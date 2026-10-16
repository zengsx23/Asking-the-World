"""Conditional body-mass interventions, separate from factual SWR fitting.

The rest baseline is an explicit hypothesis, not recovered benchmark metadata.
Finite sensitivity scans establish agreement only on the supplied hypotheses.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from atw.polyworld.rope_controls import mass_controls, replay_engine, require_independent_control
from atw.polyworld.rope_rotation import infer_pulley_rotation, summarize_center_translation


def target_control(scenes, object_id):
    matches = [(s, c) for s in scenes for c in mass_controls(s) if object_id in c['members']]
    if len(matches) != 1:
        raise ValueError('Need exactly one modeled load mass containing the requested body')
    scene, control = matches[0]
    require_independent_control(scenes, scene.rope_id)
    return scene, control


def baseline_parameters(scene, fit, mode, member_masses=None):
    if mode not in {'fitted_pre_event', 'zero_initial_velocity', 'rest', 'rest_calibrated'}:
        raise ValueError('baseline must be fitted_pre_event, zero_initial_velocity, rest or rest_calibrated')
    theta = np.asarray(fit['routing']['best_parameters'], np.float32).copy()
    controls = mass_controls(scene)
    if member_masses is not None:
        required = {name for c in controls if c['parameter_index'] is not None for name in c['members']}
        allowed = {name for c in controls for name in c['members']}
        if (not required.issubset(member_masses) or not set(member_masses).issubset(allowed)
                or any(not c['members'] for c in controls)
                or any(isinstance(v, bool) or not isinstance(v, (float, int))
                       or not np.isfinite(v) or v <= 0 for v in member_masses.values())):
            raise ValueError('member_masses must supply positive masses for every modeled dynamic member; fixed-load overrides are optional, unknown IDs are not allowed')
        if mode == 'rest_calibrated':
            raise ValueError('Explicit member masses must not be changed by baseline calibration; use rest')
    for c in controls:
        i = c['parameter_index']
        if i is None:
            continue
        mass = sum(member_masses[name] for name in c['members']) if member_masses is not None else np.exp(theta[i])
        theta[i:i+2] = np.log(mass)
    if mode != 'fitted_pre_event':
        theta[3] = 0.
    return theta


def intervention_batch(baselines, control, relative_deltas=None, *, member_fraction=None, member_factors=None):
    """Add delta*m_group to ONE member; other member masses are not scaled."""
    if member_fraction is not None and (isinstance(member_fraction, bool)
            or not isinstance(member_fraction, (int, float)) or not np.isfinite(member_fraction)
            or not 0 < member_fraction <= 1):
        raise ValueError('member_fraction must be a positive finite fraction <= 1')
    fraction = 1. if len(control['members']) == 1 else member_fraction
    factors = None
    if member_factors is not None:
        if relative_deltas is not None:
            raise ValueError('Supply either relative_deltas or member_factors, not both')
        if len(control['members']) == 1 and member_fraction not in (None, 1.):
            raise ValueError('An independent body has member_fraction 1')
        factors = np.asarray(member_factors, dtype=float)
        if (factors.ndim != 1 or not 1 <= len(factors) <= 32 or not np.isfinite(factors).all()
                or np.any(factors < .001) or np.any(factors > 1000) or np.any(factors == 1)
                or len(np.unique(factors)) != len(factors)):
            raise ValueError('Use 1..32 distinct member factors in [0.001, 1000], excluding 1')
        if fraction is None:
            raise ValueError('Assembly member factors require an explicit member_fraction hypothesis')
        if len(control['members']) > 1 and fraction >= 1:
            raise ValueError('Assembly member_fraction must leave positive mass for other members')
        deltas = (factors - 1) * fraction
    else:
        deltas = np.asarray(relative_deltas, dtype=float)
        if (deltas.ndim != 1 or not 1 <= len(deltas) <= 32 or not np.isfinite(deltas).all()
                or np.any(deltas == 0) or len(np.unique(deltas)) != len(deltas)
                or np.any(deltas <= -1) or np.any(deltas > 10)):
            raise ValueError('Use 1..32 distinct nonzero relative deltas in (-1, 10]')
    samples, variants = [], []
    i = control['parameter_index']
    for baseline_index, theta in enumerate(baselines):
        samples.append(theta.copy())
        variants.append(dict(baseline_index=baseline_index, kind='control', relative_delta=0., admissibility='known'))
        for factor_index, delta in enumerate(deltas):
            if fraction is not None and fraction + delta <= 0:
                raise ValueError('Decrement would make the target member mass nonpositive')
            sample = theta.copy()
            mass = float(control['relative_mass'] if i is None else np.exp(theta[i]))
            if i is not None:
                sample[i+1] = np.log(mass * (1 + delta))
            samples.append(sample)
            variants.append(dict(baseline_index=baseline_index, kind='intervention', relative_delta=float(delta),
                added_mass=float(delta * mass), before_group_mass=mass, after_group_mass=float(mass * (1 + delta)),
                admissibility='conditional_member_mass' if fraction is None and delta < 0 else 'known',
                required_member_fraction_above=float(-delta) if fraction is None and delta < 0 else None))
            if factors is not None:
                variants[-1].update(member_mass_factor=float(factors[factor_index]),
                    member_fraction=float(fraction), unchanged_other_members_mass=float(mass * (1-fraction)),
                    admissibility='conditional_member_fraction' if len(control['members']) > 1 else 'known')
    return np.asarray(samples, np.float32), variants


def direction_consensus(values, usable):
    if not values or len(values) != len(usable):
        raise ValueError('Consensus needs matching nonempty readouts and validity flags')
    if not all(usable):
        return dict(status='unreliable', direction=None, sampled_directions=values)
    if any(v in {None, 'uncertain'} for v in values):
        return dict(status='insufficient_readout', direction=None, sampled_directions=values)
    choices = set(values)
    return dict(status='sample_consistent' if len(choices) == 1 else 'parameter_sensitive',
        direction=values[0] if len(choices) == 1 else None, sampled_directions=values)


class RopeMassProbe:
    def __init__(self, scene, fit, *, device='cuda:0', capacity=96):
        self.scene, self.fit = scene, fit
        self.engine = replay_engine(scene, fit, device, capacity=capacity)
        self.capacity = capacity
        self._calibration_cache = {}

    def _positions(self, rope, wheels):
        return {k:v.detach().cpu().numpy().copy()
                for k,v in self.engine.predicted_body_centers(rope, wheels).items()}

    def calibrate_rest(self, initial, *, population=64, generations=12, seed=7, keep=4):
        """Fit masses to an explicit static-control hypothesis, not QA labels."""
        if not 4 <= population <= self.capacity or not 1 <= generations <= 64 or not 1 <= keep <= population:
            raise ValueError('Invalid bounded baseline calibration budget')
        indices = [c['parameter_index'] for c in mass_controls(self.scene) if c['parameter_index'] is not None]
        if not indices:
            return initial[None].copy(), dict(status='no_dynamic_masses')
        cache_key = (initial.tobytes(), population, generations, seed, keep)
        if cache_key in self._calibration_cache:
            candidates, metadata = self._calibration_cache[cache_key]
            return candidates.copy(), dict(metadata, cache_hit=True)
        import torch
        started = perf_counter()
        rng = np.random.default_rng(seed)
        mean, std = initial[indices].astype(float), np.ones(len(indices))
        lower, upper = mean - np.log(16), mean + np.log(16)
        best, best_loss = initial.copy(), float('inf')
        elite_count = max(4, population // 8)
        for _ in range(generations):
            logs = np.clip(mean + std * rng.standard_normal((population, len(indices))), lower, upper)
            samples = np.repeat(initial[None], population, axis=0)
            samples[:, indices] = samples[:, np.asarray(indices)+1] = logs
            samples[0] = best
            rope, wheels = self.engine.rollout(samples)
            losses = torch.zeros(population, device=rope.device)
            bodies = self.engine.predicted_body_centers(rope, wheels)
            dynamic = [b for b in self.scene.bodies if b.dynamic and b.role in {'endpoint', 'pulley'}]
            for body in dynamic:
                reference = torch.as_tensor(body.initial_center, device=rope.device, dtype=rope.dtype)
                losses += (bodies[body.object_id] - reference).square().sum(-1).mean(-1)
            loss = losses.detach().cpu().numpy() / max(1, len(dynamic))
            loss = np.where(np.isfinite(loss), loss, np.inf)
            order = np.argsort(loss)
            if not np.isfinite(loss[order[0]]):
                raise ValueError('All baseline calibration candidates are nonfinite')
            if loss[order[0]] < best_loss:
                best, best_loss = samples[order[0]].copy(), float(loss[order[0]])
            elite = samples[order[:elite_count]][:, indices]
            mean = .2 * mean + .8 * elite.mean(0)
            std = np.maximum(.05, .2 * std + .8 * elite.std(0))
        candidates = samples[order[:keep]].copy()
        candidates[0] = best
        metadata = dict(status='conditional_rest_fit', seconds=perf_counter()-started,
            population=population, generations=generations, seed=seed,
            best_body_rmse_px=float(np.sqrt(best_loss) * self.scene.image_height),
            prior_mass_factor_bounds=[1/16, 16], retained_candidates=len(candidates),
            objective='Mean squared dynamic-body displacement from initial centers; no question answers or GT masses',
            limitation='Static balance constrains effective masses, not a unique member split or benchmark reset state')
        self._calibration_cache[cache_key] = candidates.copy(), metadata
        return candidates, dict(metadata, cache_hit=False)

    def run(self, object_id, target_ids, relative_deltas=None, *, baseline='zero_initial_velocity',
            member_masses=None, member_fraction=None, member_factors=None, intervention_seconds=None,
            tolerance_px=3., calibration_population=64, calibration_generations=12, baseline_count=4):
        started = perf_counter()
        scene, control = target_control([self.scene], object_id)
        if not target_ids or len(set(target_ids)) != len(target_ids) or set(target_ids) - {b.object_id for b in scene.bodies}:
            raise ValueError('Targets must be distinct body IDs in the same modeled rope system')
        if not np.isfinite(tolerance_px) or tolerance_px <= 0:
            raise ValueError('tolerance_px must be positive')
        original = np.asarray(self.fit['routing']['best_parameters']).copy()
        theta = baseline_parameters(scene, self.fit, baseline, member_masses)
        calibration = None
        if member_masses is not None:
            if member_fraction is not None:
                raise ValueError('Supply either explicit member masses or a member fraction, not both')
            if control['parameter_index'] is None:
                member_fraction = 1.
                if object_id in member_masses:
                    control = dict(control, relative_mass=member_masses[object_id], mass_source='caller_supplied')
            else:
                member_fraction = member_masses[object_id] / sum(member_masses[n] for n in control['members'])
        # Reject malformed interventions before spending time calibrating a world.
        checked, _ = intervention_batch(theta[None], control, relative_deltas,
            member_fraction=member_fraction, member_factors=member_factors)
        count = baseline_count if baseline == 'rest_calibrated' else 1
        if count * len(checked) > self.capacity:
            raise ValueError('Too many hypotheses for the configured batch capacity')
        time = ((scene.frames[0]+scene.event_frame_offset)/scene.fps
                if intervention_seconds is None else intervention_seconds)
        if isinstance(time, bool) or not isinstance(time, (float, int)) or not np.isfinite(time):
            raise ValueError('intervention_seconds must be finite video time')
        offset = round(time * scene.fps) - scene.frames[0]
        if not 0 <= offset < scene.frame_count - 1:
            raise ValueError('Intervention must leave a measurable subsequent interval')
        bases = theta[None]
        if baseline == 'rest_calibrated':
            bases, calibration = self.calibrate_rest(theta, population=calibration_population,
                generations=calibration_generations, keep=baseline_count)
        samples, variants = intervention_batch(bases, control, relative_deltas,
            member_fraction=member_fraction, member_factors=member_factors)
        effective_deltas = [r['relative_delta'] for r in variants
                            if r['kind'] == 'intervention' and r['baseline_index'] == 0]
        # Epoch timing is a launch input, not a change to the retained scene/fit.
        previous_scene = self.engine.scene
        self.engine.scene = replace(scene, event_frame_offset=offset)
        try:
            rope, wheels = self.engine.rollout(samples)
            centers = self._positions(rope, wheels)
            points = rope.detach().cpu().numpy().copy()
        finally:
            self.engine.scene = previous_scene
        times = (scene.frames[0] + np.arange(scene.frame_count)) / scene.fps
        rest_lengths = np.maximum(np.linalg.norm(np.diff(scene.initial_positions, axis=0), axis=-1), 1e-8)
        body_map = {b.object_id:b for b in scene.bodies}
        control_drift = {}
        for index, row in enumerate(variants):
            strain = np.abs(np.linalg.norm(np.diff(points[index], axis=1), axis=-1)/rest_lengths-1)
            finite = bool(np.isfinite(points[index]).all() and all(np.isfinite(v[index]).all() for v in centers.values()))
            max_strain = float(strain.max()) if finite else None
            displacement = float(np.max(np.linalg.norm(points[index]-points[index, :1], axis=-1))) if finite else None
            row['stable'] = bool(finite and max_strain <= .5 and displacement <= 2*np.hypot(scene.image_width,scene.image_height)/scene.image_height)
            row['max_segment_strain'] = max_strain
            if row['kind'] == 'control':
                control_drift[row['baseline_index']] = max((float(np.linalg.norm(
                    xy[index]-np.asarray(body_map[name].initial_center), axis=-1).max()*scene.image_height)
                    for name,xy in centers.items() if body_map[name].dynamic), default=0.) if finite else None
            row['targets'] = {}
            for name in target_ids:
                body = body_map[name]
                xy = centers.get(name, np.broadcast_to(body.initial_center, (len(samples), scene.frame_count, 2)))
                translation = summarize_center_translation(xy[index, offset:]*scene.image_height,
                    times[offset:], tolerance_px=tolerance_px)
                rotation = None
                if body.role == 'pulley' and finite:
                    slack = getattr(self.engine, 'last_diagnostics', {}).get('slack_length')
                    rotation = infer_pulley_rotation(points[index, offset:], xy[index, offset:], body.radius,
                        times[offset:], contact_tolerance=max(2/scene.image_height, body.radius*.15),
                        taut_frames=None if slack is None else slack[index, offset:] <= 1e-6)
                    rotation.pop('samples', None)
                row['targets'][name] = dict(translation=translation, rotation=rotation)
        require_rest = baseline in {'rest', 'rest_calibrated'}
        controls = {r['baseline_index']:r for r in variants if r['kind'] == 'control'}
        accepted = [i for i,r in controls.items() if r['stable'] and (not require_rest
            or control_drift[i] is not None and control_drift[i] <= tolerance_px)]
        interventions = [r for r in variants if r['kind'] == 'intervention' and r['baseline_index'] in accepted]
        consensus = {}
        for label, sign in [('increase', 1), ('decrease', -1)]:
            if not any(sign * d > 0 for d in effective_deltas):
                continue
            group = [r for r in interventions if sign * r['relative_delta'] > 0]
            consensus[label] = {}
            for name in target_ids:
                values = [r['targets'][name] for r in group]
                usable = [r['stable'] for r in group]
                if not group:
                    consensus[label][name] = dict(status='invalid_baseline', reason='No control satisfies the declared baseline and stability checks')
                    continue
                outcome = dict(
                    vertical=direction_consensus([v['translation'].get('net_direction_xy', [None,None])[1] for v in values], usable),
                    horizontal=direction_consensus([v['translation'].get('net_direction_xy', [None,None])[0] for v in values], usable))
                if body_map[name].role == 'pulley':
                    outcome['rotation'] = direction_consensus([(v['rotation'] or {}).get('direction') for v in values], usable)
                outcome['required_member_fraction_above'] = max((r.get('required_member_fraction_above') or 0 for r in group), default=0.) or None
                consensus[label][name] = outcome
        assert np.array_equal(original, self.fit['routing']['best_parameters'])
        self.last_geometry = dict(rope=points, parameters=samples, times=times,
                                 **{f'body_{name}':xy for name,xy in centers.items()})
        result = dict(schema='physmind.rope_mass_probe', object_id=object_id,
            effective_group_id=control['object_id'], group_members=control['members'], rope_id=scene.rope_id,
            baseline=dict(mode=baseline, initial_rope_speed=float(theta[3]), member_masses=member_masses,
                rest_required=require_rest,
                parameters=[b.tolist() for b in bases], calibration=calibration,
                control_max_body_drift_px=control_drift, rest_tolerance_px=tolerance_px if require_rest else None,
                accepted_indices=accepted, rejected_indices=[i for i in controls if i not in accepted]),
            intervention=dict(operation='scale_requested_member_mass' if member_factors is not None else 'add_mass_to_requested_member',
                relative_deltas=effective_deltas,
                units='fraction of this baseline effective group mass; not SI kilograms',
                time_seconds=float(times[offset]), member_fraction=member_fraction,
                time_source=('explicit' if intervention_seconds is not None else
                             'initial_frame' if self.fit.get('mass_schedule') == 'constant' else 'legacy_saved_event')),
            variants=variants, consensus=consensus, seconds=perf_counter()-started,
            limitations=['Agreement covers only sampled hypotheses, not all possible masses or unseen topologies.',
                'Negative deltas on an unknown member require the reported positive-member-mass condition.',
                ('Initial velocity is zero; the control may move under gravity, with no equilibrium requirement.'
                 if baseline == 'zero_initial_velocity' else
                 'Rest is an explicit initial-state hypothesis, not benchmark reset metadata.' if require_rest else
                 'Fitted initial motion and pre-event masses are retained, not independently identified benchmark reset metadata.'),
                'Consensus describes absolute edited motion; intervention_comparison separately reports paired relative effects.',
                'Independent rope fits and no-slip spin approximation are retained.',
                'A member mass increment changes group translational mass; independent assembly COM/inertia changes are not modeled.'])
        if member_factors is not None:
            result['intervention'].update(member_factors=list(map(float, member_factors)),
                member_fraction=1. if len(control['members']) == 1 else member_fraction,
                member_fraction_source='explicit_member_masses' if member_masses is not None else
                    'single_member' if len(control['members']) == 1 else 'hypothesis')
            result['limitations'].append('Finite extreme factors are sensitivity tests, not mathematical limits; stability must be checked for every sample.')
        result['baseline']['mass_schedule'] = self.fit.get('mass_schedule', 'legacy_piecewise')
        if control['parameter_index'] is None:
            result['mass_binding'] = {k:v for k,v in control.items() if k != 'parameter_index'}
            result['limitations'].append('The load has a fixed support; changing its mass does not release that constraint. Its default mass is not identified by stationary observations.')
        from atw.polyworld.intervention_comparison import probe_comparison
        result['intervention_comparison'] = probe_comparison(result)
        result['seconds'] = perf_counter()-started
        return result
