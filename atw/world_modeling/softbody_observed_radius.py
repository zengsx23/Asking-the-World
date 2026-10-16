"""Fixed observed radius, deformable contact and observation-restarted fitting."""
from dataclasses import replace
from types import SimpleNamespace
import time
import numpy as np
from atw.polyworld import soft_body_swr as swr
from atw.world_modeling import softbody_observations as observation
from atw.world_modeling.softbody_ballistic_utils import stage_two_scene, projected, parameter_payload, projection_tolerance, surface_residual
from atw.world_modeling.scene_assets import write

MODEL = 'agentic-softbody-observed-radius-v1'
RECIPE = 'observed_radius_local_contact'


class Engine(swr.BatchWarpSoftBodyEngine):
    """No observed shape projection in dynamics; geometry correction is output-only."""
    def __init__(self, scene, *, deformable_contacts=True, **kwargs):
        super().__init__(scene, deformable_contacts=deformable_contacts, **kwargs)
        self.enable_shape_projection = False

    def rollout_numpy(self, values):
        raw, contacts = super().rollout_numpy(values)
        self.projection_diagnostics = []
        return swr._project_output_surface_positions(self.scene, raw, tolerance=self.projection_tolerance, strict=False, diagnostics=self.projection_diagnostics), contacts


def make_engine(scene, *, capacity=1, device='cuda:0', substeps=32, **kwargs):
    return Engine(stage_two_scene(scene), capacity=capacity, device=device, substeps=substeps,
                  projection_tolerance=projection_tolerance(scene), strict_projection=False, **kwargs)


def loss(raw, scene, obs, target, valid):
    center = ((observation.pixels(scene, raw, obs)-target)**2).sum(-1)[:, valid].mean(1)/1e6
    shape = (scene.radii[None, None, :, None, None]*(swr._numpy_deformation(raw)-scene.target_deformations))**2
    mask = scene.shape_observed_mask > 0
    return center + (.01*shape.mean((-1, -2))[:, mask].mean(1) if mask.any() else 0)


def search(scene, obs, base, starts, device, *, family='deformable_contact', rawbase=None,
           fixed_radius=None, generations=180, population=128):
    """Fit material parameters using observations only, with fixed initial kinematics."""
    layout = swr.make_parameter_layout(scene.object_count)
    low, high = swr.parameter_bounds(layout)
    high[layout.log_frequency_slice] = np.log(120)
    _, std = swr.initial_distribution(layout)
    if fixed_radius is not None:
        low[layout.log_radius_scale_slice] = np.log(fixed_radius)
        high[layout.log_radius_scale_slice] = np.log(fixed_radius)
    options = dict(deformable_contacts=family == 'deformable_contact', separating_normal_damping=family == 'normal_damping')
    engines = [make_engine(scene, capacity=population, device=device, substeps=k, **options) for k in (32, 64)]
    windows = []
    begins = np.array([], dtype=int)
    if rawbase is not None:
        if scene.frame_count < 31:
            raise ValueError('Local contact fitting requires at least 31 tracked frames')
        positions, velocity, vm = observation.observed_states(scene, obs)
        begins = np.unique(np.linspace(6, scene.frame_count-25, 5).astype(int))
        fields = ('target_centers','target_subset_centers','target_covariances','target_deformations','reference_subset_offsets','reference_subset_covariances','shape_observed_mask','shape_contact_recovered_mask','observed_mask')
        for begin in begins:
            stop = begin+25
            changes = {k: getattr(scene, k)[begin:stop].copy() for k in fields}
            initial = rawbase[begin, :, :2].copy()
            v = np.gradient(rawbase[:, :, :2], 1/scene.fps, axis=0)[begin].copy()
            mask = obs.valid[begin:stop].copy()
            for i in range(scene.object_count):
                if vm[begin, i]:
                    initial[i], v[i] = positions[begin, i], velocity[begin, i]
                else:
                    mask[:, i] = False
                    changes['observed_mask'][:, i] = 0
                    changes['shape_observed_mask'][:, i] = 0
            if not mask.any():
                continue
            target = replace(scene, **changes, frames=tuple(range(25)), first_offsets=tuple(0 for _ in scene.object_ids), initial_centers=initial)
            engine = make_engine(target, capacity=population, device=device, substeps=32,
                                 initial_strains=rawbase[begin, :, 2:], initial_strain_velocities=np.gradient(rawbase[:, :, 2:], 1/scene.fps, axis=0)[begin], **options)
            windows.append((begin, target, engine, v, mask))
        if not windows:
            raise ValueError('No observed local contact windows')
    fields = ['log_radius_scale_slice','log_frequency_slice','damping_ratio_slice','friction_slice']
    if family == 'deformable_contact':
        fields += ['deformation_gain_slice','log_shape_frequency_slice','shape_damping_ratio_slice']
    groups = [np.array([getattr(layout, k).start+i for k in fields]) for i in range(scene.object_count)]
    indices = np.concatenate(groups)

    def constrain(values):
        values = np.clip(values, low, high).astype('float32')
        values[:, :2*scene.object_count] = base[:2*scene.object_count]
        values[:, layout.log_gravity_scale_slice] = base[layout.log_gravity_scale_slice]
        values[:, layout.drag_slice] = 0
        return values

    def score(values):
        full = sum(loss(e.rollout(values).detach().cpu().numpy(), scene, obs, obs.centers, obs.valid) for e in engines)/2
        if not windows:
            return full
        local = np.zeros(len(values))
        for begin, target, engine, velocity, mask in windows:
            candidate = values.copy()
            candidate[:, :2*scene.object_count] = velocity.reshape(-1)
            local += loss(engine.rollout(candidate).detach().cpu().numpy(), target, obs, obs.centers[begin:begin+25], mask)
        return .5*full + .5*local/len(windows)

    starts = constrain(np.stack(starts))
    scores = score(starts)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite observation warm-start loss')
    best = starts[np.argmin(scores)].copy()
    best_score = float(scores.min())
    trace = [best_score]
    rng = np.random.default_rng(20261008)
    scales = (.3, .1, .03) if windows else (.5, .2, .06)
    for gen in range(generations):
        amplitude = scales[min(2, gen//(generations//3))]
        samples = np.tile(best, (population, 1))
        group = groups[gen % scene.object_count]
        samples[:, group] += rng.normal(size=(population, len(group)))*std[group]*amplitude
        if gen % 4 == 0:
            samples[population//2:, indices] = best[indices] + rng.normal(size=(population-population//2, len(indices)))*std[indices]*amplitude*.5
        samples = constrain(samples)
        samples[0] = best
        scores = score(samples)
        scores[~np.isfinite(scores)] = np.inf
        k = np.argmin(scores)
        if scores[k] < best_score:
            best, best_score = samples[k].copy(), float(scores[k])
        if (gen+1) % 30 == 0:
            trace.append(best_score)
            print('softbody observed contact', family, gen+1, best_score, flush=True)
    return best, dict(seed=20261008, population=population, generations=generations, trace=trace,
                      window_starts=begins.tolist(), full_weight=.5 if windows else 1., local_weight=.5 if windows else 0., integration_substeps=[32,64])


def replay(scene, fit_directory, *, device='cuda:0'):
    """Replay the saved baseline with its calibrated initial centers and force law."""
    with np.load(fit_directory/'softbody.npz', allow_pickle=False) as data:
        scene = replace(scene, initial_centers=data['initial_centers'].copy())
        theta = data['theta'].copy()
    engine = make_engine(scene, device=device)
    return engine.rollout(theta).detach().cpu().numpy()[0], projected(engine, theta)


def fit(inputs, context):
    started = time.perf_counter()
    out, device = context.output_dir, context.scene.device
    scene, source = swr.load_softbody_scene(inputs['tracking_directory'], reconstruction_dir=inputs['reconstruction_directory'])
    expected = {o['object_id'] for o in inputs['objects'] if o['physical_type'] == 'soft_body'}
    if set(scene.object_ids) != expected:
        raise ValueError('Softbody tracks do not cover classified identities')
    obstacles = {o['object_id'] for o in inputs['objects'] if o['physical_type'] == 'rigid'}
    if {o.object_id for o in scene.obstacles} != obstacles:
        raise ValueError('Static obstacle geometry does not cover classified rigids')
    obs = observation.load(scene, inputs)
    if scene.frame_count < 31:
        raise ValueError('Local contact fitting requires at least 31 tracked frames')
    initial, _ = swr.initial_distribution(swr.make_parameter_layout(scene.object_count))
    scene, calibrated, calibration = observation.calibrate_initial_state(scene, obs, initial)
    from atw.world_modeling.softbody_ballistic import fit as initialize
    warm_dir = out/'initialization'
    warm_dir.mkdir(parents=True)
    initialize(inputs, SimpleNamespace(output_dir=warm_dir, scene=context.scene),
               gravity_calibration=(calibration['gravity_scale'], [dict(method='initial_clear_mask_centers', **calibration)]))
    with np.load(warm_dir/'softbody.npz', allow_pickle=False) as data:
        base = data['theta'].copy()
    layout = swr.make_parameter_layout(scene.object_count)
    base[layout.velocity_slice] = calibrated[layout.velocity_slice]
    base[layout.log_gravity_scale_slice] = calibrated[layout.log_gravity_scale_slice]
    base[layout.drag_slice] = 0
    write(out/'initial_state.json', dict(**calibration, initial_centers=scene.initial_centers.tolist(), units='reconstruction length units and seconds'))
    seeds = [base]
    from atw.world_modeling.softbody_ballistic_utils import encode
    import json
    for name in ('warm_original.json','warm_shared_center.json'):
        seeds.append(encode(scene, json.loads((warm_dir/name).read_text())))
    warm = []
    for family in ('native','normal_damping','deformable_contact'):
        theta, info = search(scene, obs, base, seeds, device, family=family, generations=240)
        warm.append(theta)
        np.savez_compressed(out/f'warm_{family}.npz', theta=theta)
        write(out/f'warm_{family}.json', info)
    baseline = make_engine(scene, device=device).rollout(warm[-1]).detach().cpu().numpy()[0]
    scale = observation.radius_scales(scene, obs)
    theta, optimization = search(scene, obs, warm[-1], warm, device, rawbase=baseline, fixed_radius=scale)
    raws = {}
    metrics = {}
    for steps in (32,64):
        engine = make_engine(scene, device=device, substeps=steps)
        raws[f'raw_{steps}'] = engine.rollout(theta).detach().cpu().numpy()[0]
        metrics[f'observed_{steps}'] = observation.metrics(scene, raws[f'raw_{steps}'], obs)
    engine = make_engine(scene, device=device)
    predicted = projected(engine, theta)
    metrics['integration_drift_pixels'] = float(np.sqrt(((observation.pixels(scene, raws['raw_32'], obs)-observation.pixels(scene, raws['raw_64'], obs))**2).sum(-1)[obs.valid].mean()))
    np.savez_compressed(out/'softbody.npz', theta=theta, initial_centers=scene.initial_centers, raw=raws['raw_32'], predicted=predicted, **raws)
    np.savez_compressed(out/'observations.npz', centers=obs.centers, valid=obs.valid, offsets=obs.offsets, first_areas=obs.first_areas)
    payload = dict(protocol_version=1, model=MODEL, recipe=RECIPE, source=source, object_ids=list(scene.object_ids),
                   prediction_npz='softbody.npz', observations_npz='observations.npz', image_wh=list(obs.image_wh), parameters=parameter_payload(scene, theta),
                   substeps_per_frame=32, fixed_radius_scale=scale.tolist(), contact_model='deformation_dependent_support', angular_state=False,
                   initial_centers='softbody.npz:initial_centers', initialization='same_call_observation_only', optimization=optimization, metrics=metrics,
                   projection=dict(policy='best_effort', tolerance=projection_tolerance(scene), radius_fraction=.01,
                                   nonconverged_frames=engine.projection_diagnostics, max_remaining_penetration=surface_residual(scene, predicted)),
                   elapsed_seconds=time.perf_counter()-started)
    write(out/'softbody.json', payload)
    return out/'softbody.json', metrics, [
        'Non-rotating reduced-order soft bodies; not a volumetric material model.',
        'First-mask radius assumes an adequately visible, approximately circular initial silhouette.',
        'Mask centers and reconstruction scale are observational estimates, not calibrated physical truth.',
        'Output surface correction is separate from raw dynamics; nonconverged frames are retained.',
    ]
