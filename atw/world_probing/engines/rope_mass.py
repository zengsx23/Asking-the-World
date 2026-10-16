"""Select a finite qualitative mass edit using actual retained-world dynamics.

Search objectives are internal intervention conventions, never answer evidence.
Only the target mass changes; geometry, gravity, drag and supports remain fixed.
"""
from dataclasses import replace

import numpy as np

from atw.polyworld.rope_controls import mass_controls, mass_intervention_parameters, replay_engine, resolve_mass, stability


def select_mass_factor(scenes, fits, object_id, direction, device, diagnostics):
    """Return an admissible factor, preserving increase/decrease and supports.

    A 0.05 image-height/s mean vertical speed is the internal reference, shared
    with the endpoint actuator's nominal speed. It is not a fitted measurement.
    Sample in log mass, then refine locally without assuming global monotonicity:
    contact changes and slack can make the response discontinuous.
    """
    if direction not in {'increase', 'decrease'}:
        raise ValueError('Mass direction must be increase or decrease')
    binding = resolve_mass(scenes, object_id)
    scene = next(s for s in scenes if s.rope_id == binding['rope_id'])
    control = next(c for c in mass_controls(scene) if c['object_id'] == binding['object_id'])
    diagnostics.update(object_id=object_id, direction=direction, binding=binding, candidates=[])
    if control['parameter_index'] is None:
        factor = 10. if direction == 'increase' else .1
        diagnostics.update(status='fixed_support', selected_factor=factor)
        return factor, binding

    fit = next(f for f in fits if f['rope_id'] == scene.rope_id)
    duration = (scene.frame_count - 1) / scene.fps
    if duration <= 0:
        raise ValueError('A mass experiment needs a nonempty time interval')
    sign = 1 if direction == 'increase' else -1
    target_speed = .05
    diagnostics.update(target_mean_speed_image_height_per_s=target_speed,
                       duration_seconds=duration, status='searching')
    # The same bounded range is accepted by NativeExperiments for final replay.
    magnitudes = np.linspace(np.log(1.01), np.log(1000.), 17)
    engine = None

    def evaluate(log_factors):
        nonlocal engine
        parameters = []
        for log_factor in log_factors:
            theta, event, _ = mass_intervention_parameters(scene, fit,
                {binding['object_id']:float(np.exp(sign * log_factor))},
                'factual', None, initial_state='rest')
            parameters.append(theta)
        if engine is None:
            engine = replay_engine(replace(scene, event_frame_offset=event), fit, device, capacity=17)
        from atw.polyworld.rope_ideal import IdealRopeEngine
        ideal = isinstance(engine, IdealRopeEngine)
        rope, wheels = engine.rollout(np.asarray(parameters), **({'diagnostic':True} if ideal else {}))
        points = rope.detach().cpu().numpy()
        centers = engine.predicted_body_centers(rope, wheels)
        if object_id not in centers:
            raise ValueError('The requested mass has no simulated body trajectory')
        xy = centers[object_id].detach().cpu().numpy()
        for i, log_factor in enumerate(log_factors):
            valid = bool(np.isfinite(points[i]).all() and np.isfinite(xy[i]).all())
            if ideal:
                valid = valid and bool(engine.last_diagnostics['valid'][i])
            if valid:
                result = dict(geometry={scene.rope_id:points[i]})
                if ideal and engine.unilateral_rope:
                    result['rope_constraint_diagnostics'] = {scene.rope_id:engine.constraint_report(i)}
                valid = stability([scene], result)['usable_for_downstream_reasoning']
            speed = float(sign * (xy[i,-1,1]-xy[i,0,1]) / duration) if valid else None
            diagnostics['candidates'].append(dict(factor=float(np.exp(sign * log_factor)),
                log_magnitude=float(log_factor), usable=valid, signed_mean_speed=speed))

    evaluate(magnitudes)
    valid = [c for c in diagnostics['candidates'] if c['usable']]
    if valid:
        best = min(valid, key=lambda c:abs(c['signed_mean_speed']-target_speed))
        index = int(np.argmin(abs(magnitudes-best['log_magnitude'])))
        low, high = magnitudes[max(0,index-1)], magnitudes[min(len(magnitudes)-1,index+1)]
        previous = {c['log_magnitude'] for c in diagnostics['candidates']}
        evaluate([v for v in np.linspace(low,high,9) if v not in previous])
        valid = [c for c in diagnostics['candidates'] if c['usable']]
        # Use the existing 3-pixel motion tolerance, not an artificial speed band.
        # A faster/slower valid response is still evidence; reference speed only
        # ranks candidates. Never reverse the mass edit to slow an existing fall.
        moving = [c for c in valid if c['signed_mean_speed'] * duration * scene.image_height > 3.]
        if moving:
            best = min(moving, key=lambda c:abs(c['signed_mean_speed']-target_speed))
            diagnostics.update(status='selected', selected_factor=best['factor'],
                               selected_mean_speed=best['signed_mean_speed'])
            return best['factor'], binding
    diagnostics['status'] = 'unavailable'
    raise ValueError('No usable finite mass intervention was found within the supported range; '
                     'the current world cannot supply this intervention evidence')
