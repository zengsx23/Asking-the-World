"""Paired simulation measurements; relative effects are not outcome truth labels."""
import numpy as np


INTERPRETATION = (
    'Control and edited worlds use the same fitted initial state and observation interval; '
    'the control is not required to be stationary. Differences are edited minus control. '
    'A relative shift does not prove absolute motion, reversal, goal achievement or causality. '
    'Center motion does not establish collisions, fluid destinations or deformation. '
    'Missing measurements are unknown, not zero.')


def compare_rotation(control, edited, usable=True):
    keys = ('direction', 'signed_angle_rad', 'absolute_angle_rad', 'interval_seconds',
            'coverage_fraction', 'direction_seconds', 'sign_convention', 'dominant_state',
            'moving_direction', 'rotation_occurred', 'angle_by_direction_rad')
    result = dict(control={k: control[k] for k in keys if k in control},
                  edited={k: edited[k] for k in keys if k in edited}, status='unavailable')
    if (not control.get('interval_seconds') or control.get('interval_seconds') != edited.get('interval_seconds')
            or not control.get('sign_convention') or control.get('sign_convention') != edited.get('sign_convention')):
        return result | dict(reason='Rotation intervals or coordinate conventions are missing or different')
    angles = [control.get('signed_angle_rad'), edited.get('signed_angle_rad')]
    if any(x is None or not np.isfinite(x) for x in angles):
        return result | dict(reason='Finite signed angles unavailable')
    if (not usable or any(x.get('coverage_fraction', 0.) < 1. - 1e-6
            or x.get('direction_seconds', {}).get('uncertain', 0.) > 0. for x in (control, edited))):
        return result | dict(status='unreliable', reason='Unstable rollout, incomplete contact coverage or ambiguous contact rotation')
    delta = float(angles[1] - angles[0])
    return result | dict(status='measured', edited_minus_control_angle_rad=delta,
        interpretation='Signed cumulative angle difference under the stated convention, not a claim of rotation reversal.')


def probe_comparison(probe):
    controls = {v['baseline_index']: v for v in probe.get('variants', []) if v['kind'] == 'control'}
    objects = {}
    for v in probe.get('variants', []):
        if v['kind'] != 'intervention' or v['baseline_index'] not in controls:
            continue
        base = controls[v['baseline_index']]
        usable = bool(v['stable'] and base['stable'] and v['baseline_index'] in probe['baseline']['accepted_indices'])
        for name, target in v['targets'].items():
            reference = base['targets'].get(name, {})
            pair = dict(baseline_index=v['baseline_index'], relative_mass_delta=v['relative_delta'],
                        usable=usable, required_member_fraction_above=v.get('required_member_fraction_above'))
            if 'member_mass_factor' in v:
                pair.update(member_mass_factor=v['member_mass_factor'], member_fraction=v['member_fraction'],
                            admissibility=v['admissibility'])
            left, right = reference.get('translation') or {}, target.get('translation') or {}
            a, b = left.get('net_displacement_xy_px'), right.get('net_displacement_xy_px')
            if (a is not None and b is not None and left.get('interval_seconds')
                    and left['interval_seconds'] == right.get('interval_seconds')
                    and np.shape(a) == np.shape(b) == (2,) and np.isfinite([a, b]).all()):
                pair['translation'] = dict(status='measured' if usable else 'unreliable',
                    interval_seconds=left['interval_seconds'], units='pixels',
                    coordinate_system='Image x right, y down', control=left, edited=right,
                    edited_minus_control_displacement=(np.asarray(b)-a).tolist())
            if target.get('rotation') is not None or reference.get('rotation') is not None:
                pair['rotation'] = compare_rotation(reference.get('rotation') or {}, target.get('rotation') or {}, usable)
            objects.setdefault(name, []).append(pair)
    return dict(schema='physmind.intervention_comparison', source_kind='paired_simulation',
                interpretation=INTERPRETATION, objects=objects)


def compare_rollouts(control, edited, *, fps, control_id, edited_id, start_seconds=0., image_height=None):
    """Align actual frames and independently fitted components, never empty-as-zero."""
    before = control.get('trajectories', control.get('predicted_trajectories', {}))
    after = edited.get('trajectories', edited.get('predicted_trajectories', {}))
    objects = {}
    usable = all(r.get('simulation_stability', {}).get('usable_for_downstream_reasoning', True)
                 for r in (control, edited))
    fields = [('center_world', 'meters', 1.), ('plane_xy', 'meters', 1.),
              ('center_xy_px', 'pixels', 1.),
              ('xy_normalized', 'pixels' if image_height else 'image_height', image_height or 1.),
              ('center_xy_normalized', 'pixels' if image_height else 'image_height', image_height or 1.)]
    for name in sorted(set(before) | set(after)):
        if name in edited.get('remove_objects', []):
            objects[name] = dict(status='removed', reason='Removed objects have no edited motion measurement')
            continue
        left, right = before.get(name, []), after.get(name, [])
        groups = sorted({r.get('fitted_with', '') for r in left + right})
        variants = []
        for group in groups:
            a = {r['frame_index']: r for r in left if r.get('fitted_with', '') == group and r['frame_index']/fps >= start_seconds}
            b = {r['frame_index']: r for r in right if r.get('fitted_with', '') == group and r['frame_index']/fps >= start_seconds}
            common = sorted(a.keys() & b.keys())
            row = dict(fitted_with=group or None, common_frame_count=len(common), status='unavailable')
            if len(common) < 2:
                variants.append(row | dict(reason='Fewer than two common simulated frames'))
                continue
            row.update(interval_seconds=[common[0]/fps, common[-1]/fps],
                       control_frame_count=len(a), edited_frame_count=len(b),
                       coverage_complete=len(common) == len(a) == len(b))
            for field, units, scale in fields:
                if not all(field in a[f] and field in b[f] for f in common):
                    continue
                ac, bc = np.asarray([a[f][field] for f in common])*scale, np.asarray([b[f][field] for f in common])*scale
                if ac.shape != bc.shape or not np.isfinite([ac, bc]).all():
                    continue
                da, db = ac[-1]-ac[0], bc[-1]-bc[0]
                duration = (common[-1]-common[0])/fps
                row.update(status='measured' if usable else 'unreliable', position_field=field, units=units,
                    coordinate_system=('Image x right, y down' if field in {'center_xy_px', 'xy_normalized', 'center_xy_normalized'}
                                       else edited.get('coordinate_system', 'Retained SWR plane coordinates')),
                    control=dict(start_position=ac[0].tolist(), end_position=ac[-1].tolist(),
                                 net_displacement=da.tolist(), mean_velocity=(da/duration).tolist()),
                    edited=dict(start_position=bc[0].tolist(), end_position=bc[-1].tolist(),
                                net_displacement=db.tolist(), mean_velocity=(db/duration).tolist()),
                    edited_minus_control=dict(net_displacement=(db-da).tolist(),
                        final_position=(bc[-1]-ac[-1]).tolist(), mean_velocity=((db-da)/duration).tolist()))
                break
            if all('yaw_delta_rad' in a[f] and 'yaw_delta_rad' in b[f] for f in common):
                ac = a[common[-1]]['yaw_delta_rad']-a[common[0]]['yaw_delta_rad']
                bc = b[common[-1]]['yaw_delta_rad']-b[common[0]]['yaw_delta_rad']
                if np.isfinite([ac, bc]).all():
                    row['yaw'] = dict(control_angle_rad=ac, edited_angle_rad=bc,
                        edited_minus_control_angle_rad=bc-ac, convention='Retained SWR plane yaw; not image clockwise/counterclockwise')
            for field in ('active_particles', 'bounds_xyxy_px'):
                if field in a[common[-1]] and field in b[common[-1]]:
                    ac, bc = np.asarray(a[common[-1]][field]), np.asarray(b[common[-1]][field])
                    if ac.shape == bc.shape and np.isfinite([ac, bc]).all():
                        row[field] = dict(control=ac.tolist(), edited=bc.tolist(), edited_minus_control=(bc-ac).tolist(),
                                          measurement='Final common frame; particle count is not liquid volume')
            variants.append(row)
        objects[name] = dict(variants=variants) if variants else dict(status='unavailable', reason='No trajectory samples')
    return dict(schema='physmind.intervention_comparison', source_kind='paired_simulation',
        control_rollout_id=control_id, edited_rollout_id=edited_id, start_seconds=start_seconds,
        interpretation=INTERPRETATION,
        scope='Common-frame centers and mean velocities; plane yaw and fluid bounds/counts when available. '
              'No deformation or event comparison is inferred from centers. Cloth fitted_with variants remain independent.',
        objects=objects)
