"""Explicit ContPhy reset/readout conventions, separate from fitted physics.

Public generator commit39e389ef2f7f69885291388c06ba30a92ba87b1c:
https://github.com/zzcnewly/ContPhy-Gen/tree/39e389ef2f7f69885291388c06ba30a92ba87b1c
PulleyGroupConfig.cs defines mass_const=.2, pulley_load_multifier=1.975,
motion_threshold=1.2, rotation_threshold_degree=35. PulleyGroupManager.cs
initializes nominal masses and AnswerMotion/AnswerRotation take first crossings.
No instance annotations, answers, or ground-truth masses are read here.
"""
import numpy as np

NAME = 'contphy-reset-v1'
FACTORS = dict(increase=1.5, decrease=.5)


def metric_scale(fit, image_height):
    calibration = fit.get('optimization', {}).get('gravity_calibration', {})
    gravity = calibration.get('gravity_normalized_per_s2')
    world_gravity = calibration.get('world_gravity')
    if (not isinstance(gravity, (float, int)) or not isinstance(world_gravity, (float, int))
            or isinstance(gravity, bool) or isinstance(world_gravity, bool)
            or not np.isfinite([gravity, world_gravity]).all() or min(gravity, world_gravity) <= 0):
        raise ValueError('ContPhy metric motion threshold requires a saved gravity calibration with world_gravity; fitted gravity alone is insufficient')
    return gravity * image_height / world_gravity


def description():
    return dict(name=NAME, nominal_endpoint_mass=1., nominal_pulley_attached_load_mass=1.975,
        mass_factors=FACTORS, motion_threshold_world=1.2, rotation_threshold_degrees=35.,
        initialization='Nominal load masses from public generator roles, zero velocity; retain fitted geometry and gravity.',
        readout='First vertical-displacement/rotation threshold crossing, not the final displacement or majority duration.',
        limitations=['Not a factual-parameter counterfactual or a reproduction of Unity settling.',
            'Metric threshold uses locally calibrated image scale.',
            'Generator synthesizes heavier labels by reversing lighter labels; this adapter actually simulates both edits.'])


def reset_parameters(scene, theta):
    from atw.polyworld.rope_controls import mass_controls
    if scene.mass_model != 'massless':
        raise ValueError('ContPhy nominal reset requires the load-only massless rope model')
    result = np.asarray(theta).copy()
    result[3] = 0.
    for control in mass_controls(scene):
        slot = control['parameter_index']
        if slot is None:
            continue
        if control['mass_scope'] != 'body':
            raise ValueError('Nominal reset requires one identified load per mass control')
        result[slot:slot+2] = np.log(1.975 if control['role'] == 'pulley' else 1.)
    return result


def prepare_request(tool, arguments):
    args = dict(arguments)
    if args.get('extension_seconds', 0) != 0:
        raise ValueError('ContPhy reset uses extension_seconds=0; arbitrary extrapolation requires factual protocol')
    if args.get('intervention_seconds') is not None or args.get('rope_initial_state', 'rest') != 'rest':
        raise ValueError('ContPhy reset starts the requested intervention from rest')
    args.update(rope_mass_protocol=NAME, rope_initial_state='rest')
    if tool == 'change_mass':
        factor = FACTORS[args['direction']]
        if args.get('factor') is not None and args['factor'] != factor:
            raise ValueError(f'Configured {NAME} requires factor={factor}; use factual protocol for arbitrary factors')
        args['factor'] = factor
    else:
        if any(v not in FACTORS.values() for v in args.get('mass_scales', {}).values()):
            raise ValueError('ContPhy reset mass factors must be0.5 or1.5')
    return args


def translation_readout(centers_px, times, *, pixels_per_world_unit):
    points, times = np.asarray(centers_px, float), np.asarray(times, float)
    if not np.isfinite(pixels_per_world_unit) or pixels_per_world_unit <= 0:
        raise ValueError('ContPhy movement readout requires a positive calibrated metric scale')
    threshold = 1.2 * pixels_per_world_unit
    if len(points) < 2 or points.shape != (len(times), 2) or not np.isfinite(points).all():
        return dict(status='unavailable', reason='Insufficient finite trajectory samples')
    dy = points[:, 1] - points[0, 1]
    crossings = np.flatnonzero(np.abs(dy) > threshold)
    first = int(crossings[0]) if len(crossings) else None
    return dict(status='measured', protocol=NAME, direction=('stationary' if first is None else
        'down' if dy[first] > 0 else 'up'), first_crossing_seconds=None if first is None else float(times[first]),
        threshold_px=threshold, maximum_vertical_displacement_px=float(np.abs(dy).max()),
        meaning='First vertical excursion over the metric threshold; later reversal does not replace this label.')


def rotation_readout(result):
    threshold = np.deg2rad(35.)
    angle, next_time = 0., result['interval_seconds'][0]
    for row in result['samples']:
        if row['start_seconds'] > next_time + 1e-6 or row['direction'] == 'uncertain':
            return dict(status='unavailable', protocol=NAME, reason='Rotation has an uncovered or ambiguous interval before the first threshold crossing')
        angle += row['angular_velocity_rad_per_s'] * (row['end_seconds'] - row['start_seconds'])
        next_time = row['end_seconds']
        if abs(angle) > threshold:
            return dict(status='measured', protocol=NAME, direction='clockwise' if angle > 0 else 'counterclockwise',
                        first_crossing_seconds=next_time, threshold_rad=float(threshold))
    if next_time < result['interval_seconds'][1] - 1e-6:
        return dict(status='unavailable', protocol=NAME, reason='Incomplete coverage cannot establish no rotation')
    return dict(status='measured', protocol=NAME, direction='stationary', first_crossing_seconds=None,
                threshold_rad=float(threshold))
