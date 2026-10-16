"""Conservative no-effect findings from retained paired Rope worlds."""
import numpy as np


def unchanged_goal_note(backend, raw, object_ids):
    edited = backend.rollouts[raw['rollout_id']]
    comparison = edited.get('intervention_comparison', {})
    control = backend.rollouts.get(comparison.get('control_rollout_id'))
    if control is None or not all(world.get('simulation_stability', {}).get(
            'usable_for_downstream_reasoning', False) for world in (control, edited)):
        return ''
    start, end = raw.get('interval_seconds', [
        edited.get('readout_start_seconds', 0), backend._rollout_end(edited)])

    def centers(world, name):
        return [(row['frame_index'], row.get('xy_normalized', row.get('center_xy_normalized')))
                for row in world.get('trajectories', {}).get(name, [])
                if start <= row['frame_index']/backend.fps <= end]

    def equal_centers(name):
        left, right = centers(control, name), centers(edited, name)
        return (len(left) >= 2 and left == right
                and all(value is not None and np.isfinite(value).all() for _, value in left))

    def equal_rope(rope_id):
        left, right = (world.get('geometry', {}).get(rope_id) for world in (control, edited))
        if left is None or right is None or np.shape(left) != np.shape(right):
            return False
        frames = [row['frame_index'] for row in edited.get('trajectories', {}).get(rope_id, [])]
        if len(frames) != len(right):
            return False
        selected = (np.asarray(frames)/backend.fps >= start) & (np.asarray(frames)/backend.fps <= end)
        a, b = np.asarray(left)[selected], np.asarray(right)[selected]
        # Identical material geometry and centers imply identical no-slip rotation
        # throughout the interval, unlike equal final angles or sparse samples.
        return len(a) >= 2 and np.isfinite(a).all() and np.array_equal(a, b)

    findings, changed_rotations = [], []
    for name in sorted(object_ids):
        same_center = equal_centers(name)
        quantities = []
        if name in raw.get('center_translation', {}) and same_center:
            quantities.append('translation')
        rotations = comparison.get('objects', {}).get(name, {}).get('rotations', [])
        if (name in raw.get('pulley_rotation', {}).get('objects', {}) and same_center and rotations
                and all(row.get('status') == 'measured'
                        and row.get('edited_minus_control_angle_rad') == 0
                        and equal_rope(row['rope_id']) for row in rotations)):
            quantities.append('rotation')
        if quantities:
            findings.append(f"{name} ({' and '.join(quantities)})")
        if name in raw.get('pulley_rotation', {}).get('objects', {}):
            for row in rotations:
                if row.get('status') == 'measured' and row.get('edited_minus_control_angle_rad') != 0:
                    before, after = row['control']['signed_angle_rad'], row['edited']['signed_angle_rad']
                    scope = row['control']['interval_seconds']
                    changed_rotations.append(
                        f"{name} rotation CHANGED on {row['rope_id']}: baseline {before:.6g} rad, "
                        f"intervention {after:.6g} rad (net angles over {scope[0]:g}–{scope[1]:g} seconds)")
    if not findings:
        return ''
    return (' Goal intervention comparison: '+ '; '.join(findings)
            + f' is exactly unchanged from the matched unedited baseline over {start:g}–{end:g} seconds. '
            + ('. '.join(changed_rotations)+'. ' if changed_rotations else '')
            + 'Judge the goal quantity separately: unchanged translation does NOT imply unchanged rotation, '
            'or vice versa. An option with no effect on the goal quantity is unrelated to achieving that '
            'change, even if the baseline already has the requested direction. Unchanged does not mean stationary.')
