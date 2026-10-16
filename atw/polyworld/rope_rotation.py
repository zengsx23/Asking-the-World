"""No-slip pulley rotation inferred from material rope motion in image coordinates."""
from __future__ import annotations

import numpy as np
from atw.polyworld.evidence_actions import query_requirement


def summarize_center_translation(centers_xy_px, times, *, tolerance_px=1.):
    """Measure translation across all retained samples, independently of spin."""
    centers, times = np.asarray(centers_xy_px, float), np.asarray(times, float)
    result = dict(sample_count=len(times), stationary_tolerance_px=tolerance_px,
                  interval_seconds=[float(times[0]), float(times[-1])] if len(times) else None)
    if (centers.shape != (len(times), 2) or len(times) < 2
            or not np.isfinite(centers).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        return result | query_requirement("Need at least two finite, ordered center samples.",
            "Query pulley translation over a covered interval with at least two samples; do not infer stationary from insufficient samples.")
    delta = centers - centers[0]
    maximum = float(np.linalg.norm(delta, axis=1).max())
    return result | dict(status="measured", moving=maximum > tolerance_px,
        max_displacement_px=maximum, net_displacement_xy_px=delta[-1].tolist(),
        net_displacement_samples=[dict(time_seconds=float(times[i]), net_displacement_xy_px=delta[i].tolist())
            for i in np.unique(np.linspace(0, len(times)-1, min(5, len(times))).astype(int))],
        net_direction_xy=["stationary" if abs(d) <= tolerance_px else positive if d > 0 else negative
                          for d, positive, negative in zip(delta[-1], ("right", "down"), ("left", "up"))])


def infer_pulley_rotation(rope_xy, pulley_xy, radius, times, *, contact_tolerance, taut_frames=None):
    """Subtract pulley translation before measuring tangential motion; image y points down."""
    rope, centers, times = np.asarray(rope_xy, float), np.asarray(pulley_xy, float), np.asarray(times, float)
    if (rope.ndim != 3 or rope.shape[-1] != 2 or rope.shape[1] < 2
            or len(rope) != len(times) or centers.shape != (len(times), 2)
            or len(times) < 2 or np.any(np.diff(times) <= 0) or radius <= 0 or contact_tolerance <= 0
            or not np.isfinite([radius, contact_tolerance]).all()
            or not all(np.isfinite(a).all() for a in (rope, centers, times))):
        raise ValueError("Supply matching finite material-node trajectories, pulley centers, increasing times and positive geometry")
    midpoint = (rope[1:] + rope[:-1]) * .5
    center = (centers[1:] + centers[:-1]) * .5
    velocity = (np.diff(rope, axis=0) - np.diff(centers, axis=0)[:, None]) / np.diff(times)[:, None, None]
    start, segment = midpoint[:, :-1], np.diff(midpoint, axis=1)
    fraction = np.clip(np.sum((center[:, None] - start) * segment, axis=-1)
                       / np.maximum(np.sum(segment**2, axis=-1), 1e-12), 0, 1)
    radial = start + fraction[..., None] * segment - center[:, None]
    v = velocity[:, :-1] + fraction[..., None] * np.diff(velocity, axis=1)
    distance = np.linalg.norm(radial, axis=-1)
    contact = np.abs(distance - radius) <= contact_tolerance
    supported_intervals = np.ones(len(times)-1, bool)
    if taut_frames is not None:
        taut = np.asarray(taut_frames, bool)
        if taut.shape != (len(times),):
            raise ValueError('taut_frames must match the supplied time samples')
        supported_intervals = taut[:-1] & taut[1:]
        contact &= supported_intervals[:, None]
    omega = (radial[..., 0] * v[..., 1] - radial[..., 1] * v[..., 0]) / np.maximum(distance**2, 1e-12)
    rows = []
    for i in range(len(times) - 1):
        values = omega[i, contact[i]]
        if not len(values):
            continue
        median = float(np.median(values))
        moving = values[np.abs(values) > .02]
        agreement = float(np.mean(np.sign(moving) == np.sign(median))) if len(moving) else 1.
        direction = ("uncertain" if agreement < .75 else "stationary" if abs(median) <= .02
                     else "clockwise" if median > 0 else "counterclockwise")
        rows.append(dict(start_seconds=float(times[i]), end_seconds=float(times[i+1]),
                         angular_velocity_rad_per_s=median, direction=direction,
                         contact_segments=int(len(values)), sign_agreement=agreement))
    durations = {d:sum(r["end_seconds"]-r["start_seconds"] for r in rows if r["direction"]==d)
                 for d in ("clockwise","counterclockwise","stationary","uncertain")}
    interval_duration = float(times[-1] - times[0])
    rotating = ("clockwise", "counterclockwise")
    dominant = max(rotating, key=lambda name: (durations[name], name == "clockwise"))
    covered = sum(durations.values())
    moving_duration = sum(durations[d] for d in rotating)
    # Do not let a tiny directional majority turn a mostly stationary or reversing
    # wheel into evidence for sustained unidirectional rotation.
    direction = (None if not rows else
                 "uncertain" if durations['uncertain'] >= covered * .5 else
                 "stationary" if durations['stationary'] > covered * .5 else
                 dominant if moving_duration and durations[dominant] >= moving_duration * .75 else "uncertain")
    supported = [r for r in rows if r['direction'] != 'uncertain']
    angles = [(r['end_seconds'] - r['start_seconds']) * r['angular_velocity_rad_per_s'] for r in supported]
    moving_direction = (dominant if moving_duration and durations[dominant] >= moving_duration * .75
                        else 'uncertain' if moving_duration else None)
    angle_by_direction = {d: float(sum(abs(r['angular_velocity_rad_per_s']) *
        (r['end_seconds'] - r['start_seconds']) for r in supported if r['direction'] == d))
        for d in rotating}
    # Absence requires coverage; a stopped majority cannot erase earlier rotation.
    rotation_occurred = (True if moving_duration else False
        if covered >= interval_duration * (1 - 1e-6) and not durations['uncertain'] else None)
    result = dict(direction=direction,
                dominant_state=direction, moving_direction=moving_direction,
                rotation_occurred=rotation_occurred, angle_by_direction_rad=angle_by_direction,
                direction_basis=("Predominantly stationary when stationary exceeds half the covered time; "
                    "otherwise require 75% agreement among moving intervals for a direction. "
                    "Balanced reversals or conflicting contact signs are uncertain, not stationary. "
                    "This is an interval summary, not a claim that no transient rotation occurred."),
                signed_angle_rad=float(sum(angles)), absolute_angle_rad=float(sum(abs(a) for a in angles)),
                coverage_fraction=covered / interval_duration,
                interval_seconds=[float(times[0]), float(times[-1])], samples=rows,
                direction_seconds=durations, uncovered_seconds=max(0.,float(times[-1]-times[0])-sum(durations.values())),
                assumption="Taut rope contacts the pulley without tangential slip; fitted material-node motion is a model estimate, not observed wheel texture rotation.",
                sign_convention="Image x right, y down: positive angular velocity is clockwise. Pulley translation is subtracted.")
    if taut_frames is not None:
        result['slack_excluded_seconds'] = float(np.diff(times)[~supported_intervals].sum())
        result['slack_limitation'] = 'Loose-rope shape and no-slip wheel spin are not identified during slack. Such intervals are excluded, not counted as stationary.'
    # Integrate every interval before sampling cumulative angles. A missing or
    # ambiguous interval leaves later angles relative to the start unavailable.
    by_start = {r['start_seconds']: r for r in rows}
    cumulative = [0.]
    for start, stop in zip(times[:-1], times[1:]):
        row = by_start.get(float(start))
        cumulative.append(cumulative[-1] + (stop-start)*row['angular_velocity_rad_per_s']
            if cumulative[-1] is not None and row is not None and row['direction'] != 'uncertain' else None)
    result['net_angle_samples'] = [dict(time_seconds=float(times[i]),
        net_angle_rad=None if cumulative[i] is None else float(cumulative[i]))
        for i in np.unique(np.linspace(0, len(times)-1, min(5, len(times))).astype(int))]
    if not rows:
        result.update(query_requirement("No supported rope/pulley contact samples in this interval.",
            "Check the pulley binding and query an interval with material-rope contact. Rendered wheel texture cannot establish rotation direction; retain this calculation tool. If no supported contact evidence exists, request per-question original-video physical reasoning."))
    return result
