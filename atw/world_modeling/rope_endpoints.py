"""Stabilize unbound terminals only with persistent local mask evidence."""
from dataclasses import replace
import numpy as np


def stabilize_static_terminals(scenes, tracks):
    """Separate a stationary terminal region from its unstable skeleton tip.

    Explicit body bindings take precedence. A small endpoint cloud alone is
    insufficient: nearly all observed local masks must agree with a persistent
    foreground region. Large motion and missing observations remain unresolved.
    All thresholds and measurements use original-video pixels or frame fractions.
    """
    from atw.polyworld.rope_swr import _resample_polyline
    output, records = [], []
    for scene in scenes:
        targets = scene.target_positions.copy() * scene.image_height
        initial = scene.initial_positions.copy() * scene.image_height
        pinned = list(scene.endpoint_pinned)
        bound = {b.endpoint_index for b in scene.bodies if b.role == 'endpoint'}
        masks = tracks[scene.rope_track_id]
        for end in (0, 1):
            if pinned[end] or end in bound or scene.wrap_hints:
                continue
            ends = targets[:, -1 if end else 0]
            span = np.ptp(ends, axis=0)
            if np.linalg.norm(span) > .06 * scene.image_height:
                continue
            margin = max(3, round(.004 * scene.image_height))
            lower = np.maximum(np.floor(ends.min(axis=0)).astype(int)-margin, 0)
            upper = np.minimum(np.ceil(ends.max(axis=0)).astype(int)+margin+1,
                               [scene.image_width, scene.image_height])
            x0,y0 = lower; x1,y1 = upper
            observed = [masks[f][y0:y1,x0:x1] for f in scene.frames
                        if f in masks and np.any(masks[f][y0:y1,x0:x1])]
            if len(observed) < max(5, .95 * scene.frame_count):
                continue
            stack = np.stack(observed)
            persistent = stack.mean(axis=0) >= .9
            if persistent.sum() < 9:
                continue
            iou = (stack & persistent).sum(axis=(1,2)) / np.maximum((stack | persistent).sum(axis=(1,2)), 1)
            if np.mean(iou >= .75) < .95:
                continue

            def oriented(line):
                return line[::-1] if end else line

            def exit_index(line):
                outside = np.flatnonzero(np.any((line < lower) | (line >= upper), axis=1))
                return int(outside[0]) if len(outside) else None

            exits = [exit_index(oriented(line)) for line in targets]
            if any(index is None or index == 0 for index in exits):
                continue
            inner = np.median([oriented(line)[index] for line,index in zip(targets,exits)], axis=0)
            pixels = np.argwhere(persistent)[:, ::-1] + lower
            anchor = pixels[np.argmin(np.linalg.norm(pixels-inner, axis=1))].astype(float)

            def reconnect(line):
                ordered = oriented(line)
                index = exit_index(ordered)
                if index is None or index == 0:
                    return None
                revised = _resample_polyline(np.vstack((anchor, ordered[index:])), scene.node_count)
                return revised[::-1] if end else revised

            revised_initial = reconnect(initial)
            if revised_initial is None:
                continue
            targets = np.stack([reconnect(line) for line in targets])
            initial = revised_initial
            pinned[end] = True
            records.append(dict(rope_id=scene.rope_id,track_id=scene.rope_track_id,endpoint_index=end,
                anchor_pixels=anchor.tolist(),terminal_box_pixels=[int(x0),int(y0),int(x1),int(y1)],
                observed_frames=len(observed),persistent_pixels=int(persistent.sum()),
                median_local_mask_iou=float(np.median(iou)),stable_frame_fraction=float(np.mean(iou>=.75)),
                reason='Unbound skeleton tip lies in a persistent stationary terminal region.'))
        output.append(replace(scene,target_positions=targets/scene.image_height,
                              initial_positions=initial/scene.image_height,endpoint_pinned=tuple(pinned))
                      if tuple(pinned)!=scene.endpoint_pinned else scene)
    return tuple(output), records
