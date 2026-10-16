"""Observed liquid masks grouped by color, without source or particle identities."""
from atw.world_modeling.pixel_foreground import changed
import cv2
import numpy as np
from atw.perception import pixel_segmentation_cv as cv
from atw.perception.pixel_segmentation_materials import mask_archive
from atw.world_modeling.scene_assets import write


def emit(video, info, palette, reference, static, colors, output, *, domain_mask, boundary_mask):
    """Retain one mask per liquid color across the entire original clip.

    Newly occupied pixels compare adjacent observed masks at fixed image coordinates;
    they measure image change, not emitted volume or a new release event.
    """
    output.mkdir()
    masks = output/'masks'
    masks.mkdir()
    objects = []
    for i, row in enumerate(static, 1):
        objects.append(dict(object_id=row['id'], track_id=f'track_{i:04d}', mask_label=i,
            object_kind='static', name=row['name'], description=row['description']))
    for i, row in enumerate(colors, len(static)+1):
        objects.append(dict(object_id=row['object_id'], track_id=f'track_{i:04d}', mask_label=i,
            object_kind='fluid', name=row['name'], description='All observed liquid pixels of this color.',
            group_id=row['group_id'], principal_rgb=row['principal_rgb'], palette_ids=row['palette_ids']))
    boundary_id = len(objects)+1
    objects.append(dict(object_id='B1', track_id=f'track_{boundary_id:04d}', mask_label=boundary_id,
        object_kind='boundary', name='boundary', description='Observed stationary outer enclosure.'))
    fluid_objects = [o for o in objects if o['object_kind']=='fluid']
    baseline = cv.read_frame(video, 0, info.working_wh)
    previous = {o['object_id']: np.zeros(reference.shape, bool) for o in fluid_objects}
    timing = {o['object_id']: dict(first_visible_frame=None, last_visible_frame=None, visible_intervals=[]) for o in fluid_objects}
    measurements = []
    areas = []
    allowed_domain = domain_mask & ~boundary_mask & (reference==0)
    with mask_archive(masks/'identities.npz') as archive:
        for frame, raw in enumerate(cv.iter_frames(video, info.working_wh)):
            palette_labels = palette['lut'][cv.encode_bgr(raw)]
            allowed = allowed_domain.copy()
            allowed &= changed(raw, baseline)
            result = reference.copy()
            result[~domain_mask] = 0
            frame_measurements = {}
            for obj in fluid_objects:
                oid = obj['object_id']
                selected = np.isin(palette_labels, obj['palette_ids']) & allowed
                result[selected] = obj['mask_label']
                pixels = int(selected.sum())
                frame_measurements[oid] = dict(pixel_count=pixels,
                    newly_occupied_pixel_count=int((selected & ~previous[oid]).sum()))
                previous[oid] = selected
                if pixels:
                    row = timing[oid]
                    if row['first_visible_frame'] is None:
                        row['first_visible_frame'] = frame
                    row['last_visible_frame'] = frame
                    spans = row['visible_intervals']
                    if spans and spans[-1]['end_frame'] == frame-1:
                        spans[-1]['end_frame'] = frame
                    else:
                        spans.append(dict(start_frame=frame, end_frame=frame))
            result[boundary_mask] = boundary_id
            scaled = cv2.resize(result, info.mask_wh, interpolation=cv2.INTER_NEAREST)
            with archive.open(f'frame_{frame:06d}.npy', 'w') as member:
                np.save(member, scaled, allow_pickle=False)
            areas.append(dict(frame=frame, areas={o['object_id']: int((scaled==o['mask_label']).sum()) for o in objects}))
            measurements.append(dict(frame=frame, time_seconds=frame/info.fps, colors=frame_measurements))
    if len(areas) != info.frame_count:
        (masks/'identities.npz').unlink()
        raise ValueError(f'Decoded {len(areas)} frames; expected {info.frame_count}')
    for row in timing.values():
        row['first_visible_seconds'] = None if row['first_visible_frame'] is None else row['first_visible_frame']/info.fps
        row['last_visible_seconds'] = None if row['last_visible_frame'] is None else row['last_visible_frame']/info.fps
    write(output/'object_index.json', dict(objects=objects))
    write(output/'areas.json', areas)
    write(output/'color_timing.json', dict(source_video_path=str(video), frame_scope=[0, info.frame_count-1],
        fps=info.fps, coordinate_width=info.working_wh[0], coordinate_height=info.working_wh[1],
        units='working-image pixels; zero-based frame indices; seconds from frame zero',
        semantics='Visibility and adjacent-frame newly occupied pixels by color. Motion, splitting and later same-color releases can all add pixels. These are not source assignments or inferred emission events.',
        colors=timing, frames=measurements))
    return objects
