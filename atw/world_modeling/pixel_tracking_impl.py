"""Static geometry, liquid color masks and canonical observation artifacts."""
import json
from pathlib import Path
import cv2
import numpy as np

from atw.perception import pixel_segmentation_cv as cv
from atw.perception.pixel_segmentation_visuals import coordinate_grid
from atw.perception.pixel_segmentation_refinement import enclosed_domain
from atw.world_modeling import pixel_protocol as p, pixel_masks as masks, pixel_temporal as temporal, pixel_colors, pixel_sources
from atw.world_modeling.pixel_boundary import segment_boundary
from atw.world_modeling.pixel_requests import PixelRequests, prompt
from atw.world_modeling.pixel_containers import refine as refine_containers
from atw.world_modeling.scene_assets import artifact, write


def acquire_runtime(checkpoint):
    """Load GPU dependencies only when actual static inference is requested."""
    from atw.perception.fluid_boundary_sam3 import resolve_checkpoint
    from atw.perception.sam3_runtime import task_sam3_runtime_registry
    runtime, _ = task_sam3_runtime_registry().acquire(resolve_checkpoint(checkpoint))
    return runtime


def run(arguments, context):
    output = context.output_dir
    video = Path(context.inputs['video_path'])
    questions = [{k: q[k] for k in ('id', 'text', 'options')} for q in context.inputs['questions']]
    info = cv.inspect_video(video, 960, 960)
    raw = cv.read_frame(video, 0, info.source_wh)
    first_frame = output/'frame0.png'
    cv2.imwrite(str(first_frame), raw)
    requests = PixelRequests(context)
    runtime = acquire_runtime(context.sam3.checkpoint)
    wall = segment_boundary(video, info, first_frame, requests, runtime, output/'boundary', questions)
    domain = enclosed_domain(wall) & ~wall
    if not domain.any():
        raise ValueError('The observed boundary does not enclose a usable motion domain')
    grid = coordinate_grid(raw, output)
    discovered = requests.call('static_discovery', prompt('discovery')+'\nORIGINAL QUESTIONS WITH OPTIONS:\n'+json.dumps(questions),
                               p.DISCOVERY, [first_frame, grid], p.check_boxes)
    proposals = [dict(id=f'S{i}', **row) for i, row in enumerate(discovered['objects'], 1)]
    write(output/'proposals.json', dict(objects=proposals))
    if not proposals:
        raise ValueError('No stationary structural objects were observed')
    candidates = masks.generate(raw, proposals, domain, runtime)
    raw_candidates = candidates
    np.savez_compressed(output/'candidates.npz', **{c['candidate_id']: c['mask'] for c in candidates})
    write(output/'candidates.json', [{k: v for k, v in c.items() if k != 'mask'} for c in candidates])
    candidates = masks.retain_local_candidates(candidates, proposals, raw.shape[:2])
    if not candidates:
        raise ValueError('SAM3 produced no usable static candidates')
    write(output/'retained_candidates.json', [{k: v for k, v in c.items() if k != 'mask'} for c in candidates])
    pages = masks.boards(raw, proposals, candidates, output)
    decision = requests.call('static_selection', prompt('selection')+'\nPROPOSALS: '+json.dumps(proposals),
        p.static_selection([c['candidate_id'] for c in candidates]), [first_frame, *pages], p.check_selection)
    if not decision['objects']:
        raise ValueError('No supported static masks were selected')
    labels, rows, overview = masks.compose(raw, candidates, decision)
    labels, rows = refine_containers(raw, labels, rows, proposals, raw_candidates, domain, requests,
                                    output/'container_refinement', first_frame)
    overview = masks.render_overview(raw, labels, rows)
    write(output/'inventory.json', dict(objects=rows))
    np.savez_compressed(output/'selected.npz', labels=labels, wall=wall, domain=domain)
    cv2.imwrite(str(output/'static_overview.jpg'), overview)
    reference = cv2.resize(labels, info.working_wh, interpolation=cv2.INTER_NEAREST)
    wall = cv2.resize(wall.astype(np.uint8), info.working_wh, interpolation=cv2.INTER_NEAREST).astype(bool)
    domain = cv2.resize(domain.astype(np.uint8), info.working_wh, interpolation=cv2.INTER_NEAREST).astype(bool) & ~wall
    palette = cv.learn_palette(video, info)
    np.savez_compressed(output/'palette.npz', **palette)
    groups, evidence = pixel_colors.prepare(video, info, palette, reference, domain, wall, output/'liquid_groups')
    group_ids = [group['group_id'] for group in groups]
    names = requests.call('liquid_color_names', prompt('color_names')+'\nGROUP IDS: '+json.dumps(group_ids)
        +'\nORIGINAL QUESTIONS: '+json.dumps(questions), p.color_names(group_ids), [video, *evidence],
        lambda value: p.check_color_names(value, group_ids))
    write(output/'liquid_color_names.json', names)
    by_id = {row['group_id']: row['name'] for row in names['groups']}
    colors = [dict(object_id=f'F{i}', group_id=group['group_id'], name=by_id[group['group_id']],
                   principal_rgb=group['principal_rgb'], palette_ids=group['palette_ids'])
              for i, group in enumerate(groups, 1)]
    write(output/'tracking_settings.json', dict(identity_semantics='liquid_color_groups',
        mode='chromatic_change_or_brightening', chromatic_threshold=12, positive_mean_threshold=24, colors=colors))
    objects = temporal.emit(video, info, palette, reference, rows, colors,
                            output/'tracking', domain_mask=domain, boundary_mask=wall)
    areas = json.loads((output/'tracking/areas.json').read_text())
    empty = [row['object_id'] for row in colors if not any(f['areas'][row['object_id']] for f in areas)]
    if empty:
        raise ValueError('Identified liquid colors have no observed mask pixels: '+', '.join(empty))
    retained = []
    def keep(kind, path, media_type='application/json', source_kind='derived'):
        row = artifact(context, kind, path, media_type)
        row['source_kind'] = source_kind
        retained.append(row)
        return row['artifact_id']
    assets = dict(
        masks_asset_id=keep('pixel_masks', output/'tracking/masks/identities.npz', 'application/octet-stream'),
        static_masks_asset_id=keep('pixel_static_masks', output/'selected.npz', 'application/octet-stream'),
        object_index_asset_id=keep('pixel_objects', output/'tracking/object_index.json'),
        areas_asset_id=keep('pixel_areas', output/'tracking/areas.json'),
        timing_asset_id=keep('pixel_color_timing', output/'tracking/color_timing.json'),
        settings_asset_id=keep('pixel_settings', output/'tracking_settings.json'),
        color_groups_asset_id=keep('pixel_color_groups', output/'liquid_groups/groups.json'),
        color_names_asset_id=keep('pixel_color_names', output/'liquid_color_names.json'),
        overview_asset_id=keep('pixel_overview', output/'static_overview.jpg', 'image/jpeg'),
        first_frame_asset_id=keep('pixel_first_frame', first_frame, 'image/png', 'observation'),
        palette_asset_id=keep('pixel_palette', output/'palette.npz', 'application/octet-stream'))
    assets['container_refinement_asset_id'] = keep('pixel_container_refinement', output/'container_refinement/summary.json')
    source_output = output/'source_binding'
    source_manifest = pixel_sources.bind(raw, labels, rows, domain, objects,
        output/'tracking/masks/identities.npz', info.frame_count, info.fps, video, source_output)
    assets['source_masks_asset_id'] = keep('pixel_source_masks', source_output/'identities.npz', 'application/octet-stream')
    source_manifest['source_assets'] = {key: assets[key] for key in
        ('masks_asset_id', 'static_masks_asset_id', 'color_groups_asset_id', 'source_masks_asset_id')}
    write(source_output/'binding.json', source_manifest)
    assets['source_binding_asset_id'] = keep('pixel_source_binding', source_output/'binding.json')
    write(output/'tracking.json', dict(protocol_version=1, route='2D+PIXEL', representation='observed_pixel_masks',
        identity_semantics='liquid_color_groups', video_path=str(video), sources=dict(routing_call_id=arguments['routing_call_id']), objects=objects, assets=assets,
        frame_count=info.frame_count, fps=info.fps, frame_scope=[0, info.frame_count-1],
        coordinates=dict(units='pixels', x_direction='right', y_direction='down', origin='top_left',
                         mask_width=info.mask_wh[0], mask_height=info.mask_wh[1],
                         source_width=info.source_wh[0], source_height=info.source_wh[1]),
        mask_encoding=dict(dtype='uint16', archive_key='frame_{frame_index:06d}',
                           background_label=0, unknown_label=int(cv.UNKNOWN_ID),
                           static_masks_coordinates='original source frame pixels'),
        limitations=['Fixed camera and a single stationary enclosing frame are required; frame zero is assumed liquid-free without VLM verification.',
                     'Small or brief colors can fall below sampled interior support; similar colors may remain unresolved.',
                     'Color names are descriptive and may repeat; group IDs and principal RGB values identify colors.',
                     'Color-group masks share same-color liquid; separate source assets estimate outlet provenance and can remain unassigned.',
                     'Source binding assumes compact upper-boundary outlets and downward flow; long mixing or occlusion can invalidate provenance.',
                     'Newly occupied pixels include motion and splitting and are not measured emission events.',
                     'Masks describe visible material, not hidden geometry, 3D reconstruction, simulation or material particles.']))
    tracking_id = keep('pixel_tracking', output/'tracking.json')
    return dict(route='2D+PIXEL', tracking_asset_id=tracking_id, object_count=len(objects)), retained
