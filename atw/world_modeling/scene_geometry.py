"""Shared scene geometry and original support/wall gravity estimators."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from atw.world_modeling.scene_assets import classified_assets, write

ROOT=Path(__file__).resolve().parents[2]


def save_geometry(path,geometry):
    np.savez_compressed(path,rgb=geometry.rgb,intrinsics=geometry.intrinsics,
        camera_poses=geometry.camera_poses,camera_points=geometry.camera_points,
        world_points=geometry.world_points,geometry_confidence=geometry.geometry_confidence,
        geometry_valid=geometry.geometry_valid)
    return dict(archive=path.name,transform=asdict(geometry.transform),
                coordinate_frame='Track4World fixed-camera world; camera x right, y down, z forward',
                units='reconstruction scale; not independently calibrated SI units')


def load_geometry(folder,record):
    from atw.perception.track4world_tool import SceneGeometryResult, Track4WorldTransform
    path=(folder/record['archive']).resolve()
    if not path.is_relative_to(folder.resolve()):raise ValueError('Geometry archive must belong to the gravity asset directory')
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    transform=dict(record['transform'])
    for k in ('original_size_wh','output_size_wh'):transform[k]=tuple(transform[k])
    return SceneGeometryResult(transform=Track4WorldTransform(**transform),**arrays)


def estimate_3d_gravity(arguments,context):
    from atw.perception.track4world_tool import read_video_rgb, preprocess_video, load_track4world_da3_model, run_chronological_scene_geometry
    from atw.perception.fixed_camera import estimate_fixed_camera_calibration, apply_fixed_camera_calibration
    from atw.perception.object_reconstruction import read_packed_mask_archive
    from atw.perception.background_wall_gravity import estimate_background_surface_gravity
    from atw.perception.sam3_agent import GeminiSam3Agent, RIGID_SURFACE_SYSTEM, RIGID_SURFACE_INSTRUCTION
    from atw.world_modeling.physical_classification_impl import render_evidence
    from PIL import Image
    import cv2

    manifest,inventory,index,archive,objects=classified_assets(context,arguments['segmentation_asset_id'],arguments['physical_types_asset_id'])
    runtime=context.scene
    video=Path(context.inputs['video_path']);frames,fps=read_video_rgb(video)
    resized,transform=preprocess_video(frames,maximum_dimension=runtime.image_size)
    model=load_track4world_da3_model(checkpoint=runtime.checkpoint,da3_snapshot=runtime.da3_snapshot,device=runtime.device)
    try:
        geometry=run_chronological_scene_geometry(model,resized,transform=transform,device=runtime.device,
            inference_iterations=runtime.inference_iterations,geometry_threshold=0.0)
    finally:del model
    calibration=estimate_fixed_camera_calibration(geometry.camera_poses,geometry.intrinsics,sample_count=16)
    geometry=apply_fixed_camera_calibration(geometry,calibration)
    write(context.output_dir/'fixed_camera_calibration.json',calibration.to_dict())
    record=save_geometry(context.output_dir/'geometry.npz',geometry)
    # The segmentation schema supplies the actual mask-index path.
    from atw.world_modeling.scene_assets import asset_path
    segmentation=asset_path(context,arguments['segmentation_asset_id'])
    tracks=read_packed_mask_archive(archive,(segmentation.parent/manifest['mask_index']).resolve())
    masks={o['object_id']:np.stack([cv2.resize(tracks[o['track_id']].get(f,np.zeros((manifest['height'],manifest['width']),bool)).astype(np.uint8),transform.output_size_wh,interpolation=cv2.INTER_NEAREST).astype(bool) for f in range(len(frames))]) for o in objects}
    if not any(o['physical_type']=='rigid' for o in objects):
        return dict(status='skipped',reason='inventory_contains_no_rigid_objects'),record
    evidence=context.output_dir/'surface_evidence';evidence.mkdir()
    images=render_evidence(video,inventory,index,archive,evidence)
    # Preserve complete mask evidence in compact overview pages for the surface judgment.
    tiles=[]
    for p in images:
        im=Image.open(p).convert('RGB');im.thumbnail((640,200));tiles.append(im)
    sheet=Image.new('RGB',(1280,200*((len(tiles)+1)//2)),'white')
    for i,im in enumerate(tiles):sheet.paste(im,((i%2)*640,(i//2)*200))
    contact=evidence/'contact_sheet.png';sheet.save(contact)
    agent=GeminiSam3Agent(ROOT);agent.client.trace_dir=context.output_dir/'surface_requests'
    surface_inventory=dict(objects=[dict(physical_type=o['physical_type'],category_noun=o['category_noun']) for o in objects])
    try:
        decision=agent.request_json(operation='rigid_surface_initial',call_id=context.call_id,
            system=RIGID_SURFACE_SYSTEM,instruction=RIGID_SURFACE_INSTRUCTION.replace('{inventory}',json.dumps(surface_inventory)),
            media_paths=[video,contact],validate=agent._normalize_rigid_surface_decision)
        decision=agent._normalize_rigid_surface_decision(decision)
    finally:write(context.output_dir/'surface_vlm_usage.json',agent.usage_report())
    write(context.output_dir/'surface_decision.json',decision)
    if not decision['all_rigid_objects_share_one_surface']:
        return dict(status='skipped',reason='not_all_rigid_objects_share_one_surface',rigid_surface_decision=decision),record
    summary=estimate_background_surface_gravity(geometry_result=geometry,masks=masks,video_path=video,
        output_dir=context.output_dir/'surface',project_root=ROOT,scene_surface_type=decision['scene_surface_type'],device=runtime.device)
    summary['rigid_surface_decision']=decision
    return summary,record
