"""Construct object geometry and tracks from scene observations."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from atw.world_modeling.scene_assets import asset_path, write
from atw.world_modeling.scene_geometry import load_geometry, ROOT


class TrackingInputs(SimpleNamespace):
    """In-memory backend input; model confidences absent upstream remain unknown."""
    @property
    def rigid_objects(self):return tuple(o for o in self.objects if o.physical_type=='rigid')
    @property
    def cloth_objects(self):return tuple(o for o in self.objects if o.physical_type=='cloth')
    @property
    def soft_body_objects(self):return tuple(o for o in self.objects if o.physical_type=='soft_body')


def rigid_shapes(context,inventory,index,archive,objects):
    from atw.perception.sam3_agent import GeminiSam3Agent, RIGID_SHAPE_GUIDANCE
    from atw.world_modeling.physical_classification_impl import render_evidence
    from atw.world_modeling import sam3_protocol as c
    rigids=[o for o in objects if o['physical_type']=='rigid']
    if not rigids:return {}
    output=context.output_dir/'rigid_shapes';output.mkdir()
    subset=dict(inventory,objects=[o for o in inventory['objects'] if o['object_id'] in {r['object_id'] for r in rigids}])
    images=render_evidence(Path(context.inputs['video_path']),subset,index,archive,output)
    ids=[o['track_id'] for o in rigids]
    schema=c.obj(objects=c.array(c.obj(track_id=dict(type='string',enum=ids),shape=dict(type='string',enum=['box','sphere','cylinder','irregular']),
        confidence=dict(type='number',minimum=0,maximum=1),reason=c.TEXT),minItems=len(ids),maxItems=len(ids)))
    def validate(v):
        c.validate(schema,v,'rigid_shapes');c.unique_ids(v['objects'],'track_id','rigid_shapes')
        return v
    agent=GeminiSam3Agent(ROOT);agent.client.trace_dir=output/'requests'
    try:
        result=agent.request_json(operation='rigid_shapes',call_id=context.call_id,system='Classify rigid geometry without changing physical types.',
            instruction=RIGID_SHAPE_GUIDANCE+'\nAll supplied anonymous tracks are already classified as rigid. Return geometry only.\n'+json.dumps(dict(track_ids=ids))+c.prompt(schema),
            media_paths=[Path(context.inputs['video_path']),*images],validate=validate)
        validate(result)
    finally:write(output/'vlm_usage.json',agent.usage_report())
    bytrack={o['track_id']:o for o in result['objects']}
    decisions={o['object_id']:bytrack[o['track_id']] for o in rigids};write(output/'shapes.json',decisions)
    return decisions


def reconstruct(arguments,context,gravity_path,gravity,bundle):
    from atw.perception.pipeline.scene import SceneObjects, SceneData, TrackedObjectSpec, select_area_reference_frames, resized_tracking_masks, _save_result_geometry
    from atw.perception.pipeline.reconstruction import prepare_rigid_reconstruction_manifest, run_sam3d_rigid_reconstruction
    from atw.perception.pipeline.surface import wall_box_reference_priors
    from atw.perception.rigid_motion import classify_rigid_motion
    from atw.perception.reference_shape import classify_soft_body_reference_shapes
    from atw.perception.object_reconstruction import read_packed_mask_archive
    from atw.perception.track4world_tool import read_video_rgb
    from atw.perception.pointcloud_reconstruction_assets import write_reconstruction_manifest
    from atw.perception.pipeline.config import TrackingConfig
    from atw.perception.pipeline.tracking import run_tracking
    from atw.perception.pipeline.rendering import render_tracking

    manifest,inventory,index,archive,objects=bundle
    if not objects:raise ValueError('3D reconstruction requires at least one retained object')
    geometry=load_geometry(gravity_path.parent,gravity['geometry'])
    video=Path(context.inputs['video_path']);frames,fps=read_video_rgb(video)
    if len(frames)!=len(geometry.rgb):raise ValueError('Shared geometry and video frame counts differ')
    segmentation_path=asset_path(context,arguments['segmentation_asset_id'])
    tracks=read_packed_mask_archive(archive,(segmentation_path.parent/manifest['mask_index']).resolve())
    shapes=rigid_shapes(context,inventory,index,archive,objects)
    native=TrackingInputs(manifest_path=context.output_dir/'input.json',video_path=video,
        frame_count=manifest['frame_count'],width=manifest['width'],height=manifest['height'],fps=manifest['fps'],
        mask_width=manifest['width'],mask_height=manifest['height'],mask_archive_path=archive,
        mask_index_path=(segmentation_path.parent/manifest['mask_index']).resolve(),
        inventory_path=(segmentation_path.parent/manifest['inventory']).resolve(),tracking_manifest_path=segmentation_path,
        objects=tuple(SimpleNamespace(target_id=o['object_id'],track_id=o['track_id'],physical_type=o['physical_type'],
            physical_type_reason=o['physical_type_reason'],physical_type_confidence=None,
            rigid_shape=shapes.get(o['object_id'],{}).get('shape'),rigid_shape_confidence=shapes.get(o['object_id'],{}).get('confidence'),
            rigid_shape_reason=shapes.get(o['object_id'],{}).get('reason')) for o in objects))
    specs=SceneObjects.from_input(native);output=context.output_dir/'reconstruction';output.mkdir()
    selections=select_area_reference_frames(frames_rgb=frames,tracks=tracks,output_dir=output,timings={},specs=specs.all,objects=specs)
    masks=resized_tracking_masks(tracks=tracks,frame_count=len(frames),output_size_wh=geometry.transform.output_size_wh,objects=specs)
    (output/'masks').mkdir()
    for name,mask in masks.items():np.savez_compressed(output/'masks'/f'{name}.npz',masks=mask)
    _save_result_geometry(output/'geometry/reference_0000',geometry,np.logical_or.reduce(list(masks.values())))
    write(output/'fixed_camera_calibration.json',json.loads((gravity_path.parent/'fixed_camera_calibration.json').read_text()))
    write(output/'surface/gravity_estimate.json',gravity['gravity'])
    motion=classify_rigid_motion(output_dir=output/'rigid_motion',rigid_target_ids=[o.target_id for o in native.rigid_objects],
        tracks={o.target_id:tracks[o.track_id] for o in native.objects},frame_count=len(frames),mask_shape=(manifest['height'],manifest['width']))
    for name in ('rigid','soft_body'):(output/'models'/name).mkdir(parents=True)
    if specs.rigids:
        shape_values={k:v['shape'] for k,v in shapes.items()}
        priors=wall_box_reference_priors(gravity['gravity'],motion,shape_values)
        # This is the original worker entry point: it owns the <0.75 static-IoU retry.
        inputs=prepare_rigid_reconstruction_manifest(output_dir=output,geometry_result=geometry,selections=selections,tracks=tracks,
            rigid_specs=specs.rigids,rigid_shapes=shape_values,debug=context.scene.debug,reference_surface_priors=priors,
            tracking_branches={k:v['tracking_branch'] for k,v in motion['objects'].items()})
        run_sam3d_rigid_reconstruction(manifest_path=inputs,output_dir=output,sam3d_environment=context.scene.sam3d_environment)
    if native.soft_body_objects:
        soft_shapes=classify_soft_body_reference_shapes(project_root=ROOT,output_dir=output/'soft_body_reference_shapes',frames_rgb=geometry.rgb,masks=masks,
            reference_frames={o.target_id:selections[o.target_id].reference_frame for o in native.soft_body_objects})
        inputs=prepare_rigid_reconstruction_manifest(output_dir=output,geometry_result=geometry,selections=selections,tracks=tracks,
            rigid_specs=tuple(TrackedObjectSpec(o.target_id,o.track_id) for o in native.soft_body_objects),
            rigid_shapes={k:v['shape'] for k,v in soft_shapes.items()},physical_type='soft_body',debug=context.scene.debug)
        run_sam3d_rigid_reconstruction(manifest_path=inputs,output_dir=output,sam3d_environment=context.scene.sam3d_environment,physical_type='soft_body')
    write_reconstruction_manifest(output/'reconstruction_manifest.json',sam3_result_dir=segmentation_path.parent,
        original_size_wh=geometry.transform.original_size_wh,model_size_wh=geometry.transform.output_size_wh,
        tracked_objects=[dict(target_id=o.target_id,track_id=o.track_id,physical_type=o.physical_type) for o in native.objects])
    shared=SceneData(frames,fps,tracks,geometry,selections,masks)
    runtime=context.scene
    tracking=run_tracking(TrackingConfig(reconstruction_dir=output,output_dir=context.output_dir/'tracking',
        checkpoint=runtime.checkpoint,da3_snapshot=runtime.da3_snapshot,device=runtime.device,
        inference_iterations=runtime.inference_iterations,debug=runtime.debug),shared=shared,tracking_input=native)
    debug=None
    if runtime.debug:
        debug=render_tracking(tracking,output_dir=context.output_dir/'debug',workers=4)
    return dict(representation='3d',reconstruction_manifest='reconstruction/reconstruction_manifest.json',
        tracking_directory=str(tracking.relative_to(context.output_dir)),debug=debug,
        gravity_source=str(gravity_path.relative_to(context.run_dir)),
        coordinate_frame='shared fixed-camera Track4World world',
        gravity_available=gravity['gravity']['status']=='completed',
        limitations=[] if gravity['gravity']['status']=='completed' else ['The reviewed gravity asset has no usable surface constraint; tracking is unconstrained.'])
