"""Private, explicitly referenced inputs for accepted admission-review candidates."""
import copy
from dataclasses import replace
from pathlib import Path
import shutil

from atw.world_modeling.scene_assets import artifact, asset_path, classified_assets, write


def prepare_candidate(inputs, context, objects, output, surface_decision=None):
    """Reuse scene geometry, rebuild dependent native observations, then check admission."""
    from atw.world_modeling.reconstruction_3d import reconstruct
    from atw.world_modeling.scene_swr import load_inputs
    output.mkdir(parents=True)
    prefix = 'admission_'+output.name
    classification = dict(objects=[dict(object_id=o['object_id'],track_id=o['track_id'],
        physical_type=o['physical_type'],reason=o['physical_type_reason']) for o in objects])
    classification_path = output/'physical_types.json'
    write(classification_path,classification)
    arts = [artifact(context,prefix+'_physical_types',classification_path)]
    sources = dict(inputs['tracking']['sources'], physical_types_asset_id=arts[0]['artifact_id'])
    old_gravity_path = asset_path(context,sources['gravity_asset_id'])
    gravity = copy.deepcopy(inputs['gravity'])
    gravity['sources'] = dict(gravity['sources'],physical_types_asset_id=arts[0]['artifact_id'])
    geometry_folder = output/'gravity'
    geometry_folder.mkdir()
    archive = Path(gravity['geometry']['archive'])
    if archive.is_absolute() or len(archive.parts)!=1:
        raise ValueError('Expected a local scene geometry archive filename')
    shutil.copy2(old_gravity_path.parent/archive,geometry_folder/archive)
    shutil.copy2(old_gravity_path.parent/'fixed_camera_calibration.json',geometry_folder/'fixed_camera_calibration.json')
    if surface_decision is not None:
        from atw.world_modeling.scene_geometry import load_geometry, ROOT
        from atw.perception.background_wall_gravity import estimate_background_surface_gravity
        import numpy as np
        geometry = load_geometry(geometry_folder,gravity['geometry'])
        masks = {}
        for obj in objects:
            with np.load(inputs['reconstruction_directory']/'masks'/f"{obj['object_id']}.npz",allow_pickle=False) as data:
                masks[obj['object_id']] = data['masks'].astype(bool)
        summary = estimate_background_surface_gravity(geometry_result=geometry,masks=masks,
            video_path=Path(context.inputs['video_path']),output_dir=geometry_folder/'surface',
            project_root=ROOT,scene_surface_type=surface_decision['scene_surface_type'],device=context.scene.device)
        summary['rigid_surface_decision'] = surface_decision
        gravity['gravity'] = summary
    gravity_path = geometry_folder/'gravity.json'
    write(gravity_path,gravity)
    gravity_art = artifact(context,prefix+'_gravity',gravity_path)
    arts.append(gravity_art)
    sources['gravity_asset_id'] = gravity_art['artifact_id']
    if gravity['gravity']['status']!='completed':
        # No tracking or simulated input is exported for failed numerical geometry.
        return dict(inputs,gravity=gravity), arts
    native_folder = output/'native'
    native_folder.mkdir()
    ctx = replace(context,output_dir=native_folder,artifacts=(*context.artifacts,*arts))
    bundle = classified_assets(ctx,sources['segmentation_asset_id'],sources['physical_types_asset_id'])
    details = reconstruct(sources,ctx,gravity_path,gravity,bundle)
    tracking_path = native_folder/'tracking.json'
    write(tracking_path,dict(protocol_version=1,route='3D+SAM3',video_path=context.inputs['video_path'],
        sources=sources,objects=[{k:o[k] for k in ('object_id','track_id','physical_type')} for o in bundle[-1]],**details))
    tracking_art = artifact(context,prefix+'_tracking',tracking_path)
    arts.append(tracking_art)
    ctx = replace(ctx,artifacts=(*context.artifacts,*arts))
    candidate = load_inputs(dict(tracking_asset_id=tracking_art['artifact_id']),ctx)
    candidate['effective_tracking_asset_id'] = tracking_art['artifact_id']
    return candidate, arts
