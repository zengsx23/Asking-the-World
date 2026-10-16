"""Explicit reviewed-gravity handoff into physical-type reconstruction/tracking."""
import json
from atw.world_modeling import sam3_protocol as c
from atw.world_modeling.request_scope import agentic_requests
from atw.world_modeling.scene_assets import asset_path, artifact, classified_assets, write

DEFINITION=dict(protocol_version=1,name='reconstruct_and_track_scene',
    description='Reconstruct and track SAM3 objects using the selected route, physical types and reviewed gravity asset.',
    stages=['stage_one'],input_schema=c.obj(gravity_asset_id=c.TEXT,segmentation_asset_id=c.TEXT,physical_types_asset_id=c.TEXT),
    output_schema=c.obj(route=dict(type='string',enum=['3D+SAM3','2D+SAM3']),tracking_asset_id=c.TEXT,
        object_count=dict(type='integer',minimum=0)))

def validate_arguments(v):return c.validate(DEFINITION['input_schema'],v,'tool.arguments')
def validate_output(v):return c.validate(DEFINITION['output_schema'],v,'tool.data')


@agentic_requests
def execute(arguments,context):
    result=dict(protocol_version=1,call_id=context.call_id,tool_name=DEFINITION['name'],status='error',data={},artifacts=[],error=None)
    try:validate_arguments(arguments)
    except ValueError as exc:
        result['error']=dict(code='invalid_arguments',message=str(exc));return result
    try:
        gravity_path=asset_path(context,arguments['gravity_asset_id'])
        gravity=json.loads(gravity_path.read_text())
        if gravity['route']=='2D+PIXEL':
            result['error']=dict(code='unsupported_operation',message='Use track_pixel_scene with the successful 2D+PIXEL routing_call_id; this reconstruction tool requires SAM3 assets.')
            return result
        if gravity['route'] not in ('3D+SAM3','2D+SAM3'):raise ValueError('Unsupported gravity route')
        from pathlib import Path
        if Path(gravity['video_path']).resolve()!=Path(context.inputs['video_path']).resolve():raise ValueError('Gravity video mismatch')
        if gravity['route']=='3D+SAM3':
            for k in ('segmentation_asset_id','physical_types_asset_id'):
                if gravity['sources'][k]!=arguments[k]:raise ValueError(f'Gravity was computed from a different {k}')
        bundle=classified_assets(context,arguments['segmentation_asset_id'],arguments['physical_types_asset_id'])
        types={o['physical_type'] for o in bundle[-1]}
        if gravity['route']=='3D+SAM3' and 'rope' in types:raise ValueError('3D rope reconstruction is not supported; review the route or classification')
        if gravity['route']=='2D+SAM3' and not types<={'rigid','rope'}:raise ValueError('2D SAM3 tracking currently supports rigid and rope objects only')
        if gravity['route']=='3D+SAM3':
            from atw.world_modeling.reconstruction_3d import reconstruct
            details=reconstruct(arguments,context,gravity_path,gravity,bundle)
        else:
            from atw.world_modeling.reconstruction_2d import reconstruct
            details=reconstruct(context,bundle)
        path=context.output_dir/'tracking.json'
        write(path,dict(protocol_version=1,route=gravity['route'],video_path=context.inputs['video_path'],sources=arguments,
            objects=[{k:o[k] for k in ('object_id','track_id','physical_type')} for o in bundle[-1]],**details))
        art=artifact(context,'tracking',path)
        result.update(status='ok',data=dict(route=gravity['route'],tracking_asset_id=art['artifact_id'],object_count=len(bundle[-1])),artifacts=[art])
    except Exception as exc:result['error']=dict(code='execution_failed',message=f'{type(exc).__name__}: {exc}')
    return result
