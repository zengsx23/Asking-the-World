"""Route-aware gravity with an independently reviewable geometry asset."""
from atw.world_modeling import sam3_protocol as c
from atw.world_modeling.request_scope import agentic_requests
from atw.world_modeling.scene_assets import write, artifact, route_result

DEFINITION=dict(protocol_version=1,name='estimate_scene_gravity',
    description='Estimate gravity for the selected route; retain 3D surface and geometry evidence for review.',
    stages=['stage_one'],input_schema=dict(type='object',properties=dict(
        routing_call_id=c.TEXT,segmentation_asset_id=c.TEXT,physical_types_asset_id=c.TEXT),
        required=['routing_call_id'],additionalProperties=False),
    output_schema=c.obj(route=dict(type='string',enum=['3D+SAM3','2D+SAM3','2D+PIXEL']),
        gravity_asset_id=c.TEXT,availability=dict(type='string',enum=['estimated','unavailable'])))


def validate_arguments(v):return c.validate(DEFINITION['input_schema'],v,'tool.arguments')
def validate_output(v):return c.validate(DEFINITION['output_schema'],v,'tool.data')


@agentic_requests
def execute(arguments,context):
    result=dict(protocol_version=1,call_id=context.call_id,tool_name=DEFINITION['name'],status='error',data={},artifacts=[],error=None)
    try:validate_arguments(arguments)
    except ValueError as exc:
        result['error']=dict(code='invalid_arguments',message=str(exc));return result
    try:
        route=route_result(context,arguments['routing_call_id'])
        if route.startswith('2D'):
            from atw.world_modeling.gravity_2d import estimate
            gravity=estimate(context);geometry=None;availability='estimated'
        else:
            if not all(k in arguments for k in ('segmentation_asset_id','physical_types_asset_id')):
                raise ValueError('3D gravity requires segmentation_asset_id and physical_types_asset_id')
            from atw.world_modeling.scene_geometry import estimate_3d_gravity
            gravity,geometry=estimate_3d_gravity(arguments,context)
            availability='estimated' if gravity['status']=='completed' else 'unavailable'
        path=context.output_dir/'gravity.json'
        write(path,dict(protocol_version=1,route=route,video_path=context.inputs['video_path'],
            sources=arguments,gravity=gravity,geometry=geometry))
        art=artifact(context,'gravity',path)
        result.update(status='ok',data=dict(route=route,gravity_asset_id=art['artifact_id'],availability=availability),artifacts=[art])
    except Exception as exc:result['error']=dict(code='execution_failed',message=f'{type(exc).__name__}: {exc}')
    return result
