"""Stage-one observed pixel tracking selected by an explicit routing result."""
from atw.contracts import ToolDefinition, ToolResult
from atw.world_modeling.sam3_protocol import TEXT, obj, validate
from atw.world_modeling.scene_assets import route_result
from atw.world_modeling.request_scope import agentic_requests

DEFINITION: ToolDefinition = dict(
    protocol_version=1, name='track_pixel_scene', stages=['stage_one'],
    description=('Track the outer enclosure, static objects and liquid masks for the 2D+PIXEL route; '
                 'return observed pixel tracks and liquid source bindings. '
                 'Call after world_representation_routing.'),
    input_schema=obj(routing_call_id={**TEXT, 'pattern': r'^call_[0-9]{4,}$',
        'description': 'Successful same-run world_representation_routing call selecting 2D+PIXEL.'}),
    output_schema=obj(route=dict(type='string', const='2D+PIXEL'),
        tracking_asset_id={**TEXT, 'description': 'Returned observed pixel tracking manifest artifact ID.'},
        object_count=dict(type='integer', minimum=1,
            description='Static objects, liquid color groups and the outer enclosure in the original clip.')))


def validate_arguments(value):
    return validate(DEFINITION['input_schema'], value, 'arguments')


def validate_output(value):
    return validate(DEFINITION['output_schema'], value, 'data')


@agentic_requests
def execute(arguments, context) -> ToolResult:
    result = dict(protocol_version=1, call_id=context.call_id, tool_name=DEFINITION['name'],
                  status='error', data={}, artifacts=[], error=None)
    try:
        validate_arguments(arguments)
    except ValueError as exc:
        result['error'] = dict(code='invalid_arguments', message=str(exc))
        return result
    try:
        route = route_result(context, arguments['routing_call_id'])
    except (ValueError, KeyError, OSError) as exc:
        result['error'] = dict(code='invalid_prerequisite', message=str(exc))
        return result
    if route != '2D+PIXEL':
        result['error'] = dict(code='unsupported_operation', message='track_pixel_scene requires the 2D+PIXEL route')
        return result
    try:
        from atw.world_modeling.pixel_tracking_impl import run
        data, artifacts = run(arguments, context)
        result.update(status='ok', data=validate_output(data), artifacts=artifacts)
    except NotImplementedError as exc:
        result['error'] = dict(code='unsupported_operation', message=str(exc))
    except Exception as exc:
        result['error'] = dict(code='execution_failed', message=str(exc) or type(exc).__name__)
    return result
