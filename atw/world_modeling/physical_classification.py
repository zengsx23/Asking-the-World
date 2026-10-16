"""Question-aware physical-type Engineer with lazy inference dependencies."""
from atw.world_modeling import sam3_protocol as contract
from atw.world_modeling.request_scope import agentic_requests

PHYSICAL_TYPES = ('rigid', 'soft_body', 'cloth', 'rope')
CLASSIFIED_OBJECT = contract.obj(
    object_id=contract.ID, track_id=contract.TEXT,
    physical_type=dict(type='string', enum=list(PHYSICAL_TYPES)), reason=contract.TEXT)
DEFINITION = dict(
    protocol_version=1, name='classify_object_physical_types',
    description='Classify segmented objects as rigid, soft body, cloth or rope; return a reason for each.',
    stages=['stage_one'],
    input_schema=contract.obj(segmentation_asset_id=contract.TEXT),
    output_schema=contract.obj(objects=contract.array(CLASSIFIED_OBJECT)))


def selection_schema(ids):
    item = contract.obj(question_id={**contract.TEXT, 'enum': list(ids)}, reason=contract.TEXT)
    # An empty input has no possible selection and does not require inference.
    return contract.obj(selected=contract.array(item)) if ids else contract.obj(selected=dict(type='array', maxItems=0))


def classification_schema(ids):
    return contract.obj(objects=contract.array(contract.obj(
        track_id={**contract.TEXT, 'enum': list(ids)}, reason=contract.TEXT,
        physical_type=dict(type='string', enum=list(PHYSICAL_TYPES))),
        minItems=len(ids), maxItems=len(ids)))


def validate_arguments(value):
    return contract.validate(DEFINITION['input_schema'], value, 'tool.arguments')


def validate_output(value):
    contract.validate(DEFINITION['output_schema'], value, 'tool.data')
    contract.unique_ids(value['objects'], 'object_id', 'tool.data')
    contract.unique_ids(value['objects'], 'track_id', 'tool.data')
    return value


@agentic_requests
def execute(arguments, context):
    result = dict(protocol_version=1, call_id=context.call_id, tool_name=DEFINITION['name'],
                  status='error', data={}, artifacts=[], error=None)
    try:
        validate_arguments(arguments)
    except ValueError as exc:
        result['error'] = dict(code='invalid_arguments', message=str(exc))
        return result
    try:
        from atw.world_modeling.physical_classification_impl import classify
        data, artifacts = classify(arguments, context)
        result.update(status='ok', data=validate_output(data), artifacts=artifacts)
    except Exception as exc:
        result['error'] = dict(code='execution_failed', message=f'{type(exc).__name__}: {exc}')
    return result
