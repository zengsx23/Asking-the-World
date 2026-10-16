"""SAM3 Engineer tool definition with lazy GPU/model imports."""
from atw.world_modeling import sam3_protocol as contract
from atw.world_modeling.request_scope import agentic_requests

DEFINITION = dict(
    protocol_version=1, name="sam3_video_segmentation",
    description="Segment and track individual objects in the video; return masks and the object inventory.",
    stages=["stage_one"], input_schema=contract.TOOL_INPUT,
    output_schema=contract.TOOL_OUTPUT,
)


def validate_arguments(value):
    return contract.validate(DEFINITION["input_schema"], value, "tool.arguments")


def validate_output(value):
    return contract.validate(DEFINITION["output_schema"], value, "tool.data")


@agentic_requests
def execute(arguments, context):
    try:
        validate_arguments(arguments)
    except ValueError as exc:
        return dict(protocol_version=1, call_id=context.call_id, tool_name=DEFINITION["name"],
                    status="error", data={}, artifacts=[],
                    error=dict(code="invalid_arguments", message=str(exc)))
    try:
        from atw.perception.pipeline.routing import resolve_sam3_checkpoint
        from atw.world_modeling.sam3_segmentation_impl import Sam3Engineer
        config = context.sam3
        engineer = Sam3Engineer(
            checkpoint=resolve_sam3_checkpoint(config.checkpoint),
            debug=config.debug, seed_inventory=config.seed_inventory)
        return engineer.execute(arguments, context)
    except Exception as exc:
        return dict(protocol_version=1, call_id=context.call_id, tool_name=DEFINITION["name"],
                    status="error", data={}, artifacts=[],
                    error=dict(code="execution_failed", message=f"{type(exc).__name__}: {exc}"))
