"""Route selection from the original video and task text only."""
import base64
import json
import mimetypes
from pathlib import Path

from atw.contracts import ToolDefinition, ToolResult
from atw.openrouter import OpenRouterClient
from atw.world_modeling.request_scope import agentic_requests

ROUTES = ("3D+SAM3", "2D+SAM3", "2D+PIXEL")
DEFINITION: ToolDefinition = {
    "protocol_version": 1,
    "name": "world_representation_routing",
    "description": "Select a reconstruction route from the original video and questions.",
    "stages": ["stage_one"],
    "input_schema": {"type": "object", "properties": {}, "required": [],
                     "additionalProperties": False},
    "output_schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": list(ROUTES)},
            "reason": {"type": "string", "minLength": 1, "pattern": r"\S"},
        },
        "required": ["route", "reason"], "additionalProperties": False,
    },
}


def validate_arguments(arguments):
    if not isinstance(arguments, dict) or arguments:
        raise ValueError("world_representation_routing requires an empty arguments object")


def validate_output(value):
    if not isinstance(value, dict) or set(value) != {"route", "reason"}:
        raise ValueError("Expected only route and reason")
    if value["route"] not in ROUTES:
        raise ValueError("Unsupported route")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    return value


@agentic_requests
def execute(arguments, context) -> ToolResult:
    """Use a fresh request; share no Reasoning or other Engineer history."""
    result = dict(protocol_version=1, call_id=context.call_id, tool_name="world_representation_routing",
                  status="error", data={}, artifacts=[], error=None)
    try:
        validate_arguments(arguments)
    except ValueError as exc:
        result["error"] = dict(code="invalid_arguments", message=str(exc))
        return result
    try:
        video = Path(context.inputs["video_path"])
        mime = mimetypes.guess_type(video.name)[0]
        if mime is None or not mime.startswith("video/"):
            raise ValueError("Video must have a recognized video file extension")
        media = "data:" + mime + ";base64," + base64.b64encode(video.read_bytes()).decode("ascii")
        prompt = (Path(__file__).resolve().parents[1] / "prompts/route_selection_system.txt").read_text()
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": media}},
                {"type": "text", "text": json.dumps(
                    {"questions": context.inputs["questions"]}, ensure_ascii=False)},
            ]},
        ]
        data = OpenRouterClient(context.vlm).complete(
            messages, context.output_dir / "request", validate=validate_output)
        result.update(status="ok", data=validate_output(data))
    except Exception as exc:
        result["error"] = dict(code="execution_failed", message=str(exc) or type(exc).__name__)
    return result
