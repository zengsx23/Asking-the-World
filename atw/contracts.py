"""Canonical JSON boundaries for the independent agent flow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

PROTOCOL_VERSION = 1
BENCHMARKS = ("clevrer", "contphy", "other")
ANSWER_FORMATS = (
    "single_choice", "multi_select", "integer", "yes_no",
    "yes_no_unknown", "open_ended",
)
StageName = Literal["stage_one", "stage_two"]
StageStatus = Literal["completed", "not_implemented", "blocked", "failed"]


class EngineEvidence(TypedDict):
    """Engine tools supply a compact measurement plus its essential semantics.

    Each tool defines the exact evidence schema. A null world_id means an
    observation/property query, never an implicitly selected simulated world.
    Optional next_step guides collection only; note retains evidence semantics.
    """
    world_id: str | None
    evidence: dict[str, Any]
    note: str
    next_step: NotRequired[str]


class Option(TypedDict):
    id: str
    text: str


class SceneQuestion(TypedDict):
    id: str
    text: str
    options: list[Option]


class Question(SceneQuestion, total=False):
    question_type: str
    answer_format: str
    answer_instructions: str


class RunRequest(TypedDict):
    protocol_version: int
    video_path: str
    questions: list[Question]
    benchmark: str


class Artifact(TypedDict):
    artifact_id: str
    path: str
    media_type: str
    source_kind: Literal["observation", "reconstruction", "simulation", "derived"]


class Error(TypedDict):
    code: str
    message: str


class StageResult(TypedDict):
    protocol_version: int
    stage: StageName
    status: StageStatus
    data: dict[str, Any]
    artifacts: list[Artifact]
    error: Error | None


class StageOneInput(TypedDict):
    video_path: str
    questions: list[SceneQuestion]


class StageOneScene(TypedDict):
    """Effective retained scene references for the stage-two handoff."""
    route: str | None
    sources: dict[str, str]
    engine: str | None
    swr_availability: str
    ready_for_stage_two: bool
    direct_video_fallback: bool


def validate_stage_one_scene(scene: dict) -> dict:
    _object(scene, {'route','sources','engine','swr_availability','ready_for_stage_two','direct_video_fallback'}, set(), 'scene')
    if scene['route'] not in (None,'3D+SAM3','2D+SAM3','2D+PIXEL'):
        raise ValueError('Unknown scene route')
    if scene['engine'] not in (None,'rigid','cloth','softbody','rope','fluid'):
        raise ValueError('Unknown scene engine')
    if scene['swr_availability'] not in ('pending','fitted','unavailable'):
        raise ValueError('Unknown SWR availability')
    if type(scene['ready_for_stage_two']) is not bool or type(scene['direct_video_fallback']) is not bool:
        raise ValueError('Scene readiness and fallback must be booleans')
    sources=scene['sources']
    allowed={'routing_call_id','segmentation_asset_id','physical_types_asset_id','gravity_asset_id','tracking_asset_id','swr_asset_id'}
    if not isinstance(sources,dict) or not sources.keys()<=allowed:
        raise ValueError('Unknown scene source fields')
    for value in sources.values():_text(value,'scene source')
    if scene['ready_for_stage_two']:
        required={'routing_call_id','tracking_asset_id','swr_asset_id'}
        if scene['route'] in ('3D+SAM3','2D+SAM3'):
            required|={'segmentation_asset_id','physical_types_asset_id','gravity_asset_id'}
        if scene['route'] is None or scene['engine'] is None or scene['swr_availability']!='fitted' or scene['direct_video_fallback'] or not required<=sources.keys():
            raise ValueError('Ready scenes require a fitted engine and complete source references')
    return json_copy(scene)


class StageTwoInput(TypedDict):
    video_path: str
    stage_one_result: StageResult
    benchmark: str
    questions: list[Question]


class ToolDefinition(TypedDict):
    protocol_version: int
    name: str
    description: str
    stages: list[StageName]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class ToolCall(TypedDict):
    protocol_version: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolResult(TypedDict):
    protocol_version: int
    call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    data: dict[str, Any]
    artifacts: list[Artifact]
    error: Error | None


class AnswerRecord(TypedDict):
    question_id: str
    status: Literal["answered", "unresolved"]
    answer: list[str] | str | int | bool | None
    evidence_ids: list[str]
    reason: str


class QuestionClassification(TypedDict):
    reasoning_mode: Literal['property', 'count', 'explanatory', 'predictive', 'counterfactual', 'goal']
    rationale: str


class IdentityDecision(TypedDict):
    reference: str
    status: Literal['found', 'not_found_in_scene']
    object_ids: list[str]
    evidence: str


class AppearanceRecord(TypedDict):
    object_id: str
    category: str
    color: str
    shape: str
    apparent_material: str
    visual_description: str
    visual_evidence: str
    uncertainty: str
    wheel_structure: Literal['solid_disk', 'hollow_spoked', 'not_wheel', 'unknown']


class QuestionPreparation(TypedDict):
    question_id: str
    reasoning_mode: str | None
    classification_artifact_id: str | None
    binding_artifact_id: str | None
    count_artifact_id: str | None
    error: Error | None


def validate_answer_record(value: dict, question: Question, evidence_ids: set[str]) -> AnswerRecord:
    _object(value, {'question_id', 'status', 'answer', 'evidence_ids', 'reason'}, set(), 'answer')
    if value['question_id'] != question['id'] or value['status'] not in ('answered', 'unresolved'):
        raise ValueError('Invalid answer question/status')
    _text(value['reason'], 'answer.reason')
    ids = value['evidence_ids']
    if not isinstance(ids, list) or any(not isinstance(i, str) or i not in evidence_ids for i in ids):
        raise ValueError('Answer evidence must reference retained artifacts or input_video')
    answer = value['answer']
    if value['status'] == 'unresolved':
        if answer is not None:
            raise ValueError('Unresolved answers require null')
    elif question['options']:
        allowed = {o['id'] for o in question['options']}
        if not isinstance(answer, list) or any(not isinstance(i, str) for i in answer):
            raise ValueError('Choice answers require an array of option IDs')
        if len(set(answer)) != len(answer) or not set(answer) <= allowed:
            raise ValueError('Choice answers require unique supplied option IDs')
        if question.get('answer_format') == 'single_choice' and len(answer) != 1:
            raise ValueError('single_choice requires one option ID')
    else:
        fmt = question.get('answer_format')
        if answer is None or isinstance(answer, (dict, list)):
            raise ValueError('Open questions require scalar answers')
        if fmt == 'integer' and type(answer) is not int:
            raise ValueError('Integer answer required')
        if fmt in ('yes_no', 'yes_no_unknown') and answer not in (
                ('yes', 'no') if fmt == 'yes_no' else ('yes', 'no', 'can not answer')):
            raise ValueError('Invalid yes/no answer')
    return json_copy(value)


def json_copy(value: Any) -> Any:
    """Detach JSON records and reject nonfinite numbers/non-JSON objects."""
    return json.loads(json.dumps(value, allow_nan=False))


def _object(value: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"{label} has missing or unknown fields")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def parse_request(raw: Any, *, base_dir: Path) -> RunRequest:
    """Resolve the video relative to the request file; preserve question text/IDs."""
    _object(raw, {"protocol_version", "video_path", "questions"}, {"benchmark"}, "request")
    if type(raw["protocol_version"]) is not int or raw["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("protocol_version must be integer 1")
    benchmark = raw.get("benchmark", "other")
    if benchmark not in BENCHMARKS:
        raise ValueError("benchmark must be clevrer, contphy, or other")
    _text(raw["video_path"], "video_path")
    video = Path(raw["video_path"]).expanduser()
    if not video.is_absolute():
        video = base_dir / video
    video = video.resolve()
    if not video.is_file():
        raise ValueError(f"Video file does not exist: {video}")
    questions = raw["questions"]
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty array")
    seen_questions: set[str] = set()
    for question in questions:
        _object(question, {"id", "text", "options"},
                {"answer_format", "answer_instructions", "question_type"}, "question")
        _text(question["id"], "question.id")
        _text(question["text"], "question.text")
        if "question_type" in question:
            _text(question["question_type"], "question.question_type")
        if question["id"] in seen_questions:
            raise ValueError("Question IDs must be unique")
        seen_questions.add(question["id"])
        if "answer_format" in question and question["answer_format"] not in ANSWER_FORMATS:
            raise ValueError("Unsupported answer_format")
        if "answer_instructions" in question:
            _text(question["answer_instructions"], "answer_instructions")
        if not isinstance(question["options"], list):
            raise ValueError("options must be an array (empty for open-ended questions)")
        seen_options: set[str] = set()
        for option in question["options"]:
            _object(option, {"id", "text"}, set(), "option")
            _text(option["id"], "option.id")
            _text(option["text"], "option.text")
            if option["id"] in seen_options:
                raise ValueError("Option IDs must be unique within a question")
            seen_options.add(option["id"])
    return dict(protocol_version=PROTOCOL_VERSION, video_path=str(video),
                questions=json_copy(questions), benchmark=benchmark)


def stage_result(stage: StageName, status: StageStatus = "not_implemented",
                 *, error: Error | None = None, data: dict | None = None,
                 artifacts: list | None = None) -> StageResult:
    return dict(protocol_version=PROTOCOL_VERSION, stage=stage, status=status,
                data=json_copy(data) if data is not None else {},
                artifacts=json_copy(artifacts) if artifacts is not None else [], error=error)


def validate_stage_result(result: Any, stage: StageName) -> StageResult:
    _object(result, {"protocol_version", "stage", "status", "data", "artifacts", "error"},
            set(), "stage result")
    if type(result["protocol_version"]) is not int or result["protocol_version"] != 1:
        raise ValueError("Invalid stage protocol_version")
    if result["stage"] != stage:
        raise ValueError("Stage result belongs to the wrong stage")
    if result["status"] not in ("completed", "not_implemented", "blocked", "failed"):
        raise ValueError("Invalid stage status")
    if not isinstance(result["data"], dict) or not isinstance(result["artifacts"], list):
        raise ValueError("Stage data/artifacts must be an object/array")
    if result["status"] in ("blocked", "failed"):
        _object(result["error"], {"code", "message"}, set(), "stage error")
        _text(result["error"]["code"], "error.code")
        _text(result["error"]["message"], "error.message")
    elif result["error"] is not None:
        raise ValueError("Completed/unimplemented stages must have null error")
    return json_copy(result)


def tool_call_directory_name(call_id: str, tool_name: str) -> str:
    """Keep call identity separate from the human-readable invocation directory."""
    if not isinstance(tool_name, str) or not tool_name or tool_name[0] not in 'abcdefghijklmnopqrstuvwxyz' or any(
        char not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for char in tool_name
    ):
        raise ValueError('Tool name must be lower-case snake_case')
    return f'{call_id}_{tool_name}'
