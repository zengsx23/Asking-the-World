"""Stage-one orchestration and explicit stage-two handoff with retained results."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from atw import modeling as stage_one, probing as stage_two
from atw.config import StageContext, VLMConfig, SceneConfig, Sam3Config, ReasoningConfig
from atw.contracts import json_copy, parse_request, stage_result, validate_stage_result
from atw.tools import ToolRunner


def _write(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _invoke(module: Any, name: str, inputs: dict, context: StageContext) -> dict:
    try:
        return validate_stage_result(module.run(inputs, context), name)
    except Exception as exc:
        return stage_result(name, "failed", error={
            "code": "execution_failed",
            "message": str(exc) or type(exc).__name__,
        })


def run(request_path: Path, output_dir: Path, *, stage: str = 'both',
        scene: SceneConfig | None = None, sam3: Sam3Config | None = None,
        reasoning: ReasoningConfig | None = None) -> dict:
    if stage not in ('stage_one','both'):
        raise ValueError('stage must be stage_one or both')
    request_path = Path(request_path).resolve()
    request = parse_request(json.loads(request_path.read_text(encoding="utf-8")),
                            base_dir=request_path.parent)
    output_dir = Path(output_dir).resolve()
    # A run owns a new directory. Never overwrite previous results.
    output_dir.mkdir(parents=True, exist_ok=False)
    vlm = VLMConfig()
    scene=scene or SceneConfig()
    sam3=sam3 or Sam3Config()
    reasoning=reasoning or ReasoningConfig()
    _write(output_dir / "input.json", request)
    runtime=lambda value:{k:str(v) if isinstance(v,Path) else v for k,v in asdict(value).items()}
    _write(output_dir / "runner.json", {"vlm": asdict(vlm),'stage':stage,'scene':runtime(scene),'sam3':runtime(sam3),
                                      'reasoning':asdict(reasoning)})

    first_dir = output_dir / "stage_one"
    first_dir.mkdir()
    stage_questions = [
        {key: json_copy(question[key]) for key in ("id", "text", "options")}
        for question in request["questions"]
        if not (
            request["benchmark"] == "clevrer"
            and question.get("question_type") == "descriptive"
        )
    ]
    first_input = {
        "video_path": request["video_path"],
        "questions": stage_questions,
    }
    tools = ToolRunner(
        run_dir=output_dir, vlm=vlm, inputs=first_input, scene=scene, sam3=sam3, reasoning=reasoning,
    )
    _write(first_dir / "tools.json", {"tools": tools.definitions("stage_one")})
    first = _invoke(stage_one, "stage_one", first_input,
                    StageContext(output_dir, first_dir, vlm, tools))
    if tools.stage_one_failure is not None and first['status'] in ('completed','not_implemented'):
        first=tools.stage_one_failure
    _write(first_dir / "result.json", first)
    stage_paths = {"stage_one": "stage_one/result.json"}
    overall_status, error = first["status"], first["error"]

    from atw.direct_video_answer import requested as direct_video_requested
    fallback = direct_video_requested(first)
    if stage=='both' and (first["status"] in ("completed", "not_implemented") or fallback):
        second_dir = output_dir / "stage_two"
        second_dir.mkdir()
        second_input = {
            "video_path": request["video_path"],
            "stage_one_result": json_copy(first),
            "benchmark": request["benchmark"],
            "questions": [
                json_copy(question) for question in request["questions"]
                if not (
                    request["benchmark"] == "clevrer"
                    and question.get("question_type") == "descriptive"
                )
            ],
        }
        second = _invoke(stage_two, "stage_two", second_input,
                         StageContext(output_dir, second_dir, vlm, tools))
        _write(second_dir / "result.json", second)
        stage_paths["stage_two"] = "stage_two/result.json"
        if second["status"] in ("blocked", "failed"):
            overall_status, error = second["status"], second["error"]
        elif "not_implemented" in (first["status"], second["status"]):
            overall_status, error = "not_implemented", None
        elif fallback:
            # Answers may complete, but scene construction remains explicitly failed.
            overall_status, error = first['status'], first['error']
        else:
            overall_status, error = "completed", None

    result = dict(protocol_version=1, status=overall_status,
                  stage_results=stage_paths, error=error)
    _write(output_dir / "result.json", result)
    return result
