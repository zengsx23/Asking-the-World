"""Stage-scoped tools callable by Reasoning; no automatic tool execution."""
from dataclasses import dataclass, field
import json
from pathlib import Path

from atw.config import VLMConfig, Sam3Config, SceneConfig, ReasoningConfig
from atw.contracts import StageName, json_copy, tool_call_directory_name, validate_stage_result
from atw.world_modeling import route_selection, sam3_segmentation, physical_classification
from atw.world_modeling import scene_gravity, reconstruction_tracking, pixel_tracking, scene_swr
from atw.world_probing.tool_definitions import TOOLS as QUESTION_TOOLS

REGISTERED_TOOLS = {tool.DEFINITION["name"]: tool for tool in
                    (route_selection, sam3_segmentation, physical_classification,
                     scene_gravity, reconstruction_tracking, pixel_tracking, scene_swr)}
REGISTERED_TOOLS.update({tool.DEFINITION['name']: tool for tool in QUESTION_TOOLS})


@dataclass(frozen=True)
class ToolContext:
    call_id: str
    stage: StageName
    run_dir: Path
    output_dir: Path
    inputs: dict
    artifacts: tuple
    vlm: VLMConfig
    sam3: Sam3Config = Sam3Config()
    scene: SceneConfig = field(default_factory=SceneConfig)
    stage_one_result: dict | None = None
    question_id: str | None = None
    question: dict | None = None
    tools: object | None = None


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class ToolRunner:
    """One instance per run assigns call IDs and preserves invocation records."""
    def __init__(self, *, run_dir, vlm, inputs, sam3=None, scene=None,
                 artifacts=(), stage_one_result=None, call_index_start=0, reasoning=None):
        self.run_dir = Path(run_dir)
        self.vlm = vlm
        self.sam3 = sam3 or Sam3Config()
        self.scene = scene or SceneConfig()
        self.reasoning = reasoning or ReasoningConfig()
        self.inputs = json_copy(inputs)
        self._call_index = call_index_start
        self._artifacts = json_copy(list(artifacts))
        self.stage_one_result = json_copy(stage_one_result)
        self._stage_one_failure = None
        self._question_tools = {}
        self._result_ids = set()

    @property
    def evidence_ids(self):
        return self._result_ids | {a['artifact_id'] for a in self._artifacts}

    def register_question_tools(self, question_id, tools):
        """Register engine-owned operations only for their explicit question."""
        definitions = {tool.DEFINITION['name']: tool for tool in tools}
        if len(definitions) != len(tools) or set(definitions) & set(REGISTERED_TOOLS):
            raise ValueError('Question tool names must be unique and distinct from shared tools')
        self._question_tools[question_id] = definitions

    @property
    def stage_one_failure(self):
        """Explicit failed-scene handoff from exhausted SWR admission review."""
        return json_copy(self._stage_one_failure)

    def definitions(self, stage, question_id=None):
        registered = dict(REGISTERED_TOOLS)
        if stage == 'stage_two':
            registered.update(self._question_tools.get(question_id, {}))
        return [json_copy(tool.DEFINITION) for tool in registered.values()
                if stage in tool.DEFINITION["stages"]]

    def function_definitions(self, stage):
        """Provider-facing function descriptions; runtime context stays private."""
        return [{"type": "function", "function": {
            "name": definition["name"], "description": definition["description"],
            "parameters": definition["input_schema"],
        }} for definition in self.definitions(stage)]

    def call(self, tool_name, arguments, *, stage, question=None):
        if stage not in ("stage_one", "stage_two"):
            raise ValueError("Unknown stage")
        self._call_index += 1
        call_id = f"call_{self._call_index:04d}"
        self._result_ids.add(call_id)
        output = self.run_dir / stage / "tools" / tool_call_directory_name(call_id, tool_name)
        output.mkdir(parents=True)
        call = dict(protocol_version=1, call_id=call_id, tool_name=tool_name,
                    arguments=json_copy(arguments))
        _write(output / "call.json", call)
        result = dict(protocol_version=1, call_id=call_id, tool_name=tool_name,
                      status="error", data={}, artifacts=[], error=None)
        tool = REGISTERED_TOOLS.get(tool_name)
        if tool is None and stage == 'stage_two' and question is not None:
            tool = self._question_tools.get(question['id'], {}).get(tool_name)
        if tool is None or stage not in tool.DEFINITION["stages"]:
            result["error"] = dict(code="unsupported_operation", message="Tool unavailable in this stage")
        else:
            try:
                tool.validate_arguments(arguments)
            except ValueError as exc:
                result["error"] = dict(code="invalid_arguments", message=str(exc))
            else:
                inputs = json_copy(self.inputs)
                _write(output / "input.json", inputs)
                context = ToolContext(call_id, stage, self.run_dir, output, inputs,
                                      tuple(json_copy(self._artifacts)), self.vlm, self.sam3, self.scene,
                                      json_copy(self.stage_one_result) if stage == 'stage_two' else None,
                                      question['id'] if question is not None else None,
                                      json_copy(question), self)
                result = tool.execute(arguments, context)
                if result["status"] == "ok":
                    tool.validate_output(result["data"])
                    self._artifacts.extend(json_copy(result["artifacts"]))
                    if tool_name == 'fit_scene_swr':
                        if result['data']['availability']=='fitted':
                            self._stage_one_failure=None
                        else:
                            failure=next((a for a in result['artifacts']
                                if a['artifact_id']==call_id+'_stage_one_failure'),None)
                            if failure is not None:
                                from atw.world_modeling.scene_assets import asset_path
                                retained_context=ToolContext(call_id,stage,self.run_dir,output,inputs,
                                    tuple(self._artifacts),self.vlm,self.sam3,self.scene)
                                path=asset_path(retained_context,failure['artifact_id'])
                                self._stage_one_failure=validate_stage_result(json.loads(path.read_text()),'stage_one')
        _write(output / "result.json", result)
        return json_copy(result)
