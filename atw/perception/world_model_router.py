from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from atw.shared.vlm_planner import _read_env_value
from atw.shared.vlm_usage import OpenRouterJSONClient, Transport

class WorldModelRouter:
    def __init__(
        self, *, api_key: str, model: str, max_output_tokens: int,
        prompt_config: Mapping[str, Any], gravity_prompt: str, project_root: Path,
        transport: Transport | None = None, max_attempts: int = 1,
    ) -> None:
        self.project_root = project_root.resolve()
        self.prompt_config = prompt_config
        self.gravity_prompt = gravity_prompt
        self.gravity: dict[str, Any] | None = None
        self.trace: dict[str, Any] = {}
        self.client = OpenRouterJSONClient(
            api_key=api_key, model=model, max_output_tokens=max_output_tokens,
            module_name="world_model_router", transport=transport,
            max_attempts=max_attempts,
        )

    @classmethod
    def from_project(
        cls, project_root: str | Path, *, transport: Transport | None = None,
    ) -> WorldModelRouter:
        root = Path(project_root).resolve()
        model = json.loads((root / "configs/llm/openrouter_google_gemini_3_flash_preview.json")
                           .read_text(encoding="utf-8"))
        prompt = json.loads((root / "configs/prompts/world_model_router.json")
                            .read_text(encoding="utf-8"))
        if model["provider"] != "openrouter":
            raise ValueError("WorldModelRouter requires the OpenRouter provider")
        return cls(
            api_key=_read_env_value(root / ".env", "OPEN_ROUTER_KEY"),
            model=model["model"], max_output_tokens=model["max_output_tokens"],
            prompt_config=prompt, project_root=root, transport=transport,
            gravity_prompt=(root / "configs/prompts/plane_gravity.txt").read_text(encoding="utf-8"),
        )
