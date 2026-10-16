"""VLM classification of anonymous SAM3 tracks by physical object type."""


from __future__ import annotations


import json

from pathlib import Path

from atw.shared.vlm_planner import _read_env_value

from atw.shared.vlm_usage import OpenRouterJSONClient, Transport

EXPECTED_MODEL = "google/gemini-3-flash-preview"




class ObjectMaterialClassifier:
    """Classify SAM3 mask tracks independently or jointly with one VLM."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        api_key: str | None = None,
        transport: Transport | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        config = json.loads(
            (
                self.project_root
                / "configs/llm/openrouter_google_gemini_3_flash_preview.json"
            ).read_text(encoding="utf-8")
        )
        self.model = str(config["model"])
        if self.model != EXPECTED_MODEL:
            raise ValueError(
                f"material classifier requires {EXPECTED_MODEL}, found {self.model}"
            )
        resolved_key = api_key or _read_env_value(
            self.project_root / ".env", "OPEN_ROUTER_KEY"
        )
        self.client = OpenRouterJSONClient(
            api_key=resolved_key,
            model=self.model,
            max_output_tokens=int(config["max_output_tokens"]),
            module_name="object_material_classifier",
            transport=transport,
            max_attempts=max_attempts,
        )
