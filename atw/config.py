"""Runner configuration; not semantic stage input."""
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from atw.world_modeling.scene_runtime import SceneConfig

if TYPE_CHECKING:
    from atw.tools import ToolRunner

@dataclass(frozen=True)
class VLMConfig:
    # Same model settings as configs/llm/openrouter_google_gemini_3_flash_preview.json.
    # Shared request client uses these settings without generation overrides.
    provider: str = "openrouter"
    model: str = "google/gemini-3-flash-preview"
    input_mode: str = "full_video"
    max_output_tokens: int = 4096


@dataclass(frozen=True)
class StageContext:
    run_dir: Path
    output_dir: Path
    vlm: VLMConfig
    tools: "ToolRunner | None" = None


@dataclass(frozen=True)
class Sam3Config:
    """Runtime-only SAM3 settings; never model-facing tool arguments."""
    checkpoint: Path | None = None
    debug: bool = False
    seed_inventory: Path | None = None


@dataclass(frozen=True)
class ReasoningConfig:
    """Stage-two evidence routing; independent of VLM generation settings."""
    rigid_predictive_tracking_only: bool = True
