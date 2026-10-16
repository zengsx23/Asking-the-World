"""Geometry model and tracking configuration."""

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data/models/track4world/track4world_da3.pth"
DEFAULT_DA3_SNAPSHOT = (
    PROJECT_ROOT / "data/models/track4world/depth-anything/DA3NESTED-GIANT-LARGE-1.1"
)


@dataclass(kw_only=True)
class ModelConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    da3_snapshot: Path = DEFAULT_DA3_SNAPSHOT
    device: str = "cuda"
    inference_iterations: int = 4
    debug: bool = False


@dataclass(kw_only=True)
class TrackingConfig(ModelConfig):
    reconstruction_dir: Path
    output_dir: Path
    minimum_tracking_confidence: float = 0.2
    adjacent_tracking_cache: Path | None = None
    soft_body_motion_cache: Path | None = None
    rigid_workers: int = max(1, min(4, (os.cpu_count() or 1) // 3))
    debug_workers: int = min(os.cpu_count() or 1, 24)
