"""Model execution defaults owned by the reconstruction Engineers."""
from dataclasses import dataclass
from pathlib import Path
import sys
from atw.perception.pipeline.config import ModelConfig


@dataclass(kw_only=True)
class SceneConfig(ModelConfig):
    image_size: int = 640
    sam3d_environment: Path = Path(sys.prefix).with_name('physmind-v2-sam3d')
    rope_gravity_normalized_per_s2: float = 0.29
