"""CPU-only debug stage, also usable after a normal workflow run."""

from pathlib import Path
import subprocess
import sys

from atw.perception.pipeline.recording import timed_stage, write_json

ROOT = Path(__file__).resolve().parents[3]


def render_tracking(tracking_dir: Path, *, output_dir: Path, workers: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    timings = {}
    with timed_stage(timings, "render_debug_videos"):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_cloth_track4world_debug.py"),
                "--run-dir",
                str(tracking_dir),
                "--debug-dir",
                str(output_dir),
                "--workers",
                str(workers),
            ],
            check=True,
        )
    write_json(output_dir / "stage_timing.json", timings)
    return timings
