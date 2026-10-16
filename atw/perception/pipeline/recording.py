"""Stage timing and atomic JSON writes."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
import json

CURRENT = ContextVar("workflow_recorder", default=None)


def track4world_load_timings(diagnostics):
    """Report this acquisition, without charging a cached model's original load again."""
    reused = diagnostics.get("reuse_count", 0) > 0
    result = {
        f"track4world_load_{name}": float(
            0 if reused and name.endswith("_seconds") else value
        )
        for name, value in diagnostics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    result["track4world_load_cold_load_count"] = float(not reused)
    result["track4world_load_model_reused"] = float(reused)
    result["track4world_load_weight_prefetch_seconds"] = (
        0.0 if reused else float(diagnostics["weight_prefetch"]["elapsed_seconds"])
    )
    return result


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    temporary.replace(path)


@contextmanager
def timed_stage(timings, name):
    started = perf_counter()
    print(f"[timing] start {name}", flush=True)
    recorder = CURRENT.get()
    try:
        if recorder is None:
            yield
        else:
            with recorder.stage(name):
                yield
    finally:
        timings[name] = perf_counter() - started
        print(f"[timing] finish {name}: {timings[name]:.3f}s", flush=True)
