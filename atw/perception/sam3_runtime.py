"""Process-local persistent SAM3 model runtime."""

from __future__ import annotations

import atexit
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


PredictorFactory = Callable[[Path, int], Any]
ProcessorFactory = Callable[[Any, int], Any]


def _build_predictor(checkpoint_path: Path, device_index: int) -> Any:
    from sam3.model_builder import build_sam3_video_predictor

    return build_sam3_video_predictor(
        checkpoint_path=str(checkpoint_path),
        gpus_to_use=[device_index],
    )


def _build_processor(detector: Any, device_index: int) -> Any:
    from sam3.model.sam3_image_processor import Sam3Processor

    return Sam3Processor(
        detector,
        device=f"cuda:{device_index}",
        confidence_threshold=0.35,
    )


@dataclass
class Sam3Runtime:
    """One loaded SAM3 predictor shared by multiple scene requests."""

    checkpoint_path: Path
    device_index: int
    predictor: Any
    detector: Any
    tracker: Any
    image_processor: Any
    cold_load_seconds: float
    inference_lock: threading.RLock = field(default_factory=threading.RLock)
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        shutdown = getattr(self.predictor, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._closed = True


class Sam3RuntimeRegistry:
    """Own persistent SAM3 runtimes for the lifetime of one Python process."""

    def __init__(
        self,
        *,
        predictor_factory: PredictorFactory = _build_predictor,
        processor_factory: ProcessorFactory = _build_processor,
    ) -> None:
        self._predictor_factory = predictor_factory
        self._processor_factory = processor_factory
        self._runtimes: dict[tuple[Path, int], Sam3Runtime] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        checkpoint_path: str | Path,
        *,
        device_index: int | None = None,
    ) -> tuple[Sam3Runtime, bool]:
        """Return a cached runtime and whether this acquisition was a cache hit."""

        checkpoint = Path(checkpoint_path).resolve()
        if device_index is None:
            if not torch.cuda.is_available():
                raise RuntimeError("SAM3 persistent runtime requires a CUDA device")
            device_index = torch.cuda.current_device()
        key = (checkpoint, device_index)
        with self._lock:
            cached = self._runtimes.get(key)
            if cached is not None:
                return cached, True

            started = time.perf_counter()
            predictor = self._predictor_factory(checkpoint, device_index)
            try:
                model = getattr(predictor, "model", None)
                detector = getattr(model, "detector", None)
                tracker = getattr(model, "tracker", None)
                if detector is None or tracker is None:
                    raise TypeError(
                        "SAM3 video predictor must expose model.detector and model.tracker"
                    )
                if getattr(tracker, "backbone", object()) is None:
                    tracker.backbone = detector.backbone
                image_processor = self._processor_factory(detector, device_index)
            except Exception:
                shutdown = getattr(predictor, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                raise

            runtime = Sam3Runtime(
                checkpoint_path=checkpoint,
                device_index=device_index,
                predictor=predictor,
                detector=detector,
                tracker=tracker,
                image_processor=image_processor,
                cold_load_seconds=time.perf_counter() - started,
            )
            self._runtimes[key] = runtime
            return runtime, False

    def close_all(self) -> None:
        """Shutdown every cached predictor and empty the registry."""

        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
        for runtime in runtimes:
            with runtime.inference_lock:
                runtime.close()


_TASK_RUNTIME_REGISTRY = Sam3RuntimeRegistry()
atexit.register(_TASK_RUNTIME_REGISTRY.close_all)


def task_sam3_runtime_registry() -> Sam3RuntimeRegistry:
    return _TASK_RUNTIME_REGISTRY


def close_task_sam3_models() -> None:
    """Explicitly release task-scoped SAM3 models before process exit."""

    _TASK_RUNTIME_REGISTRY.close_all()
