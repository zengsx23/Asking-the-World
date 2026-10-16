"""Task-scoped helpers for predictable model loading from NFS storage."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class WeightPrefetchReport:
    paths: tuple[str, ...]
    bytes_read: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TorchLoadCache:
    """Deserialize selected torch checkpoints once across repeated consumers."""

    def __init__(
        self,
        *,
        tracked_paths: Iterable[Path],
        reusable_paths: Iterable[Path],
    ) -> None:
        self._tracked_paths = {
            str(Path(path).expanduser().resolve()) for path in tracked_paths
        }
        self._reusable_paths = {
            str(Path(path).expanduser().resolve()) for path in reusable_paths
        }
        if not self._reusable_paths <= self._tracked_paths:
            raise ValueError("reusable checkpoints must also be tracked")
        self._cache: dict[str, Any] = {}
        self._original_load: Any = None
        self.deserialize_seconds_by_path: dict[str, float] = {}
        self.loaded_paths: set[str] = set()
        self.cache_hit_count = 0

    def __enter__(self) -> "TorchLoadCache":
        import torch

        if self._original_load is not None:
            raise RuntimeError("TorchLoadCache cannot be entered twice")
        self._original_load = torch.load
        torch.load = self._load
        return self

    def __exit__(self, *_: object) -> None:
        import torch

        torch.load = self._original_load
        self._original_load = None
        self._cache.clear()

    def _load(self, path: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, (str, os.PathLike)):
            resolved = str(Path(path).expanduser().resolve())
            if resolved in self._tracked_paths:
                if resolved in self._cache:
                    self.cache_hit_count += 1
                    return self._cache.pop(resolved)
                started = time.perf_counter()
                value = self._original_load(path, *args, **kwargs)
                self.deserialize_seconds_by_path[resolved] = (
                    time.perf_counter() - started
                )
                self.loaded_paths.add(resolved)
                if resolved in self._reusable_paths:
                    self._cache[resolved] = value
                return value
        return self._original_load(path, *args, **kwargs)

    def diagnostics(self) -> dict[str, object]:
        return {
            "checkpoint_deserialize_seconds_by_path": dict(
                self.deserialize_seconds_by_path
            ),
            "checkpoint_deserialize_seconds": sum(
                self.deserialize_seconds_by_path.values()
            ),
            "unique_checkpoint_load_count": len(self.loaded_paths),
            "checkpoint_cache_hit_count": self.cache_hit_count,
        }


def prefetch_weight_files(
    paths: Iterable[Path], *, chunk_bytes: int = 32 * 1024 * 1024
) -> WeightPrefetchReport:
    """Read weights sequentially once so later tensor paging is served from RAM."""

    resolved = tuple(Path(path).expanduser().resolve() for path in paths)
    if chunk_bytes <= 0:
        raise ValueError("prefetch chunk size must be positive")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    total = 0
    buffer = bytearray(chunk_bytes)
    view = memoryview(buffer)
    for path in resolved:
        with path.open("rb", buffering=0) as stream:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    stream.fileno(),
                    0,
                    0,
                    os.POSIX_FADV_SEQUENTIAL,
                )
            while True:
                count = stream.readinto(view)
                if not count:
                    break
                total += count
    return WeightPrefetchReport(
        paths=tuple(str(path) for path in resolved),
        bytes_read=total,
        elapsed_seconds=time.perf_counter() - started,
    )
