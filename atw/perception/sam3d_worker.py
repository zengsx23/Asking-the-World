"""Task-scoped persistent SAM 3D reconstruction worker."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


_RESPONSE_PREFIX = "__PHYSMIND_SAM3D_RESPONSE__"


@dataclass(frozen=True, slots=True)
class Sam3DWorkerKey:
    environment: str
    script: str
    config: str
    cuda_visible_devices: str


class Sam3DWorker:
    """Keep SAM 3D and FoundationPose loaded for all requests in one task."""

    def __init__(
        self,
        *,
        project_root: Path,
        environment: Path,
        script: Path,
        config: Path,
    ) -> None:
        python = environment / "bin/python"
        if not python.is_file():
            raise FileNotFoundError(python)
        command = [
            str(python),
            str(script),
            "--worker",
            "--config",
            str(config),
        ]
        worker_environment = os.environ.copy()
        worker_environment["PYTHONUTF8"] = "1"
        worker_environment["CONDA_PREFIX"] = str(environment)
        worker_environment["PATH"] = (
            f"{environment / 'bin'}:{worker_environment.get('PATH', '')}"
        )
        worker_environment["LD_LIBRARY_PATH"] = ":".join(
            (
                str(environment / "lib"),
                "/usr/local/nvidia/lib",
                "/usr/local/nvidia/lib64",
            )
        )
        self._process = subprocess.Popen(
            command,
            cwd=project_root,
            env=worker_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._lock = Lock()
        self.request_count = 0

    def run(self, *, manifest: Path, output_dir: Path) -> dict[str, Any]:
        request = {
            "manifest": str(manifest.resolve()),
            "output_dir": str(output_dir.resolve()),
        }
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"SAM 3D worker exited with code {self._process.returncode}"
                )
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            for line in self._process.stdout:
                if line.startswith(_RESPONSE_PREFIX):
                    response = json.loads(line[len(_RESPONSE_PREFIX) :])
                    if not response.get("ok"):
                        raise RuntimeError(
                            "SAM 3D worker request failed: "
                            + str(response.get("error", "unknown error"))
                        )
                    self.request_count += 1
                    return response
                print(line, end="", flush=True)
        raise RuntimeError(
            "SAM 3D worker closed stdout before responding; "
            f"exit code {self._process.wait()}"
        )

    def close(self) -> None:
        with self._lock:
            if self._process.poll() is not None:
                return
            assert self._process.stdin is not None
            try:
                self._process.stdin.write('{"command":"shutdown"}\n')
                self._process.stdin.flush()
                self._process.wait(timeout=20)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()


_WORKERS: dict[Sam3DWorkerKey, Sam3DWorker] = {}


def get_task_sam3d_worker(
    *,
    project_root: Path,
    environment: Path,
    script: Path,
    config: Path,
) -> Sam3DWorker:
    key = Sam3DWorkerKey(
        environment=str(environment.resolve()),
        script=str(script.resolve()),
        config=str(config.resolve()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    worker = _WORKERS.get(key)
    if worker is None or worker._process.poll() is not None:
        worker = Sam3DWorker(
            project_root=project_root,
            environment=environment,
            script=script,
            config=config,
        )
        _WORKERS[key] = worker
    return worker


def close_task_sam3d_workers() -> None:
    for worker in tuple(_WORKERS.values()):
        worker.close()
    _WORKERS.clear()


atexit.register(close_task_sam3d_workers)
