"""Prepare SAM3D reconstruction inputs and acquire object assets."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from atw.perception.sam3d_worker import get_task_sam3d_worker
from atw.perception.pipeline import scene as tracking
from atw.perception.pipeline.config import PROJECT_ROOT
from atw.perception.pipeline.recording import write_json as _write_json


def prepare_rigid_reconstruction_manifest(
    *,
    output_dir: Path,
    geometry_result: Any,
    selections: dict[str, Any],
    tracks: dict[str, dict[int, np.ndarray]],
    rigid_specs: tuple[Any, ...],
    rigid_shapes: Mapping[str, str],
    debug: bool = False,
    physical_type: str = "rigid",
    reference_surface_priors: Mapping[str, Any] | None = None,
    tracking_branches: Mapping[str, str] | None = None,
) -> Path:
    input_dir = output_dir / f"work/{physical_type}_reconstruction_inputs"
    mask_dir = output_dir / "masks"
    input_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    frame_count, height, width = geometry_result.world_points.shape[:3]
    mask_union_cache: dict[int, np.ndarray] = {}

    def all_object_mask_union(frame_index: int) -> np.ndarray:
        cached = mask_union_cache.get(frame_index)
        if cached is not None:
            return cached
        union = np.zeros((height, width), dtype=bool)
        for track in tracks.values():
            raw_mask = track.get(frame_index)
            if raw_mask is not None:
                union |= tracking.resize_tracking_mask(
                    raw_mask, geometry_result.transform.output_size_wh
                )
        mask_union_cache[frame_index] = union
        return union

    for spec in rigid_specs:
        frame_index = selections[spec.name].reference_frame
        raw_mask = tracks[spec.track_id].get(frame_index)
        if raw_mask is None:
            raise KeyError(f"{spec.name} has no mask at selected frame {frame_index}")
        with np.load(mask_dir / f"{spec.name}.npz") as archive:
            masks = archive["masks"]
        input_path = input_dir / f"{spec.name}.npz"
        np.savez_compressed(
            input_path,
            rgb=geometry_result.rgb[frame_index],
            mask=masks[frame_index],
            camera_points=geometry_result.camera_points[frame_index],
            camera_pose=geometry_result.camera_poses[frame_index],
            intrinsics=geometry_result.intrinsics[frame_index],
            all_object_mask_union=all_object_mask_union(frame_index),
        )
        objects.append(
            {
                "track_id": spec.name,
                "mask_track_id": spec.track_id,
                "physical_type": physical_type,
                "tracking_branch": (tracking_branches or {}).get(spec.name),
                "rigid_shape": rigid_shapes[spec.name],
                "frame_index": frame_index,
                "input_npz": str(input_path),
                "seed": 42,
                "debug": debug,
                "surface_point_count": 20000,
                "reference_surface_prior": (reference_surface_priors or {}).get(spec.name),
            }
        )
    manifest_path = input_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "coordinate_frame": "track4world_base_reconstruction",
            "camera_point_convention": "opencv_x_right_y_down_z_forward",
            "objects": objects,
        },
    )
    return manifest_path


def run_sam3d_rigid_reconstruction(
    *,
    manifest_path: Path,
    output_dir: Path,
    sam3d_environment: Path,
    physical_type: str = "rigid",
) -> Path:
    model_dir = output_dir / f"models/{physical_type}"
    worker = get_task_sam3d_worker(
        project_root=PROJECT_ROOT,
        environment=sam3d_environment,
        script=PROJECT_ROOT / "scripts/run_sam3d_rigid_reconstruction.py",
        config=PROJECT_ROOT / "data/models/sam-3d-objects/pipeline.yaml",
    )
    worker.run(manifest=manifest_path, output_dir=model_dir)
    if not (model_dir / "summary.json").is_file():
        raise FileNotFoundError("SAM 3D rigid reconstruction produced no summary")
    return model_dir
