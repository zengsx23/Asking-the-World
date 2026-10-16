"""Manifest and loaders for reusable point-cloud reconstruction assets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from atw.perception.track4world_tool import SceneGeometryResult, Track4WorldTransform


SCHEMA_NAME = "physmind.pointcloud_reconstruction_assets"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PointCloudReconstructionAssets:
    manifest_path: Path
    sam3_result_dir: Path
    reference_selections_path: Path
    fixed_camera_calibration_path: Path
    geometry_dir: Path
    tracking_masks_dir: Path
    rigid_models_dir: Path
    support_plane_gravity_dir: Path
    original_size_wh: tuple[int, int]
    model_size_wh: tuple[int, int]

    soft_body_models_dir: Path

    def load_scene_geometry(self) -> SceneGeometryResult:
        """Restore dense camera/world geometry from retained depth and cameras."""

        rgb = np.load(self.geometry_dir / "rgb.npy")
        intrinsics = np.load(self.geometry_dir / "intrinsics.npy")
        camera_poses = np.load(self.geometry_dir / "camera_poses.npy")
        depth = np.load(self.geometry_dir / "depth.npy")
        geometry_confidence = np.load(
            self.geometry_dir / "geometry_confidence.npy"
        )
        geometry_valid = np.load(
            self.geometry_dir / "geometry_valid_threshold_0.npy"
        ).astype(bool)

        frame_count, height, width = depth.shape
        y, x = np.mgrid[:height, :width]
        camera_points = np.stack(
            (
                (x[None] - intrinsics[:, None, None, 0, 2])
                * depth
                / intrinsics[:, None, None, 0, 0],
                (y[None] - intrinsics[:, None, None, 1, 2])
                * depth
                / intrinsics[:, None, None, 1, 1],
                depth,
            ),
            axis=-1,
        ).astype(np.float32)
        world_points = np.einsum(
            "t...j,tij->t...i",
            camera_points,
            camera_poses[:, :3, :3],
        )
        world_points += camera_poses[:, None, None, :3, 3]
        transform = Track4WorldTransform(
            original_size_wh=self.original_size_wh,
            output_size_wh=self.model_size_wh,
            maximum_dimension=max(self.model_size_wh),
            dimension_multiple=64,
        )
        if frame_count != len(rgb):
            raise ValueError("retained geometry frame counts do not match")
        return SceneGeometryResult(
            transform=transform,
            rgb=np.asarray(rgb, dtype=np.uint8),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
            camera_poses=np.asarray(camera_poses, dtype=np.float32),
            camera_points=camera_points,
            world_points=world_points.astype(np.float32),
            geometry_confidence=np.asarray(
                geometry_confidence, dtype=np.float32
            ),
            geometry_valid=geometry_valid,
        )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def write_reconstruction_manifest(
    path: Path,
    *,
    sam3_result_dir: Path,
    original_size_wh: tuple[int, int],
    model_size_wh: tuple[int, int],
    tracked_objects: list[Mapping[str, Any]],
) -> None:
    root = path.parent
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "sam3_result_dir": os.path.relpath(sam3_result_dir.resolve(), root.resolve()),
        "original_size_wh": list(original_size_wh),
        "model_size_wh": list(model_size_wh),
        "tracked_objects": list(tracked_objects),
        "paths": {
            "reference_selections": "reference_frame_selections.json",
            "fixed_camera_calibration": "fixed_camera_calibration.json",
            "geometry": "geometry/reference_0000",
            "tracking_masks": "masks",
            "rigid_models": "models/rigid",
            "soft_body_models": "models/soft_body",
            "support_plane_gravity": (
                "surface"
            ),
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_reconstruction_assets(
    directory: Path,
) -> PointCloudReconstructionAssets:
    manifest_path = directory.resolve() / "reconstruction_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_NAME:
        raise ValueError("unexpected reconstruction manifest schema")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported reconstruction manifest version")
    paths = payload["paths"]
    root = manifest_path.parent
    return PointCloudReconstructionAssets(
        manifest_path=manifest_path,
        sam3_result_dir=_resolve(root, payload["sam3_result_dir"]),
        reference_selections_path=_resolve(root, paths["reference_selections"]),
        fixed_camera_calibration_path=_resolve(
            root, paths["fixed_camera_calibration"]
        ),
        geometry_dir=_resolve(root, paths["geometry"]),
        tracking_masks_dir=_resolve(root, paths["tracking_masks"]),
        soft_body_models_dir=_resolve(
            root,
            paths.get("soft_body_models", "intermediates/soft_body_models"),
        ),
        rigid_models_dir=_resolve(root, paths["rigid_models"]),
        support_plane_gravity_dir=_resolve(
            root, paths["support_plane_gravity"]
        ),
        original_size_wh=tuple(int(x) for x in payload["original_size_wh"]),
        model_size_wh=tuple(int(x) for x in payload["model_size_wh"]),
    )
