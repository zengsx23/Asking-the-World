"""Normalize Track4World geometry for videos captured by one fixed camera."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from atw.perception.sparse_pointcloud import uniform_sampled_frame_indices
from atw.perception.track4world_tool import SceneGeometryResult


@dataclass(frozen=True, slots=True)
class FixedCameraCalibration:
    """One camera calibration estimated from uniformly sampled predictions."""

    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    sampled_frame_indices: np.ndarray
    translation_deviation_median: float
    translation_deviation_p95: float
    rotation_deviation_degrees_median: float
    rotation_deviation_degrees_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "fixed_camera",
            "sampled_frame_indices": self.sampled_frame_indices.tolist(),
            "camera_to_world": self.camera_to_world.tolist(),
            "intrinsics": self.intrinsics.tolist(),
            "original_translation_deviation_m": {
                "median": self.translation_deviation_median,
                "p95": self.translation_deviation_p95,
            },
            "original_rotation_deviation_degrees": {
                "median": self.rotation_deviation_degrees_median,
                "p95": self.rotation_deviation_degrees_p95,
            },
        }


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    matrix = np.mean(rotations, axis=0)
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _rotation_degrees(rotations: np.ndarray, reference: np.ndarray) -> np.ndarray:
    relative = np.einsum("ij,tjk->tik", reference.T, rotations)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1, 1)
    return np.degrees(np.arccos(cosine))


def estimate_fixed_camera_calibration(
    camera_poses: np.ndarray,
    intrinsics: np.ndarray,
    *,
    sample_count: int = 16,
) -> FixedCameraCalibration:
    """Estimate one fixed camera from uniformly sampled Track4World outputs."""

    poses = np.asarray(camera_poses, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("camera poses must have shape (T, 4, 4)")
    if matrices.shape != (len(poses), 3, 3):
        raise ValueError("intrinsics must have shape (T, 3, 3)")
    if not np.isfinite(poses).all() or not np.isfinite(matrices).all():
        raise ValueError("fixed-camera inputs must be finite")

    sampled = uniform_sampled_frame_indices(len(poses), sample_count=sample_count)
    sampled_poses = poses[sampled]
    fixed_rotation = _mean_rotation(sampled_poses[:, :3, :3])
    fixed_translation = np.median(sampled_poses[:, :3, 3], axis=0)
    fixed_pose = np.eye(4, dtype=np.float64)
    fixed_pose[:3, :3] = fixed_rotation
    fixed_pose[:3, 3] = fixed_translation
    fixed_intrinsics = np.median(matrices[sampled], axis=0)

    translation_deviation = np.linalg.norm(
        poses[:, :3, 3] - fixed_translation,
        axis=1,
    )
    rotation_deviation = _rotation_degrees(poses[:, :3, :3], fixed_rotation)
    return FixedCameraCalibration(
        camera_to_world=fixed_pose.astype(np.float32),
        intrinsics=fixed_intrinsics.astype(np.float32),
        sampled_frame_indices=sampled,
        translation_deviation_median=float(np.median(translation_deviation)),
        translation_deviation_p95=float(np.quantile(translation_deviation, 0.95)),
        rotation_deviation_degrees_median=float(np.median(rotation_deviation)),
        rotation_deviation_degrees_p95=float(np.quantile(rotation_deviation, 0.95)),
    )


def apply_fixed_camera_calibration(
    geometry: SceneGeometryResult,
    calibration: FixedCameraCalibration,
) -> SceneGeometryResult:
    """Express every camera point map in one fixed world coordinate system."""

    source_camera_points = np.asarray(geometry.camera_points, dtype=np.float32)
    frame_count, height, width = source_camera_points.shape[:3]
    y, x = np.mgrid[:height, :width]
    z = source_camera_points[..., 2]
    intrinsic = np.asarray(calibration.intrinsics, dtype=np.float32)
    camera_points = np.stack(
        (
            (x[None] - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (y[None] - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ),
        axis=-1,
    ).astype(np.float32)
    rotation = np.asarray(calibration.camera_to_world[:3, :3], dtype=np.float32)
    translation = np.asarray(calibration.camera_to_world[:3, 3], dtype=np.float32)
    world_points = np.einsum(
        "...j,ij->...i",
        camera_points,
        rotation,
    )
    world_points += translation
    return replace(
        geometry,
        intrinsics=np.repeat(
            calibration.intrinsics[None], frame_count, axis=0
        ),
        camera_poses=np.repeat(
            calibration.camera_to_world[None], frame_count, axis=0
        ),
        camera_points=camera_points,
        world_points=world_points.astype(np.float32),
    )
