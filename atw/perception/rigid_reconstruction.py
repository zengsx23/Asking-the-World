"""Metric conditioning for SAM 3D Objects meshes in Track4World space."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from atw.perception.mesh_projection import render_mask_occluded_mesh


SAM3D_GLB_TO_LOCAL = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
SAM3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True, slots=True)
class ConditionedRigidMesh:
    """A frozen metric mesh plus its reference-frame rigid pose."""

    canonical_vertices: np.ndarray
    faces: np.ndarray
    reference_pose: np.ndarray
    reference_world_vertices: np.ndarray
    local_origin: np.ndarray
    sam3d_scale: float
    track4world_world_scale: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "canonical_vertex_count": int(len(self.canonical_vertices)),
            "face_count": int(len(self.faces)),
            "local_origin": self.local_origin.tolist(),
            "sam3d_scale": self.sam3d_scale,
            "track4world_world_scale": self.track4world_world_scale,
            "reference_pose": self.reference_pose.tolist(),
            "canonical_extent": np.ptp(
                self.canonical_vertices, axis=0
            ).tolist(),
        }


@dataclass(frozen=True, slots=True)
class ProjectionAlignment:
    """PhysMind-style reference-frame area and median-depth adjustment."""

    uniform_scale: float
    camera_depth_translation: float
    target_median_depth: float
    initial_visible_iou: float
    final_visible_iou: float
    initial_raw_iou: float
    final_raw_iou: float
    final_area_ratio: float
    final_depth_error: float
    final_relative_depth_error: float
    final_occluded_rendered_pixels: int
    iteration_count: int
    stop_reason: str
    alignment_trace: tuple[dict[str, Any], ...]
    method: str = "foundationpose_then_physmind_area_and_median_depth_alignment"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "uniform_scale": self.uniform_scale,
            "camera_depth_translation": self.camera_depth_translation,
            "target_median_depth": self.target_median_depth,
            "initial_visible_iou": self.initial_visible_iou,
            "final_visible_iou": self.final_visible_iou,
            "initial_raw_iou": self.initial_raw_iou,
            "final_raw_iou": self.final_raw_iou,
            "final_area_ratio": self.final_area_ratio,
            "final_depth_error": self.final_depth_error,
            "final_relative_depth_error": self.final_relative_depth_error,
            "final_occluded_rendered_pixels": (
                self.final_occluded_rendered_pixels
            ),
            "iteration_count": self.iteration_count,
            "stop_reason": self.stop_reason,
            "alignment_trace": list(self.alignment_trace),
        }


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Convert a SAM 3D scalar-first quaternion to a rotation matrix."""

    values = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.shape != (4,):
        raise ValueError("quaternion must contain exactly four values")
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("quaternion norm must be positive and finite")
    w, x, y, z = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def positive_uniform_scale(scale: np.ndarray) -> float:
    values = np.asarray(scale, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values) & (values > 0)]
    if not len(finite):
        raise ValueError("scale has no positive finite value")
    return float(np.median(finite))


def decompose_similarity_pose(
    camera_to_world: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Decompose Track4World's camera-to-world Sim(3) matrix."""

    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera-to-world pose must have shape (4, 4)")
    linear = pose[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("camera-to-world linear transform must preserve orientation")
    scale = determinant ** (1.0 / 3.0)
    left, _, right_transpose = np.linalg.svd(linear / scale)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    residual = np.linalg.norm(linear - scale * rotation)
    if residual > max(1e-5, 1e-4 * np.linalg.norm(linear)):
        raise ValueError("camera-to-world pose is not a uniform Sim(3)")
    return scale, rotation, pose[:3, 3].copy()


def sam3d_pointmap_from_opencv(camera_points: np.ndarray) -> np.ndarray:
    """Convert Track4World OpenCV camera points to SAM 3D/PyTorch3D axes."""

    points = np.asarray(camera_points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("camera point map must have shape (height, width, 3)")
    converted = points.copy()
    converted[..., :2] *= -1.0
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    converted[~valid] = np.nan
    return converted


def apply_rigid_pose(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    transform = np.asarray(pose, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("pose must have shape (4, 4)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def optimize_reference_scale_and_depth(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    object_origin_camera: np.ndarray,
    intrinsics: np.ndarray,
    target_mask: np.ndarray,
    all_object_mask_union: np.ndarray,
    target_camera_points: np.ndarray,
    area_tolerance: float = 0.01,
    depth_tolerance: float = 0.005,
    max_iterations: int = 4,
) -> tuple[ProjectionAlignment, np.ndarray, np.ndarray]:
    """Replicate PhysMind's iterative projected-area and median-depth fit."""

    vertices = np.asarray(vertices_camera, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    origin = np.asarray(object_origin_camera, dtype=np.float64).reshape(3)
    mask = np.asarray(target_mask, dtype=bool)
    all_masks = np.asarray(all_object_mask_union, dtype=bool)
    camera_points = np.asarray(target_camera_points, dtype=np.float64)
    if camera_points.shape != (*mask.shape, 3):
        raise ValueError("target camera points must match the target mask")
    valid_target_depth = (
        mask
        & np.isfinite(camera_points).all(axis=-1)
        & (camera_points[..., 2] > 0)
    )
    if not np.any(valid_target_depth):
        raise ValueError("target mask has no valid Track4World depth pixels")
    target_depth = camera_points[..., 2]
    target_median_depth = float(np.median(target_depth[valid_target_depth]))
    target_area = int(mask.sum())
    if target_area <= 0:
        raise ValueError("target mask area is zero")

    def metrics(posed_vertices: np.ndarray) -> dict[str, Any]:
        rendered = render_mask_occluded_mesh(
            vertices_camera=posed_vertices,
            faces=triangles,
            intrinsic=intrinsics,
            image_shape=mask.shape,
            occluder_mask=all_masks,
            self_mask=mask,
        )
        raw = np.asarray(rendered["rendered_mask"], dtype=bool)
        visible = np.asarray(rendered["visible_mask"], dtype=bool)
        rendered_depth = np.asarray(rendered["rendered_depth"], dtype=np.float64)
        visible_area = int(visible.sum())
        if visible_area <= 0:
            raise ValueError("rendered mesh has an empty visible silhouette")
        positive_depth = rendered_depth[
            visible & np.isfinite(rendered_depth) & (rendered_depth > 0)
        ]
        if not len(positive_depth):
            raise ValueError("rendered mesh has no positive visible depth")
        median_depth = float(np.median(positive_depth))
        return {
            "rendered": raw,
            "visible": visible,
            "occluded": raw & ~visible,
            "visible_iou": _mask_iou(visible, mask),
            "raw_iou": _mask_iou(raw, mask),
            "rendered_area": visible_area,
            "raw_rendered_area": int(raw.sum()),
            "area_ratio": float(visible_area / target_area),
            "area_error_fraction": float(
                (visible_area - target_area) / target_area
            ),
            "rendered_median_depth": median_depth,
            "depth_error": float(median_depth - target_median_depth),
        }

    initial = metrics(vertices)
    area_scale = 1.0
    depth_translation = 0.0
    final = initial
    trace: list[dict[str, Any]] = []
    stop_reason = "max_iterations"
    for iteration in range(max_iterations):
        posed_vertices = origin + (vertices - origin) * area_scale
        posed_vertices[:, 2] += depth_translation
        pre_depth = metrics(posed_vertices)
        depth_delta = (
            target_median_depth - pre_depth["rendered_median_depth"]
        )
        depth_translation += depth_delta
        posed_vertices = posed_vertices.copy()
        posed_vertices[:, 2] += depth_delta
        final = metrics(posed_vertices)
        area_error = abs(float(final["area_error_fraction"]))
        relative_depth_error = abs(float(final["depth_error"])) / max(
            target_median_depth, 1e-12
        )
        trace.append(
            {
                "iteration": iteration + 1,
                "area_scale": float(area_scale),
                "camera_depth_translation": float(depth_translation),
                "depth_delta_applied": float(depth_delta),
                "visible_iou": float(final["visible_iou"]),
                "raw_iou": float(final["raw_iou"]),
                "area_ratio": float(final["area_ratio"]),
                "area_error_fraction": float(
                    final["area_error_fraction"]
                ),
                "rendered_median_depth": float(
                    final["rendered_median_depth"]
                ),
                "depth_error": float(final["depth_error"]),
                "relative_depth_error": float(relative_depth_error),
            }
        )
        if area_error <= area_tolerance and relative_depth_error <= depth_tolerance:
            stop_reason = "area_and_depth_tolerance"
            break
        if iteration + 1 >= max_iterations:
            break
        area_step = float(np.sqrt(target_area / final["rendered_area"]))
        if not np.isfinite(area_step) or area_step <= 0:
            raise ValueError(f"invalid area alignment scale step: {area_step}")
        area_scale *= area_step

    relative_depth_error = abs(float(final["depth_error"])) / max(
        target_median_depth, 1e-12
    )
    alignment = ProjectionAlignment(
        uniform_scale=float(area_scale),
        camera_depth_translation=float(depth_translation),
        target_median_depth=target_median_depth,
        initial_visible_iou=float(initial["visible_iou"]),
        final_visible_iou=float(final["visible_iou"]),
        initial_raw_iou=float(initial["raw_iou"]),
        final_raw_iou=float(final["raw_iou"]),
        final_area_ratio=float(final["area_ratio"]),
        final_depth_error=float(final["depth_error"]),
        final_relative_depth_error=float(relative_depth_error),
        final_occluded_rendered_pixels=int(final["occluded"].sum()),
        iteration_count=len(trace),
        stop_reason=stop_reason,
        alignment_trace=tuple(trace),
    )
    return alignment, final["rendered"], final["visible"]


def reframe_box_canonical_mesh(
    conditioned: ConditionedRigidMesh,
    *,
    axes: np.ndarray,
    center: np.ndarray,
) -> ConditionedRigidMesh:
    """Use fitted box axes as canonical XYZ without moving its world geometry.

    Gravity alignment and FoundationPose expect object axes and a centered local
    mesh. Rotate/recenter the vertices and compose the inverse frame change into
    the reference pose together; changing only either side changes the object.
    """
    axes = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    vertices = (np.asarray(conditioned.canonical_vertices) - center) @ axes
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = axes
    frame[:3, 3] = center
    pose = np.asarray(conditioned.reference_pose, dtype=np.float64) @ frame
    return replace(
        conditioned,
        canonical_vertices=vertices.astype(np.float32),
        reference_pose=pose.astype(np.float32),
        reference_world_vertices=apply_rigid_pose(vertices, pose).astype(np.float32),
        local_origin=(
            conditioned.local_origin
            + center / (conditioned.sam3d_scale * conditioned.track4world_world_scale)
        ).astype(np.float32),
    )


def reference_camera_pose(
    conditioned: ConditionedRigidMesh,
    *,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Express the conditioned object's rigid pose in the reference camera."""

    world_scale, camera_rotation, camera_translation = decompose_similarity_pose(
        camera_to_world
    )
    world_pose = np.asarray(conditioned.reference_pose, dtype=np.float64)
    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[:3, :3] = camera_rotation.T @ world_pose[:3, :3]
    camera_pose[:3, 3] = (
        camera_rotation.T @ (world_pose[:3, 3] - camera_translation)
    ) / world_scale
    return camera_pose


def apply_reference_camera_pose(
    conditioned: ConditionedRigidMesh,
    *,
    camera_to_world: np.ndarray,
    reference_pose_camera: np.ndarray,
) -> ConditionedRigidMesh:
    """Replace only the reference pose, preserving the complete canonical mesh."""

    world_scale, camera_rotation, camera_translation = decompose_similarity_pose(
        camera_to_world
    )
    camera_pose = np.asarray(reference_pose_camera, dtype=np.float64).reshape(4, 4)
    reference_pose = np.eye(4, dtype=np.float64)
    reference_pose[:3, :3] = camera_rotation @ camera_pose[:3, :3]
    reference_pose[:3, 3] = (
        world_scale * (camera_rotation @ camera_pose[:3, 3])
        + camera_translation
    )
    canonical_vertices = np.asarray(
        conditioned.canonical_vertices, dtype=np.float64
    )
    reference_world_vertices = apply_rigid_pose(
        canonical_vertices, reference_pose
    )
    return ConditionedRigidMesh(
        canonical_vertices=canonical_vertices.astype(np.float32),
        faces=np.asarray(conditioned.faces, dtype=np.int32),
        reference_pose=reference_pose.astype(np.float32),
        reference_world_vertices=reference_world_vertices.astype(np.float32),
        local_origin=np.asarray(conditioned.local_origin, dtype=np.float32),
        sam3d_scale=conditioned.sam3d_scale,
        track4world_world_scale=conditioned.track4world_world_scale,
    )


def apply_projection_alignment(
    conditioned: ConditionedRigidMesh,
    *,
    camera_to_world: np.ndarray,
    alignment: ProjectionAlignment,
) -> ConditionedRigidMesh:
    """Apply reference-camera scale/depth refinement in Track4World space."""

    camera_pose = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    canonical_vertices = (
        np.asarray(conditioned.canonical_vertices, dtype=np.float64)
        * alignment.uniform_scale
    )
    reference_pose = np.asarray(conditioned.reference_pose, dtype=np.float64).copy()
    camera_depth_delta = np.asarray(
        [0.0, 0.0, alignment.camera_depth_translation], dtype=np.float64
    )
    reference_pose[:3, 3] += camera_pose[:3, :3] @ camera_depth_delta
    reference_world_vertices = apply_rigid_pose(canonical_vertices, reference_pose)
    return ConditionedRigidMesh(
        canonical_vertices=canonical_vertices.astype(np.float32),
        faces=np.asarray(conditioned.faces, dtype=np.int32),
        reference_pose=reference_pose.astype(np.float32),
        reference_world_vertices=reference_world_vertices.astype(np.float32),
        local_origin=np.asarray(conditioned.local_origin, dtype=np.float32),
        sam3d_scale=conditioned.sam3d_scale * alignment.uniform_scale,
        track4world_world_scale=conditioned.track4world_world_scale,
    )


def condition_sam3d_mesh(
    glb_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sam3d_rotation_wxyz: np.ndarray,
    sam3d_translation: np.ndarray,
    sam3d_scale: np.ndarray,
    camera_to_world: np.ndarray,
) -> ConditionedRigidMesh:
    """Freeze a SAM 3D mesh in metric units and place it in Track4World space."""

    vertices = np.asarray(glb_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("GLB vertices must have nonzero shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")

    local_vertices = vertices @ SAM3D_GLB_TO_LOCAL.T
    local_origin = (local_vertices.min(axis=0) + local_vertices.max(axis=0)) * 0.5
    object_scale = positive_uniform_scale(sam3d_scale)
    sam_rotation = quaternion_wxyz_to_rotation(sam3d_rotation_wxyz)
    sam_translation = np.asarray(sam3d_translation, dtype=np.float64).reshape(-1)
    if sam_translation.shape != (3,):
        raise ValueError("SAM 3D translation must contain exactly three values")

    world_scale, camera_rotation, camera_translation = decompose_similarity_pose(
        camera_to_world
    )
    canonical_vertices = (
        (local_vertices - local_origin) * object_scale * world_scale
    )

    pose_rotation = camera_rotation @ SAM3D_TO_OPENCV @ sam_rotation.T
    origin_in_sam_camera = (
        local_origin * object_scale @ sam_rotation + sam_translation
    )
    origin_in_opencv_camera = origin_in_sam_camera @ SAM3D_TO_OPENCV
    pose_translation = (
        world_scale * (camera_rotation @ origin_in_opencv_camera)
        + camera_translation
    )
    reference_pose = np.eye(4, dtype=np.float64)
    reference_pose[:3, :3] = pose_rotation
    reference_pose[:3, 3] = pose_translation
    reference_world_vertices = apply_rigid_pose(
        canonical_vertices, reference_pose
    )

    direct_sam_camera = (
        local_vertices * object_scale @ sam_rotation + sam_translation
    )
    direct_opencv_camera = direct_sam_camera @ SAM3D_TO_OPENCV
    direct_world = (
        world_scale * (direct_opencv_camera @ camera_rotation.T)
        + camera_translation
    )
    if not np.allclose(reference_world_vertices, direct_world, atol=1e-5):
        raise RuntimeError("conditioned rigid pose failed its frame-consistency check")

    return ConditionedRigidMesh(
        canonical_vertices=canonical_vertices.astype(np.float32),
        faces=triangles.astype(np.int32),
        reference_pose=reference_pose.astype(np.float32),
        reference_world_vertices=reference_world_vertices.astype(np.float32),
        local_origin=local_origin.astype(np.float32),
        sam3d_scale=object_scale,
        track4world_world_scale=world_scale,
    )
