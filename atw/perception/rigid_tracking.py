"""Pure point-cloud SE(3) tracking for frozen canonical rigid geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import cv2
import numpy as np
from scipy.spatial import cKDTree

from atw.perception.pointcloud_projection import sparse_point_zbuffer, sparse_point_zbuffers
from atw.perception.rigid_reconstruction import apply_rigid_pose
from atw.perception.sparse_pointcloud import deterministic_subsample_indices
from atw.perception.tracking_mask_policy import classify_tracking_mask_frames, confirmed_occlusion_termination_frame


RigidMotionModel = Literal["se3", "translation_only", "axisymmetric"]


@dataclass(frozen=True, slots=True)
class RigidRegistration:
    pose: np.ndarray
    correspondence_count: int
    inlier_count: int
    rmse: float
    observed_coverage: float
    numerically_converged: bool
    has_sufficient_correspondences: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pose"] = self.pose.tolist()
        return payload


@dataclass(frozen=True, slots=True)
class RigidPoseTrajectory:
    canonical_tracking_points: np.ndarray
    canonical_tracking_indices: np.ndarray
    tracking_poses: np.ndarray
    poses: np.ndarray
    mask_pixel_counts: np.ndarray
    occlusion_termination_frame: int
    occlusion_terminated: np.ndarray
    border_touching: np.ndarray
    nearest_nonborder_frames: np.ndarray
    border_area_ratios: np.ndarray
    border_suppressed: np.ndarray
    observed_point_counts: np.ndarray
    correspondence_counts: np.ndarray
    inlier_counts: np.ndarray
    rmse: np.ndarray
    observed_coverage: np.ndarray
    motion_initialized: np.ndarray
    motion_correspondence_counts: np.ndarray
    motion_inlier_counts: np.ndarray
    motion_rmse: np.ndarray
    segment_anchor_frames: np.ndarray
    numerically_converged: np.ndarray
    has_sufficient_correspondences: np.ndarray
    self_visible_canonical_counts: np.ndarray
    other_mask_occluded_canonical_counts: np.ndarray
    occlusion_eligible_canonical_counts: np.ndarray
    support_constrained: np.ndarray
    support_constraint_rejected: np.ndarray
    support_penetration_depth: np.ndarray
    support_ray_scale: np.ndarray
    support_correction_norm: np.ndarray
    support_clearance_after: np.ndarray


@dataclass(frozen=True, slots=True)
class PointCloudSupportConstraint:
    """One fixed world support plane used only as a non-penetration boundary."""

    up_world: np.ndarray
    plane_offset_world: float
    maximum_correction_diagonals: float = 2.0


@dataclass(frozen=True, slots=True)
class PointCloudSupportCorrection:
    """Result of moving one frozen rigid cloud along its camera sightline."""

    pose: np.ndarray
    constrained: bool
    valid: bool
    penetration_depth: float
    ray_scale: float
    correction_norm: float
    clearance_after: float
    reason: str


def enforce_pointcloud_support_nonpenetration(
    *,
    canonical_points: np.ndarray,
    pose: np.ndarray,
    camera_to_world: np.ndarray,
    constraint: PointCloudSupportConstraint,
) -> PointCloudSupportCorrection:
    """Prevent a complete rigid cloud crossing a plane by camera-ray translation."""

    points = np.asarray(canonical_points, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    camera_pose = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("canonical support point cloud must be finite and nonempty")
    if not np.isfinite(transform).all() or not np.isfinite(camera_pose).all():
        raise ValueError("support constraint poses must be finite")
    up = np.asarray(constraint.up_world, dtype=np.float64).reshape(3)
    up_norm = float(np.linalg.norm(up))
    if not np.isfinite(up_norm) or up_norm <= 1e-12:
        raise ValueError("support up direction must have positive finite norm")
    up /= up_norm
    plane = float(constraint.plane_offset_world)
    if not np.isfinite(plane):
        raise ValueError("support plane offset must be finite")
    maximum_diagonals = float(constraint.maximum_correction_diagonals)
    if not np.isfinite(maximum_diagonals) or maximum_diagonals <= 0:
        raise ValueError("maximum support correction must be positive and finite")

    extent = np.ptp(points, axis=0)
    diagonal = float(np.linalg.norm(extent))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("canonical support point cloud must have positive extent")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    rotated = points @ rotation.T
    rotated_bottom = float(np.min(rotated @ up))
    bottom = rotated_bottom + float(np.dot(translation, up))
    penetration = max(0.0, plane - bottom)
    tolerance = max(1e-9, diagonal * 1e-7)
    if penetration <= tolerance:
        return PointCloudSupportCorrection(
            pose=transform.astype(np.float32),
            constrained=False,
            valid=True,
            penetration_depth=penetration,
            ray_scale=float("nan"),
            correction_norm=0.0,
            clearance_after=bottom - plane,
            reason="not_penetrating",
        )

    centroid_local = np.mean(points, axis=0)
    rotated_centroid = rotation @ centroid_local
    centroid_world = rotated_centroid + translation
    camera_center = camera_pose[:3, 3]
    camera_ray = centroid_world - camera_center
    denominator = float(np.dot(camera_ray, up))

    def invalid(reason: str, ray_scale: float = float("nan")):
        return PointCloudSupportCorrection(
            pose=transform.astype(np.float32),
            constrained=False,
            valid=False,
            penetration_depth=penetration,
            ray_scale=ray_scale,
            correction_norm=0.0,
            clearance_after=bottom - plane,
            reason=reason,
        )

    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return invalid("camera_ray_parallel_to_support_plane")
    ray_scale = float(
        (
            plane
            - rotated_bottom
            - float(np.dot(camera_center - rotated_centroid, up))
        )
        / denominator
    )
    if not np.isfinite(ray_scale) or ray_scale <= 0.0:
        return invalid("support_solution_behind_camera", ray_scale)
    corrected_centroid = camera_center + ray_scale * camera_ray
    corrected_translation = corrected_centroid - rotated_centroid
    correction = corrected_translation - translation
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > maximum_diagonals * diagonal:
        return invalid("support_correction_too_large", ray_scale)
    corrected_bottom = rotated_bottom + float(np.dot(corrected_translation, up))
    clearance = corrected_bottom - plane
    if clearance < -tolerance:
        return invalid("residual_support_penetration", ray_scale)
    corrected_pose = transform.copy()
    corrected_pose[:3, 3] = corrected_translation
    return PointCloudSupportCorrection(
        pose=corrected_pose.astype(np.float32),
        constrained=True,
        valid=True,
        penetration_depth=penetration,
        ray_scale=ray_scale,
        correction_norm=correction_norm,
        clearance_after=clearance,
        reason="corrected_along_camera_ray",
    )


def truncate_mask_track_at_confirmed_occlusion(
    track: Mapping[int, np.ndarray],
    *,
    frame_count: int,
    minimum_occlusion_frames: int = 3,
) -> tuple[dict[int, np.ndarray], int]:
    """Discard a mask track from its first confirmed occlusion frame onward."""

    if frame_count < 0:
        raise ValueError("frame count must be non-negative")
    counts = np.zeros(frame_count, dtype=np.int64)
    for frame, mask in track.items():
        if 0 <= frame < frame_count:
            counts[frame] = np.count_nonzero(mask)
    termination = confirmed_occlusion_termination_frame(
        counts,
        minimum_occlusion_frames=minimum_occlusion_frames,
    )
    if termination < 0:
        return dict(track), -1
    return {
        frame: mask for frame, mask in track.items() if frame < termination
    }, termination


def estimate_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Estimate the SE(3) transform mapping source points to target points."""

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError("source and target point arrays must match")
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if len(source) < 3:
        raise ValueError("at least three finite point pairs are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose.astype(np.float32)


def robust_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    inlier_distance: float,
    iterations: int = 256,
    seed: int = 0,
) -> RigidRegistration:
    """RANSAC rigid fit for noisy Track4World anchor correspondences."""

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if len(source) < 3:
        raise ValueError("at least three finite point pairs are required")
    if not np.isfinite(inlier_distance) or inlier_distance <= 0:
        raise ValueError("inlier distance must be positive and finite")

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(source), dtype=bool)
    best_error = np.inf
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        centered = source[sample] - source[sample].mean(axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            continue
        pose = estimate_rigid_transform(source[sample], target[sample])
        residual = np.linalg.norm(apply_rigid_pose(source, pose) - target, axis=1)
        inliers = residual <= inlier_distance
        count = int(inliers.sum())
        error = float(np.mean(residual[inliers])) if count else np.inf
        if count > int(best_inliers.sum()) or (
            count == int(best_inliers.sum()) and error < best_error
        ):
            best_inliers = inliers
            best_error = error
    if int(best_inliers.sum()) < 3:
        raise ValueError("RANSAC found fewer than three rigid inliers")
    pose = estimate_rigid_transform(source[best_inliers], target[best_inliers])
    residual = np.linalg.norm(apply_rigid_pose(source, pose) - target, axis=1)
    inliers = residual <= inlier_distance
    rmse = float(np.sqrt(np.mean(np.square(residual[inliers]))))
    return RigidRegistration(
        pose=pose,
        correspondence_count=len(source),
        inlier_count=int(inliers.sum()),
        rmse=rmse,
        observed_coverage=float(inliers.mean()),
        numerically_converged=True,
        has_sufficient_correspondences=True,
    )


def _bilinear_sample(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    points = np.asarray(uv, dtype=np.float64)
    if values.ndim != 3 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("field must be HxWxC and UV must be Nx2")
    height, width, channels = values.shape
    output = np.full((len(points), channels), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return output
    x = points[indices, 0]
    y = points[indices, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = values[y0, x0] * (1 - wx) + values[y0, x1] * wx
    bottom = values[y1, x0] * (1 - wx) + values[y1, x1] * wx
    output[indices] = top * (1 - wy) + bottom * wy
    return output


def estimate_anchor_motion(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_world_points: np.ndarray,
    target_world_points: np.ndarray,
    source_to_target_uv: np.ndarray,
    source_tracking_confidence: np.ndarray,
    minimum_tracking_confidence: float,
    inlier_distance: float,
    spatial_stride: int = 2,
    seed: int = 0,
) -> RigidRegistration:
    """Fit frame-to-frame rigid motion from Track4World flow and point maps."""

    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    height, width = source_mask.shape
    expected_points = (height, width, 3)
    if target_mask.shape != source_mask.shape:
        raise ValueError("source and target masks must match")
    if (
        np.asarray(source_world_points).shape != expected_points
        or np.asarray(target_world_points).shape != expected_points
        or np.asarray(source_to_target_uv).shape != (height, width, 2)
        or np.asarray(source_tracking_confidence).shape != source_mask.shape
    ):
        raise ValueError("anchor motion fields must match the masks")

    grid = np.zeros_like(source_mask)
    grid[::spatial_stride, ::spatial_stride] = True
    source_valid = (
        source_mask
        & grid
        & np.isfinite(source_world_points).all(axis=-1)
        & np.isfinite(source_to_target_uv).all(axis=-1)
        & np.isfinite(source_tracking_confidence)
        & (source_tracking_confidence >= minimum_tracking_confidence)
    )
    y, x = np.nonzero(source_valid)
    uv = np.asarray(source_to_target_uv)[y, x]
    rounded = np.rint(uv).astype(np.int64)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    indices = np.flatnonzero(inside)
    rounded_inside = rounded[inside]
    on_target = target_mask[rounded_inside[:, 1], rounded_inside[:, 0]]
    indices = indices[on_target]
    uv = uv[indices]
    source_points = np.asarray(source_world_points)[y[indices], x[indices]]
    target_points = _bilinear_sample(target_world_points, uv)
    return robust_rigid_transform(
        source_points,
        target_points,
        inlier_distance=inlier_distance,
        seed=seed,
    )


def occlusion_aware_canonical_visibility(
    *,
    canonical_points: np.ndarray,
    pose: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    self_mask: np.ndarray,
    all_object_mask_union: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Select frontmost canonical points not covered by another object mask."""

    own = np.asarray(self_mask, dtype=bool)
    union = np.asarray(all_object_mask_union, dtype=bool)
    if own.shape != union.shape:
        raise ValueError("self mask and all-object mask union must match")
    posed = apply_rigid_pose(canonical_points, pose)
    point_ids, _ = sparse_point_zbuffer(
        world_points=posed,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        image_size_wh=(own.shape[1], own.shape[0]),
        point_radius=0,
    )
    return _canonical_visibility_from_zbuffer(
        point_ids=point_ids,
        canonical_point_count=len(canonical_points),
        self_mask=own,
        all_object_mask_union=union,
    )


def _canonical_visibility_from_zbuffer(
    *,
    point_ids: np.ndarray,
    canonical_point_count: int,
    self_mask: np.ndarray,
    all_object_mask_union: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Derive canonical visibility statistics from an existing Z-buffer."""

    own = np.asarray(self_mask, dtype=bool)
    union = np.asarray(all_object_mask_union, dtype=bool)
    y, x = np.nonzero(point_ids >= 0)
    visible_ids = point_ids[y, x]
    other_mask = union & ~own
    other_occluded = other_mask[y, x]
    eligible_pixels = ~other_occluded
    eligible = np.zeros(canonical_point_count, dtype=bool)
    eligible[visible_ids[eligible_pixels]] = True
    return eligible, len(visible_ids), int(np.count_nonzero(other_occluded))


def _query_nearest_eligible_canonical(
    canonical_tree: cKDTree,
    points: np.ndarray,
    eligible_canonical: np.ndarray,
    *,
    neighbor_count: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the nearest eligible point from a fixed canonical KD-Tree."""

    if np.all(eligible_canonical):
        return canonical_tree.query(points, workers=1)
    count = min(neighbor_count, len(canonical_tree.data))
    distances, ids = canonical_tree.query(points, k=count, workers=1)
    if count == 1:
        distances = distances[:, None]
        ids = ids[:, None]
    eligible_neighbors = eligible_canonical[ids]
    has_eligible = eligible_neighbors.any(axis=1)
    first_eligible = np.argmax(eligible_neighbors, axis=1)
    rows = np.arange(len(points))
    selected_distance = distances[rows, first_eligible]
    selected_ids = ids[rows, first_eligible]
    selected_distance[~has_eligible] = np.inf
    selected_ids[~has_eligible] = 0
    return selected_distance, selected_ids


def _normalized_axis(axis: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("canonical symmetry axis must be finite and nonzero")
    return vector / norm


def _minimal_axis_rotation(source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    """Return the minimum-angle rotation mapping one unit axis to another."""

    source = _normalized_axis(source_axis)
    target = _normalized_axis(target_axis)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0:
            return np.eye(3, dtype=np.float64)
        candidate = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(source)))]
        perpendicular = _normalized_axis(np.cross(source, candidate))
        return 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
    axis = cross / sine
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def transport_axisymmetric_rotation(
    anchor_rotation: np.ndarray,
    predicted_rotation: np.ndarray,
    canonical_symmetry_axis: np.ndarray,
) -> np.ndarray:
    """Transport a cylinder axis while discarding unobservable axial twist."""

    anchor = np.asarray(anchor_rotation, dtype=np.float64).reshape(3, 3)
    predicted = np.asarray(predicted_rotation, dtype=np.float64).reshape(3, 3)
    axis = _normalized_axis(canonical_symmetry_axis)
    source_axis = anchor @ axis
    target_axis = predicted @ axis
    return _minimal_axis_rotation(source_axis, target_axis) @ anchor


def _axis_tangent_basis(world_axis: np.ndarray) -> np.ndarray:
    """Return two orthonormal world rotation directions normal to an axis."""

    axis = _normalized_axis(world_axis)
    candidate = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(axis)))]
    first = _normalized_axis(np.cross(axis, candidate))
    second = np.cross(axis, first)
    return np.column_stack((first, second))


def refine_pose_with_partial_icp(
    canonical_points: np.ndarray,
    canonical_tree: cKDTree,
    observed_world_points: np.ndarray,
    observed_mask: np.ndarray,
    all_object_mask_union: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    initial_pose: np.ndarray,
    *,
    motion_model: RigidMotionModel,
    canonical_symmetry_axis: np.ndarray | None = None,
    coverage_distance: float,
    iterations: int = 30,
    translation_tolerance: float = 1e-6,
    rotation_tolerance: float = 1e-5,
) -> RigidRegistration:
    """Refine a shape-aware rigid pose from depth and silhouette residuals."""

    if motion_model not in {"se3", "translation_only", "axisymmetric"}:
        raise ValueError(f"unsupported rigid motion model: {motion_model!r}")
    symmetry_axis = (
        _normalized_axis(canonical_symmetry_axis)
        if motion_model == "axisymmetric"
        else None
    )
    if motion_model == "axisymmetric" and symmetry_axis is None:
        raise ValueError("axisymmetric motion requires a canonical symmetry axis")
    parameter_count = {
        "se3": 6,
        "axisymmetric": 5,
        "translation_only": 3,
    }[motion_model]

    canonical = np.asarray(canonical_points, dtype=np.float64).reshape(-1, 3)
    observed_map = np.asarray(observed_world_points, dtype=np.float64)
    own = np.asarray(observed_mask, dtype=bool)
    union = np.asarray(all_object_mask_union, dtype=bool)
    height, width = own.shape
    if observed_map.shape != (height, width, 3):
        raise ValueError("observed world map must match the object mask")
    if union.shape != own.shape:
        raise ValueError("all-object mask union must match the object mask")

    valid_observed = own & np.isfinite(observed_map).all(axis=-1)
    normals = np.full_like(observed_map, np.nan, dtype=np.float64)
    horizontal = observed_map[1:-1, 2:] - observed_map[1:-1, :-2]
    vertical = observed_map[2:, 1:-1] - observed_map[:-2, 1:-1]
    interior_valid = (
        valid_observed[1:-1, 1:-1]
        & valid_observed[1:-1, 2:]
        & valid_observed[1:-1, :-2]
        & valid_observed[2:, 1:-1]
        & valid_observed[:-2, 1:-1]
    )
    interior_normals = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(interior_normals, axis=-1)
    normal_valid = interior_valid & np.isfinite(lengths) & (lengths > 0)
    normalized = np.full_like(interior_normals, np.nan)
    normalized[normal_valid] = (
        interior_normals[normal_valid] / lengths[normal_valid, None]
    )
    normals[1:-1, 1:-1] = normalized
    valid_observed &= np.isfinite(normals).all(axis=-1)
    valid_observed &= ~(union & ~own)

    mask_u8 = own.astype(np.uint8)
    erosion_kernel = np.ones((3, 3), dtype=np.uint8)
    target_boundary = own & ~cv2.erode(
        mask_u8,
        erosion_kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    distance_to_boundary = cv2.distanceTransform(
        (~target_boundary).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    ).astype(np.float64)
    signed_distance = distance_to_boundary
    signed_distance[own] *= -1.0
    signed_distance[target_boundary] = 0.0
    sdf_gradient_y, sdf_gradient_x = np.gradient(signed_distance)
    world_to_camera = np.linalg.inv(
        np.asarray(camera_to_world, dtype=np.float64)
    )
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    other_mask = union & ~own

    def project_pose(
        current_pose: np.ndarray,
        point_radii: tuple[int, ...],
    ) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
        posed = apply_rigid_pose(canonical, current_pose).astype(np.float64)
        buffers = sparse_point_zbuffers(
            world_points=posed,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            image_size_wh=(width, height),
            point_radii=point_radii,
        )
        return posed, buffers

    def correspondences(
        posed: np.ndarray,
        point_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, x = np.nonzero((point_ids >= 0) & valid_observed)
        ids = point_ids[y, x]
        return posed[ids], observed_map[y, x], normals[y, x]

    def silhouette_residual_and_jacobian(
        current_pose: np.ndarray,
        posed: np.ndarray,
        point_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rendered = point_ids >= 0
        rendered_boundary = rendered & ~cv2.erode(
            rendered.astype(np.uint8),
            erosion_kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        y, x = np.nonzero(rendered_boundary & ~other_mask)
        ids = point_ids[y, x]
        if not len(ids):
            return np.empty(0, dtype=np.float64), np.empty((0, 6))

        points = posed[ids]
        camera = (
            points @ world_to_camera[:3, :3].T
            + world_to_camera[:3, 3]
        )
        finite = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
        y = y[finite]
        x = x[finite]
        points = points[finite]
        camera = camera[finite]
        if not len(points):
            return np.empty(0, dtype=np.float64), np.empty((0, 6))

        residual = signed_distance[y, x]
        image_gradient = np.column_stack(
            (sdf_gradient_x[y, x], sdf_gradient_y[y, x])
        )
        projection_jacobian = np.zeros((len(points), 2, 3), dtype=np.float64)
        inverse_z = 1.0 / camera[:, 2]
        projection_jacobian[:, 0, 0] = intrinsic[0, 0] * inverse_z
        projection_jacobian[:, 0, 2] = (
            -intrinsic[0, 0] * camera[:, 0] * np.square(inverse_z)
        )
        projection_jacobian[:, 1, 1] = intrinsic[1, 1] * inverse_z
        projection_jacobian[:, 1, 2] = (
            -intrinsic[1, 1] * camera[:, 1] * np.square(inverse_z)
        )
        image_to_camera = np.einsum(
            "ni,nij->nj", image_gradient, projection_jacobian
        )
        image_to_world = image_to_camera @ world_to_camera[:3, :3]
        centered = points - current_pose[:3, 3]
        jacobian = np.column_stack(
            (np.cross(centered, image_to_world), image_to_world)
        )
        return residual, jacobian

    def normalized_robust_system(
        residual: np.ndarray,
        jacobian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(residual):
            return jacobian, residual
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        if scale == 0.0:
            scale = float(np.sqrt(np.mean(np.square(residual))))
        scale = max(scale, np.finfo(np.float64).eps)
        normalized_residual = residual / scale
        normalized_jacobian = jacobian / scale
        weight = 1.0 / (1.0 + np.square(normalized_residual))
        group_normalizer = np.sqrt(float(len(residual)))
        row_scale = np.sqrt(weight) / group_normalizer
        return (
            normalized_jacobian * row_scale[:, None],
            normalized_residual * row_scale,
        )

    def incremental_rotation(vector: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(vector))
        if angle == 0.0:
            return np.eye(3, dtype=np.float64)
        axis = vector / angle
        x, y, z = axis
        skew = np.asarray(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        return (
            np.eye(3, dtype=np.float64)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    pose = np.asarray(initial_pose, dtype=np.float64).copy()
    final_count = 0
    final_rank = 0
    final_rmse = float("nan")
    numerically_converged = False
    for _ in range(iterations):
        rotation_basis = (
            _axis_tangent_basis(pose[:3, :3] @ symmetry_axis)
            if symmetry_axis is not None
            else None
        )
        posed, buffers = project_pose(pose, (0, 2))
        source, target, target_normals = correspondences(
            posed, buffers[0][0]
        )
        final_count = len(source)
        if final_count:
            centered_source = source - pose[:3, 3]
            depth_residual = np.einsum(
                "ij,ij->i", target_normals, source - target
            )
            full_depth_jacobian = np.column_stack(
                (np.cross(centered_source, target_normals), target_normals)
            )
            if motion_model == "se3":
                depth_jacobian = full_depth_jacobian
            elif motion_model == "axisymmetric":
                depth_jacobian = np.column_stack(
                    (
                        full_depth_jacobian[:, :3] @ rotation_basis,
                        full_depth_jacobian[:, 3:],
                    )
                )
            else:
                depth_jacobian = full_depth_jacobian[:, 3:]
        else:
            depth_residual = np.empty(0, dtype=np.float64)
            depth_jacobian = np.empty(
                (0, parameter_count), dtype=np.float64
            )
        silhouette_residual, silhouette_jacobian = (
            silhouette_residual_and_jacobian(pose, posed, buffers[2][0])
        )
        if motion_model == "translation_only":
            silhouette_jacobian = silhouette_jacobian[:, 3:]
        elif motion_model == "axisymmetric":
            silhouette_jacobian = np.column_stack(
                (
                    silhouette_jacobian[:, :3] @ rotation_basis,
                    silhouette_jacobian[:, 3:],
                )
            )
        if not len(depth_residual):
            break
        depth_system, depth_target = normalized_robust_system(
            depth_residual, depth_jacobian
        )
        silhouette_system, silhouette_target = normalized_robust_system(
            silhouette_residual, silhouette_jacobian
        )
        weighted_jacobian = np.concatenate(
            (depth_system, silhouette_system), axis=0
        )
        weighted_residual = np.concatenate(
            (depth_target, silhouette_target), axis=0
        )
        delta, _, final_rank, _ = np.linalg.lstsq(
            weighted_jacobian,
            -weighted_residual,
            rcond=None,
        )
        if not np.isfinite(delta).all():
            break
        if motion_model in {"se3", "axisymmetric"}:
            rotation_vector = (
                delta[:3]
                if motion_model == "se3"
                else rotation_basis @ delta[:2]
            )
            translation_delta = (
                delta[3:] if motion_model == "se3" else delta[2:]
            )
            rotation_delta = incremental_rotation(rotation_vector)
            pose[:3, :3] = rotation_delta @ pose[:3, :3]
            pose[:3, 3] += translation_delta
            rotation_converged = (
                np.linalg.norm(rotation_vector) <= rotation_tolerance
            )
        else:
            pose[:3, 3] += delta
            rotation_converged = True
            translation_delta = delta
        if len(depth_residual):
            final_rmse = float(
                np.sqrt(np.mean(np.square(depth_residual)))
            )
        if (
            rotation_converged
            and np.linalg.norm(translation_delta) <= translation_tolerance
        ):
            numerically_converged = True
            break

    final_posed, final_buffers = project_pose(pose, (0,))
    final_point_ids = final_buffers[0][0]
    final_source, final_target, final_normals = correspondences(
        final_posed, final_point_ids
    )
    final_count = len(final_source)
    if final_count:
        final_residual = np.einsum(
            "ij,ij->i", final_normals, final_source - final_target
        )
        final_rmse = float(np.sqrt(np.mean(np.square(final_residual))))

    eligible_canonical, _, _ = _canonical_visibility_from_zbuffer(
        point_ids=final_point_ids,
        canonical_point_count=len(canonical),
        self_mask=own,
        all_object_mask_union=union,
    )
    observed = observed_map[np.asarray(observed_mask, dtype=bool)]
    observed = observed[np.isfinite(observed).all(axis=1)]
    if len(observed):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        observed_canonical = (observed - translation) @ rotation
        distance, final_ids = _query_nearest_eligible_canonical(
            canonical_tree,
            observed_canonical,
            eligible_canonical,
        )
        coverage = float(
            np.mean(
                (distance <= coverage_distance)
                & eligible_canonical[final_ids]
            )
        )
    else:
        coverage = 0.0
    has_sufficient_correspondences = (
        final_count > 0 and final_rank == parameter_count
    )
    return RigidRegistration(
        pose=pose.astype(np.float32),
        correspondence_count=final_count,
        inlier_count=final_count,
        rmse=final_rmse,
        observed_coverage=coverage,
        numerically_converged=numerically_converged,
        has_sufficient_correspondences=has_sufficient_correspondences,
    )


def static_rigid_pose_sequence(
    *,
    canonical_points: np.ndarray,
    reference_pose: np.ndarray,
    reference_frame: int,
    masks: np.ndarray,
    all_object_mask_union: np.ndarray,
    camera_poses: np.ndarray,
    intrinsics: np.ndarray,
    support_constraint: PointCloudSupportConstraint | None,
    maximum_canonical_points: int = 4_096,
) -> RigidPoseTrajectory:
    """Freeze an aligned reference pose; no flow, depth fitting, or temporal ICP.

    Nonpenetration is evaluated ONCE using the reference camera and fixed world
    plane (support or wall). Only visibility/output eligibility changes over time.
    """
    masks = np.asarray(masks, dtype=bool)
    policy = classify_tracking_mask_frames(masks)
    eligible = (policy.mask_pixel_counts > 0) & ~policy.border_suppressed & ~policy.occlusion_terminated
    if not eligible[reference_frame]:
        raise ValueError("static reference frame must be eligible for tracking")
    canonical_full = np.asarray(canonical_points, dtype=np.float32)
    indices = deterministic_subsample_indices(len(canonical_full), maximum_points=maximum_canonical_points, seed=0)
    canonical = canonical_full[indices]
    pose = np.asarray(reference_pose, dtype=np.float32).copy()
    correction = None
    if support_constraint is not None:
        correction = enforce_pointcloud_support_nonpenetration(
            canonical_points=canonical_full, pose=pose,
            camera_to_world=camera_poses[reference_frame], constraint=support_constraint)
        if not correction.valid:
            raise ValueError(f"cannot freeze a penetrating static pose: {correction.reason}")
        pose = correction.pose
    frame_count = len(masks)
    tracking_poses = np.full((frame_count, 4, 4), np.nan, dtype=np.float32)
    poses = tracking_poses.copy()
    tracking_poses[eligible] = reference_pose
    poses[eligible] = pose
    counts = np.zeros(frame_count, np.int32)
    flags = np.zeros(frame_count, bool)
    missing = np.full(frame_count, np.nan, np.float32)
    self_visible = counts.copy()
    other_occluded = counts.copy()
    eligible_counts = counts.copy()
    anchors = np.full(frame_count, -1, np.int32)
    anchors[eligible] = reference_frame
    for frame in np.flatnonzero(eligible):
        visible, self_visible[frame], other_occluded[frame] = occlusion_aware_canonical_visibility(
            canonical_points=canonical, pose=pose, camera_to_world=camera_poses[frame],
            intrinsics=intrinsics[frame], self_mask=masks[frame],
            all_object_mask_union=all_object_mask_union[frame])
        eligible_counts[frame] = np.count_nonzero(visible)
    support_constrained = flags.copy()
    penetration, ray_scale, clearance = missing.copy(), missing.copy(), missing.copy()
    correction_norm = np.zeros(frame_count, np.float32)
    if correction is not None:
        support_constrained[eligible] = correction.constrained
        penetration[eligible] = correction.penetration_depth
        ray_scale[eligible] = correction.ray_scale
        clearance[eligible] = correction.clearance_after
        correction_norm[eligible] = correction.correction_norm
    return RigidPoseTrajectory(
        canonical_tracking_points=canonical, canonical_tracking_indices=indices,
        tracking_poses=tracking_poses, poses=poses,
        mask_pixel_counts=policy.mask_pixel_counts,
        occlusion_termination_frame=policy.occlusion_termination_frame,
        occlusion_terminated=policy.occlusion_terminated,
        border_touching=policy.border_touching, nearest_nonborder_frames=policy.nearest_nonborder_frames,
        border_area_ratios=policy.border_area_ratios, border_suppressed=policy.border_suppressed,
        observed_point_counts=counts.copy(), correspondence_counts=counts.copy(),
        inlier_counts=counts.copy(), rmse=missing.copy(), observed_coverage=missing.copy(),
        motion_initialized=flags.copy(), motion_correspondence_counts=counts.copy(),
        motion_inlier_counts=counts.copy(), motion_rmse=missing.copy(), segment_anchor_frames=anchors,
        numerically_converged=flags.copy(), has_sufficient_correspondences=flags.copy(),
        self_visible_canonical_counts=self_visible, other_mask_occluded_canonical_counts=other_occluded,
        occlusion_eligible_canonical_counts=eligible_counts, support_constrained=support_constrained,
        support_constraint_rejected=flags.copy(), support_penetration_depth=penetration,
        support_ray_scale=ray_scale, support_correction_norm=correction_norm, support_clearance_after=clearance)


def track_rigid_pose_sequence(
    *,
    canonical_points: np.ndarray,
    reference_pose: np.ndarray,
    reference_frame: int,
    masks: np.ndarray,
    all_object_mask_union: np.ndarray,
    world_points: np.ndarray,
    geometry_confidence: np.ndarray,
    camera_poses: np.ndarray,
    intrinsics: np.ndarray,
    adjacent: Any,
    motion_model: RigidMotionModel,
    canonical_symmetry_axis: np.ndarray | None = None,
    minimum_geometry_confidence: float = 0.0,
    minimum_tracking_confidence: float = 0.0,
    maximum_canonical_points: int = 4_096,
    maximum_observed_points: int = 2_048,
    support_constraint: PointCloudSupportConstraint | None = None,
) -> RigidPoseTrajectory:
    """Track one rigid model with adjacent motion and shape-aware correction."""

    if motion_model not in {"se3", "translation_only", "axisymmetric"}:
        raise ValueError(f"unsupported rigid motion model: {motion_model!r}")
    symmetry_axis = (
        _normalized_axis(canonical_symmetry_axis)
        if motion_model == "axisymmetric"
        else None
    )
    if motion_model == "axisymmetric" and symmetry_axis is None:
        raise ValueError("axisymmetric motion requires a canonical symmetry axis")

    masks = np.asarray(masks, dtype=bool)
    all_object_mask_union = np.asarray(all_object_mask_union, dtype=bool)
    world_points = np.asarray(world_points, dtype=np.float32)
    geometry_confidence = np.asarray(geometry_confidence, dtype=np.float32)
    camera_poses = np.asarray(camera_poses, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    frame_count, height, width = masks.shape
    if world_points.shape != (frame_count, height, width, 3):
        raise ValueError("world point maps must match mask sequence")
    if geometry_confidence.shape != masks.shape:
        raise ValueError("geometry confidence must match mask sequence")
    if all_object_mask_union.shape != masks.shape:
        raise ValueError("all-object mask union must match mask sequence")
    if camera_poses.shape != (frame_count, 4, 4):
        raise ValueError("camera poses must have shape Tx4x4")
    if intrinsics.shape != (frame_count, 3, 3):
        raise ValueError("intrinsics must have shape Tx3x3")
    if maximum_canonical_points < 1 or maximum_observed_points < 1:
        raise ValueError("tracking point limits must be positive")
    if not 0 <= reference_frame < frame_count:
        raise ValueError("reference frame is outside the sequence")
    if int(adjacent.frame_count) != frame_count:
        raise ValueError("adjacent tracking frame count does not match masks")
    if adjacent.source_to_target_uv.shape != (frame_count - 1, height, width, 2):
        raise ValueError("adjacent UV fields must match consecutive mask pairs")
    if adjacent.tracking_confidence.shape != (frame_count - 1, height, width):
        raise ValueError("adjacent confidence fields must match consecutive mask pairs")
    mask_policy = classify_tracking_mask_frames(masks)
    if mask_policy.occlusion_terminated[reference_frame]:
        raise ValueError("reference frame occurs after permanent occlusion")
    if mask_policy.border_suppressed[reference_frame]:
        raise ValueError("reference frame is suppressed by the border mask policy")

    canonical_full = np.asarray(canonical_points, dtype=np.float32).reshape(-1, 3)
    extent = np.ptp(canonical_full, axis=0)
    diagonal = float(np.linalg.norm(extent))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("canonical point cloud must have positive extent")
    canonical_indices = deterministic_subsample_indices(
        len(canonical_full),
        maximum_points=maximum_canonical_points,
        seed=0,
    )
    canonical = canonical_full[canonical_indices]
    canonical_tree = cKDTree(np.asarray(canonical, dtype=np.float64))
    motion_distance = 0.12 * diagonal
    icp_distance = 0.10 * diagonal

    tracking_poses = np.full((frame_count, 4, 4), np.nan, dtype=np.float32)
    observed_counts = np.zeros(frame_count, dtype=np.int32)
    correspondence_counts = np.zeros(frame_count, dtype=np.int32)
    inlier_counts = np.zeros(frame_count, dtype=np.int32)
    rmse = np.full(frame_count, np.nan, dtype=np.float32)
    coverage = np.zeros(frame_count, dtype=np.float32)
    motion_initialized = np.zeros(frame_count, dtype=bool)
    motion_correspondence_counts = np.zeros(frame_count, dtype=np.int32)
    motion_inlier_counts = np.zeros(frame_count, dtype=np.int32)
    motion_rmse = np.full(frame_count, np.nan, dtype=np.float32)
    segment_anchor_frames = np.full(frame_count, -1, dtype=np.int32)
    numerically_converged = np.zeros(frame_count, dtype=bool)
    has_sufficient_correspondences = np.zeros(frame_count, dtype=bool)
    self_visible_canonical_counts = np.zeros(frame_count, dtype=np.int32)
    other_mask_occluded_canonical_counts = np.zeros(frame_count, dtype=np.int32)
    occlusion_eligible_canonical_counts = np.zeros(frame_count, dtype=np.int32)
    tracking_poses[reference_frame] = reference_pose
    fixed_rotation = np.asarray(reference_pose[:3, :3], dtype=np.float32).copy()
    segment_anchor_frames[reference_frame] = reference_frame

    def observed(frame: int) -> np.ndarray:
        if (
            mask_policy.occlusion_terminated[frame]
            or mask_policy.border_suppressed[frame]
        ):
            return np.empty((0, 3), dtype=np.float32)
        valid = (
            masks[frame]
            & np.isfinite(world_points[frame]).all(axis=-1)
            & np.isfinite(geometry_confidence[frame])
            & (geometry_confidence[frame] >= minimum_geometry_confidence)
        )
        observed_full = world_points[frame][valid]
        observed_indices = deterministic_subsample_indices(
            len(observed_full),
            maximum_points=maximum_observed_points,
            seed=frame,
        )
        points = observed_full[observed_indices]
        observed_counts[frame] = len(points)
        return points

    reference_observed = observed(reference_frame)
    reference_eligible, reference_visible_count, reference_other_occluded = (
        occlusion_aware_canonical_visibility(
            canonical_points=canonical,
            pose=reference_pose,
            camera_to_world=camera_poses[reference_frame],
            intrinsics=intrinsics[reference_frame],
            self_mask=masks[reference_frame],
            all_object_mask_union=all_object_mask_union[reference_frame],
        )
    )
    self_visible_canonical_counts[reference_frame] = reference_visible_count
    other_mask_occluded_canonical_counts[reference_frame] = (
        reference_other_occluded
    )
    occlusion_eligible_canonical_counts[reference_frame] = np.count_nonzero(
        reference_eligible
    )
    if len(reference_observed):
        reference_rotation = np.asarray(reference_pose[:3, :3], dtype=np.float64)
        reference_translation = np.asarray(reference_pose[:3, 3], dtype=np.float64)
        reference_canonical = (
            reference_observed - reference_translation
        ) @ reference_rotation
        reference_distance, reference_ids = _query_nearest_eligible_canonical(
            canonical_tree,
            reference_canonical,
            reference_eligible,
        )
        coverage[reference_frame] = np.mean(
            (reference_distance <= icp_distance)
            & reference_eligible[reference_ids]
        )
        has_sufficient_correspondences[reference_frame] = (
            len(reference_observed) >= 20
        )

    def advance(
        *,
        pose_anchor: int,
        flow_source: int,
        flow_target: int,
        target: int,
        source_to_target_uv: np.ndarray,
        source_tracking_confidence: np.ndarray,
        inverse_motion: bool,
    ) -> None:
        segment_anchor_frames[target] = pose_anchor
        if mask_policy.occlusion_terminated[target]:
            return
        fallback_pose = tracking_poses[pose_anchor]
        if mask_policy.border_suppressed[target]:
            tracking_poses[target] = fallback_pose
            return
        initial_pose = fallback_pose.copy()
        try:
            if not mask_policy.border_suppressed[flow_source] and not (
                mask_policy.border_suppressed[flow_target]
            ):
                motion = estimate_anchor_motion(
                    source_mask=masks[flow_source],
                    target_mask=masks[flow_target],
                    source_world_points=world_points[flow_source],
                    target_world_points=world_points[flow_target],
                    source_to_target_uv=source_to_target_uv,
                    source_tracking_confidence=source_tracking_confidence,
                    minimum_tracking_confidence=minimum_tracking_confidence,
                    inlier_distance=motion_distance,
                    seed=target,
                )
                relative_pose = motion.pose
                if inverse_motion:
                    relative_pose = np.linalg.inv(relative_pose)
                initial_pose = relative_pose @ tracking_poses[pose_anchor]
                if motion_model == "translation_only":
                    initial_pose[:3, :3] = fixed_rotation
                elif motion_model == "axisymmetric":
                    initial_pose[:3, :3] = transport_axisymmetric_rotation(
                        tracking_poses[pose_anchor, :3, :3],
                        initial_pose[:3, :3],
                        symmetry_axis,
                    )
                motion_initialized[target] = True
                motion_correspondence_counts[target] = motion.correspondence_count
                motion_inlier_counts[target] = motion.inlier_count
                motion_rmse[target] = motion.rmse
        except ValueError:
            pass
        points = observed(target)
        eligible, visible_count, other_occluded = (
            occlusion_aware_canonical_visibility(
                canonical_points=canonical,
                pose=initial_pose,
                camera_to_world=camera_poses[target],
                intrinsics=intrinsics[target],
                self_mask=masks[target],
                all_object_mask_union=all_object_mask_union[target],
            )
        )
        self_visible_canonical_counts[target] = visible_count
        other_mask_occluded_canonical_counts[target] = other_occluded
        occlusion_eligible_canonical_counts[target] = np.count_nonzero(eligible)
        try:
            refined = refine_pose_with_partial_icp(
                canonical,
                canonical_tree,
                np.where(
                    (
                        np.isfinite(world_points[target]).all(axis=-1)
                        & np.isfinite(geometry_confidence[target])
                        & (
                            geometry_confidence[target]
                            >= minimum_geometry_confidence
                        )
                    )[..., None],
                    world_points[target],
                    np.nan,
                ),
                masks[target],
                all_object_mask_union[target],
                camera_poses[target],
                intrinsics[target],
                initial_pose,
                motion_model=motion_model,
                canonical_symmetry_axis=symmetry_axis,
                coverage_distance=icp_distance,
            )
        except ValueError:
            tracking_poses[target] = initial_pose
            return
        tracking_poses[target] = refined.pose
        correspondence_counts[target] = refined.correspondence_count
        inlier_counts[target] = refined.inlier_count
        rmse[target] = refined.rmse
        coverage[target] = refined.observed_coverage
        numerically_converged[target] = refined.numerically_converged
        has_sufficient_correspondences[target] = (
            refined.has_sufficient_correspondences
        )

    for target in range(reference_frame + 1, frame_count):
        source = target - 1
        advance(
            pose_anchor=source,
            flow_source=source,
            flow_target=target,
            target=target,
            source_to_target_uv=adjacent.source_to_target_uv[source],
            source_tracking_confidence=adjacent.tracking_confidence[source],
            inverse_motion=False,
        )
    for target in range(reference_frame - 1, -1, -1):
        later = target + 1
        advance(
            pose_anchor=later,
            flow_source=target,
            flow_target=later,
            target=target,
            source_to_target_uv=adjacent.source_to_target_uv[target],
            source_tracking_confidence=adjacent.tracking_confidence[target],
            inverse_motion=True,
        )

    poses = tracking_poses.copy()
    support_constrained = np.zeros(frame_count, dtype=bool)
    support_constraint_rejected = np.zeros(frame_count, dtype=bool)
    support_penetration_depth = np.full(frame_count, np.nan, dtype=np.float32)
    support_ray_scale = np.full(frame_count, np.nan, dtype=np.float32)
    support_correction_norm = np.zeros(frame_count, dtype=np.float32)
    support_clearance_after = np.full(frame_count, np.nan, dtype=np.float32)
    if support_constraint is not None:
        for frame, tracking_pose in enumerate(tracking_poses):
            if not np.isfinite(tracking_pose).all():
                continue
            result = enforce_pointcloud_support_nonpenetration(
                canonical_points=canonical_full,
                pose=tracking_pose,
                camera_to_world=camera_poses[frame],
                constraint=support_constraint,
            )
            support_penetration_depth[frame] = result.penetration_depth
            support_clearance_after[frame] = result.clearance_after
            if not result.valid:
                support_constraint_rejected[frame] = True
                continue
            poses[frame] = result.pose
            if result.constrained:
                support_constrained[frame] = True
                support_ray_scale[frame] = result.ray_scale
                support_correction_norm[frame] = result.correction_norm
    poses[mask_policy.border_suppressed] = np.nan

    return RigidPoseTrajectory(
        canonical_tracking_points=canonical,
        canonical_tracking_indices=canonical_indices,
        tracking_poses=tracking_poses,
        poses=poses,
        mask_pixel_counts=mask_policy.mask_pixel_counts,
        occlusion_termination_frame=mask_policy.occlusion_termination_frame,
        occlusion_terminated=mask_policy.occlusion_terminated,
        border_touching=mask_policy.border_touching,
        nearest_nonborder_frames=mask_policy.nearest_nonborder_frames,
        border_area_ratios=mask_policy.border_area_ratios,
        border_suppressed=mask_policy.border_suppressed,
        observed_point_counts=observed_counts,
        correspondence_counts=correspondence_counts,
        inlier_counts=inlier_counts,
        rmse=rmse,
        observed_coverage=coverage,
        motion_initialized=motion_initialized,
        motion_correspondence_counts=motion_correspondence_counts,
        motion_inlier_counts=motion_inlier_counts,
        motion_rmse=motion_rmse,
        segment_anchor_frames=segment_anchor_frames,
        numerically_converged=numerically_converged,
        has_sufficient_correspondences=has_sufficient_correspondences,
        self_visible_canonical_counts=self_visible_canonical_counts,
        other_mask_occluded_canonical_counts=(
            other_mask_occluded_canonical_counts
        ),
        occlusion_eligible_canonical_counts=occlusion_eligible_canonical_counts,
        support_constrained=support_constrained,
        support_constraint_rejected=support_constraint_rejected,
        support_penetration_depth=support_penetration_depth,
        support_ray_scale=support_ray_scale,
        support_correction_norm=support_correction_norm,
        support_clearance_after=support_clearance_after,
    )
