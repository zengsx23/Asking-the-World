"""Warp-based planar rigid-body simulation and point-cloud system identification."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from atw.perception.pointcloud_reconstruction_assets import load_reconstruction_assets


GRAVITY_M_PER_S2 = 9.81
MIN_DIMENSION_M = 0.02
VELOCITY_BOUND_M_PER_S = 5.0
ANGULAR_VELOCITY_BOUND_RAD_PER_S = 20.0
LOG_MASS_BOUND = math.log(20.0)
MIN_COLLISION_SCALE = 0.85
MAX_COLLISION_SCALE = 1.25
MIN_INERTIA_SCALE = 0.25
MAX_INERTIA_SCALE = 10.0
MAX_ABS_CONTACT_TORQUE_TRANSFER = 32.0
MIN_FRICTION = 1.0e-5
MAX_FRICTION = 1.0
DEFAULT_RESTITUTION = 0.5
DEFAULT_CONTACT_FRICTION = 0.3
DEFAULT_SPIN_FRICTION = 0.01
ROTATION_IDENTIFIABILITY_THRESHOLD_RAD = math.radians(10.0)
CONTACT_GAP_TOLERANCE_M = 2.0e-6
CONTACT_RELEASE_GAP_M = 1.0e-3
CONTACT_SPEED_TOLERANCE_M_PER_S = 1.0e-7
TOI_ITERATIONS = 64
DEFAULT_PORTFOLIO_SEEDS = (7, 11, 23, 43, 101, 137)
DEFAULT_CUDA_BLOCK_DIM = 32
PROXY_CIRCLE = 0
PROXY_BOX = 1
PROXY_ROUNDED_BOX = 2
PROXY_MODES = (
    "circle",
    "shape-aware",
    "circle-shape-response",
    "circle-shape-physical-response",
    "rounded-box",
)
HYBRID_PROXY_MODES = (
    "circle-shape-response",
    "circle-shape-physical-response",
)
ROUNDED_BOX_CORNER_FRACTION = 0.25


@dataclass(frozen=True)
class ShapeSpec:
    geometry_type: str
    dimensions_m: tuple[float, float, float]


@dataclass(frozen=True)
class RotationalSceneData:
    object_ids: tuple[str, ...]
    frames: tuple[int, ...]
    fps: float
    target_xy: np.ndarray
    target_yaw: np.ndarray
    target_centers_world: np.ndarray
    target_rotations_world: np.ndarray
    observed_mask: np.ndarray
    first_offsets: tuple[int, ...]
    last_offsets: tuple[int, ...]
    initial_xy: np.ndarray
    initial_rotations_world: np.ndarray
    initial_heights: np.ndarray
    cloud_centroids_local: np.ndarray
    cloud_factors_local: np.ndarray
    shapes: tuple[ShapeSpec, ...]
    support_plane: dict[str, Any]
    support_origin_world: np.ndarray
    support_tangent_1_world: np.ndarray
    support_tangent_2_world: np.ndarray
    support_normal_world: np.ndarray
    trajectory_paths: tuple[str, ...]
    cloud_point_counts: tuple[int, ...]
    valid_frame_counts: tuple[int, ...]

    @property
    def object_count(self) -> int:
        return len(self.object_ids)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class RotationalCollisionEvent:
    time_s: float
    object_i: int
    object_j: int
    relative_normal_velocity_m_per_s: float
    relative_tangential_velocity_m_per_s: float
    normal_impulse_magnitude: float
    tangential_impulse_magnitude: float
    contact_friction: float
    contact_mode: str


@dataclass(frozen=True)
class CollisionProxyData:
    shape_types: np.ndarray
    half_extents: np.ndarray
    corner_radii: np.ndarray
    initial_angles: np.ndarray
    spin_radii: np.ndarray
    bounding_radii: np.ndarray
    inertia_per_mass: np.ndarray


@dataclass(frozen=True)
class ParameterLayout:
    object_count: int
    pairs: tuple[tuple[int, int], ...]
    velocity_slice: slice
    angular_velocity_slice: slice
    log_collision_scale_slice: slice
    log_mass_slice: slice
    log_inertia_scale_slice: slice
    restitution_slice: slice
    friction_slice: slice
    spin_friction_slice: slice
    contact_friction_slice: slice
    contact_torque_transfer_slice: slice
    size: int


@dataclass(frozen=True)
class CEMConfig:
    population: int = 1024
    elite_fraction: float = 0.08
    generations_per_prefix: int = 64
    prefix_frames: tuple[int, ...] = (10, 21, 32, 42, 53, 64, 74, 85, 96, 106, 117, 0)
    distribution_smoothing: float = 0.2
    exploration_fraction: float = 0.08
    huber_delta_m: float = 0.02
    rotation_loss_weight: float = 1.0
    yaw_phase_loss_weight: float = 4.0
    collision_scale_prior_weight: float = 3.0e-2
    contact_torque_transfer_prior_weight: float = 0.0
    parameter_evidence_threshold_m_per_s: float = 0.05
    fit_contact_torque_transfers: bool = True
    seed: int = 23


@dataclass(frozen=True)
class RefinementConfig:
    iterations: int = 20
    initial_damping: float = 1.0e-2
    step_limit: float = 1.5


def make_parameter_layout(object_count: int) -> ParameterLayout:
    if object_count <= 0:
        raise ValueError("object_count must be positive")
    pairs = tuple(
        (object_i, object_j)
        for object_i in range(object_count)
        for object_j in range(object_i + 1, object_count)
    )
    offset = 0
    velocity_slice = slice(offset, offset + 2 * object_count)
    offset = velocity_slice.stop
    angular_velocity_slice = slice(offset, offset + object_count)
    offset = angular_velocity_slice.stop
    log_collision_scale_slice = slice(offset, offset + object_count)
    offset = log_collision_scale_slice.stop
    log_mass_slice = slice(offset, offset + max(object_count - 1, 0))
    offset = log_mass_slice.stop
    log_inertia_scale_slice = slice(offset, offset + object_count)
    offset = log_inertia_scale_slice.stop
    restitution_slice = slice(offset, offset + len(pairs))
    offset = restitution_slice.stop
    friction_slice = slice(offset, offset + object_count)
    offset = friction_slice.stop
    spin_friction_slice = slice(offset, offset + object_count)
    offset = spin_friction_slice.stop
    contact_friction_slice = slice(offset, offset + len(pairs))
    offset = contact_friction_slice.stop
    contact_torque_transfer_slice = slice(offset, offset + object_count)
    offset = contact_torque_transfer_slice.stop
    return ParameterLayout(
        object_count=object_count,
        pairs=pairs,
        velocity_slice=velocity_slice,
        angular_velocity_slice=angular_velocity_slice,
        log_collision_scale_slice=log_collision_scale_slice,
        log_mass_slice=log_mass_slice,
        log_inertia_scale_slice=log_inertia_scale_slice,
        restitution_slice=restitution_slice,
        friction_slice=friction_slice,
        spin_friction_slice=spin_friction_slice,
        contact_friction_slice=contact_friction_slice,
        contact_torque_transfer_slice=contact_torque_transfer_slice,
        size=offset,
    )


def parameter_bounds(layout: ParameterLayout) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(layout.size, dtype=np.float32)
    upper = np.empty(layout.size, dtype=np.float32)
    lower[layout.velocity_slice] = -VELOCITY_BOUND_M_PER_S
    upper[layout.velocity_slice] = VELOCITY_BOUND_M_PER_S
    lower[layout.angular_velocity_slice] = -ANGULAR_VELOCITY_BOUND_RAD_PER_S
    upper[layout.angular_velocity_slice] = ANGULAR_VELOCITY_BOUND_RAD_PER_S
    lower[layout.log_collision_scale_slice] = math.log(MIN_COLLISION_SCALE)
    upper[layout.log_collision_scale_slice] = math.log(MAX_COLLISION_SCALE)
    lower[layout.log_mass_slice] = -LOG_MASS_BOUND
    upper[layout.log_mass_slice] = LOG_MASS_BOUND
    lower[layout.log_inertia_scale_slice] = math.log(MIN_INERTIA_SCALE)
    upper[layout.log_inertia_scale_slice] = math.log(MAX_INERTIA_SCALE)
    lower[layout.restitution_slice] = 0.0
    upper[layout.restitution_slice] = 1.0
    lower[layout.friction_slice] = MIN_FRICTION
    upper[layout.friction_slice] = MAX_FRICTION
    lower[layout.spin_friction_slice] = MIN_FRICTION
    upper[layout.spin_friction_slice] = MAX_FRICTION
    lower[layout.contact_friction_slice] = 0.0
    upper[layout.contact_friction_slice] = 1.0
    lower[layout.contact_torque_transfer_slice] = -MAX_ABS_CONTACT_TORQUE_TRANSFER
    upper[layout.contact_torque_transfer_slice] = MAX_ABS_CONTACT_TORQUE_TRANSFER
    return lower, upper


def initial_distribution(layout: ParameterLayout) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(layout.size, dtype=np.float32)
    std = np.zeros(layout.size, dtype=np.float32)
    std[layout.velocity_slice] = 0.75
    std[layout.angular_velocity_slice] = 2.0
    std[layout.log_collision_scale_slice] = 0.08
    std[layout.log_mass_slice] = 0.8
    std[layout.log_inertia_scale_slice] = 0.8
    mean[layout.restitution_slice] = DEFAULT_RESTITUTION
    std[layout.restitution_slice] = 0.22
    mean[layout.friction_slice] = 0.01
    std[layout.friction_slice] = 0.04
    mean[layout.spin_friction_slice] = DEFAULT_SPIN_FRICTION
    std[layout.spin_friction_slice] = 0.04
    mean[layout.contact_friction_slice] = DEFAULT_CONTACT_FRICTION
    std[layout.contact_friction_slice] = 0.2
    mean[layout.contact_torque_transfer_slice] = 1.0
    std[layout.contact_torque_transfer_slice] = 4.0
    return mean, std


def minimum_distribution_std(layout: ParameterLayout) -> np.ndarray:
    minimum = np.empty(layout.size, dtype=np.float32)
    minimum[layout.velocity_slice] = 0.008
    minimum[layout.angular_velocity_slice] = 0.02
    minimum[layout.log_collision_scale_slice] = 0.002
    minimum[layout.log_mass_slice] = 0.02
    minimum[layout.log_inertia_scale_slice] = 0.01
    minimum[layout.restitution_slice] = 0.008
    minimum[layout.friction_slice] = 0.001
    minimum[layout.spin_friction_slice] = 0.001
    minimum[layout.contact_friction_slice] = 0.002
    minimum[layout.contact_torque_transfer_slice] = 0.01
    return minimum


def finite_difference_steps(layout: ParameterLayout) -> np.ndarray:
    steps = np.empty(layout.size, dtype=np.float64)
    steps[layout.velocity_slice] = 2.0e-3
    steps[layout.angular_velocity_slice] = 2.0e-3
    steps[layout.log_collision_scale_slice] = 2.0e-3
    steps[layout.log_mass_slice] = 3.0e-3
    steps[layout.log_inertia_scale_slice] = 3.0e-3
    steps[layout.restitution_slice] = 2.0e-3
    steps[layout.friction_slice] = 2.0e-4
    steps[layout.spin_friction_slice] = 2.0e-4
    steps[layout.contact_friction_slice] = 2.0e-3
    steps[layout.contact_torque_transfer_slice] = 3.0e-3
    return steps


def project_parameters(values: np.ndarray, layout: ParameterLayout) -> np.ndarray:
    lower, upper = parameter_bounds(layout)
    return np.clip(np.asarray(values), lower, upper)


def decode_parameters(
    values: np.ndarray, layout: ParameterLayout
) -> dict[str, np.ndarray]:
    projected = project_parameters(values, layout).astype(np.float64)
    log_masses = np.zeros(layout.object_count, dtype=np.float64)
    if layout.object_count > 1:
        log_masses[:-1] = projected[layout.log_mass_slice]
        log_masses[-1] = -float(np.sum(log_masses[:-1]))
    return {
        "velocity": projected[layout.velocity_slice].reshape(layout.object_count, 2),
        "angular_velocity": projected[layout.angular_velocity_slice],
        "collision_scale": np.exp(projected[layout.log_collision_scale_slice]),
        "mass": np.exp(log_masses),
        "inertia_scale": np.exp(projected[layout.log_inertia_scale_slice]),
        "restitution": projected[layout.restitution_slice],
        "friction": projected[layout.friction_slice],
        "spin_friction": projected[layout.spin_friction_slice],
        "contact_friction": projected[layout.contact_friction_slice],
        "contact_torque_transfer": projected[layout.contact_torque_transfer_slice],
    }


def _best_planar_yaw(
    rotation: np.ndarray, initial_rotation: np.ndarray, normal: np.ndarray,
) -> float:
    normalized = np.asarray(normal, dtype=np.float64).reshape(3)
    normalized /= np.linalg.norm(normalized)
    x, y, z = normalized
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    projection = np.eye(3) - np.outer(normalized, normalized)
    relative = np.asarray(rotation) @ np.asarray(initial_rotation).T
    sine = float(np.trace(relative.T @ skew))
    cosine = float(np.trace(relative.T @ projection))
    return math.atan2(sine, cosine)


def _cloud_statistics(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("canonical tracking points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("canonical tracking points contain non-finite values")
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / float(len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    return centroid, factor


def _load_support_plane(reconstruction_dir: Path) -> dict[str, Any]:
    assets = load_reconstruction_assets(reconstruction_dir)
    gravity_path = assets.support_plane_gravity_dir / "gravity_estimate.json"
    gravity = json.loads(gravity_path.read_text(encoding="utf-8"))
    if gravity.get("status") != "completed":
        raise ValueError("support-plane gravity estimation did not complete")
    if gravity["scene_surface_type"] != "support_plane":
        raise ValueError("SWR requires a horizontal support plane, not a wall")
    plane = gravity["selected"]["support_plane"]
    normal = np.asarray(plane["normal_world"], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1.0e-8:
        raise ValueError("support-plane normal is invalid")
    normal /= normal_norm
    plane_offset = float(plane["offset_world"])
    if not math.isfinite(plane_offset):
        raise ValueError("support-plane offset is invalid")
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_1 = reference - np.dot(reference, normal) * normal
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    origin = normal * plane_offset
    return {
        "coordinate_frame": "track4world_unified_world",
        "source": str(gravity_path.resolve()),
        "origin": origin.tolist(),
        "normal": normal.tolist(),
        "tangent_1": tangent_1.tolist(),
        "tangent_2": tangent_2.tolist(),
        "plane_offset_world": plane_offset,
    }


def _load_shape_specs(reconstruction_dir: Path) -> dict[str, ShapeSpec]:
    assets = load_reconstruction_assets(reconstruction_dir)
    summary_path = assets.rigid_models_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {}
    for item in summary.get("objects", []):
        object_id = str(item["track_id"])
        geometry_type = str(item["rigid_shape"]).lower()
        if geometry_type not in {"box", "cylinder", "sphere", "irregular"}:
            raise ValueError(
                f"unsupported rigid shape for {object_id}: {geometry_type!r}"
            )
        dimensions = tuple(float(value) for value in item["canonical_extent"])
        if len(dimensions) != 3 or not all(
            math.isfinite(value) and value > 0.0 for value in dimensions
        ):
            raise ValueError(f"invalid canonical extent for {object_id}")
        result[object_id] = ShapeSpec(
            geometry_type=geometry_type,
            dimensions_m=tuple(max(value, MIN_DIMENSION_M) for value in dimensions),
        )
    if not result:
        raise ValueError(f"rigid model summary has no objects: {summary_path}")
    return result


def load_rotational_scene(
    pointcloud_dir: Path, *, reconstruction_dir: Path | None = None,
) -> tuple[RotationalSceneData, dict[str, Any]]:
    pointcloud_dir = pointcloud_dir.resolve()
    if not (pointcloud_dir / "COMPLETED").is_file():
        raise ValueError(f"point-cloud run is incomplete: {pointcloud_dir}")
    if reconstruction_dir is None:
        reconstruction_dir = pointcloud_dir.parent / "reconstruction"
    reconstruction_dir = reconstruction_dir.resolve()
    assets = load_reconstruction_assets(reconstruction_dir)
    run_metadata_path = pointcloud_dir / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    fps = float(run_metadata["fps"])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid point-cloud run fps: {fps!r}")

    trajectory_dir = pointcloud_dir / "rigid_trajectories"
    trajectory_paths = sorted(trajectory_dir.glob("*.npz"))
    if not trajectory_paths:
        raise ValueError(f"no rigid trajectories found in {trajectory_dir}")
    object_ids = tuple(path.stem for path in trajectory_paths)
    shape_specs = _load_shape_specs(reconstruction_dir)
    missing_shapes = sorted(set(object_ids) - set(shape_specs))
    if missing_shapes:
        raise ValueError(f"missing rigid shape metadata for: {missing_shapes}")

    frame_count: int | None = None
    archives = []
    for path in trajectory_paths:
        with np.load(path) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        poses = payload["poses"]
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"invalid pose array in {path}: {poses.shape}")
        if frame_count is None:
            frame_count = len(poses)
        elif len(poses) != frame_count:
            raise ValueError("rigid trajectory frame counts do not match")
        archives.append(payload)
    assert frame_count is not None
    frames = tuple(range(frame_count))
    object_count = len(object_ids)
    observed_mask = np.zeros((frame_count, object_count), dtype=np.float64)
    target_xy = np.zeros((frame_count, object_count, 2), dtype=np.float64)
    target_yaw = np.zeros((frame_count, object_count), dtype=np.float64)
    target_centers_world = np.zeros((frame_count, object_count, 3), dtype=np.float64)
    target_rotations_world = np.broadcast_to(
        np.eye(3, dtype=np.float64), (frame_count, object_count, 3, 3)
    ).copy()

    support = _load_support_plane(reconstruction_dir)
    origin = np.asarray(support["origin"], dtype=np.float64)
    tangent_1 = np.asarray(support["tangent_1"], dtype=np.float64)
    tangent_2 = np.asarray(support["tangent_2"], dtype=np.float64)
    normal = np.asarray(support["normal"], dtype=np.float64)
    initial_xy = []
    initial_rotations = []
    initial_heights = []
    cloud_centroids = []
    cloud_factors = []
    first_offsets = []
    last_offsets = []
    point_counts = []
    valid_frame_counts = []

    for object_index, (object_id, payload) in enumerate(zip(object_ids, archives)):
        points = np.asarray(payload["canonical_tracking_points"], dtype=np.float64)
        centroid, factor = _cloud_statistics(points)
        point_counts.append(len(points))
        cloud_centroids.append(centroid)
        cloud_factors.append(factor)

        poses = np.asarray(payload["poses"], dtype=np.float64)
        if "motion_model" in payload and str(payload["motion_model"]) == "static":
            # Frozen reference poses have visibility evidence but no temporal ICP.
            sufficient = (
                (np.asarray(payload["mask_pixel_counts"]) > 0)
                & (np.asarray(payload["occlusion_eligible_canonical_counts"]) > 0)
                & ~np.asarray(payload["occlusion_terminated"], dtype=bool)
            )
        else:
            sufficient = np.asarray(payload["has_sufficient_correspondences"], dtype=bool)
        border_suppressed = np.asarray(payload["border_suppressed"], dtype=bool)
        if sufficient.shape != (frame_count,) or border_suppressed.shape != (
            frame_count,
        ):
            raise ValueError(
                f"invalid frame quality arrays in {trajectory_paths[object_index]}"
            )
        valid = np.isfinite(poses).all(axis=(1, 2)) & sufficient & ~border_suppressed
        valid_offsets = np.flatnonzero(valid)
        if len(valid_offsets) < 2:
            raise ValueError(f"fewer than two usable poses for {object_id}")
        first_offset = int(valid_offsets[0])
        last_offset = int(valid_offsets[-1])
        first_offsets.append(first_offset)
        last_offsets.append(last_offset)
        valid_frame_counts.append(len(valid_offsets))
        observed_mask[valid_offsets, object_index] = 1.0

        initial_rotation = poses[first_offset, :3, :3]
        initial_rotations.append(initial_rotation)
        raw_yaws = []
        for offset in valid_offsets:
            world_pose = poses[offset]
            rotation = world_pose[:3, :3]
            center = rotation @ centroid + world_pose[:3, 3]
            target_centers_world[offset, object_index] = center
            target_rotations_world[offset, object_index] = rotation
            delta = center - origin
            target_xy[offset, object_index] = [
                float(np.dot(delta, tangent_1)),
                float(np.dot(delta, tangent_2)),
            ]
            raw_yaws.append(_best_planar_yaw(rotation, initial_rotation, normal))

        unwrapped_yaws = np.unwrap(np.asarray(raw_yaws, dtype=np.float64))
        target_yaw[valid_offsets, object_index] = unwrapped_yaws
        first_center = target_centers_world[first_offset, object_index]
        initial_xy.append(target_xy[first_offset, object_index])
        initial_heights.append(float(np.dot(first_center - origin, normal)))

    scene = RotationalSceneData(
        object_ids=object_ids,
        frames=frames,
        fps=fps,
        target_xy=target_xy,
        target_yaw=target_yaw,
        target_centers_world=target_centers_world,
        target_rotations_world=target_rotations_world,
        observed_mask=observed_mask,
        first_offsets=tuple(first_offsets),
        last_offsets=tuple(last_offsets),
        initial_xy=np.asarray(initial_xy, dtype=np.float64),
        initial_rotations_world=np.asarray(initial_rotations, dtype=np.float64),
        initial_heights=np.asarray(initial_heights, dtype=np.float64),
        cloud_centroids_local=np.asarray(cloud_centroids, dtype=np.float64),
        cloud_factors_local=np.asarray(cloud_factors, dtype=np.float64),
        shapes=tuple(shape_specs[object_id] for object_id in object_ids),
        support_plane=support,
        support_origin_world=origin,
        support_tangent_1_world=tangent_1,
        support_tangent_2_world=tangent_2,
        support_normal_world=normal,
        trajectory_paths=tuple(str(path.resolve()) for path in trajectory_paths),
        cloud_point_counts=tuple(point_counts),
        valid_frame_counts=tuple(valid_frame_counts),
    )
    for name in (
        "target_xy",
        "target_yaw",
        "target_centers_world",
        "target_rotations_world",
        "initial_xy",
        "initial_rotations_world",
        "cloud_factors_local",
    ):
        if not np.isfinite(getattr(scene, name)).all():
            raise ValueError(f"{name} contains non-finite values")
    source = {
        "pointcloud_dir": str(pointcloud_dir),
        "run_metadata": str(run_metadata_path.resolve()),
        "reconstruction_manifest": str(assets.manifest_path),
    }
    return scene, source


def collision_proxy_data(
    scene: RotationalSceneData, *, proxy_mode: str = "shape-aware",
) -> CollisionProxyData:
    if proxy_mode not in PROXY_MODES:
        raise ValueError(f"unsupported collision proxy mode: {proxy_mode!r}")
    shape_types = []
    half_extents = []
    corner_radii = []
    initial_angles = []
    spin_radii = []
    bounding_radii = []
    inertia_per_mass = []
    normal = scene.support_normal_world

    for object_index, shape in enumerate(scene.shapes):
        rotation = scene.initial_rotations_world[object_index]
        vertical_axis = int(np.argmax(np.abs(rotation.T @ normal)))
        planar_axes = [axis for axis in range(3) if axis != vertical_axis]
        planar_dimensions = [shape.dimensions_m[axis] for axis in planar_axes]
        box_like = shape.geometry_type in {"box", "irregular"}
        if box_like and proxy_mode in {"shape-aware", "rounded-box"}:
            primary_world = rotation[:, planar_axes[0]].copy()
            primary_world -= np.dot(primary_world, normal) * normal
            primary_world /= np.linalg.norm(primary_world)
            initial_angle = math.atan2(
                float(np.dot(primary_world, scene.support_tangent_2_world)),
                float(np.dot(primary_world, scene.support_tangent_1_world)),
            )
            half_width = 0.5 * planar_dimensions[0]
            half_depth = 0.5 * planar_dimensions[1]
            if proxy_mode == "rounded-box":
                corner_radius = ROUNDED_BOX_CORNER_FRACTION * min(
                    2.0 * half_width, 2.0 * half_depth
                )
                shape_types.append(PROXY_ROUNDED_BOX)
                half_extents.append(
                    (half_width - corner_radius, half_depth - corner_radius)
                )
                corner_radii.append(corner_radius)
            else:
                shape_types.append(PROXY_BOX)
                half_extents.append((half_width, half_depth))
                corner_radii.append(0.0)
            initial_angles.append(initial_angle)
            spin_radii.append(
                math.sqrt((2.0 * half_width) * (2.0 * half_depth) / math.pi)
            )
            bounding_radii.append(math.hypot(half_width, half_depth))
            inertia_per_mass.append(
                ((2.0 * half_width) ** 2 + (2.0 * half_depth) ** 2) / 12.0
            )
        else:
            radius = (
                math.sqrt(planar_dimensions[0] * planar_dimensions[1] / math.pi)
                if box_like
                else 0.5 * max(planar_dimensions)
            )
            shape_types.append(PROXY_CIRCLE)
            half_extents.append((radius, radius))
            corner_radii.append(0.0)
            initial_angles.append(0.0)
            spin_radii.append(radius)
            bounding_radii.append(radius)
            inertia_per_mass.append(0.5 * radius * radius)

    return CollisionProxyData(
        shape_types=np.asarray(shape_types, dtype=np.int32),
        half_extents=np.asarray(half_extents, dtype=np.float32),
        corner_radii=np.asarray(corner_radii, dtype=np.float32),
        initial_angles=np.asarray(initial_angles, dtype=np.float32),
        spin_radii=np.asarray(spin_radii, dtype=np.float32),
        bounding_radii=np.asarray(bounding_radii, dtype=np.float32),
        inertia_per_mass=np.asarray(inertia_per_mass, dtype=np.float32),
    )


def pair_interaction_evidence(
    scene: RotationalSceneData, *, proximity_margin_m: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate pair interaction evidence from opposed target velocity changes."""
    layout = make_parameter_layout(scene.object_count)
    proxy = collision_proxy_data(scene, proxy_mode="circle")
    velocity = np.full_like(scene.target_xy, np.nan)
    velocity_valid = np.zeros_like(scene.observed_mask, dtype=bool)
    observed = scene.observed_mask > 0.0
    for frame_index in range(1, scene.frame_count - 1):
        valid = observed[frame_index - 1] & observed[frame_index + 1]
        velocity[frame_index, valid] = (
            scene.target_xy[frame_index + 1, valid]
            - scene.target_xy[frame_index - 1, valid]
        ) * (0.5 * scene.fps)
        velocity_valid[frame_index, valid] = True

    evidence = np.zeros(len(layout.pairs), dtype=np.float32)
    evidence_frames = np.full(len(layout.pairs), -1, dtype=np.int32)
    for pair_index, (object_i, object_j) in enumerate(layout.pairs):
        proximity = (
            proxy.bounding_radii[object_i]
            + proxy.bounding_radii[object_j]
            + proximity_margin_m
        )
        for frame_index in range(3, scene.frame_count - 3):
            if not (
                velocity_valid[frame_index - 2, object_i]
                and velocity_valid[frame_index + 2, object_i]
                and velocity_valid[frame_index - 2, object_j]
                and velocity_valid[frame_index + 2, object_j]
            ):
                continue
            separation = np.linalg.norm(
                scene.target_xy[frame_index, object_i]
                - scene.target_xy[frame_index, object_j]
            )
            if separation > proximity:
                continue
            delta_i = (
                velocity[frame_index + 2, object_i]
                - velocity[frame_index - 2, object_i]
            )
            delta_j = (
                velocity[frame_index + 2, object_j]
                - velocity[frame_index - 2, object_j]
            )
            magnitude_i = float(np.linalg.norm(delta_i))
            magnitude_j = float(np.linalg.norm(delta_j))
            alignment = max(
                0.0,
                -float(np.dot(delta_i, delta_j))
                / max(magnitude_i * magnitude_j, 1.0e-12),
            )
            score = min(magnitude_i, magnitude_j) * alignment
            if score > evidence[pair_index]:
                evidence[pair_index] = score
                evidence_frames[pair_index] = frame_index
    return evidence, evidence_frames


def optimization_active_mask(
    scene: RotationalSceneData,
    layout: ParameterLayout,
    interaction_evidence: np.ndarray,
    evidence_threshold_m_per_s: float,
    *,
    fit_contact_torque_transfers: bool = False,
) -> np.ndarray:
    """Select parameters that the observed motion can reasonably identify."""
    active = np.zeros(layout.size, dtype=bool)
    active[layout.velocity_slice] = True
    active[layout.friction_slice] = True

    rotational_objects = rotational_object_mask(scene)
    active[layout.angular_velocity_slice] = rotational_objects
    active[layout.spin_friction_slice] = rotational_objects

    active_pairs = np.asarray(interaction_evidence) >= evidence_threshold_m_per_s
    active[layout.restitution_slice] = active_pairs
    active[layout.contact_friction_slice] = active_pairs
    interacting_objects = np.zeros(scene.object_count, dtype=bool)
    for pair_index, (object_i, object_j) in enumerate(layout.pairs):
        if active_pairs[pair_index]:
            interacting_objects[object_i] = True
            interacting_objects[object_j] = True
    active[layout.log_collision_scale_slice] = interacting_objects
    active[layout.log_inertia_scale_slice] = rotational_objects & interacting_objects
    if fit_contact_torque_transfers:
        torque_transfer_objects = np.zeros(layout.object_count, dtype=bool)
        for pair_index, (object_i, object_j) in enumerate(layout.pairs):
            if active_pairs[pair_index] and (
                rotational_objects[object_i] or rotational_objects[object_j]
            ):
                torque_transfer_objects[object_i] = True
                torque_transfer_objects[object_j] = True
        active[layout.contact_torque_transfer_slice] = torque_transfer_objects
    if np.any(active_pairs):
        active[layout.log_mass_slice] = True
    return active


def rotational_object_mask(scene: RotationalSceneData) -> np.ndarray:
    """Return the objects whose planar orientation is physically observable."""
    return np.asarray(
        [shape.geometry_type in {"box", "irregular"} for shape in scene.shapes],
        dtype=bool,
    )


@wp.func
def _state_position(
    start_position: wp.vec2,
    start_velocity: wp.vec2,
    state_time: float,
    friction: float,
    query_time: float,
) -> wp.vec2:
    dt = wp.max(query_time - state_time, 0.0)
    speed = wp.length(start_velocity)
    acceleration = friction * float(GRAVITY_M_PER_S2)
    if speed <= 1.0e-12:
        return start_position
    if acceleration <= 1.0e-12:
        return start_position + start_velocity * dt
    moving_time = wp.min(dt, speed / acceleration)
    distance = speed * moving_time - 0.5 * acceleration * moving_time * moving_time
    return start_position + start_velocity * (distance / speed)


@wp.func
def _state_velocity(
    start_velocity: wp.vec2, state_time: float, friction: float, query_time: float,
) -> wp.vec2:
    dt = wp.max(query_time - state_time, 0.0)
    speed = wp.length(start_velocity)
    acceleration = friction * float(GRAVITY_M_PER_S2)
    if speed <= 1.0e-12:
        return wp.vec2(0.0, 0.0)
    if acceleration <= 1.0e-12:
        return start_velocity
    remaining_speed = wp.max(speed - acceleration * dt, 0.0)
    return start_velocity * (remaining_speed / speed)


@wp.func
def _angular_deceleration(friction: float, radius: float) -> float:
    safe_radius = wp.max(radius, 1.0e-8)
    return (4.0 / 3.0) * friction * float(GRAVITY_M_PER_S2) / safe_radius


@wp.func
def _state_angle(
    start_angle: float,
    start_angular_velocity: float,
    state_time: float,
    friction: float,
    radius: float,
    query_time: float,
) -> float:
    dt = wp.max(query_time - state_time, 0.0)
    speed = wp.abs(start_angular_velocity)
    if speed <= 1.0e-12:
        return start_angle
    acceleration = _angular_deceleration(friction, radius)
    if acceleration <= 1.0e-12:
        return start_angle + start_angular_velocity * dt
    moving_time = wp.min(dt, speed / acceleration)
    distance = speed * moving_time - 0.5 * acceleration * moving_time * moving_time
    return start_angle + wp.sign(start_angular_velocity) * distance


@wp.func
def _state_angular_velocity(
    start_angular_velocity: float,
    state_time: float,
    friction: float,
    radius: float,
    query_time: float,
) -> float:
    dt = wp.max(query_time - state_time, 0.0)
    speed = wp.abs(start_angular_velocity)
    if speed <= 1.0e-12:
        return 0.0
    acceleration = _angular_deceleration(friction, radius)
    if acceleration <= 1.0e-12:
        return start_angular_velocity
    remaining_speed = wp.max(speed - acceleration * dt, 0.0)
    return wp.sign(start_angular_velocity) * remaining_speed


@wp.func
def _cross_2d(first: wp.vec2, second: wp.vec2) -> float:
    return first[0] * second[1] - first[1] * second[0]


@wp.func
def _point_velocity(
    linear_velocity: wp.vec2, angular_velocity: float, lever: wp.vec2,
) -> wp.vec2:
    return linear_velocity + wp.vec2(
        -angular_velocity * lever[1], angular_velocity * lever[0],
    )


@wp.func
def _box_axis_x(angle: float) -> wp.vec2:
    return wp.vec2(wp.cos(angle), wp.sin(angle))


@wp.func
def _box_axis_y(angle: float) -> wp.vec2:
    return wp.vec2(-wp.sin(angle), wp.cos(angle))


@wp.func
def _box_support_radius(
    half_extents: wp.vec2, angle: float, direction: wp.vec2,
) -> float:
    return half_extents[0] * wp.abs(
        wp.dot(_box_axis_x(angle), direction)
    ) + half_extents[1] * wp.abs(wp.dot(_box_axis_y(angle), direction))


@wp.func
def _shape_support_radius(
    shape_type: int,
    half_extents: wp.vec2,
    corner_radius: float,
    angle: float,
    direction: wp.vec2,
) -> float:
    if shape_type == int(PROXY_BOX) or shape_type == int(PROXY_ROUNDED_BOX):
        return _box_support_radius(half_extents, angle, direction) + corner_radius
    return half_extents[0]


@wp.func
def _circle_circle_geometry(
    position_i: wp.vec2, radius_i: float, position_j: wp.vec2, radius_j: float,
) -> wp.vec3:
    relative_position = position_i - position_j
    distance = wp.length(relative_position)
    if distance <= 1.0e-12:
        return wp.vec3(1.0, 0.0, -(radius_i + radius_j))
    normal = relative_position / distance
    return wp.vec3(normal[0], normal[1], distance - radius_i - radius_j)


@wp.func
def _circle_box_geometry(
    circle_position: wp.vec2,
    circle_radius: float,
    box_position: wp.vec2,
    box_half_extents: wp.vec2,
    box_angle: float,
) -> wp.vec3:
    axis_x = _box_axis_x(box_angle)
    axis_y = _box_axis_y(box_angle)
    relative = circle_position - box_position
    local_x = wp.dot(relative, axis_x)
    local_y = wp.dot(relative, axis_y)
    closest_x = wp.clamp(local_x, -box_half_extents[0], box_half_extents[0])
    closest_y = wp.clamp(local_y, -box_half_extents[1], box_half_extents[1])
    closest = box_position + closest_x * axis_x + closest_y * axis_y
    delta = circle_position - closest
    distance = wp.length(delta)
    if distance > 1.0e-12:
        normal = delta / distance
        return wp.vec3(normal[0], normal[1], distance - circle_radius)

    distance_x = box_half_extents[0] - wp.abs(local_x)
    distance_y = box_half_extents[1] - wp.abs(local_y)
    normal = axis_x
    penetration = distance_x + circle_radius
    if local_x < 0.0:
        normal = -axis_x
    if distance_y < distance_x:
        normal = axis_y
        penetration = distance_y + circle_radius
        if local_y < 0.0:
            normal = -axis_y
    return wp.vec3(normal[0], normal[1], -penetration)


@wp.func
def _box_axis_geometry(
    position_i: wp.vec2,
    half_extents_i: wp.vec2,
    angle_i: float,
    position_j: wp.vec2,
    half_extents_j: wp.vec2,
    angle_j: float,
    axis: wp.vec2,
) -> wp.vec3:
    signed_distance = wp.dot(position_i - position_j, axis)
    separation = (
        wp.abs(signed_distance)
        - _box_support_radius(half_extents_i, angle_i, axis)
        - _box_support_radius(half_extents_j, angle_j, axis)
    )
    normal = axis
    if signed_distance < 0.0:
        normal = -axis
    return wp.vec3(normal[0], normal[1], separation)


@wp.func
def _box_box_geometry(
    position_i: wp.vec2,
    half_extents_i: wp.vec2,
    angle_i: float,
    position_j: wp.vec2,
    half_extents_j: wp.vec2,
    angle_j: float,
) -> wp.vec3:
    best = _box_axis_geometry(
        position_i,
        half_extents_i,
        angle_i,
        position_j,
        half_extents_j,
        angle_j,
        _box_axis_x(angle_i),
    )
    candidate = _box_axis_geometry(
        position_i,
        half_extents_i,
        angle_i,
        position_j,
        half_extents_j,
        angle_j,
        _box_axis_y(angle_i),
    )
    if candidate[2] > best[2]:
        best = candidate
    candidate = _box_axis_geometry(
        position_i,
        half_extents_i,
        angle_i,
        position_j,
        half_extents_j,
        angle_j,
        _box_axis_x(angle_j),
    )
    if candidate[2] > best[2]:
        best = candidate
    candidate = _box_axis_geometry(
        position_i,
        half_extents_i,
        angle_i,
        position_j,
        half_extents_j,
        angle_j,
        _box_axis_y(angle_j),
    )
    if candidate[2] > best[2]:
        best = candidate
    return best


@wp.func
def _contact_geometry(
    shape_type_i: int,
    position_i: wp.vec2,
    half_extents_i: wp.vec2,
    corner_radius_i: float,
    angle_i: float,
    shape_type_j: int,
    position_j: wp.vec2,
    half_extents_j: wp.vec2,
    corner_radius_j: float,
    angle_j: float,
) -> wp.vec3:
    if shape_type_i == int(PROXY_CIRCLE) and shape_type_j == int(PROXY_CIRCLE):
        return _circle_circle_geometry(
            position_i, half_extents_i[0], position_j, half_extents_j[0],
        )
    if shape_type_i == int(PROXY_CIRCLE):
        return _circle_box_geometry(
            position_i,
            half_extents_i[0] + corner_radius_j,
            position_j,
            half_extents_j,
            angle_j,
        )
    if shape_type_j == int(PROXY_CIRCLE):
        geometry = _circle_box_geometry(
            position_j,
            half_extents_j[0] + corner_radius_i,
            position_i,
            half_extents_i,
            angle_i,
        )
        return wp.vec3(-geometry[0], -geometry[1], geometry[2])
    geometry = _box_box_geometry(
        position_i, half_extents_i, angle_i, position_j, half_extents_j, angle_j,
    )
    return wp.vec3(
        geometry[0], geometry[1], geometry[2] - corner_radius_i - corner_radius_j
    )


@wp.func
def _contact_point(
    shape_type_i: int,
    position_i: wp.vec2,
    half_extents_i: wp.vec2,
    corner_radius_i: float,
    angle_i: float,
    shape_type_j: int,
    position_j: wp.vec2,
    half_extents_j: wp.vec2,
    corner_radius_j: float,
    angle_j: float,
    normal: wp.vec2,
) -> wp.vec2:
    tangent = wp.vec2(-normal[1], normal[0])
    normal_i = wp.dot(position_i, normal) - _shape_support_radius(
        shape_type_i, half_extents_i, corner_radius_i, angle_i, normal
    )
    normal_j = wp.dot(position_j, normal) + _shape_support_radius(
        shape_type_j, half_extents_j, corner_radius_j, angle_j, normal
    )
    normal_coordinate = 0.5 * (normal_i + normal_j)

    center_tangent_i = wp.dot(position_i, tangent)
    center_tangent_j = wp.dot(position_j, tangent)
    radius_tangent_i = _shape_support_radius(
        shape_type_i, half_extents_i, corner_radius_i, angle_i, tangent
    )
    radius_tangent_j = _shape_support_radius(
        shape_type_j, half_extents_j, corner_radius_j, angle_j, tangent
    )
    tangent_coordinate = 0.5 * (center_tangent_i + center_tangent_j)
    if shape_type_i == int(PROXY_CIRCLE) and shape_type_j != int(PROXY_CIRCLE):
        tangent_coordinate = wp.clamp(
            center_tangent_i,
            center_tangent_j - radius_tangent_j,
            center_tangent_j + radius_tangent_j,
        )
    elif shape_type_i != int(PROXY_CIRCLE) and shape_type_j == int(PROXY_CIRCLE):
        tangent_coordinate = wp.clamp(
            center_tangent_j,
            center_tangent_i - radius_tangent_i,
            center_tangent_i + radius_tangent_i,
        )
    elif shape_type_i != int(PROXY_CIRCLE) and shape_type_j != int(PROXY_CIRCLE):
        overlap_lower = wp.max(
            center_tangent_i - radius_tangent_i, center_tangent_j - radius_tangent_j,
        )
        overlap_upper = wp.min(
            center_tangent_i + radius_tangent_i, center_tangent_j + radius_tangent_j,
        )
        tangent_coordinate = 0.5 * (overlap_lower + overlap_upper)
    return normal_coordinate * normal + tangent_coordinate * tangent


@wp.func
def _pair_toi(
    position_i: wp.vec2,
    velocity_i: wp.vec2,
    angle_i: float,
    angular_velocity_i: float,
    state_time_i: float,
    friction_i: float,
    spin_friction_i: float,
    spin_radius_i: float,
    bounding_radius_i: float,
    shape_type_i: int,
    half_extents_i: wp.vec2,
    corner_radius_i: float,
    proxy_angle_i: float,
    position_j: wp.vec2,
    velocity_j: wp.vec2,
    angle_j: float,
    angular_velocity_j: float,
    state_time_j: float,
    friction_j: float,
    spin_friction_j: float,
    spin_radius_j: float,
    bounding_radius_j: float,
    shape_type_j: int,
    half_extents_j: wp.vec2,
    corner_radius_j: float,
    proxy_angle_j: float,
    lower: float,
    upper: float,
    contact_slack: float,
    contact_margin: float,
) -> float:
    current_time = lower
    for _iteration in range(TOI_ITERATIONS):
        query_position_i = _state_position(
            position_i, velocity_i, state_time_i, friction_i, current_time
        )
        query_position_j = _state_position(
            position_j, velocity_j, state_time_j, friction_j, current_time
        )
        query_velocity_i = _state_velocity(
            velocity_i, state_time_i, friction_i, current_time
        )
        query_velocity_j = _state_velocity(
            velocity_j, state_time_j, friction_j, current_time
        )
        query_angle_i = _state_angle(
            angle_i,
            angular_velocity_i,
            state_time_i,
            spin_friction_i,
            spin_radius_i,
            current_time,
        )
        query_angle_j = _state_angle(
            angle_j,
            angular_velocity_j,
            state_time_j,
            spin_friction_j,
            spin_radius_j,
            current_time,
        )
        query_angular_velocity_i = _state_angular_velocity(
            angular_velocity_i,
            state_time_i,
            spin_friction_i,
            spin_radius_i,
            current_time,
        )
        query_angular_velocity_j = _state_angular_velocity(
            angular_velocity_j,
            state_time_j,
            spin_friction_j,
            spin_radius_j,
            current_time,
        )
        geometry = _contact_geometry(
            shape_type_i,
            query_position_i,
            half_extents_i,
            corner_radius_i,
            query_angle_i + proxy_angle_i,
            shape_type_j,
            query_position_j,
            half_extents_j,
            corner_radius_j,
            query_angle_j + proxy_angle_j,
        )
        normal = wp.vec2(geometry[0], geometry[1])
        contact_point = _contact_point(
            shape_type_i,
            query_position_i,
            half_extents_i,
            corner_radius_i,
            query_angle_i + proxy_angle_i,
            shape_type_j,
            query_position_j,
            half_extents_j,
            corner_radius_j,
            query_angle_j + proxy_angle_j,
            normal,
        )
        relative_contact_velocity = _point_velocity(
            query_velocity_i,
            query_angular_velocity_i,
            contact_point - query_position_i,
        ) - _point_velocity(
            query_velocity_j,
            query_angular_velocity_j,
            contact_point - query_position_j,
        )
        relative_normal_speed = wp.dot(relative_contact_velocity, normal)
        gap = geometry[2] + contact_slack - contact_margin
        if gap <= float(CONTACT_GAP_TOLERANCE_M):
            if relative_normal_speed < -float(CONTACT_SPEED_TOLERANCE_M_PER_S):
                return current_time
            if relative_normal_speed <= float(CONTACT_SPEED_TOLERANCE_M_PER_S):
                return -1.0
            release_step = (float(CONTACT_RELEASE_GAP_M) - gap) / relative_normal_speed
            current_time = wp.min(
                current_time + wp.max(1.1 * release_step, 1.0e-6), upper
            )
            if current_time >= upper - 5.0e-10:
                return -1.0
            continue
        speed_bound = wp.length(query_velocity_i) + wp.length(query_velocity_j)
        if shape_type_i == int(PROXY_BOX):
            speed_bound += wp.abs(query_angular_velocity_i) * bounding_radius_i
        if shape_type_j == int(PROXY_BOX):
            speed_bound += wp.abs(query_angular_velocity_j) * bounding_radius_j
        if speed_bound <= 1.0e-12:
            return -1.0
        step = 0.8 * gap / speed_bound
        next_time = wp.min(current_time + wp.max(step, 1.0e-9), upper)
        if next_time <= current_time + 5.0e-10:
            return -1.0
        current_time = next_time
        if current_time >= upper - 5.0e-10:
            end_position_i = _state_position(
                position_i, velocity_i, state_time_i, friction_i, upper
            )
            end_position_j = _state_position(
                position_j, velocity_j, state_time_j, friction_j, upper
            )
            end_velocity_i = _state_velocity(
                velocity_i, state_time_i, friction_i, upper
            )
            end_velocity_j = _state_velocity(
                velocity_j, state_time_j, friction_j, upper
            )
            end_angle_i = _state_angle(
                angle_i,
                angular_velocity_i,
                state_time_i,
                spin_friction_i,
                spin_radius_i,
                upper,
            )
            end_angle_j = _state_angle(
                angle_j,
                angular_velocity_j,
                state_time_j,
                spin_friction_j,
                spin_radius_j,
                upper,
            )
            end_angular_velocity_i = _state_angular_velocity(
                angular_velocity_i, state_time_i, spin_friction_i, spin_radius_i, upper,
            )
            end_angular_velocity_j = _state_angular_velocity(
                angular_velocity_j, state_time_j, spin_friction_j, spin_radius_j, upper,
            )
            end_geometry = _contact_geometry(
                shape_type_i,
                end_position_i,
                half_extents_i,
                corner_radius_i,
                end_angle_i + proxy_angle_i,
                shape_type_j,
                end_position_j,
                half_extents_j,
                corner_radius_j,
                end_angle_j + proxy_angle_j,
            )
            end_normal = wp.vec2(end_geometry[0], end_geometry[1])
            end_contact_point = _contact_point(
                shape_type_i,
                end_position_i,
                half_extents_i,
                corner_radius_i,
                end_angle_i + proxy_angle_i,
                shape_type_j,
                end_position_j,
                half_extents_j,
                corner_radius_j,
                end_angle_j + proxy_angle_j,
                end_normal,
            )
            end_relative_velocity = _point_velocity(
                end_velocity_i,
                end_angular_velocity_i,
                end_contact_point - end_position_i,
            ) - _point_velocity(
                end_velocity_j,
                end_angular_velocity_j,
                end_contact_point - end_position_j,
            )
            if end_geometry[2] + contact_slack - contact_margin <= float(
                CONTACT_GAP_TOLERANCE_M
            ) and wp.dot(end_relative_velocity, end_normal) < -float(
                CONTACT_SPEED_TOLERANCE_M_PER_S
            ):
                return upper
            return -1.0
    return -1.0


@wp.func
def _relative_mass(
    theta: wp.array2d(dtype=float),
    world: int,
    object_index: int,
    object_count: int,
    log_mass_start: int,
) -> float:
    if object_count <= 1:
        return 1.0
    if object_index < object_count - 1:
        return wp.exp(theta[world, log_mass_start + object_index])
    last_log_mass = float(0.0)
    for index in range(object_count - 1):
        last_log_mass -= theta[world, log_mass_start + index]
    return wp.exp(last_log_mass)


@wp.kernel
def _simulate_worlds(
    theta: wp.array2d(dtype=float),
    initial_xy: wp.array(dtype=wp.vec2),
    initial_angle: wp.array(dtype=float),
    radii: wp.array(dtype=float),
    bounding_radii: wp.array(dtype=float),
    shape_types: wp.array(dtype=int),
    half_extents: wp.array(dtype=wp.vec2),
    corner_radii: wp.array(dtype=float),
    proxy_initial_angles: wp.array(dtype=float),
    response_shape_types: wp.array(dtype=int),
    response_half_extents: wp.array(dtype=wp.vec2),
    response_corner_radii: wp.array(dtype=float),
    response_initial_angles: wp.array(dtype=float),
    inertia_per_mass: wp.array(dtype=float),
    first_times: wp.array(dtype=float),
    end_times: wp.array(dtype=float),
    pair_i: wp.array(dtype=int),
    pair_j: wp.array(dtype=int),
    object_count: int,
    frame_count: int,
    pair_count: int,
    fps: float,
    simulation_end_time: float,
    event_cap: int,
    collision_cooldown: float,
    contact_slack: float,
    contact_margin: float,
    use_response_normal: int,
    angular_velocity_start: int,
    log_collision_scale_start: int,
    log_mass_start: int,
    log_inertia_scale_start: int,
    restitution_start: int,
    friction_start: int,
    spin_friction_start: int,
    contact_friction_start: int,
    contact_torque_transfer_start: int,
    start_position: wp.array2d(dtype=wp.vec2),
    start_velocity: wp.array2d(dtype=wp.vec2),
    start_angle: wp.array2d(dtype=float),
    start_angular_velocity: wp.array2d(dtype=float),
    state_time: wp.array2d(dtype=float),
    last_collision_time: wp.array2d(dtype=float),
    predicted: wp.array3d(dtype=wp.vec3),
    event_count_out: wp.array(dtype=int),
    event_time_out: wp.array2d(dtype=float),
    event_i_out: wp.array2d(dtype=int),
    event_j_out: wp.array2d(dtype=int),
    event_relative_normal_speed_out: wp.array2d(dtype=float),
    event_relative_tangential_speed_out: wp.array2d(dtype=float),
    event_normal_impulse_out: wp.array2d(dtype=float),
    event_tangential_impulse_out: wp.array2d(dtype=float),
):
    world = wp.tid()
    for object_index in range(object_count):
        velocity_offset = 2 * object_index
        object_position = initial_xy[object_index]
        object_velocity = wp.vec2(
            theta[world, velocity_offset], theta[world, velocity_offset + 1]
        )
        object_angle = initial_angle[object_index]
        object_angular_velocity = theta[world, angular_velocity_start + object_index]
        object_state_time = first_times[object_index]
        friction = theta[world, friction_start + object_index]
        spin_friction = theta[world, spin_friction_start + object_index]
        start_position[world, object_index] = object_position
        start_velocity[world, object_index] = object_velocity
        start_angle[world, object_index] = object_angle
        start_angular_velocity[world, object_index] = object_angular_velocity
        state_time[world, object_index] = object_state_time
        for frame_index in range(frame_count):
            query_time = float(frame_index) / fps
            query_position = _state_position(
                object_position,
                object_velocity,
                object_state_time,
                friction,
                query_time,
            )
            query_angle = _state_angle(
                object_angle,
                object_angular_velocity,
                object_state_time,
                spin_friction,
                radii[object_index],
                query_time,
            )
            predicted[world, frame_index, object_index] = wp.vec3(
                query_position[0], query_position[1], query_angle
            )
    for pair_index in range(pair_count):
        last_collision_time[world, pair_index] = -1.0e20
    event_count_out[world] = 0

    for event_index in range(event_cap):
        best_time = float(simulation_end_time + 1.0)
        best_pair = int(-1)
        for pair_index in range(pair_count):
            object_i = pair_i[pair_index]
            object_j = pair_j[pair_index]
            lower = wp.max(state_time[world, object_i], state_time[world, object_j])
            lower = wp.max(lower, first_times[object_i])
            lower = wp.max(lower, first_times[object_j])
            lower = wp.max(
                lower, last_collision_time[world, pair_index] + collision_cooldown,
            )
            lower += 1.0e-8
            upper = wp.min(simulation_end_time, end_times[object_i])
            upper = wp.min(upper, end_times[object_j])
            if upper <= lower + 1.0e-8:
                continue
            collision_scale_i = wp.exp(
                theta[world, log_collision_scale_start + object_i]
            )
            collision_scale_j = wp.exp(
                theta[world, log_collision_scale_start + object_j]
            )
            candidate_time = _pair_toi(
                start_position[world, object_i],
                start_velocity[world, object_i],
                start_angle[world, object_i],
                start_angular_velocity[world, object_i],
                state_time[world, object_i],
                theta[world, friction_start + object_i],
                theta[world, spin_friction_start + object_i],
                radii[object_i],
                collision_scale_i * bounding_radii[object_i],
                shape_types[object_i],
                collision_scale_i * half_extents[object_i],
                collision_scale_i * corner_radii[object_i],
                proxy_initial_angles[object_i],
                start_position[world, object_j],
                start_velocity[world, object_j],
                start_angle[world, object_j],
                start_angular_velocity[world, object_j],
                state_time[world, object_j],
                theta[world, friction_start + object_j],
                theta[world, spin_friction_start + object_j],
                radii[object_j],
                collision_scale_j * bounding_radii[object_j],
                shape_types[object_j],
                collision_scale_j * half_extents[object_j],
                collision_scale_j * corner_radii[object_j],
                proxy_initial_angles[object_j],
                lower,
                upper,
                contact_slack,
                contact_margin,
            )
            if candidate_time >= 0.0 and candidate_time < best_time:
                best_time = candidate_time
                best_pair = pair_index
        if best_pair < 0:
            break

        object_i = pair_i[best_pair]
        object_j = pair_j[best_pair]
        collision_scale_i = wp.exp(
            theta[world, log_collision_scale_start + object_i]
        )
        collision_scale_j = wp.exp(
            theta[world, log_collision_scale_start + object_j]
        )
        position_i = _state_position(
            start_position[world, object_i],
            start_velocity[world, object_i],
            state_time[world, object_i],
            theta[world, friction_start + object_i],
            best_time,
        )
        position_j = _state_position(
            start_position[world, object_j],
            start_velocity[world, object_j],
            state_time[world, object_j],
            theta[world, friction_start + object_j],
            best_time,
        )
        velocity_i = _state_velocity(
            start_velocity[world, object_i],
            state_time[world, object_i],
            theta[world, friction_start + object_i],
            best_time,
        )
        velocity_j = _state_velocity(
            start_velocity[world, object_j],
            state_time[world, object_j],
            theta[world, friction_start + object_j],
            best_time,
        )
        angle_i = _state_angle(
            start_angle[world, object_i],
            start_angular_velocity[world, object_i],
            state_time[world, object_i],
            theta[world, spin_friction_start + object_i],
            radii[object_i],
            best_time,
        )
        angle_j = _state_angle(
            start_angle[world, object_j],
            start_angular_velocity[world, object_j],
            state_time[world, object_j],
            theta[world, spin_friction_start + object_j],
            radii[object_j],
            best_time,
        )
        angular_velocity_i = _state_angular_velocity(
            start_angular_velocity[world, object_i],
            state_time[world, object_i],
            theta[world, spin_friction_start + object_i],
            radii[object_i],
            best_time,
        )
        angular_velocity_j = _state_angular_velocity(
            start_angular_velocity[world, object_j],
            state_time[world, object_j],
            theta[world, spin_friction_start + object_j],
            radii[object_j],
            best_time,
        )
        proxy_angle_i = angle_i + proxy_initial_angles[object_i]
        proxy_angle_j = angle_j + proxy_initial_angles[object_j]
        geometry = _contact_geometry(
            shape_types[object_i],
            position_i,
            collision_scale_i * half_extents[object_i],
            collision_scale_i * corner_radii[object_i],
            proxy_angle_i,
            shape_types[object_j],
            position_j,
            collision_scale_j * half_extents[object_j],
            collision_scale_j * corner_radii[object_j],
            proxy_angle_j,
        )
        timing_normal = wp.vec2(geometry[0], geometry[1])
        normal = timing_normal
        response_angle_i = angle_i + response_initial_angles[object_i]
        response_angle_j = angle_j + response_initial_angles[object_j]
        response_geometry = _contact_geometry(
            response_shape_types[object_i],
            position_i,
            collision_scale_i * response_half_extents[object_i],
            collision_scale_i * response_corner_radii[object_i],
            response_angle_i,
            response_shape_types[object_j],
            position_j,
            collision_scale_j * response_half_extents[object_j],
            collision_scale_j * response_corner_radii[object_j],
            response_angle_j,
        )
        response_normal = wp.vec2(response_geometry[0], response_geometry[1])
        if use_response_normal != 0:
            normal = response_normal
        contact_point = _contact_point(
            response_shape_types[object_i],
            position_i,
            collision_scale_i * response_half_extents[object_i],
            collision_scale_i * response_corner_radii[object_i],
            response_angle_i,
            response_shape_types[object_j],
            position_j,
            collision_scale_j * response_half_extents[object_j],
            collision_scale_j * response_corner_radii[object_j],
            response_angle_j,
            response_normal,
        )
        lever_i = contact_point - position_i
        lever_j = contact_point - position_j
        torque_lever_i = lever_i
        torque_lever_j = lever_j
        tangent = wp.vec2(-normal[1], normal[0])
        mass_i = _relative_mass(theta, world, object_i, object_count, log_mass_start)
        mass_j = _relative_mass(theta, world, object_j, object_count, log_mass_start)
        inertia_i = mass_i * inertia_per_mass[object_i] * wp.exp(
            theta[world, log_inertia_scale_start + object_i]
        )
        inertia_j = mass_j * inertia_per_mass[object_j] * wp.exp(
            theta[world, log_inertia_scale_start + object_j]
        )
        contact_velocity_i = _point_velocity(velocity_i, angular_velocity_i, lever_i)
        contact_velocity_j = _point_velocity(velocity_j, angular_velocity_j, lever_j)
        relative_normal_speed = wp.dot(contact_velocity_i - contact_velocity_j, normal)
        if relative_normal_speed >= -float(CONTACT_SPEED_TOLERANCE_M_PER_S):
            break
        restitution = theta[world, restitution_start + best_pair]
        normal_lever_i = _cross_2d(lever_i, normal)
        normal_lever_j = _cross_2d(lever_j, normal)
        inverse_normal_mass = (
            1.0 / mass_i
            + 1.0 / mass_j
            + normal_lever_i * normal_lever_i / inertia_i
            + normal_lever_j * normal_lever_j / inertia_j
        )
        normal_impulse = -(1.0 + restitution) * relative_normal_speed
        normal_impulse /= inverse_normal_mass
        normal_impulse_vector = normal_impulse * normal
        angular_velocity_before_contact_i = angular_velocity_i
        angular_velocity_before_contact_j = angular_velocity_j
        velocity_i += normal_impulse_vector / mass_i
        velocity_j -= normal_impulse_vector / mass_j
        contact_torque_transfer = (
            theta[world, contact_torque_transfer_start + object_i]
            * theta[world, contact_torque_transfer_start + object_j]
        )
        normal_angular_delta_i = (
            _cross_2d(torque_lever_i, normal_impulse_vector) / inertia_i
        )
        normal_angular_delta_j = (
            _cross_2d(torque_lever_j, -normal_impulse_vector) / inertia_j
        )
        angular_velocity_i += normal_angular_delta_i
        angular_velocity_j += normal_angular_delta_j

        relative_tangential_speed = wp.dot(
            _point_velocity(velocity_i, angular_velocity_i, lever_i)
            - _point_velocity(velocity_j, angular_velocity_j, lever_j),
            tangent,
        )
        tangent_lever_i = _cross_2d(lever_i, tangent)
        tangent_lever_j = _cross_2d(lever_j, tangent)
        inverse_tangent_mass = (
            1.0 / mass_i
            + 1.0 / mass_j
            + tangent_lever_i * tangent_lever_i / inertia_i
            + tangent_lever_j * tangent_lever_j / inertia_j
        )
        tangential_impulse = -relative_tangential_speed / inverse_tangent_mass
        contact_friction = theta[world, contact_friction_start + best_pair]
        tangential_limit = contact_friction * normal_impulse
        tangential_impulse = wp.clamp(
            tangential_impulse, -tangential_limit, tangential_limit
        )
        tangent_impulse_vector = tangential_impulse * tangent
        velocity_i += tangent_impulse_vector / mass_i
        velocity_j -= tangent_impulse_vector / mass_j
        angular_velocity_i += (
            _cross_2d(torque_lever_i, tangent_impulse_vector) / inertia_i
        )
        angular_velocity_j += (
            _cross_2d(torque_lever_j, -tangent_impulse_vector) / inertia_j
        )
        # Scale the complete angular impulse only after the standard contact
        # solve, so unresolved projected torque cannot alter the linear impulse.
        angular_velocity_i = angular_velocity_before_contact_i + (
            contact_torque_transfer
            * (angular_velocity_i - angular_velocity_before_contact_i)
        )
        angular_velocity_j = angular_velocity_before_contact_j + (
            contact_torque_transfer
            * (angular_velocity_j - angular_velocity_before_contact_j)
        )

        # Shape-aware response creates realistic torque, while the circle timing
        # proxy remains the non-penetration constraint for hybrid modes.
        if use_response_normal != 0:
            timing_relative_speed = wp.dot(velocity_i - velocity_j, timing_normal)
            if timing_relative_speed < 0.0:
                separation_impulse = -timing_relative_speed / (
                    1.0 / mass_i + 1.0 / mass_j
                )
                separation_impulse_vector = separation_impulse * timing_normal
                velocity_i += separation_impulse_vector / mass_i
                velocity_j -= separation_impulse_vector / mass_j

        impact_time = best_time

        start_position[world, object_i] = position_i
        start_position[world, object_j] = position_j
        start_velocity[world, object_i] = velocity_i
        start_velocity[world, object_j] = velocity_j
        start_angle[world, object_i] = angle_i
        start_angle[world, object_j] = angle_j
        start_angular_velocity[world, object_i] = angular_velocity_i
        start_angular_velocity[world, object_j] = angular_velocity_j
        state_time[world, object_i] = best_time
        state_time[world, object_j] = best_time
        for frame_index in range(frame_count):
            query_time = float(frame_index) / fps
            if query_time >= best_time:
                future_position_i = _state_position(
                    position_i,
                    velocity_i,
                    best_time,
                    theta[world, friction_start + object_i],
                    query_time,
                )
                future_position_j = _state_position(
                    position_j,
                    velocity_j,
                    best_time,
                    theta[world, friction_start + object_j],
                    query_time,
                )
                future_angle_i = _state_angle(
                    angle_i,
                    angular_velocity_i,
                    best_time,
                    theta[world, spin_friction_start + object_i],
                    radii[object_i],
                    query_time,
                )
                future_angle_j = _state_angle(
                    angle_j,
                    angular_velocity_j,
                    best_time,
                    theta[world, spin_friction_start + object_j],
                    radii[object_j],
                    query_time,
                )
                predicted[world, frame_index, object_i] = wp.vec3(
                    future_position_i[0], future_position_i[1], future_angle_i
                )
                predicted[world, frame_index, object_j] = wp.vec3(
                    future_position_j[0], future_position_j[1], future_angle_j
                )

        last_collision_time[world, best_pair] = best_time
        event_time_out[world, event_index] = impact_time
        event_i_out[world, event_index] = object_i
        event_j_out[world, event_index] = object_j
        event_relative_normal_speed_out[world, event_index] = relative_normal_speed
        event_relative_tangential_speed_out[
            world, event_index
        ] = relative_tangential_speed
        event_normal_impulse_out[world, event_index] = normal_impulse
        event_tangential_impulse_out[world, event_index] = tangential_impulse
        event_count_out[world] = event_index + 1


class BatchWarpEventEngine:
    def __init__(
        self,
        scene: RotationalSceneData,
        *,
        capacity: int,
        device: str = "cuda:0",
        event_cap: int = 32,
        collision_cooldown_s: float = 1.0e-3,
        contact_slack_m: float = 0.0,
        contact_margin_m: float = 0.0,
        proxy_mode: str = "shape-aware",
        block_dim: int = DEFAULT_CUDA_BLOCK_DIM,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 1 <= block_dim <= 1024:
            raise ValueError("block_dim must be between 1 and 1024")
        if event_cap <= 0:
            raise ValueError("event_cap must be positive")
        if contact_slack_m < 0.0:
            raise ValueError("contact_slack_m must be non-negative")
        if contact_margin_m < 0.0:
            raise ValueError("contact_margin_m must be non-negative")
        self.scene = scene
        self.capacity = int(capacity)
        self.block_dim = int(block_dim)
        self.device = device
        self.torch_device = torch.device(device)
        self.event_cap = int(event_cap)
        self.collision_cooldown_s = float(collision_cooldown_s)
        self.contact_slack_m = float(contact_slack_m)
        self.contact_margin_m = float(contact_margin_m)
        self.proxy_mode = proxy_mode
        self.layout = make_parameter_layout(scene.object_count)
        pair_i = np.asarray([pair[0] for pair in self.layout.pairs], dtype=np.int32)
        pair_j = np.asarray([pair[1] for pair in self.layout.pairs], dtype=np.int32)
        (
            self.interaction_evidence,
            self.interaction_evidence_frames,
        ) = pair_interaction_evidence(scene)
        first_times = np.asarray(scene.first_offsets, dtype=np.float32) / scene.fps
        end_times = np.asarray(scene.last_offsets, dtype=np.float32) / scene.fps
        timing_proxy_mode = "circle" if proxy_mode in HYBRID_PROXY_MODES else proxy_mode
        response_proxy_mode = (
            "shape-aware" if proxy_mode in HYBRID_PROXY_MODES else timing_proxy_mode
        )
        self.proxy_data = collision_proxy_data(scene, proxy_mode=timing_proxy_mode)
        self.response_proxy_data = collision_proxy_data(
            scene, proxy_mode=response_proxy_mode
        )

        wp.init()
        with wp.ScopedDevice(device):
            self.theta = wp.empty(
                (self.capacity, self.layout.size), dtype=wp.float32, device=device
            )
            self.initial_xy = wp.array(
                np.asarray(scene.initial_xy, dtype=np.float32),
                dtype=wp.vec2,
                device=device,
            )
            self.initial_angle = wp.zeros(
                scene.object_count, dtype=wp.float32, device=device
            )
            self.radii = wp.array(
                self.proxy_data.spin_radii, dtype=wp.float32, device=device
            )
            self.bounding_radii = wp.array(
                self.proxy_data.bounding_radii, dtype=wp.float32, device=device
            )
            self.shape_types = wp.array(
                self.proxy_data.shape_types, dtype=wp.int32, device=device
            )
            self.half_extents = wp.array(
                self.proxy_data.half_extents, dtype=wp.vec2, device=device
            )
            self.corner_radii = wp.array(
                self.proxy_data.corner_radii, dtype=wp.float32, device=device
            )
            self.proxy_initial_angles = wp.array(
                self.proxy_data.initial_angles, dtype=wp.float32, device=device
            )
            self.response_shape_types = wp.array(
                self.response_proxy_data.shape_types, dtype=wp.int32, device=device
            )
            self.response_half_extents = wp.array(
                self.response_proxy_data.half_extents, dtype=wp.vec2, device=device
            )
            self.response_corner_radii = wp.array(
                self.response_proxy_data.corner_radii, dtype=wp.float32, device=device,
            )
            self.response_initial_angles = wp.array(
                self.response_proxy_data.initial_angles,
                dtype=wp.float32,
                device=device,
            )
            self.inertia_per_mass = wp.array(
                self.proxy_data.inertia_per_mass, dtype=wp.float32, device=device
            )
            self.first_times = wp.array(first_times, dtype=wp.float32, device=device)
            self.end_times = wp.array(end_times, dtype=wp.float32, device=device)
            self.pair_i = wp.array(pair_i, dtype=wp.int32, device=device)
            self.pair_j = wp.array(pair_j, dtype=wp.int32, device=device)
            self.start_position = wp.empty(
                (self.capacity, scene.object_count), dtype=wp.vec2, device=device
            )
            self.start_velocity = wp.empty(
                (self.capacity, scene.object_count), dtype=wp.vec2, device=device
            )
            self.start_angle = wp.empty(
                (self.capacity, scene.object_count), dtype=wp.float32, device=device
            )
            self.start_angular_velocity = wp.empty(
                (self.capacity, scene.object_count), dtype=wp.float32, device=device
            )
            self.state_time = wp.empty(
                (self.capacity, scene.object_count), dtype=wp.float32, device=device
            )
            self.last_collision_time = wp.empty(
                (self.capacity, len(self.layout.pairs)), dtype=wp.float32, device=device
            )
            self.predicted = wp.empty(
                (self.capacity, scene.frame_count, scene.object_count),
                dtype=wp.vec3,
                device=device,
            )
            self.event_count = wp.empty(self.capacity, dtype=wp.int32, device=device)
            event_shape = (self.capacity, self.event_cap)
            self.event_time = wp.empty(event_shape, dtype=wp.float32, device=device)
            self.event_i = wp.empty(event_shape, dtype=wp.int32, device=device)
            self.event_j = wp.empty(event_shape, dtype=wp.int32, device=device)
            self.event_relative_normal_speed = wp.empty(
                event_shape, dtype=wp.float32, device=device
            )
            self.event_relative_tangential_speed = wp.empty(
                event_shape, dtype=wp.float32, device=device
            )
            self.event_normal_impulse = wp.empty(
                event_shape, dtype=wp.float32, device=device
            )
            self.event_tangential_impulse = wp.empty(
                event_shape, dtype=wp.float32, device=device
            )
        self.theta_torch = wp.to_torch(self.theta)
        self.predicted_torch = wp.to_torch(self.predicted)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=self.torch_device)

        self.target_centers_world_torch = tensor(scene.target_centers_world)
        self.target_rotations_world_torch = tensor(scene.target_rotations_world)
        self.target_yaw_torch = tensor(scene.target_yaw)
        self.initial_rotations_world_torch = tensor(scene.initial_rotations_world)
        self.cloud_factors_local_torch = tensor(scene.cloud_factors_local)
        self.rotation_enabled_torch = tensor(rotational_object_mask(scene))
        self.spin_radii_torch = tensor(self.proxy_data.spin_radii)
        self.support_origin_world_torch = tensor(scene.support_origin_world)
        self.support_tangent_1_world_torch = tensor(scene.support_tangent_1_world)
        self.support_tangent_2_world_torch = tensor(scene.support_tangent_2_world)
        self.support_normal_world_torch = tensor(scene.support_normal_world)
        self.initial_heights_torch = tensor(scene.initial_heights)
        float_bytes = np.dtype(np.float32).itemsize
        int_bytes = np.dtype(np.int32).itemsize
        object_count = scene.object_count
        pair_count = len(self.layout.pairs)
        self.buffer_bytes = (
            self.capacity * self.layout.size * float_bytes
            + object_count * 2 * float_bytes
            + object_count * float_bytes * 9
            + object_count * int_bytes
            + object_count * (int_bytes + 3 * float_bytes)
            + object_count * 2 * float_bytes
            + pair_count * int_bytes * 2
            + pair_count * int_bytes
            + self.capacity * object_count * 2 * float_bytes * 2
            + self.capacity * object_count * float_bytes * 3
            + self.capacity * pair_count * float_bytes
            + self.capacity * scene.frame_count * object_count * 3 * float_bytes
            + self.capacity * int_bytes
            + self.capacity * self.event_cap * (5 * float_bytes + 2 * int_bytes)
        )

    def rollout(
        self, samples: torch.Tensor | np.ndarray, end_offset: int | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32, device=self.torch_device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[1] != self.layout.size:
            raise ValueError(
                f"expected {self.layout.size} parameters, got {values.shape[1]}"
            )
        count = int(values.shape[0])
        if count > self.capacity:
            raise ValueError(
                f"batch size {count} exceeds engine capacity {self.capacity}"
            )
        final_offset = (
            self.scene.frame_count - 1
            if end_offset is None
            else min(max(int(end_offset), 0), self.scene.frame_count - 1)
        )
        self.theta_torch[:count].copy_(values)
        wp.launch(
            _simulate_worlds,
            dim=count,
            block_dim=self.block_dim,
            inputs=[
                self.theta,
                self.initial_xy,
                self.initial_angle,
                self.radii,
                self.bounding_radii,
                self.shape_types,
                self.half_extents,
                self.corner_radii,
                self.proxy_initial_angles,
                self.response_shape_types,
                self.response_half_extents,
                self.response_corner_radii,
                self.response_initial_angles,
                self.inertia_per_mass,
                self.first_times,
                self.end_times,
                self.pair_i,
                self.pair_j,
                self.scene.object_count,
                final_offset + 1,
                len(self.layout.pairs),
                float(self.scene.fps),
                float(final_offset / self.scene.fps),
                self.event_cap,
                self.collision_cooldown_s,
                self.contact_slack_m,
                self.contact_margin_m,
                int(self.proxy_mode == "circle-shape-physical-response"),
                self.layout.angular_velocity_slice.start,
                self.layout.log_collision_scale_slice.start,
                self.layout.log_mass_slice.start,
                self.layout.log_inertia_scale_slice.start,
                self.layout.restitution_slice.start,
                self.layout.friction_slice.start,
                self.layout.spin_friction_slice.start,
                self.layout.contact_friction_slice.start,
                self.layout.contact_torque_transfer_slice.start,
                self.start_position,
                self.start_velocity,
                self.start_angle,
                self.start_angular_velocity,
                self.state_time,
                self.last_collision_time,
                self.predicted,
                self.event_count,
                self.event_time,
                self.event_i,
                self.event_j,
                self.event_relative_normal_speed,
                self.event_relative_tangential_speed,
                self.event_normal_impulse,
                self.event_tangential_impulse,
            ],
            device=self.device,
        )
        return self.predicted_torch[:count, : final_offset + 1]

    def rollout_numpy(
        self, values: np.ndarray, end_offset: int | None = None,
    ) -> tuple[np.ndarray, tuple[RotationalCollisionEvent, ...]]:
        predicted = self.rollout(values, end_offset)
        wp.synchronize_device(self.device)
        predicted_numpy = predicted[0].detach().cpu().numpy().astype(np.float64)
        count = int(self.event_count.numpy()[0])
        times = self.event_time.numpy()[0]
        indices_i = self.event_i.numpy()[0]
        indices_j = self.event_j.numpy()[0]
        relative_normal_speeds = self.event_relative_normal_speed.numpy()[0]
        relative_tangential_speeds = self.event_relative_tangential_speed.numpy()[0]
        normal_impulses = self.event_normal_impulse.numpy()[0]
        tangential_impulses = self.event_tangential_impulse.numpy()[0]
        contact_frictions = decode_parameters(values, self.layout)["contact_friction"]
        pair_lookup = {
            pair: pair_index for pair_index, pair in enumerate(self.layout.pairs)
        }
        events = []
        for index in range(count):
            object_i = int(indices_i[index])
            object_j = int(indices_j[index])
            pair_index = pair_lookup[(min(object_i, object_j), max(object_i, object_j))]
            contact_friction = float(contact_frictions[pair_index])
            normal_impulse = float(normal_impulses[index])
            tangential_impulse = float(tangential_impulses[index])
            if contact_friction <= 1.0e-8:
                contact_mode = "frictionless"
            elif abs(tangential_impulse) < contact_friction * normal_impulse - 1.0e-6:
                contact_mode = "sticking"
            else:
                contact_mode = "sliding"
            events.append(
                RotationalCollisionEvent(
                    time_s=float(times[index]),
                    object_i=object_i,
                    object_j=object_j,
                    relative_normal_velocity_m_per_s=float(
                        relative_normal_speeds[index]
                    ),
                    relative_tangential_velocity_m_per_s=float(
                        relative_tangential_speeds[index]
                    ),
                    normal_impulse_magnitude=normal_impulse,
                    tangential_impulse_magnitude=tangential_impulse,
                    contact_friction=contact_friction,
                    contact_mode=contact_mode,
                )
            )
        return predicted_numpy, tuple(events)


def _curriculum_masks(
    scene: RotationalSceneData, prefixes: tuple[int, ...]
) -> list[tuple[int, np.ndarray]]:
    stages: list[tuple[int, np.ndarray]] = []
    signatures: set[bytes] = set()
    for prefix in prefixes:
        if prefix <= 0:
            selected = scene.observed_mask > 0.0
            end_offset = scene.frame_count - 1
        else:
            end_offset = min(prefix - 1, scene.frame_count - 1)
            selected = scene.observed_mask > 0.0
            selected[end_offset + 1 :] = False
        signature = selected.tobytes()
        if signature in signatures:
            continue
        signatures.add(signature)
        if np.any(selected):
            stages.append((end_offset, selected))
    full_mask = scene.observed_mask > 0.0
    if full_mask.tobytes() not in signatures:
        stages.append((scene.frame_count - 1, full_mask))
    return stages


def _point_cloud_components_torch(
    engine: BatchWarpEventEngine, predicted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal = engine.support_normal_world_torch
    normal_outer = torch.outer(normal, normal)
    projection = (
        torch.eye(3, dtype=predicted.dtype, device=predicted.device) - normal_outer
    )
    skew = torch.stack(
        (
            torch.stack((normal[0] * 0.0, -normal[2], normal[1])),
            torch.stack((normal[2], normal[0] * 0.0, -normal[0])),
            torch.stack((-normal[1], normal[0], normal[0] * 0.0)),
        )
    )
    angles = predicted[..., 2]
    delta_rotation = (
        normal_outer
        + torch.cos(angles)[..., None, None] * projection
        + torch.sin(angles)[..., None, None] * skew
    )
    predicted_rotation = torch.matmul(
        delta_rotation, engine.initial_rotations_world_torch[None, None, ...],
    )
    xy = predicted[..., :2]
    predicted_center = (
        engine.support_origin_world_torch
        + xy[..., 0, None] * engine.support_tangent_1_world_torch
        + xy[..., 1, None] * engine.support_tangent_2_world_torch
        + engine.initial_heights_torch[None, None, :, None]
        * engine.support_normal_world_torch
    )
    frame_count = predicted.shape[1]
    center_residual = predicted_center - engine.target_centers_world_torch[
        :frame_count
    ].unsqueeze(0)
    rotation_difference = predicted_rotation - engine.target_rotations_world_torch[
        :frame_count
    ].unsqueeze(0)
    rotation_residual = torch.matmul(
        rotation_difference, engine.cloud_factors_local_torch[None, None, ...],
    )
    rotation_residual = rotation_residual * engine.rotation_enabled_torch[
        None, None, :, None, None
    ]
    point_mse = torch.sum(center_residual.square(), dim=-1) + torch.sum(
        rotation_residual.square(), dim=(-1, -2)
    )
    return center_residual, rotation_residual, point_mse


def _candidate_loss(
    engine: BatchWarpEventEngine,
    samples: torch.Tensor,
    predicted: torch.Tensor,
    end_offset: int,
    mask_numpy: np.ndarray,
    huber_delta_m: float,
    rotation_loss_weight: float,
    yaw_phase_loss_weight: float,
    collision_scale_prior_weight: float,
    contact_torque_transfer_prior_weight: float,
) -> torch.Tensor:
    mask = torch.as_tensor(
        mask_numpy[: end_offset + 1], dtype=torch.float32, device=predicted.device
    )
    center_residual, rotation_residual, _point_mse = _point_cloud_components_torch(
        engine, predicted
    )
    point_mse = torch.sum(center_residual.square(), dim=-1) + float(
        rotation_loss_weight
    ) * torch.sum(rotation_residual.square(), dim=(-1, -2))
    if yaw_phase_loss_weight > 0.0:
        yaw_residual_m = (
            predicted[..., 2]
            - engine.target_yaw_torch[None, : end_offset + 1]
        ) * engine.spin_radii_torch[None, None, :]
        point_mse = point_mse + float(yaw_phase_loss_weight) * (
            yaw_residual_m.square()
            * engine.rotation_enabled_torch[None, None, :]
        )
    absolute = torch.sqrt(torch.clamp(point_mse, min=0.0))
    delta = float(huber_delta_m)
    huber = torch.where(
        absolute <= delta, 0.5 * absolute.square(), delta * (absolute - 0.5 * delta),
    )
    numerator = torch.sum(huber * mask.unsqueeze(0), dim=(1, 2))
    denominator = torch.clamp(torch.sum(mask), min=1.0)
    loss = numerator / denominator
    log_scales = samples[:, engine.layout.log_collision_scale_slice]
    if collision_scale_prior_weight > 0.0:
        loss = loss + float(collision_scale_prior_weight) * torch.mean(
            log_scales.square(), dim=1
        )
    contact_torque_transfers = samples[:, engine.layout.contact_torque_transfer_slice]
    if contact_torque_transfer_prior_weight > 0.0:
        loss = loss + float(contact_torque_transfer_prior_weight) * torch.mean(
            (contact_torque_transfers - 1.0).square(), dim=1
        )
    return loss


def _project_torch(samples: torch.Tensor, layout: ParameterLayout) -> torch.Tensor:
    lower, upper = parameter_bounds(layout)
    lower_tensor = torch.as_tensor(lower, dtype=samples.dtype, device=samples.device)
    upper_tensor = torch.as_tensor(upper, dtype=samples.dtype, device=samples.device)
    return torch.maximum(torch.minimum(samples, upper_tensor), lower_tensor)


def _contact_torque_calibration_loss(
    engine: BatchWarpEventEngine,
    samples: torch.Tensor,
    predicted: torch.Tensor,
    end_offset: int,
    rotational_mask_numpy: np.ndarray,
    rotation_loss_weight: float,
    yaw_phase_loss_weight: float,
    contact_torque_transfer_prior_weight: float,
) -> torch.Tensor:
    observed = torch.as_tensor(
        engine.scene.observed_mask[: end_offset + 1] > 0.0,
        dtype=predicted.dtype,
        device=predicted.device,
    )
    rotational_mask = torch.as_tensor(
        rotational_mask_numpy[: end_offset + 1],
        dtype=predicted.dtype,
        device=predicted.device,
    )
    center_residual, rotation_residual, _point_mse = _point_cloud_components_torch(
        engine, predicted
    )
    center_loss = torch.sum(
        torch.sum(center_residual.square(), dim=-1) * observed.unsqueeze(0),
        dim=(1, 2),
    ) / torch.clamp(torch.sum(observed), min=1.0)
    rotation_loss = torch.sum(
        torch.sum(rotation_residual.square(), dim=(-1, -2))
        * rotational_mask.unsqueeze(0),
        dim=(1, 2),
    ) / torch.clamp(torch.sum(rotational_mask), min=1.0)
    loss = center_loss + float(rotation_loss_weight) * rotation_loss
    if yaw_phase_loss_weight > 0.0:
        yaw_residual_m = (
            predicted[..., 2]
            - engine.target_yaw_torch[None, : end_offset + 1]
        ) * engine.spin_radii_torch[None, None, :]
        yaw_loss = torch.sum(
            yaw_residual_m.square() * rotational_mask.unsqueeze(0), dim=(1, 2)
        ) / torch.clamp(torch.sum(rotational_mask), min=1.0)
        loss = loss + float(yaw_phase_loss_weight) * yaw_loss
    if contact_torque_transfer_prior_weight > 0.0:
        contact_torque_transfers = samples[
            :, engine.layout.contact_torque_transfer_slice
        ]
        loss = loss + float(contact_torque_transfer_prior_weight) * torch.mean(
            (contact_torque_transfers - 1.0).square(), dim=1
        )
    return loss


def run_cem(
    scene: RotationalSceneData,
    engine: BatchWarpEventEngine,
    config: CEMConfig,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    if config.population > engine.capacity:
        raise ValueError("CEM population exceeds engine capacity")
    if not 0.0 < config.elite_fraction < 1.0:
        raise ValueError("elite_fraction must be between zero and one")
    if config.collision_scale_prior_weight < 0.0:
        raise ValueError("collision_scale_prior_weight must be non-negative")
    if config.rotation_loss_weight <= 0.0:
        raise ValueError("rotation_loss_weight must be positive")
    if config.yaw_phase_loss_weight < 0.0:
        raise ValueError("yaw_phase_loss_weight must be non-negative")
    if config.contact_torque_transfer_prior_weight < 0.0:
        raise ValueError("contact_torque_transfer_prior_weight must be non-negative")
    layout = engine.layout
    mean_numpy, std_numpy = initial_distribution(layout)
    active_numpy = optimization_active_mask(
        scene,
        layout,
        engine.interaction_evidence,
        config.parameter_evidence_threshold_m_per_s,
        fit_contact_torque_transfers=config.fit_contact_torque_transfers,
    )
    mean = torch.as_tensor(mean_numpy, device=engine.torch_device)
    std = torch.as_tensor(std_numpy, device=engine.torch_device)
    active = torch.as_tensor(active_numpy, device=engine.torch_device)
    std = torch.where(active, std, torch.zeros_like(std))
    initial_std = std.clone()
    minimum_std = torch.as_tensor(
        minimum_distribution_std(layout), device=engine.torch_device
    )
    minimum_std = torch.where(active, minimum_std, torch.zeros_like(minimum_std))
    lower_numpy, upper_numpy = parameter_bounds(layout)
    lower = torch.as_tensor(lower_numpy, device=engine.torch_device)
    upper = torch.as_tensor(upper_numpy, device=engine.torch_device)
    generator = torch.Generator(device=engine.torch_device)
    generator.manual_seed(config.seed)
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    exploration_count = int(round(config.population * config.exploration_fraction))
    carried = mean.clone()
    global_best = carried.clone()
    global_best_loss = math.inf
    trace: list[dict[str, Any]] = []

    positive_prefixes = sorted({prefix for prefix in config.prefix_frames if prefix > 0})
    curriculum_prefixes = tuple((*positive_prefixes, 0))
    previous_transfer_active = torch.zeros(
        scene.object_count, dtype=torch.bool, device=engine.torch_device
    )
    for stage_index, (end_offset, mask) in enumerate(
        _curriculum_masks(scene, curriculum_prefixes), start=1
    ):
        stage_active = active.clone()
        stage_transfer_active = torch.zeros_like(previous_transfer_active)
        if config.fit_contact_torque_transfers:
            stage_transfer_objects = np.zeros(scene.object_count, dtype=bool)
            rotational_objects = rotational_object_mask(scene)
            for pair_index, (object_i, object_j) in enumerate(layout.pairs):
                evidence_frame = int(engine.interaction_evidence_frames[pair_index])
                if (
                    evidence_frame >= 0
                    and evidence_frame + 4 <= end_offset
                    and engine.interaction_evidence[pair_index]
                    >= config.parameter_evidence_threshold_m_per_s
                    and (rotational_objects[object_i] or rotational_objects[object_j])
                ):
                    stage_transfer_objects[object_i] = True
                    stage_transfer_objects[object_j] = True
            stage_transfer_active = torch.as_tensor(
                stage_transfer_objects,
                dtype=torch.bool,
                device=engine.torch_device,
            )
            stage_active[
                layout.contact_torque_transfer_slice
            ] &= stage_transfer_active
        newly_active_transfers = stage_transfer_active & ~previous_transfer_active
        if stage_index > 1:
            std = torch.maximum(std, 0.35 * initial_std)
            transfer_std = std[layout.contact_torque_transfer_slice]
            transfer_initial_std = initial_std[
                layout.contact_torque_transfer_slice
            ]
            transfer_std = torch.where(
                newly_active_transfers,
                transfer_initial_std,
                torch.maximum(transfer_std, 0.05 * transfer_initial_std),
            )
            std[layout.contact_torque_transfer_slice] = transfer_std
        stage_best = carried.clone()
        stage_best_loss = math.inf
        stage_start = time.perf_counter()
        for generation in range(config.generations_per_prefix):
            noise = torch.randn(
                (config.population, layout.size),
                device=engine.torch_device,
                generator=generator,
            )
            samples = _project_torch(
                mean.unsqueeze(0)
                + noise
                * torch.where(stage_active, std, torch.zeros_like(std)).unsqueeze(0),
                layout,
            )
            if exploration_count:
                uniform = torch.rand(
                    (exploration_count, layout.size),
                    device=engine.torch_device,
                    generator=generator,
                )
                random_values = lower + uniform * (upper - lower)
                exploration_active = stage_active.clone()
                if config.fit_contact_torque_transfers:
                    exploration_active[
                        layout.contact_torque_transfer_slice
                    ] = newly_active_transfers
                samples[-exploration_count:] = torch.where(
                    exploration_active.unsqueeze(0),
                    random_values,
                    mean.unsqueeze(0),
                )
            samples[0] = mean
            if config.population > 1:
                samples[1] = carried
            predicted = engine.rollout(samples, end_offset)
            losses = _candidate_loss(
                engine,
                samples,
                predicted,
                end_offset,
                mask,
                config.huber_delta_m,
                config.rotation_loss_weight,
                config.yaw_phase_loss_weight,
                config.collision_scale_prior_weight,
                config.contact_torque_transfer_prior_weight,
            )
            elite_losses, elite_indices = torch.topk(
                losses, k=elite_count, largest=False, sorted=True
            )
            elites = samples[elite_indices]
            generation_loss = float(elite_losses[0].item())
            if generation_loss < stage_best_loss:
                stage_best_loss = generation_loss
                stage_best = elites[0].clone()
            new_mean = torch.mean(elites, dim=0)
            new_std = torch.std(elites, dim=0, unbiased=False)
            new_mean = torch.where(stage_active, new_mean, mean)
            new_std = torch.where(stage_active, new_std, std)
            smoothing = float(config.distribution_smoothing)
            mean = smoothing * mean + (1.0 - smoothing) * new_mean
            std = torch.maximum(
                smoothing * std + (1.0 - smoothing) * new_std, minimum_std
            )
            if verbose:
                print(
                    f"cem stage={stage_index} generation={generation + 1}/"
                    f"{config.generations_per_prefix} frame={scene.frames[end_offset]} "
                    f"loss={generation_loss:.8g}",
                    flush=True,
                )
        carried = stage_best
        mean = stage_best.clone()
        previous_transfer_active = stage_transfer_active
        global_best = stage_best.clone()
        global_best_loss = stage_best_loss
        trace.append(
            {
                "stage": stage_index,
                "end_frame": int(scene.frames[end_offset]),
                "active_observations_per_object": [
                    int(value) for value in np.sum(mask, axis=0)
                ],
                "active_parameter_count": int(torch.sum(stage_active).item()),
                "new_contact_torque_transfer_count": int(
                    torch.sum(newly_active_transfers).item()
                ),
                "best_loss": stage_best_loss,
                "elapsed_seconds": time.perf_counter() - stage_start,
            }
        )
    return {
        "best": global_best.detach().cpu().numpy().astype(np.float64),
        "best_loss": global_best_loss,
        "trace": trace,
        "active_parameter_mask": active_numpy,
    }


def _point_cloud_components_numpy(
    scene: RotationalSceneData, predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(predicted, dtype=np.float64)
    if values.ndim == 3:
        values = values[None, ...]
    normal = scene.support_normal_world
    x, y, z = normal
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    normal_outer = np.outer(normal, normal)
    projection = np.eye(3) - normal_outer
    angles = values[..., 2]
    delta_rotation = (
        normal_outer
        + np.cos(angles)[..., None, None] * projection
        + np.sin(angles)[..., None, None] * skew
    )
    predicted_rotation = np.matmul(
        delta_rotation, scene.initial_rotations_world[None, None, ...],
    )
    xy = values[..., :2]
    predicted_center = (
        scene.support_origin_world
        + xy[..., 0, None] * scene.support_tangent_1_world
        + xy[..., 1, None] * scene.support_tangent_2_world
        + scene.initial_heights[None, None, :, None] * scene.support_normal_world
    )
    frame_count = values.shape[1]
    center_residual = predicted_center - scene.target_centers_world[None, :frame_count]
    rotation_difference = (
        predicted_rotation - scene.target_rotations_world[None, :frame_count]
    )
    rotation_residual = np.matmul(
        rotation_difference, scene.cloud_factors_local[None, None, ...],
    )
    rotation_residual *= rotational_object_mask(scene)[None, None, :, None, None]
    point_mse = np.sum(center_residual ** 2, axis=-1) + np.sum(
        rotation_residual ** 2, axis=(-1, -2)
    )
    return predicted_center, predicted_rotation, center_residual, point_mse


def _residual_matrix(
    scene: RotationalSceneData,
    predicted: np.ndarray,
    mask: np.ndarray,
    rotation_loss_weight: float = 1.0,
    yaw_phase_loss_weight: float = 0.0,
) -> np.ndarray:
    values = np.asarray(predicted, dtype=np.float64)
    if values.ndim == 3:
        values = values[None, ...]
    normal = scene.support_normal_world
    x, y, z = normal
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    normal_outer = np.outer(normal, normal)
    projection = np.eye(3) - normal_outer
    angles = values[..., 2]
    delta_rotation = (
        normal_outer
        + np.cos(angles)[..., None, None] * projection
        + np.sin(angles)[..., None, None] * skew
    )
    predicted_rotation = np.matmul(
        delta_rotation, scene.initial_rotations_world[None, None, ...],
    )
    xy = values[..., :2]
    predicted_center = (
        scene.support_origin_world
        + xy[..., 0, None] * scene.support_tangent_1_world
        + xy[..., 1, None] * scene.support_tangent_2_world
        + scene.initial_heights[None, None, :, None] * scene.support_normal_world
    )
    center_residual = predicted_center - scene.target_centers_world[None, ...]
    rotation_residual = np.matmul(
        predicted_rotation - scene.target_rotations_world[None, ...],
        scene.cloud_factors_local[None, None, ...],
    )
    rotation_residual *= rotational_object_mask(scene)[None, None, :, None, None]
    rotation_residual *= math.sqrt(float(rotation_loss_weight))
    residual_parts = [
        center_residual,
        rotation_residual.reshape((*rotation_residual.shape[:3], 9)),
    ]
    if yaw_phase_loss_weight > 0.0:
        spin_radii = collision_proxy_data(scene, proxy_mode="circle").spin_radii
        yaw_residual = (
            values[..., 2] - scene.target_yaw[None, ...]
        ) * spin_radii[None, None, :]
        yaw_residual *= rotational_object_mask(scene)[None, None, :]
        yaw_residual *= math.sqrt(float(yaw_phase_loss_weight))
        residual_parts.append(yaw_residual[..., None])
    combined = np.concatenate(residual_parts, axis=-1)
    return combined[:, mask].reshape((values.shape[0], -1))


def point_cloud_metrics(
    scene: RotationalSceneData, predicted: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(predicted, dtype=np.float64)
    _centers, _rotations, center_residual, point_mse = _point_cloud_components_numpy(
        scene, values
    )
    center_residual = center_residual[0]
    point_mse = point_mse[0]
    mask = scene.observed_mask > 0.0
    count = max(int(np.sum(mask)), 1)
    overall_point_rmse = math.sqrt(float(np.sum(point_mse * mask)) / count)
    overall_translation_rmse = math.sqrt(
        float(np.sum(np.sum(center_residual ** 2, axis=-1) * mask)) / count
    )
    yaw_difference = np.arctan2(
        np.sin(values[..., 2] - scene.target_yaw),
        np.cos(values[..., 2] - scene.target_yaw),
    )
    rotation_enabled = rotational_object_mask(scene)
    yaw_mask = mask & rotation_enabled[None, :]
    yaw_count = int(np.sum(yaw_mask))
    overall_yaw_rmse = math.degrees(
        math.sqrt(float(np.sum(yaw_difference ** 2 * yaw_mask)) / max(yaw_count, 1))
    )
    per_object_point_rmse = {}
    per_object_translation_rmse = {}
    per_object_yaw_rmse = {}
    for object_index, object_id in enumerate(scene.object_ids):
        object_mask = mask[:, object_index]
        object_count = max(int(np.sum(object_mask)), 1)
        per_object_point_rmse[object_id] = math.sqrt(
            float(np.sum(point_mse[:, object_index] * object_mask)) / object_count
        )
        per_object_translation_rmse[object_id] = math.sqrt(
            float(
                np.sum(
                    np.sum(center_residual[:, object_index] ** 2, axis=-1) * object_mask
                )
            )
            / object_count
        )
        per_object_yaw_rmse[object_id] = (
            math.degrees(
                math.sqrt(
                    float(np.sum(yaw_difference[:, object_index] ** 2 * object_mask))
                    / object_count
                )
            )
            if rotation_enabled[object_index]
            else 0.0
        )
    return {
        "overall_point_rmse_m": overall_point_rmse,
        "overall_translation_rmse_m": overall_translation_rmse,
        "overall_yaw_rmse_degrees": overall_yaw_rmse,
        "per_object_point_rmse_m": per_object_point_rmse,
        "per_object_translation_rmse_m": per_object_translation_rmse,
        "per_object_yaw_rmse_degrees": per_object_yaw_rmse,
    }


def refine_gauss_newton(
    scene: RotationalSceneData,
    engine: BatchWarpEventEngine,
    initial: np.ndarray,
    config: RefinementConfig,
    *,
    active_parameter_mask: np.ndarray | None = None,
    rotation_loss_weight: float = 1.0,
    yaw_phase_loss_weight: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    layout = engine.layout
    active_indices = np.flatnonzero(
        np.ones(layout.size, dtype=bool)
        if active_parameter_mask is None
        else np.asarray(active_parameter_mask, dtype=bool)
    )
    # Collision scale changes event topology discontinuously. CEM fits it with
    # contact guidance and a size prior; local point-residual refinement does not.
    scale_indices = np.arange(
        layout.log_collision_scale_slice.start,
        layout.log_collision_scale_slice.stop,
    )
    active_indices = np.setdiff1d(active_indices, scale_indices)
    if engine.capacity < 1 + 2 * len(active_indices):
        raise ValueError("engine capacity is too small for batched finite differences")
    theta = project_parameters(initial, layout).astype(np.float64)
    differences = finite_difference_steps(layout)
    scales = np.maximum(initial_distribution(layout)[1].astype(np.float64), differences)
    mask = scene.observed_mask > 0.0
    damping = float(config.initial_damping)
    trace: list[dict[str, Any]] = []

    for iteration in range(config.iterations):
        samples = np.repeat(theta[None, :], 1 + 2 * len(active_indices), axis=0)
        for active_offset, parameter_index in enumerate(active_indices):
            difference = differences[parameter_index]
            samples[1 + 2 * active_offset, parameter_index] += difference
            samples[2 + 2 * active_offset, parameter_index] -= difference
        samples = project_parameters(samples, layout).astype(np.float32)
        predicted = engine.rollout(samples)
        wp.synchronize_device(engine.device)
        predicted_numpy = predicted.detach().cpu().numpy().astype(np.float64)
        residuals = _residual_matrix(
            scene,
            predicted_numpy,
            mask,
            rotation_loss_weight,
            yaw_phase_loss_weight,
        )
        base_residual = residuals[0]
        base_loss = float(np.mean(base_residual * base_residual))
        jacobian = np.empty((base_residual.size, len(active_indices)), dtype=np.float64)
        for active_offset, parameter_index in enumerate(active_indices):
            plus = samples[1 + 2 * active_offset, parameter_index]
            minus = samples[2 + 2 * active_offset, parameter_index]
            denominator = float(plus - minus)
            if abs(denominator) <= 1.0e-12:
                jacobian[:, active_offset] = 0.0
            else:
                jacobian[:, active_offset] = (
                    residuals[1 + 2 * active_offset]
                    - residuals[2 + 2 * active_offset]
                ) / denominator
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ base_residual
        diagonal = np.maximum(np.diag(normal), 1.0e-8)
        try:
            step = -np.linalg.solve(normal + damping * np.diag(diagonal), gradient)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue
        normalized_step = step / scales[active_indices]
        max_step = float(np.max(np.abs(normalized_step)))
        if max_step > config.step_limit:
            step *= config.step_limit / max_step

        alphas = np.asarray([1.0, 0.5, 0.25, 0.1], dtype=np.float64)
        full_step = np.zeros(layout.size, dtype=np.float64)
        full_step[active_indices] = step
        candidates = np.vstack(
            [
                theta,
                *(
                    project_parameters(theta + alpha * full_step, layout)
                    for alpha in alphas
                ),
            ]
        ).astype(np.float32)
        candidate_prediction = engine.rollout(candidates)
        wp.synchronize_device(engine.device)
        candidate_residuals = _residual_matrix(
            scene,
            candidate_prediction.detach().cpu().numpy().astype(np.float64),
            mask,
            rotation_loss_weight,
            yaw_phase_loss_weight,
        )
        losses = np.mean(candidate_residuals * candidate_residuals, axis=1)
        selected = int(np.argmin(losses))
        accepted = selected > 0 and float(losses[selected]) < base_loss
        if accepted:
            theta = candidates[selected].astype(np.float64)
            damping = max(damping * 0.35, 1.0e-7)
        else:
            damping = min(damping * 10.0, 1.0e8)
        entry = {
            "iteration": iteration + 1,
            "loss": float(losses[selected]),
            "point_rmse_m": math.sqrt(12.0 * float(losses[selected])),
            "accepted": accepted,
            "damping": damping,
        }
        trace.append(entry)
        if verbose:
            print(json.dumps({"stage": "refine", **entry}, sort_keys=True), flush=True)
        if np.linalg.norm(step / scales[active_indices]) < 1.0e-4:
            break
    predicted, events = engine.rollout_numpy(theta)
    metrics = point_cloud_metrics(scene, predicted)
    zero_rotation_prediction = predicted.copy()
    zero_rotation_prediction[..., 2] = 0.0
    zero_rotation_metrics = point_cloud_metrics(scene, zero_rotation_prediction)
    return {
        "best": theta,
        "predicted_state": predicted,
        "events": events,
        **metrics,
        "zero_rotation_point_rmse_m": zero_rotation_metrics["overall_point_rmse_m"],
        "trace": trace,
    }


def _impute_unidentifiable_parameters(
    values: np.ndarray,
    layout: ParameterLayout,
    events: tuple[RotationalCollisionEvent, ...],
    rotating_object_indices: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    imputed = project_parameters(values, layout).astype(np.float64)
    pair_lookup = {pair: index for index, pair in enumerate(layout.pairs)}
    interacting_pair_indices = sorted(
        {
            pair_lookup[
                (
                    min(event.object_i, event.object_j),
                    max(event.object_i, event.object_j),
                )
            ]
            for event in events
        }
    )
    interacting_pair_set = set(interacting_pair_indices)
    imputed_pair_indices = [
        index for index in range(len(layout.pairs)) if index not in interacting_pair_set
    ]

    restitution = imputed[layout.restitution_slice]
    contact_friction = imputed[layout.contact_friction_slice]
    if interacting_pair_indices:
        restitution_fill = float(np.mean(restitution[interacting_pair_indices]))
        contact_friction_fill = float(
            np.mean(contact_friction[interacting_pair_indices])
        )
        pair_fill_source = "interacting_pair_mean"
    else:
        restitution_fill = DEFAULT_RESTITUTION
        contact_friction_fill = DEFAULT_CONTACT_FRICTION
        pair_fill_source = "prior_mean_no_interactions"
    restitution[imputed_pair_indices] = restitution_fill
    contact_friction[imputed_pair_indices] = contact_friction_fill

    interacting_objects = sorted(
        {
            object_index
            for pair_index in interacting_pair_indices
            for object_index in layout.pairs[pair_index]
        }
    )
    interacting_object_set = set(interacting_objects)
    imputed_object_indices = [
        index
        for index in range(layout.object_count)
        if index not in interacting_object_set
    ]
    decoded = decode_parameters(imputed, layout)
    masses = decoded["mass"].copy()
    if imputed_object_indices:
        mass_fill = (
            float(np.mean(masses[interacting_objects])) if interacting_objects else 1.0
        )
        masses[imputed_object_indices] = mass_fill
        masses /= math.exp(float(np.mean(np.log(masses))))
        if layout.object_count > 1:
            imputed[layout.log_mass_slice] = np.log(masses[:-1])
        imputed = project_parameters(imputed, layout).astype(np.float64)
        final_masses = decode_parameters(imputed, layout)["mass"]
        mass_fill = float(np.mean(final_masses[imputed_object_indices]))
    else:
        mass_fill = None

    rotating_object_set = set(rotating_object_indices)
    spin_friction_identifiable_objects = sorted(
        interacting_object_set | rotating_object_set
    )
    spin_friction_identifiable_set = set(spin_friction_identifiable_objects)
    imputed_spin_friction_indices = [
        index
        for index in range(layout.object_count)
        if index not in spin_friction_identifiable_set
    ]
    spin_friction = imputed[layout.spin_friction_slice]
    spin_friction_fill = (
        float(np.mean(spin_friction[spin_friction_identifiable_objects]))
        if spin_friction_identifiable_objects
        else DEFAULT_SPIN_FRICTION
    )
    spin_friction[imputed_spin_friction_indices] = spin_friction_fill

    return (
        imputed,
        {
            "policy": "mean_of_identifiable_interactions",
            "pair_fill_source": pair_fill_source,
            "interacting_pair_indices": interacting_pair_indices,
            "imputed_pair_indices": imputed_pair_indices,
            "imputed_object_indices": imputed_object_indices,
            "rotating_object_indices": list(rotating_object_indices),
            "spin_friction_identifiable_object_indices": (
                spin_friction_identifiable_objects
            ),
            "imputed_spin_friction_indices": imputed_spin_friction_indices,
            "restitution_fill_value": restitution_fill,
            "contact_friction_fill_value": contact_friction_fill,
            "relative_mass_fill_value": mass_fill,
            "spin_friction_fill_value": spin_friction_fill,
        },
    )


def _rotation_identifiable_object_indices(
    scene: RotationalSceneData,
) -> tuple[int, ...]:
    identifiable = []
    for object_index in range(scene.object_count):
        observed = scene.observed_mask[:, object_index] > 0.0
        yaw = scene.target_yaw[observed, object_index]
        if len(yaw) >= 2 and float(np.max(yaw) - np.min(yaw)) >= (
            ROTATION_IDENTIFIABILITY_THRESHOLD_RAD
        ):
            identifiable.append(object_index)
    return tuple(identifiable)


def calibrate_contact_torque_transfers(
    scene: RotationalSceneData,
    engine: BatchWarpEventEngine,
    initial: np.ndarray,
    config: CEMConfig,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Identify object-level angular contact response in collision order."""
    layout = engine.layout
    rotational_objects = rotational_object_mask(scene)
    _baseline_prediction, baseline_events = engine.rollout_numpy(initial)
    simulated_pairs = {
        (min(event.object_i, event.object_j), max(event.object_i, event.object_j))
        for event in baseline_events
    }
    interactions = sorted(
        (
            int(engine.interaction_evidence_frames[pair_index]),
            pair_index,
            object_i,
            object_j,
        )
        for pair_index, (object_i, object_j) in enumerate(layout.pairs)
        if engine.interaction_evidence_frames[pair_index] >= 0
        and engine.interaction_evidence[pair_index]
        >= config.parameter_evidence_threshold_m_per_s
        and (object_i, object_j) in simulated_pairs
        and (rotational_objects[object_i] or rotational_objects[object_j])
    )
    if not interactions:
        return {"best": project_parameters(initial, layout), "trace": []}

    lower_numpy, upper_numpy = parameter_bounds(layout)
    lower = torch.as_tensor(lower_numpy, device=engine.torch_device)
    upper = torch.as_tensor(upper_numpy, device=engine.torch_device)
    initial_std = torch.as_tensor(
        initial_distribution(layout)[1], device=engine.torch_device
    )
    generator = torch.Generator(device=engine.torch_device)
    generator.manual_seed(config.seed + 10_000)
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    exploration_count = int(round(config.population * config.exploration_fraction))
    carried = torch.as_tensor(
        project_parameters(initial, layout).astype(np.float32),
        device=engine.torch_device,
    )
    calibrated_transfers = np.zeros(scene.object_count, dtype=bool)
    calibrated_rotation = np.zeros(scene.object_count, dtype=bool)
    trace: list[dict[str, Any]] = []

    for stage_index, (event_frame, pair_index, object_i, object_j) in enumerate(
        interactions, start=1
    ):
        next_frame = (
            interactions[stage_index][0]
            if stage_index < len(interactions)
            else scene.frame_count - 1
        )
        end_offset = min(max(event_frame + 4, next_frame), scene.frame_count - 1)
        affected_rotation = np.zeros(scene.object_count, dtype=bool)
        affected_rotation[object_i] = rotational_objects[object_i]
        affected_rotation[object_j] = rotational_objects[object_j]
        mask = np.zeros_like(scene.observed_mask, dtype=bool)
        mask[: end_offset + 1, affected_rotation] = (
            scene.observed_mask[: end_offset + 1, affected_rotation] > 0.0
        )

        active_numpy = np.zeros(layout.size, dtype=bool)
        new_transfer_objects = [
            object_index
            for object_index in (object_i, object_j)
            if not calibrated_transfers[object_index]
        ]
        if len(new_transfer_objects) == 2:
            nonrotating_new = [
                object_index
                for object_index in new_transfer_objects
                if not rotational_objects[object_index]
            ]
            transfer_object = (
                nonrotating_new[0] if nonrotating_new else new_transfer_objects[0]
            )
            active_numpy[
                layout.contact_torque_transfer_slice.start + transfer_object
            ] = True
        elif new_transfer_objects:
            active_numpy[
                layout.contact_torque_transfer_slice.start
                + new_transfer_objects[0]
            ] = True
        calibrated_transfers[object_i] = True
        calibrated_transfers[object_j] = True
        for object_index in (object_i, object_j):
            if rotational_objects[object_index] and not calibrated_rotation[object_index]:
                active_numpy[layout.spin_friction_slice.start + object_index] = True
                calibrated_rotation[object_index] = True
        active = torch.as_tensor(active_numpy, device=engine.torch_device)
        mean = carried.clone()
        std = torch.where(active, initial_std, torch.zeros_like(initial_std))
        stage_best = carried.clone()
        stage_best_loss = math.inf
        stage_start = time.perf_counter()

        for _generation in range(config.generations_per_prefix):
            noise = torch.randn(
                (config.population, layout.size),
                device=engine.torch_device,
                generator=generator,
            )
            samples = _project_torch(mean.unsqueeze(0) + noise * std.unsqueeze(0), layout)
            samples[:, ~active] = carried[~active]
            if exploration_count:
                uniform = torch.rand(
                    (exploration_count, layout.size),
                    device=engine.torch_device,
                    generator=generator,
                )
                random_values = lower + uniform * (upper - lower)
                samples[-exploration_count:] = torch.where(
                    active.unsqueeze(0), random_values, carried.unsqueeze(0)
                )
            samples[0] = stage_best
            predicted = engine.rollout(samples, end_offset)
            losses = _contact_torque_calibration_loss(
                engine,
                samples,
                predicted,
                end_offset,
                mask,
                config.rotation_loss_weight,
                config.yaw_phase_loss_weight,
                config.contact_torque_transfer_prior_weight,
            )
            elite_losses, elite_indices = torch.topk(
                losses, k=elite_count, largest=False, sorted=True
            )
            elites = samples[elite_indices]
            if float(elite_losses[0]) < stage_best_loss:
                stage_best_loss = float(elite_losses[0])
                stage_best = elites[0].clone()
            mean[active] = torch.mean(elites, dim=0)[active]
            std[active] = torch.maximum(
                torch.std(elites, dim=0, unbiased=False)[active],
                0.01 * initial_std[active],
            )
        carried = stage_best
        entry = {
            "stage": stage_index,
            "object_ids": [scene.object_ids[object_i], scene.object_ids[object_j]],
            "evidence_frame": int(scene.frames[event_frame]),
            "end_frame": int(scene.frames[end_offset]),
            "active_parameter_count": int(np.sum(active_numpy)),
            "best_loss": stage_best_loss,
            "elapsed_seconds": time.perf_counter() - stage_start,
        }
        trace.append(entry)
        if verbose:
            print(json.dumps({"stage": "contact_torque", **entry}), flush=True)
    return {"best": carried.detach().cpu().numpy().astype(np.float64), "trace": trace}


def finalize_candidate_parameters(
    scene: RotationalSceneData,
    engine: BatchWarpEventEngine,
    candidate: dict[str, Any],
    config: CEMConfig,
    refinement_config: RefinementConfig,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit initial spin with imputed material parameters fixed before scoring."""
    parameters, imputation = _impute_unidentifiable_parameters(
        candidate["best"], engine.layout, candidate["events"],
        _rotation_identifiable_object_indices(scene),
    )
    rotation_enabled = rotational_object_mask(scene)
    spin_objects = [index for index in imputation["imputed_spin_friction_indices"]
                    if rotation_enabled[index]]
    active = np.zeros(engine.layout.size, dtype=bool)
    for index in spin_objects:
        active[engine.layout.angular_velocity_slice.start + index] = True
    # Friction and initial spin are coupled: replacing friction alone can turn
    # a nearly stationary fit into a rapidly spinning exported trajectory.
    refined = refine_gauss_newton(
        scene, engine, parameters,
        refinement_config if spin_objects else replace(refinement_config, iterations=0),
        active_parameter_mask=active,
        rotation_loss_weight=config.rotation_loss_weight,
        yaw_phase_loss_weight=config.yaw_phase_loss_weight,
        verbose=verbose,
    )
    finalized = dict(candidate)
    finalized.update({key: value for key, value in refined.items() if key != "trace"})
    finalized["post_imputation_refinement_trace"] = refined["trace"]
    imputation["initial_spin_refitted_object_indices"] = spin_objects
    finalized["parameter_imputation"] = imputation
    return finalized


def fit_scene(
    scene: RotationalSceneData,
    *,
    cem_configs: tuple[CEMConfig, ...],
    refinement_config: RefinementConfig,
    device: str,
    proxy_mode: str,
    event_cap: int = 32,
    collision_cooldown_s: float = 1.0e-3,
    contact_slack_m: float = 0.0,
    contact_margin_m: float = 0.0,
    verbose: bool = True,
) -> tuple[dict[str, Any], CEMConfig, float]:
    if not cem_configs:
        raise ValueError("at least one CEM configuration is required")
    layout = make_parameter_layout(scene.object_count)
    required_capacity = max(
        max(config.population for config in cem_configs), 1 + 2 * layout.size,
    )
    engine = BatchWarpEventEngine(
        scene,
        capacity=required_capacity,
        device=device,
        event_cap=event_cap,
        collision_cooldown_s=collision_cooldown_s,
        contact_slack_m=contact_slack_m,
        contact_margin_m=contact_margin_m,
        proxy_mode=proxy_mode,
    )
    if engine.torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.torch_device)
    start = time.perf_counter()
    fit: dict[str, Any] | None = None
    selected_config: CEMConfig | None = None
    portfolio = []
    best_selection_rmse = math.inf
    rotation_enabled = rotational_object_mask(scene)
    selection_spin_radii = collision_proxy_data(
        scene, proxy_mode="circle"
    ).spin_radii
    selection_mask = scene.observed_mask > 0.0
    selection_count = max(int(np.sum(selection_mask)), 1)

    def selection_metrics(
        candidate: dict[str, Any], config: CEMConfig
    ) -> tuple[float, float]:
        rotational_point_mse = max(
            candidate["overall_point_rmse_m"] ** 2
            - candidate["overall_translation_rmse_m"] ** 2,
            0.0,
        )
        yaw_phase_residual_m = (
            candidate["predicted_state"][..., 2] - scene.target_yaw
        ) * selection_spin_radii[None, :]
        yaw_phase_mse = float(
            np.sum(
                yaw_phase_residual_m**2
                * selection_mask
                * rotation_enabled[None, :]
            )
            / selection_count
        )
        selection_rmse = math.sqrt(
            candidate["overall_translation_rmse_m"] ** 2
            + config.rotation_loss_weight * rotational_point_mse
            + config.yaw_phase_loss_weight * yaw_phase_mse
        )
        return selection_rmse, yaw_phase_mse

    for config in cem_configs:
        candidate_start = time.perf_counter()
        base_config = (
            replace(
                config,
                fit_contact_torque_transfers=False,
                yaw_phase_loss_weight=0.0,
            )
            if config.fit_contact_torque_transfers
            else config
        )
        cem = run_cem(scene, engine, base_config, verbose=verbose)
        base_refinement = (
            refine_gauss_newton(
                scene,
                engine,
                cem["best"],
                refinement_config,
                active_parameter_mask=cem["active_parameter_mask"],
                rotation_loss_weight=base_config.rotation_loss_weight,
                yaw_phase_loss_weight=base_config.yaw_phase_loss_weight,
                verbose=verbose,
            )
            if config.fit_contact_torque_transfers
            else {"best": cem["best"], "trace": []}
        )
        torque_calibration = (
            calibrate_contact_torque_transfers(
                scene, engine, base_refinement["best"], config, verbose=verbose
            )
            if config.fit_contact_torque_transfers
            else {"best": base_refinement["best"], "trace": []}
        )
        active_parameter_mask = optimization_active_mask(
            scene,
            layout,
            engine.interaction_evidence,
            config.parameter_evidence_threshold_m_per_s,
            fit_contact_torque_transfers=config.fit_contact_torque_transfers,
        )
        candidate = refine_gauss_newton(
            scene,
            engine,
            torque_calibration["best"],
            refinement_config,
            active_parameter_mask=active_parameter_mask,
            rotation_loss_weight=config.rotation_loss_weight,
            yaw_phase_loss_weight=config.yaw_phase_loss_weight,
            verbose=verbose,
        )
        candidate = finalize_candidate_parameters(
            scene, engine, candidate, config, refinement_config, verbose=verbose,
        )
        selection_rmse, yaw_phase_mse = selection_metrics(candidate, config)
        selected_path = (
            "contact_torque_calibration"
            if config.fit_contact_torque_transfers
            else "standard"
        )
        if config.fit_contact_torque_transfers:
            base_refinement = finalize_candidate_parameters(
                scene, engine, base_refinement, config, refinement_config, verbose=verbose,
            )
            baseline_selection_rmse, baseline_yaw_phase_mse = selection_metrics(
                base_refinement, config
            )
            if baseline_selection_rmse < selection_rmse:
                candidate = base_refinement
                selection_rmse = baseline_selection_rmse
                yaw_phase_mse = baseline_yaw_phase_mse
                selected_path = "baseline_fallback"
        candidate["active_parameter_mask"] = active_parameter_mask
        candidate["cem_trace"] = cem["trace"]
        candidate["base_refinement_trace"] = base_refinement["trace"]
        candidate["contact_torque_calibration_trace"] = torque_calibration["trace"]
        candidate["selected_optimization_path"] = selected_path
        portfolio.append(
            {
                "seed": config.seed,
                "overall_point_rmse_m": candidate["overall_point_rmse_m"],
                "overall_translation_rmse_m": candidate["overall_translation_rmse_m"],
                "overall_yaw_rmse_degrees": candidate["overall_yaw_rmse_degrees"],
                "selection_rmse_m": selection_rmse,
                "yaw_phase_rmse_m": math.sqrt(yaw_phase_mse),
                "selected_optimization_path": selected_path,
                "collision_count": len(candidate["events"]),
                "elapsed_seconds": time.perf_counter() - candidate_start,
            }
        )
        if fit is None or selection_rmse < best_selection_rmse:
            fit = candidate
            selected_config = config
            best_selection_rmse = selection_rmse
    if fit is None or selected_config is None:
        raise RuntimeError("CEM portfolio did not produce a candidate")
    fit["portfolio"] = portfolio
    fit["engine_buffer_bytes"] = engine.buffer_bytes
    fit["interaction_evidence"] = engine.interaction_evidence
    fit["interaction_evidence_frames"] = engine.interaction_evidence_frames
    fit["torch_peak_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(engine.torch_device))
        if engine.torch_device.type == "cuda"
        else 0
    )
    return fit, selected_config, time.perf_counter() - start


def _predicted_poses_world(
    scene: RotationalSceneData, predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers, rotations, _center_residual, _point_mse = _point_cloud_components_numpy(
        scene, predicted
    )
    centers = centers[0]
    rotations = rotations[0]
    translations = centers - np.einsum(
        "fnij,nj->fni", rotations, scene.cloud_centroids_local
    )
    world_poses = np.broadcast_to(
        np.eye(4, dtype=np.float64), (*rotations.shape[:2], 4, 4)
    ).copy()
    world_poses[..., :3, :3] = rotations
    world_poses[..., :3, 3] = translations
    return centers, world_poses


def result_payload(
    source: dict[str, Any],
    scene: RotationalSceneData,
    cem_config: CEMConfig,
    refinement_config: RefinementConfig,
    fit: dict[str, Any],
    elapsed_seconds: float,
    device: str,
    proxy_mode: str,
    contact_slack_m: float = 0.0,
    contact_margin_m: float = 0.0,
) -> dict[str, Any]:
    layout = make_parameter_layout(scene.object_count)
    parameters = decode_parameters(fit["best"], layout)
    predicted = fit["predicted_state"]
    predicted_centers, predicted_poses_world = _predicted_poses_world(scene, predicted)
    proxy_data = collision_proxy_data(scene, proxy_mode=proxy_mode)
    response_proxy_data = collision_proxy_data(
        scene,
        proxy_mode=("shape-aware" if proxy_mode in HYBRID_PROXY_MODES else proxy_mode),
    )
    events: tuple[RotationalCollisionEvent, ...] = fit["events"]
    imputation = fit["parameter_imputation"]
    interacting_pair_indices = set(imputation["interacting_pair_indices"])
    imputed_pair_indices = set(imputation["imputed_pair_indices"])
    imputed_object_indices = set(imputation["imputed_object_indices"])
    imputed_spin_friction_indices = set(imputation["imputed_spin_friction_indices"])
    rotation_enabled = rotational_object_mask(scene)
    object_parameters = {
        object_id: {
            "initial_velocity_plane_m_per_s": parameters["velocity"][index].tolist(),
            "initial_angular_velocity_rad_per_s": float(
                parameters["angular_velocity"][index]
            ),
            "relative_mass": float(parameters["mass"][index]),
            "relative_mass_imputed": index in imputed_object_indices,
            "inertia_scale": float(parameters["inertia_scale"][index]),
            "ground_friction": float(parameters["friction"][index]),
            "spin_friction": float(parameters["spin_friction"][index]),
            "spin_friction_imputed": index in imputed_spin_friction_indices,
            "collision_scale": float(parameters["collision_scale"][index]),
            "contact_torque_transfer": float(
                parameters["contact_torque_transfer"][index]
            ),
            "rotation_mode": (
                "planar_rigid"
                if rotation_enabled[index]
                else "unobserved_collision_spin"
            ),
        }
        for index, object_id in enumerate(scene.object_ids)
    }
    pair_parameters = [
        {
            "object_ids": [scene.object_ids[object_i], scene.object_ids[object_j]],
            "restitution": float(parameters["restitution"][pair_index]),
            "contact_friction": float(parameters["contact_friction"][pair_index]),
            "simulated_interaction": pair_index in interacting_pair_indices,
            "imputed_from_mean": pair_index in imputed_pair_indices,
        }
        for pair_index, (object_i, object_j) in enumerate(layout.pairs)
    ]
    return {
        "model": "physmind-v2-warp-continuous-event-swr-v7",
        "source": source,
        "device": device,
        "proxy_mode": proxy_mode,
        "contact_slack_m": contact_slack_m,
        "contact_margin_m": contact_margin_m,
        "contact_release_hysteresis_m": CONTACT_RELEASE_GAP_M,
        "cem_config": asdict(cem_config),
        "refinement_config": asdict(refinement_config),
        "object_ids": list(scene.object_ids),
        "valid_frame_counts": {
            object_id: scene.valid_frame_counts[index]
            for index, object_id in enumerate(scene.object_ids)
        },
        "fps": scene.fps,
        "frame_count": scene.frame_count,
        "overall_point_rmse_m": fit["overall_point_rmse_m"],
        "overall_translation_rmse_m": fit["overall_translation_rmse_m"],
        "overall_yaw_rmse_degrees": fit["overall_yaw_rmse_degrees"],
        "per_object_point_rmse_m": fit["per_object_point_rmse_m"],
        "per_object_translation_rmse_m": fit["per_object_translation_rmse_m"],
        "per_object_yaw_rmse_degrees": fit["per_object_yaw_rmse_degrees"],
        "zero_rotation_point_rmse_m": fit["zero_rotation_point_rmse_m"],
        "elapsed_seconds": elapsed_seconds,
        "selected_seed": cem_config.seed,
        "selected_optimization_path": fit["selected_optimization_path"],
        "portfolio": fit["portfolio"],
        "engine_buffer_bytes": fit["engine_buffer_bytes"],
        "torch_peak_memory_bytes": fit["torch_peak_memory_bytes"],
        "active_parameter_count": int(np.sum(fit["active_parameter_mask"])),
        "total_parameter_count": layout.size,
        "parameters": object_parameters,
        "pair_parameters": pair_parameters,
        "interaction_evidence": [
            {
                "object_ids": [scene.object_ids[object_i], scene.object_ids[object_j]],
                "score_m_per_s": float(fit["interaction_evidence"][pair_index]),
                "peak_frame_index": (
                    int(
                        scene.frames[
                            int(fit["interaction_evidence_frames"][pair_index])
                        ]
                    )
                    if fit["interaction_evidence_frames"][pair_index] >= 0
                    else None
                ),
                "used_as_collision_gate": False,
            }
            for pair_index, (object_i, object_j) in enumerate(layout.pairs)
        ],
        "parameter_imputation": {
            "policy": imputation["policy"],
            "pair_fill_source": imputation["pair_fill_source"],
            "interacting_pairs": [
                [scene.object_ids[object_i], scene.object_ids[object_j]]
                for pair_index, (object_i, object_j) in enumerate(layout.pairs)
                if pair_index in interacting_pair_indices
            ],
            "imputed_pairs": [
                [scene.object_ids[object_i], scene.object_ids[object_j]]
                for pair_index, (object_i, object_j) in enumerate(layout.pairs)
                if pair_index in imputed_pair_indices
            ],
            "imputed_relative_mass_objects": [
                scene.object_ids[index] for index in sorted(imputed_object_indices)
            ],
            "rotation_identifiable_objects": [
                scene.object_ids[index]
                for index in imputation["rotating_object_indices"]
            ],
            "spin_friction_identifiable_objects": [
                scene.object_ids[index]
                for index in imputation["spin_friction_identifiable_object_indices"]
            ],
            "imputed_spin_friction_objects": [
                scene.object_ids[index]
                for index in sorted(imputed_spin_friction_indices)
            ],
            "restitution_fill_value": imputation["restitution_fill_value"],
            "contact_friction_fill_value": imputation["contact_friction_fill_value"],
            "relative_mass_fill_value": imputation["relative_mass_fill_value"],
            "spin_friction_fill_value": imputation["spin_friction_fill_value"],
            "initial_spin_refitted_objects": [
                scene.object_ids[index]
                for index in imputation.get("initial_spin_refitted_object_indices", [])
            ],
        },
        "collision_proxies": {
            object_id: {
                "source_geometry_type": scene.shapes[index].geometry_type,
                "type": (
                    "oriented_box"
                    if proxy_data.shape_types[index] == PROXY_BOX
                    else (
                        "rounded_box"
                        if proxy_data.shape_types[index] == PROXY_ROUNDED_BOX
                        else "circle"
                    )
                ),
                "response_contact_type": (
                    "oriented_box"
                    if response_proxy_data.shape_types[index] == PROXY_BOX
                    else (
                        "rounded_box"
                        if response_proxy_data.shape_types[index] == PROXY_ROUNDED_BOX
                        else "circle"
                    )
                ),
                "half_extents_m": proxy_data.half_extents[index].tolist(),
                "effective_half_extents_m": (
                    proxy_data.half_extents[index]
                    * parameters["collision_scale"][index]
                ).tolist(),
                "corner_radius_m": float(proxy_data.corner_radii[index]),
                "initial_angle_rad": float(proxy_data.initial_angles[index]),
                "spin_radius_m": float(proxy_data.spin_radii[index]),
                "bounding_radius_m": float(proxy_data.bounding_radii[index]),
                "effective_bounding_radius_m": float(
                    proxy_data.bounding_radii[index]
                    * parameters["collision_scale"][index]
                ),
                "inertia_per_mass_m2": float(proxy_data.inertia_per_mass[index]),
                "effective_inertia_per_mass_m2": float(
                    proxy_data.inertia_per_mass[index]
                    * parameters["inertia_scale"][index]
                ),
            }
            for index, object_id in enumerate(scene.object_ids)
        },
        "events": [
            {
                "event_index": event_index,
                "event_time_s": event.time_s,
                "event_frame_float": scene.frames[0] + event.time_s * scene.fps,
                "object_ids": [
                    scene.object_ids[event.object_i],
                    scene.object_ids[event.object_j],
                ],
                "relative_normal_velocity_m_per_s": event.relative_normal_velocity_m_per_s,
                "relative_tangential_velocity_m_per_s": (
                    event.relative_tangential_velocity_m_per_s
                ),
                "normal_impulse_magnitude": event.normal_impulse_magnitude,
                "tangential_impulse_magnitude": event.tangential_impulse_magnitude,
                "contact_friction": event.contact_friction,
                "contact_mode": event.contact_mode,
            }
            for event_index, event in enumerate(events)
        ],
        "cem_trace": fit["cem_trace"],
        "base_refinement_trace": fit["base_refinement_trace"],
        "contact_torque_calibration_trace": fit[
            "contact_torque_calibration_trace"
        ],
        "refinement_trace": fit["trace"],
        "post_imputation_refinement_trace": fit.get("post_imputation_refinement_trace", []),
        "predicted_trajectories": {
            object_id: [
                {
                    "frame_index": int(scene.frames[frame_index]),
                    "plane_xy": predicted[frame_index, object_index, :2].tolist(),
                    "yaw_delta_rad": float(predicted[frame_index, object_index, 2]),
                    "point_cloud_center_world": predicted_centers[
                        frame_index, object_index
                    ].tolist(),
                    "pose_4x4_world": predicted_poses_world[
                        frame_index, object_index
                    ].tolist(),
                }
                for frame_index in range(scene.frame_count)
                if scene.observed_mask[frame_index, object_index] > 0.0
            ]
            for object_index, object_id in enumerate(scene.object_ids)
        },
        "point_cloud_target": {
            "pose_source": "rigid_trajectories poses",
            "coordinate_frame": "track4world_unified_world",
            "valid_frame_policy": (
                "finite poses and not border suppressed; static objects require a "
                "nonempty mask, visible canonical points, and no occlusion termination; "
                "dynamic objects require sufficient ICP correspondences"
            ),
            "correspondence": "fixed canonical_tracking_points identities",
            "loss_compression": "centroid_and_covariance_exact_mse",
            "point_counts": {
                object_id: scene.cloud_point_counts[index]
                for index, object_id in enumerate(scene.object_ids)
            },
            "trajectory_paths": {
                object_id: scene.trajectory_paths[index]
                for index, object_id in enumerate(scene.object_ids)
            },
        },
        "physics_assumptions": {
            "execution": "custom_nvidia_warp_kernel",
            "free_motion": "closed_form_coulomb_linear_and_angular_deceleration",
            "collision_time": (
                "continuous_conservative_advancement_with_selected_timing_proxy"
            ),
            "contact_slack": (
                "global_allowed_proxy_penetration_before_impulse"
                if contact_slack_m > 0.0
                else "disabled"
            ),
            "contact_margin": (
                "global_collision_detection_skin"
                if contact_margin_m > 0.0
                else "disabled"
            ),
            "collision_dimensions": (
                "per_object_isotropic_scale_fitted_for_collision_only_with_log_prior;_"
                "rendering_dimensions_and_inertia_unchanged"
            ),
            "contact_release": "one_millimeter_separation_hysteresis_before_recontact",
            "interaction_gate": "disabled;_all_pairs_use_geometric_collision_detection",
            "collision_response": (
                "standard_linear_impulse_then_signed_product_of_per_object_"
                "contact_torque_transfers_applied_to_the_total_angular_impulse"
            ),
            "effective_contact_point": (
                "shape_contact_for_linear_and_angular_impulse"
            ),
            "rotational_degrees_of_freedom": (
                "optimized_and_observed_for_box_and_irregular_only;_sphere_and_"
                "cylinder_keep_unobserved_collision_spin_for_impulse_exchange"
            ),
            "response_contact_point": (
                "shape_aware_contact_point_and_normal_with_circle_timing"
                if proxy_mode == "circle-shape-physical-response"
                else (
                    "shape_aware_contact_point_with_circle_timing_normal"
                    if proxy_mode == "circle-shape-response"
                    else "same_geometry_as_collision_timing"
                )
            ),
            "contact_friction": "one_coefficient_per_object_pair_with_stick_slip_clamp",
            "unidentifiable_parameters": (
                "noninteracting_pair_restitution_and_contact_friction_use_the_"
                "interacting_pair_mean;_isolated_object_mass_uses_the_interacting_"
                "object_mean;_objects_without_interactions_or_observable_rotation_"
                "use_the_identifiable_object_spin_friction_mean;_unidentifiable_"
                "contact_torque_transfers_remain_one"
            ),
            "ground_spin_resistance": (
                "per_object_uniform_pressure_disk_with_spin_friction_independent_"
                "of_translational_ground_friction"
            ),
            "moment_of_inertia": (
                "fitted_effective_scale_times_timing_proxy_solid_disk_for_hybrid"
                if proxy_mode in HYBRID_PROXY_MODES
                else "fitted_effective_scale_times_shape_analytic_inertia"
            ),
            "contact_proxies": (
                "rounded_rectangle_for_box_else_circle"
                if proxy_mode == "rounded-box"
                else (
                    "oriented_rectangle_for_box_else_circle"
                    if proxy_mode == "shape-aware"
                    else (
                        "circle_timing_with_shape_aware_response_contact"
                        if proxy_mode in HYBRID_PROXY_MODES
                        else "circle_for_every_object"
                    )
                )
            ),
            "initialization": "zero_linear_and_angular_velocity_distribution_without_trajectory_fit",
        },
    }
