"""Reduced-order Warp simulation and system identification for soft balls."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import warp as wp


GRAVITY_M_PER_S2 = 9.81
MIN_TRACKED_POINTS = 32
DEFAULT_SEEDS = (7, 23, 43)
DEFAULT_CUDA_BLOCK_DIM = 32
MAX_TRANSIENT_CONTACT_GAP_SECONDS = 0.5
MAX_VISIBILITY_FLICKER_GAP_FRAMES = 2
MIN_DEFORMED_SUPPORT_SCALE = 0.38
PAIR_NORMAL_SMOOTHING_SECONDS = 0.2
MIN_TARGET_PAIR_NORMAL_ALIGNMENT = 0.25
WORST_OBJECT_SELECTION_WEIGHT = 0.25
OUTPUT_PROJECTION_TOLERANCE_M = 1.0e-9
MAX_OUTPUT_PROJECTION_ITERATIONS = 8192

# Shared model metadata; these descriptions do not alter fitted coefficients.
PARAMETER_SEMANTICS = {
    "plasticity": "Not identified: no yield threshold or evolving rest shape is fitted. Slow recovery or flattening under sustained contact does not prove permanent set. This model limitation does not establish zero real plasticity.",
    "elasticity": "Distinguish contact stiffness, rebound energy recovery and reversible shape recovery; no single elasticity scalar is exported. Compare behavior under comparable loading and geometry.",
    "natural_frequency_rad_per_s": "Mass-normalized contact stiffness scales with frequency squared; not restitution or rebound height.",
    "damping_ratio": "Contact dissipation during approach; interpret with contact frequency. It does not model plastic deformation.",
    "deformation_gain": "Contact compression to target strain gain: higher gain gives more reversible strain at equal compression before saturation, not permanent set.",
    "shape_frequency_rad_per_s": "Strain restoration rate toward the current contact target. After unloading, the target returns to zero reference strain.",
    "shape_damping_ratio": "Damps shape recovery oscillations; together with shape frequency controls recovery time. No permanent rest-shape update.",
}


@dataclass(frozen=True)
class StaticObstacle:
    object_id: str
    center: tuple[float, float]
    axis_x: tuple[float, float]
    half_extents: tuple[float, float]


@dataclass(frozen=True)
class SoftBodySceneData:
    object_ids: tuple[str, ...]
    frames: tuple[int, ...]
    fps: float
    minimum_tracking_confidence: float
    target_centers: np.ndarray
    target_subset_centers: np.ndarray
    reference_subset_offsets: np.ndarray
    target_covariances: np.ndarray
    reference_subset_covariances: np.ndarray
    target_deformations: np.ndarray
    shape_target_source: str
    shape_observed_mask: np.ndarray
    shape_contact_recovered_mask: np.ndarray
    observed_mask: np.ndarray
    first_offsets: tuple[int, ...]
    initial_centers: np.ndarray
    radii: np.ndarray
    wall_lateral_world: np.ndarray
    gravity_down_world: np.ndarray
    wall_normal_world: np.ndarray
    wall_depths: np.ndarray
    trajectory_paths: tuple[str, ...]
    valid_frame_counts: tuple[int, ...]
    point_counts: tuple[int, ...]
    obstacles: tuple[StaticObstacle, ...]

    @property
    def object_count(self) -> int:
        return len(self.object_ids)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class ParameterLayout:
    object_count: int
    velocity_slice: slice
    log_gravity_scale_slice: slice
    log_radius_scale_slice: slice
    log_frequency_slice: slice
    damping_ratio_slice: slice
    friction_slice: slice
    deformation_gain_slice: slice
    log_shape_frequency_slice: slice
    shape_damping_ratio_slice: slice
    drag_slice: slice
    size: int


@dataclass(frozen=True)
class SoftBodyCEMConfig:
    population: int = 768
    elite_fraction: float = 0.08
    generations_per_prefix: int = 48
    prefix_frames: tuple[int, ...] = (24, 48, 72, 96, 0)
    distribution_smoothing: float = 0.2
    exploration_fraction: float = 0.02
    huber_delta_m: float = 0.03
    shape_weight: float = 1.0
    deformation_focus: float = 4.0
    joint_generations_per_prefix: int = 48
    joint_initial_std_scale: float = 0.5
    shape_refinement_generations: int = 48
    shape_event_weight: float = 1.0
    seed: int = 23


def make_parameter_layout(object_count: int) -> ParameterLayout:
    if object_count <= 0:
        raise ValueError("object_count must be positive")
    offset = 0

    def take(width: int) -> slice:
        nonlocal offset
        result = slice(offset, offset + width)
        offset += width
        return result

    return ParameterLayout(
        object_count=object_count,
        velocity_slice=take(2 * object_count),
        log_gravity_scale_slice=take(object_count),
        log_radius_scale_slice=take(object_count),
        log_frequency_slice=take(object_count),
        damping_ratio_slice=take(object_count),
        friction_slice=take(object_count),
        deformation_gain_slice=take(object_count),
        log_shape_frequency_slice=take(object_count),
        shape_damping_ratio_slice=take(object_count),
        drag_slice=take(object_count),
        size=offset,
    )


def parameter_bounds(layout: ParameterLayout) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(layout.size, dtype=np.float32)
    upper = np.empty(layout.size, dtype=np.float32)
    lower[layout.velocity_slice], upper[layout.velocity_slice] = -5.0, 5.0
    lower[layout.log_gravity_scale_slice] = math.log(0.03)
    upper[layout.log_gravity_scale_slice] = math.log(1.5)
    lower[layout.log_radius_scale_slice] = math.log(0.55)
    upper[layout.log_radius_scale_slice] = math.log(1.25)
    lower[layout.log_frequency_slice] = math.log(5.0)
    upper[layout.log_frequency_slice] = math.log(80.0)
    lower[layout.damping_ratio_slice], upper[layout.damping_ratio_slice] = 0.02, 2.0
    lower[layout.friction_slice], upper[layout.friction_slice] = 0.0, 1.0
    lower[layout.deformation_gain_slice] = 0.0
    upper[layout.deformation_gain_slice] = 8.0
    lower[layout.log_shape_frequency_slice] = math.log(5.0)
    upper[layout.log_shape_frequency_slice] = math.log(100.0)
    lower[layout.shape_damping_ratio_slice] = 0.02
    upper[layout.shape_damping_ratio_slice] = 2.0
    lower[layout.drag_slice], upper[layout.drag_slice] = 0.0, 2.0
    return lower, upper


def initial_distribution(layout: ParameterLayout) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(layout.size, dtype=np.float32)
    std = np.zeros(layout.size, dtype=np.float32)
    std[layout.velocity_slice] = 1.0
    mean[layout.log_gravity_scale_slice] = math.log(0.25)
    std[layout.log_gravity_scale_slice] = 0.5
    mean[layout.log_radius_scale_slice] = math.log(0.9)
    std[layout.log_radius_scale_slice] = 0.15
    mean[layout.log_frequency_slice] = math.log(28.0)
    std[layout.log_frequency_slice] = 0.55
    mean[layout.damping_ratio_slice] = 0.35
    std[layout.damping_ratio_slice] = 0.3
    mean[layout.friction_slice] = 0.15
    std[layout.friction_slice] = 0.2
    mean[layout.deformation_gain_slice] = 2.0
    std[layout.deformation_gain_slice] = 1.5
    mean[layout.log_shape_frequency_slice] = math.log(35.0)
    std[layout.log_shape_frequency_slice] = 0.6
    mean[layout.shape_damping_ratio_slice] = 0.4
    std[layout.shape_damping_ratio_slice] = 0.35
    mean[layout.drag_slice] = 0.05
    std[layout.drag_slice] = 0.15
    return mean, std


def minimum_distribution_std(layout: ParameterLayout) -> np.ndarray:
    result = np.full(layout.size, 0.005, dtype=np.float32)
    result[layout.velocity_slice] = 0.01
    result[layout.log_gravity_scale_slice] = 0.005
    result[layout.log_radius_scale_slice] = 0.003
    result[layout.log_frequency_slice] = 0.01
    result[layout.damping_ratio_slice] = 0.01
    result[layout.friction_slice] = 0.01
    result[layout.deformation_gain_slice] = 0.01
    result[layout.log_shape_frequency_slice] = 0.01
    result[layout.shape_damping_ratio_slice] = 0.01
    result[layout.drag_slice] = 0.01
    return result


def project_parameters(values: np.ndarray, layout: ParameterLayout) -> np.ndarray:
    lower, upper = parameter_bounds(layout)
    return np.clip(values, lower, upper)


def decode_parameters(values: np.ndarray, layout: ParameterLayout) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "initial_velocity": values[layout.velocity_slice].reshape(-1, 2),
        "gravity_scale": np.exp(values[layout.log_gravity_scale_slice]),
        "radius_scale": np.exp(values[layout.log_radius_scale_slice]),
        "natural_frequency_rad_per_s": np.exp(values[layout.log_frequency_slice]),
        "damping_ratio": values[layout.damping_ratio_slice],
        "contact_friction": values[layout.friction_slice],
        "deformation_gain": values[layout.deformation_gain_slice],
        "shape_frequency_rad_per_s": np.exp(
            values[layout.log_shape_frequency_slice]
        ),
        "shape_damping_ratio": values[layout.shape_damping_ratio_slice],
        "linear_drag_per_s": values[layout.drag_slice],
    }


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ValueError(f"invalid {name}")
    return value / norm


def _wall_basis(reconstruction_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = reconstruction_dir / "surface/gravity_estimate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected") or {}
    down = _unit(np.asarray(selected["gravity_down_world"], dtype=np.float64), "gravity")
    normal = _unit(
        np.asarray(selected["wall_plane"]["normal_world"], dtype=np.float64),
        "wall normal",
    )
    down = _unit(down - np.dot(down, normal) * normal, "in-wall gravity")
    lateral = _unit(np.cross(normal, down), "wall lateral")
    return lateral, down, normal


def _project(points: np.ndarray, lateral: np.ndarray, down: np.ndarray) -> np.ndarray:
    return np.column_stack((points @ lateral, points @ down))


def _weighted_moments(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1.0e-4)
    weights /= np.sum(weights)
    center = np.sum(points * weights[:, None], axis=0)
    centered = points - center
    covariance = np.einsum("ni,nj,n->ij", centered, centered, weights)
    return center, covariance


def covariance_deformation(
    reference: np.ndarray, target: np.ndarray,
) -> np.ndarray:
    """Return the symmetric map transporting one point-cloud covariance to another."""

    scale = max(float(np.trace(reference)), float(np.trace(target)), 1.0e-8)
    regularizer = np.eye(2) * (1.0e-6 * scale)

    def square_root(matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return (
            eigenvectors * np.sqrt(np.maximum(eigenvalues, 1.0e-12))
        ) @ eigenvectors.T

    reference_root = square_root(reference + regularizer)
    reference_inverse_root = np.linalg.inv(reference_root)
    middle = reference_root @ (target + regularizer) @ reference_root
    return (
        reference_inverse_root
        @ square_root(middle)
        @ reference_inverse_root
    )


def _mask_plane_points(
    mask: np.ndarray, intrinsic: np.ndarray, camera_to_world: np.ndarray,
    plane_normal: np.ndarray, plane_depth: float,
    lateral: np.ndarray, down: np.ndarray,
) -> np.ndarray:
    """Intersect mask-pixel rays with a plane parallel to the interaction wall."""

    pixel_y, pixel_x = np.nonzero(mask)
    if len(pixel_x) < MIN_TRACKED_POINTS:
        return np.empty((0, 2), dtype=np.float64)
    pixels = np.column_stack(
        (pixel_x.astype(np.float64) + 0.5, pixel_y.astype(np.float64) + 0.5,
         np.ones(len(pixel_x), dtype=np.float64))
    )
    rays_camera = pixels @ np.linalg.inv(intrinsic).T
    rotation = camera_to_world[:3, :3]
    origin = camera_to_world[:3, 3]
    rays_world = rays_camera @ rotation.T
    denominator = rays_world @ plane_normal
    valid = np.abs(denominator) > 1.0e-8
    distance = np.zeros_like(denominator)
    distance[valid] = (
        plane_depth - float(origin @ plane_normal)
    ) / denominator[valid]
    valid &= np.isfinite(distance) & (distance > 0.0)
    if np.count_nonzero(valid) < MIN_TRACKED_POINTS:
        return np.empty((0, 2), dtype=np.float64)
    world = origin + rays_world[valid] * distance[valid, None]
    return _project(world, lateral, down)


def _silhouette_deformations(
    *, mask_path: Path, geometry_dir: Path, reference_frame: int,
    frame_count: int, plane_normal: np.ndarray, plane_depth: float,
    lateral: np.ndarray, down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not mask_path.is_file():
        return None
    intrinsic_path = geometry_dir / "intrinsics.npy"
    pose_path = geometry_dir / "camera_poses.npy"
    if not intrinsic_path.is_file() or not pose_path.is_file():
        return None
    with np.load(mask_path) as archive:
        masks = np.asarray(archive["masks"], dtype=bool)
    intrinsics = np.load(intrinsic_path, mmap_mode="r")
    camera_poses = np.load(pose_path, mmap_mode="r")
    if (
        masks.shape[0] != frame_count
        or reference_frame < 0
        or reference_frame >= frame_count
        or (intrinsics.ndim == 3 and len(intrinsics) < frame_count)
        or (camera_poses.ndim == 3 and len(camera_poses) < frame_count)
    ):
        return None

    def frame_value(values: np.ndarray, frame_index: int) -> np.ndarray:
        return np.asarray(values[frame_index] if values.ndim == 3 else values)

    reference_points = _mask_plane_points(
        masks[reference_frame], frame_value(intrinsics, reference_frame),
        frame_value(camera_poses, reference_frame), plane_normal, plane_depth,
        lateral, down,
    )
    if len(reference_points) < MIN_TRACKED_POINTS:
        return None
    _, reference_covariance = _weighted_moments(
        reference_points, np.ones(len(reference_points))
    )
    result = np.broadcast_to(np.eye(2), (frame_count, 2, 2)).copy()
    valid = np.zeros(frame_count, dtype=bool)
    visibility_loss = np.zeros(frame_count, dtype=bool)
    for frame_index, mask in enumerate(masks):
        if (
            np.any(mask[0]) or np.any(mask[-1])
            or np.any(mask[:, 0]) or np.any(mask[:, -1])
        ):
            continue
        points = _mask_plane_points(
            mask, frame_value(intrinsics, frame_index),
            frame_value(camera_poses, frame_index), plane_normal, plane_depth,
            lateral, down,
        )
        if len(points) < MIN_TRACKED_POINTS:
            continue
        _, covariance = _weighted_moments(points, np.ones(len(points)))
        deformation = covariance_deformation(reference_covariance, covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(deformation)
        eigenvalues = np.clip(eigenvalues, 0.45, 1.8)
        result[frame_index] = (eigenvectors * eigenvalues) @ eigenvectors.T
        # A large area loss without transverse expansion is usually partial
        # visibility, not a physically observable in-plane deformation.
        area_scale = float(np.prod(eigenvalues))
        visibility_loss[frame_index] = (
            area_scale < 0.65 and float(eigenvalues[-1]) < 1.12
        )
        valid[frame_index] = not visibility_loss[frame_index]
    return result, valid, visibility_loss


def _target_contact_mask(
    target_centers: np.ndarray, observed_mask: np.ndarray, radii: np.ndarray,
    obstacles: tuple[StaticObstacle, ...],
) -> np.ndarray:
    """Find target frames geometrically compatible with 2D contact."""
    frame_count, object_count, _ = target_centers.shape
    contact = np.zeros((frame_count, object_count), dtype=bool)
    observed = observed_mask > 0.0
    for left in range(object_count):
        for right in range(left + 1, object_count):
            distance = np.linalg.norm(
                target_centers[:, left] - target_centers[:, right], axis=-1
            )
            pair_contact = (
                observed[:, left]
                & observed[:, right]
                & (distance <= radii[left] + radii[right])
            )
            contact[:, left] |= pair_contact
            contact[:, right] |= pair_contact
    for object_index in range(object_count):
        centers = target_centers[:, object_index]
        for obstacle in obstacles:
            axis_x = np.asarray(obstacle.axis_x, dtype=np.float64)
            axis_y = np.asarray([-axis_x[1], axis_x[0]])
            offset = centers - np.asarray(obstacle.center, dtype=np.float64)
            local = np.column_stack((offset @ axis_x, offset @ axis_y))
            excess = np.maximum(
                np.abs(local) - np.asarray(obstacle.half_extents), 0.0
            )
            distance = np.linalg.norm(excess, axis=-1)
            contact[:, object_index] |= (
                observed[:, object_index] & (distance <= radii[object_index])
            )
    return contact


def _obb_signed_distance_and_normal(
    point: np.ndarray, obstacle: StaticObstacle,
) -> tuple[float, np.ndarray]:
    axis_x = np.asarray(obstacle.axis_x, dtype=np.float64)
    axis_y = np.asarray([-axis_x[1], axis_x[0]])
    offset = np.asarray(point, dtype=np.float64) - np.asarray(obstacle.center)
    local = np.asarray([offset @ axis_x, offset @ axis_y])
    distance = np.abs(local) - np.asarray(obstacle.half_extents)
    outside = np.maximum(distance, 0.0)
    outside_length = float(np.linalg.norm(outside))
    if outside_length > 1.0e-8:
        local_normal = np.sign(local) * outside / outside_length
    elif distance[0] > distance[1]:
        local_normal = np.asarray([1.0 if local[0] >= 0.0 else -1.0, 0.0])
    else:
        local_normal = np.asarray([0.0, 1.0 if local[1] >= 0.0 else -1.0])
    normal = local_normal[0] * axis_x + local_normal[1] * axis_y
    signed_distance = outside_length + min(float(np.max(distance)), 0.0)
    return signed_distance, normal


def _target_penetration_allowances(
    scene: SoftBodySceneData,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure proxy overlap already present in reliable tracked targets."""
    disabled = np.float32(1.0e6)
    static = np.full(
        (scene.frame_count, scene.object_count, len(scene.obstacles)),
        disabled,
        dtype=np.float32,
    )
    pair = np.full(
        (scene.frame_count, scene.object_count, scene.object_count),
        disabled,
        dtype=np.float32,
    )
    shape_valid = scene.shape_observed_mask > 0.0
    for frame_index in range(scene.frame_count):
        for object_index in range(scene.object_count):
            if not shape_valid[frame_index, object_index]:
                continue
            center = scene.target_centers[frame_index, object_index]
            deformation = scene.target_deformations[frame_index, object_index]
            radius = scene.radii[object_index]
            for obstacle_index, obstacle in enumerate(scene.obstacles):
                signed_distance, normal = _obb_signed_distance_and_normal(
                    center, obstacle
                )
                support = radius * np.linalg.norm(deformation.T @ normal)
                static[frame_index, object_index, obstacle_index] = max(
                    float(support - signed_distance), 0.0
                )
        for first_index in range(scene.object_count):
            if not shape_valid[frame_index, first_index]:
                continue
            for second_index in range(first_index + 1, scene.object_count):
                if not shape_valid[frame_index, second_index]:
                    continue
                delta = (
                    scene.target_centers[frame_index, first_index]
                    - scene.target_centers[frame_index, second_index]
                )
                distance = float(np.linalg.norm(delta))
                if distance <= 1.0e-8:
                    continue
                normal = delta / distance
                first_support = scene.radii[first_index] * np.linalg.norm(
                    scene.target_deformations[frame_index, first_index].T @ normal
                )
                second_support = scene.radii[second_index] * np.linalg.norm(
                    scene.target_deformations[frame_index, second_index].T @ normal
                )
                allowance = max(
                    float(first_support + second_support - distance), 0.0
                )
                pair[frame_index, first_index, second_index] = allowance
                pair[frame_index, second_index, first_index] = allowance
    return static, pair


def _transient_contact_recovery_mask(
    visibility_loss_mask: np.ndarray, shape_observed_mask: np.ndarray,
    observed_mask: np.ndarray, target_centers: np.ndarray, radii: np.ndarray,
    obstacles: tuple[StaticObstacle, ...], fps: float,
) -> np.ndarray:
    """Recover short, bracketed 2D compression events from visibility filtering."""
    candidate = visibility_loss_mask & (observed_mask > 0.0)
    contact = _target_contact_mask(
        target_centers, observed_mask, radii, obstacles
    )
    recovered = np.zeros_like(candidate)
    maximum_frames = max(1, int(round(MAX_TRANSIENT_CONTACT_GAP_SECONDS * fps)))
    frame_count, object_count = candidate.shape
    for object_index in range(object_count):
        episode = candidate[:, object_index].copy()
        candidate_frames = np.flatnonzero(episode)
        for left, right in zip(candidate_frames[:-1], candidate_frames[1:]):
            if right - left - 1 <= MAX_VISIBILITY_FLICKER_GAP_FRAMES:
                episode[left:right + 1] = True
        frame_index = 0
        while frame_index < frame_count:
            if not episode[frame_index]:
                frame_index += 1
                continue
            start = frame_index
            while (
                frame_index + 1 < frame_count
                and episode[frame_index + 1]
            ):
                frame_index += 1
            end = frame_index
            bracketed = (
                start > 0
                and end + 1 < frame_count
                and shape_observed_mask[start - 1, object_index] > 0.0
                and shape_observed_mask[end + 1, object_index] > 0.0
            )
            short = end - start + 1 <= maximum_frames
            contact_supported = bool(np.any(contact[start:end + 1, object_index]))
            if bracketed and short and contact_supported:
                recovered[start:end + 1, object_index] = candidate[
                    start:end + 1, object_index
                ]
            frame_index += 1
    return recovered


def _load_static_obstacles(
    tracking_dir: Path, lateral: np.ndarray, down: np.ndarray,
) -> tuple[StaticObstacle, ...]:
    result = []
    for path in sorted((tracking_dir / "rigid_trajectories").glob("*.npz")):
        with np.load(path) as archive:
            motion_model = str(np.asarray(archive["motion_model"]).item())
            if motion_model != "static":
                continue
            points = np.asarray(archive["canonical_tracking_points"], dtype=np.float64)
            poses = np.asarray(archive["poses"], dtype=np.float64)
        valid = np.flatnonzero(np.isfinite(poses).all(axis=(1, 2)))
        if not len(valid) or len(points) < 4:
            continue
        pose = poses[int(valid[0])]
        world = points @ pose[:3, :3].T + pose[:3, 3]
        plane = _project(world, lateral, down).astype(np.float32)
        rectangle = cv2.minAreaRect(plane)
        corners = cv2.boxPoints(rectangle)
        edge_x = corners[1] - corners[0]
        edge_y = corners[3] - corners[0]
        length_x = float(np.linalg.norm(edge_x))
        length_y = float(np.linalg.norm(edge_y))
        if min(length_x, length_y) < 1.0e-4:
            continue
        axis_x = edge_x / length_x
        result.append(
            StaticObstacle(
                object_id=path.stem,
                center=tuple(float(value) for value in corners.mean(axis=0)),
                axis_x=tuple(float(value) for value in axis_x),
                half_extents=(0.5 * length_x, 0.5 * length_y),
            )
        )
    if not result:
        raise ValueError(f"no static rigid obstacle trajectories in {tracking_dir}")
    return tuple(result)


def load_softbody_scene(
    tracking_dir: Path, *, reconstruction_dir: Path | None = None,
    minimum_tracking_confidence: float = 0.2,
) -> tuple[SoftBodySceneData, dict[str, Any]]:
    tracking_dir = tracking_dir.resolve()
    if not (tracking_dir / "COMPLETED").is_file():
        raise ValueError(f"tracking run is incomplete: {tracking_dir}")
    if reconstruction_dir is None:
        reconstruction_dir = tracking_dir.parent / "reconstruction"
    reconstruction_dir = reconstruction_dir.resolve()
    metadata_path = tracking_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fps = float(metadata["fps"])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid tracking fps: {fps!r}")
    lateral, down, normal = _wall_basis(reconstruction_dir)

    paths = []
    for path in sorted((tracking_dir / "trajectories").glob("*.npz")):
        canonical_path = tracking_dir / "canonical" / path.name
        if not canonical_path.is_file():
            continue
        with np.load(canonical_path) as canonical:
            physical_type = str(np.asarray(canonical["physical_type"]).item())
        if physical_type == "soft_body":
            paths.append(path)
    if not paths:
        raise ValueError(f"no soft-body trajectories found in {tracking_dir}")

    frame_count = None
    targets = []
    for path in paths:
        with np.load(path) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        xyz = payload["model_xyz"]
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"invalid model_xyz in {path}: {xyz.shape}")
        if frame_count is None:
            frame_count = len(xyz)
        elif len(xyz) != frame_count:
            raise ValueError("soft-body trajectory frame counts do not match")
        targets.append(payload)
    assert frame_count is not None

    object_count = len(paths)
    target_centers = np.zeros((frame_count, object_count, 2), dtype=np.float64)
    target_subset_centers = np.zeros_like(target_centers)
    reference_subset_offsets = np.zeros_like(target_centers)
    target_covariances = np.zeros((frame_count, object_count, 2, 2), dtype=np.float64)
    reference_covariances = np.zeros_like(target_covariances)
    target_deformations = np.broadcast_to(
        np.eye(2), (frame_count, object_count, 2, 2)
    ).copy()
    observed_mask = np.zeros((frame_count, object_count), dtype=np.float64)
    shape_observed_mask = np.zeros_like(observed_mask)
    shape_contact_recovered_mask = np.zeros_like(observed_mask)
    initial_centers = []
    radii = []
    wall_depths = []
    first_offsets = []
    valid_frame_counts = []
    point_counts = []
    geometry_directories = sorted(
        (reconstruction_dir / "geometry").glob("reference_*")
    )
    geometry_dir = next(
        (
            path for path in geometry_directories
            if (path / "intrinsics.npy").is_file()
            and (path / "camera_poses.npy").is_file()
        ),
        None,
    )
    used_silhouette_targets = False
    silhouette_candidates: list[np.ndarray | None] = []
    silhouette_visibility_losses: list[np.ndarray | None] = []

    for object_index, (path, payload) in enumerate(zip(paths, targets)):
        xyz = np.asarray(payload["model_xyz"], dtype=np.float64)
        confidence = np.asarray(payload["tracking_confidence"], dtype=np.float64)
        border = np.asarray(payload["border_suppressed"], dtype=bool)
        reference_frame = int(np.asarray(payload["reference_frame"]).item())
        if confidence.shape != xyz.shape[:2] or border.shape != (frame_count,):
            raise ValueError(f"invalid trajectory quality arrays in {path}")
        reference_world = xyz[reference_frame]
        reference_plane = _project(reference_world, lateral, down)
        reference_valid = np.isfinite(reference_world).all(axis=1)
        if np.count_nonzero(reference_valid) < MIN_TRACKED_POINTS:
            raise ValueError(f"insufficient reference points in {path}")
        reference_center = np.mean(reference_plane[reference_valid], axis=0)
        reference_depth = float(np.mean(reference_world[reference_valid] @ normal))
        centered_reference = reference_plane[reference_valid] - reference_center
        radial_distances = np.linalg.norm(centered_reference, axis=1)
        radius = float(np.percentile(radial_distances, 95.0))
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"invalid soft-body radius in {path}")

        valid_offsets = []
        for frame_index in range(frame_count):
            valid = (
                np.isfinite(xyz[frame_index]).all(axis=1)
                & reference_valid
                & np.isfinite(confidence[frame_index])
                & (confidence[frame_index] >= minimum_tracking_confidence)
            )
            if border[frame_index] or np.count_nonzero(valid) < MIN_TRACKED_POINTS:
                continue
            weights = confidence[frame_index, valid]
            current_plane = _project(xyz[frame_index, valid], lateral, down)
            reference_subset = reference_plane[valid]
            current_center, current_covariance = _weighted_moments(current_plane, weights)
            reference_subset_center, reference_covariance = _weighted_moments(
                reference_subset, weights
            )
            corrected_center = reference_center + current_center - reference_subset_center
            target_centers[frame_index, object_index] = corrected_center
            target_subset_centers[frame_index, object_index] = current_center
            reference_subset_offsets[frame_index, object_index] = (
                reference_subset_center - reference_center
            )
            target_covariances[frame_index, object_index] = current_covariance
            reference_covariances[frame_index, object_index] = reference_covariance
            target_deformations[frame_index, object_index] = covariance_deformation(
                reference_covariance, current_covariance
            )
            observed_mask[frame_index, object_index] = 1.0
            shape_observed_mask[frame_index, object_index] = 1.0
            valid_offsets.append(frame_index)
        if len(valid_offsets) < 2:
            raise ValueError(f"fewer than two usable soft-body frames for {path.stem}")
        first_offset = int(valid_offsets[0])
        initial_centers.append(target_centers[first_offset, object_index])
        first_offsets.append(first_offset)
        valid_frame_counts.append(len(valid_offsets))
        point_counts.append(xyz.shape[1])
        radii.append(radius)
        wall_depths.append(reference_depth)

        if geometry_dir is not None:
            silhouette = _silhouette_deformations(
                mask_path=reconstruction_dir / "masks" / path.name,
                geometry_dir=geometry_dir,
                reference_frame=reference_frame,
                frame_count=frame_count,
                plane_normal=normal,
                plane_depth=reference_depth,
                lateral=lateral,
                down=down,
            )
            if silhouette is not None:
                (
                    silhouette_deformation,
                    silhouette_valid,
                    silhouette_visibility_loss,
                ) = silhouette
                shape_observed_mask[:, object_index] = 0.0
                replace = silhouette_valid & (observed_mask[:, object_index] > 0.0)
                target_deformations[replace, object_index] = (
                    silhouette_deformation[replace]
                )
                used_silhouette_targets |= bool(np.any(replace))
                shape_observed_mask[replace, object_index] = 1.0
                silhouette_candidates.append(silhouette_deformation)
                silhouette_visibility_losses.append(silhouette_visibility_loss)
            else:
                silhouette_candidates.append(None)
                silhouette_visibility_losses.append(None)
        else:
            silhouette_candidates.append(None)
            silhouette_visibility_losses.append(None)

    obstacles = _load_static_obstacles(tracking_dir, lateral, down)
    visibility_loss_mask = np.zeros_like(observed_mask, dtype=bool)
    for object_index, visibility_loss in enumerate(silhouette_visibility_losses):
        if visibility_loss is not None:
            visibility_loss_mask[:, object_index] = visibility_loss
    recovered = _transient_contact_recovery_mask(
        visibility_loss_mask,
        shape_observed_mask,
        observed_mask,
        target_centers,
        np.asarray(radii, dtype=np.float64),
        obstacles,
        fps,
    )
    for object_index, candidate_deformation in enumerate(silhouette_candidates):
        if candidate_deformation is None:
            continue
        replace = recovered[:, object_index]
        target_deformations[replace, object_index] = candidate_deformation[replace]
    shape_observed_mask[recovered] = 1.0
    shape_contact_recovered_mask[recovered] = 1.0

    scene = SoftBodySceneData(
        object_ids=tuple(path.stem for path in paths),
        frames=tuple(range(frame_count)),
        fps=fps,
        minimum_tracking_confidence=float(minimum_tracking_confidence),
        target_centers=target_centers,
        target_subset_centers=target_subset_centers,
        reference_subset_offsets=reference_subset_offsets,
        target_covariances=target_covariances,
        reference_subset_covariances=reference_covariances,
        target_deformations=target_deformations,
        shape_target_source=(
            "wall_plane_silhouette" if used_silhouette_targets else "point_cloud"
        ),
        shape_observed_mask=shape_observed_mask,
        shape_contact_recovered_mask=shape_contact_recovered_mask,
        observed_mask=observed_mask,
        first_offsets=tuple(first_offsets),
        initial_centers=np.asarray(initial_centers, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        wall_lateral_world=lateral,
        gravity_down_world=down,
        wall_normal_world=normal,
        wall_depths=np.asarray(wall_depths, dtype=np.float64),
        trajectory_paths=tuple(str(path.resolve()) for path in paths),
        valid_frame_counts=tuple(valid_frame_counts),
        point_counts=tuple(point_counts),
        obstacles=obstacles,
    )
    source = {
        "tracking_dir": str(tracking_dir),
        "run_metadata": str(metadata_path.resolve()),
        "reconstruction_dir": str(reconstruction_dir),
    }
    return scene, source


@wp.func
def _obb_contact(
    point: wp.vec2, center: wp.vec2, axis_x: wp.vec2, half: wp.vec2,
) -> wp.vec3:
    axis_y = wp.vec2(-axis_x[1], axis_x[0])
    delta = point - center
    q = wp.vec2(wp.dot(delta, axis_x), wp.dot(delta, axis_y))
    distance = wp.vec2(wp.abs(q[0]) - half[0], wp.abs(q[1]) - half[1])
    outside = wp.vec2(wp.max(distance[0], 0.0), wp.max(distance[1], 0.0))
    outside_length = wp.length(outside)
    local_normal = wp.vec2(0.0, 0.0)
    if outside_length > 1.0e-8:
        sign_x = 1.0 if q[0] >= 0.0 else -1.0
        sign_y = 1.0 if q[1] >= 0.0 else -1.0
        local_normal = wp.vec2(
            sign_x * outside[0] / outside_length,
            sign_y * outside[1] / outside_length,
        )
    elif distance[0] > distance[1]:
        local_normal = wp.vec2(1.0 if q[0] >= 0.0 else -1.0, 0.0)
    else:
        local_normal = wp.vec2(0.0, 1.0 if q[1] >= 0.0 else -1.0)
    normal = local_normal[0] * axis_x + local_normal[1] * axis_y
    signed_distance = outside_length + wp.min(wp.max(distance[0], distance[1]), 0.0)
    return wp.vec3(normal[0], normal[1], signed_distance)


@wp.func
def _contact_strain(normal: wp.vec2, amplitude: float) -> wp.vec3:
    poisson = float(0.5)
    return wp.vec3(
        amplitude * (
            poisson * normal[1] * normal[1] - normal[0] * normal[0]
        ),
        amplitude * (
            poisson * normal[0] * normal[0] - normal[1] * normal[1]
        ),
        -amplitude * (1.0 + poisson) * normal[0] * normal[1],
    )


@wp.func
def _deformed_support_radius(
    radius: float, strain: wp.vec3, normal: wp.vec2,
) -> float:
    transformed = wp.vec2(
        (1.0 + strain[0]) * normal[0] + strain[2] * normal[1],
        strain[2] * normal[0] + (1.0 + strain[1]) * normal[1],
    )
    return radius * wp.length(transformed)


@wp.func
def _project_strain_support(
    radius: float, strain: wp.vec3, normal: wp.vec2, maximum_support: float,
) -> wp.vec3:
    support = _deformed_support_radius(radius, strain, normal)
    penetration = support - maximum_support
    if penetration > 0.0:
        strain += _contact_strain(
            normal, wp.clamp(penetration / radius, 0.0, 0.7)
        )
    return strain


@wp.kernel
def _simulate_soft_worlds(
    theta: wp.array2d(dtype=float),
    initial_centers: wp.array(dtype=wp.vec2),
    initial_strains: wp.array(dtype=wp.vec3),
    initial_strain_velocities: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    first_offsets: wp.array(dtype=int),
    obstacle_centers: wp.array(dtype=wp.vec2),
    obstacle_axes: wp.array(dtype=wp.vec2),
    obstacle_half_extents: wp.array(dtype=wp.vec2),
    static_penetration_allowance: wp.array3d(dtype=float),
    pair_penetration_allowance: wp.array3d(dtype=float),
    object_count: int,
    obstacle_count: int,
    frame_count: int,
    fps: float,
    substeps: int,
    log_gravity_scale_start: int,
    log_radius_scale_start: int,
    log_frequency_start: int,
    damping_start: int,
    friction_start: int,
    deformation_gain_start: int,
    log_shape_frequency_start: int,
    shape_damping_start: int,
    drag_start: int,
    enable_soft_body_contacts: int,
    enable_shape_projection: int,
    deformable_contacts: int,
    separating_normal_damping: int,
    positions: wp.array2d(dtype=wp.vec2),
    velocities: wp.array2d(dtype=wp.vec2),
    strains: wp.array2d(dtype=wp.vec3),
    strain_velocities: wp.array2d(dtype=wp.vec3),
    accelerations: wp.array2d(dtype=wp.vec2),
    target_strains: wp.array2d(dtype=wp.vec3),
    contact_strengths: wp.array2d(dtype=float),
    predicted_positions: wp.array3d(dtype=wp.vec2),
    predicted_strains: wp.array3d(dtype=wp.vec3),
    contact_obstacle: wp.array3d(dtype=int),
):
    world = wp.tid()
    dt = 1.0 / (fps * float(substeps))
    for object_index in range(object_count):
        velocity_offset = 2 * object_index
        positions[world, object_index] = initial_centers[object_index]
        velocities[world, object_index] = wp.vec2(
            theta[world, velocity_offset], theta[world, velocity_offset + 1]
        )
        strains[world, object_index] = initial_strains[object_index]
        strain_velocities[world, object_index] = initial_strain_velocities[object_index]

    for frame_index in range(frame_count):
        for object_index in range(object_count):
            contact_obstacle[world, frame_index, object_index] = -1
        if frame_index > 0:
            for _substep in range(substeps):
                for object_index in range(object_count):
                    accelerations[world, object_index] = wp.vec2(0.0, 0.0)
                    target_strains[world, object_index] = wp.vec3(0.0, 0.0, 0.0)
                    contact_strengths[world, object_index] = 0.0
                    if frame_index <= first_offsets[object_index]:
                        continue
                    position = positions[world, object_index]
                    velocity = velocities[world, object_index]
                    radius = radii[object_index] * wp.exp(
                        theta[world, log_radius_scale_start + object_index]
                    )
                    frequency = wp.exp(
                        theta[world, log_frequency_start + object_index]
                    )
                    damping = theta[world, damping_start + object_index]
                    friction = theta[world, friction_start + object_index]
                    drag = theta[world, drag_start + object_index]
                    gravity_scale = wp.exp(
                        theta[world, log_gravity_scale_start + object_index]
                    )
                    acceleration = (
                        wp.vec2(0.0, float(GRAVITY_M_PER_S2) * gravity_scale)
                        - drag * velocity
                    )
                    target_strain = wp.vec3(0.0, 0.0, 0.0)
                    for obstacle_index in range(obstacle_count):
                        geometry = _obb_contact(
                            position,
                            obstacle_centers[obstacle_index],
                            obstacle_axes[obstacle_index],
                            obstacle_half_extents[obstacle_index],
                        )
                        normal = wp.vec2(geometry[0], geometry[1])
                        support = radius
                        if deformable_contacts != 0:
                            support = _deformed_support_radius(radius, strains[world, object_index], normal)
                        penetration = support - geometry[2]
                        if penetration <= 0.0:
                            continue
                        contact_strength = penetration / radius
                        if contact_strength > contact_strengths[world, object_index]:
                            contact_strengths[world, object_index] = contact_strength
                            contact_obstacle[world, frame_index, object_index] = (
                                obstacle_index
                            )
                        normal_speed = wp.dot(velocity, normal)
                        normal_acceleration = frequency * frequency * penetration
                        if normal_speed < 0.0 or separating_normal_damping != 0:
                            normal_acceleration -= 2.0 * damping * frequency * normal_speed
                        normal_acceleration = wp.clamp(normal_acceleration, 0.0, 2000.0)
                        acceleration += normal_acceleration * normal
                        tangent = wp.vec2(-normal[1], normal[0])
                        tangent_speed = wp.dot(velocity, tangent)
                        friction_acceleration = wp.min(
                            wp.abs(tangent_speed) / dt,
                            friction * normal_acceleration,
                        )
                        if tangent_speed > 0.0:
                            acceleration -= friction_acceleration * tangent
                        elif tangent_speed < 0.0:
                            acceleration += friction_acceleration * tangent
                        compression = wp.clamp(
                            normal_acceleration
                            / (frequency * frequency * radius),
                            0.0,
                            0.7,
                        )
                        deformation_gain = theta[
                            world, deformation_gain_start + object_index
                        ]
                        amplitude = deformation_gain * compression
                        target_strain += _contact_strain(normal, amplitude)
                    accelerations[world, object_index] = acceleration
                    target_strains[world, object_index] = target_strain

                for first_index in range(object_count):
                    if enable_soft_body_contacts == 0:
                        continue
                    if frame_index <= first_offsets[first_index]:
                        continue
                    for second_index in range(first_index + 1, object_count):
                        if frame_index <= first_offsets[second_index]:
                            continue
                        first_position = positions[world, first_index]
                        second_position = positions[world, second_index]
                        delta = first_position - second_position
                        distance = wp.length(delta)
                        normal = wp.vec2(1.0, 0.0)
                        if distance > 1.0e-8:
                            normal = delta / distance
                        first_radius = radii[first_index] * wp.exp(
                            theta[world, log_radius_scale_start + first_index]
                        )
                        second_radius = radii[second_index] * wp.exp(
                            theta[world, log_radius_scale_start + second_index]
                        )
                        support = first_radius + second_radius
                        if deformable_contacts != 0:
                            support = _deformed_support_radius(first_radius, strains[world, first_index], normal) + _deformed_support_radius(second_radius, strains[world, second_index], normal)
                        penetration = support - distance
                        if penetration <= 0.0:
                            continue
                        first_frequency = wp.exp(
                            theta[world, log_frequency_start + first_index]
                        )
                        second_frequency = wp.exp(
                            theta[world, log_frequency_start + second_index]
                        )
                        frequency = wp.sqrt(first_frequency * second_frequency)
                        damping = 0.5 * (
                            theta[world, damping_start + first_index]
                            + theta[world, damping_start + second_index]
                        )
                        relative_velocity = (
                            velocities[world, first_index]
                            - velocities[world, second_index]
                        )
                        normal_speed = wp.dot(relative_velocity, normal)
                        pair_acceleration = 0.5 * frequency * frequency * penetration
                        if normal_speed < 0.0 or separating_normal_damping != 0:
                            pair_acceleration -= damping * frequency * normal_speed
                        pair_acceleration = wp.clamp(
                            pair_acceleration, 0.0, 2000.0
                        )
                        accelerations[world, first_index] = (
                            accelerations[world, first_index]
                            + pair_acceleration * normal
                        )
                        accelerations[world, second_index] = (
                            accelerations[world, second_index]
                            - pair_acceleration * normal
                        )
                        tangent = wp.vec2(-normal[1], normal[0])
                        tangent_speed = wp.dot(relative_velocity, tangent)
                        friction = 0.5 * (
                            theta[world, friction_start + first_index]
                            + theta[world, friction_start + second_index]
                        )
                        friction_acceleration = wp.min(
                            0.5 * wp.abs(tangent_speed) / dt,
                            friction * pair_acceleration,
                        )
                        if tangent_speed > 0.0:
                            accelerations[world, first_index] -= (
                                friction_acceleration * tangent
                            )
                            accelerations[world, second_index] += (
                                friction_acceleration * tangent
                            )
                        elif tangent_speed < 0.0:
                            accelerations[world, first_index] += (
                                friction_acceleration * tangent
                            )
                            accelerations[world, second_index] -= (
                                friction_acceleration * tangent
                            )
                        first_compression = wp.clamp(
                            pair_acceleration
                            / (
                                0.5 * frequency * frequency * first_radius
                            ),
                            0.0,
                            0.7,
                        )
                        second_compression = wp.clamp(
                            pair_acceleration
                            / (
                                0.5 * frequency * frequency * second_radius
                            ),
                            0.0,
                            0.7,
                        )
                        first_amplitude = (
                            theta[world, deformation_gain_start + first_index]
                            * first_compression
                        )
                        second_amplitude = (
                            theta[world, deformation_gain_start + second_index]
                            * second_compression
                        )
                        target_strains[world, first_index] = (
                            target_strains[world, first_index]
                            + _contact_strain(normal, first_amplitude)
                        )
                        target_strains[world, second_index] = (
                            target_strains[world, second_index]
                            + _contact_strain(-normal, second_amplitude)
                        )
                        first_strength = 0.5 * penetration / first_radius
                        second_strength = 0.5 * penetration / second_radius
                        if first_strength > contact_strengths[world, first_index]:
                            contact_strengths[world, first_index] = first_strength
                            contact_obstacle[world, frame_index, first_index] = (
                                obstacle_count + second_index
                            )
                        if second_strength > contact_strengths[world, second_index]:
                            contact_strengths[world, second_index] = second_strength
                            contact_obstacle[world, frame_index, second_index] = (
                                obstacle_count + first_index
                            )

                for object_index in range(object_count):
                    if frame_index <= first_offsets[object_index]:
                        continue
                    position = positions[world, object_index]
                    velocity = velocities[world, object_index]
                    strain = strains[world, object_index]
                    strain_velocity = strain_velocities[world, object_index]
                    shape_frequency = wp.exp(
                        theta[world, log_shape_frequency_start + object_index]
                    )
                    shape_damping = theta[
                        world, shape_damping_start + object_index
                    ]
                    velocity += accelerations[world, object_index] * dt
                    position += velocity * dt
                    strain_acceleration = (
                        shape_frequency * shape_frequency
                        * (target_strains[world, object_index] - strain)
                        - 2.0 * shape_damping * shape_frequency * strain_velocity
                    )
                    strain_velocity += strain_acceleration * dt
                    strain += strain_velocity * dt
                    strain_norm = wp.length(strain)
                    if strain_norm > 0.7:
                        strain *= 0.7 / strain_norm
                        strain_velocity *= 0.5
                    positions[world, object_index] = position
                    velocities[world, object_index] = velocity
                    strains[world, object_index] = strain
                    strain_velocities[world, object_index] = strain_velocity

                # Soft bodies accommodate geometric overlap through strain rather
                # than by moving observed centers away from their fitted paths.
                for _projection_iteration in range(2):
                    if enable_shape_projection == 0:
                        continue
                    for object_index in range(object_count):
                        if frame_index <= first_offsets[object_index]:
                            continue
                        position = positions[world, object_index]
                        strain = strains[world, object_index]
                        radius = radii[object_index]
                        for obstacle_index in range(obstacle_count):
                            geometry = _obb_contact(
                                position,
                                obstacle_centers[obstacle_index],
                                obstacle_axes[obstacle_index],
                                obstacle_half_extents[obstacle_index],
                            )
                            normal = wp.vec2(geometry[0], geometry[1])
                            minimum_support = float(MIN_DEFORMED_SUPPORT_SCALE) * radius
                            support = _deformed_support_radius(radius, strain, normal)
                            maximum_support = wp.max(
                                geometry[2]
                                + static_penetration_allowance[
                                    frame_index, object_index, obstacle_index
                                ],
                                minimum_support,
                            )
                            penetration = support - maximum_support
                            if penetration <= 0.0:
                                continue
                            strain = _project_strain_support(
                                radius,
                                strain,
                                normal,
                                maximum_support,
                            )
                            strength = penetration / radius
                            if strength > contact_strengths[world, object_index]:
                                contact_strengths[world, object_index] = strength
                                contact_obstacle[
                                    world, frame_index, object_index
                                ] = obstacle_index
                        strain_norm = wp.length(strain)
                        if strain_norm > 0.7:
                            strain *= 0.7 / strain_norm
                        strains[world, object_index] = strain

                    if enable_soft_body_contacts != 0:
                        for first_index in range(object_count):
                            if frame_index <= first_offsets[first_index]:
                                continue
                            for second_index in range(first_index + 1, object_count):
                                if frame_index <= first_offsets[second_index]:
                                    continue
                                delta = (
                                    positions[world, first_index]
                                    - positions[world, second_index]
                                )
                                distance = wp.length(delta)
                                normal = wp.vec2(1.0, 0.0)
                                if distance > 1.0e-8:
                                    normal = delta / distance
                                first_radius = radii[first_index]
                                second_radius = radii[second_index]
                                first_strain = strains[world, first_index]
                                second_strain = strains[world, second_index]
                                first_support = _deformed_support_radius(
                                    first_radius, first_strain, normal
                                )
                                second_support = _deformed_support_radius(
                                    second_radius, second_strain, normal
                                )
                                maximum_support_sum = (
                                    distance
                                    + pair_penetration_allowance[
                                        frame_index, first_index, second_index
                                    ]
                                )
                                penetration = (
                                    first_support + second_support
                                    - maximum_support_sum
                                )
                                if penetration <= 0.0:
                                    continue
                                radius_sum = first_radius + second_radius
                                first_limit = wp.max(
                                    maximum_support_sum
                                    * first_radius / radius_sum,
                                    float(MIN_DEFORMED_SUPPORT_SCALE) * first_radius,
                                )
                                second_limit = wp.max(
                                    maximum_support_sum
                                    * second_radius / radius_sum,
                                    float(MIN_DEFORMED_SUPPORT_SCALE) * second_radius,
                                )
                                first_strain = _project_strain_support(
                                    first_radius, first_strain, normal, first_limit
                                )
                                second_strain = _project_strain_support(
                                    second_radius, second_strain, -normal, second_limit
                                )
                                strains[world, first_index] = first_strain
                                strains[world, second_index] = second_strain
                                first_strength = 0.5 * penetration / first_radius
                                second_strength = 0.5 * penetration / second_radius
                                if first_strength > contact_strengths[world, first_index]:
                                    contact_strengths[world, first_index] = first_strength
                                    contact_obstacle[
                                        world, frame_index, first_index
                                    ] = obstacle_count + second_index
                                if second_strength > contact_strengths[world, second_index]:
                                    contact_strengths[world, second_index] = second_strength
                                    contact_obstacle[
                                        world, frame_index, second_index
                                    ] = obstacle_count + first_index
        for object_index in range(object_count):
            position = positions[world, object_index]
            strain = strains[world, object_index]
            predicted_positions[world, frame_index, object_index] = position
            predicted_strains[world, frame_index, object_index] = strain


class BatchWarpSoftBodyEngine:
    def __init__(
        self, scene: SoftBodySceneData, *, capacity: int, device: str = "cuda:0",
        substeps: int = 8,
        projection_tolerance: float = OUTPUT_PROJECTION_TOLERANCE_M,
        strict_projection: bool = True,
        initial_strains: np.ndarray | None = None,
        initial_strain_velocities: np.ndarray | None = None,
        block_dim: int = DEFAULT_CUDA_BLOCK_DIM,
        deformable_contacts: bool = False,
        separating_normal_damping: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        if not 1 <= block_dim <= 1024:
            raise ValueError("block_dim must be between 1 and 1024")
        self.scene = scene
        self.projection_tolerance = float(projection_tolerance)
        self.strict_projection = strict_projection
        self.projection_diagnostics = []
        self.capacity = int(capacity)
        self.device = device
        self.torch_device = torch.device(device)
        self.substeps = int(substeps)
        self.block_dim = int(block_dim)
        self.layout = make_parameter_layout(scene.object_count)
        self.enable_soft_body_contacts = True
        self.enable_shape_projection = True
        self.deformable_contacts = bool(deformable_contacts)
        self.separating_normal_damping = bool(separating_normal_damping)
        static_allowance, pair_allowance = _target_penetration_allowances(scene)
        obstacle_centers = np.asarray([item.center for item in scene.obstacles], np.float32)
        obstacle_axes = np.asarray([item.axis_x for item in scene.obstacles], np.float32)
        obstacle_half = np.asarray([item.half_extents for item in scene.obstacles], np.float32)
        wp.init()
        with wp.ScopedDevice(device):
            self.theta = wp.empty((capacity, self.layout.size), dtype=float, device=device)
            self.initial_centers = wp.array(scene.initial_centers.astype(np.float32), dtype=wp.vec2, device=device)
            self.initial_strains = wp.array(np.zeros((scene.object_count, 3), np.float32) if initial_strains is None else np.asarray(initial_strains, np.float32), dtype=wp.vec3, device=device)
            self.initial_strain_velocities = wp.array(np.zeros((scene.object_count, 3), np.float32) if initial_strain_velocities is None else np.asarray(initial_strain_velocities, np.float32), dtype=wp.vec3, device=device)
            self.radii = wp.array(scene.radii.astype(np.float32), dtype=float, device=device)
            self.first_offsets = wp.array(np.asarray(scene.first_offsets, np.int32), dtype=int, device=device)
            self.obstacle_centers = wp.array(obstacle_centers, dtype=wp.vec2, device=device)
            self.obstacle_axes = wp.array(obstacle_axes, dtype=wp.vec2, device=device)
            self.obstacle_half = wp.array(obstacle_half, dtype=wp.vec2, device=device)
            self.static_penetration_allowance = wp.array(
                static_allowance, dtype=float, device=device
            )
            self.pair_penetration_allowance = wp.array(
                pair_allowance, dtype=float, device=device
            )
            state_shape = (capacity, scene.object_count)
            self.positions = wp.empty(state_shape, dtype=wp.vec2, device=device)
            self.velocities = wp.empty(state_shape, dtype=wp.vec2, device=device)
            self.strains = wp.empty(state_shape, dtype=wp.vec3, device=device)
            self.strain_velocities = wp.empty(state_shape, dtype=wp.vec3, device=device)
            self.accelerations = wp.empty(
                state_shape, dtype=wp.vec2, device=device
            )
            self.target_strains = wp.empty(
                state_shape, dtype=wp.vec3, device=device
            )
            self.contact_strengths = wp.empty(
                state_shape, dtype=float, device=device
            )
            self.predicted_positions = wp.empty(
                (capacity, scene.frame_count, scene.object_count), dtype=wp.vec2, device=device
            )
            self.predicted_strains = wp.empty(
                (capacity, scene.frame_count, scene.object_count), dtype=wp.vec3, device=device
            )
            self.contacts = wp.empty(
                (capacity, scene.frame_count, scene.object_count), dtype=int, device=device
            )
        self.theta_torch = wp.to_torch(self.theta)
        self.predicted_positions_torch = wp.to_torch(self.predicted_positions)
        self.predicted_strains_torch = wp.to_torch(self.predicted_strains)
        self.contacts_torch = wp.to_torch(self.contacts)
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                value, dtype=torch.float32, device=self.torch_device
            )
        self.target_subset_centers = tensor(scene.target_subset_centers)
        self.reference_subset_offsets = tensor(scene.reference_subset_offsets)
        self.target_deformations = tensor(scene.target_deformations)
        self.observed_mask = tensor(scene.observed_mask)
        self.shape_observed_mask = tensor(scene.shape_observed_mask)
        self.radii_torch = tensor(scene.radii)
        identity = torch.eye(2, device=self.torch_device).reshape(1, 1, 2, 2)
        self.deformation_signal = torch.sqrt(
            (self.target_deformations - identity).square().mean(dim=(-1, -2))
        )

    def rollout(self, samples: torch.Tensor | np.ndarray, end_offset: int | None = None) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32, device=self.torch_device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[1] != self.layout.size:
            raise ValueError(f"expected {self.layout.size} parameters, got {values.shape[1]}")
        count = int(values.shape[0])
        if count > self.capacity:
            raise ValueError(f"batch size {count} exceeds engine capacity {self.capacity}")
        final_offset = self.scene.frame_count - 1 if end_offset is None else min(
            max(int(end_offset), 0), self.scene.frame_count - 1
        )
        self.theta_torch[:count].copy_(values)
        layout = self.layout
        wp.launch(
            _simulate_soft_worlds,
            dim=count,
            block_dim=self.block_dim,
            inputs=[
                self.theta, self.initial_centers, self.initial_strains, self.initial_strain_velocities, self.radii, self.first_offsets,
                self.obstacle_centers, self.obstacle_axes, self.obstacle_half,
                self.static_penetration_allowance,
                self.pair_penetration_allowance,
                self.scene.object_count, len(self.scene.obstacles), final_offset + 1,
                float(self.scene.fps), self.substeps,
                layout.log_gravity_scale_slice.start,
                layout.log_radius_scale_slice.start, layout.log_frequency_slice.start,
                layout.damping_ratio_slice.start, layout.friction_slice.start,
                layout.deformation_gain_slice.start,
                layout.log_shape_frequency_slice.start,
                layout.shape_damping_ratio_slice.start,
                layout.drag_slice.start,
                int(self.enable_soft_body_contacts),
                int(self.enable_shape_projection),
                int(self.deformable_contacts),
                int(self.separating_normal_damping),
                self.positions, self.velocities, self.strains, self.strain_velocities,
                self.accelerations, self.target_strains, self.contact_strengths,
                self.predicted_positions, self.predicted_strains, self.contacts,
            ],
            device=self.device,
        )
        return torch.cat(
            (
                self.predicted_positions_torch[:count, : final_offset + 1],
                self.predicted_strains_torch[:count, : final_offset + 1],
            ),
            dim=-1,
        )

    def rollout_numpy(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predicted = self.rollout(values)
        wp.synchronize_device(self.device)
        result = predicted[0].detach().cpu().numpy().astype(np.float64)
        if self.enable_shape_projection:
            self.projection_diagnostics = []
            result = _project_output_surface_positions(self.scene, result, tolerance=self.projection_tolerance, strict=self.strict_projection, diagnostics=self.projection_diagnostics)
        return result, self.contacts_torch[0].detach().cpu().numpy().astype(np.int32)


def _deformation_matrices(predicted: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((*predicted.shape[:-1], 2, 2), dtype=predicted.dtype, device=predicted.device)
    result[..., 0, 0] = 1.0 + predicted[..., 2]
    result[..., 1, 1] = 1.0 + predicted[..., 3]
    result[..., 0, 1] = predicted[..., 4]
    result[..., 1, 0] = predicted[..., 4]
    return result


def candidate_loss(
    engine: BatchWarpSoftBodyEngine, predicted: torch.Tensor, *, end_offset: int,
    huber_delta_m: float, shape_weight: float, deformation_focus: float,
    per_object: bool = False,
) -> torch.Tensor:
    frame_count = predicted.shape[1]
    deformation = _deformation_matrices(predicted)
    reference_offsets = engine.reference_subset_offsets[:frame_count]
    predicted_subset_centers = predicted[..., :2] + torch.einsum(
        "pfnij,fnj->pfni", deformation, reference_offsets
    )
    center_residual = predicted_subset_centers - engine.target_subset_centers[:frame_count]
    absolute = torch.abs(center_residual)
    center_loss = torch.where(
        absolute <= huber_delta_m,
        0.5 * center_residual.square(),
        huber_delta_m * (absolute - 0.5 * huber_delta_m),
    ).mean(dim=-1)
    radius = engine.radii_torch.reshape(1, 1, -1, 1, 1)
    shape_loss = (
        radius * (deformation - engine.target_deformations[:frame_count])
    ).square().mean(dim=(-1, -2))
    shape_available = engine.shape_observed_mask[:frame_count].unsqueeze(0)
    deformation_weight = 1.0 + deformation_focus * torch.clamp(
        engine.deformation_signal[:frame_count] / 0.15, 0.0, 1.0
    )
    mask = engine.observed_mask[:frame_count].unsqueeze(0)
    frame_indices = torch.arange(frame_count, device=predicted.device)
    prefix_mask = (frame_indices <= end_offset).float().reshape(1, -1, 1)
    weights = mask * prefix_mask
    object_losses = (
        (
            (
                center_loss
                + shape_weight * deformation_weight * shape_loss * shape_available
            )
            * weights
        ).sum(dim=1)
        / weights.sum(dim=1).clamp_min(1.0)
    )
    return object_losses if per_object else object_losses.mean(dim=1)


def _object_parameter_indices(
    layout: ParameterLayout, object_index: int,
) -> tuple[int, ...]:
    return (
        2 * object_index,
        2 * object_index + 1,
        layout.log_gravity_scale_slice.start + object_index,
        layout.log_radius_scale_slice.start + object_index,
        layout.log_frequency_slice.start + object_index,
        layout.damping_ratio_slice.start + object_index,
        layout.friction_slice.start + object_index,
        layout.deformation_gain_slice.start + object_index,
        layout.log_shape_frequency_slice.start + object_index,
        layout.shape_damping_ratio_slice.start + object_index,
        layout.drag_slice.start + object_index,
    )


def _shape_parameter_indices(
    layout: ParameterLayout, object_index: int,
) -> tuple[int, ...]:
    return (
        layout.deformation_gain_slice.start + object_index,
        layout.log_shape_frequency_slice.start + object_index,
        layout.shape_damping_ratio_slice.start + object_index,
    )


def candidate_shape_loss(
    engine: BatchWarpSoftBodyEngine, predicted: torch.Tensor, *,
    event_weight: float,
) -> torch.Tensor:
    deformation = _deformation_matrices(predicted)
    radius = engine.radii_torch.reshape(1, 1, -1, 1, 1)
    frame_loss = (
        radius * (deformation - engine.target_deformations)
    ).square().mean(dim=(-1, -2))
    observed = engine.shape_observed_mask.unsqueeze(0)
    base = (frame_loss * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
    signal_weight = observed * engine.deformation_signal.unsqueeze(0)
    event = (frame_loss * signal_weight).sum(dim=1) / signal_weight.sum(
        dim=1
    ).clamp_min(1.0e-8)
    return base + event_weight * event


def refine_shape_parameters(
    engine: BatchWarpSoftBodyEngine, config: SoftBodyCEMConfig,
    initial: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    generations = config.shape_refinement_generations
    if config.shape_event_weight < 0.0:
        raise ValueError("shape_event_weight cannot be negative")
    if generations <= 0:
        return np.asarray(initial).copy(), {"generations": 0}
    layout = engine.layout
    device = engine.torch_device
    lower_np, upper_np = parameter_bounds(layout)
    _, std_np = initial_distribution(layout)
    minimum_std_np = minimum_distribution_std(layout)
    lower = torch.as_tensor(lower_np, device=device)
    upper = torch.as_tensor(upper_np, device=device)
    mean = torch.as_tensor(initial, dtype=torch.float32, device=device).clone()
    std = torch.as_tensor(std_np, device=device)
    minimum_std = torch.as_tensor(minimum_std_np, device=device)
    best = mean.clone()
    shape_indices = tuple(
        _shape_parameter_indices(layout, object_index)
        for object_index in range(engine.scene.object_count)
    )
    flat_indices = torch.as_tensor(
        tuple(index for indices in shape_indices for index in indices), device=device
    )
    generator = torch.Generator(device=device).manual_seed(config.seed + 100_003)
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    best_losses = candidate_shape_loss(
        engine, engine.rollout(best), event_weight=config.shape_event_weight
    )[0]
    initial_losses = best_losses.clone()
    for _ in range(generations):
        samples = best.unsqueeze(0).repeat(config.population, 1)
        sampled_shape = (
            mean[flat_indices]
            + std[flat_indices]
            * torch.randn(
                (config.population, len(flat_indices)),
                generator=generator,
                device=device,
            )
        )
        sampled_shape.clamp_(lower[flat_indices], upper[flat_indices])
        samples[:, flat_indices] = sampled_shape
        samples[0] = best
        losses = candidate_shape_loss(
            engine, engine.rollout(samples), event_weight=config.shape_event_weight
        )
        for object_index, object_indices in enumerate(shape_indices):
            indices = torch.as_tensor(object_indices, device=device)
            elite_losses, elite_rows = torch.topk(
                losses[:, object_index], elite_count, largest=False
            )
            elites = samples[elite_rows][:, indices]
            smoothing = config.distribution_smoothing
            mean[indices] = (
                smoothing * mean[indices]
                + (1.0 - smoothing) * elites.mean(dim=0)
            )
            std[indices] = (
                smoothing * std[indices]
                + (1.0 - smoothing) * elites.std(dim=0, unbiased=False)
            )
            if elite_losses[0] < best_losses[object_index]:
                best_losses[object_index] = elite_losses[0]
                best[indices] = samples[elite_rows[0], indices]
        std[flat_indices] = torch.maximum(
            std[flat_indices], minimum_std[flat_indices]
        )
        if config.exploration_fraction > 0.0:
            std[flat_indices] = torch.maximum(
                std[flat_indices],
                config.exploration_fraction
                * (upper[flat_indices] - lower[flat_indices]),
            )
    return best.detach().cpu().numpy(), {
        "generations": generations,
        "event_weight": config.shape_event_weight,
        "initial_per_object_loss": initial_losses.detach().cpu().tolist(),
        "final_per_object_loss": best_losses.detach().cpu().tolist(),
    }


def run_cem(
    scene: SoftBodySceneData, engine: BatchWarpSoftBodyEngine,
    config: SoftBodyCEMConfig, *, initial: np.ndarray | None = None,
    initial_std_scale: float = 1.0, joint: bool = True,
    generations_per_prefix: int | None = None, verbose: bool = True,
) -> dict[str, Any]:
    if config.population > engine.capacity:
        raise ValueError("CEM population exceeds engine capacity")
    if initial_std_scale <= 0.0:
        raise ValueError("initial_std_scale must be positive")
    layout = engine.layout
    lower_np, upper_np = parameter_bounds(layout)
    mean_np, std_np = initial_distribution(layout)
    if initial is not None:
        mean_np = project_parameters(initial, layout).astype(np.float32)
        std_np *= float(initial_std_scale)
    minimum_std_np = minimum_distribution_std(layout)
    device = engine.torch_device
    lower = torch.as_tensor(lower_np, device=device)
    upper = torch.as_tensor(upper_np, device=device)
    mean = torch.as_tensor(mean_np, device=device)
    std = torch.as_tensor(std_np, device=device)
    minimum_std = torch.as_tensor(minimum_std_np, device=device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    generation_count = (
        config.generations_per_prefix
        if generations_per_prefix is None
        else int(generations_per_prefix)
    )
    if config.population < 2 or generation_count <= 0:
        raise ValueError("CEM requires at least two samples and one generation")
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    trace = []
    best = mean.clone()
    best_loss = math.inf
    prefixes = tuple(
        scene.frame_count - 1 if value <= 0 else min(value - 1, scene.frame_count - 1)
        for value in config.prefix_frames
    )
    for stage_index, end_offset in enumerate(prefixes):
        active_objects = np.any(
            scene.observed_mask[: end_offset + 1] > 0.0, axis=0
        )
        active_object_indices = torch.as_tensor(
            np.flatnonzero(active_objects), device=device
        )
        stage_best_loss = math.inf
        stage_object_best = np.where(active_objects, math.inf, 0.0)
        stage_candidate = best.clone()
        for generation in range(generation_count):
            joint_indices = None
            if joint:
                active_sequence = np.flatnonzero(active_objects)
                object_index = int(active_sequence[generation % len(active_sequence)])
                joint_indices = torch.as_tensor(
                    _object_parameter_indices(layout, object_index), device=device
                )
                samples = best.unsqueeze(0).repeat(config.population, 1)
                samples[:, joint_indices] = (
                    mean[joint_indices]
                    + std[joint_indices]
                    * torch.randn(
                        (config.population, len(joint_indices)),
                        generator=generator,
                        device=device,
                    )
                )
                samples[:, joint_indices] = torch.clamp(
                    samples[:, joint_indices],
                    lower[joint_indices],
                    upper[joint_indices],
                )
            else:
                samples = mean + std * torch.randn(
                    (config.population, layout.size),
                    generator=generator,
                    device=device,
                )
                samples.clamp_(lower, upper)
            if generation == 0:
                samples[0] = best
            predicted = engine.rollout(samples, end_offset)
            object_losses = candidate_loss(
                engine, predicted, end_offset=end_offset,
                huber_delta_m=config.huber_delta_m, shape_weight=config.shape_weight,
                deformation_focus=config.deformation_focus,
                per_object=True,
            )
            smoothing = config.distribution_smoothing
            if joint:
                assert joint_indices is not None
                losses = object_losses[:, active_object_indices].mean(dim=1)
                elite_losses, elite_indices = torch.topk(
                    losses, elite_count, largest=False
                )
                elites = samples[elite_indices][:, joint_indices]
                mean[joint_indices] = (
                    smoothing * mean[joint_indices]
                    + (1.0 - smoothing) * elites.mean(dim=0)
                )
                std[joint_indices] = (
                    smoothing * std[joint_indices]
                    + (1.0 - smoothing) * elites.std(dim=0, unbiased=False)
                )
                generation_loss = float(elite_losses[0].item())
                if generation_loss < stage_best_loss:
                    stage_best_loss = generation_loss
                    best[joint_indices] = samples[elite_indices[0], joint_indices]
                    stage_candidate = best.clone()
            else:
                for object_index in np.flatnonzero(active_objects):
                    elite_losses, elite_indices = torch.topk(
                        object_losses[:, object_index], elite_count, largest=False
                    )
                    indices = torch.as_tensor(
                        _object_parameter_indices(layout, int(object_index)),
                        device=device,
                    )
                    elites = samples[elite_indices][:, indices]
                    mean[indices] = (
                        smoothing * mean[indices]
                        + (1.0 - smoothing) * elites.mean(dim=0)
                    )
                    std[indices] = (
                        smoothing * std[indices]
                        + (1.0 - smoothing)
                        * elites.std(dim=0, unbiased=False)
                    )
                    object_loss = float(elite_losses[0].item())
                    if object_loss < stage_object_best[object_index]:
                        stage_object_best[object_index] = object_loss
                        stage_candidate[indices] = samples[elite_indices[0], indices]
            std = torch.maximum(std, minimum_std)
            if config.exploration_fraction > 0.0:
                std = torch.maximum(std, config.exploration_fraction * (upper - lower))
        best = stage_candidate
        best_loss = (
            stage_best_loss if joint else float(np.mean(stage_object_best))
        )
        entry = {
            "stage": stage_index + 1,
            "end_frame": int(scene.frames[end_offset]),
            "best_loss": best_loss,
            "optimization": "joint" if joint else "independent",
        }
        trace.append(entry)
        mean = best.clone()
        if verbose:
            print(json.dumps({"phase": "softbody_cem", **entry}, sort_keys=True), flush=True)
    return {"best": best.detach().cpu().numpy(), "best_loss": best_loss, "trace": trace}


def _numpy_deformation(predicted: np.ndarray) -> np.ndarray:
    result = np.zeros((*predicted.shape[:-1], 2, 2), dtype=np.float64)
    result[..., 0, 0] = 1.0 + predicted[..., 2]
    result[..., 1, 1] = 1.0 + predicted[..., 3]
    result[..., 0, 1] = predicted[..., 4]
    result[..., 1, 0] = predicted[..., 4]
    return result


def _smoothed_target_pair_normals(scene: SoftBodySceneData) -> np.ndarray:
    """Estimate temporally continuous separating directions from tracked centers."""
    result = np.zeros(
        (scene.frame_count, scene.object_count, scene.object_count, 2),
        dtype=np.float64,
    )
    half_window = max(1, int(round(PAIR_NORMAL_SMOOTHING_SECONDS * scene.fps)))
    ramp = np.arange(1, half_window + 2, dtype=np.float64)
    kernel = np.concatenate((ramp, ramp[-2::-1]))
    kernel /= np.sum(kernel)
    frame_indices = np.arange(scene.frame_count)
    observed = scene.observed_mask > 0.0
    for first_index in range(scene.object_count):
        for second_index in range(first_index + 1, scene.object_count):
            delta = (
                scene.target_centers[:, first_index]
                - scene.target_centers[:, second_index]
            )
            distance = np.linalg.norm(delta, axis=-1)
            valid = (
                observed[:, first_index]
                & observed[:, second_index]
                & (distance > 1.0e-8)
            )
            if not np.any(valid):
                continue
            normals = np.zeros_like(delta)
            normals[valid] = delta[valid] / distance[valid, None]
            valid_indices = frame_indices[valid]
            for axis in range(2):
                normals[:, axis] = np.interp(
                    frame_indices, valid_indices, normals[valid, axis]
                )
                normals[:, axis] = np.convolve(
                    np.pad(normals[:, axis], (half_window, half_window), mode="edge"),
                    kernel,
                    mode="valid",
                )
            length = np.linalg.norm(normals, axis=-1)
            normalized = normals / np.maximum(length[:, None], 1.0e-12)
            result[:, first_index, second_index] = normalized
            result[:, second_index, first_index] = -normalized
    return result


def _output_pair_correction_normals(
    scene: SoftBodySceneData, predicted: np.ndarray,
) -> np.ndarray:
    """Choose one stable separating direction per pair and frame."""
    target_normals = _smoothed_target_pair_normals(scene)
    result = np.zeros_like(target_normals)
    for frame_index in range(scene.frame_count):
        for first_index in range(scene.object_count):
            for second_index in range(first_index + 1, scene.object_count):
                delta = (
                    predicted[frame_index, first_index, :2]
                    - predicted[frame_index, second_index, :2]
                )
                distance = float(np.linalg.norm(delta))
                physical_normal = (
                    delta / distance
                    if distance > 1.0e-8
                    else np.asarray([1.0, 0.0])
                )
                target_normal = target_normals[
                    frame_index, first_index, second_index
                ]
                target_consistent = (
                    np.linalg.norm(target_normal) > 1.0e-8
                    and np.dot(target_normal, physical_normal)
                    >= MIN_TARGET_PAIR_NORMAL_ALIGNMENT
                )
                normal = target_normal if target_consistent else physical_normal
                result[frame_index, first_index, second_index] = normal
                result[frame_index, second_index, first_index] = -normal
    return result


def _project_output_surface_positions(
    scene: SoftBodySceneData, predicted: np.ndarray, *,
    tolerance: float = OUTPUT_PROJECTION_TOLERANCE_M,
    strict: bool = True, diagnostics: list | None = None,
) -> np.ndarray:
    """Apply per-frame minimum center corrections for strict surface separation."""
    result = np.asarray(predicted, dtype=np.float64).copy()
    deformation = _numpy_deformation(result)
    valid = scene.observed_mask > 0.0
    pair_correction_normals = _output_pair_correction_normals(scene, result)
    for frame_index in range(scene.frame_count):
        converged = False
        for _projection_iteration in range(MAX_OUTPUT_PROJECTION_ITERATIONS):
            maximum_correction = 0.0
            for object_index in range(scene.object_count):
                if not valid[frame_index, object_index]:
                    continue
                center = result[frame_index, object_index, :2]
                transform = deformation[frame_index, object_index]
                radius = scene.radii[object_index]
                for obstacle in scene.obstacles:
                    signed_distance, normal = _obb_signed_distance_and_normal(
                        center, obstacle
                    )
                    support = radius * np.linalg.norm(transform.T @ normal)
                    penetration = support - signed_distance
                    if penetration > 0.0:
                        center += penetration * normal
                        maximum_correction = max(maximum_correction, penetration)
                result[frame_index, object_index, :2] = center
            for first_index in range(scene.object_count):
                if not valid[frame_index, first_index]:
                    continue
                for second_index in range(first_index + 1, scene.object_count):
                    if not valid[frame_index, second_index]:
                        continue
                    delta = (
                        result[frame_index, first_index, :2]
                        - result[frame_index, second_index, :2]
                    )
                    distance = float(np.linalg.norm(delta))
                    correction_normal = pair_correction_normals[
                        frame_index, first_index, second_index
                    ]
                    if distance <= 1.0e-8:
                        normal = correction_normal
                    else:
                        normal = delta / distance
                    first_support = scene.radii[first_index] * np.linalg.norm(
                        deformation[frame_index, first_index].T @ normal
                    )
                    second_support = scene.radii[second_index] * np.linalg.norm(
                        deformation[frame_index, second_index].T @ normal
                    )
                    penetration = first_support + second_support - distance
                    if penetration <= 0.0:
                        continue
                    correction = 0.5 * penetration * correction_normal
                    result[frame_index, first_index, :2] += correction
                    result[frame_index, second_index, :2] -= correction
                    maximum_correction = max(maximum_correction, penetration)
            if maximum_correction <= tolerance:
                converged = True
                break
        if not converged and diagnostics is not None:
            diagnostics.append(dict(frame=int(scene.frames[frame_index]), maximum_correction=float(maximum_correction)))
        if not converged and strict:
            raise RuntimeError(
                "strict output surface projection did not converge for "
                f"frame {scene.frames[frame_index]}"
            )
    return result


def softbody_metrics(scene: SoftBodySceneData, predicted: np.ndarray) -> dict[str, Any]:
    deformation = _numpy_deformation(predicted)
    predicted_subset_centers = predicted[..., :2] + np.einsum(
        "fnij,fnj->fni", deformation, scene.reference_subset_offsets
    )
    valid = scene.observed_mask > 0.0
    shape_valid = scene.shape_observed_mask > 0.0
    translation_error = np.linalg.norm(predicted[..., :2] - scene.target_centers, axis=-1)
    subset_error = np.linalg.norm(predicted_subset_centers - scene.target_subset_centers, axis=-1)
    deformation_error = np.sqrt(
        np.mean(
            (
                scene.radii[None, :, None, None]
                * (deformation - scene.target_deformations)
            )
            ** 2,
            axis=(-1, -2),
        )
    )
    per_object_translation = {}
    per_object_shape = {}
    per_object_surface = {}
    surface_squared = subset_error**2 + np.where(
        shape_valid, deformation_error**2, 0.0
    )
    for object_index, object_id in enumerate(scene.object_ids):
        object_valid = valid[:, object_index]
        per_object_translation[object_id] = float(
            np.sqrt(np.mean(translation_error[object_valid, object_index] ** 2))
        )
        object_shape_valid = shape_valid[:, object_index]
        per_object_shape[object_id] = float(np.sqrt(np.mean(
            deformation_error[object_shape_valid, object_index] ** 2
        )))
        per_object_surface[object_id] = float(
            np.sqrt(np.mean(surface_squared[object_valid, object_index]))
        )
    return {
        "overall_translation_rmse_m": float(np.sqrt(np.mean(translation_error[valid] ** 2))),
        "overall_shape_moment_rmse_m": float(np.sqrt(
            np.mean(deformation_error[shape_valid] ** 2)
        )),
        "overall_surface_moment_rmse_m": float(np.sqrt(np.mean(surface_squared[valid]))),
        "per_object_translation_rmse_m": per_object_translation,
        "per_object_shape_moment_rmse_m": per_object_shape,
        "per_object_surface_moment_rmse_m": per_object_surface,
    }


def _portfolio_selection_score(metrics: dict[str, Any]) -> float:
    worst_object = max(metrics["per_object_surface_moment_rmse_m"].values())
    return float(
        metrics["overall_surface_moment_rmse_m"]
        + WORST_OBJECT_SELECTION_WEIGHT * worst_object
    )


def fit_softbody_scene(
    scene: SoftBodySceneData, *, configs: tuple[SoftBodyCEMConfig, ...] | None = None,
    device: str = "cuda:0", substeps: int = 8, verbose: bool = True,
    projection_tolerance: float = OUTPUT_PROJECTION_TOLERANCE_M,
    strict_projection: bool = True,
) -> tuple[dict[str, Any], SoftBodyCEMConfig, float]:
    if configs is None:
        configs = tuple(SoftBodyCEMConfig(seed=seed) for seed in DEFAULT_SEEDS)
    if not configs:
        raise ValueError("at least one CEM config is required")
    start = time.perf_counter()
    engine = BatchWarpSoftBodyEngine(
        scene, capacity=max(config.population for config in configs), device=device,
        substeps=substeps, projection_tolerance=projection_tolerance, strict_projection=strict_projection,
    )
    selected = None
    selected_config = None
    portfolio = []
    for config in configs:
        candidate_start = time.perf_counter()
        engine.enable_shape_projection = False
        engine.enable_soft_body_contacts = False
        independent = run_cem(
            scene, engine, config, joint=False, verbose=verbose
        )
        engine.enable_soft_body_contacts = True
        cem = run_cem(
            scene,
            engine,
            config,
            initial=independent["best"],
            initial_std_scale=config.joint_initial_std_scale,
            joint=True,
            generations_per_prefix=config.joint_generations_per_prefix,
            verbose=verbose,
        )
        engine.enable_shape_projection = True
        refined, shape_refinement = refine_shape_parameters(
            engine, config, cem["best"]
        )
        cem["best"] = refined
        predicted, contacts = engine.rollout_numpy(cem["best"])
        metrics = softbody_metrics(scene, predicted)
        selection_score = _portfolio_selection_score(metrics)
        row = {
            "seed": config.seed,
            **metrics,
            "portfolio_selection_score_m": selection_score,
            "elapsed_seconds": time.perf_counter() - candidate_start,
        }
        portfolio.append(row)
        if (
            selected is None
            or selection_score < selected["portfolio_selection_score_m"]
        ):
            selected = {
                "best": cem["best"],
                "predicted": predicted,
                "contacts": contacts,
                "cem_trace": independent["trace"] + cem["trace"],
                "shape_refinement": shape_refinement,
                **metrics,
                "portfolio_selection_score_m": selection_score,
            }
            selected_config = config
    assert selected is not None and selected_config is not None
    selected["portfolio"] = portfolio
    return selected, selected_config, time.perf_counter() - start


def _contact_events(scene: SoftBodySceneData, contacts: np.ndarray) -> list[dict[str, Any]]:
    events = []
    for object_index, object_id in enumerate(scene.object_ids):
        previous = -1
        for frame_index in range(scene.frame_count):
            obstacle_index = int(contacts[frame_index, object_index])
            if obstacle_index >= 0 and obstacle_index != previous:
                if obstacle_index < len(scene.obstacles):
                    contact_id = scene.obstacles[obstacle_index].object_id
                    contact_type = "static_obstacle"
                else:
                    partner_index = obstacle_index - len(scene.obstacles)
                    contact_id = scene.object_ids[partner_index]
                    contact_type = "soft_body"
                events.append(
                    {
                        "frame_index": int(scene.frames[frame_index]),
                        "time_s": frame_index / scene.fps,
                        "soft_body_id": object_id,
                        "contact_id": contact_id,
                        "contact_type": contact_type,
                        "obstacle_id": contact_id,
                    }
                )
            previous = obstacle_index
    return events


def result_payload(
    source: dict[str, Any], scene: SoftBodySceneData, config: SoftBodyCEMConfig,
    fit: dict[str, Any], elapsed_seconds: float, *, device: str, substeps: int,
) -> dict[str, Any]:
    parameters = decode_parameters(fit["best"], make_parameter_layout(scene.object_count))
    predicted = fit["predicted"]
    deformation = _numpy_deformation(predicted)
    object_parameters = {}
    for index, object_id in enumerate(scene.object_ids):
        object_parameters[object_id] = {
            "initial_velocity_plane_m_per_s": parameters["initial_velocity"][index].tolist(),
            "effective_gravity_m_per_s2": float(
                GRAVITY_M_PER_S2 * parameters["gravity_scale"][index]
            ),
            "gravity_scale": float(parameters["gravity_scale"][index]),
            "collision_radius_m": float(scene.radii[index] * parameters["radius_scale"][index]),
            "collision_radius_scale": float(parameters["radius_scale"][index]),
            "natural_frequency_rad_per_s": float(parameters["natural_frequency_rad_per_s"][index]),
            "damping_ratio": float(parameters["damping_ratio"][index]),
            "contact_friction": float(parameters["contact_friction"][index]),
            "deformation_gain": float(parameters["deformation_gain"][index]),
            "shape_frequency_rad_per_s": float(
                parameters["shape_frequency_rad_per_s"][index]
            ),
            "shape_damping_ratio": float(
                parameters["shape_damping_ratio"][index]
            ),
            "linear_drag_per_s": float(parameters["linear_drag_per_s"][index]),
        }
    trajectories = {}
    for object_index, object_id in enumerate(scene.object_ids):
        rows = []
        for frame_index in range(scene.frame_count):
            if scene.observed_mask[frame_index, object_index] <= 0.0:
                continue
            plane = predicted[frame_index, object_index, :2]
            world = (
                plane[0] * scene.wall_lateral_world
                + plane[1] * scene.gravity_down_world
                + scene.wall_depths[object_index] * scene.wall_normal_world
            )
            rows.append(
                {
                    "frame_index": int(scene.frames[frame_index]),
                    "plane_lateral_down_m": plane.tolist(),
                    "center_world": world.tolist(),
                    "deformation_2x2": deformation[frame_index, object_index].tolist(),
                }
            )
        trajectories[object_id] = rows
    return {
        "model": "physmind-v2-warp-reduced-softbody-swr-v11",
        "source": source,
        "device": device,
        "substeps_per_frame": substeps,
        "fps": scene.fps,
        "frame_count": scene.frame_count,
        "object_ids": list(scene.object_ids),
        "valid_frame_counts": dict(zip(scene.object_ids, scene.valid_frame_counts)),
        "cem_config": asdict(config),
        "selected_seed": config.seed,
        "elapsed_seconds": elapsed_seconds,
        "parameters": object_parameters,
        "parameter_semantics": PARAMETER_SEMANTICS,
        "obstacles": [asdict(obstacle) for obstacle in scene.obstacles],
        "events": _contact_events(scene, fit["contacts"]),
        "predicted_trajectories": trajectories,
        "portfolio": fit["portfolio"],
        "cem_trace": fit["cem_trace"],
        "shape_refinement": fit["shape_refinement"],
        "overall_translation_rmse_m": fit["overall_translation_rmse_m"],
        "overall_shape_moment_rmse_m": fit["overall_shape_moment_rmse_m"],
        "overall_surface_moment_rmse_m": fit["overall_surface_moment_rmse_m"],
        "portfolio_selection_score_m": fit["portfolio_selection_score_m"],
        "per_object_translation_rmse_m": fit["per_object_translation_rmse_m"],
        "per_object_shape_moment_rmse_m": fit["per_object_shape_moment_rmse_m"],
        "per_object_surface_moment_rmse_m": fit["per_object_surface_moment_rmse_m"],
        "target": {
            "coordinate_frame": "wall_lateral_and_gravity_down",
            "translation_source": (
                "tracking/trajectories model_xyz with fixed material IDs"
            ),
            "deformation_source": scene.shape_target_source,
            "validity": (
                f"finite points, confidence >= {scene.minimum_tracking_confidence:g}, "
                f"at least {MIN_TRACKED_POINTS} points, non-border frame"
            ),
            "deformation_validity": (
                "complete non-border silhouette; severe area-only loss is "
                "retained at full weight for short, bracketed 2D contact events"
            ),
            "loss_compression": (
                "visibility-corrected point-cloud centroid and wall-plane "
                "silhouette second moment"
            ),
            "point_counts": dict(zip(scene.object_ids, scene.point_counts)),
            "deformation_valid_frame_counts": dict(zip(
                scene.object_ids,
                scene.shape_observed_mask.sum(axis=0).astype(int).tolist(),
            )),
            "deformation_contact_recovered_frame_counts": dict(zip(
                scene.object_ids,
                scene.shape_contact_recovered_mask.sum(axis=0).astype(int).tolist(),
            )),
            "trajectory_paths": dict(zip(scene.object_ids, scene.trajectory_paths)),
        },
        "physics_assumptions": {
            "execution": "custom_nvidia_warp_kernel",
            "center_dynamics": "2D gravity with linear drag in the vertical wall plane",
            "contact": (
                "mass-normalized Kelvin-Voigt compliant contact against static "
                "oriented boxes and other soft bodies"
            ),
            "deformation": "independent symmetric 2D modal strain with fixed out-of-plane Poisson response driven by contact compression",
            "surface_contact_consistency": (
                "two-pass strain-space projection limits static and soft-body "
                "proxy penetration to the allowance observed in reliable targets; "
                "final per-frame center projection enforces zero surface penetration"
            ),
            "optimization": (
                "independent-to-coupled staged derivative-free CEM in PyTorch, "
                "followed by shape-only refinement"
            ),
            "soft_body_interactions": (
                "symmetric compliant pair contact using shared existing material "
                "parameters"
            ),
        },
    }
