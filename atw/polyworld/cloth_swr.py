"""Reduced 3D Warp cloth simulation and system identification."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


GRAVITY_M_PER_S2 = 9.81
CLOTH_RIGID_FRICTION = 1.0
RIGID_RIGID_FRICTION = 0.5
GROUND_STATIC_FRICTION = 0.8
GROUND_DYNAMIC_FRICTION = 0.6
DEFAULT_SEEDS = (7, 23)
DEFAULT_OBSTACLE_MODE = "obb-dynamic"
DEFAULT_REFINEMENT_CAPACITY = 256
DEFAULT_CUDA_BLOCK_DIM = 32
DEFAULT_EXECUTION_MODE = "colored-batched"
DEFAULT_GRAPH_CHUNK_FRAMES = 16
DEFAULT_CONTACT_ITERATION_STRIDE = 2
CLOTH_EXECUTION_MODES = (
    "baseline", "interleaved", "batched-seeds", "interleaved-batched",
    "colored", "colored-batched",
)


@dataclass(frozen=True)
class ClothObstacle:
    object_id: str
    center: tuple[float, float, float]
    axis_x: tuple[float, float, float]
    axis_y: tuple[float, float, float]
    axis_z: tuple[float, float, float]
    half_extents: tuple[float, float, float]


@dataclass(frozen=True)
class ClothSceneData:
    object_id: str
    fps: float
    frames: tuple[int, ...]
    reference_frame: int
    start_frame: int
    vertex_ids: np.ndarray
    vertex_uv: np.ndarray
    initial_positions: np.ndarray
    rest_positions: np.ndarray
    target_positions: np.ndarray
    raw_target_positions: np.ndarray
    target_weights: np.ndarray
    faces: np.ndarray
    structural_edges: np.ndarray
    structural_rest_lengths: np.ndarray
    bending_edges: np.ndarray
    bending_rest_lengths: np.ndarray
    gravity_down_world: np.ndarray
    support_normal_world: np.ndarray
    support_offset_world: float
    obstacles: tuple[ClothObstacle, ...]
    source_trajectory: str
    source_mesh: str
    camera_centers_world: np.ndarray | None = None
    target_depth_mode: str = "camera-initial"

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def node_count(self) -> int:
        return len(self.vertex_ids)


@dataclass(frozen=True)
class ClothCEMConfig:
    population: int = 48
    elite_fraction: float = 0.08
    generations_per_prefix: int = 6
    prefix_frames: tuple[int, ...] = (16, 32, 0)
    distribution_smoothing: float = 0.2
    exploration_fraction: float = 0.02
    huber_delta_m: float = 1.0
    seed: int = 23
    release_refinement_passes: int = 2
    joint_dynamics_refinement: bool = True
    joint_dynamics_population: int = 64
    joint_dynamics_generations: int = 24


def _bending_edges(faces: np.ndarray) -> np.ndarray:
    adjacent: dict[tuple[int, int], list[int]] = {}
    for first, second, third in np.asarray(faces, dtype=np.int32):
        for left, right, opposite in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            edge = tuple(sorted((int(left), int(right))))
            adjacent.setdefault(edge, []).append(int(opposite))
    result = {
        tuple(sorted(opposites))
        for opposites in adjacent.values()
        if len(opposites) == 2 and opposites[0] != opposites[1]
    }
    return np.asarray(sorted(result), dtype=np.int32).reshape(-1, 2)


def _sample_coarse_mesh(
    vertex_uv: np.ndarray, rest_positions: np.ndarray, valid_at_start: np.ndarray,
    source_faces: np.ndarray, *, grid_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid_resolution < 4:
        raise ValueError("grid_resolution must be at least four")
    uv = np.asarray(vertex_uv, dtype=np.float64)
    rest = np.asarray(rest_positions, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int32)
    valid = np.asarray(valid_at_start, dtype=bool)
    if uv.ndim != 2 or uv.shape[1] != 2 or rest.shape != (len(uv), 3):
        raise ValueError("reference mesh vertices have invalid shapes")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("reference cloth mesh has no faces")
    if valid.shape != (len(uv),) or np.count_nonzero(valid) < 16:
        raise ValueError("release frame has too few valid cloth vertices")
    if np.any(faces < 0) or np.any(faces >= len(uv)):
        raise ValueError("reference cloth faces contain invalid vertex IDs")

    lower = np.min(uv, axis=0)
    span = np.maximum(np.max(uv, axis=0) - lower, 1.0)
    bins = np.rint((uv - lower) / span * (grid_resolution - 1)).astype(np.int32)
    cell = bins[:, 0] + grid_resolution * bins[:, 1]
    all_fine_edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    all_fine_edges.sort(axis=1)
    fine_edges, fine_edge_counts = np.unique(
        all_fine_edges, axis=0, return_counts=True
    )
    fine_boundary = np.zeros(len(uv), dtype=bool)
    fine_boundary[fine_edges[fine_edge_counts == 1].ravel()] = True
    within_cell = fine_edges[cell[fine_edges[:, 0]] == cell[fine_edges[:, 1]]]
    adjacency = coo_matrix(
        (np.ones(len(within_cell), dtype=np.uint8),
         (within_cell[:, 0], within_cell[:, 1])),
        shape=(len(uv), len(uv)),
    ).tocsr()
    component_count, labels = connected_components(adjacency, directed=False)

    coarse_labels = labels[faces]
    nondegenerate = (
        (coarse_labels[:, 0] != coarse_labels[:, 1])
        & (coarse_labels[:, 1] != coarse_labels[:, 2])
        & (coarse_labels[:, 2] != coarse_labels[:, 0])
    )
    coarse_labels = coarse_labels[nondegenerate]
    if not len(coarse_labels):
        raise ValueError("reference cloth mesh has no coarse faces")
    used_labels = np.unique(coarse_labels)
    remap = np.full(component_count, -1, dtype=np.int32)
    remap[used_labels] = np.arange(len(used_labels), dtype=np.int32)
    coarse_faces = remap[coarse_labels]
    _, first = np.unique(np.sort(coarse_faces, axis=1), axis=0, return_index=True)
    coarse_faces = coarse_faces[np.sort(first)]
    all_coarse_edges = np.concatenate(
        (coarse_faces[:, [0, 1]], coarse_faces[:, [1, 2]],
         coarse_faces[:, [2, 0]]), axis=0
    )
    all_coarse_edges.sort(axis=1)
    structural, coarse_edge_counts = np.unique(
        all_coarse_edges, axis=0, return_counts=True
    )
    coarse_boundary = np.unique(structural[coarse_edge_counts == 1])

    sizes = np.bincount(labels, minlength=component_count)
    centers = np.column_stack([
        np.bincount(labels, weights=uv[:, axis], minlength=component_count) / sizes
        for axis in range(2)
    ])
    vertex_ids = np.empty(len(used_labels), dtype=np.int32)
    for node, component in enumerate(used_labels):
        members = np.flatnonzero(labels == component)
        tracked = members[valid[members]]
        candidates = tracked if len(tracked) else members
        distances = np.sum(((uv[candidates] - centers[component]) / span)**2, axis=1)
        vertex_ids[node] = candidates[np.argmin(distances)]
    interior_vertex_ids = vertex_ids.copy()
    for node in coarse_boundary:
        members = np.flatnonzero((labels == used_labels[node]) & fine_boundary)
        if not len(members):
            continue
        tracked = members[valid[members]]
        candidates = tracked if len(tracked) else members
        center = np.mean(uv[members], axis=0)
        distances = np.sum((uv[candidates] - center)**2, axis=1)
        vertex_ids[node] = candidates[np.argmin(distances)]

    while True:
        coarse_uv = uv[vertex_ids[coarse_faces]]
        signed_area = np.cross(
            coarse_uv[:, 1] - coarse_uv[:, 0],
            coarse_uv[:, 2] - coarse_uv[:, 0],
        )
        degenerate = np.abs(signed_area) <= 1.0e-8
        if not np.any(degenerate):
            break
        affected = np.unique(coarse_faces[degenerate])
        restore = affected[vertex_ids[affected] != interior_vertex_ids[affected]]
        if not len(restore):
            raise ValueError("coarse cloth mesh contains degenerate faces")
        vertex_ids[restore] = interior_vertex_ids[restore]
    flipped = signed_area < 0.0
    coarse_faces[flipped] = coarse_faces[flipped][:, [0, 2, 1]]
    bending = _bending_edges(coarse_faces)
    if len(vertex_ids) < 16 or not len(structural):
        raise ValueError("coarse cloth mesh has too few connected vertices")
    return vertex_ids, coarse_faces, structural, bending


def _fit_static_obstacles(tracking_dir: Path) -> tuple[ClothObstacle, ...]:
    result = []
    for path in sorted((tracking_dir / "rigid_trajectories").glob("*.npz")):
        with np.load(path) as archive:
            if str(np.asarray(archive["motion_model"]).item()) != "static":
                continue
            points = np.asarray(archive["canonical_tracking_points"], np.float64)
            poses = np.asarray(archive["poses"], np.float64)
        valid = np.flatnonzero(np.isfinite(poses).all(axis=(1, 2)))
        if not len(valid) or len(points) < 8:
            continue
        pose = poses[int(valid[0])]
        world = points @ pose[:3, :3].T + pose[:3, 3]
        center = np.mean(world, axis=0)
        _, _, basis_t = np.linalg.svd(world - center, full_matrices=False)
        basis = basis_t.copy()
        if np.linalg.det(basis) < 0.0:
            basis[2] *= -1.0
        local = (world - center) @ basis.T
        lower, upper = np.percentile(local, (0.5, 99.5), axis=0)
        local_center = 0.5 * (lower + upper)
        center = center + local_center @ basis
        half = np.maximum(0.5 * (upper - lower), 0.005)
        result.append(ClothObstacle(
            object_id=path.stem,
            center=tuple(float(value) for value in center),
            axis_x=tuple(float(value) for value in basis[0]),
            axis_y=tuple(float(value) for value in basis[1]),
            axis_z=tuple(float(value) for value in basis[2]),
            half_extents=tuple(float(value) for value in half),
        ))
    return tuple(result)


def _camera_initial_depth_target(
    target: np.ndarray, initial: np.ndarray, camera_poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_poses)
    camera_target = np.einsum(
        "fij,fnj->fni", world_to_camera[:, :3, :3], target
    ) + world_to_camera[:, None, :3, 3]
    camera_initial = np.einsum(
        "fij,nj->fni", world_to_camera[:, :3, :3], initial
    ) + world_to_camera[:, None, :3, 3]
    positive_depth = (
        np.isfinite(camera_target).all(axis=-1)
        & (camera_target[..., 2] > 1.0e-5)
        & (camera_initial[..., 2] > 1.0e-5)
    )
    scale = np.ones_like(camera_target[..., 2])
    np.divide(
        camera_initial[..., 2], camera_target[..., 2],
        out=scale, where=positive_depth,
    )
    normalized_camera = camera_target * scale[..., None]
    normalized_world = np.einsum(
        "fij,fnj->fni", camera_poses[:, :3, :3], normalized_camera
    ) + camera_poses[:, None, :3, 3]
    return normalized_world, positive_depth


def load_cloth_scenes(
    tracking_dir: Path, *, grid_resolution: int = 16,
    minimum_tracking_confidence: float = 0.2,
    target_depth_mode: str = "camera-initial",
) -> tuple[tuple[ClothSceneData, ...], dict[str, Any]]:
    tracking_dir = Path(tracking_dir)
    if target_depth_mode not in {"raw", "camera-initial"}:
        raise ValueError(f"unsupported cloth target depth mode: {target_depth_mode}")
    if target_depth_mode == "camera-initial":
        inputs = json.loads((tracking_dir / "input_sources.json").read_text())
        manifest_path = Path(inputs["reconstruction_manifest"])
        manifest = json.loads(manifest_path.read_text())
        geometry_dir = manifest_path.parent / manifest["paths"]["geometry"]
        camera_poses = np.load(geometry_dir / "camera_poses.npy", mmap_mode="r")
    reconstruction_dir = tracking_dir.parent / "reconstruction"
    metadata = json.loads((tracking_dir / "run_metadata.json").read_text())
    gravity_payload = json.loads(
        (reconstruction_dir / "surface/gravity_estimate.json").read_text()
    )
    selected = gravity_payload["selected"]
    gravity_down = np.asarray(selected["gravity_down_world"], dtype=np.float64)
    gravity_down /= np.linalg.norm(gravity_down)
    support = selected["support_plane"]
    support_normal = np.asarray(support["normal_world"], dtype=np.float64)
    support_normal /= np.linalg.norm(support_normal)
    support_offset = float(support["offset_world"])
    obstacles = _fit_static_obstacles(tracking_dir)
    scenes = []
    for trajectory_path in sorted((tracking_dir / "trajectories").glob("*.npz")):
        object_id = trajectory_path.stem
        mesh_path = tracking_dir / "cloth_meshes" / trajectory_path.name
        if not mesh_path.is_file():
            continue
        with np.load(trajectory_path) as archive:
            reference_frame = int(archive["reference_frame"])
            positions = np.asarray(archive["model_xyz"], dtype=np.float64)
            confidence = np.asarray(archive["tracking_confidence"], dtype=np.float64)
        with np.load(mesh_path) as archive:
            vertex_uv = np.asarray(archive["vertex_uv"], dtype=np.float64)
            rest_positions_full = np.asarray(archive["vertices_world"], dtype=np.float64)
            source_faces = np.asarray(archive["faces"], dtype=np.int32)
        if positions.shape[1] != len(vertex_uv):
            raise ValueError(f"cloth mesh and trajectory vertex IDs differ: {object_id}")
        start_frame = reference_frame
        valid_start = (
            np.isfinite(positions[start_frame]).all(axis=-1)
            & (confidence[start_frame] >= minimum_tracking_confidence)
        )
        vertex_ids, faces, structural_edges, bending_edges = _sample_coarse_mesh(
            vertex_uv, rest_positions_full, valid_start, source_faces,
            grid_resolution=grid_resolution,
        )
        initial = positions[start_frame, vertex_ids].copy()
        initial[~valid_start[vertex_ids]] = rest_positions_full[
            vertex_ids[~valid_start[vertex_ids]]
        ]
        target = positions[start_frame:, vertex_ids].copy()
        raw_target = target.copy()
        weights = confidence[start_frame:, vertex_ids].copy()
        if target_depth_mode == "camera-initial":
            target, positive_depth = _camera_initial_depth_target(
                target,
                initial,
                np.asarray(camera_poses[start_frame:len(positions)]),
            )
            weights[~positive_depth] = 0.0
        valid = np.isfinite(target).all(axis=-1) & (
            weights >= minimum_tracking_confidence
        )
        weights = np.where(valid, np.clip(weights, 0.0, 1.0), 0.0)
        target[~valid] = 0.0
        raw_target[~valid] = 0.0
        rest = rest_positions_full[vertex_ids].copy()
        structural_rest = np.linalg.norm(
            rest[structural_edges[:, 1]] - rest[structural_edges[:, 0]], axis=-1
        )
        bending_rest = np.linalg.norm(
            rest[bending_edges[:, 1]] - rest[bending_edges[:, 0]], axis=-1
        )
        scenes.append(ClothSceneData(
            object_id=object_id,
            fps=float(metadata["fps"]),
            frames=tuple(range(start_frame, len(positions))),
            reference_frame=reference_frame,
            start_frame=start_frame,
            vertex_ids=vertex_ids,
            vertex_uv=vertex_uv[vertex_ids].astype(np.float32),
            initial_positions=initial.astype(np.float32),
            rest_positions=rest.astype(np.float32),
            target_positions=target.astype(np.float32),
            raw_target_positions=raw_target.astype(np.float32),
            target_weights=weights.astype(np.float32),
            faces=faces,
            structural_edges=structural_edges,
            structural_rest_lengths=structural_rest.astype(np.float32),
            bending_edges=bending_edges,
            bending_rest_lengths=bending_rest.astype(np.float32),
            gravity_down_world=gravity_down.astype(np.float32),
            support_normal_world=support_normal.astype(np.float32),
            support_offset_world=support_offset,
            obstacles=obstacles,
            source_trajectory=str(trajectory_path),
            source_mesh=str(mesh_path),
            target_depth_mode=target_depth_mode,
            camera_centers_world=(
                np.asarray(
                    camera_poses[start_frame:len(positions), :3, 3],
                    dtype=np.float32,
                ) if target_depth_mode == "camera-initial" else None
            ),
        ))
    if not scenes:
        raise ValueError(f"no cloth trajectories found in {tracking_dir}")
    source = {
        "tracking_dir": str(tracking_dir.resolve()),
        "reconstruction_dir": str(reconstruction_dir.resolve()),
        "run_metadata": str((tracking_dir / "run_metadata.json").resolve()),
        "grid_resolution": grid_resolution,
        "topology_source": "reference-mesh-face-contraction",
        "minimum_tracking_confidence": minimum_tracking_confidence,
        "target_depth_mode": target_depth_mode,
    }
    return tuple(scenes), source


def _boundary_shifts(
    scene: ClothSceneData, support: np.ndarray,
) -> np.ndarray:
    shifts = np.full((scene.frame_count, 3), np.nan, dtype=np.float64)
    shifts[0] = 0.0
    for frame in range(1, scene.frame_count):
        observed = support & (scene.target_weights[frame] > 0.0)
        if np.count_nonzero(observed) >= 3:
            displacements = (
                scene.target_positions[frame, observed]
                - scene.initial_positions[observed]
            )
            shifts[frame] = np.median(displacements, axis=0)
    frames = np.arange(scene.frame_count)
    for axis in range(3):
        valid = np.isfinite(shifts[:, axis])
        shifts[:, axis] = np.interp(frames, frames[valid], shifts[valid, axis])
    endpoint = shifts[-1].copy()
    shifts = gaussian_filter1d(shifts, sigma=1.0, axis=0, mode="nearest")
    shifts -= shifts[0]
    shifts += np.linspace(0.0, 1.0, scene.frame_count)[:, None] * (
        endpoint - shifts[-1]
    )
    return shifts


def _tracked_boundary(
    scene: ClothSceneData, mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    uv = scene.vertex_uv
    lower = np.min(uv, axis=0)
    upper = np.max(uv, axis=0)
    span = np.maximum(upper - lower, 1.0)
    positions = np.broadcast_to(
        scene.initial_positions, (scene.frame_count, scene.node_count, 3)
    ).copy().astype(np.float64)
    kinematic = np.zeros(scene.node_count, dtype=np.int32)
    if mode == "tracked-corners":
        for u in (lower[0], upper[0]):
            for v in (lower[1], upper[1]):
                distance = np.sum(((uv - [u, v]) / span) ** 2, axis=1)
                anchor = int(np.argmin(distance))
                support = (
                    (np.abs(uv[:, 0] - u) <= 0.3 * span[0])
                    & (np.abs(uv[:, 1] - v) <= 0.3 * span[1])
                )
                if np.count_nonzero(support) < 3:
                    support = np.zeros(scene.node_count, dtype=bool)
                    support[np.argsort(distance)[:6]] = True
                kinematic[anchor] = 1
                positions[:, anchor] += _boundary_shifts(
                    scene, support
                )
    else:
        vertical_span = span[1]
        bands = [(
            uv[:, 1] <= lower[1] + 0.10 * vertical_span,
            uv[:, 1] <= lower[1] + 0.25 * vertical_span,
        )]
        if mode in {"tracked-edges", "tracked-edges-release"}:
            bands.append((
                uv[:, 1] >= upper[1] - 0.10 * vertical_span,
                uv[:, 1] >= upper[1] - 0.25 * vertical_span,
            ))
        for anchors, support in bands:
            kinematic[anchors] = 1
            positions[:, anchors] += _boundary_shifts(
                scene, support
            )[:, None, :]
    return kinematic, positions.astype(np.float32)


def _edge_drive_basis(
    scene: ClothSceneData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv = scene.vertex_uv
    lower = np.min(uv[:, 1])
    upper = np.max(uv[:, 1])
    span = max(float(upper - lower), 1.0)
    driven = uv[:, 1] <= lower + 0.10 * span
    opposite = uv[:, 1] >= upper - 0.10 * span
    if not np.any(driven) or not np.any(opposite):
        raise ValueError("cloth mesh is missing a drive edge")
    direction = (
        np.median(scene.initial_positions[opposite], axis=0)
        - np.median(scene.initial_positions[driven], axis=0)
    )
    normal = scene.support_normal_world
    direction -= np.dot(direction, normal) * normal
    length = np.linalg.norm(direction)
    if length < 1.0e-6:
        raise ValueError("cloth drive direction is undefined")
    inward = direction / length
    sideways = np.cross(normal, inward)
    sideways /= np.linalg.norm(sideways)
    return (
        driven.astype(np.int32), inward.astype(np.float32),
        sideways.astype(np.float32),
    )


def _released_boundary_mask(
    scene: ClothSceneData, initial_mask: np.ndarray,
) -> np.ndarray:
    mask = np.broadcast_to(
        initial_mask, (scene.frame_count, scene.node_count)
    ).copy()
    uv = scene.vertex_uv
    vertical_span = float(np.ptp(uv[:, 1]))
    horizontal = scene.initial_positions[np.argmax(uv[:, 0])] - (
        scene.initial_positions[np.argmin(uv[:, 0])]
    )
    horizontal /= np.linalg.norm(horizontal)
    for top in (True, False):
        if top:
            anchors = uv[:, 1] <= np.min(uv[:, 1]) + 0.10 * vertical_span
            support = uv[:, 1] <= np.min(uv[:, 1]) + 0.25 * vertical_span
        else:
            anchors = uv[:, 1] >= np.max(uv[:, 1]) - 0.10 * vertical_span
            support = uv[:, 1] >= np.max(uv[:, 1]) - 0.25 * vertical_span
        initial_projection = scene.initial_positions[support] @ horizontal
        initial_span = np.percentile(initial_projection, 90) - np.percentile(
            initial_projection, 10
        )
        if initial_span <= 1.0e-6:
            continue
        streak = 0
        for frame in range(1, scene.frame_count):
            observed = support & (scene.target_weights[frame] > 0.0)
            if np.count_nonzero(observed) < 5:
                streak += 1
            else:
                projected = scene.target_positions[frame, observed] @ horizontal
                span = np.percentile(projected, 90) - np.percentile(
                    projected, 10
                )
                streak = streak + 1 if span < 0.8 * initial_span else 0
            if streak >= 2:
                mask[frame - 1:, anchors] = 0
                break
    return mask


@wp.func
def _project_obb_particle(
    position: wp.vec3, center: wp.vec3, axis_x: wp.vec3, axis_y: wp.vec3,
    axis_z: wp.vec3, half: wp.vec3, thickness: float,
) -> wp.vec3:
    offset = position - center
    local = wp.vec3(
        wp.dot(offset, axis_x), wp.dot(offset, axis_y), wp.dot(offset, axis_z)
    )
    expanded = half + wp.vec3(thickness, thickness, thickness)
    distance = wp.vec3(
        wp.abs(local[0]) - expanded[0],
        wp.abs(local[1]) - expanded[1],
        wp.abs(local[2]) - expanded[2],
    )
    outside = wp.vec3(
        wp.max(distance[0], 0.0),
        wp.max(distance[1], 0.0),
        wp.max(distance[2], 0.0),
    )
    outside_length = wp.length(outside)
    if outside_length > 0.0:
        return position
    face = 0
    clearance = -distance[0]
    if -distance[1] < clearance:
        face = 1
        clearance = -distance[1]
    if -distance[2] < clearance:
        face = 2
        clearance = -distance[2]
    sign = 1.0 if local[face] >= 0.0 else -1.0
    normal = axis_x
    if face == 1:
        normal = axis_y
    elif face == 2:
        normal = axis_z
    return position + sign * clearance * normal


@wp.func
def _rotate_vector(vector: wp.vec3, rotation: wp.vec3) -> wp.vec3:
    angle = wp.length(rotation)
    if angle < 1.0e-8:
        return vector + wp.cross(rotation, vector)
    axis = rotation / angle
    return (
        wp.cos(angle) * vector
        + wp.sin(angle) * wp.cross(axis, vector)
        + (1.0 - wp.cos(angle)) * wp.dot(axis, vector) * axis
    )


@wp.func
def _box_inverse_inertia(
    torque: wp.vec3, axis_x: wp.vec3, axis_y: wp.vec3, axis_z: wp.vec3,
    half: wp.vec3, inverse_mass: float,
) -> wp.vec3:
    return 3.0 * inverse_mass * (
        wp.dot(torque, axis_x) / (half[1] * half[1] + half[2] * half[2]) * axis_x
        + wp.dot(torque, axis_y) / (half[0] * half[0] + half[2] * half[2]) * axis_y
        + wp.dot(torque, axis_z) / (half[0] * half[0] + half[1] * half[1]) * axis_z
    )


@wp.func
def _box_corner_lever(
    corner_index: int, axis_x: wp.vec3, axis_y: wp.vec3,
    axis_z: wp.vec3, half: wp.vec3,
) -> wp.vec3:
    sign_x = -1.0
    sign_y = -1.0
    sign_z = -1.0
    if corner_index % 2 != 0:
        sign_x = 1.0
    if (corner_index // 2) % 2 != 0:
        sign_y = 1.0
    if corner_index // 4 != 0:
        sign_z = 1.0
    return (
        sign_x * half[0] * axis_x
        + sign_y * half[1] * axis_y
        + sign_z * half[2] * axis_z
    )


@wp.func
def _solve_box_floor_contact(
    world: int, obstacle: int, dt: float, normal: wp.vec3, offset: float,
    half: wp.vec3, inverse_mass: float,
    centers: wp.array2d(dtype=wp.vec3),
    axis_x_values: wp.array2d(dtype=wp.vec3),
    axis_y_values: wp.array2d(dtype=wp.vec3),
    axis_z_values: wp.array2d(dtype=wp.vec3),
    velocity_values: wp.array2d(dtype=wp.vec3),
    angular_values: wp.array2d(dtype=wp.vec3),
):
    center = centers[world, obstacle]
    axis_x = axis_x_values[world, obstacle]
    axis_y = axis_y_values[world, obstacle]
    axis_z = axis_z_values[world, obstacle]
    velocity = velocity_values[world, obstacle]
    angular = angular_values[world, obstacle]
    maximum_depth = -1.0e6
    for corner_index in range(8):
        lever = _box_corner_lever(
            corner_index, axis_x, axis_y, axis_z, half,
        )
        depth = offset - wp.dot(center + lever, normal)
        maximum_depth = wp.max(maximum_depth, depth)
    if maximum_depth > 0.0:
        manifold_lever = wp.vec3(0.0, 0.0, 0.0)
        manifold_count = 0.0
        for corner_index in range(8):
            lever = _box_corner_lever(
                corner_index, axis_x, axis_y, axis_z, half,
            )
            depth = offset - wp.dot(center + lever, normal)
            if depth >= maximum_depth - 1.0e-4:
                manifold_lever += lever
                manifold_count += 1.0
        manifold_lever /= wp.max(manifold_count, 1.0)
        normal_torque = wp.cross(manifold_lever, normal)
        inverse_normal_torque = _box_inverse_inertia(
            normal_torque, axis_x, axis_y, axis_z, half, inverse_mass,
        )
        denominator = inverse_mass + wp.dot(
            normal_torque, inverse_normal_torque
        )
        position_impulse = maximum_depth / wp.max(denominator, 1.0e-8)
        center += inverse_mass * position_impulse * normal
        angle_delta = position_impulse * inverse_normal_torque
        axis_x = _rotate_vector(axis_x, angle_delta)
        axis_y = _rotate_vector(axis_y, angle_delta)
        axis_z = _rotate_vector(axis_z, angle_delta)

        contact_velocity = velocity + wp.cross(angular, manifold_lever)
        normal_speed = wp.dot(contact_velocity, normal)
        normal_impulse = 0.0
        if normal_speed < 0.0:
            normal_impulse = -normal_speed / wp.max(denominator, 1.0e-8)
            velocity += inverse_mass * normal_impulse * normal
            angular += normal_impulse * inverse_normal_torque

        contact_velocity = velocity + wp.cross(angular, manifold_lever)
        tangent_velocity = contact_velocity - wp.dot(
            contact_velocity, normal
        ) * normal
        tangent_speed = wp.length(tangent_velocity)
        if tangent_speed > 1.0e-6 and normal_impulse > 0.0:
            tangent_direction = tangent_velocity / tangent_speed
            tangent_torque = wp.cross(manifold_lever, tangent_direction)
            inverse_tangent_torque = _box_inverse_inertia(
                tangent_torque, axis_x, axis_y, axis_z,
                half, inverse_mass,
            )
            tangent_denominator = inverse_mass + wp.dot(
                tangent_torque, inverse_tangent_torque
            )
            static_impulse = tangent_speed / wp.max(
                tangent_denominator, 1.0e-8
            )
            friction_impulse = static_impulse
            if static_impulse > float(GROUND_STATIC_FRICTION) * normal_impulse:
                friction_impulse = float(GROUND_DYNAMIC_FRICTION) * normal_impulse
            velocity -= inverse_mass * friction_impulse * tangent_direction
            angular -= friction_impulse * inverse_tangent_torque
    centers[world, obstacle] = center
    axis_x_values[world, obstacle] = axis_x
    axis_y_values[world, obstacle] = axis_y
    axis_z_values[world, obstacle] = axis_z
    velocity_values[world, obstacle] = velocity
    angular_values[world, obstacle] = angular


@wp.func
def _obb_separating_axis(
    index: int, ax: wp.vec3, ay: wp.vec3, az: wp.vec3,
    bx: wp.vec3, by: wp.vec3, bz: wp.vec3,
) -> wp.vec3:
    result = ax
    if index == 1:
        result = ay
    elif index == 2:
        result = az
    elif index == 3:
        result = bx
    elif index == 4:
        result = by
    elif index == 5:
        result = bz
    elif index >= 6:
        first = ax
        second = bx
        row = (index - 6) // 3
        column = (index - 6) % 3
        if row == 1:
            first = ay
        elif row == 2:
            first = az
        if column == 1:
            second = by
        elif column == 2:
            second = bz
        result = wp.cross(first, second)
    return result


@wp.func
def _obb_contact_point(
    center: wp.vec3, ax: wp.vec3, ay: wp.vec3, az: wp.vec3,
    half: wp.vec3, direction: wp.vec3,
) -> wp.vec3:
    result = center
    dx = wp.dot(ax, direction)
    dy = wp.dot(ay, direction)
    dz = wp.dot(az, direction)
    if dx > 0.05:
        result += half[0] * ax
    elif dx < -0.05:
        result -= half[0] * ax
    if dy > 0.05:
        result += half[1] * ay
    elif dy < -0.05:
        result -= half[1] * ay
    if dz > 0.05:
        result += half[2] * az
    elif dz < -0.05:
        result -= half[2] * az
    return result


@wp.kernel
def _simulate_cloth_worlds(
    theta: wp.array2d(dtype=float),
    initial_positions: wp.array(dtype=wp.vec3),
    gravity_down: wp.vec3,
    support_normal: wp.vec3,
    support_offset: float,
    support_mask: wp.array(dtype=int),
    obstacle_centers: wp.array(dtype=wp.vec3),
    obstacle_axis_x: wp.array(dtype=wp.vec3),
    obstacle_axis_y: wp.array(dtype=wp.vec3),
    obstacle_axis_z: wp.array(dtype=wp.vec3),
    obstacle_half: wp.array(dtype=wp.vec3),
    obstacle_inverse_volume: wp.array(dtype=float),
    kinematic_mask: wp.array2d(dtype=int),
    kinematic_positions: wp.array2d(dtype=wp.vec3),
    drive_inward: wp.vec3,
    drive_sideways: wp.vec3,
    edge_drive: int,
    structural_first: wp.array(dtype=int),
    structural_second: wp.array(dtype=int),
    structural_rest: wp.array(dtype=float),
    bending_first: wp.array(dtype=int),
    bending_second: wp.array(dtype=int),
    bending_rest: wp.array(dtype=float),
    node_count: int,
    structural_count: int,
    bending_count: int,
    obstacle_count: int,
    dynamic_obstacles: int,
    rigid_mass_parameter: int,
    frame_count: int,
    fps: float,
    substeps: int,
    constraint_iterations: int,
    thickness: float,
    positions: wp.array2d(dtype=wp.vec3),
    previous_positions: wp.array2d(dtype=wp.vec3),
    velocities: wp.array2d(dtype=wp.vec3),
    predicted: wp.array3d(dtype=wp.vec3),
    rigid_centers: wp.array2d(dtype=wp.vec3),
    rigid_axis_x: wp.array2d(dtype=wp.vec3),
    rigid_axis_y: wp.array2d(dtype=wp.vec3),
    rigid_axis_z: wp.array2d(dtype=wp.vec3),
    rigid_velocities: wp.array2d(dtype=wp.vec3),
    rigid_angular_velocities: wp.array2d(dtype=wp.vec3),
    predicted_rigid_centers: wp.array3d(dtype=wp.vec3),
    predicted_rigid_axis_x: wp.array3d(dtype=wp.vec3),
    predicted_rigid_axis_y: wp.array3d(dtype=wp.vec3),
    predicted_rigid_axis_z: wp.array3d(dtype=wp.vec3),
):
    world = wp.tid()
    dt = 1.0 / (fps * float(substeps))
    gravity_scale = wp.exp(theta[world, 3])
    structural_stiffness = theta[world, 4]
    bending_stiffness = theta[world, 5]
    drag = theta[world, 6]
    damping = wp.exp(-drag * dt)
    initial_velocity = wp.vec3(
        theta[world, 0], theta[world, 1], theta[world, 2]
    )
    for node in range(node_count):
        positions[world, node] = initial_positions[node]
        velocities[world, node] = initial_velocity
        predicted[world, 0, node] = initial_positions[node]
    for obstacle in range(obstacle_count):
        rigid_centers[world, obstacle] = obstacle_centers[obstacle]
        rigid_axis_x[world, obstacle] = obstacle_axis_x[obstacle]
        rigid_axis_y[world, obstacle] = obstacle_axis_y[obstacle]
        rigid_axis_z[world, obstacle] = obstacle_axis_z[obstacle]
        rigid_velocities[world, obstacle] = wp.vec3(0.0, 0.0, 0.0)
        rigid_angular_velocities[world, obstacle] = wp.vec3(0.0, 0.0, 0.0)
        predicted_rigid_centers[world, 0, obstacle] = obstacle_centers[obstacle]
        predicted_rigid_axis_x[world, 0, obstacle] = obstacle_axis_x[obstacle]
        predicted_rigid_axis_y[world, 0, obstacle] = obstacle_axis_y[obstacle]
        predicted_rigid_axis_z[world, 0, obstacle] = obstacle_axis_z[obstacle]
    for frame in range(1, frame_count):
        released = False
        if edge_drive != 0:
            released = float(frame) >= theta[world, 10]
        for substep in range(substeps):
            if dynamic_obstacles != 0:
                for obstacle in range(obstacle_count):
                    center = rigid_centers[world, obstacle]
                    velocity = rigid_velocities[world, obstacle]
                    angular = rigid_angular_velocities[world, obstacle]
                    axis_x = rigid_axis_x[world, obstacle]
                    axis_y = rigid_axis_y[world, obstacle]
                    axis_z = rigid_axis_z[world, obstacle]
                    half = obstacle_half[obstacle]
                    velocity += dt * float(GRAVITY_M_PER_S2) * gravity_down
                    center += dt * velocity
                    rotation = dt * angular
                    axis_x = _rotate_vector(axis_x, rotation)
                    axis_y = _rotate_vector(axis_y, rotation)
                    axis_z = _rotate_vector(axis_z, rotation)
                    rigid_centers[world, obstacle] = center
                    rigid_axis_x[world, obstacle] = axis_x
                    rigid_axis_y[world, obstacle] = axis_y
                    rigid_axis_z[world, obstacle] = axis_z
                    rigid_velocities[world, obstacle] = velocity
                    rigid_angular_velocities[world, obstacle] = angular
                    inverse_mass = wp.exp(
                        -theta[world, rigid_mass_parameter]
                    ) * obstacle_inverse_volume[obstacle]
                    _solve_box_floor_contact(
                        world, obstacle, dt, support_normal, support_offset,
                        half, inverse_mass, rigid_centers, rigid_axis_x,
                        rigid_axis_y, rigid_axis_z, rigid_velocities,
                        rigid_angular_velocities,
                    )
            for node in range(node_count):
                position = positions[world, node]
                previous_positions[world, node] = position
                if kinematic_mask[frame, node] != 0 and not released:
                    fraction = float(substep + 1) / float(substeps)
                    anchor = (
                        (1.0 - fraction) * kinematic_positions[frame - 1, node]
                        + fraction * kinematic_positions[frame, node]
                    )
                    if edge_drive != 0:
                        drive_frame = wp.min(
                            float(frame - 1) + fraction, theta[world, 9]
                        )
                        drive_time = drive_frame / fps
                        anchor = initial_positions[node] + (
                            theta[world, 7] * drive_inward
                            + theta[world, 8] * drive_sideways
                        ) * drive_time
                    positions[world, node] = anchor
                    velocities[world, node] = (anchor - position) / dt
                else:
                    velocity = damping * velocities[world, node]
                    velocity += (
                        dt * float(GRAVITY_M_PER_S2) * gravity_scale * gravity_down
                    )
                    position += dt * velocity
                    positions[world, node] = position
                    velocities[world, node] = velocity
            for _iteration in range(constraint_iterations):
                for edge in range(structural_count):
                    first = structural_first[edge]
                    second = structural_second[edge]
                    delta = positions[world, second] - positions[world, first]
                    length = wp.length(delta)
                    if length > 1.0e-8:
                        correction = 0.5 * structural_stiffness * (
                            length - structural_rest[edge]
                        ) * delta / length
                        if kinematic_mask[frame, first] == 0 or released:
                            factor = 2.0 if kinematic_mask[frame, second] != 0 and not released else 1.0
                            positions[world, first] += factor * correction
                        if kinematic_mask[frame, second] == 0 or released:
                            factor = 2.0 if kinematic_mask[frame, first] != 0 and not released else 1.0
                            positions[world, second] -= factor * correction
                for edge in range(bending_count):
                    first = bending_first[edge]
                    second = bending_second[edge]
                    delta = positions[world, second] - positions[world, first]
                    length = wp.length(delta)
                    if length > 1.0e-8:
                        correction = 0.5 * bending_stiffness * (
                            length - bending_rest[edge]
                        ) * delta / length
                        if kinematic_mask[frame, first] == 0 or released:
                            factor = 2.0 if kinematic_mask[frame, second] != 0 and not released else 1.0
                            positions[world, first] += factor * correction
                        if kinematic_mask[frame, second] == 0 or released:
                            factor = 2.0 if kinematic_mask[frame, first] != 0 and not released else 1.0
                            positions[world, second] -= factor * correction
                for node in range(node_count):
                    kinematic = kinematic_mask[frame, node] != 0 and not released
                    if kinematic and dynamic_obstacles == 0:
                        continue
                    position = positions[world, node]
                    signed_height = wp.dot(support_normal, position) - support_offset
                    if not kinematic and support_mask[node] != 0 and signed_height < thickness:
                        position += (thickness - signed_height) * support_normal
                    for obstacle in range(obstacle_count):
                        if dynamic_obstacles == 0:
                            position = _project_obb_particle(
                                position, obstacle_centers[obstacle],
                                obstacle_axis_x[obstacle], obstacle_axis_y[obstacle],
                                obstacle_axis_z[obstacle], obstacle_half[obstacle],
                                thickness,
                            )
                        else:
                            center = rigid_centers[world, obstacle]
                            axis_x = rigid_axis_x[world, obstacle]
                            axis_y = rigid_axis_y[world, obstacle]
                            axis_z = rigid_axis_z[world, obstacle]
                            half = obstacle_half[obstacle]
                            projection = _project_obb_particle(
                                position, center, axis_x, axis_y, axis_z,
                                half, thickness,
                            )
                            separation = projection - position
                            depth = wp.length(separation)
                            if depth > 1.0e-8:
                                normal = separation / depth
                                inverse_mass = wp.exp(-theta[world, rigid_mass_parameter]) * obstacle_inverse_volume[obstacle]
                                lever = projection - center
                                torque_axis = wp.cross(lever, normal)
                                inverse_inertia_torque = _box_inverse_inertia(
                                    torque_axis, axis_x, axis_y, axis_z,
                                    half, inverse_mass,
                                )
                                particle_inverse_mass = 1.0
                                if kinematic:
                                    particle_inverse_mass = 0.0
                                denominator = particle_inverse_mass + inverse_mass + wp.dot(torque_axis, inverse_inertia_torque)
                                impulse = depth / wp.max(denominator, 1.0e-8)
                                position += particle_inverse_mass * impulse * normal
                                center -= inverse_mass * impulse * normal
                                angle_delta = -impulse * inverse_inertia_torque
                                rigid_centers[world, obstacle] = center
                                rigid_axis_x[world, obstacle] = _rotate_vector(axis_x, angle_delta)
                                rigid_axis_y[world, obstacle] = _rotate_vector(axis_y, angle_delta)
                                rigid_axis_z[world, obstacle] = _rotate_vector(axis_z, angle_delta)
                                rigid_velocities[world, obstacle] -= inverse_mass * impulse * normal / dt
                                rigid_angular_velocities[world, obstacle] += angle_delta / dt
                                surface_velocity = (
                                    rigid_velocities[world, obstacle]
                                    + wp.cross(rigid_angular_velocities[world, obstacle], lever)
                                )
                                # PBD constraints change displacement during each
                                # iteration; friction must use the remaining slip.
                                particle_velocity = (
                                    position - previous_positions[world, node]
                                ) / dt
                                relative_velocity = particle_velocity - surface_velocity
                                tangent_velocity = relative_velocity - wp.dot(relative_velocity, normal) * normal
                                tangent_speed = wp.length(tangent_velocity)
                                if tangent_speed > 1.0e-6:
                                    tangent_direction = tangent_velocity / tangent_speed
                                    tangent_axis = wp.cross(lever, tangent_direction)
                                    inverse_tangent_inertia = _box_inverse_inertia(
                                        tangent_axis, axis_x, axis_y, axis_z,
                                        half, inverse_mass,
                                    )
                                    tangent_denom = (
                                        particle_inverse_mass + inverse_mass
                                        + wp.dot(tangent_axis, inverse_tangent_inertia)
                                    )
                                    tangent_impulse = wp.min(
                                        tangent_speed / wp.max(tangent_denom, 1.0e-8),
                                        float(CLOTH_RIGID_FRICTION) * impulse / dt,
                                    )
                                    position -= particle_inverse_mass * tangent_impulse * dt * tangent_direction
                                    center += inverse_mass * tangent_impulse * dt * tangent_direction
                                    tangent_angle = dt * tangent_impulse * inverse_tangent_inertia
                                    rigid_centers[world, obstacle] = center
                                    rigid_axis_x[world, obstacle] = _rotate_vector(
                                        rigid_axis_x[world, obstacle], tangent_angle
                                    )
                                    rigid_axis_y[world, obstacle] = _rotate_vector(
                                        rigid_axis_y[world, obstacle], tangent_angle
                                    )
                                    rigid_axis_z[world, obstacle] = _rotate_vector(
                                        rigid_axis_z[world, obstacle], tangent_angle
                                    )
                                    rigid_velocities[world, obstacle] += inverse_mass * tangent_impulse * tangent_direction
                                    rigid_angular_velocities[world, obstacle] += tangent_impulse * inverse_tangent_inertia
                    positions[world, node] = position
                if dynamic_obstacles != 0:
                    for first_obstacle in range(obstacle_count):
                        for second_obstacle in range(first_obstacle + 1, obstacle_count):
                            pair_a_center = rigid_centers[world, first_obstacle]
                            pair_b_center = rigid_centers[world, second_obstacle]
                            pair_ax = rigid_axis_x[world, first_obstacle]
                            pair_ay = rigid_axis_y[world, first_obstacle]
                            pair_az = rigid_axis_z[world, first_obstacle]
                            pair_bx = rigid_axis_x[world, second_obstacle]
                            pair_by = rigid_axis_y[world, second_obstacle]
                            pair_bz = rigid_axis_z[world, second_obstacle]
                            pair_a_half = obstacle_half[first_obstacle]
                            pair_b_half = obstacle_half[second_obstacle]
                            pair_delta = pair_b_center - pair_a_center
                            pair_depth = 1.0e6
                            pair_normal = wp.vec3(1.0, 0.0, 0.0)
                            pair_separated = False
                            for axis_index in range(15):
                                candidate_axis = _obb_separating_axis(
                                    axis_index, pair_ax, pair_ay, pair_az,
                                    pair_bx, pair_by, pair_bz,
                                )
                                axis_length = wp.length(candidate_axis)
                                if axis_length > 1.0e-6:
                                    candidate_axis /= axis_length
                                    if wp.dot(candidate_axis, pair_delta) < 0.0:
                                        candidate_axis = -candidate_axis
                                    radius_a = (
                                        pair_a_half[0] * wp.abs(wp.dot(candidate_axis, pair_ax))
                                        + pair_a_half[1] * wp.abs(wp.dot(candidate_axis, pair_ay))
                                        + pair_a_half[2] * wp.abs(wp.dot(candidate_axis, pair_az))
                                    )
                                    radius_b = (
                                        pair_b_half[0] * wp.abs(wp.dot(candidate_axis, pair_bx))
                                        + pair_b_half[1] * wp.abs(wp.dot(candidate_axis, pair_by))
                                        + pair_b_half[2] * wp.abs(wp.dot(candidate_axis, pair_bz))
                                    )
                                    candidate_depth = (
                                        radius_a + radius_b
                                        - wp.dot(candidate_axis, pair_delta)
                                    )
                                    if candidate_depth < 0.0:
                                        pair_separated = True
                                    if candidate_depth < pair_depth:
                                        pair_depth = candidate_depth
                                        pair_normal = candidate_axis
                            if not pair_separated and pair_depth > 1.0e-6:
                                pair_a_point = _obb_contact_point(
                                    pair_a_center, pair_ax, pair_ay, pair_az,
                                    pair_a_half, pair_normal,
                                )
                                pair_b_point = _obb_contact_point(
                                    pair_b_center, pair_bx, pair_by, pair_bz,
                                    pair_b_half, -pair_normal,
                                )
                                pair_a_lever = pair_a_point - pair_a_center
                                pair_b_lever = pair_b_point - pair_b_center
                                pair_a_inverse_mass = wp.exp(-theta[world, rigid_mass_parameter]) * obstacle_inverse_volume[first_obstacle]
                                pair_b_inverse_mass = wp.exp(-theta[world, rigid_mass_parameter]) * obstacle_inverse_volume[second_obstacle]
                                pair_a_torque = wp.cross(pair_a_lever, pair_normal)
                                pair_b_torque = wp.cross(pair_b_lever, pair_normal)
                                pair_a_inverse_torque = _box_inverse_inertia(
                                    pair_a_torque, pair_ax, pair_ay, pair_az,
                                    pair_a_half, pair_a_inverse_mass,
                                )
                                pair_b_inverse_torque = _box_inverse_inertia(
                                    pair_b_torque, pair_bx, pair_by, pair_bz,
                                    pair_b_half, pair_b_inverse_mass,
                                )
                                pair_denom = (
                                    pair_a_inverse_mass + pair_b_inverse_mass
                                    + wp.dot(pair_a_torque, pair_a_inverse_torque)
                                    + wp.dot(pair_b_torque, pair_b_inverse_torque)
                                )
                                pair_lambda = pair_depth / wp.max(pair_denom, 1.0e-8)
                                pair_a_center -= pair_a_inverse_mass * pair_lambda * pair_normal
                                pair_b_center += pair_b_inverse_mass * pair_lambda * pair_normal
                                pair_a_angle = -pair_lambda * pair_a_inverse_torque
                                pair_b_angle = pair_lambda * pair_b_inverse_torque
                                pair_ax = _rotate_vector(pair_ax, pair_a_angle)
                                pair_ay = _rotate_vector(pair_ay, pair_a_angle)
                                pair_az = _rotate_vector(pair_az, pair_a_angle)
                                pair_bx = _rotate_vector(pair_bx, pair_b_angle)
                                pair_by = _rotate_vector(pair_by, pair_b_angle)
                                pair_bz = _rotate_vector(pair_bz, pair_b_angle)
                                pair_a_velocity = rigid_velocities[world, first_obstacle]
                                pair_b_velocity = rigid_velocities[world, second_obstacle]
                                pair_a_angular = rigid_angular_velocities[world, first_obstacle]
                                pair_b_angular = rigid_angular_velocities[world, second_obstacle]
                                pair_relative = (
                                    pair_b_velocity + wp.cross(pair_b_angular, pair_b_lever)
                                    - pair_a_velocity - wp.cross(pair_a_angular, pair_a_lever)
                                )
                                pair_normal_speed = wp.dot(pair_relative, pair_normal)
                                pair_normal_impulse = 0.0
                                if pair_normal_speed < 0.0:
                                    pair_normal_impulse = -pair_normal_speed / wp.max(pair_denom, 1.0e-8)
                                    pair_a_velocity -= pair_a_inverse_mass * pair_normal_impulse * pair_normal
                                    pair_b_velocity += pair_b_inverse_mass * pair_normal_impulse * pair_normal
                                    pair_a_angular -= pair_normal_impulse * pair_a_inverse_torque
                                    pair_b_angular += pair_normal_impulse * pair_b_inverse_torque
                                    pair_relative = (
                                        pair_b_velocity + wp.cross(pair_b_angular, pair_b_lever)
                                        - pair_a_velocity - wp.cross(pair_a_angular, pair_a_lever)
                                    )
                                pair_tangent = pair_relative - wp.dot(pair_relative, pair_normal) * pair_normal
                                pair_tangent_speed = wp.length(pair_tangent)
                                if pair_tangent_speed > 1.0e-6:
                                    pair_friction_dir = -pair_tangent / pair_tangent_speed
                                    pair_a_friction_torque = wp.cross(pair_a_lever, pair_friction_dir)
                                    pair_b_friction_torque = wp.cross(pair_b_lever, pair_friction_dir)
                                    pair_a_inverse_friction = _box_inverse_inertia(
                                        pair_a_friction_torque, pair_ax, pair_ay, pair_az,
                                        pair_a_half, pair_a_inverse_mass,
                                    )
                                    pair_b_inverse_friction = _box_inverse_inertia(
                                        pair_b_friction_torque, pair_bx, pair_by, pair_bz,
                                        pair_b_half, pair_b_inverse_mass,
                                    )
                                    pair_friction_denom = (
                                        pair_a_inverse_mass + pair_b_inverse_mass
                                        + wp.dot(pair_a_friction_torque, pair_a_inverse_friction)
                                        + wp.dot(pair_b_friction_torque, pair_b_inverse_friction)
                                    )
                                    pair_friction_impulse = wp.min(
                                        pair_tangent_speed / wp.max(pair_friction_denom, 1.0e-8),
                                        float(RIGID_RIGID_FRICTION) * pair_normal_impulse,
                                    )
                                    pair_a_velocity -= pair_a_inverse_mass * pair_friction_impulse * pair_friction_dir
                                    pair_b_velocity += pair_b_inverse_mass * pair_friction_impulse * pair_friction_dir
                                    pair_a_angular -= pair_friction_impulse * pair_a_inverse_friction
                                    pair_b_angular += pair_friction_impulse * pair_b_inverse_friction
                                rigid_centers[world, first_obstacle] = pair_a_center
                                rigid_centers[world, second_obstacle] = pair_b_center
                                rigid_axis_x[world, first_obstacle] = pair_ax
                                rigid_axis_y[world, first_obstacle] = pair_ay
                                rigid_axis_z[world, first_obstacle] = pair_az
                                rigid_axis_x[world, second_obstacle] = pair_bx
                                rigid_axis_y[world, second_obstacle] = pair_by
                                rigid_axis_z[world, second_obstacle] = pair_bz
                                rigid_velocities[world, first_obstacle] = pair_a_velocity
                                rigid_velocities[world, second_obstacle] = pair_b_velocity
                                rigid_angular_velocities[world, first_obstacle] = pair_a_angular
                                rigid_angular_velocities[world, second_obstacle] = pair_b_angular
                    for obstacle in range(obstacle_count):
                        inverse_mass = wp.exp(
                            -theta[world, rigid_mass_parameter]
                        ) * obstacle_inverse_volume[obstacle]
                        _solve_box_floor_contact(
                            world, obstacle, dt, support_normal, support_offset,
                            obstacle_half[obstacle], inverse_mass,
                            rigid_centers, rigid_axis_x, rigid_axis_y,
                            rigid_axis_z, rigid_velocities,
                            rigid_angular_velocities,
                        )
            for node in range(node_count):
                if kinematic_mask[frame, node] == 0 or released:
                    velocities[world, node] = (
                        positions[world, node] - previous_positions[world, node]
                    ) / dt
        for node in range(node_count):
            predicted[world, frame, node] = positions[world, node]
        for obstacle in range(obstacle_count):
            predicted_rigid_centers[world, frame, obstacle] = rigid_centers[world, obstacle]
            predicted_rigid_axis_x[world, frame, obstacle] = rigid_axis_x[world, obstacle]
            predicted_rigid_axis_y[world, frame, obstacle] = rigid_axis_y[world, obstacle]
            predicted_rigid_axis_z[world, frame, obstacle] = rigid_axis_z[world, obstacle]


def parameter_bounds(
    *, edge_drive: bool = False, frame_count: int | None = None,
    dynamic_obstacles: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [-1.5, -1.5, -1.5, math.log(0.05), 0.08, 0.0, 0.0],
        dtype=np.float32,
    )
    upper = np.asarray(
        [1.5, 1.5, 1.5, math.log(1.5), 1.0, 0.6, 6.0],
        dtype=np.float32,
    )
    if edge_drive:
        if frame_count is None:
            raise ValueError("edge drive bounds require frame count")
        lower = np.append(lower, np.asarray([-1.5, -1.5, 0.0, 0.0], np.float32))
        upper = np.append(
            upper, np.asarray([1.5, 1.5, frame_count, frame_count], np.float32)
        )
    if dynamic_obstacles:
        lower = np.append(lower, np.float32(math.log(2.0)))
        upper = np.append(upper, np.float32(math.log(6400.0)))
    return lower, upper


def initial_distribution(
    *, edge_drive: bool = False, frame_count: int | None = None,
    dynamic_obstacles: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(
        [0.0, 0.0, 0.0, math.log(0.6), 0.75, 0.08, 0.8],
        dtype=np.float32,
    )
    std = np.asarray([0.35, 0.35, 0.35, 0.6, 0.2, 0.12, 0.8], np.float32)
    if edge_drive:
        if frame_count is None:
            raise ValueError("edge drive initialization requires frame count")
        mean = np.append(
            mean, np.asarray([
                0.3, 0.0, min(24.0, 0.3 * frame_count),
                min(40.0, 0.5 * frame_count),
            ], np.float32),
        )
        std = np.append(
            std, np.asarray([
                0.3, 0.3, max(10.0, 0.15 * frame_count),
                max(12.0, 0.15 * frame_count),
            ], np.float32),
        )
    if dynamic_obstacles:
        mean = np.append(mean, np.float32(math.log(40.0)))
        std = np.append(std, np.float32(0.8))
    return mean, std


DEFAULT_THICKNESS_M = 0.006


class BatchWarpClothEngine:
    def __init__(
        self, scene: ClothSceneData, *, capacity: int, device: str = "cuda:0",
        substeps: int = 3, constraint_iterations: int = 4,
        thickness_m: float = DEFAULT_THICKNESS_M, support_mode: str = "none",
        boundary_mode: str = "none", obstacle_mode: str = DEFAULT_OBSTACLE_MODE,
        edge_stop_frame: int | None = None,
        edge_release_frame: int | None = None,
        block_dim: int = DEFAULT_CUDA_BLOCK_DIM,
    ) -> None:
        if capacity < 1:
            raise ValueError("cloth capacity must be positive")
        if block_dim < 32 or block_dim > 1024 or block_dim % 32:
            raise ValueError("cloth CUDA block dimension must be a multiple of 32 in [32, 1024]")
        self.scene = scene
        self.capacity = capacity
        self.block_dim = block_dim
        self.device = device
        self.torch_device = torch.device(device)
        self.substeps = substeps
        self.constraint_iterations = constraint_iterations
        self.thickness_m = thickness_m
        if support_mode not in {
            "initial-above", "initial-envelope", "none", "infinite"
        }:
            raise ValueError(f"unsupported cloth support mode: {support_mode}")
        self.support_mode = support_mode
        if boundary_mode not in {
            "none", "tracked-top", "tracked-edges", "tracked-edges-release",
            "tracked-corners", "edge-drive",
        }:
            raise ValueError(f"unsupported cloth boundary mode: {boundary_mode}")
        if obstacle_mode not in {"none", "obb", "obb-dynamic"}:
            raise ValueError(f"unsupported cloth obstacle mode: {obstacle_mode}")
        if (edge_stop_frame is not None or edge_release_frame is not None) and boundary_mode != "edge-drive":
            raise ValueError("edge drive timing requires edge-drive mode")
        if (edge_stop_frame is not None and edge_release_frame is not None
                and edge_release_frame < edge_stop_frame):
            raise ValueError("edge release must not precede edge stop")
        self.boundary_mode = boundary_mode
        self.edge_stop_frame = edge_stop_frame
        self.edge_release_frame = edge_release_frame
        self.dynamic_obstacles = obstacle_mode == "obb-dynamic"
        self.rigid_mass_parameter = 11 if boundary_mode == "edge-drive" else 7
        self.parameter_count = self.rigid_mass_parameter + int(self.dynamic_obstacles)
        self.obstacle_mode = obstacle_mode
        obstacles = scene.obstacles if obstacle_mode != "none" else ()
        obstacle_centers = np.asarray([item.center for item in obstacles], np.float32)
        obstacle_axis_x = np.asarray([item.axis_x for item in obstacles], np.float32)
        obstacle_axis_y = np.asarray([item.axis_y for item in obstacles], np.float32)
        obstacle_axis_z = np.asarray([item.axis_z for item in obstacles], np.float32)
        obstacle_half = np.asarray([item.half_extents for item in obstacles], np.float32)
        obstacle_volumes = np.asarray([
            np.prod(item.half_extents) for item in obstacles
        ], np.float32)
        inverse_volume = (
            np.median(obstacle_volumes) / obstacle_volumes
            if len(obstacle_volumes) else np.zeros(0, np.float32)
        ).astype(np.float32)
        if not obstacles:
            obstacle_centers = obstacle_axis_x = obstacle_axis_y = obstacle_axis_z = obstacle_half = np.zeros((0, 3), np.float32)
        self.drive_inward = np.zeros(3, dtype=np.float32)
        self.drive_sideways = np.zeros(3, dtype=np.float32)
        if boundary_mode == "edge-drive":
            kinematic_mask, self.drive_inward, self.drive_sideways = (
                _edge_drive_basis(scene)
            )
            kinematic_positions = np.broadcast_to(
                scene.initial_positions, (scene.frame_count, scene.node_count, 3)
            ).copy()
        elif boundary_mode != "none":
            kinematic_mask, kinematic_positions = _tracked_boundary(scene, boundary_mode)
        else:
            kinematic_mask = np.zeros(scene.node_count, dtype=np.int32)
            kinematic_positions = np.broadcast_to(
                scene.initial_positions, (scene.frame_count, scene.node_count, 3)
            ).copy()
        self.interior_mask = kinematic_mask == 0
        kinematic_frame_mask = (
            _released_boundary_mask(
                scene, kinematic_mask
            )
            if boundary_mode == "tracked-edges-release"
            else np.broadcast_to(
                kinematic_mask, (scene.frame_count, scene.node_count)
            ).copy()
        )
        structural = scene.structural_edges
        bending = scene.bending_edges
        initial_height = (
            scene.initial_positions @ scene.support_normal_world
            - scene.support_offset_world
        )
        self.support_offset_world = scene.support_offset_world
        if support_mode == "initial-envelope":
            self.support_offset_world += float(np.min(initial_height)) - thickness_m
        if support_mode in {"infinite", "initial-envelope"}:
            support_mask = np.ones(scene.node_count, dtype=np.int32)
        elif support_mode == "initial-above":
            support_mask = (initial_height >= thickness_m).astype(np.int32)
        else:
            support_mask = np.zeros(scene.node_count, dtype=np.int32)
        wp.init()
        with wp.ScopedDevice(device):
            self.theta = wp.empty(
                (capacity, self.parameter_count), dtype=float, device=device
            )
            self.initial_positions = wp.array(scene.initial_positions, dtype=wp.vec3, device=device)
            self.support_mask = wp.array(support_mask, dtype=int, device=device)
            self.obstacle_centers = wp.array(obstacle_centers, dtype=wp.vec3, device=device)
            self.obstacle_axis_x = wp.array(obstacle_axis_x, dtype=wp.vec3, device=device)
            self.obstacle_axis_y = wp.array(obstacle_axis_y, dtype=wp.vec3, device=device)
            self.obstacle_axis_z = wp.array(obstacle_axis_z, dtype=wp.vec3, device=device)
            self.obstacle_half = wp.array(obstacle_half, dtype=wp.vec3, device=device)
            self.obstacle_inverse_volume = wp.array(inverse_volume, dtype=float, device=device)
            self.kinematic_mask = wp.array(
                kinematic_frame_mask, dtype=int, device=device
            )
            self.kinematic_positions = wp.array(
                kinematic_positions, dtype=wp.vec3, device=device
            )
            self.structural_first = wp.array(structural[:, 0], dtype=int, device=device)
            self.structural_second = wp.array(structural[:, 1], dtype=int, device=device)
            self.structural_rest = wp.array(scene.structural_rest_lengths, dtype=float, device=device)
            self.bending_first = wp.array(bending[:, 0], dtype=int, device=device)
            self.bending_second = wp.array(bending[:, 1], dtype=int, device=device)
            self.bending_rest = wp.array(scene.bending_rest_lengths, dtype=float, device=device)
            shape = (capacity, scene.node_count)
            self.positions = wp.empty(shape, dtype=wp.vec3, device=device)
            self.previous_positions = wp.empty(shape, dtype=wp.vec3, device=device)
            self.velocities = wp.empty(shape, dtype=wp.vec3, device=device)
            self.predicted = wp.empty(
                (capacity, scene.frame_count, scene.node_count),
                dtype=wp.vec3,
                device=device,
            )
            rigid_shape = (capacity, max(1, len(obstacles)))
            rigid_trajectory_shape = (
                capacity, scene.frame_count, max(1, len(obstacles))
            )
            self.rigid_centers = wp.empty(rigid_shape, dtype=wp.vec3, device=device)
            self.rigid_axis_x = wp.empty(rigid_shape, dtype=wp.vec3, device=device)
            self.rigid_axis_y = wp.empty(rigid_shape, dtype=wp.vec3, device=device)
            self.rigid_axis_z = wp.empty(rigid_shape, dtype=wp.vec3, device=device)
            self.rigid_velocities = wp.empty(rigid_shape, dtype=wp.vec3, device=device)
            self.rigid_angular_velocities = wp.empty(rigid_shape, dtype=wp.vec3, device=device)
            self.predicted_rigid_centers = wp.empty(
                rigid_trajectory_shape, dtype=wp.vec3, device=device
            )
            self.predicted_rigid_axis_x = wp.empty(
                rigid_trajectory_shape, dtype=wp.vec3, device=device
            )
            self.predicted_rigid_axis_y = wp.empty(
                rigid_trajectory_shape, dtype=wp.vec3, device=device
            )
            self.predicted_rigid_axis_z = wp.empty(
                rigid_trajectory_shape, dtype=wp.vec3, device=device
            )
        self.theta_torch = wp.to_torch(self.theta)
        self.predicted_torch = wp.to_torch(self.predicted)
        self.predicted_rigid_centers_torch = wp.to_torch(self.predicted_rigid_centers)
        self.predicted_rigid_axis_x_torch = wp.to_torch(self.predicted_rigid_axis_x)
        self.predicted_rigid_axis_y_torch = wp.to_torch(self.predicted_rigid_axis_y)
        self.predicted_rigid_axis_z_torch = wp.to_torch(self.predicted_rigid_axis_z)
        self.target = torch.as_tensor(
            scene.target_positions, dtype=torch.float32, device=self.torch_device
        )
        self.weights = torch.as_tensor(
            scene.target_weights, dtype=torch.float32, device=self.torch_device
        )

    def rollout(
        self, values: torch.Tensor | np.ndarray, end_offset: int | None = None,
    ) -> torch.Tensor:
        samples = torch.as_tensor(values, dtype=torch.float32, device=self.torch_device)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        count = len(samples)
        if count > self.capacity or samples.shape[1] != self.parameter_count:
            raise ValueError("invalid cloth parameter batch")
        final_offset = self.scene.frame_count - 1 if end_offset is None else min(
            max(int(end_offset), 0), self.scene.frame_count - 1
        )
        self.theta_torch[:count].copy_(samples)
        self._launch(count, final_offset)
        return self.predicted_torch[:count, : final_offset + 1]

    def _launch(self, count: int, final_offset: int) -> None:
        wp.launch(
            _simulate_cloth_worlds,
            dim=count,
            inputs=self._kernel_inputs(final_offset),
            device=self.device,
            block_dim=self.block_dim,
        )

    def _kernel_inputs(self, final_offset: int) -> list:
        return [
            self.theta,
            self.initial_positions,
            wp.vec3(*self.scene.gravity_down_world),
            wp.vec3(*self.scene.support_normal_world),
            self.support_offset_world,
            self.support_mask,
            self.obstacle_centers,
            self.obstacle_axis_x,
            self.obstacle_axis_y,
            self.obstacle_axis_z,
            self.obstacle_half,
            self.obstacle_inverse_volume,
            self.kinematic_mask,
            self.kinematic_positions,
            wp.vec3(*self.drive_inward),
            wp.vec3(*self.drive_sideways),
            1 if self.boundary_mode == "edge-drive" else 0,
            self.structural_first,
            self.structural_second,
            self.structural_rest,
            self.bending_first,
            self.bending_second,
            self.bending_rest,
            self.scene.node_count,
            len(self.scene.structural_edges),
            len(self.scene.bending_edges),
            len(self.scene.obstacles) if self.obstacle_mode != "none" else 0,
            int(self.dynamic_obstacles),
            self.rigid_mass_parameter,
            final_offset + 1,
            self.scene.fps,
            self.substeps,
            self.constraint_iterations,
            self.thickness_m,
            self.positions,
            self.previous_positions,
            self.velocities,
            self.predicted,
            self.rigid_centers,
            self.rigid_axis_x,
            self.rigid_axis_y,
            self.rigid_axis_z,
            self.rigid_velocities,
            self.rigid_angular_velocities,
            self.predicted_rigid_centers,
            self.predicted_rigid_axis_x,
            self.predicted_rigid_axis_y,
            self.predicted_rigid_axis_z,
        ]


def _candidate_loss(
    engine: BatchWarpClothEngine, predicted: torch.Tensor, *, end_offset: int,
    huber_delta_m: float,
) -> torch.Tensor:
    target = engine.target[: end_offset + 1]
    weights = engine.weights[: end_offset + 1]
    distance = torch.linalg.vector_norm(predicted - target.unsqueeze(0), dim=-1)
    delta = float(huber_delta_m)
    huber = torch.where(
        distance <= delta,
        0.5 * distance.square(),
        delta * (distance - 0.5 * delta),
    )
    return (huber * weights.unsqueeze(0)).sum(dim=(1, 2)) / torch.clamp(
        weights.sum(), min=1.0
    )


def _cem_iterations(
    engine: BatchWarpClothEngine, config: ClothCEMConfig, *, verbose: bool = True,
):
    """Yield evaluation batches while retaining this seed's independent state."""
    edge_drive = engine.boundary_mode == "edge-drive"
    frame_count = engine.scene.frame_count if edge_drive else None
    lower_np, upper_np = parameter_bounds(
        edge_drive=edge_drive, frame_count=frame_count,
        dynamic_obstacles=getattr(engine, "dynamic_obstacles", False),
    )
    mean_np, std_np = initial_distribution(
        edge_drive=edge_drive, frame_count=frame_count,
        dynamic_obstacles=getattr(engine, "dynamic_obstacles", False),
    )
    if edge_drive:
        for index, fixed_frame in (
            (9, engine.edge_stop_frame), (10, engine.edge_release_frame)
        ):
            if fixed_frame is not None:
                offset = np.clip(
                    fixed_frame - engine.scene.start_frame, 0, frame_count
                )
                lower_np[index] = upper_np[index] = mean_np[index] = offset
                std_np[index] = 0.0
        upper_np[9] = min(upper_np[9], upper_np[10])
        lower_np[10] = max(lower_np[10], lower_np[9])
        mean_np = np.clip(mean_np, lower_np, upper_np)
        mean_np[10] = max(mean_np[10], mean_np[9])
    lower = torch.as_tensor(lower_np, device=engine.torch_device)
    upper = torch.as_tensor(upper_np, device=engine.torch_device)
    mean = torch.as_tensor(mean_np, device=engine.torch_device)
    std = torch.as_tensor(std_np, device=engine.torch_device)
    timing_mean = mean[9:11].clone() if edge_drive else None
    timing_std = std[9:11].clone() if edge_drive else None
    minimum_std = 0.01 * (upper - lower)
    if edge_drive:
        minimum_std[9:11] = torch.where(
            upper[9:11] > lower[9:11],
            torch.maximum(minimum_std[9:11], torch.tensor(1.5, device=engine.torch_device)),
            0.0,
        )
    generator = torch.Generator(device=engine.torch_device).manual_seed(config.seed)
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    best = mean.clone()
    best_loss = math.inf
    trace = []
    prefixes = tuple(
        engine.scene.frame_count - 1 if value <= 0 else min(
            value - 1, engine.scene.frame_count - 1
        )
        for value in config.prefix_frames
    )
    for stage_index, end_offset in enumerate(prefixes):
        timing_active = edge_drive and (
            end_offset == engine.scene.frame_count - 1
            or stage_index == len(prefixes) - 1
        )
        generation_count = config.generations_per_prefix
        stage_best = best.clone()
        stage_loss = math.inf
        for generation in range(generation_count):
            samples = mean + std * torch.randn(
                (config.population, engine.parameter_count), generator=generator,
                device=engine.torch_device,
            )
            samples.clamp_(lower, upper)
            if getattr(engine, "dynamic_obstacles", False):
                mass_explorers = max(4, config.population // 4)
                samples[-mass_explorers:, engine.rigid_mass_parameter] = (
                    lower[engine.rigid_mass_parameter]
                    + torch.rand(
                        mass_explorers, generator=generator,
                        device=engine.torch_device,
                    ) * (
                        upper[engine.rigid_mass_parameter]
                        - lower[engine.rigid_mass_parameter]
                    )
                )
            if edge_drive:
                if timing_active:
                    samples[:, 10] = torch.maximum(samples[:, 10], samples[:, 9])
                else:
                    samples[:, 9:11] = timing_mean
            samples[0] = best
            losses = yield samples, end_offset
            elite_losses, elite_indices = torch.topk(
                losses, elite_count, largest=False
            )
            elites = samples[elite_indices]
            smoothing = config.distribution_smoothing
            mean = smoothing * mean + (1.0 - smoothing) * elites.mean(dim=0)
            std = smoothing * std + (1.0 - smoothing) * elites.std(
                dim=0, unbiased=False
            )
            std = torch.maximum(std, minimum_std)
            if config.exploration_fraction > 0.0:
                std = torch.maximum(
                    std, config.exploration_fraction * (upper - lower)
                )
            if edge_drive and not timing_active:
                mean[9:11] = timing_mean
                std[9:11] = timing_std
            if elite_losses[0] < stage_loss:
                stage_loss = float(elite_losses[0].item())
                stage_best = samples[elite_indices[0]].clone()
        best = stage_best
        mean = best.clone()
        if edge_drive and not timing_active:
            mean[9:11] = timing_mean
            std[9:11] = timing_std
        best_loss = stage_loss
        entry = {
            "stage": stage_index + 1,
            "end_frame": engine.scene.frames[end_offset],
            "best_loss": best_loss,
            "generations": generation_count,
            "timing_frozen": bool(edge_drive and not timing_active),
        }
        trace.append(entry)
        if verbose:
            print(json.dumps({"phase": "cloth_cem", **entry}), flush=True)
    return {"best": best.detach().cpu().numpy(), "best_loss": best_loss, "trace": trace}


def run_cem(
    engine: BatchWarpClothEngine, config: ClothCEMConfig, *, verbose: bool = True,
) -> dict[str, Any]:
    iterations = _cem_iterations(engine, config, verbose=verbose)
    try:
        request = next(iterations)
    except StopIteration as result:
        return result.value
    while True:
        samples, end_offset = request
        losses = _candidate_loss(
            engine, engine.rollout(samples, end_offset), end_offset=end_offset,
            huber_delta_m=config.huber_delta_m,
        )
        try:
            request = iterations.send(losses)
        except StopIteration as result:
            return result.value


def _rollout_release_profile(engine, samples):
    if getattr(engine, "reuse_release_prefix", False):
        return engine.rollout_release_candidates(samples)
    return engine.rollout(samples)


def refine_cloth_release(
    engine: BatchWarpClothEngine, initial: np.ndarray, *,
    huber_delta_m: float = 1.0,
) -> dict[str, Any]:
    """Profile release time without changing the drive, stop, or material fit."""
    if engine.boundary_mode != "edge-drive":
        raise ValueError("release refinement requires edge-drive")
    if engine.edge_release_frame is not None:
        raise ValueError("release refinement cannot change a fixed release frame")
    if not math.isfinite(huber_delta_m) or huber_delta_m <= 0.0:
        raise ValueError("Huber delta must be finite and positive")
    initial = np.asarray(initial, dtype=np.float32)
    if initial.shape != (engine.parameter_count,) or not np.isfinite(initial).all():
        raise ValueError("invalid initial cloth parameters")
    if not 0.0 <= initial[9] <= initial[10] <= engine.scene.frame_count:
        raise ValueError("invalid stop/release offsets")
    releases = np.concatenate((
        initial[10:11],
        # Release is tested at the start of a frame. Let the last driven
        # substep reach the stop position before considering a new release.
        np.arange(math.ceil(float(initial[9])) + 1, engine.scene.frame_count + 1),
    ))
    profile = []
    for offset in range(0, len(releases), engine.capacity):
        offsets = releases[offset:offset + engine.capacity]
        samples = np.tile(initial, (len(offsets), 1))
        samples[:, 10] = offsets
        predicted = _rollout_release_profile(engine, samples)
        whole_loss = _candidate_loss(
            engine, predicted, end_offset=engine.scene.frame_count - 1,
            huber_delta_m=huber_delta_m,
        )
        for release, whole in zip(offsets, whole_loss.detach().cpu().tolist()):
            profile.append({
                "release_offset_frames": float(release),
                "release_frame": engine.scene.start_frame + math.ceil(float(release)),
                "vertex_loss": whole, "selection_loss": whole,
            })
    # Keep the saved candidate on ties; absence of evidence must not force an
    # earlier release merely because it appears first in the integer grid.
    finite = [i for i, row in enumerate(profile) if math.isfinite(row["selection_loss"])]
    if not finite or 0 not in finite:
        raise ValueError("non-finite saved cloth rollout loss")
    selected = min(finite, key=lambda i: profile[i]["selection_loss"])
    best = initial.copy()
    best[10] = releases[selected]
    return {
        "best": best,
        "optimization_loss": profile[selected]["vertex_loss"],
        "metadata": {
            "method": "integer-release-profile-original-vertex-loss",
            "huber_delta_m": huber_delta_m,
            "stop_and_other_parameters_fixed": True,
            "baseline": profile[0], "selected": profile[selected], "profile": profile,
        },
    }


def refine_cloth_release_materials(
    engine: BatchWarpClothEngine, initial: np.ndarray, *,
    huber_delta_m: float = 1.0, passes: int = 2,
) -> dict[str, Any]:
    """Profile release jointly with each physical parameter, retaining the drive."""
    if passes < 1:
        raise ValueError("joint refinement requires at least one pass")
    release_fit = refine_cloth_release(engine, initial, huber_delta_m=huber_delta_m)
    best = release_fit["best"].copy()
    best_loss = release_fit["optimization_loss"]
    release_only_loss = best_loss
    dynamic = getattr(engine, "dynamic_obstacles", False)
    lower, upper = parameter_bounds(
        edge_drive=True, frame_count=engine.scene.frame_count,
        dynamic_obstacles=dynamic,
    )
    coordinates = [(3, "log_gravity"), (4, "structural_stiffness"),
                   (5, "bending_stiffness"), (6, "drag")]
    if dynamic:
        coordinates.append((engine.rigid_mass_parameter, "log_rigid_mass"))
    trace = []
    candidate_count = 0
    for pass_index in range(passes):
        for index, name in coordinates:
            previous = best.copy()
            previous_loss = best_loss
            # Halve the neighborhood each pass, but keep an absolute floor so
            # a zero stiffness or drag can still leave the bound.
            scale = (
                math.log(4.0) if name.startswith("log_") else
                max(abs(float(best[index])), 0.1 * float(upper[index] - lower[index]))
            ) / (2 ** pass_index)
            values = np.unique(np.clip(
                best[index] + scale * np.asarray([-0.5, -0.25, 0.0, 0.25, 0.5]),
                lower[index], upper[index],
            ))
            releases = np.unique(np.concatenate((
                best[10:11],
                np.arange(math.ceil(float(best[9])) + 1, engine.scene.frame_count + 1),
            )))
            # Candidate zero is the incumbent, so ties and unobserved motion
            # never cause arbitrary parameter changes.
            samples = np.tile(best, (1 + len(values) * len(releases), 1))
            samples[1:, index] = np.repeat(values, len(releases))
            samples[1:, 10] = np.tile(releases, len(values))
            for offset in range(0, len(samples), engine.capacity):
                batch = samples[offset:offset + engine.capacity]
                predicted = _rollout_release_profile(engine, batch)
                losses = _candidate_loss(
                    engine, predicted, end_offset=engine.scene.frame_count - 1,
                    huber_delta_m=huber_delta_m,
                ).detach().cpu().numpy()
                finite = np.flatnonzero(np.isfinite(losses))
                if not len(finite):
                    continue
                row = int(finite[np.argmin(losses[finite])])
                if float(losses[row]) < best_loss:
                    best = batch[row].copy()
                    best_loss = float(losses[row])
            candidate_count += len(samples)
            trace.append({
                "pass": pass_index + 1, "parameter": name,
                "before_value": float(previous[index]), "after_value": float(best[index]),
                "before_release_frame": engine.scene.start_frame + math.ceil(float(previous[10])),
                "after_release_frame": engine.scene.start_frame + math.ceil(float(best[10])),
                "before_loss": previous_loss, "after_loss": best_loss,
            })
    return {
        "best": best, "optimization_loss": best_loss,
        "metadata": {
            "method": "release-material-coordinate-profile-original-vertex-loss",
            "huber_delta_m": huber_delta_m, "passes": passes,
            "stop_and_drive_fixed": True,
            "fixed_parameter_indices": [0, 1, 2, 7, 8, 9],
            "optimized_parameter_indices": [index for index, _ in coordinates] + [10],
            "baseline": release_fit["metadata"]["baseline"],
            "release_only_loss": release_only_loss,
            "selected": {
                "release_offset_frames": float(best[10]),
                "release_frame": engine.scene.start_frame + math.ceil(float(best[10])),
                "vertex_loss": best_loss, "selection_loss": best_loss,
            },
            "candidate_count": candidate_count, "trace": trace,
        },
    }


def refine_cloth_dynamics(
    engine: BatchWarpClothEngine, initial: np.ndarray, *,
    huber_delta_m: float = 1.0, strata: int = 4, population: int = 64,
    generations: int = 24, seed: int = 7,
) -> dict[str, Any]:
    """Optional full-parameter CEM polish, retaining separate release hypotheses."""
    if engine.boundary_mode != "edge-drive":
        raise ValueError("joint dynamics refinement requires edge-drive")
    if engine.edge_release_frame is not None or engine.edge_stop_frame is not None:
        raise ValueError("joint dynamics refinement cannot change fixed drive timing")
    if strata < 1 or population < 4 or generations < 1:
        raise ValueError("invalid joint dynamics search budget")
    if not math.isfinite(huber_delta_m) or huber_delta_m <= 0:
        raise ValueError("Huber delta must be finite and positive")
    best = np.asarray(initial, np.float32).copy()
    if best.shape != (engine.parameter_count,) or not np.isfinite(best).all():
        raise ValueError("invalid initial cloth parameters")
    if not 0 <= best[9] <= best[10] <= engine.scene.frame_count:
        raise ValueError("invalid stop/release offsets")
    dynamic = engine.dynamic_obstacles
    lower, upper = parameter_bounds(
        edge_drive=True, frame_count=engine.scene.frame_count, dynamic_obstacles=dynamic,
    )
    _, initial_std = initial_distribution(
        edge_drive=True, frame_count=engine.scene.frame_count, dynamic_obstacles=dynamic,
    )

    def evaluate(samples: np.ndarray) -> np.ndarray:
        losses = []
        for offset in range(0, len(samples), engine.capacity):
            predicted = engine.rollout(samples[offset:offset + engine.capacity])
            losses.append(_candidate_loss(
                engine, predicted, end_offset=engine.scene.frame_count - 1,
                huber_delta_m=huber_delta_m,
            ).detach().cpu().numpy())
        return np.concatenate(losses)

    best_loss = float(evaluate(best[None])[0])
    if not math.isfinite(best_loss):
        raise ValueError("non-finite saved cloth rollout loss")
    baseline_loss = best_loss
    means = np.tile(best, (strata, 1))
    deviations = np.tile(initial_std * 0.5, (strata, 1))
    floors = (upper - lower) * 0.003
    floors[9:11] = 1.0
    boundaries = np.linspace(0, engine.scene.frame_count, strata + 1)
    incumbents = means.copy()
    incumbent_losses = np.full(strata, np.inf)
    for k in range(strata):
        means[k, 10] = 0.5 * (boundaries[k] + boundaries[k + 1])
        means[k, 9] = min(means[k, 9], means[k, 10])
        deviations[k, 10] = (boundaries[k + 1] - boundaries[k]) * 0.4
    rng = np.random.default_rng(seed)
    elite_count = max(2, population // 8)
    trace = []
    for generation in range(generations):
        samples = rng.normal(
            means[:, None, :], deviations[:, None, :],
            (strata, population, len(best)),
        ).astype(np.float32)
        samples = np.clip(samples, lower, upper)
        for k in range(strata):
            samples[k, :, 10] = np.clip(samples[k, :, 10], boundaries[k], boundaries[k + 1])
            samples[k, :, 9] = np.minimum(samples[k, :, 9], samples[k, :, 10])
            if np.isfinite(incumbent_losses[k]):
                samples[k, 0] = incumbents[k]
        losses = evaluate(samples.reshape(-1, len(best))).reshape(strata, population)
        for k in range(strata):
            finite = np.flatnonzero(np.isfinite(losses[k]))
            if not len(finite):
                continue
            ids = finite[np.argsort(losses[k, finite])[:elite_count]]
            elite = samples[k, ids]
            means[k] = 0.2 * means[k] + 0.8 * elite.mean(axis=0)
            deviations[k] = np.maximum(
                0.2 * deviations[k] + 0.8 * elite.std(axis=0), floors,
            )
            incumbents[k] = elite[0]
            incumbent_losses[k] = losses[k, ids[0]]
        k = int(np.argmin(incumbent_losses))
        # Retain the supplied fit on ties, including unobserved sequences.
        if incumbent_losses[k] < best_loss:
            best = incumbents[k].copy()
            best_loss = float(incumbent_losses[k])
        trace.append({
            "generation": generation, "loss": best_loss,
            "release_frame": engine.scene.start_frame + math.ceil(float(best[10])),
            "stratum_losses": [float(x) if np.isfinite(x) else None for x in incumbent_losses],
        })
    return {
        "best": best, "optimization_loss": best_loss,
        "metadata": {
            "method": "uniform-release-strata-joint-dynamics-CEM",
            "huber_delta_m": huber_delta_m, "seed": seed,
            "strata": strata, "population_per_stratum": population,
            "generations": generations,
            "candidate_count": 1 + strata * population * generations,
            "stop_and_drive_fixed": False,
            "optimized_parameter_indices": list(range(engine.parameter_count)),
            "baseline_loss": baseline_loss, "trace": trace,
        },
    }


def cloth_metrics(scene: ClothSceneData, predicted: np.ndarray) -> dict[str, float]:
    valid = scene.target_weights > 0.0
    error = np.linalg.norm(predicted - scene.target_positions, axis=-1)
    vertex_rmse = float(np.sqrt(np.mean(error[valid] ** 2)))
    centroid_errors = []
    shape_squared = []
    for frame in range(scene.frame_count):
        frame_valid = valid[frame]
        if not np.any(frame_valid):
            continue
        predicted_center = np.mean(predicted[frame, frame_valid], axis=0)
        target_center = np.mean(scene.target_positions[frame, frame_valid], axis=0)
        centroid_errors.append(np.linalg.norm(predicted_center - target_center) ** 2)
        predicted_relative = predicted[frame, frame_valid] - predicted_center
        target_relative = scene.target_positions[frame, frame_valid] - target_center
        shape_squared.extend(np.sum((predicted_relative - target_relative) ** 2, axis=-1))
    edge_squared = []
    for first, second in scene.structural_edges:
        edge_valid = valid[:, first] & valid[:, second]
        if not np.any(edge_valid):
            continue
        predicted_length = np.linalg.norm(
            predicted[edge_valid, second] - predicted[edge_valid, first], axis=-1
        )
        target_length = np.linalg.norm(
            scene.target_positions[edge_valid, second]
            - scene.target_positions[edge_valid, first],
            axis=-1,
        )
        edge_squared.extend((predicted_length - target_length) ** 2)
    return {
        "vertex_rmse_m": vertex_rmse,
        "centroid_rmse_m": float(np.sqrt(np.mean(centroid_errors))),
        "shape_rmse_m": float(np.sqrt(np.mean(shape_squared))),
        "edge_length_rmse_m": float(np.sqrt(np.mean(edge_squared))),
    }


def _evaluate_cloth_candidate(
    engine: BatchWarpClothEngine, theta: np.ndarray, *, device: str,
) -> dict[str, Any]:
    scene = engine.scene
    predicted_tensor = engine.rollout(theta)
    wp.synchronize_device(device)
    predicted = predicted_tensor[0].detach().cpu().numpy().astype(np.float64)
    valid = scene.target_weights > 0.0
    raw_error = predicted - scene.raw_target_positions
    result = {
        "predicted": predicted,
        "raw_vertex_rmse_m": float(np.sqrt(np.mean(
            np.sum(raw_error[valid] ** 2, axis=-1)
        ))),
        **cloth_metrics(scene, predicted),
    }
    if engine.dynamic_obstacles:
        rigid_count = len(scene.obstacles)
        result["rigid_trajectories"] = {
            "object_ids": [item.object_id for item in scene.obstacles],
            **{
                key: getattr(engine, attribute)[0, :, :rigid_count]
                .detach().cpu().numpy().tolist()
                for key, attribute in (
                    ("centers_world", "predicted_rigid_centers_torch"),
                    ("axis_x_world", "predicted_rigid_axis_x_torch"),
                    ("axis_y_world", "predicted_rigid_axis_y_torch"),
                    ("axis_z_world", "predicted_rigid_axis_z_torch"),
                )
            },
        }
    return result


def make_cloth_engine(scene: ClothSceneData, *, execution_mode: str = DEFAULT_EXECUTION_MODE, **options):
    if execution_mode not in CLOTH_EXECUTION_MODES:
        raise ValueError(f"unknown cloth execution mode: {execution_mode}")
    if execution_mode in {"colored", "colored-batched"}:
        from atw.polyworld.cloth_swr_parallel import InterleavedSplitClothEngine
        options.setdefault("graph_chunk_frames", DEFAULT_GRAPH_CHUNK_FRAMES)
        options.setdefault("contact_iteration_stride", DEFAULT_CONTACT_ITERATION_STRIDE)
        return InterleavedSplitClothEngine(scene, colored=True, **options)
    if execution_mode in {"interleaved", "interleaved-batched"}:
        from atw.polyworld.cloth_swr_execution import InterleavedClothEngine
        return InterleavedClothEngine(scene, **options)
    return BatchWarpClothEngine(scene, **options)


def fit_cloth_scene(
    scene: ClothSceneData, *, configs: tuple[ClothCEMConfig, ...] | None = None,
    device: str = "cuda:0", substeps: int = 3, constraint_iterations: int = 4,
    support_mode: str = "initial-above", boundary_mode: str = "edge-drive",
    obstacle_mode: str = DEFAULT_OBSTACLE_MODE,
    edge_stop_frame: int | None = None,
    edge_release_frame: int | None = None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
    release_prefix_reuse: bool = False,
    graph_chunk_frames: int | None = None,
    contact_iteration_stride: int | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], ClothCEMConfig, float]:
    if execution_mode not in CLOTH_EXECUTION_MODES:
        raise ValueError(f"unknown cloth execution mode: {execution_mode}")
    requested_execution_mode = execution_mode
    if contact_iteration_stride is None:
        contact_iteration_stride = DEFAULT_CONTACT_ITERATION_STRIDE if execution_mode in {"colored", "colored-batched"} else 1
    if graph_chunk_frames is None:
        graph_chunk_frames = DEFAULT_GRAPH_CHUNK_FRAMES if execution_mode in {"colored", "colored-batched"} else 0
    if graph_chunk_frames and execution_mode not in {"colored", "colored-batched"}:
        raise ValueError("frame blocks require colored cloth execution")
    if contact_iteration_stride != 1 and execution_mode not in {"colored", "colored-batched"}:
        raise ValueError("contact cadence requires colored cloth execution")
    batch_seeds = execution_mode in {"batched-seeds", "interleaved-batched", "colored-batched"}
    if boundary_mode not in {"none", "edge-drive"}:
        raise ValueError("open-loop fitting requires no boundary or edge-drive")
    configs = configs or tuple(ClothCEMConfig(seed=seed) for seed in DEFAULT_SEEDS)
    if batch_seeds and (
        len({(tuple(c.prefix_frames), c.generations_per_prefix) for c in configs}) != 1
        or any(c.generations_per_prefix <= 0 or not c.prefix_frames for c in configs)
    ):
        # Preserve custom seed schedules that cannot share evaluation batches.
        batch_seeds = False
        execution_mode = {
            "interleaved-batched": "interleaved", "colored-batched": "colored",
            "batched-seeds": "baseline",
        }[execution_mode]
    if any(config.release_refinement_passes < 0 for config in configs):
        raise ValueError("release refinement passes must be nonnegative")
    if any(config.joint_dynamics_population < 4 or config.joint_dynamics_generations < 1
           for config in configs):
        raise ValueError("invalid joint dynamics search budget")
    start = time.perf_counter()
    capacity = max(config.population for config in configs)
    if batch_seeds:
        capacity = sum(config.population for config in configs)
    needs_refinement = (
        boundary_mode == "edge-drive" and edge_release_frame is None
        and any(config.release_refinement_passes > 0 or (
            config.joint_dynamics_refinement and edge_stop_frame is None
        ) for config in configs)
    )
    if needs_refinement:
        # Capacity controls only independent-world batching, not CEM population.
        joint_capacity = max((
            4 * config.joint_dynamics_population
            for config in configs
            if config.joint_dynamics_refinement and edge_stop_frame is None
        ), default=0)
        capacity = max(capacity, DEFAULT_REFINEMENT_CAPACITY, joint_capacity)
    engine = make_cloth_engine(
        scene, execution_mode=execution_mode, capacity=capacity, device=device,
        substeps=substeps, constraint_iterations=constraint_iterations,
        support_mode=support_mode, boundary_mode=boundary_mode,
        obstacle_mode=obstacle_mode,
        edge_stop_frame=edge_stop_frame,
        edge_release_frame=edge_release_frame,
        **({'graph_chunk_frames': graph_chunk_frames, 'contact_iteration_stride': contact_iteration_stride}
           if execution_mode in {"colored", "colored-batched"} else {}),
    )
    if release_prefix_reuse:
        if not hasattr(engine, "rollout_release_candidates"):
            raise ValueError("release prefix reuse requires a split cloth execution mode")
        engine.reuse_release_prefix = True
    selected = None
    selected_config = None
    portfolio = []
    portfolio_start = time.perf_counter()
    shared_cem_seconds = 0.0
    if batch_seeds:
        from atw.polyworld.cloth_swr_execution import run_cem_portfolio
        cems = run_cem_portfolio(engine, configs, verbose=verbose)
        shared_cem_seconds = time.perf_counter() - portfolio_start
    for index, config in enumerate(configs):
        candidate_start = time.perf_counter()
        cem = cems[index] if batch_seeds else run_cem(engine, config, verbose=verbose)
        evaluated = _evaluate_cloth_candidate(engine, cem["best"], device=device)
        row = {
            "seed": config.seed,
            **{key: value for key, value in evaluated.items() if key.endswith("rmse_m")},
            "optimization_loss": cem["best_loss"],
            "elapsed_seconds": time.perf_counter() - candidate_start,
        }
        if batch_seeds:
            row["elapsed_seconds"] += shared_cem_seconds / len(configs)
            row["timing_mode"] = "amortized_shared_cem_plus_evaluation"
        portfolio.append(row)
        if selected is None or cem["best_loss"] < selected["optimization_loss"]:
            selected = {
                "best": cem["best"],
                "cem_trace": cem["trace"],
                "optimization_loss": cem["best_loss"],
                **evaluated,
            }
            selected_config = config
    portfolio_seconds = time.perf_counter() - portfolio_start
    assert selected is not None and selected_config is not None
    if (boundary_mode == "edge-drive" and edge_release_frame is None
            and selected_config.release_refinement_passes > 0):
        refinement_start = time.perf_counter()
        before_metrics = {
            key: value for key, value in selected.items()
            if key.endswith("rmse_m") or key == "optimization_loss"
        }
        refined = refine_cloth_release_materials(
            engine, selected["best"], huber_delta_m=selected_config.huber_delta_m,
            passes=selected_config.release_refinement_passes,
        )
        selected.update(
            best=refined["best"], optimization_loss=refined["optimization_loss"],
            **_evaluate_cloth_candidate(engine, refined["best"], device=device),
        )
        selected["release_refinement"] = {
            **refined["metadata"], "before_metrics": before_metrics,
            "elapsed_seconds": time.perf_counter() - refinement_start,
            "engine_capacity": engine.capacity,
        }
        if verbose:
            print(f"release/material refinement: {before_metrics['optimization_loss']:.6g}"
                  f" -> {selected['optimization_loss']:.6g}", flush=True)
    initial_fit_seconds = time.perf_counter() - start
    joint_seconds = 0.0
    if (selected_config.joint_dynamics_refinement and boundary_mode == "edge-drive"
            and edge_release_frame is None and edge_stop_frame is None):
        refinement_start = time.perf_counter()
        before_metrics = {
            key: value for key, value in selected.items()
            if key.endswith("rmse_m") or key == "optimization_loss"
        }
        refined = refine_cloth_dynamics(
            engine, selected["best"], huber_delta_m=selected_config.huber_delta_m,
            population=selected_config.joint_dynamics_population,
            generations=selected_config.joint_dynamics_generations,
        )
        selected.update(
            best=refined["best"], optimization_loss=refined["optimization_loss"],
            **_evaluate_cloth_candidate(engine, refined["best"], device=device),
        )
        joint_seconds = time.perf_counter() - refinement_start
        selected["joint_dynamics_refinement"] = {
            **refined["metadata"], "before_metrics": before_metrics,
            "elapsed_seconds": joint_seconds, "engine_capacity": engine.capacity,
        }
        if verbose:
            print(f"joint dynamics refinement: {before_metrics['optimization_loss']:.6g}"
                  f" -> {selected['optimization_loss']:.6g} ({joint_seconds:.1f}s)", flush=True)
    if boundary_mode == "edge-drive":
        velocity = (
            selected["best"][7] * engine.drive_inward
            + selected["best"][8] * engine.drive_sideways
        )
        speed = float(np.linalg.norm(velocity))
        selected["drive_velocity_world"] = velocity.tolist()
        selected["drive_direction_world"] = (
            (velocity / speed).tolist() if speed > 1.0e-8 else None
        )
        stop_frame = scene.start_frame + float(selected["best"][9])
        release_frame = scene.start_frame + math.ceil(float(selected["best"][10]))
        selected["drive_stop_frame"] = (
            stop_frame if stop_frame <= scene.frames[-1] else None
        )
        selected["drive_release_frame"] = (
            release_frame if release_frame <= scene.frames[-1] else None
        )
    selected["portfolio"] = portfolio
    selected["execution_config"] = {
        "engine_capacity": engine.capacity,
        "cuda_block_dim": engine.block_dim,
        "release_prefix_reuse": release_prefix_reuse,
        "graph_chunk_frames": graph_chunk_frames,
        "contact_iteration_stride": contact_iteration_stride,
    }
    if execution_mode != "baseline":
        selected["execution_config"].update(
            mode=execution_mode, shared_cem_seconds=shared_cem_seconds,
        )
    if requested_execution_mode != execution_mode:
        selected["execution_config"].update(
            mode=execution_mode, requested_mode=requested_execution_mode,
            seed_batching_fallback="incompatible_or_empty_seed_schedules",
        )
    if execution_mode in {"colored", "colored-batched"}:
        selected["execution_config"].update(
            forward_solver="colored", constraint_order="edge-colors",
            cuda_graph_count=len(engine.graphs),
        )
    elapsed_seconds = time.perf_counter() - start
    selected["stage_timings_seconds"] = {
        "initial_fit": initial_fit_seconds,
        "initial_cem_portfolio": portfolio_seconds,
        "release_material_refinement": selected.get("release_refinement", {}).get("elapsed_seconds", 0.0),
        "joint_dynamics_refinement": joint_seconds,
        "total_fit": elapsed_seconds,
    }
    return selected, selected_config, elapsed_seconds


def result_payload(
    source: dict[str, Any], scene: ClothSceneData, config: ClothCEMConfig,
    fit: dict[str, Any], elapsed_seconds: float, *, device: str, substeps: int,
    constraint_iterations: int, support_mode: str = "initial-above",
    boundary_mode: str = "edge-drive", obstacle_mode: str = DEFAULT_OBSTACLE_MODE,
    edge_stop_frame: int | None = None,
    edge_release_frame: int | None = None,
) -> dict[str, Any]:
    theta = fit["best"]
    predicted = fit["predicted"]
    fitted_parameter_count = len(theta) - (
        int(edge_stop_frame is not None) + int(edge_release_frame is not None)
        if boundary_mode == "edge-drive" else 0
    )
    return {
        "model": (
            "physmind-v2-warp-reduced-cloth-swr-v14"
            if obstacle_mode == "obb-dynamic" else
            "physmind-v2-warp-reduced-cloth-swr-v10"
        ),
        "source": source,
        "object_id": scene.object_id,
        "device": device,
        "fps": scene.fps,
        "reference_frame": scene.reference_frame,
        "start_frame": scene.start_frame,
        "frames": list(scene.frames),
        "node_count": scene.node_count,
        "vertex_ids": scene.vertex_ids.tolist(),
        "vertex_uv": scene.vertex_uv.tolist(),
        "faces": scene.faces.tolist(),
        "parameters": {
            "initial_velocity_world_m_per_s": theta[:3].tolist(),
            "gravity_scale": float(np.exp(theta[3])),
            "structural_stiffness": float(theta[4]),
            "bending_stiffness": float(theta[5]),
            "linear_drag_per_s": float(theta[6]),
            "edge_velocity_components_m_per_s": (
                theta[7:9].tolist() if boundary_mode == "edge-drive" else None
            ),
            "edge_stop_offset_frames": (
                float(theta[9]) if boundary_mode == "edge-drive" else None
            ),
            "edge_release_offset_frames": (
                float(theta[10]) if boundary_mode == "edge-drive" else None
            ),
            **({"rigid_mass_per_median_box_in_cloth_node_masses": float(
                np.exp(theta[11 if boundary_mode == "edge-drive" else 7])
            )} if obstacle_mode == "obb-dynamic" else {}),
            "edge_velocity_world_m_per_s": (
                fit["drive_velocity_world"] if boundary_mode == "edge-drive" else None
            ),
            "edge_speed_world_m_per_s": (
                float(np.linalg.norm(fit["drive_velocity_world"]))
                if boundary_mode == "edge-drive" else None
            ),
        },
        "predicted_positions_world": predicted.tolist(),
        **({
            "predicted_rigid_trajectories": fit["rigid_trajectories"],
            "obstacle_half_extents_m": {
                item.object_id: list(item.half_extents) for item in scene.obstacles
            },
        } if obstacle_mode == "obb-dynamic" else {}),
        "cem_config": asdict(config),
        "selected_seed": config.seed,
        "elapsed_seconds": elapsed_seconds,
        "optimization_loss": fit["optimization_loss"],
        "portfolio": fit["portfolio"],
        "raw_vertex_rmse_m": fit["raw_vertex_rmse_m"],
        "cem_trace": fit["cem_trace"],
        **({
            "release_refinement": fit["release_refinement"],
            "cem_trace_scope": "before_release_material_refinement",
            "portfolio_scope": "before_release_material_refinement",
        } if "release_refinement" in fit else {}),
        **({
            "joint_dynamics_refinement": fit["joint_dynamics_refinement"],
            "cem_trace_scope": "before_post_cem_refinement",
            "portfolio_scope": "before_post_cem_refinement",
        } if "joint_dynamics_refinement" in fit else {}),
        **({"stage_timings_seconds": fit["stage_timings_seconds"]}
           if "stage_timings_seconds" in fit else {}),
        **({"execution_config": fit["execution_config"]}
           if "execution_config" in fit else {}),
        "vertex_rmse_m": fit["vertex_rmse_m"],
        "centroid_rmse_m": fit["centroid_rmse_m"],
        "shape_rmse_m": fit["shape_rmse_m"],
        "edge_length_rmse_m": fit["edge_length_rmse_m"],
        "physics_assumptions": {
            "solver": "batched 3D position-based cloth dynamics in NVIDIA Warp",
            "geometry": "coarse open triangle mesh sampled from fixed tracked vertex IDs",
            "constraints": (
                "structural, shear-like triangle edges, opposite-vertex bending, "
                + ("two-way cloth-rigid contact, rigid-rigid OBB contact, and floor support" if obstacle_mode == "obb-dynamic" else "static OBB contact")
            ),
            "inverse_method": (
                f"PyTorch cross-entropy method over "
                f"{fitted_parameter_count} global "
                "parameters (open-loop rollout)"
                + ("; joint release/material profile refinement"
                   if "release_refinement" in fit else "")
                + ("; uniform-release-strata joint dynamics CEM refinement"
                   if "joint_dynamics_refinement" in fit else "")
            ),
            "omissions": (
                "cloth self-collision and cloth-cloth collision; cloths are fitted separately"
                if obstacle_mode == "obb-dynamic" else
                "cloth self-collision and cloth-cloth collision"
            ),
            "support_plane_mode": support_mode,
            "boundary_mode": boundary_mode,
            "edge_drive": (
                {
                    "driven_edge": "minimum reference-image v",
                    "direction_world": fit["drive_direction_world"],
                    "velocity_world_m_per_s": fit["drive_velocity_world"],
                    "velocity_estimator": "joint CEM fit to tracked point trajectories",
                    "motion": "constant-velocity kinematic edge while driven",
                    "stop_frame": fit["drive_stop_frame"],
                    "release_frame": fit["drive_release_frame"],
                    "stop_frame_source": (
                        "fixed" if edge_stop_frame is not None else "fitted"
                    ),
                    "release_frame_source": (
                        "fixed" if edge_release_frame is not None else "fitted"
                    ),
                } if boundary_mode == "edge-drive" else None
            ),
            "obstacle_mode": obstacle_mode,
            **({
                "cloth_rigid_friction_coefficient": CLOTH_RIGID_FRICTION,
                "cloth_rigid_friction_velocity": "current-substep-displacement-per-iteration",
                "rigid_rigid_friction_coefficient": RIGID_RIGID_FRICTION,
                "ground_contact": (
                    "eight-corner support manifold with normal impulses, "
                    "static friction, and dynamic friction"
                ),
                "ground_static_friction_coefficient": GROUND_STATIC_FRICTION,
                "ground_dynamic_friction_coefficient": GROUND_DYNAMIC_FRICTION,
            } if obstacle_mode == "obb-dynamic" else {}),
            "fit_mode": "open-loop",
            "observation_policy": "loss_only;_no_state_correction",
            "substeps_per_frame": substeps,
            "constraint_iterations": constraint_iterations,
        },
    }
