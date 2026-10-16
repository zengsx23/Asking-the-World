"""Visible-surface cloth tracking with an open reference mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from atw.perception.mesh_projection import render_mesh_depth
from atw.perception.pointcloud_projection import apply_pose


@dataclass(frozen=True, slots=True)
class ClothReferenceMesh:
    """Reference cloth surface whose vertex IDs are canonical track IDs."""

    image_shape: tuple[int, int]
    vertex_uv: np.ndarray
    vertices_world: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    rest_edge_lengths: np.ndarray


@dataclass(frozen=True, slots=True)
class ClothMeshObservation:
    rendered_mask: np.ndarray
    rendered_depth: np.ndarray
    mask_iou: float
    observed_mask_coverage: float
    rendered_mask_precision: float
    compared_depth_pixels: int
    depth_absolute_error_mean: float | None
    depth_absolute_error_median: float | None
    depth_absolute_error_p90: float | None

    def metrics(self) -> dict[str, Any]:
        return {
            "mask_iou": self.mask_iou,
            "observed_mask_coverage": self.observed_mask_coverage,
            "rendered_mask_precision": self.rendered_mask_precision,
            "compared_depth_pixels": self.compared_depth_pixels,
            "depth_absolute_error_mean": self.depth_absolute_error_mean,
            "depth_absolute_error_median": self.depth_absolute_error_median,
            "depth_absolute_error_p90": self.depth_absolute_error_p90,
        }


@dataclass(frozen=True, slots=True)
class ClothFrameSolution:
    """One visible-only step; inactive IDs emit no position in this frame."""

    uv: np.ndarray
    positions_world: np.ndarray
    observations_world: np.ndarray
    active: np.ndarray
    unobserved: np.ndarray
    edge_relative_strain: np.ndarray


@dataclass(frozen=True, slots=True)
class ConstrainedClothTrajectory:
    uv: np.ndarray
    positions_world: np.ndarray
    tracking_confidence: np.ndarray
    active: np.ndarray
    trackable: np.ndarray
    frame_metrics: tuple[dict[str, Any], ...]


def _edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.empty(0, dtype=np.float32)
    delta = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def _unique_face_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0).astype(np.int32)


def build_cloth_reference_mesh(
    *,
    mask: np.ndarray,
    vertex_uv: np.ndarray,
    vertices_world: np.ndarray,
    maximum_edge_length_ratio: float = 4.0,
) -> ClothReferenceMesh:
    """Triangulate valid mask pixels without closing the cloth boundary.

    Triangles only join vertices sharing one reference-image cell. The shorter
    3D diagonal is chosen for complete cells, and unusually long faces are
    rejected so separate depth layers are not stitched together.
    """

    cloth_mask = np.asarray(mask, dtype=bool)
    uv = np.asarray(vertex_uv, dtype=np.float32)
    world = np.asarray(vertices_world, dtype=np.float32)
    if cloth_mask.ndim != 2:
        raise ValueError("cloth mask must be two-dimensional")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("vertex UV must have shape (vertices, 2)")
    if world.shape != (len(uv), 3):
        raise ValueError("world vertices must have shape (vertices, 3)")

    pixels = np.rint(uv).astype(np.int32)
    if not np.allclose(uv, pixels, atol=1e-4):
        raise ValueError("reference vertices must lie on image pixels")
    height, width = cloth_mask.shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    if not inside.all() or not cloth_mask[pixels[:, 1], pixels[:, 0]].all():
        raise ValueError("reference vertices must belong to the cloth mask")
    flat_pixels = pixels[:, 1].astype(np.int64) * width + pixels[:, 0]
    if len(np.unique(flat_pixels)) != len(flat_pixels):
        raise ValueError("reference vertices must have unique image pixels")
    if not np.isfinite(world).all():
        raise ValueError("reference world vertices must be finite")

    vertex_grid = np.full((height, width), -1, dtype=np.int32)
    vertex_grid[pixels[:, 1], pixels[:, 0]] = np.arange(len(uv), dtype=np.int32)

    neighbor_chunks: list[np.ndarray] = []
    for first, second in (
        (vertex_grid[:, :-1], vertex_grid[:, 1:]),
        (vertex_grid[:-1], vertex_grid[1:]),
    ):
        valid = (first >= 0) & (second >= 0)
        if valid.any():
            neighbor_chunks.append(np.column_stack((first[valid], second[valid])))
    neighbor_edges = (
        np.concatenate(neighbor_chunks).astype(np.int32)
        if neighbor_chunks
        else np.empty((0, 2), dtype=np.int32)
    )
    neighbor_lengths = _edge_lengths(world, neighbor_edges)
    positive_lengths = neighbor_lengths[
        np.isfinite(neighbor_lengths) & (neighbor_lengths > 0)
    ]
    typical_edge_length = (
        float(np.median(positive_lengths)) if len(positive_lengths) else 0.0
    )
    maximum_edge_length = typical_edge_length * maximum_edge_length_ratio

    corners = np.stack(
        (
            vertex_grid[:-1, :-1],
            vertex_grid[:-1, 1:],
            vertex_grid[1:, :-1],
            vertex_grid[1:, 1:],
        ),
        axis=-1,
    ).reshape(-1, 4)
    faces: list[tuple[int, int, int]] = []
    for top_left, top_right, bottom_left, bottom_right in corners:
        cell = (top_left, top_right, bottom_left, bottom_right)
        present = [int(vertex) for vertex in cell if vertex >= 0]
        if len(present) < 3:
            continue
        if len(present) == 3:
            faces.append(tuple(present))
            continue
        left_diagonal = np.linalg.norm(world[top_left] - world[bottom_right])
        right_diagonal = np.linalg.norm(world[top_right] - world[bottom_left])
        if left_diagonal <= right_diagonal:
            faces.extend(
                (
                    (int(top_left), int(top_right), int(bottom_right)),
                    (int(top_left), int(bottom_right), int(bottom_left)),
                )
            )
        else:
            faces.extend(
                (
                    (int(top_left), int(top_right), int(bottom_left)),
                    (int(top_right), int(bottom_right), int(bottom_left)),
                )
            )

    face_array = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if len(face_array) and typical_edge_length > 0:
        triangle_edges = np.stack(
            (
                face_array[:, [0, 1]],
                face_array[:, [1, 2]],
                face_array[:, [2, 0]],
            ),
            axis=1,
        )
        triangle_lengths = np.linalg.norm(
            world[triangle_edges[..., 1]] - world[triangle_edges[..., 0]],
            axis=-1,
        )
        cross = np.cross(
            world[face_array[:, 1]] - world[face_array[:, 0]],
            world[face_array[:, 2]] - world[face_array[:, 0]],
        )
        twice_area = np.linalg.norm(cross, axis=1)
        keep = (
            np.isfinite(triangle_lengths).all(axis=1)
            & (triangle_lengths <= maximum_edge_length).all(axis=1)
            & np.isfinite(twice_area)
            & (twice_area > typical_edge_length**2 * 1e-5)
        )
        face_array = face_array[keep]

    edges = _unique_face_edges(face_array)
    return ClothReferenceMesh(
        image_shape=(height, width),
        vertex_uv=uv.copy(),
        vertices_world=world.copy(),
        faces=face_array,
        edges=edges,
        rest_edge_lengths=_edge_lengths(world, edges),
    )


def evaluate_cloth_mesh_observation(
    mesh: ClothReferenceMesh,
    *,
    vertices_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    active_vertices: np.ndarray | None = None,
) -> ClothMeshObservation:
    """Compare a visible mesh surface with the segmented observed depth."""

    vertices = np.asarray(vertices_world, dtype=np.float32)
    if vertices.shape != mesh.vertices_world.shape:
        raise ValueError("mesh vertex positions have an unexpected shape")
    faces = mesh.faces
    if active_vertices is not None:
        active = np.asarray(active_vertices, dtype=bool)
        if active.shape != (len(vertices),):
            raise ValueError("active vertex mask has an unexpected shape")
        faces = faces[active[faces].all(axis=1)]

    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float32))
    vertices_camera = apply_pose(vertices, world_to_camera)
    rendered_mask, rendered_depth = render_mesh_depth(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=mesh.image_shape,
    )
    target_mask = np.asarray(observed_mask, dtype=bool)
    target_depth = np.asarray(observed_depth, dtype=np.float32)
    if target_mask.shape != mesh.image_shape or target_depth.shape != mesh.image_shape:
        raise ValueError("observed mask and depth must match the reference image")

    valid_target = target_mask & np.isfinite(target_depth) & (target_depth > 0)
    intersection = rendered_mask & valid_target
    union = rendered_mask | valid_target
    errors = np.abs(rendered_depth[intersection] - target_depth[intersection])
    return ClothMeshObservation(
        rendered_mask=rendered_mask,
        rendered_depth=rendered_depth,
        mask_iou=float(intersection.sum() / union.sum()) if union.any() else 1.0,
        observed_mask_coverage=(
            float(intersection.sum() / valid_target.sum())
            if valid_target.any()
            else 1.0
        ),
        rendered_mask_precision=(
            float(intersection.sum() / rendered_mask.sum())
            if rendered_mask.any()
            else float(not valid_target.any())
        ),
        compared_depth_pixels=int(len(errors)),
        depth_absolute_error_mean=(float(errors.mean()) if len(errors) else None),
        depth_absolute_error_median=(
            float(np.median(errors)) if len(errors) else None
        ),
        depth_absolute_error_p90=(
            float(np.quantile(errors, 0.9)) if len(errors) else None
        ),
    )


def _project_world_points(
    points_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float32))
    camera = apply_pose(points_world, world_to_camera)
    projected = camera @ np.asarray(intrinsic, dtype=np.float32).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    uv[camera[:, 2] <= 0] = np.nan
    return uv.astype(np.float32)


def _nearest_mask_pixels(
    uv: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest observed cloth pixel within a small boundary radius."""

    target = np.asarray(mask, dtype=bool)
    points = np.asarray(uv, dtype=np.float32)
    height, width = target.shape
    best = np.full((len(points), 2), -1, dtype=np.int32)
    if not target.any():
        return best, np.zeros(len(points), dtype=bool)

    _, nearest_indices = distance_transform_edt(
        ~target,
        return_indices=True,
    )
    finite = np.isfinite(points).all(axis=1)
    rounded = np.rint(np.where(finite[:, None], points, 0)).astype(np.int32)
    lookup_x = np.clip(rounded[:, 0], 0, width - 1)
    lookup_y = np.clip(rounded[:, 1], 0, height - 1)
    best[:, 0] = nearest_indices[1, lookup_y, lookup_x]
    best[:, 1] = nearest_indices[0, lookup_y, lookup_x]
    distance = np.linalg.norm(best.astype(np.float32) - points, axis=1)
    valid = finite & np.isfinite(distance) & (distance <= maximum_distance)
    best[~valid] = -1
    return best, valid


def render_active_cloth_mask(
    mesh: ClothReferenceMesh,
    *,
    vertex_uv: np.ndarray,
    active_vertices: np.ndarray,
) -> np.ndarray:
    """Rasterize the union of currently visible reference-mesh faces."""

    import cv2

    uv = np.asarray(vertex_uv, dtype=np.float32)
    active = np.asarray(active_vertices, dtype=bool)
    faces = mesh.faces[active[mesh.faces].all(axis=1)]
    output = np.zeros(mesh.image_shape, dtype=np.uint8)
    if len(faces):
        triangles = np.rint(uv[faces]).astype(np.int32)
        cv2.fillPoly(output, triangles, 1)
    return output.astype(bool)


def solve_visible_cloth_frame(
    mesh: ClothReferenceMesh,
    *,
    proposed_uv: np.ndarray,
    tracking_confidence: np.ndarray,
    previously_active: np.ndarray,
    observed_mask: np.ndarray,
    observed_world_points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    minimum_tracking_confidence: float = 0.05,
    iterations: int = 24,
    stretch_stiffness: float = 2.5,
    observation_stiffness: float = 0.0,
    maximum_relative_edge_strain: float = 2.0,
    maximum_mask_snap_distance: float = 5.0,
) -> ClothFrameSolution:
    """Fit visible observations while approximately preserving rest edge lengths."""

    uv = np.asarray(proposed_uv, dtype=np.float32)
    confidence = np.asarray(tracking_confidence, dtype=np.float32)
    previous = np.asarray(previously_active, dtype=bool)
    target_mask = np.asarray(observed_mask, dtype=bool)
    point_map = np.asarray(observed_world_points, dtype=np.float32)
    vertex_count = len(mesh.vertex_uv)
    if uv.shape != (vertex_count, 2):
        raise ValueError("proposed UV has an unexpected shape")
    if confidence.shape != (vertex_count,) or previous.shape != (vertex_count,):
        raise ValueError("confidence and active state must match mesh vertices")
    if (
        target_mask.shape != mesh.image_shape
        or point_map.shape != (*mesh.image_shape, 3)
    ):
        raise ValueError("observations must match the mesh image shape")

    finite_uv = np.isfinite(uv).all(axis=1)
    sample_pixels, on_mask = _nearest_mask_pixels(
        uv,
        target_mask,
        maximum_distance=maximum_mask_snap_distance,
    )
    observations = np.full((vertex_count, 3), np.nan, dtype=np.float32)
    observed_ids = np.flatnonzero(on_mask)
    if len(observed_ids):
        observations[observed_ids] = point_map[
            sample_pixels[observed_ids, 1], sample_pixels[observed_ids, 0]
        ]
    active = (
        previous
        & finite_uv
        & on_mask
        & np.isfinite(observations).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= minimum_tracking_confidence)
    )

    positions = observations.copy()
    active_edges = active[mesh.edges].all(axis=1)
    edge_ids = np.flatnonzero(active_edges)
    for _ in range(iterations):
        if not len(edge_ids):
            break
        edges = mesh.edges[edge_ids]
        rest = mesh.rest_edge_lengths[edge_ids]
        delta = positions[edges[:, 1]] - positions[edges[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        valid = np.isfinite(length) & (length > 1e-8) & (rest > 1e-8)
        correction = np.zeros_like(delta)
        correction[valid] = (
            0.5
            * ((length[valid] - rest[valid]) / length[valid])[:, None]
            * delta[valid]
        )
        accumulated = np.zeros_like(positions)
        degree = np.zeros(vertex_count, dtype=np.float32)
        np.add.at(accumulated, edges[:, 0], correction)
        np.add.at(accumulated, edges[:, 1], -correction)
        np.add.at(degree, edges[:, 0], 1.0)
        np.add.at(degree, edges[:, 1], 1.0)
        constrained = active & (degree > 0)
        positions[constrained] += (
            stretch_stiffness
            * accumulated[constrained]
            / degree[constrained, None]
        )
        attachment = observation_stiffness * np.clip(confidence, 0.0, 1.0)
        positions[active] = observations[active] + (
            positions[active] - observations[active]
        ) * (1.0 - attachment[active, None])

    edge_strain = np.full(len(mesh.edges), np.nan, dtype=np.float32)
    if len(edge_ids):
        length = _edge_lengths(positions, mesh.edges[edge_ids])
        edge_strain[edge_ids] = np.abs(
            length / mesh.rest_edge_lengths[edge_ids] - 1.0
        )
        per_vertex_strain = np.zeros(vertex_count, dtype=np.float32)
        np.maximum.at(per_vertex_strain, mesh.edges[edge_ids, 0], edge_strain[edge_ids])
        np.maximum.at(per_vertex_strain, mesh.edges[edge_ids, 1], edge_strain[edge_ids])
        active &= per_vertex_strain <= maximum_relative_edge_strain

    corrected_uv = _project_world_points(positions, camera_to_world, intrinsic)
    corrected_pixels, corrected_on_mask = _nearest_mask_pixels(
        corrected_uv,
        target_mask,
        maximum_distance=maximum_mask_snap_distance,
    )
    active &= corrected_on_mask
    corrected_ids = np.flatnonzero(active)
    rounded = np.rint(corrected_uv[corrected_ids]).astype(np.int32)
    height, width = mesh.image_shape
    rounded_inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    exactly_on_mask = np.zeros(len(corrected_ids), dtype=bool)
    inside_ids = np.flatnonzero(rounded_inside)
    exactly_on_mask[inside_ids] = target_mask[
        rounded[inside_ids, 1], rounded[inside_ids, 0]
    ]
    snap_ids = corrected_ids[~exactly_on_mask]
    corrected_uv[snap_ids] = corrected_pixels[snap_ids].astype(np.float32)

    unobserved = previous & ~active
    corrected_uv[~active] = np.nan
    positions[~active] = np.nan
    observations[~active] = np.nan
    return ClothFrameSolution(
        uv=corrected_uv,
        positions_world=positions,
        observations_world=observations,
        active=active,
        unobserved=unobserved,
        edge_relative_strain=edge_strain,
    )


def _bilinear_sample(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float32)
    points = np.asarray(uv, dtype=np.float32)
    scalar = values.ndim == 2
    if scalar:
        values = values[..., None]
    height, width, channels = values.shape
    output = np.full((len(points), channels), np.nan, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    valid = (
        finite
        & (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )
    ids = np.flatnonzero(valid)
    if len(ids):
        x = points[ids, 0]
        y = points[ids, 1]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        wx = (x - x0)[:, None]
        wy = (y - y0)[:, None]
        top = values[y0, x0] * (1.0 - wx) + values[y0, x1] * wx
        bottom = values[y1, x0] * (1.0 - wx) + values[y1, x1] * wx
        output[ids] = top * (1.0 - wy) + bottom * wy
    return output[:, 0] if scalar else output


def _trajectory_frame_metrics(
    mesh: ClothReferenceMesh,
    *,
    frame_index: int,
    uv: np.ndarray,
    positions_world: np.ndarray,
    active: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    camera_to_world: np.ndarray,
    edge_strain: np.ndarray,
    newly_unobserved: int,
    newly_terminated: int,
) -> dict[str, Any]:
    rendered = render_active_cloth_mask(
        mesh,
        vertex_uv=uv,
        active_vertices=active,
    )
    target = np.asarray(observed_mask, dtype=bool)
    intersection = rendered & target
    union = rendered | target
    active_faces = active[mesh.faces].all(axis=1)
    active_edges = active[mesh.edges].all(axis=1)
    strains = edge_strain[active_edges]
    strains = strains[np.isfinite(strains)]

    active_ids = np.flatnonzero(active)
    depth_errors = np.empty(0, dtype=np.float32)
    if len(active_ids):
        world_to_camera = np.linalg.inv(
            np.asarray(camera_to_world, dtype=np.float32)
        )
        camera = apply_pose(positions_world[active_ids], world_to_camera)
        pixels = np.rint(uv[active_ids]).astype(np.int32)
        sampled_depth = observed_depth[pixels[:, 1], pixels[:, 0]]
        valid_depth = (
            np.isfinite(camera[:, 2])
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0)
        )
        depth_errors = np.abs(camera[valid_depth, 2] - sampled_depth[valid_depth])

    return {
        "frame_index": frame_index,
        "observed_mask_pixels": int(target.sum()),
        "active_vertices": int(active.sum()),
        "active_vertex_fraction": float(active.mean()),
        "newly_unobserved_vertices": newly_unobserved,
        "newly_terminated_identities": newly_terminated,
        "active_faces": int(active_faces.sum()),
        "active_edges": int(active_edges.sum()),
        "mesh_mask_iou": (
            float(intersection.sum() / union.sum()) if union.any() else 1.0
        ),
        "observed_mask_coverage": (
            float(intersection.sum() / target.sum()) if target.any() else 1.0
        ),
        "rendered_mask_precision": (
            float(intersection.sum() / rendered.sum())
            if rendered.any()
            else float(not target.any())
        ),
        "edge_strain_median": float(np.median(strains)) if len(strains) else None,
        "edge_strain_p95": (
            float(np.quantile(strains, 0.95)) if len(strains) else None
        ),
        "edge_strain_maximum": float(strains.max()) if len(strains) else None,
        "visible_depth_error_median": (
            float(np.median(depth_errors)) if len(depth_errors) else None
        ),
        "visible_depth_error_p90": (
            float(np.quantile(depth_errors, 0.9)) if len(depth_errors) else None
        ),
    }


def propagate_constrained_cloth_tracks(
    mesh: ClothReferenceMesh,
    forward: Any,
    backward: Any,
    *,
    reference_frame: int,
    world_points: np.ndarray,
    camera_points: np.ndarray,
    intrinsics: np.ndarray,
    camera_poses: np.ndarray,
    masks: np.ndarray,
    policy_paused_frames: np.ndarray,
    minimum_tracking_confidence: float = 0.05,
    maximum_mask_snap_distance: float = 5.0,
    solver_iterations: int = 24,
    stretch_stiffness: float = 2.5,
    observation_stiffness: float = 0.0,
    maximum_relative_edge_strain: float = 2.0,
    maximum_reassociation_gap: int = 3,
) -> ConstrainedClothTrajectory:
    """Propagate cloth IDs, excluding nonempty policy pauses from gap expiry.

    The canonical reference frame always emits its initialization even if the
    border policy marks it as paused. Empty masks still consume the invisible-gap
    budget. Other policy pauses emit no positions.
    """

    world = np.asarray(world_points, dtype=np.float32)
    camera = np.asarray(camera_points, dtype=np.float32)
    raw_masks = np.asarray(masks, dtype=bool)
    frame_count, height, width = world.shape[:3]
    expected_uv_shape = (frame_count - 1, height, width, 2)
    if forward.source_to_target_uv.shape != expected_uv_shape:
        raise ValueError("forward adjacent flow does not match cloth geometry")
    if backward.source_to_target_uv.shape != expected_uv_shape:
        raise ValueError("backward adjacent flow does not match cloth geometry")
    if raw_masks.shape != (frame_count, height, width):
        raise ValueError("cloth masks do not match cloth geometry")
    paused = np.asarray(policy_paused_frames, dtype=bool)
    if paused.shape != (frame_count,):
        raise ValueError("cloth pause flags must have one value per frame")
    raw_mask_pixels = raw_masks.sum(axis=(1, 2))
    paused = paused & (raw_mask_pixels > 0)
    paused[reference_frame] = False
    object_masks = raw_masks.copy()
    object_masks[paused] = False

    vertex_count = len(mesh.vertex_uv)
    trajectory_uv = np.full(
        (frame_count, vertex_count, 2), np.nan, dtype=np.float32
    )
    trajectory_world = np.full(
        (frame_count, vertex_count, 3), np.nan, dtype=np.float32
    )
    trajectory_confidence = np.full(
        (frame_count, vertex_count), np.nan, dtype=np.float32
    )
    trajectory_active = np.zeros((frame_count, vertex_count), dtype=bool)
    state_uv = np.full_like(trajectory_uv, np.nan)
    trackable = np.zeros((frame_count, vertex_count), dtype=bool)
    invisible_gap = np.full(
        (frame_count, vertex_count), maximum_reassociation_gap + 1, dtype=np.int16
    )
    if not paused[reference_frame]:
        trajectory_uv[reference_frame] = mesh.vertex_uv
        trajectory_world[reference_frame] = mesh.vertices_world
        trajectory_confidence[reference_frame] = 1.0
        trajectory_active[reference_frame] = True
    state_uv[reference_frame] = mesh.vertex_uv
    trackable[reference_frame] = True
    invisible_gap[reference_frame] = 0
    frame_metrics: list[dict[str, Any] | None] = [None] * frame_count
    frame_metrics[reference_frame] = _trajectory_frame_metrics(
        mesh,
        frame_index=reference_frame,
        uv=trajectory_uv[reference_frame],
        positions_world=trajectory_world[reference_frame],
        active=trajectory_active[reference_frame],
        observed_mask=object_masks[reference_frame],
        observed_depth=camera[reference_frame, ..., 2],
        camera_to_world=camera_poses[reference_frame],
        edge_strain=np.zeros(len(mesh.edges), dtype=np.float32),
        newly_unobserved=0,
        newly_terminated=0,
    )

    def advance(
        *,
        source: int,
        target: int,
        adjacent: Any,
        pair_index: int,
    ) -> None:
        proposed_uv = _bilinear_sample(
            adjacent.source_to_target_uv[pair_index], state_uv[source]
        )
        step_confidence = _bilinear_sample(
            adjacent.tracking_confidence[pair_index], state_uv[source]
        )
        solution = solve_visible_cloth_frame(
            mesh,
            proposed_uv=proposed_uv,
            tracking_confidence=step_confidence,
            previously_active=trackable[source],
            observed_mask=object_masks[target],
            observed_world_points=world[target],
            camera_to_world=camera_poses[target],
            intrinsic=intrinsics[target],
            minimum_tracking_confidence=minimum_tracking_confidence,
            iterations=solver_iterations,
            stretch_stiffness=stretch_stiffness,
            observation_stiffness=observation_stiffness,
            maximum_relative_edge_strain=maximum_relative_edge_strain,
            maximum_mask_snap_distance=maximum_mask_snap_distance,
        )
        trajectory_uv[target] = solution.uv
        trajectory_world[target] = solution.positions_world
        trajectory_confidence[target, solution.active] = step_confidence[
            solution.active
        ]
        trajectory_active[target] = solution.active
        proposed_valid = np.isfinite(proposed_uv).all(axis=1)
        # A deliberate skip is neither a failed observation nor a new one:
        # preserve the existing gap, without incrementing or resetting it.
        next_gap = invisible_gap[source] + (0 if paused[target] else 1)
        next_gap[solution.active] = 0
        trackable[target] = (
            trackable[source]
            & proposed_valid
            & (next_gap <= maximum_reassociation_gap)
        )
        invisible_gap[target] = next_gap
        state_uv[target, trackable[target]] = proposed_uv[trackable[target]]
        state_uv[target, solution.active] = solution.uv[solution.active]
        newly_terminated = trackable[source] & ~trackable[target]
        newly_unobserved = trajectory_active[source] & ~solution.active
        frame_metrics[target] = _trajectory_frame_metrics(
            mesh,
            frame_index=target,
            uv=solution.uv,
            positions_world=solution.positions_world,
            active=solution.active,
            observed_mask=object_masks[target],
            observed_depth=camera[target, ..., 2],
            camera_to_world=camera_poses[target],
            edge_strain=solution.edge_relative_strain,
            newly_unobserved=int(newly_unobserved.sum()),
            newly_terminated=int(newly_terminated.sum()),
        )

    for source in range(reference_frame, frame_count - 1):
        advance(
            source=source,
            target=source + 1,
            adjacent=forward,
            pair_index=source,
        )
    for source in range(reference_frame, 0, -1):
        advance(
            source=source,
            target=source - 1,
            adjacent=backward,
            pair_index=frame_count - 1 - source,
        )
    for frame, metrics in enumerate(frame_metrics):
        metrics.update(
            raw_mask_pixels=int(raw_mask_pixels[frame]),
            policy_paused=bool(paused[frame]),
            observation_state=(
                "policy_paused" if paused[frame]
                else "observed" if raw_mask_pixels[frame] else "empty_mask"
            ),
        )
    return ConstrainedClothTrajectory(
        uv=trajectory_uv,
        positions_world=trajectory_world,
        tracking_confidence=trajectory_confidence,
        active=trajectory_active,
        trackable=trackable,
        frame_metrics=tuple(item for item in frame_metrics if item is not None),
    )
