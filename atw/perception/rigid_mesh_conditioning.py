"""PhysMind-style primitive fitting for canonical rigid meshes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Literal

import numpy as np


RigidShape = Literal["sphere", "box", "cylinder", "irregular"]
CLEVRER_IRREGULAR_TARGET_FACES = 2_000


@dataclass(frozen=True, slots=True)
class RigidMeshConditioning:
    mesh: Any
    action: str
    fit: dict[str, Any]
    representative_color: dict[str, Any]


def _local_bbox(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return (lower + upper) * 0.5, upper - lower


def _sphere_mesh(vertices: np.ndarray) -> tuple[Any, dict[str, Any]]:
    import trimesh

    center, extent = _local_bbox(vertices)
    radius = float(np.median(extent) * 0.5)
    mesh = trimesh.creation.uv_sphere(radius=radius, count=[32, 16])
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) + center.reshape(1, 3)
    )
    return mesh, {
        "primitive": "sphere",
        "center": center.tolist(),
        "radius": radius,
        "source_extent": extent.tolist(),
    }


def _box_mesh(vertices: np.ndarray) -> tuple[Any, dict[str, Any]]:
    import trimesh

    _, original_extent = _local_bbox(vertices)
    to_origin, minimum_extent = trimesh.bounds.oriented_bounds(
        vertices, angle_digits=2
    )
    basis = np.linalg.inv(to_origin)[:3, :3]
    # A box has 24 equivalent right-handed axis labelings. Match the original
    # local XYZ before comparing directions or exporting a canonical frame.
    equivalent_axes = (
        basis[:, permutation] * np.asarray(signs)[None, :]
        for permutation in permutations(range(3))
        for signs in product((-1, 1), repeat=3)
    )
    minimum_axes = max(
        (axes for axes in equivalent_axes if np.linalg.det(axes) > 0),
        key=np.trace,
    )
    original_volume = float(np.prod(original_extent))
    minimum_volume = float(np.prod(minimum_extent))
    axes = minimum_axes
    box_center, extent = _local_bbox(vertices @ axes)
    center = axes @ box_center
    mesh = trimesh.creation.box(extents=extent)
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) @ axes.T + center.reshape(1, 3)
    )
    return mesh, {
        "primitive": "box",
        "center": center.tolist(),
        "extent": extent.tolist(),
        "axes": axes.tolist(),
        "axes_coordinate_frame": "input_canonical_mesh",
        "axis_source": "mesh_minimum_volume_box",
        "fit_method": "minimum_volume_bounding_box",
        "axis_labeling": "closest_right_handed_signed_permutation_to_input_xyz",
        "original_aabb_volume": original_volume,
        "minimum_box_volume": minimum_volume,
        "axis_deviation_degrees": np.rad2deg(
            np.arccos(np.clip(np.diag(axes), -1, 1))
        ).tolist(),
    }


def _cylinder_mesh(vertices: np.ndarray) -> tuple[Any, dict[str, Any]]:
    import trimesh

    center, extent = _local_bbox(vertices)
    candidates: list[tuple[float, float, int]] = []
    for axis_index in range(3):
        plane_indices = [index for index in range(3) if index != axis_index]
        radial = np.linalg.norm(
            vertices[:, plane_indices] - center[plane_indices].reshape(1, 2),
            axis=1,
        )
        positive = radial[np.isfinite(radial) & (radial > 1e-12)]
        score = (
            float(np.std(positive) / max(float(np.mean(positive)), 1e-12))
            if len(positive)
            else float("inf")
        )
        candidates.append((score, -float(extent[axis_index]), axis_index))
    candidates.sort(key=lambda item: (item[0], item[1]))
    radial_score, _, axis_index = candidates[0]
    plane_indices = [index for index in range(3) if index != axis_index]
    height = float(extent[axis_index])
    radius = float(np.median(extent[plane_indices]) * 0.5)

    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    local_axes = np.eye(3, dtype=np.float64)
    cylinder_axes = np.eye(3, dtype=np.float64)
    cylinder_axes[:, 2] = local_axes[:, axis_index]
    cylinder_axes[:, 0] = local_axes[:, plane_indices[0]]
    cylinder_axes[:, 1] = local_axes[:, plane_indices[1]]
    if np.linalg.det(cylinder_axes) < 0:
        cylinder_axes[:, 1] *= -1.0
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) @ cylinder_axes.T
        + center.reshape(1, 3)
    )
    return mesh, {
        "primitive": "cylinder",
        "center": center.tolist(),
        "radius": radius,
        "height": height,
        "axis_index": int(axis_index),
        "axis": local_axes[:, axis_index].tolist(),
        "radial_distance_coefficient_of_variation": radial_score,
        "axis_source": "sam3d_canonical_local_xyz",
        "axis_selection_method": "local_radial_distance_consistency",
        "axis_candidates": [
            {
                "axis_index": int(candidate_axis),
                "radial_distance_coefficient_of_variation": float(score),
                "axis_extent": float(-negative_extent),
            }
            for score, negative_extent, candidate_axis in candidates
        ],
        "source_extent": extent.tolist(),
    }


def _decimate_mesh(mesh: Any, target_faces: int) -> tuple[Any, dict[str, Any]]:
    original_faces = int(len(mesh.faces))
    original_vertices = int(len(mesh.vertices))
    if original_faces <= target_faces:
        return mesh.copy(), {
            "status": "skipped",
            "target_faces": int(target_faces),
            "original_faces": original_faces,
            "original_vertices": original_vertices,
            "conditioned_faces": original_faces,
            "conditioned_vertices": original_vertices,
        }
    import open3d as o3d
    import trimesh

    source = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices, dtype=np.float64)
        ),
        triangles=o3d.utility.Vector3iVector(
            np.asarray(mesh.faces, dtype=np.int32)
        ),
    )
    simplified = source.simplify_quadric_decimation(
        target_number_of_triangles=int(target_faces)
    )
    simplified.remove_duplicated_vertices()
    simplified.remove_duplicated_triangles()
    simplified.remove_degenerate_triangles()
    simplified.remove_unreferenced_vertices()
    vertices = np.asarray(simplified.vertices, dtype=np.float64)
    faces = np.asarray(simplified.triangles, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("mesh decimation produced an empty mesh")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), {
        "status": "decimated",
        "target_faces": int(target_faces),
        "original_faces": original_faces,
        "original_vertices": original_vertices,
        "conditioned_faces": int(len(faces)),
        "conditioned_vertices": int(len(vertices)),
    }


def _representative_vertex_color(mesh: Any) -> tuple[np.ndarray, dict[str, Any]]:
    raw = getattr(mesh.visual, "vertex_colors", None)
    if raw is None or not len(raw):
        color = np.asarray([160, 160, 160, 255], dtype=np.uint8)
        return color, {"method": "default_neutral_rgba", "rgba": color.tolist()}
    colors = np.asarray(raw)
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError(f"unsupported source vertex colors: {colors.shape}")
    if colors.shape[1] < 4:
        colors = np.column_stack(
            (colors[:, :3], np.full(len(colors), 255, dtype=colors.dtype))
        )
    else:
        colors = colors[:, :4]
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    color = np.median(colors, axis=0).astype(np.uint8)
    color[3] = 255
    return color, {
        "method": "median_source_vertex_color",
        "source_vertex_color_count": int(len(colors)),
        "rgba": color.tolist(),
    }


def condition_rigid_mesh(
    mesh: Any,
    rigid_shape: RigidShape,
    *,
    irregular_target_faces: int = CLEVRER_IRREGULAR_TARGET_FACES,
) -> RigidMeshConditioning:
    """Replace regular CLEVRER geometry with a deterministic fitted primitive."""

    import trimesh

    if rigid_shape not in {"sphere", "box", "cylinder", "irregular"}:
        raise ValueError(f"unsupported rigid shape: {rigid_shape!r}")
    if irregular_target_faces < 4:
        raise ValueError("irregular target face count must be at least four")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("mesh must contain at least four 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("mesh must contain triangular faces")

    if rigid_shape == "sphere":
        conditioned, fit = _sphere_mesh(vertices)
        action = "sphere_fit"
    elif rigid_shape == "box":
        conditioned, fit = _box_mesh(vertices)
        action = "box_fit"
    elif rigid_shape == "cylinder":
        conditioned, fit = _cylinder_mesh(vertices)
        action = "cylinder_fit"
    else:
        conditioned, fit = _decimate_mesh(mesh, irregular_target_faces)
        action = "decimated"

    color, color_metadata = _representative_vertex_color(mesh)
    conditioned = trimesh.Trimesh(
        vertices=np.asarray(conditioned.vertices, dtype=np.float64),
        faces=np.asarray(conditioned.faces, dtype=np.int64),
        process=False,
    )
    conditioned.visual.vertex_colors = np.tile(
        color.reshape(1, 4), (len(conditioned.vertices), 1)
    )
    return RigidMeshConditioning(
        mesh=conditioned,
        action=action,
        fit=fit,
        representative_color=color_metadata,
    )
