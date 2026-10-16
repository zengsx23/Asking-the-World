"""Triangle Z-buffer projection used by rigid reference-frame alignment."""

from __future__ import annotations

from typing import Any

import numpy as np


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    target_h, target_w = shape
    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def render_mesh_depth(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the nearest triangle depth in an OpenCV camera frame."""

    height, width = image_shape
    vertices_camera = np.asarray(vertices_camera, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)

    z = vertices_camera[:, 2]
    valid = (
        np.isfinite(vertices_camera).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-6)
    )
    uv = np.full((vertices_camera.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = (
        intrinsic[0, 0] * vertices_camera[valid, 0] / z[valid]
        + intrinsic[0, 2]
    )
    uv[valid, 1] = (
        intrinsic[1, 1] * vertices_camera[valid, 1] / z[valid]
        + intrinsic[1, 2]
    )

    depth = np.full((height, width), np.inf, dtype=np.float32)
    for face in faces:
        if face.shape[0] != 3 or not valid[face].all():
            continue
        tri_uv = uv[face]
        tri_z = z[face]
        if not np.isfinite(tri_uv).all() or not np.isfinite(tri_z).all():
            continue

        min_xy = np.floor(tri_uv.min(axis=0)).astype(int)
        max_xy = np.ceil(tri_uv.max(axis=0)).astype(int)
        x0 = max(0, int(min_xy[0]))
        y0 = max(0, int(min_xy[1]))
        x1 = min(width - 1, int(max_xy[0]))
        y1 = min(height - 1, int(max_xy[1]))
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        p0, p1, p2 = tri_uv.astype(np.float32)
        denom = (
            (p1[1] - p2[1]) * (p0[0] - p2[0])
            + (p2[0] - p1[0]) * (p0[1] - p2[1])
        )
        if abs(float(denom)) < 1e-6:
            continue

        w0 = (
            (p1[1] - p2[1]) * (grid_x - p2[0])
            + (p2[0] - p1[0]) * (grid_y - p2[1])
        ) / denom
        w1 = (
            (p2[1] - p0[1]) * (grid_x - p2[0])
            + (p0[0] - p2[0]) * (grid_y - p2[1])
        ) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue

        tri_depth = (
            w0 * float(tri_z[0])
            + w1 * float(tri_z[1])
            + w2 * float(tri_z[2])
        )
        region_depth = depth[y0 : y1 + 1, x0 : x1 + 1]
        update = inside & (tri_depth < region_depth)
        if update.any():
            region_depth[update] = tri_depth[update]

    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return mask, depth


def mask_occluded_mesh_mask(
    *,
    rendered_mask: np.ndarray,
    occluder_mask: np.ndarray | None,
    self_mask: np.ndarray | None,
) -> np.ndarray:
    """Remove pixels assigned by segmentation to another recognized object."""

    visible = np.asarray(rendered_mask).astype(bool).copy()
    if occluder_mask is None:
        return visible

    occluder = np.asarray(occluder_mask).astype(bool)
    if occluder.shape != visible.shape:
        occluder = resize_mask_nearest(occluder, visible.shape)

    if self_mask is not None:
        own = np.asarray(self_mask).astype(bool)
        if own.shape != visible.shape:
            own = resize_mask_nearest(own, visible.shape)
        occluder = occluder & ~own

    return visible & ~occluder


def render_mask_occluded_mesh(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    occluder_mask: np.ndarray | None = None,
    self_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Render triangle depth, then apply depth-free recognized-mask occlusion."""

    rendered_mask, rendered_depth = render_mesh_depth(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsic=intrinsic,
        image_shape=image_shape,
    )
    visible_mask = mask_occluded_mesh_mask(
        rendered_mask=rendered_mask,
        occluder_mask=occluder_mask,
        self_mask=self_mask,
    )
    return {
        "rendered_mask": rendered_mask,
        "rendered_depth": rendered_depth,
        "visible_mask": visible_mask,
    }
