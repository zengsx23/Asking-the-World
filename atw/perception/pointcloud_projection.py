"""Camera projection and z-buffer splatting for sparse tracking point clouds."""

from __future__ import annotations

import numpy as np


def apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    transform = np.asarray(pose, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError("pose must have shape (4, 4)")
    return array @ transform[:3, :3].T + transform[:3, 3]


def sparse_point_zbuffer(
    *,
    world_points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size_wh: tuple[int, int],
    point_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize sparse points and retain the nearest source ID per pixel."""

    return sparse_point_zbuffers(
        world_points=world_points,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        image_size_wh=image_size_wh,
        point_radii=(point_radius,),
    )[point_radius]


def sparse_point_zbuffers(
    *,
    world_points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size_wh: tuple[int, int],
    point_radii: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Rasterize multiple point radii after one camera transform and projection."""

    width, height = image_size_wh
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if not point_radii:
        raise ValueError("at least one point radius is required")
    if len(set(point_radii)) != len(point_radii):
        raise ValueError("point radii must be unique")
    if any(point_radius < 0 for point_radius in point_radii):
        raise ValueError("point radius cannot be negative")
    points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float32))
    camera = apply_pose(points, world_to_camera)
    projected = camera @ np.asarray(intrinsics, dtype=np.float32).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    center = np.rint(np.where(np.isfinite(uv), uv, -1)).astype(np.int32)
    valid_source = (
        np.isfinite(camera).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (camera[:, 2] > 0)
    )
    source_ids = np.flatnonzero(valid_source).astype(np.int32)
    if not len(source_ids):
        return {
            point_radius: (
                np.full((height, width), -1, dtype=np.int32),
                np.full((height, width), np.nan, dtype=np.float32),
            )
            for point_radius in point_radii
        }

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for point_radius in point_radii:
        pixel_chunks: list[np.ndarray] = []
        source_chunks: list[np.ndarray] = []
        for offset_y in range(-point_radius, point_radius + 1):
            for offset_x in range(-point_radius, point_radius + 1):
                if (
                    offset_x * offset_x + offset_y * offset_y
                    > point_radius * point_radius
                ):
                    continue
                x = center[source_ids, 0] + offset_x
                y = center[source_ids, 1] + offset_y
                inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                pixel_chunks.append(
                    (y[inside] * width + x[inside]).astype(np.int64)
                )
                source_chunks.append(source_ids[inside])
        pixels = np.concatenate(pixel_chunks)
        sources = np.concatenate(source_chunks)
        id_image = np.full(height * width, -1, dtype=np.int32)
        depth_image = np.full(height * width, np.nan, dtype=np.float32)
        if len(pixels):
            depths = camera[sources, 2]
            if point_radius == 0:
                order = np.lexsort((depths, pixels))
                sorted_pixels = pixels[order]
                first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
                selected = order[first]
                id_image[pixels[selected]] = sources[selected]
                depth_image[pixels[selected]] = depths[selected]
            else:
                minimum_depth = np.full(height * width, np.inf, dtype=np.float32)
                np.minimum.at(minimum_depth, pixels, depths)
                frontmost = depths == minimum_depth[pixels]
                candidate_order = np.arange(len(pixels), dtype=np.int64)
                first_frontmost = np.full(
                    height * width,
                    np.iinfo(np.int64).max,
                    dtype=np.int64,
                )
                np.minimum.at(
                    first_frontmost,
                    pixels[frontmost],
                    candidate_order[frontmost],
                )
                occupied = np.isfinite(minimum_depth)
                selected = first_frontmost[occupied]
                id_image[occupied] = sources[selected]
                depth_image[occupied] = depths[selected]
        results[point_radius] = (
            id_image.reshape(height, width),
            depth_image.reshape(height, width),
        )
    return results
