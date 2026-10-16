"""Shared sparse point-cloud policies for tracking, storage, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def deterministic_subsample_indices(
    point_count: int,
    *,
    maximum_points: int,
    seed: int,
) -> np.ndarray:
    """Return sorted, reproducible indices without changing retained point IDs."""

    if point_count < 0:
        raise ValueError("point count cannot be negative")
    if maximum_points < 1:
        raise ValueError("maximum point count must be positive")
    if point_count <= maximum_points:
        return np.arange(point_count, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(
        generator.choice(point_count, maximum_points, replace=False)
    ).astype(np.int64)


@dataclass(frozen=True, slots=True)
class StaticScenePointCloud:
    """One temporally fused cloud containing only observed background points."""

    world_points: np.ndarray
    colors: np.ndarray
    observation_counts: np.ndarray
    sampled_frame_indices: np.ndarray
    height: int
    width: int
    background_stride: int
    voxel_size: float
    input_point_count: int
    initially_masked_sample_count: int
    initially_masked_observed_later_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.world_points)
        colors = np.asarray(self.colors)
        counts = np.asarray(self.observation_counts)
        frames = np.asarray(self.sampled_frame_indices)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("world points must have shape (N, 3)")
        if colors.shape != points.shape:
            raise ValueError("point colors must match world points")
        if counts.shape != (len(points),):
            raise ValueError("observation counts must match world points")
        if frames.ndim != 1 or not len(frames):
            raise ValueError("sampled frame indices must be nonempty")
        if self.height < 1 or self.width < 1 or self.background_stride < 1:
            raise ValueError("scene dimensions and stride must be positive")
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel size must be positive and finite")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            world_points=np.asarray(self.world_points, dtype=np.float32),
            colors=np.asarray(self.colors, dtype=np.uint8),
            observation_counts=np.asarray(self.observation_counts, dtype=np.int16),
            sampled_frame_indices=np.asarray(
                self.sampled_frame_indices, dtype=np.int32
            ),
            height=np.asarray(self.height, dtype=np.int32),
            width=np.asarray(self.width, dtype=np.int32),
            background_stride=np.asarray(
                self.background_stride, dtype=np.int32
            ),
            voxel_size=np.asarray(self.voxel_size, dtype=np.float32),
            input_point_count=np.asarray(self.input_point_count, dtype=np.int64),
            initially_masked_sample_count=np.asarray(
                self.initially_masked_sample_count, dtype=np.int64
            ),
            initially_masked_observed_later_count=np.asarray(
                self.initially_masked_observed_later_count, dtype=np.int64
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "StaticScenePointCloud":
        with np.load(path) as archive:
            return cls(
                world_points=np.asarray(archive["world_points"], dtype=np.float32),
                colors=np.asarray(archive["colors"], dtype=np.uint8),
                observation_counts=np.asarray(
                    archive["observation_counts"], dtype=np.int16
                ),
                sampled_frame_indices=np.asarray(
                    archive["sampled_frame_indices"], dtype=np.int32
                ),
                height=int(archive["height"]),
                width=int(archive["width"]),
                background_stride=int(archive["background_stride"]),
                voxel_size=float(archive["voxel_size"]),
                input_point_count=int(archive["input_point_count"]),
                initially_masked_sample_count=int(
                    archive["initially_masked_sample_count"]
                ),
                initially_masked_observed_later_count=int(
                    archive["initially_masked_observed_later_count"]
                ),
            )


def uniform_sampled_frame_indices(
    frame_count: int,
    *,
    sample_count: int = 16,
) -> np.ndarray:
    """Return chronological, uniformly distributed frame indices."""

    if frame_count < 1 or sample_count < 1:
        raise ValueError("frame and sample counts must be positive")
    return np.unique(
        np.rint(np.linspace(0, frame_count - 1, min(frame_count, sample_count)))
        .astype(np.int32)
    )


def build_static_scene_pointcloud(
    *,
    world_points: np.ndarray,
    rgb: np.ndarray,
    tracked_region_masks: np.ndarray,
    geometry_valid: np.ndarray,
    sample_count: int = 16,
    background_stride: int = 2,
    voxel_size: float | None = None,
) -> StaticScenePointCloud:
    """Fuse uniformly sampled observed background into one voxelized cloud."""

    world = np.asarray(world_points, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    excluded = np.asarray(tracked_region_masks, dtype=bool)
    geometry = np.asarray(geometry_valid, dtype=bool)
    if world.ndim != 4 or world.shape[-1] != 3:
        raise ValueError("world point maps must have shape (T, H, W, 3)")
    if colors.shape != world.shape:
        raise ValueError("RGB frames must match the world point-map shape")
    if excluded.shape != world.shape[:3]:
        raise ValueError("tracked-region masks must match the point maps")
    if geometry.shape != world.shape[:3]:
        raise ValueError("geometry validity must match the point maps")
    if background_stride < 1:
        raise ValueError("background stride must be positive")

    frame_count, height, width = world.shape[:3]
    grid = np.zeros((height, width), dtype=bool)
    grid[::background_stride, ::background_stride] = True
    sampled_frames = uniform_sampled_frame_indices(
        frame_count, sample_count=sample_count
    )
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    observed_by_frame: list[np.ndarray] = []
    for frame_index in sampled_frames:
        valid = (
            grid
            & ~excluded[frame_index]
            & geometry[frame_index]
            & np.isfinite(world[frame_index]).all(axis=-1)
        )
        point_chunks.append(world[frame_index][valid])
        color_chunks.append(colors[frame_index][valid])
        frame_chunks.append(np.full(valid.sum(), frame_index, dtype=np.int32))
        observed_by_frame.append(valid)
    points = np.concatenate(point_chunks)
    point_colors = np.concatenate(color_chunks)
    point_frames = np.concatenate(frame_chunks)
    if not len(points):
        raise ValueError("sampled frames contain no valid background points")
    if voxel_size is None:
        lower, upper = np.percentile(points, (1, 99), axis=0)
        voxel_size = max(
            0.0035 * float(np.linalg.norm(upper - lower)),
            1e-6,
        )
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel size must be positive and finite")

    origin = points.min(axis=0)
    voxel = np.floor((points - origin) / voxel_size).astype(np.int32)
    order = np.lexsort((voxel[:, 2], voxel[:, 1], voxel[:, 0]))
    sorted_voxel = voxel[order]
    sorted_points = points[order].astype(np.float64)
    sorted_colors = point_colors[order].astype(np.float64)
    sorted_frames = point_frames[order]
    starts = np.r_[
        True,
        np.any(sorted_voxel[1:] != sorted_voxel[:-1], axis=1),
    ]
    group_starts = np.flatnonzero(starts)
    sample_counts = np.diff(np.r_[group_starts, len(points)])
    fused_points = np.add.reduceat(sorted_points, group_starts, axis=0)
    fused_points /= sample_counts[:, None]
    fused_colors = np.add.reduceat(sorted_colors, group_starts, axis=0)
    fused_colors /= sample_counts[:, None]

    voxel_group = np.cumsum(starts) - 1
    new_frame_observation = starts.copy()
    new_frame_observation[1:] |= sorted_frames[1:] != sorted_frames[:-1]
    observation_counts = np.bincount(
        voxel_group[new_frame_observation], minlength=len(group_starts)
    )
    initially_masked = grid & excluded[sampled_frames[0]]
    observed_later = (
        np.logical_or.reduce(observed_by_frame[1:])
        if len(observed_by_frame) > 1
        else np.zeros_like(grid)
    )
    return StaticScenePointCloud(
        world_points=fused_points.astype(np.float32),
        colors=np.clip(np.rint(fused_colors), 0, 255).astype(np.uint8),
        observation_counts=observation_counts.astype(np.int16),
        sampled_frame_indices=sampled_frames,
        height=height,
        width=width,
        background_stride=background_stride,
        voxel_size=float(voxel_size),
        input_point_count=len(points),
        initially_masked_sample_count=int(initially_masked.sum()),
        initially_masked_observed_later_count=int(
            np.count_nonzero(initially_masked & observed_later)
        ),
    )
