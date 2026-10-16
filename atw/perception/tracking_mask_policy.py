"""Shared per-frame mask visibility policy for tracked physical objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrackingMaskFramePolicy:
    mask_pixel_counts: np.ndarray
    occlusion_termination_frame: int
    occlusion_terminated: np.ndarray
    border_touching: np.ndarray
    nearest_nonborder_frames: np.ndarray
    border_area_ratios: np.ndarray
    border_suppressed: np.ndarray


def confirmed_occlusion_termination_frame(
    mask_pixel_counts: np.ndarray,
    *,
    minimum_occlusion_frames: int = 3,
) -> int:
    """Return the first frame of a confirmed empty-mask run after visibility."""

    counts = np.asarray(mask_pixel_counts).reshape(-1)
    if minimum_occlusion_frames < 1:
        raise ValueError("minimum occlusion frames must be positive")
    visible = counts > 0
    visible_frames = np.flatnonzero(visible)
    if not len(visible_frames):
        return -1
    frame = int(visible_frames[0]) + 1
    while frame < len(visible):
        if visible[frame]:
            frame += 1
            continue
        run_start = frame
        while frame < len(visible) and not visible[frame]:
            frame += 1
        if frame - run_start >= minimum_occlusion_frames:
            return run_start
    return -1


def classify_tracking_mask_frames(
    masks: np.ndarray,
    *,
    minimum_occlusion_frames: int = 3,
) -> TrackingMaskFramePolicy:
    """Suppress every border frame unless the whole visible track is border-only."""

    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("tracking masks must have shape TxHxW")
    if minimum_occlusion_frames < 1:
        raise ValueError("minimum occlusion frames must be positive")
    frame_count = len(masks)
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    visible = areas > 0
    termination_frame = confirmed_occlusion_termination_frame(
        areas,
        minimum_occlusion_frames=minimum_occlusion_frames,
    )
    occlusion_terminated = np.zeros(frame_count, dtype=bool)
    if termination_frame >= 0:
        occlusion_terminated[termination_frame:] = True
    touching = visible & (
        masks[:, 0, :].any(axis=1)
        | masks[:, -1, :].any(axis=1)
        | masks[:, :, 0].any(axis=1)
        | masks[:, :, -1].any(axis=1)
    )
    interior_frames = np.flatnonzero(visible & ~touching)
    nearest = np.full(frame_count, -1, dtype=np.int32)
    ratios = np.full(frame_count, np.nan, dtype=np.float32)
    suppressed = np.zeros(frame_count, dtype=bool)
    if len(interior_frames):
        suppressed = touching.copy()
        for frame in np.flatnonzero(touching):
            distances = np.abs(interior_frames - frame)
            nearest_frame = int(interior_frames[np.argmin(distances)])
            nearest[frame] = nearest_frame
            ratios[frame] = float(areas[frame] / areas[nearest_frame])
    return TrackingMaskFramePolicy(
        mask_pixel_counts=areas.astype(np.int32),
        occlusion_termination_frame=termination_frame,
        occlusion_terminated=occlusion_terminated,
        border_touching=touching,
        nearest_nonborder_frames=nearest,
        border_area_ratios=ratios,
        border_suppressed=suppressed,
    )
