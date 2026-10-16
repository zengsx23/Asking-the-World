"""Mask proximity and reconstruction-frame selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


CONTACT_MARGIN_PIXELS_AT_480 = 2


@dataclass(frozen=True, slots=True)
class ReconstructionFrameSelection:
    track_id: str
    frame_index: int
    mask_area: int
    touches_other_mask: bool
    touches_frame_edge: bool
    used_contact_fallback: bool
    used_edge_fallback: bool
    positive_mask_frames: int
    non_contact_frames: int
    non_edge_frames: int
    contact_margin_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unpack_mask(array: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    unpacked = np.unpackbits(array)[: int(metadata["num_bits"])]
    return unpacked.reshape(
        int(metadata["height"]), int(metadata["width"])
    ).astype(bool)


def read_packed_mask_archive(
    archive_path: Path,
    index_path: Path,
) -> dict[str, dict[int, np.ndarray]]:
    """Read the packed mask format emitted by the SAM3 video tool."""

    index = json.loads(index_path.read_text(encoding="utf-8"))
    tracks: dict[str, dict[int, np.ndarray]] = {}
    with np.load(archive_path) as archive:
        for key, metadata in index.items():
            track_id = str(metadata["track_id"])
            frame_index = int(metadata["frame_index"])
            tracks.setdefault(track_id, {})[frame_index] = _unpack_mask(
                archive[key], metadata
            )
    return tracks


def default_contact_margin_pixels(mask_width: int) -> int:
    """Scale the two-pixel contact margin from a reference width of 480 pixels."""

    return max(1, round(CONTACT_MARGIN_PIXELS_AT_480 * mask_width / 480))


def masks_touch(
    first: np.ndarray,
    second: np.ndarray,
    *,
    margin_pixels: int | None = None,
) -> bool:
    """Test proximity; default to 2 pixels at width 480, scaled to mask width.

    An explicit margin is measured directly in the supplied mask's pixels.
    """

    if first.shape != second.shape:
        raise ValueError(f"mask shape mismatch: {first.shape} != {second.shape}")
    if margin_pixels is None:
        margin_pixels = default_contact_margin_pixels(first.shape[1])
    if margin_pixels < 0:
        raise ValueError("margin_pixels must be non-negative")
    if not np.any(first) or not np.any(second):
        return False
    if margin_pixels == 0:
        return bool(np.any(first & second))
    width = margin_pixels * 2 + 1
    dilated = cv2.dilate(
        second.astype(np.uint8),
        np.ones((width, width), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    return bool(np.any(first & dilated))


def mask_touches_frame_edge(mask: np.ndarray) -> bool:
    """Return whether a non-empty mask reaches any image boundary."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got shape {mask.shape}")
    if not np.any(mask):
        return False
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def select_reconstruction_frame(
    track_id: str,
    tracks: Mapping[str, Mapping[int, np.ndarray]],
    *,
    contact_margin_pixels: int | None = None,
) -> ReconstructionFrameSelection:
    """Prefer non-edge, non-contact, then area with resolution-aware contact."""

    if track_id not in tracks:
        raise KeyError(f"unknown track: {track_id}")
    candidates: list[tuple[int, int, bool, bool]] = []
    for frame_index, mask in tracks[track_id].items():
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        touching = any(
            masks_touch(
                mask,
                other_frames[frame_index],
                margin_pixels=contact_margin_pixels,
            )
            for other_id, other_frames in tracks.items()
            if other_id != track_id
            and frame_index in other_frames
            and np.any(other_frames[frame_index])
        )
        candidates.append(
            (area, frame_index, touching, mask_touches_frame_edge(mask))
        )
    if not candidates:
        raise ValueError(f"track has no positive-area masks: {track_id}")

    non_edge = [candidate for candidate in candidates if not candidate[3]]
    edge_eligible = non_edge or candidates
    non_contact = [candidate for candidate in edge_eligible if not candidate[2]]
    eligible = non_contact or edge_eligible
    area, frame_index, touching, touching_edge = max(
        eligible,
        key=lambda candidate: (candidate[0], -candidate[1]),
    )
    return ReconstructionFrameSelection(
        track_id=track_id,
        frame_index=frame_index,
        mask_area=area,
        touches_other_mask=touching,
        touches_frame_edge=touching_edge,
        used_contact_fallback=not non_contact,
        used_edge_fallback=not non_edge,
        positive_mask_frames=len(candidates),
        non_contact_frames=sum(not candidate[2] for candidate in candidates),
        non_edge_frames=len(non_edge),
        contact_margin_pixels=(
            default_contact_margin_pixels(tracks[track_id][frame_index].shape[1])
            if contact_margin_pixels is None else contact_margin_pixels
        ),
    )
