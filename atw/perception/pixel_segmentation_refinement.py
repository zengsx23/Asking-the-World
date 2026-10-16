"""Extract the interior domain enclosed by a boundary mask."""

from __future__ import annotations


import cv2
import numpy as np


def enclosed_domain(wall: np.ndarray) -> np.ndarray:
    """Return the largest free component enclosed by a measured wall mask."""
    wall = np.asarray(wall, dtype=bool)
    if wall.ndim != 2 or not wall.any():
        raise ValueError("A present boundary needs a nonempty 2D mask")
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (~wall).astype(np.uint8), 8
    )
    border = set(np.unique(np.concatenate((
        components[0], components[-1], components[:, 0], components[:, -1]
    ))).tolist())
    candidates = [label for label in range(1, count) if label not in border]
    if not candidates:
        raise ValueError("The boundary mask does not enclose an interior domain")
    chosen = max(candidates, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    return components == chosen
