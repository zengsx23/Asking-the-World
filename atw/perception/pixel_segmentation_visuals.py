"""Render coordinate grids over original video frames."""

from pathlib import Path

import cv2
import numpy as np


def coordinate_grid(raw: np.ndarray, out: Path) -> Path:
    grid = raw.copy()
    h, w = raw.shape[:2]
    for n in range(100, 1000, 100):
        x, y = round(n * (w - 1) / 1000), round(n * (h - 1) / 1000)
        cv2.line(grid, (x, 0), (x, h - 1), (110, 110, 110), 1)
        cv2.line(grid, (0, y), (w - 1, y), (110, 110, 110), 1)
        cv2.putText(grid, str(n), (x + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
        cv2.putText(grid, str(n), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    path = out / "coordinates.jpg"
    cv2.imwrite(str(path), grid)
    return path
