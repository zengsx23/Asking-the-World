"""Mask archives and image evidence cards."""

from contextlib import contextmanager
import zipfile
import cv2
import numpy as np


@contextmanager
def mask_archive(path):
    """Keep completed records; remove only an incomplete archive on failure."""
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            yield archive
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def card(raw, title, size=(320, 220)):
    width, height = size
    result = np.zeros((height, width, 3), np.uint8)
    scale = min(width / raw.shape[1], (height - 32) / raw.shape[0])
    wh = (max(1, round(raw.shape[1] * scale)), max(1, round(raw.shape[0] * scale)))
    result[32 : 32 + wh[1], : wh[0]] = cv2.resize(raw, wh)
    cv2.putText(
        result, title[:70], (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )
    return result


def board(tiles, path, columns=3):
    tiles = list(tiles)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(
        str(path),
        np.vstack(
            [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
        ),
    )
    return path
