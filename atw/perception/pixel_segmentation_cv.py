"""CPU color regions and persistent outlines for planar pixel segmentation."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


UNKNOWN_ID = 65535


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    source_wh: tuple[int, int]
    working_wh: tuple[int, int]
    mask_wh: tuple[int, int]


def inspect_video(path: Path, working_width: int = 960, mask_width: int = 480) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = (int(cap.get(key)) for key in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if count < 1 or fps <= 0 or min(width, height) < 1:
            raise ValueError("Input must be a decodable video with frames and FPS")
        if min(working_width, mask_width) < 2:
            raise ValueError("Working and mask widths must be at least two pixels")
        return VideoInfo(count, fps, (width, height),
                         (working_width, max(2, round(height * working_width / width))),
                         (mask_width, max(2, round(height * mask_width / width))))
    finally:
        cap.release()


def sample_frames(count: int, fractions=(0., .25, .5, .75, 1.)) -> list[int]:
    return sorted({round((count - 1) * fraction) for fraction in fractions})


def read_frame(path: Path, frame: int, size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, raw = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {frame}")
        return cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
    finally:
        cap.release()


def iter_frames(path: Path, size: tuple[int, int]):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            yield cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
    finally:
        cap.release()


def encode_bgr(image: np.ndarray) -> np.ndarray:
    quantized = image.astype(np.uint32) // 4
    return quantized[..., 0] * 4096 + quantized[..., 1] * 64 + quantized[..., 2]


def learn_palette(video: Path, info: VideoInfo, size: int = 32) -> dict:
    rng = np.random.default_rng(41)
    samples = []
    for frame in sample_frames(info.frame_count, np.linspace(0, 1, 11)):
        raw = read_frame(video, frame, info.working_wh)
        lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        samples.append(lab[rng.choice(len(lab), min(9000, len(lab)), replace=False)])
    points = np.concatenate(samples).astype(np.float32)
    size = min(size, len(points))
    cv2.setRNGSeed(41)
    _, _, centers = cv2.kmeans(points, size, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, .1),
        3, cv2.KMEANS_PP_CENTERS)
    centers = centers[np.argsort(centers[:, 0])]
    grid = np.indices((64, 64, 64)).reshape(3, -1).T.astype(np.uint8) * 4 + 2
    lab_grid = cv2.cvtColor(grid[:, None], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float32)
    lookup = np.empty(len(grid), np.uint8)
    # Chunking keeps the RGB lookup independent of available RAM.
    for start in range(0, len(grid), 8192):
        distances = ((lab_grid[start:start+8192, None] - centers[None]) ** 2).sum(2)
        lookup[start:start+8192] = distances.argmin(axis=1)
    colors = cv2.cvtColor(np.rint(centers).astype(np.uint8)[None], cv2.COLOR_LAB2BGR)[0]
    return dict(centers=centers, lut=lookup, bgr=colors)


def components(labels: np.ndarray, colors: list[dict], min_area: int = 30,
               limit: int = 90) -> list[dict]:
    """One palette cluster per candidate; material groups need not be connected."""
    candidates = []
    for color in colors:
        for palette_id in color['palette_ids']:
            binary = (labels == palette_id).astype(np.uint8)
            count, regions, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
            for region in range(1, count):
                x, y, width, height, area = map(int, stats[region])
                if area >= min_area:
                    candidates.append(dict(color=color['name'], palette_ids=[palette_id],
                        box=[x, y, x+width, y+height], area=area,
                        center=centers[region].tolist(),
                        mask=regions[y:y+height, x:x+width] == region))
    candidates.sort(key=lambda row: -row['area'])
    candidates = candidates[:limit]
    for number, row in enumerate(candidates):
        row['id'] = f'R{number:04d}'
    return candidates
