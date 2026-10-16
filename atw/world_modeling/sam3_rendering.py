"""Candidate views with unchanged object pixels and 70% context brightness."""
from pathlib import Path
from typing import Any
import math
import cv2
import numpy as np
from atw.perception.sam3_tool import _labeled_image_tile

def render_mask_focus(image, mask):
    """Keep mask pixels unchanged; dim surrounding context by 30 percent."""
    source = np.asarray(image)
    candidate = np.asarray(mask, dtype=bool)
    if source.ndim != 3 or source.shape[2] != 3 or candidate.shape != source.shape[:2]:
        raise ValueError('Expected an HxWx3 image and matching mask')
    if not candidate.any():
        raise ValueError('Candidate mask must be nonempty')
    focused = np.rint(source.astype(np.float32) * 0.70).astype(np.uint8)
    focused[candidate] = source[candidate]
    contours, _ = cv2.findContours(candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pixels = np.zeros(candidate.shape, dtype=np.uint8)
    cv2.drawContours(contour_pixels, contours, -1, 255, 5, cv2.LINE_AA)
    focused[(contour_pixels > 0) & ~candidate] = (0, 0, 255)
    return focused


def render_candidate_sheet(
    image_path: Path,
    candidates: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(image_path)
    height, width = image.shape[:2]
    tile_width = 640
    raw = _labeled_image_tile(
        image,
        label="RAW",
        tile_width=tile_width,
        header_height=58,
        maximum_font_scale=1.0,
        text_thickness=3,
    )
    tiles = [raw]
    for candidate in candidates:
        overlay = render_mask_focus(image, candidate["mask"])
        label = str(candidate["candidate_id"])
        tiles.append(
            _labeled_image_tile(
                overlay,
                label=label,
                tile_width=tile_width,
                header_height=58,
                maximum_font_scale=0.9,
                text_thickness=2,
            )
        )
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(output_path)




def write_named_overlay(*, video_path, masks_by_frame, objects, output_path):
    """Render numbered masks and complete inventory names outside the source image."""
    import colorsys

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    panel_width = 420
    output_width = width + panel_width
    output_height = max(height + 44, 80 + len(objects) * 32)
    output_width += output_width % 2
    output_height += output_height % 2
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (output_width, output_height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Unable to create debug video: {output_path}")
    colors = [
        tuple(round(c * 255) for c in
              colorsys.hsv_to_rgb(i / max(1, len(objects)), .7, 1)[::-1])
        for i in range(len(objects))
    ]
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            canvas = np.full((output_height, output_width, 3), (28, 23, 19), np.uint8)
            frame_masks = masks_by_frame.get(frame_index, {})
            for i, obj in enumerate(objects):
                color = colors[i]
                mask = frame_masks.get(obj["track_id"])
                if mask is not None and np.any(mask):
                    mask = np.asarray(mask, dtype=bool)
                    if mask.shape != (height, width):
                        raise ValueError("Debug masks must use original video coordinates")
                    frame[mask] = (frame[mask] * .7 + np.asarray(color) * .3).astype(np.uint8)
                    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, color, 2)
                    ys, xs = np.where(mask)
                    x = max(0, min(width - 32, int(xs.mean()) - 10))
                    y = max(18, min(height - 4, int(ys.mean())))
                    for ink, thickness in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
                        cv2.putText(frame, f"{i+1:02d}", (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, ink, thickness, cv2.LINE_AA)
                label = f"{i+1:02d}  {obj['name']}"
                measured = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .65, 1)[0][0]
                scale = .65 * min(1, (panel_width - 28) / max(1, measured))
                cv2.putText(canvas, label, (width + 14, 84 + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, scale,
                            color if mask is not None else (130, 130, 130), 1, cv2.LINE_AA)
            canvas[44:44+height, :width] = frame
            cv2.putText(canvas, f"SAM3 segmentation | frame {frame_index:03d}", (12, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Object inventory", (width + 14, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(canvas)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
