"""Focused original-color mask video for qualitative rope-tension comparison."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


def _overlap_area(left, right):
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _label_position(mask, text, occupied, union, previous=None):
    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
    scale = max(.45, min(.8, width / 1600))
    thickness = max(1, round(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad, gap = 7, 12
    box_width, box_height = text_width + pad * 2, text_height + baseline + pad * 2
    center_y = (y0 + y1) // 2
    candidates = [
        (x0, y0 - gap - box_height), (x1 - box_width, y0 - gap - box_height),
        (x0, y1 + gap), (x1 - box_width, y1 + gap),
        (x0 - gap - box_width, center_y - box_height // 2),
        (x1 + gap, center_y - box_height // 2),
    ]
    best = None
    for x, y in candidates:
        x = max(2, min(width - box_width - 2, x))
        y = max(2, min(height - box_height - 2, y))
        box = (x, y, x + box_width, y + box_height)
        selected_overlap = int(union[y:y + box_height, x:x + box_width].sum())
        label_overlap = sum(_overlap_area(box, other) for other in occupied)
        cx, cy = x + box_width // 2, y + box_height // 2
        distance = abs(cx - (x0 + x1) // 2) + abs(cy - center_y)
        jitter = 0 if previous is None else abs(cx - previous[0]) + abs(cy - previous[1])
        score = selected_overlap * 10000 + label_overlap * 100 + distance + jitter * .25
        if best is None or score < best[0]:
            best = (score, box, scale, thickness, baseline)
    return best[1:], (x0, y0, x1, y1)


def _draw_label(image, mask, text, occupied, union, previous=None):
    placement, target = _label_position(mask, text, occupied, union, previous)
    box, scale, thickness, baseline = placement
    x0, y0, x1, y1 = box
    tx0, ty0, tx1, ty1 = target
    label_center = ((x0 + x1) // 2, (y0 + y1) // 2)
    anchor = (min(max(label_center[0], tx0), tx1), min(max(label_center[1], ty0), ty1))
    edge = (min(max(anchor[0], x0), x1), min(max(anchor[1], y0), y1))
    cv2.line(image, anchor, edge, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (245, 245, 245), 1)
    cv2.putText(image, text, (x0 + 7, y1 - baseline - 7),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    occupied.append(box)
    return label_center


def render_rope_tension_video(backend, labels, output_dir: Path):
    """Keep two rope masks in original color and darken all other pixels by 35%."""
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "rope_tension.mp4"
    intermediate = output_dir / "encoding_intermediate.mp4"
    capture = cv2.VideoCapture(str(backend.video))
    if not capture.isOpened():
        raise ValueError("Original video is unavailable")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot open rope-tension video encoder")
    visible_frames = {row["object_id"]: 0 for row in labels}
    previous = {}
    frame = 0
    encoded = False
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            masks = []
            for row in labels:
                try:
                    mask = backend._mask(row["object_id"], frame)
                except (KeyError, ValueError, OSError):
                    mask = np.zeros((height, width), dtype=bool)
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                if mask.any():
                    visible_frames[row["object_id"]] += 1
                masks.append(mask)
            union = np.logical_or.reduce(masks)
            focused = np.rint(raw.astype(np.float32) * .65).astype(np.uint8)
            focused[union] = raw[union]
            occupied = []
            for row, mask in zip(labels, masks):
                if mask.any():
                    previous[row["object_id"]] = _draw_label(
                        focused, mask, row["label"], occupied, union,
                        previous.get(row["object_id"]))
            writer.write(focused)
            frame += 1
        writer.release()
        missing = [name for name, count in visible_frames.items() if count == 0]
        if missing:
            raise ValueError("No mask frames are available for: " + ", ".join(missing))
        try:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
                "-i", str(intermediate), "-c:v", "libx264", "-threads", "2", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError("Rope-tension video encoding failed") from error
        encoded = True
    finally:
        capture.release()
        writer.release()
        intermediate.unlink(missing_ok=True)
        if not encoded:
            target.unlink(missing_ok=True)
    manifest = dict(source_kind="processed_original_video", property_name="tension",
        object_labels=labels, background_brightness=0.65,
        selected_masks_original_color=True, visible_frames=visible_frames,
        frame_count=frame, video=str(target))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
