"""Primary-only Gemini + SAM3 shared-state video segmentation tool."""


from __future__ import annotations


import gc

import json

import math

import os

import time

from collections.abc import Mapping

from pathlib import Path

from typing import Any

import cv2

import numpy as np

import torch

from PIL import Image

from atw.shared.protocols import JsonValue



class EmptySeedSelection(RuntimeError):
    """The bounded candidate search produced no selected mask."""


SEED_SELECTION_WORKERS = 4


SEED_DUPLICATE_IOU = 0.9




def _configure_tracker_non_overlap(tracker: Any) -> dict[str, bool]:
    """Apply the project's explicit visible-instance tracking configuration."""

    required_attributes = (
        "non_overlap_masks_for_mem_enc",
        "non_overlap_masks_for_output",
    )
    missing = [name for name in required_attributes if not hasattr(tracker, name)]
    if missing:
        raise TypeError(
            "SAM3 tracker is missing required non-overlap settings: "
            + ", ".join(missing)
        )
    tracker.non_overlap_masks_for_mem_enc = True
    tracker.non_overlap_masks_for_output = True
    return {
        "non_overlap_masks_for_mem_enc": True,
        "non_overlap_masks_for_output": True,
    }


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("cannot compare masks with different shapes")
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def _write_mask_archive(
    *,
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    output_path: Path,
) -> dict[str, dict[str, JsonValue]]:
    arrays: dict[str, np.ndarray] = {}
    index: dict[str, dict[str, JsonValue]] = {}
    for frame_index, frame_masks in masks_by_frame.items():
        for track_id, mask in frame_masks.items():
            key = f"{track_id}__frame_{frame_index:05d}"
            flattened = np.ascontiguousarray(mask.astype(bool)).reshape(-1)
            arrays[key] = np.packbits(flattened)
            index[key] = {
                "track_id": track_id,
                "frame_index": frame_index,
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "num_bits": int(flattened.size),
            }
    temporary = output_path.with_name(f".{output_path.name}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return index


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _video_metadata(video_path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        capture.release()
    if frame_count < 1:
        raise ValueError(f"video contains no frames: {video_path}")
    return frame_count, width, height, fps


def _sampled_frame_indices(frame_count: int, maximum: int = 12) -> list[int]:
    return sorted(set(np.linspace(0, frame_count - 1, maximum, dtype=int).tolist()))


def _labeled_image_tile(
    image: np.ndarray,
    *,
    label: str,
    tile_width: int,
    header_height: int,
    maximum_font_scale: float,
    text_thickness: int,
) -> np.ndarray:
    """Place a label above an aspect-preserved image without covering pixels."""

    source = np.asarray(image, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("labeled tile image must have shape (height, width, 3)")
    if tile_width < 1 or header_height < 1:
        raise ValueError("labeled tile dimensions must be positive")
    image_height = max(1, int(round(source.shape[0] * tile_width / source.shape[1])))
    resized = cv2.resize(
        source,
        (tile_width, image_height),
        interpolation=(
            cv2.INTER_AREA if tile_width < source.shape[1] else cv2.INTER_LINEAR
        ),
    )
    header = np.zeros((header_height, tile_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    available_width = max(1, tile_width - 28)
    measured_width = cv2.getTextSize(
        label,
        font,
        maximum_font_scale,
        text_thickness,
    )[0][0]
    font_scale = maximum_font_scale * min(
        1.0,
        available_width / max(measured_width, 1),
    )
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        text_thickness,
    )
    del text_width
    origin_y = max(
        text_height,
        (header_height + text_height - baseline) // 2,
    )
    cv2.putText(
        header,
        label,
        (14, origin_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return np.vstack((header, resized))


def _write_indexed_contact_sheet(
    video_path: Path,
    frame_indices: list[int],
    output_path: Path,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")
    tiles: list[np.ndarray] = []
    tile_width = 480
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"unable to read sampled frame {frame_index}")
            tiles.append(
                _labeled_image_tile(
                    frame,
                    label=f"FRAME {frame_index:04d}",
                    tile_width=tile_width,
                    header_height=52,
                    maximum_font_scale=1.05,
                    text_thickness=3,
                )
            )
    finally:
        capture.release()
    columns = 3
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(output_path)


def _extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"unable to read frame {frame_index} from {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise OSError(output_path)


def _extract_tracker_frames(video_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output_path = output_dir / f"{frame_index:05d}.jpg"
            if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(output_path)
            frame_index += 1
    finally:
        capture.release()
    if frame_index == 0:
        raise ValueError(f"no frames extracted from {video_path}")
    return output_dir


def _predict_phrase_candidates(
    processor: Any,
    image: Image.Image,
    phrase: str,
) -> dict[str, list[dict[str, Any]]]:
    """Decode filtered queries and the first two nonempty queries from one prediction."""

    width, height = image.size
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        state["backbone_out"].update(
            processor.model.backbone.forward_text([phrase], device=processor.device)
        )
        outputs = processor.model.forward_grounding(
            backbone_out=state["backbone_out"], find_input=processor.find_stage,
            geometric_prompt=processor.model._get_dummy_prompt(), find_target=None,
        )
        scores = (outputs["pred_logits"].sigmoid()
                  * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).reshape(-1)
        logits = outputs["pred_masks"].flatten(0, 1)
        boxes = outputs["pred_boxes"].reshape(-1, 4)
        decoded = {}

        def decode(index):
            if index not in decoded:
                mask = torch.nn.functional.interpolate(
                    logits[index][None, None], (height, width),
                    mode="bilinear", align_corners=False,
                ).sigmoid()[0, 0].gt(0.5).cpu().numpy()
                if not mask.any():
                    decoded[index] = None
                    return None
                cx, cy, w, h = boxes[index].detach().float().cpu().tolist()
                decoded[index] = dict(phrase=phrase, score=float(scores[index]),
                    box=[(cx - w / 2) * width, (cy - h / 2) * height,
                         (cx + w / 2) * width, (cy + h / 2) * height], mask=mask)
            return decoded[index]

        # Preserve the processor's query order and strict score comparison.
        filtered = [decode(i) for i in (scores > processor.confidence_threshold).nonzero().flatten().tolist()]
        filtered = [item for item in filtered if item is not None]
        ranked = []
        for index in scores.argsort(descending=True).tolist():
            item = decode(index)
            if item is not None:
                ranked.append(item)
            if len(ranked) == 2:
                break
    return {"filtered": filtered, "ranked": ranked}


def _generate_candidates(
    processor: Any,
    image_path: Path,
    phrases: list[str],
    *, prediction_cache: dict[str, Any] | None = None, rank: int | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate filtered candidates or one fixed rank per phrase from cached predictions."""

    if rank not in (None, 1, 2):
        raise ValueError("candidate rank must be 1 or 2")
    cache = prediction_cache if prediction_cache is not None else {}
    candidates: list[dict[str, Any]] = []
    image = Image.open(image_path).convert("RGB")
    for phrase in dict.fromkeys(phrases):
        if phrase not in cache:
            cache[phrase] = _predict_phrase_candidates(processor, image, phrase)
        pool = cache[phrase]["filtered"] if rank is None else cache[phrase]["ranked"][rank-1:rank]
        for item in pool:
            if any(_mask_iou(item["mask"], old["mask"]) >= 0.94 for old in candidates):
                continue
            candidates.append({**item, "candidate_id": f"C{len(candidates):02d}"})
    return candidates


def _render_candidate_sheet(
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
        overlay = _render_mask_focus(image, candidate["mask"])
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


def _render_mask_focus(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Preserve candidate RGB while dimming context and outlining it in red."""

    source = np.asarray(image)
    candidate = np.asarray(mask, dtype=bool)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("candidate image must have shape HxWx3")
    if candidate.shape != source.shape[:2]:
        raise ValueError("candidate mask must match the image size")
    if not candidate.any():
        raise ValueError("candidate mask must be nonempty")

    focused = np.rint(source.astype(np.float32) * 0.35).astype(np.uint8)
    focused[candidate] = source[candidate]

    contours, _ = cv2.findContours(
        candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_pixels = np.zeros(candidate.shape, dtype=np.uint8)
    cv2.drawContours(contour_pixels, contours, -1, 255, 5, cv2.LINE_AA)
    outside_contour = (contour_pixels > 0) & ~candidate
    focused[outside_contour] = (0, 0, 255)
    return focused


def _run_shared_tracker(
    *,
    tracker: Any,
    tracker_resource: Path,
    objects: list[dict[str, Any]],
    seeds: Mapping[str, Mapping[str, Any]],
    frame_count: int,
    resolve_overlap: bool = True,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Track seed-synchronized lifecycle cohorts in independent shared states."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not objects:
        raise ValueError("shared tracking requires at least one object")
    target_ids = [str(target["target_id"]) for target in objects]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("shared tracking target IDs must be unique")
    target_to_obj_id = {
        target_id: index + 1 for index, target_id in enumerate(target_ids)
    }
    obj_id_to_target = {value: key for key, value in target_to_obj_id.items()}
    normalized_seeds: dict[str, dict[str, Any]] = {}
    for target_id in target_ids:
        seed = seeds.get(target_id)
        if seed is None:
            raise RuntimeError(f"no Primary seed for {target_id}")
        frame_index = int(seed["frame_index"])
        if not 0 <= frame_index < frame_count:
            raise ValueError(
                f"Primary seed frame {frame_index} for {target_id} is outside "
                f"a {frame_count}-frame video"
            )
        mask = np.asarray(seed["mask"]).astype(bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError(f"Primary seed mask for {target_id} must be nonempty 2D")
        normalized_seeds[target_id] = {
            "frame_index": frame_index,
            "mask": mask,
        }

    anchors = sorted(
        {int(seed["frame_index"]) for seed in normalized_seeds.values()}
    )

    cohort_outputs: dict[str, dict[int, tuple[float, np.ndarray, str]]] = {
        target_id: {} for target_id in target_ids
    }
    cohorts: list[dict[str, Any]] = []
    cohort_tracking_started = time.perf_counter()
    for anchor in anchors:
        cohort_targets = [
            target_id
            for target_id in target_ids
            if int(normalized_seeds[target_id]["frame_index"]) == anchor
        ]
        state = tracker.init_state(
            video_path=str(tracker_resource), offload_video_to_cpu=True
        )
        cohort_record: dict[str, Any] = {
            "anchor_frame": anchor,
            "targets": cohort_targets,
        }
        try:
            for target_id in cohort_targets:
                tracker.add_new_mask(
                    inference_state=state,
                    frame_idx=anchor,
                    obj_id=target_to_obj_id[target_id],
                    mask=torch.from_numpy(normalized_seeds[target_id]["mask"]),
                )
            tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
            for direction, reverse in (("forward", False), ("reverse", True)):
                started = time.perf_counter()
                for frame, obj_ids, _, logits, object_scores in tracker.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=anchor,
                    max_frame_num_to_track=None,
                    reverse=reverse,
                    tqdm_disable=True,
                    propagate_preflight=False,
                ):
                    if reverse and int(frame) >= anchor:
                        continue
                    for obj_index, raw_obj_id in enumerate(obj_ids):
                        target_id = obj_id_to_target[int(raw_obj_id)]
                        score = float(
                            torch.sigmoid(object_scores[obj_index]).item()
                        )
                        mask = (
                            (logits[obj_index, 0] > 0.0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        if mask.any():
                            cohort_outputs[target_id][int(frame)] = (
                                score,
                                mask,
                                direction,
                            )
                cohort_record[f"{direction}_elapsed_seconds"] = (
                    time.perf_counter() - started
                )
        finally:
            del state
            gc.collect()
        cohorts.append(cohort_record)

    direction_seconds: dict[str, float] = {
        "cohort_tracking": time.perf_counter() - cohort_tracking_started
    }

    started = time.perf_counter()
    frame_outputs: dict[str, dict[int, tuple[float, np.ndarray, str]]] = {
        target_id: {} for target_id in target_ids
    }
    overlap_pixels_removed = 0
    fully_occluded_candidates = 0
    for frame in range(frame_count):
        candidates: list[tuple[bool, float, int, str, np.ndarray, str]] = []
        for target_id in target_ids:
            anchor = int(normalized_seeds[target_id]["frame_index"])
            if frame == anchor:
                candidates.append(
                    (
                        True,
                        1.0,
                        target_to_obj_id[target_id],
                        target_id,
                        normalized_seeds[target_id]["mask"],
                        "primary_seed",
                    )
                )
                continue
            item = cohort_outputs[target_id].get(frame)
            if item is not None:
                candidates.append(
                    (
                        False,
                        item[0],
                        target_to_obj_id[target_id],
                        target_id,
                        item[1],
                        item[2],
                    )
                )
        if not candidates:
            continue
        occupied = np.zeros_like(candidates[0][4], dtype=bool)
        for _, score, _, target_id, mask, source in sorted(
            candidates,
            key=lambda item: (-int(item[0]), -item[1], item[2]),
        ):
            if mask.shape != occupied.shape:
                raise ValueError("lifecycle candidates have inconsistent mask shapes")
            resolved = np.logical_and(mask, ~occupied) if resolve_overlap else mask
            overlap_pixels_removed += int(mask.sum() - resolved.sum())
            if not resolved.any():
                fully_occluded_candidates += 1
                continue
            frame_outputs[target_id][frame] = (score, resolved, source)
            occupied = np.logical_or(occupied, resolved)
    direction_seconds["global_overlap_resolution"] = time.perf_counter() - started
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # The tracker returns masks in original-video coordinates. Keep their pixels
    # intact for export; downstream consumers resize to their own working grids.
    tracks = {
        target_id: {
            frame: item[1]
            for frame, item in sorted(outputs.items())
        }
        for target_id, outputs in frame_outputs.items()
    }
    manifest = {
        target_id: {
            "obj_id": target_to_obj_id[target_id],
            "anchor_frames": [int(seeds[target_id]["frame_index"])],
            "active_frames": len(tracks[target_id]),
            "mean_object_score": float(
                np.mean([item[0] for item in frame_outputs[target_id].values()])
            )
            if frame_outputs[target_id]
            else None,
            "forward_selected_frames": sum(
                item[2] == "forward" for item in frame_outputs[target_id].values()
            ),
            "reverse_selected_frames": sum(
                item[2] == "reverse" for item in frame_outputs[target_id].values()
            ),
            "primary_seed_selected_frames": sum(
                item[2] == "primary_seed"
                for item in frame_outputs[target_id].values()
            ),
        }
        for target_id in tracks
    }
    manifest["_lifecycle_segmentation"] = {
        "method": "seed_cohort_bidirectional_shared_tracking",
        "unique_anchor_frames": anchors,
        "cohorts": cohorts,
        "overlap_pixels_removed": overlap_pixels_removed,
        "fully_occluded_candidates": fully_occluded_candidates,
    }
    manifest["_timing_seconds"] = direction_seconds
    return tracks, manifest


def _filter_duplicate_seeds(
    *, objects: list[dict[str, Any]], seeds: Mapping[str, Mapping[str, Any]],
    seed_manifest: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Drop same-frame duplicate identities before initializing tracker memory."""
    scores = {
        str(item["target"]["target_id"]): float(item["selection"]["sam_score"])
        for item in seed_manifest
    }
    target_ids = [str(item["target_id"]) for item in objects]
    # Stable sorting resolves equal scores in the original inventory order.
    ranked = sorted(target_ids, key=lambda target_id: -scores[target_id])
    kept: list[str] = []
    dropped = []
    for target_id in ranked:
        seed = seeds[target_id]
        for kept_id in kept:
            other = seeds[kept_id]
            if int(seed["frame_index"]) != int(other["frame_index"]):
                continue
            overlap = _mask_iou(seed["mask"], other["mask"])
            if overlap >= SEED_DUPLICATE_IOU:
                dropped.append({
                    "code": "duplicate_primary_seed_dropped",
                    "dropped_target_id": target_id,
                    "kept_target_id": kept_id,
                    "frame_index": int(seed["frame_index"]),
                    "mask_iou": overlap,
                    "dropped_sam_score": scores[target_id],
                    "kept_sam_score": scores[kept_id],
                    "reason": "Same-frame seed masks overlap above the duplicate threshold; retain higher SAM3 score, then inventory order.",
                })
                break
        else:
            kept.append(target_id)
    retained = set(kept)
    active_objects = [item for item in objects if str(item["target_id"]) in retained]
    active_ids = [str(item["target_id"]) for item in active_objects]
    return active_objects, {key: seeds[key] for key in active_ids}, {
        "method": "same_frame_seed_mask_iou",
        "iou_threshold": SEED_DUPLICATE_IOU,
        "input_seed_count": len(target_ids),
        "retained_seed_count": len(active_ids),
        "retained_target_ids": active_ids,
        "dropped": dropped,
    }
