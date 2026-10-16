"""Fixed-camera, occlusion-aware rigid motion classification from SAM3 masks."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from atw.perception.tracking_mask_policy import classify_tracking_mask_frames

CONFIG = dict(boundary_radius_at_480=2, mismatch_fraction=.08,
              minimum_witness_pixels_at_480=12, comparable_fraction=.25,
              static_temporal_coverage=.7, minimum_eligible_frames=3)
POPCOUNT = np.array([i.bit_count() for i in range(256)], np.uint8)


def dilate(mask, radius):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def erode(mask, radius):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def eligible_frames(masks):
    policy = classify_tracking_mask_frames(masks)
    eligible = np.flatnonzero((policy.mask_pixel_counts > 0) & ~policy.border_suppressed & ~policy.occlusion_terminated)
    return eligible, policy


def classify_object_motion(masks, others, config=CONFIG):
    start = time.perf_counter()
    frames, policy = eligible_frames(masks)
    scale = max(masks.shape[1:]) / 480
    radius = max(1, round(config["boundary_radius_at_480"] * scale))
    minimum_pixels = max(3, round(config["minimum_witness_pixels_at_480"] * scale ** 2))
    result = dict(state="unknown", eligible_frames=frames.tolist(),
        termination_frame=int(policy.occlusion_termination_frame),
        border_suppressed_frames=np.flatnonzero(policy.border_suppressed).tolist(),
        post_termination_frames=np.flatnonzero(policy.occlusion_terminated).tolist(),
        empty_frames=np.flatnonzero(policy.mask_pixel_counts == 0).tolist(),
        radius=radius, config=config, frame_count=len(masks))
    if len(frames) < config["minimum_eligible_frames"]:
        result.update(reason="fewer than three eligible tracking frames", elapsed_seconds=time.perf_counter()-start,
                      max_pair=None, temporal_coverage=0., candidate_pairs=0, supported_pairs=0)
        return result, []
    union = masks[frames].any(axis=0)
    yy, xx = np.where(union)
    x0, x1 = max(0, int(xx.min())-radius-1), min(masks.shape[2], int(xx.max())+radius+2)
    y0, y1 = max(0, int(yy.min())-radius-1), min(masks.shape[1], int(yy.max())+radius+2)
    cores, backgrounds = [], []
    for f in frames:
        target = masks[f, y0:y1, x0:x1]
        other = others[f, y0:y1, x0:x1]
        # Other-object masks remain occluders even if their OWN tracking is inactive.
        occluded = dilate(other, radius)
        cores.append(erode(target, radius) & ~occluded)
        backgrounds.append(~dilate(target | other, radius))
    core_areas = np.array([c.sum() for c in cores])
    core_bits = np.stack([np.packbits(c) for c in cores])
    background_bits = np.stack([np.packbits(b) for b in backgrounds])
    candidates = np.zeros((len(frames), len(frames)), bool)
    temporal_support = np.zeros(len(frames), bool)
    rows = []
    for i, f in enumerate(frames[:-1]):
        intersections = POPCOUNT[core_bits[i] & core_bits[i+1:]].sum(axis=1)
        differences = POPCOUNT[(core_bits[i] & background_bits[i+1:]) | (background_bits[i] & core_bits[i+1:])].sum(axis=1)
        for k, (intersection, difference) in enumerate(zip(intersections, differences)):
            j = i+k+1
            total = int(intersection + difference)
            comparable = total / max(1, int(max(core_areas[i], core_areas[j])))
            mismatch = int(difference) / max(1, total)
            adequate = total >= minimum_pixels and comparable >= config["comparable_fraction"]
            candidate = adequate and difference >= minimum_pixels and mismatch >= config["mismatch_fraction"]
            candidates[i, j] = candidates[j, i] = candidate
            if adequate and frames[j]-f >= max(1, .2 * (frames[-1]-frames[0])):
                temporal_support[i] = temporal_support[j] = True
            rows.append(dict(frame_a=int(f), frame_b=int(frames[j]), mismatch=float(mismatch),
                comparable_fraction=float(comparable), witness_pixels=int(difference),
                comparable_pixels=total, iou=float(intersection/total) if total else None,
                candidate=bool(candidate), adequate=bool(adequate)))
    # An isolated bad frame forms a star, not supported changes on both sides.
    degree = candidates.sum(axis=1)
    supported = candidates & (degree[:, None] >= 2) & (degree[None, :] >= 2)
    coverage = float(temporal_support.mean())
    max_pair = max(rows, key=lambda p: (p["adequate"], p["mismatch"], p["witness_pixels"]))
    if supported.any():
        state, reason = "moving", "unoccluded silhouette contradiction supported by multiple frames on both sides"
    elif candidates.any():
        state, reason = "unknown", "silhouette contradiction lacks multi-frame support; could be an isolated segmentation error or brief motion"
    elif coverage >= config["static_temporal_coverage"]:
        state, reason = "static_compatible", "adequate cross-time observations with no reliable unoccluded silhouette contradiction; rotation invisible to silhouettes remains unresolved"
    else:
        state, reason = "unknown", "insufficient comparable cross-time visible area"
    result.update(state=state, reason=reason, max_pair=max_pair, temporal_coverage=coverage,
        candidate_pairs=int(candidates.sum()//2), supported_pairs=int(supported.sum()//2),
        elapsed_seconds=time.perf_counter()-start)
    return result, rows


def _save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def classify_rigid_motion(*, tracks, rigid_target_ids, frame_count, mask_shape, output_dir: Path):
    """Classify rigid targets using all original objects as possible occluders.

    Tracks map original frame indices to native-resolution visible masks.
    No video, reconstruction, model, credentials or VLM is required.
    """
    started = time.perf_counter()
    output_dir.mkdir(parents=True)
    masks = {}
    for name, track in tracks.items():
        sequence = np.zeros((frame_count, *mask_shape), dtype=bool)
        for frame, mask in track.items():
            sequence[frame] = mask
        masks[name] = sequence
    report = dict(module="rigid_motion", method="fixed_camera_occlusion_aware_masks",
        fixed_camera=True, objects={}, config=CONFIG,
        policy="all eligible frame pairs; original masks of all other objects remain possible occluders; static_compatible uses static tracking; moving and unknown use dynamic tracking",
        frame_count=frame_count, mask_shape=list(mask_shape))
    for name in rigid_target_ids:
        other_masks = [mask for other, mask in masks.items() if other != name]
        others = np.logical_or.reduce(other_masks) if other_masks else np.zeros_like(masks[name])
        decision, pairs = classify_object_motion(masks[name], others)
        decision.update(target_id=name, source="mask_consistency",
            tracking_branch="static" if decision["state"] == "static_compatible" else "dynamic")
        object_dir = output_dir / name
        object_dir.mkdir()
        _save_json(object_dir / "decision.json", decision)
        if pairs:
            np.savez_compressed(object_dir / "pairs.npz",
                **{key: np.array([pair[key] for pair in pairs], dtype=float) for key in pairs[0]})
        report["objects"][name] = decision
    states = {item["state"] for item in report["objects"].values()}
    if not states:
        final_state = "no_rigid"
    elif "unknown" in states:
        final_state = "unknown"
    elif len(states) > 1:
        final_state = "mixed"
    elif states == {"static_compatible"}:
        final_state = "all_static"
    else:
        final_state = "all_moving"
    report.update(final_state=final_state, status="completed",
        elapsed_seconds=time.perf_counter()-started,
        classifier_seconds=sum(item["elapsed_seconds"] for item in report["objects"].values()),
        vlm_calls=0, api_cost_usd=0)
    _save_json(output_dir / "decision.json", report)
    return report
