"""Fixed-camera support/wall segmentation, background plane fitting, and gravity."""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image

from atw.perception.track4world_tool import SceneGeometryResult


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def segmentation_frames(frame_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("surface segmentation needs video frames")
    return sorted({0, frame_count // 2, frame_count - 1})


def _top_two_surface_candidates(processor, state, prompts, output_size_wh):
    """Rank unfiltered queries across this frame's expansion phrases, without changing the processor."""
    import torch.nn.functional as F

    pool = []
    for phrase in prompts:
        processor.reset_all_prompts(state)
        state["backbone_out"].update(
            processor.model.backbone.forward_text([phrase], device=processor.device))
        outputs = processor.model.forward_grounding(
            backbone_out=state["backbone_out"], find_input=processor.find_stage,
            geometric_prompt=processor.model._get_dummy_prompt(), find_target=None)
        scores = (outputs["pred_logits"].sigmoid()
                  * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).reshape(-1)
        logits = outputs["pred_masks"].detach().cpu()
        logits = logits.reshape(-1, *logits.shape[-2:])
        pool.extend(dict(score=float(score), logits=mask, prompts=[phrase])
                    for score, mask in zip(scores.detach().float().cpu(), logits))
    selected = []
    for item in sorted(pool, key=lambda row: row["score"], reverse=True):
        # Preserve SAM3's original-resolution binarization before display resizing.
        mask = F.interpolate(item["logits"][None, None].float(),
            (state["original_height"], state["original_width"]),
            mode="bilinear", align_corners=False).sigmoid()[0, 0].numpy() > .5
        mask = cv2.resize(mask.astype(np.uint8), output_size_wh,
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any() or any(np.count_nonzero(mask & c["mask"])
                / np.count_nonzero(mask | c["mask"]) > .97 for c in selected):
            continue
        selected.append(dict(mask=mask, score=item["score"], prompts=item["prompts"],
                             candidate_source="low_confidence_top2_fallback"))
        if len(selected) == 2:
            break
    return selected


def _generate_surface_candidates(
    *, video_path: Path, frames: list[int], output_size_wh: tuple[int, int],
    output_dir: Path, device: str, scene_surface_type: str, prompts: list[str], round_index: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate one retained round, reusing the process-local SAM3 model."""
    import torch
    from huggingface_hub import hf_hub_download
    from atw.perception.sam3_runtime import task_sam3_runtime_registry
    from scripts.prepare_sam3 import REPO_ID, REVISION

    checkpoint = hf_hub_download(REPO_ID, "sam3.pt", revision=REVISION, local_files_only=True)
    requested_device = torch.device(device)
    if requested_device.type != "cuda":
        raise ValueError("SAM3 background segmentation requires a CUDA device")
    runtime, cache_hit = task_sam3_runtime_registry().acquire(checkpoint, device_index=requested_device.index)
    candidates: dict[str, dict[str, Any]] = {}
    metadata = []
    outcomes = []
    sheets = []
    warning_records = []
    width, height = output_size_wh
    cap = cv2.VideoCapture(str(video_path))
    try:
        with runtime.inference_lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            processor = runtime.image_processor
            for frame in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"cannot read background frame {frame} from {video_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                display = cv2.resize(rgb, output_size_wh)
                state = processor.set_image(Image.fromarray(rgb))
                local = []
                for phrase in prompts:
                    processor.reset_all_prompts(state)
                    result = processor.set_text_prompt(state=state, prompt=phrase)
                    masks = result["masks"].detach().cpu().numpy()
                    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
                    outcomes.append(dict(frame=frame, prompt=phrase, returned_count=len(scores),
                                         maximum_returned_score=float(scores.max()) if len(scores) else None))
                    for raw, score in zip(masks, scores):
                        mask = cv2.resize(raw.squeeze().astype(np.uint8), output_size_wh,
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                        if not mask.any():
                            continue
                        duplicate = next((c for c in local if
                            np.count_nonzero(mask & c["mask"]) / np.count_nonzero(mask | c["mask"]) > .97), None)
                        if duplicate is not None:
                            duplicate["prompts"].append(phrase)
                            continue
                        local.append(dict(id=f"R{round_index}_F{frame:03d}_C{len(local):02d}", frame=frame,
                                          mask=mask, score=float(score), prompts=[phrase]))
                if round_index > 0 and prompts and not local:
                    local = _top_two_surface_candidates(processor, state, prompts, output_size_wh)
                    for index, candidate in enumerate(local):
                        candidate.update(id=f"R{round_index}_F{frame:03d}_C{index:02d}", frame=frame)
                    warning = dict(code="background_surface_top2_fallback", round=round_index,
                        frame=frame, prompts=prompts, confidence_threshold=float(processor.confidence_threshold),
                        reason="no_candidates_after_expansion_filtering",
                        candidate_ids=[c["id"] for c in local], scores=[c["score"] for c in local])
                    warning_records.append(warning)
                    _write_json(output_dir / "warnings.json", warning_records)
                    logging.getLogger(__name__).warning("%s", json.dumps(warning, ensure_ascii=False))
                panels = []
                for c in local:
                    candidates[c["id"]] = c
                    np.save(output_dir / f"{c['id']}.npy", c["mask"])
                    metadata.append({k: v for k, v in c.items() if k != "mask"})
                    canvas = display.copy()
                    canvas[~c["mask"]] = (canvas[~c["mask"]] * .25).astype(np.uint8)
                    contours, _ = cv2.findContours(c["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, (255, 60, 60), 2)
                    panel = np.zeros((height + 48, width, 3), np.uint8)
                    panel[48:] = canvas
                    cv2.putText(panel, f"{c['id']} | {','.join(c['prompts'])}",
                                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                    panels.append(panel)
                if panels:
                    if len(panels) % 2:
                        panels.append(np.zeros_like(panels[0]))
                    sheet = np.concatenate([np.concatenate(panels[i:i+2], axis=1)
                                            for i in range(0, len(panels), 2)])
                    path = output_dir / f"candidates_{frame:04d}.png"
                    Image.fromarray(sheet).save(path)
                    sheets.append(path)
                del state
    finally:
        cap.release()
    _write_json(output_dir / "candidates.json", metadata)
    _write_json(output_dir / "sam3_metadata.json", dict(model=REPO_ID, revision=REVISION,
        frames=frames, prompts=prompts, scene_surface_type=scene_surface_type,
        runtime_cache_hit=cache_hit, confidence_threshold=.35))
    _write_json(output_dir / "prompt_outcomes.json", outcomes)
    _write_json(output_dir / "warnings.json", warning_records)
    return candidates, metadata, sheets, outcomes, warning_records


def select_background_masks(
    *, video_path: Path, frame_count: int, output_size_wh: tuple[int, int],
    output_dir: Path, project_root: Path, device: str, scene_surface_type: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Accept two frames; otherwise expand prompts for missing frames at most twice."""
    from atw.perception.sam3_agent import GeminiSam3Agent

    if scene_surface_type not in {"support_plane", "wall_plane"}:
        raise ValueError("background segmentation requires the known support/wall label")
    output_dir.mkdir(parents=True)
    frames = segmentation_frames(frame_count)
    originals = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"cannot read background frame {frame} from {video_path}")
            original = output_dir / f"original_{frame:04d}.png"
            Image.fromarray(cv2.cvtColor(cv2.resize(bgr, output_size_wh), cv2.COLOR_BGR2RGB)).save(original)
            originals[frame] = original
    finally:
        cap.release()
    agent = GeminiSam3Agent(project_root)
    started = time.perf_counter()
    timings = {}
    candidates = {}
    metadata, outcomes, history, tried = [], [], [], []
    warning_records = []
    accepted = {f: {"frame": f, "candidate_ids": []} for f in frames}
    try:
        stage_started = time.perf_counter()
        plan = agent.plan_background_surface(surface_type=scene_surface_type, frames=frames,
                                             image_paths=list(originals.values()))
        timings["vlm_prompt_planning_seconds"] = time.perf_counter() - stage_started
        _write_json(output_dir / "prompt_plan.json", plan)
        prompts = plan["short_noun_phrases"]
        for round_index in range(3):
            missing = [f for f in frames if not accepted[f]["candidate_ids"]]
            round_dir = output_dir / f"round_{round_index:02d}"
            round_dir.mkdir()
            tried.extend(prompts)
            _write_json(round_dir / "prompts.json", {"frames": missing, "prompts": prompts})
            current, rows, sheets, current_outcomes, current_warnings = ({}, [], [], [], [])
            if prompts:
                stage_started = time.perf_counter()
                current, rows, sheets, current_outcomes, current_warnings = _generate_surface_candidates(
                    video_path=video_path, frames=missing, output_size_wh=output_size_wh,
                    output_dir=round_dir, device=device, scene_surface_type=scene_surface_type,
                    prompts=prompts, round_index=round_index)
                timings[f"round_{round_index:02d}_sam3_candidates_seconds"] = time.perf_counter() - stage_started
            candidates.update(current)
            metadata.extend(rows)
            outcomes.extend(current_outcomes)
            warning_records.extend(current_warnings)
            _write_json(output_dir / "warnings.json", warning_records)
            if current:
                stage_started = time.perf_counter()
                selection = agent.choose_background_surface_masks(plan=plan, frames=missing,
                    candidates=rows, image_paths=[originals[f] for f in missing],
                    candidate_sheet_paths=sheets, round_index=round_index)
                timings[f"round_{round_index:02d}_vlm_selection_seconds"] = time.perf_counter() - stage_started
                selected_consensus(selection, current, missing)
            else:
                selection = {"frames": [{"frame": f, "candidate_ids": []} for f in missing],
                             "notes": "no_sam3_candidates"}
            for row in selection["frames"]:
                if row["candidate_ids"]:
                    accepted[row["frame"]] = row
            _write_json(round_dir / "selection.json", selection)
            usable = sum(bool(row["candidate_ids"]) for row in accepted.values())
            history.append(dict(round=round_index, frames=missing, prompts=prompts,
                                accepted_frame_count=usable, warnings=current_warnings))
            if usable >= 2 or round_index == 2:
                break
            missing = [f for f in frames if not accepted[f]["candidate_ids"]]
            stage_started = time.perf_counter()
            expansion = agent.expand_background_surface_prompts(
                plan=plan, tried_prompts=tried, missing_frames=missing,
                feedback={"selection": list(accepted.values()), "last_round_selection": selection,
                          "prompt_outcomes": outcomes}, image_paths=[originals[f] for f in missing],
                candidate_sheet_paths=sheets, round_index=round_index+1)
            timings[f"round_{round_index+1:02d}_vlm_expansion_seconds"] = time.perf_counter() - stage_started
            _write_json(output_dir / f"prompt_expansion_{round_index+1:02d}.json", expansion)
            prompts = expansion["short_noun_phrases"]
            if not prompts:
                break
        usable = sum(bool(row["candidate_ids"]) for row in accepted.values())
        final = dict(frames=list(accepted.values()), status="completed" if usable >= 2 else "insufficient",
                     reason="two_usable_frames" if usable >= 2 else "fewer_than_two_usable_frames",
                     required_frame_count=2, accepted_frame_count=usable, rounds=history,
                     warnings=warning_records)
        _write_json(output_dir / "vlm_selection.json", final)
        _write_json(output_dir / "candidates.json", metadata)
        return final, candidates
    finally:
        timings["total_seconds"] = time.perf_counter() - started
        _write_json(output_dir / "timings.json", timings)
        _write_json(output_dir / "vlm_usage.json", agent.usage_report())
        _write_json(output_dir / "vlm_trace.json", agent.trace)



def selected_consensus(
    selection: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]],
    frames: list[int],
) -> np.ndarray | None:
    rows = selection["frames"]
    if sorted(row["frame"] for row in rows) != frames:
        raise ValueError("background selection must contain each requested frame exactly once")
    masks = []
    for row in rows:
        ids = row["candidate_ids"]
        if not isinstance(ids, list) or any(i not in candidates or candidates[i]["frame"] != row["frame"] for i in ids):
            raise ValueError("background selection refers to an unknown or wrong-frame candidate")
        if ids:
            mask = np.logical_or.reduce([candidates[i]["mask"] for i in ids])
            if not mask.any():
                raise ValueError("selected background candidate is empty")
            masks.append(mask)
    if len(masks) < 2:
        return None
    # Strict majority among frames with an accepted mask; empty selections abstain.
    return np.sum(masks, axis=0) >= len(masks) // 2 + 1


def fit_background_plane(points: np.ndarray, *, seed: int = 42) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 100 or not np.isfinite(points).all():
        raise ValueError("background plane needs at least 100 finite points")
    rng = np.random.default_rng(seed)
    search = points[rng.choice(len(points), min(5000, len(points)), replace=False)]
    threshold = float(np.median(points[:, 2]) * .005)
    best = None
    for _ in range(1000):
        a, b, c = search[rng.choice(len(search), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        offset = float(normal @ a)
        count = int((np.abs(search @ normal-offset) < threshold).sum())
        if best is None or count > best[0]:
            best = (count, normal, offset)
    if best is None:
        raise ValueError("background points do not span a plane")
    _, normal, offset = best
    for _ in range(8):
        inside = np.abs(points @ normal-offset) < threshold
        selected = points[inside]
        if len(selected) < 100:
            raise ValueError("too few background plane inliers")
        center = selected.mean(axis=0)
        eigenvalues, vectors = np.linalg.eigh((selected-center).T @ (selected-center))
        if eigenvalues[1] <= 1e-10:
            raise ValueError("background points are collinear")
        normal = vectors[:, 0]
        offset = float(normal @ center)
    if normal[2] < 0:
        normal, offset = -normal, -offset
    residual = np.abs(points @ normal-offset)
    return dict(normal_camera=normal.tolist(), offset_camera=offset, threshold=threshold,
                point_count=len(points), rmse=float(np.sqrt(np.mean(residual**2))),
                p95=float(np.quantile(residual, .95)), inlier_fraction=float((residual <= threshold).mean()))


def wall_gravity_coordinates(normal: np.ndarray, offset: float, camera_to_world: np.ndarray) -> dict[str, Any]:
    normal = np.asarray(normal, dtype=float)
    gravity = np.array([0., normal[2], -normal[1]])
    if np.linalg.norm(gravity) < 1e-6:
        raise ValueError("wall normal parallel to camera X does not determine no-roll gravity")
    gravity /= np.linalg.norm(gravity)
    if gravity[1] < 0:
        gravity = -gravity
    rotation, translation = camera_to_world[:3, :3], camera_to_world[:3, 3]
    # The nonpenetration normal faces the camera/free-space side, unlike gravity.
    wall_normal = -(rotation @ normal)
    wall_offset = -float(offset) + float(wall_normal @ translation)
    down_world = rotation @ gravity
    return dict(gravity_down_base_camera=gravity.tolist(), gravity_down_world=down_world.tolist(),
                up_direction_base_camera=(-gravity).tolist(), up_direction_world=(-down_world).tolist(),
                wall_plane=dict(normal_world=wall_normal.tolist(), offset_world=wall_offset,
                                admissible_half_space="normal_world dot point_world >= offset_world"))


def support_gravity_coordinates(normal: np.ndarray, offset: float, camera_to_world: np.ndarray) -> dict[str, Any]:
    """A horizontal support plane's upward normal points toward the camera side."""
    normal = np.asarray(normal, dtype=float)
    # Plane fitting reports a unit normal. The camera is at the origin; choose
    # its half-space as free space, independently of the fitter's normal sign.
    if offset > 0:
        normal, offset = -normal, -float(offset)
    rotation, translation = camera_to_world[:3, :3], camera_to_world[:3, 3]
    up = rotation @ normal
    plane_offset = float(offset + up @ translation)
    return dict(gravity_down_base_camera=(-normal).tolist(), gravity_down_world=(-up).tolist(),
                up_direction_base_camera=normal.tolist(), up_direction_world=up.tolist(),
                support_plane=dict(normal_world=up.tolist(), offset_world=plane_offset,
                                   admissible_half_space="normal_world dot point_world >= offset_world"))


def estimate_background_surface_gravity(
    *, geometry_result: SceneGeometryResult, masks: Mapping[str, np.ndarray],
    video_path: Path, output_dir: Path, project_root: Path, scene_surface_type: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fit the classified interaction surface using predicted background geometry."""
    if scene_surface_type not in {"support_plane", "wall_plane"}:
        raise ValueError("background gravity requires a support or wall label")
    surface = "wall" if scene_surface_type == "wall_plane" else "support"
    summary = dict(method=f"sam3_vlm_background_{surface}", scene_surface_type=scene_surface_type,
                   assumptions=(["fixed_camera", "zero_camera_roll", "vertical_wall"]
                                if surface == "wall" else ["fixed_camera", "horizontal_support_surface", "camera_above_support"]),
                   reference_annotations_used=False)
    if not np.allclose(geometry_result.camera_poses, geometry_result.camera_poses[0], atol=1e-5):
        return dict(summary, status="skipped", reason="background_surface_requires_fixed_camera")
    count, height, width = geometry_result.rgb.shape[:3]
    frames = segmentation_frames(count)
    started = time.perf_counter()
    selection, candidates = select_background_masks(video_path=video_path, frame_count=count,
        output_size_wh=(width, height), output_dir=output_dir / "segmentation", project_root=project_root,
        device=device, scene_surface_type=scene_surface_type)
    segmentation_seconds = time.perf_counter() - started
    fit_started = time.perf_counter()
    union = np.logical_or.reduce(list(masks.values())) if masks else np.zeros((count, height, width), bool)
    sampled = np.unique(np.linspace(0, count-1, min(16, count)).round().astype(int))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    y, x = np.mgrid[:height, :width]
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(geometry_result.intrinsics[0]).T
    overlay = geometry_result.rgb[0].copy()
    consensus = selected_consensus(selection, candidates, frames)
    plane = dict(status="unavailable", reason="fewer_than_two_usable_frames")
    if consensus is not None:
        consensus = cv2.erode(consensus.astype(np.uint8), kernel).astype(bool)
        observations = []
        for frame in sampled:
            depth = geometry_result.camera_points[frame, ..., 2]
            keep = consensus & ~cv2.dilate(union[frame].astype(np.uint8), kernel).astype(bool)
            keep &= geometry_result.geometry_valid[frame] & np.isfinite(depth) & (depth > 0)
            observations.append(np.where(keep, depth, np.nan))
        observed = np.stack(observations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            depth = np.nanmedian(observed, axis=0)
        usable = np.isfinite(depth) & (np.isfinite(observed).sum(axis=0) >= min(3, len(sampled)))
        points = rays[usable] * depth[usable, None]
        np.savez_compressed(output_dir / f"{surface}_fit_inputs.npz", consensus=consensus,
                            usable=usable, camera_points=points.astype(np.float32), frames=sampled)
        try:
            result = fit_background_plane(points)
        except ValueError as exc:
            plane = dict(status="unavailable", reason=str(exc))
        else:
            plane = dict(status="completed", **result)
            overlay[usable] = (.65 * overlay[usable] + .35 * np.array([255, 90, 70])).astype(np.uint8)
    Image.fromarray(overlay).save(output_dir / "fit_regions.png")
    results = {surface: plane}
    _write_json(output_dir / "background_planes.json", results)
    summary.update(segmentation_frames=frames, geometry_frames=sampled.tolist(), background_planes=results)
    summary["timings"] = dict(segmentation_seconds=segmentation_seconds,
                             background_plane_seconds=time.perf_counter() - fit_started,
                             total_seconds=time.perf_counter() - started)
    if plane["status"] != "completed":
        return dict(summary, status="skipped", reason=f"no_usable_background_{surface}")
    try:
        coordinates = wall_gravity_coordinates if surface == "wall" else support_gravity_coordinates
        selected = coordinates(np.array(plane["normal_camera"]), plane["offset_camera"],
                               geometry_result.camera_poses[0])
    except ValueError as exc:
        return dict(summary, status="skipped", reason=str(exc))
    return dict(summary, status="completed", selected=selected)
