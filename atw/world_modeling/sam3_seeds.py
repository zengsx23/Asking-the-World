"""Single-instance seed selection with explicit identity correction handoff."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any
import numpy as np
from atw.shared.protocols import JsonValue
from atw.perception.sam3_agent import NoVisiblePrimaryFrame
from atw.perception.sam3_tool import EmptySeedSelection, SEED_SELECTION_WORKERS, _generate_candidates, _extract_frame, _write_json
from atw.world_modeling.sam3_rendering import render_candidate_sheet as _render_candidate_sheet


def select_primary_masks(
    *, agent: Any, runtime: Any, video_path: Path, request_dir: Path,
    call_prefix: str, objects: list[dict[str, Any]], timings: dict[str, JsonValue],
    contact_sheet_path: Path, sampled_frames: list[int], seed_recovery=None, candidate_filter=None,
    candidate_exclusion=None, fixed_identity=False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run bounded per-object retries; serialize GPU inference, overlap VLM work."""
    if fixed_identity and seed_recovery is not None:
        raise ValueError('Locked identities cannot use inventory-changing seed recovery')
    seeds, seed_manifest, seed_arrays = {}, [], {}
    omissions = []
    selection_wait_seconds = 0.0
    pipeline_started = time.perf_counter()
    object_times = [dict(sam3=0.0, selection=0.0, expansion=0.0,
                         visibility=0.0, reselection=0.0) for _ in objects]

    def select_object(target, elapsed):
        target_id = str(target["target_id"])
        frame_index = int(target["primary_seed_frame"])
        image_path = request_dir / "seed_frames" / f"{target_id}_primary_{frame_index:04d}.jpg"
        evidence_dir = request_dir / "seed_selection" / target_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        cache, history = {}, []
        original_phrases = list(target["short_noun_phrases"])
        report = dict(target_id=target_id, frame_index=frame_index,
                      original_primary_frame=frame_index, rounds=history)
        last_sheet = None

        def run_round(name, phrases, rank=None):
            nonlocal last_sheet
            round_index = len(history)
            row = dict(round=name, frame_index=frame_index, phrases=phrases, rank=rank)
            history.append(row)
            # Do not count waiting for another object's GPU inference as compute time.
            with runtime.inference_lock:
                started = time.perf_counter()
                candidates = _generate_candidates(
                    runtime.image_processor, image_path, phrases,
                    prediction_cache=cache, rank=rank,
                )
                row["sam3_seconds"] = time.perf_counter() - started
            elapsed["sam3"] += row["sam3_seconds"]
            row["candidate_count"] = len(candidates)
            row["candidates"] = [{k: v for k, v in c.items() if k != "mask"} for c in candidates]
            np.savez_compressed(evidence_dir / f"{name}_masks.npz", **{
                c["candidate_id"]: c["mask"] for c in candidates})
            if candidate_exclusion is not None:
                accepted, rejected = [], []
                for candidate in candidates:
                    conflicts = candidate_exclusion(target_id, frame_index, candidate['mask'])
                    if conflicts:
                        rejected.append(dict(candidate_id=candidate['candidate_id'], conflicting_object_ids=conflicts))
                    else:
                        accepted.append(candidate)
                candidates = accepted
                row['duplicate_rejections'] = rejected
            if candidate_filter is not None:
                accepted, rejected = [], []
                for candidate in candidates:
                    conflicts = candidate_filter(target_id, frame_index, candidate["mask"])
                    if len(conflicts) >= 2:
                        rejected.append(dict(candidate_id=candidate["candidate_id"], contained=conflicts))
                    else:
                        accepted.append(candidate)
                candidates = accepted
                row["containment_rejections"] = rejected
            decision = {"candidate_id": None, "reason": "no candidates after filtering"}
            last_sheet = None
            if candidates:
                suffix = "" if round_index == 0 else f"_round_{round_index:02d}"
                last_sheet = request_dir / "candidate_sheets" / f"{target_id}_primary{suffix}.png"
                _render_candidate_sheet(image_path, candidates, last_sheet)
                row["candidate_sheet"] = str(last_sheet)
                _write_json(evidence_dir / "rounds.json", report)
                started = time.perf_counter()
                try:
                    decision = agent.choose_seed_mask(
                        call_prefix=call_prefix if round_index == 0 else f"{call_prefix}_round_{round_index:02d}",
                        target=target, frame_index=frame_index, image_path=image_path,
                        candidate_sheet_path=last_sheet,
                        candidate_ids={item["candidate_id"] for item in candidates},
                    )
                finally:
                    row["vlm_selection_seconds"] = time.perf_counter() - started
                    elapsed["selection"] += row["vlm_selection_seconds"]
            row["selection"] = decision
            _write_json(evidence_dir / "rounds.json", report)
            candidate = next((c for c in candidates if c["candidate_id"] == decision["candidate_id"]), None)
            if candidate is None:
                return None
            return candidate, dict(target=target, candidate_count=len(candidates),
                selection={**decision, "phrase": candidate["phrase"], "sam_score": candidate["score"]},
                selected_round=name, rounds=history)

        def reselect_frame(prefix, feedback, frames):
            nonlocal frame_index, image_path, cache, original_phrases
            started = time.perf_counter()
            try:
                reselection = agent.reselect_primary_frame(
                    call_prefix=call_prefix + prefix, target=target,
                    contact_sheet_path=contact_sheet_path, sampled_frames=frames,
                    failed_frame=frame_index, feedback=feedback,
                )
            finally:
                elapsed["reselection"] += time.perf_counter() - started
            report[prefix + "reselection"] = reselection
            if not fixed_identity:
                target.update(reselection["identity"])
            original_phrases = list(target["short_noun_phrases"])
            frame_index = reselection["primary_seed_frame"]
            target["primary_seed_frame"] = frame_index
            target["primary_seed_reason"] = reselection["reason"]
            report["frame_index"] = frame_index
            image_path = request_dir / "seed_frames" / f"{target_id}_primary_{frame_index:04d}.jpg"
            cache = {}
            _extract_frame(video_path, frame_index, image_path)
            return run_round(prefix + "reselected", original_phrases)

        def expand_candidates(prefix=""):
            started = time.perf_counter()
            try:
                expansion = agent.expand_seed_prompts(
                    call_prefix=call_prefix + prefix, target=target, frame_index=frame_index,
                    image_path=image_path, tried_phrases=original_phrases,
                    feedback=history[-1], candidate_sheet_path=last_sheet,
                )
            finally:
                elapsed["expansion"] += time.perf_counter() - started
            report[prefix + "expansion"] = expansion
            new_phrases = expansion["short_noun_phrases"]
            selected = run_round(prefix + "expanded", new_phrases)
            if selected is not None:
                return selected
            all_phrases = list(dict.fromkeys(original_phrases + new_phrases))
            for rank in (1, 2):
                selected = run_round(f"{prefix}rank{rank}", all_phrases, rank=rank)
                if selected is not None:
                    return selected
            raise EmptySeedSelection(f"Gemini selected no valid Primary mask for {target_id} after phrase expansion and ranks 1–2")

        def recover(exc):
            # Called only at a semantic empty-selection exit, never on tool/provider errors.
            failure = dict(error=str(exc), visibility=report.get("visibility"), attempts=[
                {key: row[key] for key in ("round", "frame_index", "phrases", "rank")} |
                dict(candidate_ids=[c["candidate_id"] for c in row["candidates"]],
                     selection={k: row["selection"][k] for k in ("candidate_id", "reason") if k in row["selection"]})
                for row in history])
            media = [("failed_frame", image_path), ("indexed_frames", contact_sheet_path)]
            if last_sheet is not None:
                media.append(("failed_candidates", last_sheet))
            decision = seed_recovery(video=video_path, target=dict(target), failure=failure, media=media)
            report["recovery"] = dict(decision=decision)
            action = decision["action"]
            if action in {"reselect_frame", "retry_masks"}:
                keys = ("discriminative_description", "primary_seed_reason")
                changed = any(target.get(k) != decision[k] for k in keys)
                target.update({k: decision[k] for k in keys})
                try:
                    if action == "reselect_frame":
                        tried = {row["frame_index"] for row in history}
                        frames = [frame for frame in sampled_frames if frame not in tried]
                        if not frames:
                            raise NoVisiblePrimaryFrame("All printed Primary frames have already been tried")
                        selected = reselect_frame("recovery_", failure | dict(diagnosis=decision["reason"]), frames)
                    elif changed:
                        last = history[-1]
                        selected = run_round("recovery_candidates", last["phrases"], last["rank"])
                    else:
                        selected = None
                    if selected is None:
                        # A new frame or identity context permits one existing candidate chain.
                        # With unchanged inputs, only run the expansion if it was never reached.
                        if action == "reselect_frame" or changed or "expansion" not in report:
                            selected = expand_candidates("recovery_")
                    if selected is not None:
                        report["recovery"]["outcome"] = "recovered"
                        return selected
                except (NoVisiblePrimaryFrame, EmptySeedSelection) as retry_error:
                    report["recovery"]["error"] = str(retry_error)
            outcome = "omitted_inventory_error" if action == "omit" else "omitted_unresolved"
            report["recovery"]["outcome"] = outcome
            return None, dict(target_id=target_id, outcome=outcome, reason=decision["reason"],
                              evidence=f"seed_selection/{target_id}/rounds.json")

        try:
            _extract_frame(video_path, frame_index, image_path)
            selected = run_round("initial", original_phrases)
            if selected is not None:
                return selected
            started = time.perf_counter()
            try:
                visibility = agent.judge_seed_frame_visibility(
                    call_prefix=call_prefix, target=target, frame_index=frame_index,
                    image_path=image_path, feedback=history[0],
                    candidate_sheet_path=last_sheet,
                )
            finally:
                elapsed["visibility"] += time.perf_counter() - started
            report["visibility"] = visibility
            _write_json(evidence_dir / "rounds.json", report)
            if visibility["entity_visible"]:
                update = visibility["identity"]
                changed = not fixed_identity and any(target[key] != value for key, value in update.items())
                if not fixed_identity:
                    target.update(update)
                original_phrases = list(target["short_noun_phrases"])
                target["primary_seed_reason"] = visibility["reason"]
                if changed:
                    selected = run_round("identity_corrected", original_phrases)
                    if selected is not None:
                        return selected
            if not visibility["entity_visible"]:
                selected = reselect_frame("", {"initial_selection": history[0]["selection"], "visibility": visibility},
                                          sampled_frames)
                if selected is not None:
                    return selected
            # This is deliberately not a loop: no second visibility/reselection call.
            return expand_candidates()
        except (NoVisiblePrimaryFrame, EmptySeedSelection) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            if seed_recovery is None:
                return None, dict(target_id=target_id, outcome="no_mask_selected",
                    reason=report["error"], evidence=f"seed_selection/{target_id}/rounds.json")
            return recover(exc)
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            report["timing_seconds"] = dict(elapsed)
            _write_json(evidence_dir / "rounds.json", report)

    try:
        with ThreadPoolExecutor(max_workers=SEED_SELECTION_WORKERS) as executor:
            pending = [executor.submit(copy_context().run, select_object, target, elapsed)
                       for target, elapsed in zip(objects, object_times)]
            # Completion may vary; preserve inventory order in the tracker handoff.
            for target, future in zip(objects, pending):
                started = time.perf_counter()
                try:
                    candidate, manifest = future.result()
                finally:
                    selection_wait_seconds += time.perf_counter() - started
                if candidate is None:
                    omissions.append(manifest)
                    continue
                target_id = str(target["target_id"])
                frame_index = int(target["primary_seed_frame"])
                seeds[target_id] = {"frame_index": frame_index, "mask": candidate["mask"]}
                seed_arrays[f"{target_id}__{frame_index:04d}"] = candidate["mask"].astype(np.uint8)
                seed_manifest.append(manifest)
    finally:
        if omissions:
            _write_json(request_dir / "seed_omissions.json", omissions)
        timings.update({
            "sam3_candidate_seconds": sum(t["sam3"] for t in object_times),
            "gemini_mask_selection_seconds": sum(t["selection"] for t in object_times),
            "gemini_seed_prompt_expansion_seconds": sum(t["expansion"] for t in object_times),
            "gemini_seed_visibility_seconds": sum(t["visibility"] for t in object_times),
            "gemini_primary_reselection_seconds": sum(t["reselection"] for t in object_times),
            "gemini_mask_selection_wait_seconds": selection_wait_seconds,
            "primary_seed_pipeline_wall_seconds": time.perf_counter() - pipeline_started,
            "gemini_mask_selection_workers": SEED_SELECTION_WORKERS,
        })
    return seeds, seed_manifest, seed_arrays
