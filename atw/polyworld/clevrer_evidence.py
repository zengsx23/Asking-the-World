"""Rigid scene event evidence. Never converts timing or paths into answer labels."""
from collections import deque
from atw.polyworld.evidence_actions import query_requirement

ENTRY_GUIDANCE = (
    "For rigid scene questions, entry/entrance means the object's presence in frame. "
    "These are the same condition, including when the object is on screen at the interval's start. "
    "The query returns one entry record for each object in frame during the requested interval; "
    "the recorded time is its first retained in-frame sample in that interval. "
    "Judge entry/entrance and presence options using the same controlled object-removal evidence "
    "and original-target event comparison. Honor the event-identity rule. "
    "Being in frame alone does not establish causal responsibility. "
    "An empty incomplete result cannot establish absence or non-responsibility.")


GUIDANCE = """Choose experiments from the literal question. Explanations need baseline evidence (saved fit or replay) and relevant controlled edits; predictions need continuation; counterfactuals replay the requested edit from the modeled beginning. Property/count may use retained observations.
Each object's first usable 3D state is its appearance in this modeled world. It is absent beforehand, not subject to an early-coverage uncertainty. Baseline and edited collision logs use the same rule.
For a collision-event cause, inspect event order and time-increasing shared-object paths. Earlier timing or a path is supporting evidence, not a causal verdict. Removing a body removes all its interactions, not one chosen collision; event suppression is unsupported. State when no faithful intervention is available.
First locate the question's historical target event in the baseline, including its participants and time. A later extrapolated collision of the same pair is not a substitute for a missing historical target. For candidate collision events, inspect target/candidate times and intervening contact paths; an event strictly after the target cannot cause it. Equal/near times alone do not resolve order or causality. For presence/entrance choices, object removal can test the object's enabling role; do not reject an established enabling condition merely because a separate choice describes the immediate collision. Object-level controls may support event analysis but cannot isolate a specific collision from the removed body's other interactions.
Compare the specific target event across worlds. Use the five-frame event identity rule: matching participants within five frames retain the original event; a larger shift is a different event. Do not treat a later different event as preservation of the target.
For each collision option, use query_event_outcome with the exact two participants, or include contact queries in simulate/query_world. An already observed target event in the requested world needs no extension. An absent target is automatically checked up to THREE TIMES the original video duration (total horizon, not three additional durations); absence there counts as not happening under this scene's evaluation convention. Original-window and extrapolated events remain separately labeled. Set end_seconds only for a time limit explicitly requested by the question. A raw simulate event list is not a completed negative option query. Missing identities, capped/incomplete logs or unsupported queries require correcting or completing the query, never a false verdict. If required evidence cannot be obtained, request original-video reasoning for this question; other questions remain independent."""


GUIDANCE += "\n" + ENTRY_GUIDANCE


def tools():
    return [dict(name="query_event_outcome", arguments=dict(rollout_id="existing ID", object_ids="REQUIRED exactly two known participant IDs",
        start_seconds="default 0; use observed end for future-only events", end_seconds="omit for automatic 3x-video horizon; set only for a literal time bound"),
        description="Check one collision in the requested world. Return immediately if present; otherwise replay the SAME edit to 3x video duration before judging absence. Reuses extended rollouts. Missing IDs or incomplete logs return needs_query with next_action, never false; if unresolved, request per-question video reasoning."),
        dict(name="query_event_timeline", arguments=dict(rollout_id="existing ID", kind="collision|exit|entry, default collision",
        object_ids="optional existing IDs", start_seconds="default 0", end_seconds="default rollout end"),
        description="List individual events with stable IDs, times and evidence coverage. " + ENTRY_GUIDANCE +
            " Exit queries use projected center crossings of the image boundary."),
        dict(name="query_event_path", arguments=dict(rollout_id="existing ID", candidate_event_id="timeline event ID", target_event_id="timeline event ID"),
        description="Return a strictly time-increasing shared-object event path, or no recorded path with coverage limits. This is structural evidence, not a responsibility verdict.")]


def event_outcome(backend, rollout_id, object_ids, start_seconds=0., end_seconds=None):
    """Scene-local bounded absence convention, independent of answer labels."""
    import math
    from atw.polyworld.swr_measurements import event_verdict
    rigid = backend._rigid
    original = backend.rollouts[rollout_id]
    ids = object_ids
    if not isinstance(ids, list) or len(ids) != 2 or any(not isinstance(i, str) for i in ids) or len(set(ids)) != 2:
        return query_requirement("A collision query needs two distinct resolved object IDs",
                    "Resolve both participants from identity assets and retry query_event_outcome on the same rollout.") | dict(event_status="needs_query", event_occurs=None,
                    rollout_id=rollout_id, reason="A collision query needs two distinct resolved object IDs")
    if not set(ids) <= set(rigid.scene.object_ids):
        return query_requirement("Requested participant is absent from the modeled object inventory",
                    "Check the participant binding against identity assets; correct the ID and retry. If the required object has no model, request request_video_fallback for this question.") | dict(event_status="needs_query", event_occurs=None,
                    rollout_id=rollout_id, reason="Requested participant is absent from the modeled object inventory")
    start = float(start_seconds)
    video_duration = backend.frame_count / backend.fps
    horizon = 3 * video_duration if end_seconds is None else float(end_seconds)
    if not all(math.isfinite(v) for v in (start, horizon)) or not 0 <= start <= horizon:
        raise ValueError("Event query needs a finite nonnegative ordered interval")
    def read(rid, end):
        return event_verdict(rigid.query_events(rid, ids, start, end))
    result = read(rollout_id, min(original['end_seconds'], horizon)) if start <= original['end_seconds'] else None
    if not (result and result.get('event_occurs') is True) and original['end_seconds'] < horizon:
        same_edit = sorted(original.get('remove_objects', []))
        existing = next((rid for rid, row in backend.rollouts.items()
                         if sorted(row.get('remove_objects', [])) == same_edit and row['end_seconds'] >= horizon), None)
        if existing is None:
            # All rigid replays use the same fitted parameters and modeled origin.
            extension = horizon - (rigid.scene.frame_count - 1) / rigid.scene.fps
            rigid.output, rigid.rollouts = backend.output, backend.rollouts
            extended = rigid.simulate(max(0., extension), same_edit)
            existing = extended['rollout_id']
        result = read(existing, horizon)
    assert result is not None
    events = result.get('events', [])
    actual_rollout = backend.rollouts[result['rollout_id']]
    occurrence = result.get('event_occurs')
    summary = dict(rollout_id=result['rollout_id'], remove_objects=actual_rollout.get('remove_objects', []),
        queried_pair=ids, interval_seconds=result['interval_seconds'], event_occurs=occurrence,
        event_times_seconds=[e['event_time_s'] for e in events],
        statement=("Queried collision OCCURRED in this world." if occurrence is True else
                   "Queried collision DID NOT OCCUR within this covered interval." if occurrence is False else
                   "Complete or correct the event query; no event verdict is available. Follow next_action."))
    return dict(evidence_summary=summary) | result | dict(requested_rollout_id=rollout_id,
        continuation_of_rollout=rollout_id if result['rollout_id'] != rollout_id else None,
        experiment_role="readout", result_kind="events", status=result.get('event_status', 'needs_query'),
        horizon_policy=dict(video_duration_seconds=video_duration, total_horizon_seconds=horizon,
            basis="explicit_question_interval" if end_seconds is not None else "three_video_durations",
            horizon_reached=result['interval_seconds'][1] >= horizon,
            stopped_on_observed_event=result.get('event_occurs') is True),
        observed_window_event_times_seconds=[e['event_time_s'] for e in events if e['event_time_s'] <= backend.end],
        extrapolated_event_times_seconds=[e['event_time_s'] for e in events if e['event_time_s'] > backend.end],
        interpretation="A present event establishes occurrence. With complete coverage, absence at the stated horizon counts as nonoccurrence under the scene convention; this is not an infinite-time physical theorem. Missing evidence requires a corrected/completed query, then per-question video reasoning if still unresolved.")


def timeline(backend, rollout_id, kind="collision", object_ids=None, start_seconds=0., end_seconds=None):
    import numpy as np
    rigid = getattr(backend, "_rigid", None) or backend
    rollout = backend.rollouts[rollout_id]
    end = rollout["end_seconds"] if end_seconds is None else float(end_seconds)
    start = float(start_seconds)
    if not np.isfinite([start, end]).all() or not 0 <= start <= end <= rollout["end_seconds"]:
        raise ValueError("Query interval must lie inside this rollout")
    selected = set(object_ids or [])
    if not selected <= set(rigid.scene.object_ids):
        raise ValueError("Unrecognized object IDs")
    if kind == "collision":
        result = rigid.query_events(rollout_id, object_ids, start, end)
        events = [dict(e, event_id=f"{rollout_id}:collision:{i}", kind="collision")
                  for i, e in enumerate(rollout["events"])
                  if selected <= set(e["object_ids"]) and start <= e["event_time_s"] <= end]
    elif kind in {"entry", "exit"}:
        if not hasattr(backend, "paths"):
            return dict(status="unsupported", reason="Camera projection assets unavailable", rollout_id=rollout_id)
        from atw.polyworld.experiment_views import image_projection
        project = image_projection(backend, rollout)
        events = []
        limitations = []
        removed = set(rollout.get("remove_objects", []))
        names = [name for name in rigid.scene.object_ids if not selected or name in selected]
        for name in names:
            if name in removed:
                continue
            rows = rollout["predicted_trajectories"].get(name, [])
            if not rows:
                if kind == "entry":limitations.append(f"No retained trajectory for {name}.")
                continue
            pixels = project([r["point_cloud_center_world"] for r in rows])
            inside = np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < backend.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < backend.height)
            if kind == "entry":
                frames = np.asarray([r["frame_index"] for r in rows])
                times = frames / rigid.scene.fps
                window = (times >= start) & (times <= end)
                indices = np.flatnonzero(inside & window)[:1]
                # The modeled body activates at its first retained state and remains active.
                first = max(int(frames[0]), int(np.ceil(start * rigid.scene.fps - 1e-6)))
                last = int(np.floor(end * rigid.scene.fps + 1e-6))
                if (not np.array_equal(frames[window], np.arange(first, last + 1))
                        or not np.isfinite(pixels[window]).all()):
                    limitations.append(f"Incomplete projected entry coverage for {name}.")
            else:
                indices = np.flatnonzero(inside[:-1] & ~inside[1:]) + 1
            for index in indices:
                t = rows[index]["frame_index"] / rigid.scene.fps
                if start <= t <= end:
                    events.append(dict(event_id=f"{rollout_id}:{kind}:{name}:{rows[index]['frame_index']}", kind=kind,
                        object_ids=[name], event_time_s=t, frame_float=t * rigid.scene.fps,
                        image_xy_px=pixels[index].tolist()))
        result = dict(source_kind="projected_simulation_centers", rollout_id=rollout_id,
            event_log_complete=kind == "entry" and not limitations, coverage_limitations=limitations,
            interpretation=ENTRY_GUIDANCE if kind == "entry" else
                "Exit records are center crossings of the image rectangle, not observed silhouette exit or physical disappearance.")
    else:
        raise ValueError("kind must be collision, entry or exit")
    return result | dict(events=sorted(events, key=lambda e:e["event_time_s"]), interval_seconds=[start, end],
                         rollout_id=rollout_id, experiment_role="readout", result_kind="events")


def event(backend, rollout_id, event_id):
    parts = event_id.split(":")
    if len(parts) < 3 or parts[0] != rollout_id:
        raise ValueError("Use an event ID from this rollout's timeline")
    result = timeline(backend, rollout_id, kind=parts[1])
    match = next((e for e in result.get("events", []) if e["event_id"] == event_id), None)
    if match is None and parts[1] == "entry" and len(parts) >= 4:
        name = ":".join(parts[2:-1])
        try:
            frame = int(parts[-1])
        except ValueError:
            raise ValueError("Use an entry event ID returned by the timeline") from None
        sample = next((r for r in backend.rollouts[rollout_id]["predicted_trajectories"].get(name, [])
                       if r["frame_index"] == frame), None)
        if sample is not None:
            rigid = getattr(backend, "_rigid", None) or backend
            time_s = sample["frame_index"] / rigid.scene.fps
            result = timeline(backend, rollout_id, kind="entry", object_ids=[name],
                              start_seconds=time_s, end_seconds=time_s)
            match = next((e for e in result.get("events", []) if e["event_id"] == event_id), None)
    if match is None:
        raise ValueError("Event ID not found")
    return match


def event_path(backend, rollout_id, candidate_event_id, target_event_id):
    candidate, target = event(backend, rollout_id, candidate_event_id), event(backend, rollout_id, target_event_id)
    result = timeline(backend, rollout_id)
    candidates = sorted([e for e in result["events"] if candidate["event_time_s"] < e["event_time_s"] < target["event_time_s"]] + [target], key=lambda e:e["event_time_s"])
    queue, visited, path = deque([(candidate, [candidate])]), {candidate_event_id}, None
    while queue and path is None:
        left, prefix = queue.popleft()
        for right in candidates:
            if (right["event_id"] not in visited and left["event_time_s"] < right["event_time_s"]
                    and set(left["object_ids"]) & set(right["object_ids"])):
                visited.add(right["event_id"])
                if right["event_id"] == target_event_id:
                    path = prefix + [right]
                    break
                queue.append((right, prefix + [right]))
    return dict(rollout_id=rollout_id, candidate=candidate, target=target, path=path,
        coverage_limitations=result.get("coverage_limitations", []), event_capacity_reached=result.get("event_capacity_reached", False),
        interpretation="A path is possible interaction transmission, not proof of causality. No recorded path is not proof of independence when observations/model events are incomplete.",
        experiment_role="readout", result_kind="events")


def attach_event_identity(backend, tool, arguments, result):
    """Add first-event identity evidence to explanatory rigid readouts."""
    if tool not in {"simulate", "query_world", "query_events", "query_event_outcome", "query_event_timeline"}:
        return result
    rigid = getattr(backend, "_rigid", None) or backend
    baselines = [(rid, row) for rid, row in backend.rollouts.items()
                 if row.get("remove_objects") == []]
    baseline = baselines[0] if baselines else None
    rows = result.get("measurements", {}).get("results", result.get("results", [result]))
    for row in rows:
        rid = row.get("rollout_id", result.get("rollout_id"))
        rollout = backend.rollouts.get(rid, {})
        ids = row.get("object_ids", arguments.get("object_ids"))
        kind = arguments.get("kind", "collision") if tool == "query_event_timeline" else "collision"
        if (not rollout.get("remove_objects") or not ids
                or len(set(ids)) != (2 if kind == "collision" else 1)
                or not set(ids) <= set(rigid.scene.object_ids)
                or not ("events" in row or "event_occurs" in row)):
            continue
        comparison = dict(same_event_frame_tolerance=5, same_event_with_base=None)
        row["target_comparison"] = comparison
        if baseline is None:
            comparison.update(status="needs_baseline")
            continue
        base_id, base = baseline
        # Explanations concern the historical target, never a later extrapolation.
        base_end = min(base["end_seconds"], getattr(backend, "end", base["end_seconds"]))
        base_events = timeline(backend, base_id, kind, ids, 0., base_end)
        interval = row.get("interval_seconds", [0., rollout["end_seconds"]])
        edited_events = timeline(backend, rid, kind, ids, *interval)
        def complete(readout):
            return (readout.get("event_log_complete") is True
                    and not readout.get("event_capacity_reached")
                    and not readout.get("coverage_limitations")
                    and readout.get("status") not in {"unsupported", "unavailable"})
        base_rows, edited_rows = base_events.get("events", []), edited_events.get("events", [])
        comparison.update(base_rollout_id=base_id, baseline_interval_seconds=[0., base_end],
            baseline_complete=complete(base_events), edited_complete=complete(edited_events))
        if not complete(base_events) or not complete(edited_events):
            comparison.update(status="unknown", event_identity_note="Incomplete event coverage; event identity is unknown.")
            continue
        first_base = base_rows[0] if base_rows else None
        first_edit = edited_rows[0] if edited_rows else None
        comparison.update(base_first_frame=round(first_base["event_time_s"] * rigid.scene.fps, 6) if first_base else None,
            edited_first_frame=round(first_edit["event_time_s"] * rigid.scene.fps, 6) if first_edit else None)
        if first_base is None:
            comparison.update(status="missing_baseline_target", event_identity_note="No original matching event found; responsibility cannot be established.")
        elif first_edit is None:
            comparison.update(status="compared", same_event_with_base=False,
                event_identity_note="The original event has no matching event in the covered edited interval.")
        else:
            delta = round(comparison["edited_first_frame"] - comparison["base_first_frame"], 6)
            same = abs(delta) <= 5
            comparison.update(status="compared", event_frame_delta=delta, same_event_with_base=same,
                event_identity_note=("Edited event matches the original event timing (within five frames)." if same else
                    "Edited world still has this event, but timing differs by more than five frames, so it is not the same event."))
    return result
