"""Read-only assets and fixed-parameter experiments for a planar rigid world.

This backend contains no benchmark question rules. Observation windows are not
physical lifetimes: experiment bodies remain active after their last tracked pose.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from atw.perception.rigid_motion import classify_object_motion
from atw.polyworld.experiment_views import labeled_observation
from atw.polyworld.swr_sysid import BatchWarpEventEngine, _predicted_poses_world, make_parameter_layout


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mask_visibility(index, masks, track_id, fps, termination_frame):
    """QA entry and visibility within the upstream-retained mask segment."""
    frames = sorted({info["frame_index"] for key, info in index.items()
                     if info["track_id"] == track_id
                     and (termination_frame < 0 or info["frame_index"] < termination_frame)
                     and np.unpackbits(masks[key])[:info["num_bits"]].any()})
    intervals = []
    for frame in frames:
        if intervals and frame == intervals[-1][1] + 1:
            intervals[-1][1] = frame
        else:
            intervals.append([frame, frame])
    return dict(source_kind="mask_observation", visible_frame_count=len(frames),
                entry=(dict(frame=frames[0], seconds=frames[0]/fps,
                            already_present_at_video_start=frames[0] == 0) if frames else None),
                intervals=[dict(first_frame=a, last_frame=b, start_seconds=a/fps, end_seconds=b/fps)
                           for a, b in intervals],
                interpretation="Mask observation: entry is the first nonempty mask in the retained segment; frame 0 means already visible at video start. Masks from the upstream confirmed-occlusion termination onward are excluded. Short gaps do not create new mask entries. Border pauses do not remove mask visibility.")


def translation_measurement(rows):
    """Local descriptive measurements only; no motion threshold or state correction."""
    if len(rows) < 2:
        return dict(status="insufficient_temporal_samples", sample_count=len(rows),
                    interpretation="No velocity can be established; missing data is not stationarity.")
    times = np.array([r["time_s"] for r in rows])
    xy = np.array([r["plane_xy"] for r in rows])
    centered = times - times.mean()
    velocity = centered @ (xy - xy.mean(axis=0)) / (centered @ centered)
    residual = xy - xy.mean(axis=0) - centered[:, None] * velocity
    return dict(status="measured", sample_count=len(rows),
                duration_seconds=float(times[-1]-times[0]),
                net_displacement_m=float(np.linalg.norm(xy[-1]-xy[0])),
                linear_velocity_m_per_s=velocity.tolist(),
                linear_fit_residual_rms_m=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                interpretation="A local linear trend, not a moving/stationary verdict. Tracking noise, curved motion and collisions affect this estimate; inspect video when ambiguous.")


def encode_saved_parameters(scene, fit):
    layout = make_parameter_layout(scene.object_count)
    objects = [fit["parameters"][name] for name in scene.object_ids]
    pairs = {frozenset(p["object_ids"]): p for p in fit["pair_parameters"]}
    theta = np.zeros(layout.size, dtype=np.float32)
    theta[layout.velocity_slice] = np.asarray([p["initial_velocity_plane_m_per_s"] for p in objects]).ravel()
    theta[layout.angular_velocity_slice] = [p["initial_angular_velocity_rad_per_s"] for p in objects]
    theta[layout.log_collision_scale_slice] = np.log([p["collision_scale"] for p in objects])
    theta[layout.log_mass_slice] = np.log([p["relative_mass"] for p in objects[:-1]])
    theta[layout.log_inertia_scale_slice] = np.log([p["inertia_scale"] for p in objects])
    theta[layout.friction_slice] = [p["ground_friction"] for p in objects]
    theta[layout.spin_friction_slice] = [p["spin_friction"] for p in objects]
    theta[layout.contact_torque_transfer_slice] = [p["contact_torque_transfer"] for p in objects]
    for index, (i, j) in enumerate(layout.pairs):
        pair = pairs[frozenset((scene.object_ids[i], scene.object_ids[j]))]
        theta[layout.restitution_slice.start + index] = pair["restitution"]
        theta[layout.contact_friction_slice.start + index] = pair["contact_friction"]
    return theta


def experiment_scene(scene, extension_seconds=0.0, remove_objects=()):
    if not np.isfinite(extension_seconds) or extension_seconds < 0:
        raise ValueError("extension_seconds must be finite and nonnegative")
    unknown = set(remove_objects) - set(scene.object_ids)
    if unknown:
        raise ValueError(f"Unrecognized object ids: {sorted(unknown)}")
    extra = int(np.ceil(extension_seconds * scene.fps))
    count = scene.frame_count + extra
    arrays = {}
    for name in ("target_xy", "target_yaw", "target_centers_world", "target_rotations_world", "observed_mask"):
        value = getattr(scene, name)
        pad = np.zeros((extra, *value.shape[1:]), dtype=value.dtype)
        arrays[name] = np.concatenate((value, pad), axis=0)
    # Keep parameter indexing unchanged. Removed bodies never activate, so they
    # cannot participate in collision search and are omitted from all exports.
    first = tuple(count + 1 if name in remove_objects else scene.first_offsets[i]
                  for i, name in enumerate(scene.object_ids))
    return replace(scene, frames=tuple(range(count)), first_offsets=first,
                   last_offsets=(count - 1,) * scene.object_count, **arrays)


class PhysicsExperiments:

    def _engine(self, scene, *, event_cap=128):
        return BatchWarpEventEngine(scene, capacity=1, device=self.device, event_cap=event_cap,
                                    proxy_mode=self.fit["proxy_mode"],
                                    contact_slack_m=self.fit["contact_slack_m"],
                                    contact_margin_m=self.fit["contact_margin_m"])


    def assets(self):
        from atw.polyworld.clevrer_evidence import GUIDANCE
        descriptions = {o["target_id"]: o.get("discriminative_description", "") for o in self.inventory}
        return dict(
            experiment_protocol=dict(result_kind="events", guidance=GUIDANCE),
            capabilities=dict(edited_rollouts=True, swr_engine="rigid"),
            grounded_appearance=self.grounded_appearance,
            object_identity_index=[dict(object_id=name, shape=self.scene.shapes[i].geometry_type,
                                        visual_description=descriptions.get(name, ""))
                                   for i, name in enumerate(self.scene.object_ids)],
            coordinate_system="metric support-plane XY; seconds from video start; fixed camera",
            observed_end_seconds=(self.scene.frame_count - 1) / self.scene.fps,
            fps=self.scene.fps,
            objects=[dict(object_id=name, description=descriptions.get(name, ""),
                          physical_type="rigid", geometry=self.scene.shapes[i].geometry_type,
                          model_appearance_seconds=self.scene.first_offsets[i] / self.scene.fps,
                          visibility=self.visibility[name],
                          parameters=self.fit["parameters"][name])
                     for i, name in enumerate(self.scene.object_ids)],
            fit_point_rmse_m=self.fit["overall_point_rmse_m"],
            pair_parameters=self.fit["pair_parameters"],
            limitations=["Planar rigid motion only; fixed heights; collision proxies, not full mesh contacts.",
                         "Parameters for unobserved interactions may be imputed, not identified.",
                         "Each object appears in the modeled world at model_appearance_seconds, its first usable 3D state. Before that time it is absent, not an unmodeled interval. This applies equally to baseline and edited rollouts.",
                         "Experiments keep bodies alive after their last observation, including outside the image.",
                         "Removal excludes a body throughout the modeled experiment, not just from a particular collision.",
                         "A finite rollout with no collision is not proof of no collision for all future time."],
            paths=dict(tracking=str(self.root / "tracking"), swr=str(self.root / "swr/result.json"),
                       reconstruction=str(self.root / "reconstruction")),
            available_rollouts=list(self.rollouts),
        )

    def simulate(self, extension_seconds, remove_objects=None, option_id=None):
        started = perf_counter()
        removed = sorted(set(remove_objects or []))
        scene = experiment_scene(self.scene, float(extension_seconds), removed)
        predicted, events = self._engine(scene).rollout_numpy(self.theta)
        rollout_id = f"rollout_{len(self.rollouts) + 1:03d}"
        centers, poses = _predicted_poses_world(scene, predicted)
        trajectories = {}
        for i, name in enumerate(scene.object_ids):
            trajectories[name] = [] if name in removed else [
                dict(frame_index=f, time_s=f / scene.fps, plane_xy=predicted[f, i, :2].tolist(),
                     yaw_delta_rad=float(predicted[f, i, 2]), point_cloud_center_world=centers[f, i].tolist(),
                     pose_4x4_world=poses[f, i].tolist())
                for f in range(scene.first_offsets[i], scene.frame_count)]
        payload = dict(
            rollout_id=rollout_id, object_ids=list(scene.object_ids), source_kind="simulation",
            remove_objects=removed, extension_seconds=float(extension_seconds),
            intervention=dict(type="remove_objects" if removed else "none",
                              semantics="The edited world is replayed from its modeled beginning, NOT edited at the original video cutoff. Events shared with the unedited world are still events of this edited world."),
            simulator_initialization={name: dict(first_state_seconds=self.scene.first_offsets[i] / scene.fps,
                                                last_observed_state_seconds=self.scene.last_offsets[i] / scene.fps)
                                      for i, name in enumerate(self.scene.object_ids)},
            end_seconds=(scene.frame_count - 1) / scene.fps,
            observed_end_seconds=(self.scene.frame_count - 1) / scene.fps,
            covers_future=bool(extension_seconds > 0),
            coverage_note=("This rollout covers ONLY the original observation interval; it cannot establish absence of future outcomes."
                           if extension_seconds == 0 else "Absence of events is established only up to end_seconds, not indefinitely."),
            events=[dict(event_time_s=e.time_s, frame_float=e.time_s * scene.fps,
                         seconds_after_observation=e.time_s - (self.scene.frame_count - 1) / scene.fps,
                         temporal_scope="after_observation" if e.time_s > (self.scene.frame_count - 1) / scene.fps else "within_observation",
                         object_ids=[scene.object_ids[e.object_i], scene.object_ids[e.object_j]],
                         normal_impulse=e.normal_impulse_magnitude) for e in events],
            predicted_trajectories=trajectories, seconds=perf_counter() - started,
            event_capacity_reached=len(events) >= 128,
            assumptions=self.assets()["limitations"],
        )
        path = self.output / "rollouts" / f"{rollout_id}.json"
        write_json(path, payload)
        self.rollouts[rollout_id] = payload
        if removed:
            from atw.polyworld.intervention_comparison import compare_rollouts
            try:
                control_id = next((rid for rid, r in self.rollouts.items()
                    if not r.get('remove_objects') and r.get('extension_seconds') == float(extension_seconds)), None)
                if control_id is None:
                    control_id = self.simulate(extension_seconds, [])['rollout_id']
                payload['intervention_comparison'] = compare_rollouts(self.rollouts[control_id], payload,
                    fps=scene.fps, control_id=control_id, edited_id=rollout_id)
            except (ValueError, RuntimeError, OSError) as error:
                payload['intervention_comparison'] = dict(status='unavailable', reason=f'Paired control unavailable: {error}')
            write_json(path, payload)
        return {k: v for k, v in payload.items() if k not in {"predicted_trajectories", "simulator_initialization"}} | {"artifact": str(path)}

    def query_events(self, rollout_id, object_ids=None, start_seconds=0., end_seconds=None):
        rollout = self.rollouts[rollout_id]
        end = rollout["end_seconds"] if end_seconds is None else float(end_seconds)
        if not 0 <= start_seconds <= end <= rollout["end_seconds"]:
            raise ValueError("Requested interval must lie within the simulated horizon")
        selected = set(object_ids or [])
        if not selected <= set(self.scene.object_ids):
            raise ValueError("Unrecognized object ids")
        events = [e for e in rollout["events"] if selected <= set(e["object_ids"])
                  and start_seconds <= e["event_time_s"] <= end]
        edited_world = bool(rollout.get("remove_objects"))
        return dict(source_kind="simulation", rollout_id=rollout_id, object_ids=object_ids,
                    interval_seconds=[start_seconds, end], events=events,
                    event_log_complete=True,
                    event_capacity_reached=rollout["event_capacity_reached"],
                    coverage_limitations=[],
                    model_appearance_seconds={name: (None if name in rollout.get("remove_objects", []) else
                        self.scene.first_offsets[self.scene.object_ids.index(name)] / self.scene.fps)
                        for name in (object_ids or self.scene.object_ids)},
                    coverage_basis=("completed_edited_rollout" if edited_world else
                                    "completed_rollout"),
                    interpretation="Objects are absent before their model appearance time. With a complete uncapped log, no listed event means no collision occurred in this modeled world during the queried interval.")

    def query_trajectory(self, object_id, rollout_id=None, start_seconds=0., end_seconds=None, sample_count=12):
        i = self.scene.object_ids.index(object_id)
        if rollout_id is None:
            rows = [dict(frame_index=f, time_s=f / self.scene.fps,
                         plane_xy=self.scene.target_xy[f, i].tolist(),
                         yaw_delta_rad=float(self.scene.target_yaw[f, i]))
                    for f in range(self.scene.frame_count) if self.scene.observed_mask[f, i]]
        else:
            rows = self.rollouts[rollout_id]["predicted_trajectories"][object_id]
        end = (self.scene.frame_count - 1) / self.scene.fps if rollout_id is None else self.rollouts[rollout_id]["end_seconds"]
        available_end = end
        end = end if end_seconds is None else float(end_seconds)
        if not 0 <= start_seconds <= end <= available_end or int(sample_count) < 1:
            raise ValueError("Invalid interval or sample count")
        rows = [r for r in rows if start_seconds <= r["time_s"] <= end]
        indices = np.unique(np.linspace(0, len(rows) - 1, min(int(sample_count), len(rows))).astype(int)) if rows else []
        return dict(source_kind="tracked_observation" if rollout_id is None else "simulation",
                    object_id=object_id, rollout_id=rollout_id, available_count=len(rows),
                    translation_measurement=translation_measurement(rows),
                    model_appearance_seconds=(None if rollout_id is not None and object_id in self.rollouts[rollout_id].get("remove_objects", []) else self.scene.first_offsets[i] / self.scene.fps),
                    interpretation="The object is absent before its model appearance time or throughout a rollout that removes it. Empty samples alone do not establish motion or stationarity after appearance.",
                    samples=[{k: rows[j][k] for k in ("time_s", "plane_xy", "yaw_delta_rad")} for j in indices])

    def compare(self, base_rollout_id, edited_rollout_id, object_ids=None):
        base, edit = self.rollouts[base_rollout_id], self.rollouts[edited_rollout_id]
        result = {}
        for name in object_ids or self.scene.object_ids:
            left = {r["frame_index"]: r for r in base["predicted_trajectories"][name]}
            right = {r["frame_index"]: r for r in edit["predicted_trajectories"][name]}
            common = sorted(left.keys() & right.keys())
            delta = [float(np.linalg.norm(np.array(left[f]["plane_xy"]) - right[f]["plane_xy"])) for f in common]
            result[name] = dict(common_frame_count=len(common), max_center_displacement_m=max(delta, default=None))
        return dict(source_kind="simulation_comparison", base_rollout_id=base_rollout_id,
                    edited_rollout_id=edited_rollout_id, trajectories=result,
                    base_events=base["events"], edited_events=edit["events"],
                    base_horizon=base["end_seconds"], edited_horizon=edit["end_seconds"])

    def query_mask_motion(self, object_id, start_seconds=0., end_seconds=None):
        """Read-only local mask-motion observation."""
        end = (self.scene.frame_count-1)/self.scene.fps if end_seconds is None else float(end_seconds)
        if not 0 <= start_seconds <= end <= (self.scene.frame_count-1)/self.scene.fps:
            raise ValueError("Requested mask interval must lie within the observed video")
        frames = list(range(int(np.ceil(start_seconds*self.scene.fps)), int(np.floor(end*self.scene.fps))+1))
        track_id = self.track_ids[object_id]
        info = next(iter(self.mask_index.values()))
        target = np.zeros((len(frames), info["height"], info["width"]), dtype=bool)
        others = np.zeros_like(target)
        offsets = {frame:i for i, frame in enumerate(frames)}
        for key, info in self.mask_index.items():
            frame = info["frame_index"]
            if frame not in offsets:
                continue
            mask = np.unpackbits(self.masks[key])[:info["num_bits"]].reshape(info["height"], info["width"]).astype(bool)
            if info["track_id"] == track_id:
                if self._retains_mask_frame(object_id, frame):
                    target[offsets[frame]] = mask
            else:
                others[offsets[frame]] |= mask
        decision, _ = classify_object_motion(target, others)
        observations = []
        for offset in np.unique(np.linspace(0, len(frames)-1, min(12, len(frames))).astype(int)):
            yy, xx = np.where(target[offset])
            observations.append(dict(frame=frames[offset], time_s=frames[offset]/self.scene.fps,
                                     visible_pixels=len(xx),
                                     centroid_px=[float(xx.mean()), float(yy.mean())] if len(xx) else None,
                                     touches_border=bool(len(xx) and (xx.min()==0 or yy.min()==0 or xx.max()==target.shape[2]-1 or yy.max()==target.shape[1]-1))))
        witness = decision["max_pair"]
        if witness is not None:
            witness = dict(witness, frame_a=frames[witness["frame_a"]], frame_b=frames[witness["frame_b"]])
        centroid_rows = [dict(time_s=o["time_s"], plane_xy=o["centroid_px"]) for o in observations if o["centroid_px"] is not None]
        measured = translation_measurement(centroid_rows)
        centroid_summary = dict(status=measured["status"],
                                interpretation="Image-mask centroid trend only, not material-point motion; occlusion or clipping can also cause it.")
        if measured["status"] == "measured":
            centroid_summary.update(displacement_px=measured["net_displacement_m"],
                                    trend_px_per_s=measured["linear_velocity_m_per_s"],
                                    trend_residual_rms_px=measured["linear_fit_residual_rms_m"])
        return dict(source_kind="mask_observation", object_id=object_id,
                    interval_seconds=[start_seconds, end],
                    visible_at_interval_end=bool(len(target) and target[-1].any()),
                    centroid_summary=centroid_summary,
                    supported_silhouette_change=decision["state"] == "moving",
                    test_conclusive=decision["state"] != "unknown",
                    eligible_frames=[frames[f] for f in decision["eligible_frames"]],
                    mask_observations=observations,
                    temporal_coverage=decision["temporal_coverage"], strongest_comparison=witness,
                    interpretation="No supported silhouette change means no change above the test's spatial tolerance, NOT zero motion: slow motion and rotation remain possible. Inspect raw temporal trends too. Centroids are not correspondences and shift under occlusion/clipping. Test inconclusive means missing evidence.")

    def _retains_mask_frame(self, object_id, frame):
        termination = self.mask_termination_frames[object_id]
        return termination < 0 or frame < termination

    def inspect_frames(self, frame_indices, object_id=None):
        images = []
        capture = cv2.VideoCapture(str(self.video))
        try:
            for frame in frame_indices:
                frame = int(frame)
                if not 0 <= frame < self.scene.frame_count:
                    raise ValueError("Frame is outside the observed video")
                if object_id is not None and not self._retains_mask_frame(object_id, frame):
                    raise ValueError(f"Mask identity for {object_id} is terminated at frame {frame}; use the original video for unbound visual evidence")
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Cannot read frame {frame}")
                mask = None
                if object_id is not None:
                    key = f"{self.track_ids[object_id]}__frame_{frame:05d}"
                    if key not in self.mask_index:
                        raise ValueError(f"No observed mask for {object_id} at {frame}")
                    info = self.mask_index[key]
                    mask = np.unpackbits(self.masks[key])[:info["num_bits"]].reshape(info["height"], info["width"]).astype(bool)
                label = f"{object_id or 'Original video'} | frame {frame} | {frame / self.scene.fps:.2f}s"
                image = labeled_observation(image, label, mask)
                path = self.output / "images" / f"{object_id or 'video'}_{frame:05d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    cv2.imwrite(str(path), image)
                images.append(str(path))
        finally:
            capture.release()
        return dict(source_kind="video_observation", images=images)
