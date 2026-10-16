"""Read-only measurement data exported by each SWR adapter, without running a solver."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class PointSeries:
    object_id: str
    variant: str
    frames: np.ndarray
    points: list
    kind: str


def event_verdict(result):
    """Three-valued engine-event evidence; incomplete logs cannot prove absence."""
    gaps = result.get("coverage_limitations", [])
    def verified(event):
        time = event.get("time_s", event.get("event_time_s"))
        for gap in gaps:
            a, b = gap["unverified_interval_seconds"]
            if a <= time and (time < b if gap.get("end_exclusive") else time <= b):
                return False
        return True
    positive = any(verified(e) for e in result.get("events", []))
    complete = (result.get("event_log_complete", True) and not gaps
                and not result.get("event_capacity_reached", False)
                and result.get("status") != "unavailable")
    verdict = result | dict(event_occurs=True if positive else (False if complete else None),
                         coverage_complete=bool(complete),
                         event_status="known" if positive or complete else "needs_query")
    if not positive and not complete:
        verdict.update(reason="The requested event has no positive witness and the event log is incomplete.",
            next_action="Read a complete replay of the same world and interval. If the event log cannot be recovered, request original-video reasoning for this question; do not treat this as no collision.")
    return verdict


class SWRMeasurements:
    """Engine-specific extraction and capability descriptions live here, not in the agent."""
    def __init__(self, backend, rollout_id=None):
        self.backend, self.rollout_id = backend, rollout_id
        self.engine, self.fit = backend.engine, backend.fit
        self.data = backend.rollouts[rollout_id] if rollout_id is not None else self.fit
        self._particles = None

    def capabilities(self):
        modes = {"fluid": ["center", "members"], "cloth": ["center", "members"],
                 "rope": ["center", "members"], "soft_body": ["center"], "rigid": ["center"]}
        return dict(point_modes=modes.get(self.engine, []), default_point_mode="members" if self.engine == "fluid" else "center",
            members={"fluid": "Born PBF particles, equal count weights; not calibrated fluid volume.",
                     "cloth": "Cloth nodes only; independently fitted systems stay separate.",
                     "rope": "Rope material nodes only."}.get(self.engine, "Not exported"),
            orientation={"cloth": "Exported rigid body axes, separately per cloth fit.",
                         "rigid": "Retained pose only; this planar engine cannot predict overturning."}.get(self.engine, "Not exported"),
            box_overlap="cloth: exported rigid obstacle boxes; signed geometric gap/proximity proxy, not a contact event log; positive gap is separation, do not invent a contact tolerance" if self.engine == "cloth" else "Not exported",
            contact_events={"rigid": "Collision events. Completed baseline and edited rollouts provide exhaustive finite-horizon event logs unless event capacity is reached. Objects are absent before their modeled appearance times.",
                            "soft_body": "Saved/new contact transitions; one contact label per frame, not exhaustive negative evidence."}.get(self.engine, "Not exported; image proximity is not physical contact."))

    def provenance(self):
        b = self.backend
        return dict(source_kind="saved_swr_fit" if self.rollout_id is None else "new_simulation",
            engine=self.engine, rollout_id=self.rollout_id,
            artifact=self.data.get("evidence_artifact",str(b.paths["swr"] if self.rollout_id is None else b.output / self.rollout_id)),
            assumptions=self.data.get("assumptions", []),
            replay_check=self.data.get("replay_check"),
            interpretation="Measurement of this modeled world, not validation against reality.")

    @property
    def end(self):
        if self.rollout_id is None:
            return self.backend.end
        if "end_seconds" in self.data:
            return self.data["end_seconds"]
        return self.backend._rollout_end(self.data)

    def trajectories(self, name):
        if self.engine == "rope" and self.rollout_id is None:
            rows = []
            for fit in self.fit["ropes"]:
                if name == fit["rope_id"]:
                    rows.extend(dict(frame_index=r["frame_index"],center_xy_normalized=np.mean(r["xy_normalized"],axis=0).tolist())
                                for r in fit["predicted_centerline"])
                else:
                    rows.extend(dict(r,fitted_with=fit["rope_id"]) for r in fit.get("predicted_body_centers",{}).get(name,[]))
            return rows
        if self.rollout_id is None:
            return self.backend._saved_trajectory(name)
        return self.data.get("trajectories", self.data.get("predicted_trajectories", {})).get(name, [])

    def groups(self, name):
        groups = defaultdict(list)
        for row in self.trajectories(name):
            groups[row.get("fitted_with", "joint")].append(row)
        return {key: sorted(rows, key=lambda r:r["frame_index"]) for key, rows in groups.items()}

    def project(self, points):
        from atw.polyworld.experiment_views import image_projection
        return image_projection(self.backend, self.data)(points)

    def points(self, name, mode):
        if self.engine == "fluid":
            return self._fluid_points(name, mode)
        if mode == "center":
            result = []
            for variant, rows in self.groups(name).items():
                key = next((k for k in ("center_world", "point_cloud_center_world", "xy_normalized",
                    "center_xy_normalized", "center_xy_px") if k in rows[0]), None)
                if key is None:
                    continue
                pixels = self.project([row[key] for row in rows])
                result.append(PointSeries(name, variant, np.array([r["frame_index"] for r in rows]),
                                          [p[None] for p in pixels], "projected_center"))
            return result
        if mode != "members":
            raise ValueError("point_mode must be center or members")
        if self.engine not in {"cloth", "rope"}:
            return []
        if self.rollout_id is not None:
            points = self.data.get("geometry", {}).get(name)
            if points is None:
                return []
            frames = sorted({r["frame_index"] for r in self.trajectories(name)})
        elif self.engine == "cloth":
            fit = self.fit["objects"].get(name)
            if fit is None:
                return []
            points, frames = fit["predicted_positions_world"], fit["frames"]
        else:
            fit = next((r for r in self.fit["ropes"] if r["rope_id"] == name), None)
            if fit is None:
                return []
            points = [r["xy_normalized"] for r in fit["predicted_centerline"]]
            frames = [r["frame_index"] for r in fit["predicted_centerline"]]
        if len(points) != len(frames):
            return []
        return [PointSeries(name, "joint", np.asarray(frames), [self.project(p) for p in points], "material_nodes")]

    def _fluid_points(self, name, mode):
        if name not in self.fit["fluid_ids"]:
            return []
        index = self.fit["fluid_ids"].index(name)
        if self.rollout_id is None:
            if self._particles is None:
                with np.load(self.backend.paths["swr"] / self.fit["prediction_npz"]) as data:
                    self._particles = {key: data[key] for key in ("predicted_xy", "particle_fluid", "particle_birth")}
            data = self._particles
            selection = data["particle_fluid"] == index
            points, births = data["predicted_xy"][:, selection], data["particle_birth"][selection]
        else:
            points = self.data["geometry"].get(name)
            if points is None or "particle_birth" not in self.data:
                return []
            births = self.data["particle_birth"][self.data["particle_fluid"] == index]
        pixels = []
        for frame, xy in enumerate(points):
            active = self.project(xy[births <= frame])
            pixels.append(active if mode == "members" or not len(active) else active.mean(0, keepdims=True))
        return [PointSeries(name, "joint", np.arange(len(points)), pixels,
                            "born_particles" if mode == "members" else "projected_center")]

    def orientations(self, name):
        # Soft-body and rope engines do not export rigid axes; no guessed orientation.
        return self.groups(name) if self.engine in {"cloth", "rigid"} else {}

    def static_proximity_target(self, name):
        # PBF keeps non-fluid identities fixed as image masks, including sources.
        if self.engine != "fluid" or name in self.fit["fluid_ids"]:
            return None
        mask = self.backend._mask(name,0)
        ys,xs = np.where(mask)
        if not len(xs):
            raise LookupError("The engine's static target mask is empty")
        scale = np.array([self.backend.width/mask.shape[1],self.backend.height/mask.shape[0]])
        return np.column_stack((xs,ys))*scale, dict(object_id=name,source_kind="engine_static_mask",
            mask_frame=0,resolution_px=scale.tolist(),assumption="PBF holds this retained mask fixed; this is image proximity, not an exported collision event.")

    def boxes(self, name):
        if self.engine != "cloth":
            return {}
        result = {}
        for variant, rows in self.groups(name).items():
            extents = self.fit.get("objects", {}).get(variant, {}).get("obstacle_half_extents_m", {}).get(name)
            result[variant] = {r["frame_index"]: (np.asarray(r["center_world"]), np.asarray(r["rotation_world"]),
                np.asarray(r.get("half_extents_m", extents))) for r in rows
                if "rotation_world" in r and (r.get("half_extents_m") is not None or extents is not None)}
        return {k:v for k,v in result.items() if v}

    def cloth_fit_assignments(self):
        """Assign each rigid body to the independently fitted cloth that reaches it."""
        if self.engine != "cloth":
            return {}
        cached = getattr(self.backend, "_cloth_fit_assignments", None)
        if cached is not None:
            return cached
        candidates = defaultdict(list)
        for variant, fit in self.fit.get("objects", {}).items():
            cloth = np.asarray(fit.get("predicted_positions_world", []), float)
            bodies = fit.get("predicted_rigid_trajectories", {})
            ids = bodies.get("object_ids", [])
            centers = np.asarray(bodies.get("centers_world", []), float)
            axes = [np.asarray(bodies.get(f"axis_{axis}_world", []), float) for axis in "xyz"]
            extents = fit.get("obstacle_half_extents_m", {})
            if cloth.ndim != 3 or centers.ndim != 3 or any(axis.ndim != 3 for axis in axes):
                continue
            for index, object_id in enumerate(ids):
                half = np.asarray(extents.get(object_id), float)
                if half.shape != (3,) or index >= centers.shape[1] or any(index >= axis.shape[1] for axis in axes):
                    continue
                frame_count = min(len(cloth), len(centers), *(len(axis) for axis in axes))
                minimum, near_frames = np.inf, 0
                for frame in range(frame_count):
                    points, center = cloth[frame], centers[frame, index]
                    rotation = np.column_stack([axis[frame, index] for axis in axes])
                    valid = (np.isfinite(points).all(1) if points.ndim == 2 else np.zeros(0, bool))
                    if not valid.any() or not np.isfinite(center).all() or not np.isfinite(rotation).all():
                        continue
                    local_gap = np.abs((points[valid] - center) @ rotation) - half
                    signed = (np.linalg.norm(np.maximum(local_gap, 0), axis=1)
                              + np.minimum(np.max(local_gap, axis=1), 0))
                    frame_minimum = float(np.min(signed))
                    minimum = min(minimum, frame_minimum)
                    near_frames += int(frame_minimum <= .01)
                if np.isfinite(minimum):
                    candidates[object_id].append(dict(
                        variant=variant, near_frame_count=near_frames,
                        minimum_cloth_box_gap_m=minimum))
        assignments = {}
        for object_id, rows in candidates.items():
            rows.sort(key=lambda row: (-row["near_frame_count"], row["minimum_cloth_box_gap_m"], row["variant"]))
            best, runner = rows[0], rows[1] if len(rows) > 1 else None
            unique_contact = best["near_frame_count"] > 0 and (
                runner is None or best["near_frame_count"] > runner["near_frame_count"])
            clear_nearest = runner is None or (
                best["near_frame_count"] == runner["near_frame_count"] == 0
                and runner["minimum_cloth_box_gap_m"] - best["minimum_cloth_box_gap_m"] >= .05)
            assignments[object_id] = dict(
                status="assigned" if unique_contact or clear_nearest else "ambiguous",
                variant=best["variant"] if unique_contact or clear_nearest else None,
                candidates=rows,
                basis=("Observed-window fitted cloth nodes versus each rigid OBB; "
                       "question text and answer labels are not used."))
        self.backend._cloth_fit_assignments = assignments
        return assignments

    def events(self, ids, start, end):
        if self.engine == "rigid" and self.rollout_id is not None:
            self.backend._rigid.rollouts = self.backend.rollouts
            return event_verdict(self.backend._rigid.query_events(self.rollout_id, ids, start, end))
        payload = dict(source_kind="saved_swr_fit" if self.rollout_id is None else "new_simulation",
                       interval_seconds=[start, end], event_log_complete=False)
        if "events" not in self.data:
            return event_verdict(payload | dict(status="unavailable", events=[], reason="Engine does not export contact events for this source"))
        events = [e for e in self.data["events"]
                  if start <= e.get("time_s", e.get("event_time_s", -1)) <= end
                  and set(ids) <= set(e.get("object_ids", [e.get("soft_body_id"), e.get("contact_id")]))]
        # Saved soft-body logs retain one contact label per sampled frame, not every contact.
        return event_verdict(payload | dict(events=events, reason="Recorded contact transitions; missing events are not exhaustive negative evidence"))
