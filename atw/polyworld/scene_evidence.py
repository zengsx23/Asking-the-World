"""Read-only second-stage evidence over a first-stage handoff.

The handoff selects files and records uncertainty. It never grants this backend
an upstream action. Recorded fits are evidence, not new intervention rollouts.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from atw.polyworld.experiment_views import labeled_observation




def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class SceneEvidence:




    def _ids(self, values):
        if not isinstance(values, (list, tuple)) or any(not isinstance(x, str) for x in values):
            raise ValueError("object_ids must be a list of existing string IDs")
        unknown = set(values) - self.objects.keys()
        if unknown:
            raise ValueError(f"Unrecognized object IDs: {sorted(unknown)}")
        return values

    def _interval(self, start_seconds, end_seconds, sample_count=12):
        end = self.end if end_seconds is None else float(end_seconds)
        start = float(start_seconds)
        if not np.isfinite([start, end]).all() or not 0 <= start <= end <= self.end:
            raise ValueError(f"Interval must lie within observed video [0, {self.end}]")
        if isinstance(sample_count, bool) or int(sample_count) != sample_count or not 1 <= sample_count <= 32:
            raise ValueError("sample_count must be an integer from 1 to 32")
        return start, end, int(sample_count)





    def inspect_frames(self, frame_indices, object_id=None, *, identity_view=False):
        if not isinstance(frame_indices, list) or not 1 <= len(frame_indices) <= 8 or any(
                isinstance(f, bool) or not isinstance(f, int) or not 0 <= f < self.frame_count for f in frame_indices):
            raise ValueError(f"Supply 1..8 integer frame_indices within [0, {self.frame_count - 1}]")
        if object_id is not None:
            self._ids([object_id])
        capture = cv2.VideoCapture(str(self.video))
        images, highlighted, unavailable = [], [], []
        try:
            for frame in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Cannot read original frame {frame}")
                label, identity_mask = "Original", None
                if object_id is not None:
                    try:
                        mask = self._mask(object_id, frame)
                    except (KeyError, ValueError):
                        mask = None
                    if mask is not None and mask.any():
                        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                        identity_mask = mask
                        highlighted.append(frame)
                        label = object_id
                    else:
                        unavailable.append(frame)
                        label = f"Original (mask unavailable: {object_id})"
                image = labeled_observation(image, f"{label} | frame {frame} | {frame/self.fps:.2f}s",
                                            identity_mask, identity_view=identity_view)
                directory = "identity_images" if identity_view else "images"
                path = self.output / directory / f"{object_id or 'video'}_{frame:05d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    cv2.imwrite(str(path), image)
                images.append(str(path))
        finally:
            capture.release()
        result = dict(source_kind="video_observation", images=images)
        if object_id is not None:
            result["highlighting"] = dict(object_id=object_id, highlighted_frames=highlighted, unavailable_frames=unavailable,
                interpretation="Unavailable masks return the original frame; this does not establish object absence.")
        return result

    def query_mask_motion(self, object_id, start_seconds=0., end_seconds=None):
        if self._rigid is not None:
            return self._rigid.query_mask_motion(object_id, start_seconds, end_seconds)
        self._ids([object_id])
        start, end, _ = self._interval(start_seconds, end_seconds)
        frames = self._frames(start, end, 12)
        observations = []
        for frame in frames:
            try:
                mask = self._mask(object_id, int(frame))
            except (KeyError, ValueError):
                continue
            yy, xx = np.where(mask)
            observations.append(dict(frame=int(frame), time_s=float(frame/self.fps), visible_pixels=len(xx),
                centroid_px=[float(xx.mean()), float(yy.mean())] if len(xx) else None,
                bbox_xyxy=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if len(xx) else None))
        return dict(source_kind="mask_observation", object_id=object_id, interval_seconds=[start, end],
                    mask_observations=observations, coordinate_system="mask pixels; x right, y down",
                    interpretation="Region measurements, not material-point velocity. Missing masks cannot prove rest or absence.")

    def query_mask_assets(self, object_ids=None, frame_index=None):
        ids = list(self.objects) if object_ids is None else list(self._ids(object_ids))
        ids = [name for name in ids if name not in self.background_object_ids]
        if frame_index is not None and (isinstance(frame_index, bool) or not isinstance(frame_index, int)
                                       or not 0 <= frame_index < self.frame_count):
            raise ValueError(f"frame_index must be an integer within [0, {self.frame_count - 1}]")
        rows = []
        for name in ids:
            obj = self.objects[name]
            if frame_index is not None:
                frame = frame_index
            else:
                frame = self.best_frames.get(name)
                if frame is None:
                    frame = (self.visibility.get(name, {}).get("entry") or {}).get("frame")
            row = dict(object_id=name,
                description=obj.get("discriminative_description", obj.get("category", "")),
                physical_type=("fluid" if name in self.fit.get("fluid_ids", []) else "rigid")
                    if self.engine == "fluid" else obj.get("physical_type"),
                geometry=obj.get("rigid_shape"),
                **{key:obj[key] for key in ("kind", "origin_id", "is_origin", "color") if key in obj},
                visibility=self.visibility.get(name), frame_index=frame)
            if frame is None:
                row.update(mask_available=False, reason="No nonempty identity mask was retained")
            else:
                try:
                    mask = self._mask(name, frame).astype(np.uint8)
                except (KeyError, ValueError, OSError) as error:
                    row.update(mask_available=False, reason=str(error))
                else:
                    yy, xx = np.where(mask)
                    components = cv2.connectedComponents(mask)[0] - 1 if len(xx) else 0
                    row.update(mask_available=bool(len(xx)), mask_pixels=int(len(xx)),
                        bbox_xyxy=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if len(xx) else None,
                        connected_components=int(components), mask_wh=[int(mask.shape[1]), int(mask.shape[0])])
            rows.append(row)
        visible = [row["object_id"] for row in rows if row.get("mask_available")]
        return dict(source_kind="mask_observation", requested_frame=frame_index,
            distinct_identity_count=len(visible), visible_identity_ids=visible, objects=rows,
            interpretation="Counted retained identity IDs, not connected components. Metadata supplies candidate color/shape labels; verify ambiguous appearance in original frames. Missing masks do not prove absence.")

    def _frames(self, start, end, count):
        first, last = int(np.ceil(start*self.fps)), int(np.floor(end*self.fps))
        return np.unique(np.linspace(first, last, min(count, last-first+1)).astype(int)) if first <= last else []

    def query_trajectory(self, object_id, rollout_id=None, start_seconds=0., end_seconds=None, sample_count=12):
        if self._rigid is not None:
            return self._rigid.query_trajectory(object_id, rollout_id, start_seconds, end_seconds, sample_count)
        self._ids([object_id])
        if rollout_id is not None:
            rollout = self.rollouts[rollout_id]
            end = self._rollout_end(rollout) if end_seconds is None else float(end_seconds)
            if not 0 <= start_seconds <= end <= self._rollout_end(rollout) or not isinstance(sample_count,int) or not 1 <= sample_count <= 32:
                raise ValueError("Invalid rollout interval or sample_count (1..32)")
            values = [r for r in rollout["trajectories"].get(object_id,[]) if start_seconds <= r["frame_index"]/self.fps <= end]
            # Preserve independent cloth model variants at the sampled frames.
            frames = sorted({r["frame_index"] for r in values})
            sampled = {frames[i] for i in np.unique(np.linspace(0,len(frames)-1,min(sample_count,len(frames))).astype(int))} if frames else set()
            samples = [r | dict(time_s=r["frame_index"]/self.fps) for r in values if r["frame_index"] in sampled]
            from atw.polyworld.experiment_views import image_projection
            project = image_projection(self, rollout)
            for row in samples:
                key = next(k for k in ("center_world", "xy_normalized", "center_xy_normalized", "center_xy_px") if k in row)
                row["image_xy_px"] = project([row[key]])[0].tolist()
            result = dict(source_kind="new_simulation", rollout_id=rollout_id, object_id=object_id,
                samples=samples, image_coordinates=dict(source_wh=[self.width,self.height], axes="x right, y down; outside-image coordinates are retained"),
                coordinate_system=rollout["coordinate_system"], assumptions=rollout["assumptions"])
            if self.engine == "rope":
                result["center_translation"] = self._rope_center_translation(values, rollout_id)
            return result
        start, end, count = self._interval(start_seconds, end_seconds, sample_count)
        tracking = self.paths["tracking"]
        path = tracking / "rigid_trajectories" / f"{object_id}.npz"
        rigid = path.is_file()
        if not rigid:
            path = tracking / "trajectories" / f"{object_id}.npz"
        rows = []
        with np.load(path) as data:
            values = data["poses" if rigid else "model_xyz"]
            confidence = None if rigid else data["tracking_confidence"]
            for frame in self._frames(start, end, count):
                if frame >= len(values) or 0 <= self.termination.get(object_id, -1) <= frame:
                    continue
                if not any(span["first_frame"] <= frame <= span["last_frame"]
                           for span in self.visibility.get(object_id, {}).get("intervals", [])):
                    continue
                value = values[frame]
                points = value if rigid else value[(confidence[frame] > 0) & np.isfinite(value).all(axis=1)]
                if not rigid and not len(points):
                    continue
                center = value[:3, 3] if rigid else points.mean(axis=0)
                if np.isfinite(center).all():
                    row = dict(frame_index=int(frame), time_s=float(frame/self.fps), center_world=center.tolist())
                    if rigid:
                        row["rotation_world"] = value[:3, :3].tolist()
                    rows.append(row)
        return dict(source_kind="tracked_observation", object_id=object_id, samples=rows,
                    coordinate_system="retained 3D world coordinates", interpretation="Tracking estimates; verify apparent motion against video. Missing samples are not disappearance.")

    def simulate(self, extension_seconds, remove_objects=None, mass_scales=None, continue_emission=False,
                 emission_fluid_id=None, measurements=None, queries=None, option_id=None,
                 rope_mass_protocol="factual", intervention_seconds=None, rope_initial_state=None, _with_comparison=True):
        if not self.simulation_ready:
            raise ValueError("Experiments unavailable; use retained evidence")
        if (rope_mass_protocol != 'factual' or intervention_seconds is not None or rope_initial_state is not None) and self.engine != 'rope':
            raise ValueError('Rope mass protocols are unavailable for this solver')
        if rope_initial_state is not None and rope_initial_state not in {'fitted', 'rest'}:
            raise ValueError('rope_initial_state must be fitted or rest')
        if rope_mass_protocol == 'contphy-reset-v1':
            from atw.polyworld.rope_qa_protocol import prepare_request
            prepare_request('simulate', dict(mass_scales=mass_scales or {}, extension_seconds=extension_seconds,
                rope_initial_state=rope_initial_state or 'rest', intervention_seconds=intervention_seconds))
            rope_initial_state = 'rest'
        if self._contphy:
            request = dict(extension_seconds=extension_seconds,remove_objects=sorted(remove_objects or []),
                           mass_scales=mass_scales or {},continue_emission=continue_emission,
                           emission_fluid_id=emission_fluid_id)
            if rope_mass_protocol != "factual" or intervention_seconds is not None:
                request.update(rope_mass_protocol=rope_mass_protocol, intervention_seconds=intervention_seconds)
            if self.engine == 'rope':
                request['rope_initial_state'] = rope_initial_state or ('rest' if mass_scales and rope_mass_protocol == 'factual' else 'fitted')
            existing = next(((name,data) for name,data in self.rollouts.items()
                             if (option_id is None or data.get("option_id") == option_id)
                             and not data.get("endpoint_motion")
                             and data.get('simulation_request', {k:data.get(k) for k in request}) == request),None)
            if existing:
                rollout_id,result = existing
            else:
                result = self._contphy.run(**request)
                result['simulation_request'] = request
                rollout_id = f"experiment_{len(self.rollouts)+1}"
                if option_id is not None:
                    result["option_id"] = option_id
                self.rollouts[rollout_id] = result
                directory = self.output / rollout_id
                directory.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(directory / "geometry.npz", **result["geometry"])
                persisted = {k:v for k,v in result.items() if k not in {"geometry","particle_birth","particle_fluid"}}
                save(directory / "result.json", persisted)
                if self.engine == "fluid":
                    np.savez_compressed(directory / "particles.npz", birth=result["particle_birth"], fluid=result["particle_fluid"])
            outcomes = {name:self.query_trajectory(name,rollout_id,sample_count=3)["samples"] for name in result["trajectories"]}
            detail_tools = ("inspect_fluid_rollout with the question's target material ID; it renders only that liquid while all liquids remain in the simulation"
                            if self.engine == "fluid" else
                            "query_world for scene contact, named regions, pose and motion; query_trajectory for detail")
            summary = dict(source_kind="new_simulation", rollout_id=rollout_id, end_seconds=self._rollout_end(result),
                remove_objects=result["remove_objects"],mass_scales=result["mass_scales"],continue_emission=continue_emission,
                coordinate_system=result["coordinate_system"],outcome_samples=outcomes,
                image_coordinates=dict(source_wh=[self.width,self.height], axes="image_xy_px: x right, y down"),
                assumptions=result["assumptions"],replay_check=result["replay_check"],
                detail_tools=detail_tools)
            for key in ('simulation_stability', 'mass_edits', 'rope_mass_protocol',
                        'intervention_seconds', 'readout_start_seconds', 'rope_initial_state'):
                if key in result:
                    summary[key] = result[key]
            if 'rope_constraint_diagnostics' in result:
                summary['rope_constraint_diagnostics'] = {name:{k:v for k,v in d.items()
                    if k != 'slack_length_normalized_per_frame'} for name,d in result['rope_constraint_diagnostics'].items()}
            if option_id is not None:
                summary["option_id"] = option_id
            if emission_fluid_id is not None:
                summary["emission_fluid_id"] = emission_fluid_id
            if self.engine in {"cloth", "soft_body"}:
                summary["requires_result_measurement"] = True
            from atw.polyworld.experiment_views import last_in_frame_samples
            in_frame = last_in_frame_samples(self, result)
            if in_frame:
                summary["last_in_frame_samples"] = in_frame
                summary["image_coordinates"]["interpretation"] = (
                    "These are projected centers, not visibility or goal labels. "
                    "Offscreen endpoints alone cannot locate a destination; inspect the path through the scene.")
            if self.engine == "rope":
                readout_start = result.get('readout_start_seconds', 0.)
                summary["center_translation"] = {name:self._rope_center_translation(
                    [r for r in rows if r['frame_index'] / self.fps >= readout_start])
                    for name, rows in result["trajectories"].items()}
                rotations = {}
                for name in dict.fromkeys(p["object_id"] for p in result["pulleys"]):
                    try:
                        rotations[name] = [{k:v for k,v in row.items() if k not in {"assumption", "sign_convention"}}
                            for row in self.query_pulley_rotation(name, rollout_id, start_seconds=readout_start, sample_count=2)["results"]]
                    except (ValueError, KeyError, IndexError) as error:
                        rotations[name] = dict(status="unavailable", reason=str(error))
                summary["pulley_rotation"] = dict(objects=rotations,
                    interpretation="Spin is separate from center translation; inferred from no-slip rope contact, not observed wheel texture. Positive angular velocity is clockwise in image coordinates. Samples show first and last contact intervals; query_pulley_rotation provides details.")
            if measurements is not None:
                summary["measurements"] = self._completed_measurements(measurements, rollout_id)
            if queries is not None:
                summary["measurements"] = self._completed_world_queries(queries,rollout_id)
            control = self._contphy.control_arguments(**request)
            if control is not None:
                try:
                    preserved = self.simulate(**control,measurements=measurements,queries=queries)
                    summary["input_control"] = {key:preserved[key] for key in ("rollout_id","end_seconds","continue_emission","measurements") if key in preserved}
                except (ValueError, RuntimeError, OSError) as error:
                    summary["input_control"] = dict(status="unavailable",reason=str(error))
                summary["input_control"]["purpose"] = "Preserves fitted source inputs and isolates the object edit. The requested rollout also changes future material supply; use the control for the pure object edit."
            if _with_comparison and (remove_objects or mass_scales or continue_emission or emission_fluid_id):
                paired_request = dict(request, remove_objects=[], mass_scales={})
                if not (remove_objects or mass_scales):
                    paired_request.update(continue_emission=False, emission_fluid_id=None)
                if rope_mass_protocol == 'pre_event':
                    paired_request['intervention_seconds'] = result['intervention_seconds']
                self._attach_intervention_comparison(summary, paired_request)
            return summary
        if mass_scales or continue_emission:
            raise ValueError("Rigid experiments do not support these arguments")
        result = self._rigid.simulate(extension_seconds, remove_objects)
        rollout = self.rollouts[result["rollout_id"]]
        comparisons = {}
        for name, rows in rollout["predicted_trajectories"].items():
            baseline = {r["frame_index"]: r["plane_xy"] for r in self.fit["predicted_trajectories"].get(name, [])}
            differences = [float(np.linalg.norm(np.asarray(r["plane_xy"]) - baseline[r["frame_index"]]))
                           for r in rows if r["frame_index"] in baseline]
            if differences:
                comparisons[name] = dict(common_frame_count=len(differences), max_center_displacement_m=max(differences))
        if measurements is not None:
            result["measurements"] = self._completed_measurements(measurements, result["rollout_id"])
        if queries is not None:
            result["measurements"] = self._completed_world_queries(queries,result["rollout_id"])
            result["evidence_summary"] = [dict(query_id=row.get("id"), **row["evidence_summary"])
                for row in result["measurements"].get("results", []) if "evidence_summary" in row]
        result["negative_event_claims_require_pair_query"] = queries is None
        return result | dict(saved_baseline_comparison=dict(source_kind="simulation_comparison", objects=comparisons,
            interpretation="Displacement from retained unedited SWR over common observed frames; not a visibility or causality verdict."))

    def _attach_intervention_comparison(self, summary, control_request):
        from atw.polyworld.intervention_comparison import compare_rollouts, compare_rotation
        cache = getattr(self, '_intervention_controls', None)
        if cache is None:
            self._intervention_controls = cache = {}
        key = json.dumps(control_request, sort_keys=True)
        edited_id = summary['rollout_id']
        try:
            if key not in cache:
                cache[key] = self.simulate(**control_request, _with_comparison=False)['rollout_id']
            control_id = cache[key]
            base, edit = self.rollouts[control_id], self.rollouts[edited_id]
            start = edit.get('readout_start_seconds') or 0.
            if (self.engine == 'fluid' and not edit.get('remove_objects')
                    and (edit.get('continue_emission') or edit.get('emission_fluid_id'))):
                start = edit.get('observed_end_seconds', start)
            comparison = compare_rollouts(base, edit, fps=self.fps, control_id=control_id,
                edited_id=edited_id, start_seconds=start, image_height=self.height)
            comparison['control_request'] = control_request
            if self.engine == 'rope':
                for name in sorted({p['object_id'] for p in edit.get('pulleys', [])}):
                    try:
                        start = edit.get('readout_start_seconds') or 0.
                        end = min(self._rollout_end(base), self._rollout_end(edit))
                        a = self.query_pulley_rotation(name, control_id, start_seconds=start, end_seconds=end)['results']
                        b = self.query_pulley_rotation(name, edited_id, start_seconds=start, end_seconds=end)['results']
                        rotations = [dict(rope_id=right['rope_id'], **compare_rotation(left, right,
                            all(r.get('simulation_stability', {}).get('usable_for_downstream_reasoning', True) for r in (base, edit))))
                            for right in b for left in a if left['rope_id'] == right['rope_id']]
                        comparison['objects'].setdefault(name, {})['rotations'] = rotations
                    except (ValueError, KeyError, IndexError) as error:
                        comparison['objects'].setdefault(name, {})['rotation'] = dict(status='unavailable', reason=str(error))
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            comparison = dict(status='unavailable', reason=f'Paired control unavailable: {error}',
                              edited_rollout_id=edited_id, control_request=control_request)
        summary['intervention_comparison'] = comparison
        self.rollouts[edited_id]['intervention_comparison'] = comparison
        save(self.output / edited_id / 'intervention_comparison.json', comparison)

    def _rollout_end(self, rollout):
        return max(r["frame_index"] for values in rollout["trajectories"].values() for r in values)/self.fps

    def nearest_rope_endpoint(self, object_id, scenes):
        """Geometric diagnostic only; interventions use exact modeled bindings."""
        candidates = []
        for scene in scenes:
            frame = int(scene.frames[0])
            try:
                mask = self._mask(object_id, frame)
                yy, xx = np.where(mask)
                if not len(xx):
                    raise ValueError('Object mask is empty at model start')
                center = np.array([xx.mean() * self.width / mask.shape[1], yy.mean() * self.height / mask.shape[0]])
                source = 'object_mask_centroid'
            except (KeyError, ValueError, OSError):
                body = next((b for b in scene.bodies if b.object_id == object_id), None)
                if body is None:
                    continue
                center = np.asarray(body.initial_center) * self.height
                source = 'modeled_body_center'
            for endpoint, node in enumerate((0, -1)):
                point = np.asarray(scene.initial_positions[node]) * self.height
                candidates.append(dict(control_kind='rope_endpoint', rope_id=scene.rope_id, endpoint_index=endpoint,
                    distance_px=float(np.linalg.norm(point - center)), frame_index=frame,
                    object_center_xy_px=center.tolist(), endpoint_xy_px=point.tolist(), position_source=source))
            for body in scene.bodies:
                if body.role == 'pulley' and body.dynamic:
                    point = np.asarray(body.initial_center) * self.height
                    candidates.append(dict(control_kind='dynamic_pulley', rope_id=scene.rope_id,
                        object_id=body.object_id, distance_px=float(np.linalg.norm(point - center)),
                        frame_index=frame, object_center_xy_px=center.tolist(),
                        control_center_xy_px=point.tolist(), position_source=source))
        if not candidates:
            raise ValueError('Cannot locate selected object relative to modeled rope ends; use original-video reasoning')
        candidates.sort(key=lambda row: row['distance_px'])
        nearest = [row for row in candidates if np.isclose(
            row['distance_px'], candidates[0]['distance_px'], rtol=0, atol=1e-6)]
        dynamic = [row for row in nearest if row['control_kind'] == 'dynamic_pulley']
        if dynamic:
            names = sorted({row['object_id'] for row in dynamic})
            raise ValueError('Nearest physical control is dynamic pulley ' + ','.join(names)
                + '; prescribed pulley-center motion is unsupported, so use original-video reasoning')
        if len(nearest) > 1:
            raise ValueError('Selected object is equally close to multiple rope ends; endpoint binding is unresolved')
        return candidates[0]

    def query_rope_controls(self):
        from atw.polyworld.rope_controls import control_catalog
        if self.engine != "rope":
            raise ValueError("Rope controls require a rope SWR")
        scenes = self._contphy._load()
        modeled = {s.rope_id for s in scenes} | {b.object_id for s in scenes for b in s.bodies}
        return dict(source_kind="saved_swr_fit", unmodeled_object_ids=sorted(set(self.objects)-modeled),
                    **control_catalog(scenes))

    def drive_rope_endpoint(self, object_id, direction, extension_seconds=0., option_id=None,
                            endpoint_index=None, speed_image_height_per_s=.05,
                            max_displacement_image_height=.1):
        from atw.polyworld.rope_controls import resolve_endpoint
        if self.engine != "rope" or not self.simulation_ready or direction not in {"up", "down"}:
            raise ValueError("Use a ready rope SWR and direction up/down")
        if isinstance(extension_seconds, bool) or not np.isfinite(extension_seconds) or not 0 <= extension_seconds <= 2:
            raise ValueError("Endpoint extension must be within 0..2 seconds")
        self._ids([object_id])
        for name, value in (("speed_image_height_per_s", speed_image_height_per_s),
                            ("max_displacement_image_height", max_displacement_image_height)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= .5:
                raise ValueError(f"{name} must be finite, positive and <= 0.5")
        scenes = self._contphy._load()
        binding = resolve_endpoint(scenes, object_id, endpoint_index)
        control = dict(**binding, direction=direction, speed_image_height_per_s=speed_image_height_per_s,
                       max_displacement_image_height=max_displacement_image_height)
        result = self._contphy._rope(scenes, extension_seconds, set(), {}, False, False,
                                    endpoint_motion=control)
        driven = {binding['rope_id']}
        stability = result['simulation_stability']
        original = next(s for s in scenes if s.rope_id == binding['rope_id'])
        duration = (len(result['geometry'][original.rope_id]) - 1) / original.fps
        result.update(source_kind='new_simulation', engine='rope', remove_objects=[], mass_scales={},
            extension_seconds=extension_seconds, option_id=option_id,
            endpoint_motion=dict(object_id=object_id, **control, masses_changed=False,
                effective_speed_image_height_per_s=min(speed_image_height_per_s, max_displacement_image_height/max(duration, 1/original.fps)),
                interval_seconds=[original.frames[0]/original.fps, original.frames[0]/original.fps+duration]),
            assumptions=['Prescribed exact-end vertical motion from model start; this is an actuator, not a mass intervention.',
                         'Spin requires supported taut no-slip contact; fixed-center spin is not translation.'])
        rid = f"experiment_{len(self.rollouts)+1}"
        self.rollouts[rid] = result
        pulley_ids = sorted({body.object_id for scene in scenes if scene.rope_id in driven
                             for body in scene.bodies if body.role == 'pulley'})
        pulley_motion = []
        for pulley_id in pulley_ids:
            readout = self.query_pulley_rotation(
                pulley_id, rid, start_seconds=0., end_seconds=self._rollout_end(result))
            translation = readout.get('center_translation', {})
            moving = translation.get('moving')
            translation_state = ('moving' if moving is True else 'stationary' if moving is False
                                 else 'unknown')
            pulley_motion.append(dict(
                object_id=pulley_id,
                center_translation={k: translation[k] for k in (
                    'status', 'moving', 'max_displacement_px', 'net_displacement_xy_px',
                    'net_direction_xy', 'interval_seconds') if k in translation}
                    | {'translation_state': translation_state},
                rotations=[{k: row[k] for k in (
                    'rope_id', 'direction', 'interval_seconds') if k in row}
                    for row in readout.get('results', [])],
                movement_interpretation=('Pulley movement means center translation. Rotation without '
                    'center translation is rotation only and is not pulley movement.')))
        result['pulley_motion'] = pulley_motion
        directory = self.output / rid
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / 'geometry.npz', **result['geometry'])
        save(directory / 'result.json', {k:v for k,v in result.items() if k != 'geometry'})
        summary = dict(rollout_id=rid, source_kind='new_simulation', end_seconds=self._rollout_end(result),
                    endpoint_motion=result['endpoint_motion'], simulation_stability=stability,
                    pulley_motion=pulley_motion, assumptions=result['assumptions'])
        self._attach_intervention_comparison(summary, dict(extension_seconds=extension_seconds,
            remove_objects=[], mass_scales={}, continue_emission=False, emission_fluid_id=None))
        return summary

    def change_mass(self, object_id, direction, extension_seconds=0., option_id=None,
                    factor=None, mass_scope="body", member_mass_fraction=None,
                    rope_mass_protocol="factual", intervention_seconds=None, rope_initial_state=None):
        from atw.polyworld.swr_queries import MASS_FACTORS, MOTION_MEANING
        from atw.polyworld.rope_controls import resolve_mass
        self._ids([object_id])
        if self.engine != "rope" or direction not in MASS_FACTORS:
            raise ValueError("Use the rope engine and direction increase or decrease")
        if isinstance(extension_seconds, bool) or not np.isfinite(extension_seconds) or not 0 <= extension_seconds <= 2:
            raise ValueError("change_mass extension_seconds must be within [0, 2] to limit unstable extrapolation")
        if rope_mass_protocol == 'contphy-reset-v1':
            from atw.polyworld.rope_qa_protocol import prepare_request
            factor = prepare_request('change_mass', dict(direction=direction, factor=factor,
                rope_initial_state=rope_initial_state or 'rest', intervention_seconds=intervention_seconds))['factor']
        factor = MASS_FACTORS[direction] if factor is None else factor
        if (isinstance(factor, bool) or not isinstance(factor, (int, float)) or not np.isfinite(factor)
                or not .1 <= factor <= 10 or (factor <= 1 if direction == "increase" else factor >= 1)):
            raise ValueError("factor must be in [0.1, 10], >1 for increase and <1 for decrease")
        binding = resolve_mass(self._contphy._load(), object_id, mass_scope, member_mass_fraction)
        effective_factor = 1 + member_mass_fraction * (factor - 1) if member_mass_fraction is not None else factor
        kwargs = {}
        if rope_mass_protocol != "factual" or intervention_seconds is not None:
            kwargs.update(rope_mass_protocol=rope_mass_protocol, intervention_seconds=intervention_seconds)
        if rope_initial_state is not None:
            kwargs['rope_initial_state'] = rope_initial_state
        result = self.simulate(extension_seconds,mass_scales={binding['object_id']:effective_factor},option_id=option_id, **kwargs)
        samples = result.get("outcome_samples", {}).get(object_id, [])
        points = [row.get("image_xy_px") for row in samples if row.get("image_xy_px") is not None]
        if len(points) >= 2:
            delta_y = float(points[-1][1]-points[0][1])
            vertical = "down" if delta_y > 0 else "up" if delta_y < 0 else "stationary"
            simulated_target = dict(status="measured", direction=vertical, net_delta_y_px=delta_y,
                image_axis="positive y is down")
        else:
            simulated_target = dict(status="unavailable", direction=None,
                reason="Fewer than two projected target samples")
        if result.get('readout_start_seconds', 0.) > 0:
            translation = result.get('center_translation', {}).get(object_id, {})
            delta = translation.get('net_displacement_xy_px')
            simulated_target = (dict(status='measured', direction=translation['net_direction_xy'][1],
                net_delta_y_px=delta[1], interval_seconds=translation.get('interval_seconds'))
                if delta is not None else dict(status='unavailable', direction=None))
        return result | dict(object_id=object_id, direction=direction, factor=factor,
            effective_mass_factor=effective_factor, mass_binding=binding,
            simulated_target_vertical_motion=simulated_target,
            interpretation=MOTION_MEANING + " Mass is multiplied, not prescribed motion. Increased mass may still move up if other loads dominate. Assembly edits affect the combined mass, not an identified member mass.")

    def probe_rope_mass(self, object_id, target_ids, relative_deltas=None, **options):
        """Explicit numerical sensitivity API; not an automatic QA default."""
        from atw.polyworld.rope_mass_probe import RopeMassProbe, target_control
        self._ids([object_id, *target_ids])
        if self.engine != 'rope' or not self.simulation_ready:
            raise ValueError('A completed rope SWR fit is required')
        scene, _ = target_control(self._contphy._load(), object_id)
        fit = next(r for r in self.fit['ropes'] if r['rope_id'] == scene.rope_id)
        values = options.get('member_factors') if options.get('member_factors') is not None else relative_deltas
        if values is None:
            raise ValueError('Supply relative_deltas or member_factors')
        capacity = max(options.get('calibration_population', 64),
                       options.get('baseline_count', 4) * (len(values)+1))
        if not 1 <= capacity <= 512:
            raise ValueError('Mass-probe batch capacity must be within 1..512')
        if not hasattr(self, '_mass_probe_engines'):
            self._mass_probe_engines = {}
            self._mass_probe_count = 0
        key = (scene.rope_id, capacity)
        if key not in self._mass_probe_engines:
            self._mass_probe_engines[key] = RopeMassProbe(scene, fit, device=self._contphy.device, capacity=capacity)
        probe = self._mass_probe_engines[key]
        result = probe.run(object_id, target_ids, relative_deltas, **options)
        self._mass_probe_count += 1
        folder = self.output / f'mass_probe_{self._mass_probe_count}'
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / 'geometry.npz', **probe.last_geometry)
        result.update(probe_id=folder.name, geometry_archive=str(folder / 'geometry.npz'))
        save(folder / 'result.json', result)
        return result

    def query_world(self, queries, rollout_id=None):
        from atw.polyworld.swr_queries import query_world
        if self._rigid is None or rollout_id is None:
            return query_world(self,queries,rollout_id)
        if not isinstance(queries, list) or not 1 <= len(queries) <= 6:
            raise ValueError("queries must contain 1..6 requests")
        labels = [q.get('id') for q in queries if isinstance(q, dict)]
        if len(labels) != len(queries) or any(not isinstance(i, str) or not i.strip() for i in labels) or len(set(labels)) != len(labels):
            raise ValueError("Each query needs a unique nonempty id")
        results = []
        for request in queries:
            if request.get("kind") == "contact":
                if request.get("temporal", "any") != "any":
                    raise ValueError("Contact events require temporal=any")
                outcome = self.query_event_outcome(rollout_id, request.get("object_ids"),
                    request.get("start_seconds", 0.), request.get("end_seconds"))
                results.append(dict(outcome, id=request['id'], metric='contact',
                    condition_met=outcome.get('event_occurs'), temporal='any'))
            else:
                results.extend(query_world(self,[request],rollout_id)['results'])
        return dict(source_kind="new_simulation", engine=self.engine, rollout_id=rollout_id,
                    semantic_queries=True, results=results)

    def _completed_world_queries(self, queries, rollout_id):
        try:
            return self.query_world(queries,rollout_id)
        except (ValueError,KeyError,TypeError,OSError) as error:
            return dict(status="invalid_query",rollout_id=rollout_id,reason=str(error),
                instruction="Simulation completed. Correct query_world on this rollout; do not resimulate.")

    def _rope_center_translation(self, rows, rollout_id=None):
        from atw.polyworld.rope_rotation import summarize_center_translation
        points = [r.get("xy_normalized", r.get("center_xy_normalized")) for r in rows]
        centers = np.asarray(points, float)*self.height
        times = [r["frame_index"]/self.fps for r in rows]
        result = summarize_center_translation(centers, times)
        if self.rollouts.get(rollout_id, {}).get('rope_mass_protocol') == 'contphy-reset-v1':
            from atw.polyworld.rope_qa_protocol import translation_readout, metric_scale
            scales = [metric_scale(r, self.height) for r in self.fit['ropes']]
            result['benchmark_motion'] = translation_readout(centers, times,
                pixels_per_world_unit=float(np.median(scales)))
        return result

    def query_pulley_rotation(self, object_id, rollout_id=None, start_seconds=0., end_seconds=None, sample_count=12):
        from atw.polyworld.rope_rotation import infer_pulley_rotation
        self._ids([object_id])
        scenes = self._contphy._load()
        if rollout_id is None:
            geometries = {r["rope_id"]: np.asarray([p["xy_normalized"] for p in r["predicted_centerline"]]) for r in self.fit["ropes"]}
            trajectories = {name:self._saved_trajectory(name) for name in self.objects}
            end = self.end
        else:
            rollout = self.rollouts[rollout_id]
            geometries, trajectories, end = rollout["geometry"], rollout["trajectories"], self._rollout_end(rollout)
        end = end if end_seconds is None else float(end_seconds)
        if not np.isfinite([start_seconds,end]).all() or not 0 <= start_seconds < end or not 1 <= sample_count <= 32:
            raise ValueError("Rotation requires a nonempty finite interval and 1..32 samples")
        if rollout_id is not None and rollout.get('simulation_stability', {}).get('usable_for_downstream_reasoning') is False:
            from atw.polyworld.evidence_actions import query_requirement
            pulleys = [s.rope_id for s in scenes if any(b.object_id == object_id and b.role == 'pulley' for b in s.bodies)]
            if not pulleys:
                raise ValueError("Object is not a pulley in the retained rope topology")
            unsupported = query_requirement("The rope rollout is unstable or excessively stretched; its spin is not reliable.",
                "Inspect the retained geometry and intervention assumptions; do not infer rotation from this rollout.")
            return dict(source_kind='new_simulation', object_id=object_id, **unsupported,
                center_translation=unsupported,
                results=[dict(rope_id=name, direction=None, **unsupported) for name in pulleys])
        results = []
        for scene in scenes:
            body = next((b for b in scene.bodies if b.object_id==object_id and b.role=="pulley"), None)
            if body is None:
                continue
            points = geometries[scene.rope_id]
            centers = np.broadcast_to(body.initial_center,(len(points),2)).copy()
            for row in trajectories.get(object_id,[]):
                i = row["frame_index"]-scene.frames[0]
                if 0 <= i < len(centers):
                    centers[i] = row["xy_normalized"]
            times = (np.arange(len(points))+scene.frames[0])/self.fps
            selection = (times>=start_seconds)&(times<=end)
            diagnostics = (rollout.get('rope_constraint_diagnostics', {}).get(scene.rope_id, {})
                           if rollout_id is not None else next(r for r in self.fit['ropes'] if r['rope_id']==scene.rope_id).get('constraint_diagnostics', {}))
            slack = diagnostics.get('slack_length_normalized_per_frame')
            taut = None if slack is None else np.asarray(slack)[selection] <= 1e-6
            result = infer_pulley_rotation(points[selection],centers[selection],body.radius,times[selection],
                                           contact_tolerance=max(2/self.height,body.radius*.15), taut_frames=taut)
            if rollout_id is not None and rollout.get('rope_mass_protocol') == 'contphy-reset-v1':
                from atw.polyworld.rope_qa_protocol import rotation_readout
                result['benchmark_rotation'] = rotation_readout(result)
            samples = result["samples"]
            result["samples"] = [samples[i] for i in np.unique(np.linspace(0,len(samples)-1,min(sample_count,len(samples))).astype(int))] if samples else []
            results.append(dict(rope_id=scene.rope_id,**result))
        if not results:
            raise ValueError("Object is not a pulley in the retained rope topology")
        translation = self._rope_center_translation([r for r in trajectories.get(object_id, [])
            if start_seconds <= r["frame_index"]/self.fps <= end], rollout_id)
        return dict(source_kind="saved_swr_fit" if rollout_id is None else "new_simulation",
            object_id=object_id, center_translation=translation, results=results,
            movement_interpretation=("Pulley movement means center translation. Rotation without "
                "center translation is rotation only and is not pulley movement."))

    def inspect_rollout(self, rollout_id, start_seconds=None, end_seconds=None, option_id=None,
                        object_labels=None):
        from atw.polyworld.experiment_views import render_rollout_video
        return render_rollout_video(self, rollout_id, start_seconds, end_seconds,
                                    object_labels=object_labels)

    def inspect_material_response(self, object_ids, property_name):
        from atw.polyworld.material_response import inspect_material_response
        return inspect_material_response(self, object_ids, property_name)

    def query_event_timeline(self, **arguments):
        from atw.polyworld.clevrer_evidence import timeline
        return timeline(self, **arguments)

    def query_event_path(self, **arguments):
        from atw.polyworld.clevrer_evidence import event_path
        return event_path(self, **arguments)

    def inspect_fluid_rollout(self, rollout_id, object_id, start_seconds=None, end_seconds=None,
                              option_id=None, object_labels=None):
        from atw.polyworld.experiment_views import render_fluid_rollout
        result = render_fluid_rollout(self, rollout_id, object_id, start_seconds, end_seconds,
                                      object_labels=object_labels)
        if option_id is not None:
            result["option_id"] = option_id
        return result

    def query_events(self, rollout_id, object_ids=None, start_seconds=0., end_seconds=None):
        from atw.polyworld.swr_measurements import event_verdict
        if object_ids and len(object_ids) == 2:
            return self.query_event_outcome(rollout_id, object_ids, start_seconds, end_seconds)
        result = self._rigid.query_events(rollout_id, object_ids, start_seconds, end_seconds)
        if object_ids:
            result = event_verdict(result)
        return result

    def query_event_outcome(self, rollout_id, object_ids, start_seconds=0., end_seconds=None):
        from atw.polyworld.clevrer_evidence import event_outcome
        return event_outcome(self, rollout_id, object_ids, start_seconds, end_seconds)

    def measure_results(self, queries, rollout_id=None):
        from atw.polyworld.evidence_measurements import measure_results
        return measure_results(self, queries, rollout_id)

    def _completed_measurements(self, queries, rollout_id):
        try:
            return self.measure_results(queries, rollout_id)
        except (ValueError, KeyError, TypeError, OSError) as error:
            return dict(status="invalid_query",rollout_id=rollout_id,reason=str(error),
                instruction="The experiment completed and remains available; correct the query using measure_results, without repeating simulation.")

    def query_swr(self, section, object_ids=None, start_seconds=0., end_seconds=None, sample_count=12, property_name=None):
        ids = list(self.objects) if object_ids is None else self._ids(object_ids)
        start, end, count = self._interval(start_seconds, end_seconds, sample_count)
        result = dict(source_kind="saved_swr_fit", engine=self.engine, section=section,
                      limitations=self.limitations, asset_cautions=self.cautions)
        if self.engine == "rope":
            result["coordinate_system"] = "Image x right, y down; both axes normalized by source image height"
        elif self.engine == "fluid":
            result["coordinate_system"] = "SWR image pixels; x right, y down"
            result["image_wh"] = self.fit["image_wh"]
        if section == "property":
            from atw.polyworld.swr_properties import property_evidence
            return property_evidence(self,property_name,ids)
        if section == "summary":
            if self.engine == "rope":
                result.update(self._contphy.evidence_catalog())
            return result | dict(summary={k:v for k,v in self.summary.items()
                if k not in {"paths", "object_ids", "segmentation_guidance", "wall_seconds", "fit_seconds", "limitations"}})
        if section == "parameters":
            if self.engine == "cloth":
                data = {name: row["parameters"] for name, row in self.fit["objects"].items() if name in ids}
            elif self.engine == "rope":
                data = {r["rope_id"]: {k:r.get(k) for k in ("parameters", "bodies", "event_frame", "event_type", "mass_schedule")}
                        for r in self.fit["ropes"] if r["rope_id"] in ids or any(b["object_id"] in ids for b in r["bodies"])}
                try:
                    scenes = self._contphy._load()
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    result["mass_parameter_mapping_error"] = f"Topology unavailable: {error}; do not infer endpoint IDs from body order."
                else:
                    for scene in scenes:
                        if scene.rope_id not in data:
                            continue
                        names = {(f"endpoint_{b.endpoint_index}" if b.role == "endpoint" else b.object_id): b.object_id
                                 for b in scene.bodies if b.role in {"endpoint", "pulley"}}
                        data[scene.rope_id]["mass_parameter_objects"] = {
                            key: names.get(key) for key in data[scene.rope_id]["parameters"]["relative_masses_before"]}
            elif self.engine == "fluid":
                data = {"parameters": self.fit["parameters"], "fluid_ids": self.fit["fluid_ids"],
                        "emitters": [{k:v for k,v in e.items() if k != "active_frames"} for e in self.fit["emitters"]]}
            else:
                data = {name: p for name,p in self.fit.get("parameters", {}).items() if name in ids}
            if self.engine in {"cloth","soft_body"} and self._contphy:
                semantics = self._contphy.parameter_semantics()
                if semantics:
                    result["parameter_semantics"] = semantics
                if len(data) == 2:
                    left, right = data
                    result["numeric_comparison"] = dict(object_order=[left,right], parameters={
                        key:dict(left=float(a),right=float(b),difference=float(a-b),
                                 ratio_left_to_right=float(a/b) if b else None)
                        for key,a in data[left].items() if isinstance(a,(int,float)) and not isinstance(a,bool)
                        and isinstance((b:=data[right].get(key)),(int,float)) and not isinstance(b,bool)},
                        interpretation="Arithmetic only, not a calibrated material-property comparison or a threshold for 'much'. For a physical-property judgment use section=property; not_identified cannot establish either side of a comparison.")
            return result | dict(parameters=data, interpretation="Fitted effective parameters, not ground truth. Units and identifiability vary by engine; compare with observed behavior. No new intervention was executed.")
        if section == "events":
            if "events" not in self.fit:
                return result | dict(status="unavailable", reason="This saved solver does not export an event list; do not infer no events")
            rows = []
            for event in self.fit["events"]:
                time = event.get("time_s", event.get("event_time_s"))
                involved = event.get("object_ids", [event.get("soft_body_id"), event.get("contact_id")])
                if start <= time <= end and (object_ids is None or set(ids) <= set(involved)):
                    rows.append(event)
            return result | dict(events=rows, interval_seconds=[start,end], interpretation="Recorded model events, not a complete observed event list.")
        if section != "trajectory" or not 1 <= len(ids) <= 6:
            raise ValueError("section must be summary, parameters, events or trajectory; trajectory requires 1..6 object_ids")
        trajectories = {}
        for name in ids:
            rows = [r for r in self._saved_trajectory(name) if start <= r["frame_index"] / self.fps <= end]
            indices = np.unique(np.linspace(0, len(rows)-1, min(count, len(rows))).astype(int)) if rows else []
            trajectories[name] = dict(samples=[rows[i] | dict(time_s=rows[i]["frame_index"]/self.fps) for i in indices],
                                      available_count=len(rows))
            if self.engine == "rope":
                trajectories[name]["center_translation"] = self._rope_center_translation(rows)
        return result | dict(trajectories=trajectories,
            interpretation="Saved baseline predictions only, not a changed world or independent observation."
            + (" up_axis_tilt_degrees measures the body's initially most vertical axis against reconstructed up: 0 is vertical, 90 is horizontal; world Z is not height." if self.engine == "cloth" else ""))

    def _saved_trajectory(self, name):
        if self.engine in {"rigid", "soft_body"}:
            return self.fit.get("predicted_trajectories", {}).get(name, [])
        if self.engine == "cloth":
            row = self.fit["objects"].get(name)
            if row:
                centers = np.mean(np.asarray(row["predicted_positions_world"]), axis=1)
                return [dict(frame_index=f, center_world=c.tolist()) for f,c in zip(row["frames"], centers)]
            rows = []
            for cloth, fit in self.fit["objects"].items():
                bodies = fit.get("predicted_rigid_trajectories", {})
                if name in bodies.get("object_ids", []):
                    index = bodies["object_ids"].index(name)
                    for offset,frame in enumerate(fit["frames"]):
                        item = dict(frame_index=frame, fitted_with=cloth, center_world=bodies["centers_world"][offset][index])
                        axes = [bodies.get(f"axis_{axis}_world") for axis in "xyz"]
                        if all(axis is not None for axis in axes):
                            rotation = np.column_stack([axis[offset][index] for axis in axes])
                            item["rotation_world"] = rotation.tolist()
                            item["half_extents_m"] = fit["obstacle_half_extents_m"][name]
                            down = getattr(self, "world_frame", {}).get("gravity", {}).get("gravity_down_world")
                            if down is not None:
                                up = -np.asarray(down)/np.linalg.norm(down)
                                initial = np.column_stack([axis[0][index] for axis in axes])
                                vertical_axis = int(np.argmax(np.abs(initial.T@up)))
                                item["initial_up_axis"] = vertical_axis
                                item["up_axis_tilt_degrees"] = float(np.degrees(np.arccos(np.clip(abs(rotation[:,vertical_axis]@up),0,1))))
                                item["center_height_along_up_m"] = float(np.asarray(item["center_world"])@up)
                        for key in ("rotations_world", "rotation_vectors_world", "quaternions_world"):
                            if key in bodies:
                                item[key] = bodies[key][offset][index]
                        rows.append(item)
            return rows
        if self.engine == "rope":
            rows = []
            for rope in self.fit["ropes"]:
                if rope["rope_id"] == name:
                    rows += [dict(frame_index=r["frame_index"], center_xy_normalized=np.mean(r["xy_normalized"], axis=0).tolist())
                             for r in rope["predicted_centerline"]]
                elif name in rope.get("predicted_body_centers", {}):
                    rows += rope["predicted_body_centers"][name]
            return rows
        if self.engine == "fluid":
            if name not in self.fit["fluid_ids"]:
                return []
            index = self.fit["fluid_ids"].index(name)
            with np.load(self.paths["swr"] / self.fit["prediction_npz"]) as data:
                predicted, labels, birth = data["predicted_xy"], data["particle_fluid"], data["particle_birth"]
                rows = []
                for frame, xy in enumerate(predicted):
                    points = xy[(labels == index) & (birth <= frame) & np.isfinite(xy).all(axis=1)]
                    if len(points):
                        rows.append(dict(frame_index=frame, center_xy_px=points.mean(axis=0).tolist(),
                            bounds_xyxy_px=[*points.min(axis=0).tolist(), *points.max(axis=0).tolist()], active_particles=len(points)))
                return rows
        return []
