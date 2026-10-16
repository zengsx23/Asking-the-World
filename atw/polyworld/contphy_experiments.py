"""Fixed-parameter forward experiments over retained ContPhy SWR assets.

No optimizer, segmentation, tracking, reconstruction or VLM is invoked here.
Each experiment owns a scene copy and writes only under the second-stage output.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


def extend(values, count, *, fill=None):
    extra = count - len(values)
    if extra <= 0:
        return values.copy()
    tail = np.repeat(values[-1:], extra, axis=0) if fill is None else np.full((extra, *values.shape[1:]), fill, dtype=values.dtype)
    return np.concatenate((values, tail))


def rows(frames, positions, key, **extra):
    return [dict(frame_index=int(f), **{key: p.tolist()}, **extra) for f, p in zip(frames, positions)]


class ContPhyExperiments:
    mass_factor_bounds = (.1, 10.)

    def __init__(self, backend, device):
        self.backend, self.device = backend, device
        self.kind, self.fit, self.paths = backend.engine, backend.fit, backend.paths
        self.loaded = None
        self.replay_check = None

    def capabilities(self):
        common = dict(extension_seconds="0..10 beyond observed end; all fitted parameters remain fixed")
        if self.kind == "rope":
            return common | dict(endpoint_motion="query_rope_controls lists explicit rope ends and attached IDs. drive_rope_endpoint prescribes up/down motion without changing mass; it is NOT a substitute for heavier/lighter.",
                mass_change="change_mass or simulate(mass_scales=...) changes fitted effective mass and simulates the response. Hanging-load assemblies require explicit mass_scope=assembly; individual member masses are not identified.",
                pulley_rotation="No-slip material-rope contact with center translation subtracted",
                movement_meaning="Center translation and rotation are separate readouts.")
        if self.kind == "cloth":
            return common | dict(remove_objects="Rigid obstacle IDs only; separately fitted cloth systems remain separate",
                independent_fits="Each fitted_with ID identifies a separate cloth-driven world. For local effects choose the fit whose cloth interacts with the bodies, using observations; unrelated cloth fits are not votes against that interaction. Measurements accept variant=fit ID and retain conditional provenance.")
        if self.kind == "soft_body":
            return common | dict(remove_objects="Soft-body IDs and static obstacle IDs; remove throughout the modeled experiment")
        return common | dict(remove_objects="Static obstacle IDs excluding emitters; remove throughout the experiment",
                             continue_emission="Default false preserves the observed emission schedule while existing material keeps moving. True adds NEW material from sources active at the cutoff, changing the population and its fractions. Use it only when the requested future includes further emission; settling existing material does not require it.")

    def evidence_catalog(self):
        if self.kind == "rope":
            ids = list(dict.fromkeys(r["rope_id"] for r in self.fit["ropes"]))
            missing = [name for name, obj in self.backend.objects.items()
                       if obj.get("physical_type") == "rope" and name not in ids]
            return dict(source_kind="saved_swr_fit", rope_count=len(ids), rope_ids=ids,
                scope="Distinct modeled rope identities, including chain-like members of physical_type=rope; not individual links or visible segments.",
                unmodeled_rope_ids=missing,
                physical_family_definitions=dict(rope="Rope-family flexible connectors, including cords and linked chains. Unqualified 'rope(s)' includes these subtypes; restrict only when a subtype is explicitly requested."))
        if self.kind == "cloth":
            return dict(material_response_convention={
                "harder_to_bend": "After interaction, the visible cloth area generally changes less.",
                "easier_to_wrinkle": "After interaction, the visible cloth area generally decreases more.",
                "evidence": "Compare spaced before, contact and after mask panels under the observed interaction. Missing or strongly occluded coverage is inconclusive."})
        if self.kind == "soft_body":
            return dict(material_response_convention={
                "elasticity": "Large visible deformation AND strong recovery after unloading. Both are required under this scene convention.",
                "plasticity": "Large visible deformation AND weak recovery after unloading; substantial residual deformation.",
                "little_deformation": "Little visible deformation means weak elasticity AND weak plasticity under this scene convention; it does not mean high elasticity because the body is stiff.",
                "evidence": "Compare original shapes before contact, during deformation and after unloading. Continued pressure or missing recovery coverage is inconclusive. Justify these qualitative labels from observed shapes, not fitted stiffness, damping, gain or frequency; this solver has no permanent-shape state."})
        if self.kind == "fluid":
            sources = []
            visibility = getattr(self.backend,"visibility",{})
            objects = getattr(self.backend,"objects",{})
            for emitter in self.fit["emitters"]:
                source, material = emitter["object_id"], emitter["fluid_id"]
                frames = emitter["active_frames"]
                entry = lambda name:(visibility.get(name,{}).get("entry") or {}).get("seconds")
                sources.append(dict(source_id=source,material_id=material,
                    color=objects.get(source,{}).get("color",objects.get(material,{}).get("color")),
                    source_first_visible_seconds=entry(source),material_first_visible_seconds=entry(material),
                    emission_interval_seconds=[min(frames)/self.backend.fps,max(frames)/self.backend.fps] if frames else None,
                    active_at_cutoff=bool(frames and max(frames)>=self.backend.frame_count-2)))
            starts = sorted({s["emission_interval_seconds"][0] for s in sources if s["emission_interval_seconds"]})
            for source in sources:
                interval=source["emission_interval_seconds"]
                source["emission_start_order"] = starts.index(interval[0])+1 if interval else None
            same_color = []
            colors = {str(v["color"]).strip().lower() for v in sources if v.get("color")}
            for color in sorted(colors):
                group = [v for v in sources if str(v.get("color", "")).strip().lower() == color]
                if len(group) < 2:
                    continue
                times = sorted({v["emission_interval_seconds"][0] for v in group if v["emission_interval_seconds"]})
                for v in group:
                    interval = v["emission_interval_seconds"]
                    v["same_color_emission_order"] = times.index(interval[0])+1 if interval else None
                order = [dict(
                    source_id=v["source_id"],material_id=v["material_id"],
                    emission_start_seconds=v["emission_interval_seconds"][0] if v["emission_interval_seconds"] else None,
                    rank=v["same_color_emission_order"]) for v in sorted(group,key=lambda v:(v["same_color_emission_order"] is None,v["same_color_emission_order"] or 0))]
                distinct = len(order) == 2 and [v["rank"] for v in order] == [1, 2]
                same_color.append(dict(color=color, order=order,
                    temporal_roles=(dict(
                        default_source_id=order[0]["source_id"], default_material_id=order[0]["material_id"],
                        other_source_id=order[1]["source_id"], other_material_id=order[1]["material_id"])
                        if distinct else None),
                    meaning=("Rank 1 emits first among this color. With exactly two distinct ranks, an unqualified "
                             "singular color material is the default first-emitted material and 'other' is rank 2. "
                             "Equal or missing ranks leave these roles unresolved. This is emission order, not nozzle appearance order.")))
            identity = dict(sources=sources,same_color_emission_order=same_color,reference_rule=
                "For 'the other emitter', compare same-appearance candidates using the question's contrast and appearance/emission order. "
                "The nozzle can be visible before it emits; material first-visible time and emission start are distinct evidence. "
                "With exactly two distinct same-color emission ranks, bind an unqualified singular color material to rank 1 "
                "and 'other', 'second' or 'later' to rank 2; use this timeline, never ID suffixes. "
                "Equal or missing times do not distinguish identities; keep unresolved references null. "
                "An emitter's material_id is its emitted fluid, not the emitter itself.")
            return dict(source_kind="saved_swr_fit",sources=sources,identity_evidence=identity,
                interpretation="Fitted source-to-material bindings and observed timing. Use chronological evidence for relative references; never infer order from names.")
        return {}

    def control_arguments(self, extension_seconds, remove_objects, mass_scales, continue_emission,
                          emission_fluid_id=None, rope_mass_protocol="factual", intervention_seconds=None,
                          rope_initial_state=None):
        if self.kind == "fluid" and continue_emission and remove_objects:
            return dict(extension_seconds=extension_seconds,remove_objects=remove_objects,
                        mass_scales=mass_scales,continue_emission=False)
        return None

    def parameter_semantics(self):
        if self.kind == "cloth":
            return dict(elasticity=dict(status="not_identified",reason="Dimensionless solver stiffness is not a calibrated elastic modulus, extensibility ratio or elastic recovery measurement. Its reciprocal is not a measured elasticity."),
                plasticity=dict(status="not_identified",reason="No evolving rest shape or yield law is fitted."),
                structural_stiffness="Effective constraint stiffness under this timestep, grid and loading; compare observed deformation only under comparable geometry and load.")
        if self.kind == "soft_body":
            from atw.polyworld.soft_body_swr import PARAMETER_SEMANTICS
            return PARAMETER_SEMANTICS
        return {}

    def _load(self):
        if self.loaded is not None:
            return self.loaded
        if self.kind == "cloth":
            from atw.polyworld.cloth_swr import load_cloth_scenes
            source = next(iter(self.fit["objects"].values()))["source"]
            self.loaded = load_cloth_scenes(self.paths["tracking"], grid_resolution=source["grid_resolution"],
                minimum_tracking_confidence=source["minimum_tracking_confidence"], target_depth_mode=source["target_depth_mode"])[0]
        elif self.kind == "soft_body":
            from atw.polyworld.soft_body_swr import load_softbody_scene
            self.loaded = load_softbody_scene(self.paths["tracking"], reconstruction_dir=self.paths["reconstruction"])[0]
        elif self.kind == "fluid":
            from atw.polyworld.fluid_swr import load_fluid_scene
            self.loaded = load_fluid_scene(self.paths["segmentation"], boundary_dir=self.paths.get("boundary"))
            if (self.loaded.image_width, self.loaded.image_height) != tuple(self.fit["image_wh"]):
                raise ValueError("Saved PBF and segmentation image dimensions differ")
        else:
            from atw.polyworld.rope_swr import load_rope_scenes
            from atw.polyworld.rope_topology import available_pulleys
            from atw.perception.object_reconstruction import read_packed_mask_archive
            fits = {r["rope_id"]: r for r in self.fit["ropes"]}
            scenes, source = load_rope_scenes(self.paths["segmentation"], node_count=next(iter(fits.values()))["node_count"],
                roles=self.fit.get("source", {}).get("rope_roles"),
                endpoint_policy=self.fit.get("source", {}).get("rope_endpoint_policy", "legacy"),
                tracking_policy=self.fit.get("source", {}).get("rope_tracking_policy", "legacy"))
            first = scenes[0]
            tracks = read_packed_mask_archive(Path(first.mask_archive), Path(first.mask_index))
            selected = []
            for scene in scenes:
                if scene.rope_id not in fits:
                    continue
                fit = fits[scene.rope_id]
                catalog = {b.object_id: b for b in (*scene.bodies, *available_pulleys(scene, tracks, source))}
                bodies = tuple(replace(catalog[b["object_id"]], role=b["role"], dynamic=b["dynamic"],
                    radius=b["radius_normalized"], parent_id=b.get("parent_id"),
                    initial_center=tuple(b.get("initial_center_normalized", catalog[b["object_id"]].initial_center)),
                    endpoint_offset=tuple(b.get("endpoint_offset_normalized", catalog[b["object_id"]].endpoint_offset)),
                    attachment_source=b.get("attachment_source", catalog[b["object_id"]].attachment_source),
                    default_relative_mass=b.get("default_relative_mass", 1.)) for b in fit["bodies"])
                saved = {b.object_id: b for b in bodies}
                bodies += tuple(b for b in scene.bodies if b.object_id not in saved
                    and b.role == 'attachment' and not b.dynamic and b.parent_id in saved
                    and not saved[b.parent_id].dynamic)
                selected.append(replace(scene, bodies=bodies, initial_positions=np.asarray(fit["initial_centerline_normalized"]),
                    mass_model=fit["optimization"].get("mass_model", "particle"),
                    route_visits=tuple(tuple(v) for v in fit["routing"].get("visits", ())),
                    endpoint_pinned=tuple(fit["endpoint_pinned"]),
                    event_frame_offset=(int(fit['event_frame']) - scene.frames[0]
                        if fit.get('event_frame') is not None else 0)))
            self.loaded = tuple(selected)
        return self.loaded

    def run(self, extension_seconds, remove_objects=None, mass_scales=None, continue_emission=False,
            emission_fluid_id=None, *, audit=False, rope_mass_protocol="factual", intervention_seconds=None,
            rope_initial_state=None):
        if isinstance(extension_seconds, bool) or not np.isfinite(extension_seconds) or not 0 <= extension_seconds <= 10:
            raise ValueError("extension_seconds must be a finite number from 0 to 10")
        removed = set(self.backend._ids(remove_objects or []))
        mass_scales = mass_scales or {}
        lower, upper = self.mass_factor_bounds
        if not isinstance(mass_scales, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not np.isfinite(v) or not lower <= v <= upper for v in mass_scales.values()):
            raise ValueError(f"mass_scales must map existing dynamic body IDs to factors in [{lower:g}, {upper:g}]")
        self.backend._ids(list(mass_scales))
        if not isinstance(continue_emission, bool):
            raise ValueError("continue_emission must be boolean")
        if mass_scales and self.kind != "rope" or continue_emission and self.kind != "fluid":
            raise ValueError("Requested edit is unavailable for this solver; inspect experiment capabilities")
        if (rope_mass_protocol != "factual" or intervention_seconds is not None or rope_initial_state is not None) and self.kind != "rope":
            raise ValueError("Rope mass protocols are unavailable for this solver")
        if rope_initial_state is not None and rope_initial_state not in {'fitted', 'rest'}:
            raise ValueError('rope_initial_state must be fitted or rest')
        source = self._load()
        if emission_fluid_id is not None:
            if self.kind != "fluid" or emission_fluid_id not in source.fluid_ids:
                raise ValueError("emission_fluid_id must be a modeled fluid material ID")
            if continue_emission:
                raise ValueError("Use target-specific predictive emission or the explicit all-source edit, not both")
        if self.kind == "fluid":
            result = self._fluid(source, extension_seconds, removed, mass_scales,
                                 continue_emission, audit, emission_fluid_id)
        elif self.kind == "rope":
            if rope_mass_protocol == 'contphy-reset-v1' and rope_initial_state == 'fitted':
                raise ValueError('ContPhy reset requires zero initial velocity')
            state = rope_initial_state or ('rest' if rope_mass_protocol == 'contphy-reset-v1'
                or mass_scales and rope_mass_protocol == 'factual' else 'fitted')
            result = self._rope(source, extension_seconds, removed, mass_scales, continue_emission, audit,
                mass_protocol=rope_mass_protocol, intervention_seconds=intervention_seconds, initial_state=state)
        else:
            result = getattr(self, "_" + self.kind)(source, extension_seconds, removed,
                                                     mass_scales, continue_emission, audit)
        result.update(source_kind="new_simulation", engine=self.kind,
                      extension_seconds=extension_seconds, remove_objects=sorted(removed), mass_scales=mass_scales,
                      continue_emission=continue_emission, emission_fluid_id=emission_fluid_id,
                      observed_end_seconds=self.backend.end,
                      replay_check=self.replay_check)
        if any(not np.isfinite(a).all() for a in result.get("geometry", {}).values()):
            raise ValueError("Nonfinite simulated geometry; use retained evidence instead")
        return result

    def _rope(self, scenes, extension, removed, masses, emission, audit, endpoint_motion=None,
              mass_protocol="factual", intervention_seconds=None, initial_state="fitted"):
        from atw.polyworld.rope_controls import mass_controls, mass_intervention_parameters, replay_engine, require_independent_control, stability
        from atw.polyworld.rope_ideal import IdealRopeEngine
        if removed:
            raise ValueError("Rope experiments support mass changes, not topology/body removal")
        allowed = {c['object_id'] for s in scenes for c in mass_controls(s)}
        if set(masses) - allowed:
            raise ValueError(f"Mass edits require a modeled load mass; supported IDs: {sorted(allowed)}")
        trajectories, geometry, pulleys, constraints = {}, {}, [], {}
        fits = {r["rope_id"]: r for r in self.fit["ropes"]}
        if mass_protocol == "pre_event" and intervention_seconds is None:
            edited = [s for s in scenes if any(b.object_id in masses for b in s.bodies)]
            if not edited:
                raise ValueError("pre_event requires a mass edit or an explicit intervention_seconds for its control")
            intervention_seconds = min((s.frames[0] + s.event_frame_offset) / s.fps for s in edited)
        edits = []
        for original in scenes:
            fit = fits[original.rope_id]
            count = original.frame_count + round(extension * original.fps)
            scene = replace(original, frames=tuple(range(original.frames[0], original.frames[0] + count)),
                            target_positions=extend(original.target_positions, count))
            theta, event, scene_edits = mass_intervention_parameters(
                scene, fit, masses, mass_protocol, intervention_seconds, initial_state=initial_state)
            if scene_edits:
                require_independent_control(scenes, scene.rope_id)
            scene = replace(scene, event_frame_offset=event)
            edits.extend(scene_edits)
            engine = replay_engine(scene, fit, self.device)
            drive = None
            if endpoint_motion is not None:
                direction = endpoint_motion['direction']
                if scene.rope_id == endpoint_motion['rope_id']:
                    duration = max((count - 1) / scene.fps, 1 / scene.fps)
                    speed = min(endpoint_motion.get('speed_image_height_per_s', .05),
                                endpoint_motion.get('max_displacement_image_height', .1) / duration)
                    speed *= 1 if direction == "down" else -1
                    drive = (endpoint_motion['endpoint_index'], speed)
            diagnostic = dict(diagnostic=True) if isinstance(engine, IdealRopeEngine) else {}
            rope, wheels = engine.rollout(theta, endpoint_motion=drive, **diagnostic)
            if diagnostic and not engine.last_diagnostics['valid'].all():
                failures = engine.last_diagnostics.get('first_failures', [])
                failure = next((f for f in failures if f), {})
                reason = failure.get('reason', 'length_not_converged')
                time = failure.get('time_seconds')
                when = f" at {time:.6f}s" if time is not None and np.isfinite(time) else ""
                raise ValueError(f"Nonfinite rope rollout: {scene.rope_id}: {reason}{when}; "
                                 "the solver rejected this physical state, not necessarily a numerical overflow. "
                                 "Do not retry the same edit unchanged or query an uncreated rollout.")
            centers = {k:v[0].detach().cpu().numpy() for k,v in engine.predicted_body_centers(rope, wheels).items()}
            if isinstance(engine, IdealRopeEngine) and engine.unilateral_rope:
                constraints[scene.rope_id] = engine.constraint_report()
            points = rope[0].detach().cpu().numpy()
            geometry[scene.rope_id] = points
            trajectories[scene.rope_id] = rows(scene.frames, points.mean(1), "center_xy_normalized")
            for body in scene.bodies:
                value = centers.get(body.object_id, np.broadcast_to(body.initial_center, (count, 2)))
                trajectories[body.object_id] = rows(scene.frames, value, "xy_normalized")
                if body.role == "pulley":
                    pulleys.append(dict(object_id=body.object_id, rope_id=scene.rope_id, radius=body.radius))
        result = dict(trajectories=trajectories, geometry=geometry, pulleys=pulleys, mass_edits=edits,
            rope_constraint_diagnostics=constraints,
            rope_mass_protocol=mass_protocol, intervention_seconds=intervention_seconds,
            rope_initial_state=initial_state,
            readout_start_seconds=intervention_seconds if mass_protocol == "pre_event" else 0.,
            coordinate_system="Image x right, y down; both axes normalized by source image height",
            assumptions=["Retained guide set with zero-wrap release and contact-side-aware recapture; no independent wheel angular dynamics.",
                         "Tension-only rope: slack has zero tension and recovers on re-tautening. Loose shape is not uniquely determined by a massless rope; displayed geometry is its path envelope, and slack is excluded from no-slip spin readouts.",
                         "Dynamic relative masses are effective fitted parameters. Fixed loads use explicitly uncalibrated defaults; mass edits preserve their support constraints."])
        result['assumptions'].append('All initial rope/load velocities are zero; this reset world is not factual replay.'
                                    if initial_state == 'rest' else 'Saved fitted initial motion is retained.')
        if mass_protocol == "pre_event":
            result['assumptions'].append("All factual post-event mass changes are suppressed. Only the requested target changes after the intervention time. Pre-event fitted masses are retained, not guaranteed equilibrium or the benchmark's reset masses. Initial motion follows rope_initial_state.")
        if mass_protocol == 'contphy-reset-v1':
            from atw.polyworld.rope_qa_protocol import description
            result['benchmark_reset'] = description()
            result['assumptions'].append('Explicit ContPhy nominal-mass reset: endpoint loads1, moving-pulley attached loads1.975. These are public generator rules, not fitted masses or instance truth. Fitted geometry/gravity are retained; residual baseline motion is measured, not suppressed.')
        result['simulation_stability'] = stability(scenes, result)
        return result

    def _cloth(self, scenes, extension, removed, masses, emission, audit):
        import warp as wp
        from atw.polyworld.cloth_swr import BatchWarpClothEngine
        allowed = {b.object_id for s in scenes for b in s.obstacles}
        if removed - allowed:
            raise ValueError(f"Cloth removal supports rigid obstacles only: {sorted(allowed)}")
        trajectories, geometry = {}, {}
        for original in scenes:
            fit = self.fit["objects"][original.object_id]
            params, cfg = fit["parameters"], fit["physics_assumptions"]
            if cfg["boundary_mode"] not in {"none", "edge-drive"}:
                raise ValueError("Tracked cloth boundary control is not an independent future experiment")
            count = original.frame_count + round(extension * original.fps)
            scene = replace(original, frames=tuple(range(original.start_frame, original.start_frame+count)),
                target_positions=extend(original.target_positions, count), raw_target_positions=extend(original.raw_target_positions, count),
                target_weights=extend(original.target_weights, count, fill=0),
                obstacles=tuple(b for b in original.obstacles if b.object_id not in removed))
            theta = [*params["initial_velocity_world_m_per_s"], np.log(params["gravity_scale"]),
                     params["structural_stiffness"], params["bending_stiffness"], params["linear_drag_per_s"]]
            if cfg["boundary_mode"] == "edge-drive":
                theta += [*params["edge_velocity_components_m_per_s"], params["edge_stop_offset_frames"], params["edge_release_offset_frames"]]
            if cfg["obstacle_mode"] == "obb-dynamic":
                theta += [np.log(params["rigid_mass_per_median_box_in_cloth_node_masses"])]
            engine = BatchWarpClothEngine(scene, capacity=1, device=self.device,
                substeps=cfg["substeps_per_frame"], constraint_iterations=cfg["constraint_iterations"],
                support_mode=cfg["support_plane_mode"], boundary_mode=cfg["boundary_mode"], obstacle_mode=cfg["obstacle_mode"])
            # Removing a body must not change the surviving bodies' fitted mass reference.
            if removed and scene.obstacles:
                median = np.median([np.prod(b.half_extents) for b in original.obstacles])
                inverse = np.asarray([median/np.prod(b.half_extents) for b in scene.obstacles], np.float32)
                wp.copy(engine.obstacle_inverse_volume, wp.array(inverse, dtype=float, device=self.device))
            points = engine.rollout(np.asarray(theta, np.float32))[0].detach().cpu().numpy()
            geometry[scene.object_id] = points
            trajectories[scene.object_id] = rows(scene.frames, points.mean(1), "center_world")
            if engine.dynamic_obstacles:
                centers = engine.predicted_rigid_centers_torch[0].detach().cpu().numpy()
                rotations = np.stack([getattr(engine, f"predicted_rigid_axis_{a}_torch")[0].detach().cpu().numpy() for a in "xyz"], axis=-1)
                for index, body in enumerate(scene.obstacles):
                    record = rows(scene.frames, centers[:, index], "center_world", fitted_with=scene.object_id)
                    for t, row in enumerate(record):
                        row["rotation_world"] = rotations[t, index].tolist()
                        row["axis_up_alignment"] = np.abs(rotations[t,index].T @ scene.support_normal_world).tolist()
                    trajectories.setdefault(body.object_id, []).extend(record)
        return dict(trajectories=trajectories, geometry=geometry, coordinate_system="Retained 3D world coordinates",
                    assumptions=["Cloths are fitted separately; trajectories tagged fitted_with are separate model predictions, not a jointly coupled world.",
                                 "Original friction, drive/release timing and fitted mass reference are preserved; self/cloth-cloth collision is unsupported."])

    def _soft_body(self, original, extension, removed, masses, emission, audit):
        from atw.polyworld.soft_body_swr import BatchWarpSoftBodyEngine, make_parameter_layout, _contact_events, _numpy_deformation
        allowed = set(original.object_ids) | {o.object_id for o in original.obstacles}
        if removed - allowed:
            raise ValueError(f"Objects are absent from the fitted soft-body world: {sorted(removed-allowed)}")
        selected = [i for i,n in enumerate(original.object_ids) if n not in removed]
        if not selected:
            raise ValueError("At least one soft body must remain")
        count = original.frame_count + round(extension * original.fps)
        changes = dict(object_ids=tuple(original.object_ids[i] for i in selected),
            frames=tuple(range(original.frames[0], original.frames[0]+count)),
            obstacles=tuple(o for o in original.obstacles if o.object_id not in removed))
        for key in ("target_centers", "target_subset_centers", "target_covariances", "target_deformations",
                    "reference_subset_offsets", "reference_subset_covariances",
                    "shape_observed_mask", "shape_contact_recovered_mask", "observed_mask"):
            changes[key] = extend(getattr(original,key)[:,selected], count, fill=0)
        for key in ("initial_centers", "radii", "wall_depths"):
            changes[key] = getattr(original,key)[selected].copy()
        for key in ("first_offsets", "trajectory_paths", "valid_frame_counts", "point_counts"):
            changes[key] = tuple(getattr(original,key)[i] for i in selected)
        scene = replace(original, **changes)
        if not audit:
            # No observed pair normals or trajectory-derived penetration allowances in edited/future motion.
            scene = replace(scene, observed_mask=np.zeros_like(scene.observed_mask), target_centers=np.zeros_like(scene.target_centers))
        engine = BatchWarpSoftBodyEngine(scene, capacity=1, device=self.device, substeps=self.fit["substeps_per_frame"])
        layout = make_parameter_layout(scene.object_count)
        theta = np.zeros(layout.size, np.float32)
        for i,name in enumerate(scene.object_ids):
            p = self.fit["parameters"][name]
            theta[2*i:2*i+2] = p["initial_velocity_plane_m_per_s"]
            for field,key,log in [("log_gravity_scale_slice","gravity_scale",True), ("log_radius_scale_slice","collision_radius_scale",True),
                ("log_frequency_slice","natural_frequency_rad_per_s",True), ("damping_ratio_slice","damping_ratio",False),
                ("friction_slice","contact_friction",False), ("deformation_gain_slice","deformation_gain",False),
                ("log_shape_frequency_slice","shape_frequency_rad_per_s",True), ("shape_damping_ratio_slice","shape_damping_ratio",False),
                ("drag_slice","linear_drag_per_s",False)]:
                theta[getattr(layout,field).start+i] = np.log(p[key]) if log else p[key]
        if not audit:
            active = np.arange(count)[:,None] >= np.asarray(scene.first_offsets)[None]
            engine.scene = replace(scene, observed_mask=active.astype(float))
        predicted, contacts = engine.rollout_numpy(theta)
        trajectories, geometry = {}, {}
        for i,name in enumerate(scene.object_ids):
            world = (predicted[:,i,:1]*scene.wall_lateral_world + predicted[:,i,1:2]*scene.gravity_down_world
                     + scene.wall_depths[i]*scene.wall_normal_world)
            trajectories[name] = rows(scene.frames[scene.first_offsets[i]:], world[scene.first_offsets[i]:], "center_world")
            for row, value in zip(trajectories[name], predicted[scene.first_offsets[i]:,i]):
                row["plane_lateral_down_m"] = value[:2].tolist()
            with np.load(scene.trajectory_paths[i]) as archive:
                reference_frame = int(np.asarray(archive["reference_frame"]).item())
                reference_world = np.asarray(archive["model_xyz"][reference_frame], dtype=np.float64)
            valid = np.isfinite(reference_world).all(axis=1)
            reference_world = reference_world[valid]
            if len(reference_world) > 2048:
                reference_world = reference_world[np.linspace(0, len(reference_world)-1, 2048).astype(int)]
            reference_plane = np.stack((reference_world @ scene.wall_lateral_world,
                                        reference_world @ scene.gravity_down_world), axis=-1)
            plane_offsets = reference_plane - reference_plane.mean(axis=0)
            depth_offsets = reference_world @ scene.wall_normal_world - scene.wall_depths[i]
            deformation = _numpy_deformation(predicted[:, i])
            deformed_plane = predicted[:, i, None, :2] + np.einsum(
                "tij,pj->tpi", deformation, plane_offsets)
            geometry[name] = (deformed_plane[..., :1] * scene.wall_lateral_world
                + deformed_plane[..., 1:2] * scene.gravity_down_world
                + (scene.wall_depths[i] + depth_offsets)[None, :, None] * scene.wall_normal_world)
        return dict(trajectories=trajectories, geometry=geometry, coordinate_system="Retained 3D world coordinates",
            events=[dict(event,time_s=event["frame_index"]/scene.fps) for event in _contact_events(scene,contacts)],
            obstacles=[dict(object_id=o.object_id, center_plane=o.center, half_extents=o.half_extents) for o in scene.obstacles],
            assumptions=["Fixed effective soft-body parameters and static walls; surviving balls retain their modeled entry states.",
                         "New experiments use physical contact separation without observed pair-normal guidance or fitted penetration allowances; edited motion may diverge from the saved fit."])

    def _fluid(self, original, extension, removed, masses, emission, audit,
               emission_fluid_id=None):
        import warp as wp
        from atw.polyworld.fluid_pbf import BatchPBF, PBFConfig, estimate_flux
        source_ids = {e.object_id for e in original.emitters}
        allowed = set(self.backend.objects) - set(original.fluid_ids) - source_ids
        if removed - allowed:
            raise ValueError("Fluid removal supports static obstacles, not sources or fluid identities")
        count = original.frame_count + round(extension * original.fps)
        obstacle, collision = original.obstacle_mask.copy(), original.collision_obstacle_mask.copy()
        for name in removed:
            mask = self.backend._mask(name, 0)
            mask = cv2.resize(mask.astype(np.uint8), (original.image_width, original.image_height),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            obstacle[mask], collision[mask] = False, False
        if removed and original.boundary_source:
            with np.load(Path(original.boundary_source)/"boundary_mask.npz") as archive:
                wall = cv2.resize((archive["labels"]!=0).astype(np.uint8), (original.image_width,original.image_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            obstacle |= wall
            collision |= wall
        predictive_stop = min(count, original.frame_count + round(.5 * original.fps))
        emitters = tuple(replace(e, active_frames=tuple(sorted(set(e.active_frames) | set(
            range(original.frame_count, count if emission else predictive_stop)))))
            if e.active_frames and ((emission and e.active_frames[-1] >= original.frame_count-2)
                                    or e.fluid_id == emission_fluid_id)
            else e for e in original.emitters)
        scene = replace(original, target_masks=extend(original.target_masks,count,fill=False), emitters=emitters,
                        obstacle_mask=obstacle, collision_obstacle_mask=collision)
        config = PBFConfig(**(self.fit["optimization"]["config"] | dict(population=1)))
        flux = estimate_flux(original)
        engine = BatchPBF(scene, config, self.device, capacity_flux=flux)
        _, births = engine.rollout(np.asarray(self.fit["parameter_vector"],np.float32))
        points = wp.to_torch(engine.prediction)[0].detach().cpu().numpy()
        trajectories, geometry = {}, {}
        birth_frames = (births[0]//config.substeps).astype(int)
        for i,name in enumerate(scene.fluid_ids):
            mask = engine.group_np == i
            xy, born = points[:,mask], birth_frames[mask]
            geometry[name] = xy
            record = []
            for frame in range(count):
                active = xy[frame,born<=frame]
                if len(active):
                    record.append(dict(frame_index=frame,center_xy_px=active.mean(0).tolist(),
                        bounds_xyxy_px=[*active.min(0).tolist(),*active.max(0).tolist()],active_particles=len(active)))
            trajectories[name] = record
        return dict(trajectories=trajectories, geometry=geometry, particle_birth=birth_frames,
            particle_fluid=engine.group_np.copy(), coordinate_system="SWR image pixels; x right, y down", image_wh=self.fit["image_wh"],
            assumptions=["Fixed PBF parameters, collision masks and observed emission schedule; no refit.",
                         ("Extra post-cutoff material is an assumed source change, separate from object edits. For an edit to the observed setup, also compare with continue_emission=false unless future emission is explicitly requested."
                          if emission else "The selected target's active source continues briefly for prediction; all other sources keep the fitted schedule, and existing material continues moving."
                          if emission_fluid_id else "Only the fitted observed emission schedule is replayed; existing material continues moving.")])

    def check_replay(self):
        result = self.run(0)
        differences = []
        for name, predicted in result["trajectories"].items():
            saved = {(r["frame_index"], r.get("fitted_with")):r for r in self.backend._saved_trajectory(name)}
            for row in predicted:
                baseline = saved.get((row["frame_index"],row.get("fitted_with")))
                if baseline:
                    key = next((k for k in ("center_world","xy_normalized","center_xy_normalized","center_xy_px") if k in row and k in baseline), None)
                    if key:
                        differences.append(float(np.linalg.norm(np.asarray(row[key])-baseline[key])))
        scale = self.backend.height if self.kind == "rope" else 1.
        maximum = max(differences, default=float("inf"))*scale
        tolerance = 2. if self.kind in {"rope","fluid"} else .002
        self.replay_check = dict(status="matched" if maximum <= tolerance else "limited", compared_samples=len(differences),
            maximum_center_error=maximum if np.isfinite(maximum) else None, units="pixels" if self.kind in {"rope","fluid"} else "meters",
            interpretation="Unedited experiment dynamics compared with the retained fit; not validation of unseen outcomes")
        return self.replay_check
