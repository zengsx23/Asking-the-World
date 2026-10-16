"""Explicit controls for retained rope fits, without geometric ID substitution."""
from __future__ import annotations

import numpy as np


def mass_controls(scene):
    controls, parameter = [], 4
    for endpoint, pinned in enumerate(scene.endpoint_pinned):
        if pinned:
            continue
        members = [b.object_id for b in scene.bodies
                   if b.role == "endpoint" and b.endpoint_index == endpoint]
        controls.append(dict(rope_id=scene.rope_id, object_id=members[0] if members else None,
            members=members, role="endpoint", endpoint_index=endpoint,
            parameter_index=parameter, mass_scope="body" if len(members) == 1 else "assembly"))
        parameter += 2
    for body in scene.bodies:
        if body.role != "pulley" or not body.dynamic:
            continue
        members = [body.object_id] + [b.object_id for b in scene.bodies if b.parent_id == body.object_id]
        if scene.mass_model == "massless":
            members = members[1:]
        controls.append(dict(rope_id=scene.rope_id, object_id=body.object_id, members=members,
            role="pulley", parameter_index=parameter,
            mass_scope="body" if len(members) == 1 else "assembly"))
        parameter += 2
    for body in scene.bodies:
        if body.dynamic or body.role not in {"endpoint", "attachment"}:
            continue
        if body.role == "attachment" and not any(
                p.object_id == body.parent_id and not p.dynamic for p in scene.bodies):
            continue
        if not np.isfinite(body.default_relative_mass) or body.default_relative_mass <= 0:
            raise ValueError(f"Default load mass must be positive and finite: {body.object_id}")
        controls.append(dict(rope_id=scene.rope_id, object_id=body.object_id,
            members=[body.object_id], role=body.role, parameter_index=None, mass_scope="body",
            mass_source="default_unidentified", relative_mass=body.default_relative_mass,
            motion_constraint="fixed_parent" if body.parent_id else "pinned_endpoint",
            parent_id=body.parent_id))
    return controls


def require_independent_control(scenes, rope_id):
    owners = {}
    for scene in scenes:
        for body in scene.bodies:
            if body.dynamic:
                owners.setdefault(body.object_id, set()).add(scene.rope_id)
    shared = sorted(name for name, ropes in owners.items() if rope_id in ropes and len(ropes) > 1)
    if shared:
        raise ValueError(f"Shared dynamic bodies {shared} require a coupled rope solver; independent fits cannot propagate this edit")


def control_catalog(scenes):
    ropes = []
    for scene in scenes:
        try:
            require_independent_control(scenes, scene.rope_id)
            supported, reason = True, None
        except ValueError as error:
            supported, reason = False, str(error)
        ropes.append(dict(rope_id=scene.rope_id, intervention_supported=supported, reason=reason,
            mass_model=scene.mass_model,
            massless_objects=([scene.rope_id] + [b.object_id for b in scene.bodies if b.role == "pulley"]
                              if scene.mass_model == "massless" else []),
            endpoints=[dict(endpoint_index=i, xy_normalized=scene.initial_positions[node].tolist(),
                            pinned=bool(scene.endpoint_pinned[i]),
                            object_ids=[b.object_id for b in scene.bodies if b.role == "endpoint" and b.endpoint_index == i])
                       for i, node in enumerate((0, -1))],
            mass_controls=[{k:v for k,v in item.items() if k != "parameter_index"} for item in mass_controls(scene)],
            pulleys=[dict(object_id=b.object_id, center_dynamic=b.dynamic,
                          center_xy_normalized=list(b.initial_center)) for b in scene.bodies if b.role == "pulley"]))
    return dict(ropes=ropes, coordinate_system="Image x right, y down; both axes normalized by image height",
        semantics=dict(endpoint_motion="Prescribed vertical motion, not a mass change. Endpoint 0/1 follows retained material-node order, not left/right. Driving a pinned end replaces its fixed position with an actuator.",
            mass_change="Finite multiplier on a fitted effective mass. Ordinary mass edits and paired controls start with zero initial velocity, without equilibrium calibration. New fits keep mass constant; factual edits apply from the initial frame. Unedited replay retains saved velocity and masses. Legacy factual replay retains both saved epochs. Explicit pre_event retains fitted initial motion and freezes baseline masses until the requested edit time; callers can separately choose a rest initial state.",
            fixed_load="Static loads retain an editable default relative mass of 1, marked default_unidentified. Changing mass preserves the pinned endpoint or fixed-parent constraint, so it does not imply motion or a fitted mass estimate.",
            assembly="Legacy particle fits combine pulley and load mass. Massless fits assign zero mass to all pulleys and ropes: only attached loads carry mass. A single attached load owns the full fitted mass; multiple loads still require a known member_mass_fraction or mass_scope=assembly. Never guess the split.",
            rotation="No-slip estimate from material rope velocity relative to pulley center; not an independently simulated wheel inertia or bearing friction."))


def resolve_endpoint(scenes, object_id, endpoint_index=None):
    if endpoint_index is not None and (isinstance(endpoint_index, bool) or not isinstance(endpoint_index, (int, np.integer))
                                       or endpoint_index not in (0, 1)):
        raise ValueError("endpoint_index must be integer 0 or 1")
    candidates = []
    for scene in scenes:
        if object_id == scene.rope_id:
            if endpoint_index is None:
                raise ValueError("A rope ID requires endpoint_index=0 or 1; query_rope_controls lists their positions")
            indices = [endpoint_index]
        else:
            indices = [b.endpoint_index for b in scene.bodies if b.object_id == object_id and b.role == "endpoint"
                       and (endpoint_index is None or endpoint_index == b.endpoint_index)]
        for index in set(indices):
            candidates.append(dict(control_kind="rope_endpoint", rope_id=scene.rope_id, endpoint_index=index,
                frame_index=int(scene.frames[0]), endpoint_xy_normalized=scene.initial_positions[0 if index == 0 else -1].tolist(),
                binding_source="explicit_rope_endpoint" if object_id == scene.rope_id else "modeled_attachment"))
    if len(candidates) != 1:
        raise ValueError("No unique modeled endpoint for this object. Use query_rope_controls and an explicit rope ID plus endpoint_index; a pulley or its hanging load is not a rope end")
    require_independent_control(scenes, candidates[0]["rope_id"])
    return candidates[0]


def resolve_mass(scenes, object_id, mass_scope="body", member_mass_fraction=None):
    if mass_scope not in {"body", "assembly"}:
        raise ValueError("mass_scope must be body or assembly")
    candidates = [item for scene in scenes for item in mass_controls(scene) if object_id in item["members"]]
    if len(candidates) != 1:
        raise ValueError("No unique modeled load mass for this object; inspect query_rope_controls. Unmodeled objects and massless pulley inertia are not load mass controls")
    control = candidates[0]
    require_independent_control(scenes, control["rope_id"])
    if member_mass_fraction is not None:
        if (mass_scope != "body" or control["mass_scope"] != "assembly"
                or isinstance(member_mass_fraction, bool)
                or not isinstance(member_mass_fraction, (int, float))
                or not np.isfinite(member_mass_fraction) or not 0 < member_mass_fraction < 1):
            raise ValueError("member_mass_fraction requires a body within an assembly and a known fraction strictly between 0 and 1")
        return {k:v for k,v in control.items() if k != "parameter_index"} | dict(
            requested_object_id=object_id, requested_mass_scope="body",
            member_mass_fraction=float(member_mass_fraction), fraction_source="caller_supplied_not_fitted")
    if control["mass_scope"] == "assembly" and mass_scope != "assembly":
        raise ValueError(f"Only the combined mass of {control['members']} is fitted. Supply a known member_mass_fraction for a body edit, or mass_scope=assembly to edit the whole assembly; do not substitute one for the other")
    return {k:v for k,v in control.items() if k != "parameter_index"}


def mass_intervention_parameters(scene, fit, masses, protocol, intervention_seconds, *, initial_state="fitted"):
    """Copy fitted parameters; never leak another factual mass change into a reset experiment."""
    if protocol not in {"factual", "pre_event", "contphy-reset-v1"}:
        raise ValueError("rope_mass_protocol must be factual, pre_event or contphy-reset-v1")
    theta = np.asarray(fit["routing"]["best_parameters"], np.float32).copy()
    if protocol == 'contphy-reset-v1':
        from atw.polyworld.rope_qa_protocol import reset_parameters, metric_scale
        metric_scale(fit, scene.image_height)
        theta = reset_parameters(scene, theta)
    if initial_state not in {'fitted', 'rest'}:
        raise ValueError('rope_initial_state must be fitted or rest')
    if initial_state == 'rest':
        theta[3] = 0.
    event = scene.event_frame_offset
    if intervention_seconds is not None:
        if (protocol != "pre_event" or isinstance(intervention_seconds, bool)
                or not isinstance(intervention_seconds, (int, float)) or not np.isfinite(intervention_seconds)):
            raise ValueError("intervention_seconds requires pre_event and a finite absolute video time")
        event = round(intervention_seconds * scene.fps) - scene.frames[0]
        if not 0 <= event < scene.frame_count - 1:
            raise ValueError("Mass intervention must leave at least one subsequent simulated frame")
    edits = []
    for control in mass_controls(scene):
        index = control["parameter_index"]
        if index is None:
            if control['object_id'] in masses:
                factor = masses[control['object_id']]
                edits.append({k:v for k,v in control.items() if k != "parameter_index"} | dict(
                    factor=factor, edited_relative_mass=control['relative_mass']*factor,
                    interpretation="Mass changed from an uncalibrated default; the saved support constraint is unchanged."))
            continue
        if protocol == "pre_event":
            theta[index + 1] = theta[index]
        if control["object_id"] in masses:
            factor = masses[control["object_id"]]
            theta[index + (protocol == "pre_event"):index + 2] += np.log(factor)
            edits.append({k:v for k,v in control.items() if k != "parameter_index"} | dict(factor=factor))
    return theta, event, edits


def stability(scenes, result):
    strains, displacements, feasible = {}, [], []
    for scene in scenes:
        points = result["geometry"][scene.rope_id]
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite rope rollout")
        rest = np.maximum(np.linalg.norm(np.diff(scene.initial_positions, axis=0), axis=-1), 1e-8)
        strains[scene.rope_id] = float(np.max(np.abs(np.linalg.norm(np.diff(points, axis=1), axis=-1)/rest - 1)))
        displacements.append(float(np.max(np.linalg.norm(points - points[:1], axis=-1))))
        d = result.get('rope_constraint_diagnostics', {}).get(scene.rope_id, {})
        feasible.append(d['max_stretch_normalized'] < 1e-5 and d['max_contact_penetration_normalized'] < 1e-6
                        and d['minimum_tension'] >= -1e-7 if d.get('model') == 'unilateral-path-length'
                        else strains[scene.rope_id] <= .5)
    max_strain, max_displacement = max(strains.values(), default=0.), max(displacements, default=0.)
    limit = min(2*np.hypot(s.image_width, s.image_height)/s.image_height for s in scenes)
    usable = all(feasible) and max_displacement <= limit
    return dict(status="usable" if usable else "unstable_or_stretched", usable_for_downstream_reasoning=bool(usable),
        max_relative_segment_strain=max_strain, per_rope_segment_strain=strains,
        max_displacement_image_height=max_displacement,
        definition="Finite geometry and displacement <= two image diagonals. Unilateral ropes require bounded extension/penetration and nonnegative tension; slack is not compression strain. Legacy solvers require segment strain <= 0.5. No assumed direction after a mass change.")


def replay_engine(scene, fit, device, *, capacity=1):
    from atw.polyworld.rope_swr import BatchWarpRopeEngine
    execution = fit["optimization"].get("execution_config", {})
    options = dict(capacity=capacity, device=device, substeps=fit["optimization"]["substeps"],
        constraint_iterations=fit["optimization"].get("constraint_iterations", 6),
        simulate_static_system=fit["optimization"].get("simulate_static_system", False),
        rope_linear_density=fit["parameters"].get("rope_linear_density"))
    if execution.get("solver") == "ideal-massless":
        from atw.polyworld.rope_ideal import IdealRopeEngine
        engine_class = IdealRopeEngine
        backend = execution.get('forward_backend', 'numpy-batched')
        if backend == 'warp-fp64-worlds':
            from atw.polyworld.rope_ideal_warp import WarpIdealRopeEngine
            engine_class = WarpIdealRopeEngine
            options['block_dim'] = execution.get('cuda_block_dim', 32)
        elif backend != 'numpy-batched':
            raise ValueError(f'Unknown ideal rope forward backend: {backend}')
        options["constraint_iterations"] = execution["constraint_iterations"]
        options["contact_model"] = execution.get("contact_model", "none")
        options["contact_iterations"] = execution.get("contact_iterations", 12)
        options["unilateral_rope"] = execution.get("unilateral_rope")
        return engine_class(scene, visits=execution["visits"], **options)
    if execution.get("solver", "monolithic") == "monolithic":
        return BatchWarpRopeEngine(scene, **options)
    if execution["solver"] != "split":
        raise ValueError("Unknown saved rope execution solver")
    from atw.polyworld.rope_swr_parallel import SplitRopeEngine
    return SplitRopeEngine(scene, **options,
        block_dim=execution.get("cuda_block_dim", 32),
        colored=execution["stretch_order"] == "two-color",
        parallel_static_contacts=execution["contact_order"] == "two-color",
        parallel_static_nodes=execution["contact_order"] == "parallel-nodes-original-edges",
        graph_chunk_frames=execution.get("graph_chunk_frames", 0))
