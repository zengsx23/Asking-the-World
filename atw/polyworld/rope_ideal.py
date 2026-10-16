"""Massless tension-only rope with frictionless, massless circular pulley guides.

Only endpoint/attached loads carry inertia. A global path-length constraint
replaces massive rope particles. Contacts are inelastic and frictionless;
Guides may release/recover at zero wrap. Slack has zero tension until re-tautening;
its geodesic envelope is not a reconstruction of the undetermined loose shape.
"""
from __future__ import annotations

import numpy as np
import torch

from atw.polyworld.rope_swr import BatchWarpRopeEngine, DEFAULT_MASSLESS_SUBSTEPS


class IdealRopeEngine:
    predicted_body_centers = BatchWarpRopeEngine.predicted_body_centers

    def __init__(self, scene, *, visits, capacity, device="cpu", substeps=DEFAULT_MASSLESS_SUBSTEPS,
                 constraint_iterations=4, contact_model="load-guide-v1", contact_iterations=12,
                 unilateral_rope=None, **unused):
        if capacity < 1 or substeps < 1 or constraint_iterations < 1:
            raise ValueError("Positive capacity and solver iterations required")
        self.scene, self.capacity = scene, capacity
        self.torch_device = torch.device(device)
        self.substeps, self.constraint_iterations = substeps, constraint_iterations
        if contact_model not in {"none", "load-guide-v1"} or contact_iterations < 1:
            raise ValueError("Invalid ideal rope contact model")
        self.contact_model, self.contact_iterations = contact_model, contact_iterations
        self.unilateral_rope = contact_model != "none" if unilateral_rope is None else bool(unilateral_rope)
        self.visits = tuple(tuple(v) for v in visits)
        self.pulleys = tuple(b for b in scene.bodies if b.role == "pulley")
        lookup = {b.object_id: i + 2 for i, b in enumerate(self.pulleys)}
        if any(name not in lookup or sign not in (-1, 1) or turns < 0
               for name, sign, turns in self.visits):
            raise ValueError("Invalid ideal rope route")
        self.order = np.array([0, *[lookup[n] for n, _, _ in self.visits], 1])
        self.sign = np.array([1, *[s for _, s, _ in self.visits], 1])
        self.turns = np.array([t for _, _, t in self.visits])
        self.initial = np.array([scene.initial_positions[0], scene.initial_positions[-1],
                                 *[b.initial_center for b in self.pulleys]], dtype=float)
        radii = np.array([0., 0., *[b.radius for b in self.pulleys]])
        self.guide_radii = radii[2:]
        self.radii = radii[self.order]
        self.mass_slots = np.full(len(self.initial), -1, dtype=int)
        slot = 4
        for i, pinned in enumerate(scene.endpoint_pinned):
            if not pinned:
                loads = [b for b in scene.bodies if b.role == "endpoint" and b.endpoint_index == i]
                if not loads:
                    raise ValueError("A free massless rope end needs an explicit load")
                self.mass_slots[i], slot = slot, slot + 2
        for i, pulley in enumerate(self.pulleys, 2):
            if pulley.dynamic:
                if pulley.object_id not in {v[0] for v in self.visits}:
                    raise ValueError("Dynamic pulley must occur in the ideal route")
                if not any(b.parent_id == pulley.object_id for b in scene.bodies):
                    raise ValueError("Unloaded massless moving pulley requires a coupled algebraic solver")
                self.mass_slots[i], slot = slot, slot + 2
        self.parameter_count = slot
        from atw.polyworld.rope_contacts import RopeContacts
        self.contacts = RopeContacts(scene, self.pulleys)
        # Global mass scale is unidentifiable; stiffness is not an ideal-rope DOF.
        self.fixed_parameters = {2: 1., **({4: 0.} if slot > 4 else {})}
        self.target = torch.as_tensor(scene.target_positions, dtype=torch.float32, device=device)
        self.body_targets = {b.object_id: torch.as_tensor(b.target_centers, dtype=torch.float32,
                              device=device) for b in scene.bodies if b.dynamic}
        geometry = self.path(self.initial[None])
        if not geometry[-1][0]:
            raise ValueError("Invalid initial tangent geometry")
        self.rest_length = geometry[0][0]
        self.initial_sweep = geometry[5][0].copy()

    @property
    def execution_config(self):
        return dict(solver="ideal-massless", visits=self.visits, rope_mass=0.,
                    pulley_mass=0., pulley_inertia=0.,
                    constraint="unilateral-path-length" if self.unilateral_rope else "global-taut-path-length",
                    unilateral_rope=self.unilateral_rope, slack_geometry="geodesic-envelope-not-material-shape",
                    constraint_iterations=self.constraint_iterations,
                    contact_model=self.contact_model, contact_iterations=self.contact_iterations,
                    contact_restitution=0., contact_friction=0.,
                    guide_release="zero-wrap", winding_state="continuous-signed-angle",
                    endpoint_geometry="load-and-protruding-end-guide", effective_substeps=self.substeps,
                    geometry_samples="material-arclength-when-taut; envelope-when-slack",
                    integration="load-only-split-impulse" if self.unilateral_rope else "load-only-rattle",
                    forward_backend="numpy-batched")

    def constraint_report(self, world=0):
        d = self.last_diagnostics
        return dict(model='unilateral-path-length' if self.unilateral_rope else 'global-taut-path-length',
            max_stretch_normalized=float(d['max_length_error'][world]),
            max_contact_penetration_normalized=float(d['max_contact_penetration'][world]),
            minimum_tension=float(d['minimum_tension'][world]),
            max_slack_length_px=float(d['max_slack_length'][world]*self.scene.image_height),
            slack_length_normalized_per_frame=d['slack_length'][world].tolist(),
            geometry_semantics='Shortest admissible routed path; during slack it is an envelope, not material rope shape.')

    def path(self, positions, active_guides=None, reference_sweep=None, route_signs=None):
        """Exact tangent/arc length and its envelope-theorem Jacobian."""
        c = positions[:, self.order].copy()
        radii = np.broadcast_to(self.radii, c.shape[:2]).copy()
        route_signs = np.broadcast_to(self.sign, c.shape[:2]) if route_signs is None else route_signs
        if active_guides is not None and not active_guides.all():
            # Inactive guides become zero-radius points on the common tangent
            # of their active neighbors. They contribute neither length nor force.
            for pattern in np.unique(active_guides, axis=0):
                rows = np.flatnonzero(np.all(active_guides == pattern, axis=1))
                active = np.r_[True, pattern, True]
                radii[np.ix_(rows, np.flatnonzero(~active))] = 0.
                keep = np.flatnonzero(active)
                for left, right in zip(keep[:-1], keep[1:]):
                    if right == left+1:
                        continue
                    d = c[rows, right]-c[rows, left]
                    length = np.linalg.norm(d, axis=1)
                    offset = route_signs[rows, left]*self.radii[left]-route_signs[rows, right]*self.radii[right]
                    phi = np.arctan2(d[:, 1], d[:, 0])+np.arcsin(np.clip(offset/np.maximum(length, 1e-12), -1., 1.))
                    normal = np.stack((-np.sin(phi), np.cos(phi)), axis=-1)
                    start = c[rows, left]-(route_signs[rows, left]*self.radii[left])[:, None]*normal
                    end = c[rows, right]-(route_signs[rows, right]*self.radii[right])[:, None]*normal
                    for index in range(left+1, right):
                        c[rows, index] = start+(end-start)*(index-left)/(right-left)
        delta = np.diff(c, axis=1)
        distance = np.linalg.norm(delta, axis=-1)
        offset = np.diff(-route_signs * radii, axis=1)
        if self.contact_model == "none":
            valid = np.all(distance > np.abs(offset) + 1e-9, axis=1)
        else:
            # Opposite-side guides have a zero-length internal tangent when
            # they touch. This is the closed boundary of the feasible path,
            # not overlap. Match the position solver's numerical tolerance.
            valid = np.all((distance > 1e-12) & (distance >= np.abs(offset)-1e-10), axis=1)
        for i in range(len(self.pulleys)):
            for j in range(i):
                valid &= np.linalg.norm(positions[:, i+2]-positions[:, j+2], axis=1) >= (
                    self.guide_radii[i]+self.guide_radii[j]-1e-9)
        phi = np.arctan2(delta[..., 1], delta[..., 0]) + np.arcsin(
            np.clip(offset / np.maximum(distance, 1e-12), -1., 1.))
        tangent = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
        normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
        starts = c[:, :-1] - (route_signs[:, :-1] * radii[:, :-1])[..., None] * normal
        ends = c[:, 1:] - (route_signs[:, 1:] * radii[:, 1:])[..., None] * normal
        line_lengths = np.sqrt(np.maximum(distance**2 - offset**2, 0.))
        a0 = np.arctan2((ends[:, :-1] - c[:, 1:-1])[..., 1],
                        (ends[:, :-1] - c[:, 1:-1])[..., 0])
        a1 = np.arctan2((starts[:, 1:] - c[:, 1:-1])[..., 1],
                        (starts[:, 1:] - c[:, 1:-1])[..., 0])
        signs = route_signs[:, 1:-1]
        sweep = signs * (np.remainder(signs * (a1 - a0), 2*np.pi) + 2*np.pi*self.turns)
        if reference_sweep is not None:
            # Winding is a continuous state, not a fresh modulo choice each step.
            sweep += 2*np.pi*np.floor((reference_sweep-sweep)/(2*np.pi)+.5)
        if active_guides is not None:
            sweep = np.where(active_guides, sweep, 0.)
        arc_lengths = np.abs(sweep) * radii[:, 1:-1]
        lengths = np.zeros((len(c), 2*len(self.visits)+1))
        lengths[:, ::2], lengths[:, 1::2] = line_lengths, arc_lengths
        jac = np.zeros_like(positions)
        for i in range(len(tangent[0])):
            jac[:, self.order[i]] -= tangent[:, i]
            jac[:, self.order[i+1]] += tangent[:, i]
        return lengths.sum(1), jac, starts, ends, a0, sweep, lengths, c, radii, valid

    def material_points(self, positions, geometry):
        length, _, starts, ends, a0, sweep, lengths, centers, radii, _ = geometry
        cumulative = np.cumsum(lengths, axis=1)
        coordinates = length[:, None] * np.linspace(0., 1., self.scene.node_count)
        piece = (coordinates[:, :, None] > cumulative[:, None, :]).sum(-1)
        piece = np.minimum(piece, lengths.shape[1]-1)
        before = np.concatenate((np.zeros((len(length), 1)), cumulative[:, :-1]), axis=1)
        fraction = (coordinates - np.take_along_axis(before, piece, 1)) / np.maximum(
            np.take_along_axis(lengths, piece, 1), 1e-12)
        result = np.zeros((len(length), self.scene.node_count, 2))
        for j in range(lengths.shape[1]):
            i = j//2
            if j % 2 == 0:
                xy = starts[:, i, None] + fraction[..., None]*(ends-starts)[:, i, None]
            else:
                angle = a0[:, i, None] + fraction * sweep[:, i, None]
                xy = centers[:, i+1, None] + radii[:, i+1, None, None] * np.stack(
                    (np.cos(angle), np.sin(angle)), axis=-1)
            result = np.where((piece == j)[..., None], xy, result)
        return result

    def rollout(self, values, end_offset=None, *, endpoint_motion=None, diagnostic=False):
        theta = np.atleast_2d(torch.as_tensor(values).detach().cpu().numpy()).astype(float)
        if len(theta) > self.capacity or theta.shape[1] != self.parameter_count or not np.isfinite(theta).all():
            raise ValueError("Invalid ideal rope parameter batch")
        if endpoint_motion is not None:
            raise ValueError("Ideal rope endpoint actuator is not implemented")
        final = self.scene.frame_count-1 if end_offset is None else min(int(end_offset), self.scene.frame_count-1)
        if final < 0:
            raise ValueError("Negative final frame")
        count = len(theta)
        q = np.broadcast_to(self.initial, (count, *self.initial.shape)).copy()
        velocity = np.zeros_like(q)
        geometry = self.path(q)
        # Initial material flow is carried by the loads, not by rope inertia.
        for endpoint, index in ((0, 0), (1, -1)):
            if self.mass_slots[endpoint] >= 0:
                tangent = geometry[3][:, index] - geometry[2][:, index]
                tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
                velocity[:, endpoint] = theta[:, 3, None] * tangent
        dynamic = self.mass_slots >= 0
        inverse = np.zeros((count, len(q[0])))
        inverse[:, dynamic] = np.exp(-theta[:, self.mass_slots[dynamic]])
        jac = geometry[1]
        normal_speed = (jac*velocity).sum((1, 2)) / np.maximum(
            (inverse[..., None]*jac**2).sum((1, 2)), 1e-30)
        if self.unilateral_rope:
            normal_speed = np.maximum(normal_speed, 0.)
        velocity -= inverse[..., None]*jac*normal_speed[:, None, None]
        rope = np.zeros((count, final+1, self.scene.node_count, 2))
        pulleys = np.zeros((count, final+1, len(self.pulleys), 2))
        valid = np.ones(count, bool)
        max_error, min_tension = np.zeros(count), np.full(count, np.inf)
        max_slack = np.zeros(count)
        slack_history = np.zeros((count, final+1))
        max_penetration = np.zeros(count)
        first_contact = np.full(count, np.nan)
        contact_enabled = self.contact_model != "none" and bool(self.contacts.pairs)
        dt = 1. / (self.scene.fps*self.substeps)
        first_failures = [None] * count
        active_guides = np.ones((count, len(self.visits)), bool)
        route_signs = np.broadcast_to(self.sign, (count, len(self.sign))).copy()
        previous_sweep = geometry[5].copy()
        route_events = []
        # A guide cannot pull the rope toward itself after its wrap reaches
        # zero. Released moving guides retain their attached load's inertia.
        def current_path(state, time_seconds):
            result = self.path(state, active_guides if contact_enabled else None,
                               previous_sweep if contact_enabled else None, route_signs)
            if contact_enabled and len(self.visits):
                unwrapped = active_guides & (result[5]*route_signs[:, 1:-1] < 0.)
                if unwrapped.any():
                    for world, guide in np.argwhere(unwrapped):
                        route_events.append(dict(world=int(world), object_id=self.visits[guide][0],
                                                 time_seconds=float(time_seconds), action="detach_zero_wrap"))
                    active_guides[unwrapped] = False
                    result = self.path(state, active_guides, previous_sweep, route_signs)
                # A released guide may be encountered again from either side;
                # never invent an extra full winding at the new contact.
                for guide in range(len(self.visits)):
                    inactive = ~active_guides[:, guide]
                    if not inactive.any():
                        continue
                    k = guide+1
                    start, end = result[2][:, k-1], result[3][:, k]
                    line = end-start
                    center = state[:, self.order[k]]
                    fraction = np.clip(((center-start)*line).sum(1)/np.maximum((line**2).sum(1), 1e-20), 0., 1.)
                    gap = np.linalg.norm(start+fraction[:, None]*line-center, axis=1)-self.radii[k]
                    hit = inactive & (gap < -1e-8)
                    if hit.any():
                        candidate_active = active_guides.copy()
                        candidate_active[hit, guide] = True
                        candidate_signs = route_signs.copy()
                        if self.unilateral_rope:
                            # Once fully detached, the strand may return from the
                            # other side. Reusing the old side creates a wrap jump.
                            relative = center-start
                            side = line[:, 0]*relative[:, 1]-line[:, 1]*relative[:, 0]
                            candidate_signs[hit, k] = np.where(side[hit] >= 0., 1, -1)
                        candidate = self.path(state, candidate_active, previous_sweep, candidate_signs)
                        angle = candidate[5][:, guide]*candidate_signs[:, k]
                        restore = hit & (angle >= 0.) & (angle < np.pi)
                        for world in np.flatnonzero(restore):
                            route_events.append(dict(world=int(world), object_id=self.visits[guide][0],
                                                     time_seconds=float(time_seconds), action="restore_zero_wrap"))
                        active_guides[restore, guide] = True
                        route_signs[restore, k] = candidate_signs[restore, k]
                        result = self.path(state, active_guides, previous_sweep, route_signs)
                        result[-1][hit & ~restore] = False
            return result

        def record_failure(mask, reason, state, geometry, time_seconds, phase, tension=None):
            if not diagnostic:
                return
            for i in np.flatnonzero(mask):
                if first_failures[i] is not None:
                    continue
                centers = state[i, self.order]
                margin = np.linalg.norm(np.diff(centers, axis=0), axis=1) - np.abs(
                    np.diff(-self.sign*self.radii))
                guides = []
                for a in range(len(self.pulleys)):
                    for b in range(a):
                        gap = np.linalg.norm(state[i, a+2]-state[i, b+2]) - (
                            self.guide_radii[a]+self.guide_radii[b])
                        guides.append(dict(objects=[self.pulleys[a].object_id, self.pulleys[b].object_id],
                                           gap_px=float(gap*self.scene.image_height)))
                first_failures[i] = dict(reason=reason, time_seconds=float(time_seconds), phase=phase,
                    tangent_margins_px=(margin*self.scene.image_height).tolist(),
                    guide_gaps=guides, positions_normalized=state[i].tolist(),
                    positions_order=["endpoint_0", "endpoint_1", *[b.object_id for b in self.pulleys]],
                    length_error_px=float((geometry[0][i]-self.rest_length)*self.scene.image_height),
                    arc_change_rad=(geometry[5][i]-self.initial_sweep).tolist(),
                    arc_step_change_rad=(geometry[5][i]-previous_sweep[i]).tolist(),
                    arc_sweep_rad=geometry[5][i].tolist(),
                    tension=None if tension is None else float(tension[i]))

        for frame in range(final+1):
            if contact_enabled and not valid.any():
                break
            if frame:
                inverse = np.zeros((count, len(q[0])))
                indices = self.mass_slots[dynamic] + int(frame >= self.scene.event_frame_offset)
                inverse[:, dynamic] = np.exp(-theta[:, indices])
                for substep in range(self.substeps):
                    if contact_enabled and not valid.all():
                        # Rejected CEM worlds cannot become valid later. Keep
                        # them out of contact solves and quarantine their state.
                        q[~valid] = self.initial
                        velocity[~valid] = 0.
                        inverse[~valid] = 0.
                        active_guides[~valid] = True
                        route_signs[~valid] = self.sign
                        previous_sweep[~valid] = self.initial_sweep
                    time_seconds = (self.scene.frames[0] + frame-1 + (substep+1)/self.substeps)/self.scene.fps
                    old = q.copy()
                    velocity *= np.exp(-theta[:, 1, None, None] * dt)
                    velocity[:, dynamic, 1] += np.exp(theta[:, 0, None]) * dt
                    if contact_enabled:
                        velocity[~valid] = 0.
                    q += velocity * dt
                    impulse = np.zeros(count)
                    iterations = self.contact_iterations if contact_enabled else self.constraint_iterations
                    contact_impulse = np.zeros((count, len(self.contacts.pairs)))
                    for iteration in range(iterations):
                        geometry = current_path(q, time_seconds)
                        if diagnostic and not contact_enabled:
                            record_failure(~geometry[-1], "invalid_geometry", q, geometry, time_seconds,
                                           f"position_projection_{iteration}")
                        error, jac = geometry[0]-self.rest_length, geometry[1]
                        if contact_enabled:
                            from atw.polyworld.rope_contacts import constrained_correction
                            gaps, normals = self.contacts.geometry(q)
                            delta, multipliers, solved = constrained_correction(inverse, jac,
                                np.where(valid, -error, 0.), normals, -gaps, eligible=np.broadcast_to(valid[:, None], gaps.shape),
                                unilateral_rope=self.unilateral_rope)
                            impulse -= multipliers[:, 0]
                            contact_impulse += multipliers[:, 1:]
                            q += delta
                            valid &= solved
                            if diagnostic:
                                record_failure(~solved, "contact_position_not_converged", q,
                                               current_path(q, time_seconds), time_seconds, f"position_projection_{iteration}")
                            if iteration >= self.constraint_iterations-1 and np.max(np.maximum(error[valid], 0.) if self.unilateral_rope else np.abs(error[valid]), initial=0.) < 1e-10 and np.min(gaps[valid], initial=0.) >= -1e-10:
                                break
                        else:
                            denominator = (inverse[..., None] * jac**2).sum((1, 2))
                            correction = error / np.maximum(denominator, 1e-30)
                            correction = np.where(denominator > 1e-20, correction, 0.)
                            if self.unilateral_rope:
                                correction = np.maximum(correction, 0.)
                            impulse += correction
                            q -= inverse[..., None] * jac * correction[:, None, None]
                            valid &= geometry[-1]
                    geometry = current_path(q, time_seconds)
                    jac = geometry[1]
                    denominator = (inverse[..., None] * jac**2).sum((1, 2))
                    # Split stabilization from physical impulses: contact/length
                    # position repair must not create a launch velocity.
                    if not self.unilateral_rope:
                        velocity = (q-old)/dt
                    if contact_enabled:
                        gaps, normals = self.contacts.geometry(q)
                        delta, _, solved = constrained_correction(inverse, jac, np.where(valid, -(jac*velocity).sum((1, 2)), 0.),
                            normals, -np.einsum('bknd,bnd->bk', normals, velocity), eligible=(gaps <= 1e-7) & valid[:, None],
                            unilateral_rope=self.unilateral_rope, rope_eligible=geometry[0] >= self.rest_length-1e-8)
                        velocity += delta
                        valid &= solved & geometry[-1]
                        penetration = np.maximum(0., -gaps.min(1))
                        max_penetration = np.maximum(max_penetration, penetration)
                        valid &= penetration < 1e-6
                        made_contact = (contact_impulse > 1e-12).any(1)
                        first_contact[np.isnan(first_contact) & made_contact] = time_seconds
                        if diagnostic:
                            record_failure(~geometry[-1], "invalid_geometry", q, geometry, time_seconds, "after_contact_projection")
                            record_failure(penetration >= 1e-6, "contact_not_converged", q, geometry, time_seconds, "after_contact_projection")
                            record_failure(~solved, "contact_velocity_not_converged", q, geometry, time_seconds, "after_contact_projection")
                    else:
                        normal_speed = (jac*velocity).sum((1, 2)) / np.maximum(denominator, 1e-30)
                        if self.unilateral_rope:
                            normal_speed = np.where(geometry[0] >= self.rest_length-1e-8, np.maximum(normal_speed, 0.), 0.)
                        velocity -= inverse[..., None]*jac*normal_speed[:, None, None]
                    # Compression is never a valid rope force. The unilateral
                    # solve releases this constraint before it would push.
                    tension = impulse/dt**2
                    min_tension = np.minimum(min_tension, tension)
                    valid &= tension >= -1e-7
                    if diagnostic:
                        record_failure(tension < -1e-7, "negative_tension", q, geometry,
                                       time_seconds, "after_projection", tension)
                    if len(self.visits):
                        # A continuous arc may change by more than pi over the
                        # rollout. Reject per-step winding jumps, not accumulated
                        # motion; zero-wrap releases were handled by current_path.
                        reference_sweep = previous_sweep if contact_enabled else self.initial_sweep
                        branch_valid = np.all(~active_guides | (np.abs(geometry[5]-reference_sweep) < np.pi), axis=1)
                        valid &= branch_valid
                        if diagnostic:
                            record_failure(~branch_valid, "arc_branch_change", q, geometry,
                                           time_seconds, "after_projection", tension)
                    length_error = (np.maximum(geometry[0]-self.rest_length, 0.) if self.unilateral_rope
                                    else np.abs(geometry[0]-self.rest_length))
                    if diagnostic:
                        record_failure(length_error >= 1e-5,
                                       "length_not_converged", q, geometry,
                                       time_seconds, "after_projection", tension)
                    max_error = np.maximum(max_error, length_error)
                    max_slack = np.maximum(max_slack, self.rest_length-geometry[0])
                    slack_history[:, frame] = np.maximum(slack_history[:, frame], self.rest_length-geometry[0])
                    previous_sweep = geometry[5].copy()
            rope[:, frame] = self.material_points(q, geometry)
            slack_history[:, frame] = np.maximum(slack_history[:, frame], self.rest_length-geometry[0])
            pulleys[:, frame] = q[:, 2:]
        valid &= max_error < 1e-5
        rope[~valid], pulleys[~valid] = np.nan, np.nan
        self.last_diagnostics = dict(valid=valid, max_length_error=max_error,
                                     minimum_tension=min_tension,
                                     max_contact_penetration=max_penetration, first_contact_seconds=first_contact,
                                     route_events=route_events)
        self.last_diagnostics.update(max_slack_length=max_slack, slack_length=slack_history)
        if diagnostic:
            self.last_diagnostics["first_failures"] = first_failures
        return (torch.as_tensor(rope, dtype=torch.float32, device=self.torch_device),
                torch.as_tensor(pulleys, dtype=torch.float32, device=self.torch_device))
