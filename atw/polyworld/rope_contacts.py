"""Frictionless load/guide contacts for the massless rope's load coordinates."""
from __future__ import annotations

import numpy as np


class RopeContacts:
    def __init__(self, scene, pulleys):
        lookup = {p.object_id: i+2 for i, p in enumerate(pulleys)}
        self.pairs = []
        for body in scene.bodies:
            if body.role not in {"endpoint", "attachment"}:
                continue
            root = body.endpoint_index if body.role == "endpoint" else lookup[body.parent_id]
            for i, pulley in enumerate(pulleys, 2):
                if root == i or not (body.dynamic or pulley.dynamic):
                    continue
                half = body.half_extent if body.shape in {"box", "irregular"} and any(body.half_extent) else None
                self.pairs.append((root, i, np.array(body.endpoint_offset), half,
                                   body.radius, pulley.radius, (body.object_id, pulley.object_id)))
                offset = np.asarray(body.endpoint_offset)
                outside = (np.any(np.abs(offset) > np.asarray(half)) if half is not None
                           else np.linalg.norm(offset) > body.radius)
                if body.role == "endpoint" and outside:
                    # An observed rope attachment may protrude beyond its load's
                    # collision proxy; the endpoint must still admit a tangent.
                    self.pairs.append((root, i, np.zeros(2), None, 0., pulley.radius,
                                       (body.object_id + ":rope_end", pulley.object_id)))
        for i, a in enumerate(pulleys, 2):
            for j, b in enumerate(pulleys[:i-2], 2):
                if a.dynamic or b.dynamic:
                    self.pairs.append((i, j, np.zeros(2), None, a.radius, b.radius,
                                       (a.object_id, b.object_id)))

    def geometry(self, positions):
        count, nodes, _ = positions.shape
        gaps = np.empty((count, len(self.pairs)))
        jac = np.zeros((count, len(self.pairs), nodes, 2))
        for k, (a, b, offset, half, radius, guide_radius, _) in enumerate(self.pairs):
            delta = positions[:, b] - positions[:, a] - offset
            if half is None:
                distance = np.linalg.norm(delta, axis=1)
                normal = delta / np.maximum(distance[:, None], 1e-15)
                normal[distance < 1e-15] = (1., 0.)
                gaps[:, k] = distance-radius-guide_radius
            else:
                # Circle against an axis-aligned box, including the inside case.
                half = np.asarray(half)
                outside = delta-np.clip(delta, -half, half)
                distance = np.linalg.norm(outside, axis=1)
                normal = outside/np.maximum(distance[:, None], 1e-15)
                gaps[:, k] = distance-guide_radius
                inside = distance < 1e-15
                depth = half-np.abs(delta)
                axis = np.argmin(depth, axis=1)
                rows = np.flatnonzero(inside)
                normal[rows] = 0.
                normal[rows, axis[rows]] = np.where(delta[rows, axis[rows]] >= 0., 1., -1.)
                gaps[rows, k] = -depth[rows, axis[rows]]-guide_radius
            jac[:, k, a] = -normal
            jac[:, k, b] = normal
        return gaps, jac


def constrained_correction(inverse, rope_jac, rope_rhs, contact_jac, contact_rhs, eligible=None,
                           *, unilateral_rope=False, rope_eligible=None):
    """Minimum mass-metric correction: one equality and unilateral contacts.

    Active-set solve of J*dq=rhs (rope) and N*dq>=rhs (contacts).
    With unilateral_rope, J*dq<=rhs and the rope multiplier is nonpositive.
    Contact multipliers must be nonnegative. No artificial masses/compliance.
    """
    count, contacts = contact_rhs.shape
    jac = np.concatenate((rope_jac[:, None], contact_jac), axis=1)
    rhs = np.concatenate((rope_rhs[:, None], contact_rhs), axis=1)
    weights = np.repeat(inverse, 2, axis=1)
    flat = jac.reshape(count, contacts+1, -1)
    denom = (rope_jac**2*inverse[..., None]).sum((1, 2))
    lam = np.zeros((count, contacts+1))
    lam[:, 0] = rope_rhs/np.maximum(denom, 1e-30)
    rope_eligible = np.ones(count, bool) if rope_eligible is None else rope_eligible
    if unilateral_rope:
        lam[:, 0] = np.where(rope_eligible, np.minimum(lam[:, 0], 0.), 0.)
    delta = inverse[..., None]*rope_jac*lam[:, 0, None, None]
    if not contacts:
        return delta, lam, np.ones(count, bool)
    eligible = np.ones_like(contact_rhs, bool) if eligible is None else eligible
    violation = contact_rhs-np.einsum('bknd,bnd->bk', contact_jac, delta)
    needs = np.any(eligible & (violation > 1e-12), axis=1)
    converged = np.ones(count, bool)
    if not needs.any():
        return delta, lam, converged
    # Only contact-bearing worlds enter the small dense solve.
    ids = np.flatnonzero(needs)
    a, w, r = flat[ids], weights[ids], rhs[ids]
    matrix = (a*w[:, None]) @ a.transpose(0, 2, 1)
    active = np.zeros((len(ids), contacts+1), bool)
    active[:, 0] = (rope_eligible[ids] & (lam[ids, 0] < 0.)) if unilateral_rope else True
    active[:, 1:] = eligible[ids] & (violation[ids] > 1e-12)
    solution = np.zeros_like(r)
    done = np.zeros(len(ids), bool)
    for _ in range(4*(contacts+1)):
        system = matrix*active[:, :, None]*active[:, None, :]
        diagonal = np.arange(contacts+1)
        system[:, diagonal, diagonal] += (~active).astype(float)
        # Multiple contacts can describe the same normal constraint. A direct
        # solve of the nearly singular Gram matrix creates spurious impulses.
        solution = (np.linalg.pinv(system, rcond=1e-12, hermitian=True)
                    @ (r*active)[..., None])[..., 0]
        negative = active[:, 1:] & (solution[:, 1:] < -1e-10)
        residual = r[:, 1:]-np.einsum('bij,bj->bi', matrix[:, 1:], solution)
        violated = eligible[ids] & ~active[:, 1:] & (residual > 1e-10)
        rope_drop = active[:, 0] & (solution[:, 0] > 1e-10) if unilateral_rope else np.zeros(len(ids), bool)
        rope_add = (rope_eligible[ids] & ~active[:, 0]
                    & (np.einsum('bj,bj->b', matrix[:, 0], solution)-r[:, 0] > 1e-10)) if unilateral_rope else np.zeros(len(ids), bool)
        done = ~negative.any(1) & ~violated.any(1) & ~rope_drop & ~rope_add
        if done.all():
            break
        for row in np.flatnonzero(~done):
            if rope_drop[row]:
                active[row, 0] = False
            elif negative[row].any():
                col = np.argmin(np.where(negative[row], solution[row, 1:], np.inf))+1
                active[row, col] = False
            elif rope_add[row]:
                active[row, 0] = True
            else:
                col = np.argmax(np.where(violated[row], residual[row], -np.inf))+1
                active[row, col] = True
    lam[ids] = solution
    delta[ids] = (w*np.einsum('bkd,bk->bd', a, solution)).reshape(len(ids), -1, 2)
    residual = np.einsum('bkd,bd->bk', flat, delta.reshape(count, -1))-rhs
    converged[ids] = done
    converged &= (~rope_eligible | (residual[:, 0] <= 1e-8)) if unilateral_rope else (np.abs(residual[:, 0]) < 1e-8)
    converged &= np.all(~eligible | (residual[:, 1:] >= -1e-8), axis=1)
    return delta, lam, converged
