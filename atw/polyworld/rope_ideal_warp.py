"""FP64, load-only ideal rope forward solver.

One Warp thread integrates one CEM world.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from atw.polyworld.rope_ideal import IdealRopeEngine

wp.set_module_options({'enable_backward': False})
D = wp.float64
V = wp.vec2d


@wp.struct
class Model:
    n: int
    unilateral: int
    guides: int
    contacts: int
    order: wp.array(dtype=int)
    signs: wp.array(dtype=D)
    turns: wp.array(dtype=int)
    radius: wp.array(dtype=D)
    initial: wp.array(dtype=V)
    slots: wp.array(dtype=int)
    pairs: wp.array2d(dtype=int)
    offsets: wp.array(dtype=V)
    half: wp.array(dtype=V)
    pair_radius: wp.array(dtype=D)
    guide_radius: wp.array(dtype=D)
    initial_sweep: wp.array(dtype=D)
    rest: D


@wp.struct
class State:
    q: wp.array2d(dtype=V)
    old: wp.array2d(dtype=V)
    velocity: wp.array2d(dtype=V)
    inverse: wp.array2d(dtype=D)
    centers: wp.array2d(dtype=V)
    radii: wp.array2d(dtype=D)
    starts: wp.array2d(dtype=V)
    ends: wp.array2d(dtype=V)
    angles: wp.array2d(dtype=D)
    sweeps: wp.array2d(dtype=D)
    previous: wp.array2d(dtype=D)
    active: wp.array2d(dtype=int)
    signs: wp.array2d(dtype=D)
    lengths: wp.array2d(dtype=D)
    jac: wp.array3d(dtype=V)
    gaps: wp.array2d(dtype=D)
    rhs: wp.array2d(dtype=D)
    eligible: wp.array2d(dtype=int)
    working: wp.array2d(dtype=int)
    multiplier: wp.array2d(dtype=D)
    delta: wp.array2d(dtype=V)
    gram: wp.array3d(dtype=D)
    matrix: wp.array3d(dtype=D)
    eigenvectors: wp.array3d(dtype=D)
    # Per-world diagnostics: valid, length error, tension, penetration, contact time.
    stats: wp.array2d(dtype=D)
    failures: wp.array2d(dtype=D)
    events: wp.array3d(dtype=D)
    event_count: wp.array(dtype=int)


@wp.func
def normal(v: V):
    return V(-v[1], v[0])


@wp.func
def path(m: Model, s: State, w: int):
    g = m.guides
    for k in range(g+2):
        s.centers[w, k] = s.q[w, m.order[k]]
        s.radii[w, k] = m.radius[m.order[k]]
    left = int(0)
    for right in range(1, g+2):
        keep = right == g+1
        if right <= g:
            keep = s.active[w, right-1] != 0
        if keep:
            if right > left+1:
                d = s.centers[w, right]-s.centers[w, left]
                off = s.signs[w, left]*s.radii[w, left]-s.signs[w, right]*s.radii[w, right]
                phi = wp.atan2(d[1], d[0])+wp.asin(wp.clamp(off/wp.max(wp.length(d), D(1e-12)), D(-1.0), D(1.0)))
                norm = V(-wp.sin(phi), wp.cos(phi))
                a = s.centers[w, left]-s.signs[w, left]*s.radii[w, left]*norm
                b = s.centers[w, right]-s.signs[w, right]*s.radii[w, right]*norm
                for k in range(left+1, right):
                    s.centers[w, k] = a+(b-a)*D(k-left)/D(right-left)
                    s.radii[w, k] = D(0.0)
            left = right
    for k in range(m.n):
        s.jac[w, 0, k] = V(D(0.0))
    valid = bool(True)
    length = D(0.0)
    for k in range(g+1):
        d = s.centers[w, k+1]-s.centers[w, k]
        distance = wp.length(d)
        off = s.signs[w, k]*s.radii[w, k]-s.signs[w, k+1]*s.radii[w, k+1]
        valid = valid and distance > D(1e-12) and distance >= wp.abs(off)-D(1e-10)
        phi = wp.atan2(d[1], d[0])+wp.asin(wp.clamp(off/wp.max(distance, D(1e-12)), D(-1.0), D(1.0)))
        tangent = V(wp.cos(phi), wp.sin(phi))
        s.starts[w, k] = s.centers[w, k]-s.signs[w, k]*s.radii[w, k]*normal(tangent)
        s.ends[w, k] = s.centers[w, k+1]-s.signs[w, k+1]*s.radii[w, k+1]*normal(tangent)
        straight = wp.sqrt(wp.max(distance*distance-off*off, D(0.0)))
        s.lengths[w, 2*k] = straight
        length += straight
        node_a = m.order[k]
        node_b = m.order[k+1]
        s.jac[w, 0, node_a] = s.jac[w, 0, node_a]-tangent
        s.jac[w, 0, node_b] = s.jac[w, 0, node_b]+tangent
    for k in range(g):
        a = s.ends[w, k]-s.centers[w, k+1]
        b = s.starts[w, k+1]-s.centers[w, k+1]
        a0 = wp.atan2(a[1], a[0])
        a1 = wp.atan2(b[1], b[0])
        sign = s.signs[w, k+1]
        angle = sign*(a1-a0)
        period = D(6.283185307179586)
        angle = angle-wp.floor(angle/period)*period
        sweep = sign*(angle+period*D(m.turns[k]))
        sweep += period*wp.floor((s.previous[w, k]-sweep)/period+D(0.5))
        if s.active[w, k] == 0:
            sweep = D(0.0)
        s.angles[w, k] = a0
        s.sweeps[w, k] = sweep
        arc = wp.abs(sweep)*s.radii[w, k+1]
        s.lengths[w, 2*k+1] = arc
        length += arc
    for node_a in range(2, m.n):
        for node_b in range(2, node_a):
            valid = valid and wp.length(s.q[w, node_a]-s.q[w, node_b]) >= m.radius[node_a]+m.radius[node_b]-D(1e-9)
    return length, valid and wp.isfinite(length)


@wp.func
def route_event(s: State, w: int, guide: int, step: int, action: int):
    index = s.event_count[w]
    if index < s.events.shape[1]:
        s.events[w, index, 0] = D(step)
        s.events[w, index, 1] = D(guide)
        s.events[w, index, 2] = D(action)
    s.event_count[w] = index+1


@wp.func
def current_path(m: Model, s: State, w: int, step: int):
    length, valid = path(m, s, w)
    changed = bool(False)
    pi = D(3.141592653589793)
    for k in range(m.guides):
        if s.active[w, k] != 0:
            if s.sweeps[w, k]*s.signs[w, k+1] < D(0.0):
                s.active[w, k] = 0
                changed = True
                route_event(s, w, k, step, -1)
    if changed:
        length, valid = path(m, s, w)
    for k in range(m.guides):
        if s.active[w, k] == 0:
            start = s.starts[w, k]
            end = s.ends[w, k+1]
            line = end-start
            center = s.q[w, m.order[k+1]]
            fraction = wp.clamp(wp.dot(center-start, line)/wp.max(wp.dot(line, line), D(1e-20)), D(0.0), D(1.0))
            gap = wp.length(start+fraction*line-center)-m.radius[m.order[k+1]]
            if gap < D(-1e-8):
                previous_sign = s.signs[w, k+1]
                if m.unilateral != 0:
                    relative = center-start
                    side = line[0]*relative[1]-line[1]*relative[0]
                    s.signs[w, k+1] = D(1.0)
                    if side < D(0.0):
                        s.signs[w, k+1] = D(-1.0)
                s.active[w, k] = 1
                length, valid = path(m, s, w)
                angle = s.sweeps[w, k]*s.signs[w, k+1]
                restore = angle >= D(0.0) and angle < pi
                if restore:
                    route_event(s, w, k, step, 1)
                else:
                    s.active[w, k] = 0
                    s.signs[w, k+1] = previous_sign
                length, valid = path(m, s, w)
                valid = valid and restore
    return length, valid


@wp.func
def contacts(m: Model, s: State, w: int):
    for k in range(m.contacts):
        a = m.pairs[k, 0]
        b = m.pairs[k, 1]
        d = s.q[w, b]-s.q[w, a]-m.offsets[k]
        half = m.half[k]
        outside = d
        if m.pairs[k, 2] != 0:
            outside = V(d[0]-wp.clamp(d[0], -half[0], half[0]), d[1]-wp.clamp(d[1], -half[1], half[1]))
        distance = wp.length(outside)
        norm = outside/wp.max(distance, D(1e-15))
        gap = distance-m.guide_radius[k]
        if m.pairs[k, 2] == 0:
            gap -= m.pair_radius[k]
            if distance < D(1e-15):
                norm = V(D(1.0), D(0.0))
        elif distance < D(1e-15):
            axis = int(0)
            if half[1]-wp.abs(d[1]) < half[0]-wp.abs(d[0]):
                axis = 1
            norm = V(D(0.0))
            norm[axis] = D(1.0)
            if d[axis] < D(0.0):
                norm[axis] = D(-1.0)
            gap = -(half[axis]-wp.abs(d[axis]))-m.guide_radius[k]
        s.gaps[w, k] = gap
        for node in range(m.n):
            s.jac[w, k+1, node] = V(D(0.0))
        s.jac[w, k+1, a] = -norm
        s.jac[w, k+1, b] = norm


@wp.func
def pseudoinverse_solve(m: Model, s: State, w: int):
    # Symmetric Jacobi eigensolve reproduces the reference Gram pseudoinverse,
    # including redundant contact rows, without adding artificial compliance.
    size = m.contacts+1
    scale = D(0.0)
    for i in range(size):
        for j in range(size):
            value = s.gram[w, i, j]*D(s.working[w, i]*s.working[w, j])
            if i == j:
                value += D(1-s.working[w, i])
            s.matrix[w, i, j] = value
            s.eigenvectors[w, i, j] = D(0.0)
            if i == j:
                s.eigenvectors[w, i, j] = D(1.0)
                scale = wp.max(scale, wp.abs(value))
    for sweep in range(32):
        largest = D(0.0)
        for p in range(size):
            for q in range(p+1, size):
                off = s.matrix[w, p, q]
                largest = wp.max(largest, wp.abs(off))
                if wp.abs(off) > wp.max(scale*D(1e-15), D(1e-30)):
                    tau = (s.matrix[w, q, q]-s.matrix[w, p, p])/(D(2.0)*off)
                    t = D(1.0)/(wp.abs(tau)+wp.sqrt(D(1.0)+tau*tau))
                    if tau < D(0.0):
                        t = -t
                    c = D(1.0)/wp.sqrt(D(1.0)+t*t)
                    sn = t*c
                    ap = s.matrix[w, p, p]
                    aq = s.matrix[w, q, q]
                    s.matrix[w, p, p] = ap-t*off
                    s.matrix[w, q, q] = aq+t*off
                    s.matrix[w, p, q] = D(0.0)
                    s.matrix[w, q, p] = D(0.0)
                    for k in range(size):
                        if k != p and k != q:
                            x = s.matrix[w, k, p]
                            y = s.matrix[w, k, q]
                            s.matrix[w, k, p] = c*x-sn*y
                            s.matrix[w, p, k] = c*x-sn*y
                            s.matrix[w, k, q] = sn*x+c*y
                            s.matrix[w, q, k] = sn*x+c*y
                        x = s.eigenvectors[w, k, p]
                        y = s.eigenvectors[w, k, q]
                        s.eigenvectors[w, k, p] = c*x-sn*y
                        s.eigenvectors[w, k, q] = sn*x+c*y
        if largest <= wp.max(scale*D(1e-14), D(1e-30)):
            break
    maximum = D(0.0)
    for i in range(size):
        maximum = wp.max(maximum, wp.abs(s.matrix[w, i, i]))
        s.multiplier[w, i] = D(0.0)
    for j in range(size):
        eigenvalue = s.matrix[w, j, j]
        if wp.abs(eigenvalue) > maximum*D(1e-12):
            value = D(0.0)
            for k in range(size):
                value += s.eigenvectors[w, k, j]*s.rhs[w, k]*D(s.working[w, k])
            value /= eigenvalue
            for i in range(size):
                s.multiplier[w, i] = s.multiplier[w, i]+s.eigenvectors[w, i, j]*value


@wp.func
def correction(m: Model, s: State, w: int):
    size = m.contacts+1
    denom = D(0.0)
    for node in range(m.n):
        jacobian = s.jac[w, 0, node]
        denom += s.inverse[w, node]*wp.dot(jacobian, jacobian)
    lam = s.rhs[w, 0]/wp.max(denom, D(1e-30))
    if m.unilateral != 0:
        lam = wp.min(lam, D(0.0))
        if s.eligible[w, 0] == 0:
            lam = D(0.0)
    if m.contacts == 0 and denom <= D(1e-20):
        lam = D(0.0)
    for node in range(m.n):
        s.delta[w, node] = s.inverse[w, node]*s.jac[w, 0, node]*lam
    s.multiplier[w, 0] = lam
    s.working[w, 0] = 1
    if m.unilateral != 0:
        s.working[w, 0] = int(lam < D(0.0))
    needs = bool(False)
    for k in range(1, size):
        violation = s.rhs[w, k]
        for node in range(m.n):
            violation -= wp.dot(s.jac[w, k, node], s.delta[w, node])
        active = s.eligible[w, k] != 0 and violation > D(1e-12)
        s.working[w, k] = int(active)
        s.multiplier[w, k] = D(0.0)
        needs = needs or active
    done = bool(True)
    if needs:
        for i in range(size):
            for j in range(size):
                value = D(0.0)
                for node in range(m.n):
                    value += s.inverse[w, node]*wp.dot(s.jac[w, i, node], s.jac[w, j, node])
                s.gram[w, i, j] = value
        done = bool(False)
        for iteration in range(4*size):
            pseudoinverse_solve(m, s, w)
            worst_negative = D(-1e-10)
            drop = int(-1)
            worst_violation = D(1e-10)
            add = int(-1)
            if m.unilateral != 0:
                if s.working[w, 0] != 0 and s.multiplier[w, 0] > D(1e-10):
                    drop = 0
                    worst_negative = -s.multiplier[w, 0]
                residual = -s.rhs[w, 0]
                for j in range(size):
                    residual += s.gram[w, 0, j]*s.multiplier[w, j]
                if s.eligible[w, 0] != 0 and s.working[w, 0] == 0 and residual > D(1e-10):
                    add = 0
                    worst_violation = residual
            for i in range(1, size):
                if s.working[w, i] != 0 and s.multiplier[w, i] < worst_negative:
                    worst_negative = s.multiplier[w, i]
                    drop = i
                residual = s.rhs[w, i]
                for j in range(size):
                    residual -= s.gram[w, i, j]*s.multiplier[w, j]
                if s.eligible[w, i] != 0 and s.working[w, i] == 0 and residual > worst_violation:
                    worst_violation = residual
                    add = i
            if drop < 0 and add < 0:
                done = True
                break
            if drop >= 0:
                s.working[w, drop] = 0
            elif add >= 0:
                s.working[w, add] = 1
        for node in range(m.n):
            vector = V(D(0.0))
            for i in range(size):
                vector += s.jac[w, i, node]*s.multiplier[w, i]
            s.delta[w, node] = s.inverse[w, node]*vector
        for i in range(size):
            residual = -s.rhs[w, i]
            for node in range(m.n):
                residual += wp.dot(s.jac[w, i, node], s.delta[w, node])
            if i == 0:
                if m.unilateral != 0:
                    done = done and (s.eligible[w, 0] == 0 or residual <= D(1e-8))
                else:
                    done = done and wp.abs(residual) < D(1e-8)
            elif s.eligible[w, i] != 0:
                done = done and residual >= D(-1e-8)
    return done


@wp.kernel
def simulate(m: Model, s: State, theta: wp.array2d(dtype=D), frames: int,
             event_frame: int, substeps: int, dt: D, iterations: int, minimum_iterations: int,
             nodes: int, rope: wp.array3d(dtype=wp.vec2), wheels: wp.array3d(dtype=wp.vec2),
             slack: wp.array2d(dtype=D)):
    w = wp.tid()
    for node in range(m.n):
        s.q[w, node] = m.initial[node]
        s.velocity[w, node] = V(D(0.0))
        slot = m.slots[node]
        inverse = D(0.0)
        if slot >= 0:
            inverse = wp.exp(-theta[w, slot])
        s.inverse[w, node] = inverse
    for k in range(m.guides):
        s.active[w, k] = 1
        s.previous[w, k] = m.initial_sweep[k]
    for k in range(m.guides+2):
        s.signs[w, k] = m.signs[k]
    s.event_count[w] = 0
    s.failures[w, 0] = D(0.0)
    s.failures[w, 1] = D(-1.0)
    length, geometry_valid = path(m, s, w)
    for endpoint in range(2):
        if m.slots[endpoint] >= 0:
            segment = int(0)
            if endpoint == 1:
                segment = m.guides
            t = s.ends[w, segment]-s.starts[w, segment]
            s.velocity[w, endpoint] = theta[w, 3]*t/wp.max(wp.length(t), D(1e-12))
    denom = D(0.0)
    speed = D(0.0)
    for node in range(m.n):
        jacobian = s.jac[w, 0, node]
        denom += s.inverse[w, node]*wp.dot(jacobian, jacobian)
        speed += wp.dot(jacobian, s.velocity[w, node])
    if m.unilateral != 0:
        speed = wp.max(speed, D(0.0))
    for node in range(m.n):
        s.velocity[w, node] = s.velocity[w, node]-s.inverse[w, node]*s.jac[w, 0, node]*speed/wp.max(denom, D(1e-30))
    max_error = D(0.0)
    min_tension = D(1e300)
    max_penetration = D(0.0)
    max_slack = D(0.0)
    first_contact = D(-1.0)
    valid = bool(True)
    for frame in range(frames):
        if frame > 0:
            for node in range(m.n):
                slot = m.slots[node]
                if slot >= 0:
                    slot += int(frame >= event_frame)
                    s.inverse[w, node] = wp.exp(-theta[w, slot])
            for substep in range(substeps):
                step = (frame-1)*substeps+substep+1
                for node in range(m.n):
                    s.old[w, node] = s.q[w, node]
                    v = s.velocity[w, node]*wp.exp(-theta[w, 1]*dt)
                    if m.slots[node] >= 0:
                        v += V(D(0.0), wp.exp(theta[w, 0])*dt)
                    s.velocity[w, node] = v
                    s.q[w, node] = s.q[w, node]+v*dt
                impulse = D(0.0)
                made_contact = bool(False)
                for iteration in range(iterations):
                    length, geometry_valid = current_path(m, s, w, step)
                    error = length-m.rest
                    contacts(m, s, w)
                    s.rhs[w, 0] = -error
                    s.eligible[w, 0] = 1
                    min_gap = D(0.0)
                    for k in range(m.contacts):
                        s.rhs[w, k+1] = -s.gaps[w, k]
                        s.eligible[w, k+1] = 1
                        min_gap = wp.min(min_gap, s.gaps[w, k])
                    solved = correction(m, s, w)
                    impulse -= s.multiplier[w, 0]
                    for k in range(m.contacts):
                        made_contact = made_contact or s.multiplier[w, k+1] > D(1e-12)
                    for node in range(m.n):
                        s.q[w, node] = s.q[w, node]+s.delta[w, node]
                    if not solved:
                        valid = False
                        s.failures[w, 0] = D(1.0)
                        break
                    length_error = wp.abs(error)
                    if m.unilateral != 0:
                        length_error = wp.max(error, D(0.0))
                    if iteration >= minimum_iterations-1 and length_error < D(1e-10) and min_gap >= D(-1e-10):
                        break
                if valid:
                    length, geometry_valid = current_path(m, s, w, step)
                    contacts(m, s, w)
                    if m.unilateral == 0:
                        for node in range(m.n):
                            s.velocity[w, node] = (s.q[w, node]-s.old[w, node])/dt
                    for k in range(m.contacts+1):
                        rhs = D(0.0)
                        for node in range(m.n):
                            rhs -= wp.dot(s.jac[w, k, node], s.velocity[w, node])
                        s.rhs[w, k] = rhs
                        if k == 0:
                            s.eligible[w, k] = int(length >= m.rest-D(1e-8))
                        if k > 0:
                            s.eligible[w, k] = int(s.gaps[w, k-1] <= D(1e-7))
                    solved = correction(m, s, w)
                    for node in range(m.n):
                        s.velocity[w, node] = s.velocity[w, node]+s.delta[w, node]
                    penetration = D(0.0)
                    for k in range(m.contacts):
                        penetration = wp.max(penetration, -s.gaps[w, k])
                    max_penetration = wp.max(max_penetration, penetration)
                    if made_contact and first_contact < D(0.0):
                        first_contact = D(step)*dt
                    tension = impulse/(dt*dt)
                    min_tension = wp.min(min_tension, tension)
                    branch = bool(True)
                    for k in range(m.guides):
                        if s.active[w, k] != 0:
                            branch = branch and wp.abs(s.sweeps[w, k]-s.previous[w, k]) < D(3.141592653589793)
                        s.previous[w, k] = s.sweeps[w, k]
                    length_error = wp.abs(length-m.rest)
                    if m.unilateral != 0:
                        length_error = wp.max(length-m.rest, D(0.0))
                    max_error = wp.max(max_error, length_error)
                    max_slack = wp.max(max_slack, m.rest-length)
                    slack[w, frame] = wp.max(slack[w, frame], m.rest-length)
                    code = int(0)
                    if not solved:
                        code = 2
                    elif not geometry_valid:
                        code = 3
                    elif penetration >= D(1e-6):
                        code = 4
                    elif tension < D(-1e-7):
                        code = 5
                    elif not branch:
                        code = 6
                    if code != 0:
                        valid = False
                        s.failures[w, 0] = D(code)
                if not valid:
                    s.failures[w, 1] = D(step)*dt
                    break
        if not valid:
            break
        slack[w, frame] = wp.max(slack[w, frame], m.rest-length)
        for k in range(nodes):
            coordinate = length*D(k)/D(nodes-1)
            before = D(0.0)
            piece = int(0)
            for j in range(2*m.guides+1):
                piece = j
                if coordinate <= before+s.lengths[w, j] or j == 2*m.guides:
                    break
                before += s.lengths[w, j]
            fraction = (coordinate-before)/wp.max(s.lengths[w, piece], D(1e-12))
            i = piece//2
            point = V(D(0.0))
            if piece%2 == 0:
                point = s.starts[w, i]+fraction*(s.ends[w, i]-s.starts[w, i])
            else:
                angle = s.angles[w, i]+fraction*s.sweeps[w, i]
                point = s.centers[w, i+1]+s.radii[w, i+1]*V(wp.cos(angle), wp.sin(angle))
            rope[w, frame, k] = wp.vec2(wp.float32(point[0]), wp.float32(point[1]))
        for k in range(m.n-2):
            point = s.q[w, k+2]
            wheels[w, frame, k] = wp.vec2(wp.float32(point[0]), wp.float32(point[1]))
    s.stats[w, 0] = D(int(valid and max_error < D(1e-5)))
    s.stats[w, 1] = max_error
    s.stats[w, 2] = min_tension
    s.stats[w, 3] = max_penetration
    s.stats[w, 4] = first_contact
    s.stats[w, 5] = max_slack


class WarpIdealRopeEngine(IdealRopeEngine):
    """Same host API and objective as the NumPy reference, compiled forward only."""

    def __init__(self, scene, *, block_dim=32, **options):
        super().__init__(scene, **options)
        if block_dim not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError('Warp ideal block_dim must be a power of two from 1 to 256')
        self.block_dim = block_dim
        if self.contact_model != 'load-guide-v1':
            raise ValueError('Warp ideal backend requires load-guide-v1; use NumPy for legacy contact-free replay')
        self.wp_device = wp.get_device(str(self.torch_device))
        m = Model()
        m.n, m.guides, m.contacts = len(self.initial), len(self.visits), len(self.contacts.pairs)
        m.unilateral = int(self.unilateral_rope)
        def array(values, dtype):
            return wp.array(np.asarray(values), dtype=dtype, device=self.wp_device)
        m.order = array(self.order, int)
        m.signs = array(self.sign, D)
        m.turns = array(self.turns, int)
        m.radius = array(np.r_[0., 0., self.guide_radii], D)
        m.initial = array(self.initial, V)
        m.slots = array(self.mass_slots, int)
        m.initial_sweep = array(self.initial_sweep, D)
        m.rest = float(self.rest_length)
        pairs = self.contacts.pairs
        m.pairs = array(np.asarray([(a, b, int(half is not None)) for a, b, _, half, _, _, _ in pairs]).reshape(-1, 3), int)
        m.offsets = array(np.asarray([p[2] for p in pairs]).reshape(-1, 2), V)
        m.half = array(np.asarray([p[3] if p[3] is not None else (0., 0.) for p in pairs]).reshape(-1, 2), V)
        m.pair_radius = array([p[4] for p in pairs], D)
        m.guide_radius = array([p[5] for p in pairs], D)
        self.model = m
        s = State()
        b, n, g, k = self.capacity, m.n, m.guides, m.contacts+1
        shapes = dict(q=(b,n), old=(b,n), velocity=(b,n), inverse=(b,n),
                      centers=(b,g+2), radii=(b,g+2), starts=(b,g+1), ends=(b,g+1),
                      angles=(b,g), sweeps=(b,g), previous=(b,g), active=(b,g), signs=(b,g+2), lengths=(b,2*g+1),
                      jac=(b,k,n), gaps=(b,k-1), rhs=(b,k), eligible=(b,k), working=(b,k),
                      multiplier=(b,k), delta=(b,n), gram=(b,k,k), matrix=(b,k,k),
                      eigenvectors=(b,k,k), stats=(b,6), failures=(b,2), events=(b,256,3), event_count=(b,))
        vectors = {'q','old','velocity','centers','starts','ends','jac','delta'}
        integers = {'active','eligible','working','event_count'}
        for name, shape in shapes.items():
            setattr(s, name, wp.empty(shape, dtype=V if name in vectors else int if name in integers else D,
                                     device=self.wp_device))
        self.state = s

    @property
    def execution_config(self):
        return {**super().execution_config, 'forward_backend': 'warp-fp64-worlds',
                'cuda_block_dim': self.block_dim}

    def rollout(self, values, end_offset=None, *, endpoint_motion=None, diagnostic=False):
        if endpoint_motion is not None:
            raise ValueError('Ideal rope endpoint actuator is not implemented')
        theta = torch.as_tensor(values, device=self.torch_device).detach().to(dtype=torch.float64)
        theta = torch.atleast_2d(theta).contiguous()
        count = len(theta)
        if count < 1 or count > self.capacity or theta.shape[1] != self.parameter_count or not torch.isfinite(theta).all():
            raise ValueError('Invalid ideal rope parameter batch')
        final = self.scene.frame_count-1 if end_offset is None else min(int(end_offset), self.scene.frame_count-1)
        if final < 0:
            raise ValueError('Negative final frame')
        rope = torch.empty((count,final+1,self.scene.node_count,2), device=self.torch_device)
        wheels = torch.empty((count,final+1,len(self.pulleys),2), device=self.torch_device)
        slack = torch.zeros((count,final+1), dtype=torch.float64, device=self.torch_device)
        with wp.ScopedStream(wp.stream_from_torch(self.torch_device)) if self.torch_device.type == 'cuda' else wp.ScopedDevice(self.wp_device):
            wp.launch(simulate, dim=count, inputs=[self.model, self.state, wp.from_torch(theta, dtype=D),
                      final+1, self.scene.event_frame_offset, self.substeps, 1./(self.scene.fps*self.substeps),
                      self.contact_iterations, self.constraint_iterations, self.scene.node_count,
                      wp.from_torch(rope, dtype=wp.vec2), wp.from_torch(wheels, dtype=wp.vec2), wp.from_torch(slack, dtype=D)],
                      device=self.wp_device, block_dim=self.block_dim)
            stats = self.state.stats.numpy()[:count].copy()
            failed = self.state.failures.numpy()[:count].copy() if diagnostic else None
            event_count = self.state.event_count.numpy()[:count].copy()
            events = self.state.events.numpy()[:count].copy() if event_count.any() else None
        valid = stats[:,0] != 0
        invalid = torch.as_tensor(~valid, device=self.torch_device)
        rope[invalid], wheels[invalid] = float('nan'), float('nan')
        offset = self.scene.frames[0]/self.scene.fps
        routes = []
        for world, total in enumerate(event_count):
            for step, guide, action in events[world, :min(total,256)] if total else []:
                routes.append(dict(world=world, object_id=self.visits[int(guide)][0],
                                   time_seconds=offset+step/(self.scene.fps*self.substeps),
                                   action='detach_zero_wrap' if action < 0 else 'restore_zero_wrap'))
        self.last_diagnostics = dict(valid=valid, max_length_error=stats[:,1], minimum_tension=stats[:,2],
                                     max_slack_length=stats[:,5], slack_length=slack.cpu().numpy(),
                                     max_contact_penetration=stats[:,3],
                                     first_contact_seconds=np.where(stats[:,4] >= 0, stats[:,4]+offset, np.nan),
                                     route_events=routes, route_events_truncated=bool((event_count > 256).any()))
        if diagnostic:
            reasons = ('none','contact_position_not_converged','contact_velocity_not_converged',
                       'invalid_geometry','contact_not_converged','negative_tension','arc_branch_change')
            self.last_diagnostics['first_failures'] = [None if ok else dict(
                reason=reasons[int(f[0])] if f[0] else 'length_not_converged',
                time_seconds=float(f[1]+offset)) for ok, f in zip(valid, failed)]
        return rope, wheels
