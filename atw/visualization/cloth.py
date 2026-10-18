"""Build cloth display geometry without modifying input arrays."""

import numpy as np


def intersects(tri, ext):
    hit = np.ones(len(tri), bool)
    axes = [np.tile(v, (len(tri), 1)) for v in np.eye(3)]
    edges = np.roll(tri, -1, axis=1) - tri
    axes.append(np.cross(edges[:, 0], edges[:, 1]))
    for e in range(3):
        for v in np.eye(3):
            axes.append(np.cross(edges[:, e], v))
    for a in axes:
        p = np.einsum("ntj,nj->nt", tri, a)
        r = np.abs(a) @ ext
        hit &= (p.min(1) < r - 1e-10) & (p.max(1) > -r + 1e-10) | (
            np.linalg.norm(a, axis=1) < 1e-12
        )
    return hit


def separate_along(points, faces, bodies, direction):
    direction = np.array(direction, dtype=float, copy=True)
    direction /= np.linalg.norm(direction)
    points = points.copy()
    constraints = []
    for T, inv, center, extent in bodies:
        local_direction = inv[:3, :3] @ direction
        axis = int(np.argmax(np.abs(local_direction)))
        target = np.sign(local_direction[axis]) * (extent[axis] + 0.00005)
        constraints.append((inv, center, extent, local_direction, axis, target))
    for _ in range(len(points) * max(1, len(bodies)) + 1):
        changed = False
        for inv, center, extent, local_direction, axis, target in constraints:
            local = points @ inv[:3, :3].T + inv[:3, 3] - center
            hit = intersects(local[faces], extent)
            if not hit.any():
                continue
            ids = np.unique(faces[hit])
            distance = np.maximum(
                (target - local[ids, axis]) / local_direction[axis], 0
            )
            points[ids] += distance[:, None] * direction
            changed = True
        if not changed:
            return points
    raise RuntimeError("Cloth display separation did not converge")


def prepare(info, z):
    result = {}
    report = []
    for obj in info["objects"]:
        if obj["kind"] != "cloth":
            continue
        key = obj["key"]
        vs = z[key + "_vertices_frames"].copy()
        faces = z[key + "_faces"].copy()
        for level in range(2):
            edges = {}
            pairs = []
            new = []
            nv = vs.shape[1]

            def mid(a, b):
                edge = tuple(sorted((int(a), int(b))))
                if edge not in edges:
                    edges[edge] = nv + len(pairs)
                    pairs.append(edge)
                return edges[edge]

            for a, b, c in faces:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                new.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
            pairs = np.array(pairs)
            vs = np.concatenate(
                [vs, (vs[:, pairs[:, 0]] + vs[:, pairs[:, 1]]) / 2], axis=1
            )
            faces = np.array(new)
        for f in range(info["start"], info["frames"]):
            points = vs[f].copy()
            before = points.copy()
            bodies = []
            if not np.isfinite(points).all():
                continue
            for b in info["objects"]:
                if b["kind"] != "mesh":
                    continue
                k = b["key"]
                T = z[k + "_poses"][f]
                if not np.isfinite(T).all():
                    continue
                v = z[k + "_vertices"]
                center = (v.min(0) + v.max(0)) / 2
                ext = (v.max(0) - v.min(0)) / 2 + 0.0015
                bodies.append((T, np.linalg.inv(T), center, ext))
            initial = 0
            for iteration in range(12):
                count = 0
                for T, inv, c, e in bodies:
                    local = points @ inv[:3, :3].T + inv[:3, 3] - c
                    tri = local[faces]
                    ids = np.flatnonzero(intersects(tri, e))
                    count += len(ids)
                    if iteration == 0:
                        initial += len(ids)
                    for idx in ids:
                        ids3 = faces[idx]
                        t = local[ids3]
                        if not intersects(t[None], e)[0]:
                            continue
                        distances = np.concatenate([e - t.min(0), e + t.max(0)])
                        choice = int(np.argmin(distances))
                        axis = choice % 3
                        if choice < 3:
                            local[ids3, axis] = np.maximum(
                                local[ids3, axis], e[axis] + 0.00005
                            )
                        else:
                            local[ids3, axis] = np.minimum(
                                local[ids3, axis], -e[axis] - 0.00005
                            )
                    points = (local + c) @ T[:3, :3].T + T[:3, 3]
                if count == 0:
                    break
            remaining = 0
            monotone_separation = False
            for T, inv, c, e in bodies:
                remaining += int(
                    intersects(
                        (points @ inv[:3, :3].T + inv[:3, 3] - c)[faces], e - 0.0002
                    ).sum()
                )
            if remaining:
                _, _, basis = np.linalg.svd(
                    before - before.mean(0), full_matrices=False
                )
                directions = [*np.eye(3), *(-np.eye(3)), basis[-1], -basis[-1]]
                candidates = [
                    separate_along(points, faces, bodies, d) for d in directions
                ]
                points = min(
                    candidates, key=lambda p: float(np.square(p - before).sum())
                )
                remaining = sum(
                    int(
                        intersects(
                            (points @ inv[:3, :3].T + inv[:3, 3] - c)[faces], e - 0.0002
                        ).sum()
                    )
                    for T, inv, c, e in bodies
                )
                monotone_separation = True
                if remaining:
                    raise RuntimeError(
                        (obj["name"], f, "unresolved triangles", remaining)
                    )
            vs[f] = points
            move = np.linalg.norm(points - before, axis=1)
            report.append(
                {
                    "cloth": obj["name"],
                    "frame": f,
                    "initial_intersections": initial,
                    "remaining_intersections": remaining,
                    "moved_vertices": int((move > 1e-7).sum()),
                    "max_displacement_m": float(move.max()),
                    "iterations": iteration + 1,
                    "monotone_separation": monotone_separation,
                }
            )
        result[key + "_vertices_frames"] = vs
        result[key + "_faces"] = faces

    return result, report
