"""Omit redundant connector solves while retaining observed object identities."""
import numpy as np


def find_local_attachment_connectors(scenes, objects, tracks):
    """Find redundant local connectors independently of their physical labels.

    Classification as rigid must not add a second load for a connector between
    an already attached load and its pulley. Require the existing co-moving
    relation, thin mask geometry, and both ends inside that local bridge in at
    least 95% of jointly observed frames. Never remove endpoints or pulleys.
    """
    protected = {b.object_id for s in scenes for b in s.bodies if b.role != 'attachment'}
    protected.update(s.rope_id for s in scenes if s.wrap_hints)
    omitted = {}
    for scene in scenes:
        bodies = {b.object_id: b for b in scene.bodies}
        attachments = [b for b in scene.bodies if b.role == 'attachment'
                       and b.attachment_source == 'co-moving-load']
        for load in attachments:
            parent = bodies.get(load.parent_id)
            if parent is None or parent.role != 'pulley':
                continue
            for connector in objects:
                identifier = connector['target_id']
                if identifier in protected or identifier in (load.object_id, parent.object_id):
                    continue
                valid = np.flatnonzero(load.observed & parent.observed)
                if len(valid) < 5:
                    continue
                extents = [np.asarray(b.half_extent) for b in (parent, load)]
                thickness_limit = min(float(e.min()) for e in extents)
                margins = [max(6.0 / scene.image_height, .25 * float(e.min())) for e in extents]
                matched = 0
                for f in valid:
                    mask = tracks.get(connector['track_id'], {}).get(scene.frames[f])
                    if mask is None:
                        continue
                    y, x = np.nonzero(mask)
                    if len(x) < 4:
                        continue
                    points = np.column_stack((x, y)) / scene.image_height
                    center = points.mean(axis=0)
                    _, axes = np.linalg.eigh((points-center).T @ (points-center))
                    along = (points-center) @ axes[:, 1]
                    across = (points-center) @ axes[:, 0]
                    if np.ptp(along) < 2 * np.ptp(across) or np.ptp(across) > thickness_limit:
                        continue
                    ends = center + np.array([along.min(), along.max()])[:, None] * axes[:, 1]
                    centers = [b.target_centers[f] for b in (parent, load)]
                    near = [np.all(np.abs(ends-c) <= e+m, axis=1)
                            for c, e, m in zip(centers, extents, margins)]
                    lower = np.minimum(*(c-e-m for c, e, m in zip(centers, extents, margins)))
                    upper = np.maximum(*(c+e+m for c, e, m in zip(centers, extents, margins)))
                    local = np.all((points >= lower) & (points <= upper))
                    matched += bool(local and ((near[0][0] and near[1][1]) or (near[0][1] and near[1][0])))
                fraction = matched / len(valid)
                if fraction >= .95:
                    omitted[identifier] = dict(object_id=identifier,
                        track_id=connector['track_id'], represented_by_rope_id=scene.rope_id,
                        parent_object_id=parent.object_id, attachment_object_id=load.object_id,
                        endpoint_and_local_fraction=fraction,
                        reason='Thin local connector already represented by the co-moving load; no separate mass.')
    # A retained load must continue to represent every omitted connector.
    loads = {r['attachment_object_id'] for r in omitted.values()}
    omitted = {key: value for key, value in omitted.items() if key not in loads}
    return list(omitted.values())


def without_attachment_connectors(source, records):
    """Use a solver-only object view; retain excluded observation identities."""
    if not records:
        return source
    omitted = {r['object_id'] for r in records}
    roles = source['rope_roles']
    return dict(source,
        objects=[o for o in source['objects'] if o['target_id'] not in omitted],
        rope_roles=dict(roles,
            objects=[o for o in roles['objects'] if o['object_id'] not in omitted],
            pulley_ids=[i for i in roles['pulley_ids'] if i not in omitted]),
        ignored_local_connectors=[dict(r, observed_object=next(o for o in source['objects']
            if o['target_id'] == r['object_id'])) for r in records])
