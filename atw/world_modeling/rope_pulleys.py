"""Constrain static circular proxies using disjoint observed pulley masks."""
import numpy as np
from scipy.optimize import minimize


def constrain_static_pulleys(source, tracks, roles):
    from atw.polyworld.rope_swr import _interpolate_centers, _track_motion
    ids = {item['object_id'] for item in roles['objects'] if item['role'] == 'pulley'}
    entries = []
    for item in source['objects']:
        if item['target_id'] not in ids:
            continue
        masks = tracks.get(item['track_id'], {})
        centers, observed, radius = _interpolate_centers(
            masks, source['video']['frame_count'], circular=True, reject_truncated=True)
        if observed.sum() < 5 or _track_motion(centers, observed) > 8:
            continue
        entries.append((item['target_id'], masks, np.median(centers[observed], axis=0), radius))
    pairs = []
    for i, (_, a, ca, ra) in enumerate(entries):
        for j in range(i):
            _, b, cb, rb = entries[j]
            distance = float(np.linalg.norm(ca-cb))
            if distance >= ra+rb or distance <= max(ra, rb):
                continue
            frames = [f for f in a.keys() & b.keys() if np.any(a[f]) and np.any(b[f])]
            if len(frames) >= 5 and all(not np.any(a[f] & b[f]) for f in frames):
                pairs.append((i, j, distance))
    if not pairs:
        return {}, []
    radii = np.array([entry[3] for entry in entries])
    # Minimize relative radius changes while preserving measured centers. A tiny
    # numerical clearance keeps the tangent solver away from exact tangency.
    fit = minimize(lambda r: float(np.sum(((r-radii)/radii)**2)), radii,
        method='SLSQP', bounds=[(r*0.5, r) for r in radii],
        constraints=[{'type':'ineq', 'fun':lambda r,i=i,j=j,d=d: d-1e-3-r[i]-r[j]}
                     for i,j,d in pairs], options={'ftol':1e-12, 'maxiter':200})
    if not fit.success:
        raise ValueError('Static pulley proxy constraints could not be satisfied: '+fit.message)
    overrides = {entry[0]:float(r) for entry,r in zip(entries,fit.x) if abs(r-entry[3])>1e-5}
    records = [dict(object_ids=[entries[i][0],entries[j][0]], center_distance_px=d,
                    original_radii_px=[float(radii[i]),float(radii[j])],
                    adjusted_radii_px=[float(fit.x[i]),float(fit.x[j])],
                    evidence='Disjoint observed masks; static circular approximation, not recovered physical radii.')
               for i,j,d in pairs]
    return overrides, records
