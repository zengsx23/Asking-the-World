"""Visible depth charts: mask-aware sampling and local curvature regularization."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def membership(uv, mask):
    finite = np.isfinite(uv).all(-1)
    p = np.rint(np.where(np.isfinite(uv), uv, -1)).astype(int)
    h, w = mask.shape
    inside = finite & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    out = np.zeros(len(p), bool)
    out[inside] = mask[p[inside, 1], p[inside, 0]]
    return out


def masked_sample(field, uv, mask):
    """Bilinear, renormalized over this surface only; never mix background depth."""
    value = np.asarray(field)
    scalar = value.ndim == 2
    if scalar:
        value = value[..., None]
    h, w = mask.shape
    safe = np.where(np.isfinite(uv), uv, -10)
    base = np.floor(safe).astype(int)
    fraction = safe-base
    result = np.zeros((len(uv), value.shape[-1]), float)
    total = np.zeros(len(uv), float)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = base[:, 0]+dx, base[:, 1]+dy
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        ix = np.flatnonzero(inside)
        ix = ix[mask[y[ix], x[ix]] & np.isfinite(value[y[ix], x[ix]]).all(1)]
        weight = (fraction[ix, 0] if dx else 1-fraction[ix, 0]) * (fraction[ix, 1] if dy else 1-fraction[ix, 1])
        result[ix] += value[y[ix], x[ix]]*weight[:, None]
        total[ix] += weight
    good = (total > 1e-9) & membership(uv, mask)
    result[good] /= total[good, None]
    result[~good] = np.nan
    return result[:, 0] if scalar else result


def chart_geometry(mask, observed):
    """Second-difference stencils inside connected observed regions only."""
    y, x = np.nonzero(mask)
    uv = np.column_stack((x, y)).astype(float)
    index = np.full(mask.shape, -1, int)
    index[y, x] = np.arange(len(x))
    triples = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ay, ax, by, bx = y-dy, x-dx, y+dy, x+dx
        ok = (ay >= 0) & (ay < mask.shape[0]) & (ax >= 0) & (ax < mask.shape[1])
        ok &= (by >= 0) & (by < mask.shape[0]) & (bx >= 0) & (bx < mask.shape[1])
        ids = np.flatnonzero(ok)
        ids = ids[mask[ay[ids], ax[ids]] & mask[by[ids], bx[ids]]]
        tri = np.column_stack((index[ay[ids], ax[ids]], ids, index[by[ids], bx[ids]]))
        triples.append((tri, float(dx*dx+dy*dy)))
    z = observed[mask].astype(float)
    differences = [np.abs(np.diff(z[t], axis=1)).ravel() for t, _ in triples if len(t)]
    scale = 3*float(np.median(np.concatenate(differences))) if differences else 0.
    scale = max(scale, np.finfo(float).eps*max(1., float(np.median(np.abs(z)))))
    rows, cols, values = [], [], []
    offset = 0
    for t, distance in triples:
        jump = np.abs(np.diff(z[t], axis=1)).max(1)
        conductance = 1/(1+(jump/scale)**2)
        coeff = np.sqrt(conductance)/distance
        rows.extend(np.repeat(np.arange(offset, offset+len(t)), 3))
        cols.extend(t.ravel())
        values.extend((coeff[:, None]*[1., -2., 1.]).ravel())
        offset += len(t)
    curvature = sparse.csr_matrix((values, (rows, cols)), shape=(offset, len(uv)))
    return uv, index, curvature


def fit_surface(observed, mask):
    """Fit current visible depth, never smoothing across an unobserved mask gap."""
    if not mask.any():
        return np.full(mask.shape, np.nan, np.float32)
    _, _, curvature = chart_geometry(mask, observed)
    matrix = sparse.eye(int(mask.sum())) + curvature.T @ curvature
    z = spsolve(matrix.tocsc(), observed[mask].astype(float))
    depth = np.full(mask.shape, np.nan, np.float32)
    depth[mask] = z
    return depth


def open_surface_topology(mask):
    """Triangulate adjacent observed pixel centers, preserving holes and boundaries."""
    y, x = np.nonzero(mask)
    uv = np.column_stack((x, y)).astype(np.float32)
    index = np.full(mask.shape, -1, int)
    index[mask] = np.arange(len(uv))
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    candidates = [np.stack((a, b, d), -1), np.stack((a, d, c), -1)]
    tri = np.stack((a, b, c), -1)
    tri[d >= 0] = -1
    candidates.append(tri)
    tri = np.stack((b, d, c), -1)
    tri[a >= 0] = -1
    candidates.append(tri)
    faces = np.concatenate([v.reshape(-1, 3) for v in candidates])
    faces = faces[(faces >= 0).all(1)].astype(np.int32)
    covered = np.zeros_like(mask)
    used = np.unique(faces)
    covered[y[used], x[used]] = True
    return uv, faces, covered


def depth_statistics(predicted, observed, extent):
    """Same-ray residuals, not ground-truth shape or material-identity accuracy."""
    valid = np.isfinite(predicted) & np.isfinite(observed) & (observed > 0)
    error = np.abs(predicted[valid] - observed[valid])
    return {
        "depth_samples": int(valid.sum()),
        "depth_absolute_p50": float(np.median(error)) if len(error) else None,
        "depth_absolute_p90": float(np.quantile(error, .9)) if len(error) else None,
        "depth_extent_p50_percent": float(100*np.median(error)/extent) if len(error) else None,
        "depth_extent_p90_percent": float(100*np.quantile(error, .9)/extent) if len(error) else None,
    }
