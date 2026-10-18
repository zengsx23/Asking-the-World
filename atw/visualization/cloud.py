"""Subpixel dense point splats and persistent colored surface trajectories."""

import cv2
import numpy as np


def project(points, K, camera):
    inverse = np.linalg.inv(camera)
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    pixels = local @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return pixels[:, :2] / pixels[:, 2, None], local[:, 2]


def render_cloud(info, assets, frame, scale=2, support_tolerance=None):
    points = assets["background_points"]
    colors = assets["background_colors"][:, :3]
    if support_tolerance is not None:
        plane = assets["support_plane"]
        keep = points @ plane[:3] - plane[3] <= support_tolerance
        points, colors = points[keep], colors[keep]
    if not len(points):
        raise ValueError("Empty background after filtering")
    bg = np.median(colors, axis=0)
    ps = [points]
    cs = [colors]
    for obj in info["objects"]:
        key = obj["key"]
        T = assets[key + "_poses"][frame]
        if np.isfinite(T).all():
            ps.append(assets[key + "_points"] @ T[:3, :3].T + T[:3, 3])
            cs.append(assets[key + "_colors"])
    points = np.concatenate(ps)
    colors = np.concatenate(cs)
    uv, depth = project(points, assets["K"][frame], assets["camera_to_world"][frame])
    uv -= 0.5
    good = np.isfinite(uv).all(1) & np.isfinite(depth) & (depth > 0)
    uv, depth, colors = uv[good], depth[good], colors[good]
    ij = np.floor(uv).astype(np.int64)
    frac = uv - ij
    w, h = info["width"], info["height"]
    indices = []
    depths = []
    rgb = []
    weights = []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x = ij[:, 0] + dx
        y = ij[:, 1] + dy
        a = (frac[:, 0] if dx else 1 - frac[:, 0]) * (
            frac[:, 1] if dy else 1 - frac[:, 1]
        )
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (a > 1e-5)
        indices.append(y[valid] * w + x[valid])
        depths.append(depth[valid])
        rgb.append(colors[valid])
        weights.append(a[valid])
    ids = np.concatenate(indices)
    ds = np.concatenate(depths)
    cs = np.concatenate(rgb)
    ws = np.concatenate(weights)
    z = np.full(w * h, np.inf)
    np.minimum.at(z, ids, ds)
    visible = ds <= z[ids] + 0.001
    ids, cs, ws = ids[visible], cs[visible], ws[visible]
    den = np.bincount(ids, ws, minlength=w * h)
    image = np.tile(bg, (w * h, 1))
    valid = den > 0
    for c in range(3):
        image[valid, c] = (
            np.bincount(ids, ws * cs[:, c], minlength=w * h)[valid] / den[valid]
        )
    image = np.clip(image.reshape(h, w, 3) * 255, 0, 255).astype(np.uint8)[..., ::-1]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def overlay_trails(
    image, info, assets, frame, history=24, samples=48, width=2, opacity=0.5
):
    base = image.copy()
    scale = image.shape[1] / info["width"]
    K = assets["K"][frame].copy()
    K[:2] *= scale
    palette = cv2.applyColorMap(
        np.linspace(8, 247, samples, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO
    )[:, 0]
    for obj in info["objects"]:
        key = obj["key"]
        if obj["kind"] not in ("mesh", "sphere"):
            continue
        poses = assets[key + "_poses"]
        if not np.isfinite(poses[frame]).all():
            continue
        if key + "_track_points" in assets:
            points = assets[key + "_track_points"]
        elif obj["kind"] == "sphere":
            i = np.arange(samples)
            y = 1 - 2 * (i + 0.5) / samples
            angle = i * np.pi * (3 - np.sqrt(5))
            r = np.sqrt(1 - y * y)
            points = np.column_stack([r * np.cos(angle), y, r * np.sin(angle)])
        else:
            vertices = assets[key + "_vertices"]
            points = vertices[
                np.linspace(
                    0, len(vertices) - 1, min(samples, len(vertices)), dtype=int
                )
            ]
        for point, color in zip(points, palette):
            current = []
            # Preserve gaps instead of drawing across missing saved states.
            for t in range(max(info["start"], frame - history + 1), frame + 1):
                T = poses[t]
                if not np.isfinite(T).all():
                    current = []
                    continue
                uv, depth = project(
                    (T[:3, :3] @ point + T[:3, 3])[None],
                    K,
                    assets["camera_to_world"][frame],
                )
                if not np.isfinite(uv).all() or depth[0] <= 0:
                    current = []
                    continue
                current.append(np.rint(uv[0]).astype(np.int32))
            if not current:
                continue
            col = tuple(map(int, color))
            if len(current) > 1:
                cv2.polylines(
                    image, [np.asarray(current)], False, col, width, cv2.LINE_AA
                )
            cv2.circle(image, tuple(current[-1]), width, col, -1, cv2.LINE_AA)
    return cv2.addWeighted(image, opacity, base, 1 - opacity, 0)
