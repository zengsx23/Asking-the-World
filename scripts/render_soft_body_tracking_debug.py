#!/usr/bin/env python3
"""Global full-frame identity and visible-depth review, without object recentering."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atw.perception.pointcloud_reconstruction_assets import load_reconstruction_assets

from atw.perception.soft_body_tracking import unproject_world


def identity_colors(count, object_index):
    rng = np.random.default_rng(8103+object_index)
    # Stable ID colors across the entire video, unrelated to depth error colors.
    return rng.integers(65, 256, size=(count, 3), dtype=np.uint8)


def depth_colors(relative_error):
    t = np.clip(np.nan_to_num(relative_error, nan=.2)/.2, 0, 1)
    return np.column_stack((255*np.minimum(2*t, 1), 230*np.minimum(2-2*t, 1),
                            np.full(len(t), 40))).astype(np.uint8)


def text(canvas, label, xy, scale=.43):
    cv2.putText(canvas, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (245, 245, 245), 1, cv2.LINE_AA)


def draw_points(canvas, xy, colors, radius=1):
    h, w = canvas.shape[:2]
    finite = np.isfinite(xy).all(1)
    p = np.rint(xy[finite]).astype(int)
    colors = colors[finite]
    inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    for q, color in zip(p[inside], colors[inside]):
        cv2.circle(canvas, tuple(q), radius, tuple(map(int, color)), -1, cv2.LINE_AA)


def global_view_bounds(points, axes, width, height):
    projected = points[:, axes]
    lo, hi = np.quantile(projected, [.005, .995], axis=0)
    center = (lo+hi)/2
    span = np.maximum(hi-lo, 1e-6)
    scale = min((width-30)/span[0], (height-30)/span[1])
    return center, scale


def render_soft_body_review(run_dir: Path, *, output: Path | None = None) -> Path:
    directory = run_dir/"soft_body_surfaces"
    names = sorted(path.stem for path in directory.glob("*.npz"))
    custom_output = output is not None
    output = output if custom_output else run_dir/"debug/soft_body_global_review.mp4"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite retained review: {output}")
    if not names:
        raise ValueError("no soft-body tracking results")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_dir/"run_metadata.json").read_text())
    sources = json.loads((run_dir / 'input_sources.json').read_text())
    assets = load_reconstruction_assets(Path(sources['reconstruction_manifest']).parent)
    geometry = assets.geometry_dir
    rgb = np.load(geometry/"rgb.npy", mmap_mode="r")
    depth = np.load(geometry/"depth.npy", mmap_mode="r")
    camera = np.load(geometry/"camera_poses.npy")
    intrinsic = np.load(geometry/"intrinsics.npy")
    valid = np.load(geometry/"geometry_valid_threshold_0.npy", mmap_mode="r")
    count, h, w = depth.shape
    objects, summary = {}, {}
    for index, name in enumerate(names):
        # NpzFile key access decompresses the entire array, even before slicing.
        # Retain only the arrays used below and close the archives before rendering.
        with np.load(directory/f"{name}.npz") as archive:
            surface = {key: archive[key] for key in ("surface_depth", "reference_extent")}
        with np.load(run_dir/f"trajectories/{name}.npz") as archive:
            tracks = {"uv": archive["uv"]}
        with np.load(run_dir/f"associations/{name}.npz") as archive:
            associations = {key: archive[key] for key in ("frame_offsets", "current_uv", "canonical_id")}
        rows = json.loads((directory/f"{name}_metrics.json").read_text())
        colors = identity_colors(tracks["uv"].shape[1], index)
        objects[name] = (surface, tracks, associations, rows, colors)
        scored = [r for r in rows if r["mask_pixels"] > 0]
        summary[name] = {
            "reference_points":len(colors),
            "mean_point_iou":float(np.mean([r["corrected_visible_iou"] for r in scored])),
            "mean_surface_iou":float(np.mean([r["surface_iou"] for r in scored])),
            "mean_depth_extent_p90_percent":float(np.mean([r["depth_extent_p90_percent"]
                for r in scored if r["depth_extent_p90_percent"] is not None])),
            "last_eligible_frame_tracked_points":scored[-1]["tracked_points"],
            "identity_colors_are_not_confidence":True,
        }
    # One fixed scene basis and framing for all objects and times, never a ball crop.
    origin = camera[0, :3, 3]
    rotation = camera[0, :3, :3]
    background = []
    for t in np.unique(np.linspace(0, count-1, min(12, count)).astype(int)):
        y, x = np.nonzero(valid[t] & np.isfinite(depth[t]) & (depth[t] > 0))
        y, x = y[::32], x[::32]
        world = unproject_world(np.column_stack((x, y)), depth[t, y, x], camera[t], intrinsic[t])
        background.append((world-origin)@rotation)
    scene = np.concatenate(background)
    panel_h = h+80+22*len(names)
    bounds = [global_view_bounds(scene, axes, w, h) for axes in ((0, 2), (1, 2))]
    samples = set(np.unique(np.minimum([0, 35, 55, 82, count-1], count-1)))
    temporary = output.with_name(".soft_body_global_review.partial.mp4")
    try:
        with imageio.get_writer(temporary, fps=metadata["fps"], codec="libx264", quality=8,
                               pixelformat="yuv420p", macro_block_size=2,
                               ffmpeg_params=["-threads", "4"]) as writer:
            for t in range(count):
                canvas = np.zeros((2*panel_h, 2*w, 3), np.uint8)
                panels = [canvas[:panel_h, :w], canvas[:panel_h, w:],
                          canvas[panel_h:, :w], canvas[panel_h:, w:]]
                titles = ("GLOBAL RGB / identities", "GLOBAL RGB / surface depth residual",
                          "GLOBAL X-Z / identities", "GLOBAL Y-Z / depth residual")
                for panel, title in zip(panels, titles):
                    text(panel, f"{title} | frame {t:03d}", (6, 24), .48)
                panels[0][35:35+h] = rgb[t]
                panels[1][35:35+h] = rgb[t]
                views = [p[35:35+h] for p in panels]
                y, x = np.nonzero(valid[t] & np.isfinite(depth[t]) & (depth[t] > 0))
                y, x = y[::12], x[::12]
                obs_world = unproject_world(np.column_stack((x, y)), depth[t, y, x], camera[t], intrinsic[t])
                obs_local = (obs_world-origin)@rotation
                for side, axes in enumerate(((0, 2), (1, 2))):
                    center, scale = bounds[side]
                    xy = (obs_local[:, axes]-center)*scale+[w/2, h/2]
                    draw_points(views[side+2], xy, np.full((len(xy), 3), 75, np.uint8), 1)
                for object_index, name in enumerate(names):
                    surface, tracks, associations, rows, colors = objects[name]
                    smooth = surface["surface_depth"][t]
                    domain = np.isfinite(smooth)
                    yy, xx = np.nonzero(domain)
                    uv = np.column_stack((xx, yy)).astype(float)
                    world = unproject_world(uv, smooth[domain], camera[t], intrinsic[t])
                    local = (world-origin)@rotation
                    errors = np.abs(smooth[domain]-depth[t][domain])/float(surface["reference_extent"])
                    heat = depth_colors(errors)
                    # Orange means observed surface with no assigned ID. It is not depth error.
                    id_colors = np.tile(np.array([255, 145, 35], np.uint8), (len(uv), 1))
                    index_map = np.full((h, w), -1, int)
                    index_map[yy, xx] = np.arange(len(uv))
                    start, end = associations["frame_offsets"][t:t+2]
                    observed_uv = associations["current_uv"][start:end]
                    ids = associations["canonical_id"][start:end]
                    p = np.rint(observed_uv).astype(int)
                    inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h) & (ids >= 0)
                    chosen = np.flatnonzero(inside)
                    mapped = index_map[p[chosen, 1], p[chosen, 0]]
                    good = mapped >= 0
                    id_colors[mapped[good]] = colors[ids[chosen[good]]]
                    views[0][yy, xx] = (.4*views[0][yy, xx]+.6*id_colors).astype(np.uint8)
                    views[1][yy, xx] = (.25*views[1][yy, xx]+.75*heat).astype(np.uint8)
                    for identity in np.linspace(0, len(colors)-1, min(40, len(colors))).astype(int):
                        trail = tracks["uv"][max(0, t-7):t+1, identity]
                        for a, b in zip(trail[:-1], trail[1:]):
                            if np.isfinite([a, b]).all():
                                cv2.line(views[0], tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                                         tuple(map(int, colors[identity])), 1, cv2.LINE_AA)
                    for side, axes in enumerate(((0, 2), (1, 2))):
                        center, scale = bounds[side]
                        xy = (local[:, axes]-center)*scale+[w/2, h/2]
                        draw_points(views[side+2], xy, id_colors if side == 0 else heat, 1)
                    row = rows[t]
                    point_iou = row["corrected_visible_iou"]
                    p90 = row["depth_extent_p90_percent"]
                    identity_count = int(good.sum())
                    label = (f"{name}: IDs {row['tracked_points']}/{len(colors)}; assigned pixels {identity_count}/{len(uv)}")
                    text(panels[0], label, (6, 35+h+18+22*object_index), .39)
                    text(panels[2], label, (6, 35+h+18+22*object_index), .39)
                    stats = f"{name}: point IoU {point_iou:.3f}; track depth P90 {p90:.2f}% D" if p90 is not None and point_iou is not None else f"{name}: no visible tracked depth"
                    text(panels[1], stats, (6, 35+h+18+22*object_index), .39)
                    text(panels[3], stats, (6, 35+h+18+22*object_index), .39)
                for i, panel in enumerate(panels):
                    legend = "ID colors = labels, not proof; orange = no assigned ID" if i in (0, 2) else "Surface dz / D: green 0%, yellow 10%, red >=20%; not GT"
                    text(panel, legend, (6, panel_h-18), .39)
                writer.append_data(canvas)
                if t in samples:
                    prefix = output.stem if custom_output else "soft_body_global"
                    cv2.imwrite(str(output.parent/f"{prefix}_frame_{t:03d}.jpg"),
                                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    summary["display"] = {"global_frame":True, "object_recentered":False,
                          "depth_and_identity_separate":True, "frames":count,
                          "side_axes":"fixed first-camera axes in the common world; not gravity axes"}
    summary_path = output.with_suffix(".json") if custom_output else run_dir/"soft_body_tracking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new output path for a retained review version")
    args = parser.parse_args()
    print(render_soft_body_review(args.run_dir, output=args.output))
