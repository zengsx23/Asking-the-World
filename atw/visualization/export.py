"""Export retained SWR fits into a renderer-only scene bundle."""

import json
from pathlib import Path
import cv2
import numpy as np
import trimesh


def read(path):
    return json.loads(Path(path).read_text())


def export_run(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    first = read(run / "stage_one/result.json")
    artifacts = {a["artifact_id"]: run / a["path"] for a in first["artifacts"]}
    scene = first["data"]["scene"]
    swr = read(artifacts[scene["sources"]["swr_asset_id"]])
    if swr["availability"] != "fitted":
        raise ValueError("A fitted SWR is required")
    fit_path = artifacts[swr["result_asset_id"]]
    fit = read(fit_path)
    tracking_path = artifacts[scene["sources"]["tracking_asset_id"]]
    tracking = read(tracking_path)
    output.mkdir(parents=True, exist_ok=False)
    if scene["engine"] in {"rigid", "cloth", "softbody"}:
        return export_3d(
            scene["engine"], fit_path, fit, tracking_path, tracking, output
        )
    if scene["engine"] == "fluid":
        return export_fluid(fit_path, fit, output)
    if scene["engine"] == "rope":
        segpath = artifacts[scene["sources"]["segmentation_asset_id"]]
        return export_rope(fit, segpath, output)
    raise ValueError("Unsupported SWR engine")


def finish(output, info, arrays):
    info["schema_version"] = 1
    np.savez_compressed(output / "assets.npz", **arrays)
    (output / "scene.json").write_text(json.dumps(info, indent=2) + "\n")
    return output


def export_3d(kind, fit_path, fit, track_path, track, output):
    manifest_path = track_path.parent / track["reconstruction_manifest"]
    manifest = read(manifest_path)
    base = manifest_path.parent
    tracking = track_path.parent / track["tracking_directory"]
    geom = base / manifest["paths"]["geometry"]
    rgb = np.load(geom / "rgb.npy", mmap_mode="r")
    depth = np.load(geom / "depth.npy", mmap_mode="r")
    K = np.load(geom / "intrinsics.npy")
    camera = np.load(geom / "camera_poses.npy")
    N, h, w = rgb.shape[:3]
    if K.ndim == 2:
        K = np.repeat(K[None], N, 0)
    if camera.ndim == 2:
        camera = np.repeat(camera[None], N, 0)
    if not np.allclose(camera, camera[0], atol=1e-4) or not np.allclose(
        K, K[0], atol=1e-4
    ):
        raise ValueError("Temporal background export requires a fixed estimated camera")
    masks = {
        p.stem: np.load(p)["masks"].astype(bool)
        for p in (base / manifest["paths"]["tracking_masks"]).glob("*.npz")
    }
    union = np.logical_or.reduce(list(masks.values()))
    valid = np.stack(
        [
            ~cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
            for m in union
        ]
    )
    valid &= np.isfinite(depth) & (depth > 0)
    count = valid.sum(0)
    color = (rgb.astype(np.float32) * valid[..., None]).sum(0) / np.maximum(
        count[..., None], 1
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.nanmedian(np.where(valid, depth, np.nan), axis=0)
    holes = ~np.isfinite(d)
    if holes.all():
        raise ValueError("No observed static background")
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
    d[holes] = d[tuple(idx[:, holes])]
    color[holes] = color[tuple(idx[:, holes])]
    y, x = np.mgrid[:h, :w]
    cam = np.stack(
        [
            (x + 0.5 - K[0, 0, 2]) * d / K[0, 0, 0],
            (y + 0.5 - K[0, 1, 2]) * d / K[0, 1, 1],
            d,
        ],
        -1,
    ).reshape(-1, 3)
    arrays = dict(
        K=K,
        camera_to_world=camera,
        background_points=cam @ camera[0, :3, :3].T + camera[0, :3, 3],
        background_colors=color.reshape(-1, 3) / 255,
    )
    cv2.imwrite(str(output / "background.png"), color.astype(np.uint8)[..., ::-1])
    objects = []

    def sample_color(name):
        pixels = np.asarray(rgb)[masks[name]]
        if not len(pixels):
            raise ValueError(f"Empty object mask: {name}")
        return (pixels.mean(0) / 255).tolist()

    def mesh(name, poses):
        directory = base / manifest["paths"]["rigid_models"] / name
        model = trimesh.load(
            directory / "canonical_metric.glb", force="scene"
        ).to_geometry()
        key = f"object_{len(objects)}"
        arrays.update(
            {
                key + "_vertices": np.asarray(model.vertices),
                key + "_faces": np.asarray(model.faces),
                key + "_poses": poses,
            }
        )
        objects.append(dict(name=name, key=key, kind="mesh", color=sample_color(name)))
        if kind == "rigid":
            with np.load(directory / "canonical_rigid_model.npz") as a:
                points = a["canonical_surface_points"].copy()
            with np.load(tracking / "rigid_trajectories" / f"{name}.npz") as a:
                ref = int(a["reference_frame"])
                T = a["poses"][ref]
                points_track = a["canonical_tracking_points"].copy()
            inv = np.linalg.inv(camera[ref])
            world = points @ T[:3, :3].T + T[:3, 3]
            cp = world @ inv[:3, :3].T + inv[:3, 3]
            uv = cp @ K[ref].T
            uv = uv[:, :2] / uv[:, 2, None]
            colors = np.tile(sample_color(name), (len(points), 1))
            mask = masks[name][ref].astype(np.uint8)
            # OpenCV remap has a signed-short dimension limit.
            for start in range(0, len(points), 16000):
                end = min(len(points), start + 16000)
                coords = uv[start:end].astype(np.float32)
                mx, my = coords[:, 0, None], coords[:, 1, None]
                visible = (cv2.remap(mask, mx, my, cv2.INTER_NEAREST).ravel() > 0) & (
                    cp[start:end, 2] > 0
                )
                sampled = (
                    cv2.remap(np.asarray(rgb[ref]), mx, my, cv2.INTER_LINEAR).reshape(
                        -1, 3
                    )
                    / 255
                )
                colors[start:end][visible] = sampled[visible]
            arrays[key + "_points"] = points
            arrays[key + "_colors"] = colors
            arrays[key + "_track_points"] = points_track[
                np.linspace(
                    0, len(points_track) - 1, min(48, len(points_track)), dtype=int
                )
            ]

    start = 0
    if kind == "rigid":
        for name in fit["object_ids"]:
            poses = np.full((N, 4, 4), np.nan)
            for row in fit["predicted_trajectories"][name]:
                poses[row["frame_index"]] = row["pose_4x4_world"]
            mesh(name, poses)
        gravity = read(base / "surface/gravity_estimate.json")
        plane = gravity["selected"]["support_plane"]
        # Stored for explicit opt-in filtering at render time.
        arrays["support_plane"] = np.r_[plane["normal_world"], plane["offset_world"]]
    elif kind == "softbody":
        from atw.polyworld.soft_body_swr import load_softbody_scene, _numpy_deformation

        ss, _ = load_softbody_scene(tracking, reconstruction_dir=base)
        with np.load(fit_path.parent / fit["prediction_npz"]) as a:
            pred = a["predicted"].copy()
        deformation = _numpy_deformation(pred)
        basis = np.stack(
            [ss.wall_lateral_world, ss.gravity_down_world, ss.wall_normal_world], axis=1
        )
        for j, name in enumerate(ss.object_ids):
            key = f"object_{len(objects)}"
            poses = np.full((N, 4, 4), np.nan)
            for i, f in enumerate(ss.frames):
                if i < ss.first_offsets[j]:
                    continue
                A = np.eye(3)
                A[:2, :2] = deformation[i, j]
                poses[f] = np.eye(4)
                poses[f, :3, :3] = basis @ A * ss.radii[j]
                poses[f, :3, 3] = (
                    pred[i, j, 0] * ss.wall_lateral_world
                    + pred[i, j, 1] * ss.gravity_down_world
                    + ss.wall_depths[j] * ss.wall_normal_world
                )
            arrays[key + "_poses"] = poses
            objects.append(
                dict(name=name, key=key, kind="sphere", color=sample_color(name))
            )
        for name in masks:
            if name in ss.object_ids:
                continue
            with np.load(
                base
                / manifest["paths"]["rigid_models"]
                / name
                / "canonical_rigid_model.npz"
            ) as a:
                poses = np.repeat(a["reference_pose"][None], N, 0)
            mesh(name, poses)
    else:
        start = max(obj["start_frame"] for obj in fit["objects"].values())
        for name, obj in fit["objects"].items():
            key = f"object_{len(objects)}"
            vertices = np.full((N, len(obj["vertex_ids"]), 3), np.nan)
            vertices[obj["frames"]] = obj["predicted_positions_world"]
            arrays[key + "_vertices_frames"] = vertices
            arrays[key + "_faces"] = np.asarray(obj["faces"])
            objects.append(
                dict(name=name, key=key, kind="cloth", color=sample_color(name))
            )
        # A display scene needs one explicit trajectory for shared obstacles.
        branches = fit["objects"]
        rigids = next(iter(branches.values()))["predicted_rigid_trajectories"][
            "object_ids"
        ]
        selection = {}
        for name in rigids:
            candidates = []
            for cname, obj in branches.items():
                rr = obj["predicted_rigid_trajectories"]
                j = rr["object_ids"].index(name)
                distance = np.linalg.norm(
                    np.asarray(rr["centers_world"][0][j])
                    - np.mean(obj["predicted_positions_world"][0], axis=0)
                )
                candidates.append((distance, cname, j))
            _, chosen, j = min(candidates)
            obj = branches[chosen]
            rr = obj["predicted_rigid_trajectories"]
            poses = np.full((N, 4, 4), np.nan)
            initial = np.stack(
                [rr[a][0][j] for a in ("axis_x_world", "axis_y_world", "axis_z_world")],
                axis=1,
            )
            with np.load(
                base
                / manifest["paths"]["rigid_models"]
                / name
                / "canonical_rigid_model.npz"
            ) as a:
                align = initial.T @ a["reference_pose"][:3, :3]
            for i, f in enumerate(obj["frames"]):
                poses[f] = np.eye(4)
                poses[f, :3, 3] = rr["centers_world"][i][j]
                poses[f, :3, :3] = (
                    np.stack(
                        [
                            rr[a][i][j]
                            for a in ("axis_x_world", "axis_y_world", "axis_z_world")
                        ],
                        axis=1,
                    )
                    @ align
                )
            mesh(name, poses)
            selection[name] = chosen
        (output / "cloth-assembly.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
    return finish(
        output,
        dict(
            kind=kind,
            width=w,
            height=h,
            frames=N,
            start=start,
            fps=float(fit.get("fps", 30)),
            objects=objects,
            background_mean=(color.mean((0, 1)) / 255).tolist(),
        ),
        arrays,
    )


def export_fluid(path, fit, output):
    obs = read(path.parent / fit["observations"])
    w, h = obs["image_wh"]
    N = obs["frame_count"]
    with np.load(path.parent / fit["prediction_npz"]) as a:
        arrays = {
            k: a[k].copy() for k in ("predicted_xy", "particle_fluid", "particle_birth")
        }
    with np.load(path.parent / "observations.npz") as a:
        masks = a["target_masks"]
        arrays["obstacle_mask"] = a["obstacle_mask"]
    sums = np.zeros((h, w, 3))
    counts = np.zeros((h, w))
    cs = np.zeros((len(obs["fluid_ids"]), 3))
    ct = np.zeros(len(cs))
    cap = cv2.VideoCapture(obs["video_path"])
    try:
        for f in range(N):
            ok, im = cap.read()
            if not ok:
                raise ValueError(f"Video ended at frame {f}")
            im = cv2.cvtColor(cv2.resize(im, (w, h)), cv2.COLOR_BGR2RGB)
            valid = ~masks[f].any(0)
            sums[valid] += im[valid]
            counts[valid] += 1
            for j in range(len(cs)):
                cs[j] += im[masks[f, j]].sum(0)
                ct[j] += masks[f, j].sum()
    finally:
        cap.release()
    avg = sums / np.maximum(counts[..., None], 1)
    avg[counts == 0] = avg[counts > 0].mean(0)
    cv2.imwrite(str(output / "background.png"), avg.astype(np.uint8)[..., ::-1])
    return finish(
        output,
        dict(
            kind="fluid",
            width=w,
            height=h,
            frames=N,
            start=0,
            fps=obs["fps"],
            aspect=w / h,
            objects=[],
            colors=(cs / np.maximum(ct[:, None], 1) / 255).tolist(),
            particle_radius_pixels=fit["particle_radius_pixels"],
            background_mean=(avg.mean((0, 1)) / 255).tolist(),
        ),
        arrays,
    )


def export_rope(fit, segpath, output):
    from atw.perception.object_reconstruction import _unpack_mask

    seg = read(segpath)
    src = fit["source"]
    video = src["video"]
    w, h = video["width"], video["height"]
    N = video["frame_count"]
    bindings = {o["target_id"]: o["track_id"] for o in src["objects"]}
    index = read(segpath.parent / seg["mask_index"])
    archive = np.load(segpath.parent / seg["mask_archive"])
    lookup = {(v["track_id"], v["frame_index"]): (k, v) for k, v in index.items()}
    sums = {n: np.zeros(3) for n in bindings}
    count = {n: 0 for n in bindings}
    bg = np.zeros(3)
    nbg = 0
    cap = cv2.VideoCapture(video["path"])
    try:
        for f in range(N):
            ok, im = cap.read()
            if not ok:
                raise ValueError(f"Video ended at frame {f}")
            if f % 10:
                continue
            im = im[..., ::-1]
            union = np.zeros((h, w), bool)
            for name, track in bindings.items():
                pair = lookup.get((track, f))
                if pair is None:
                    continue
                mask = _unpack_mask(archive[pair[0]], pair[1])
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                union |= mask
                sums[name] += im[mask].sum(0)
                count[name] += mask.sum()
            bg += im[~union].sum(0)
            nbg += (~union).sum()
    finally:
        cap.release()
        archive.close()
    colors = {n: (sums[n] / max(count[n], 1) / 255).tolist() for n in bindings}
    objects = []
    arrays = {}
    seen = set()
    for rope in fit["ropes"]:
        key = f"object_{len(objects)}"
        rows = rope["predicted_centerline"]
        line = np.full((N, rope["node_count"], 2), np.nan)
        for row in rows:
            line[row["frame_index"]] = row["xy_normalized"]
        arrays[key + "_line"] = line
        objects.append(
            dict(
                name=rope["rope_id"],
                kind="rope",
                key=key,
                color=colors[rope["rope_id"]],
                radius=0.004,
            )
        )
        for body in rope["bodies"]:
            name = body["object_id"]
            if name in seen:
                continue
            seen.add(name)
            key = f"object_{len(objects)}"
            centers = np.repeat(
                np.asarray(body["initial_center_normalized"])[None], N, 0
            )
            for row in rope["predicted_body_centers"].get(name, []):
                centers[row["frame_index"]] = row["xy_normalized"]
            arrays[key + "_centers"] = centers
            objects.append(
                dict(
                    name=name,
                    key=key,
                    kind=(
                        "pulley"
                        if body["role"] == "pulley"
                        else (
                            "box"
                            if body["shape"] in ("box", "irregular")
                            else "sphere2d"
                        )
                    ),
                    radius=body["radius_normalized"],
                    half_extent=body["half_extent_normalized"],
                    color=colors[name],
                )
            )
    return finish(
        output,
        dict(
            kind="rope",
            width=640,
            height=360,
            frames=N,
            start=0,
            fps=video["fps"],
            aspect=w / h,
            background_mean=(bg / max(nbg, 1) / 255).tolist(),
            objects=objects,
        ),
        arrays,
    )
