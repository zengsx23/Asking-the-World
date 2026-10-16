"""Small original-camera diagrams of new simulation results."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np


IDENTITY_PANEL_MAX_WIDTH = 960


SEMANTIC_BGR = {
    "orange": (0, 130, 255), "yellow": (0, 220, 255), "blue": (255, 80, 20),
    "cyan": (255, 220, 0), "pink": (190, 90, 255), "light blue": (255, 200, 80),
    "green": (50, 210, 50), "white": (245, 245, 245), "teal": (180, 170, 20),
    "red": (30, 30, 230), "purple": (180, 50, 170), "brown": (45, 80, 130),
}


def _mask_color(backend, object_id):
    """Estimate an object's displayed BGR color from its clearest original masked pixels."""
    best = None
    for frame in range(int(backend.frame_count)):
        try:
            mask = np.asarray(backend._mask(object_id, frame), dtype=bool)
        except (KeyError, ValueError, OSError, IndexError):
            continue
        area = int(mask.sum())
        if area and (best is None or area > best[0]):
            best = area, frame, mask
    if best is not None:
        _, frame, mask = best
        capture = cv2.VideoCapture(str(backend.video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
        finally:
            capture.release()
        if ok:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            interior = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
            if not interior.any():
                interior = mask.astype(bool)
            pixels = image[interior]
            if len(pixels):
                return tuple(int(v) for v in np.median(pixels, axis=0).round())
    metadata = backend.objects.get(object_id, {})
    color_name = str(metadata.get("color", metadata.get("appearance", {}).get("base_color", ""))).strip().lower()
    return SEMANTIC_BGR.get(color_name, (120, 120, 120))


def _focused_identity(image, mask, background_scale, outline_width):
    focused = np.where(
        mask[..., None], image,
        np.rint(image.astype(np.float32) * background_scale).astype(np.uint8))
    # Draw just outside the mask so every foreground pixel keeps its color.
    expanded = cv2.dilate(mask.astype(np.uint8), np.ones((2*outline_width+1, 2*outline_width+1), np.uint8))
    focused[expanded.astype(bool) & ~mask] = (0, 0, 255)
    return focused


def _local_crop_panel(image, mask, background_scale, outline_width):
    """Fit a padded target crop into one frame-sized panel without changing aspect ratio."""
    y, x = np.where(mask)
    height, width = mask.shape
    span = max(int(x.max() - x.min() + 1), int(y.max() - y.min() + 1))
    padding = max(8, round(span * .35))
    left, right = max(0, int(x.min()) - padding), min(width, int(x.max()) + padding + 1)
    top, bottom = max(0, int(y.min()) - padding), min(height, int(y.max()) + padding + 1)
    crop = image[top:bottom, left:right]
    crop_mask = mask[top:bottom, left:right]
    scale = min(width / crop.shape[1], height / crop.shape[0])
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(crop, size, interpolation=interpolation)
    resized_mask = cv2.resize(crop_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    resized = _focused_identity(resized, resized_mask, background_scale, outline_width)
    panel = np.zeros_like(image)
    x0, y0 = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    panel[y0:y0+resized.shape[0], x0:x0+resized.shape[1]] = resized
    return panel


def labeled_observation(image, label, mask=None, *, identity_view=False):
    """Render a full frame; identity views also carry an aspect-preserving local crop."""
    panel_boundary = None
    if identity_view and image.shape[1] > IDENTITY_PANEL_MAX_WIDTH:
        scale = IDENTITY_PANEL_MAX_WIDTH / image.shape[1]
        image = cv2.resize(image, (IDENTITY_PANEL_MAX_WIDTH,
            max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        if mask.any():
            original_width = image.shape[1]
            background_scale = 1.0 if identity_view else .25
            outline_width = max(1, round(max(image.shape[:2]) / 960)) + int(identity_view)
            focused = _focused_identity(image, mask, background_scale, outline_width)
            if identity_view:
                image = np.hstack((focused, _local_crop_panel(image, mask, background_scale, outline_width)))
                panel_boundary = original_width
            else:
                image = focused
    left_label = f"FULL FRAME | {label}" if identity_view and panel_boundary is not None else label
    label_width = cv2.getTextSize(left_label, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)[0][0]
    image = cv2.copyMakeBorder(image, 30, 0, 0, max(0, label_width+16-image.shape[1]), cv2.BORDER_CONSTANT)
    cv2.putText(image, left_label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    if panel_boundary is not None:
        cv2.putText(image, "LOCAL CROP | SAME FRAME", (panel_boundary + 8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    return image


def identity_mask_observation(mask, *, touches_frame_edge=None, touches_other_mask=None):
    """Return a short visibility note, not reconstruction provenance or selection rationale."""
    mask = np.asarray(mask, dtype=bool)
    if touches_frame_edge is None:
        touches_frame_edge = bool(
            mask.any() and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
    touches_frame_edge = bool(touches_frame_edge)
    touches_other_mask = bool(touches_other_mask) if touches_other_mask is not None else False
    if touches_frame_edge and touches_other_mask:
        status = "edge_and_neighbor_contact"
        note = "Mask touches the frame edge and another object; the visible shape may be partial or occluded."
    elif touches_frame_edge:
        status = "frame_edge"
        note = "Mask touches the frame edge; the object may be only partly visible."
    elif touches_other_mask:
        status = "neighbor_contact"
        note = "Mask touches another object; part of the outline may be occluded."
    else:
        status = "complete_clear"
        note = "Mask is fully inside the frame and separated from other object masks."
    return dict(status=status, touches_frame_edge=touches_frame_edge,
                touches_other_mask=touches_other_mask, note=note)


def reconstruction_identity_images(video, reconstruction, object_ids, output, fps):
    """Use the retained reconstruction mask, never reselect a larger tracking mask."""
    reconstruction, output = Path(reconstruction), Path(output)
    references = json.loads((reconstruction / "reference_frame_selections.json").read_text())
    inputs = {}
    for manifest in sorted((reconstruction / "work").glob("*_reconstruction_inputs/manifest.json")):
        for row in json.loads(manifest.read_text())["objects"]:
            inputs[row["track_id"]] = row
    images, sources = [], {}
    capture = cv2.VideoCapture(str(video))
    try:
        for name in object_ids:
            if name in inputs:
                row = inputs[name]
                frame = int(row["frame_index"])
                path = Path(row["input_npz"])
                with np.load(path) as data:
                    mask = data["mask"].astype(bool)
                source = "sam3d_input_mask"
            else:
                # Cloth and other non-SAM3D assets still use their reconstruction reference.
                frame = int(references[name]["reference_frame"])
                path = reconstruction / "masks" / f"{name}.npz"
                with np.load(path) as data:
                    mask = data["masks"][frame].astype(bool)
                source = "reconstruction_reference_mask"
            if not mask.any():
                raise ValueError(f"Empty reconstruction identity mask: {name}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot read reconstruction reference frame {frame}")
            image = labeled_observation(
                image, f"{name} | frame {frame} | {frame / fps:.2f}s", mask, identity_view=True)
            target = output / "identity_images" / f"{name}_{frame:05d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), image):
                raise OSError(f"Cannot write identity image {target}")
            images.append(str(target))
            selection = references.get(name, {}).get("agent_response", {})
            observation = identity_mask_observation(
                mask, touches_frame_edge=selection.get("touches_frame_edge"),
                touches_other_mask=selection.get("touches_other_mask"))
            sources[name] = dict(frame_index=frame, mask_source=source, mask_path=str(path),
                                 image=str(target), mask_observation=observation)
    finally:
        capture.release()
    (output / "identity_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    return images


def image_projection(backend, rollout):
    """Project physical coordinates into the original fixed camera's pixels."""
    if backend.engine in {"soft_body", "cloth", "rigid"}:
        if not hasattr(backend, "_camera_projection"):
            from atw.polyworld.scene_evidence import read
            inputs = read(backend.paths["tracking"]/"input_sources.json")
            manifest_path = Path(inputs["reconstruction_manifest"])
            manifest = read(manifest_path)
            geometry = manifest_path.parent/manifest["paths"]["geometry"]
            intrinsic = np.load(geometry/"intrinsics.npy")
            pose = np.load(geometry/"camera_poses.npy")
            intrinsic = intrinsic[0] if intrinsic.ndim == 3 else intrinsic
            pose = pose[0] if pose.ndim == 3 else pose
            scale = np.asarray([backend.width, backend.height])/manifest["model_size_wh"]
            backend._camera_projection = (intrinsic, np.linalg.inv(pose), scale)
        intrinsic, world_to_camera, scale = backend._camera_projection
        def project(points):
            camera = np.asarray(points)@world_to_camera[:3,:3].T+world_to_camera[:3,3]
            pixel = camera@intrinsic.T
            return pixel[:,:2]/np.maximum(pixel[:,2:],1e-8)*scale
    else:
        scale = np.asarray([backend.width, backend.height])/rollout["image_wh"] if backend.engine == "fluid" else backend.height
        def project(points):
            return np.asarray(points)*scale
    return project


def last_in_frame_samples(backend, rollout):
    """Retain the image boundary portion when a long rollout extends offscreen."""
    project, result = image_projection(backend, rollout), {}
    for name, rows in rollout["trajectories"].items():
        if not rows:
            continue
        key = next(k for k in ("center_world", "xy_normalized", "center_xy_normalized", "center_xy_px") if k in rows[0])
        pixels = project([row[key] for row in rows])
        inside = (np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < backend.width)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < backend.height))
        if inside.any() and not inside.all():
            result[name] = [dict(rows[i], time_s=rows[i]["frame_index"] / backend.fps,
                                 image_xy_px=pixels[i].tolist()) for i in np.flatnonzero(inside)[-2:]]
    return result


def _observation_duration_seconds(backend):
    """Playback duration of the complete original observation video."""
    return float(backend.frame_count) / float(backend.fps)


def _render_rollout_video(backend, rollout_id, start_seconds, end_seconds, variant=None):
    rollout = backend.rollouts[rollout_id]
    end = backend._rollout_end(rollout) if end_seconds is None else float(end_seconds)
    start = 0. if start_seconds is None else float(start_seconds)
    if not np.isfinite([start,end]).all() or not 0 <= start < end <= backend._rollout_end(rollout):
        raise ValueError("Use a nonempty interval inside the rollout")
    end = min(end, start + _observation_duration_seconds(backend))
    capture = cv2.VideoCapture(str(backend.video))
    ok, original = capture.read()
    capture.release()
    if not ok:
        raise ValueError("Original camera frame unavailable")
    height,width = original.shape[:2]
    canvas = np.full_like(original,245)
    moving = set(rollout["trajectories"])
    removed = set(rollout["remove_objects"])
    for name in backend.objects:
        if name in moving or name in removed:
            continue
        try:
            mask = cv2.resize(backend._mask(name,0).astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool)
        except (KeyError,ValueError):
            continue
        canvas[mask] = original[mask]
    project = image_projection(backend, rollout)
    frames = range(int(np.ceil(start*backend.fps)), int(np.floor(end*backend.fps))+1)
    if len(frames) > backend.frame_count:
        frames = range(frames.start, frames.start + backend.frame_count)
    if len(frames) < 1:
        raise ValueError("No saved frame lies in the requested interval")
    directory = backend.output / rollout_id / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{variant or backend.engine}_{frames.start:05d}_{frames.stop-1:05d}.mp4"
    mask_colors = ({name: _mask_color(backend, name) for name in rollout["geometry"]}
                   if backend.engine == "soft_body" else {})
    out_height = max(2, round(height*960/width)//2*2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), backend.fps, (960, out_height+36))
    if not writer.isOpened():
        raise OSError(f"Cannot create rollout video {path}")
    try:
        for frame in frames:
            image = canvas.copy()
            for index,(name,geometry) in enumerate(rollout["geometry"].items()):
                if backend.engine == "cloth":
                    continue  # Hide cloth occlusion only in this rigid-body readout.
                if variant is not None and name != variant:
                    continue
                rows = rollout["trajectories"].get(name, [])
                if not rows or frame < min(r["frame_index"] for r in rows):
                    continue
                first = min(r["frame_index"] for r in rollout["trajectories"][name]) if backend.engine=="cloth" else 0
                offset = frame-first
                if not 0 <= offset < len(geometry):
                    continue
                points = geometry[offset]
                if backend.engine=="fluid":
                    label = backend.fit["fluid_ids"].index(name)
                    born = rollout["particle_birth"][rollout["particle_fluid"]==label]
                    points = points[born<=frame]
                if not len(points):
                    continue
                pixels = project(points)
                valid = np.isfinite(pixels).all(1)&(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
                pixels = pixels[valid].astype(int)
                color = (mask_colors[name] if backend.engine == "soft_body" else
                         tuple(int(v) for v in cv2.cvtColor(
                             np.uint8([[[index*37%180,210,190]]]), cv2.COLOR_HSV2BGR)[0,0]))
                if geometry.shape[1] == 1:
                    # A center-only export needs a motion trace; sparse endpoint dots can miss a crossing.
                    active = min(r["frame_index"] for r in rollout["trajectories"][name])-first
                    trace = project(geometry[active:offset+1, 0])
                    for left, right in zip(trace[:-1], trace[1:]):
                        if np.isfinite([left,right]).all() and np.max(np.abs([left,right])) < 1e6:
                            cv2.line(image,tuple(left.astype(int)),tuple(right.astype(int)),color,2)
                    visible = trace[np.isfinite(trace).all(1)&(trace[:,0]>=0)&(trace[:,0]<width)&(trace[:,1]>=0)&(trace[:,1]<height)]
                    if len(visible) and not len(pixels):
                        cv2.putText(image,name,tuple(visible[-1].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
                for x,y in pixels:
                    cv2.circle(image,(x,y),6 if geometry.shape[1] == 1 else max(2,round(width/480)),color,-1)
                if len(pixels):
                    center = np.mean(pixels,0).astype(int)
                    cv2.putText(image,name,tuple(center),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
            # Rigid orientation is rendered from the actual predicted axes, per independent cloth model.
            if backend.engine=="rope":
                for rope in backend.fit["ropes"]:
                    for body in rope["bodies"]:
                        name=body["object_id"]
                        row=next((r for r in rollout["trajectories"].get(name,[]) if r["frame_index"]==frame),None)
                        if row is None: continue
                        center=(np.asarray(row["xy_normalized"])*height).astype(int)
                        if np.max(np.abs(center))>1e6: continue
                        if body["role"]=="pulley":
                            cv2.circle(image,tuple(center),round(body["radius_normalized"]*height),(90,90,90),2)
                        else:
                            half=(np.asarray(body["half_extent_normalized"])*height).astype(int)
                            cv2.rectangle(image,tuple(center-half),tuple(center+half),(90,90,90),2)
                        cv2.putText(image,name,tuple(center),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,40,40),1)
            if backend.engine=="cloth":
                fits = backend.fit["objects"]
                for name,values in rollout["trajectories"].items():
                    for row in values:
                        if row["frame_index"]!=frame or "rotation_world" not in row or (variant is not None and row.get("fitted_with") != variant):
                            continue
                        half = fits[row["fitted_with"]]["obstacle_half_extents_m"][name]
                        corners = np.asarray(list(itertools.product((-1,1),repeat=3)))*half
                        corners = corners@np.asarray(row["rotation_world"]).T+row["center_world"]
                        pixel = project(corners)
                        if np.isfinite(pixel).all() and np.max(np.abs(pixel))<1e6:
                            hull = cv2.convexHull(pixel.astype(np.int32))
                            cv2.polylines(image,[hull],True,(70,70,70),2)
                            cv2.putText(image,name,tuple(pixel.mean(0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
            image = cv2.resize(image,(960,out_height))
            image = cv2.copyMakeBorder(image,36,0,0,0,cv2.BORDER_CONSTANT)
            color_note = "MASK colors" if backend.engine == "soft_body" else "ID colors"
            cv2.putText(image,f"SIMULATION {rollout_id} | {frame/backend.fps:.2f}s | {variant or backend.engine} | {color_note}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            writer.write(image)
    except Exception:
        writer.release()
        path.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
    return dict(source_kind="new_simulation", rollout_id=rollout_id, images=[str(path)],
        interval_seconds=[frames.start/backend.fps,(frames.stop-1)/backend.fps], frame_count=len(frames), fps=backend.fps, variant=variant,
        interpretation=("Simulated interval in the fixed camera; colors label IDs. " +
            ("Cloth is hidden visually but still applies forces. Rigid wireframes show modeled extents and orientation, not exact surface contact. Independent cloth fits are separate worlds."
             if backend.engine == "cloth" else "Center traces are not shape or pulley-spin evidence.")))


def render_rollout_video(backend, rollout_id, start_seconds=None, end_seconds=None,
                         object_labels=None):
    """Full result movies only for ContPhy scenes; independent fits stay separate."""
    if backend.engine not in {"cloth", "soft_body", "rope"}:
        raise ValueError("Use this scene's advertised result inspection tool")
    variants = list(backend.rollouts[rollout_id]["geometry"]) if backend.engine == "cloth" else [None]
    if not variants:
        return dict(status="unsupported",rollout_id=rollout_id,reason="No simulated cloth geometry was exported",images=[])
    results = [_render_rollout_video(backend, rollout_id, start_seconds, end_seconds, variant) for variant in variants]
    return dict(source_kind="new_simulation", rollout_id=rollout_id,
        images=[p for r in results for p in r["images"]], views=results,
        interval_seconds=results[0]['interval_seconds'],
        interpretation=("Cloth is hidden only in rendering; its forces remain. Each video is an independent fitted cloth world. Rigid wireframes are modeled poses, not exact surface contact."
                        if backend.engine == "cloth" else "Read the complete permitted simulated interval; projected geometry is model evidence."))


def render_fluid_rollout(backend, rollout_id, object_id, start_seconds=None, end_seconds=None,
                         object_labels=None):
    """Render one simulated fluid while every fluid remains active in the solver."""
    if backend.engine != "fluid":
        raise ValueError("Target-fluid rollout video is available only for the fluid engine")
    if rollout_id not in backend.rollouts:
        raise ValueError("Use an existing rollout ID")
    if object_id not in backend.fit["fluid_ids"]:
        raise ValueError("object_id must be one modeled fluid material ID, not an emitter or solid")
    rollout = backend.rollouts[rollout_id]
    geometry = np.asarray(rollout["geometry"][object_id])
    label = backend.fit["fluid_ids"].index(object_id)
    selection = np.asarray(rollout["particle_fluid"]) == label
    births = np.asarray(rollout["particle_birth"])[selection]
    if geometry.ndim != 3 or geometry.shape[1] != len(births):
        raise ValueError("Fluid particle geometry and birth records do not align")
    born = births < len(geometry)
    if not born.any():
        raise ValueError("The selected fluid has no born particles in this rollout")
    default_start = max(0., float(births[born].min()) / backend.fps - .5)
    start = default_start if start_seconds is None else float(start_seconds)
    end = backend._rollout_end(rollout) if end_seconds is None else float(end_seconds)
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= backend._rollout_end(rollout):
        raise ValueError("Use a nonempty interval inside this rollout")
    end = min(end, start + _observation_duration_seconds(backend))
    first = int(np.ceil(start * backend.fps))
    last = int(np.floor(end * backend.fps))
    if not np.any(births <= last):
        raise ValueError("The selected fluid is not emitted in the requested interval")

    capture = cv2.VideoCapture(str(backend.video))
    ok, original = capture.read()
    capture.release()
    if not ok:
        raise ValueError("Original camera frame unavailable")
    output_width = min(960, backend.width)
    output_height = max(1, round(backend.height * output_width / backend.width))
    size = (output_width, output_height)
    original = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    canvas = np.full_like(original, 245)
    removed = set(rollout["remove_objects"])
    labels = []
    for name, obj in backend.objects.items():
        if name in backend.fit["fluid_ids"] or name in removed:
            continue
        try:
            mask = cv2.resize(backend._mask(name, 0).astype(np.uint8), size,
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        except (KeyError, ValueError, OSError):
            continue
        canvas[mask] = original[mask]
        description = str(obj.get("description", "")).lower()
        if mask.any() and ("stick" in description or "container" in description or "funnel" in description):
            yy, xx = np.where(mask)
            labels.append((name, (int(np.median(xx)), int(np.median(yy)))))

    color = _mask_color(backend, object_id)
    outline_color = tuple(max(0, min(255, round(channel * .62))) for channel in color)
    solver_wh = np.asarray(rollout["image_wh"], float)
    scale = np.asarray(size, float) / solver_wh
    base_radius = max(2, round(float(backend.fit["particle_radius_pixels"]) * float(scale.mean())))
    radius = max(3, round(base_radius * 1.5))
    stride = max(1, round(backend.fps / 15))
    frames = list(range(first, last + 1, stride))
    if frames[-1] != last:
        frames.append(last)
    maximum_video_frames = max(1, int(np.floor(_observation_duration_seconds(backend) * backend.fps / stride)))
    frames = frames[:maximum_video_frames]
    last, end = frames[-1], frames[-1] / backend.fps

    def visible_particle_count(frame):
        active = (births <= frame) & np.isfinite(geometry[frame]).all(axis=1)
        pixels = geometry[frame, active] * scale
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < output_width) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] < output_height))
        return int(inside.sum())

    visible_counts = {frame: visible_particle_count(frame) for frame in range(first, last + 1)}
    visible_frames = [frame for frame, count in visible_counts.items() if count]
    if not visible_frames:
        raise ValueError("The selected fluid has no visible particles in the requested interval")
    directory = backend.output / rollout_id / "fluid_views"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{object_id}_{first:05d}_{last:05d}"
    video_path = directory / f"{stem}.mp4"
    grid_path = directory / f"{stem}.png"

    def draw(frame):
        image = canvas.copy()
        active = (births <= frame) & np.isfinite(geometry[frame]).all(axis=1)
        pixels = np.rint(geometry[frame, active] * scale).astype(int)
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < output_width) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] < output_height))
        for x, y in pixels[inside]:
            # Rendering only: preserve every simulated center while enlarging the
            # mask-colored drop and adding a one-pixel darker same-hue edge.
            cv2.circle(image, (int(x), int(y)), radius + 1, outline_color, -1, cv2.LINE_AA)
            cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        for name, (x, y) in labels:
            cv2.putText(image, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(image, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (20, 20, 20), 1, cv2.LINE_AA)
        banner = f"SIMULATION {rollout_id} | target={object_id} | t={frame/backend.fps:.2f}s"
        if removed:
            banner += " | removed=" + ",".join(sorted(removed))
        cv2.rectangle(image, (0, 0), (output_width, 30), (0, 0, 0), -1)
        cv2.putText(image, banner, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return image

    if not video_path.exists():
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 backend.fps / stride, size)
        if not writer.isOpened():
            raise OSError(f"Cannot create rollout video {video_path}")
        try:
            for frame in frames:
                writer.write(draw(frame))
        finally:
            writer.release()
    if not grid_path.exists():
        # Keep the complete-world endpoints while reserving half the sheet for the
        # target's visible interval. This prevents a late-emitted stream from
        # appearing absent in a uniformly sampled contact sheet.
        samples = sorted(set(np.linspace(first, last, 3).astype(int).tolist()) |
                         set(np.linspace(visible_frames[0], visible_frames[-1], 3).astype(int).tolist()))
        thumbs = [cv2.resize(draw(int(frame)), (output_width // 3, output_height // 2),
                             interpolation=cv2.INTER_AREA) for frame in samples]
        while len(thumbs) < 6:
            thumbs.append(thumbs[-1].copy())
        grid = np.vstack((np.hstack(thumbs[:3]), np.hstack(thumbs[3:6])))
        if not cv2.imwrite(str(grid_path), grid):
            raise OSError(f"Cannot create rollout contact sheet {grid_path}")
    return dict(source_kind="new_simulation", rollout_id=rollout_id, object_id=object_id,
        interval_seconds=[start, end], images=[str(video_path), str(grid_path)],
        visible_interval_seconds=[visible_frames[0] / backend.fps, visible_frames[-1] / backend.fps],
        visible_frame_count=len(visible_frames), max_visible_particles=max(visible_counts.values()),
        visible_dynamic_objects=[object_id], hidden_dynamic_objects=[name for name in backend.fit["fluid_ids"] if name != object_id],
        all_fluids_participated_in_simulation=True, removed_objects=sorted(removed),
        interpretation=("Only the selected target fluid is visible; all fluids still participated in the physical simulation. "
            "Use the complete video and contact sheet to judge the main stream. A pass requires the main stream to contact "
            "the stick's upper surface; proximity, endpoint near-misses, side or underside contact, and isolated splash "
            "particles do not count. "
            "For destination questions inspect the target fluid's final stable location."))
