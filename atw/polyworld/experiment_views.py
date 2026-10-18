"""Original-camera identity views and shared physical projection helpers."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


IDENTITY_PANEL_MAX_WIDTH = 960


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
