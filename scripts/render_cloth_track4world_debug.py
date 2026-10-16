#!/usr/bin/env python3
"""Render human-review videos from retained point-cloud tracking products."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atw.perception.pointcloud_reconstruction_assets import load_reconstruction_assets

from atw.perception.sparse_pointcloud import StaticScenePointCloud


CAMERA_RENDER_SCALE = 2
RIGID_MINIMUM_ICP_POINTS = 20


@dataclass(slots=True)
class ClothRenderData:
    reference_frame: int
    rgb: np.ndarray
    trajectory: np.ndarray
    trajectory_colors: np.ndarray
    canonical_colors: np.ndarray
    frame_offsets: np.ndarray
    current_uv: np.ndarray
    current_xyz: np.ndarray
    canonical_id: np.ndarray

    def frame_observations(
        self, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = int(self.frame_offsets[frame_index])
        end = int(self.frame_offsets[frame_index + 1])
        return (
            self.current_uv[start:end],
            self.current_xyz[start:end],
            self.canonical_id[start:end] >= 0,
        )


@dataclass(slots=True)
class RigidRenderData:
    reference_frame: int
    canonical_points: np.ndarray
    point_colors: np.ndarray
    identity_colors: np.ndarray
    poses: np.ndarray
    masks: np.ndarray
    mask_pixel_counts: np.ndarray
    occlusion_termination_frame: int
    occlusion_terminated: np.ndarray
    border_touching: np.ndarray
    nearest_nonborder_frames: np.ndarray
    border_area_ratios: np.ndarray
    border_suppressed: np.ndarray
    observed_point_counts: np.ndarray
    observed_coverage: np.ndarray
    numerically_converged: np.ndarray
    has_sufficient_correspondences: np.ndarray
    trajectory_indices: np.ndarray
    trajectory_colors: np.ndarray
    motion_model: str = "se3"

    def is_observed(self, frame_index: int) -> bool:
        if self.motion_model == "static":
            return bool(
                self.mask_pixel_counts[frame_index] > 0
                and not self.border_suppressed[frame_index]
                and not self.occlusion_terminated[frame_index]
                and np.isfinite(self.poses[frame_index]).all()
            )
        return bool(self.observed_point_counts[frame_index] > 0)


@dataclass(slots=True)
class DebugRenderState:
    scene_name: str
    fps: float
    base_reference: int
    base_rgb: np.ndarray
    base_depth: np.ndarray
    static_scene: StaticScenePointCloud
    tracked_region_mask_union: np.ndarray
    base_intrinsics: np.ndarray
    base_world_to_camera: np.ndarray
    cloths: dict[str, ClothRenderData]
    rigids: dict[str, RigidRenderData]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    elevation_basis: np.ndarray
    elevation_center: np.ndarray
    elevation_radius: float
    trail_length: int
    depth_min: float
    depth_max: float
    gravity_known: bool = True
    fitted_plane_vertices: np.ndarray | None = None
    fitted_plane_label: str = ""


_STATE: DebugRenderState | None = None


def _turbo_colors(count: int) -> np.ndarray:
    values = np.linspace(8, 247, count, dtype=np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(-1, 3)
    return bgr[:, ::-1].astype(np.float32) / 255.0


def _load_array(path: Path, *, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)


def _apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _project_camera_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projected = points @ intrinsics.T
    depth = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    return uv, depth


def _nearest_projected_samples(
    uv: np.ndarray,
    depth: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_uv = np.where(np.isfinite(uv), uv, -1.0)
    pixels = np.rint(safe_uv).astype(np.int32)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    source = np.flatnonzero(valid)
    if not len(source):
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty
    valid_pixels = pixels[source]
    flat = valid_pixels[:, 1] * width + valid_pixels[:, 0]
    order = np.lexsort((depth[source], flat))
    sorted_flat = flat[order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    selected = source[order[first]]
    return pixels[selected, 0], pixels[selected, 1], selected


def _draw_pixel_points(
    canvas: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    colors: np.ndarray,
    *,
    point_radius: int,
) -> None:
    height, width = canvas.shape[:2]
    colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    for offset_y in range(-point_radius, point_radius + 1):
        target_y = y + offset_y
        for offset_x in range(-point_radius, point_radius + 1):
            target_x = x + offset_x
            valid = (
                (target_x >= 0)
                & (target_x < width)
                & (target_y >= 0)
                & (target_y < height)
            )
            canvas[target_y[valid], target_x[valid]] = colors_uint8[valid]


def _load_gravity_aligned_basis(
    surface_dir: Path,
    base_camera_pose: np.ndarray,
) -> np.ndarray:
    gravity_path = (
        surface_dir
        / "gravity_estimate.json"
    )
    gravity = json.loads(gravity_path.read_text(encoding="utf-8"))
    if gravity.get("status") != "completed":
        # A display basis only; no physical gravity estimate is inferred here.
        rotation = np.asarray(base_camera_pose[:3, :3], dtype=np.float64)
        return np.stack((rotation[:, 0], -rotation[:, 2], -rotation[:, 1])).astype(np.float32)
    up = np.asarray(
        gravity["selected"]["up_direction_world"], dtype=np.float64
    )
    up /= np.linalg.norm(up)
    camera_forward = np.asarray(base_camera_pose[:3, 2], dtype=np.float64)
    front_view = camera_forward - np.dot(camera_forward, up) * up
    view = -front_view
    view /= np.linalg.norm(view)
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    return np.stack((right, view, up)).astype(np.float32)


def _fitted_plane_patch(
    summary: dict[str, Any], points: np.ndarray
) -> tuple[np.ndarray | None, str]:
    """Bound the retained plane n.xyz = offset by the tracked scene's extent."""
    if summary.get("status") != "completed":
        return None, ""
    selected = summary["selected"]
    surface_type = summary.get("scene_surface_type")
    if surface_type in {"wall_plane", "support_plane"}:
        plane = selected[surface_type]
        normal = np.asarray(plane["normal_world"], dtype=np.float64)
        offset = float(plane["offset_world"])
        label = "Fitted wall plane" if surface_type == "wall_plane" else "Fitted support plane"
    else:
        normal = np.asarray(selected["up_direction_world"], dtype=np.float64)
        offset = float(selected["full_mesh"]["support_plane_offset_world"])
        label = "Fitted support plane"
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = np.cross(normal, axis)
    tangent /= np.linalg.norm(tangent)
    basis = np.stack((tangent, np.cross(normal, tangent)))
    lower, upper = np.percentile(points @ basis.T, (1, 99), axis=0)
    padding = max(float(np.max(upper - lower)) * 0.08, 1e-3)
    lower -= padding
    upper += padding
    corners = np.asarray([
        [lower[0], lower[1]], [upper[0], lower[1]],
        [upper[0], upper[1]], [lower[0], upper[1]],
    ])
    return corners @ basis + offset * normal, label


def load_render_state(
    run_dir: Path,
    *,
    trail_length: int = 24,
) -> DebugRenderState:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    base_reference = int(metadata["base_reference_frame"])
    sources = json.loads((run_dir / 'input_sources.json').read_text())
    assets = load_reconstruction_assets(Path(sources['reconstruction_manifest']).parent)
    base_root = assets.geometry_dir
    base_rgb = _load_array(base_root / "rgb.npy", mmap=True)
    base_depth = _load_array(base_root / "depth.npy", mmap=True)
    static_scene = StaticScenePointCloud.load(
        base_root / "static_scene_points.npz"
    )
    tracked_region_mask_union = _load_array(
        base_root / "tracked_region_mask_union.npy", mmap=True
    )
    base_intrinsics = _load_array(base_root / "intrinsics.npy", mmap=True)
    base_camera_poses = _load_array(base_root / "camera_poses.npy")
    base_world_to_camera = np.linalg.inv(base_camera_poses)
    tracked_order = tuple(str(name) for name in metadata["tracked_objects"])
    canonical_root = run_dir / "canonical"
    rigid_trajectory_root = run_dir / "rigid_trajectories"
    discovered_cloths = {
        path.stem for path in canonical_root.glob("*.npz")
    } if canonical_root.is_dir() else set()
    discovered_rigids = {
        path.stem for path in rigid_trajectory_root.glob("*.npz")
    } if rigid_trajectory_root.is_dir() else set()
    cloth_names = tuple(name for name in tracked_order if name in discovered_cloths)
    rigid_names = tuple(name for name in tracked_order if name in discovered_rigids)
    if not cloth_names and not rigid_names:
        raise ValueError("run contains no retained cloth or rigid trajectories")
    cloths: dict[str, ClothRenderData] = {}
    rigids: dict[str, RigidRenderData] = {}
    bounds_chunks: list[np.ndarray] = []
    for name in cloth_names:
        with np.load(run_dir / f"canonical/{name}.npz") as archive:
            reference = int(archive["reference_frame"])
            uv = np.asarray(archive["uv"])
        rgb = base_rgb
        uv_int = np.rint(uv).astype(int)
        selected = np.flatnonzero(
            (uv_int[:, 0] % 10 == 0) & (uv_int[:, 1] % 10 == 0)
        )
        if len(selected) > 560:
            selected = selected[
                np.linspace(0, len(selected) - 1, 560).astype(int)
            ]
        with np.load(run_dir / f"associations/{name}.npz") as archive:
            frame_offsets = np.asarray(archive["frame_offsets"])
            current_uv = np.asarray(archive["current_uv"])
            current_xyz = np.asarray(archive["current_xyz"])
            canonical_id = np.asarray(archive["canonical_id"])
            image_distance = np.asarray(archive["image_distance"])
        if (run_dir / f"soft_body_surfaces/{name}.npz").is_file():
            # Show the tracked SAM3D surface, rather than substituting raw depth
            # observations at associated pixels or showing untracked new points.
            with np.load(run_dir / f"trajectories/{name}.npz") as archive:
                surface_xyz = np.asarray(archive["model_xyz"])
                surface_uv = np.asarray(archive["uv"])
                surface_confidence = np.asarray(archive["tracking_confidence"])
            chunks_xyz, chunks_uv, chunks_ids = [], [], []
            frame_offsets = [0]
            for frame in range(len(rgb)):
                keep = np.isfinite(surface_xyz[frame]).all(axis=1) & (surface_confidence[frame] >= metadata["minimum_tracking_confidence"])
                ids = np.flatnonzero(keep)
                chunks_xyz.append(surface_xyz[frame, ids])
                chunks_uv.append(surface_uv[frame, ids])
                chunks_ids.append(ids)
                frame_offsets.append(frame_offsets[-1] + len(ids))
            frame_offsets = np.asarray(frame_offsets)
            current_xyz = np.concatenate(chunks_xyz)
            current_uv = np.concatenate(chunks_uv)
            canonical_id = np.concatenate(chunks_ids)
            image_distance = np.zeros(len(canonical_id), dtype=np.float32)
        id_to_slot = np.full(len(uv), -1, dtype=np.int32)
        id_to_slot[selected] = np.arange(len(selected), dtype=np.int32)
        trajectory = np.full(
            (len(rgb), len(selected), 3), np.nan, dtype=np.float32
        )
        for frame_index in range(len(rgb)):
            start = int(frame_offsets[frame_index])
            end = int(frame_offsets[frame_index + 1])
            ids = canonical_id[start:end]
            matched = ids >= 0
            slots = id_to_slot[ids[matched]]
            selected_observations = slots >= 0
            if not np.any(selected_observations):
                continue
            slots = slots[selected_observations]
            xyz = current_xyz[start:end][matched][selected_observations]
            distances = image_distance[start:end][matched][selected_observations]
            order = np.argsort(distances)
            _, first = np.unique(slots[order], return_index=True)
            choices = order[first]
            trajectory[frame_index, slots[choices]] = xyz[choices]
        finite_sample = current_xyz[::8]
        finite_sample = finite_sample[np.isfinite(finite_sample).all(axis=1)]
        bounds_chunks.append(finite_sample)
        cloths[name] = ClothRenderData(
            reference_frame=reference,
            rgb=rgb,
            trajectory=trajectory,
            trajectory_colors=_turbo_colors(len(selected)),
            canonical_colors=_turbo_colors(len(uv)),
            frame_offsets=frame_offsets,
            current_uv=current_uv,
            current_xyz=current_xyz,
            canonical_id=canonical_id,
        )
    fallback_colors = _turbo_colors(len(rigid_names))
    for rigid_slot, name in enumerate(rigid_names):
        with np.load(
            run_dir / f"rigid_trajectories/{name}.npz"
        ) as archive:
            points = np.asarray(archive["canonical_tracking_points"])
            motion_model = str(archive["motion_model"])
            colors = np.asarray(archive["canonical_tracking_colors"])
            reference = int(archive["reference_frame"])
            poses = np.asarray(archive["poses"])
            mask_pixel_counts = np.asarray(archive["mask_pixel_counts"])
            occlusion_termination_frame = int(
                archive["occlusion_termination_frame"]
            )
            occlusion_terminated = np.asarray(
                archive["occlusion_terminated"], dtype=bool
            )
            border_touching = np.asarray(archive["border_touching"])
            nearest_nonborder_frames = np.asarray(
                archive["nearest_nonborder_frames"]
            )
            border_area_ratios = np.asarray(archive["border_area_ratios"])
            border_suppressed = np.asarray(archive["border_suppressed"])
            observed_point_counts = np.asarray(archive["observed_point_counts"])
            observed_coverage = np.asarray(archive["observed_coverage"])
            numerically_converged = np.asarray(
                archive["numerically_converged"], dtype=bool
            )
            has_sufficient_correspondences = np.asarray(
                archive["has_sufficient_correspondences"], dtype=bool
            )
        with np.load(
            assets.tracking_masks_dir / f"{name}.npz"
        ) as archive:
            masks = np.asarray(archive["masks"], dtype=bool)
        if masks.shape != base_rgb.shape[:3]:
            raise ValueError(
                f"rigid mask shape mismatch for {name}: "
                f"{masks.shape} != {base_rgb.shape[:3]}"
            )
        if colors.ndim != 2 or len(colors) != len(points) or colors.shape[1] < 3:
            colors = np.tile(fallback_colors[rigid_slot], (len(points), 1))
        else:
            colors = colors[:, :3].astype(np.float32)
            if colors.max(initial=0) > 1.0:
                colors /= 255.0
        selected = np.linspace(
            0, len(points) - 1, min(48, len(points)), dtype=np.int32
        )
        for frame_index in range(0, len(poses), 8):
            if np.isfinite(poses[frame_index]).all():
                bounds_chunks.append(
                    _apply_pose(points[::80], poses[frame_index])
                )
        rigids[name] = RigidRenderData(
            reference_frame=reference,
            canonical_points=points,
            point_colors=colors,
            identity_colors=_turbo_colors(len(points)),
            poses=poses,
            masks=masks,
            mask_pixel_counts=mask_pixel_counts,
            occlusion_termination_frame=occlusion_termination_frame,
            occlusion_terminated=occlusion_terminated,
            border_touching=border_touching,
            nearest_nonborder_frames=nearest_nonborder_frames,
            border_area_ratios=border_area_ratios,
            border_suppressed=border_suppressed,
            observed_point_counts=observed_point_counts,
            observed_coverage=observed_coverage,
            numerically_converged=numerically_converged,
            has_sufficient_correspondences=has_sufficient_correspondences,
            trajectory_indices=selected,
            trajectory_colors=_turbo_colors(len(selected)),
            motion_model=motion_model,
        )
    points = np.concatenate(bounds_chunks)
    gravity = json.loads(
        (assets.support_plane_gravity_dir / "gravity_estimate.json").read_text()
    )
    fitted_plane_vertices, fitted_plane_label = _fitted_plane_patch(gravity, points)
    lower = np.percentile(points, 1, axis=0)
    upper = np.percentile(points, 99, axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) * 0.65, 1e-3)
    elevation_basis = _load_gravity_aligned_basis(
        assets.support_plane_gravity_dir,
        base_camera_poses[0],
    )
    elevation_points = points @ elevation_basis.T
    elevation_lower, elevation_upper = np.percentile(
        elevation_points[:, (0, 2)], (1, 99), axis=0
    )
    elevation_center = (elevation_lower + elevation_upper) / 2
    elevation_radius = max(
        float(np.max(elevation_upper - elevation_lower)) * 0.65,
        1e-3,
    )
    sampled_depth = np.asarray(base_depth[:, ::8, ::8]).reshape(-1)
    sampled_depth = sampled_depth[np.isfinite(sampled_depth) & (sampled_depth > 0)]
    if not len(sampled_depth):
        raise ValueError("base reconstruction contains no valid positive depth")
    depth_min, depth_max = np.percentile(sampled_depth, (1, 99))
    return DebugRenderState(
        scene_name=str(metadata.get("scene_name", "scene 378")),
        fps=float(metadata["fps"]),
        base_reference=base_reference,
        base_rgb=base_rgb,
        base_depth=base_depth,
        static_scene=static_scene,
        tracked_region_mask_union=tracked_region_mask_union,
        base_intrinsics=base_intrinsics,
        base_world_to_camera=base_world_to_camera,
        cloths=cloths,
        rigids=rigids,
        bounds_min=center - radius,
        bounds_max=center + radius,
        elevation_basis=elevation_basis,
        elevation_center=elevation_center.astype(np.float32),
        elevation_radius=elevation_radius,
        trail_length=trail_length,
        depth_min=float(depth_min),
        depth_max=float(depth_max),
        gravity_known=gravity.get("status") == "completed",
        fitted_plane_vertices=fitted_plane_vertices,
        fitted_plane_label=fitted_plane_label,
    )


def _project_elevation_points(
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert _STATE is not None
    aligned = np.asarray(points) @ _STATE.elevation_basis.T
    top, bottom, margin = 46, 24, 30
    usable_height = height - top - bottom
    scale = min(width - 2 * margin, usable_height) / (
        2.0 * _STATE.elevation_radius
    )
    uv = np.empty((len(aligned), 2), dtype=np.float32)
    uv[:, 0] = width / 2 + (
        aligned[:, 0] - _STATE.elevation_center[0]
    ) * scale
    uv[:, 1] = top + usable_height / 2 - (
        aligned[:, 2] - _STATE.elevation_center[1]
    ) * scale
    finite_depth = aligned[:, 1][np.isfinite(aligned[:, 1])]
    depth = aligned[:, 1] - finite_depth.min() + 1e-6
    return uv, depth


def _draw_elevation_trail(
    canvas: np.ndarray,
    trail: np.ndarray,
    color: np.ndarray,
    *,
    thickness: int,
) -> None:
    uv, _ = _project_elevation_points(
        trail,
        width=canvas.shape[1],
        height=canvas.shape[0],
    )
    valid = np.isfinite(uv).all(axis=1)
    pixels = np.rint(uv[valid]).astype(np.int32)
    if len(pixels) < 2:
        return
    rgb = tuple(int(value) for value in np.clip(color * 255, 0, 255))
    cv2.polylines(canvas, [pixels], False, rgb, thickness, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pixels[-1]), thickness + 1, rgb, -1, cv2.LINE_AA)


def _draw_height_axis(canvas: np.ndarray, label: str = "HEIGHT") -> None:
    bottom = canvas.shape[0] - 28
    cv2.arrowedLine(
        canvas,
        (28, bottom),
        (28, bottom - 70),
        (90, 255, 110),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    cv2.putText(
        canvas,
        label,
        (38, bottom - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 255, 110),
        1,
        cv2.LINE_AA,
    )


def _draw_fitted_elevation_plane(canvas: np.ndarray) -> None:
    assert _STATE is not None
    if _STATE.fitted_plane_vertices is None:
        return
    uv, _ = _project_elevation_points(
        _STATE.fitted_plane_vertices, width=canvas.shape[1], height=canvas.shape[0]
    )
    polygon = np.rint(uv).astype(np.int32).reshape(-1, 1, 2)
    color = (80, 220, 255)  # RGB; diagnostic overlay stays visible through clouds.
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, polygon, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, dst=canvas)
    # Keep the outline even for an edge-on plane, whose projection is a line.
    cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas, _STATE.fitted_plane_label + " (cyan)",
        (14, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 1, cv2.LINE_AA,
    )


def _render_elevation_frame_chunk(task: tuple[list[int], str]) -> list[int]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    assert _STATE is not None
    frame_indices, frame_dir_value = task
    frame_dir = Path(frame_dir_value)
    panel_size = 800
    for frame_index in frame_indices:
        _, _, world_points, world_colors = _frame_composited_rgb_point_clouds(
            frame_index
        )
        uv, depth = _project_elevation_points(
            world_points,
            width=panel_size,
            height=panel_size,
        )
        x, y, selected = _nearest_projected_samples(
            uv,
            depth,
            width=panel_size,
            height=panel_size,
        )
        left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        _draw_pixel_points(
            left,
            x,
            y,
            world_colors[selected],
            point_radius=1,
        )
        _draw_fitted_elevation_plane(left)
        right = left.copy()
        trail_start = max(0, frame_index - _STATE.trail_length + 1)
        for cloth in _STATE.cloths.values():
            for point_index in range(cloth.trajectory.shape[1]):
                trail = cloth.trajectory[
                    trail_start : frame_index + 1, point_index
                ]
                trail = trail[np.isfinite(trail).all(axis=1)]
                if len(trail) >= 2:
                    _draw_elevation_trail(
                        right,
                        trail,
                        cloth.trajectory_colors[point_index],
                        thickness=2,
                    )
        for rigid in _STATE.rigids.values():
            selected_points = rigid.canonical_points[rigid.trajectory_indices]
            for slot, point in enumerate(selected_points):
                trail = []
                for trail_frame in range(trail_start, frame_index + 1):
                    pose = rigid.poses[trail_frame]
                    if rigid.is_observed(trail_frame) and np.isfinite(pose).all():
                        trail.append(_apply_pose(point[None], pose)[0])
                if len(trail) >= 2:
                    _draw_elevation_trail(
                        right,
                        np.asarray(trail),
                        rigid.trajectory_colors[slot],
                        thickness=2,
                    )
        view_label = (
            "Gravity-aligned back elevation" if _STATE.gravity_known
            else "Camera-oriented view (gravity not estimated)"
        )
        _draw_title(
            left,
            f"{view_label} | frame {frame_index:03d} | "
            "no trails",
        )
        _draw_title(
            right,
            f"{view_label} | frame {frame_index:03d} | "
            f"{_STATE.trail_length}-frame colored trails",
        )
        axis_label = "HEIGHT" if _STATE.gravity_known else "CAMERA UP"
        _draw_height_axis(left, axis_label)
        _draw_height_axis(right, axis_label)
        output = np.concatenate((left, right), axis=1)
        frame_path = frame_dir / f"{frame_index:05d}.png"
        if not cv2.imwrite(
            str(frame_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        ):
            raise OSError(f"failed to write {frame_path}")
    return frame_indices


def _draw_title(image: np.ndarray, text: str) -> None:
    cv2.putText(
        image,
        text,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _rigid_tracking_status(
    name: str,
    rigid: RigidRenderData,
    frame_index: int,
) -> str:
    display_name = name.replace("_", " ")
    observed_count = int(rigid.observed_point_counts[frame_index])
    sample_count = len(rigid.trajectory_indices)
    canonical_count = len(rigid.canonical_points)
    if bool(rigid.occlusion_terminated[frame_index]):
        return (
            f"{display_name} | tracking terminated by occlusion at "
            f"F{rigid.occlusion_termination_frame:03d}"
        )
    if bool(rigid.border_suppressed[frame_index]):
        ratio = 100.0 * float(rigid.border_area_ratios[frame_index])
        baseline_frame = int(rigid.nearest_nonborder_frames[frame_index])
        return (
            f"{display_name} | border clipped | area {ratio:.1f}% of "
            f"F{baseline_frame:03d} | omitted"
        )
    if rigid.motion_model == "static" and rigid.is_observed(frame_index):
        return f"{display_name} | static reference pose"
    if observed_count == 0:
        return (
            f"{display_name} | not visible | "
            f"trails {sample_count}/{canonical_count}"
        )
    if observed_count < RIGID_MINIMUM_ICP_POINTS:
        return (
            f"{display_name} | visible geom N/A | Track4World initialization: "
            f"insufficient points "
            f"({observed_count}<{RIGID_MINIMUM_ICP_POINTS}) | "
            f"trails {sample_count}/{canonical_count}"
        )
    coverage = 100.0 * float(rigid.observed_coverage[frame_index])
    if bool(rigid.numerically_converged[frame_index]):
        pose_status = "ICP converged"
    else:
        pose_status = "ICP final iterate"
    if not bool(rigid.has_sufficient_correspondences[frame_index]):
        pose_status += " | correspondences insufficient"
    border_status = ""
    if bool(rigid.border_touching[frame_index]):
        ratio = float(rigid.border_area_ratios[frame_index])
        border_status = (
            f" | border kept {100.0 * ratio:.1f}%"
            if np.isfinite(ratio)
            else " | border kept: no interior baseline"
        )
    return (
        f"{display_name} | visible geom {coverage:.1f}% | {pose_status}"
        f"{border_status} | "
        f"trails {sample_count}/{canonical_count}"
    )


def _draw_status_lines(image: np.ndarray, lines: list[str]) -> None:
    for line_index, line in enumerate(lines):
        origin = (14, 50 + 19 * line_index)
        cv2.putText(
            image,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_world_trail(
    canvas: np.ndarray,
    trail_world: np.ndarray,
    color: np.ndarray,
    *,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    thickness: int,
) -> None:
    trail_camera = _apply_pose(trail_world, world_to_camera)
    uv, depth = _project_camera_points(trail_camera, intrinsics)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
    if np.count_nonzero(valid) < 2:
        return
    pixels = np.rint(uv[valid]).astype(np.int32)
    rgb = tuple(int(value) for value in np.clip(color * 255.0, 0, 255))
    cv2.polylines(
        canvas,
        [pixels.reshape(-1, 1, 2)],
        isClosed=False,
        color=rgb,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    endpoint = pixels[-1]
    cv2.circle(canvas, tuple(endpoint), thickness + 1, rgb, -1, cv2.LINE_AA)


def _project_topdown_points(
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthographically project world points onto XZ while looking along +Y."""
    assert _STATE is not None
    center = (_STATE.bounds_min + _STATE.bounds_max) / 2.0
    span = np.maximum(_STATE.bounds_max - _STATE.bounds_min, 1e-8)
    scale = 0.92 * min(width / float(span[0]), height / float(span[2]))
    uv = np.column_stack(
        (
            (points[:, 0] - center[0]) * scale + width / 2.0,
            (points[:, 2] - center[2]) * scale + height / 2.0,
        )
    )
    camera_y = _STATE.bounds_min[1] - span[1]
    depth = points[:, 1] - camera_y
    return uv, depth


def _draw_topdown_trail(
    canvas: np.ndarray,
    trail_world: np.ndarray,
    color: np.ndarray,
    *,
    thickness: int,
) -> None:
    height, width = canvas.shape[:2]
    uv, _ = _project_topdown_points(
        trail_world,
        width=width,
        height=height,
    )
    valid = np.isfinite(uv).all(axis=1)
    if np.count_nonzero(valid) < 2:
        return
    pixels = np.rint(uv[valid]).astype(np.int32)
    rgb = tuple(int(value) for value in np.clip(color * 255.0, 0, 255))
    cv2.polylines(
        canvas,
        [pixels.reshape(-1, 1, 2)],
        isClosed=False,
        color=rgb,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(canvas, tuple(pixels[-1]), thickness + 1, rgb, -1, cv2.LINE_AA)


def _frame_composited_rgb_point_clouds(
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine one static background with current tracked-object geometry."""
    assert _STATE is not None
    background_world = _STATE.static_scene.world_points
    background_colors = _STATE.static_scene.colors.astype(np.float32) / 255.0
    world_to_camera = _STATE.base_world_to_camera[frame_index]
    background_camera = _apply_pose(background_world, world_to_camera)
    background_uv, background_depth = _project_camera_points(
        background_camera, _STATE.base_intrinsics[frame_index]
    )
    pixels = np.rint(np.where(np.isfinite(background_uv), background_uv, -1)).astype(
        np.int32
    )
    height, width = _STATE.tracked_region_mask_union.shape[1:]
    in_frame = (
        np.isfinite(background_uv).all(axis=1)
        & np.isfinite(background_depth)
        & (background_depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    visible_background = np.zeros(len(background_world), dtype=bool)
    visible_ids = np.flatnonzero(in_frame)
    visible_pixels = pixels[visible_ids]
    visible_background[visible_ids] = ~_STATE.tracked_region_mask_union[
        frame_index, visible_pixels[:, 1], visible_pixels[:, 0]
    ]
    camera_world_chunks = [background_world[visible_background]]
    camera_color_chunks = [background_colors[visible_background]]
    world_chunks = [background_world]
    world_color_chunks = [background_colors]
    for cloth in _STATE.cloths.values():
        uv, xyz, _ = cloth.frame_observations(frame_index)
        valid = np.isfinite(xyz).all(axis=1)
        if not np.any(valid):
            continue
        pixels = np.rint(uv[valid]).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, cloth.rgb.shape[2] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, cloth.rgb.shape[1] - 1)
        current_colors = (
            cloth.rgb[frame_index, pixels[:, 1], pixels[:, 0]].astype(np.float32)
            / 255.0
        )
        camera_world_chunks.append(xyz[valid])
        camera_color_chunks.append(current_colors)
        world_chunks.append(xyz[valid])
        world_color_chunks.append(current_colors)
    for rigid in _STATE.rigids.values():
        if not rigid.is_observed(frame_index):
            continue
        pose = rigid.poses[frame_index]
        if not np.isfinite(pose).all():
            continue
        posed_world = _apply_pose(rigid.canonical_points, pose)
        camera_world_chunks.append(posed_world)
        camera_color_chunks.append(rigid.point_colors)
        world_chunks.append(posed_world)
        world_color_chunks.append(rigid.point_colors)
    camera_world = np.concatenate(camera_world_chunks)
    combined_world = np.concatenate(world_chunks)
    return (
        _apply_pose(camera_world, world_to_camera),
        np.concatenate(camera_color_chunks),
        combined_world,
        np.concatenate(world_color_chunks),
    )


def _render_camera_frame_chunk(task: tuple[list[int], str]) -> list[int]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    assert _STATE is not None
    frame_indices, frame_dir_value = task
    frame_dir = Path(frame_dir_value)
    for frame_index in frame_indices:
        source_rgb = np.asarray(_STATE.base_rgb[frame_index])
        source_height, source_width = source_rgb.shape[:2]
        height = source_height * CAMERA_RENDER_SCALE
        width = source_width * CAMERA_RENDER_SCALE
        rgb_panel = cv2.resize(
            source_rgb,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        camera_panel = np.zeros_like(rgb_panel)
        topdown_panel = np.zeros_like(rgb_panel)
        camera_points, camera_colors, world_points, world_colors = (
            _frame_composited_rgb_point_clouds(frame_index)
        )
        intrinsics = np.asarray(_STATE.base_intrinsics[frame_index]).copy()
        intrinsics[:2] *= CAMERA_RENDER_SCALE
        background_uv, background_depth = _project_camera_points(
            camera_points,
            intrinsics,
        )
        x, y, selected = _nearest_projected_samples(
            background_uv,
            background_depth,
            width=width,
            height=height,
        )
        _draw_pixel_points(
            camera_panel,
            x,
            y,
            camera_colors[selected],
            point_radius=1,
        )
        topdown_uv, topdown_depth = _project_topdown_points(
            world_points,
            width=width,
            height=height,
        )
        topdown_x, topdown_y, topdown_selected = _nearest_projected_samples(
            topdown_uv,
            topdown_depth,
            width=width,
            height=height,
        )
        _draw_pixel_points(
            topdown_panel,
            topdown_x,
            topdown_y,
            world_colors[topdown_selected],
            point_radius=1,
        )
        world_to_camera = _STATE.base_world_to_camera[frame_index]

        trail_start = max(0, frame_index - _STATE.trail_length + 1)
        for cloth in _STATE.cloths.values():
            for point_index in range(cloth.trajectory.shape[1]):
                trail = cloth.trajectory[trail_start : frame_index + 1, point_index]
                trail = trail[np.isfinite(trail).all(axis=1)]
                if len(trail) >= 2:
                    _draw_world_trail(
                        camera_panel,
                        trail,
                        cloth.trajectory_colors[point_index],
                        world_to_camera=world_to_camera,
                        intrinsics=intrinsics,
                        thickness=2,
                    )
                    _draw_topdown_trail(
                        topdown_panel,
                        trail,
                        cloth.trajectory_colors[point_index],
                        thickness=2,
                    )
        for rigid in _STATE.rigids.values():
            if not rigid.is_observed(frame_index):
                continue
            for slot, point_index in enumerate(rigid.trajectory_indices):
                trail = []
                for trail_frame in range(trail_start, frame_index + 1):
                    pose = rigid.poses[trail_frame]
                    if rigid.is_observed(trail_frame) and np.isfinite(pose).all():
                        trail.append(
                            _apply_pose(
                                rigid.canonical_points[point_index][None], pose
                            )[0]
                        )
                if len(trail) >= 2:
                    _draw_world_trail(
                        camera_panel,
                        np.asarray(trail),
                        rigid.trajectory_colors[slot],
                        world_to_camera=world_to_camera,
                        intrinsics=intrinsics,
                        thickness=2,
                    )
                    _draw_topdown_trail(
                        topdown_panel,
                        np.asarray(trail),
                        rigid.trajectory_colors[slot],
                        thickness=2,
                    )

        _draw_title(rgb_panel, f"RGB video | frame {frame_index:03d}")
        _draw_title(
            camera_panel,
            "Camera-view RGB point cloud | SAM3D mesh replaces rigid-mask geometry",
        )
        _draw_title(
            topdown_panel,
            "Fixed top-down RGB point cloud | SAM3D mesh replaces rigid masks",
        )
        status_lines = [
            _rigid_tracking_status(name, rigid, frame_index)
            for name, rigid in _STATE.rigids.items()
        ]
        _draw_status_lines(camera_panel, status_lines)
        _draw_status_lines(topdown_panel, status_lines)
        combined = np.concatenate((rgb_panel, camera_panel, topdown_panel), axis=1)
        frame_path = frame_dir / f"{frame_index:05d}.png"
        if not cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write {frame_path}")
    return frame_indices


def _render_depth_frame_chunk(task: tuple[list[int], str]) -> list[int]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    assert _STATE is not None
    frame_indices, frame_dir_value = task
    frame_dir = Path(frame_dir_value)
    scale = max(_STATE.depth_max - _STATE.depth_min, 1e-8)
    for frame_index in frame_indices:
        depth = np.asarray(_STATE.base_depth[frame_index])
        valid = np.isfinite(depth) & (depth > 0)
        normalized = np.zeros(depth.shape, dtype=np.float32)
        normalized[valid] = np.clip(
            (depth[valid] - _STATE.depth_min) / scale,
            0.0,
            1.0,
        )
        color_indices = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(color_indices, cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        colored[~valid] = 0
        _draw_title(
            colored,
            (
                f"Reconstructed depth | frame {frame_index:03d} | "
                f"fixed range {_STATE.depth_min:.3f}-{_STATE.depth_max:.3f}"
            ),
        )
        frame_path = frame_dir / f"{frame_index:05d}.png"
        if not cv2.imwrite(str(frame_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write {frame_path}")
    return frame_indices


def _render_named_frame_chunk(
    task: tuple[str, list[int], str],
) -> tuple[str, float, int]:
    view_name, frame_indices, frame_dir = task
    renderers = {
        "elevation": _render_elevation_frame_chunk,
        "camera": _render_camera_frame_chunk,
        "depth": _render_depth_frame_chunk,
    }
    started = time.perf_counter()
    renderers[view_name]((frame_indices, frame_dir))
    return view_name, time.perf_counter() - started, len(frame_indices)


def _encode_rendered_frames(
    *,
    frame_dir: Path,
    output_path: Path,
    frame_count: int,
    fps: float,
) -> float:
    import imageio.v2 as imageio

    started = time.perf_counter()
    temporary_output = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.partial.mp4"
    )
    writer = imageio.get_writer(
        temporary_output,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_params=["-threads", "4"],
    )
    try:
        for frame_index in range(frame_count):
            writer.append_data(imageio.imread(frame_dir / f"{frame_index:05d}.png"))
    finally:
        writer.close()
    temporary_output.replace(output_path)
    return time.perf_counter() - started


def render_debug_videos(
    *,
    run_dir: Path,
    elevation_output: Path,
    camera_output: Path,
    depth_output: Path,
    workers: int,
) -> dict[str, Any]:
    """Render all review videos with one shared, bounded process pool."""

    if workers < 1:
        raise ValueError("workers must be positive")
    outputs = {
        "elevation": elevation_output,
        "camera": camera_output,
        "depth": depth_output,
    }
    for output_path in outputs.values():
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite debug video: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    global _STATE
    if _STATE is None:
        _STATE = load_render_state(run_dir)
    frame_count = len(_STATE.base_rgb)
    worker_count = min(workers, frame_count * len(outputs))
    chunk_count = min(frame_count, max(worker_count * 2, 1))
    temporary_dirs = {
        name: Path(
            tempfile.mkdtemp(prefix=f".{name}-frames-", dir=output.parent)
        )
        for name, output in outputs.items()
    }
    render_started = time.perf_counter()
    task_seconds = {name: 0.0 for name in outputs}
    rendered_frames = {name: 0 for name in outputs}
    try:
        tasks = []
        chunks = [
            chunk.tolist()
            for chunk in np.array_split(np.arange(frame_count), chunk_count)
            if len(chunk)
        ]
        for chunk in chunks:
            tasks.extend(
                (view_name, chunk, str(frame_dir))
                for view_name, frame_dir in temporary_dirs.items()
            )
        context = mp.get_context("fork")
        with context.Pool(worker_count) as pool:
            for view_name, elapsed, count in pool.imap_unordered(
                _render_named_frame_chunk, tasks
            ):
                task_seconds[view_name] += elapsed
                rendered_frames[view_name] += count
        render_seconds = time.perf_counter() - render_started

        encode_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(outputs)) as executor:
            futures = {
                name: executor.submit(
                    _encode_rendered_frames,
                    frame_dir=temporary_dirs[name],
                    output_path=output_path,
                    frame_count=frame_count,
                    fps=_STATE.fps,
                )
                for name, output_path in outputs.items()
            }
            encode_seconds_by_view = {
                name: future.result() for name, future in futures.items()
            }
        encode_seconds = time.perf_counter() - encode_started
    finally:
        for temporary in temporary_dirs.values():
            shutil.rmtree(temporary, ignore_errors=True)
        _STATE = None

    timing: dict[str, Any] = {
        "worker_count": worker_count,
        "frame_count": frame_count,
        "parallel_all_views_render_seconds": render_seconds,
        "parallel_all_views_encode_seconds": encode_seconds,
        "render_debug_video_seconds": time.perf_counter() - started,
    }
    for name in outputs:
        timing[f"{name}_render_task_cpu_seconds"] = task_seconds[name]
        timing[f"{name}_rendered_frame_count"] = rendered_frames[name]
        timing[f"{name}_video_encode_seconds"] = encode_seconds_by_view[name]
    debug_dir = elevation_output.parent
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "render_timing.json").write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )
    return timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--elevation-output", type=Path)
    parser.add_argument("--camera-output", type=Path)
    parser.add_argument("--depth-output", type=Path)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 24),
        help="total CPU workers shared by all three video renderers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    debug_dir = args.debug_dir.resolve() if args.debug_dir else run_dir / "debug"
    elevation_output = (
        args.elevation_output.resolve()
        if args.elevation_output is not None
        else debug_dir / "pointcloud_tracks_elevation.mp4"
    )
    camera_output = (
        args.camera_output.resolve()
        if args.camera_output is not None
        else debug_dir / "pointcloud_tracks_camera.mp4"
    )
    depth_output = (
        args.depth_output.resolve()
        if args.depth_output is not None
        else debug_dir / "reconstructed_depth.mp4"
    )
    timing = render_debug_videos(
        run_dir=run_dir,
        elevation_output=elevation_output,
        camera_output=camera_output,
        depth_output=depth_output,
        workers=args.workers,
    )
    if (run_dir / "soft_body_surfaces").is_dir():
        from scripts.render_soft_body_tracking_debug import render_soft_body_review

        started = time.perf_counter()
        render_soft_body_review(run_dir, output=debug_dir / "soft_body_global_review.mp4")
        timing["soft_body_review_seconds"] = time.perf_counter() - started
    (debug_dir / "render_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    (debug_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
