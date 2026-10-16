#!/usr/bin/env python3
"""Shared scene data, mask operations and reference selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


from atw.perception.cloth_reference_selector import ReferenceFrameSelection
from atw.perception.object_reconstruction import mask_touches_frame_edge, masks_touch, read_packed_mask_archive, select_reconstruction_frame
from atw.perception.pointcloud_tracking_input import PointCloudTrackingInput
from atw.perception.track4world_tool import SceneGeometryResult
from atw.perception.rigid_tracking import truncate_mask_track_at_confirmed_occlusion
from atw.perception.sparse_pointcloud import build_static_scene_pointcloud


from atw.perception.pipeline.recording import write_json as _write_json, timed_stage


@dataclass(frozen=True, slots=True)
class TrackedObjectSpec:
    name: str
    track_id: str


@dataclass(frozen=True, slots=True)
class SceneObjects:
    rigids: tuple[TrackedObjectSpec, ...]
    deformables: tuple[TrackedObjectSpec, ...]

    @property
    def all(self) -> tuple[TrackedObjectSpec, ...]:
        return self.rigids + self.deformables

    @classmethod
    def from_input(cls, data: PointCloudTrackingInput) -> SceneObjects:
        unsupported = [o.target_id for o in data.objects if o.physical_type == "rope"]
        if unsupported:
            raise ValueError(
                f"Rope identities support 2D segmentation only; 3D reconstruction is unavailable: {unsupported}"
            )
        result = cls(
            tuple(
                TrackedObjectSpec(o.target_id, o.track_id) for o in data.rigid_objects
            ),
            tuple(
                TrackedObjectSpec(o.target_id, o.track_id)
                for o in (*data.cloth_objects, *data.soft_body_objects)
            ),
        )
        if not result.all:
            raise ValueError("SAM3 handoff contains no trackable physical objects")
        return result


@dataclass(slots=True)
class SceneData:
    frames: np.ndarray
    fps: float
    tracks: dict[str, dict[int, np.ndarray]]
    geometry: SceneGeometryResult
    selections: dict[str, ReferenceFrameSelection]
    masks: dict[str, np.ndarray]


def load_tracks_and_targets(
    tracking_input: PointCloudTrackingInput, *, objects: SceneObjects
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[str, Any]]]:
    tracks = read_packed_mask_archive(
        tracking_input.mask_archive_path, tracking_input.mask_index_path
    )
    inventory = json.loads(tracking_input.inventory_path.read_text(encoding="utf-8"))
    inventory_objects = inventory.get("objects", [])
    targets = {
        str(item["target_id"]): dict(item)
        for item in inventory_objects
        if isinstance(item, Mapping) and "target_id" in item
    }
    for spec in objects.all:
        if spec.track_id not in tracks:
            raise KeyError(f"missing mask track {spec.track_id}")
        if spec.name not in targets:
            raise KeyError(f"missing inventory target {spec.name}")
    return (tracks, targets)


def select_area_reference_frames(
    *,
    frames_rgb: np.ndarray,
    tracks: Mapping[str, Mapping[int, np.ndarray]],
    output_dir: Path,
    timings: dict[str, float],
    specs: tuple[TrackedObjectSpec, ...] | None = None,
    objects: SceneObjects,
) -> dict[str, ReferenceFrameSelection]:
    """Select each object by non-edge, non-contact, then area priority."""
    specs = objects.deformables if specs is None else specs
    selections: dict[str, ReferenceFrameSelection] = {}
    for spec in specs:
        with timed_stage(timings, f"area_reference_selection_{spec.name}"):
            eligible_track = dict(tracks[spec.track_id])
            termination_frame = -1
            if spec in objects.rigids:
                eligible_track, termination_frame = (
                    truncate_mask_track_at_confirmed_occlusion(
                        eligible_track, frame_count=len(frames_rgb)
                    )
                )
            eligible_tracks = dict(tracks)
            eligible_tracks[spec.track_id] = eligible_track
            selected = select_reconstruction_frame(spec.track_id, eligible_tracks)
            rows: list[dict[str, Any]] = []
            for frame_index, raw_mask in sorted(eligible_track.items()):
                mask = np.asarray(raw_mask, dtype=bool)
                area = int(np.count_nonzero(mask))
                if area == 0:
                    continue
                touches_other = any(
                    (
                        masks_touch(
                            mask, np.asarray(other_frames[frame_index], dtype=bool)
                        )
                        for other_id, other_frames in tracks.items()
                        if other_id != spec.track_id
                        and frame_index in other_frames
                        and np.any(other_frames[frame_index])
                    )
                )
                rows.append(
                    {
                        "frame_index": int(frame_index),
                        "mask_area": area,
                        "touches_other_mask": touches_other,
                        "touches_frame_edge": mask_touches_frame_edge(mask),
                    }
                )
            has_non_edge = any(not row["touches_frame_edge"] for row in rows)
            for row in rows:
                row["edge_eligible"] = (
                    not row["touches_frame_edge"] if has_non_edge else True
                )
            has_non_contact = any(
                (
                    row["edge_eligible"] and (not row["touches_other_mask"])
                    for row in rows
                )
            )
            for row in rows:
                row["final_eligible"] = bool(
                    row["edge_eligible"]
                    and (not row["touches_other_mask"] if has_non_contact else True)
                )
                row["selected"] = row["frame_index"] == selected.frame_index
            ranked = sorted(
                (row for row in rows if row["final_eligible"]),
                key=lambda row: (-row["mask_area"], row["frame_index"]),
            )
            if not ranked or ranked[0]["frame_index"] != selected.frame_index:
                raise RuntimeError(
                    f"area diagnostics disagree with selector for {spec.name}"
                )
            alternatives = tuple((int(row["frame_index"]) for row in ranked[1:4]))
            diagnostics = selected.to_dict()
            diagnostics["strategy"] = "largest_non_edge_non_contact_mask"
            diagnostics["occlusion_termination_frame"] = (
                termination_frame if termination_frame >= 0 else None
            )
            selection = ReferenceFrameSelection(
                target_id=spec.name,
                reference_frame=selected.frame_index,
                confidence=1.0,
                reason="deterministic largest mask after preferring frames without frame-edge truncation, then without contact",
                alternative_frames=alternatives,
                active_frame_count=selected.positive_mask_frames,
                contact_sheets=(),
                agent_response=diagnostics,
            )
            selections[spec.name] = selection
    _write_json(
        output_dir / "reference_frame_selections.json",
        {name: selection.to_dict() for name, selection in selections.items()},
    )
    return selections


def load_saved_reference_frames(
    path: Path, *, objects: SceneObjects
) -> dict[str, ReferenceFrameSelection]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("saved reference selections must be a JSON object")
    selections = {
        name: ReferenceFrameSelection.from_dict(payload[name])
        for name in (spec.name for spec in objects.all)
    }
    return selections


def resize_tracking_mask(
    mask: np.ndarray, output_size_wh: tuple[int, int]
) -> np.ndarray:
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        output_size_wh,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _save_result_geometry(
    directory: Path, result: SceneGeometryResult, tracked_region_masks: np.ndarray
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "rgb": result.rgb,
        "intrinsics": result.intrinsics,
        "camera_poses": result.camera_poses,
        "depth": result.camera_points[..., 2],
        "geometry_confidence": result.geometry_confidence,
        "geometry_valid_threshold_0": result.geometry_valid,
    }
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array)
    static_background = build_static_scene_pointcloud(
        world_points=result.world_points,
        rgb=result.rgb,
        tracked_region_masks=tracked_region_masks,
        geometry_valid=result.geometry_valid,
        sample_count=16,
        background_stride=2,
    )
    static_background.save(directory / "static_scene_points.npz")
    np.save(directory / "tracked_region_mask_union.npy", tracked_region_masks)


def resized_tracking_masks(
    *,
    tracks: Mapping[str, Mapping[int, np.ndarray]],
    frame_count: int,
    output_size_wh: tuple[int, int],
    objects: SceneObjects,
) -> dict[str, np.ndarray]:
    width, height = output_size_wh
    output: dict[str, np.ndarray] = {}
    for spec in objects.all:
        masks = np.zeros((frame_count, height, width), dtype=bool)
        track = tracks[spec.track_id]
        for frame_index, mask in track.items():
            if 0 <= frame_index < frame_count:
                masks[frame_index] = resize_tracking_mask(mask, output_size_wh)
        output[spec.name] = masks
    return output


def load_complete_canonical_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the complete rigid mesh retained as the canonical GLB."""
    import trimesh

    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"unsupported canonical mesh type: {type(mesh).__name__}")
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
