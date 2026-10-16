#!/usr/bin/env python3
"""Temporal rigid and deformable tracking over one reconstructed scene."""

from __future__ import annotations

import csv
import gc
import json
import logging
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np


from atw.perception.cloth_reference_selector import ReferenceFrameSelection
from atw.perception.cloth_mesh_tracking import ClothReferenceMesh, ConstrainedClothTrajectory, build_cloth_reference_mesh, evaluate_cloth_mesh_observation, propagate_constrained_cloth_tracks
from atw.perception.pointcloud_tracking_input import load_pointcloud_tracking_input
from atw.perception.pointcloud_reconstruction_assets import load_reconstruction_assets
from atw.perception.rigid_motion import classify_rigid_motion
from atw.perception.soft_body_tracking import reference_surface_from_mesh, track_soft_body_surface
from atw.perception.soft_body_motion import local_soft_body_motion
from atw.perception.track4world_tool import CanonicalTrajectory, ObservedPointAssociations, SceneGeometryResult, geometry_completeness, load_adjacent_track4world_cache, load_track4world_da3_model, read_video_rgb, run_adjacent_track4world_tracking, save_adjacent_track4world_cache, sample_canonical_tracks_on_observed_mask
from atw.perception.tracking_mask_policy import TrackingMaskFramePolicy, classify_tracking_mask_frames


GEOMETRY_THRESHOLDS = (0.0, 1.0, 2.0, 3.0)


from atw.perception.pipeline.recording import write_json as _write_json, timed_stage, track4world_load_timings

from atw.perception.pipeline.config import TrackingConfig
from atw.perception.pipeline.scene import SceneObjects, SceneData, TrackedObjectSpec, load_complete_canonical_mesh, load_saved_reference_frames, load_tracks_and_targets, resize_tracking_mask
from atw.perception.pipeline.surface import support_constraint_from_gravity_summary
from atw.perception.pipeline.rigid import track_and_save_rigid_objects


@dataclass(slots=True)
class CanonicalDeformable:
    spec: TrackedObjectSpec
    reference_frame: int
    mask: np.ndarray
    uv: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    geometry_confidence: np.ndarray
    physical_type: str = "cloth"


def apply_deformable_mask_policy(
    trajectory: CanonicalTrajectory, masks: np.ndarray, *,
    reference_frame: int | None = None,
) -> TrackingMaskFramePolicy:
    """Omit clipped border output while retaining canonical initialization."""
    policy = classify_tracking_mask_frames(masks)
    if reference_frame is not None:
        if not 0 <= reference_frame < len(policy.border_suppressed):
            raise ValueError("reference frame is outside the deformable trajectory")
        suppressed = policy.border_suppressed.copy()
        suppressed[reference_frame] = False
        policy = replace(policy, border_suppressed=suppressed)
    trajectory.uv[policy.border_suppressed] = np.nan
    trajectory.model_xyz[policy.border_suppressed] = np.nan
    trajectory.tracking_confidence[policy.border_suppressed] = np.nan
    return policy


def create_canonical_deformables(
    *,
    geometry_result: SceneGeometryResult,
    selections: Mapping[str, ReferenceFrameSelection],
    tracks: Mapping[str, Mapping[int, np.ndarray]],
    soft_body_models: Mapping[str, Path] | None = None,
    objects: SceneObjects,
) -> dict[str, CanonicalDeformable]:
    canonical: dict[str, CanonicalDeformable] = {}
    for spec in objects.deformables:
        reference_frame = selections[spec.name].reference_frame
        source_mask = tracks[spec.track_id].get(reference_frame)
        if source_mask is None:
            raise KeyError(
                f"{spec.name} has no tracked mask at selected frame {reference_frame}"
            )
        mask = resize_tracking_mask(
            source_mask, geometry_result.transform.output_size_wh
        )
        reference_camera = geometry_result.camera_points[reference_frame]
        valid = (
            mask
            & np.isfinite(reference_camera).all(axis=-1)
            & (reference_camera[..., 2] > 0)
            & np.isfinite(geometry_result.world_points[reference_frame]).all(axis=-1)
        )
        y, x = np.nonzero(valid)
        uv = np.column_stack((x, y)).astype(np.float32)
        xyz = geometry_result.world_points[reference_frame, y, x]
        physical_type = "cloth"
        if soft_body_models is not None and spec.name in soft_body_models:
            model_dir = soft_body_models[spec.name]
            vertices, faces = load_complete_canonical_mesh(
                model_dir / "canonical_metric.glb"
            )
            with np.load(model_dir / "canonical_rigid_model.npz") as archive:
                pose = np.asarray(archive["reference_pose"])
            uv, xyz = reference_surface_from_mesh(
                vertices_world=vertices @ pose[:3, :3].T + pose[:3, 3],
                faces=faces,
                camera_to_world=geometry_result.camera_poses[reference_frame],
                intrinsic=geometry_result.intrinsics[reference_frame],
                target_mask=mask,
            )
            x, y = np.rint(uv).astype(int).T
            physical_type = "soft_body"
        canonical[spec.name] = CanonicalDeformable(
            spec=spec,
            reference_frame=reference_frame,
            mask=mask,
            uv=uv,
            xyz=xyz,
            rgb=geometry_result.rgb[reference_frame, y, x],
            geometry_confidence=geometry_result.geometry_confidence[
                reference_frame, y, x
            ],
            physical_type=physical_type,
        )
    return canonical


def build_observed_associations(
    geometry_result: SceneGeometryResult,
    canonical: Mapping[str, CanonicalDeformable],
    trajectories: Mapping[str, CanonicalTrajectory],
    mask_policies: Mapping[str, TrackingMaskFramePolicy],
    tracks: Mapping[str, Mapping[int, np.ndarray]],
    *,
    minimum_tracking_confidence: float,
    objects: SceneObjects,
) -> tuple[dict[str, list[ObservedPointAssociations]], list[dict[str, Any]]]:
    all_associations: dict[str, list[ObservedPointAssociations]] = {}
    metrics: list[dict[str, Any]] = []
    for spec in objects.deformables:
        deformable = canonical[spec.name]
        trajectory = trajectories[spec.name]
        frame_count = len(geometry_result.rgb)
        height, width = geometry_result.rgb.shape[1:3]
        frame_associations: list[ObservedPointAssociations] = []
        for frame_index in range(frame_count):
            source_mask = tracks[spec.track_id].get(frame_index)
            observed_mask = (
                np.zeros((height, width), dtype=bool)
                if source_mask is None
                else resize_tracking_mask(
                    source_mask, geometry_result.transform.output_size_wh
                )
            )
            if frame_index == deformable.reference_frame:
                observed_mask = deformable.mask
            if mask_policies[spec.name].border_suppressed[frame_index]:
                observed_mask = np.zeros_like(observed_mask)
            association = sample_canonical_tracks_on_observed_mask(
                observed_mask=observed_mask,
                observed_world_points=geometry_result.world_points[frame_index],
                observed_camera_points=geometry_result.camera_points[frame_index],
                observed_geometry_confidence=geometry_result.geometry_confidence[
                    frame_index
                ],
                canonical_to_frame_uv=trajectory.uv[frame_index],
                canonical_to_frame_xyz=trajectory.model_xyz[frame_index],
                canonical_tracking_confidence=trajectory.tracking_confidence[
                    frame_index
                ],
                minimum_tracking_confidence=minimum_tracking_confidence,
                seed=frame_index,
            )
            frame_associations.append(association)
            associated = association.associated
            unique_ids = np.unique(association.canonical_ids[associated])
            row: dict[str, Any] = {
                "deformable": spec.name,
                "reference_frame": deformable.reference_frame,
                "frame_index": frame_index,
                "mask_pixels": int(observed_mask.sum()),
                "reconstructed_points": int(len(association.current_uv)),
                "associated_points": int(associated.sum()),
                "unique_canonical_ids": int(len(unique_ids)),
                "mask_geometry_coverage": (
                    float(len(association.current_uv) / observed_mask.sum())
                    if observed_mask.any()
                    else None
                ),
                "mask_tracking_coverage": (
                    float(associated.sum() / observed_mask.sum())
                    if observed_mask.any()
                    else None
                ),
                "observed_association_rate": (
                    float(associated.mean()) if len(associated) else None
                ),
                "canonical_observed_rate": (
                    float(len(unique_ids) / len(deformable.uv))
                    if len(deformable.uv)
                    else None
                ),
                "propagated_point_survival": float(
                    np.mean(np.isfinite(trajectory.uv[frame_index]).all(axis=1))
                ),
            }
            for threshold in GEOMETRY_THRESHOLDS:
                row[f"geometry_completeness_conf_ge_{threshold:g}"] = (
                    geometry_completeness(
                        observed_mask,
                        geometry_result.camera_points[frame_index],
                        geometry_result.geometry_confidence[frame_index],
                        threshold=threshold,
                    )
                )
            metrics.append(row)
        all_associations[spec.name] = frame_associations
    return (all_associations, metrics)


def summarize_mask_coverage(
    metrics: list[dict[str, Any]],
    *,
    minimum_tracking_confidence: float,
    objects: SceneObjects,
) -> dict[str, Any]:
    deformable_summaries: dict[str, Any] = {}
    for spec in objects.deformables:
        rows = [
            row
            for row in metrics
            if row["deformable"] == spec.name and int(row["mask_pixels"]) > 0
        ]
        if not rows:
            raise ValueError(f"no positive-area mask metrics for {spec.name}")
        mask_pixels = sum((int(row["mask_pixels"]) for row in rows))
        reconstructed = sum((int(row["reconstructed_points"]) for row in rows))
        associated = sum((int(row["associated_points"]) for row in rows))
        per_frame = np.asarray(
            [float(row["mask_tracking_coverage"]) for row in rows], dtype=np.float64
        )
        deformable_summaries[spec.name] = {
            "active_frame_count": len(rows),
            "mask_pixels": mask_pixels,
            "reconstructed_points": reconstructed,
            "confident_associated_points": associated,
            "pixel_weighted_mask_geometry_coverage": reconstructed / mask_pixels,
            "pixel_weighted_mask_tracking_coverage": associated / mask_pixels,
            "association_rate_given_geometry": (
                associated / reconstructed if reconstructed else None
            ),
            "per_frame_mask_tracking_coverage": {
                "mean": float(per_frame.mean()),
                "median": float(np.median(per_frame)),
                "p10": float(np.quantile(per_frame, 0.1)),
                "minimum": float(per_frame.min()),
                "maximum": float(per_frame.max()),
            },
        }
    return {
        "minimum_tracking_confidence": minimum_tracking_confidence,
        "coverage_definition": "valid current-frame mask pixels selected by confident forward canonical trajectories divided by all current-frame mask pixels",
        "deformables": deformable_summaries,
    }


def summarize_observed_errors(values: np.ndarray) -> dict[str, Any]:
    """Missing visible observations have no error estimate, rather than zero error."""
    return {
        "evaluated_frame_count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
    }


def save_tracking_outputs(
    output_dir: Path,
    scene_geometry: SceneGeometryResult,
    canonical: Mapping[str, CanonicalDeformable],
    trajectories: Mapping[str, CanonicalTrajectory],
    mask_policies: Mapping[str, TrackingMaskFramePolicy],
    associations: Mapping[str, list[ObservedPointAssociations]],
    metrics: list[dict[str, Any]],
    cloth_meshes: Mapping[str, ClothReferenceMesh],
    cloth_physics: Mapping[str, ConstrainedClothTrajectory],
    cloth_reference_metrics: Mapping[str, Mapping[str, Any]],
    *,
    selections: Mapping[str, ReferenceFrameSelection],
    fps: float,
    minimum_tracking_confidence: float,
    objects: SceneObjects,
) -> None:
    canonical_dir = output_dir / "canonical"
    trajectory_dir = output_dir / "trajectories"
    association_dir = output_dir / "associations"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    association_dir.mkdir(parents=True, exist_ok=True)
    for name, deformable in canonical.items():
        np.savez_compressed(
            canonical_dir / f"{name}.npz",
            reference_frame=np.asarray(deformable.reference_frame, dtype=np.int32),
            mask=deformable.mask,
            uv=deformable.uv,
            xyz=deformable.xyz,
            rgb=deformable.rgb,
            geometry_confidence=deformable.geometry_confidence,
            physical_type=np.asarray(deformable.physical_type),
        )
        trajectory = trajectories[name]
        trajectory_arrays = {
            "reference_frame": np.asarray(deformable.reference_frame, dtype=np.int32),
            "uv": trajectory.uv,
            "model_xyz": trajectory.model_xyz,
            "tracking_confidence": trajectory.tracking_confidence,
            "border_touching": mask_policies[name].border_touching,
            "nearest_nonborder_frames": mask_policies[name].nearest_nonborder_frames,
            "border_area_ratios": mask_policies[name].border_area_ratios,
            "border_suppressed": mask_policies[name].border_suppressed,
        }
        if name in cloth_physics:
            trajectory_arrays["trackable_identity_hypothesis"] = cloth_physics[
                name
            ].trackable
        np.savez_compressed(trajectory_dir / f"{name}.npz", **trajectory_arrays)
        items = associations[name]
        offsets = np.zeros(len(items) + 1, dtype=np.int64)
        for index, item in enumerate(items):
            offsets[index + 1] = offsets[index] + len(item.current_uv)
        np.savez_compressed(
            association_dir / f"{name}.npz",
            reference_frame=np.asarray(deformable.reference_frame, dtype=np.int32),
            frame_offsets=offsets,
            current_uv=np.concatenate([item.current_uv for item in items]),
            current_xyz=np.concatenate([item.current_xyz for item in items]),
            canonical_id=np.concatenate([item.canonical_ids for item in items]),
            geometry_confidence=np.concatenate(
                [item.geometry_confidence for item in items]
            ),
            tracking_confidence=np.concatenate(
                [item.tracking_confidence for item in items]
            ),
            image_distance=np.concatenate([item.image_distance for item in items]),
        )
    cloth_mesh_dir = output_dir / "cloth_meshes"
    cloth_mesh_dir.mkdir(parents=True, exist_ok=True)
    cloth_metric_rows: list[dict[str, Any]] = []
    cloth_summaries: dict[str, Any] = {}
    warnings = []
    for name, mesh in cloth_meshes.items():
        np.savez_compressed(
            cloth_mesh_dir / f"{name}.npz",
            image_shape=np.asarray(mesh.image_shape, dtype=np.int32),
            vertex_uv=mesh.vertex_uv,
            vertices_world=mesh.vertices_world,
            faces=mesh.faces,
            edges=mesh.edges,
            rest_edge_lengths=mesh.rest_edge_lengths,
        )
        physics = cloth_physics[name]
        rows = [{"cloth": name, **row} for row in physics.frame_metrics]
        cloth_metric_rows.extend(rows)
        durations = physics.active.sum(axis=0)
        evaluated_rows = [
            row for row in physics.frame_metrics if row["observed_mask_pixels"] > 0
        ]
        mask_ious = np.asarray(
            [float(row["mesh_mask_iou"]) for row in evaluated_rows], dtype=np.float64
        )
        active_fractions = np.asarray(
            [float(row["active_vertex_fraction"]) for row in evaluated_rows],
            dtype=np.float64,
        )
        strain_p95 = np.asarray(
            [
                float(row["edge_strain_p95"])
                for row in evaluated_rows
                if row["edge_strain_p95"] is not None
            ],
            dtype=np.float64,
        )
        depth_p90 = np.asarray(
            [
                float(row["visible_depth_error_p90"])
                for row in evaluated_rows
                if row["visible_depth_error_p90"] is not None
            ],
            dtype=np.float64,
        )
        active = physics.active.astype(np.int8)
        starts = (
            np.diff(np.pad(active, ((1, 1), (0, 0)), constant_values=0), axis=0) == 1
        )
        track_run_count = starts.sum(axis=0)
        reference_frame = canonical[name].reference_frame
        if mask_policies[name].border_suppressed[reference_frame]:
            warning = dict(
                code="cloth_reference_frame_suppressed",
                object_id=name,
                reference_frame=reference_frame,
                reason="Selected reference is suppressed by the existing tracking border policy.",
            )
            warnings.append(warning)
            logging.getLogger(__name__).warning("%s", warning)
        terminal_rebirths = np.count_nonzero(
            np.diff(physics.trackable[reference_frame:].astype(np.int8), axis=0) > 0
        ) + np.count_nonzero(
            np.diff(
                physics.trackable[: reference_frame + 1][::-1].astype(np.int8), axis=0
            )
            > 0
        )
        cloth_summaries[name] = {
            "reference_surface": dict(cloth_reference_metrics[name]),
            "mesh": {
                "vertices": len(mesh.vertex_uv),
                "faces": len(mesh.faces),
                "edges": len(mesh.edges),
                "closed_surface": False,
            },
            "identity_tracks": {
                "hidden_positions_output": False,
                "maximum_reassociation_gap_frames": 3,
                "multiple_visible_run_identity_count": int((track_run_count > 1).sum()),
                "terminal_identity_rebirth_count": int(terminal_rebirths),
                "duration_frames": {
                    "median": float(np.median(durations)),
                    "p10": float(np.quantile(durations, 0.1)),
                    "p90": float(np.quantile(durations, 0.9)),
                    "maximum": int(durations.max()),
                },
            },
            "mesh_mask_iou": {
                "evaluated_frame_count": len(evaluated_rows),
                "mean": float(mask_ious.mean()),
                "median": float(np.median(mask_ious)),
                "p10": float(np.quantile(mask_ious, 0.1)),
                "minimum": float(mask_ious.min()),
            },
            "active_vertex_fraction": {
                "mean": float(active_fractions.mean()),
                "median": float(np.median(active_fractions)),
                "p10": float(np.quantile(active_fractions, 0.1)),
                "minimum": float(active_fractions.min()),
            },
            "edge_strain_p95_per_frame": summarize_observed_errors(strain_p95),
            "visible_depth_error_p90_per_frame": summarize_observed_errors(depth_p90),
        }
    _write_json(output_dir / "warnings.json", warnings)
    _write_json(output_dir / "cloth_physics_metrics.json", cloth_metric_rows)
    _write_json(output_dir / "cloth_physics_summary.json", cloth_summaries)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        if metrics:
            writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "coverage_summary.json",
        summarize_mask_coverage(
            metrics,
            minimum_tracking_confidence=minimum_tracking_confidence,
            objects=objects,
        ),
    )
    _write_tracking_metadata(
        output_dir=output_dir,
        scene_geometry=scene_geometry,
        selections=selections,
        fps=fps,
        minimum_tracking_confidence=minimum_tracking_confidence,
    )


def _write_tracking_metadata(
    *,
    output_dir: Path,
    scene_geometry: SceneGeometryResult,
    selections: Mapping[str, ReferenceFrameSelection],
    fps: float,
    minimum_tracking_confidence: float,
) -> None:
    _write_json(
        output_dir / "run_metadata.json",
        {
            "reference_frames": {
                name: selection.reference_frame
                for name, selection in selections.items()
            },
            "base_reference_frame": 0,
            "fps": fps,
            "original_size_wh": scene_geometry.transform.original_size_wh,
            "model_size_wh": scene_geometry.transform.output_size_wh,
            "geometry_inference_threshold": 0.0,
            "reported_geometry_thresholds": GEOMETRY_THRESHOLDS,
            "trajectory_mode": "adjacent_recursive_by_physical_type",
            "association_mode": "canonical_forward_sampling",
            "adjacent_deformable_policy": {
                "propagation": "each deformable object propagates bidirectionally from its own reference through consecutive frame pairs",
                "confidence_accumulation": "current adjacent-step confidence; failed visible identities are not reborn",
                "lost_identity_policy": "no track birth after propagation failure",
                "xyz_policy": "cloth uses observed point maps with its surface prior; soft bodies apply observed displacement increments to the aligned SAM3D surface, followed by mask correction",
            },
            "cloth_physics_policy": {
                "reference_surface": "open image-grid triangle mesh with depth-discontinuity edge rejection",
                "motion_proposal": "Track4World adjacent-frame optical flow",
                "visibility_confidence": "current adjacent step rather than minimum confidence over the complete trajectory prefix",
                "constraints": "visible observed surface attachment and approximate rest edge-length preservation; free bending",
                "solver_parameters": {
                    "iterations": 24,
                    "stretch_relaxation": 2.5,
                    "observation_pull_per_iteration": 0.0,
                    "maximum_relative_edge_strain": 2.0,
                    "maximum_mask_snap_distance_pixels": 5.0,
                    "maximum_reassociation_gap_frames": 3,
                },
                "recursion": "feed constrained projected UV into the next frame",
                "visibility": "withhold output outside the cloth mask, without valid geometry, or below current-step confidence; expire identities using the non-paused invisible-gap budget; snap boundary proposals to a nearby observed cloth pixel",
                "hidden_surface_completion": False,
                "terminated_identity_rebirth": False,
                "short_gap_reassociation": "retain an internal UV hypothesis for at most three non-paused unobserved steps; write no hidden position",
                "border_pause": "nonempty border-suppressed frames emit no positions; propagate finite internal UV while preserving, not resetting, the invisible-gap count",
                "complete_occlusion": "empty masks always advance the invisible-gap count; policy pauses cannot revive terminated identities",
                "soft_body_uses_cloth_prior": False,
            },
            "soft_body_policy": {
                "reference": "SAM3D with rigid-taxonomy reference shape fitting, FoundationPose and scale/depth alignment",
                "canonical_points": "reference-visible mesh ray intersections with fixed IDs",
                "motion": "independent adjacent world-point-map displacement increments",
                "correction": "soft mask IoU and confidence-weighted tracking attachment at fixed predicted depth",
                "mask_initialization": "project nearby predictions into target mask before IoU optimization; retain best hard IoU including proposal",
                "shape_constraints": [],
                "hidden_surface_completion": False,
                "lost_identity_rebirth": False,
                "splat_radius_pixels": 1,
            },
            "canonical_forward_sampling_policy": {
                "pixel_sampling": "round predicted UV to nearest current pixel",
                "mask_policy": "accept only valid reconstructed deformable-mask pixels",
                "collision_policy": "one canonical ID per current pixel; prefer confidence, then subpixel residual, then canonical ID",
                "new_track_policy": "do not create IDs outside the reference mask",
            },
            "minimum_tracking_confidence": minimum_tracking_confidence,
            "unmatched_canonical_id": -1,
            "border_frame_policy": "omit every border-touching object frame when its visible lifecycle contains any non-border frame; retain all frames for a border-only lifecycle",
            "shared_geometry_policy": {
                "mode": "single_chronological_track4world_pass",
                "geometry_reference_frame": 0,
                "description": "All objects share one chronological depth, camera-pose, and world-point reconstruction. Rigid objects share one complete forward adjacent-pair inference; deformable objects reuse it forward and use one reverse-video adjacent inference backward.",
            },
            "static_background_policy": {
                "sample_count": 16,
                "sampling": "uniform across the full video",
                "foreground_exclusion": "union of every SAM3 tracked-object mask independent of physical type",
                "fusion": "observed points only, adaptive voxel averaging",
                "spatial_completion": False,
            },
            "camera_policy": {
                "mode": "fixed_camera",
                "calibration_sample_count": 16,
                "geometry_policy": "express every Track4World camera point map in one fixed camera-to-world coordinate system before object tracking",
            },
        },
    )


def run_tracking(config: TrackingConfig, *, shared: SceneData | None = None,
                 tracking_input=None) -> Path:
    args = config
    if args.rigid_workers < 1:
        raise ValueError("--rigid-workers must be positive")
    assets = load_reconstruction_assets(args.reconstruction_dir)
    if tracking_input is None:
        tracking_input = load_pointcloud_tracking_input(
            assets.sam3_result_dir / "pointcloud_tracking_input.json"
        )
    objects = SceneObjects.from_input(tracking_input)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite retained task result: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    with timed_stage(timings, "load_video_and_masks"):
        if shared is None:
            frames, fps = read_video_rgb(tracking_input.video_path)
            tracks, _ = load_tracks_and_targets(tracking_input, objects=objects)
        else:
            frames, fps, tracks = shared.frames, shared.fps, shared.tracks
        if len(frames) != tracking_input.frame_count:
            raise ValueError(
                f"source video frame count changed after SAM3: {len(frames)} != {tracking_input.frame_count}"
            )
    with timed_stage(timings, "mask_rigid_motion"):
        prepared_motion = assets.manifest_path.parent / 'rigid_motion'
        if (prepared_motion / 'decision.json').is_file():
            shutil.copytree(prepared_motion, output_dir / 'rigid_motion')
            motion_decision = json.loads((prepared_motion / 'decision.json').read_text())
        else:
            motion_decision = classify_rigid_motion(
                output_dir=output_dir / "rigid_motion",
                rigid_target_ids=[item.target_id for item in tracking_input.rigid_objects],
                tracks={
                    item.target_id: tracks[item.track_id] for item in tracking_input.objects
                },
                frame_count=len(frames),
                mask_shape=(tracking_input.mask_height, tracking_input.mask_width),
            )
    needs_adjacent = bool(tracking_input.cloth_objects) or any(
        (
            item["tracking_branch"] == "dynamic"
            for item in motion_decision["objects"].values()
        )
    )
    with timed_stage(timings, "load_reconstruction_assets"):
        selections = (
            shared.selections
            if shared is not None
            else load_saved_reference_frames(
                assets.reference_selections_path, objects=objects
            )
        )
        scene_geometry = (
            shared.geometry if shared is not None else assets.load_scene_geometry()
        )
    _write_json(
        output_dir / "input_sources.json",
        {
            "reconstruction_manifest": str(assets.manifest_path),
            "pointcloud_tracking_input": str(tracking_input.manifest_path),
            "video": str(tracking_input.video_path),
            "mask_archive": str(tracking_input.mask_archive_path),
            "mask_index": str(tracking_input.mask_index_path),
            "inventory": str(tracking_input.inventory_path),
            "rigid_motion_decision": str(output_dir / "rigid_motion/decision.json"),
            "tracking_manifest": str(tracking_input.tracking_manifest_path),
            "adjacent_tracking_cache": (
                str(args.adjacent_tracking_cache.resolve())
                if args.adjacent_tracking_cache is not None
                else None
            ),
            "objects": [
                {
                    "target_id": item.target_id,
                    "track_id": item.track_id,
                    "physical_type": item.physical_type,
                    "physical_type_confidence": item.physical_type_confidence,
                    "rigid_shape": item.rigid_shape,
                    "rigid_shape_confidence": item.rigid_shape_confidence,
                }
                for item in tracking_input.objects
            ],
        },
    )
    if shared is not None:
        model_masks = shared.masks
    else:
        model_masks = {}
        for spec in objects.all:
            with np.load(assets.tracking_masks_dir / f"{spec.name}.npz") as archive:
                model_masks[spec.name] = archive["masks"]
    with timed_stage(timings, "preprocess_video"):
        resized = scene_geometry.rgb
    adjacent = None
    adjacent_output = output_dir / "cache/adjacent_tracking"
    if needs_adjacent and args.adjacent_tracking_cache is not None:
        with timed_stage(timings, "track4world_adjacent_cache_load"):
            adjacent = load_adjacent_track4world_cache(
                args.adjacent_tracking_cache,
                source_video_path=tracking_input.video_path,
                frame_count=len(resized),
                image_size_wh=(resized.shape[2], resized.shape[1]),
            )
        adjacent_output.parent.mkdir(parents=True, exist_ok=True)
        adjacent_output.symlink_to(
            args.adjacent_tracking_cache.resolve(), target_is_directory=True
        )
    model = None
    if (
        tracking_input.cloth_objects
        or (tracking_input.soft_body_objects and args.soft_body_motion_cache is None)
        or (needs_adjacent and adjacent is None)
    ):
        with timed_stage(timings, "track4world_model_load"):
            model = load_track4world_da3_model(
                checkpoint=args.checkpoint,
                da3_snapshot=args.da3_snapshot,
                device=args.device,
            )
        timings.update(track4world_load_timings(model.physmind_load_diagnostics))
    if needs_adjacent and adjacent is None:
        assert model is not None
        with timed_stage(timings, "track4world_whole_video_adjacent_inference"):
            adjacent = run_adjacent_track4world_tracking(
                model,
                resized,
                device=args.device,
                inference_iterations=args.inference_iterations,
            )
        with timed_stage(timings, "track4world_adjacent_cache_save"):
            save_adjacent_track4world_cache(
                adjacent_output,
                adjacent,
                source_video_path=tracking_input.video_path,
                inference_iterations=args.inference_iterations,
            )
    else:
        timings["track4world_whole_video_adjacent_inference"] = 0.0
        timings["track4world_adjacent_cache_save"] = 0.0
    reverse_adjacent = None
    if tracking_input.cloth_objects:
        assert model is not None
        with timed_stage(timings, "track4world_whole_video_reverse_adjacent_inference"):
            reverse_adjacent = run_adjacent_track4world_tracking(
                model,
                np.ascontiguousarray(resized[::-1]),
                device=args.device,
                inference_iterations=args.inference_iterations,
            )
    with timed_stage(timings, "adjacent_deformable_tracking_and_sampling"):
        canonical = create_canonical_deformables(
            geometry_result=scene_geometry,
            selections=selections,
            tracks=tracks,
            soft_body_models={
                item.target_id: assets.soft_body_models_dir / item.target_id
                for item in tracking_input.soft_body_objects
            },
            objects=objects,
        )
        trajectories: dict[str, CanonicalTrajectory] = {}
        deformable_mask_policies: dict[str, TrackingMaskFramePolicy] = {}
        cloth_meshes: dict[str, ClothReferenceMesh] = {}
        cloth_physics: dict[str, ConstrainedClothTrajectory] = {}
        cloth_reference_metrics: dict[str, Mapping[str, Any]] = {}
        cloth_names = {item.target_id for item in tracking_input.cloth_objects}
        for name, deformable in canonical.items():
            width, height = scene_geometry.transform.output_size_wh
            object_masks = model_masks[name]
            mask_policy = classify_tracking_mask_frames(object_masks)
            effective_masks = object_masks.copy()
            effective_masks[mask_policy.border_suppressed] = False
            if name in cloth_names:
                assert reverse_adjacent is not None
                mesh = build_cloth_reference_mesh(
                    mask=deformable.mask,
                    vertex_uv=deformable.uv,
                    vertices_world=deformable.xyz,
                )
                constrained = propagate_constrained_cloth_tracks(
                    mesh,
                    adjacent,
                    reverse_adjacent,
                    reference_frame=deformable.reference_frame,
                    world_points=scene_geometry.world_points,
                    camera_points=scene_geometry.camera_points,
                    intrinsics=scene_geometry.intrinsics,
                    camera_poses=scene_geometry.camera_poses,
                    masks=object_masks,
                    policy_paused_frames=mask_policy.border_suppressed,
                    minimum_tracking_confidence=args.minimum_tracking_confidence,
                )
                cloth_meshes[name] = mesh
                cloth_physics[name] = constrained
                reference_observation = evaluate_cloth_mesh_observation(
                    mesh,
                    vertices_world=mesh.vertices_world,
                    camera_to_world=scene_geometry.camera_poses[
                        deformable.reference_frame
                    ],
                    intrinsic=scene_geometry.intrinsics[deformable.reference_frame],
                    observed_mask=deformable.mask,
                    observed_depth=scene_geometry.camera_points[
                        deformable.reference_frame, ..., 2
                    ],
                )
                cloth_reference_metrics[name] = reference_observation.metrics()
                trajectories[name] = CanonicalTrajectory(
                    uv=constrained.uv,
                    model_xyz=constrained.positions_world,
                    tracking_confidence=constrained.tracking_confidence,
                )
            else:
                effective_masks[mask_policy.occlusion_terminated] = False
                with timed_stage(timings, f"soft_body_local_motion_{name}"):
                    local_forward, local_backward = local_soft_body_motion(
                        model,
                        frames,
                        effective_masks,
                        video_path=tracking_input.video_path,
                        output_dir=output_dir / "cache/soft_body_motion" / name,
                        device=args.device,
                        inference_iterations=args.inference_iterations,
                        cache_dir=(
                            args.soft_body_motion_cache / name
                            if args.soft_body_motion_cache is not None
                            else None
                        ),
                    )
                model_dir = assets.soft_body_models_dir / name
                reference_vertices, _ = load_complete_canonical_mesh(
                    model_dir / "canonical_metric.glb"
                )
                with np.load(model_dir / "canonical_rigid_model.npz") as archive:
                    reference_pose = archive["reference_pose"]
                reference_extent = float(
                    np.ptp(
                        reference_vertices @ reference_pose[:3, :3].T
                        + reference_pose[:3, 3],
                        axis=0,
                    ).max()
                )
                with timed_stage(timings, f"soft_body_surface_tracking_{name}"):
                    soft_result = track_soft_body_surface(
                        initial_uv=deformable.uv,
                        initial_xyz=deformable.xyz,
                        reference_frame=deformable.reference_frame,
                        forward=local_forward,
                        backward=local_backward,
                        world_points=scene_geometry.world_points,
                        camera_poses=scene_geometry.camera_poses,
                        intrinsics=scene_geometry.intrinsics,
                        masks=effective_masks,
                        observed_depth=scene_geometry.camera_points[..., 2],
                        geometry_valid=scene_geometry.geometry_valid,
                        reference_extent=reference_extent,
                        all_object_mask_union=np.load(
                            assets.geometry_dir / "tracked_region_mask_union.npy",
                            mmap_mode="r",
                        ),
                        minimum_tracking_confidence=args.minimum_tracking_confidence,
                        device=args.device,
                    )
                trajectories[name] = soft_result.trajectory
                soft_dir = output_dir / "soft_body_surfaces"
                soft_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    soft_dir / f"{name}.npz",
                    proposal_uv=soft_result.proposal_uv,
                    surface_depth=soft_result.surface_depth,
                    reference_extent=np.asarray(soft_result.reference_extent),
                )
                _write_json(soft_dir / f"{name}_metrics.json", soft_result.metrics)
                del local_forward, local_backward, soft_result
            deformable_mask_policies[name] = apply_deformable_mask_policy(
                trajectories[name], object_masks,
                reference_frame=deformable.reference_frame,
            )
            gc.collect()
            if args.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
        associations, metrics = build_observed_associations(
            scene_geometry,
            canonical,
            trajectories,
            deformable_mask_policies,
            tracks,
            minimum_tracking_confidence=args.minimum_tracking_confidence,
            objects=objects,
        )
    with timed_stage(timings, "save_machine_outputs"):
        save_tracking_outputs(
            output_dir,
            scene_geometry,
            canonical,
            trajectories,
            deformable_mask_policies,
            associations,
            metrics,
            cloth_meshes,
            cloth_physics,
            cloth_reference_metrics,
            selections=selections,
            fps=fps,
            minimum_tracking_confidence=args.minimum_tracking_confidence,
            objects=objects,
        )
        metadata_path = output_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["reconstruction_assets"] = {
            "manifest": str(assets.manifest_path),
            "shared_by_reference": True,
            "sam3d_invoked_by_tracking": False,
        }
        metadata["track4world_adjacent_flow"] = {
            "required": needs_adjacent,
            "cache_directory": (
                str(adjacent_output.resolve()) if needs_adjacent else None
            ),
            "cache_reused": needs_adjacent and args.adjacent_tracking_cache is not None,
            "arrays": ["source_to_target_uv", "tracking_confidence"],
            "forward_shared_by_dynamic_rigids_and_cloth": needs_adjacent,
            "reverse_video_inference_for_cloth": bool(tracking_input.cloth_objects),
            "reverse_video_fields_retained": False,
            "source_to_target_xyz_retained": False,
        }
        metadata["soft_body_tracking"] = {
            "motion": "fixed-scale native RGB crops, adjacent forward/reverse; endpoints restored to scene pixels",
            "motion_cache": "cache/soft_body_motion/<object>/{forward,backward}",
            "surface": "open observed depth chart, curvature weight 1, mask-aware sampling",
            "depth_attachment": "after UV recursion; never feeds back into image trajectories",
            "hidden_surface_completion": False,
            "identity_rebirth": False,
            "depth_ground_truth": False,
            "debug_coordinates": "global full frame and common scene frame",
        }
        metadata["rigid_motion"] = motion_decision
        _write_json(metadata_path, metadata)
    if objects.rigids:
        rigid_model_dir = assets.rigid_models_dir
        gravity_summary = json.loads(
            (assets.support_plane_gravity_dir / "gravity_estimate.json").read_text(
                encoding="utf-8"
            )
        )
        support_constraint = support_constraint_from_gravity_summary(gravity_summary)
        with timed_stage(timings, "rigid_pointcloud_pose_tracking"):
            rigid_trajectories = track_and_save_rigid_objects(
                output_dir=output_dir,
                model_dir=rigid_model_dir,
                adjacent=adjacent,
                geometry_result=scene_geometry,
                selections=selections,
                tracks=tracks,
                rigid_shapes={
                    item.target_id: str(item.rigid_shape)
                    for item in tracking_input.rigid_objects
                },
                motion_decision=motion_decision,
                minimum_tracking_confidence=args.minimum_tracking_confidence,
                support_constraint=support_constraint,
                rigid_workers=args.rigid_workers,
                objects=objects,
                model_masks=model_masks,
            )
    del model, resized, reverse_adjacent, frames
    gc.collect()
    if args.device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    timings["compute_total_seconds"] = time.perf_counter() - total_started
    timings["current_process_total"] = timings["compute_total_seconds"]
    timings["end_to_end_total"] = timings["current_process_total"]
    _write_json(output_dir / "timings.json", timings)
    (output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return output_dir
