#!/usr/bin/env python3
"""Static and dynamic rigid tracking with per-object results."""

from __future__ import annotations

import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np


from atw.perception.cloth_reference_selector import ReferenceFrameSelection
from atw.perception.track4world_tool import AdjacentTrack4WorldResult, SceneGeometryResult
from atw.perception.rigid_tracking import PointCloudSupportConstraint, RigidPoseTrajectory, static_rigid_pose_sequence, track_rigid_pose_sequence


from atw.perception.pipeline.recording import write_json as _write_json

from atw.perception.pipeline.scene import SceneObjects, TrackedObjectSpec


def track_and_save_rigid_objects(
    *,
    output_dir: Path,
    model_dir: Path,
    adjacent: AdjacentTrack4WorldResult | None,
    geometry_result: SceneGeometryResult,
    selections: Mapping[str, ReferenceFrameSelection],
    tracks: Mapping[str, Mapping[int, np.ndarray]],
    rigid_shapes: Mapping[str, str],
    motion_decision: Mapping[str, Any],
    minimum_tracking_confidence: float,
    support_constraint: PointCloudSupportConstraint | None,
    rigid_workers: int,
    objects: SceneObjects,
    model_masks: Mapping[str, np.ndarray],
) -> dict[str, RigidPoseTrajectory]:
    trajectory_dir = output_dir / "rigid_trajectories"
    trajectory_dir.mkdir(parents=True)
    frame_count, height, width = geometry_result.world_points.shape[:3]
    trajectories: dict[str, RigidPoseTrajectory] = {}
    summaries: dict[str, Any] = {}
    mask_by_name = model_masks
    all_object_mask_union = np.logical_or.reduce(tuple(mask_by_name.values()))
    motion_models = {
        spec.name: (
            "static"
            if motion_decision["objects"][spec.name]["tracking_branch"] == "static"
            else {
                "sphere": "translation_only",
                "cylinder": "axisymmetric",
                "box": "se3",
                "irregular": "se3",
            }[rigid_shapes[spec.name]]
        )
        for spec in objects.rigids
    }

    def track_one(
        spec: TrackedObjectSpec,
    ) -> tuple[TrackedObjectSpec, np.ndarray, np.ndarray, RigidPoseTrajectory]:
        model_path = model_dir / spec.name / "canonical_rigid_model.npz"
        with np.load(model_path) as archive:
            canonical_points = np.asarray(
                archive["canonical_surface_points"], dtype=np.float32
            )
            canonical_colors = np.asarray(archive["canonical_surface_colors"])
            reference_pose = np.asarray(archive["reference_pose"], dtype=np.float32)
        masks = mask_by_name[spec.name]
        rigid_shape = rigid_shapes[spec.name]
        motion_model = motion_models[spec.name]
        if motion_model == "static":
            trajectory = static_rigid_pose_sequence(
                canonical_points=canonical_points,
                reference_pose=reference_pose,
                reference_frame=selections[spec.name].reference_frame,
                masks=masks,
                all_object_mask_union=all_object_mask_union,
                camera_poses=geometry_result.camera_poses,
                intrinsics=geometry_result.intrinsics,
                support_constraint=support_constraint,
            )
            return (spec, canonical_points, canonical_colors, trajectory)
        if adjacent is None:
            raise ValueError("dynamic rigid tracking requires adjacent motion fields")
        canonical_symmetry_axis = None
        if rigid_shape == "cylinder":
            model_metadata = json.loads(
                (model_path.parent / "metadata.json").read_text(encoding="utf-8")
            )
            canonical_symmetry_axis = np.asarray(
                model_metadata["mesh_conditioning"]["fit"]["axis"], dtype=np.float32
            )
        trajectory = track_rigid_pose_sequence(
            canonical_points=canonical_points,
            reference_pose=reference_pose,
            reference_frame=selections[spec.name].reference_frame,
            masks=masks,
            all_object_mask_union=all_object_mask_union,
            world_points=geometry_result.world_points,
            geometry_confidence=geometry_result.geometry_confidence,
            camera_poses=geometry_result.camera_poses,
            intrinsics=geometry_result.intrinsics,
            adjacent=adjacent,
            motion_model=motion_model,
            canonical_symmetry_axis=canonical_symmetry_axis,
            minimum_geometry_confidence=0.0,
            minimum_tracking_confidence=minimum_tracking_confidence,
            support_constraint=support_constraint,
        )
        return (spec, canonical_points, canonical_colors, trajectory)

    worker_count = min(rigid_workers, len(objects.rigids))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        tracked = list(executor.map(track_one, objects.rigids))
    for spec, canonical_points, canonical_colors, trajectory in tracked:
        trajectories[spec.name] = trajectory
        np.savez_compressed(
            trajectory_dir / f"{spec.name}.npz",
            motion_state=np.asarray(motion_decision["objects"][spec.name]["state"]),
            motion_model=np.asarray(motion_models[spec.name]),
            reference_frame=np.asarray(
                selections[spec.name].reference_frame, dtype=np.int32
            ),
            canonical_tracking_points=trajectory.canonical_tracking_points,
            canonical_tracking_indices=trajectory.canonical_tracking_indices,
            canonical_tracking_colors=(
                canonical_colors[trajectory.canonical_tracking_indices]
                if len(canonical_colors) == len(canonical_points)
                else np.empty((0, 4), dtype=np.uint8)
            ),
            tracking_poses=trajectory.tracking_poses,
            poses=trajectory.poses,
            mask_pixel_counts=trajectory.mask_pixel_counts,
            occlusion_termination_frame=np.asarray(
                trajectory.occlusion_termination_frame, dtype=np.int32
            ),
            occlusion_terminated=trajectory.occlusion_terminated,
            border_touching=trajectory.border_touching,
            nearest_nonborder_frames=trajectory.nearest_nonborder_frames,
            border_area_ratios=trajectory.border_area_ratios,
            border_suppressed=trajectory.border_suppressed,
            observed_point_counts=trajectory.observed_point_counts,
            correspondence_counts=trajectory.correspondence_counts,
            inlier_counts=trajectory.inlier_counts,
            rmse=trajectory.rmse,
            observed_coverage=trajectory.observed_coverage,
            motion_initialized=trajectory.motion_initialized,
            motion_correspondence_counts=trajectory.motion_correspondence_counts,
            motion_inlier_counts=trajectory.motion_inlier_counts,
            motion_rmse=trajectory.motion_rmse,
            segment_anchor_frames=trajectory.segment_anchor_frames,
            numerically_converged=trajectory.numerically_converged,
            has_sufficient_correspondences=trajectory.has_sufficient_correspondences,
            self_visible_canonical_counts=trajectory.self_visible_canonical_counts,
            other_mask_occluded_canonical_counts=trajectory.other_mask_occluded_canonical_counts,
            occlusion_eligible_canonical_counts=trajectory.occlusion_eligible_canonical_counts,
            support_constrained=trajectory.support_constrained,
            support_constraint_rejected=trajectory.support_constraint_rejected,
            support_penetration_depth=trajectory.support_penetration_depth,
            support_ray_scale=trajectory.support_ray_scale,
            support_correction_norm=trajectory.support_correction_norm,
            support_clearance_after=trajectory.support_clearance_after,
        )
        active = np.isfinite(trajectory.poses).all(axis=(1, 2))
        numerically_converged_active = active & trajectory.numerically_converged
        sufficient_active = active & trajectory.has_sufficient_correspondences
        summaries[spec.name] = {
            "reference_frame": selections[spec.name].reference_frame,
            "rigid_shape": rigid_shapes[spec.name],
            "motion_model": motion_models[spec.name],
            "motion_decision": motion_decision["objects"][spec.name],
            "temporal_pose_fitting": motion_models[spec.name] != "static",
            "support_correction_evaluations": (
                (
                    1
                    if motion_models[spec.name] == "static"
                    else int(
                        np.isfinite(trajectory.tracking_poses).all(axis=(1, 2)).sum()
                    )
                )
                if support_constraint is not None
                else 0
            ),
            "occlusion_termination_frame": (
                trajectory.occlusion_termination_frame
                if trajectory.occlusion_termination_frame >= 0
                else None
            ),
            "occlusion_terminated_frame_count": int(
                trajectory.occlusion_terminated.sum()
            ),
            "border_touching_frame_count": int(trajectory.border_touching.sum()),
            "border_suppressed_frame_count": int(trajectory.border_suppressed.sum()),
            "border_retained_frame_count": int(
                (trajectory.border_touching & ~trajectory.border_suppressed).sum()
            ),
            "border_retained_by_area_frame_count": int(
                (
                    trajectory.border_touching
                    & ~trajectory.border_suppressed
                    & np.isfinite(trajectory.border_area_ratios)
                ).sum()
            ),
            "border_retained_without_interior_baseline_frame_count": int(
                (
                    trajectory.border_touching
                    & ~trajectory.border_suppressed
                    & ~np.isfinite(trajectory.border_area_ratios)
                ).sum()
            ),
            "active_frame_count": int(active.sum()),
            "numerically_converged_frame_count": int(
                numerically_converged_active.sum()
            ),
            "numerically_converged_active_rate": (
                float(numerically_converged_active.sum() / active.sum())
                if active.any()
                else None
            ),
            "sufficient_correspondence_frame_count": int(sufficient_active.sum()),
            "sufficient_correspondence_active_rate": (
                float(sufficient_active.sum() / active.sum()) if active.any() else None
            ),
            "mean_observed_coverage": (
                float(np.mean(trajectory.observed_coverage[active]))
                if active.any() and motion_models[spec.name] != "static"
                else None
            ),
            "median_observed_coverage": (
                float(np.median(trajectory.observed_coverage[active]))
                if active.any() and motion_models[spec.name] != "static"
                else None
            ),
            "motion_initialized_frame_count": int(trajectory.motion_initialized.sum()),
            "support_constrained_frame_count": int(
                trajectory.support_constrained.sum()
            ),
            "support_constraint_rejected_frame_count": int(
                trajectory.support_constraint_rejected.sum()
            ),
            "maximum_support_penetration_depth": (
                float(np.nanmax(trajectory.support_penetration_depth))
                if np.isfinite(trajectory.support_penetration_depth).any()
                else None
            ),
            "maximum_support_correction_norm": float(
                np.max(trajectory.support_correction_norm)
            ),
            "mean_self_visible_canonical_points": float(
                np.mean(trajectory.self_visible_canonical_counts[active])
            ),
            "mean_other_mask_occluded_canonical_points": float(
                np.mean(trajectory.other_mask_occluded_canonical_counts[active])
            ),
            "mean_occlusion_eligible_canonical_points": float(
                np.mean(trajectory.occlusion_eligible_canonical_counts[active])
            ),
        }
        gc.collect()
    _write_json(
        output_dir / "rigid_tracking_summary.json",
        {
            "method": "fresh fixed-camera mask motion decision; static-compatible objects freeze the aligned reference pose after one nonpenetration correction; dynamic objects use shared Track4World adjacent-pair inference; recursive frame-to-frame RANSAC initialization followed at every frame by occlusion-aware projective robust depth-and-silhouette refinement; sphere rotation is fixed, cylinder axial twist is suppressed, and other rigid shapes retain SE(3)",
            "icp_visibility_policy": "re-render the canonical tracking cloud at every iteration; keep the frontmost point per pixel with a point-cloud Z-buffer; pair it with the same-pixel observed depth and normal inside the SAM mask; complement point-to-plane residuals with the SAM-mask signed-distance silhouette; exclude other recognized object masks",
            "border_frame_policy": "omit every border-touching frame when any non-border mask exists; retain all when the visible lifecycle is border-only",
            "pose_output_policy": "static: fixed corrected reference world pose on eligible mask frames; dynamic: use the pose returned by ICP; when ICP cannot run, use the Track4World initialization",
            "terminated_pose_policy": "write NaN poses from the termination frame",
            "occlusion_policy": "after first visibility, three consecutive empty masks confirm occlusion; permanently omit from the first empty frame",
            "minimum_geometry_confidence": 0.0,
            "minimum_tracking_confidence": minimum_tracking_confidence,
            "rigid_worker_count": worker_count,
            "support_nonpenetration": {
                "enabled": support_constraint is not None,
                "geometry": "complete frozen canonical point cloud",
                "correction_direction": "static: reference-camera centroid ray once; dynamic: current-frame camera centroid ray",
                "rotation_modified": False,
                "airborne_pose_modified": False,
                "propagation": "static: correct reference once and repeat identically; dynamic: postprocess completed poses without feedback into tracking",
            },
            "objects": summaries,
        },
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tracked_objects"] = [spec.name for spec in objects.all]
    metadata["rigid_tracking"] = {
        "object_count": len(objects.rigids),
        "worker_count": worker_count,
        "canonical_geometry": "SAM 3D Objects, frozen after reference frame",
        "pose_model": "VLM static: fixed world pose; dynamic shape-aware: translation-only with reference rotation fixed for spheres; five-DoF axisymmetric motion for cylinders; SE(3) for boxes and irregular objects; shape and metric scale fixed",
        "motion_models": motion_models,
        "motion_decision": "rigid_motion/decision.json",
        "initialization": "static: retained aligned reference only; dynamic: load dense adjacent-flow cache or run one shared whole-video Track4World infer_pair call",
        "refinement": "static: no temporal fitting; dynamic: occlusion-aware projective point-to-plane and mask signed-distance silhouette refinement; each residual family is independently MAD-scaled and robustly weighted; each refined pose becomes the next recursive pose anchor",
        "icp_visibility_policy": "iterative canonical point-cloud Z-buffer rendering followed by same-pixel observed depth/normal association and rendered-boundary alignment to the complete SAM mask, with other recognized object mask exclusion",
        "border_frame_policy": "omit every border-touching frame when any non-border mask exists; retain all when the visible lifecycle is border-only",
        "pose_output_policy": "static: fixed corrected reference world pose; dynamic: use the pose returned by ICP; when ICP cannot run, use the Track4World initialization",
        "terminated_pose_policy": "write NaN poses from the termination frame",
        "occlusion_policy": "after first visibility, three consecutive empty masks confirm occlusion; permanently omit from the first empty frame",
        "foundationpose_used": True,
        "foundationpose_scope": "five-iteration reference-frame 6DoF refinement only; only dynamic temporal poses use Track4World and point-cloud ICP",
        "support_nonpenetration": {
            "enabled": support_constraint is not None,
            "geometry": "complete frozen canonical point cloud",
            "correction_direction": "static: reference-camera ray once; dynamic: current-frame camera ray",
            "tracking_pose_field": "tracking_poses",
            "output_pose_field": "poses",
            "rotation_modified": False,
            "airborne_pose_modified": False,
            "propagation": "static: repeat one corrected pose; dynamic: postprocess only, no tracking feedback",
        },
    }
    _write_json(metadata_path, metadata)
    return trajectories
