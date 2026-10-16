from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atw.perception.rigid_reconstruction import apply_projection_alignment, apply_reference_camera_pose, apply_rigid_pose, condition_sam3d_mesh, reference_camera_pose, reframe_box_canonical_mesh, sam3d_pointmap_from_opencv
from atw.perception.foundationpose_refinement import FoundationPoseReferenceRefiner
from atw.perception.reference_mesh_alignment import align_reference_mesh
from atw.perception.mesh_projection import render_mask_occluded_mesh
from atw.perception.model_loading import TorchLoadCache, prefetch_weight_files
from atw.perception.rigid_mesh_conditioning import condition_rigid_mesh


DEFAULT_CONFIG = PROJECT_ROOT / "data/models/sam-3d-objects/pipeline.yaml"


def _checkpoint_paths(config: Any, config_path: Path) -> tuple[Path, ...]:
    paths = []
    for key, value in config.items():
        if key.endswith("_ckpt_path") and value:
            path = Path(str(value))
            if not path.is_absolute():
                path = config_path.parent / path
            paths.append(path.resolve())
    return tuple(dict.fromkeys(paths))


def _load_pipeline(config_path: Path):
    os.environ["LIDRA_SKIP_INIT"] = "true"
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = str(config_path.parent)
    config.depth_model = None
    checkpoint_paths = _checkpoint_paths(config, config_path)
    reusable_checkpoint_paths = {
        (config_path.parent / str(config[key])).resolve()
        for key in ("ss_generator_ckpt_path", "slat_generator_ckpt_path")
    }
    prefetch = prefetch_weight_files(checkpoint_paths)

    instantiate_started = perf_counter()
    with TorchLoadCache(
        tracked_paths=checkpoint_paths,
        reusable_paths=reusable_checkpoint_paths,
    ) as checkpoint_cache:
        pipeline = instantiate(config)
    instantiate_seconds = perf_counter() - instantiate_started
    diagnostics = {
        "weight_prefetch": prefetch.to_dict(),
        "pipeline_instantiate_seconds": instantiate_seconds,
        **checkpoint_cache.diagnostics(),
    }
    return pipeline, diagnostics


def _tensor_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _merged_mesh(glb: Any):
    import trimesh

    if isinstance(glb, trimesh.Scene):
        geometries = tuple(glb.geometry.values())
        if not geometries:
            raise ValueError("SAM 3D returned an empty GLB scene")
        return trimesh.util.concatenate(geometries)
    if isinstance(glb, trimesh.Trimesh):
        return glb.copy()
    raise TypeError(f"unsupported SAM 3D GLB type: {type(glb).__name__}")


def _projection_iou(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    target_mask: np.ndarray,
    all_object_mask_union: np.ndarray,
) -> float:
    rendered = render_mask_occluded_mesh(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsic=intrinsics,
        image_shape=target_mask.shape,
        occluder_mask=all_object_mask_union,
        self_mask=target_mask,
    )["visible_mask"]
    intersection = int(np.count_nonzero(rendered & target_mask))
    union = int(np.count_nonzero(rendered | target_mask))
    return float(intersection / union) if union else 0.0


def _save_tracking_model(
    path: Path,
    *,
    canonical_surface_points: np.ndarray,
    canonical_surface_colors: np.ndarray,
    reference_pose: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    """Save only the lightweight data consumed by temporal rigid tracking."""

    np.savez_compressed(
        path,
        canonical_surface_points=np.asarray(
            canonical_surface_points, dtype=np.float32
        ),
        canonical_surface_colors=np.asarray(canonical_surface_colors),
        reference_pose=np.asarray(reference_pose, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
    )


def _run_object_attempt(
    pipeline: Any,
    foundationpose: FoundationPoseReferenceRefiner,
    *,
    record: dict[str, Any],
    object_dir: Path,
) -> dict[str, Any]:
    import torch
    import trimesh

    track_id = str(record["track_id"])
    physical_type = str(record.get("physical_type", "rigid"))
    rigid_shape = str(record["rigid_shape"])
    input_path = Path(record["input_npz"])
    object_dir.mkdir(parents=True, exist_ok=True)
    with np.load(input_path) as archive:
        rgb = np.asarray(archive["rgb"], dtype=np.uint8)
        mask = np.asarray(archive["mask"], dtype=bool)
        all_object_mask_union = np.asarray(
            archive["all_object_mask_union"], dtype=bool
        )
        camera_points = np.asarray(archive["camera_points"], dtype=np.float32)
        camera_pose = np.asarray(archive["camera_pose"], dtype=np.float32)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float32)
    if rgb.shape[:2] != mask.shape or camera_points.shape != (*mask.shape, 3):
        raise ValueError(f"inconsistent SAM 3D inputs for {track_id}")
    if all_object_mask_union.shape != mask.shape:
        raise ValueError(f"inconsistent occluder mask input for {track_id}")

    rgba = np.dstack((rgb, mask.astype(np.uint8) * 255))
    pointmap = torch.from_numpy(sam3d_pointmap_from_opencv(camera_points))
    sam3d_started = perf_counter()
    result = pipeline.run(
        rgba,
        None,
        int(record.get("seed", 42)),
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=True,
        use_vertex_color=True,
        pointmap=pointmap,
    )
    sam3d_elapsed = perf_counter() - sam3d_started
    glb = result.get("glb")
    if glb is None:
        raise RuntimeError(f"SAM 3D produced no GLB for {track_id}")
    if record.get("debug", False) or record.get("tracking_branch") == "static":
        glb.export(object_dir / "sam3d_raw.glb")
    if "iou" not in result:
        raise RuntimeError(
            f"SAM 3D layout post-optimization failed for {track_id}; "
            "refusing to freeze an unadjusted mesh"
        )
    merge_started = perf_counter()
    mesh = _merged_mesh(glb)
    sam3d_source_vertex_count = int(len(mesh.vertices))
    sam3d_source_face_count = int(len(mesh.faces))
    conditioned = condition_sam3d_mesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        sam3d_rotation_wxyz=_tensor_array(result["rotation"]),
        sam3d_translation=_tensor_array(result["translation"]),
        sam3d_scale=_tensor_array(result["scale"]),
        camera_to_world=camera_pose,
    )
    source_canonical_mesh = mesh.copy()
    source_canonical_mesh.vertices = np.asarray(
        conditioned.canonical_vertices, dtype=np.float64
    )
    mesh_conditioning = condition_rigid_mesh(
        source_canonical_mesh, rigid_shape
    )
    canonical_mesh = mesh_conditioning.mesh
    canonical_vertices = np.asarray(
        canonical_mesh.vertices, dtype=np.float64
    )
    conditioned = replace(
        conditioned,
        canonical_vertices=canonical_vertices.astype(np.float32),
        faces=np.asarray(canonical_mesh.faces, dtype=np.int32),
        reference_world_vertices=apply_rigid_pose(
            canonical_vertices, conditioned.reference_pose
        ).astype(np.float32),
    )
    if rigid_shape == "box":
        conditioned = reframe_box_canonical_mesh(
            conditioned,
            axes=np.asarray(mesh_conditioning.fit["axes"]),
            center=np.asarray(mesh_conditioning.fit["center"]),
        )
        canonical_mesh.vertices = conditioned.canonical_vertices
    merge_condition_elapsed = perf_counter() - merge_started
    metric_local_mesh = canonical_mesh.copy()
    metric_local_mesh.vertices = (
        np.asarray(conditioned.canonical_vertices, dtype=np.float64)
        / conditioned.track4world_world_scale
    )
    proxy_mesh = metric_local_mesh
    seed_pose_camera = reference_camera_pose(
        conditioned, camera_to_world=camera_pose
    )
    seed_proxy_camera = apply_rigid_pose(
        np.asarray(proxy_mesh.vertices), seed_pose_camera
    )
    seed_visible_iou = _projection_iou(
        vertices_camera=seed_proxy_camera,
        faces=np.asarray(proxy_mesh.faces),
        intrinsics=intrinsics,
        target_mask=mask,
        all_object_mask_union=all_object_mask_union,
    )
    reference_alignment = align_reference_mesh(
        foundationpose=foundationpose, mesh=proxy_mesh,
        seed_pose_camera=seed_pose_camera, rgb=rgb, mask=mask,
        camera_points=camera_points, intrinsics=intrinsics,
        all_object_mask_union=all_object_mask_union, rigid_shape=rigid_shape,
        reference_surface_prior=record.get('reference_surface_prior'), camera_to_world=camera_pose,
    )
    refined_pose_camera = reference_alignment.pose_camera
    foundationpose_diagnostics = reference_alignment.diagnostics
    alignment = reference_alignment.alignment
    rendered_mask = reference_alignment.rendered_mask
    visible_rendered_mask = reference_alignment.visible_mask
    alignment_elapsed = reference_alignment.timings["area_depth_alignment_seconds"]
    foundationpose_elapsed = reference_alignment.timings["total_seconds"] - alignment_elapsed
    ratio_vertices = np.asarray(reference_alignment.mesh.vertices) * conditioned.track4world_world_scale
    conditioned = replace(conditioned, canonical_vertices=ratio_vertices.astype(np.float32))
    conditioned = apply_reference_camera_pose(
        conditioned, camera_to_world=camera_pose,
        reference_pose_camera=refined_pose_camera,
    )
    conditioned = apply_projection_alignment(
        conditioned, camera_to_world=camera_pose, alignment=alignment,
    )

    np.savez_compressed(
        object_dir / "reference_projection_alignment.npz",
        target_mask=mask,
        all_object_mask_union=all_object_mask_union,
        other_object_mask=all_object_mask_union & ~mask,
        rendered_mask=rendered_mask,
        visible_rendered_mask=visible_rendered_mask,
    )
    canonical_mesh = canonical_mesh.copy()
    canonical_mesh.vertices = conditioned.canonical_vertices
    canonical_path = object_dir / "canonical_metric.glb"
    export_started = perf_counter()
    canonical_mesh.export(canonical_path)
    canonical_export_elapsed = perf_counter() - export_started

    sample_count = int(record.get("surface_point_count", 20_000))
    sampling_started = perf_counter()
    sample_points, sample_faces = trimesh.sample.sample_surface(
        canonical_mesh, sample_count, seed=int(record.get("seed", 42))
    )
    colors = np.empty((0, 4), dtype=np.uint8)
    vertex_colors = getattr(canonical_mesh.visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == len(canonical_mesh.vertices):
        colors = np.asarray(vertex_colors, dtype=np.uint8)[
            np.asarray(canonical_mesh.faces)[sample_faces, 0]
        ]
    sampling_elapsed = perf_counter() - sampling_started
    npz_path = object_dir / "canonical_rigid_model.npz"
    tracking_model_started = perf_counter()
    _save_tracking_model(
        npz_path,
        canonical_surface_points=sample_points,
        canonical_surface_colors=colors,
        reference_pose=conditioned.reference_pose,
        intrinsics=intrinsics,
    )
    tracking_model_elapsed = perf_counter() - tracking_model_started

    metadata = {
        "track_id": track_id,
        "rigid_shape": rigid_shape,
        "physical_type": physical_type,
        "tracking_branch": record.get("tracking_branch"),
        "seed": int(record.get("seed", 42)),
        "reference_surface_prior": record.get('reference_surface_prior'),
        "frame_index": int(record["frame_index"]),
        "input_npz": str(input_path),
        "canonical_glb": str(canonical_path),
        "canonical_model": str(npz_path),
        "complete_mesh_storage": "canonical_glb_only",
        "tracking_model_storage": "sampled_points_pose_and_intrinsics_only",
        "layout_postprocess": True,
        "pointmap_source": "track4world_reference_camera_points",
        "sam3d_elapsed_seconds": sam3d_elapsed,
        "foundationpose_elapsed_seconds": foundationpose_elapsed,
        "area_depth_alignment_elapsed_seconds": alignment_elapsed,
        "reference_adjustment_elapsed_seconds": (
            foundationpose_elapsed + alignment_elapsed
        ),
        "mesh_merge_and_condition_elapsed_seconds": merge_condition_elapsed,
        "canonical_glb_export_elapsed_seconds": canonical_export_elapsed,
        "surface_sampling_elapsed_seconds": sampling_elapsed,
        "tracking_model_save_elapsed_seconds": tracking_model_elapsed,
        "surface_point_count": sample_count,
        "layout_iou": (
            float(_tensor_array(result["iou"]).reshape(-1)[0])
            if "iou" in result
            else None
        ),
        "reference_projection_alignment": alignment.to_metadata(),
        "foundationpose_reference_refinement": {
            "seed_pose_camera_4x4": seed_pose_camera.tolist(),
            "refined_pose_camera_4x4": refined_pose_camera.tolist(),
            "seed_visible_iou": seed_visible_iou,
            "refined_visible_iou_before_scale_depth": (
                alignment.initial_visible_iou
            ),
            "proxy_method": "physmind_conditioned_mesh_direct",
            "proxy_face_count": int(len(proxy_mesh.faces)),
            "complete_mesh_face_count": int(len(canonical_mesh.faces)),
            "iterations": 5,
            "diagnostics": foundationpose_diagnostics,
        },
        "mesh_conditioning": {
            "method": "physmind_clevrer_primitive_or_qem",
            "action": mesh_conditioning.action,
            "fit": mesh_conditioning.fit,
            "representative_color": mesh_conditioning.representative_color,
            "sam3d_source_vertex_count": sam3d_source_vertex_count,
            "sam3d_source_face_count": sam3d_source_face_count,
            "conditioned_vertex_count": int(len(canonical_mesh.vertices)),
            "conditioned_face_count": int(len(canonical_mesh.faces)),
        },
        **conditioned.to_metadata(),
    }
    metadata_path = object_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _run_object(
    pipeline: Any,
    foundationpose: FoundationPoseReferenceRefiner,
    *,
    record: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Retry low-IoU static rigids once, with independent geometry and alignment."""
    object_dir = output_root / str(record["track_id"])
    if record.get("physical_type", "rigid") != "rigid" or record.get("tracking_branch") != "static":
        return _run_object_attempt(
            pipeline, foundationpose, record=record, object_dir=object_dir,
        )

    started = perf_counter()
    seed = int(record.get("seed", 42))
    attempts_dir = object_dir / "attempts"
    attempts_dir.mkdir(parents=True)
    first_dir = attempts_dir / f"seed_{seed}"
    first = _run_object_attempt(
        pipeline, foundationpose, record=dict(record), object_dir=first_dir,
    )
    def score(item):
        return float(item["reference_projection_alignment"]["final_visible_iou"])

    selected, selected_dir = first, first_dir
    attempts = [dict(seed=seed, status="completed", final_visible_iou=score(first),
                     metadata=str(first_dir / "metadata.json"))]
    triggered = score(first) < 0.75
    if triggered:
        retry_seed = 7 if seed != 7 else 42
        retry_dir = attempts_dir / f"seed_{retry_seed}"
        try:
            retry = _run_object_attempt(
                pipeline, foundationpose, record={**record, "seed": retry_seed},
                object_dir=retry_dir,
            )
        except Exception as exc:
            # Preserve the usable first result and remove only incomplete retry output.
            if retry_dir.exists():
                shutil.rmtree(retry_dir)
            attempts.append(dict(seed=retry_seed, status="failed", error=str(exc)))
        else:
            attempts.append(dict(seed=retry_seed, status="completed", final_visible_iou=score(retry),
                                 metadata=str(retry_dir / "metadata.json")))
            if score(retry) > score(first):
                selected, selected_dir = retry, retry_dir

    # Publish one complete candidate; retained attempt metadata keeps its own paths.
    for path in selected_dir.iterdir():
        if path.is_file() and path.name != "metadata.json":
            shutil.copy2(path, object_dir / path.name)
    metadata = dict(selected)
    for key in ("canonical_glb", "canonical_model"):
        metadata[key] = str(object_dir / Path(selected[key]).name)
    metadata["static_iou_retry"] = dict(
        threshold=0.75, metric="reference_projection_alignment.final_visible_iou",
        triggered=triggered, attempts=attempts, selected_seed=selected["seed"],
        selected_metadata=str(selected_dir / "metadata.json"),
        elapsed_seconds=perf_counter() - started,
    )
    (object_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8",
    )
    return metadata


def _run_manifest(
    *,
    pipeline: Any,
    foundationpose: FoundationPoseReferenceRefiner,
    manifest_path: Path,
    output_dir: Path,
    config_path: Path,
    load_diagnostics: dict[str, Any],
    foundationpose_cold_start_seconds: float,
    worker_request_index: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("objects")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must contain a non-empty objects list")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    outputs = [
        _run_object(
            pipeline,
            foundationpose,
            record=record,
            output_root=output_dir,
        )
        for record in records
    ]
    summary = {
        "config": str(config_path),
        "object_count": len(outputs),
        "foundationpose_cold_start_seconds": foundationpose_cold_start_seconds,
        "model_load": load_diagnostics,
        "worker_request_index": worker_request_index,
        "model_reused_from_task_worker": worker_request_index > 0,
        "elapsed_seconds": perf_counter() - started,
        "objects": outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _initialize_models(
    config_path: Path,
) -> tuple[
    Any,
    FoundationPoseReferenceRefiner,
    dict[str, Any],
    float,
]:
    pipeline, load_diagnostics = _load_pipeline(config_path)
    foundationpose_started = perf_counter()
    foundationpose = FoundationPoseReferenceRefiner()
    foundationpose_cold_start_seconds = perf_counter() - foundationpose_started
    return (
        pipeline,
        foundationpose,
        load_diagnostics,
        foundationpose_cold_start_seconds,
    )


def _run_worker(config_path: Path) -> None:
    (
        pipeline,
        foundationpose,
        load_diagnostics,
        foundationpose_cold_start_seconds,
    ) = _initialize_models(config_path)
    request_index = 0
    response_prefix = "__PHYSMIND_SAM3D_RESPONSE__"
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                return
            summary = _run_manifest(
                pipeline=pipeline,
                foundationpose=foundationpose,
                manifest_path=Path(request["manifest"]),
                output_dir=Path(request["output_dir"]),
                config_path=config_path,
                load_diagnostics=load_diagnostics,
                foundationpose_cold_start_seconds=(
                    foundationpose_cold_start_seconds if request_index == 0 else 0.0
                ),
                worker_request_index=request_index,
            )
            response = {
                "ok": True,
                "request_index": request_index,
                "object_count": summary["object_count"],
            }
            request_index += 1
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        print(response_prefix + json.dumps(response), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if args.manifest is not None or args.output_dir is not None:
            parser.error("--worker cannot be combined with one-shot paths")
        _run_worker(args.config)
        return
    if args.manifest is None or args.output_dir is None:
        parser.error("one-shot mode requires --manifest and --output-dir")
    (
        pipeline,
        foundationpose,
        load_diagnostics,
        foundationpose_cold_start_seconds,
    ) = _initialize_models(args.config)
    _run_manifest(
        pipeline=pipeline,
        foundationpose=foundationpose,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config_path=args.config,
        load_diagnostics=load_diagnostics,
        foundationpose_cold_start_seconds=foundationpose_cold_start_seconds,
        worker_request_index=0,
    )


if __name__ == "__main__":
    main()
