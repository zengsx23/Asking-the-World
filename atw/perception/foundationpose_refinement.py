"""Seeded single-candidate FoundationPose refinement for one reference frame."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONPOSE_ROOT = PROJECT_ROOT / "third_party" / "FoundationPose"


def add_foundationpose_to_path() -> None:
    if not FOUNDATIONPOSE_ROOT.exists():
        raise FileNotFoundError(
            f"FoundationPose submodule not found: {FOUNDATIONPOSE_ROOT}"
        )
    root = str(FOUNDATIONPOSE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def float32_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def _required_erode_depth_thresholds(
    *,
    depth: np.ndarray,
    ob_mask: np.ndarray,
    radius: int,
    ratio_thres: float,
    zfar: float = 100.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_np = np.asarray(depth, dtype=np.float32)
    mask_np = np.asarray(ob_mask) > 0
    valid_mask = (depth_np >= 0.001) & (depth_np < zfar) & mask_np
    height, width = depth_np.shape[:2]
    required: list[float] = []
    impossible_count = 0
    valid_count = 0

    for y, x in np.argwhere(valid_mask):
        center_depth = float(depth_np[y, x])
        diffs: list[float] = []
        invalid_count = 0
        total = 0
        y0 = max(0, int(y) - radius)
        y1 = min(height, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(width, int(x) + radius + 1)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                total += 1
                neighbor_depth = float(depth_np[yy, xx])
                if neighbor_depth < 0.001 or neighbor_depth >= zfar:
                    invalid_count += 1
                else:
                    diffs.append(abs(neighbor_depth - center_depth))

        valid_count += 1
        allowed_bad = int(np.floor(float(ratio_thres) * float(total)))
        allowed_diff_bad = allowed_bad - invalid_count
        if allowed_diff_bad < 0:
            impossible_count += 1
            continue
        if not diffs:
            required.append(0.0)
            continue
        sorted_diffs = np.sort(np.asarray(diffs, dtype=np.float32))
        keep_count = len(sorted_diffs) - int(allowed_diff_bad)
        if keep_count <= 0:
            required.append(0.0)
        else:
            required.append(
                float(sorted_diffs[min(keep_count - 1, len(sorted_diffs) - 1)])
            )

    required_np = np.asarray(required, dtype=np.float32)
    metadata: dict[str, Any] = {
        "mask_valid_depth_pixel_count": valid_count,
        "impossible_to_keep_pixel_count": impossible_count,
        "required_threshold_count": int(required_np.size),
    }
    if required_np.size:
        metadata.update(
            {
                "min": float(np.min(required_np)),
                "median": float(np.median(required_np)),
                "p90": float(np.percentile(required_np, 90.0)),
                "p95": float(np.percentile(required_np, 95.0)),
                "max": float(np.max(required_np)),
            }
        )
    return required_np, metadata


def foundationpose_adaptive_depth_preprocess(
    *,
    depth: np.ndarray,
    ob_mask: np.ndarray,
    radius: int = 2,
    min_valid_pixels: int = 4,
    device: str = "cuda",
    initial_depth_diff_thres: float = 0.001,
    max_depth_diff_thres: float = 0.1,
    target_max_filtered_ratio: float = 0.10,
    ratio_thres: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    add_foundationpose_to_path()
    from Utils import erode_depth

    required_thresholds, threshold_metadata = _required_erode_depth_thresholds(
        depth=depth,
        ob_mask=ob_mask,
        radius=radius,
        ratio_thres=ratio_thres,
    )
    if required_thresholds.size:
        percentile = 100.0 * (1.0 - float(target_max_filtered_ratio))
        selected_threshold = float(
            np.percentile(required_thresholds, percentile)
        )
        selected_threshold = max(
            float(initial_depth_diff_thres), selected_threshold
        )
        selected_threshold = min(
            float(max_depth_diff_thres), selected_threshold
        )
    else:
        selected_threshold = float(max_depth_diff_thres)

    selected_depth = float32_contiguous(
        erode_depth(
            depth,
            radius=radius,
            depth_diff_thres=selected_threshold,
            ratio_thres=ratio_thres,
            device=device,
        )
    )
    original_valid = int(
        ((np.asarray(depth) >= 0.001) & (np.asarray(ob_mask) > 0)).sum()
    )
    kept_valid = int(
        ((selected_depth >= 0.001) & (np.asarray(ob_mask) > 0)).sum()
    )
    filtered_ratio = float(1.0 - kept_valid / max(original_valid, 1))
    metadata = {
        "method": "adaptive_erode_depth",
        "threshold_selection": "per_pixel_required_threshold_percentile",
        "radius": radius,
        "min_valid_pixels": min_valid_pixels,
        "initial_depth_diff_thres": float(initial_depth_diff_thres),
        "max_depth_diff_thres": float(max_depth_diff_thres),
        "selected_depth_diff_thres": float(selected_threshold),
        "ratio_thres": float(ratio_thres),
        "target_max_filtered_ratio": float(target_max_filtered_ratio),
        "original_valid_depth_pixel_count": original_valid,
        "kept_valid_depth_pixel_count": kept_valid,
        "filtered_depth_pixel_ratio": filtered_ratio,
        "required_threshold_stats": threshold_metadata,
        "relaxed": bool(
            float(selected_threshold) > float(initial_depth_diff_thres)
        ),
    }
    return float32_contiguous(selected_depth), metadata



@dataclass(slots=True)
class PreparedReferenceMesh:
    mesh: Any
    tensors: dict[str, Any]
    diameter: float
    to_original: Any
    to_original_cpu: np.ndarray = field(init=False)
    to_centered_cpu: np.ndarray = field(init=False)

    def __post_init__(self):
        self.to_original_cpu = self.to_original.detach().cpu().numpy().copy()
        self.to_centered_cpu = np.linalg.inv(self.to_original_cpu)


class FoundationPoseReferenceSession:
    """One reference RGB/depth/mask, shared across explicitly prepared meshes."""

    def __init__(self, owner: FoundationPoseReferenceRefiner, *, rgb, depth, mask, intrinsic):
        import torch
        from Utils import bilateral_filter_depth, depth2xyzmap

        started = perf_counter()
        self.owner = owner
        self.rgb = np.ascontiguousarray(rgb)
        self.K = float32_contiguous(intrinsic)
        mask = np.asarray(mask, dtype=bool)
        filtered, self.depth_metadata = foundationpose_adaptive_depth_preprocess(
            depth=float32_contiguous(depth), ob_mask=mask, radius=2,
            min_valid_pixels=4, device="cuda")
        self.depth = bilateral_filter_depth(filtered, radius=2, device="cuda")
        self.valid_count = int(((self.depth >= .001) & mask).sum())
        self.xyz = depth2xyzmap(self.depth, self.K)
        # The upstream predictor accepts tensors unchanged when dtype/device match.
        # Keep these allocations scoped to this reference, across all candidates.
        self.rgb = torch.as_tensor(self.rgb, device="cuda", dtype=torch.float)
        self.depth = torch.as_tensor(self.depth, device="cuda", dtype=torch.float)
        self.xyz = torch.as_tensor(self.xyz, device="cuda", dtype=torch.float)
        self.timings = {"preprocess_seconds": perf_counter() - started,
                        "mesh_prepare_seconds": 0., "refine_seconds": 0.}

    def prepare_mesh(self, mesh: Any) -> PreparedReferenceMesh:
        """Preserve FP mesh conditioning without unused registration setup/IO."""
        import torch
        from Utils import compute_mesh_diameter, make_mesh_tensors, set_seed

        started = perf_counter()
        set_seed(0)
        conditioned = mesh.copy()
        conditioned.vertices = float32_contiguous(conditioned.vertices)
        conditioned.vertex_normals = float32_contiguous(conditioned.vertex_normals)
        center = (conditioned.vertices.max(axis=0) + conditioned.vertices.min(axis=0)) / 2
        centered = conditioned.copy()
        centered.vertices = centered.vertices - center.reshape(1, 3)
        diameter = float(compute_mesh_diameter(model_pts=centered.vertices, n_sample=10000))
        tensors = make_mesh_tensors(centered)
        centered.vertices = float32_contiguous(centered.vertices)
        centered.vertex_normals = float32_contiguous(centered.vertex_normals)
        transform = torch.eye(4, device="cuda", dtype=torch.float)
        transform[:3, 3] = -torch.as_tensor(center, device="cuda", dtype=torch.float)
        prepared = PreparedReferenceMesh(centered, tensors, diameter, transform)
        self.timings["mesh_prepare_seconds"] += perf_counter() - started
        return prepared

    def refine(self, *, prepared: PreparedReferenceMesh, seed_pose_4x4: np.ndarray,
               iterations: int = 5, pose_constraint=None) -> tuple[np.ndarray, dict[str, Any]]:
        from Utils import set_seed

        if iterations < 1:
            raise ValueError("FoundationPose refinement iterations must be positive")
        started = perf_counter()
        set_seed(0)
        seed = float32_contiguous(seed_pose_4x4).reshape(4, 4)
        applied = self.valid_count >= 4
        if pose_constraint is not None:
            pose = pose_constraint(seed)
            # The network renders centered vertices; constraints use original
            # canonical coordinates. Convert explicitly at every iteration.
            for _ in range(iterations):
                centered_seed = pose @ prepared.to_centered_cpu
                if applied:
                    poses, _ = self.owner.refiner.predict(
                        mesh=prepared.mesh, mesh_tensors=prepared.tensors,
                        rgb=self.rgb, depth=self.depth, K=self.K,
                        ob_in_cams=float32_contiguous(centered_seed).reshape(1, 4, 4), normal_map=None,
                        xyz_map=self.xyz, glctx=self.owner.glctx,
                        mesh_diameter=prepared.diameter, iteration=1, get_vis=False,
                        clear_cuda_cache=False)
                    pose = (poses[0] @ prepared.to_original).detach().cpu().numpy()
                pose = pose_constraint(pose)
        elif applied:
            poses, _ = self.owner.refiner.predict(
                mesh=prepared.mesh, mesh_tensors=prepared.tensors,
                rgb=self.rgb, depth=self.depth, K=self.K,
                ob_in_cams=seed.reshape(1, 4, 4), normal_map=None,
                xyz_map=self.xyz, glctx=self.owner.glctx,
                mesh_diameter=prepared.diameter, iteration=iterations, get_vis=False,
                clear_cuda_cache=False)
            pose = (poses[0] @ prepared.to_original).detach().cpu().numpy()
        else:
            pose = seed @ prepared.to_original_cpu
        self.timings["refine_seconds"] += perf_counter() - started
        metadata = {
            "registration_strategy": "keyframe_seeded_single_candidate_refine",
            "candidate_count": 1, "best_candidate_index": 0,
            "seed_centered_pose_4x4": seed.tolist(),
            "refined_pose_4x4": pose.tolist(), "refiner_applied": applied,
            "scorer_applied": False, "register_score": None,
            "scorer_skip_reason": "single_pose_candidate_selected_by_external_mask_iou",
            "valid_depth_pixel_count": self.valid_count,
            "adaptive_depth_preprocess": self.depth_metadata,
            "foundationpose_iterations": iterations,
            "pose_constraint_each_iteration": pose_constraint is not None,
        }
        if not applied:
            metadata["reason"] = "valid_depth_pixel_count_lt_4"
        return float32_contiguous(pose), metadata


class FoundationPoseReferenceRefiner:
    """Persistent learned refiner/context; per-reference inputs have scoped lifetime."""

    def __init__(self) -> None:
        add_foundationpose_to_path()
        from estimater import PoseRefinePredictor, dr, set_logging_format, set_seed

        set_logging_format()
        set_seed(0)
        self.glctx = dr.RasterizeCudaContext()
        self.refiner = PoseRefinePredictor()

    def prepare_reference(self, *, rgb, depth, mask, intrinsic) -> FoundationPoseReferenceSession:
        return FoundationPoseReferenceSession(
            self, rgb=rgb, depth=depth, mask=mask, intrinsic=intrinsic)

    def refine(self, *, mesh: Any, seed_pose_4x4: np.ndarray, rgb: np.ndarray,
               depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray,
               debug_dir: Path | None = None, iterations: int = 5):
        session = self.prepare_reference(rgb=rgb, depth=depth, mask=mask, intrinsic=intrinsic)
        prepared = session.prepare_mesh(mesh)
        pose, metadata = session.refine(
            prepared=prepared, seed_pose_4x4=seed_pose_4x4, iterations=iterations)
        metadata["timings"] = dict(session.timings)
        metadata["foundationpose_debug_dir"] = str(debug_dir) if debug_dir is not None else None
        return pose, metadata
