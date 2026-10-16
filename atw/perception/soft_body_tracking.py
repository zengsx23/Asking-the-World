"""Local-motion identities attached to the current smooth visible depth surface.

The complete, optionally primitive-fitted reference mesh is retained by
reconstruction. Only its visible surface is seeded here. Subsequent deformation
uses no hidden-surface completion, rigid fitting, primitive fitting, rest-length,
volume, or elasticity constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from atw.perception.mesh_projection import render_mesh_depth
from atw.perception.visible_surface import fit_surface, masked_sample, open_surface_topology, depth_statistics
from atw.perception.track4world_tool import AdjacentTrack4WorldResult, CanonicalTrajectory, _bilinear_sample


@dataclass(frozen=True)
class SoftBodyTrackingResult:
    trajectory: CanonicalTrajectory
    proposal_uv: np.ndarray
    metrics: list[dict[str, Any]]
    surface_depth: np.ndarray
    reference_extent: float


def project_world(
    points: np.ndarray, camera: np.ndarray, intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    xyz = (homogeneous @ np.linalg.inv(camera).T)[:, :3]
    projected = xyz @ intrinsic.T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    return uv.astype(np.float32), xyz[:, 2].astype(np.float32)


def unproject_world(
    uv: np.ndarray, depth: np.ndarray, camera: np.ndarray, intrinsic: np.ndarray,
) -> np.ndarray:
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(intrinsic).T
    xyz = rays * np.asarray(depth)[:, None]
    return (xyz @ camera[:3, :3].T + camera[:3, 3]).astype(np.float32)


def reference_surface_from_mesh(
    *, vertices_world: np.ndarray, faces: np.ndarray,
    camera_to_world: np.ndarray, intrinsic: np.ndarray, target_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample nearest SAM3D triangle intersections at reference mask pixels."""
    camera_vertices = (
        np.column_stack((vertices_world, np.ones(len(vertices_world))))
        @ np.linalg.inv(camera_to_world).T
    )[:, :3]
    rendered, depth = render_mesh_depth(
        vertices_camera=camera_vertices, faces=faces, intrinsic=intrinsic,
        image_shape=target_mask.shape,
    )
    valid = target_mask & rendered & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    if not len(x):
        raise ValueError("aligned SAM3D soft-body mesh has no visible mask intersection")
    uv = np.column_stack((x, y)).astype(np.float32)
    xyz = unproject_world(uv, depth[y, x], camera_to_world, intrinsic)
    return uv, xyz


def point_mask(uv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fixed one-pixel-radius disks; radius is never optimized to improve IoU."""
    mask = np.zeros(shape, dtype=np.uint8)
    finite = np.isfinite(uv).all(axis=1)
    pixels = np.rint(uv[finite]).astype(np.int64)
    h, w = shape
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    pixels = pixels[inside]
    mask[pixels[:, 1], pixels[:, 0]] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel).astype(bool)


def visible_iou(
    uv: np.ndarray, mask: np.ndarray, occluder: np.ndarray,
) -> float | None:
    rendered = point_mask(uv, mask.shape) & ~occluder
    union = np.count_nonzero(rendered | mask)
    return float(np.count_nonzero(rendered & mask) / union) if union else None


def correct_silhouette(
    uv: np.ndarray, confidence: np.ndarray, mask: np.ndarray,
    occluder: np.ndarray, *, iterations: int = 24, device: str = "cpu",
) -> np.ndarray:
    """Optimize independent image-plane coordinates at fixed predicted depth.

Track attachment is a data term, not a distance-preserving shape prior. Bilinear
point splats provide the differentiable silhouette; hard fixed-radius IoU selects
the retained iterate, so correction never degrades that measured score.
Target-mask projection supplies gradients with a useful starting point when
nearby splats and the visible target are disjoint; it adds no shape prior.
"""
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import distance_transform_edt

    if not len(uv) or not mask.any() or iterations == 0:
        return uv.copy()
    height, width = mask.shape
    nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)

    def project_to_mask(candidate):
        """Mask membership projection gives nearby disjoint splats a valid start."""
        candidate = candidate.copy()
        pixels = np.rint(candidate).astype(int)
        px = np.clip(pixels[:, 0], 0, width - 1)
        py = np.clip(pixels[:, 1], 0, height - 1)
        outside = ~mask[py, px] | (px != pixels[:, 0]) | (py != pixels[:, 1])
        candidate[outside] = nearest[:, py[outside], px[outside]][::-1].T
        return candidate
    # Work in a crop without changing pixel scale or splat radius.
    y, x = np.nonzero(mask)
    lower = np.floor(
        np.minimum(uv.min(axis=0), [x.min(), y.min()]) - 4
    ).astype(int)
    upper = np.ceil(
        np.maximum(uv.max(axis=0), [x.max(), y.max()]) + 5
    ).astype(int)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, [width, height])
    left, top = lower
    right, bottom = upper
    h, w = bottom - top, right - left
    if h < 1 or w < 1:
        return uv.copy()
    origin = np.asarray([left, top], dtype=np.float32)
    anchor = torch.as_tensor(uv - origin, dtype=torch.float32, device=device)
    initial = project_to_mask(uv)
    positions = torch.as_tensor(
        initial - origin, dtype=torch.float32, device=device
    ).clone().requires_grad_(True)
    weights = torch.as_tensor(confidence, dtype=torch.float32, device=device)
    target = torch.as_tensor(
        mask[top:bottom, left:right], dtype=torch.float32, device=device
    )
    allowed = torch.as_tensor(
        ~occluder[top:bottom, left:right], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam([positions], lr=0.35)
    best = uv.copy()
    best_iou = visible_iou(best, mask, occluder)
    initial_iou = visible_iou(initial, mask, occluder)
    if initial_iou > best_iou:
        best, best_iou = initial, initial_iou
    for _ in range(iterations):
        base = torch.floor(positions).detach().long()
        fraction = positions - base
        density = torch.zeros(h * w, dtype=torch.float32, device=device)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            px, py = base[:, 0] + dx, base[:, 1] + dy
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            mass = (
                (fraction[:, 0] if dx else 1 - fraction[:, 0])
                * (fraction[:, 1] if dy else 1 - fraction[:, 1])
            )
            density = density.scatter_add(0, py[valid] * w + px[valid], mass[valid])
        alpha = density.reshape(1, 1, h, w).clamp(0, 1)
        # Cross-shaped dilation matches point_mask's fixed-radius splats.
        padded = F.pad(alpha, (1, 1, 1, 1))
        alpha = torch.stack((
            padded[..., 1:-1, 1:-1], padded[..., :-2, 1:-1],
            padded[..., 2:, 1:-1], padded[..., 1:-1, :-2],
            padded[..., 1:-1, 2:],
        )).amax(0)[0, 0] * allowed
        intersection = (alpha * target).sum()
        union = (alpha + target - alpha * target).sum().clamp_min(1)
        attachment = (
            (weights * (positions - anchor).square().sum(-1)).sum()
            / weights.sum().clamp_min(1)
        )
        loss = 1 - intersection / union + 0.002 * attachment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        candidate = project_to_mask(positions.detach().cpu().numpy() + origin)
        with torch.no_grad():
            positions.copy_(torch.as_tensor(candidate - origin, device=device))
        score = visible_iou(candidate, mask, occluder)
        if score > best_iou:
            best_iou, best = score, candidate.copy()
    return best


def track_soft_body_surface(
    *, initial_uv: np.ndarray, initial_xyz: np.ndarray, reference_frame: int,
    forward: AdjacentTrack4WorldResult, backward: AdjacentTrack4WorldResult,
    world_points: np.ndarray, camera_poses: np.ndarray, intrinsics: np.ndarray,
    masks: np.ndarray, all_object_mask_union: np.ndarray,
    observed_depth: np.ndarray, geometry_valid: np.ndarray, reference_extent: float,
    minimum_tracking_confidence: float = 0.2,
    silhouette_iterations: int = 24, device: str = "cpu",
) -> SoftBodyTrackingResult:
    """Generate adjacent UV/IDs, then attach to depth without feeding back into UV.

The recursive XYZ below is only an internal image-proposal state. It is not the
exported geometry: replacing it during recursion would change the computed UV
and ID result. No hidden trajectories or new identities are inferred.
"""
    frame_count, height, width = masks.shape
    point_count = len(initial_uv)
    uv = np.full((frame_count, point_count, 2), np.nan, dtype=np.float32)
    xyz = np.full((frame_count, point_count, 3), np.nan, dtype=np.float32)
    confidence = np.zeros((frame_count, point_count), dtype=np.float32)
    proposals = np.full_like(uv, np.nan)
    uv[reference_frame], xyz[reference_frame] = initial_uv, initial_xyz
    proposals[reference_frame] = initial_uv
    confidence[reference_frame] = 1
    metrics = []

    def record(frame, proposal, corrected, conf):
        occluder = all_object_mask_union[frame] & ~masks[frame]
        metrics.append({
            "frame_index": frame,
            "mask_pixels": int(masks[frame].sum()),
            "tracked_points": int(len(corrected)),
            "reference_points": point_count,
            "proposal_visible_iou": visible_iou(proposal, masks[frame], occluder),
            "corrected_visible_iou": visible_iou(corrected, masks[frame], occluder),
            "mean_correction_pixels": (
                float(np.linalg.norm(corrected - proposal, axis=1).mean())
                if len(corrected) else None
            ),
            "mean_step_confidence": float(conf.mean()) if len(conf) else None,
        })

    record(reference_frame, initial_uv, initial_uv, confidence[reference_frame])

    def advance(source, target, motion, pair):
        source_ids = np.flatnonzero(np.isfinite(xyz[source]).all(axis=1))
        if not len(source_ids) or not masks[target].any():
            record(target, np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
            return
        # A rounded-pixel collision between independent surface samples is not
        # evidence of physical occlusion. Keep subpixel identities; mask ownership
        # and correspondence confidence determine whether propagation is usable.
        # A point-cloud Z buffer belongs to rendering, not irreversible ID culling.
        source_uv = uv[source, source_ids]
        target_uv = _bilinear_sample(motion.source_to_target_uv[pair], source_uv)
        step_conf = _bilinear_sample(motion.tracking_confidence[pair], source_uv)
        first = _bilinear_sample(world_points[source], source_uv)
        second = _bilinear_sample(world_points[target], target_uv)
        predicted = xyz[source, source_ids] + second - first
        predicted_uv, depth = project_world(
            predicted, camera_poses[target], intrinsics[target]
        )
        valid = (
            np.isfinite(predicted).all(1) & np.isfinite(predicted_uv).all(1)
            & np.isfinite(step_conf) & (step_conf >= minimum_tracking_confidence)
            & (depth > 0)
        )
        rounded = np.rint(np.where(np.isfinite(target_uv), target_uv, -1)).astype(int)
        inside = (
            (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
        )
        valid &= inside
        occluder = all_object_mask_union[target] & ~masks[target]
        # Give near-boundary mismatches a chance to undergo mask correction.
        # Rejecting other-mask pixels here permanently erased useful identities
        # around contacts before the silhouette term could correct their UVs.
        near_mask = cv2.dilate(
            masks[target].astype(np.uint8), np.ones((11, 11), np.uint8)
        ).astype(bool)
        valid[inside] &= near_mask[rounded[inside, 1], rounded[inside, 0]]
        ids = source_ids[valid]
        proposal = predicted_uv[valid]
        corrected = correct_silhouette(
            proposal, step_conf[valid], masks[target], occluder,
            iterations=silhouette_iterations, device=device,
        )
        proposals[target, ids] = proposal
        uv[target, ids] = corrected
        xyz[target, ids] = unproject_world(
            corrected, depth[valid], camera_poses[target], intrinsics[target]
        )
        confidence[target, ids] = step_conf[valid]
        record(target, proposal, corrected, step_conf[valid])

    for source in range(reference_frame, frame_count - 1):
        advance(source, source + 1, forward, source)
    for source in range(reference_frame, 0, -1):
        advance(source, source - 1, backward, frame_count - 1 - source)
    surface_depth = np.full(masks.shape, np.nan, np.float32)
    by_frame = {row["frame_index"]: row for row in metrics}
    for frame, mask in enumerate(masks):
        valid = mask & geometry_valid[frame] & np.isfinite(observed_depth[frame]) & (observed_depth[frame] > 0)
        depth = fit_surface(observed_depth[frame], valid)
        surface_depth[frame] = depth
        z = masked_sample(depth, uv[frame], valid)
        visible = np.isfinite(z)
        uv[frame, ~visible] = np.nan
        confidence[frame, ~visible] = np.nan
        xyz[frame] = unproject_world(uv[frame], z, camera_poses[frame], intrinsics[frame])
        sampled = masked_sample(observed_depth[frame], uv[frame], valid)
        row = by_frame[frame]
        row.update(depth_statistics(z, sampled, reference_extent))
        row["tracked_points"] = int(visible.sum())
        row["corrected_visible_iou"] = visible_iou(uv[frame], mask, all_object_mask_union[frame] & ~mask)
        _, _, covered = open_surface_topology(valid)
        row["surface_iou"] = float((covered & mask).sum()/max(1, (covered | mask).sum())) if mask.any() else None
        row["surface_depth_coverage"] = float(covered.sum()/max(1, valid.sum())) if valid.any() else None
        row["surface_depth"] = depth_statistics(depth[covered], observed_depth[frame][covered], reference_extent)
    return SoftBodyTrackingResult(
        CanonicalTrajectory(uv=uv, model_xyz=xyz, tracking_confidence=confidence),
        proposals, sorted(metrics, key=lambda item: item["frame_index"]),
        surface_depth, reference_extent,
    )
