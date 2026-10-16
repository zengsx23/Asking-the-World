"""Normalized spatial transport and stable-window quantity supervision for fluids."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from atw.polyworld.fluid_swr import FluidScene


@dataclass(frozen=True)
class StableVolumeWindow:
    start_frame: int
    end_frame: int
    visible_area_pixels2: float
    relative_tolerance: float


def stable_volume_windows(scene: FluidScene) -> list[StableVolumeWindow]:
    """Find observation plateaus after emission, without assuming the last frame is settled."""
    union = scene.target_masks.any(axis=1)
    area = union.sum(axis=(1, 2)).astype(np.float64)
    centers = np.zeros((scene.frame_count, 2))
    for frame, mask in enumerate(union):
        y, x = np.where(mask)
        if len(x):
            centers[frame] = [x.mean(), y.mean()]
    emitting = np.zeros(scene.frame_count, bool)
    for emitter in scene.emitters:
        emitting[list(emitter.active_frames)] = True
    width = max(5, round(scene.fps))
    cooldown = max(1, round(scene.fps * 0.5))
    stable = np.zeros(scene.frame_count, bool)
    for start in range(scene.frame_count - width + 1):
        end = start + width
        if emitting[max(0, start - cooldown) : end].any():
            continue
        a = area[start:end]
        median = np.median(a)
        if median < 16 or np.median(abs(a - median)) > 0.05 * median:
            continue
        third = max(1, width // 3)
        if abs(np.median(a[:third]) - np.median(a[-third:])) > 0.05 * median:
            continue
        c = centers[start:end]
        if np.max(np.quantile(c, 0.95, axis=0) - np.quantile(c, 0.05, axis=0)) > 2:
            continue
        stable[start:end] = True
    edges = np.diff(np.r_[False, stable, False].astype(int))
    windows = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a = area[start:end]
        median = float(np.median(a))
        tolerance = max(0.05, float(2 * np.median(abs(a - median)) / median))
        windows.append(StableVolumeWindow(int(start), int(end - 1), median, tolerance))
    return windows


def projected_quantiles(
    points: torch.Tensor,
    valid: torch.Tensor,
    directions: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    """Inverse empirical CDF with unit-sum equal-area weights; no spatial alignment."""
    if points.shape[-2] == 0:
        return points.new_zeros((*points.shape[:-2], len(directions), len(levels)))
    projection = (points @ directions.T).transpose(-1, -2)
    ordered = projection.masked_fill(~valid.unsqueeze(-2), float("inf")).sort(-1).values
    count = valid.sum(-1)
    indices = (count[..., None, None] * levels).long()
    indices = indices.clamp(0, points.shape[-2] - 1).expand(
        *points.shape[:-2], len(directions), len(levels)
    )
    values = torch.gather(ordered, -1, indices)
    return torch.where(count[..., None, None] > 0, values, 0.0)


class FluidDistributionLoss:
    def __init__(
        self,
        scene: FluidScene,
        frames: list[int],
        device: str,
        direction_count: int = 16,
        quantile_count: int = 64,
        volume_weight: float = 4.0,
        settled_weight: float = 0.0,
        spatial_mode: str = "multiscale",
    ):
        if (
            direction_count < 2
            or quantile_count < 2
            or min(volume_weight, settled_weight) < 0
        ):
            raise ValueError("invalid distribution-loss configuration")
        self.scene, self.frames, self.volume_weight = scene, frames, volume_weight
        self.settled_weight = settled_weight
        if spatial_mode not in ("swd", "multiscale"):
            raise ValueError("unknown spatial distribution loss")
        self.spatial_mode = spatial_mode
        self.multiscale = None
        if spatial_mode == "multiscale":
            from atw.polyworld.fluid_spatial_loss import MultiscaleColorLoss

            self.multiscale = MultiscaleColorLoss(scene, frames, device)
        angles = np.arange(direction_count) * math.pi / direction_count
        directions = np.column_stack((np.cos(angles), np.sin(angles))).astype(
            np.float32
        )
        levels = (np.arange(quantile_count) + 0.5) / quantile_count
        self.directions = torch.as_tensor(directions, device=device)
        self.levels = torch.as_tensor(levels, dtype=torch.float32, device=device)
        self.windows = stable_volume_windows(scene)
        phases = (
            len(scene.fluid_ids) + 1
        )  # Include the union to retain inter-color allocation.
        target = np.zeros(
            (len(frames), phases, direction_count, quantile_count), np.float32
        )
        present = np.zeros((len(frames), phases), bool)
        visible = []
        for slot, frame in enumerate(frames):
            masks = scene.target_masks[frame]
            union = masks.any(axis=0)
            # Only known foreground-obstacle pixels are treated as occluded.
            visible.append(~scene.obstacle_mask | union)
            for phase, mask in enumerate([*masks, union]):
                y, x = np.where(mask)
                if len(x):
                    points = np.column_stack((x + 0.5, y + 0.5)).astype(np.float32)
                    ordered = np.sort(points @ directions.T, axis=0)
                    target[slot, phase] = ordered[(levels * len(x)).astype(int)].T
                    present[slot, phase] = True
        self.target = torch.as_tensor(target, device=device)
        self.present = torch.as_tensor(present, device=device)
        self.visible = torch.as_tensor(np.asarray(visible), device=device)
        self.window_steps = torch.as_tensor(
            [w.end_frame for w in self.windows], device=device
        )
        self.window_area = torch.as_tensor(
            [w.visible_area_pixels2 for w in self.windows], device=device
        )
        self.window_tolerance = torch.as_tensor(
            [w.relative_tolerance for w in self.windows], device=device
        )
        self.settled_regions = []
        self.vertical = torch.tensor([[0.0, 1.0]], device=device)
        for name, bounds in (scene.basin_bounds or {}).items():
            x0, y0, x1, y1 = bounds
            target_y = np.zeros(
                (len(frames), len(scene.fluid_ids), quantile_count), np.float32
            )
            enabled = np.zeros((len(frames), len(scene.fluid_ids)), bool)
            depth = np.ones(len(frames), np.float32)
            for slot, frame in enumerate(frames):
                if not any(w.start_frame <= frame <= w.end_frame for w in self.windows):
                    continue
                masks = scene.target_masks[frame, :, y0:y1, x0:x1]
                occupied = masks.sum((1, 2)) >= 8
                if occupied.sum() < 2:
                    continue
                y, _ = np.where(masks.any(axis=0))
                depth[slot] = max(
                    4.0, float(np.quantile(y, 0.95) - np.quantile(y, 0.05))
                )
                for phase in np.flatnonzero(occupied):
                    y, _ = np.where(masks[phase])
                    target_y[slot, phase] = np.sort(y + y0 + 0.5)[
                        (levels * len(y)).astype(int)
                    ]
                    enabled[slot, phase] = True
            if enabled.any():
                self.settled_regions.append(
                    (
                        name,
                        bounds,
                        torch.as_tensor(target_y, device=device),
                        torch.as_tensor(enabled, device=device),
                        torch.as_tensor(depth, device=device),
                    )
                )

    def __call__(
        self,
        positions: torch.Tensor,
        groups: torch.Tensor,
        births: torch.Tensor,
        particle_area: float,
        substeps: int,
        *,
        diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length = positions.shape[:2]
        finite = torch.isfinite(positions).all(-1)
        valid = finite & (positions >= 0).all(-1)
        safe = torch.nan_to_num(positions)
        x = safe[..., 0].long().clamp(0, self.scene.image_width - 1)
        y = safe[..., 1].long().clamp(0, self.scene.image_height - 1)
        times = torch.arange(length, device=positions.device)[None, :, None]
        valid &= self.visible[times, y, x]
        errors = []
        squared_pixels = []
        # SWD is report-only under the multiscale objective. Always evaluate
        # terms with nonzero weights, even when report diagnostics are deferred.
        compute_swd = diagnostics or self.multiscale is None
        swd_phases = range(len(self.scene.fluid_ids) + 1) if compute_swd else ()
        for phase in swd_phases:
            selected = (
                groups == phase
                if phase < len(self.scene.fluid_ids)
                else torch.ones_like(groups, dtype=torch.bool)
            )
            p, mask = safe[:, :, selected], valid[:, :, selected]
            predicted = projected_quantiles(p, mask, self.directions, self.levels)
            distance = ((predicted - self.target[None, :length, phase]) ** 2).mean(
                (-1, -2)
            )
            observed = self.present[None, :length, phase]
            simulated = mask.any(-1)
            missing = float(self.scene.image_width**2 + self.scene.image_height**2)
            distance = torch.where(observed & ~simulated, missing, distance)
            distance = torch.where(observed, distance, 0.0)
            # Known-visible false presence is penalized, without rewarding fewer parcels.
            errors.append(distance / 20.0**2 + (~observed & simulated).float())
            squared_pixels.append(distance)
        shape = (
            torch.stack(errors, -1).mean((1, 2))
            if compute_swd else positions.new_zeros(batch)
        )
        swd_shape = shape
        if self.multiscale is not None:
            shape = self.multiscale(safe, groups, valid, self.present)
        observed_count = self.present[:length].sum().clamp_min(1)
        swd_pixels2 = (
            torch.stack(squared_pixels, -1).sum((1, 2)) / observed_count
            if compute_swd else None
        )
        quantity = positions.new_zeros(batch)
        relative_error = positions.new_zeros(batch)
        if self.windows:
            end_step = (self.window_steps + 1) * substeps
            volume = (births[:, None] < end_step[None, :, None]).sum(-1) * particle_area
            error = (volume - self.window_area) / self.window_area.clamp_min(1)
            relative_error = error.abs().mean(-1)
            quantity = (
                (error.abs() - self.window_tolerance).clamp_min(0).square().mean(-1)
            )
        settled = positions.new_zeros(batch)
        settled_pixels2 = positions.new_zeros(batch)
        settled_count = positions.new_zeros(())
        compute_settled = diagnostics or self.settled_weight > 0
        regions = self.settled_regions if compute_settled else ()
        for _, (x0, y0, x1, y1), target_y, enabled, depth in regions:
            in_basin = valid & (safe[..., 0] >= x0) & (safe[..., 0] < x1)
            in_basin &= (safe[..., 1] >= y0) & (safe[..., 1] < y1)
            for phase in range(len(self.scene.fluid_ids)):
                selected = groups == phase
                mask = in_basin[:, :, selected]
                quantiles = projected_quantiles(
                    safe[:, :, selected], mask, self.vertical, self.levels
                )
                error = (
                    (quantiles.squeeze(-2) - target_y[None, :length, phase])
                    .square()
                    .mean(-1)
                )
                # Missing a locally observed phase must not erase its layer penalty.
                error = torch.where(mask.any(-1), error, float((y1 - y0) ** 2))
                active = enabled[None, :length, phase]
                settled += (error / depth[None, :length].square() * active).sum(-1)
                settled_pixels2 += (error * active).sum(-1)
                settled_count += active.sum()
        settled /= settled_count.clamp_min(1)
        settled_pixels2 /= settled_count.clamp_min(1)
        total = shape + self.volume_weight * quantity + self.settled_weight * settled
        components = dict(
            distribution=shape,
            stable_quantity=quantity,
            stable_quantity_relative_error=relative_error,
        )
        if compute_swd:
            components.update(
                swd_distribution=swd_shape, swd_pixels=torch.sqrt(swd_pixels2)
            )
        if compute_settled:
            components.update(
                settled_distribution=settled,
                settled_vertical_rmse_pixels=torch.sqrt(settled_pixels2),
            )
        return total, components

    def metadata(self) -> dict:
        return dict(
            mode=self.spatial_mode,
            spatial_reference=(
                "five normalized color-field levels, 2px base grid, smoothed 2x decimation; squared L2 / target energy"
                if self.multiscale is not None
                else "global sliced Wasserstein"
            ),
            directions=len(self.directions),
            quantiles=len(self.levels),
            stable_volume_weight=self.volume_weight,
            settled_distribution_weight=self.settled_weight,
            settled_regions=[item[0] for item in self.settled_regions],
            settled_constraint_active=bool(self.settled_regions)
            and self.settled_weight > 0,
            settled_reference="per-color vertical quantiles in observed multi-phase basins during stable windows; normalized by observed union depth",
            quantity_constraint_active=bool(self.windows),
            stable_windows=[asdict(w) for w in self.windows],
            quantity_reference="visible union area in quiet stable windows; 2D proxy, not GT volume",
            visibility="known static obstacle pixels only; unknown segmentation omissions remain unresolved",
        )
