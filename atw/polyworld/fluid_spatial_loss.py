"""Normalized multiscale color fields without basin or gravity-direction priors."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from atw.polyworld.fluid_swr import FluidScene


def splat_points(points, valid, height, width, spacing=2):
    """Bilinear area deposition, with equal weights for valid particles."""
    gh, gw = math.ceil(height / spacing), math.ceil(width / spacing)
    xy = torch.where(valid[..., None], points, 0.0) / spacing - 0.5
    lower = xy.floor().long()
    fraction = xy - lower
    field = points.new_zeros((*points.shape[:-2], gh * gw))
    for dy in (0, 1):
        for dx in (0, 1):
            x = (lower[..., 0] + dx).clamp(0, gw - 1)
            y = (lower[..., 1] + dy).clamp(0, gh - 1)
            wx = fraction[..., 0] if dx else 1 - fraction[..., 0]
            wy = fraction[..., 1] if dy else 1 - fraction[..., 1]
            field.scatter_add_(-1, y * gw + x, wx * wy * valid)
    return field.reshape(*points.shape[:-2], gh, gw)


def normalized_pyramid(field, levels=5):
    """Smooth before decimation; normalize each level independently of parcel count."""
    shape = field.shape[:-2]
    x = field.reshape(-1, 1, *field.shape[-2:])
    result = []
    for level in range(levels):
        x = F.avg_pool2d(
            x, 3, stride=1 if level == 0 else 2, padding=1, count_include_pad=False
        )
        normalized = x / x.sum((-2, -1), keepdim=True).clamp_min(1e-12)
        result.append(normalized.reshape(*shape, *x.shape[-2:]))
    return result


class MultiscaleColorLoss:
    def __init__(self, scene: FluidScene, frames: list[int], device: str):
        self.height, self.width = scene.image_height, scene.image_width
        self.fluids = len(scene.fluid_ids)
        self.targets = []
        for phase in range(self.fluids + 1):
            fields = []
            for frame in frames:
                masks = scene.target_masks[frame]
                mask = masks[phase] if phase < self.fluids else masks.any(0)
                y, x = np.where(mask)
                points = torch.as_tensor(
                    np.column_stack((x + 0.5, y + 0.5)),
                    dtype=torch.float32,
                    device=device,
                )
                fields.append(
                    splat_points(
                        points,
                        torch.ones(len(x), dtype=torch.bool, device=device),
                        self.height,
                        self.width,
                    )
                )
            self.targets.append(normalized_pyramid(torch.stack(fields)))

    def __call__(self, positions, groups, valid, present):
        length = positions.shape[1]
        errors = []
        for phase in range(self.fluids + 1):
            selected = (
                groups == phase
                if phase < self.fluids
                else torch.ones_like(groups, dtype=torch.bool)
            )
            mask = valid[:, :, selected]
            field = splat_points(
                positions[:, :, selected], mask, self.height, self.width
            )
            predicted = normalized_pyramid(field)
            error = positions.new_zeros(positions.shape[:2])
            for p, target in zip(predicted, self.targets[phase]):
                q = target[None, :length]
                # Reference energy makes each spatial scale dimensionless.
                error += (p - q).square().sum((-2, -1)) / q.square().sum(
                    (-2, -1)
                ).clamp_min(1e-12)
            error /= len(predicted)
            observed = present[None, :length, phase]
            simulated = mask.any(-1)
            error = torch.where(observed, error, simulated.float())
            errors.append(error)
        return torch.stack(errors, -1).mean((1, 2))
