"""Batched, open-loop 2D PBF with finite-area parcels and inferred emission rates.

Density projection follows Macklin & Mueller (2013). XSPH supplies effective
viscosity; compact pairwise cohesion is a surface-tension approximation, not a
calibrated SI surface-tension coefficient. Observations enter only the loss and
the emitter setup, never the simulated state after emission.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
import warp as wp
from scipy.optimize import nnls

from atw.polyworld.fluid_swr import FluidScene, _collision_sdf, fluid_chamfer_rmse
from atw.polyworld.fluid_distribution_loss import FluidDistributionLoss


@dataclass(frozen=True)
class PBFConfig:
    spacing: float = 2.0
    nozzle_width: float = 4.0
    substeps: int = 4
    iterations: int = 4
    population: int = 24
    generations: int = 10
    restarts: int = 2
    seed: int = 7
    loss_stride: int = 4
    grid_scale: int = 2
    loss_mode: str = "multiscale"
    swd_directions: int = 16
    swd_quantiles: int = 64
    stable_volume_weight: float = 4.0
    settled_distribution_weight: float = 0.0
    restart_full_horizon: bool = False
    buffer_pingpong: bool = False
    graph_chunk_frames: int = 16


def _restart_cutoffs(frame_count: int, restart: int, full_horizon: bool) -> list[int]:
    if restart > 0 and full_horizon:
        return [frame_count - 1]
    return sorted({min(59, frame_count - 1), min(129, frame_count - 1), frame_count - 1})


def estimate_flux(scene: FluidScene) -> np.ndarray:
    """Visible-area growth initializes flux, without using original simulator GT."""
    design = np.zeros((scene.frame_count, len(scene.emitters)), np.float64)
    for source, emitter in enumerate(scene.emitters):
        design[list(emitter.active_frames), source] = 1.0 / scene.fps
    design = np.cumsum(design, axis=0)
    result = np.zeros(len(scene.emitters), np.float32)
    for group, fluid in enumerate(scene.fluid_ids):
        indices = [i for i, e in enumerate(scene.emitters) if e.fluid_id == fluid]
        area = scene.target_masks[:, group].sum(axis=(1, 2)).astype(float)
        result[indices] = nnls(design[:, indices], area)[0]
    return np.maximum(result, 10.0)


def lattice_rest_density(spacing: float, h: float) -> float:
    """Discrete kernel sum for an uncompressed square parcel lattice."""
    extent = math.ceil(h / spacing)
    axis = np.arange(-extent, extent + 1) * spacing
    x, y = np.meshgrid(axis, axis)
    q = np.minimum(np.sqrt(x * x + y * y) / h, 1.0)
    return float(
        (spacing**2 * 7 / (math.pi * h * h) * (1 - q) ** 4 * (1 + 4 * q)).sum()
    )


def emission_schedule(
    scene: FluidScene,
    rates: np.ndarray,
    area: float,
    substeps: int,
    capacities: np.ndarray,
    ages: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate continuous area flux; only emitted, equal-area parcels activate."""
    rates = np.atleast_2d(rates)
    if (
        area <= 0
        or substeps < 1
        or rates.shape[1] != len(scene.emitters)
        or len(capacities) != len(scene.emitters)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
    ):
        raise ValueError("invalid emission rates or parcel dimensions")
    birth = np.full(
        (len(rates), int(capacities.sum())), scene.frame_count * substeps + 1, np.int32
    )
    if ages is not None:
        if ages.shape != birth.shape:
            raise ValueError("birth ages must match the emission pool")
        ages.fill(0)
    offset = 0
    for source, emitter in enumerate(scene.emitters):
        steps = np.asarray(
            [f * substeps + s for f in emitter.active_frames for s in range(substeps)],
            np.int32,
        )
        for world, rate in enumerate(rates[:, source]):
            count = int(np.floor(len(steps) * rate / (scene.fps * substeps * area)))
            if count > capacities[source]:
                raise ValueError("emission pool too small for requested flux")
            if count:
                times = np.arange(1, count + 1) * area * scene.fps * substeps / rate
                indices = np.ceil(times - 1e-7).astype(int) - 1
                birth[world, offset : offset + count] = steps[indices]
                if ages is not None:
                    ages[world, offset : offset + count] = np.clip(
                        indices + 1 - times, 0, 1
                    ) / (scene.fps * substeps)
        offset += int(capacities[source])
    return birth


@wp.func
def _w(r: float, h: float) -> float:
    q = wp.clamp(r / h, 0.0, 1.0)
    s = 1.0 - q
    return 7.0 / (wp.pi * h * h) * s * s * s * s * (1.0 + 4.0 * q)


@wp.func
def _grad(d: wp.vec3, h: float) -> wp.vec3:
    q = wp.clamp(wp.length(d) / h, 0.0, 1.0)
    s = 1.0 - q
    return -140.0 / (wp.pi * h * h * h * h) * s * s * s * d


@wp.func
def _sample_sdf(
    p: wp.vec3, sdf: wp.array2d(dtype=float), width: int, height: int
) -> wp.vec3:
    x = wp.clamp(int(wp.floor(p[0])), 0, width - 2)
    y = wp.clamp(int(wp.floor(p[1])), 0, height - 2)
    tx = wp.clamp(p[0] - float(x), 0.0, 1.0)
    ty = wp.clamp(p[1] - float(y), 0.0, 1.0)
    a = sdf[y, x]
    b = sdf[y, x + 1]
    c = sdf[y + 1, x]
    d = sdf[y + 1, x + 1]
    value = (1.0 - ty) * ((1.0 - tx) * a + tx * b) + ty * ((1.0 - tx) * c + tx * d)
    return wp.vec3(
        value, (1.0 - ty) * (b - a) + ty * (d - c), (1.0 - tx) * (c - a) + tx * (d - b)
    )


@wp.func
def _wall(
    p: wp.vec3,
    sdf: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec2),
    radius: float,
    width: int,
    height: int,
) -> wp.vec3:
    p = wp.vec3(
        wp.clamp(p[0], radius, float(width - 1) - radius),
        wp.clamp(p[1], radius, float(height - 1) - radius),
        p[2],
    )
    for k in range(2):
        sample = _sample_sdf(p, sdf, width, height)
        if sample[0] < radius:
            g = wp.vec3(sample[1], sample[2], 0.0)
            if wp.dot(g, g) < 1e-6:
                x = wp.clamp(int(wp.round(p[0])), 0, width - 1)
                y = wp.clamp(int(wp.round(p[1])), 0, height - 1)
                fallback = gradient[y, x]
                g = wp.vec3(fallback[0], fallback[1], 0.0)
            # Newton projection handles the non-unit gradient of a pixel SDF.
            p += (radius - sample[0]) * g / wp.max(wp.dot(g, g), 1e-6)
    return p


@wp.kernel
def _reset(
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    n: int,
    h: float,
    height: int,
):
    i = wp.tid()
    # Reserve a cell outside the physical domain. An arbitrary huge coordinate
    # can hash onto a basin cell and make every fluid query scan inactive parcels.
    p[i] = wp.vec3(0.0, float(height) + 4.0 * h, float(i // n) * 4.0 * h)
    v[i] = wp.vec3()


@wp.kernel
def _predict(
    theta: wp.array2d(dtype=float),
    birth: wp.array2d(dtype=int),
    birth_age: wp.array2d(dtype=float),
    initial: wp.array2d(dtype=wp.vec2),
    source: wp.array(dtype=int),
    p: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    sdf: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec2),
    step: int,
    n: int,
    fluids: int,
    sources: int,
    dt: float,
    h: float,
    radius: float,
    width: int,
    height: int,
):
    i = wp.tid()
    world = i // n
    j = i % n
    if step < birth[world, j]:
        return
    x = p[i]
    vel = v[i]
    elapsed = dt
    if step == birth[world, j]:
        xy = initial[world, j]
        x = wp.vec3(xy[0], xy[1], float(world) * 4.0 * h)
        flux = theta[world, 3 + 2 * fluids + source[j]]
        nozzle_width = theta[world, 3 + 2 * fluids + sources + source[j]]
        vel = wp.vec3(0.0, flux / nozzle_width, 0.0)
        elapsed = birth_age[world, j]
    vel += wp.vec3(0.0, theta[world, 0] * elapsed, 0.0)
    # Preserve inlet velocity when displacement-to-velocity uses a full substep.
    old[i] = x - vel * (dt - elapsed)
    displacement = vel * elapsed
    steps = wp.clamp(int(wp.ceil(wp.length(displacement) / wp.max(radius, 0.5))), 1, 32)
    for k in range(steps):
        x = _wall(x + displacement / float(steps), sdf, gradient, radius, width, height)
    p[i] = x


@wp.kernel
def _lambda(
    grid: wp.uint64,
    theta: wp.array2d(dtype=float),
    p: wp.array(dtype=wp.vec3),
    birth: wp.array2d(dtype=int),
    group: wp.array(dtype=int),
    lam: wp.array(dtype=float),
    density: wp.array(dtype=float),
    step: int,
    n: int,
    fluids: int,
    area: float,
    h: float,
    rest_density: float,
):
    i = wp.tid()
    world = i // n
    if step < birth[world, i % n]:
        lam[i] = 0.0
        density[i] = 0.0
        return
    x = p[i]
    rho = float(0.0)
    denominator = float(0.0)
    gradient_i = wp.vec3()
    mass_i = theta[world, 3 + fluids + group[i % n]]
    query = wp.hash_grid_query(grid, x, h)
    for j in query:
        if j // n != world or step < birth[world, j % n]:
            continue
        d = x - p[j]
        rho += area / rest_density * _w(wp.length(d), h)
        g = area / rest_density * _grad(d, h)
        gradient_i += g
        if j != i:
            mass_ratio = mass_i / theta[world, 3 + fluids + group[j % n]]
            denominator += mass_ratio * wp.dot(g, g)
    # Store lambda / mass_i so regularization is invariant to common mass scaling.
    denominator += wp.dot(gradient_i, gradient_i)
    lam[i] = -wp.max(rho - 1.0, 0.0) / (denominator + 0.0001 / (h * h))
    density[i] = rho


@wp.kernel
def _project(
    grid: wp.uint64,
    theta: wp.array2d(dtype=float),
    p: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
    birth: wp.array2d(dtype=int),
    group: wp.array(dtype=int),
    lam: wp.array(dtype=float),
    sdf: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec2),
    step: int,
    n: int,
    fluids: int,
    area: float,
    h: float,
    rest_density: float,
    radius: float,
    width: int,
    height: int,
):
    i = wp.tid()
    world = i // n
    x = p[i]
    if step < birth[world, i % n]:
        out[i] = x
        return
    correction = wp.vec3()
    mass_i = theta[world, 3 + fluids + group[i % n]]
    query = wp.hash_grid_query(grid, x, h)
    for j in query:
        if j // n == world and step >= birth[world, j % n] and j != i:
            mass_ratio = theta[world, 3 + fluids + group[j % n]] / mass_i
            correction += (
                area / rest_density * (lam[i] + mass_ratio * lam[j]) * _grad(x - p[j], h)
            )
    length = wp.length(correction)
    if length > radius:
        correction *= radius / length
    out[i] = _wall(x + correction, sdf, gradient, radius, width, height)


@wp.kernel
def _velocity(
    p: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    birth: wp.array2d(dtype=int),
    step: int,
    n: int,
    dt: float,
):
    i = wp.tid()
    if step >= birth[i // n, i % n]:
        v[i] = (p[i] - old[i]) / dt


@wp.kernel
def _viscosity(
    grid: wp.uint64,
    theta: wp.array2d(dtype=float),
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
    birth: wp.array2d(dtype=int),
    group: wp.array(dtype=int),
    sdf: wp.array2d(dtype=float),
    gradient: wp.array2d(dtype=wp.vec2),
    step: int,
    n: int,
    fluids: int,
    dt: float,
    area: float,
    h: float,
    radius: float,
    width: int,
    height: int,
):
    i = wp.tid()
    world = i // n
    if step < birth[world, i % n]:
        out[i] = wp.vec3()
        return
    phase = group[i % n]
    density_i = theta[world, 3 + fluids + phase]
    mu_i = theta[world, 3 + phase]
    diffusion = wp.vec3()
    cohesion = wp.vec3()
    weight_sum = float(0.0)
    query = wp.hash_grid_query(grid, p[i], h)
    for j in query:
        if j // n != world or j == i or step < birth[world, j % n]:
            continue
        d = p[j] - p[i]
        distance = wp.length(d)
        phase_j = group[j % n]
        weight = area * _w(distance, h)
        density_j = theta[world, 3 + fluids + phase_j]
        mu = 0.5 * (mu_i + theta[world, 3 + phase_j])
        # Symmetric momentum exchange for unequal-density, equal-area parcels.
        weight *= 2.0 * density_j / (density_i + density_j)
        diffusion += mu * weight * (v[j] - v[i])
        weight_sum += mu * weight
        if phase == phase_j and distance > 0.001:
            cohesion += area * _w(distance, h) * d / h
    # A bounded explicit XSPH step. mu is an effective inverse-time rate.
    vel = v[i] + dt / wp.max(1.0, dt * weight_sum) * diffusion
    vel += dt * theta[world, 1] / density_i * cohesion
    sample = _sample_sdf(p[i], sdf, width, height)
    if sample[0] < radius + 0.75:
        normal = wp.normalize(wp.vec3(sample[1], sample[2], 0.0))
        vn = wp.dot(vel, normal)
        vel -= wp.min(vn, 0.0) * normal
        tangent = vel - wp.dot(vel, normal) * normal
        vel -= (1.0 - wp.exp(-theta[world, 2] * dt)) * tangent
    out[i] = vel


@wp.kernel
def _record(
    p: wp.array(dtype=wp.vec3),
    birth: wp.array2d(dtype=int),
    density: wp.array(dtype=float),
    prediction: wp.array3d(dtype=wp.vec2),
    diagnostic: wp.array3d(dtype=float),
    frame: int,
    step: int,
    n: int,
):
    i = wp.tid()
    w = i // n
    j = i % n
    prediction[w, frame, j] = wp.vec2(-1.0, -1.0)
    diagnostic[w, frame, j] = 0.0
    if step >= birth[w, j]:
        prediction[w, frame, j] = wp.vec2(p[i][0], p[i][1])
        diagnostic[w, frame, j] = density[i]


@wp.kernel
def _splat(
    p: wp.array(dtype=wp.vec3),
    birth: wp.array2d(dtype=int),
    group: wp.array(dtype=int),
    field: wp.array4d(dtype=float),
    step: int,
    n: int,
    area: float,
    h: float,
    scale: float,
    width: int,
    height: int,
):
    i = wp.tid()
    world = i // n
    if step < birth[world, i % n]:
        return
    x = p[i][0]
    y = p[i][1]
    lo_x = wp.max(0, int(wp.floor((x - h) / scale)))
    hi_x = wp.min(width - 1, int(wp.ceil((x + h) / scale)))
    lo_y = wp.max(0, int(wp.floor((y - h) / scale)))
    hi_y = wp.min(height - 1, int(wp.ceil((y + h) / scale)))
    for py in range(lo_y, hi_y + 1):
        for px in range(lo_x, hi_x + 1):
            d = wp.vec2((float(px) + 0.5) * scale - x, (float(py) + 0.5) * scale - y)
            wp.atomic_add(
                field, world, group[i % n], py, px, area * _w(wp.length(d), h)
            )


@wp.kernel
def _store_field(
    field: wp.array4d(dtype=float),
    history: wp.array4d(dtype=float),
    slot: int,
    width: int,
):
    world, fluid, y, x = wp.tid()
    history[world, slot, fluid, y * width + x] = field[world, fluid, y, x]


class BatchPBF:
    def __init__(self, scene: FluidScene, config: PBFConfig, device: str, *, capacity_flux=None):
        if not isinstance(config.graph_chunk_frames, int) or config.graph_chunk_frames < 0:
            raise ValueError('graph_chunk_frames must be a nonnegative integer')
        if (
            min(
                config.spacing,
                config.nozzle_width,
                config.substeps,
                config.iterations,
                config.population,
                config.grid_scale,
                config.loss_stride,
            )
            <= 0
        ):
            raise ValueError("PBF dimensions and iteration counts must be positive")
        if config.nozzle_width < config.spacing:
            raise ValueError("nozzle width must resolve at least one parcel")
        if config.loss_mode not in ("occupancy", "swd", "multiscale"):
            raise ValueError("unknown PBF loss mode")
        self.scene, self.config, self.device = scene, config, device
        self.area = config.spacing**2
        self.h = config.spacing * 2.2
        self.rest_density = lattice_rest_density(config.spacing, self.h)
        self.fluids, self.sources = len(scene.fluid_ids), len(scene.emitters)
        # Forward experiments may extend emission without fitting to synthetic targets.
        # This reference sizes storage only; rollout parameters still set actual flux.
        self.initial_flux = estimate_flux(scene) if capacity_flux is None else np.asarray(capacity_flux, np.float32)
        if self.initial_flux.shape != (self.sources,) or not np.isfinite(self.initial_flux).all() or np.any(self.initial_flux <= 0):
            raise ValueError("capacity_flux requires one positive finite flux per source")
        self.max_flux = self.initial_flux * 2.5
        self.capacities = np.asarray(
            [
                math.ceil(len(e.active_frames) / scene.fps * q / self.area) + 1
                for e, q in zip(scene.emitters, self.max_flux)
            ],
            np.int32,
        )
        self.n = int(self.capacities.sum())
        self.source_np = np.repeat(np.arange(self.sources), self.capacities).astype(
            np.int32
        )
        self.group_np = np.asarray(
            [scene.fluid_ids.index(scene.emitters[s].fluid_id) for s in self.source_np],
            np.int32,
        )
        self.initial = wp.zeros(
            (config.population, self.n), dtype=wp.vec2, device=device
        )
        self.source = wp.array(self.source_np, dtype=int, device=device)
        self.group = wp.array(self.group_np, dtype=int, device=device)
        self.theta = wp.zeros(
            (config.population, 3 + 2 * self.fluids + 2 * self.sources),
            dtype=float,
            device=device,
        )
        self.birth = wp.zeros((config.population, self.n), dtype=int, device=device)
        self.birth_age = wp.zeros(
            (config.population, self.n), dtype=float, device=device
        )
        total = config.population * self.n
        self.p = wp.empty(total, dtype=wp.vec3, device=device)
        self.old = wp.empty_like(self.p)
        self.temp = wp.empty_like(self.p)
        self.v = wp.empty_like(self.p)
        self.vtemp = wp.empty_like(self.p)
        self.lam = wp.empty(total, dtype=float, device=device)
        self.density = wp.empty_like(self.lam)
        mask = scene.collision_obstacle_mask
        sdf, gradient = _collision_sdf(scene.obstacle_mask if mask is None else mask)
        self.sdf = wp.array(sdf, dtype=float, device=device)
        self.gradient = wp.array(gradient, dtype=wp.vec2, device=device)
        self.prediction = wp.empty(
            (config.population, scene.frame_count, self.n), dtype=wp.vec2, device=device
        )
        self.diagnostic = wp.empty(
            (config.population, scene.frame_count, self.n), dtype=float, device=device
        )
        grid_dims = [
            math.ceil(scene.image_width / self.h) + 8,
            math.ceil(scene.image_height / self.h) + 8,
            config.population * 4 + 4,
        ]
        self.grid = wp.HashGrid(
            *(2 ** math.ceil(math.log2(d)) for d in grid_dims), device=device
        )
        self.grid.reserve(total)
        self.gh = math.ceil(scene.image_height / config.grid_scale)
        self.gw = math.ceil(scene.image_width / config.grid_scale)
        self.field = wp.zeros(
            (config.population, self.fluids, self.gh, self.gw),
            dtype=float,
            device=device,
        )
        self.frames = list(range(0, scene.frame_count, config.loss_stride))
        if self.frames[-1] != scene.frame_count - 1:
            self.frames.append(scene.frame_count - 1)
        self.field_history = wp.empty(
            (config.population, len(self.frames), self.fluids, self.gh * self.gw),
            dtype=float,
            device=device,
        )
        self.fields = wp.to_torch(self.field_history).view(
            config.population, len(self.frames), self.fluids, self.gh, self.gw
        )
        target, distances = [], []
        for frame in self.frames:
            masks, ds = [], []
            for phase in range(self.fluids):
                m = cv2.resize(
                    scene.target_masks[frame, phase].astype(np.float32),
                    (self.gw, self.gh),
                    interpolation=cv2.INTER_AREA,
                )
                masks.append(m)
                ds.append(
                    cv2.distanceTransform((m < 0.1).astype(np.uint8), cv2.DIST_L2, 5)
                    * config.grid_scale
                    if m.any()
                    else np.zeros_like(m)
                )
            target.append(masks)
            distances.append(ds)
        self.target = torch.as_tensor(np.asarray(target), device=device)
        self.distance = torch.as_tensor(np.asarray(distances), device=device)
        samples = np.zeros((len(self.frames), self.fluids, 64, 2), np.float32)
        present = np.zeros((len(self.frames), self.fluids), np.float32)
        for slot, frame in enumerate(self.frames):
            for phase in range(self.fluids):
                yy, xx = np.where(scene.target_masks[frame, phase])
                if len(xx):
                    indices = np.linspace(0, len(xx) - 1, 64).astype(int)
                    samples[slot, phase] = np.column_stack((xx[indices], yy[indices]))
                    present[slot, phase] = 1
        self.target_samples = torch.as_tensor(samples, device=device)
        self.target_present = torch.as_tensor(present, device=device)
        self.distribution_loss = (
            FluidDistributionLoss(
                scene,
                self.frames,
                device,
                config.swd_directions,
                config.swd_quantiles,
                config.stable_volume_weight,
                config.settled_distribution_weight,
                config.loss_mode,
            )
            if config.loss_mode in ("swd", "multiscale")
            else None
        )
        self.group_tensor = torch.as_tensor(self.group_np, device=device)
        self.loss_components = {}
        self.compute_loss_diagnostics = True
        self.graphs = {}
        self.graph_warmed = False
        self.frame_base = wp.zeros(1, dtype=int, device=device)
        slots = np.full(scene.frame_count, -1, np.int32)
        slots[self.frames] = np.arange(len(self.frames))
        self.field_slots = wp.array(slots, dtype=int, device=device)
        self.clock_kernels = None
        if config.graph_chunk_frames:
            from atw.polyworld.fluid_pbf_graph import clock_kernels
            self.clock_kernels = clock_kernels()
        wp.load_module(device=device)

    def _simulate(self, cutoff: int, *, reset=True, dynamic=False):
        c, s = self.config, self.scene
        dt = 1.0 / (s.fps * c.substeps)
        total = c.population * self.n
        p, temp = self.p, self.temp
        v, vtemp = self.v, self.vtemp

        def launch(kernel, dim, inputs, *, device):
            if dynamic:
                kernel = getattr(self.clock_kernels, kernel.key)
                inputs = [*inputs, self.frame_base, c.substeps, self.field_slots]
            wp.launch(kernel, dim, inputs, device=device)

        if reset:
            wp.launch(_reset, total, [self.p, self.v, self.n, self.h, s.image_height], device=self.device)
        for frame in range(cutoff + 1):
            for sub in range(c.substeps):
                step = frame * c.substeps + sub
                launch(
                    _predict,
                    total,
                    [
                        self.theta,
                        self.birth,
                        self.birth_age,
                        self.initial,
                        self.source,
                        p,
                        self.old,
                        v,
                        self.sdf,
                        self.gradient,
                        step,
                        self.n,
                        self.fluids,
                        self.sources,
                        dt,
                        self.h,
                        c.spacing * 0.5,
                        s.image_width,
                        s.image_height,
                    ],
                    device=self.device,
                )
                for _ in range(c.iterations):
                    self.grid.build(p, self.h)
                    launch(
                        _lambda,
                        total,
                        [
                            self.grid.id,
                            self.theta,
                            p,
                            self.birth,
                            self.group,
                            self.lam,
                            self.density,
                            step,
                            self.n,
                            self.fluids,
                            self.area,
                            self.h,
                            self.rest_density,
                        ],
                        device=self.device,
                    )
                    launch(
                        _project,
                        total,
                        [
                            self.grid.id,
                            self.theta,
                            p,
                            temp,
                            self.birth,
                            self.group,
                            self.lam,
                            self.sdf,
                            self.gradient,
                            step,
                            self.n,
                            self.fluids,
                            self.area,
                            self.h,
                            self.rest_density,
                            c.spacing * 0.5,
                            s.image_width,
                            s.image_height,
                        ],
                        device=self.device,
                    )
                    if c.buffer_pingpong:
                        p, temp = temp, p
                    else:
                        wp.copy(p, temp)
                self.grid.build(p, self.h)
                launch(
                    _lambda,
                    total,
                    [
                        self.grid.id,
                        self.theta,
                        p,
                        self.birth,
                        self.group,
                        self.lam,
                        self.density,
                        step,
                        self.n,
                        self.fluids,
                        self.area,
                        self.h,
                        self.rest_density,
                    ],
                    device=self.device,
                )
                launch(
                    _velocity,
                    total,
                    [p, self.old, v, self.birth, step, self.n, dt],
                    device=self.device,
                )
                launch(
                    _viscosity,
                    total,
                    [
                        self.grid.id,
                        self.theta,
                        p,
                        v,
                        vtemp,
                        self.birth,
                        self.group,
                        self.sdf,
                        self.gradient,
                        step,
                        self.n,
                        self.fluids,
                        dt,
                        self.area,
                        self.h,
                        c.spacing * 0.5,
                        s.image_width,
                        s.image_height,
                    ],
                    device=self.device,
                )
                if c.buffer_pingpong:
                    v, vtemp = vtemp, v
                else:
                    wp.copy(v, vtemp)
            launch(
                _record,
                total,
                [
                    p,
                    self.birth,
                    self.density,
                    self.prediction,
                    self.diagnostic,
                    frame,
                    step,
                    self.n,
                ],
                device=self.device,
            )
            if (dynamic or frame in self.frames) and self.distribution_loss is None:
                self.field.zero_()
                launch(
                    _splat,
                    total,
                    [
                        p,
                        self.birth,
                        self.group,
                        self.field,
                        step,
                        self.n,
                        self.area,
                        self.h,
                        float(c.grid_scale),
                        self.gw,
                        self.gh,
                    ],
                    device=self.device,
                )
                launch(
                    _store_field,
                    (c.population, self.fluids, self.gh, self.gw),
                    [self.field, self.field_history, frame if dynamic else self.frames.index(frame), self.gw],
                    device=self.device,
                )

        if p is not self.p:
            wp.copy(self.p, p)
        if v is not self.v:
            wp.copy(self.v, v)
        if dynamic:
            wp.launch(self.clock_kernels.advance_frame, 1,
                      [self.frame_base, cutoff + 1], device=self.device)

    def _launch_chunked(self, cutoff):
        def segment(length):
            if length == 0:
                wp.launch(self.clock_kernels.set_frame, 1, [self.frame_base, 0], device=self.device)
                wp.launch(_reset, self.config.population * self.n,
                          [self.p, self.v, self.n, self.h, self.scene.image_height], device=self.device)
            else:
                self._simulate(length - 1, reset=False, dynamic=True)

        if self.device.startswith('cuda') and not self.graph_warmed:
            self._simulate(min(1, cutoff))
            wp.synchronize_device(self.device)
            wp.load_module(self.clock_kernels, device=self.device, block_dim=256)
            self.graph_warmed = True
        lengths = [0] + [min(self.config.graph_chunk_frames, cutoff + 1 - start)
                         for start in range(0, cutoff + 1, self.config.graph_chunk_frames)]
        for length in lengths:
            if self.device.startswith('cuda'):
                key = ('chunk', length)
                if key not in self.graphs:
                    with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
                        segment(length)
                    self.graphs[key] = capture.graph
                wp.capture_launch(self.graphs[key])
            else:
                segment(length)

    def resolved_parameters(self, parameters: np.ndarray) -> np.ndarray:
        params = np.asarray(parameters, np.float32).copy()
        if params.shape[-1] != 3 + 2 * self.fluids + 2 * self.sources:
            raise ValueError("PBF v2 parameters require flux and nozzle width")
        # Whole transverse cells preserve parcel area and an uncompressed inlet.
        params[..., -self.sources :] = (
            np.maximum(1, np.rint(params[..., -self.sources :] / self.config.spacing))
            * self.config.spacing
        )
        return params

    def rollout(self, parameters: np.ndarray, cutoff: int | None = None):
        cutoff = self.scene.frame_count - 1 if cutoff is None else cutoff
        params = np.atleast_2d(self.resolved_parameters(parameters))
        if len(params) not in (1, self.config.population):
            raise ValueError("use one candidate or a full population")
        if len(params) == 1:
            params = np.repeat(params, self.config.population, axis=0)
        offset = 3 + 2 * self.fluids
        initial = np.zeros((self.config.population, self.n, 2), np.float32)
        start = 0
        for source, (emitter, capacity) in enumerate(
            zip(self.scene.emitters, self.capacities)
        ):
            columns = np.rint(
                params[:, offset + self.sources + source] / self.config.spacing
            ).astype(int)
            index = np.arange(capacity)[None]
            initial[:, start : start + capacity, 0] = (
                emitter.xy[0]
                + (index % columns[:, None] - (columns[:, None] - 1) / 2)
                * self.config.spacing
            )
            initial[:, start : start + capacity, 1] = emitter.xy[1]
            start += capacity
        ages = np.zeros((self.config.population, self.n), np.float32)
        births = emission_schedule(
            self.scene,
            params[:, offset : offset + self.sources],
            self.area,
            self.config.substeps,
            self.capacities,
            ages,
        )
        wp.copy(self.theta, wp.array(params, dtype=float, device=self.device))
        wp.copy(self.birth, wp.array(births, dtype=int, device=self.device))
        wp.copy(self.birth_age, wp.array(ages, dtype=float, device=self.device))
        wp.copy(self.initial, wp.array(initial, dtype=wp.vec2, device=self.device))
        if self.config.graph_chunk_frames:
            self._launch_chunked(cutoff)
        elif self.device.startswith("cuda"):
            if cutoff not in self.graphs:
                # Warm up lazy grid kernels before capturing the complete rollout.
                self._simulate(min(1, cutoff))
                wp.synchronize_device(self.device)
                with wp.ScopedCapture(device=self.device) as capture:
                    self._simulate(cutoff)
                self.graphs[cutoff] = capture.graph
            wp.capture_launch(self.graphs[cutoff])
        else:
            self._simulate(cutoff)
        wp.synchronize_device(self.device)
        length = sum(f <= cutoff for f in self.frames)
        if self.distribution_loss is not None:
            score, self.loss_components = self.distribution_loss(
                wp.to_torch(self.prediction)[:, self.frames[:length]],
                self.group_tensor,
                wp.to_torch(self.birth),
                self.area,
                self.config.substeps,
                diagnostics=self.compute_loss_diagnostics,
            )
            return torch.nan_to_num(score, nan=1e8, posinf=1e8).cpu().numpy(), births
        fields = self.fields[:, :length]
        target = self.target[:length]
        norm = target.sum((-2, -1)).clamp_min(4.0)
        occupancy = fields.clamp(0, 1)
        shape = ((occupancy - target) ** 2).sum((-2, -1)) / norm
        transport = (fields * (self.distance[:length] / 20.0) ** 2).sum((-2, -1)) / norm
        area = ((fields.sum((-2, -1)) - target.sum((-2, -1))) / norm) ** 2
        score = (shape + transport + 0.5 * area).mean((1, 2))
        # Reverse distance prevents a spatially wrong, small puddle from winning
        # merely because its total area is right and occupancy gradients saturate.
        positions = wp.to_torch(self.prediction)[:, self.frames[:length]]
        reverse = torch.zeros(len(params), device=self.device)
        for phase in range(self.fluids):
            points = positions[:, :, self.group_np == phase]
            distances = torch.cdist(self.target_samples[:length, phase][None], points)
            distances.masked_fill_((points[..., 0] < 0).unsqueeze(-2), 1e4)
            nearest = distances.min(dim=-1).values.clamp_max(200.0)
            reverse += (
                (nearest / 20.0) ** 2
                * self.target_present[:length, phase][None, :, None]
            ).sum((1, 2))
        score += reverse / (self.target_present[:length].sum().clamp_min(1) * 64)
        score = torch.nan_to_num(score, nan=1e8, posinf=1e8, neginf=1e8)
        return score.cpu().numpy(), births


def parameter_bounds(engine: BatchPBF):
    f, s = engine.fluids, engine.sources
    # g, cohesion, wall drag, per-fluid viscosity and relative density, flux, width.
    lower = np.r_[
        40.0,
        0.0,
        0.0,
        np.full(f, 0.0),
        np.full(f, 0.4),
        engine.initial_flux * 0.35,
        np.full(s, engine.config.spacing),
    ].astype(np.float32)
    upper = np.r_[
        600.0,
        1500.0,
        25.0,
        np.full(f, 40.0),
        np.full(f, 3.0),
        engine.max_flux,
        np.full(s, engine.config.nozzle_width * 2),
    ].astype(np.float32)
    mean = np.r_[
        200.0,
        150.0,
        3.0,
        np.full(f, 5.0),
        np.ones(f),
        engine.initial_flux,
        np.full(s, engine.config.nozzle_width),
    ].astype(np.float32)
    # Only density ratios are identifiable from this experiment.
    lower[3 + f] = upper[3 + f] = mean[3 + f] = 1.0
    return lower, upper, mean


def fit_pbf(
    scene: FluidScene,
    config: PBFConfig,
    device: str,
    warm_parameters: dict | None = None,
    warm_vector: np.ndarray | None = None,
    verbose: bool = True,
    *,
    defer_loss_diagnostics: bool = False,
    engine_factory=None,
    bounds_fn=None,
) -> dict:
    if config.population < 4 or config.generations < 1 or config.restarts < 1:
        raise ValueError("PBF CEM requires population >= 4 and positive iterations")
    start = time.perf_counter()
    engine = (engine_factory or BatchPBF)(scene, config, device)
    engine.compute_loss_diagnostics = not defer_loss_diagnostics
    lower, upper, initial = (bounds_fn or parameter_bounds)(engine)
    if warm_parameters:
        initial[0] = warm_parameters.get("gravity_pixels_per_s2", initial[0])
    if warm_vector is not None:
        if warm_vector.shape != initial.shape or not np.isfinite(warm_vector).all():
            raise ValueError("warm-start parameter vector is incompatible")
        initial = np.clip(warm_vector, lower, upper).astype(np.float32)
    global_score, global_best = math.inf, initial.copy()
    trace = []
    std_floor = 0.035 * (upper - lower)
    # Width is discrete: retain enough variance to cross a parcel-cell boundary.
    std_floor[-engine.sources :] = 0.5 * config.spacing
    elites = max(2, config.population // 4)
    for restart in range(config.restarts):
        cutoffs = _restart_cutoffs(scene.frame_count, restart, config.restart_full_horizon)
        rng = np.random.default_rng(config.seed + restart * 11)
        mean = initial.copy() if restart == 0 else global_best.copy()
        std = 0.22 * (upper - lower)
        best = mean.copy()
        for cutoff in cutoffs:
            # Early free-flight frames scarcely identify relative densities.
            # Reopen their search when the temporal horizon grows to settling.
            density_slice = slice(3 + engine.fluids, 3 + 2 * engine.fluids)
            std[density_slice] = 0.22 * (upper - lower)[density_slice]
            best_score = math.inf
            for generation in range(config.generations):
                batch = np.clip(
                    mean + rng.standard_normal((config.population, len(mean))) * std,
                    lower,
                    upper,
                ).astype(np.float32)
                batch[0] = best
                scores, _ = engine.rollout(batch, cutoff)
                order = np.argsort(scores)[:elites]
                selected = batch[order]
                mean = 0.25 * mean + 0.75 * selected.mean(axis=0)
                std = np.maximum(0.25 * std + 0.75 * selected.std(axis=0), std_floor)
                if scores[order[0]] < best_score:
                    best_score, best = float(scores[order[0]]), selected[0].copy()
            mean = best.copy()
            item = dict(
                restart=restart,
                end_frame=cutoff,
                loss=best_score,
                seconds=time.perf_counter() - start,
            )
            trace.append(item)
            if verbose:
                print(item, flush=True)
        if best_score < global_score:
            global_score, global_best = best_score, best.copy()
    global_best = engine.resolved_parameters(global_best)
    engine.compute_loss_diagnostics = True
    _, birth = engine.rollout(global_best)
    wp.synchronize_device(device)
    fit_seconds = time.perf_counter() - start
    replay_start = time.perf_counter()
    engine.rollout(global_best)
    batch_replay_seconds = time.perf_counter() - replay_start
    predicted = wp.to_torch(engine.prediction)[0].cpu().numpy().copy()
    densities = wp.to_torch(engine.diagnostic)[0].cpu().numpy().copy()
    selected_births = birth[0] // config.substeps
    active = selected_births < scene.frame_count
    fitted_scene = replace(
        scene,
        particle_xy=engine.initial.numpy()[0, active],
        particle_birth=selected_births[active],
        particle_fluid=engine.group_np[active],
        particle_emitter=engine.source_np[active],
    )
    predicted = predicted[:, active]
    densities = densities[:, active]
    chamfer = fluid_chamfer_rmse(fitted_scene, predicted)
    d = densities[densities > 0]
    return dict(
        parameters=global_best,
        predicted=predicted,
        scene=fitted_scene,
        birth_substeps=birth[0, active],
        birth_age_seconds=engine.birth_age.numpy()[0, active],
        densities=densities,
        mean_loss=global_score,
        loss_details=(
            {
                **engine.distribution_loss.metadata(),
                "components": {
                    k: (
                        None
                        if k == "stable_quantity_relative_error"
                        and not engine.distribution_loss.windows
                        else float(v[0].item())
                    )
                    for k, v in engine.loss_components.items()
                },
            }
            if engine.distribution_loss is not None
            else {"mode": "occupancy"}
        ),
        trace=trace,
        elapsed_seconds=fit_seconds,
        batch_replay_seconds=batch_replay_seconds,
        particle_area=engine.area,
        smoothing_length=engine.h,
        lattice_rest_density=engine.rest_density,
        initial_flux=engine.initial_flux,
        chamfer_pixels=chamfer,
        density_compression_rmse=float(np.sqrt(np.mean(np.maximum(d - 1, 0) ** 2))),
        density_p99=float(np.quantile(d, 0.99)),
        mean_chamfer_pixels=float(np.mean(list(chamfer.values()))),
    )


def density_images(
    positions: np.ndarray,
    groups: np.ndarray,
    fluid_count: int,
    height: int,
    width: int,
    area: float,
    h: float,
) -> np.ndarray:
    """Same finite-area kernel as the optimizer, sampled at full pixel resolution."""
    result = np.zeros((fluid_count, height, width), np.float32)
    for point, group in zip(positions, groups):
        if not np.isfinite(point).all() or np.any(point < 0):
            continue
        x, y = point
        x0, x1 = max(0, math.floor(x - h)), min(width, math.ceil(x + h) + 1)
        y0, y1 = max(0, math.floor(y - h)), min(height, math.ceil(y + h) + 1)
        xx, yy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        q = np.minimum(np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / h, 1)
        result[group, y0:y1, x0:x1] += (
            area * 7 / (math.pi * h * h) * (1 - q) ** 4 * (1 + 4 * q)
        )
    return result


def coverage_metrics(
    scene: FluidScene,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    area: float | None = None,
    h: float | None = None,
    marker_radius: int = 2,
) -> dict:
    """Full-resolution mask comparison for particle fields and disks."""
    intersection = np.zeros(len(scene.fluid_ids))
    union = np.zeros_like(intersection)
    area_error = np.zeros_like(intersection)
    observed_area = np.zeros_like(intersection)
    for frame, points in enumerate(predicted):
        if area is not None:
            density = density_images(
                points,
                groups,
                len(scene.fluid_ids),
                scene.image_height,
                scene.image_width,
                area,
                h,
            )
            simulated = density >= 0.3
        else:
            simulated = np.zeros(
                (len(scene.fluid_ids), scene.image_height, scene.image_width), np.uint8
            )
            for point, group in zip(points, groups):
                if np.isfinite(point).all() and np.all(point >= 0):
                    cv2.circle(
                        simulated[group],
                        tuple(np.rint(point).astype(int)),
                        marker_radius,
                        1,
                        -1,
                    )
            simulated = simulated.astype(bool)
        target = scene.target_masks[frame]
        intersection += (simulated & target).sum((1, 2))
        union += (simulated | target).sum((1, 2))
        area_error += np.abs(simulated.sum((1, 2)).astype(float) - target.sum((1, 2)))
        observed_area += target.sum((1, 2))
    iou = intersection / np.maximum(union, 1)
    relative_error = area_error / np.maximum(observed_area, 1)
    return {
        "mask_iou": dict(zip(scene.fluid_ids, iou.tolist())),
        "area_relative_l1": dict(zip(scene.fluid_ids, relative_error.tolist())),
        "mean_mask_iou": float(iou.mean()),
        "mean_area_relative_l1": float(relative_error.mean()),
    }
