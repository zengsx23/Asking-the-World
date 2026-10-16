"""Open-loop 2D particle-fluid fitting from PhysMind pixel-segmentation masks."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import warp as wp
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FluidEmitter:
    object_id: str
    fluid_id: str
    xy: tuple[float, float]
    active_frames: tuple[int, ...]


@dataclass(frozen=True)
class FluidScene:
    scene_dir: Path
    source_video: str
    fps: float
    image_width: int
    image_height: int
    fluid_ids: tuple[str, ...]
    target_masks: np.ndarray  # [frame, fluid, y, x]
    obstacle_mask: np.ndarray
    emitters: tuple[FluidEmitter, ...]
    particle_xy: np.ndarray
    particle_birth: np.ndarray
    particle_fluid: np.ndarray
    emission_stride: int
    particle_emitter: np.ndarray | None = None
    collision_obstacle_mask: np.ndarray | None = None
    basin_bounds: dict[str, tuple[int, int, int, int]] | None = None
    boundary_source: str | None = None

    @property
    def frame_count(self) -> int:
        return self.target_masks.shape[0]


def _source_object(item: dict) -> bool:
    if "is_origin" in item:
        return item["is_origin"] is True
    text = " ".join(
        str(item.get(key, "")) for key in ("target_id", "category", "role")
    ).lower()
    return any(token in text for token in ("emitter", "dispenser", "spawner", "source"))


def _color_token(value: str) -> str:
    words = value.lower().replace("-", "_").split("_")
    aliases = {
        "pale": "",
        "light": "",
        "grey": "white",
        "gray": "white",
        "mint": "teal",
        "cyan": "teal",
    }
    for word in words:
        word = aliases.get(word, word)
        if word in {
            "pink",
            "yellow",
            "blue",
            "white",
            "orange",
            "green",
            "teal",
            "red",
        }:
            return word
    return ""


def _enclosed_boundary_domain(wall: np.ndarray) -> np.ndarray:
    """Return the largest free region enclosed by a highest-priority wall mask."""

    wall = np.asarray(wall, dtype=bool)
    if wall.ndim != 2 or not wall.any():
        raise ValueError("A present fluid boundary needs a nonempty 2D mask")
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (~wall).astype(np.uint8), 8
    )
    border_labels = set(np.unique(np.concatenate((
        components[0], components[-1], components[:, 0], components[:, -1]
    ))).tolist())
    enclosed = [
        label for label in range(1, count) if label not in border_labels
    ]
    if not enclosed:
        raise ValueError("The fluid boundary mask does not enclose an interior domain")
    label = max(enclosed, key=lambda item: int(stats[item, cv2.CC_STAT_AREA]))
    return components == label


def _source_assignment(
    masks: np.ndarray,
    source_id: int,
    fluid_indices: tuple[int, ...],
    bounds: tuple[int, int, int, int],
    names: dict[int, str],
) -> int:
    x0, y0, x1, y1 = bounds
    cx = (x0 + x1) // 2
    roi = masks[:, y1 + 1 : y1 + 12, max(0, cx - 8) : cx + 9]
    scores = [
        int(np.count_nonzero(roi[: min(40, len(masks))] == index))
        for index in fluid_indices
    ]
    if not max(scores, default=0):
        scores = [int(np.count_nonzero(roi == index)) for index in fluid_indices]
    if max(scores, default=0) > 0:
        return fluid_indices[int(np.argmax(scores))]
    source_color = _color_token(names[source_id])
    for index in fluid_indices:
        if source_color and source_color == _color_token(names[index]):
            return index
    raise ValueError(f"cannot assign fluid to source {names[source_id]}")


def load_fluid_scene(
    scene_dir: Path, *, emission_stride: int = 1, emission_layout: str = "centerline",
    boundary_dir: Path | None = None,
) -> FluidScene:
    if emission_stride < 1:
        raise ValueError("emission stride must be positive")
    if emission_layout not in {"centerline", "parallel"}:
        raise ValueError("unknown emission layout")
    scene_dir = Path(scene_dir)
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    if manifest.get("status") != "succeeded":
        raise ValueError(f"segmentation is not succeeded: {scene_dir}")
    index = {
        int(key): value
        for key, value in json.loads(
            (scene_dir / "final/object_index.json").read_text()
        ).items()
    }
    names = {key: value["target_id"] for key, value in index.items()}
    fluid_indices = tuple(
        key
        for key, value in index.items()
        if value.get("kind") == "appearance_collection"
        or (value.get("kind") == "dynamic_instance" and (
            (value.get("grouping") == "collection" and manifest.get("prompt_branch") == "fluid")
            or "fluid" in value.get("category", "").lower().split()))
    )
    source_indices = tuple(
        key
        for key, value in index.items()
        if value.get("kind") == "persistent_instance" and _source_object(value)
    )
    if not fluid_indices or not source_indices:
        raise ValueError("fluid masks and fixed emitters are required")
    mask_file = scene_dir / "final/masks/identities.npz"
    with np.load(mask_file) as archive:
        frames = manifest["mask_frames"]
        masks = np.stack([archive[f"frame_{frame:06d}"] for frame in range(frames)])
    height, width = masks.shape[1:]
    boundary_source = None
    boundary_collision = np.zeros((height, width), bool)
    if boundary_dir is not None:
        boundary_dir = Path(boundary_dir).resolve()
        boundary = json.loads((boundary_dir / "boundary.json").read_text())
        if boundary["status"] == "present":
            with np.load(boundary_dir / "boundary_mask.npz") as archive:
                wall = cv2.resize((archive["labels"] != 0).astype(np.uint8),
                                  (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            interior = _enclosed_boundary_domain(wall)
            # Boundary ownership is absolute. Remove every foreground identity
            # from both the wall and exterior before deriving any SWR inputs.
            masks = masks.copy()
            masks[:, ~interior] = 0
            # Treat the complete exterior as solid as well as the visible wall,
            # so numerical particles cannot tunnel through and continue outside.
            boundary_collision = ~interior
            boundary_source = str(boundary_dir)
    obstacle_mask = (
        np.isin(masks[0], tuple(index.keys()))
        & ~np.isin(masks[0], fluid_indices)
    ) | boundary_collision
    collision_obstacle_mask = (
        obstacle_mask & ~np.isin(masks[0], source_indices)
    ) | boundary_collision
    target_masks = np.stack([(masks == key) for key in fluid_indices], axis=1)
    basin_bounds = {}
    for key, item in index.items():
        if item.get("category") not in {
            "basin",
            "bin",
            "container",
            "cup",
        } and item.get("role") not in {"sink", "collector", "receiver"}:
            continue
        y, x = np.where(masks[0] == key)
        if len(x):
            basin_bounds[names[key]] = (
                int(x.min()),
                int(y.min()),
                int(x.max()) + 1,
                int(y.max()) + 1,
            )
    emitters: list[FluidEmitter] = []
    particles: list[tuple[float, float]] = []
    births: list[int] = []
    groups: list[int] = []
    particle_emitters: list[int] = []
    for source_id in source_indices:
        yy, xx = np.where(masks[0] == source_id)
        if not len(xx):
            continue
        bounds = int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())
        bound = [key for key in fluid_indices if index[key].get("origin_id") == names[source_id]]
        if len(bound) == 1:
            fluid_index = bound[0]
        elif any(index[key].get("kind") == "dynamic_instance" for key in fluid_indices):
            raise ValueError(f"Expected one explicit fluid identity for origin {names[source_id]}")
        else:
            fluid_index = _source_assignment(masks, source_id, fluid_indices, bounds, names)
        group = fluid_indices.index(fluid_index)
        x0, _y0, x1, y1 = bounds
        cx = 0.5 * (x0 + x1)
        half_width = min(2.0, 0.2 * (x1 - x0))
        fluid_mask = target_masks[:, group]
        visible = np.any(
            fluid_mask[:, y1 + 1 : y1 + 9, max(0, int(cx) - 7) : int(cx) + 8],
            axis=(1, 2),
        )
        visible = binary_closing(visible, structure=np.ones(3), border_value=0)
        active = tuple(int(frame) for frame in np.flatnonzero(visible))
        if not active:
            continue
        emitter_number = len(emitters)
        emitters.append(
            FluidEmitter(names[source_id], names[fluid_index], (cx, y1 + 2.0), active)
        )
        for frame in active:
            if frame % emission_stride:
                continue
            positions = (
                ((cx, y1 + 2.0), (cx, y1 + 5.0))
                if emission_layout == "centerline"
                else ((cx - half_width, y1 + 2.0), (cx + half_width, y1 + 2.0))
            )
            for position in positions:
                particles.append(position)
                births.append(frame)
                groups.append(group)
                particle_emitters.append(emitter_number)
    if not particles:
        raise ValueError("no active fluid sources found")
    info = manifest["video_info"]
    return FluidScene(
        scene_dir=scene_dir,
        source_video=manifest["video"],
        fps=float(info["fps"]),
        image_width=width,
        image_height=height,
        fluid_ids=tuple(names[key] for key in fluid_indices),
        target_masks=target_masks,
        obstacle_mask=obstacle_mask,
        emitters=tuple(emitters),
        particle_xy=np.asarray(particles, dtype=np.float32),
        particle_birth=np.asarray(births, dtype=np.int32),
        particle_fluid=np.asarray(groups, dtype=np.int32),
        emission_stride=emission_stride,
        particle_emitter=np.asarray(particle_emitters, dtype=np.int32),
        collision_obstacle_mask=collision_obstacle_mask,
        basin_bounds=basin_bounds,
        boundary_source=boundary_source,
    )


def source_stream_masks(
    scene: FluidScene, grid_height: int, grid_width: int
) -> np.ndarray:
    """Split only the observable upper streams; a pooled color has no source ID."""
    result = np.zeros(
        (scene.frame_count, len(scene.emitters), grid_height, grid_width), dtype=bool
    )
    xx = np.arange(grid_width, dtype=np.float32)[None, :]
    yy = np.arange(grid_height, dtype=np.float32)[:, None]
    scale_x = scene.image_width / grid_width
    scale_y = scene.image_height / grid_height
    for source, emitter in enumerate(scene.emitters):
        same_color = [
            other.xy[0]
            for other in scene.emitters
            if other.fluid_id == emitter.fluid_id and other is not emitter
        ]
        cx, cy = emitter.xy
        local = (np.abs(xx * scale_x - cx) <= 36.0) & (
            (yy * scale_y >= cy - 2.0)
            & (yy * scale_y < min(cy + 50.0, 0.45 * scene.image_height))
        )
        for other_x in same_color:
            local &= np.abs(xx * scale_x - cx) <= np.abs(xx * scale_x - other_x)
        fluid = scene.fluid_ids.index(emitter.fluid_id)
        for frame in emitter.active_frames:
            resized = cv2.resize(
                scene.target_masks[frame, fluid].astype(np.uint8),
                (grid_width, grid_height),
                interpolation=cv2.INTER_AREA,
            )
            result[frame, source] = (resized > 0) & local
    return result


@wp.func
def _grid_value(
    grid: wp.array3d(dtype=float), world: int, y: int, x: int, height: int, width: int
) -> float:
    return grid[world, wp.clamp(y, 0, height - 1), wp.clamp(x, 0, width - 1)]


@wp.func
def _jet_lateral_scale(p: wp.vec2, origin: wp.vec2, jet_length: float) -> float:
    if jet_length <= 0.0:
        return 1.0
    return wp.clamp((p[1] - origin[1]) / jet_length, 0.0, 1.0)


@wp.kernel
def _reset_particles(
    initial: wp.array(dtype=wp.vec2),
    position: wp.array2d(dtype=wp.vec2),
    velocity: wp.array2d(dtype=wp.vec2),
):
    world, particle = wp.tid()
    position[world, particle] = initial[particle]
    velocity[world, particle] = wp.vec2()


@wp.kernel
def _clear_density(
    total: wp.array3d(dtype=float),
    per_fluid: wp.array4d(dtype=float),
    fluid_count: int,
):
    world, y, x = wp.tid()
    total[world, y, x] = 0.0
    for fluid in range(fluid_count):
        per_fluid[world, fluid, y, x] = 0.0


@wp.kernel
def _deposit_particles(
    theta: wp.array2d(dtype=float),
    position: wp.array2d(dtype=wp.vec2),
    birth: wp.array(dtype=int),
    group: wp.array(dtype=int),
    source: wp.array(dtype=int),
    total: wp.array3d(dtype=float),
    per_fluid: wp.array4d(dtype=float),
    frame: int,
    source_count: int,
    grid_scale: float,
    grid_height: int,
    grid_width: int,
):
    world, particle = wp.tid()
    if frame < birth[particle]:
        return
    p = position[world, particle] / grid_scale
    x = wp.clamp(int(wp.floor(p[0])), 0, grid_width - 2)
    y = wp.clamp(int(wp.floor(p[1])), 0, grid_height - 2)
    tx = wp.clamp(p[0] - float(x), 0.0, 1.0)
    ty = wp.clamp(p[1] - float(y), 0.0, 1.0)
    fluid = group[particle]
    parcel_area = 1.0
    if source_count > 0:
        parcel_area = theta[world, 7 + source[particle]]
    for dy in range(2):
        for dx in range(2):
            weight = (
                parcel_area
                * (tx if dx == 1 else 1.0 - tx)
                * (ty if dy == 1 else 1.0 - ty)
            )
            wp.atomic_add(total, world, y + dy, x + dx, weight)
            wp.atomic_add(per_fluid, world, fluid, y + dy, x + dx, weight)


@wp.kernel
def _advance_particles(
    theta: wp.array2d(dtype=float),
    initial: wp.array(dtype=wp.vec2),
    birth: wp.array(dtype=int),
    position: wp.array2d(dtype=wp.vec2),
    velocity: wp.array2d(dtype=wp.vec2),
    total: wp.array3d(dtype=float),
    sdf: wp.array2d(dtype=float),
    sdf_gradient: wp.array2d(dtype=wp.vec2),
    frame: int,
    substep: int,
    dt: float,
    grid_scale: float,
    grid_height: int,
    grid_width: int,
    image_height: int,
    image_width: int,
    jet_length: float,
):
    world, particle = wp.tid()
    if frame < birth[particle]:
        return
    radius = theta[world, 6]
    p = position[world, particle]
    v = velocity[world, particle]
    if frame == birth[particle] and substep == 0:
        p = initial[particle]
        v = wp.vec2(0.0, theta[world, 5])
    q = p / grid_scale
    x = wp.clamp(int(wp.floor(q[0])), 0, grid_width - 2)
    y = wp.clamp(int(wp.floor(q[1])), 0, grid_height - 2)
    tx = wp.clamp(q[0] - float(x), 0.0, 1.0)
    ty = wp.clamp(q[1] - float(y), 0.0, 1.0)
    gx = (
        (1.0 - ty)
        * (
            _grid_value(total, world, y, x + 1, grid_height, grid_width)
            - _grid_value(total, world, y, x, grid_height, grid_width)
        )
        + ty
        * (
            _grid_value(total, world, y + 1, x + 1, grid_height, grid_width)
            - _grid_value(total, world, y + 1, x, grid_height, grid_width)
        )
    ) / grid_scale
    gy = (
        (1.0 - tx)
        * (
            _grid_value(total, world, y + 1, x, grid_height, grid_width)
            - _grid_value(total, world, y, x, grid_height, grid_width)
        )
        + tx
        * (
            _grid_value(total, world, y + 1, x + 1, grid_height, grid_width)
            - _grid_value(total, world, y, x + 1, grid_height, grid_width)
        )
    ) / grid_scale
    gx -= ((1.0 - ty) * (1.0 - ty) + ty * ty) * (2.0 * tx - 1.0) / grid_scale
    gy -= ((1.0 - tx) * (1.0 - tx) + tx * tx) * (2.0 * ty - 1.0) / grid_scale
    lateral_pressure = theta[world, 2] * _jet_lateral_scale(
        p, initial[particle], jet_length
    )
    acceleration = wp.vec2(
        -lateral_pressure * gx, theta[world, 0] - theta[world, 2] * gy
    )
    v = (v + acceleration * dt) * wp.exp(-theta[world, 1] * dt)
    start = p
    p = p + v * dt
    for _contact in range(2):
        displacement = p - start
        steps = wp.clamp(
            int(wp.ceil(wp.length(displacement) / wp.max(0.5, radius * 0.5))),
            1,
            24,
        )
        collided = bool(False)
        for step in range(1, steps + 1):
            fraction = float(step) / float(steps)
            sample = start + fraction * displacement
            ix = wp.clamp(int(wp.round(sample[0])), 1, image_width - 2)
            iy = wp.clamp(int(wp.round(sample[1])), 1, image_height - 2)
            clearance = sdf[iy, ix]
            if clearance >= radius:
                continue
            normal = wp.normalize(sdf_gradient[iy, ix])
            if wp.length(normal) <= 0.01:
                continue
            contact = sample + (radius - clearance) * normal
            remainder = (1.0 - fraction) * displacement
            inward = wp.min(wp.dot(remainder, normal), 0.0)
            remainder -= inward * normal
            p = contact + (1.0 - theta[world, 4]) * remainder
            vn = wp.dot(v, normal)
            if vn < 0.0:
                tangent = v - vn * normal
                v = (1.0 - theta[world, 4]) * tangent - theta[world, 3] * vn * normal
            start = contact
            collided = True
            break
        if not collided:
            break
    p = wp.vec2(
        wp.clamp(p[0], radius, float(image_width - 1) - radius),
        wp.clamp(p[1], radius, float(image_height - 1) - radius),
    )
    position[world, particle] = p
    velocity[world, particle] = v


@wp.kernel
def _separate_particles(
    theta: wp.array2d(dtype=float),
    initial: wp.array(dtype=wp.vec2),
    position: wp.array2d(dtype=wp.vec2),
    birth: wp.array(dtype=int),
    group: wp.array(dtype=int),
    source: wp.array(dtype=int),
    projected: wp.array2d(dtype=wp.vec2),
    sdf: wp.array2d(dtype=float),
    sdf_gradient: wp.array2d(dtype=wp.vec2),
    frame: int,
    particle_count: int,
    image_height: int,
    image_width: int,
    jet_length: float,
    cross_color_only: int,
    source_count: int,
):
    world, particle = wp.tid()
    p = position[world, particle]
    if frame < birth[particle]:
        projected[world, particle] = p
        return
    radius = theta[world, 6]
    parcel_area = 1.0
    if source_count > 0:
        parcel_area = theta[world, 7 + source[particle]]
    correction = wp.vec2()
    for neighbor in range(particle_count):
        if neighbor == particle or frame < birth[neighbor]:
            continue
        if cross_color_only != 0 and group[neighbor] == group[particle]:
            continue
        same_source = wp.length(initial[particle] - initial[neighbor]) < 6.0
        delta = p - position[world, neighbor]
        distance = wp.length(delta)
        if distance > 1.0e-5 and distance < 2.0 * radius:
            neighbor_area = 1.0
            if source_count > 0:
                neighbor_area = theta[world, 7 + source[neighbor]]
            pair_correction = (
                neighbor_area
                / (parcel_area + neighbor_area)
                * (2.0 * radius - distance)
                * delta
                / distance
            )
            if same_source:
                scale = wp.min(
                    _jet_lateral_scale(p, initial[particle], jet_length),
                    _jet_lateral_scale(
                        position[world, neighbor], initial[neighbor], jet_length
                    ),
                )
                pair_correction[0] *= scale
            correction += pair_correction
    magnitude = wp.length(correction)
    if magnitude > radius:
        correction *= radius / magnitude
    p += correction
    ix = wp.clamp(int(wp.round(p[0])), 1, image_width - 2)
    iy = wp.clamp(int(wp.round(p[1])), 1, image_height - 2)
    clearance = sdf[iy, ix]
    if clearance < radius:
        normal = wp.normalize(sdf_gradient[iy, ix])
        if wp.length(normal) > 0.01:
            p += (radius - clearance) * normal
    projected[world, particle] = wp.vec2(
        wp.clamp(p[0], radius, float(image_width - 1) - radius),
        wp.clamp(p[1], radius, float(image_height - 1) - radius),
    )


@wp.kernel
def _commit_projection(
    position: wp.array2d(dtype=wp.vec2),
    projected: wp.array2d(dtype=wp.vec2),
    velocity: wp.array2d(dtype=wp.vec2),
    birth: wp.array(dtype=int),
    frame: int,
    dt: float,
):
    world, particle = wp.tid()
    if frame < birth[particle]:
        return
    delta = projected[world, particle] - position[world, particle]
    velocity[world, particle] += 0.15 * delta / dt
    position[world, particle] = projected[world, particle]


@wp.kernel
def _record_particles(
    position: wp.array2d(dtype=wp.vec2),
    birth: wp.array(dtype=int),
    predicted: wp.array3d(dtype=wp.vec2),
    frame: int,
):
    world, particle = wp.tid()
    predicted[world, frame, particle] = (
        position[world, particle] if frame >= birth[particle] else wp.vec2(-1.0, -1.0)
    )


@wp.kernel
def _occupancy_loss(
    predicted: wp.array4d(dtype=float),
    target: wp.array4d(dtype=float),
    normalizer: wp.array2d(dtype=float),
    loss: wp.array2d(dtype=float),
    frame: int,
    fluid_count: int,
):
    world, y, x = wp.tid()
    for fluid in range(fluid_count):
        observed = target[frame, fluid, y, x]
        simulated = wp.min(1.0, predicted[world, fluid, y, x])
        difference = simulated - observed
        weight = 1.0 if observed > 0.05 else 0.06
        if observed <= 0.05:
            for other in range(fluid_count):
                if other != fluid and target[frame, other, y, x] > 0.05:
                    weight = 1.0
        wp.atomic_add(
            loss,
            world,
            frame,
            weight * difference * difference / normalizer[frame, fluid],
        )


@wp.kernel
def _particle_distance_loss(
    position: wp.array2d(dtype=wp.vec2),
    birth: wp.array(dtype=int),
    group: wp.array(dtype=int),
    target_distance: wp.array4d(dtype=float),
    active_count: wp.array2d(dtype=float),
    loss: wp.array2d(dtype=float),
    frame: int,
    grid_scale: float,
    grid_height: int,
    grid_width: int,
):
    world, particle = wp.tid()
    if frame < birth[particle]:
        return
    fluid = group[particle]
    p = position[world, particle] / grid_scale
    x = wp.clamp(int(wp.round(p[0])), 0, grid_width - 1)
    y = wp.clamp(int(wp.round(p[1])), 0, grid_height - 1)
    distance = wp.min(target_distance[frame, fluid, y, x], 20.0)
    wp.atomic_add(
        loss,
        world,
        frame,
        2.0 * distance * distance / (400.0 * active_count[frame, fluid]),
    )


@wp.kernel
def _source_stream_loss(
    position: wp.array2d(dtype=wp.vec2),
    birth: wp.array(dtype=int),
    source: wp.array(dtype=int),
    source_distance: wp.array4d(dtype=float),
    source_weight: wp.array2d(dtype=float),
    loss: wp.array2d(dtype=float),
    frame: int,
    grid_scale: float,
    grid_height: int,
    grid_width: int,
):
    world, particle = wp.tid()
    age = frame - birth[particle]
    if age < 0 or age > 18:
        return
    emitter = source[particle]
    weight = source_weight[frame, emitter]
    if weight <= 0.0:
        return
    p = position[world, particle] / grid_scale
    x = wp.clamp(int(wp.round(p[0])), 0, grid_width - 1)
    y = wp.clamp(int(wp.round(p[1])), 0, grid_height - 1)
    distance_pixels = wp.min(grid_scale * source_distance[frame, emitter, y, x], 20.0)
    wp.atomic_add(
        loss, world, emitter, weight * distance_pixels * distance_pixels / 400.0
    )


def _collision_sdf(obstacle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    free = (~obstacle).astype(np.uint8)
    solid = obstacle.astype(np.uint8)
    sdf = cv2.distanceTransform(free, cv2.DIST_L2, 5) - cv2.distanceTransform(
        solid, cv2.DIST_L2, 5
    )
    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return sdf.astype(np.float32), np.stack((gx, gy), axis=-1)


class BatchWarpFluidEngine:
    def __init__(
        self,
        scene: FluidScene,
        *,
        capacity: int,
        device: str = "cuda:0",
        substeps: int = 2,
        grid_scale: int = 4,
        particle_radius: float = 1.5,
        jet_length: float | None = None,
        contact_iterations: int = 1,
    ) -> None:
        if capacity < 1 or substeps < 1 or grid_scale < 1 or contact_iterations < 1:
            raise ValueError("invalid simulation capacity or resolution")
        self.scene = scene
        self.capacity = capacity
        self.device = device
        self.substeps = substeps
        self.grid_scale = float(grid_scale)
        self.particle_radius = particle_radius
        self.contact_iterations = contact_iterations
        self.jet_length = (
            0.18 * scene.image_height if jet_length is None else float(jet_length)
        )
        if self.jet_length < 0:
            raise ValueError("jet length cannot be negative")
        self.grid_height = math.ceil(scene.image_height / grid_scale) + 1
        self.grid_width = math.ceil(scene.image_width / grid_scale) + 1
        self.particle_count = len(scene.particle_birth)
        self.fluid_count = len(scene.fluid_ids)
        self.source_count = len(scene.emitters)
        if self.source_count and scene.particle_emitter is None:
            raise ValueError("particle emitter IDs are required for source flux")
        self.parameter_count = 7 + self.source_count
        self.theta = wp.empty(
            (capacity, self.parameter_count), dtype=float, device=device
        )
        self.theta_torch = wp.to_torch(self.theta)
        self.initial = wp.array(scene.particle_xy, dtype=wp.vec2, device=device)
        self.birth = wp.array(scene.particle_birth, dtype=int, device=device)
        self.group = wp.array(scene.particle_fluid, dtype=int, device=device)
        self.source = wp.array(
            scene.particle_emitter
            if scene.particle_emitter is not None
            else np.zeros(self.particle_count, dtype=np.int32),
            dtype=int,
            device=device,
        )
        sdf, gradient = _collision_sdf(
            scene.collision_obstacle_mask
            if scene.collision_obstacle_mask is not None
            else scene.obstacle_mask
        )
        self.sdf = wp.array(sdf, dtype=float, device=device)
        self.sdf_gradient = wp.array(gradient, dtype=wp.vec2, device=device)
        grid_shape = (capacity, self.grid_height, self.grid_width)
        group_shape = (capacity, self.fluid_count, self.grid_height, self.grid_width)
        self.total = wp.zeros(grid_shape, dtype=float, device=device)
        self.per_fluid = wp.zeros(group_shape, dtype=float, device=device)
        self.position = wp.empty(
            (capacity, self.particle_count), dtype=wp.vec2, device=device
        )
        self.velocity = wp.empty(
            (capacity, self.particle_count), dtype=wp.vec2, device=device
        )
        self.projected = wp.empty(
            (capacity, self.particle_count), dtype=wp.vec2, device=device
        )
        self.predicted = wp.empty(
            (capacity, scene.frame_count, self.particle_count),
            dtype=wp.vec2,
            device=device,
        )
        self.predicted_torch = wp.to_torch(self.predicted)
        target = np.empty(
            (scene.frame_count, self.fluid_count, self.grid_height, self.grid_width),
            dtype=np.float32,
        )
        target_distance = np.empty_like(target)
        for frame in range(scene.frame_count):
            for fluid in range(self.fluid_count):
                resized = cv2.resize(
                    scene.target_masks[frame, fluid].astype(np.uint8),
                    (self.grid_width, self.grid_height),
                    interpolation=cv2.INTER_AREA,
                )
                target[frame, fluid] = cv2.GaussianBlur(
                    (resized > 0).astype(np.float32), (3, 3), 0.75
                )
                target_distance[frame, fluid] = (
                    cv2.distanceTransform(
                        (resized == 0).astype(np.uint8), cv2.DIST_L2, 5
                    )
                    if np.any(resized)
                    else 0.0
                )
        normalizer = np.maximum(target.sum(axis=(2, 3)), 1.0).astype(np.float32)
        active_count = np.empty((scene.frame_count, self.fluid_count), np.float32)
        for frame in range(scene.frame_count):
            for fluid in range(self.fluid_count):
                active_count[frame, fluid] = max(
                    1,
                    np.count_nonzero(
                        (scene.particle_birth <= frame)
                        & (scene.particle_fluid == fluid)
                    ),
                )
        self.target = wp.array(target, dtype=float, device=device)
        self.target_distance = wp.array(target_distance, dtype=float, device=device)
        self.normalizer = wp.array(normalizer, dtype=float, device=device)
        self.active_count = wp.array(active_count, dtype=float, device=device)
        self.loss = wp.zeros((capacity, scene.frame_count), dtype=float, device=device)
        self.loss_torch = wp.to_torch(self.loss)
        self.source_loss = wp.zeros(
            (capacity, max(1, self.source_count)), dtype=float, device=device
        )
        self.source_loss_torch = wp.to_torch(self.source_loss)
        if scene.particle_emitter is not None and self.source_count:
            source_masks = source_stream_masks(scene, self.grid_height, self.grid_width)
            self.constrained_sources = np.flatnonzero(source_masks.any(axis=(0, 2, 3)))
            source_distance = np.full(source_masks.shape, 20.0, np.float32)
            source_weight = np.zeros((scene.frame_count, self.source_count), np.float32)
            for source in self.constrained_sources:
                valid_frames = np.flatnonzero(source_masks[:, source].any(axis=(1, 2)))
                for frame in valid_frames:
                    source_distance[frame, source] = cv2.distanceTransform(
                        (~source_masks[frame, source]).astype(np.uint8),
                        cv2.DIST_L2,
                        5,
                    )
                    recent_count = np.count_nonzero(
                        (scene.particle_emitter == source)
                        & (scene.particle_birth <= frame)
                        & (scene.particle_birth >= frame - 18)
                    )
                    source_weight[frame, source] = 1.0 / (
                        len(valid_frames) * max(1, recent_count)
                    )
            self.source_distance = wp.array(source_distance, dtype=float, device=device)
            self.source_weight = wp.array(source_weight, dtype=float, device=device)
        else:
            self.constrained_sources = np.empty(0, np.int32)

    def rollout(
        self, values: torch.Tensor | np.ndarray, end_frame: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)
        if parameters.shape[1] == 6:
            defaults = torch.ones(
                (len(parameters), 1 + self.source_count),
                device=self.device,
                dtype=torch.float32,
            )
            defaults[:, 0] = self.particle_radius
            parameters = torch.cat((parameters, defaults), dim=1)
        count = len(parameters)
        if count > self.capacity or parameters.shape[1] != self.parameter_count:
            raise ValueError("invalid fluid parameter batch")
        last = (
            self.scene.frame_count - 1
            if end_frame is None
            else min(end_frame, self.scene.frame_count - 1)
        )
        self.theta_torch[:count].copy_(parameters)
        wp.launch(
            _reset_particles,
            dim=(count, self.particle_count),
            inputs=[self.initial, self.position, self.velocity],
            device=self.device,
        )
        self.loss_torch[:count, : last + 1].zero_()
        self.source_loss_torch[:count].zero_()
        dt = 1.0 / self.scene.fps / self.substeps
        for frame in range(last + 1):
            for substep in range(self.substeps):
                wp.launch(
                    _clear_density,
                    dim=(count, self.grid_height, self.grid_width),
                    inputs=[self.total, self.per_fluid, self.fluid_count],
                    device=self.device,
                )
                wp.launch(
                    _deposit_particles,
                    dim=(count, self.particle_count),
                    inputs=[
                        self.theta,
                        self.position,
                        self.birth,
                        self.group,
                        self.source,
                        self.total,
                        self.per_fluid,
                        frame,
                        self.source_count,
                        self.grid_scale,
                        self.grid_height,
                        self.grid_width,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _advance_particles,
                    dim=(count, self.particle_count),
                    inputs=[
                        self.theta,
                        self.initial,
                        self.birth,
                        self.position,
                        self.velocity,
                        self.total,
                        self.sdf,
                        self.sdf_gradient,
                        frame,
                        substep,
                        dt,
                        self.grid_scale,
                        self.grid_height,
                        self.grid_width,
                        self.scene.image_height,
                        self.scene.image_width,
                        self.jet_length,
                    ],
                    device=self.device,
                )
                for iteration in range(self.contact_iterations):
                    wp.launch(
                        _separate_particles,
                        dim=(count, self.particle_count),
                        inputs=[
                            self.theta,
                            self.initial,
                            self.position,
                            self.birth,
                            self.group,
                            self.source,
                            self.projected,
                            self.sdf,
                            self.sdf_gradient,
                            frame,
                            self.particle_count,
                            self.scene.image_height,
                            self.scene.image_width,
                            self.jet_length,
                            int(iteration > 0),
                            self.source_count,
                        ],
                        device=self.device,
                    )
                    wp.launch(
                        _commit_projection,
                        dim=(count, self.particle_count),
                        inputs=[
                            self.position,
                            self.projected,
                            self.velocity,
                            self.birth,
                            frame,
                            dt,
                        ],
                        device=self.device,
                    )
            wp.launch(
                _clear_density,
                dim=(count, self.grid_height, self.grid_width),
                inputs=[self.total, self.per_fluid, self.fluid_count],
                device=self.device,
            )
            wp.launch(
                _deposit_particles,
                dim=(count, self.particle_count),
                inputs=[
                    self.theta,
                    self.position,
                    self.birth,
                    self.group,
                    self.source,
                    self.total,
                    self.per_fluid,
                    frame,
                    self.source_count,
                    self.grid_scale,
                    self.grid_height,
                    self.grid_width,
                ],
                device=self.device,
            )
            wp.launch(
                _occupancy_loss,
                dim=(count, self.grid_height, self.grid_width),
                inputs=[
                    self.per_fluid,
                    self.target,
                    self.normalizer,
                    self.loss,
                    frame,
                    self.fluid_count,
                ],
                device=self.device,
            )
            wp.launch(
                _particle_distance_loss,
                dim=(count, self.particle_count),
                inputs=[
                    self.position,
                    self.birth,
                    self.group,
                    self.target_distance,
                    self.active_count,
                    self.loss,
                    frame,
                    self.grid_scale,
                    self.grid_height,
                    self.grid_width,
                ],
                device=self.device,
            )
            if len(self.constrained_sources):
                wp.launch(
                    _source_stream_loss,
                    dim=(count, self.particle_count),
                    inputs=[
                        self.position,
                        self.birth,
                        self.source,
                        self.source_distance,
                        self.source_weight,
                        self.source_loss,
                        frame,
                        self.grid_scale,
                        self.grid_height,
                        self.grid_width,
                    ],
                    device=self.device,
                )
            wp.launch(
                _record_particles,
                dim=(count, self.particle_count),
                inputs=[self.position, self.birth, self.predicted, frame],
                device=self.device,
            )
        return self.predicted_torch[:count, : last + 1], self.loss_torch[
            :count, : last + 1
        ]


@dataclass(frozen=True)
class FluidCEMConfig:
    population: int = 16
    generations: int = 4
    seed: int = 7
    elite_fraction: float = 0.25
    restarts: int = 3
    source_loss_weight: float = 4.0


def fit_fluid_scene(
    scene: FluidScene,
    *,
    config: FluidCEMConfig = FluidCEMConfig(),
    device: str = "cuda:0",
    substeps: int = 2,
    contact_iterations: int | None = None,
    jet_length: float | None = None,
    warm_start: np.ndarray | None = None,
    infer_flux_and_radius: bool = False,
    verbose: bool = True,
) -> dict:
    if config.population < 2 or config.generations < 1 or config.restarts < 1:
        raise ValueError("CEM population, generations, and restarts must be positive")
    if config.source_loss_weight < 0:
        raise ValueError("source loss weight cannot be negative")
    if jet_length is not None and jet_length < 0:
        raise ValueError("jet length cannot be negative")
    if contact_iterations is None:
        started = time.perf_counter()
        candidates = [
            fit_fluid_scene(
                scene,
                config=config,
                device=device,
                substeps=substeps,
                contact_iterations=iterations,
                jet_length=jet_length,
                warm_start=warm_start,
                infer_flux_and_radius=infer_flux_and_radius,
                verbose=verbose,
            )
            for iterations in (1, 3)
        ]
        selected = min(candidates, key=lambda item: item["selection_score"])
        return {
            **selected,
            "elapsed_seconds": time.perf_counter() - started,
            "contact_solver_candidates": [
                {
                    "contact_iterations": item["contact_iterations"],
                    "selection_score": item["selection_score"],
                    "mean_chamfer_pixels": item["mean_chamfer_pixels"],
                    "mean_source_rmse_pixels": item["mean_source_rmse_pixels"],
                    "mean_shape_loss": item["mean_shape_loss"],
                }
                for item in candidates
            ],
        }
    started = time.perf_counter()
    engine = BatchWarpFluidEngine(
        scene,
        capacity=config.population,
        device=device,
        substeps=substeps,
        contact_iterations=contact_iterations,
        jet_length=jet_length,
    )
    lower = torch.tensor([40.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    upper = torch.tensor([600.0, 5.0, 300.0, 0.2, 1.0, 100.0], device=device)
    initial_mean = torch.tensor([180.0, 0.5, 30.0, 0.1, 0.2, 10.0], device=device)
    initial_std = torch.tensor([80.0, 0.6, 60.0, 0.2, 0.2, 20.0], device=device)
    lower = torch.cat(
        (
            lower,
            torch.tensor([0.75], device=device),
            torch.full((engine.source_count,), 0.35, device=device),
        )
    )
    upper = torch.cat(
        (
            upper,
            torch.tensor([2.25], device=device),
            torch.full((engine.source_count,), 2.5, device=device),
        )
    )
    initial_mean = torch.cat(
        (
            initial_mean,
            torch.tensor([1.5], device=device),
            torch.ones(engine.source_count, device=device),
        )
    )
    initial_std = torch.cat(
        (
            initial_std,
            torch.tensor([0.25], device=device),
            torch.full((engine.source_count,), 0.35, device=device),
        )
    )
    if not infer_flux_and_radius:
        lower[6:] = initial_mean[6:]
        upper[6:] = initial_mean[6:]
        initial_std[6:] = 0.0
    floor = 0.03 * (upper - lower)
    elites = max(2, round(config.population * config.elite_fraction))
    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=np.float32)
        if warm_start.shape == (6,):
            warm_start = np.concatenate(
                (
                    warm_start,
                    [engine.particle_radius],
                    np.ones(engine.source_count, np.float32),
                )
            )
        if (
            warm_start.shape != (engine.parameter_count,)
            or not np.isfinite(warm_start).all()
        ):
            raise ValueError("warm start must be a finite fluid parameter vector")
        warm_start_torch = torch.as_tensor(warm_start, device=device).clamp(
            lower, upper
        )
    weights = (
        (0.0, config.source_loss_weight)
        if len(engine.constrained_sources) and config.source_loss_weight
        else (0.0,)
    )
    fits = []
    trials_per_weight = config.restarts + (
        2
        if infer_flux_and_radius and warm_start is not None
        else int(warm_start is not None)
    )
    for trial in range(trials_per_weight * len(weights)):
        restart = trial % trials_per_weight
        source_loss_weight = weights[trial // trials_per_weight]
        seed = config.seed + 11 * restart
        generator = torch.Generator(device=device).manual_seed(seed)
        if (
            infer_flux_and_radius
            and warm_start is not None
            and restart == config.restarts
        ):
            mean = warm_start_torch.clone()
            std = 0.2 * (upper - lower)
            std[:6] = 0.0
            fixed_dynamics = True
        elif (
            infer_flux_and_radius
            and warm_start is not None
            and restart == config.restarts + 1
        ):
            mean = torch.as_tensor(fits[-1]["parameters"], device=device).clone()
            std = 0.08 * (upper - lower)
            fixed_dynamics = False
        elif warm_start is not None and restart == config.restarts:
            mean = warm_start_torch.clone()
            std = 0.15 * (upper - lower)
            fixed_dynamics = False
        else:
            mean, std = initial_mean.clone(), initial_std.clone()
            fixed_dynamics = False
        best, best_loss = mean.clone(), math.inf
        trace = []
        for cutoff in dict.fromkeys(
            [
                min(59, scene.frame_count - 1),
                min(129, scene.frame_count - 1),
                scene.frame_count - 1,
            ]
        ):
            for generation in range(config.generations):
                batch = (
                    mean
                    + std
                    * torch.randn(
                        (config.population, engine.parameter_count),
                        device=device,
                        generator=generator,
                    )
                ).clamp(lower, upper)
                if fixed_dynamics:
                    batch[:, :6] = warm_start_torch[:6]
                batch[0] = best
                _trajectory, losses = engine.rollout(batch, cutoff)
                score = losses[:, : cutoff + 1].mean(dim=1)
                if source_loss_weight:
                    score += source_loss_weight * engine.source_loss_torch[
                        : config.population, engine.constrained_sources
                    ].mean(dim=1)
                elite_score, elite_indices = torch.topk(score, elites, largest=False)
                elite = batch[elite_indices]
                mean = 0.25 * mean + 0.75 * elite.mean(dim=0)
                std = torch.maximum(
                    0.25 * std + 0.75 * elite.std(dim=0, unbiased=False), floor
                )
                if fixed_dynamics:
                    mean[:6] = warm_start_torch[:6]
                    std[:6] = 0.0
                if elite_score[0].item() < best_loss or generation == 0:
                    best, best_loss = elite[0].clone(), float(elite_score[0].item())
            mean = best.clone()
            trace.append(
                {
                    "end_frame": cutoff,
                    "loss": best_loss,
                    "seed": seed,
                    "source_loss_weight": source_loss_weight,
                }
            )
            if verbose:
                print(json.dumps({"phase": "fluid_cem", **trace[-1]}), flush=True)
        predicted, losses = engine.rollout(best)
        positions = predicted[0].detach().cpu().numpy().copy()
        frame_losses = losses[0].detach().cpu().numpy().copy()
        chamfer = fluid_chamfer_rmse(scene, positions)
        source_losses = (
            engine.source_loss_torch[0, engine.constrained_sources]
            .detach()
            .cpu()
            .numpy()
        )
        source_rmse = 20.0 * np.sqrt(np.maximum(source_losses, 0.0))
        mean_chamfer = float(np.mean(list(chamfer.values())))
        mean_source_rmse = float(np.mean(source_rmse)) if len(source_rmse) else 0.0
        mean_shape_loss = float(np.mean(frame_losses))
        shape_weight = 0.1 if infer_flux_and_radius else 0.0
        fits.append(
            {
                "parameters": best.detach().cpu().numpy().copy(),
                "predicted": positions,
                "frame_losses": frame_losses,
                "trace": trace,
                "seed": seed,
                "source_loss_weight": source_loss_weight,
                "chamfer_pixels": chamfer,
                "source_rmse_pixels": {
                    scene.emitters[int(source)].object_id: float(value)
                    for source, value in zip(engine.constrained_sources, source_rmse)
                },
                "mean_chamfer_pixels": mean_chamfer,
                "mean_source_rmse_pixels": mean_source_rmse,
                "mean_shape_loss": mean_shape_loss,
                "selection_score": (
                    mean_chamfer
                    + 0.5 * mean_source_rmse
                    + shape_weight * mean_shape_loss
                ),
            }
        )
    if warm_start is not None:
        predicted, losses = engine.rollout(warm_start_torch)
        positions = predicted[0].detach().cpu().numpy().copy()
        chamfer = fluid_chamfer_rmse(scene, positions)
        source_losses = (
            engine.source_loss_torch[0, engine.constrained_sources]
            .detach()
            .cpu()
            .numpy()
        )
        source_rmse = 20.0 * np.sqrt(np.maximum(source_losses, 0.0))
        mean_chamfer = float(np.mean(list(chamfer.values())))
        mean_source_rmse = float(np.mean(source_rmse)) if len(source_rmse) else 0.0
        frame_losses = losses[0].detach().cpu().numpy().copy()
        mean_shape_loss = float(np.mean(frame_losses))
        shape_weight = 0.1 if infer_flux_and_radius else 0.0
        fits.append(
            {
                "parameters": warm_start_torch.detach().cpu().numpy().copy(),
                "predicted": positions,
                "frame_losses": frame_losses,
                "trace": [],
                "seed": "warm_start",
                "source_loss_weight": 0.0,
                "chamfer_pixels": chamfer,
                "source_rmse_pixels": {
                    scene.emitters[int(source)].object_id: float(value)
                    for source, value in zip(engine.constrained_sources, source_rmse)
                },
                "mean_chamfer_pixels": mean_chamfer,
                "mean_source_rmse_pixels": mean_source_rmse,
                "mean_shape_loss": mean_shape_loss,
                "selection_score": (
                    mean_chamfer
                    + 0.5 * mean_source_rmse
                    + shape_weight * mean_shape_loss
                ),
            }
        )
    selected = min(fits, key=lambda item: item["selection_score"])
    return {
        **selected,
        "restart_scores": [
            {
                "seed": item["seed"],
                "source_loss_weight": item["source_loss_weight"],
                "mean_chamfer_pixels": item["mean_chamfer_pixels"],
                "mean_source_rmse_pixels": item["mean_source_rmse_pixels"],
                "mean_shape_loss": item["mean_shape_loss"],
                "selection_score": item["selection_score"],
            }
            for item in fits
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "device": device,
        "substeps": substeps,
        "contact_iterations": contact_iterations,
        "grid_scale": engine.grid_scale,
        "particle_radius": float(selected["parameters"][6]),
        "emitter_area_scales": selected["parameters"][7:].copy(),
        "infer_flux_and_radius": infer_flux_and_radius,
        "jet_length": engine.jet_length,
        "source_loss_weight": selected["source_loss_weight"],
    }


def fluid_chamfer_rmse(scene: FluidScene, predicted: np.ndarray) -> dict[str, float]:
    """Symmetric pixel-to-particle distance on frames with visible fluid."""
    if predicted.shape != (scene.frame_count, len(scene.particle_birth), 2):
        raise ValueError("predicted particle trajectory shape does not match scene")
    result = {}
    for group, fluid_id in enumerate(scene.fluid_ids):
        squared = []
        group_particles = scene.particle_fluid == group
        for frame in range(scene.frame_count):
            yy, xx = np.where(scene.target_masks[frame, group])
            active = group_particles & (scene.particle_birth <= frame)
            observed = np.column_stack((xx, yy))
            simulated = predicted[frame, active]
            simulated = simulated[np.isfinite(simulated).all(axis=1)]
            simulated = simulated[(simulated >= 0).all(axis=1)]
            if not len(observed) or not len(simulated):
                continue
            squared.extend(cKDTree(observed).query(simulated)[0] ** 2)
            squared.extend(cKDTree(simulated).query(observed)[0] ** 2)
        result[fluid_id] = float(np.sqrt(np.mean(squared))) if squared else math.nan
    return result


def result_payload(scene: FluidScene, fit: dict, prediction_file: Path) -> dict:
    values = fit["parameters"]
    emitter_area_scales = fit.get(
        "emitter_area_scales", np.ones(len(scene.emitters), dtype=np.float32)
    )
    relative_flux = {
        emitter.object_id: float(scale)
        for emitter, scale in zip(scene.emitters, emitter_area_scales, strict=True)
    }
    return {
        "schema": "physmind.fluid_swr_scene",
        "schema_version": 1,
        "model": "physmind-v2-warp-grid-particle-fluid-v4",
        "source": str(scene.scene_dir),
        "source_video": scene.source_video,
        "collision_geometry": "fixed barriers and basins; emitter shells excluded",
        "coordinate_system": {
            "x": "image_right",
            "y": "image_down",
            "units": "mask_pixels",
        },
        "gravity_direction": [0.0, 1.0],
        "gravity_direction_source": "ContPhy 2D scene prior: image down",
        "image_wh": [scene.image_width, scene.image_height],
        "fps": scene.fps,
        "frame_count": scene.frame_count,
        "fluid_ids": list(scene.fluid_ids),
        "emitters": [
            {
                "object_id": emitter.object_id,
                "fluid_id": emitter.fluid_id,
                "xy": list(emitter.xy),
                "active_frames": list(emitter.active_frames),
            }
            for emitter in scene.emitters
        ],
        "particle_count": len(scene.particle_birth),
        "source_local_stream_constraint": {
            "scope": "same-color emitter streams above mixing region",
            "pooled_fluid_source_attribution": "unidentifiable from color masks",
            "observation_npz_key": "source_stream_masks",
            "observation_shape": [
                scene.frame_count,
                len(scene.emitters),
                scene.image_height,
                scene.image_width,
            ],
        },
        "emission_stride": scene.emission_stride,
        "particle_radius_pixels": float(fit["particle_radius"]),
        "emission_2d_flux": {
            "unit": "baseline parcel-area units per second while active",
            "inferred": bool(fit.get("infer_flux_and_radius", False)),
            "sample_markers_per_emission_frame": 2,
            "emitter_relative_flux": relative_flux,
            "emitter_nominal_rate": {
                emitter.object_id: float(
                    2.0 * scene.fps / scene.emission_stride * scale
                )
                for emitter, scale in zip(
                    scene.emitters, emitter_area_scales, strict=True
                )
            },
        },
        "jet_length_pixels": float(fit["jet_length"]),
        "prediction_npz": prediction_file.name,
        "parameters": {
            "gravity_pixels_per_s2": float(values[0]),
            "linear_drag_per_s": float(values[1]),
            "density_pressure_pixels_per_s2": float(values[2]),
            "collision_restitution": float(values[3]),
            "collision_tangent_damping": float(values[4]),
            "emission_speed_pixels_per_s": float(values[5]),
            "effective_contact_radius_pixels": float(fit["particle_radius"]),
            "emitter_relative_flux": relative_flux,
        },
        "optimization": {
            "trace": fit["trace"],
            "selected_seed": fit["seed"],
            "restart_scores": fit["restart_scores"],
            "elapsed_seconds": fit["elapsed_seconds"],
            "device": fit["device"],
            "substeps": fit["substeps"],
            "contact_iterations": fit.get("contact_iterations", 1),
            "contact_solver_candidates": fit.get("contact_solver_candidates", []),
            "source_loss_weight": fit["source_loss_weight"],
        },
        "metrics": {
            "mean_shape_loss": float(np.mean(fit["frame_losses"])),
            "fluid_chamfer_rmse_pixels": fit["chamfer_pixels"],
            "source_stream_rmse_pixels": fit["source_rmse_pixels"],
        },
    }
