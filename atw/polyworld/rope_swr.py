"""Open-loop 2D rope simulation and system identification from SAM3 masks."""

from __future__ import annotations

import heapq
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from skimage.morphology import skeletonize

from atw.perception.object_reconstruction import read_packed_mask_archive


DEFAULT_SEEDS = (7, 23, 43)
DEFAULT_CUDA_BLOCK_DIM = 32
DEFAULT_EXECUTION_MODE = "auto"
ROPE_EXECUTION_MODES = ("auto", "monolithic", "guarded-colored", "guarded-chunked", "ideal-warp")
DEFAULT_NODE_COUNT = 48
DEFAULT_MASSLESS_SUBSTEPS = 16
DEFAULT_TRACKING_POLICY = "visible-centers-v2"
DEFAULT_ENDPOINT_POLICY = "surface-midpoint-v1"
MIN_COMPONENT_PIXELS = 4
ROPE_PARTICLE_INVERSE_MASS = 20.0


@dataclass(frozen=True, slots=True)
class RopeBody:
    object_id: str
    track_id: str
    role: str
    shape: str
    dynamic: bool
    radius: float
    initial_center: tuple[float, float]
    target_centers: np.ndarray
    observed: np.ndarray
    endpoint_index: int
    endpoint_offset: tuple[float, float]
    half_extent: tuple[float, float] = (0.0, 0.0)
    parent_id: str | None = None
    attachment_source: str | None = None
    default_relative_mass: float = 1.0


@dataclass(frozen=True, slots=True)
class RopeSceneData:
    rope_id: str
    rope_track_id: str
    frames: tuple[int, ...]
    fps: float
    image_width: int
    image_height: int
    event_frame_offset: int
    target_positions: np.ndarray
    initial_positions: np.ndarray
    endpoint_pinned: tuple[bool, bool]
    bodies: tuple[RopeBody, ...]
    source_video: str
    mask_archive: str
    mask_index: str
    wrap_hints: tuple[tuple[str, int], ...] = ()
    mass_model: str = "particle"
    route_visits: tuple[tuple[str, int, int], ...] = ()

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def node_count(self) -> int:
        return int(self.initial_positions.shape[0])


@dataclass(frozen=True, slots=True)
class RopeCEMConfig:
    population: int = 512
    elite_fraction: float = 0.08
    generations_per_prefix: int = 36
    prefix_frames: tuple[int, ...] = (24, 48, 72, 0)
    distribution_smoothing: float = 0.2
    exploration_fraction: float = 0.025
    huber_delta: float = 0.02
    body_weight: float = 2.0
    seed: int = 23


def _mask_center(mask: np.ndarray) -> tuple[np.ndarray, float]:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return np.full(2, np.nan), 0.0
    center = np.asarray([x.mean(), y.mean()], dtype=np.float64)
    radius = math.sqrt(float(len(x)) / math.pi)
    return center, radius


def _mask_outer_circle(mask: np.ndarray) -> tuple[np.ndarray, float]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.full(2, np.nan), 0.0
    contour = max(contours, key=cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(contour)
    return np.asarray(center, dtype=np.float64), float(radius)


def _mask_touches_border(mask: np.ndarray) -> bool:
    return any(np.any(edge) for edge in (mask[:2], mask[-2:], mask[:, :2], mask[:, -2:]))


def _track_motion(centers: np.ndarray, observed: np.ndarray) -> float:
    """Reject isolated centroid errors without smoothing the fitting targets."""
    valid = centers[observed]
    if not len(valid):
        raise ValueError("Motion classification requires visible centers")
    filtered = median_filter(valid, size=(5, 1), mode="reflect") if len(valid) >= 5 else valid
    return float(np.linalg.norm(filtered - np.median(filtered[:8], axis=0), axis=1).max())


def _interpolate_centers(
    masks: Mapping[int, np.ndarray],
    frame_count: int,
    *,
    circular: bool = False,
    reject_truncated: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    centers = np.full((frame_count, 2), np.nan, dtype=np.float64)
    radii = []
    observed = np.zeros(frame_count, dtype=bool)
    for frame, mask in masks.items():
        if 0 <= frame < frame_count:
            if reject_truncated and _mask_touches_border(mask):
                continue
            center, radius = (
                _mask_outer_circle(mask) if circular else _mask_center(mask)
            )
            if np.isfinite(center).all():
                centers[frame] = center
                observed[frame] = True
                radii.append(radius)
    valid = np.flatnonzero(observed)
    if len(valid) == 0:
        raise ValueError("object track has no reliable non-empty masks")
    for axis in range(2):
        centers[:, axis] = np.interp(
            np.arange(frame_count), valid, centers[valid, axis]
        )
    return centers, observed, float(np.median(radii))


def _track_half_extent(
    masks: Mapping[int, np.ndarray], frame_count: int, *, reject_truncated: bool = False,
) -> tuple[float, float]:
    samples = []
    for frame in sorted({0, min(48, frame_count - 1), frame_count - 1}):
        mask = masks.get(frame)
        if mask is None or not np.any(mask):
            continue
        if reject_truncated and _mask_touches_border(mask):
            continue
        x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
        samples.append((width / 2.0, height / 2.0))
    return tuple(np.median(samples, axis=0)) if samples else (0.0, 0.0)


def _connect_skeleton_components(skeleton: np.ndarray) -> np.ndarray:
    result = skeleton.astype(np.uint8).copy()
    while True:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
        keep = [
            index
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PIXELS
        ]
        if not keep:
            return result.astype(bool)
        result[~np.isin(labels, keep)] = 0
        if len(keep) == 1:
            return result.astype(bool)
        components = [np.column_stack(np.nonzero(labels == index)) for index in keep]
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for first in range(len(components)):
            tree = cKDTree(components[first])
            for second in range(first + 1, len(components)):
                distance, index = tree.query(components[second], k=1)
                nearest = int(np.argmin(distance))
                candidate = (
                    float(distance[nearest]),
                    components[first][int(index[nearest])],
                    components[second][nearest],
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        assert best is not None
        start, end = best[1], best[2]
        cv2.line(
            result,
            (int(start[1]), int(start[0])),
            (int(end[1]), int(end[0])),
            1,
            1,
            cv2.LINE_8,
        )


def _dijkstra_farthest(
    skeleton: np.ndarray,
    start: tuple[int, int],
    *,
    keep_parent: bool,
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    height, width = skeleton.shape
    distances = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    farthest = start
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        if distance > distances[farthest]:
            farthest = current
        y, x = current
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx == 0 and dy == 0) or not (
                    0 <= y + dy < height and 0 <= x + dx < width
                ):
                    continue
                neighbor = (y + dy, x + dx)
                if not skeleton[neighbor]:
                    continue
                candidate = distance + (math.sqrt(2.0) if dx and dy else 1.0)
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    if keep_parent:
                        parents[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
    return farthest, parents


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("rope centerline has fewer than two points")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segment)))
    keep = np.concatenate(([True], np.diff(distance) > 1.0e-8))
    points, distance = points[keep], distance[keep]
    if distance[-1] <= 1.0e-8:
        raise ValueError("rope centerline has zero length")
    samples = np.linspace(0.0, distance[-1], count)
    return np.column_stack(
        [np.interp(samples, distance, points[:, axis]) for axis in range(2)]
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / length if length > 1.0e-8 else np.zeros(2, dtype=np.float64)


def _endpoint_branch_direction(
    skeleton: np.ndarray, endpoint: tuple[int, int], steps: int = 18
) -> np.ndarray:
    current = endpoint
    previous = None
    for _ in range(steps):
        y, x = current
        neighbors = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx)
            and 0 <= y + dy < skeleton.shape[0]
            and 0 <= x + dx < skeleton.shape[1]
            and skeleton[y + dy, x + dx]
            and (y + dy, x + dx) != previous
        ]
        if len(neighbors) != 1:
            break
        previous, current = current, neighbors[0]
    return _unit(np.asarray(current[::-1], dtype=float) - endpoint[::-1])


def _tangent_contact(
    endpoint: tuple[int, int],
    branch: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    point = np.asarray(endpoint[::-1], dtype=np.float64)
    if not np.any(branch):
        return center + (radius + 4.0) * _unit(point - center)
    normal = np.asarray((-branch[1], branch[0]))
    options = (center + (radius + 4.0) * normal, center - (radius + 4.0) * normal)
    return min(options, key=lambda option: float(np.linalg.norm(option - point)))


def _choose_arc_delta(
    start: float,
    end: float,
    incoming: np.ndarray,
    outgoing: np.ndarray,
) -> float:
    shortest = (end - start + math.pi) % (2.0 * math.pi) - math.pi
    if abs(shortest) < 1.0e-8:
        return shortest
    tangent_start = np.asarray((-math.sin(start), math.cos(start)))
    tangent_end = np.asarray((-math.sin(end), math.cos(end)))
    score = float(
        np.dot(tangent_start, _unit(incoming)) + np.dot(tangent_end, _unit(outgoing))
    ) * math.copysign(1.0, shortest)
    return (
        shortest if score >= -0.3 else shortest - math.copysign(2.0 * math.pi, shortest)
    )


def _topology_aware_path(
    skeleton: np.ndarray,
    circles: list[tuple[np.ndarray, float]],
) -> np.ndarray | None:
    """Route through pulley arcs, allowing overlapping strands to reuse a segment."""
    degree = (
        cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.uint8))
        - skeleton
    )
    endpoints = [
        tuple(int(value) for value in point)
        for point in np.column_stack(np.nonzero(skeleton & (degree == 1)))
    ]
    arcs = []
    used: set[tuple[int, int]] = set()
    for center, radius in circles:
        nearby = [
            point
            for point in endpoints
            if abs(np.linalg.norm(np.asarray(point[::-1]) - center) - radius) < 12.0
            and point not in used
        ]
        if len(nearby) != 2:
            continue
        first, second = nearby
        first_branch = _endpoint_branch_direction(skeleton, first)
        second_branch = _endpoint_branch_direction(skeleton, second)
        first_contact = _tangent_contact(first, first_branch, center, radius)
        second_contact = _tangent_contact(second, second_branch, center, radius)
        angle0 = math.atan2(first_contact[1] - center[1], first_contact[0] - center[0])
        angle1 = math.atan2(
            second_contact[1] - center[1], second_contact[0] - center[0]
        )
        delta = _choose_arc_delta(angle0, angle1, -first_branch, second_branch)
        if abs(delta) < 0.35:
            continue
        angles = angle0 + np.linspace(0.0, delta, max(8, int(abs(delta) * radius / 3)))
        points = np.column_stack(
            (
                center[0] + (radius + 4.0) * np.cos(angles),
                center[1] + (radius + 4.0) * np.sin(angles),
            )
        )
        arcs.append((first, second, points))
        used.update((first, second))
    free = [
        point
        for point in endpoints
        if point not in used
        and all(
            np.linalg.norm(np.asarray(point[::-1]) - center) > radius + 12.0
            for center, radius in circles
        )
    ]
    if len(free) != 2 or not arcs or len(arcs) > 3:
        return None
    adjacency: dict[tuple[int, int], list[tuple[float, tuple[int, int]]]] = {}
    for y, x in np.column_stack(np.nonzero(skeleton)):
        current = (int(y), int(x))
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (
                    (dy or dx)
                    and 0 <= y + dy < skeleton.shape[0]
                    and 0 <= x + dx < skeleton.shape[1]
                    and skeleton[y + dy, x + dx]
                ):
                    neighbors.append((math.hypot(dy, dx), (int(y + dy), int(x + dx))))
        adjacency[current] = neighbors
    cache: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray | None] = {}

    def shortest(start: tuple[int, int], end: tuple[int, int]) -> np.ndarray | None:
        if (start, end) in cache:
            return cache[start, end]
        queue = [(0.0, start)]
        distances = {start: 0.0}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances.get(current):
                continue
            if current == end:
                break
            for cost, neighbor in adjacency[current]:
                candidate = distance + cost
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    parents[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        if end not in distances:
            cache[start, end] = None
            return None
        path = [end]
        while path[-1] != start:
            path.append(parents[path[-1]])
        path.reverse()
        result = np.asarray([(x, y) for y, x in path], dtype=np.float64)
        cache[start, end] = result
        return result

    best: np.ndarray | None = None
    best_length = math.inf
    for order in itertools.permutations(arcs):
        for directions in itertools.product((False, True), repeat=len(arcs)):
            path_parts = []
            current = free[0]
            valid = True
            for (first, second, arc), reverse in zip(order, directions):
                entry, exit = (second, first) if reverse else (first, second)
                route = shortest(current, entry)
                if route is None:
                    valid = False
                    break
                path_parts.extend((route, arc[::-1] if reverse else arc))
                current = exit
            tail = shortest(current, free[1]) if valid else None
            if tail is None:
                continue
            path_parts.append(tail)
            path = np.concatenate(path_parts)
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if length < best_length:
                best, best_length = path, length
    return best


def extract_rope_centerline(
    mask: np.ndarray,
    count: int = DEFAULT_NODE_COUNT,
    *,
    pulley_circles: list[tuple[np.ndarray, float]] | None = None,
    endpoint_hints: np.ndarray | None = None,
) -> np.ndarray:
    """Return an ordered, arc-length sampled centerline in pixel coordinates."""
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("rope mask must be a non-empty 2D array")
    y, x = np.nonzero(mask)
    margin = 3
    y0, y1 = (
        max(0, int(y.min()) - margin),
        min(mask.shape[0], int(y.max()) + margin + 1),
    )
    x0, x1 = (
        max(0, int(x.min()) - margin),
        min(mask.shape[1], int(x.max()) + margin + 1),
    )
    crop = mask[y0:y1, x0:x1].astype(bool)
    raw_skeleton = skeletonize(crop)
    circles = [
        (np.asarray(center, dtype=np.float64) - (x0, y0), radius)
        for center, radius in (pulley_circles or [])
    ]
    skeleton = None
    topology_path = _topology_aware_path(raw_skeleton, circles)
    if topology_path is None:
        skeleton = _connect_skeleton_components(raw_skeleton)
        topology_path = _topology_aware_path(skeleton, circles)
    if topology_path is not None and endpoint_hints is not None:
        ends = topology_path[[0, -1]] + (x0, y0)
        if (
            np.linalg.norm(ends[::-1] - endpoint_hints, axis=1).sum()
            < np.linalg.norm(ends - endpoint_hints, axis=1).sum()
        ):
            topology_path = topology_path[::-1].copy()
            ends = ends[::-1]
        if np.max(np.linalg.norm(ends - endpoint_hints, axis=1)) > 4.0:
            topology_path = None
    if topology_path is not None:
        topology_path += (x0, y0)
        return _reroute_inside_circles(
            _resample_polyline(topology_path, count),
            pulley_circles or [],
            thickness=2.0,
        )
    if skeleton is None:
        skeleton = _connect_skeleton_components(raw_skeleton)
    coordinates = np.column_stack(np.nonzero(skeleton))
    if len(coordinates) < 2:
        raise ValueError("failed to skeletonize rope mask")
    if endpoint_hints is None:
        start = tuple(int(value) for value in coordinates[0])
        first, _ = _dijkstra_farthest(skeleton, start, keep_parent=False)
        second, parents = _dijkstra_farthest(skeleton, first, keep_parent=True)
    else:
        local_hints = (np.asarray(endpoint_hints) - (x0, y0))[:, ::-1]
        nearest = cKDTree(coordinates).query(local_hints)[1]
        first, second = (tuple(int(v) for v in coordinates[i]) for i in nearest)
        _, parents = _dijkstra_farthest(skeleton, first, keep_parent=True)
    path = [second]
    while path[-1] != first:
        parent = parents.get(path[-1])
        if parent is None:
            raise ValueError("failed to recover rope skeleton path")
        path.append(parent)
    path.reverse()
    pixels = np.asarray([(point[1] + x0, point[0] + y0) for point in path])
    return _resample_polyline(pixels.astype(np.float64), count)


def _orient_centerlines(lines: list[np.ndarray]) -> np.ndarray:
    result = [lines[0]]
    for line in lines[1:]:
        previous = result[-1]
        direct = np.linalg.norm(line[0] - previous[0]) + np.linalg.norm(
            line[-1] - previous[-1]
        )
        reverse = np.linalg.norm(line[-1] - previous[0]) + np.linalg.norm(
            line[0] - previous[-1]
        )
        result.append(line[::-1].copy() if reverse < direct else line)
    return np.stack(result)


def _repair_short_centerlines(lines: list[np.ndarray]) -> np.ndarray:
    lengths = np.asarray(
        [np.linalg.norm(np.diff(line, axis=0), axis=1).sum() for line in lines]
    )
    valid = np.flatnonzero(lengths >= 0.8 * np.median(lengths))
    if len(valid) < 2 or len(valid) == len(lines):
        return _orient_centerlines(lines)
    ordered = _orient_centerlines([lines[index] for index in valid])
    result = np.empty((len(lines), *ordered.shape[1:]), dtype=np.float64)
    for node in range(ordered.shape[1]):
        for axis in range(2):
            result[:, node, axis] = np.interp(
                np.arange(len(lines)), valid, ordered[:, node, axis]
            )
    return result


def _object_distance(rope_mask: np.ndarray, object_mask: np.ndarray) -> float:
    if not np.any(object_mask):
        return math.inf
    distance = cv2.distanceTransform((~rope_mask).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.min(distance[object_mask]))


def _endpoint_attachment_evidence(
    endpoints: np.ndarray,
    centers: np.ndarray,
    observed: np.ndarray,
    half_extent: tuple[float, float],
) -> np.ndarray | None:
    """Require repeated surface proximity, not proximity to any rope segment."""
    relative = endpoints - centers
    extent = np.asarray(half_extent)
    tolerance = max(14.0, 0.45 * float(np.max(extent)))
    gap = np.linalg.norm(np.maximum(np.abs(relative) - extent, 0.0), axis=1)
    valid = observed & (gap <= tolerance)
    observed_count = int(np.count_nonzero(observed))
    support = int(np.count_nonzero(valid))
    if (
        observed_count == 0
        or support < min(3, observed_count)
        or support / observed_count < 0.5
    ):
        return None
    return np.median(relative[valid], axis=0)


def _surface_attachment(lines, endpoint, centers, observed, extent, radius, shape):
    """Infer the load face from the approaching rope, not its mask's last pixel."""
    if shape not in {"box", "sphere"}:
        return None
    extent = np.asarray(extent)
    directions = []
    for frame in np.flatnonzero(observed):
        ordered = lines[frame] if endpoint == 0 else lines[frame, ::-1]
        delta = ordered-centers[frame]
        outside = np.flatnonzero(np.linalg.norm(delta, axis=1) > 2*max(radius, float(extent.max())))
        if len(outside):
            direction = delta[outside[0]]
            directions.append(direction/np.linalg.norm(direction))
    if len(directions) < min(3, int(np.count_nonzero(observed))) or not directions:
        return None
    direction = np.median(directions, axis=0)
    if np.linalg.norm(direction) < .5:
        return None
    if shape == "sphere":
        return radius*direction/np.linalg.norm(direction)
    axis = int(np.argmax(np.abs(direction)))
    offset = np.zeros(2)
    offset[axis] = np.copysign(extent[axis], direction[axis])
    return offset


def _moving_load_evidence(centers, parent_centers, observed, extent, parent_radius):
    """Bound robust relative drift and require coherent motion below the pulley."""
    if np.count_nonzero(observed) < 10:
        return False
    load, pulley = centers[observed], parent_centers[observed]
    relative = load - pulley
    offset = np.median(relative, axis=0)
    scale = max(float(np.max(extent)), parent_radius)
    deviation = np.linalg.norm(relative-offset, axis=1)
    p50, p90 = np.percentile(deviation, [50, 90])
    # Pixel caps prevent large proxies from admitting unrelated trajectories.
    if p50 > min(15., max(6., .2*scale)) or p90 > min(30., max(10., .35*scale)):
        return False
    if not (0 < offset[1] <= parent_radius + 2*extent[1] + 60.
            and abs(offset[0]) <= max(12., .6*scale)):
        return False
    # Compare robust temporal blocks rather than single noisy endpoint frames.
    a = np.array([np.median(v, axis=0) for v in np.array_split(load, 5)])
    b = np.array([np.median(v, axis=0) for v in np.array_split(pulley, 5)])
    da, db = a[-1]-a[0], b[-1]-b[0]
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if min(na, nb) <= 8.:
        return False
    if np.dot(da, db)/(na*nb) < .95 or not .65 <= na/nb <= 1.5:
        return False
    changes_a, changes_b = np.diff(a, axis=0), np.diff(b, axis=0)
    significant = np.linalg.norm(changes_b, axis=1) > 3.
    return bool(np.any(significant) and np.all(
        np.sum(changes_a[significant]*changes_b[significant], axis=1) > 0))


def _match_rope_endpoints(candidates, lines, rigid_cache, pulley_ids):
    """Resolve competing endpoint candidates with temporal one-to-one evidence."""
    assignments = [None, None]
    frame = min(48, len(lines)-1)
    loads = [c for c in candidates if c[0]['target_id'] not in pulley_ids]
    for candidate in loads:
        distances = np.linalg.norm(lines[frame, (0, -1)]-candidate[1][frame], axis=1)
        end = int(np.argmin(distances))
        if assignments[end] is None or distances[end] < assignments[end][-1]:
            assignments[end] = (*candidate, float(distances[end]))
    assigned = sum(c is not None for c in assignments)
    # Do not turn occlusion at an already distinct attachment into a missing load.
    # Temporal reassignment is needed when candidates compete and leave an end empty.
    if assigned == 2 or len(loads) <= assigned:
        return assignments
    choices = [[None], [None]]
    for candidate in candidates:
        item, centers, observed, radius, motion = candidate
        if item['target_id'] in pulley_ids:
            continue
        extent = rigid_cache[item['target_id']][3]
        for end, node in enumerate((0, -1)):
            if _endpoint_attachment_evidence(lines[:, node], centers, observed, extent) is None:
                continue
            gaps = np.linalg.norm(np.maximum(np.abs(lines[:, node]-centers)-extent, 0.), axis=1)
            choices[end].append((*candidate, float(np.median(gaps[observed]))))
    pairs = [(a, b) for a in choices[0] for b in choices[1]
             if a is None or b is None or a[0]['target_id'] != b[0]['target_id']]
    matched = list(min(pairs, key=lambda pair: (
        -sum(c is not None for c in pair),
        sum(c[-1] for c in pair if c is not None),
        tuple(c[0]['target_id'] if c else '' for c in pair))))
    return matched if sum(c is not None for c in matched) > assigned else assignments


def _static_load_alignment(relative, extent, parent_radius):
    """A still load must be below and aligned with its guide, not merely nearby."""
    offset = np.median(relative, axis=0)
    gap = offset[1]-parent_radius-extent[1]
    return (abs(offset[0]) <= max(8., .35*parent_radius)
            and -8. <= gap <= max(40., .5*parent_radius))


def _recover_attached_endpoints(
    lines: np.ndarray,
    attachments: list[
        tuple[np.ndarray, np.ndarray, tuple[float, float], np.ndarray] | None
    ],
    masks: Mapping[int, np.ndarray],
    circles_by_frame: list[list[tuple[np.ndarray, float]]],
) -> np.ndarray:
    """Repair outlier boundary estimates; never copy a later rope configuration."""
    result = lines.copy()
    for frame, line in enumerate(lines):
        hints = line[[0, -1]].copy()
        changed = False
        for end, attachment in enumerate(attachments):
            if attachment is None:
                continue
            centers, observed, half_extent, relative = attachment
            if not observed[frame]:
                continue
            extent = np.asarray(half_extent)
            tolerance = max(14.0, 0.45 * float(np.max(extent)))
            gap = np.linalg.norm(
                np.maximum(np.abs(hints[end] - centers[frame]) - extent, 0.0)
            )
            if gap > tolerance:
                hints[end] = centers[frame] + relative
                changed = True
        mask = masks.get(frame)
        if changed and mask is not None and np.any(mask):
            repaired = extract_rope_centerline(
                mask,
                len(line),
                pulley_circles=circles_by_frame[frame],
                endpoint_hints=hints,
            )
            # No arbitrary long attachment offset if the inferred boundary is
            # absent from this frame's mask (e.g. hidden behind its mount).
            repaired[[0, -1]] = hints
            result[frame] = repaired
    return result


def _reroute_inside_circles(
    points: np.ndarray,
    circles: list[tuple[np.ndarray, float]],
    thickness: float,
) -> np.ndarray:
    result = points.copy()
    for center, radius in circles:
        safe_radius = radius + thickness
        inside = np.linalg.norm(result - center, axis=1) < safe_radius
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        for group in groups:
            lo, hi = max(0, int(group[0]) - 1), min(len(result) - 1, int(group[-1]) + 1)
            if lo == hi:
                continue
            a0 = math.atan2(*(result[lo] - center)[::-1])
            a1 = math.atan2(*(result[hi] - center)[::-1])
            incoming = result[lo] - result[lo - 1] if lo > 0 else np.zeros(2)
            outgoing = (
                result[hi + 1] - result[hi] if hi + 1 < len(result) else np.zeros(2)
            )
            delta = _choose_arc_delta(a0, a1, incoming, outgoing)
            angles = a0 + np.linspace(0.0, delta, hi - lo + 1)
            result[lo : hi + 1, 0] = center[0] + safe_radius * np.cos(angles)
            result[lo : hi + 1, 1] = center[1] + safe_radius * np.sin(angles)
    return _resample_polyline(result, len(points))


def _insert_forced_wrap(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
    turns: int,
) -> np.ndarray:
    if turns < 1:
        return points
    distances = np.linalg.norm(points - center, axis=1)
    indices = np.flatnonzero(np.abs(distances - radius) < 25.0)
    if len(indices) < 2:
        return points
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    group = max(groups, key=len)
    if len(group) < 2:
        return points
    lo, hi = int(group[0]), int(group[-1])
    a0 = math.atan2(points[lo, 1] - center[1], points[lo, 0] - center[0])
    a1 = math.atan2(points[hi, 1] - center[1], points[hi, 0] - center[0])
    shortest = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(shortest) > 1.3:
        return points
    direction = -1.0 if shortest > 0.0 else 1.0
    winding = shortest + direction * turns * 2.0 * math.pi
    angles = a0 + np.linspace(0.0, winding, max(24, int(abs(winding) * radius / 3.0)))
    arc = np.column_stack(
        (
            center[0] + (radius + 4.0) * np.cos(angles),
            center[1] + (radius + 4.0) * np.sin(angles),
        )
    )
    return _resample_polyline(
        np.concatenate((points[: lo + 1], arc, points[hi:])), len(points)
    )


def pulley_ids_for_source(source):
    rigid = {obj["target_id"]: obj for obj in source["objects"] if obj.get("physical_type") == "rigid"}
    if "rope_roles" not in source:
        return {name for name, obj in rigid.items() if obj.get("rigid_shape") == "cylinder"}
    ids = source["rope_roles"]["pulley_ids"]
    if not isinstance(ids, list) or any(not isinstance(name, str) for name in ids) or not set(ids) <= rigid.keys():
        raise ValueError("Pulley roles require retained rigid identities")
    return set(ids)


def load_rope_scenes(
    segmentation_dir: Path,
    *,
    node_count: int = DEFAULT_NODE_COUNT,
    topology_hints: Mapping[str, Mapping[str, int]] | None = None,
    roles: Mapping[str, Any] | None = None,
    tracking_policy: str = DEFAULT_TRACKING_POLICY,
    endpoint_policy: str = DEFAULT_ENDPOINT_POLICY,
) -> tuple[tuple[RopeSceneData, ...], dict[str, Any]]:
    segmentation_dir = Path(segmentation_dir)
    source = json.loads(
        (segmentation_dir / "pointcloud_tracking_input.json").read_text()
    )
    return prepare_rope_scenes(
        source,
        mask_archive=segmentation_dir / source["sam3_artifacts"]["mask_archive"],
        mask_index=segmentation_dir / source["sam3_artifacts"]["mask_index"],
        node_count=node_count, topology_hints=topology_hints, roles=roles,
        tracking_policy=tracking_policy, endpoint_policy=endpoint_policy,
    )


def prepare_rope_scenes(
    source: dict[str, Any], *, mask_archive: Path, mask_index: Path,
    node_count: int = DEFAULT_NODE_COUNT,
    topology_hints: Mapping[str, Mapping[str, int]] | None = None,
    roles: Mapping[str, Any] | None = None,
    tracking_policy: str = DEFAULT_TRACKING_POLICY,
    endpoint_policy: str = DEFAULT_ENDPOINT_POLICY,
) -> tuple[tuple[RopeSceneData, ...], dict[str, Any]]:
    """Prepare fitting targets from explicit video/object records and mask assets."""
    source = dict(source)
    if tracking_policy not in {"legacy", DEFAULT_TRACKING_POLICY}:
        raise ValueError(f"Unknown rope tracking policy: {tracking_policy}")
    robust_tracking = tracking_policy != "legacy"
    source["rope_tracking_policy"] = tracking_policy
    if endpoint_policy not in {"legacy", DEFAULT_ENDPOINT_POLICY}:
        raise ValueError(f"Unknown rope endpoint policy: {endpoint_policy}")
    source["rope_endpoint_policy"] = endpoint_policy
    if roles is not None:
        source["rope_roles"] = dict(roles)
    pulley_ids = pulley_ids_for_source(source)
    archive_path = Path(mask_archive)
    index_path = Path(mask_index)
    tracks = read_packed_mask_archive(archive_path, index_path)
    frame_count = int(source["video"]["frame_count"])
    width, height = int(source["video"]["width"]), int(source["video"]["height"])
    fps = float(source["video"]["fps"])
    objects = source["objects"]
    rigid = [item for item in objects if item.get("physical_type") == "rigid"]
    rope_objects = [item for item in objects if item.get("physical_type") == "rope"]
    rigid_cache = {}
    for item in rigid:
        item_track = tracks.get(item["track_id"], {})
        centers, observed, radius = _interpolate_centers(
            item_track,
            frame_count,
            circular=item["target_id"] in pulley_ids,
            reject_truncated=robust_tracking,
        )
        radius = source.get("pulley_radius_overrides_px", {}).get(item["target_id"], radius)
        rigid_cache[item["target_id"]] = (
            centers,
            observed,
            radius,
            _track_half_extent(item_track, frame_count, reject_truncated=robust_tracking),
        )
    scenes = []
    for rope in rope_objects:
        wraps = tuple(
            (str(body_id), int(turns))
            for body_id, turns in (topology_hints or {})
            .get(rope["target_id"], {})
            .items()
        )
        rope_track = tracks[rope["track_id"]]
        raw_lines = []
        last_line = None
        for frame in range(frame_count):
            mask = rope_track.get(frame)
            circles = [
                (
                    rigid_cache[item["target_id"]][0][frame],
                    rigid_cache[item["target_id"]][2],
                )
                for item in rigid
                if item["target_id"] in pulley_ids
            ]
            try:
                line = (
                    extract_rope_centerline(mask, node_count, pulley_circles=circles)
                    if mask is not None
                    else None
                )
            except ValueError:
                line = None
            if line is not None:
                for body_id, turns in wraps:
                    centers, _observed, radius, _extent = rigid_cache[body_id]
                    line = _insert_forced_wrap(line, centers[frame], radius, turns)
            if line is None:
                if last_line is None:
                    raise ValueError(f"{rope['target_id']} has no usable initial mask")
                line = last_line.copy()
            raw_lines.append(line)
            last_line = line
        lines_px = _repair_short_centerlines(raw_lines)
        representative_frames = sorted({0, min(48, frame_count - 1), frame_count - 1})
        candidates = []
        for item in rigid:
            item_track = tracks.get(item["track_id"], {})
            centers_px, observed, radius_px, _extent = rigid_cache[item["target_id"]]
            distances = []
            for frame in representative_frames:
                rope_mask = rope_track.get(frame)
                object_mask = item_track.get(frame)
                if rope_mask is not None and object_mask is not None:
                    distances.append(_object_distance(rope_mask, object_mask))
            if not distances or min(distances) > max(14.0, 0.45 * radius_px):
                continue
            baseline = np.median(centers_px[:8], axis=0)
            motion = (_track_motion(centers_px, observed) if robust_tracking else
                      float(np.max(np.linalg.norm(centers_px - baseline, axis=1))))
            candidates.append((item, centers_px, observed, radius_px, motion))

        endpoint_assignments = _match_rope_endpoints(
            candidates, lines_px, rigid_cache, pulley_ids)
        pulleys = [c for c in candidates if c[0]['target_id'] in pulley_ids]

        attachments = []
        for endpoint, assignment in enumerate(endpoint_assignments):
            if assignment is None:
                attachments.append(None)
                continue
            item, centers_px, observed, _radius, _motion, _score = assignment
            extent = rigid_cache[item["target_id"]][3]
            evidence = _endpoint_attachment_evidence(
                lines_px[:, -1 if endpoint else 0], centers_px, observed, extent
            )
            attachments.append(
                None if evidence is None else (centers_px, observed, extent, evidence)
            )
        circles_by_frame = [
            [
                (
                    rigid_cache[item["target_id"]][0][frame],
                    rigid_cache[item["target_id"]][2],
                )
                for item in rigid
                if item["target_id"] in pulley_ids
            ]
            for frame in range(frame_count)
        ]
        # Explicit wrap hints alter the path by construction; do not replace
        # those user-specified configurations with a skeleton shortest path.
        if not wraps:
            lines_px = _recover_attached_endpoints(
                lines_px, attachments, rope_track, circles_by_frame
            )
        if endpoint_policy != "legacy":
            for endpoint, assignment in enumerate(endpoint_assignments):
                if assignment is None or attachments[endpoint] is None:
                    continue
                item, centers, observed, radius, motion, _ = assignment
                if motion <= 8.0:
                    continue
                extent = rigid_cache[item["target_id"]][3]
                anchor = _surface_attachment(lines_px, endpoint, centers, observed,
                                             extent, radius, item.get("rigid_shape"))
                if anchor is not None:
                    lines_px[:, -1 if endpoint else 0] = centers+anchor

        bodies = []
        for endpoint, assignment in enumerate(endpoint_assignments):
            if assignment is None:
                continue
            item, centers_px, observed, radius_px, motion, _ = assignment
            dynamic = motion > 8.0
            particle = lines_px[0, -1 if endpoint else 0]
            offset_px = centers_px[0] - particle
            bodies.append(
                RopeBody(
                    object_id=item["target_id"],
                    track_id=item["track_id"],
                    role="endpoint",
                    shape=str(item.get("rigid_shape") or "unknown"),
                    dynamic=dynamic,
                    radius=radius_px / height,
                    initial_center=tuple(centers_px[0] / height),
                    target_centers=centers_px / height,
                    observed=observed,
                    endpoint_index=endpoint,
                    endpoint_offset=tuple(offset_px / height),
                    half_extent=tuple(
                        np.asarray(rigid_cache[item["target_id"]][3]) / height
                    ),
                )
            )
        for item, centers_px, observed, radius_px, motion in pulleys:
            dynamic = motion > 8.0
            initial_center = (np.median(centers_px[observed], axis=0)
                              if robust_tracking and not dynamic else centers_px[0])
            bodies.append(
                RopeBody(
                    object_id=item["target_id"],
                    track_id=item["track_id"],
                    role="pulley",
                    shape="cylinder",
                    dynamic=dynamic,
                    radius=radius_px / height,
                    initial_center=tuple(initial_center / height),
                    target_centers=centers_px / height,
                    observed=observed,
                    endpoint_index=-1,
                    endpoint_offset=(0.0, 0.0),
                    half_extent=tuple(
                        np.asarray(rigid_cache[item["target_id"]][3]) / height
                    ),
                )
            )

        modeled_ids = {body.object_id for body in bodies}
        for item in rigid:
            if item["target_id"] in modeled_ids or item.get("rigid_shape") not in (
                "box",
                "sphere",
            ):
                continue
            centers_px, observed, radius_px, extent = rigid_cache[item["target_id"]]
            motion = (_track_motion(centers_px, observed) if robust_tracking else
                      float(np.max(np.linalg.norm(centers_px - centers_px[0], axis=1))))
            parents = []
            for parent in bodies:
                if parent.role != "pulley" or parent.dynamic != (motion > 8.0):
                    continue
                parent_px = np.asarray(parent.target_centers) * height
                parent_observed = parent.observed & observed
                if parent_observed.sum() < 5:
                    continue
                relative = centers_px[parent_observed] - parent_px[parent_observed]
                distance = float(np.linalg.norm(centers_px[0] - parent_px[0]))
                if parent.dynamic:
                    attached = (distance <= parent.radius*height + 2*radius_px + 60.
                        and np.linalg.norm(relative.std(axis=0)) < 6.0)
                    if not attached:
                        attached = _moving_load_evidence(
                            centers_px, parent_px, parent_observed, extent, parent.radius*height)
                else:
                    attached = (distance <= parent.radius*height + 2*radius_px + 60.
                        and _static_load_alignment(relative, extent, parent.radius*height)
                        and np.linalg.norm(relative.std(axis=0)) < 6.0)
                if attached:
                    parents.append((distance, parent))
            if not parents:
                continue
            parent = min(parents, key=lambda value: value[0])[1]
            offset_px = centers_px[0] - np.asarray(parent.initial_center) * height
            bodies.append(
                RopeBody(
                    object_id=item["target_id"],
                    track_id=item["track_id"],
                    role="attachment",
                    shape=str(item.get("rigid_shape")),
                    dynamic=parent.dynamic,
                    radius=radius_px / height,
                    initial_center=tuple(centers_px[0] / height),
                    target_centers=centers_px / height,
                    observed=observed,
                    endpoint_index=-1,
                    endpoint_offset=tuple(offset_px / height),
                    half_extent=tuple(np.asarray(extent) / height),
                    parent_id=parent.object_id,
                    attachment_source="co-moving-load" if parent.dynamic else "aligned-static-load",
                )
            )

        # Factual fitting has no inferred mass-change event. Explicit edits may
        # select a different event offset in a copied simulation scene.
        event = 0
        endpoint_baseline = np.median(lines_px[:8, (0, -1)], axis=0)
        endpoint_motion = np.max(
            np.linalg.norm(lines_px[:, (0, -1)] - endpoint_baseline, axis=2),
            axis=0,
        )
        pinned = [bool(value < 8.0) for value in endpoint_motion]
        for body in bodies:
            if body.role == "endpoint" and body.dynamic:
                pinned[body.endpoint_index] = False
            elif body.role == "endpoint" and not body.dynamic:
                pinned[body.endpoint_index] = True
        initial_px = lines_px[0].copy()
        circles = [
            (np.asarray(body.initial_center) * height, body.radius * height)
            for body in bodies
            if body.role == "pulley"
            and body.object_id not in {item[0] for item in wraps}
        ]
        initial_px = _reroute_inside_circles(initial_px, circles, thickness=2.0)
        scenes.append(
            RopeSceneData(
                rope_id=rope["target_id"],
                rope_track_id=rope["track_id"],
                frames=tuple(range(frame_count)),
                fps=fps,
                image_width=width,
                image_height=height,
                event_frame_offset=event,
                target_positions=lines_px / height,
                initial_positions=initial_px / height,
                endpoint_pinned=(pinned[0], pinned[1]),
                bodies=tuple(bodies),
                source_video=str(source["video"]["path"]),
                mask_archive=str(archive_path),
                mask_index=str(index_path),
                wrap_hints=wraps,
            )
        )
    return tuple(scenes), source


@wp.kernel
def _simulate_rope_worlds(
    theta: wp.array2d(dtype=float),
    initial: wp.array(dtype=wp.vec2),
    rest_lengths: wp.array(dtype=float),
    particle_inverse_masses: wp.array(dtype=float),
    endpoint_pinned: wp.array(dtype=int),
    endpoint_mass_parameter_before: wp.array(dtype=int),
    endpoint_mass_parameter_after: wp.array(dtype=int),
    pulley_initial: wp.array(dtype=wp.vec2),
    pulley_radius: wp.array(dtype=float),
    pulley_dynamic: wp.array(dtype=int),
    pulley_mass_parameter_before: wp.array(dtype=int),
    pulley_mass_parameter_after: wp.array(dtype=int),
    system_dynamic: int,
    node_count: int,
    pulley_count: int,
    event_frame: int,
    frame_count: int,
    fps: float,
    substeps: int,
    iterations: int,
    thickness: float,
    positions: wp.array2d(dtype=wp.vec2),
    previous: wp.array2d(dtype=wp.vec2),
    velocities: wp.array2d(dtype=wp.vec2),
    pulley_positions: wp.array2d(dtype=wp.vec2),
    pulley_previous: wp.array2d(dtype=wp.vec2),
    pulley_velocities: wp.array2d(dtype=wp.vec2),
    predicted: wp.array3d(dtype=wp.vec2),
    predicted_pulleys: wp.array3d(dtype=wp.vec2),
    driven_node: int,
    drive_speed: float,
):
    world = wp.tid()
    dt = 1.0 / (fps * float(substeps))
    gravity = wp.exp(theta[world, 0])
    damping = wp.exp(-theta[world, 1] * dt)
    stiffness = theta[world, 2]
    for node in range(node_count):
        positions[world, node] = initial[node]
        tangent = wp.vec2(0.0, 0.0)
        if node == 0:
            tangent = wp.normalize(initial[1] - initial[0])
        elif node == node_count - 1:
            tangent = wp.normalize(initial[node] - initial[node - 1])
        else:
            tangent = wp.normalize(initial[node + 1] - initial[node - 1])
        velocities[world, node] = theta[world, 3] * tangent
    for pulley in range(pulley_count):
        pulley_positions[world, pulley] = pulley_initial[pulley]
        pulley_velocities[world, pulley] = wp.vec2(0.0, 0.0)
    for frame in range(frame_count):
        if frame > 0 and system_dynamic != 0:
            for _substep in range(substeps):
                for node in range(node_count):
                    if node == driven_node:
                        positions[world, node] += wp.vec2(0.0, drive_speed * dt)
                    previous[world, node] = positions[world, node]
                    pinned = (node == 0 and endpoint_pinned[0] != 0) or (
                        node == node_count - 1 and endpoint_pinned[1] != 0
                    ) or node == driven_node
                    if not pinned:
                        velocity = damping * velocities[world, node]
                        velocity[1] += dt * gravity
                        positions[world, node] += dt * velocity
                        velocities[world, node] = velocity
                for pulley in range(pulley_count):
                    pulley_previous[world, pulley] = pulley_positions[world, pulley]
                    if pulley_dynamic[pulley] != 0:
                        velocity = damping * pulley_velocities[world, pulley]
                        velocity[1] += dt * gravity
                        pulley_positions[world, pulley] += dt * velocity
                        pulley_velocities[world, pulley] = velocity
                for _iteration in range(iterations):
                    for edge in range(node_count - 1):
                        first, second = edge, edge + 1
                        delta = positions[world, second] - positions[world, first]
                        length = wp.length(delta)
                        if length > 1.0e-8:
                            first_inv = particle_inverse_masses[first]
                            second_inv = particle_inverse_masses[second]
                            if first == 0:
                                parameter = endpoint_mass_parameter_before[0]
                                if frame >= event_frame:
                                    parameter = endpoint_mass_parameter_after[0]
                                first_inv = (
                                    0.0
                                    if endpoint_pinned[0] != 0 or driven_node == 0
                                    else wp.exp(-theta[world, parameter])
                                )
                            if second == node_count - 1:
                                parameter = endpoint_mass_parameter_before[1]
                                if frame >= event_frame:
                                    parameter = endpoint_mass_parameter_after[1]
                                second_inv = (
                                    0.0
                                    if endpoint_pinned[1] != 0 or driven_node == node_count - 1
                                    else wp.exp(-theta[world, parameter])
                                )
                            total = first_inv + second_inv
                            if total > 1.0e-8:
                                correction = (
                                    stiffness
                                    * (length - rest_lengths[edge])
                                    * delta
                                    / length
                                )
                                positions[world, first] += (
                                    first_inv / total * correction
                                )
                                positions[world, second] -= (
                                    second_inv / total * correction
                                )
                    for node in range(node_count):
                        pinned = (node == 0 and endpoint_pinned[0] != 0) or (
                            node == node_count - 1 and endpoint_pinned[1] != 0
                        ) or node == driven_node
                        particle_inv = particle_inverse_masses[node]
                        if node == 0:
                            parameter = endpoint_mass_parameter_before[0]
                            if frame >= event_frame:
                                parameter = endpoint_mass_parameter_after[0]
                            particle_inv = (
                                0.0
                                if endpoint_pinned[0] != 0 or driven_node == 0
                                else wp.exp(-theta[world, parameter])
                            )
                        elif node == node_count - 1:
                            parameter = endpoint_mass_parameter_before[1]
                            if frame >= event_frame:
                                parameter = endpoint_mass_parameter_after[1]
                            particle_inv = (
                                0.0
                                if endpoint_pinned[1] != 0 or driven_node == node_count - 1
                                else wp.exp(-theta[world, parameter])
                            )
                        for pulley in range(pulley_count):
                            center = pulley_positions[world, pulley]
                            delta = positions[world, node] - center
                            distance = wp.length(delta)
                            minimum = pulley_radius[pulley] + thickness
                            if distance < minimum and distance > 1.0e-8:
                                normal = delta / distance
                                pulley_inv = 0.0
                                if pulley_dynamic[pulley] != 0:
                                    parameter = pulley_mass_parameter_before[pulley]
                                    if frame >= event_frame:
                                        parameter = pulley_mass_parameter_after[pulley]
                                    pulley_inv = wp.exp(-theta[world, parameter])
                                total = particle_inv + pulley_inv
                                if total > 1.0e-8:
                                    correction = (minimum - distance) * normal
                                    if not pinned:
                                        positions[world, node] += (
                                            particle_inv / total * correction
                                        )
                                    if pulley_dynamic[pulley] != 0:
                                        pulley_positions[world, pulley] -= (
                                            pulley_inv / total * correction
                                        )
                    for edge in range(node_count - 1):
                        first, second = edge, edge + 1
                        first_inv = particle_inverse_masses[first]
                        second_inv = particle_inverse_masses[second]
                        if first == 0:
                            parameter = endpoint_mass_parameter_before[0]
                            if frame >= event_frame:
                                parameter = endpoint_mass_parameter_after[0]
                            first_inv = (
                                0.0
                                if endpoint_pinned[0] != 0 or driven_node == 0
                                else wp.exp(-theta[world, parameter])
                            )
                        if second == node_count - 1:
                            parameter = endpoint_mass_parameter_before[1]
                            if frame >= event_frame:
                                parameter = endpoint_mass_parameter_after[1]
                            second_inv = (
                                0.0
                                if endpoint_pinned[1] != 0 or driven_node == node_count - 1
                                else wp.exp(-theta[world, parameter])
                            )
                        start = positions[world, first]
                        direction = positions[world, second] - start
                        length_sq = wp.dot(direction, direction)
                        if length_sq < 1.0e-10:
                            continue
                        for pulley in range(pulley_count):
                            center = pulley_positions[world, pulley]
                            t = wp.clamp(
                                wp.dot(center - start, direction) / length_sq, 0.0, 1.0
                            )
                            if t <= 0.02 or t >= 0.98:
                                continue
                            closest = start + t * direction
                            delta = closest - center
                            distance = wp.length(delta)
                            minimum = pulley_radius[pulley] + thickness
                            if distance >= minimum or distance <= 1.0e-8:
                                continue
                            pulley_inv = 0.0
                            if pulley_dynamic[pulley] != 0:
                                parameter = pulley_mass_parameter_before[pulley]
                                if frame >= event_frame:
                                    parameter = pulley_mass_parameter_after[pulley]
                                pulley_inv = wp.exp(-theta[world, parameter])
                            first_share = 1.0 - t
                            total = (
                                first_inv * first_share * first_share
                                + second_inv * t * t
                                + pulley_inv
                            )
                            if total <= 1.0e-8:
                                continue
                            correction = (minimum - distance) / total * delta / distance
                            positions[world, first] += (
                                first_inv * first_share * correction
                            )
                            positions[world, second] += second_inv * t * correction
                            if pulley_dynamic[pulley] != 0:
                                pulley_positions[world, pulley] -= (
                                    pulley_inv * correction
                                )
                for node in range(node_count):
                    pinned = (node == 0 and endpoint_pinned[0] != 0) or (
                        node == node_count - 1 and endpoint_pinned[1] != 0
                    ) or node == driven_node
                    if not pinned:
                        velocities[world, node] = (
                            positions[world, node] - previous[world, node]
                        ) / dt
                for pulley in range(pulley_count):
                    if pulley_dynamic[pulley] != 0:
                        pulley_velocities[world, pulley] = (
                            pulley_positions[world, pulley]
                            - pulley_previous[world, pulley]
                        ) / dt
        for node in range(node_count):
            predicted[world, frame, node] = positions[world, node]
        for pulley in range(pulley_count):
            predicted_pulleys[world, frame, pulley] = pulley_positions[world, pulley]


def rope_particle_inverse_masses(
    rest_lengths: np.ndarray, linear_density: float | None = None,
) -> np.ndarray:
    rest = np.asarray(rest_lengths, dtype=np.float64)
    if rest.ndim != 1 or not len(rest) or not np.isfinite(rest).all() or np.any(rest <= 0):
        raise ValueError("rope rest lengths must be finite and positive")
    if linear_density is None:
        return np.full(len(rest) + 1, ROPE_PARTICLE_INVERSE_MASS, np.float32)
    if not math.isfinite(linear_density) or linear_density <= 0:
        raise ValueError("rope linear density must be finite and positive")
    # Interior nodes represent half of each adjacent segment. Endpoint inverse
    # masses are overridden in the solver by fitted total endpoint/body masses.
    dual_lengths = np.r_[rest, 0.] * .5 + np.r_[0., rest] * .5
    return (1. / (linear_density * dual_lengths)).astype(np.float32)


class BatchWarpRopeEngine:
    def __init__(
        self,
        scene: RopeSceneData,
        *,
        capacity: int,
        device: str = "cuda:0",
        substeps: int = 8,
        constraint_iterations: int = 6,
        simulate_static_system: bool = False,
        rope_linear_density: float | None = None,
        block_dim: int = DEFAULT_CUDA_BLOCK_DIM,
    ) -> None:
        if not 1 <= block_dim <= 1024:
            raise ValueError("block_dim must be between 1 and 1024")
        self.scene = scene
        self.capacity = capacity
        self.block_dim = int(block_dim)
        self.device = device
        self.torch_device = torch.device(device)
        self.substeps = substeps
        self.constraint_iterations = constraint_iterations
        self.simulate_static_system = simulate_static_system
        self.pulleys = tuple(body for body in scene.bodies if body.role == "pulley")
        dynamic_entities = 0
        endpoint_parameters_before = [-1, -1]
        endpoint_parameters_after = [-1, -1]
        for endpoint in range(2):
            if not scene.endpoint_pinned[endpoint]:
                endpoint_parameters_before[endpoint] = 4 + 2 * dynamic_entities
                endpoint_parameters_after[endpoint] = 5 + 2 * dynamic_entities
                dynamic_entities += 1
        pulley_parameters_before = []
        pulley_parameters_after = []
        for body in self.pulleys:
            pulley_parameters_before.append(
                4 + 2 * dynamic_entities if body.dynamic else -1
            )
            pulley_parameters_after.append(
                5 + 2 * dynamic_entities if body.dynamic else -1
            )
            dynamic_entities += int(body.dynamic)
        self.parameter_count = 4 + 2 * dynamic_entities
        rest = np.linalg.norm(np.diff(scene.initial_positions, axis=0), axis=1)
        self.rope_linear_density = rope_linear_density
        node_inverse_masses = rope_particle_inverse_masses(rest, rope_linear_density)
        pulley_storage = max(1, len(self.pulleys))
        pulley_initial = np.zeros((pulley_storage, 2), dtype=np.float32)
        pulley_radius = np.zeros(pulley_storage, dtype=np.float32)
        pulley_dynamic = np.zeros(pulley_storage, dtype=np.int32)
        pulley_parameter_before_array = np.full(pulley_storage, -1, dtype=np.int32)
        pulley_parameter_after_array = np.full(pulley_storage, -1, dtype=np.int32)
        for index, body in enumerate(self.pulleys):
            pulley_initial[index] = body.initial_center
            pulley_radius[index] = body.radius
            pulley_dynamic[index] = int(body.dynamic)
            pulley_parameter_before_array[index] = pulley_parameters_before[index]
            pulley_parameter_after_array[index] = pulley_parameters_after[index]
        self.theta = wp.empty(
            (capacity, self.parameter_count), dtype=float, device=device
        )
        self.initial = wp.array(
            scene.initial_positions.astype(np.float32), dtype=wp.vec2, device=device
        )
        self.rest = wp.array(rest.astype(np.float32), dtype=float, device=device)
        self.particle_inverse_masses = wp.array(node_inverse_masses, dtype=float, device=device)
        self.endpoint_pinned = wp.array(
            np.asarray(scene.endpoint_pinned, np.int32), dtype=int, device=device
        )
        self.endpoint_mass_parameter_before = wp.array(
            np.asarray(endpoint_parameters_before, np.int32), dtype=int, device=device
        )
        self.endpoint_mass_parameter_after = wp.array(
            np.asarray(endpoint_parameters_after, np.int32), dtype=int, device=device
        )
        self.pulley_initial = wp.array(pulley_initial, dtype=wp.vec2, device=device)
        self.pulley_radius = wp.array(pulley_radius, dtype=float, device=device)
        self.pulley_dynamic = wp.array(pulley_dynamic, dtype=int, device=device)
        self.pulley_mass_parameter_before = wp.array(
            pulley_parameter_before_array, dtype=int, device=device
        )
        self.pulley_mass_parameter_after = wp.array(
            pulley_parameter_after_array, dtype=int, device=device
        )
        state_shape = (capacity, scene.node_count)
        pulley_shape = (capacity, pulley_storage)
        self.positions = wp.empty(state_shape, dtype=wp.vec2, device=device)
        self.previous = wp.empty(state_shape, dtype=wp.vec2, device=device)
        self.velocities = wp.empty(state_shape, dtype=wp.vec2, device=device)
        self.pulley_positions = wp.empty(pulley_shape, dtype=wp.vec2, device=device)
        self.pulley_previous = wp.empty(pulley_shape, dtype=wp.vec2, device=device)
        self.pulley_velocities = wp.empty(pulley_shape, dtype=wp.vec2, device=device)
        self.predicted = wp.empty(
            (capacity, scene.frame_count, scene.node_count),
            dtype=wp.vec2,
            device=device,
        )
        self.predicted_pulleys = wp.empty(
            (capacity, scene.frame_count, pulley_storage), dtype=wp.vec2, device=device
        )
        self.theta_torch = wp.to_torch(self.theta)
        self.predicted_torch = wp.to_torch(self.predicted)
        self.predicted_pulleys_torch = wp.to_torch(self.predicted_pulleys)
        self.target = torch.as_tensor(
            scene.target_positions, dtype=torch.float32, device=device
        )
        self.body_targets = {
            body.object_id: torch.as_tensor(
                body.target_centers, dtype=torch.float32, device=device
            )
            for body in scene.bodies
            if body.dynamic
        }

    def rollout(
        self, values: torch.Tensor | np.ndarray, end_offset: int | None = None,
        *, endpoint_motion: tuple[int, float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = torch.as_tensor(values, dtype=torch.float32, device=self.torch_device)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        count = len(samples)
        if count > self.capacity or samples.shape[1] != self.parameter_count:
            raise ValueError("invalid rope parameter batch")
        final = (
            self.scene.frame_count - 1
            if end_offset is None
            else min(int(end_offset), self.scene.frame_count - 1)
        )
        self._driven_node, self._drive_speed = -1, 0.
        if endpoint_motion is not None:
            endpoint, speed = endpoint_motion
            if endpoint not in (0, 1) or not np.isfinite(speed):
                raise ValueError("Drive requires endpoint 0/1 and finite speed")
            self._driven_node = 0 if endpoint == 0 else self.scene.node_count - 1
            self._drive_speed = float(speed)
        self.theta_torch[:count].copy_(samples)
        self._launch(count, final)
        return (
            self.predicted_torch[:count, : final + 1],
            self.predicted_pulleys_torch[:count, : final + 1, : len(self.pulleys)],
        )

    def _launch(self, count: int, final: int) -> None:
        wp.launch(
            _simulate_rope_worlds,
            dim=count,
            block_dim=self.block_dim,
            device=self.device,
            inputs=self._kernel_inputs(final),
        )

    @property
    def execution_config(self) -> dict:
        return dict(solver="monolithic", cuda_block_dim=self.block_dim,
                    stretch_order="original", contact_order="original", cuda_graphs=False)

    def _kernel_inputs(self, final: int) -> list:
        return [
            self.theta,
            self.initial,
            self.rest,
            self.particle_inverse_masses,
            self.endpoint_pinned,
            self.endpoint_mass_parameter_before,
            self.endpoint_mass_parameter_after,
            self.pulley_initial,
            self.pulley_radius,
            self.pulley_dynamic,
            self.pulley_mass_parameter_before,
            self.pulley_mass_parameter_after,
            int(getattr(self, "_driven_node", -1) >= 0 or self.simulate_static_system or any(
                body.dynamic for body in self.scene.bodies
            )),
            self.scene.node_count,
            len(self.pulleys),
            self.scene.event_frame_offset,
            final + 1,
            self.scene.fps,
            self.substeps,
            self.constraint_iterations,
            2.0 / self.scene.image_height,
            self.positions,
            self.previous,
            self.velocities,
            self.pulley_positions,
            self.pulley_previous,
            self.pulley_velocities,
            self.predicted,
            self.predicted_pulleys,
            getattr(self, "_driven_node", -1),
            getattr(self, "_drive_speed", 0.),
        ]

    def predicted_body_centers(
        self,
        rope: torch.Tensor,
        pulleys: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = {}
        pulley_lookup = {
            body.object_id: index for index, body in enumerate(self.pulleys)
        }
        for body in self.scene.bodies:
            if not body.dynamic:
                continue
            if body.role == "pulley":
                result[body.object_id] = pulleys[:, :, pulley_lookup[body.object_id]]
            elif body.role == "attachment":
                offset = torch.tensor(body.endpoint_offset, device=self.torch_device)
                result[body.object_id] = (
                    pulleys[:, :, pulley_lookup[body.parent_id]] + offset
                )
            else:
                node = 0 if body.endpoint_index == 0 else self.scene.node_count - 1
                offset = torch.tensor(body.endpoint_offset, device=self.torch_device)
                result[body.object_id] = rope[:, :, node] + offset
        return result


def parameter_bounds(parameter_count: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(parameter_count, math.log(0.03), dtype=np.float32)
    upper = np.full(parameter_count, math.log(100.0), dtype=np.float32)
    lower[0], upper[0] = math.log(0.03), math.log(4.0)
    lower[1], upper[1] = 0.0, 3.0
    lower[2], upper[2] = 0.82, 1.0
    lower[3], upper[3] = -0.5, 0.5
    return lower, upper


def initial_distribution(parameter_count: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(parameter_count, math.log(1.0), dtype=np.float32)
    std = np.full(parameter_count, 1.0, dtype=np.float32)
    mean[0], std[0] = math.log(0.7), 0.7
    mean[1], std[1] = 0.15, 0.25
    mean[2], std[2] = 0.97, 0.04
    mean[3], std[3] = 0.0, 0.08
    return mean, std


def constant_mass_parameter_indices(parameter_count: int) -> np.ndarray:
    """Optimize one mass per body; paired kernel slots remain for explicit edits."""
    if parameter_count < 4 or (parameter_count - 4) % 2:
        raise ValueError('Expected four global parameters and paired mass slots')
    return np.r_[np.arange(4), np.arange(4, parameter_count, 2)]


def expand_constant_masses(values):
    """Expand compact CEM samples without introducing an independent second mass."""
    if isinstance(values, torch.Tensor):
        return torch.cat((values[..., :4], values[..., 4:].repeat_interleave(2, dim=-1)), dim=-1)
    values = np.asarray(values)
    return np.concatenate((values[..., :4], np.repeat(values[..., 4:], 2, axis=-1)), axis=-1)


def _candidate_loss(
    engine: BatchWarpRopeEngine,
    rope: torch.Tensor,
    pulleys: torch.Tensor,
    *,
    end_offset: int,
    config: RopeCEMConfig,
) -> torch.Tensor:
    target = engine.target[: end_offset + 1]
    pair_distance = torch.cdist(rope, target.unsqueeze(0))
    distance = torch.cat(
        (pair_distance.min(dim=-1).values, pair_distance.min(dim=-2).values),
        dim=-1,
    )
    delta = config.huber_delta
    loss = torch.where(
        distance <= delta, 0.5 * distance.square(), delta * (distance - 0.5 * delta)
    ).mean(dim=(1, 2))
    for object_id, predicted in engine.predicted_body_centers(rope, pulleys).items():
        body_distance = torch.linalg.vector_norm(
            predicted - engine.body_targets[object_id][: end_offset + 1].unsqueeze(0),
            dim=-1,
        )
        body_loss = torch.where(
            body_distance <= delta,
            0.5 * body_distance.square(),
            delta * (body_distance - 0.5 * delta),
        ).mean(dim=1)
        loss += config.body_weight * body_loss
    return loss


def centerline_chamfer_rmse(target: np.ndarray, predicted: np.ndarray) -> float:
    if target.shape != predicted.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("centerline arrays must have matching [frame, node, 2] shapes")
    squared = []
    for target_frame, predicted_frame in zip(target, predicted):
        delta = target_frame[:, None, :] - predicted_frame[None, :, :]
        distance_squared = np.sum(delta * delta, axis=-1)
        squared.extend(np.min(distance_squared, axis=0))
        squared.extend(np.min(distance_squared, axis=1))
    return float(np.sqrt(np.mean(squared)))


def run_cem(
    engine: BatchWarpRopeEngine, config: RopeCEMConfig, *,
    verbose: bool = True,
    loss_fn: Callable[..., torch.Tensor] | None = None,
) -> dict[str, Any]:
    lower_np, upper_np = parameter_bounds(engine.parameter_count)
    mean_np, std_np = initial_distribution(engine.parameter_count)
    for index, value in getattr(engine, "fixed_parameters", {}).items():
        lower_np[index] = upper_np[index] = mean_np[index] = value
        std_np[index] = 0.
    indices = constant_mass_parameter_indices(engine.parameter_count)
    lower_np, upper_np = lower_np[indices], upper_np[indices]
    mean_np, std_np = mean_np[indices], std_np[indices]
    lower = torch.as_tensor(lower_np, device=engine.torch_device)
    upper = torch.as_tensor(upper_np, device=engine.torch_device)
    mean = torch.as_tensor(mean_np, device=engine.torch_device)
    std = torch.as_tensor(std_np, device=engine.torch_device)
    minimum_std = config.exploration_fraction * (upper - lower)
    generator = torch.Generator(device=engine.torch_device).manual_seed(config.seed)
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    best, best_loss, trace = mean.clone(), math.inf, []
    # Keep the stage budget while decoupling the curriculum from mass edits.
    prefixes = np.linspace(0, engine.scene.frame_count, len(config.prefix_frames) + 2,
                           dtype=int)[1:] - 1
    prefixes = np.maximum(prefixes, 0)
    for stage, end_offset in enumerate(dict.fromkeys(prefixes), start=1):
        stage_best, stage_loss = best.clone(), math.inf
        for _generation in range(config.generations_per_prefix):
            samples = (
                mean
                + std
                * torch.randn(
                    (config.population, len(indices)),
                    generator=generator,
                    device=engine.torch_device,
                )
            ).clamp_(lower, upper)
            samples[0] = best
            predicted, pulleys = engine.rollout(expand_constant_masses(samples), end_offset)
            losses = (loss_fn or _candidate_loss)(
                engine, predicted, pulleys, end_offset=end_offset, config=config
            )
            losses = torch.nan_to_num(losses, nan=math.inf, posinf=math.inf, neginf=math.inf)
            elite_losses, elite_indices = torch.topk(losses, elite_count, largest=False)
            elites = samples[elite_indices]
            mean = config.distribution_smoothing * mean + (
                1.0 - config.distribution_smoothing
            ) * elites.mean(0)
            std = config.distribution_smoothing * std + (
                1.0 - config.distribution_smoothing
            ) * elites.std(0, unbiased=False)
            std = torch.maximum(std, minimum_std)
            if elite_losses[0] < stage_loss:
                stage_loss = float(elite_losses[0].item())
                stage_best = samples[elite_indices[0]].clone()
        best, mean, best_loss = stage_best, stage_best.clone(), stage_loss
        entry = {
            "stage": stage,
            "end_frame": engine.scene.frames[end_offset],
            "best_loss": best_loss,
        }
        trace.append(entry)
        if verbose:
            print(
                json.dumps(
                    {"phase": "rope_cem", "rope_id": engine.scene.rope_id, **entry}
                ),
                flush=True,
            )
    return {"best": expand_constant_masses(best).detach().cpu().numpy(),
            "best_loss": best_loss, "trace": trace,
            "mass_schedule": "constant", "optimized_parameter_count": int(np.count_nonzero(upper_np > lower_np))}


def make_rope_engine(
    scene: RopeSceneData, *, execution_mode: str = DEFAULT_EXECUTION_MODE, **options: Any,
) -> BatchWarpRopeEngine:
    if execution_mode == "auto":
        execution_mode = "ideal-warp" if scene.mass_model == "massless" else "guarded-chunked"
    if scene.mass_model == "massless":
        if execution_mode == "ideal-warp":
            from atw.polyworld.rope_ideal_warp import WarpIdealRopeEngine
            return WarpIdealRopeEngine(scene, visits=scene.route_visits, **options)
        from atw.polyworld.rope_ideal import IdealRopeEngine
        return IdealRopeEngine(scene, visits=scene.route_visits, **options)
    if execution_mode == 'ideal-warp':
        raise ValueError('ideal-warp execution requires the massless rope model')
    if execution_mode == "monolithic":
        return BatchWarpRopeEngine(scene, **options)
    if execution_mode in {"guarded-colored", "guarded-chunked"}:
        from atw.polyworld.rope_swr_parallel import SplitRopeEngine

        if execution_mode == "guarded-chunked":
            options.setdefault("graph_chunk_frames", 16)
        return SplitRopeEngine(scene, colored=True, parallel_static_contacts=True,
                               preserve_coupled_contacts=True, **options)
    raise ValueError(f"unknown rope execution mode: {execution_mode}")


def result_payload(
    scene: RopeSceneData,
    config: RopeCEMConfig,
    fit: dict[str, Any],
    elapsed: float,
    *,
    device: str,
    substeps: int,
) -> dict[str, Any]:
    theta = fit["best"]
    constant_masses = np.array_equal(theta[4::2], theta[5::2])
    masses_before = {}
    masses_after = {}
    parameter = 4
    for endpoint in range(2):
        if not scene.endpoint_pinned[endpoint]:
            masses_before[f"endpoint_{endpoint}"] = float(math.exp(theta[parameter]))
            masses_after[f"endpoint_{endpoint}"] = float(math.exp(theta[parameter + 1]))
            parameter += 2
    for body in (
        item for item in scene.bodies if item.role == "pulley" and item.dynamic
    ):
        masses_before[body.object_id] = float(math.exp(theta[parameter]))
        masses_after[body.object_id] = float(math.exp(theta[parameter + 1]))
        parameter += 2
    body_predictions = {
        object_id: [
            {"frame_index": frame, "xy_normalized": position.tolist()}
            for frame, position in zip(scene.frames, values)
        ]
        for object_id, values in fit["predicted_bodies"].items()
    }
    return {
        "schema": "physmind.rope_swr",
        "schema_version": 1,
        "model": ("physmind-v2-ideal-massless-rope-v1" if scene.mass_model == "massless"
                  else "physmind-v2-warp-rope-swr-v1"),
        "rope_id": scene.rope_id,
        "node_count": scene.node_count,
        "coordinate_system": {
            "x": "image_right",
            "y": "image_down",
            "scale": "image_height",
        },
        "gravity_direction": [0.0, 1.0],
        "mass_schedule": "constant" if constant_masses else "legacy_piecewise",
        "event_frame": None if constant_masses else scene.frames[scene.event_frame_offset],
        "event_type": None if constant_masses else "mass_intervention",
        "endpoint_pinned": list(scene.endpoint_pinned),
        "parameters": {
            **({"rope_mass": 0., "pulley_masses": {b.object_id: 0. for b in scene.bodies
                if b.role == "pulley"}, "mass_semantics": "attached-load-only"}
               if scene.mass_model == "massless" else {}),
            "gravity_normalized_per_s2": float(math.exp(theta[0])),
            "linear_drag_per_s": float(theta[1]),
            "distance_stiffness": float(theta[2]),
            "initial_rope_speed_normalized_per_s": float(theta[3]),
            "relative_masses_before": masses_before,
            "relative_masses_after": masses_after,
            **({"relative_masses": masses_before} if constant_masses else {}),
        },
        "bodies": [
            {
                "object_id": body.object_id,
                "role": body.role,
                "shape": body.shape,
                "dynamic": body.dynamic,
                "radius_normalized": body.radius,
                "half_extent_normalized": body.half_extent,
                "parent_id": body.parent_id,
                "attachment_source": body.attachment_source,
                "default_relative_mass": 0. if body.role == "pulley" and scene.mass_model == "massless" else body.default_relative_mass,
                "initial_center_normalized": list(body.initial_center),
                "endpoint_offset_normalized": list(body.endpoint_offset),
            }
            for body in scene.bodies
        ],
        "predicted_centerline": [
            {"frame_index": frame, "xy_normalized": positions.tolist()}
            for frame, positions in zip(scene.frames, fit["predicted"])
        ],
        "predicted_body_centers": body_predictions,
        "constraint_diagnostics": fit.get('constraint_diagnostics', {}),
        "metrics": {
            "centerline_chamfer_rmse_normalized": fit["centerline_rmse"],
            "centerline_chamfer_rmse_pixels": fit["centerline_rmse"]
            * scene.image_height,
            "dynamic_body_rmse_normalized": fit["body_rmse"],
            "dynamic_body_rmse_pixels": fit["body_rmse"] * scene.image_height,
        },
        "optimization": {
            "seed": config.seed,
            "trace": fit["trace"],
            "elapsed_seconds": elapsed,
            "device": device,
            "substeps": substeps,
            "execution_config": fit.get("execution_config", {}),
            "mass_schedule": "constant" if constant_masses else "legacy_piecewise",
            "optimized_parameter_count": fit.get("optimized_parameter_count",
                (4 + len(masses_before)) if constant_masses else len(theta)),
            "prefix_schedule": "uniform-full-sequence" if fit.get('mass_schedule') == 'constant' else None,
        },
        "source": {
            "video": scene.source_video,
            "mask_archive": scene.mask_archive,
            "mask_index": scene.mask_index,
            "rope_track_id": scene.rope_track_id,
        },
    }
