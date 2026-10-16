"""Discrete rope routing with raw, visibility-aware observations.

Only initial geometry is reconstructed. Subsequent frames are open-loop Warp
rollouts, not independently rerouted or corrected from observations.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import ndtri
from scipy.stats import qmc
from skimage.morphology import skeletonize

from atw.perception.object_reconstruction import read_packed_mask_archive
from atw.polyworld.rope_swr import DEFAULT_EXECUTION_MODE, DEFAULT_TRACKING_POLICY, RopeBody, RopeCEMConfig, RopeSceneData, ROPE_PARTICLE_INVERSE_MASS, _interpolate_centers, _track_motion, _resample_polyline, centerline_chamfer_rmse, initial_distribution, parameter_bounds, pulley_ids_for_source, make_rope_engine, run_cem


@dataclass
class RopeObservations:
    frames: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    free_distance: np.ndarray
    initial_points: np.ndarray
    initial_free_distance: np.ndarray
    image_height: int
    image_width: int
    visible_lengths: np.ndarray
    initial_route_outlier_pixels: int = 0
    route_tolerance_pixels: float = 2.0


@dataclass
class RouteCandidate:
    name: str
    positions: np.ndarray
    visits: tuple[tuple[str, int, int], ...]
    geometry_loss: float
    length: float


DEFAULT_LENGTH_PRIOR_WEIGHT = 0.1
DEFAULT_MASS_MODEL = "massless"
MASS_REFERENCE_NODE_COUNT = 96


def route_selection_score(
    observation_loss: float,
    length: float,
    visible_length: float,
    *,
    weight: float = DEFAULT_LENGTH_PRIOR_WEIGHT,
    huber_delta: float = 0.02,
) -> dict[str, float]:
    if (
        not all(math.isfinite(x) for x in (length, visible_length, weight, huber_delta))
        or length <= 0
        or visible_length <= 0
        or weight < 0
        or huber_delta <= 0
    ):
        raise ValueError("invalid length-prior settings")
    # This is a complexity prior, not measured invisible length or a physical
    # constraint: the visible skeleton is a lower bound affected by occlusion.
    excess = max(0.0, length - visible_length) / visible_length
    penalty = weight * huber_delta**2 * excess
    return {
        "length_excess_ratio": excess,
        "length_prior_penalty": penalty,
        "selection_loss": observation_loss + penalty,
    }


def _skeleton_length(skeleton: np.ndarray) -> float:
    horizontal = np.count_nonzero(skeleton[:, :-1] & skeleton[:, 1:])
    vertical = np.count_nonzero(skeleton[:-1] & skeleton[1:])
    diagonal_a = (
        skeleton[:-1, :-1] & skeleton[1:, 1:] & ~skeleton[:-1, 1:] & ~skeleton[1:, :-1]
    )
    diagonal_b = (
        skeleton[:-1, 1:] & skeleton[1:, :-1] & ~skeleton[:-1, :-1] & ~skeleton[1:, 1:]
    )
    return float(
        horizontal + vertical + math.sqrt(2.0) * (diagonal_a.sum() + diagonal_b.sum())
    )


def _initial_route_mask(
    mask: np.ndarray, track: dict, other_masks: list[np.ndarray]
) -> np.ndarray:
    """Exclude small cross-rope fragments contradicted by both following frames.

    This only prepares route proposals. Raw observations remain the fitting target.
    """
    following = [track.get(frame) for frame in (1, 2)]
    if not other_masks or any(
        value is None or not np.any(value) for value in following
    ):
        return mask
    others = np.logical_or.reduce(other_masks)
    if not np.any(others):
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    small = [
        i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < 0.05 * mask.sum()
    ]
    if not small:
        return mask
    future_distance = [
        cv2.distanceTransform((~value.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
        for value in following
    ]
    other_distance = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 3)
    result = mask.copy()
    for label in small:
        region = labels == label
        if np.median(other_distance[region]) <= max(3.0, 0.005 * mask.shape[0]) and all(
            np.median(distance[region]) > max(14.0, 0.03 * mask.shape[0])
            for distance in future_distance
        ):
            result[region] = False
    return result


def load_observations(
    scene: RopeSceneData,
    tracks: dict,
    source: dict,
    *,
    stride: int = 6,
    point_count: int = 192,
    map_scale: int = 4,
) -> RopeObservations:
    if stride < 1 or point_count < 2 or map_scale < 1:
        raise ValueError("observation sampling settings must be positive")
    frames = np.unique(
        np.r_[np.arange(0, scene.frame_count, stride), scene.frame_count - 1]
    )
    points = np.zeros((len(frames), point_count, 2), np.float32)
    valid = np.zeros((len(frames), point_count), bool)
    maps = []
    visible_lengths = []
    initial_points = None
    initial_route_outlier_pixels = 0
    route_tolerance_pixels = 2.0
    rigid_tracks = [
        item["track_id"]
        for item in source["objects"]
        if item.get("physical_type") == "rigid"
    ]
    for offset, frame in enumerate(frames):
        mask = tracks.get(scene.rope_track_id, {}).get(int(frame))
        shape = (scene.image_height, scene.image_width)
        mask = np.zeros(shape, bool) if mask is None else mask.astype(bool)
        skeleton = skeletonize(mask)
        ys, xs = np.nonzero(skeleton)
        visible_lengths.append(
            max(_skeleton_length(skeleton), float(bool(len(xs)))) / scene.image_height
        )
        observed = np.column_stack((xs, ys)).astype(np.float32)
        if offset == 0:
            other_masks = [
                tracks[item["track_id"]][0]
                for item in source["objects"]
                if item.get("physical_type") == "rope"
                and item["track_id"] != scene.rope_track_id
                and 0 in tracks.get(item["track_id"], {})
            ]
            route_mask = _initial_route_mask(
                mask, tracks.get(scene.rope_track_id, {}), other_masks
            )
            initial_route_outlier_pixels = int(mask.sum() - route_mask.sum())
            widths = cv2.distanceTransform(route_mask.astype(np.uint8), cv2.DIST_L2, 5)[
                skeletonize(route_mask)
            ]
            if len(widths):
                route_tolerance_pixels = max(2.0, float(np.median(widths)))
            initial_points = (
                np.column_stack(np.nonzero(skeletonize(route_mask)))[:, ::-1].astype(
                    np.float32
                )
                / scene.image_height
            )
        if len(observed):
            take = min(len(observed), point_count)
            selected = np.linspace(0, len(observed) - 1, take).astype(int)
            points[offset, :take] = observed[selected] / scene.image_height
            valid[offset, :take] = True
        # Filled rigid silhouettes are possible occluders, not observed rope.
        # This allowance affects visibility loss only, never collision geometry.
        allowed = mask.astype(np.uint8)
        for track in rigid_tracks:
            obstacle = tracks.get(track, {}).get(int(frame))
            if obstacle is not None:
                contours, _ = cv2.findContours(
                    obstacle.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(allowed, contours, -1, 1, cv2.FILLED)
        allowed = cv2.dilate(allowed, np.ones((5, 5), np.uint8))
        distance = cv2.distanceTransform(
            1 - allowed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        maps.append(
            cv2.resize(
                distance,
                (
                    math.ceil(scene.image_width / map_scale),
                    math.ceil(scene.image_height / map_scale),
                ),
            )
            / scene.image_height
        )
    if initial_points is None or not len(initial_points):
        raise ValueError("routing needs visible rope in the initial frame")
    return RopeObservations(
        frames,
        points,
        valid,
        np.asarray(maps, np.float32),
        initial_points,
        maps[0],
        scene.image_height,
        scene.image_width,
        np.asarray(visible_lengths, np.float32),
        initial_route_outlier_pixels,
        route_tolerance_pixels,
    )


def available_pulleys(
    scene: RopeSceneData, tracks: dict, source: dict
) -> tuple[RopeBody, ...]:
    """Include detected pulleys missing from the current body association."""
    known = {body.object_id: body for body in scene.bodies if body.role == "pulley"}
    robust_tracking = source.get("rope_tracking_policy", DEFAULT_TRACKING_POLICY) != "legacy"
    pulley_ids = pulley_ids_for_source(source)
    for item in source["objects"]:
        if item["target_id"] not in pulley_ids or item["target_id"] in known:
            continue
        masks = tracks.get(item["track_id"], {})
        if not masks:
            continue
        centers, observed, radius = _interpolate_centers(
            masks, scene.frame_count, circular=True, reject_truncated=robust_tracking
        )
        radius = source.get("pulley_radius_overrides_px", {}).get(item["target_id"], radius)
        motion = (_track_motion(centers, observed) if robust_tracking else
            np.max(np.linalg.norm(centers - np.median(centers[:8], axis=0), axis=1))
        )
        dynamic = motion > 8.0
        initial_center = (np.median(centers[observed], axis=0)
                          if robust_tracking and not dynamic else centers[0])
        known[item["target_id"]] = RopeBody(
            item["target_id"],
            item["track_id"],
            "pulley",
            "cylinder",
            bool(dynamic),
            radius / scene.image_height,
            tuple(initial_center / scene.image_height),
            centers / scene.image_height,
            observed,
            -1,
            (0.0, 0.0),
        )
    return tuple(known.values())


def oriented_tangent(
    c0: np.ndarray, r0: float, s0: int, c1: np.ndarray, r1: float, s1: int
):
    delta = np.asarray(c1) - np.asarray(c0)
    length = np.linalg.norm(delta)
    offset = s0 * r0 - s1 * r1
    if length <= abs(offset) + 1.0e-8:
        return None
    phi = math.atan2(delta[1], delta[0]) + math.asin(offset / length)
    normal = np.array([-math.sin(phi), math.cos(phi)])
    return c0 - s0 * r0 * normal, c1 - s1 * r1 * normal


def tangent_route(
    endpoints: np.ndarray,
    circles: tuple[RopeBody, ...],
    order: tuple[int, ...],
    directions: tuple[int, ...],
    turns: tuple[int, ...],
    *,
    clearance: float,
) -> np.ndarray | None:
    centers = [
        endpoints[0],
        *[np.asarray(circles[i].initial_center) for i in order],
        endpoints[1],
    ]
    radii = [0.0, *[circles[i].radius + clearance for i in order], 0.0]
    signs = [1, *directions, 1]
    tangents = [
        oriented_tangent(
            centers[i], radii[i], signs[i], centers[i + 1], radii[i + 1], signs[i + 1]
        )
        for i in range(len(centers) - 1)
    ]
    if any(tangent is None for tangent in tangents):
        return None
    pieces = []
    for i, (start, end) in enumerate(tangents):
        pieces.append(
            np.linspace(
                start, end, max(2, math.ceil(np.linalg.norm(end - start) / 0.003))
            )
        )
        if i < len(order):
            center, radius, sign = centers[i + 1], radii[i + 1], signs[i + 1]
            following = tangents[i + 1][0]
            a0 = math.atan2(end[1] - center[1], end[0] - center[0])
            a1 = math.atan2(following[1] - center[1], following[0] - center[0])
            sweep = sign * ((sign * (a1 - a0)) % (2 * math.pi) + turns[i] * 2 * math.pi)
            angles = a0 + np.linspace(
                0.0, sweep, max(3, math.ceil(abs(sweep) * radius / 0.003))
            )
            pieces.append(
                center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            )
    path = np.concatenate(pieces)
    # Test the whole route, including against pulleys not visited by it.
    for circle in circles:
        if (
            np.min(np.linalg.norm(path - circle.initial_center, axis=1))
            < circle.radius - clearance * 0.25
        ):
            return None
    return path


def _huber_np(distance: np.ndarray, delta: float = 0.02) -> np.ndarray:
    return np.where(
        distance <= delta, 0.5 * distance**2, delta * (distance - 0.5 * delta)
    )


def geometry_loss(path: np.ndarray, obs: RopeObservations) -> float:
    dense = _resample_polyline(path, 512)
    coverage = cKDTree(dense).query(obs.initial_points)[0]
    height, width = obs.initial_free_distance.shape
    x = np.clip(
        dense[:, 0] * obs.image_height / obs.image_width * (width - 1), 0, width - 1
    ).astype(int)
    y = np.clip(dense[:, 1] * (height - 1), 0, height - 1).astype(int)
    free = obs.initial_free_distance[y, x]
    in_view = (
        (dense[:, 0] >= 0)
        & (dense[:, 0] <= obs.image_width / obs.image_height)
        & (dense[:, 1] >= 0)
        & (dense[:, 1] <= 1)
    )
    free = free * in_view
    length = np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()
    return float(
        np.mean(_huber_np(coverage))
        + np.mean(_huber_np(free)) * length / max(obs.visible_lengths[0], 1.0e-8)
    )


def propose_routes(
    scene: RopeSceneData,
    circles: tuple[RopeBody, ...],
    obs: RopeObservations,
    *,
    keep: int = 4,
    max_visits: int = 5,
) -> tuple[list[RouteCandidate], dict]:
    if keep < 1 or max_visits < 0:
        raise ValueError("invalid route search size")
    baseline = RouteCandidate(
        "legacy-initial",
        scene.initial_positions.copy(),
        (),
        geometry_loss(scene.initial_positions, obs),
        float(np.linalg.norm(np.diff(scene.initial_positions, axis=0), axis=1).sum()),
    )
    candidates = []
    tested, feasible = 0, 0
    endpoints = scene.initial_positions[[0, -1]]
    # First rank base routes; expanding turns only for retained routes bounds
    # the search. This is a finite hypothesis search, not exhaustive knot recovery.
    for count in range(min(max_visits, len(circles)) + 1):
        for order in itertools.permutations(range(len(circles)), count):
            for signs in itertools.product((-1, 1), repeat=count):
                tested += 1
                path = tangent_route(
                    endpoints,
                    circles,
                    order,
                    signs,
                    (0,) * count,
                    clearance=2.0 / scene.image_height,
                )
                if path is None:
                    continue
                feasible += 1
                score = geometry_loss(path, obs)
                candidates.append((score, order, signs, path))
        # Keep enough route diversity to allow a covered extra turn to win.
        candidates = sorted(candidates, key=lambda item: item[0])[: max(32, keep * 8)]
    expanded = []
    for score, order, signs, path in candidates:
        for extra in [-1, *range(len(order))]:
            turns = tuple(int(i == extra) for i in range(len(order)))
            if extra >= 0:
                path_extra = tangent_route(
                    endpoints,
                    circles,
                    order,
                    signs,
                    turns,
                    clearance=2.0 / scene.image_height,
                )
                if path_extra is None:
                    continue
                path_value, score_value = path_extra, geometry_loss(path_extra, obs)
            else:
                path_value, score_value = path, score
            visits = tuple(
                (circles[i].object_id, sign, turn)
                for i, sign, turn in zip(order, signs, turns)
            )
            positions = _resample_polyline(path_value, scene.node_count)
            length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            name = (
                "/".join(f"{body}:{sign:+d}:{turn}" for body, sign, turn in visits)
                or "direct"
            )
            expanded.append(
                RouteCandidate(name, positions, visits, score_value, length)
            )
    ranked = sorted(expanded, key=lambda item: item.geometry_loss)
    return [baseline, *ranked[:keep]], {
        "base_routes_tested": tested,
        "feasible_base_routes": feasible,
        "expanded_routes": len(expanded),
        "max_visits": max_visits,
        "extra_turn_limit": 1,
        "initial_endpoint_pixels": (endpoints * scene.image_height).tolist(),
        "initial_route_outlier_pixels": obs.initial_route_outlier_pixels,
    }


def _route_penetration(path: np.ndarray, circles: tuple[RopeBody, ...]) -> float:
    """Maximum circle penetration along segments, not just at rope nodes."""
    start, segment = path[:-1], np.diff(path, axis=0)
    squared = np.maximum(np.sum(segment**2, axis=1), 1.0e-16)
    penetration = 0.0
    for body in circles:
        center = np.asarray(body.initial_center)
        fraction = np.clip(np.sum((center - start) * segment, axis=1) / squared, 0, 1)
        distance = np.linalg.norm(start + fraction[:, None] * segment - center, axis=1)
        penetration = max(penetration, float(body.radius - distance.min()))
    return penetration


def _observed_route_path(
    candidate: RouteCandidate,
    scene: RopeSceneData,
    circles: tuple[RopeBody, ...],
    frame: int,
    tolerance: float,
) -> np.ndarray | None:
    """Kinematic geometry proxy, not a simulated or corrected output trajectory."""
    if frame == 0:
        return candidate.positions
    endpoints = candidate.positions[[0, -1]].copy()
    for end in range(2):
        body = next(
            (
                b
                for b in scene.bodies
                if b.role == "endpoint" and b.endpoint_index == end
            ),
            None,
        )
        if body is not None:
            if not body.observed[frame]:
                return None
            endpoints[end] += body.target_centers[frame] - body.initial_center
        elif not scene.endpoint_pinned[end]:
            # An unobserved free end is not an inferred motion constraint.
            return None
    if candidate.name != "legacy-initial":
        lookup = {body.object_id: i for i, body in enumerate(circles)}
        if any(
            not circles[lookup[name]].observed[frame] for name, _, _ in candidate.visits
        ):
            return None
        moved = tuple(
            replace(body, initial_center=tuple(body.target_centers[frame]))
            for body in circles
        )
        return tangent_route(
            endpoints,
            moved,
            tuple(lookup[name] for name, _, _ in candidate.visits),
            tuple(sign for _, sign, _ in candidate.visits),
            tuple(turn for _, _, turn in candidate.visits),
            clearance=2.0 / scene.image_height,
        )
    # Transport the observed skeleton by its endpoint/contact supports. Free
    # spans interpolate displacement, without inventing a taut winding count.
    path = candidate.positions
    displacement = np.zeros_like(path)
    anchored = np.zeros(len(path), bool)
    nearest_gap = np.full(len(path), np.inf)
    for body in circles:
        gap = np.abs(np.linalg.norm(path - body.initial_center, axis=1) - body.radius)
        near = (gap <= tolerance + 2.0 / scene.image_height) & (gap < nearest_gap)
        if not near.any():
            continue
        if not body.observed[frame]:
            return None
        displacement[near] = body.target_centers[frame] - body.initial_center
        anchored[near], nearest_gap[near] = True, gap[near]
    anchored[[0, -1]] = True
    displacement[[0, -1]] = endpoints - path[[0, -1]]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path + np.column_stack(
        [
            np.interp(distance, distance[anchored], displacement[anchored, axis])
            for axis in range(2)
        ]
    )


def select_observed_route(
    scene: RopeSceneData,
    circles: tuple[RopeBody, ...],
    candidates: list[RouteCandidate],
    obs: RopeObservations,
    *,
    length_prior_weight: float = DEFAULT_LENGTH_PRIOR_WEIGHT,
) -> tuple[RouteCandidate, dict]:
    """Choose one initial route from observed geometry without a rollout.

    Observed endpoint/pulley motion transports each topology for a cheap
    geometric consistency check. It does not replace the open-loop rollout.
    The mask half-width is a measurement tolerance, not a collision margin.
    """
    if not candidates:
        raise ValueError("no route candidates")
    tolerance = obs.route_tolerance_pixels / scene.image_height
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid route observation tolerance")
    for candidate in candidates:
        path = np.asarray(candidate.positions)
        if (
            path.ndim != 2
            or path.shape[1] != 2
            or len(path) < 2
            or not np.isfinite(path).all()
        ):
            raise ValueError("invalid route positions")
    paths = [
        [
            (
                _observed_route_path(candidate, scene, circles, int(frame), tolerance)
                if obs.valid[offset].any()
                else None
            )
            for offset, frame in enumerate(obs.frames)
        ]
        for candidate in candidates
    ]
    # Compare every hypothesis on the same frames. A taut proxy ceases to be
    # applicable after contact changes; skipping that frame for just one route
    # would give it an unfair advantage. Dynamics still fits every observation.
    offsets = [
        i for i in range(len(obs.frames)) if all(row[i] is not None for row in paths)
    ]
    if not offsets:
        raise ValueError("no shared observed frames for route selection")
    rows = []
    for candidate_index, candidate in enumerate(candidates):
        path = np.asarray(candidate.positions)
        penetration = _route_penetration(path, circles)
        frame_losses = []
        used_frames = []
        length_errors = []
        for offset in offsets:
            frame = obs.frames[offset]
            transported = paths[candidate_index][offset]
            used_frames.append(int(frame))
            length = float(np.linalg.norm(np.diff(transported, axis=0), axis=1).sum())
            length_errors.append(
                float(_huber_np(max(0.0, length - candidate.length - tolerance)))
            )
            dense = _resample_polyline(transported, max(512, 4 * len(path)))
            tree = cKDTree(dense)
            points = (
                obs.initial_points
                if offset == 0
                else obs.points[offset, obs.valid[offset]]
            )
            coverage = np.maximum(tree.query(points)[0] - tolerance, 0.0)
            distance_map = obs.free_distance[offset]
            height, width = distance_map.shape
            x = np.clip(
                dense[:, 0] * scene.image_height / scene.image_width * (width - 1),
                0,
                width - 1,
            ).astype(int)
            y = np.clip(dense[:, 1] * (height - 1), 0, height - 1).astype(int)
            in_view = (
                (dense[:, 0] >= 0)
                & (dense[:, 0] <= scene.image_width / scene.image_height)
                & (dense[:, 1] >= 0)
                & (dense[:, 1] <= 1)
            )
            free = np.maximum(distance_map[y, x] - tolerance, 0.0) * in_view
            frame_losses.append(
                float(
                    np.mean(_huber_np(coverage))
                    + np.mean(_huber_np(free))
                    * length
                    / max(float(obs.visible_lengths[0]), 1.0e-8)
                )
            )
        # This kinematic proxy cannot model slack or contact release. Its
        # apparent extension is diagnostic only, not an inextensibility loss.
        geometry = float(np.mean(frame_losses))
        score = route_selection_score(
            geometry,
            candidate.length,
            float(obs.visible_lengths[0]),
            weight=length_prior_weight,
        )
        rows.append(
            {
                "name": candidate.name,
                "geometry_observation_loss": geometry,
                "kinematic_extension_huber_diagnostic": float(np.mean(length_errors)),
                "observation_frames": used_frames,
                **score,
                "length": candidate.length,
                "max_initial_penetration_pixels": penetration * scene.image_height,
                # Permit polygonal arc discretization, not macroscopic shortcuts.
                "feasible": penetration * scene.image_height <= 0.5,
            }
        )
    feasible = [i for i, row in enumerate(rows) if row["feasible"]]
    if not feasible:
        raise ValueError(
            "no nonpenetrating initial rope route; inspect masks and pulley geometry"
        )
    ranked = sorted(
        feasible,
        key=lambda i: (rows[i]["selection_loss"], rows[i]["length"], rows[i]["name"]),
    )
    chosen = ranked[0]
    gap = (
        rows[ranked[1]]["selection_loss"] - rows[chosen]["selection_loss"]
        if len(ranked) > 1
        else None
    )
    return candidates[chosen], {
        "method": "observed-geometry-single-fit",
        "selected": candidates[chosen].name,
        "uses_previous_fit": False,
        "planned_physics_fit_count": 1,
        "observation_frames": rows[chosen]["observation_frames"],
        "tolerance_pixels": obs.route_tolerance_pixels,
        "score_gap": gap,
        "ambiguous": gap is not None and gap <= 0.5 * tolerance**2,
        "candidates": rows,
    }


class VisibleRopeLoss:
    def __init__(
        self, obs: RopeObservations, device: str, body_ids: set[str] | None = None
    ):
        self.obs = obs
        self.frames = torch.as_tensor(obs.frames, device=device)
        self.points = torch.as_tensor(obs.points, device=device)
        self.valid = torch.as_tensor(obs.valid, device=device)
        self.maps = torch.as_tensor(obs.free_distance, device=device)
        self.visible_lengths = torch.as_tensor(obs.visible_lengths, device=device)
        self.body_ids = body_ids

    def terms(self, rope: torch.Tensor, end_offset: int):
        use = self.frames <= end_offset
        indices = self.frames[use]
        selected = rope[:, indices]
        alpha = torch.linspace(0.0, 1.0, 4, device=rope.device)
        dense = (
            selected[:, :, :-1, None] * (1 - alpha[None, None, None, :, None])
            + selected[:, :, 1:, None] * alpha[None, None, None, :, None]
        ).flatten(2, 3)
        observed = self.points[use]
        valid = self.valid[use]
        coverage = torch.cdist(observed[None], dense).min(-1).values
        grid = dense.clone()
        grid[..., 0] = (
            grid[..., 0] * self.obs.image_height / self.obs.image_width * 2 - 1
        )
        grid[..., 1] = grid[..., 1] * 2 - 1
        # grid_sample batches frames, with candidate and node as grid dimensions.
        free = torch.nn.functional.grid_sample(
            self.maps[use, None],
            grid.permute(1, 0, 2, 3),
            align_corners=True,
            padding_mode="border",
        )[:, 0].permute(1, 0, 2)
        # Outside the camera is unobserved, not evidence of empty space.
        in_view = ((grid >= -1) & (grid <= 1)).all(-1)
        free = free * in_view
        # Entirely missing masks contain no evidence of visible free space.
        frame_valid = valid.any(-1)
        return coverage, valid, free, frame_valid

    def free_space_integral(self, values, rope, end_offset, frame_valid):
        use = self.frames <= end_offset
        segments = torch.linalg.vector_norm(
            torch.diff(rope[:, self.frames[use]], dim=2), dim=-1
        )
        quadrature = torch.tensor([1.0, 2.0, 2.0, 1.0], device=rope.device) / 6.0
        weights = (segments[..., None] * quadrature).flatten(2, 3)
        # A fixed observed denominator prevents extra hidden length from
        # diluting the penalty on an unchanged visible wrong segment.
        return (values * weights * frame_valid[None, :, None]).sum(
            (1, 2)
        ) / self.visible_lengths[use].sum().clamp_min(1.0e-8)

    def __call__(self, engine, rope, pulleys, *, end_offset, config):
        coverage, valid, free, frame_valid = self.terms(rope, end_offset)
        delta = config.huber_delta

        def huber(x):
            return torch.where(x <= delta, 0.5 * x.square(), delta * (x - 0.5 * delta))

        loss = (huber(coverage) * valid).sum((1, 2)) / valid.sum().clamp_min(1)
        loss += self.free_space_integral(huber(free), rope, end_offset, frame_valid)
        for object_id, predicted in engine.predicted_body_centers(
            rope, pulleys
        ).items():
            if self.body_ids is not None and object_id not in self.body_ids:
                continue
            body = next(
                body for body in engine.scene.bodies if body.object_id == object_id
            )
            mask = torch.as_tensor(body.observed[: end_offset + 1], device=rope.device)
            distance = torch.linalg.vector_norm(
                predicted - engine.body_targets[object_id][: end_offset + 1], dim=-1
            )
            loss += (
                config.body_weight
                * (huber(distance) * mask).sum(1)
                / mask.sum().clamp_min(1)
            )
        return loss

    def metrics(self, rope: np.ndarray) -> dict[str, float]:
        tensor = torch.as_tensor(
            rope[None], dtype=torch.float32, device=self.points.device
        )
        coverage, valid, free, frame_valid = self.terms(tensor, len(rope) - 1)
        c = (coverage.square() * valid).sum() / valid.sum().clamp_min(1)
        f = (free.square() * frame_valid[None, :, None]).sum() / (
            frame_valid.sum() * free.shape[-1]
        ).clamp_min(1)
        return {
            "visible_coverage_rmse_pixels": float(c.sqrt()) * self.obs.image_height,
            "visible_free_space_rmse_pixels": float(f.sqrt()) * self.obs.image_height,
            "free_space_integral_rms_pixels": float(
                self.free_space_integral(
                    free.square(), tensor, len(rope) - 1, frame_valid
                )[0].sqrt()
            )
            * self.obs.image_height,
        }


def physical_metrics(scene: RopeSceneData, predicted: np.ndarray, bodies: dict) -> dict:
    rest = np.linalg.norm(np.diff(scene.initial_positions, axis=0), axis=-1)
    lengths = np.linalg.norm(np.diff(predicted, axis=1), axis=-1)
    strain = np.abs(lengths / np.maximum(rest, 1.0e-8) - 1.0)
    penetration = 0.0
    start, end = predicted[:, :-1], predicted[:, 1:]
    delta = end - start
    for body in scene.bodies:
        if body.role != "pulley":
            continue
        centers = bodies.get(
            body.object_id, np.broadcast_to(body.initial_center, (len(predicted), 2))
        )
        t = np.clip(
            np.sum((centers[:, None] - start) * delta, axis=-1)
            / np.maximum(np.sum(delta * delta, axis=-1), 1.0e-12),
            0.0,
            1.0,
        )
        distance = np.linalg.norm(
            start + t[..., None] * delta - centers[:, None], axis=-1
        )
        penetration = max(penetration, float(np.max(body.radius - distance)))
    return {
        "max_pulley_penetration_pixels": penetration * scene.image_height,
        "p95_relative_segment_length_error": float(np.percentile(strain, 95)),
        "max_relative_segment_length_error": float(np.max(strain)),
    }


def _route_scene(scene, circles, candidate):
    used = {visit[0] for visit in candidate.visits}
    existing = {body.object_id for body in scene.bodies}
    bodies = (
        *scene.bodies,
        *[
            body
            for body in circles
            if body.object_id in used and body.object_id not in existing
        ],
    )
    return replace(scene, initial_positions=candidate.positions, bodies=bodies,
                   route_visits=tuple(candidate.visits))


def _reference_linear_density(scene):
    length = float(
        np.linalg.norm(np.diff(scene.initial_positions, axis=0), axis=1).sum()
    )
    return (MASS_REFERENCE_NODE_COUNT - 1) / (ROPE_PARTICLE_INVERSE_MASS * length)


def _ideal_candidates(scene, candidates):
    required = {b.object_id for b in scene.bodies if b.role == "pulley" and b.dynamic}
    result = [c for c in candidates if c.name != "legacy-initial"
              and required.issubset({v[0] for v in c.visits})]
    if not result:
        raise ValueError("No explicit tangent route supports every dynamic pulley for massless fitting")
    return result


def _parameter_keys(scene):
    keys = ["gravity", "drag", "stiffness", "initial_speed"]
    entities = [f"endpoint:{end}" for end in range(2) if not scene.endpoint_pinned[end]]
    entities += [
        f"pulley:{body.object_id}"
        for body in scene.bodies
        if body.role == "pulley" and body.dynamic
    ]
    return keys + [
        f"{entity}:mass" for entity in entities for _ in range(2)
    ]


def _screening_routes(scene, circles, candidates, obs):
    """Do not use invisible duplicate turns to compensate for a coarse bank."""
    accepted, rejected = {}, []
    lookup = {body.object_id: i for i, body in enumerate(circles)}
    noise = 0.5 * (obs.route_tolerance_pixels / scene.image_height) ** 2
    for candidate in candidates:
        if any(turn > 0 for _, _, turn in candidate.visits):
            visits = tuple((name, sign, 0) for name, sign, _ in candidate.visits)
            path = tangent_route(
                scene.initial_positions[[0, -1]],
                circles,
                tuple(lookup[name] for name, _, _ in visits),
                tuple(sign for _, sign, _ in visits),
                (0,) * len(visits),
                clearance=2.0 / scene.image_height,
            )
            if path is not None:
                name = "/".join(f"{body}:{sign:+d}:0" for body, sign, _ in visits)
                base = next((value for value in candidates if value.name == name), None)
                if base is None:
                    positions = _resample_polyline(path, scene.node_count)
                    base = RouteCandidate(
                        name,
                        positions,
                        visits,
                        geometry_loss(path, obs),
                        float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                    )
                gain = base.geometry_loss - candidate.geometry_loss
                if gain <= noise:
                    accepted[base.name] = base
                    rejected.append(
                        {
                            "name": candidate.name,
                            "reason": "extra-turn-without-visible-support",
                            "geometry_gain": gain,
                            "noise_tolerance": noise,
                            "replacement": base.name,
                        }
                    )
                    continue
        accepted[candidate.name] = candidate
    result = []
    for candidate in accepted.values():
        penetration = (
            _route_penetration(candidate.positions, circles) * scene.image_height
        )
        if penetration > 0.5:
            rejected.append(
                {
                    "name": candidate.name,
                    "reason": "initial-pulley-penetration",
                    "penetration_pixels": penetration,
                }
            )
        else:
            result.append(candidate)
    if not result:
        raise ValueError("no nonpenetrating route for forward screening")
    return result, rejected


def screen_route_candidates(
    scene: RopeSceneData,
    circles: tuple[RopeBody, ...],
    candidates: list[RouteCandidate],
    obs: RopeObservations,
    *,
    config: RopeCEMConfig,
    device: str,
    substeps: int,
    sample_count: int = 64,
    length_prior_weight: float = DEFAULT_LENGTH_PRIOR_WEIGHT,
    mass_model: str = DEFAULT_MASS_MODEL,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
    gravity_normalized_per_s2: float | None = None,
    fit_initial_speed: bool = False,
) -> tuple[RouteCandidate, dict]:
    """One fixed parameter bank per route; no CEM, updates or fitted warm start.

    The bank is shared by semantic parameter ID, including when a route adds a
    previously unassociated dynamic pulley. Screening cannot use saved winners.
    """
    from atw.polyworld.rope_gravity import require_gravity, fix_engine_gravity
    gravity = require_gravity(gravity_normalized_per_s2)
    if sample_count < 1 or not candidates:
        raise ValueError("screening needs candidates and a positive sample count")
    if mass_model not in {"linear-density", "uniform-node", "massless"}:
        raise ValueError("invalid rope mass model")
    if mass_model == "massless":
        scene = replace(scene, mass_model="massless")
        candidates = _ideal_candidates(scene, candidates)
    started = time.perf_counter()
    candidates, rejected = _screening_routes(scene, circles, candidates, obs)
    trials = [_route_scene(scene, circles, candidate) for candidate in candidates]
    common = _parameter_keys(scene)[:4]
    keys = common + sorted(
        {key for trial in trials for key in _parameter_keys(trial)[4:]}
    )
    mean, std = initial_distribution(len(keys))
    lower, upper = parameter_bounds(len(keys))
    unit = qmc.Sobol(d=len(keys), scramble=True, seed=config.seed).random_base2(
        int(math.ceil(math.log2(sample_count)))
    )[:sample_count]
    bank = np.clip(
        mean + std * ndtri(np.clip(unit, 1.0e-6, 1 - 1.0e-6)), lower, upper
    ).astype(np.float32)
    bank[0] = mean
    bank[:, 0] = math.log(gravity)
    density = _reference_linear_density(scene)
    objective = VisibleRopeLoss(obs, device, {body.object_id for body in scene.bodies})
    rows = []
    unsupported_route_errors = {
        "A free massless rope end needs an explicit load",
        "Unloaded massless moving pulley requires a coupled algebraic solver",
        "Invalid initial tangent geometry",
    }
    for candidate, trial in zip(candidates, trials):
        route_started = time.perf_counter()
        values = bank[:, [keys.index(key) for key in _parameter_keys(trial)]]
        try:
            engine = make_rope_engine(
                trial,
                capacity=sample_count,
                device=device,
                substeps=substeps,
                simulate_static_system=True,
                rope_linear_density=density if mass_model == "linear-density" else None,
                execution_mode=execution_mode,
            )
        except ValueError as error:
            if str(error) not in unsupported_route_errors:
                raise
            rows.append({
                "name": candidate.name,
                "screen_observation_loss": None,
                "selection_loss": None,
                "length_prior_penalty": None,
                "best_bank_index": None,
                "finite_samples": 0,
                "forward_rollouts": 0,
                "rejection_reason": str(error),
                "elapsed_seconds": time.perf_counter() - route_started,
            })
            continue
        fix_engine_gravity(engine, gravity)
        if not fit_initial_speed:
            engine.fixed_parameters[3] = 0.
        for index, value in engine.fixed_parameters.items():
            values[:, index] = value
            if index >= 4:
                values[:, index+1] = value
        predicted, pulleys = engine.rollout(values)
        losses = objective(
            engine, predicted, pulleys, end_offset=trial.frame_count - 1, config=config
        )
        losses = losses.detach().cpu().numpy()
        finite = np.flatnonzero(np.isfinite(losses))
        index = int(finite[np.argmin(losses[finite])]) if len(finite) else None
        value = float(losses[index]) if index is not None else None
        score = route_selection_score(
            value if value is not None else math.inf,
            candidate.length,
            float(obs.visible_lengths[0]),
            weight=length_prior_weight,
            huber_delta=config.huber_delta,
        )
        rows.append(
            {
                "name": candidate.name,
                "screen_observation_loss": value,
                "selection_loss": (
                    score["selection_loss"] if index is not None else None
                ),
                "length_prior_penalty": score["length_prior_penalty"],
                "best_bank_index": index,
                "finite_samples": len(finite),
                "forward_rollouts": sample_count,
                "execution_config": getattr(engine, "execution_config", {}),
                "elapsed_seconds": time.perf_counter() - route_started,
            }
        )
        del engine, predicted, pulleys, losses
    ranked = sorted(
        (i for i, row in enumerate(rows) if row["selection_loss"] is not None),
        key=lambda i: (
            rows[i]["selection_loss"],
            candidates[i].length,
            candidates[i].name,
        ),
    )
    if not ranked:
        raise ValueError("no finite route screening rollout: " + "; ".join(
            sorted({row.get("rejection_reason", "non-finite forward loss") for row in rows})))
    chosen = ranked[0]
    return candidates[chosen], {
        "method": "fixed-bank-forward-screen-single-fit",
        "selected": candidates[chosen].name,
        "uses_previous_fit": False,
        "planned_physics_fit_count": 1,
        "cem_calls": 0,
        "sample_count_per_route": sample_count,
        "bank_distribution": "sobol-normal-initial-prior-with-fixed-gravity",
        "gravity_normalized_per_s2": gravity,
        "initial_velocity_policy": "fitted" if fit_initial_speed else "fixed-zero",
        "forward_rollouts": len(candidates) * sample_count,
        "seed": config.seed,
        "warm_start_used": False,
        "parameter_keys": keys,
        "score_gap": (
            rows[ranked[1]]["selection_loss"] - rows[chosen]["selection_loss"]
            if len(ranked) > 1
            else None
        ),
        "candidates": rows,
        "geometric_rejections": rejected,
        "elapsed_seconds": time.perf_counter() - started,
    }


def fit_route_candidates(
    scene: RopeSceneData,
    circles: tuple[RopeBody, ...],
    candidates: list[RouteCandidate],
    obs: RopeObservations,
    *,
    config: RopeCEMConfig,
    device: str,
    substeps: int,
    length_prior_weight: float = DEFAULT_LENGTH_PRIOR_WEIGHT,
    mass_model: str = DEFAULT_MASS_MODEL,
    verbose: bool = True,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
    gravity_normalized_per_s2: float | None = None,
    fit_initial_speed: bool = False,
) -> tuple[dict, RopeSceneData, list[dict]]:
    from atw.polyworld.rope_gravity import require_gravity, fix_engine_gravity
    gravity = require_gravity(gravity_normalized_per_s2)
    if mass_model not in {"linear-density", "uniform-node", "massless"}:
        raise ValueError("invalid rope mass model")
    if mass_model == "massless":
        scene = replace(scene, mass_model="massless")
        candidates = _ideal_candidates(scene, candidates)
    reference_length = float(
        np.linalg.norm(np.diff(scene.initial_positions, axis=0), axis=1).sum()
    )
    # One fixed density per rope, shared by all route hypotheses. The 96-node
    # calibration is independent of the requested simulation node count.
    density = _reference_linear_density(scene)
    visible_length = float(obs.visible_lengths[0])
    route_selection_score(
        0.0,
        reference_length,
        visible_length,
        weight=length_prior_weight,
        huber_delta=config.huber_delta,
    )
    objective = VisibleRopeLoss(obs, device, {body.object_id for body in scene.bodies})
    selected, selected_scene = None, None
    report = []
    for candidate in candidates:
        started = time.perf_counter()
        existing = {body.object_id for body in scene.bodies}
        # Retain the same original body observations in every candidate. Add
        # previously unassociated pulleys only when a route actually uses them.
        trial = _route_scene(scene, circles, candidate)
        engine = make_rope_engine(
            trial,
            capacity=config.population,
            device=device,
            substeps=substeps,
            simulate_static_system=True,
            rope_linear_density=density if mass_model == "linear-density" else None,
            execution_mode=execution_mode,
        )
        if mass_model == "massless":
            initial = engine.material_points(engine.initial[None], engine.path(engine.initial[None]))[0]
            trial = replace(trial, initial_positions=initial)
            engine.scene = trial
        fix_engine_gravity(engine, gravity)
        if not fit_initial_speed:
            engine.fixed_parameters[3] = 0.
        cem = run_cem(engine, config, verbose=False, loss_fn=objective)
        rope, pulleys = engine.rollout(cem["best"])
        final_loss = float(
            objective(
                engine, rope, pulleys, end_offset=trial.frame_count - 1, config=config
            )[0]
        )
        predicted = rope[0].detach().cpu().numpy().copy()
        predicted_pulleys = pulleys[0].detach().cpu().numpy().copy()
        body_np = {
            key: value[0].detach().cpu().numpy().copy()
            for key, value in engine.predicted_body_centers(rope, pulleys).items()
        }
        body_errors = [
            np.sum(
                (
                    body_np[body.object_id][body.observed]
                    - body.target_centers[body.observed]
                )
                ** 2,
                axis=-1,
            )
            for body in trial.bodies
            if body.dynamic and np.any(body.observed) and body.object_id in existing
        ]
        body_rmse = (
            float(np.sqrt(np.mean(np.concatenate(body_errors)))) if body_errors else 0.0
        )
        row = {
            "name": candidate.name,
            "visits": candidate.visits,
            "initial_length_normalized": candidate.length,
            "geometry_loss": candidate.geometry_loss,
            "observation_loss": final_loss,
            **route_selection_score(
                final_loss,
                candidate.length,
                visible_length,
                weight=length_prior_weight,
                huber_delta=config.huber_delta,
            ),
            "length_prior_weight": length_prior_weight,
            "visible_reference_length_normalized": visible_length,
            "mass_model": mass_model,
            "rope_linear_density": (density if mass_model == "linear-density" else
                                    0. if mass_model == "massless" else None),
            "best_parameters": cem["best"].tolist(),
            "execution_config": getattr(engine, "execution_config", {}),
            **objective.metrics(predicted),
            "dynamic_body_rmse_pixels": body_rmse * scene.image_height,
            **physical_metrics(trial, predicted, body_np),
            "elapsed_seconds": time.perf_counter() - started,
        }
        report.append(row)
        if verbose:
            print(json.dumps({"rope": scene.rope_id, **row}), flush=True)
        if math.isfinite(final_loss) and (
            selected is None or row["selection_loss"] < selected["selection_loss"]
        ):
            selected = {
                **cem,
                "best_loss": final_loss,
                "selection_loss": row["selection_loss"],
                "predicted": predicted,
                "predicted_pulleys": predicted_pulleys,
                "predicted_bodies": body_np,
                "body_rmse": body_rmse,
                "centerline_rmse": centerline_chamfer_rmse(
                    scene.target_positions, predicted
                ),
                "route": row,
                "execution_config": row["execution_config"],
                "constraint_diagnostics": engine.constraint_report() if hasattr(engine, 'constraint_report') else {},
            }
            selected_scene = trial
    if selected is None:
        raise ValueError("no finite route fit")
    return selected, selected_scene, report


def read_observation_tracks(scene: RopeSceneData) -> dict:
    from pathlib import Path

    return read_packed_mask_archive(Path(scene.mask_archive), Path(scene.mask_index))
