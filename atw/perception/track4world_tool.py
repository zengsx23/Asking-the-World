"""Track4World DA3 inference and cloth-point correspondence utilities."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import cv2
import numpy as np

from atw.perception.model_loading import prefetch_weight_files
from atw.perception.sparse_pointcloud import deterministic_subsample_indices


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACK4WORLD_ROOT = PROJECT_ROOT / "third_party/track4world"
UTILS3D_ROOT = PROJECT_ROOT / "third_party/utils3d"
PI3_ROOT = PROJECT_ROOT / "third_party/pi3"
DA3_REPO_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
TORCH_HUB_ROOT = PROJECT_ROOT / "data/models/track4world/torch/hub"
CONVNEXT_CHECKPOINT = TORCH_HUB_ROOT / "checkpoints/convnext_tiny-983f1562.pth"
TRACK4WORLD_SEGMENT_LENGTH = 8
ADJACENT_TRACKING_CACHE_SCHEMA = "physmind.track4world_adjacent_flow"
ADJACENT_TRACKING_CACHE_VERSION = 1


_TRACK4WORLD_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def close_task_track4world_models() -> None:
    """Release persistent models at the end of a worker session, not each stage."""
    _TRACK4WORLD_MODEL_CACHE.clear()


@dataclass(frozen=True, slots=True)
class Track4WorldTransform:
    original_size_wh: tuple[int, int]
    output_size_wh: tuple[int, int]
    maximum_dimension: int
    dimension_multiple: int


@dataclass(slots=True)
class Track4WorldBranchResult:
    frame_indices: np.ndarray
    rgb: np.ndarray
    intrinsics: np.ndarray
    camera_poses: np.ndarray
    camera_points: np.ndarray
    world_points: np.ndarray
    geometry_confidence: np.ndarray
    geometry_valid: np.ndarray
    anchor_to_frame_uv: np.ndarray
    anchor_to_frame_xyz: np.ndarray
    tracking_confidence: np.ndarray


@dataclass(slots=True)
class AdjacentTrack4WorldResult:
    """Dense chronological correspondences for every adjacent frame pair."""

    frame_count: int
    source_to_target_uv: np.ndarray
    tracking_confidence: np.ndarray


def save_adjacent_track4world_cache(
    directory: Path,
    result: AdjacentTrack4WorldResult,
    *,
    source_video_path: Path,
    inference_iterations: int,
) -> Path:
    """Persist dense adjacent UV and confidence fields for solver replay."""

    root = directory.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite adjacent cache: {root}")
    root.mkdir(parents=True)
    uv = np.asarray(result.source_to_target_uv, dtype=np.float32)
    confidence = np.asarray(result.tracking_confidence, dtype=np.float32)
    if uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError("adjacent UV cache must have shape pairs x height x width x 2")
    if confidence.shape != uv.shape[:-1]:
        raise ValueError("adjacent confidence cache must match UV fields")
    if len(uv) != result.frame_count - 1:
        raise ValueError("adjacent cache pair count must equal frame count minus one")
    np.save(root / "source_to_target_uv.npy", uv)
    np.save(root / "tracking_confidence.npy", confidence)
    payload = {
        "schema": ADJACENT_TRACKING_CACHE_SCHEMA,
        "schema_version": ADJACENT_TRACKING_CACHE_VERSION,
        "source_video": str(source_video_path.resolve()),
        "frame_count": int(result.frame_count),
        "pair_count": int(len(uv)),
        "image_size_wh": [int(uv.shape[2]), int(uv.shape[1])],
        "dtype": "float32",
        "inference_iterations": int(inference_iterations),
        "direction": "chronological_t_to_t_plus_1",
        "backward_pose_policy": "invert_forward_rigid_increment",
        "arrays": {
            "source_to_target_uv": "source_to_target_uv.npy",
            "tracking_confidence": "tracking_confidence.npy",
        },
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_adjacent_track4world_cache(
    directory: Path,
    *,
    source_video_path: Path,
    frame_count: int,
    image_size_wh: tuple[int, int],
) -> AdjacentTrack4WorldResult:
    """Load a retained adjacent-flow cache without copying its dense arrays."""

    root = directory.resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema") != ADJACENT_TRACKING_CACHE_SCHEMA:
        raise ValueError("unexpected adjacent tracking cache schema")
    if payload.get("schema_version") != ADJACENT_TRACKING_CACHE_VERSION:
        raise ValueError("unsupported adjacent tracking cache version")
    if Path(payload["source_video"]).resolve() != source_video_path.resolve():
        raise ValueError("adjacent cache belongs to a different source video")
    if int(payload["frame_count"]) != frame_count:
        raise ValueError("adjacent cache frame count does not match the video")
    if tuple(int(value) for value in payload["image_size_wh"]) != image_size_wh:
        raise ValueError("adjacent cache image size does not match preprocessing")
    arrays = payload["arrays"]
    uv = np.load(root / arrays["source_to_target_uv"], mmap_mode="r")
    confidence = np.load(root / arrays["tracking_confidence"], mmap_mode="r")
    expected_uv = (frame_count - 1, image_size_wh[1], image_size_wh[0], 2)
    if uv.shape != expected_uv or uv.dtype != np.float32:
        raise ValueError("adjacent cache UV array has an unexpected layout")
    if confidence.shape != expected_uv[:-1] or confidence.dtype != np.float32:
        raise ValueError("adjacent cache confidence array has an unexpected layout")
    return AdjacentTrack4WorldResult(
        frame_count=frame_count,
        source_to_target_uv=uv,
        tracking_confidence=confidence,
    )


@dataclass(slots=True)
class SceneGeometryResult:
    transform: Track4WorldTransform
    rgb: np.ndarray
    intrinsics: np.ndarray
    camera_poses: np.ndarray
    camera_points: np.ndarray
    world_points: np.ndarray
    geometry_confidence: np.ndarray
    geometry_valid: np.ndarray


@dataclass(slots=True)
class CanonicalTrajectory:
    uv: np.ndarray
    model_xyz: np.ndarray
    tracking_confidence: np.ndarray


@dataclass(frozen=True, slots=True)
class ObservedPointAssociations:
    current_uv: np.ndarray
    current_xyz: np.ndarray
    canonical_ids: np.ndarray
    geometry_confidence: np.ndarray
    tracking_confidence: np.ndarray
    image_distance: np.ndarray

    @property
    def associated(self) -> np.ndarray:
        return self.canonical_ids >= 0


def build_track4world_transform(
    width: int,
    height: int,
    *,
    maximum_dimension: int = 640,
    dimension_multiple: int = 64,
) -> Track4WorldTransform:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if maximum_dimension < dimension_multiple or dimension_multiple <= 0:
        raise ValueError("invalid resize configuration")
    candidates = range(dimension_multiple, maximum_dimension + 1, dimension_multiple)
    resized_width, resized_height = min(
        (
            (candidate_width, candidate_height)
            for candidate_width in candidates
            for candidate_height in candidates
        ),
        key=lambda size: (
            abs(size[0] * height - size[1] * width) / (size[1] * width),
            -(size[0] * size[1]),
        ),
    )
    return Track4WorldTransform(
        original_size_wh=(width, height),
        output_size_wh=(resized_width, resized_height),
        maximum_dimension=maximum_dimension,
        dimension_multiple=dimension_multiple,
    )


def apply_rgb_resize(
    frame_rgb: np.ndarray,
    transform: Track4WorldTransform,
) -> np.ndarray:
    frame = np.asarray(frame_rgb, dtype=np.uint8)
    expected = (
        transform.original_size_wh[1],
        transform.original_size_wh[0],
        3,
    )
    if frame.shape != expected:
        raise ValueError(f"RGB frame shape {frame.shape} does not match {expected}")
    return cv2.resize(
        frame,
        transform.output_size_wh,
        interpolation=cv2.INTER_LINEAR,
    )


def read_video_rgb(video_path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video has no decodable frames: {video_path}")
    return np.stack(frames), fps


def preprocess_video(
    frames_rgb: np.ndarray,
    *,
    maximum_dimension: int = 640,
) -> tuple[np.ndarray, Track4WorldTransform]:
    frames = np.asarray(frames_rgb, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("video must have shape (frames, height, width, 3)")
    transform = build_track4world_transform(
        int(frames.shape[2]),
        int(frames.shape[1]),
        maximum_dimension=maximum_dimension,
    )
    resized = np.stack(
        [apply_rgb_resize(frame, transform) for frame in frames]
    )
    return resized, transform


def sample_canonical_tracks_on_observed_mask(
    *,
    observed_mask: np.ndarray,
    observed_world_points: np.ndarray,
    observed_camera_points: np.ndarray,
    observed_geometry_confidence: np.ndarray,
    canonical_to_frame_uv: np.ndarray,
    canonical_to_frame_xyz: np.ndarray,
    canonical_tracking_confidence: np.ndarray,
    minimum_tracking_confidence: float,
    maximum_observed_points: int | None = None,
    seed: int = 0,
) -> ObservedPointAssociations:
    """Sample reconstructed cloth points using forward canonical trajectories.

    Each canonical ID proposes one rounded current-frame pixel. Proposals outside
    the valid cloth surface are rejected. If multiple IDs propose the same pixel,
    confidence, subpixel residual, and canonical ID deterministically break ties.
    """
    if not 0.0 <= minimum_tracking_confidence <= 1.0:
        raise ValueError("minimum tracking confidence must be in [0, 1]")
    if maximum_observed_points is not None and maximum_observed_points < 1:
        raise ValueError("maximum observed points must be positive")
    mask = np.asarray(observed_mask, dtype=bool)
    height, width = mask.shape
    world = np.asarray(observed_world_points, dtype=np.float32)
    camera = np.asarray(observed_camera_points, dtype=np.float32)
    geometry_confidence = np.asarray(
        observed_geometry_confidence, dtype=np.float32
    )
    if world.shape != (height, width, 3) or camera.shape != world.shape:
        raise ValueError("observed point maps must match the mask")
    if geometry_confidence.shape != mask.shape:
        raise ValueError("geometry confidence must match the mask")

    observed_valid = (
        mask
        & np.isfinite(world).all(axis=-1)
        & np.isfinite(camera).all(axis=-1)
        & (camera[..., 2] > 0)
        & np.isfinite(geometry_confidence)
    )
    observed_y, observed_x = np.nonzero(observed_valid)
    if maximum_observed_points is not None:
        observed_indices = deterministic_subsample_indices(
            len(observed_y),
            maximum_points=maximum_observed_points,
            seed=seed,
        )
        observed_y = observed_y[observed_indices]
        observed_x = observed_x[observed_indices]
    current_uv = np.column_stack((observed_x, observed_y)).astype(np.float32)
    current_xyz = world[observed_y, observed_x]
    current_geometry_confidence = geometry_confidence[observed_y, observed_x]

    predicted_uv = np.asarray(canonical_to_frame_uv, dtype=np.float32)
    predicted_xyz = np.asarray(canonical_to_frame_xyz, dtype=np.float32)
    predicted_confidence = np.asarray(
        canonical_tracking_confidence, dtype=np.float32
    )
    if predicted_uv.ndim != 2 or predicted_uv.shape[1] != 2:
        raise ValueError("canonical UV must have shape (points, 2)")
    if predicted_xyz.shape != (len(predicted_uv), 3):
        raise ValueError("canonical XYZ must have shape (points, 3)")
    if predicted_confidence.shape != (len(predicted_uv),):
        raise ValueError("tracking confidence must have shape (points,)")

    canonical_ids = np.full(len(current_uv), -1, dtype=np.int32)
    tracking_confidence = np.full(len(current_uv), np.nan, dtype=np.float32)
    image_distance = np.full(len(current_uv), np.inf, dtype=np.float32)
    if not len(current_uv) or not len(predicted_uv):
        return ObservedPointAssociations(
            current_uv=current_uv,
            current_xyz=current_xyz,
            canonical_ids=canonical_ids,
            geometry_confidence=current_geometry_confidence,
            tracking_confidence=tracking_confidence,
            image_distance=image_distance,
        )

    finite_uv = np.isfinite(predicted_uv).all(axis=1)
    rounded_uv = np.zeros(predicted_uv.shape, dtype=np.int64)
    rounded_uv[finite_uv] = np.rint(predicted_uv[finite_uv]).astype(np.int64)
    predicted_valid = (
        finite_uv
        & np.isfinite(predicted_xyz).all(axis=1)
        & np.isfinite(predicted_confidence)
        & (predicted_confidence >= minimum_tracking_confidence)
        & (rounded_uv[:, 0] >= 0)
        & (rounded_uv[:, 0] < width)
        & (rounded_uv[:, 1] >= 0)
        & (rounded_uv[:, 1] < height)
    )
    candidate_ids = np.flatnonzero(predicted_valid)
    candidate_pixels = rounded_uv[candidate_ids]
    on_observed_surface = observed_valid[
        candidate_pixels[:, 1], candidate_pixels[:, 0]
    ]
    candidate_ids = candidate_ids[on_observed_surface]
    candidate_pixels = candidate_pixels[on_observed_surface]
    if not len(candidate_ids):
        return ObservedPointAssociations(
            current_uv=current_uv,
            current_xyz=current_xyz,
            canonical_ids=canonical_ids,
            geometry_confidence=current_geometry_confidence,
            tracking_confidence=tracking_confidence,
            image_distance=image_distance,
        )

    candidate_flat = candidate_pixels[:, 1] * width + candidate_pixels[:, 0]
    residual = np.linalg.norm(
        predicted_uv[candidate_ids] - candidate_pixels, axis=1
    )
    order = np.lexsort(
        (
            candidate_ids,
            residual,
            -predicted_confidence[candidate_ids],
            candidate_flat,
        )
    )
    sorted_flat = candidate_flat[order]
    first_per_pixel = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    winners = order[first_per_pixel]

    observation_lookup = np.full(height * width, -1, dtype=np.int64)
    observation_lookup[observed_y * width + observed_x] = np.arange(
        len(current_uv), dtype=np.int64
    )
    observation_indices = observation_lookup[candidate_flat[winners]]
    winning_ids = candidate_ids[winners]
    canonical_ids[observation_indices] = winning_ids.astype(np.int32)
    tracking_confidence[observation_indices] = predicted_confidence[winning_ids]
    image_distance[observation_indices] = residual[winners].astype(np.float32)
    return ObservedPointAssociations(
        current_uv=current_uv,
        current_xyz=current_xyz,
        canonical_ids=canonical_ids,
        geometry_confidence=current_geometry_confidence,
        tracking_confidence=tracking_confidence,
        image_distance=image_distance,
    )


def _bilinear_sample(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float32)
    points = np.asarray(uv, dtype=np.float32)
    scalar = values.ndim == 2
    if scalar:
        values = values[..., None]
    if values.ndim != 3 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("field and UV dimensions are incompatible")
    height, width, channels = values.shape
    output = np.full((len(points), channels), np.nan, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    valid = (
        finite
        & (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )
    indices = np.flatnonzero(valid)
    if len(indices):
        x = points[indices, 0]
        y = points[indices, 1]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        wx = (x - x0)[:, None]
        wy = (y - y0)[:, None]
        top = values[y0, x0] * (1 - wx) + values[y0, x1] * wx
        bottom = values[y1, x0] * (1 - wx) + values[y1, x1] * wx
        output[indices] = top * (1 - wy) + bottom * wy
    return output[:, 0] if scalar else output


def geometry_completeness(
    mask: np.ndarray,
    camera_points: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: float,
) -> float:
    cloth = np.asarray(mask, dtype=bool)
    total = int(cloth.sum())
    if total == 0:
        return float("nan")
    points = np.asarray(camera_points)
    valid = (
        np.isfinite(points).all(axis=-1)
        & (points[..., 2] > 0)
        & np.isfinite(confidence)
        & (np.asarray(confidence) >= threshold)
    )
    return float(np.count_nonzero(cloth & valid) / total)


def _add_track4world_sources() -> None:
    import importlib.util

    for path in (TRACK4WORLD_ROOT, UTILS3D_ROOT, PI3_ROOT):
        if not path.is_dir():
            raise FileNotFoundError(path)
    # utils3d is a package at the checkout root. Register that package directly:
    # adding its parent exposes other checkouts as namespace packages (e.g. sam3)
    # and shadows their editable installations.
    if "utils3d" not in sys.modules:
        spec = importlib.util.spec_from_file_location("utils3d", UTILS3D_ROOT / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils3d"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules["utils3d"]
            raise
    for path in (TRACK4WORLD_ROOT, PI3_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import pi3
    import track4world.nets.external

    sys.modules["track4world.nets.external.pi3"] = pi3


def load_track4world_da3_model(
    *,
    checkpoint: Path,
    da3_snapshot: Path,
    device: str = "cuda",
) -> Any:
    _add_track4world_sources()
    import torch
    from safetensors import safe_open
    from track4world.nets.external.depth_anything_3.api import DepthAnything3
    from track4world.nets.model import Track4World

    checkpoint = checkpoint.expanduser().resolve()
    da3_snapshot = da3_snapshot.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (da3_snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(da3_snapshot / "model.safetensors")
    if not CONVNEXT_CHECKPOINT.is_file():
        raise FileNotFoundError(CONVNEXT_CHECKPOINT)
    cache_key = (str(checkpoint), str(da3_snapshot), str(device))
    cached = _TRACK4WORLD_MODEL_CACHE.get(cache_key)
    if cached is not None:
        diagnostics = cached.physmind_load_diagnostics
        diagnostics["reuse_count"] = int(diagnostics["reuse_count"]) + 1
        diagnostics["last_reuse_seconds"] = 0.0
        return cached

    diagnostics: dict[str, Any] = {
        "cold_load_count": 1,
        "reuse_count": 0,
        "da3_loading_policy": "construct nested DA3 and load only metric branch",
    }
    prefetch = prefetch_weight_files((checkpoint, CONVNEXT_CHECKPOINT))
    diagnostics["weight_prefetch"] = prefetch.to_dict()
    torch.hub.set_dir(str(TORCH_HUB_ROOT))
    config = json.loads(
        (TRACK4WORLD_ROOT / "track4world/config/eval/v1.json").read_text(
            encoding="utf-8"
        )
    )

    def load_pinned_da3(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        if str(path) != DA3_REPO_ID:
            raise ValueError(f"unexpected DA3 model request: {path}")
        if args or kwargs:
            raise ValueError("the pinned DA3 loader does not accept overrides")
        started = time.perf_counter()
        config = json.loads(
            (da3_snapshot / "config.json").read_text(encoding="utf-8")
        )
        backbone = DepthAnything3(model_name=str(config["model_name"]))
        metric_state = {}
        safetensor_path = da3_snapshot / "model.safetensors"
        with safe_open(safetensor_path, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                if key.startswith("model.da3_metric."):
                    metric_state[key] = archive.get_tensor(key)
        missing, unexpected = backbone.load_state_dict(metric_state, strict=False)
        missing_metric = [
            key for key in missing if key.startswith("model.da3_metric.")
        ]
        if missing_metric or unexpected:
            raise RuntimeError(
                "selective DA3 metric loading was incomplete: "
                f"missing={missing_metric[:5]}, unexpected={unexpected[:5]}"
            )
        diagnostics["da3_metric_tensor_count"] = len(metric_state)
        diagnostics["da3_metric_load_seconds"] = time.perf_counter() - started
        return backbone

    construct_started = time.perf_counter()
    with patch.object(
        DepthAnything3,
        "from_pretrained",
        side_effect=load_pinned_da3,
    ):
        model = Track4World(
            **config["model"],
            seqlen=16,
            use_3d=True,
            use_model="depthanythingv3",
        )
    diagnostics["track4world_construct_seconds"] = (
        time.perf_counter() - construct_started
    )
    checkpoint_started = time.perf_counter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    diagnostics["checkpoint_deserialize_seconds"] = (
        time.perf_counter() - checkpoint_started
    )
    remap_started = time.perf_counter()
    missing, unexpected = model.load_pretrained_with_remap(state)
    missing_anyview = [
        key for key in missing if key.startswith("backbone.model.da3.")
    ]
    if missing_anyview:
        raise RuntimeError(
            "Track4World checkpoint did not completely initialize the DA3 "
            f"any-view branch: {missing_anyview[:5]}"
        )
    diagnostics["checkpoint_remap_seconds"] = time.perf_counter() - remap_started
    diagnostics["checkpoint_missing_key_count"] = len(missing)
    diagnostics["checkpoint_unexpected_key_count"] = len(unexpected)
    del state
    model.use_metric_scale = True
    cuda_started = time.perf_counter()
    model.to(device)
    diagnostics["model_to_device_seconds"] = time.perf_counter() - cuda_started
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    diagnostics["cold_load_total_seconds"] = sum(
        float(diagnostics[key])
        for key in (
            "track4world_construct_seconds",
            "checkpoint_deserialize_seconds",
            "checkpoint_remap_seconds",
            "model_to_device_seconds",
        )
    ) + float(prefetch.elapsed_seconds)
    model.physmind_load_diagnostics = diagnostics
    _TRACK4WORLD_MODEL_CACHE[cache_key] = model
    return model


@contextmanager
def _capture_geometry_confidence(
    model: Any,
    *,
    geometry_threshold: float,
) -> Iterator[list[Any]]:
    captured: list[Any] = []
    original = model.get_fmaps

    def wrapped_get_fmaps(_self: Any, *args: Any, **kwargs: Any) -> Any:
        outputs = original(*args, **kwargs)
        captured.append(outputs[5].detach().float().cpu())
        model.mask_threshold = geometry_threshold
        return outputs

    model.get_fmaps = MethodType(wrapped_get_fmaps, model)
    try:
        yield captured
    finally:
        model.get_fmaps = original


def run_track4world_branch(
    model: Any,
    frames_rgb: np.ndarray,
    frame_indices: Sequence[int],
    *,
    device: str = "cuda",
    inference_iterations: int = 4,
    geometry_threshold: float = 0.0,
) -> Track4WorldBranchResult:
    import torch

    frames = np.asarray(frames_rgb, dtype=np.uint8)
    indices = np.asarray(frame_indices, dtype=np.int32)
    if len(frames) != len(indices) or not len(frames):
        raise ValueError("frames and frame indices must have the same nonzero length")
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    tensor = tensor.unsqueeze(0).float().to(device)
    with _capture_geometry_confidence(
        model,
        geometry_threshold=geometry_threshold,
    ) as captured:
        with torch.inference_mode():
            outputs, _ = model.infer(
                tensor,
                iters=inference_iterations,
                sw=None,
                is_training=False,
                tracking3d=True,
                force_projection=True,
            )
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one geometry-confidence capture, found {len(captured)}"
        )

    geometry, motion = outputs

    def cpu_array(value: Any) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    intrinsics = cpu_array(geometry["intrinsics"])
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    if intrinsics.shape != (len(frames), 3, 3):
        raise RuntimeError(f"unexpected intrinsics shape: {intrinsics.shape}")
    intrinsics = intrinsics.copy()
    height, width = frames.shape[1:3]
    intrinsics[:, 0, :] *= width
    intrinsics[:, 1, :] *= height
    camera_poses = cpu_array(geometry["camera_poses"])
    if camera_poses.ndim == 4:
        camera_poses = camera_poses[0]
    if camera_poses.shape != (len(frames), 4, 4):
        raise RuntimeError(f"unexpected camera-pose shape: {camera_poses.shape}")
    camera_points = cpu_array(geometry["points"])[0]
    world_points = cpu_array(geometry["world_points"])[0]
    geometry_valid = cpu_array(geometry["mask"])[0].astype(bool)
    confidence = cpu_array(captured[0])
    confidence = confidence.reshape(
        -1, confidence.shape[-2], confidence.shape[-1]
    )[: len(frames)]
    uv = cpu_array(motion["flow_2d"])[0].transpose(0, 2, 3, 1)
    xyz = cpu_array(motion["flow_3d"])[0]
    visibility_channels = cpu_array(motion["visconf_maps_e"])[0]
    tracking_confidence = (
        visibility_channels[:, 0] * visibility_channels[:, 1]
    )
    del outputs, tensor
    torch.cuda.empty_cache()
    return Track4WorldBranchResult(
        frame_indices=indices,
        rgb=frames,
        intrinsics=intrinsics.astype(np.float32),
        camera_poses=camera_poses.astype(np.float32),
        camera_points=camera_points,
        world_points=world_points,
        geometry_confidence=confidence,
        geometry_valid=geometry_valid,
        anchor_to_frame_uv=uv,
        anchor_to_frame_xyz=xyz,
        tracking_confidence=tracking_confidence,
    )


def run_adjacent_track4world_tracking(
    model: Any,
    frames_rgb: np.ndarray,
    *,
    device: str = "cuda",
    inference_iterations: int = 4,
) -> AdjacentTrack4WorldResult:
    """Infer all chronological ``t -> t+1`` fields in one model call."""

    import torch

    frames = np.asarray(frames_rgb, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2:
        raise ValueError("adjacent tracking requires at least two RGB frames")
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    tensor = tensor.unsqueeze(0).float().to(device)
    with torch.inference_mode():
        with torch.autocast(
            device_type=tensor.device.type,
            dtype=torch.float16,
            enabled=tensor.is_cuda,
        ):
            outputs = model.infer_pair(
                tensor,
                iters=inference_iterations,
                sw=None,
                is_training=False,
                tracking3d=True,
                force_projection=True,
            )
    _, motion = outputs

    def cpu_array(value: Any) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    uv = cpu_array(motion["flow_2d"])[0].transpose(0, 2, 3, 1)
    xyz = cpu_array(motion["flow_3d"])[0]
    visibility_channels = cpu_array(motion["visconf_maps_e"])[0]
    tracking_confidence = visibility_channels[:, 0] * visibility_channels[:, 1]
    pair_count = len(frames) - 1
    height, width = frames.shape[1:3]
    if uv.shape != (pair_count, height, width, 2):
        raise RuntimeError(f"unexpected adjacent UV shape: {uv.shape}")
    if tracking_confidence.shape != (pair_count, height, width):
        raise RuntimeError(
            "unexpected adjacent confidence shape: "
            f"{tracking_confidence.shape}"
        )
    del outputs, tensor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return AdjacentTrack4WorldResult(
        frame_count=len(frames),
        source_to_target_uv=uv,
        tracking_confidence=tracking_confidence,
    )


def scene_geometry_from_chronological_branch(
    branch: Track4WorldBranchResult,
    transform: Track4WorldTransform,
) -> SceneGeometryResult:
    """Keep one branch's shared geometry and discard its anchor-specific tracks."""

    expected = np.arange(len(branch.frame_indices), dtype=np.int32)
    if not np.array_equal(branch.frame_indices, expected):
        raise ValueError("scene geometry branch must cover all frames chronologically")
    return SceneGeometryResult(
        transform=transform,
        rgb=branch.rgb,
        intrinsics=branch.intrinsics,
        camera_poses=branch.camera_poses,
        camera_points=branch.camera_points,
        world_points=branch.world_points,
        geometry_confidence=branch.geometry_confidence,
        geometry_valid=branch.geometry_valid,
    )


def run_chronological_scene_geometry(
    model: Any,
    frames_rgb: np.ndarray,
    *,
    transform: Track4WorldTransform,
    device: str = "cuda",
    inference_iterations: int = 4,
    geometry_threshold: float = 0.0,
) -> SceneGeometryResult:
    """Reconstruct every frame once in one temporally consistent world."""

    frames = np.asarray(frames_rgb, dtype=np.uint8)
    branch = run_track4world_branch(
        model,
        frames,
        np.arange(len(frames), dtype=np.int32),
        device=device,
        inference_iterations=inference_iterations,
        geometry_threshold=geometry_threshold,
    )
    return scene_geometry_from_chronological_branch(branch, transform)
