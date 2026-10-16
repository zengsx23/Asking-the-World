"""Strict handoff contract from SAM3 segmentation to point-cloud tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "physmind.pointcloud_tracking_input"
SCHEMA_VERSION = 2
PHYSICAL_TYPES = frozenset({"rigid", "cloth", "soft_body", "rope"})
RIGID_SHAPES = frozenset({"sphere", "cylinder", "box", "irregular"})


@dataclass(frozen=True, slots=True)
class PointCloudTrackingObject:
    """One SAM3 identity and its audited physical behavior."""

    target_id: str
    track_id: str
    physical_type: str
    physical_type_confidence: float
    physical_type_reason: str
    rigid_shape: str | None
    rigid_shape_confidence: float | None
    rigid_shape_reason: str | None


@dataclass(frozen=True, slots=True)
class RigidSurfaceDecision:
    """First-pass scene-level relationship consumed by gravity estimation."""

    all_rigid_objects_share_one_surface: bool
    scene_surface_type: str
    contact_alignment_type: str
    confidence: float
    evidence: str


@dataclass(frozen=True, slots=True)
class PointCloudTrackingInput:
    """Resolved and validated inputs consumed by point-cloud tracking."""

    manifest_path: Path
    video_path: Path
    frame_count: int
    width: int
    height: int
    fps: float
    mask_width: int
    mask_height: int
    mask_archive_path: Path
    mask_index_path: Path
    inventory_path: Path
    tracking_manifest_path: Path
    objects: tuple[PointCloudTrackingObject, ...]
    rigid_surface_decision: RigidSurfaceDecision | None

    @property
    def rigid_objects(self) -> tuple[PointCloudTrackingObject, ...]:
        return tuple(item for item in self.objects if item.physical_type == "rigid")

    @property
    def cloth_objects(self) -> tuple[PointCloudTrackingObject, ...]:
        return tuple(item for item in self.objects if item.physical_type == "cloth")

    @property
    def soft_body_objects(self) -> tuple[PointCloudTrackingObject, ...]:
        return tuple(
            item for item in self.objects if item.physical_type == "soft_body"
        )


def _load_rigid_surface_decision(
    raw_relation: Any,
) -> RigidSurfaceDecision | None:
    relation = _require_mapping(raw_relation, "rigid_surface_relation")
    status = relation.get("status")
    if status == "not_applicable":
        reason = str(relation.get("reason", "")).strip()
        if not reason:
            raise ValueError("not-applicable rigid surface relation needs a reason")
        return None
    if status != "evaluated":
        raise ValueError(f"invalid rigid surface relation status: {status!r}")
    initial = _require_mapping(relation.get("initial"), "initial rigid surface")
    all_share = initial.get("all_rigid_objects_share_one_surface")
    if not isinstance(all_share, bool):
        raise TypeError("all_rigid_objects_share_one_surface must be a boolean")
    scene_surface_type = str(initial.get("scene_surface_type", "")).strip()
    expected_alignment = {
        "support_plane": "bottom_alignment",
        "wall_plane": "side_or_back_alignment",
    }.get(scene_surface_type)
    if expected_alignment is None:
        raise ValueError(f"invalid scene_surface_type: {scene_surface_type!r}")
    contact_alignment_type = str(
        initial.get("contact_alignment_type", "")
    ).strip()
    if contact_alignment_type != expected_alignment:
        raise ValueError("contact alignment does not match scene surface type")
    raw_confidence = initial.get("confidence")
    if isinstance(raw_confidence, bool):
        raise TypeError("rigid surface confidence must be numeric")
    confidence = float(raw_confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("rigid surface confidence must be in [0, 1]")
    evidence = str(initial.get("evidence", "")).strip()
    if not evidence:
        raise ValueError("rigid surface evidence must be non-empty")
    return RigidSurfaceDecision(
        all_rigid_objects_share_one_surface=all_share,
        scene_surface_type=scene_surface_type,
        contact_alignment_type=contact_alignment_type,
        confidence=confidence,
        evidence=evidence,
    )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _resolve_existing_path(root: Path, raw_path: Any, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(raw_path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def load_pointcloud_tracking_input(
    manifest_path: str | Path,
) -> PointCloudTrackingInput:
    """Load a SAM3 handoff manifest and reject incomplete or inconsistent input."""

    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"point-cloud tracking input does not exist: {manifest}")
    payload = _require_mapping(
        json.loads(manifest.read_text(encoding="utf-8")), "handoff manifest"
    )
    if payload.get("schema") != SCHEMA_NAME:
        raise ValueError(f"unsupported handoff schema: {payload.get('schema')!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "unsupported handoff schema version: "
            f"{payload.get('schema_version')!r}"
        )

    video = _require_mapping(payload.get("video"), "video")
    artifacts = _require_mapping(payload.get("sam3_artifacts"), "sam3_artifacts")
    mask_resolution = _require_mapping(
        payload.get("mask_resolution"), "mask_resolution"
    )
    frame_count = int(video.get("frame_count", 0))
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    fps = float(video.get("fps", 0.0))
    expected_mask_width = int(mask_resolution.get("width", 0))
    expected_mask_height = int(mask_resolution.get("height", 0))
    if frame_count < 1 or width < 1 or height < 1 or fps <= 0.0:
        raise ValueError(
            "video metadata must contain positive frame_count, width, height, and fps"
        )
    if expected_mask_width < 1 or expected_mask_height < 1:
        raise ValueError("mask resolution must contain positive width and height")

    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("handoff manifest must contain at least one object")
    objects: list[PointCloudTrackingObject] = []
    target_ids: set[str] = set()
    track_ids: set[str] = set()
    for index, raw_item in enumerate(raw_objects):
        item = _require_mapping(raw_item, f"objects[{index}]")
        target_id = str(item.get("target_id", "")).strip()
        track_id = str(item.get("track_id", "")).strip()
        physical_type = str(item.get("physical_type", "")).strip()
        reason = str(item.get("physical_type_reason", "")).strip()
        if not target_id or Path(target_id).name != target_id:
            raise ValueError(f"unsafe or empty target_id: {target_id!r}")
        if not track_id:
            raise ValueError(f"missing track_id for {target_id}")
        if target_id in target_ids:
            raise ValueError(f"duplicate target_id: {target_id}")
        if track_id in track_ids:
            raise ValueError(f"duplicate track_id: {track_id}")
        if physical_type not in PHYSICAL_TYPES:
            raise ValueError(
                f"invalid physical_type for {target_id}: {physical_type!r}"
            )
        raw_confidence = item.get("physical_type_confidence")
        if isinstance(raw_confidence, bool):
            raise ValueError(f"invalid physical type confidence for {target_id}")
        confidence = float(raw_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid physical type confidence for {target_id}")
        if not reason:
            raise ValueError(f"missing physical type reason for {target_id}")
        missing_shape_fields = [
            field
            for field in (
                "rigid_shape",
                "rigid_shape_confidence",
                "rigid_shape_reason",
            )
            if field not in item
        ]
        if missing_shape_fields:
            raise ValueError(
                f"missing rigid shape fields for {target_id}: {missing_shape_fields}"
            )
        rigid_shape = item["rigid_shape"]
        rigid_shape_confidence = item["rigid_shape_confidence"]
        rigid_shape_reason = item["rigid_shape_reason"]
        if physical_type == "rigid":
            rigid_shape = str(rigid_shape).strip()
            if rigid_shape not in RIGID_SHAPES:
                raise ValueError(
                    f"invalid rigid_shape for {target_id}: {rigid_shape!r}"
                )
            if isinstance(rigid_shape_confidence, bool):
                raise ValueError(f"invalid rigid shape confidence for {target_id}")
            rigid_shape_confidence = float(rigid_shape_confidence)
            if not 0.0 <= rigid_shape_confidence <= 1.0:
                raise ValueError(f"invalid rigid shape confidence for {target_id}")
            rigid_shape_reason = str(rigid_shape_reason).strip()
            if not rigid_shape_reason:
                raise ValueError(f"missing rigid shape reason for {target_id}")
        elif any(
            value is not None
            for value in (rigid_shape, rigid_shape_confidence, rigid_shape_reason)
        ):
            raise ValueError(
                f"non-rigid object {target_id} must have null rigid shape fields"
            )
        target_ids.add(target_id)
        track_ids.add(track_id)
        objects.append(
            PointCloudTrackingObject(
                target_id=target_id,
                track_id=track_id,
                physical_type=physical_type,
                physical_type_confidence=confidence,
                physical_type_reason=reason,
                rigid_shape=rigid_shape,
                rigid_shape_confidence=rigid_shape_confidence,
                rigid_shape_reason=rigid_shape_reason,
            )
        )

    root = manifest.parent
    mask_index_path = _resolve_existing_path(
        root, artifacts.get("mask_index"), "SAM3 mask index"
    )
    mask_index = _require_mapping(
        json.loads(mask_index_path.read_text(encoding="utf-8")), "SAM3 mask index"
    )
    indexed_tracks: set[str] = set()
    for key, raw_metadata in mask_index.items():
        metadata = _require_mapping(raw_metadata, f"mask index entry {key}")
        track_id = str(metadata.get("track_id", ""))
        frame_index = int(metadata.get("frame_index", -1))
        if track_id not in track_ids:
            raise ValueError(f"mask index contains unknown track_id: {track_id}")
        if not 0 <= frame_index < frame_count:
            raise ValueError(f"mask index frame is outside the video: {frame_index}")
        if (int(metadata.get("height", 0)), int(metadata.get("width", 0))) != (
            expected_mask_height,
            expected_mask_width,
        ):
            raise ValueError(
                f"inconsistent SAM3 mask size for {key}: "
                f"{metadata.get('width')}x{metadata.get('height')} != "
                f"{expected_mask_width}x{expected_mask_height}"
            )
        indexed_tracks.add(track_id)
    missing_tracks = sorted(track_ids - indexed_tracks)
    if missing_tracks:
        raise ValueError(f"SAM3 mask index has no masks for tracks: {missing_tracks}")

    return PointCloudTrackingInput(
        manifest_path=manifest,
        video_path=_resolve_existing_path(root, video.get("path"), "source video"),
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        mask_width=expected_mask_width,
        mask_height=expected_mask_height,
        mask_archive_path=_resolve_existing_path(
            root, artifacts.get("mask_archive"), "SAM3 mask archive"
        ),
        mask_index_path=mask_index_path,
        inventory_path=_resolve_existing_path(
            root, artifacts.get("inventory"), "SAM3 inventory"
        ),
        tracking_manifest_path=_resolve_existing_path(
            root, artifacts.get("tracking_manifest"), "SAM3 tracking manifest"
        ),
        objects=tuple(objects),
        rigid_surface_decision=_load_rigid_surface_decision(
            payload.get("rigid_surface_relation")
        ),
    )
