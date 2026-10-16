"""Gemini 3 Flash decisions used by the Primary-only SAM3 pipeline."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from atw.shared.vlm_planner import _read_env_value
from atw.shared.vlm_usage import OpenRouterJSONClient, Transport


EXPECTED_MODEL = "google/gemini-3-flash-preview"

INVENTORY_SYSTEM = """You are a meticulous video object-discovery agent. Enumerate
distinct physical entities inside the primary experimental interaction workspace: the
support region where deliberately placed entities move, fall, collide, deform, or are
covered. Include visually distinct apparatus components that constrain the experiment,
including fixed obstacles, support blocks, and dividers. Do not exclude a component
merely because it is stationary or supports another object. Exclude the overall
tabletop, floor, background wall, and unrelated furniture that host the experiment;
do not split these continuous environment surfaces into objects. Also exclude background, shadows,
reflections, textures, annotations, and rendering artifacts. Inspect the whole video
because workspace entities can enter, leave, overlap, deform, or become occluded.
Return strict JSON only."""

PHYSICAL_TYPE_GUIDANCE = """Classify physical_type using the object's behavior over
time, especially the frames immediately before, during, and after contact:
- rigid: the intrinsic 3D shape remains fixed while the object translates or rotates;
- soft_body: a bulk, volumetric deformable or elastic object whose material shape
  changes through compression, flattening, stretching, bulging, indentation, or
  squashing. Deformation may be brief or subtle and followed by recovery. Examples
  include elastic balls, rubber balls, sponges, foam objects, and jelly-like objects.
  Do not put rope-like, slender flexible cords or cables in soft_body; use rope.
- rope: a slender, line-like flexible object such as a rope, cord, string, or cable.
  Its overall configuration changes mainly by bending, winding, wrapping, or becoming
  taut or slack, rather than the bulk shape deformation of a soft_body.
- cloth: a thin open sheet or fabric that folds, drapes, wrinkles, or flaps.

A ball or sphere is not automatically a soft body: rigid balls and rigid spheres also
exist. Conversely, an elastic ball can remain approximately spherical for most of the
video and still be a soft body if it visibly deforms at impact. Decide from observed
temporal evidence, not from the category noun. Do not mistake viewpoint change, rigid
3D rotation, partial occlusion, overlap, illumination, or perspective for deformation."""

PHYSICAL_TYPES = frozenset({"rigid", "soft_body", "cloth", "rope"})
RIGID_SHAPES = frozenset({"sphere", "cylinder", "box", "irregular"})

SHAPE_CATEGORY_GUIDANCE = """Classify geometry using these four categories:
- sphere: a spherical or ball-shaped object;
- cylinder: an object whose principal body is cylindrical;
- box: a cube, cuboid, rectangular block, or rectangular prism;
- irregular: any other geometry."""

RIGID_SHAPE_GUIDANCE = f"""For every object whose physical_type is rigid, also
classify its intrinsic geometry:
{SHAPE_CATEGORY_GUIDANCE}

Judge the complete intrinsic shape rather than the visible silhouette in one frame.
This label describes geometry only; it must not change physical_type. For cloth, rope, and
soft_body objects, rigid_shape, rigid_shape_confidence, and rigid_shape_reason must
all be null."""

INVENTORY_INSTRUCTION = f"""Inventory every independent physical entity participating
inside the primary experimental interaction workspace. Include both rigid and
deformable entities and keep same-category instances separate. Include visually distinct apparatus components that constrain the experiment,
including fixed obstacles, support blocks, and dividers. Do not exclude a component
merely because it is stationary or supports another object. Exclude the overall
tabletop, floor, background wall, and unrelated furniture that host the experiment;
do not split these continuous environment surfaces into objects.

For each entity return:
- target_id: unique lower_snake_case visual identity, not a dataset ID;
- category_noun: concise ordinary noun;
- discriminative_description: appearance and relative location sufficient to
  distinguish this instance without defining it by another object's mask;
- short_noun_phrases: 2 to 4 simple singular noun phrases suitable for an
  open-vocabulary segmenter; no actions, counting, long relationships, or sentences;
- discovery_evidence: short explanation of where it is seen;
- physical_type: exactly one of rigid, soft_body, cloth, or rope;
- physical_type_confidence: number from 0 to 1;
- physical_type_reason: concise observed temporal evidence from the video.
- rigid_shape: exactly one of sphere, cylinder, box, or irregular for rigid objects,
  and null for non-rigid objects;
- rigid_shape_confidence: number from 0 to 1 for rigid objects, and null otherwise;
- rigid_shape_reason: concise shape evidence for rigid objects, and null otherwise.

{PHYSICAL_TYPE_GUIDANCE}

{RIGID_SHAPE_GUIDANCE}

Return {{"objects": [...], "excluded_regions": [...], "notes": "..."}}. Do not infer
the desired object count in advance. The final attached image is a uniformly sampled
contact sheet with exact frame labels. A separate step will select one Primary frame."""

AUDIT_INSTRUCTION = f"""Audit the proposed workspace-object inventory below against
the entire video and labeled contact sheet. Look especially for small, thin, dark,
briefly visible, partially occluded, same-category, and deforming entities. Merge
duplicates, split merged independent entities, add omissions, and remove artifacts.
Separate disconnected rigid obstacles even when their color and static state match.

Include visually distinct apparatus components that constrain the experiment,
including fixed obstacles, support blocks, and dividers. Do not exclude a component
merely because it is stationary or supports another object. Exclude the overall
tabletop, floor, background wall, and unrelated furniture that host the experiment;
do not split these continuous environment surfaces into objects. Keep each retained target_id stable.

Independently re-evaluate physical_type for every final object using this guidance:
{PHYSICAL_TYPE_GUIDANCE}

Independently re-evaluate rigid_shape for every rigid object using this guidance:
{RIGID_SHAPE_GUIDANCE}

Preserve all fields, including physical_type, physical_type_confidence in [0, 1],
physical_type_reason, rigid_shape, rigid_shape_confidence, and rigid_shape_reason.
Return a complete revised
{{"objects": [...], "excluded_regions": [...], "notes": "..."}}.

PROPOSED INVENTORY:
{{inventory}}"""

RIGID_SURFACE_SYSTEM = """You are a careful visual physical-scene analyst. Determine
whether all already inventoried rigid objects share one common physical interaction
surface. Base the decision on temporal physical evidence across the complete video.
Return only valid JSON matching the requested schema. Do not estimate a numeric
gravity direction, camera orientation, plane normal, roll, pitch, or image-space
direction."""

RIGID_SURFACE_INSTRUCTION = """The object inventory below is already complete. Answer
one scene-level question: are ALL objects whose physical_type is exactly rigid located
on the same common support plane or wall plane?

Definitions:
- support_plane: a load-bearing floor, tray, platform, or tabletop. The scene's
  physical motion unfolds on top of this surface.
- wall_plane: a wall-like or board-like interaction plane. The scene's physical
  motion and rigid structures are constrained within this plane, with rigid objects
  mounted or aligned through their side/back geometry.

How to decide:
1. Use temporal physical evidence across the complete video, including falling,
   tipping, rolling, sliding, collisions, contact constraints, and settling.
2. Distinguish the common interaction surface from collision boundaries or obstacles
   located inside that surface. A visually horizontal collision edge does not by
   itself prove that the common surface is a support plane. Decide whether the scene's
   motion unfolds on top of a load-bearing surface or within a wall-like plane.
3. This is a scene-level relationship, not a test of each object's exact contact
   topology. A rigid sphere still counts as being on a support plane even though its
   geometric contact is a point rather than a planar face.
4. Treat object names, colors, labels, apparent image height, and apparent image
   orientation as non-authoritative.
5. Cloth and soft_body objects are not counted in "all rigid objects", although their
   motion may be used as physical evidence.
6. Do not list or classify individual object IDs. Do not estimate any direction.

Return exactly:
{
  "all_rigid_objects_share_one_surface": true,
  "scene_surface_type": "support_plane",
  "contact_alignment_type": "bottom_alignment",
  "confidence": number from 0 to 1,
  "evidence": "concise temporal physical evidence"
}

The example above is strict JSON. scene_surface_type must be "support_plane" or
"wall_plane". contact_alignment_type must be "bottom_alignment" for a support_plane
and "side_or_back_alignment" for a wall_plane. Return one top-level JSON object,
never an array and never any per-object list.

OBJECT INVENTORY:
{inventory}
"""

PRIMARY_SELECTION_SYSTEM = """Select one Primary seed frame for each workspace entity,
using only printed frame numbers. Confirm visibility, then prioritize the entire
outline being inside the image before comparing occlusion and identity clarity.
An outline cut by any image edge is incomplete. State image position and outline
completeness in the reason. Return strict JSON only."""

SEED_SELECTION_SYSTEM = """You are an instance-mask selection agent. Compare the raw
image with all labeled SAM3 candidate masks. Select a candidate only if it covers the
requested whole foreground entity and excludes adjacent entities and
support/background. Thin, dark, deforming, and partially occluded objects still count.
Return strict JSON only."""

SEED_PROMPT_EXPANSION_SYSTEM = """You plan image-grounded text prompts to segment
one fixed foreground instance. Use the supplied image and prior results to propose
alternative names for the same entity. Return strict JSON only."""

SEED_PROMPT_EXPANSION = """The current SAM3 candidates did not provide an acceptable
mask for the requested foreground entity. Inspect the SAME Primary frame, the
fixed target description, previously attempted phrases, and the selection feedback.
If a candidate sheet is supplied, use it to understand what went wrong. An empty
candidate set or rejected candidates do not by themselves mean the target is absent.
Propose 2 to 4 NEW, distinct short noun phrases for this SAME entity, avoiding all
previously attempted phrases. Ground the alternatives in visible appearance: try
another ordinary category noun, simplify an overly specific name, or use a useful
visual qualifier. Do not change the target, frame, physical category, or thresholds;
do not broaden the target to a connected assembly, neighboring objects, or background.
Return {"short_noun_phrases":["...","..."],"reason":"..."}.
"""

BACKGROUND_SURFACE_SYSTEM = """You identify the environment surface of an already
classified interaction workspace from images. The supplied surface type is fixed;
do not reclassify it or plan segmentation of the other surface type. Use visual
evidence, without dataset-specific assumptions about colors, materials, viewpoint,
or scene layout. Return strict JSON only."""

BACKGROUND_SURFACE_PLAN = """Inspect the supplied first, middle, and last frames.
The known interaction surface type is {surface_type}.
{surface_definition}

Identify ONE common target surface across the frames. Describe its visible extent
and appearance, then propose 2 to 4 diverse short noun phrases for SAM3 to segment
that SAME surface. Choose the nouns and optional visual qualifiers from these
images, not from a predetermined vocabulary. Try useful alternative names or
granularity because synonymous prompts can produce different masks. Prefer simple
noun phrases, not sentences, actions, frame indices, or complex relationships.
For furniture, target the supporting planar part, not the whole furniture assembly.
Do not target the foreground entities, their shadows/reflections, or a distinct
background surface. Occluded portions need not be hallucinated. If the target is
not visibly identifiable in any frame, set visible=false and return no phrases.
Do not estimate a plane normal, gravity, camera orientation, or select new frames.
Return {"visible":true,"target_description":"...",
"short_noun_phrases":["...","..."],"reason":"..."}.
"""

BACKGROUND_SURFACE_SELECTION = """Select masks for the planned environment surface
in each requested frame. The supplied surface type and target description are
fixed. Compare every numbered SAM3 candidate with the original frame; its text
prompt does not establish correctness. Select the most complete clean
mask of the visible target. Complementary candidates may be combined only if they
cover different visible parts of the SAME plane; avoid redundant copies. Reject
masks of foreground entities, a different surface, furniture legs/sides, shadows
alone, or broad mixtures of unrelated surfaces. Do not reject an otherwise correct
surface merely because its texture or illumination varies. Moving entities can
occlude the surface; do not require hidden portions to be filled in.
Candidate panels keep original colors inside the mask and dim the outside. The
red contour is a display annotation, not scene evidence. If no candidate is usable
for a frame, return an empty list. Do not invent IDs or propagate between frames.
Return {"frames":[{"frame":0,"candidate_ids":[],"confidence":0.0,
"reason":"..."}],"notes":"..."}, with exactly one row per requested frame.
"""

BACKGROUND_SURFACE_EXPANSION = """The current SAM3 results do not provide enough
usable observations of the already identified environment surface. Inspect the
original frames and the supplied per-prompt outcomes, masks, and rejection reasons.
Keep the supplied surface type and physical target fixed. Propose 2 to 4 NEW short
noun phrases for the SAME visible surface, avoiding phrases already attempted.
Adapt to the failure: simplify overly specific descriptions, try a different
ordinary category name, or use a different visually supported qualifier. Select
these words from the images, not from a fixed dataset vocabulary. Do not broaden
the target to foreground entities, another surface, or an entire furniture
assembly. A lack of SAM3 candidates alone does not prove that the surface is absent.
Only the listed missing frames will be retried; accepted frames are retained.
Do not change frames, thresholds, physical labels, or infer hidden geometry.
Return {"short_noun_phrases":["...","..."],"reason":"..."}.
If visual evidence clearly rules out a visible target in all missing frames,
return an empty list and explain that evidence instead of inventing a surface.
"""


def _media_data_url(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        mime_type = "video/mp4"
        kind = "video_url"
    elif suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
        kind = "image_url"
    elif suffix == ".png":
        mime_type = "image/png"
        kind = "image_url"
    else:
        raise ValueError(f"unsupported Gemini media type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return kind, f"data:{mime_type};base64,{encoded}"


class NoVisiblePrimaryFrame(RuntimeError):
    """The frame selector found no usable alternative for the requested entity."""


class GeminiSam3Agent:
    """Identify objects, select primary frames, and choose seed masks."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        transport: Transport | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        config = json.loads(
            (
                self.project_root
                / "configs/llm/openrouter_google_gemini_3_flash_preview.json"
            ).read_text(encoding="utf-8")
        )
        self.model = str(config["model"])
        if self.model != EXPECTED_MODEL:
            raise ValueError(
                f"SAM3 agent requires configured {EXPECTED_MODEL}, found {self.model}"
            )
        self.client = OpenRouterJSONClient(
            api_key=_read_env_value(self.project_root / ".env", "OPEN_ROUTER_KEY"),
            model=self.model,
            max_output_tokens=int(config["max_output_tokens"]),
            module_name="sam3",
            transport=transport,
            max_attempts=max_attempts,
        )
        self.trace: list[dict[str, Any]] = []

    def request_json(
        self,
        *,
        operation: str,
        call_id: str,
        system: str,
        instruction: str,
        media_paths: Iterable[Path],
        object_id: str | None = None,
        validate=None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        media_names: list[str] = []
        for media_path in media_paths:
            resolved = media_path.resolve()
            kind, data_url = _media_data_url(resolved)
            media_names.append(resolved.name)
            content.append({"type": kind, kind: {"url": data_url}})
        from atw.perception.pipeline.recording import CURRENT
        from atw.shared.vlm_usage import REQUEST_RETRY_FACTORY
        recorder = CURRENT.get()
        validation = dict(validate=validate) if REQUEST_RETRY_FACTORY.get() is not None or (recorder is not None and recorder.harness_vlm_retries) else {}
        parsed = self.client.request_json(
            operation=operation,
            call_id=call_id,
            system=system,
            content=content,
            object_id=object_id,
            **validation,
        )
        self.trace.append(
            {
                "call_id": call_id,
                "model": self.model,
                "media": media_names,
                "instruction": instruction,
                "response": parsed,
            }
        )
        return parsed

    def plan_background_surface(
        self, *, surface_type: str, frames: list[int], image_paths: list[Path],
    ) -> dict[str, Any]:
        if surface_type not in {"support_plane", "wall_plane"}:
            raise ValueError("background planning needs the known support/wall label")
        definition = {
            "support_plane": "Locate the load-bearing surface on which the workspace objects rest or move. It may be a tabletop, floor, platform, tray, or another support; identify its actual appearance in these images.",
            "wall_plane": "Locate the wall-like or board-like interaction surface associated with the workspace, where objects are mounted or aligned through side/back geometry; identify its actual appearance in these images.",
        }[surface_type]
        plan = self.request_json(
            operation="background_surface_prompt_planning", call_id="background_surface_plan",
            system=BACKGROUND_SURFACE_SYSTEM,
            instruction=BACKGROUND_SURFACE_PLAN.replace("{surface_type}", surface_type)
                .replace("{surface_definition}", definition)
                + f"\nImage frame order: {frames}", media_paths=image_paths,
            validate=lambda value: self._normalize_background_plan(value, surface_type),
        )
        return self._normalize_background_plan(plan, surface_type)

    @staticmethod
    def _normalize_background_plan(plan, surface_type):
        visible = plan.get("visible")
        phrases = plan.get("short_noun_phrases")
        if not isinstance(visible, bool) or not isinstance(phrases, list):
            raise ValueError("background plan needs visibility and a phrase list")
        if any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError("background phrases must be nonempty strings")
        phrases = list(dict.fromkeys(p.strip() for p in phrases))
        if (visible and not 2 <= len(phrases) <= 4) or (not visible and phrases):
            raise ValueError("visible background needs 2–4 unique phrases; invisible needs none")
        if not isinstance(plan.get("target_description"), str) or (visible and not plan["target_description"].strip()):
            raise ValueError("visible background needs a target description")
        return {**plan, "short_noun_phrases": phrases, "scene_surface_type": surface_type}

    def expand_background_surface_prompts(
        self, *, plan: Mapping[str, Any], tried_prompts: list[str],
        missing_frames: list[int], feedback: Mapping[str, Any],
        image_paths: list[Path], candidate_sheet_paths: list[Path], round_index: int,
    ) -> dict[str, Any]:
        expanded = self.request_json(
            operation="background_surface_prompt_expansion", call_id=f"background_surface_expansion_{round_index}",
            system=BACKGROUND_SURFACE_SYSTEM,
            instruction=BACKGROUND_SURFACE_EXPANSION
                + f"\nFixed plan: {json.dumps(plan)}\nAlready tried: {json.dumps(tried_prompts)}"
                + f"\nMissing frames: {missing_frames}\nFeedback: {json.dumps(feedback)}",
            media_paths=image_paths + candidate_sheet_paths,
            validate=lambda value: self._normalize_background_expansion(value, tried_prompts),
        )
        return self._normalize_background_expansion(expanded, tried_prompts)

    @staticmethod
    def _normalize_background_expansion(expanded, tried_prompts):
        phrases = expanded.get("short_noun_phrases")
        if not isinstance(phrases, list) or any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError("expanded background prompts must be a list of nonempty strings")
        tried = {p.casefold() for p in tried_prompts}
        unique = {}
        for phrase in phrases:
            phrase = phrase.strip()
            if phrase.casefold() not in tried:
                unique.setdefault(phrase.casefold(), phrase)
        fresh = list(unique.values())
        if len(fresh) > 4:
            raise ValueError("background expansion allows at most four new phrases")
        return {**expanded, "short_noun_phrases": fresh}

    def choose_background_surface_masks(
        self, *, plan: Mapping[str, Any], frames: list[int], candidates: list[dict[str, Any]],
        image_paths: list[Path], candidate_sheet_paths: list[Path],
        round_index: int = 0,
    ) -> dict[str, Any]:
        selection_candidates = [
            {key: candidate[key] for key in ("id", "frame", "prompts")}
            for candidate in candidates
        ]
        def validate_selection(value):
            rows = value["frames"]
            if sorted(row["frame"] for row in rows) != sorted(frames):
                raise ValueError("background selection must contain each requested frame exactly once")
            candidate_frames = {c["id"]: c["frame"] for c in candidates}
            for row in rows:
                ids = row["candidate_ids"]
                if not isinstance(ids, list) or any(i not in candidate_frames or candidate_frames[i] != row["frame"] for i in ids):
                    raise ValueError("background selection refers to an unknown or wrong-frame candidate")
            return value
        return self.request_json(
            operation="background_surface_mask_selection", call_id=f"background_surface_selection_{round_index}",
            system=BACKGROUND_SURFACE_SYSTEM,
            instruction=BACKGROUND_SURFACE_SELECTION
                + f"\nPlan: {json.dumps(plan)}\nRequested frames: {frames}\nCandidates: {json.dumps(selection_candidates)}",
            media_paths=image_paths + candidate_sheet_paths,
            validate=validate_selection,
        )

    def discover_primary_inventory(
        self,
        *,
        call_prefix: str,
        video_path: Path,
        contact_sheet_path: Path,
        sampled_frames: list[int],
    ) -> dict[str, Any]:
        initial = self.request_json(
            operation="inventory_initial",
            call_id=f"inventory_initial_{call_prefix}",
            system=INVENTORY_SYSTEM,
            instruction=INVENTORY_INSTRUCTION,
            media_paths=[video_path, contact_sheet_path],
        )
        audited = self.request_json(
            operation="inventory_audit",
            call_id=f"inventory_audit_{call_prefix}",
            system=INVENTORY_SYSTEM,
            instruction=AUDIT_INSTRUCTION.replace(
                "{inventory}", json.dumps(initial, ensure_ascii=False, indent=2)
            ),
            media_paths=[video_path, contact_sheet_path],
            validate=lambda result: self._normalize_inventory_objects(result.get("objects")),
        )
        rigid_surface_relation = self.infer_rigid_surface_relation(
            call_prefix=call_prefix,
            video_path=video_path,
            contact_sheet_path=contact_sheet_path,
            audited_inventory=audited,
        )
        primary_selection = self.request_json(
            operation="primary_frame_selection",
            call_id=f"primary_frame_selection_{call_prefix}",
            system=PRIMARY_SELECTION_SYSTEM,
            instruction=f"""For every inventory entity below, choose exactly one printed
Primary frame. Select only from {sampled_frames}. Return
{{"objects": [{{"target_id": "...", "primary_seed_frame": integer,
"reason": "..."}}]}}.

INVENTORY: {json.dumps(audited.get('objects', []), ensure_ascii=False)}""",
            media_paths=[contact_sheet_path],
            validate=lambda result: self._normalize_primary_objects(audited.get("objects"),
                {str(item.get("target_id")): item for item in result.get("objects", [])
                 if isinstance(item, Mapping)}, sampled_frames),
        )
        selections = {
            str(item.get("target_id")): item
            for item in primary_selection.get("objects", [])
            if isinstance(item, Mapping)
        }
        objects = self._normalize_primary_objects(
            audited.get("objects"), selections, sampled_frames
        )
        return {
            "initial": initial,
            "audited": audited,
            "rigid_surface_relation": rigid_surface_relation,
            "primary_selection": primary_selection,
            "sampled_frame_indices": sampled_frames,
            "objects": objects,
        }

    def infer_rigid_surface_relation(
        self,
        *,
        call_prefix: str,
        video_path: Path,
        contact_sheet_path: Path,
        audited_inventory: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Classify the shared surface once; the initial answer is final."""

        raw_objects = audited_inventory.get("objects")
        if not isinstance(raw_objects, list):
            raise ValueError("audited inventory must contain an objects list")
        has_rigid_objects = any(
            isinstance(item, Mapping) and item.get("physical_type") == "rigid"
            for item in raw_objects
        )
        if not has_rigid_objects:
            return {
                "status": "not_applicable",
                "reason": "inventory_contains_no_rigid_objects",
            }

        # Keep category evidence without importing upstream surface interpretations.
        surface_inventory = {
            "objects": [
                {
                    "physical_type": item["physical_type"],
                    "category_noun": item["category_noun"],
                }
                for item in raw_objects
            ]
        }
        inventory_text = json.dumps(
            surface_inventory, ensure_ascii=False, indent=2
        )
        initial_raw = self.request_json(
            operation="rigid_surface_initial",
            call_id=f"rigid_surface_initial_{call_prefix}",
            system=RIGID_SURFACE_SYSTEM,
            instruction=RIGID_SURFACE_INSTRUCTION.replace(
                "{inventory}", inventory_text
            ),
            media_paths=[video_path, contact_sheet_path],
            validate=self._normalize_rigid_surface_decision,
        )
        initial = self._normalize_rigid_surface_decision(initial_raw)
        return {
            "status": "evaluated",
            "initial": initial,
        }

    def judge_seed_frame_visibility(
        self, *, call_prefix: str, target: Mapping[str, Any], frame_index: int,
        image_path: Path, feedback: Mapping[str, Any],
        candidate_sheet_path: Path | None,
    ) -> dict[str, Any]:
        def validate(response):
            if type(response.get("entity_visible")) is not bool:
                raise ValueError("seed frame visibility must be true or false")

        response = self.request_json(
            operation="seed_frame_visibility",
            object_id=str(target["target_id"]),
            call_id=f"seed_visibility_{call_prefix}_{target['target_id']}",
            system="Judge whether one requested entity is visible in an image. Return strict JSON only.",
            instruction=f"""The first seed-mask selection failed. Inspect raw frame {frame_index}
for this fixed entity: {json.dumps(dict(target), ensure_ascii=False)}
Previous result: {json.dumps(feedback.get('selection', feedback), ensure_ascii=False)}
Decide whether the entity itself is visible, not whether a candidate is acceptable.
Visible but missing/incorrect masks means entity_visible=true; out of frame or
fully occluded means false. Do not substitute a neighboring entity. Treat the
previous planning and selection reasons as fallible; ground your decision in the
raw image. State the visible position/appearance or evidence of absence.
Return {{"entity_visible": true/false, "reason": "..."}}.""",
            media_paths=[image_path] + ([candidate_sheet_path] if candidate_sheet_path else []),
            validate=validate,
        )
        validate(response)
        return response

    def reselect_primary_frame(
        self, *, call_prefix: str, target: Mapping[str, Any],
        contact_sheet_path: Path, sampled_frames: list[int],
        failed_frame: int, feedback: Mapping[str, Any],
    ) -> dict[str, Any]:
        allowed = [frame for frame in sampled_frames if frame != failed_frame]
        if not allowed:
            raise NoVisiblePrimaryFrame("no alternative printed Primary frame available")
        def validate(response):
            frame = response.get("primary_seed_frame")
            if frame is not None and (type(frame) is not int or frame not in allowed):
                raise ValueError("reselected Primary must be an allowed printed frame, excluding the failed frame")

        response = self.request_json(
            operation="primary_frame_reselection",
            object_id=str(target["target_id"]),
            call_id=f"primary_frame_reselection_{call_prefix}_{target['target_id']}",
            system=PRIMARY_SELECTION_SYSTEM,
            instruction=f"""Reselect the Primary frame once for this SAME entity:
{json.dumps(dict(target), ensure_ascii=False)}
Failed frame: {failed_frame}. Previous rejection and visibility judgment:
{json.dumps(feedback, ensure_ascii=False)}
Use the original indexed contact sheet. Exclude frame {failed_frame}; choose only
from {allowed}. First prioritize the entire target being inside the image, with
no part of its outline cut off by any image edge; then compare occlusion. Do not
describe a partly out-of-frame target as fully visible.
Do not change the entity. If no suitable visible frame exists,
return null. Return {{"primary_seed_frame": integer or null, "reason": "..."}}.""",
            media_paths=[contact_sheet_path],
            validate=validate,
        )
        frame = response.get("primary_seed_frame")
        if frame is None:
            raise NoVisiblePrimaryFrame(f"no visible alternative Primary frame for {target['target_id']}")
        validate(response)
        return response

    def expand_seed_prompts(
        self, *, call_prefix: str, target: Mapping[str, Any], frame_index: int,
        image_path: Path, tried_phrases: list[str], feedback: Mapping[str, Any],
        candidate_sheet_path: Path | None,
    ) -> dict[str, Any]:
        def normalize(response):
            phrases = response.get("short_noun_phrases")
            if not isinstance(phrases, list) or any(not isinstance(p, str) or not p.strip() for p in phrases):
                raise ValueError("seed prompt expansion needs nonempty noun phrases")
            tried = {p.strip().casefold() for p in tried_phrases}
            fresh = {}
            for phrase in phrases:
                phrase = phrase.strip()
                if phrase.casefold() not in tried:
                    fresh.setdefault(phrase.casefold(), phrase)
            if not 2 <= len(fresh) <= 4:
                raise ValueError("seed prompt expansion needs 2–4 new distinct phrases")
            return {**response, "short_noun_phrases": list(fresh.values())}

        response = self.request_json(
            operation="seed_prompt_expansion",
            object_id=str(target["target_id"]),
            call_id=f"seed_expand_{call_prefix}_{target['target_id']}",
            system=SEED_PROMPT_EXPANSION_SYSTEM,
            instruction=SEED_PROMPT_EXPANSION + "\nFixed target: " + json.dumps({
                key: target[key] for key in
                ("target_id", "category_noun", "discriminative_description")
            }) + f"\nPrimary frame: {frame_index}\nAlready tried: {json.dumps(tried_phrases)}"
                + f"\nPrevious result: {json.dumps(feedback)}",
            media_paths=[image_path] + ([candidate_sheet_path] if candidate_sheet_path else []),
            validate=normalize,
        )
        return normalize(response)

    def choose_seed_mask(
        self,
        *,
        call_prefix: str,
        target: Mapping[str, Any],
        frame_index: int,
        image_path: Path,
        candidate_sheet_path: Path,
        candidate_ids: set[str],
    ) -> dict[str, Any]:
        instruction = f"""Select the best candidate for the supplied Primary frame.
The requested entity is:
target_id: {target['target_id']}
category noun: {target['category_noun']}
description: {target['discriminative_description']}
Primary-frame planning reason: {target.get('primary_seed_reason', '')}

Treat target_id only as an identifier. Do not infer position, size, appearance,
or instance identity from words embedded in it, such as left, right, or tall.
Use the description and planning reason as fallible context for identifying the
requested entity. If they conflict with the images, prioritize visible evidence
in the raw frame and candidate masks.

Media order is: raw frame {frame_index}, candidates for frame {frame_index}.
In each candidate image, the outlined region preserves its original appearance,
the surrounding context is dimmed, and the red contour is only an annotation:
it does not indicate the object's color. Compare candidates with the raw frame
using appearance, shape, position, and instance identity.
Return {{"selections": [{{"frame_index": {frame_index}, "candidate_id": "C00"
or null, "confidence": 0..1, "reason": "..."}}], "entity_visible": true/false,
"notes": "..."}}. Reject every candidate that belongs to another instance or
includes large background/support regions."""
        def normalize(response):
            selections = response.get("selections", [])
            decision = next((item for item in selections if isinstance(item, Mapping)
                             and int(item.get("frame_index", -1)) == frame_index), {})
            candidate_id = decision.get("candidate_id")
            if candidate_id not in candidate_ids:
                candidate_id = None
            return {
                "candidate_id": candidate_id,
                "confidence": float(decision.get("confidence", 0.0)),
                "reason": str(decision.get("reason", "no valid agent selection")),
                "agent_response": response,
            }

        response = self.request_json(
            operation="seed_mask_selection",
            object_id=str(target["target_id"]),
            call_id=f"seed_select_{call_prefix}_{target['target_id']}",
            system=SEED_SELECTION_SYSTEM,
            instruction=instruction,
            media_paths=[image_path, candidate_sheet_path],
            validate=normalize,
        )
        return normalize(response)

    def usage_report(self) -> dict[str, Any]:
        """Return provider-reported token and USD-cost accounting."""

        return self.client.usage_report()

    @staticmethod
    def _normalize_rigid_surface_decision(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("rigid surface decision must be a JSON object")
        required_keys = {
            "all_rigid_objects_share_one_surface",
            "scene_surface_type",
            "contact_alignment_type",
            "confidence",
            "evidence",
        }
        missing_keys = sorted(required_keys - raw.keys())
        if missing_keys:
            raise ValueError(
                f"rigid surface decision omitted fields: {missing_keys}"
            )
        unexpected_keys = sorted(raw.keys() - required_keys)
        if unexpected_keys:
            raise ValueError(
                f"rigid surface decision has unexpected fields: {unexpected_keys}"
            )

        all_share = raw["all_rigid_objects_share_one_surface"]
        if not isinstance(all_share, bool):
            raise TypeError(
                "all_rigid_objects_share_one_surface must be a boolean"
            )
        surface_type = str(raw["scene_surface_type"]).strip()
        expected_alignment = {
            "support_plane": "bottom_alignment",
            "wall_plane": "side_or_back_alignment",
        }.get(surface_type)
        if expected_alignment is None:
            raise ValueError(f"invalid scene_surface_type: {surface_type!r}")
        alignment_type = str(raw["contact_alignment_type"]).strip()
        if alignment_type != expected_alignment:
            raise ValueError(
                "contact_alignment_type does not match scene_surface_type: "
                f"{alignment_type!r} versus {surface_type!r}"
            )
        raw_confidence = raw["confidence"]
        if isinstance(raw_confidence, bool):
            raise TypeError("rigid surface confidence must be numeric")
        confidence = float(raw_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"rigid surface confidence must be in [0, 1], found {confidence}"
            )
        evidence = str(raw["evidence"]).strip()
        if not evidence:
            raise ValueError("rigid surface evidence must be non-empty")
        return {
            "all_rigid_objects_share_one_surface": all_share,
            "scene_surface_type": surface_type,
            "contact_alignment_type": alignment_type,
            "confidence": confidence,
            "evidence": evidence,
        }

    @staticmethod
    def _normalize_inventory_objects(raw_objects: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_objects, list) or not raw_objects:
            raise ValueError("audited inventory must contain objects")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_objects):
            if not isinstance(item, Mapping):
                raise TypeError("inventory object must be a JSON object")
            original_id = str(item.get("target_id", ""))
            raw_id = re.sub(r"[^a-z0-9]+", "_", original_id.lower())
            target_id = raw_id.strip("_") or f"entity_{index:02d}"
            if target_id in seen:
                raise ValueError(f"duplicate target_id after normalization: {target_id}")
            seen.add(target_id)
            category = str(item.get("category_noun", "object")).strip() or "object"
            phrases = [
                str(value).strip()
                for value in item.get("short_noun_phrases", [])
                if str(value).strip()
            ]
            if category not in phrases:
                phrases.append(category)
            if "physical_type" not in item:
                raise ValueError(
                    f"audited inventory omitted physical_type for {target_id}"
                )
            physical_type = str(item["physical_type"]).strip()
            if physical_type not in PHYSICAL_TYPES:
                raise ValueError(
                    f"invalid physical_type for {target_id}: {physical_type!r}"
                )
            if "physical_type_confidence" not in item:
                raise ValueError(
                    "audited inventory omitted physical_type_confidence "
                    f"for {target_id}"
                )
            raw_confidence = item["physical_type_confidence"]
            if isinstance(raw_confidence, bool):
                raise ValueError(
                    f"invalid physical_type_confidence for {target_id}: "
                    f"{raw_confidence!r}"
                )
            confidence = float(raw_confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"invalid physical_type_confidence for {target_id}: {confidence}"
                )
            reason = str(item.get("physical_type_reason", "")).strip()
            if not reason:
                raise ValueError(
                    f"audited inventory omitted physical_type_reason for {target_id}"
                )
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
                    "audited inventory omitted rigid shape fields for "
                    f"{target_id}: {missing_shape_fields}"
                )
            rigid_shape = item["rigid_shape"]
            rigid_shape_confidence = item.get("rigid_shape_confidence")
            rigid_shape_reason = item.get("rigid_shape_reason")
            if physical_type == "rigid":
                rigid_shape = str(rigid_shape).strip()
                if rigid_shape not in RIGID_SHAPES:
                    raise ValueError(
                        f"invalid rigid_shape for {target_id}: {rigid_shape!r}"
                    )
                if isinstance(rigid_shape_confidence, bool):
                    raise ValueError(
                        f"invalid rigid_shape_confidence for {target_id}: "
                        f"{rigid_shape_confidence!r}"
                    )
                rigid_shape_confidence = float(rigid_shape_confidence)
                if not 0.0 <= rigid_shape_confidence <= 1.0:
                    raise ValueError(
                        f"invalid rigid_shape_confidence for {target_id}: "
                        f"{rigid_shape_confidence}"
                    )
                rigid_shape_reason = str(rigid_shape_reason).strip()
                if not rigid_shape_reason:
                    raise ValueError(
                        f"audited inventory omitted rigid_shape_reason for {target_id}"
                    )
            elif any(
                value is not None
                for value in (
                    rigid_shape,
                    rigid_shape_confidence,
                    rigid_shape_reason,
                )
            ):
                raise ValueError(
                    f"non-rigid object {target_id} must have null rigid shape fields"
                )
            normalized.append(
                {
                    "target_id": target_id,
                    "category_noun": category,
                    "discriminative_description": str(
                        item.get("discriminative_description", category)
                    ),
                    "short_noun_phrases": phrases[:4],
                    "discovery_evidence": str(item.get("discovery_evidence", "")),
                    "physical_type": physical_type,
                    "physical_type_confidence": confidence,
                    "physical_type_reason": reason,
                    "rigid_shape": rigid_shape,
                    "rigid_shape_confidence": rigid_shape_confidence,
                    "rigid_shape_reason": rigid_shape_reason,
                }
            )
        return normalized

    @staticmethod
    def _normalize_primary_objects(
        raw_objects: Any,
        selections: Mapping[str, Mapping[str, Any]],
        sampled_frames: list[int],
    ) -> list[dict[str, Any]]:
        normalized = GeminiSam3Agent._normalize_inventory_objects(raw_objects)
        for raw, obj in zip(raw_objects, normalized):
            target_id = obj["target_id"]
            selection = selections.get(str(raw.get("target_id", "")))
            if selection is None or "primary_seed_frame" not in selection:
                raise ValueError(f"Gemini omitted Primary selection for {target_id}")
            primary = int(selection["primary_seed_frame"])
            if primary not in sampled_frames:
                raise ValueError(f"Primary frame {primary} for {target_id} is not in {sampled_frames}")
            obj.update(primary_seed_frame=primary,
                       primary_seed_reason=str(selection.get("reason", "")).strip(), alternate_seed_frames=[])
        return normalized
