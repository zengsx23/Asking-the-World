"""SAM3 Engineer implementation; model settings are inherited from SAM3."""
from pathlib import Path
import json
import shutil
import time
from copy import deepcopy
import numpy as np
from atw.perception import sam3_tool as sam
from atw.perception.sam3_agent import GeminiSam3Agent, INVENTORY_SYSTEM, PRIMARY_SELECTION_SYSTEM, SEED_SELECTION_SYSTEM, SEED_PROMPT_EXPANSION_SYSTEM, SEED_PROMPT_EXPANSION, NoVisiblePrimaryFrame, _media_data_url
from typing import Mapping, Any
from atw.perception.sam3_runtime import task_sam3_runtime_registry
from atw.world_modeling import sam3_protocol as contract
from atw.world_modeling.sam3_seeds import select_primary_masks
from atw.world_modeling.sam3_containment import contained_objects, resolve_seed_containment
from atw.world_modeling.sam3_rendering import write_named_overlay

ROOT = Path(__file__).resolve().parents[2]

INVENTORY = """Inventory every independent entity inside the experimental interaction workspace.
Include fixed obstacles, support blocks and dividers. Exclude the overall tabletop,
floor, background wall, unrelated furniture, shadows, reflections and annotations.
Inspect the complete video. Each inventory entry must refer to exactly one
independent physical instance, with one target_id and singular segmentation phrases.
List separate supports, blocks or wheels separately even when they look alike;
never use one entry for a collection such as "all anchors" or an entire mechanism.
Keep one continuous object as one instance across occluded visible portions.
Describe visible color/pattern, shape, and image position at a cited frame, plus
nearby landmarks or visible connections. For similar instances, explicitly contrast
left/right, front/behind, width, orientation or which neighbor each overlaps or
connects to. Relative size alone is insufficient. State only observable details;
do not invent hidden connections or uncertain colors.
Do not classify physical behavior, material properties, rigid shape categories,
or support/wall relationships. Ordinary visual descriptions are allowed.
The final image is a contact sheet with exact frame labels. Primary-frame
selection is performed separately."""


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def normalize_objects(rows):
    contract.validate(contract.INVENTORY['properties']['objects'], rows, 'inventory_initial $.objects')
    contract.unique_ids(rows, 'target_id', 'inventory_initial')
    result = []
    for row in rows:
        # Preserve the existing generic-category fallback and four-phrase cap.
        phrases = list(row['short_noun_phrases'])
        if row['category_noun'] not in phrases:
            phrases.append(row['category_noun'])
        result.append({**row, 'short_noun_phrases': phrases[:4]})
    return result


def primary_objects(objects, response, frames):
    contract.validate(contract.primary([o['target_id'] for o in objects], frames), response, 'primary_frame_selection')
    contract.unique_ids(response['objects'], 'target_id', 'primary_frame_selection')
    selections = {r['target_id']: r for r in response['objects']}
    return [{**o, 'primary_seed_frame': selections[o['target_id']]['primary_seed_frame'],
        'primary_seed_reason': selections[o['target_id']]['reason'], 'alternate_seed_frames': []} for o in objects]


class SegmentationAgent(GeminiSam3Agent):
    """Schema-validated scene segmentation and seed selection."""
    def request_json(self, *, operation, call_id, system, instruction, media_paths,
                     response_schema, object_id=None, validate=None):
        instruction += contract.prompt(response_schema)
        content = [{'type': 'text', 'text': instruction}]
        names = []
        for path in media_paths:
            path = Path(path).resolve()
            kind, url = _media_data_url(path)
            names.append(path.name)
            content.append({'type': kind, kind: {'url': url}})
        def check(response):
            normalized, _ = contract.normalize_response(response_schema, response)
            contract.validate(response_schema, normalized, operation)
            if validate is not None:
                validate(normalized)
        parsed = self.client.request_json(operation=operation, call_id=call_id,
            system=system, content=content, object_id=object_id, validate=check)
        normalized, removed = contract.normalize_response(response_schema, parsed)
        self.trace.append(dict(call_id=call_id, model=self.model, media=names,
            instruction=instruction, raw_response=parsed, response=normalized,
            removed_schema_fields=removed))
        return normalized

    def discover_primary_inventory(self, *, call_prefix, video_path, contact_sheet_path, sampled_frames):
        initial = self.request_json(operation='inventory_initial', call_id=f'inventory_initial_{call_prefix}',
            system=INVENTORY_SYSTEM, instruction=INVENTORY, response_schema=contract.INVENTORY,
            media_paths=[video_path, contact_sheet_path],
            validate=lambda result: normalize_objects(result.get('objects')))
        objects = normalize_objects(initial.get('objects'))
        selection = self.request_json(operation='primary_frame_selection',
            call_id=f'primary_frame_selection_{call_prefix}', system=PRIMARY_SELECTION_SYSTEM,
            response_schema=contract.primary([o['target_id'] for o in objects], sampled_frames),
            instruction=f'''For every inventory entity below, choose exactly one printed
Primary frame. Select only from {sampled_frames}. In the reason, locate that
specific instance in this frame and distinguish it from similar neighbors using
visible position, shape and relationships. Do not merely repeat its name or claim
that it is fully visible. Prior descriptions and relative-size labels are fallible.

INVENTORY: {json.dumps(objects, ensure_ascii=False)}''', media_paths=[contact_sheet_path],
            validate=lambda result: primary_objects(objects, result, sampled_frames))
        return dict(initial=initial, primary_selection=selection,
                    sampled_frame_indices=sampled_frames,
                    objects=primary_objects(objects, selection, sampled_frames))


    def judge_seed_frame_visibility(
        self, *, call_prefix: str, target: Mapping[str, Any], frame_index: int,
        image_path: Path, feedback: Mapping[str, Any],
        candidate_sheet_path: Path | None,
    ) -> dict[str, Any]:
        response = self.request_json(
            operation="seed_frame_visibility",
            object_id=str(target["target_id"]),
            call_id=f"seed_visibility_{call_prefix}_{target['target_id']}",
            system="Judge whether one requested entity is visible in an image. Return strict JSON only.",
            instruction=f"""The first seed-mask selection failed. Inspect raw frame {frame_index}
for this fixed entity: {json.dumps(dict(target), ensure_ascii=False)}
Previous result: {json.dumps(feedback.get('selection', feedback), ensure_ascii=False)}
Decide whether the entity itself is visible, not whether a candidate is acceptable.
Match the intended instance using location, shape, neighboring objects and visible
connections together; prior color, name and size descriptions may be wrong.
A color/name mismatch alone does not mean absence. If a visible instance best fits
these relationships, set entity_visible=true and return its grounded description
and 2-4 singular SAM3 phrases in identity, correcting mistaken attributes.
Distinguish similar neighbors explicitly. A missing candidate is not absence.
Use false with identity=null only when the intended instance cannot be located
in this frame; explain the evidence. Do not substitute an unrelated lookalike.
""",
            media_paths=[image_path] + ([candidate_sheet_path] if candidate_sheet_path else []),
            response_schema=contract.VISIBILITY,
        )
        return response

    def reselect_primary_frame(
        self, *, call_prefix: str, target: Mapping[str, Any],
        contact_sheet_path: Path, sampled_frames: list[int],
        failed_frame: int, feedback: Mapping[str, Any],
    ) -> dict[str, Any]:
        allowed = [frame for frame in sampled_frames if frame != failed_frame]
        if not allowed:
            raise NoVisiblePrimaryFrame("no alternative printed Primary frame available")
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
First re-identify the intended instance from its location, shape, neighbors and
visible connections. Prior colors, names and size labels are fallible, not hard
filters. Choose the frame showing the best-supported matching instance even when
its appearance contradicts the old description. Return that instance's grounded
description and 2-4 singular SAM3 phrases in identity; explain corrections in reason.
Keep the target_id and intended identity fixed, and distinguish similar neighbors.
Do not reject a real match merely for a color/name mismatch or choose an unrelated
lookalike. Return primary_seed_frame=null and identity=null only when no intended
instance can be located in any allowed frame; explain which evidence is missing.""",
            media_paths=[contact_sheet_path],
            response_schema=contract.reselection(allowed),
        )
        frame = response.get("primary_seed_frame")
        if frame is None:
            raise NoVisiblePrimaryFrame(f"no visible alternative Primary frame for {target['target_id']}")
        return response

    def expand_seed_prompts(
        self, *, call_prefix: str, target: Mapping[str, Any], frame_index: int,
        image_path: Path, tried_phrases: list[str], feedback: Mapping[str, Any],
        candidate_sheet_path: Path | None,
    ) -> dict[str, Any]:
        response = self.request_json(
            operation="seed_prompt_expansion",
            object_id=str(target["target_id"]),
            call_id=f"seed_expand_{call_prefix}_{target['target_id']}",
            system=SEED_PROMPT_EXPANSION_SYSTEM,
            instruction=SEED_PROMPT_EXPANSION.split("Return {")[0].rstrip() + "\nFixed target: " + json.dumps({
                key: target[key] for key in
                ("target_id", "category_noun", "discriminative_description")
            }) + f"\nPrimary frame: {frame_index}\nAlready tried: {json.dumps(tried_phrases)}"
                + f"\nPrevious result: {json.dumps(feedback)}",
            media_paths=[image_path] + ([candidate_sheet_path] if candidate_sheet_path else []),
            response_schema=contract.EXPANSION,
            validate=lambda response: contract.fresh_phrases(response, tried_phrases),
        )
        return response

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
in the raw frame and candidate masks. For similar candidates, explicitly compare
their positions, shapes, occlusion order and visible relationships. Match one
intended instance; do not select the same-looking neighbor based on color or
relative size alone. Explain the distinguishing evidence in reason.

Media order is: raw frame {frame_index}, candidates for frame {frame_index}.
In each candidate image, the outlined region preserves its original appearance,
the surrounding context is dimmed, and the red contour is only an annotation:
it does not indicate the object's color. Compare candidates with the raw frame
using appearance, shape, position, and instance identity.
Reject every candidate that belongs to another instance or
includes large background/support regions."""
        extra_media = []
        if target.get('seed_review_feedback'):
            instruction += '\nSelection review: ' + target['seed_review_feedback']
        if target.get("seed_containment_feedback"):
            instruction += "\n" + target["seed_containment_feedback"]
            extra_media = [Path(target["seed_containment_evidence"])]
        response = self.request_json(
            operation="seed_mask_selection",
            object_id=str(target["target_id"]),
            call_id=f"seed_select_{call_prefix}_{target['target_id']}",
            system=SEED_SELECTION_SYSTEM,
            instruction=instruction,
            media_paths=[image_path, candidate_sheet_path, *extra_media],
            response_schema=contract.selection(frame_index, candidate_ids),
        )
        decision = response['selections'][0]
        return {key: decision[key] for key in ('candidate_id', 'confidence', 'reason')} | {'agent_response': response}

def publish(*, output, video, objects, seeds, seed_manifest, tracks, debug, locked_inventory=False):
    """Export one inventory item per retained mask trajectory, using actual IDs."""
    objects = [obj for obj in objects if obj['target_id'] in seeds]
    if locked_inventory:
        dedup = dict(executed=False, reason='Inventory locked before propagation')
    else:
        objects, seeds, dedup = sam._filter_duplicate_seeds(
            objects=objects, seeds=seeds, seed_manifest=seed_manifest)
    kept, masks, omitted = [], {}, []
    for obj in objects:
        target = obj['target_id']
        visible = {f: m for f, m in tracks.get(target, {}).items() if np.any(m)}
        if not visible:
            if locked_inventory:
                raise ValueError(f'Locked inventory object has no propagated masks: {target}')
            omitted.append(dict(target_id=target, reason='no_nonempty_tracking_masks'))
            continue
        track_id = f'track_{len(kept)+1:03d}'
        kept.append(dict(object_id=target, name=target.replace('_', ' '), track_id=track_id,
            category_noun=obj['category_noun'], description=obj['discriminative_description'],
            primary_frame=obj['primary_seed_frame'], visible_frames=sorted(visible)))
        masks[track_id] = visible
    frames = {}
    for track, trajectory in masks.items():
        for f, mask in trajectory.items():
            frames.setdefault(f, {})[track] = mask
    index = sam._write_mask_archive(masks_by_frame=frames, output_path=output/'masks.npz')
    inventory = dict(objects=kept,
        object_to_track={o['object_id']:o['track_id'] for o in kept},
        track_to_object={o['track_id']:o['object_id'] for o in kept})
    contract.validate_assets(inventory, index)
    write(output/'mask_index.json', index)
    write(output/'objects.json', inventory)
    write(output/'finalization.json', dict(seed_deduplication=dedup, omitted_empty_tracks=omitted))
    if debug:
        write_named_overlay(video_path=video, masks_by_frame=frames, objects=kept,
                            output_path=output/'segmentation_named.mp4')
    return kept



def load_saved_seeds(path, video, frame_count, width, height):
    """Load an explicit seed-only result without rediscovery or VLM calls."""
    saved = json.loads(path.read_text())
    if saved.get('stage') != 'ready_for_propagation':
        raise ValueError('Seed inventory must be ready_for_propagation')
    if Path(saved['video_path']).resolve() != video:
        raise ValueError('Seed inventory video does not match the input video')
    if (saved['frame_count'], saved['width'], saved['height']) != (frame_count, width, height):
        raise ValueError('Seed inventory dimensions/frame count do not match the video')
    rows = saved['objects']
    contract.unique_ids(rows, 'object_id', 'seed inventory')
    contract.unique_ids(rows, 'mask_id', 'seed inventory')
    forward = {o['object_id']: o['mask_id'] for o in rows}
    if saved['object_to_mask'] != forward or saved['mask_to_object'] != {v:k for k,v in forward.items()}:
        raise ValueError('Seed inventory requires exact inverse object/mask mappings')
    objects, seeds, arrays = [], {}, {}
    with np.load(path.parent/saved['masks']) as archive:
        if set(archive.files) != set(forward.values()):
            raise ValueError('Seed archive keys must match inventory masks')
        for row in rows:
            frame = row['primary_frame']
            if type(frame) is not int or not 0 <= frame < frame_count:
                raise ValueError('Seed primary_frame must be an in-range integer')
            mask = archive[row['mask_id']].astype(bool)
            if mask.shape != (height, width) or not mask.any():
                raise ValueError('Seed mask must be nonempty at original video resolution')
            target = row['object_id']
            objects.append(dict(target_id=target, category_noun=row['category_noun'],
                discriminative_description=row['description'], primary_seed_frame=frame))
            seeds[target] = dict(frame_index=frame, mask=mask)
            arrays[f'{target}__{frame:04d}'] = mask
    manifest = json.loads((path.parent/'seed_manifest.json').read_text())
    original = json.loads((path.parent/'inventory_before_dedup.json').read_text())
    return dict(initial=original, objects=objects), seeds, manifest, arrays

class Sam3Engineer:
    from atw.world_modeling.sam3_segmentation import DEFINITION as definition

    def __init__(self, *, checkpoint, debug=False, seed_inventory=None):
        self.checkpoint, self.debug = Path(checkpoint), debug
        self.seed_inventory = Path(seed_inventory) if seed_inventory is not None else None

    def execute(self, arguments, context):
        result = dict(protocol_version=1, call_id=context.call_id, tool_name=self.definition['name'],
                      status='error', data={}, artifacts=[], error=None)
        try:
            contract.validate(self.definition['input_schema'], arguments, 'tool.arguments')
        except ValueError as exc:
            result['error'] = dict(code='invalid_arguments', message=str(exc))
            return result
        output = context.output_dir
        output.mkdir(parents=True, exist_ok=True)
        agent = None
        try:
            video = Path(context.inputs['video_path']).resolve()
            if context.stage != 'stage_one':
                raise ValueError('SAM3 Engineer is available only in stage one')
            count, width, height, fps = sam._video_metadata(video)
            timings = {}
            started = time.perf_counter()
            if self.seed_inventory is not None:
                inventory, seeds, manifest, arrays = load_saved_seeds(
                    self.seed_inventory, video, count, width, height)
                write(output/'seed_source.json', dict(
                    inventory_path=str(self.seed_inventory.resolve()), seed_selection_reused=True))
            else:
                agent = SegmentationAgent(ROOT)
                agent.client.trace_dir = output/'requests'
                sampled = sam._sampled_frame_indices(count, maximum=8)
                sheet = output/'indexed_frame_contact_sheet.jpg'
                sam._write_indexed_contact_sheet(video, sampled, sheet)
                inventory = agent.discover_primary_inventory(call_prefix=context.call_id, video_path=video,
                    contact_sheet_path=sheet, sampled_frames=sampled)
                write(output/'inventory_before_dedup.json',inventory)
            runtime, _ = task_sam3_runtime_registry().acquire(self.checkpoint)
            if self.seed_inventory is None:
                seeds, manifest, arrays = select_primary_masks(agent=agent, runtime=runtime, video_path=video,
                    request_dir=output, call_prefix=context.call_id, objects=inventory['objects'],timings=timings,
                    contact_sheet_path=sheet, sampled_frames=sampled)
            else:
                write(output/'inventory_before_dedup.json',inventory)
            timings['seed_preparation_seconds'] = time.perf_counter() - started
            np.savez_compressed(output/'selected_primary_masks.npz',**arrays)
            write(output/'seed_manifest_initial.json',manifest)
            seeded_objects = [obj for obj in inventory['objects'] if obj['target_id'] in seeds]
            def reselect_large(target, conflict, current_seeds):
                nonlocal agent
                if agent is None:
                    agent = SegmentationAgent(ROOT)
                    agent.client.trace_dir = output/'requests'
                repair_dir = output/'containment_reselection'/target['target_id']
                repair_dir.mkdir(parents=True, exist_ok=False)
                sampled = sam._sampled_frame_indices(count, maximum=8)
                sheet = repair_dir/'indexed_frames.jpg'
                sam._write_indexed_contact_sheet(video, sampled, sheet)
                original = next(r['target'] for r in manifest
                                if r['target']['target_id'] == target['target_id'])
                retry_target = {**deepcopy(original), **deepcopy(target)}
                frame = conflict['frame_index']
                raw = repair_dir/'conflict_frame.jpg'
                sam._extract_frame(video, frame, raw)
                from atw.world_modeling.sam3_rendering import render_candidate_sheet
                evidence = repair_dir/'containment_evidence.jpg'
                evidence_rows = [
                    dict(candidate_id=key, mask=current_seeds[key]['mask'])
                    for key in [target['target_id'], *[c['target_id'] for c in conflict['contained']]]
                ]
                render_candidate_sheet(raw, evidence_rows, evidence)
                retry_target['seed_containment_feedback'] = (
                    'The previous mask for this target is rejected: it contains multiple '
                    'other inventory objects on the same frame. Select only the requested '
                    'instance, excluding those other objects. Keep its target_id unchanged. '
                    'The final image shows the rejected mask and the separate contained objects '
                    f"at frame {frame}: " + json.dumps(conflict['contained']))
                retry_target['seed_containment_evidence'] = str(evidence)
                repair_timings = {}
                new_seeds, records, arrays = select_primary_masks(
                    agent=agent, runtime=runtime, video_path=video, request_dir=repair_dir,
                    call_prefix=context.call_id+'_containment_'+target['target_id'],
                    objects=[retry_target], timings=repair_timings,
                    contact_sheet_path=sheet, sampled_frames=sampled,
                    candidate_filter=lambda key, f, mask: contained_objects(key, f, mask, current_seeds))
                write(repair_dir/'timings.json', repair_timings)
                np.savez_compressed(repair_dir/'selected_primary_masks.npz', **arrays)
                write(repair_dir/'seed_manifest.json', records)
                if target['target_id'] not in new_seeds:
                    return None
                return retry_target, new_seeds[target['target_id']], records[0]

            active, seeds, manifest, containment = resolve_seed_containment(
                objects=seeded_objects, seeds=seeds, manifest=manifest, reselect=reselect_large)
            write(output/'seed_containment.json', containment)
            active, seeds, dedup = sam._filter_duplicate_seeds(
                objects=active, seeds=seeds, seed_manifest=manifest)
            write(output/'seed_deduplication.json', dedup)
            manifest = [r for r in manifest if r['target']['target_id'] in seeds]
            from atw.world_modeling.sam3_seed_review import review, repair
            decision = review(objects=active, seeds=seeds, video_path=video,
                output=output/'seed_visual_review', vlm=context.vlm)
            if decision['resegment']:
                if agent is None:
                    agent = SegmentationAgent(ROOT)
                    agent.client.trace_dir = output/'requests'
                seeds, manifest = repair(decision=decision, objects=active, seeds=seeds,
                    manifest=manifest, video_path=video, output=output/'seed_visual_repair',
                    call_prefix=context.call_id+'_visual_repair', runtime=runtime, agent=agent)
            # Seed review is complete; geometric temporal deduplication follows propagation.
            active = [dict(o, primary_seed_frame=seeds[o['target_id']]['frame_index']) for o in active]
            write(output/'inventory_before_temporal_dedup.json', dict(objects=active))
            write(output/'locked_seed_inventory.json', dict(objects=active))
            write(output/'seed_manifest.json', manifest)
            np.savez_compressed(output/'checked_primary_masks.npz', **{
                f"{key}__{seed['frame_index']:04d}": seed['mask'] for key, seed in seeds.items()})
            active_seeds = seeds
            tracker_configuration, tracks = {}, {}
            tracking = dict(executed=False, reason='no_selected_seeds')
            started = time.perf_counter()
            if active:
                tracker_configuration = sam._configure_tracker_non_overlap(runtime.tracker)
                frames = sam._extract_tracker_frames(video, output/'tracker_frames')
                with runtime.inference_lock:
                    from atw.perception.sam3_temporal_dedup import track_with_deduplication
                    active, seeds, tracks, tracking, temporal_dedup = track_with_deduplication(
                        tracker=runtime.tracker, tracker_resource=frames,
                        objects=active, seeds=active_seeds, frame_count=count)
                write(output/'temporal_deduplication.json', temporal_dedup)
                manifest = [r for r in manifest if r['target']['target_id'] in seeds]
                write(output/'seed_manifest.json', manifest)
                write(output/'locked_seed_inventory.json', dict(objects=active))
                np.savez_compressed(output/'checked_primary_masks.npz', **{
                    f"{key}__{seed['frame_index']:04d}": seed['mask'] for key, seed in seeds.items()})
            timings['propagation_seconds'] = time.perf_counter() - started
            write(output/'tracking_manifest.json',dict(configuration=tracker_configuration,tracking=tracking))
            kept = publish(output=output,video=video,objects=active,seeds=seeds,
                seed_manifest=manifest,tracks=tracks,debug=self.debug,locked_inventory=True)
            segmentation = dict(video_path=str(video),frame_count=count,width=width,height=height,
                fps=fps,mask_archive='masks.npz',mask_index='mask_index.json',inventory='objects.json',
                coordinate_frame='original video pixels; x right, y down; zero-based frames',
                mask_semantics='SAM3-predicted visible instance masks; absent frames are not synthesized')
            contract.validate(contract.SEGMENTATION, segmentation, 'segmentation.json')
            write(output/'segmentation.json', segmentation)
            files=[('initial_inventory','inventory_before_dedup.json','application/json'),
                   ('segmentation','segmentation.json','application/json'),('inventory','objects.json','application/json'),
                   ('masks','masks.npz','application/x-npz'),('mask_index','mask_index.json','application/json')]
            files.extend([('seed_visual_review','seed_visual_review/decision.json','application/json'),
                          ('locked_seed_inventory','locked_seed_inventory.json','application/json')])
            if active:
                files.append(('temporal_deduplication','temporal_deduplication.json','application/json'))
            if self.debug:files.append(('debug_video','segmentation_named.mp4','video/mp4'))
            data = dict(object_count=len(kept),
                segmentation_asset_id=f'{context.call_id}_segmentation',inventory_asset_id=f'{context.call_id}_inventory')
            contract.validate(self.definition['output_schema'], data, 'tool.data')
            result['artifacts']=[dict(artifact_id=f'{context.call_id}_{kind}',path=str((output/name).relative_to(context.run_dir)),
                media_type=mime,source_kind='derived') for kind,name,mime in files]
            write(output/'timings.json',timings)
            result.update(status='ok', data=data)
        except Exception as exc:
            result['error']=dict(code='execution_failed',message=f'{type(exc).__name__}: {exc}')
        finally:
            shutil.rmtree(output/'tracker_frames',ignore_errors=True)
            if agent is not None:
                write(output/'agent_trace.json',agent.trace)
                write(output/'vlm_usage.json',agent.usage_report())
        return result
