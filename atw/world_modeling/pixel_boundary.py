"""Boundary-first observation pipeline with native requests and retained SAM3 evidence."""
import json
import cv2
import numpy as np

from atw.perception import pixel_segmentation_cv as cv
from atw.perception.pixel_segmentation_materials import board, card
from atw.world_modeling import pixel_protocol as p
from atw.world_modeling.pixel_requests import prompt
from atw.world_modeling.scene_assets import write


def boundary_cleanup(wall):
    count, labels, stats, _=cv2.connectedComponentsWithStats(wall.astype(np.uint8),8)
    if count<2:
        return wall.copy()
    area=stats[:,cv2.CC_STAT_AREA]
    largest=1+area[1:].argmax()
    # Relative to the observed frame material, not any dataset geometry.
    tiny_limit=max(1,int(area[largest]*.001))
    keep=area>tiny_limit
    keep[0]=False;keep[largest]=True
    return keep[labels]


def generate_candidates(runtime, first_frame, phrases, cache):
    import torch
    from atw.perception.sam3_tool import _generate_candidates
    with runtime.inference_lock, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        return _generate_candidates(runtime.image_processor, first_frame, phrases, prediction_cache=cache)


def render_candidate_sheet(first_frame, candidates, sheet):
    from atw.perception.sam3_tool import _render_candidate_sheet
    _render_candidate_sheet(first_frame, candidates, sheet)


def segment_boundary(video, info, first_frame, requests, runtime, output, questions):
    output.mkdir()
    frames = cv.sample_frames(info.frame_count)
    timeline = board([card(cv.read_frame(video, f, info.working_wh),
                     f'ORIGINAL FRAME {f}', (480, 300)) for f in frames], output/'timeline.jpg', len(frames))
    reference = output/'reference.jpg'
    cv2.imwrite(str(reference), cv.read_frame(video, 0, info.working_wh))
    context = dict(camera_fixed=True, frame_count=info.frame_count, questions=questions)
    decision = requests.call('boundary_presence', json.dumps(context), p.presence(info.frame_count),
                            [video, reference, timeline], system=prompt('boundary_presence'))
    if decision['status'] != 'present':
        raise ValueError('Pixel tracking requires a confirmed enclosing physical frame: '+decision['status'])
    plan = requests.call('boundary_plan', prompt('boundary_plan_instruction').format(decision=json.dumps(decision)),
                         p.PLAN, [first_frame], p.check_phrases, system=prompt('boundary_plan_system'))
    phrases = plan['short_noun_phrases']
    tried, cache = set(), {}
    for round_index in range(2):
        tried.update(s.strip().casefold() for s in phrases)
        candidates = generate_candidates(runtime, first_frame, phrases, cache)
        summaries = [{k: c[k] for k in ('candidate_id', 'phrase', 'score', 'box')} for c in candidates]
        write(output/f'candidates_{round_index}.json', summaries)
        np.savez_compressed(output/f'candidates_{round_index}.npz', **{c['candidate_id']: c['mask'] for c in candidates})
        media = [first_frame]
        if candidates:
            sheet = output/f'candidates_{round_index}.jpg'
            render_candidate_sheet(first_frame, candidates, sheet)
            media.append(sheet)
            selected = requests.call(f'boundary_selection_{round_index}',
                prompt('boundary_selection_instruction').format(description=plan['description'], candidates=json.dumps(summaries)),
                p.boundary_selection([c['candidate_id'] for c in candidates]), media,
                system=prompt('boundary_selection_system'))
            if selected['candidate_id'] is not None:
                wall = next(c['mask'] for c in candidates if c['candidate_id'] == selected['candidate_id']).astype(bool)
                np.savez_compressed(output/'before_cleanup.npz', wall=wall)
                clean = boundary_cleanup(wall)
                write(output/'cleanup.json', dict(method='remove_tiny_disconnected_components',
                    relative_area_limit=.001, removed_pixels=int((wall & ~clean).sum())))
                wall = clean
                np.savez_compressed(output/'boundary_mask.npz', wall=wall)
                return wall
        if round_index == 0:
            expansion = requests.call('boundary_expansion',
                prompt('boundary_expansion_instruction').format(tried=json.dumps(sorted(tried))),
                p.EXPANSION, media, lambda value: p.check_phrases(value, tried),
                system=prompt('boundary_plan_system'))
            phrases = expansion['short_noun_phrases']
            if not phrases:
                break
    raise ValueError('No acceptable observed outer-enclosure mask')
