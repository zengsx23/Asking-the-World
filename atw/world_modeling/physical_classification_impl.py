"""Semantic question selection followed by one joint mask-aware classification."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from atw.perception.object_material_classifier import ObjectMaterialClassifier
from atw.perception.sam3_agent import _media_data_url
from atw.world_modeling import sam3_protocol as contract
from atw.world_modeling.physical_classification import classification_schema, selection_schema

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).resolve().parents[1] / 'prompts'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_assets(asset_id, context):
    artifact = next((a for a in context.artifacts if a['artifact_id'] == asset_id), None)
    if artifact is None:
        raise ValueError(f'Unknown segmentation_asset_id: {asset_id}')
    root = context.run_dir.resolve()
    manifest_path = (root / artifact['path']).resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError('Segmentation artifact must belong to this run')
    manifest = json.loads(manifest_path.read_text())
    contract.validate(contract.SEGMENTATION, manifest, 'segmentation')
    if Path(manifest['video_path']).resolve() != Path(context.inputs['video_path']).resolve():
        raise ValueError('Segmentation video does not match the original input video')
    paths = {}
    for key in ('inventory', 'mask_index', 'mask_archive'):
        path = (manifest_path.parent / manifest[key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'segmentation.{key} must belong to this run')
        paths[key] = path
    inventory = json.loads(paths['inventory'].read_text())
    index = json.loads(paths['mask_index'].read_text())
    contract.validate_assets(inventory, index)
    return manifest, inventory, index, paths['mask_archive']


def _fit(image, width=640, height=360):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1]*scale)),
                                 max(1, round(image.shape[0]*scale))))
    canvas = np.full((height, width, 3), 245, np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


def render_evidence(video, inventory, index, archive_path, output):
    """One original-RGB context/crop pair per track; no descriptions or scores."""
    capture = cv2.VideoCapture(str(video))
    pairs = {(v['track_id'], v['frame_index']): (k, v) for k, v in index.items()}
    images, evidence = [], []
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            for obj in inventory['objects']:
                frame = obj['primary_frame'] if obj['primary_frame'] in obj['visible_frames'] else obj['visible_frames'][0]
                key, meta = pairs[obj['track_id'], frame]
                mask = np.unpackbits(archive[key])[:meta['num_bits']].reshape(meta['height'], meta['width']).astype(bool)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, raw = capture.read()
                if not ok or raw.shape[:2] != mask.shape or not mask.any():
                    raise ValueError(f"Invalid video/mask evidence for {obj['track_id']} at frame {frame}")
                ys, xs = np.where(mask)
                pad = max(12, round(max(xs.max()-xs.min(), ys.max()-ys.min())*.12))
                x0, x1 = max(0, int(xs.min())-pad), min(raw.shape[1], int(xs.max())+pad+1)
                y0, y1 = max(0, int(ys.min())-pad), min(raw.shape[0], int(ys.max())+pad+1)
                outlined = raw.copy()
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(outlined, contours, -1, (0, 0, 255), 3)
                pair = np.hstack([_fit(outlined), _fit(outlined[y0:y1, x0:x1])])
                image = Image.new('RGB', (1280, 400), 'white')
                image.paste(Image.fromarray(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB)), (0, 40))
                ImageDraw.Draw(image).text((12, 8),
                    f"{obj['track_id']} | frame {frame} | LEFT: scene context | RIGHT: local crop",
                    font=ImageFont.load_default(size=23), fill='black')
                path = output / f"{obj['track_id']}.png"
                image.save(path)
                images.append(path)
                evidence.append(dict(track_id=obj['track_id'], frame=frame,
                    crop_xyxy=[x0, y0, x1, y1], image=path.name))
    finally:
        capture.release()
    write(output / 'evidence.json', evidence)
    return images


def request(client, schema, *, operation, call_id, prompt_name, content, output, filename, key, id_key):
    def validate(value):
        value, _ = contract.normalize_response(schema, value)
        contract.validate(schema, value, operation)
        contract.unique_ids(value[key], id_key, operation)
    raw = client.request_json(operation=operation, call_id=call_id,
        system=(PROMPTS / prompt_name).read_text(), content=content, validate=validate)
    value, _ = contract.normalize_response(schema, raw)
    validate(value)
    write(output / filename, value)
    return value


def classify(arguments, context):
    manifest, inventory, index, archive = load_assets(arguments['segmentation_asset_id'], context)
    output = context.output_dir
    video = Path(context.inputs['video_path'])
    questions = [{k: q[k] for k in ('id', 'text', 'options')} for q in context.inputs['questions']]
    write(output / 'selection_input.json', dict(questions=questions))
    clients = {}
    try:
        if not inventory['objects']:
            write(output / 'question_selection.json', dict(selected=[]))
            write(output / 'classification.json', dict(objects=[]))
            write(output / 'evidence.json', [])
            write(output / 'execution.json', dict(skipped='empty_inventory'))
            data = dict(objects=[])
        else:
            images = render_evidence(video, inventory, index, archive, output)
            if questions:
                selector = clients['selection'] = ObjectMaterialClassifier(ROOT).client
                selector.trace_dir = output / 'selection_requests'
                schema = selection_schema([q['id'] for q in questions])
                write(output / 'selection_schema.json', schema)
                selection = request(selector, schema, operation='physical_question_selection', call_id='selection',
                    prompt_name='physical_question_selection_system.txt',
                    content=[dict(type='text', text=json.dumps(dict(questions=questions))+contract.prompt(schema))],
                    output=output, filename='question_selection.json', key='selected', id_key='question_id')
            else:
                selection = dict(selected=[])
                write(output / 'question_selection.json', selection)
            selected = {q['question_id'] for q in selection['selected']}
            kept = [q for q in questions if q['id'] in selected]
            objects = [dict(track_id=o['track_id']) for o in inventory['objects']]
            inp = dict(questions=kept, objects=objects)
            write(output / 'classification_input.json', inp)
            schema = classification_schema([o['track_id'] for o in objects])
            write(output / 'classification_schema.json', schema)
            content = [dict(type='text', text=json.dumps(inp)+contract.prompt(schema))]
            for path in [video, *images]:
                kind, url = _media_data_url(path)
                content.append(dict(type=kind, **{kind: dict(url=url)}))
            closing = dict(explicit_mechanical_property_questions=kept,
                note='These are the original questions selected for mechanical-property relevance. They are context, not physical labels or established facts. Judge visual evidence from the video as well.')
            write(output / 'mechanical_question_evidence.json', closing)
            content.append(dict(type='text', text=json.dumps(closing)+
                '\nNow classify every supplied track. Read the mechanical-property evidence before deciding. '+contract.prompt(schema)))
            classifier = clients['classification'] = ObjectMaterialClassifier(ROOT).client
            classifier.trace_dir = output / 'classification_requests'
            answer = request(classifier, schema, operation='physical_type_classification_joint', call_id='classification',
                prompt_name='physical_classification_system.txt', content=content, output=output,
                filename='classification.json', key='objects', id_key='track_id')
            by_track = {o['track_id']: o for o in answer['objects']}
            data = dict(objects=[dict(object_id=o['object_id'], **by_track[o['track_id']]) for o in inventory['objects']])
        write(output / 'physical_types.json', data)
    finally:
        write(output / 'vlm_usage.json', {stage: client.usage_report() for stage, client in clients.items()})
    artifacts = [dict(artifact_id=f'{context.call_id}_{kind}',
        path=str((output/name).relative_to(context.run_dir)), media_type='application/json', source_kind='derived')
        for kind, name in [('physical_types', 'physical_types.json'), ('question_selection', 'question_selection.json'),
                           ('classification_evidence', 'evidence.json'), ('vlm_usage', 'vlm_usage.json')]]
    return data, artifacts
