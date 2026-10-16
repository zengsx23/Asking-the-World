"""Read explicit upstream assets and render observation cards without simulators."""
import base64
import json
import mimetypes
from pathlib import Path

import cv2
import numpy as np

from atw.contracts import validate_stage_one_scene
from atw.world_modeling import sam3_protocol


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def media(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime is None or mime.split('/')[0] not in ('image', 'video'):
        raise ValueError('Expected an original video or evidence image')
    kind = mime.split('/')[0] + '_url'
    return dict(type=kind, **{kind: dict(url=f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode())})


class Evidence:
    def __init__(self, inputs, context):
        self.context = context
        self.video = Path(inputs['video_path']).resolve()
        self.artifacts = {a['artifact_id']: a for a in inputs['stage_one_result']['artifacts']}
        self.scene = validate_stage_one_scene(inputs['stage_one_result']['data']['scene'])
        self.sources = self.scene['sources']

    def local(self, path):
        path = path.resolve()
        if not path.is_relative_to(self.context.run_dir.resolve()) or not path.is_file():
            raise ValueError('Referenced evidence must be an existing retained run file')
        return path

    def path(self, asset_id):
        if asset_id not in self.artifacts:
            raise ValueError(f'Unknown retained artifact: {asset_id}')
        return self.local(self.context.run_dir / self.artifacts[asset_id]['path'])

    def load(self, asset_id):
        return json.loads(self.path(asset_id).read_text())

    def count_labels(self, catalog, appearance):
        """Reuse retained semantic labels for counting, without exposing them to binding."""
        from atw.perception.labeled_mask_counter import infer_color
        if self.scene['route'] == '2D+PIXEL':
            tracking = self.load(self.sources['tracking_asset_id'])
            original = {o['object_id']:o for o in tracking['objects'] if o['object_kind'] != 'fluid'}
            groups = {o['group_id']:o for o in tracking['objects'] if o['object_kind'] == 'fluid'}
            binding = self.load(tracking['assets']['source_binding_asset_id'])
            for source in binding['sources']:
                original[source['source_id']] = dict(category='fluid', color=groups[source['group_id']]['name'])
        else:
            segmentation_path = self.path(self.sources['segmentation_asset_id'])
            segmentation = json.loads(segmentation_path.read_text())
            inventory = json.loads(self.local(segmentation_path.parent/segmentation['inventory']).read_text())
            original = {o['object_id']:o for o in inventory['objects']}
        descriptions = {o['object_id']:o for o in appearance['objects']}
        labels = []
        for card in catalog['objects']:
            identifier = card['object_id']
            obj, description = original[identifier], descriptions[identifier]
            category = obj.get('category') or obj.get('category_noun') or obj.get('name')
            try:
                color = infer_color(dict(color=obj.get('color'),
                    short_noun_phrases=[obj.get('name','')], description=obj.get('description','')))
            except ValueError:
                color = None
            labels.append(dict(object_id=identifier, category=category or description['category'],
                color=color or description['color'], category_source='stage_one' if category else 'appearance',
                color_source='stage_one' if color else 'appearance'))
        return dict(objects=labels)

    def prepare(self, output):
        output.mkdir(parents=True, exist_ok=False)
        tracking = self.load(self.sources['tracking_asset_id'])
        if Path(tracking['video_path']).resolve() != self.video or tracking['route'] != self.scene['route']:
            raise ValueError('Tracking video/route differs from the explicit scene handoff')
        if self.scene['route'] == '2D+PIXEL':
            objects = [obj for obj in tracking['objects'] if obj['object_kind'] != 'fluid']
            color_groups = {obj['group_id']: obj for obj in tracking['objects'] if obj['object_kind']=='fluid'}
            fps = tracking['fps']
            samples = {}
            # Every label is resolved through the declared manifest, never track-ID digits.
            with np.load(self.path(tracking['assets']['masks_asset_id']), allow_pickle=False) as masks:
                for frame in range(tracking['frame_count']):
                    labels = masks[tracking['mask_encoding']['archive_key'].format(frame_index=frame)]
                    for obj in objects:
                        mask = labels == obj['mask_label']
                        area = int(mask.sum())
                        if area:
                            previous = samples.get(obj['object_id'])
                            first = frame if previous is None else previous[3]
                            if previous is None or area > previous[0]:
                                samples[obj['object_id']] = (area, frame, mask, first)
            candidates = [(obj, samples[obj['object_id']][1], samples[obj['object_id']][2], samples[obj['object_id']][3]) for obj in objects]
            binding = self.load(tracking['assets']['source_binding_asset_id'])
            if (Path(binding['source_video_path']).resolve() != self.video
                    or binding['fps'] != fps
                    or binding['frame_scope'] != [0, tracking['frame_count'] - 1]):
                raise ValueError('Liquid identity video/time scope differs from tracking')
            # An inactive outlet remains a static identity, not an observed liquid.
            sources = [source for source in binding['sources'] if source['first_visible_frame'] is not None]
            liquid_samples = {}
            with np.load(self.path(tracking['assets']['source_masks_asset_id']), allow_pickle=False) as masks:
                for frame in range(tracking['frame_count']):
                    labels = masks[binding['mask_encoding']['archive_key'].format(frame_index=frame)]
                    for source in sources:
                        identifier = source['source_id']
                        mask = labels == source['mask_label']
                        area = int(mask.sum())
                        if area:
                            previous = liquid_samples.get(identifier)
                            first = frame if previous is None else previous[3]
                            if previous is None or area > previous[0]:
                                liquid_samples[identifier] = (area, frame, mask, first)
            for source in sources:
                identifier = source['source_id']
                if identifier not in liquid_samples:
                    raise ValueError(f'No observed liquid mask for identity {identifier}')
                _, frame, mask, first = liquid_samples[identifier]
                obj = dict(object_id=identifier, object_kind='fluid', physical_type='fluid',
                           source_id=identifier, mask_label=source['mask_label'],
                           mask_asset_id=tracking['assets']['source_masks_asset_id'])
                color = color_groups[source['group_id']]
                obj.update(color_group_id=source['group_id'], pixel_color_rgb=color['principal_rgb'])
                candidates.append((obj, frame, mask, first))
            semantics = 'retained_instances_and_separate_liquid_masks'
        else:
            seg_path = self.path(self.sources['segmentation_asset_id'])
            seg = json.loads(seg_path.read_text())
            sam3_protocol.validate(sam3_protocol.SEGMENTATION, seg, 'segmentation')
            if Path(seg['video_path']).resolve() != self.video:
                raise ValueError('Segmentation video differs from the original input')
            fps = seg['fps']
            inventory = json.loads(self.local(seg_path.parent / seg['inventory']).read_text())
            index = json.loads(self.local(seg_path.parent / seg['mask_index']).read_text())
            sam3_protocol.validate_assets(inventory, index)
            types = self.load(self.sources['physical_types_asset_id'])['objects']
            by_id = {o['object_id']: o for o in types}
            expected = {(o['object_id'], o['track_id']) for o in inventory['objects']}
            if expected != {(o['object_id'], o['track_id']) for o in types} or expected != {
                    (o['object_id'], o['track_id']) for o in tracking['objects']}:
                raise ValueError('Scene identity mappings disagree')
            pairs = {(v['track_id'], v['frame_index']): (k, v) for k, v in index.items()}
            candidates = []
            with np.load(self.local(seg_path.parent / seg['mask_archive']), allow_pickle=False) as masks:
                for obj in inventory['objects']:
                    selected, first = None, None
                    for frame in sorted(f for track,f in pairs if track == obj['track_id']):
                        key, meta = pairs[obj['track_id'], frame]
                        mask = np.unpackbits(masks[key])[:meta['num_bits']].reshape(meta['height'], meta['width']).astype(bool)
                        if mask.any():
                            if first is None:
                                first = frame
                            area = int(mask.sum())
                            if selected is None or area > selected[0]:
                                selected = (area, frame, mask)
                    if selected is None:
                        raise ValueError(f"No observed mask for identity {obj['object_id']}")
                    candidates.append((dict(obj, physical_type=by_id[obj['object_id']]['physical_type']), selected[1], selected[2], first))
            semantics = 'retained_segmented_instances'
        capture = cv2.VideoCapture(str(self.video))
        cards = []
        try:
            for ordinal, (obj, frame, mask, first) in enumerate(candidates, 1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, raw = capture.read()
                if not ok or not mask.any():
                    raise ValueError('Representative original frame/mask is unavailable')
                if mask.shape != raw.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                path = output / f'identity_{ordinal:04d}.png'
                render_card(raw, mask, obj['object_id'], path)
                card = dict(object_id=obj['object_id'],
                    object_kind=obj.get('object_kind', 'instance'),
                    physical_type=obj.get('physical_type'), image_frame_index=frame,
                    first_observed_frame=first, first_observed_seconds=first / fps,
                    image=str(path.relative_to(self.context.run_dir)))
                # E IDs are native source-mask identities, not invented tracking IDs.
                for key in ('track_id', 'source_id', 'mask_label', 'mask_asset_id',
                            'color_group_id', 'pixel_color_rgb'):
                    if key in obj:
                        card[key] = obj[key]
                cards.append(card)
        finally:
            capture.release()
        catalog = dict(video_path=str(self.video), route=self.scene['route'],
                       identity_semantics=semantics, sources=dict(self.sources), objects=cards)
        write(output / 'catalog.json', catalog)
        return catalog


def render_card(raw, mask, object_id, path):
    """Proportional original RGB panels; only the full context has a red contour."""
    from PIL import Image, ImageDraw
    ys, xs = np.where(mask)
    pad = max(12, round(max(xs.max()-xs.min(), ys.max()-ys.min()) * .3))
    bounds = (slice(max(0, ys.min()-pad),ys.max()+pad+1), slice(max(0, xs.min()-pad),xs.max()+pad+1))
    crop = raw[bounds]
    isolated = np.full_like(raw, 255)
    isolated[mask] = raw[mask]
    isolated = isolated[bounds]
    outlined = raw.copy()
    border = cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool) & ~mask
    outlined[border] = (0,0,255)
    canvas = Image.new('RGB', (1920, 420), 'white')
    for offset, array in [(0, outlined), (640, crop), (1280, isolated)]:
        panel = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
        scale = min(640/panel.width,360/panel.height)
        panel = panel.resize((max(1,round(panel.width*scale)),max(1,round(panel.height*scale))),Image.Resampling.BICUBIC)
        canvas.paste(panel, (offset+(640-panel.width)//2, 50+(360-panel.height)//2))
    draw = ImageDraw.Draw(canvas)
    for x, label in [(0, 'original context / red locator'), (640, 'original crop'), (1280, 'selected mask RGB / white outside mask')]:
        draw.text((x+12, 12), f'{object_id} | {label}', fill='black')
    canvas.save(path)
