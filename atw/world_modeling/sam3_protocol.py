"""Canonical SAM3 Engineer schemas for prompts, responses and saved assets."""
import json
import math
from jsonschema import Draft202012Validator, validators


def obj(**properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


TEXT = dict(type='string', minLength=1, pattern=r'\S')
ID = dict(type='string', pattern=r'^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$')
FRAME = dict(type='integer', minimum=0)


def array(items, **limits):
    return dict(type='array', items=items, **limits)


IDENTITY = obj(target_id=ID, category_noun=TEXT, discriminative_description=TEXT,
    short_noun_phrases=array(TEXT, minItems=2, maxItems=4), discovery_evidence=TEXT)
IDENTITY['properties'] = {k: {**v, 'description': d} for (k, v), d in zip(
    IDENTITY['properties'].items(), [
        'Stable visual instance ID; not a physical classification.',
        'Ordinary visual category noun.', 'Visible color/pattern, shape, position at a cited frame, and relations that uniquely distinguish this single instance; contrast similar neighbors.',
        'Simple singular phrases for one instance, never a plural collection.', 'One textual account of where the entity is observed.'])}
INVENTORY = obj(objects=array(IDENTITY, minItems=1), excluded_regions=array(TEXT), notes=dict(type='string'))
IDENTITY_UPDATE = obj(discriminative_description=IDENTITY['properties']['discriminative_description'],
    short_noun_phrases=IDENTITY['properties']['short_noun_phrases'])
VISIBILITY = obj(entity_visible=dict(type='boolean'), reason=TEXT, identity=dict(anyOf=[IDENTITY_UPDATE, dict(type='null')]))
VISIBILITY['allOf'] = [dict(
    **{'if': dict(properties=dict(entity_visible=dict(const=True))),
       'then': dict(properties=dict(identity=IDENTITY_UPDATE)),
       'else': dict(properties=dict(identity=dict(type='null')))})]
EXPANSION = obj(short_noun_phrases=array(TEXT, minItems=2, maxItems=4, uniqueItems=True), reason=TEXT)


def primary(ids, frames):
    return obj(objects=array(obj(target_id={**ID, 'enum': list(ids)},
        primary_seed_frame={**FRAME, 'enum': list(frames)}, reason=TEXT), minItems=len(ids), maxItems=len(ids)))


def selection(frame, candidates):
    return obj(selections=array(obj(frame_index={**FRAME, 'const': frame},
        candidate_id=dict(type=['string', 'null'], enum=[*sorted(candidates), None],
            description='Candidate label, or null when no candidate is acceptable.'),
        confidence=dict(type='number', minimum=0, maximum=1), reason=TEXT), minItems=1, maxItems=1),
        entity_visible=dict(type='boolean'), notes=dict(type='string'))


def reselection(frames):
    schema = obj(primary_seed_frame=dict(type=['integer', 'null'], enum=[*frames, None],
        description='Zero-based alternative sampled frame for the best-supported intended instance; null only when none can be located.'),
        reason=TEXT, identity=dict(anyOf=[IDENTITY_UPDATE, dict(type='null')]))
    schema['allOf'] = [{'if': dict(properties=dict(primary_seed_frame=dict(type='integer'))),
        'then': dict(properties=dict(identity=IDENTITY_UPDATE)),
        'else': dict(properties=dict(identity=dict(type='null')))}]
    return schema


# JSON boundary types are strict: do not accept Python bool as integer, 12.0 as
# an integer, or non-JSON NaN/Infinity as a number. Do not coerce model output.
checker = Draft202012Validator.TYPE_CHECKER.redefine_many({
    'integer': lambda _, v: type(v) is int,
    'number': lambda _, v: type(v) in (int, float) and math.isfinite(v),
})
Validator = validators.extend(Draft202012Validator, type_checker=checker)


def validate(schema, value, label='response'):
    error = next(Validator(schema).iter_errors(value), None)
    if error is not None:
        path = '$' + ''.join(f'[{x}]' if isinstance(x, int) else f'.{x}' for x in error.absolute_path)
        raise ValueError(f'{label} {path}: {error.message} (actual type: {type(error.instance).__name__})')
    return value


def normalize_response(schema, value):
    """Remove only observed schema echoes when all required answer fields exist."""
    if not isinstance(value, dict) or not set(schema.get('required', [])) <= value.keys():
        return value, []
    echoed = {'type', 'properties', 'required', 'additionalProperties'}
    removed = sorted((value.keys() - schema['properties'].keys()) & echoed)
    return {key: item for key, item in value.items() if key not in removed}, removed


def prompt(schema):
    return '\nOutput contract (JSON Schema; all listed fields required; no other fields):\n' + json.dumps(schema, ensure_ascii=False, separators=(',', ':'))


def unique_ids(rows, key, label):
    seen = set()
    for i, row in enumerate(rows):
        if row[key] in seen:
            raise ValueError(f'{label} $.objects[{i}].{key}: duplicate ID {row[key]!r}')
        seen.add(row[key])


def fresh_phrases(response, tried):
    seen = {p.strip().casefold() for p in tried}
    for i, phrase in enumerate(response['short_noun_phrases']):
        normalized = phrase.strip().casefold()
        if normalized in seen:
            raise ValueError(f'seed_prompt_expansion $.short_noun_phrases[{i}]: phrase is repeated or already tried')
        seen.add(normalized)


FINAL_OBJECT = obj(object_id=ID, name=TEXT, track_id=TEXT, category_noun=TEXT,
    description=TEXT, primary_frame=FRAME, visible_frames=array(FRAME, minItems=1, uniqueItems=True))
STRING_MAP = dict(type='object', additionalProperties=TEXT)
FINAL_INVENTORY = obj(objects=array(FINAL_OBJECT, minItems=0), object_to_track=STRING_MAP, track_to_object=STRING_MAP)
MASK_RECORD = obj(track_id=TEXT, frame_index=FRAME, height=dict(type='integer', minimum=1),
    width=dict(type='integer', minimum=1), num_bits=dict(type='integer', minimum=1))
MASK_INDEX = dict(type='object', minProperties=0, additionalProperties=MASK_RECORD)
SEGMENTATION = obj(video_path=TEXT, frame_count=dict(type='integer', minimum=1),
    width=dict(type='integer', minimum=1), height=dict(type='integer', minimum=1),
    fps=dict(type='number', exclusiveMinimum=0), mask_archive=TEXT, mask_index=TEXT,
    inventory=TEXT, coordinate_frame=TEXT, mask_semantics=TEXT)
TOOL_INPUT = obj()
TOOL_OUTPUT = obj(object_count=dict(type='integer', minimum=0), segmentation_asset_id=TEXT, inventory_asset_id=TEXT)


def validate_assets(inventory, index):
    validate(FINAL_INVENTORY, inventory, 'objects.json')
    validate(MASK_INDEX, index, 'mask_index.json')
    rows = inventory['objects']
    unique_ids(rows, 'object_id', 'objects.json')
    unique_ids(rows, 'track_id', 'objects.json')
    forward = {r['object_id']: r['track_id'] for r in rows}
    inverse = {v: k for k, v in forward.items()}
    if inventory['object_to_track'] != forward or inventory['track_to_object'] != inverse:
        raise ValueError('objects.json $: object_to_track and track_to_object must be exact inverse mappings of objects')
    pairs = set()
    for key, record in index.items():
        pair = record['track_id'], record['frame_index']
        if pair in pairs:
            raise ValueError(f'mask_index.json $.{key}: duplicate track/frame pair {pair}')
        pairs.add(pair)
        if record['num_bits'] != record['height'] * record['width']:
            raise ValueError(f'mask_index.json $.{key}.num_bits: must equal height * width')
    expected = {(r['track_id'], f) for r in rows for f in r['visible_frames']}
    if pairs != expected:
        raise ValueError('mask_index.json $: track/frame pairs must match inventory visible_frames exactly')
