"""Schemas shared by pixel prompts, validation and retained measurements."""
from atw.world_modeling.sam3_protocol import TEXT, array, obj

COORD = dict(type='number', minimum=0, maximum=1000)
BOX = obj(**{key: COORD for key in ('top', 'left', 'bottom', 'right')})
BOX['description'] = 'Full original frame coordinates normalized to 0..1000; x right, y down.'
RECEIVING = dict(type='boolean', description='True only for an observed stationary receiving structure that collects fluid; false for sources, obstacles and the enclosing frame. This is a visual role, independent of its name.')
DISCOVERY = obj(objects=array(obj(name=TEXT, description=TEXT, box=BOX, is_receiving_structure=RECEIVING)))
PLAN = obj(description=TEXT, short_noun_phrases=array(TEXT, minItems=2, maxItems=4, uniqueItems=True))
EXPANSION = obj(short_noun_phrases=array(TEXT, maxItems=4, uniqueItems=True), reason=TEXT)


def presence(frame_count):
    return obj(status=dict(type='string', enum=['present', 'absent', 'uncertain']), reason=TEXT,
               evidence=array(obj(frame=dict(type='integer', minimum=0, maximum=frame_count-1), observation=TEXT)))


def selection(ids):
    return obj(objects=array(obj(name=TEXT, candidate_ids=array(
        dict(type='string', enum=list(ids)), minItems=1, uniqueItems=True))))


def static_selection(ids):
    schema = selection(ids)
    item = schema['properties']['objects']['items']
    item['properties']['is_receiving_structure'] = RECEIVING
    item['required'].append('is_receiving_structure')
    return schema


def color_names(ids):
    return obj(groups=array(obj(group_id=dict(type='string', enum=list(ids)), name=TEXT)))


def check_color_names(value, ids):
    returned = [row['group_id'] for row in value['groups']]
    if len(returned) != len(ids) or set(returned) != set(ids):
        raise ValueError('Name every supplied color group exactly once')


def boundary_selection(ids):
    return obj(candidate_id=dict(type=['string', 'null'], enum=[*ids, None]),
               confidence=dict(type='number', minimum=0, maximum=1), reason=TEXT)


def check_boxes(value):
    for row in value['objects']:
        box = row['box']
        if box['top'] >= box['bottom'] or box['left'] >= box['right']:
            raise ValueError('Each box must have positive width and height')


def check_selection(value):
    used = [cid for row in value['objects'] for cid in row['candidate_ids']]
    if len(used) != len(set(used)):
        raise ValueError('Use each selected candidate at most once')


def check_phrases(value, tried=()):
    phrases = [s.strip().casefold() for s in value['short_noun_phrases']]
    if len(set(phrases)) != len(phrases) or set(phrases) & set(tried):
        raise ValueError('Boundary prompts must be distinct and new')
