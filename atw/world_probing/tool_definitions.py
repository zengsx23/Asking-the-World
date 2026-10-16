"""Stage-two tools use the same flat canonical call/result envelopes as stage one."""
import json
from types import SimpleNamespace

from atw.world_modeling import sam3_protocol as s
from atw.world_modeling.scene_assets import artifact, asset_path
from atw.world_probing.evidence import Evidence, write
from atw.world_probing import identities, question_types, counting, existence

IDENTITY_DETAILS_NOTE = (
    'First observation is the first nonempty retained mask: zero-based frame and seconds from video start. '
    'It may miss earlier visibility and is not an emission time. Static emitter visibility does not date its liquid. '
    'Cards still show maximum-area masks, with time labels outside the original image. '
    'Pixel RGB is an observed 0..255 prototype; color groups do not imply temporal order.')


def current_question(arguments, context):
    if context.question is None or arguments['question_id'] != context.question_id:
        raise ValueError('The explicit current question must match question_id')
    return context.question


def classification(arguments, context):
    question = current_question(arguments, context)
    value = question_types.classify(question, context.output_dir/'classification', context.vlm)
    item = artifact(context, 'classification', context.output_dir/'classification/request/result.json')
    return value, [item]


def existence_classification(arguments, context):
    question = current_question(arguments, context)
    value = existence.classify(question, context.output_dir/'existence_classification', context.vlm)
    return value, [artifact(context, 'existence_classification',
                           context.output_dir/'existence_classification/request/result.json')]


def existence_answer(arguments, context):
    question = current_question(arguments, context)
    classification = existence.load(arguments, context, 'classification_asset_id')
    binding = existence.load(arguments, context, 'binding_asset_id')
    value = existence.answer(question, classification, binding, context, context.output_dir/'existence',
                             arguments['binding_asset_id'])
    return value, [artifact(context, 'existence', context.output_dir/'existence/result.json')]


def appearance(arguments, context):
    upstream = context.stage_one_result
    if upstream is None or upstream['data']['scene']['sources']['tracking_asset_id'] != arguments['tracking_asset_id']:
        raise ValueError('Identity preparation requires the explicitly selected tracking asset')
    catalog = Evidence(dict(video_path=context.inputs['video_path'], stage_one_result=upstream), context).prepare(context.output_dir/'cards')
    value = identities.ground(catalog, context.inputs['questions'], context, context.output_dir/'appearance')
    by_id = {o['object_id']:o for o in value['objects']}
    path = context.output_dir/'identities.json'
    write(path, dict(protocol_version=1, video_path=catalog['video_path'],
        identity_semantics=catalog['identity_semantics'], sources=catalog['sources'],
        objects=[dict(o, **{k:v for k,v in by_id[o['object_id']].items() if k!='object_id'}) for o in catalog['objects']]))
    items = [artifact(context,'identities',path)]
    for index, obj in enumerate(catalog['objects'],1):
        items.append(artifact(context,f'identity_image_{index:04d}',context.run_dir/obj['image'],'image/png'))
    return dict(identity_asset_id=items[0]['artifact_id'], object_count=len(value['objects'])), items


def load_identity(arguments, context):
    value = json.loads(asset_path(context, arguments['identity_asset_id']).read_text())
    if value['protocol_version'] != 1 or value['video_path'] != context.inputs['video_path']:
        raise ValueError('Identity asset video/protocol does not match this run')
    keys = identities.APPEARANCE_SCHEMA['properties']['objects']['items']['properties']
    appearance = dict(objects=[{key:obj[key] for key in keys} for obj in value['objects']])
    return value, appearance


def binding(arguments, context):
    question = current_question(arguments, context)
    catalog, appearance = load_identity(arguments, context)
    value = identities.bind(question, catalog, appearance, context, context.output_dir/'binding', arguments['identity_asset_id'])
    return value, [artifact(context,'binding',context.output_dir/'binding/result.json')]


def query_details(arguments, context):
    from PIL import Image, ImageDraw, ImageFont
    catalog, _ = load_identity(arguments, context)
    rows, items = [], []
    for index, obj in enumerate(catalog['objects'], 1):
        object_id = obj['object_id']
        with Image.open(context.run_dir / obj['image']) as original:
            card = Image.new('RGB', (original.width, original.height + 48), 'white')
            card.paste(original, (0, 48))
        label = f"{object_id} | First observed frame: {obj['first_observed_frame']} (zero-based)"
        ImageDraw.Draw(card).text((12, 12), label, fill='black', font=ImageFont.load_default(size=20))
        path = context.output_dir / f'identity_{index:04d}.png'
        card.save(path)
        item = artifact(context, f'identity_time_{index:04d}', path, 'image/png')
        items.append(item)
        row = dict(object_id=object_id, first_observed_frame=obj['first_observed_frame'],
                   first_observed_seconds=obj['first_observed_seconds'], image_artifact_id=item['artifact_id'])
        if catalog['identity_semantics'] == 'retained_instances_and_separate_liquid_masks':
            row['object_kind'] = obj['object_kind']
        for key in ('color_group_id','pixel_color_rgb'):
            if key in obj:
                row[key] = obj[key]
        rows.append(row)
    return dict(objects=rows,note=IDENTITY_DETAILS_NOTE), items


def count(arguments, context):
    question = current_question(arguments, context)
    catalog, appearance = load_identity(arguments, context)
    labels = Evidence(dict(video_path=context.inputs['video_path'],stage_one_result=context.stage_one_result),context).count_labels(catalog,appearance)
    write(context.output_dir/'count_labels.json', labels)
    value = counting.count(question, catalog, labels, context, context.output_dir/'count')
    item = artifact(context,'count',context.output_dir/'count/result.json')
    if value['execution_error']:
        raise RuntimeError(value['execution_error'])
    return dict(question_id=question['id'], status='answered' if value['answer'] is not None else 'unresolved',
                answer=value['answer'], evidence_ids=[item['artifact_id']], reason=value['reason']), [item]


def tool(name, description, input_schema, output_schema, function):
    definition = dict(protocol_version=1, name=name, description=description, stages=['stage_two'],
                      input_schema=input_schema, output_schema=output_schema)
    def validate_arguments(value):
        return s.validate(input_schema,value,'arguments')
    def validate_output(value):
        return s.validate(output_schema,value,'data')
    def execute(arguments, context):
        result = dict(protocol_version=1, call_id=context.call_id, tool_name=name,
                      status='error', data={}, artifacts=[], error=None)
        try:
            data, items = function(arguments, context)
            validate_output(data)
            result.update(status='ok', data=data, artifacts=items)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            result['error']=dict(code='execution_failed', message=f'{type(exc).__name__}: {exc}')
        return result
    return SimpleNamespace(DEFINITION=definition,validate_arguments=validate_arguments,
                           validate_output=validate_output,execute=execute)


DETAIL_SCHEMA = s.obj(object_id=s.TEXT,
    first_observed_frame=dict(type='integer',minimum=0),
    first_observed_seconds=dict(type='number',minimum=0), image_artifact_id=s.TEXT)
DETAIL_SCHEMA['properties'].update(object_kind=s.TEXT, color_group_id=s.TEXT,
    pixel_color_rgb=dict(type='array',items=dict(type='integer',minimum=0,maximum=255),minItems=3,maxItems=3))

TOOLS = [
    tool('classify_question_reasoning_mode', 'Classify the current question from literal text and options only.',
         s.obj(question_id=s.TEXT), s.obj(reasoning_mode=dict(type='string',enum=['property','count','explanatory','predictive','counterfactual','goal']),rationale=s.TEXT), classification),
    tool('classify_property_existence', 'Determine whether a property question asks about entity presence, and encode conditional answers from wording only; no scene judgment.',
         s.obj(question_id=s.TEXT), existence.CLASSIFICATION_SCHEMA, existence_classification),
    tool('ground_scene_identities', 'Describe retained identities using original video and original-frame mask cards; upstream assets are read-only.',
         s.obj(tracking_asset_id=s.TEXT), s.obj(identity_asset_id=s.TEXT,object_count=dict(type='integer',minimum=0)), appearance),
    tool('bind_question_identities', 'Bind the current question references to supplied identities with found or not_found_in_scene decisions.',
         s.obj(question_id=s.TEXT,identity_asset_id=s.TEXT), identities.BINDING_SCHEMA, binding),
    tool('query_identity_details',
         'Read first-observation times and labeled cards for all identities, plus pixel attributes when available. '
         'Use for temporal ambiguity, especially same-color liquids; ordinary appearance binding does not need this query. Query once.',
         s.obj(identity_asset_id=s.TEXT), s.obj(objects=s.array(DETAIL_SCHEMA),note=s.TEXT), query_details),
    tool('count_scene_identities', 'Count matching retained identities in the current question using binary visual decisions; does not simulate.',
         s.obj(question_id=s.TEXT,identity_asset_id=s.TEXT),
         s.obj(question_id=s.TEXT,status=dict(type='string',enum=['answered','unresolved']),
               answer=dict(type=['integer','array','null'],items=s.TEXT),evidence_ids=s.array(s.TEXT),reason=s.TEXT), count),
    tool('answer_scene_existence', 'Answer a confirmed existence question: matched identity references establish presence; otherwise independently inspect the complete original video.',
         s.obj(question_id=s.TEXT,classification_asset_id=s.TEXT,binding_asset_id=s.TEXT),
         existence.ANSWER_SCHEMA, existence_answer),
]
