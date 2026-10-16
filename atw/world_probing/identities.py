"""Scene appearance and isolated, two-state question identity binding."""
import json
from pathlib import Path

from atw.world_modeling import sam3_protocol as schema
from atw.openrouter import OpenRouterClient
from atw.world_probing.evidence import media, write
from atw.world_probing.collection_input import ToolFeedback

APPEARANCE_SCHEMA = schema.obj(objects=schema.array(schema.obj(
    object_id=schema.TEXT, category=schema.TEXT, color=schema.TEXT, shape=schema.TEXT,
    apparent_material=schema.TEXT, visual_description=schema.TEXT,
    visual_evidence=schema.TEXT, uncertainty=schema.TEXT,
    wheel_structure=dict(type='string', enum=['solid_disk', 'hollow_spoked', 'not_wheel', 'unknown']))))
BINDING_SCHEMA = schema.obj(identity_evidence=schema.array(schema.obj(
    reference=schema.TEXT, status=dict(type='string', enum=['found', 'not_found_in_scene']),
    object_ids=schema.array(schema.TEXT, uniqueItems=True), evidence=schema.TEXT)))
REFERENCE_SCHEMA = schema.obj(references=schema.array(schema.obj(
    reference=schema.TEXT, scope=dict(type='string',enum=['single','group']))))
BINDING_OUTPUT = ('Return only {"identity_evidence":[{"reference":"supplied reference",'
    '"status":"found","object_ids":["matching ID"],"evidence":"brief visual reason"}]}. '
    'Include every supplied reference once. Use found with matching IDs, or not_found_in_scene with []. '
    'Single references have one ID; groups contain all matches. No extra fields.')


def reference_key(text):
    key = ' '.join(text.casefold().split())
    return key[4:] if key.startswith('the ') else key


def request(config, output, prompt, payload, definition, validate, paths=(), *, media_labels=None,
            include_schema_in_prompt=True, normalize_response=None):
    output.mkdir(parents=True, exist_ok=False)
    saved=dict(system=prompt, context=payload, media=[str(p) for p in paths])
    if include_schema_in_prompt:
        saved['schema']=definition
        instructions=('\nReturn only the completed JSON data object matching the schema below. '
            'Populate its fields with your observations and decisions. Do not return or repeat the JSON Schema itself.\n'
            +schema.prompt(definition))
    else:
        write(output/'validation_schema.json',definition)
        instructions='\nReturn only the JSON answer specified by output_format.'
    write(output / 'input.json', saved)
    def checked(value):
        schema.validate(definition, value, 'response')
        return validate(value)
    options = dict(normalize_response=normalize_response) if normalize_response is not None else {}
    return OpenRouterClient(config).complete([
        dict(role='system', content=prompt + instructions),
        dict(role='user', content=[dict(type='text', text=json.dumps(payload)),
            *labeled_media(paths,payload.get('objects',[]),media_labels)]),
    ], output / 'request', validate=checked, **options)


def labeled_media(paths, objects, labels=None):
    content=[]
    for index,path in enumerate(paths):
        label=(labels[index] if labels is not None else 'Complete original video' if Path(path).suffix=='.mp4' else
               'Identity card for object_id: '+objects[index-1]['object_id'] if objects else Path(path).name)
        content.extend([dict(type='text',text=label),media(path)])
    return content


def replace_card_images(messages):
    """Annotated copies replace clean copies of the same cards, retaining video."""
    for message in messages:
        if isinstance(message.get('content'), list):
            message['content'] = [part for part in message['content'] if part.get('type') != 'image_url']


def ground(catalog, questions, context, output):
    if len(catalog['objects']) > 4:
        objects = []
        for start in range(0, len(catalog['objects']), 4):
            batch = dict(catalog, objects=catalog['objects'][start:start+4])
            result = ground(batch, questions, context, output/f'batch_{start//4+1:04d}')
            objects.extend(result['objects'])
        result = dict(objects=objects)
        write(output/'result.json', result)
        return result
    ids = [o['object_id'] for o in catalog['objects']]
    if not ids:
        write(output / 'result.json', dict(objects=[]))
        return dict(objects=[])
    def validate(value):
        returned = [o['object_id'] for o in value['objects']]
        if len(returned) != len(ids) or set(returned) != set(ids):
            raise ValueError('Appearance must cover every supplied identity exactly once')
        return value
    prompt = (Path(__file__).parents[1] / 'prompts/stage_two_appearance.txt').read_text()
    observation_keys = ('object_id','object_kind','color_group_id','pixel_color_rgb')
    payload = dict(objects=[{k:o[k] for k in observation_keys if k in o} for o in catalog['objects']],
                   identity_semantics=catalog['identity_semantics'],
                   questions_for_visual_attention=[{k:q[k] for k in ('id', 'text', 'options')} for q in questions])
    paths = [Path(catalog['video_path']), *[context.run_dir/o['image'] for o in catalog['objects']]]
    result = request(context.vlm, output, prompt, payload, APPEARANCE_SCHEMA, validate, paths)
    write(output / 'result.json', result)
    return result


def bind(question, catalog, appearance, context, output, identity_asset_id):
    output.mkdir(parents=True, exist_ok=False)
    def check_references(value):
        keys = [reference_key(r['reference']) for r in value['references']]
        if len(keys) != len(set(keys)):
            raise ValueError('Extract each distinct concrete reference once')
        return value
    reference_prompt = (Path(__file__).parents[1]/'prompts/stage_two_references.txt').read_text()
    references = request(context.vlm, output/'references', reference_prompt,
        dict(question={k:question[k] for k in ('id','text','options')}), REFERENCE_SCHEMA,
        check_references)['references']
    scopes = {reference_key(r['reference']):r['scope'] for r in references}
    ids = {o['object_id'] for o in catalog['objects']}
    def validate(value):
        seen = set()
        for row in value['identity_evidence']:
            reference = reference_key(row['reference'])
            if not reference or reference in seen:
                raise ValueError('References must be nonempty and unique')
            seen.add(reference)
            if not set(row['object_ids']) <= ids:
                raise ValueError('Binding references an unknown object ID')
            if (row['status'] == 'found') != bool(row['object_ids']):
                raise ValueError('found requires IDs; not_found_in_scene requires an empty list')
            if scopes.get(reference)=='single' and len(row['object_ids'])>1:
                raise ValueError('A single reference needs one identity, not alternative candidates')
        if seen != set(scopes):
            raise ValueError('Cover every extracted reference exactly once')
        return value
    prompt = (Path(__file__).parents[1] / 'prompts/stage_two_binding.txt').read_text()
    payload = dict(question={k:question[k] for k in ('id', 'text', 'options')},
                   references=references, objects=appearance['objects'], identity_asset_id=identity_asset_id)
    paths = [Path(catalog['video_path']), *[context.run_dir/o['image'] for o in catalog['objects']]]
    definitions = [d for d in context.tools.definitions('stage_two') if d['name']=='query_identity_details']
    variants=[]
    for row in references:
        object_ids=schema.array(schema.TEXT,uniqueItems=True)
        if row['scope']=='single':
            object_ids['maxItems']=1
        variants.append(schema.obj(reference=dict(type='string',const=row['reference']),
            evidence=schema.TEXT,status=dict(type='string',enum=['found','not_found_in_scene']), object_ids=object_ids))
    definition=schema.obj(identity_evidence=dict(type='array',items=dict(anyOf=variants) if variants else {},
        minItems=len(references),maxItems=len(references)))
    write(output/'input.json', dict(system=prompt, context=payload, schema=definition,
                                    tools=definitions, media=[str(p) for p in paths]))
    messages = [dict(role='system', content=prompt+'\n'+BINDING_OUTPUT),
                dict(role='user', content=[*labeled_media(paths,catalog['objects']), dict(type='text',text=json.dumps(payload))])]
    def checked(value):
        if 'tool_calls' in value:
            for call in value['tool_calls']:
                if call['arguments']['identity_asset_id'] != identity_asset_id:
                    raise ValueError('Query must use the supplied identity asset')
            return value
        from atw.world_probing.binding_reader import read_binding
        normalized,changes=read_binding(value,ids)
        if changes:
            write(output/f'normalization_turn_{turns:04d}.json',dict(changes=changes,raw=value,normalized=normalized))
        return validate(normalized)
    turns, trace = 0, []
    feedback=ToolFeedback()
    while True:
        turns += 1
        request_dir = output / f'turn_{turns:04d}' / 'request'
        from atw.world_probing.binding_reader import binding_record
        result = OpenRouterClient(context.vlm).complete(messages, request_dir, validate=checked,
            tool_definitions=definitions, normalize_response=binding_record)
        if 'tool_calls' not in result:
            write(output/'result.json', result)
            return result
        response = json.loads((request_dir/'provider_response.json').read_text())
        assistant = response['choices'][0]['message']
        messages.append(dict(role='assistant', content=assistant.get('content'), tool_calls=assistant['tool_calls']))
        provider_ids = json.loads((request_dir/'provider_call_ids.json').read_text())
        images = []
        for call, provider_id in zip(result['tool_calls'], provider_ids):
            observed = context.tools.call(call['tool_name'], call['arguments'], stage='stage_two', question=question)
            trace.append(dict(turn=turns, provider_call_id=provider_id, call_id=observed['call_id']))
            messages.append(dict(role='tool', tool_call_id=provider_id, content=json.dumps(feedback.view(observed))))
            if observed['status']=='ok':
                for item in observed['artifacts']:
                    if item['media_type']=='image/png':
                        images.extend([dict(type='text',text=f"Identity details card: {item['artifact_id']}"),
                                       media(context.run_dir/item['path'])])
        write(output/'tool_trace.json', trace)
        if images:
            replace_card_images(messages)
            messages.append(dict(role='user',content=images))
            # The one scene-wide query has returned all supplementary observations.
            definitions = []
