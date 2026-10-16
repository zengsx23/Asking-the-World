"""Independent identity judgements, programmatic sums and canonical count answers."""
from pathlib import Path
from atw.world_modeling import sam3_protocol as schema
from atw.world_probing.identities import request
from atw.world_probing.evidence import write

SCOPE_SCHEMA = schema.obj(supported=dict(type='boolean'), reason=schema.TEXT)
INCREMENT_SCHEMA = schema.obj(increment=dict(type='integer', enum=[0, 1]),
                            identity_description=schema.TEXT, reason=schema.TEXT)
OPTIONS_SCHEMA = schema.obj(option_counts=schema.array(schema.obj(option_id=schema.TEXT,
                           count=dict(type='integer', minimum=0))))


def read_increment(value):
    """Accept unambiguous answer fields without treating echoed schema metadata as answers."""
    if isinstance(value, dict) and all(k in value for k in INCREMENT_SCHEMA['required']):
        return {k:value[k] for k in INCREMENT_SCHEMA['required']}
    return value


def count(question, catalog, labels, context, output):
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(question=question['text'], identity_semantics=catalog['identity_semantics'])
    scope = request(context.vlm, output/'scope',
        'Decide only whether summing one binary match per retained scene identity answers this count question. '
        'Representative identity cards support visible category/color/shape counts over scene identities. '
        'Return supported=false for counts conditioned on motion, events, a specific instant, hidden entities, '
        'or components not represented as separate identities. A liquid mask represents retained liquid '
        'from one observed outlet, not individual particles, release events or connected components. '
        'Count static emitters only through their own static identities. Decide from question semantics only: '
        'a category with no matching identities is a valid zero count, not an unsupported question. '
        'Do not infer which objects exist. Return {"supported":true,"reason":"brief reason"}.',
        dict(payload, output_format='{"supported":true,"reason":"brief reason"}'),
        SCOPE_SCHEMA, lambda value:value, include_schema_in_prompt=False)
    records, total, failure = [], None, None
    if scope['supported'] and catalog['objects']:
        by_id = {o['object_id']:o for o in labels['objects']}
        prompt = (Path(__file__).parents[1]/'prompts/stage_two_count.txt').read_text()
        for index, obj in enumerate(catalog['objects'], 1):
            card = by_id[obj['object_id']]
            try:
                decision = request(context.vlm, output/f'identity_{index:04d}',
                    prompt,
                    dict(question=question['text'], identity_id=f'{index:02d}',
                         identity_label={k:card[k] for k in ('category', 'color')},
                         output_format='{"increment":0,"identity_description":"brief description","reason":"brief reason"}'),
                    INCREMENT_SCHEMA, lambda value:value, [context.run_dir/obj['image']],
                    include_schema_in_prompt=False, normalize_response=read_increment)
                records.append(dict(object_id=obj['object_id'], **decision))
            except (RuntimeError, ValueError, OSError) as exc:
                failure = f'{type(exc).__name__}: {exc}'
                records.append(dict(object_id=obj['object_id'], error=failure))
            write(output/'identity_results.json', records)
        if failure is None:
            total = sum(r['increment'] for r in records)
    reason = (f'{total} of {len(records)} retained identities match the question.' if total is not None
              else scope['reason'] if not scope['supported'] else 'Complete identity counting evidence is unavailable.')
    answer = total
    if total is not None and question['options']:
        ids = {o['id'] for o in question['options']}
        def validate(value):
            rows = value['option_counts']
            if len(rows) != len(ids) or {r['option_id'] for r in rows} != ids:
                raise ValueError('Every supplied count option must occur exactly once')
            return value
        mapping = request(context.vlm, output/'option_counts',
            'Translate each supplied answer option into its nonnegative integer count. Do not answer the '
            'question. Do not change option IDs. Each option must denote one exact count.',
            dict(options=question['options'],output_format='{"option_counts":[{"option_id":"supplied ID","count":0}]}'),
            OPTIONS_SCHEMA, validate, include_schema_in_prompt=False)
        answer = [r['option_id'] for r in mapping['option_counts'] if r['count'] == total]
        if not answer or (question.get('answer_format') == 'single_choice' and len(answer) != 1):
            answer, reason = None, 'The measured count does not resolve the supplied option format.'
    elif total is not None and question.get('answer_format') not in (None, 'integer', 'open_ended'):
        answer, reason = None, 'This count route requires an integer-compatible scalar answer format.'
    result = dict(scope=scope, identity_results=records, count=total, answer=answer, reason=reason,
                  execution_error=failure)
    write(output/'result.json', result)
    return result
