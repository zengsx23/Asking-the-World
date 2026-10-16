"""The sole extension point for dataset-specific answer conventions."""
from atw.contracts import BENCHMARKS, Question, json_copy, validate_answer_record
from atw.world_modeling import sam3_protocol as s


def prepare_questions(benchmark: str, questions: list[Question]) -> list[Question]:
    """Preserve source questions; dataset conventions only affect final output."""
    if benchmark not in BENCHMARKS:
        raise ValueError("Unsupported benchmark")
    return json_copy(questions)


def answer_question(benchmark, question, mode=None, engine=None):
    """Apply routed answer vocabulary only at the final-answer boundary."""
    if benchmark == 'contphy' and engine == 'softbody' and mode == 'property' and not question['options']:
        return dict(question, answer_format='yes_no')
    return question


def final_format(benchmark, question, *, allow_unresolved=True):
    """Model response format, translated explicitly to the shared answer record."""
    if benchmark not in BENCHMARKS:
        raise ValueError('Unsupported benchmark')
    ids = s.array(s.TEXT, minItems=1, uniqueItems=True)
    options = question['options']
    per_option = bool(options) and benchmark != 'contphy'
    instructions = ('Return only JSON with exactly the example fields, replacing placeholders. '
                    'Cite supplied evidence IDs; give brief reasons. ')
    instructions += ('Use the unresolved form for missing necessary evidence. ' if allow_unresolved else
                     'This is the final original-video fallback: give your best-supported answer from the video. '
                     'When observations are ambiguous, choose the most plausible answer using visual and physical '
                     'reasoning, and briefly explain the inference. Do not return unresolved. ')
    if per_option:
        row = s.obj(option_id=dict(type='string', enum=[o['id'] for o in options]),
                    decision=dict(type='string', enum=['yes', 'no']), evidence_ids=ids, reason=s.TEXT)
        answered = s.obj(action=dict(const='answer'), per_option=s.array(row, minItems=len(options), maxItems=len(options)))
        example = dict(action='answer',per_option=[dict(option_id='<option_id>',decision='yes',
                       evidence_ids=['<evidence_id>'],reason='<brief reason>')])
        instructions += ('Cover every option exactly once; decision is yes or no. Yes selects an option matching '
                         'the literal question, including negation. ')
        instructions += ('Select exactly one option. ' if question.get('answer_format')=='single_choice' else
                         'Multiple or zero selections are allowed. ')
    else:
        if options:
            choice = dict(type='string', enum=[o['id'] for o in options])
            single = question.get('answer_format') == 'single_choice'
            answer = choice if single else s.array(choice, minItems=1, uniqueItems=True)
            example_answer = '<option_id>' if single else ['<option_id>']
            instructions += ('Return one supplied option ID. ' if single else
                             'Return a nonempty array of all applicable, unique option IDs. ')
        else:
            fmt = question.get('answer_format')
            answer = dict(type=['string', 'number', 'boolean'])
            example_answer = '<scalar>'
            instructions += 'Answer must be a scalar. '
            if benchmark == 'contphy' and fmt in (None, 'open_ended'):
                instructions += ('For a supported conclusion that the requested comparison cannot be made '
                                 '(including independent physical systems or a confirmed absent comparison object), '
                                 'set answer to "can not answer" and keep action="answer". ')
                if allow_unresolved:
                    instructions += 'Request failures or missing necessary evidence remain unresolved. '
            if fmt == 'integer':
                answer = dict(type='integer');example_answer=0
                instructions += 'Use an integer. '
            if fmt in ('yes_no', 'yes_no_unknown'):
                allowed = ['yes', 'no']
                if fmt == 'yes_no_unknown':allowed.append('can not answer')
                answer = dict(type='string', enum=allowed)
                example_answer = allowed[0]
                instructions += 'Allowed scalar answers: '+', '.join(allowed)+'. '
                if benchmark == 'contphy' and fmt == 'yes_no_unknown':
                    instructions += ('Set answer to "can not answer" with action="answer" for a supported non-comparability conclusion. ')
                    if allow_unresolved:
                        instructions += 'Request failure or missing necessary evidence remains unresolved. '
        answered = s.obj(action=dict(const='answer'), answer=answer, evidence_ids=ids, reason=s.TEXT)
        example = dict(action='answer',answer=example_answer,evidence_ids=['<evidence_id>'],reason='<brief reason>')
    schema = dict(oneOf=[answered, s.obj(action=dict(const='unresolved'), reason=s.TEXT)]) if allow_unresolved else answered
    result = dict(schema=schema, instructions=instructions, per_option=per_option, answered_example=example)
    if allow_unresolved:
        result['unresolved_example'] = dict(action='unresolved',reason='<specific missing evidence>')
    return result


def read_final_response(value):
    """Repair a scalar answer repeated in action; all other validation stays strict."""
    if isinstance(value, dict):
        action = value.get('action')
        if isinstance(action, str) and action not in ('answer', 'unresolved') and action == value.get('answer'):
            return dict(value, action='answer')
    return value


def decode_final(value, benchmark, question, evidence_ids, *, allow_unresolved=True):
    """Dataset-shaped model JSON never crosses a stage/tool answer boundary."""
    convention = final_format(benchmark, question, allow_unresolved=allow_unresolved)
    s.validate(convention['schema'], value, 'final answer')
    record = dict(question_id=question['id'], status='unresolved', answer=None, evidence_ids=[], reason=value.get('reason', ''))
    if value['action'] == 'unresolved':return validate_answer_record(record, question, evidence_ids)
    def references(rows):
        ids = list(dict.fromkeys(i for row in rows for i in row['evidence_ids']))
        if not set(ids) <= evidence_ids:raise ValueError('Cite only evidence supplied to this final request')
        return ids
    def coverage(rows):
        returned = [r['option_id'] for r in rows]
        allowed = [o['id'] for o in question['options']]
        if len(returned) != len(allowed) or set(returned) != set(allowed):
            raise ValueError('Cover every supplied option exactly once')
    if convention['per_option']:
        rows = value['per_option'];coverage(rows)
        selected = {r['option_id'] for r in rows if r['decision'] == 'yes'}
        answer = [o['id'] for o in question['options'] if o['id'] in selected]
        ids = references(rows)
        reason = '; '.join(f"{r['option_id']}: {r['reason']}" for r in rows)
    else:
        answer = value['answer'];ids = references([value]);reason = value['reason']
        if question['options'] and isinstance(answer, str):answer = [answer]
    record.update(status='answered', answer=answer, evidence_ids=ids, reason=reason)
    return validate_answer_record(record, question, evidence_ids)
