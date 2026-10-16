"""Question-only existence routing and binding-first presence answers."""
import json
from pathlib import Path

from atw.contracts import validate_answer_record
from atw.world_modeling import sam3_protocol as s
from atw.world_modeling.scene_assets import asset_path
from atw.world_probing.identities import request
from atw.world_probing.evidence import write

CONDITIONAL_ANSWER = dict(type=['string', 'array', 'null'], items=s.TEXT)
CLASSIFICATION_SCHEMA = s.obj(is_existence=dict(type='boolean'), rationale=s.TEXT,
    answer_when_present=CONDITIONAL_ANSWER, answer_when_absent=CONDITIONAL_ANSWER)
VIDEO_SCHEMA = s.obj(exists=dict(type='boolean'), reason=s.TEXT)
ANSWER_SCHEMA = s.obj(question_id=s.TEXT, status=dict(type='string', enum=['answered', 'unresolved']),
    answer=dict(type=['string', 'array', 'null'], items=s.TEXT), evidence_ids=s.array(s.TEXT), reason=s.TEXT)
PROMPTS = Path(__file__).parents[1] / 'prompts'


def classify(question, output, config):
    def validate(value):
        for key in ('answer_when_present', 'answer_when_absent'):
            answer = value[key]
            if not value['is_existence']:
                if answer is not None:
                    raise ValueError('Non-existence questions require null conditional answers')
            else:
                validate_answer_record(dict(question_id=question['id'], status='answered',
                    answer=answer, evidence_ids=[], reason=value['rationale']), question, set())
                if not question['options'] and answer not in ('yes', 'no'):
                    raise ValueError('Scalar existence answers must be yes or no')
                if question['options'] and not answer:
                    raise ValueError('Conditional choice answers require at least one supplied option')
        if value['is_existence'] and value['answer_when_present'] == value['answer_when_absent']:
            raise ValueError('Presence and absence must resolve to different answers')
        return value
    return request(config, output, (PROMPTS/'stage_two_existence_classification.txt').read_text(),
        dict(question=question['text'], options=question['options']), CLASSIFICATION_SCHEMA, validate)


def answer(question, classification, binding, context, output, binding_asset_id):
    output.mkdir(parents=True, exist_ok=False)
    if not classification['is_existence']:
        raise ValueError('Existence answering requires a positive existence classification')
    rows = binding['identity_evidence']
    found = bool(rows) and all(row['status'] == 'found' and row['object_ids'] for row in rows)
    if found:
        observed = dict(exists=True, reason='All described entity references have matching retained identities.')
        evidence_ids = [binding_asset_id]
        basis = 'identity_binding'
    else:
        observed = request(context.vlm, output/'video_review',
            (PROMPTS/'stage_two_existence_video.txt').read_text(),
            dict(question=question['text'], options=question['options']), VIDEO_SCHEMA,
            lambda value:value, [Path(context.inputs['video_path'])])
        evidence_ids = ['input_video']
        basis = 'original_video'
    value = dict(question_id=question['id'], status='answered',
        answer=classification['answer_when_present' if observed['exists'] else 'answer_when_absent'],
        evidence_ids=evidence_ids, reason=observed['reason'])
    validate_answer_record(value, question, set(evidence_ids))
    write(output/'decision.json', dict(basis=basis, **observed))
    write(output/'result.json', value)
    return value


def load(arguments, context, key):
    return json.loads(asset_path(context, arguments[key]).read_text())
