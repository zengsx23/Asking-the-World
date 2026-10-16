"""Fresh final-answer requests over literal compact tool findings, without summaries."""
from pathlib import Path

from atw.answer_conventions import answer_question, final_format, decode_final, read_final_response
from atw.contracts import json_copy
from atw.world_probing.identities import request
from atw.world_probing.collection_input import model_error
from atw.world_probing.evidence import write

PROMPT = Path(__file__).parents[1]/'prompts/stage_two_final_answer.txt'


class VideoFallbackFailed(RuntimeError):
    """The final video request exhausted its shared budget; do not retry it externally."""


def question_input(question):
    return {k:question[k] for k in ('id','text','options','answer_format','answer_instructions') if k in question}


def run(config, output, question, benchmark, evidence, *, binding=(), paths=(), mode=None, engine=None,
        allow_unresolved=True, observed_end_seconds=None):
    question = answer_question(benchmark, question, mode, engine)
    convention = final_format(benchmark, question, allow_unresolved=allow_unresolved)
    payload = dict(question=question_input(question),
                   output_format={k:convention[k] for k in ('instructions','answered_example','unresolved_example') if k in convention},
                   identity_bindings=[{k:row[k] for k in ('reference','status','object_ids')} for row in binding],
                   **prepare_evidence(evidence))
    from atw.world_probing.prediction_scope import applies, PREDICTION_GUIDANCE
    if applies(mode, engine):
        scope = dict(reasoning_mode='predictive', time_scope='strictly after the observed video cutoff',
                     instructions=PREDICTION_GUIDANCE)
        if observed_end_seconds is not None:
            scope['observed_end_seconds'] = observed_end_seconds
        payload['prediction_scope'] = scope
    allowed = {row['evidence_id'] for row in evidence if 'error' not in row}
    allowed.update(a['artifact_id'] for row in evidence if 'error' not in row for a in row.get('media', []))
    labels = [f"{row['evidence_id']} / {a['artifact_id']} ({a['source_kind']})" for row in evidence for a in row.get('media', [])]
    if paths and not labels:labels = ['input_video: complete original video']
    return request(config, output, PROMPT.read_text(), payload, convention['schema'],
                   lambda value:decode_final(value, benchmark, question, allowed, allow_unresolved=allow_unresolved), paths,media_labels=labels,
                   include_schema_in_prompt=False, normalize_response=read_final_response)


def prepare_evidence(evidence):
    """Deduplicate exact fields/notes without interpreting physics or changing findings."""
    rows=json_copy(evidence)
    notes={}
    for row in rows:
        row.pop('next_step',None)
        if row.get('note'):notes.setdefault(row['note'],[]).append(row['evidence_id'])
        # Only same-named, exactly equal fields are redundant. In particular,
        # keep the requested world when a readout returns an extended world.
        supplied={**row.get('evidence',{}),**{k:v for k,v in row.items() if k!='arguments'}}
        arguments={k:v for k,v in row.get('arguments',{}).items() if k not in supplied or supplied[k]!=v}
        if arguments:row['arguments']=arguments
        else:row.pop('arguments',None)
    shared=[]
    for note,ids in notes.items():
        if len(ids)>1:
            shared.append(dict(evidence_ids=ids,note=note))
            for row in rows:
                if row.get('note')==note:del row['note']
    result=dict(evidence=rows)
    if shared:result['evidence_notes']=shared
    return result


def collect(trace, arguments, run_dir):
    """Keep actual arguments, findings and notes; never copy planning conclusions."""
    evidence, paths = [], []
    for receipt in trace:
        if receipt['tool_name']=='finish_evidence_collection':continue
        row = dict(evidence_id=receipt['call_id'], tool_name=receipt['tool_name'],
                   arguments=arguments[receipt['call_id']])
        if receipt['status'] == 'error':
            row['error'] = model_error(receipt['error'])
        else:
            row.update(receipt['data'])
            images = [a for a in receipt['artifacts'] if a['media_type'].startswith(('image/', 'video/'))]
            if images:
                row['media'] = [dict(artifact_id=a['artifact_id'], source_kind=a['source_kind']) for a in images]
                paths.extend(run_dir/a['path'] for a in images)
        evidence.append(row)
    return evidence, paths


def from_video(config, output, question, benchmark, video, *, mode=None, engine=None, reason=None):
    if reason is not None:
        write(output.parent/'video_fallback_trigger.json',dict(reason=reason))
    evidence = [dict(evidence_id='input_video', source_kind='observation',
                     note='Independently reason from the complete attached original video. No simulation evidence is supplied.')]
    try:
        from atw.world_probing.prediction_scope import applies, video_cutoff
        scope = dict(observed_end_seconds=video_cutoff(video)) if applies(mode, engine) else {}
        return run(config, output, question, benchmark, evidence, paths=[Path(video)], mode=mode, engine=engine,
                   allow_unresolved=False, **scope)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        failure = f'{type(exc).__name__}: {exc}'
        write(output.parent/'video_fallback_error.json',dict(reason=failure))
        raise VideoFallbackFailed(failure) from exc
