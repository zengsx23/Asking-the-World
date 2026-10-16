"""Explicit admission fallback using the shared per-question final-answer path."""
from atw.contracts import stage_result
from atw.world_probing import final_answer
from atw.world_probing.evidence import write


def requested(stage_one_result):
    return (stage_one_result['status']=='failed'
            and stage_one_result.get('error',{}).get('code')=='swr_admission_unresolved'
            and stage_one_result.get('data',{}).get('scene',{}).get('direct_video_fallback') is True)


def run(inputs, context):
    answers, failures = [], []
    for index, question in enumerate(inputs['questions']):
        output = context.output_dir/'reasoning'/f'q_{index:04d}'
        try:
            answer = final_answer.from_video(context.vlm, output/'video_fallback', question,
                                             inputs['benchmark'], inputs['video_path'])
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
            answer = dict(question_id=question['id'], status='unresolved', answer=None,
                          evidence_ids=[], reason=reason)
            failures.append(question['id'])
        answers.append(answer)
        write(output/'result.json', answer)
        write(context.output_dir/'answers.json', dict(answers=answers))
    if failures:
        return stage_result('stage_two', 'failed', data=dict(answers=answers), error=dict(
            code='question_processing_failed', message='Original-video inference failed for: '+', '.join(failures)))
    return stage_result('stage_two', 'completed', data=dict(answers=answers))
