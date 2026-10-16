"""Programmatic stage-two scheduling through the shared canonical ToolRunner."""
from atw.contracts import stage_result, validate_answer_record, json_copy
from atw.world_probing.evidence import write
from atw.world_probing import final_answer


def run(inputs, context, questions):
    from atw.tools import ToolRunner
    tools = context.tools or ToolRunner(run_dir=context.run_dir, vlm=context.vlm,
        inputs=dict(video_path=inputs['video_path'], questions=questions),
        artifacts=inputs['stage_one_result']['artifacts'])
    tools.stage_one_result = json_copy(inputs['stage_one_result'])
    artifacts, preparation, answers = [], [], []
    root = context.output_dir
    write(root/'tools.json', dict(tools=tools.definitions('stage_two')))

    def call(name, arguments, question=None):
        result = tools.call(name, arguments, stage='stage_two', question=question)
        artifacts.extend(result['artifacts'])
        return result

    for question in questions:
        result = call('classify_question_reasoning_mode', dict(question_id=question['id']), question)
        preparation.append(dict(question_id=question['id'], reasoning_mode=result['data'].get('reasoning_mode'),
            classification_artifact_id=result['artifacts'][0]['artifact_id'] if result['status']=='ok' else None,
            is_existence=None, existence_classification_artifact_id=None, existence_artifact_id=None,
            binding_artifact_id=None, count_artifact_id=None, error=result['error']))
        row = preparation[-1]
        if row['error'] is None and row['reasoning_mode'] == 'property':
            check = call('classify_property_existence', dict(question_id=question['id']), question)
            row['error'] = check['error']
            if check['status'] == 'ok':
                row['is_existence'] = check['data']['is_existence']
                row['existence_classification_artifact_id'] = check['artifacts'][0]['artifact_id']
        write(root/'preparation.json', dict(questions=preparation))

    identity_id, shared_error = None, None
    if any(row['error'] is None for row in preparation):
        result = call('ground_scene_identities',
                      dict(tracking_asset_id=inputs['stage_one_result']['data']['scene']['sources']['tracking_asset_id']))
        identity_id, shared_error = result['data'].get('identity_asset_id'), result['error']
    for index, (question, row) in enumerate(zip(questions, preparation)):
        fallback_failed = False
        answer = dict(question_id=question['id'], status='unresolved', answer=None, evidence_ids=[],
                      reason='Required question evidence is unavailable.')
        if row['error'] is None and shared_error is not None:
            row['error'] = shared_error
        if row['error'] is None:
            is_count = row['reasoning_mode']=='count'
            result = call('count_scene_identities' if is_count else 'bind_question_identities',
                          dict(question_id=question['id'],identity_asset_id=identity_id),question)
            row['error'] = result['error']
            if result['status']=='ok':
                row['count_artifact_id' if is_count else 'binding_artifact_id'] = result['artifacts'][0]['artifact_id']
                if is_count:
                    answer = result['data']
                elif row['is_existence']:
                    result = call('answer_scene_existence', dict(question_id=question['id'],
                        classification_asset_id=row['existence_classification_artifact_id'],
                        binding_asset_id=row['binding_artifact_id']), question)
                    row['error'] = result['error']
                    if result['status'] == 'ok':
                        answer = result['data']
                        row['existence_artifact_id'] = result['artifacts'][0]['artifact_id']
                else:
                    from atw.world_probing import reasoning
                    try:
                        answer, retained = reasoning.run(inputs,tools,question,row['reasoning_mode'],
                            result['data'],root/'reasoning'/f'q_{index:04d}')
                        artifacts.extend(retained)
                    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
                        fallback_failed=isinstance(exc,final_answer.VideoFallbackFailed)
                        row['error']=dict(code='video_fallback_failed' if fallback_failed else 'reasoning_failed',
                                          message=f'{type(exc).__name__}: {exc}')
        if row['error'] is not None:
            answer.update(status='unresolved', answer=None, reason=row['error']['message'])
        if answer['status']=='unresolved' and not fallback_failed:
            output=root/'reasoning'/f'q_{index:04d}'
            try:
                answer=final_answer.from_video(context.vlm,output/'video_fallback',question,
                    inputs['benchmark'],inputs['video_path'],mode=row['reasoning_mode'],
                    engine=inputs['stage_one_result']['data']['scene']['engine'],reason=answer['reason'])
                row['error']=None
            except final_answer.VideoFallbackFailed as exc:
                row['error']=dict(code='video_fallback_failed',message=str(exc))
                answer.update(status='unresolved',answer=None,evidence_ids=[],reason=str(exc))
            write(output/'result.json',answer)
        answers.append(validate_answer_record(answer,question,{'input_video',*tools.evidence_ids}))
        write(root/'answers.json',dict(answers=answers))
        write(root/'preparation.json',dict(questions=preparation))
    artifacts.append(dict(artifact_id='stage_two_preparation',path=str((root/'preparation.json').relative_to(context.run_dir)),
                          media_type='application/json',source_kind='derived'))
    data=dict(answers=answers,preparation_artifact_id='stage_two_preparation')
    if any(row['error'] for row in preparation):
        status,failure='failed',dict(code='question_processing_failed',message='One or more questions failed; see retained tool results and answers.')
    elif all(a['status']=='answered' for a in answers):
        status,failure='completed',None
    else:
        status,failure='blocked',dict(code='answer_evidence_unavailable',message='Available evidence does not resolve every question.')
    return stage_result('stage_two',status,data=data,artifacts=artifacts,error=failure)
