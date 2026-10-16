"""Image-plane gravity decision, independent of routing/reselection decisions."""
import json
from pathlib import Path

from atw.world_modeling import sam3_protocol as contract

DECISION = contract.obj(gravity=dict(type='string', enum=['screen_down', 'screen_inward']),
                        reason=contract.TEXT)


def estimate(context, *, admission_review=None):
    """Reuse the existing 2D router client's configuration without overrides."""
    from atw.perception.world_model_router import WorldModelRouter
    from atw.perception.plane_gravity import gravity_artifact
    from atw.perception.sam3_agent import _media_data_url

    root = Path(__file__).resolve().parents[2]
    client = WorldModelRouter.from_project(root).client
    client.trace_dir = context.output_dir / 'requests'
    questions = [{k: q[k] for k in ('id', 'text', 'options')} for q in context.inputs['questions']]
    video = Path(context.inputs['video_path'])
    kind, url = _media_data_url(video)
    evidence=dict(questions=questions)
    if admission_review is not None:evidence['admission_review']=admission_review
    content = [dict(type=kind, **{kind: dict(url=url)}),
               dict(type='text', text=json.dumps(evidence)+contract.prompt(DECISION))]
    prompt = (Path(__file__).resolve().parents[1] / 'prompts/gravity_2d_system.txt').read_text()
    if admission_review is not None:
        prompt += '\nThe original gravity direction is very likely wrong (原来的重力方向很可能有问题): it failed SWR admission. Re-examine the original video, the original direction and explanation, and the admission rejection. Correct the gravity direction based on visible support, suspension and motion evidence. A wooden backdrop or a supported stationary object alone does not establish inward gravity. Do not change direction solely to satisfy the engine: if the original direction remains supported, retain it and explain why. Return the same gravity and reason schema.'
        (context.output_dir/'gravity_admission_request.json').write_text(json.dumps(admission_review,indent=2)+'\n')
    def validate(value):
        value, _ = contract.normalize_response(DECISION, value)
        contract.validate(DECISION, value, 'gravity_2d')
    try:
        raw = client.request_json(operation='gravity_2d_admission_review' if admission_review is not None else 'gravity_2d', call_id=context.call_id,
                                  system=prompt, content=content, validate=validate,
                                  allow_single_object_array=True)
        decision, _ = contract.normalize_response(DECISION, raw)
        validate(decision)
        result = gravity_artifact(decision)
        result['reason'] = decision['reason']
        (context.output_dir / 'gravity_2d.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        (context.output_dir / 'vlm_usage.json').write_text(json.dumps(client.usage_report(), indent=2)+'\n')
