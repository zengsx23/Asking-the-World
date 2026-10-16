"""Stage two: experiment and answer using explicitly handed-off assets."""
from atw import answer_conventions
from atw.config import StageContext
from atw.contracts import StageResult, StageTwoInput, stage_result


def run(inputs: StageTwoInput, context: StageContext) -> StageResult:
    """Prepare identities, collect engine evidence and independently answer.

    Only explicit exhausted admission review permits original-video fallback
    from a failed upstream result.
    """
    from atw import direct_video_answer
    if direct_video_answer.requested(inputs['stage_one_result']):
        return direct_video_answer.run(inputs,context)
    if inputs["stage_one_result"]["status"] != "completed":
        if inputs["stage_one_result"]["status"] == "not_implemented":
            return stage_result("stage_two")
        return stage_result("stage_two", "blocked", error={
            "code": "upstream_not_completed",
            "message": "Stage two requires a completed stage-one result.",
        })
    questions = answer_conventions.prepare_questions(inputs["benchmark"], inputs["questions"])
    if not questions:
        return stage_result('stage_two')
    from atw.world_probing import pipeline
    return pipeline.run(inputs, context, questions)
