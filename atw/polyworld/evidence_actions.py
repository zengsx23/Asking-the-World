"""Actionable missing-evidence receipts shared by second-stage scene tools."""


def query_requirement(reason, next_action):
    return dict(status="needs_query", reason=reason, next_action=next_action,
        if_unresolved="Request request_video_fallback for this question only. Do not infer a negative answer from missing evidence.")


def actionable_result(value):
    """Annotate local gaps, not the entire scene or unrelated successful queries."""
    if isinstance(value, list):
        rows = [actionable_result(row) for row in value]
        return value if all(a is b for a, b in zip(rows, value)) else rows
    if not isinstance(value, dict):
        return value
    result = {key: actionable_result(row) for key, row in value.items()}
    status = result.get("status")
    if status in {"needs_query", "invalid_query", "unsupported", "unavailable", "not_available",
                  "unreliable", "not_identified", "not_comparable"}:
        result.setdefault("next_action", "Correct the requested IDs, interval and supported readout using the tool catalog and this receipt's reason; query the same completed world again where possible. Do not repeat an unchanged failed query.")
        result.setdefault("if_unresolved", "If no supported query can obtain this question's required evidence, request request_video_fallback for this question only. This is not a negative physical result.")
    return value if result.keys() == value.keys() and all(result[k] is value[k] for k in value) else result
