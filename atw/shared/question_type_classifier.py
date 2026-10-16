MODES = {"property", "count", "explanatory", "predictive", "counterfactual", "goal"}


SYSTEM = """Classify the reasoning mode required to answer one physical-scene question. Do not answer the question. Use only the literal question and its answer options.

Choose exactly one mode:
- property: asks for an intrinsic physical attribute, an attribute comparison, a static relationship, or existence. Examples include mass, density, tension, elasticity, plasticity, material, color, shape, and whether an object exists. It does not ask for a number or for the outcome of physical evolution.
- count: asks how many objects or entities there are.
- explanatory: asks what caused, enabled, or was responsible for an observed event.
- predictive: asks what happens next or what the final outcome will be in the unmodified world. Treat questions about whether a dynamic event occurs—such as collision, falling over, or making contact—and questions about final contact or final pose as predictive, even if they omit words such as "will" or "next."
- counterfactual: asks what would occur in an explicitly modified hypothetical world, such as after removing an object or changing a physical property.
- goal: asks which intervention or action would achieve a desired outcome.

Apply these precedence rules when wording overlaps:
1. A desired outcome plus a requested intervention is goal.
2. An explicit hypothetical modification whose outcome is requested is counterfactual.
3. A causal or responsibility question about an observed event is explanatory.
4. A request for a number is count, even if a dataset might call it a color or shape question.
5. A next/final outcome or the occurrence of a dynamic event in the unmodified world is predictive.
6. Other intrinsic attributes, static relationships, comparisons, and existence questions are property.

Return exactly one JSON object with this schema:
{"reasoning_mode":"property|count|explanatory|predictive|counterfactual|goal","rationale":"one short sentence grounded only in the wording"}"""


def validate_classification(value):
    if not isinstance(value, dict) or set(value) != {"reasoning_mode", "rationale"}:
        raise ValueError("Return exactly reasoning_mode and rationale")
    if value["reasoning_mode"] not in MODES:
        raise ValueError("reasoning_mode must be property, count, explanatory, predictive, counterfactual or goal")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("rationale must be a nonempty string")
    return dict(reasoning_mode=value["reasoning_mode"], rationale=value["rationale"].strip())
