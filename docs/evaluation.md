# Datasets and evaluation

Prepare the official data:

```bash
python scripts/prepare_clevrer.py
python scripts/prepare_contphy.py --variant full
python scripts/prepare_evaluation.py --data-root data --output-dir evaluation
```

The subset contains CLEVRER scenes **10000–10999 inclusive**, with explanatory, predictive and counterfactual questions (4,280 questions), and ContPhy scenes **350–499 inclusive** in each of cloth, softbody, rope and fluid (600 scenes, 1,950 questions). Original question wording and options are retained. Descriptive CLEVRER questions are excluded.

Preparation writes:

- `inputs/<scene>.json`: inference requests, without answer labels.
- `reference.json`: offline scoring labels, question types and source-file hashes.
- `runs.json`: a scene-to-output-directory mapping, relative to this file.

Run each request with `scripts/run_atw.py`, using the corresponding output directory from `runs.json`. An external batch scheduler may distribute independent scene runs. To score a smaller selection or a new attempt, create a manifest containing only the desired scene IDs and output directories; do not combine multiple attempts for one scene.

```json
{"clevrer_10000": "runs/clevrer_10000", "rope_350": "runs/rope_350"}
```

Score saved predictions without invoking a model:

```bash
python scripts/evaluate.py --reference evaluation/reference.json \
  --runs evaluation/runs.json --output evaluation/scores.json
```

CLEVRER and ContPhy are reported separately. The report includes exact question accuracy, option accuracy, perfect-scene accuracy, per-scene/per-type results, coverage, and video-fallback counts and accuracy. ContPhy also reports property, predictive, counterfactual and goal-driven categories for each physical domain.

For multiple-choice questions, exact accuracy requires the predicted option set to match the reference set. Option accuracy counts every option's selected/not-selected decision. Open answers use case/whitespace normalization, integer normalization, and equivalent “cannot answer” spelling. Missing or invalid answers count as incorrect in `accuracy_all`; `accuracy_answered` is also reported. A perfect scene requires every question to be answered correctly. Fallback is detected from the saved trigger or `input_video` evidence, without changing the scoring rules.
