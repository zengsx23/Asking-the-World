# Usage

The entry point is `scripts/run_atw.py`. By default it executes both stages in a **new** output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_atw.py \
  --input evaluation/inputs/rope_350.json \
  --output-dir evaluation/runs/rope_350 --device cuda:0
```

`--stage stage_one` builds assets only. `--stage both` builds them and answers the questions. `cuda:0` denotes the first GPU exposed by `CUDA_VISIBLE_DEVICES`. Concurrent jobs must use separate output directories and have sufficient GPU memory.

## Input

```json
{
  "protocol_version": 1,
  "benchmark": "contphy",
  "video_path": "/absolute/path/video.mp4",
  "questions": [
    {
      "id": "q1",
      "text": "Is there any orange rope in the video?",
      "options": [],
      "answer_format": "open_ended"
    }
  ]
}
```

Multiple-choice questions use `"answer_format": "multi_select"` and `"options": [{"id": "A", "text": "..."}]`. Question IDs are strings and unique within a scene. Benchmark labels are `clevrer` or `contphy`. CLEVRER requests may include `question_type`; descriptive questions are excluded. Ground-truth answers do not belong in a request.

## Outputs and reuse

Each run contains `input.json`, `runner.json`, `result.json`, `stage_one/result.json`, and, when reached, `stage_two/result.json`. Stage-two answers and evidence traces are saved under `stage_two/`. Inspect status and error fields rather than treating the presence of a directory as success.

Stage-one results retain references to masks, identities, reconstructed objects and fitted physical assets. Keep the referenced files together and preserve their recorded paths when reusing an existing run. No asset conversion is required between the two stages.

For programmatic reuse, the stage-two implementation is `atw.probing.run(inputs, context)` and the shared runtime is `atw.tools.ToolRunner`. Supply the saved stage-one result, the original video and the requested questions; construct `StageContext` with the new run/output directories and the runtime. See the handoff in `atw/runner.py` for the exact input structure and result handling.

Rope assets can cover only some independent pulley groups. Capability queries expose the controllable objects and groups; unavailable interventions must use the existing video fallback. Partial coverage remains recorded in diagnostics and limitations. A completed answer does not convert a failed world-modeling stage into a successful one.
