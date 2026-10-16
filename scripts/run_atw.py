#!/usr/bin/env python3
"""Run independent stage-one scene construction and optional stage-two handoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atw.runner import run
from atw.config import SceneConfig, ReasoningConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical request JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory")
    parser.add_argument('--stage',choices=('stage_one','both'),default='both',
                        help='Run stage one only, or also invoke the stage-two handoff (default: both)')
    parser.add_argument('--device',help='Runtime model device, e.g. cuda:0 within CUDA_VISIBLE_DEVICES; does not change VLM settings')
    parser.add_argument('--rigid-predictive-tracking-only', action=argparse.BooleanOptionalAction, default=True,
                        help='Use observed tracks and masks for rigid predictions (default: enabled); disable to use SWR simulation')
    args = parser.parse_args()
    try:
        scene=SceneConfig(device=args.device) if args.device else None
        result = run(args.input, args.output_dir,stage=args.stage,scene=scene,
                     reasoning=ReasoningConfig(rigid_predictive_tracking_only=args.rigid_predictive_tracking_only))
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(1, f"atw: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["status"] in ("blocked", "failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
