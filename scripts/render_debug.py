#!/usr/bin/env python3
"""Render saved segmentation, tracking/depth and fitted-world debug views in one command."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir", type=Path, help="Completed ATW run with stage_one/result.json"
    )
    source.add_argument(
        "--bundle",
        type=Path,
        help="Previously exported scene.json and assets.npz directory",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory; existing output is never overwritten",
    )
    p.add_argument(
        "--blender",
        type=Path,
        help="Blender 4.5 executable (not needed for rigid point clouds)",
    )
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="Optional QA frames; gaps produce still images, not a retimed movie",
    )
    p.add_argument(
        "--cloth-contact", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--support-tolerance",
        type=float,
        help="Optional rigid background height cutoff in reconstruction units",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="CPU workers for retained tracking/depth renderers",
    )
    args = p.parse_args()
    from atw.visualization.debug import render_run
    from atw.visualization.render import render_bundle

    output = args.output_dir.resolve()
    options = dict(
        blender=args.blender,
        device=args.device,
        frames=args.frames,
        scale=args.scale,
        samples=args.samples,
        cloth_contact=args.cloth_contact,
        support_tolerance=args.support_tolerance,
    )
    if args.run_dir:
        result = render_run(args.run_dir, output, workers=args.workers, **options)
    else:
        bundle = args.bundle.resolve()
        if output == bundle or output.is_relative_to(bundle):
            p.error("Debug output must be outside the input bundle")
        output.mkdir(parents=True, exist_ok=False)
        result = render_bundle(bundle, output / "render", **options)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
