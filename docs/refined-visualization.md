# Debug visualization

Render saved assets from a completed run with one command. The command preserves
existing segmentation, reconstruction/tracking and depth rendering, and selects
the refined SWR renderer automatically from the scene's engine. It does not rerun
inference, fitting or VLM calls.

## Usage

Install the optional dependencies in the existing ATW environment. Blender 4.5
is required for mesh and particle scenes; rigid point clouds render on the CPU.

```bash
python -m pip install -r requirements/visualization.txt
CUDA_VISIBLE_DEVICES=0 python scripts/render_debug.py \
  --run-dir /path/to/completed-run \
  --output-dir /path/to/new-debug \
  --blender /path/to/blender --device gpu --workers 4
```

Keep the run's referenced assets and original video available. Use a new output
directory outside the source run; input files remain unchanged. `--device cpu`
is also supported. `--workers` controls the retained tracking/depth renderers.

## Outputs

- `segmentation/`: SAM3 overlays, or fluid's retained overview, first-frame image and palette.
- `tracking/`: existing 3D tracking and depth views. Rope and fluid use planar masks instead.
- `swr/bundle/` and `swr/render/`: exported fitted state and refined SWR views.
- `report.json`: completion status, output paths and inapplicable views.

Rigid scenes use dense colored points and trajectories; cloth uses triangular
surfaces and rigid meshes; fluid uses particles; rope uses curves and pulleys;
softbody uses deformed surfaces and trajectories. The current softbody adapter
supports the sphere-deformation model. Rendering uses the estimated camera or
fitted image coordinates, source RGB, and saved frame indices and frame rate.

## Appearance and options

Adjust materials, lighting, point size and trails in `atw/visualization/` for
development or presentation needs. These views do not change world models,
probing inputs or evaluation results.

`--scale` controls resolution and `--samples` controls Cycles samples. `--frames`
selects SWR preview frames while preceding debug views keep their full length.
`--bundle` can replace `--run-dir` to render only a previously exported SWR bundle.
`--support-tolerance` optionally filters rigid background points above the support
plane; omit it when elevated static background objects should remain. See
`python scripts/render_debug.py --help` for all options.
