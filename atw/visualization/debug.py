"""One offline entry point for retained observations and fitted-world views."""

import json
from pathlib import Path


def read(path):
    return Path(path).read_text(encoding="utf-8")


def render_run(run, output, *, workers=4, **render_options):
    from atw.perception.object_reconstruction import read_packed_mask_archive
    from atw.perception.pipeline.rendering import render_tracking
    from atw.world_modeling.sam3_rendering import write_named_overlay
    from atw.visualization.export import export_run
    from atw.visualization.render import render_bundle

    run, output = Path(run).resolve(), Path(output).resolve()
    if workers < 1:
        raise ValueError("Workers must be positive")
    if output == run or output.is_relative_to(run):
        raise ValueError("Debug output must be outside the input run")
    first = json.loads(read(run / "stage_one/result.json"))
    scene = first["data"]["scene"]
    artifacts = {a["artifact_id"]: a["path"] for a in first["artifacts"]}

    def retained(base, relative):
        path = (base / relative).resolve()
        if not path.is_relative_to(run):
            raise ValueError("Retained artifacts must belong to the input run")
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    trackpath = retained(run, artifacts[scene["sources"]["tracking_asset_id"]])
    tracking = json.loads(read(trackpath))
    pixel_views = None
    if tracking["representation"] == "observed_pixel_masks":
        pixel_views = {
            name: retained(run, artifacts[tracking["assets"][name + "_asset_id"]])
            for name in ("overview", "first_frame", "palette")
        }
    else:
        segpath = retained(run, artifacts[scene["sources"]["segmentation_asset_id"]])
        segmentation = json.loads(read(segpath))
        inventory = json.loads(
            read(retained(segpath.parent, segmentation["inventory"]))
        )
        masks = retained(segpath.parent, segmentation["mask_archive"])
        index = retained(segpath.parent, segmentation["mask_index"])
        video = Path(segmentation["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
    tracking_dir = (
        retained(trackpath.parent, tracking["tracking_directory"])
        if tracking["representation"] == "3d"
        else None
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {"engine": scene["engine"], "status": "running", "stages": {}}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        (output / "segmentation").mkdir()
        if pixel_views is not None:
            import shutil

            paths = []
            for name, source in pixel_views.items():
                target = output / "segmentation" / (name + source.suffix)
                shutil.copyfile(source, target)
                paths.append(str(target.relative_to(output)))
            report["stages"]["segmentation"] = {
                "status": "completed",
                "kind": "retained_pixel_views",
                "paths": paths,
            }
        else:
            tracks = read_packed_mask_archive(masks, index)
            frames = {}
            for track, trajectory in tracks.items():
                for frame, mask in trajectory.items():
                    frames.setdefault(frame, {})[track] = mask
            write_named_overlay(
                video_path=video,
                masks_by_frame=frames,
                objects=inventory["objects"],
                output_path=output / "segmentation/segmentation_named.mp4",
            )
            del tracks, frames
            report["stages"]["segmentation"] = {
                "status": "completed",
                "path": "segmentation/segmentation_named.mp4",
            }
        save()
        if tracking_dir is not None:
            render_tracking(
                tracking_dir, output_dir=output / "tracking", workers=workers
            )
            report["stages"]["tracking"] = {"status": "completed", "path": "tracking"}
        else:
            report["stages"]["tracking"] = {
                "status": "not_applicable",
                "reason": "This representation retains planar masks, not 3D tracks or depth.",
            }
        save()
        bundle = export_run(run, output / "swr/bundle")
        result = render_bundle(bundle, output / "swr/render", **render_options)
        report["stages"]["swr"] = {
            "status": "completed",
            "path": "swr/render",
            "report": result,
        }
        report["status"] = "completed"
        save()
        return report
    except Exception as exc:
        report.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        save()
        raise
