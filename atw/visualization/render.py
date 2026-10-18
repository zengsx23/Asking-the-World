"""Render saved scene assets into an isolated output directory."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from atw.visualization.cloud import render_cloud, overlay_trails


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_bundle(
    bundle,
    output,
    *,
    blender=None,
    device="cpu",
    frames=None,
    scale=2,
    samples=20,
    cloth_contact=True,
    support_tolerance=None,
):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if scale < 1 or samples < 1:
        raise ValueError("Scale and samples must be positive")
    info = json.loads((bundle / "scene.json").read_text())
    if info["kind"] not in ("rigid", "cloth", "fluid", "rope", "softbody"):
        raise ValueError("Unknown renderer kind")
    if info["fps"] <= 0 or info["frames"] <= 0:
        raise ValueError("Invalid frame timing")
    if info["width"] * scale % 2 or info["height"] * scale % 2:
        raise ValueError("Video dimensions must be even")
    frames = (
        list(range(info["start"], info["frames"])) if frames is None else list(frames)
    )
    if (
        not frames
        or frames != sorted(set(frames))
        or frames[0] < info["start"]
        or frames[-1] >= info["frames"]
    ):
        raise ValueError("Use ordered unique frames within the saved SWR range")
    full = frames == list(range(frames[0], frames[-1] + 1))
    if device == "gpu" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly for GPU rendering")
    before = {p.name: digest(p) for p in bundle.iterdir() if p.is_file()}
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(frames=frames, scale=scale, samples=samples, device=device)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    with np.load(bundle / "assets.npz", allow_pickle=False) as archive:
        assets = dict(archive)
    if info["kind"] == "cloth" and cloth_contact:
        from atw.visualization.cloth import prepare

        display, report = prepare(info, assets)
        np.savez_compressed(output / "cloth-display.npz", **display)
        (output / "cloth-display-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    if info["kind"] != "rigid":
        if not blender:
            raise ValueError("Blender executable required for this engine")
        with (output / "blender.log").open("w") as log:
            subprocess.run(
                [
                    str(blender),
                    "-b",
                    "-t",
                    "4",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("blender_scene.py")),
                    "--",
                    str(bundle),
                    str(output),
                    str(output / "settings.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    for f in frames:
        path = output / f"{f:05d}.png"
        im = (
            render_cloud(info, assets, f, scale, support_tolerance)
            if info["kind"] == "rigid"
            else cv2.imread(str(path))
        )
        if im is None:
            raise RuntimeError(f"Missing rendered frame {f}")
        if info["kind"] in ("rigid", "softbody"):
            im = overlay_trails(im, info, assets, f)
        if not cv2.imwrite(str(path), im):
            raise OSError(path)
    # Sparse QA frames are images, never a falsely timed movie.
    if full:
        target = output / "swr.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-framerate",
                str(info["fps"]),
                "-start_number",
                str(frames[0]),
                "-i",
                str(output / "%05d.png"),
                "-frames:v",
                str(len(frames)),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ],
            check=True,
        )
        cap = cv2.VideoCapture(str(target))
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        if decoded != len(frames):
            raise RuntimeError("Encoded video frame count mismatch")
    after = {p.name: digest(p) for p in bundle.iterdir() if p.is_file()}
    if before != after:
        raise RuntimeError("Input bundle changed during rendering")
    report = dict(
        engine=info["kind"],
        frames=frames,
        fps=info["fps"],
        video="swr.mp4" if full else None,
        input_sha256=before,
        input_unchanged=True,
        device=device,
        visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        settings=settings,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
