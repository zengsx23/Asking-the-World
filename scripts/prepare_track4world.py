#!/usr/bin/env python3
"""Download and verify the pinned official Track4World DA3 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_REVISIONS = {
    "third_party/track4world": "fbd59ffadf2de9fccba5ea017e13af239075fbe9",
    "third_party/utils3d": "2072c024c73f7c0f83e0da23eef5f2d9ac575249",
    "third_party/pi3": "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b",
}


@dataclass(frozen=True, slots=True)
class OfficialAsset:
    filename: str
    size_bytes: int
    sha256: str


TRACK4WORLD_REPO = "TencentARC/Track4World"
TRACK4WORLD_REVISION = "93ae34410efd05d1b7abfdb0f749bc93f7655404"
TRACK4WORLD_CHECKPOINT = OfficialAsset(
    filename="track4world_da3.pth",
    size_bytes=5_527_438_950,
    sha256="6c589c0f41632a2329920087dcd8cbd6465d4198d574f0d4186eeada78c492c7",
)

DA3_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
DA3_ASSETS = (
    OfficialAsset(
        filename="config.json",
        size_bytes=3_113,
        sha256="09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
    ),
    OfficialAsset(
        filename="model.safetensors",
        size_bytes=6_759_558_100,
        sha256="8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c",
    ),
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "data/models/track4world/track4world_da3.pth"
)
DEFAULT_DA3_DIRECTORY = (
    PROJECT_ROOT
    / "data/models/track4world/depth-anything"
    / "DA3NESTED-GIANT-LARGE-1.1"
)
DEFAULT_TORCH_HUB = PROJECT_ROOT / "data/models/track4world/torch/hub"
CONVNEXT_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
CONVNEXT_CHECKPOINT = OfficialAsset(
    filename="convnext_tiny-983f1562.pth",
    size_bytes=114_419_221,
    sha256="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--da3-directory", type=Path, default=DEFAULT_DA3_DIRECTORY)
    parser.add_argument("--torch-hub", type=Path, default=DEFAULT_TORCH_HUB)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--source-file",
        type=Path,
        help="verified Track4World checkpoint bytes to reuse instead of downloading",
    )
    parser.add_argument(
        "--convnext-source-file",
        type=Path,
        help="verified ConvNeXt-Tiny checkpoint bytes to reuse",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, asset: OfficialAsset) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != asset.size_bytes:
        raise ValueError(
            f"size mismatch for {path}: expected {asset.size_bytes}, found {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {asset.sha256}, found {actual_sha256}"
        )


def verify_source_revisions() -> None:
    for relative_path, expected_revision in SOURCE_REVISIONS.items():
        source = PROJECT_ROOT / relative_path
        if not source.is_dir():
            raise FileNotFoundError(source)
        marker = source / ".git"
        if marker.is_file():
            prefix = "gitdir: "
            marker_text = marker.read_text(encoding="utf-8").strip()
            if not marker_text.startswith(prefix):
                raise ValueError(f"invalid gitdir marker: {marker}")
            git_directory = (source / marker_text[len(prefix) :]).resolve()
        else:
            git_directory = marker
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            reference = head.removeprefix("ref: ")
            loose_reference = git_directory / reference
            if loose_reference.is_file():
                actual_revision = loose_reference.read_text(
                    encoding="utf-8"
                ).strip()
            else:
                packed_references = (git_directory / "packed-refs").read_text(
                    encoding="utf-8"
                ).splitlines()
                matches = [
                    line.split(" ", 1)[0]
                    for line in packed_references
                    if line.endswith(f" {reference}")
                ]
                if len(matches) != 1:
                    raise ValueError(f"unable to resolve {reference} in {source}")
                actual_revision = matches[0]
        else:
            actual_revision = head
        if actual_revision != expected_revision:
            raise ValueError(
                f"source revision mismatch for {source}: "
                f"expected {expected_revision}, found {actual_revision}"
            )


def materialize(source: Path, destination: Path, asset: OfficialAsset) -> str:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    method = "hardlink"
    try:
        try:
            os.link(source, temporary)
        except OSError:
            method = "copy"
            shutil.copyfile(source, temporary)
        if method == "copy":
            verify_asset(temporary, asset)
        elif temporary.stat().st_size != asset.size_bytes:
            raise ValueError(f"size mismatch after hardlinking {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return method


def download_official(url: str, destination: Path, asset: OfficialAsset) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file, length=16 * 1024 * 1024)
        verify_asset(temporary, asset)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_track4world_assets(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    da3_directory: Path = DEFAULT_DA3_DIRECTORY,
    torch_hub: Path = DEFAULT_TORCH_HUB,
    cache_dir: Path | None = None,
    source_file: Path | None = None,
    convnext_source_file: Path | None = None,
    verify_only: bool = False,
) -> dict[str, object]:
    verify_source_revisions()
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        verify_asset(checkpoint, TRACK4WORLD_CHECKPOINT)
        checkpoint_method = "existing"
    elif verify_only:
        raise FileNotFoundError(checkpoint)
    else:
        if source_file is None:
            source = Path(
                hf_hub_download(
                    repo_id=TRACK4WORLD_REPO,
                    filename=TRACK4WORLD_CHECKPOINT.filename,
                    revision=TRACK4WORLD_REVISION,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                )
            ).resolve()
        else:
            source = source_file.expanduser().resolve()
        verify_asset(source, TRACK4WORLD_CHECKPOINT)
        checkpoint_method = materialize(
            source, checkpoint, TRACK4WORLD_CHECKPOINT
        )

    da3_cache_snapshot = Path(
        snapshot_download(
            repo_id=DA3_REPO,
            revision=DA3_REVISION,
            allow_patterns=[asset.filename for asset in DA3_ASSETS],
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=verify_only,
        )
    ).resolve()
    da3_directory = da3_directory.expanduser().resolve()
    da3_materialization: dict[str, str] = {}
    for asset in DA3_ASSETS:
        cached_asset = da3_cache_snapshot / asset.filename
        verify_asset(cached_asset, asset)
        destination = da3_directory / asset.filename
        if destination.is_file():
            verify_asset(destination, asset)
            da3_materialization[asset.filename] = "existing"
        elif verify_only:
            raise FileNotFoundError(destination)
        else:
            da3_materialization[asset.filename] = materialize(
                cached_asset, destination, asset
            )

    convnext_destination = (
        torch_hub.expanduser().resolve()
        / "checkpoints"
        / CONVNEXT_CHECKPOINT.filename
    )
    if convnext_destination.is_file():
        verify_asset(convnext_destination, CONVNEXT_CHECKPOINT)
        convnext_method = "existing"
    elif verify_only:
        raise FileNotFoundError(convnext_destination)
    elif convnext_source_file is not None:
        verify_asset(convnext_source_file.expanduser().resolve(), CONVNEXT_CHECKPOINT)
        convnext_method = materialize(
            convnext_source_file,
            convnext_destination,
            CONVNEXT_CHECKPOINT,
        )
    else:
        download_official(
            CONVNEXT_URL,
            convnext_destination,
            CONVNEXT_CHECKPOINT,
        )
        convnext_method = "download"

    return {
        "sources": SOURCE_REVISIONS,
        "track4world": {
            "repo_id": TRACK4WORLD_REPO,
            "revision": TRACK4WORLD_REVISION,
            "checkpoint": str(checkpoint),
            "size_bytes": TRACK4WORLD_CHECKPOINT.size_bytes,
            "sha256": TRACK4WORLD_CHECKPOINT.sha256,
            "materialization": checkpoint_method,
        },
        "depth_anything_3": {
            "repo_id": DA3_REPO,
            "revision": DA3_REVISION,
            "cache_snapshot": str(da3_cache_snapshot),
            "runtime_directory": str(da3_directory),
            "materialization": da3_materialization,
            "assets": [
                {
                    "filename": asset.filename,
                    "size_bytes": asset.size_bytes,
                    "sha256": asset.sha256,
                }
                for asset in DA3_ASSETS
            ],
        },
        "convnext_tiny": {
            "url": CONVNEXT_URL,
            "checkpoint": str(convnext_destination),
            "size_bytes": CONVNEXT_CHECKPOINT.size_bytes,
            "sha256": CONVNEXT_CHECKPOINT.sha256,
            "materialization": convnext_method,
        },
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            prepare_track4world_assets(
                checkpoint=args.checkpoint,
                da3_directory=args.da3_directory,
                torch_hub=args.torch_hub,
                cache_dir=args.cache_dir,
                source_file=args.source_file,
                convnext_source_file=args.convnext_source_file,
                verify_only=args.verify_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
