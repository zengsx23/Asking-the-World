#!/usr/bin/env python3
"""Download and verify the pinned official SAM3 inference assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SOURCE_COMMIT = "c3a42ff67487eb489f74dce7a747477324734e44"


@dataclass(frozen=True, slots=True)
class OfficialAsset:
    filename: str
    size_bytes: int
    sha256: str


OFFICIAL_ASSETS = (
    OfficialAsset(
        filename="config.json",
        size_bytes=25_843,
        sha256="4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
    ),
    OfficialAsset(
        filename="sam3.pt",
        size_bytes=3_450_062_241,
        sha256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face Hub cache directory; defaults to HF_HUB_CACHE",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require pinned assets to exist locally and verify them without downloading",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, asset: OfficialAsset) -> None:
    actual_size = path.stat().st_size
    if actual_size != asset.size_bytes:
        raise ValueError(
            f"size mismatch for {asset.filename}: expected {asset.size_bytes}, "
            f"found {actual_size} at {path}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        raise ValueError(
            f"SHA256 mismatch for {asset.filename}: expected {asset.sha256}, "
            f"found {actual_sha256} at {path}"
        )


def prepare_asset(
    asset: OfficialAsset,
    *,
    cache_dir: Path | None,
    verify_only: bool,
) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=asset.filename,
            revision=REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=verify_only,
        )
    )
    verify_asset(path, asset)
    return path


def prepare_sam3_assets(
    *,
    cache_dir: Path | None = None,
    verify_only: bool = False,
) -> dict[str, Path]:
    """Resolve official assets through the Hub cache and verify their bytes."""

    return {
        asset.filename: prepare_asset(
            asset,
            cache_dir=cache_dir,
            verify_only=verify_only,
        )
        for asset in OFFICIAL_ASSETS
    }


def main() -> None:
    args = parse_args()
    paths = prepare_sam3_assets(
        cache_dir=args.cache_dir,
        verify_only=args.verify_only,
    )
    print(
        json.dumps(
            {
                "source_commit": SOURCE_COMMIT,
                "repo_id": REPO_ID,
                "revision": REVISION,
                "assets": {
                    asset.filename: {
                        "path": str(paths[asset.filename]),
                        "size_bytes": asset.size_bytes,
                        "sha256": asset.sha256,
                    }
                    for asset in OFFICIAL_ASSETS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
