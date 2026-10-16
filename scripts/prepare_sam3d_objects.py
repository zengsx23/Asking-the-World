#!/usr/bin/env python3
"""Install and verify the pinned official SAM 3D Objects inference assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "data/models/sam-3d-objects"
REPOSITORY = "facebook/sam-3d-objects"
REVISION = "05929e2a63f234014031f9941f4aabefea5f382e"


@dataclass(frozen=True, slots=True)
class OfficialAsset:
    filename: str
    size_bytes: int
    sha256: str


ASSETS = (
    OfficialAsset("pipeline.yaml", 3_548, "53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c"),
    OfficialAsset("ss_generator.yaml", 5_076, "3c265448bca7c057f94e3ef56adea3a895a10bcd9f15f992a41dd03fa35412cd"),
    OfficialAsset("ss_generator.ckpt", 6_690_136_964, "225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146"),
    OfficialAsset("slat_generator.yaml", 1_986, "53029fadff6fe34a0344381a16d64d65d0567d603484952712e8969319559c4e"),
    OfficialAsset("slat_generator.ckpt", 4_906_537_684, "91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb"),
    OfficialAsset("ss_decoder.yaml", 244, "baacff269b664f84f7aa1896ebd66128065f6568cdb88d419d5c3d2ccb4193ae"),
    OfficialAsset("ss_decoder.ckpt", 147_609_242, "6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340"),
    OfficialAsset("slat_decoder_gs.yaml", 576, "53f054e02a0c185f0a6885d30bb6ca0ec92efe0a98148688e7f05f7c26afe670"),
    OfficialAsset("slat_decoder_gs.ckpt", 171_476_155, "f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111"),
    OfficialAsset("slat_decoder_gs_4.yaml", 575, "3d1dfd4c56cdac56f30e0cf5eb310d4df973fe25c60ac0a462c86ca4e3bf8b48"),
    OfficialAsset("slat_decoder_gs_4.ckpt", 170_269_801, "731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186"),
    OfficialAsset("slat_decoder_mesh.yaml", 300, "8f46952764aa985c50109a56f0b4f07625cdb06457aaded74957d26f3520c69e"),
    OfficialAsset("slat_decoder_mesh.ckpt", 363_726_862, "85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        help="optional verified local directory containing the official files",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FileIdentity = tuple[int, int, int, int]


def verify(
    path: Path,
    asset: OfficialAsset,
    digest_cache: dict[FileIdentity, str],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if stat.st_size != asset.size_bytes:
        raise ValueError(
            f"size mismatch for {path}: expected {asset.size_bytes}, found {stat.st_size}"
        )
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    digest = digest_cache.get(identity)
    if digest is None:
        print(f"Verifying {asset.filename}...", file=sys.stderr, flush=True)
        digest = sha256_file(path)
        digest_cache[identity] = digest
    if digest != asset.sha256:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {asset.sha256}, found {digest}"
        )


def install_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source.resolve(), temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(target)


def reuse_candidate(directory: Path, filename: str) -> Path | None:
    candidates = (directory / filename, directory / "checkpoints" / filename)
    return next((path for path in candidates if path.is_file()), None)


def prepare(
    *,
    target: Path,
    cache_dir: Path | None,
    reuse_dir: Path | None,
    verify_only: bool,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    digest_cache: dict[FileIdentity, str] = {}
    if reuse_dir is not None and not verify_only:
        for asset in ASSETS:
            destination = target / asset.filename
            if destination.is_file() and destination.stat().st_size == asset.size_bytes:
                continue
            source = reuse_candidate(reuse_dir, asset.filename)
            if source is None:
                continue
            verify(source, asset, digest_cache)
            install_file(source, destination)

    missing = [asset for asset in ASSETS if not (target / asset.filename).is_file()]
    if missing and verify_only:
        raise FileNotFoundError(
            "missing SAM3D assets: " + ", ".join(asset.filename for asset in missing)
        )
    if missing:
        snapshot = Path(
            snapshot_download(
                repo_id=REPOSITORY,
                revision=REVISION,
                allow_patterns=[f"checkpoints/{asset.filename}" for asset in ASSETS],
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
        )
        for asset in missing:
            source = snapshot / "checkpoints" / asset.filename
            verify(source, asset, digest_cache)
            install_file(source, target / asset.filename)

    for asset in ASSETS:
        verify(target / asset.filename, asset, digest_cache)
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "assets": [
            {
                "filename": asset.filename,
                "size_bytes": asset.size_bytes,
                "sha256": asset.sha256,
            }
            for asset in ASSETS
        ],
    }
    (target / "assets.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    prepare(
        target=args.target.resolve(),
        cache_dir=args.cache_dir.resolve() if args.cache_dir is not None else None,
        reuse_dir=args.reuse_dir.resolve() if args.reuse_dir is not None else None,
        verify_only=args.verify_only,
    )
    print(json.dumps({"target": str(args.target.resolve()), "status": "verified"}))


if __name__ == "__main__":
    main()
