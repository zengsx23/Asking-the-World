#!/usr/bin/env python3
"""Download, verify, extract, and summarize the official ContPhy dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REPOSITORY = "zzcnewly/ContPhy_Dataset"
REVISION = "7c1f624ef2aa2676c456323310e3636e08dccf03"
CHUNK_SIZE = 16 * 1024 * 1024
MARKER_NAME = ".contphy-extract.json"


@dataclass(frozen=True)
class Asset:
    filename: str
    size: int
    sha256: str
    variant: str
    extract_relative: Path

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
            f"{REVISION}/{self.filename}?download=true"
        )


ASSETS = (
    Asset(
        filename="contphy_mini.zip",
        size=440_826_887,
        sha256="780d2fc6b5e85281631773f72ed2d7ccb15b96b53dc8be7af14974c2be63cc3e",
        variant="mini",
        extract_relative=Path("mini"),
    ),
    Asset(
        filename="fluid_full.zip",
        size=2_254_502_143,
        sha256="61e6aa8562864719da991a5c47e80359103ec1667d4a46b0f4f64e8fd952b0ab",
        variant="full",
        extract_relative=Path("full/fluid"),
    ),
    Asset(
        filename="rope_full.zip",
        size=1_270_357_423,
        sha256="562a4ba900f9168b65f5fffab2eba59b9d67ef0e07c70c4bfb9d034649e26054",
        variant="full",
        extract_relative=Path("full/rope"),
    ),
    Asset(
        filename="cloth_full.zip",
        size=5_697_486_939,
        sha256="005bdd7485f0f9272c81a392ac00d108565712e46ba4b6be635c12b1015cd99b",
        variant="full",
        extract_relative=Path("full/cloth"),
    ),
    Asset(
        filename="ball_full.zip",
        size=1_804_421_895,
        sha256="b891207eccd2cf97601889a556a9b3b3990788d765d7525e30c64484e0f6d50b",
        variant="full",
        extract_relative=Path("full/ball"),
    ),
)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, asset: Asset) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive: {path}")
    actual_size = path.stat().st_size
    if actual_size != asset.size:
        raise ValueError(
            f"Size mismatch for {path}: expected {asset.size}, got {actual_size}"
        )
    print(f"Verifying {path.name} ({format_bytes(actual_size)}) ...", flush=True)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {asset.sha256}, "
            f"got {actual_sha256}"
        )
    print(f"Verified {path.name}: {actual_sha256}", flush=True)


def download_archive(path: Path, asset: Asset, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            verify_archive(path, asset)
            return
        except (OSError, ValueError):
            if not force:
                raise
            path.unlink()

    partial = path.with_name(f"{path.name}.part")
    if force and partial.exists():
        partial.unlink()
    if partial.exists() and partial.stat().st_size > asset.size:
        raise ValueError(f"Partial file is larger than expected: {partial}")
    if partial.exists() and partial.stat().st_size == asset.size:
        verify_archive(partial, asset)
        partial.replace(path)
        return

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Physmind-v2 ContPhy downloader"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(
            f"Resuming {asset.filename} at {format_bytes(offset)} / "
            f"{format_bytes(asset.size)}",
            flush=True,
        )
    else:
        print(f"Downloading {asset.filename} ({format_bytes(asset.size)})", flush=True)

    request = urllib.request.Request(asset.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code == 416 and partial.exists():
            verify_archive(partial, asset)
            partial.replace(path)
            return
        raise

    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        print("Server ignored the range request; restarting the download.", flush=True)
        offset = 0
        mode = "wb"
    else:
        mode = "ab" if offset else "wb"

    downloaded = offset
    last_report = time.monotonic()
    with response, partial.open(mode) as handle:
        while chunk := response.read(CHUNK_SIZE):
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report >= 5:
                percent = downloaded * 100 / asset.size
                print(
                    f"  {asset.filename}: {format_bytes(downloaded)} / "
                    f"{format_bytes(asset.size)} ({percent:.1f}%)",
                    flush=True,
                )
                last_report = now
        handle.flush()
        os.fsync(handle.fileno())

    if downloaded != asset.size:
        raise ValueError(
            f"Incomplete download for {asset.filename}: expected {asset.size}, "
            f"got {downloaded}. Rerun the command to resume."
        )
    verify_archive(partial, asset)
    partial.replace(path)


def marker_matches(destination: Path, asset: Asset) -> bool:
    marker = destination / MARKER_NAME
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == {
        "archive": asset.filename,
        "repository": REPOSITORY,
        "revision": REVISION,
        "sha256": asset.sha256,
    }


def safe_destination(root: Path, member: zipfile.ZipInfo) -> Path:
    member_path = PurePosixPath(member.filename)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe ZIP member path: {member.filename!r}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"Symbolic-link ZIP member is not allowed: {member.filename!r}")
    destination = root.joinpath(*member_path.parts)
    root_resolved = root.resolve()
    destination_resolved = destination.resolve()
    if not destination_resolved.is_relative_to(root_resolved):
        raise ValueError(f"ZIP member escapes destination: {member.filename!r}")
    return destination


def extract_archive(
    archive: Path,
    destination: Path,
    asset: Asset,
    force: bool,
) -> None:
    if destination.exists() and marker_matches(destination, asset) and not force:
        print(f"Extraction already verified: {destination}", flush=True)
        return
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"Extraction destination exists without a matching marker: "
                f"{destination}. Use --force-extract to replace it."
            )
        shutil.rmtree(destination)

    temporary = destination.with_name(f".{destination.name}.extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    print(f"Extracting {archive.name} to {destination} ...", flush=True)

    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for index, member in enumerate(members, start=1):
                output = safe_destination(temporary, member)
                if member.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target, length=CHUNK_SIZE)
                if index % 500 == 0 or index == len(members):
                    print(f"  extracted {index}/{len(members)} entries", flush=True)

        marker = {
            "archive": asset.filename,
            "repository": REPOSITORY,
            "revision": REVISION,
            "sha256": asset.sha256,
        }
        (temporary / MARKER_NAME).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def summarize_tree(path: Path) -> None:
    counts: Counter[str] = Counter()
    total_size = 0
    files = 0
    directories = 0
    top_level = []
    if not path.exists():
        print(f"Missing extracted path: {path}")
        return
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if child.name != MARKER_NAME:
            top_level.append(child.name)
    for item in path.rglob("*"):
        if item.is_dir():
            directories += 1
        elif item.is_file():
            files += 1
            total_size += item.stat().st_size
            counts[item.suffix.lower() or "<no suffix>"] += 1
    common_types = ", ".join(
        f"{suffix}={count}" for suffix, count in counts.most_common(12)
    )
    print(f"Summary for {path}:")
    print(f"  top-level: {top_level}")
    print(f"  directories: {directories}")
    print(f"  files: {files}")
    print(f"  extracted size: {format_bytes(total_size)}")
    print(f"  common file types: {common_types}")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("mini", "full", "all"),
        default="all",
        help="Dataset release to prepare (default: all).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repository_root / "data" / "contphy",
        help="Dataset root (default: <repository>/data/contphy).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify archives without extracting them.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify archives already on disk without downloading or extracting.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an invalid archive or partial download.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Replace an existing extraction that lacks a matching marker.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    selected = [
        asset
        for asset in ASSETS
        if args.variant == "all" or asset.variant == args.variant
    ]
    archives = root / "archives"

    try:
        for asset in selected:
            archive = archives / asset.filename
            if args.verify_only:
                verify_archive(archive, asset)
            else:
                download_archive(archive, asset, force=args.force_download)

        if args.verify_only or args.download_only:
            return 0

        for asset in selected:
            archive = archives / asset.filename
            destination = root / asset.extract_relative
            extract_archive(
                archive,
                destination,
                asset,
                force=args.force_extract,
            )
            summarize_tree(destination)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
