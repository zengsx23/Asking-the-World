#!/usr/bin/env python3
"""Download, verify, extract, and validate the official CLEVRER validation set."""

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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


OFFICIAL_PROJECT_PAGE = "https://clevrer.csail.mit.edu/"
CHUNK_SIZE = 16 * 1024 * 1024
SCENE_START = 10_000
SCENE_END = 15_000
BUCKET_SIZE = 1_000


@dataclass(frozen=True)
class Asset:
    filename: str
    url: str
    size: int
    sha256: str


VIDEO_ARCHIVE = Asset(
    filename="video_validation.zip",
    url=(
        "https://data.csail.mit.edu/clevrer/videos/validation/"
        "video_validation.zip"
    ),
    size=6_206_059_321,
    sha256="07f6d48bd708c80428684fcf17a7874f6c01fbb82a59b2f79ede199a63db7aae",
)
QUESTIONS = Asset(
    filename="validation.json",
    url="https://data.csail.mit.edu/clevrer/questions/validation.json",
    size=43_327_533,
    sha256="fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d",
)
ASSETS = (VIDEO_ARCHIVE, QUESTIONS)


class DatasetValidationError(ValueError):
    """Raised when downloaded or extracted CLEVRER data is inconsistent."""


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


def verify_asset(path: Path, asset: Asset) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official asset: {path}")
    actual_size = path.stat().st_size
    if actual_size != asset.size:
        raise DatasetValidationError(
            f"Size mismatch for {path}: expected {asset.size}, got {actual_size}"
        )
    print(f"Verifying {path.name} ({format_bytes(actual_size)}) ...", flush=True)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        raise DatasetValidationError(
            f"SHA256 mismatch for {path}: expected {asset.sha256}, "
            f"got {actual_sha256}"
        )
    print(f"Verified {path.name}: {actual_sha256}", flush=True)


def download_asset(path: Path, asset: Asset, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            verify_asset(path, asset)
            return
        except (OSError, DatasetValidationError):
            if not force:
                raise
            path.unlink()

    partial = path.with_name(f"{path.name}.part")
    if force and partial.exists():
        partial.unlink()
    if partial.exists() and not partial.is_file():
        raise DatasetValidationError(f"Partial download is not a file: {partial}")
    if partial.exists() and partial.stat().st_size > asset.size:
        raise DatasetValidationError(f"Partial file is larger than expected: {partial}")
    if partial.exists() and partial.stat().st_size == asset.size:
        verify_asset(partial, asset)
        partial.replace(path)
        return

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Physmind-v2 CLEVRER downloader"}
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
            verify_asset(partial, asset)
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
        raise DatasetValidationError(
            f"Incomplete download for {asset.filename}: expected {asset.size}, "
            f"got {downloaded}. Rerun the command to resume."
        )
    verify_asset(partial, asset)
    partial.replace(path)


def bucket_name(scene_index: int) -> str:
    bucket_start = scene_index - scene_index % BUCKET_SIZE
    return f"video_{bucket_start:05d}-{bucket_start + BUCKET_SIZE:05d}"


def expected_video_paths() -> set[str]:
    return {
        f"{bucket_name(index)}/video_{index:05d}.mp4"
        for index in range(SCENE_START, SCENE_END)
    }


def safe_destination(root: Path, member: zipfile.ZipInfo) -> Path:
    member_path = PurePosixPath(member.filename)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise DatasetValidationError(
            f"Unsafe ZIP member path: {member.filename!r}"
        )
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise DatasetValidationError(
            f"Symbolic-link ZIP member is not allowed: {member.filename!r}"
        )
    destination = root.joinpath(*member_path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise DatasetValidationError(
            f"ZIP member escapes extraction root: {member.filename!r}"
        )
    return destination


def validate_archive_layout(bundle: zipfile.ZipFile) -> None:
    expected = expected_video_paths()
    actual: set[str] = set()
    for member in bundle.infolist():
        safe_destination(Path.cwd(), member)
        if member.is_dir():
            continue
        normalized = PurePosixPath(member.filename).as_posix()
        if normalized in actual:
            raise DatasetValidationError(f"Duplicate ZIP member: {normalized}")
        actual.add(normalized)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise DatasetValidationError(
            "Unexpected video archive layout: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


def validate_video_tree(root: Path) -> None:
    expected = expected_video_paths()
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.mp4")
        if path.is_file()
    }
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise DatasetValidationError(
            "CLEVRER video tree is incomplete or contains unexpected MP4 files: "
            f"missing={len(missing)} {sorted(missing)[:3]}, "
            f"unexpected={len(unexpected)} {sorted(unexpected)[:3]}"
        )
    empty = [relative for relative in expected if (root / relative).stat().st_size == 0]
    if empty:
        raise DatasetValidationError(
            f"CLEVRER video tree contains {len(empty)} empty files: {sorted(empty)[:3]}"
        )


def validate_questions(path: Path) -> None:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(records, list):
        raise DatasetValidationError(f"Expected a JSON list in {path}")
    if len(records) != SCENE_END - SCENE_START:
        raise DatasetValidationError(
            f"Expected {SCENE_END - SCENE_START} validation records, got {len(records)}"
        )

    seen: set[int] = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetValidationError(f"Record {position} is not a JSON object")
        scene_index = record.get("scene_index")
        if not isinstance(scene_index, int):
            raise DatasetValidationError(
                f"Record {position} has invalid scene_index: {scene_index!r}"
            )
        expected_filename = f"video_{scene_index:05d}.mp4"
        if record.get("video_filename") != expected_filename:
            raise DatasetValidationError(
                f"Record {position} filename does not match scene {scene_index}"
            )
        if not isinstance(record.get("questions"), list):
            raise DatasetValidationError(
                f"Record {position} has no valid questions list"
            )
        if scene_index in seen:
            raise DatasetValidationError(f"Duplicate scene_index: {scene_index}")
        seen.add(scene_index)

    expected_indices = set(range(SCENE_START, SCENE_END))
    if seen != expected_indices:
        missing = sorted(expected_indices - seen)[:3]
        unexpected = sorted(seen - expected_indices)[:3]
        raise DatasetValidationError(
            "Question scene indices do not match the validation split: "
            f"missing={missing}, unexpected={unexpected}"
        )


def extract_videos(archive: Path, root: Path, force: bool) -> None:
    try:
        validate_video_tree(root)
    except DatasetValidationError:
        pass
    else:
        if not force:
            print(f"Extracted videos already verified: {root}", flush=True)
            return

    video_directories = [
        root / f"video_{start:05d}-{start + BUCKET_SIZE:05d}"
        for start in range(SCENE_START, SCENE_END, BUCKET_SIZE)
    ]
    existing = [path for path in video_directories if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Incomplete CLEVRER extraction already exists. Use --force-extract "
            "to replace the five validation video directories."
        )
    if force:
        for path in existing:
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)

    temporary = root / ".video_validation.extracting"
    if temporary.exists():
        if not force:
            raise FileExistsError(
                f"Incomplete extraction workspace exists: {temporary}. "
                "Use --force-extract to replace it."
            )
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    print(f"Extracting {archive.name} into {root} ...", flush=True)

    try:
        with zipfile.ZipFile(archive) as bundle:
            validate_archive_layout(bundle)
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

        validate_video_tree(temporary)
        for path in video_directories:
            (temporary / path.name).replace(path)
        temporary.rmdir()
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def print_summary(root: Path) -> None:
    video_count = sum(1 for _ in root.rglob("*.mp4"))
    video_bytes = sum(path.stat().st_size for path in root.rglob("*.mp4"))
    print("CLEVRER validation set is ready:")
    print(f"  official project: {OFFICIAL_PROJECT_PAGE}")
    print(f"  root: {root}")
    print(f"  videos: {video_count}")
    print(f"  extracted video size: {format_bytes(video_bytes)}")
    print(f"  questions: {root / QUESTIONS.filename}")
    print(f"  retained archive: {root / VIDEO_ARCHIVE.filename}")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=repository_root / "data" / "clevrer",
        help="Dataset root (default: <repository>/data/clevrer).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify official files without extracting videos.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify downloaded files and extracted data without network access.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an invalid official file or partial download.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Replace incomplete validation video directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    try:
        if args.verify_only:
            for asset in ASSETS:
                verify_asset(root / asset.filename, asset)
            validate_questions(root / QUESTIONS.filename)
            validate_video_tree(root)
            print_summary(root)
            return 0

        for asset in ASSETS:
            download_asset(root / asset.filename, asset, force=args.force_download)
        validate_questions(root / QUESTIONS.filename)
        if args.download_only:
            return 0

        extract_videos(
            root / VIDEO_ARCHIVE.filename,
            root,
            force=args.force_extract,
        )
        validate_video_tree(root)
        print_summary(root)
    except (
        DatasetValidationError,
        FileExistsError,
        OSError,
        urllib.error.URLError,
        zipfile.BadZipFile,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
