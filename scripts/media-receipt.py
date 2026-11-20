#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bind portable media bytes to pinned sources and exact build configuration."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from elf_audit import tree
from release_platform import native_target

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def current() -> dict:
    binaries = ROOT / "build/media-runtime/bin"
    tree(binaries, native_target())
    checksums = {name: digest(binaries / name) for name in ("ffmpeg", "ffprobe")}
    inputs = {"scripts/build_media.sh": digest(ROOT / "scripts/build_media.sh")}
    for name in ("ffmpeg", "x264", "zlib"):
        lock_path = ROOT / "packaging/locks" / f"{name}-source.json"
        lock = json.loads(lock_path.read_text())
        source = ROOT / "build/sources" / lock["url"].rsplit("/", 1)[-1]
        if digest(source) != lock["sha256"]:
            raise ValueError("media source differs from lock")
        inputs[str(lock_path.relative_to(ROOT))] = digest(lock_path)
    config = subprocess.check_output(
        [str(binaries / "ffmpeg"), "-buildconf"], stderr=subprocess.STDOUT
    ).decode()
    return {
        "schema": 2,
        "minimum_glibc": "2.35",
        "target": native_target(),
        "inputs": inputs,
        "binaries": checksums,
        "buildconf_sha256": hashlib.sha256(config.encode()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "verify"))
    args = parser.parse_args()
    actual = current()
    receipt = ROOT / "build/media-runtime/build-receipt.json"
    if args.action == "write":
        receipt.write_text(json.dumps(actual, indent=2) + "\n")
    elif json.loads(receipt.read_text()) != actual:
        raise ValueError(
            "portable media receipt does not match source/configuration/binaries"
        )
    print(json.dumps({"media": "verified", "action": args.action}))
