#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fetch pinned release inputs and reconstruct derived assets, never new models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from release_platform import contract_paths, native_target

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = False


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url: str, path: Path, checksum: str | None = None, size: int | None = None) -> Path:
    if path.exists():
        if checksum and digest(path) != checksum:
            raise ValueError(f"existing input differs from lock: {path}")
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"existing input size differs from lock: {path}")
        return path
    if OFFLINE:
        raise FileNotFoundError(f"offline build input missing: {path}")
    if urlsplit(url).scheme != "https":
        raise ValueError("release inputs require HTTPS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            request = Request(url, headers={"User-Agent": "Tracefolio-release-inputs/1"})
            with urlopen(request, timeout=90) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            if checksum and digest(temp) != checksum:
                raise ValueError(f"download checksum mismatch: {path.name}")
            if size is not None and temp.stat().st_size != size:
                raise ValueError(f"download size mismatch: {path.name}")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return path


def models() -> None:
    # URLs remain provenance. Model construction never fetches weights or
    # rebuilds dictionaries from upstream data.
    spec = json.loads((ROOT / "packaging/manifests/models.json").read_text())
    for asset in spec["assets"]:
        target = ROOT / asset["development_path"]
        if not target.is_file():
            raise FileNotFoundError(
                f"retained local model missing: {target}; restore the pinned model-assets archive"
            )
        if digest(target) != asset["sha256"] or target.stat().st_size != asset["bytes"]:
            raise ValueError(f"retained local model differs from lock: {target}")


def source_inputs() -> None:
    locks = ROOT / "packaging/locks"
    target = native_target()
    contracts = contract_paths(ROOT, target)
    standalone = json.loads(contracts["python_lock"].read_text())
    for entry in standalone["archives"].values():
        fetch(
            entry["url"],
            ROOT / "build/sources" / entry["filename"],
            entry["sha256"],
            entry["bytes"],
        )
    source_locks = sorted(locks.glob("*-source.json"))
    if target == "linux-arm64":
        native = json.loads(contracts["native_manifest"].read_text())
        required = {component["source_lock"] for component in native["components"]}
        required.update(
            {
                "ffmpeg-source.json",
                "x264-source.json",
                "zlib-source.json",
                "llama-source.json",
            }
        )
        source_locks = [locks / name for name in sorted(required)]
    for lock in source_locks:
        entry = json.loads(lock.read_text())
        url = entry["url"]
        if entry["component"] == "onnxruntime-sherpa":
            url = json.loads((locks / "source-mirrors.json").read_text())["sources"][0]["url"]
        fetch(
            url,
            ROOT
            / "build/sources"
            / entry.get("filename", unquote(urlsplit(entry["url"]).path.rsplit("/", 1)[-1])),
            entry["sha256"],
            entry["bytes"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    OFFLINE = args.offline
    source_inputs()
    if not args.sources_only:
        models()
    print(
        json.dumps(
            {
                "sources": "verified",
                "models": "skipped" if args.sources_only else "verified",
            }
        )
    )
