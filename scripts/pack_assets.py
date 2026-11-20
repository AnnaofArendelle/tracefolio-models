#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Ship runtime source, corresponding native source, licenses and checksums."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def package(runtime: Path, destination: Path):
    target = json.loads((runtime / "runtime.json").read_text())["target"]
    suffix = "-arm64" if target.endswith("arm64") else ""
    inventory = json.loads(
        (ROOT / f"packaging/manifests/native-components{suffix}.json").read_text()
    )
    with tempfile.TemporaryDirectory(prefix="pack-source-") as temporary:
        stage = Path(temporary)
        source = stage / "source"
        source.mkdir()
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                "--output",
                str(stage / "checkout.tar"),
                "HEAD",
            ],
            cwd=ROOT,
            check=True,
        )
        with tarfile.open(stage / "checkout.tar") as archive:
            archive.extractall(source / "tracefolio-models", filter="data")
        required = {c["source_lock"] for c in inventory["components"]} | {
            "ffmpeg-source.json",
            "x264-source.json",
            "zlib-source.json",
            "llama-source.json",
        }
        sources = []
        for name in sorted(required):
            lock = json.loads((ROOT / "packaging/locks" / name).read_text())
            filename = lock.get("filename", lock["url"].rsplit("/", 1)[-1])
            path = ROOT / "build/sources" / filename
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != lock["sha256"] or path.stat().st_size != lock["bytes"]:
                raise ValueError("corresponding source differs: " + filename)
            (source / "native").mkdir(exist_ok=True)
            shutil.copy2(path, source / "native" / filename)
            sources.append({**lock, "retained_filename": filename})
        # Preserve Python source and all license metadata from the actual runtime.
        site = runtime / "python/lib/python3.12/site-packages"
        notices = stage / "licenses"
        shutil.copytree(runtime / "licenses", notices)
        for distribution in site.glob("*.dist-info"):
            location = notices / "python" / distribution.name
            location.mkdir(parents=True, exist_ok=True)
            shutil.copy2(distribution / "METADATA", location / "METADATA")
            for entry in distribution.rglob("*"):
                if entry.is_file() and entry.name.upper().startswith(
                    ("LICENSE", "LICENCE", "COPYING", "NOTICE")
                ):
                    dest = location / entry.relative_to(distribution)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry, dest)
        for path in site.rglob("*"):
            if path.is_file() and path.suffix in {
                ".py",
                ".json",
                ".txt",
                ".h",
                ".c",
                ".cpp",
            }:
                dest = source / "python" / path.relative_to(site)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        shutil.copy2(ROOT / "LICENSE", notices / "Tracefolio-AGPL-3.0.txt")
        shutil.copy2(ROOT / "NOTICE", notices / "NOTICE")
        for kind, path in [("sources", source), ("licenses", notices)]:
            with tarfile.open(
                destination / f"tracefolio-models-{kind}-{target}.tar.gz", "w:gz"
            ) as archive:
                archive.add(path, arcname=kind)
        (destination / f"tracefolio-models-provenance-{target}.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "source_revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "runtime_manifest": json.loads(
                        (runtime / "runtime.json").read_text()
                    ),
                    "native_sources": sources,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    shutil.copy2(
        ROOT / "build/reports" / f"models-{target}.json",
        destination / f"tracefolio-models-acceptance-{target}.json",
    )
    lines = []
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            lines.append(checksum + "  " + path.name)
    (destination / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    package(args.runtime, args.directory)
