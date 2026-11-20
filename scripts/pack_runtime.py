#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Create a deterministic runtime tar.gz from a verified staging generation."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def pack(source: Path, destination: Path):
    raw = (source / "runtime.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["total_bytes"] > 1536 * 1024**2:
        raise ValueError("expanded runtime exceeds budget")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    with temp.open("wb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=output, mtime=0
        ) as gz:
            with tarfile.open(
                fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT
            ) as archive:
                info = tarfile.TarInfo("runtime.json")
                info.size = len(raw)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(raw))
                for entry in sorted(manifest["files"], key=lambda x: x["path"]):
                    path = source / entry["path"]
                    if path.is_symlink():
                        raise ValueError("symlink in runtime staging")
                    with path.open("rb") as stream:
                        if (
                            hashlib.file_digest(stream, "sha256").hexdigest()
                            != entry["sha256"]
                        ):
                            raise ValueError("staged file changed: " + entry["path"])
                        if path.stat().st_size != entry["bytes"]:
                            raise ValueError("staged file size changed")
                        stream.seek(0)
                        info = tarfile.TarInfo(entry["path"])
                        info.size = entry["bytes"]
                        info.mode = 0o700 if entry["executable"] else 0o600
                        archive.addfile(info, stream)
    if temp.stat().st_size > 1024**3:
        raise ValueError("compressed runtime exceeds single-file budget")
    temp.replace(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": digest,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pack(args.source, args.destination)
