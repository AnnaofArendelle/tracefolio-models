#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Restore retained model inputs from the project's own fixed archive."""

import hashlib
import json
from pathlib import Path
import tarfile
from verify_local_models import verify

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    lock = json.loads((ROOT / "packaging/locks/model-assets.json").read_text())
    archive = ROOT / "build/inputs" / lock["filename"]
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != lock["sha256"] or archive.stat().st_size != lock["bytes"]:
        raise ValueError("retained model archive differs from lock")
    destination = ROOT / "assets"
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    print(json.dumps(verify(destination)))
