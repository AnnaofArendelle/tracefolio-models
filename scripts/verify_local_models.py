#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Check retained weights without opening a network connection."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(directory: Path) -> dict:
    manifest = json.loads((ROOT / "packaging/manifests/models.json").read_text())
    total = 0
    for asset in manifest["assets"]:
        path = directory / asset["path"]
        if not path.is_file():
            raise FileNotFoundError(f"retained local model missing: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset["sha256"] or path.stat().st_size != asset["bytes"]:
            raise ValueError(f"retained local model differs from lock: {path}")
        total += asset["bytes"]
    return {
        "verified": True,
        "models": len(manifest["assets"]),
        "bytes": total,
        "network_used": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory)))
