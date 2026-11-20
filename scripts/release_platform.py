#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Select the native Linux release contract; cross-architecture staging is denied."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import struct


def native_target(system: str | None = None, machine: str | None = None) -> str:
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    architectures = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    if system.lower() != "linux" or machine.lower() not in architectures:
        raise ValueError("complete releases require native Linux amd64 or arm64")
    return "linux-" + architectures[machine.lower()]


def contract_paths(root: Path, target: str | None = None) -> dict[str, Path]:
    target = native_target() if target is None else target
    if target not in {"linux-amd64", "linux-arm64"}:
        raise ValueError("unsupported release target")
    suffix = "-arm64" if target == "linux-arm64" else ""
    return {
        "python_lock": root / f"packaging/locks/python-standalone{suffix}.json",
        "python_metadata": root / f"packaging/manifests/python-build{suffix}.json",
        "python_licenses": root / f"packaging/licenses/python-standalone{suffix}",
        "native_manifest": root / f"packaging/manifests/native-components{suffix}.json",
    }


def validate_elf_header(header: bytes, target: str) -> None:
    """Reject foreign native members before an otherwise valid bundle is built."""
    if not header.startswith(b"\x7fELF"):
        return
    if target not in {"linux-amd64", "linux-arm64"}:
        raise ValueError("unsupported ELF release target")
    if len(header) < 20 or header[4:6] != b"\x02\x01":
        raise ValueError("release native files require little-endian ELF64")
    if struct.unpack_from("<H", header, 18)[0] != {"linux-amd64": 62, "linux-arm64": 183}[target]:
        raise ValueError("native ELF member differs from the release architecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target")
    args = parser.parse_args()
    target = native_target()
    if args.target is not None and args.target != target:
        raise SystemExit("release target differs from the native build host")
    print(target)
