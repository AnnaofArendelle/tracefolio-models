#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Audit packaged ELF architecture, required glibc versions and loader dependencies."""

from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path
from release_platform import native_target, validate_elf_header

BASELINE = (2, 35)
SYSTEM = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "libpthread.so.0",
    "librt.so.1",
    "libutil.so.1",
    "libresolv.so.2",
    "libgcc_s.so.1",
    "libstdc++.so.6",
    "libcrypt.so.1",
    "libz.so.1",
    "ld-linux-x86-64.so.2",
    "ld-linux-aarch64.so.1",
}


def audit(paths, target, root=None):
    paths = list(paths)
    elf = []
    for path in paths:
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    elf.append(path.resolve())
    bundled = {p.name for p in elf}
    root = root.resolve() if root is not None else None
    out = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            header = stream.read(20)
        if not header.startswith(b"\x7fELF"):
            continue
        validate_elf_header(header, target)
        symbols = subprocess.check_output(
            ["readelf", "--wide", "--dyn-syms", str(path)], text=True
        )
        required = sorted(
            set(
                re.findall(
                    r"@GLIBC_([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
                    "\n".join(row for row in symbols.splitlines() if " UND " in row),
                )
            )
        )
        if any(tuple(map(int, v.split("."))) > BASELINE for v in required):
            raise ValueError(f"{path}: requires glibc beyond 2.35: {required}")
        dynamic = subprocess.check_output(
            ["readelf", "--wide", "--dynamic", str(path)], text=True
        )
        needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
        expanded = []
        for name in needed:
            if "/" in name:
                candidate = Path(name.replace("$ORIGIN", str(path.parent))).resolve()
                if (
                    root is not None and not candidate.is_relative_to(root)
                ) or candidate not in elf:
                    raise ValueError(
                        f"{path}: absent relative loader dependency {name}"
                    )
                expanded.append(candidate.name)
            else:
                expanded.append(name)
        missing = set(expanded) - SYSTEM - bundled
        if missing:
            raise ValueError(
                f"{path}: unbundled native dependencies: {sorted(missing)}"
            )
        out.append({"path": str(path), "required_glibc": required, "needed": needed})
    return {"schema": 1, "target": target, "minimum_glibc": "2.35", "elf_files": out}


def tree(root, target):
    return audit(
        root.rglob("*") if root.is_dir() else [root],
        target,
        root if root.is_dir() else root.parent,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--target", default=native_target())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = tree(args.root, args.target)
    raw = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    print(
        json.dumps(
            {
                "target": args.target,
                "minimum_glibc": "2.35",
                "elf_files": len(receipt["elf_files"]),
            }
        )
    )
