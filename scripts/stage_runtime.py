#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage a relocatable locked worker runtime without copying research venvs.

Run with workers/.venv/bin/python. Existing verified assets are hard-linked
where possible; never mutate staged files in place (links share source bytes).
The output is a build candidate, not an accepted end-user distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tarfile
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from elf_audit import tree as audit_elf_tree
from release_platform import contract_paths, native_target, validate_elf_header

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_distributions() -> list:
    project = tomllib.loads((ROOT / "workers/pyproject.toml").read_text())
    pending = [Requirement(s) for s in project["project"]["dependencies"]]
    selected = {}
    while pending:
        req = pending.pop()
        if req.marker and not req.marker.evaluate():
            continue
        key = canonicalize_name(req.name)
        dist = metadata.distribution(req.name)
        if dist.version not in req.specifier:
            raise RuntimeError(
                f"installed dependency violates lock constraint: {req.name}"
            )
        if key in selected:
            continue
        selected[key] = dist
        pending.extend(Requirement(s) for s in dist.requires or [])
    return [selected[key] for key in sorted(selected)]


def stage(
    destination: Path,
    include_models: bool,
    media: Path | None,
    description: Path | None = None,
) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("use the locked Python 3.12 worker environment")
    target_platform = native_target()
    contracts = contract_paths(ROOT, target_platform)
    if destination.exists():
        raise RuntimeError("destination exists; choose a new staging generation")
    destination.mkdir(parents=True)
    files = {}

    def install(source: Path, relative: Path | str, *, copy_file: bool = False) -> None:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe runtime path")
        source = source.resolve(strict=True)
        if not source.is_file():
            return
        with source.open("rb") as stream:
            validate_elf_header(stream.read(20), target_platform)
        key = relative.as_posix()
        checksum = digest(source)
        if key in files:
            if files[key]["sha256"] != checksum:
                raise RuntimeError(f"conflicting package files: {key}")
            return
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy_file:
            shutil.copy2(source, target)
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        files[key] = {
            "path": key,
            "sha256": checksum,
            "bytes": target.stat().st_size,
            "executable": bool(target.stat().st_mode & 0o111),
        }

    base = Path(sys.base_prefix)
    # uv patches _sysconfigdata with local paths. Package pristine upstream
    # bytes, not that installation mutation or regenerated .pyc files. Matching
    # local bytes can still be hardlinked to avoid duplicate staging space.
    python_lock = json.loads(contracts["python_lock"].read_text())
    archive_spec = python_lock["archives"]["install"]
    archive_path = ROOT / "build/sources" / archive_spec["filename"]
    if digest(archive_path) != archive_spec["sha256"]:
        raise RuntimeError("Python standalone archive does not match lock")
    omitted_stdlib = {
        "site-packages",
        "test",
        "ensurepip",
        "idlelib",
        "tkinter",
        "turtledemo",
    }
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            rel = Path(member.name).relative_to("python")
            if ".." in rel.parts or rel.is_absolute():
                raise ValueError("unsafe Python archive member")
            selected = rel.as_posix() == "bin/python3.12" or (
                rel.parent.as_posix() == "lib" and rel.name.startswith("libpython")
            )
            if rel.parts[:2] == ("lib", "python3.12"):
                selected = len(rel.parts) > 2 and rel.parts[2] not in omitted_stdlib
            if (
                not selected
                or "__pycache__" in rel.parts
                or rel.suffix == ".pyc"
                or rel.name.startswith("_tkinter")
            ):
                continue
            data = archive.extractfile(member).read()
            validate_elf_header(data[:20], target_platform)
            checksum = hashlib.sha256(data).hexdigest()
            local = base / rel
            key = (Path("python") / rel).as_posix()
            if local.is_file() and digest(local) == checksum:
                install(local, key)
            else:
                target = destination / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
                files[key] = {
                    "path": key,
                    "bytes": len(data),
                    "sha256": checksum,
                    "executable": bool(member.mode & 0o111),
                }
    for entry in python_lock["license_files"]:
        license_file = contracts["python_licenses"] / entry["path"]
        if digest(license_file) != entry["sha256"]:
            raise RuntimeError("Python bundled license changed")
        install(license_file, Path("licenses/python-standalone") / entry["path"])
    metadata_file = contracts["python_metadata"]
    if digest(metadata_file) != python_lock["metadata_sha256"]:
        raise RuntimeError("Python build metadata changed")
    install(metadata_file, "sources/python-build.json")
    install(
        contracts["python_lock"],
        "sources/python-standalone.json",
    )

    purelib = Path(sysconfig.get_path("purelib")).resolve()
    distributions = runtime_distributions()
    for dist in distributions:
        for entry in dist.files or []:
            source = Path(dist.locate_file(entry))
            if not source.is_file():
                continue
            try:
                rel = source.absolute().relative_to(purelib)
            except ValueError:
                continue  # console scripts are not the worker entry point
            if (
                ".." in rel.parts
                or "__pycache__" in rel.parts
                or source.suffix == ".pyc"
            ):
                continue
            # Library test suites are not imported by production adapters.
            if "tests" in rel.parts or "test" in rel.parts:
                continue
            if rel.parts[:2] == ("rapidocr", "models"):
                continue  # wheel defaults include the explicitly excluded small models
            if source.name == "direct_url.json":
                continue
            install(source, Path("python/lib/python3.12/site-packages") / rel)
    for path in (ROOT / "workers/src").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".json"}:
            continue
        install(
            path,
            Path("python/lib/python3.12/site-packages")
            / path.relative_to(ROOT / "workers/src"),
            copy_file=True,
        )
    for path in (ROOT / "packaging/licenses").iterdir():
        if path.is_file():
            install(path, Path("licenses") / path.name, copy_file=True)
    install(ROOT / "LICENSE", "licenses/Tracefolio-AGPL-3.0.txt", copy_file=True)
    install(ROOT / "NOTICE", "NOTICE", copy_file=True)
    install(
        contracts["native_manifest"],
        "sources/native-components.json",
        copy_file=True,
    )

    if media is not None:
        for name in ("ffmpeg", "ffprobe"):
            install(media / "bin" / name, Path("media/bin") / name)
        for notice in (media / "licenses").iterdir():
            if notice.is_file():
                install(notice, Path("licenses/media") / notice.name)
        for lock in (ROOT / "packaging/locks").glob("*-source.json"):
            install(lock, Path("sources") / lock.name, copy_file=True)

    if include_models:
        if description is None:
            raise RuntimeError(
                "complete model runtime requires the built-in description executable"
            )
        install(description / "bin/llama-server", "description/bin/llama-server")
        install(
            ROOT / "packaging/locks/llama-source.json",
            "sources/llama-source.json",
            copy_file=True,
        )
        spec = json.loads((ROOT / "packaging/manifests/models.json").read_text())
        spec["target"] = target_platform
        for asset in spec["assets"]:
            source = ROOT / asset["development_path"]
            if (
                digest(source) != asset["sha256"]
                or source.stat().st_size != asset["bytes"]
            ):
                raise RuntimeError("model differs from frozen manifest: " + asset["id"])
            install(source, asset["path"])
        # Development source paths are neither useful nor exposed in the bundle.
        for asset in spec["assets"]:
            del asset["development_path"]
        target = destination / "models.json"
        target.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
        files["models.json"] = {
            "path": "models.json",
            "sha256": digest(target),
            "bytes": target.stat().st_size,
            "executable": False,
        }

    elf_receipt = audit_elf_tree(destination, target_platform)
    (ROOT / "build/reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "build/reports/runtime-elf-audit.json").write_text(
        json.dumps(elf_receipt, indent=2) + "\n"
    )
    worker_entries = [
        {"path": p.relative_to(ROOT / "workers").as_posix(), "sha256": digest(p)}
        for p in sorted((ROOT / "workers/src").rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json"}
    ]
    worker_digest = hashlib.sha256(
        json.dumps(worker_entries, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema": 1,
        "target": target_platform,
        "stage": "candidate",
        "pack": {
            "version": os.environ.get(
                "TRACEFOLIO_MODEL_PACK_VERSION", (ROOT / "VERSION").read_text().strip()
            ),
            "contract": "tracefolio-worker-v1",
            "protocol": 1,
            "worker_source_sha256": worker_digest,
        },
        "minimum_glibc": "2.35",
        "python": sys.version,
        "python_distribution": {
            "version": python_lock["version"],
            "build": python_lock["build"],
            "sha256": archive_spec["sha256"],
        },
        "worker_lock_sha256": digest(ROOT / "workers/uv.lock"),
        "dependencies": [
            {"name": d.metadata["Name"], "version": d.version} for d in distributions
        ],
        "files": [files[k] for k in sorted(files)],
        "total_bytes": sum(f["bytes"] for f in files.values()),
    }
    # Immutable dependency groups bind dispatch verification to build-owned paths.
    shared = [
        name
        for name in sorted(files)
        if not name.startswith(("models/", "description/"))
        and name != "sources/llama-source.json"
    ]
    groups = {
        "media": [
            name
            for name in sorted(files)
            if name.startswith("media/") or name == "sources/ffmpeg-source.json"
        ]
    }
    if include_models:
        assets = {a["id"]: a["path"] for a in spec["assets"]}
        groups["document-render"] = sorted(shared)
        profiles = {p["id"]: p for p in spec["models"]}
        for profile in spec["models"]:
            required_assets = list(profile["assets"])
            if profile["id"] == "pose.rtmpose-t":
                required_assets += profiles["vision.yolox-nano"]["assets"]
            groups[profile["id"]] = sorted(
                set(shared + [assets[name] for name in required_assets])
            )
            if profile["id"] == "description.lfm25-450m":
                groups[profile["id"]] += [
                    "description/bin/llama-server",
                    "sources/llama-source.json",
                ]
                groups[profile["id"]].sort()
    manifest["dependency_groups"] = {
        key: value for key, value in groups.items() if value
    }
    (destination / "runtime.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "directory": str(destination),
                "files": len(files),
                "dependencies": len(distributions),
                "bytes": manifest["total_bytes"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument("--media", type=Path)
    parser.add_argument("--description", type=Path)
    args = parser.parse_args()
    stage(args.destination.resolve(), args.include_models, args.media, args.description)
