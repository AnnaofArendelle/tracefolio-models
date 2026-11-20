#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Ship runtime source, corresponding native source, licenses and checksums."""

import argparse
import csv
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def package(runtime: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    target = json.loads((runtime / "runtime.json").read_text())["target"]
    suffix = "-arm64" if target.endswith("arm64") else ""
    inventory = json.loads(
        (ROOT / f"packaging/manifests/native-components{suffix}.json").read_text()
    )
    manifest = json.loads((runtime / "runtime.json").read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    packages = []
    relationships = []
    for dependency in manifest["dependencies"]:
        identifier = "SPDXRef-python-" + re.sub(r"[^A-Za-z0-9.-]", "-", dependency["name"])
        packages.append(
            {
                "SPDXID": identifier,
                "name": dependency["name"],
                "versionInfo": dependency["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )
    models = json.loads((runtime / "models.json").read_text())
    for asset in models["assets"]:
        packages.append(
            {
                "SPDXID": "SPDXRef-model-" + asset["id"],
                "name": asset["id"],
                "versionInfo": asset["source_revision"],
                "downloadLocation": asset["source_url"],
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": asset["license"],
                "copyrightText": asset["attribution"],
                "checksums": [{"algorithm": "SHA256", "checksumValue": asset["sha256"]}],
                "comment": "Weight license: " + asset["license"],
            }
        )
    for component in inventory["components"]:
        native = runtime / component["runtime_path"]
        with native.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != component["sha256"] or native.stat().st_size != component["bytes"]:
            raise ValueError("native source inventory differs: " + component["id"])
        for notice in component["licenses"]:
            if (
                hashlib.sha256((runtime / notice["path"]).read_bytes()).hexdigest()
                != notice["sha256"]
            ):
                raise ValueError("native license differs: " + notice["path"])
        packages.append(
            {
                "SPDXID": "SPDXRef-native-" + re.sub(r"[^A-Za-z0-9.-]", "-", component["id"]),
                "name": component["name"],
                "versionInfo": component["version"],
                "downloadLocation": component["upstream_source_url"],
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": component["license"],
                "copyrightText": "NOASSERTION",
                "checksums": [{"algorithm": "SHA256", "checksumValue": component["sha256"]}],
            }
        )
    for name, path in [
        ("ffmpeg", "media/bin/ffmpeg"),
        ("ffprobe", "media/bin/ffprobe"),
        ("llama-server", "description/bin/llama-server"),
    ]:
        entry = next(item for item in manifest["files"] if item["path"] == path)
        packages.append(
            {
                "SPDXID": "SPDXRef-executable-" + name,
                "name": name,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "MIT" if name == "llama-server" else "GPL-3.0-or-later",
                "copyrightText": "NOASSERTION",
                "checksums": [{"algorithm": "SHA256", "checksumValue": entry["sha256"]}],
            }
        )
    packages.append(
        {
            "SPDXID": "SPDXRef-python-runtime",
            "name": "CPython standalone",
            "versionInfo": manifest["python_distribution"]["version"],
            "downloadLocation": "https://github.com/astral-sh/python-build-standalone",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": manifest["python_distribution"]["sha256"],
                }
            ],
        }
    )
    packages.append(
        {
            "SPDXID": "SPDXRef-worker",
            "name": "Tracefolio model worker",
            "versionInfo": revision,
            "downloadLocation": "https://github.com/AnnaofArendelle/tracefolio-models",
            "filesAnalyzed": False,
            "licenseConcluded": "AGPL-3.0-only",
            "licenseDeclared": "AGPL-3.0-only",
            "copyrightText": "Copyright 2026 Tracefolio contributors",
        }
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
            filename = lock.get("filename", unquote(urlsplit(lock["url"]).path.rsplit("/", 1)[-1]))
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
        license_names = re.compile(
            r"^((license|licence|copying|notice|copyright)([._-].*)?|third[-_]?party[-_]?(notices?|licenses?)([._-].*)?)$",
            re.I,
        )
        for distribution in site.glob("*.dist-info"):
            fields = Parser().parsestr((distribution / "METADATA").read_text(errors="replace"))
            location = notices / "python" / distribution.name
            location.mkdir(parents=True, exist_ok=True)
            shutil.copy2(distribution / "METADATA", location / "METADATA")
            found = {
                entry
                for entry in distribution.rglob("*")
                if entry.is_file() and license_names.match(entry.name)
            }
            for record in csv.reader((distribution / "RECORD").read_text().splitlines()):
                relative = Path(record[0])
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                entry = site / relative
                if entry.is_file() and license_names.match(entry.name):
                    found.add(entry)
            for filename in fields.get_all("License-File", []):
                entry = distribution / "licenses" / filename
                if not entry.is_file():
                    entry = distribution / filename
                if not entry.is_file():
                    raise FileNotFoundError("declared license missing: " + str(entry))
                found.add(entry)
            if not found and not fields.get("License") and not fields.get("License-Expression"):
                raise ValueError("missing Python license metadata: " + distribution.name)
            for entry in sorted(found):
                dest = location / entry.relative_to(site)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, dest)
            for package in packages:
                if package["SPDXID"] == "SPDXRef-python-" + re.sub(
                    r"[^A-Za-z0-9.-]", "-", fields["Name"].lower().replace("_", "-")
                ):
                    package["licenseDeclared"] = fields.get("License-Expression", "NOASSERTION")
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
                    "runtime_manifest": json.loads((runtime / "runtime.json").read_text()),
                    "native_sources": sources,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    for package in packages:
        relationships.append(
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": package["SPDXID"],
            }
        )
    (destination / f"tracefolio-models-sbom-{target}.spdx.json").write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "dataLicense": "CC0-1.0",
                "SPDXID": "SPDXRef-DOCUMENT",
                "name": "Tracefolio Models " + target,
                "documentNamespace": "https://github.com/AnnaofArendelle/tracefolio-models/sbom/"
                + revision
                + "/"
                + target,
                "creationInfo": {
                    "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "creators": ["Tool: Tracefolio model pack builder"],
                },
                "hasExtractedLicensingInfos": [
                    {
                        "licenseId": "LicenseRef-FunASR-Model-1.1",
                        "extractedText": (
                            ROOT / "packaging/licenses/FunASR-model-v1.1.txt"
                        ).read_text(),
                    },
                    {
                        "licenseId": "LicenseRef-LFM-Open-1.0",
                        "extractedText": (
                            ROOT / "packaging/licenses/LFM-Open-License-1.0.txt"
                        ).read_text(),
                    },
                ],
                "packages": packages,
                "relationships": relationships,
                "documentComment": "Actual runtime inventory; NOASSERTION preserves model and dependency license terms provided in the accompanying license archive.",
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
