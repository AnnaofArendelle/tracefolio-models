# SPDX-License-Identifier: AGPL-3.0-only
"""One task per process, bounded JSONL, acknowledged real preflight/artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAX_EVENT = 8 * 1024 * 1024
MAX_REQUEST = 65536


def read_line(stream, limit=MAX_REQUEST):
    raw = stream.readline(limit + 1)
    if not raw or len(raw) > limit or not raw.endswith(b"\n"):
        raise ValueError("missing or oversized protocol line")
    result = json.loads(
        raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON"))
    )
    if not isinstance(result, dict):
        raise ValueError("protocol message must be an object")
    return result


def request(stream):
    req = read_line(stream)
    if set(req) != {"schema", "task_id", "attempt", "config_hash", "module", "input"}:
        raise ValueError("invalid request fields")
    if (
        req["schema"] != 1
        or type(req["attempt"]) is not int
        or req["attempt"] < 1
        or not isinstance(req["task_id"], str)
        or not 1 <= len(req["task_id"]) <= 256
        or not isinstance(req["config_hash"], str)
        or len(req["config_hash"]) != 64
        or any(c not in "0123456789abcdef" for c in req["config_hash"])
        or req["module"]
        not in {
            "ocr",
            "objects",
            "pose",
            "asr",
            "pdf",
            "video",
            "description",
            "description-render",
        }
        or not isinstance(req["input"], dict)
    ):
        raise ValueError("unsupported request contract")
    return req


class Wire:
    def __init__(self, incoming, outgoing, req):
        self.incoming, self.outgoing = incoming, outgoing
        self.identity = {k: req[k] for k in ("task_id", "attempt", "config_hash")}
        self.sequence = 0

    def emit(self, kind, payload):
        event = {
            "schema": 1,
            **self.identity,
            "type": kind,
            "sequence": self.sequence,
            "payload": payload,
        }
        data = (
            json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
        if len(data) > MAX_EVENT:
            raise ValueError("artifact exceeds protocol budget")
        remaining = memoryview(data)
        while remaining:
            written = self.outgoing.write(remaining)
            if not written:
                raise BrokenPipeError("protocol output closed")
            remaining = remaining[written:]
        self.outgoing.flush()
        self.sequence += 1
        if kind not in {"done", "error"}:
            ack = read_line(self.incoming)
            if ack != {"schema": 1, "ack": event["sequence"]}:
                raise ValueError("missing or incorrect durable acknowledgment")


def staged_file(root: Path, spec: dict) -> Path:
    """Private supervisor-owned input, regular file, exact frozen bytes."""
    if set(spec) != {"path", "sha256"}:
        raise ValueError("invalid staged file descriptor")
    relative = Path(spec["path"])
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in {".", ".."} for p in relative.parts)
    ):
        raise ValueError("input path outside staging")
    current = root
    if current.is_symlink() or not current.is_dir():
        raise ValueError("invalid staging root")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("staged input cannot contain symlinks")
    if not current.is_file():
        raise ValueError("staged input is not a regular file")
    with current.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise ValueError("staged input hash changed")
    return current
