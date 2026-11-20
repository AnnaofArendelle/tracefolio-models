# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import io
import json

import pytest

from tracefolio_worker.protocol.wire import Wire, read_line, request, staged_file


def test_identity_and_acknowledged_artifact():
    req = {
        "schema": 1,
        "task_id": "task",
        "attempt": 3,
        "config_hash": "a" * 64,
        "module": "ocr",
        "input": {},
    }
    assert request(io.BytesIO(json.dumps(req).encode() + b"\n")) == req
    out = io.BytesIO()
    wire = Wire(io.BytesIO(b'{"schema":1,"ack":0}\n{"schema":1,"ack":1}\n'), out, req)
    wire.emit("ready", {"inferred": True})
    wire.emit("artifact", {"text": "归档合同"})
    wire.emit("done", {"status": "completed"})
    lines = [json.loads(s) for s in out.getvalue().splitlines()]
    assert [s["sequence"] for s in lines] == [0, 1, 2]
    assert all(s["task_id"] == "task" and s["attempt"] == 3 for s in lines)
    wire = Wire(io.BytesIO(b'{"schema":1,"ack":9}\n'), io.BytesIO(), req)
    with pytest.raises(ValueError, match="acknowledgment"):
        wire.emit("artifact", {})


def test_protocol_limits_and_file_identity(tmp_path):
    for value in [b"[]\n", b'{"v":NaN}\n', b'{"v":1}', b"x" * 100]:
        with pytest.raises(ValueError):
            read_line(io.BytesIO(value), limit=80)
    source = tmp_path / "input.txt"
    source.write_text("合同")
    descriptor = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert staged_file(tmp_path, descriptor) == source
    source.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        staged_file(tmp_path, descriptor)
    for path in ["../escape.txt", "/tmp/escape.txt"]:
        with pytest.raises(ValueError, match="outside staging"):
            staged_file(tmp_path, {**descriptor, "path": path})
    link = tmp_path / "linked.txt"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        staged_file(tmp_path, {**descriptor, "path": link.name})
