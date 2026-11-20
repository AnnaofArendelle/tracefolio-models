# SPDX-License-Identifier: AGPL-3.0-only
"""Task-scoped, offline LFM captions using the packaged llama.cpp server."""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import math
from pathlib import Path
import socket
import subprocess
import time

from PIL import Image, ImageOps

from tracefolio_worker.ocr.rapid import check_source_dimensions

PROFILE = json.loads(Path(__file__).with_name("description_profile.json").read_text())


class LocalCaption:
    def __init__(self, runtime, staging, assets):
        self.runtime, self.staging, self.assets = runtime, staging, assets
        self.process = None
        # The supervisor starts the worker in its private staging directory.
        self.socket = "caption.sock"

    def __enter__(self):
        # Asset bytes and runtime are already checked by the supervisor at dispatch.
        args = [
            str(self.runtime / "description/bin/llama-server"),
            "-m",
            str(self.assets["language"].path),
            "--mmproj",
            str(self.assets["projector"].path),
            "--alias",
            "lfm25",
            "--host",
            self.socket,
            "-ngl",
            "0",
            "--no-mmproj-offload",
            "-c",
            "4096",
            "-np",
            "1",
            "-t",
            "2",
            "-tb",
            "2",
            "-b",
            "512",
            "-ub",
            "128",
            "--jinja",
            "--image-min-tokens",
            "64",
            "--image-max-tokens",
            "128",
            "--cache-ram",
            "1",
            "--no-cache-idle-slots",
            "--no-webui",
            "--no-slots",
            "--offline",
            "--threads-http",
            "1",
        ]
        self.process = subprocess.Popen(args, stdin=subprocess.DEVNULL)
        return self

    def ready(self):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("local description runtime exited before ready")
            try:
                status, _ = self.request("GET", "/health")
                if status == 200:
                    return
            except (FileNotFoundError, ConnectionRefusedError):
                pass  # The private socket is created during normal server startup.
            time.sleep(0.1)
        raise TimeoutError("local description runtime startup timed out")

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("localhost", timeout=180)
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.settimeout(180)
        try:
            connection.sock.connect(self.socket)
            connection.request(method, path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read(65537)
            if len(data) > 65536:
                raise ValueError("local description response exceeds budget")
            return response.status, data
        finally:
            connection.close()

    def describe(self, pixels, max_tokens=180):
        body = {
            "model": "lfm25",
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
            "cache_prompt": False,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROFILE["prompt"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64.b64encode(pixels).decode()
                            },
                        },
                    ],
                }
            ],
        }
        status, raw = self.request("POST", "/v1/chat/completions", json.dumps(body).encode())
        if status != 200:
            raise RuntimeError(f"local description returned HTTP {status}")
        response = json.loads(raw)
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("invalid local description choices")
        text = choices[0].get("message", {}).get("content")
        finish = choices[0].get("finish_reason")
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 8192:
            raise ValueError("empty or oversized local description")
        if finish not in {"stop", "length"}:
            raise ValueError("unsupported local description finish reason")
        return {
            "text": text,
            "finish_reason": finish,
            "response": response,
            "image_sha256": hashlib.sha256(pixels).hexdigest(),
        }

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()


def caption_image(source, crop=None, orientation=None):
    with Image.open(source) as original:
        check_source_dimensions(*original.size)
        if orientation is not None:
            original.getexif()[274] = orientation
        with ImageOps.exif_transpose(original) as upright:
            if crop is not None:
                if not (
                    0 <= crop[0] < crop[2] <= upright.width
                    and 0 <= crop[1] < crop[3] <= upright.height
                ):
                    raise ValueError("description crop outside image")
                image = upright.crop(crop).convert("RGB")
            else:
                image = upright.convert("RGB")
            try:
                image.thumbnail((384, 384), Image.Resampling.LANCZOS)
                out = io.BytesIO()
                image.save(out, format="JPEG", quality=90)
                return out.getvalue()
            finally:
                image.close()


def caption_page(page, region=None):
    from tracefolio_worker.documents.pdf import render_region

    width, height = page.get_size()
    if not all(math.isfinite(v) and 0 < v <= 100000 for v in (width, height)):
        raise ValueError("invalid description PDF page size")
    scale = 384 / max(width, height)
    if region:
        scale = 384 / max(region["right"] - region["left"], region["top"] - region["bottom"])
    crop, _, _ = render_region(
        page, math.ceil(width * scale), math.ceil(height * scale), region, scale
    )
    bitmap = page.render(scale=scale, crop=crop, limit_image_cache=True)
    try:
        with bitmap.to_pil() as image:
            image.thumbnail((384, 384), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            with image.convert("RGB") as rgb:
                rgb.save(out, format="JPEG", quality=90)
            return out.getvalue()
    finally:
        bitmap.close()


def run_description(config, wire, render_only=False, models=None, resources=None):
    from tracefolio_worker.ocr.rapid import Asset
    from tracefolio_worker.protocol.wire import staged_file

    required = {
        "runtime",
        "staging",
        "source",
        "kind",
        "page",
        "crop",
        "pdf_region",
        "orientation_override",
        "next_ordinal",
    }
    if not render_only and "probe" in config:
        required.add("probe")
    if set(config) != required or config["kind"] not in {"image", "pdf"}:
        raise ValueError("invalid local description input")
    if (
        type(config["page"]) is not int
        or not 0 <= config["page"] <= 1000000
        or type(config["next_ordinal"]) is not int
        or not 0 <= config["next_ordinal"] <= 1000000
    ):
        raise ValueError("invalid description page or checkpoint")
    orientation, crop = config["orientation_override"], config["crop"]
    if orientation is not None and (type(orientation) is not int or not 1 <= orientation <= 8):
        raise ValueError("invalid description orientation")
    if crop is not None and (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in crop)
    ):
        raise ValueError("invalid description image region")
    if config["kind"] == "image" and (config["page"] != 0 or config["pdf_region"] is not None):
        raise ValueError("PDF selection used for image description")
    if config["kind"] == "pdf" and (
        crop is not None
        or orientation is not None
        or config["pdf_region"] is not None
        and config["page"] == 0
    ):
        raise ValueError("invalid PDF description selection")
    runtime, staging = Path(config["runtime"]), Path(config["staging"])
    if not runtime.is_absolute() or not staging.is_absolute():
        raise ValueError("description roots must be absolute")
    source = staged_file(staging, config["source"])
    if render_only:
        import pypdfium2 as pdf

        if config["kind"] != "pdf" or config["page"] < 1:
            raise ValueError("description render requires one PDF page")
        with pdf.PdfDocument(source) as document:
            if config["page"] > len(document):
                raise ValueError("description page outside PDF")
            page = document[config["page"] - 1]
            try:
                pixels = caption_page(page, config["pdf_region"])
            finally:
                page.close()
            wire.emit("ready", {"module": "description-render"})
            wire.emit(
                "artifact",
                {
                    "page": config["page"],
                    "total_pages": len(document),
                    "image": base64.b64encode(pixels).decode(),
                    "image_sha256": hashlib.sha256(pixels).hexdigest(),
                },
            )
            wire.emit("done", {"status": "completed"})
        return
    document = None
    try:
        if config["kind"] == "pdf":
            import pypdfium2 as pdf

            document = pdf.PdfDocument(source)
            total = len(document)
            first = max(0, config["page"] - 1)
            stop = config["page"] or min(total, 5000)
            if not first < stop <= total:
                raise ValueError("description page outside PDF")
        else:
            total, first, stop = 1, 0, 1
        start = max(first, config["next_ordinal"])
        if start >= stop:
            raise ValueError("description checkpoint already completed")
        if "caption" not in models:
            manifest = json.loads((runtime / "models.json").read_text())
            assets = {
                asset["role"]: Asset(runtime / asset["path"], asset["sha256"])
                for asset in manifest["assets"]
                if asset["role"] in {"language", "projector"}
            }
            adapter = resources.enter_context(LocalCaption(runtime, staging, assets))
            adapter.ready()
            if "probe" in config:
                probe = staged_file(staging, config["probe"])
                adapter.describe(caption_image(probe), max_tokens=8)
            models["caption"] = adapter
        adapter = models["caption"]
        for ordinal in range(start, stop):
            if document is not None:
                page = document[ordinal]
                try:
                    pixels = caption_page(page, config["pdf_region"])
                finally:
                    page.close()
            else:
                pixels = caption_image(source, crop, orientation)
            result = adapter.describe(pixels)
            result.update(
                schema=1,
                page=ordinal + 1 if document is not None else 0,
                total_pages=total,
                model=PROFILE["profile"],
                prompt_sha256=hashlib.sha256(PROFILE["prompt"].encode()).hexdigest(),
            )
            if ordinal == start:
                wire.emit(
                    "ready",
                    {
                        "module": "description",
                        "model": PROFILE["profile"],
                        "total_pages": total,
                        "inferred": True,
                    },
                )
            wire.emit(
                "artifact",
                {
                    "module": "description",
                    "ordinal": ordinal,
                    "result": result,
                    "checkpoint": {"next_ordinal": ordinal + 1},
                },
            )
        wire.emit("done", {"status": "completed", "next_ordinal": stop, "total_pages": total})
    finally:
        if document is not None:
            document.close()
