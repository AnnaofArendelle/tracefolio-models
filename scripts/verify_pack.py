#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run fixed real inference probes with the packaged interpreter, in sequence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def probe(runtime: Path) -> dict:
    import wave
    from tracefolio_worker.ocr.rapid import Asset, OCRModels, RapidAdapter
    from tracefolio_worker.asr.sensevoice import SenseVoiceAdapter
    from tracefolio_worker.asr.vad import SileroAdapter
    from tracefolio_worker.vision.objects import ObjectAdapter, load_image
    from tracefolio_worker.vision.pose import PoseAdapter
    from tracefolio_worker.description import LocalCaption, caption_image

    manifest = json.loads((runtime / "models.json").read_text())
    assets = {
        a["role"]: Asset(runtime / a["path"], a["sha256"]) for a in manifest["assets"]
    }
    result = {}
    started = time.monotonic()
    ocr = RapidAdapter(
        OCRModels(assets["det"], assets["rec"], assets["cls"], assets["dictionary"])
    )
    proof = ocr.preflight(ROOT / "packaging/probes/ocr.png", "TRACEFOLIO")
    chinese = ocr.extract(ROOT / "tests/fixtures/vision/meeting.png")
    assert "青禾" in "\n".join(s["text"] for s in chinese["segments"])
    result["ocr"] = {
        "seconds": time.monotonic() - started,
        "preflight_segments": len(proof["segments"]),
        "chinese_text": [s["text"] for s in chinese["segments"]],
    }
    del ocr
    started = time.monotonic()
    asr = SenseVoiceAdapter(assets["asr"], assets["tokens"])
    sample = ROOT / "packaging/probes/asr.wav"
    with wave.open(str(sample)) as audio:
        length = audio.getnframes()
    proof = asr.preflight(sample, 0, length, "交易")
    vad = SileroAdapter(assets["vad"])
    ranges = list(vad.ranges(sample))
    assert ranges
    result["asr_vad"] = {
        "seconds": time.monotonic() - started,
        "text": proof["text"],
        "speech_ranges": ranges,
    }
    del asr, vad
    started = time.monotonic()
    detector = ObjectAdapter(assets["object"])
    photo = ROOT / "packaging/probes/vision.jpg"
    proof = detector.preflight(photo, "person")
    person = next(o for o in proof["objects"] if o["label"] == "person")
    pose = PoseAdapter(assets["pose"])
    pixels, _ = load_image(photo)
    posed = pose.extract_pixels(pixels, person)
    assert len(posed["points"]) == 17
    result["objects_pose"] = {
        "seconds": time.monotonic() - started,
        "objects": len(proof["objects"]),
        "points": len(posed["points"]),
    }
    del detector, pose, pixels
    started = time.monotonic()
    # The same real llama-server must serve both sequential images.
    for a in (assets["language"], assets["projector"]):
        a.verify()
    with tempfile.TemporaryDirectory(prefix="caption-probe-") as temporary:
        previous = Path.cwd()
        os.chdir(temporary)
        try:
            with LocalCaption(runtime, Path(temporary), assets) as caption:
                caption.ready()
                pid = caption.process.pid
                first = caption.describe(caption_image(photo), max_tokens=80)
                second = caption.describe(
                    caption_image(ROOT / "tests/fixtures/vision/meeting.png"),
                    max_tokens=80,
                )
                assert caption.process.pid == pid and caption.process.poll() is None
                result["description"] = {
                    "seconds": time.monotonic() - started,
                    "server_reused": True,
                    "texts": [first["text"], second["text"]],
                }
        finally:
            os.chdir(previous)
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(ROOT / "tests/fixtures/pdf/mixed.pdf") as document:
        assert len(document) > 0
        page = document[0]
        textpage = page.get_textpage()
        assert textpage.get_text_range()
        textpage.close()
        page.close()
    for tool in ("ffmpeg", "ffprobe"):
        subprocess.run(
            [str(runtime / "media/bin" / tool), "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return {
        "passed": True,
        "runtime_manifest_sha256": hashlib.sha256(
            (runtime / "runtime.json").read_bytes()
        ).hexdigest(),
        "modules": result,
        "pdf_and_media": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inside", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if not args.inside:
        env = dict(
            os.environ,
            PYTHONHOME=str(runtime / "python"),
            PYTHONNOUSERSITE="1",
            PYTHONPATH="",
            LD_LIBRARY_PATH=str(runtime / "python/lib"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        subprocess.run(
            [
                str(runtime / "python/bin/python3.12"),
                str(Path(__file__).resolve()),
                "--runtime",
                str(runtime),
                "--report",
                str(args.report.resolve()),
                "--inside",
            ],
            env=env,
            check=True,
        )
    else:
        receipt = probe(runtime)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(receipt, ensure_ascii=False))
