# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded subtitles and real-PTS video observations; sampling is always partial."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import threading

from tracefolio_worker.asr.media import number_ns, read_exact

CELL_NS = 2_000_000_000
MAX_AUTO_FRAMES = 240
MAX_FINE_CELLS = 60
MAX_STREAMS = 32
TIME_BASE = re.compile(r"config in time_base:\s*(\d+)/(\d+)")
FRAME = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(\S+)\s+pts_time:(\S+).*?\bs:(\d+)x(\d+)")
TIME = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})$")


def probe(runtime: Path, source: Path):
    command = [
        str(runtime / "media/bin/ffprobe"),
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-show_entries",
        "format=start_time,duration:stream=index,codec_type,codec_name,width,height,start_time,duration,time_base",
        "-of",
        "json",
        str(source),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        data = process.stdout.read(1_048_577)
        if len(data) > 1_048_576:
            raise ValueError("video metadata budget exceeded")
        errors = process.stderr.read(65_537)
        if process.wait(timeout=30) != 0:
            raise ValueError("video probing failed: " + errors[:4096].decode(errors="replace"))
        value = json.loads(data)
        streams = value.get("streams", [])
        if len(streams) > 256 or any(
            type(s.get("index")) is not int or s["index"] < 0 for s in streams
        ):
            raise ValueError("invalid video streams")
        origin = number_ns(value.get("format", {}).get("start_time"))
        duration = number_ns(value.get("format", {}).get("duration"))
        return streams, origin, duration
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def schedule(duration_ns: int, region=None):
    if duration_ns <= 0 or duration_ns > 1_000_000_000 * 86400 * 365:
        raise ValueError("video duration unavailable or exceeds schedule bound")
    if region is not None:
        start, end = region["start_ns"], region["end_ns"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= duration_ns
            or end - start > MAX_FINE_CELLS * CELL_NS
        ):
            raise ValueError("fine scan requires up to120 seconds within media duration")
        return list(range(start // CELL_NS, (end + CELL_NS - 1) // CELL_NS)), "fine_5hz"
    count = math.ceil(duration_ns / CELL_NS)
    # Two observations per selected cell; sparse cells bound actual heavy calls.
    step = max(1, math.ceil(count / (MAX_AUTO_FRAMES // 2)))
    return list(range(0, count, step)), "budget_activity"


def timestamp(value):
    match = TIME.fullmatch(value.strip())
    if not match:
        raise ValueError("invalid subtitle timestamp")
    hour, minute, second, millis = match.groups()
    if int(minute) >= 60 or int(second) >= 60:
        raise ValueError("invalid subtitle timestamp")
    return (
        (int(hour or 0) * 3600 + int(minute) * 60 + int(second)) * 1000 + int(millis)
    ) * 1_000_000


def subtitle_cues(stream):
    """Stream VTT cue blocks; preserve markup as source text, no inferred words."""
    block, size, ordinal = [], 0, 0
    while True:
        raw = stream.readline(65_537)
        if len(raw) > 65_536:
            raise ValueError("subtitle line budget exceeded")
        if not raw or raw.strip() == b"":
            if block:
                timing = next((i for i, text in enumerate(block) if " --> " in text), None)
                if timing is not None:
                    parts = block[timing].split(" --> ")
                    if len(parts) != 2:
                        raise ValueError("invalid subtitle cue")
                    first, last = timestamp(parts[0]), timestamp(parts[1].split()[0])
                    if last <= first:
                        raise ValueError("nonpositive subtitle interval")
                    text = "\n".join(block[timing + 1 :])
                    if text.strip():
                        yield ordinal, first, last, text
                        ordinal += 1
                        if ordinal > 200000:
                            raise ValueError("subtitle cue budget exceeded")
                block, size = [], 0
            if not raw:
                break
            continue
        text = raw.decode("utf-8", errors="strict").rstrip("\r\n")
        size += len(raw)
        if size > 262144:
            raise ValueError("subtitle cue budget exceeded")
        block.append(text)


def subtitles(runtime, source, index):
    command = [
        str(runtime / "media/bin/ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-copyts",
        "-i",
        str(source),
        "-map",
        f"0:{index}",
        "-vn",
        "-an",
        "-dn",
        "-c:s",
        "webvtt",
        "-f",
        "webvtt",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield from subtitle_cues(process.stdout)
        errors = process.stderr.read(65537)
        if process.wait(timeout=30) != 0:
            raise ValueError(
                "subtitle conversion unavailable: " + errors[:4096].decode(errors="replace")
            )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def frames(runtime, source, index, origin, start, end, progress=None):
    """Seek a cell, emit exact selected decoded-frame timestamps and RGB pixels."""

    # ffmpeg input -ss is relative to container origin. copyts retains source PTS.
    absolute_start, absolute_end = start + origin, end + origin
    vf = f"trim=start={absolute_start / 1e9:.9f}:end={absolute_end / 1e9:.9f},select='isnan(prev_selected_t)+gte(t-prev_selected_t,0.199999)',showinfo"
    command = [
        str(runtime / "media/bin/ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-v",
        "info",
        "-protocol_whitelist",
        "file,pipe",
        "-copyts",
        "-ss",
        f"{start / 1e9:.9f}",
        "-i",
        str(source),
        "-map",
        f"0:{index}",
        "-an",
        "-sn",
        "-dn",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-vf",
        vf,
        "-fps_mode",
        "passthrough",
        "-to",
        f"{absolute_end / 1e9:.9f}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    events, stop, tail = queue.Queue(maxsize=16), threading.Event(), bytearray()

    def send(item):
        while not stop.is_set():
            try:
                events.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def lines():
        time_base = None
        try:
            while line := process.stderr.readline(65537):
                if len(line) > 65536:
                    raise ValueError("video diagnostic line budget exceeded")
                text = line.decode(errors="replace")
                if "showinfo" in text and "config in time_base:" in text:
                    match = TIME_BASE.search(text)
                    if not match or int(match[2]) == 0:
                        raise ValueError("invalid video time base")
                    time_base = (int(match[1]), int(match[2]))
                elif "showinfo" in text and "pts_time:" in text:
                    match = FRAME.search(text)
                    if not match:
                        raise ValueError("unsupported decoded video frame clock")
                    if time_base is None:
                        raise ValueError("missing decoded-frame time base")
                    send((*match.groups(), time_base))
                else:
                    tail.extend(line)
                    if len(tail) > 65536:
                        del tail[:-65536]
            send(None)
        except Exception as exc:
            send(exc)

    thread = threading.Thread(target=lines, daemon=True)
    thread.start()
    previous = None
    try:
        while True:
            item = events.get(timeout=30)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            ordinal, raw_pts, seconds, width, height, time_base = item
            width, height = int(width), int(height)
            if (
                width <= 0
                or height <= 0
                or width * height > 16_000_000
                or max(width, height) > 16384
            ):
                raise ValueError("video frame source exceeds pixel budget")
            pixels = read_exact(process.stdout, width * height * 3)
            pts = (
                int(raw_pts) * time_base[0] * 1_000_000_000 // time_base[1]
                if raw_pts not in {"NOPTS", "N/A"}
                else None
            )
            if (
                pts is None
                or raw_pts in {"NOPTS", "N/A"}
                or pts < absolute_start
                or pts >= absolute_end
                or previous is not None
                and pts <= previous
            ):
                raise ValueError("video timestamp unavailable, backward or outside cell")
            previous = pts
            if progress:
                progress()
            yield {
                "raw_pts": int(raw_pts),
                "time_base": list(time_base),
                "raw_pts_ns": pts,
                "media_time_ns": pts - origin,
                "width": width,
                "height": height,
                "pixels": pixels,
                "sha256": hashlib.sha256(pixels).hexdigest(),
            }
        extra = process.stdout.read(1)
        if extra or process.wait(timeout=30) != 0:
            raise ValueError(
                "video frame conversion incomplete: " + tail[-4096:].decode(errors="replace")
            )
    finally:
        stop.set()
        if process.poll() is None:
            process.kill()
        process.wait()
        thread.join(timeout=5)
        process.stdout.close()
        process.stderr.close()


def select_frames(candidates, policy):
    if policy == "fine_5hz" or len(candidates) <= 2:
        return candidates
    import cv2
    import numpy as np

    previous = None
    scores = []
    for candidate in candidates:
        pixels = np.frombuffer(candidate["pixels"], dtype=np.uint8).reshape(
            candidate["height"], candidate["width"], 3
        )
        gray = cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), (160, 90)).astype(np.float32)
        change = (
            0.0
            if previous is None
            else max(
                float(np.abs(gray - previous).mean()),
                float(np.abs(gray[54:] - previous[54:]).mean()),
            )
        )
        score = change * (1 + min(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 100) / 100)
        scores.append(score)
        previous = gray
    return [
        candidates[0],
        candidates[max(range(1, len(candidates)), key=lambda i: (scores[i], -i))],
    ]


def bounded_selection(candidates, policy):
    """Keep only first/best full-resolution frames while scoring streaming candidates."""
    if policy == "fine_5hz":
        for frame in candidates:
            yield frame
        return
    import cv2
    import numpy as np

    previous = None
    first, best, best_score = None, None, -1.0
    for candidate in candidates:
        pixels = np.frombuffer(candidate["pixels"], dtype=np.uint8).reshape(
            candidate["height"], candidate["width"], 3
        )
        gray = cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), (160, 90)).astype(np.float32)
        score = (
            0.0
            if previous is None
            else max(
                float(np.abs(gray - previous).mean()),
                float(np.abs(gray[54:] - previous[54:]).mean()),
            )
            * (1 + min(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 100) / 100)
        )
        previous = gray
        if first is None:
            first = candidate
        elif score > best_score:
            best, best_score = candidate, score
    if first is not None:
        yield first
    if best is not None:
        yield best


def run_video(config, wire, assets, source, probe_source, runtime, staging, models):
    """Shared worker dispatch entry. Saved outputs account for streams/cells/cues."""
    from PIL import Image
    import cv2
    import numpy as np
    from tracefolio_worker.ocr.rapid import InputBudgetExceeded
    from tracefolio_worker.protocol.dispatch import checked_ocr
    from tracefolio_worker.vision.objects import ObjectAdapter
    from tracefolio_worker.vision.pose import PoseAdapter
    from tracefolio_worker.protocol.wire import staged_file

    if (
        any(type(config[key]) is not bool for key in ("with_ocr", "with_objects", "with_pose"))
        or config["with_pose"]
        and not config["with_objects"]
    ):
        raise ValueError("invalid video module selection")
    all_streams, origin, duration = probe(runtime, source)
    video = [s for s in all_streams if s["codec_type"] == "video"]
    subtitle = [s for s in all_streams if s["codec_type"] == "subtitle"]
    if len(video) > MAX_STREAMS or len(subtitle) > MAX_STREAMS:
        raise ValueError("video/subtitle stream budget exceeded")
    selected_cells, policy = (
        schedule(duration, config["region"])
        if video and origin is not None and duration is not None
        else ([], "timing_unavailable")
    )
    ocr, detector, pose = None, None, None
    if config["with_ocr"] and video:
        ocr, _ = checked_ocr(models, assets, probe_source, config["sentinel"])
    detected = None
    visual_probe = None
    if config["with_objects"] and video:
        visual_probe = staged_file(staging, config["vision_probe"])
        if "objects" not in models:
            detector = ObjectAdapter(assets["object"])
            models["objects"] = detector, detector.preflight(visual_probe, "person")
        detector, detected = models["objects"]
    # Pose initialization remains lazy; frames without persons do not load it.
    wire.emit(
        "ready",
        {
            "module": "video",
            "probe_sha256": config["probe"]["sha256"],
            "proof": {
                "sentinel": config["sentinel"],
                "video_streams": len(video),
                "subtitle_streams": len(subtitle),
                "media_origin_ns": origin,
                "duration_ns": duration,
                "selected_cells": selected_cells,
                "policy": policy,
            },
        },
    )
    ordinal = 0

    def artifact(result):
        nonlocal ordinal
        if ordinal >= config["next_ordinal"]:
            wire.emit(
                "artifact",
                {
                    "module": "video",
                    "ordinal": ordinal,
                    "result": result,
                    "checkpoint": {"next_ordinal": ordinal + 1},
                },
            )
        ordinal += 1

    if config["region"] is None:
        for stream in subtitle:
            index = stream["index"]
            chunk, first = [], 0
            try:
                for cue, a, b, text in subtitles(runtime, source, index):
                    chunk.append({"ordinal": cue, "start_ns": a, "end_ns": b, "text": text})
                    if len(chunk) >= 32 or sum(len(c["text"].encode()) for c in chunk) >= 262144:
                        artifact(
                            {
                                "schema": 1,
                                "kind": "video_subtitle",
                                "stream": f"subtitle:{index}",
                                "codec": stream["codec_name"],
                                "first": first,
                                "last": cue + 1,
                                "origin_ns": origin,
                                "cues": chunk,
                                "coverage": "complete",
                                "issues": [],
                            }
                        )
                        chunk, first = [], cue + 1
                if chunk:
                    artifact(
                        {
                            "schema": 1,
                            "kind": "video_subtitle",
                            "stream": f"subtitle:{index}",
                            "codec": stream["codec_name"],
                            "first": first,
                            "last": chunk[-1]["ordinal"] + 1,
                            "origin_ns": origin,
                            "cues": chunk,
                            "coverage": "complete",
                            "issues": [],
                        }
                    )
            except (ValueError, UnicodeError) as exc:
                artifact(
                    {
                        "schema": 1,
                        "kind": "video_subtitle",
                        "stream": f"subtitle:{index}",
                        "codec": stream["codec_name"],
                        "first": first,
                        "last": first + 1,
                        "origin_ns": origin,
                        "cues": [],
                        "coverage": "partial",
                        "issues": ["subtitle_stream_not_fully_decoded:" + str(exc)[:1024]],
                    }
                )
    for stream in video:
        index = stream["index"]
        for cell in selected_cells:
            start, end = cell * CELL_NS, min((cell + 1) * CELL_NS, duration)
            if config["region"] is not None:
                start = max(start, config["region"]["start_ns"])
                end = min(end, config["region"]["end_ns"])
            observations, issues = [], []
            if ordinal < config["next_ordinal"]:
                ordinal += 1
                continue
            try:

                def progress():
                    wire.emit("progress", {"module": "video", "cell": cell, "stream": index})

                for frame in bounded_selection(
                    frames(runtime, source, index, origin, start, end, progress), policy
                ):
                    pixels = np.frombuffer(frame.pop("pixels"), dtype=np.uint8).reshape(
                        frame["height"], frame["width"], 3
                    )
                    result = {**frame, "ocr": None, "visual": None}
                    if ocr is not None:
                        temporary = staging / "video-frame.png"
                        try:
                            Image.fromarray(pixels).save(temporary)
                            result["ocr"] = ocr.extract(temporary)
                        except InputBudgetExceeded as exc:
                            issues.append("frame_ocr_budget:" + str(exc))
                        finally:
                            temporary.unlink(missing_ok=True)
                    if detector is not None:
                        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                        visual = detector.extract_pixels(bgr)
                        visual["exif_orientation"] = 1
                        people = [o for o in visual["objects"] if o["label"] == "person"]
                        if config["with_pose"]:
                            if people and pose is None:
                                if "pose" not in models:
                                    pose = PoseAdapter(assets["pose"])
                                    probe_pixels = cv2.imread(str(visual_probe))
                                    pose.extract_pixels(
                                        probe_pixels,
                                        next(
                                            o for o in detected["objects"] if o["label"] == "person"
                                        ),
                                    )
                                    models["pose"] = pose
                                pose = models["pose"]
                            visual["poses"] = [pose.extract_pixels(bgr, o) for o in people]
                        result["visual"] = visual
                    observations.append(result)
                    if len(observations) > 11:
                        raise ValueError("video cell frame count exceeds5Hz budget")
            except ValueError as exc:
                issues.append("frame_cell_incomplete:" + str(exc)[:1024])
            artifact(
                {
                    "schema": 1,
                    "kind": "video_cell",
                    "stream": f"video:{index}",
                    "cell": cell,
                    "start_ns": start,
                    "end_ns": end,
                    "origin_ns": origin,
                    "policy": policy,
                    "frames": observations,
                    "coverage": "partial",
                    "issues": issues,
                }
            )
    if config["next_ordinal"] > ordinal:
        raise ValueError("video checkpoint beyond deterministic outputs")
    if not selected_cells and video:
        artifact(
            {
                "schema": 1,
                "kind": "video_issue",
                "issues": ["video_timing_or_duration_unavailable"],
                "coverage": "partial",
            }
        )
    wire.emit(
        "done",
        {
            "status": "completed",
            "artifacts": ordinal,
            "sampled": bool(video),
            "policy": policy,
            "duration_ns": duration,
            "selected_cells": selected_cells,
            "video_streams": [s["index"] for s in video],
            "subtitle_streams": [s["index"] for s in subtitle],
        },
    )
