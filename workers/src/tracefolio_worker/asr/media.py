# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded final-resampler PCM/PTS ledger and per-clock VAD/SenseVoice pipeline.

FFmpeg's final ashowinfo is AFTER async=0 resampling. Audio samples are never
stretched to hide gaps. Unknown/negative/backward PTS preserve text with an
unavailable location and never form trusted cross-block bridges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import errno
import json
from pathlib import Path
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
import wave

import numpy as np

from tracefolio_worker.asr.sensevoice import MAX_SAMPLES, SenseVoiceAdapter, bounded_chunks
from tracefolio_worker.asr.vad import SileroAdapter

RATE = 16000
MAX_PCM_BYTES = 32 * 1024**3
RESERVE_BYTES = 64 * 1024**2
MAX_CLOCKS = 100000
MAX_STREAMS = 32
MAX_CHUNKS = 2000000
FRAME = re.compile(r"\bn:(\d+)\s+pts:(\S+)\s+pts_time:(\S+).*?\brate:(\d+)\s+nb_samples:(\d+)")
DECODE_ERROR = re.compile(
    r"Error while decoding stream|Invalid PCM packet|[Pp]acket corrupt|corrupt decoded frame"
)


def merge_touching_ranges(ranges):
    """VAD max-speech cuts are algorithm boundaries, not observed silence gaps."""
    pending = None
    for start, end in ranges:
        if start < 0 or end <= start or pending and start < pending[1]:
            raise ValueError("nonmonotonic VAD selection")
        if pending and start == pending[1]:
            pending = (pending[0], end)
        else:
            if pending:
                yield pending
            pending = (start, end)
    if pending:
        yield pending


@dataclass
class Clock:
    stream: str
    start_sample: int
    end_sample: int
    raw_start_pts: int | None
    media_start_ns: int | None
    reason: str
    frame_count: int


def number_ns(value):
    try:
        value = Decimal(str(value))
        if not value.is_finite():
            return None
        return int(value * 1000000000)
    except (InvalidOperation, ValueError):
        return None


def probe_media(runtime: Path, source: Path):
    command = [
        str(runtime / "media/bin/ffprobe"),
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-show_entries",
        "format=format_name,start_time,duration:stream=index,codec_type,sample_rate,start_time,duration",
        "-of",
        "json",
        str(source),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Metadata is bounded even for a malicious container with enormous streams.
        raw = process.stdout.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("media metadata budget exceeded")
        errors = process.stderr.read(65537)
        if process.wait(timeout=30) != 0:
            raise ValueError("media probing failed: " + errors[:4096].decode(errors="replace"))
        data = json.loads(raw)
        all_streams = data.get("streams", [])
        if len(all_streams) > 256:
            raise ValueError("media stream budget exceeded")
        streams = [s for s in all_streams if s.get("codec_type") == "audio"]
        if any(type(s.get("index")) is not int or s["index"] < 0 for s in streams):
            raise ValueError("invalid audio stream identity")
        origin = number_ns(data.get("format", {}).get("start_time"))
        if origin is None and data.get("format", {}).get("format_name") in {
            "wav",
            "flac",
            "amr",
            "aac",
        }:
            # Elementary/sample-container audio has no independent presentation
            # origin. Its demuxer timestamp timeline is defined to start at zero.
            origin = 0
            data["format"]["origin_semantics"] = "elementary_audio_zero"
        # A missing container origin is explicit; no guessed player timestamp.
        return streams, origin, data.get("format", {})
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def read_exact(stream, count):
    raw = bytearray()
    while len(raw) < count:
        block = stream.read(count - len(raw))
        if not block:
            raise ValueError("FFmpeg final PCM/frame sample count mismatch")
        raw.extend(block)
    return raw


def consume_wave_header(stream):
    """Portable build has WAV muxing; discard only its validated pipe header."""
    head = read_exact(stream, 12)
    if head[:4] != b"RIFF" or head[8:] != b"WAVE":
        raise ValueError("invalid final PCM WAV stream")
    remaining = 65536
    found_format = False
    while remaining > 0:
        part = read_exact(stream, 8)
        length = struct.unpack("<I", part[4:])[0]
        if part[:4] == b"data":
            if not found_format:
                raise ValueError("final PCM WAV data lacks format")
            return
        if length > remaining:
            raise ValueError("final PCM WAV header exceeds budget")
        data = read_exact(stream, length + length % 2)
        remaining -= length + length % 2 + 8
        if part[:4] == b"fmt ":
            if length < 16 or struct.unpack("<HHIIHH", data[:16]) != (1, 1, RATE, RATE * 2, 2, 16):
                raise ValueError("final PCM WAV format mismatch")
            found_format = True


def decode_audio(
    runtime: Path,
    source: Path,
    staging: Path,
    index: int,
    origin: int | None,
    progress=None,
    budget=MAX_PCM_BYTES,
    waveform=False,
):
    """Disk-bounded PCM and JSONL clocks; memory independent of audio duration."""
    target = staging / f"audio-{index}.pcm"
    ledger = staging / f"audio-{index}.clocks.jsonl"
    free = shutil.disk_usage(staging).free
    # The frozen hard budget must not shrink with changing free disk space: a
    # shorter decode on retry would change already saved clock identities.
    if free < RESERVE_BYTES + min(budget, 4 * 1024**2):
        return target, ledger, 0, 0, ["decoded_audio_disk_budget_unavailable"]
    budget -= budget % 2
    if budget < MAX_SAMPLES * 2:
        return target, ledger, 0, 0, ["decoded_audio_disk_budget_unavailable"]
    args = [
        str(runtime / "media/bin/ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-v",
        "info",
        "-protocol_whitelist",
        "file,pipe",
        "-copyts",
        "-i",
        str(source),
        "-map",
        f"0:{index}",
        "-vn",
        "-sn",
        "-dn",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-af",
        "aresample=16000:async=0,aformat=sample_fmts=s16:channel_layouts=mono,ashowinfo",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "pipe:1",
    ]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    events = queue.Queue(maxsize=64)
    stop = threading.Event()
    tail = bytearray()
    decode_error = False

    def send(value):
        while not stop.is_set():
            try:
                events.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def lines():
        nonlocal decode_error
        try:
            while line := process.stderr.readline(65537):
                if len(line) > 65536 or not line.endswith(b"\n"):
                    send(ValueError("FFmpeg diagnostic line exceeds budget"))
                    return
                text = line.decode(errors="replace")
                if "ashowinfo" in text and "nb_samples:" in text:
                    match = FRAME.search(text)
                    if not match:
                        send(ValueError("unsupported final resampler frame ledger"))
                        return
                    send(match.groups())
                else:
                    # FFmpeg may return success after dropping a corrupt final
                    # packet. Retain only the actual complete decoded frames,
                    # while recording explicit decoder errors as partial.
                    if DECODE_ERROR.search(text):
                        decode_error = True
                    tail.extend(line)
                    if len(tail) > 65536:
                        del tail[:-65536]
        finally:
            send(None)

    reader = threading.Thread(target=lines, daemon=True)
    reader.start()
    cursor = count = expected = 0
    active = None
    issues = []
    try:
        consume_wave_header(process.stdout)
        with target.open("xb") as pcm, ledger.open("x", encoding="utf-8") as out:
            peaks = []
            bin_samples = 1600

            def flush():
                nonlocal count, bin_samples
                if active and active.end_sample > active.start_sample:
                    record = asdict(active)
                    if waveform:
                        record["waveform"] = {"bin_samples": bin_samples, "peaks": peaks}
                    out.write(json.dumps(record, separators=(",", ":")) + "\n")
                    peaks.clear()
                    bin_samples = 1600
                    count += 1

            while True:
                event = events.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event
                ordinal, pts, _seconds, rate, samples = event
                ordinal, rate, samples = int(ordinal), int(rate), int(samples)
                if ordinal != expected or rate != RATE or not 0 < samples <= 262144:
                    raise ValueError("FFmpeg final frame sequence/rate/size invalid")
                expected += 1
                if (cursor + samples) * 2 > budget or count >= MAX_CLOCKS:
                    issues.append("decoded_audio_disk_or_clock_budget_unprocessed_tail")
                    process.kill()
                    break
                try:
                    pts = int(pts)
                except ValueError:
                    pts = None
                media = (
                    None
                    if pts is None or origin is None
                    else int(Decimal(pts) * 1000000000 / RATE) - origin
                )
                reason = "continuous"
                if media is None:
                    reason = "media_pts_unavailable"
                elif media < 0:
                    media, reason = None, "negative_media_pts_unavailable"
                if active:
                    predicted = (
                        None
                        if active.raw_start_pts is None
                        else active.raw_start_pts + cursor - active.start_sample
                    )
                    aligned = (predicted is None and pts is None) or (
                        predicted is not None and pts == predicted
                    )
                    # Encoder padding may begin before the playable origin.
                    # Once an actual output frame reaches a nonnegative media
                    # PTS, it owns a new known clock. Do not let the initial
                    # negative frame make the entire recording unlocatable.
                    availability_changed = (active.media_start_ns is None) != (media is None)
                    if not aligned or availability_changed:
                        if pts is not None and predicted is not None and pts < predicted:
                            media, reason = None, "backward_or_overlapping_pts_unavailable"
                        elif reason == "continuous":
                            reason = "pts_discontinuity"
                        flush()
                        active = None
                if active is None:
                    active = Clock(str(index), cursor, cursor, pts, media, reason, 0)
                raw = read_exact(process.stdout, samples * 2)
                try:
                    if pcm.write(raw) != len(raw):
                        raise OSError(errno.ENOSPC, "short PCM write")
                except OSError as error:
                    if error.errno != errno.ENOSPC:
                        raise
                    # A disk-full boundary is an explicit unprocessed tail.
                    # Exclude the incomplete frame and retain only full frames.
                    pcm.truncate(cursor * 2)
                    issues.append("audio_staging_disk_full_unprocessed_tail")
                    process.kill()
                    break
                if waveform:
                    signal = np.frombuffer(raw, dtype="<i2")
                    consumed = 0
                    while consumed < samples:
                        position = cursor - active.start_sample + consumed
                        slot = position // bin_samples
                        if slot == 1024:
                            peaks = [max(a, b) for a, b in zip(peaks[::2], peaks[1::2])]
                            bin_samples *= 2
                            slot //= 2
                        size = min(samples - consumed, bin_samples - position % bin_samples)
                        part = signal[consumed : consumed + size]
                        peak = max(-int(part.min()), int(part.max()))
                        if slot == len(peaks):
                            peaks.append(peak)
                        else:
                            peaks[slot] = max(peaks[slot], peak)
                        consumed += size
                cursor += samples
                active.end_sample, active.frame_count = cursor, active.frame_count + 1
                if progress:
                    progress(cursor)
            flush()
            pcm.flush()
            if not issues:
                if process.stdout.read(1):
                    raise ValueError("FFmpeg emitted PCM without a final frame ledger")
                if process.wait(timeout=30) != 0 or decode_error:
                    issues.append("audio_decode_error_partial_tail")
                if cursor == 0:
                    issues.append("audio_decode_empty")
    except OSError as error:
        if error.errno != errno.ENOSPC:
            raise
        # Published inputs cannot pretend a partially written frame is valid.
        raise ValueError("audio staging disk became full; retry with more space") from error
    finally:
        stop.set()
        if process.poll() is None:
            process.kill()
        process.stdout.close()
        process.wait()
        process.stderr.close()
        reader.join(timeout=5)
    if count and target.stat().st_size != cursor * 2:
        raise ValueError("final PCM sample count differs from committed ledger")
    return target, ledger, cursor, count, issues


def checked_models(models, assets, probe, sentinel):
    if "asr" not in models:
        vad = SileroAdapter(assets["vad"])
        asr = SenseVoiceAdapter(assets["asr"], assets["tokens"])
        with wave.open(str(probe)) as audio:
            length = audio.getnframes()
        ranges = list(vad.ranges(probe))
        asr.preflight(probe, 0, length, sentinel)
        if not ranges:
            raise ValueError("VAD preflight sentinel produced no speech")
        models["asr"] = vad, asr, ranges
    return models["asr"]


def run_asr(config, wire, assets, source, probe, models):
    """Input schema2; old schema1 WAV adapter remains in dispatch for replay."""
    expected = {
        "runtime",
        "staging",
        "source",
        "probe",
        "sentinel",
        "next_ordinal",
        "input_schema",
        "next_clock",
    }
    if (
        set(config) != expected
        or config["input_schema"] != 2
        or type(config["input_schema"]) is not int
    ):
        raise ValueError("unsupported media ASR configuration")
    if any(
        type(config[key]) is not int or config[key] < 0 for key in ("next_ordinal", "next_clock")
    ):
        raise ValueError("invalid media ASR checkpoint")

    runtime, staging = Path(config["runtime"]), Path(config["staging"])
    if not runtime.is_absolute() or not staging.is_absolute():
        raise ValueError("ASR supervisor roots must be absolute")
    vad, asr, ranges = checked_models(models, assets, probe, config["sentinel"])
    streams, origin, metadata = probe_media(runtime, source)
    wire.emit(
        "ready",
        {
            "module": "asr",
            "probe_sha256": config["probe"]["sha256"],
            "proof": {
                "sentinel": config["sentinel"],
                "vad_speech_ranges": len(ranges),
                "audio_streams": len(streams),
                "media_origin_ns": origin,
            },
        },
    )
    ordinal = clock_ordinal = 0
    issues = []
    last_progress = time.monotonic()

    def heartbeat(stage, stream, samples):
        nonlocal last_progress
        if time.monotonic() - last_progress > 2:
            wire.emit(
                "artifact",
                {
                    "module": "asr",
                    "kind": "progress",
                    "stage": stage,
                    "stream": str(stream),
                    "sample": samples,
                },
            )
            last_progress = time.monotonic()

    for stream in streams[:MAX_STREAMS]:
        index = stream["index"]
        pcm, ledger, total, clocks, decode_issues = decode_audio(
            runtime,
            source,
            staging,
            index,
            origin,
            lambda at: heartbeat("decode", index, at),
            waveform=True,
        )
        issues.extend(decode_issues)
        if not clocks:
            continue
        try:
            with ledger.open(encoding="utf-8") as lines:
                for line in lines:
                    clock = json.loads(line)
                    waveform = clock.pop("waveform")
                    first, last = clock["start_sample"], clock["end_sample"]
                    speech_samples = speech_ranges = 0
                    selected = []
                    for a, b in merge_touching_ranges(
                        vad.ranges_pcm(pcm, first, last, lambda at: heartbeat("vad", index, at))
                    ):
                        speech_ranges += 1
                        speech_samples += b - a
                        if len(selected) < 4096:
                            selected.append([a, b])
                        for start, end in bounded_chunks(a, b):
                            if ordinal >= MAX_CHUNKS:
                                issues.append("asr_chunk_budget_unprocessed_tail")
                                break
                            if ordinal >= config["next_ordinal"]:
                                result = asr.extract_raw(pcm, start, end)
                                result.update(
                                    {
                                        "schema": 2,
                                        "clock": clock,
                                        "media_origin_ns": origin,
                                        "mapping_semantics": "final_resampler_pts",
                                        "kind": "chunk",
                                    }
                                )
                                wire.emit(
                                    "artifact",
                                    {
                                        "module": "asr",
                                        "kind": "chunk",
                                        "ordinal": ordinal,
                                        "result": result,
                                        "checkpoint": {"next_ordinal": ordinal + 1},
                                    },
                                )
                            ordinal += 1
                    if clock_ordinal >= config["next_clock"]:
                        result = {
                            "schema": 2,
                            "kind": "clock",
                            "clock": clock,
                            "media_origin_ns": origin,
                            "speech_samples": speech_samples,
                            "speech_ranges": speech_ranges,
                            "vad_semantics": "not_speech_selected_is_not_proof_of_silence",
                            "waveform": waveform,
                            "vad_selected": selected,
                            "vad_known_until": last
                            if speech_ranges <= len(selected)
                            else selected[-1][1],
                        }
                        wire.emit(
                            "artifact",
                            {
                                "module": "asr",
                                "kind": "clock",
                                "ordinal": clock_ordinal,
                                "result": result,
                                "checkpoint": {"next_clock": clock_ordinal + 1},
                            },
                        )
                    clock_ordinal += 1
        finally:
            pcm.unlink(missing_ok=True)
            ledger.unlink(missing_ok=True)
    if len(streams) > MAX_STREAMS:
        issues.append("audio_stream_budget_unprocessed_tracks")
    if not streams:
        issues.append("media_has_no_audio_stream")
    if origin is None:
        issues.append("media_origin_unavailable")
    if config["next_ordinal"] > ordinal or config["next_clock"] > clock_ordinal:
        raise ValueError("ASR checkpoint beyond deterministic final audio blocks")
    wire.emit(
        "done",
        {
            "status": "partial" if issues else "completed",
            "chunks": ordinal,
            "clocks": clock_ordinal,
            "audio_streams": len(streams),
            "issues": issues,
            "media_origin_ns": origin,
            "metadata": metadata,
        },
    )


def run_asr_review(config, wire, assets, source, probe, models):
    """A human-selected range runs every bounded block, including VAD misses."""

    expected = {
        "runtime",
        "staging",
        "source",
        "probe",
        "sentinel",
        "next_ordinal",
        "input_schema",
        "review",
        "media_origin_ns",
    }
    if (
        set(config) != expected
        or config["input_schema"] != 3
        or type(config["next_ordinal"]) is not int
    ):
        raise ValueError("unsupported ASR range contract")
    blocks = config["review"]
    if (
        not isinstance(blocks, list)
        or not 1 <= len(blocks) <= 64
        or not 0 <= config["next_ordinal"] <= len(blocks)
    ):
        raise ValueError("ASR review block budget")
    runtime, staging = Path(config["runtime"]), Path(config["staging"])
    vad, asr, spans = checked_models(models, assets, probe, config["sentinel"])
    streams, origin, _ = probe_media(runtime, source)
    if origin != config["media_origin_ns"]:
        raise ValueError("ASR review media origin changed")
    wire.emit(
        "ready",
        {
            "module": "asr",
            "probe_sha256": config["probe"]["sha256"],
            "proof": {
                "sentinel": config["sentinel"],
                "vad_speech_ranges": len(spans),
                "audio_streams": len(streams),
                "media_origin_ns": origin,
            },
        },
    )
    decoded = {}
    last_progress = time.monotonic()

    def heartbeat(stage, stream, samples):
        nonlocal last_progress
        if time.monotonic() - last_progress > 2:
            wire.emit(
                "artifact",
                {
                    "module": "asr",
                    "kind": "progress",
                    "stage": stage,
                    "stream": str(stream),
                    "sample": samples,
                },
            )
            last_progress = time.monotonic()

    try:
        for ordinal, block in enumerate(blocks):
            clock = block["clock"]
            stream = int(clock["stream"])
            first, last = block["start_sample"], block["end_sample"]
            if (
                type(first) is not int
                or type(last) is not int
                or first < clock["start_sample"]
                or last > clock["end_sample"]
                or not 0 < last - first <= MAX_SAMPLES
            ):
                raise ValueError("invalid ASR range block")
            if stream not in decoded:
                pcm, ledger, _, count, issues = decode_audio(
                    runtime,
                    source,
                    staging,
                    stream,
                    origin,
                    lambda at: heartbeat("decode", stream, at),
                )
                if issues or not count:
                    raise ValueError(
                        "ASR review decode incomplete; frozen clock cannot be confirmed"
                    )
                decoded[stream] = (pcm, ledger)
            pcm, ledger = decoded[stream]
            found = False
            with ledger.open(encoding="utf-8") as lines:
                for line in lines:
                    if json.loads(line) == clock:
                        found = True
                        break
            if not found:
                raise ValueError("ASR review frozen final resampler clock differs")
            if ordinal < config["next_ordinal"]:
                continue
            result = asr.extract_raw(pcm, first, last)
            result.update(
                {
                    "schema": 2,
                    "kind": "chunk",
                    "clock": clock,
                    "media_origin_ns": origin,
                    "mapping_semantics": "final_resampler_pts",
                }
            )
            wire.emit(
                "artifact",
                {
                    "module": "asr",
                    "kind": "chunk",
                    "ordinal": ordinal,
                    "result": result,
                    "checkpoint": {"next_ordinal": ordinal + 1},
                },
            )
    finally:
        for pcm, ledger in decoded.values():
            pcm.unlink(missing_ok=True)
            ledger.unlink(missing_ok=True)
    wire.emit(
        "done",
        {
            "status": "completed",
            "chunks": len(blocks),
            "clocks": 0,
            "audio_streams": len(streams),
            "issues": [],
            "media_origin_ns": origin,
            "metadata": {"range_review": True},
        },
    )
