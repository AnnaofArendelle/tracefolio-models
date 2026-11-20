# SPDX-License-Identifier: AGPL-3.0-only
"""SenseVoice INT8 bounded waveform adapter; token timestamps are points."""

from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np

from tracefolio_worker.ocr.rapid import Asset

SAMPLE_RATE = 16000
MAX_SAMPLES = 12 * SAMPLE_RATE
OVERLAP_SAMPLES = round(0.6 * SAMPLE_RATE)


def bounded_chunks(start: int, end: int):
    """Keep every selected sample, including a short tail, with bounded overlap."""
    if start < 0 or end < start:
        raise ValueError("invalid audio range")
    while start < end:
        stop = min(end, start + MAX_SAMPLES)
        yield start, stop
        if stop == end:
            break
        start = stop - OVERLAP_SAMPLES


def read_pcm16(path: Path, start: int, end: int) -> np.ndarray:
    if start < 0 or end <= start or end - start > MAX_SAMPLES:
        raise ValueError("ASR input must be a bounded positive range <=12 seconds")
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getframerate() != SAMPLE_RATE
            or audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("expected mono 16kHz PCM16 WAV staging input")
        if end > audio.getnframes():
            raise ValueError("audio range outside staging input")
        audio.setpos(start)
        raw = audio.readframes(end - start)
        if len(raw) != 2 * (end - start):
            raise ValueError("truncated staging audio")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


class SenseVoiceAdapter:
    def __init__(self, model: Asset, tokens: Asset):
        model.verify()
        tokens.verify()
        import sherpa_onnx

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model.path),
            tokens=str(tokens.path),
            num_threads=1,
            language="zh",
            use_itn=False,
        )
        self.ready = False

    def extract(self, path: Path, start: int, end: int) -> dict:
        pcm = read_pcm16(path, start, end)
        return self.extract_pcm(pcm, start, end)

    def extract_raw(self, path: Path, start: int, end: int) -> dict:
        if start < 0 or end <= start or end - start > MAX_SAMPLES:
            raise ValueError("ASR input must be a bounded positive range <=12 seconds")
        with path.open("rb") as audio:
            audio.seek(start * 2)
            raw = audio.read((end - start) * 2)
        if len(raw) != (end - start) * 2:
            raise ValueError("truncated ASR PCM staging input")
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        return self.extract_pcm(pcm, start, end)

    def extract_pcm(self, pcm: np.ndarray, start: int, end: int) -> dict:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, pcm)
        self.recognizer.decode_stream(stream)
        result = stream.result
        tokens = list(result.tokens)
        timestamps = list(result.timestamps)
        issues = []
        points = []
        if (
            len(tokens) != len(timestamps)
            or any(not math.isfinite(t) or t < 0 or t > len(pcm) / SAMPLE_RATE for t in timestamps)
            or any(a > b for a, b in zip(timestamps, timestamps[1:]))
        ):
            issues.append("token_timing_unavailable")
        else:
            points = [
                {"token": token, "sample_point": start + round(t * SAMPLE_RATE)}
                for token, t in zip(tokens, timestamps, strict=True)
            ]
        return {
            "text": result.text,
            "anchor": {
                "kind": "decoded_audio",
                "sample_rate": SAMPLE_RATE,
                "start_sample": start,
                "end_sample": end,
            },
            "token_points": points,
            "timing_semantics": "token_points_not_word_start_end",
            "coverage": "complete",
            "issues": issues,
        }

    def preflight(self, path: Path, start: int, end: int, sentinel: str) -> dict:
        result = self.extract(path, start, end)
        if not sentinel or sentinel not in result["text"]:
            raise ValueError("ASR inference preflight sentinel missing")
        self.ready = True
        return result
