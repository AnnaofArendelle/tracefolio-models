# SPDX-License-Identifier: MIT
# Copyright (c) 2020-present Silero Team
# Copyright 2026 Tracefolio contributors
#
# Derived from Silero v6.2.3 get_speech_timestamps_from_probs at commit
# 5cd7945676eb32225748052e2e6a0580e4686a08. Modified to stream probabilities and
# completed ranges with bounded state; removed plotting/resampling/UI options.
# Full MIT license: packaging/licenses/Silero-MIT.txt.
"""Official Silero probability/state semantics with bounded stream memory."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import math
import wave

import numpy as np

from tracefolio_worker.ocr.rapid import Asset
from tracefolio_worker.vision.objects import session

RATE = 16_000
WINDOW = 512
PADDING = 3200


def raw_ranges(probabilities: Iterable[float], total: int) -> Iterator[tuple[int, int]]:
    triggered = False
    start = None
    temp_end = prev_end = next_start = 0
    possible_ends = []
    max_speech_samples = RATE * 20 - WINDOW - 2 * PADDING
    count = 0
    for i, probability in enumerate(probabilities):
        count += 1
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("invalid VAD probability")
        current = WINDOW * i
        if current >= total:
            raise ValueError("VAD produced frames beyond audio")
        if probability >= 0.5 and temp_end:
            silence_duration = current - temp_end
            if silence_duration > RATE * 0.098:
                possible_ends.append((temp_end, silence_duration))
            temp_end = 0
            if next_start < prev_end:
                next_start = current
        if probability >= 0.5 and not triggered:
            triggered = True
            start = current
            continue
        if triggered and current - start > max_speech_samples:
            if possible_ends:
                prev_end, duration = max(possible_ends, key=lambda pair: pair[1])
                yield start, prev_end
                next_start = prev_end + duration
                if next_start < prev_end + current:
                    start = next_start
                else:
                    triggered, start = False, None
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                yield start, current
                triggered, start = False, None
                prev_end = next_start = temp_end = 0
                possible_ends = []
                continue
        if probability < 0.35 and triggered:
            if not temp_end:
                temp_end = current
            if current - temp_end < RATE * 0.5:
                continue
            if temp_end - start > RATE * 0.25:
                yield start, temp_end
            triggered, start = False, None
            prev_end = next_start = temp_end = 0
            possible_ends = []
    if count != (total + WINDOW - 1) // WINDOW:
        raise ValueError("VAD frame stream incomplete")
    if start is not None and total - start > RATE * 0.25:
        yield start, total


def speech_ranges(probabilities: Iterable[float], total: int) -> Iterator[tuple[int, int]]:
    if total < 0:
        raise ValueError("invalid decoded audio length")
    pending = None
    for start, end in raw_ranges(probabilities, total):
        if pending is None:
            pending = (max(0, start - PADDING), end)
            continue
        previous_start, previous_end = pending
        silence = start - previous_end
        if silence < 2 * PADDING:
            half = silence // 2
            yield previous_start, previous_end + half
            pending = (max(0, start - half), end)
        else:
            yield previous_start, min(total, previous_end + PADDING)
            pending = (max(0, start - PADDING), end)
    if pending is not None:
        yield pending[0], min(total, pending[1] + PADDING)


class SileroAdapter:
    def __init__(self, model: Asset):
        self.session = session(model)
        if {i.name for i in self.session.get_inputs()} != {"input", "state", "sr"}:
            raise ValueError("unsupported Silero input contract")

    def probabilities(self, path: Path) -> Iterator[float]:
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros((1, 64), np.float32)
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getcomptype(),
            ) != (RATE, 1, 2, "NONE"):
                raise ValueError("VAD requires mono 16kHz PCM16 WAV")
            total = audio.getnframes()
            consumed = 0
            while consumed < total:
                raw = audio.readframes(WINDOW)
                expected = min(WINDOW, total - consumed)
                if len(raw) != expected * 2:
                    raise ValueError("truncated VAD staging audio")
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                consumed += len(samples)
                samples = np.pad(samples, (0, WINDOW - len(samples)))[None]
                inputs = np.concatenate([context, samples], axis=1)
                probability, state = self.session.run(
                    None,
                    {
                        "input": inputs,
                        "state": state,
                        "sr": np.array(RATE, dtype=np.int64),
                    },
                )
                context = inputs[:, -64:]
                yield float(probability.squeeze())

    def probabilities_pcm(self, path: Path, start: int, end: int, progress=None):
        """Raw final resampler PCM, one continuous clock, bounded 512 sample state."""
        if start < 0 or end < start:
            raise ValueError("invalid VAD PCM range")
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros((1, 64), np.float32)
        with path.open("rb") as audio:
            audio.seek(start * 2)
            consumed = start
            while consumed < end:
                expected = min(WINDOW, end - consumed)
                raw = audio.read(expected * 2)
                if len(raw) != expected * 2:
                    raise ValueError("truncated VAD PCM staging input")
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                consumed += expected
                samples = np.pad(samples, (0, WINDOW - len(samples)))[None]
                inputs = np.concatenate([context, samples], axis=1)
                probability, state = self.session.run(
                    None, {"input": inputs, "state": state, "sr": np.array(RATE, dtype=np.int64)}
                )
                context = inputs[:, -64:]
                if progress:
                    progress(consumed)
                yield float(probability.squeeze())

    def ranges_pcm(self, path: Path, start: int, end: int, progress=None):
        for first, last in speech_ranges(
            self.probabilities_pcm(path, start, end, progress), end - start
        ):
            yield start + first, start + last

    def ranges(self, path: Path) -> Iterator[tuple[int, int]]:
        with wave.open(str(path), "rb") as audio:
            total = audio.getnframes()
        yield from speech_ranges(self.probabilities(path), total)
