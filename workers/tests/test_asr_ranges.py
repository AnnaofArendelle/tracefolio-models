# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
import wave

import pytest

from tracefolio_worker.asr.sensevoice import (
    MAX_SAMPLES,
    OVERLAP_SAMPLES,
    bounded_chunks,
    read_pcm16,
)


@pytest.mark.parametrize("length", [1, 16_000, MAX_SAMPLES, MAX_SAMPLES + 1, 2 * 3600 * 16_000])
def test_chunks_cover_complete_selected_interval(length):
    chunks = list(bounded_chunks(1234, 1234 + length))
    assert chunks[0][0] == 1234
    assert chunks[-1][1] == 1234 + length
    assert all(0 < end - start <= MAX_SAMPLES for start, end in chunks)
    assert all(left[1] - right[0] == OVERLAP_SAMPLES for left, right in zip(chunks, chunks[1:]))


def test_waveform_read_keeps_tail_and_rejects_wrong_format(tmp_path: Path):
    p = tmp_path / "audio.wav"
    with wave.open(str(p), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x40" * 100)
    tail = read_pcm16(p, 99, 100)
    assert tail.tolist() == [0.5]
    for start, end in [(0, 101), (100, 100), (-1, 2), (0, MAX_SAMPLES + 1)]:
        with pytest.raises(ValueError):
            read_pcm16(p, start, end)
