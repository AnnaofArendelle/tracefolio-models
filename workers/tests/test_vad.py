# SPDX-License-Identifier: AGPL-3.0-only
import itertools

import pytest

from tracefolio_worker.asr.vad import speech_ranges
from tracefolio_worker.asr.sensevoice import MAX_SAMPLES, bounded_chunks


def test_vad_eof_padding_and_negative_controls():
    assert list(speech_ranges([0.9] * 11, 5137)) == [(0, 5137)]
    assert list(speech_ranges([0.0] * 11, 5137)) == []
    assert list(speech_ranges([0.9] * 3, 1536)) == []
    assert list(speech_ranges([0.9] * 20 + [0.0] * 20, 20480)) == [(0, 13440)]
    # Separate speech intervals retain their padding and the intervening gap.
    probs = [0.9] * 20 + [0.0] * 50 + [0.9] * 20
    assert list(speech_ranges(probs, len(probs) * 512)) == [(0, 13440), (32640, 46080)]


def test_long_speech_stream_is_bounded_and_all_selected_samples_are_routed():
    total = 2 * 3600 * 16_000 + 17
    ranges = speech_ranges(itertools.repeat(0.9, (total + 511) // 512), total)
    previous_end = 0
    for start, end in ranges:
        assert start >= previous_end and end - start <= 20 * 16_000
        blocks = list(bounded_chunks(start, end))
        assert blocks[0][0] == start and blocks[-1][1] == end
        assert all(b - a <= MAX_SAMPLES for a, b in blocks)
        previous_end = end
    assert previous_end == total


def test_invalid_or_incomplete_probabilities_are_errors():
    for probabilities, total in [([0.5], 1024), ([0.5, 0.5], 512), ([float("nan")], 512)]:
        with pytest.raises(ValueError):
            list(speech_ranges(probabilities, total))
