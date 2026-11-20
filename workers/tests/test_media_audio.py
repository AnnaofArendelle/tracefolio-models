# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path
import subprocess
import wave
import math
import struct

import pytest

from tracefolio_worker.asr.media import Clock, decode_audio, number_ns, probe_media


def test_pts_and_resampler_clocks(tmp_path):
    # Uses the portable production FFmpeg, without inference or benchmarks.
    root = Path(__file__).resolve().parents[2]
    runtime = root / "build/media-runtime"
    if not (runtime / "bin/ffmpeg").exists():
        pytest.skip("portable FFmpeg not built")
    (tmp_path / "media/bin").mkdir(parents=True)
    for tool in ("ffmpeg", "ffprobe"):
        (tmp_path / "media/bin" / tool).symlink_to(runtime / "bin" / tool)
    source = tmp_path / "gap.mkv"
    original = tmp_path / "source.wav"
    with wave.open(str(original), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(
            b"".join(
                struct.pack("<h", round(3000 * math.sin(2 * math.pi * 300 * i / 48000)))
                for i in range(96000)
            )
        )
    subprocess.run(
        [
            str(runtime / "bin/ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(original),
            "-af",
            r"asetpts=PTS+if(gte(T\,1)\,2/TB\,0)",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
    )
    streams, origin, _ = probe_media(tmp_path, source)
    pcm, ledger, samples, count, issues = decode_audio(tmp_path, source, tmp_path, 0, origin)
    clocks = [json.loads(s) for s in ledger.read_text().splitlines()]
    assert not issues
    assert samples == 32000 == pcm.stat().st_size // 2
    assert count == len(clocks) >= 2
    assert clocks[0]["start_sample"] == 0 and clocks[-1]["end_sample"] == samples
    assert all(a["end_sample"] == b["start_sample"] for a, b in zip(clocks, clocks[1:]))
    # Output timestamp jumps by ~2 seconds; actual sample count is unchanged.
    jumps = [c for c in clocks if c["reason"] == "pts_discontinuity"]
    assert any(
        c["media_start_ns"] - c["start_sample"] * 1000000000 // 16000 >= 1900000000 for c in jumps
    )
    assert len(streams) == 1


def test_resampler_pts_unknown_are_not_guessed():
    assert number_ns("N/A") is None
    assert number_ns("NaN") is None
    assert number_ns("-0.064") == -64000000
    assert number_ns("1.000000001") == 1000000001
    assert Clock("0", 0, 512, None, None, "media_pts_unavailable", 1).media_start_ns is None


def test_corrupt_wav_tail_with_success_exit_is_explicitly_partial(tmp_path):
    root = Path(__file__).resolve().parents[2]
    runtime = root / "build/media-runtime"
    if not (runtime / "bin/ffmpeg").exists():
        pytest.skip("portable FFmpeg not built")
    (tmp_path / "media/bin").mkdir(parents=True)
    for tool in ("ffmpeg", "ffprobe"):
        (tmp_path / "media/bin" / tool).symlink_to(runtime / "bin" / tool)
    original = root / "packaging/probes/asr.wav"
    source = tmp_path / "corrupt-tail.wav"
    source.write_bytes(original.read_bytes()[:-3])
    with wave.open(str(original)) as audio:
        expected = audio.getnframes() - 2
    diagnostic = subprocess.run(
        [
            str(runtime / "bin/ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    assert diagnostic.returncode == 0
    assert b"Invalid PCM packet" in diagnostic.stderr
    _, origin, _ = probe_media(tmp_path, source)
    pcm, ledger, total, count, issues = decode_audio(tmp_path, source, tmp_path, 0, origin)
    assert issues == ["audio_decode_error_partial_tail"]
    clocks = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert count == len(clocks) == 1
    assert clocks[-1]["end_sample"] == total == expected == pcm.stat().st_size // 2


def test_decode_formats_and_budget_keep_only_committed_frames(tmp_path):
    root = Path(__file__).resolve().parents[2]
    runtime = root / "build/media-runtime"
    if not (runtime / "bin/ffmpeg").exists():
        pytest.skip("portable FFmpeg not built")
    (tmp_path / "media/bin").mkdir(parents=True)
    for tool in ("ffmpeg", "ffprobe"):
        (tmp_path / "media/bin" / tool).symlink_to(runtime / "bin" / tool)
    fixture = root / "tests/fixtures/audio"
    paths = [root / "packaging/probes/asr.wav", fixture / "clean.mp3", fixture / "clean.amr"]
    for path in paths:
        assert path.is_file(), "committed audio fixture missing"
        target = tmp_path / path.name
        target.mkdir()
        _, origin, _ = probe_media(tmp_path, path)
        pcm, ledger, total, count, issues = decode_audio(tmp_path, path, target, 0, origin)
        assert not issues and total > 64000 and count > 0
        clocks = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert clocks[-1]["end_sample"] == total == pcm.stat().st_size // 2
        if path.suffix == ".mp3":
            assert any(c["reason"] == "negative_media_pts_unavailable" for c in clocks)
            positive = [c for c in clocks if c["media_start_ns"] is not None]
            assert positive and positive[-1]["end_sample"] == total
            assert all(c["media_start_ns"] >= 0 for c in positive)
    # Budget truncation retains exact full-frame prefixes and explicit tail issue.
    target = tmp_path / "limited"
    target.mkdir()
    pcm, ledger, total, count, issues = decode_audio(
        tmp_path, paths[0], target, 0, 0, budget=400000
    )
    assert not issues  # This short sample fits.
    target = tmp_path / "no-space"
    target.mkdir()
    _, _, total, count, issues = decode_audio(tmp_path, paths[0], target, 0, 0, budget=1)
    assert total == count == 0 and issues == ["decoded_audio_disk_budget_unavailable"]


def test_unknown_clock_and_sparse_large_pcm_keep_memory_bounded(tmp_path):
    from tracefolio_worker.asr.sensevoice import SenseVoiceAdapter

    # No full recording read: raw PCM reader seeks directly to the bounded tail.
    path = tmp_path / "large.pcm"
    with path.open("wb") as file:
        file.seek(2 * 16000 * 8 * 3600)
        file.write(b"\x00\x40" * 100)
    adapter = object.__new__(SenseVoiceAdapter)
    adapter.extract_pcm = lambda pcm, start, end: (pcm.tolist(), start, end)
    start = 16000 * 8 * 3600
    samples, first, last = adapter.extract_raw(path, start, start + 100)
    assert first == start and last == start + 100 and samples == [0.5] * 100


def test_vad_forced_long_speech_boundaries_retain_overlap():
    import itertools
    from tracefolio_worker.asr.media import merge_touching_ranges
    from tracefolio_worker.asr.vad import speech_ranges
    from tracefolio_worker.asr.sensevoice import bounded_chunks, OVERLAP_SAMPLES, MAX_SAMPLES

    total = 16000 * 90 + 17
    selected = merge_touching_ranges(
        speech_ranges(itertools.repeat(0.9, (total + 511) // 512), total)
    )
    blocks = [block for first, last in selected for block in bounded_chunks(first, last)]
    assert blocks[0][0] == 0 and blocks[-1][1] == total
    assert all(a[1] - b[0] == OVERLAP_SAMPLES for a, b in zip(blocks, blocks[1:]))
    assert all(0 < last - first <= MAX_SAMPLES for first, last in blocks)


def test_review_runs_vad_missed_range_skips_committed_inference_and_checks_clock(
    tmp_path, monkeypatch
):
    import hashlib
    import shutil
    from tracefolio_worker.asr import media

    root = Path(__file__).resolve().parents[2]
    runtime = root / "build/media-runtime"
    if not (runtime / "bin/ffmpeg").exists():
        pytest.skip("portable FFmpeg not built")
    (tmp_path / "media/bin").mkdir(parents=True)
    for tool in ("ffmpeg", "ffprobe"):
        (tmp_path / "media/bin" / tool).symlink_to(runtime / "bin" / tool)
    source = tmp_path / "source.wav"
    shutil.copyfile(root / "packaging/probes/asr.wav", source)
    _, origin, _ = probe_media(tmp_path, source)
    initial = tmp_path / "initial"
    initial.mkdir()
    _, ledger, total, _, issues = decode_audio(tmp_path, source, initial, 0, origin)
    assert not issues and total > 32000
    clock = json.loads(ledger.read_text().splitlines()[0])
    assert clock["end_sample"] > 32000
    calls = []

    class FakeVAD:
        def __init__(self, _):
            pass

        def ranges(self, _):
            return [(0, 16000)]

        def ranges_pcm(self, *_):
            raise AssertionError("human range review cannot skip audio by VAD")

    class FakeASR:
        def __init__(self, *_):
            pass

        def preflight(self, *_):
            pass

        def extract_raw(self, pcm, first, last):
            with pcm.open("rb") as audio:
                audio.seek(first * 2)
                assert len(audio.read((last - first) * 2)) == (last - first) * 2
            calls.append((first, last))
            return {"text": "人工指定补测"}

    class Sink:
        def __init__(self):
            self.events = []

        def emit(self, kind, payload):
            self.events.append((kind, payload))

    monkeypatch.setattr(media, "SileroAdapter", FakeVAD)
    monkeypatch.setattr(media, "SenseVoiceAdapter", FakeASR)
    descriptor = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    config = {
        "runtime": str(tmp_path),
        "staging": str(tmp_path),
        "source": descriptor,
        "probe": descriptor,
        "sentinel": "交易",
        "input_schema": 3,
        "next_ordinal": 1,
        "media_origin_ns": origin,
        "review": [
            {"clock": clock, "start_sample": 0, "end_sample": 16000},
            {"clock": clock, "start_sample": 16000, "end_sample": 32000},
        ],
    }
    sink = Sink()
    media.run_asr_review(
        config, sink, {"vad": "fake", "asr": "fake", "tokens": "fake"}, source, source, {}
    )
    assert calls == [(16000, 32000)]
    artifact = [payload for kind, payload in sink.events if kind == "artifact"]
    assert len(artifact) == 1 and artifact[0]["ordinal"] == 1
    assert sink.events[-1][0] == "done" and sink.events[-1][1]["chunks"] == 2
    assert not list(tmp_path.glob("audio-*"))
    config["review"][1]["clock"] = {**clock, "frame_count": clock["frame_count"] + 1}
    with pytest.raises(ValueError, match="frozen final resampler clock differs"):
        media.run_asr_review(
            config, Sink(), {"vad": "fake", "asr": "fake", "tokens": "fake"}, source, source, {}
        )
    assert calls == [(16000, 32000)]
