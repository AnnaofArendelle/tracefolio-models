# SPDX-License-Identifier: AGPL-3.0-only
import io
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tracefolio_worker.media.video import frames, probe, schedule, subtitle_cues, subtitles


def test_video_schedule_budgets_and_partial_fine_cells():
    cells, policy = schedule(24 * 3600 * 1_000_000_000)
    assert len(cells) <= 120 and cells[0] == 0 and policy == "budget_activity"
    assert schedule(10_000_000_000, {"start_ns": 500_000_000, "end_ns": 3_500_000_000})[0] == [0, 1]
    with pytest.raises(ValueError):
        schedule(1000_000_000_000, {"start_ns": 0, "end_ns": 121_000_000_000})


def test_vtt_cues_streams_multiline_and_preserves_markup():
    data = b"WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.500\n<b>Chinese</b>\nline 2\n\n"
    assert list(subtitle_cues(io.BytesIO(data))) == [
        (0, 1_000_000_000, 2_500_000_000, "<b>Chinese</b>\nline 2")
    ]
    with pytest.raises(ValueError):
        list(subtitle_cues(io.BytesIO(b"WEBVTT\n\n00:00:02.000 --> 00:00:01.000\na\n\n")))


def runtime(tmp_path):
    explicit = os.environ.get("TRACEFOLIO_TEST_MEDIA_RUNTIME")
    if explicit:
        root = Path(explicit)
        assert all((root / "media/bin" / command).is_file() for command in ["ffmpeg", "ffprobe"])
        return root
    # Decode with production media bytes; host FFmpeg only authors fixtures.
    portable = Path(__file__).resolve().parents[2] / "build/media-runtime/bin"
    if not all((portable / command).is_file() for command in ["ffmpeg", "ffprobe"]):
        pytest.skip("portable FFmpeg not built")
    for command in ["ffmpeg", "ffprobe"]:
        target = tmp_path / "media/bin" / command
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(portable / command)
    return tmp_path


def fixture_ffmpeg():
    command = shutil.which("ffmpeg")
    if not command:
        pytest.skip("system FFmpeg needed only to author bounded fixtures")
    return command


def test_real_video_source_pts_and_text_subtitle(tmp_path):
    rt = runtime(tmp_path)
    source = tmp_path / "source.mkv"
    srt = tmp_path / "cue.srt"
    srt.write_text("1\n00:00:00,200 --> 00:00:01,400\n张雨桐合同\n\n")
    subprocess.run(
        [
            fixture_ffmpeg(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=10:duration=2",
            "-i",
            str(srt),
            "-map",
            "0:v",
            "-map",
            "1:s",
            "-c:v",
            "ffv1",
            "-c:s",
            "srt",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
    )
    streams, origin, duration = probe(rt, source)
    assert origin == 0 and duration == 2_000_000_000
    assert [s["codec_type"] for s in streams] == ["video", "subtitle"]
    selected = list(frames(rt, source, 0, origin, 0, 2_000_000_000))
    assert 9 <= len(selected) <= 11
    assert all(frame["media_time_ns"] == frame["raw_pts_ns"] for frame in selected)
    assert selected[0]["media_time_ns"] == 0
    cues = list(subtitles(rt, source, 1))
    assert cues == [(0, 200_000_000, 1_400_000_000, "张雨桐合同")]


def test_real_video_offset_and_seek_cell_keep_raw_pts(tmp_path):
    rt = runtime(tmp_path)
    source = tmp_path / "offset.mkv"
    subprocess.run(
        [
            fixture_ffmpeg(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=10:duration=4",
            "-vf",
            "setpts=PTS+5/TB",
            "-c:v",
            "ffv1",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
    )
    _, origin, _ = probe(rt, source)
    assert origin == 5_000_000_000
    selected = list(frames(rt, source, 0, origin, 2_000_000_000, 4_000_000_000))
    assert selected and 9 <= len(selected) <= 11
    assert 2_000_000_000 <= selected[0]["media_time_ns"] < 2_200_000_000
    assert all(frame["raw_pts_ns"] == frame["media_time_ns"] + origin for frame in selected)


@pytest.mark.parametrize(
    "extension,encoder,codec", [("mov", "libx264", "h264"), ("webm", "libvpx-vp9", "vp9")]
)
def test_portable_mov_webm_decode_and_source_clock(tmp_path, extension, encoder, codec):
    # System FFmpeg only authors the fixture. All reading uses production bytes.
    rt = runtime(tmp_path)
    source = tmp_path / f"offset.{extension}"
    subprocess.run(
        [
            fixture_ffmpeg(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=10:duration=2",
            "-vf",
            "setpts=PTS+5/TB",
            "-vsync",
            "0",
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
    )
    streams, origin, duration = probe(rt, source)
    assert len(streams) == 1 and streams[0]["codec_type"] == "video"
    assert streams[0]["codec_name"] == codec
    assert origin == 5_000_000_000 and duration >= 2_000_000_000
    selected = list(frames(rt, source, streams[0]["index"], origin, 0, 2_000_000_000))
    assert len(selected) == 10
    assert selected[0]["media_time_ns"] == 0
    assert selected[-1]["media_time_ns"] == 1_800_000_000
    assert [frame["media_time_ns"] for frame in selected] == [i * 200_000_000 for i in range(10)]
    for frame in selected:
        assert frame["raw_pts_ns"] == frame["media_time_ns"] + origin
        numerator, denominator = frame["time_base"]
        assert frame["raw_pts_ns"] == frame["raw_pts"] * numerator * 1_000_000_000 // denominator
        assert (frame["width"], frame["height"]) == (160, 90)
        assert len(frame["pixels"]) == 160 * 90 * 3
        assert frame["sha256"] == hashlib.sha256(frame["pixels"]).hexdigest()
