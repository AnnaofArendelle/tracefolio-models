# SPDX-License-Identifier: AGPL-3.0-only
import numpy as np
import pytest
from PIL import Image, ImageOps
from tracefolio_worker.vision.objects import detector_resize, load_image


def test_detector_resize_has_a_cpu_independent_frozen_input():
    import hashlib
    from pathlib import Path

    image = Path(__file__).resolve().parents[2] / "tests/fixtures/vision/group.png"
    pixels, _ = load_image(image)
    resized = detector_resize(pixels, (416, 305))
    assert resized.shape == (305, 416, 3) and resized.dtype == np.uint8
    assert hashlib.sha256(resized.tobytes()).hexdigest() == (
        "2b6b0fb69e2ea914cba1ea8ba8d7217eee9f830a799c35217521730a743a2112"
    )


def test_visual_crop_uses_exif_display_pixels_and_preserves_original(tmp_path):
    pixels = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    source = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(source, exif=exif)
    original = source.read_bytes()
    actual, orientation = load_image(source, [1, 2, 6, 10])
    with Image.open(source) as image:
        expected = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))[2:10, 1:6, ::-1]
    assert orientation == 6 and actual.shape == (8, 5, 3)
    assert np.array_equal(actual, expected)
    assert source.read_bytes() == original
    for invalid in ([0, 0, 9, 12], [1, 1, 1, 2], [True, 0, 2, 3], [0.0, 0, 2, 3], [0, 1, 2]):
        with pytest.raises(ValueError, match="crop"):
            load_image(source, invalid)


def test_pose_skips_loading_for_empty_and_animal_only_inputs(tmp_path, monkeypatch):
    import json
    from tracefolio_worker.protocol import dispatch as module
    from tracefolio_worker.vision import objects, pose

    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "assets": [
                    {"role": "object", "path": "detector", "sha256": "0" * 64},
                    {"role": "pose", "path": "pose", "sha256": "1" * 64},
                ]
            }
        )
    )
    monkeypatch.setattr(module, "staged_file", lambda root, spec: root / spec["path"])
    monkeypatch.setattr(objects, "load_image", lambda *args: (np.zeros((20, 20, 3)), 1))

    def forbidden(*args):
        raise AssertionError("human pose runtime loaded for a nonhuman input")

    monkeypatch.setattr(pose, "PoseAdapter", forbidden)

    class Detector:
        calls = 0
        observations = []

        def __init__(self, *args):
            pass

        def preflight(self, *args):
            return {"objects": [{"label": "person"}]}

        def extract_pixels(self, pixels):
            Detector.calls += 1
            return {
                "objects": self.observations,
                "width": 20,
                "height": 20,
                "coverage": "complete",
                "issues": [],
            }

    monkeypatch.setattr(objects, "ObjectAdapter", Detector)

    class Events:
        def __init__(self):
            self.rows = []

        def emit(self, kind, payload):
            self.rows.append((kind, payload))

    req = {
        "module": "pose",
        "input": {
            "runtime": str(tmp_path),
            "staging": str(tmp_path),
            "source": {"path": "source"},
            "probe": {"path": "probe", "sha256": "2" * 64},
            "sentinel": "person",
            "next_ordinal": 0,
            "input_schema": 2,
            "crop": None,
        },
    }
    for observations in ([], [{"label": "dog"}], [{"label": "bus"}]):
        Detector.observations = observations
        Detector.calls = 0
        events = Events()
        module.dispatch(req, events)
        assert Detector.calls == 1
        assert events.rows[0][1]["proof"]["pose_inferred"] is False
        assert events.rows[1][1]["result"]["poses"] == []
        assert events.rows[1][1]["result"]["objects"] == observations
