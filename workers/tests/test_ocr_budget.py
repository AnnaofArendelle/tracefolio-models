# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

import pytest

from tracefolio_worker.ocr.rapid import Asset, InputBudgetExceeded, check_dimensions, detector_shape


def test_crop_uses_exif_display_pixels_and_reports_local_coordinates(tmp_path):
    import numpy as np
    from PIL import Image, ImageOps
    from types import SimpleNamespace
    from tracefolio_worker.ocr.rapid import RapidAdapter

    source = Image.new("RGB", (80, 40), "red")
    source.paste("blue", (0, 0, 40, 40))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    source.save(path, exif=exif)
    seen = []
    adapter = RapidAdapter.__new__(RapidAdapter)
    adapter.infer = lambda pixels, *args: (
        seen.append(pixels.copy()) or SimpleNamespace(txts=None, boxes=None, scores=None),
        {"schema": 1, "candidates": [], "unexamined_contours": 0},
    )
    result = adapter.extract(path, [5, 10, 30, 60])
    with Image.open(path) as encoded:
        expected = np.asarray(ImageOps.exif_transpose(encoded).convert("RGB").crop((5, 10, 30, 60)))
    assert np.array_equal(seen[0], expected[:, :, ::-1])
    assert (result["width"], result["height"], result["exif_orientation"]) == (25, 50, 1)
    for invalid in ([0, 0, 41, 80], [0, 0, 40, 81], [True, 0, 20, 20], [0, 0, 0, 10]):
        with pytest.raises(ValueError, match="crop"):
            adapter.extract(path, invalid)


def test_crop_allows_bounded_tensor_but_checks_source_before_decode(tmp_path, monkeypatch):
    from PIL import Image, ImageOps
    from types import SimpleNamespace
    from tracefolio_worker.ocr.rapid import RapidAdapter

    path = tmp_path / "strip.png"
    Image.new("L", (3000, 200)).save(path)
    adapter = RapidAdapter.__new__(RapidAdapter)
    adapter.infer = lambda _, *args: (
        SimpleNamespace(txts=None, boxes=None, scores=None),
        {"schema": 1, "candidates": [], "unexamined_contours": 0},
    )
    assert adapter.extract(path)["processing"]["downscaled"]
    assert adapter.extract(path, [0, 0, 1000, 200])["width"] == 1000
    large = tmp_path / "large.png"
    Image.new("L", (4001, 4000)).save(large)

    def no_decode(*args):
        raise AssertionError("oversized source reached decode")

    monkeypatch.setattr(ImageOps, "exif_transpose", no_decode)
    with pytest.raises(InputBudgetExceeded, match="full decode"):
        adapter.extract(large, [0, 0, 100, 100])


def test_large_or_extreme_aspect_image_uses_bounded_detector():
    for shape in [
        (1500, 850),
        (2800, 220),
        (2500, 3500),
        (4096, 3072),
        (2544, 3296),
        (2550, 3300),
        (16384, 1),
    ]:
        check_dimensions(*shape)
        dw, dh = detector_shape(*shape)
        assert dw > 0 and dh > 0 and dw * dh <= 4_000_000
    # A tiny compressed long strip can otherwise create a huge detector tensor.
    for shape in [(4001, 4000), (20_000, 900), (0, 100)]:
        with pytest.raises((InputBudgetExceeded, ValueError)):
            check_dimensions(*shape)


def test_model_integrity_fails_before_engine_import(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="digest"):
        Asset(model, "0" * 64).verify()
    link = tmp_path / "linked.onnx"
    link.symlink_to(model)
    with pytest.raises(ValueError, match="regular file"):
        Asset(link, "0" * 64).verify()


def test_actual_detector_tensor_box_mapping_and_original_crops():
    import cv2
    import numpy as np
    from rapidocr.ch_ppocr_det.main import TextDetector
    from rapidocr.ch_ppocr_det.utils import DBPostProcess, DetPreProcess
    from rapidocr.main import RapidOCR
    from tracefolio_worker.ocr.rapid import detector_preprocess

    pixels = np.zeros((3300, 2550, 3), np.uint8)
    pixels[800:1400, 600:1700] = [10, 90, 240]
    preprocess = detector_preprocess(736, "min", None, None)
    # Admitted inputs continue through the exact original preprocessing path.
    small = pixels[:900, :1500]
    assert np.array_equal(preprocess(small), DetPreProcess()(small))
    expected_width, expected_height = detector_shape(2550, 3300)
    observed = []

    def inference(tensor):
        observed.append(tensor.shape)
        _, _, height, width = tensor.shape
        prediction = np.zeros((1, 1, height, width), np.float32)
        prediction[:, :, height // 4 : height // 3, width // 4 : width // 2] = 1
        return prediction

    detector = TextDetector.__new__(TextDetector)
    detector.get_preprocess = lambda: preprocess
    detector.session = inference
    detector.postprocess_op = DBPostProcess(
        thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=1.6
    )
    output = detector(pixels)
    assert observed == [(1, 3, expected_height, expected_width)]
    assert expected_width * expected_height <= 4_000_000
    assert output.img is pixels
    assert len(output.boxes) == 1
    box = output.boxes[0]
    # DBPostProcess maps the tensor's box back to the ORIGINAL display pixels.
    assert box[:, 0].max() > expected_width / 2
    assert box[:, 1].max() > expected_height / 3
    engine = RapidOCR.__new__(RapidOCR)
    crop = engine.crop_text_regions(pixels, output.boxes)[0]
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[0] - box[3]))
    assert crop.shape[:2] in ((height, width), (width, height))
    assert np.any(crop[:, :, 2] > 200)
    # Resizing occurs before normalization, preserving the selected exact kernel.
    resized = preprocess.resize(pixels)
    assert np.array_equal(
        resized,
        cv2.resize(pixels, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR_EXACT),
    )


def test_diagnostics_preserve_detector_output_and_scores_follow_geometry():
    import numpy as np
    from rapidocr.ch_ppocr_det.main import TextDetector
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    from tracefolio_worker.ocr.diagnostics import DiagnosticPostProcess

    prediction = np.zeros((1, 1, 200, 300), np.float32)
    prediction[:, :, 20:45, 30:160] = 0.95
    prediction[:, :, 90:115, 30:160] = 0.65
    prediction[:, :, 150:175, 30:160] = 0.4
    prediction[:, :, 185:186, 240:241] = 0.9
    stock = DBPostProcess(thresh=0.3, box_thresh=0.5, use_dilation=False)
    diagnostic = DiagnosticPostProcess(thresh=0.3, box_thresh=0.5, use_dilation=False)
    old_boxes, old_scores = stock(prediction, (400, 600))
    boxes, scores = diagnostic(prediction, (400, 600))
    assert np.array_equal(boxes, old_boxes)
    assert scores == old_scores
    ordered = TextDetector.sorted_boxes(boxes)
    observed_scores = [
        diagnostic.candidates[diagnostic.by_geometry[tuple(box.flat)][0]]["detection_score"]
        for box in ordered
    ]
    assert observed_scores[0] > observed_scores[1]  # top rectangle has higher confidence
    assert scores[0] < scores[1]  # upstream output has the opposite order
    assert {c["reason"] for c in diagnostic.candidates} == {
        "unread",
        "det_low_score",
        "det_small_before_expand",
    }
    diagnostic.max_candidates = 2
    diagnostic(prediction, (400, 600))
    assert len(diagnostic.candidates) == 2
    assert diagnostic.unexamined == 2


def test_fast_detector_uses_tested_960_path_and_single_line_skips_detector():
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from tracefolio_worker.ocr.rapid import RapidAdapter, detector_preprocess

    pixels = np.zeros((1800, 1200, 3), np.uint8)
    dw, dh = detector_shape(1200, 1800, "fast")
    assert (dw, dh) == (640, 960)
    assert np.array_equal(
        detector_preprocess(736, "min", None, None, "fast").resize(pixels),
        cv2.resize(pixels, (dw, dh)),
    )
    observed = []
    adapter = RapidAdapter.__new__(RapidAdapter)
    adapter.engine = SimpleNamespace(
        text_score=0.5,
        cls_and_rotate=lambda images: (images, None),
        recognize_txt=lambda images: (
            observed.append(images[0]) or SimpleNamespace(txts=["一行文字"], scores=[0.9])
        ),
    )
    output, diagnostic = adapter.infer(pixels[:50, :220], "standard", True)
    assert observed[0].shape == (50, 220, 3)
    assert output.txts == ["一行文字"]
    assert diagnostic["candidates"][0]["detection_score"] is None
    assert diagnostic["candidates"][0]["polygon"] == [[0, 0], [220, 0], [220, 50], [0, 50]]

    from tracefolio_worker.ocr.rapid import OCRThresholds

    output, diagnostic = adapter.infer(
        pixels[:50, :220], "standard", True, OCRThresholds(text=0.95)
    )
    assert output.txts == [] and diagnostic["candidates"][0]["reason"] == "low_score"
    output, _ = adapter.infer(pixels[:50, :220], "standard", True)
    assert output.txts == ["一行文字"]  # the next file restores the default


def test_reused_detector_resets_thresholds_before_each_file():
    import numpy as np
    from types import SimpleNamespace
    from tracefolio_worker.ocr.rapid import OCRThresholds, RapidAdapter

    detector = SimpleNamespace(postprocess_op=SimpleNamespace(candidates=[], unexamined=0))
    observed = []
    adapter = RapidAdapter.__new__(RapidAdapter)

    def run(image, operations):
        observed.append(
            (
                detector.postprocess_op.thresh,
                detector.postprocess_op.box_thresh,
                adapter.engine.text_score,
            )
        )
        return SimpleNamespace(boxes=None), None, None, None

    adapter.engine = SimpleNamespace(
        _load_det_model=lambda: detector,
        text_det=detector,
        preprocess_img=lambda pixels: (pixels, {}),
        run_ocr_steps=run,
        build_final_output=lambda *args: None,
    )
    pixels = np.zeros((32, 32, 3), np.uint8)
    adapter.infer(
        pixels,
        thresholds=OCRThresholds.from_request({"detection": 0.12, "box": 0.23, "text": 0.34}),
    )
    adapter.infer(pixels)
    assert observed == [(0.12, 0.23, 0.34), (0.3, 0.5, 0.5)]
