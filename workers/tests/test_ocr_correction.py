# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace

import numpy as np
import pypdfium2 as pdf
import pytest
from PIL import Image

from tracefolio_worker.ocr.rapid import RapidAdapter
from tracefolio_worker.ocr.correction import validate_correction
from tracefolio_worker.documents.pdf import PDFAdapter, probe_pdf, page_coordinates


def adapter_with_box(seen):
    adapter = RapidAdapter.__new__(RapidAdapter)

    def infer(pixels, *_):
        seen.append(pixels.copy())
        h, w = pixels.shape[:2]
        box = [[0, 0], [w, 0], [w, h], [0, h]]
        return SimpleNamespace(txts=["测试"], scores=[0.9], boxes=np.array([box])), {
            "schema": 1,
            "unexamined_contours": 0,
            "candidates": [
                {
                    "polygon": box,
                    "text": "测试",
                    "recognition_score": 0.9,
                    "detection_score": 0.8,
                    "reason": "recognized",
                    "segment": 0,
                }
            ],
        }

    adapter.infer = infer
    return adapter


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_manual_correction_pixels_candidates_and_original_mapping(tmp_path, turns):
    image = Image.new("RGB", (400, 200), "red")
    image.paste("blue", (0, 0, 200, 200))
    path = tmp_path / "source.png"
    image.save(path)
    original = path.read_bytes()
    seen = []
    correction = {"quad": [[0.1, 0.1], [0.9, 0.2], [0.8, 0.9], [0.2, 0.8]], "turns": turns}
    validate_correction(correction)
    out = adapter_with_box(seen).extract(path, correction=correction)
    expected = np.roll(np.array(correction["quad"]) * [400, 200], turns, axis=0)
    assert np.allclose(out["segments"][0]["anchor"]["polygon"], expected, atol=1e-5)
    assert out["segments"][0]["anchor"]["polygon"] == out["diagnostics"]["candidates"][0]["polygon"]
    assert out["processing"]["input_width"] == seen[0].shape[1]
    assert (out["width"], out["height"]) == (400, 200)
    assert path.read_bytes() == original
    # The center retains actual source pixels after the manual transform.
    assert seen[0][seen[0].shape[0] // 2, seen[0].shape[1] // 4].max() > 200


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_corrected_pdf_high_scale_tile_is_bounded_and_maps_back(tmp_path, rotation):
    with pdf.PdfDocument(probe_pdf()) as document:
        page = document[0]
        page.set_mediabox(0, 0, 2400, 800)
        page.set_rotation(rotation)
        seen = []
        adapter = PDFAdapter(lambda: adapter_with_box(seen))
        correction = {"quad": [[0.1, 0.1], [0.9, 0.2], [0.8, 0.9], [0.2, 0.8]], "turns": 2}
        options = {
            "depth": "detailed",
            "single_line": False,
            "tile_start": 0,
            "correction": correction,
        }
        units, _, total = adapter.units(page, None, options)
        assert total > 1
        region, scale, raster, _ = units[1]
        out = adapter.page(
            page, 1, 1, tmp_path, region, scale=scale, raster_region=raster, correction=correction
        )
        assert seen[0].shape[:2] == (1536, 1536)
        assert out["coverage"] == "complete"
        anchor = next(s["anchor"] for s in out["segments"] if s["origin"] == "ocr")
        for source, display in zip(anchor["polygon"], anchor["display_polygon"]):
            mapped = page_coordinates(page, out["render_width"], out["render_height"], *source)
            assert np.allclose(mapped, display, atol=1)
        page.close()


def test_manual_selection_rejects_crossed_or_outside_points():
    for quad in ([[0, 0], [1, 1], [1, 0], [0, 1]], [[-0.1, 0], [1, 0], [1, 1], [0, 1]]):
        with pytest.raises(ValueError):
            validate_correction({"quad": quad, "turns": 0})
