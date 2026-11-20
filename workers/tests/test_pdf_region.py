# SPDX-License-Identifier: AGPL-3.0-only
import math
from pathlib import Path

import pypdfium2 as pdf
import pytest
from PIL import Image

from tracefolio_worker.documents.pdf import (
    PDFAdapter,
    SCALE,
    page_coordinates,
    probe_pdf,
    render_region,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/pdf"


class OCR:
    def __init__(self):
        self.sizes = []

    def extract(self, source):
        with Image.open(source) as image:
            width, height = image.size
        self.sizes.append((width, height))
        return {
            "width": width,
            "height": height,
            "issues": [],
            "segments": [
                {
                    "text": "区域",
                    "confidence": 0.9,
                    "anchor": {
                        "polygon": [
                            [1, 1],
                            [width - 1, 1],
                            [width - 1, height - 1],
                            [1, height - 1],
                        ]
                    },
                }
            ],
        }


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pdf_region_renders_small_rotated_crop_before_allocation(tmp_path, rotation):
    with pdf.PdfDocument(probe_pdf()) as document:
        page = document[0]
        try:
            # A full bitmap is too large for OCR, but a 100x40 point region is cheap.
            page.set_mediabox(-50, -20, 20000, 20000)
            page.set_cropbox(-30, -10, 19000, 19000)
            page.set_rotation(rotation)
            region = {"left": 0, "bottom": 20, "right": 100, "top": 60}
            ocr = OCR()
            result = PDFAdapter(lambda: ocr).page(page, 1, 1, tmp_path, region)
            assert result["coverage"] == "complete"
            assert result["segments"][0]["origin"] == "ocr"
            assert ocr.sizes[0] in [(150, 60), (60, 150)]
            anchor = result["segments"][0]["anchor"]
            for point, display in zip(anchor["polygon"], anchor["display_polygon"], strict=True):
                back = page_coordinates(
                    page, result["render_width"], result["render_height"], *point
                )
                assert all(abs(a - b) <= 1 for a, b in zip(back, display, strict=True))
                assert -1 <= point[0] <= 101 and 19 <= point[1] <= 61
            assert not (tmp_path / "pdf-page.png").exists()
        finally:
            page.close()


def test_pdf_region_uses_effective_visible_bbox_not_raw_cropbox():
    with pdf.PdfDocument(probe_pdf()) as document:
        page = document[0]
        try:
            page.set_cropbox(-100, -100, 1000, 1000)
            width, height = page.get_size()
            with pytest.raises(ValueError, match="outside visible"):
                render_region(
                    page,
                    math.ceil(width * SCALE),
                    math.ceil(height * SCALE),
                    {"left": -1, "bottom": 0, "right": 5, "top": 5},
                )
        finally:
            page.close()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_detailed_units_render_real_high_scale_pixels_and_continue(tmp_path, rotation):
    with pdf.PdfDocument(probe_pdf()) as document:
        page = document[0]
        try:
            page.set_mediabox(0, 0, 2400, 800)
            page.set_rotation(rotation)
            ocr = OCR()
            adapter = PDFAdapter(lambda: ocr)
            options = {"depth": "detailed", "single_line": False, "tile_start": 0}
            units, next_tile, total = adapter.units(page, None, options)
            assert len(units) == 10 and next_tile == 9 and total > 9
            region, scale, raster, supplement = units[1]
            result = adapter.page(page, 1, 1, tmp_path, region, scale=scale, raster_region=raster)
            assert scale == 4 and supplement
            assert ocr.sizes == [(1536, 1536)]
            assert result["render_scale"] == 4
            assert result["segments"][0]["anchor"]["polygon"]
            options["tile_start"] = next_tile
            rest, end, same_total = adapter.units(page, None, options)
            assert same_total == total and end == min(18, total)
            assert all(u[3] for u in rest)
            assert rest[0][2] != units[-1][2]
        finally:
            page.close()


def test_pdf_batch_protocol_resumes_regions_and_reuses_preflight(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    from dataclasses import asdict
    from tracefolio_worker.protocol import dispatch as module

    source = tmp_path / "source.pdf"
    source.write_bytes(probe_pdf())
    probe = tmp_path / "probe.png"
    Image.new("RGB", (20, 20), "white").save(probe)
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "assets": [
                    {"role": role, "path": role, "sha256": "0" * 64}
                    for role in ("det", "rec", "cls", "dictionary")
                ]
            }
        )
    )
    calls, preflights = [], []

    class Adapter:
        def __init__(self, models):
            pass

        def preflight(self, path, sentinel):
            preflights.append(sentinel)
            return {"segments": []}

        def extract(self, path, *, thresholds, **options):
            with Image.open(path) as image:
                width, height = image.size
            calls.append(asdict(thresholds))
            return {
                "width": width,
                "height": height,
                "coverage": "complete",
                "issues": [],
                "segments": [],
            }

    monkeypatch.setattr(module, "RapidAdapter", Adapter)
    regions = [
        {"left": 0, "bottom": 0, "right": 100, "top": 70},
        {"left": 100, "bottom": 0, "right": 230, "top": 70},
    ]
    request = {
        "module": "pdf",
        "input": {
            "runtime": str(tmp_path),
            "staging": str(tmp_path),
            "source": {
                "path": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "probe": {"path": probe.name, "sha256": hashlib.sha256(probe.read_bytes()).hexdigest()},
            "sentinel": "TRACEFOLIO",
            "next_ordinal": 0,
            "next_unit": 0,
            "next_region": 0,
            "input_schema": 2,
            "first_page": 1,
            "last_page": 1,
            "pdf_region": None,
            "pdf_regions": regions,
            "options": {
                "depth": "standard",
                "single_line": False,
                "tile_start": 0,
                "thresholds": {"detection": 0.12, "box": 0.23, "text": 0.34},
            },
        },
    }

    class Events:
        def __init__(self, interrupt=False):
            self.rows = []
            self.interrupt = interrupt

        def emit(self, kind, payload):
            self.rows.append((kind, payload))
            if kind == "artifact" and self.interrupt:
                raise InterruptedError("saved before ACK")

    models = {}
    events = Events(interrupt=True)
    with pytest.raises(InterruptedError, match="saved before ACK"):
        module.dispatch(copy.deepcopy(request), events, models)
    saved = events.rows[-1][1]
    assert saved["checkpoint"] == {"next_ordinal": 0, "next_region": 1, "next_unit": 0}
    assert saved["result"]["region_index"] == 0
    resumed = copy.deepcopy(request)
    resumed["input"].update(saved["checkpoint"])
    events = Events()
    module.dispatch(resumed, events, models)
    output = [row for kind, row in events.rows if kind == "artifact"]
    assert [row["result"]["region_index"] for row in output] == [1]
    assert output[0]["checkpoint"] == {"next_ordinal": 1, "next_region": 0, "next_unit": 0}
    assert output[0]["result"]["pdf_region"] == regions[1]
    # The next file gets defaults while the loaded OCR/preflight remains warm.
    next_file = copy.deepcopy(request)
    del next_file["input"]["options"]["thresholds"]
    module.dispatch(next_file, Events(), models)
    assert preflights == ["TRACEFOLIO"]
    assert (
        calls
        == [{"detection": 0.12, "box": 0.23, "text": 0.34}] * 2
        + [{"detection": 0.3, "box": 0.5, "text": 0.5}] * 2
    )
