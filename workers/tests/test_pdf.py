# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdf
import pytest
from PIL import Image

from tracefolio_worker.documents.pdf import (
    MAX_CHARACTERS,
    MAX_SEGMENTS,
    PDFAdapter,
    native_polygon,
    page_coordinates,
    probe_pdf,
    verify_pdfium,
)


FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/pdf"


def native_pdf(stream, width=240, height=1800):
    """Self-authored PDF content; no private source or generated binary fixture."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] ".encode()
        + b"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data, offsets = b"%PDF-1.7\n", [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return (
        data
        + (f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n").encode()
    )


def no_ocr():
    raise AssertionError("native-only PDF unnecessarily loaded OCR")


def test_pdf_native_more_than_256_lines_have_complete_individual_geometry(tmp_path):
    lines = [f"Line-{index:03d} archive evidence" for index in range(300)]
    stream = b"BT /F1 4 Tf 20 1700 Td 5 TL\n"
    stream += b"\n".join(f"({line}) Tj T*".encode() for line in lines) + b"\nET"
    with pdf.PdfDocument(native_pdf(stream)) as document:
        page = document[0]
        textpage = page.get_textpage()
        try:
            original = textpage.get_text_range()
            assert textpage.count_chars() < MAX_CHARACTERS
            assert textpage.count_rects() > 256
            result = PDFAdapter(no_ocr).page(page, 1, 1, tmp_path)
            assert result["coverage"] == "complete" and not result["ocr_required"]
            assert len(result["segments"]) == len(lines)
            assert "".join(segment["text"] for segment in result["segments"]) == original
            previous = 0
            for index, segment in enumerate(result["segments"]):
                anchor = segment["anchor"]
                assert segment["text"].strip() == lines[index]
                assert previous <= anchor["char_start"] < anchor["char_end"]
                assert (
                    textpage.get_text_range(
                        anchor["char_start"], anchor["char_end"] - anchor["char_start"]
                    )
                    == segment["text"]
                )
                previous = anchor["char_end"]
                # Independent tight character boxes establish each complete
                # line's extent, including the last line beyond the old cap.
                boxes = [
                    textpage.get_charbox(i)
                    for i in range(anchor["char_start"], anchor["char_end"])
                    if textpage.get_text_range(i, 1).strip()
                ]
                polygon = anchor["polygon"]
                assert len(polygon) == 4
                assert polygon[0][0] <= min(box[0] for box in boxes) + 0.01
                assert polygon[0][1] >= max(box[3] for box in boxes) - 0.01
                assert polygon[2][0] >= max(box[2] for box in boxes) - 0.01
                assert polygon[2][1] <= min(box[1] for box in boxes) + 0.01
                if index:
                    previous_polygon = result["segments"][index - 1]["anchor"]["polygon"]
                    assert polygon[0][1] < previous_polygon[2][1]
        finally:
            textpage.close()
            page.close()


def test_pdf_native_more_than_256_font_runs_in_one_line_are_not_truncated(tmp_path):
    stream = (
        b"BT 20 30 Td\n"
        + b"\n".join(
            f"/F{index % 2 + 1} {index % 2 + 4} Tf (A) Tj".encode() for index in range(400)
        )
        + b"\nET"
    )
    with pdf.PdfDocument(native_pdf(stream, width=1800, height=80)) as document:
        page = document[0]
        textpage = page.get_textpage()
        try:
            assert textpage.count_rects() > 256
            original = textpage.get_text_range()
            last = textpage.get_charbox(textpage.count_chars() - 1)
            result = PDFAdapter(no_ocr).page(page, 1, 1, tmp_path)
            assert result["coverage"] == "complete" and result["issues"] == []
            assert len(result["segments"]) == 1
            assert result["segments"][0]["text"] == original == "A" * 400
            assert result["segments"][0]["anchor"]["polygon"][2][0] >= last[2] - 0.01
        finally:
            textpage.close()
            page.close()


def test_pdf_native_segment_budget_keeps_unpositioned_tail_text(tmp_path):
    stream = b"BT /F1 2 Tf 20 3400 Td 3 TL\n"
    stream += b"\n".join(f"(Line-{index:04d}) Tj T*".encode() for index in range(1100))
    stream += b"\nET"
    with pdf.PdfDocument(native_pdf(stream, height=3500)) as document:
        page = document[0]
        textpage = page.get_textpage()
        try:
            original = textpage.get_text_range()
            result = PDFAdapter(no_ocr).page(page, 1, 1, tmp_path)
            assert result["coverage"] == "partial"
            assert result["issues"] == ["page_segment_budget_exceeded"]
            assert len(result["segments"]) == MAX_SEGMENTS
            assert "".join(segment["text"] for segment in result["segments"]) == original
            assert "Line-1099" in result["segments"][-1]["text"]
            assert result["segments"][-1]["anchor"]["polygon"] == []
        finally:
            textpage.close()
            page.close()


def test_pdf_native_rectangle_budget_and_invalid_geometry_never_return_prefix_box():
    class TextPage:
        def __init__(self, boxes):
            self.boxes = boxes
            self.read = 0

        def count_rects(self, start, count):
            assert (start, count) == (0, 3)
            return len(self.boxes)

        def get_rect(self, index):
            self.read += 1
            return self.boxes[index]

    textpage = TextPage([(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10)])
    assert native_polygon(textpage, 0, 3, 2) == ([], 0, "native_rectangle_budget_exceeded")
    assert textpage.read == 0
    textpage.boxes[1] = (10, 0, float("nan"), 10)
    assert native_polygon(textpage, 0, 3, 3) == ([], 3, "native_rectangle_geometry_invalid")
    assert textpage.read == 3


def test_pdf_native_sentinel_never_loads_ocr(tmp_path):
    verify_pdfium()

    def forbidden():
        raise AssertionError("native-only PDF unnecessarily loaded OCR")

    with pdf.PdfDocument(probe_pdf()) as document:
        page = document[0]
        try:
            result = PDFAdapter(forbidden).page(page, 1, 1, tmp_path)
        finally:
            page.close()
    assert result["coverage"] == "complete" and not result["ocr_required"]
    assert result["segments"][0]["text"] == "TRACEFOLIO PDF"
    assert result["segments"][0]["origin"] == "native"


def test_pdf_mixed_stale_and_rotated_coordinates(tmp_path):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())

    class OCR:
        def extract(self, source):
            with Image.open(source) as image:
                width, height = image.size
            return {
                "width": width,
                "height": height,
                "issues": [],
                "segments": [
                    {
                        "text": "rendered sentinel",
                        "confidence": 0.9,
                        "anchor": {"polygon": [[10, 10], [100, 10], [100, 100], [10, 100]]},
                    }
                ],
            }

    calls = []

    def create():
        calls.append(True)
        return OCR()

    adapter = PDFAdapter(create)
    for row in manifest["rows"]:
        source = FIXTURES / row["path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
        if row["expected_status"] != "ok":
            with pytest.raises(pdf.PdfiumError):
                pdf.PdfDocument(source)
            continue
        with pdf.PdfDocument(source) as document:
            page = document[0]
            try:
                result = adapter.page(page, 1, 1, tmp_path)
                assert result["ocr_required"]
                assert any(s["origin"] == "ocr" for s in result["segments"])
                for segment in result["segments"]:
                    anchor = segment["anchor"]
                    for point, display in zip(
                        anchor["polygon"], anchor["display_polygon"], strict=True
                    ):
                        back = page_coordinates(
                            page, result["render_width"], result["render_height"], *point
                        )
                        assert all(abs(a - b) <= 1 for a, b in zip(back, display, strict=True))
                if row["id"] == "stale":
                    assert any(
                        "STALE CONTENT" in s["text"] and s["anchor"]["layer"] == "native_unverified"
                        for s in result["segments"]
                    )
            finally:
                page.close()
    assert len(calls) == 1  # one model instance per task, reused across pages
    assert not (tmp_path / "pdf-page.png").exists()


def test_pdf_mixed_budget_exposes_ocr_tail_without_losing_native(tmp_path):
    from tracefolio_worker.documents.pdf import page_coordinates

    stream = b"BT /F1 2 Tf 20 3400 Td 3 TL\n"
    stream += b"\n".join(f"(Line-{index:04d}) Tj T*".encode() for index in range(1100))
    stream += b"\nET"

    class OCR:
        def extract(self, source):
            with Image.open(source) as image:
                width, height = image.size
            polygon = [[10, 10], [100, 10], [100, 30], [10, 30]]
            return {
                "width": width,
                "height": height,
                "issues": [],
                "segments": [
                    {"text": "saved tail", "confidence": 0.9, "anchor": {"polygon": polygon}}
                ],
                "diagnostics": {
                    "schema": 1,
                    "unexamined_contours": 0,
                    "candidates": [
                        {
                            "polygon": polygon,
                            "text": "saved tail",
                            "detection_score": 0.8,
                            "recognition_score": 0.9,
                            "segment": 0,
                            "reason": "recognized",
                        }
                    ],
                },
            }

    with pdf.PdfDocument(native_pdf(stream, height=3500)) as document:
        page = document[0]
        image = pdf.PdfImage.new(document)
        bitmap = pdf.PdfBitmap.from_pil(Image.new("RGB", (2, 2), "white"))
        try:
            image.set_bitmap(bitmap)
            image.set_matrix(pdf.PdfMatrix(2, 0, 0, 2, 0, 0))
            page.insert_obj(image)
            page.gen_content()
            adapter = PDFAdapter(OCR)
            out = adapter.page(page, 1, 1, tmp_path)
            assert len(out["segments"]) == 1024
            assert "Line-1099" in out["segments"][-1]["text"]
            assert out["unprocessed_ocr_segments"] == 1
            assert out["issues"] == ["ocr_segment_budget_exceeded", "page_segment_budget_exceeded"]
            candidate = out["diagnostics"]["candidates"][0]
            assert candidate["text"] == "saved tail" and candidate["segment"] is None
            assert candidate["reason"] == "page_segment_budget"
            region = dict(zip(("left", "bottom", "right", "top"), page.get_bbox()))
            recovered = adapter.page(page, 1, 1, tmp_path, region)
            assert recovered["coverage"] == "complete"
            assert recovered["segments"][0]["text"] == "saved tail"
            point = recovered["segments"][0]["anchor"]["polygon"][0]
            assert page_coordinates(page, out["render_width"], out["render_height"], *point) == [
                10,
                10,
            ]
        finally:
            bitmap.close()
            page.close()
