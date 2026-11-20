# SPDX-License-Identifier: AGPL-3.0-only
"""PDFium page evidence with lazy OCR and bounded, explicitly partial coverage."""

from __future__ import annotations

import ctypes
import math
import re
from pathlib import Path

from tracefolio_worker.ocr.rapid import InputBudgetExceeded, check_dimensions

MAX_PAGES = 5000
MAX_CHARACTERS = 65536
MAX_SEGMENTS = 1024
MAX_NATIVE_RECTANGLES = MAX_CHARACTERS
SCALE = 1.5


def probe_pdf() -> bytes:
    """Self-authored native text sentinel, no fonts/assets outside the runtime."""
    stream = b"BT /F1 12 Tf 20 40 Td (TRACEFOLIO PDF) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 80] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data = b"%PDF-1.7\n"
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return data + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


def verify_pdfium():
    import pypdfium2 as pdf

    with pdf.PdfDocument(probe_pdf()) as document:
        page = document[0]
        text = page.get_textpage()
        try:
            if text.get_text_range().strip() != "TRACEFOLIO PDF":
                raise ValueError("PDFium text preflight failed")
        finally:
            text.close()
            page.close()


def page_coordinates(page, width, height, x, y, *, inverse=False):
    import pypdfium2.raw as raw

    if inverse:
        a, b = ctypes.c_double(), ctypes.c_double()
        ok = raw.FPDF_DeviceToPage(page.raw, 0, 0, width, height, 0, round(x), round(y), a, b)
    else:
        a, b = ctypes.c_int(), ctypes.c_int()
        ok = raw.FPDF_PageToDevice(page.raw, 0, 0, width, height, 0, x, y, a, b)
    if not ok:
        raise ValueError("PDF page coordinate conversion failed")
    return [a.value, b.value]


def native_lines(textpage, text, count):
    """Map complete CR/LF-preserving lines to PDFium's internal char list.

    Text indices count UTF-16 units and can differ from character indices.
    Only line boundaries are translated; a range read verifies each mapping.
    Blank separators stay attached to neighboring text rather than disappearing.
    """
    import pypdfium2.raw as raw

    lines, blank = [], ""
    for match in re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", text):
        line = match.group()
        if not line:
            continue
        if not line.strip():
            blank += line
        else:
            lines.append(blank + line)
            blank = ""
    if blank and lines:
        lines[-1] += blank
    if "".join(lines) != text:
        raise ValueError("native line text mapping differs")
    truncated = len(lines) > MAX_SEGMENTS
    if truncated:
        # Preserve all bounded text, but do not pretend a multiline tail is a
        # precise line box when the page's segment budget has been reached.
        lines[MAX_SEGMENTS - 1 :] = ["".join(lines[MAX_SEGMENTS - 1 :])]
    first = next(
        (
            index
            for index in range(count)
            if raw.FPDFText_GetTextIndexFromCharIndex(textpage.raw, index) >= 0
        ),
        None,
    )
    if first is None:
        raise ValueError("native text has no character mapping")
    text_index = raw.FPDFText_GetTextIndexFromCharIndex(textpage.raw, first)
    out, previous_end = [], 0
    for ordinal, line in enumerate(lines):
        units = len(line.encode("utf-16-le")) // 2
        start = raw.FPDFText_GetCharIndexFromTextIndex(textpage.raw, text_index)
        end = raw.FPDFText_GetCharIndexFromTextIndex(textpage.raw, text_index + units - 1) + 1
        if not (previous_end <= start < end <= count):
            raise ValueError("native line character mapping is invalid")
        if textpage.get_text_range(start, end - start) != line:
            raise ValueError("native line character range differs")
        out.append((start, end, line, truncated and ordinal == MAX_SEGMENTS - 1))
        previous_end = end
        text_index += units
    return out, truncated


def native_polygon(textpage, start, end, remaining):
    """Union every rectangle for one exact line, never a truncated prefix."""
    import pypdfium2 as pdf

    try:
        rects = textpage.count_rects(start, end - start)
    except pdf.PdfiumError:
        return [], 0, "native_rectangle_geometry_invalid"
    if rects > remaining:
        return [], 0, "native_rectangle_budget_exceeded"
    if not rects:
        return [], 0, "native_text_without_rectangle"
    left, bottom, right, top = math.inf, math.inf, -math.inf, -math.inf
    valid = True
    for index in range(rects):
        try:
            box = textpage.get_rect(index)
        except pdf.PdfiumError:
            valid = False
            continue
        if (
            not all(math.isfinite(value) and abs(value) <= 1000000 for value in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            valid = False
            continue
        left, bottom = min(left, box[0]), min(bottom, box[1])
        right, top = max(right, box[2]), max(top, box[3])
    if not valid:
        return [], rects, "native_rectangle_geometry_invalid"
    return [[left, top], [right, top], [right, bottom], [left, bottom]], rects, None


class PDFAdapter:
    def __init__(self, ocr_factory):
        self.ocr_factory = ocr_factory
        self.ocr = None

    def units(self, page, region, options):
        if options["depth"] != "detailed":
            return [(region, SCALE, None, False)], 0, 0
        width, height = page.get_size()
        rw, rh = math.ceil(width * 4), math.ceil(height * 4)
        _, offset, size = render_region(page, rw, rh, region, 4)
        correction = options.get("correction")
        if correction is not None:
            from tracefolio_worker.ocr.correction import correction_geometry

            _, size = correction_geometry(correction, *size)
            offset = (0, 0)
        xs = list(range(offset[0], max(offset[0] + 1, offset[0] + size[0] - 1536), 1344))
        ys = list(range(offset[1], max(offset[1] + 1, offset[1] + size[1] - 1536), 1344))
        xs = sorted(set(xs + [max(offset[0], offset[0] + size[0] - 1536)]))
        ys = sorted(set(ys + [max(offset[1], offset[1] + size[1] - 1536)]))
        total = len(xs) * len(ys)
        start = options["tile_start"]
        if start >= total and start != 0:
            raise ValueError("PDF continuation outside detailed grid")
        stop = min(start + 9, total)
        units = [(region, SCALE, None, False)] if start == 0 else []
        for index in range(start, stop):
            x, y = xs[index % len(xs)], ys[index // len(xs)]
            right, bottom = min(offset[0] + size[0], x + 1536), min(offset[1] + size[1], y + 1536)
            if correction is not None:
                units.append((region, 4, (x, y, right, bottom), True))
                continue
            points = [
                page_coordinates(page, rw, rh, a, b, inverse=True)
                for a, b in ((x, y), (right, bottom))
            ]
            bounds = {
                "left": min(p[0] for p in points),
                "right": max(p[0] for p in points),
                "bottom": min(p[1] for p in points),
                "top": max(p[1] for p in points),
            }
            units.append((bounds, 4, (x, y, right, bottom), True))
        return units, stop, total

    def page(
        self,
        page,
        number: int,
        total: int,
        staging: Path,
        region=None,
        *,
        scale=SCALE,
        depth="standard",
        single_line=False,
        raster_region=None,
        correction=None,
        thresholds=None,
    ):
        import pypdfium2.raw as raw

        width, height = page.get_size()
        if not all(math.isfinite(v) and 0 < v <= 100000 for v in (width, height)):
            raise ValueError("invalid PDF page dimensions")
        raster_w, raster_h = math.ceil(width * scale), math.ceil(height * scale)
        result = {
            "schema": 1,
            "page": number,
            "total_pages": total,
            "width": width,
            "height": height,
            "rotation": page.get_rotation(),
            "crop_box": list(page.get_cropbox()),
            "page_box": list(page.get_bbox()),
            "pdf_region": region,
            "render_scale": scale,
            "render_width": raster_w,
            "render_height": raster_h,
            "segments": [],
            "coverage": "complete",
            "issues": [],
            "ocr_required": False,
        }
        segments, issues = result["segments"], result["issues"]
        textpage = page.get_textpage() if region is None else None
        try:
            count = textpage.count_chars() if textpage is not None else 0
            if count > MAX_CHARACTERS:
                issues.append("native_character_budget_exceeded")
            bounded_count = min(count, MAX_CHARACTERS)
            native_text = textpage.get_text_range(0, bounded_count) if bounded_count else ""
            if native_text.strip():
                try:
                    lines, truncated = native_lines(textpage, native_text, bounded_count)
                except ValueError:
                    # Keep the original extracted text searchable, without an
                    # invented position when PDFium's two index views disagree.
                    issues.append("native_text_mapping_unverified")
                    lines, truncated = [(0, bounded_count, native_text, True)], False
                if truncated:
                    issues.append("page_segment_budget_exceeded")
                remaining = MAX_NATIVE_RECTANGLES
                for start, end, text, without_geometry in lines:
                    polygon = []
                    if not without_geometry:
                        polygon, used, issue = native_polygon(textpage, start, end, remaining)
                        remaining -= used
                        if issue:
                            issues.append(issue)
                    segments.append(
                        {
                            "text": text,
                            "origin": "native",
                            "anchor": {
                                "kind": "pdf",
                                "unit": "pdf_point",
                                "page": number,
                                "polygon": polygon,
                                "display_polygon": [
                                    page_coordinates(page, raster_w, raster_h, *point)
                                    for point in polygon
                                ],
                                "char_start": start,
                                "char_end": end,
                                "layer": "native_unverified",
                            },
                        }
                    )
        finally:
            if textpage is not None:
                textpage.close()
        image_present = False
        for index, obj in enumerate(page.get_objects()):
            if index >= 4096:
                issues.append("page_object_budget_exceeded")
                image_present = True
                break
            if obj.type == raw.FPDF_PAGEOBJ_IMAGE:
                image_present = True
                break
        # Nonempty text does not prove the visual page is covered. Every image
        # containing page is rendered, including mixed and stale hidden layers.
        required = (
            correction is not None
            or region is not None
            or image_present
            or not native_text.strip()
            or "\ufffd" in native_text
        )
        result["ocr_required"] = required
        if required:
            crop, offset, size = render_region(page, raster_w, raster_h, region, scale)
            prepared_geometry, corrected_crop = None, None
            render_size = size
            if correction is not None:
                import cv2
                import numpy as np
                from tracefolio_worker.ocr.correction import correction_geometry

                matrix, corrected_size = correction_geometry(correction, *size)
                x0, y0, x1, y1 = raster_region or (0, 0, *corrected_size)
                if raster_region is not None:
                    corrected_crop = dict(zip(("left", "top", "right", "bottom"), raster_region))
                original = cv2.perspectiveTransform(
                    np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float64),
                    np.linalg.inv(matrix),
                )[0]
                # Render a two-pixel interpolation margin around the transformed
                # tile, instead of decoding a full high-resolution PDF page.
                left = max(0, math.floor(min(p[0] for p in original)) - 2)
                top = max(0, math.floor(min(p[1] for p in original)) - 2)
                right = min(size[0], math.ceil(max(p[0] for p in original)) + 2)
                bottom = min(size[1], math.ceil(max(p[1] for p in original)) + 2)
                gx0, gy0, gx1, gy1 = (
                    left + offset[0],
                    top + offset[1],
                    right + offset[0],
                    bottom + offset[1],
                )
                margins = (gx0, raster_h - gy1, raster_w - gx1, gy0)
                crop = tuple(max(0, (v - 1e-7) / scale) for v in margins)
                render_size = (right - left, bottom - top)
                prepared_geometry = (*size, left, top)
                result["correction"] = correction
                result["correction_polygon"] = [
                    page_coordinates(
                        page,
                        raster_w,
                        raster_h,
                        p[0] * size[0] + offset[0],
                        p[1] * size[1] + offset[1],
                        inverse=True,
                    )
                    for p in correction["quad"]
                ]
            elif raster_region is not None:
                x0, y0, x1, y1 = raster_region
                margins = (x0, raster_h - y1, raster_w - x1, y0)
                crop = tuple(max(0, (v - 1e-7) / scale) for v in margins)
                offset, size = (x0, y0), (x1 - x0, y1 - y0)
                render_size = size
            try:
                check_dimensions(*render_size)
            except InputBudgetExceeded:
                issues.append("page_render_budget_exceeded_requires_region")
            else:
                if self.ocr is None:
                    self.ocr = self.ocr_factory()  # includes real OCR sentinel
                bitmap = page.render(scale=scale, crop=crop, limit_image_cache=True)
                temporary = staging / "pdf-page.png"
                try:
                    image = bitmap.to_pil()
                    try:
                        image.save(temporary)
                    finally:
                        image.close()
                    extracted = (
                        self.ocr.extract(
                            temporary,
                            depth=depth,
                            single_line=single_line,
                            correction=correction,
                            corrected_crop=corrected_crop,
                            prepared_geometry=prepared_geometry,
                            **({"thresholds": thresholds} if thresholds is not None else {}),
                        )
                        if correction is not None
                        else self.ocr.extract(
                            temporary,
                            depth=depth,
                            single_line=single_line,
                            **({"thresholds": thresholds} if thresholds is not None else {}),
                        )
                        if depth != "standard" or single_line or thresholds is not None
                        else self.ocr.extract(temporary)
                    )
                finally:
                    bitmap.close()
                    temporary.unlink(missing_ok=True)
                if (extracted["width"], extracted["height"]) != size:
                    raise ValueError("PDF render and OCR geometry differ")
                native_count = len(segments)
                if "diagnostics" in extracted:
                    diagnostic = extracted["diagnostics"]
                    result["diagnostics"] = {
                        "schema": 1,
                        "unexamined_contours": diagnostic["unexamined_contours"],
                        "candidates": [],
                    }
                    for candidate in diagnostic["candidates"]:
                        display = [[x + offset[0], y + offset[1]] for x, y in candidate["polygon"]]
                        original_segment = candidate["segment"]
                        accepted = (
                            original_segment is not None
                            and original_segment < MAX_SEGMENTS - native_count
                        )
                        result["diagnostics"]["candidates"].append(
                            {
                                **candidate,
                                "polygon": [
                                    page_coordinates(page, raster_w, raster_h, *point, inverse=True)
                                    for point in display
                                ],
                                "segment": native_count + original_segment if accepted else None,
                                "reason": "page_segment_budget"
                                if original_segment is not None and not accepted
                                else candidate["reason"],
                            }
                        )

                for item in extracted["segments"][: MAX_SEGMENTS - native_count]:
                    display = [[x + offset[0], y + offset[1]] for x, y in item["anchor"]["polygon"]]
                    segments.append(
                        {
                            "text": item["text"],
                            "origin": "ocr",
                            "anchor": {
                                **(
                                    {"processing": extracted["processing"]}
                                    if correction is not None
                                    else {}
                                ),
                                "kind": "pdf",
                                "unit": "pdf_point",
                                "page": number,
                                "polygon": [
                                    page_coordinates(page, raster_w, raster_h, *point, inverse=True)
                                    for point in display
                                ],
                                "display_polygon": display,
                                "confidence": item["confidence"],
                                "layer": "rendered_ocr",
                            },
                        }
                    )
                if len(extracted["segments"]) + native_count > MAX_SEGMENTS:
                    issues.append("ocr_segment_budget_exceeded")
                    result["unprocessed_ocr_segments"] = (
                        len(extracted["segments"]) + native_count - MAX_SEGMENTS
                    )
                issues.extend(extracted["issues"])
        if issues:
            result["issues"] = sorted(set(issues))
            result["coverage"] = "partial"
        return result


def render_region(page, raster_w, raster_h, region, scale=SCALE):
    """Compute small bitmap crop in the rotated visible viewport before allocation."""
    if region is None:
        return (0, 0, 0, 0), (0, 0), (raster_w, raster_h)
    if not isinstance(region, dict) or set(region) != {"left", "bottom", "right", "top"}:
        raise ValueError("invalid PDF region")
    left, bottom, right, top = (region[k] for k in ("left", "bottom", "right", "top"))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (left, bottom, right, top)):
        raise ValueError("nonfinite PDF region")
    box = page.get_bbox()
    if not (box[0] <= left < right <= box[2] and box[1] <= bottom < top <= box[3]):
        raise ValueError("PDF region outside visible page")
    points = [
        page_coordinates(page, raster_w, raster_h, x, y)
        for x, y in ((left, bottom), (right, bottom), (right, top), (left, top))
    ]
    x0, y0 = max(0, min(p[0] for p in points)), max(0, min(p[1] for p in points))
    x1, y1 = min(raster_w, max(p[0] for p in points)), min(raster_h, max(p[1] for p in points))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("PDF region is smaller than one render pixel")
    # PdfPage.render applies ceil to each scaled margin. A tiny inward epsilon
    # prevents floating division roundoff adding an unintended extra pixel.
    margins = (x0, raster_h - y1, raster_w - x1, y0)
    crop = tuple(max(0, (value - 1e-7) / scale) for value in margins)
    return crop, (x0, y0), (x1 - x0, y1 - y0)
