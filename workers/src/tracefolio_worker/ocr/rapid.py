# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned RapidOCR adapter with bounded detection and original-resolution text crops."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


# Packaged with the adapter and pinned by the Go task contract. Runtime paths
# are inserted separately; they never form part of an inference identity.
PROFILE = json.loads(Path(__file__).with_name("profile.json").read_text())
if PROFILE["schema"] != 2 or PROFILE["detector_resize"] != "bounded-area-linear-exact-v1":
    raise ValueError("unsupported OCR profile schema")
INPUT_LIMITS = PROFILE["input_limits"]


class InputBudgetExceeded(ValueError):
    """The caller should retain the original and offer an explicit bounded region."""


@dataclass(frozen=True)
class OCRThresholds:
    detection: float = 0.3
    box: float = 0.5
    text: float = 0.5

    @classmethod
    def from_request(cls, value):
        if value is None:
            return cls()
        if (
            not isinstance(value, dict)
            or set(value) != {"detection", "box", "text"}
            or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in value.values())
        ):
            raise ValueError("OCR thresholds require detection, box and text between 0 and 1")
        return cls(**value)


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str

    def verify(self) -> None:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("model asset must be a regular file")
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.sha256:
            raise ValueError("model asset digest mismatch")


@dataclass(frozen=True)
class OCRModels:
    detection: Asset
    recognition: Asset
    classification: Asset
    dictionary: Asset


def detector_shape(width: int, height: int, depth: str = "standard") -> tuple[int, int]:
    """Keep admitted preprocessing; fit oversized tensors within the area budget."""
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    if depth == "fast":
        scale = min(1.0, 960 / max(width, height))
        return tuple(max(32, round(int(side * scale) / 32) * 32) for side in (width, height))
    scale = max(1.0, INPUT_LIMITS["detector_min_side"] / min(width, height))
    stride = INPUT_LIMITS["detector_stride"]
    dw, dh = (
        round(int(width * scale) / stride) * stride,
        round(int(height * scale) / stride) * stride,
    )
    budget = INPUT_LIMITS["detector_pixels"]
    if dw * dh <= budget:
        return dw, dh
    # Do not let the minimum-side enlargement undo the area limit for strips.
    # Floor to stride and keep at least one cell, even for a one-pixel source.
    scale = min(1.0, math.sqrt(budget / (width * height)))
    dw = max(stride, int(width * scale) // stride * stride)
    dh = max(stride, int(height * scale) // stride * stride)
    while dw * dh > budget:
        if dw >= dh:
            dw -= stride
        else:
            dh -= stride
    return dw, dh


def detector_preprocess(limit_side_len, limit_type, mean, std, depth="standard"):
    from rapidocr.ch_ppocr_det.utils import DetPreProcess

    class BoundedDetPreProcess(DetPreProcess):
        def resize(self, img):
            import cv2

            height, width = img.shape[:2]
            dw, dh = detector_shape(width, height, depth)
            if depth == "fast":
                return cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
            # Preserve the prior byte-exact path for already admitted shapes.
            scale = max(1.0, INPUT_LIMITS["detector_min_side"] / min(width, height))
            stride = INPUT_LIMITS["detector_stride"]
            old_w = round(int(width * scale) / stride) * stride
            old_h = round(int(height * scale) / stride) * stride
            if old_w * old_h <= INPUT_LIMITS["detector_pixels"]:
                return super().resize(img)
            return cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR_EXACT)

    return BoundedDetPreProcess(limit_side_len, limit_type, mean, std)


def detector_downscaled(width: int, height: int) -> bool:
    scale = max(1.0, INPUT_LIMITS["detector_min_side"] / min(width, height))
    stride = INPUT_LIMITS["detector_stride"]
    return (
        round(int(width * scale) / stride) * stride * round(int(height * scale) / stride) * stride
        > INPUT_LIMITS["detector_pixels"]
    )


def check_source_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    if (
        width * height > INPUT_LIMITS["source_pixels"]
        or max(width, height) > INPUT_LIMITS["source_max_side"]
    ):
        raise InputBudgetExceeded("source image exceeds full decode budget")


def check_dimensions(width: int, height: int) -> None:
    check_source_dimensions(width, height)
    dw, dh = detector_shape(width, height)
    if dw * dh > INPUT_LIMITS["detector_pixels"]:
        raise InputBudgetExceeded("detector tensor exceeds budget; select a region")


class RapidAdapter:
    def __init__(self, models: OCRModels):
        for asset in (
            models.detection,
            models.recognition,
            models.classification,
            models.dictionary,
        ):
            asset.verify()
        import cv2
        from rapidocr import ModelType, OCRVersion, RapidOCR

        cv2.setNumThreads(1)
        params = dict(PROFILE["parameters"])
        params.update(
            {
                "Det.model_path": str(models.detection.path),
                "Rec.model_path": str(models.recognition.path),
                "Cls.model_path": str(models.classification.path),
                "Rec.rec_keys_path": str(models.dictionary.path),
            }
        )
        for part in ("Det", "Rec"):
            params[part + ".model_type"] = ModelType[params[part + ".model_type"]]
            params[part + ".ocr_version"] = OCRVersion[params[part + ".ocr_version"]]
        self.engine = RapidOCR(params=params)
        detector = self.engine._load_det_model()
        from tracefolio_worker.ocr.diagnostics import DiagnosticPostProcess

        original = detector.postprocess_op
        detector.postprocess_op = DiagnosticPostProcess(
            thresh=original.thresh,
            box_thresh=original.box_thresh,
            max_candidates=original.max_candidates,
            unclip_ratio=original.unclip_ratio,
            use_dilation=original.dilation_kernel is not None,
            score_mode=original.score_mode,
        )
        detector.get_preprocess = lambda: detector_preprocess(
            detector.limit_side_len, detector.limit_type, detector.mean, detector.std
        )
        self.ready = False

    def infer(self, pixels, depth="standard", single_line=False, thresholds=OCRThresholds()):
        import numpy as np

        # Each task owns its thresholds, including tasks that use defaults.
        self.engine.text_score = thresholds.text

        if single_line:
            from types import SimpleNamespace

            crops, _ = self.engine.cls_and_rotate([pixels])
            recognized = self.engine.recognize_txt(crops)
            if len(recognized.txts) != 1 or len(recognized.scores) != 1:
                raise ValueError("single-line OCR returned multiple results")
            text, score = recognized.txts[0], float(recognized.scores[0])
            if not isinstance(text, str) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid single-line OCR result")
            height, width = pixels.shape[:2]
            polygon = [[0, 0], [width, 0], [width, height], [0, height]]
            reason = (
                "empty"
                if not text.strip()
                else "low_score"
                if score < self.engine.text_score
                else "recognized"
            )
            accepted = reason == "recognized"
            output = SimpleNamespace(
                txts=[text] if accepted else [],
                scores=[score] if accepted else [],
                boxes=np.array([polygon] if accepted else []),
            )
            return output, {
                "schema": 1,
                "unexamined_contours": 0,
                "candidates": [
                    {
                        "polygon": polygon,
                        "text": text,
                        "detection_score": None,
                        "recognition_score": score,
                        "reason": reason,
                        "segment": 0 if accepted else None,
                    }
                ],
            }

        detector = self.engine._load_det_model()
        detector.postprocess_op.thresh = thresholds.detection
        detector.postprocess_op.box_thresh = thresholds.box
        detector.get_preprocess = lambda: detector_preprocess(
            detector.limit_side_len, detector.limit_type, detector.mean, detector.std, depth
        )
        # Global resizing/padding is disabled in the pinned profile; detector
        # boxes and recognition crops already use this input's original pixels.
        image, operations = self.engine.preprocess_img(pixels)
        detected, classified, recognized, crops = self.engine.run_ocr_steps(image, operations)
        diagnostic = self.engine.text_det.postprocess_op
        if detected.boxes is not None:
            texts = recognized.txts
            scores = recognized.scores
            if texts is not None and (
                scores is None or len(texts) != len(detected.boxes) or len(scores) != len(texts)
            ):
                raise ValueError("OCR returned mismatched pre-filter arrays")
            ordinal = 0
            for index, box in enumerate(detected.boxes):
                candidate = diagnostic.candidates[diagnostic.by_geometry[tuple(box.flat)].pop(0)]
                if texts is None:
                    continue
                text, score = texts[index], float(scores[index])
                if not isinstance(text, str) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("invalid pre-filter OCR text or score")
                reason = (
                    "empty"
                    if not text.strip()
                    else "low_score"
                    if score < self.engine.text_score
                    else "recognized"
                )
                candidate.update(text=text, recognition_score=score, reason=reason)
                if reason == "recognized":
                    candidate["segment"] = ordinal
                    ordinal += 1
        output = self.engine.build_final_output(
            pixels, detected, classified, recognized, crops, operations
        )
        return output, {
            "schema": 1,
            "candidates": diagnostic.candidates,
            "unexamined_contours": diagnostic.unexamined,
        }

    def preflight(self, image: Path, sentinel: str) -> dict:
        """Successful construction is not readiness: exercise det/cls/rec."""
        result = self.extract(image)
        if not sentinel or sentinel not in "\n".join(s["text"] for s in result["segments"]):
            raise ValueError("OCR inference preflight sentinel missing")
        self.ready = True
        return result

    def extract(
        self,
        path: Path,
        crop: list[int] | None = None,
        orientation_override: int | None = None,
        depth="standard",
        single_line=False,
        correction=None,
        corrected_crop=None,
        prepared_geometry=None,
        thresholds=OCRThresholds(),
    ) -> dict:
        import cv2
        import numpy as np
        from PIL import Image
        from tracefolio_worker.vision.direction import oriented_image

        # The supervisor supplies a private immutable staging path, never a URL.
        # Read the header before decoding. EXIF transposition defines display
        # coordinates and is recorded, rather than silently rotating the source.
        with Image.open(path) as source:
            check_source_dimensions(*source.size)
            orientation = (
                orientation_override
                if orientation_override is not None
                else source.getexif().get(274, 1)
            )
            width, height = source.size
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            if crop is not None:
                if (
                    not isinstance(crop, list)
                    or len(crop) != 4
                    or any(type(v) is not int for v in crop)
                    or not 0 <= crop[0] < crop[2] <= width
                    or not 0 <= crop[1] < crop[3] <= height
                ):
                    raise ValueError("invalid oriented image crop")
            display = oriented_image(source, orientation).convert("RGB")
            if crop is not None:
                display = display.crop(tuple(crop))
                orientation = 1
            width, height = display.size
            pixels = cv2.cvtColor(np.asarray(display), cv2.COLOR_RGB2BGR)
        if prepared_geometry is not None:
            # PDFium rendered only the source bounding box needed by this tile.
            width, height, offset_x, offset_y = prepared_geometry
        else:
            offset_x, offset_y = 0, 0
        inverse = None
        if correction is not None:
            from tracefolio_worker.ocr.correction import correction_geometry

            matrix, size = correction_geometry(correction, width, height)
            x0, y0 = 0, 0
            if corrected_crop is not None:
                x0, y0, x1, y1 = (corrected_crop[k] for k in ("left", "top", "right", "bottom"))
                if not 0 <= x0 < x1 <= size[0] or not 0 <= y0 < y1 <= size[1]:
                    raise ValueError("corrected tile outside manual image")
                size = x1 - x0, y1 - y0
            check_source_dimensions(*size)
            local = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=float) @ matrix
            inverse = np.linalg.inv(local)
            local = local @ np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=float)
            pixels = cv2.warpPerspective(
                pixels, local, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
            )
        processing_height, processing_width = pixels.shape[:2]
        output, diagnostics = self.infer(pixels, depth, single_line, thresholds)
        texts = list(output.txts) if output.txts is not None else []
        boxes = output.boxes.tolist() if output.boxes is not None else []
        scores = list(output.scores) if output.scores is not None else []
        if inverse is not None:
            # Both accepted and rejected candidates keep source coordinates.
            # Only floating-point noise at an exact source edge is clamped.
            def original_polygon(polygon):
                mapped = cv2.perspectiveTransform(np.array([polygon], dtype=np.float64), inverse)[0]
                for point in mapped:
                    for axis, bound in enumerate((width, height)):
                        if -1e-5 <= point[axis] <= bound + 1e-5:
                            point[axis] = max(0, min(bound, point[axis]))
                return mapped.tolist()

            boxes = [original_polygon(box) for box in boxes]
            for candidate in diagnostics["candidates"]:
                candidate["polygon"] = original_polygon(candidate["polygon"])
        if not (len(texts) == len(boxes) == len(scores)):
            raise ValueError("OCR returned mismatched text/box/score arrays")
        segments = []
        for text, box, score in zip(texts, boxes, scores, strict=True):
            if len(box) != 4 or any(len(p) != 2 for p in box):
                raise ValueError("OCR returned invalid quadrilateral")
            if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                raise ValueError("OCR returned invalid confidence")
            if any(
                not math.isfinite(v) or v < 0 or v > bound
                for point in box
                for v, bound in zip(point, (width, height), strict=True)
            ):
                raise ValueError("OCR coordinates outside display image")
            segments.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "anchor": {"kind": "image", "polygon": box, "unit": "display_pixel"},
                }
            )
        dw, dh = (
            (processing_width, processing_height)
            if single_line
            else detector_shape(processing_width, processing_height, depth)
        )
        return {
            "segments": segments,
            "width": width,
            "height": height,
            "exif_orientation": orientation,
            "coverage": "complete",
            "issues": [],
            "diagnostics": diagnostics,
            "processing": {
                "thresholds": asdict(thresholds),
                **({"correction": correction} if correction is not None else {}),
                **({"corrected_crop": corrected_crop} if corrected_crop is not None else {}),
                "depth": depth,
                "single_line": single_line,
                "policy": PROFILE["detector_resize"],
                "input_width": processing_width,
                "input_height": processing_height,
                "detector_width": dw,
                "detector_height": dh,
                "scale_x": dw / processing_width,
                "scale_y": dh / processing_height,
                "downscaled": False
                if single_line
                else (dw < processing_width or dh < processing_height)
                if depth == "fast"
                else detector_downscaled(processing_width, processing_height),
                "recognition_input": "manually_corrected_text_crops"
                if correction is not None
                else "original_resolution_text_crops",
            },
        }
