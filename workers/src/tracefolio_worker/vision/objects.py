# SPDX-License-Identifier: AGPL-3.0-only
"""YOLOX official ONNX 416 raw-BGR contract, bounded single-frame observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tracefolio_worker.ocr.rapid import Asset, InputBudgetExceeded

with (Path(__file__).with_name("classes.json")).open(encoding="utf-8") as classes_file:
    COCO_LABELS = [row["label"] for row in json.load(classes_file)["classes"]]


def session(asset: Asset):
    import onnxruntime as ort

    asset.verify()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 3
    return ort.InferenceSession(str(asset.path), options, providers=["CPUExecutionProvider"])


def load_image(path: Path, crop=None, orientation_override=None) -> tuple[np.ndarray, int]:
    import cv2
    from PIL import Image
    from tracefolio_worker.vision.direction import oriented_image

    cv2.setNumThreads(1)
    with Image.open(path) as source:
        if source.width * source.height > 16_000_000 or max(source.size) > 16_384:
            raise InputBudgetExceeded("vision source exceeds decode budget")
        orientation = (
            orientation_override
            if orientation_override is not None
            else source.getexif().get(274, 1)
        )
        display = oriented_image(source, orientation).convert("RGB")
        if crop is not None:
            if (
                not isinstance(crop, list)
                or len(crop) != 4
                or any(type(v) is not int for v in crop)
                or not 0 <= crop[0] < crop[2] <= display.width
                or not 0 <= crop[1] < crop[3] <= display.height
            ):
                raise ValueError("invalid visual crop in display pixels")
            display = display.crop(tuple(crop))
        return cv2.cvtColor(np.asarray(display), cv2.COLOR_RGB2BGR), orientation


def intersection_over_union(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    intersections = np.maximum(
        0, np.minimum(box[2:], others[:, 2:]) - np.maximum(box[:2], others[:, :2])
    )
    area = intersections.prod(axis=1)
    denominator = np.maximum(
        1e-9, (box[2:] - box[:2]).prod() + (others[:, 2:] - others[:, :2]).prod(axis=1) - area
    )
    return area / denominator


def detector_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    # Default INTER_LINEAR rounds differently in native ARM and x86 kernels.
    # The exact integer interpolation produces identical uint8 model inputs.
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR_EXACT)


class ObjectAdapter:
    def __init__(self, model: Asset):
        self.session = session(model)
        inp = self.session.get_inputs()
        if len(inp) != 1 or inp[0].shape != [1, 3, 416, 416]:
            raise ValueError("unsupported YOLOX tensor contract")
        self.input_name = inp[0].name
        self.ready = False
        grids = []
        for stride in (8, 16, 32):
            gx, gy = np.meshgrid(np.arange(416 // stride), np.arange(416 // stride))
            grids.append(np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, stride)], axis=1))
        self.grid = np.concatenate(grids).astype(np.float32)

    def extract_pixels(self, image: np.ndarray, threshold: float = 0.35) -> dict:
        if not 0.1 <= threshold <= 0.95:
            raise ValueError("object threshold outside supported range")
        height, width = image.shape[:2]
        ratio = min(416 / width, 416 / height)
        resized_width, resized_height = max(1, int(width * ratio)), max(1, int(height * ratio))
        tensor = np.full((416, 416, 3), 114, dtype=np.float32)
        tensor[:resized_height, :resized_width] = detector_resize(
            image, (resized_width, resized_height)
        )
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.shape != (1, 3549, 85) or not np.isfinite(output).all():
            raise ValueError("invalid YOLOX output")
        prediction = output[0]
        scores = prediction[:, 4:5] * prediction[:, 5:]
        centers = (prediction[:, :2] + self.grid[:, :2]) * self.grid[:, 2:]
        dimensions = np.exp(np.clip(prediction[:, 2:4], -30, 30)) * self.grid[:, 2:]
        boxes = np.concatenate((centers - dimensions / 2, centers + dimensions / 2), axis=1) / ratio
        labels, confidences = scores.argmax(axis=1), scores.max(axis=1)
        selected = []
        for label in np.unique(labels):
            candidates = np.flatnonzero((labels == label) & (confidences >= threshold))
            candidates = candidates[np.argsort(-confidences[candidates], kind="stable")]
            while len(candidates):
                winner = candidates[0]
                selected.append(int(winner))
                candidates = candidates[1:]
                candidates = candidates[
                    intersection_over_union(boxes[winner], boxes[candidates]) <= 0.45
                ]
        selected.sort(key=lambda i: (-float(confidences[i]), i))
        observations = []
        for i in selected[:256]:
            box = np.clip(boxes[i], 0, [width, height, width, height])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            observations.append(
                {
                    "ordinal": len(observations),
                    "class_id": int(labels[i]),
                    "label": COCO_LABELS[labels[i]],
                    "score": float(confidences[i]),
                    "box": box.tolist(),
                    "coordinate_unit": "display_pixel",
                    "source": "local_detection",
                    "identity": None,
                }
            )
        truncated = len(selected) > 256
        return {
            "objects": observations,
            "width": width,
            "height": height,
            "coverage": "partial" if truncated else "complete",
            "issues": ["object_limit_reached"] if truncated else [],
        }

    def extract(self, path: Path) -> dict:
        image, orientation = load_image(path)
        result = self.extract_pixels(image)
        result["exif_orientation"] = orientation
        return result

    def preflight(self, path: Path, expected_label: str) -> dict:
        result = self.extract(path)
        if expected_label not in {o["label"] for o in result["objects"]}:
            raise ValueError("object preflight sentinel missing")
        self.ready = True
        return result
