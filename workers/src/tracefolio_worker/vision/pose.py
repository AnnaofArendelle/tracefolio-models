# SPDX-License-Identifier: AGPL-3.0-only
"""RTMPose-t original SimCC points; no identity, action or temporal inference."""

from __future__ import annotations

import numpy as np

from tracefolio_worker.ocr.rapid import Asset
from tracefolio_worker.vision.objects import session


class PoseAdapter:
    def __init__(self, model: Asset):
        self.session = session(model)
        inputs = self.session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].shape[1:] != [3, 256, 192]
            or inputs[0].shape[0] not in (1, "batch")
        ):
            raise ValueError("unsupported RTMPose tensor contract")
        self.input_name = inputs[0].name

    def extract_pixels(self, image: np.ndarray, observation: dict) -> dict:
        import cv2

        if observation.get("label") != "person":
            raise ValueError("pose requires a person observation")
        box = np.asarray(observation["box"], dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError("invalid pose parent box")
        center = (box[:2] + box[2:]) / 2
        scale = (box[2:] - box[:2]) * 1.25
        # Same zero-rotation, center/scale transform as the official SDK. The
        # inverse transform preserves off-image/occluded raw points separately.
        if scale[0] > scale[1] * 0.75:
            scale[1] = scale[0] / 0.75
        else:
            scale[0] = scale[1] * 0.75
        factors = np.asarray([192, 256], dtype=np.float32) / scale
        matrix = np.asarray(
            [
                [factors[0], 0, 96 - center[0] * factors[0]],
                [0, factors[1], 128 - center[1] * factors[1]],
            ],
            dtype=np.float32,
        )
        crop = cv2.warpAffine(image, matrix, (192, 256), flags=cv2.INTER_LINEAR)
        normalized = (crop[:, :, ::-1].astype(np.float32) - [123.675, 116.28, 103.53]) / [
            58.395,
            57.12,
            57.375,
        ]
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self.session.run(["simcc_x", "simcc_y"], {self.input_name: tensor})
        x, y = outputs
        if (
            x.shape != (1, 17, 384)
            or y.shape != (1, 17, 512)
            or not all(np.isfinite(o).all() for o in outputs)
        ):
            raise ValueError("invalid RTMPose output")
        scores = np.minimum(x.max(axis=2)[0], y.max(axis=2)[0])
        locations = (
            np.stack([x.argmax(axis=2)[0], y.argmax(axis=2)[0]], axis=1).astype(np.float32) / 2
        )
        points = locations / np.asarray([192, 256]) * scale + center - scale / 2
        height, width = image.shape[:2]
        visible = (
            (scores >= 0.3)
            & (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        return {
            "parent_ordinal": observation["ordinal"],
            "parent_box": box.tolist(),
            "points": points.tolist(),
            "scores": scores.tolist(),
            "visible": visible.tolist(),
            "coordinate_unit": "display_pixel",
            "association": "unverified",
            "action": None,
            "identity": None,
            "derived": False,
            "transform": {"center": center.tolist(), "scale": scale.tolist()},
        }
