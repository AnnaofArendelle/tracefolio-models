# SPDX-License-Identifier: AGPL-3.0-only
"""Observe the pinned DB postprocessor before its score and size filters.

The contour algorithm follows RapidOCR 3.10.0 (Apache-2.0, PaddlePaddle
Authors / SWHL). Thresholds, clipping and output order stay unchanged.
"""

import cv2
import numpy as np
from rapidocr.ch_ppocr_det.utils import DBPostProcess


class DiagnosticPostProcess(DBPostProcess):
    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        self.candidates = []
        self.unexamined = max(0, len(contours) - self.max_candidates)
        self.passed = []
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            points, side = self.get_mini_boxes(contour)
            polygon, score = points.copy(), None
            reason = "det_small_before_expand"
            if side >= self.min_size:
                score = float(
                    self.box_score_fast(pred, points.reshape(-1, 2))
                    if self.score_mode == "fast"
                    else self.box_score_slow(pred, contour)
                )
                reason = "det_low_score"
                if score >= self.box_thresh:
                    polygon, side = self.get_mini_boxes(self.unclip(points))
                    reason = "det_small_after_expand" if side < self.min_size + 2 else "unread"
            polygon[:, 0] = np.clip(np.round(polygon[:, 0] / width * dest_width), 0, dest_width)
            polygon[:, 1] = np.clip(np.round(polygon[:, 1] / height * dest_height), 0, dest_height)
            self.candidates.append(
                {
                    "polygon": polygon.tolist(),
                    "detection_score": score,
                    "text": "",
                    "recognition_score": None,
                    "reason": reason,
                    "segment": None,
                }
            )
            if reason == "unread":
                self.passed.append(len(self.candidates) - 1)
                boxes.append(polygon.astype(np.int32))
                scores.append(score)
        return np.array(boxes, dtype=np.int32), scores

    def filter_det_res(self, boxes, scores, height, width):
        # Bind diagnostics to geometry BEFORE the upstream detector sorts boxes
        # without sorting scores. Recognition must never zip that scores array.
        self.by_geometry = {}
        kept_boxes, kept_scores = [], []
        for index, box, score in zip(self.passed, boxes, scores, strict=True):
            box = self.clip_det_res(self.order_points_clockwise(box), height, width)
            candidate = self.candidates[index]
            candidate["polygon"] = box.tolist()
            if (
                int(np.linalg.norm(box[0] - box[1])) <= 3
                or int(np.linalg.norm(box[0] - box[3])) <= 3
            ):
                candidate["reason"] = "det_small_after_clip"
                continue
            self.by_geometry.setdefault(tuple(box.flat), []).append(index)
            kept_boxes.append(box)
            kept_scores.append(score)
        return np.array(kept_boxes), kept_scores
