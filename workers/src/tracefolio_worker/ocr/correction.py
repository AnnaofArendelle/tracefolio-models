# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit manual perspective/rotation; coordinates always return to the source."""

import math


def validate_correction(value):
    if value is None:
        return
    if (
        not isinstance(value, dict)
        or set(value) != {"quad", "turns"}
        or type(value["turns"]) is not int
        or not 0 <= value["turns"] <= 3
    ):
        raise ValueError("invalid manual correction")
    quad = value["quad"]
    if (
        not isinstance(quad, list)
        or len(quad) != 4
        or any(
            not isinstance(p, list)
            or len(p) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p
            )
            for p in quad
        )
    ):
        raise ValueError("manual correction points outside selection")
    for i, p in enumerate(quad):
        q, r = quad[(i + 1) % 4], quad[(i + 2) % 4]
        if (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]) <= 0:
            raise ValueError("manual correction requires clockwise convex points")


def correction_geometry(value, width, height):
    import cv2
    import numpy as np

    points = np.array(value["quad"], dtype=np.float64) * [width, height]
    w = max(1, math.ceil(max(math.dist(points[0], points[1]), math.dist(points[3], points[2]))))
    h = max(1, math.ceil(max(math.dist(points[0], points[3]), math.dist(points[1], points[2]))))
    turns = value["turns"]
    if turns % 2:
        w, h = h, w
    # Clockwise rotation makes the old bottom-left the new top-left.
    points = np.roll(points, turns, axis=0)
    target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), target)
    return matrix, (w, h)
