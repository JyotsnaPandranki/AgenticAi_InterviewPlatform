from __future__ import annotations

import math
from typing import Iterable, List, Tuple

Point2D = Tuple[float, float]


def _dist(p1: Point2D, p2: Point2D) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye: List[Point2D]) -> float:
    if len(eye) != 6:
        raise ValueError("eye must have 6 points")
    vertical1 = _dist(eye[1], eye[5])
    vertical2 = _dist(eye[2], eye[4])
    horizontal = _dist(eye[0], eye[3])
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth: List[Point2D]) -> float:
    if len(mouth) != 8:
        raise ValueError("mouth must have 8 points")
    vertical1 = _dist(mouth[1], mouth[7])
    vertical2 = _dist(mouth[2], mouth[6])
    vertical3 = _dist(mouth[3], mouth[5])
    horizontal = _dist(mouth[0], mouth[4])
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)


def angle_between(p1: Point2D, p2: Point2D, p3: Point2D) -> float:
    a = _dist(p2, p3)
    b = _dist(p1, p3)
    c = _dist(p1, p2)
    if a * b == 0:
        return 0.0
    cos_val = max(min((a * a + c * c - b * b) / (2 * a * c), 1.0), -1.0)
    return math.degrees(math.acos(cos_val))


def shoulder_slope(left_shoulder: Point2D, right_shoulder: Point2D) -> float:
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    if dx == 0:
        return 90.0
    return abs(math.degrees(math.atan2(dy, dx)))


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.0
    clamped = min(max(value, min_val), max_val)
    return (clamped - min_val) / (max_val - min_val)


def center_distance_ratio(points: Iterable[Point2D], center: Point2D) -> float:
    points = list(points)
    if not points:
        return 1.0
    avg_x = sum(p[0] for p in points) / len(points)
    avg_y = sum(p[1] for p in points) / len(points)
    return _dist((avg_x, avg_y), center)
