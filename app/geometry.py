import cv2
import numpy as np

from app.messages import Reason
from app.schemas import GeometryResult, GeometryScores
from settings import settings


def area_share(quad: np.ndarray, width: int, height: int) -> float:
    return float(cv2.contourArea(quad.astype(np.float32)) / (width * height))


# perspective makes opposite sides of a rectangle unequal
def side_ratio(quad: np.ndarray) -> float:
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    ratio = lambda a, b: max(a, b) / max(min(a, b), 1.0)
    return float(max(ratio(sides[0], sides[2]), ratio(sides[1], sides[3])))


def max_angle_dev(quad: np.ndarray) -> float:
    devs = []
    for i in range(4):
        v1, v2 = quad[i - 1] - quad[i], quad[(i + 1) % 4] - quad[i]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        devs.append(abs(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) - 90))
    return float(max(devs))


def check_geometry(bgr: np.ndarray, quad: np.ndarray) -> GeometryResult:
    h, w = bgr.shape[:2]
    scores = GeometryScores(area_share=area_share(quad, w, h), side_ratio=side_ratio(quad), max_angle_dev=max_angle_dev(quad))

    reasons = []
    if scores.area_share < settings.GEOM_MIN_AREA_SHARE:
        reasons.append(Reason.TOO_SMALL)
    if scores.side_ratio > settings.GEOM_MAX_SIDE_RATIO or scores.max_angle_dev > settings.GEOM_MAX_ANGLE_DEV:
        reasons.append(Reason.TOO_TILTED)
    return GeometryResult(scores=scores, reasons=reasons)
