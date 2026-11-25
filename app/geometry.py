import cv2
import numpy as np

from app.schemas import GeometryResult, GeometryScores
from settings import settings


# share of the frame covered by the document quad
def area_share(quad: np.ndarray, width: int, height: int) -> float:
    return float(cv2.contourArea(quad.astype(np.float32)) / (width * height))


# perspective strength: longer / shorter side for both pairs of opposite sides, the worse pair wins
def side_ratio(quad: np.ndarray) -> float:
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    ratio = lambda a, b: max(a, b) / max(min(a, b), 1.0)
    return float(max(ratio(sides[0], sides[2]), ratio(sides[1], sides[3])))


# largest deviation of a corner angle from 90 degrees (very high for broken or extremely skewed quads)
def max_angle_dev(quad: np.ndarray) -> float:
    devs = []
    for i in range(4):
        v1, v2 = quad[i - 1] - quad[i], quad[(i + 1) % 4] - quad[i]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        devs.append(abs(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) - 90))
    return float(max(devs))


# document big enough in the frame and not shot at a too sharp angle
def check_geometry(bgr: np.ndarray, quad: np.ndarray) -> GeometryResult:
    h, w = bgr.shape[:2]
    scores = GeometryScores(area_share=area_share(quad, w, h), side_ratio=side_ratio(quad), max_angle_dev=max_angle_dev(quad))

    reasons = []
    if scores.area_share < settings.GEOM_MIN_AREA_SHARE:
        reasons.append("TOO_SMALL")
    if scores.side_ratio > settings.GEOM_MAX_SIDE_RATIO or scores.max_angle_dev > settings.GEOM_MAX_ANGLE_DEV:
        reasons.append("TOO_TILTED")
    return GeometryResult(scores=scores, reasons=reasons)
