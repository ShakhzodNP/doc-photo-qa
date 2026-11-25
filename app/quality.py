from typing import Optional

import cv2
import numpy as np

from app.document import warp_document
from app.messages import Reason
from app.schemas import BrightnessStats, QualityResult, QualityScores
from settings import settings


# second derivative along x, y and both diagonals
DIRECTIONAL_KERNELS = [
    np.array([[1, -2, 1]], np.float32),
    np.array([[1], [-2], [1]], np.float32),
    np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], np.float32),
    np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], np.float32),
]


def resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale == 1:
        return image
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=interpolation)


# straightened document at its own resolution (no upscaling), inset so the background around it is not measured
def document_region(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    native = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl), np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    crop = warp_document(bgr, quad, int(min(native, settings.QUALITY_WORK_SIZE)))
    h, w = crop.shape[:2]
    dy, dx = int(h * settings.QUALITY_DOC_INSET), int(w * settings.QUALITY_DOC_INSET)
    return crop[dy:h - dy, dx:w - dx]


# without it dark or low-contrast photos get a low sharpness score and are wrongly called blurry
def stretch_contrast(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (1, 99))
    if hi - lo < 1:
        return gray
    return np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


# Fixed size makes the score independent of resolution; light smoothing removes sensor noise,
# which otherwise looks like sharp detail and hides blur.
# Minimum over directions: motion blur lowers only the direction of movement, plain Laplacian misses it.
def sharpness_score(gray: np.ndarray) -> float:
    g = resize_long_side(gray, settings.SHARP_WORK_SIZE)
    g = cv2.GaussianBlur(g, (0, 0), settings.SHARP_PRE_SMOOTH)
    g = stretch_contrast(g).astype(np.float32)
    return float(min(cv2.filter2D(g, cv2.CV_32F, k).var() for k in DIRECTIONAL_KERNELS))


def brightness_stats(gray: np.ndarray) -> BrightnessStats:
    return BrightnessStats(
        mean=float(gray.mean()),
        dark_share=float((gray <= settings.LIGHT_DARK_PIXEL).mean()),
        bright_share=float((gray >= settings.LIGHT_BRIGHT_PIXEL).mean()),
    )


# Glare must be clearly brighter than the paper, otherwise pale or white documents count as one big glare.
# Paper level is a low percentile: a big glare spot would pull the median up.
def glare_share(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    paper = float(np.percentile(value, settings.GLARE_PAPER_PERCENTILE))
    min_value = max(settings.GLARE_MIN_VALUE, paper + settings.GLARE_MIN_ABOVE_PAPER)
    mask = ((value >= min_value) & (hsv[..., 1] <= settings.GLARE_MAX_SATURATION)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop thin white strokes

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_blob = settings.GLARE_MIN_BLOB_SHARE * mask.size
    glare_area = sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_blob)
    return float(glare_area / mask.size)


def check_quality(bgr: np.ndarray, quad: Optional[np.ndarray] = None) -> QualityResult:
    region = document_region(bgr, quad) if quad is not None else resize_long_side(bgr, settings.QUALITY_WORK_SIZE)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    sharpness = sharpness_score(gray)
    light = brightness_stats(gray)
    glare = glare_share(region)

    reasons = []
    if sharpness < settings.SHARP_MIN_SCORE:
        reasons.append(Reason.BLURRY)
    if light.mean < settings.LIGHT_MIN_MEAN or light.dark_share > settings.LIGHT_MAX_DARK_SHARE:
        reasons.append(Reason.TOO_DARK)
    if light.mean > settings.LIGHT_MAX_MEAN or light.bright_share > settings.LIGHT_MAX_BRIGHT_SHARE:
        reasons.append(Reason.OVEREXPOSED)
    elif glare > settings.LIGHT_MAX_GLARE_SHARE:  # an overexposed photo looks like one big glare spot
        reasons.append(Reason.GLARE)

    return QualityResult(
        scores=QualityScores(sharpness=sharpness, brightness=light, glare_share=glare),
        reasons=reasons,
    )
