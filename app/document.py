import os
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime as ort

from app.schemas import DocumentResult, DocumentScores
from settings import settings


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# load an ONNX model once and reuse it for every request
@lru_cache
def load_model(rel_path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.ONNX_THREADS
    return ort.InferenceSession(os.path.join(ROOT, rel_path), options, providers=["CPUExecutionProvider"])


# DocCornerNet: 4 document corners (TL, TR, BR, BL) in image pixels + document-present probability
def predict_corners(bgr: np.ndarray) -> tuple:
    model = load_model(settings.CORNER_MODEL_PATH)
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR)
    x = ((x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)[None]
    coords, logit = model.run(None, {model.get_inputs()[0].name: x})
    score = float(1.0 / (1.0 + np.exp(-logit.ravel()[0])))
    quad = coords.reshape(4, 2) * np.array([w, h], np.float32)
    return quad.astype(np.float32), score


# DocAligner lcnet100 (fallback for small documents): corners from 4 heatmaps + confidence (weakest corner peak)
def predict_corners_fallback(bgr: np.ndarray) -> tuple:
    model = load_model(settings.FALLBACK_MODEL_PATH)
    h, w = bgr.shape[:2]
    x = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    heatmaps = model.run(None, {model.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]  # 4 x 128 x 128

    points = []
    for heatmap in heatmaps[:4]:
        mask = (heatmap >= settings.FALLBACK_HEATMAP_THRESHOLD).astype(np.uint8)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n <= 1:  # this corner was not found
            return None, float(heatmaps.max(axis=(1, 2)).min())
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx, cy = centroids[biggest]
        points.append(((cx + 0.5) * w / heatmap.shape[1], (cy + 0.5) * h / heatmap.shape[0]))
    confidence = float(heatmaps[:4].max(axis=(1, 2)).min())
    return np.array(points, np.float32), confidence


# cut out the document and straighten it, long side = long_side pixels
def warp_document(bgr: np.ndarray, quad: np.ndarray, long_side: int) -> np.ndarray:
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = long_side / max(width, height, 1.0)
    out_w, out_h = max(int(width * scale), 32), max(int(height * scale), 32)
    target = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return cv2.warpPerspective(bgr, matrix, (out_w, out_h))


# PP-OCRv4 text detection: number of text regions (text is only found, not read)
def count_text_boxes(bgr: np.ndarray) -> int:
    model = load_model(settings.TEXT_MODEL_PATH)
    h, w = bgr.shape[:2]
    scale = min(1.0, settings.DOC_TEXT_LONG_SIDE / max(h, w))
    new_h, new_w = max(32, round(h * scale / 32) * 32), max(32, round(w * scale / 32) * 32)  # network needs x32 sizes
    x = cv2.resize(bgr, (new_w, new_h)).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    prob = model.run(None, {model.get_inputs()[0].name: x})[0][0, 0]

    mask = cv2.dilate((prob > settings.TEXT_PIXEL_THRESHOLD).astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        if min(cv2.minAreaRect(contour)[1]) < 3:  # too thin to be text
            continue
        x0, y0, bw, bh = cv2.boundingRect(contour)
        if prob[y0:y0 + bh, x0:x0 + bw].mean() >= settings.TEXT_BOX_THRESHOLD:
            count += 1
    return count


# share of 8x8 blocks with almost no pixel variation: camera photos always have noise, screenshots have flat areas
def flat_share(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = (gray.shape[0] // 8) * 8, (gray.shape[1] // 8) * 8
    block_std = gray[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3))
    return float((block_std < settings.SCREENSHOT_BLOCK_STD).mean())


# is there a document: not a screenshot, document found, document has text (not a blank sheet)
def check_document(bgr: np.ndarray) -> DocumentResult:
    flat = flat_share(bgr)
    quad, corner_score = predict_corners(bgr)
    found = corner_score >= settings.DOC_MIN_CORNER_SCORE

    fallback_score = None
    if not found and flat <= settings.SCREENSHOT_MIN_FLAT_SHARE:  # main model misses small documents
        fallback_quad, fallback_score = predict_corners_fallback(bgr)
        if fallback_quad is not None and fallback_score >= settings.DOC_MIN_FALLBACK_SCORE:
            quad, found = fallback_quad, True

    text_boxes = None
    if found and flat <= settings.SCREENSHOT_MIN_FLAT_SHARE:
        text_boxes = count_text_boxes(warp_document(bgr, quad, settings.DOC_TEXT_LONG_SIDE))

    reasons = []
    if flat > settings.SCREENSHOT_MIN_FLAT_SHARE:
        reasons.append("SCREENSHOT")
    elif not found:
        reasons.append("NO_DOCUMENT")
    elif text_boxes < settings.DOC_MIN_TEXT_BOXES:
        reasons.append("BLANK_PAGE")

    return DocumentResult(
        scores=DocumentScores(corner_score=corner_score, fallback_score=fallback_score,
                              text_boxes=text_boxes, flat_share=flat),
        quad=quad.round(1).tolist() if found else None,
        reasons=reasons,
    )
