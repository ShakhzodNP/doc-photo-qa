import numpy as np

from app.document import predict_corners_fallback
from app.messages import Reason
from app.schemas import FramingResult, FramingScores
from settings import settings


def edge_margin(quad: np.ndarray, width: int, height: int) -> float:
    x, y = quad[:, 0] / width, quad[:, 1] / height
    return float(min(x.min(), (1 - x).min(), y.min(), (1 - y).min()))


# The corner model puts a cut-off corner ~1% inside the frame, so a corner near the edge is only suspicious.
# The second model decides: on cut documents it finds no 4th corner or invents it with a weak peak.
def check_framing(bgr: np.ndarray, quad: np.ndarray) -> FramingResult:
    h, w = bgr.shape[:2]
    margin = edge_margin(quad, w, h)

    fallback_sees_corners = None
    if margin < settings.FRAME_EDGE_MARGIN:
        fallback_quad, fallback_score = predict_corners_fallback(bgr)
        fallback_sees_corners = fallback_quad is not None and fallback_score >= settings.FRAME_MIN_FALLBACK_SCORE

    reasons = [Reason.DOCUMENT_CUT] if fallback_sees_corners is False else []
    return FramingResult(
        scores=FramingScores(edge_margin=margin, fallback_sees_corners=fallback_sees_corners),
        reasons=reasons,
    )
