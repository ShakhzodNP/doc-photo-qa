from typing import List, Optional

from pydantic import BaseModel

from app.messages import Reason


class DocumentScores(BaseModel):
    corner_score: float
    fallback_score: Optional[float]
    text_boxes: Optional[int]
    flat_share: float


class DocumentResult(BaseModel):
    scores: DocumentScores
    quad: Optional[List[List[float]]]  # TL, TR, BR, BL in image pixels
    reasons: List[Reason]
