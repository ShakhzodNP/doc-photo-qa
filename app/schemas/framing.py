from typing import List, Optional

from pydantic import BaseModel

from app.messages import Reason


class FramingScores(BaseModel):
    edge_margin: float
    fallback_sees_corners: Optional[bool]


class FramingResult(BaseModel):
    scores: FramingScores
    reasons: List[Reason]
