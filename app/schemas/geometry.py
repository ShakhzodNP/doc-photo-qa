from typing import List

from pydantic import BaseModel

from app.messages import Reason


class GeometryScores(BaseModel):
    area_share: float
    side_ratio: float
    max_angle_dev: float


class GeometryResult(BaseModel):
    scores: GeometryScores
    reasons: List[Reason]
