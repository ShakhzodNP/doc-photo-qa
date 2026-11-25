from typing import List

from pydantic import BaseModel

from app.messages import Reason


class BrightnessStats(BaseModel):
    mean: float
    dark_share: float
    bright_share: float


class QualityScores(BaseModel):
    sharpness: float
    brightness: BrightnessStats
    glare_share: float


class QualityResult(BaseModel):
    scores: QualityScores
    reasons: List[Reason]
