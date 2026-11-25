from typing import List, Literal

from pydantic import BaseModel


QualityReason = Literal["BLURRY", "TOO_DARK", "OVEREXPOSED", "GLARE"]


# brightness statistics of an image
class BrightnessStats(BaseModel):
    mean: float          # mean gray value, 0..255
    dark_share: float    # share of almost black pixels, 0..1
    bright_share: float  # share of almost white pixels, 0..1


# raw measurements of all quality checks
class QualityScores(BaseModel):
    sharpness: float
    brightness: BrightnessStats
    glare_share: float   # glare area / image area, 0..1


# quality check result: measurements + rejection reasons
class QualityResult(BaseModel):
    scores: QualityScores
    reasons: List[QualityReason]
