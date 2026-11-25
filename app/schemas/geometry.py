from typing import List, Literal

from pydantic import BaseModel


GeometryReason = Literal["TOO_SMALL", "TOO_TILTED"]


# raw measurements of the size and angle check
class GeometryScores(BaseModel):
    area_share: float        # document area / frame area, 0..1
    side_ratio: float        # longer / shorter of two opposite sides (worst pair), 1 = no perspective
    max_angle_dev: float     # largest corner angle deviation from 90 degrees


# geometry result: measurements + rejection reasons
class GeometryResult(BaseModel):
    scores: GeometryScores
    reasons: List[GeometryReason]
