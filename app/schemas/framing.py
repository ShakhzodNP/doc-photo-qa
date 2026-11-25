from typing import List, Literal, Optional

from pydantic import BaseModel


FramingReason = Literal["DOCUMENT_CUT"]


# raw measurements of the "whole document in frame" check
class FramingScores(BaseModel):
    edge_margin: float                    # closest corner distance to the frame edge, as share of that side, 0..0.5
    fallback_sees_corners: Optional[bool]  # second model found all 4 corners, null when it was not needed


# framing result: measurements + rejection reasons
class FramingResult(BaseModel):
    scores: FramingScores
    reasons: List[FramingReason]
