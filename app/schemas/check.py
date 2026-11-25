from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from app.schemas.quality import QualityReason, QualityScores


# reasons when the file itself can't be checked
FileReason = Literal["FILE_TOO_LARGE", "RESOLUTION_TOO_HIGH", "INVALID_IMAGE"]


# one rejection reason: machine code + short text for the user
class ReasonItem(BaseModel):
    code: Union[QualityReason, FileReason]
    message: str


# /api/check response (also used for 413 / 415 errors)
class CheckResponse(BaseModel):
    ok: bool                         # photo is good for further processing
    reasons: List[ReasonItem]        # all rejection reasons, empty when ok
    scores: Optional[QualityScores]  # numeric score per check, null when the file could not be read
