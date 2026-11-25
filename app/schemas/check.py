from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from app.schemas.document import DocumentReason, DocumentScores
from app.schemas.framing import FramingReason, FramingScores
from app.schemas.geometry import GeometryReason, GeometryScores
from app.schemas.quality import QualityReason, QualityScores


# reasons when the file itself can't be checked
FileReason = Literal["FILE_TOO_LARGE", "RESOLUTION_TOO_HIGH", "INVALID_IMAGE"]


# one rejection reason: machine code + short text for the user
class ReasonItem(BaseModel):
    code: Union[DocumentReason, FramingReason, GeometryReason, QualityReason, FileReason]
    message: str


# numeric scores of every check
class CheckScores(BaseModel):
    document: DocumentScores
    framing: Optional[FramingScores]  # null when no document was found
    geometry: Optional[GeometryScores]  # null when no document was found
    quality: QualityScores


# /api/check response (also used for 413 / 415 errors)
class CheckResponse(BaseModel):
    ok: bool                         # photo is good for further processing
    reasons: List[ReasonItem]        # all rejection reasons, empty when ok
    scores: Optional[CheckScores]    # numeric score per check, null when the file could not be read
