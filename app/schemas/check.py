from typing import List, Optional

from pydantic import BaseModel

from app.messages import Reason
from app.schemas.document import DocumentScores
from app.schemas.framing import FramingScores
from app.schemas.geometry import GeometryScores
from app.schemas.quality import QualityScores


class ReasonItem(BaseModel):
    code: Reason
    message: str


class CheckScores(BaseModel):
    document: DocumentScores
    framing: Optional[FramingScores]
    geometry: Optional[GeometryScores]
    quality: QualityScores


class CheckResponse(BaseModel):
    ok: bool
    reasons: List[ReasonItem]
    scores: Optional[CheckScores]
