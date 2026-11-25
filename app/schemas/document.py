from typing import List, Literal, Optional

from pydantic import BaseModel


DocumentReason = Literal["NO_DOCUMENT", "SCREENSHOT", "BLANK_PAGE"]


# raw measurements of the document presence check
class DocumentScores(BaseModel):
    corner_score: float          # document-present probability from the corner model, 0..1
    fallback_score: Optional[float]  # small-document model confidence, null when the main model found the document
    text_boxes: Optional[int]    # text regions inside the document, null when no document was found
    flat_share: float            # share of perfectly flat 8x8 blocks, high for screenshots, 0..1


# document presence result: measurements, document corners, rejection reasons
class DocumentResult(BaseModel):
    scores: DocumentScores
    quad: Optional[List[List[float]]]  # 4 corners [x, y] TL, TR, BR, BL in image pixels, null when no document
    reasons: List[DocumentReason]
