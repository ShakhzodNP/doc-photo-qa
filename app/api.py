import uuid
from typing import List

import numpy as np
from fastapi import APIRouter, UploadFile, status
from fastapi.responses import JSONResponse

from app.document import check_document
from app.framing import check_framing
from app.geometry import check_geometry
from app.image_io import ImageDecodeError, ImageTooLargeError, decode_image
from app.logger import logger, task_id_var
from app.messages import REASON_MESSAGES, Reason
from app.quality import check_quality
from app.schemas import CheckResponse, CheckScores, Msg, ReasonItem
from settings import settings


router = APIRouter(prefix='/api')
responses = {
    200: {"description": 'Success'},
    204: {"description": 'Not found valid content in request'},
    404: {"description": 'Bad request'},
    413: {"description": 'File or image resolution too large'},
    415: {"description": 'Unsupported media type in request'},
    501: {"description": 'Not implemented'},
}


def build_response(reasons: List[Reason], scores: CheckScores = None) -> CheckResponse:
    return CheckResponse(
        ok=not reasons,
        reasons=[ReasonItem(code=reason, message=REASON_MESSAGES[reason]) for reason in reasons],
        scores=scores,
    )


def file_error(reason: Reason, status_code: int) -> JSONResponse:
    return JSONResponse(content=build_response([reason]).model_dump(mode="json"), status_code=status_code)


@router.get("/health", response_model=Msg, responses={**responses})
async def health() -> Msg:
    return Msg(msg="OK")


@router.post("/check", response_model=CheckResponse, responses={**responses})
async def check(file: UploadFile):
    task_id_var.set(uuid.uuid4().hex)
    max_bytes = int(settings.MAX_UPLOAD_MB * 1024 * 1024)

    logger.info("Decoding %s", file.filename)
    contents = await file.read(max_bytes + 1)  # one extra byte is enough to detect an oversized file
    if len(contents) > max_bytes:
        logger.info("Rejected: file larger than %s MB", settings.MAX_UPLOAD_MB)
        return file_error(Reason.FILE_TOO_LARGE, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    try:
        bgr_image = decode_image(contents)
    except ImageTooLargeError as error:
        logger.info("Rejected: %s", error)
        return file_error(Reason.RESOLUTION_TOO_HIGH, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    except ImageDecodeError as error:
        logger.info("Rejected: %s", error)
        return file_error(Reason.INVALID_IMAGE, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    document = check_document(bgr_image)
    logger.info("Document check: %s", document.reasons)

    framing, geometry = None, None
    if document.quad is not None:
        quad = np.array(document.quad, np.float32)
        framing = check_framing(bgr_image, quad)
        geometry = check_geometry(bgr_image, quad)
        logger.info("Framing and geometry checks: %s", framing.reasons + geometry.reasons)

    quality = check_quality(bgr_image)
    logger.info("Quality checks: %s", quality.reasons)

    reasons = document.reasons
    if framing and geometry:
        reasons = reasons + framing.reasons + geometry.reasons
    reasons = reasons + quality.reasons
    scores = CheckScores(document=document.scores,
                         framing=framing.scores if framing else None,
                         geometry=geometry.scores if geometry else None,
                         quality=quality.scores)
    return build_response(reasons, scores)
