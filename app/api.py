import uuid

import numpy as np

from fastapi import APIRouter, UploadFile, status
from fastapi.responses import JSONResponse

from app.image_io import ImageDecodeError, ImageTooLargeError, decode_image
from app.logger import logger
from app.document import check_document
from app.framing import check_framing
from app.messages import REASON_MESSAGES
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

# build a response from reason codes (empty list = photo is ok)
def build_response(codes: list, scores=None) -> CheckResponse:
    return CheckResponse(
        ok=not codes,
        reasons=[ReasonItem(code=code, message=REASON_MESSAGES[code]) for code in codes],
        scores=scores,
    )


# error response for a file that can't be checked
def file_error(code: str, status_code: int) -> JSONResponse:
    return JSONResponse(content=build_response([code]).model_dump(), status_code=status_code)


@router.get("/health", response_model=Msg, responses={**responses})
async def health() -> Msg:
    return Msg(msg="OK")


@router.post("/check", response_model=CheckResponse, responses={**responses})
async def check(file: UploadFile):
    TASK_ID = uuid.uuid4().hex
    max_bytes = int(settings.MAX_UPLOAD_MB * 1024 * 1024)

    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Start image decoding ({file.filename})')
    contents = await file.read(max_bytes + 1)  # one extra byte is enough to detect an oversized file
    if len(contents) > max_bytes:
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Rejected: file larger than {settings.MAX_UPLOAD_MB} MB')
        return file_error("FILE_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    try:
        bgr_image = decode_image(contents)
    except ImageTooLargeError as error:
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Rejected: {error}')
        return file_error("RESOLUTION_TOO_HIGH", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    except ImageDecodeError as error:
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Rejected: {error}')
        return file_error("INVALID_IMAGE", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] End image decoding, size {bgr_image.shape[1]}x{bgr_image.shape[0]}')

    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Start document check')
    document = check_document(bgr_image)
    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] End document check, reasons: {document.reasons}')

    framing = None
    if document.quad is not None:
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Start framing check')
        framing = check_framing(bgr_image, np.array(document.quad, np.float32))
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] End framing check, reasons: {framing.reasons}')

    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Start quality checks')
    quality = check_quality(bgr_image)
    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] End quality checks, reasons: {quality.reasons}')

    framing_reasons = framing.reasons if framing else []
    scores = CheckScores(document=document.scores, framing=framing.scores if framing else None, quality=quality.scores)
    return build_response(document.reasons + framing_reasons + quality.reasons, scores)
