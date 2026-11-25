import uuid

from fastapi import APIRouter, UploadFile, status
from fastapi.responses import JSONResponse

from app.image_io import ImageDecodeError, ImageTooLargeError, decode_image
from app.logger import logger
from app.quality import check_quality
from app.schemas import Msg, QualityResult
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

MSG_TOO_LARGE = "Файл слишком большой. Загрузите фото до 10 МБ."
MSG_TOO_MANY_PIXELS = "Разрешение фото слишком большое. Уменьшите его или сделайте снимок обычной камерой телефона."
MSG_INVALID = "Не удалось открыть файл. Загрузите фото в формате JPG или PNG."


@router.get("/health", response_model=Msg, responses={**responses})
async def health() -> Msg:
    return Msg(msg="OK")


@router.post("/check", response_model=QualityResult, responses={**responses})
async def check(file: UploadFile):
    TASK_ID = uuid.uuid4().hex
    max_bytes = int(settings.MAX_UPLOAD_MB * 1024 * 1024)

    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Start image decoding ({file.filename})')
    contents = await file.read(max_bytes + 1)  # one extra byte is enough to detect an oversized file
    if len(contents) > max_bytes:
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Rejected: file larger than {settings.MAX_UPLOAD_MB} MB')
        return JSONResponse(content=Msg(msg=MSG_TOO_LARGE).model_dump(),
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    try:
        bgr_image = decode_image(contents)
    except ImageTooLargeError as error:
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Rejected: {error}')
        return JSONResponse(content=Msg(msg=MSG_TOO_MANY_PIXELS).model_dump(),
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    except ImageDecodeError as error:
        logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Rejected: {error}')
        return JSONResponse(content=Msg(msg=MSG_INVALID).model_dump(),
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] End image decoding, size {bgr_image.shape[1]}x{bgr_image.shape[0]}')

    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] Start quality checks')
    result = check_quality(bgr_image)
    logger.info(f'[CHECK][TASK_ID: {TASK_ID}] End quality checks, reasons: {result.reasons}')

    return result
