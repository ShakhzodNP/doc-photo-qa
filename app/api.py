from fastapi import APIRouter

from app.schemas import Msg


router = APIRouter(prefix='/api')
responses = {
    200: {"description": 'Success'},
    204: {"description": 'Not found valid content in request'},
    404: {"description": 'Bad request'},
    501: {"description": 'Not implemented'},
    415: {"description": 'Unsupported media type in request'}
}


@router.get("/health", response_model=Msg, responses={**responses})
async def health() -> Msg:
    return Msg(msg="OK")
