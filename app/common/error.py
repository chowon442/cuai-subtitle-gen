from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        if error["type"] == "int_parsing":
            errors.append({
                "loc": error["loc"],
                "msg": "유효한 정수를 입력해주세요",
                "type": error["type"]
            })
        else:
            errors.append(error)
    
    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )
