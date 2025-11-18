from fastapi import FastAPI
from app.routers import task

from fastapi.exceptions import RequestValidationError
from app.common.error import validation_exception_handler

app = FastAPI()
app.include_router(task.router)

@app.exception_handler(RequestValidationError)
async def handle_validation_error(request, exc):
    return await validation_exception_handler(request, exc)

@app.get("/")
def read_root():
    return {"Hello": "World"}