from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.routers import task, llm, whisper

from fastapi.exceptions import RequestValidationError
from app.common.error import validation_exception_handler
from app.common.db import ping_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ping_database()
    print("✅ Database connection verified")
    yield
    # Shutdown
    print("👋 Shutting down")


app = FastAPI(lifespan=lifespan)
app.include_router(llm.router)
app.include_router(task.router)
app.include_router(whisper.router)

@app.exception_handler(RequestValidationError)
async def handle_validation_error(request, exc):
    return await validation_exception_handler(request, exc)

@app.get("/")
def read_root():
    return {"Hello": "World"}