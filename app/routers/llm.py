from fastapi import APIRouter

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)