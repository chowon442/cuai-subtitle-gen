from fastapi import APIRouter
from ..common.enum import TaskStatus

router = APIRouter(
    prefix="/task",
    tags=["task"],
)

# ID로 진행 상태 조회
@router.get("/{task_id}/status")
async def get_task_status(task_id: int):
    return {"task_id": task_id, "status": TaskStatus.RUNNING.value}