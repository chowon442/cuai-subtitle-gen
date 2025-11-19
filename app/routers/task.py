import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.common.db import get_db
from ..common.enum import TaskStatus

router = APIRouter(
    prefix="/task",
    tags=["task"],
)


class TaskCreateRequest(BaseModel):
    title: str = Field(..., max_length=255)
    link: str | None = None
    subtitle: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("제목을 입력해주세요.")
        return value


class TaskCompleteRequest(BaseModel):
    subtitle: Dict[str, Any] | None = None


class TaskStatusResponse(BaseModel):
    task_id: int
    status: TaskStatus


def _get_task_status(db: Session, task_id: int) -> TaskStatus:
    query = text(
        """
        SELECT status
        FROM public.tasks
        WHERE id = :task_id
        """
    )
    record = db.execute(query, {"task_id": task_id}).mappings().first()
    if record is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return TaskStatus(record["status"])

# GET /task/{task_id}/status
# 작업 상태 조회
@router.get("/{task_id}/status", response_model=TaskStatusResponse)
def get_task_status(task_id: int, db: Session = Depends(get_db)):
    status_value = _get_task_status(db, task_id)
    return {"task_id": task_id, "status": status_value}

# POST /task/create
# 새로운 작업 생성
@router.post("/create", response_model=TaskStatusResponse, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreateRequest, db: Session = Depends(get_db)):
    subtitle_payload = payload.subtitle or {}
    query = text(
        """
        INSERT INTO public.tasks (title, link, subtitle)
        VALUES (:title, :link, CAST(:subtitle AS jsonb))
        RETURNING id, status
        """
    )
    result = db.execute(
        query,
        {"title": payload.title, "link": payload.link, "subtitle": json.dumps(subtitle_payload)},
    )
    record = result.mappings().first()
    if record is None:
        db.rollback()
        raise HTTPException(status_code=500, detail="작업 생성에 실패했습니다.")
    db.commit()
    return {"task_id": record["id"], "status": TaskStatus(record["status"])}

# POST /task/{task_id}/complete
# 작업 완료 처리
@router.post("/{task_id}/complete", response_model=TaskStatusResponse)
def complete_task(task_id: int, payload: TaskCompleteRequest, db: Session = Depends(get_db)):
    params = {"task_id": task_id, "status": TaskStatus.DONE.value}
    set_clauses = ["status = :status", "updated_at = NOW()"]
    if payload.subtitle is not None:
        params["subtitle"] = json.dumps(payload.subtitle)
        set_clauses.append("subtitle = CAST(:subtitle AS jsonb)")

    query = text(
        f"""
        UPDATE public.tasks
        SET {', '.join(set_clauses)}
        WHERE id = :task_id
        RETURNING id, status
        """
    )
    result = db.execute(query, params)
    record = result.mappings().first()
    if record is None:
        db.rollback()
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    db.commit()
    return {"task_id": record["id"], "status": TaskStatus(record["status"])}