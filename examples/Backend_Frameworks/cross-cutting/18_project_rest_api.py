"""
Capstone Project — REST API Structure
Demonstrates: modular project layout, repository pattern, service layer,
              error handling, and dependency injection for a task management API.

Run: pip install fastapi uvicorn && uvicorn 18_project_rest_api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum
import uuid


# ========== Domain Models ==========

class TaskStatus(str, Enum):
    todo = "todo"
    in_progress = "in_progress"
    done = "done"


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = ""
    assignee: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    assignee: Optional[str] = None


class Task(BaseModel):
    id: str
    title: str
    description: str
    status: TaskStatus
    assignee: Optional[str]
    created_at: str
    updated_at: str


# ========== Repository Layer ==========

class TaskRepository:
    """In-memory repository (swap with SQLAlchemy/MongoDB in production)."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def create(self, data: dict) -> dict:
        task_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        record = {"id": task_id, "status": "todo", "created_at": now, "updated_at": now, **data}
        self._store[task_id] = record
        return record

    def get(self, task_id: str) -> Optional[dict]:
        return self._store.get(task_id)

    def list_all(self, status_filter: Optional[str] = None) -> list[dict]:
        tasks = list(self._store.values())
        if status_filter:
            tasks = [t for t in tasks if t["status"] == status_filter]
        return sorted(tasks, key=lambda t: t["created_at"], reverse=True)

    def update(self, task_id: str, changes: dict) -> Optional[dict]:
        if task_id not in self._store:
            return None
        record = self._store[task_id]
        for k, v in changes.items():
            if v is not None:
                record[k] = v
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        return record

    def delete(self, task_id: str) -> bool:
        return self._store.pop(task_id, None) is not None


# ========== Service Layer ==========

class TaskService:
    """Business logic layer between routes and repository."""

    def __init__(self, repo: TaskRepository):
        self.repo = repo

    def create_task(self, data: TaskCreate) -> Task:
        record = self.repo.create(data.model_dump())
        return Task(**record)

    def get_task(self, task_id: str) -> Task:
        record = self.repo.get(task_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return Task(**record)

    def list_tasks(self, status_filter: Optional[str] = None) -> list[Task]:
        return [Task(**r) for r in self.repo.list_all(status_filter)]

    def update_task(self, task_id: str, data: TaskUpdate) -> Task:
        changes = data.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(status_code=400, detail="No fields to update")
        record = self.repo.update(task_id, changes)
        if not record:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return Task(**record)

    def delete_task(self, task_id: str):
        if not self.repo.delete(task_id):
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


# ========== Dependency Injection ==========

_repo = TaskRepository()
_service = TaskService(_repo)


def get_task_service() -> TaskService:
    return _service


# ========== API Routes ==========

app = FastAPI(title="Task Manager API", version="1.0.0")


@app.post("/api/tasks", response_model=Task, status_code=201)
async def create_task(body: TaskCreate, svc: TaskService = Depends(get_task_service)):
    return svc.create_task(body)


@app.get("/api/tasks", response_model=list[Task])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    svc: TaskService = Depends(get_task_service),
):
    return svc.list_tasks(status.value if status else None)


@app.get("/api/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str, svc: TaskService = Depends(get_task_service)):
    return svc.get_task(task_id)


@app.patch("/api/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, body: TaskUpdate, svc: TaskService = Depends(get_task_service)):
    return svc.update_task(task_id, body)


@app.delete("/api/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str, svc: TaskService = Depends(get_task_service)):
    svc.delete_task(task_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
