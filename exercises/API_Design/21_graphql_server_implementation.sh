#!/bin/bash
# Exercises for Lesson 21: GraphQL Server Implementation
# Topic: API_Design
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Full Server Setup (Task Management) ==="
    cat << 'SOLUTION'
# types/task.py
import strawberry
from enum import Enum
from typing import Optional
from datetime import datetime

@strawberry.enum
class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"

@strawberry.type
class Task:
    id: strawberry.ID
    title: str
    description: Optional[str]
    status: TaskStatus
    project_id: strawberry.ID
    assignee_id: Optional[strawberry.ID]
    created_at: datetime

    @strawberry.field
    async def project(self, info) -> "Project":
        return await info.context.dataloaders.project_loader.load(self.project_id)

    @strawberry.field
    async def assignee(self, info) -> Optional["User"]:
        if self.assignee_id:
            return await info.context.dataloaders.user_loader.load(self.assignee_id)
        return None

@strawberry.input
class CreateTaskInput:
    title: str
    description: Optional[str] = None
    project_id: strawberry.ID
    assignee_id: Optional[strawberry.ID] = None

@strawberry.type
class CreateTaskPayload:
    task: Optional[Task] = None
    user_errors: list["UserError"] = strawberry.field(default_factory=list)

# mutations/task_mutations.py
@strawberry.type
class TaskMutation:
    @strawberry.mutation
    async def create_task(self, info, input: CreateTaskInput) -> CreateTaskPayload:
        user = require_auth(info)
        errors = []
        if not input.title.strip():
            errors.append(UserError(["input","title"], "Required", "BLANK"))
        project = await info.context.db.projects.find_by_id(input.project_id)
        if not project:
            errors.append(UserError(["input","projectId"], "Not found", "NOT_FOUND"))
        if errors:
            return CreateTaskPayload(user_errors=errors)
        task = await info.context.db.tasks.create(
            title=input.title, description=input.description,
            project_id=input.project_id, assignee_id=input.assignee_id,
            status=TaskStatus.TODO, creator_id=user.id,
        )
        return CreateTaskPayload(task=task)

# context.py with JWT auth
async def get_context(request: Request) -> RequestContext:
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    current_user = None
    if token:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        current_user = await db.users.find_by_id(payload["sub"])
    return RequestContext(
        db=db, current_user=current_user,
        dataloaders=DataLoaders(db),
    )
SOLUTION
}

exercise_3() {
    echo "=== Exercise 3: Middleware Pipeline ==="
    cat << 'SOLUTION'
import hashlib, json, time
from collections import defaultdict
from datetime import datetime, timedelta

class QueryCostExtension(SchemaExtension):
    MAX_COST = 500
    def on_operation(self):
        cost = self._estimate_cost()
        if cost > self.MAX_COST:
            raise ValueError(f"Query cost {cost} exceeds {self.MAX_COST}")
        yield
    def _estimate_cost(self):
        return self.execution_context.query.count("{") * 2

class CacheExtension(SchemaExtension):
    _cache = {}
    TTL = 60
    def on_operation(self):
        key = hashlib.sha256(
            (self.execution_context.query or "").encode()
        ).hexdigest()
        cached = self._cache.get(key)
        if cached and time.time() - cached["ts"] < self.TTL:
            self.execution_context.result = cached["result"]
            return
        yield
        result = self.execution_context.result
        if result and not result.errors:
            self._cache[key] = {"result": result, "ts": time.time()}

class RateLimitExtension(SchemaExtension):
    _usage = defaultdict(list)
    MAX_OPS = 100
    WINDOW = 60
    def on_operation(self):
        user = self.execution_context.context.current_user
        key = user.id if user else "anon"
        now = time.time()
        self._usage[key] = [t for t in self._usage[key] if now - t < self.WINDOW]
        if len(self._usage[key]) >= self.MAX_OPS:
            raise ValueError("Rate limit exceeded")
        self._usage[key].append(now)
        yield
SOLUTION
}

main() { exercise_1; echo ""; exercise_3; }
main "$@"
