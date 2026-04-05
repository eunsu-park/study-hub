# Exercise: Capstone Project — REST API
# Build a complete task management API with repository pattern, validation, and filtering.

from datetime import datetime, timezone
from typing import Optional
import uuid


# Exercise 1: Repository Pattern
class TaskRepository:
    """In-memory task repository with CRUD and query support.

    Each task: {
        "id": str,
        "title": str,
        "description": str,
        "status": "todo" | "in_progress" | "done",
        "priority": int (1=low, 2=medium, 3=high),
        "assignee": str | None,
        "created_at": str (ISO 8601),
        "updated_at": str (ISO 8601),
    }
    """

    def __init__(self):
        self._store = {}

    def create(self, title: str, description: str = "", priority: int = 2,
               assignee: str | None = None) -> dict:
        """Create a new task. Auto-generate id, timestamps, status='todo'."""
        # TODO: Implement
        pass

    def get(self, task_id: str) -> Optional[dict]:
        """Get task by ID. Return None if not found."""
        # TODO: Implement
        pass

    def update(self, task_id: str, **fields) -> Optional[dict]:
        """Partial update. Only update provided fields. Update updated_at.
        Return updated task or None if not found.
        Raise ValueError if unknown field is passed.
        """
        # TODO: Implement
        pass

    def delete(self, task_id: str) -> bool:
        """Delete task. Return True if existed."""
        # TODO: Implement
        pass

    def list(self, status: str | None = None, assignee: str | None = None,
             sort_by: str = "created_at", ascending: bool = True) -> list[dict]:
        """List tasks with optional filtering and sorting.

        Supported sort fields: created_at, priority, title
        """
        # TODO: Implement
        pass

    def count(self, status: str | None = None) -> int:
        """Count tasks, optionally filtered by status."""
        # TODO: Implement
        pass


# Test
# repo = TaskRepository()
# t1 = repo.create("Write tests", priority=3, assignee="alice")
# t2 = repo.create("Deploy app", priority=1, assignee="bob")
# t3 = repo.create("Review PR", priority=2, assignee="alice")
# assert repo.count() == 3
# assert repo.get(t1["id"])["title"] == "Write tests"
# repo.update(t1["id"], status="done")
# assert repo.get(t1["id"])["status"] == "done"
# alice_tasks = repo.list(assignee="alice")
# assert len(alice_tasks) == 2
# by_priority = repo.list(sort_by="priority", ascending=False)
# assert by_priority[0]["priority"] == 3


# Exercise 2: Request Validator
class RequestValidator:
    """Validate API request data against a schema.

    Schema format: {
        "field_name": {
            "type": str,          # "str", "int", "bool"
            "required": bool,
            "min_length": int,    # for str
            "max_length": int,    # for str
            "min_value": number,  # for int
            "max_value": number,  # for int
            "choices": list,      # allowed values
        }
    }
    """

    def __init__(self, schema: dict):
        self.schema = schema

    def validate(self, data: dict) -> tuple[bool, list[str]]:
        """Validate data against schema.

        Returns: (is_valid, list_of_error_messages)
        Error messages should be descriptive, e.g.:
          "title is required"
          "priority must be between 1 and 3"
          "status must be one of: todo, in_progress, done"
        """
        # TODO: Implement
        pass


# Test
# schema = {
#     "title": {"type": "str", "required": True, "min_length": 1, "max_length": 200},
#     "priority": {"type": "int", "required": False, "min_value": 1, "max_value": 3},
#     "status": {"type": "str", "required": False, "choices": ["todo", "in_progress", "done"]},
# }
# v = RequestValidator(schema)
# ok, errors = v.validate({"title": "Test"})
# assert ok and len(errors) == 0
# ok, errors = v.validate({})
# assert not ok and "title is required" in errors[0]
# ok, errors = v.validate({"title": "X", "priority": 5})
# assert not ok


# Exercise 3: API Response Builder
class ResponseBuilder:
    """Build consistent API responses."""

    @staticmethod
    def success(data, status: int = 200, meta: dict | None = None) -> dict:
        """Build success response: {"data": ..., "meta": ..., "status": int}."""
        # TODO: Implement
        pass

    @staticmethod
    def error(message: str, status: int = 400, errors: list | None = None) -> dict:
        """Build error response: {"error": {"message": ..., "errors": ...}, "status": int}."""
        # TODO: Implement
        pass

    @staticmethod
    def paginated(items: list, total: int, page: int, per_page: int) -> dict:
        """Build paginated response with data + pagination meta.

        Meta: total, page, per_page, total_pages, has_next, has_prev
        """
        # TODO: Implement
        pass


# Test
# r = ResponseBuilder.success({"id": 1}, status=201)
# assert r["status"] == 201
# assert r["data"]["id"] == 1
# r = ResponseBuilder.error("Not found", 404)
# assert r["error"]["message"] == "Not found"
# r = ResponseBuilder.paginated([1, 2, 3], total=10, page=1, per_page=3)
# assert r["meta"]["total_pages"] == 4
# assert r["meta"]["has_next"] is True


# Exercise 4: Service Layer with Validation
class TaskService:
    """Business logic layer that coordinates repository and validation."""

    CREATE_SCHEMA = {
        "title": {"type": "str", "required": True, "min_length": 1, "max_length": 200},
        "description": {"type": "str", "required": False, "max_length": 2000},
        "priority": {"type": "int", "required": False, "min_value": 1, "max_value": 3},
        "assignee": {"type": "str", "required": False, "max_length": 50},
    }

    def __init__(self, repo: TaskRepository):
        self.repo = repo
        self.validator = RequestValidator(self.CREATE_SCHEMA)

    def create_task(self, data: dict) -> dict:
        """Validate input, create task, return ResponseBuilder response.

        Return error response if validation fails.
        Return success response (201) with task data if ok.
        """
        # TODO: Implement
        pass

    def get_task(self, task_id: str) -> dict:
        """Get task by ID. Return 404 error response if not found."""
        # TODO: Implement
        pass


# Test
# repo = TaskRepository()
# svc = TaskService(repo)
# resp = svc.create_task({"title": "Test task", "priority": 2})
# assert resp["status"] == 201
# resp = svc.create_task({})
# assert resp["status"] == 400


if __name__ == "__main__":
    print("Capstone REST API Exercise")
    print("Implement each class and verify with the test cases.")
