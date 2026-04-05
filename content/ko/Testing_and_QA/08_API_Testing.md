# API 테스트 (API Testing)

**이전**: [통합 테스트](./07_Integration_Testing.md) | **다음**: [엔드투엔드 테스트](./09_End_to_End_Testing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프레임워크 제공 테스트 클라이언트를 사용하여 REST API를 테스트할 수 있다
2. Flask 및 FastAPI 애플리케이션에 대한 요청/응답 테스트를 작성할 수 있다
3. API 엔드포인트에서 인증, 오류 처리, 엣지 케이스를 테스트할 수 있다
4. `pytest-httpx`와 `responses`로 외부 HTTP 호출을 모킹할 수 있다
5. API 호환성을 검증하기 위한 계약 테스트(contract testing) 원칙을 적용할 수 있다

---

## API를 테스트하는 이유

API는 백엔드와 소비자(프론트엔드, 모바일 앱, 다른 서비스) 간의 계약입니다. API 테스트는 다음을 검증합니다:

- **올바른 상태 코드** — 성공 시 200, 리소스 미존재 시 404, 유효성 검증 오류 시 422
- **응답 구조** — JSON 형태, 필수 필드, 올바른 타입
- **비즈니스 로직** — 권한 규칙, 데이터 변환, 부수 효과
- **오류 처리** — 우아한 실패, 유용한 오류 메시지

API 테스트는 테스트 피라미드에서 단위 테스트와 E2E 테스트 사이에 위치합니다. 브라우저 기반 E2E 테스트보다 빠르면서도, 라우팅, 직렬화, 미들웨어, 유효성 검증을 실행하기 때문에 순수 단위 테스트보다 현실적입니다.

---

## Flask 애플리케이션 테스트

Flask는 서버를 시작하지 않고 HTTP 요청을 시뮬레이션하는 내장 테스트 클라이언트를 제공합니다.

### 애플리케이션 설정

```python
# app.py
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# In-memory store for simplicity
BOOKS = {}
_next_id = 1


@app.post("/api/books")
def create_book():
    global _next_id
    data = request.get_json()
    if not data or "title" not in data:
        abort(400, description="Title is required")
    if not data.get("author"):
        abort(400, description="Author is required")

    book = {
        "id": _next_id,
        "title": data["title"],
        "author": data["author"],
        "year": data.get("year"),
    }
    BOOKS[_next_id] = book
    _next_id += 1
    return jsonify(book), 201


@app.get("/api/books")
def list_books():
    return jsonify(list(BOOKS.values()))


@app.get("/api/books/<int:book_id>")
def get_book(book_id):
    book = BOOKS.get(book_id)
    if book is None:
        abort(404, description="Book not found")
    return jsonify(book)


@app.delete("/api/books/<int:book_id>")
def delete_book(book_id):
    if book_id not in BOOKS:
        abort(404, description="Book not found")
    del BOOKS[book_id]
    return "", 204


@app.put("/api/books/<int:book_id>")
def update_book(book_id):
    if book_id not in BOOKS:
        abort(404, description="Book not found")
    data = request.get_json()
    if not data:
        abort(400, description="Request body is required")

    book = BOOKS[book_id]
    book["title"] = data.get("title", book["title"])
    book["author"] = data.get("author", book["author"])
    book["year"] = data.get("year", book["year"])
    return jsonify(book)
```

### 테스트 설정

```python
# conftest.py
import pytest
from app import app, BOOKS


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the in-memory store before each test."""
    global _next_id
    BOOKS.clear()
    import app as app_module
    app_module._next_id = 1
```

### API 테스트 작성하기

```python
# test_books_api.py
import json


class TestCreateBook:
    def test_create_book_success(self, client):
        response = client.post(
            "/api/books",
            json={"title": "Clean Code", "author": "Robert Martin", "year": 2008},
        )
        assert response.status_code == 201
        data = response.get_json()
        assert data["title"] == "Clean Code"
        assert data["author"] == "Robert Martin"
        assert data["year"] == 2008
        assert "id" in data

    def test_create_book_without_title(self, client):
        response = client.post(
            "/api/books",
            json={"author": "Someone"},
        )
        assert response.status_code == 400

    def test_create_book_without_author(self, client):
        response = client.post(
            "/api/books",
            json={"title": "Some Book"},
        )
        assert response.status_code == 400

    def test_create_book_empty_body(self, client):
        response = client.post(
            "/api/books",
            data="",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_create_book_assigns_sequential_ids(self, client):
        r1 = client.post("/api/books", json={"title": "Book 1", "author": "A"})
        r2 = client.post("/api/books", json={"title": "Book 2", "author": "B"})
        assert r1.get_json()["id"] == 1
        assert r2.get_json()["id"] == 2


class TestListBooks:
    def test_list_empty(self, client):
        response = client.get("/api/books")
        assert response.status_code == 200
        assert response.get_json() == []

    def test_list_after_adding(self, client):
        client.post("/api/books", json={"title": "Book A", "author": "Auth A"})
        client.post("/api/books", json={"title": "Book B", "author": "Auth B"})
        response = client.get("/api/books")
        data = response.get_json()
        assert len(data) == 2


class TestGetBook:
    def test_get_existing_book(self, client):
        create_resp = client.post(
            "/api/books", json={"title": "Refactoring", "author": "Martin Fowler"}
        )
        book_id = create_resp.get_json()["id"]

        response = client.get(f"/api/books/{book_id}")
        assert response.status_code == 200
        assert response.get_json()["title"] == "Refactoring"

    def test_get_nonexistent_book(self, client):
        response = client.get("/api/books/9999")
        assert response.status_code == 404


class TestUpdateBook:
    def test_update_title(self, client):
        client.post("/api/books", json={"title": "Old Title", "author": "Author"})
        response = client.put("/api/books/1", json={"title": "New Title"})
        assert response.status_code == 200
        assert response.get_json()["title"] == "New Title"
        assert response.get_json()["author"] == "Author"  # Unchanged

    def test_update_nonexistent(self, client):
        response = client.put("/api/books/9999", json={"title": "X"})
        assert response.status_code == 404


class TestDeleteBook:
    def test_delete_existing(self, client):
        client.post("/api/books", json={"title": "To Delete", "author": "Author"})
        response = client.delete("/api/books/1")
        assert response.status_code == 204

        # Verify it is gone
        get_response = client.get("/api/books/1")
        assert get_response.status_code == 404

    def test_delete_nonexistent(self, client):
        response = client.delete("/api/books/9999")
        assert response.status_code == 404
```

---

## FastAPI 애플리케이션 테스트

FastAPI는 Starlette의 `TestClient`를 제공하며, 동기 및 비동기 엔드포인트를 모두 지원합니다.

```python
# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class TodoCreate(BaseModel):
    title: str
    description: str = ""
    priority: int = 1


class Todo(BaseModel):
    id: int
    title: str
    description: str
    priority: int
    completed: bool = False


TODOS: dict[int, dict] = {}
_counter = 0


@app.post("/api/todos", response_model=Todo, status_code=201)
def create_todo(todo: TodoCreate):
    global _counter
    _counter += 1
    item = {
        "id": _counter,
        "title": todo.title,
        "description": todo.description,
        "priority": todo.priority,
        "completed": False,
    }
    TODOS[_counter] = item
    return item


@app.get("/api/todos", response_model=list[Todo])
def list_todos(completed: bool | None = None):
    items = list(TODOS.values())
    if completed is not None:
        items = [t for t in items if t["completed"] == completed]
    return items


@app.patch("/api/todos/{todo_id}")
def toggle_complete(todo_id: int):
    if todo_id not in TODOS:
        raise HTTPException(status_code=404, detail="Todo not found")
    TODOS[todo_id]["completed"] = not TODOS[todo_id]["completed"]
    return TODOS[todo_id]
```

```python
# test_fastapi_app.py
import pytest
from fastapi.testclient import TestClient
from fastapi_app import app, TODOS


@pytest.fixture
def client():
    TODOS.clear()
    import fastapi_app
    fastapi_app._counter = 0
    return TestClient(app)


class TestCreateTodo:
    def test_create_success(self, client):
        response = client.post(
            "/api/todos",
            json={"title": "Buy groceries", "priority": 2},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Buy groceries"
        assert data["priority"] == 2
        assert data["completed"] is False

    def test_create_with_defaults(self, client):
        response = client.post("/api/todos", json={"title": "Simple task"})
        data = response.json()
        assert data["description"] == ""
        assert data["priority"] == 1

    def test_create_missing_title(self, client):
        """FastAPI validates automatically via Pydantic."""
        response = client.post("/api/todos", json={"priority": 1})
        assert response.status_code == 422  # Validation error
        detail = response.json()["detail"]
        assert any("title" in str(e).lower() for e in detail)


class TestListTodos:
    def test_filter_completed(self, client):
        client.post("/api/todos", json={"title": "Task 1"})
        client.post("/api/todos", json={"title": "Task 2"})
        client.patch("/api/todos/1")  # Complete task 1

        response = client.get("/api/todos?completed=true")
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Task 1"

    def test_filter_incomplete(self, client):
        client.post("/api/todos", json={"title": "Task 1"})
        client.post("/api/todos", json={"title": "Task 2"})
        client.patch("/api/todos/1")  # Complete task 1

        response = client.get("/api/todos?completed=false")
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Task 2"
```

---

## 인증 테스트

많은 API는 인증이 필요합니다. 인증된 접근과 인증되지 않은 접근 모두를 테스트합니다.

```python
# Auth helper for tests
@pytest.fixture
def auth_headers():
    """Generate valid authentication headers."""
    token = create_test_token(user_id=1, role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def viewer_headers():
    """Generate headers for a read-only user."""
    token = create_test_token(user_id=2, role="viewer")
    return {"Authorization": f"Bearer {token}"}


class TestAuthenticatedEndpoints:
    def test_create_requires_auth(self, client):
        response = client.post("/api/books", json={"title": "X", "author": "Y"})
        assert response.status_code == 401

    def test_create_with_valid_token(self, client, auth_headers):
        response = client.post(
            "/api/books",
            json={"title": "X", "author": "Y"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_create_with_expired_token(self, client):
        expired_token = create_test_token(user_id=1, expired=True)
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.post(
            "/api/books",
            json={"title": "X", "author": "Y"},
            headers=headers,
        )
        assert response.status_code == 401

    def test_viewer_cannot_delete(self, client, viewer_headers):
        # First create as admin
        admin_token = create_test_token(user_id=1, role="admin")
        client.post(
            "/api/books",
            json={"title": "X", "author": "Y"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # Viewer tries to delete
        response = client.delete("/api/books/1", headers=viewer_headers)
        assert response.status_code == 403
```

---

## 오류 응답 테스트

API가 유용한 오류 정보를 반환하는지 항상 테스트합니다:

```python
class TestErrorResponses:
    def test_404_includes_message(self, client):
        response = client.get("/api/books/9999")
        assert response.status_code == 404
        data = response.get_json()
        assert "not found" in data.get("description", "").lower()

    def test_400_includes_field_info(self, client):
        response = client.post("/api/books", json={})
        assert response.status_code == 400

    def test_405_method_not_allowed(self, client):
        response = client.patch("/api/books")  # PATCH not supported on collection
        assert response.status_code == 405

    def test_content_type_is_json(self, client):
        response = client.get("/api/books")
        assert response.content_type == "application/json"
```

---

## 외부 HTTP 호출 모킹

API가 외부 서비스를 호출하는 경우, 해당 호출을 모킹하여 테스트를 빠르고 결정적으로 유지합니다.

### responses 라이브러리 사용하기

```bash
pip install responses
```

```python
import responses
import requests


def get_weather(city: str) -> dict:
    """Call external weather API."""
    resp = requests.get(
        "https://api.weather.com/current",
        params={"city": city},
    )
    resp.raise_for_status()
    return resp.json()


@responses.activate
def test_get_weather_success():
    responses.add(
        responses.GET,
        "https://api.weather.com/current",
        json={"city": "Seoul", "temp": 22.5},
        status=200,
    )

    result = get_weather("Seoul")
    assert result["temp"] == 22.5


@responses.activate
def test_get_weather_api_error():
    responses.add(
        responses.GET,
        "https://api.weather.com/current",
        json={"error": "Service unavailable"},
        status=503,
    )

    with pytest.raises(requests.HTTPError):
        get_weather("Seoul")
```

### pytest-httpx 사용하기 (httpx 기반 앱용)

```bash
pip install pytest-httpx
```

```python
import httpx
import pytest


async def fetch_user(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()


@pytest.mark.asyncio
async def test_fetch_user(httpx_mock):
    httpx_mock.add_response(
        url="https://api.example.com/users/1",
        json={"id": 1, "name": "Alice"},
    )

    user = await fetch_user(1)
    assert user["name"] == "Alice"


@pytest.mark.asyncio
async def test_fetch_user_not_found(httpx_mock):
    httpx_mock.add_response(
        url="https://api.example.com/users/999",
        status_code=404,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_user(999)
```

---

## 응답 검증

상태 코드 외에도 응답 구조를 엄격하게 검증합니다:

```python
def test_book_response_structure(client):
    """Verify the response contains all expected fields with correct types."""
    client.post("/api/books", json={"title": "Test", "author": "Author", "year": 2024})
    response = client.get("/api/books/1")
    data = response.get_json()

    # Required fields exist
    assert "id" in data
    assert "title" in data
    assert "author" in data

    # Correct types
    assert isinstance(data["id"], int)
    assert isinstance(data["title"], str)
    assert isinstance(data["author"], str)
    assert data["year"] is None or isinstance(data["year"], int)

    # No unexpected fields leaked (e.g., internal IDs, passwords)
    allowed_fields = {"id", "title", "author", "year"}
    assert set(data.keys()) <= allowed_fields
```

### jsonschema를 사용한 검증

```bash
pip install jsonschema
```

```python
from jsonschema import validate

BOOK_SCHEMA = {
    "type": "object",
    "required": ["id", "title", "author"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "title": {"type": "string", "minLength": 1},
        "author": {"type": "string", "minLength": 1},
        "year": {"type": ["integer", "null"]},
    },
    "additionalProperties": False,
}


def test_response_matches_schema(client):
    client.post("/api/books", json={"title": "Schema Test", "author": "Tester"})
    response = client.get("/api/books/1")
    validate(instance=response.get_json(), schema=BOOK_SCHEMA)
```

---

## 계약 테스트 소개

계약 테스트(contract testing)는 제공자(API)와 소비자(클라이언트)가 API의 구조에 합의하는지 검증합니다. 양쪽을 함께 테스트하는 통합 테스트와 달리, 계약 테스트는 공유된 계약에 대해 각 쪽을 독립적으로 검증합니다.

```
Consumer Test:             Provider Test:
"I expect GET /users/1     "When called GET /users/1
 to return {id, name}"      I return {id, name}"
         │                           │
         └──── Contract File ────────┘
              (shared JSON)
```

**Pact**와 같은 도구가 이를 정형화합니다. 소비자가 계약(pact 파일)을 생성하고, 제공자가 이를 검증합니다:

```python
# Consumer side: generate the contract
# Provider side: verify against the contract

# This ensures API changes do not break consumers,
# even without running both services together.
```

계약 테스트는 서로 다른 팀이 다른 서비스를 소유하고 API 변경을 안전하게 조율해야 하는 마이크로서비스 아키텍처에서 가장 유용합니다.

---

## 연습 문제

1. **전체 CRUD 테스트 스위트**: "노트" 관리를 위한 간단한 Flask API를 구축합니다(생성, 목록, 조회, 수정, 삭제). 유효성 검증 오류, 404, 빈 상태를 포함하여 모든 엔드포인트를 다루는 완전한 테스트 스위트를 작성합니다.

2. **인증 테스트**: 노트 API에 토큰 기반 인증을 추가합니다. 인증되지 않은 요청이 거부되는지, 만료된 토큰이 거부되는지, 유효한 토큰이 성공하는지, 역할 기반 접근이 적용되는지 검증하는 테스트를 작성합니다.

3. **외부 API 모킹**: 외부 API(예: 환율 변환)를 호출하는 서비스를 작성합니다. `responses` 라이브러리를 사용하여 외부 API를 모킹합니다. 성공 시나리오와 오류 시나리오(타임아웃, 500 오류, 잘못된 형식의 JSON) 모두를 테스트합니다.

---

**License**: CC BY-NC 4.0
