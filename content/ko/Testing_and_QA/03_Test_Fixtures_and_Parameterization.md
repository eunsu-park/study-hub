# 테스트 Fixture와 매개변수화 (Test Fixtures and Parameterization)

**이전**: [pytest를 이용한 단위 테스팅](./02_Unit_Testing_with_pytest.md) | **다음**: [Mocking과 Patching](./04_Mocking_and_Patching.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. pytest fixture를 생성하여 테스트 설정과 해제를 관리할 수 있다
2. scope 수준으로 fixture의 생명주기를 제어할 수 있다
3. `conftest.py`를 사용하여 테스트 모듈 간에 fixture를 공유할 수 있다
4. 유연한 테스트 데이터 생성을 위한 fixture 팩토리를 구축할 수 있다
5. `@pytest.mark.parametrize`로 중복된 테스트를 제거할 수 있다

---

## Fixture란 무엇인가?

Fixture는 테스트 데이터, 객체 또는 리소스를 제공하는 함수입니다. `unittest`의 `setUp` / `tearDown` 패턴을 의존성 주입 기반의 더 유연하고 조합 가능한 접근 방식으로 대체합니다.

핵심 아이디어: 테스트 함수가 fixture 매개변수의 이름을 지정하여 필요한 것을 *선언*하면, pytest가 자동으로 이를 제공합니다.

```python
import pytest


@pytest.fixture
def sample_user():
    """Provide a sample user dictionary for tests."""
    return {
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30,
    }


def test_user_has_name(sample_user):
    assert sample_user["name"] == "Alice"

def test_user_has_valid_email(sample_user):
    assert "@" in sample_user["email"]
```

`sample_user`를 요청하는 각 테스트는 **새로운** 인스턴스를 받습니다. Fixture는 기본적으로 요청하는 테스트마다 한 번 실행됩니다.

---

## Fixture Scope

`scope` 매개변수는 fixture가 생성되고 소멸되는 빈도를 제어합니다:

| Scope      | 생성 시점                    | 소멸 시점                     |
|------------|----------------------------|------------------------------|
| `function` | 각 테스트 함수마다           | 테스트 함수 종료 후           |
| `class`    | 각 테스트 클래스마다         | 클래스의 마지막 테스트 후     |
| `module`   | 각 테스트 모듈마다           | 모듈의 마지막 테스트 후       |
| `session`  | 전체 테스트 실행에서 한 번   | 모든 테스트 완료 후           |

```python
import pytest
import time


@pytest.fixture(scope="function")
def fresh_list():
    """New list for every test."""
    return []


@pytest.fixture(scope="module")
def db_connection():
    """Shared DB connection for all tests in this module."""
    conn = create_connection("test.db")
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def app_config():
    """Loaded once for the entire test session."""
    return load_config("test_config.yaml")
```

**경험 법칙**: 합리적인 범위에서 가장 좁은 scope를 사용하세요. `function` scope가 각 테스트가 깨끗한 상태를 얻으므로 가장 안전합니다. 데이터베이스 연결이나 Docker 컨테이너 같은 비용이 큰 리소스에는 `module` 또는 `session` scope를 사용하세요.

### Scope와 가변성 경고

`module` scope의 fixture가 가변 객체를 반환하고 하나의 테스트가 이를 변경하면, 이후 테스트가 변경된 상태를 보게 됩니다. 이는 테스트 순서 버그를 유발합니다.

```python
@pytest.fixture(scope="module")
def shared_list():
    return [1, 2, 3]

def test_append(shared_list):
    shared_list.append(4)
    assert len(shared_list) == 4  # passes

def test_original_length(shared_list):
    # DANGER: shared_list is [1, 2, 3, 4] if test_append ran first!
    assert len(shared_list) == 3  # may FAIL
```

---

## yield를 이용한 설정과 해제

Fixture에서 `yield`를 사용하여 설정과 해제를 분리합니다. `yield` 이전의 코드는 설정 시 실행되고, `yield` 이후의 코드는 테스트가 실패하더라도 해제 시 실행됩니다.

```python
import pytest
import tempfile
import os


@pytest.fixture
def temp_file():
    """Create a temporary file and clean it up after the test."""
    # Setup
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.write(fd, b"test data")
    os.close(fd)

    yield path  # This value is passed to the test

    # Teardown — always runs
    if os.path.exists(path):
        os.remove(path)


def test_temp_file_exists(temp_file):
    assert os.path.exists(temp_file)

def test_temp_file_content(temp_file):
    with open(temp_file) as f:
        assert f.read() == "test data"
```

### 데이터베이스 트랜잭션 롤백 패턴

데이터베이스 테스트를 위한 강력한 패턴: 설정에서 트랜잭션을 시작하고, 해제에서 롤백합니다. 각 테스트가 비용이 큰 재생성 없이 깨끗한 데이터베이스를 보게 됩니다.

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture
def db_session():
    """Provide a database session that rolls back after each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    session = Session(engine)
    session.begin_nested()  # SAVEPOINT

    yield session

    session.rollback()
    session.close()


def test_create_user(db_session):
    user = User(name="Alice", email="alice@example.com")
    db_session.add(user)
    db_session.flush()
    assert user.id is not None
    # Rollback happens automatically — no user persists
```

---

## conftest.py: Fixture 공유

`conftest.py`에 fixture를 배치하면 여러 테스트 파일에서 공유할 수 있습니다. pytest는 `conftest.py` 파일을 자동으로 탐색합니다 — import가 필요하지 않습니다.

```
tests/
├── conftest.py              # Fixtures available to ALL tests
├── unit/
│   ├── conftest.py          # Fixtures for unit tests only
│   ├── test_models.py
│   └── test_services.py
├── integration/
│   ├── conftest.py          # Fixtures for integration tests only
│   └── test_api.py
```

```python
# tests/conftest.py
import pytest

@pytest.fixture
def app():
    """Create a test Flask application."""
    from myapp import create_app
    app = create_app(testing=True)
    return app

@pytest.fixture
def client(app):
    """Create a test client from the app fixture."""
    return app.test_client()
```

```python
# tests/unit/test_models.py — no import needed for fixtures
def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
```

### Fixture 조합

Fixture는 다른 fixture에 의존할 수 있습니다. pytest가 의존성 체인을 자동으로 해결합니다:

```python
@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture
def db_session(db_engine):
    """Depends on db_engine fixture."""
    session = Session(db_engine)
    yield session
    session.close()

@pytest.fixture
def user_repo(db_session):
    """Depends on db_session, which depends on db_engine."""
    return UserRepository(db_session)

def test_add_user(user_repo):
    """Receives user_repo, which pulls in db_session and db_engine."""
    user_repo.add(User(name="Bob"))
    assert user_repo.count() == 1
```

---

## Fixture 팩토리

테스트 데이터의 여러 변형이 필요할 때, 팩토리 fixture를 사용합니다 — 객체를 생성하는 *함수*를 반환하는 fixture입니다.

```python
@pytest.fixture
def make_user():
    """Factory fixture: call with custom attributes."""
    created_users = []

    def _make_user(name="Test User", email=None, role="viewer"):
        email = email or f"{name.lower().replace(' ', '.')}@test.com"
        user = User(name=name, email=email, role=role)
        created_users.append(user)
        return user

    yield _make_user

    # Cleanup all created users
    for user in created_users:
        user.delete()


def test_admin_can_delete(make_user):
    admin = make_user(name="Admin", role="admin")
    target = make_user(name="Target")
    admin.delete_user(target)
    assert target.is_deleted

def test_viewer_cannot_delete(make_user):
    viewer = make_user(name="Viewer", role="viewer")
    target = make_user(name="Target")
    with pytest.raises(PermissionError):
        viewer.delete_user(target)
```

---

## 매개변수화된 테스트 (Parameterized Tests)

`@pytest.mark.parametrize`는 동일한 테스트 함수를 서로 다른 입력과 기대 출력으로 실행합니다. 복사-붙여넣기 테스트 중복을 제거합니다.

### 기본 매개변수화

```python
import pytest

def is_palindrome(s: str) -> bool:
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


@pytest.mark.parametrize("input_str, expected", [
    ("racecar", True),
    ("hello", False),
    ("A man a plan a canal Panama", True),
    ("", True),
    ("a", True),
    ("ab", False),
    ("Madam", True),
])
def test_is_palindrome(input_str, expected):
    assert is_palindrome(input_str) == expected
```

pytest는 각각 명확하게 라벨이 붙은 7개의 테스트를 실행합니다:

```
test_palindrome.py::test_is_palindrome[racecar-True]        PASSED
test_palindrome.py::test_is_palindrome[hello-False]          PASSED
test_palindrome.py::test_is_palindrome[A man a plan...-True] PASSED
...
```

### ID를 사용한 매개변수화

각 테스트 케이스에 읽기 쉬운 이름을 부여합니다:

```python
@pytest.mark.parametrize("a, b, expected", [
    pytest.param(2, 3, 5, id="positive"),
    pytest.param(-1, 1, 0, id="neg_plus_pos"),
    pytest.param(0, 0, 0, id="zeros"),
    pytest.param(-3, -7, -10, id="both_negative"),
    pytest.param(1_000_000, 1, 1_000_001, id="large_number"),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

### 여러 Parametrize 데코레이터

데코레이터를 쌓아서 테스트 케이스의 교차곱을 생성합니다:

```python
@pytest.mark.parametrize("x", [1, 2, 3])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply(x, y):
    """Runs 6 tests: (1,10), (1,20), (2,10), (2,20), (3,10), (3,20)"""
    result = x * y
    assert result == x * y
```

### 예상 예외를 포함한 매개변수화

```python
@pytest.mark.parametrize("input_val, expected_error", [
    ("", "cannot be empty"),
    ("abc", "must be numeric"),
    ("-5", "must be positive"),
    ("0", "must be positive"),
])
def test_parse_positive_int_errors(input_val, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        parse_positive_int(input_val)
```

---

## 간접 매개변수화 (Indirect Parameterization)

때때로 테스트 입력이 아닌 *fixture* 입력을 매개변수화하고 싶을 수 있습니다. `indirect=True`를 사용합니다:

```python
@pytest.fixture
def user_by_role(request):
    """Create a user with the role passed via parametrize."""
    role = request.param
    return User(name=f"Test {role}", role=role)


@pytest.mark.parametrize("user_by_role", ["admin", "editor", "viewer"], indirect=True)
def test_user_can_view(user_by_role):
    """All roles should be able to view content."""
    assert user_by_role.can_view() is True


@pytest.mark.parametrize("user_by_role", ["admin", "editor"], indirect=True)
def test_user_can_edit(user_by_role):
    """Only admin and editor can edit."""
    assert user_by_role.can_edit() is True
```

---

## 내장 Fixture

pytest는 여러 내장 fixture를 제공합니다. 가장 유용한 것들:

```python
# tmp_path — provides a temporary directory unique to each test
def test_write_file(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("hello")
    assert file.read_text() == "hello"

# tmp_path_factory — for session/module scoped temp directories
@pytest.fixture(scope="session")
def shared_data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

# capsys — capture stdout/stderr
def test_print_output(capsys):
    print("hello")
    captured = capsys.readouterr()
    assert captured.out == "hello\n"

# monkeypatch — modify objects, environment, or sys.path
def test_with_env_var(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key-123")
    from myapp.config import get_api_key
    assert get_api_key() == "test-key-123"

# request — access test metadata from within a fixture
@pytest.fixture
def data_file(request):
    """Load test data based on the test's module name."""
    module = request.module.__name__
    return load_test_data(f"{module}_data.json")
```

---

## 실전 예제: 작업 관리자 테스팅

```python
# task_manager.py
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    title: str
    priority: Priority = Priority.MEDIUM
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    due_date: datetime | None = None


class TaskManager:
    def __init__(self):
        self._tasks: list[Task] = []

    def add(self, task: Task) -> None:
        if not task.title.strip():
            raise ValueError("Task title cannot be empty")
        self._tasks.append(task)

    def complete(self, title: str) -> None:
        task = self._find(title)
        task.completed = True

    def pending(self) -> list[Task]:
        return [t for t in self._tasks if not t.completed]

    def overdue(self) -> list[Task]:
        now = datetime.now()
        return [
            t for t in self._tasks
            if not t.completed and t.due_date and t.due_date < now
        ]

    def _find(self, title: str) -> Task:
        for task in self._tasks:
            if task.title == title:
                return task
        raise KeyError(f"Task not found: {title}")
```

```python
# test_task_manager.py
import pytest
from datetime import datetime, timedelta
from task_manager import TaskManager, Task, Priority


@pytest.fixture
def manager():
    return TaskManager()


@pytest.fixture
def make_task():
    def _make(title="Test Task", priority=Priority.MEDIUM, due_date=None):
        return Task(title=title, priority=priority, due_date=due_date)
    return _make


class TestAddTask:
    def test_add_single_task(self, manager, make_task):
        manager.add(make_task(title="Buy groceries"))
        assert len(manager.pending()) == 1

    def test_add_empty_title_raises(self, manager):
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.add(Task(title="  "))

    @pytest.mark.parametrize("priority", list(Priority))
    def test_add_task_with_any_priority(self, manager, priority):
        task = Task(title="Task", priority=priority)
        manager.add(task)
        assert manager.pending()[0].priority == priority


class TestCompleteTask:
    def test_complete_marks_task_done(self, manager, make_task):
        manager.add(make_task(title="Write tests"))
        manager.complete("Write tests")
        assert len(manager.pending()) == 0

    def test_complete_nonexistent_raises(self, manager):
        with pytest.raises(KeyError, match="not found"):
            manager.complete("Ghost task")


class TestOverdue:
    def test_overdue_task_detected(self, manager):
        yesterday = datetime.now() - timedelta(days=1)
        task = Task(title="Urgent", due_date=yesterday)
        manager.add(task)
        assert len(manager.overdue()) == 1

    def test_future_task_not_overdue(self, manager):
        tomorrow = datetime.now() + timedelta(days=1)
        task = Task(title="Later", due_date=tomorrow)
        manager.add(task)
        assert len(manager.overdue()) == 0

    def test_completed_task_not_overdue(self, manager):
        yesterday = datetime.now() - timedelta(days=1)
        task = Task(title="Done", due_date=yesterday)
        manager.add(task)
        manager.complete("Done")
        assert len(manager.overdue()) == 0
```

---

## 연습 문제

1. **Fixture 체인**: 세 단계의 fixture 체인(`database` -> `user_table` -> `user_record`)을 만드세요. 각 fixture는 이전 것에 의존합니다. `user_record`를 사용하는 테스트를 작성하고 전체 체인이 동작하는지 검증하세요.

2. **매개변수화된 유효성 검사**: 규칙을 적용하는 `validate_username(name)` 함수를 작성하세요 (3-20자, 영숫자 + 밑줄, 문자로 시작해야 함). 유효한 입력과 유효하지 않은 입력을 포함하여 최소 10개의 케이스로 `@pytest.mark.parametrize`를 사용하세요.

3. **팩토리 fixture**: 제품, 수량, 할인을 설정할 수 있는 주문 객체를 생성하는 팩토리 fixture `make_order`를 만드세요. 최소 5가지 다른 설정으로 팩토리를 사용하여 `calculate_total` 함수에 대한 테스트를 작성하세요.

---

**License**: CC BY-NC 4.0
