# Test Fixtures and Parameterization

**Previous**: [Unit Testing with pytest](./02_Unit_Testing_with_pytest.md) | **Next**: [Mocking and Patching](./04_Mocking_and_Patching.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create pytest fixtures to manage test setup and teardown
2. Control fixture lifecycle with scope levels
3. Share fixtures across test modules using `conftest.py`
4. Build fixture factories for flexible test data creation
5. Eliminate duplicated tests with `@pytest.mark.parametrize`

---

## What Are Fixtures?

Fixtures are functions that provide test data, objects, or resources. They replace the `setUp` / `tearDown` pattern from `unittest` with a more flexible, composable approach based on dependency injection.

The key idea: a test function declares what it *needs* by naming fixture parameters, and pytest automatically provides them.

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

Each test that requests `sample_user` receives a **fresh** instance. Fixtures run once per requesting test by default.

---

## Fixture Scopes

The `scope` parameter controls how often a fixture is created and destroyed:

| Scope      | Created                     | Destroyed                    |
|------------|----------------------------|------------------------------|
| `function` | Once per test function     | After the test function ends |
| `class`    | Once per test class        | After the last test in class |
| `module`   | Once per test module       | After the last test in module|
| `session`  | Once per entire test run   | After all tests finish       |

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

**Rule of thumb**: Use the narrowest scope that makes sense. `function` scope is safest because each test gets clean state. Use `module` or `session` scope for expensive resources like database connections or Docker containers.

### Scope and Mutability Warning

If a `module`-scoped fixture returns a mutable object and one test mutates it, subsequent tests see the mutation. This causes test ordering bugs.

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

## Setup and Teardown with yield

Use `yield` in a fixture to separate setup from teardown. Code before `yield` runs during setup; code after `yield` runs during teardown, even if the test fails.

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

### Database Transaction Rollback Pattern

A powerful pattern for database tests: start a transaction in setup, rollback in teardown. Every test sees a clean database without expensive re-creation.

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

## conftest.py: Sharing Fixtures

Place fixtures in `conftest.py` to share them across multiple test files. pytest automatically discovers `conftest.py` files — no imports needed.

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

### Fixture Composition

Fixtures can depend on other fixtures. pytest resolves the dependency chain automatically:

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

## Fixture Factories

When you need multiple variations of test data, use a factory fixture — a fixture that returns a *function* for creating objects.

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

## Parameterized Tests

`@pytest.mark.parametrize` runs the same test function with different inputs and expected outputs. This eliminates copy-paste test duplication.

### Basic Parameterization

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

pytest runs seven tests, each clearly labeled:

```
test_palindrome.py::test_is_palindrome[racecar-True]        PASSED
test_palindrome.py::test_is_palindrome[hello-False]          PASSED
test_palindrome.py::test_is_palindrome[A man a plan...-True] PASSED
...
```

### Parameterize with IDs

Give each test case a readable name:

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

### Multiple Parameterize Decorators

Stack decorators to create a cross-product of test cases:

```python
@pytest.mark.parametrize("x", [1, 2, 3])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply(x, y):
    """Runs 6 tests: (1,10), (1,20), (2,10), (2,20), (3,10), (3,20)"""
    result = x * y
    assert result == x * y
```

### Parameterize with Expected Exceptions

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

## Indirect Parameterization

Sometimes you want to parametrize the *fixture* input, not the test input directly. Use `indirect=True`:

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

## Built-in Fixtures

pytest provides several built-in fixtures. The most useful ones:

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

## Practical Example: Testing a Task Manager

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

## Exercises

1. **Fixture chain**: Create a three-level fixture chain (`database` -> `user_table` -> `user_record`) where each fixture depends on the previous one. Write tests that use `user_record` and verify the full chain works.

2. **Parameterized validation**: Write a `validate_username(name)` function that enforces rules (3-20 chars, alphanumeric + underscore, must start with a letter). Use `@pytest.mark.parametrize` with at least 10 cases covering valid and invalid inputs.

3. **Factory fixture**: Create a factory fixture `make_order` that generates order objects with configurable product, quantity, and discount. Write tests for a `calculate_total` function using the factory with at least 5 different configurations.

---

**License**: CC BY-NC 4.0
