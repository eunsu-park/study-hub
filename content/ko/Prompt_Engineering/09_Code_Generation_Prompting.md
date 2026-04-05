# 09. 코드 생성 프롬프팅(Code Generation Prompting)

**이전**: [멀티모달 프롬프팅](./08_Multimodal_Prompting.md) | **다음**: [RAG 프롬프트 패턴](./10_RAG_Prompt_Patterns.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 올바르고 프로덕션에 바로 적용 가능한 코드를 생성하는 명세 기반 프롬프트(Specification-Driven Prompts)를 설계하기
2. 테스트 주도 프롬프팅(Test-Driven Prompting)을 적용하여 LLM 출력을 제한하고 정확성을 검증하기
3. 효과적인 디버깅 및 오류 진단(Error-Diagnosis) 프롬프트를 구성하기
4. 실행 가능한 피드백을 제공하는 코드 리뷰(Code Review) 및 리팩토링(Refactoring) 프롬프트를 작성하기
5. 구조화된 프롬프트를 통해 다중 파일 프로젝트 구조와 문서를 생성하기

---

코드 생성(Code Generation)은 대규모 언어 모델의 가장 높은 가치의 응용 분야 중 하나이지만, 동시에 가장 오류가 발생하기 쉬운 분야이기도 합니다. "웹 서버를 작성해줘"와 같은 모호한 프롬프트는 세 줄짜리 Flask 앱부터 과도하게 설계된 마이크로서비스까지 무엇이든 생성할 수 있으며, 결과물이 컴파일되는지, 엣지 케이스를 처리하는지, 보안 모범 사례를 따르는지 보장할 수 없습니다. 아마추어와 전문적인 코드 생성 프롬프팅의 차이는 *명세(Specification)*에 있습니다: 코드가 무엇을 해야 하는지, 어떻게 구조화되어야 하는지, 어떤 제약 조건을 충족해야 하는지를 더 정밀하게 설명할수록 출력의 신뢰성이 높아집니다.

이 레슨에서는 코드 생성 프롬프팅에 대한 체계적 접근 방식을 다룹니다. 명세 기반 프롬프팅(모델에게 정확히 무엇을 만들어야 하는지 알려주기)에서 테스트 주도 프롬프팅(테스트로 정확성을 정의하기), 디버깅, 리뷰, 리팩토링 워크플로우까지 순서대로 진행합니다. 각 기법은 이전 레슨의 프롬프트 엔지니어링 기초 위에 구축됩니다.

## 목차

1. [명세 기반 프롬프팅](#1-명세-기반-프롬프팅)
2. [테스트 주도 프롬프팅](#2-테스트-주도-프롬프팅)
3. [디버깅 및 오류 진단 프롬프트](#3-디버깅-및-오류-진단-프롬프트)
4. [코드 리뷰 프롬프트](#4-코드-리뷰-프롬프트)
5. [리팩토링 프롬프트](#5-리팩토링-프롬프트)
6. [다중 파일 생성](#6-다중-파일-생성)
7. [언어별 프롬프팅 전략](#7-언어별-프롬프팅-전략)
8. [문서 생성](#8-문서-생성)
9. [코드 설명 프롬프트](#9-코드-설명-프롬프트)
10. [모범 사례와 주의사항](#10-모범-사례와-주의사항)

---

## 1. 명세 기반 프롬프팅

명세 기반 프롬프팅(Specification-Driven Prompting)은 프롬프트를 소프트웨어 명세 문서처럼 취급합니다. 코드를 높은 수준에서 설명하는 대신, 입력, 출력, 제약 조건, 오류 처리, 성능 요구사항을 명시적으로 정의합니다.

### 1.1 명세 템플릿

좋은 코드 생성 프롬프트에는 다음 요소들이 포함됩니다:

| 요소 | 설명 | 예시 |
|------|------|------|
| **함수 시그니처(Function Signature)** | 이름, 매개변수, 반환 타입 | `def parse_csv(path: str) -> list[dict]` |
| **동작 설명(Behavior Description)** | 함수가 하는 일 | "CSV 파일을 파싱하고 행을 dict로 반환" |
| **입력 제약 조건(Input Constraints)** | 유효한 범위, 타입, 형식 | "경로가 존재해야 함; CSV에 헤더 행 있음" |
| **출력 명세(Output Specification)** | 정확한 반환 형식 | "컬럼 헤더를 키로 사용하는 dict의 리스트" |
| **오류 처리(Error Handling)** | 잘못된 입력 처리 방법 | "잘못된 행에 대해 ValueError 발생" |
| **성능 참고사항(Performance Notes)** | 복잡도 또는 리소스 제한 | "스트리밍 필수; 전체 파일을 메모리에 로드하지 않을 것" |
| **의존성(Dependencies)** | 허용되는 라이브러리 | "stdlib csv 모듈만 사용" |

### 1.2 기본 명세 프롬프트

```python
import anthropic

client = anthropic.Anthropic()

spec_prompt = """
Write a Python function with the following specification:

Function: rate_limiter
Signature: def rate_limiter(max_calls: int, period_seconds: float) -> Callable
Purpose: A decorator that limits how many times a function can be called within
         a sliding time window.

Behavior:
- Track call timestamps for the decorated function
- If max_calls have been made within the last period_seconds, raise a
  RateLimitExceeded exception (define this exception class too)
- Thread-safe using threading.Lock
- The decorator should preserve the original function's signature and docstring

Constraints:
- Use only Python standard library (collections.deque, time, threading, functools)
- Time complexity for each call check: O(n) worst case where n = max_calls
- Must work with both sync functions (not async)

Return: The complete implementation including the exception class, the decorator
        factory, and a brief usage example in a __main__ block.
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[
        {"role": "user", "content": spec_prompt}
    ]
)

print(message.content[0].text)
```

### 1.3 점진적 명세(Incremental Specification)

복잡한 코드의 경우, 모든 것을 한 번에 넣기보다 점진적으로 명세를 구축합니다. 이렇게 하면 모델이 한 번에 하나의 부분에 집중할 수 있어 환각(Hallucination)이 줄어듭니다.

```python
import anthropic

client = anthropic.Anthropic()

# Step 1: Define the data model
step1 = """
Design a Python data model for a task management system.

Requirements:
- Task has: id (UUID), title (str), description (str), status (enum: todo,
  in_progress, done), priority (enum: low, medium, high, critical),
  created_at (datetime), due_date (optional datetime), tags (list[str])
- Use Python dataclasses with type hints
- Include a from_dict() classmethod and a to_dict() method
- Status transitions: todo -> in_progress -> done (no skipping, no going back)
- Add a method transition_to(new_status) that enforces valid transitions

Write ONLY the data model. No persistence, no API.
"""

response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1500,
    messages=[{"role": "user", "content": step1}]
)
data_model = response1.content[0].text

# Step 2: Build on the data model
step2 = f"""
Here is a Task data model:

{data_model}

Now write a TaskRepository class that provides in-memory CRUD operations:

Requirements:
- __init__(self) initializes an empty dict[UUID, Task] store
- add(task: Task) -> Task: stores the task, raises ValueError if ID exists
- get(task_id: UUID) -> Task: returns task, raises KeyError if not found
- list_tasks(status: Status | None = None, tag: str | None = None) -> list[Task]:
  filter by status and/or tag; both optional
- update(task_id: UUID, **fields) -> Task: partial update of allowed fields
  (title, description, priority, tags, due_date). Raise ValueError for
  unknown fields. Return updated task.
- delete(task_id: UUID) -> None: remove task, raise KeyError if not found

Write ONLY the TaskRepository class. Import the data model from above.
"""

response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1500,
    messages=[
        {"role": "user", "content": step1},
        {"role": "assistant", "content": data_model},
        {"role": "user", "content": step2}
    ]
)

print(response2.content[0].text)
```

### 1.4 제약 조건이 많은 명세

특정 기술적 요구사항을 충족하는 코드가 필요한 경우, 제약 조건을 명시적이고 눈에 띄게 만드세요:

```python
constraint_prompt = """
Write a Python function that computes the nth Fibonacci number.

HARD CONSTRAINTS (must satisfy ALL):
1. Time complexity: O(log n) -- use matrix exponentiation
2. Space complexity: O(1) auxiliary (no memoization table)
3. Must handle n up to 10^6 without overflow (use Python's arbitrary precision int)
4. Must handle negative indices using negafibonacci extension
5. No external libraries (no numpy, no sympy)
6. Include type hints and a docstring with complexity analysis

DO NOT use the naive recursive approach or simple iterative approach.
The matrix exponentiation method is required.
"""
```

핵심 인사이트: LLM은 문제에 대해 *가장 흔한* 해법을 생성하는 경향이 있습니다. 특정 알고리즘이 필요하다면 그 이름을 명시적으로 지정하고 일반적인 대안을 제외하세요.

---

## 2. 테스트 주도 프롬프팅

테스트 주도 프롬프팅(Test-Driven Prompting)은 일반적인 워크플로우를 뒤집습니다: 코드가 무엇을 해야 하는지를 산문으로 설명하는 대신, 코드가 통과해야 하는 테스트를 제공합니다. 이것은 LLM에게 모호하지 않은 정확성 기준을 제공합니다.

### 2.1 테스트 우선 패턴(Tests-First Pattern)

```python
import anthropic

client = anthropic.Anthropic()

tdd_prompt = """
Write a Python class that passes ALL of the following tests.
Do NOT modify the tests. Write ONLY the implementation.

```python
import pytest
from datetime import datetime, timedelta

def test_create_booking():
    system = BookingSystem()
    booking = system.create_booking(
        room="Conference A",
        start=datetime(2025, 1, 15, 9, 0),
        end=datetime(2025, 1, 15, 10, 0),
        organizer="alice@example.com"
    )
    assert booking.room == "Conference A"
    assert booking.organizer == "alice@example.com"
    assert booking.id is not None

def test_no_double_booking():
    system = BookingSystem()
    system.create_booking(
        room="Conference A",
        start=datetime(2025, 1, 15, 9, 0),
        end=datetime(2025, 1, 15, 10, 0),
        organizer="alice@example.com"
    )
    with pytest.raises(ConflictError):
        system.create_booking(
            room="Conference A",
            start=datetime(2025, 1, 15, 9, 30),
            end=datetime(2025, 1, 15, 10, 30),
            organizer="bob@example.com"
        )

def test_adjacent_bookings_ok():
    system = BookingSystem()
    system.create_booking(
        room="Conference A",
        start=datetime(2025, 1, 15, 9, 0),
        end=datetime(2025, 1, 15, 10, 0),
        organizer="alice@example.com"
    )
    # Adjacent booking (starts exactly when previous ends) should succeed
    booking = system.create_booking(
        room="Conference A",
        start=datetime(2025, 1, 15, 10, 0),
        end=datetime(2025, 1, 15, 11, 0),
        organizer="bob@example.com"
    )
    assert booking is not None

def test_cancel_booking():
    system = BookingSystem()
    booking = system.create_booking(
        room="Conference A",
        start=datetime(2025, 1, 15, 9, 0),
        end=datetime(2025, 1, 15, 10, 0),
        organizer="alice@example.com"
    )
    system.cancel_booking(booking.id)
    # Slot is now free; re-booking should succeed
    new_booking = system.create_booking(
        room="Conference A",
        start=datetime(2025, 1, 15, 9, 0),
        end=datetime(2025, 1, 15, 10, 0),
        organizer="charlie@example.com"
    )
    assert new_booking is not None

def test_list_bookings_by_room():
    system = BookingSystem()
    system.create_booking(
        room="Conference A",
        start=datetime(2025, 1, 15, 9, 0),
        end=datetime(2025, 1, 15, 10, 0),
        organizer="alice@example.com"
    )
    system.create_booking(
        room="Conference B",
        start=datetime(2025, 1, 15, 9, 0),
        end=datetime(2025, 1, 15, 10, 0),
        organizer="bob@example.com"
    )
    a_bookings = system.list_bookings(room="Conference A")
    assert len(a_bookings) == 1
    assert a_bookings[0].room == "Conference A"
```

Implement the BookingSystem class, Booking dataclass, and ConflictError exception
that make all tests pass. Use only Python standard library.
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[{"role": "user", "content": tdd_prompt}]
)

print(message.content[0].text)
```

### 2.2 속성 기반 테스트 프롬프팅(Property-Based Test Prompting)

예제 기반 테스트를 넘어, 코드가 충족해야 하는 *속성(Properties)*을 지정할 수 있습니다:

```python
property_prompt = """
Write a Python function `serialize(obj)` and `deserialize(data)` pair where:

PROPERTIES (must hold for ALL valid inputs):
1. Round-trip: deserialize(serialize(x)) == x for any x of supported types
2. Supported types: int, float, str, bool, None, list, dict (nested arbitrarily)
3. serialize returns bytes
4. deserialize raises DeserializeError for corrupted input
5. Format must NOT be JSON (implement a custom binary format)

PROPERTY TEST (your implementation must pass this):
```python
from hypothesis import given, strategies as st

json_like = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=50
)

@given(json_like)
def test_roundtrip(obj):
    assert deserialize(serialize(obj)) == obj
```

Write the complete implementation.
"""
```

### 2.3 명세와 테스트의 결합

가장 견고한 접근 방식은 명세와 테스트를 모두 사용합니다:

```python
hybrid_prompt = """
Write a Python class `LRUCache` with the following specification:

SPECIFICATION:
- __init__(self, capacity: int): Initialize with max capacity
- get(self, key: str) -> Any: Return value if key exists (and mark as recently used),
  otherwise raise KeyError
- put(self, key: str, value: Any) -> None: Insert or update; if at capacity, evict
  the least recently used item first
- Time complexity: O(1) for both get and put

TESTS (must all pass):
```python
def test_basic_operations():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # evicts "b" (least recently used)
    with pytest.raises(KeyError):
        cache.get("b")

def test_update_refreshes():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)  # update "a", refreshes its position
    cache.put("c", 3)   # evicts "b", not "a"
    assert cache.get("a") == 10
    with pytest.raises(KeyError):
        cache.get("b")

def test_capacity_one():
    cache = LRUCache(1)
    cache.put("a", 1)
    cache.put("b", 2)  # evicts "a"
    with pytest.raises(KeyError):
        cache.get("a")
    assert cache.get("b") == 2
```

Use collections.OrderedDict for O(1) operations.
"""
```

---

## 3. 디버깅 및 오류 진단 프롬프트

디버깅 프롬프트(Debugging Prompts)는 LLM이 기존 코드의 오류를 진단하고 수정하도록 도와줍니다. 진단의 품질은 얼마나 많은 컨텍스트를 제공하느냐에 달려 있습니다.

### 3.1 진단 프롬프트 템플릿

```python
import anthropic

client = anthropic.Anthropic()

debug_prompt = """
I have a bug in my Python code. Help me diagnose and fix it.

CODE:
```python
import asyncio
import aiohttp

async def fetch_all(urls: list[str]) -> list[str]:
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch_one(session, url))
            tasks.append(task)
        for task in tasks:
            result = await task
            results.append(result)
    return results

async def fetch_one(session, url):
    async with session.get(url) as response:
        return await response.text()

# Usage
urls = ["https://example.com/api/1", "https://example.com/api/2"]
results = asyncio.run(fetch_all(urls))
```

ERROR MESSAGE:
```
RuntimeError: Event loop is closed
```

ENVIRONMENT:
- Python 3.11.5
- aiohttp 3.9.1
- Windows 11

WHAT I ALREADY TRIED:
- Adding asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  fixes it but I want to understand WHY and if there is a better solution.

Please:
1. Explain the root cause of this error
2. Explain why the Windows policy fix works
3. Provide the most robust fix that works cross-platform
4. Show the corrected code
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[{"role": "user", "content": debug_prompt}]
)

print(message.content[0].text)
```

### 3.2 오류 패턴 진단(Error Pattern Diagnosis)

단일 버그가 아닌 오류 패턴이 있는 경우, 패턴을 중심으로 프롬프트를 구성합니다:

```python
pattern_debug = """
Analyze this pattern of test failures and identify the root cause.

TEST RESULTS (5 of 20 tests failing):

PASS: test_process_ascii_text
PASS: test_process_empty_string
FAIL: test_process_unicode_emoji - Expected "Hello 👋 World" got "Hello \ud83d World"
PASS: test_process_latin_accents - "café" works fine
FAIL: test_process_cjk - Expected "日本語" got garbled bytes
PASS: test_process_simple_utf8
FAIL: test_process_surrogate_pairs - AssertionError on 4-byte UTF-8
PASS: test_process_bmp_only - Basic Multilingual Plane chars work
FAIL: test_roundtrip_emoji - Data corrupted after save/load cycle
FAIL: test_process_musical_symbols - U+1D11E (treble clef) fails

RELEVANT CODE:
```python
def process_text(text: str) -> str:
    encoded = text.encode('utf-16-le')
    # Process two bytes at a time
    result = bytearray()
    for i in range(0, len(encoded), 2):
        char_bytes = encoded[i:i+2]
        result.extend(transform(char_bytes))
    return result.decode('utf-16-le')
```

What is the root cause? What is the correct fix?
"""
```

### 3.3 스택 트레이스 분석 프롬프트(Stack Trace Analysis Prompts)

```python
stack_trace_prompt = """
Analyze this Python stack trace and explain:
1. What the immediate error is
2. What the root cause is (may be different from the immediate error)
3. The fix

```
Traceback (most recent call last):
  File "/app/main.py", line 45, in handle_request
    result = await process_batch(items)
  File "/app/processor.py", line 112, in process_batch
    async with asyncio.TaskGroup() as tg:
  File "/usr/lib/python3.11/asyncio/taskgroups.py", line 145, in __aexit__
    raise me from None
  ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
    +-+---------------- 1 ----------------
      | Traceback (most recent call last):
      |   File "/app/processor.py", line 115, in process_batch
      |     tg.create_task(process_one(item, db_pool))
      |   File "/app/processor.py", line 128, in process_one
      |     async with db_pool.acquire() as conn:
      |   File "/usr/lib/python3.11/contextlib.py", line 210, in __aenter__
      |     return await anext(self.gen)
      |   File "/app/db.py", line 34, in acquire
      |     conn = await asyncio.wait_for(self._pool.get(), timeout=5.0)
      | asyncio.TimeoutError
      +------------------------------------
```

CONTEXT:
- This happens under load (>100 concurrent requests)
- db_pool max_size is set to 10
- Each process_one call holds the connection for ~200ms
- Batch sizes are typically 50-100 items
"""
```

### 3.4 구조화된 디버깅 워크플로우

복잡한 버그의 경우, 모델을 구조화된 분석으로 안내합니다:

```python
structured_debug = """
Perform a systematic debugging analysis of the code below.

Follow this exact structure:
1. SYMPTOMS: What observable behavior indicates the bug?
2. HYPOTHESES: List 3 possible root causes, ranked by likelihood
3. EVIDENCE: For each hypothesis, what evidence supports or refutes it?
4. ROOT CAUSE: Which hypothesis is correct and why?
5. FIX: Minimal code change to resolve the issue
6. PREVENTION: What test or check would catch this in the future?

CODE:
```python
class ConnectionPool:
    def __init__(self, max_size: int = 10):
        self._pool: list[Connection] = []
        self._max_size = max_size
        self._current_size = 0
        self._lock = threading.Lock()

    def acquire(self) -> Connection:
        with self._lock:
            if self._pool:
                return self._pool.pop()
            if self._current_size < self._max_size:
                self._current_size += 1
                return Connection()
        # Wait for a connection to be released
        while True:
            time.sleep(0.01)
            with self._lock:
                if self._pool:
                    return self._pool.pop()

    def release(self, conn: Connection):
        if conn.is_healthy():
            with self._lock:
                self._pool.append(conn)
        else:
            with self._lock:
                self._current_size -= 1
```

OBSERVED PROBLEM: After running for several hours under moderate load,
the pool reports _current_size == 10 but _pool is empty, and new calls
to acquire() block indefinitely.
"""
```

---

## 4. 코드 리뷰 프롬프트

코드 리뷰 프롬프트(Code Review Prompts)는 LLM에게 기존 코드를 정확성, 스타일, 성능, 보안 문제에 대해 평가하도록 요청합니다.

### 4.1 종합 리뷰 프롬프트

```python
import anthropic

client = anthropic.Anthropic()

review_prompt = """
Review the following Python code. Organize your feedback into these categories:

1. **BUGS**: Actual correctness issues that will cause wrong behavior
2. **SECURITY**: Vulnerabilities (injection, path traversal, etc.)
3. **PERFORMANCE**: Inefficiencies that matter at scale
4. **MAINTAINABILITY**: Readability, naming, structure issues
5. **BEST PRACTICES**: Idiomatic Python improvements

For each issue:
- Quote the specific line(s)
- Explain the problem
- Provide the corrected code

Rate severity: CRITICAL / HIGH / MEDIUM / LOW

```python
import os
import sqlite3
import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)
DB_PATH = "users.db"

def get_db():
    return sqlite3.connect(DB_PATH)

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data["username"]
    password = data["password"]
    email = data["email"]

    password_hash = hashlib.md5(password.encode()).hexdigest()

    db = get_db()
    db.execute(
        f"INSERT INTO users (username, password, email) VALUES ('{username}', '{password_hash}', '{email}')"
    )
    db.commit()
    db.close()

    return jsonify({"status": "ok"})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data["username"]
    password = data["password"]

    password_hash = hashlib.md5(password.encode()).hexdigest()

    db = get_db()
    cursor = db.execute(
        f"SELECT * FROM users WHERE username='{username}' AND password='{password_hash}'"
    )
    user = cursor.fetchone()
    db.close()

    if user:
        return jsonify({"token": hashlib.sha256(username.encode()).hexdigest()})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/profile/<username>")
def profile(username):
    db = get_db()
    cursor = db.execute(f"SELECT username, email FROM users WHERE username='{username}'")
    user = cursor.fetchone()
    db.close()
    if user:
        return jsonify({"username": user[0], "email": user[1]})
    return jsonify({"error": "Not found"}), 404

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    path = os.path.join("/uploads", file.filename)
    file.save(path)
    return jsonify({"path": path})
```
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    messages=[{"role": "user", "content": review_prompt}]
)

print(message.content[0].text)
```

### 4.2 집중 리뷰 프롬프트(Focused Review Prompts)

모델이 단일 차원에 집중하길 원할 때가 있습니다:

```python
# Security-focused review
security_review = """
Perform a security-focused code review. Identify ALL potential vulnerabilities.

For each vulnerability:
- Name the vulnerability class (e.g., CWE-89: SQL Injection)
- Show the vulnerable code
- Explain the attack vector with a concrete exploit example
- Provide the secure alternative

Focus ONLY on security. Ignore style, naming, and performance.

CODE:
[... code here ...]
"""

# Performance-focused review
perf_review = """
Analyze this code for performance issues. Assume it processes 1M+ records daily.

For each issue:
- Current time/space complexity
- Why it is a problem at scale
- Optimized alternative with complexity analysis
- Estimated improvement factor

CODE:
[... code here ...]
"""
```

### 4.3 Diff 리뷰 프롬프트

전체 파일이 아닌 코드 변경사항을 리뷰합니다:

```python
diff_review = """
Review this git diff. Focus on whether the change is correct and complete.

```diff
--- a/auth/middleware.py
+++ b/auth/middleware.py
@@ -15,8 +15,12 @@ class AuthMiddleware:
     def authenticate(self, request):
         token = request.headers.get("Authorization", "").removeprefix("Bearer ")
-        if not token:
+        if not token or token == "":
             return None
-        payload = jwt.decode(token, self.secret, algorithms=["HS256"])
-        return User.from_payload(payload)
+        try:
+            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
+            return User.from_payload(payload)
+        except jwt.ExpiredSignatureError:
+            return None
+        except jwt.InvalidTokenError:
+            return None
```

Questions to answer:
1. Does this change fix the intended problem?
2. Does it introduce any new issues?
3. Are there any missing edge cases?
4. Is the error handling appropriate (should expired tokens return None or raise)?
"""
```

---

## 5. 리팩토링 프롬프트

리팩토링 프롬프트(Refactoring Prompts)는 모델에게 동작을 유지하면서 코드를 재구성하도록 요청합니다. 핵심 과제는 *동작 동등성(Behavioral Equivalence)* 보장 -- 리팩토링된 코드가 원본이 했던 것과 정확히 같은 일을 해야 합니다.

### 5.1 패턴 기반 리팩토링

```python
import anthropic

client = anthropic.Anthropic()

refactor_prompt = """
Refactor the following code to use the Strategy pattern.

CURRENT CODE:
```python
class ReportGenerator:
    def generate(self, data: list[dict], format: str) -> str:
        if format == "csv":
            header = ",".join(data[0].keys())
            rows = []
            for row in data:
                rows.append(",".join(str(v) for v in row.values()))
            return header + "\\n" + "\\n".join(rows)
        elif format == "json":
            import json
            return json.dumps(data, indent=2)
        elif format == "html":
            html = "<table>\\n<tr>"
            html += "".join(f"<th>{k}</th>" for k in data[0].keys())
            html += "</tr>\\n"
            for row in data:
                html += "<tr>"
                html += "".join(f"<td>{v}</td>" for v in row.values())
                html += "</tr>\\n"
            html += "</table>"
            return html
        elif format == "markdown":
            headers = list(data[0].keys())
            lines = ["| " + " | ".join(headers) + " |"]
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in data:
                lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
            return "\\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")
```

REQUIREMENTS:
1. Use the Strategy pattern with a base class and one concrete class per format
2. New formats should require adding only a new class (Open/Closed Principle)
3. Include a registry mechanism so formats can be looked up by name
4. Preserve exact output behavior for all four formats
5. Add type hints throughout
6. Show a brief demonstration that the refactored version produces identical output

DO NOT change the output format of any existing format type.
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    messages=[{"role": "user", "content": refactor_prompt}]
)

print(message.content[0].text)
```

### 5.2 성능 리팩토링

```python
perf_refactor = """
Refactor this code for performance. It currently processes 10,000 records in ~45 seconds.
Target: under 2 seconds.

```python
import re

def find_duplicates(records: list[dict]) -> list[tuple[int, int]]:
    \"\"\"Find all pairs of records that are likely duplicates based on name similarity.\"\"\"
    duplicates = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            name_i = normalize(records[i]["name"])
            name_j = normalize(records[j]["name"])
            if similarity(name_i, name_j) > 0.85:
                duplicates.append((i, j))
    return duplicates

def normalize(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9\\s]', '', name)
    name = re.sub(r'\\s+', ' ', name)
    return name

def similarity(a: str, b: str) -> float:
    \"\"\"Jaccard similarity on character trigrams.\"\"\"
    trigrams_a = set(a[i:i+3] for i in range(len(a) - 2))
    trigrams_b = set(b[i:i+3] for i in range(len(b) - 2))
    if not trigrams_a or not trigrams_b:
        return 0.0
    intersection = trigrams_a & trigrams_b
    union = trigrams_a | trigrams_b
    return len(intersection) / len(union)
```

CONSTRAINTS:
- Must produce the exact same results as the original
- May use standard library only (no external packages)
- Explain each optimization and its expected impact

Show the optimized code with comments explaining the performance improvements.
"""
```

### 5.3 현대화 리팩토링(Modernization Refactoring)

```python
modernize_prompt = """
Modernize this Python 2-style code to idiomatic Python 3.12+.

```python
class Config(object):
    def __init__(self, **kwargs):
        self._data = dict()
        for key, value in kwargs.iteritems():
            self._data[key] = value

    def get(self, key, default=None):
        if self._data.has_key(key):
            return self._data[key]
        return default

    def set(self, key, value):
        self._data[key] = value

    def keys(self):
        return self._data.keys()

    def __repr__(self):
        return u"Config(%s)" % unicode(self._data)

    def merge(self, other):
        if not isinstance(other, Config):
            raise TypeError, "Expected Config instance"
        for k, v in other._data.iteritems():
            if not self._data.has_key(k):
                self._data[k] = v

    def to_env(self):
        import os
        for k, v in self._data.iteritems():
            os.environ[k.upper()] = str(v)
```

Apply ALL relevant modernizations:
- Python 3 syntax (no object base, f-strings, etc.)
- Type hints
- Data class or __slots__ if appropriate
- Modern dict operations
- Proper exception syntax
- Any other Python 3.12+ improvements

Preserve ALL existing functionality.
"""
```

---

## 6. 다중 파일 생성

실제 프로젝트는 여러 파일에 걸쳐 있습니다. 다중 파일 생성(Multi-File Generation) 프롬프트는 프로젝트 구조, 파일 간 의존성, 컴포넌트 통합 방법을 지정해야 합니다.

### 6.1 프로젝트 스캐폴드 프롬프트(Project Scaffold Prompt)

```python
import anthropic

client = anthropic.Anthropic()

scaffold_prompt = """
Generate a Python project structure for a CLI todo application.

PROJECT STRUCTURE:
```
todo_cli/
├── __init__.py
├── __main__.py          # Entry point (python -m todo_cli)
├── cli.py               # Click-based CLI commands
├── models.py            # SQLAlchemy models (Task, Tag)
├── database.py          # DB session management
├── services.py          # Business logic layer
└── config.py            # Configuration (DB path, defaults)
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_models.py
└── test_services.py
pyproject.toml           # Project metadata and dependencies
```

SPECIFICATIONS:
- CLI framework: Click 8.x
- Database: SQLite via SQLAlchemy 2.x (async not required)
- Commands: add, list, complete, delete, tag, search
- Task fields: id, title, description, status, priority, due_date, created_at, tags
- Use dependency injection for the database session in services

For each file, generate the COMPLETE implementation. Mark each file clearly with
its path as a header:

## todo_cli/__init__.py
```python
...
```

## todo_cli/cli.py
```python
...
```

(continue for all files)

Important: Files must be mutually consistent -- imports must match actual
module contents, model fields must match service method signatures, CLI
arguments must match service parameters.
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    messages=[{"role": "user", "content": scaffold_prompt}]
)

print(message.content[0].text)
```

### 6.2 컴포넌트 통합 프롬프트(Component Integration Prompt)

기존 코드와 통합해야 하는 새 컴포넌트를 생성할 때:

```python
integration_prompt = """
I have an existing Flask API. Generate a new module that adds WebSocket support.

EXISTING CODE CONTEXT:

```python
# app.py (existing - DO NOT modify)
from flask import Flask
app = Flask(__name__)

# Already has: /api/tasks (CRUD), /api/users, authentication middleware
# Uses: SQLAlchemy for DB, Redis for caching
# Auth: JWT tokens in Authorization header
```

```python
# models.py (existing - DO NOT modify)
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    status = db.Column(db.String(20))  # 'todo', 'in_progress', 'done'
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'))
```

GENERATE these new files:

1. websocket.py - Flask-SocketIO setup with events:
   - task_updated: broadcast when any task changes status
   - user_typing: broadcast to task assignees when someone is typing a comment
   - presence: track online users per task board

2. events.py - Event handlers that bridge REST changes to WebSocket broadcasts

3. test_websocket.py - Tests using flask-socketio test client

Requirements:
- Must authenticate WebSocket connections using the same JWT tokens
- Must work with the existing Redis instance for pub/sub
- Must NOT require changes to existing files
"""
```

---

## 7. 언어별 프롬프팅 전략

프로그래밍 언어마다 관용구, 주의사항, 규칙이 다릅니다. 프롬프트를 그에 맞게 조정하세요.

### 7.1 Python 전용 프롬프트

```python
python_prompt = """
Write a Python context manager for database transactions.

PYTHON-SPECIFIC REQUIREMENTS:
- Use @contextmanager decorator from contextlib
- Support both `with` statement and async `async with` (provide both versions)
- Follow PEP 8 naming conventions
- Include type hints using Python 3.12+ syntax (PEP 695 type aliases if appropriate)
- Use __all__ to control public API
- Include docstrings in Google style format
"""
```

### 7.2 TypeScript 전용 프롬프트

```python
ts_prompt = """
Write a TypeScript generic Result type and utility functions.

TYPESCRIPT-SPECIFIC REQUIREMENTS:
- Use discriminated unions (not classes) for Result<T, E>
- Provide type guards (isOk, isErr)
- Ensure full type inference (no explicit type parameters needed at call sites)
- Use 'satisfies' operator where it improves type safety
- Include JSDoc comments
- Target ES2022+ (use modern syntax)
- Export types and functions separately
"""
```

### 7.3 Rust 전용 프롬프트

```python
rust_prompt = """
Write a Rust implementation of a thread-safe LRU cache.

RUST-SPECIFIC REQUIREMENTS:
- Use proper lifetime annotations
- Implement Send + Sync
- Use Arc<Mutex<>> for thread safety (explain why RwLock might be better)
- Implement the standard library traits: Debug, Clone, Default
- Handle the borrow checker properly (no unsafe code)
- Include comprehensive documentation comments (///)
- Add #[cfg(test)] module with tests
"""
```

### 7.4 언어 간 프롬프팅(Prompting Across Languages)

```python
cross_lang_prompt = """
Implement the same algorithm (Levenshtein edit distance) in three languages.
For each version, follow that language's conventions:

1. Python: Type hints, docstring, snake_case
2. Go: Exported function, error handling, GoDoc comment
3. Rust: Generic over AsRef<str>, documentation, #[test]

For each implementation:
- Use the language's idiomatic style (not a line-by-line port)
- Include the standard testing approach for that language
- Note any language-specific optimizations
"""
```

---

## 8. 문서 생성

LLM은 코드에서 문서를 생성하는 데 뛰어나지만, 품질은 문서 형식과 대상 독자를 지정하는 것에 달려 있습니다.

### 8.1 API 문서 프롬프트

```python
import anthropic

client = anthropic.Anthropic()

doc_prompt = """
Generate API documentation for the following Python module.

FORMAT: Google-style docstrings + a module-level overview

AUDIENCE: Developers who will use this library (not contributors)

REQUIREMENTS:
- Module docstring with overview, quick start example, and installation note
- Each public function/class gets a docstring with:
  - One-line summary
  - Extended description (if the behavior is non-obvious)
  - Args section with types and descriptions
  - Returns section
  - Raises section (if applicable)
  - Example section with runnable code
- Do NOT document private methods (single underscore prefix)
- Do NOT change any code logic -- only add/improve docstrings

```python
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

@dataclass
class TokenConfig:
    secret: bytes
    ttl_seconds: int = 3600
    algorithm: str = "sha256"

class TokenManager:
    def __init__(self, config: TokenConfig):
        self._config = config

    def create_token(self, user_id: str, scopes: list[str] | None = None) -> str:
        timestamp = str(int(time.time()))
        scope_str = ",".join(sorted(scopes or []))
        payload = f"{user_id}:{timestamp}:{scope_str}"
        signature = hmac.new(
            self._config.secret,
            payload.encode(),
            self._config.algorithm
        ).hexdigest()
        token = f"{payload}:{signature}"
        return token

    def verify_token(self, token: str) -> dict:
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            raise InvalidTokenError("Malformed token")
        payload, signature = parts
        expected = hmac.new(
            self._config.secret,
            payload.encode(),
            self._config.algorithm
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise InvalidTokenError("Invalid signature")
        user_id, timestamp, scope_str = payload.split(":", 2)
        age = time.time() - int(timestamp)
        if age > self._config.ttl_seconds:
            raise TokenExpiredError(f"Token expired {age - self._config.ttl_seconds:.0f}s ago")
        scopes = [s for s in scope_str.split(",") if s]
        return {"user_id": user_id, "scopes": scopes, "age_seconds": age}

    def _generate_nonce(self) -> str:
        return secrets.token_hex(16)

class InvalidTokenError(Exception):
    pass

class TokenExpiredError(InvalidTokenError):
    pass
```

Generate the documented version of this module.
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    messages=[{"role": "user", "content": doc_prompt}]
)

print(message.content[0].text)
```

### 8.2 README 생성

```python
readme_prompt = """
Generate a README.md for a Python library based on the following source files.

TONE: Professional, concise, developer-friendly
SECTIONS: Overview, Installation, Quick Start, API Reference (summary table),
          Configuration, Error Handling, Contributing, License

SOURCE FILES:
[... paste key source files ...]

CONSTRAINTS:
- Quick Start example must be copy-paste runnable
- API Reference should be a table, not full docs (link to full docs)
- Include badges: PyPI version, Python version, License
- Keep total length under 200 lines
"""
```

### 8.3 Git Diff에서 변경 로그 생성

```python
changelog_prompt = """
Generate a CHANGELOG entry from this git diff. Follow Keep a Changelog format.

CATEGORIES: Added, Changed, Deprecated, Removed, Fixed, Security

```diff
[... paste git diff ...]
```

Rules:
- Each entry is a single sentence, user-facing language (not implementation details)
- Group by category
- If a change is breaking, prefix with **BREAKING:**
- Include PR/issue numbers if they appear in commit messages
"""
```

---

## 9. 코드 설명 프롬프트

설명 프롬프트(Explanation Prompts)는 방향을 반대로 합니다: 코드를 생성하는 대신, 모델이 기존 코드를 분석하고 설명합니다.

### 9.1 다단계 설명(Multi-Level Explanation)

```python
import anthropic

client = anthropic.Anthropic()

explain_prompt = """
Explain the following code at three levels of detail.

LEVEL 1 - Executive Summary (2-3 sentences):
What does this code do and why would someone use it?

LEVEL 2 - Technical Overview (1 paragraph per major component):
How is it structured? What are the key design decisions?

LEVEL 3 - Line-by-Line Analysis:
Walk through the non-obvious parts, explaining WHY each decision was made
(not just WHAT the code does).

```python
from __future__ import annotations
import sys
from typing import Any, Callable, TypeVar

T = TypeVar("T")

class Signal:
    __slots__ = ("_receivers", "_sender_cache")

    def __init__(self):
        self._receivers: list[tuple[int, Callable]] = []
        self._sender_cache: dict[int, list[Callable]] = {}

    def connect(self, receiver: Callable, sender: Any = None, weak: bool = True) -> None:
        lookup_key = id(sender) if sender is not None else 0
        if weak:
            import weakref
            ref = weakref.ref(receiver, self._cleanup)
            receiver_id = id(receiver)
            self._receivers.append((lookup_key, ref))
        else:
            self._receivers.append((lookup_key, receiver))
        self._sender_cache.clear()

    def send(self, sender: Any = None, **kwargs) -> list[tuple[Callable, Any]]:
        lookup_key = id(sender) if sender is not None else 0
        responses = []
        for receiver in self._live_receivers(lookup_key):
            response = receiver(signal=self, sender=sender, **kwargs)
            responses.append((receiver, response))
        return responses

    def _live_receivers(self, sender_key: int) -> list[Callable]:
        if sender_key in self._sender_cache:
            return self._sender_cache[sender_key]
        receivers = []
        for (key, receiver) in self._receivers[:]:
            if key in (0, sender_key):
                if isinstance(receiver, type(lambda: None)):
                    receivers.append(receiver)
                else:
                    strong = receiver()
                    if strong is not None:
                        receivers.append(strong)
                    else:
                        self._receivers.remove((key, receiver))
        self._sender_cache[sender_key] = receivers
        return receivers

    def _cleanup(self, ref):
        self._receivers = [(k, r) for k, r in self._receivers if r is not ref]
        self._sender_cache.clear()
```
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    messages=[{"role": "user", "content": explain_prompt}]
)

print(message.content[0].text)
```

### 9.2 복잡도 분석 프롬프트

```python
complexity_prompt = """
Analyze the time and space complexity of each function in this code.

For each function, provide:
1. Time complexity (Big-O) with justification
2. Space complexity (Big-O) with justification
3. Best case, average case, worst case if they differ
4. Any hidden costs (e.g., string concatenation in a loop, hash collisions)

Present as a table, then give detailed analysis for any function worse than O(n log n).

```python
[... code here ...]
```
"""
```

### 9.3 아키텍처 설명

```python
architecture_prompt = """
Analyze this codebase structure and explain the architecture.

FILE TREE:
```
src/
├── api/
│   ├── routes.py (150 lines)
│   ├── middleware.py (80 lines)
│   └── schemas.py (200 lines)
├── core/
│   ├── events.py (120 lines)
│   ├── commands.py (180 lines)
│   └── queries.py (90 lines)
├── domain/
│   ├── models.py (300 lines)
│   ├── services.py (250 lines)
│   └── repositories.py (100 lines)
├── infrastructure/
│   ├── database.py (80 lines)
│   ├── cache.py (60 lines)
│   └── messaging.py (90 lines)
└── config.py (40 lines)
```

KEY FILES:
[... paste contents of 2-3 key files ...]

Questions:
1. What architectural pattern is this? (MVC, Clean Architecture, CQRS, etc.)
2. Draw an ASCII dependency diagram showing which layers depend on which
3. What are the benefits and trade-offs of this architecture?
4. Where would you add a new feature that requires a background job?
"""
```

---

## 10. 모범 사례와 주의사항

### 10.1 모범 사례 요약

| 사례 | 효과적인 이유 |
|------|-------------|
| 함수 시그니처 포함 | 정확한 타입과 이름 지정으로 모델 고정 |
| 테스트 케이스 제공 | 모호하지 않은 정확성 기준 제공 |
| 라이브러리와 버전 지정 | 오래된 API 사용 방지 |
| 원하지 않는 접근 방식 제외 | LLM은 가장 흔한 해법을 기본으로 사용 |
| 복잡한 작업을 단계로 분리 | 환각 감소 및 일관성 향상 |
| 오류 예시 포함 | 모델에게 엣지 케이스에 대해 교육 |
| 코딩 스타일 지정 | 생성 간 불일치 방지 |

### 10.2 주의사항(Common Pitfalls)

**주의사항 1: 모호한 명세**

```python
# BAD: Vague prompt
"Write a function to process data"

# GOOD: Specific prompt
"Write a function process_csv(path: str) -> pd.DataFrame that reads a CSV file,
drops rows where all values are NaN, converts the 'date' column to datetime,
and returns the cleaned DataFrame. Raise FileNotFoundError if path doesn't exist."
```

**주의사항 2: 언어 버전과 의존성 미지정**

```python
# BAD: Model might use deprecated APIs
"Write an async HTTP client in Python"

# GOOD: Pinned versions
"Write an async HTTP client using Python 3.12 and aiohttp 3.9.
Use the modern async with syntax, not the deprecated callback API."
```

**주의사항 3: 한 번에 너무 많은 것을 요청**

```python
# BAD: Entire application in one prompt
"Write a complete e-commerce backend with authentication, products, orders,
payments, shipping, reviews, admin panel, and search."

# GOOD: One module at a time
"Write the order processing module. Here are the existing models it must
integrate with: [models]. Here are the API contracts it must implement: [specs]."
```

**주의사항 4: 생성된 코드를 검증하지 않음**

항상 생성된 코드를 최소한 다음을 통해 실행하세요:
1. 구문 검사(compile)
2. 타입 검사(mypy/pyright)
3. 린터(ruff/flake8)
4. 단위 테스트
5. 엣지 케이스에 대한 수동 리뷰

**주의사항 5: import와 의존성을 신뢰하기**

LLM은 때때로 존재하지 않는 모듈을 import하거나 존재하지 않는 API 함수를 사용합니다. import된 모듈과 호출된 함수가 실제로 존재하는지 항상 확인하세요.

### 10.3 코드 생성 프롬프트 체크리스트

코드 생성 프롬프트를 보내기 전에 확인하세요:

- [ ] 함수/클래스 이름과 시그니처가 지정되었는가
- [ ] 입력 타입과 제약 조건이 정의되었는가
- [ ] 출력 형식과 타입이 지정되었는가
- [ ] 오류 처리 동작이 설명되었는가
- [ ] 엣지 케이스가 언급되었는가
- [ ] 성능 요구사항이 명시되었는가 (관련된 경우)
- [ ] 언어 버전과 의존성이 고정되었는가
- [ ] 코딩 스타일/규칙이 지정되었는가
- [ ] 하지 말아야 할 것 (부정적 제약 조건)
- [ ] 테스트 케이스 또는 수락 기준이 포함되었는가

---

## 연습문제

### 연습문제 1: 명세 기반 생성

Python `PasswordValidator` 클래스를 생성하는 명세 프롬프트를 작성하세요. 검증기는 구성 가능한 규칙(최소 길이, 대문자 필수, 숫자 필수, 특수 문자 필수, 일반적인 비밀번호 금지)에 대해 비밀번호를 검사해야 합니다. 섹션 1.1의 명세 요소를 최소 6개 포함하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

prompt = """
Write a Python class with the following specification:

Class: PasswordValidator
File: password_validator.py

Constructor:
    __init__(self, min_length: int = 8, max_length: int = 128,
             require_uppercase: bool = True, require_lowercase: bool = True,
             require_digit: bool = True, require_special: bool = True,
             banned_passwords_file: str | None = None)

Methods:
    validate(password: str) -> ValidationResult
        - Returns a ValidationResult dataclass with:
          - is_valid: bool
          - errors: list[str] (human-readable error messages)
        - Checks applied in this order:
          1. Length check (min and max)
          2. Character class checks (uppercase, lowercase, digit, special)
          3. Common password check (if banned_passwords_file provided)
        - Special characters defined as: !@#$%^&*()_+-=[]{}|;:,.<>?

    estimate_strength(password: str) -> str
        - Returns "weak", "moderate", "strong", or "very_strong"
        - Based on: length, character diversity, and entropy estimation

Dependencies: Python standard library only (no external packages)
Error handling:
    - Raise ValueError if min_length < 1 or max_length < min_length
    - Raise FileNotFoundError if banned_passwords_file does not exist
    - Raise TypeError if password is not a string

Performance: validate() must be O(n) where n = len(password), except for
    banned password lookup which should use a set for O(1) average case.

Include:
- The ValidationResult dataclass
- The PasswordValidator class
- A __main__ block demonstrating usage with all features
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    messages=[{"role": "user", "content": prompt}]
)
print(message.content[0].text)
```

포함된 핵심 요소: 함수 시그니처, 동작 설명, 입력 제약 조건, 출력 명세, 오류 처리, 성능 요구사항, 의존성 제약 조건.

</details>

### 연습문제 2: 테스트 주도 프롬프팅

`RingBuffer` 클래스에 대한 테스트 주도 프롬프트를 작성하세요. 동작을 완전히 지정하는 최소 5개의 테스트 함수를 제공한 다음, 모든 테스트를 통과하는 클래스를 구현하도록 모델에게 요청하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

prompt = """
Write a Python class `RingBuffer` that passes ALL of the following tests.
Do NOT modify the tests. Write ONLY the implementation.

```python
import pytest

def test_basic_append_and_read():
    buf = RingBuffer(capacity=3)
    buf.append(1)
    buf.append(2)
    buf.append(3)
    assert list(buf) == [1, 2, 3]
    assert len(buf) == 3

def test_overwrites_oldest():
    buf = RingBuffer(capacity=3)
    buf.append(1)
    buf.append(2)
    buf.append(3)
    buf.append(4)  # overwrites 1
    assert list(buf) == [2, 3, 4]
    assert len(buf) == 3

def test_getitem():
    buf = RingBuffer(capacity=3)
    buf.append("a")
    buf.append("b")
    buf.append("c")
    assert buf[0] == "a"  # oldest
    assert buf[-1] == "c"  # newest
    buf.append("d")  # overwrites "a"
    assert buf[0] == "b"  # new oldest
    assert buf[-1] == "d"

def test_empty_buffer():
    buf = RingBuffer(capacity=5)
    assert len(buf) == 0
    assert list(buf) == []
    with pytest.raises(IndexError):
        _ = buf[0]

def test_is_full():
    buf = RingBuffer(capacity=2)
    assert not buf.is_full
    buf.append(1)
    assert not buf.is_full
    buf.append(2)
    assert buf.is_full
    buf.append(3)  # overwrite
    assert buf.is_full

def test_clear():
    buf = RingBuffer(capacity=3)
    buf.append(1)
    buf.append(2)
    buf.clear()
    assert len(buf) == 0
    assert list(buf) == []
    assert not buf.is_full

def test_capacity_one():
    buf = RingBuffer(capacity=1)
    buf.append("x")
    assert list(buf) == ["x"]
    buf.append("y")
    assert list(buf) == ["y"]
    assert len(buf) == 1
```

Implement the RingBuffer class using a fixed-size list and index tracking.
Use __iter__, __len__, __getitem__ for the protocol methods.
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    messages=[{"role": "user", "content": prompt}]
)
print(message.content[0].text)
```

테스트는 다음을 커버합니다: 기본 연산, 덮어쓰기 동작, 인덱싱(양수 및 음수), 빈 상태, 가득 찬 상태 추적, clear 메서드, 엣지 케이스(capacity=1).

</details>

### 연습문제 3: 디버깅 프롬프트

다음 버그가 있는 코드가 있습니다. 모델이 버그를 식별하고 수정할 수 있을 만큼 충분한 컨텍스트를 제공하는 디버깅 프롬프트를 작성하세요.

```python
import threading
import time

class Throttle:
    def __init__(self, rate: float):
        """Allow one call per `rate` seconds."""
        self.rate = rate
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.rate:
                time.sleep(self.rate - elapsed)
            self.last_call = time.time()
```

버그: 동시 접근 시, 여러 스레드가 동시에 sleep할 수 있고 그 후 모두 한꺼번에 진행되어 스로틀이 무효화됩니다.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

prompt = """
I have a bug in my Throttle class. Help me diagnose and fix it.

CODE:
```python
import threading
import time

class Throttle:
    def __init__(self, rate: float):
        self.rate = rate
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.rate:
                time.sleep(self.rate - elapsed)
            self.last_call = time.time()
```

OBSERVED PROBLEM:
When 10 threads call wait() concurrently, they should be serialized
(one call every `rate` seconds). Instead, multiple threads get through
simultaneously at the throttle boundary.

MY ANALYSIS:
The lock is held during sleep(), which should serialize access. But
under load testing with rate=0.1 and 10 threads, I see bursts of
3-4 calls within 10ms.

ENVIRONMENT: Python 3.12, Linux, 8-core CPU

Please:
1. Identify the root cause (why does the lock not prevent concurrent calls?)
2. Explain the race condition step by step
3. Provide a corrected implementation
4. Add a test that demonstrates the fix works
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    messages=[{"role": "user", "content": prompt}]
)
print(message.content[0].text)
```

근본 원인: `time.sleep()`이 잠금 *내부*에서 호출되어 다른 스레드의 확인을 차단합니다. 그러나 `self.last_call = time.time()`이 sleep *후에* 설정됩니다. sleep 지속 시간이 매우 짧은 경우(예: elapsed가 rate에 거의 동일한 경우), 여러 스레드가 업데이트하기 전에 같은 오래된 `self.last_call` 값을 읽을 수 있습니다. 수정 방법은 sleep *전에* `self.last_call`을 업데이트하고(목표 시간으로 설정) 잠금 *외부*에서 sleep하는 것입니다:

```python
def wait(self):
    with self.lock:
        now = time.time()
        target = self.last_call + self.rate
        if target > now:
            sleep_time = target - now
        else:
            sleep_time = 0
            target = now
        self.last_call = target  # Reserve the slot immediately
    if sleep_time > 0:
        time.sleep(sleep_time)
```

</details>

### 연습문제 4: 코드 리뷰 프롬프트

다음 함수에 대한 코드 리뷰 프롬프트를 작성하세요. 프롬프트는 모델에게 최소 세 가지 구체적인 품질 차원을 검사하도록 요청해야 합니다.

```python
def sync_users(source_db, target_db):
    source_users = source_db.query("SELECT * FROM users")
    for user in source_users:
        existing = target_db.query(f"SELECT id FROM users WHERE email='{user['email']}'")
        if existing:
            target_db.execute(f"UPDATE users SET name='{user['name']}', role='{user['role']}' WHERE email='{user['email']}'")
        else:
            target_db.execute(f"INSERT INTO users (email, name, role) VALUES ('{user['email']}', '{user['name']}', '{user['role']}')")
    target_db.commit()
```

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

prompt = """
Review the following Python function across these dimensions:

1. **SECURITY**: Check for SQL injection, data leakage, input validation
2. **CORRECTNESS**: Check for edge cases, error handling, data integrity
3. **PERFORMANCE**: Check for N+1 queries, missing batching, scalability
4. **RESILIENCE**: Check for failure modes, partial completion, rollback

For each issue found:
- Category and severity (CRITICAL / HIGH / MEDIUM / LOW)
- The problematic code (quote it)
- Attack vector or failure scenario (be specific)
- Corrected code

After the review, provide a fully rewritten version that fixes ALL issues.

```python
def sync_users(source_db, target_db):
    source_users = source_db.query("SELECT * FROM users")
    for user in source_users:
        existing = target_db.query(f"SELECT id FROM users WHERE email='{user['email']}'")
        if existing:
            target_db.execute(f"UPDATE users SET name='{user['name']}', role='{user['role']}' WHERE email='{user['email']}'")
        else:
            target_db.execute(f"INSERT INTO users (email, name, role) VALUES ('{user['email']}', '{user['name']}', '{user['role']}')")
    target_db.commit()
```
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    messages=[{"role": "user", "content": prompt}]
)
print(message.content[0].text)
```

예상되는 발견사항:
- **CRITICAL (보안)**: f-string 포매팅을 통한 SQL 인젝션(SQL Injection) -- 매개변수화된 쿼리 사용
- **HIGH (성능)**: N+1 쿼리 패턴 -- 사용자당 하나의 SELECT; 배치 UPSERT 사용
- **HIGH (정확성)**: 오류 처리 없음 -- 한 사용자가 실패하면 전체에 대한 commit이 건너뛰어짐
- **MEDIUM (정확성)**: SELECT *는 소스에서 불필요한 데이터를 가져옴
- **MEDIUM (복원력)**: target_db에 트랜잭션 없음 -- 실패 시 부분 업데이트
- **LOW (정확성)**: 아포스트로피가 있는 이름(O'Brien)이 SQL을 깨뜨림

</details>

### 연습문제 5: 다중 파일 생성

데이터 접근 계층, 비즈니스 로직 계층, API 계층의 세 파일로 구성된 Python 패키지를 생성하는 프롬프트를 작성하세요. 패키지는 간단한 재고 시스템(이름, 수량, 가격이 있는 제품)을 관리해야 합니다. 모든 import와 메서드 호출이 일관되도록 파일 간 계약을 충분히 명확하게 지정하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

prompt = """
Generate a Python package for an inventory management system.

STRUCTURE:
```
inventory/
├── __init__.py        # Export public API
├── repository.py      # Data access layer (SQLite)
├── service.py         # Business logic layer
└── api.py             # FastAPI routes
```

CONTRACTS BETWEEN LAYERS:

repository.py exports class ProductRepository:
    __init__(self, db_path: str)
    create(self, name: str, quantity: int, price_cents: int) -> Product
    get(self, product_id: int) -> Product | None
    list_all(self) -> list[Product]
    update_quantity(self, product_id: int, delta: int) -> Product
    delete(self, product_id: int) -> bool

    Product is a dataclass with: id (int), name (str), quantity (int),
    price_cents (int), created_at (datetime), updated_at (datetime)

service.py exports class InventoryService:
    __init__(self, repo: ProductRepository)  # Dependency injection
    add_product(self, name: str, quantity: int, price: float) -> Product
        - Converts price (dollars) to price_cents
        - Validates: name non-empty, quantity >= 0, price > 0
        - Raises ValueError with descriptive message on invalid input
    restock(self, product_id: int, quantity: int) -> Product
        - quantity must be > 0
        - Raises ProductNotFoundError if product doesn't exist
    sell(self, product_id: int, quantity: int) -> Product
        - Raises InsufficientStockError if quantity > current stock
        - Raises ProductNotFoundError if product doesn't exist
    get_low_stock(self, threshold: int = 10) -> list[Product]
        - Returns products with quantity <= threshold

api.py exports a FastAPI router:
    POST /products  (body: {name, quantity, price})  -> Product JSON
    GET  /products  -> list of Product JSON
    POST /products/{id}/restock  (body: {quantity})  -> Product JSON
    POST /products/{id}/sell     (body: {quantity})  -> Product JSON
    GET  /products/low-stock?threshold=10  -> list of Product JSON
    - All routes use InventoryService
    - Return proper HTTP status codes (201 for create, 404 for not found, 409 for insufficient stock)

REQUIREMENTS:
- All files must use type hints
- repository.py uses sqlite3 from stdlib
- service.py depends ONLY on repository.py (no direct DB access)
- api.py depends ONLY on service.py (no direct repository access)
- Include error classes in service.py: ProductNotFoundError, InsufficientStockError
- __init__.py re-exports the key classes

Generate the COMPLETE implementation for all four files.
Mark each file clearly with its path as a header.
"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=6000,
    messages=[{"role": "user", "content": prompt}]
)
print(message.content[0].text)
```

이 프롬프트의 핵심은 계층 간에 공유되는 정확한 메서드 시그니처와 타입을 정의하는 명시적 CONTRACT 섹션입니다. 이것은 가장 흔한 다중 파일 생성 오류인 모듈 간 인터페이스 불일치를 방지합니다.

</details>

---

**이전**: [멀티모달 프롬프팅](./08_Multimodal_Prompting.md) | **다음**: [RAG 프롬프트 패턴](./10_RAG_Prompt_Patterns.md)
