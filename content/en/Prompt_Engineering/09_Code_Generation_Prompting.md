# 09. Code Generation Prompting

**Previous**: [Multimodal Prompting](./08_Multimodal_Prompting.md) | **Next**: [RAG Prompt Patterns](./10_RAG_Prompt_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design specification-driven prompts that produce correct, production-ready code
2. Apply test-driven prompting to constrain LLM output and verify correctness
3. Construct effective debugging and error-diagnosis prompts
4. Write code review and refactoring prompts that yield actionable feedback
5. Generate multi-file project structures and documentation through structured prompts

---

Code generation is one of the highest-value applications of large language models, yet it is also one of the most error-prone. A vague prompt like "write me a web server" can produce anything from a three-line Flask app to an over-engineered microservice -- and the result may or may not compile, handle edge cases, or follow security best practices. The difference between amateur and professional code generation prompting lies in *specification*: the more precisely you describe what the code should do, how it should be structured, and what constraints it must satisfy, the more reliable the output becomes.

This lesson covers a systematic approach to code generation prompting. We move from specification-driven prompting (telling the model exactly what to build) through test-driven prompting (letting tests define correctness) to debugging, review, and refactoring workflows. Each technique builds on the prompt engineering fundamentals from earlier lessons.

## Table of Contents

1. [Specification-Driven Prompting](#1-specification-driven-prompting)
2. [Test-Driven Prompting](#2-test-driven-prompting)
3. [Debugging and Error-Diagnosis Prompts](#3-debugging-and-error-diagnosis-prompts)
4. [Code Review Prompts](#4-code-review-prompts)
5. [Refactoring Prompts](#5-refactoring-prompts)
6. [Multi-File Generation](#6-multi-file-generation)
7. [Language-Specific Prompting Strategies](#7-language-specific-prompting-strategies)
8. [Documentation Generation](#8-documentation-generation)
9. [Code Explanation Prompts](#9-code-explanation-prompts)
10. [Best Practices and Pitfalls](#10-best-practices-and-pitfalls)

---

## 1. Specification-Driven Prompting

Specification-driven prompting treats the prompt as a software specification document. Instead of describing code at a high level, you define inputs, outputs, constraints, error handling, and performance requirements explicitly.

### 1.1 The Specification Template

A good code generation prompt includes these elements:

| Element | Description | Example |
|---------|-------------|---------|
| **Function signature** | Name, parameters, return type | `def parse_csv(path: str) -> list[dict]` |
| **Behavior description** | What the function does | "Parses a CSV file and returns rows as dicts" |
| **Input constraints** | Valid ranges, types, formats | "Path must exist; CSV has a header row" |
| **Output specification** | Exact return format | "List of dicts keyed by column headers" |
| **Error handling** | How to handle invalid input | "Raise ValueError for malformed rows" |
| **Performance notes** | Complexity or resource limits | "Must stream; do not load entire file into memory" |
| **Dependencies** | Allowed libraries | "Use only stdlib csv module" |

### 1.2 Basic Specification Prompt

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

### 1.3 Incremental Specification

For complex code, build the specification incrementally rather than dumping everything at once. This reduces hallucination by letting the model focus on one piece at a time.

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

### 1.4 Constraint-Heavy Specifications

When you need code that meets specific technical requirements, make constraints explicit and prominent:

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

The key insight: LLMs tend to produce the *most common* solution to a problem. If you need a specific algorithm, name it explicitly and exclude the common alternatives.

---

## 2. Test-Driven Prompting

Test-driven prompting inverts the usual workflow: instead of describing what code should do in prose, you provide tests that the code must pass. This gives the LLM an unambiguous correctness criterion.

### 2.1 Tests-First Pattern

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

### 2.2 Property-Based Test Prompting

Beyond example-based tests, you can specify *properties* the code must satisfy:

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

### 2.3 Combining Specs with Tests

The most robust approach uses both a specification and tests:

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

## 3. Debugging and Error-Diagnosis Prompts

Debugging prompts help the LLM diagnose and fix errors in existing code. The quality of the diagnosis depends on how much context you provide.

### 3.1 The Diagnostic Prompt Template

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

### 3.2 Error Pattern Diagnosis

When you have a pattern of errors rather than a single bug, frame the prompt around the pattern:

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

### 3.3 Stack Trace Analysis Prompts

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

### 3.4 Structured Debugging Workflow

For complex bugs, guide the model through a structured analysis:

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

## 4. Code Review Prompts

Code review prompts ask the LLM to evaluate existing code for correctness, style, performance, and security issues.

### 4.1 Comprehensive Review Prompt

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

### 4.2 Focused Review Prompts

Sometimes you want the model to focus on a single dimension:

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

### 4.3 Diff Review Prompts

Review a code change rather than the full file:

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

## 5. Refactoring Prompts

Refactoring prompts ask the model to restructure code while preserving behavior. The key challenge is ensuring *behavioral equivalence* -- the refactored code must do exactly what the original did.

### 5.1 Pattern-Based Refactoring

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

### 5.2 Performance Refactoring

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

### 5.3 Modernization Refactoring

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

## 6. Multi-File Generation

Real projects span multiple files. Multi-file generation prompts must specify the project structure, inter-file dependencies, and how components integrate.

### 6.1 Project Scaffold Prompt

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

### 6.2 Component Integration Prompt

When generating a new component that must integrate with existing code:

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

## 7. Language-Specific Prompting Strategies

Different programming languages have different idioms, pitfalls, and conventions. Tailor your prompts accordingly.

### 7.1 Python-Specific Prompts

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

### 7.2 TypeScript-Specific Prompts

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

### 7.3 Rust-Specific Prompts

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

### 7.4 Prompting Across Languages

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

## 8. Documentation Generation

LLMs excel at generating documentation from code, but the quality depends on specifying the documentation format and audience.

### 8.1 API Documentation Prompt

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

### 8.2 README Generation

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

### 8.3 Changelog Generation from Git Diff

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

## 9. Code Explanation Prompts

Explanation prompts reverse the direction: instead of generating code, the model analyzes and explains existing code.

### 9.1 Multi-Level Explanation

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

### 9.2 Complexity Analysis Prompt

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

### 9.3 Architecture Explanation

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

## 10. Best Practices and Pitfalls

### 10.1 Best Practices Summary

| Practice | Why It Works |
|----------|-------------|
| Include function signatures | Anchors the model on exact types and naming |
| Provide test cases | Gives unambiguous correctness criteria |
| Specify libraries and versions | Prevents outdated API usage |
| Exclude approaches you do not want | LLMs default to the most common solution |
| Break complex tasks into steps | Reduces hallucination and improves coherence |
| Include error examples | Teaches the model about edge cases |
| Specify the coding style | Prevents inconsistency across generations |

### 10.2 Common Pitfalls

**Pitfall 1: Ambiguous specifications**

```python
# BAD: Vague prompt
"Write a function to process data"

# GOOD: Specific prompt
"Write a function process_csv(path: str) -> pd.DataFrame that reads a CSV file,
drops rows where all values are NaN, converts the 'date' column to datetime,
and returns the cleaned DataFrame. Raise FileNotFoundError if path doesn't exist."
```

**Pitfall 2: Not specifying language version and dependencies**

```python
# BAD: Model might use deprecated APIs
"Write an async HTTP client in Python"

# GOOD: Pinned versions
"Write an async HTTP client using Python 3.12 and aiohttp 3.9.
Use the modern async with syntax, not the deprecated callback API."
```

**Pitfall 3: Requesting too much at once**

```python
# BAD: Entire application in one prompt
"Write a complete e-commerce backend with authentication, products, orders,
payments, shipping, reviews, admin panel, and search."

# GOOD: One module at a time
"Write the order processing module. Here are the existing models it must
integrate with: [models]. Here are the API contracts it must implement: [specs]."
```

**Pitfall 4: Not validating generated code**

Always run generated code through at minimum:
1. Syntax check (compile)
2. Type check (mypy/pyright)
3. Linter (ruff/flake8)
4. Unit tests
5. Manual review of edge cases

**Pitfall 5: Trusting imports and dependencies**

LLMs sometimes import non-existent modules or use API functions that do not exist. Always verify that imported modules and called functions are real.

### 10.3 The Code Generation Prompt Checklist

Before sending a code generation prompt, verify:

- [ ] Function/class name and signature specified
- [ ] Input types and constraints defined
- [ ] Output format and type specified
- [ ] Error handling behavior described
- [ ] Edge cases mentioned
- [ ] Performance requirements stated (if relevant)
- [ ] Language version and dependencies pinned
- [ ] Coding style/conventions specified
- [ ] What NOT to do (negative constraints)
- [ ] Test cases or acceptance criteria included

---

## Exercises

### Exercise 1: Specification-Driven Generation

Write a specification prompt that generates a Python `PasswordValidator` class. The validator should check passwords against configurable rules (minimum length, require uppercase, require digits, require special characters, no common passwords). Include at least 6 specification elements from Section 1.1.

<details><summary>Show Answer</summary>

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

The key elements included: function signatures, behavior description, input constraints, output specification, error handling, performance requirements, and dependency constraints.

</details>

### Exercise 2: Test-Driven Prompting

Write a test-driven prompt for a `RingBuffer` class. Provide at least 5 test functions that fully specify the behavior, then ask the model to implement the class that passes all tests.

<details><summary>Show Answer</summary>

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

The tests cover: basic operations, overwrite behavior, indexing (positive and negative), empty state, full state tracking, clear method, and edge case (capacity=1).

</details>

### Exercise 3: Debugging Prompt

You have the following buggy code. Write a debugging prompt that provides sufficient context for the model to identify and fix the bug.

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

The bug: under concurrent access, multiple threads can sleep simultaneously and then all proceed at once, defeating the throttle.

<details><summary>Show Answer</summary>

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

The root cause: `time.sleep()` is called *inside* the lock, which blocks other threads from even checking. However, the `self.last_call = time.time()` is set *after* the sleep. If the sleep duration is very short (e.g., elapsed is almost equal to rate), multiple threads can read `self.last_call` as the same old value before any of them update it. The fix is to update `self.last_call` *before* sleeping (set it to the target time) and sleep *outside* the lock:

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

### Exercise 4: Code Review Prompt

Write a code review prompt for the following function. Your prompt should ask the model to check for at least three specific quality dimensions.

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

<details><summary>Show Answer</summary>

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

Expected findings:
- **CRITICAL (Security)**: SQL injection via f-string formatting -- use parameterized queries
- **HIGH (Performance)**: N+1 query pattern -- one SELECT per user; use batch UPSERT
- **HIGH (Correctness)**: No error handling -- if one user fails, commit is skipped for all
- **MEDIUM (Correctness)**: SELECT * fetches unnecessary data from source
- **MEDIUM (Resilience)**: No transaction on target_db -- partial updates on failure
- **LOW (Correctness)**: Names with apostrophes (O'Brien) will break the SQL

</details>

### Exercise 5: Multi-File Generation

Write a prompt that generates a Python package with three files: a data access layer, a business logic layer, and an API layer. The package should manage a simple inventory system (products with names, quantities, and prices). Ensure your prompt specifies the inter-file contracts clearly enough that all imports and method calls are consistent.

<details><summary>Show Answer</summary>

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

The key to this prompt is the explicit CONTRACT section that defines the exact method signatures and types shared between layers. This prevents the most common multi-file generation error: inconsistent interfaces between modules.

</details>

---

**Previous**: [Multimodal Prompting](./08_Multimodal_Prompting.md) | **Next**: [RAG Prompt Patterns](./10_RAG_Prompt_Patterns.md)
