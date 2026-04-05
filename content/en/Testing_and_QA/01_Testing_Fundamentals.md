# Testing Fundamentals

**Previous**: [Overview](./00_Overview.md) | **Next**: [Unit Testing with pytest](./02_Unit_Testing_with_pytest.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between unit, integration, end-to-end, and acceptance tests
2. Apply the test pyramid to balance test suite composition
3. Evaluate the economic trade-offs of testing effort
4. Identify and avoid common testing anti-patterns
5. Assess test quality beyond simple pass/fail metrics

---

## Why Testing Matters

Software testing is not about proving code works. It is about finding where it does not. Every bug that reaches production carries a cost: lost revenue, damaged reputation, engineering time diverted from new features, and sometimes human safety. Testing is the systematic practice of reducing that risk.

A common misconception is that testing slows development down. In reality, untested code slows you down *later*. Debugging production incidents, manually verifying changes, and fear of refactoring are far more expensive than writing tests upfront.

```python
# Without tests: "I think this works"
def calculate_discount(price, percentage):
    return price * percentage / 100

# With tests: "I know this works — and I know when it breaks"
def test_calculate_discount():
    assert calculate_discount(100, 10) == 10.0
    assert calculate_discount(200, 50) == 100.0
    assert calculate_discount(0, 25) == 0.0
```

---

## Types of Tests

### Unit Tests

Unit tests verify a single function, method, or class in isolation. They are fast, deterministic, and form the foundation of a reliable test suite.

**Characteristics:**
- Execute in milliseconds
- No I/O (no database, no network, no filesystem)
- Test one behavior per test function
- Easy to pinpoint failures

```python
# Function under test
def validate_email(email: str) -> bool:
    """Return True if email has a valid format."""
    import re
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# Unit tests
def test_valid_email():
    assert validate_email("user@example.com") is True

def test_email_without_at_sign():
    assert validate_email("userexample.com") is False

def test_email_without_domain():
    assert validate_email("user@") is False

def test_email_with_subdomain():
    assert validate_email("user@mail.example.com") is True
```

### Integration Tests

Integration tests verify that multiple components work together correctly. They cross module boundaries and often involve real external systems (databases, file systems, APIs).

**Characteristics:**
- Slower than unit tests (seconds to minutes)
- May require setup/teardown of external resources
- Test the *seams* between components
- Catch wiring bugs that unit tests miss

```python
import sqlite3

def test_user_repository_stores_and_retrieves():
    """Integration test: repository + database working together."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

    # Store
    conn.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    conn.commit()

    # Retrieve
    row = conn.execute("SELECT name FROM users WHERE id = 1").fetchone()
    assert row[0] == "Alice"
    conn.close()
```

### End-to-End (E2E) Tests

E2E tests exercise the entire application from the user's perspective. For a web application, this means launching a browser, navigating pages, clicking buttons, and verifying outcomes.

**Characteristics:**
- Slowest to run (seconds to minutes per test)
- Most realistic — simulate actual user workflows
- Most brittle — break when UI changes
- Require a running application stack

```python
# Conceptual E2E test (Playwright example)
def test_user_can_log_in(page):
    page.goto("http://localhost:8000/login")
    page.fill("#username", "alice")
    page.fill("#password", "secret123")
    page.click("button[type=submit]")
    assert page.text_content("h1") == "Welcome, Alice"
```

### Acceptance Tests

Acceptance tests verify that the software meets business requirements. They are often written in collaboration with stakeholders and describe *what* the system should do, not *how* it does it.

```python
# Acceptance test: business language, not implementation details
def test_premium_users_get_free_shipping():
    """Business rule: orders over $50 from premium users ship free."""
    user = create_user(membership="premium")
    order = create_order(user=user, total=75.00)
    assert order.shipping_cost == 0.0

def test_regular_users_pay_shipping():
    """Business rule: regular users always pay shipping."""
    user = create_user(membership="regular")
    order = create_order(user=user, total=75.00)
    assert order.shipping_cost > 0.0
```

---

## The Test Pyramid

The test pyramid is a model for balancing test types, proposed by Mike Cohn. It recommends many unit tests at the base, fewer integration tests in the middle, and even fewer E2E tests at the top.

```
        /  E2E  \          Slow, expensive, realistic
       /----------\
      / Integration \      Medium speed, cross-boundary
     /----------------\
    /    Unit Tests     \  Fast, cheap, isolated
   /____________________\
```

**Why this shape?**

| Property         | Unit   | Integration | E2E     |
|-----------------|--------|-------------|---------|
| Speed           | ~1 ms  | ~100 ms     | ~5 sec  |
| Cost to write   | Low    | Medium      | High    |
| Cost to maintain| Low    | Medium      | High    |
| Failure clarity | High   | Medium      | Low     |
| Confidence      | Low    | Medium      | High    |

The pyramid is a guideline, not a law. Some applications (e.g., CRUD apps with thin logic) may benefit from more integration tests. The key insight is: **push tests down to the lowest level that provides sufficient confidence**.

### The Ice Cream Cone Anti-Pattern

The inverse of the test pyramid. Teams with mostly manual or E2E tests and few unit tests experience:

- Slow feedback loops (CI takes hours)
- Flaky test suites that erode trust
- Expensive maintenance when UI changes
- Developers who avoid running tests locally

---

## Testing Economics

### The Cost of Bugs Over Time

The later a bug is found, the more expensive it is to fix:

| Stage              | Relative Cost |
|-------------------|---------------|
| During coding      | 1x            |
| Code review        | 2-5x          |
| QA / Testing       | 5-15x         |
| Production         | 30-100x       |

This is why shifting testing *left* (earlier in the development process) pays dividends.

### What to Test

Not everything needs the same level of testing. Prioritize:

1. **Business-critical paths** — Payment processing, authentication, data integrity
2. **Complex logic** — Algorithms, state machines, calculations
3. **Edge cases** — Empty inputs, boundary values, error conditions
4. **Bug-prone areas** — Code that has broken before will break again
5. **Public APIs** — Interfaces that other code or users depend on

### What NOT to Over-Test

- Trivial getters/setters with no logic
- Framework-provided functionality (do not test that Flask returns 200)
- Implementation details that change frequently
- Third-party library internals

---

## Test Quality vs Quantity

Having 1,000 tests means nothing if they all test the same happy path. Quality dimensions include:

### Behavior Coverage

Does your test suite cover the important *behaviors*, not just lines of code?

```python
# Poor: tests the happy path only
def test_divide():
    assert divide(10, 2) == 5.0

# Better: tests behavior including edge cases
def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_divide_negative_numbers():
    assert divide(-10, 2) == -5.0

def test_divide_returns_float():
    assert isinstance(divide(7, 2), float)
```

### Test Independence

Each test should be able to run in isolation, in any order. Tests that depend on execution order are a maintenance nightmare.

```python
# BAD: test_b depends on test_a having run first
class TestBadDependency:
    result = None

    def test_a_create(self):
        TestBadDependency.result = create_item("widget")
        assert self.result is not None

    def test_b_read(self):
        # Fails if test_a did not run first!
        item = get_item(TestBadDependency.result.id)
        assert item.name == "widget"

# GOOD: each test sets up its own data
def test_create_item():
    item = create_item("widget")
    assert item is not None
    assert item.name == "widget"

def test_read_item():
    item = create_item("widget")  # Own setup
    retrieved = get_item(item.id)
    assert retrieved.name == "widget"
```

### Determinism

Flaky tests — tests that sometimes pass and sometimes fail without code changes — destroy confidence. Common causes:

- Time-dependent logic (`datetime.now()`)
- Random data without seeding
- Shared mutable state between tests
- Network calls to external services
- Race conditions in async code

---

## Testing Anti-Patterns

### 1. The Giant Test

A single test function that verifies 20 things. When it fails, you have no idea which behavior broke.

### 2. Testing Implementation, Not Behavior

```python
# Anti-pattern: testing HOW, not WHAT
def test_sort_uses_quicksort():
    with patch("mymodule.quicksort") as mock_qs:
        sort_data([3, 1, 2])
        mock_qs.assert_called_once()

# Better: testing WHAT the function achieves
def test_sort_returns_sorted_list():
    assert sort_data([3, 1, 2]) == [1, 2, 3]
```

### 3. The Liar Test

A test that always passes regardless of correctness, usually because the assertion is wrong or missing.

```python
# Anti-pattern: no real assertion
def test_process_data():
    result = process_data([1, 2, 3])
    assert result is not None  # This tells you almost nothing
```

### 4. Excessive Mocking

When you mock so much that you are testing the mocks, not the code. If a test requires more than 2-3 mocks, consider whether the code under test has too many dependencies.

### 5. Copy-Paste Tests

Dozens of near-identical tests differing in one value. Use parameterized tests instead (covered in Lesson 03).

### 6. Ignoring Test Maintenance

Tests are code. They need refactoring, documentation, and review just like production code. Neglected test suites become liabilities.

---

## When to Write Tests

| Approach                | When It Fits                                |
|------------------------|---------------------------------------------|
| Test-First (TDD)       | Well-understood requirements, algorithmic code |
| Test-After             | Exploratory/prototype code, UI experiments   |
| Test-During            | Most production code — write tests alongside |
| Regression Test        | After a bug is found — prevent recurrence    |

The best time to write a test is when you can clearly articulate the expected behavior. The worst time to *not* write a test is right before a deploy.

---

## Setting Up a Test Project

A minimal Python project with testing:

```
my_project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       └── calculator.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_calculator.py
├── pyproject.toml
└── README.md
```

```toml
# pyproject.toml
[project]
name = "my-project"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

```python
# src/my_project/calculator.py
def add(a: float, b: float) -> float:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
```

```python
# tests/test_calculator.py
import pytest
from my_project.calculator import add, divide

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -1) == -2

def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        divide(10, 0)
```

Run with:

```bash
pip install pytest
pytest
```

---

## Exercises

1. **Classify the tests**: Given a list of test descriptions, classify each as unit, integration, E2E, or acceptance test. Justify your reasoning for each.

2. **Find the anti-pattern**: Review the following test suite and identify at least three anti-patterns. Rewrite the tests to fix them.

3. **Design a test plan**: For a simple library management system (add book, borrow book, return book, search by title), outline which behaviors you would test at each level of the test pyramid. Explain why you placed each test where you did.

---

**License**: CC BY-NC 4.0
