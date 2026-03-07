# Test-Driven Development

**Previous**: [Test Coverage and Quality](./05_Test_Coverage_and_Quality.md) | **Next**: [Integration Testing](./07_Integration_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the Red-Green-Refactor cycle to develop code incrementally
2. Use baby steps and triangulation to build implementations gradually
3. Apply the Transformation Priority Premise to guide refactoring decisions
4. Evaluate TDD's benefits and limitations in different contexts
5. Perform a complete TDD walkthrough building a real module

---

## What Is TDD?

Test-Driven Development is a software development practice where you write a failing test *before* writing the production code that makes it pass. Invented by Kent Beck as part of Extreme Programming, TDD inverts the traditional "write code, then test" workflow.

The fundamental cycle has three steps:

```
    ┌──────────┐
    │   RED    │   Write a test that fails
    └────┬─────┘
         │
    ┌────▼─────┐
    │  GREEN   │   Write the simplest code to pass the test
    └────┬─────┘
         │
    ┌────▼─────┐
    │ REFACTOR │   Improve the code without changing behavior
    └────┬─────┘
         │
         └──────── Repeat
```

### The Three Laws of TDD (Robert C. Martin)

1. You may not write production code until you have written a failing unit test
2. You may not write more of a unit test than is sufficient to fail (including compilation failure)
3. You may not write more production code than is sufficient to pass the currently failing test

---

## Red-Green-Refactor in Practice

### RED: Write a Failing Test

Write the smallest test that describes one piece of behavior. Run it and confirm it fails. The failure message should be clear and specific.

```python
# test_stack.py
from stack import Stack

def test_new_stack_is_empty():
    s = Stack()
    assert s.is_empty() is True
```

```bash
$ pytest test_stack.py
E   ModuleNotFoundError: No module named 'stack'
```

The test fails because `Stack` does not exist yet. This is the RED phase.

### GREEN: Make It Pass

Write the minimum amount of code to make the test pass. Do not anticipate future requirements. Do not optimize. Just make it green.

```python
# stack.py
class Stack:
    def is_empty(self):
        return True
```

```bash
$ pytest test_stack.py
PASSED
```

Yes, `return True` is cheating. That is deliberate. The current tests only require `is_empty()` to return `True`. More tests will force a real implementation.

### REFACTOR: Clean Up

Look at both the test code and production code. Is there duplication? Poor naming? Unnecessary complexity? Clean it up while keeping all tests green.

In this case, the code is simple enough — no refactoring needed yet.

---

## Baby Steps and Triangulation

### Baby Steps

Each cycle should be tiny — a few minutes at most. If you spend 20 minutes in the RED phase writing a complex test, you are taking too big a step.

Break large features into the smallest possible increments:

```
Bad:  "Test that the calculator handles all arithmetic operations"
Good: "Test that add(2, 3) returns 5"
      "Test that add(-1, 1) returns 0"
      "Test that subtract(5, 3) returns 2"
```

### Triangulation

When you are not sure how to generalize, add more specific test cases until the pattern forces a general implementation.

```python
# Test 1: forces add() to exist and return something
def test_add_two_and_three():
    assert add(2, 3) == 5

# stack.py — the "cheat" implementation
def add(a, b):
    return 5  # Passes, but obviously fake
```

```python
# Test 2: triangulates — forces a real implementation
def test_add_one_and_one():
    assert add(1, 1) == 2

# Now the fake implementation fails. We must generalize:
def add(a, b):
    return a + b
```

Two test cases "triangulated" the real implementation. One test case allowed cheating; two forced the truth.

---

## Transformation Priority Premise

Kent Beck and Robert Martin observed that TDD implementations evolve through predictable transformations, from simple to complex. Prefer simpler transformations:

| Priority | Transformation | Example |
|----------|---------------|---------|
| 1 | `{}` -> nil | No code -> return None |
| 2 | nil -> constant | Return None -> return 42 |
| 3 | constant -> variable | return 42 -> return x |
| 4 | unconditional -> conditional | statement -> if/else |
| 5 | scalar -> collection | single value -> list |
| 6 | statement -> recursion | loop -> recursive call |
| 7 | value -> mutated value | immutable -> mutable state |

When choosing how to make a test pass, prefer the transformation that is *higher* (simpler) on this list. This leads to cleaner, more incremental designs.

---

## TDD Walkthrough: Building a RomanNumeral Converter

Let us build a function that converts integers to Roman numerals using strict TDD.

### Cycle 1: The Simplest Case

**RED:**

```python
# test_roman.py
from roman import to_roman

def test_1_is_I():
    assert to_roman(1) == "I"
```

**GREEN:**

```python
# roman.py
def to_roman(number: int) -> str:
    return "I"
```

### Cycle 2: Triangulate

**RED:**

```python
def test_2_is_II():
    assert to_roman(2) == "II"
```

**GREEN:**

```python
def to_roman(number: int) -> str:
    return "I" * number
```

This passes both tests. The pattern `"I" * number` generalizes from the two test cases.

### Cycle 3: Introduce a New Symbol

**RED:**

```python
def test_5_is_V():
    assert to_roman(5) == "V"
```

`"I" * 5 = "IIIII"` which is wrong. We need a conditional.

**GREEN:**

```python
def to_roman(number: int) -> str:
    if number == 5:
        return "V"
    return "I" * number
```

### Cycle 4: Combination

**RED:**

```python
def test_4_is_IV():
    assert to_roman(4) == "IV"

def test_6_is_VI():
    assert to_roman(6) == "VI"
```

Time to refactor toward a general algorithm.

**GREEN + REFACTOR:**

```python
def to_roman(number: int) -> str:
    values = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    result = ""
    for value, numeral in values:
        while number >= value:
            result += numeral
            number -= value
    return result
```

### Cycle 5: Comprehensive Tests

Now we add more test cases to build confidence:

```python
import pytest

@pytest.mark.parametrize("number, expected", [
    (1, "I"),
    (3, "III"),
    (4, "IV"),
    (5, "V"),
    (9, "IX"),
    (10, "X"),
    (14, "XIV"),
    (40, "XL"),
    (42, "XLII"),
    (90, "XC"),
    (99, "XCIX"),
    (100, "C"),
    (400, "CD"),
    (500, "D"),
    (900, "CM"),
    (1000, "M"),
    (1994, "MCMXCIV"),
    (3999, "MMMCMXCIX"),
])
def test_to_roman(number, expected):
    assert to_roman(number) == expected
```

All pass. Notice how TDD guided us from a fake `return "I"` to a complete algorithm through small, verified steps.

---

## TDD Walkthrough: Building a Stack

A more substantial example showing TDD with a data structure.

```python
# test_stack.py — tests written in TDD order
import pytest
from stack import Stack


def test_new_stack_is_empty():
    s = Stack()
    assert s.is_empty() is True


def test_stack_after_push_is_not_empty():
    s = Stack()
    s.push(42)
    assert s.is_empty() is False


def test_pop_returns_pushed_value():
    s = Stack()
    s.push(42)
    assert s.pop() == 42


def test_pop_removes_element():
    s = Stack()
    s.push(42)
    s.pop()
    assert s.is_empty() is True


def test_lifo_order():
    s = Stack()
    s.push("first")
    s.push("second")
    assert s.pop() == "second"
    assert s.pop() == "first"


def test_peek_returns_top_without_removing():
    s = Stack()
    s.push(99)
    assert s.peek() == 99
    assert s.is_empty() is False


def test_pop_empty_stack_raises():
    s = Stack()
    with pytest.raises(IndexError, match="empty"):
        s.pop()


def test_peek_empty_stack_raises():
    s = Stack()
    with pytest.raises(IndexError, match="empty"):
        s.peek()


def test_size():
    s = Stack()
    assert s.size() == 0
    s.push("a")
    s.push("b")
    assert s.size() == 2
    s.pop()
    assert s.size() == 1
```

```python
# stack.py — implementation driven by the tests above
class Stack:
    def __init__(self):
        self._items: list = []

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Cannot peek at empty stack")
        return self._items[-1]

    def size(self) -> int:
        return len(self._items)
```

---

## Benefits of TDD

### 1. Design Feedback

TDD forces you to think about the *interface* before the implementation. If a function is hard to test, it is usually hard to use. TDD naturally pushes toward:

- Smaller functions with clear responsibilities
- Dependency injection instead of hidden dependencies
- Pure functions where possible

### 2. Built-in Regression Suite

Every feature has tests from birth. When you refactor or add features later, existing tests catch regressions immediately.

### 3. Confidence to Refactor

Without tests, refactoring is terrifying. With TDD, you have a safety net. You can aggressively simplify code knowing that tests will catch mistakes.

### 4. Documentation by Example

Tests written in TDD style describe what the code does in concrete, executable examples. They serve as living documentation that never goes stale.

---

## Criticisms and Limitations

### 1. Overhead for Exploratory Work

When you do not know what you are building (prototyping, research), writing tests first is premature. You may throw away both the tests and the code. TDD works best when requirements are reasonably clear.

### 2. Over-Specification

Poorly practiced TDD can lock you into an implementation. If tests are too coupled to internal structure, refactoring becomes impossible without rewriting tests.

### 3. False Confidence

TDD does not guarantee correctness. You can have 100% coverage and still miss edge cases. TDD tests reflect the developer's understanding, which may be incomplete.

### 4. Learning Curve

TDD requires discipline and practice. Beginners often write tests that are too large or too coupled to implementation details. It takes time to develop the instinct for good test granularity.

### When to Use TDD

| Situation | TDD? |
|-----------|------|
| Well-defined business logic | Yes |
| Algorithm implementation | Yes |
| Library/API design | Yes |
| UI layout and styling | Usually no |
| Exploratory prototyping | No |
| One-off scripts | No |
| Debugging production issues | Yes (write a failing test for the bug first) |

---

## TDD and the Wider Process

TDD is one practice in a broader quality toolkit:

```
Developer writes failing test (TDD)
  → Implements code to pass
  → Refactors
  → Pushes to CI
    → CI runs full test suite
    → CI runs linter and type checker
    → CI runs coverage check
    → Code review
    → Merge
```

TDD is not a substitute for integration tests, code review, or monitoring. It is a development discipline that produces well-tested, well-designed units of code.

---

## Exercises

1. **TDD a FizzBuzz**: Using strict TDD (write one test, make it pass, refactor), implement FizzBuzz. Document each Red-Green-Refactor cycle. Show the test and implementation at each step.

2. **TDD a StringCalculator**: Build a `string_calculator(expression: str) -> int` that handles:
   - Empty string returns 0
   - Single number returns that number
   - Two numbers separated by comma returns their sum
   - Newline as delimiter (`"1\n2,3"` returns 6)
   - Negative numbers raise ValueError
   Apply TDD strictly — one test case at a time.

3. **TDD critique**: Take existing code you have written (without TDD). Attempt to add tests after the fact. Identify at least 2 design decisions that make testing harder. Explain how TDD might have led to a different design.

---

**License**: CC BY-NC 4.0
