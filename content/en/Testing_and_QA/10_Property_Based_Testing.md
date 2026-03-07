# Lesson 10: Property-Based Testing

**Previous**: [End-to-End Testing](./09_End_to_End_Testing.md) | **Next**: [Performance Testing](./11_Performance_Testing.md)

---

Traditional example-based tests verify that specific inputs produce specific outputs. But how do you know your chosen examples actually cover the edge cases that matter? Property-based testing flips the script: instead of specifying individual examples, you describe **properties** that should always hold, and the testing framework generates hundreds or thousands of random inputs to try to break them. When it finds a failure, it automatically shrinks the input to the smallest possible counterexample.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Comfortable writing pytest tests (Lessons 02–03)
- Understanding of test design principles (Lesson 05)
- Basic knowledge of Python type hints

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why property-based testing catches bugs that example-based tests miss
2. Use the Hypothesis library to write property-based tests with `@given` and strategies
3. Design effective properties for functions, data structures, and APIs
4. Apply stateful testing to verify systems with complex state transitions
5. Configure Hypothesis settings and integrate it into CI pipelines

---

## 1. The Limits of Example-Based Testing

Consider a function that sorts a list:

```python
def test_sort_basic():
    assert sort_list([3, 1, 2]) == [1, 2, 3]
    assert sort_list([]) == []
    assert sort_list([1]) == [1]
```

This covers three cases. But what about negative numbers? Duplicates? Very large lists? Lists with `None` values? You are limited by your imagination — and your imagination is biased toward cases that work.

Property-based testing asks a different question: **what properties must always be true about the output of `sort_list`, regardless of input?**

- The output has the same length as the input
- The output is in non-decreasing order
- The output contains exactly the same elements as the input

These properties hold for *every* valid input — and a property-based testing framework will try to find inputs where they do not.

---

## 2. Getting Started with Hypothesis

[Hypothesis](https://hypothesis.readthedocs.io/) is the standard property-based testing library for Python. Install it alongside pytest:

```bash
pip install hypothesis
```

### 2.1 Your First Property Test

```python
from hypothesis import given
from hypothesis import strategies as st


def sort_list(xs):
    return sorted(xs)


@given(st.lists(st.integers()))
def test_sort_preserves_length(xs):
    result = sort_list(xs)
    assert len(result) == len(xs)


@given(st.lists(st.integers()))
def test_sort_produces_ordered_output(xs):
    result = sort_list(xs)
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]


@given(st.lists(st.integers()))
def test_sort_preserves_elements(xs):
    result = sort_list(xs)
    assert sorted(result) == sorted(xs)  # Same multiset
```

When you run `pytest`, Hypothesis will generate 100 random lists of integers by default and check each property. If any fails, it shrinks the input to the minimal failing example.

### 2.2 How Hypothesis Works Internally

Hypothesis does not just generate random data. Its process is:

1. **Generation**: Create random examples using the specified strategy
2. **Testing**: Run the test function with each example
3. **Shrinking**: When a failure is found, systematically reduce the input to find the smallest counterexample
4. **Database**: Store failing examples in `.hypothesis/` so they are replayed in future runs

The shrinking step is crucial. If your function fails on a list of 47 random integers, the raw failure is hard to debug. Hypothesis will shrink it to something like `[0, 1]` or `[-1]` — the simplest input that still triggers the bug.

---

## 3. Strategies in Depth

Strategies are Hypothesis's way of describing the shape of test data. The `hypothesis.strategies` module (conventionally imported as `st`) provides composable building blocks.

### 3.1 Primitive Strategies

```python
from hypothesis import strategies as st

# Integers with optional bounds
st.integers()                        # Any integer
st.integers(min_value=0, max_value=100)  # Bounded

# Floating point
st.floats()                          # Includes NaN, inf, -inf
st.floats(allow_nan=False, allow_infinity=False)

# Text
st.text()                            # Any Unicode string
st.text(min_size=1, max_size=50)     # Bounded length
st.text(alphabet="abcdef")           # Restricted alphabet

# Booleans
st.booleans()                        # True or False

# Binary
st.binary(min_size=0, max_size=100)  # bytes objects

# None
st.none()                            # Always None
```

### 3.2 Collection Strategies

```python
# Lists
st.lists(st.integers())                          # List[int]
st.lists(st.integers(), min_size=1, max_size=10) # Non-empty, bounded

# Sets (no duplicates)
st.sets(st.integers(), min_size=1)

# Dictionaries
st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.integers()
)

# Tuples (fixed structure)
st.tuples(st.integers(), st.text(), st.booleans())  # (int, str, bool)

# Frozensets
st.frozensets(st.integers())
```

### 3.3 Composing Strategies

The real power of strategies comes from composition:

```python
# Union: one of several types
st.one_of(st.integers(), st.text(), st.none())

# Mapping: transform generated values
st.integers(min_value=1, max_value=12).map(lambda m: f"2024-{m:02d}-01")

# Filtering: reject values (use sparingly — can slow generation)
st.integers().filter(lambda x: x % 2 == 0)  # Even integers

# Flatmap: dependent generation
def list_and_index(draw):
    """Generate a non-empty list and a valid index into it."""
    xs = draw(st.lists(st.integers(), min_size=1))
    i = draw(st.integers(min_value=0, max_value=len(xs) - 1))
    return (xs, i)

# Using @st.composite for complex strategies
@st.composite
def valid_email(draw):
    local = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=1, max_size=20
    ))
    domain = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=1, max_size=10
    ))
    tld = draw(st.sampled_from(["com", "org", "net", "io"]))
    return f"{local}@{domain}.{tld}"
```

### 3.4 The `@st.composite` Decorator

`@st.composite` is the most flexible way to build custom strategies. The decorated function receives a `draw` callable that pulls values from other strategies:

```python
@st.composite
def sorted_pair(draw):
    """Generate two integers where a <= b."""
    a = draw(st.integers())
    b = draw(st.integers(min_value=a))
    return (a, b)


@given(sorted_pair())
def test_range_is_valid(pair):
    a, b = pair
    assert a <= b
```

---

## 4. Providing Explicit Examples with `@example`

Sometimes you want to ensure specific edge cases are always tested, in addition to generated ones:

```python
from hypothesis import given, example
from hypothesis import strategies as st


@given(st.integers(), st.integers())
@example(0, 0)          # Always test zero/zero
@example(1, -1)         # Always test opposite signs
@example(2**31, 2**31)  # Always test large values
def test_addition_is_commutative(a, b):
    assert a + b == b + a
```

`@example` runs the given input *every* time, before the random generation phase. This is useful for:
- Regression tests for previously found bugs
- Known edge cases (empty strings, zero, boundary values)
- Cases that random generation might take long to discover

---

## 5. Configuring Hypothesis with `settings`

The `settings` decorator controls generation behavior:

```python
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


@settings(
    max_examples=500,          # Default is 100; increase for thorough testing
    deadline=1000,             # Max ms per example (None to disable)
    suppress_health_check=[
        HealthCheck.too_slow,  # Allow slow tests
    ],
    database=None,             # Disable example database
)
@given(st.lists(st.integers()))
def test_with_custom_settings(xs):
    result = sorted(xs)
    assert len(result) == len(xs)
```

### 5.1 Profiles for Different Environments

```python
from hypothesis import settings, Phase

# CI profile: more thorough
settings.register_profile("ci", max_examples=1000)

# Development profile: fast feedback
settings.register_profile("dev", max_examples=50)

# Debug profile: minimal, with verbose output
settings.register_profile("debug", max_examples=10, verbosity=3)

# Load profile from environment variable
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

Set the profile in CI:

```bash
HYPOTHESIS_PROFILE=ci pytest tests/
```

---

## 6. Stateful Testing

Property tests on pure functions are straightforward. But what about systems with mutable state — databases, caches, APIs? Hypothesis provides **stateful testing** using rule-based state machines.

### 6.1 Rule-Based State Machines

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant
from hypothesis import strategies as st


class SetMachine(RuleBasedStateMachine):
    """Test that our CustomSet behaves like Python's built-in set."""

    def __init__(self):
        super().__init__()
        self.model = set()           # Reference (known-correct)
        self.actual = CustomSet()    # Implementation under test

    @rule(value=st.integers())
    def add_element(self, value):
        self.model.add(value)
        self.actual.add(value)

    @rule(value=st.integers())
    def remove_element(self, value):
        try:
            self.model.remove(value)
            self.actual.remove(value)
        except KeyError:
            pass  # Both should raise or neither

    @rule(value=st.integers())
    def check_contains(self, value):
        assert (value in self.model) == (value in self.actual)

    @invariant()
    def size_matches(self):
        assert len(self.model) == len(self.actual)


# Hypothesis will generate random sequences of add/remove/check operations
TestSetMachine = SetMachine.TestCase
```

### 6.2 Why Stateful Testing Matters

Stateful testing is particularly powerful for:
- **Data structures**: Verify your implementation matches a reference
- **APIs**: Verify that any sequence of API calls leaves the system in a valid state
- **Protocols**: Verify that state machines handle all valid transitions
- **Caches**: Verify that cached results always match fresh computations

Hypothesis generates sequences of operations and, when a failure is found, shrinks both the operations and their arguments to the minimal failing sequence.

---

## 7. When Property-Based Testing Shines

Property-based testing is not a replacement for example-based testing. It excels in specific situations:

### 7.1 Roundtrip Properties

If you can encode and decode, the roundtrip should be an identity:

```python
import json

@given(st.dictionaries(st.text(), st.integers()))
def test_json_roundtrip(data):
    assert json.loads(json.dumps(data)) == data
```

### 7.2 Invariant Properties

Operations that should preserve certain characteristics:

```python
@given(st.lists(st.integers()))
def test_reverse_preserves_length(xs):
    assert len(list(reversed(xs))) == len(xs)

@given(st.lists(st.integers()))
def test_reverse_is_involution(xs):
    assert list(reversed(list(reversed(xs)))) == xs
```

### 7.3 Oracle Testing

Compare your implementation against a known-correct (but slower) reference:

```python
@given(st.lists(st.integers()))
def test_my_sort_matches_builtin(xs):
    assert my_sort(xs) == sorted(xs)
```

### 7.4 Algebraic Properties

Mathematical properties your code should satisfy:

```python
@given(st.integers(), st.integers(), st.integers())
def test_addition_is_associative(a, b, c):
    assert (a + b) + c == a + (b + c)
```

### 7.5 Idempotence

Applying an operation twice should give the same result as once:

```python
@given(st.text())
def test_normalize_is_idempotent(s):
    assert normalize(normalize(s)) == normalize(s)
```

---

## 8. Practical Example: Testing a URL Parser

Let us build a more realistic example — property-testing a URL parser:

```python
from urllib.parse import urlparse, urlunparse
from hypothesis import given
from hypothesis import strategies as st


@st.composite
def urls(draw):
    scheme = draw(st.sampled_from(["http", "https"]))
    host = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
        min_size=1, max_size=20
    ))
    tld = draw(st.sampled_from(["com", "org", "net"]))
    path = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz/",
        min_size=0, max_size=30
    ))
    if path and not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{host}.{tld}{path}"


@given(urls())
def test_parse_roundtrip(url):
    """Parsing and reconstructing a URL should produce the original."""
    parsed = urlparse(url)
    reconstructed = urlunparse(parsed)
    assert reconstructed == url


@given(urls())
def test_scheme_is_preserved(url):
    parsed = urlparse(url)
    assert parsed.scheme in ("http", "https")


@given(urls())
def test_netloc_is_non_empty(url):
    parsed = urlparse(url)
    assert len(parsed.netloc) > 0
```

---

## 9. Common Pitfalls and Best Practices

### 9.1 Avoid Overly Strict Filters

```python
# BAD: Hypothesis may give up if too many values are rejected
@given(st.integers().filter(lambda x: x % 1000 == 42))
def test_bad_filter(x):
    ...

# GOOD: Generate directly
@given(st.integers().map(lambda x: x * 1000 + 42))
def test_good_generation(x):
    ...
```

### 9.2 Keep Properties Simple and Independent

Each test should verify one property. A test that checks "sorted, same length, and same elements" all at once is harder to debug than three separate tests.

### 9.3 Use `@example` for Regression Tests

When Hypothesis finds a bug, add `@example(failing_input)` to lock it in permanently:

```python
@given(st.text())
@example("")           # Found this edge case in CI
@example("\x00\xff")   # Null and high bytes
def test_my_encoder(s):
    assert decode(encode(s)) == s
```

### 9.4 Set Reasonable Deadlines

If your function is inherently slow, adjust the deadline rather than letting tests time out:

```python
@settings(deadline=5000)  # 5 seconds per example
@given(st.lists(st.integers(), max_size=10000))
def test_slow_function(xs):
    ...
```

### 9.5 Use `assume()` for Preconditions

When inputs must satisfy a precondition that is hard to express as a strategy:

```python
from hypothesis import assume

@given(st.floats(), st.floats())
def test_division(a, b):
    assume(b != 0)
    assume(not (a == 0 and b == 0))
    result = a / b
    assert result * b == pytest.approx(a, rel=1e-6)
```

---

## 10. Integrating Hypothesis into Your Workflow

### 10.1 Project Configuration

Add to `pyproject.toml`:

```toml
[tool.hypothesis]
database_backend = "directory"

[tool.pytest.ini_options]
addopts = "--hypothesis-show-statistics"
```

### 10.2 CI Configuration

```yaml
# .github/workflows/test.yml
- name: Run property tests
  env:
    HYPOTHESIS_PROFILE: ci
  run: pytest tests/ -x --hypothesis-show-statistics
```

### 10.3 When to Write Property Tests vs Example Tests

| Situation | Preferred Approach |
|---|---|
| Business logic with specific rules | Example-based |
| Serialization/deserialization | Property-based (roundtrip) |
| Data transformations | Property-based (invariants) |
| Algorithm correctness | Property-based (oracle) |
| UI behavior | Example-based |
| Edge case regression | `@example` on property tests |
| Mathematical operations | Property-based (algebraic) |

---

## Exercises

1. **Roundtrip Testing**: Write a property test for a `compress`/`decompress` function pair that verifies the roundtrip property for arbitrary byte strings.

2. **Invariant Discovery**: Given a function `deduplicate(xs: list) -> list` that removes duplicates while preserving order, identify and test at least three properties it must satisfy.

3. **Custom Strategy**: Build a `@st.composite` strategy that generates valid JSON-like nested structures (dicts containing lists containing dicts, etc.) up to a configurable depth. Write a property test using it.

4. **Stateful Testing**: Implement a `RuleBasedStateMachine` that tests a simple key-value store (dict wrapper with `get`, `set`, `delete`) against Python's built-in `dict`.

5. **Bug Hunting**: Intentionally introduce a subtle bug into a sorting function (e.g., off-by-one in a merge step) and demonstrate that Hypothesis finds and shrinks the counterexample. Compare this to how many example-based tests you would need to catch the same bug.

---

**License**: CC BY-NC 4.0
