# Test Coverage and Quality

**Previous**: [Mocking and Patching](./04_Mocking_and_Patching.md) | **Next**: [Test-Driven Development](./06_Test_Driven_Development.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Measure line and branch coverage using `coverage.py`
2. Configure coverage thresholds and exclusions with `.coveragerc`
3. Interpret coverage reports and identify meaningful gaps
4. Apply mutation testing with `mutmut` to assess test effectiveness
5. Distinguish between high coverage and high-quality tests

---

## What Is Test Coverage?

Test coverage measures what percentage of your code is executed during testing. It answers: "Which lines (or branches) did my tests actually run?"

Coverage is a **necessary but not sufficient** indicator of test quality. Code that is executed is not necessarily code that is *verified*. A test can run a function without checking its output.

```python
# This test "covers" the function but verifies nothing meaningful
def test_misleading_coverage():
    result = complex_calculation(42)
    assert True  # 100% coverage, 0% confidence
```

---

## Setting Up coverage.py

`coverage.py` is the standard Python coverage tool. pytest integrates with it through the `pytest-cov` plugin.

```bash
pip install coverage pytest-cov
```

### Running Coverage

```bash
# Using coverage directly
coverage run -m pytest
coverage report
coverage html  # Generate HTML report in htmlcov/

# Using pytest-cov (simpler)
pytest --cov=mypackage
pytest --cov=mypackage --cov-report=html
pytest --cov=mypackage --cov-report=term-missing
```

### Reading the Report

```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
mypackage/__init__.py       2      0   100%
mypackage/calculator.py    25      3    88%   42-44
mypackage/validator.py     40     12    70%   15-18, 33-40, 55
-----------------------------------------------------
TOTAL                      67     15    78%
```

- **Stmts**: Total executable statements
- **Miss**: Statements never executed during tests
- **Cover**: Percentage of statements executed
- **Missing**: Line numbers not covered

---

## Line Coverage vs Branch Coverage

### Line Coverage

Line coverage counts which *lines* were executed. It can miss untested conditional branches.

```python
def categorize_age(age: int) -> str:
    if age < 0:
        return "invalid"
    elif age < 18:
        return "minor"
    elif age < 65:
        return "adult"
    else:
        return "senior"


# This test covers only 2 of 4 branches
def test_adult():
    assert categorize_age(30) == "adult"

def test_minor():
    assert categorize_age(10) == "minor"

# Line coverage might show 70%+, but "invalid" and "senior" paths are untested
```

### Branch Coverage

Branch coverage tracks both outcomes of each conditional (True and False). Enable it:

```bash
pytest --cov=mypackage --cov-branch
```

Or in configuration:

```ini
# .coveragerc or pyproject.toml
[tool.coverage.run]
branch = true
```

Branch coverage reveals that the test above misses the `age < 0` and `age >= 65` branches.

```
Name                 Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------
mypackage/age.py        8      2      6      2    67%   3->4, 9->10
```

- **Branch**: Total branch points
- **BrPart**: Partially covered branches (some outcomes tested, not all)

---

## Configuring coverage.py

### pyproject.toml Configuration

```toml
[tool.coverage.run]
source = ["src/mypackage"]
branch = true
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/__main__.py",
]

[tool.coverage.report]
# Fail if coverage drops below this threshold
fail_under = 85

# Lines matching these patterns are excluded from coverage
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.",
    "raise NotImplementedError",
    "pass",
    "\\.\\.\\.",
]

# Show missing lines in terminal report
show_missing = true

# Precision of coverage percentage
precision = 1

[tool.coverage.html]
directory = "htmlcov"
```

### .coveragerc (Alternative)

```ini
[run]
source = src/mypackage
branch = True
omit =
    */tests/*
    */migrations/*

[report]
fail_under = 85
exclude_lines =
    pragma: no cover
    def __repr__
    if __name__ == .__main__.
show_missing = True

[html]
directory = htmlcov
```

### Excluding Code from Coverage

Use the `# pragma: no cover` comment for code that should not count toward coverage:

```python
def platform_specific_init():  # pragma: no cover
    """Only runs on Linux; tested in CI, not locally."""
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))


if __name__ == "__main__":  # pragma: no cover
    main()
```

Use exclusions sparingly. Every exclusion is a line you are choosing not to test.

---

## Setting Coverage Thresholds

### Minimum Coverage in CI

Set `fail_under` to make coverage a CI gate:

```toml
[tool.coverage.report]
fail_under = 85
```

```bash
# CI pipeline
pytest --cov=mypackage --cov-fail-under=85
```

### Choosing a Threshold

| Threshold | Appropriate For |
|-----------|----------------|
| 60-70%    | Early-stage projects, rapid prototyping |
| 75-85%    | Most production applications |
| 85-95%    | Libraries, financial/medical software |
| 95%+      | Safety-critical systems, core algorithms |

**Do not chase 100%**. Covering the last 5% often means testing trivial code (string representations, defensive error handling) with diminishing returns. The goal is meaningful coverage, not a vanity metric.

### Ratcheting Coverage

A practical approach: never let coverage *decrease*. Record current coverage, and fail CI if new changes lower it:

```bash
# Record baseline
pytest --cov=mypackage --cov-report=json
# Store .coverage or coverage.json as baseline

# In CI: compare against baseline
pytest --cov=mypackage --cov-fail-under=$(cat .coverage-baseline)
```

---

## Coverage Pitfalls

### Pitfall 1: High Coverage, Low Confidence

```python
def calculate_discount(price, membership):
    if membership == "gold":
        return price * 0.20
    elif membership == "silver":
        return price * 0.10
    else:
        return 0


# 100% line coverage, but no assertions on values!
def test_all_paths():
    calculate_discount(100, "gold")
    calculate_discount(100, "silver")
    calculate_discount(100, "bronze")
    # No assert statements — tests always pass
```

### Pitfall 2: Testing Implementation, Not Behavior

High coverage from testing internal state rather than observable behavior. The tests break on any refactoring, even when behavior is preserved.

### Pitfall 3: Coverage as a Target

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure." If engineers are rewarded for coverage numbers, they write shallow tests to boost the metric without improving quality.

### Pitfall 4: Ignoring Untested Edge Cases

Coverage shows which lines ran, not which *inputs* were tested. A function might be covered by one test case while critical edge cases (null, empty, boundary values, overflow) remain untested.

---

## Mutation Testing

Mutation testing answers: "If I introduce a bug, will my tests catch it?" It systematically modifies your source code (creating *mutants*) and checks whether tests fail. If a mutant survives (tests still pass), your tests have a gap.

### How It Works

1. The tool creates a *mutant* by changing one piece of source code (e.g., `>` becomes `>=`, `+` becomes `-`, `True` becomes `False`)
2. The full test suite runs against the mutant
3. If tests fail, the mutant is **killed** (good)
4. If tests pass, the mutant **survives** (your tests missed the bug)

### Using mutmut

```bash
pip install mutmut

# Run mutation testing
mutmut run --paths-to-mutate=src/mypackage/

# View results
mutmut results

# Show a specific surviving mutant
mutmut show 42

# Generate HTML report
mutmut html
```

### Example

```python
# discount.py
def apply_discount(price: float, percentage: float) -> float:
    if percentage < 0 or percentage > 100:
        raise ValueError("Invalid percentage")
    return price * (1 - percentage / 100)
```

```python
# test_discount.py
def test_apply_discount():
    assert apply_discount(100, 10) == 90.0
```

mutmut might create these mutants:

| Mutant | Change | Survives? |
|--------|--------|-----------|
| 1 | `percentage < 0` -> `percentage <= 0` | Yes (no test for 0%) |
| 2 | `percentage > 100` -> `percentage >= 100` | Yes (no test for 100%) |
| 3 | `1 - percentage / 100` -> `1 + percentage / 100` | No (caught!) |
| 4 | `percentage / 100` -> `percentage / 101` | Yes (only one test case) |

Surviving mutants reveal real gaps. Adding more tests kills them:

```python
def test_zero_discount():
    assert apply_discount(100, 0) == 100.0

def test_full_discount():
    assert apply_discount(100, 100) == 0.0

def test_half_discount():
    assert apply_discount(200, 50) == 100.0

def test_negative_percentage_raises():
    with pytest.raises(ValueError):
        apply_discount(100, -1)

def test_over_100_percentage_raises():
    with pytest.raises(ValueError):
        apply_discount(100, 101)
```

### Mutation Testing Limitations

- **Slow**: Runs the entire test suite for every mutant (can be hours for large codebases)
- **Equivalent mutants**: Some mutations do not change behavior (e.g., reordering independent statements)
- **Best for critical code**: Apply selectively to core business logic, not the entire codebase

---

## Code Quality Beyond Coverage

### Cyclomatic Complexity

High complexity means more paths to test. Tools like `radon` measure it:

```bash
pip install radon
radon cc src/mypackage/ -s -a
```

```
src/mypackage/processor.py
    F 12:0 process_order - C (14)   # Complexity 14 = many branches
    F 45:0 validate_input - A (3)   # Complexity 3 = simple
```

Aim for complexity under 10 per function. High-complexity functions need more tests *and* should be refactored.

### Static Analysis

Type checkers and linters catch bugs that tests might miss:

```bash
# Type checking
pip install mypy
mypy src/mypackage/

# Linting
pip install ruff
ruff check src/mypackage/
```

### Property-Based Testing

Instead of writing specific test cases, describe properties that should always hold. Hypothesis generates hundreds of random inputs:

```bash
pip install hypothesis
```

```python
from hypothesis import given
from hypothesis import strategies as st


def reverse_string(s: str) -> str:
    return s[::-1]


@given(st.text())
def test_reverse_is_involution(s):
    """Reversing twice returns the original string."""
    assert reverse_string(reverse_string(s)) == s

@given(st.text())
def test_reverse_preserves_length(s):
    assert len(reverse_string(s)) == len(s)

@given(st.lists(st.integers()))
def test_sort_is_idempotent(lst):
    """Sorting a sorted list returns the same list."""
    assert sorted(sorted(lst)) == sorted(lst)
```

---

## A Balanced Test Quality Strategy

1. **Start with behavior-focused unit tests** covering happy paths and critical edge cases
2. **Enable branch coverage** and target 80-85% as a reasonable floor
3. **Use mutation testing** on critical business logic to find assertion gaps
4. **Add property-based tests** for pure functions and data transformations
5. **Run static analysis** (mypy, ruff) in CI alongside tests
6. **Monitor coverage trends** — prevent decreases, do not obsess over maximizing

```toml
# A balanced pyproject.toml test configuration
[tool.pytest.ini_options]
addopts = "--cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80"
testpaths = ["tests"]

[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.",
    "raise NotImplementedError",
]
```

---

## Exercises

1. **Coverage gap analysis**: Take a small Python module (50-100 lines) and run `pytest --cov --cov-branch --cov-report=term-missing`. Identify the uncovered branches. Write tests to reach 95% branch coverage and explain what the remaining 5% covers.

2. **Mutation testing**: Install `mutmut` and run it against a module with its test suite. Identify at least 3 surviving mutants. For each, explain why the mutant survived and write a test that kills it.

3. **Coverage vs quality debate**: Write a module with 100% line coverage but at least 3 surviving mutants. Then write a module with 80% line coverage but zero surviving mutants. Explain which test suite is better and why.

---

**License**: CC BY-NC 4.0
