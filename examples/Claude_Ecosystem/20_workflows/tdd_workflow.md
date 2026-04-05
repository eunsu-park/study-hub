# TDD Workflow with Claude Code

A step-by-step guide for Test-Driven Development using Claude Code.

## The Red-Green-Refactor Cycle

### Step 1: RED — Write a Failing Test

```
User: Write a failing test for a function that validates email addresses.
      Put it in tests/test_email.py. Do NOT implement the function yet.
```

Claude writes:
```python
# tests/test_email.py
from src.email import validate_email

def test_valid_email():
    assert validate_email("user@example.com") is True

def test_invalid_no_at():
    assert validate_email("userexample.com") is False

def test_invalid_no_domain():
    assert validate_email("user@") is False
```

### Step 2: GREEN — Minimal Implementation

```
User: Now implement validate_email in src/email.py.
      Write the minimum code to pass the tests. Run the tests.
```

### Step 3: REFACTOR — Improve Code Quality

```
User: Refactor validate_email to use a regex pattern. Keep the tests passing.
      Add type hints. Run the tests.
```

## Best Practices for TDD with Claude Code

1. **Be explicit about the phase**: Tell Claude whether you're in RED, GREEN, or REFACTOR
2. **Always include "run the tests"**: Claude will verify each change
3. **One test at a time**: Start with the simplest case
4. **Let Claude iterate**: If tests fail, Claude will fix and retry
5. **Use /compact between cycles**: Keep context fresh for long sessions

## Example Prompts by Phase

| Phase | Prompt Pattern |
|-------|---------------|
| RED | "Write a failing test for [feature]. Do NOT implement yet." |
| GREEN | "Implement the minimum code to pass the tests. Run them." |
| REFACTOR | "Refactor [target] to use [pattern]. Keep tests passing." |
