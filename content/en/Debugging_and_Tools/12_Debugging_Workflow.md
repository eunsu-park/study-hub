# Debugging Workflow

**Previous**: [Version Control for Debugging](./11_Version_Control_for_Debugging.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Combine multiple debugging techniques into a cohesive workflow
2. Apply the REPRODUCE-ISOLATE-FIX-VERIFY cycle to real bugs
3. Triage bugs by severity and choose the appropriate debugging depth
4. Debug data processing bugs using systematic data tracing
5. Debug API and integration bugs by isolating the boundary
6. Debug performance bugs using profiling-driven optimization
7. Apply a post-mortem process to learn from resolved bugs
8. Build a personal debugging toolkit and checklist

---

This final lesson brings together everything you've learned: error reading, print debugging, debuggers, bug patterns, strategy, logging, testing, linters, type checking, profiling, and version control. Real-world bugs rarely fit neatly into a single category. They require you to combine multiple techniques, switching tools as you narrow down the cause. This lesson walks through complete debugging case studies to show how experienced developers approach bugs in practice.

> **The Master Debugger:** An expert debugger doesn't know more tools than a beginner -- they know when to use each one and how to switch between them efficiently.

---

## 1. The Universal Debugging Workflow

Every bug, regardless of complexity, follows this cycle:

```
┌──────────────────────────────────────────────────┐
│            The Debugging Cycle                    │
│                                                   │
│   ┌─────────────┐                                │
│   │  REPRODUCE  │  Can you make it happen?       │
│   └──────┬──────┘                                │
│          ▼                                        │
│   ┌─────────────┐                                │
│   │   ISOLATE   │  Strip away irrelevant code    │
│   └──────┬──────┘                                │
│          ▼                                        │
│   ┌─────────────┐                                │
│   │   LOCATE    │  Find the exact line/value     │
│   └──────┬──────┘                                │
│          ▼                                        │
│   ┌─────────────┐                                │
│   │ UNDERSTAND  │  Why does it go wrong?         │
│   └──────┬──────┘                                │
│          ▼                                        │
│   ┌─────────────┐                                │
│   │     FIX     │  Change the code               │
│   └──────┬──────┘                                │
│          ▼                                        │
│   ┌─────────────┐                                │
│   │   VERIFY    │  Test the fix, add regression  │
│   └──────┬──────┘  test, check for side effects  │
│          ▼                                        │
│   ┌─────────────┐                                │
│   │   REFLECT   │  What can you learn from this? │
│   └─────────────┘                                │
└──────────────────────────────────────────────────┘
```

---

## 2. Case Study 1: The Silent Data Corruption

### Scenario

A data processing script reads CSV files and produces summary statistics. Users report that the average age is unreasonably high (150+) for some datasets.

### Step 1: Reproduce

```python
# reproduce.py
from data_processor import calculate_stats

# Use the reported dataset
stats = calculate_stats("users_2024.csv")
print(f"Average age: {stats['avg_age']}")  # 167.5 -- confirmed!
```

### Step 2: Isolate

```python
# Create minimal data that shows the bug
test_csv = """name,age,city
Alice,30,Seoul
Bob,25,Busan
Charlie,N/A,Daegu
Diana,28,Incheon"""

# Write to temp file and test
import tempfile, os
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write(test_csv)
    temp_path = f.name

stats = calculate_stats(temp_path)
print(f"Average age: {stats['avg_age']}")  # Still high? Or correct?
os.unlink(temp_path)
```

### Step 3: Locate with Print Debugging

```python
def calculate_stats(csv_path):
    ages = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = row["age"]
            print(f"  Row: {row['name']}, raw age={age!r}, type={type(age).__name__}")
            ages.append(int(age))  # CRASH on "N/A"!
    return {"avg_age": sum(ages) / len(ages)}
```

But the user said it doesn't crash -- it just gives wrong numbers. Let's look at the actual code:

```python
def calculate_stats(csv_path):
    ages = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ages.append(int(row["age"]))
            except ValueError:
                ages.append(0)  # BUG: Treating invalid as 0
    # But wait, average is HIGH, not LOW...
```

Hmm, treating invalid ages as 0 would make the average *lower*, not higher. Let's look more carefully:

```python
# Print every raw value
for row in reader:
    print(f"  age={row['age']!r}")
```

Output reveals:
```
  age='30'
  age='25'
  age='1990'    # This is a birth YEAR, not an age!
  age='28'
```

**Root cause**: Some rows contain birth years (1990) instead of ages (34). The original code silently treated them as ages.

### Step 4: Fix

```python
import datetime

def parse_age(value):
    """Parse age, handling both ages and birth years."""
    try:
        num = int(value)
    except (ValueError, TypeError):
        return None
    
    current_year = datetime.date.today().year
    if num > 120:  # Likely a birth year
        return current_year - num
    if num < 0:
        return None
    return num
```

### Step 5: Verify with Tests

```python
def test_parse_age_normal():
    assert parse_age("30") == 30

def test_parse_age_birth_year():
    assert parse_age("1990") == datetime.date.today().year - 1990

def test_parse_age_invalid():
    assert parse_age("N/A") is None
    assert parse_age("") is None
    assert parse_age(None) is None
```

### Tools Used

1. **Print debugging** -- to see raw data values
2. **repr()** -- to identify unexpected data formats
3. **Testing** -- to verify the fix and prevent regression

---

## 3. Case Study 2: The Intermittent Crash

### Scenario

A web API occasionally returns 500 errors. It works most of the time, but fails unpredictably.

### Step 1: Reproduce

Check the logs:
```python
import logging
logger = logging.getLogger(__name__)

@app.route("/api/users/<int:user_id>")
def get_user(user_id):
    logger.info(f"Request: GET /api/users/{user_id}")
    user = db.find_user(user_id)
    logger.debug(f"Found user: {user!r}")
    profile = user.get_profile()  # Crashes HERE sometimes
    return jsonify(profile.to_dict())
```

Log output on failure:
```
INFO: Request: GET /api/users/42
DEBUG: Found user: None
ERROR: 'NoneType' has no attribute 'get_profile'
```

### Step 2: Locate

The bug is clear from the log: `user` is `None` when the user doesn't exist.

### Step 3: Fix

```python
@app.route("/api/users/<int:user_id>")
def get_user(user_id):
    user = db.find_user(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    profile = user.get_profile()
    return jsonify(profile.to_dict())
```

### Step 4: Prevent with Type Hints

```python
def find_user(user_id: int) -> User | None:
    ...
```

Now mypy would have caught the missing None check.

### Tools Used

1. **Logging** -- to capture the state at failure time
2. **None pattern recognition** -- from Lesson 4
3. **Type hints + mypy** -- to prevent similar bugs

---

## 4. Case Study 3: The Performance Regression

### Scenario

A report generation script that used to complete in 5 seconds now takes 2 minutes.

### Step 1: Profile

```python
import cProfile

cProfile.run('generate_report()', sort='cumtime')
```

Output:
```
   ncalls  tottime  cumtime
   100000    95.2    95.2   db.py:15(get_user_by_id)
        1     0.5     0.5   report.py:30(format_report)
        1     0.1     0.1   report.py:10(load_data)
```

### Step 2: Locate

`get_user_by_id` is called 100,000 times! That's the N+1 query problem.

### Step 3: Understand

```python
# Current code: one database query per user
def generate_report():
    orders = load_all_orders()       # 1 query
    for order in orders:
        user = get_user_by_id(order.user_id)  # 100,000 queries!
        ...
```

### Step 4: Fix

```python
# Fixed: batch load all users
def generate_report():
    orders = load_all_orders()
    user_ids = {o.user_id for o in orders}
    users = get_users_by_ids(user_ids)  # 1 query with IN clause
    user_map = {u.id: u for u in users}
    for order in orders:
        user = user_map[order.user_id]  # O(1) dict lookup
        ...
```

### Step 5: Verify

```bash
python -m timeit -n 1 "generate_report()"
# Before: 120.3 sec per loop
# After:    4.8 sec per loop
```

### Tools Used

1. **cProfile** -- to find the bottleneck
2. **N+1 query pattern recognition** -- common performance bug
3. **timeit** -- to verify the improvement

---

## 5. Case Study 4: The Git Bisect Investigation

### Scenario

The test suite passed last week but now 3 tests are failing. Nobody knows which change broke them.

### Step 1: Find a Known-Good Commit

```bash
git log --oneline -20
# Find last week's commits
git checkout abc123
python -m pytest tests/test_calc.py  # All pass!
git checkout main
```

### Step 2: Bisect

```bash
git bisect start
git bisect bad HEAD
git bisect good abc123
git bisect run python -m pytest tests/test_calc.py -x
```

After ~5 steps:
```
def456 is the first bad commit
Author: Charlie
Date: Wed Jan 17
Message: "Refactor: extract helper functions"
```

### Step 3: Examine the Breaking Commit

```bash
git diff def456~1 def456
```

Found: A refactoring accidentally changed the order of arguments in a helper function.

### Tools Used

1. **git bisect** -- automated binary search through history
2. **pytest** -- automated test execution
3. **git diff** -- to see what changed

---

## 6. Choosing the Right Tool

```
What kind of bug?
│
├─ Crash with traceback?
│   └─ Read the error message (Lesson 1)
│      Then: print debug or use pdb (Lessons 2-3)
│
├─ Wrong output (no crash)?
│   └─ Print debugging to trace data flow (Lesson 2)
│      Then: Check common patterns (Lesson 4)
│      Then: Binary search debugging (Lesson 5)
│
├─ Intermittent failure?
│   └─ Add logging to capture state (Lesson 6)
│      Then: Write a test that reproduces it (Lesson 7)
│
├─ "It worked before"?
│   └─ git bisect to find breaking commit (Lesson 11)
│      Then: git diff to see what changed
│
├─ Performance regression?
│   └─ cProfile to find bottleneck (Lesson 10)
│      Then: timeit to compare solutions
│
├─ Type-related error?
│   └─ Add type hints and run mypy (Lesson 9)
│
└─ Code smell / style issue?
    └─ Run ruff (Lesson 8)
```

---

## 7. Building Your Personal Debugging Toolkit

### Essential Tools

```
□ Python built-ins: breakpoint(), pdb, traceback
□ Print debugging: f-strings with !r, pprint
□ Logging: logging module configured for your project
□ Testing: pytest installed and configured
□ Linting: ruff (replaces pylint, flake8, black, isort)
□ Type checking: mypy with pyproject.toml config
□ Profiling: cProfile, timeit, tracemalloc
□ Version control: git (bisect, blame, diff, log)
```

### Editor Setup

```
□ Debugger integration (breakpoints, variable inspector)
□ Ruff extension (auto-lint on save)
□ Mypy extension (type error highlighting)
□ Git integration (blame, diff view)
```

### Personal Checklist

Create your own debugging checklist based on the bugs you encounter most often:

```
My Debugging Checklist:
□ Read the error message carefully (bottom-to-top)
□ Can I reproduce it consistently?
□ What changed recently? (git log, git diff)
□ Is it a known bug pattern? (off-by-one, None, mutable default)
□ What does the data look like at the crash point? (print with !r)
□ Does the test suite still pass?
□ Have I checked for None?
□ Have I checked the types?
```

---

## 8. The Debugging Post-Mortem

After fixing a significant bug, ask:

1. **What was the root cause?** (not the symptom)
2. **How did it get past code review?** (process issue?)
3. **How did it get past tests?** (coverage gap?)
4. **Could linters or type checking have caught it?**
5. **What test should we add to prevent regression?**
6. **Is this a pattern that might exist elsewhere?** (search for similar bugs)

### Example Post-Mortem Entry

```
## Bug: Discount calculation wrong for orders > $1000
Date: 2024-01-20
Severity: High (affected billing)
Time to fix: 2 hours

### Root Cause
Commit def456 changed the discount threshold from 1000 to 100
during a "code cleanup" refactor. The magic number 1000 wasn't
documented, so the refactorer assumed 100 was correct.

### How It Was Found
Customer complaint → reproduced → git bisect → found commit

### Prevention
1. Added unit test for threshold boundary
2. Replaced magic number with named constant LARGE_ORDER_THRESHOLD
3. Added code comment explaining the business rule

### Takeaway
Magic numbers should always be named constants with documentation.
```

---

## 9. Debugging Mindset

### Be Systematic, Not Heroic

- Don't try to solve it in your head -- use tools
- Don't assume -- verify with data
- Don't chase symptoms -- find root causes
- Don't skip the test -- prevent regressions

### Stay Calm

- Bugs are normal; they're not a reflection of your competence
- Every bug you fix teaches you something
- The best developers aren't the ones who write bug-free code -- they're the ones who find and fix bugs efficiently

### Keep Learning

- Every new bug is a new pattern for your mental library
- Review other people's bug fixes to learn their techniques
- Practice debugging deliberately -- don't just "get it working"

---

## 10. Quick Reference: All Techniques

| Technique | Best For | Lesson |
|-----------|----------|--------|
| Reading tracebacks | Understanding crash errors | 1 |
| Print debugging | Quick data flow inspection | 2 |
| pdb / breakpoint() | Line-by-line execution analysis | 3 |
| Bug pattern recognition | Preventing known bug types | 4 |
| Scientific debugging method | Systematic bug hunting | 5 |
| Logging | Production diagnostics | 6 |
| Testing | Verifying fixes, preventing regression | 7 |
| Linters (ruff) | Catching bugs before runtime | 8 |
| Type checking (mypy) | Catching type errors statically | 9 |
| Profiling (cProfile) | Finding performance bottlenecks | 10 |
| Git tools (bisect, blame) | Finding when/who introduced a bug | 11 |

---

## Summary

- Real debugging combines multiple tools and techniques
- Follow the REPRODUCE-ISOLATE-LOCATE-UNDERSTAND-FIX-VERIFY cycle
- Choose the right tool based on the type of bug
- Build a personal debugging toolkit and checklist
- Conduct post-mortems to learn from bugs and prevent recurrence
- Debugging is a skill that improves with deliberate practice
- The goal is not just to fix the bug but to understand it and prevent similar ones

---

## Exercises

1. Debug a multi-step data pipeline using the full debugging workflow
2. Identify the right debugging technique for each of 5 different bug scenarios
3. Write a post-mortem for a bug you've fixed
4. Create a personal debugging checklist

**Previous**: [Version Control for Debugging](./11_Version_Control_for_Debugging.md)
