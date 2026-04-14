"""
12 Debugging Workflow
=====================
Demonstrates end-to-end debugging case studies combining
multiple techniques: print, pdb, logging, testing, profiling.
"""
import datetime
import logging
import time

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --- Case Study 1: Data Corruption ---

def case_study_data_corruption():
    """Debug a data processing pipeline with corrupted ages."""
    print("=" * 60)
    print("Case Study 1: Silent Data Corruption")
    print("=" * 60)

    raw_data = [
        {"name": "Alice", "age": "30"},
        {"name": "Bob", "age": "25"},
        {"name": "Charlie", "age": "1990"},  # Birth year, not age!
        {"name": "Diana", "age": "28"},
        {"name": "Eve", "age": "N/A"},       # Invalid
    ]

    # Buggy version
    def calculate_avg_age_buggy(data):
        ages = []
        for record in data:
            try:
                ages.append(int(record["age"]))
            except ValueError:
                ages.append(0)
        return sum(ages) / len(ages) if ages else 0

    buggy_avg = calculate_avg_age_buggy(raw_data)
    print(f"\n  Buggy average age: {buggy_avg:.1f} (unreasonably high!)")

    # Debug with print
    print("\n  Debugging with print:")
    for record in raw_data:
        raw = record["age"]
        try:
            parsed = int(raw)
        except ValueError:
            parsed = None
        print(f"    name={record['name']}, raw={raw!r}, parsed={parsed}")

    # Fixed version
    def parse_age(value):
        try:
            num = int(value)
        except (ValueError, TypeError):
            return None
        current_year = datetime.date.today().year
        if num > 120:
            return current_year - num
        if num < 0:
            return None
        return num

    def calculate_avg_age_fixed(data):
        ages = [parse_age(r["age"]) for r in data]
        valid = [a for a in ages if a is not None]
        return sum(valid) / len(valid) if valid else 0

    fixed_avg = calculate_avg_age_fixed(raw_data)
    print(f"\n  Fixed average age: {fixed_avg:.1f}")

    # Verify with tests
    print("\n  Tests:")
    assert parse_age("30") == 30, "Normal age"
    birth_year_age = parse_age("1990")
    assert birth_year_age is not None and 30 <= birth_year_age <= 40, "Birth year"
    assert parse_age("N/A") is None, "Invalid"
    assert parse_age("-5") is None, "Negative"
    print("    All tests passed!")
    print()


# --- Case Study 2: Intermittent None Error ---

def case_study_none_error():
    """Debug an intermittent NoneType error."""
    print("=" * 60)
    print("Case Study 2: Intermittent NoneType Error")
    print("=" * 60)

    users_db = {1: "Alice", 2: "Bob"}

    def get_user(user_id):
        return users_db.get(user_id)

    # Buggy version
    def greet_user_buggy(user_id):
        user = get_user(user_id)
        return f"Hello, {user.upper()}!"  # Crashes if user is None

    # Test
    print(f"\n  greet_user(1): {greet_user_buggy(1)}")
    try:
        greet_user_buggy(999)
    except AttributeError as e:
        print(f"  greet_user(999): AttributeError: {e}")

    # Fixed version
    def greet_user_fixed(user_id):
        user = get_user(user_id)
        if user is None:
            logger.warning("User %d not found", user_id)
            return "Hello, Guest!"
        return f"Hello, {user.upper()}!"

    print(f"\n  Fixed greet_user(1): {greet_user_fixed(1)}")
    print(f"  Fixed greet_user(999): {greet_user_fixed(999)}")
    print()


# --- Case Study 3: Performance Regression ---

def case_study_performance():
    """Debug a performance regression using profiling."""
    print("=" * 60)
    print("Case Study 3: Performance Regression")
    print("=" * 60)

    # Simulate N+1 query problem
    users = {i: f"User_{i}" for i in range(1000)}

    def get_user_slow(user_id):
        time.sleep(0.0001)  # Simulate DB query
        return users.get(user_id)

    def get_users_batch(user_ids):
        time.sleep(0.001)  # Simulate single batch query
        return {uid: users.get(uid) for uid in user_ids}

    orders = [{"user_id": i % 100, "amount": 10 + i} for i in range(500)]

    # Slow version (N+1)
    start = time.perf_counter()
    slow_results = []
    for order in orders:
        user = get_user_slow(order["user_id"])
        slow_results.append(f"{user}: ${order['amount']}")
    slow_time = time.perf_counter() - start
    print(f"\n  Slow (N+1 queries): {slow_time:.3f}s for {len(orders)} orders")

    # Fast version (batch)
    start = time.perf_counter()
    user_ids = {o["user_id"] for o in orders}
    user_map = get_users_batch(user_ids)
    fast_results = []
    for order in orders:
        user = user_map[order["user_id"]]
        fast_results.append(f"{user}: ${order['amount']}")
    fast_time = time.perf_counter() - start
    print(f"  Fast (batch query): {fast_time:.3f}s for {len(orders)} orders")
    print(f"  Speedup: {slow_time / fast_time:.0f}x faster")
    print()


# --- Tool Selection Guide ---

def tool_selection_guide():
    """Print the debugging tool selection guide."""
    print("=" * 60)
    print("Debugging Tool Selection Guide")
    print("=" * 60)

    guides = [
        ("Crash with traceback", "Read error message (Lesson 1) → print/pdb (2-3)"),
        ("Wrong output (no crash)", "Print debugging (2) → Check patterns (4) → Binary search (5)"),
        ("Intermittent failure", "Add logging (6) → Write reproducing test (7)"),
        ("'It worked before'", "git bisect (11) → git diff"),
        ("Performance regression", "cProfile (10) → timeit to compare"),
        ("Type-related error", "Add type hints → run mypy (9)"),
        ("Code smell/style", "Run ruff (8)"),
    ]
    for problem, solution in guides:
        print(f"\n  {problem}:")
        print(f"    → {solution}")
    print()


# --- Debugging Checklist ---

def print_debugging_checklist():
    """Print the universal debugging checklist."""
    print("=" * 60)
    print("Universal Debugging Checklist")
    print("=" * 60)

    checklist = [
        "1. REPRODUCE - Can you make it happen reliably?",
        "2. ISOLATE   - Create a minimal example",
        "3. LOCATE    - Binary search + tracing to find exact line",
        "4. UNDERSTAND- Why does this line produce wrong result?",
        "5. FIX       - Change the code",
        "6. VERIFY    - Does fix work? Anything else break?",
        "7. PREVENT   - Add a test for regression",
    ]
    for item in checklist:
        print(f"  {item}")
    print()


if __name__ == "__main__":
    case_study_data_corruption()
    case_study_none_error()
    case_study_performance()
    tool_selection_guide()
    print_debugging_checklist()
