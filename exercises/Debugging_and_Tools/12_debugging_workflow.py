"""
Exercise 12: Debugging Workflow

Practice end-to-end debugging by combining multiple techniques.
"""


def debug_data_pipeline(records):
    """Debug and fix this complete data pipeline.

    The pipeline should:
    1. Parse ages from string to int (handle invalid values)
    2. Filter to adults only (age >= 18)
    3. Calculate the average age
    4. Return a summary dict

    There are MULTIPLE bugs. Use the full debugging workflow:
    - Read error messages
    - Add strategic prints
    - Check common bug patterns
    - Fix each bug
    - Verify with test cases

    Args:
        records: List of dicts with "name" and "age" (age as string).

    Returns:
        dict: {"count": int, "average_age": float, "names": list}
    """
    # TODO: Find and fix ALL bugs in this pipeline
    ages = []
    names = []

    for record in records:
        age = int(record["age"])  # BUG 1: crashes on non-numeric
        if age > 18:              # BUG 2: should be >= 18
            ages.append(age)
            names.append(record["name"])

    avg = sum(ages) / len(ages)   # BUG 3: crashes on empty list

    return {
        "count": len(names),
        "average_age": avg,
        "names": names,
    }


def choose_debugging_technique(scenario):
    """Choose the best debugging technique for each scenario.

    Args:
        scenario: A string describing the bug scenario.

    Returns:
        str: The recommended technique. One of:
            "traceback", "print", "debugger", "logging",
            "testing", "linter", "type_check", "profiling",
            "git_bisect", "git_blame"
    """
    # TODO: Return the best technique for each scenario
    pass


def write_postmortem(bug_description, root_cause, fix_description,
                     prevention_steps):
    """Write a structured post-mortem for a resolved bug.

    Args:
        bug_description: What the bug was.
        root_cause: Why it happened.
        fix_description: How it was fixed.
        prevention_steps: List of steps to prevent recurrence.

    Returns:
        str: Formatted post-mortem report.
    """
    # TODO: Create a structured post-mortem report
    pass


def create_debugging_checklist():
    """Create a personal debugging checklist.

    Return a list of debugging steps in the recommended order.
    Each step should be a dict with "step" (number), "action"
    (what to do), and "tool" (which tool to use).

    Returns:
        list: List of checklist step dicts.
    """
    # TODO: Create a comprehensive debugging checklist
    # Should include at least 7 steps covering:
    # - Reading errors
    # - Reproducing
    # - Isolating
    # - Locating
    # - Understanding
    # - Fixing
    # - Verifying
    pass


if __name__ == "__main__":
    # Test debug_data_pipeline
    records = [
        {"name": "Alice", "age": "30"},
        {"name": "Bob", "age": "17"},
        {"name": "Charlie", "age": "N/A"},
        {"name": "Diana", "age": "18"},
        {"name": "Eve", "age": "25"},
    ]
    result = debug_data_pipeline(records)
    assert result["count"] == 3, f"Expected 3 adults, got {result['count']}"
    assert 20 <= result["average_age"] <= 30, f"Avg age: {result['average_age']}"
    assert "Alice" in result["names"]
    assert "Diana" in result["names"]
    assert "Bob" not in result["names"]  # Under 18
    print("debug_data_pipeline: PASSED")

    # Test with empty valid data
    empty_records = [{"name": "Kid", "age": "5"}]
    result = debug_data_pipeline(empty_records)
    assert result["count"] == 0
    assert result["average_age"] == 0.0
    print("debug_data_pipeline (empty): PASSED")

    # Test choose_debugging_technique
    scenarios = {
        "Program crashes with ZeroDivisionError traceback": "traceback",
        "Function returns wrong value, no error": "print",
        "Need to inspect many variables at one point": "debugger",
        "Intermittent failure in production": "logging",
        "Code worked last week, broken now": "git_bisect",
    }
    for scenario, expected in scenarios.items():
        result = choose_debugging_technique(scenario)
        assert result == expected, f"For '{scenario}': got {result!r}, expected {expected!r}"
    print("choose_debugging_technique: PASSED")

    # Test write_postmortem
    report = write_postmortem(
        bug_description="Average age calculation returns 0",
        root_cause="Division by zero when no valid ages exist",
        fix_description="Added check for empty list before division",
        prevention_steps=["Add unit test for empty input", "Add type hints"],
    )
    assert report is not None
    assert "root cause" in report.lower() or "Root Cause" in report
    print("write_postmortem: PASSED")

    # Test create_debugging_checklist
    checklist = create_debugging_checklist()
    assert checklist is not None
    assert len(checklist) >= 7, f"Need at least 7 steps, got {len(checklist)}"
    assert all("step" in item and "action" in item for item in checklist)
    print("create_debugging_checklist: PASSED")
