"""
Exercise 05: Debugging Strategy

Practice systematic debugging: binary search, MRE creation,
and structured bug reports.
"""


def find_bug_binary_search(data):
    """Use binary search debugging to find the bug in this pipeline.

    The pipeline should:
    1. Filter out negative numbers
    2. Square each remaining number
    3. Sum the results

    Example: [1, -2, 3, -4, 5] → [1, 3, 5] → [1, 9, 25] → 35

    One step has a bug. Use binary search (check intermediate
    results) to find and fix it.

    Args:
        data: A list of integers.

    Returns:
        int: Sum of squares of non-negative numbers.
    """
    # TODO: Find and fix the bug using binary search debugging
    # Step 1: Filter negatives
    filtered = [x for x in data if x > 0]

    # Step 2: Square (BUG: cubes instead of squares)
    squared = [x ** 3 for x in filtered]

    # Step 3: Sum
    return sum(squared)


def create_mre(full_dataset):
    """Create a minimal reproducible example for a bug.

    The full_dataset has many records. The bug is:
    calculate_average returns wrong results for datasets
    containing the string "N/A" as an age value.

    Create and return the SMALLEST list that reproduces the bug.
    Then fix calculate_average to handle "N/A" values.

    Args:
        full_dataset: A large list of dicts with "name" and "age".

    Returns:
        tuple: (mre_data, fixed_average)
            - mre_data: minimal list that shows the bug
            - fixed_average: correct average of valid ages
    """
    # TODO: Create MRE and fix the calculation

    def calculate_average(records):
        """Buggy: crashes on non-numeric ages."""
        ages = [int(r["age"]) for r in records]  # Crashes on "N/A"
        return sum(ages) / len(ages)

    # Create your MRE here (smallest dataset that shows the bug)
    mre_data = None  # TODO: Create minimal example

    # Fix the function
    def calculate_average_fixed(records):
        # TODO: Handle non-numeric ages
        pass

    fixed_avg = None  # TODO: Calculate fixed average on full_dataset
    return mre_data, fixed_avg


def write_bug_report(symptom, expected, actual, steps_tried):
    """Write a structured bug report.

    Args:
        symptom: Description of what went wrong.
        expected: What should have happened.
        actual: What actually happened.
        steps_tried: List of debugging steps already attempted.

    Returns:
        str: A formatted bug report string.
    """
    # TODO: Implement a structured bug report formatter
    # Include: Summary, Expected, Actual, Steps to Reproduce, What Tried
    pass


def apply_scientific_method(buggy_func, test_input, expected_output):
    """Apply the scientific debugging method to find a bug.

    1. Observe: Call buggy_func and note the wrong output
    2. Hypothesize: Form a hypothesis about the cause
    3. Test: Verify or reject the hypothesis
    4. Fix: Return the corrected output

    Args:
        buggy_func: A function that produces wrong output.
        test_input: Input to pass to the function.
        expected_output: What the function should return.

    Returns:
        tuple: (actual_output, hypothesis, corrected_output)
    """
    # TODO: Implement the scientific debugging method
    pass


if __name__ == "__main__":
    # Test find_bug_binary_search
    result = find_bug_binary_search([1, -2, 3, -4, 5])
    assert result == 35, f"Expected 35, got {result}"
    result = find_bug_binary_search([2, 4])
    assert result == 20, f"Expected 20, got {result}"
    print("find_bug_binary_search: PASSED")

    # Test create_mre
    full_data = [
        {"name": "Alice", "age": "30"},
        {"name": "Bob", "age": "25"},
        {"name": "Charlie", "age": "N/A"},
        {"name": "Diana", "age": "28"},
    ]
    mre, avg = create_mre(full_data)
    assert mre is not None, "MRE should not be None"
    assert len(mre) <= 2, f"MRE should be minimal, got {len(mre)} records"
    assert avg is not None, "Fixed average should not be None"
    assert 25 <= avg <= 30, f"Average should be ~27.7, got {avg}"
    print("create_mre: PASSED")

    # Test write_bug_report
    report = write_bug_report(
        symptom="Function returns 0",
        expected="Sum of positive numbers (13)",
        actual="0",
        steps_tried=["Checked input data", "Added print statements"],
    )
    assert report is not None
    assert "expected" in report.lower() or "Expected" in report
    print("write_bug_report: PASSED")
