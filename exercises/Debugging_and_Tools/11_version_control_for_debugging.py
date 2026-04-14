"""
Exercise 11: Version Control for Debugging

Practice git debugging concepts: bisect, blame, diff, and log.
"""


def simulate_git_bisect(commits, test_func):
    """Simulate git bisect to find the first bad commit.

    Given a list of commits (oldest to newest) and a test function
    that returns True for "good" commits and False for "bad" commits,
    find the first bad commit using binary search.

    Args:
        commits: List of commit identifiers (strings).
        test_func: A function that takes a commit and returns
                   True if good, False if bad.

    Returns:
        str: The identifier of the first bad commit.
    """
    # TODO: Implement binary search to find first bad commit
    pass


def parse_blame_output(blame_lines):
    """Parse simulated git blame output.

    Each line has format: "HASH (AUTHOR DATE) LINE_NUM) CODE"
    Extract the author and line number for each line.

    Args:
        blame_lines: List of blame output strings.

    Returns:
        list: List of dicts with "author", "line", and "code" keys.
    """
    # TODO: Parse each blame line into structured data
    pass


def find_changed_functions(diff_text):
    """Find which functions were modified in a diff.

    Given a unified diff text, identify Python function names
    that were added or modified (lines starting with + or -
    that contain "def ").

    Args:
        diff_text: A string containing unified diff output.

    Returns:
        set: Set of function names that were changed.
    """
    # TODO: Parse the diff and find changed function names
    pass


def write_bisect_script(test_command):
    """Generate a git bisect run script.

    Create a shell script that can be used with
    `git bisect run ./script.sh`.

    The script should:
    1. Run the test command
    2. Exit with 0 if the test passes (good commit)
    3. Exit with 1 if the test fails (bad commit)

    Args:
        test_command: The command to run (e.g., "python -m pytest tests/")

    Returns:
        str: The shell script content.
    """
    # TODO: Generate the bisect run script
    pass


if __name__ == "__main__":
    # Test simulate_git_bisect
    commits = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    # Bug introduced at c5
    test_func = lambda c: int(c[1]) < 5

    result = simulate_git_bisect(commits, test_func)
    assert result == "c5", f"Expected 'c5', got {result!r}"
    print("simulate_git_bisect: PASSED")

    # Test parse_blame_output
    blame_lines = [
        'a1b2c3d (Alice   2024-01-10 14:30)  1) def calculate(x, y):',
        'a1b2c3d (Alice   2024-01-10 14:30)  2)     return x + y',
        'e5f6g7h (Bob     2024-01-15 09:15)  3)     # modified',
    ]
    parsed = parse_blame_output(blame_lines)
    assert parsed is not None
    assert len(parsed) == 3
    assert parsed[0]["author"] == "Alice"
    assert parsed[2]["author"] == "Bob"
    print("parse_blame_output: PASSED")

    # Test find_changed_functions
    diff_text = """
diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,5 +1,5 @@
-def add(a, b):
-    return a + b
+def add(a, b):
+    return a + b + 0  # changed

-def multiply(a, b):
+def multiply(a, b):
     return a * b
+
+def new_function():
+    pass
"""
    changed = find_changed_functions(diff_text)
    assert changed is not None
    assert "add" in changed
    assert "new_function" in changed
    print("find_changed_functions: PASSED")

    # Test write_bisect_script
    script = write_bisect_script("python -m pytest tests/test_calc.py")
    assert script is not None
    assert "pytest" in script
    assert "exit" in script.lower() or "$?" in script
    print("write_bisect_script: PASSED")
