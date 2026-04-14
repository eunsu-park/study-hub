"""
11 Version Control for Debugging
=================================
Demonstrates git debugging techniques: diff, blame, bisect,
and log search patterns (conceptual examples).
"""


def git_diff_examples():
    """Show git diff usage patterns for debugging."""
    print("=== git diff for Debugging ===")
    commands = [
        ("git diff", "Show unstaged changes"),
        ("git diff --staged", "Show staged changes"),
        ("git diff HEAD~3 HEAD", "Changes in last 3 commits"),
        ("git diff abc123 HEAD -- src/calc.py", "Diff specific file between commits"),
        ("git diff --name-only abc123 HEAD", "List changed files only"),
        ("git diff --stat abc123 HEAD", "Show stats per file"),
    ]
    for cmd, desc in commands:
        print(f"  {cmd}")
        print(f"    → {desc}")
    print()


def git_blame_examples():
    """Show git blame usage for finding who changed code."""
    print("=== git blame for Debugging ===")

    # Simulated blame output
    blame_output = """
    a1b2c3d (Alice  2024-01-10)  1) def calculate(x, y):
    a1b2c3d (Alice  2024-01-10)  2)     '''Calculate the sum.'''
    e5f6g7h (Bob    2024-01-15)  3)     return x * y  # Changed + to *
    a1b2c3d (Alice  2024-01-10)  4)
    """
    print("  Simulated blame output:")
    print(blame_output)
    print("  Analysis:")
    print("  - Line 3 was changed by Bob on Jan 15")
    print("  - Original: return x + y (addition)")
    print("  - Changed to: return x * y (multiplication)")
    print("  - Run: git show e5f6g7h to see the full commit")
    print()

    commands = [
        ("git blame file.py", "Show who changed each line"),
        ("git blame -L 10,20 file.py", "Blame specific line range"),
        ("git blame -w file.py", "Ignore whitespace changes"),
        ("git show COMMIT", "See full commit details"),
    ]
    for cmd, desc in commands:
        print(f"  {cmd}")
        print(f"    → {desc}")
    print()


def git_bisect_demo():
    """Demonstrate git bisect concept with simulation."""
    print("=== git bisect Simulation ===")

    # Simulate a series of commits
    commits = [
        ("abc001", "v1.0", True),
        ("abc002", "Add validation", True),
        ("abc003", "Refactor utils", True),
        ("abc004", "Optimize calc", False),  # BUG INTRODUCED HERE
        ("abc005", "Add logging", False),
        ("abc006", "Update docs", False),
        ("abc007", "Fix typo", False),
        ("abc008", "HEAD", False),
    ]

    print("  Commit history:")
    for sha, msg, good in commits:
        status = "✓ good" if good else "✗ bad"
        print(f"    {sha} {msg:<20} {status}")

    print("\n  Binary search steps:")
    lo, hi = 0, len(commits) - 1
    step = 1
    while lo < hi:
        mid = (lo + hi) // 2
        sha, msg, good = commits[mid]
        result = "good" if good else "bad"
        print(f"    Step {step}: Testing {sha} ({msg}) → {result}")
        if good:
            lo = mid + 1
        else:
            hi = mid
        step += 1

    sha, msg, _ = commits[lo]
    print(f"\n  Result: {sha} ({msg}) is the first bad commit!")
    print(f"  Found in {step - 1} steps (log2({len(commits)}) ≈ {len(commits).bit_length() - 1})")
    print()

    print("  Commands:")
    print("    git bisect start")
    print("    git bisect bad HEAD")
    print("    git bisect good abc001")
    print("    git bisect run python -m pytest tests/")
    print("    git bisect reset")
    print()


def git_log_search():
    """Show git log search patterns for debugging."""
    print("=== git log Search Patterns ===")
    commands = [
        ("git log --grep='fix' --oneline", "Find commits mentioning 'fix'"),
        ("git log -S 'calculate' --oneline", "Find commits changing 'calculate'"),
        ("git log -G 'def calc' --oneline", "Find commits matching regex"),
        ("git log --since='1 week ago'", "Commits from last week"),
        ("git log --author='Bob' -- src/", "Bob's changes to src/"),
        ("git log -p -- file.py", "Full diff history of a file"),
    ]
    for cmd, desc in commands:
        print(f"  {cmd}")
        print(f"    → {desc}")
    print()


def commit_message_examples():
    """Show good vs bad commit messages."""
    print("=== Commit Message Quality ===")

    print("  BAD messages (useless for debugging):")
    bad = ["fix bug", "update code", "stuff", "wip", "asdf"]
    for msg in bad:
        print(f"    - {msg}")

    print("\n  GOOD messages (enable git log/bisect/grep):")
    good = [
        "fix: correct off-by-one in pagination calculation",
        "fix: handle None return from find_user()",
        "perf: replace bubble_sort with sorted() in pipeline",
        "refactor: extract discount calculation to helper function",
    ]
    for msg in good:
        print(f"    - {msg}")
    print()


def debugging_workflow_with_git():
    """Show a complete git-based debugging workflow."""
    print("=== Git Debugging Workflow ===")
    steps = [
        "1. Bug reported: 'Discount wrong for large orders'",
        "2. git log --oneline -10 -- src/pricing.py",
        "3. Spot suspicious commit: 'Optimize discount calc'",
        "4. git bisect start && git bisect bad && git bisect good v2.0",
        "5. git bisect run python test_discount.py",
        "6. Found: commit abc123 changed threshold from 1000 to 100",
        "7. git diff abc123~1 abc123 -- src/pricing.py",
        "8. Fix the bug, write a test, commit with good message",
    ]
    for step in steps:
        print(f"  {step}")
    print()


if __name__ == "__main__":
    git_diff_examples()
    git_blame_examples()
    git_bisect_demo()
    git_log_search()
    commit_message_examples()
    debugging_workflow_with_git()
