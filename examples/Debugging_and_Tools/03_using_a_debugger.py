"""
03 Using a Debugger
===================
Demonstrates pdb usage patterns including breakpoints,
stepping, inspection, and post-mortem debugging.
"""


def basic_breakpoint_demo():
    """Show how breakpoint() pauses execution.

    Run this file normally to see the output.
    To actually use the debugger, uncomment the breakpoint() call
    and run: python -m pdb 03_using_a_debugger.py
    """
    print("=== Breakpoint Demo ===")

    def calculate_average(numbers):
        total = 0
        for n in numbers:
            total += n
        # Uncomment to try: breakpoint()
        return total / len(numbers)

    result = calculate_average([10, 20, 30])
    print(f"Average: {result}")
    print("(Uncomment breakpoint() in source to try interactive debugging)")
    print()


def debugging_walkthrough():
    """Walk through finding a bug with debugger-style inspection."""
    print("=== Debugging Walkthrough ===")

    def compute_stats_buggy(data):
        """Buggy version: missing **2 in variance calculation."""
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) for x in data) / n  # BUG: missing **2
        std_dev = variance ** 0.5
        return {"mean": mean, "std_dev": std_dev}

    def compute_stats_fixed(data):
        """Fixed version: correct variance calculation."""
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std_dev = variance ** 0.5
        return {"mean": mean, "std_dev": std_dev}

    data = [2, 4, 4, 4, 5, 5, 7, 9]
    buggy = compute_stats_buggy(data)
    fixed = compute_stats_fixed(data)

    print(f"Data: {data}")
    print(f"Buggy: mean={buggy['mean']}, std_dev={buggy['std_dev']:.4f}")
    print(f"Fixed: mean={fixed['mean']}, std_dev={fixed['std_dev']:.4f}")
    print()
    print("Debugging steps (simulated):")
    mean = sum(data) / len(data)
    deviations = [(x - mean) for x in data]
    print(f"  mean = {mean}")
    print(f"  deviations = {deviations}")
    print(f"  sum(deviations) = {sum(deviations)} → always 0 (BUG!)")
    print(f"  sum(deviations**2) = {sum(d**2 for d in deviations)} → correct")
    print()


def conditional_breakpoint_demo():
    """Demonstrate conditional breakpoint patterns."""
    print("=== Conditional Breakpoint Pattern ===")

    def process_records(records):
        results = []
        for i, record in enumerate(records):
            # In pdb: b <line>, record["status"] == "error"
            if record.get("status") == "error":
                print(f"  [STOP] Would breakpoint at record {i}: {record}")
            results.append(record.get("value", 0))
        return results

    records = [
        {"value": 10, "status": "ok"},
        {"value": 20, "status": "ok"},
        {"value": -1, "status": "error"},
        {"value": 30, "status": "ok"},
        {"value": -5, "status": "error"},
    ]
    process_records(records)
    print()


def post_mortem_demo():
    """Demonstrate post-mortem debugging pattern."""
    print("=== Post-Mortem Debugging ===")
    import traceback

    def buggy_function():
        data = {"name": "Alice", "scores": [90, 85]}
        return data["grade"]  # KeyError!

    try:
        buggy_function()
    except Exception:
        print("Exception caught! In a real session, call pdb.post_mortem()")
        print("Traceback:")
        traceback.print_exc()
        print()
        print("To use post-mortem debugging:")
        print("  import pdb; pdb.post_mortem()")
        print("  Or run: python -m pdb script.py")
    print()


def stack_navigation_demo():
    """Show how stack frames work for up/down navigation."""
    print("=== Stack Navigation Demo ===")

    def level_3(x):
        """Innermost function."""
        return x * 2

    def level_2(x):
        return level_3(x + 10)

    def level_1(x):
        return level_2(x + 5)

    result = level_1(1)
    print(f"level_1(1) = {result}")
    print()
    print("Call stack (use 'w' in pdb to see this):")
    print("  level_1(x=1)")
    print("    └── level_2(x=6)")
    print("        └── level_3(x=16) → returns 32")
    print()
    print("Use 'u' to move up, 'd' to move down in pdb")
    print()


def pdb_commands_reference():
    """Print a quick reference of essential pdb commands."""
    print("=== Essential pdb Commands ===")
    commands = [
        ("n", "next", "Step over (execute line, skip into functions)"),
        ("s", "step", "Step into (enter function calls)"),
        ("c", "continue", "Continue until next breakpoint"),
        ("r", "return", "Continue until current function returns"),
        ("p expr", "print", "Print value of expression"),
        ("pp expr", "pprint", "Pretty-print value"),
        ("l", "list", "Show source around current line"),
        ("w", "where", "Show call stack"),
        ("u", "up", "Move up one stack frame"),
        ("d", "down", "Move down one stack frame"),
        ("b N", "break", "Set breakpoint at line N"),
        ("q", "quit", "Quit debugger"),
    ]
    for cmd, name, desc in commands:
        print(f"  {cmd:<12} ({name:<10}) {desc}")
    print()


if __name__ == "__main__":
    basic_breakpoint_demo()
    debugging_walkthrough()
    conditional_breakpoint_demo()
    post_mortem_demo()
    stack_navigation_demo()
    pdb_commands_reference()
