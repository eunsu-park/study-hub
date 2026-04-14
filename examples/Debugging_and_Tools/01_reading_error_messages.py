"""
01 Reading Error Messages
=========================
Demonstrates how to read and interpret Python error messages,
tracebacks, and common exception types.
"""


def demonstrate_traceback():
    """Show how a traceback is generated through nested calls."""
    def divide(a, b):
        return a / b

    def calculate():
        return divide(10, 0)

    try:
        calculate()
    except ZeroDivisionError:
        import traceback
        print("=== Traceback Demo ===")
        traceback.print_exc()
        print()


def demonstrate_common_exceptions():
    """Show the most common Python exceptions with examples."""

    errors = []

    # 1. NameError
    try:
        print(undefined_variable)
    except NameError as e:
        errors.append(("NameError", str(e)))

    # 2. TypeError
    try:
        "age: " + 25
    except TypeError as e:
        errors.append(("TypeError", str(e)))

    # 3. ValueError
    try:
        int("hello")
    except ValueError as e:
        errors.append(("ValueError", str(e)))

    # 4. IndexError
    try:
        items = [1, 2, 3]
        _ = items[10]
    except IndexError as e:
        errors.append(("IndexError", str(e)))

    # 5. KeyError
    try:
        data = {"name": "Alice"}
        _ = data["age"]
    except KeyError as e:
        errors.append(("KeyError", str(e)))

    # 6. AttributeError
    try:
        x = 42
        x.append(1)
    except AttributeError as e:
        errors.append(("AttributeError", str(e)))

    # 7. ZeroDivisionError
    try:
        _ = 100 / 0
    except ZeroDivisionError as e:
        errors.append(("ZeroDivisionError", str(e)))

    # 8. FileNotFoundError
    try:
        with open("__nonexistent_file__.txt") as f:
            f.read()
    except FileNotFoundError as e:
        errors.append(("FileNotFoundError", str(e)))

    print("=== Common Python Exceptions ===")
    for error_type, message in errors:
        print(f"  {error_type:25s} → {message}")
    print()


def demonstrate_chained_exceptions():
    """Show how exception chaining works with 'from'."""

    def load_config(path):
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError as e:
            raise RuntimeError(f"Config missing: {path}") from e

    print("=== Chained Exception Demo ===")
    try:
        load_config("nonexistent_config.yaml")
    except RuntimeError:
        import traceback
        traceback.print_exc()
    print()


def demonstrate_error_classification():
    """Classify errors into syntax, runtime, and logical."""
    print("=== Error Classification ===")

    # Runtime error example
    print("Runtime error (caught):")
    try:
        result = int("not_a_number")
    except ValueError as e:
        print(f"  ValueError: {e}")

    # Logical error example (no exception raised, wrong result)
    print("\nLogical error (no exception, wrong result):")

    def buggy_average(numbers):
        total = 0
        for n in numbers:
            total += n
        return total / len(numbers) + 1  # Bug: extra +1

    data = [10, 20, 30]
    result = buggy_average(data)
    expected = 20.0
    print(f"  buggy_average({data}) = {result} (expected {expected})")
    print(f"  Bug: function adds +1 to the result")
    print()


def demonstrate_reading_strategy():
    """Show the bottom-to-top reading strategy for tracebacks."""
    print("=== Traceback Reading Strategy ===")
    print("Given this traceback:")
    print("""
    Traceback (most recent call last):
      File "main.py", line 12, in <module>        ← Step 4: Entry point
        app.run()
      File "app.py", line 45, in run               ← Step 3: Caller
        result = self.processor.process(data)
      File "processor.py", line 23, in process      ← Step 2: Crash site
        return int(item["count"])
    ValueError: invalid literal for int(): 'three'  ← Step 1: START HERE
    """)
    print("Reading order:")
    print("  1. Bottom: ValueError - tried to convert 'three' to int")
    print("  2. Crash site: processor.py line 23, int(item['count'])")
    print("  3. Caller: app.py line 45 called process(data)")
    print("  4. Entry: main.py line 12 called app.run()")
    print()


if __name__ == "__main__":
    demonstrate_traceback()
    demonstrate_common_exceptions()
    demonstrate_chained_exceptions()
    demonstrate_error_classification()
    demonstrate_reading_strategy()
