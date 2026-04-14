"""
02 Print Debugging
==================
Demonstrates strategic print debugging techniques including
labeled output, repr(), data flow tracing, and debug flags.
"""
import sys


def strategic_print_demo():
    """Show strategic print placement at decision points."""
    print("=== Strategic Print Debugging ===")

    def process_orders(orders):
        print(f"[process_orders] Received {len(orders)} orders")
        results = []
        for i, order in enumerate(orders):
            total = sum(item["price"] * item["qty"] for item in order["items"])
            print(f"  Order {i}: total={total}, items={len(order['items'])}")
            results.append(total)
        print(f"[process_orders] Results: {results}")
        return results

    orders = [
        {"items": [{"price": 10, "qty": 2}, {"price": 5, "qty": 1}]},
        {"items": [{"price": 100, "qty": 1}]},
    ]
    process_orders(orders)
    print()


def repr_vs_str_demo():
    """Show why repr() is essential for debugging."""
    print("=== repr() vs str() Demo ===")
    values = {
        "normal": "hello",
        "trailing_space": "hello ",
        "trailing_tab": "hello\t",
        "empty": "",
        "none": None,
        "zero": 0,
        "false": False,
    }
    print(f"{'Name':<20} {'str()':<15} {'repr()':<20}")
    print("-" * 55)
    for name, val in values.items():
        print(f"{name:<20} {str(val):<15} {repr(val):<20}")
    print()


def fstring_debug_demo():
    """Show f-string debugging tricks (Python 3.8+)."""
    print("=== f-string Debugging (Python 3.8+) ===")
    x = 42
    data = [1, 2, 3]
    name = "Alice"
    print(f"{x = }")
    print(f"{len(data) = }")
    print(f"{name.upper() = }")
    print(f"{x * 2 + 1 = }")
    print()


def data_flow_tracing():
    """Trace data through a multi-step transformation pipeline."""
    print("=== Data Flow Tracing ===")

    def clean_username(raw_input):
        print(f"[clean] step 0 (raw):     {raw_input!r}")
        stripped = raw_input.strip()
        print(f"[clean] step 1 (strip):   {stripped!r}")
        lowered = stripped.lower()
        print(f"[clean] step 2 (lower):   {lowered!r}")
        cleaned = "".join(c for c in lowered if c.isalnum() or c == "_")
        print(f"[clean] step 3 (filter):  {cleaned!r}")
        return cleaned

    result = clean_username("  Hello_World! @#$ ")
    print(f"Final result: {result!r}")
    print()


def function_entry_exit():
    """Demonstrate function entry/exit logging pattern."""
    print("=== Function Entry/Exit Pattern ===")

    def calculate_tax(income, deductions):
        print(f">>> calculate_tax(income={income}, deductions={deductions})")
        taxable = income - deductions
        if taxable <= 0:
            print(f"<<< calculate_tax -> 0 (no taxable income)")
            return 0
        rate = 0.3 if taxable > 50000 else 0.2
        tax = taxable * rate
        print(f"<<< calculate_tax -> {tax} (rate={rate}, taxable={taxable})")
        return tax

    calculate_tax(80000, 20000)
    calculate_tax(10000, 15000)
    print()


def debug_flag_demo():
    """Show how to use a debug flag to control output."""
    print("=== Debug Flag Pattern ===")

    class DebugPrinter:
        def __init__(self, enabled=True):
            self.enabled = enabled

        def __call__(self, *args, **kwargs):
            if self.enabled:
                print("[DEBUG]", *args, **kwargs)

    debug = DebugPrinter(enabled=True)

    def calculate(x, y):
        debug(f"calculate({x}, {y})")
        result = x + y
        debug(f"result = {result}")
        return result

    print("With debug enabled:")
    calculate(3, 4)
    print()

    debug.enabled = False
    print("With debug disabled:")
    calculate(3, 4)
    print("(No debug output)")
    print()


def stderr_demo():
    """Show separating debug output from program output."""
    print("=== stderr Separation Demo ===")
    print("Program output goes to stdout")
    print("[DEBUG] Debug output goes to stderr", file=sys.stderr)
    print("Use: python script.py > output.txt  (debug still visible)")
    print()


def type_debugging():
    """Show type inspection for debugging."""
    print("=== Type Debugging ===")

    def debug_value(name, value):
        print(f"  {name}: value={value!r}, type={type(value).__name__}")

    debug_value("count_str", "5")
    debug_value("count_int", 5)
    debug_value("items_none", None)
    debug_value("flag", True)
    debug_value("scores", [90, 85, 78])
    print()


if __name__ == "__main__":
    strategic_print_demo()
    repr_vs_str_demo()
    fstring_debug_demo()
    data_flow_tracing()
    function_entry_exit()
    debug_flag_demo()
    stderr_demo()
    type_debugging()
