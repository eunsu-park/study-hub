"""
05 Debugging Strategy
=====================
Demonstrates systematic debugging strategies: scientific method,
binary search, minimal reproducible examples, and working backward.
"""


def scientific_method_demo():
    """Demonstrate the scientific debugging method."""
    print("=== Scientific Method Demo ===")

    # Buggy function: returns 0 for valid input
    def sum_positive(numbers):
        total = 0
        for n in numbers:
            if n > 0:
                total += n
            else:
                return 0  # BUG: returns instead of skipping
        return total

    data = [1, -2, 3, 4, -1, 5]
    result = sum_positive(data)
    print(f"sum_positive({data}) = {result}")
    print(f"Expected: {sum(n for n in data if n > 0)}")
    print()

    # Scientific method walkthrough
    print("Scientific method:")
    print("  1. OBSERVE:     Returns 0 instead of 13")
    print("  2. HYPOTHESIS:  Maybe the function stops early")
    print("  3. PREDICT:     If it stops early, adding a print inside")
    print("                  the loop should show fewer iterations")
    print("  4. TEST:")

    total = 0
    for i, n in enumerate(data):
        print(f"     iteration {i}: n={n}", end="")
        if n > 0:
            total += n
            print(f" → total={total}")
        else:
            print(f" → return 0 (BUG: should skip, not return)")
            break
    print()
    print("  5. ROOT CAUSE: 'return 0' should be 'continue'")

    # Fixed version
    def sum_positive_fixed(numbers):
        total = 0
        for n in numbers:
            if n > 0:
                total += n
        return total

    print(f"  6. FIX: sum_positive_fixed({data}) = {sum_positive_fixed(data)}")
    print()


def binary_search_debugging():
    """Demonstrate binary search debugging on a pipeline."""
    print("=== Binary Search Debugging ===")

    def step_a(data):
        return [x.strip() for x in data]

    def step_b(data):
        return [x.lower() for x in data]

    def step_c(data):
        return [x.replace(" ", "") for x in data]  # BUG: removes spaces

    def step_d(data):
        return [x.title() for x in data]

    data = ["  Alice Smith  ", "  Bob Jones  ", "  Charlie Brown  "]
    expected = ["Alice Smith", "Bob Jones", "Charlie Brown"]

    # Run full pipeline
    result = step_d(step_c(step_b(step_a(data))))
    print(f"Input:    {data}")
    print(f"Expected: {expected}")
    print(f"Got:      {result}")
    print()

    # Binary search: check midpoint
    print("Binary search debugging:")
    mid = step_b(step_a(data))
    print(f"  After step A+B: {mid}")
    print(f"  → Looks correct (names are lowercase but intact)")

    mid2 = step_c(mid)
    print(f"  After step C:   {mid2}")
    print(f"  → BUG FOUND! step_c removes spaces from names")
    print(f"  → step_c should only remove leading/trailing, not internal spaces")
    print()


def minimal_reproducible_example():
    """Show how to create a minimal reproducible example."""
    print("=== Minimal Reproducible Example ===")

    # Original: complex function with many parameters
    def process_invoice(customer, items, tax_rate, discount_code,
                        shipping_method, notes):
        subtotal = sum(i["price"] * i["qty"] for i in items)
        discount = 0
        if discount_code == "SAVE10":
            discount = subtotal * 0.1
        elif discount_code == "SAVE20":
            discount = subtotal * 0.20
        elif discount_code and discount_code.endswith("%"):
            pct = int(discount_code[:-1])  # BUG: "10%" → "10" → 10
            discount = subtotal * pct  # Missing /100!
        total = (subtotal - discount) * (1 + tax_rate)
        return total

    # Full call (hard to debug)
    total = process_invoice(
        customer={"name": "Alice", "id": 123},
        items=[{"name": "Widget", "price": 100, "qty": 2}],
        tax_rate=0.1,
        discount_code="10%",
        shipping_method="express",
        notes="Rush order",
    )
    print(f"Full call result: ${total:.2f} (expected ~$198, got wrong value)")

    # MRE (10 lines, same bug)
    print("\nMinimal Reproducible Example:")
    print("  def apply_discount(subtotal, code):")
    print("      pct = int(code[:-1])  # '10%' → 10")
    print("      return subtotal * pct  # BUG: 200 * 10 = 2000!")
    print("      # FIX: return subtotal * pct / 100  # 200 * 0.1 = 20")

    # Fixed
    def apply_discount_fixed(subtotal, code):
        pct = int(code[:-1])
        return subtotal * pct / 100

    print(f"\n  apply_discount(200, '10%') = {apply_discount_fixed(200, '10%')}")
    print()


def work_backward_demo():
    """Demonstrate working backward from wrong output."""
    print("=== Work Backward from Symptom ===")

    def generate_report(sales):
        filtered = [s for s in sales if s["date"].startswith("2024-01")]
        amounts = [s["amount"] for s in filtered]
        total = sum(amounts)
        count = len(amounts)
        avg = total / count if count > 0 else 0
        return {"total": total, "count": count, "avg": avg}

    sales = [
        {"date": "2024-01-05", "amount": 100},
        {"date": "2024-01-15", "amount": 200},
        {"date": "2024-02-01", "amount": 300},  # Different month
        {"date": "2024-1-20", "amount": 150},    # BUG: missing leading zero
    ]

    report = generate_report(sales)
    print(f"Sales data: {len(sales)} records")
    print(f"Report: {report}")
    print(f"Expected total: 450 (100+200+150), got: {report['total']}")
    print()
    print("Working backward:")
    print(f"  1. total={report['total']} → wrong")
    print(f"  2. amounts from filtered sales → check filter")
    filtered = [s for s in sales if s["date"].startswith("2024-01")]
    print(f"  3. filtered = {filtered}")
    print(f"  4. Missing: {{'date': '2024-1-20'}} → doesn't match '2024-01'!")
    print(f"  5. Root cause: inconsistent date format (missing leading zero)")
    print()


if __name__ == "__main__":
    scientific_method_demo()
    binary_search_debugging()
    minimal_reproducible_example()
    work_backward_demo()
