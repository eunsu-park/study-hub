"""
Exercise 10: Profiling Basics

Practice timing, profiling, and optimizing Python code.
"""
import timeit
import time


def compare_string_building():
    """Compare string concatenation vs join.

    Time both approaches for building a string of numbers 0-9999
    separated by commas. Return which is faster and by how much.

    Returns:
        tuple: (concat_time, join_time, speedup_factor)
    """
    # TODO: Implement both approaches and time them with timeit
    pass


def find_bottleneck(data):
    """Find and fix the performance bottleneck.

    This function processes data through multiple steps.
    Profile it to find the bottleneck, then optimize.

    Steps:
    1. Remove duplicates
    2. Sort the data
    3. Find the median

    Args:
        data: A list of integers.

    Returns:
        float: The median value.
    """
    # TODO: Profile this function, find the bottleneck, and optimize

    # Step 1: Remove duplicates (SLOW: O(n^2) approach)
    unique = []
    for item in data:
        found = False
        for existing in unique:
            if item == existing:
                found = True
                break
        if not found:
            unique.append(item)

    # Step 2: Sort
    unique.sort()

    # Step 3: Find median
    n = len(unique)
    if n == 0:
        return 0
    if n % 2 == 0:
        return (unique[n // 2 - 1] + unique[n // 2]) / 2
    return unique[n // 2]


def measure_lookup_performance():
    """Compare list vs set vs dict lookup performance.

    Create a collection of 100,000 items and measure the time
    to check if an item exists.

    Returns:
        dict: Mapping of structure name to lookup time.
    """
    # TODO: Compare lookup times for list, set, and dict
    pass


def optimize_with_profiling(records):
    """Optimize this function based on profiling results.

    This function calculates the total value of records
    grouped by category. It's currently slow due to
    inefficient lookups.

    Args:
        records: A list of dicts with "category" and "value".

    Returns:
        dict: Category to total value mapping.
    """
    # TODO: Optimize this function
    # Current slow approach: linear search for category
    categories = []
    totals = []

    for record in records:
        cat = record["category"]
        # Find category index (O(n) each time!)
        idx = -1
        for i, c in enumerate(categories):
            if c == cat:
                idx = i
                break

        if idx >= 0:
            totals[idx] += record["value"]
        else:
            categories.append(cat)
            totals.append(record["value"])

    return dict(zip(categories, totals))


if __name__ == "__main__":
    # Test compare_string_building
    result = compare_string_building()
    if result:
        concat_t, join_t, speedup = result
        print(f"compare_string_building: concat={concat_t:.4f}s, "
              f"join={join_t:.4f}s, speedup={speedup:.1f}x")
    else:
        print("compare_string_building: Not implemented yet")

    # Test find_bottleneck
    import random
    random.seed(42)
    data = [random.randint(0, 1000) for _ in range(5000)]

    start = time.perf_counter()
    result = find_bottleneck(data)
    elapsed = time.perf_counter() - start
    print(f"find_bottleneck: median={result}, time={elapsed:.4f}s")
    assert isinstance(result, (int, float))

    # Test measure_lookup_performance
    result = measure_lookup_performance()
    if result:
        for name, t in result.items():
            print(f"  {name}: {t:.6f}s")
    else:
        print("measure_lookup_performance: Not implemented yet")

    # Test optimize_with_profiling
    records = [
        {"category": f"cat_{i % 50}", "value": i}
        for i in range(10000)
    ]
    start = time.perf_counter()
    result = optimize_with_profiling(records)
    elapsed = time.perf_counter() - start
    print(f"optimize_with_profiling: {len(result)} categories, time={elapsed:.4f}s")
    assert len(result) == 50
