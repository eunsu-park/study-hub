"""
14 Choosing the Right Data Structure
====================================
Demonstrates practical examples of choosing the right
data structure for different problem patterns.
"""

import heapq
import time
from collections import Counter, deque, OrderedDict


def demo_frequency_counting():
    """Pattern: Frequency counting -> Counter."""
    items = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    counts = Counter(items)
    print(f"Items: {items}")
    print(f"Top 3: {counts.most_common(3)}")


def demo_dedup_preserve_order():
    """Pattern: Dedup preserving order -> dict."""
    items = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    unique = list(dict.fromkeys(items))
    print(f"Input:  {items}")
    print(f"Unique: {unique}")


def demo_top_k():
    """Pattern: Top-K -> heap."""
    data = [10, 4, 3, 50, 23, 90, 1, 77, 45]
    print(f"Data: {data}")
    print(f"Top 3:    {heapq.nlargest(3, data)}")
    print(f"Bottom 3: {heapq.nsmallest(3, data)}")


def demo_sliding_window():
    """Pattern: Sliding window -> monotonic deque."""
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    dq = deque(); result = []
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1: dq.popleft()
        while dq and nums[dq[-1]] < nums[i]: dq.pop()
        dq.append(i)
        if i >= k - 1: result.append(nums[dq[0]])
    print(f"Sliding window max (k={k}): {result}")


def demo_wrong_vs_right():
    """Show list vs set lookup performance."""
    n = 50000
    data = list(range(n))
    queries = list(range(n - 100, n))

    start = time.perf_counter()
    sum(1 for q in queries if q in data)
    list_time = time.perf_counter() - start

    data_set = set(data)
    start = time.perf_counter()
    sum(1 for q in queries if q in data_set)
    set_time = time.perf_counter() - start

    print(f"List lookup: {list_time:.4f}s")
    print(f"Set lookup:  {set_time:.6f}s")
    if set_time > 0:
        print(f"Speedup: {list_time / set_time:.0f}x")


if __name__ == "__main__":
    for title, func in [
        ("Frequency Counting", demo_frequency_counting),
        ("Dedup Preserve Order", demo_dedup_preserve_order),
        ("Top-K Elements", demo_top_k),
        ("Sliding Window", demo_sliding_window),
        ("Wrong vs Right DS", demo_wrong_vs_right),
    ]:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
