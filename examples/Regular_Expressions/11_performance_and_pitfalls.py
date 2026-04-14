"""
11 Performance and Pitfalls
============================
Demonstrates catastrophic backtracking detection, safe pattern
alternatives, and performance comparison techniques.
"""

import re
import time


def backtracking_demo():
    """Show how backtracking time grows with input size."""
    pattern = r'^(a+)+$'
    print("Testing dangerous pattern: (a+)+$")
    for n in [5, 10, 15, 18, 20]:
        text = 'a' * n + '!'
        start = time.time()
        re.search(pattern, text)
        elapsed = time.time() - start
        warn = " *** SLOW" if elapsed > 0.1 else ""
        print(f"  n={n:2d}: {elapsed:.4f}s{warn}")
        if elapsed > 2.0:
            print("  Stopping -- too slow!")
            break


def safe_alternatives():
    """Show safe pattern rewrites."""
    text = '<div class="name">value</div>'

    # Greedy (potentially slow on complex input)
    t1 = time.time()
    r1 = re.search(r'<.*>', text).group()
    t1 = time.time() - t1

    # Lazy
    t2 = time.time()
    r2 = re.search(r'<.*?>', text).group()
    t2 = time.time() - t2

    # Negated class (fastest)
    t3 = time.time()
    r3 = re.search(r'<[^>]+>', text).group()
    t3 = time.time() - t3

    print(f"Greedy <.*>:      {r1}")
    print(f"Lazy <.*?>:       {r2}")
    print(f"Negated <[^>]+>:  {r3}")


def compile_benchmark():
    """Compare compiled vs uncompiled pattern performance."""
    pattern_str = r'\b\w{3,}\b'
    compiled = re.compile(pattern_str)
    text = "The quick brown fox jumps over the lazy dog " * 100

    n = 5000

    start = time.time()
    for _ in range(n):
        compiled.findall(text)
    compiled_time = time.time() - start

    start = time.time()
    for _ in range(n):
        re.findall(pattern_str, text)
    string_time = time.time() - start

    print(f"Compiled: {compiled_time:.4f}s")
    print(f"String:   {string_time:.4f}s")
    print(f"Speedup:  {string_time/compiled_time:.2f}x")


def string_vs_regex():
    """Compare string methods and regex for simple operations."""
    text = "Hello World " * 1000
    n = 10000

    # 'in' operator vs re.search
    start = time.time()
    for _ in range(n):
        "Hello" in text
    str_time = time.time() - start

    start = time.time()
    for _ in range(n):
        re.search(r'Hello', text)
    re_time = time.time() - start

    print(f"'in' operator: {str_time:.4f}s")
    print(f"re.search():   {re_time:.4f}s")
    print(f"String method is {re_time/str_time:.1f}x faster")


def common_pitfalls():
    """Demonstrate and fix common regex mistakes."""
    # Pitfall 1: forgetting re.escape
    user_input = "price (USD)"
    safe = re.escape(user_input)
    print(f"Escaped user input: {safe}")

    # Pitfall 2: . doesn't match newline
    text = "line1\nline2"
    print(f"Without DOTALL: {re.search(r'line1.*line2', text)}")
    print(f"With DOTALL:    {re.search(r'line1.*line2', text, re.S)}")

    # Pitfall 3: match vs search
    print(f"match('file', 'Error: file'): {re.match(r'file', 'Error: file')}")
    print(f"search('file', 'Error: file'): {re.search(r'file', 'Error: file')}")

    # Pitfall 4: unintended capture
    print(f"findall with group:    {re.findall(r'colo(u?)r', 'color colour')}")
    print(f"findall without group: {re.findall(r'colou?r', 'color colour')}")


if __name__ == "__main__":
    sections = [
        ("Backtracking Demo", backtracking_demo),
        ("Safe Alternatives", safe_alternatives),
        ("Compile Benchmark", compile_benchmark),
        ("String vs Regex", string_vs_regex),
        ("Common Pitfalls", common_pitfalls),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
