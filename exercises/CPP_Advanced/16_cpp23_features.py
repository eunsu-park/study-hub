"""
Exercises for Lesson 21: C++23 Features
Topic: CPP

Solutions to practice problems covering std::expected patterns,
deducing this, std::mdspan concepts, generator patterns,
and range adaptor compositions — all demonstrated in Python
equivalents to illustrate the concepts.
"""

import math
from collections import namedtuple


# ============================================================
# Exercise 1: Error Pipeline (std::expected pattern)
# ============================================================
def exercise_1():
    """
    Build a data processing pipeline using Result type (Python equivalent
    of std::expected). Chain operations with and_then/transform.
    """
    print("=== Exercise 1: Error Pipeline (std::expected) ===\n")

    class Result:
        """Python equivalent of std::expected<T, E>."""
        def __init__(self, value=None, error=None):
            self._value = value
            self._error = error
            self._has_value = error is None

        @staticmethod
        def ok(value):
            return Result(value=value)

        @staticmethod
        def err(error):
            return Result(error=error)

        def has_value(self):
            return self._has_value

        def value(self):
            if not self._has_value:
                raise RuntimeError(f"No value: {self._error}")
            return self._value

        def error(self):
            return self._error

        def value_or(self, default):
            return self._value if self._has_value else default

        def and_then(self, fn):
            """Chain: if has value, apply fn(value) → Result."""
            if self._has_value:
                return fn(self._value)
            return self

        def transform(self, fn):
            """Map: if has value, wrap fn(value) in Result.ok."""
            if self._has_value:
                return Result.ok(fn(self._value))
            return self

        def or_else(self, fn):
            """If error, try recovery with fn(error) → Result."""
            if not self._has_value:
                return fn(self._error)
            return self

    # Pipeline: read_file → parse_json → validate → apply
    def read_file(path):
        if path == "missing.json":
            return Result.err("FileNotFound")
        return Result.ok('{"name": "MyApp", "version": 2, "port": 8080}')

    def parse_json(content):
        try:
            # Simplified JSON parsing
            if "{" not in content:
                return Result.err("InvalidJSON")
            pairs = content.strip("{}").split(",")
            data = {}
            for pair in pairs:
                k, v = pair.split(":")
                k = k.strip().strip('"')
                v = v.strip().strip('"')
                try:
                    v = int(v)
                except ValueError:
                    pass
                data[k] = v
            return Result.ok(data)
        except Exception as e:
            return Result.err(f"ParseError: {e}")

    def validate(config):
        if "name" not in config:
            return Result.err("MissingField: name")
        if "port" not in config:
            return Result.err("MissingField: port")
        return Result.ok(config)

    # Test the pipeline
    test_cases = ["config.json", "missing.json", "bad.json"]

    for path in test_cases:
        if path == "bad.json":
            result = Result.ok("not valid json {{{")
        else:
            result = read_file(path)

        final = (result
                 .and_then(parse_json)
                 .and_then(validate)
                 .transform(lambda c: f"App '{c.get('name', '?')}' on port {c.get('port', '?')}"))

        if final.has_value():
            print(f"  {path:>15}: OK → {final.value()}")
        else:
            print(f"  {path:>15}: Error → {final.error()}")

    # value_or demonstration
    port = (read_file("missing.json")
            .and_then(parse_json)
            .transform(lambda c: c.get("port", 80))
            .value_or(8080))
    print(f"\n  Default port: {port}")
    print(f"\n  C++ equivalent: read_file(path).and_then(parse).and_then(validate)")
    print()


# ============================================================
# Exercise 2: Deducing this / CRTP Replacement
# ============================================================
def exercise_2():
    """
    Demonstrate the CRTP pattern and its simplification with
    deducing this (using Python's method resolution order).
    """
    print("=== Exercise 2: CRTP Replacement ===\n")

    # Old CRTP pattern (Python equivalent)
    class PrintableCRTP:
        """Base class that calls derived to_string()."""
        def print_info(self):
            print(f"    {self.to_string()}")

    class Point2D(PrintableCRTP):
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def to_string(self):
            return f"Point2D({self.x}, {self.y})"

    class Color(PrintableCRTP):
        def __init__(self, r, g, b):
            self.r, self.g, self.b = r, g, b
        def to_string(self):
            return f"Color({self.r}, {self.g}, {self.b})"

    # C++23 deducing this equivalent:
    # In C++23, the base class doesn't need a template parameter.
    # The explicit object parameter 'this' deduces the derived type.

    print("  CRTP pattern (both Python and C++ achieve polymorphism):")
    Point2D(1.5, 2.7).print_info()
    Color(255, 128, 0).print_info()

    # Recursive lambda (C++23 deducing this enables this directly)
    print("\n  Recursive lambda (C++23: auto fib = [](this auto self, int n)...):")
    fib = lambda self, n: n if n <= 1 else self(self, n - 1) + self(self, n - 2)
    for n in [5, 10, 15]:
        print(f"    fib({n}) = {fib(fib, n)}")

    # Value-category aware member function
    print("\n  Value-category aware (C++23 deducing this):")
    print("    template<typename Self>")
    print("    auto&& name(this Self&& self) { return fwd<Self>(self).name_; }")
    print("    // One function handles: lvalue→ref, rvalue→moved, const→const ref")
    print()


# ============================================================
# Exercise 3: Multidimensional Array View (std::mdspan)
# ============================================================
def exercise_3():
    """
    Implement an mdspan-like multidimensional view over flat arrays.
    Demonstrate matrix operations without copying.
    """
    print("=== Exercise 3: Multidimensional View (mdspan) ===\n")

    import numpy as np

    class MDSpan:
        """Python equivalent of std::mdspan — non-owning multidim view."""

        def __init__(self, data, shape, layout='row_major'):
            self.data = data  # underlying flat array (not owned)
            self.shape = shape
            self.layout = layout
            self._compute_strides()

        def _compute_strides(self):
            if self.layout == 'row_major':
                self.strides = []
                stride = 1
                for s in reversed(self.shape):
                    self.strides.insert(0, stride)
                    stride *= s
            else:  # column_major
                self.strides = []
                stride = 1
                for s in self.shape:
                    self.strides.append(stride)
                    stride *= s

        def __getitem__(self, indices):
            offset = sum(i * s for i, s in zip(indices, self.strides))
            return self.data[offset]

        def __setitem__(self, indices, value):
            offset = sum(i * s for i, s in zip(indices, self.strides))
            self.data[offset] = value

        def extent(self, dim):
            return self.shape[dim]

    # Create 3×4 matrix view over flat storage
    flat_data = list(range(1, 13))  # [1, 2, ..., 12]
    mat = MDSpan(flat_data, (3, 4), layout='row_major')

    print("  3×4 row-major matrix:")
    for i in range(mat.extent(0)):
        row = [mat[i, j] for j in range(mat.extent(1))]
        print(f"    {row}")

    # Column-major view over SAME data
    mat_col = MDSpan(flat_data, (3, 4), layout='col_major')
    print("\n  Same data, column-major view:")
    for i in range(mat_col.extent(0)):
        row = [mat_col[i, j] for j in range(mat_col.extent(1))]
        print(f"    {row}")

    # Matrix-vector multiply using mdspan
    vec = [1, 2, 3, 4]
    result = [0] * 3
    for i in range(mat.extent(0)):
        for j in range(mat.extent(1)):
            result[i] += mat[i, j] * vec[j]
    print(f"\n  Mat × [1,2,3,4] = {result}")

    # Zero-copy: modifying flat_data changes the view
    flat_data[0] = 100
    print(f"\n  After flat_data[0] = 100:")
    print(f"    mat[0,0] = {mat[0, 0]} (zero-copy view)")

    print(f"\n  C++ std::mdspan provides this for C arrays, GPU buffers, etc.")
    print()


# ============================================================
# Exercise 4: Lazy Generator (std::generator)
# ============================================================
def exercise_4():
    """
    Implement lazy generators (Python equivalents of std::generator).
    """
    print("=== Exercise 4: Lazy Generators ===\n")

    # 1. Infinite prime sequence
    def primes():
        """Infinite prime number generator."""
        yield 2
        n = 3
        while True:
            is_prime = True
            for d in range(2, int(n ** 0.5) + 1):
                if n % d == 0:
                    is_prime = False
                    break
            if is_prime:
                yield n
            n += 2

    print("  First 15 primes:", end=" ")
    gen = primes()
    for _ in range(15):
        print(next(gen), end=" ")
    print()

    # 2. Flatten nested containers
    def flatten(nested):
        """Flatten arbitrarily nested iterables."""
        for item in nested:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    nested = [[1, 2], [3, [4, 5]], [6], [[7, 8], 9]]
    print(f"  Flatten {nested}:")
    print(f"    → {list(flatten(nested))}")

    # 3. Interleave two generators
    def interleave(gen1, gen2):
        """Alternate between two generators."""
        it1, it2 = iter(gen1), iter(gen2)
        while True:
            try:
                yield next(it1)
            except StopIteration:
                yield from it2
                return
            try:
                yield next(it2)
            except StopIteration:
                yield from it1
                return

    a = range(1, 6)  # [1, 2, 3, 4, 5]
    b = range(10, 14)  # [10, 11, 12, 13]
    print(f"  Interleave [1..5] and [10..13]:")
    print(f"    → {list(interleave(a, b))}")

    # 4. Fibonacci generator (std::generator equivalent)
    def fibonacci():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b

    from itertools import islice
    fibs = list(islice(fibonacci(), 12))
    print(f"  First 12 Fibonacci: {fibs}")

    print(f"\n  C++23: std::generator<T> uses co_yield for lazy sequences.")
    print(f"  Python generators map directly to this concept.")
    print()


# ============================================================
# Exercise 5: Range Pipeline (C++23 views)
# ============================================================
def exercise_5():
    """
    Implement C++23-style range pipelines using Python functional tools.
    """
    print("=== Exercise 5: Range Pipelines ===\n")

    from itertools import islice

    words = ["hello", "to", "the", "wonderful", "world", "of", "C++",
             "ranges", "and", "views"]

    # Pipeline: enumerate → filter(len>3) → chunk(2) → format
    print(f"  Input: {words}")
    print(f"  Pipeline: enumerate | filter(len>3) | chunk(2) | format\n")

    # Step 1: enumerate
    enumerated = list(enumerate(words))
    print(f"  After enumerate: {enumerated[:5]}...")

    # Step 2: filter by length > 3
    filtered = [(i, w) for i, w in enumerated if len(w) > 3]
    print(f"  After filter(len>3): {filtered}")

    # Step 3: chunk into groups of 2
    def chunk(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                break
            yield batch

    chunked = list(chunk(filtered, 2))
    print(f"  After chunk(2): {chunked}")

    # Step 4: format each chunk
    formatted = []
    for grp in chunked:
        parts = [f"[{i}] {w}" for i, w in grp]
        formatted.append(", ".join(parts))
    print(f"  Formatted: {formatted}")

    # C++23 equivalent
    print(f"\n  C++23 equivalent:")
    print(f"    words | views::enumerate")
    print(f"         | views::filter([](auto& p) {{ return p.second.size() > 3; }})")
    print(f"         | views::chunk(2)")
    print(f"         | views::transform(format_chunk)")

    # Bonus: sliding window
    print(f"\n  Sliding window (views::slide):")
    nums = [1, 2, 3, 4, 5, 6]

    def slide(seq, window):
        for i in range(len(seq) - window + 1):
            yield seq[i:i + window]

    for w in slide(nums, 3):
        print(f"    {w}")

    # Bonus: zip
    print(f"\n  Zip (views::zip):")
    names = ["Alice", "Bob", "Charlie"]
    scores = [95, 87, 92]
    for name, score in zip(names, scores):
        print(f"    {name}: {score}")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
