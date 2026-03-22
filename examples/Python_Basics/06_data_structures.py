"""
06 Data Structures
==================
Demonstrates lists, tuples, dictionaries, sets, comprehensions,
namedtuple, and common operations on each.
"""

from collections import namedtuple, defaultdict, Counter


def list_operations():
    """List creation, manipulation, and common methods."""
    # Creation
    nums = [1, 2, 3, 4, 5]
    mixed = [1, "hello", 3.14, True, None]
    nested = [[1, 2], [3, 4], [5, 6]]
    from_range = list(range(0, 20, 3))
    print(f"nums:       {nums}")
    print(f"mixed:      {mixed}")
    print(f"from_range: {from_range}")

    # Indexing and slicing
    print(f"\nnums[0]    = {nums[0]}")
    print(f"nums[-1]   = {nums[-1]}")
    print(f"nums[1:4]  = {nums[1:4]}")
    print(f"nums[::2]  = {nums[::2]}")
    print(f"nums[::-1] = {nums[::-1]}")

    # Mutation
    nums.append(6)
    nums.insert(0, 0)
    nums.extend([7, 8])
    print(f"\nAfter append/insert/extend: {nums}")

    removed = nums.pop(3)
    print(f"pop(3) removed {removed}: {nums}")

    # Sorting
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"\nSorted:  {sorted(data)}")
    print(f"Reverse: {sorted(data, reverse=True)}")
    print(f"Original unchanged: {data}")

    # List as stack
    stack = []
    for item in ["a", "b", "c"]:
        stack.append(item)
    print(f"\nStack: {stack}")
    print(f"Pop: {stack.pop()}, stack now: {stack}")


def tuple_operations():
    """Tuples: immutable sequences."""
    # Creation
    point = (3, 4)
    single = (42,)           # Note the comma
    from_list = tuple([1, 2, 3])
    empty = ()

    print(f"point:     {point}")
    print(f"single:    {single} (type: {type(single).__name__})")
    print(f"not tuple: {type((42)).__name__}")  # int, not tuple!

    # Unpacking
    x, y = point
    print(f"\nUnpacked: x={x}, y={y}")

    # Extended unpacking
    first, *rest = [1, 2, 3, 4, 5]
    print(f"first={first}, rest={rest}")

    head, *middle, tail = [1, 2, 3, 4, 5]
    print(f"head={head}, middle={middle}, tail={tail}")

    # Tuples as dict keys (hashable)
    grid = {}
    grid[(0, 0)] = "origin"
    grid[(1, 0)] = "right"
    print(f"\nGrid: {grid}")

    # namedtuple
    Point = namedtuple("Point", ["x", "y"])
    p = Point(3, 4)
    print(f"\nNamedTuple: {p}")
    print(f"p.x={p.x}, p.y={p.y}")
    print(f"As dict: {p._asdict()}")


def dict_operations():
    """Dictionary creation, access patterns, and methods."""
    # Creation methods
    d1 = {"name": "Alice", "age": 30}
    d2 = dict(name="Bob", age=25)
    d3 = dict.fromkeys(["a", "b", "c"], 0)
    print(f"d1: {d1}")
    print(f"d2: {d2}")
    print(f"d3: {d3}")

    # Safe access
    print(f"\nd1['name']:          {d1['name']}")
    print(f"d1.get('email'):     {d1.get('email')}")
    print(f"d1.get('email', ?):  {d1.get('email', 'N/A')}")

    # setdefault: get or set-and-get
    d1.setdefault("email", "alice@example.com")
    print(f"After setdefault:    {d1}")

    # Update and merge
    d1.update({"age": 31, "city": "NYC"})
    print(f"After update:        {d1}")

    # Merge operator (Python 3.9+)
    merged = d1 | {"country": "US", "age": 32}
    print(f"Merged (|):          {merged}")

    # Iteration
    print("\nIteration:")
    for key in d1:
        print(f"  {key}: {d1[key]}")

    # Dictionary views
    print(f"\nKeys:   {list(d1.keys())}")
    print(f"Values: {list(d1.values())}")
    print(f"Items:  {list(d1.items())}")

    # defaultdict
    word_count = defaultdict(int)
    for word in "the cat sat on the mat the cat".split():
        word_count[word] += 1
    print(f"\nWord count: {dict(word_count)}")

    # Counter
    counter = Counter("mississippi")
    print(f"Counter: {counter}")
    print(f"Most common 3: {counter.most_common(3)}")


def set_operations():
    """Set creation and mathematical set operations."""
    a = {1, 2, 3, 4, 5}
    b = {4, 5, 6, 7, 8}

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a | b (union):        {a | b}")
    print(f"a & b (intersection): {a & b}")
    print(f"a - b (difference):   {a - b}")
    print(f"a ^ b (symmetric):    {a ^ b}")
    print(f"a <= a|b (subset):    {a <= a | b}")

    # Set for deduplication
    data = [1, 3, 2, 3, 1, 4, 2, 5, 3]
    unique = list(set(data))
    print(f"\nDeduplicated {data} -> {sorted(unique)}")

    # Frozen set (immutable, hashable)
    fs = frozenset([1, 2, 3])
    print(f"frozenset: {fs}")

    # Set of sets requires frozenset
    set_of_sets = {frozenset([1, 2]), frozenset([3, 4])}
    print(f"Set of frozensets: {set_of_sets}")


def comprehension_showcase():
    """Advanced comprehension patterns."""
    # Nested list comprehension: matrix transpose
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    transpose = [[row[i] for row in matrix] for i in range(3)]
    print(f"Matrix:    {matrix}")
    print(f"Transpose: {transpose}")

    # Dict comprehension: invert a dictionary
    original = {"a": 1, "b": 2, "c": 3}
    inverted = {v: k for k, v in original.items()}
    print(f"\nOriginal: {original}")
    print(f"Inverted: {inverted}")

    # Filtering with comprehension
    scores = {"Alice": 95, "Bob": 62, "Charlie": 88, "Diana": 45}
    passed = {k: v for k, v in scores.items() if v >= 70}
    print(f"\nAll scores: {scores}")
    print(f"Passed:     {passed}")

    # Nested dict comprehension
    multiplication = {i: {j: i * j for j in range(1, 6)} for i in range(1, 4)}
    print(f"\nMultiplication table:")
    for row, cols in multiplication.items():
        print(f"  {row}: {cols}")


def unpacking_patterns():
    """Advanced unpacking and structuring."""
    # Swap
    a, b = 1, 2
    a, b = b, a
    print(f"Swapped: a={a}, b={b}")

    # Nested unpacking
    data = (1, (2, 3), 4)
    a, (b, c), d = data
    print(f"Nested unpack: a={a}, b={b}, c={c}, d={d}")

    # Star unpacking in assignments
    first, *middle, last = range(10)
    print(f"first={first}, middle={middle}, last={last}")

    # Merging dicts with unpacking
    defaults = {"color": "blue", "size": 10, "font": "Arial"}
    overrides = {"size": 14, "bold": True}
    config = {**defaults, **overrides}
    print(f"\nMerged config: {config}")

    # Merging lists with unpacking
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    combined = [*list1, 0, *list2]
    print(f"Combined: {combined}")


if __name__ == "__main__":
    sections = [
        ("List Operations", list_operations),
        ("Tuple Operations", tuple_operations),
        ("Dict Operations", dict_operations),
        ("Set Operations", set_operations),
        ("Comprehension Showcase", comprehension_showcase),
        ("Unpacking Patterns", unpacking_patterns),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
