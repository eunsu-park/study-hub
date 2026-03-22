# 표준 라이브러리 핵심

**이전**: [예외 처리](./12_Exception_Handling.md) | **다음**: [Python 관용구와 모범 사례](./14_Python_Idioms_and_Best_Practices.md)

> **주제**: Python 기초
> **수업**: 14개 중 13번째
> **선수 지식**: 함수, 자료 구조, 모듈과 패키지, 예외 처리

## 학습 목표

이 수업을 완료하면 다음을 할 수 있습니다:

1. `collections` 타입(`Counter`, `defaultdict`, `deque`, `namedtuple`, `OrderedDict`)을 사용하여 데이터 조작을 단순화하기
2. `itertools` 함수(`chain`, `product`, `combinations`, `permutations`, `groupby`, `islice`)를 적용하여 효율적으로 반복하기
3. `functools` 유틸리티(`partial`, `lru_cache`, `reduce`)를 활용하여 함수형 프로그래밍 패턴 구현하기
4. `datetime` 모듈(`date`, `time`, `datetime`, `timedelta`, `strftime`/`strptime`)로 날짜와 시간을 조작하기
5. `os`와 `sys`(`os.environ`, `sys.argv`, `os.getcwd`)를 사용하여 운영 체제 및 런타임과 상호 작용하기
6. 크로스 플랫폼 파일 시스템 작업을 위해 `pathlib.Path`로 경로 작업 수행하기
7. `math`, `random`, `copy` 모듈을 사용하여 수학 연산, 난수 생성, 객체 복제하기
8. `argparse`로 명령줄 인터페이스 구축하기

---

## 소개

Python의 모토는 "배터리 포함"입니다. 표준 라이브러리는 텍스트 처리부터 네트워킹, 수학까지 모든 것을 다루는 200개 이상의 모듈을 포함하고 있습니다. 서드파티 패키지에 손을 뻗기 전에, 이미 설치된 도구로 문제를 해결할 수 있는 경우가 많습니다.

이 수업은 일상 프로그래밍에 가장 유용한 표준 라이브러리 모듈을 다룹니다. 이것들을 숙달하면 생산성이 크게 향상됩니다.

---

## `collections` — 특수 컨테이너 타입

### `Counter` — 개수 세기

`Counter`는 해시 가능한 객체를 세기 위한 딕셔너리 하위 클래스입니다:

```python
from collections import Counter

# Count characters in a string
char_counts = Counter("mississippi")
print(char_counts)
# Counter({'s': 4, 'i': 4, 'p': 2, 'm': 1})

# Count words in a list
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
word_counts = Counter(words)
print(word_counts)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# Most common elements
print(word_counts.most_common(2))
# [('apple', 3), ('banana', 2)]

# Arithmetic operations
inventory_a = Counter(apples=3, oranges=2)
inventory_b = Counter(apples=1, oranges=5, bananas=4)

# Combine inventories
print(inventory_a + inventory_b)
# Counter({'oranges': 7, 'bananas': 4, 'apples': 4})

# Difference
print(inventory_b - inventory_a)
# Counter({'bananas': 4, 'oranges': 3})

# Total count
print(word_counts.total())  # 6 (Python 3.10+)

# Iterate over elements (repeats by count)
print(list(word_counts.elements()))
# ['apple', 'apple', 'apple', 'banana', 'banana', 'cherry']
```

### 실용적인 `Counter` 예제

```python
from collections import Counter

def analyze_text(text):
    """Analyze word frequency in text."""
    # Normalize: lowercase, split on whitespace
    words = text.lower().split()
    # Remove punctuation from each word
    words = [word.strip(".,!?;:\"'()") for word in words]
    words = [w for w in words if w]  # Remove empty strings

    counter = Counter(words)

    print(f"Total words: {sum(counter.values())}")
    print(f"Unique words: {len(counter)}")
    print(f"Top 10 words:")
    for word, count in counter.most_common(10):
        bar = "#" * count
        print(f"  {word:15s} {count:4d} {bar}")

    return counter

text = """
Python is a powerful programming language. Python is used for web development,
data science, machine learning, and more. Python is known for its simplicity
and readability. Many developers love Python because Python makes coding fun.
"""

analyze_text(text)
```

### `defaultdict` — 기본값이 있는 딕셔너리

`defaultdict`는 팩토리 함수로 누락된 키를 자동으로 생성합니다:

```python
from collections import defaultdict

# Regular dict raises KeyError for missing keys
regular = {}
# regular["missing"]  # KeyError!

# defaultdict returns a default value
int_dict = defaultdict(int)       # Default: 0
list_dict = defaultdict(list)     # Default: []
set_dict = defaultdict(set)       # Default: set()

# Counting without checking
word = "abracadabra"
counts = defaultdict(int)
for char in word:
    counts[char] += 1  # No need to check if key exists
print(dict(counts))  # {'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}

# Grouping items
students = [
    ("Alice", "Math"),
    ("Bob", "Science"),
    ("Charlie", "Math"),
    ("Diana", "Science"),
    ("Eve", "Math"),
]

by_subject = defaultdict(list)
for name, subject in students:
    by_subject[subject].append(name)

print(dict(by_subject))
# {'Math': ['Alice', 'Charlie', 'Eve'], 'Science': ['Bob', 'Diana']}
```

### 실용적인 `defaultdict` 예제

```python
from collections import defaultdict

def build_index(documents):
    """Build an inverted index from documents.

    Args:
        documents: List of (doc_id, text) tuples.

    Returns:
        Dictionary mapping each word to the set of document IDs containing it.
    """
    index = defaultdict(set)
    for doc_id, text in documents:
        words = text.lower().split()
        for word in words:
            word = word.strip(".,!?;:")
            index[word].add(doc_id)
    return dict(index)

docs = [
    (1, "Python is great for data science"),
    (2, "Data engineering uses Python and SQL"),
    (3, "Machine learning with Python is powerful"),
]

index = build_index(docs)
print(index["python"])  # {1, 2, 3}
print(index["data"])    # {1, 2}
print(index["sql"])     # {2}
```

### `deque` — 양방향 큐 (Double-Ended Queue)

`deque`는 양쪽 끝에서 O(1)의 추가와 삭제를 제공합니다 (리스트는 왼쪽 작업에 O(n)):

```python
from collections import deque

# Basic operations
d = deque([1, 2, 3, 4, 5])
d.append(6)        # Add to right: [1, 2, 3, 4, 5, 6]
d.appendleft(0)    # Add to left:  [0, 1, 2, 3, 4, 5, 6]
d.pop()            # Remove from right: 6
d.popleft()        # Remove from left: 0
print(d)           # deque([1, 2, 3, 4, 5])

# Extend from both sides
d.extend([6, 7, 8])         # deque([1, 2, 3, 4, 5, 6, 7, 8])
d.extendleft([-2, -1, 0])   # deque([0, -1, -2, 1, 2, 3, 4, 5, 6, 7, 8])

# Rotate
d = deque([1, 2, 3, 4, 5])
d.rotate(2)   # Rotate right: deque([4, 5, 1, 2, 3])
d.rotate(-2)  # Rotate left:  deque([1, 2, 3, 4, 5])

# Fixed-size deque (automatically drops oldest items)
recent = deque(maxlen=3)
recent.append("first")
recent.append("second")
recent.append("third")
print(recent)          # deque(['first', 'second', 'third'], maxlen=3)
recent.append("fourth")
print(recent)          # deque(['second', 'third', 'fourth'], maxlen=3)
```

### 실용적인 `deque` 예제

```python
from collections import deque
from datetime import datetime

class RecentActivity:
    """Track the N most recent activities."""

    def __init__(self, max_items=100):
        self._items = deque(maxlen=max_items)

    def add(self, activity):
        self._items.append({
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
        })

    def recent(self, n=10):
        """Return the n most recent activities."""
        items = list(self._items)
        return items[-n:]

    def __len__(self):
        return len(self._items)

# Usage
tracker = RecentActivity(max_items=5)
tracker.add("User logged in")
tracker.add("Viewed dashboard")
tracker.add("Updated profile")
tracker.add("Uploaded file")
tracker.add("Sent message")
tracker.add("Logged out")

for item in tracker.recent(3):
    print(f"  [{item['timestamp']}] {item['activity']}")
# Only shows the 3 most recent (and oldest was dropped when 6th was added)
```

### `namedtuple` — 경량 데이터 클래스

```python
from collections import namedtuple

# Define a named tuple type
Point = namedtuple("Point", ["x", "y"])
Color = namedtuple("Color", "red green blue")

# Create instances
p = Point(3, 4)
c = Color(255, 128, 0)

# Access by name or index
print(p.x, p.y)        # 3 4
print(p[0], p[1])      # 3 4
print(c.red, c.green)  # 255 128

# Unpack like a tuple
x, y = p
print(f"({x}, {y})")  # (3, 4)

# Immutable (like tuples)
# p.x = 10  # AttributeError: can't set attribute

# Create modified copy
p2 = p._replace(x=10)
print(p2)  # Point(x=10, y=4)

# Convert to dictionary
print(p._asdict())  # {'x': 3, 'y': 4}

# Default values
Employee = namedtuple("Employee", ["name", "dept", "role"], defaults=["Engineering", "Developer"])
e = Employee("Alice")
print(e)  # Employee(name='Alice', dept='Engineering', role='Developer')
```

### 실용적인 `namedtuple` 예제

```python
from collections import namedtuple

# Database record representation
User = namedtuple("User", ["id", "name", "email", "role"])
Address = namedtuple("Address", ["street", "city", "state", "zip_code"])

def parse_user_csv(line):
    """Parse a CSV line into a User namedtuple."""
    parts = line.strip().split(",")
    return User(
        id=int(parts[0]),
        name=parts[1],
        email=parts[2],
        role=parts[3]
    )

csv_data = [
    "1,Alice,alice@example.com,admin",
    "2,Bob,bob@example.com,editor",
    "3,Charlie,charlie@example.com,viewer",
]

users = [parse_user_csv(line) for line in csv_data]
admins = [u for u in users if u.role == "admin"]
print(f"Admin users: {[u.name for u in admins]}")  # ['Alice']
```

### `OrderedDict`

Python 3.7 이전에는 일반 dict가 삽입 순서를 보장하지 않았습니다. `OrderedDict`는 그때 필수였으며 여전히 고유한 기능을 제공합니다:

```python
from collections import OrderedDict

# Regular dict preserves insertion order (Python 3.7+)
# OrderedDict adds extra features:

od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])

# Move to end
od.move_to_end("a")
print(list(od.keys()))  # ['b', 'c', 'a']

# Move to beginning
od.move_to_end("c", last=False)
print(list(od.keys()))  # ['c', 'b', 'a']

# Pop last item
print(od.popitem())              # ('a', 1)
print(od.popitem(last=False))    # ('c', 3)

# Equality comparison considers order
od1 = OrderedDict([("a", 1), ("b", 2)])
od2 = OrderedDict([("b", 2), ("a", 1)])
print(od1 == od2)  # False (different order)

# Regular dict comparison ignores order
d1 = {"a": 1, "b": 2}
d2 = {"b": 2, "a": 1}
print(d1 == d2)  # True
```

---

## `itertools` — 효율적인 반복자 함수

### `chain` — 여러 이터러블 연결하기

```python
from itertools import chain

# Chain multiple iterables into one
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

for item in chain(list1, list2, list3):
    print(item, end=" ")  # 1 2 3 4 5 6 7 8 9
print()

# Flatten a list of lists
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(chain.from_iterable(nested))
print(flat)  # [1, 2, 3, 4, 5, 6]

# Combine different iterable types
combined = list(chain("abc", [1, 2, 3], (True, False)))
print(combined)  # ['a', 'b', 'c', 1, 2, 3, True, False]
```

### `product` — 데카르트 곱 (Cartesian Product)

```python
from itertools import product

# All combinations of two lists
colors = ["red", "blue"]
sizes = ["S", "M", "L"]

for color, size in product(colors, sizes):
    print(f"  {color}-{size}", end="")
print()
# red-S red-M red-L blue-S blue-M blue-L

# Equivalent to nested loops
for color in colors:
    for size in sizes:
        pass  # Same pairs as product()

# Repeat parameter: product with itself
dice_rolls = list(product(range(1, 7), repeat=2))
print(f"Two dice: {len(dice_rolls)} combinations")  # 36

# All binary strings of length 3
binary = list(product("01", repeat=3))
print(["".join(b) for b in binary])
# ['000', '001', '010', '011', '100', '101', '110', '111']
```

### `combinations`와 `permutations`

```python
from itertools import combinations, permutations

# Combinations: order does not matter, no repetition
items = ["A", "B", "C", "D"]

pairs = list(combinations(items, 2))
print(f"Pairs: {pairs}")
# [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]

triples = list(combinations(items, 3))
print(f"Triples: {triples}")
# [('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'C', 'D'), ('B', 'C', 'D')]

# Permutations: order matters, no repetition
perms = list(permutations(items, 2))
print(f"Permutations of 2: {perms}")
# [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'A'), ('B', 'C'), ...]

# All permutations of a sequence
all_perms = list(permutations([1, 2, 3]))
print(f"All permutations: {all_perms}")
# [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
```

### 실용적인 조합 예제

```python
from itertools import combinations

def find_pairs_summing_to(numbers, target):
    """Find all pairs of numbers that sum to the target."""
    return [
        (a, b) for a, b in combinations(numbers, 2)
        if a + b == target
    ]

numbers = [1, 3, 5, 7, 9, 11, 13]
pairs = find_pairs_summing_to(numbers, 14)
print(f"Pairs summing to 14: {pairs}")
# [(1, 13), (3, 11), (5, 9)]

def team_combinations(players, team_size):
    """Generate all possible teams of a given size."""
    teams = list(combinations(players, team_size))
    print(f"{len(teams)} possible teams of {team_size} from {len(players)} players:")
    for team in teams:
        print(f"  {', '.join(team)}")

team_combinations(["Alice", "Bob", "Charlie", "Diana", "Eve"], 3)
```

### `groupby` — 연속 요소 그룹화

```python
from itertools import groupby
from operator import itemgetter

# IMPORTANT: Data must be sorted by the grouping key first!
data = [
    {"name": "Alice", "dept": "Engineering"},
    {"name": "Bob", "dept": "Engineering"},
    {"name": "Charlie", "dept": "Marketing"},
    {"name": "Diana", "dept": "Marketing"},
    {"name": "Eve", "dept": "Sales"},
]

# Already sorted by dept (alphabetically)
for dept, members in groupby(data, key=itemgetter("dept")):
    member_list = [m["name"] for m in members]
    print(f"  {dept}: {member_list}")
# Engineering: ['Alice', 'Bob']
# Marketing: ['Charlie', 'Diana']
# Sales: ['Eve']

# Group consecutive identical values
text = "aaabbbccddddee"
for char, group in groupby(text):
    count = sum(1 for _ in group)
    print(f"  '{char}' x {count}")
# 'a' x 3, 'b' x 3, 'c' x 2, 'd' x 4, 'e' x 2
```

### `islice` — 반복자 슬라이싱

```python
from itertools import islice

# Slice without creating a full list
def fibonacci():
    """Generate Fibonacci numbers infinitely."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Get the first 10 Fibonacci numbers
first_10 = list(islice(fibonacci(), 10))
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Get elements 5 through 10
fib_5_to_10 = list(islice(fibonacci(), 5, 10))
print(fib_5_to_10)  # [5, 8, 13, 21, 34]

# Get every 3rd element from first 20
every_third = list(islice(fibonacci(), 0, 20, 3))
print(every_third)  # [0, 2, 8, 34, 144, 610, 2584]

# Read first 5 lines of a large file efficiently
from pathlib import Path

def head(filepath, n=5):
    """Print the first n lines of a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in islice(f, n):
            print(line.rstrip())
```

### 기타 유용한 `itertools` 함수

```python
from itertools import accumulate, repeat, cycle, zip_longest, starmap

# accumulate: running totals
numbers = [1, 2, 3, 4, 5]
running_sum = list(accumulate(numbers))
print(running_sum)  # [1, 3, 6, 10, 15]

import operator
running_product = list(accumulate(numbers, operator.mul))
print(running_product)  # [1, 2, 6, 24, 120]

# cycle: repeat an iterable forever
colors = cycle(["red", "green", "blue"])
for i, color in zip(range(7), colors):
    print(f"  Item {i}: {color}")
# Item 0: red, Item 1: green, Item 2: blue, Item 3: red, ...

# repeat: repeat a value
fives = list(repeat(5, 3))
print(fives)  # [5, 5, 5]

# zip_longest: zip with fill value for uneven iterables
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87]
paired = list(zip_longest(names, scores, fillvalue=0))
print(paired)  # [('Alice', 95), ('Bob', 87), ('Charlie', 0)]

# starmap: map with unpacked arguments
pairs = [(2, 3), (4, 5), (6, 7)]
products = list(starmap(operator.mul, pairs))
print(products)  # [6, 20, 42]
```

---

## `functools` — 고차 함수

### `partial` — 함수 인자 미리 채우기

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# Create specialized functions
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(3))    # 27

# Practical: pre-configured logging
import logging

def log_message(level, module, message):
    print(f"[{level}] ({module}) {message}")

# Create module-specific loggers
db_log = partial(log_message, module="database")
api_log = partial(log_message, module="api")

db_log("INFO", message="Connection established")
api_log("ERROR", message="Request timeout")
```

### `lru_cache` — 메모이제이션 (Memoization)

```python
from functools import lru_cache
import time

# Without cache: exponential time
def fibonacci_slow(n):
    if n < 2:
        return n
    return fibonacci_slow(n - 1) + fibonacci_slow(n - 2)

# With cache: linear time
@lru_cache(maxsize=128)
def fibonacci_fast(n):
    if n < 2:
        return n
    return fibonacci_fast(n - 1) + fibonacci_fast(n - 2)

# Timing comparison
start = time.time()
result = fibonacci_fast(100)
elapsed = time.time() - start
print(f"fib(100) = {result}")
print(f"Time: {elapsed:.6f}s")

# Cache statistics
print(fibonacci_fast.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# Clear cache
fibonacci_fast.cache_clear()
```

### 실용적인 `lru_cache` 예제

```python
from functools import lru_cache
from pathlib import Path
import json

@lru_cache(maxsize=32)
def load_config(filepath):
    """Load and cache a configuration file.

    The cache means repeated calls with the same filepath
    return instantly without re-reading the file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=None)  # Unlimited cache
def is_prime(n):
    """Check if n is prime (with memoization)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

primes_under_100 = [n for n in range(100) if is_prime(n)]
print(f"Primes under 100: {primes_under_100}")
```

### `reduce` — 누적 계산

```python
from functools import reduce
import operator

# Sum of a list (better to use sum(), but shows the concept)
numbers = [1, 2, 3, 4, 5]
total = reduce(operator.add, numbers)
print(total)  # 15

# Product of a list
product = reduce(operator.mul, numbers)
print(product)  # 120

# With initial value
total_with_start = reduce(operator.add, numbers, 100)
print(total_with_start)  # 115

# Flatten nested lists
nested = [[1, 2], [3, 4], [5, 6]]
flat = reduce(lambda a, b: a + b, nested)
print(flat)  # [1, 2, 3, 4, 5, 6]

# Find the longest string
words = ["cat", "elephant", "dog", "hippopotamus", "bee"]
longest = reduce(lambda a, b: a if len(a) >= len(b) else b, words)
print(longest)  # hippopotamus

# Build a nested dictionary access
def deep_get(data, keys):
    """Get a value from nested dictionaries using a list of keys."""
    return reduce(lambda d, key: d[key], keys, data)

config = {"database": {"primary": {"host": "localhost", "port": 5432}}}
print(deep_get(config, ["database", "primary", "host"]))  # localhost
```

---

## `datetime` — 날짜와 시간 처리

### `date`, `time`, `datetime` 객체

```python
from datetime import date, time, datetime, timedelta

# Date
today = date.today()
print(today)            # 2024-01-15
print(today.year)       # 2024
print(today.month)      # 1
print(today.day)        # 15
print(today.weekday())  # 0 = Monday, 6 = Sunday

specific_date = date(2024, 12, 25)
print(specific_date)    # 2024-12-25

# Time
t = time(14, 30, 0)
print(t)           # 14:30:00
print(t.hour)      # 14
print(t.minute)    # 30

# Datetime (combines date and time)
now = datetime.now()
print(now)  # 2024-01-15 14:30:45.123456

specific = datetime(2024, 6, 15, 9, 30, 0)
print(specific)  # 2024-06-15 09:30:00
```

### `timedelta` — 시간 연산

```python
from datetime import datetime, timedelta, date

now = datetime.now()

# Add/subtract time
tomorrow = now + timedelta(days=1)
next_week = now + timedelta(weeks=1)
two_hours_ago = now - timedelta(hours=2)
in_90_minutes = now + timedelta(minutes=90)

print(f"Now:          {now}")
print(f"Tomorrow:     {tomorrow}")
print(f"Next week:    {next_week}")
print(f"2 hours ago:  {two_hours_ago}")

# Difference between dates
birthday = date(2024, 7, 4)
today = date.today()
days_until = (birthday - today).days
print(f"Days until birthday: {days_until}")

# Complex timedelta
delta = timedelta(days=5, hours=3, minutes=30)
print(f"Total seconds: {delta.total_seconds()}")  # 448200.0
```

### 형식 지정과 파싱

```python
from datetime import datetime

now = datetime.now()

# strftime: datetime -> string
print(now.strftime("%Y-%m-%d"))           # 2024-01-15
print(now.strftime("%B %d, %Y"))          # January 15, 2024
print(now.strftime("%I:%M %p"))           # 02:30 PM
print(now.strftime("%Y-%m-%d %H:%M:%S"))  # 2024-01-15 14:30:45
print(now.strftime("%A, %B %d"))          # Monday, January 15

# strptime: string -> datetime
date_str = "2024-03-15 09:30:00"
parsed = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
print(parsed)  # 2024-03-15 09:30:00

# ISO format (recommended for data exchange)
iso_str = now.isoformat()
print(iso_str)  # 2024-01-15T14:30:45.123456

parsed_iso = datetime.fromisoformat(iso_str)
print(parsed_iso == now)  # True
```

### 일반적인 형식 코드

| 코드 | 의미 | 예시 |
|------|------|------|
| `%Y` | 4자리 연도 | 2024 |
| `%m` | 0이 붙은 월 | 01-12 |
| `%d` | 0이 붙은 일 | 01-31 |
| `%H` | 24시간 시 | 00-23 |
| `%I` | 12시간 시 | 01-12 |
| `%M` | 분 | 00-59 |
| `%S` | 초 | 00-59 |
| `%p` | AM/PM | AM, PM |
| `%A` | 전체 요일 | Monday |
| `%B` | 전체 월 | January |
| `%a` | 축약 요일 | Mon |
| `%b` | 축약 월 | Jan |

### 실용적인 `datetime` 예제

```python
from datetime import datetime, timedelta

def format_relative_time(dt):
    """Format a datetime as a human-readable relative time."""
    now = datetime.now()
    diff = now - dt

    seconds = diff.total_seconds()
    if seconds < 0:
        return "in the future"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    if seconds < 604800:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"

    return dt.strftime("%B %d, %Y")

# Test
print(format_relative_time(datetime.now() - timedelta(seconds=30)))  # just now
print(format_relative_time(datetime.now() - timedelta(minutes=5)))   # 5 minutes ago
print(format_relative_time(datetime.now() - timedelta(hours=3)))     # 3 hours ago
print(format_relative_time(datetime.now() - timedelta(days=2)))      # 2 days ago
```

---

## `os`와 `sys` — 시스템 상호 작용

### `os` — 운영 체제 인터페이스

```python
import os

# Environment variables
print(os.environ.get("HOME"))         # /home/user
print(os.environ.get("PATH"))         # /usr/bin:/usr/local/bin:...
print(os.getenv("DEBUG", "false"))    # Get with default

# Set an environment variable (for current process only)
os.environ["MY_APP_MODE"] = "testing"

# Current directory
print(os.getcwd())  # /home/user/project

# List directory contents
print(os.listdir("."))  # ['file1.py', 'file2.py', 'data/']

# Create directories
os.makedirs("data/output/reports", exist_ok=True)

# Check existence
print(os.path.exists("data.txt"))
print(os.path.isfile("data.txt"))
print(os.path.isdir("data"))

# File metadata
if os.path.exists("data.txt"):
    size = os.path.getsize("data.txt")
    print(f"Size: {size} bytes")
```

### `sys` — Python 런타임

```python
import sys

# Command-line arguments
print(sys.argv)  # ['script.py', 'arg1', 'arg2']

# Python version
print(sys.version)         # 3.12.0 (main, Oct  2 2023, ...)
print(sys.version_info)    # sys.version_info(major=3, minor=12, micro=0, ...)

# Platform
print(sys.platform)        # 'linux', 'darwin', 'win32'

# Module search path
for p in sys.path[:3]:
    print(f"  {p}")

# Memory size of an object
data = list(range(1000))
print(f"Size: {sys.getsizeof(data)} bytes")

# Exit the program
# sys.exit(0)   # Exit with success code
# sys.exit(1)   # Exit with error code
# sys.exit("Error message")  # Exit with message to stderr

# Standard streams
sys.stdout.write("Hello to stdout\n")
sys.stderr.write("Hello to stderr\n")

# Maximum integer size, recursion limit
print(sys.maxsize)            # 9223372036854775807 (on 64-bit)
print(sys.getrecursionlimit())  # 1000 (default)
```

### 실용적인 `sys.argv` 예제

```python
import sys

def main():
    """A simple command-line tool."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <command> [options]")
        print("Commands: count, upper, lower, reverse")
        sys.exit(1)

    command = sys.argv[1]

    if command == "count":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Enter text: ")
        print(f"Characters: {len(text)}")
        print(f"Words: {len(text.split())}")
    elif command == "upper":
        text = " ".join(sys.argv[2:])
        print(text.upper())
    elif command == "lower":
        text = " ".join(sys.argv[2:])
        print(text.lower())
    elif command == "reverse":
        text = " ".join(sys.argv[2:])
        print(text[::-1])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## `math`와 `random`

### `math` — 수학 함수

```python
import math

# Constants
print(math.pi)    # 3.141592653589793
print(math.e)     # 2.718281828459045
print(math.tau)   # 6.283185307179586 (2 * pi)
print(math.inf)   # inf
print(math.nan)   # nan

# Rounding
print(math.ceil(4.1))    # 5  (round up)
print(math.floor(4.9))   # 4  (round down)
print(math.trunc(4.9))   # 4  (truncate toward zero)
print(math.trunc(-4.9))  # -4

# Powers and roots
print(math.sqrt(16))       # 4.0
print(math.pow(2, 10))     # 1024.0
print(math.isqrt(17))      # 4 (integer square root)

# Logarithms
print(math.log(math.e))    # 1.0 (natural log)
print(math.log(100, 10))   # 2.0 (log base 10)
print(math.log2(1024))     # 10.0
print(math.log10(1000))    # 3.0

# Trigonometry (radians)
print(math.sin(math.pi / 2))  # 1.0
print(math.cos(0))            # 1.0
print(math.degrees(math.pi))  # 180.0
print(math.radians(180))      # 3.141592653589793

# Factorial and combinatorics
print(math.factorial(10))    # 3628800
print(math.comb(10, 3))     # 120 (10 choose 3)
print(math.perm(10, 3))     # 720 (permutations)

# Useful functions
print(math.gcd(48, 18))      # 6
print(math.lcm(12, 18))      # 36
print(math.fabs(-3.14))      # 3.14
print(math.copysign(1.0, -3))  # -1.0
print(math.fsum([0.1] * 10))   # 1.0 (precise floating-point sum)
print(math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9))  # True
```

### `random` — 난수

```python
import random

# Set seed for reproducibility
random.seed(42)

# Random float in [0.0, 1.0)
print(random.random())       # 0.6394267984578837

# Random integer in [a, b] (inclusive)
print(random.randint(1, 6))  # 1 (dice roll)

# Random float in [a, b]
print(random.uniform(1.0, 10.0))  # 7.66...

# Choose from a sequence
colors = ["red", "green", "blue", "yellow"]
print(random.choice(colors))  # green

# Choose k items (with replacement)
print(random.choices(colors, k=5))
# ['blue', 'red', 'blue', 'green', 'yellow']

# Choose k items (without replacement)
print(random.sample(colors, k=2))
# ['yellow', 'red']

# Weighted choices
print(random.choices(
    ["common", "uncommon", "rare", "legendary"],
    weights=[70, 20, 8, 2],
    k=10
))

# Shuffle a list in place
deck = list(range(1, 53))
random.shuffle(deck)
hand = deck[:5]
print(f"Your hand: {hand}")

# Normal distribution
height = random.gauss(mu=170, sigma=10)  # mean=170cm, std=10cm
print(f"Random height: {height:.1f} cm")
```

### 실용적인 Random 예제

```python
import random
import string

def generate_password(length=16, include_special=True):
    """Generate a random password."""
    chars = string.ascii_letters + string.digits
    if include_special:
        chars += string.punctuation

    # Ensure at least one of each type
    password = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
    ]
    if include_special:
        password.append(random.choice(string.punctuation))

    # Fill remaining length
    password.extend(random.choices(chars, k=length - len(password)))

    # Shuffle so the required characters are not always at the start
    random.shuffle(password)
    return "".join(password)

for i in range(5):
    print(f"  {generate_password(12)}")
```

---

## `copy` — 얕은 복사와 깊은 복사

### 문제

```python
# Assignment creates a reference, not a copy
original = [1, [2, 3], [4, 5]]
reference = original

reference[0] = 99
print(original)  # [99, [2, 3], [4, 5]] - original is modified!
```

### 얕은 복사 (Shallow Copy)

```python
import copy

original = [1, [2, 3], [4, 5]]
shallow = copy.copy(original)
# Or: shallow = original.copy()
# Or: shallow = list(original)

shallow[0] = 99
print(original)  # [1, [2, 3], [4, 5]] - top-level unchanged

# But nested objects are shared!
shallow[1][0] = 99
print(original)  # [1, [99, 3], [4, 5]] - nested object is shared!
```

### 깊은 복사 (Deep Copy)

```python
import copy

original = [1, [2, 3], [4, 5]]
deep = copy.deepcopy(original)

deep[0] = 99
deep[1][0] = 99
print(original)  # [1, [2, 3], [4, 5]] - completely independent!
print(deep)      # [99, [99, 3], [4, 5]]
```

### 어떤 것을 사용할지

```python
import copy

# Shallow copy: flat data structures
flat_list = [1, 2, 3, 4, 5]
copied = flat_list.copy()  # Shallow is fine

flat_dict = {"a": 1, "b": 2}
copied = flat_dict.copy()  # Shallow is fine

# Deep copy: nested data structures
nested = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
safe_copy = copy.deepcopy(nested)
safe_copy["users"][0]["name"] = "Charlie"
print(nested["users"][0]["name"])  # Alice (unchanged)

# Deep copy handles circular references
a = [1, 2]
a.append(a)  # Circular reference
b = copy.deepcopy(a)  # Works correctly
```

---

## `argparse` — 명령줄 인자 파싱

### 기본 사용법

```python
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="A simple file processing tool"
    )

    # Positional argument (required)
    parser.add_argument("filename", help="File to process")

    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)",
        default=None
    )
    parser.add_argument(
        "-n", "--lines",
        type=int,
        default=10,
        help="Number of lines to process (default: 10)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Processing: {args.filename}")
        print(f"Output: {args.output or 'stdout'}")
        print(f"Lines: {args.lines}")

    # Use args.filename, args.output, args.lines, args.verbose
    process(args.filename, args.output, args.lines, args.verbose)

if __name__ == "__main__":
    main()
```

```bash
$ python tool.py data.csv -o result.txt -n 100 -v
Processing: data.csv
Output: result.txt
Lines: 100

$ python tool.py --help
usage: tool.py [-h] [-o OUTPUT] [-n LINES] [-v] filename

A simple file processing tool

positional arguments:
  filename              File to process

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file (default: stdout)
  -n LINES, --lines LINES
                        Number of lines to process (default: 10)
  -v, --verbose         Enable verbose output
```

### 인자 타입과 선택지

```python
import argparse

parser = argparse.ArgumentParser(description="Data converter")

# Choices restrict allowed values
parser.add_argument(
    "--format",
    choices=["json", "csv", "xml"],
    default="json",
    help="Output format"
)

# Type conversion
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Score threshold (0.0-1.0)"
)

# Multiple values
parser.add_argument(
    "--tags",
    nargs="+",
    help="Tags to filter by (one or more)"
)

# Optional list
parser.add_argument(
    "--exclude",
    nargs="*",
    default=[],
    help="Patterns to exclude"
)

# Count occurrences (e.g., -vvv for verbosity level 3)
parser.add_argument(
    "-v", "--verbose",
    action="count",
    default=0,
    help="Increase verbosity"
)

args = parser.parse_args()
```

### 실용적인 `argparse` 예제

```python
import argparse
import json
import csv
from pathlib import Path

def convert(input_file, output_file, input_format, output_format):
    """Convert between JSON and CSV formats."""
    # Load
    if input_format == "json":
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif input_format == "csv":
        with open(input_file, "r", encoding="utf-8") as f:
            data = list(csv.DictReader(f))

    # Save
    if output_format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif output_format == "csv":
        if data:
            with open(output_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    print(f"Converted {input_file} ({input_format}) -> {output_file} ({output_format})")
    print(f"Records: {len(data)}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert between JSON and CSV formats",
        epilog="Example: python converter.py data.json output.csv"
    )
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument(
        "--input-format",
        choices=["json", "csv"],
        help="Input format (auto-detected from extension if not specified)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        help="Output format (auto-detected from extension if not specified)"
    )

    args = parser.parse_args()

    # Auto-detect formats from file extensions
    in_fmt = args.input_format or Path(args.input).suffix.lstrip(".")
    out_fmt = args.output_format or Path(args.output).suffix.lstrip(".")

    if in_fmt not in ("json", "csv") or out_fmt not in ("json", "csv"):
        parser.error("Cannot detect format. Use --input-format and --output-format")

    convert(args.input, args.output, in_fmt, out_fmt)

if __name__ == "__main__":
    main()
```

---

## 요약

| 모듈 | 주요 타입/함수 | 사용 사례 |
|------|---------------|----------|
| `collections` | `Counter`, `defaultdict`, `deque`, `namedtuple` | 특수 컨테이너 |
| `itertools` | `chain`, `product`, `combinations`, `groupby`, `islice` | 효율적인 반복 |
| `functools` | `partial`, `lru_cache`, `reduce` | 함수 조작 |
| `datetime` | `date`, `datetime`, `timedelta`, `strftime`/`strptime` | 날짜/시간 처리 |
| `os` | `environ`, `getcwd`, `listdir`, `makedirs` | OS 상호 작용 |
| `sys` | `argv`, `path`, `exit`, `version` | Python 런타임 |
| `pathlib` | `Path`, `/` 연산자, `glob`, `stat` | 파일 경로 |
| `math` | `sqrt`, `log`, `ceil`, `floor`, `pi` | 수학 |
| `random` | `random`, `randint`, `choice`, `shuffle`, `seed` | 난수 |
| `copy` | `copy`, `deepcopy` | 객체 복제 |
| `argparse` | `ArgumentParser`, `add_argument` | CLI 인자 파싱 |

핵심 내용:
- 표준 라이브러리는 방대합니다 — **서드파티 패키지를 설치하기 전에 먼저 확인하세요**
- `collections` 타입은 복잡한 dict/list 조작이 필요한 코드를 종종 단순화합니다
- `itertools` 함수는 메모리 효율적입니다 (항목을 한 번에 하나씩 생성)
- `lru_cache`는 순수 함수의 속도를 극적으로 높이는 간단한 방법입니다
- 파일 경로에는 항상 `pathlib.Path`를, 날짜/시간 작업에는 `datetime`을 사용하세요
- `argparse`는 최소한의 코드로 전문적인 CLI 도구를 만듭니다

---

## 추가 자료

- [Python 표준 라이브러리 문서](https://docs.python.org/3/library/index.html)
- [collections — 컨테이너 데이터 타입](https://docs.python.org/3/library/collections.html)
- [itertools — 반복자 빌딩 블록](https://docs.python.org/3/library/itertools.html)
- [functools — 고차 함수](https://docs.python.org/3/library/functools.html)
- [datetime — 날짜와 시간](https://docs.python.org/3/library/datetime.html)

---

**이전**: [예외 처리](./12_Exception_Handling.md) | **다음**: [Python 관용구와 모범 사례](./14_Python_Idioms_and_Best_Practices.md)
