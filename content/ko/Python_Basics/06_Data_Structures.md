# 자료구조 (Data Structures)

**이전**: [함수](./05_Functions.md) | **다음**: [문자열과 텍스트 처리](./07_Strings_and_Text_Processing.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 내장 메서드와 리스트 컴프리헨션 (List Comprehension)을 사용하여 리스트를 생성, 인덱싱, 슬라이싱, 조작한다
2. 패킹, 언패킹, 네임드 튜플 (Named Tuple)을 포함하여 불변 시퀀스인 튜플 (Tuple)을 사용한다
3. 일반 메서드와 딕셔너리 컴프리헨션 (Dictionary Comprehension)으로 딕셔너리 (Dictionary)를 구축하고 질의한다
4. 집합 연산 (합집합, 교집합, 차집합)을 수행하고 세트 컴프리헨션 (Set Comprehension)을 적용한다
5. 가변성, 순서, 검색 속도와 같은 특성에 따라 다른 문제에 적합한 자료구조를 선택한다
6. 중첩 자료구조를 다루고 그 접근 패턴을 이해한다
7. 얕은 복사 (Shallow Copy)와 깊은 복사 (Deep Copy)를 구분하고 흔한 별칭 버그를 피한다

---

자료구조는 데이터를 효율적으로 접근하고 수정할 수 있도록 조직하고 저장하는 컨테이너입니다. 파이썬은 네 가지 기본 내장 컬렉션 타입 -- 리스트, 튜플, 딕셔너리, 세트 -- 을 제공하며, 각각 서로 다른 작업에 적합한 고유한 속성을 가지고 있습니다. 올바른 자료구조를 선택하는 것은 깔끔하고 성능 좋은 코드를 작성하는 데 있어 가장 중요한 결정 중 하나입니다.

## 1. 리스트 (List)

리스트는 순서가 있는 가변 시퀀스입니다. 모든 타입의 항목을 담을 수 있으며 파이썬에서 가장 많이 사용되는 자료구조입니다.

### 리스트 생성

```python
# Empty list
empty = []
also_empty = list()

# List with values
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, None]

# From other iterables
from_range = list(range(5))        # [0, 1, 2, 3, 4]
from_string = list("hello")       # ['h', 'e', 'l', 'l', 'o']
from_tuple = list((10, 20, 30))   # [10, 20, 30]

# Repetition
zeros = [0] * 5                    # [0, 0, 0, 0, 0]
pattern = [1, 2] * 3               # [1, 2, 1, 2, 1, 2]
```

### 인덱싱

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# Positive indexing (from start, 0-based)
print(fruits[0])    # Output: apple
print(fruits[2])    # Output: cherry

# Negative indexing (from end)
print(fruits[-1])   # Output: elderberry
print(fruits[-2])   # Output: date

# Modifying by index
fruits[1] = "blueberry"
print(fruits)  # ['apple', 'blueberry', 'cherry', 'date', 'elderberry']
```

### 슬라이싱

슬라이싱은 `list[start:stop:step]` 구문을 사용합니다. `stop` 인덱스는 포함되지 않습니다.

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:5])     # [2, 3, 4]
print(numbers[:4])      # [0, 1, 2, 3]       (from beginning)
print(numbers[6:])      # [6, 7, 8, 9]       (to end)
print(numbers[::2])     # [0, 2, 4, 6, 8]    (every 2nd element)
print(numbers[1::2])    # [1, 3, 5, 7, 9]    (odd-indexed elements)
print(numbers[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  (reversed)
print(numbers[7:2:-1])  # [7, 6, 5, 4, 3]    (reverse slice)

# Slice assignment
numbers[2:5] = [20, 30, 40]
print(numbers)  # [0, 1, 20, 30, 40, 5, 6, 7, 8, 9]

# Delete via slice
numbers[2:5] = []
print(numbers)  # [0, 1, 5, 6, 7, 8, 9]
```

### 리스트 메서드

```python
fruits = ["apple", "banana", "cherry"]

# append: add one item to the end
fruits.append("date")
print(fruits)  # ['apple', 'banana', 'cherry', 'date']

# extend: add all items from another iterable
fruits.extend(["elderberry", "fig"])
print(fruits)  # ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']

# insert: add item at a specific position
fruits.insert(1, "blueberry")
print(fruits)  # ['apple', 'blueberry', 'banana', 'cherry', 'date', 'elderberry', 'fig']

# remove: remove the first occurrence of a value
fruits.remove("banana")
print(fruits)  # ['apple', 'blueberry', 'cherry', 'date', 'elderberry', 'fig']

# pop: remove and return item at index (default: last)
last = fruits.pop()
print(last)    # fig
print(fruits)  # ['apple', 'blueberry', 'cherry', 'date', 'elderberry']

second = fruits.pop(1)
print(second)  # blueberry
print(fruits)  # ['apple', 'cherry', 'date', 'elderberry']

# index: find position of first occurrence
print(fruits.index("cherry"))  # 1

# count: count occurrences
numbers = [1, 2, 3, 2, 4, 2, 5]
print(numbers.count(2))  # 3

# sort: sort in place
numbers.sort()
print(numbers)  # [1, 2, 2, 2, 3, 4, 5]

numbers.sort(reverse=True)
print(numbers)  # [5, 4, 3, 2, 2, 2, 1]

# reverse: reverse in place
numbers.reverse()
print(numbers)  # [1, 2, 2, 2, 3, 4, 5]

# clear: remove all items
numbers.clear()
print(numbers)  # []
```

### 리스트 멤버십과 길이

```python
colors = ["red", "green", "blue"]

print("red" in colors)      # True
print("yellow" in colors)   # False
print("yellow" not in colors)  # True
print(len(colors))          # 3
```

### 리스트 컴프리헨션

리스트 컴프리헨션 (List Comprehension)은 기존 이터러블을 기반으로 리스트를 생성하는 간결한 방법을 제공합니다.

```python
# Basic syntax: [expression for item in iterable]
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition: [expression for item in iterable if condition]
evens = [x for x in range(20) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# With transformation and condition
words = ["Hello", "WORLD", "Python", "CODE"]
lower_short = [w.lower() for w in words if len(w) <= 5]
print(lower_short)  # ['hello', 'world', 'code']

# Nested loops in comprehension
pairs = [(x, y) for x in range(3) for y in range(3)]
print(pairs)
# [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

# Flattening a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# If-else in comprehension (expression part, not filter)
labels = ["even" if x % 2 == 0 else "odd" for x in range(6)]
print(labels)  # ['even', 'odd', 'even', 'odd', 'even', 'odd']
```

---

## 2. 튜플 (Tuple)

튜플은 순서가 있는 **불변** 시퀀스입니다. 한 번 생성되면 요소를 변경할 수 없습니다.

### 튜플 생성

```python
# With parentheses
point = (3, 4)
rgb = (255, 128, 0)

# Without parentheses (tuple packing)
coordinates = 10, 20, 30
print(type(coordinates))  # <class 'tuple'>

# Single-element tuple (comma is required!)
single = (42,)
print(type(single))   # <class 'tuple'>

not_tuple = (42)
print(type(not_tuple))  # <class 'int'>

# Empty tuple
empty = ()
also_empty = tuple()

# From iterable
from_list = tuple([1, 2, 3])
from_string = tuple("abc")  # ('a', 'b', 'c')
```

### 튜플 연산

```python
point = (3, 4, 5)

# Indexing and slicing (same as lists)
print(point[0])     # 3
print(point[-1])    # 5
print(point[1:])    # (4, 5)

# Immutability: cannot modify
# point[0] = 10  # TypeError: 'tuple' object does not support item assignment

# Concatenation and repetition
a = (1, 2)
b = (3, 4)
print(a + b)       # (1, 2, 3, 4)
print(a * 3)       # (1, 2, 1, 2, 1, 2)

# Membership
print(3 in point)  # True
print(len(point))  # 3

# Methods
numbers = (1, 2, 3, 2, 4, 2)
print(numbers.count(2))  # 3
print(numbers.index(3))  # 2
```

### 튜플 패킹과 언패킹

```python
# Packing
person = "Alice", 30, "Seoul"

# Unpacking
name, age, city = person
print(f"{name}, {age}, {city}")  # Alice, 30, Seoul

# Swap variables (Python idiom)
a, b = 10, 20
a, b = b, a
print(a, b)  # 20 10

# Extended unpacking with *
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5

# Ignore values with _
_, age, _ = person
print(age)  # 30

# Unpacking in loops
pairs = [(1, "a"), (2, "b"), (3, "c")]
for number, letter in pairs:
    print(f"{number} -> {letter}")
# 1 -> a
# 2 -> b
# 3 -> c
```

### 튜플 vs 리스트 사용 시점

| 튜플 | 리스트 |
|--------|-------|
| 불변 (안전, 해시 가능) | 가변 (유연) |
| 고정된 항목 모음 | 늘어나거나 줄어드는 컬렉션 |
| 딕셔너리 키로 사용 가능 | 딕셔너리 키로 사용 불가 |
| 약간 더 빠름 | 약간 더 느림 |
| 레코드를 표현 (이름, 나이) | 같은 타입의 컬렉션을 표현 |

### 네임드 튜플

네임드 튜플 (Named Tuple)은 각 위치에 의미 있는 이름을 부여하여 코드를 더 읽기 쉽게 만듭니다.

```python
from collections import namedtuple

# Define a named tuple type
Point = namedtuple("Point", ["x", "y"])

p = Point(3, 4)
print(p.x)       # 3
print(p.y)       # 4
print(p[0])      # 3 (still works like a tuple)
print(p)         # Point(x=3, y=4)

# Real-world example
Student = namedtuple("Student", ["name", "grade", "age"])

students = [
    Student("Alice", 95, 20),
    Student("Bob", 88, 22),
    Student("Charlie", 92, 21),
]

# Access by name is much clearer
for s in students:
    print(f"{s.name}: grade={s.grade}, age={s.age}")

# Named tuples are still immutable
# p.x = 10  # AttributeError

# Create modified copy with _replace
p2 = p._replace(x=10)
print(p2)  # Point(x=10, y=4)

# Convert to dictionary
print(p._asdict())  # {'x': 3, 'y': 4}
```

---

## 3. 딕셔너리 (Dictionary)

딕셔너리는 키-값 쌍의 **비순서** (Python 3.7부터 삽입 순서 유지) 가변 매핑입니다. 키는 해시 가능해야 합니다 (문자열, 숫자, 해시 가능한 요소로 구성된 튜플).

### 딕셔너리 생성

```python
# Curly braces
student = {"name": "Alice", "age": 20, "grade": 95}

# dict() constructor
config = dict(host="localhost", port=8080, debug=True)

# From list of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
d = dict(pairs)
print(d)  # {'a': 1, 'b': 2, 'c': 3}

# dict.fromkeys
keys = ["name", "age", "city"]
defaults = dict.fromkeys(keys, "unknown")
print(defaults)  # {'name': 'unknown', 'age': 'unknown', 'city': 'unknown'}

# Empty dictionary
empty = {}
also_empty = dict()
```

### 접근과 수정

```python
student = {"name": "Alice", "age": 20, "grade": 95}

# Access by key
print(student["name"])   # Alice

# KeyError if key does not exist
# print(student["email"])  # KeyError: 'email'

# get() returns None (or default) instead of raising error
print(student.get("email"))           # None
print(student.get("email", "N/A"))    # N/A

# Add or update
student["email"] = "alice@example.com"   # Add new key
student["age"] = 21                       # Update existing key
print(student)
# {'name': 'Alice', 'age': 21, 'grade': 95, 'email': 'alice@example.com'}

# Delete
del student["email"]
print(student)  # {'name': 'Alice', 'age': 21, 'grade': 95}
```

### 딕셔너리 메서드

```python
student = {"name": "Alice", "age": 20, "grade": 95}

# keys(), values(), items()
print(list(student.keys()))    # ['name', 'age', 'grade']
print(list(student.values()))  # ['Alice', 20, 95]
print(list(student.items()))   # [('name', 'Alice'), ('age', 20), ('grade', 95)]

# update: merge another dictionary
student.update({"age": 21, "city": "Seoul"})
print(student)  # {'name': 'Alice', 'age': 21, 'grade': 95, 'city': 'Seoul'}

# setdefault: get value or set it if missing
email = student.setdefault("email", "unknown@example.com")
print(email)     # unknown@example.com
print(student["email"])  # unknown@example.com

# setdefault does not overwrite existing keys
name = student.setdefault("name", "Bob")
print(name)  # Alice (not overwritten)

# pop: remove and return value
grade = student.pop("grade")
print(grade)    # 95

# pop with default (no KeyError)
missing = student.pop("phone", "not found")
print(missing)  # not found

# popitem: remove and return last inserted pair
last = student.popitem()
print(last)  # ('email', 'unknown@example.com')

# copy: shallow copy
copy_student = student.copy()

# clear: remove all items
student.clear()
print(student)  # {}
```

### 딕셔너리 순회

```python
scores = {"Alice": 95, "Bob": 88, "Charlie": 92}

# Iterate over keys (default)
for name in scores:
    print(name)

# Iterate over values
for score in scores.values():
    print(score)

# Iterate over key-value pairs
for name, score in scores.items():
    print(f"{name}: {score}")

# Check membership (checks keys by default)
print("Alice" in scores)     # True
print(95 in scores)           # False (checks keys, not values)
print(95 in scores.values())  # True
```

### 딕셔너리 컴프리헨션

```python
# Basic dictionary comprehension
squares = {x: x ** 2 for x in range(6)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# With condition
even_squares = {x: x ** 2 for x in range(10) if x % 2 == 0}
print(even_squares)  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Invert a dictionary (swap keys and values)
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}

# From two lists
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
name_age = {name: age for name, age in zip(names, ages)}
print(name_age)  # {'Alice': 25, 'Bob': 30, 'Charlie': 35}

# Filter and transform
words = ["Hello", "World", "Python", "Go", "JS"]
long_words = {w: len(w) for w in words if len(w) > 3}
print(long_words)  # {'Hello': 5, 'World': 5, 'Python': 6}
```

### 병합 연산자 (Python 3.9+)

```python
defaults = {"color": "blue", "size": "medium", "theme": "light"}
user_prefs = {"color": "red", "font": "Arial"}

# Merge with | (creates new dict, right side wins on conflicts)
merged = defaults | user_prefs
print(merged)
# {'color': 'red', 'size': 'medium', 'theme': 'light', 'font': 'Arial'}

# In-place merge with |=
defaults |= user_prefs
print(defaults)
# {'color': 'red', 'size': 'medium', 'theme': 'light', 'font': 'Arial'}
```

---

## 4. 세트 (Set)

세트는 **고유한** 요소의 **비순서** 컬렉션입니다. 멤버십 테스트, 중복 제거, 수학적 집합 연산에 이상적입니다.

### 세트 생성

```python
# Curly braces (but {} creates a dict, not a set!)
colors = {"red", "green", "blue"}
print(type(colors))  # <class 'set'>

# Empty set must use set()
empty = set()
print(type(empty))   # <class 'set'>

# From iterable (automatically removes duplicates)
numbers = set([1, 2, 2, 3, 3, 3])
print(numbers)  # {1, 2, 3}

from_string = set("mississippi")
print(from_string)  # {'m', 'i', 's', 'p'} (order may vary)
```

### 세트 연산

```python
# Adding and removing
colors = {"red", "green", "blue"}

colors.add("yellow")
print(colors)  # {'red', 'green', 'blue', 'yellow'}

colors.discard("green")   # Remove if present (no error if missing)
colors.remove("blue")     # Remove (raises KeyError if missing)
# colors.remove("purple")  # KeyError

popped = colors.pop()     # Remove and return arbitrary element
print(f"Popped: {popped}")
```

### 수학적 집합 연산

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# Union: elements in either set
print(a | b)                # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))           # {1, 2, 3, 4, 5, 6, 7, 8}

# Intersection: elements in both sets
print(a & b)                # {4, 5}
print(a.intersection(b))    # {4, 5}

# Difference: elements in a but not in b
print(a - b)                # {1, 2, 3}
print(a.difference(b))      # {1, 2, 3}

# Symmetric difference: elements in either but not both
print(a ^ b)                        # {1, 2, 3, 6, 7, 8}
print(a.symmetric_difference(b))    # {1, 2, 3, 6, 7, 8}
```

### 세트 관계

```python
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}
c = {6, 7}

# Subset and superset
print(a.issubset(b))      # True (all of a is in b)
print(b.issuperset(a))    # True (b contains all of a)
print(a <= b)             # True (operator form of issubset)

# Disjoint: no common elements
print(a.isdisjoint(c))    # True
print(a.isdisjoint(b))    # False
```

### 실용적인 세트 예제

```python
# Remove duplicates from a list (order not preserved)
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
unique = list(set(data))
print(sorted(unique))  # [1, 2, 3, 4, 5, 6, 9]

# Remove duplicates preserving order (Python 3.7+)
unique_ordered = list(dict.fromkeys(data))
print(unique_ordered)  # [3, 1, 4, 5, 9, 2, 6]

# Find common elements
list_a = [1, 2, 3, 4, 5]
list_b = [4, 5, 6, 7, 8]
common = set(list_a) & set(list_b)
print(common)  # {4, 5}

# Find items in one list but not another
only_a = set(list_a) - set(list_b)
print(only_a)  # {1, 2, 3}

# Efficient membership testing
valid_usernames = {"alice", "bob", "charlie", "diana"}
username = "bob"
if username in valid_usernames:  # O(1) average time
    print(f"{username} is valid")
```

### 세트 컴프리헨션

```python
# Basic set comprehension
squares = {x ** 2 for x in range(-5, 6)}
print(squares)  # {0, 1, 4, 9, 16, 25}

# With condition
even_squares = {x ** 2 for x in range(10) if x % 2 == 0}
print(even_squares)  # {0, 4, 16, 36, 64}

# Extract unique first characters
words = ["apple", "avocado", "banana", "blueberry", "cherry"]
first_chars = {w[0] for w in words}
print(first_chars)  # {'a', 'b', 'c'}
```

### 프로즌세트 (Frozen Set)

프로즌세트 (Frozen Set)는 불변 세트입니다. 딕셔너리 키나 다른 세트의 요소로 사용할 수 있습니다.

```python
fs = frozenset([1, 2, 3])
# fs.add(4)  # AttributeError: 'frozenset' has no method 'add'

# Can be used as a dictionary key
permissions = {
    frozenset({"read"}): "viewer",
    frozenset({"read", "write"}): "editor",
    frozenset({"read", "write", "admin"}): "admin",
}
user_perms = frozenset({"read", "write"})
print(permissions[user_perms])  # editor
```

---

## 5. 올바른 자료구조 선택

| 특성 | 리스트 | 튜플 | 딕셔너리 | 세트 |
|---------|------|-------|------|-----|
| 순서 유지 | 예 | 예 | 예 (3.7+) | 아니오 |
| 가변 | 예 | 아니오 | 예 | 예 |
| 중복 허용 | 예 | 예 | 키: 아니오 | 아니오 |
| 인덱싱 가능 | 예 | 예 | 키로 | 아니오 |
| 해시 가능 | 아니오 | 예* | 아니오 | 아니오 |
| 사용 사례 | 일반 시퀀스 | 불변 레코드 | 키-값 매핑 | 고유 요소 |

\* 튜플은 모든 요소가 해시 가능할 때만 해시 가능합니다.

### 선택 가이드

```python
# Need ordered, changeable collection? -> list
shopping = ["milk", "eggs", "bread"]

# Need immutable record or dictionary key? -> tuple
coordinate = (40.7128, -74.0060)

# Need to look up values by key? -> dict
phonebook = {"Alice": "555-0100", "Bob": "555-0200"}

# Need unique elements or set math? -> set
tags = {"python", "tutorial", "beginner"}

# Need to count occurrences? -> dict or Counter
from collections import Counter
words = ["the", "cat", "sat", "on", "the", "mat"]
word_count = Counter(words)
print(word_count)  # Counter({'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1})
```

---

## 6. 중첩 자료구조

실세계 데이터는 종종 복잡하여 자료구조의 중첩이 필요합니다.

### 딕셔너리의 리스트

```python
students = [
    {"name": "Alice", "grades": [95, 88, 92]},
    {"name": "Bob", "grades": [78, 85, 90]},
    {"name": "Charlie", "grades": [92, 95, 88]},
]

# Access nested data
print(students[0]["name"])        # Alice
print(students[0]["grades"][1])   # 88

# Calculate average for each student
for student in students:
    avg = sum(student["grades"]) / len(student["grades"])
    print(f"{student['name']}: {avg:.1f}")
# Alice: 91.7
# Bob: 84.3
# Charlie: 91.7
```

### 리스트의 딕셔너리

```python
class_roster = {
    "math": ["Alice", "Bob", "Charlie"],
    "science": ["Bob", "Diana", "Eve"],
    "history": ["Alice", "Charlie", "Eve"],
}

# Find students taking all subjects
all_students = set()
for subject_students in class_roster.values():
    all_students.update(subject_students)
print(all_students)  # {'Alice', 'Bob', 'Charlie', 'Diana', 'Eve'}

# Find students in both math and science
math_students = set(class_roster["math"])
science_students = set(class_roster["science"])
print(math_students & science_students)  # {'Bob'}
```

### 중첩 딕셔너리

```python
company = {
    "engineering": {
        "backend": {"lead": "Alice", "members": 5},
        "frontend": {"lead": "Bob", "members": 3},
    },
    "marketing": {
        "digital": {"lead": "Charlie", "members": 4},
    },
}

# Deep access
print(company["engineering"]["backend"]["lead"])  # Alice

# Safe deep access
def deep_get(data, *keys, default=None):
    """Safely get a value from a nested dictionary."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

print(deep_get(company, "engineering", "backend", "lead"))      # Alice
print(deep_get(company, "sales", "team", "lead", default="N/A")) # N/A
```

### defaultdict로 중첩 구조 구축

```python
from collections import defaultdict

# Group items by category
items = [
    ("fruit", "apple"),
    ("vegetable", "carrot"),
    ("fruit", "banana"),
    ("vegetable", "broccoli"),
    ("fruit", "cherry"),
]

grouped = defaultdict(list)
for category, item in items:
    grouped[category].append(item)

print(dict(grouped))
# {'fruit': ['apple', 'banana', 'cherry'], 'vegetable': ['carrot', 'broccoli']}

# Nested defaultdict for two-level grouping
sales = [
    ("2024", "Q1", 100),
    ("2024", "Q2", 150),
    ("2024", "Q1", 200),
    ("2025", "Q1", 300),
]

yearly = defaultdict(lambda: defaultdict(list))
for year, quarter, amount in sales:
    yearly[year][quarter].append(amount)

print(yearly["2024"]["Q1"])  # [100, 200]
```

---

## 7. 복사: 얕은 복사 vs 깊은 복사

### 별칭 문제

```python
# Assignment creates an alias, NOT a copy
original = [1, 2, 3]
alias = original

alias.append(4)
print(original)  # [1, 2, 3, 4] -- both names point to the same list!
print(alias is original)  # True
```

### 얕은 복사

얕은 복사 (Shallow Copy)는 새로운 외부 컨테이너를 생성하지만 내부 객체에 대한 참조를 공유합니다.

```python
import copy

# Ways to create a shallow copy of a list
original = [1, 2, 3, [4, 5]]

copy1 = original.copy()        # list.copy() method
copy2 = original[:]            # slice
copy3 = list(original)         # constructor
copy4 = copy.copy(original)    # copy module

# The outer list is independent
copy1.append(6)
print(original)  # [1, 2, 3, [4, 5]] -- not affected
print(copy1)     # [1, 2, 3, [4, 5], 6]

# But nested objects are still shared!
copy1[3].append(99)
print(original)  # [1, 2, 3, [4, 5, 99]] -- nested list IS affected!
print(copy1)     # [1, 2, 3, [4, 5, 99], 6]
```

### 깊은 복사

깊은 복사 (Deep Copy)는 모든 중첩 객체의 완전히 독립적인 복사본을 생성합니다.

```python
import copy

original = [1, 2, [3, 4, [5, 6]]]

# Deep copy
deep = copy.deepcopy(original)

deep[2][2].append(7)
print(original)  # [1, 2, [3, 4, [5, 6]]]  -- not affected
print(deep)      # [1, 2, [3, 4, [5, 6, 7]]]
```

### 딕셔너리 복사

```python
import copy

original = {
    "name": "Alice",
    "scores": [95, 88, 92],
    "address": {"city": "Seoul", "zip": "12345"},
}

# Shallow copy
shallow = original.copy()
shallow["scores"].append(100)
print(original["scores"])  # [95, 88, 92, 100] -- shared!

# Deep copy
original["scores"].pop()  # Remove the 100 we just added
deep = copy.deepcopy(original)
deep["scores"].append(100)
deep["address"]["city"] = "Busan"
print(original["scores"])         # [95, 88, 92] -- independent
print(original["address"]["city"]) # Seoul -- independent
```

### 각각을 사용해야 할 때

| 시나리오 | 방법 |
|----------|--------|
| 단순한 평면 리스트/딕셔너리 | 얕은 복사로 충분 |
| 중첩된 가변 객체 | 깊은 복사 필요 |
| 성능 중요, 읽기 전용 | 참조 공유 (별칭) |
| 불변 데이터 (튜플, 문자열) | 복사 불필요 |

---

## 8. 실용적인 예제

### 예제: 재고 관리

```python
inventory = {}

def add_item(name, quantity, price):
    """Add or update an item in inventory."""
    if name in inventory:
        inventory[name]["quantity"] += quantity
    else:
        inventory[name] = {"quantity": quantity, "price": price}

def remove_item(name, quantity):
    """Remove quantity of an item. Remove entry if quantity reaches 0."""
    if name not in inventory:
        print(f"{name} not in inventory")
        return
    inventory[name]["quantity"] -= quantity
    if inventory[name]["quantity"] <= 0:
        del inventory[name]

def get_total_value():
    """Calculate total inventory value."""
    return sum(
        item["quantity"] * item["price"]
        for item in inventory.values()
    )

def get_report():
    """Generate inventory report sorted by value."""
    items = []
    for name, info in inventory.items():
        value = info["quantity"] * info["price"]
        items.append((name, info["quantity"], info["price"], value))

    items.sort(key=lambda x: -x[3])  # Sort by value descending
    return items

add_item("apple", 50, 1.20)
add_item("banana", 30, 0.50)
add_item("cherry", 100, 3.00)
add_item("apple", 20, 1.20)  # Add more apples

print(f"Total value: ${get_total_value():.2f}")
# Total value: $399.00

for name, qty, price, value in get_report():
    print(f"  {name}: {qty} x ${price:.2f} = ${value:.2f}")
# cherry: 100 x $3.00 = $300.00
# apple: 70 x $1.20 = $84.00
# banana: 30 x $0.50 = $15.00
```

### 예제: 중첩 리스트를 활용한 행렬 연산

```python
def create_matrix(rows, cols, fill=0):
    """Create a matrix (list of lists)."""
    return [[fill] * cols for _ in range(rows)]

def print_matrix(matrix):
    """Pretty-print a matrix."""
    for row in matrix:
        print("  ".join(f"{val:4}" for val in row))

def matrix_add(a, b):
    """Add two matrices."""
    rows = len(a)
    cols = len(a[0])
    return [[a[r][c] + b[r][c] for c in range(cols)] for r in range(rows)]

def matrix_transpose(matrix):
    """Transpose a matrix."""
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]

m1 = [[1, 2, 3],
      [4, 5, 6]]

m2 = [[7, 8, 9],
      [10, 11, 12]]

print("Matrix 1:")
print_matrix(m1)

print("\nMatrix 2:")
print_matrix(m2)

print("\nSum:")
print_matrix(matrix_add(m1, m2))
# 8  10  12
# 14  16  18

print("\nTranspose of Matrix 1:")
print_matrix(matrix_transpose(m1))
# 1   4
# 2   5
# 3   6
```

---

## 9. 요약

| 자료구조 | 순서 유지 | 가변 | 중복 허용 | 핵심 특징 |
|----------------|---------|---------|------------|-------------|
| **리스트** | 예 | 예 | 예 | 범용 시퀀스 |
| **튜플** | 예 | 아니오 | 예 | 불변, 해시 가능 |
| **딕셔너리** | 예 (3.7+) | 예 | 키: 아니오 | 키-값 매핑, O(1) 조회 |
| **세트** | 아니오 | 예 | 아니오 | 고유 요소, 집합 연산 |

핵심 요점:
- **리스트**는 순서가 있는 컬렉션의 기본 선택입니다
- **튜플**은 우발적인 수정으로부터 데이터를 보호하고 딕셔너리 키로 사용할 수 있습니다
- **딕셔너리**는 빠른 조회와 구조화된 데이터 표현에 탁월합니다
- **세트**는 고유성과 집합 연산을 효율적으로 처리합니다
- **컴프리헨션**은 네 가지 타입 모두를 구축하는 파이썬다운 방법을 제공합니다
- 중첩 구조를 다룰 때는 항상 **얕은 복사 vs 깊은 복사**를 인식해야 합니다

---

## 연습문제

1. 숫자 리스트를 받아 `"min"`, `"max"`, `"mean"`, `"median"` 키를 가진 딕셔너리를 반환하는 함수를 작성하세요.
2. 두 개의 딕셔너리가 주어졌을 때, 양쪽 모두에 존재하는 키의 딕셔너리를 반환하는 함수를 작성하세요. 값은 `(dict1의_값, dict2의_값)` 튜플이어야 합니다.
3. 딕셔너리를 사용하여 간단한 전화번호부를 구현하세요: 추가, 삭제, 검색, 전체 연락처 목록 기능을 포함합니다.
4. 세트를 사용하여 리스트에서 모든 중복 요소를 찾는 함수를 작성하세요.
5. 학교를 나타내는 중첩 자료구조 (학과 -> 수업 -> 학생)를 생성하고 이를 질의하는 함수를 작성하세요 (예: 학과의 모든 학생 찾기, 수업별 학생 수 세기).

---

**이전**: [함수](./05_Functions.md) | **다음**: [문자열과 텍스트 처리](./07_Strings_and_Text_Processing.md)
