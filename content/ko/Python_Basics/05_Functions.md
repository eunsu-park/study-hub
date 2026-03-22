# 함수 (Functions)

**이전**: [제어 흐름](./04_Control_Flow.md) | **다음**: [자료구조](./06_Data_Structures.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `def` 키워드를 사용하여 매개변수, 반환값, 독스트링이 포함된 함수를 정의한다
2. 위치 인수, 키워드 인수, 기본 매개변수, 가변 길이 인수(`*args`, `**kwargs`)를 구분한다
3. 지역, 전역, 둘러싸는 스코프를 포함한 파이썬의 스코프 규칙(LEGB 규칙)을 설명한다
4. 짧은 익명 함수 (Anonymous Function)를 위한 람다 표현식 (Lambda Expression)을 작성하고 고차 함수와 함께 사용한다
5. 필수 내장 함수(`len`, `max`, `min`, `sum`, `sorted`, `map`, `filter`, `zip`, `enumerate`, `any`, `all`)를 활용한다
6. 기본적인 재귀 함수 (Recursive Function)를 구현하고 호출 스택 동작을 이해한다
7. 중첩 함수 (Nested Function)를 생성하고 클로저 (Closure) 기초를 이해한다
8. 파이썬 독스트링 (Docstring) 관례에 따라 잘 문서화된 함수를 작성한다

---

함수는 체계적이고 재사용 가능한 파이썬 코드의 기본 구성 요소입니다. 같은 로직을 반복해서 작성하는 대신 함수로 캡슐화하고, 의미 있는 이름을 부여하고, 필요할 때마다 호출합니다. 함수는 프로그램을 읽기 쉽고, 테스트하기 쉽고, 디버그하기 쉽고, 유지보수하기 쉽게 만듭니다. 간단한 스크립트부터 대규모 웹 애플리케이션까지 모든 본격적인 파이썬 프로그램은 함께 동작하는 함수들로 구성됩니다.

## 1. 함수 정의

함수는 `def` 키워드 뒤에 함수 이름, 매개변수를 위한 괄호, 콜론을 사용하여 생성합니다. 그 아래 들여쓴 블록이 함수 본문입니다.

```python
# Basic function definition
def greet():
    print("Hello, World!")

# Calling the function
greet()  # Output: Hello, World!
```

### 매개변수가 있는 함수

매개변수 (Parameter)는 함수가 작업을 수행하는 데 필요한 값의 자리표시자입니다. 함수를 호출할 때 그 자리표시자를 채우는 인수 (Argument)를 전달합니다.

```python
def greet_user(name):
    print(f"Hello, {name}!")

greet_user("Alice")   # Output: Hello, Alice!
greet_user("Bob")     # Output: Hello, Bob!
```

### 여러 매개변수

```python
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")

add(3, 5)       # Output: 3 + 5 = 8
add(10, 20)     # Output: 10 + 20 = 30
```

### 매개변수와 인수의 차이

- **매개변수 (Parameter)**는 함수 정의에 나열된 이름입니다
- **인수 (Argument)**는 함수 호출 시 전달되는 실제 값입니다

```python
def multiply(x, y):    # x and y are parameters
    return x * y

result = multiply(3, 4) # 3 and 4 are arguments
print(result)            # Output: 12
```

---

## 2. 반환값

함수는 `return` 문을 사용하여 호출자에게 결과를 돌려보낼 수 있습니다. 명시적인 `return`이 없으면 함수는 `None`을 반환합니다.

```python
def add(a, b):
    return a + b

result = add(5, 3)
print(result)       # Output: 8

# A function without return gives None
def greet(name):
    print(f"Hello, {name}!")

value = greet("Alice")   # Output: Hello, Alice!
print(value)             # Output: None
```

### 여러 값 반환

파이썬 함수는 튜플 (Tuple)로 여러 값을 반환할 수 있습니다.

```python
def divide_and_remainder(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide_and_remainder(17, 5)
print(f"Quotient: {q}, Remainder: {r}")  # Output: Quotient: 3, Remainder: 2

# The return is actually a tuple
result = divide_and_remainder(17, 5)
print(result)        # Output: (3, 2)
print(type(result))  # Output: <class 'tuple'>
```

### 조기 반환

`return`은 즉시 함수를 종료합니다. 같은 블록에서 `return` 이후의 코드는 실행되지 않습니다.

```python
def absolute_value(n):
    if n >= 0:
        return n
    return -n

print(absolute_value(5))    # Output: 5
print(absolute_value(-3))   # Output: 3


def find_first_negative(numbers):
    for num in numbers:
        if num < 0:
            return num
    return None   # No negative found

data = [3, 7, -2, 5, -8]
print(find_first_negative(data))  # Output: -2
print(find_first_negative([1, 2, 3]))  # Output: None
```

### 다른 타입 반환

파이썬은 다른 타입의 반환을 허용하지만, 반환 타입을 일관되게 유지하는 것이 모범 사례입니다.

```python
def safe_divide(a, b):
    if b == 0:
        return None   # Signals an error condition
    return a / b

result = safe_divide(10, 3)
if result is not None:
    print(f"Result: {result:.2f}")  # Output: Result: 3.33
```

---

## 3. 기본 매개변수

기본 매개변수 값을 사용하면 모든 인수를 제공하지 않고도 함수를 호출할 수 있습니다.

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")               # Output: Hello, Alice!
greet("Bob", "Good morning") # Output: Good morning, Bob!
```

### 여러 기본값

```python
def create_profile(name, age=0, city="Unknown", active=True):
    return {
        "name": name,
        "age": age,
        "city": city,
        "active": active,
    }

print(create_profile("Alice"))
# Output: {'name': 'Alice', 'age': 0, 'city': 'Unknown', 'active': True}

print(create_profile("Bob", 25, "Seoul"))
# Output: {'name': 'Bob', 'age': 25, 'city': 'Seoul', 'active': True}
```

### 가변 기본 인수 함정

흔한 실수는 리스트와 같은 가변 객체 (Mutable Object)를 기본값으로 사용하는 것입니다. 기본값은 한 번만 생성되어 모든 호출에서 공유됩니다.

```python
# BAD: mutable default argument
def add_item_bad(item, items=[]):
    items.append(item)
    return items

print(add_item_bad("a"))  # Output: ['a']
print(add_item_bad("b"))  # Output: ['a', 'b']  -- Unexpected!

# GOOD: use None as default, create new list inside
def add_item_good(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item_good("a"))  # Output: ['a']
print(add_item_good("b"))  # Output: ['b']  -- Correct!
```

---

## 4. 키워드 인수

함수를 호출할 때 이름으로 인수를 지정할 수 있습니다. 이렇게 하면 호출이 더 명확해지고 인수를 순서에 관계없이 전달할 수 있습니다.

```python
def describe_pet(animal, name, age):
    print(f"{name} is a {age}-year-old {animal}.")

# Positional arguments (order matters)
describe_pet("dog", "Rex", 5)

# Keyword arguments (order does not matter)
describe_pet(name="Whiskers", age=3, animal="cat")

# Mix of positional and keyword (positional must come first)
describe_pet("hamster", name="Pip", age=1)
```

### 키워드 전용 인수 강제

매개변수 목록에서 `*`를 사용하면 이후의 모든 매개변수를 키워드 전용으로 강제할 수 있습니다.

```python
def connect(host, port, *, timeout=30, use_ssl=False):
    print(f"Connecting to {host}:{port}")
    print(f"  timeout={timeout}, ssl={use_ssl}")

connect("localhost", 8080)                          # OK
connect("localhost", 8080, timeout=60, use_ssl=True) # OK
# connect("localhost", 8080, 60, True)  # TypeError! timeout and use_ssl are keyword-only
```

### 위치 전용 매개변수 (Python 3.8+)

`/`를 사용하면 그 앞의 매개변수를 위치 전용으로 강제합니다.

```python
def power(base, exp, /):
    return base ** exp

print(power(2, 10))    # Output: 1024
# power(base=2, exp=10)  # TypeError! base and exp are positional-only
```

### 위치 전용, 일반, 키워드 전용 결합

```python
def example(pos_only, /, regular, *, kw_only):
    print(f"pos_only={pos_only}, regular={regular}, kw_only={kw_only}")

example(1, 2, kw_only=3)          # OK
example(1, regular=2, kw_only=3)  # OK
# example(pos_only=1, regular=2, kw_only=3)  # TypeError!
```

---

## 5. 가변 길이 인수: `*args`와 `**kwargs`

### `*args` -- 가변 위치 인수

`*args`는 추가 위치 인수를 튜플로 수집합니다.

```python
def add_all(*args):
    print(f"args = {args}")
    return sum(args)

print(add_all(1, 2, 3))        # args = (1, 2, 3) -> Output: 6
print(add_all(10, 20, 30, 40)) # args = (10, 20, 30, 40) -> Output: 100
```

### `**kwargs` -- 가변 키워드 인수

`**kwargs`는 추가 키워드 인수를 딕셔너리로 수집합니다.

```python
def print_info(**kwargs):
    print(f"kwargs = {kwargs}")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

print_info(name="Alice", age=30, city="Seoul")
# kwargs = {'name': 'Alice', 'age': 30, 'city': 'Seoul'}
#   name: Alice
#   age: 30
#   city: Seoul
```

### `*args`와 `**kwargs` 결합

```python
def universal_function(*args, **kwargs):
    print(f"Positional: {args}")
    print(f"Keyword: {kwargs}")

universal_function(1, 2, 3, name="Alice", active=True)
# Positional: (1, 2, 3)
# Keyword: {'name': 'Alice', 'active': True}
```

### 인수 언패킹

함수를 호출할 때 시퀀스와 딕셔너리를 언패킹할 수 있습니다.

```python
def greet(first, last, greeting="Hello"):
    print(f"{greeting}, {first} {last}!")

# Unpack a list/tuple with *
names = ["Alice", "Smith"]
greet(*names)  # Output: Hello, Alice Smith!

# Unpack a dictionary with **
config = {"first": "Bob", "last": "Jones", "greeting": "Good morning"}
greet(**config)  # Output: Good morning, Bob Jones!
```

### 실용적인 예제: 유연한 로거

```python
def log(level, message, *tags, **metadata):
    tag_str = ", ".join(tags) if tags else "none"
    meta_str = " | ".join(f"{k}={v}" for k, v in metadata.items())
    print(f"[{level.upper()}] {message}")
    print(f"  Tags: {tag_str}")
    if meta_str:
        print(f"  Meta: {meta_str}")

log("info", "User logged in", "auth", "security", user="alice", ip="192.168.1.1")
# [INFO] User logged in
#   Tags: auth, security
#   Meta: user=alice | ip=192.168.1.1
```

---

## 6. 독스트링

독스트링 (Docstring)은 함수 본문의 첫 번째 문장으로 나타나는 문자열 리터럴입니다. 함수가 무엇을 하는지 설명하며 `help()` 함수와 `__doc__` 속성을 통해 접근할 수 있습니다.

```python
def calculate_area(length, width):
    """Calculate the area of a rectangle.

    Args:
        length: The length of the rectangle (must be positive).
        width: The width of the rectangle (must be positive).

    Returns:
        The area as a float.

    Raises:
        ValueError: If length or width is negative.
    """
    if length < 0 or width < 0:
        raise ValueError("Dimensions must be non-negative")
    return length * width

# Accessing the docstring
help(calculate_area)
print(calculate_area.__doc__)
```

### 독스트링 스타일

```python
# Google style (shown above)
def func_google(param1, param2):
    """Summary line.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.
    """
    pass

# NumPy/SciPy style
def func_numpy(param1, param2):
    """Summary line.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.
    """
    pass

# reStructuredText style (Sphinx)
def func_rst(param1, param2):
    """Summary line.

    :param param1: Description of param1.
    :type param1: int
    :param param2: Description of param2.
    :type param2: str
    :returns: Description of return value.
    :rtype: bool
    """
    pass
```

---

## 7. 스코프: 지역 vs 전역 (LEGB 규칙)

파이썬은 LEGB 규칙을 사용하여 이름을 해석합니다: **L**ocal (지역), **E**nclosing (둘러싸는), **G**lobal (전역), **B**uilt-in (내장).

```python
# Global scope
x = "global"

def outer():
    # Enclosing scope
    x = "enclosing"

    def inner():
        # Local scope
        x = "local"
        print(f"inner: {x}")    # Output: local

    inner()
    print(f"outer: {x}")        # Output: enclosing

outer()
print(f"module: {x}")           # Output: global
```

### `global` 키워드

`global` 키워드는 함수가 전역 변수를 수정할 수 있게 합니다.

```python
counter = 0

def increment():
    global counter
    counter += 1

increment()
increment()
print(counter)  # Output: 2
```

### `nonlocal` 키워드

`nonlocal` 키워드는 내부 함수가 둘러싸는 스코프의 변수를 수정할 수 있게 합니다.

```python
def make_counter():
    count = 0

    def increment():
        nonlocal count
        count += 1
        return count

    return increment

counter = make_counter()
print(counter())  # Output: 1
print(counter())  # Output: 2
print(counter())  # Output: 3
```

### 스코프 함정: UnboundLocalError

```python
x = 10

def broken():
    # Python sees x = ... below, so it treats x as local in the ENTIRE function
    # Trying to read x before assigning it causes an error
    # print(x)  # UnboundLocalError: local variable 'x' referenced before assignment
    x = 20
    print(x)

broken()       # Output: 20
print(x)       # Output: 10 (global x unchanged)
```

---

## 8. 중첩 함수

다른 함수 안에 정의된 함수입니다. 둘러싸는 스코프의 변수에 접근할 수 있습니다.

```python
def greet_builder(greeting):
    def greet(name):
        return f"{greeting}, {name}!"
    return greet

hello = greet_builder("Hello")
hi = greet_builder("Hi")

print(hello("Alice"))  # Output: Hello, Alice!
print(hi("Bob"))       # Output: Hi, Bob!
```

### 실용적 활용: 유효성 검사 래퍼

```python
def validated_operation(operation_name):
    def validate_and_run(a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return f"Error: {operation_name} requires numeric inputs"
        if operation_name == "divide" and b == 0:
            return "Error: cannot divide by zero"

        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b,
        }
        return operations.get(operation_name, "Unknown operation")

    return validate_and_run

divide = validated_operation("divide")
print(divide(10, 3))    # Output: 3.3333333333333335
print(divide(10, 0))    # Output: Error: cannot divide by zero
print(divide(10, "a"))  # Output: Error: divide requires numeric inputs
```

### 클로저

클로저 (Closure)는 외부 함수가 실행을 마친 후에도 둘러싸는 스코프의 변수를 기억하는 중첩 함수입니다.

```python
def create_multiplier(factor):
    def multiply(number):
        return number * factor   # factor is "closed over"
    return multiply

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))   # Output: 10
print(triple(5))   # Output: 15

# Inspect closure variables
print(double.__closure__[0].cell_contents)  # Output: 2
```

---

## 9. 람다 표현식

람다 표현식 (Lambda Expression)은 한 줄로 작은 익명 함수를 생성합니다. 짧은 기간 동안 간단한 함수가 필요할 때 유용합니다.

```python
# Regular function
def add(a, b):
    return a + b

# Equivalent lambda
add_lambda = lambda a, b: a + b

print(add(3, 5))         # Output: 8
print(add_lambda(3, 5))  # Output: 8
```

### 정렬에서의 람다

```python
students = [
    {"name": "Alice", "grade": 88},
    {"name": "Bob", "grade": 95},
    {"name": "Charlie", "grade": 72},
    {"name": "Diana", "grade": 91},
]

# Sort by grade
by_grade = sorted(students, key=lambda s: s["grade"])
print([s["name"] for s in by_grade])
# Output: ['Charlie', 'Alice', 'Diana', 'Bob']

# Sort by name length
by_name_len = sorted(students, key=lambda s: len(s["name"]))
print([s["name"] for s in by_name_len])
# Output: ['Bob', 'Alice', 'Diana', 'Charlie']
```

### 조건식이 있는 람다

```python
classify = lambda x: "positive" if x > 0 else ("negative" if x < 0 else "zero")

print(classify(5))    # Output: positive
print(classify(-3))   # Output: negative
print(classify(0))    # Output: zero
```

### 람다 vs def 사용 시점

| 람다 사용 | def 사용 |
|------------|---------|
| 간단한 단일 표현식 | 여러 문장이나 복잡한 로직 |
| `sorted`, `map`, `filter`의 인수로 사용 | 명확성을 위해 이름이 필요한 함수 |
| 일회성 사용 | 재사용되거나 테스트될 함수 |
| 로직이 즉시 명확한 경우 | 독스트링이 필요한 경우 |

---

## 10. 내장 함수 개요

파이썬은 많은 유용한 내장 함수를 제공합니다. 이들을 알면 바퀴를 재발명하지 않아도 됩니다.

### `len()` -- 길이

```python
print(len("Hello"))       # Output: 5
print(len([1, 2, 3, 4]))  # Output: 4
print(len({"a": 1, "b": 2}))  # Output: 2
```

### `max()`와 `min()` -- 최댓값과 최솟값

```python
print(max(3, 7, 2, 9))          # Output: 9
print(min(3, 7, 2, 9))          # Output: 2
print(max([10, 20, 5]))         # Output: 20

# With key function
words = ["apple", "hi", "banana", "cat"]
print(max(words, key=len))      # Output: banana
print(min(words, key=len))      # Output: hi
```

### `sum()` -- 이터러블의 합

```python
print(sum([1, 2, 3, 4, 5]))     # Output: 15
print(sum(range(1, 101)))       # Output: 5050

# With start value
print(sum([1, 2, 3], 10))       # Output: 16 (10 + 1 + 2 + 3)
```

### `sorted()`와 `reversed()`

```python
numbers = [3, 1, 4, 1, 5, 9]

# sorted returns a new list
print(sorted(numbers))              # Output: [1, 1, 3, 4, 5, 9]
print(sorted(numbers, reverse=True)) # Output: [9, 5, 4, 3, 1, 1]

# reversed returns an iterator
print(list(reversed(numbers)))  # Output: [9, 5, 1, 4, 1, 3]

# Sort with custom key
names = ["Charlie", "alice", "Bob"]
print(sorted(names, key=str.lower))  # Output: ['alice', 'Bob', 'Charlie']
```

### `map()` -- 각 요소에 함수 적용

`map()`은 이터러블의 모든 항목에 함수를 적용하고 이터레이터를 반환합니다.

```python
numbers = [1, 2, 3, 4, 5]

# Square each number
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # Output: [1, 4, 9, 16, 25]

# Convert strings to integers
str_numbers = ["10", "20", "30"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # Output: [10, 20, 30]

# Multiple iterables
a = [1, 2, 3]
b = [10, 20, 30]
sums = list(map(lambda x, y: x + y, a, b))
print(sums)  # Output: [11, 22, 33]
```

### `filter()` -- 조건으로 요소 선택

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Keep even numbers only
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # Output: [2, 4, 6, 8, 10]

# Remove empty strings
words = ["hello", "", "world", "", "python"]
non_empty = list(filter(None, words))  # None removes falsy values
print(non_empty)  # Output: ['hello', 'world', 'python']
```

### `zip()` -- 이터러블 결합

```python
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]

# Pair elements together
pairs = list(zip(names, scores))
print(pairs)
# Output: [('Alice', 85), ('Bob', 92), ('Charlie', 78)]

# Common pattern: create a dictionary
score_dict = dict(zip(names, scores))
print(score_dict)
# Output: {'Alice': 85, 'Bob': 92, 'Charlie': 78}

# Unzipping with zip(*)
paired = [("a", 1), ("b", 2), ("c", 3)]
letters, nums = zip(*paired)
print(letters)  # Output: ('a', 'b', 'c')
print(nums)     # Output: (1, 2, 3)
```

### `enumerate()` -- 인덱스 + 요소

```python
fruits = ["apple", "banana", "cherry"]

# Instead of manual indexing
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# Custom start index
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")
# 1. apple
# 2. banana
# 3. cherry
```

### `any()`와 `all()`

```python
numbers = [2, 4, 6, 8, 10]

# all: True if ALL elements are truthy (or iterable is empty)
print(all(n > 0 for n in numbers))    # Output: True
print(all(n % 2 == 0 for n in numbers))  # Output: True

# any: True if ANY element is truthy
print(any(n > 5 for n in numbers))    # Output: True
print(any(n > 100 for n in numbers))  # Output: False

# Practical example: validation
def validate_user(name, email, age):
    checks = [
        len(name) > 0,
        "@" in email,
        age >= 18,
    ]
    return all(checks)

print(validate_user("Alice", "alice@example.com", 25))  # Output: True
print(validate_user("", "alice@example.com", 25))        # Output: False
```

### 비교: map/filter vs 리스트 컴프리헨션

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map + filter approach
result1 = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# List comprehension approach (generally preferred in Python)
result2 = [x ** 2 for x in numbers if x % 2 == 0]

print(result1)  # Output: [4, 16, 36, 64, 100]
print(result2)  # Output: [4, 16, 36, 64, 100]
```

---

## 11. 재귀 기초

재귀 함수 (Recursive Function)는 자기 자신을 호출합니다. 모든 재귀 함수는 무한 재귀를 방지하기 위한 **기저 사례** (Base Case, 종료 조건)가 필요합니다.

### 팩토리얼

```python
def factorial(n):
    """Calculate n! recursively."""
    if n <= 1:       # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

print(factorial(5))   # Output: 120 (5 * 4 * 3 * 2 * 1)
print(factorial(0))   # Output: 1
```

### 호출 스택 동작 방식

```
factorial(4)
  -> 4 * factorial(3)
       -> 3 * factorial(2)
            -> 2 * factorial(1)
                 -> return 1        # Base case
            -> return 2 * 1 = 2
       -> return 3 * 2 = 6
  -> return 4 * 6 = 24
```

### 피보나치 수열

```python
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(10):
    print(fibonacci(i), end=" ")
# Output: 0 1 1 2 3 5 8 13 21 34
print()
```

### 리스트의 합 (재귀)

```python
def recursive_sum(lst):
    """Sum all elements in a list recursively."""
    if not lst:        # Base case: empty list
        return 0
    return lst[0] + recursive_sum(lst[1:])

print(recursive_sum([1, 2, 3, 4, 5]))  # Output: 15
```

### 재귀 vs 반복

```python
# Recursive countdown
def countdown_recursive(n):
    if n <= 0:
        print("Go!")
        return
    print(n)
    countdown_recursive(n - 1)

# Iterative countdown
def countdown_iterative(n):
    while n > 0:
        print(n)
        n -= 1
    print("Go!")

countdown_recursive(3)
# 3
# 2
# 1
# Go!
```

### 재귀 제한

파이썬은 스택 오버플로를 방지하기 위해 기본 재귀 제한(보통 1000)이 있습니다.

```python
import sys

print(sys.getrecursionlimit())  # Output: 1000 (default)

# You can change it, but be careful
# sys.setrecursionlimit(2000)

# For deep recursion, prefer iterative solutions
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print(factorial_iterative(100))  # Works fine, no recursion limit issue
```

---

## 12. 종합 예제

### 예제: 텍스트 분석 함수

```python
def analyze_text(text, *, case_sensitive=False, min_word_length=1):
    """Analyze text and return word statistics.

    Args:
        text: The input text to analyze.
        case_sensitive: Whether to treat uppercase and lowercase
            as different words. Defaults to False.
        min_word_length: Minimum word length to include. Defaults to 1.

    Returns:
        A dictionary containing word count, unique words,
        and word frequency.
    """
    if not case_sensitive:
        text = text.lower()

    words = text.split()
    words = [w for w in words if len(w) >= min_word_length]

    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    return {
        "total_words": len(words),
        "unique_words": len(frequency),
        "most_common": max(frequency, key=frequency.get) if frequency else None,
        "frequency": dict(sorted(frequency.items(), key=lambda x: -x[1])),
    }

sample = "the quick brown fox jumps over the lazy dog the fox"
result = analyze_text(sample, min_word_length=3)
print(f"Total words: {result['total_words']}")
print(f"Unique words: {result['unique_words']}")
print(f"Most common: {result['most_common']}")
print(f"Frequency: {result['frequency']}")
# Total words: 9
# Unique words: 7
# Most common: the
# Frequency: {'the': 3, 'fox': 2, 'quick': 1, 'brown': 1, 'jumps': 1, 'over': 1, 'lazy': 1}
```

### 예제: 유연한 데이터 프로세서

```python
def process_data(data, *transformations, verbose=False):
    """Apply a series of transformations to data.

    Args:
        data: The input list of numbers.
        *transformations: Functions to apply in sequence.
        verbose: If True, print intermediate results.

    Returns:
        The transformed data as a list.
    """
    result = list(data)

    for i, transform in enumerate(transformations):
        result = list(map(transform, result))
        if verbose:
            print(f"  Step {i + 1} ({transform.__name__}): {result}")

    return result

def double(x):
    return x * 2

def add_one(x):
    return x + 1

def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
output = process_data(numbers, double, add_one, square, verbose=True)
#   Step 1 (double): [2, 4, 6, 8, 10]
#   Step 2 (add_one): [3, 5, 7, 9, 11]
#   Step 3 (square): [9, 25, 49, 81, 121]
print(f"Final: {output}")
# Final: [9, 25, 49, 81, 121]
```

---

## 13. 요약

| 개념 | 핵심 포인트 |
|---------|------------|
| `def` | 매개변수와 본문으로 함수를 정의 |
| `return` | 값을 반환; 생략하면 `None` |
| 기본 매개변수 | 대체 값을 제공; 가변 기본값 사용을 피할 것 |
| `*args` | 추가 위치 인수를 튜플로 수집 |
| `**kwargs` | 추가 키워드 인수를 딕셔너리로 수집 |
| 독스트링 | 함수 본문의 첫 번째 문자열; Google/NumPy 스타일 사용 |
| 스코프 (LEGB) | Local > Enclosing > Global > Built-in |
| `global`/`nonlocal` | 외부 스코프의 변수를 수정 |
| 람다 | 익명 단일 표현식 함수 |
| 내장 함수 | `len`, `max`, `min`, `sum`, `sorted`, `map`, `filter`, `zip`, `enumerate`, `any`, `all` |
| 재귀 | 함수가 자기 자신을 호출; 항상 기저 사례를 정의할 것 |

---

## 연습문제

1. 재귀를 사용하여 `base ** exp`를 계산하는 함수 `power(base, exp)`를 작성하세요 (`**` 연산자 사용 불가).
2. `[[1, 2], [3, [4, 5]], 6]`과 같은 중첩 리스트를 받아 `[1, 2, 3, 4, 5, 6]`을 반환하는 함수 `flatten(nested_list)`를 재귀를 사용하여 작성하세요.
3. 유효성 검사 함수를 반환하는 `make_validator(**rules)` 함수를 생성하세요. 검사기는 딕셔너리를 규칙에 대해 검사해야 합니다 (예: `make_validator(name=str, age=int)`는 타입을 검사하는 함수를 반환).
4. `map`, `filter`, `zip`을 사용하여: 두 개의 이름 리스트와 나이 리스트가 주어졌을 때, 18세 이상인 사람의 이름 목록을 생성하세요.
5. `func()`를 최대 `attempts`번까지 호출하여 성공 시 결과를 반환하고 실패 시 `None`을 반환하는 데코레이터 유사 함수 `retry(func, attempts=3)`를 작성하세요.

---

**이전**: [제어 흐름](./04_Control_Flow.md) | **다음**: [자료구조](./06_Data_Structures.md)
