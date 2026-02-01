# 함수형 프로그래밍 (Functional Programming)

## 1. 함수형 프로그래밍이란?

함수형 프로그래밍은 순수 함수와 불변 데이터를 중심으로 프로그램을 구성하는 패러다임입니다.

### 핵심 원칙

| 원칙 | 설명 |
|------|------|
| 순수 함수 | 같은 입력 → 같은 출력, 부작용 없음 |
| 불변성 | 데이터를 변경하지 않고 새로 생성 |
| 일급 함수 | 함수를 값처럼 전달/반환 |
| 선언적 | "어떻게"보다 "무엇을" 기술 |

### 명령형 vs 선언적

```python
numbers = [1, 2, 3, 4, 5]

# 명령형 (어떻게)
result = []
for n in numbers:
    if n % 2 == 0:
        result.append(n ** 2)
print(result)  # [4, 16]

# 선언적/함수형 (무엇을)
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))
print(result)  # [4, 16]

# 더 파이썬다운 방식 (리스트 컴프리헨션)
result = [n ** 2 for n in numbers if n % 2 == 0]
print(result)  # [4, 16]
```

---

## 2. 일급 함수 (First-Class Function)

파이썬에서 함수는 일급 객체입니다.

### 변수에 할당

```python
def greet(name):
    return f"Hello, {name}!"

# 함수를 변수에 할당
say_hello = greet
print(say_hello("Python"))  # Hello, Python!

# 함수도 객체
print(type(greet))  # <class 'function'>
print(greet.__name__)  # greet
```

### 함수를 인자로 전달

```python
def apply_twice(func, value):
    return func(func(value))

def add_ten(x):
    return x + 10

result = apply_twice(add_ten, 5)
print(result)  # 25 (5 + 10 + 10)
```

### 함수를 반환

```python
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

---

## 3. 고차 함수 (Higher-Order Function)

함수를 인자로 받거나 함수를 반환하는 함수입니다.

### map()

모든 요소에 함수를 적용합니다.

```python
numbers = [1, 2, 3, 4, 5]

# 제곱
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# 문자열 변환
words = ["hello", "world"]
upper_words = list(map(str.upper, words))
print(upper_words)  # ['HELLO', 'WORLD']

# 여러 이터러블
a = [1, 2, 3]
b = [10, 20, 30]
sums = list(map(lambda x, y: x + y, a, b))
print(sums)  # [11, 22, 33]
```

### filter()

조건을 만족하는 요소만 선택합니다.

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 짝수만
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# 빈 문자열 제거
words = ["hello", "", "world", "", "python"]
non_empty = list(filter(None, words))  # falsy 값 제거
print(non_empty)  # ['hello', 'world', 'python']

# 사용자 정의 조건
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 17},
    {"name": "Charlie", "age": 25},
]
adults = list(filter(lambda u: u["age"] >= 18, users))
print(adults)  # [{'name': 'Alice', ...}, {'name': 'Charlie', ...}]
```

### reduce()

요소들을 누적하여 단일 값으로 줄입니다.

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# 합계
total = reduce(lambda acc, x: acc + x, numbers)
print(total)  # 15

# 곱
product = reduce(lambda acc, x: acc * x, numbers)
print(product)  # 120

# 초기값 지정
total_with_init = reduce(lambda acc, x: acc + x, numbers, 100)
print(total_with_init)  # 115

# 최댓값 찾기
max_val = reduce(lambda a, b: a if a > b else b, numbers)
print(max_val)  # 5
```

### map + filter + reduce 조합

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 짝수의 제곱의 합
result = reduce(
    lambda acc, x: acc + x,
    map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers))
)
print(result)  # 220 (4 + 16 + 36 + 64 + 100)
```

---

## 4. lambda 표현식

익명 함수를 간결하게 정의합니다.

### 기본 문법

```python
# lambda 인자: 표현식
add = lambda x, y: x + y
print(add(3, 4))  # 7

# 기본값
greet = lambda name="World": f"Hello, {name}!"
print(greet())        # Hello, World!
print(greet("Python"))  # Hello, Python!

# 가변 인자
sum_all = lambda *args: sum(args)
print(sum_all(1, 2, 3, 4))  # 10
```

### 정렬에 활용

```python
# 튜플 정렬
points = [(1, 2), (3, 1), (5, 4), (2, 2)]

# y 좌표로 정렬
sorted_by_y = sorted(points, key=lambda p: p[1])
print(sorted_by_y)  # [(3, 1), (1, 2), (2, 2), (5, 4)]

# 딕셔너리 정렬
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
]
sorted_by_age = sorted(users, key=lambda u: u["age"])
print([u["name"] for u in sorted_by_age])  # ['Bob', 'Alice', 'Charlie']
```

### lambda 주의사항

```python
# 복잡한 로직은 일반 함수로
# 나쁜 예
complex_op = lambda x: (x ** 2 + 3 * x - 5) if x > 0 else (x ** 2 - 3 * x + 5)

# 좋은 예
def complex_operation(x):
    if x > 0:
        return x ** 2 + 3 * x - 5
    return x ** 2 - 3 * x + 5
```

---

## 5. functools 모듈

함수형 프로그래밍을 위한 도구 모음입니다.

### partial (부분 적용)

함수의 일부 인자를 고정합니다.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# 제곱 함수
square = partial(power, exponent=2)
print(square(5))  # 25

# 세제곱 함수
cube = partial(power, exponent=3)
print(cube(5))  # 125

# 밑이 2인 거듭제곱
power_of_two = partial(power, 2)
print(power_of_two(10))  # 1024
```

### lru_cache (메모이제이션)

함수 결과를 캐싱합니다.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 캐시 덕분에 빠름
print(fibonacci(100))  # 354224848179261915075

# 캐시 정보
print(fibonacci.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# 캐시 초기화
fibonacci.cache_clear()
```

### cache (Python 3.9+)

무제한 캐시 (lru_cache(maxsize=None)와 동일).

```python
from functools import cache

@cache
def factorial(n):
    return n * factorial(n - 1) if n else 1

print(factorial(10))  # 3628800
```

### wraps (데코레이터 메타데이터 보존)

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper docstring"""
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def say_hello():
    """Say hello function"""
    return "Hello!"

# 원본 함수 정보 유지
print(say_hello.__name__)  # say_hello
print(say_hello.__doc__)   # Say hello function
```

### singledispatch (함수 오버로딩)

타입에 따라 다른 함수를 호출합니다.

```python
from functools import singledispatch

@singledispatch
def process(value):
    raise NotImplementedError(f"Cannot process {type(value)}")

@process.register(int)
def _(value):
    return f"Processing integer: {value * 2}"

@process.register(str)
def _(value):
    return f"Processing string: {value.upper()}"

@process.register(list)
def _(value):
    return f"Processing list of {len(value)} items"

print(process(10))       # Processing integer: 20
print(process("hello"))  # Processing string: HELLO
print(process([1, 2]))   # Processing list of 2 items
```

---

## 6. operator 모듈

연산자를 함수로 사용할 수 있게 합니다.

### 산술 연산자

```python
from operator import add, sub, mul, truediv, mod, pow
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# lambda 대신 operator 사용
total = reduce(add, numbers)
print(total)  # 15

product = reduce(mul, numbers)
print(product)  # 120
```

### 비교 연산자

```python
from operator import lt, le, eq, ne, ge, gt

print(lt(3, 5))  # True (3 < 5)
print(eq(3, 3))  # True (3 == 3)
print(ge(5, 3))  # True (5 >= 3)
```

### itemgetter, attrgetter

```python
from operator import itemgetter, attrgetter

# itemgetter - 인덱스/키로 접근
data = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
sorted_data = sorted(data, key=itemgetter(1))  # 나이로 정렬
print(sorted_data)  # [('Bob', 25), ('Alice', 30), ('Charlie', 35)]

# 딕셔너리에서 사용
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
]
get_name = itemgetter("name")
names = list(map(get_name, users))
print(names)  # ['Alice', 'Bob']

# 여러 필드
get_info = itemgetter("name", "age")
print(get_info(users[0]))  # ('Alice', 30)

# attrgetter - 속성 접근
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person("Alice", 30), Person("Bob", 25)]
sorted_people = sorted(people, key=attrgetter("age"))
print([p.name for p in sorted_people])  # ['Bob', 'Alice']
```

### methodcaller

```python
from operator import methodcaller

words = ["hello", "WORLD", "Python"]

# 메서드 호출
upper = methodcaller("upper")
print(list(map(upper, words)))  # ['HELLO', 'WORLD', 'PYTHON']

# 인자가 있는 메서드
replace_o = methodcaller("replace", "o", "0")
print(replace_o("hello world"))  # hell0 w0rld
```

---

## 7. 순수 함수와 불변성

### 순수 함수

```python
# 순수 함수: 부작용 없음
def pure_add(a, b):
    return a + b

# 비순수 함수: 외부 상태 변경
total = 0
def impure_add(value):
    global total
    total += value
    return total

# 비순수 함수: 입력 데이터 변경
def impure_append(lst, value):
    lst.append(value)  # 원본 변경
    return lst

# 순수 버전
def pure_append(lst, value):
    return lst + [value]  # 새 리스트 반환
```

### 불변성 유지

```python
# 리스트: 새 리스트 생성
original = [1, 2, 3]
modified = original + [4]  # 원본 유지
modified = [*original, 4]  # 언패킹

# 딕셔너리: 새 딕셔너리 생성
config = {"debug": True, "port": 8080}
new_config = {**config, "port": 9090}  # 원본 유지

# 불변 자료구조
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p1 = Point(1, 2)
p2 = p1._replace(x=10)  # 새 객체 생성
print(p1)  # Point(x=1, y=2)
print(p2)  # Point(x=10, y=2)
```

### frozenset과 tuple

```python
# 불변 집합
mutable_set = {1, 2, 3}
immutable_set = frozenset([1, 2, 3])

# 딕셔너리 키로 사용 가능
cache = {frozenset([1, 2]): "result"}

# 불변 리스트 (tuple)
mutable_list = [1, 2, 3]
immutable_list = (1, 2, 3)
```

---

## 8. 컴프리헨션

파이썬스러운 함수형 패턴입니다.

### 리스트 컴프리헨션

```python
# map 대체
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# filter 대체
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# map + filter
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# 중첩
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 딕셔너리/집합 컴프리헨션

```python
# 딕셔너리 컴프리헨션
names = ["Alice", "Bob", "Charlie"]
name_lengths = {name: len(name) for name in names}
print(name_lengths)  # {'Alice': 5, 'Bob': 3, 'Charlie': 7}

# 집합 컴프리헨션
unique_lengths = {len(name) for name in names}
print(unique_lengths)  # {3, 5, 7}

# 딕셔너리 키/값 뒤집기
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}
```

### 제너레이터 표현식

```python
# 메모리 효율적인 지연 평가
large_squares = (x ** 2 for x in range(1000000))
print(next(large_squares))  # 0
print(next(large_squares))  # 1

# sum, any, all과 함께
numbers = [1, 2, 3, 4, 5]
total = sum(x ** 2 for x in numbers)  # 괄호 생략 가능
print(total)  # 55

# 조건 검사
all_positive = all(x > 0 for x in numbers)
any_even = any(x % 2 == 0 for x in numbers)
print(all_positive, any_even)  # True True
```

---

## 9. 파이프라인 패턴

함수를 연결하여 데이터 처리 파이프라인을 만듭니다.

### 함수 합성

```python
def compose(*functions):
    """오른쪽에서 왼쪽으로 함수 합성"""
    def composed(x):
        for func in reversed(functions):
            x = func(x)
        return x
    return composed

def pipe(*functions):
    """왼쪽에서 오른쪽으로 함수 합성"""
    def piped(x):
        for func in functions:
            x = func(x)
        return x
    return piped

# 사용 예
add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x ** 2

# compose: square(double(add_one(5))) = square(double(6)) = square(12) = 144
composed = compose(square, double, add_one)
print(composed(5))  # 144

# pipe: square(double(add_one(5)))와 동일하지만 읽기 쉬움
piped = pipe(add_one, double, square)
print(piped(5))  # 144
```

### 데이터 처리 파이프라인

```python
from functools import reduce

# 파이프 연산자 흉내
class Pipe:
    def __init__(self, value):
        self.value = value

    def __or__(self, func):
        return Pipe(func(self.value))

    def __repr__(self):
        return repr(self.value)

# 사용
result = (
    Pipe([1, 2, 3, 4, 5])
    | (lambda x: [n * 2 for n in x])      # 두 배
    | (lambda x: [n for n in x if n > 4]) # 4 초과만
    | sum                                  # 합계
)
print(result)  # 24 (6 + 8 + 10)
```

### 실용적인 파이프라인

```python
from functools import partial

def process_data(data):
    """데이터 처리 파이프라인"""
    pipeline = [
        # 1. 문자열 정리
        lambda items: [s.strip().lower() for s in items],
        # 2. 빈 문자열 제거
        lambda items: [s for s in items if s],
        # 3. 중복 제거 (순서 유지)
        lambda items: list(dict.fromkeys(items)),
        # 4. 정렬
        sorted,
    ]

    result = data
    for transform in pipeline:
        result = transform(result)
    return result

raw_data = ["  Hello ", "world", "  ", "HELLO", "Python", "world"]
print(process_data(raw_data))  # ['hello', 'python', 'world']
```

---

## 10. itertools 활용

### 조합/순열

```python
from itertools import permutations, combinations, product

# 순열
print(list(permutations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# 조합
print(list(combinations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 3)]

# 데카르트 곱
print(list(product([1, 2], ['a', 'b'])))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
```

### 그룹화

```python
from itertools import groupby

data = [
    ("fruit", "apple"),
    ("fruit", "banana"),
    ("vegetable", "carrot"),
    ("vegetable", "daikon"),
    ("fruit", "elderberry"),
]

# 정렬 후 그룹화 (연속된 동일 키만 그룹화됨)
sorted_data = sorted(data, key=lambda x: x[0])
for key, group in groupby(sorted_data, key=lambda x: x[0]):
    print(f"{key}: {[item[1] for item in group]}")
# fruit: ['apple', 'banana', 'elderberry']
# vegetable: ['carrot', 'daikon']
```

### 무한 이터레이터

```python
from itertools import count, cycle, repeat, islice

# count: 무한 카운터
for i in islice(count(10, 2), 5):
    print(i, end=" ")  # 10 12 14 16 18

# cycle: 무한 반복
colors = cycle(["red", "green", "blue"])
for _ in range(5):
    print(next(colors), end=" ")  # red green blue red green

# repeat: 값 반복
threes = list(repeat(3, 4))
print(threes)  # [3, 3, 3, 3]
```

### 체이닝과 슬라이싱

```python
from itertools import chain, islice, takewhile, dropwhile

# chain: 여러 이터러블 연결
combined = list(chain([1, 2], [3, 4], [5, 6]))
print(combined)  # [1, 2, 3, 4, 5, 6]

# islice: 슬라이싱
first_three = list(islice(range(10), 3))
print(first_three)  # [0, 1, 2]

# takewhile: 조건이 참인 동안
nums = [1, 3, 5, 8, 2, 4]
result = list(takewhile(lambda x: x < 6, nums))
print(result)  # [1, 3, 5]

# dropwhile: 조건이 참인 동안 건너뛰기
result = list(dropwhile(lambda x: x < 6, nums))
print(result)  # [8, 2, 4]
```

---

## 11. 실전 예제

### 데이터 변환

```python
from functools import reduce
from operator import itemgetter

# 원본 데이터
orders = [
    {"product": "apple", "quantity": 10, "price": 1.5},
    {"product": "banana", "quantity": 5, "price": 0.8},
    {"product": "apple", "quantity": 3, "price": 1.5},
    {"product": "orange", "quantity": 8, "price": 2.0},
]

# 제품별 총 매출 계산
from itertools import groupby

def calculate_sales(orders):
    # 제품별로 정렬 및 그룹화
    sorted_orders = sorted(orders, key=itemgetter("product"))

    result = {}
    for product, group in groupby(sorted_orders, key=itemgetter("product")):
        total = sum(o["quantity"] * o["price"] for o in group)
        result[product] = total

    return result

print(calculate_sales(orders))
# {'apple': 19.5, 'banana': 4.0, 'orange': 16.0}
```

### 함수형 검증

```python
from functools import reduce
from typing import Callable, List, Any

def validate(value: Any, *validators: Callable) -> tuple[bool, list[str]]:
    """여러 검증 함수를 순차 적용"""
    errors = []
    for validator in validators:
        result = validator(value)
        if result is not None:
            errors.append(result)
    return len(errors) == 0, errors

# 검증 함수들
def not_empty(s): return "Cannot be empty" if not s else None
def min_length(n): return lambda s: f"Min length is {n}" if len(s) < n else None
def max_length(n): return lambda s: f"Max length is {n}" if len(s) > n else None

# 사용
is_valid, errors = validate(
    "hi",
    not_empty,
    min_length(3),
    max_length(10)
)
print(is_valid, errors)  # False ['Min length is 3']
```

### 커링 (Currying)

```python
def curry(func):
    """함수를 커링"""
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more: curried(*args, *more)
    return curried

@curry
def add_three(a, b, c):
    return a + b + c

# 다양한 호출 방식
print(add_three(1, 2, 3))     # 6
print(add_three(1)(2)(3))     # 6
print(add_three(1, 2)(3))     # 6
print(add_three(1)(2, 3))     # 6
```

---

## 12. 요약

| 개념 | 설명 |
|------|------|
| 일급 함수 | 함수를 값처럼 전달/반환 |
| map/filter/reduce | 고차 함수로 컬렉션 변환 |
| lambda | 익명 함수 |
| partial | 부분 적용 |
| lru_cache | 메모이제이션 |
| operator | 연산자 함수화 |
| 순수 함수 | 부작용 없는 함수 |
| 불변성 | 데이터 변경 대신 새로 생성 |
| 파이프라인 | 함수 체이닝 |

---

## 13. 연습 문제

### 연습 1: 함수 합성

두 함수를 받아 합성된 함수를 반환하는 `compose2` 함수를 작성하세요.

### 연습 2: 트랜스듀서

`map`과 `filter`를 조합하여 한 번의 순회로 처리하는 함수를 작성하세요.

### 연습 3: 메모이제이션

직접 메모이제이션 데코레이터를 구현하세요.

---

## 다음 단계

[10_Performance_Optimization.md](./10_Performance_Optimization.md)에서 파이썬 성능 최적화 기법을 배워봅시다!
