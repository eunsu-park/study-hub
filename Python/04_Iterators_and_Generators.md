# 이터레이터와 제너레이터 (Iterators & Generators)

## 1. 이터러블과 이터레이터

### 개념 구분

| 용어 | 설명 | 예시 |
|------|------|------|
| Iterable | `__iter__` 메서드가 있는 객체 | list, str, dict, set |
| Iterator | `__iter__`와 `__next__` 메서드가 있는 객체 | iter(list), 파일 객체 |

```
┌──────────────────────────────────────────┐
│              Iterable                     │
│  ┌────────────────────────────────────┐  │
│  │    Iterator                         │  │
│  │    __iter__() → self               │  │
│  │    __next__() → 다음 값 or StopIteration │
│  └────────────────────────────────────┘  │
│  __iter__() → Iterator 반환              │
└──────────────────────────────────────────┘
```

### for문의 동작 원리

```python
# for item in iterable:
#     ...

# 위 코드는 아래와 동일
iterator = iter(iterable)  # __iter__() 호출
while True:
    try:
        item = next(iterator)  # __next__() 호출
        # 루프 본문 실행
    except StopIteration:
        break
```

### 예시

```python
numbers = [1, 2, 3]

# iter()로 이터레이터 생성
iterator = iter(numbers)

print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration 예외!
```

---

## 2. 커스텀 이터레이터

### __iter__와 __next__ 구현

```python
class Counter:
    """1부터 max까지 카운트하는 이터레이터"""

    def __init__(self, max_count):
        self.max_count = max_count
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current > self.max_count:
            raise StopIteration
        return self.current

# 사용
for num in Counter(5):
    print(num, end=" ")  # 1 2 3 4 5
```

### 이터러블과 이터레이터 분리

재사용 가능한 이터러블을 만들려면 분리가 필요합니다.

```python
class Range:
    """재사용 가능한 range"""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return RangeIterator(self.start, self.end)

class RangeIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# 여러 번 순회 가능
r = Range(1, 4)
print(list(r))  # [1, 2, 3]
print(list(r))  # [1, 2, 3] (다시 사용 가능)
```

---

## 3. 제너레이터 함수

`yield` 키워드를 사용하면 간단하게 이터레이터를 만들 수 있습니다.

### 기본 문법

```python
def count_up_to(max_count):
    count = 1
    while count <= max_count:
        yield count  # 값을 반환하고 일시 정지
        count += 1

# 사용
for num in count_up_to(5):
    print(num, end=" ")  # 1 2 3 4 5

# 또는
gen = count_up_to(3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
```

### 동작 원리

```
count_up_to(3) 호출
    │
    ▼
┌─────────────────┐
│  count = 1      │
│  while True:    │
│    yield 1 ─────┼──▶ 반환, 일시 정지
│                 │
│  (next 호출)    │
│    count = 2    │
│    yield 2 ─────┼──▶ 반환, 일시 정지
│                 │
│  (next 호출)    │
│    count = 3    │
│    yield 3 ─────┼──▶ 반환, 일시 정지
│                 │
│  (next 호출)    │
│    while 종료   │
│  StopIteration  │
└─────────────────┘
```

### 여러 값을 yield

```python
def multi_yield():
    yield "첫 번째"
    yield "두 번째"
    yield "세 번째"

for value in multi_yield():
    print(value)
```

---

## 4. 제너레이터 표현식

리스트 컴프리헨션과 비슷하지만 괄호를 사용합니다.

```python
# 리스트 컴프리헨션 - 메모리에 전체 저장
squares_list = [x**2 for x in range(10)]

# 제너레이터 표현식 - 필요할 때 생성
squares_gen = (x**2 for x in range(10))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>

# 제너레이터는 한 번만 순회 가능
print(list(squares_gen))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(list(squares_gen))  # [] (이미 소진됨)
```

### 메모리 효율성

```python
import sys

# 리스트: 전체 메모리 사용
list_comp = [x for x in range(1000000)]
print(sys.getsizeof(list_comp))  # ~8MB

# 제너레이터: 최소 메모리
gen_exp = (x for x in range(1000000))
print(sys.getsizeof(gen_exp))    # ~200 bytes
```

---

## 5. yield from

다른 이터러블의 값을 위임합니다.

```python
def chain(*iterables):
    for it in iterables:
        yield from it  # for item in it: yield item 과 동일

result = list(chain([1, 2], [3, 4], [5, 6]))
print(result)  # [1, 2, 3, 4, 5, 6]
```

### 재귀적 제너레이터

```python
def flatten(nested):
    """중첩 리스트를 평탄화"""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, 3, [4, 5]], 6, [7]]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7]
```

---

## 6. 제너레이터 고급 기능

### send() - 값 전달

제너레이터에 값을 보낼 수 있습니다.

```python
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

gen = accumulator()
print(next(gen))      # 0 (초기화)
print(gen.send(10))   # 10
print(gen.send(20))   # 30
print(gen.send(5))    # 35
```

### throw() - 예외 전달

```python
def generator():
    try:
        yield 1
        yield 2
        yield 3
    except ValueError as e:
        yield f"예외 처리됨: {e}"

gen = generator()
print(next(gen))              # 1
print(gen.throw(ValueError, "테스트"))  # 예외 처리됨: 테스트
```

### close() - 제너레이터 종료

```python
def generator():
    try:
        yield 1
        yield 2
        yield 3
    finally:
        print("정리 작업")

gen = generator()
print(next(gen))  # 1
gen.close()       # "정리 작업" 출력
```

---

## 7. itertools 모듈

효율적인 이터레이터 도구를 제공합니다.

### 무한 이터레이터

```python
from itertools import count, cycle, repeat

# count: 무한 카운터
for i in count(10, 2):  # 10부터 2씩 증가
    if i > 20:
        break
    print(i, end=" ")  # 10 12 14 16 18 20

# cycle: 무한 반복
colors = cycle(["빨강", "파랑", "초록"])
for _ in range(5):
    print(next(colors), end=" ")  # 빨강 파랑 초록 빨강 파랑

# repeat: 반복
for item in repeat("Hello", 3):
    print(item)  # Hello Hello Hello
```

### 조합 이터레이터

```python
from itertools import chain, zip_longest, product, permutations, combinations

# chain: 여러 이터러블 연결
print(list(chain([1, 2], [3, 4])))  # [1, 2, 3, 4]

# zip_longest: 길이가 다른 이터러블 묶기
a = [1, 2, 3]
b = ["a", "b"]
print(list(zip_longest(a, b, fillvalue="-")))
# [(1, 'a'), (2, 'b'), (3, '-')]

# product: 데카르트 곱
print(list(product("AB", [1, 2])))
# [('A', 1), ('A', 2), ('B', 1), ('B', 2)]

# permutations: 순열
print(list(permutations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# combinations: 조합
print(list(combinations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'C')]
```

### 필터링 이터레이터

```python
from itertools import takewhile, dropwhile, filterfalse, compress

numbers = [1, 3, 5, 2, 4, 6]

# takewhile: 조건이 참인 동안
print(list(takewhile(lambda x: x < 5, numbers)))  # [1, 3]

# dropwhile: 조건이 참인 동안 건너뛰기
print(list(dropwhile(lambda x: x < 5, numbers)))  # [5, 2, 4, 6]

# filterfalse: 조건이 거짓인 것만
print(list(filterfalse(lambda x: x % 2, numbers)))  # [2, 4, 6]

# compress: 선택자로 필터링
data = ["A", "B", "C", "D"]
selectors = [1, 0, 1, 0]
print(list(compress(data, selectors)))  # ['A', 'C']
```

### 그룹화

```python
from itertools import groupby

data = [
    {"name": "Alice", "dept": "HR"},
    {"name": "Bob", "dept": "IT"},
    {"name": "Charlie", "dept": "HR"},
    {"name": "David", "dept": "IT"},
]

# 정렬 필수!
data.sort(key=lambda x: x["dept"])

for dept, group in groupby(data, key=lambda x: x["dept"]):
    print(f"{dept}: {[p['name'] for p in group]}")
# HR: ['Alice', 'Charlie']
# IT: ['Bob', 'David']
```

### 슬라이싱

```python
from itertools import islice

# 무한 이터레이터에서 일부만 추출
from itertools import count

first_10 = list(islice(count(1), 10))
print(first_10)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 시작, 끝, 스텝 지정
result = list(islice(range(100), 10, 20, 2))
print(result)  # [10, 12, 14, 16, 18]
```

---

## 8. 지연 평가 (Lazy Evaluation)

제너레이터는 값을 미리 계산하지 않고 필요할 때 생성합니다.

### 대용량 파일 처리

```python
def read_large_file(filepath):
    """한 줄씩 읽는 제너레이터"""
    with open(filepath, "r") as f:
        for line in f:
            yield line.strip()

# 메모리 효율적으로 처리
for line in read_large_file("huge_file.txt"):
    if "ERROR" in line:
        print(line)
```

### 파이프라인 처리

```python
def numbers():
    for i in range(1, 1000001):
        yield i

def even_only(nums):
    for n in nums:
        if n % 2 == 0:
            yield n

def squared(nums):
    for n in nums:
        yield n ** 2

def less_than(nums, limit):
    for n in nums:
        if n >= limit:
            break
        yield n

# 파이프라인: 메모리 효율적
pipeline = less_than(squared(even_only(numbers())), 100)
print(list(pipeline))  # [4, 16, 36, 64]
```

---

## 9. 무한 시퀀스

### 피보나치 수열

```python
def fibonacci():
    """무한 피보나치 수열"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 처음 10개
from itertools import islice
print(list(islice(fibonacci(), 10)))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### 소수 생성기

```python
def primes():
    """무한 소수 생성"""
    yield 2
    candidate = 3
    found = [2]
    while True:
        if all(candidate % p != 0 for p in found):
            found.append(candidate)
            yield candidate
        candidate += 2

# 처음 10개 소수
from itertools import islice
print(list(islice(primes(), 10)))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

---

## 10. 요약

| 개념 | 설명 |
|------|------|
| Iterable | `__iter__` 구현, for문에 사용 가능 |
| Iterator | `__iter__` + `__next__` 구현 |
| Generator | `yield`를 사용하는 함수 |
| Generator Expression | `(x for x in iterable)` |
| `yield from` | 다른 이터러블 위임 |
| `send()` | 제너레이터에 값 전달 |
| Lazy Evaluation | 필요할 때 값 생성 |

---

## 11. 연습 문제

### 연습 1: 청크 분할

리스트를 지정된 크기의 청크로 나누는 제너레이터를 작성하세요.

```python
# chunk([1,2,3,4,5], 2) → [1,2], [3,4], [5]
```

### 연습 2: 윈도우 슬라이딩

슬라이딩 윈도우를 생성하는 제너레이터를 작성하세요.

```python
# sliding_window([1,2,3,4,5], 3) → (1,2,3), (2,3,4), (3,4,5)
```

### 연습 3: 트리 순회

이진 트리를 순회하는 제너레이터를 작성하세요.

---

## 다음 단계

[05_Closures_and_Scope.md](./05_Closures_and_Scope.md)에서 변수 스코프와 클로저를 배워봅시다!
