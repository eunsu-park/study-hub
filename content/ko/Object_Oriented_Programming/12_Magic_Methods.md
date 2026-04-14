# 레슨 12: 매직 메서드

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 매직 메서드(던더 메서드)가 무엇이고 Python이 어떻게 사용하는지 설명할 수 있다
2. 사람이 읽기 쉬운 표현과 디버그 표현을 위한 `__str__`과 `__repr__`을 구현할 수 있다
3. `__eq__`와 `__hash__`로 동등성과 해싱을 정의할 수 있다
4. `__iter__`와 `__next__`로 이터러블 객체를 만들 수 있다
5. `__getitem__`, `__len__`, `__contains__`로 컨테이너 프로토콜을 구현할 수 있다
6. `__call__`로 호출 가능 객체를 만들 수 있다
7. `__enter__`와 `__exit__`로 컨텍스트 매니저를 사용할 수 있다

## 매직 메서드란?

매직 메서드(또는 **던더 메서드** — "double underscore")는 이중 언더스코어로 둘러싸인 특수 메서드로, 연산자, 내장 함수, 언어 구조에 대응하여 Python이 자동으로 호출합니다.

```
┌────────────────────────────────────────────┐
│  Python 구문       →  호출되는 매직 메서드  │
├────────────────────────────────────────────┤
│  str(obj)          →  obj.__str__()        │
│  repr(obj)         →  obj.__repr__()       │
│  len(obj)          →  obj.__len__()        │
│  obj[key]          →  obj.__getitem__(key) │
│  obj == other      →  obj.__eq__(other)    │
│  for x in obj:     →  obj.__iter__()       │
│  obj(args)         →  obj.__call__(args)   │
│  with obj as x:    →  obj.__enter__()      │
└────────────────────────────────────────────┘
```

## `__str__`과 `__repr__`

```
┌──────────────────────────────────────────────────┐
│  __repr__: 모호하지 않은, 개발자용               │
│  - 가능하면 유효한 Python처럼 보여야 함          │
│  - repr(), 디버거, 컨테이너가 사용               │
│                                                  │
│  __str__: 읽기 쉬운, 최종 사용자용               │
│  - 친근하고 읽기 쉬운 출력                        │
│  - str(), print(), f-string이 사용               │
│  - 정의되지 않으면 __repr__로 대체               │
└──────────────────────────────────────────────────┘
```

```python
class Product:
    def __init__(self, name, price, sku):
        self.name = name
        self.price = price
        self.sku = sku

    def __repr__(self):
        return f"Product({self.name!r}, {self.price}, {self.sku!r})"

    def __str__(self):
        return f"{self.name} - ${self.price:.2f}"

p = Product("Laptop", 999.99, "SKU-001")
print(repr(p))   # Product('Laptop', 999.99, 'SKU-001')
print(str(p))    # Laptop - $999.99
print([p])       # [Product('Laptop', 999.99, 'SKU-001')]
```

## `__eq__`와 `__hash__`

```python
class Card:
    """커스텀 동등성을 가진 카드."""

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        """__eq__를 정의하면 __hash__도 구현해야 합니다.
        동등한 객체는 반드시 같은 해시를 가져야 합니다.
        """
        return hash((self.rank, self.suit))

c1 = Card("A", "Spades")
c2 = Card("A", "Spades")
c3 = Card("K", "Hearts")

print(c1 == c2)   # True
print(c1 == c3)   # False

hand = {c1, c2, c3}
print(len(hand))  # 2 (c1과 c2가 동등하므로 중복 제거)
```

## 순서: `__lt__`, `@total_ordering`

```python
from functools import total_ordering

@total_ordering
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __eq__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius == other.celsius

    def __lt__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius

    def __hash__(self):
        return hash(self.celsius)

    def __repr__(self):
        return f"Temperature({self.celsius}C)"

temps = [Temperature(100), Temperature(0), Temperature(37)]
print(sorted(temps))  # [Temperature(0C), Temperature(37C), Temperature(100C)]
```

## `__iter__`와 `__next__`

`for` 루프와 작동하는 객체 만들기:

```python
class Fibonacci:
    """이터러블 피보나치 수열."""

    def __init__(self, max_count):
        self.max_count = max_count

    def __iter__(self):
        return FibonacciIterator(self.max_count)

class FibonacciIterator:
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.max_count:
            raise StopIteration
        value = self.a
        self.a, self.b = self.b, self.a + self.b
        self.count += 1
        return value

fib = Fibonacci(10)
print(list(fib))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

## 컨테이너 프로토콜

```python
class Matrix:
    """컨테이너 프로토콜을 지원하는 2D 행렬."""

    def __init__(self, rows):
        self._data = [list(row) for row in rows]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self._data[row][col]
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            self._data[row][col] = value

    def __len__(self):
        return len(self._data)

    def __contains__(self, value):
        return any(value in row for row in self._data)

    def __iter__(self):
        return iter(self._data)

m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(m[0, 1])      # 2
print(len(m))        # 3
print(5 in m)        # True
```

## `__call__`: 호출 가능 객체

```python
class Validator:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, value):
        if not self.min_val <= value <= self.max_val:
            raise ValueError(f"{value}는 [{self.min_val}, {self.max_val}] 범위에 없습니다")
        return True

validate_age = Validator(0, 150)
print(validate_age(25))  # True
```

## 컨텍스트 매니저: `__enter__`와 `__exit__`

```python
class Timer:
    """실행 시간을 측정하는 컨텍스트 매니저."""

    def __init__(self, label="Block"):
        self.label = label

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.perf_counter() - self._start
        print(f"{self.label}: {self.elapsed:.4f}초")
        return False

with Timer("계산"):
    total = sum(range(1_000_000))
```

## 요약

- 매직 메서드는 Python의 구문과 내장 함수가 객체와 상호작용하는 방식을 커스터마이즈합니다
- `__repr__`은 개발자용(모호하지 않은), `__str__`은 사용자용(읽기 쉬운)
- `__eq__`와 `__hash__`는 계약을 따라야 합니다: 동등한 객체는 같은 해시를 가져야 함
- `__iter__`와 `__next__`는 `for` 루프와 작동하게 합니다
- `__getitem__`, `__len__`, `__contains__`는 컨테이너 프로토콜을 구현합니다
- `__call__`은 객체를 함수처럼 호출 가능하게 합니다
- `__enter__`와 `__exit__`는 컨텍스트 매니저를 구현합니다

## 다음 단계

[레슨 13: 데이터클래스와 모던 OOP](13_Dataclasses_and_Modern_OOP.md)에서 OOP 보일러플레이트를 줄이는 Python의 현대적 도구를 탐구합니다.
