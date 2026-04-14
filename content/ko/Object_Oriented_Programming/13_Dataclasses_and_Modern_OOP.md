# 레슨 13: 데이터클래스와 모던 OOP

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. `@dataclass`를 사용하여 데이터 클래스의 보일러플레이트를 제거할 수 있다
2. 데이터클래스 옵션(frozen, ordering, slots)을 설정할 수 있다
3. 기본 팩토리와 메타데이터를 위한 `field()`를 사용할 수 있다
4. frozen 데이터클래스로 불변 값 객체를 만들 수 있다
5. 경량 불변 레코드를 위한 `NamedTuple`을 사용할 수 있다
6. 상속 없는 구조적 타이핑을 위한 `Protocol`을 적용할 수 있다
7. 데이터클래스, NamedTuple, 일반 클래스 중 적절한 것을 선택할 수 있다

## 보일러플레이트 문제

전통적인 Python 클래스는 반복적인 코드가 필요합니다:

```python
# 전통적인 클래스 — 많은 보일러플레이트
class PointOld:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"PointOld({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        if not isinstance(other, PointOld):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)
```

## 데이터클래스: 현대적 해결책

```python
from dataclasses import dataclass

@dataclass
class Point:
    """같은 기능, 최소한의 코드."""
    x: float
    y: float
    z: float = 0.0

# 자동 생성: __init__, __repr__, __eq__
p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)
print(p1)          # Point(x=1.0, y=2.0, z=0.0)
print(p1 == p2)    # True (자동 __eq__)
```

## 데이터클래스 옵션

```python
from dataclasses import dataclass, field

# Frozen: 불변 (자동으로 __hash__도 생성)
@dataclass(frozen=True)
class Color:
    red: int
    green: int
    blue: int

    def hex(self):
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"

c = Color(255, 128, 0)
# c.red = 0  # FrozenInstanceError! 수정 불가

# 딕셔너리 키나 집합에 사용 가능 (해싱 가능)
colors = {Color(255, 0, 0): "빨강", Color(0, 255, 0): "초록"}


# Ordering: 비교 연산자 생성
@dataclass(order=True)
class Student:
    gpa: float
    name: str = ""

students = [Student(3.5, "Alice"), Student(3.8, "Bob")]
print(sorted(students))
```

## `field()` 함수

고급 필드 설정을 위해:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class ShoppingCart:
    owner: str
    items: List[str] = field(default_factory=list)  # 가변 기본값!
    _total: float = field(default=0.0, repr=False)  # repr에서 숨김
    created: str = field(default="", init=False)     # __init__에 포함 안 됨

    def __post_init__(self):
        """__init__ 후 호출 — 파생/계산 값을 위해."""
        from datetime import datetime
        self.created = datetime.now().isoformat()

    def add_item(self, item, price):
        self.items.append(item)
        self._total += price
```

## `__post_init__`: 검증과 계산 필드

```python
from dataclasses import dataclass

@dataclass
class Temperature:
    celsius: float

    def __post_init__(self):
        """자동 __init__ 후 검증."""
        if self.celsius < -273.15:
            raise ValueError(f"온도 {self.celsius}C는 절대 영도 아래입니다")

    @property
    def fahrenheit(self):
        return self.celsius * 9 / 5 + 32
```

## NamedTuple

`NamedTuple`은 경량의 불변 레코드를 생성합니다 — 이름이 있는 필드로 튜플과 같은 동작이 필요할 때 유용합니다:

```python
from typing import NamedTuple
from math import sqrt

class Vector2D(NamedTuple):
    x: float
    y: float

    @property
    def magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self):
        mag = self.magnitude
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

v = Vector2D(3.0, 4.0)
print(v.magnitude)     # 5.0
print(v.normalized())  # Vector2D(x=0.6, y=0.8)
print(v[0])            # 3.0 (인덱스 접근 — 튜플이니까!)

# 불변
# v.x = 5.0  # AttributeError!
```

## Protocol: 구조적 타이핑

Protocol은 상속이 아닌 구조를 통해 인터페이스를 정의합니다:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Renderable(Protocol):
    """문자열로 렌더링할 수 있는 모든 것."""
    def render(self) -> str:
        ...

# Renderable을 상속하지 않음 — 구조만 일치
class HTMLElement:
    def __init__(self, tag, content):
        self.tag = tag
        self.content = content

    def render(self) -> str:
        return f"<{self.tag}>{self.content}</{self.tag}>"

class MarkdownHeader:
    def __init__(self, text, level=1):
        self.text = text
        self.level = level

    def render(self) -> str:
        return f"{'#' * self.level} {self.text}"

def render_all(items: list[Renderable]) -> str:
    return "\n".join(item.render() for item in items)

print(isinstance(HTMLElement("p", "hi"), Renderable))  # True
```

## 올바른 도구 선택

```
┌─────────────────┬──────────────────────────────────────┐
│  도구           │  적합한 경우                         │
├─────────────────┼──────────────────────────────────────┤
│ 일반 클래스     │ 복잡한 동작, 가변 상태, 커스텀       │
│                 │ __init__, 비데이터 클래스             │
├─────────────────┼──────────────────────────────────────┤
│ @dataclass      │ 동작이 있는 데이터 보유 클래스,      │
│                 │ 기본적으로 가변, 풍부한 기능          │
├─────────────────┼──────────────────────────────────────┤
│ @dataclass      │ 값 객체, 딕셔너리 키, 설정 객체,     │
│ (frozen=True)   │ 불변 레코드                          │
├─────────────────┼──────────────────────────────────────┤
│ NamedTuple      │ 경량 불변 레코드, 튜플 호환성,       │
│                 │ 단순 데이터                          │
├─────────────────┼──────────────────────────────────────┤
│ Protocol        │ 구조적 인터페이스, 타입 검사기        │
│                 │ 지원 덕 타이핑                       │
└─────────────────┴──────────────────────────────────────┘
```

## 요약

- `@dataclass`는 `__init__`, `__repr__`, `__eq__`를 자동 생성하여 보일러플레이트를 제거합니다
- 불변 데이터클래스에는 `frozen=True` 사용 (값 객체, 딕셔너리 키)
- 가변 기본값, 숨김 필드, 계산 값에는 `field()` 사용
- `__post_init__`은 `__init__` 후에 검증과 파생 속성을 위해 실행됩니다
- `NamedTuple`은 불변의 튜플 호환 레코드를 만듭니다
- `Protocol`은 상속 없이 인터페이스를 매칭하는 구조적 타이핑을 가능하게 합니다

## 다음 단계

[레슨 14: OOP 모범 사례](14_OOP_Best_Practices.md)에서 피해야 할 안티패턴과 깔끔한 OOP 코드를 작성하기 위한 실전 가이드라인을 다룹니다.
