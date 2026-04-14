# 레슨 07: 다형성

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 다형성을 정의하고 왜 OOP에 필수적인지 설명할 수 있다
2. 메서드 오버라이딩을 통한 서브타입 다형성을 구현할 수 있다
3. 덕 타이핑 — Python이 선호하는 다형성 접근법을 적용할 수 있다
4. 매직 메서드를 사용하여 연산자를 오버로딩할 수 있다
5. 구조적 타이핑을 위한 프로토콜(Python 3.8+)을 사용할 수 있다
6. 매개변수적, 애드혹, 서브타입 다형성을 구분할 수 있다
7. 확장 가능한 시스템을 위한 다형적 인터페이스를 설계할 수 있다

## 다형성이란?

다형성(그리스어: "많은 형태")은 **같은 인터페이스**가 객체의 타입에 따라 **다른 동작**을 만들어내는 것을 의미합니다.

```
           shape.area()
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼────┐ ┌──▼─────┐ ┌──▼──────┐
│ Circle │ │ Rect-  │ │Triangle │
│        │ │ angle  │ │         │
│ pi*r^2 │ │ w * h  │ │ 0.5*b*h │
└────────┘ └────────┘ └─────────┘

같은 메서드 이름, 다른 구현
```

## 서브타입 다형성 (상속 기반)

가장 전통적인 형태: 서브클래스가 부모 메서드를 오버라이드합니다:

```python
from math import pi

class Shape:
    def area(self):
        raise NotImplementedError("서브클래스가 area()를 구현해야 합니다")

    def perimeter(self):
        raise NotImplementedError("서브클래스가 perimeter()를 구현해야 합니다")

    def describe(self):
        return f"{self.__class__.__name__}: 면적={self.area():.2f}, 둘레={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return pi * self.radius ** 2

    def perimeter(self):
        return 2 * pi * self.radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# 다형성: 같은 인터페이스, 다른 동작
shapes = [Circle(5), Rectangle(4, 6)]
for shape in shapes:
    print(shape.describe())
```

## 덕 타이핑

Python이 선호하는 다형성 접근법: "오리처럼 걷고 오리처럼 꽥꽥거리면, 그것은 오리다." 상속이 필요 없습니다 — 적절한 메서드만 구현하면 됩니다.

```python
# 공유 기본 클래스가 필요 없음!
class Dog:
    def speak(self):
        return "멍멍!"

class Cat:
    def speak(self):
        return "야옹!"

class Robot:
    def speak(self):
        return "삐 삡!"

# 이 함수는 speak()가 있는 모든 객체와 작동
def make_them_speak(things):
    for thing in things:
        print(f"{thing.__class__.__name__}: {thing.speak()}")

# 공통 부모 없지만 다형성 작동!
make_them_speak([Dog(), Cat(), Robot()])
```

## 연산자 오버로딩

Python에서 연산자(`+`, `-`, `*`, `==` 등)가 객체와 어떻게 작동하는지 정의할 수 있습니다:

```python
class Vector:
    """연산자 오버로딩이 있는 2D 벡터."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)
print(v1 + v2)       # Vector(4, 6)
print(v1 * 3)        # Vector(9, 12)
print(abs(v1))        # 5.0
```

### 일반적으로 오버로딩하는 연산자

```
┌────────────────┬────────────────┬──────────────────┐
│  연산자        │  메서드        │  예시             │
├────────────────┼────────────────┼──────────────────┤
│  +             │  __add__       │  a + b           │
│  -             │  __sub__       │  a - b           │
│  *             │  __mul__       │  a * b           │
│  ==            │  __eq__        │  a == b          │
│  <             │  __lt__        │  a < b           │
│  len()         │  __len__       │  len(a)          │
│  str()         │  __str__       │  str(a)          │
│  []            │  __getitem__   │  a[key]          │
│  in            │  __contains__  │  x in a          │
└────────────────┴────────────────┴──────────────────┘
```

## 프로토콜 (구조적 타이핑)

Python 3.8+에서 `Protocol`을 도입하여 공식적인 덕 타이핑을 제공합니다:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    """draw() 메서드를 가진 모든 객체."""
    def draw(self) -> str:
        ...

# 이 클래스들은 Drawable을 상속하지 않지만 프로토콜을 만족
class Circle:
    def __init__(self, radius):
        self.radius = radius
    def draw(self) -> str:
        return f"반지름 {self.radius}의 원을 그립니다"

class TextBox:
    def __init__(self, text):
        self.text = text
    def draw(self) -> str:
        return f"텍스트를 그립니다: {self.text}"

def render(item: Drawable) -> None:
    print(item.draw())

# 런타임 검사
print(isinstance(Circle(5), Drawable))   # True
print(isinstance(TextBox("hi"), Drawable))  # True
```

## 실전 예제: 결제 처리

```python
class PaymentProcessor:
    def charge(self, amount):
        raise NotImplementedError
    def refund(self, amount):
        raise NotImplementedError

class CreditCard(PaymentProcessor):
    def __init__(self, card_number):
        self.card_number = card_number[-4:]
    def charge(self, amount):
        return f"카드 ***{self.card_number}에 ${amount:.2f} 청구됨"
    def refund(self, amount):
        return f"카드 ***{self.card_number}에 ${amount:.2f} 환불됨"

class PayPal(PaymentProcessor):
    def __init__(self, email):
        self.email = email
    def charge(self, amount):
        return f"PayPal ({self.email})로 ${amount:.2f} 청구됨"
    def refund(self, amount):
        return f"PayPal ({self.email})로 ${amount:.2f} 환불됨"

def process_order(payment: PaymentProcessor, amount: float):
    """이 함수는 결제 유형을 알 필요도 없고 신경 쓸 필요도 없습니다."""
    print(payment.charge(amount))

# 같은 함수, 다른 결제 수단
process_order(CreditCard("4111111111111234"), 99.99)
process_order(PayPal("alice@example.com"), 49.99)
```

## 요약

- 다형성은 객체 타입에 따라 "같은 인터페이스, 다른 동작"을 의미합니다
- **서브타입 다형성**: 서브클래스가 부모 메서드를 오버라이드 (고전적 OOP)
- **덕 타이핑**: Python이 선호하는 스타일 — 상속 없이 메서드 이름만 일치하면 됨
- **연산자 오버로딩**: `__add__`, `__eq__` 등으로 커스텀 연산자 동작 정의
- **프로토콜**: 타입 검사가 되는 구조적 인터페이스 (Python 3.8+)
- 다형성은 확장 가능한 코드를 가능하게 합니다 — 기존 함수를 변경하지 않고 새 타입 추가

## 다음 단계

[레슨 08: 추상화](08_Abstraction.md)에서 서브클래스에 계약을 강제하는 추상 인터페이스를 정의하는 방법을 배웁니다.
