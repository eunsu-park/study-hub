# 레슨 10: SOLID 원칙

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 다섯 가지 SOLID 원칙 각각을 명명하고 설명할 수 있다
2. 기존 코드에서 SOLID 원칙 위반을 식별할 수 있다
3. 단일 책임 원칙(SRP)을 따르도록 코드를 리팩토링할 수 있다
4. 확장에는 열려 있고 수정에는 닫힌 클래스를 설계할 수 있다 (OCP)
5. 리스코프 치환 원칙을 적용하여 상속 계층을 검증할 수 있다
6. 인터페이스 분리 원칙에 따라 비대한 인터페이스를 분할할 수 있다
7. 의존성 역전 원칙을 사용하여 의존성을 역전시킬 수 있다

## SOLID란?

SOLID는 Robert C. Martin("Uncle Bob")이 소개한 다섯 가지 설계 원칙의 약어로, 객체 지향 시스템을 더 유지보수 가능하고 유연하며 이해하기 쉽게 만듭니다.

```
┌───┬──────────────────────────────────────────────┐
│ S │ 단일 책임 원칙 (Single Responsibility)       │
│   │ 클래스는 변경 이유가 하나만 있어야 한다      │
├───┼──────────────────────────────────────────────┤
│ O │ 개방/폐쇄 원칙 (Open/Closed)                 │
│   │ 확장에는 열려 있고, 수정에는 닫혀 있어야 한다│
├───┼──────────────────────────────────────────────┤
│ L │ 리스코프 치환 원칙 (Liskov Substitution)     │
│   │ 하위 타입은 상위 타입을 대체할 수 있어야 한다│
├───┼──────────────────────────────────────────────┤
│ I │ 인터페이스 분리 원칙 (Interface Segregation)  │
│   │ 사용하지 않는 메서드에 의존하면 안 된다      │
├───┼──────────────────────────────────────────────┤
│ D │ 의존성 역전 원칙 (Dependency Inversion)      │
│   │ 구체 클래스가 아닌 추상화에 의존해야 한다    │
└───┴──────────────────────────────────────────────┘
```

## S — 단일 책임 원칙 (SRP)

> "클래스는 변경 이유가 하나만 있어야 한다."

```python
# 나쁜 예: 여러 책임을 가진 갓 클래스
class UserManager:
    def validate_email(self): ...    # 검증
    def save_to_database(self): ...  # 영속화
    def send_welcome_email(self): ...# 이메일
    def generate_report(self): ...   # 보고서

# 좋은 예: 각 클래스가 단일 책임
class User:
    """책임: 사용자 데이터와 검증."""
    def validate_email(self): ...

class UserRepository:
    """책임: 사용자 데이터 영속화."""
    def save(self, user): ...

class EmailService:
    """책임: 이메일 발송."""
    def send_welcome(self, user): ...
```

## O — 개방/폐쇄 원칙 (OCP)

> "소프트웨어 엔티티는 확장에는 열려 있고 수정에는 닫혀 있어야 한다."

기존 코드를 **변경하지 않고** 새로운 동작을 추가할 수 있어야 합니다.

```python
# 나쁜 예: 새 도형 추가마다 기존 함수를 수정해야 함
def calculate_area_bad(shape):
    if shape["type"] == "circle":
        return 3.14 * shape["radius"] ** 2
    elif shape["type"] == "rectangle":
        return shape["width"] * shape["height"]
    # 새 도형마다 이 함수를 수정해야 함

# 좋은 예: 새 도형 추가 시 기존 코드 변경 불필요
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * self.radius ** 2

# 이 함수는 절대 변경할 필요 없음
def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)
```

## L — 리스코프 치환 원칙 (LSP)

> "상위 클래스의 객체는 프로그램을 깨뜨리지 않고 하위 클래스의 객체로 대체 가능해야 한다."

```python
# 나쁜 예: LSP 위반 — Square가 Rectangle의 계약을 깨뜨림
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

class Square(Rectangle):
    @Rectangle.width.setter
    def width(self, value):
        self._width = value
        self._height = value  # 예상치 못한 부작용!

# 좋은 예: 별도 클래스 또는 합성 사용
class Shape(ABC):
    @abstractmethod
    def area(self): pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height

class Square(Shape):
    def __init__(self, side):
        self.side = side
    def area(self):
        return self.side ** 2
```

## I — 인터페이스 분리 원칙 (ISP)

> "클라이언트는 사용하지 않는 메서드에 의존하도록 강제되어서는 안 된다."

```python
# 나쁜 예: 비대한 인터페이스
class Worker(ABC):
    @abstractmethod
    def work(self): pass
    @abstractmethod
    def program(self): pass
    @abstractmethod
    def manage(self): pass

# 청소부는 프로그래밍하거나 관리하지 않습니다!

# 좋은 예: 분리된 인터페이스
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Programmable(ABC):
    @abstractmethod
    def program(self): pass

class Manageable(ABC):
    @abstractmethod
    def manage(self): pass

class Janitor(Workable):
    def work(self):
        return "청소 중"

class Developer(Workable, Programmable):
    def work(self):
        return "소프트웨어 개발 중"
    def program(self):
        return "Python 코드 작성 중"
```

## D — 의존성 역전 원칙 (DIP)

> "상위 수준 모듈은 하위 수준 모듈에 의존해서는 안 된다. 둘 다 추상화에 의존해야 한다."

```
┌─────────────────────────────────────────────────┐
│  DIP 없이:                                      │
│  OrderService ──의존──▶ MySQLDatabase            │
│  문제: 데이터베이스 교체 시 OrderService 수정 필요│
├─────────────────────────────────────────────────┤
│  DIP 적용:                                      │
│  OrderService ──의존──▶ Database (ABC)           │
│                            ▲                     │
│                    ┌───────┼───────┐             │
│                  MySQL  Postgres  SQLite          │
└─────────────────────────────────────────────────┘
```

```python
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def query(self, sql: str) -> str:
        pass

class MySQLDatabase(Database):
    def query(self, sql):
        return f"MySQL: {sql}"

class PostgresDatabase(Database):
    def query(self, sql):
        return f"Postgres: {sql}"

class OrderService:
    """상위 수준 모듈이 추상화에 의존합니다."""
    def __init__(self, db: Database):  # 의존성 주입!
        self.db = db

    def get_orders(self):
        return self.db.query("SELECT * FROM orders")

# 구현 교체가 쉬움
service = OrderService(PostgresDatabase())
```

## 요약

- **SRP**: 하나의 클래스, 하나의 책임, 하나의 변경 이유
- **OCP**: 새 클래스를 추가하여 동작을 확장하고, 기존 코드는 수정하지 않기
- **LSP**: 서브클래스는 부모 클래스를 완전히 대체할 수 있어야 함
- **ISP**: 하나의 큰 인터페이스보다 작고 집중된 여러 인터페이스를 선호
- **DIP**: 추상화(인터페이스/ABC)에 의존하고, 구체 구현을 주입
- SOLID 원칙은 함께 작동하여 유지보수 가능하고 테스트 가능하며 확장 가능한 시스템을 만듭니다

## 다음 단계

[레슨 11: 디자인 패턴 입문](11_Design_Patterns_Intro.md)에서 SOLID 원칙을 구현하는 고전적인 디자인 패턴을 탐구합니다.
