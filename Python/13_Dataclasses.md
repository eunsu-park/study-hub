# 13. 데이터클래스 (Dataclasses)

## 학습 목표
- dataclasses 모듈의 목적과 장점 이해
- @dataclass 데코레이터와 옵션 마스터
- field() 함수를 활용한 고급 필드 설정
- 상속, 불변성, 슬롯 등 고급 기능 활용
- Pydantic과의 비교 및 선택 기준 이해

## 목차
1. [데이터클래스 기초](#1-데이터클래스-기초)
2. [@dataclass 옵션](#2-dataclass-옵션)
3. [field() 함수](#3-field-함수)
4. [고급 기능](#4-고급-기능)
5. [상속과 조합](#5-상속과-조합)
6. [Pydantic 비교](#6-pydantic-비교)
7. [연습 문제](#7-연습-문제)

---

## 1. 데이터클래스 기초

### 1.1 데이터클래스란?

```python
# 일반 클래스로 데이터 구조 정의
class PersonManual:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def __repr__(self):
        return f"PersonManual(name={self.name!r}, age={self.age}, email={self.email!r})"

    def __eq__(self, other):
        if not isinstance(other, PersonManual):
            return NotImplemented
        return (self.name, self.age, self.email) == (other.name, other.age, other.email)


# 데이터클래스로 동일한 기능 (간결!)
from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int
    email: str
```

### 1.2 자동 생성 메서드

```python
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


# 자동 생성되는 메서드들:

# 1. __init__
p = Point(3.0, 4.0)

# 2. __repr__
print(p)  # Point(x=3.0, y=4.0)

# 3. __eq__
p1 = Point(3.0, 4.0)
p2 = Point(3.0, 4.0)
print(p1 == p2)  # True

# 비교에 사용되는 것은 모든 필드
p3 = Point(3.0, 5.0)
print(p1 == p3)  # False
```

### 1.3 기본값 설정

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    name: str
    email: str
    age: int = 0                    # 기본값
    active: bool = True             # 기본값
    nickname: Optional[str] = None  # Optional with default


# 기본값 있는 필드는 기본값 없는 필드 뒤에 와야 함
user = User("Alice", "alice@example.com")
print(user)  # User(name='Alice', email='alice@example.com', age=0, active=True, nickname=None)

user2 = User("Bob", "bob@example.com", age=30, nickname="bobby")
```

---

## 2. @dataclass 옵션

### 2.1 주요 옵션

```python
@dataclass(
    init=True,          # __init__ 생성 (기본: True)
    repr=True,          # __repr__ 생성 (기본: True)
    eq=True,            # __eq__ 생성 (기본: True)
    order=False,        # 비교 메서드 생성 (기본: False)
    unsafe_hash=False,  # __hash__ 생성 (기본: False)
    frozen=False,       # 불변 객체 (기본: False)
    match_args=True,    # 패턴 매칭 지원 (Python 3.10+)
    kw_only=False,      # 모든 필드 키워드 전용 (Python 3.10+)
    slots=False,        # __slots__ 사용 (Python 3.10+)
)
class MyClass:
    pass
```

### 2.2 order - 비교 연산자

```python
@dataclass(order=True)
class Student:
    name: str
    grade: float
    age: int


students = [
    Student("Alice", 3.8, 20),
    Student("Bob", 3.5, 22),
    Student("Charlie", 3.8, 21),
]

# 정렬 (모든 필드 순서대로 비교)
sorted_students = sorted(students)
# name으로 먼저 비교, 같으면 grade, 같으면 age

# 특정 필드로 정렬
sorted_by_grade = sorted(students, key=lambda s: s.grade, reverse=True)
```

### 2.3 frozen - 불변 객체

```python
@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float


point = ImmutablePoint(1.0, 2.0)

# 수정 시도시 에러
try:
    point.x = 3.0
except AttributeError as e:
    print(f"Error: {e}")  # cannot assign to field 'x'

# 불변이므로 해시 가능 (dict 키, set 요소로 사용 가능)
points = {point: "origin"}
point_set = {point, ImmutablePoint(3.0, 4.0)}
```

### 2.4 slots - 메모리 최적화

```python
@dataclass(slots=True)  # Python 3.10+
class OptimizedPoint:
    x: float
    y: float


# slots=True 장점:
# 1. 메모리 사용량 감소
# 2. 속성 접근 속도 향상
# 3. __dict__ 없음 (동적 속성 추가 불가)

point = OptimizedPoint(1.0, 2.0)
# point.z = 3.0  # AttributeError - slots에 없는 속성


# 메모리 비교
import sys

@dataclass
class RegularPoint:
    x: float
    y: float

regular = RegularPoint(1.0, 2.0)
optimized = OptimizedPoint(1.0, 2.0)

print(sys.getsizeof(regular.__dict__))  # dict 크기
# OptimizedPoint는 __dict__ 없음
```

### 2.5 kw_only - 키워드 전용 인자

```python
@dataclass(kw_only=True)  # Python 3.10+
class Config:
    host: str
    port: int
    debug: bool = False


# 반드시 키워드로 전달
config = Config(host="localhost", port=8080)
# config = Config("localhost", 8080)  # TypeError
```

---

## 3. field() 함수

### 3.1 field() 기본 사용

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class Team:
    name: str
    members: List[str] = field(default_factory=list)  # 가변 기본값
    score: int = field(default=0)


# 잘못된 방법 (가변 객체 직접 할당)
# members: List[str] = []  # 모든 인스턴스가 같은 리스트 공유!

team1 = Team("Alpha")
team2 = Team("Beta")
team1.members.append("Alice")
print(team2.members)  # [] - 올바르게 분리됨
```

### 3.2 field() 옵션

```python
from dataclasses import dataclass, field


@dataclass
class Product:
    # 기본 필드
    name: str

    # 기본값 팩토리
    tags: list = field(default_factory=list)

    # repr에서 제외
    internal_id: str = field(repr=False, default="")

    # 비교에서 제외
    cache: dict = field(compare=False, default_factory=dict)

    # __init__에서 제외 (후처리로 설정)
    computed: str = field(init=False)

    # 해시에서 제외 (frozen=True 시)
    mutable_data: list = field(hash=False, default_factory=list)

    # 메타데이터 추가
    price: float = field(metadata={"unit": "USD", "min": 0})

    def __post_init__(self):
        self.computed = f"{self.name}_computed"


product = Product("Widget", price=9.99)
print(product)  # internal_id 숨김
print(product.computed)  # Widget_computed

# 메타데이터 접근
from dataclasses import fields
for f in fields(Product):
    if f.metadata:
        print(f"{f.name}: {f.metadata}")
```

### 3.3 __post_init__ 활용

```python
from dataclasses import dataclass, field


@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    perimeter: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)


rect = Rectangle(5, 3)
print(f"Area: {rect.area}")  # 15
print(f"Perimeter: {rect.perimeter}")  # 16
```

### 3.4 InitVar - 초기화 전용 변수

```python
from dataclasses import dataclass, field, InitVar


@dataclass
class User:
    name: str
    email: str
    password: InitVar[str]  # 초기화에만 사용, 필드로 저장 안 됨
    password_hash: str = field(init=False)

    def __post_init__(self, password: str):
        import hashlib
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()


user = User("Alice", "alice@example.com", "secret123")
print(user)  # password 없음, password_hash 있음
# User(name='Alice', email='alice@example.com', password_hash='...')

# user.password  # AttributeError - InitVar는 저장되지 않음
```

---

## 4. 고급 기능

### 4.1 데이터클래스 유틸리티 함수

```python
from dataclasses import dataclass, field, fields, asdict, astuple, replace


@dataclass
class Person:
    name: str
    age: int
    email: str = ""


person = Person("Alice", 30, "alice@example.com")

# fields(): 필드 정보 조회
for f in fields(person):
    print(f"Name: {f.name}, Type: {f.type}, Default: {f.default}")

# asdict(): 딕셔너리로 변환
person_dict = asdict(person)
print(person_dict)  # {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# astuple(): 튜플로 변환
person_tuple = astuple(person)
print(person_tuple)  # ('Alice', 30, 'alice@example.com')

# replace(): 일부 필드만 변경한 새 인스턴스 생성
person2 = replace(person, name="Bob", age=25)
print(person2)  # Person(name='Bob', age=25, email='alice@example.com')
```

### 4.2 커스텀 __hash__

```python
@dataclass(eq=True, frozen=True)
class HashableItem:
    id: int
    name: str
    # frozen=True면 자동으로 __hash__ 생성


# 또는 수동 해시
@dataclass(eq=True)
class CustomHashItem:
    id: int
    name: str
    data: list  # 가변 필드

    def __hash__(self):
        return hash(self.id)  # id만으로 해시


item = CustomHashItem(1, "test", [1, 2, 3])
items_set = {item}  # 해시 가능
```

### 4.3 JSON 직렬화

```python
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from typing import List


@dataclass
class Event:
    name: str
    date: datetime
    attendees: List[str]


def serialize_event(event: Event) -> str:
    def encoder(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(asdict(event), default=encoder)


def deserialize_event(json_str: str) -> Event:
    data = json.loads(json_str)
    data["date"] = datetime.fromisoformat(data["date"])
    return Event(**data)


event = Event("Meeting", datetime.now(), ["Alice", "Bob"])
json_str = serialize_event(event)
print(json_str)

restored = deserialize_event(json_str)
print(restored)
```

---

## 5. 상속과 조합

### 5.1 데이터클래스 상속

```python
from dataclasses import dataclass


@dataclass
class Animal:
    name: str
    age: int


@dataclass
class Dog(Animal):
    breed: str
    is_trained: bool = False


dog = Dog("Buddy", 3, "Labrador")
print(dog)  # Dog(name='Buddy', age=3, breed='Labrador', is_trained=False)
```

### 5.2 기본값과 상속 주의점

```python
from dataclasses import dataclass, field


@dataclass
class Base:
    x: int
    y: int = 0  # 기본값 있음


# 오류! 기본값 없는 필드가 기본값 있는 필드 뒤에 올 수 없음
# @dataclass
# class Derived(Base):
#     z: int  # Error!

# 해결책 1: 기본값 제공
@dataclass
class Derived1(Base):
    z: int = 0


# 해결책 2: kw_only 사용 (Python 3.10+)
@dataclass
class Derived2(Base):
    z: int = field(kw_only=True, default=0)
```

### 5.3 조합 (Composition)

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str


@dataclass
class ContactInfo:
    email: str
    phone: str


@dataclass
class Employee:
    name: str
    employee_id: str
    address: Address          # 조합
    contact: ContactInfo      # 조합
    skills: List[str] = field(default_factory=list)


employee = Employee(
    name="Alice",
    employee_id="E001",
    address=Address("123 Main St", "Seoul", "Korea", "12345"),
    contact=ContactInfo("alice@company.com", "010-1234-5678"),
    skills=["Python", "SQL"]
)
```

### 5.4 믹스인 패턴

```python
from dataclasses import dataclass, field
from datetime import datetime


class TimestampMixin:
    """타임스탬프 기능 믹스인"""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def touch(self):
        self.updated_at = datetime.now()


@dataclass
class Article(TimestampMixin):
    title: str
    content: str
    author: str


article = Article("Hello", "World", "Alice")
print(article.created_at)
article.touch()
print(article.updated_at)
```

---

## 6. Pydantic 비교

### 6.1 dataclass vs Pydantic

```python
# dataclass - 간단한 데이터 구조
from dataclasses import dataclass


@dataclass
class UserDataclass:
    name: str
    age: int
    email: str


# 타입 검증 없음!
user = UserDataclass("Alice", "not an int", "invalid-email")
print(user)  # 에러 없이 생성됨


# Pydantic - 런타임 검증
from pydantic import BaseModel, EmailStr, Field


class UserPydantic(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    email: EmailStr


try:
    user = UserPydantic(name="Alice", age="25", email="alice@example.com")
    print(user)  # age가 자동으로 int로 변환
    print(user.age, type(user.age))  # 25 <class 'int'>
except Exception as e:
    print(f"Validation error: {e}")
```

### 6.2 선택 기준

```
┌─────────────────────────────────────────────────────────────────┐
│                 dataclass vs Pydantic 선택                      │
│                                                                 │
│   dataclass 선택:                                               │
│   - 내부 데이터 구조                                             │
│   - 성능이 중요한 경우                                           │
│   - 단순한 데이터 컨테이너                                       │
│   - 외부 의존성 최소화                                           │
│   - 타입 검증이 필요 없는 경우                                   │
│                                                                 │
│   Pydantic 선택:                                                │
│   - API 입력 검증                                               │
│   - 설정 파일 파싱                                              │
│   - 외부 데이터 처리                                             │
│   - JSON 직렬화/역직렬화                                        │
│   - 복잡한 검증 로직                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Pydantic dataclass

```python
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field


@pydantic_dataclass
class Person:
    """Pydantic 검증 + dataclass 인터페이스"""
    name: str
    age: int = Field(ge=0)


# 검증 수행
try:
    person = Person(name="Alice", age=-1)  # ValidationError
except Exception as e:
    print(e)

# 유효한 데이터
person = Person(name="Bob", age=30)
print(person)
```

---

## 7. 연습 문제

### 연습 1: 불변 설정 클래스
불변(frozen) 설정 클래스를 작성하세요.

```python
# 예시 답안
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    version: str
    debug: bool = False
    settings: tuple = field(default_factory=tuple)  # 불변 컬렉션

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        return cls(
            app_name=data["app_name"],
            version=data["version"],
            debug=data.get("debug", False),
            settings=tuple(data.get("settings", [])),
        )


config = AppConfig("MyApp", "1.0.0")
config_dict = {"app_name": "MyApp", "version": "1.0.0", "debug": True}
config2 = AppConfig.from_dict(config_dict)
```

### 연습 2: 자동 계산 필드
면적과 둘레를 자동 계산하는 Circle 클래스를 작성하세요.

```python
# 예시 답안
from dataclasses import dataclass, field
import math


@dataclass
class Circle:
    radius: float
    area: float = field(init=False)
    circumference: float = field(init=False)

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("Radius cannot be negative")
        self.area = math.pi * self.radius ** 2
        self.circumference = 2 * math.pi * self.radius


circle = Circle(5)
print(f"Area: {circle.area:.2f}")  # ~78.54
print(f"Circumference: {circle.circumference:.2f}")  # ~31.42
```

### 연습 3: 중첩 데이터클래스와 JSON
중첩된 데이터클래스를 JSON으로 직렬화/역직렬화하세요.

```python
# 예시 답안
from dataclasses import dataclass, asdict, field
from typing import List
import json


@dataclass
class OrderItem:
    product_name: str
    quantity: int
    price: float


@dataclass
class Order:
    order_id: str
    customer: str
    items: List[OrderItem] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Order":
        data = json.loads(json_str)
        items = [OrderItem(**item) for item in data.pop("items", [])]
        return cls(**data, items=items)


order = Order(
    "ORD001",
    "Alice",
    [OrderItem("Book", 2, 15.99), OrderItem("Pen", 5, 1.99)]
)
json_str = order.to_json()
print(json_str)

restored = Order.from_json(json_str)
print(restored)
```

---

## 다음 단계
- [14. 패턴 매칭](./14_Pattern_Matching.md)
- [11. 테스트 및 품질 관리](./11_Testing_and_Quality.md)

## 참고 자료
- [Python dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html)
- [PEP 557 - Data Classes](https://peps.python.org/pep-0557/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [attrs Library](https://www.attrs.org/) (대안)
