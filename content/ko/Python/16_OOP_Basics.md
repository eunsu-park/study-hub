# 객체지향 프로그래밍 기초

**이전**: [파이썬 기초](./15_Python_Basics.md)

> **참고**: 이 레슨은 선수 지식 복습용입니다. 고급 레슨(데코레이터, 메타클래스 등)을 시작하기 전에 OOP 기초가 부족하다면 이 내용을 먼저 학습하세요.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 클래스(class)와 객체(object)/인스턴스(instance)의 관계를 설명하고, 클래스 정의로부터 인스턴스를 생성할 수 있습니다
2. 생성자(`__init__`)를 작성하고, 인스턴스 변수(instance variable), 클래스 변수(class variable), `self`의 역할을 구분할 수 있습니다
3. 인스턴스 메서드(instance method), 클래스 메서드(`@classmethod`), 정적 메서드(`@staticmethod`)를 구현할 수 있습니다
4. 단일·다중 상속(inheritance)을 적용하고, `super()`를 사용하며, 메서드 결정 순서(MRO, Method Resolution Order)를 설명할 수 있습니다
5. 명명 관례(public, protected, private)와 `@property` 데코레이터를 활용해 캡슐화(encapsulation)를 구현할 수 있습니다
6. 다형성(polymorphism)과 덕 타이핑(duck typing)을 설명하고, 공통 인터페이스(interface)를 통해 객체를 다루는 코드를 작성할 수 있습니다
7. 특수(매직) 메서드(special/magic method)를 사용해 연산자 오버로딩(operator overloading)과 컨테이너 프로토콜(container protocol)을 구현할 수 있습니다
8. `abc` 모듈을 사용해 추상 기반 클래스(abstract base class)를 정의하고 인터페이스 계약(interface contract)을 강제할 수 있습니다

---

객체지향 프로그래밍(OOP, Object-Oriented Programming)은 중대형 파이썬 애플리케이션을 구조화하는 지배적인 패러다임입니다. 현실 세계의 개념을 명확한 인터페이스를 가진 클래스로 모델링하는 방법, 상속(inheritance)을 통해 코드를 재사용하는 방법, 캡슐화(encapsulation)로 내부 상태를 보호하는 방법을 이해하면 이후 레슨에서 다루는 데코레이터(decorator), 메타클래스(metaclass), 디스크립터(descriptor) 같은 고급 패턴에 대비할 수 있습니다.

---

## 1. 클래스와 객체

### 1.1 기본 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    객체지향 프로그래밍 (OOP)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  클래스 (Class):                                                 │
│  - 객체의 청사진(설계도)                                          │
│  - 속성(변수)과 행동(메서드) 정의                                 │
│                                                                 │
│  객체 (Object) / 인스턴스 (Instance):                            │
│  - 클래스로부터 생성된 실체                                       │
│  - 각 객체는 독립적인 상태(데이터)를 가짐                          │
│                                                                 │
│  예시:                                                           │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │ class Dog   │ ──────▶ │ my_dog      │  # 객체 1             │
│  │  name       │         │ name="Max"  │                       │
│  │  age        │         │ age=3       │                       │
│  │  bark()     │         └─────────────┘                       │
│  │  eat()      │                                                │
│  └─────────────┘ ──────▶ ┌─────────────┐                       │
│                          │ your_dog    │  # 객체 2             │
│                          │ name="Bella"│                       │
│                          │ age=5       │                       │
│                          └─────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 클래스 정의

```python
class Dog:
    """강아지 클래스"""

    # 클래스 변수 (모든 인스턴스가 공유)
    species = "Canis familiaris"

    # 생성자 (초기화 메서드)
    def __init__(self, name, age):
        """
        Args:
            name: 강아지 이름
            age: 강아지 나이
        """
        # 인스턴스 변수 (각 객체마다 독립)
        self.name = name
        self.age = age

    # 인스턴스 메서드
    def bark(self):
        """짖기"""
        print(f"{self.name} says: Woof!")

    def describe(self):
        """설명"""
        return f"{self.name} is {self.age} years old"

    def birthday(self):
        """생일"""
        self.age += 1
        print(f"Happy birthday, {self.name}! Now {self.age} years old.")


# 객체 생성
my_dog = Dog("Max", 3)
your_dog = Dog("Bella", 5)

# 속성 접근
print(my_dog.name)     # Max
print(my_dog.species)  # Canis familiaris

# 메서드 호출
my_dog.bark()           # Max says: Woof!
print(my_dog.describe()) # Max is 3 years old
my_dog.birthday()        # Happy birthday, Max! Now 4 years old.

# 클래스 변수 vs 인스턴스 변수
print(Dog.species)       # 클래스로 접근
print(my_dog.species)    # 인스턴스로 접근 (같은 값)
```

### 1.3 self의 의미

```python
class Circle:
    def __init__(self, radius):
        # self = 현재 생성되는 인스턴스를 참조
        self.radius = radius

    def area(self):
        # self.radius = 이 인스턴스의 radius
        return 3.14159 * self.radius ** 2

    def circumference(self):
        return 2 * 3.14159 * self.radius


# 객체 생성 시
c1 = Circle(5)   # self = c1
c2 = Circle(10)  # self = c2

# 메서드 호출 시
print(c1.area())  # self = c1, self.radius = 5
print(c2.area())  # self = c2, self.radius = 10

# 실제로는 이렇게 동작:
# Circle.area(c1) → self = c1
```

---

## 2. 클래스/인스턴스 변수와 메서드

### 2.1 변수 종류

```python
class Counter:
    # 클래스 변수: 모든 인스턴스가 공유
    total_count = 0

    def __init__(self, name):
        # 인스턴스 변수: 각 객체마다 독립
        self.name = name
        self.count = 0

        # 클래스 변수 수정
        Counter.total_count += 1

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count


c1 = Counter("Counter 1")
c2 = Counter("Counter 2")

c1.increment()
c1.increment()
c2.increment()

print(c1.count)  # 2 (인스턴스 변수)
print(c2.count)  # 1 (인스턴스 변수)
print(Counter.total_count)  # 2 (클래스 변수)

# 주의: 인스턴스로 클래스 변수를 "할당"하면 인스턴스 변수가 됨
c1.total_count = 100  # c1만의 인스턴스 변수 생성
print(c1.total_count)       # 100 (인스턴스)
print(Counter.total_count)  # 2 (클래스 변수 그대로)
```

### 2.2 메서드 종류

```python
class MyClass:
    class_var = 0

    def __init__(self, value):
        self.instance_var = value

    # 인스턴스 메서드: self 사용, 인스턴스 데이터 접근
    def instance_method(self):
        return f"Instance: {self.instance_var}"

    # 클래스 메서드: cls 사용, 클래스 데이터 접근
    @classmethod
    def class_method(cls):
        cls.class_var += 1
        return f"Class var: {cls.class_var}"

    # 정적 메서드: self도 cls도 없음, 독립적인 함수
    @staticmethod
    def static_method(x, y):
        return x + y


obj = MyClass(42)

# 인스턴스 메서드
print(obj.instance_method())  # Instance: 42

# 클래스 메서드 (클래스나 인스턴스로 호출)
print(MyClass.class_method())  # Class var: 1
print(obj.class_method())      # Class var: 2

# 정적 메서드 (클래스나 인스턴스로 호출)
print(MyClass.static_method(3, 4))  # 7
print(obj.static_method(3, 4))      # 7
```

### 2.3 팩토리 메서드 패턴

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

    # 팩토리 메서드: 다양한 형식에서 객체 생성
    @classmethod
    def from_string(cls, date_string):
        """'YYYY-MM-DD' 형식에서 생성"""
        year, month, day = map(int, date_string.split("-"))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        """오늘 날짜로 생성"""
        import datetime
        t = datetime.date.today()
        return cls(t.year, t.month, t.day)


# 다양한 방법으로 객체 생성
d1 = Date(2024, 1, 15)
d2 = Date.from_string("2024-06-20")
d3 = Date.today()

print(d1)  # Date(2024, 1, 15)
print(d2)  # Date(2024, 6, 20)
print(d3)  # Date(현재 날짜)
```

---

## 3. 상속 (Inheritance)

### 3.1 기본 상속

```python
# 부모 클래스 (슈퍼클래스, 기반 클래스)
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement")

    def describe(self):
        return f"I am {self.name}"


# 자식 클래스 (서브클래스, 파생 클래스)
class Dog(Animal):
    def __init__(self, name, breed):
        # 부모 클래스 초기화
        super().__init__(name)
        self.breed = breed

    def speak(self):
        return f"{self.name} says Woof!"

    def fetch(self):
        return f"{self.name} is fetching the ball"


class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

    def scratch(self):
        return f"{self.name} is scratching"


# 사용
dog = Dog("Max", "Golden Retriever")
cat = Cat("Whiskers")

print(dog.describe())  # I am Max (부모 메서드)
print(dog.speak())     # Max says Woof! (오버라이드)
print(dog.fetch())     # Max is fetching the ball (자식 전용)
print(dog.breed)       # Golden Retriever

print(cat.speak())     # Whiskers says Meow!

# 상속 관계 확인
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True
print(issubclass(Dog, Animal))  # True
```

### 3.2 메서드 오버라이딩

```python
class Vehicle:
    def __init__(self, brand):
        self.brand = brand

    def info(self):
        return f"Brand: {self.brand}"

    def move(self):
        return "Moving..."


class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model

    # 메서드 오버라이딩 (재정의)
    def info(self):
        # 부모 메서드 호출 + 확장
        parent_info = super().info()
        return f"{parent_info}, Model: {self.model}"

    def move(self):
        # 완전히 새로운 구현
        return "Driving on the road"

    def honk(self):
        return "Beep beep!"


class Motorcycle(Vehicle):
    def move(self):
        return "Riding on two wheels"


car = Car("Toyota", "Camry")
print(car.info())  # Brand: Toyota, Model: Camry
print(car.move())  # Driving on the road

moto = Motorcycle("Harley")
print(moto.move())  # Riding on two wheels
```

### 3.3 다중 상속

```python
class Flyable:
    def fly(self):
        return "Flying in the sky"


class Swimmable:
    def swim(self):
        return "Swimming in the water"


class Duck(Animal, Flyable, Swimmable):
    def speak(self):
        return f"{self.name} says Quack!"


duck = Duck("Donald")
print(duck.describe())  # I am Donald (Animal)
print(duck.fly())       # Flying in the sky (Flyable)
print(duck.swim())      # Swimming in the water (Swimmable)
print(duck.speak())     # Donald says Quack!

# MRO (Method Resolution Order) 확인
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Animal'>, <class 'Flyable'>, <class 'Swimmable'>, <class 'object'>)
```

---

## 4. 캡슐화 (Encapsulation)

### 4.1 접근 제어

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner          # public (공개)
        self._balance = balance     # protected (보호, 관례상)
        self.__pin = "1234"         # private (비공개, 네임 맹글링)

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return True
        return False

    def withdraw(self, amount, pin):
        if pin != self.__pin:
            raise ValueError("Invalid PIN")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        return amount

    def get_balance(self):
        return self._balance


account = BankAccount("Alice", 1000)

# public 접근
print(account.owner)  # Alice

# protected 접근 (가능하지만 권장하지 않음)
print(account._balance)  # 1000

# private 접근 (직접 불가)
# print(account.__pin)  # AttributeError

# 네임 맹글링으로 접근 가능 (권장하지 않음)
print(account._BankAccount__pin)  # 1234

# 메서드를 통한 안전한 접근
account.deposit(500)
print(account.get_balance())  # 1500
```

### 4.2 프로퍼티 (Property)

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    # getter
    @property
    def celsius(self):
        return self._celsius

    # setter
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    # 계산된 프로퍼티
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9


temp = Temperature()

# 프로퍼티 사용 (속성처럼)
temp.celsius = 25
print(temp.celsius)      # 25
print(temp.fahrenheit)   # 77.0

temp.fahrenheit = 100
print(temp.celsius)      # 37.777...

# 검증 동작
# temp.celsius = -300  # ValueError
```

---

## 5. 다형성 (Polymorphism)

### 5.1 메서드 다형성

```python
class Shape:
    def area(self):
        raise NotImplementedError


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2


class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height


# 다형성: 같은 메서드명, 다른 동작
def print_area(shape: Shape):
    print(f"Area: {shape.area()}")


shapes = [
    Rectangle(4, 5),
    Circle(3),
    Triangle(6, 8)
]

for shape in shapes:
    print_area(shape)

# 출력:
# Area: 20
# Area: 28.27431
# Area: 24.0
```

### 5.2 덕 타이핑 (Duck Typing)

```python
# "오리처럼 걷고 오리처럼 꽥꽥거리면, 그것은 오리다"
# 파이썬은 객체의 타입보다 객체가 가진 메서드/속성에 집중

class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Robot:
    def speak(self):
        return "Beep boop!"

# 상속 관계 없어도 같은 메서드가 있으면 동일하게 처리
def animal_sound(animal):
    print(animal.speak())

# 모두 speak() 메서드가 있으므로 동작
animal_sound(Dog())    # Woof!
animal_sound(Cat())    # Meow!
animal_sound(Robot())  # Beep boop!
```

---

## 6. 특수 메서드 (매직 메서드)

### 6.1 기본 특수 메서드

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 문자열 표현 (사용자용)
    def __str__(self):
        return f"Point({self.x}, {self.y})"

    # 문자열 표현 (개발자용, 디버깅)
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    # 동등성 비교
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    # 해시 (dict 키, set 요소로 사용 가능)
    def __hash__(self):
        return hash((self.x, self.y))

    # 길이
    def __len__(self):
        return 2  # x, y 두 좌표

    # 불리언 변환
    def __bool__(self):
        return self.x != 0 or self.y != 0


p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(0, 0)

print(str(p1))   # Point(3, 4)
print(repr(p1))  # Point(x=3, y=4)
print(p1 == p2)  # True
print(len(p1))   # 2
print(bool(p1))  # True
print(bool(p3))  # False
```

### 6.2 연산자 오버로딩

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    # 덧셈: v1 + v2
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    # 뺄셈: v1 - v2
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    # 곱셈 (스칼라): v * 3
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    # 역방향 곱셈: 3 * v
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    # 내적: v1 @ v2
    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y

    # 음수: -v
    def __neg__(self):
        return Vector(-self.x, -self.y)

    # 절대값 (크기)
    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)     # Vector(4, 6)
print(v1 - v2)     # Vector(2, 2)
print(v1 * 2)      # Vector(6, 8)
print(3 * v1)      # Vector(9, 12)
print(v1 @ v2)     # 11 (내적)
print(-v1)         # Vector(-3, -4)
print(abs(v1))     # 5.0
```

### 6.3 컨테이너 프로토콜

```python
class Deck:
    """카드 덱 클래스"""

    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [f"{rank} of {suit}" for suit in suits for rank in ranks]

    # 길이
    def __len__(self):
        return len(self.cards)

    # 인덱싱: deck[0]
    def __getitem__(self, index):
        return self.cards[index]

    # 할당: deck[0] = "Joker"
    def __setitem__(self, index, value):
        self.cards[index] = value

    # 삭제: del deck[0]
    def __delitem__(self, index):
        del self.cards[index]

    # 포함 여부: "Ace of Spades" in deck
    def __contains__(self, item):
        return item in self.cards

    # 이터레이션
    def __iter__(self):
        return iter(self.cards)


deck = Deck()

print(len(deck))          # 52
print(deck[0])            # 2 of Hearts
print(deck[-1])           # A of Spades
print("A of Spades" in deck)  # True

# 슬라이싱도 자동 지원
print(deck[0:3])  # ['2 of Hearts', '3 of Hearts', '4 of Hearts']

# 반복
for card in deck[:5]:
    print(card)
```

---

## 7. 추상 클래스

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """추상 기반 클래스"""

    @abstractmethod
    def area(self):
        """면적 계산 (반드시 구현 필요)"""
        pass

    @abstractmethod
    def perimeter(self):
        """둘레 계산 (반드시 구현 필요)"""
        pass

    def describe(self):
        """일반 메서드 (상속됨)"""
        return f"Area: {self.area()}, Perimeter: {self.perimeter()}"


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14159 * self.radius


# 추상 클래스는 인스턴스화 불가
# shape = Shape()  # TypeError

rect = Rectangle(4, 5)
print(rect.describe())  # Area: 20, Perimeter: 18

circle = Circle(3)
print(circle.describe())  # Area: 28.27..., Perimeter: 18.84...
```

---

## 정리

### OOP 핵심 개념

| 개념 | 설명 |
|------|------|
| 클래스 | 객체의 설계도 |
| 객체/인스턴스 | 클래스로 생성된 실체 |
| `__init__` | 생성자, 초기화 메서드 |
| `self` | 현재 인스턴스 참조 |
| 상속 | 부모 클래스의 속성/메서드 물려받기 |
| 오버라이딩 | 부모 메서드 재정의 |
| 캡슐화 | 데이터 보호, 접근 제어 |
| 다형성 | 같은 인터페이스, 다른 동작 |
| 추상 클래스 | 미완성 메서드를 가진 기반 클래스 |

### 접근 제어 관례

| 형식 | 의미 | 예시 |
|------|------|------|
| `name` | public | 자유롭게 접근 |
| `_name` | protected | 내부 사용 권장 |
| `__name` | private | 네임 맹글링 적용 |

### 다음 단계

OOP 기초를 마쳤다면 다음 레슨으로:
- [데코레이터 (Decorators)](./02_Decorators.md): 데코레이터
- [메타클래스 (Metaclasses)](./06_Metaclasses.md): 메타클래스
- [디스크립터 (Descriptors)](./07_Descriptors.md): 디스크립터

---

## 연습 문제

### 연습 1: 클래스 기초

객체 모델에 대한 이해를 다지기 위해 클래스를 처음부터 만드세요.

1. 다음을 갖는 `BankAccount` 클래스를 정의하세요:
   - 인스턴스 변수: `owner` (str), `balance` (float, 기본값 0.0)
   - 클래스 변수: `interest_rate` (float, 기본값 0.02)
   - 메서드: `deposit(amount)`, `withdraw(amount)`, `apply_interest()`, `__str__`
2. `deposit`과 `withdraw`에 유효성 검사를 추가하세요: 두 금액 모두 양수여야 하며, `withdraw`는 잔액이 충분한지 확인해야 합니다. 실패 시 설명적인 메시지와 함께 `ValueError`를 발생시키세요.
3. 딕셔너리 `{"owner": "Alice", "balance": 500.0}`에서 `BankAccount`를 생성하는 클래스 메서드 `from_dict(cls, data)`를 추가하세요.
4. `amount > 0`이면 `True`를 반환하는 정적 메서드 `is_valid_amount(amount)`를 추가하세요.
5. 두 계좌를 만들고 입출금하고, 이자를 적용하세요. `BankAccount.interest_rate = 0.03`을 통해 `interest_rate` 변경이 모든 인스턴스에 적용됨을 시연하세요.

### 연습 2: 상속 계층

`super()`와 메서드 재정의(method overriding)를 연습하기 위해 3단계 상속 계층을 만드세요.

1. 다음을 갖는 기본 클래스 `Employee`를 정의하세요:
   - `__init__(self, name, emp_id, base_salary)`
   - `calculate_pay()` → `base_salary` 반환
   - `__str__` → 예: `"Employee Alice (E001): $3000.00/month"`
2. `team_size`를 추가하고 팀원당 $500 보너스를 더하도록 `calculate_pay()`를 재정의하는 `Manager(Employee)`를 생성하세요.
3. `hourly_rate`와 `hours_worked`를 추가하고, `base_salary`를 무시하고 `hourly_rate * hours_worked`를 반환하도록 `calculate_pay()`를 재정의하는 `Contractor(Employee)`를 생성하세요.
4. `Employee`, `Manager`, `Contractor` 객체의 혼합 리스트를 받아 형식화된 급여 요약을 출력하는 함수 `payroll_report(employees)`를 작성하세요.
5. `isinstance()`와 `issubclass()`로 계층 구조가 올바른지 검증하고, `Manager`의 MRO(메서드 결정 순서, Method Resolution Order)를 출력하세요.

### 연습 3: 프로퍼티(Property)를 이용한 캡슐화

속성 접근 인터페이스를 유지하면서 불변 조건을 강제하는 `@property` 데코레이터를 사용하세요.

1. `radius`가 프로퍼티인 `Circle` 클래스를 만드세요:
   - getter는 저장된 반지름을 반환합니다.
   - setter는 반지름이 음수이면 `ValueError`를 발생시킵니다.
   - 읽기 전용 프로퍼티 `area`와 `circumference`를 추가하세요.
2. `width`와 `height`가 프로퍼티(둘 다 양수여야 함)인 `Rectangle` 클래스를 만드세요. 계산된 프로퍼티 `aspect_ratio = width / height`를 추가하세요.
3. 관련 치수를 `factor`만큼 곱하는 `scale(factor)` 메서드를 두 클래스 모두에 추가하세요. `factor`가 음수이면 setter의 유효성 검사가 실행되는지 확인하세요.
4. 네임 맹글링(name mangling)을 시연하세요: `__init__`에서 `uuid.uuid4()`로 생성한 private `__id` 속성을 추가하고, `obj.__id`로는 접근할 수 없지만 `obj._ClassName__id`로는 접근할 수 있음을 보이세요.
5. `@property`를 일반 속성 대신 언제 선택할지 설명하세요. 접근 시 유효성 검사 또는 계산이 필요한 실제 예시 두 가지를 제시하세요.

### 연습 4: 매직 메서드(Magic Method)와 연산자 오버로딩

매직 메서드를 사용하여 풍부한 `Fraction` 클래스를 구현하세요.

1. 정수 `numerator`와 `denominator`를 저장하는 `Fraction` 클래스를 만드세요:
   - `__init__`: `math.gcd`를 사용하여 기약분수 형태로 저장; 분모가 0이면 `ZeroDivisionError` 발생.
   - `__str__`: `"3/4"`
   - `__repr__`: `"Fraction(3, 4)"`
2. 산술 연산자 `__add__`, `__sub__`, `__mul__`, `__truediv__`를 구현하세요. 각각 기약분수 형태의 새 `Fraction`을 반환해야 합니다.
3. 비교 연산자 `__eq__`, `__lt__`, `__le__`, `__gt__`, `__ge__`를 구현하세요 (비교에 교차 곱셈 사용).
4. `__abs__`, `__neg__`, `__float__` (Python float으로 변환)를 구현하세요.
5. `__hash__`를 구현하여 `Fraction` 객체를 집합에 저장하거나 딕셔너리 키로 사용할 수 있도록 검증하세요. 시연: `{Fraction(1,2), Fraction(2,4)}`는 하나의 요소만 가져야 합니다.

### 연습 5: 추상 기반 클래스(Abstract Base Class)와 다형성

추상 기반 클래스를 사용하여 플러그인 방식의 시스템을 설계하세요.

1. `abc.ABC`와 `@abstractmethod`를 사용하여 다음 추상 메서드를 가진 추상 클래스 `DataExporter`를 정의하세요:
   - `export(data: list, filepath: str) -> None`
   - `get_extension() -> str`
   - `data`가 리스트가 아니면 `TypeError`를 발생시키는 구체 메서드 `validate(data)`.
2. 두 개의 구체 익스포터를 구현하세요:
   - `CSVExporter`: `csv` 모듈을 사용하여 데이터(딕셔너리 리스트)를 CSV 파일에 씁니다.
   - `JSONExporter`: `json` 모듈과 `indent=2`를 사용하여 데이터를 JSON 파일에 씁니다.
3. `DataExporter` 인스턴스 리스트를 받아 각각을 호출하면서 `filepath_base`에 올바른 확장자를 추가하는 함수 `export_all(data, filepath_base, exporters)`를 작성하세요.
4. 덕 타이핑(duck typing) 시연: `DataExporter`를 상속받지 않지만 `export()`와 `get_extension()`을 구현하는 `MarkdownExporter`를 작성하세요. 덕 타이핑 덕분에 `export_all`과 함께 작동함을 보이세요.
5. `isinstance()`를 사용하여 혼합 리스트에서 어느 익스포터가 `DataExporter`의 진정한 서브클래스인지, 어느 것이 덕 타입 구현인지 확인하세요. 추상 기반 클래스 계약을 강제하는 것과 덕 타이핑에 의존하는 것 중 언제 무엇을 선택할지 성찰하세요.

---

## 참고 자료

- [Python OOP 공식 문서](https://docs.python.org/3/tutorial/classes.html)
- [Real Python - OOP](https://realpython.com/python3-object-oriented-programming/)
- [Data Model 문서](https://docs.python.org/3/reference/datamodel.html)
