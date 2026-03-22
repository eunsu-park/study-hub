# OOP 고급 (OOP Advanced)

**이전**: [OOP 기초](./08_OOP_Basics.md) | **다음**: [모듈과 패키지](./10_Modules_and_Packages.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단일 상속 (Single Inheritance)을 구현하고 `super()`를 사용하여 부모 클래스 메서드를 호출한다
2. 서브클래스에서 메서드를 오버라이드하고 다중 상속에서의 메서드 해석 순서 (Method Resolution Order, MRO)를 이해한다
3. `abc` 모듈을 사용하여 인터페이스 계약을 강제하는 추상 기반 클래스 (Abstract Base Class)를 정의한다
4. 파이썬에서 다형성 (Polymorphism)과 덕 타이핑 (Duck Typing) 원칙을 적용한다
5. 연산자 오버로딩 (`__add__`, `__eq__`, `__lt__`, `__len__`, `__getitem__` 등)을 사용하여 클래스가 내장 타입처럼 동작하게 한다
6. 코드 재사용을 위해 조합 (Composition)과 상속 (Inheritance) 중 적절한 것을 선택한다
7. `@dataclass` 데코레이터를 사용하여 데이터 보유 클래스의 보일러플레이트를 줄인다
8. 파이썬에서 기본적인 디자인 패턴 (팩토리, 싱글톤)을 인식하고 구현한다

---

이전 레슨의 OOP 기초를 바탕으로, 이 레슨은 객체 지향 파이썬을 진정으로 강력하게 만드는 기법들을 다룹니다. 상속은 클래스 계층 구조를 구축할 수 있게 합니다. 추상 클래스는 계약을 정의합니다. 연산자 오버로딩은 여러분의 객체가 파이썬 문법과 매끄럽게 동작하게 합니다. 조합은 깊은 상속 트리에 대한 유연한 대안을 제공합니다. 이 도구들을 잘 사용하면 표현력 있고 유지보수하기 쉬운 코드를 만들 수 있습니다.

## 1. 상속 (Inheritance)

상속은 새로운 클래스 (자식/서브클래스)가 기존 클래스 (부모/슈퍼클래스)의 속성과 메서드를 물려받을 수 있게 합니다.

### 기본 상속

```python
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def speak(self):
        return f"{self.name} makes a sound"

    def __str__(self):
        return f"{self.name} ({self.species})"


class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")  # Call parent __init__
        self.breed = breed

    def speak(self):  # Override parent method
        return f"{self.name} barks: Woof!"

    def fetch(self, item):  # New method specific to Dog
        return f"{self.name} fetches the {item}"


class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, "Cat")
        self.indoor = indoor

    def speak(self):
        return f"{self.name} meows: Meow!"

    def purr(self):
        return f"{self.name} purrs contentedly"


# Usage
dog = Dog("Rex", "German Shepherd")
cat = Cat("Whiskers", indoor=True)

print(dog)             # Rex (Dog)
print(dog.speak())     # Rex barks: Woof!
print(dog.fetch("ball"))  # Rex fetches the ball

print(cat)             # Whiskers (Cat)
print(cat.speak())     # Whiskers meows: Meow!
print(cat.purr())      # Whiskers purrs contentedly

# Inheritance checks
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True
print(issubclass(Dog, Animal))  # True
```

### `super()` 함수

`super()`는 부모 클래스에 메서드 호출을 위임하는 프록시 객체를 반환합니다. 클래스 계층 구조에서 적절한 초기화를 위해 필수적입니다.

```python
class Vehicle:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def describe(self):
        return f"{self.year} {self.make} {self.model}"


class ElectricVehicle(Vehicle):
    def __init__(self, make, model, year, battery_kwh):
        super().__init__(make, model, year)
        self.battery_kwh = battery_kwh

    def describe(self):
        base = super().describe()  # Call parent's describe
        return f"{base} (Electric, {self.battery_kwh} kWh)"


class Tesla(ElectricVehicle):
    def __init__(self, model, year, battery_kwh, autopilot=False):
        super().__init__("Tesla", model, year, battery_kwh)
        self.autopilot = autopilot

    def describe(self):
        base = super().describe()
        ap_status = "with Autopilot" if self.autopilot else "no Autopilot"
        return f"{base} - {ap_status}"


car = Tesla("Model 3", 2024, 75, autopilot=True)
print(car.describe())
# 2024 Tesla Model 3 (Electric, 75 kWh) - with Autopilot
```

### 부모 메서드 확장

```python
class Logger:
    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def get_logs(self):
        return self.logs


class TimestampLogger(Logger):
    def log(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        super().log(f"[{timestamp}] {message}")  # Extend, not replace


class PriorityLogger(TimestampLogger):
    def log(self, message, priority="INFO"):
        super().log(f"({priority}) {message}")  # Further extend


logger = PriorityLogger()
logger.log("System started", "INFO")
logger.log("Disk full", "CRITICAL")

for entry in logger.get_logs():
    print(entry)
# [2024-01-15 10:30:00] (INFO) System started
# [2024-01-15 10:30:00] (CRITICAL) Disk full
```

---

## 2. 메서드 오버라이딩

서브클래스가 부모 메서드와 같은 이름의 메서드를 정의하면, 서브클래스 버전이 우선합니다.

```python
class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement area()")

    def perimeter(self):
        raise NotImplementedError("Subclasses must implement perimeter()")

    def describe(self):
        return f"{self.__class__.__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}"


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
        import math
        return math.pi * self.radius ** 2

    def perimeter(self):
        import math
        return 2 * math.pi * self.radius


class Triangle(Shape):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def area(self):
        s = (self.a + self.b + self.c) / 2
        return (s * (s - self.a) * (s - self.b) * (s - self.c)) ** 0.5

    def perimeter(self):
        return self.a + self.b + self.c


shapes = [Rectangle(5, 3), Circle(4), Triangle(3, 4, 5)]
for shape in shapes:
    print(shape.describe())
# Rectangle: area=15.00, perimeter=16.00
# Circle: area=50.27, perimeter=25.13
# Triangle: area=6.00, perimeter=12.00
```

---

## 3. 다중 상속과 MRO

파이썬은 다중 상속 (Multiple Inheritance)을 지원합니다 -- 클래스가 하나 이상의 부모로부터 상속받을 수 있습니다.

```python
class Flyable:
    def fly(self):
        return f"{self.__class__.__name__} is flying"

class Swimmable:
    def swim(self):
        return f"{self.__class__.__name__} is swimming"

class Walkable:
    def walk(self):
        return f"{self.__class__.__name__} is walking"


class Duck(Flyable, Swimmable, Walkable):
    def __init__(self, name):
        self.name = name

    def quack(self):
        return f"{self.name} says: Quack!"

duck = Duck("Donald")
print(duck.fly())    # Duck is flying
print(duck.swim())   # Duck is swimming
print(duck.walk())   # Duck is walking
print(duck.quack())  # Donald says: Quack!
```

### 메서드 해석 순서 (MRO)

여러 부모가 같은 메서드를 정의할 때, 파이썬은 C3 선형화 알고리즘을 사용하여 어떤 메서드를 호출할지 결정합니다.

```python
class A:
    def greet(self):
        return "Hello from A"

class B(A):
    def greet(self):
        return "Hello from B"

class C(A):
    def greet(self):
        return "Hello from C"

class D(B, C):
    pass

d = D()
print(d.greet())  # Hello from B (B comes before C in MRO)

# Inspect the MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# Or use mro() method
for cls in D.mro():
    print(cls.__name__)
# D -> B -> C -> A -> object
```

### `super()`를 사용한 협력적 다중 상속

```python
class Base:
    def __init__(self, **kwargs):
        # Absorb remaining kwargs so the chain does not break
        pass

class PowerSource(Base):
    def __init__(self, fuel_type="electric", **kwargs):
        super().__init__(**kwargs)
        self.fuel_type = fuel_type

class Navigation(Base):
    def __init__(self, gps=True, **kwargs):
        super().__init__(**kwargs)
        self.gps = gps

class Communication(Base):
    def __init__(self, radio_freq=None, **kwargs):
        super().__init__(**kwargs)
        self.radio_freq = radio_freq

class Drone(PowerSource, Navigation, Communication):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def status(self):
        return (f"Drone '{self.name}': fuel={self.fuel_type}, "
                f"GPS={'on' if self.gps else 'off'}, "
                f"radio={self.radio_freq or 'none'}")

drone = Drone("Scout", fuel_type="battery", gps=True, radio_freq="2.4GHz")
print(drone.status())
# Drone 'Scout': fuel=battery, GPS=on, radio=2.4GHz

print(Drone.__mro__)
# Drone -> PowerSource -> Navigation -> Communication -> Base -> object
```

### 다이아몬드 문제

```python
class A:
    def __init__(self):
        print("A.__init__")
        super().__init__()

class B(A):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A):
    def __init__(self):
        print("C.__init__")
        super().__init__()

class D(B, C):
    def __init__(self):
        print("D.__init__")
        super().__init__()

d = D()
# D.__init__
# B.__init__
# C.__init__
# A.__init__
# Each class's __init__ is called exactly once (thanks to C3 linearization)
```

---

## 4. 추상 기반 클래스

추상 기반 클래스 (Abstract Base Class, ABC)는 서브클래스가 반드시 구현해야 하는 인터페이스를 정의합니다. 추상 클래스를 직접 인스턴스화할 수 없습니다.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes."""

    @abstractmethod
    def area(self):
        """Calculate the area of the shape."""
        pass

    @abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape."""
        pass

    def describe(self):
        """Non-abstract method -- inherited by all subclasses."""
        return f"{self.__class__.__name__}: area={self.area():.2f}"

# Cannot instantiate abstract class
# shape = Shape()  # TypeError: Can't instantiate abstract class Shape

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

    def perimeter(self):
        return 4 * self.side

# Must implement ALL abstract methods
class IncompleteShape(Shape):
    def area(self):
        return 0
    # Missing perimeter() -- cannot instantiate!

# incomplete = IncompleteShape()
# TypeError: Can't instantiate abstract class IncompleteShape
# with abstract method perimeter

s = Square(5)
print(s.describe())     # Square: area=25.00
print(s.perimeter())    # 20
```

### 추상 프로퍼티

```python
from abc import ABC, abstractmethod

class DatabaseAdapter(ABC):
    """Abstract interface for database adapters."""

    @property
    @abstractmethod
    def connection_string(self):
        """Return the connection string."""
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def execute(self, query):
        pass

    @abstractmethod
    def close(self):
        pass


class PostgresAdapter(DatabaseAdapter):
    def __init__(self, host, port, database):
        self.host = host
        self.port = port
        self.database = database
        self._connected = False

    @property
    def connection_string(self):
        return f"postgresql://{self.host}:{self.port}/{self.database}"

    def connect(self):
        print(f"Connecting to {self.connection_string}")
        self._connected = True

    def execute(self, query):
        if not self._connected:
            raise RuntimeError("Not connected")
        print(f"Executing: {query}")
        return []

    def close(self):
        self._connected = False
        print("Connection closed")


db = PostgresAdapter("localhost", 5432, "mydb")
db.connect()
db.execute("SELECT * FROM users")
db.close()
```

---

## 5. 다형성과 덕 타이핑

### 다형성

다형성 (Polymorphism)은 "여러 형태"를 의미합니다 -- 같은 인터페이스를 공유하면 서로 다른 클래스를 상호 교환적으로 사용할 수 있습니다.

```python
class PaymentProcessor:
    def process(self, amount):
        raise NotImplementedError

class CreditCardProcessor(PaymentProcessor):
    def __init__(self, card_number):
        self.card_number = card_number

    def process(self, amount):
        return f"Charged ${amount:.2f} to card ending in {self.card_number[-4:]}"

class PayPalProcessor(PaymentProcessor):
    def __init__(self, email):
        self.email = email

    def process(self, amount):
        return f"Sent ${amount:.2f} via PayPal to {self.email}"

class CryptoProcessor(PaymentProcessor):
    def __init__(self, wallet_address):
        self.wallet = wallet_address

    def process(self, amount):
        return f"Transferred ${amount:.2f} in crypto to {self.wallet[:8]}..."


def checkout(processor, amount):
    """Works with ANY payment processor -- polymorphism in action."""
    print(processor.process(amount))

# Same function, different behaviors
checkout(CreditCardProcessor("4111111111111111"), 99.99)
checkout(PayPalProcessor("alice@example.com"), 49.50)
checkout(CryptoProcessor("0xABCDEF1234567890"), 150.00)
# Charged $99.99 to card ending in 1111
# Sent $49.50 via PayPal to alice@example.com
# Transferred $150.00 in crypto to 0xABCDEF...
```

### 덕 타이핑

"오리처럼 걷고 오리처럼 꽥꽥거리면, 그것은 오리이다." 파이썬은 타입을 검사하지 않습니다 -- 동작을 검사합니다.

```python
# These classes share no common parent, but all have a write() method
class FileWriter:
    def write(self, data):
        print(f"Writing to file: {data}")

class NetworkSender:
    def write(self, data):
        print(f"Sending over network: {data}")

class Logger:
    def write(self, data):
        print(f"[LOG] {data}")

class NullWriter:
    def write(self, data):
        pass  # Silently discard

def save_report(writer, report):
    """Works with anything that has a write() method."""
    writer.write(f"Report: {report}")

# All work, no shared base class needed
save_report(FileWriter(), "Q4 Results")
save_report(NetworkSender(), "Q4 Results")
save_report(Logger(), "Q4 Results")
save_report(NullWriter(), "Q4 Results")
```

### 프로토콜 클래스 (구조적 서브타이핑)

Python 3.8+는 정적 타입 검사와 함께 명시적인 덕 타이핑을 위한 `Protocol`을 제공합니다.

```python
from typing import Protocol

class Renderable(Protocol):
    def render(self) -> str:
        ...

class HTMLPage:
    def render(self) -> str:
        return "<html><body>Hello</body></html>"

class JSONResponse:
    def render(self) -> str:
        return '{"message": "Hello"}'

def display(item: Renderable) -> None:
    """Type checker verifies that item has render() method."""
    print(item.render())

# Both work -- they satisfy the Renderable protocol structurally
display(HTMLPage())
display(JSONResponse())
```

---

## 6. 연산자 오버로딩

연산자 오버로딩 (Operator Overloading)은 특수 (던더) 메서드를 구현하여 객체가 파이썬 연산자 (`+`, `-`, `==`, `<`, `[]` 등)와 함께 동작하게 합니다.

### 산술 연산자

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """self + other"""
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        """self - other"""
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        """self * scalar"""
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        """scalar * self (reflected multiplication)"""
        return self.__mul__(scalar)

    def __neg__(self):
        """-self"""
        return Vector(-self.x, -self.y)

    def __abs__(self):
        """abs(self) -- magnitude"""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # Vector(4, 6)
print(v1 - v2)      # Vector(2, 2)
print(v1 * 3)       # Vector(9, 12)
print(2 * v1)       # Vector(6, 8) -- uses __rmul__
print(-v1)           # Vector(-3, -4)
print(abs(v1))       # 5.0
```

### 비교 연산자

```python
class Student:
    def __init__(self, name, gpa):
        self.name = name
        self.gpa = gpa

    def __eq__(self, other):
        """self == other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa == other.gpa

    def __lt__(self, other):
        """self < other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa < other.gpa

    def __le__(self, other):
        """self <= other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa <= other.gpa

    def __gt__(self, other):
        """self > other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa > other.gpa

    def __ge__(self, other):
        """self >= other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa >= other.gpa

    def __repr__(self):
        return f"Student({self.name!r}, gpa={self.gpa})"

students = [
    Student("Alice", 3.8),
    Student("Bob", 3.5),
    Student("Charlie", 3.9),
    Student("Diana", 3.5),
]

# Sorting uses __lt__
print(sorted(students))
# [Student('Bob', gpa=3.5), Student('Diana', gpa=3.5),
#  Student('Alice', gpa=3.8), Student('Charlie', gpa=3.9)]

print(Student("Alice", 3.8) == Student("Bob", 3.8))  # True (same GPA)
print(Student("Alice", 3.8) > Student("Bob", 3.5))   # True
```

### `functools.total_ordering` 사용

`__eq__`와 하나의 순서 메서드만 정의하면 `total_ordering`이 나머지를 도출합니다.

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

    def __repr__(self):
        return f"Temperature({self.celsius}°C)"

t1 = Temperature(100)
t2 = Temperature(37)

print(t1 > t2)    # True  (derived from __lt__ and __eq__)
print(t1 >= t2)   # True
print(t2 <= t1)   # True
```

### 컨테이너 연산자: `__len__`과 `__getitem__`

```python
class Playlist:
    def __init__(self, name):
        self.name = name
        self._songs = []

    def add(self, song):
        self._songs.append(song)
        return self  # For chaining

    def __len__(self):
        """len(playlist)"""
        return len(self._songs)

    def __getitem__(self, index):
        """playlist[index] and playlist[start:stop]"""
        return self._songs[index]

    def __setitem__(self, index, value):
        """playlist[index] = value"""
        self._songs[index] = value

    def __delitem__(self, index):
        """del playlist[index]"""
        del self._songs[index]

    def __contains__(self, song):
        """song in playlist"""
        return song in self._songs

    def __iter__(self):
        """for song in playlist"""
        return iter(self._songs)

    def __repr__(self):
        return f"Playlist({self.name!r}, {len(self)} songs)"


pl = Playlist("Road Trip")
pl.add("Hotel California").add("Bohemian Rhapsody").add("Stairway to Heaven")

print(len(pl))           # 3
print(pl[0])             # Hotel California
print(pl[-1])            # Stairway to Heaven
print(pl[0:2])           # ['Hotel California', 'Bohemian Rhapsody']
print("Bohemian Rhapsody" in pl)  # True

for song in pl:
    print(f"  Playing: {song}")

pl[1] = "We Will Rock You"
print(pl[1])             # We Will Rock You
```

### 호출 가능 객체: `__call__`

```python
class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return self.n + x

add_five = Adder(5)
print(add_five(10))     # 15
print(add_five(20))     # 25
print(callable(add_five))  # True


class Polynomial:
    """Represent a polynomial and evaluate it at a point."""

    def __init__(self, *coefficients):
        # coefficients are in order: a0 + a1*x + a2*x^2 + ...
        self.coefficients = coefficients

    def __call__(self, x):
        return sum(c * x ** i for i, c in enumerate(self.coefficients))

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coefficients):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}x")
            else:
                terms.append(f"{c}x^{i}")
        return " + ".join(terms) if terms else "0"

# p(x) = 1 + 2x + 3x^2
p = Polynomial(1, 2, 3)
print(p)         # 1 + 2x + 3x^2
print(p(0))      # 1
print(p(1))      # 6  (1 + 2 + 3)
print(p(2))      # 17 (1 + 4 + 12)
```

### 일반적인 던더 메서드 참조

| 연산자/내장함수 | 메서드 | 예시 |
|---------|--------|---------|
| `+` | `__add__` | `a + b` |
| `-` | `__sub__` | `a - b` |
| `*` | `__mul__` | `a * b` |
| `/` | `__truediv__` | `a / b` |
| `//` | `__floordiv__` | `a // b` |
| `%` | `__mod__` | `a % b` |
| `**` | `__pow__` | `a ** b` |
| `==` | `__eq__` | `a == b` |
| `!=` | `__ne__` | `a != b` |
| `<` | `__lt__` | `a < b` |
| `<=` | `__le__` | `a <= b` |
| `>` | `__gt__` | `a > b` |
| `>=` | `__ge__` | `a >= b` |
| `len()` | `__len__` | `len(a)` |
| `[]` | `__getitem__` | `a[key]` |
| `[]=` | `__setitem__` | `a[key] = val` |
| `del[]` | `__delitem__` | `del a[key]` |
| `in` | `__contains__` | `x in a` |
| `()` | `__call__` | `a()` |
| `str()` | `__str__` | `str(a)` |
| `repr()` | `__repr__` | `repr(a)` |
| `bool()` | `__bool__` | `if a:` |
| `hash()` | `__hash__` | `hash(a)` |
| `iter()` | `__iter__` | `for x in a:` |
| `next()` | `__next__` | `next(a)` |

---

## 7. 조합 vs 상속

### 상속: "is-a" 관계

```python
class Engine:
    def start(self):
        return "Engine started"

class ElectricEngine(Engine):
    def start(self):
        return "Electric engine humming"
```

`ElectricEngine`은 `Engine`**이다** (is-a).

### 조합: "has-a" 관계

```python
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower

    def start(self):
        return f"{self.horsepower}HP engine started"

class Transmission:
    def __init__(self, type_name):
        self.type_name = type_name

    def shift(self, gear):
        return f"{self.type_name} shifting to gear {gear}"

class GPS:
    def navigate(self, destination):
        return f"Navigating to {destination}"


class Car:
    """Car is composed of engine, transmission, and optional GPS."""

    def __init__(self, make, model, engine, transmission, gps=None):
        self.make = make
        self.model = model
        self.engine = engine            # has-a Engine
        self.transmission = transmission # has-a Transmission
        self.gps = gps                  # has-a GPS (optional)

    def start(self):
        return f"{self.make} {self.model}: {self.engine.start()}"

    def drive(self, gear, destination=None):
        actions = [self.transmission.shift(gear)]
        if destination and self.gps:
            actions.append(self.gps.navigate(destination))
        return " | ".join(actions)


# Compose a car from parts
car = Car(
    "Toyota", "Camry",
    engine=Engine(203),
    transmission=Transmission("automatic"),
    gps=GPS()
)

print(car.start())
# Toyota Camry: 203HP engine started

print(car.drive(3, "Seoul"))
# automatic shifting to gear 3 | Navigating to Seoul
```

### 각각을 사용해야 할 때

| 상속 선호 | 조합 선호 |
|---|---|
| 진정한 "is-a" 관계 | "has-a" 또는 "uses-a" 관계 |
| 타입 계층 구조에서 동작 공유 | 독립적인 기능 결합 |
| 프레임워크가 요구 (예: ABC, Django 모델) | 런타임에 구성요소 교체 필요 |
| 얕은 계층 구조 (1-2단계) | 깊은 계층 구조는 취약해짐 |

### 조합 우선 원칙

```python
# Instead of a complex inheritance tree:
# Animal -> FlyingAnimal -> FlyingSwimmingAnimal -> ...

# Use composition with capability objects:
class FlyAbility:
    def fly(self, owner):
        return f"{owner.name} soars through the sky"

class SwimAbility:
    def swim(self, owner):
        return f"{owner.name} glides through the water"

class RunAbility:
    def run(self, owner):
        return f"{owner.name} runs swiftly"


class Animal:
    def __init__(self, name, abilities=None):
        self.name = name
        self.abilities = abilities or []

    def perform(self, action):
        for ability in self.abilities:
            method = getattr(ability, action, None)
            if method:
                return method(self)
        return f"{self.name} cannot {action}"


duck = Animal("Duck", [FlyAbility(), SwimAbility(), RunAbility()])
eagle = Animal("Eagle", [FlyAbility()])
fish = Animal("Fish", [SwimAbility()])

print(duck.perform("fly"))    # Duck soars through the sky
print(duck.perform("swim"))   # Duck glides through the water
print(eagle.perform("swim"))  # Eagle cannot swim
print(fish.perform("swim"))   # Fish glides through the water
```

---

## 8. 데이터클래스

`@dataclass` 데코레이터 (Python 3.7+)는 `__init__`, `__repr__`, `__eq__`와 선택적으로 다른 메서드들을 자동으로 생성합니다.

### 기본 데이터클래스

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Automatically generates __init__, __repr__, __eq__
p1 = Point(3.0, 4.0)
p2 = Point(3.0, 4.0)

print(p1)          # Point(x=3.0, y=4.0)
print(p1 == p2)    # True (compares all fields)
print(p1.x)        # 3.0
```

### 기본값

```python
from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int
    grade: float = 0.0
    courses: list = field(default_factory=list)  # Mutable default

s = Student("Alice", 20)
s.courses.append("Math")
print(s)  # Student(name='Alice', age=20, grade=0.0, courses=['Math'])

# Each instance gets its own list
s2 = Student("Bob", 22)
print(s2.courses)  # [] (independent)
```

### 프로즌 데이터클래스 (불변)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Color:
    red: int
    green: int
    blue: int

c = Color(255, 128, 0)
print(c)  # Color(red=255, green=128, blue=0)

# Cannot modify
# c.red = 200  # FrozenInstanceError

# Frozen dataclasses are hashable (can be dict keys or set elements)
colors = {Color(255, 0, 0): "red", Color(0, 255, 0): "green"}
print(colors[Color(255, 0, 0)])  # red
```

### 데이터클래스 정렬

```python
from dataclasses import dataclass

@dataclass(order=True)
class Version:
    major: int
    minor: int
    patch: int

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

versions = [Version(2, 1, 0), Version(1, 9, 5), Version(2, 0, 3)]
print(sorted(versions))
# [Version(major=1, minor=9, patch=5), Version(major=2, minor=0, patch=3),
#  Version(major=2, minor=1, patch=0)]
```

### 후처리 초기화

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Not in __init__, computed

    def __post_init__(self):
        """Called after __init__."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dimensions must be positive")
        self.area = self.width * self.height

r = Rectangle(5, 3)
print(r)       # Rectangle(width=5, height=3, area=15)
print(r.area)  # 15
```

### 데이터클래스 상속

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: str
    department: str
    salary: float = 50000.0

emp = Employee("Alice", 30, "E001", "Engineering", 75000)
print(emp)
# Employee(name='Alice', age=30, employee_id='E001',
#          department='Engineering', salary=75000)
```

---

## 9. 일반적인 디자인 패턴

### 팩토리 패턴

팩토리 패턴 (Factory Pattern)은 생성 로직을 노출하지 않고 객체를 생성합니다.

```python
from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, message):
        pass

class EmailNotification(Notification):
    def __init__(self, email):
        self.email = email

    def send(self, message):
        return f"Email to {self.email}: {message}"

class SMSNotification(Notification):
    def __init__(self, phone):
        self.phone = phone

    def send(self, message):
        return f"SMS to {self.phone}: {message}"

class PushNotification(Notification):
    def __init__(self, device_id):
        self.device_id = device_id

    def send(self, message):
        return f"Push to {self.device_id}: {message}"


class NotificationFactory:
    """Factory class to create notification objects."""

    _registry = {
        "email": EmailNotification,
        "sms": SMSNotification,
        "push": PushNotification,
    }

    @classmethod
    def create(cls, channel, destination):
        """Create a notification based on channel type."""
        notification_class = cls._registry.get(channel)
        if notification_class is None:
            raise ValueError(f"Unknown channel: {channel}")
        return notification_class(destination)

    @classmethod
    def register(cls, channel, notification_class):
        """Register a new notification type."""
        cls._registry[channel] = notification_class


# Usage
notif = NotificationFactory.create("email", "alice@example.com")
print(notif.send("Hello!"))
# Email to alice@example.com: Hello!

notif = NotificationFactory.create("sms", "+1-555-0100")
print(notif.send("Hello!"))
# SMS to +1-555-0100: Hello!

# Send to multiple channels
def broadcast(message, targets):
    """Send message to multiple channels."""
    for channel, destination in targets:
        notif = NotificationFactory.create(channel, destination)
        print(notif.send(message))

targets = [
    ("email", "alice@example.com"),
    ("sms", "+1-555-0100"),
    ("push", "device-abc-123"),
]
broadcast("System maintenance at 10 PM", targets)
```

### 싱글톤 패턴

싱글톤 패턴 (Singleton Pattern)은 클래스가 하나의 인스턴스만 가지도록 보장합니다.

```python
class Singleton:
    """Basic singleton using __new__."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value=None):
        # __init__ is called every time, so guard against re-init
        if not hasattr(self, "_initialized"):
            self.value = value
            self._initialized = True


s1 = Singleton("first")
s2 = Singleton("second")

print(s1 is s2)       # True (same instance)
print(s1.value)        # first (not overwritten)
print(id(s1) == id(s2))  # True
```

### 데코레이터를 통한 싱글톤

```python
def singleton(cls):
    """Decorator that turns a class into a singleton."""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class AppConfig:
    def __init__(self):
        self.settings = {}

    def set(self, key, value):
        self.settings[key] = value

    def get(self, key, default=None):
        return self.settings.get(key, default)


config1 = AppConfig()
config1.set("debug", True)

config2 = AppConfig()
print(config2.get("debug"))  # True (same instance)
print(config1 is config2)    # True
```

---

## 10. 종합 예제

### 예제: ABC, 팩토리, 덕 타이핑을 사용한 플러그인 시스템

```python
from abc import ABC, abstractmethod

class Plugin(ABC):
    """Abstract base for all plugins."""

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def execute(self, data):
        pass


class UpperCasePlugin(Plugin):
    @property
    def name(self):
        return "uppercase"

    def execute(self, data):
        return data.upper()


class ReversePlugin(Plugin):
    @property
    def name(self):
        return "reverse"

    def execute(self, data):
        return data[::-1]


class CensorPlugin(Plugin):
    def __init__(self, banned_words=None):
        self.banned_words = banned_words or []

    @property
    def name(self):
        return "censor"

    def execute(self, data):
        result = data
        for word in self.banned_words:
            result = result.replace(word, "*" * len(word))
        return result


class PluginManager:
    """Manage and run plugins in sequence."""

    def __init__(self):
        self._plugins = []

    def register(self, plugin):
        if not isinstance(plugin, Plugin):
            raise TypeError(f"Expected Plugin, got {type(plugin).__name__}")
        self._plugins.append(plugin)
        print(f"Registered plugin: {plugin.name}")

    def process(self, data):
        result = data
        for plugin in self._plugins:
            result = plugin.execute(result)
        return result

    def list_plugins(self):
        return [p.name for p in self._plugins]


# Build a processing pipeline
manager = PluginManager()
manager.register(CensorPlugin(["bad", "ugly"]))
manager.register(UpperCasePlugin())

text = "This is a bad and ugly example"
result = manager.process(text)
print(f"Original: {text}")
print(f"Processed: {result}")
# Original: This is a bad and ugly example
# Processed: THIS IS A *** AND **** EXAMPLE
```

---

## 11. 요약

| 개념 | 핵심 포인트 |
|---------|------------|
| 상속 | `class Child(Parent):`; 부모 메서드 호출에 `super()` 사용 |
| 메서드 오버라이딩 | 서브클래스가 부모 메서드를 재정의; MRO가 해석 순서 결정 |
| 다중 상속 | 쉼표로 구분된 여러 부모; C3 선형화 (MRO) |
| ABC (`abc` 모듈) | `@abstractmethod`로 인터페이스 강제; 직접 인스턴스화 불가 |
| 다형성 | 서로 다른 클래스, 같은 인터페이스; 적합한 모든 객체와 동작 |
| 덕 타이핑 | 타입이 아닌 동작을 검사; "`.write()`가 있으면, writer이다" |
| 연산자 오버로딩 | 던더 메서드 (`__add__`, `__eq__`, `__getitem__` 등) |
| 조합 | "has-a" 관계; 객체가 다른 객체를 포함 |
| `@dataclass` | `__init__`, `__repr__`, `__eq__` 자동 생성; 기본값에 `field()` 사용 |
| 팩토리 패턴 | 매개변수에 기반한 중앙화된 객체 생성 |
| 싱글톤 패턴 | 클래스의 단일 인스턴스 보장 |

---

## 연습문제

1. `Dog`, `Cat`, `Bird` 서브클래스가 있는 `Animal` 계층 구조를 생성하세요. 각각 `speak()`과 `move()` 메서드를 가져야 합니다. 모든 `Animal`을 받아 두 메서드를 호출하는 함수를 사용하세요 (다형성).
2. `serialize(data)`와 `deserialize(text)` 메서드가 있는 추상 `Serializer` 클래스를 정의하세요. `JSONSerializer`와 `CSVSerializer` 서브클래스를 구현하세요.
3. 연산자 오버로딩이 있는 `Money` 클래스를 생성하세요: `+`, `-`, `*` (스칼라), `==`, `<`, `str()`을 지원합니다. 통화를 처리하세요 (예: `Money(10, "USD") + Money(5, "USD")`).
4. 조합을 사용하여 `TaskQueue`를 구축하세요: `Task` 데이터클래스 (이름, 우선순위, 상태 포함)의 리스트를 포함하고 `add`, `pop_highest_priority`, `__len__`, `__iter__`를 지원해야 합니다.
5. 간단한 옵저버 패턴을 구현하세요: `on(event, callback)`, `off(event, callback)`, `emit(event, *args)`를 지원하는 `EventEmitter` 클래스. 여러 리스너로 테스트하세요.

---

**이전**: [OOP 기초](./08_OOP_Basics.md) | **다음**: [모듈과 패키지](./10_Modules_and_Packages.md)
