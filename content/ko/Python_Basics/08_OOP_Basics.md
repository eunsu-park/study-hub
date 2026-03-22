# OOP 기초 (OOP Basics)

**이전**: [문자열과 텍스트 처리](./07_Strings_and_Text_Processing.md) | **다음**: [OOP 고급](./09_OOP_Advanced.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `__init__` 생성자와 `self` 매개변수를 사용하여 클래스를 정의하고 객체를 생성한다
2. 인스턴스 변수 (Instance Variable)와 클래스 변수 (Class Variable)를 구분하고 각각의 사용 시점을 안다
3. 인스턴스 메서드, 클래스 메서드 (`@classmethod`), 정적 메서드 (`@staticmethod`)를 작성한다
4. 읽기 쉬운 객체 표현을 위해 `__str__`과 `__repr__`을 구현한다
5. `@property` 데코레이터를 사용하여 게터와 세터가 있는 관리 속성을 생성한다
6. 캡슐화 (Encapsulation)를 위한 파이썬 명명 규칙 (`public`, `_protected`, `__private`)을 적용한다
7. 생성과 `__del__` 소멸자를 포함한 객체 수명 주기를 이해한다
8. 적절한 관심사 분리로 실세계 엔티티를 모델링하는 간단한 클래스를 설계한다

---

객체 지향 프로그래밍 (Object-Oriented Programming, OOP)은 코드를 객체 중심으로 조직하는 패러다임입니다 -- 객체는 실세계 엔티티를 모델링하는 데이터 (속성)와 동작 (메서드)의 묶음입니다. 별도의 데이터에 대해 동작하는 프로시저를 작성하는 대신, 상태와 동작을 모두 캡슐화하는 클래스를 정의합니다. OOP는 코드 재사용, 모듈성, 유지보수성을 촉진하며, 대규모 파이썬 애플리케이션, 프레임워크, 라이브러리에서 지배적인 패러다임입니다.

## 1. 클래스와 객체

**클래스 (Class)**는 청사진이고, **객체 (Object)** (또는 인스턴스)는 그 청사진의 구체적인 실현입니다.

```python
# Define a class
class Dog:
    pass

# Create objects (instances)
dog1 = Dog()
dog2 = Dog()

print(type(dog1))       # <class '__main__.Dog'>
print(isinstance(dog1, Dog))  # True
print(dog1 is dog2)     # False (different objects)
```

### `__init__` 생성자

`__init__` 메서드는 새로운 객체가 생성될 때 자동으로 호출됩니다. 객체의 속성을 초기화합니다.

```python
class Dog:
    def __init__(self, name, breed, age):
        self.name = name
        self.breed = breed
        self.age = age

# Create objects with initial values
dog1 = Dog("Rex", "German Shepherd", 5)
dog2 = Dog("Buddy", "Golden Retriever", 3)

print(dog1.name)   # Rex
print(dog2.breed)  # Golden Retriever
```

### `self` 매개변수

`self`는 현재 인스턴스를 참조합니다. 항상 인스턴스 메서드의 첫 번째 매개변수이지만, 명시적으로 전달하지는 않습니다.

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius  # self.radius is an instance attribute

    def area(self):
        return 3.14159 * self.radius ** 2

    def circumference(self):
        return 2 * 3.14159 * self.radius

c = Circle(5)
print(f"Area: {c.area():.2f}")             # Area: 78.54
print(f"Circumference: {c.circumference():.2f}")  # Circumference: 31.42

# Behind the scenes, Python translates c.area() to Circle.area(c)
print(f"Area: {Circle.area(c):.2f}")       # Area: 78.54
```

### 기본값이 있는 생성자

```python
class Student:
    def __init__(self, name, grade=0, courses=None):
        self.name = name
        self.grade = grade
        self.courses = courses if courses is not None else []

    def enroll(self, course):
        if course not in self.courses:
            self.courses.append(course)

    def display(self):
        courses_str = ", ".join(self.courses) if self.courses else "None"
        print(f"{self.name} (Grade: {self.grade}) - Courses: {courses_str}")

s1 = Student("Alice", 95)
s1.enroll("Math")
s1.enroll("Physics")
s1.display()  # Alice (Grade: 95) - Courses: Math, Physics

s2 = Student("Bob")
s2.display()  # Bob (Grade: 0) - Courses: None
```

---

## 2. 인스턴스 변수 vs 클래스 변수

### 인스턴스 변수

인스턴스 변수 (Instance Variable)는 각 객체에 속합니다. `__init__` 안에서 `self`를 사용하여 정의됩니다.

```python
class Player:
    def __init__(self, name, score=0):
        self.name = name      # instance variable
        self.score = score    # instance variable

p1 = Player("Alice", 100)
p2 = Player("Bob", 200)

# Each instance has its own copy
print(p1.score)  # 100
print(p2.score)  # 200

p1.score = 150
print(p1.score)  # 150
print(p2.score)  # 200 (unchanged)
```

### 클래스 변수

클래스 변수 (Class Variable)는 모든 인스턴스에서 공유됩니다. 클래스 본문에 직접 정의됩니다.

```python
class Player:
    # Class variable: shared by all instances
    game_name = "Adventure Quest"
    player_count = 0

    def __init__(self, name, score=0):
        self.name = name      # instance variable
        self.score = score    # instance variable
        Player.player_count += 1  # modify class variable

p1 = Player("Alice")
p2 = Player("Bob")
p3 = Player("Charlie")

# Access class variable through class or instance
print(Player.game_name)     # Adventure Quest
print(p1.game_name)         # Adventure Quest
print(Player.player_count)  # 3
```

### 클래스 변수 섀도잉

인스턴스를 통해 속성에 할당하면 클래스 변수를 가리는 새로운 인스턴스 변수가 생성됩니다.

```python
class Config:
    debug = False
    version = "1.0"

c1 = Config()
c2 = Config()

# Both see the class variable
print(c1.debug)  # False
print(c2.debug)  # False

# Assigning via instance creates an instance variable
c1.debug = True
print(c1.debug)  # True  (instance variable)
print(c2.debug)  # False (still class variable)
print(Config.debug)  # False (class variable unchanged)

# Check where the attribute lives
print("debug" in c1.__dict__)  # True (instance has its own)
print("debug" in c2.__dict__)  # False (c2 uses class variable)
```

### 가변 클래스 변수 함정

```python
# BAD: mutable class variable shared by all instances
class StudentBad:
    courses = []  # Shared mutable list!

    def __init__(self, name):
        self.name = name

s1 = StudentBad("Alice")
s2 = StudentBad("Bob")

s1.courses.append("Math")
print(s2.courses)  # ['Math'] -- both students share the same list!

# GOOD: initialize mutable attributes in __init__
class StudentGood:
    def __init__(self, name):
        self.name = name
        self.courses = []  # Each instance gets its own list

s1 = StudentGood("Alice")
s2 = StudentGood("Bob")

s1.courses.append("Math")
print(s2.courses)  # [] -- independent
```

---

## 3. 인스턴스 메서드

인스턴스 메서드는 특정 인스턴스에서 동작하며 `self`를 통해 그 속성에 접근하고 수정할 수 있습니다.

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.transactions = []

    def deposit(self, amount):
        """Add funds to the account."""
        if amount <= 0:
            print("Deposit amount must be positive")
            return
        self.balance += amount
        self.transactions.append(("deposit", amount))
        print(f"Deposited ${amount:.2f}. Balance: ${self.balance:.2f}")

    def withdraw(self, amount):
        """Withdraw funds from the account."""
        if amount <= 0:
            print("Withdrawal amount must be positive")
            return
        if amount > self.balance:
            print(f"Insufficient funds. Balance: ${self.balance:.2f}")
            return
        self.balance -= amount
        self.transactions.append(("withdraw", amount))
        print(f"Withdrew ${amount:.2f}. Balance: ${self.balance:.2f}")

    def get_statement(self):
        """Print account statement."""
        print(f"\n--- Statement for {self.owner} ---")
        for action, amount in self.transactions:
            symbol = "+" if action == "deposit" else "-"
            print(f"  {symbol}${amount:.2f}")
        print(f"  Current Balance: ${self.balance:.2f}")
        print("---")

account = BankAccount("Alice", 1000)
account.deposit(500)       # Deposited $500.00. Balance: $1500.00
account.withdraw(200)      # Withdrew $200.00. Balance: $1300.00
account.withdraw(2000)     # Insufficient funds. Balance: $1300.00
account.deposit(100)       # Deposited $100.00. Balance: $1400.00
account.get_statement()
```

### 메서드 체이닝

메서드에서 `self`를 반환하면 체이닝이 가능합니다.

```python
class QueryBuilder:
    def __init__(self, table):
        self.table = table
        self._columns = "*"
        self._conditions = []
        self._order = None
        self._limit = None

    def select(self, *columns):
        self._columns = ", ".join(columns)
        return self  # Enable chaining

    def where(self, condition):
        self._conditions.append(condition)
        return self

    def order_by(self, column, desc=False):
        direction = "DESC" if desc else "ASC"
        self._order = f"{column} {direction}"
        return self

    def limit(self, n):
        self._limit = n
        return self

    def build(self):
        query = f"SELECT {self._columns} FROM {self.table}"
        if self._conditions:
            query += " WHERE " + " AND ".join(self._conditions)
        if self._order:
            query += f" ORDER BY {self._order}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query

# Fluent interface with method chaining
query = (QueryBuilder("users")
         .select("name", "email", "age")
         .where("age >= 18")
         .where("active = true")
         .order_by("name")
         .limit(10)
         .build())

print(query)
# SELECT name, email, age FROM users WHERE age >= 18 AND active = true ORDER BY name ASC LIMIT 10
```

---

## 4. 클래스 메서드와 정적 메서드

### 클래스 메서드 (`@classmethod`)

클래스 메서드는 인스턴스 대신 클래스 (`cls`)를 첫 번째 인수로 받습니다. 클래스 상태에 접근하고 수정할 수 있습니다.

```python
class Employee:
    raise_percentage = 1.05  # 5% raise
    employee_count = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.employee_count += 1

    def apply_raise(self):
        self.salary *= self.raise_percentage

    @classmethod
    def set_raise_percentage(cls, percentage):
        """Set raise percentage for all employees."""
        cls.raise_percentage = percentage

    @classmethod
    def from_string(cls, employee_str):
        """Alternative constructor from a dash-separated string."""
        name, salary = employee_str.split("-")
        return cls(name, float(salary))

    @classmethod
    def get_count(cls):
        return cls.employee_count

# Regular construction
emp1 = Employee("Alice", 50000)

# Alternative constructor via classmethod
emp2 = Employee.from_string("Bob-60000")
print(emp2.name)     # Bob
print(emp2.salary)   # 60000.0

# Modify class variable
Employee.set_raise_percentage(1.10)
emp1.apply_raise()
print(f"Alice's salary: {emp1.salary:.2f}")  # 55000.00

print(f"Total employees: {Employee.get_count()}")  # 2
```

### 정적 메서드 (`@staticmethod`)

정적 메서드는 `self`나 `cls`를 받지 않습니다. 논리적으로 클래스에 속하지만 인스턴스나 클래스 상태에 접근할 필요가 없는 유틸리티 함수입니다.

```python
class MathUtils:
    @staticmethod
    def is_prime(n):
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def factorial(n):
        """Calculate factorial."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def gcd(a, b):
        """Calculate greatest common divisor."""
        while b:
            a, b = b, a % b
        return a

# Call without creating an instance
print(MathUtils.is_prime(17))     # True
print(MathUtils.factorial(5))     # 120
print(MathUtils.gcd(48, 18))     # 6
```

### 비교: 인스턴스 vs 클래스 vs 정적 메서드

```python
class MyClass:
    class_var = "I am a class variable"

    def __init__(self, value):
        self.instance_var = value

    def instance_method(self):
        """Access instance and class data via self."""
        return f"instance: {self.instance_var}, class: {self.class_var}"

    @classmethod
    def class_method(cls):
        """Access class data via cls. No instance access."""
        return f"class: {cls.class_var}"

    @staticmethod
    def static_method(x, y):
        """No access to instance or class data."""
        return x + y

obj = MyClass("hello")
print(obj.instance_method())   # instance: hello, class: I am a class variable
print(MyClass.class_method())  # class: I am a class variable
print(MyClass.static_method(3, 4))  # 7
```

| 특성 | 인스턴스 메서드 | 클래스 메서드 | 정적 메서드 |
|---------|----------------|--------------|---------------|
| 첫 번째 매개변수 | `self` | `cls` | 없음 |
| 인스턴스 접근? | 예 | 아니오 | 아니오 |
| 클래스 접근? | 예 (`self.__class__` 통해) | 예 (`cls` 통해) | 아니오 |
| 호출 대상 | 인스턴스 | 클래스 또는 인스턴스 | 클래스 또는 인스턴스 |
| 일반적 용도 | 객체 동작 | 대안 생성자, 클래스 수준 연산 | 유틸리티 함수 |

---

## 5. `__str__`과 `__repr__`

이 특수 메서드들은 객체가 어떻게 표시되는지를 제어합니다.

- `__str__`: 사람이 읽기 쉬운 표현 (`print()`와 `str()`에서 사용)
- `__repr__`: 개발자 지향 표현 (`repr()`, 디버거, 대화형 프롬프트에서 사용)

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

p = Point(3, 4)

print(p)           # (3, 4)         -- uses __str__
print(repr(p))     # Point(3, 4)    -- uses __repr__
print(f"Point: {p}")  # Point: (3, 4) -- uses __str__

# In a list, __repr__ is used for elements
points = [Point(1, 2), Point(3, 4)]
print(points)      # [Point(1, 2), Point(3, 4)]  -- uses __repr__
```

### `__repr__`과 `__str__` 가이드라인

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __repr__(self):
        # Should ideally be a valid Python expression that recreates the object
        return f"Temperature({self.celsius})"

    def __str__(self):
        # Should be readable for end users
        return f"{self.celsius}°C ({self.celsius * 9/5 + 32:.1f}°F)"

t = Temperature(100)
print(repr(t))  # Temperature(100)
print(str(t))   # 100°C (212.0°F)

# If only __repr__ is defined, it is used as fallback for __str__
class Simple:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Simple({self.value!r})"

s = Simple("hello")
print(s)        # Simple('hello')  -- __repr__ used as fallback
print(repr(s))  # Simple('hello')
```

---

## 6. `@property` 데코레이터

프로퍼티 (Property)는 속성처럼 접근되는 메서드를 정의할 수 있게 합니다. 계산된 속성과 제어된 접근을 가능하게 합니다.

### 기본 프로퍼티 (게터)

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """Get the radius."""
        return self._radius

    @property
    def area(self):
        """Computed property: area of the circle."""
        return 3.14159 * self._radius ** 2

    @property
    def diameter(self):
        """Computed property: diameter of the circle."""
        return self._radius * 2

c = Circle(5)
print(c.radius)    # 5       (accessed like an attribute)
print(c.area)      # 78.53975
print(c.diameter)  # 10

# Cannot set read-only properties
# c.area = 100  # AttributeError: cannot set attribute
```

### 세터가 있는 프로퍼티

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius  # Store in private attribute

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5 / 9  # Reuses celsius setter validation

t = Temperature(25)
print(f"{t.celsius}°C = {t.fahrenheit}°F")  # 25°C = 77.0°F

t.fahrenheit = 212
print(f"{t.celsius}°C = {t.fahrenheit}°F")  # 100.0°C = 212.0°F

# Validation works
try:
    t.celsius = -300
except ValueError as e:
    print(e)  # Temperature below absolute zero is not possible
```

### 딜리터가 있는 프로퍼티

```python
class CachedData:
    def __init__(self, source):
        self.source = source
        self._cache = None

    @property
    def data(self):
        if self._cache is None:
            print("Loading data from source...")
            self._cache = f"Data from {self.source}"
        return self._cache

    @data.deleter
    def data(self):
        print("Clearing cache...")
        self._cache = None

obj = CachedData("database")
print(obj.data)   # Loading data from source... / Data from database
print(obj.data)   # Data from database (cached, no loading message)

del obj.data      # Clearing cache...
print(obj.data)   # Loading data from source... / Data from database
```

---

## 7. 캡슐화와 명명 규칙

파이썬은 속성과 메서드의 의도된 가시성을 나타내기 위해 명명 규칙을 사용합니다 (Java/C++의 엄격한 접근 제어자가 아닙니다).

### 공개 속성

```python
class User:
    def __init__(self, name, email):
        self.name = name    # public: accessible from anywhere
        self.email = email  # public

user = User("Alice", "alice@example.com")
print(user.name)      # Alice
user.name = "Bob"     # OK, can modify directly
```

### 보호 속성 (`_단일_밑줄`)

단일 선행 밑줄은 "내부 사용"을 의미하는 규칙입니다. 클래스 외부에서 접근하지 말아야 한다는 신호이지만, 파이썬은 이를 강제하지 않습니다.

```python
class Account:
    def __init__(self, owner, balance):
        self.owner = owner
        self._balance = balance  # protected by convention

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def get_balance(self):
        return self._balance

acc = Account("Alice", 1000)
print(acc.get_balance())  # 1000

# Still accessible (Python trusts the developer)
print(acc._balance)       # 1000 -- works, but discouraged
```

### 비공개 속성 (`__이중_밑줄`)

이중 선행 밑줄은 **이름 맹글링 (Name Mangling)**을 유발합니다: 파이썬이 속성 이름을 `_ClassName__attribute`로 변경하여 우발적 접근을 더 어렵게 만듭니다.

```python
class SecureAccount:
    def __init__(self, owner, balance, pin):
        self.owner = owner
        self._balance = balance   # protected
        self.__pin = pin          # private (name-mangled)

    def verify_pin(self, pin):
        return self.__pin == pin

    def get_balance(self, pin):
        if self.verify_pin(pin):
            return self._balance
        return "Invalid PIN"

acc = SecureAccount("Alice", 5000, "1234")
print(acc.get_balance("1234"))  # 5000
print(acc.get_balance("0000"))  # Invalid PIN

# Direct access fails
# print(acc.__pin)  # AttributeError: 'SecureAccount' has no attribute '__pin'

# But name mangling can be bypassed (not truly private)
print(acc._SecureAccount__pin)  # 1234 -- possible but strongly discouraged
```

### 명명 규칙 요약

| 규칙 | 예시 | 의미 |
|------------|---------|---------|
| `name` | `self.name` | 공개 -- 어디서나 자유롭게 사용 |
| `_name` | `self._name` | 보호 -- 내부 사용, 외부 접근 자제 |
| `__name` | `self.__name` | 비공개 -- `_Class__name`으로 이름 맹글링 |
| `__name__` | `self.__init__` | 던더/매직 -- 파이썬 특수 메서드 |

---

## 8. 객체 수명 주기

### 생성: `__new__`와 `__init__`

`__new__`는 인스턴스를 생성하고, `__init__`은 그것을 초기화합니다. 대부분의 경우 `__init__`만 오버라이드합니다.

```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        print(f"1. __new__ called (creating instance of {cls.__name__})")
        instance = super().__new__(cls)
        return instance

    def __init__(self, value):
        print(f"2. __init__ called (initializing with {value})")
        self.value = value

obj = MyClass(42)
# 1. __new__ called (creating instance of MyClass)
# 2. __init__ called (initializing with 42)
```

### 소멸: `__del__`

`__del__`은 객체가 가비지 컬렉션되기 직전에 호출됩니다. 실무에서는 거의 필요하지 않습니다.

```python
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Resource '{self.name}' created")

    def __del__(self):
        print(f"Resource '{self.name}' destroyed")

# Normal lifecycle
r = Resource("file_handler")  # Resource 'file_handler' created
del r                          # Resource 'file_handler' destroyed

# Also triggered when reference count drops to zero
def demo():
    r = Resource("temp")  # Resource 'temp' created
    print("Inside function")
    # r goes out of scope when function returns

demo()
# Inside function
# Resource 'temp' destroyed (eventually, when garbage collected)
```

### `__del__`에 대한 중요 참고사항

```python
# __del__ is NOT guaranteed to be called immediately
# Prefer context managers (with statement) for cleanup

class FileHandler:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        print(f"Opened {filename}")

    def write(self, data):
        self.file.write(data)

    def __del__(self):
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()
            print(f"Closed {self.filename}")

# Better approach: context manager protocol
class BetterFileHandler:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        print(f"Closed {self.filename}")
        return False

# Usage
# with BetterFileHandler("output.txt") as handler:
#     handler.file.write("Hello!")
# File is guaranteed to close here
```

---

## 9. 실용적인 예제

### 예제: 학생 성적 추적기

```python
class GradeTracker:
    """Track and analyze student grades."""

    # Class variable: grading scale
    GRADE_SCALE = {
        "A+": (97, 100), "A": (93, 96), "A-": (90, 92),
        "B+": (87, 89),  "B": (83, 86), "B-": (80, 82),
        "C+": (77, 79),  "C": (73, 76), "C-": (70, 72),
        "D+": (67, 69),  "D": (63, 66), "D-": (60, 62),
        "F": (0, 59),
    }

    def __init__(self, student_name):
        self.student_name = student_name
        self._grades = {}  # subject -> list of scores

    def add_grade(self, subject, score):
        """Add a grade for a subject."""
        if not 0 <= score <= 100:
            raise ValueError(f"Score must be 0-100, got {score}")
        if subject not in self._grades:
            self._grades[subject] = []
        self._grades[subject].append(score)

    @property
    def subjects(self):
        """List of all subjects."""
        return list(self._grades.keys())

    @property
    def gpa(self):
        """Calculate overall GPA on a 4.0 scale."""
        if not self._grades:
            return 0.0
        all_scores = []
        for scores in self._grades.values():
            all_scores.extend(scores)
        avg = sum(all_scores) / len(all_scores)
        return min(4.0, avg / 25)  # Simple 4.0 scale approximation

    def get_average(self, subject=None):
        """Get average score, optionally for a specific subject."""
        if subject:
            scores = self._grades.get(subject, [])
            return sum(scores) / len(scores) if scores else 0
        all_scores = [s for scores in self._grades.values() for s in scores]
        return sum(all_scores) / len(all_scores) if all_scores else 0

    @staticmethod
    def score_to_letter(score):
        """Convert numeric score to letter grade."""
        for letter, (low, high) in GradeTracker.GRADE_SCALE.items():
            if low <= score <= high:
                return letter
        return "N/A"

    def __str__(self):
        avg = self.get_average()
        letter = self.score_to_letter(avg) if avg else "N/A"
        return f"{self.student_name} - Average: {avg:.1f} ({letter})"

    def __repr__(self):
        return f"GradeTracker({self.student_name!r})"

# Usage
tracker = GradeTracker("Alice")
tracker.add_grade("Math", 95)
tracker.add_grade("Math", 88)
tracker.add_grade("Science", 92)
tracker.add_grade("English", 87)

print(tracker)  # Alice - Average: 90.5 (A-)
print(f"Math average: {tracker.get_average('Math'):.1f}")  # Math average: 91.5
print(f"Subjects: {tracker.subjects}")  # Subjects: ['Math', 'Science', 'English']
print(f"GPA: {tracker.gpa:.2f}")  # GPA: 3.62
```

### 예제: 프로퍼티가 있는 재고 항목

```python
class InventoryItem:
    """Represent an item in inventory with price and quantity validation."""

    _tax_rate = 0.10  # Class-level tax rate (10%)

    def __init__(self, name, price, quantity=0):
        self.name = name
        self.price = price        # Uses property setter
        self.quantity = quantity   # Uses property setter

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError(f"Price cannot be negative: {value}")
        self._price = round(value, 2)

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Quantity must be a non-negative integer: {value}")
        self._quantity = value

    @property
    def total_value(self):
        """Total value of this item in inventory."""
        return self._price * self._quantity

    @property
    def price_with_tax(self):
        """Price including tax."""
        return self._price * (1 + self._tax_rate)

    @classmethod
    def set_tax_rate(cls, rate):
        """Set tax rate for all items."""
        if not 0 <= rate <= 1:
            raise ValueError("Tax rate must be between 0 and 1")
        cls._tax_rate = rate

    @classmethod
    def from_dict(cls, data):
        """Create an InventoryItem from a dictionary."""
        return cls(
            name=data["name"],
            price=data["price"],
            quantity=data.get("quantity", 0),
        )

    def restock(self, amount):
        """Add items to inventory."""
        self.quantity += amount

    def sell(self, amount):
        """Remove items from inventory."""
        if amount > self._quantity:
            raise ValueError(f"Cannot sell {amount}, only {self._quantity} in stock")
        self.quantity -= amount

    def __str__(self):
        return f"{self.name}: ${self._price:.2f} x {self._quantity} = ${self.total_value:.2f}"

    def __repr__(self):
        return f"InventoryItem({self.name!r}, {self._price}, {self._quantity})"

# Usage
item = InventoryItem("Widget", 9.99, 100)
print(item)  # Widget: $9.99 x 100 = $999.00

item.sell(30)
print(item)  # Widget: $9.99 x 70 = $699.30

print(f"Price with tax: ${item.price_with_tax:.2f}")  # Price with tax: $10.99

# From dictionary
data = {"name": "Gadget", "price": 24.99, "quantity": 50}
item2 = InventoryItem.from_dict(data)
print(item2)  # Gadget: $24.99 x 50 = $1249.50

# Validation
try:
    item.price = -5
except ValueError as e:
    print(e)  # Price cannot be negative: -5
```

### 예제: 간단한 연결 리스트 노드

```python
class Node:
    """A node in a singly linked list."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data!r})"

    def __str__(self):
        parts = []
        current = self
        while current:
            parts.append(str(current.data))
            current = current.next
        return " -> ".join(parts) + " -> None"

# Build a linked list: 1 -> 2 -> 3 -> None
head = Node(1, Node(2, Node(3)))
print(head)  # 1 -> 2 -> 3 -> None

# Traverse
current = head
while current:
    print(f"Visiting: {current.data}")
    current = current.next
# Visiting: 1
# Visiting: 2
# Visiting: 3
```

---

## 10. 요약

| 개념 | 핵심 포인트 |
|---------|------------|
| 클래스 | `class`로 정의된 청사진; `ClassName()`으로 객체 생성 |
| `__init__` | 생성자; 인스턴스 속성 초기화 |
| `self` | 현재 인스턴스에 대한 참조; 항상 첫 번째 매개변수 |
| 인스턴스 변수 | 객체별 데이터 (`self.attr`); `__init__`에서 정의 |
| 클래스 변수 | 공유 데이터; 클래스 본문에서 정의 |
| 인스턴스 메서드 | `self`에서 동작; 인스턴스와 클래스 데이터에 접근 |
| `@classmethod` | `cls`에서 동작; 대안 생성자, 클래스 수준 연산 |
| `@staticmethod` | `self`나 `cls` 없음; 클래스에 그룹화된 유틸리티 함수 |
| `__str__`/`__repr__` | 사람 읽기용 / 개발자 표현 |
| `@property` | 게터, 세터, 딜리터가 있는 관리 속성 |
| 캡슐화 | `public`, `_protected`, `__private` (이름 맹글링) |
| `__del__` | 소멸자; 가비지 컬렉션 전에 호출 (타이밍 불확실) |

---

## 연습문제

1. `width`와 `height` 프로퍼티 (양수 검증), 계산된 `area`와 `perimeter` 프로퍼티, `__str__`/`__repr__` 메서드가 있는 `Rectangle` 클래스를 생성하세요.
2. 노래 목록을 관리하는 `Playlist` 클래스를 구축하세요. `add_song`, `remove_song`, `shuffle`, `total_duration`과 문자열 리스트에서 노래 데이터를 읽는 `from_file` 클래스 메서드를 포함하세요.
3. 항목의 카운트를 추적하는 `Counter` 클래스를 구현하세요 (`collections.Counter`와 유사). `add`, `remove`, `most_common(n)`을 포함하고 표시를 위한 `__str__`을 지원하세요.
4. 2D 숫자 그리드를 저장하는 `Matrix` 클래스를 생성하세요. `rows`, `cols`, `shape`에 대한 프로퍼티를 추가하세요. n x n 단위 행렬을 생성하는 `@classmethod` 팩토리 `identity(n)`을 포함하세요.
5. `__private` 잔액, 잔액에 대한 `@property` (읽기 전용), 유효성 검사가 포함된 `deposit`/`withdraw` 메서드, 거래 내역이 있는 `BankAccount` 클래스를 설계하세요.

---

**이전**: [문자열과 텍스트 처리](./07_Strings_and_Text_Processing.md) | **다음**: [OOP 고급](./09_OOP_Advanced.md)
