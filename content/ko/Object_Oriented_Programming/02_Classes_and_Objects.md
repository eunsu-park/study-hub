# 레슨 02: 클래스와 객체

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. Python으로 속성과 메서드가 있는 클래스를 정의할 수 있다
2. 클래스 속성과 인스턴스 속성의 차이를 구분할 수 있다
3. 객체 인스턴스를 생성하고 사용할 수 있다
4. 클래스와 인스턴스의 관계를 설명할 수 있다
5. 클래스 메서드와 정적 메서드를 적절히 사용할 수 있다
6. 객체 생명주기(생성, 사용, 소멸)를 이해할 수 있다
7. 속성 조회 시 네임스페이스 해결 규칙을 적용할 수 있다

## 클래스: 청사진

**클래스**는 객체의 구조와 동작을 정의하는 청사진입니다. 건축 설계도와 같습니다 — 설계도 자체는 집이 아니지만, 같은 설계도로 많은 집을 지을 수 있습니다.

```
┌────────────────────────────┐
│       CLASS: Car           │  <-- 청사진 (템플릿)
├────────────────────────────┤
│  속성:                     │
│    - make                  │
│    - model                 │
│    - year                  │
│    - mileage               │
├────────────────────────────┤
│  메서드:                   │
│    - start()               │
│    - drive(distance)       │
│    - stop()                │
│    - describe()            │
└─────────┬──────────────────┘
          │ 인스턴스화
          ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  객체 #1     │  │  객체 #2     │  │  객체 #3     │
│  make: Toyota│  │  make: BMW   │  │  make: Ford  │
│  model: Camry│  │  model: X3   │  │  model: F150 │
│  year: 2022  │  │  year: 2023  │  │  year: 2021  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 클래스 정의하기

```python
class Car:
    """자동차를 나타내는 클래스."""

    # 클래스 속성: 모든 인스턴스가 공유
    wheel_count = 4

    def __init__(self, make, model, year):
        """새로운 Car 인스턴스 초기화."""
        # 인스턴스 속성: 각 인스턴스에 고유
        self.make = make
        self.model = model
        self.year = year
        self.mileage = 0.0
        self.is_running = False

    def start(self):
        """자동차 엔진을 시작합니다."""
        if self.is_running:
            print(f"{self.make} {self.model}는 이미 작동 중입니다.")
        else:
            self.is_running = True
            print(f"{self.make} {self.model} 시동이 걸렸습니다.")

    def drive(self, distance):
        """일정 거리를 운전합니다."""
        if not self.is_running:
            print("먼저 시동을 걸어주세요!")
            return
        if distance <= 0:
            raise ValueError("거리는 양수여야 합니다")
        self.mileage += distance
        print(f"{distance}마일 주행. 총: {self.mileage}")

    def stop(self):
        """자동차 엔진을 정지합니다."""
        self.is_running = False
        print(f"{self.make} {self.model} 정지했습니다.")

    def describe(self):
        """자동차 설명을 반환합니다."""
        return f"{self.year} {self.make} {self.model} ({self.mileage:.0f}마일)"
```

## 객체: 인스턴스

**객체**는 클래스로부터 생성된 특정 인스턴스입니다. 각 객체는 인스턴스 속성의 자체 사본을 가지지만 클래스 속성은 공유합니다.

```python
# 객체 생성 (인스턴스화)
car1 = Car("Toyota", "Camry", 2022)
car2 = Car("BMW", "X3", 2023)

# 각 객체는 독립적
car1.start()           # Toyota Camry 시동이 걸렸습니다.
car1.drive(50)         # 50마일 주행. 총: 50.0

car2.start()           # BMW X3 시동이 걸렸습니다.
car2.drive(30)         # 30마일 주행. 총: 30.0

# 서로 다른 주행거리 (인스턴스 속성은 독립적)
print(car1.mileage)    # 50.0
print(car2.mileage)    # 30.0

# 같은 바퀴 수 (클래스 속성은 공유)
print(car1.wheel_count)  # 4
print(car2.wheel_count)  # 4
```

### 정체성, 동등성, 타입

Python에서 모든 객체는 세 가지 기본 속성을 가집니다:

```python
car1 = Car("Toyota", "Camry", 2022)
car2 = Car("Toyota", "Camry", 2022)

# 정체성: 메모리의 고유 ID
print(id(car1))              # 예: 140234567890
print(id(car2))              # 다른 숫자
print(car1 is car2)          # False — 다른 객체

# 타입: 어떤 클래스에서 왔는지
print(type(car1))            # <class '__main__.Car'>
print(isinstance(car1, Car)) # True

# 동등성: 기본적으로 객체는 자기 자신에게만 동등
print(car1 == car2)          # False (기본: `is`와 동일)
```

## 클래스 속성 vs 인스턴스 속성

차이를 이해하는 것이 매우 중요합니다:

```
┌──────────────────────────────────────────────────┐
│            Class: Employee                        │
│  ┌──────────────────────────────────┐            │
│  │ 클래스 속성 (공유)              │            │
│  │   company = "Acme Corp"         │            │
│  │   employee_count = 0            │            │
│  └──────────────────────────────────┘            │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐               │
│  │ 인스턴스 #1 │  │ 인스턴스 #2 │               │
│  │ name: Alice │  │ name: Bob   │               │
│  │ salary: 75k │  │ salary: 80k │               │
│  │ dept: Eng   │  │ dept: Sales │               │
│  └─────────────┘  └─────────────┘               │
└──────────────────────────────────────────────────┘
```

### 속성 조회 순서 (네임스페이스 해결)

`obj.attr`에 접근할 때, Python은 다음 순서로 검색합니다:

```
1. 인스턴스 네임스페이스  (obj.__dict__)
       │
       ▼ 찾지 못함?
2. 클래스 네임스페이스     (type(obj).__dict__)
       │
       ▼ 찾지 못함?
3. 부모 클래스(들)         (MRO를 통해)
       │
       ▼ 찾지 못함?
4. AttributeError 발생
```

## 메서드: 인스턴스, 클래스, 정적

Python은 세 가지 유형의 메서드를 지원합니다:

```python
class MathHelper:
    """세 가지 메서드 유형을 보여줍니다."""

    precision = 2  # 클래스 속성

    def __init__(self, name):
        self.name = name  # 인스턴스 속성

    # 인스턴스 메서드: 인스턴스(self)와 클래스에 접근 가능
    def greet(self):
        return f"저는 {self.name}이고, 정밀도={self.precision}입니다"

    # 클래스 메서드: 클래스(cls)에 접근 가능하지만 인스턴스에는 불가
    @classmethod
    def set_precision(cls, value):
        cls.precision = value

    # 정적 메서드: 인스턴스도 클래스도 접근 불가
    @staticmethod
    def add(a, b):
        return a + b
```

### 언제 어떤 것을 사용할까

```
┌─────────────────────────────────────────────────────────┐
│  메서드 유형    │ self? │ cls? │ 용도               │
├─────────────────┼───────┼──────┼────────────────────┤
│ 인스턴스 메서드 │  예   │(self │ 대부분의 메서드:   │
│  def foo(self)  │       │통해) │ 인스턴스 데이터    │
│                 │       │      │ 조작               │
├─────────────────┼───────┼──────┼────────────────────┤
│ 클래스 메서드   │ 아니오│  예  │ 대체 생성자,       │
│  @classmethod   │       │      │ 클래스 전체 작업   │
├─────────────────┼───────┼──────┼────────────────────┤
│ 정적 메서드     │ 아니오│아니오│ 유틸리티 함수      │
│  @staticmethod  │       │      │ (클래스와 논리적   │
│                 │       │      │ 으로 그룹화)       │
└─────────────────┴───────┴──────┴────────────────────┘
```

### 대체 생성자로서의 클래스 메서드

`@classmethod`를 사용하여 객체를 생성하는 여러 방법을 제공하는 일반적인 패턴:

```python
class Date:
    """여러 생성자를 가진 날짜 클래스."""

    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def from_string(cls, date_string):
        """'YYYY-MM-DD' 형식의 문자열에서 Date 생성."""
        year, month, day = map(int, date_string.split("-"))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        """오늘 날짜의 Date 생성."""
        import datetime
        t = datetime.date.today()
        return cls(t.year, t.month, t.day)

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"
```

## 객체 생명주기

```
    ┌────────────┐
    │  클래스     │
    │  정의       │
    └──────┬─────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ __new__()  │────▶│ 메모리 할당  │
    └──────┬─────┘     └──────────────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ __init__() │────▶│ 속성 초기화  │
    └──────┬─────┘     └──────────────┘
           │
    ┌──────▼─────┐
    │  객체 사용 │◄──── 메서드 호출, 속성 접근
    └──────┬─────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ 참조 없음  │────▶│ 참조 카운트  │
    │            │     │ = 0          │
    └──────┬─────┘     └──────────────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ __del__()  │────▶│ 가비지       │
    │            │     │ 컬렉션       │
    └────────────┘     └──────────────┘
```

## 실전 예제: 도서관 시스템

```python
class Book:
    """도서관 시스템의 도서."""

    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_checked_out = False
        self.borrower = None

    def check_out(self, borrower_name):
        """도서를 대출합니다."""
        if self.is_checked_out:
            raise RuntimeError(f"'{self.title}'은(는) 이미 대출 중입니다")
        self.is_checked_out = True
        self.borrower = borrower_name

    def return_book(self):
        """도서를 반납합니다."""
        self.is_checked_out = False
        self.borrower = None

    def __repr__(self):
        status = f"({self.borrower}에게 대출 중)" if self.is_checked_out else "(이용 가능)"
        return f"Book('{self.title}' by {self.author}) {status}"


class Library:
    """도서 컬렉션을 보유하는 도서관."""

    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def find_by_title(self, title):
        return [b for b in self.books if title.lower() in b.title.lower()]

    def available_books(self):
        return [b for b in self.books if not b.is_checked_out]

    def __len__(self):
        return len(self.books)
```

## 요약

- **클래스**는 청사진이고, **객체**는 그 청사진에서 생성된 인스턴스입니다
- **클래스 속성**은 모든 인스턴스가 공유하고, **인스턴스 속성**은 객체마다 고유합니다
- Python은 속성을 인스턴스 -> 클래스 -> 부모 클래스 순서로 조회합니다
- 세 가지 메서드 유형: **인스턴스 메서드** (`self`), **클래스 메서드** (`cls`), **정적 메서드** (둘 다 없음)
- 클래스 메서드는 일반적으로 **대체 생성자**로 사용됩니다
- 객체는 생명주기를 거칩니다: `__new__` (할당) -> `__init__` (초기화) -> 사용 -> `__del__` (소멸)

## 다음 단계

[레슨 03: 생성자와 초기화](03_Constructors_and_Initialization.md)에서 `__init__` 메서드를 깊이 탐구하며, 매개변수 검증, 기본값, 초기화 패턴을 다룹니다.
