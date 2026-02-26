# 데코레이터 (Decorators)

**이전**: [타입 힌트](./01_Type_Hints.md) | **다음**: [컨텍스트 매니저](./03_Context_Managers.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `@decorator` 문법이 함수 래핑(wrapping)의 syntactic sugar임을 설명할 수 있습니다
2. `*args`와 `**kwargs`를 받는 기본 데코레이터를 구현할 수 있습니다
3. `functools.wraps`를 적용해 데코레이트된 함수의 메타데이터를 보존할 수 있습니다
4. 자체 설정 인자를 받는 데코레이터를 작성할 수 있습니다(3단계 중첩 구조)
5. `__call__` 메서드를 사용해 클래스 기반(class-based) 데코레이터를 구현할 수 있습니다
6. `@property`, `@staticmethod`, `@classmethod`, `@lru_cache`의 차이를 구분할 수 있습니다
7. 데코레이터 체이닝(chaining)을 적용하고 아래에서 위로의 평가 순서를 설명할 수 있습니다
8. 클래스 전체를 수정하거나 확장하는 클래스 데코레이터를 작성할 수 있습니다(예: 싱글톤 패턴)

---

데코레이터는 파이썬에서 가장 강력한 추상화 메커니즘 중 하나입니다. 로깅 추가, 인증 강제, 비용이 큰 결과 캐싱, 입력 검증 등을 처리할 때마다 동일한 문제에 직면하게 됩니다. 바로 핵심 함수를 지저분하게 만드는 횡단 관심사(cross-cutting logic)입니다. 데코레이터를 사용하면 이러한 반복 코드를 재사용 가능한 래퍼로 분리해 비즈니스 로직을 깔끔하게 유지하면서, 단 하나의 `@` 줄로 선언적으로 기능을 추가할 수 있습니다. Flask, Django, pytest 같은 프레임워크는 라우팅, 미들웨어, 테스트 픽스처에 데코레이터를 적극적으로 활용합니다.

> **비유:** 데코레이터는 선물 포장지가 선물을 감싸는 방식으로 함수를 감쌉니다. 안의 선물은 변하지 않지만, 포장이 리본(로깅), 카드(타이밍 측정), 보안 테이프(인증)를 바깥에 덧붙입니다.

## 1. 데코레이터란?

데코레이터는 함수나 클래스를 수정하지 않고 기능을 추가하는 패턴입니다. `@` 문법을 사용하여 적용합니다.

```python
@decorator
def function():
    pass

# 위 코드는 아래와 동일
def function():
    pass
function = decorator(function)
```

### 데코레이터의 구조

```
┌─────────────────────────────────────────┐
│              데코레이터                   │
│  ┌─────────────────────────────────┐    │
│  │         wrapper 함수             │    │
│  │  ┌─────────────────────────┐    │    │
│  │  │     원본 함수 호출        │    │    │
│  │  └─────────────────────────┘    │    │
│  │  + 추가 기능 (전/후)            │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## 2. 기본 데코레이터

### 가장 단순한 형태

```python
def my_decorator(func):
    def wrapper():
        print("함수 실행 전")
        func()
        print("함수 실행 후")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

출력:
```
함수 실행 전
Hello!
함수 실행 후
```

### 인자를 받는 함수에 적용

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"인자: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"결과: {result}")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

add(3, 5)
```

출력:
```
인자: (3, 5), {}
결과: 8
```

---

## 3. @wraps로 메타데이터 보존

데코레이터를 적용하면 원본 함수의 메타데이터(이름, docstring 등)가 사라집니다.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """인사 함수"""
    return f"Hello, {name}"

print(greet.__name__)  # wrapper (원본 이름이 아님!)
print(greet.__doc__)   # None (docstring 손실!)
```

### functools.wraps 사용

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # @wraps가 없으면 greet.__name__이 'wrapper'가 되어 help(), 로깅(logging),
                  # 디버거나 문서 생성기 같은 인트로스펙션(introspection) 기반 도구들이 잘못된 정보를 보게 됨
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """인사 함수"""
    return f"Hello, {name}"

print(greet.__name__)  # greet (보존됨!)
print(greet.__doc__)   # 인사 함수 (보존됨!)
```

---

## 4. 인자를 받는 데코레이터

데코레이터 자체에 인자를 전달하려면 한 단계 더 감싸야 합니다.

```python
from functools import wraps

def repeat(times):
    """함수를 n번 반복 실행하는 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hi():
    print("Hi!")

say_hi()
```

출력:
```
Hi!
Hi!
Hi!
```

### 실용 예제: 권한 검사

```python
from functools import wraps

def require_role(role):
    """특정 역할이 필요한 함수에 적용"""
    def decorator(func):
        @wraps(func)
        def wrapper(user, *args, **kwargs):
            if user.get("role") != role:
                raise PermissionError(f"'{role}' 권한이 필요합니다")
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

@require_role("admin")
def delete_user(user, target_id):
    print(f"사용자 {target_id} 삭제됨")

admin = {"name": "Alice", "role": "admin"}
guest = {"name": "Bob", "role": "guest"}

delete_user(admin, 123)  # OK
# delete_user(guest, 123)  # PermissionError!
```

---

## 5. 클래스 기반 데코레이터

`__call__` 메서드를 구현하면 클래스를 데코레이터로 사용할 수 있습니다.

```python
class CountCalls:
    """함수 호출 횟수를 추적하는 데코레이터"""

    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} 호출 횟수: {self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()  # 호출 횟수: 1
say_hello()  # 호출 횟수: 2
say_hello()  # 호출 횟수: 3
```

### 인자를 받는 클래스 데코레이터

```python
class Retry:
    """실패 시 재시도하는 데코레이터.

    왜 인라인(inline) try/except 대신 데코레이터를 사용하는가? 재시도 로직은 횡단 관심사(cross-cutting concern)로,
    취약한 호출 지점마다 동일한 반복 코드가 필요하게 됨. 데코레이터를 사용하면 호출 지점은 깔끔하게
    유지되고, 재시도 정책을 한 곳에서 변경할 수 있음.
    """

    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"시도 {attempt} 실패: {e}")
                    if attempt == self.max_attempts:
                        raise  # 마지막 시도에서 원래 예외를 다시 발생시킴
        return wrapper

@Retry(max_attempts=3)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("랜덤 실패!")
    return "성공!"

result = unstable_function()
```

---

## 6. 내장 데코레이터

### @property

getter/setter/deleter를 정의합니다.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """반지름 (읽기)"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """반지름 (쓰기)"""
        if value < 0:
            raise ValueError("반지름은 양수여야 합니다")
        self._radius = value

    @property
    def area(self):
        """면적 (계산된 속성)"""
        return 3.14159 * self._radius ** 2

circle = Circle(5)
print(circle.radius)  # 5
print(circle.area)    # 78.53975

circle.radius = 10
print(circle.area)    # 314.159
```

### @staticmethod

인스턴스나 클래스에 접근하지 않는 메서드입니다.

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def multiply(a, b):
        return a * b

# 인스턴스 없이 호출
print(Math.add(3, 5))       # 8
print(Math.multiply(3, 5))  # 15
```

### @classmethod

클래스를 첫 번째 인자로 받는 메서드입니다.

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def from_string(cls, date_string):
        """문자열에서 Date 생성"""
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        """오늘 날짜로 Date 생성"""
        import datetime
        t = datetime.date.today()
        return cls(t.year, t.month, t.day)

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

date1 = Date.from_string("2024-01-23")
date2 = Date.today()
print(date1)  # Date(2024, 1, 23)
```

---

## 7. 실용적 데코레이터 패턴

### 타이밍 측정

```python
import time
from functools import wraps

def timer(func):
    """함수 실행 시간 측정"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}초")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "완료"

slow_function()  # slow_function: 1.0012초
```

### 로깅

```python
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)

def log_calls(func):
    """함수 호출을 로깅"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"호출: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        logging.info(f"반환: {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

add(3, 5)
```

### 캐싱 (메모이제이션)

```python
from functools import wraps

def memoize(func):
    """결과를 캐싱하는 데코레이터"""
    cache = {}  # 클로저(closure)에 캡처된 딕셔너리가 호출 간에 유지됨; 키는 인자 튜플
                # (해시 가능(hashable)해야 함 — 리스트나 딕셔너리를 인자로 전달하면 실패)

    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # 캐싱 없이는 매우 느림
```

**참고**: Python 내장 `functools.lru_cache`를 사용하면 더 편리합니다.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### 입력 검증

```python
from functools import wraps

def validate_types(**expected_types):
    """인자 타입을 검증하는 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 키워드 인자 검증
            for name, expected in expected_types.items():
                if name in kwargs:
                    if not isinstance(kwargs[name], expected):
                        raise TypeError(
                            f"{name}은(는) {expected.__name__} 타입이어야 합니다"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}

create_user(name="Alice", age=30)  # OK
# create_user(name="Alice", age="30")  # TypeError!
```

---

## 8. 데코레이터 체이닝

여러 데코레이터를 동시에 적용할 수 있습니다. 적용 순서는 **아래에서 위**입니다.

```python
@decorator1
@decorator2
@decorator3
def func():
    pass

# 위 코드는 아래와 동일
func = decorator1(decorator2(decorator3(func)))
```

### 예제

```python
from functools import wraps

def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def greet(name):
    return f"Hello, {name}"

print(greet("World"))  # <b><i>Hello, World</i></b>
```

---

## 9. 클래스 데코레이터

클래스 전체에 데코레이터를 적용할 수 있습니다.

```python
def singleton(cls):
    """싱글톤 패턴 데코레이터"""
    instances = {}  # 문자열 이름이 아닌 클래스 객체를 키로 사용하여 서브클래스를 독립적으로 관리

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("데이터베이스 연결 생성")

db1 = Database()  # 데이터베이스 연결 생성
db2 = Database()  # (출력 없음 - 같은 인스턴스)
print(db1 is db2)  # True
```

### dataclass (내장 클래스 데코레이터)

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3, 4)
print(p)              # Point(x=3, y=4)
print(p.distance())   # 5.0
print(p == Point(3, 4))  # True
```

---

## 10. 요약

| 패턴 | 설명 | 예시 |
|------|------|------|
| 기본 데코레이터 | 함수를 감싸서 기능 추가 | `@timer` |
| 인자 있는 데코레이터 | 데코레이터에 설정 전달 | `@repeat(3)` |
| 클래스 기반 데코레이터 | 상태 유지가 필요할 때 | `@CountCalls` |
| @wraps | 메타데이터 보존 | `@wraps(func)` |
| @property | getter/setter 정의 | `@property` |
| @staticmethod | 정적 메서드 | `@staticmethod` |
| @classmethod | 클래스 메서드 | `@classmethod` |
| @lru_cache | 결과 캐싱 | `@lru_cache(128)` |

---

## 11. 연습 문제

### 연습 1: 실행 시간 제한

함수가 지정된 시간 내에 완료되지 않으면 TimeoutError를 발생시키는 데코레이터를 작성하세요.

### 연습 2: 결과 로깅

함수의 입력과 출력을 파일에 로깅하는 데코레이터를 작성하세요.

### 연습 3: 디버그 모드

DEBUG 플래그가 True일 때만 디버그 정보를 출력하는 데코레이터를 작성하세요.

---

## 다음 단계

[컨텍스트 매니저 (Context Managers)](./03_Context_Managers.md)에서 with문과 리소스 관리를 배워봅시다!
