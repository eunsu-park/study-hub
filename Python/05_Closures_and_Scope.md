# 클로저와 스코프 (Closures & Scope)

## 1. 변수 스코프

파이썬에서 변수는 정의된 위치에 따라 접근 범위가 결정됩니다.

### LEGB 규칙

변수를 찾는 순서입니다.

```
┌─────────────────────────────────────────────┐
│ B - Built-in (내장)                          │
│   print, len, range, ...                    │
│ ┌─────────────────────────────────────────┐ │
│ │ G - Global (전역)                        │ │
│ │   모듈 레벨에서 정의된 변수              │ │
│ │ ┌─────────────────────────────────────┐ │ │
│ │ │ E - Enclosing (감싸는 함수)          │ │ │
│ │ │   바깥 함수의 지역 변수              │ │ │
│ │ │ ┌─────────────────────────────────┐ │ │ │
│ │ │ │ L - Local (지역)                │ │ │ │
│ │ │ │   현재 함수 내부                │ │ │ │
│ │ │ └─────────────────────────────────┘ │ │ │
│ │ └─────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### 예제

```python
# Built-in (B)
# print, len, str, ...

# Global (G)
x = "global"

def outer():
    # Enclosing (E)
    x = "enclosing"

    def inner():
        # Local (L)
        x = "local"
        print(x)  # local

    inner()
    print(x)  # enclosing

outer()
print(x)  # global
```

---

## 2. global 키워드

함수 내에서 전역 변수를 수정할 때 사용합니다.

```python
count = 0

def increment():
    global count  # 전역 변수 사용 선언
    count += 1

increment()
increment()
print(count)  # 2
```

### global 없이 수정하면?

```python
count = 0

def increment():
    count += 1  # UnboundLocalError!
    # count = count + 1 에서 count를 지역 변수로 인식

increment()
```

### 읽기만 할 때는 불필요

```python
name = "Python"

def greet():
    print(f"Hello, {name}")  # global 없이 읽기 가능

greet()  # Hello, Python
```

---

## 3. nonlocal 키워드

감싸는 함수의 변수를 수정할 때 사용합니다.

```python
def outer():
    count = 0

    def inner():
        nonlocal count  # 바깥 함수의 변수 사용 선언
        count += 1

    inner()
    inner()
    print(count)  # 2

outer()
```

### global vs nonlocal

```python
x = "global"

def outer():
    x = "outer"

    def inner():
        nonlocal x  # outer의 x 수정
        x = "inner"

    inner()
    print(f"outer에서: {x}")  # inner

outer()
print(f"global에서: {x}")    # global (변경 안됨)
```

---

## 4. 클로저 (Closure)

클로저는 자신이 정의된 환경(스코프)을 기억하는 함수입니다.

### 클로저의 조건

1. 중첩 함수가 있어야 함
2. 내부 함수가 외부 함수의 변수를 참조
3. 외부 함수가 내부 함수를 반환

### 기본 예제

```python
def make_multiplier(n):
    """n을 곱하는 함수를 반환"""
    def multiplier(x):
        return x * n  # n을 기억함
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15
print(double(10))  # 20
```

### 클로저가 기억하는 변수 확인

```python
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

c = make_counter()
print(c.__closure__)  # 클로저 셀 확인
print(c.__closure__[0].cell_contents)  # 0

c()
print(c.__closure__[0].cell_contents)  # 1
```

---

## 5. 클로저 활용 패턴

### 팩토리 함수

```python
def make_power(exp):
    """거듭제곱 함수 생성"""
    def power(base):
        return base ** exp
    return power

square = make_power(2)
cube = make_power(3)

print(square(4))  # 16
print(cube(4))    # 64
```

### 상태 유지

```python
def make_accumulator(initial=0):
    """누적 합계기 생성"""
    total = initial

    def add(value):
        nonlocal total
        total += value
        return total

    return add

acc = make_accumulator(100)
print(acc(10))   # 110
print(acc(20))   # 130
print(acc(30))   # 160
```

### 설정 저장

```python
def make_logger(prefix):
    """접두어가 붙는 로거 생성"""
    def log(message):
        print(f"[{prefix}] {message}")
    return log

error_log = make_logger("ERROR")
info_log = make_logger("INFO")

error_log("문제 발생!")   # [ERROR] 문제 발생!
info_log("시작합니다")    # [INFO] 시작합니다
```

### 함수 커스터마이징

```python
def make_formatter(template):
    """템플릿 기반 포매터 생성"""
    def format_data(**kwargs):
        return template.format(**kwargs)
    return format_data

user_format = make_formatter("이름: {name}, 나이: {age}")
product_format = make_formatter("{name} - {price}원")

print(user_format(name="Alice", age=30))
# 이름: Alice, 나이: 30

print(product_format(name="사과", price=1000))
# 사과 - 1000원
```

---

## 6. 클로저 vs 클래스

같은 기능을 클로저와 클래스로 구현할 수 있습니다.

### 클로저 버전

```python
def make_counter():
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter

c = make_counter()
print(c())  # 1
print(c())  # 2
```

### 클래스 버전

```python
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

c = Counter()
print(c())  # 1
print(c())  # 2
```

### 언제 무엇을 사용할까?

| 상황 | 권장 |
|------|------|
| 단순한 상태 유지 | 클로저 |
| 여러 메서드 필요 | 클래스 |
| 상속/확장 필요 | 클래스 |
| 함수형 스타일 | 클로저 |
| 복잡한 상태 관리 | 클래스 |

---

## 7. 클로저 주의사항

### 루프와 클로저

```python
# 잘못된 예
functions = []
for i in range(3):
    def f():
        return i
    functions.append(f)

# 모두 2를 반환! (마지막 i 값)
print([f() for f in functions])  # [2, 2, 2]
```

### 해결책 1: 기본 인자 사용

```python
functions = []
for i in range(3):
    def f(x=i):  # 기본 인자로 값 캡처
        return x
    functions.append(f)

print([f() for f in functions])  # [0, 1, 2]
```

### 해결책 2: 클로저로 감싸기

```python
def make_func(i):
    def f():
        return i
    return f

functions = [make_func(i) for i in range(3)]
print([f() for f in functions])  # [0, 1, 2]
```

### 해결책 3: lambda 사용

```python
functions = [lambda x=i: x for i in range(3)]
print([f() for f in functions])  # [0, 1, 2]
```

---

## 8. 실용적 예제

### 메모이제이션 (캐싱)

```python
def memoize(func):
    cache = {}

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

print(fibonacci(100))  # 빠르게 계산됨
```

### 디바운스 (연속 호출 제한)

```python
import time

def debounce(wait):
    """지정 시간 내 연속 호출 무시"""
    def decorator(func):
        last_call = [0]

        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_call[0] >= wait:
                last_call[0] = now
                return func(*args, **kwargs)
        return wrapper
    return decorator

@debounce(1.0)  # 1초 내 재호출 무시
def save_data():
    print("데이터 저장됨")
```

### 재시도 로직

```python
import time

def retry(max_attempts=3, delay=1):
    """실패 시 재시도"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    print(f"재시도 {attempts}/{max_attempts}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unstable_api_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("연결 실패")
    return "성공"
```

### 속성 검증

```python
def validated(validator, message):
    """값 검증 클로저"""
    def getter_setter():
        value = [None]

        def getter():
            return value[0]

        def setter(new_value):
            if not validator(new_value):
                raise ValueError(message)
            value[0] = new_value

        return getter, setter

    return getter_setter

# 사용
get_age, set_age = validated(
    lambda x: isinstance(x, int) and 0 <= x <= 150,
    "나이는 0-150 사이의 정수여야 합니다"
)()

set_age(25)
print(get_age())  # 25

# set_age(-1)  # ValueError!
```

---

## 9. 스코프 관련 함수

### locals()와 globals()

```python
x = 10

def func():
    y = 20
    print(f"지역 변수: {locals()}")   # {'y': 20}
    print(f"전역 변수 x: {globals()['x']}")  # 10

func()
```

### vars()

```python
class MyClass:
    def __init__(self):
        self.a = 1
        self.b = 2

obj = MyClass()
print(vars(obj))  # {'a': 1, 'b': 2}
```

---

## 10. 요약

| 키워드/개념 | 설명 |
|-------------|------|
| LEGB | Local → Enclosing → Global → Built-in 순서로 변수 탐색 |
| global | 전역 변수 수정 시 사용 |
| nonlocal | 감싸는 함수의 변수 수정 시 사용 |
| 클로저 | 외부 함수의 환경을 기억하는 내부 함수 |
| `__closure__` | 클로저가 참조하는 변수 확인 |

---

## 11. 연습 문제

### 연습 1: 카운터 팩토리

시작값과 증가값을 설정할 수 있는 카운터 팩토리를 작성하세요.

```python
# counter = make_counter(start=10, step=5)
# counter() → 10
# counter() → 15
# counter() → 20
```

### 연습 2: 함수 호출 기록

함수 호출 기록을 저장하는 클로저를 작성하세요.

```python
# tracked_add, get_history = track_calls(add)
# tracked_add(1, 2)
# tracked_add(3, 4)
# get_history() → [(1, 2, 3), (3, 4, 7)]
```

### 연습 3: Rate Limiter

초당 호출 횟수를 제한하는 클로저를 작성하세요.

---

## 다음 단계

[06_Metaclasses.md](./06_Metaclasses.md)에서 클래스의 클래스, 메타클래스를 배워봅시다!
