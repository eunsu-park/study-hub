# 타입 체킹

**이전**: [린터와 포매터](./08_Linters_and_Formatters.md) | **다음**: [프로파일링 기초](./10_Profiling_Basics.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 함수, 변수, 클래스 속성에 타입 힌트 추가하기
2. 내장 타입(`list`, `dict`, `tuple`)과 `typing` 모듈 타입(`Optional`, `Union`) 사용하기
3. `mypy`를 실행하여 타입 정확성을 정적으로 검사하기
4. mypy 에러 메시지를 해석하고 타입 관련 버그 수정하기
5. 기존 코드베이스에 점진적 타이핑을 적용하여 단계적으로 타입 추가하기
6. `TypeAlias`, `Literal`, `TypedDict` 등 고급 타입 구조 사용하기
7. 타입 체킹의 한계와 언제 버그를 잡는지 이해하기
8. `pyproject.toml`로 mypy 설정하기

---

Python은 동적 타입 언어입니다: 변수가 어떤 시점에도 어떤 타입이든 가질 수 있습니다. 이 유연성은 강력하지만 위험합니다 -- 타입 관련 버그는 Python 코드에서 가장 흔한 에러 중 하나입니다. 타입 힌트로 예상 타입을 주석으로 달면, `mypy` 같은 도구가 코드가 실행되기 **전에** 그 주석을 검사합니다. 이렇게 하면 런타임이 아닌 개발 시점에 전체 범주의 버그를 잡을 수 있습니다.

> **핵심 통찰:** 타입 힌트는 Python이 코드를 실행하는 방식을 바꾸지 않습니다. 사람과 도구를 위한 주석입니다. Python은 런타임에 이를 무시하지만, `mypy`는 이를 읽고 타입 불일치를 경고합니다.

---

## 1. 기본 타입 힌트

### 1.1 함수 주석

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def process(data: list) -> None:
    for item in data:
        print(item)
```

### 1.2 변수 주석

```python
name: str = "Alice"
age: int = 30
scores: list[float] = [95.5, 87.0, 92.3]
settings: dict[str, int] = {"width": 800, "height": 600}
```

### 1.3 컬렉션 타입 (Python 3.9+)

```python
names: list[str] = ["Alice", "Bob"]
scores: dict[str, float] = {"Alice": 95.0}
coordinates: tuple[float, float] = (3.14, 2.71)
unique_ids: set[int] = {1, 2, 3}
matrix: list[list[int]] = [[1, 2], [3, 4]]
```

---

## 2. 특수 타입

### 2.1 Optional (None 가능 값)

```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    """사용자명 또는 None 반환."""
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Python 3.10+ 문법
def find_user(user_id: int) -> str | None:
    ...
```

### 2.2 Union (여러 가능 타입)

```python
# Python 3.10+ 문법
def normalize(value: int | float | str) -> float:
    if isinstance(value, str):
        return float(value)
    return float(value)
```

### 2.3 Literal (특정 값)

```python
from typing import Literal

def set_direction(direction: Literal["north", "south", "east", "west"]) -> None:
    print(f"Moving {direction}")

set_direction("north")   # OK
set_direction("up")      # mypy 에러: "up"은 유효한 값이 아님
```

---

## 3. mypy 실행

### 3.1 설치 및 기본 사용

```bash
pip install mypy
mypy my_script.py
```

### 3.2 흔한 mypy 에러

| 에러 | 의미 |
|------|------|
| `Incompatible types in assignment` | 변수에 잘못된 타입 할당 |
| `Incompatible return value type` | 함수가 잘못된 타입 반환 |
| `Argument has incompatible type` | 함수에 잘못된 타입 전달 |
| `Item "None" of "Optional[X]" has no attribute` | None 검사 없이 Optional 사용 |
| `Missing return statement` | 반환 타입을 선언했으나 항상 반환하지 않음 |

### 3.3 Optional 올바르게 처리

```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice"}
    return users.get(user_id)

# 나쁜 예: mypy가 잡음!
name = find_user(1)
print(name.upper())  # 에러: "str | None"의 Item "None"에 "upper" 속성 없음

# 좋은 예: 먼저 None 확인
name = find_user(1)
if name is not None:
    print(name.upper())  # OK: mypy가 여기서 name이 str임을 앎
```

---

## 4. 점진적 타이핑

한꺼번에 모든 곳에 타입을 추가할 필요 없습니다. 점진적으로 추가하세요:

```
1. 새 코드부터 → 항상 타입 힌트 추가
2. 공개 함수 시그니처에 타입 추가
3. 자주 변경되는 파일에 타입 추가
4. CI에서 mypy를 사용하여 회귀 방지
5. 점차 엄격도 증가
```

### mypy 설정

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
check_untyped_defs = true

# 새 모듈에 대한 엄격 모드
[[tool.mypy.overrides]]
module = "myapp.new_module"
disallow_untyped_defs = true

# 스텁이 없는 서드파티 라이브러리 무시
[[tool.mypy.overrides]]
module = "some_library.*"
ignore_missing_imports = true
```

---

## 5. 타입 체킹이 잡는 실제 버그

### 5.1 None 안전성

```python
def get_config(key: str) -> str | None:
    config = {"host": "localhost", "port": "8080"}
    return config.get(key)

# 타입 체킹 없이는 이 버그가 런타임까지 발견되지 않음:
port = get_config("port")
url = f"http://localhost:{port + 1}"  # port가 None이면 TypeError!

# mypy와 함께:
# error: Unsupported operand types for + ("str | None" and "int")
```

### 5.2 잘못된 반환 타입

```python
def parse_age(text: str) -> int:
    if text.isdigit():
        return int(text)
    # 반환 문 누락! mypy가 잡음: Missing return statement

# 수정:
def parse_age(text: str) -> int:
    if text.isdigit():
        return int(text)
    raise ValueError(f"Invalid age: {text!r}")
```

---

## 6. 클래스와 TypedDict

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
```

```python
from typing import TypedDict

class UserProfile(TypedDict):
    name: str
    age: int
    email: str

def process_user(user: UserProfile) -> str:
    return f"{user['name']} ({user['age']})"
```

---

## 요약

- 타입 힌트는 런타임 동작을 바꾸지 않고 예상 타입을 주석으로 표시
- `mypy`가 타입 정확성을 정적으로 검사하여 코드 실행 전에 버그 포착
- None이 될 수 있는 값에는 `Optional[X]` (또는 `X | None`) 사용
- 타입 체킹이 None 안전성 버그, 잘못된 인자 타입, 누락된 반환을 포착
- 점진적 타이핑으로 기존 코드베이스에 단계적으로 타입 추가 가능
- `pyproject.toml`에서 프로젝트 전체 설정으로 mypy 구성
- 타입 체킹은 린터 및 포매터와 결합할 때 가장 효과적

---

## 연습문제

1. 타입이 없는 함수 집합에 타입 힌트 추가하기
2. 타입이 있는 코드베이스에서 mypy 에러 수정하기
3. `Optional`을 사용하여 None을 반환하는 함수를 올바르게 처리하기
4. 구조화된 설정 딕셔너리를 위한 `TypedDict` 만들기

**이전**: [린터와 포매터](./08_Linters_and_Formatters.md) | **다음**: [프로파일링 기초](./10_Profiling_Basics.md)
