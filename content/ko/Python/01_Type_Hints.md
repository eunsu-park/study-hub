# 타입 힌팅 (Type Hints)

**다음**: [데코레이터](./02_Decorators.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 파이썬이 동적 타입 언어임에도 타입 힌트(Type Hints)가 코드 품질을 높이는 이유를 설명할 수 있습니다
2. 변수, 함수 매개변수, 반환값에 기본 타입 어노테이션(annotation)을 작성할 수 있습니다
3. 컬렉션(Collection) 타입에 파라미터화된 타입 인자를 적용할 수 있습니다(`list`, `dict`, `set`, `tuple`)
4. `typing` 모듈의 `Optional`, `Union`, `Any`, `Callable`, `TypeVar`의 차이를 구분할 수 있습니다
5. `Generic[T]`를 사용해 제네릭(Generic) 클래스를 구현하고, `Protocol`을 사용해 구조적 서브타이핑(structural subtyping)을 적용할 수 있습니다
6. 복잡한 타입 별칭(type alias)을 정의하고, `TypedDict`로 구조화된 딕셔너리 타입을 사용할 수 있습니다
7. `mypy`로 정적 타입 검사를 수행하고, 런타임 타입 강제와 정적 타입 강제의 차이를 이해할 수 있습니다
8. `@overload`를 사용해 오버로드된 함수 시그니처(signature)를 작성할 수 있습니다

---

타입 힌트(Type Hints)는 파이썬의 동적 유연성과 정적 타입 언어의 안전성 보장 사이의 간극을 메워줍니다. 수십 명의 기여자가 참여하는 프로덕션 코드베이스에서 타입 어노테이션은 머신이 검사할 수 있는 문서 역할을 하며, 코드가 실행되기 전에 잘못된 인자 타입, 반환값 불일치, 누락된 `None` 검사 등 특정 범주의 버그를 사전에 잡아냅니다. 현대적인 파이썬 도구(IDE, 린터, CI 파이프라인)는 타입 힌트를 활용해 자동완성, 리팩토링 지원, 대규모 자동 검증 기능을 제공합니다.

## 1. 타입 힌팅이란?

타입 힌팅은 파이썬 3.5+에서 도입된 기능으로, 변수나 함수의 타입을 명시적으로 표시합니다. 파이썬은 여전히 동적 타입 언어이지만, 타입 힌트를 통해 코드의 가독성과 안정성을 높일 수 있습니다.

```python
# 타입 힌트 없이
def greet(name):
    return f"Hello, {name}"

# 타입 힌트 사용
def greet(name: str) -> str:
    return f"Hello, {name}"
```

### 타입 힌팅의 장점

| 장점 | 설명 |
|------|------|
| 가독성 | 함수의 입력/출력 타입을 명확히 알 수 있음 |
| IDE 지원 | 자동완성, 타입 오류 경고 |
| 문서화 | 코드 자체가 문서 역할 |
| 버그 예방 | 정적 분석으로 런타임 전 오류 발견 |

---

## 2. 기본 타입 힌트

### 기본 자료형

```python
# 변수 타입 힌트
name: str = "Python"
age: int = 30
price: float = 19.99
is_active: bool = True
data: bytes = b"hello"

# 함수 타입 힌트
def add(a: int, b: int) -> int:
    return a + b

def say_hello(name: str) -> None:
    print(f"Hello, {name}")
```

### 컬렉션 타입

```python
# Python 3.9+ (내장 타입 직접 사용)
numbers: list[int] = [1, 2, 3]
names: set[str] = {"Alice", "Bob"}
scores: dict[str, int] = {"math": 95, "english": 88}
point: tuple[int, int] = (10, 20)
coordinates: tuple[float, ...] = (1.0, 2.0, 3.0)  # 가변 길이

# Python 3.8 이하 (typing 모듈 사용)
from typing import List, Set, Dict, Tuple

numbers: List[int] = [1, 2, 3]
names: Set[str] = {"Alice", "Bob"}
scores: Dict[str, int] = {"math": 95}
point: Tuple[int, int] = (10, 20)
```

---

## 3. typing 모듈

### Optional

값이 None일 수 있음을 표시합니다.

```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    """사용자를 찾거나 None 반환"""
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Python 3.10+ 대안
def find_user(user_id: int) -> str | None:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)
```

### Union

여러 타입 중 하나를 허용합니다.

```python
from typing import Union

def process(value: Union[int, str]) -> str:
    """int 또는 str을 받아 문자열로 변환"""
    return str(value)

# Python 3.10+ 대안
def process(value: int | str) -> str:
    return str(value)
```

### Any

모든 타입을 허용합니다 (타입 체크 비활성화).

```python
from typing import Any

def log(message: Any) -> None:
    print(message)

# 어떤 값이든 가능
log("hello")
log(123)
log([1, 2, 3])
```

### Callable

함수 타입을 지정합니다.

```python
from typing import Callable

# Callable[[인자타입들], 반환타입]
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

def add(x: int, y: int) -> int:
    return x + y

result = apply(add, 3, 4)  # 7
```

### TypeVar

제네릭 타입 변수를 정의합니다.

```python
from typing import TypeVar, List

T = TypeVar('T')

def first(items: List[T]) -> T:
    """리스트의 첫 번째 요소 반환"""
    return items[0]

# 타입이 자동으로 추론됨
num = first([1, 2, 3])      # int
name = first(["a", "b"])    # str
```

---

## 4. 고급 타입 힌팅

### Generic 클래스

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def is_empty(self) -> bool:
        return len(self._items) == 0

# 사용
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")
```

### TypedDict

딕셔너리의 키와 값 타입을 정의합니다.

```python
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str

# 정확한 키와 타입이 필요
user: User = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
}

# 선택적 키
class UserOptional(TypedDict, total=False):
    name: str
    age: int
    nickname: str  # 선택적
```

### Protocol (구조적 서브타이핑)

덕 타이핑을 타입 힌트로 표현합니다.

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None:
        ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

def render(shape: Drawable) -> None:
    shape.draw()

# Circle, Square는 Drawable을 상속하지 않지만
# draw() 메서드가 있으므로 사용 가능
render(Circle())  # OK
render(Square())  # OK
```

### Literal

특정 리터럴 값만 허용합니다.

```python
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"Mode: {mode}")

set_mode("read")    # OK
set_mode("write")   # OK
# set_mode("delete")  # 타입 에러!
```

### Final

상수를 표시합니다 (재할당 불가).

```python
from typing import Final

MAX_SIZE: Final = 100
PI: Final[float] = 3.14159

# MAX_SIZE = 200  # 타입 체커가 경고
```

---

## 5. 타입 별칭

복잡한 타입에 이름을 붙입니다.

```python
from typing import Dict, List, Tuple

# 타입 별칭
UserId = int
UserData = Dict[str, str]
UserList = List[Tuple[UserId, UserData]]

def get_users() -> UserList:
    return [
        (1, {"name": "Alice", "email": "alice@example.com"}),
        (2, {"name": "Bob", "email": "bob@example.com"}),
    ]

# Python 3.10+ TypeAlias
from typing import TypeAlias

Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]
```

---

## 6. 런타임 vs 정적 분석

### 타입 힌트는 런타임에 강제되지 않음

```python
def greet(name: str) -> str:
    return f"Hello, {name}"

# 런타임에는 문제없이 실행됨!
result = greet(123)  # "Hello, 123"
```

### 정적 타입 체커 (mypy)

```bash
# 설치
pip install mypy

# 타입 체크 실행
mypy your_script.py
```

```python
# example.py
def add(a: int, b: int) -> int:
    return a + b

result = add("hello", "world")  # mypy가 오류 감지
```

```
$ mypy example.py
example.py:4: error: Argument 1 to "add" has incompatible type "str"; expected "int"
```

### 런타임 타입 체크 (선택적)

```python
from typing import get_type_hints

def validate_types(func):
    """런타임 타입 검증 데코레이터"""
    hints = get_type_hints(func)

    def wrapper(*args, **kwargs):
        # 인자 타입 검증
        for (name, value), expected_type in zip(
            kwargs.items(), hints.values()
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be {expected_type}")
        return func(*args, **kwargs)
    return wrapper
```

---

## 7. 실용적 패턴

### API 응답 타입 정의

```python
from typing import TypedDict, List, Optional

class APIResponse(TypedDict):
    success: bool
    data: Optional[dict]
    error: Optional[str]

class User(TypedDict):
    id: int
    name: str
    email: str

class UsersResponse(TypedDict):
    users: List[User]
    total: int
    page: int

def fetch_users(page: int = 1) -> UsersResponse:
    # API 호출 로직
    return {
        "users": [{"id": 1, "name": "Alice", "email": "alice@example.com"}],
        "total": 100,
        "page": page
    }
```

### 설정 클래스

```python
from typing import Final, ClassVar

class Config:
    # 클래스 변수 (모든 인스턴스 공유)
    DEBUG: ClassVar[bool] = False
    VERSION: ClassVar[str] = "1.0.0"

    # 상수
    MAX_CONNECTIONS: Final = 100
    TIMEOUT: Final[int] = 30
```

### 오버로드

같은 함수의 여러 시그니처를 정의합니다.

```python
from typing import overload, Union

@overload
def process(value: int) -> int: ...

@overload
def process(value: str) -> str: ...

def process(value: Union[int, str]) -> Union[int, str]:
    if isinstance(value, int):
        return value * 2
    return value.upper()

# 타입 체커가 올바른 반환 타입을 추론
num: int = process(5)       # int
text: str = process("hi")   # str
```

---

## 8. 자주 쓰는 타입 정리

| 타입 | 설명 | 예시 |
|------|------|------|
| `int`, `str`, `float`, `bool` | 기본 타입 | `x: int = 1` |
| `list[T]` | 리스트 | `nums: list[int]` |
| `dict[K, V]` | 딕셔너리 | `data: dict[str, int]` |
| `set[T]` | 집합 | `ids: set[int]` |
| `tuple[T, ...]` | 튜플 | `point: tuple[int, int]` |
| `Optional[T]` | T 또는 None | `name: Optional[str]` |
| `Union[T1, T2]` | T1 또는 T2 | `value: Union[int, str]` |
| `Callable[[Args], R]` | 함수 타입 | `fn: Callable[[int], str]` |
| `Any` | 모든 타입 | `data: Any` |
| `TypeVar` | 제네릭 변수 | `T = TypeVar('T')` |
| `Protocol` | 구조적 타이핑 | `class Sized(Protocol)` |

---

## 9. 연습 문제

### 연습 1: 함수 타입 힌트

다음 함수에 적절한 타입 힌트를 추가하세요.

```python
def calculate_average(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
```

### 연습 2: 제네릭 함수

리스트에서 조건을 만족하는 첫 번째 요소를 찾는 제네릭 함수를 작성하세요.

```python
# find_first([1, 2, 3, 4], lambda x: x > 2) -> 3
# find_first(["a", "bb", "ccc"], lambda s: len(s) > 1) -> "bb"
```

### 연습 3: TypedDict

사용자 프로필을 나타내는 TypedDict를 정의하세요.
- 필수: id (int), username (str), email (str)
- 선택: bio (str), avatar_url (str)

---

## 다음 단계

[데코레이터 (Decorators)](./02_Decorators.md)에서 함수와 클래스를 확장하는 데코레이터를 배워봅시다!
