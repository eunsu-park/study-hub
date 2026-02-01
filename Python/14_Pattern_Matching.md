# 14. 패턴 매칭 (Pattern Matching)

## 학습 목표
- Python 3.10+ match/case 문법 이해
- 다양한 패턴 유형 마스터
- 구조적 패턴 매칭 활용
- 가드와 OR 패턴 사용
- 실무 활용 패턴 학습

## 목차
1. [패턴 매칭 기초](#1-패턴-매칭-기초)
2. [리터럴 패턴](#2-리터럴-패턴)
3. [구조적 패턴](#3-구조적-패턴)
4. [클래스 패턴](#4-클래스-패턴)
5. [가드와 OR 패턴](#5-가드와-or-패턴)
6. [실전 활용](#6-실전-활용)
7. [연습 문제](#7-연습-문제)

---

## 1. 패턴 매칭 기초

### 1.1 match/case 소개

```python
# Python 3.10+ 필요

def http_status(status: int) -> str:
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:  # 와일드카드 (default)
            return f"Unknown status: {status}"


print(http_status(200))  # OK
print(http_status(404))  # Not Found
print(http_status(999))  # Unknown status: 999
```

### 1.2 기존 if-elif와 비교

```python
# if-elif 방식
def get_day_type_if(day: str) -> str:
    if day in ("Saturday", "Sunday"):
        return "Weekend"
    elif day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"):
        return "Weekday"
    else:
        return "Invalid day"


# match/case 방식
def get_day_type_match(day: str) -> str:
    match day:
        case "Saturday" | "Sunday":  # OR 패턴
            return "Weekend"
        case "Monday" | "Tuesday" | "Wednesday" | "Thursday" | "Friday":
            return "Weekday"
        case _:
            return "Invalid day"
```

### 1.3 변수 캡처

```python
def describe_point(point):
    match point:
        case (0, 0):
            return "Origin"
        case (x, 0):  # x에 값 캡처
            return f"On X-axis at x={x}"
        case (0, y):  # y에 값 캡처
            return f"On Y-axis at y={y}"
        case (x, y):  # 둘 다 캡처
            return f"Point at ({x}, {y})"
        case _:
            return "Not a point"


print(describe_point((0, 0)))    # Origin
print(describe_point((5, 0)))    # On X-axis at x=5
print(describe_point((0, 3)))    # On Y-axis at y=3
print(describe_point((2, 4)))    # Point at (2, 4)
```

---

## 2. 리터럴 패턴

### 2.1 다양한 리터럴

```python
def check_value(value):
    match value:
        # 숫자 리터럴
        case 0:
            return "Zero"
        case 1 | 2 | 3:  # OR 패턴
            return "Small positive"

        # 문자열 리터럴
        case "":
            return "Empty string"
        case "hello":
            return "Greeting"

        # Boolean
        case True:
            return "True value"
        case False:
            return "False value"

        # None
        case None:
            return "None value"

        case _:
            return "Other"


print(check_value(0))       # Zero
print(check_value(2))       # Small positive
print(check_value("hello")) # Greeting
print(check_value(None))    # None value
```

### 2.2 상수 비교

```python
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def describe_color(color: Color) -> str:
    match color:
        case Color.RED:
            return "Stop"
        case Color.GREEN:
            return "Go"
        case Color.BLUE:
            return "Cool"
        case _:
            return "Unknown color"


print(describe_color(Color.RED))    # Stop
print(describe_color(Color.GREEN))  # Go
```

### 2.3 상수 패턴 (점 표기법)

```python
# 상수를 패턴에서 사용하려면 점(.)이 포함되어야 함
class HttpStatus:
    OK = 200
    NOT_FOUND = 404
    ERROR = 500


def check_status(status: int) -> str:
    match status:
        case HttpStatus.OK:
            return "Success"
        case HttpStatus.NOT_FOUND:
            return "Not found"
        case HttpStatus.ERROR:
            return "Server error"
        case _:
            return "Unknown"


# 단순 변수는 상수가 아닌 캡처 변수로 해석됨
# OK = 200
# case OK:  # 이건 200과 비교가 아닌 변수 캡처!
```

---

## 3. 구조적 패턴

### 3.1 시퀀스 패턴

```python
def analyze_sequence(seq):
    match seq:
        case []:
            return "Empty sequence"
        case [single]:
            return f"Single element: {single}"
        case [first, second]:
            return f"Two elements: {first}, {second}"
        case [first, *middle, last]:  # 언패킹
            return f"First: {first}, Middle: {middle}, Last: {last}"


print(analyze_sequence([]))           # Empty sequence
print(analyze_sequence([1]))          # Single element: 1
print(analyze_sequence([1, 2]))       # Two elements: 1, 2
print(analyze_sequence([1, 2, 3, 4])) # First: 1, Middle: [2, 3], Last: 4
```

### 3.2 딕셔너리 패턴

```python
def process_event(event: dict):
    match event:
        case {"type": "click", "x": x, "y": y}:
            return f"Click at ({x}, {y})"

        case {"type": "keypress", "key": key}:
            return f"Key pressed: {key}"

        case {"type": "scroll", "direction": direction, **rest}:
            return f"Scroll {direction}, extra: {rest}"

        case {"type": event_type}:
            return f"Unknown event type: {event_type}"

        case _:
            return "Invalid event"


print(process_event({"type": "click", "x": 100, "y": 200}))
# Click at (100, 200)

print(process_event({"type": "keypress", "key": "Enter"}))
# Key pressed: Enter

print(process_event({"type": "scroll", "direction": "down", "speed": 10}))
# Scroll down, extra: {'speed': 10}
```

### 3.3 중첩 구조

```python
def process_response(response: dict):
    match response:
        case {"status": "success", "data": {"users": [first_user, *_]}}:
            return f"First user: {first_user}"

        case {"status": "success", "data": {"count": count}}:
            return f"Count: {count}"

        case {"status": "error", "error": {"code": code, "message": msg}}:
            return f"Error {code}: {msg}"

        case {"status": status}:
            return f"Status: {status}"


response1 = {"status": "success", "data": {"users": ["Alice", "Bob"]}}
print(process_response(response1))  # First user: Alice

response2 = {"status": "error", "error": {"code": 404, "message": "Not found"}}
print(process_response(response2))  # Error 404: Not found
```

---

## 4. 클래스 패턴

### 4.1 데이터클래스와 패턴 매칭

```python
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Circle:
    center: Point
    radius: float


@dataclass
class Rectangle:
    top_left: Point
    width: float
    height: float


def describe_shape(shape):
    match shape:
        case Point(x=0, y=0):
            return "Origin point"
        case Point(x=x, y=y):
            return f"Point at ({x}, {y})"

        case Circle(center=Point(x=0, y=0), radius=r):
            return f"Circle at origin with radius {r}"
        case Circle(center=c, radius=r):
            return f"Circle at ({c.x}, {c.y}) with radius {r}"

        case Rectangle(width=w, height=h) if w == h:
            return f"Square with side {w}"
        case Rectangle(top_left=tl, width=w, height=h):
            return f"Rectangle at ({tl.x}, {tl.y}), {w}x{h}"

        case _:
            return "Unknown shape"


print(describe_shape(Point(0, 0)))        # Origin point
print(describe_shape(Point(3, 4)))        # Point at (3, 4)
print(describe_shape(Circle(Point(0, 0), 5)))  # Circle at origin with radius 5
print(describe_shape(Rectangle(Point(1, 2), 10, 10)))  # Square with side 10
```

### 4.2 위치 인자 패턴 (__match_args__)

```python
from dataclasses import dataclass


@dataclass
class Vector:
    x: float
    y: float
    z: float = 0.0
    # dataclass는 자동으로 __match_args__ 생성


def describe_vector(v):
    match v:
        case Vector(0, 0, 0):  # 위치 인자로 매칭
            return "Zero vector"
        case Vector(x, 0, 0):
            return f"X-axis vector: {x}"
        case Vector(0, y, 0):
            return f"Y-axis vector: {y}"
        case Vector(0, 0, z):
            return f"Z-axis vector: {z}"
        case Vector(x, y, z):
            return f"Vector ({x}, {y}, {z})"


print(describe_vector(Vector(0, 0, 0)))  # Zero vector
print(describe_vector(Vector(5, 0, 0)))  # X-axis vector: 5
print(describe_vector(Vector(1, 2, 3)))  # Vector (1, 2, 3)
```

### 4.3 일반 클래스

```python
class Animal:
    __match_args__ = ("name", "species")

    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species


class Dog(Animal):
    __match_args__ = ("name", "breed")

    def __init__(self, name: str, breed: str):
        super().__init__(name, "dog")
        self.breed = breed


def greet_animal(animal):
    match animal:
        case Dog(name, breed="Labrador"):
            return f"Good dog, {name}! Labs are the best!"
        case Dog(name, breed):
            return f"Hello, {name} the {breed}!"
        case Animal(name, species):
            return f"Hello, {name} the {species}!"


print(greet_animal(Dog("Buddy", "Labrador")))  # Good dog, Buddy!
print(greet_animal(Dog("Max", "Beagle")))      # Hello, Max the Beagle!
print(greet_animal(Animal("Whiskers", "cat"))) # Hello, Whiskers the cat!
```

---

## 5. 가드와 OR 패턴

### 5.1 가드 (if 조건)

```python
def categorize_number(n: int) -> str:
    match n:
        case n if n < 0:
            return "Negative"
        case 0:
            return "Zero"
        case n if n % 2 == 0:
            return "Positive even"
        case n if n % 2 == 1:
            return "Positive odd"
        case _:
            return "Unknown"


print(categorize_number(-5))  # Negative
print(categorize_number(0))   # Zero
print(categorize_number(4))   # Positive even
print(categorize_number(7))   # Positive odd
```

### 5.2 복합 가드

```python
from dataclasses import dataclass


@dataclass
class User:
    name: str
    age: int
    role: str


def check_access(user: User, resource: str) -> str:
    match (user, resource):
        case (User(role="admin"), _):
            return "Full access"

        case (User(age=age), "adult_content") if age < 18:
            return "Access denied: Age restriction"

        case (User(role="user"), "admin_panel"):
            return "Access denied: Admin only"

        case (User(name=name), resource):
            return f"{name} can access {resource}"


admin = User("Alice", 30, "admin")
teen = User("Bob", 16, "user")
user = User("Charlie", 25, "user")

print(check_access(admin, "admin_panel"))     # Full access
print(check_access(teen, "adult_content"))    # Access denied: Age restriction
print(check_access(user, "admin_panel"))      # Access denied: Admin only
print(check_access(user, "profile"))          # Charlie can access profile
```

### 5.3 OR 패턴 (|)

```python
def classify_char(char: str) -> str:
    match char:
        case 'a' | 'e' | 'i' | 'o' | 'u':
            return "Lowercase vowel"
        case 'A' | 'E' | 'I' | 'O' | 'U':
            return "Uppercase vowel"
        case ' ' | '\t' | '\n':
            return "Whitespace"
        case '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9':
            return "Digit"
        case _:
            return "Other"


print(classify_char('a'))  # Lowercase vowel
print(classify_char('E'))  # Uppercase vowel
print(classify_char(' '))  # Whitespace
print(classify_char('5'))  # Digit
print(classify_char('x'))  # Other
```

### 5.4 AS 패턴 (별칭)

```python
def process_command(command):
    match command:
        case ["quit" | "exit" | "q" as cmd]:
            return f"Exit command: {cmd}"

        case ["load", filename] | ["open", filename] as full_cmd:
            return f"Loading file: {filename} (command: {full_cmd})"

        case [("get" | "fetch") as action, *args]:
            return f"Action: {action}, Args: {args}"

        case _:
            return "Unknown command"


print(process_command(["quit"]))           # Exit command: quit
print(process_command(["exit"]))           # Exit command: exit
print(process_command(["load", "data.txt"])) # Loading file: data.txt
print(process_command(["get", "user", "123"])) # Action: get, Args: ['user', '123']
```

---

## 6. 실전 활용

### 6.1 JSON API 응답 처리

```python
from typing import Any


def handle_api_response(response: dict[str, Any]) -> str:
    match response:
        case {"status": 200, "data": {"items": [first, *rest]}}:
            return f"Success: First item = {first}, {len(rest)} more items"

        case {"status": 200, "data": data}:
            return f"Success: {data}"

        case {"status": 400, "error": {"field": field, "message": msg}}:
            return f"Validation error in '{field}': {msg}"

        case {"status": 401}:
            return "Unauthorized: Please login"

        case {"status": 403, "error": {"reason": reason}}:
            return f"Forbidden: {reason}"

        case {"status": 404}:
            return "Not found"

        case {"status": status} if 500 <= status < 600:
            return f"Server error: {status}"

        case {"status": status}:
            return f"Unknown status: {status}"

        case _:
            return "Invalid response format"


# 테스트
responses = [
    {"status": 200, "data": {"items": ["a", "b", "c"]}},
    {"status": 400, "error": {"field": "email", "message": "Invalid format"}},
    {"status": 401},
    {"status": 500},
]

for r in responses:
    print(handle_api_response(r))
```

### 6.2 상태 머신

```python
from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


@dataclass
class Event:
    pass


@dataclass
class Start(Event):
    pass


@dataclass
class Pause(Event):
    pass


@dataclass
class Resume(Event):
    pass


@dataclass
class Stop(Event):
    pass


def transition(state: State, event: Event) -> State:
    match (state, event):
        case (State.IDLE, Start()):
            return State.RUNNING

        case (State.RUNNING, Pause()):
            return State.PAUSED

        case (State.RUNNING, Stop()):
            return State.STOPPED

        case (State.PAUSED, Resume()):
            return State.RUNNING

        case (State.PAUSED, Stop()):
            return State.STOPPED

        case (state, event):
            print(f"Invalid transition: {state} + {event}")
            return state


# 상태 전이 테스트
state = State.IDLE
print(f"Initial: {state}")

state = transition(state, Start())
print(f"After Start: {state}")

state = transition(state, Pause())
print(f"After Pause: {state}")

state = transition(state, Resume())
print(f"After Resume: {state}")

state = transition(state, Stop())
print(f"After Stop: {state}")
```

### 6.3 AST 처리 (인터프리터 패턴)

```python
from dataclasses import dataclass
from typing import Union


@dataclass
class Num:
    value: float


@dataclass
class BinOp:
    left: "Expr"
    op: str
    right: "Expr"


@dataclass
class UnaryOp:
    op: str
    operand: "Expr"


Expr = Union[Num, BinOp, UnaryOp]


def evaluate(expr: Expr) -> float:
    match expr:
        case Num(value):
            return value

        case BinOp(left, "+", right):
            return evaluate(left) + evaluate(right)

        case BinOp(left, "-", right):
            return evaluate(left) - evaluate(right)

        case BinOp(left, "*", right):
            return evaluate(left) * evaluate(right)

        case BinOp(left, "/", right):
            right_val = evaluate(right)
            if right_val == 0:
                raise ValueError("Division by zero")
            return evaluate(left) / right_val

        case UnaryOp("-", operand):
            return -evaluate(operand)

        case _:
            raise ValueError(f"Unknown expression: {expr}")


# (3 + 4) * -2
expr = BinOp(
    BinOp(Num(3), "+", Num(4)),
    "*",
    UnaryOp("-", Num(2))
)
print(evaluate(expr))  # -14.0
```

### 6.4 CLI 명령 파서

```python
import sys


def parse_command(args: list[str]) -> dict:
    match args:
        case []:
            return {"command": "help"}

        case ["--version" | "-v"]:
            return {"command": "version"}

        case ["--help" | "-h"]:
            return {"command": "help"}

        case ["init", name]:
            return {"command": "init", "name": name}

        case ["init", name, "--template", template]:
            return {"command": "init", "name": name, "template": template}

        case ["run", *files] if files:
            return {"command": "run", "files": files}

        case ["config", "set", key, value]:
            return {"command": "config_set", "key": key, "value": value}

        case ["config", "get", key]:
            return {"command": "config_get", "key": key}

        case [unknown, *_]:
            return {"command": "error", "message": f"Unknown command: {unknown}"}


# 테스트
commands = [
    [],
    ["--version"],
    ["init", "myproject"],
    ["init", "myproject", "--template", "fastapi"],
    ["run", "app.py", "tests.py"],
    ["config", "set", "debug", "true"],
    ["unknown", "arg"],
]

for cmd in commands:
    print(f"{cmd} -> {parse_command(cmd)}")
```

---

## 7. 연습 문제

### 연습 1: 도형 면적 계산기
다양한 도형의 면적을 계산하는 함수를 작성하세요.

```python
# 예시 답안
from dataclasses import dataclass
import math


@dataclass
class Circle:
    radius: float


@dataclass
class Rectangle:
    width: float
    height: float


@dataclass
class Triangle:
    base: float
    height: float


Shape = Circle | Rectangle | Triangle


def calculate_area(shape: Shape) -> float:
    match shape:
        case Circle(radius=r):
            return math.pi * r ** 2
        case Rectangle(width=w, height=h):
            return w * h
        case Triangle(base=b, height=h):
            return 0.5 * b * h


print(calculate_area(Circle(5)))          # ~78.54
print(calculate_area(Rectangle(4, 5)))    # 20
print(calculate_area(Triangle(6, 4)))     # 12
```

### 연습 2: HTTP 요청 라우터
간단한 HTTP 요청 라우터를 구현하세요.

```python
# 예시 답안
def route_request(method: str, path: str) -> str:
    match (method, path.split("/")):
        case ("GET", ["", ""]):
            return "Home page"

        case ("GET", ["", "users"]):
            return "List users"

        case ("GET", ["", "users", user_id]):
            return f"Get user {user_id}"

        case ("POST", ["", "users"]):
            return "Create user"

        case ("PUT", ["", "users", user_id]):
            return f"Update user {user_id}"

        case ("DELETE", ["", "users", user_id]):
            return f"Delete user {user_id}"

        case (method, _):
            return f"404 Not Found: {method} {path}"


print(route_request("GET", "/"))              # Home page
print(route_request("GET", "/users"))         # List users
print(route_request("GET", "/users/123"))     # Get user 123
print(route_request("POST", "/users"))        # Create user
print(route_request("DELETE", "/users/456"))  # Delete user 456
```

### 연습 3: 재귀 트리 순회
트리 구조를 패턴 매칭으로 순회하세요.

```python
# 예시 답안
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    value: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def sum_tree(node: Optional[TreeNode]) -> int:
    match node:
        case None:
            return 0
        case TreeNode(value=v, left=l, right=r):
            return v + sum_tree(l) + sum_tree(r)


tree = TreeNode(
    1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3, None, TreeNode(6))
)
print(sum_tree(tree))  # 21 (1+2+3+4+5+6)
```

---

## 다음 단계
- [11. 테스트 및 품질 관리](./11_Testing_and_Quality.md)
- [13. 데이터클래스](./13_Dataclasses.md)

## 참고 자료
- [PEP 634 - Structural Pattern Matching](https://peps.python.org/pep-0634/)
- [PEP 635 - Motivation and Rationale](https://peps.python.org/pep-0635/)
- [PEP 636 - Tutorial](https://peps.python.org/pep-0636/)
- [Python 3.10 What's New](https://docs.python.org/3/whatsnew/3.10.html)
