# 컨텍스트 매니저 (Context Managers)

## 1. 컨텍스트 매니저란?

컨텍스트 매니저는 `with`문과 함께 사용되어 리소스의 설정(setup)과 정리(cleanup)를 자동으로 처리합니다.

```python
# 컨텍스트 매니저 없이
file = open("example.txt", "w")
try:
    file.write("Hello")
finally:
    file.close()

# 컨텍스트 매니저 사용
with open("example.txt", "w") as file:
    file.write("Hello")
# 자동으로 file.close() 호출됨
```

### 동작 흐름

```
with 표현식 as 변수:
    │
    ▼
┌─────────────────────┐
│  __enter__() 호출   │ ← 리소스 설정
│  반환값 → 변수      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   with 블록 실행     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  __exit__() 호출    │ ← 리소스 정리 (예외 발생해도 실행)
└─────────────────────┘
```

---

## 2. 클래스로 구현하기

`__enter__`와 `__exit__` 메서드를 구현합니다.

### 기본 구조

```python
class MyContextManager:
    def __enter__(self):
        print("리소스 설정")
        return self  # as 절에 바인딩될 값

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("리소스 정리")
        return False  # 예외를 다시 발생시킴

with MyContextManager() as cm:
    print("작업 수행")
```

출력:
```
리소스 설정
작업 수행
리소스 정리
```

### 파일 관리자 예제

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False

with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
```

### 데이터베이스 연결 예제

```python
class DatabaseConnection:
    def __init__(self, host, database):
        self.host = host
        self.database = database
        self.connection = None

    def __enter__(self):
        print(f"연결: {self.host}/{self.database}")
        self.connection = {"host": self.host, "db": self.database}
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("연결 종료")
        self.connection = None
        return False

with DatabaseConnection("localhost", "mydb") as conn:
    print(f"사용 중: {conn}")
```

---

## 3. __exit__의 예외 처리

`__exit__` 메서드는 예외 정보를 받아 처리할 수 있습니다.

### 매개변수

| 매개변수 | 설명 |
|----------|------|
| exc_type | 예외 클래스 (예: `ValueError`) |
| exc_val | 예외 인스턴스 |
| exc_tb | 트레이스백 객체 |

예외가 없으면 모두 `None`입니다.

### 예외 처리 예제

```python
class ErrorHandler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"예외 발생: {exc_type.__name__}: {exc_val}")
            # True 반환 시 예외를 억제 (전파하지 않음)
            return True
        return False

with ErrorHandler():
    raise ValueError("테스트 에러")

print("이 줄이 실행됨 (예외가 억제됨)")
```

출력:
```
예외 발생: ValueError: 테스트 에러
이 줄이 실행됨 (예외가 억제됨)
```

### 특정 예외만 처리

```python
class IgnoreValueError:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ValueError만 억제
        if exc_type is ValueError:
            print(f"ValueError 무시됨: {exc_val}")
            return True
        return False  # 다른 예외는 전파

with IgnoreValueError():
    raise ValueError("이 에러는 무시됨")

# with IgnoreValueError():
#     raise TypeError("이 에러는 전파됨")  # 프로그램 중단
```

---

## 4. contextlib 모듈

### @contextmanager 데코레이터

제너레이터 함수로 컨텍스트 매니저를 간단히 만들 수 있습니다.

```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("설정")       # __enter__ 부분
    yield "리소스"       # as 절에 바인딩될 값
    print("정리")       # __exit__ 부분

with my_context() as value:
    print(f"사용: {value}")
```

출력:
```
설정
사용: 리소스
정리
```

### 예외 처리 포함

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("리소스 획득")
    try:
        yield "resource"
    except Exception as e:
        print(f"예외 처리: {e}")
        raise  # 예외 재발생 (억제하려면 제거)
    finally:
        print("리소스 해제")

with managed_resource() as r:
    print(f"사용: {r}")
    # raise ValueError("테스트")
```

### 파일 관리자 (contextmanager 버전)

```python
from contextlib import contextmanager

@contextmanager
def open_file(path, mode):
    f = open(path, mode)
    try:
        yield f
    finally:
        f.close()

with open_file("test.txt", "w") as f:
    f.write("Hello!")
```

---

## 5. contextlib 유틸리티

### suppress - 예외 억제

```python
from contextlib import suppress

# 기존 방식
try:
    import json
    data = json.loads("invalid")
except json.JSONDecodeError:
    pass

# suppress 사용
with suppress(json.JSONDecodeError):
    data = json.loads("invalid")
# 예외가 발생해도 무시됨
```

### redirect_stdout - 출력 리다이렉트

```python
from contextlib import redirect_stdout
import io

# 출력을 문자열로 캡처
f = io.StringIO()
with redirect_stdout(f):
    print("이 출력은 캡처됨")

output = f.getvalue()
print(f"캡처된 내용: {output}")
```

### closing - close() 자동 호출

```python
from contextlib import closing
from urllib.request import urlopen

# urlopen은 컨텍스트 매니저가 아님 (Python 2 호환용)
with closing(urlopen("https://example.com")) as page:
    content = page.read()
```

### ExitStack - 동적 컨텍스트 관리

여러 컨텍스트 매니저를 동적으로 관리합니다.

```python
from contextlib import ExitStack

files = ["file1.txt", "file2.txt", "file3.txt"]

with ExitStack() as stack:
    file_objects = [
        stack.enter_context(open(f, "w"))
        for f in files
    ]
    # 모든 파일에 쓰기
    for f in file_objects:
        f.write("Hello\n")
# 모든 파일이 자동으로 닫힘
```

---

## 6. 중첩 컨텍스트 매니저

### 여러 with문

```python
with open("input.txt") as infile:
    with open("output.txt", "w") as outfile:
        outfile.write(infile.read())
```

### 한 줄로 작성

```python
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read())
```

### 괄호로 여러 줄

```python
# Python 3.10+
with (
    open("file1.txt") as f1,
    open("file2.txt") as f2,
    open("file3.txt") as f3,
):
    # 모든 파일 사용
    pass
```

---

## 7. 실용적 패턴

### 타이머

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name="작업"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}초")

with timer("데이터 처리"):
    # 시간이 걸리는 작업
    time.sleep(0.5)
```

### 임시 디렉토리 변경

```python
from contextlib import contextmanager
import os

@contextmanager
def change_dir(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)

with change_dir("/tmp"):
    print(f"현재: {os.getcwd()}")
# 자동으로 원래 디렉토리로 복원
```

### 임시 환경 변수

```python
from contextlib import contextmanager
import os

@contextmanager
def temp_env(**kwargs):
    old_env = {k: os.environ.get(k) for k in kwargs}
    os.environ.update(kwargs)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

with temp_env(DEBUG="true", API_KEY="test"):
    print(os.environ["DEBUG"])  # true
# 원래 환경 복원
```

### 락 (Lock)

```python
from contextlib import contextmanager
import threading

@contextmanager
def locked(lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

# 실제로는 Lock 자체가 컨텍스트 매니저
lock = threading.Lock()
with lock:
    # 임계 영역
    pass
```

### 트랜잭션 패턴

```python
from contextlib import contextmanager

class Transaction:
    def __init__(self):
        self.operations = []

    def add(self, op):
        self.operations.append(op)

    def commit(self):
        for op in self.operations:
            print(f"실행: {op}")
        self.operations.clear()

    def rollback(self):
        print("롤백!")
        self.operations.clear()

@contextmanager
def transaction(tx):
    try:
        yield tx
        tx.commit()
    except Exception:
        tx.rollback()
        raise

tx = Transaction()
with transaction(tx):
    tx.add("INSERT INTO users VALUES (1, 'Alice')")
    tx.add("UPDATE accounts SET balance = 100")
    # raise ValueError("오류!")  # 주석 해제 시 롤백
```

---

## 8. 비동기 컨텍스트 매니저

`async with`를 사용하려면 `__aenter__`와 `__aexit__`를 구현합니다.

```python
class AsyncResource:
    async def __aenter__(self):
        print("비동기 설정")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("비동기 정리")
        return False

async def main():
    async with AsyncResource() as r:
        print("비동기 작업")

import asyncio
asyncio.run(main())
```

### contextlib의 asynccontextmanager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_timer(name):
    import time
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.4f}초")

async def main():
    async with async_timer("비동기 작업"):
        await asyncio.sleep(0.5)
```

---

## 9. 요약

| 방법 | 사용 시점 |
|------|----------|
| 클래스 (`__enter__`, `__exit__`) | 상태 관리가 필요할 때 |
| `@contextmanager` | 간단한 설정/정리 로직 |
| `suppress` | 특정 예외 무시 |
| `redirect_stdout` | 출력 리다이렉트 |
| `ExitStack` | 동적 컨텍스트 관리 |
| `closing` | close() 메서드 자동 호출 |

---

## 10. 연습 문제

### 연습 1: 타임아웃 컨텍스트 매니저

지정된 시간이 지나면 TimeoutError를 발생시키는 컨텍스트 매니저를 작성하세요.

### 연습 2: 로그 레벨 변경

임시로 로깅 레벨을 변경했다가 복원하는 컨텍스트 매니저를 작성하세요.

### 연습 3: 테스트 더블

테스트용으로 함수를 임시로 대체하는 컨텍스트 매니저를 작성하세요.

---

## 다음 단계

[04_Iterators_and_Generators.md](./04_Iterators_and_Generators.md)에서 이터레이터와 yield를 배워봅시다!
