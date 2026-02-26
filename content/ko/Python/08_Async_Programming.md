# 비동기 프로그래밍 (Async Programming)

**이전**: [디스크립터](./07_Descriptors.md) | **다음**: [함수형 프로그래밍](./09_Functional_Programming.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 동기(synchronous)와 비동기(asynchronous) 실행의 차이를 설명하고, 비동기가 유리한 I/O 바운드(I/O-bound) 시나리오를 식별할 수 있습니다
2. `async def`로 코루틴(coroutine)을 정의하고, `await`를 사용해 I/O 경계에서 실행을 일시 중단할 수 있습니다
3. asyncio 이벤트 루프(event loop)의 역할을 설명하고, `asyncio.run()`으로 코루틴을 실행할 수 있습니다
4. `asyncio.create_task()`를 사용해 동시 태스크(task)를 생성하고 예약할 수 있습니다
5. 동시 실행 패턴에 `asyncio.gather()`, `asyncio.wait()`, `asyncio.as_completed()`를 적용할 수 있습니다
6. `asyncio.wait_for()`와 `asyncio.timeout()`으로 타임아웃(timeout)을 구현할 수 있습니다
7. 비동기 컨텍스트 매니저(async context manager) (`async with`)와 비동기 이터레이터/제너레이터(async iterator/generator) (`async for`)를 작성할 수 있습니다
8. `run_in_executor()`와 `asyncio.to_thread()`를 사용해 동기 블로킹 코드를 비동기 코드와 통합할 수 있습니다

---

현대 애플리케이션은 대부분의 시간을 기다리는 데 씁니다 — 네트워크 응답, 데이터베이스 쿼리, 파일 읽기, API 호출을 위해서요. 비동기 프로그래밍(Asynchronous Programming)은 블로킹 대신 대기 시간 동안 태스크를 전환함으로써, 단일 스레드가 수천 개의 동시 I/O 작업을 처리할 수 있게 합니다. Python의 `asyncio` 라이브러리와 `async`/`await` 구문을 조합하면, 스레드나 멀티프로세싱의 복잡성 없이 고성능 동시 웹 서버, 스크레이퍼, 마이크로서비스를 작성할 수 있습니다.

## 1. 동기 vs 비동기

### 동기 (Synchronous)

작업이 순차적으로 실행되며, 하나가 끝나야 다음이 시작됩니다.

```python
import time

def task(name, duration):
    print(f"{name} 시작")
    time.sleep(duration)  # 블로킹
    print(f"{name} 완료")

# 총 6초
task("작업1", 2)
task("작업2", 2)
task("작업3", 2)
```

### 비동기 (Asynchronous)

I/O 대기 중에 다른 작업을 수행할 수 있습니다.

```python
import asyncio

async def task(name, duration):
    print(f"{name} 시작")
    await asyncio.sleep(duration)  # 비블로킹
    print(f"{name} 완료")

async def main():
    # 동시 실행 - 총 2초
    await asyncio.gather(
        task("작업1", 2),
        task("작업2", 2),
        task("작업3", 2)
    )

asyncio.run(main())
```

### 비교 다이어그램

```
동기 실행:
작업1: ████████
작업2:         ████████
작업3:                  ████████
시간:  0       2        4        6초

비동기 실행 (I/O 바운드):
작업1: ████████
작업2: ████████
작업3: ████████
시간:  0       2초
```

---

## 2. async/await 기초

### 코루틴 정의

```python
async def my_coroutine():
    return "Hello, Async!"

# 코루틴 호출 → 코루틴 객체 반환
coro = my_coroutine()
print(coro)  # <coroutine object my_coroutine at ...>

# 실행하려면 await 또는 asyncio.run() 필요
result = asyncio.run(my_coroutine())
print(result)  # Hello, Async!
```

### await 키워드

`await`는 코루틴, Task, Future를 기다립니다.

```python
async def fetch_data():
    print("데이터 가져오는 중...")
    await asyncio.sleep(1)  # 이곳이 I/O 경계(I/O boundary) — 이벤트 루프(event loop)가
                            # 네트워크/디스크 응답을 기다리는 동안 다른 코루틴을 처리할 수 있도록
                            # 실행 제어권을 여기서 양보(yield)함
    return {"data": "value"}

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

### 주의: await는 async 함수 안에서만

```python
# 에러!
# result = await fetch_data()  # SyntaxError

# 올바른 사용
async def main():
    result = await fetch_data()
```

---

## 3. asyncio 이벤트 루프

### 기본 실행

```python
import asyncio

async def main():
    print("메인 코루틴")

# Python 3.7+
asyncio.run(main())

# 또는 수동으로
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
```

### 현재 루프 가져오기

```python
async def show_loop():
    loop = asyncio.get_running_loop()
    print(f"현재 루프: {loop}")

asyncio.run(show_loop())
```

---

## 4. Task 생성

### asyncio.create_task()

코루틴을 태스크로 래핑하여 동시 실행을 예약합니다.

```python
async def task(name, seconds):
    print(f"{name} 시작")
    await asyncio.sleep(seconds)
    print(f"{name} 완료")
    return name

async def main():
    # 태스크 생성 (즉시 예약됨)
    task1 = asyncio.create_task(task("A", 2))
    task2 = asyncio.create_task(task("B", 1))

    # 다른 작업 수행 가능
    print("태스크 생성됨")

    # 결과 대기
    result1 = await task1
    result2 = await task2

    print(f"결과: {result1}, {result2}")

asyncio.run(main())
```

출력:
```
태스크 생성됨
A 시작
B 시작
B 완료
A 완료
결과: A, B
```

---

## 5. 동시 실행

### asyncio.gather()

여러 코루틴을 동시에 실행하고 모든 결과를 기다립니다.

```python
async def fetch(url, delay):
    await asyncio.sleep(delay)
    return f"{url} 데이터"

async def main():
    results = await asyncio.gather(
        fetch("url1", 1),
        fetch("url2", 2),
        fetch("url3", 1),
    )
    print(results)  # ['url1 데이터', 'url2 데이터', 'url3 데이터']

asyncio.run(main())
```

### return_exceptions=True

예외가 발생해도 다른 태스크는 계속 실행됩니다.

```python
async def might_fail(n):
    if n == 2:
        raise ValueError("Error!")
    await asyncio.sleep(1)
    return n

async def main():
    results = await asyncio.gather(
        might_fail(1),
        might_fail(2),
        might_fail(3),
        return_exceptions=True  # 이 옵션 없이는 첫 번째 예외가 나머지 태스크를 모두 취소함;
                                # 이 옵션을 사용하면 예외가 값으로 반환되어 부분 실패(partial failure)를
                                # 처리할 수 있음 — 모든 결과(성공이든 실패든)가 필요한 팬아웃(fan-out) 패턴에 필수
    )
    print(results)  # [1, ValueError('Error!'), 3]

asyncio.run(main())
```

### asyncio.wait()

태스크 집합을 기다리며 더 세밀한 제어가 가능합니다.

```python
async def main():
    tasks = [
        asyncio.create_task(fetch("url1", 2)),
        asyncio.create_task(fetch("url2", 1)),
        asyncio.create_task(fetch("url3", 3)),
    ]

    # 첫 번째 완료 시 반환
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    print(f"완료: {len(done)}, 대기 중: {len(pending)}")

    # 나머지 취소
    for task in pending:
        task.cancel()
```

### asyncio.as_completed()

완료되는 순서대로 결과를 받습니다.

```python
async def main():
    tasks = [
        fetch("url1", 3),
        fetch("url2", 1),
        fetch("url3", 2),
    ]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result)  # 완료 순서: url2, url3, url1

asyncio.run(main())
```

---

## 6. 타임아웃

### asyncio.wait_for()

```python
async def slow_operation():
    await asyncio.sleep(10)
    return "완료"

async def main():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("타임아웃!")

asyncio.run(main())
```

### asyncio.timeout() (Python 3.11+)

```python
async def main():
    async with asyncio.timeout(2.0):
        await slow_operation()
```

---

## 7. 비동기 컨텍스트 매니저

`async with`를 사용합니다.

```python
class AsyncResource:
    async def __aenter__(self):
        print("리소스 획득")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("리소스 해제")
        await asyncio.sleep(0.1)

async def main():
    async with AsyncResource() as resource:
        print("리소스 사용")

asyncio.run(main())
```

### contextlib 버전

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_resource():
    print("획득")
    yield "resource"
    print("해제")

async def main():
    async with async_resource() as r:
        print(f"사용: {r}")
```

---

## 8. 비동기 이터레이터

`async for`를 사용합니다.

```python
class AsyncRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __aiter__(self):
        self.current = self.start
        return self

    async def __anext__(self):
        if self.current >= self.end:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)  # 비동기 작업
        value = self.current
        self.current += 1
        return value

async def main():
    async for num in AsyncRange(0, 5):
        print(num)

asyncio.run(main())
```

### 비동기 제너레이터

```python
async def async_range(start, end):
    for i in range(start, end):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for num in async_range(0, 5):
        print(num)
```

---

## 9. 실전 예제: HTTP 요청

### aiohttp 사용

```python
import aiohttp
import asyncio

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "https://example.com",
        "https://httpbin.org/get",
        "https://api.github.com",
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

        for url, result in zip(urls, results):
            print(f"{url}: {len(result)} bytes")

asyncio.run(main())
```

### 비동기 파일 I/O (aiofiles)

```python
import aiofiles
import asyncio

async def read_file(path):
    async with aiofiles.open(path, 'r') as f:
        return await f.read()

async def write_file(path, content):
    async with aiofiles.open(path, 'w') as f:
        await f.write(content)

async def main():
    await write_file("test.txt", "Hello, Async!")
    content = await read_file("test.txt")
    print(content)

asyncio.run(main())
```

---

## 10. 동기 코드와 혼합

### run_in_executor()

동기 함수를 비동기적으로 실행합니다.

```python
import asyncio
import time

def blocking_io():
    """동기 I/O 작업"""
    time.sleep(2)
    return "결과"

async def main():
    loop = asyncio.get_running_loop()

    # 스레드 풀에서 실행
    result = await loop.run_in_executor(
        None,  # 기본 ThreadPoolExecutor
        blocking_io
    )
    print(result)

asyncio.run(main())
```

### to_thread() (Python 3.9+)

```python
async def main():
    result = await asyncio.to_thread(blocking_io)
    print(result)
```

### 비동기 함수를 동기적으로 호출

```python
async def async_func():
    await asyncio.sleep(1)
    return "결과"

# 동기 컨텍스트에서 호출
result = asyncio.run(async_func())
print(result)
```

---

## 11. 세마포어와 락

### Semaphore (동시 실행 제한)

```python
async def limited_task(sem, n):
    async with sem:
        print(f"작업 {n} 시작")
        await asyncio.sleep(1)
        print(f"작업 {n} 완료")

async def main():
    sem = asyncio.Semaphore(3)  # 다운스트림(downstream) 속도 제한이나 커넥션 풀(connection pool) 크기에 맞게 조정;
                                # 3은 예시용 — 실제 프로덕션(production) API는 보통 스로틀링(throttling)이나
                                # 연결 거부 전까지 10~100개의 동시 요청을 허용함

    tasks = [limited_task(sem, i) for i in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

### Lock

```python
async def worker(lock, name):
    async with lock:
        print(f"{name} 획득")
        await asyncio.sleep(1)
        print(f"{name} 해제")

async def main():
    lock = asyncio.Lock()

    await asyncio.gather(
        worker(lock, "A"),
        worker(lock, "B"),
        worker(lock, "C"),
    )

asyncio.run(main())
```

---

## 12. 에러 처리

### 태스크 예외 처리

```python
async def risky_task():
    await asyncio.sleep(1)
    raise ValueError("오류!")

async def main():
    task = asyncio.create_task(risky_task())

    try:
        await task
    except ValueError as e:
        print(f"예외 발생: {e}")

asyncio.run(main())
```

### 여러 태스크 예외

```python
async def main():
    tasks = [
        asyncio.create_task(task1()),
        asyncio.create_task(task2()),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            print(f"에러: {result}")
        else:
            print(f"성공: {result}")
```

---

## 13. 요약

| 개념 | 설명 |
|------|------|
| `async def` | 코루틴 정의 |
| `await` | 코루틴 실행 대기 |
| `asyncio.run()` | 이벤트 루프 실행 |
| `asyncio.create_task()` | 태스크 생성 |
| `asyncio.gather()` | 여러 코루틴 동시 실행 |
| `asyncio.wait()` | 세밀한 태스크 관리 |
| `async with` | 비동기 컨텍스트 매니저 |
| `async for` | 비동기 이터레이션 |
| `Semaphore` | 동시 실행 제한 |

---

## 14. 연습 문제

### 연습 1: 웹 크롤러

여러 URL을 동시에 가져오는 비동기 크롤러를 작성하세요.

### 연습 2: 동시 파일 처리

여러 파일을 동시에 읽고 처리하는 함수를 작성하세요.

### 연습 3: Rate Limiter

초당 요청 수를 제한하는 비동기 함수를 작성하세요.

---

## 다음 단계

[함수형 프로그래밍 (Functional Programming)](./09_Functional_Programming.md)에서 함수형 프로그래밍을 배워봅시다!
