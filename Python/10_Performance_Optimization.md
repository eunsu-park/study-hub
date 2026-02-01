# 성능 최적화 (Performance Optimization)

## 1. 성능 측정 기초

최적화 전에 반드시 측정하세요. "추측하지 말고 측정하라."

### timeit 모듈

```python
import timeit

# 문자열로 코드 측정
time = timeit.timeit(
    'sum(range(1000))',
    number=10000
)
print(f"실행 시간: {time:.4f}초")

# 함수 측정
def my_sum():
    return sum(range(1000))

time = timeit.timeit(my_sum, number=10000)
print(f"실행 시간: {time:.4f}초")
```

### IPython/Jupyter에서

```python
# 한 줄 측정
%timeit sum(range(1000))

# 셀 전체 측정
%%timeit
total = 0
for i in range(1000):
    total += i
```

### 시간 측정 데코레이터

```python
import time
from functools import wraps

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}초")
        return result
    return wrapper

@timing
def slow_function():
    time.sleep(1)
    return "완료"

slow_function()  # slow_function: 1.0012초
```

---

## 2. 프로파일링

### cProfile

```python
import cProfile
import pstats

def expensive_function():
    total = 0
    for i in range(10000):
        total += sum(range(100))
    return total

# 프로파일링
profiler = cProfile.Profile()
profiler.enable()

expensive_function()

profiler.disable()

# 결과 출력
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 상위 10개
```

### 명령줄에서 실행

```bash
# 기본 프로파일링
python -m cProfile my_script.py

# 결과 정렬
python -m cProfile -s cumulative my_script.py

# 파일로 저장
python -m cProfile -o output.prof my_script.py
```

### 프로파일 결과 분석

```python
import pstats

# 저장된 프로파일 로드
stats = pstats.Stats('output.prof')

# 정렬 기준
# 'calls': 호출 횟수
# 'time': 내부 시간
# 'cumulative': 누적 시간
stats.sort_stats('cumulative')
stats.print_stats(20)

# 특정 함수만 보기
stats.print_stats('my_function')
```

### line_profiler (라인별 분석)

```bash
pip install line_profiler
```

```python
# my_script.py
@profile  # 데코레이터 추가
def slow_function():
    result = []
    for i in range(10000):
        result.append(i ** 2)
    return result

slow_function()
```

```bash
kernprof -l -v my_script.py
```

---

## 3. 메모리 프로파일링

### memory_profiler

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def memory_hungry():
    big_list = [i for i in range(1000000)]
    del big_list
    small_list = [i for i in range(1000)]
    return small_list

memory_hungry()
```

출력:
```
Line #    Mem usage    Increment  Occurrences   Line Contents
     3     38.5 MiB     38.5 MiB           1   @profile
     4                                         def memory_hungry():
     5     76.8 MiB     38.3 MiB           1       big_list = [i for i in range(1000000)]
     6     38.5 MiB    -38.3 MiB           1       del big_list
     7     38.5 MiB      0.0 MiB           1       small_list = [i for i in range(1000)]
     8     38.5 MiB      0.0 MiB           1       return small_list
```

### sys.getsizeof()

```python
import sys

# 객체 크기 확인
print(sys.getsizeof([]))         # 56 (빈 리스트)
print(sys.getsizeof([1, 2, 3]))  # 88
print(sys.getsizeof({}))         # 64 (빈 딕셔너리)
print(sys.getsizeof("hello"))    # 54

# 중첩 객체는 포함하지 않음
nested = [[1, 2, 3] for _ in range(10)]
print(sys.getsizeof(nested))  # 외부 리스트만
```

### tracemalloc (메모리 추적)

```python
import tracemalloc

tracemalloc.start()

# 메모리 사용 코드
big_list = [i ** 2 for i in range(100000)]

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 메모리 사용 ]")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

---

## 4. 데이터 구조 선택

### 리스트 vs 튜플

```python
import sys

# 튜플이 더 작고 빠름
lst = [1, 2, 3, 4, 5]
tup = (1, 2, 3, 4, 5)

print(sys.getsizeof(lst))  # 104
print(sys.getsizeof(tup))  # 80

# 생성 속도
%timeit [1, 2, 3, 4, 5]  # 느림
%timeit (1, 2, 3, 4, 5)  # 빠름 (상수)
```

### 리스트 vs 집합 (멤버십 테스트)

```python
import timeit

data_list = list(range(10000))
data_set = set(range(10000))

# 리스트: O(n)
print(timeit.timeit('9999 in data_list', globals=globals(), number=10000))
# 약 0.8초

# 집합: O(1)
print(timeit.timeit('9999 in data_set', globals=globals(), number=10000))
# 약 0.001초
```

### 딕셔너리 vs 리스트 (검색)

```python
# 이름으로 검색할 때
users_list = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    # ... 10000개
]

users_dict = {
    "Alice": {"age": 30},
    "Bob": {"age": 25},
    # ... 10000개
}

# 리스트: O(n)
def find_in_list(name):
    for user in users_list:
        if user["name"] == name:
            return user

# 딕셔너리: O(1)
def find_in_dict(name):
    return users_dict.get(name)
```

### 시간 복잡도 정리

| 연산 | list | dict | set |
|------|------|------|-----|
| 인덱스 접근 | O(1) | - | - |
| 검색 (in) | O(n) | O(1) | O(1) |
| 삽입 (끝) | O(1) | O(1) | O(1) |
| 삽입 (중간) | O(n) | - | - |
| 삭제 | O(n) | O(1) | O(1) |

---

## 5. 문자열 최적화

### 문자열 연결

```python
# 나쁜 예: O(n²)
result = ""
for i in range(10000):
    result += str(i)

# 좋은 예: O(n)
result = "".join(str(i) for i in range(10000))

# 더 좋은 예 (리스트 사용)
parts = []
for i in range(10000):
    parts.append(str(i))
result = "".join(parts)
```

### f-string vs format vs %

```python
name = "Alice"
age = 30

# f-string (가장 빠름, Python 3.6+)
s = f"Name: {name}, Age: {age}"

# format (중간)
s = "Name: {}, Age: {}".format(name, age)

# % 포맷 (느림)
s = "Name: %s, Age: %d" % (name, age)
```

### 문자열 인터닝

```python
# 파이썬은 짧은 문자열을 자동으로 인터닝
a = "hello"
b = "hello"
print(a is b)  # True

# 긴 문자열
a = "hello world" * 100
b = "hello world" * 100
print(a is b)  # False

# 수동 인터닝
import sys
a = sys.intern("hello world" * 100)
b = sys.intern("hello world" * 100)
print(a is b)  # True
```

---

## 6. 루프 최적화

### 지역 변수 사용

```python
import math

# 느림: 매번 전역 조회
def slow():
    result = []
    for i in range(10000):
        result.append(math.sqrt(i))
    return result

# 빠름: 지역 변수로 캐싱
def fast():
    result = []
    sqrt = math.sqrt  # 지역 변수로
    append = result.append  # 지역 변수로
    for i in range(10000):
        append(sqrt(i))
    return result
```

### 리스트 컴프리헨션

```python
# for 루프
result = []
for i in range(10000):
    result.append(i ** 2)

# 리스트 컴프리헨션 (더 빠름)
result = [i ** 2 for i in range(10000)]

# map (비슷한 속도)
result = list(map(lambda x: x ** 2, range(10000)))
```

### 불필요한 작업 제거

```python
# 나쁜 예: 매번 len() 호출
for i in range(len(my_list)):
    if i < len(my_list) - 1:
        pass

# 좋은 예: 미리 계산
length = len(my_list)
for i in range(length):
    if i < length - 1:
        pass

# 더 좋은 예: enumerate 사용
for i, item in enumerate(my_list):
    pass
```

---

## 7. 함수 최적화

### 메모이제이션

```python
from functools import lru_cache

# 메모이제이션 없이 (느림)
def fib_slow(n):
    if n < 2:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)

# 메모이제이션 사용 (빠름)
@lru_cache(maxsize=None)
def fib_fast(n):
    if n < 2:
        return n
    return fib_fast(n - 1) + fib_fast(n - 2)

# fib_slow(35) → 수 초
# fib_fast(35) → 즉시
```

### 제너레이터 사용

```python
# 메모리 많이 사용
def get_squares_list(n):
    return [i ** 2 for i in range(n)]

# 메모리 효율적
def get_squares_gen(n):
    for i in range(n):
        yield i ** 2

# 사용
for square in get_squares_gen(1000000):
    if square > 100:
        break
```

### 내장 함수 활용

```python
# 직접 구현 (느림)
def my_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total

# 내장 함수 (빠름, C로 구현)
total = sum(numbers)

# 다른 예시
max_val = max(numbers)      # 최댓값
min_val = min(numbers)      # 최솟값
sorted_nums = sorted(numbers)  # 정렬
any_positive = any(n > 0 for n in numbers)  # 조건 검사
```

---

## 8. __slots__ 최적화

클래스 인스턴스의 메모리 사용량을 줄입니다.

```python
import sys

# 일반 클래스
class PointRegular:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# __slots__ 사용
class PointSlots:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# 메모리 비교
p1 = PointRegular(1, 2)
p2 = PointSlots(1, 2)

print(sys.getsizeof(p1.__dict__))  # 104 (딕셔너리 크기)
# PointSlots는 __dict__ 없음

# 많은 인스턴스에서 차이 큼
regular_points = [PointRegular(i, i) for i in range(100000)]
slots_points = [PointSlots(i, i) for i in range(100000)]
```

### __slots__ 주의사항

```python
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)

# 동적 속성 추가 불가
# p.z = 3  # AttributeError

# 상속 시 주의
class Point3D(Point):
    __slots__ = ['z']  # 추가 슬롯만 정의

    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z
```

---

## 9. 병렬 처리

### multiprocessing (CPU 바운드)

```python
from multiprocessing import Pool
import time

def cpu_bound_task(n):
    """CPU 집약적 작업"""
    return sum(i * i for i in range(n))

if __name__ == '__main__':
    numbers = [10000000] * 4

    # 순차 실행
    start = time.time()
    results = [cpu_bound_task(n) for n in numbers]
    print(f"순차: {time.time() - start:.2f}초")

    # 병렬 실행
    start = time.time()
    with Pool(4) as pool:
        results = pool.map(cpu_bound_task, numbers)
    print(f"병렬: {time.time() - start:.2f}초")
```

### concurrent.futures

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

def task(n):
    return sum(range(n))

# ProcessPoolExecutor (CPU 바운드)
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(task, [10000000] * 4))

# ThreadPoolExecutor (I/O 바운드)
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, n) for n in [10000000] * 4]
    results = [f.result() for f in futures]
```

### asyncio (I/O 바운드)

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = ["http://example.com"] * 10

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    return results

# asyncio.run(main())
```

---

## 10. NumPy 활용

과학 계산에서 성능 향상을 제공합니다.

### 기본 비교

```python
import numpy as np

# 순수 파이썬
def python_sum_squares(n):
    return sum(i ** 2 for i in range(n))

# NumPy
def numpy_sum_squares(n):
    arr = np.arange(n)
    return np.sum(arr ** 2)

# NumPy가 훨씬 빠름
%timeit python_sum_squares(1000000)  # ~100ms
%timeit numpy_sum_squares(1000000)   # ~2ms
```

### 벡터화 연산

```python
import numpy as np

# 파이썬 루프
def normalize_python(data):
    result = []
    mean = sum(data) / len(data)
    std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    for x in data:
        result.append((x - mean) / std)
    return result

# NumPy 벡터화
def normalize_numpy(data):
    arr = np.array(data)
    return (arr - arr.mean()) / arr.std()
```

---

## 11. Cython 소개

파이썬을 C로 컴파일하여 성능을 높입니다.

### 간단한 예제

```python
# fib.pyx
def fib_py(int n):
    cdef int a = 0
    cdef int b = 1
    cdef int i
    for i in range(n):
        a, b = b, a + b
    return a
```

### 빌드

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fib.pyx")
)
```

```bash
python setup.py build_ext --inplace
```

---

## 12. 최적화 체크리스트

### 일반 원칙

1. **측정 먼저**: 추측하지 말고 프로파일링
2. **병목 찾기**: 전체의 20%가 80% 시간 소모
3. **알고리즘 개선**: 자료구조와 알고리즘이 가장 중요
4. **가독성 유지**: 과도한 최적화는 금물

### 빠른 승리 (Quick Wins)

| 항목 | 방법 |
|------|------|
| 멤버십 테스트 | list → set |
| 문자열 연결 | + → join() |
| 루프 | for → 컴프리헨션 |
| 함수 호출 | 지역 변수 캐싱 |
| 반복 계산 | @lru_cache |
| 많은 객체 | __slots__ |

### 상황별 선택

| 상황 | 해결책 |
|------|--------|
| CPU 바운드 | multiprocessing, NumPy, Cython |
| I/O 바운드 | asyncio, threading |
| 메모리 부족 | 제너레이터, __slots__ |
| 반복 계산 | 메모이제이션 |

---

## 13. 요약

| 도구 | 용도 |
|------|------|
| timeit | 실행 시간 측정 |
| cProfile | 함수 단위 프로파일링 |
| line_profiler | 라인 단위 프로파일링 |
| memory_profiler | 메모리 프로파일링 |
| tracemalloc | 메모리 추적 |

| 최적화 기법 | 효과 |
|------------|------|
| 적절한 자료구조 | O(n) → O(1) |
| 리스트 컴프리헨션 | 루프보다 빠름 |
| 지역 변수 캐싱 | 조회 비용 감소 |
| 메모이제이션 | 중복 계산 제거 |
| __slots__ | 메모리 절약 |
| 병렬 처리 | CPU 활용 극대화 |

---

## 14. 연습 문제

### 연습 1: 프로파일링

느린 코드를 프로파일링하고 병목을 찾아 개선하세요.

```python
def slow_function(n):
    result = ""
    for i in range(n):
        if i in [j for j in range(i)]:
            result += str(i)
    return result
```

### 연습 2: 메모리 최적화

대용량 CSV 파일을 메모리 효율적으로 처리하는 함수를 작성하세요.

### 연습 3: 병렬 처리

CPU 바운드 작업을 병렬화하여 성능을 개선하세요.

---

## 마무리

이 가이드에서 파이썬 고급 문법을 학습했습니다. 실제 프로젝트에 적용하면서 경험을 쌓아가세요!

### 학습 완료 체크리스트

- [ ] 타입 힌팅으로 코드 품질 향상
- [ ] 데코레이터로 코드 재사용
- [ ] 컨텍스트 매니저로 리소스 관리
- [ ] 제너레이터로 메모리 효율화
- [ ] 클로저로 상태 캡슐화
- [ ] 메타클래스로 클래스 커스터마이징
- [ ] 디스크립터로 속성 제어
- [ ] asyncio로 비동기 처리
- [ ] 함수형 패턴 활용
- [ ] 성능 측정 및 최적화
