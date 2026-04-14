# 프로파일링 기초

**이전**: [타입 체킹](./09_Type_Checking.md) | **다음**: [디버깅을 위한 버전 관리](./11_Version_Control_for_Debugging.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프로파일링과 벤치마킹의 차이 설명하기
2. `timeit`으로 코드 조각의 실행 시간을 정확히 측정하기
3. `cProfile`로 프로그램에서 가장 느린 함수 찾기
4. 프로파일링 출력(ncalls, tottime, cumtime) 읽고 해석하기
5. `time.perf_counter()`로 코드 구간의 수동 타이밍
6. `tracemalloc`과 `memory_profiler`로 메모리 사용량 프로파일링
7. 프로파일링 후 최적화하여 성급한 최적화 피하기
8. 80/20 규칙 적용: 80% 느림을 유발하는 20% 코드 찾기

---

"작동하게 만들고, 올바르게 만들고, 빠르게 만들라 -- 이 순서대로." 코드를 최적화하기 전에 **측정**해야 합니다. 프로파일링은 프로그램이 시간과 메모리를 정확히 어디에 쓰는지 알려주며, 추측을 데이터로 대체합니다. 프로파일링 없이는 0.001초 동안 실행되는 코드를 최적화하느라 몇 시간을 낭비할 수 있는 반면, 실제 병목은 다른 곳에 숨어 있습니다.

> **성급한 최적화:** Donald Knuth는 "성급한 최적화는 만악의 근원"이라 했습니다. 최적화 전에 항상 먼저 프로파일링하여 실제 병목을 식별하세요.

---

## 1. 코드 타이밍

### 1.1 `time.perf_counter()` -- 수동 타이밍

```python
import time

start = time.perf_counter()
result = expensive_function()
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f}초 소요")
```

### 1.2 재사용 가능한 타이머

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(label="Block"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{label}] {elapsed:.4f}s")

with timer("정렬"):
    sorted_data = sorted(large_list)
```

### 1.3 `timeit` -- 정확한 마이크로벤치마크

```python
import timeit

t = timeit.timeit('sum(range(1000))', number=10000)
print(f"총: {t:.4f}s, 호출당: {t/10000:.6f}s")
```

---

## 2. cProfile: 느린 함수 찾기

### 2.1 기본 사용

```bash
python -m cProfile my_script.py
```

### 2.2 cProfile 출력 읽기

```
         1000003 function calls in 2.543 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.543    2.543 script.py:1(main)
  1000000    2.100    0.000    2.100    0.000 script.py:12(transform)
        1    0.431    0.431    0.431    0.431 script.py:8(generate_data)
```

### 열 의미

```
ncalls   함수가 호출된 횟수
tottime  함수 안에서의 총 시간 (하위 호출 제외)
percall  tottime / ncalls
cumtime  함수 안에서의 총 시간 (하위 호출 포함)
```

**핵심**: `tottime`을 보면 시간이 실제로 어디서 쓰이는지 알 수 있습니다. `cumtime`을 보면 어떤 상위 함수가 전반적으로 느린지 알 수 있습니다.

### 2.3 정렬 및 필터링

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 상위 20개 함수
```

---

## 3. 메모리 프로파일링

### 3.1 `tracemalloc` (내장)

```python
import tracemalloc

tracemalloc.start()

data = [i ** 2 for i in range(100000)]
big_dict = {str(i): i ** 2 for i in range(100000)}

snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')

print("메모리 사용 상위 10:")
for stat in stats[:10]:
    print(stat)
```

### 3.2 메모리 증가 추적

```python
import tracemalloc
tracemalloc.start()

snapshot1 = tracemalloc.take_snapshot()
process_data()
snapshot2 = tracemalloc.take_snapshot()

stats = snapshot2.compare_to(snapshot1, 'lineno')
print("메모리 변화:")
for stat in stats[:10]:
    print(stat)
```

### 3.3 `memory_profiler` (서드파티, 줄별)

```bash
pip install memory-profiler
```

```python
from memory_profiler import profile

@profile
def memory_hungry():
    a = [i for i in range(100000)]
    b = {i: i**2 for i in range(100000)}
    del a
    return b
```

---

## 4. 흔한 성능 함정

### 4.1 문자열 연결 vs join

```python
# 느림: 문자열 복사 때문에 O(n^2)
result = ""
for item in items:
    result += str(item)

# 빠름: O(n)
result = "".join(str(item) for item in items)
```

### 4.2 리스트 vs 집합에서 검색

```python
# 느림: 검색마다 O(n)
items = list(range(100000))
if target in items: ...

# 빠름: 검색마다 O(1)
items = set(range(100000))
if target in items: ...
```

---

## 5. 최적화의 80/20 규칙

```
1. 먼저 프로파일 -- 병목이 어디인지 추측하지 마세요
2. tottime 기준 상위 1-3개 함수 식별
3. 그 함수들만 최적화
4. 개선을 확인하기 위해 다시 프로파일
5. 성능이 "충분히 좋으면" 중단
```

---

## 6. 빠른 참조

| 도구 | 측정 대상 | 사용 시점 |
|------|----------|----------|
| `time.perf_counter()` | 벽시계 시간 | 빠른 수동 타이밍 |
| `timeit` | 실행 시간 (평균) | 마이크로벤치마크, 접근법 비교 |
| `cProfile` | 함수 호출 횟수와 시간 | 느린 함수 찾기 |
| `tracemalloc` | 메모리 할당 | 메모리 집약적 코드 찾기 |
| `memory_profiler` | 줄별 메모리 사용량 | 상세 메모리 분석 |
| `snakeviz` | 시각적 프로필 뷰어 | 복잡한 프로필 이해 |

---

## 요약

- 최적화 전에 항상 프로파일 -- 병목이 어디인지 추측하지 말 것
- `timeit`이 두 접근법을 비교하는 올바른 도구
- `cProfile`이 가장 많은 시간을 소비하는 함수를 식별
- `tracemalloc`과 `memory_profiler`가 메모리 병목을 식별
- 상위 1-3개 병목 함수에 최적화를 집중
- 최적화 후 다시 프로파일하여 개선 확인
- 성급한 최적화는 시간 낭비; 데이터 기반 최적화가 결과를 냄

---

## 연습문제

1. `timeit`으로 문자열 연결 vs `join()` 비교하기
2. `cProfile`로 느린 함수를 프로파일링하고 병목 식별하기
3. `tracemalloc`으로 스크립트에서 가장 메모리를 많이 쓰는 줄 찾기
4. 프로파일링 결과를 바탕으로 함수 최적화하고 속도 향상 확인하기

**이전**: [타입 체킹](./09_Type_Checking.md) | **다음**: [디버깅을 위한 버전 관리](./11_Version_Control_for_Debugging.md)
