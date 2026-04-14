# 탐색 기초

**이전**: [정렬 기초](./12_Sorting_Fundamentals.md) | **다음**: [올바른 자료구조 선택](./14_Choosing_the_Right_Data_Structure.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 선형 탐색을 구현하고 분석할 수 있다
2. 이진 탐색과 일반적인 변형을 구현할 수 있다
3. 이진 탐색 구현에서 오프바이원 오류를 피할 수 있다
4. 단순 조회를 넘어서 이진 탐색을 적용할 수 있다 (하한/상한)
5. O(1) 평균 조회를 위한 해시 기반 탐색을 사용할 수 있다
6. 탐색과 정렬을 결합하여 효율적인 쿼리 처리를 할 수 있다
7. 데이터와 쿼리 패턴에 따라 적절한 탐색 전략을 선택할 수 있다

---

**탐색**은 컬렉션에서 특정 요소를 찾거나 그 부재를 확인하는 과정입니다. 탐색 알고리즘의 선택은 자료구조, 데이터의 정렬 여부, 수행할 쿼리의 수에 따라 달라집니다.

## 선형 탐색

대상을 찾거나 끝에 도달할 때까지 모든 요소를 검사합니다:

```python
def linear_search(arr, target):
    """선형 탐색 -- O(n) 시간, O(1) 공간."""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```

## 이진 탐색

**정렬된** 데이터가 필요합니다. 탐색 공간을 반복적으로 반으로 줄입니다:

```python
def binary_search(arr, target):
    """이진 탐색 -- O(log n) 시간, O(1) 공간.
    
    arr은 오름차순으로 정렬되어야 합니다.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2  # 오버플로 방지
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

## 이진 탐색 변형

### 하한 (bisect_left)

정렬 순서를 유지하면서 target을 삽입할 수 있는 가장 왼쪽 위치를 찾습니다:

```python
def lower_bound(arr, target):
    """target 이상인 첫 위치 찾기 -- O(log n).
    
    >>> lower_bound([1, 2, 4, 4, 4, 6, 8], 4)
    2
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left
```

### 상한 (bisect_right)

```python
def upper_bound(arr, target):
    """target 초과인 첫 위치 찾기 -- O(log n).
    
    >>> upper_bound([1, 2, 4, 4, 4, 6, 8], 4)
    5
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left
```

### 출현 횟수 세기

```python
def count_occurrences(arr, target):
    """이진 탐색을 사용한 출현 횟수 -- O(log n)."""
    return upper_bound(arr, target) - lower_bound(arr, target)
```

## 답에 대한 이진 탐색

문제가 단조 속성을 가질 때 **답**에 대해 이진 탐색할 수 있습니다:

```python
def sqrt_integer(n):
    """이진 탐색으로 floor(sqrt(n)) 찾기."""
    if n < 2:
        return n
    left, right = 1, n // 2
    while left <= right:
        mid = left + (right - left) // 2
        if mid * mid == n:
            return mid
        elif mid * mid < n:
            left = mid + 1
        else:
            right = mid - 1
    return right
```

## 해시 기반 탐색

```python
# 평균 O(1) 조회를 위한 Python set
lookup_set = set(large_list)
if target in lookup_set:  # O(1) 평균
    print("Found!")

# 키-값 조회를 위한 Python dict
lookup_dict = {item.id: item for item in items}
result = lookup_dict.get(target_id)  # O(1) 평균
```

### 해시 vs 이진 탐색

| 기준 | 해시 기반 | 이진 탐색 |
|------|----------|----------|
| 전처리 | O(n) 해시 테이블 구축 | O(n log n) 정렬 |
| 단일 쿼리 | O(1) 평균 | O(log n) |
| 범위 쿼리 | 지원 안 됨 | 효율적 |
| 공간 | O(n) | O(1) 추가 (정렬된 경우) |
| 정렬된 결과 | 아니오 | 예 |

## Python의 `bisect` 모듈

```python
import bisect

sorted_list = [1, 3, 5, 7, 9, 11, 13, 15]

bisect.bisect_left(sorted_list, 7)   # 3
bisect.bisect_right(sorted_list, 7)  # 4
bisect.insort(sorted_list, 8)        # 정렬 유지하며 삽입
```

## 탐색 비교

| 알고리즘 | 시간 | 공간 | 정렬 필요 | 적합한 용도 |
|---------|------|------|---------|-----------|
| 선형 | O(n) | O(1) | 아니오 | 작은/비정렬 데이터 |
| 이진 | O(log n) | O(1) | **예** | 정렬 배열, 범위 쿼리 |
| 해시 | O(1)* | O(n) | 아니오 | 빈번한 조회, 정확한 매칭 |

*평균

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 선형 탐색 | O(n), 어떤 데이터에서든 동작 |
| 이진 탐색 | O(log n), 정렬된 데이터 필요 |
| 하한/상한 | 삽입 지점 찾기, 출현 횟수 세기 |
| 답에 대한 이진 탐색 | 문제가 단조적일 때 적용 |
| 해시 기반 탐색 | 평균 O(1), O(n) 공간 |
| `bisect` 모듈 | Python의 내장 이진 탐색 유틸리티 |
| 트레이드오프 | 정확 매칭에는 해시; 범위 쿼리에는 이진 |

---

**다음**: [올바른 자료구조 선택](./14_Choosing_the_Right_Data_Structure.md) -- 모든 것을 실용적인 의사결정 프레임워크로 종합합니다.
