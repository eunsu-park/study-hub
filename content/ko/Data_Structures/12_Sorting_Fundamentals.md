# 정렬 기초

**이전**: [문자열 자료구조](./11_Strings_as_Data_Structures.md) | **다음**: [탐색 기초](./13_Searching_Fundamentals.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 버블 정렬, 선택 정렬, 삽입 정렬을 구현할 수 있다
2. 분할 정복으로 병합 정렬과 퀵 정렬을 구현할 수 있다
3. 각 정렬 알고리즘의 시간/공간 복잡도를 분석할 수 있다
4. 정렬의 안정성 개념을 설명할 수 있다
5. 비교 기반 정렬의 O(n log n) 하한을 이해할 수 있다
6. Python의 내장 `sorted()`와 `list.sort()`를 효과적으로 사용할 수 있다
7. 데이터 특성에 따라 적절한 정렬 알고리즘을 선택할 수 있다

---

**정렬**은 요소를 정의된 순서 (일반적으로 오름차순 또는 내림차순)로 배열하는 과정입니다. 정렬된 데이터는 효율적인 탐색, 병합, 분석을 가능하게 하므로, 컴퓨터 과학에서 가장 많이 연구된 문제 중 하나입니다.

## 버블 정렬

인접한 요소가 잘못된 순서이면 반복적으로 교환합니다:

```python
def bubble_sort(arr):
    """버블 정렬 -- O(n^2) 시간, O(1) 공간, 안정."""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
```

## 선택 정렬

최솟값을 찾아 앞으로 교환합니다:

```python
def selection_sort(arr):
    """선택 정렬 -- O(n^2) 시간, O(1) 공간, 불안정."""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

## 삽입 정렬

정렬된 부분을 한 요소씩 구축하며, 각각을 올바른 위치에 삽입합니다:

```python
def insertion_sort(arr):
    """삽입 정렬 -- O(n^2) 시간, O(1) 공간, 안정, 적응적."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

**삽입 정렬이 중요한 이유**: 거의 정렬된 데이터에서 O(n)이며, Timsort와 같은 하이브리드 알고리즘의 기본 사례로 사용됩니다.

## 병합 정렬

배열을 반으로 나누고, 각 반을 정렬한 후, 정렬된 반을 병합합니다:

```python
def merge_sort(arr):
    """병합 정렬 -- O(n log n) 시간, O(n) 공간, 안정."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    """두 정렬된 배열을 하나의 정렬된 배열로 병합."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

## 퀵 정렬

피벗을 선택하고, 요소를 피벗 주위로 분할한 후, 각 파티션을 재귀적으로 정렬합니다:

```python
def quick_sort(arr):
    """퀵 정렬 -- 평균 O(n log n), 최악 O(n^2), O(log n) 공간."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

## 비교 요약

| 알고리즘 | 최선 | 평균 | 최악 | 공간 | 안정 |
|---------|------|------|------|------|------|
| 버블 | O(n) | O(n^2) | O(n^2) | O(1) | 예 |
| 선택 | O(n^2) | O(n^2) | O(n^2) | O(1) | 아니오 |
| 삽입 | **O(n)** | O(n^2) | O(n^2) | O(1) | 예 |
| 병합 | O(n log n) | O(n log n) | O(n log n) | O(n) | 예 |
| 퀵 | O(n log n) | O(n log n) | O(n^2) | O(log n) | 아니오 |
| Timsort | **O(n)** | O(n log n) | O(n log n) | O(n) | 예 |

## O(n log n) 하한

어떤 비교 기반 정렬 알고리즘이든 최악의 경우 최소 O(n log n) 비교를 해야 합니다. 비비교 정렬 (카운팅 정렬, 기수 정렬)은 요소의 속성을 활용하여 이 하한을 깰 수 있습니다.

## Python의 `sorted()`와 Timsort

Python은 병합 정렬과 삽입 정렬의 하이브리드인 **Timsort**를 사용합니다:

```python
sorted([3, 1, 4, 1, 5])         # [1, 1, 3, 4, 5]
sorted([3, 1, 4], reverse=True) # [4, 3, 1]

nums = [3, 1, 4]
nums.sort()  # 제자리 정렬

# 사용자 정의 키 함수
words = ["banana", "apple", "cherry"]
sorted(words, key=len)  # ['apple', 'banana', 'cherry']

# 객체 정렬
students = [("Alice", 88), ("Bob", 95), ("Charlie", 88)]
sorted(students, key=lambda s: (-s[1], s[0]))
```

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 버블 정렬 | 단순하지만 느림; 교육용으로 좋음 |
| 선택 정렬 | 최소 교환; 불안정 |
| 삽입 정렬 | 작은/거의 정렬된 데이터에 좋음 |
| 병합 정렬 | 보장된 O(n log n); 안정; O(n) 공간 필요 |
| 퀵 정렬 | 실제로 빠름; 최악 O(n^2); 불안정 |
| 안정성 | 동일 요소의 상대적 순서 보존 |
| 하한 | 비교 정렬은 O(n log n)을 이길 수 없음 |
| Timsort | Python 내장; 병합+삽입 하이브리드 |

---

**다음**: [탐색 기초](./13_Searching_Fundamentals.md) -- 효율적인 탐색 기법을 배웁니다.
