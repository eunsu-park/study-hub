# 탐색 알고리즘 (Search Algorithms)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 선형 탐색(linear search), 이진 탐색(binary search), 해시 탐색(hash search)을 시간 복잡도, 전제 조건, 적합한 사용 사례 측면에서 비교할 수 있다
2. 반복문과 재귀를 이용한 이진 탐색을 구현하고, 정렬된 배열에서 O(log n) 시간이 달성되는 이유를 설명할 수 있다
3. 이진 탐색 변형(lower bound, upper bound)을 적용하여 목표 값의 첫 번째·마지막 등장 위치를 찾는 문제를 해결할 수 있다
4. 최적화 문제를 정답에 대한 이진 탐색(parametric search, 파라메트릭 서치) 문제로 변환하여 솔루션을 설계할 수 있다
5. 해시 기반 조회가 평균 O(1) 시간을 달성하는 이유와 정렬 배열 이진 탐색 대비 한계를 설명할 수 있다
6. 데이터 구조(정렬 여부, 범위 제약, 중복 존재)에 따라 가장 효율적인 탐색 전략을 선택할 수 있다

---

## 개요

탐색은 데이터에서 원하는 값을 찾는 기본적인 연산입니다. 이 레슨에서는 선형 탐색, 이진 탐색, 파라메트릭 서치, 해시 탐색 등 다양한 탐색 기법을 학습합니다.

---

## 목차

1. [탐색 알고리즘 비교](#1-탐색-알고리즘-비교)
2. [선형 탐색](#2-선형-탐색)
3. [이진 탐색](#3-이진-탐색)
4. [이진 탐색 변형](#4-이진-탐색-변형)
5. [파라메트릭 서치](#5-파라메트릭-서치)
6. [해시 탐색](#6-해시-탐색)
7. [연습 문제](#7-연습-문제)

---

## 1. 탐색 알고리즘 비교

```
┌──────────────┬─────────────┬─────────────┬───────────────────┐
│  알고리즘    │ 시간 복잡도 │ 조건        │ 특징              │
├──────────────┼─────────────┼─────────────┼───────────────────┤
│ 선형 탐색    │ O(n)        │ 없음        │ 단순, 범용        │
│ 이진 탐색    │ O(log n)    │ 정렬됨      │ 빠름, 분할정복    │
│ 해시 탐색    │ O(1) 평균   │ 해시 테이블 │ 가장 빠름         │
│ 보간 탐색    │ O(log log n)│ 균등 분포   │ 특수 상황         │
└──────────────┴─────────────┴─────────────┴───────────────────┘
```

---

## 2. 선형 탐색 (Linear Search)

### 원리

```
처음부터 끝까지 순차적으로 검사

배열: [5, 3, 8, 4, 2], 찾는 값: 8

인덱스 0: 5 != 8 → 다음
인덱스 1: 3 != 8 → 다음
인덱스 2: 8 == 8 → 찾음! 인덱스 2 반환
```

### 구현

```c
// C
int linearSearch(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;  // 못 찾음
}
```

```cpp
// C++
int linearSearch(const vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

// STL
auto it = find(arr.begin(), arr.end(), target);
if (it != arr.end()) {
    int index = distance(arr.begin(), it);
}
```

```python
# Python
def linear_search(arr, target):
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1

# 내장 함수
# arr.index(target)  # 없으면 ValueError
# target in arr      # 존재 여부만 확인
```

---

## 3. 이진 탐색 (Binary Search)

### 원리

```
정렬된 배열에서 중간값과 비교하여 탐색 범위를 절반씩 줄임

배열: [1, 3, 5, 7, 9, 11, 13], 찾는 값: 9

1단계: left=0, right=6, mid=3
       arr[3]=7 < 9 → left=4

       [1, 3, 5, 7, 9, 11, 13]
                   ↑
                  left

2단계: left=4, right=6, mid=5
       arr[5]=11 > 9 → right=4

       [1, 3, 5, 7, 9, 11, 13]
                   ↑
               left,right

3단계: left=4, right=4, mid=4
       arr[4]=9 == 9 → 찾음! 인덱스 4 반환
```

### 구현

```c
// C - 반복문
int binarySearch(int arr[], int n, int target) {
    int left = 0;
    int right = n - 1;

    while (left <= right) {
        // (left + right) / 2 대신 left + (right - left) / 2를 사용:
        // 두 값이 클 때 (left + right)가 32비트 정수 오버플로우(overflow)를 일으킬 수 있음
        // (예: left = right = 1.5 x 10^9 → 합이 INT_MAX 초과).
        // 이 형태는 먼저 빼서 범위 내에 유지 — C/C++/Java처럼 정수 너비가
        // 고정된 언어에서 필수. Python의 임의 정밀도 정수에서는 단순 형태도 안전하지만
        // 이 스타일이 모범 사례로 간주됨.
        int mid = left + (right - left) / 2;  // 오버플로우 방지

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;   // 오른쪽 절반에 있음
        } else {
            right = mid - 1;  // 왼쪽 절반에 있음
        }
    }

    return -1;  // 모든 후보를 소진 — 존재하지 않음
}

// C - 재귀
int binarySearchRecursive(int arr[], int left, int right, int target) {
    // 기저 조건(base case): 빈 탐색 범위 — 원소가 존재하지 않음
    if (left > right) {
        return -1;
    }

    int mid = left + (right - left) / 2;  // 동일한 오버플로우 안전 중간점

    if (arr[mid] == target) {
        return mid;
    } else if (arr[mid] < target) {
        return binarySearchRecursive(arr, mid + 1, right, target);
    } else {
        return binarySearchRecursive(arr, left, mid - 1, target);
    }
}
```

```cpp
// C++
int binarySearch(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

// STL
#include <algorithm>

// binary_search: 존재 여부만 반환
bool found = binary_search(arr.begin(), arr.end(), target);

// lower_bound: target 이상인 첫 위치
auto it = lower_bound(arr.begin(), arr.end(), target);

// upper_bound: target 초과인 첫 위치
auto it = upper_bound(arr.begin(), arr.end(), target);
```

```python
# Python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# bisect 모듈
import bisect

# bisect_left: target 이상인 첫 위치
idx = bisect.bisect_left(arr, target)

# bisect_right: target 초과인 첫 위치
idx = bisect.bisect_right(arr, target)
```

---

## 4. 이진 탐색 변형

### 4.1 Lower Bound (이상인 첫 위치)

```
target 이상인 첫 번째 원소의 인덱스

배열: [1, 2, 4, 4, 4, 6, 8], target=4

lower_bound(4) = 2  (첫 번째 4의 인덱스)
lower_bound(5) = 5  (5보다 크거나 같은 6의 인덱스)
lower_bound(0) = 0  (모든 원소가 0보다 큼)
lower_bound(9) = 7  (없음, 배열 끝)
```

```c
// C
int lowerBound(int arr[], int n, int target) {
    int left = 0;
    // right = n (마지막 유효 인덱스 + 1), n - 1이 아님:
    // target이 모든 원소보다 클 때 결과가 n이 될 수 있으며,
    // 이는 "끝에 삽입"을 의미 — n - 1을 사용하면 이 경우를 놓침
    int right = n;

    // 루프 조건이 left < right (<= 아님): [left, right) 범위는 반개구간(half-open)이므로
    // left == right이면 범위가 비어서 완료
    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < target) {
            left = mid + 1;  // mid는 확실히 답이 아님 — 제외
        } else {
            // arr[mid] >= target: mid가 답일 수 있으므로 범위에 유지
            right = mid;
        }
    }

    return left;  // left == right: target의 삽입 지점
}
```

```cpp
// C++
int lowerBound(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

```python
# Python
def lower_bound(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

### 4.2 Upper Bound (초과인 첫 위치)

```
target 초과인 첫 번째 원소의 인덱스

배열: [1, 2, 4, 4, 4, 6, 8], target=4

upper_bound(4) = 5  (4보다 큰 첫 원소 6의 인덱스)
upper_bound(5) = 5  (5보다 큰 첫 원소 6의 인덱스)

특정 값의 개수 = upper_bound - lower_bound
4의 개수 = 5 - 2 = 3
```

```c
// C
int upperBound(int arr[], int n, int target) {
    int left = 0;
    int right = n;  // lowerBound와 동일한 반개구간(half-open) [left, right) 규약

    while (left < right) {
        int mid = left + (right - left) / 2;

        // lowerBound와의 유일한 차이: < 대신 <=.
        // 이로 인해 동일 값 원소가 left를 앞으로 밀어, 결과가
        // target의 모든 등장 *다음* 한 위치에 오게 됨
        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

```cpp
// C++
int upperBound(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

```python
# Python
def upper_bound(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2

        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left
```

### 4.3 회전 정렬 배열 탐색

```
회전된 정렬 배열에서 탐색

원본: [0, 1, 2, 4, 5, 6, 7]
회전: [4, 5, 6, 7, 0, 1, 2]

특징: 둘 중 하나는 항상 정렬됨
```

```cpp
// C++
int searchRotated(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;  // 오버플로우 안전 중간점

        if (arr[mid] == target) {
            return mid;
        }

        // 핵심 통찰: 회전된 배열에서 두 절반 중 적어도 하나는 항상
        // 연속적으로 정렬되어 있음. 어느 쪽이 정렬되었는지 판단한 후,
        // target이 그 정렬된 범위 안에 있는지 확인.

        // 왼쪽이 정렬됨: arr[left] <= arr[mid]이면 여기에 회전 지점 없음
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid]) {
                right = mid - 1;  // target이 정렬된 왼쪽 절반 안에 있음
            } else {
                left = mid + 1;   // target은 (회전 가능한) 오른쪽 절반에 있어야 함
            }
        }
        // 오른쪽이 정렬됨: 회전 지점이 왼쪽 절반에 있음
        else {
            if (arr[mid] < target && target <= arr[right]) {
                left = mid + 1;   // target이 정렬된 오른쪽 절반 안에 있음
            } else {
                right = mid - 1;  // target은 (회전된) 왼쪽 절반에 있어야 함
            }
        }
    }

    return -1;
}
```

```python
# Python
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        # 왼쪽 부분이 정렬됨
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 오른쪽 부분이 정렬됨
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

---

## 5. 파라메트릭 서치 (Parametric Search)

### 개념

```
최적화 문제를 결정 문제로 변환하여 이진 탐색으로 해결

"최솟값을 구하라" → "x 이하로 가능한가?"
"최댓값을 구하라" → "x 이상으로 가능한가?"

조건: 답이 단조성(monotonic)을 가져야 함
      - 가능 → 가능 → 가능 → 불가능 → 불가능 (경계 찾기)
```

### 예제 1: 랜선 자르기

```
문제: N개의 랜선을 K개의 같은 길이 랜선으로 자를 때,
      최대 길이는?

랜선 길이: [802, 743, 457, 539], K=11

길이 100으로 자르면: 8+7+4+5 = 24개 ≥ 11 (가능)
길이 200으로 자르면: 4+3+2+2 = 11개 ≥ 11 (가능)
길이 201으로 자르면: 3+3+2+2 = 10개 < 11 (불가능)

→ 최대 길이는 200
```

```cpp
// C++
bool canMake(const vector<int>& cables, int k, long long length) {
    long long count = 0;
    for (int cable : cables) {
        count += cable / length;  // 정수 나눗셈으로 이 길이의 조각이 몇 개 나오는지 계산
    }
    return count >= k;
}

long long maxCableLength(vector<int>& cables, int k) {
    long long left = 1;
    long long right = *max_element(cables.begin(), cables.end());
    long long answer = 0;  // 0 = "아직 유효한 답을 찾지 못함"

    while (left <= right) {
        long long mid = (left + right) / 2;

        if (canMake(cables, k, mid)) {
            // mid가 가능 — 기록하고 더 큰(더 좋은) 답을 탐색
            answer = mid;
            left = mid + 1;  // 더 긴 길이 시도
        } else {
            // mid가 너무 큼 — 탐색 범위를 축소
            right = mid - 1;
        }
    }

    return answer;
}
```

```python
# Python
def can_make(cables, k, length):
    count = sum(cable // length for cable in cables)
    return count >= k

def max_cable_length(cables, k):
    left, right = 1, max(cables)
    answer = 0

    while left <= right:
        mid = (left + right) // 2

        if can_make(cables, k, mid):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

### 예제 2: 나무 자르기

```
문제: N개의 나무를 높이 H로 자를 때,
      M 미터 이상의 나무를 얻을 수 있는 최대 H는?

나무 높이: [20, 15, 10, 17], M=7

H=15로 자르면: 5+0+0+2 = 7m (가능)
H=16으로 자르면: 4+0+0+1 = 5m < 7 (불가능)

→ 최대 H는 15
```

```python
# Python
def can_get_wood(trees, m, height):
    wood = sum(max(0, tree - height) for tree in trees)
    return wood >= m

def max_cut_height(trees, m):
    left, right = 0, max(trees)
    answer = 0

    while left <= right:
        mid = (left + right) // 2

        if can_get_wood(trees, m, mid):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

### 예제 3: 공유기 설치

```
문제: N개의 집에 C개의 공유기를 설치할 때,
      가장 인접한 두 공유기 사이 최대 거리는?

집 위치: [1, 2, 8, 4, 9], C=3 → 정렬: [1, 2, 4, 8, 9]

거리 3으로 설치: 1, 4, 8 또는 1, 4, 9 (가능)
거리 4로 설치: 1, 8 (2개만 가능, 불가)

→ 최대 거리는 3
```

```cpp
// C++
bool canInstall(const vector<int>& houses, int c, int dist) {
    // 탐욕법(greedy): 마지막 공유기에서 충분히 먼 첫 번째 집에 항상 설치 —
    // 더 일찍 설치하는 것이 향후 설치를 방해하지 않으므로 탐욕적 선택이 유효
    // (오른쪽에 더 많은 여유 공간만 남김)
    int count = 1;
    int lastPos = houses[0];

    for (int i = 1; i < houses.size(); i++) {
        if (houses[i] - lastPos >= dist) {
            count++;
            lastPos = houses[i];
        }
    }

    return count >= c;
}

int maxMinDistance(vector<int>& houses, int c) {
    // 먼저 정렬 필수: canInstall은 연속 위치가 인접한 것에 의존하며,
    // 이진 탐색 범위 [1, max_gap]은 정렬된 입력에서만 의미 있음
    sort(houses.begin(), houses.end());

    int left = 1;
    int right = houses.back() - houses.front();
    int answer = 0;

    while (left <= right) {
        int mid = (left + right) / 2;

        if (canInstall(houses, c, mid)) {
            answer = mid;   // mid가 가능 — 기록하고 더 큰 값 시도
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return answer;
}
```

```python
# Python
def can_install(houses, c, dist):
    count = 1
    last_pos = houses[0]

    for house in houses[1:]:
        if house - last_pos >= dist:
            count += 1
            last_pos = house

    return count >= c

def max_min_distance(houses, c):
    houses.sort()

    left, right = 1, houses[-1] - houses[0]
    answer = 0

    while left <= right:
        mid = (left + right) // 2

        if can_install(houses, c, mid):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

---

## 6. 해시 탐색

### 개념

```
해시 함수를 사용하여 O(1) 시간에 탐색

키 → 해시 함수 → 해시값(인덱스) → 데이터

예: "apple" → hash("apple") = 3 → arr[3] = 데이터
```

### 해시 테이블 사용

```cpp
// C++
#include <unordered_map>
#include <unordered_set>

// 해시 맵 (키-값 쌍)
unordered_map<string, int> map;
map["apple"] = 5;
map["banana"] = 3;

// 탐색 O(1)
if (map.count("apple")) {
    cout << map["apple"] << endl;  // 5
}

// 해시 셋 (키만)
unordered_set<int> set;
set.insert(1);
set.insert(2);

// 존재 확인 O(1)
if (set.count(1)) {
    cout << "Found" << endl;
}
```

```python
# Python
# 딕셔너리 (해시 맵)
d = {"apple": 5, "banana": 3}

# 탐색 O(1)
if "apple" in d:
    print(d["apple"])  # 5

# 세트 (해시 셋)
s = {1, 2, 3}

# 존재 확인 O(1)
if 1 in s:
    print("Found")
```

### Two Sum 문제 (해시 활용)

```
문제: 배열에서 두 수의 합이 target인 인덱스 쌍 찾기

배열: [2, 7, 11, 15], target=9
답: [0, 1] (2 + 7 = 9)
```

```cpp
// C++ - O(n) 해시 풀이
vector<int> twoSum(const vector<int>& nums, int target) {
    unordered_map<int, int> seen;  // 값 → 인덱스

    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];

        if (seen.count(complement)) {
            return {seen[complement], i};
        }

        seen[nums[i]] = i;
    }

    return {-1, -1};
}
```

```python
# Python
def two_sum(nums, target):
    seen = {}  # 값 → 인덱스

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i

    return [-1, -1]
```

---

## 7. 연습 문제

### 문제 1: 제곱근 구하기

정수 x의 제곱근을 정수로 반환하세요 (내림).

```
입력: 8
출력: 2 (2.828...의 내림)
```

<details>
<summary>힌트</summary>

mid * mid <= x인 최대 mid 찾기 (이진 탐색)

</details>

<details>
<summary>정답 코드</summary>

```python
def sqrt(x):
    if x < 2:
        return x

    left, right = 1, x // 2
    answer = 1

    while left <= right:
        mid = (left + right) // 2

        if mid * mid <= x:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer
```

</details>

### 문제 2: 배열에서 피크 원소 찾기

피크 원소: 이웃한 원소들보다 큰 원소

```
입력: [1, 2, 3, 1]
출력: 2 (인덱스 2의 원소 3이 피크)
```

<details>
<summary>힌트</summary>

이진 탐색으로 O(log n)에 해결 가능
- mid > mid+1이면 왼쪽에 피크 존재
- mid < mid+1이면 오른쪽에 피크 존재

</details>

<details>
<summary>정답 코드</summary>

```python
def find_peak_element(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        mid = (left + right) // 2

        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [Binary Search](https://leetcode.com/problems/binary-search/) | LeetCode | 기본 이진 탐색 |
| ⭐ | [수 찾기](https://www.acmicpc.net/problem/1920) | 백준 | 이진 탐색 |
| ⭐⭐ | [Search Insert Position](https://leetcode.com/problems/search-insert-position/) | LeetCode | Lower Bound |
| ⭐⭐ | [랜선 자르기](https://www.acmicpc.net/problem/1654) | 백준 | 파라메트릭 서치 |
| ⭐⭐ | [나무 자르기](https://www.acmicpc.net/problem/2805) | 백준 | 파라메트릭 서치 |
| ⭐⭐⭐ | [Search in Rotated Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) | LeetCode | 회전 배열 |
| ⭐⭐⭐ | [공유기 설치](https://www.acmicpc.net/problem/2110) | 백준 | 파라메트릭 서치 |

---

## 이진 탐색 템플릿 정리

### 기본 템플릿

```python
# 정확히 target 찾기
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### Lower Bound / Upper Bound

```python
# target 이상인 첫 위치
def lower_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

# target 초과인 첫 위치
def upper_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left
```

### 파라메트릭 서치

```python
# 조건을 만족하는 최댓값
def parametric_max(check, low, high):
    answer = low
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            answer = mid
            low = mid + 1
        else:
            high = mid - 1
    return answer

# 조건을 만족하는 최솟값
def parametric_min(check, low, high):
    answer = high
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    return answer
```

---

## 다음 단계

- [07_Divide_and_Conquer.md](./07_Divide_and_Conquer.md) - 분할 정복

---

## 참고 자료

- [Binary Search Tutorial](https://www.topcoder.com/thrive/articles/Binary%20Search)
- [Parametric Search Guide](https://cp-algorithms.com/num_methods/binary_search.html)
- [이진 탐색 정리](https://www.acmicpc.net/blog/view/109)
