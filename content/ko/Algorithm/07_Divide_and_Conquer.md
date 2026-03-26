# 분할 정복 (Divide and Conquer)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 분할 정복(divide and conquer)의 세 단계(분할, 정복, 결합)를 설명하고, 어떤 문제가 이 패러다임에 적합한지 식별할 수 있다
2. 중복 부분 문제(overlapping subproblems)와 메모이제이션(memoization)의 역할을 통해 분할 정복(divide and conquer)과 동적 프로그래밍(dynamic programming, DP)을 구분할 수 있다
3. 분할 정복 패턴을 사용하여 병합 정렬(Merge Sort)을 구현하고, 재귀 트리를 추적하여 O(n log n) 복잡도를 도출할 수 있다
4. 파티션 기반 분할 정복으로 퀵 정렬(Quick Sort)을 구현하고, 최선·평균·최악 케이스 동작을 분석할 수 있다
5. 빠른 거듭제곱(fast exponentiation, 거듭제곱 분할)을 적용하여 큰 거듭제곱을 O(log n) 시간에 계산할 수 있다
6. 마스터 정리(Master Theorem)를 사용하여 분할 정복 알고리즘의 점화식(recurrence relation)을 분석할 수 있다

---

## 개요

분할 정복은 큰 문제를 작은 부분 문제로 나누고, 각각을 해결한 후 결합하는 알고리즘 설계 기법입니다.

---

## 목차

1. [분할 정복 개념](#1-분할-정복-개념)
2. [병합 정렬](#2-병합-정렬)
3. [퀵 정렬](#3-퀵-정렬)
4. [이진 탐색](#4-이진-탐색)
5. [거듭제곱](#5-거듭제곱)
6. [연습 문제](#6-연습-문제)

---

## 1. 분할 정복 개념

### 기본 단계

```
분할 정복 3단계:

1. 분할 (Divide)
   - 문제를 더 작은 부분 문제로 나눔

2. 정복 (Conquer)
   - 부분 문제를 재귀적으로 해결
   - 충분히 작으면 직접 해결

3. 결합 (Combine)
   - 부분 문제의 해를 합쳐 원래 문제의 해 구성
```

### 시각화

```
        [문제]
       /      \
   [부분1]  [부분2]
   /    \    /    \
 [a]   [b] [c]   [d]
   \    /    \    /
   [부분1]  [부분2]
       \      /
        [해답]
```

### 분할 정복 vs DP

```
┌────────────────┬─────────────────┬─────────────────┐
│                │ 분할 정복       │ DP              │
├────────────────┼─────────────────┼─────────────────┤
│ 부분 문제      │ 서로 독립       │ 중복 존재       │
│ 저장           │ 불필요          │ 메모이제이션    │
│ 예시           │ 병합정렬        │ 피보나치        │
└────────────────┴─────────────────┴─────────────────┘
```

---

## 2. 병합 정렬 (Merge Sort)

### 원리

```
1. 배열을 반으로 나눔
2. 각 부분을 재귀적으로 정렬
3. 정렬된 두 부분을 병합

[38, 27, 43, 3, 9, 82, 10]
           ↓ 분할
[38, 27, 43, 3]  [9, 82, 10]
      ↓              ↓
[38, 27] [43, 3] [9, 82] [10]
    ↓       ↓       ↓      ↓
[38][27] [43][3] [9][82]  [10]
    ↓       ↓       ↓      ↓
[27, 38] [3, 43] [9, 82] [10]
      ↓              ↓
[3, 27, 38, 43]  [9, 10, 82]
           ↓ 결합
[3, 9, 10, 27, 38, 43, 82]
```

### 구현

```c
// C
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // VLA(가변 길이 배열)는 간결함을 위해 사용; 프로덕션 C에서는
    // 큰 n에서 스택 오버플로를 방지하기 위해 malloc 사용 권장
    int L[n1], R[n2];

    // 병합 전에 양쪽 절반을 복사 — arr에 쓰기 시작하면
    // arr에서 직접 읽을 경우 원본 값이 손실되기 때문
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        // <=로 안정 정렬(stability) 보장: 같은 값의 왼쪽 원소가 오른쪽보다 먼저 배치됨
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    // 정확히 하나의 루프만 실행되어 아직 처리되지 않은 절반의 나머지 원소를 비움
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        // 오버플로 안전한 중간점 계산 — 인덱스가 클 때를 대비한
        // 좋은 습관 ((left+right)/2도 left=0, right=1일 때 정상 동작하지만)
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
```

```python
def merge_sort(arr):
    # 기저 조건: 원소가 0개 또는 1개이면 이미 정렬됨 — 무한 분할을 방지하는
    # 재귀 종료 조건이기도 함
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    # arr[:mid]와 arr[mid:]는 새 리스트 객체를 생성(레벨당 O(n) 공간),
    # 이것이 Python 병합 정렬이 순진한 형태에서 O(n log n) 공간을 쓰는 이유;
    # 제자리(in-place) 변형은 이를 피하지만 상당히 복잡해짐
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        # <=로 안정 정렬(stability) 보장: 같은 값의 원소 중 왼쪽 부분 배열의
        # 원소가 항상 우선하여 원래 입력 순서를 유지함
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # extend는 나머지 원소에 대해 O(k) — 원소별 Python 오버헤드를 피하므로
    # append 루프보다 훨씬 효율적
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### 복잡도 분석

```
T(n) = 2T(n/2) + O(n)

분할: O(1)
정복: 2 × T(n/2)
결합: O(n)

마스터 정리:
a = 2, b = 2, f(n) = n
n^(log_b(a)) = n^1 = n
f(n) = Θ(n)

→ T(n) = Θ(n log n)

공간: O(n) - 임시 배열
```

---

## 3. 퀵 정렬 (Quick Sort)

### 원리

```
1. 피벗 선택
2. 피벗보다 작은 원소는 왼쪽, 큰 원소는 오른쪽
3. 각 부분을 재귀적으로 정렬

[5, 3, 8, 4, 2, 7, 1, 6], 피벗=5
           ↓ 파티션
[3, 4, 2, 1] [5] [8, 7, 6]
      ↓              ↓
[1, 2, 3, 4]    [6, 7, 8]
           ↓
[1, 2, 3, 4, 5, 6, 7, 8]
```

### 구현

```cpp
// C++
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    // i는 low - 1에서 시작: 피벗보다 작다고 확인된 마지막 원소의 인덱스;
    // 작은 원소를 찾을 때마다 [low..i] 영역이 확장됨
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            // 피벗보다 작은 영역을 확장하고 새 원소를 해당 영역으로 교환
            swap(arr[++i], arr[j]);
        }
    }

    // 피벗을 올바른 정렬 위치에 배치 — 왼쪽은 모두 작고,
    // 오른쪽은 모두 큰 값 (제자리, 추가 메모리 불필요)
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // 피벗은 이제 최종 위치에 있으므로 정렬되지 않은 파티션만 재귀
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

```python
def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

### 복잡도 분석

```
평균: T(n) = 2T(n/2) + O(n) = O(n log n)
최악: T(n) = T(n-1) + O(n) = O(n²)
      (이미 정렬된 배열 + 첫/마지막 피벗)

공간: O(log n) - 재귀 스택
```

---

## 4. 이진 탐색 (Binary Search)

### 분할 정복 관점

```
문제: 정렬된 배열에서 target 찾기

분할: 중간값 기준으로 왼쪽/오른쪽 나눔
정복: target이 있는 쪽만 탐색
결합: 필요 없음 (찾으면 반환)

[1, 3, 5, 7, 9, 11, 13], target=9
           ↓
       [7] 중간값
      7 < 9
           ↓
    오른쪽 탐색: [9, 11, 13]
           ↓
       [11] 중간값
      11 > 9
           ↓
    왼쪽 탐색: [9]
           ↓
       찾음!
```

### 재귀 구현

```cpp
// C++
int binarySearchRecursive(const vector<int>& arr, int left, int right, int target) {
    if (left > right) return -1;

    int mid = left + (right - left) / 2;

    if (arr[mid] == target) return mid;
    if (arr[mid] > target) return binarySearchRecursive(arr, left, mid - 1, target);
    return binarySearchRecursive(arr, mid + 1, right, target);
}
```

```python
def binary_search_recursive(arr, left, right, target):
    if left > right:
        return -1

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    if arr[mid] > target:
        return binary_search_recursive(arr, left, mid - 1, target)
    return binary_search_recursive(arr, mid + 1, right, target)
```

---

## 5. 거듭제곱 (Exponentiation)

### 단순 방법 vs 분할 정복

```
a^n 계산

단순: a × a × a × ... × a (n번) → O(n)

분할 정복:
a^n = a^(n/2) × a^(n/2)        (n이 짝수)
a^n = a^(n/2) × a^(n/2) × a    (n이 홀수)

→ O(log n)

예: 2^10
= 2^5 × 2^5
= (2^2 × 2^2 × 2) × (2^2 × 2^2 × 2)
= ((2 × 2)^2 × 2)^2
```

### 구현

```c
// C - 재귀
long long power(long long a, int n) {
    // n == 0: 어떤 수의 0승이든 1 (빈 곱, empty product)
    if (n == 0) return 1;
    // n == 1은 명시적 단축 경로로, 꼭 필요하지는 않지만
    // 일반적인 단일 단계 케이스에서 불필요한 제곱 단계를 방지
    if (n == 1) return a;

    // a^(n/2)를 한 번 계산하고 재사용 — O(n) 곱셈을
    // O(log n)으로 줄이는 분할 단계 (선형 곱셈 대신 제곱 활용)
    long long half = power(a, n / 2);

    if (n % 2 == 0) {
        return half * half;
    } else {
        // 홀수 지수: a^n = a^(n/2) * a^(n/2) * a — 정수 나눗셈에서
        // n/2가 내림될 때 나머지를 추가 'a'로 처리
        return half * half * a;
    }
}

// C - 반복 (비트 연산)
long long powerIterative(long long a, int n) {
    long long result = 1;

    while (n > 0) {
        // n의 비트를 하나씩 처리: 현재 비트가 1이면 현재 거듭제곱 값을
        // 결과에 곱함 (비트 = 1은 이 항이 기여함을 의미)
        if (n & 1) {  // n이 홀수
            result *= a;
        }
        // 매 단계마다 a를 제곱: k번 반복 후 a = original_a^(2^k)
        a *= a;
        // 오른쪽 시프트하여 지수의 다음 비트를 검사
        n >>= 1;
    }

    return result;
}
```

```python
def power(a, n):
    if n == 0:
        return 1
    if n == 1:
        return a

    half = power(a, n // 2)

    if n % 2 == 0:
        return half * half
    else:
        return half * half * a

# 모듈러 거듭제곱 (큰 수)
def power_mod(a, n, mod):
    result = 1
    a %= mod

    while n > 0:
        if n & 1:
            result = (result * a) % mod
        a = (a * a) % mod
        n >>= 1

    return result
```

### 행렬 거듭제곱

```
피보나치를 O(log n)에 계산

[F(n+1)]   [1 1]^n   [F(1)]
[F(n)  ] = [1 0]   × [F(0)]

행렬 거듭제곱을 분할 정복으로!
```

```python
def matrix_mult(A, B, mod=10**9+7):
    # 표준 2x2 행렬 곱셈; 매 단계마다 mod를 적용하여
    # Python 정수가 수백만 자릿수로 커지는 것을 방지 (성능)
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]
    ]

def matrix_power(M, n, mod=10**9+7):
    # 기저 조건: M^1 = M (항등 행렬은 M^0이지만 n=1부터 시작)
    if n == 1:
        return M

    if n % 2 == 0:
        # 절반 거듭제곱의 결과를 제곱 — 스칼라 거듭제곱과 동일한
        # 분할 정복 아이디어; 재귀 호출을 O(n)에서 O(log n)으로 축소
        half = matrix_power(M, n // 2, mod)
        return matrix_mult(half, half, mod)
    else:
        # 홀수 지수: M 하나를 분리하고 짝수 부분을 재귀적으로 처리
        return matrix_mult(M, matrix_power(M, n - 1, mod), mod)

def fibonacci(n):
    if n <= 1:
        return n

    # 점화식 [[F(n+1)], [F(n)]] = [[1,1],[1,0]]^n × [[F(1)],[F(0)]]을 통해
    # O(n) 대신 O(log n) 곱셈으로 피보나치를 계산할 수 있음
    M = [[1, 1], [1, 0]]
    result = matrix_power(M, n)
    return result[1][0]  # n번의 행렬 곱셈 후 result[1][0] = F(n)
```

---

## 6. 연습 문제

### 문제 1: 배열의 역순 쌍 개수

i < j이고 arr[i] > arr[j]인 쌍의 개수

<details>
<summary>힌트</summary>

병합 정렬 과정에서 카운트

</details>

<details>
<summary>정답 코드</summary>

```python
def count_inversions(arr):
    def merge_count(arr, temp, left, mid, right):
        i = left
        j = mid + 1
        k = left
        inv_count = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                inv_count += (mid - i + 1)  # 핵심!
                j += 1
            k += 1

        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1

        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1

        for i in range(left, right + 1):
            arr[i] = temp[i]

        return inv_count

    def merge_sort_count(arr, temp, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += merge_sort_count(arr, temp, left, mid)
            inv_count += merge_sort_count(arr, temp, mid + 1, right)
            inv_count += merge_count(arr, temp, left, mid, right)
        return inv_count

    n = len(arr)
    temp = [0] * n
    return merge_sort_count(arr, temp, 0, n - 1)
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐ | [행렬 제곱](https://www.acmicpc.net/problem/10830) | 백준 | 행렬 거듭제곱 |
| ⭐⭐ | [피보나치 수 6](https://www.acmicpc.net/problem/11444) | 백준 | 행렬 거듭제곱 |
| ⭐⭐ | [K번째 수](https://www.acmicpc.net/problem/11004) | 백준 | Quick Select |
| ⭐⭐⭐ | [버블 소트](https://www.acmicpc.net/problem/1517) | 백준 | 역순 쌍 |
| ⭐⭐⭐ | [가장 가까운 두 점](https://www.acmicpc.net/problem/2261) | 백준 | 분할 정복 |

---

## 마스터 정리 (Master Theorem)

```
T(n) = aT(n/b) + f(n) 형태의 점화식 해결

Case 1: f(n) = O(n^(log_b(a) - ε))
        → T(n) = Θ(n^(log_b(a)))

Case 2: f(n) = Θ(n^(log_b(a)))
        → T(n) = Θ(n^(log_b(a)) log n)

Case 3: f(n) = Ω(n^(log_b(a) + ε))
        → T(n) = Θ(f(n))

예:
- 병합 정렬: T(n) = 2T(n/2) + n → O(n log n) [Case 2]
- 이진 탐색: T(n) = T(n/2) + 1 → O(log n) [Case 2]
- 거듭제곱: T(n) = T(n/2) + 1 → O(log n) [Case 2]
```

---

## 다음 단계

- [08_Backtracking.md](./08_Backtracking.md) - 백트래킹

---

## 참고 자료

- Introduction to Algorithms (CLRS) - Chapter 4
- [Divide and Conquer](https://www.geeksforgeeks.org/divide-and-conquer/)
