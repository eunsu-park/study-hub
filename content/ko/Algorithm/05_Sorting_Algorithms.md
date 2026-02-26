# 정렬 알고리즘 (Sorting Algorithms)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 버블 정렬(Bubble Sort), 선택 정렬(Selection Sort), 삽입 정렬(Insertion Sort), 병합 정렬(Merge Sort), 퀵 정렬(Quick Sort), 힙 정렬(Heap Sort), 계수 정렬(Counting Sort)의 최선·평균·최악 시간 복잡도와 공간 복잡도를 비교할 수 있다
2. 안정 정렬(stable sort)과 불안정 정렬(unstable sort)을 구분하고 안정성이 중요한 상황을 설명할 수 있다
3. 병합 정렬과 퀵 정렬을 직접 구현하며, 각 알고리즘의 분할 정복(divide-and-conquer) 전략을 명확히 설명할 수 있다
4. 계수 정렬(Counting Sort)이 O(n+k) 시간을 달성하는 이유와 적용 가능한 제약 조건을 설명할 수 있다
5. 퀵 정렬(Quick Sort)의 최악 케이스 동작을 분석하고 이를 완화하는 피벗(pivot) 선택 전략을 설명할 수 있다
6. 입력 크기, 값의 범위, 메모리 제약, 안정성 요구 사항을 바탕으로 주어진 문제에 가장 적합한 정렬 알고리즘을 선택할 수 있다

---

## 개요

정렬은 데이터를 특정 순서로 배열하는 기본적이면서도 중요한 알고리즘입니다. 다양한 정렬 알고리즘의 원리, 구현, 시간/공간 복잡도를 학습합니다.

---

## 목차

1. [정렬 알고리즘 비교](#1-정렬-알고리즘-비교)
2. [버블 정렬](#2-버블-정렬)
3. [선택 정렬](#3-선택-정렬)
4. [삽입 정렬](#4-삽입-정렬)
5. [병합 정렬](#5-병합-정렬)
6. [퀵 정렬](#6-퀵-정렬)
7. [힙 정렬](#7-힙-정렬)
8. [계수 정렬](#8-계수-정렬)
9. [정렬 알고리즘 선택](#9-정렬-알고리즘-선택)
10. [연습 문제](#10-연습-문제)

---

## 1. 정렬 알고리즘 비교

### 복잡도 비교표

```
┌─────────────┬───────────────────────────────────┬─────────┬────────┐
│  알고리즘   │         시간 복잡도               │  공간   │ 안정성 │
│             │  최선    │  평균    │  최악     │ 복잡도  │        │
├─────────────┼──────────┼──────────┼───────────┼─────────┼────────┤
│ 버블 정렬   │ O(n)     │ O(n²)    │ O(n²)     │ O(1)    │ 안정   │
│ 선택 정렬   │ O(n²)    │ O(n²)    │ O(n²)     │ O(1)    │ 불안정 │
│ 삽입 정렬   │ O(n)     │ O(n²)    │ O(n²)     │ O(1)    │ 안정   │
│ 병합 정렬   │ O(nlogn) │ O(nlogn) │ O(nlogn)  │ O(n)    │ 안정   │
│ 퀵 정렬     │ O(nlogn) │ O(nlogn) │ O(n²)     │ O(logn) │ 불안정 │
│ 힙 정렬     │ O(nlogn) │ O(nlogn) │ O(nlogn)  │ O(1)    │ 불안정 │
│ 계수 정렬   │ O(n+k)   │ O(n+k)   │ O(n+k)    │ O(k)    │ 안정   │
└─────────────┴──────────┴──────────┴───────────┴─────────┴────────┘
* k: 값의 범위
```

### 안정 정렬 (Stable Sort)

```
안정 정렬: 같은 값을 가진 원소들의 상대적 순서가 정렬 후에도 유지

예: [(A,3), (B,1), (C,3)] 을 숫자로 정렬

안정 정렬:   [(B,1), (A,3), (C,3)]  ← A가 C보다 앞 (원래 순서 유지)
불안정 정렬: [(B,1), (C,3), (A,3)]  ← 순서 바뀔 수 있음
```

---

## 2. 버블 정렬 (Bubble Sort)

### 원리

```
인접한 두 원소를 비교하여 교환, 가장 큰 원소가 끝으로 "버블"처럼 이동

배열: [5, 3, 8, 4, 2]

1회차: 5와 3 비교/교환 → [3, 5, 8, 4, 2]
       5와 8 비교 (유지) → [3, 5, 8, 4, 2]
       8와 4 비교/교환 → [3, 5, 4, 8, 2]
       8와 2 비교/교환 → [3, 5, 4, 2, 8] ← 8 확정

2회차: [3, 5, 4, 2, 8]
       → [3, 4, 2, 5, 8] ← 5 확정

3회차: → [3, 2, 4, 5, 8] ← 4 확정

4회차: → [2, 3, 4, 5, 8] ← 완료
```

### 구현

```c
// C
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        // 내부 루프가 n - i - 1에서 멈춤: i번째 패스 후 마지막 i개 원소는
        // 이미 최종 정렬 위치에 있으므로 다시 확인할 필요가 없음
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// 최적화: 교환이 없으면 조기 종료
void bubbleSortOptimized(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;

        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = 1;
            }
        }

        // 교환이 발생하지 않았으면 배열이 이미 정렬된 것 — 거의 정렬된
        // 입력에 대해 최선 복잡도를 O(n²)에서 O(n)으로 개선
        if (!swapped) break;  // 이미 정렬됨
    }
}
```

```cpp
// C++
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;

        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped) break;
    }
}
```

```python
# Python
def bubble_sort(arr):
    n = len(arr)

    for i in range(n - 1):
        swapped = False

        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return arr
```

---

## 3. 선택 정렬 (Selection Sort)

### 원리

```
매 단계에서 최솟값을 찾아 맨 앞과 교환

배열: [5, 3, 8, 4, 2]

1회차: 최솟값 2 → 맨 앞과 교환
       [2, 3, 8, 4, 5] ← 2 확정

2회차: [2] 제외, 나머지에서 최솟값 3
       [2, 3, 8, 4, 5] ← 3 확정 (이미 위치)

3회차: [2, 3] 제외, 나머지에서 최솟값 4
       [2, 3, 4, 8, 5] ← 4 확정

4회차: [2, 3, 4] 제외, 나머지에서 최솟값 5
       [2, 3, 4, 5, 8] ← 완료
```

### 구현

```c
// C
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;

        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }

        if (minIdx != i) {
            int temp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = temp;
        }
    }
}
```

```cpp
// C++
void selectionSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;

        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }

        if (minIdx != i) {
            swap(arr[i], arr[minIdx]);
        }
    }
}
```

```python
# Python
def selection_sort(arr):
    n = len(arr)

    for i in range(n - 1):
        min_idx = i

        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr
```

---

## 4. 삽입 정렬 (Insertion Sort)

### 원리

```
현재 원소를 이미 정렬된 부분의 적절한 위치에 삽입

배열: [5, 3, 8, 4, 2]

초기: [5] ← 첫 원소는 정렬됨

1회차: 3 삽입 → [3, 5]
       3 < 5 이므로 5를 오른쪽으로 이동, 3 삽입

2회차: 8 삽입 → [3, 5, 8]
       8 > 5 이므로 그대로

3회차: 4 삽입 → [3, 4, 5, 8]
       4 < 8, 4 < 5, 4 > 3 → 5,8 이동 후 삽입

4회차: 2 삽입 → [2, 3, 4, 5, 8]
       2 < 8, 2 < 5, 2 < 4, 2 < 3 → 모두 이동 후 삽입
```

### 구현

```c
// C
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        // key보다 큰 원소들을 오른쪽으로 이동
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}
```

```cpp
// C++
void insertionSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}
```

```python
# Python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr
```

### 삽입 정렬의 장점

```
1. 거의 정렬된 배열에서 O(n) - 매우 빠름
2. 작은 데이터셋에 효율적
3. 안정 정렬
4. 제자리 정렬 (추가 메모리 불필요)
5. 온라인 알고리즘 (데이터가 들어올 때마다 정렬 가능)
```

---

## 5. 병합 정렬 (Merge Sort)

### 원리

```
분할 정복: 배열을 반으로 나누고, 각각 정렬한 후 병합

배열: [5, 3, 8, 4, 2, 7, 1, 6]

분할:
         [5, 3, 8, 4, 2, 7, 1, 6]
              /              \
       [5, 3, 8, 4]      [2, 7, 1, 6]
        /      \          /      \
     [5, 3]  [8, 4]    [2, 7]  [1, 6]
     /   \    /   \    /   \    /   \
   [5]  [3] [8]  [4] [2]  [7] [1]  [6]

병합:
   [5]  [3] [8]  [4] [2]  [7] [1]  [6]
     \  /     \  /     \  /     \  /
    [3, 5]  [4, 8]   [2, 7]   [1, 6]
        \    /           \    /
      [3, 4, 5, 8]    [1, 2, 6, 7]
            \              /
         [1, 2, 3, 4, 5, 6, 7, 8]
```

### 구현

```c
// C
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // 임시 배열이 필요함 — arr에 직접 병합하면 반대쪽 비교에 아직 필요한
    // 원소를 덮어쓰게 됨
    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        // <= (<가 아님)로 안정성(stability)을 보장: 왼쪽 부분 배열의 동일 값 원소가
        // 먼저 배치되어 원래 상대적 순서가 유지됨
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    // 남은 원소를 비움 — 이 두 루프 중 최대 하나만 실행됨
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        // 오버플로우(overflow) 안전한 중간점: left + right가 INT_MAX를 초과할 때
        // 정수 오버플로우를 방지
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
```

```cpp
// C++
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);

    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
```

```python
# Python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
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

---

## 6. 퀵 정렬 (Quick Sort)

### 원리

```
피벗을 선택하여 피벗보다 작은 원소는 왼쪽, 큰 원소는 오른쪽으로 분할

배열: [5, 3, 8, 4, 2, 7, 1, 6], 피벗 = 5 (첫 원소)

파티셔닝:
  피벗=5
  작은 것: [3, 4, 2, 1]
  큰 것:   [8, 7, 6]

  → [3, 4, 2, 1] + [5] + [8, 7, 6]

재귀적으로 좌우 정렬:
  [1, 2, 3, 4] + [5] + [6, 7, 8]

결과: [1, 2, 3, 4, 5, 6, 7, 8]
```

### Lomuto 파티셔닝

```
마지막 원소를 피벗으로 사용

배열: [5, 3, 8, 4, 2], 피벗 = 2

i = -1 (피벗보다 작은 영역의 끝)

j=0: 5 > 2 → 건너뜀
j=1: 3 > 2 → 건너뜀
j=2: 8 > 2 → 건너뜀
j=3: 4 > 2 → 건너뜀

피벗을 i+1 위치로 이동:
[2, 3, 8, 4, 5]
 ↑
피벗 위치
```

### 구현

```c
// C - Lomuto 파티셔닝
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    // i는 부분 배열 시작 바로 이전 위치에서 시작 — "피벗보다 작은 원소" 영역의
    // 경계를 표시하며, 영역을 확장할 때만 i가 증가
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            // 피벗보다 작은 영역을 확장하고 새 원소를 그 안으로 교환
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // 피벗을 최종 위치(i + 1)에 배치: 왼쪽은 모두 작고 오른쪽은 모두 큼
    // — 이것이 O(n) 파티션을 보장
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // 피벗은 이미 최종 정렬 위치에 있음 — 양쪽 재귀 호출에서 제외하여
        // 무한 루프를 방지하고 불필요한 비교를 피함
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

```cpp
// C++
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

```python
# Python
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

# 사용
# arr = [5, 3, 8, 4, 2, 7, 1, 6]
# quick_sort(arr, 0, len(arr) - 1)
```

### 피벗 선택 전략

```
1. 첫 원소 / 마지막 원소: 간단하지만 정렬된 배열에서 O(n²)
2. 랜덤 피벗: 평균적으로 좋은 성능
3. 중간값 (Median of Three): 첫, 중간, 끝 원소 중 중간값

import random

def partition_random(arr, low, high):
    # 랜덤 피벗 선택
    rand_idx = random.randint(low, high)
    arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
    return partition(arr, low, high)
```

---

## 7. 힙 정렬 (Heap Sort)

### 원리

```
최대 힙을 구성한 후, 루트(최댓값)를 꺼내어 정렬

배열을 힙으로 시각화:
        [16]                  인덱스:
       /    \                    0
     [14]   [10]              1    2
     /  \   /  \            3  4  5  6
   [8] [7] [9] [3]
   /\
 [2][4]

배열: [16, 14, 10, 8, 7, 9, 3, 2, 4]

정렬 과정:
1. 최대 힙 구성
2. 루트(16)와 마지막 원소 교환 → [16] 확정
3. 나머지로 힙 재구성
4. 반복
```

### 구현

```c
// C
void heapify(int arr[], int n, int i) {
    int largest = i;
    // 0-인덱스 이진 힙(binary heap)에서 노드 i의 자식은 2*i+1(왼쪽)과
    // 2*i+2(오른쪽) — 이 산술이 명시적 트리 구조를 대체
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;

        // 교환으로 인해 영향받은 서브트리를 재귀적으로 수정 —
        // 한 번의 교환이 트리 아래쪽의 힙 속성을 추가로 위반할 수 있음
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    // 마지막 내부 노드부터 위로 heapify하여 최대 힙 구성 —
    // 리프 노드는 자명하게 유효한 힙이므로 n/2-1부터 시작하여 건너뛰면
    // O(n log n)이 아닌 O(n) 빌드 시간을 달성
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // 힙의 루트(현재 최댓값)를 정렬된 접미사로 반복 이동한 후,
    // 남은 n-i-1개 원소에 대해 힙 속성을 복원
    for (int i = n - 1; i > 0; i--) {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        heapify(arr, i, 0);
    }
}
```

```cpp
// C++
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();

    // 최대 힙 구성
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // 하나씩 추출
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

```python
# Python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # 최대 힙 구성
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 하나씩 추출
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr
```

---

## 8. 계수 정렬 (Counting Sort)

### 원리

```
원소의 개수를 세어 정렬 (비교 기반이 아님)
조건: 원소가 정수이고 범위가 제한적일 때

배열: [4, 2, 2, 8, 3, 3, 1]
범위: 1~8

개수 세기:
값:    1  2  3  4  5  6  7  8
개수: [1, 2, 2, 1, 0, 0, 0, 1]

누적 합:
      [1, 3, 5, 6, 6, 6, 6, 7]

결과 배열 생성 (뒤에서부터):
원소 1 → 위치 1 → result[0] = 1
원소 3 → 위치 5 → result[4] = 3
...

결과: [1, 2, 2, 3, 3, 4, 8]
```

### 구현

```c
// C
void countingSort(int arr[], int n) {
    // 최댓값 찾기
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
    }

    // 개수 배열
    int* count = (int*)calloc(max + 1, sizeof(int));

    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }

    // 누적 합
    for (int i = 1; i <= max; i++) {
        count[i] += count[i - 1];
    }

    // 결과 배열
    int* output = (int*)malloc(n * sizeof(int));

    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(count);
    free(output);
}
```

```cpp
// C++
void countingSort(vector<int>& arr) {
    if (arr.empty()) return;

    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;

    vector<int> count(range, 0);
    vector<int> output(arr.size());

    for (int x : arr) {
        count[x - minVal]++;
    }

    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }

    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i] - minVal] - 1] = arr[i];
        count[arr[i] - minVal]--;
    }

    arr = output;
}
```

```python
# Python
def counting_sort(arr):
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    # 값을 이동시켜 min_val이 인덱스 0에 대응 — 음수나 큰 오프셋(offset)의
    # 정수도 0~max_val까지의 거대한 배열 할당 없이 처리 가능
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    for x in arr:
        count[x - min_val] += 1

    # 개수를 누적 합으로 변환: count[i]는 이제 값 (min_val + i)의 *마지막*
    # 등장이 출력에서 위치할 1-기반 위치를 나타냄
    for i in range(1, range_val):
        count[i] += count[i - 1]

    # 입력을 뒤에서부터 순회하여 안정성(stability)을 달성: 동일 값 원소 중
    # 입력에서 나중에 나온 것이 출력에서도 나중에 배치되어
    # 원래 상대적 순서가 보존됨 (안정 정렬의 보장)
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1  # 다음 동일 원소가 한 칸 앞에 배치되도록 감소

    return output
```

---

## 9. 정렬 알고리즘 선택

### 상황별 추천

```
┌────────────────────────────────┬─────────────────────────┐
│ 상황                           │ 추천 알고리즘           │
├────────────────────────────────┼─────────────────────────┤
│ 소규모 데이터 (n < 50)         │ 삽입 정렬               │
│ 거의 정렬된 데이터             │ 삽입 정렬               │
│ 메모리 제한 있음               │ 힙 정렬, 퀵 정렬        │
│ 안정 정렬 필요                 │ 병합 정렬               │
│ 최악의 경우도 O(n log n) 보장  │ 병합 정렬, 힙 정렬      │
│ 평균적으로 가장 빠름           │ 퀵 정렬                 │
│ 정수, 범위 제한적              │ 계수 정렬               │
│ 범용 라이브러리                │ Timsort (병합+삽입)     │
└────────────────────────────────┴─────────────────────────┘
```

### 실제 사용

```cpp
// C++ STL
#include <algorithm>

vector<int> arr = {5, 3, 8, 4, 2};

// 오름차순 정렬
sort(arr.begin(), arr.end());

// 내림차순 정렬
sort(arr.begin(), arr.end(), greater<int>());

// 커스텀 비교 함수
sort(arr.begin(), arr.end(), [](int a, int b) {
    return abs(a) < abs(b);  // 절댓값 기준
});

// 안정 정렬
stable_sort(arr.begin(), arr.end());
```

```python
# Python
arr = [5, 3, 8, 4, 2]

# 오름차순 (Timsort)
sorted_arr = sorted(arr)

# 내림차순
sorted_arr = sorted(arr, reverse=True)

# 커스텀 키
sorted_arr = sorted(arr, key=lambda x: abs(x))

# 제자리 정렬
arr.sort()
```

---

## 10. 연습 문제

### 문제 1: K번째 큰 원소

정렬을 사용하지 않고 K번째로 큰 원소를 O(n) 평균 시간에 찾으세요.

<details>
<summary>힌트</summary>

Quick Select 알고리즘: 퀵 정렬의 파티셔닝을 활용

</details>

<details>
<summary>정답 코드</summary>

```python
def find_kth_largest(arr, k):
    k = len(arr) - k  # k번째 큰 = (n-k)번째 작은

    def quick_select(left, right):
        pivot = arr[right]
        i = left

        for j in range(left, right):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1

        arr[i], arr[right] = arr[right], arr[i]

        if i == k:
            return arr[i]
        elif i < k:
            return quick_select(i + 1, right)
        else:
            return quick_select(left, i - 1)

    return quick_select(0, len(arr) - 1)
```

</details>

### 문제 2: 색깔 정렬 (Dutch National Flag)

0, 1, 2로만 구성된 배열을 한 번의 순회로 정렬하세요.

```
입력: [2, 0, 2, 1, 1, 0]
출력: [0, 0, 1, 1, 2, 2]
```

<details>
<summary>정답 코드</summary>

```python
def sort_colors(arr):
    low, mid, high = 0, 0, len(arr) - 1

    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:  # arr[mid] == 2
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr

# 시간: O(n), 공간: O(1)
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [수 정렬하기](https://www.acmicpc.net/problem/2750) | 백준 | 기초 정렬 |
| ⭐ | [Sort Colors](https://leetcode.com/problems/sort-colors/) | LeetCode | 3-way 파티셔닝 |
| ⭐⭐ | [수 정렬하기 2](https://www.acmicpc.net/problem/2751) | 백준 | O(n log n) |
| ⭐⭐ | [Merge Intervals](https://leetcode.com/problems/merge-intervals/) | LeetCode | 정렬 응용 |
| ⭐⭐ | [Kth Largest Element](https://leetcode.com/problems/kth-largest-element-in-an-array/) | LeetCode | Quick Select |
| ⭐⭐⭐ | [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) | LeetCode | 이진 탐색 |

---

## 다음 단계

- [06_Searching_Algorithms.md](./06_Searching_Algorithms.md) - 이진 탐색, 파라메트릭 서치

---

## 참고 자료

- [Sorting Algorithms Visualized](https://www.toptal.com/developers/sorting-algorithms)
- [VisuAlgo - Sorting](https://visualgo.net/en/sorting)
- Introduction to Algorithms (CLRS) - Chapter 7, 8
