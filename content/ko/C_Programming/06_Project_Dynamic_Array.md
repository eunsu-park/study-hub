# 프로젝트 4: 동적 배열 (Dynamic Array)

**이전**: [프로젝트 3: 주소록 프로그램](./05_Project_Address_Book.md) | **다음**: [프로젝트 5: 연결 리스트](./07_Project_Linked_List.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컴파일 시점에 요소 개수를 알 수 없을 때 정적 배열이 왜 부족한지 설명할 수 있습니다
2. `malloc`, `calloc`, `realloc`, `free`를 사용하여 NULL 검사를 포함한 동적 메모리 할당을 구현할 수 있습니다
3. `data`, `size`, `capacity`를 별도로 추적하는 확장 가능한 배열 구조체를 설계할 수 있습니다
4. 분할 상환 O(1) push 성능을 달성하기 위한 두 배 증가(doubling) 전략을 적용할 수 있습니다
5. 요소를 이동시켜 연속 저장을 유지하는 삽입 및 삭제 연산을 구축할 수 있습니다
6. `void*` 포인터와 `memcpy`를 사용하여 타입에 독립적인 범용(generic) 동적 배열을 구현할 수 있습니다
7. 메모리 누수(memory leak), 댕글링 포인터(dangling pointer), 이중 해제(double free), 해제 후 사용(use-after-free)과 같은 일반적인 메모리 버그를 식별하고 예방할 수 있습니다

---

Python의 `list`, JavaScript의 `Array`, Java의 `ArrayList`처럼 고급 언어는 크기 조정 가능한 배열을 기본으로 제공합니다. C에서는 그 메커니즘을 직접 구축해야 합니다. 이 프로젝트를 통해 그 방법을 배우고, 그 과정에서 고급 추상화 내부에서 일어나는 일을 정확히 이해하게 됩니다. 힙 메모리 블록을 할당하고, 블록이 너무 작아지면 데이터를 복사하고, 운영체제가 재사용할 수 있도록 이전 블록을 해제하는 것이 바로 그 내부 동작입니다.

## 동적 메모리가 필요한 이유

### 정적 배열의 한계

```c
// Static array: fixed size
int arr[100];  // Size determined at compile time

// Problem 1: Must know size in advance
// Problem 2: Cannot change size
// Problem 3: Wastes unused space
```

### 동적 배열의 장점

```c
// Dynamic array: size can be determined and changed at runtime
int *arr = malloc(n * sizeof(int));  // Size determined at runtime
arr = realloc(arr, m * sizeof(int)); // Size can be changed!
```

---

## 1단계: 동적 메모리 함수 이해

### malloc - Memory Allocation

> **비유 — 주소가 적힌 포스트잇**: C 포인터는 집 주소가 적힌 포스트잇 메모와 같습니다. 메모 자체는 작지만(64비트 시스템에서 8바이트), 그 주소를 따라가면 어떤 크기도 될 수 있는 건물에 도달합니다. `malloc`은 새 건물을 짓고 포스트잇을 건네줍니다. `free`는 그 건물을 철거합니다. 메모를 잃어버리면(`free` 없이 `p = NULL` 로 덮어쓰면), 건물은 여전히 땅을 차지하고 있습니다 — 이것이 메모리 누수(memory leak)입니다.

```c
#include <stdio.h>
#include <stdlib.h>  // malloc, free

int main(void) {
    // Allocate memory for 5 ints
    int *arr = (int *)malloc(5 * sizeof(int));

    // Check for allocation failure (required!)
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Use
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);  // 0 10 20 30 40
    }
    printf("\n");

    // Free (required!)
    free(arr);
    arr = NULL;  // Prevent dangling pointer

    return 0;
}
```

### calloc - Clear Allocation

```c
// calloc: allocate + initialize to 0
int *arr = (int *)calloc(5, sizeof(int));
// arr[0] ~ arr[4] all initialized to 0

// malloc vs calloc
int *m = malloc(5 * sizeof(int));  // Not initialized (garbage values)
int *c = calloc(5, sizeof(int));   // Initialized to 0
```

### realloc - Re-allocation

```c
int *arr = malloc(5 * sizeof(int));

// Expand size (5 -> 10)
int *new_arr = realloc(arr, 10 * sizeof(int));
if (new_arr == NULL) {
    // On failure, original arr remains valid
    free(arr);
    return 1;
}
arr = new_arr;

// Shrink size (10 -> 3)
arr = realloc(arr, 3 * sizeof(int));

free(arr);
```

### realloc 동작 방식

```
+-----------------------------------------------------+
|  realloc(ptr, new_size)                             |
|                                                     |
|  1. If expansion possible at current location:      |
|     [existing data][new space      ]                |
|                                                     |
|  2. If expansion not possible -> copy to new loc    |
|     [original location: freed]                      |
|     [new location: existing data copied][new space] |
|                                                     |
|  3. On failure -> returns NULL (original preserved) |
+-----------------------------------------------------+
```

---

## 2단계: 동적 배열 구조체 설계

### 설계

```c
typedef struct {
    int *data;      // Actual data storage
    int size;       // Current element count
    int capacity;   // Allocated space size
} DynamicArray;
```

### 동작 원리

```
Initial state (capacity=4, size=0):
+---+---+---+---+
|   |   |   |   |  data
+---+---+---+---+

After adding 3 items (capacity=4, size=3):
+---+---+---+---+
| 1 | 2 | 3 |   |  data
+---+---+---+---+

Adding 5th item -> auto expand! (capacity=8, size=5):
+---+---+---+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |   |   |   |  data
+---+---+---+---+---+---+---+---+
```

---

## 3단계: 기본 구현

```c
// dynamic_array.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 4
#define GROWTH_FACTOR 2

// Dynamic array struct
typedef struct {
    int *data;
    int size;
    int capacity;
} DynamicArray;

// Function declarations
DynamicArray* da_create(void);
void da_destroy(DynamicArray *arr);
int da_push(DynamicArray *arr, int value);
int da_pop(DynamicArray *arr, int *value);
int da_get(DynamicArray *arr, int index, int *value);
int da_set(DynamicArray *arr, int index, int value);
int da_insert(DynamicArray *arr, int index, int value);
int da_remove(DynamicArray *arr, int index);
void da_print(DynamicArray *arr);
static int da_resize(DynamicArray *arr, int new_capacity);

// Create
DynamicArray* da_create(void) {
    DynamicArray *arr = (DynamicArray *)malloc(sizeof(DynamicArray));
    if (arr == NULL) {
        return NULL;
    }

    arr->data = (int *)malloc(INITIAL_CAPACITY * sizeof(int));
    if (arr->data == NULL) {
        free(arr);
        return NULL;
    }

    arr->size = 0;
    arr->capacity = INITIAL_CAPACITY;
    return arr;
}

// Destroy
void da_destroy(DynamicArray *arr) {
    if (arr != NULL) {
        free(arr->data);
        free(arr);
    }
}

// Resize (internal function)
static int da_resize(DynamicArray *arr, int new_capacity) {
    int *new_data = (int *)realloc(arr->data, new_capacity * sizeof(int));
    if (new_data == NULL) {
        return -1;  // Failure
    }

    arr->data = new_data;
    arr->capacity = new_capacity;
    return 0;  // Success
}

// Push to end
int da_push(DynamicArray *arr, int value) {
    // Expand if not enough space
    if (arr->size >= arr->capacity) {
        if (da_resize(arr, arr->capacity * GROWTH_FACTOR) != 0) {
            return -1;
        }
    }

    arr->data[arr->size] = value;
    arr->size++;
    return 0;
}

// Pop from end
int da_pop(DynamicArray *arr, int *value) {
    if (arr->size == 0) {
        return -1;  // Empty array
    }

    arr->size--;
    if (value != NULL) {
        *value = arr->data[arr->size];
    }

    // Shrink if too large (optional)
    if (arr->size > 0 && arr->size <= arr->capacity / 4) {
        da_resize(arr, arr->capacity / 2);
    }

    return 0;
}

// Get value by index
int da_get(DynamicArray *arr, int index, int *value) {
    if (index < 0 || index >= arr->size) {
        return -1;  // Out of range
    }

    *value = arr->data[index];
    return 0;
}

// Set value at index
int da_set(DynamicArray *arr, int index, int value) {
    if (index < 0 || index >= arr->size) {
        return -1;
    }

    arr->data[index] = value;
    return 0;
}

// Insert at specific position
int da_insert(DynamicArray *arr, int index, int value) {
    if (index < 0 || index > arr->size) {
        return -1;
    }

    // Ensure space
    if (arr->size >= arr->capacity) {
        if (da_resize(arr, arr->capacity * GROWTH_FACTOR) != 0) {
            return -1;
        }
    }

    // Shift elements right
    for (int i = arr->size; i > index; i--) {
        arr->data[i] = arr->data[i - 1];
    }

    arr->data[index] = value;
    arr->size++;
    return 0;
}

// Remove at specific position
int da_remove(DynamicArray *arr, int index) {
    if (index < 0 || index >= arr->size) {
        return -1;
    }

    // Shift elements left
    for (int i = index; i < arr->size - 1; i++) {
        arr->data[i] = arr->data[i + 1];
    }

    arr->size--;
    return 0;
}

// Print array
void da_print(DynamicArray *arr) {
    printf("DynamicArray(size=%d, capacity=%d): [", arr->size, arr->capacity);
    for (int i = 0; i < arr->size; i++) {
        printf("%d", arr->data[i]);
        if (i < arr->size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

// Test
int main(void) {
    printf("=== Dynamic Array Test ===\n\n");

    // Create
    DynamicArray *arr = da_create();
    if (arr == NULL) {
        printf("Array creation failed\n");
        return 1;
    }

    da_print(arr);

    // Push test
    printf("\n[Push Test]\n");
    for (int i = 1; i <= 10; i++) {
        da_push(arr, i * 10);
        da_print(arr);
    }

    // Get/set test
    printf("\n[Get/Set Test]\n");
    int value;
    da_get(arr, 3, &value);
    printf("arr[3] = %d\n", value);

    da_set(arr, 3, 999);
    da_print(arr);

    // Insert test
    printf("\n[Insert Test]\n");
    da_insert(arr, 0, -100);  // Insert at front
    da_print(arr);

    da_insert(arr, 5, -500);  // Insert in middle
    da_print(arr);

    // Remove test
    printf("\n[Remove Test]\n");
    da_remove(arr, 0);  // Remove from front
    da_print(arr);

    // Pop test
    printf("\n[Pop Test]\n");
    while (arr->size > 0) {
        da_pop(arr, &value);
        printf("Popped: %d, ", value);
        da_print(arr);
    }

    // Destroy
    da_destroy(arr);
    printf("\nArray destroyed\n");

    return 0;
}
```

---

## 4단계: 범용 동적 배열 (void 포인터)

어떤 타입이든 저장할 수 있는 버전:

```c
// generic_array.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *data;
    int size;
    int capacity;
    size_t element_size;  // Size of one element
} GenericArray;

GenericArray* ga_create(size_t element_size) {
    GenericArray *arr = malloc(sizeof(GenericArray));
    if (!arr) return NULL;

    arr->capacity = 4;
    arr->size = 0;
    arr->element_size = element_size;
    arr->data = malloc(arr->capacity * element_size);

    if (!arr->data) {
        free(arr);
        return NULL;
    }

    return arr;
}

void ga_destroy(GenericArray *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

int ga_push(GenericArray *arr, const void *element) {
    if (arr->size >= arr->capacity) {
        int new_cap = arr->capacity * 2;
        void *new_data = realloc(arr->data, new_cap * arr->element_size);
        if (!new_data) return -1;
        arr->data = new_data;
        arr->capacity = new_cap;
    }

    // Copy element
    void *dest = (char *)arr->data + (arr->size * arr->element_size);
    memcpy(dest, element, arr->element_size);
    arr->size++;
    return 0;
}

void* ga_get(GenericArray *arr, int index) {
    if (index < 0 || index >= arr->size) return NULL;
    return (char *)arr->data + (index * arr->element_size);
}

// Test
int main(void) {
    // int array
    printf("=== int array ===\n");
    GenericArray *int_arr = ga_create(sizeof(int));

    for (int i = 0; i < 5; i++) {
        int val = i * 100;
        ga_push(int_arr, &val);
    }

    for (int i = 0; i < int_arr->size; i++) {
        int *val = ga_get(int_arr, i);
        printf("%d ", *val);
    }
    printf("\n");
    ga_destroy(int_arr);

    // double array
    printf("\n=== double array ===\n");
    GenericArray *double_arr = ga_create(sizeof(double));

    for (int i = 0; i < 5; i++) {
        double val = i * 1.5;
        ga_push(double_arr, &val);
    }

    for (int i = 0; i < double_arr->size; i++) {
        double *val = ga_get(double_arr, i);
        printf("%.2f ", *val);
    }
    printf("\n");
    ga_destroy(double_arr);

    // struct array
    printf("\n=== struct array ===\n");
    typedef struct { int x, y; } Point;
    GenericArray *point_arr = ga_create(sizeof(Point));

    Point points[] = {{1, 2}, {3, 4}, {5, 6}};
    for (int i = 0; i < 3; i++) {
        ga_push(point_arr, &points[i]);
    }

    for (int i = 0; i < point_arr->size; i++) {
        Point *p = ga_get(point_arr, i);
        printf("(%d, %d) ", p->x, p->y);
    }
    printf("\n");
    ga_destroy(point_arr);

    return 0;
}
```

---

## 컴파일 및 실행

```bash
gcc -Wall -Wextra -std=c11 dynamic_array.c -o dynamic_array
./dynamic_array
```

---

## 실행 결과

```
=== Dynamic Array Test ===

DynamicArray(size=0, capacity=4): []

[Push Test]
DynamicArray(size=1, capacity=4): [10]
DynamicArray(size=2, capacity=4): [10, 20]
DynamicArray(size=3, capacity=4): [10, 20, 30]
DynamicArray(size=4, capacity=4): [10, 20, 30, 40]
DynamicArray(size=5, capacity=8): [10, 20, 30, 40, 50]  <- Auto expand!
DynamicArray(size=6, capacity=8): [10, 20, 30, 40, 50, 60]
...
```

---

## 배운 내용 정리

| 함수 | 설명 |
|------|------|
| `malloc(size)` | size 바이트 메모리 할당 |
| `calloc(n, size)` | n개 요소, 0으로 초기화 |
| `realloc(ptr, size)` | 크기 변경 |
| `free(ptr)` | 메모리 해제 |
| `memcpy(dest, src, n)` | n 바이트 복사 |

### 메모리 관리 규칙

1. **할당 후 NULL 체크** 필수
2. **사용 후 free()** 필수
3. **free 후 NULL 할당** 권장 (댕글링 포인터(dangling pointer) 방지)
4. **이중 해제(double free) 금지**

---

## 연습 문제

1. **da_find**: 값을 검색하여 인덱스 반환

2. **da_reverse**: 배열 뒤집기

3. **da_sort**: 정렬 기능 추가 (qsort 활용)

4. **문자열 동적 배열**: `char*` 배열 구현

---

## 다음 단계

[프로젝트 5: 연결 리스트 (Linked List)](./07_Project_Linked_List.md) → 포인터의 꽃, 연결 리스트를 배워봅시다!

---

**이전**: [프로젝트 3: 주소록 프로그램](./05_Project_Address_Book.md) | **다음**: [프로젝트 5: 연결 리스트](./07_Project_Linked_List.md)
