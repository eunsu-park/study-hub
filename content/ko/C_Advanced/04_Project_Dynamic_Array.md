# 프로젝트: 동적 배열

**이전**: [비트 연산](./03_Bit_Operations.md) | **다음**: [프로젝트: 연결 리스트](./05_Project_Linked_List.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컴파일 시점에 요소 수를 알 수 없을 때 정적 배열이 부족한 이유를 설명할 수 있다
2. `malloc`, `calloc`, `realloc`, `free`를 적절한 NULL 검사와 함께 사용하여 동적 메모리 할당을 구현할 수 있다
3. `data`, `size`, `capacity`를 별도로 추적하는 가변 배열 구조체를 설계할 수 있다
4. 분할 상환(amortized) O(1) push 성능을 달성하기 위한 2배 성장 전략을 적용할 수 있다
5. 요소를 시프트하고 연속 저장을 유지하는 삽입 및 삭제 연산을 구축할 수 있다
6. `void*` 포인터와 `memcpy`를 사용한 제네릭 동적 배열을 구현하여 타입 독립적 저장이 가능하다
7. 일반적인 메모리 버그(누수, 댕글링 포인터, 이중 해제, use-after-free)를 식별하고 방지할 수 있다

---

모든 고수준 언어는 크기 조절 가능한 배열을 기본 제공합니다 -- Python의 `list`, JavaScript의 `Array`, Java의 `ArrayList`. C에서는 이 메커니즘을 직접 구축합니다. 이 프로젝트가 그 방법을 가르쳐주며, 그 과정에서 고수준 추상화가 내부적으로 무엇을 하는지 정확히 드러납니다: 힙 메모리 블록을 할당하고, 블록이 너무 작으면 데이터를 복사하며, 운영체제가 재사용할 수 있도록 이전 블록을 해제합니다.

## 동적 메모리가 필요한 이유

### 정적 배열의 한계

```c
// 정적 배열: 고정 크기
int arr[100];  // 크기가 컴파일 시점에 결정됨

// 문제 1: 크기를 미리 알아야 함
// 문제 2: 크기를 변경할 수 없음
// 문제 3: 사용하지 않는 공간이 낭비됨
```

### 동적 배열의 장점

```c
// 동적 배열: 런타임에 크기를 결정하고 변경 가능
int *arr = malloc(n * sizeof(int));  // 런타임에 크기 결정
arr = realloc(arr, m * sizeof(int)); // 크기 변경 가능!
```

---

## 단계 1: 동적 메모리 함수 이해

### malloc - 메모리 할당(Memory Allocation)

> **비유 -- 주소가 적힌 포스트잇**: C 포인터는 거리 주소가 적힌 포스트잇과 같습니다. 메모 자체는 작지만 (64비트 시스템에서 8바이트), 주소를 따라가면 어떤 크기든 될 수 있는 건물에 도착합니다. `malloc`은 새 건물을 짓고 포스트잇을 건네주며, `free`는 건물을 철거합니다. 메모를 잃어버리면 (`p = NULL`을 해제 없이), 건물은 여전히 토지를 차지합니다 -- 이것이 메모리 누수입니다.

```c
#include <stdio.h>
#include <stdlib.h>  // malloc, free

int main(void) {
    // int 5개를 위한 메모리 할당
    int *arr = (int *)malloc(5 * sizeof(int));

    // 할당 실패 확인 (필수!)
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // 사용
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);  // 0 10 20 30 40
    }
    printf("\n");

    // 해제 (필수!)
    free(arr);
    arr = NULL;  // 댕글링 포인터 방지

    return 0;
}
```

### calloc - 클리어 할당(Clear Allocation)

```c
// calloc: 할당 + 0으로 초기화
int *arr = (int *)calloc(5, sizeof(int));
// arr[0] ~ arr[4] 모두 0으로 초기화됨

// malloc vs calloc
int *m = malloc(5 * sizeof(int));  // 초기화되지 않음 (쓰레기 값)
int *c = calloc(5, sizeof(int));   // 0으로 초기화됨
```

### realloc - 재할당(Re-allocation)

```c
int *arr = malloc(5 * sizeof(int));

// 크기 확장 (5 -> 10)
int *new_arr = realloc(arr, 10 * sizeof(int));
if (new_arr == NULL) {
    // 실패 시 원래 arr은 유효한 상태로 유지
    free(arr);
    return 1;
}
arr = new_arr;

// 크기 축소 (10 -> 3)
arr = realloc(arr, 3 * sizeof(int));

free(arr);
```

### realloc 작동 방식

```
+-----------------------------------------------------+
|  realloc(ptr, new_size)                             |
|                                                     |
|  1. 현재 위치에서 확장 가능한 경우:                 |
|     [기존 데이터][새 공간      ]                    |
|                                                     |
|  2. 확장 불가 -> 새 위치로 복사                     |
|     [원래 위치: 해제됨]                             |
|     [새 위치: 기존 데이터 복사됨][새 공간]          |
|                                                     |
|  3. 실패 시 -> NULL 반환 (원본 유지)               |
+-----------------------------------------------------+
```

---

## 단계 2: 동적 배열 구조체 설계

### 설계

```c
typedef struct {
    int *data;      // 실제 데이터 저장소
    int size;       // 현재 요소 수
    int capacity;   // 할당된 공간 크기
} DynamicArray;
```

### 작동 방식

```
초기 상태 (capacity=4, size=0):
+---+---+---+---+
|   |   |   |   |  data
+---+---+---+---+

3개 추가 후 (capacity=4, size=3):
+---+---+---+---+
| 1 | 2 | 3 |   |  data
+---+---+---+---+

5번째 항목 추가 -> 자동 확장! (capacity=8, size=5):
+---+---+---+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |   |   |   |  data
+---+---+---+---+---+---+---+---+
```

---

## 단계 3: 기본 구현

```c
// dynamic_array.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 4
#define GROWTH_FACTOR 2

// 동적 배열 구조체
typedef struct {
    int *data;
    int size;
    int capacity;
} DynamicArray;

// 함수 선언
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

// 생성
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

// 소멸
void da_destroy(DynamicArray *arr) {
    if (arr != NULL) {
        free(arr->data);
        free(arr);
    }
}

// 크기 변경 (내부 함수)
static int da_resize(DynamicArray *arr, int new_capacity) {
    int *new_data = (int *)realloc(arr->data, new_capacity * sizeof(int));
    if (new_data == NULL) {
        return -1;  // 실패
    }

    arr->data = new_data;
    arr->capacity = new_capacity;
    return 0;  // 성공
}

// 끝에 추가
int da_push(DynamicArray *arr, int value) {
    // 공간이 부족하면 확장
    if (arr->size >= arr->capacity) {
        if (da_resize(arr, arr->capacity * GROWTH_FACTOR) != 0) {
            return -1;
        }
    }

    arr->data[arr->size] = value;
    arr->size++;
    return 0;
}

// 끝에서 제거
int da_pop(DynamicArray *arr, int *value) {
    if (arr->size == 0) {
        return -1;  // 빈 배열
    }

    arr->size--;
    if (value != NULL) {
        *value = arr->data[arr->size];
    }

    // 너무 크면 축소 (선택 사항)
    if (arr->size > 0 && arr->size <= arr->capacity / 4) {
        da_resize(arr, arr->capacity / 2);
    }

    return 0;
}

// 인덱스로 값 가져오기
int da_get(DynamicArray *arr, int index, int *value) {
    if (index < 0 || index >= arr->size) {
        return -1;  // 범위 초과
    }

    *value = arr->data[index];
    return 0;
}

// 인덱스에 값 설정
int da_set(DynamicArray *arr, int index, int value) {
    if (index < 0 || index >= arr->size) {
        return -1;
    }

    arr->data[index] = value;
    return 0;
}

// 특정 위치에 삽입
int da_insert(DynamicArray *arr, int index, int value) {
    if (index < 0 || index > arr->size) {
        return -1;
    }

    // 공간 확보
    if (arr->size >= arr->capacity) {
        if (da_resize(arr, arr->capacity * GROWTH_FACTOR) != 0) {
            return -1;
        }
    }

    // 요소를 오른쪽으로 시프트
    for (int i = arr->size; i > index; i--) {
        arr->data[i] = arr->data[i - 1];
    }

    arr->data[index] = value;
    arr->size++;
    return 0;
}

// 특정 위치에서 제거
int da_remove(DynamicArray *arr, int index) {
    if (index < 0 || index >= arr->size) {
        return -1;
    }

    // 요소를 왼쪽으로 시프트
    for (int i = index; i < arr->size - 1; i++) {
        arr->data[i] = arr->data[i + 1];
    }

    arr->size--;
    return 0;
}

// 배열 출력
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

// 테스트
int main(void) {
    printf("=== 동적 배열 테스트 ===\n\n");

    // 생성
    DynamicArray *arr = da_create();
    if (arr == NULL) {
        printf("배열 생성 실패\n");
        return 1;
    }

    da_print(arr);

    // Push 테스트
    printf("\n[Push 테스트]\n");
    for (int i = 1; i <= 10; i++) {
        da_push(arr, i * 10);
        da_print(arr);
    }

    // Get/Set 테스트
    printf("\n[Get/Set 테스트]\n");
    int value;
    da_get(arr, 3, &value);
    printf("arr[3] = %d\n", value);

    da_set(arr, 3, 999);
    da_print(arr);

    // Insert 테스트
    printf("\n[Insert 테스트]\n");
    da_insert(arr, 0, -100);  // 맨 앞에 삽입
    da_print(arr);

    da_insert(arr, 5, -500);  // 중간에 삽입
    da_print(arr);

    // Remove 테스트
    printf("\n[Remove 테스트]\n");
    da_remove(arr, 0);  // 맨 앞에서 제거
    da_print(arr);

    // Pop 테스트
    printf("\n[Pop 테스트]\n");
    while (arr->size > 0) {
        da_pop(arr, &value);
        printf("Popped: %d, ", value);
        da_print(arr);
    }

    // 소멸
    da_destroy(arr);
    printf("\n배열 소멸됨\n");

    return 0;
}
```

---

## 단계 4: 제네릭 동적 배열 (void 포인터)

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
    size_t element_size;  // 하나의 요소 크기
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

    // 요소 복사
    void *dest = (char *)arr->data + (arr->size * arr->element_size);
    memcpy(dest, element, arr->element_size);
    arr->size++;
    return 0;
}

void* ga_get(GenericArray *arr, int index) {
    if (index < 0 || index >= arr->size) return NULL;
    return (char *)arr->data + (index * arr->element_size);
}

// 테스트
int main(void) {
    // int 배열
    printf("=== int 배열 ===\n");
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

    // double 배열
    printf("\n=== double 배열 ===\n");
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

    // 구조체 배열
    printf("\n=== 구조체 배열 ===\n");
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

## 예제 출력

```
=== 동적 배열 테스트 ===

DynamicArray(size=0, capacity=4): []

[Push 테스트]
DynamicArray(size=1, capacity=4): [10]
DynamicArray(size=2, capacity=4): [10, 20]
DynamicArray(size=3, capacity=4): [10, 20, 30]
DynamicArray(size=4, capacity=4): [10, 20, 30, 40]
DynamicArray(size=5, capacity=8): [10, 20, 30, 40, 50]  <- 자동 확장!
DynamicArray(size=6, capacity=8): [10, 20, 30, 40, 50, 60]
...
```

---

## 요약

| 함수 | 설명 |
|------|------|
| `malloc(size)` | size 바이트 할당 |
| `calloc(n, size)` | n개 요소 할당, 0으로 초기화 |
| `realloc(ptr, size)` | 크기 변경 |
| `free(ptr)` | 메모리 해제 |
| `memcpy(dest, src, n)` | n바이트 복사 |

### 메모리 관리 규칙

1. **할당 후 NULL 검사** 필수
2. **사용 후 free()** 필수
3. **free 후 NULL 대입** 권장 (댕글링 포인터 방지)
4. **이중 해제 금지**

---

## 연습문제

1. **da_find**: 값을 검색하여 인덱스 반환

2. **da_reverse**: 배열 뒤집기

3. **da_sort**: 정렬 기능 추가 (qsort 활용)

4. **문자열 동적 배열**: `char*` 배열 구현

5. **축소 정책**: size가 25% 이하로 떨어지면 capacity를 절반으로 줄이는 축소 전략을 구현하고, 축소하지 않는 경우와 벤치마크하세요

---

## 다음 단계

[프로젝트: 연결 리스트](./05_Project_Linked_List.md) -> 포인터의 꽃, 연결 리스트를 배워봅시다!
