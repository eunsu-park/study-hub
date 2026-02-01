# 프로젝트 4: 동적 배열 (Dynamic Array)

## 학습 목표

이 프로젝트를 통해 배우는 내용:
- 동적 메모리 할당 (`malloc`, `calloc`, `realloc`, `free`)
- 메모리 누수 방지
- 크기가 자동으로 늘어나는 배열 구현
- Python의 리스트, JavaScript의 배열과 유사한 자료구조

---

## 동적 메모리가 필요한 이유

### 정적 배열의 한계

```c
// 정적 배열: 크기가 고정됨
int arr[100];  // 컴파일 시 크기 결정

// 문제 1: 크기를 미리 알아야 함
// 문제 2: 크기 변경 불가
// 문제 3: 사용하지 않는 공간 낭비
```

### 동적 배열의 장점

```c
// 동적 배열: 실행 중 크기 결정 및 변경 가능
int *arr = malloc(n * sizeof(int));  // 실행 시 크기 결정
arr = realloc(arr, m * sizeof(int)); // 크기 변경 가능!
```

---

## 1단계: 동적 메모리 함수 이해

### malloc - Memory Allocation

```c
#include <stdio.h>
#include <stdlib.h>  // malloc, free

int main(void) {
    // int 5개 크기의 메모리 할당
    int *arr = (int *)malloc(5 * sizeof(int));

    // 할당 실패 체크 (필수!)
    if (arr == NULL) {
        printf("메모리 할당 실패\n");
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
    arr = NULL;  // dangling pointer 방지

    return 0;
}
```

### calloc - Clear Allocation

```c
// calloc: 할당 + 0으로 초기화
int *arr = (int *)calloc(5, sizeof(int));
// arr[0] ~ arr[4] 모두 0으로 초기화됨

// malloc vs calloc
int *m = malloc(5 * sizeof(int));  // 초기화 안 됨 (쓰레기 값)
int *c = calloc(5, sizeof(int));   // 0으로 초기화
```

### realloc - Re-allocation

```c
int *arr = malloc(5 * sizeof(int));

// 크기 확장 (5 → 10)
int *new_arr = realloc(arr, 10 * sizeof(int));
if (new_arr == NULL) {
    // 실패 시 원본 arr은 그대로 유지됨
    free(arr);
    return 1;
}
arr = new_arr;

// 크기 축소 (10 → 3)
arr = realloc(arr, 3 * sizeof(int));

free(arr);
```

### realloc 동작 방식

```
┌─────────────────────────────────────────────────────┐
│  realloc(ptr, new_size)                             │
│                                                     │
│  1. 현재 위치에서 확장 가능하면 → 확장              │
│     [기존 데이터][새 공간      ]                    │
│                                                     │
│  2. 확장 불가능하면 → 새 위치로 복사                │
│     [기존 위치: 해제됨]                             │
│     [새 위치: 기존 데이터 복사][새 공간]            │
│                                                     │
│  3. 실패하면 → NULL 반환 (원본 유지)               │
└─────────────────────────────────────────────────────┘
```

---

## 2단계: 동적 배열 구조체 설계

### 설계

```c
typedef struct {
    int *data;      // 실제 데이터 저장
    int size;       // 현재 요소 개수
    int capacity;   // 할당된 공간 크기
} DynamicArray;
```

### 동작 원리

```
초기 상태 (capacity=4, size=0):
┌───┬───┬───┬───┐
│   │   │   │   │  data
└───┴───┴───┴───┘

3개 추가 후 (capacity=4, size=3):
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │   │  data
└───┴───┴───┴───┘

5번째 추가 시 → 자동 확장! (capacity=8, size=5):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │   │   │   │  data
└───┴───┴───┴───┴───┴───┴───┴───┘
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

// 해제
void da_destroy(DynamicArray *arr) {
    if (arr != NULL) {
        free(arr->data);
        free(arr);
    }
}

// 크기 조정 (내부 함수)
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

    // 공간이 너무 크면 축소 (선택적)
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

    // 뒤의 요소들을 한 칸씩 이동
    for (int i = arr->size; i > index; i--) {
        arr->data[i] = arr->data[i - 1];
    }

    arr->data[index] = value;
    arr->size++;
    return 0;
}

// 특정 위치 제거
int da_remove(DynamicArray *arr, int index) {
    if (index < 0 || index >= arr->size) {
        return -1;
    }

    // 뒤의 요소들을 한 칸씩 앞으로
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

    // push 테스트
    printf("\n[Push 테스트]\n");
    for (int i = 1; i <= 10; i++) {
        da_push(arr, i * 10);
        da_print(arr);
    }

    // get/set 테스트
    printf("\n[Get/Set 테스트]\n");
    int value;
    da_get(arr, 3, &value);
    printf("arr[3] = %d\n", value);

    da_set(arr, 3, 999);
    da_print(arr);

    // insert 테스트
    printf("\n[Insert 테스트]\n");
    da_insert(arr, 0, -100);  // 맨 앞에 삽입
    da_print(arr);

    da_insert(arr, 5, -500);  // 중간에 삽입
    da_print(arr);

    // remove 테스트
    printf("\n[Remove 테스트]\n");
    da_remove(arr, 0);  // 맨 앞 제거
    da_print(arr);

    // pop 테스트
    printf("\n[Pop 테스트]\n");
    while (arr->size > 0) {
        da_pop(arr, &value);
        printf("Popped: %d, ", value);
        da_print(arr);
    }

    // 해제
    da_destroy(arr);
    printf("\n배열 해제 완료\n");

    return 0;
}
```

---

## 4단계: 제네릭 동적 배열 (void 포인터)

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
    size_t element_size;  // 요소 하나의 크기
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

## 실행 결과

```
=== 동적 배열 테스트 ===

DynamicArray(size=0, capacity=4): []

[Push 테스트]
DynamicArray(size=1, capacity=4): [10]
DynamicArray(size=2, capacity=4): [10, 20]
DynamicArray(size=3, capacity=4): [10, 20, 30]
DynamicArray(size=4, capacity=4): [10, 20, 30, 40]
DynamicArray(size=5, capacity=8): [10, 20, 30, 40, 50]  ← 자동 확장!
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
3. **free 후 NULL 할당** 권장 (dangling pointer 방지)
4. **이중 free 금지**

---

## 연습 문제

1. **da_find**: 값을 검색하여 인덱스 반환

2. **da_reverse**: 배열 뒤집기

3. **da_sort**: 정렬 기능 추가 (qsort 활용)

4. **문자열 동적 배열**: `char*` 배열 구현

---

## 다음 단계

[07_Project_Linked_List.md](./07_Project_Linked_List.md) → 포인터의 꽃, 연결 리스트를 배워봅시다!
