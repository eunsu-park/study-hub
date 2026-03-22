# 동적 메모리

**이전**: [구조체와 공용체](./08_Structs_and_Unions.md) | **다음**: [파일 입출력](./10_File_IO.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `malloc`과 `calloc`을 사용하여 힙 메모리를 할당하고 각각 적합한 경우를 설명하기
2. `realloc`으로 할당 크기를 변경하고 반환된 포인터를 올바르게 처리하기
3. `free`로 메모리를 해제하고 댕글링 참조를 방지하기 위해 포인터를 `NULL`로 설정하기
4. 일반적인 메모리 오류(누수, 이중 해제, 해제 후 사용) 식별 및 방지하기
5. 프로그램에서 일관된 할당-검사-사용-해제 패턴 적용하기

---

스택 변수는 간단하고 빠르지만, 두 가지 중요한 제한이 있습니다: 크기를 컴파일 타임에 알아야 하고, 함수가 반환되면 소멸됩니다. 동적 메모리 할당을 사용하면 런타임에 메모리를 요청하고, 수명을 제어하며, 필요에 따라 늘어나고 줄어드는 데이터 구조를 구축할 수 있습니다. 대신 **여러분이** 그 메모리를 해제할 책임을 지게 됩니다.

## 1. 스택 vs 힙

| 속성 | 스택 (Stack) | 힙 (Heap) |
|----------|-------|------|
| 할당 | 자동 (함수 진입 시) | 수동 (`malloc`, `calloc`) |
| 해제 | 자동 (함수 종료 시) | 수동 (`free`) |
| 크기 | 컴파일 타임에 고정 | 런타임에 결정 |
| 일반적 한도 | 1-8 MB (OS 기본값) | 사용 가능한 RAM으로 제한 |
| 속도 | 매우 빠름 (포인터 이동) | 느림 (부기 오버헤드) |
| 단편화 | 없음 | 시간이 지나면 발생 가능 |

**힙을 사용해야 하는 경우**:
- 런타임까지 크기를 모를 때 (사용자 입력, 파일 데이터)
- 데이터가 생성 함수보다 오래 살아야 할 때
- 데이터가 스택에 담기에 너무 클 때 (큰 배열, 버퍼)

---

## 2. malloc

`malloc` (memory allocate)은 힙에서 초기화되지 않은 바이트 블록을 요청합니다. 블록에 대한 `void *` 포인터를 반환하며, 할당에 실패하면 `NULL`을 반환합니다.

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n;
    printf("How many numbers? ");
    scanf("%d", &n);

    /* Allocate array of n ints */
    int *arr = malloc(n * sizeof(int));
    if (arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Use the memory */
    for (int i = 0; i < n; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < n; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    /* Release the memory */
    free(arr);
    arr = NULL;

    return 0;
}
```

### 핵심 포인트

- 항상 가리키는 타입에 `sizeof`를 사용하세요: `malloc(n * sizeof(int))` 또는 `arr`의 타입을 변경해도 올바르게 유지되는 더 안전한 관용구 `malloc(n * sizeof(*arr))`.
- **항상 반환 값을 확인하세요**. `malloc`은 실패 시 `NULL`을 반환합니다.
- C에서 `malloc`의 반환 값을 캐스팅하는 것은 불필요합니다(많은 스타일 가이드에서 권장하지 않습니다). `void *`는 모든 포인터 타입으로 암시적으로 변환되기 때문입니다. C++에서는 캐스팅이 필요하지만, C++에서는 `malloc`을 사용하면 안 됩니다.

```c
/* Preferred: sizeof applied to the variable, not the type */
int *data = malloc(count * sizeof(*data));
```

---

## 3. calloc

`calloc` (clear allocate)은 `malloc`처럼 작동하지만 두 개의 인수(개수와 요소 크기)를 받으며, 메모리를 **0으로 초기화**합니다.

```c
#include <stdlib.h>

int main(void) {
    /* Allocate 100 ints, all initialized to 0 */
    int *arr = calloc(100, sizeof(int));
    if (arr == NULL) {
        return 1;
    }

    /* arr[0] through arr[99] are all 0 */

    free(arr);
    arr = NULL;
    return 0;
}
```

| 함수 | 인수 | 초기화 여부 | 사용 시기 |
|----------|-----------|-------------|----------|
| `malloc` | 전체 바이트 수 | 아니오 (쓰레기 값) | 모든 바이트에 즉시 쓸 때 |
| `calloc` | 개수, 요소 크기 | 예 (0으로) | 0으로 초기화된 메모리가 필요할 때 |

`calloc`에는 안전성 이점도 있습니다: `count * size`에서 정수 오버플로를 내부적으로 검사하지만, `malloc(count * size)`는 조용히 오버플로되어 메모리를 너무 적게 할당할 수 있습니다.

---

## 4. realloc

`realloc`은 이전에 할당된 블록의 크기를 변경합니다. 제자리에서 확장할 공간이 충분하지 않으면 데이터를 새 위치로 이동시킬 수 있습니다.

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    size_t capacity = 4;
    size_t size = 0;
    int *arr = malloc(capacity * sizeof(*arr));
    if (arr == NULL) return 1;

    /* Simulate adding elements */
    for (int i = 0; i < 20; i++) {
        if (size == capacity) {
            capacity *= 2;
            int *temp = realloc(arr, capacity * sizeof(*temp));
            if (temp == NULL) {
                fprintf(stderr, "realloc failed\n");
                free(arr);   /* free the original block */
                return 1;
            }
            arr = temp;
            printf("Grew to capacity %zu\n", capacity);
        }
        arr[size++] = i * 10;
    }

    for (size_t i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);
    arr = NULL;
    return 0;
}
```

### 중요한 규칙: 절대 이렇게 하지 마세요

```c
arr = realloc(arr, new_size);   /* DANGEROUS */
```

`realloc`이 실패하여 `NULL`을 반환하면, 원래 블록에 대한 유일한 포인터를 덮어쓴 것입니다 — 그 메모리는 이제 **누수**됩니다. 항상 임시 포인터를 사용하세요:

```c
int *temp = realloc(arr, new_size);
if (temp == NULL) {
    /* handle error — arr is still valid */
} else {
    arr = temp;
}
```

### realloc 특수 케이스

| 호출 | 동작 |
|------|----------|
| `realloc(NULL, size)` | `malloc(size)`과 동일 |
| `realloc(ptr, 0)` | 구현 정의 (해제하거나 작은 블록 반환 가능) — 피하세요 |
| `realloc(ptr, smaller)` | 제자리에서 축소하거나 새 포인터를 반환할 수 있음 |

---

## 5. free

`free`는 동적으로 할당된 메모리 블록을 시스템에 반환합니다. 해제 후 실수로 재사용하는 것을 방지하기 위해 포인터를 `NULL`로 설정하세요.

```c
int *p = malloc(sizeof(int));
*p = 42;

free(p);
p = NULL;   /* good practice — prevents dangling pointer use */
```

**규칙**:
- `malloc`, `calloc`, 또는 `realloc`이 반환한 메모리만 해제하세요.
- 스택 변수, 전역 변수, 문자열 리터럴은 해제하지 마세요.
- 같은 포인터를 두 번 해제하지 마세요.
- `free(NULL)`은 안전하며 아무것도 하지 않습니다 — 이것이 해제된 포인터를 `NULL`로 설정하는 것이 도움이 되는 이유입니다.

---

## 6. 일반적인 메모리 오류

### 메모리 누수 (Memory Leak)

할당된 메모리가 해제되지 않는 경우. 프로그램의 메모리 사용량이 한없이 증가합니다.

```c
void process(void) {
    char *buf = malloc(1024);
    if (some_condition) {
        return;   /* BUG: buf is leaked on this path */
    }
    /* ... use buf ... */
    free(buf);
}
```

### 이중 해제 (Double Free)

같은 블록을 두 번 해제하면 메모리 할당자의 내부 데이터 구조가 손상됩니다.

```c
int *p = malloc(sizeof(int));
free(p);
free(p);   /* UNDEFINED BEHAVIOR — heap corruption */
```

**수정**: 해제 후 포인터를 `NULL`로 설정하세요. `free(NULL)`은 아무 동작도 하지 않습니다.

### 해제 후 사용 (Use-After-Free)

메모리가 해제된 후 접근하는 경우. 그 메모리는 다른 용도로 재할당되었을 수 있습니다.

```c
int *p = malloc(sizeof(int));
*p = 42;
free(p);
printf("%d\n", *p);   /* UNDEFINED BEHAVIOR */
```

### 버퍼 오버플로 (Buffer Overflow)

할당된 블록의 끝을 넘어서 쓰는 경우.

```c
int *arr = malloc(5 * sizeof(int));
arr[5] = 99;   /* UNDEFINED BEHAVIOR — out of bounds */
```

| 오류 | 원인 | 증상 | 방지 |
|-------|-------|---------|------------|
| 누수 | `free` 누락 | 메모리 사용량 증가 | 모든 경로에서 해제 |
| 이중 해제 | `free` 두 번 호출 | 크래시 / 손상 | 해제 후 NULL 설정 |
| 해제 후 사용 | 해제된 메모리 접근 | 쓰레기 데이터 / 크래시 | NULL 설정, 사용하지 않기 |
| 오버플로 | 할당 범위를 넘어서 쓰기 | 손상 / 크래시 | 크기를 주의 깊게 추적 |

---

## 7. 메모리 관리 패턴

### 소유권 (Ownership)

명확한 소유권을 확립하세요: 할당하는 함수(또는 모듈)가 해제할 책임이 있습니다.

```c
/* Caller owns the returned memory */
char *create_greeting(const char *name) {
    size_t len = strlen("Hello, ") + strlen(name) + 2;
    char *buf = malloc(len);
    if (buf == NULL) return NULL;
    snprintf(buf, len, "Hello, %s!", name);
    return buf;   /* caller must free */
}

int main(void) {
    char *msg = create_greeting("Alice");
    if (msg) {
        printf("%s\n", msg);
        free(msg);   /* caller frees */
    }
    return 0;
}
```

### goto 정리 패턴 (Goto Cleanup Pattern)

함수에서 서로 의존하는 여러 할당을 할 때, 어떤 오류에서든 모두 해제되도록 `goto cleanup`을 사용합니다:

```c
#include <stdio.h>
#include <stdlib.h>

int process_data(size_t n) {
    int *buffer = NULL;
    char *name = NULL;
    int result = -1;

    buffer = malloc(n * sizeof(*buffer));
    if (buffer == NULL) goto cleanup;

    name = malloc(256);
    if (name == NULL) goto cleanup;

    /* ... do work with buffer and name ... */

    result = 0;  /* success */

cleanup:
    free(name);     /* free(NULL) is safe */
    free(buffer);
    return result;
}
```

이 패턴은 리눅스 커널과 기타 시스템 수준 C 코드에서 널리 사용됩니다.

---

## 8. 구조체의 동적 배열

`malloc`, 구조체, 적절한 정리를 결합하는 실용적인 예시:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[50];
    int age;
    double score;
} Student;

Student *create_students(size_t count) {
    Student *students = calloc(count, sizeof(Student));
    return students;  /* NULL if allocation failed */
}

void print_students(const Student *students, size_t count) {
    printf("%-20s %5s %7s\n", "Name", "Age", "Score");
    printf("%-20s %5s %7s\n", "----", "---", "-----");
    for (size_t i = 0; i < count; i++) {
        printf("%-20s %5d %7.2f\n",
               students[i].name,
               students[i].age,
               students[i].score);
    }
}

int main(void) {
    size_t n = 3;
    Student *roster = create_students(n);
    if (roster == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Populate */
    strcpy(roster[0].name, "Alice");
    roster[0].age = 20;
    roster[0].score = 95.5;

    strcpy(roster[1].name, "Bob");
    roster[1].age = 22;
    roster[1].score = 88.0;

    strcpy(roster[2].name, "Charlie");
    roster[2].age = 21;
    roster[2].score = 92.3;

    print_students(roster, n);

    /* Grow the array to add one more student */
    n = 4;
    Student *temp = realloc(roster, n * sizeof(Student));
    if (temp == NULL) {
        fprintf(stderr, "realloc failed\n");
        free(roster);
        return 1;
    }
    roster = temp;

    strcpy(roster[3].name, "Diana");
    roster[3].age = 23;
    roster[3].score = 97.1;

    printf("\nAfter adding Diana:\n");
    print_students(roster, n);

    free(roster);
    roster = NULL;
    return 0;
}
```

출력:

```
Name                   Age   Score
----                   ---   -----
Alice                   20   95.50
Bob                     22   88.00
Charlie                 21   92.30

After adding Diana:
Name                   Age   Score
----                   ---   -----
Alice                   20   95.50
Bob                     22   88.00
Charlie                 21   92.30
Diana                   23   97.10
```

---

## 연습문제

**연습문제 1 — 동적 정수 배열**: 사용자가 -1을 입력할 때까지 정수를 읽는 프로그램을 작성하세요. 동적으로 늘어나는 배열에 저장하세요 (용량 4로 시작, 가득 차면 2배). 모든 값을 출력하고 메모리를 해제하세요.

**연습문제 2 — 문자열 복제기**: 메모리를 할당하고, 문자열을 복사한 후, 새 문자열을 반환하는 `char *my_strdup(const char *s)` 함수를 작성하세요. 호출자가 해제할 책임이 있습니다. 여러 문자열로 테스트하세요.

**연습문제 3 — 행렬 할당**: 2D 행렬(`int **`)을 동적으로 할당하고, 구구단(행 * 열)으로 채우고, 출력하고, 모든 메모리를 해제하는 함수를 작성하세요. 차원은 런타임에 제공되어야 합니다.

**연습문제 4 — 파일에서 구조체 배열**: 사용자에게 학생 수를 묻고, `Student` 구조체 배열을 할당하고, 사용자 입력으로 채우고, 가장 높은 점수의 학생을 찾고, 모든 메모리를 해제하는 프로그램을 작성하세요.

**연습문제 5 — 메모리 오류 찾기**: 다음 프로그램에는 세 가지 메모리 오류가 있습니다. 각각을 식별하고 수정하세요:

```c
#include <stdlib.h>
#include <string.h>

int main(void) {
    int *a = malloc(5 * sizeof(int));
    a[5] = 100;

    char *s = malloc(10);
    strcpy(s, "Hello, World!");

    int *b = malloc(sizeof(int));
    free(b);
    *b = 42;

    free(a);
    free(s);
    return 0;
}
```

---

## 다음 단계

이제 C에서 메모리 할당과 해제를 완전히 제어할 수 있습니다. 다음 레슨 [파일 입출력](./10_File_IO.md)에서는 파일에 데이터를 읽고 쓰는 방법을 배웁니다 — 동적 메모리와 파일 작업을 결합하여 실행 간에 데이터를 유지하는 프로그램을 구축합니다.
