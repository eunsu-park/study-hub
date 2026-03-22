# 고급 C 포인터

**이전**: [C 고급](./00_Overview.md) | **다음**: [고급 메모리 관리](./02_Advanced_Memory_Management.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 포인터 산술을 수행하여 배열을 순회하고 요소 간 거리를 계산할 수 있다
2. 포인터 배열(`int *arr[]`)과 배열 포인터(`int (*p)[N]`)를 구분할 수 있다
3. 이중 포인터를 사용하여 함수 내에서 호출자의 포인터를 수정할 수 있다
4. 함수 포인터를 선언, 할당, 호출할 수 있으며, `typedef`와 `qsort`를 활용할 수 있다
5. `const`를 포인터와 올바르게 사용하여 함수 인터페이스에서 읽기 전용 의도를 표현할 수 있다
6. 포인터 기반 할당을 사용하여 일반적인 자료 구조(연결 리스트, 동적 2D 배열)를 구현할 수 있다
7. `<stdarg.h>`를 사용한 가변 인자 함수를 작성하고, 최적화를 위한 `restrict` 한정자를 적용할 수 있다

---

포인터(Pointer)는 C에서 가장 강력하면서도 가장 위험한 기능입니다. 메모리에 직접 접근할 수 있어 효율적인 자료 구조, 제로 카피 인터페이스, 하드웨어 제어가 가능하지만, 잘못된 역참조 하나가 프로그램을 충돌시키거나 데이터를 조용히 손상시킬 수 있습니다. 이 레슨은 기초를 넘어 자신감 있는 C 프로그래머와 조심스러운 C 프로그래머를 구분하는 깊고 실용적인 포인터 이해를 구축합니다.

**난이도**: 고급

---

## 1. 포인터 산술(Pointer Arithmetic)

### 포인터 증가/감소

포인터에 1을 더하면 **가리키는 타입의 크기**만큼 주소가 증가합니다.

```c
int arr[] = {10, 20, 30, 40, 50};
int *p = arr;

printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[0] = 10
p++;
printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[1] = 20
p += 2;
printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[3] = 40
```

### 포인터를 이용한 배열 순회

```c
int arr[] = {1, 2, 3, 4, 5};
int n = sizeof(arr) / sizeof(arr[0]);

// 방법 1: 인덱스 사용
for (int i = 0; i < n; i++) {
    printf("%d ", arr[i]);
}

// 방법 2: 포인터 산술
for (int *p = arr; p < arr + n; p++) {
    printf("%d ", *p);
}

// 방법 3: 포인터와 인덱스 혼합
int *p = arr;
for (int i = 0; i < n; i++) {
    printf("%d ", *(p + i));  // p[i]와 동일
}
```

### 포인터 뺄셈

두 포인터 사이의 **요소 수**를 반환합니다.

```c
int arr[] = {10, 20, 30, 40, 50};
int *start = &arr[0];
int *end = &arr[4];

ptrdiff_t diff = end - start;  // 4 (바이트가 아닌 요소 수)
printf("Element count: %td\n", diff);
```

### 포인터 비교

```c
int arr[] = {1, 2, 3, 4, 5};
int *p1 = &arr[1];
int *p2 = &arr[3];

if (p1 < p2) {
    printf("p1 is at a lower address\n");  // 이 줄이 출력됨
}

// 같은 배열 내의 포인터만 비교 가능
// 서로 다른 배열의 포인터를 비교하면 정의되지 않은 동작
```

---

## 2. 배열과 포인터

### 배열 인덱싱의 진실

`arr[i]`는 `*(arr + i)`의 문법적 설탕(syntactic sugar)입니다.

```c
int arr[] = {10, 20, 30};

// 모두 동일
printf("%d\n", arr[1]);       // 20
printf("%d\n", *(arr + 1));   // 20
printf("%d\n", *(1 + arr));   // 20
printf("%d\n", 1[arr]);       // 20 (이상하지만 합법!)
```

### 포인터 배열 vs 배열 포인터

```c
// 포인터 배열: 포인터들의 배열
int *ptr_arr[3];  // int* 3개를 담는 배열

int a = 1, b = 2, c = 3;
ptr_arr[0] = &a;
ptr_arr[1] = &b;
ptr_arr[2] = &c;

// 배열 포인터: 배열을 가리키는 포인터
int (*arr_ptr)[4];  // int[4] 배열을 가리키는 포인터

int arr[4] = {1, 2, 3, 4};
arr_ptr = &arr;

printf("%d\n", (*arr_ptr)[2]);  // 3
```

**선언 읽는 법**:
```c
int *ptr_arr[3];   // [3] 먼저 -> ptr_arr은 크기 3의 배열
                   // * 다음 -> 요소는 포인터
                   // int -> int에 대한 포인터

int (*arr_ptr)[4]; // * 먼저 (괄호) -> arr_ptr은 포인터
                   // [4] 다음 -> 크기 4의 배열을 가리킴
                   // int -> int 배열
```

### 2D 배열과 포인터 관계

```c
int matrix[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

// 요소 접근
printf("%d\n", matrix[1][2]);           // 7
printf("%d\n", *(*(matrix + 1) + 2));   // 7

// matrix는 int[4] 배열에 대한 포인터로 변환됨
// matrix[i]는 행 i의 첫 번째 요소 주소
```

---

## 3. 다중 간접 참조(Multiple Indirection)

### 이중 포인터(Pointer to Pointer)

```c
int x = 42;
int *p = &x;
int **pp = &p;

printf("x:   %d\n", x);       // 42
printf("*p:  %d\n", *p);      // 42
printf("**pp: %d\n", **pp);   // 42

// 주소 관계
printf("&x:  %p\n", (void*)&x);   // x의 주소
printf("p:   %p\n", (void*)p);    // x의 주소
printf("&p:  %p\n", (void*)&p);   // p의 주소
printf("pp:  %p\n", (void*)pp);   // p의 주소
```

### 이중 포인터 활용: 함수에서 포인터 수정

```c
#include <stdio.h>
#include <stdlib.h>

// 잘못된 방법: 포인터의 복사본이 전달됨
void allocate_wrong(int *p, int size) {
    p = malloc(size * sizeof(int));  // 로컬 p만 수정됨
    // 호출자의 포인터는 변경되지 않음
}

// 올바른 방법: 이중 포인터 사용
void allocate_correct(int **pp, int size) {
    *pp = malloc(size * sizeof(int));  // 호출자의 포인터를 수정
}

int main(void) {
    int *arr = NULL;

    allocate_wrong(arr, 5);
    printf("wrong: %p\n", (void*)arr);  // NULL

    allocate_correct(&arr, 5);
    printf("correct: %p\n", (void*)arr);  // 유효한 주소

    free(arr);
    return 0;
}
```

### 동적 2D 배열

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int rows = 3, cols = 4;

    // 방법 1: 포인터 배열 (행별 개별 할당)
    int **matrix = malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(int));
    }

    // 사용
    matrix[1][2] = 42;
    printf("%d\n", matrix[1][2]);

    // 해제 (역순으로!)
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);

    // 방법 2: 연속 메모리 할당 (캐시 효율적)
    int *flat = malloc(rows * cols * sizeof(int));
    // flat[i * cols + j]로 접근
    flat[1 * cols + 2] = 42;
    free(flat);

    return 0;
}
```

### 문자열 배열 (명령줄 인자)

```c
#include <stdio.h>

int main(int argc, char *argv[]) {
    // argv는 char*의 배열
    // argv[0]: 프로그램 이름
    // argv[1] ~ argv[argc-1]: 인자들

    printf("Argument count: %d\n", argc);

    for (int i = 0; i < argc; i++) {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    return 0;
}
```

```c
// 문자열 배열을 직접 생성
char *fruits[] = {"apple", "banana", "cherry"};
int n = sizeof(fruits) / sizeof(fruits[0]);

for (int i = 0; i < n; i++) {
    printf("%s\n", fruits[i]);
}
```

---

## 4. 함수 포인터(Function Pointers)

### 기본 선언과 사용

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int mul(int a, int b) { return a * b; }

int main(void) {
    // 함수 포인터 선언
    int (*fp)(int, int);

    // 함수 주소 할당
    fp = add;  // 또는 fp = &add;
    printf("add: %d\n", fp(3, 4));  // 7

    fp = sub;
    printf("sub: %d\n", fp(3, 4));  // -1

    fp = mul;
    printf("mul: %d\n", fp(3, 4));  // 12

    return 0;
}
```

### typedef로 가독성 향상

```c
// 함수 포인터 타입 정의
typedef int (*Operation)(int, int);

int add(int a, int b) { return a + b; }

int main(void) {
    Operation op = add;
    printf("%d\n", op(5, 3));  // 8

    // 함수 포인터 배열
    Operation ops[] = {add, sub, mul};
    for (int i = 0; i < 3; i++) {
        printf("%d\n", ops[i](10, 3));
    }

    return 0;
}
```

### 콜백 함수(Callback Functions)

```c
#include <stdio.h>

// 콜백 타입 정의
typedef void (*Callback)(int);

void process_array(int *arr, int size, Callback cb) {
    for (int i = 0; i < size; i++) {
        cb(arr[i]);
    }
}

void print_value(int x) {
    printf("%d ", x);
}

void print_double(int x) {
    printf("%d ", x * 2);
}

int main(void) {
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("원본: ");
    process_array(arr, n, print_value);
    printf("\n");

    printf("2배: ");
    process_array(arr, n, print_double);
    printf("\n");

    return 0;
}
```

### qsort 사용법

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 비교 함수: 오름차순
int compare_int_asc(const void *a, const void *b) {
    return *(int*)a - *(int*)b;
}

// 비교 함수: 내림차순
int compare_int_desc(const void *a, const void *b) {
    return *(int*)b - *(int*)a;
}

// 문자열 비교
int compare_str(const void *a, const void *b) {
    return strcmp(*(char**)a, *(char**)b);
}

int main(void) {
    // 정수 정렬
    int nums[] = {3, 1, 4, 1, 5, 9, 2, 6};
    int n = sizeof(nums) / sizeof(nums[0]);

    qsort(nums, n, sizeof(int), compare_int_asc);

    for (int i = 0; i < n; i++) {
        printf("%d ", nums[i]);
    }
    printf("\n");  // 1 1 2 3 4 5 6 9

    // 문자열 정렬
    char *words[] = {"banana", "apple", "cherry"};
    int wn = sizeof(words) / sizeof(words[0]);

    qsort(words, wn, sizeof(char*), compare_str);

    for (int i = 0; i < wn; i++) {
        printf("%s ", words[i]);
    }
    printf("\n");  // apple banana cherry

    return 0;
}
```

---

## 5. void 포인터와 제네릭 프로그래밍

어떤 타입이든 가리킬 수 있는 범용 포인터입니다.

```c
void *generic;

int x = 42;
double d = 3.14;
char c = 'A';

generic = &x;  // OK
generic = &d;  // OK
generic = &c;  // OK

// 역참조를 위해 캐스팅 필요
printf("%d\n", *(int*)generic);  // 캐스팅 후 역참조
```

**void 포인터 용도**:
- `malloc()`의 반환 타입
- 제네릭 함수 작성 (예: `qsort`, `memcpy`)
- C에서 다형성 인터페이스 구현

### 제네릭 스왑 함수

```c
#include <stdio.h>
#include <string.h>

void generic_swap(void *a, void *b, size_t size) {
    unsigned char temp[size];  // VLA를 임시 버퍼로 사용
    memcpy(temp, a, size);
    memcpy(a, b, size);
    memcpy(b, temp, size);
}

int main(void) {
    int x = 10, y = 20;
    generic_swap(&x, &y, sizeof(int));
    printf("x=%d, y=%d\n", x, y);  // x=20, y=10

    double a = 1.5, b = 2.5;
    generic_swap(&a, &b, sizeof(double));
    printf("a=%.1f, b=%.1f\n", a, b);  // a=2.5, b=1.5

    return 0;
}
```

---

## 6. const와 포인터

### 네 가지 조합

```c
int x = 10;
int y = 20;

// 1. 일반 포인터
int *p1 = &x;
*p1 = 30;   // OK: 값 수정 가능
p1 = &y;    // OK: 다른 주소를 가리킬 수 있음

// 2. const int* (const int에 대한 포인터)
// = int const *
const int *p2 = &x;
// *p2 = 30;  // 오류: 값 수정 불가
p2 = &y;      // OK: 다른 주소를 가리킬 수 있음

// 3. int* const (int에 대한 const 포인터)
int *const p3 = &x;
*p3 = 30;     // OK: 값 수정 가능
// p3 = &y;   // 오류: 다른 주소를 가리킬 수 없음

// 4. const int* const (const int에 대한 const 포인터)
const int *const p4 = &x;
// *p4 = 30;  // 오류: 값 수정 불가
// p4 = &y;   // 오류: 다른 주소를 가리킬 수 없음
```

### 읽는 방법

오른쪽에서 왼쪽으로 읽습니다:

```c
const int *p;      // p는 포인터, const int를 가리킴
int *const p;      // p는 const 포인터, int를 가리킴
const int *const p; // p는 const 포인터, const int를 가리킴
```

### 함수 매개변수에서의 const

```c
// 입력 전용: 값이 수정되지 않음을 나타냄
void print_array(const int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
        // arr[i] = 0;  // 컴파일 오류!
    }
}

// 문자열은 항상 const char*로 받기
void print_str(const char *str) {
    while (*str) {
        putchar(*str++);
    }
}
```

---

## 7. 자기 참조 구조체(Self-referential Structures)

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;  // 자기 자신을 가리키는 포인터
} Node;

// 노드 생성
Node *create_node(int data) {
    Node *node = malloc(sizeof(Node));
    if (node) {
        node->data = data;
        node->next = NULL;
    }
    return node;
}

// 앞에 추가
void push_front(Node **head, int data) {
    Node *new_node = create_node(data);
    if (new_node) {
        new_node->next = *head;
        *head = new_node;
    }
}

// 출력
void print_list(Node *head) {
    while (head) {
        printf("%d -> ", head->data);
        head = head->next;
    }
    printf("NULL\n");
}

// 전체 해제
void free_list(Node *head) {
    while (head) {
        Node *temp = head;
        head = head->next;
        free(temp);
    }
}

int main(void) {
    Node *list = NULL;

    push_front(&list, 3);
    push_front(&list, 2);
    push_front(&list, 1);

    print_list(list);  // 1 -> 2 -> 3 -> NULL

    free_list(list);
    return 0;
}
```

---

## 8. 가변 인자 함수(Variadic Functions)와 `restrict` 한정자

### `<stdarg.h>`를 사용한 가변 인자 함수

C는 `<stdarg.h>` 헤더를 통해 가변 개수의 인자를 받는 함수를 지원합니다. `printf`, `scanf` 등의 함수가 내부적으로 이 방식을 사용합니다.

```c
#include <stdio.h>
#include <stdarg.h>

/*
 * va_list  - 인자 목록을 순회하는 데 필요한 상태를 저장하는 타입
 * va_start - va_list를 첫 번째 가변 인자를 가리키도록 초기화
 * va_arg   - 다음 인자를 가져오고 내부 포인터를 전진
 * va_end   - 정리 (이식성을 위해 필수; 일부 ABI는 메모리를 할당)
 */

/* 가변 개수의 정수를 합산.
 * 호출자는 반드시 첫 번째 인자로 개수를 전달해야 함 --
 * 함수가 인자 수를 알아낼 방법이 없음. */
int sum(int count, ...) {
    va_list args;
    va_start(args, count);  /* 초기화: 'count'는 마지막 명명된 매개변수 */

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);  /* 다음 int 가져오기 */
    }

    va_end(args);  /* 정의되지 않은 동작을 피하려면 항상 va_end 호출 */
    return total;
}

int main(void) {
    printf("Sum: %d\n", sum(3, 10, 20, 30));   /* 60 */
    printf("Sum: %d\n", sum(5, 1, 2, 3, 4, 5)); /* 15 */
    return 0;
}
```

### printf 스타일 함수 구현

실무에서 흔한 패턴은 로깅을 위해 `printf`를 래핑하는 것입니다:

```c
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

/* 타임스탬프를 앞에 붙이는 로깅 함수.
 * 포맷 문자열 + 가변 인자는 vfprintf로 전달됨.
 * vfprintf는 fprintf의 va_list 버전. */
void log_message(const char *level, const char *fmt, ...) {
    /* 타임스탬프 출력 */
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    fprintf(stderr, "[%02d:%02d:%02d] [%s] ",
            t->tm_hour, t->tm_min, t->tm_sec, level);

    /* 가변 인자를 vfprintf로 전달.
     * fprintf 대신 vfprintf를 쓰는 이유: 가변 인자를 이미
     * va_list로 받았기 때문 -- fprintf는 va_list를 받을 수 없음. */
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
}

int main(void) {
    log_message("INFO",  "Server started on port %d", 8080);
    log_message("ERROR", "Failed to open file: %s", "config.yaml");
    return 0;
}
```

### 가변 인자 함수의 타입 안전성 문제

가변 인자 함수는 본질적으로 **타입 안전하지 않습니다**: 컴파일러가 인자가 예상 타입과 일치하는지 검증할 수 없습니다.

**주요 위험**:
- 가변 인자에 대한 컴파일러 타입 검사 없음
- 잘못된 타입으로 `va_arg`를 호출하면 잘못된 바이트를 읽음 (정의되지 않은 동작)
- 예상보다 적은 인자를 전달하면 스택 쓰레기를 읽음
- 기본 인자 승격이 적용됨: `float` -> `double`, `char`/`short` -> `int`

### `restrict` 한정자

`restrict` 한정자(C99)는 프로그래머가 컴파일러에게 하는 약속입니다: **해당 포인터가 가리키는 메모리에 접근하는 유일한 방법이 이 포인터뿐**이라는 것. 이를 통해 컴파일러는 앨리어싱(aliasing) 우려 때문에 불가능했던 최적화를 수행할 수 있습니다.

```c
#include <stdio.h>
#include <string.h>

/* restrict 없이: 컴파일러는 a와 b가 겹칠 수 있다고 가정해야 함.
 * *a에 쓸 때마다 *b가 변경될 수 있으므로 매번 다시 읽어야 함. */
void add_arrays_slow(int *a, const int *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];  /* a==b 가능하면 매 반복마다 b[i]를 다시 읽어야 함 */
    }
}

/* restrict 사용: a와 b가 겹치지 않음을 약속.
 * 컴파일러가 공격적으로 벡터화(SIMD)하고, 로드/스토어를 재정렬하며,
 * 메모리에서 다시 읽지 않고 레지스터에 값을 유지할 수 있음. */
void add_arrays_fast(int *restrict a, const int *restrict b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];  /* b[i]를 캐시하고 벡터화해도 안전 */
    }
}

int main(void) {
    int x[] = {1, 2, 3, 4};
    int y[] = {10, 20, 30, 40};

    /* 올바름: x와 y는 별개의 배열 */
    add_arrays_fast(x, y, 4);

    /* 잘못됨: 겹치는 메모리를 restrict로 전달 -- 정의되지 않은 동작!
     * add_arrays_fast(x, x+1, 3);  <- restrict 계약 위반 */

    for (int i = 0; i < 4; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");  /* 11 22 33 44 */

    return 0;
}
```

### 표준 라이브러리에서의 restrict

C 표준 라이브러리는 `restrict`를 광범위하게 사용합니다. `memcpy`와 `memmove`의 시그니처를 비교해보세요:

```c
/* memcpy: 소스와 대상이 겹치면 안 됨.
 * restrict가 이를 컴파일러에 알려주어 최적화된 블록 복사가 가능. */
void *memcpy(void *restrict dest, const void *restrict src, size_t n);

/* memmove: 소스와 대상이 겹쳐도 됨.
 * restrict 없음 -> 컴파일러가 겹침을 처리해야 함 (임시 버퍼를 통해 복사). */
void *memmove(void *dest, const void *src, size_t n);

/* memcpy가 memmove보다 빠른 이유:
 * restrict 덕분에 컴파일러가 소스 데이터를 읽기 전에
 * 덮어쓸 걱정 없이 더 넓은 로드/스토어를 사용할 수 있음. */
```

**`restrict` 사용 지침**:
1. 앨리어싱이 없음을 보장할 수 있을 때 함수 매개변수에 사용
2. `restrict` 계약을 위반하면 정의되지 않은 동작 -- 컴파일러가 당신을 신뢰함
3. `restrict`는 C(C99+)에서만 존재하며, 표준 C++에는 없음 (컴파일러가 `__restrict` 제공)
4. 전후 프로파일링 필요: 최적화 이점은 루프와 대상 아키텍처에 따라 다름

---

## 연습문제

### 연습문제 1: 배열 뒤집기

포인터만 사용하여 배열을 제자리에서(in-place) 뒤집는 함수를 작성하세요.

```c
void reverse_array(int *arr, int size);

// 예: {1, 2, 3, 4, 5} -> {5, 4, 3, 2, 1}
```

### 연습문제 2: 문자열 단어 뒤집기

포인터 조작을 사용하여 "Hello World"를 "World Hello"로 변환하세요.

### 연습문제 3: 연결 리스트 뒤집기

단일 연결 리스트를 뒤집는 함수를 작성하세요.

```c
Node *reverse_list(Node *head);
```

### 연습문제 4: 함수 포인터 계산기

함수 포인터 배열을 사용하여 사칙연산을 구현하세요.

```c
// 입력: "3 + 4" -> 출력: 7
```

### 연습문제 5: 제네릭 이진 탐색

`<stdlib.h>`의 `bsearch`와 유사하게, `void*`와 비교 콜백을 사용하는 제네릭 이진 탐색 함수를 구현하세요.

```c
void *generic_bsearch(const void *key, const void *base,
                      size_t nmemb, size_t size,
                      int (*compar)(const void *, const void *));
```

---

## 다음 단계

고급 포인터를 마스터했다면 다음으로 진행하세요:
- [02. 고급 메모리 관리](./02_Advanced_Memory_Management.md) - 메모리 레이아웃, 커스텀 할당자, 디버깅 도구 심화
