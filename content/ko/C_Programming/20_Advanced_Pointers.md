# C 언어 포인터 심화

**이전**: [고급 임베디드 프로토콜](./19_Advanced_Embedded_Protocols.md) | **다음**: [C 네트워크 프로그래밍](./21_Network_Programming.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 포인터 산술(pointer arithmetic)을 사용하여 배열을 순회하고 요소 간 거리를 계산할 수 있다
2. 포인터 배열(`int *arr[]`)과 배열 포인터(`int (*p)[N]`)의 차이를 구분할 수 있다
3. 이중 포인터(double pointer)를 사용하여 함수 내부에서 호출자의 포인터를 수정할 수 있다
4. 함수 포인터를 선언·대입·호출하고, `typedef` 및 `qsort`와 함께 활용할 수 있다
5. `malloc`, `calloc`, `realloc`, `free`로 동적 메모리를 안전하게 관리하며 누수와 댕글링 참조(dangling reference)를 방지할 수 있다
6. 포인터에 `const`를 올바르게 적용하여 함수 인터페이스에서 읽기 전용 의도를 표현할 수 있다
7. 포인터 기반 할당을 사용하여 연결 리스트(linked list), 동적 2차원 배열 등 일반적인 자료구조를 구현할 수 있다
8. Valgrind와 AddressSanitizer를 사용하여 댕글링 포인터(dangling pointer), double free, 버퍼 오버플로우(buffer overflow) 버그를 탐지하고 수정할 수 있다

---

포인터는 C에서 가장 강력하면서 동시에 가장 위험한 기능입니다. 포인터는 메모리에 직접 접근하여 효율적인 자료구조, 복사 없는 인터페이스(zero-copy interface), 하드웨어 제어를 가능하게 하지만, 잘못된 역참조(dereference) 하나가 프로그램을 충돌시키거나 데이터를 조용히 오염시킬 수 있습니다. 이 레슨은 기초를 넘어 포인터에 대한 깊고 실용적인 이해를 쌓아, 조심스러운 C 프로그래머에서 자신 있는 C 프로그래머로 성장하도록 합니다.

**난이도**: 중급

---

## 1. 포인터 기초 복습

### 메모리와 주소

컴퓨터 메모리는 바이트 단위로 주소가 부여된 연속적인 공간입니다.

```c
#include <stdio.h>

int main(void) {
    int x = 42;

    printf("값: %d\n", x);           // 42
    printf("주소: %p\n", (void*)&x); // 0x7ffd12345678 (예시)
    printf("크기: %zu 바이트\n", sizeof(x)); // 4

    return 0;
}
```

### 포인터 선언과 초기화

```c
int x = 10;
int *p;      // 포인터 선언
p = &x;      // 주소 할당

// 선언과 동시에 초기화 (권장)
int *q = &x;

// 초기화하지 않은 포인터는 위험!
int *danger; // 쓰레기 값 - 사용하면 안 됨
```

### 역참조 연산자(*)

```c
int x = 42;
int *p = &x;

printf("p가 가리키는 값: %d\n", *p);  // 42

*p = 100;  // x의 값이 100으로 변경
printf("x의 새 값: %d\n", x);         // 100
```

### NULL 포인터

```c
int *p = NULL;  // 아무것도 가리키지 않음

// NULL 체크는 필수!
if (p != NULL) {
    printf("%d\n", *p);
} else {
    printf("포인터가 NULL입니다\n");
}

// C11부터 nullptr도 사용 가능 (일부 컴파일러)
```

### void 포인터

어떤 타입이든 가리킬 수 있는 범용 포인터입니다.

```c
void *generic;

int x = 42;
double d = 3.14;
char c = 'A';

generic = &x;  // OK
generic = &d;  // OK
generic = &c;  // OK

// 역참조 시 캐스팅 필요
printf("%d\n", *(int*)generic);  // 타입 캐스팅 후 역참조
```

**void 포인터 용도**:
- `malloc()` 반환 타입
- 범용 함수 작성 (예: `qsort`, `memcpy`)

---

## 2. 포인터 산술

### 포인터 증가/감소

포인터에 1을 더하면 **가리키는 타입의 크기만큼** 주소가 증가합니다.

```c
int arr[] = {10, 20, 30, 40, 50};
int *p = arr;

printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[0] = 10
p++;
printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[1] = 20
p += 2;
printf("p: %p, *p: %d\n", (void*)p, *p);      // arr[3] = 40
```

### 포인터로 배열 순회

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

### 포인터 간 뺄셈

두 포인터 사이의 **요소 개수**를 반환합니다.

```c
int arr[] = {10, 20, 30, 40, 50};
int *start = &arr[0];
int *end = &arr[4];

ptrdiff_t diff = end - start;  // 4 (바이트가 아닌 요소 수)
printf("요소 개수: %td\n", diff);
```

### 포인터 비교

```c
int arr[] = {1, 2, 3, 4, 5};
int *p1 = &arr[1];
int *p2 = &arr[3];

if (p1 < p2) {
    printf("p1이 더 앞쪽 주소\n");  // 이 줄이 출력됨
}

// 같은 배열의 포인터만 비교 가능
// 다른 배열 포인터 비교는 정의되지 않은 동작
```

---

## 3. 배열과 포인터

### 배열 이름의 의미

배열 이름은 대부분의 상황에서 **첫 번째 요소의 주소**로 변환됩니다.

```c
int arr[5] = {1, 2, 3, 4, 5};

printf("arr:     %p\n", (void*)arr);      // 같은 주소
printf("&arr[0]: %p\n", (void*)&arr[0]);  // 같은 주소

int *p = arr;  // int *p = &arr[0];과 동일
```

**예외 상황**:
```c
// sizeof는 전체 배열 크기 반환
printf("sizeof(arr): %zu\n", sizeof(arr));  // 20 (5 * 4바이트)

// &arr은 배열 전체의 주소 (타입이 다름)
printf("arr:  %p\n", (void*)arr);           // int* 타입
printf("&arr: %p\n", (void*)&arr);          // int(*)[5] 타입

// 주소는 같지만 +1의 의미가 다름
printf("arr + 1:  %p\n", (void*)(arr + 1));   // 4바이트 증가
printf("&arr + 1: %p\n", (void*)(&arr + 1));  // 20바이트 증가
```

### 배열 인덱싱의 진실

`arr[i]`는 `*(arr + i)`의 문법적 설탕(syntactic sugar)입니다.

```c
int arr[] = {10, 20, 30};

// 모두 동일한 값
printf("%d\n", arr[1]);       // 20
printf("%d\n", *(arr + 1));   // 20
printf("%d\n", *(1 + arr));   // 20
printf("%d\n", 1[arr]);       // 20 (이상하지만 합법!)
```

### 2차원 배열

```c
int matrix[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

// 요소 접근
printf("%d\n", matrix[1][2]);           // 7
printf("%d\n", *(*(matrix + 1) + 2));   // 7

// matrix는 int[4] 배열을 가리키는 포인터로 변환됨
// matrix[i]는 i번째 행의 첫 번째 요소 주소
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
int *ptr_arr[3];   // [3]이 먼저 → ptr_arr은 크기 3인 배열
                   // *이 다음 → 요소가 포인터
                   // int → int에 대한 포인터

int (*arr_ptr)[4]; // *이 먼저 (괄호) → arr_ptr은 포인터
                   // [4]가 다음 → 크기 4인 배열을 가리킴
                   // int → int 배열
```

---

## 4. 다중 포인터

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
    p = malloc(size * sizeof(int));  // 로컬 p만 변경됨
    // 호출자의 포인터는 변경되지 않음
}

// 올바른 방법: 이중 포인터 사용
void allocate_correct(int **pp, int size) {
    *pp = malloc(size * sizeof(int));  // 호출자의 포인터를 변경
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

### 동적 2차원 배열

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int rows = 3, cols = 4;

    // 방법 1: 포인터 배열 (행마다 별도 할당)
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
    // argv는 char* 배열
    // argv[0]: 프로그램 이름
    // argv[1] ~ argv[argc-1]: 인자들

    printf("인자 개수: %d\n", argc);

    for (int i = 0; i < argc; i++) {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    return 0;
}
```

```c
// 문자열 배열 직접 만들기
char *fruits[] = {"apple", "banana", "cherry"};
int n = sizeof(fruits) / sizeof(fruits[0]);

for (int i = 0; i < n; i++) {
    printf("%s\n", fruits[i]);
}
```

---

## 5. 함수 포인터

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

### typedef로 가독성 높이기

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

### 콜백 함수(Callback)

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

    printf("두 배: ");
    process_array(arr, n, print_double);
    printf("\n");

    return 0;
}
```

### qsort 활용

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

## 6. 동적 메모리 관리

### malloc, calloc, realloc, free

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    // malloc: 초기화 없이 할당
    int *arr1 = malloc(5 * sizeof(int));
    // 값이 쓰레기! 초기화 필요

    // calloc: 0으로 초기화하여 할당
    int *arr2 = calloc(5, sizeof(int));
    // 모든 값이 0

    // realloc: 크기 변경
    arr1 = realloc(arr1, 10 * sizeof(int));
    // 기존 값 유지, 추가 공간은 초기화 안 됨

    // NULL 체크 필수!
    if (arr1 == NULL || arr2 == NULL) {
        fprintf(stderr, "메모리 할당 실패\n");
        return 1;
    }

    // 사용 후 해제
    free(arr1);
    free(arr2);

    // 해제 후 NULL로 설정 (선택적이지만 권장)
    arr1 = NULL;
    arr2 = NULL;

    return 0;
}
```

### 메모리 누수 방지

```c
// 잘못된 패턴: 메모리 누수
void memory_leak(void) {
    int *p = malloc(100);
    // free 없이 함수 종료 → 누수!
}

// 올바른 패턴
void no_leak(void) {
    int *p = malloc(100);
    if (p == NULL) return;

    // 작업 수행...

    free(p);  // 반드시 해제
}

// 에러 처리 시 주의
int process(void) {
    int *a = malloc(100);
    int *b = malloc(200);

    if (a == NULL || b == NULL) {
        free(a);  // NULL이어도 free 호출 가능
        free(b);
        return -1;
    }

    // 작업 수행...

    free(a);
    free(b);
    return 0;
}
```

### realloc 안전하게 사용하기

```c
// 위험한 패턴
p = realloc(p, new_size);  // 실패 시 원본 주소 유실!

// 안전한 패턴
int *temp = realloc(p, new_size);
if (temp == NULL) {
    // p는 여전히 유효
    free(p);
    return NULL;
}
p = temp;
```

---

## 7. const와 포인터

### 네 가지 조합

```c
int x = 10;
int y = 20;

// 1. 일반 포인터
int *p1 = &x;
*p1 = 30;   // OK: 값 변경 가능
p1 = &y;    // OK: 다른 주소 가리키기 가능

// 2. const int* (pointer to const int)
// = int const *
const int *p2 = &x;
// *p2 = 30;  // 에러: 값 변경 불가
p2 = &y;      // OK: 다른 주소 가리키기 가능

// 3. int* const (const pointer to int)
int *const p3 = &x;
*p3 = 30;     // OK: 값 변경 가능
// p3 = &y;   // 에러: 다른 주소 가리키기 불가

// 4. const int* const (const pointer to const int)
const int *const p4 = &x;
// *p4 = 30;  // 에러: 값 변경 불가
// p4 = &y;   // 에러: 다른 주소 가리키기 불가
```

### 읽는 방법

오른쪽에서 왼쪽으로 읽으세요:

```c
const int *p;      // p는 포인터, int const를 가리킴
int *const p;      // p는 const 포인터, int를 가리킴
const int *const p; // p는 const 포인터, int const를 가리킴
```

### 함수 매개변수에서의 const

```c
// 입력 전용: 값을 변경하지 않음을 명시
void print_array(const int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
        // arr[i] = 0;  // 컴파일 에러!
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

## 8. 문자열과 포인터

### 문자열 리터럴 vs 문자 배열

```c
// 문자열 리터럴: 읽기 전용 메모리
char *str1 = "Hello";
// str1[0] = 'h';  // 정의되지 않은 동작! (대부분 크래시)

// 문자 배열: 수정 가능
char str2[] = "Hello";
str2[0] = 'h';  // OK

// const 사용 권장
const char *str3 = "Hello";  // 의도를 명확히
```

### 문자열 함수 직접 구현

```c
#include <stdio.h>

// strlen 구현
size_t my_strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}

// strcpy 구현
char *my_strcpy(char *dest, const char *src) {
    char *ret = dest;
    while ((*dest++ = *src++));
    return ret;
}

// strcmp 구현
int my_strcmp(const char *s1, const char *s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *(unsigned char*)s1 - *(unsigned char*)s2;
}

// strcat 구현
char *my_strcat(char *dest, const char *src) {
    char *ret = dest;
    while (*dest) dest++;  // 끝으로 이동
    while ((*dest++ = *src++));
    return ret;
}

int main(void) {
    char buffer[100] = "Hello";

    printf("길이: %zu\n", my_strlen(buffer));  // 5

    my_strcat(buffer, " World");
    printf("%s\n", buffer);  // Hello World

    return 0;
}
```

### 문자열 배열

```c
// 방법 1: 포인터 배열 (다른 길이 가능)
const char *names1[] = {
    "Alice",
    "Bob",
    "Charlie"
};

// 방법 2: 2차원 배열 (고정 길이)
char names2[][10] = {
    "Alice",
    "Bob",
    "Charlie"
};

// 차이점
printf("sizeof(names1[0]): %zu\n", sizeof(names1[0]));  // 8 (포인터 크기)
printf("sizeof(names2[0]): %zu\n", sizeof(names2[0]));  // 10 (배열 크기)
```

---

## 9. 구조체와 포인터

### 구조체 포인터 기본

```c
#include <stdio.h>
#include <string.h>

typedef struct {
    char name[50];
    int age;
    double height;
} Person;

int main(void) {
    Person p1 = {"Alice", 25, 165.5};
    Person *ptr = &p1;

    // 멤버 접근: -> 연산자
    printf("이름: %s\n", ptr->name);      // (*ptr).name과 동일
    printf("나이: %d\n", ptr->age);

    // 값 수정
    ptr->age = 26;

    return 0;
}
```

### 동적 구조체

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *name;  // 동적 할당할 문자열
    int age;
} Person;

Person *create_person(const char *name, int age) {
    Person *p = malloc(sizeof(Person));
    if (p == NULL) return NULL;

    p->name = malloc(strlen(name) + 1);
    if (p->name == NULL) {
        free(p);
        return NULL;
    }

    strcpy(p->name, name);
    p->age = age;

    return p;
}

void free_person(Person *p) {
    if (p) {
        free(p->name);
        free(p);
    }
}

int main(void) {
    Person *alice = create_person("Alice", 25);
    if (alice) {
        printf("%s, %d\n", alice->name, alice->age);
        free_person(alice);
    }
    return 0;
}
```

### 자기참조 구조체(연결 리스트)

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

## 10. 흔한 실수와 디버깅

### 댕글링 포인터(Dangling Pointer)

해제된 메모리를 가리키는 포인터입니다.

```c
// 위험한 코드
int *p = malloc(sizeof(int));
*p = 42;
free(p);
// p는 여전히 같은 주소를 가리킴 (댕글링 포인터)
printf("%d\n", *p);  // 정의되지 않은 동작!

// 해결책
free(p);
p = NULL;  // 명시적으로 NULL 설정

if (p != NULL) {
    printf("%d\n", *p);  // NULL 체크로 방어
}
```

### Use After Free

```c
// 위험한 패턴
char *str = malloc(100);
strcpy(str, "Hello");
free(str);
// ...
printf("%s\n", str);  // 해제된 메모리 접근!
```

### Double Free

```c
// 위험한 코드
int *p = malloc(sizeof(int));
free(p);
free(p);  // 같은 메모리 두 번 해제 → 크래시 가능

// 해결책
free(p);
p = NULL;
free(p);  // NULL free는 안전함
```

### 버퍼 오버플로우(Buffer Overflow)

```c
// 위험한 코드
char buffer[10];
strcpy(buffer, "This is a very long string");  // 오버플로우!

// 안전한 코드
char buffer[10];
strncpy(buffer, "This is a very long string", sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';

// 또는 snprintf 사용
snprintf(buffer, sizeof(buffer), "%s", "This is a very long string");
```

### Valgrind로 메모리 오류 찾기

```bash
# 컴파일 (디버그 정보 포함)
gcc -g -o myprogram myprogram.c

# Valgrind 실행
valgrind --leak-check=full ./myprogram
```

**Valgrind 출력 예시**:
```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 100 bytes in 1 blocks
==12345==   total heap usage: 5 allocs, 4 frees, 500 bytes allocated
==12345==
==12345== 100 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4C2BBAF: malloc (vg_replace_malloc.c:299)
==12345==    by 0x400547: main (myprogram.c:10)
```

### 디버깅 팁

1. **포인터 출력하기**
```c
printf("ptr = %p, *ptr = %d\n", (void*)ptr, ptr ? *ptr : -1);
```

2. **assert 사용하기**
```c
#include <assert.h>

void process(int *arr, int size) {
    assert(arr != NULL);
    assert(size > 0);
    // ...
}
```

3. **AddressSanitizer 사용** (GCC/Clang)
```bash
gcc -fsanitize=address -g myprogram.c -o myprogram
./myprogram
```

---

## 11. 가변 인자 함수와 `restrict` 한정자(Variadic Functions and restrict Qualifier)

### `<stdarg.h>`를 이용한 가변 인자 함수(Variadic Functions)

C는 `<stdarg.h>` 헤더를 통해 가변 개수의 인자를 받는 함수를 지원합니다. `printf`, `scanf` 등이 내부적으로 이 방식을 사용합니다.

```c
#include <stdio.h>
#include <stdarg.h>

/*
 * va_list  - 인자 목록을 순회하는 데 필요한 상태를 담는 타입
 * va_start - va_list를 첫 번째 가변 인자를 가리키도록 초기화
 * va_arg   - 다음 인자를 꺼내면서 내부 포인터를 이동
 * va_end   - 정리 (이식성을 위해 반드시 호출; 일부 ABI는 메모리 할당)
 */

/* 가변 개수의 정수를 합산합니다.
 * 호출자는 첫 번째 인자로 개수를 전달해야 합니다 -- 함수가
 * 인자 개수를 스스로 알아낼 방법은 없습니다. */
int sum(int count, ...) {
    va_list args;
    va_start(args, count);  /* 초기화: 'count'는 마지막 명명된 매개변수 */

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);  /* 다음 int 꺼내기 */
    }

    va_end(args);  /* 정의되지 않은 동작 방지를 위해 반드시 호출 */
    return total;
}

int main(void) {
    printf("Sum: %d\n", sum(3, 10, 20, 30));   /* 60 */
    printf("Sum: %d\n", sum(5, 1, 2, 3, 4, 5)); /* 15 */
    return 0;
}
```

### printf와 유사한 함수 구현

실무에서 흔한 패턴은 `printf`를 감싸 로깅 함수를 만드는 것입니다:

```c
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

/* 타임스탬프를 앞에 붙이는 로깅 함수.
 * 서식 문자열 + 가변 인자를 vfprintf에 전달합니다.
 * vfprintf는 fprintf의 va_list 버전입니다. */
void log_message(const char *level, const char *fmt, ...) {
    /* 타임스탬프 출력 */
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    fprintf(stderr, "[%02d:%02d:%02d] [%s] ",
            t->tm_hour, t->tm_min, t->tm_sec, level);

    /* 가변 인자를 vfprintf로 전달.
     * vfprintf를 쓰는 이유: 이미 가변 인자를 va_list로 소비했기 때문에
     * fprintf는 va_list를 받을 수 없습니다. */
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

### 가변 인자 함수의 타입 안전성 문제(Type Safety Issues)

가변 인자 함수는 본질적으로 **타입 안전하지 않습니다**: 컴파일러가 인자가 기대하는 타입과 일치하는지 검증할 수 없습니다.

```c
#include <stdio.h>
#include <stdarg.h>

double average(int count, ...) {
    va_list args;
    va_start(args, count);

    double total = 0.0;
    for (int i = 0; i < count; i++) {
        /* 위험: 호출자가 double을 기대하는 곳에 int를 전달하면
         * va_arg가 잘못된 바이트 수를 읽어 쓰레기 값이 됩니다.
         * 컴파일러는 이 불일치에 대해 경고하지 않습니다! */
        total += va_arg(args, double);
    }

    va_end(args);
    return total / count;
}

int main(void) {
    /* 올바름: double을 전달 */
    printf("%.2f\n", average(3, 1.0, 2.0, 3.0));  /* 2.00 */

    /* 버그: double 대신 int를 전달 -- 정의되지 않은 동작!
     * average(3, 1, 2, 3);  ← int는 4바이트, double은 8바이트 */

    return 0;
}
```

**주요 위험 사항**:
- 컴파일러가 가변 인자에 대한 타입 검사를 하지 않음
- 잘못된 타입으로 `va_arg`를 호출하면 잘못된 바이트를 읽음 (정의되지 않은 동작)
- 기대보다 적은 인자를 전달하면 스택의 쓰레기 값을 읽음
- 기본 인자 승격(default argument promotion)이 적용됨: `float` → `double`, `char`/`short` → `int`

### `restrict` 한정자(restrict Qualifier)

`restrict` 한정자(C99)는 프로그래머가 컴파일러에게 하는 약속입니다: **해당 포인터가 수명 동안 그 메모리에 접근하는 유일한 방법**이라는 것입니다. 이를 통해 컴파일러는 앨리어싱(aliasing) 우려 때문에 불가능했던 최적화를 수행할 수 있습니다.

```c
#include <stdio.h>
#include <string.h>

/* restrict 없이: 컴파일러는 a와 b가 겹칠 수 있다고 가정해야 합니다.
 * *a에 쓸 때마다 *b가 바뀔 수 있으므로 매번 다시 읽어야 합니다. */
void add_arrays_slow(int *a, const int *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];  /* a==b가 가능하면 매 반복마다 b[i]를 다시 읽어야 함 */
    }
}

/* restrict 사용: a와 b가 겹치지 않는다고 약속.
 * 컴파일러가 공격적으로 벡터화(SIMD)하고, 로드/스토어를 재정렬하며,
 * 값을 메모리에서 다시 읽지 않고 레지스터에 유지할 수 있습니다. */
void add_arrays_fast(int *restrict a, const int *restrict b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];  /* b[i]를 캐싱하고 벡터화 가능 */
    }
}

int main(void) {
    int x[] = {1, 2, 3, 4};
    int y[] = {10, 20, 30, 40};

    /* 올바름: x와 y는 별도의 배열 */
    add_arrays_fast(x, y, 4);

    /* 잘못됨: 겹치는 메모리를 restrict와 함께 전달 -- 정의되지 않은 동작!
     * add_arrays_fast(x, x+1, 3);  ← restrict 계약 위반 */

    for (int i = 0; i < 4; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");  /* 11 22 33 44 */

    return 0;
}
```

### restrict와 앨리어싱(Aliasing): 왜 중요한가

```c
#include <stdio.h>

/* 고전적인 예: restrict 없이 이 함수는 성능이 저하됩니다.
 * a == b인 경우를 생각해 보세요: *a에 쓰면 *b가 바뀝니다! */
void multiply(int *a, int *b, int *result) {
    *result = *a * *b;
    /* 이후에 *a가 다시 필요하면 메모리에서 다시 읽어야 함
     * result == a일 수 있으므로 *result를 쓰면 *a가 변했을 수 있음 */
}

/* restrict 사용: 컴파일러는 a, b, result가 모두 다른 메모리임을 압니다.
 * *a와 *b를 레지스터에 유지하고 다시 읽기를 생략할 수 있습니다. */
void multiply_fast(int *restrict a, int *restrict b, int *restrict result) {
    *result = *a * *b;
    /* 컴파일러가 *a와 *b가 변경되지 않았다고 신뢰 가능 */
}
```

### 표준 라이브러리에서의 restrict

C 표준 라이브러리는 `restrict`를 광범위하게 사용합니다. `memcpy`와 `memmove`의 시그니처를 비교해 보세요:

```c
/* memcpy: 소스와 대상이 겹치면 안 됨.
 * restrict가 이를 컴파일러에 알려 최적화된 블록 복사가 가능. */
void *memcpy(void *restrict dest, const void *restrict src, size_t n);

/* memmove: 소스와 대상이 겹칠 수 있음.
 * restrict 없음 → 컴파일러가 겹침을 처리해야 함 (임시 버퍼 경유 복사). */
void *memmove(void *dest, const void *src, size_t n);

/* memcpy가 memmove보다 빠른 이유:
 * restrict 덕분에 컴파일러가 소스 데이터를 읽기 전에 덮어쓸 걱정 없이
 * 더 넓은 로드/스토어를 사용할 수 있습니다. */
```

### 실용적인 restrict 사용 패턴

```c
#include <stddef.h>

/* 패턴 1: 함수 매개변수 -- 가장 흔한 사용처 */
void process(float *restrict output,
             const float *restrict input,
             size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] * 2.0f;
    }
}

/* 패턴 2: 구조체 멤버 (드물지만 C99에서 유효) */
struct Buffer {
    float *restrict data;  /* 이 포인터만 버퍼에 접근 */
    size_t size;
};

/* 패턴 3: 지역 변수 */
void compute(float *base, size_t n) {
    /* 이 지역 뷰들이 서로 앨리어스하지 않는다고 컴파일러에 알림 */
    float *restrict first_half  = base;
    float *restrict second_half = base + n / 2;
    /* 두 포인터로 같은 요소에 접근하지 않는 경우에만 유효 */
}
```

**`restrict` 사용 지침**:
1. 앨리어싱이 없다고 보장할 수 있는 함수 매개변수에 사용
2. `restrict` 계약을 위반하면 정의되지 않은 동작 -- 컴파일러가 여러분을 신뢰합니다
3. `restrict`는 C(C99+)에만 존재하며, 표준 C++에는 없음 (컴파일러가 `__restrict`를 제공하기도 함)
4. 적용 전후에 프로파일링: 최적화 효과는 루프와 대상 아키텍처에 따라 다름

---

## 연습 문제

### 문제 1: 배열 뒤집기

포인터만 사용하여 배열을 제자리에서 뒤집는 함수를 작성하세요.

```c
void reverse_array(int *arr, int size);

// 예시: {1, 2, 3, 4, 5} → {5, 4, 3, 2, 1}
```

### 문제 2: 문자열 단어 뒤집기

"Hello World"를 "World Hello"로 변환하세요.

### 문제 3: 연결 리스트 뒤집기

단일 연결 리스트를 뒤집는 함수를 작성하세요.

```c
Node *reverse_list(Node *head);
```

### 문제 4: 함수 포인터 계산기

사칙연산을 함수 포인터 배열로 구현하세요.

```c
// 입력: "3 + 4" → 출력: 7
```

---

## 요약

| 개념 | 핵심 포인트 |
|------|------------|
| 포인터 기본 | `&`(주소), `*`(역참조), NULL 체크 필수 |
| 포인터 산술 | 타입 크기만큼 증가/감소 |
| 배열과 포인터 | `arr[i] == *(arr + i)` |
| 다중 포인터 | 함수에서 포인터 수정 시 사용 |
| 함수 포인터 | 콜백, qsort 비교 함수 |
| 동적 메모리 | malloc/free, 누수 방지, realloc 안전 패턴 |
| const 포인터 | `const int*` vs `int* const` |
| 디버깅 | Valgrind, AddressSanitizer |

---

## 참고 자료

- [C Programming: A Modern Approach (K.N. King)](http://knking.com/books/c2/)
- [The C Programming Language (K&R)](https://en.wikipedia.org/wiki/The_C_Programming_Language)
- [Valgrind Documentation](https://valgrind.org/docs/manual/quick-start.html)
- [cdecl: C declaration decoder](https://cdecl.org/)

---

**이전**: [고급 임베디드 프로토콜](./19_Advanced_Embedded_Protocols.md) | **다음**: [C 네트워크 프로그래밍](./21_Network_Programming.md)
