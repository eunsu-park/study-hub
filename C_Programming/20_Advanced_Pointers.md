# C 언어 포인터 심화

## 목표

- 포인터의 동작 원리를 깊이 이해한다
- 다양한 포인터 활용 패턴을 익힌다
- 포인터 관련 흔한 실수를 피하는 방법을 배운다

**난이도**: ⭐⭐⭐ (중급)

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

### 역참조 연산자 (*)

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

### 이중 포인터 (Pointer to Pointer)

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

### 콜백 함수

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

### 자기참조 구조체 (연결 리스트)

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

### 댕글링 포인터 (Dangling Pointer)

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

### 버퍼 오버플로우

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
