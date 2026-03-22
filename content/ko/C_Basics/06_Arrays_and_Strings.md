# 배열과 문자열

**이전**: [함수와 스코프](./05_Functions_and_Scope.md) | **다음**: [포인터 기초](./07_Pointers_Fundamentals.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 1차원 배열을 선언, 초기화, 순회한다
2. 다차원 배열을 다루고 메모리 배치를 이해한다
3. `sizeof`를 사용하여 배열 크기를 계산하고 함수에 배열을 전달한다
4. 표준 라이브러리 함수(`strlen`, `strcpy`, `strcat`, `strcmp`, `strncpy`, `snprintf`)를 사용하여 C 문자열을 조작한다
5. 널 종단자(null terminator)를 설명하고 문자열 리터럴과 char 배열을 구별한다

---

배열은 같은 타입의 고정된 수의 값을 연속된 메모리에 저장할 수 있게 합니다. C에서 문자열은 특수한 널 바이트로 끝나는 문자 배열에 불과합니다. 거의 모든 비자명(non-trivial) C 프로그램이 데이터 저장과 텍스트 처리를 위해 이 둘에 의존하므로, 둘 다 이해하는 것이 필수적입니다.

## 1. 배열 선언과 초기화

배열 선언은 요소 타입과 컴파일 시간 상수 크기를 지정합니다.

```c
int scores[5];               /* 초기화되지 않음 — 쓰레기 값 포함 */
int primes[5] = {2, 3, 5, 7, 11};  /* 완전 초기화 */
int zeros[5] = {0};          /* 부분 초기화 — 첫 요소 0, 나머지 자동 0 */
int partial[5] = {10, 20};   /* 10, 20, 0, 0, 0 */
```

초기화 리스트를 제공하면 컴파일러가 크기를 추론할 수 있습니다:

```c
int data[] = {1, 2, 3, 4};   /* 컴파일러가 크기 = 4로 추론 */
```

**C99 지정 초기화(Designated Initializer)**로 특정 인덱스를 설정할 수 있습니다:

```c
int sparse[10] = {
    [0] = 100,
    [5] = 500,
    [9] = 900
};
/* 다른 인덱스의 요소는 0 */
```

---

## 2. 요소 접근과 수정

배열 요소는 `[]` 연산자로 0부터 시작하는 인덱싱을 사용하여 접근합니다.

```c
#include <stdio.h>

int main(void) {
    int temps[7] = {22, 25, 19, 28, 31, 27, 23};

    /* 읽기 */
    printf("Monday: %d°C\n", temps[0]);

    /* 쓰기 */
    temps[2] = 21;

    /* 순회 */
    for (int i = 0; i < 7; i++) {
        printf("Day %d: %d°C\n", i + 1, temps[i]);
    }
    return 0;
}
```

**범위 검사 없음**: C는 인덱스가 선언된 크기 내에 있는지 확인하지 않습니다. `temps[7]`이나 `temps[-1]`에 접근하면 미정의 동작입니다 -- 프로그램이 충돌하거나, 잘못된 결과를 내거나, 그때까지 정상적으로 작동하는 것처럼 보일 수 있습니다.

| 흔한 버그 | 설명 |
|----------|------|
| Off-by-one | `i < size` 대신 `i <= size`로 루프 |
| 초기화되지 않은 읽기 | 초기화되지 않은 배열에서 읽기 |
| 음수 인덱스 | 부호 있는 변수가 음수가 되는 경우 |

---

## 3. 다차원 배열

2차원 배열은 배열의 배열입니다. C는 **행 우선 순서(row-major order)**로 저장합니다 -- 행 0의 모든 요소가 메모리에서 먼저 오고, 그 다음 행 1, 이런 식입니다.

```c
#include <stdio.h>

int main(void) {
    int matrix[3][4] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };

    /* 행렬 출력 */
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 4; c++) {
            printf("%3d ", matrix[r][c]);
        }
        printf("\n");
    }
    return 0;
}
```

`matrix[3][4]`의 메모리 배치:

```
Address:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11]
Element:   1    2    3    4    5    6    7    8    9   10   11   12
Row:      |--- row 0 ---|--- row 1 ---|--- row 2 ---|
```

요소 `matrix[r][c]`는 시작점에서 오프셋 `r * 4 + c`에 위치합니다.

---

## 4. 배열 크기

`sizeof` 연산자는 배열 이름에 직접 적용할 때 배열의 총 바이트 수를 반환합니다.

```c
#include <stdio.h>

int main(void) {
    int data[] = {10, 20, 30, 40, 50};

    size_t total_bytes = sizeof(data);         /* 예: int = 4바이트인 시스템에서 20 */
    size_t element_size = sizeof(data[0]);     /* 4 */
    size_t count = sizeof(data) / sizeof(data[0]);  /* 5 */

    printf("Array has %zu elements\n", count);
    return 0;
}
```

**포인터에서 실패하는 이유**: 배열이 함수에 전달되면 포인터로 퇴화(decay)합니다. `sizeof(pointer)`는 배열 크기가 아닌 포인터 크기(64비트에서 보통 8바이트)를 제공합니다. 이것은 가장 흔한 C 함정 중 하나입니다.

```c
void print_size(int arr[]) {
    /* sizeof(arr) == sizeof(int *) == 8, 배열 크기가 아님 */
    printf("Inside function: %zu\n", sizeof(arr));  /* 8 */
}
```

---

## 5. 함수에 배열 전달

배열이 함수에 전달될 때 포인터로 퇴화하므로, 크기를 항상 별도의 매개변수로 전달해야 합니다.

```c
#include <stdio.h>

double average(const int arr[], size_t n) {
    long sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += arr[i];
    }
    return (double)sum / n;
}

int main(void) {
    int scores[] = {85, 92, 78, 96, 88};
    size_t n = sizeof(scores) / sizeof(scores[0]);

    printf("Average: %.1f\n", average(scores, n));
    return 0;
}
```

| 매개변수 스타일 | 동등 표현 | 비고 |
|---------------|----------|------|
| `int arr[]` | `int *arr` | 크기 정보 손실 |
| `int arr[5]` | `int *arr` | `5`는 컴파일러가 무시 |
| `const int arr[]` | `const int *arr` | 수정 방지 |

2차원 배열의 경우 열 수를 지정해야 합니다:

```c
void print_matrix(int rows, int cols, int mat[][4]) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%d ", mat[r][c]);
        }
        printf("\n");
    }
}
```

---

## 6. 문자 배열과 문자열

**C 문자열**은 마지막 의미 있는 문자 뒤에 **널 종단자** `'\0'` (바이트 값 0)이 오는 문자 배열입니다.

```c
char greeting[6] = {'H', 'e', 'l', 'l', 'o', '\0'};
char greeting2[] = "Hello";  /* 컴파일러가 자동으로 '\0' 추가, 크기 = 6 */
```

**문자열 리터럴**(예: `"Hello"`)은 읽기 전용 메모리에 저장됩니다. `char *`에 할당하면 불변 데이터에 대한 포인터를 얻게 됩니다:

```c
char arr[] = "Hello";    /* 스택의 수정 가능한 복사본 — 수정 안전 */
char *ptr  = "Hello";    /* 읽기 전용 리터럴에 대한 포인터 — 수정은 미정의 동작 */

arr[0] = 'J';   /* OK: arr은 이제 "Jello" */
/* ptr[0] = 'J';  미정의 동작 — 문자열 리터럴을 수정하지 마세요 */
```

| 측면 | `char arr[] = "Hi"` | `char *ptr = "Hi"` |
|------|---------------------|---------------------|
| 저장소 | 스택 (수정 가능한 복사본) | 읽기 전용 섹션 |
| `sizeof` | 3 (`'\0'` 포함) | 8 (포인터 크기) |
| 수정 가능 | 예 | 아니오 (미정의 동작) |

---

## 7. 문자열 함수

모든 문자열 함수에는 `<string.h>`를 포함하세요. 대상 버퍼가 충분히 큰지 항상 확인하세요.

### 길이

```c
#include <string.h>

char msg[] = "Hello";
size_t len = strlen(msg);  /* 5 — '\0'은 세지 않음 */
```

### 복사

```c
char dest[20];
strcpy(dest, "Hello");        /* "Hello\0"을 dest에 복사 */
strncpy(dest, "Hello", 19);   /* 최대 19문자 복사, 널 종단을 보장하지 않을 수 있음 */
dest[19] = '\0';              /* strncpy 후 항상 널 종단 보장 */
```

### 연결

```c
char buf[50] = "Hello";
strcat(buf, ", ");            /* buf = "Hello, " */
strcat(buf, "World!");        /* buf = "Hello, World!" */

/* 길이 제한이 있는 안전한 버전 */
strncat(buf, " Bye", sizeof(buf) - strlen(buf) - 1);
```

### 비교

```c
int result = strcmp("apple", "banana");
/* result < 0: "apple"이 "banana" 앞에 옴 */
/* result == 0: 문자열이 같음 */
/* result > 0: 첫 번째 문자열이 두 번째 뒤에 옴 */

/* 최대 n문자까지 비교 */
int cmp = strncmp("Hello", "Help", 3);  /* 0 — 처음 3문자 일치 */
```

### 문자열에 형식화된 출력

```c
char buf[100];
int age = 30;
snprintf(buf, sizeof(buf), "Age: %d years", age);
/* buf = "Age: 30 years" */
/* snprintf는 버퍼 크기를 초과하여 쓰지 않음 — sprintf보다 항상 선호 */
```

### 메모리 함수

`<string.h>`는 널 종단 문자열이 아닌 원시 바이트를 다루는 세 가지 핵심 함수도 제공합니다. 이 함수들은 문자뿐만 아니라 모든 데이터 타입에 작동합니다.

| 함수 | 시그니처 | 용도 |
|------|---------|------|
| `memcpy` | `void *memcpy(void *dest, const void *src, size_t n)` | `n`바이트 복사; 영역이 **겹치면 안 됨** |
| `memmove` | `void *memmove(void *dest, const void *src, size_t n)` | `n`바이트 복사; 영역이 겹쳐도 안전 |
| `memset` | `void *memset(void *dest, int val, size_t n)` | `n`바이트를 `val`로 채움 |

```c
#include <string.h>
#include <stdio.h>

int main(void) {
    int src[5] = {1, 2, 3, 4, 5};
    int dst[5];

    memcpy(dst, src, sizeof(src));   /* 빠른 복사 — src와 dst는 겹치지 않음 */

    /* 배열을 0으로 초기화 — memset의 가장 흔한 용도 */
    int arr[100];
    memset(arr, 0, sizeof(arr));     /* 모든 바이트를 0으로 설정 */

    /* 겹치는 복사: 요소를 오른쪽으로 한 칸 이동 */
    /* 여기서 memcpy를 사용하면 미정의 동작 — memmove를 사용 */
    memmove(src + 1, src, 4 * sizeof(int));  /* 겹침이 있어도 안전 */
    src[0] = 0;
    /* src는 이제 {0, 1, 2, 3, 4} */

    return 0;
}
```

> **구분이 중요한 이유**: `memcpy`는 바이트보다 큰 청크를 읽고 쓰는 SIMD 명령어로 구현될 수 있어 더 빠르지만 출발지와 목적지가 겹칠 때는 안전하지 않습니다. `memmove`는 겹치는 영역에서도 정확성을 보장하며, 일반적으로 임시 버퍼를 통해 복사하거나 주소 순서에 따라 복사 방향을 선택합니다.

---

## 8. 문자열 입력

### fgets (권장)

`fgets`는 최대 `n-1`자를 읽고 항상 널 종단합니다. 공간이 있으면 개행 문자를 포함합니다.

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char name[50];

    printf("Enter your name: ");
    if (fgets(name, sizeof(name), stdin) != NULL) {
        /* 있으면 후행 개행 제거 */
        name[strcspn(name, "\n")] = '\0';
        printf("Hello, %s!\n", name);
    }
    return 0;
}
```

### scanf로 문자열 입력 (위험)

```c
char word[20];
scanf("%19s", word);  /* 한 단어 읽기, 최대 19자 + '\0' */
/* 너비 제한 없이 scanf를 사용하면 버퍼를 오버플로우할 수 있음 */
```

| 함수 | 버퍼 오버플로우? | 공백 읽기? | 개행 처리 |
|------|----------------|-----------|----------|
| `fgets` | 아니오 (크기가 올바르면) | 예 | 버퍼에 포함 |
| `scanf("%s", ...)` | 예 (너비 없으면) | 아니오 (공백에서 멈춤) | 입력 버퍼에 남김 |
| `gets` | **항상** | 예 | 제거됨 — **절대 사용 금지** |

`gets` 함수는 어떤 상황에서도 버퍼 오버플로우를 방지할 수 없기 때문에 C11 표준에서 제거되었습니다.

---

## 연습문제

**연습문제 1 -- 배열 통계**: 10개의 정수를 배열에 읽어 최솟값, 최댓값, 평균을 출력하는 프로그램을 작성하세요.

**연습문제 2 -- 배열 뒤집기**: 배열을 제자리에서 뒤집는 함수 `void reverse(int arr[], size_t n)`를 작성하세요. 홀수와 짝수 길이의 배열 모두로 테스트하세요.

**연습문제 3 -- 행렬 전치**: 3x3 정수 행렬을 받아 전치(행이 열이 됨)를 출력하는 함수를 작성하세요.

**연습문제 4 -- 문자열 뒤집기**: 라이브러리 함수를 사용하지 않고 C 문자열을 제자리에서 뒤집는 함수 `void str_reverse(char s[])`를 작성하세요. 빈 문자열 경우를 처리하세요.

**연습문제 5 -- 단어 카운터**: `fgets`로 한 줄의 텍스트를 읽고 단어(공백이 아닌 문자의 연속) 수를 세는 프로그램을 작성하세요. 연속된 여러 공백을 올바르게 처리하세요.

---

## 다음 단계

이제 배열에 데이터 컬렉션을 저장하고 C 문자열로 텍스트를 조작하는 방법을 알게 되었습니다. 다음 레슨 [포인터 기초](./07_Pointers_Fundamentals.md)에서는 포인터가 내부적으로 어떻게 작동하는지 배울 것입니다 -- C에서 배열, 문자열, 동적 메모리를 가능하게 하는 메커니즘입니다.
