# 구조체와 공용체

**이전**: [포인터 기초](./07_Pointers_Fundamentals.md) | **다음**: [동적 메모리](./09_Dynamic_Memory.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 여러 필드를 가진 구조체를 정의하고 점(`.`) 연산자와 화살표(`->`) 연산자로 멤버에 접근하기
2. `typedef`를 사용하여 구조체에 편리한 타입 별칭 만들기
3. 구조체와 공용체의 차이점과 메모리 레이아웃 차이 설명하기
4. 이름이 있는 정수 상수를 위한 열거형 정의 및 사용하기
5. 비트 필드를 적용하여 데이터를 컴팩트하게 표현하기

---

지금까지 사용한 모든 변수는 단일 값을 담고 있었습니다 — 정수 하나, 실수 하나, 문자 하나. 실제 프로그램은 여러 속성을 가진 엔티티를 모델링합니다: 학생은 이름, 학번, 학점을 가지고 있으며, 픽셀은 빨강, 초록, 파랑 컴포넌트를 가지고 있습니다. 구조체를 사용하면 이러한 관련 데이터 조각들을 하나의 커스텀 타입으로 묶을 수 있습니다.

## 1. 구조체 정의

`struct` 키워드는 새로운 복합 타입을 도입합니다. 내부의 각 데이터 조각을 **필드(field)** 또는 **멤버(member)**라고 합니다.

```c
struct Point {
    double x;
    double y;
};

struct Student {
    char name[50];
    int id;
    double gpa;
};
```

구조체 타입의 변수를 선언하고 초기화할 수 있습니다:

```c
#include <stdio.h>

struct Point {
    double x;
    double y;
};

int main(void) {
    /* Declaration and initialization */
    struct Point origin = {0.0, 0.0};
    struct Point p1 = {3.5, -2.1};

    /* Declaration then assignment (field by field) */
    struct Point p2;
    p2.x = 1.0;
    p2.y = 4.0;

    printf("p1 = (%.1f, %.1f)\n", p1.x, p1.y);
    return 0;
}
```

---

## 2. 멤버 접근

구조체 변수의 멤버에 접근하려면 **점 연산자**(`.`)를 사용합니다.

```c
#include <stdio.h>
#include <string.h>

struct Student {
    char name[50];
    int id;
    double gpa;
};

int main(void) {
    struct Student s;
    strcpy(s.name, "Alice");
    s.id = 1001;
    s.gpa = 3.85;

    printf("Name: %s, ID: %d, GPA: %.2f\n", s.name, s.id, s.gpa);
    return 0;
}
```

### C99 지정 초기화자 (Designated Initializers)

초기화 시 필드 이름을 지정할 수 있으며, 순서는 상관없습니다:

```c
struct Student s = {
    .name = "Bob",
    .gpa = 3.92,
    .id = 1002
};
```

이 방식은 구조체에 필드가 많을 때 코드를 자체 문서화하는 데 특히 유용합니다.

### 구조체 대입

하나의 구조체를 다른 구조체에 대입하면 모든 멤버가 복사됩니다:

```c
struct Point a = {1.0, 2.0};
struct Point b = a;  /* b.x = 1.0, b.y = 2.0 — full copy */
```

**주의**: 구조체에 포인터가 포함되어 있으면 복사는 **얕은 복사(shallow copy)**가 됩니다 — 두 구조체가 같은 메모리를 가리킵니다.

---

## 3. typedef

`struct Point`를 매번 작성하는 것은 번거롭습니다. `typedef` 키워드로 별칭을 만들 수 있습니다.

```c
typedef struct {
    double x;
    double y;
} Point;

/* Now you can write: */
Point p = {1.0, 2.0};
```

자기 참조 구조체(예: 연결 리스트)에 대한 일반적인 관례는 구조체 태그에 이름을 붙이는 것입니다:

```c
typedef struct Node {
    int data;
    struct Node *next;   /* must use "struct Node" here, not "Node" */
} Node;
```

| 스타일 | 선언 | 사용법 |
|-------|------------|-------|
| typedef 없이 | `struct Point { ... };` | `struct Point p;` |
| typedef 사용 | `typedef struct { ... } Point;` | `Point p;` |
| 태그와 typedef 모두 | `typedef struct Point { ... } Point;` | 어느 형태든 가능 |

---

## 4. 구조체와 포인터

구조체에 대한 포인터가 있을 때는 **화살표 연산자**(`->`)로 멤버에 접근합니다.

```c
#include <stdio.h>
#include <math.h>

typedef struct {
    double x;
    double y;
} Point;

double distance(const Point *a, const Point *b) {
    double dx = a->x - b->x;   /* equivalent to (*a).x - (*b).x */
    double dy = a->y - b->y;
    return sqrt(dx * dx + dy * dy);
}

int main(void) {
    Point p1 = {0.0, 0.0};
    Point p2 = {3.0, 4.0};

    printf("Distance: %.2f\n", distance(&p1, &p2));  /* 5.00 */
    return 0;
}
```

화살표 연산자 `p->member`는 `(*p).member`의 문법적 편의(syntactic sugar)입니다. `.`가 `*`보다 우선순위가 높기 때문에 괄호가 필요합니다.

### 함수에 구조체 전달하기

| 방법 | 문법 | 데이터 복사? | 원본 수정 가능? |
|--------|--------|-------------|---------------------|
| 값 전달 | `void f(Point p)` | 예 — 전체 복사 | 아니오 |
| 포인터 전달 | `void f(Point *p)` | 아니오 — 주소만 | 예 |
| const 포인터 전달 | `void f(const Point *p)` | 아니오 | 아니오 (읽기 전용) |

작은 구조체(2-3개 필드)의 경우 값 전달이 괜찮습니다. 큰 구조체의 경우 비용이 큰 복사를 피하기 위해 `const` 포인터로 전달합니다.

---

## 5. 중첩 구조체

구조체는 다른 구조체를 포함하여 계층적 데이터를 자연스럽게 모델링할 수 있습니다.

```c
#include <stdio.h>

typedef struct {
    int day;
    int month;
    int year;
} Date;

typedef struct {
    char name[50];
    int id;
    Date hire_date;
    Date birth_date;
} Employee;

int main(void) {
    Employee emp = {
        .name = "Alice",
        .id = 42,
        .hire_date = {15, 3, 2023},
        .birth_date = {.day = 10, .month = 7, .year = 1995}
    };

    printf("%s was hired on %02d/%02d/%04d\n",
           emp.name,
           emp.hire_date.day,
           emp.hire_date.month,
           emp.hire_date.year);

    return 0;
}
```

포인터가 있는 경우: `emp_ptr->hire_date.day` (첫 번째 레벨은 화살표, 중첩 구조체는 점 연산자 — `hire_date`는 그 자체가 포인터가 아니기 때문).

---

## 6. 공용체 (Unions)

**공용체(union)**는 구조체처럼 보이지만 모든 멤버가 **같은 메모리를 공유**합니다. 공용체의 크기는 가장 큰 멤버의 크기와 같습니다. 주어진 시점에 하나의 멤버만 유효한 값을 가집니다.

```c
#include <stdio.h>

union Data {
    int i;
    float f;
    char str[20];
};

int main(void) {
    union Data d;

    printf("Size of union: %zu bytes\n", sizeof(d));  /* 20 — size of str */

    d.i = 42;
    printf("d.i = %d\n", d.i);   /* 42 */

    d.f = 3.14f;
    printf("d.f = %.2f\n", d.f); /* 3.14 */
    printf("d.i = %d\n", d.i);   /* garbage — overwritten by d.f */

    return 0;
}
```

### 태그된 공용체 패턴 (Tagged Union Pattern)

어떤 멤버가 현재 유효한지 알기 위해, 공용체에 열거형 "태그"를 결합합니다:

```c
#include <stdio.h>

typedef enum { VAL_INT, VAL_FLOAT, VAL_STRING } ValueType;

typedef struct {
    ValueType type;
    union {
        int i;
        float f;
        char s[32];
    } data;
} Value;

void print_value(const Value *v) {
    switch (v->type) {
        case VAL_INT:    printf("int: %d\n", v->data.i);    break;
        case VAL_FLOAT:  printf("float: %.2f\n", v->data.f); break;
        case VAL_STRING: printf("string: %s\n", v->data.s);  break;
    }
}

int main(void) {
    Value v1 = {.type = VAL_INT, .data.i = 42};
    Value v2 = {.type = VAL_STRING, .data.s = "Hello"};

    print_value(&v1);  /* int: 42 */
    print_value(&v2);  /* string: Hello */
    return 0;
}
```

| 특성 | struct | union |
|---------|--------|-------|
| 메모리 | 모든 멤버 크기의 합 (+ 패딩) | 가장 큰 멤버의 크기 |
| 활성 멤버 | 모두 동시에 | 한 번에 하나 |
| 사용 사례 | 관련 데이터 그룹화 | 변형/다형적 데이터 |

---

## 7. 열거형 (Enumerations)

`enum`은 이름이 있는 정수 상수의 집합을 정의합니다.

```c
#include <stdio.h>

enum Direction { NORTH, EAST, SOUTH, WEST };
/* NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3 */

enum HttpStatus {
    HTTP_OK         = 200,
    HTTP_NOT_FOUND  = 404,
    HTTP_SERVER_ERR = 500
};

int main(void) {
    enum Direction dir = NORTH;

    switch (dir) {
        case NORTH: printf("Going north\n"); break;
        case EAST:  printf("Going east\n");  break;
        case SOUTH: printf("Going south\n"); break;
        case WEST:  printf("Going west\n");  break;
    }

    printf("HTTP OK = %d\n", HTTP_OK);  /* 200 */
    return 0;
}
```

열거형은 타입 안전한 문서화입니다. 코드 곳곳에 매직 넘버를 흩뿌리는 대신 의미 있는 이름을 부여합니다.

| 특성 | 설명 |
|---------|-------------|
| 기본값 | 0부터 시작, 자동 증가 |
| 명시적 값 | `= value`로 할당 |
| 기본 타입 | `int` (표준 C에서) |
| 스코프 | 전역 (열거형 이름에 한정되지 않음) |

---

## 8. 비트 필드 (Bit Fields)

비트 필드를 사용하면 구조체 멤버가 차지할 정확한 비트 수를 지정할 수 있습니다. 플래그, 하드웨어 레지스터 맵, 메모리가 제한된 환경에서 유용합니다.

```c
#include <stdio.h>

typedef struct {
    unsigned int is_active : 1;   /* 1 bit: 0 or 1 */
    unsigned int priority  : 3;   /* 3 bits: 0-7 */
    unsigned int category  : 4;   /* 4 bits: 0-15 */
} TaskFlags;

int main(void) {
    TaskFlags task = {
        .is_active = 1,
        .priority = 5,
        .category = 12
    };

    printf("Active: %u, Priority: %u, Category: %u\n",
           task.is_active, task.priority, task.category);

    printf("Size of TaskFlags: %zu bytes\n", sizeof(TaskFlags));
    /* Likely 4 bytes — the compiler packs bits into an unsigned int */

    return 0;
}
```

### 하드웨어 레지스터 예시

```c
typedef struct {
    unsigned int enable     : 1;
    unsigned int mode       : 2;
    unsigned int interrupt  : 1;
    unsigned int reserved   : 4;
} ControlRegister;
```

**이식성 참고사항**:
- 바이트 내 비트 필드의 순서는 구현 정의(implementation-defined)입니다 (컴파일러와 아키텍처에 따라 다를 수 있음).
- 비트 필드 멤버의 주소를 취할 수 없습니다 (`&task.priority`는 불법).
- 비트 필드는 컴파일러 간 이식성이 중요하지 않은 내부 데이터 구조에 가장 적합합니다. 와이어 프로토콜의 경우 명시적 비트 연산을 대신 사용하세요.

---

## 연습문제

**연습문제 1 — Rectangle 구조체**: `width`와 `height` (모두 `double`)를 가진 `Rectangle` 구조체를 정의하세요. `double area(const Rectangle *r)` 함수와 `double perimeter(const Rectangle *r)` 함수를 작성하세요. 여러 사각형을 만들어 면적과 둘레를 출력하세요.

**연습문제 2 — 학생 기록**: 이름, ID, 5개의 성적 배열을 가진 `Student` 구조체를 정의하세요. `const Student *`를 받아 평균 성적을 반환하는 함수를 작성하세요. 3명의 학생 배열을 만들어 보고서를 출력하세요.

**연습문제 3 — 태그된 공용체 계산기**: `int` 또는 `double`을 담을 수 있는 태그된 공용체 `Number`를 만드세요. `void print_number(const Number *n)` 함수와 두 Number를 더하는 `Number add_numbers(const Number *a, const Number *b)` 함수를 작성하세요 (어느 쪽이든 double이면 double로 승격).

**연습문제 4 — Color 열거형과 구조체**: RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA를 가진 `Color` 열거형을 정의하세요. `int x`, `int y`, `Color color`를 가진 `Pixel` 구조체를 정의하세요. 색상 이름(숫자가 아닌)으로 픽셀 정보를 출력하는 함수를 작성하세요.

**연습문제 5 — 패킹된 플래그**: 소유자, 그룹, 기타에 대한 읽기, 쓰기, 실행 비트를 가진 비트 필드 구조체 `FilePermissions`를 정의하세요 (총 9비트). 권한을 설정, 해제, `rwxrwxrwx` 형식(`ls -l`과 같이)으로 표시하는 함수를 작성하세요.

---

## 다음 단계

이제 실세계 엔티티를 모델링하는 커스텀 데이터 타입을 만들 수 있습니다. 다음 레슨 [동적 메모리](./09_Dynamic_Memory.md)에서는 컴파일 타임에 크기를 알 수 없는 데이터 구조를 구축하는 데 필수적인, 런타임에 힙에 구조체와 배열을 할당하는 방법을 배웁니다.
