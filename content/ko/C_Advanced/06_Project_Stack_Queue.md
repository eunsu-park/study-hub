# 프로젝트: 스택과 큐

**이전**: [프로젝트: 연결 리스트](./05_Project_Linked_List.md) | **다음**: [프로젝트: 해시 테이블](./07_Project_Hash_Table.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LIFO(스택)와 FIFO(큐) 순서를 구분하고 각각의 실생활 예시를 식별할 수 있다
2. push, pop, peek, isEmpty, isFull 연산을 갖춘 고정 크기 배열 기반 스택을 구현할 수 있다
3. head 삽입을 사용한 O(1) push와 pop을 갖춘 연결 리스트 기반 스택을 구현할 수 있다
4. 모듈러 산술로 배열 슬롯을 재사용하여 공간 낭비를 방지하는 원형 큐를 구축할 수 있다
5. front와 rear 포인터를 사용한 O(1) enqueue와 dequeue를 갖춘 연결 리스트 기반 큐를 구현할 수 있다
6. 스택을 적용하여 여는 괄호와 닫는 괄호를 짝짓는 괄호 검사 문제를 해결할 수 있다
7. 스택을 사용하여 산술 표현식을 평가하는 후위 표기법 계산기를 구축할 수 있다

---

스택과 큐는 가장 단순한 두 가지 추상 자료형이지만 어디에나 등장합니다: 프로그램을 실행하는 함수 호출 스택, 에디터의 실행 취소 버퍼, 운영체제의 인쇄 큐, 그래프 알고리즘의 BFS 프론티어. C에서 이들의 구현을 마스터하면 자료 구조 자체뿐만 아니라, 일관된 오류 처리를 갖춘 깔끔하고 재사용 가능한 ADT 인터페이스를 작성하는 규율도 배우게 됩니다.

## 스택(Stack)

### 개념: LIFO (Last In, First Out)

```
마지막에 들어간 것이 먼저 나옴 (접시 쌓기와 같음)

Push 3 -> Push 7 -> Push 1 -> Pop

+---+      +---+      +---+      +---+
|   |      |   |      | 1 | <-   |   |
+---+      +---+      +---+      +---+
|   |      | 7 |      | 7 |      | 7 |
+---+  ->  +---+  ->  +---+  ->  +---+
| 3 |      | 3 |      | 3 |      | 3 |
+---+      +---+      +---+      +---+
                                 Pop -> 1
```

### 주요 연산

| 연산 | 설명 | 시간 복잡도 |
|------|------|------------|
| `push` | 꼭대기에 추가 | O(1) |
| `pop` | 꼭대기에서 제거 및 반환 | O(1) |
| `peek/top` | 꼭대기 값 확인 | O(1) |
| `isEmpty` | 비어있는지 확인 | O(1) |

---

## 단계 1: 배열 기반 스택

```c
// array_stack.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 100

typedef struct {
    int data[MAX_SIZE];
    int top;
} Stack;

// 스택 초기화
void stack_init(Stack *s) {
    s->top = -1;
}

// 비어있는지 확인
bool stack_isEmpty(Stack *s) {
    return s->top == -1;
}

// 가득 찼는지 확인
bool stack_isFull(Stack *s) {
    return s->top == MAX_SIZE - 1;
}

// Push
bool stack_push(Stack *s, int value) {
    if (stack_isFull(s)) {
        printf("Stack Overflow!\n");
        return false;
    }
    s->data[++s->top] = value;
    return true;
}

// Pop
bool stack_pop(Stack *s, int *value) {
    if (stack_isEmpty(s)) {
        printf("Stack Underflow!\n");
        return false;
    }
    *value = s->data[s->top--];
    return true;
}

// Peek
bool stack_peek(Stack *s, int *value) {
    if (stack_isEmpty(s)) {
        return false;
    }
    *value = s->data[s->top];
    return true;
}

// 스택 출력
void stack_print(Stack *s) {
    printf("Stack (top=%d): ", s->top);
    for (int i = 0; i <= s->top; i++) {
        printf("%d ", s->data[i]);
    }
    printf("\n");
}

// 테스트
int main(void) {
    Stack s;
    stack_init(&s);

    printf("=== 배열 기반 스택 테스트 ===\n\n");

    // Push
    for (int i = 1; i <= 5; i++) {
        stack_push(&s, i * 10);
        stack_print(&s);
    }

    // Peek
    int top;
    stack_peek(&s, &top);
    printf("\nTop: %d\n", top);

    // Pop
    printf("\nPop 연산:\n");
    int value;
    while (stack_pop(&s, &value)) {
        printf("Popped: %d, ", value);
        stack_print(&s);
    }

    return 0;
}
```

---

## 단계 2: 연결 리스트 기반 스택

```c
// linked_stack.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

typedef struct {
    Node *top;
    int size;
} LinkedStack;

// 생성
LinkedStack* lstack_create(void) {
    LinkedStack *s = malloc(sizeof(LinkedStack));
    if (s) {
        s->top = NULL;
        s->size = 0;
    }
    return s;
}

// 소멸
void lstack_destroy(LinkedStack *s) {
    Node *current = s->top;
    while (current) {
        Node *next = current->next;
        free(current);
        current = next;
    }
    free(s);
}

bool lstack_isEmpty(LinkedStack *s) {
    return s->top == NULL;
}

// Push - O(1)
bool lstack_push(LinkedStack *s, int value) {
    Node *node = malloc(sizeof(Node));
    if (!node) return false;

    node->data = value;
    node->next = s->top;
    s->top = node;
    s->size++;
    return true;
}

// Pop - O(1)
bool lstack_pop(LinkedStack *s, int *value) {
    if (lstack_isEmpty(s)) return false;

    Node *node = s->top;
    *value = node->data;
    s->top = node->next;
    free(node);
    s->size--;
    return true;
}

// Peek - O(1)
bool lstack_peek(LinkedStack *s, int *value) {
    if (lstack_isEmpty(s)) return false;
    *value = s->top->data;
    return true;
}

void lstack_print(LinkedStack *s) {
    printf("Stack (size=%d): ", s->size);
    Node *current = s->top;
    while (current) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("(top)\n");
}

int main(void) {
    LinkedStack *s = lstack_create();

    printf("=== 연결 리스트 기반 스택 ===\n\n");

    for (int i = 1; i <= 5; i++) {
        lstack_push(s, i * 10);
        lstack_print(s);
    }

    int value;
    while (lstack_pop(s, &value)) {
        printf("Popped: %d\n", value);
    }

    lstack_destroy(s);
    return 0;
}
```

---

## 단계 3: 스택 응용 - 괄호 검사

```c
// bracket_check.c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define MAX_SIZE 100

typedef struct {
    char data[MAX_SIZE];
    int top;
} CharStack;

void stack_init(CharStack *s) { s->top = -1; }
bool stack_isEmpty(CharStack *s) { return s->top == -1; }
void stack_push(CharStack *s, char c) { s->data[++s->top] = c; }
char stack_pop(CharStack *s) { return s->data[s->top--]; }
char stack_peek(CharStack *s) { return s->data[s->top]; }

// 짝이 맞는지 확인
bool isMatchingPair(char open, char close) {
    return (open == '(' && close == ')') ||
           (open == '{' && close == '}') ||
           (open == '[' && close == ']');
}

// 괄호 검사
bool checkBrackets(const char *expr) {
    CharStack s;
    stack_init(&s);

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        // 여는 괄호
        if (c == '(' || c == '{' || c == '[') {
            stack_push(&s, c);
        }
        // 닫는 괄호
        else if (c == ')' || c == '}' || c == ']') {
            if (stack_isEmpty(&s)) {
                printf("오류: '%c' (위치 %d) - 짝이 없음\n", c, i);
                return false;
            }

            char open = stack_pop(&s);
            if (!isMatchingPair(open, c)) {
                printf("오류: '%c'와 '%c' 불일치 (위치 %d)\n", open, c, i);
                return false;
            }
        }
    }

    if (!stack_isEmpty(&s)) {
        printf("오류: 닫히지 않은 괄호가 있음\n");
        return false;
    }

    return true;
}

int main(void) {
    const char *tests[] = {
        "(a + b) * (c - d)",
        "((a + b) * c",
        "{[()]}",
        "{[(])}",
        "((()))",
        ")("
    };

    int n = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < n; i++) {
        printf("\n검사: \"%s\"\n", tests[i]);
        if (checkBrackets(tests[i])) {
            printf("결과: 유효한 괄호\n");
        } else {
            printf("결과: 유효하지 않은 괄호\n");
        }
    }

    return 0;
}
```

---

## 큐(Queue)

### 개념: FIFO (First In, First Out)

```
먼저 들어간 것이 먼저 나옴 (줄 서기와 같음)

Enqueue 3 -> Enqueue 7 -> Enqueue 1 -> Dequeue

front                     rear
  |                        |
+---+---+---+---+      +---+---+---+---+
| 3 |   |   |   |  ->  | 3 | 7 | 1 |   |
+---+---+---+---+      +---+---+---+---+

Dequeue -> 3
    front             rear
      |                |
+---+---+---+---+
|   | 7 | 1 |   |
+---+---+---+---+
```

### 주요 연산

| 연산 | 설명 | 시간 복잡도 |
|------|------|------------|
| `enqueue` | 뒤에 추가 | O(1) |
| `dequeue` | 앞에서 제거 | O(1) |
| `front` | 앞의 값 확인 | O(1) |
| `isEmpty` | 비어있는지 확인 | O(1) |

---

## 단계 4: 원형 큐 (배열 기반)

```c
// circular_queue.c
#include <stdio.h>
#include <stdbool.h>

#define MAX_SIZE 5

typedef struct {
    int data[MAX_SIZE];
    int front;
    int rear;
    int count;
} CircularQueue;

void queue_init(CircularQueue *q) {
    q->front = 0;
    q->rear = -1;
    q->count = 0;
}

bool queue_isEmpty(CircularQueue *q) {
    return q->count == 0;
}

bool queue_isFull(CircularQueue *q) {
    return q->count == MAX_SIZE;
}

bool queue_enqueue(CircularQueue *q, int value) {
    if (queue_isFull(q)) {
        printf("큐가 가득 참!\n");
        return false;
    }

    q->rear = (q->rear + 1) % MAX_SIZE;  // 원형 래핑
    q->data[q->rear] = value;
    q->count++;
    return true;
}

bool queue_dequeue(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        printf("큐가 비어있음!\n");
        return false;
    }

    *value = q->data[q->front];
    q->front = (q->front + 1) % MAX_SIZE;  // 원형 래핑
    q->count--;
    return true;
}

bool queue_front(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) return false;
    *value = q->data[q->front];
    return true;
}

void queue_print(CircularQueue *q) {
    printf("Queue (count=%d): [", q->count);
    if (!queue_isEmpty(q)) {
        int i = q->front;
        for (int c = 0; c < q->count; c++) {
            printf("%d", q->data[i]);
            if (c < q->count - 1) printf(", ");
            i = (i + 1) % MAX_SIZE;
        }
    }
    printf("] (front=%d, rear=%d)\n", q->front, q->rear);
}

int main(void) {
    CircularQueue q;
    queue_init(&q);

    printf("=== 원형 큐 테스트 ===\n\n");

    // Enqueue
    for (int i = 1; i <= 5; i++) {
        queue_enqueue(&q, i * 10);
        queue_print(&q);
    }

    // 2개 Dequeue
    int value;
    printf("\n2개 Dequeue:\n");
    queue_dequeue(&q, &value);
    printf("Dequeued: %d, ", value);
    queue_print(&q);

    queue_dequeue(&q, &value);
    printf("Dequeued: %d, ", value);
    queue_print(&q);

    // 다시 Enqueue (원형 동작 테스트)
    printf("\n2개 더 Enqueue:\n");
    queue_enqueue(&q, 60);
    queue_print(&q);
    queue_enqueue(&q, 70);
    queue_print(&q);

    return 0;
}
```

---

## 단계 5: 연결 리스트 기반 큐

```c
// linked_queue.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

typedef struct {
    Node *front;
    Node *rear;
    int size;
} LinkedQueue;

LinkedQueue* lqueue_create(void) {
    LinkedQueue *q = malloc(sizeof(LinkedQueue));
    if (q) {
        q->front = NULL;
        q->rear = NULL;
        q->size = 0;
    }
    return q;
}

void lqueue_destroy(LinkedQueue *q) {
    Node *current = q->front;
    while (current) {
        Node *next = current->next;
        free(current);
        current = next;
    }
    free(q);
}

bool lqueue_isEmpty(LinkedQueue *q) {
    return q->front == NULL;
}

// Enqueue - O(1)
bool lqueue_enqueue(LinkedQueue *q, int value) {
    Node *node = malloc(sizeof(Node));
    if (!node) return false;

    node->data = value;
    node->next = NULL;

    if (q->rear == NULL) {
        q->front = q->rear = node;
    } else {
        q->rear->next = node;
        q->rear = node;
    }
    q->size++;
    return true;
}

// Dequeue - O(1)
bool lqueue_dequeue(LinkedQueue *q, int *value) {
    if (lqueue_isEmpty(q)) return false;

    Node *node = q->front;
    *value = node->data;
    q->front = node->next;

    if (q->front == NULL) {
        q->rear = NULL;
    }

    free(node);
    q->size--;
    return true;
}

void lqueue_print(LinkedQueue *q) {
    printf("Queue (size=%d): front -> ", q->size);
    Node *current = q->front;
    while (current) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("<- rear\n");
}

int main(void) {
    LinkedQueue *q = lqueue_create();

    printf("=== 연결 리스트 기반 큐 ===\n\n");

    for (int i = 1; i <= 5; i++) {
        lqueue_enqueue(q, i * 10);
        lqueue_print(q);
    }

    printf("\n");
    int value;
    while (lqueue_dequeue(q, &value)) {
        printf("Dequeued: %d\n", value);
    }

    lqueue_destroy(q);
    return 0;
}
```

---

## 단계 6: 스택 응용 - 후위 표기법 계산기

```c
// postfix_calc.c
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#define MAX_SIZE 100

typedef struct {
    double data[MAX_SIZE];
    int top;
} Stack;

void stack_init(Stack *s) { s->top = -1; }
void stack_push(Stack *s, double v) { s->data[++s->top] = v; }
double stack_pop(Stack *s) { return s->data[s->top--]; }

// 후위 표기법 계산
// 예: "3 4 + 5 *" = (3 + 4) * 5 = 35
double evaluatePostfix(const char *expr) {
    Stack s;
    stack_init(&s);

    char *str = strdup(expr);
    char *token = strtok(str, " ");

    while (token) {
        if (isdigit(token[0]) || (token[0] == '-' && isdigit(token[1]))) {
            // 숫자
            stack_push(&s, atof(token));
        } else {
            // 연산자
            double b = stack_pop(&s);
            double a = stack_pop(&s);
            double result;

            switch (token[0]) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': result = a / b; break;
                default:
                    printf("알 수 없는 연산자: %s\n", token);
                    free(str);
                    return 0;
            }
            stack_push(&s, result);
        }
        token = strtok(NULL, " ");
    }

    double result = stack_pop(&s);
    free(str);
    return result;
}

int main(void) {
    const char *expressions[] = {
        "3 4 +",           // 3 + 4 = 7
        "3 4 + 5 *",       // (3 + 4) * 5 = 35
        "10 2 / 3 +",      // 10 / 2 + 3 = 8
        "5 1 2 + 4 * + 3 -" // 5 + ((1 + 2) * 4) - 3 = 14
    };

    int n = sizeof(expressions) / sizeof(expressions[0]);

    printf("=== 후위 표기법 계산기 ===\n\n");

    for (int i = 0; i < n; i++) {
        printf("표현식: %s\n", expressions[i]);
        printf("결과: %.2f\n\n", evaluatePostfix(expressions[i]));
    }

    return 0;
}
```

---

## 컴파일 및 실행

```bash
gcc -Wall -std=c11 array_stack.c -o array_stack
gcc -Wall -std=c11 bracket_check.c -o bracket_check
gcc -Wall -std=c11 circular_queue.c -o circular_queue
gcc -Wall -std=c11 postfix_calc.c -o postfix_calc
```

---

## 요약

| 자료 구조 | 특성 | 활용 |
|----------|------|------|
| 스택(Stack) | LIFO | 괄호 검사, 함수 호출, Undo |
| 큐(Queue) | FIFO | 작업 대기열, BFS, 버퍼 |
| 원형 큐(Circular Queue) | 공간 재사용 | 고정 크기 버퍼 |

### 스택 vs 큐

| 비교 | 스택 | 큐 |
|------|------|-----|
| 원리 | LIFO | FIFO |
| 삽입 | push (꼭대기) | enqueue (뒤) |
| 제거 | pop (꼭대기) | dequeue (앞) |

---

## 연습문제

1. **중위 -> 후위 변환**: `(3 + 4) * 5` -> `3 4 + 5 *`

2. **덱(Deque) 구현**: 양쪽 끝에서 삽입/삭제가 가능한 자료 구조

3. **우선순위 큐**: 값의 우선순위에 따라 정렬되는 큐

4. **Min-Stack**: push와 pop 외에 O(1) 시간에 `getMin()`을 지원하는 스택을 구현하세요

5. **두 스택으로 큐 구현**: 두 개의 스택만 사용하여 큐 ADT를 구현하세요

---

## 다음 단계

[프로젝트: 해시 테이블](./07_Project_Hash_Table.md) -> 해시 테이블을 배워봅시다!
