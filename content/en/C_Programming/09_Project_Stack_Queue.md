# Project 7: Stack and Queue

**Previous**: [Project 6: File Encryption Tool](./08_Project_File_Encryption.md) | **Next**: [Project 8: Hash Table](./10_Project_Hash_Table.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish LIFO (stack) and FIFO (queue) ordering and identify real-world examples of each
2. Implement a fixed-size array-based stack with push, pop, peek, isEmpty, and isFull operations
3. Implement a linked-list-based stack with O(1) push and pop using head insertion
4. Build a circular queue that reuses array slots via modular arithmetic to avoid wasted space
5. Implement a linked-list-based queue with O(1) enqueue and dequeue using front and rear pointers
6. Apply a stack to solve the bracket-matching problem by pairing opening and closing delimiters
7. Build a postfix-notation calculator that uses a stack to evaluate arithmetic expressions

---

Stacks and queues are the two simplest abstract data types, yet they appear everywhere: the function call stack that keeps your program running, the undo buffer in your editor, the print queue on your operating system, and the BFS frontier in graph algorithms. Mastering their implementation in C teaches you not only the data structures themselves but also the discipline of writing clean, reusable ADT interfaces with consistent error handling.

## Stack

### Concept: LIFO (Last In, First Out)

```
Last in is first out (like stacking plates)

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

### Main Operations

| Operation | Description | Time Complexity |
|-----------|-------------|-----------------|
| `push` | Add to top | O(1) |
| `pop` | Remove and return top | O(1) |
| `peek/top` | View top value | O(1) |
| `isEmpty` | Check if empty | O(1) |

---

## Step 1: Array-Based Stack

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

// Initialize stack
void stack_init(Stack *s) {
    s->top = -1;
}

// Check if empty
bool stack_isEmpty(Stack *s) {
    return s->top == -1;
}

// Check if full
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

// Print stack
void stack_print(Stack *s) {
    printf("Stack (top=%d): ", s->top);
    for (int i = 0; i <= s->top; i++) {
        printf("%d ", s->data[i]);
    }
    printf("\n");
}

// Test
int main(void) {
    Stack s;
    stack_init(&s);

    printf("=== Array-Based Stack Test ===\n\n");

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
    printf("\nPop operations:\n");
    int value;
    while (stack_pop(&s, &value)) {
        printf("Popped: %d, ", value);
        stack_print(&s);
    }

    return 0;
}
```

---

## Step 2: Linked List-Based Stack

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

// Create
LinkedStack* lstack_create(void) {
    LinkedStack *s = malloc(sizeof(LinkedStack));
    if (s) {
        s->top = NULL;
        s->size = 0;
    }
    return s;
}

// Destroy
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

    printf("=== Linked List-Based Stack ===\n\n");

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

## Step 3: Stack Application - Bracket Checking

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

// Check if matching pair
bool isMatchingPair(char open, char close) {
    return (open == '(' && close == ')') ||
           (open == '{' && close == '}') ||
           (open == '[' && close == ']');
}

// Check brackets
bool checkBrackets(const char *expr) {
    CharStack s;
    stack_init(&s);

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        // Opening bracket
        if (c == '(' || c == '{' || c == '[') {
            stack_push(&s, c);
        }
        // Closing bracket
        else if (c == ')' || c == '}' || c == ']') {
            if (stack_isEmpty(&s)) {
                printf("Error: '%c' (position %d) - no match\n", c, i);
                return false;
            }

            char open = stack_pop(&s);
            if (!isMatchingPair(open, c)) {
                printf("Error: '%c' and '%c' mismatch (position %d)\n", open, c, i);
                return false;
            }
        }
    }

    if (!stack_isEmpty(&s)) {
        printf("Error: Unclosed brackets exist\n");
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
        printf("\nChecking: \"%s\"\n", tests[i]);
        if (checkBrackets(tests[i])) {
            printf("Result: Valid brackets\n");
        } else {
            printf("Result: Invalid brackets\n");
        }
    }

    return 0;
}
```

---

## Queue

### Concept: FIFO (First In, First Out)

```
First in is first out (like standing in line)

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

### Main Operations

| Operation | Description | Time Complexity |
|-----------|-------------|-----------------|
| `enqueue` | Add to rear | O(1) |
| `dequeue` | Remove from front | O(1) |
| `front` | View front value | O(1) |
| `isEmpty` | Check if empty | O(1) |

---

## Step 4: Circular Queue (Array-Based)

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
        printf("Queue is full!\n");
        return false;
    }

    q->rear = (q->rear + 1) % MAX_SIZE;  // Circular wrap
    q->data[q->rear] = value;
    q->count++;
    return true;
}

bool queue_dequeue(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        printf("Queue is empty!\n");
        return false;
    }

    *value = q->data[q->front];
    q->front = (q->front + 1) % MAX_SIZE;  // Circular wrap
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

    printf("=== Circular Queue Test ===\n\n");

    // Enqueue
    for (int i = 1; i <= 5; i++) {
        queue_enqueue(&q, i * 10);
        queue_print(&q);
    }

    // Dequeue 2 items
    int value;
    printf("\nDequeue 2 items:\n");
    queue_dequeue(&q, &value);
    printf("Dequeued: %d, ", value);
    queue_print(&q);

    queue_dequeue(&q, &value);
    printf("Dequeued: %d, ", value);
    queue_print(&q);

    // Enqueue again (test circular behavior)
    printf("\nEnqueue 2 more:\n");
    queue_enqueue(&q, 60);
    queue_print(&q);
    queue_enqueue(&q, 70);
    queue_print(&q);

    return 0;
}
```

---

## Step 5: Linked List-Based Queue

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

    printf("=== Linked List-Based Queue ===\n\n");

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

## Step 6: Stack Application - Postfix Notation Calculator

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

// Postfix notation calculation
// Example: "3 4 + 5 *" = (3 + 4) * 5 = 35
double evaluatePostfix(const char *expr) {
    Stack s;
    stack_init(&s);

    char *str = strdup(expr);
    char *token = strtok(str, " ");

    while (token) {
        if (isdigit(token[0]) || (token[0] == '-' && isdigit(token[1]))) {
            // Number
            stack_push(&s, atof(token));
        } else {
            // Operator
            double b = stack_pop(&s);
            double a = stack_pop(&s);
            double result;

            switch (token[0]) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': result = a / b; break;
                default:
                    printf("Unknown operator: %s\n", token);
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

    printf("=== Postfix Notation Calculator ===\n\n");

    for (int i = 0; i < n; i++) {
        printf("Expression: %s\n", expressions[i]);
        printf("Result: %.2f\n\n", evaluatePostfix(expressions[i]));
    }

    return 0;
}
```

---

## Compile and Run

```bash
gcc -Wall -std=c11 array_stack.c -o array_stack
gcc -Wall -std=c11 bracket_check.c -o bracket_check
gcc -Wall -std=c11 circular_queue.c -o circular_queue
gcc -Wall -std=c11 postfix_calc.c -o postfix_calc
```

---

## Summary

| Data Structure | Characteristic | Applications |
|----------------|----------------|--------------|
| Stack | LIFO | Bracket checking, function calls, Undo |
| Queue | FIFO | Task queues, BFS, buffers |
| Circular Queue | Space reuse | Fixed-size buffers |

### Stack vs Queue

| Comparison | Stack | Queue |
|------------|-------|-------|
| Principle | LIFO | FIFO |
| Insert | push (top) | enqueue (rear) |
| Remove | pop (top) | dequeue (front) |

---

## Exercises

1. **Infix to Postfix Conversion**: `(3 + 4) * 5` -> `3 4 + 5 *`

2. **Deque Implementation**: Data structure that can insert/delete from both ends

3. **Priority Queue**: Queue sorted by value priority

---

## Next Step

[Project 8: Hash Table](./10_Project_Hash_Table.md) -> Let's learn about hash tables!
