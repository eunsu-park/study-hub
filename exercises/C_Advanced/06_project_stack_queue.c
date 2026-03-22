/*
 * Exercises for Lesson 09: Project Stack Queue
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex09 09_project_stack_queue.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define STACK_CAP 64

/* === Exercise 1: Array-Based Stack === */
/* Problem: Implement a stack using a fixed-size array with error handling. */

typedef struct {
    int data[STACK_CAP];
    int top;  /* Index of the top element; -1 when empty */
} Stack;

void stack_init(Stack *s) { s->top = -1; }
int  stack_empty(const Stack *s) { return s->top == -1; }
int  stack_full(const Stack *s) { return s->top == STACK_CAP - 1; }
int  stack_size(const Stack *s) { return s->top + 1; }

int stack_push(Stack *s, int val) {
    /*
     * Push: O(1) -- simply increment top and store.
     * Must check for overflow (stack full).
     */
    if (stack_full(s)) return -1;
    s->data[++s->top] = val;
    return 0;
}

int stack_pop(Stack *s, int *val) {
    /* Pop: O(1) -- read top and decrement. */
    if (stack_empty(s)) return -1;
    *val = s->data[s->top--];
    return 0;
}

int stack_peek(const Stack *s, int *val) {
    if (stack_empty(s)) return -1;
    *val = s->data[s->top];
    return 0;
}

void exercise_1(void) {
    printf("=== Exercise 1: Array-Based Stack ===\n");

    Stack s;
    stack_init(&s);

    printf("Push 10, 20, 30, 40, 50:\n");
    for (int v = 10; v <= 50; v += 10) {
        stack_push(&s, v);
        int top;
        stack_peek(&s, &top);
        printf("  push(%d) -> top=%d, size=%d\n", v, top, stack_size(&s));
    }

    printf("\nPop all elements (LIFO order):\n");
    int val;
    while (stack_pop(&s, &val) == 0) {
        printf("  pop() -> %d, size=%d\n", val, stack_size(&s));
    }

    printf("\nEdge cases:\n");
    printf("  Pop from empty: %s\n",
           stack_pop(&s, &val) == -1 ? "ERROR (correct)" : "unexpected");
    printf("  Peek on empty:  %s\n",
           stack_peek(&s, &val) == -1 ? "ERROR (correct)" : "unexpected");

    /* Fill to capacity */
    for (int i = 0; i < STACK_CAP; i++) stack_push(&s, i);
    printf("  Push to full stack: %s\n",
           stack_push(&s, 999) == -1 ? "OVERFLOW (correct)" : "unexpected");
}

/* === Exercise 2: Linked Queue === */
/* Problem: Implement a FIFO queue using a linked list. */

typedef struct QNode {
    int data;
    struct QNode *next;
} QNode;

typedef struct {
    QNode *front;
    QNode *rear;
    int count;
} Queue;

void queue_init(Queue *q) { q->front = q->rear = NULL; q->count = 0; }
int  queue_empty(const Queue *q) { return q->front == NULL; }

int queue_enqueue(Queue *q, int val) {
    /*
     * Enqueue at rear: O(1) with tail pointer.
     * Without tail pointer, enqueue would be O(n) -- a common
     * beginner mistake when implementing queues with linked lists.
     */
    QNode *node = malloc(sizeof(QNode));
    if (!node) return -1;
    node->data = val;
    node->next = NULL;

    if (q->rear) q->rear->next = node;
    else q->front = node;
    q->rear = node;
    q->count++;
    return 0;
}

int queue_dequeue(Queue *q, int *val) {
    /* Dequeue from front: O(1) */
    if (queue_empty(q)) return -1;

    QNode *tmp = q->front;
    *val = tmp->data;
    q->front = tmp->next;
    if (!q->front) q->rear = NULL; /* Queue became empty */
    free(tmp);
    q->count--;
    return 0;
}

void queue_free(Queue *q) {
    int val;
    while (queue_dequeue(q, &val) == 0);
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Linked Queue ===\n");

    Queue q;
    queue_init(&q);

    printf("Enqueue 10, 20, 30, 40, 50:\n");
    for (int v = 10; v <= 50; v += 10) {
        queue_enqueue(&q, v);
        printf("  enqueue(%d) -> front=%d, count=%d\n",
               v, q.front->data, q.count);
    }

    printf("\nDequeue all elements (FIFO order):\n");
    int val;
    while (queue_dequeue(&q, &val) == 0) {
        printf("  dequeue() -> %d, count=%d\n", val, q.count);
    }

    printf("\nInterleaved operations:\n");
    queue_enqueue(&q, 1);
    queue_enqueue(&q, 2);
    queue_dequeue(&q, &val);
    printf("  enqueue(1), enqueue(2), dequeue() -> %d\n", val);
    queue_enqueue(&q, 3);
    queue_dequeue(&q, &val);
    printf("  enqueue(3), dequeue() -> %d\n", val);
    queue_dequeue(&q, &val);
    printf("  dequeue() -> %d\n", val);

    queue_free(&q);
}

/* === Exercise 3: Bracket Matching === */
/* Problem: Use a stack to check if brackets are balanced. */

int is_matching(char open, char close) {
    return (open == '(' && close == ')') ||
           (open == '[' && close == ']') ||
           (open == '{' && close == '}');
}

typedef struct {
    char data[256];
    int top;
} CharStack;

void cstack_init(CharStack *s) { s->top = -1; }
void cstack_push(CharStack *s, char c) { s->data[++s->top] = c; }
int  cstack_pop(CharStack *s, char *c) {
    if (s->top < 0) return -1;
    *c = s->data[s->top--];
    return 0;
}

int check_brackets(const char *expr) {
    /*
     * Algorithm:
     * 1. For each character:
     *    - If opening bracket, push onto stack
     *    - If closing bracket, pop from stack and check match
     *    - If stack is empty when popping, unmatched closing bracket
     * 2. After processing, stack should be empty
     *    - Non-empty stack means unmatched opening brackets
     *
     * Time: O(n), Space: O(n) worst case (all opening brackets)
     */
    CharStack s;
    cstack_init(&s);

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];
        if (c == '(' || c == '[' || c == '{') {
            cstack_push(&s, c);
        } else if (c == ')' || c == ']' || c == '}') {
            char top;
            if (cstack_pop(&s, &top) == -1) return 0; /* Extra closing */
            if (!is_matching(top, c)) return 0;         /* Mismatch */
        }
    }
    return s.top == -1; /* Should be empty */
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Bracket Matching ===\n");

    const char *tests[] = {
        "()",
        "()[]{}",
        "{[()]}",
        "((()))",
        "({[]})",
        "(]",           /* Mismatch */
        "([)]",         /* Interleaved */
        "(((",          /* Unclosed */
        ")))",          /* Extra closing */
        "",             /* Empty */
        "a + (b * [c - {d / e}])",
        "func(arr[i], map{key})",
        "((())",        /* Missing one closing */
    };
    int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    printf("%-30s  %-8s\n", "Expression", "Balanced");
    printf("------------------------------  --------\n");

    for (int i = 0; i < n_tests; i++) {
        printf("%-30s  %s\n", strlen(tests[i]) > 0 ? tests[i] : "(empty)",
               check_brackets(tests[i]) ? "YES" : "NO");
    }
}

/* === Exercise 4: Postfix Expression Evaluator === */
/* Problem: Evaluate postfix (reverse Polish notation) expressions using a stack. */

int evaluate_postfix(const char *expr, double *result) {
    /*
     * Postfix evaluation algorithm:
     * - Numbers: push onto stack
     * - Operators: pop two operands, compute, push result
     * - At end, stack should contain exactly one value
     *
     * Postfix eliminates the need for parentheses and operator precedence.
     * "3 4 + 2 *" = (3 + 4) * 2 = 14
     *
     * Time: O(n), Space: O(n)
     */
    double stack[64];
    int top = -1;

    char buf[256];
    strncpy(buf, expr, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *token = strtok(buf, " ");
    while (token) {
        if (isdigit((unsigned char)token[0]) ||
            (token[0] == '-' && isdigit((unsigned char)token[1]))) {
            /* Number: push */
            if (top >= 62) return -1; /* Stack overflow */
            stack[++top] = atof(token);
        } else if (strlen(token) == 1 && strchr("+-*/", token[0])) {
            /* Operator: need at least 2 operands */
            if (top < 1) return -1;
            double b = stack[top--];
            double a = stack[top--];

            switch (token[0]) {
                case '+': stack[++top] = a + b; break;
                case '-': stack[++top] = a - b; break;
                case '*': stack[++top] = a * b; break;
                case '/':
                    if (b == 0) return -2; /* Division by zero */
                    stack[++top] = a / b;
                    break;
            }
        } else {
            return -1; /* Invalid token */
        }
        token = strtok(NULL, " ");
    }

    if (top != 0) return -1; /* Should have exactly one result */
    *result = stack[0];
    return 0;
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Postfix Expression Evaluator ===\n");

    struct {
        const char *expr;
        const char *infix;
    } tests[] = {
        {"3 4 +",           "3 + 4"},
        {"3 4 + 2 *",       "(3 + 4) * 2"},
        {"5 1 2 + 4 * + 3 -", "5 + ((1 + 2) * 4) - 3"},
        {"2 3 * 4 5 * +",   "2*3 + 4*5"},
        {"10 2 /",          "10 / 2"},
        {"7 2 -",           "7 - 2"},
        {"15 7 1 1 + - / 3 * 2 1 1 + + -",
                            "((15/(7-(1+1)))*3)-(2+(1+1))"},
    };
    int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    printf("%-35s  %-25s  %-8s\n", "Postfix", "Infix", "Result");
    printf("-----------------------------------  -------------------------  --------\n");

    for (int i = 0; i < n_tests; i++) {
        double result;
        int status = evaluate_postfix(tests[i].expr, &result);

        printf("%-35s  %-25s  ", tests[i].expr, tests[i].infix);
        if (status == 0) printf("%.2f\n", result);
        else if (status == -2) printf("DIV/0\n");
        else printf("ERROR\n");
    }
}

/* === Exercise 5: Min-Stack === */
/* Problem: Stack that supports O(1) getMin in addition to push/pop. */

typedef struct {
    int data[STACK_CAP];
    int mins[STACK_CAP];  /* Auxiliary stack tracking minimums */
    int top;
} MinStack;

void minstack_init(MinStack *s) { s->top = -1; }

int minstack_push(MinStack *s, int val) {
    /*
     * Key insight: maintain a parallel stack of minimums.
     * When pushing, the new minimum is min(val, current_min).
     * When popping, the minimum automatically restores.
     *
     * This achieves O(1) for all operations at the cost of O(n) extra space.
     * Alternative: store (value, min_at_this_point) pairs.
     */
    if (s->top >= STACK_CAP - 1) return -1;
    s->top++;
    s->data[s->top] = val;
    if (s->top == 0) {
        s->mins[s->top] = val;
    } else {
        s->mins[s->top] = val < s->mins[s->top - 1] ? val : s->mins[s->top - 1];
    }
    return 0;
}

int minstack_pop(MinStack *s, int *val) {
    if (s->top < 0) return -1;
    *val = s->data[s->top--];
    return 0;
}

int minstack_getmin(const MinStack *s, int *val) {
    if (s->top < 0) return -1;
    *val = s->mins[s->top];
    return 0;
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Min-Stack ===\n");

    MinStack ms;
    minstack_init(&ms);

    int operations[] = {5, 3, 7, 1, 4, 2};
    int n_ops = (int)(sizeof(operations) / sizeof(operations[0]));

    printf("Push operations:\n");
    printf("%-10s  %-6s  %-6s\n", "Operation", "Top", "Min");
    printf("----------  ------  ------\n");

    for (int i = 0; i < n_ops; i++) {
        minstack_push(&ms, operations[i]);
        int top, min;
        top = ms.data[ms.top];
        minstack_getmin(&ms, &min);
        printf("push(%d)     %-6d  %-6d\n", operations[i], top, min);
    }

    printf("\nPop operations:\n");
    printf("%-10s  %-8s  %-6s\n", "Operation", "Popped", "New Min");
    printf("----------  --------  ------\n");

    int val, min;
    for (int i = 0; i < n_ops; i++) {
        minstack_pop(&ms, &val);
        if (ms.top >= 0) {
            minstack_getmin(&ms, &min);
            printf("pop()       %-8d  %-6d\n", val, min);
        } else {
            printf("pop()       %-8d  (empty)\n", val);
        }
    }

    printf("\nComplexity: push O(1), pop O(1), getMin O(1)\n");
    printf("Space overhead: O(n) for the auxiliary mins array.\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
