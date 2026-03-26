/*
 * Stack and Queue
 * Stack, Queue, Deque, Monotonic Stack
 *
 * Fundamental data structures and their applications.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* =============================================================================
 * 1. Stack Implementation
 * ============================================================================= */

typedef struct {
    int* data;
    int top;
    int capacity;
} Stack;

Stack* stack_create(int capacity) {
    Stack* s = malloc(sizeof(Stack));
    s->data = malloc(capacity * sizeof(int));
    s->top = -1;
    s->capacity = capacity;
    return s;
}

void stack_free(Stack* s) {
    free(s->data);
    free(s);
}

bool stack_is_empty(Stack* s) {
    return s->top == -1;
}

bool stack_is_full(Stack* s) {
    return s->top == s->capacity - 1;
}

void stack_push(Stack* s, int value) {
    if (!stack_is_full(s)) {
        s->data[++s->top] = value;
    }
}

int stack_pop(Stack* s) {
    if (!stack_is_empty(s)) {
        return s->data[s->top--];
    }
    return -1;
}

int stack_peek(Stack* s) {
    if (!stack_is_empty(s)) {
        return s->data[s->top];
    }
    return -1;
}

/* =============================================================================
 * 2. Queue Implementation (Circular Queue)
 * ============================================================================= */

typedef struct {
    int* data;
    int front;
    int rear;
    int size;
    int capacity;
} Queue;

Queue* queue_create(int capacity) {
    Queue* q = malloc(sizeof(Queue));
    q->data = malloc(capacity * sizeof(int));
    q->front = 0;
    q->rear = -1;
    q->size = 0;
    q->capacity = capacity;
    return q;
}

void queue_free(Queue* q) {
    free(q->data);
    free(q);
}

bool queue_is_empty(Queue* q) {
    return q->size == 0;
}

bool queue_is_full(Queue* q) {
    return q->size == q->capacity;
}

void queue_enqueue(Queue* q, int value) {
    if (!queue_is_full(q)) {
        q->rear = (q->rear + 1) % q->capacity;
        q->data[q->rear] = value;
        q->size++;
    }
}

int queue_dequeue(Queue* q) {
    if (!queue_is_empty(q)) {
        int value = q->data[q->front];
        q->front = (q->front + 1) % q->capacity;
        q->size--;
        return value;
    }
    return -1;
}

int queue_front(Queue* q) {
    if (!queue_is_empty(q)) {
        return q->data[q->front];
    }
    return -1;
}

/* =============================================================================
 * 3. Deque Implementation (Double-ended Queue)
 * ============================================================================= */

typedef struct {
    int* data;
    int front;
    int rear;
    int size;
    int capacity;
} Deque;

Deque* deque_create(int capacity) {
    Deque* d = malloc(sizeof(Deque));
    d->data = malloc(capacity * sizeof(int));
    d->front = 0;
    d->rear = 0;
    d->size = 0;
    d->capacity = capacity;
    return d;
}

void deque_free(Deque* d) {
    free(d->data);
    free(d);
}

bool deque_is_empty(Deque* d) {
    return d->size == 0;
}

void deque_push_front(Deque* d, int value) {
    if (d->size < d->capacity) {
        d->front = (d->front - 1 + d->capacity) % d->capacity;
        d->data[d->front] = value;
        d->size++;
    }
}

void deque_push_back(Deque* d, int value) {
    if (d->size < d->capacity) {
        d->data[d->rear] = value;
        d->rear = (d->rear + 1) % d->capacity;
        d->size++;
    }
}

int deque_pop_front(Deque* d) {
    if (!deque_is_empty(d)) {
        int value = d->data[d->front];
        d->front = (d->front + 1) % d->capacity;
        d->size--;
        return value;
    }
    return -1;
}

int deque_pop_back(Deque* d) {
    if (!deque_is_empty(d)) {
        d->rear = (d->rear - 1 + d->capacity) % d->capacity;
        d->size--;
        return d->data[d->rear];
    }
    return -1;
}

int deque_front(Deque* d) {
    if (!deque_is_empty(d)) {
        return d->data[d->front];
    }
    return -1;
}

int deque_back(Deque* d) {
    if (!deque_is_empty(d)) {
        return d->data[(d->rear - 1 + d->capacity) % d->capacity];
    }
    return -1;
}

/* =============================================================================
 * 4. Parentheses Validation
 * ============================================================================= */

bool is_valid_parentheses(const char* s) {
    Stack* stack = stack_create(strlen(s));

    for (int i = 0; s[i]; i++) {
        char c = s[i];
        if (c == '(' || c == '{' || c == '[') {
            stack_push(stack, c);
        } else {
            if (stack_is_empty(stack)) {
                stack_free(stack);
                return false;
            }
            char top = stack_pop(stack);
            if ((c == ')' && top != '(') ||
                (c == '}' && top != '{') ||
                (c == ']' && top != '[')) {
                stack_free(stack);
                return false;
            }
        }
    }

    bool valid = stack_is_empty(stack);
    stack_free(stack);
    return valid;
}

/* =============================================================================
 * 5. Postfix Expression Evaluation
 * ============================================================================= */

int evaluate_postfix(const char* expr) {
    Stack* stack = stack_create(strlen(expr));

    for (int i = 0; expr[i]; i++) {
        char c = expr[i];

        if (c >= '0' && c <= '9') {
            stack_push(stack, c - '0');
        } else if (c == ' ') {
            continue;
        } else {
            int b = stack_pop(stack);
            int a = stack_pop(stack);
            int result;

            switch (c) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': result = a / b; break;
                default: result = 0;
            }
            stack_push(stack, result);
        }
    }

    int result = stack_pop(stack);
    stack_free(stack);
    return result;
}

/* =============================================================================
 * 6. Monotonic Stack - Next Greater Element
 * ============================================================================= */

int* next_greater_element(int arr[], int n) {
    int* result = malloc(n * sizeof(int));
    Stack* stack = stack_create(n);

    for (int i = n - 1; i >= 0; i--) {
        while (!stack_is_empty(stack) && stack_peek(stack) <= arr[i]) {
            stack_pop(stack);
        }
        result[i] = stack_is_empty(stack) ? -1 : stack_peek(stack);
        stack_push(stack, arr[i]);
    }

    stack_free(stack);
    return result;
}

/* =============================================================================
 * 7. Sliding Window Maximum (Using Deque)
 * ============================================================================= */

int* sliding_window_max(int arr[], int n, int k, int* result_size) {
    *result_size = n - k + 1;
    int* result = malloc((*result_size) * sizeof(int));
    Deque* dq = deque_create(n);

    for (int i = 0; i < n; i++) {
        /* Remove elements outside the window */
        while (!deque_is_empty(dq) && deque_front(dq) <= i - k) {
            deque_pop_front(dq);
        }

        /* Remove elements smaller than the current value */
        while (!deque_is_empty(dq) && arr[deque_back(dq)] < arr[i]) {
            deque_pop_back(dq);
        }

        deque_push_back(dq, i);

        if (i >= k - 1) {
            result[i - k + 1] = arr[deque_front(dq)];
        }
    }

    deque_free(dq);
    return result;
}

/* =============================================================================
 * 8. Largest Rectangle in Histogram
 * ============================================================================= */

int largest_rectangle_histogram(int heights[], int n) {
    Stack* stack = stack_create(n + 1);
    int max_area = 0;

    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];

        while (!stack_is_empty(stack) && heights[stack_peek(stack)] > h) {
            int height = heights[stack_pop(stack)];
            int width = stack_is_empty(stack) ? i : i - stack_peek(stack) - 1;
            int area = height * width;
            if (area > max_area) max_area = area;
        }

        stack_push(stack, i);
    }

    stack_free(stack);
    return max_area;
}

/* =============================================================================
 * Test
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]");
}

int main(void) {
    printf("============================================================\n");
    printf("Stack and Queue Examples\n");
    printf("============================================================\n");

    /* 1. Stack */
    printf("\n[1] Stack Basic Operations\n");
    Stack* s = stack_create(10);
    stack_push(s, 1);
    stack_push(s, 2);
    stack_push(s, 3);
    printf("    Push: 1, 2, 3\n");
    printf("    Top: %d\n", stack_peek(s));
    printf("    Pop: %d\n", stack_pop(s));
    printf("    Pop: %d\n", stack_pop(s));
    stack_free(s);

    /* 2. Queue */
    printf("\n[2] Queue Basic Operations\n");
    Queue* q = queue_create(10);
    queue_enqueue(q, 1);
    queue_enqueue(q, 2);
    queue_enqueue(q, 3);
    printf("    Enqueue: 1, 2, 3\n");
    printf("    Front: %d\n", queue_front(q));
    printf("    Dequeue: %d\n", queue_dequeue(q));
    printf("    Dequeue: %d\n", queue_dequeue(q));
    queue_free(q);

    /* 3. Parentheses Validation */
    printf("\n[3] Parentheses Validation\n");
    printf("    '()[]{}': %s\n", is_valid_parentheses("()[]{}") ? "valid" : "invalid");
    printf("    '([)]': %s\n", is_valid_parentheses("([)]") ? "valid" : "invalid");
    printf("    '{[()]}': %s\n", is_valid_parentheses("{[()]}") ? "valid" : "invalid");

    /* 4. Postfix Expression */
    printf("\n[4] Postfix Expression Evaluation\n");
    printf("    '2 3 + 4 *' = %d\n", evaluate_postfix("2 3 + 4 *"));
    printf("    '5 1 2 + 4 * + 3 -' = %d\n", evaluate_postfix("5 1 2 + 4 * + 3 -"));

    /* 5. Next Greater Element */
    printf("\n[5] Next Greater Element (Monotonic Stack)\n");
    int arr5[] = {4, 5, 2, 25};
    int* nge = next_greater_element(arr5, 4);
    printf("    Array: [4, 5, 2, 25]\n");
    printf("    NGE:  ");
    print_array(nge, 4);
    printf("\n");
    free(nge);

    /* 6. Sliding Window Maximum */
    printf("\n[6] Sliding Window Maximum\n");
    int arr6[] = {1, 3, -1, -3, 5, 3, 6, 7};
    int result_size;
    int* max_vals = sliding_window_max(arr6, 8, 3, &result_size);
    printf("    Array: [1,3,-1,-3,5,3,6,7], k=3\n");
    printf("    Maximums: ");
    print_array(max_vals, result_size);
    printf("\n");
    free(max_vals);

    /* 7. Histogram */
    printf("\n[7] Largest Rectangle in Histogram\n");
    int heights[] = {2, 1, 5, 6, 2, 3};
    printf("    Heights: [2,1,5,6,2,3]\n");
    printf("    Maximum area: %d\n", largest_rectangle_histogram(heights, 6));

    /* 8. Data Structure Comparison */
    printf("\n[8] Data Structure Comparison\n");
    printf("    | Structure | Insert  | Delete  | Feature           |\n");
    printf("    |-----------|---------|---------|-------------------|\n");
    printf("    | Stack     | O(1)    | O(1)    | LIFO              |\n");
    printf("    | Queue     | O(1)    | O(1)    | FIFO              |\n");
    printf("    | Deque     | O(1)    | O(1)    | Both-end ins/del  |\n");

    printf("\n============================================================\n");

    return 0;
}
