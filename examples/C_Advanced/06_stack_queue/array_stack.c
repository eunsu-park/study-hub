// array_stack.c
// Array-based stack implementation
// LIFO (Last In, First Out) data structure

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

// Push - add element to the top (O(1))
bool stack_push(Stack *s, int value) {
    if (stack_isFull(s)) {
        printf("Stack Overflow!\n");
        return false;
    }
    s->data[++s->top] = value;
    return true;
}

// Pop - remove and return top element (O(1))
bool stack_pop(Stack *s, int *value) {
    if (stack_isEmpty(s)) {
        printf("Stack Underflow!\n");
        return false;
    }
    *value = s->data[s->top--];
    return true;
}

// Peek - view top element without removing (O(1))
bool stack_peek(Stack *s, int *value) {
    if (stack_isEmpty(s)) {
        return false;
    }
    *value = s->data[s->top];
    return true;
}

// Print stack (for debugging)
void stack_print(Stack *s) {
    printf("Stack (top=%d): ", s->top);
    for (int i = 0; i <= s->top; i++) {
        printf("%d ", s->data[i]);
    }
    printf("\n");
}

// Return stack size
int stack_size(Stack *s) {
    return s->top + 1;
}

// Test code
int main(void) {
    Stack s;
    stack_init(&s);

    printf("=== Array-Based Stack Test ===\n\n");

    // Push test
    printf("[ Push Operations ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Push %d -> ", i * 10);
        stack_push(&s, i * 10);
        stack_print(&s);
    }

    // Peek test
    printf("\n[ Peek Operation ]\n");
    int top;
    if (stack_peek(&s, &top)) {
        printf("Top value: %d (stack unchanged)\n", top);
        stack_print(&s);
    }

    // Pop test
    printf("\n[ Pop Operations ]\n");
    int value;
    while (stack_pop(&s, &value)) {
        printf("Popped: %d, ", value);
        stack_print(&s);
    }

    // Underflow test
    printf("\n[ Underflow Test ]\n");
    printf("Attempting Pop on empty stack: ");
    stack_pop(&s, &value);

    // Overflow test
    printf("\n[ Overflow Test ]\n");
    Stack s2;
    stack_init(&s2);
    printf("Attempting Push beyond MAX_SIZE...\n");
    for (int i = 0; i <= MAX_SIZE; i++) {
        if (!stack_push(&s2, i)) {
            printf("Overflow occurred after inserting %d elements\n", i);
            break;
        }
    }

    return 0;
}
