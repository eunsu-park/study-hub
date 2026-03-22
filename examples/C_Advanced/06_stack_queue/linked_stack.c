// linked_stack.c
// Linked list-based stack implementation
// No size limit thanks to dynamic memory allocation

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Node struct
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// Stack struct
typedef struct {
    Node *top;
    int size;
} LinkedStack;

// Create stack
LinkedStack* lstack_create(void) {
    LinkedStack *s = malloc(sizeof(LinkedStack));
    if (s) {
        s->top = NULL;
        s->size = 0;
    }
    return s;
}

// Destroy stack (free all memory)
// Why: every malloc must have a matching free — without this traversal-and-free,
// every remaining node would be a memory leak (N nodes = N leaks)
void lstack_destroy(LinkedStack *s) {
    Node *current = s->top;
    while (current) {
        Node *next = current->next;
        free(current);
        current = next;
    }
    free(s);
}

// Check if empty
bool lstack_isEmpty(LinkedStack *s) {
    return s->top == NULL;
}

// Push - add element to the top (O(1))
// Why: linked-list stack has no capacity limit unlike array-based stacks — each
// push is a single malloc, so the only limit is available heap memory
bool lstack_push(LinkedStack *s, int value) {
    Node *node = malloc(sizeof(Node));
    if (!node) {
        printf("Memory allocation failed!\n");
        return false;
    }

    node->data = value;
    node->next = s->top;  // New node points to current top
    s->top = node;        // Update top to new node
    s->size++;
    return true;
}

// Pop - remove and return top element (O(1))
bool lstack_pop(LinkedStack *s, int *value) {
    if (lstack_isEmpty(s)) {
        printf("Stack Underflow!\n");
        return false;
    }

    // Why: saving data and advancing top BEFORE freeing prevents use-after-free —
    // accessing node->data or node->next after free() is undefined behavior
    Node *node = s->top;
    *value = node->data;
    s->top = node->next;  // Move top to the next node
    free(node);           // Free the removed node's memory
    s->size--;
    return true;
}

// Peek - view top element without removing (O(1))
bool lstack_peek(LinkedStack *s, int *value) {
    if (lstack_isEmpty(s)) {
        return false;
    }
    *value = s->top->data;
    return true;
}

// Print stack (from top to bottom)
void lstack_print(LinkedStack *s) {
    printf("Stack (size=%d): ", s->size);
    Node *current = s->top;
    while (current) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("(top -> bottom)\n");
}

// Return stack size
int lstack_size(LinkedStack *s) {
    return s->size;
}

// Test code
int main(void) {
    LinkedStack *s = lstack_create();
    if (!s) {
        printf("Failed to create stack!\n");
        return 1;
    }

    printf("=== Linked List-Based Stack Test ===\n\n");

    // Push test
    printf("[ Push Operations ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Push %d -> ", i * 10);
        lstack_push(s, i * 10);
        lstack_print(s);
    }

    // Peek test
    printf("\n[ Peek Operation ]\n");
    int top;
    if (lstack_peek(s, &top)) {
        printf("Top value: %d (stack unchanged)\n", top);
        lstack_print(s);
    }

    // Pop test
    printf("\n[ Pop Operations ]\n");
    int value;
    while (lstack_pop(s, &value)) {
        printf("Popped: %d, ", value);
        lstack_print(s);
    }

    // Underflow test
    printf("\n[ Underflow Test ]\n");
    printf("Attempting Pop on empty stack: ");
    lstack_pop(s, &value);

    // Bulk insertion test (advantage of dynamic allocation)
    printf("\n[ Bulk Insertion Test ]\n");
    printf("Inserting 10000 elements...\n");
    for (int i = 0; i < 10000; i++) {
        lstack_push(s, i);
    }
    printf("Insertion complete! Current size: %d\n", lstack_size(s));

    // Memory cleanup
    printf("\nFreeing memory...\n");
    lstack_destroy(s);
    printf("Stack destroyed!\n");

    return 0;
}
