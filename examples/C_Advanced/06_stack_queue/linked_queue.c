// linked_queue.c
// Linked list-based queue implementation
// No size limit thanks to dynamic memory allocation

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Node struct
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// Queue struct
typedef struct {
    Node *front;  // First node (removal position)
    Node *rear;   // Last node (insertion position)
    int size;
} LinkedQueue;

// Create queue
LinkedQueue* lqueue_create(void) {
    LinkedQueue *q = malloc(sizeof(LinkedQueue));
    if (q) {
        q->front = NULL;
        q->rear = NULL;
        q->size = 0;
    }
    return q;
}

// Destroy queue (free all memory)
void lqueue_destroy(LinkedQueue *q) {
    Node *current = q->front;
    while (current) {
        Node *next = current->next;
        free(current);
        current = next;
    }
    free(q);
}

// Check if empty
bool lqueue_isEmpty(LinkedQueue *q) {
    return q->front == NULL;
}

// Enqueue - add element to the rear (O(1))
bool lqueue_enqueue(LinkedQueue *q, int value) {
    Node *node = malloc(sizeof(Node));
    if (!node) {
        printf("Memory allocation failed!\n");
        return false;
    }

    node->data = value;
    node->next = NULL;

    // If queue is empty, both front and rear point to the new node
    if (q->rear == NULL) {
        q->front = q->rear = node;
    } else {
        // Append new node after the existing rear
        q->rear->next = node;
        q->rear = node;
    }
    q->size++;
    return true;
}

// Dequeue - remove element from the front (O(1))
bool lqueue_dequeue(LinkedQueue *q, int *value) {
    if (lqueue_isEmpty(q)) {
        printf("Queue is empty!\n");
        return false;
    }

    Node *node = q->front;
    *value = node->data;
    q->front = node->next;

    // If the last node was removed, set rear to NULL as well
    if (q->front == NULL) {
        q->rear = NULL;
    }

    free(node);
    q->size--;
    return true;
}

// Front - view front value without removing
bool lqueue_front(LinkedQueue *q, int *value) {
    if (lqueue_isEmpty(q)) {
        return false;
    }
    *value = q->front->data;
    return true;
}

// Rear - view rear value
bool lqueue_rear(LinkedQueue *q, int *value) {
    if (lqueue_isEmpty(q)) {
        return false;
    }
    *value = q->rear->data;
    return true;
}

// Print queue (from front to rear)
void lqueue_print(LinkedQueue *q) {
    printf("Queue (size=%d): front -> ", q->size);
    Node *current = q->front;
    while (current) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("<- rear\n");
}

// Return queue size
int lqueue_size(LinkedQueue *q) {
    return q->size;
}

// Print all elements and clear the queue
void lqueue_clear(LinkedQueue *q) {
    int value;
    while (lqueue_dequeue(q, &value)) {
        // Simply remove all elements
    }
}

// Test code
int main(void) {
    LinkedQueue *q = lqueue_create();
    if (!q) {
        printf("Failed to create queue!\n");
        return 1;
    }

    printf("=== Linked List-Based Queue Test ===\n\n");

    // Enqueue test
    printf("[ Step 1: Enqueue 5 items ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Enqueue %d -> ", i * 10);
        lqueue_enqueue(q, i * 10);
        lqueue_print(q);
    }

    // Check Front/Rear
    printf("\n[ Step 2: Check Front/Rear ]\n");
    int front_val, rear_val;
    if (lqueue_front(q, &front_val) && lqueue_rear(q, &rear_val)) {
        printf("Front value: %d (first in)\n", front_val);
        printf("Rear value: %d (last in)\n", rear_val);
        lqueue_print(q);
    }

    // Dequeue test
    printf("\n[ Step 3: Dequeue 2 items ]\n");
    int value;
    for (int i = 0; i < 2; i++) {
        if (lqueue_dequeue(q, &value)) {
            printf("Dequeued: %d -> ", value);
            lqueue_print(q);
        }
    }

    // Additional Enqueue
    printf("\n[ Step 4: Enqueue 2 more ]\n");
    for (int i = 6; i <= 7; i++) {
        printf("Enqueue %d -> ", i * 10);
        lqueue_enqueue(q, i * 10);
        lqueue_print(q);
    }

    // Dequeue all elements
    printf("\n[ Step 5: Dequeue all elements ]\n");
    while (lqueue_dequeue(q, &value)) {
        printf("Dequeued: %d -> ", value);
        lqueue_print(q);
    }

    // Attempt dequeue when empty
    printf("\n[ Step 6: Attempt Dequeue when empty ]\n");
    printf("Dequeue -> ");
    lqueue_dequeue(q, &value);

    // Bulk insertion test
    printf("\n[ Step 7: Bulk Insertion Test ]\n");
    printf("Inserting 10000 elements...\n");
    for (int i = 0; i < 10000; i++) {
        lqueue_enqueue(q, i);
    }
    printf("Insertion complete! Current size: %d\n", lqueue_size(q));

    // Bulk removal test
    printf("\nChecking first 5 elements:\n");
    for (int i = 0; i < 5; i++) {
        lqueue_dequeue(q, &value);
        printf("  Dequeued: %d\n", value);
    }

    // Memory cleanup
    printf("\nFreeing memory...\n");
    lqueue_destroy(q);
    printf("Queue destroyed!\n");

    return 0;
}
