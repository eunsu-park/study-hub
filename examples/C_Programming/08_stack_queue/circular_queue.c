// circular_queue.c
// Circular Queue implementation
// Improved space efficiency by reusing the front portion of the array

#include <stdio.h>
#include <stdbool.h>

#define MAX_SIZE 5

// Circular queue struct
// Why: separate count field avoids the classic circular queue dilemma — without it,
// front==rear is ambiguous (empty or full), and one slot must be wasted to distinguish
typedef struct {
    int data[MAX_SIZE];
    int front;  // Position of first element
    int rear;   // Position of last element
    int count;  // Current number of elements
} CircularQueue;

// Initialize queue
void queue_init(CircularQueue *q) {
    q->front = 0;
    q->rear = -1;
    q->count = 0;
}

// Check if empty
bool queue_isEmpty(CircularQueue *q) {
    return q->count == 0;
}

// Check if full
bool queue_isFull(CircularQueue *q) {
    return q->count == MAX_SIZE;
}

// Enqueue - add element to the rear (O(1))
bool queue_enqueue(CircularQueue *q, int value) {
    if (queue_isFull(q)) {
        printf("Queue is full!\n");
        return false;
    }

    // Why: modulo arithmetic wraps the index back to 0 after reaching MAX_SIZE-1,
    // reusing slots freed by dequeue — a linear queue would waste all dequeued space
    q->rear = (q->rear + 1) % MAX_SIZE;  // Circular wrap-around
    q->data[q->rear] = value;
    q->count++;
    return true;
}

// Dequeue - remove element from the front (O(1))
bool queue_dequeue(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        printf("Queue is empty!\n");
        return false;
    }

    // Why: copying value BEFORE advancing front ensures we read valid data —
    // advancing first would lose track of where the element was
    *value = q->data[q->front];
    q->front = (q->front + 1) % MAX_SIZE;  // Circular wrap-around
    q->count--;
    return true;
}

// Front - view front value without removing
bool queue_front(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        return false;
    }
    *value = q->data[q->front];
    return true;
}

// Rear - view rear value
bool queue_rear(CircularQueue *q, int *value) {
    if (queue_isEmpty(q)) {
        return false;
    }
    *value = q->data[q->rear];
    return true;
}

// Print queue (from front to rear)
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

// Visualize array state (for debugging)
void queue_visualize(CircularQueue *q) {
    printf("Array state: [");
    for (int i = 0; i < MAX_SIZE; i++) {
        if (q->count > 0) {
            int start = q->front;
            int end = q->rear;
            bool inRange = false;

            // Check if index i is within the valid range in the circular queue
            if (start <= end) {
                inRange = (i >= start && i <= end);
            } else {
                inRange = (i >= start || i <= end);
            }

            if (inRange) {
                printf("%d", q->data[i]);
            } else {
                printf("-");
            }
        } else {
            printf("-");
        }
        if (i < MAX_SIZE - 1) printf(" ");
    }
    printf("]\n");

    // Display front and rear positions
    printf("           ");
    for (int i = 0; i < MAX_SIZE; i++) {
        if (i == q->front && i == q->rear && q->count > 0) {
            printf("FR");
        } else if (i == q->front && q->count > 0) {
            printf("F ");
        } else if (i == q->rear && q->count > 0) {
            printf("R ");
        } else {
            printf("  ");
        }
    }
    printf("\n");
}

// Return queue size
int queue_size(CircularQueue *q) {
    return q->count;
}

// Test code
int main(void) {
    CircularQueue q;
    queue_init(&q);

    printf("=== Circular Queue Test ===\n\n");

    // Enqueue test (fill the queue)
    printf("[ Step 1: Enqueue 5 items (fill the queue) ]\n");
    for (int i = 1; i <= 5; i++) {
        printf("Enqueue %d -> ", i * 10);
        queue_enqueue(&q, i * 10);
        queue_print(&q);
        queue_visualize(&q);
        printf("\n");
    }

    // Attempt enqueue when full
    printf("[ Step 2: Attempt Enqueue when full ]\n");
    printf("Enqueue 60 -> ");
    queue_enqueue(&q, 60);
    printf("\n");

    // Dequeue 2 items
    int value;
    printf("[ Step 3: Dequeue 2 items ]\n");
    for (int i = 0; i < 2; i++) {
        queue_dequeue(&q, &value);
        printf("Dequeued: %d -> ", value);
        queue_print(&q);
        queue_visualize(&q);
        printf("\n");
    }

    // Enqueue again (verify circular behavior)
    printf("[ Step 4: Enqueue 2 more items (verify circular wrap-around) ]\n");
    for (int i = 6; i <= 7; i++) {
        printf("Enqueue %d -> ", i * 10);
        queue_enqueue(&q, i * 10);
        queue_print(&q);
        queue_visualize(&q);
        printf("\n");
    }

    // Check Front and Rear
    printf("[ Step 5: Check Front/Rear ]\n");
    int front_val, rear_val;
    if (queue_front(&q, &front_val) && queue_rear(&q, &rear_val)) {
        printf("Front value: %d, Rear value: %d\n", front_val, rear_val);
    }
    printf("\n");

    // Dequeue all elements
    printf("[ Step 6: Dequeue all elements ]\n");
    while (queue_dequeue(&q, &value)) {
        printf("Dequeued: %d -> ", value);
        queue_print(&q);
    }

    // Attempt dequeue when empty
    printf("\n[ Step 7: Attempt Dequeue when empty ]\n");
    printf("Dequeue -> ");
    queue_dequeue(&q, &value);

    return 0;
}
