// race_condition.c
// Race Condition demonstration
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define ITERATIONS 100000

// Shared variable
// Why: global (shared) mutable state without synchronization is the root cause
// of race conditions — each thread sees and modifies the same memory location
int counter = 0;

void* increment(void* arg) {
    (void)arg;

    for (int i = 0; i < ITERATIONS; i++) {
        // Why: counter++ compiles to load-increment-store (3 CPU instructions) —
        // two threads can load the same value, both increment to N+1, and store
        // N+1 twice, losing one increment entirely
        counter++;  // Not atomic!
        // Actually: temp = counter; temp = temp + 1; counter = temp;
    }

    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }

    // Why: pthread_join blocks until the thread finishes — without it, main could
    // exit and print counter before all threads have completed their increments
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Expected: NUM_THREADS * ITERATIONS = 1,000,000
    // Actual: a value less than that (lost due to race condition)
    printf("Expected: %d\n", NUM_THREADS * ITERATIONS);
    printf("Actual:   %d\n", counter);
    printf("Lost:     %d\n", NUM_THREADS * ITERATIONS - counter);

    return 0;
}
