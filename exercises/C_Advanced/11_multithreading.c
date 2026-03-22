/*
 * Exercises for Lesson 13: Project Multithreading
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -pthread -o ex13 13_project_multithreading.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

/* === Exercise 1: Basic Thread Creation === */
/* Problem: Create threads, pass arguments, and collect results. */

typedef struct {
    int thread_id;
    int start;
    int end;
    long result;  /* Output: sum of [start, end) */
} SumArgs;

void *sum_range(void *arg) {
    /*
     * Thread function signature: void *func(void *arg)
     * - Takes a single void* argument (cast to actual type inside)
     * - Returns void* (result or NULL)
     * - Must be careful with shared data (more in Exercise 2)
     */
    SumArgs *args = (SumArgs *)arg;
    args->result = 0;
    for (int i = args->start; i < args->end; i++) {
        args->result += i;
    }
    return NULL;
}

void exercise_1(void) {
    printf("=== Exercise 1: Basic Thread Creation ===\n");

    int num_threads = 4;
    int total = 1000;
    int chunk = total / num_threads;

    pthread_t threads[4];
    SumArgs args[4];

    printf("Computing sum of 0..%d using %d threads:\n", total - 1, num_threads);

    /* Create threads */
    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].start = i * chunk;
        args[i].end = (i == num_threads - 1) ? total : (i + 1) * chunk;
        args[i].result = 0;

        int ret = pthread_create(&threads[i], NULL, sum_range, &args[i]);
        if (ret != 0) {
            fprintf(stderr, "pthread_create failed: %d\n", ret);
            return;
        }
        printf("  Thread %d: computing sum[%d, %d)\n",
               i, args[i].start, args[i].end);
    }

    /* Join all threads and collect results */
    long total_sum = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        printf("  Thread %d: partial sum = %ld\n", i, args[i].result);
        total_sum += args[i].result;
    }

    /* Verify: sum of 0..n-1 = n*(n-1)/2 */
    long expected = (long)total * (total - 1) / 2;
    printf("\nTotal sum: %ld (expected: %ld) -> %s\n",
           total_sum, expected, total_sum == expected ? "CORRECT" : "WRONG");

    /*
     * Key points:
     * - pthread_create starts the thread immediately
     * - pthread_join blocks until the thread completes
     * - Arguments must remain valid until the thread completes
     *   (don't pass local variables that go out of scope!)
     */
}

/* === Exercise 2: Mutex for Shared Counter === */
/* Problem: Demonstrate race condition and fix with mutex. */

#define COUNT_ITERS 100000

/* Shared state for race condition demo */
typedef struct {
    int counter;
    pthread_mutex_t lock;
    int use_mutex;
} SharedCounter;

void *increment_counter(void *arg) {
    SharedCounter *sc = (SharedCounter *)arg;

    for (int i = 0; i < COUNT_ITERS; i++) {
        if (sc->use_mutex) {
            pthread_mutex_lock(&sc->lock);
            sc->counter++;
            pthread_mutex_unlock(&sc->lock);
        } else {
            /*
             * Race condition: counter++ is NOT atomic!
             * It's actually: temp = counter; temp = temp + 1; counter = temp;
             * Two threads can read the same value, both increment to the
             * same result, losing one increment.
             */
            sc->counter++;
        }
    }
    return NULL;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Mutex for Shared Counter ===\n");

    int num_threads = 4;
    int expected = num_threads * COUNT_ITERS;

    /* Test WITHOUT mutex (race condition) */
    SharedCounter no_mutex = { .counter = 0, .use_mutex = 0 };
    pthread_mutex_init(&no_mutex.lock, NULL);

    pthread_t threads[4];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, increment_counter, &no_mutex);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Without mutex: counter = %d (expected %d, lost %d)\n",
           no_mutex.counter, expected, expected - no_mutex.counter);
    pthread_mutex_destroy(&no_mutex.lock);

    /* Test WITH mutex */
    SharedCounter with_mutex = { .counter = 0, .use_mutex = 1 };
    pthread_mutex_init(&with_mutex.lock, NULL);

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, increment_counter, &with_mutex);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("With mutex:    counter = %d (expected %d) -> %s\n",
           with_mutex.counter, expected,
           with_mutex.counter == expected ? "CORRECT" : "WRONG");
    pthread_mutex_destroy(&with_mutex.lock);

    printf("\nMutex overhead: The mutex version is slower because of\n");
    printf("lock/unlock syscalls, but it guarantees correctness.\n");
    printf("For high-contention cases, consider atomic operations\n");
    printf("(__atomic_fetch_add) or lock-free data structures.\n");
}

/* === Exercise 3: Producer-Consumer with Condition Variable === */
/* Problem: Implement a bounded buffer with producer and consumer threads. */

#define BUF_SIZE 5

typedef struct {
    int buffer[BUF_SIZE];
    int in;        /* Write index */
    int out;       /* Read index */
    int count;     /* Current number of items */
    int done;      /* Producer finished flag */
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} BoundedBuffer;

void bb_init(BoundedBuffer *bb) {
    memset(bb->buffer, 0, sizeof(bb->buffer));
    bb->in = bb->out = bb->count = bb->done = 0;
    pthread_mutex_init(&bb->mutex, NULL);
    pthread_cond_init(&bb->not_full, NULL);
    pthread_cond_init(&bb->not_empty, NULL);
}

void bb_destroy(BoundedBuffer *bb) {
    pthread_mutex_destroy(&bb->mutex);
    pthread_cond_destroy(&bb->not_full);
    pthread_cond_destroy(&bb->not_empty);
}

void *producer(void *arg) {
    BoundedBuffer *bb = (BoundedBuffer *)arg;

    for (int i = 1; i <= 10; i++) {
        pthread_mutex_lock(&bb->mutex);

        /* Wait while buffer is full */
        while (bb->count == BUF_SIZE) {
            /*
             * pthread_cond_wait atomically:
             * 1. Releases the mutex
             * 2. Puts the thread to sleep
             * 3. Re-acquires the mutex when woken
             *
             * MUST use while loop, not if, because of spurious wakeups.
             */
            pthread_cond_wait(&bb->not_full, &bb->mutex);
        }

        bb->buffer[bb->in] = i;
        bb->in = (bb->in + 1) % BUF_SIZE;
        bb->count++;
        printf("  Produced: %2d  [buf: %d/%d]\n", i, bb->count, BUF_SIZE);

        pthread_cond_signal(&bb->not_empty);
        pthread_mutex_unlock(&bb->mutex);

        usleep(10000); /* Simulate work */
    }

    pthread_mutex_lock(&bb->mutex);
    bb->done = 1;
    pthread_cond_signal(&bb->not_empty);
    pthread_mutex_unlock(&bb->mutex);

    return NULL;
}

void *consumer(void *arg) {
    BoundedBuffer *bb = (BoundedBuffer *)arg;

    while (1) {
        pthread_mutex_lock(&bb->mutex);

        while (bb->count == 0 && !bb->done) {
            pthread_cond_wait(&bb->not_empty, &bb->mutex);
        }

        if (bb->count == 0 && bb->done) {
            pthread_mutex_unlock(&bb->mutex);
            break;
        }

        int item = bb->buffer[bb->out];
        bb->out = (bb->out + 1) % BUF_SIZE;
        bb->count--;
        printf("  Consumed: %2d  [buf: %d/%d]\n", item, bb->count, BUF_SIZE);

        pthread_cond_signal(&bb->not_full);
        pthread_mutex_unlock(&bb->mutex);

        usleep(20000); /* Consumer is slower -> buffer will fill up */
    }

    return NULL;
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Producer-Consumer ===\n");
    printf("Buffer size: %d, Producing 10 items:\n\n", BUF_SIZE);

    BoundedBuffer bb;
    bb_init(&bb);

    pthread_t prod, cons;
    pthread_create(&prod, NULL, producer, &bb);
    pthread_create(&cons, NULL, consumer, &bb);

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    bb_destroy(&bb);
    printf("\nProducer-consumer completed successfully.\n");
}

/* === Exercise 4: Thread Pool Basics === */
/* Problem: Implement a simple thread pool pattern. */

#define POOL_SIZE 3
#define TASK_COUNT 8

typedef struct {
    int task_id;
    int data;
    int result;
} Task;

typedef struct {
    Task tasks[TASK_COUNT];
    int next_task;  /* Next task to be picked up */
    int completed;
    pthread_mutex_t mutex;
} TaskQueue;

void *worker(void *arg) {
    TaskQueue *tq = (TaskQueue *)arg;

    while (1) {
        pthread_mutex_lock(&tq->mutex);

        if (tq->next_task >= TASK_COUNT) {
            pthread_mutex_unlock(&tq->mutex);
            break;
        }

        /* Grab a task */
        int idx = tq->next_task++;
        pthread_mutex_unlock(&tq->mutex);

        /* Execute task (simulate computation) */
        Task *t = &tq->tasks[idx];
        t->result = t->data * t->data; /* Square the input */
        usleep(50000); /* Simulate work */

        pthread_mutex_lock(&tq->mutex);
        tq->completed++;
        printf("  Worker %lu: task %d, compute(%d) = %d  [%d/%d done]\n",
               (unsigned long)pthread_self() % 1000,
               t->task_id, t->data, t->result, tq->completed, TASK_COUNT);
        pthread_mutex_unlock(&tq->mutex);
    }
    return NULL;
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Thread Pool Basics ===\n");
    printf("Pool size: %d, Tasks: %d\n\n", POOL_SIZE, TASK_COUNT);

    /*
     * Thread pool pattern:
     * - Fixed number of worker threads (avoids thread creation overhead)
     * - Shared task queue protected by mutex
     * - Workers grab tasks until queue is empty
     *
     * Benefits:
     * - Bounded resource usage (known number of threads)
     * - Amortized thread creation cost
     * - Natural load balancing (fast workers take more tasks)
     */

    TaskQueue tq = { .next_task = 0, .completed = 0 };
    pthread_mutex_init(&tq.mutex, NULL);

    for (int i = 0; i < TASK_COUNT; i++) {
        tq.tasks[i].task_id = i;
        tq.tasks[i].data = (i + 1) * 5;
        tq.tasks[i].result = 0;
    }

    pthread_t pool[POOL_SIZE];
    for (int i = 0; i < POOL_SIZE; i++) {
        pthread_create(&pool[i], NULL, worker, &tq);
    }
    for (int i = 0; i < POOL_SIZE; i++) {
        pthread_join(pool[i], NULL);
    }

    printf("\nResults:\n");
    for (int i = 0; i < TASK_COUNT; i++) {
        printf("  Task %d: %d^2 = %d\n",
               tq.tasks[i].task_id, tq.tasks[i].data, tq.tasks[i].result);
    }

    pthread_mutex_destroy(&tq.mutex);
}

/* === Exercise 5: Deadlock Detection and Prevention === */
/* Problem: Demonstrate deadlock scenarios and prevention strategies. */

typedef struct {
    pthread_mutex_t lock_a;
    pthread_mutex_t lock_b;
    int resource_a;
    int resource_b;
} DeadlockDemo;

void *safe_thread_1(void *arg) {
    DeadlockDemo *dd = (DeadlockDemo *)arg;

    /* Always acquire locks in consistent order: A then B */
    pthread_mutex_lock(&dd->lock_a);
    usleep(1000); /* Simulate some work */
    pthread_mutex_lock(&dd->lock_b);

    dd->resource_a += 10;
    dd->resource_b += 20;

    pthread_mutex_unlock(&dd->lock_b);
    pthread_mutex_unlock(&dd->lock_a);
    return NULL;
}

void *safe_thread_2(void *arg) {
    DeadlockDemo *dd = (DeadlockDemo *)arg;

    /* Same order as thread 1: A then B -- no deadlock possible */
    pthread_mutex_lock(&dd->lock_a);
    usleep(1000);
    pthread_mutex_lock(&dd->lock_b);

    dd->resource_a += 30;
    dd->resource_b += 40;

    pthread_mutex_unlock(&dd->lock_b);
    pthread_mutex_unlock(&dd->lock_a);
    return NULL;
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Deadlock Detection and Prevention ===\n");

    /*
     * Deadlock conditions (ALL four must hold simultaneously):
     * 1. Mutual exclusion: resources held exclusively
     * 2. Hold and wait: thread holds one resource, waits for another
     * 3. No preemption: resources can't be forcibly taken
     * 4. Circular wait: thread A waits for B, B waits for A
     *
     * Deadlock scenario:
     *   Thread 1: lock(A) -> lock(B)
     *   Thread 2: lock(B) -> lock(A)
     *   If T1 holds A and T2 holds B, both wait forever.
     *
     * Prevention strategies:
     * 1. Lock ordering: always acquire locks in the same order
     * 2. Try-lock: use pthread_mutex_trylock, back off if failed
     * 3. Timeout: use pthread_mutex_timedlock
     * 4. Lock hierarchy: assign levels to locks, only acquire higher levels
     */

    printf("Deadlock scenario (DO NOT RUN):\n");
    printf("  Thread 1: lock(A) -> sleep -> lock(B)  // Holds A, wants B\n");
    printf("  Thread 2: lock(B) -> sleep -> lock(A)  // Holds B, wants A\n");
    printf("  Result: DEADLOCK! Both threads blocked forever.\n");

    printf("\nPrevention: consistent lock ordering (A before B):\n");

    DeadlockDemo dd = { .resource_a = 0, .resource_b = 0 };
    pthread_mutex_init(&dd.lock_a, NULL);
    pthread_mutex_init(&dd.lock_b, NULL);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, safe_thread_1, &dd);
    pthread_create(&t2, NULL, safe_thread_2, &dd);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("  Both threads completed (no deadlock)\n");
    printf("  resource_a = %d (expected 40)\n", dd.resource_a);
    printf("  resource_b = %d (expected 60)\n", dd.resource_b);

    pthread_mutex_destroy(&dd.lock_a);
    pthread_mutex_destroy(&dd.lock_b);

    printf("\nTrylock pattern (deadlock avoidance):\n");
    printf("  if (pthread_mutex_trylock(&b) != 0) {\n");
    printf("    pthread_mutex_unlock(&a);  // Release and retry\n");
    printf("    usleep(rand() %% 1000);    // Random backoff\n");
    printf("    continue;                  // Try again\n");
    printf("  }\n");
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
