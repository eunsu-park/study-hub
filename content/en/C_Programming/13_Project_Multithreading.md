# Project 12: Multithreaded Programming

**Previous**: [Project 11: Mini Shell](./12_Project_Mini_Shell.md) | **Next**: [Embedded Programming Basics](./14_Embedded_Basics.md)

Learn multithreaded programming using the pthread library.

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and manage POSIX threads using `pthread_create` and `pthread_join`
2. Pass data to threads through structures and retrieve return values
3. Identify race conditions caused by unsynchronized access to shared variables
4. Synchronize shared data access using mutex locks to prevent data corruption
5. Coordinate thread execution using condition variables with proper wait loops
6. Implement the producer-consumer pattern with a bounded buffer
7. Build a thread pool that dispatches tasks from a concurrent queue
8. Use read-write locks to allow concurrent readers with exclusive writers

---

Concurrency is one of the most challenging aspects of systems programming, but it is also one of the most rewarding. Whether you are building a web server that handles thousands of connections or a data pipeline that crunches numbers in parallel, understanding threads, mutexes, and condition variables gives you the tools to harness modern multi-core hardware. This project takes you from your first thread all the way to a reusable thread pool.

## Prerequisites
- Pointers
- Structures
- Function pointers

---

## Stage 1: Thread Basics

### First Thread Program

```c
// thread_basic.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// Thread function: void* return, void* argument
void* print_message(void* arg) {
    char* message = (char*)arg;

    for (int i = 0; i < 5; i++) {
        printf("[Thread] %s - %d\n", message, i);
        sleep(1);
    }

    return NULL;
}

int main(void) {
    pthread_t thread;
    const char* msg = "Hello from thread";

    // Create thread
    int result = pthread_create(&thread, NULL, print_message, (void*)msg);
    if (result != 0) {
        fprintf(stderr, "Thread creation failed: %d\n", result);
        return 1;
    }

    // Main thread also does work
    for (int i = 0; i < 5; i++) {
        printf("[Main] Main thread - %d\n", i);
        sleep(1);
    }

    // Wait for thread to finish
    pthread_join(thread, NULL);

    printf("All tasks completed\n");
    return 0;
}
```

### Compilation

```bash
# Linux
gcc -o thread_basic thread_basic.c -pthread

# macOS
gcc -o thread_basic thread_basic.c -lpthread
```

### Creating Multiple Threads

```c
// multi_threads.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

// Data to pass to thread
typedef struct {
    int id;
    char name[32];
} ThreadData;

void* thread_func(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    printf("Thread %d (%s) started\n", data->id, data->name);

    // Simulate work
    int sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }

    printf("Thread %d completed: sum = %d\n", data->id, sum);

    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    ThreadData data[NUM_THREADS];

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].id = i;
        snprintf(data[i].name, sizeof(data[i].name), "Worker-%d", i);

        int result = pthread_create(&threads[i], NULL, thread_func, &data[i]);
        if (result != 0) {
            fprintf(stderr, "Thread %d creation failed\n", i);
            exit(1);
        }
    }

    printf("All threads created. Waiting...\n");

    // Wait for all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Program finished\n");
    return 0;
}
```

### Receiving Thread Return Values

```c
// thread_return.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* calculate_sum(void* arg) {
    int n = *(int*)arg;

    // Dynamically allocate result
    long* result = malloc(sizeof(long));
    *result = 0;

    for (int i = 1; i <= n; i++) {
        *result += i;
    }

    printf("Thread: Sum from 1 to %d calculated\n", n);
    return result;
}

int main(void) {
    pthread_t thread;
    int n = 100;

    pthread_create(&thread, NULL, calculate_sum, &n);

    // Receive return value
    void* ret_val;
    pthread_join(thread, &ret_val);

    long* result = (long*)ret_val;
    printf("Result: %ld\n", *result);

    free(result);  // Free dynamically allocated memory
    return 0;
}
```

---

## Stage 2: Race Condition

Problems occur when multiple threads access shared data simultaneously.

### Race Condition Example

```c
// race_condition.c
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define ITERATIONS 100000

// Shared variable
int counter = 0;

void* increment(void* arg) {
    (void)arg;

    for (int i = 0; i < ITERATIONS; i++) {
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

    // Wait
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Expected: NUM_THREADS * ITERATIONS = 1,000,000
    // Actual: Less (loss due to race condition)
    printf("Expected: %d\n", NUM_THREADS * ITERATIONS);
    printf("Actual: %d\n", counter);
    printf("Lost: %d\n", NUM_THREADS * ITERATIONS - counter);

    return 0;
}
```

Execution result:
```
Expected: 1000000
Actual: 847293
Lost: 152707
```

---

## Stage 3: Mutex

Synchronize access to shared resources with mutexes.

### Using Mutex

```c
// mutex_example.c
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define ITERATIONS 100000

int counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_safe(void* arg) {
    (void)arg;

    for (int i = 0; i < ITERATIONS; i++) {
        pthread_mutex_lock(&mutex);    // Lock
        counter++;                      // Critical section
        pthread_mutex_unlock(&mutex);  // Unlock
    }

    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, increment_safe, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Expected: %d\n", NUM_THREADS * ITERATIONS);
    printf("Actual: %d\n", counter);

    pthread_mutex_destroy(&mutex);
    return 0;
}
```

### Bank Account with Mutex

```c
// bank_account.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

typedef struct {
    int balance;
    pthread_mutex_t lock;
} Account;

Account* account_create(int initial_balance) {
    Account* acc = malloc(sizeof(Account));
    acc->balance = initial_balance;
    pthread_mutex_init(&acc->lock, NULL);
    return acc;
}

void account_destroy(Account* acc) {
    pthread_mutex_destroy(&acc->lock);
    free(acc);
}

int account_deposit(Account* acc, int amount) {
    pthread_mutex_lock(&acc->lock);

    acc->balance += amount;
    int new_balance = acc->balance;

    pthread_mutex_unlock(&acc->lock);
    return new_balance;
}

int account_withdraw(Account* acc, int amount) {
    pthread_mutex_lock(&acc->lock);

    if (acc->balance >= amount) {
        acc->balance -= amount;
        int new_balance = acc->balance;
        pthread_mutex_unlock(&acc->lock);
        return new_balance;
    }

    pthread_mutex_unlock(&acc->lock);
    return -1;  // Insufficient balance
}

int account_get_balance(Account* acc) {
    pthread_mutex_lock(&acc->lock);
    int balance = acc->balance;
    pthread_mutex_unlock(&acc->lock);
    return balance;
}

// Transfer between accounts
int account_transfer(Account* from, Account* to, int amount) {
    // Prevent deadlock: always lock in same order
    // Lock account with smaller address first
    Account* first = (from < to) ? from : to;
    Account* second = (from < to) ? to : from;

    pthread_mutex_lock(&first->lock);
    pthread_mutex_lock(&second->lock);

    int result = -1;
    if (from->balance >= amount) {
        from->balance -= amount;
        to->balance += amount;
        result = from->balance;
    }

    pthread_mutex_unlock(&second->lock);
    pthread_mutex_unlock(&first->lock);

    return result;
}

// Thread data for testing
typedef struct {
    Account* acc;
    int thread_id;
} ThreadArg;

void* depositor(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;

    for (int i = 0; i < 100; i++) {
        int new_balance = account_deposit(ta->acc, 100);
        printf("[Depositor %d] Deposited 100 -> Balance: %d\n", ta->thread_id, new_balance);
        usleep(rand() % 10000);
    }

    return NULL;
}

void* withdrawer(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;

    for (int i = 0; i < 100; i++) {
        int result = account_withdraw(ta->acc, 100);
        if (result >= 0) {
            printf("[Withdrawer %d] Withdrew 100 -> Balance: %d\n", ta->thread_id, result);
        } else {
            printf("[Withdrawer %d] Insufficient balance\n", ta->thread_id);
        }
        usleep(rand() % 10000);
    }

    return NULL;
}

int main(void) {
    srand(time(NULL));

    Account* acc = account_create(10000);
    printf("Initial balance: %d\n\n", account_get_balance(acc));

    pthread_t depositors[3];
    pthread_t withdrawers[3];
    ThreadArg args[6];

    // 3 depositors
    for (int i = 0; i < 3; i++) {
        args[i].acc = acc;
        args[i].thread_id = i;
        pthread_create(&depositors[i], NULL, depositor, &args[i]);
    }

    // 3 withdrawers
    for (int i = 0; i < 3; i++) {
        args[i + 3].acc = acc;
        args[i + 3].thread_id = i;
        pthread_create(&withdrawers[i], NULL, withdrawer, &args[i + 3]);
    }

    // Wait
    for (int i = 0; i < 3; i++) {
        pthread_join(depositors[i], NULL);
        pthread_join(withdrawers[i], NULL);
    }

    printf("\nFinal balance: %d\n", account_get_balance(acc));
    printf("Expected balance: %d (initial 10000 + deposits 30000 - withdrawals max 30000)\n", 10000);

    account_destroy(acc);
    return 0;
}
```

---

## Stage 4: Condition Variable

Wait for threads until a specific condition is met.

### Condition Variable Basics

```c
// condition_basic.c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool ready = false;

void* waiter(void* arg) {
    int id = *(int*)arg;

    pthread_mutex_lock(&mutex);

    while (!ready) {  // Wait while condition is false
        printf("[Waiter %d] Waiting for condition...\n", id);
        pthread_cond_wait(&cond, &mutex);  // Wait (mutex released)
    }
    // When awakened from pthread_cond_wait, mutex is reacquired

    printf("[Waiter %d] Condition satisfied! Starting work\n", id);

    pthread_mutex_unlock(&mutex);
    return NULL;
}

void* signaler(void* arg) {
    (void)arg;

    sleep(2);  // Wait 2 seconds

    pthread_mutex_lock(&mutex);
    ready = true;
    printf("[Signaler] Condition set. Broadcasting signal!\n");
    pthread_cond_broadcast(&cond);  // Signal all waiters
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(void) {
    pthread_t waiters[3];
    pthread_t sig;
    int ids[] = {1, 2, 3};

    // Create waiting threads
    for (int i = 0; i < 3; i++) {
        pthread_create(&waiters[i], NULL, waiter, &ids[i]);
    }

    // Create signaling thread
    pthread_create(&sig, NULL, signaler, NULL);

    // Wait
    for (int i = 0; i < 3; i++) {
        pthread_join(waiters[i], NULL);
    }
    pthread_join(sig, NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}
```

---

## Stage 5: Producer-Consumer Pattern

One of the most important synchronization patterns.

### Bounded Buffer

```c
// producer_consumer.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>

#define BUFFER_SIZE 5
#define NUM_ITEMS 20

// Bounded buffer
typedef struct {
    int buffer[BUFFER_SIZE];
    int count;      // Current item count
    int in;         // Next insertion position
    int out;        // Next removal position

    pthread_mutex_t mutex;
    pthread_cond_t not_full;   // Buffer not full
    pthread_cond_t not_empty;  // Buffer not empty

    bool done;      // Production complete flag
} BoundedBuffer;

BoundedBuffer* buffer_create(void) {
    BoundedBuffer* bb = malloc(sizeof(BoundedBuffer));
    bb->count = 0;
    bb->in = 0;
    bb->out = 0;
    bb->done = false;

    pthread_mutex_init(&bb->mutex, NULL);
    pthread_cond_init(&bb->not_full, NULL);
    pthread_cond_init(&bb->not_empty, NULL);

    return bb;
}

void buffer_destroy(BoundedBuffer* bb) {
    pthread_mutex_destroy(&bb->mutex);
    pthread_cond_destroy(&bb->not_full);
    pthread_cond_destroy(&bb->not_empty);
    free(bb);
}

void buffer_put(BoundedBuffer* bb, int item) {
    pthread_mutex_lock(&bb->mutex);

    // Wait if buffer is full
    while (bb->count == BUFFER_SIZE) {
        printf("[Producer] Buffer full. Waiting...\n");
        pthread_cond_wait(&bb->not_full, &bb->mutex);
    }

    // Insert item
    bb->buffer[bb->in] = item;
    bb->in = (bb->in + 1) % BUFFER_SIZE;
    bb->count++;

    printf("[Producer] Item %d produced (buffer: %d/%d)\n",
           item, bb->count, BUFFER_SIZE);

    // Notify consumer
    pthread_cond_signal(&bb->not_empty);

    pthread_mutex_unlock(&bb->mutex);
}

int buffer_get(BoundedBuffer* bb, int* item) {
    pthread_mutex_lock(&bb->mutex);

    // Wait if buffer is empty and production not done
    while (bb->count == 0 && !bb->done) {
        printf("[Consumer] Buffer empty. Waiting...\n");
        pthread_cond_wait(&bb->not_empty, &bb->mutex);
    }

    // If buffer empty and production done, exit
    if (bb->count == 0 && bb->done) {
        pthread_mutex_unlock(&bb->mutex);
        return 0;  // No more items
    }

    // Remove item
    *item = bb->buffer[bb->out];
    bb->out = (bb->out + 1) % BUFFER_SIZE;
    bb->count--;

    printf("[Consumer] Item %d consumed (buffer: %d/%d)\n",
           *item, bb->count, BUFFER_SIZE);

    // Notify producer
    pthread_cond_signal(&bb->not_full);

    pthread_mutex_unlock(&bb->mutex);
    return 1;  // Success
}

void buffer_set_done(BoundedBuffer* bb) {
    pthread_mutex_lock(&bb->mutex);
    bb->done = true;
    pthread_cond_broadcast(&bb->not_empty);  // Wake all consumers
    pthread_mutex_unlock(&bb->mutex);
}

// Producer thread
void* producer(void* arg) {
    BoundedBuffer* bb = (BoundedBuffer*)arg;

    for (int i = 1; i <= NUM_ITEMS; i++) {
        usleep((rand() % 500) * 1000);  // 0~500ms wait
        buffer_put(bb, i);
    }

    printf("[Producer] Production complete\n");
    buffer_set_done(bb);

    return NULL;
}

// Consumer thread
void* consumer(void* arg) {
    BoundedBuffer* bb = (BoundedBuffer*)arg;
    int item;

    while (buffer_get(bb, &item)) {
        usleep((rand() % 800) * 1000);  // 0~800ms processing time
    }

    printf("[Consumer] Consumption complete\n");
    return NULL;
}

int main(void) {
    srand(time(NULL));

    BoundedBuffer* bb = buffer_create();

    pthread_t prod;
    pthread_t cons[2];

    // 1 producer
    pthread_create(&prod, NULL, producer, bb);

    // 2 consumers
    pthread_create(&cons[0], NULL, consumer, bb);
    pthread_create(&cons[1], NULL, consumer, bb);

    // Wait
    pthread_join(prod, NULL);
    pthread_join(cons[0], NULL);
    pthread_join(cons[1], NULL);

    buffer_destroy(bb);
    printf("\nProgram finished\n");

    return 0;
}
```

---

## Stage 6: Thread Pool

A pattern commonly used in real server programs.

### Thread Pool Implementation

```c
// thread_pool.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>

#define POOL_SIZE 4
#define QUEUE_SIZE 100

// Task definition
typedef struct Task {
    void (*function)(void* arg);
    void* arg;
} Task;

// Task queue
typedef struct {
    Task tasks[QUEUE_SIZE];
    int front;
    int rear;
    int count;

    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;

    bool shutdown;
} TaskQueue;

// Thread pool
typedef struct {
    pthread_t threads[POOL_SIZE];
    TaskQueue queue;
    int thread_count;
} ThreadPool;

// Initialize task queue
void queue_init(TaskQueue* q) {
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->shutdown = false;

    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

// Destroy task queue
void queue_destroy(TaskQueue* q) {
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

// Add task
bool queue_push(TaskQueue* q, Task task) {
    pthread_mutex_lock(&q->mutex);

    while (q->count == QUEUE_SIZE && !q->shutdown) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }

    if (q->shutdown) {
        pthread_mutex_unlock(&q->mutex);
        return false;
    }

    q->tasks[q->rear] = task;
    q->rear = (q->rear + 1) % QUEUE_SIZE;
    q->count++;

    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);

    return true;
}

// Get task
bool queue_pop(TaskQueue* q, Task* task) {
    pthread_mutex_lock(&q->mutex);

    while (q->count == 0 && !q->shutdown) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }

    if (q->count == 0 && q->shutdown) {
        pthread_mutex_unlock(&q->mutex);
        return false;
    }

    *task = q->tasks[q->front];
    q->front = (q->front + 1) % QUEUE_SIZE;
    q->count--;

    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);

    return true;
}

// Worker thread function
void* worker_thread(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    Task task;

    printf("[Worker] Thread started (TID: %lu)\n", pthread_self());

    while (queue_pop(&pool->queue, &task)) {
        printf("[Worker %lu] Executing task\n", pthread_self());
        task.function(task.arg);
    }

    printf("[Worker %lu] Thread exiting\n", pthread_self());
    return NULL;
}

// Create thread pool
ThreadPool* pool_create(int size) {
    ThreadPool* pool = malloc(sizeof(ThreadPool));
    pool->thread_count = size;

    queue_init(&pool->queue);

    for (int i = 0; i < size; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }

    return pool;
}

// Submit task
bool pool_submit(ThreadPool* pool, void (*function)(void*), void* arg) {
    Task task = { .function = function, .arg = arg };
    return queue_push(&pool->queue, task);
}

// Shutdown thread pool
void pool_shutdown(ThreadPool* pool) {
    pthread_mutex_lock(&pool->queue.mutex);
    pool->queue.shutdown = true;
    pthread_cond_broadcast(&pool->queue.not_empty);
    pthread_mutex_unlock(&pool->queue.mutex);

    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    queue_destroy(&pool->queue);
    free(pool);
}

// ============ Test ============

typedef struct {
    int id;
    int value;
} WorkItem;

void process_work(void* arg) {
    WorkItem* item = (WorkItem*)arg;

    printf("Processing task %d (value: %d)...\n", item->id, item->value);
    usleep((rand() % 500 + 100) * 1000);  // 100~600ms processing
    printf("Task %d completed!\n", item->id);

    free(item);
}

int main(void) {
    srand(time(NULL));

    printf("Creating thread pool (size: %d)\n\n", POOL_SIZE);
    ThreadPool* pool = pool_create(POOL_SIZE);

    // Submit tasks
    for (int i = 0; i < 10; i++) {
        WorkItem* item = malloc(sizeof(WorkItem));
        item->id = i;
        item->value = rand() % 100;

        printf("Submitting task %d (value: %d)\n", i, item->value);
        pool_submit(pool, process_work, item);

        usleep(100000);  // 100ms interval
    }

    printf("\nAll tasks submitted. Waiting for pool shutdown...\n\n");
    sleep(2);  // Wait for task processing

    pool_shutdown(pool);
    printf("\nProgram finished\n");

    return 0;
}
```

---

## Stage 7: Read-Write Lock

Allow concurrent reads, exclusive writes.

```c
// rwlock_example.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_READERS 5
#define NUM_WRITERS 2

// Shared data
typedef struct {
    int data;
    pthread_rwlock_t lock;
} SharedData;

SharedData shared = { .data = 0 };

void* reader(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 5; i++) {
        pthread_rwlock_rdlock(&shared.lock);  // Read lock

        printf("[Reader %d] Read data: %d\n", id, shared.data);
        usleep(100000);  // Reading...

        pthread_rwlock_unlock(&shared.lock);

        usleep(rand() % 200000);
    }

    return NULL;
}

void* writer(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 3; i++) {
        pthread_rwlock_wrlock(&shared.lock);  // Write lock (exclusive)

        shared.data = rand() % 1000;
        printf("[Writer %d] Wrote data: %d\n", id, shared.data);
        usleep(200000);  // Writing...

        pthread_rwlock_unlock(&shared.lock);

        usleep(rand() % 500000);
    }

    return NULL;
}

int main(void) {
    srand(time(NULL));

    pthread_rwlock_init(&shared.lock, NULL);

    pthread_t readers[NUM_READERS];
    pthread_t writers[NUM_WRITERS];
    int reader_ids[NUM_READERS];
    int writer_ids[NUM_WRITERS];

    // Create readers
    for (int i = 0; i < NUM_READERS; i++) {
        reader_ids[i] = i;
        pthread_create(&readers[i], NULL, reader, &reader_ids[i]);
    }

    // Create writers
    for (int i = 0; i < NUM_WRITERS; i++) {
        writer_ids[i] = i;
        pthread_create(&writers[i], NULL, writer, &writer_ids[i]);
    }

    // Wait
    for (int i = 0; i < NUM_READERS; i++) {
        pthread_join(readers[i], NULL);
    }
    for (int i = 0; i < NUM_WRITERS; i++) {
        pthread_join(writers[i], NULL);
    }

    pthread_rwlock_destroy(&shared.lock);
    printf("Complete\n");

    return 0;
}
```

---

## Stage 8: Real Example - Parallel Sorting

### Multithreaded Merge Sort

```c
// parallel_sort.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define THRESHOLD 10000  // Use single thread if smaller

typedef struct {
    int* arr;
    int left;
    int right;
} SortTask;

// Merge
void merge(int* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = malloc(n1 * sizeof(int));
    int* R = malloc(n2 * sizeof(int));

    memcpy(L, arr + left, n1 * sizeof(int));
    memcpy(R, arr + mid + 1, n2 * sizeof(int));

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// Single-threaded merge sort
void merge_sort_single(int* arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_single(arr, left, mid);
        merge_sort_single(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Multithreaded merge sort
void* merge_sort_parallel(void* arg) {
    SortTask* task = (SortTask*)arg;
    int* arr = task->arr;
    int left = task->left;
    int right = task->right;

    if (left >= right) return NULL;

    // Use single thread for small arrays
    if (right - left < THRESHOLD) {
        merge_sort_single(arr, left, right);
        return NULL;
    }

    int mid = left + (right - left) / 2;

    // Left half: new thread
    SortTask left_task = { arr, left, mid };
    pthread_t left_thread;
    pthread_create(&left_thread, NULL, merge_sort_parallel, &left_task);

    // Right half: current thread
    SortTask right_task = { arr, mid + 1, right };
    merge_sort_parallel(&right_task);

    // Wait for left thread
    pthread_join(left_thread, NULL);

    // Merge
    merge(arr, left, mid, right);

    return NULL;
}

// Print array
void print_array(int* arr, int n) {
    for (int i = 0; i < n && i < 20; i++) {
        printf("%d ", arr[i]);
    }
    if (n > 20) printf("...");
    printf("\n");
}

// Verify array
int is_sorted(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) return 0;
    }
    return 1;
}

int main(void) {
    srand(time(NULL));

    int n = 1000000;  // One million
    int* arr1 = malloc(n * sizeof(int));
    int* arr2 = malloc(n * sizeof(int));

    // Generate random array
    for (int i = 0; i < n; i++) {
        arr1[i] = rand();
        arr2[i] = arr1[i];  // Copy
    }

    printf("Array size: %d\n\n", n);

    // Single-threaded sort
    clock_t start = clock();
    merge_sort_single(arr1, 0, n - 1);
    clock_t end = clock();
    double single_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Single-threaded: %.3f seconds\n", single_time);
    printf("Sort verification: %s\n\n", is_sorted(arr1, n) ? "OK" : "FAIL");

    // Multithreaded sort
    start = clock();
    SortTask task = { arr2, 0, n - 1 };
    merge_sort_parallel(&task);
    end = clock();
    double parallel_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Multithreaded: %.3f seconds\n", parallel_time);
    printf("Sort verification: %s\n\n", is_sorted(arr2, n) ? "OK" : "FAIL");

    printf("Speedup: %.2fx\n", single_time / parallel_time);

    free(arr1);
    free(arr2);

    return 0;
}
```

---

## Exercises

### Exercise 1: Dining Philosophers
5 philosophers sit at a round table with 5 chopsticks.
- Philosophers think or eat
- Need both chopsticks to eat
- Implement without deadlock

### Exercise 2: Barrier
Implement a barrier that waits until N threads arrive.

```c
typedef struct {
    int count;
    int threshold;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Barrier;

void barrier_wait(Barrier* b);
```

### Exercise 3: Semaphore Implementation
Implement a counting semaphore using mutex and condition variables.

```c
typedef struct {
    int value;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Semaphore;

void sem_wait(Semaphore* sem);
void sem_post(Semaphore* sem);
```

### Exercise 4: Parallel Matrix Multiplication
Compute N×N matrix multiplication using multiple threads.

---

## Key Concepts Summary

| Function | Description |
|----------|-------------|
| `pthread_create()` | Create thread |
| `pthread_join()` | Wait for thread to finish |
| `pthread_mutex_lock()` | Lock mutex |
| `pthread_mutex_unlock()` | Unlock mutex |
| `pthread_cond_wait()` | Wait on condition |
| `pthread_cond_signal()` | Wake one waiter |
| `pthread_cond_broadcast()` | Wake all waiters |

| Concept | Description |
|---------|-------------|
| Race condition | Bugs from concurrent access by multiple threads |
| Mutex | Mutual exclusion (only one access at a time) |
| Condition variable | Wait until condition is satisfied |
| Deadlock | Stuck waiting for each other's resources |
| Producer-consumer | Pattern separating data production/processing |
| Thread pool | Process tasks with pre-created threads |

---

## Debugging Tips

### 1. Using ThreadSanitizer

```bash
gcc -fsanitize=thread -g program.c -o program -lpthread
./program
```

### 2. Helgrind (Valgrind)

```bash
valgrind --tool=helgrind ./program
```

### 3. Common Mistakes

- Forgetting to unlock mutex → Check `unlock` on all paths
- Using `if` instead of `while` with condition variables → Always use `while`
- Inconsistent lock ordering → Always lock in same order

---

## C Programming Projects Complete!

If you've completed these 12 projects, you've experienced all core concepts of C:

1. ✅ Environment setup
2. ✅ C basics quick review
3. ✅ Calculator (functions, conditionals)
4. ✅ Number guessing (loops, random)
5. ✅ Address book (structures, file I/O)
6. ✅ Dynamic arrays (memory management)
7. ✅ Linked lists (advanced pointers)
8. ✅ File encryption (bit operations)
9. ✅ Stack and queue (data structures)
10. ✅ Hash table (hash functions)
11. ✅ Snake game (terminal control)
12. ✅ Mini shell (process management)
13. ✅ Multithreading (concurrency)

Recommended next learning:
- Network programming (sockets)
- Advanced system calls
- Linux kernel modules
- Embedded systems
