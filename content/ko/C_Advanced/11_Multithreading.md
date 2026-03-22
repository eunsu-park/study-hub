# C 멀티스레딩

**이전**: [프로젝트: 미니 셸](./10_Project_Mini_Shell.md) | **다음**: [네트워크 프로그래밍](./12_Network_Programming.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `pthread_create`와 `pthread_join`을 사용하여 POSIX 스레드를 생성하고 관리할 수 있다
2. 구조체를 통해 스레드에 데이터를 전달하고 반환 값을 받을 수 있다
3. 동기화되지 않은 공유 변수 접근으로 인한 경쟁 조건을 식별할 수 있다
4. 뮤텍스 잠금을 사용하여 공유 데이터 접근을 동기화하고 데이터 손상을 방지할 수 있다
5. 적절한 대기 루프와 함께 조건 변수를 사용하여 스레드 실행을 조율할 수 있다
6. 제한된 버퍼를 가진 생산자-소비자 패턴을 구현할 수 있다
7. 동시성 큐에서 작업을 분배하는 스레드 풀을 구축할 수 있다
8. 읽기-쓰기 잠금을 사용하여 동시 읽기와 배타적 쓰기를 허용할 수 있다

---

동시성은 시스템 프로그래밍에서 가장 도전적인 측면 중 하나이지만, 가장 보람 있는 것이기도 합니다. 수천 개의 연결을 처리하는 웹 서버를 구축하든, 데이터를 병렬로 처리하는 데이터 파이프라인을 만들든, 스레드, 뮤텍스, 조건 변수를 이해하면 최신 멀티코어 하드웨어를 활용하는 도구를 갖게 됩니다. 이 레슨에서는 첫 번째 스레드부터 재사용 가능한 스레드 풀까지 안내합니다.

## 사전 지식
- 포인터
- 구조체
- 함수 포인터

---

## 단계 1: 스레드 기초

### 첫 번째 스레드 프로그램

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

### 컴파일

```bash
# Linux
gcc -o thread_basic thread_basic.c -pthread

# macOS
gcc -o thread_basic thread_basic.c -lpthread
```

### 여러 스레드 생성

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

### 스레드 반환 값 받기

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

## 단계 2: 경쟁 조건 (Race Condition)

여러 스레드가 공유 데이터에 동시에 접근하면 문제가 발생합니다.

### 경쟁 조건 예제

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

실행 결과:
```
Expected: 1000000
Actual: 847293
Lost: 152707
```

---

## 단계 3: 뮤텍스 (Mutex)

뮤텍스로 공유 자원에 대한 접근을 동기화합니다.

### 뮤텍스 사용

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

### 뮤텍스를 사용한 은행 계좌

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

    account_destroy(acc);
    return 0;
}
```

---

## 단계 4: 조건 변수 (Condition Variable)

특정 조건이 충족될 때까지 스레드를 대기시킵니다.

### 조건 변수 기초

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

## 단계 5: 생산자-소비자 패턴

가장 중요한 동기화 패턴 중 하나입니다.

### 제한된 버퍼

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

## 단계 6: 스레드 풀

실제 서버 프로그램에서 흔히 사용되는 패턴입니다.

### 스레드 풀 구현

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

## 단계 7: 읽기-쓰기 잠금 (Read-Write Lock)

동시 읽기는 허용하고, 쓰기는 배타적으로 처리합니다.

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
    printf("완료\n");

    return 0;
}
```

---

## 단계 8: C11 원자적 연산(_Atomic)과 메모리 순서(memory ordering)

C11은 `<stdatomic.h>`를 통해 네이티브 원자적 연산(atomic operations)을 도입하여, 플랫폼별 인트린식(intrinsics)이나 뮤텍스로 보호된 단일 변수의 이식성 있는 대안을 제공합니다.

### _Atomic 타입 한정자

```c
#include <stdatomic.h>
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define ITERATIONS  100000

_Atomic int counter = 0;  // C11 원자적 정수

void* increment_atomic(void* arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_fetch_add(&counter, 1);  // Atomic read-modify-write
    }
    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_atomic, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    printf("Counter: %d (expected %d)\n", counter, NUM_THREADS * ITERATIONS);
    return 0;
}
```

### 메모리 순서(Memory Ordering)

각 원자적 연산은 컴파일러와 CPU가 주변 메모리 접근을 재배열하는 방식을 제어하는 선택적 메모리 순서 인수를 받습니다:

| 순서 | 설명 |
|------|------|
| `memory_order_relaxed` | 순서 보장 없음; 원자성만 보장. 가장 빠름. |
| `memory_order_acquire` | 이후의 모든 읽기/쓰기가 이 로드 이후에 발생. |
| `memory_order_release` | 이전의 모든 읽기/쓰기가 이 저장 이전에 발생. |
| `memory_order_seq_cst` | 완전한 순차 일관성 (기본값). 가장 느림. |

```c
// 생산자: 데이터 쓰기 후 플래그 공개
atomic_store_explicit(&flag, 1, memory_order_release);

// 소비자: 플래그 대기 후 데이터 읽기
while (!atomic_load_explicit(&flag, memory_order_acquire))
    ;  // spin
```

`acquire`/`release` 쌍은 대부분의 생산자-소비자 플래그에 충분하며, 약한 순서의 아키텍처(ARM, POWER)에서 `seq_cst`보다 비용이 적습니다.

### 스핀락(Spinlock) vs 뮤텍스(Mutex) 트레이드오프

스핀락(spinlock)은 원자적 연산만으로 구축된 뮤텍스입니다:

```c
typedef _Atomic int Spinlock;

void spin_lock(Spinlock* lock) {
    int expected = 0;
    // 0 -> 1로 성공적으로 변경할 때까지 회전
    while (!atomic_compare_exchange_weak_explicit(
               lock, &expected, 1,
               memory_order_acquire, memory_order_relaxed)) {
        expected = 0;  // Reset after failed CAS
    }
}

void spin_unlock(Spinlock* lock) {
    atomic_store_explicit(lock, 0, memory_order_release);
}
```

**스핀락을 사용하는 경우**: 임계 구역이 매우 짧고 (몇 개의 명령) 경합이 낮을 때. 스핀락은 `pthread_mutex_lock`의 시스템 콜 오버헤드를 피하지만 대기하는 동안 CPU 사이클을 소모합니다. 단일 코어 머신이나 높은 경합 상황에서는 항상 뮤텍스를 사용하는 것이 좋습니다.

---

## 단계 9: 실용 예제 -- 병렬 정렬

### 멀티스레드 병합 정렬

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

// Verify array
int is_sorted(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) return 0;
    }
    return 1;
}

int main(void) {
    srand(time(NULL));

    int n = 1000000;  // 백만 개
    int* arr1 = malloc(n * sizeof(int));
    int* arr2 = malloc(n * sizeof(int));

    // Generate random array
    for (int i = 0; i < n; i++) {
        arr1[i] = rand();
        arr2[i] = arr1[i];  // Copy
    }

    printf("배열 크기: %d\n\n", n);

    // Single-threaded sort
    clock_t start = clock();
    merge_sort_single(arr1, 0, n - 1);
    clock_t end = clock();
    double single_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("단일 스레드: %.3f초\n", single_time);
    printf("정렬 검증: %s\n\n", is_sorted(arr1, n) ? "OK" : "FAIL");

    // Multithreaded sort
    start = clock();
    SortTask task = { arr2, 0, n - 1 };
    merge_sort_parallel(&task);
    end = clock();
    double parallel_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("멀티스레드: %.3f초\n", parallel_time);
    printf("정렬 검증: %s\n\n", is_sorted(arr2, n) ? "OK" : "FAIL");

    printf("속도 향상: %.2f배\n", single_time / parallel_time);

    free(arr1);
    free(arr2);

    return 0;
}
```

---

## 연습 문제

### 연습 문제 1: 식사하는 철학자
5명의 철학자가 5개의 젓가락이 놓인 원탁에 앉아 있습니다.
- 철학자는 생각하거나 먹습니다
- 먹으려면 양쪽 젓가락이 모두 필요합니다
- 교착 상태(deadlock) 없이 구현하세요

### 연습 문제 2: 배리어
N개의 스레드가 도착할 때까지 대기하는 배리어를 구현하세요.

```c
typedef struct {
    int count;
    int threshold;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Barrier;

void barrier_wait(Barrier* b);
```

### 연습 문제 3: 세마포어 구현
뮤텍스와 조건 변수를 사용하여 카운팅 세마포어를 구현하세요.

```c
typedef struct {
    int value;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Semaphore;

void sem_wait(Semaphore* sem);
void sem_post(Semaphore* sem);
```

### 연습 문제 4: 병렬 행렬 곱셈
여러 스레드를 사용하여 N x N 행렬 곱셈을 계산하세요.

### 연습 문제 5: 동시성 해시 맵
여러 스레드에서 동시에 삽입, 조회, 삭제를 지원하는 버킷별 잠금 방식의 해시 맵을 구현하세요.

---

## 핵심 개념 요약

| 함수 | 설명 |
|------|------|
| `pthread_create()` | 스레드 생성 |
| `pthread_join()` | 스레드 종료 대기 |
| `pthread_mutex_lock()` | 뮤텍스 잠금 |
| `pthread_mutex_unlock()` | 뮤텍스 해제 |
| `pthread_cond_wait()` | 조건 대기 |
| `pthread_cond_signal()` | 대기자 하나 깨우기 |
| `pthread_cond_broadcast()` | 모든 대기자 깨우기 |

| 개념 | 설명 |
|------|------|
| 경쟁 조건 (Race condition) | 여러 스레드의 동시 접근으로 인한 버그 |
| 뮤텍스 (Mutex) | 상호 배제 (한 번에 하나만 접근) |
| 조건 변수 (Condition variable) | 조건이 충족될 때까지 대기 |
| 교착 상태 (Deadlock) | 서로의 자원을 기다리며 멈춤 |
| 생산자-소비자 (Producer-consumer) | 데이터 생산/처리를 분리하는 패턴 |
| 스레드 풀 (Thread pool) | 미리 생성된 스레드로 작업 처리 |

---

## 디버깅 팁

### 1. ThreadSanitizer 사용

```bash
gcc -fsanitize=thread -g program.c -o program -lpthread
./program
```

### 2. Helgrind (Valgrind)

```bash
valgrind --tool=helgrind ./program
```

### 3. 흔한 실수

- 뮤텍스 잠금 해제를 잊음 -- 모든 경로에서 `unlock`을 확인하세요
- 조건 변수에서 `while` 대신 `if` 사용 -- 항상 `while`을 사용하세요
- 일관성 없는 잠금 순서 -- 항상 같은 순서로 잠금하세요

---

## 다음 단계

스레드와 동기화를 마스터했다면 다음으로 진행하세요:
- [네트워크 프로그래밍](./12_Network_Programming.md) -- Berkeley 소켓 API로 TCP/UDP 서버와 클라이언트 구축
