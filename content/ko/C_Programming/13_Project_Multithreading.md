# 프로젝트 12: 멀티스레드 프로그래밍

**이전**: [프로젝트 11: 미니 쉘](./12_Project_Mini_Shell.md) | **다음**: [임베디드 프로그래밍 기초](./14_Embedded_Basics.md)

pthread 라이브러리를 사용한 멀티스레드 프로그래밍을 배웁니다.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `pthread_create`와 `pthread_join`을 사용하여 POSIX 스레드(POSIX thread)를 생성하고 관리할 수 있습니다
2. 구조체를 통해 스레드에 데이터를 전달하고 반환값을 받을 수 있습니다
3. 공유 변수에 대한 비동기 접근으로 인한 경쟁 조건(race condition)을 식별할 수 있습니다
4. 뮤텍스 잠금(mutex lock)을 사용하여 공유 데이터 접근을 동기화하고 데이터 손상을 방지할 수 있습니다
5. 적절한 대기 루프와 함께 조건 변수(condition variable)를 사용하여 스레드 실행을 조율할 수 있습니다
6. 경계 버퍼(bounded buffer)를 사용하여 생산자-소비자(producer-consumer) 패턴을 구현할 수 있습니다
7. 동시성 큐에서 작업을 분배하는 스레드 풀(thread pool)을 구축할 수 있습니다
8. 읽기-쓰기 잠금(read-write lock)을 사용하여 동시 읽기는 허용하고 쓰기는 배타적으로 처리할 수 있습니다

---

동시성(concurrency)은 시스템 프로그래밍에서 가장 도전적인 영역 중 하나이지만, 동시에 가장 보람 있는 영역이기도 합니다. 수천 개의 연결을 처리하는 웹 서버를 구축하든, 병렬로 숫자를 처리하는 데이터 파이프라인을 만들든, 스레드(thread)·뮤텍스(mutex)·조건 변수(condition variable)를 이해하면 현대의 멀티코어 하드웨어를 최대한 활용할 수 있는 도구를 갖추게 됩니다. 이 프로젝트는 첫 번째 스레드 생성부터 재사용 가능한 스레드 풀 구현까지 단계별로 안내합니다.

## 사전 지식
- 포인터
- 구조체
- 함수 포인터

---

## 1단계: 스레드 기초

### 첫 번째 스레드 프로그램

```c
// thread_basic.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// 스레드 함수: void* 반환, void* 인자
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

    // 스레드 생성
    int result = pthread_create(&thread, NULL, print_message, (void*)msg);
    if (result != 0) {
        fprintf(stderr, "Thread creation failed: %d\n", result);
        return 1;
    }

    // 메인 스레드도 작업 수행
    for (int i = 0; i < 5; i++) {
        printf("[Main] Main thread - %d\n", i);
        sleep(1);
    }

    // 스레드 종료 대기
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

// 스레드에 전달할 데이터
typedef struct {
    int id;
    char name[32];
} ThreadData;

void* thread_func(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    printf("Thread %d (%s) started\n", data->id, data->name);

    // 작업 시뮬레이션
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

    // 스레드 생성
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

    // 모든 스레드 대기
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Program finished\n");
    return 0;
}
```

### 스레드 반환값 받기

```c
// thread_return.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* calculate_sum(void* arg) {
    int n = *(int*)arg;

    // 동적 할당하여 결과 반환
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

    // 반환값 받기
    void* ret_val;
    pthread_join(thread, &ret_val);

    long* result = (long*)ret_val;
    printf("Result: %ld\n", *result);

    free(result);  // 동적 할당된 메모리 해제
    return 0;
}
```

---

## 2단계: 경쟁 조건(Race Condition)

여러 스레드가 동시에 공유 데이터에 접근하면 문제가 발생합니다.

### 경쟁 조건 예제

```c
// race_condition.c
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define ITERATIONS 100000

// 공유 변수
int counter = 0;

void* increment(void* arg) {
    (void)arg;

    for (int i = 0; i < ITERATIONS; i++) {
        counter++;  // 원자적이지 않음!
        // 실제로는: temp = counter; temp = temp + 1; counter = temp;
    }

    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];

    // 스레드 생성
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }

    // 대기
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // 예상: NUM_THREADS * ITERATIONS = 1,000,000
    // 실제: 그보다 적은 값 (경쟁 조건으로 인한 손실)
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

## 3단계: 뮤텍스(Mutex)

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
        pthread_mutex_lock(&mutex);    // 잠금
        counter++;                      // 임계 구역(critical section)
        pthread_mutex_unlock(&mutex);  // 해제
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

### 뮤텍스를 이용한 은행 계좌

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
    return -1;  // 잔액 부족
}

int account_get_balance(Account* acc) {
    pthread_mutex_lock(&acc->lock);
    int balance = acc->balance;
    pthread_mutex_unlock(&acc->lock);
    return balance;
}

// 이체 (두 계좌 간)
int account_transfer(Account* from, Account* to, int amount) {
    // 데드락(deadlock) 방지: 항상 같은 순서로 잠금
    // 주소값이 작은 계좌 먼저 잠금
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

// 테스트용 스레드 데이터
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

    // 입금자 3명
    for (int i = 0; i < 3; i++) {
        args[i].acc = acc;
        args[i].thread_id = i;
        pthread_create(&depositors[i], NULL, depositor, &args[i]);
    }

    // 출금자 3명
    for (int i = 0; i < 3; i++) {
        args[i + 3].acc = acc;
        args[i + 3].thread_id = i;
        pthread_create(&withdrawers[i], NULL, withdrawer, &args[i + 3]);
    }

    // 대기
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

## 4단계: 조건 변수(Condition Variable)

특정 조건이 만족될 때까지 스레드를 대기시킵니다.

### 조건 변수 기본

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

    while (!ready) {  // 조건이 false인 동안 대기
        printf("[Waiter %d] Waiting for condition...\n", id);
        pthread_cond_wait(&cond, &mutex);  // 대기 (뮤텍스 해제됨)
    }
    // pthread_cond_wait에서 깨어나면 뮤텍스 다시 획득됨

    printf("[Waiter %d] Condition satisfied! Starting work\n", id);

    pthread_mutex_unlock(&mutex);
    return NULL;
}

void* signaler(void* arg) {
    (void)arg;

    sleep(2);  // 2초 대기

    pthread_mutex_lock(&mutex);
    ready = true;
    printf("[Signaler] Condition set. Broadcasting signal!\n");
    pthread_cond_broadcast(&cond);  // 모든 대기자에게 신호
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(void) {
    pthread_t waiters[3];
    pthread_t sig;
    int ids[] = {1, 2, 3};

    // 대기 스레드 생성
    for (int i = 0; i < 3; i++) {
        pthread_create(&waiters[i], NULL, waiter, &ids[i]);
    }

    // 신호 스레드 생성
    pthread_create(&sig, NULL, signaler, NULL);

    // 대기
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

## 5단계: 생산자-소비자(Producer-Consumer) 패턴

가장 중요한 동기화 패턴 중 하나입니다.

### 경계 버퍼(Bounded Buffer)

```c
// producer_consumer.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>

#define BUFFER_SIZE 5
#define NUM_ITEMS 20

// 경계 버퍼
typedef struct {
    int buffer[BUFFER_SIZE];
    int count;      // 현재 아이템 수
    int in;         // 다음 삽입 위치
    int out;        // 다음 추출 위치

    pthread_mutex_t mutex;
    pthread_cond_t not_full;   // 버퍼가 가득 차지 않음
    pthread_cond_t not_empty;  // 버퍼가 비어있지 않음

    bool done;      // 생산 완료 플래그
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

    // 버퍼가 가득 찼으면 대기
    while (bb->count == BUFFER_SIZE) {
        printf("[Producer] Buffer full. Waiting...\n");
        pthread_cond_wait(&bb->not_full, &bb->mutex);
    }

    // 아이템 삽입
    bb->buffer[bb->in] = item;
    bb->in = (bb->in + 1) % BUFFER_SIZE;
    bb->count++;

    printf("[Producer] Item %d produced (buffer: %d/%d)\n",
           item, bb->count, BUFFER_SIZE);

    // 소비자에게 알림
    pthread_cond_signal(&bb->not_empty);

    pthread_mutex_unlock(&bb->mutex);
}

int buffer_get(BoundedBuffer* bb, int* item) {
    pthread_mutex_lock(&bb->mutex);

    // 버퍼가 비어있고 생산 완료 아니면 대기
    while (bb->count == 0 && !bb->done) {
        printf("[Consumer] Buffer empty. Waiting...\n");
        pthread_cond_wait(&bb->not_empty, &bb->mutex);
    }

    // 버퍼가 비어있고 생산 완료면 종료
    if (bb->count == 0 && bb->done) {
        pthread_mutex_unlock(&bb->mutex);
        return 0;  // 더 이상 아이템 없음
    }

    // 아이템 추출
    *item = bb->buffer[bb->out];
    bb->out = (bb->out + 1) % BUFFER_SIZE;
    bb->count--;

    printf("[Consumer] Item %d consumed (buffer: %d/%d)\n",
           *item, bb->count, BUFFER_SIZE);

    // 생산자에게 알림
    pthread_cond_signal(&bb->not_full);

    pthread_mutex_unlock(&bb->mutex);
    return 1;  // 성공
}

void buffer_set_done(BoundedBuffer* bb) {
    pthread_mutex_lock(&bb->mutex);
    bb->done = true;
    pthread_cond_broadcast(&bb->not_empty);  // 모든 소비자 깨움
    pthread_mutex_unlock(&bb->mutex);
}

// 생산자 스레드
void* producer(void* arg) {
    BoundedBuffer* bb = (BoundedBuffer*)arg;

    for (int i = 1; i <= NUM_ITEMS; i++) {
        usleep((rand() % 500) * 1000);  // 0~500ms 대기
        buffer_put(bb, i);
    }

    printf("[Producer] Production complete\n");
    buffer_set_done(bb);

    return NULL;
}

// 소비자 스레드
void* consumer(void* arg) {
    BoundedBuffer* bb = (BoundedBuffer*)arg;
    int item;

    while (buffer_get(bb, &item)) {
        usleep((rand() % 800) * 1000);  // 0~800ms 처리 시간
    }

    printf("[Consumer] Consumption complete\n");
    return NULL;
}

int main(void) {
    srand(time(NULL));

    BoundedBuffer* bb = buffer_create();

    pthread_t prod;
    pthread_t cons[2];

    // 생산자 1명
    pthread_create(&prod, NULL, producer, bb);

    // 소비자 2명
    pthread_create(&cons[0], NULL, consumer, bb);
    pthread_create(&cons[1], NULL, consumer, bb);

    // 대기
    pthread_join(prod, NULL);
    pthread_join(cons[0], NULL);
    pthread_join(cons[1], NULL);

    buffer_destroy(bb);
    printf("\nProgram finished\n");

    return 0;
}
```

---

## 6단계: 스레드 풀(Thread Pool)

실제 서버 프로그램에서 많이 사용하는 패턴입니다.

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

// 작업 정의
typedef struct Task {
    void (*function)(void* arg);
    void* arg;
} Task;

// 작업 큐
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

// 스레드 풀
typedef struct {
    pthread_t threads[POOL_SIZE];
    TaskQueue queue;
    int thread_count;
} ThreadPool;

// 작업 큐 초기화
void queue_init(TaskQueue* q) {
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->shutdown = false;

    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

// 작업 큐 정리
void queue_destroy(TaskQueue* q) {
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

// 작업 추가
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

// 작업 가져오기
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

// 워커 스레드 함수
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

// 스레드 풀 생성
ThreadPool* pool_create(int size) {
    ThreadPool* pool = malloc(sizeof(ThreadPool));
    pool->thread_count = size;

    queue_init(&pool->queue);

    for (int i = 0; i < size; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }

    return pool;
}

// 작업 제출
bool pool_submit(ThreadPool* pool, void (*function)(void*), void* arg) {
    Task task = { .function = function, .arg = arg };
    return queue_push(&pool->queue, task);
}

// 스레드 풀 종료
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

// ============ 테스트 ============

typedef struct {
    int id;
    int value;
} WorkItem;

void process_work(void* arg) {
    WorkItem* item = (WorkItem*)arg;

    printf("Processing task %d (value: %d)...\n", item->id, item->value);
    usleep((rand() % 500 + 100) * 1000);  // 100~600ms 처리
    printf("Task %d completed!\n", item->id);

    free(item);
}

int main(void) {
    srand(time(NULL));

    printf("Creating thread pool (size: %d)\n\n", POOL_SIZE);
    ThreadPool* pool = pool_create(POOL_SIZE);

    // 작업 제출
    for (int i = 0; i < 10; i++) {
        WorkItem* item = malloc(sizeof(WorkItem));
        item->id = i;
        item->value = rand() % 100;

        printf("Submitting task %d (value: %d)\n", i, item->value);
        pool_submit(pool, process_work, item);

        usleep(100000);  // 100ms 간격
    }

    printf("\nAll tasks submitted. Waiting for pool shutdown...\n\n");
    sleep(2);  // 작업 처리 대기

    pool_shutdown(pool);
    printf("\nProgram finished\n");

    return 0;
}
```

---

## 7단계: 읽기-쓰기 잠금(Read-Write Lock)

읽기는 동시에, 쓰기는 배타적으로 허용합니다.

```c
// rwlock_example.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_READERS 5
#define NUM_WRITERS 2

// 공유 데이터
typedef struct {
    int data;
    pthread_rwlock_t lock;
} SharedData;

SharedData shared = { .data = 0 };

void* reader(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 5; i++) {
        pthread_rwlock_rdlock(&shared.lock);  // 읽기 잠금

        printf("[Reader %d] Read data: %d\n", id, shared.data);
        usleep(100000);  // 읽기 중...

        pthread_rwlock_unlock(&shared.lock);

        usleep(rand() % 200000);
    }

    return NULL;
}

void* writer(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 3; i++) {
        pthread_rwlock_wrlock(&shared.lock);  // 쓰기 잠금 (배타적)

        shared.data = rand() % 1000;
        printf("[Writer %d] Wrote data: %d\n", id, shared.data);
        usleep(200000);  // 쓰기 중...

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

    // 독자 생성
    for (int i = 0; i < NUM_READERS; i++) {
        reader_ids[i] = i;
        pthread_create(&readers[i], NULL, reader, &reader_ids[i]);
    }

    // 작가 생성
    for (int i = 0; i < NUM_WRITERS; i++) {
        writer_ids[i] = i;
        pthread_create(&writers[i], NULL, writer, &writer_ids[i]);
    }

    // 대기
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

## 8단계: 실전 예제 - 병렬 정렬

### 멀티스레드 병합 정렬

```c
// parallel_sort.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define THRESHOLD 10000  // 이보다 작으면 단일 스레드

typedef struct {
    int* arr;
    int left;
    int right;
} SortTask;

// 병합
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

// 단일 스레드 병합 정렬
void merge_sort_single(int* arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_single(arr, left, mid);
        merge_sort_single(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// 멀티스레드 병합 정렬
void* merge_sort_parallel(void* arg) {
    SortTask* task = (SortTask*)arg;
    int* arr = task->arr;
    int left = task->left;
    int right = task->right;

    if (left >= right) return NULL;

    // 작은 배열은 단일 스레드로
    if (right - left < THRESHOLD) {
        merge_sort_single(arr, left, right);
        return NULL;
    }

    int mid = left + (right - left) / 2;

    // 왼쪽 절반: 새 스레드
    SortTask left_task = { arr, left, mid };
    pthread_t left_thread;
    pthread_create(&left_thread, NULL, merge_sort_parallel, &left_task);

    // 오른쪽 절반: 현재 스레드
    SortTask right_task = { arr, mid + 1, right };
    merge_sort_parallel(&right_task);

    // 왼쪽 스레드 대기
    pthread_join(left_thread, NULL);

    // 병합
    merge(arr, left, mid, right);

    return NULL;
}

// 배열 출력
void print_array(int* arr, int n) {
    for (int i = 0; i < n && i < 20; i++) {
        printf("%d ", arr[i]);
    }
    if (n > 20) printf("...");
    printf("\n");
}

// 배열 검증
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

    // 랜덤 배열 생성
    for (int i = 0; i < n; i++) {
        arr1[i] = rand();
        arr2[i] = arr1[i];  // 복사
    }

    printf("Array size: %d\n\n", n);

    // 단일 스레드 정렬
    clock_t start = clock();
    merge_sort_single(arr1, 0, n - 1);
    clock_t end = clock();
    double single_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Single-threaded: %.3f seconds\n", single_time);
    printf("Sort verification: %s\n\n", is_sorted(arr1, n) ? "OK" : "FAIL");

    // 멀티스레드 정렬
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

## 연습 문제

### 연습 1: 식사하는 철학자(Dining Philosophers)
5명의 철학자가 원탁에 앉아 있고, 젓가락 5개가 있습니다.
- 철학자는 생각하거나 식사합니다
- 식사하려면 양쪽 젓가락이 필요합니다
- 데드락(deadlock) 없이 구현하세요

### 연습 2: 장벽(Barrier)
N개의 스레드가 모두 도착할 때까지 대기하는 장벽을 구현하세요.

```c
typedef struct {
    int count;
    int threshold;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Barrier;

void barrier_wait(Barrier* b);
```

### 연습 3: 세마포어(Semaphore) 구현
뮤텍스와 조건 변수를 사용하여 카운팅 세마포어(counting semaphore)를 구현하세요.

```c
typedef struct {
    int value;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Semaphore;

void sem_wait(Semaphore* sem);
void sem_post(Semaphore* sem);
```

### 연습 4: 병렬 행렬 곱셈
N×N 행렬 곱셈을 여러 스레드로 나누어 계산하세요.

---

## 핵심 개념 정리

| 함수 | 설명 |
|------|------|
| `pthread_create()` | 스레드 생성 |
| `pthread_join()` | 스레드 종료 대기 |
| `pthread_mutex_lock()` | 뮤텍스 잠금 |
| `pthread_mutex_unlock()` | 뮤텍스 해제 |
| `pthread_cond_wait()` | 조건 대기 |
| `pthread_cond_signal()` | 하나의 대기자 깨움 |
| `pthread_cond_broadcast()` | 모든 대기자 깨움 |

| 개념 | 설명 |
|------|------|
| 경쟁 조건(race condition) | 여러 스레드의 동시 접근으로 인한 버그 |
| 뮤텍스(mutex) | 상호 배제(mutual exclusion) - 한 번에 하나만 접근 |
| 조건 변수(condition variable) | 조건 만족까지 대기 |
| 데드락(deadlock) | 서로 상대방의 자원을 기다리며 멈춤 |
| 생산자-소비자(producer-consumer) | 데이터 생성/처리 분리 패턴 |
| 스레드 풀(thread pool) | 미리 생성된 스레드로 작업 처리 |

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

### 3. 일반적인 실수

- 뮤텍스 해제 잊음 → 모든 경로에서 `unlock` 확인
- 조건 변수에서 `while` 대신 `if` 사용 → 항상 `while` 사용
- 잠금 순서 불일치 → 항상 같은 순서로 잠금

---

## C 언어 프로젝트 완료!

이 12개의 프로젝트를 완료했다면 C 언어의 핵심 개념을 모두 경험한 것입니다:

1. ✅ 환경 설정
2. ✅ C 기초 빠른 복습
3. ✅ 계산기 (함수, 조건문)
4. ✅ 숫자 맞추기 (반복문, 랜덤)
5. ✅ 주소록 (구조체, 파일 I/O)
6. ✅ 동적 배열 (메모리 관리)
7. ✅ 연결 리스트 (포인터 심화)
8. ✅ 파일 암호화 (비트 연산)
9. ✅ 스택과 큐 (자료구조)
10. ✅ 해시 테이블 (해시 함수)
11. ✅ 뱀 게임 (터미널 제어)
12. ✅ 미니 쉘 (프로세스 관리)
13. ✅ 멀티스레드(multithreading) (동시성)

다음 학습 추천:
- 네트워크 프로그래밍 (소켓)
- 시스템 콜 심화
- 리눅스 커널 모듈
- 임베디드 시스템

---

**이전**: [프로젝트 11: 미니 쉘](./12_Project_Mini_Shell.md) | **다음**: [임베디드 프로그래밍 기초](./14_Embedded_Basics.md)
