// thread_pool.c
// Thread Pool implementation
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>

#define POOL_SIZE 4
#define QUEUE_SIZE 100

// Task definition
// Why: function pointer + void* arg is the standard C callback pattern — void*
// erases the type so any data can be passed, making the pool generic
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
// Why: broadcast wakes ALL workers blocked in queue_pop — setting shutdown=true
// under the lock ensures workers see the flag consistently and exit their loops
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
    printf("Task %d complete!\n", item->id);

    // Why: the worker (not the submitter) frees the work item — ownership
    // transfers through the queue, so freeing before processing would be UAF
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
    printf("\nProgram terminated\n");

    return 0;
}
