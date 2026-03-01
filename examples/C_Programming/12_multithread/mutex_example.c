// mutex_example.c
// 뮤텍스를 이용한 동기화
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define ITERATIONS 100000

int counter = 0;
// Why: PTHREAD_MUTEX_INITIALIZER is a static initializer — it avoids the need for
// pthread_mutex_init() for global/static mutexes, reducing boilerplate
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_safe(void* arg) {
    (void)arg;

    for (int i = 0; i < ITERATIONS; i++) {
        // Why: lock-increment-unlock ensures only one thread executes counter++
        // at a time — this makes the load-increment-store sequence atomic, fixing
        // the race condition shown in race_condition.c
        pthread_mutex_lock(&mutex);    // 잠금
        counter++;                      // 임계 구역
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

    printf("예상값: %d\n", NUM_THREADS * ITERATIONS);
    printf("실제값: %d\n", counter);

    // Why: destroy releases OS resources held by the mutex — not strictly required
    // before exit, but essential in libraries/long-running programs to avoid leaks
    pthread_mutex_destroy(&mutex);
    return 0;
}
