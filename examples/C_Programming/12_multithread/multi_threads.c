// multi_threads.c
// Multiple thread creation example
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

// Data to pass to threads
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

    printf("Thread %d finished: sum = %d\n", data->id, sum);

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

    printf("Program terminated\n");
    return 0;
}
