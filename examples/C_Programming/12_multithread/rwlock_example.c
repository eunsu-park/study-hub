// rwlock_example.c
// Read-Write Lock example
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define NUM_READERS 5
#define NUM_WRITERS 2

// Shared data
// Why: rwlock is ideal when reads vastly outnumber writes (5 readers vs 2 writers
// here) — a plain mutex would serialize all 7 threads, but rwlock lets 5 readers
// proceed in parallel
typedef struct {
    int data;
    pthread_rwlock_t lock;
} SharedData;

SharedData shared = { .data = 0 };

void* reader(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 5; i++) {
        // Why: rdlock allows multiple readers simultaneously — unlike mutex, readers
        // don't block each other, giving much better throughput for read-heavy workloads
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
        // Why: wrlock is exclusive — it blocks ALL readers and other writers,
        // ensuring the write is seen atomically (no partial updates observed)
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
    printf("Done\n");

    return 0;
}
