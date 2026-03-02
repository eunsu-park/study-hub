// producer_consumer.c
// Producer-Consumer pattern (bounded buffer)
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>

#define BUFFER_SIZE 5
#define NUM_ITEMS 20

// Bounded buffer
typedef struct {
    int buffer[BUFFER_SIZE];
    int count;      // Current item count
    int in;         // Next insert position
    int out;        // Next extract position

    pthread_mutex_t mutex;
    pthread_cond_t not_full;   // Buffer is not full
    pthread_cond_t not_empty;  // Buffer is not empty

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
    // Why: "while" loop (not "if") protects against spurious wakeups — the POSIX
    // spec allows cond_wait to return even without a signal, so the condition must
    // be re-checked after every wakeup
    while (bb->count == BUFFER_SIZE) {
        printf("[Producer] Buffer full. Waiting...\n");
        pthread_cond_wait(&bb->not_full, &bb->mutex);
    }

    // Insert item
    bb->buffer[bb->in] = item;
    bb->in = (bb->in + 1) % BUFFER_SIZE;
    bb->count++;

    printf("[Producer] Produced item %d (buffer: %d/%d)\n",
           item, bb->count, BUFFER_SIZE);

    // Why: signal wakes ONE waiting consumer — if no consumer is waiting, the
    // signal is lost (not queued), but that's fine because the consumer will
    // check the count when it next tries to get an item
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

    // If buffer is empty and production is done, exit
    if (bb->count == 0 && bb->done) {
        pthread_mutex_unlock(&bb->mutex);
        return 0;  // No more items
    }

    // Extract item
    *item = bb->buffer[bb->out];
    bb->out = (bb->out + 1) % BUFFER_SIZE;
    bb->count--;

    printf("[Consumer] Consumed item %d (buffer: %d/%d)\n",
           *item, bb->count, BUFFER_SIZE);

    // Notify producer
    pthread_cond_signal(&bb->not_full);

    pthread_mutex_unlock(&bb->mutex);
    return 1;  // Success
}

void buffer_set_done(BoundedBuffer* bb) {
    pthread_mutex_lock(&bb->mutex);
    bb->done = true;
    // Why: broadcast (not signal) wakes ALL consumers — if only one is woken,
    // the others remain blocked forever waiting for items that will never come
    pthread_cond_broadcast(&bb->not_empty);  // Wake all consumers
    pthread_mutex_unlock(&bb->mutex);
}

// Producer thread
void* producer(void* arg) {
    BoundedBuffer* bb = (BoundedBuffer*)arg;

    for (int i = 1; i <= NUM_ITEMS; i++) {
        usleep((rand() % 500) * 1000);  // Wait 0~500ms
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
    printf("\nProgram terminated\n");

    return 0;
}
