// condition_basic.c
// Condition Variable basics
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

    // Why: "while (!ready)" instead of "if (!ready)" guards against spurious
    // wakeups and lost signals — the condition must be re-verified after each return
    while (!ready) {  // Wait while condition is false
        printf("[Waiter %d] Waiting on condition...\n", id);
        // Why: cond_wait atomically releases the mutex and puts the thread to sleep —
        // this prevents the "missed signal" bug where the signal fires between
        // unlocking and sleeping
        pthread_cond_wait(&cond, &mutex);  // Wait (mutex is released)
    }
    // After waking from pthread_cond_wait, the mutex is re-acquired

    printf("[Waiter %d] Condition met! Starting work\n", id);

    pthread_mutex_unlock(&mutex);
    return NULL;
}

void* signaler(void* arg) {
    (void)arg;

    sleep(2);  // Wait 2 seconds

    // Why: modifying the shared flag (ready) and signaling must both happen under
    // the same mutex — otherwise a waiter might check ready between the set and
    // the signal, see false, and sleep forever
    pthread_mutex_lock(&mutex);
    ready = true;
    printf("[Signaler] Condition set. Sending signal!\n");
    pthread_cond_broadcast(&cond);  // Signal all waiters
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(void) {
    pthread_t waiters[3];
    pthread_t sig;
    int ids[] = {1, 2, 3};

    // Create waiter threads
    for (int i = 0; i < 3; i++) {
        pthread_create(&waiters[i], NULL, waiter, &ids[i]);
    }

    // Create signaler thread
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
