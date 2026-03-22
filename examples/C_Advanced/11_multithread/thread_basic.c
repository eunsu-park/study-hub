// thread_basic.c
// Basic thread program

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// Thread function
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

    printf("=== Basic Thread Example ===\n\n");

    // Create thread
    int result = pthread_create(&thread, NULL, print_message, (void*)msg);
    if (result != 0) {
        fprintf(stderr, "Thread creation failed: %d\n", result);
        return 1;
    }

    // Main thread also performs work
    for (int i = 0; i < 5; i++) {
        printf("[Main] Main thread - %d\n", i);
        sleep(1);
    }

    // Wait for thread to finish
    pthread_join(thread, NULL);

    printf("\nAll work completed\n");
    return 0;
}

/*
 * How to compile:
 * gcc thread_basic.c -o thread_basic -pthread
 *
 * Run:
 * ./thread_basic
 *
 * The main thread and the created thread run concurrently.
 */
