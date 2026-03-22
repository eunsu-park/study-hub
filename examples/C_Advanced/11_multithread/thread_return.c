// thread_return.c
// Receiving thread return values
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* calculate_sum(void* arg) {
    int n = *(int*)arg;

    // Dynamically allocate to return result
    long* result = malloc(sizeof(long));
    *result = 0;

    for (int i = 1; i <= n; i++) {
        *result += i;
    }

    printf("Thread: sum from 1 to %d calculated\n", n);
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
