// bank_account.c
// Thread-safe bank account using mutex
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

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

// Why: the balance check and deduction must be atomic (under the same lock) —
// without the lock, two threads could both see balance=100, both pass the check,
// and overdraw the account
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

// Why: even a simple read needs the lock — on some architectures, reading an int
// while another thread writes it can produce a torn (partially updated) value
int account_get_balance(Account* acc) {
    pthread_mutex_lock(&acc->lock);
    int balance = acc->balance;
    pthread_mutex_unlock(&acc->lock);
    return balance;
}

// Transfer (between two accounts)
// Why: acquiring two locks in arbitrary order causes deadlock — thread A locks
// account 1 then waits for 2, while thread B locks 2 then waits for 1.
// Ordering by address guarantees a global lock acquisition order
int account_transfer(Account* from, Account* to, int amount) {
    // Deadlock prevention: always lock in the same order
    // Lock the account with the smaller address first
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
    printf("Expected balance: %d (initial 10000 + deposits 30000 - withdrawals max 30000)\n", 10000);

    account_destroy(acc);
    return 0;
}
