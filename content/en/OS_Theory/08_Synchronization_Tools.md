# Synchronization Tools

**Previous**: [Synchronization Basics](./07_Synchronization_Basics.md) | **Next**: [Deadlock](./09_Deadlock.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish a mutex from a binary semaphore and explain the ownership difference
2. Implement counting semaphores for resource pooling
3. Solve the Producer-Consumer problem using semaphores
4. Solve the Readers-Writers and Dining Philosophers problems
5. Explain how monitors simplify synchronization by encapsulating shared state

---

Raw locks and test-and-set are too low-level for most programmers. Higher-level tools -- semaphores, monitors, condition variables -- provide structured ways to coordinate concurrent access. These are the building blocks used inside every database, web server, and operating system kernel.

## Table of Contents

1. [Mutex](#1-mutex)
2. [Semaphore](#2-semaphore)
3. [Monitor](#3-monitor)
4. [Condition Variable](#4-condition-variable)
5. [Classic Synchronization Problems](#5-classic-synchronization-problems)
6. [Practice Problems](#6-practice-problems)

---

## 1. Mutex

### Concept

```
┌─────────────────────────────────────────────────────────┐
│                    Mutex (Mutual Exclusion)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Mutex = Short for Mutual Exclusion                     │
│        = Binary Lock                                    │
│        = Allows only one thread to enter critical       │
│          section at a time                              │
│                                                         │
│  States:                                                │
│  • Locked: One thread owns the lock                     │
│  • Unlocked: Available for use                          │
│                                                         │
│  Basic Operations:                                      │
│  • lock(): Acquire lock (other threads wait)            │
│  • unlock(): Release lock                               │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │          Mutex Operation Visualization          │    │
│  │                                                 │    │
│  │  Thread1: ─lock()─┬───critical section───┬─unlock()─ │
│  │  Thread2: ─lock()─│─waiting──────────────│─critical─ │
│  │                   │                      ▲           │
│  │                   └──────────────────────┘           │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Using pthread_mutex

```c
#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        pthread_mutex_lock(&mutex);    // Acquire lock
        counter++;                      // Critical section
        pthread_mutex_unlock(&mutex);  // Release lock
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", counter);  // 2000000 (correct!)
    return 0;
}
```

### Mutex Initialization Methods

```c
// Method 1: Static initialization
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Method 2: Dynamic initialization
pthread_mutex_t mutex;
pthread_mutex_init(&mutex, NULL);  // NULL attributes = default

// Cleanup (for dynamic initialization)
pthread_mutex_destroy(&mutex);

// Attribute setting example
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);  // Recursive mutex
pthread_mutex_init(&mutex, &attr);
pthread_mutexattr_destroy(&attr);
```

### Mutex Types

```
┌─────────────────────────────────────────────────────────┐
│                    Mutex Types                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. PTHREAD_MUTEX_NORMAL (default)                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Recursive lock causes deadlock                 │  │
│  │  • Unlock by non-owner thread is undefined        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. PTHREAD_MUTEX_RECURSIVE                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Same thread can lock multiple times            │  │
│  │  • Requires unlock as many times as locked        │  │
│  │  • Useful in recursive functions                  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. PTHREAD_MUTEX_ERRORCHECK                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Returns error on recursive lock                │  │
│  │  • Returns error on unlock by non-owner           │  │
│  │  • For debugging                                  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Semaphore

### Concept

```
┌─────────────────────────────────────────────────────────┐
│                   Semaphore                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Proposed by Dijkstra in 1965                           │
│  = Synchronization object with an integer value         │
│  = Can manage multiple resources                        │
│                                                         │
│  Basic Operations:                                      │
│  • wait() / P() / down() / acquire()                    │
│    - If value > 0, decrement and proceed                │
│    - If value = 0, wait                                 │
│                                                         │
│  • signal() / V() / up() / release()                    │
│    - Increment value                                    │
│    - Wake up waiting process if any                     │
│                                                         │
│  Types:                                                 │
│  • Binary semaphore: 0 or 1 (similar to mutex)          │
│  • Counting semaphore: Integer ≥ 0                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### P/V Operations (wait/signal)

```
┌─────────────────────────────────────────────────────────┐
│                    P/V Operation Definition              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  P(S) / wait(S):                                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  wait(S) {                                        │  │
│  │      while (S <= 0) {                             │  │
│  │          // wait (busy waiting or blocking)       │  │
│  │      }                                            │  │
│  │      S = S - 1;                                   │  │
│  │  }                                                │  │
│  │  // P = Proberen (Dutch: to test)                │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  V(S) / signal(S):                                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  signal(S) {                                      │  │
│  │      S = S + 1;                                   │  │
│  │      // wake up waiting process if any            │  │
│  │  }                                                │  │
│  │  // V = Verhogen (Dutch: to increment)           │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Important: P and V must be performed atomically        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Using POSIX Semaphore

```c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t semaphore;
int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        sem_wait(&semaphore);    // P operation: wait and decrement
        counter++;
        sem_post(&semaphore);    // V operation: increment and signal
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    // Initialize semaphore (value=1: binary semaphore)
    sem_init(&semaphore, 0, 1);  // 0=shared between threads

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    sem_destroy(&semaphore);

    printf("Counter: %d\n", counter);  // 2000000
    return 0;
}
```

### Counting Semaphore Example

```c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define MAX_CONNECTIONS 3

sem_t connection_sem;

void* client(void* arg) {
    int id = *(int*)arg;

    printf("Client %d: waiting for connection\n", id);

    sem_wait(&connection_sem);  // Acquire connection slot

    printf("Client %d: connected (working...)\n", id);
    sleep(2);  // Simulate work

    printf("Client %d: disconnected\n", id);
    sem_post(&connection_sem);  // Return connection slot

    return NULL;
}

int main() {
    pthread_t threads[10];
    int ids[10];

    // Allow maximum 3 simultaneous connections
    sem_init(&connection_sem, 0, MAX_CONNECTIONS);

    for (int i = 0; i < 10; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, client, &ids[i]);
    }

    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }

    sem_destroy(&connection_sem);
    return 0;
}

/*
Example output:
Client 0: waiting for connection
Client 0: connected (working...)
Client 1: waiting for connection
Client 1: connected (working...)
Client 2: waiting for connection
Client 2: connected (working...)
Client 3: waiting for connection       <- More than 3 wait
...
*/
```

### Mutex vs Semaphore

```
┌──────────────────┬─────────────────────┬─────────────────────┐
│      Feature      │       Mutex         │      Semaphore      │
├──────────────────┼─────────────────────┼─────────────────────┤
│ Value range      │ 0 or 1              │ ≥ 0                 │
├──────────────────┼─────────────────────┼─────────────────────┤
│ Ownership        │ Yes (only owner can │ No (anyone can      │
│                  │ unlock)             │ signal)             │
├──────────────────┼─────────────────────┼─────────────────────┤
│ Purpose          │ Mutual exclusion    │ Resource counting,  │
│                  │                     │ signaling           │
├──────────────────┼─────────────────────┼─────────────────────┤
│ Recursive        │ Possible (RECURSIVE)│ Not possible        │
│ acquisition      │                     │                     │
├──────────────────┼─────────────────────┼─────────────────────┤
│ Priority         │ Can be supported    │ Typically not       │
│ inheritance      │                     │ supported           │
├──────────────────┼─────────────────────┼─────────────────────┤
│ Use cases        │ Protect shared data │ Producer-consumer,  │
│                  │                     │ connection pool     │
└──────────────────┴─────────────────────┴─────────────────────┘
```

---

## 3. Monitor

### Concept

```
┌─────────────────────────────────────────────────────────┐
│                    Monitor                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Monitor = High-level abstraction encapsulating sync    │
│          = Combines shared data + operations + sync     │
│          = Only one process can enter monitor at a time │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                   Monitor                       │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         Shared Data (Private)           │   │    │
│  │  │         int counter;                    │   │    │
│  │  │         int buffer[N];                  │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         Condition Variables             │   │    │
│  │  │         condition notEmpty;             │   │    │
│  │  │         condition notFull;              │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         Procedures (Public)             │   │    │
│  │  │         void insert(int item);          │   │    │
│  │  │         int remove();                   │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                │    │
│  │          ← Entry Queue                         │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Features:                                              │
│  • Compiler automatically ensures mutual exclusion      │
│  • Java synchronized, Python Lock, etc.                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Monitor in Java

```java
// Monitor using Java synchronized
public class Counter {
    private int count = 0;

    // synchronized method = monitor procedure
    public synchronized void increment() {
        count++;  // Mutual exclusion automatically guaranteed
    }

    public synchronized void decrement() {
        count--;
    }

    public synchronized int getCount() {
        return count;
    }
}

// Usage example
Counter counter = new Counter();

Thread t1 = new Thread(() -> {
    for (int i = 0; i < 1000000; i++) {
        counter.increment();
    }
});

Thread t2 = new Thread(() -> {
    for (int i = 0; i < 1000000; i++) {
        counter.increment();
    }
});

t1.start(); t2.start();
t1.join(); t2.join();
System.out.println(counter.getCount());  // 2000000
```

---

## 4. Condition Variable

### Concept

```
┌─────────────────────────────────────────────────────────┐
│               Condition Variable                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Condition Variable = Tool to wait until a condition    │
│                       is true                           │
│                     = Used with mutex                   │
│                                                         │
│  Operations:                                            │
│  • wait(cond, mutex):                                   │
│    1. Release mutex                                     │
│    2. Wait on condition variable                        │
│    3. Re-acquire mutex when awakened                    │
│                                                         │
│  • signal(cond) / pthread_cond_signal():                │
│    - Wake up one waiting thread                         │
│                                                         │
│  • broadcast(cond) / pthread_cond_broadcast():          │
│    - Wake up all waiting threads                        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                   Operation Flow                │    │
│  │                                                 │    │
│  │  Thread A           Thread B                   │    │
│  │  ─────────           ─────────                 │    │
│  │  lock(mutex)                                   │    │
│  │  while (!condition)                            │    │
│  │      wait(cond) ───┐                           │    │
│  │          │ release  │                          │    │
│  │          │ mutex    │  lock(mutex)             │    │
│  │          │ wait     │  modify condition        │    │
│  │          │          │  signal(cond)            │    │
│  │          │◀─────────│  unlock(mutex)           │    │
│  │  re-acquire mutex                              │    │
│  │  continue critical section                     │    │
│  │  unlock(mutex)                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Using pthread Condition Variable

```c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool ready = false;
int data = 0;

void* producer(void* arg) {
    pthread_mutex_lock(&mutex);

    data = 42;
    ready = true;
    printf("Producer: data ready\n");

    pthread_cond_signal(&cond);  // Wake up consumer
    pthread_mutex_unlock(&mutex);

    return NULL;
}

void* consumer(void* arg) {
    pthread_mutex_lock(&mutex);

    while (!ready) {  // Use while loop! (prevent spurious wakeup)
        printf("Consumer: waiting for data...\n");
        pthread_cond_wait(&cond, &mutex);
    }

    printf("Consumer: received data = %d\n", data);
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main() {
    pthread_t prod, cons;

    pthread_create(&cons, NULL, consumer, NULL);
    sleep(1);  // Let consumer wait first
    pthread_create(&prod, NULL, producer, NULL);

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    return 0;
}
```

### Spurious Wakeup

```
┌─────────────────────────────────────────────────────────┐
│               Spurious Wakeup                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Problem: Can wake up from wait without signal          │
│                                                         │
│  Incorrect code:                                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  if (!ready) {                                    │  │
│  │      pthread_cond_wait(&cond, &mutex);  // Danger! │  │
│  │  }                                                │  │
│  │  // May execute when condition is not true        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Correct code:                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  while (!ready) {                                 │  │
│  │      pthread_cond_wait(&cond, &mutex);            │  │
│  │  }                                                │  │
│  │  // Recheck condition when awakened               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Rule: Always call wait() inside a while loop!          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Classic Synchronization Problems

### Producer-Consumer Problem (Bounded Buffer)

```
┌─────────────────────────────────────────────────────────┐
│              Producer-Consumer Problem                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Setup:                                                 │
│  • Fixed-size buffer (N items)                          │
│  • Producer: Add items to buffer                        │
│  • Consumer: Remove items from buffer                   │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                                                   │  │
│  │  Producer ──▶ [  Buffer  ] ──▶ Consumer          │  │
│  │            ┌───────────┐                          │  │
│  │            │ ? │ ? │ ? │                          │  │
│  │            └───────────┘                          │  │
│  │            N = 3                                  │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Synchronization requirements:                          │
│  1. Producer waits when buffer is full                  │
│  2. Consumer waits when buffer is empty                 │
│  3. Mutual exclusion when accessing buffer              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 5

int buffer[BUFFER_SIZE];
int in = 0, out = 0;

sem_t empty;  // Number of empty slots (initial: BUFFER_SIZE)
sem_t full;   // Number of full slots (initial: 0)
pthread_mutex_t mutex;

void* producer(void* arg) {
    for (int i = 0; i < 10; i++) {
        int item = i;

        sem_wait(&empty);           // Wait for empty slot
        pthread_mutex_lock(&mutex);

        buffer[in] = item;
        printf("Produce: %d (position %d)\n", item, in);
        in = (in + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&mutex);
        sem_post(&full);            // Increment full slots

        usleep(100000);  // 0.1s
    }
    return NULL;
}

void* consumer(void* arg) {
    for (int i = 0; i < 10; i++) {
        sem_wait(&full);            // Wait for full slot
        pthread_mutex_lock(&mutex);

        int item = buffer[out];
        printf("Consume: %d (position %d)\n", item, out);
        out = (out + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&mutex);
        sem_post(&empty);           // Increment empty slots

        usleep(150000);  // 0.15s
    }
    return NULL;
}

int main() {
    pthread_t prod, cons;

    sem_init(&empty, 0, BUFFER_SIZE);
    sem_init(&full, 0, 0);
    pthread_mutex_init(&mutex, NULL);

    pthread_create(&prod, NULL, producer, NULL);
    pthread_create(&cons, NULL, consumer, NULL);

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    return 0;
}
```

### Readers-Writers Problem

```
┌─────────────────────────────────────────────────────────┐
│                 Readers-Writers Problem                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Setup:                                                 │
│  • Shared database                                      │
│  • Readers: Only read data                              │
│  • Writers: Modify data                                 │
│                                                         │
│  Rules:                                                 │
│  • Multiple readers can read simultaneously             │
│  • No access while writer is writing                    │
│  • Writer needs exclusive access                        │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                                                   │  │
│  │  Reader1 ──read──┐                                │  │
│  │  Reader2 ──read──┼──▶ [ Database ]                │  │
│  │  Reader3 ──read──┘          ↑                     │  │
│  │                             │                     │  │
│  │  Writer  ────────write──────┘                     │  │
│  │       (exclusive access)                          │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t write_lock = PTHREAD_MUTEX_INITIALIZER;
int read_count = 0;
int shared_data = 0;

void* reader(void* arg) {
    int id = *(int*)arg;

    pthread_mutex_lock(&mutex);
    read_count++;
    if (read_count == 1) {
        pthread_mutex_lock(&write_lock);  // First reader blocks writers
    }
    pthread_mutex_unlock(&mutex);

    // Reading (not critical section, multiple readers allowed)
    printf("Reader %d: data = %d\n", id, shared_data);
    usleep(100000);

    pthread_mutex_lock(&mutex);
    read_count--;
    if (read_count == 0) {
        pthread_mutex_unlock(&write_lock);  // Last reader allows writers
    }
    pthread_mutex_unlock(&mutex);

    return NULL;
}

void* writer(void* arg) {
    int id = *(int*)arg;

    pthread_mutex_lock(&write_lock);

    // Writing (exclusive access)
    shared_data++;
    printf("Writer %d: changed data to %d\n", id, shared_data);
    usleep(200000);

    pthread_mutex_unlock(&write_lock);

    return NULL;
}

int main() {
    pthread_t readers[5], writers[2];
    int ids[5] = {1, 2, 3, 4, 5};
    int wids[2] = {1, 2};

    for (int i = 0; i < 5; i++)
        pthread_create(&readers[i], NULL, reader, &ids[i]);
    for (int i = 0; i < 2; i++)
        pthread_create(&writers[i], NULL, writer, &wids[i]);

    for (int i = 0; i < 5; i++)
        pthread_join(readers[i], NULL);
    for (int i = 0; i < 2; i++)
        pthread_join(writers[i], NULL);

    return 0;
}
```

### Dining Philosophers Problem

```
┌─────────────────────────────────────────────────────────┐
│              Dining Philosophers Problem                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Setup:                                                 │
│  • 5 philosophers, 5 chopsticks                         │
│  • Each philosopher either thinks or eats               │
│  • Need both adjacent chopsticks to eat                 │
│                                                         │
│           ┌───────────────────────────┐                 │
│           │          P0              │                 │
│           │      ◇       ◇          │                 │
│           │     C4       C0          │                 │
│           │                          │                 │
│        P4 ◇                      ◇ P1│                 │
│           │   C3             C1     │                 │
│           │                          │                 │
│        P3 ◇──────C2──────◇ P2       │                 │
│           │                          │                 │
│           └───────────────────────────┘                 │
│                                                         │
│  Problem: Deadlock possible!                            │
│  If all philosophers pick up left chopstick → no one eats│
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define N 5
#define LEFT(i) (i)
#define RIGHT(i) ((i + 1) % N)

pthread_mutex_t chopsticks[N];

void* philosopher(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 3; i++) {
        // Thinking
        printf("Philosopher %d: thinking...\n", id);
        usleep(100000);

        // Deadlock prevention: even philosophers pick left first, odd pick right first
        if (id % 2 == 0) {
            pthread_mutex_lock(&chopsticks[LEFT(id)]);
            pthread_mutex_lock(&chopsticks[RIGHT(id)]);
        } else {
            pthread_mutex_lock(&chopsticks[RIGHT(id)]);
            pthread_mutex_lock(&chopsticks[LEFT(id)]);
        }

        // Eating
        printf("Philosopher %d: eating...\n", id);
        usleep(200000);

        // Put down chopsticks
        pthread_mutex_unlock(&chopsticks[LEFT(id)]);
        pthread_mutex_unlock(&chopsticks[RIGHT(id)]);
    }

    return NULL;
}

int main() {
    pthread_t philosophers[N];
    int ids[N];

    for (int i = 0; i < N; i++)
        pthread_mutex_init(&chopsticks[i], NULL);

    for (int i = 0; i < N; i++) {
        ids[i] = i;
        pthread_create(&philosophers[i], NULL, philosopher, &ids[i]);
    }

    for (int i = 0; i < N; i++)
        pthread_join(philosophers[i], NULL);

    return 0;
}
```

---

## 6. Practice Problems

### Problem 1: Semaphore Value

For a semaphore with initial value 5, what is the final value after performing P, P, V, P, P, P operations in order?

<details>
<summary>Show Answer</summary>

**Operation sequence and value changes:**
- Initial value: 5
- P: 5 - 1 = 4
- P: 4 - 1 = 3
- V: 3 + 1 = 4
- P: 4 - 1 = 3
- P: 3 - 1 = 2
- P: 2 - 1 = 1

**Final value: 1**

</details>

### Problem 2: Producer-Consumer

Explain the roles of the empty and full semaphores in the producer-consumer problem, and why their order is important.

<details>
<summary>Show Answer</summary>

**empty semaphore:**
- Manages number of empty buffer slots
- Initial value: buffer size (N)
- Producer performs P operation (allocate slot)
- Consumer performs V operation (return slot)

**full semaphore:**
- Manages number of filled buffer slots
- Initial value: 0
- Producer performs V operation (notify item added)
- Consumer performs P operation (wait for item)

**Why order is important:**
- Order of semaphore P operation and mutex acquisition
- Wrong order: mutex lock → sem_wait → possible deadlock
- Correct order: sem_wait → mutex lock → prevent deadlock

</details>

### Problem 3: Monitor Implementation

Convert the following semaphore code to monitor (condition variable) style.

```c
sem_t sem;
sem_init(&sem, 0, 0);

// Producer
sem_post(&sem);

// Consumer
sem_wait(&sem);
```

<details>
<summary>Show Answer</summary>

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int count = 0;

// Producer
pthread_mutex_lock(&mutex);
count++;
pthread_cond_signal(&cond);
pthread_mutex_unlock(&mutex);

// Consumer
pthread_mutex_lock(&mutex);
while (count == 0) {
    pthread_cond_wait(&cond, &mutex);
}
count--;
pthread_mutex_unlock(&mutex);
```

</details>

### Problem 4: Dining Philosophers Deadlock

Explain a deadlock scenario in the dining philosophers problem and propose three solution methods.

<details>
<summary>Show Answer</summary>

**Deadlock scenario:**
If all philosophers pick up their left chopstick simultaneously:
- P0: holds C0, waits for C4
- P1: holds C1, waits for C0
- P2: holds C2, waits for C1
- P3: holds C3, waits for C2
- P4: holds C4, waits for C3
→ Circular wait → Deadlock!

**Solution methods:**

1. **Asymmetric lock acquisition:**
   - Even philosophers: pick left first
   - Odd philosophers: pick right first
   - Breaks circular wait

2. **Simultaneous chopstick acquisition:**
   - Pick up both chopsticks only when both are available
   - Protected by central mutex

3. **Limit to N-1 philosophers:**
   - Use semaphore to limit maximum 4 at table
   - At least one can use both chopsticks

</details>

### Problem 5: Condition Variable Usage

Explain why wait() calls on condition variables should be inside a while loop.

<details>
<summary>Show Answer</summary>

**Reason 1: Spurious Wakeup**
- Can wake up without signal from the system
- Will proceed in incorrect state without rechecking condition

**Reason 2: Multiple Waiters**
- Multiple threads may be waiting on same condition variable
- After broadcast, all wake up but only one may satisfy condition
- Others must wait again

**Reason 3: Condition Change**
- Condition may become false again after waking up
- Another thread may use the resource first

**Correct pattern:**
```c
pthread_mutex_lock(&mutex);
while (!condition) {
    pthread_cond_wait(&cond, &mutex);
}
// Condition is guaranteed to be true
pthread_mutex_unlock(&mutex);
```

</details>

---

## Hands-On Exercises

### Exercise 1: Producer-Consumer Variants

Run `examples/OS_Theory/08_producer_consumer.py` and experiment with the bounded buffer.

**Tasks:**
1. Change the buffer capacity to 1 (single-element buffer). How does this affect throughput compared to capacity=5?
2. Create an unbalanced scenario: 5 fast producers (0.01s delay) and 1 slow consumer (0.1s delay). Observe the buffer utilization pattern
3. Implement a priority producer-consumer where high-priority items are consumed first (hint: use a heap instead of a deque)

### Exercise 2: Dining Philosophers Solutions

Explore different solutions to the dining philosophers problem using `examples/OS_Theory/08_producer_consumer.py`.

**Tasks:**
1. Implement the "waiter" solution: add a semaphore initialized to N-1 that limits how many philosophers can attempt to eat simultaneously
2. Implement the "chandy-misra" solution: forks start dirty, after use forks are marked dirty, dirty forks must be given to requesting neighbors
3. Compare all three solutions (resource hierarchy, waiter, chandy-misra) by running 1000 meals and measuring average wait time per philosopher

### Exercise 3: Read-Write Lock

Implement a readers-writers lock that allows concurrent reads but exclusive writes:

**Tasks:**
1. Create a `ReadWriteLock` class with `acquire_read()`, `release_read()`, `acquire_write()`, `release_write()` methods
2. Implement the "readers-preference" policy (readers never wait if no writer is active)
3. Test with 5 reader threads (100 reads each) and 2 writer threads (20 writes each). Verify no read occurs during a write

---

## Exercises

### Exercise 1: Mutex vs Semaphore Semantics

For each synchronization scenario, choose **mutex** or **semaphore** and justify your choice. If semaphore, specify whether it is binary or counting and its initial value.

| Scenario | Tool (mutex/semaphore) | Initial Value | Justification |
|----------|----------------------|---------------|---------------|
| Protecting a shared `int counter` updated by 4 threads | | | |
| Limiting simultaneous database connections to 10 | | | |
| Signaling a worker thread that a new task has been added to a queue | | | |
| Ensuring a config file is read by exactly one thread at a time | | | |
| Coordinating a producer that runs ahead of a consumer (bounded buffer, size=5) | | | |

### Exercise 2: Semaphore Trace

A system uses two semaphores to coordinate a producer and consumer with a buffer of size 3:
- `empty` (initialized to 3): counts empty slots
- `full` (initialized to 0): counts full slots
- `mutex` (initialized to 1): protects buffer access

Trace the following sequence of operations and record the semaphore values after each call. Write "BLOCK" if a process would block.

| Step | Operation | empty | full | mutex | Notes |
|------|-----------|-------|------|-------|-------|
| Initial | — | 3 | 0 | 1 | |
| 1 | Producer: P(empty) | | | | |
| 2 | Producer: P(mutex) | | | | |
| 3 | Producer: add item | | | | |
| 4 | Producer: V(mutex) | | | | |
| 5 | Producer: V(full) | | | | |
| 6 | Producer: P(empty) | | | | |
| 7 | Producer: P(mutex) | | | | |
| 8 | Producer: add item | | | | |
| 9 | Producer: V(mutex) | | | | |
| 10 | Producer: V(full) | | | | |
| 11 | Producer: P(empty) | | | | |
| 12 | Producer: P(mutex) | | | | |
| 13 | Producer: add item | | | | |
| 14 | Producer: V(mutex) | | | | |
| 15 | Producer: V(full) | | | | |
| 16 | Producer: P(empty) | | | | |

### Exercise 3: Monitor and Condition Variable Design

Design a `BoundedQueue` monitor in pseudocode that supports `enqueue(item)` and `dequeue()` with a maximum capacity of N. The monitor must:
- Block `enqueue()` when the queue is full
- Block `dequeue()` when the queue is empty
- Use two condition variables: `not_full` and `not_empty`

```pseudocode
monitor BoundedQueue:
    queue = []
    capacity = N
    condition not_full
    condition not_empty

    procedure enqueue(item):
        // TODO: fill in

    procedure dequeue():
        // TODO: fill in
        return item
```

1. Fill in the `enqueue` and `dequeue` procedures using `wait()` and `signal()`
2. Why must the condition check be in a `while` loop rather than an `if` statement? (Hint: spurious wakeup)
3. If `signal()` uses **signal-and-wait** semantics, what changes in the implementation?

### Exercise 4: Dining Philosophers Deadlock Analysis

Five philosophers sit at a round table. Each picks up the left fork then the right fork. Consider this flawed implementation:

```c
void philosopher(int i) {
    while (true) {
        think();
        lock(fork[i]);           // pick up left fork
        lock(fork[(i+1) % 5]);  // pick up right fork
        eat();
        unlock(fork[i]);
        unlock(fork[(i+1) % 5]);
    }
}
```

1. Demonstrate a deadlock scenario: show the specific state where all 5 philosophers hold exactly one fork and none can proceed
2. Apply the **resource hierarchy** solution: philosophers must always acquire the lower-numbered fork first. Rewrite `philosopher()` to use this fix and explain why it breaks the circular wait condition.
3. With the resource hierarchy fix, can starvation still occur? Give a specific scenario or prove it cannot.

### Exercise 5: Readers-Writers Problem Fairness

The classic readers-writers solution with readers' preference can starve writers. Consider a system where readers arrive at rate λ_r = 10/second and writers at λ_w = 1/second, and reads take 50ms while writes take 20ms.

1. Under readers' preference, a writer arrives and finds 5 active readers. New readers keep arriving. Does the writer ever get access? Explain the starvation mechanism.
2. Implement writers' preference (a reader waits if any writer is waiting or writing) using pseudocode with semaphores. What variables do you need?
3. Under writers' preference, a writer holds the lock. 20 readers are waiting. The writer finishes. How many readers can proceed? All at once or one at a time?
4. Describe a fair solution that prevents both reader and writer starvation. What is the key mechanism?

---

## Next Steps

- [09_Deadlock.md](./09_Deadlock.md) - Deadlock conditions, prevention, avoidance, detection

---

## References

- [OSTEP - Condition Variables](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-cv.pdf)
- [POSIX Threads Programming](https://computing.llnl.gov/tutorials/pthreads/)
- [Java Concurrency in Practice](https://jcip.net/)
