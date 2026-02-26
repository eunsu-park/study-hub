# Threads and Multithreading

**Previous**: [Process Concepts](./02_Process_Concepts.md) | **Next**: [CPU Scheduling Basics](./04_CPU_Scheduling_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish threads from processes and explain what resources threads share versus keep private
2. Explain why threads are considered lightweight compared to processes in terms of creation, switching, and memory overhead
3. Compare user-level threads and kernel-level threads, identifying the tradeoffs of each approach
4. Describe the three multithreading models (many-to-one, one-to-one, many-to-many) and identify which model modern operating systems use
5. Implement basic multithreaded programs using the POSIX pthread API
6. Analyze thread safety issues such as race conditions in shared-memory concurrency

---

Modern CPUs have multiple cores, but even a single core can run many threads. Threads let a single process do multiple things at once -- handle user input while downloading a file while rendering the UI. Understanding threads is essential for writing responsive, efficient software. This lesson covers the differences between threads and processes, user-level versus kernel-level threads, multithreading models, and the pthread API that brings these concepts to life in C.

## Table of Contents

1. [What is a Thread?](#1-what-is-a-thread)
2. [Thread vs Process](#2-thread-vs-process)
3. [Thread Control Block (TCB)](#3-thread-control-block-tcb)
4. [User Threads and Kernel Threads](#4-user-threads-and-kernel-threads)
5. [Multithreading Models](#5-multithreading-models)
6. [pthread API Basics](#6-pthread-api-basics)
7. [Practice Problems](#7-practice-problems)

---

## 1. What is a Thread?

### Definition

```
Thread = Execution flow within a process
       = Basic unit of CPU scheduling
       = Lightweight Process

┌─────────────────────────────────────────────────────────┐
│                Single-threaded Process                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Code  │  Data  │       Heap       │    Stack   │   │
│  └─────────────────────────────────────────────────┘   │
│               Only one execution flow exists            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                Multi-threaded Process                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Code  │  Data  │       Heap       │               │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Stack1  │  │  Stack2  │  │  Stack3  │              │
│  │ Thread1  │  │ Thread2  │  │ Thread3  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│           Three execution flows (can run in parallel)    │
└─────────────────────────────────────────────────────────┘
```

### Why Use Threads?

```
┌────────────────────────────────────────────────────────┐
│                   Thread Benefits                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│ 1. Responsiveness                                      │
│    - Other threads continue even if one blocks         │
│    - Separate UI thread from worker threads            │
│                                                        │
│ 2. Resource Sharing                                    │
│    - Share same address space                          │
│    - Exchange data without IPC                         │
│                                                        │
│ 3. Economy                                             │
│    - Thread creation faster than process creation      │
│    - Reduced context switch cost                       │
│                                                        │
│ 4. Scalability                                         │
│    - Utilize multicore CPUs                            │
│    - Each thread runs on different core                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Thread vs Process

### Shared vs Private Resources

```
┌─────────────────────────────────────────────────────────┐
│                   Thread Shared/Private                  │
├────────────────────────┬────────────────────────────────┤
│       Shared           │       Private                  │
├────────────────────────┼────────────────────────────────┤
│ • Code section         │ • Thread ID                    │
│ • Data section         │ • Program Counter (PC)         │
│ • Heap area            │ • Register set                 │
│ • Open files           │ • Stack                        │
│ • Signal handlers      │ • Scheduling info (priority)   │
│ • Current directory    │ • Signal mask                  │
│ • User/Group ID        │ • errno value                  │
└────────────────────────┴────────────────────────────────┘
```

### Memory Layout Comparison

```
Process Memory:                Multithreaded Process Memory:

┌─────────────┐                ┌─────────────────────────┐
│   Kernel    │                │         Kernel          │
├─────────────┤                ├─────────────────────────┤
│   Stack     │                │Thread1 │Thread2 │Thread3│
│     ↓       │                │ Stack  │ Stack  │ Stack │
│             │                │   ↓    │   ↓    │   ↓   │
├ ─ ─ ─ ─ ─ ─ ┤                ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
│             │                │                         │
│     ↑       │                │           ↑             │
│   Heap      │                │      Heap (shared)      │
├─────────────┤                ├─────────────────────────┤
│   Data      │                │     Data (shared)       │
├─────────────┤                ├─────────────────────────┤
│   Code      │                │     Code (shared)       │
└─────────────┘                └─────────────────────────┘
```

### Cost Comparison

```
┌─────────────────────────────────────────────────────────┐
│               Process vs Thread Cost Comparison          │
├───────────────────┬─────────────┬───────────────────────┤
│       Item         │   Process   │        Thread         │
├───────────────────┼─────────────┼───────────────────────┤
│ Creation time     │    Slow     │         Fast          │
│ (Linux baseline)  │  ~10ms      │        ~1ms           │
├───────────────────┼─────────────┼───────────────────────┤
│ Context switch    │    Slow     │         Fast          │
│                   │ TLB flush   │ TLB can be preserved  │
├───────────────────┼─────────────┼───────────────────────┤
│ Memory usage      │    High     │         Low           │
│                   │ Separate    │  Shared space         │
├───────────────────┼─────────────┼───────────────────────┤
│ Communication cost│    High     │         Low           │
│                   │  IPC needed │  Direct memory access │
├───────────────────┼─────────────┼───────────────────────┤
│ Stability         │    High     │         Low           │
│                   │ Isolated    │  One crash affects all│
└───────────────────┴─────────────┴───────────────────────┘
```

---

## 3. Thread Control Block (TCB)

### TCB Structure

```
┌───────────────────────────────────────────────────────┐
│                   TCB (Thread Control Block)          │
├───────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Thread ID (TID)                                 │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Thread State (Running, Ready, Blocked...)       │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Program Counter (PC)                            │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ CPU Registers (general purpose, flags...)       │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Stack Pointer                                   │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Scheduling Information (priority)               │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Pointer to parent PCB                           │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  TCB is much smaller than PCB (why thread switch      │
│  is faster)                                           │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### PCB and TCB Relationship

```
┌─────────────────────────────────────────────────────────┐
│                        PCB                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ PID, memory info, open files, signal handlers...  │  │
│  │                                                   │  │
│  │  Thread list:                                     │  │
│  │  ┌───────────────────────────────────────────┐   │  │
│  │  │     TCB1      │     TCB2      │    TCB3   │   │  │
│  │  │ TID, PC, SP,  │ TID, PC, SP,  │ TID, PC,  │   │  │
│  │  │ registers,    │ registers,    │ SP, ...   │   │  │
│  │  │ state         │ state         │           │   │  │
│  │  └───────────────────────────────────────────┘   │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 4. User Threads and Kernel Threads

### User-Level Threads (ULT)

```
┌─────────────────────────────────────────────────────────┐
│              User-Level Threads (ULT)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User space:                                           │
│   ┌───────────────────────────────────────────────────┐ │
│   │              Application                          │ │
│   │  ┌─────────────────────────────────────────────┐  │ │
│   │  │  Thread1    Thread2    Thread3              │  │ │
│   │  └─────────────────────────────────────────────┘  │ │
│   │              ↑    ↑    ↑                         │ │
│   │              └────┼────┘                         │ │
│   │                   ↓                              │ │
│   │         Thread Library                           │ │
│   │         (scheduling, creation, synchronization)  │ │
│   └───────────────────────────────────────────────────┘ │
│                       │                                 │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                       │                                 │
│   Kernel space:       ↓                                 │
│   ┌───────────────────────────────────────────────────┐ │
│   │       Kernel (sees as single thread)              │ │
│   │                                                   │ │
│   │   Kernel knows process only, not threads         │ │
│   └───────────────────────────────────────────────────┘ │
│                                                         │
│   Advantages: Fast context switch, portability          │
│   Disadvantages: Blocking blocks all, no multicore use  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Kernel-Level Threads (KLT)

```
┌─────────────────────────────────────────────────────────┐
│              Kernel-Level Threads (KLT)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User space:                                           │
│   ┌───────────────────────────────────────────────────┐ │
│   │              Application                          │ │
│   │  ┌─────────────────────────────────────────────┐  │ │
│   │  │  Thread1    Thread2    Thread3              │  │ │
│   │  └─────────────────────────────────────────────┘  │ │
│   └───────────────────────────────────────────────────┘ │
│              │         │         │                      │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│              ↓         ↓         ↓                      │
│   Kernel space:                                         │
│   ┌───────────────────────────────────────────────────┐ │
│   │              Kernel Scheduler                     │ │
│   │  ┌─────────────────────────────────────────────┐  │ │
│   │  │ K-Thread1  K-Thread2  K-Thread3             │  │ │
│   │  └─────────────────────────────────────────────┘  │ │
│   │                                                   │ │
│   │   Kernel schedules each thread individually      │ │
│   └───────────────────────────────────────────────────┘ │
│                                                         │
│   Advantages: Multicore use, blocking doesn't affect    │
│               other threads                             │
│   Disadvantages: Slower context switch (syscall needed) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Comparison Table

```
┌────────────────────┬────────────────────┬────────────────────┐
│       Feature       │    User Thread     │    Kernel Thread   │
├────────────────────┼────────────────────┼────────────────────┤
│ Managed by         │ Thread library     │ OS kernel          │
├────────────────────┼────────────────────┼────────────────────┤
│ Creation/switch    │ Fast (no syscall)  │ Slow (syscall)     │
├────────────────────┼────────────────────┼────────────────────┤
│ Multicore use      │ Not possible       │ Possible           │
├────────────────────┼────────────────────┼────────────────────┤
│ Blocking syscall   │ Blocks all process │ Only blocks thread │
├────────────────────┼────────────────────┼────────────────────┤
│ OS support needed  │ Not required       │ Required           │
├────────────────────┼────────────────────┼────────────────────┤
│ Examples           │ GNU Portable       │ Linux NPTL,        │
│                    │ Threads            │ Windows threads    │
└────────────────────┴────────────────────┴────────────────────┘
```

---

## 5. Multithreading Models

### Many-to-One Model (N:1)

```
┌─────────────────────────────────────────────────────────┐
│                  Many-to-One Model (N:1)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User space:                                           │
│   ┌─────────────────────────────────────────────────┐   │
│   │  ULT1    ULT2    ULT3    ULT4    ULT5          │   │
│   │   │       │       │       │       │            │   │
│   │   └───────┼───────┼───────┼───────┘            │   │
│   │           └───────┼───────┘                    │   │
│   │                   ↓                            │   │
│   │           Thread Library                       │   │
│   └─────────────────────────────────────────────────┘   │
│                       │                                 │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                       ↓                                 │
│   Kernel space:    ┌──────────┐                         │
│                    │  KLT 1   │                         │
│                    └──────────┘                         │
│                                                         │
│   • Advantages: Efficient thread management             │
│   • Disadvantages: No multicore, blocking blocks all    │
│   • Examples: Early Green Threads                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### One-to-One Model (1:1)

```
┌─────────────────────────────────────────────────────────┐
│                  One-to-One Model (1:1)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User space:                                           │
│   ┌─────────────────────────────────────────────────┐   │
│   │  ULT1    ULT2    ULT3    ULT4                  │   │
│   │   │       │       │       │                    │   │
│   └───┼───────┼───────┼───────┼────────────────────┘   │
│       │       │       │       │                        │
│   ━━━━│━━━━━━━│━━━━━━━│━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━ │
│       ↓       ↓       ↓       ↓                        │
│   Kernel space:                                         │
│   ┌───────────────────────────────────────────────┐     │
│   │  KLT1    KLT2    KLT3    KLT4                │     │
│   └───────────────────────────────────────────────┘     │
│                                                         │
│   • Advantages: Multicore use, blocking doesn't affect  │
│                 other threads                           │
│   • Disadvantages: Kernel thread creation overhead      │
│   • Examples: Linux NPTL, Windows, macOS                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Many-to-Many Model (M:N)

```
┌─────────────────────────────────────────────────────────┐
│                  Many-to-Many Model (M:N)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User space:                                           │
│   ┌─────────────────────────────────────────────────┐   │
│   │  ULT1  ULT2  ULT3  ULT4  ULT5  ULT6  ULT7     │   │
│   │   │     │     │     │     │     │     │       │   │
│   │   └─────┼─────┼─────┼─────┼─────┼─────┘       │   │
│   │         └─────┼─────┼─────┼─────┘             │   │
│   │               ↓     ↓     ↓                   │   │
│   │           Thread Scheduler                     │   │
│   └─────────────────────────────────────────────────┘   │
│                   │     │     │                        │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                   ↓     ↓     ↓                        │
│   Kernel space:                                         │
│   ┌───────────────────────────────────────────────┐     │
│   │      KLT1      KLT2      KLT3                │     │
│   └───────────────────────────────────────────────┘     │
│                                                         │
│   7 user threads mapped to 3 kernel threads             │
│                                                         │
│   • Advantages: Flexibility, balance of efficiency      │
│                 and parallelism                         │
│   • Disadvantages: Implementation complexity            │
│   • Examples: Solaris, Go goroutines                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Model Comparison

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│     Feature   │   N:1       │    1:1      │    M:N      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Parallel exec │ Not possible │ Possible     │ Possible     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Blocking issue│ Blocks all   │ No impact    │ Partial      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Thread create │ Fast         │ Slow         │ Medium       │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Complexity    │ Low          │ Low          │ High         │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Scalability   │ Low          │ High         │ High         │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Modern OS     │ Rarely used  │ Commonly used│ Some use     │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 6. pthread API Basics

### Thread Creation and Termination

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// Function executed by thread
void* thread_function(void* arg) {
    int thread_num = *(int*)arg;
    printf("Thread %d starting\n", thread_num);

    // Perform work
    sleep(1);

    printf("Thread %d terminating\n", thread_num);
    return NULL;
}

int main() {
    pthread_t threads[3];
    int thread_args[3] = {1, 2, 3};

    // Create threads
    for (int i = 0; i < 3; i++) {
        int result = pthread_create(&threads[i], NULL,
                                    thread_function, &thread_args[i]);
        if (result != 0) {
            perror("pthread_create failed");
            exit(1);
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
        printf("Thread %d joined\n", thread_args[i]);
    }

    printf("Main thread terminating\n");
    return 0;
}

/*
Compile: gcc -pthread thread_example.c -o thread_example

Output (order may vary):
Thread 1 starting
Thread 2 starting
Thread 3 starting
Thread 1 terminating
Thread 2 terminating
Thread 3 terminating
Thread 1 joined
Thread 2 joined
Thread 3 joined
Main thread terminating
*/
```

### pthread API Main Functions

```
┌────────────────────────┬────────────────────────────────────┐
│        Function         │              Description           │
├────────────────────────┼────────────────────────────────────┤
│ pthread_create()       │ Create new thread                  │
│ pthread_join()         │ Wait for thread termination        │
│ pthread_exit()         │ Exit current thread                │
│ pthread_self()         │ Return current thread ID           │
│ pthread_equal()        │ Compare two thread IDs             │
│ pthread_detach()       │ Detach thread (auto cleanup)       │
│ pthread_cancel()       │ Request thread cancellation        │
├────────────────────────┼────────────────────────────────────┤
│ pthread_mutex_init()   │ Initialize mutex                   │
│ pthread_mutex_lock()   │ Lock mutex                         │
│ pthread_mutex_unlock() │ Unlock mutex                       │
│ pthread_mutex_destroy()│ Destroy mutex                      │
├────────────────────────┼────────────────────────────────────┤
│ pthread_cond_init()    │ Initialize condition variable      │
│ pthread_cond_wait()    │ Wait on condition variable         │
│ pthread_cond_signal()  │ Signal condition variable          │
│ pthread_cond_broadcast()│ Signal all waiting threads        │
└────────────────────────┴────────────────────────────────────┘
```

### Receiving Return Values

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Calculate and return square of number
void* calculate_square(void* arg) {
    int num = *(int*)arg;
    int* result = malloc(sizeof(int));
    *result = num * num;

    printf("Thread: %d squared = %d\n", num, *result);

    return (void*)result;  // Return heap-allocated result
}

int main() {
    pthread_t thread;
    int num = 5;
    void* result;

    // Create thread
    pthread_create(&thread, NULL, calculate_square, &num);

    // Wait for thread and receive result
    pthread_join(thread, &result);

    printf("Main: result = %d\n", *(int*)result);

    free(result);  // Free memory

    return 0;
}

/*
Output:
Thread: 5 squared = 25
Main: result = 25
*/
```

### Thread Detachment

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void* detached_thread(void* arg) {
    printf("Detached thread starting\n");
    sleep(2);
    printf("Detached thread terminating\n");
    return NULL;
}

int main() {
    pthread_t thread;
    pthread_attr_t attr;

    // Initialize thread attributes
    pthread_attr_init(&attr);

    // Set detached state
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    // Create detached thread
    pthread_create(&thread, &attr, detached_thread, NULL);

    // Destroy attribute object
    pthread_attr_destroy(&attr);

    printf("Main thread: no need to join detached thread\n");

    // Give detached thread time to execute
    sleep(3);

    printf("Main thread terminating\n");
    return 0;
}

/*
Detached thread:
- Automatically cleaned up on termination
- pthread_join() unnecessary (causes error)
- Suitable for background tasks
*/
```

### Thread Safety Issue Example

```c
#include <stdio.h>
#include <pthread.h>

int counter = 0;  // Shared variable

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        counter++;  // Race condition!
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // Expected: 2000000, Actual: less than that (varies)
    printf("Counter: %d\n", counter);

    return 0;
}

/*
counter++ operation is not atomic:
1. Read counter value from memory
2. Increment value
3. Store result to memory

When two threads execute simultaneously, values are lost
(solved in next lesson)
*/
```

---

## 7. Practice Problems

### Problem 1: Thread vs Process

Choose all that threads do NOT share.

A. Code section
B. Data section
C. Stack
D. Heap
E. Program Counter
F. Open files

<details>
<summary>Show Answer</summary>

**C, E**

- Stack: Each thread has its own stack (local variables, function call info)
- Program Counter: Each thread can execute different code locations

All others are shared between threads.

</details>

### Problem 2: Multithreading Models

Match each description to the correct multithreading model.

1. If one user thread blocks, the entire process blocks.
2. The model used by Linux NPTL.
3. Can efficiently manage more user threads than kernel threads.

Options: N:1, 1:1, M:N

<details>
<summary>Show Answer</summary>

1. **N:1 (Many-to-One)** - All user threads map to one kernel thread, so blocking one blocks all
2. **1:1 (One-to-One)** - Linux NPTL, Windows, macOS use this model
3. **M:N (Many-to-Many)** - Maps many user threads to fewer kernel threads efficiently

</details>

### Problem 3: pthread Code Analysis

What are the possible outputs of the following code?

```c
#include <stdio.h>
#include <pthread.h>

void* print_msg(void* arg) {
    char* msg = (char*)arg;
    printf("%s", msg);
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, print_msg, "A");
    pthread_create(&t2, NULL, print_msg, "B");

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("C");
    return 0;
}
```

A. ABC
B. BAC
C. CAB
D. ACB

<details>
<summary>Show Answer</summary>

**A, B**

- Order of A and B can vary depending on thread scheduling
- C is always last (printed after both threads join)
- Possible outputs: ABC or BAC

CAB, ACB are impossible (C only prints after join)

</details>

### Problem 4: Thread Creation

Explain the problem with the following code.

```c
#include <stdio.h>
#include <pthread.h>

void* thread_func(void* arg) {
    int* num = (int*)arg;
    printf("Thread: %d\n", *num);
    return NULL;
}

int main() {
    pthread_t threads[5];

    for (int i = 0; i < 5; i++) {
        pthread_create(&threads[i], NULL, thread_func, &i);
    }

    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
```

<details>
<summary>Show Answer</summary>

**Problem: Race Condition**

All threads receive the address of the same variable `i`. When threads execute, `i`'s value may have already changed.

Expected output: 0, 1, 2, 3, 4
Actual possible output: 5, 5, 5, 5, 5 or 2, 3, 4, 5, 5, etc.

**Solution:**

```c
int thread_args[5];

for (int i = 0; i < 5; i++) {
    thread_args[i] = i;
    pthread_create(&threads[i], NULL, thread_func, &thread_args[i]);
}
```

Must allocate separate argument space for each thread.

</details>

### Problem 5: TCB and PCB

Explain why TCB is smaller than PCB in relation to thread characteristics.

<details>
<summary>Show Answer</summary>

Why TCB is smaller than PCB:

1. **Resource Sharing**: Threads share code, data, heap, open files with other threads in the same process. This information is stored only in PCB, not in TCB.

2. **Information stored in TCB is minimal**:
   - Thread ID
   - Program Counter
   - Register set
   - Stack Pointer
   - Thread state
   - Priority

3. **Additional information in PCB**:
   - Memory management info (page table)
   - Open file list
   - I/O status
   - Accounting information
   - Signal handlers

Result: Thread context switch is faster than process context switch.

</details>

---

## Hands-On Exercises

### Exercise 1: Race Condition and GIL

Run `examples/OS_Theory/03_threading_demo.py` and observe the race condition and GIL demos.

**Tasks:**
1. In `demo_race_condition()`, change the iteration count to 1,000,000. Does the unsafe counter ever produce the correct result? Run 10 times and record the error rate
2. Explain why Python's GIL prevents the race condition for simple integer increments but NOT for check-then-act patterns (like `if balance > 0: balance -= 1`)
3. Modify the threading demo to use `threading.RLock` (reentrant lock) and demonstrate a scenario where a regular `Lock` would deadlock but `RLock` succeeds

### Exercise 2: Thread vs Process Performance

Write a benchmark comparing thread creation vs process creation overhead:

```python
import threading, multiprocessing, time

def worker():
    pass  # minimal work

def benchmark(label, create_func, n=1000):
    start = time.perf_counter()
    items = [create_func(target=worker) for _ in range(n)]
    for item in items:
        item.start()
    for item in items:
        item.join()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed*1000:.1f} ms for {n} workers")

benchmark("Threads", threading.Thread)
benchmark("Processes", multiprocessing.Process, n=100)
```

**Tasks:**
1. Run the benchmark and compare thread vs process creation time
2. Why is process creation slower? What additional setup does fork() require?
3. Add a CPU-bound task (sum 1 to 1M) and compare total execution time — when do processes win?

### Exercise 3: Thread Pool Pattern

Implement a simple thread pool that reuses threads instead of creating new ones:

**Tasks:**
1. Create a `ThreadPool(n_workers)` class with a `submit(func, *args)` method
2. Use a `queue.Queue` for task distribution
3. Test with 100 tasks on a pool of 4 workers and verify all tasks complete
4. Compare the total time vs creating 100 individual threads

---

## Exercises

### Exercise 1: Thread vs Process Resource Sharing

A web server creates a new **thread** for each incoming request. For each resource listed below, state whether it is **shared** among all threads or **private** to each thread, and explain why.

| Resource | Shared or Private? | Why? |
|----------|--------------------|------|
| Global variable `int total_requests` | | |
| Local variable `char buf[4096]` inside the handler | | |
| Open file descriptor for the access log | | |
| errno value | | |
| Heap-allocated `malloc()`'d request object | | |
| Signal disposition (SIGTERM → graceful shutdown) | | |

### Exercise 2: Multithreading Model Comparison

A student is choosing a threading model for a new application. Evaluate each scenario and recommend the most appropriate model (N:1, 1:1, or M:N), explaining your reasoning.

1. A scientific simulation that parallelizes matrix multiplication across 8 CPU cores on a 12-core machine
2. A legacy embedded system with no kernel thread support that needs cooperative multitasking
3. A high-concurrency server handling 10,000 simultaneous connections on a 4-core system
4. A simple desktop GUI app that offloads one background task to keep the UI responsive

### Exercise 3: Race Condition Analysis

Examine the following code run by two threads simultaneously. Identify all race conditions and for each one, describe: (a) the shared resource involved, (b) a concrete interleaving that causes incorrect behavior, and (c) a fix.

```c
#include <stdio.h>
#include <pthread.h>

int counter = 0;
int log_count = 0;
FILE *log_file;

void *worker(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 1000; i++) {
        counter++;                            // (A)
        fprintf(log_file, "thread %d: %d\n", id, counter);  // (B)
        log_count++;                          // (C)
    }
    return NULL;
}
```

1. How many race conditions exist? List each one.
2. What is the expected value of `counter` after both threads finish if no synchronization is used? What range of values is actually possible?
3. Fix the race conditions using `pthread_mutex_t`.

### Exercise 4: TCB Fields Under Context Switch

A thread is preempted mid-execution. List the fields that must be saved into the Thread Control Block (TCB) at the moment of preemption, and explain what would go wrong if each field were NOT saved.

| TCB Field | Consequence if NOT saved |
|-----------|--------------------------|
| Program Counter | |
| Stack Pointer | |
| General-purpose registers | |
| Floating-point registers | |
| Thread state | |

### Exercise 5: Designing for Concurrency

A single-threaded image-processing server sequentially: (1) receives an image over the network, (2) resizes it, (3) applies a filter, (4) saves to disk. The measured times are: network I/O 200ms, resize 50ms, filter 150ms, disk I/O 100ms. The server handles one request at a time.

1. What is the total latency for one request and the maximum throughput (requests/second)?
2. Redesign the server using multithreading to maximize throughput. Describe your design (number of threads, what each does, how they communicate).
3. Which steps can benefit from true parallelism on a multi-core CPU? Which are limited by I/O even with threads?
4. What is the theoretical maximum throughput of your multithreaded design if CPU steps can overlap with I/O steps?

---

## Next Steps

- [04_CPU_Scheduling_Basics.md](./04_CPU_Scheduling_Basics.md) - Basic concepts of CPU scheduling

---

## References

- [POSIX Threads Programming](https://computing.llnl.gov/tutorials/pthreads/)
- [Linux man pages - pthreads](https://man7.org/linux/man-pages/man7/pthreads.7.html)
- [OSTEP - Concurrency: Threads](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-intro.pdf)
