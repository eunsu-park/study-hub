# Synchronization Basics

**Previous**: [Advanced Scheduling](./06_Advanced_Scheduling.md) | **Next**: [Synchronization Tools](./08_Synchronization_Tools.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define a race condition and explain why it occurs in concurrent programs
2. Identify the three requirements for a critical section solution (mutual exclusion, progress, bounded waiting)
3. Trace Peterson's algorithm step-by-step and explain why both `flag[]` and `turn` are necessary
4. Explain the test-and-set and compare-and-swap hardware instructions
5. Distinguish busy waiting from blocking synchronization

---

Concurrency bugs are among the hardest to find and reproduce because they depend on timing. A race condition might crash your program once in a million runs -- but that one time could corrupt a database or cause a security breach. Mastering synchronization is what separates programs that work "most of the time" from programs that are provably correct.

## Table of Contents

1. [Race Condition](#1-race-condition)
2. [Critical Section Problem](#2-critical-section-problem)
3. [Critical Section Solution Requirements](#3-critical-section-solution-requirements)
4. [Peterson's Solution](#4-petersons-solution)
5. [Hardware Support](#5-hardware-support)
6. [Practice Problems](#6-practice-problems)

---

## 1. Race Condition

> Imagine two people trying to walk through a revolving door at the same time from opposite sides. Without coordination, they jam the door and neither gets through. This is the essence of a race condition -- multiple actors accessing a shared resource without proper synchronization, leading to unpredictable results.

### Definition

```
Race Condition
= A situation where the result varies depending on the execution order
  when multiple processes/threads access shared data simultaneously

┌─────────────────────────────────────────────────────────┐
│                    Race Condition Example                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Shared variable: counter = 5                           │
│                                                         │
│  Thread 1: counter++                                    │
│  Thread 2: counter++                                    │
│                                                         │
│  Expected result: counter = 7                           │
│  Actual result: counter = 6 (or 7, non-deterministic)   │
│                                                         │
│  Why?                                                   │
│  counter++ is not atomic                                │
│  1. register = counter (read)                           │
│  2. register = register + 1 (increment)                 │
│  3. counter = register (write)                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Race Condition Execution Process

```
┌─────────────────────────────────────────────────────────┐
│              counter++ Race Condition Details            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Initial value: counter = 5                             │
│                                                         │
│  Thread 1                 Thread 2                      │
│  ─────────                ─────────                     │
│  R1 = counter (R1=5)                                    │
│                          R2 = counter (R2=5)            │
│  R1 = R1 + 1 (R1=6)                                     │
│                          R2 = R2 + 1 (R2=6)             │
│  counter = R1 (counter=6)                               │
│                          counter = R2 (counter=6)       │
│                                                         │
│  Final counter = 6 (not 7!)                             │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Timeline:                                        │   │
│  │                                                  │   │
│  │  T1: ─read─────increment─────write──             │   │
│  │  T2: ────read─────increment──────write──         │   │
│  │            ↑                                     │   │
│  │     Read before T1 writes                        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### C Code Example

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

    // Expected: 2000000
    // Actual: A value less than that (varies each run)
    printf("Counter: %d\n", counter);

    return 0;
}

/*
Example execution results:
$ ./race_condition
Counter: 1523847
$ ./race_condition
Counter: 1678234
$ ./race_condition
Counter: 1432156
*/
```

### Bank Account Example

```
┌─────────────────────────────────────────────────────────┐
│                 Bank Account Race Condition              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Account balance: balance = 1000                        │
│                                                         │
│  Thread 1 (withdraw 200)      Thread 2 (deposit 500)    │
│  ───────────────────          ───────────────────       │
│  temp = balance (1000)                                  │
│                               temp = balance (1000)     │
│  temp = temp - 200 (800)                                │
│                               temp = temp + 500 (1500)  │
│  balance = temp (800)                                   │
│                               balance = temp (1500)     │
│                                                         │
│  Final: balance = 1500                                  │
│  Correct result: 1000 - 200 + 500 = 1300                │
│                                                         │
│  200 units disappeared!                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
// Bank account race condition code
#include <stdio.h>
#include <pthread.h>

int balance = 1000;

void* withdraw(void* arg) {
    int amount = *(int*)arg;
    int temp = balance;
    // Context switch may occur here
    temp = temp - amount;
    balance = temp;
    return NULL;
}

void* deposit(void* arg) {
    int amount = *(int*)arg;
    int temp = balance;
    // Context switch may occur here
    temp = temp + amount;
    balance = temp;
    return NULL;
}

int main() {
    pthread_t t1, t2;
    int withdraw_amount = 200;
    int deposit_amount = 500;

    pthread_create(&t1, NULL, withdraw, &withdraw_amount);
    pthread_create(&t2, NULL, deposit, &deposit_amount);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Final balance: %d\n", balance);
    // Expected: 1300, Actual: 800 or 1500 or 1300

    return 0;
}
```

---

## 2. Critical Section Problem

### Critical Section Definition

```
┌─────────────────────────────────────────────────────────┐
│                    Critical Section Concept              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Critical Section                                       │
│  = Code region that accesses shared resources           │
│  = Region where only one process should execute at a time│
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                Process Structure                 │    │
│  │                                                 │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         Entry Section                   │   │    │
│  │  │         - Request permission to enter    │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                      │                         │    │
│  │                      ▼                         │    │
│  │  ╔═════════════════════════════════════════╗   │    │
│  │  ║         Critical Section                 ║   │    │
│  │  ║         - Code accessing shared resource ║   │    │
│  │  ╚═════════════════════════════════════════╝   │    │
│  │                      │                         │    │
│  │                      ▼                         │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         Exit Section                    │   │    │
│  │  │         - Signal critical section exit   │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                      │                         │    │
│  │                      ▼                         │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         Remainder Section               │   │    │
│  │  │         - Code not using shared resource │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Code Structure

```c
// General critical section structure
while (true) {
    // Entry Section
    // Acquire permission to enter critical section

    // ===== Critical Section =====
    // Code accessing shared resources
    counter++;
    // ==========================================

    // Exit Section
    // Signal critical section completion

    // Remainder Section
    // Code not using shared resources
}
```

---

## 3. Critical Section Solution Requirements

### Three Essential Requirements

```
┌─────────────────────────────────────────────────────────┐
│              Three Requirements for Critical Section     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Mutual Exclusion                                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │  When one process is in the critical section,     │  │
│  │  no other process can enter                       │  │
│  │                                                   │  │
│  │  ┌──────────────────────────────────────────────┐ │  │
│  │  │                                              │ │  │
│  │  │  P1:  ╔════ Critical Section ════╗           │ │  │
│  │  │  P2:  ═══waiting═══▶│            │           │ │  │
│  │  │                     │            │           │ │  │
│  │  │                     ╚════════════╝           │ │  │
│  │  │                                              │ │  │
│  │  └──────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. Progress                                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  If the critical section is empty and processes   │  │
│  │  want to enter, a decision must be made on which  │  │
│  │  process can enter, and this decision cannot be   │  │
│  │  postponed indefinitely                           │  │
│  │                                                   │  │
│  │  → No indefinite denial of entry                  │  │
│  │  → Processes in remainder section cannot participate│ │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. Bounded Waiting                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  After a process requests entry to critical section│ │
│  │  there is a limit on the number of times other    │  │
│  │  processes can enter                              │  │
│  │                                                   │  │
│  │  → Prevent starvation                             │  │
│  │  → Guarantee eventual entry                       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Violation Examples

```
┌─────────────────────────────────────────────────────────┐
│                    Violation Examples                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Mutual Exclusion Violation:                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  P1 and P2 are both in critical section           │  │
│  │  → Race condition occurs                          │  │
│  │                                                   │  │
│  │  P1:  ╔══════════════════╗                        │  │
│  │  P2:       ╔══════════════════╗  ← Simultaneous!  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. Progress Violation:                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Critical section is empty but no one can enter   │  │
│  │                                                   │  │
│  │  P1: waiting ────────────────▶                    │  │
│  │  P2: waiting ────────────────▶                    │  │
│  │  Critical section: [empty]  ← Both cannot enter!  │  │
│  │                                                   │  │
│  │  Example: Incorrectly used turn variable          │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. Bounded Waiting Violation:                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  P1 keeps entering critical section while P2 waits│ │
│  │  forever                                          │  │
│  │                                                   │  │
│  │  P1: CS → CS → CS → ...                          │  │
│  │  P2: waiting ────────────────────────▶ (starvation)│ │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Peterson's Solution

### Algorithm Description

```
┌─────────────────────────────────────────────────────────┐
│                 Peterson's Solution                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Software solution for mutual exclusion between two     │
│  processes                                              │
│                                                         │
│  Shared variables:                                      │
│  • flag[2]: Each process's intention to enter          │
│  • turn: Whose turn it is                              │
│                                                         │
│  Core idea:                                             │
│  • Indicate desire to enter (flag[i] = true)           │
│  • Yield to the other (turn = j)                       │
│  • Wait if other wants to enter and it's their turn    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation Code

```c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

// Shared variables
volatile bool flag[2] = {false, false};
volatile int turn = 0;

int shared_counter = 0;

void* process_0(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // Entry Section
        flag[0] = true;      // Indicate intention to enter
        turn = 1;            // Yield to the other
        while (flag[1] && turn == 1) {
            // Busy Waiting
            // Wait if other is in critical section and it's their turn
        }

        // Critical Section
        shared_counter++;

        // Exit Section
        flag[0] = false;     // Exit critical section

        // Remainder Section
    }
    return NULL;
}

void* process_1(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // Entry Section
        flag[1] = true;      // Indicate intention to enter
        turn = 0;            // Yield to the other
        while (flag[0] && turn == 0) {
            // Busy waiting
        }

        // Critical Section
        shared_counter++;

        // Exit Section
        flag[1] = false;

        // Remainder Section
    }
    return NULL;
}

int main() {
    pthread_t t0, t1;

    pthread_create(&t0, NULL, process_0, NULL);
    pthread_create(&t1, NULL, process_1, NULL);

    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    printf("Counter: %d\n", shared_counter);  // 2000000
    return 0;
}
```

### Correctness Proof

```
┌─────────────────────────────────────────────────────────┐
│              Peterson's Solution Correctness             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Mutual Exclusion ✓                                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  For P0 to be in critical section:                │  │
│  │    flag[1] = false OR turn = 0                    │  │
│  │                                                   │  │
│  │  For P1 to be in critical section:                │  │
│  │    flag[0] = false OR turn = 1                    │  │
│  │                                                   │  │
│  │  For both to be in critical section:             │  │
│  │    turn = 0 AND turn = 1 (impossible!)            │  │
│  │                                                   │  │
│  │  → Mutual exclusion guaranteed                    │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. Progress ✓                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  If only P0 wants to enter:                       │  │
│  │    flag[0] = true, turn = 1, flag[1] = false      │  │
│  │    while condition: flag[1](false) && turn==1     │  │
│  │    → Exit while loop, can enter                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. Bounded Waiting ✓                                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  If P0 is waiting and P1 finishes critical section:│ │
│  │    For P1 to re-enter, must set turn = 0          │  │
│  │    → P0 enters (since turn == 0, P0 has priority) │  │
│  │                                                   │  │
│  │  Entry guaranteed after at most 1 wait            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Peterson's Solution Limitations

```
┌─────────────────────────────────────────────────────────┐
│            Peterson's Solution Limitations               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Only applicable to two processes                    │
│     → Extension to n processes becomes complex          │
│                                                         │
│  2. Busy Waiting                                        │
│     → Wastes CPU time                                   │
│     → Also called spinlock                              │
│                                                         │
│  3. Not guaranteed to work on modern CPUs               │
│     → Compiler/CPU may reorder instructions             │
│     → Memory barriers required                          │
│                                                         │
│  4. Performance issues                                  │
│     → Cache coherence overhead on multicore systems     │
│                                                         │
│  In modern systems:                                     │
│  • Hardware support (atomic instructions)               │
│  • OS-provided synchronization tools (mutex, semaphore) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Hardware Support

### Test-and-Set (TAS)

```
┌─────────────────────────────────────────────────────────┐
│                    Test-and-Set                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Hardware instruction executed atomically:              │
│  1. Read current value                                  │
│  2. Set to new value (usually true)                     │
│                                                         │
│  Pseudocode:                                            │
│  ```                                                    │
│  bool test_and_set(bool *target) {                      │
│      bool rv = *target;    // Read current value        │
│      *target = true;       // Set to true               │
│      return rv;            // Return previous value     │
│  }                                                      │
│  // Entire operation is atomic (non-interruptible)      │
│  ```                                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Mutual Exclusion Using TAS

```c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>

// Atomic boolean lock
atomic_bool lock = false;

int shared_counter = 0;

// test_and_set implementation (actually a hardware instruction)
bool test_and_set(atomic_bool *target) {
    // C11 atomic: equivalent to atomic_exchange
    return atomic_exchange(target, true);
}

void* critical_section(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // Entry section: Attempt to acquire lock
        while (test_and_set(&lock)) {
            // Busy waiting (spinning)
            // If lock is already true, keep waiting
        }

        // Critical section
        shared_counter++;

        // Exit section: Release lock
        lock = false;
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, critical_section, NULL);
    pthread_create(&t2, NULL, critical_section, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", shared_counter);  // 2000000
    return 0;
}
```

### Compare-and-Swap (CAS)

```
┌─────────────────────────────────────────────────────────┐
│                  Compare-and-Swap (CAS)                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Hardware instruction executed atomically:              │
│  1. Compare current value with expected value           │
│  2. If equal, replace with new value                    │
│  3. Return previous value                               │
│                                                         │
│  Pseudocode:                                            │
│  ```                                                    │
│  bool compare_and_swap(int *word, int expected, int new_val) { │
│      int temp = *word;                                  │
│      if (temp == expected) {                            │
│          *word = new_val;                               │
│          return true;                                   │
│      }                                                  │
│      return false;                                      │
│  }                                                      │
│  // Entire operation is atomic                          │
│  ```                                                    │
│                                                         │
│  x86: CMPXCHG instruction                               │
│  ARM: LDREX/STREX instruction combination               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Mutual Exclusion Using CAS

```c
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int lock = 0;  // 0: available, 1: in use

int shared_counter = 0;

void* critical_section(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // Entry section: Attempt to acquire lock using CAS
        int expected = 0;
        while (!atomic_compare_exchange_weak(&lock, &expected, 1)) {
            // If CAS fails, expected is updated with current lock value
            expected = 0;  // Reset to 0 and retry
            // Busy waiting
        }

        // Critical section
        shared_counter++;

        // Exit section: Release lock
        atomic_store(&lock, 0);
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, critical_section, NULL);
    pthread_create(&t2, NULL, critical_section, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", shared_counter);  // 2000000
    return 0;
}
```

### Lock-Free Counter Using CAS

```c
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        int old_val, new_val;
        do {
            old_val = atomic_load(&counter);
            new_val = old_val + 1;
        } while (!atomic_compare_exchange_weak(&counter, &old_val, new_val));
        // Repeat until CAS succeeds
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", counter);  // 2000000
    return 0;
}
```

### Hardware Instruction Comparison

```
┌──────────────────┬─────────────────────────┬─────────────────────────┐
│      Feature      │      Test-and-Set      │    Compare-and-Swap    │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ Return value     │ Previous value          │ Success/failure + current│
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ Conditional      │ Not possible (always    │ Possible (change only   │
│ modification     │ sets)                   │ if matches)             │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ Usage            │ Spinlocks               │ Lock-free algorithms    │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ ABA problem      │ Not applicable          │ Can occur               │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ Complexity       │ Simple                  │ Slightly more complex   │
└──────────────────┴─────────────────────────┴─────────────────────────┘

ABA problem:
Value changes from A → B → A, but CAS cannot detect the change
Solution: Add version number or use double-word CAS
```

---

## 6. Practice Problems

### Problem 1: Identifying Race Conditions

Find and explain where a race condition can occur in the following code.

```c
int balance = 1000;

void transfer(int from, int to, int amount) {
    if (balance >= amount) {
        balance = balance - amount;
        // ... transfer processing ...
    }
}
```

<details>
<summary>Show Answer</summary>

**Race condition location:**

A race condition can occur between `if (balance >= amount)` and `balance = balance - amount`

**Scenario:**
- Balance: 1000, two threads each trying to transfer 700

```
Thread 1: if (1000 >= 700) → true
Thread 2: if (1000 >= 700) → true (not yet decreased)
Thread 1: balance = 1000 - 700 = 300
Thread 2: balance = 1000 - 700 = 300 (incorrect operation!)
```

Result: 1400 transferred (should be balance of -400, but is 300)

**Solution:** Critical section protection needed (mutex, etc.)

</details>

### Problem 2: Critical Section Requirements

Which of the following is NOT a requirement for solving the critical section problem?

A. Mutual Exclusion
B. Progress
C. Bounded Waiting
D. Fairness

<details>
<summary>Show Answer</summary>

**Answer: D. Fairness**

Three essential requirements for critical section problem:
1. Mutual Exclusion - only one at a time
2. Progress - no indefinite waiting
3. Bounded Waiting - guaranteed entry within finite attempts

Fairness is desirable but not an essential requirement.

</details>

### Problem 3: Peterson's Solution Analysis

Explain why two processes cannot simultaneously enter the critical section in Peterson's Solution.

<details>
<summary>Show Answer</summary>

**Core logic:**

For P0 to be in the critical section:
- `flag[1] == false` OR `turn == 0`

For P1 to be in the critical section:
- `flag[0] == false` OR `turn == 1`

For both processes to enter simultaneously, both conditions must be true.

However:
- For both to enter, `flag[0] == true` AND `flag[1] == true`
- Therefore the second condition (turn) becomes decisive
- `turn` can only be 0 or 1
- `turn == 0 AND turn == 1` is impossible!

Therefore, at most one process can enter the critical section.

</details>

### Problem 4: TAS Implementation

Implement a lock using Test-and-Set that satisfies the following requirements:
- `lock()`: Acquire lock
- `unlock()`: Release lock
- Safe for use by multiple threads

<details>
<summary>Show Answer</summary>

```c
#include <stdatomic.h>
#include <stdbool.h>

typedef struct {
    atomic_bool locked;
} spinlock_t;

void spinlock_init(spinlock_t *lock) {
    atomic_store(&lock->locked, false);
}

void lock(spinlock_t *lock) {
    while (atomic_exchange(&lock->locked, true)) {
        // Spin (busy waiting)
        // Optional: yield CPU
        // sched_yield();
    }
}

void unlock(spinlock_t *lock) {
    atomic_store(&lock->locked, false);
}

// Usage example:
spinlock_t my_lock;
spinlock_init(&my_lock);

lock(&my_lock);
// Critical section
unlock(&my_lock);
```

</details>

### Problem 5: Busy Waiting Problems

Explain the problems with busy waiting and suggest solutions.

<details>
<summary>Show Answer</summary>

**Problems with busy waiting:**

1. **CPU time waste**
   - Consumes CPU cycles even while waiting
   - Takes away time that other processes could use

2. **Priority inversion**
   - High priority process spins
   - Cannot proceed because low priority process holds the lock

3. **Power consumption**
   - Battery drain on mobile/embedded systems

**Solutions:**

1. **Blocking locks**
   - Transition process to sleep state while waiting
   - Wake up when lock is released
   - Example: mutex, semaphore

2. **Hybrid approach**
   - Spin for a short time, then block
   - Linux futex, Java synchronized

3. **Using yield**
   - Yield to other threads instead of spinning
   - Call `sched_yield()`

</details>

---

## Hands-On Exercises

### Exercise 1: Peterson's Algorithm Verification

Run `examples/OS_Theory/07_sync_primitives.py` and analyze the Peterson's algorithm demo.

**Tasks:**
1. Modify `PetersonLock` to remove the `turn` variable. Run `demo_peterson()` and explain why mutual exclusion fails
2. Increase the iteration count to 500,000 and run 5 times. Record the actual count vs expected. What's the maximum error you observe?
3. Implement a `FilterLock` that generalizes Peterson's algorithm to N threads (use N-1 levels of flag arrays)

### Exercise 2: Atomic Operations Comparison

Compare the performance of different synchronization primitives:

```python
import threading, time

def benchmark_lock(lock_type, label, n=1_000_000):
    counter = [0]
    lock = lock_type()

    def worker():
        for _ in range(n // 2):
            lock.acquire()
            counter[0] += 1
            lock.release()

    t0 = threading.Thread(target=worker)
    t1 = threading.Thread(target=worker)
    start = time.perf_counter()
    t0.start(); t1.start()
    t0.join(); t1.join()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s, count={counter[0]}")
```

**Tasks:**
1. Benchmark `threading.Lock`, `threading.RLock`, and `threading.Semaphore(1)`. Rank by performance
2. Explain why `Lock` is typically faster than `RLock` and `Semaphore`
3. Test with no lock at all and report the error rate — how does Python's GIL affect the result?

### Exercise 3: Barrier Implementation

Study the `SimpleBarrier` in `examples/OS_Theory/07_sync_primitives.py`.

**Tasks:**
1. The current barrier uses an `Event` per generation. Implement an alternative using a `Condition` variable instead
2. Add a `barrier.wait()` return value that identifies one thread as the "leader" (returns True for exactly one thread per barrier phase)
3. Test your barrier with 8 threads and 5 phases, verifying all Phase N completes before any Phase N+1 starts

---

## Exercises

### Exercise 1: Race Condition Identification

Examine the following code that implements a simple bank account. Two threads run `transfer()` concurrently.

```c
#include <stdio.h>
#include <pthread.h>

int account_A = 1000;
int account_B = 500;

void transfer(int *from, int *to, int amount) {
    if (*from >= amount) {       // Step 1: Check balance
        *from -= amount;         // Step 2: Deduct
        *to   += amount;         // Step 3: Credit
    }
}

void *thread1(void *arg) { transfer(&account_A, &account_B, 200); return NULL; }
void *thread2(void *arg) { transfer(&account_A, &account_B, 900); return NULL; }
```

1. Identify the critical section(s) in this code
2. Describe a specific interleaving of thread1 and thread2 that results in account_A having a negative balance, given that both transfers initially pass the check
3. What is the total money in the system at the start? Can a race condition change this total? Explain
4. Fix the code using a global `pthread_mutex_t` to ensure correct behavior

### Exercise 2: Critical Section Requirements Analysis

For each proposed "solution" to the two-process critical section problem, identify which of the three requirements it violates (mutual exclusion, progress, bounded waiting). Assume two processes P0 and P1.

**Proposed Solution A** — Simple turn variable:
```c
int turn = 0;
// Process Pi:
while (turn != i);  // wait
// Critical section
turn = 1 - i;       // give turn to other
```

**Proposed Solution B** — Flag only, no turn:
```c
bool flag[2] = {false, false};
// Process Pi:
flag[i] = true;
while (flag[1-i]);  // wait
// Critical section
flag[i] = false;
```

For each solution:
1. Does it satisfy mutual exclusion? Prove or give a counterexample
2. Does it satisfy progress? Prove or give a counterexample
3. Does it satisfy bounded waiting? Explain

### Exercise 3: Peterson's Algorithm Step-Through

Peterson's algorithm for processes P0 and P1:
```c
bool flag[2] = {false, false};
int turn;

// Process Pi (i=0 or 1):
flag[i] = true;
turn = 1 - i;
while (flag[1-i] && turn == 1-i);  // busy wait
// --- Critical Section ---
flag[i] = false;
```

Trace through the following interleaving where P0 and P1 both try to enter at the same time. Record the values of `flag[0]`, `flag[1]`, and `turn` after each step. Which process enters the critical section first?

| Step | Action | flag[0] | flag[1] | turn | Result |
|------|--------|---------|---------|------|--------|
| 1 | P0: flag[0]=true | | | | |
| 2 | P1: flag[1]=true | | | | |
| 3 | P0: turn=1 | | | | |
| 4 | P1: turn=0 | | | | |
| 5 | P0: checks while | | | | |
| 6 | P1: checks while | | | | |

Explain: why does assigning `turn` last guarantee that exactly one process proceeds?

### Exercise 4: TAS and CAS Semantics

**Part A**: Show how Test-and-Set (TAS) can implement a spinlock. Write pseudocode for `lock()` and `unlock()` using `TestAndSet(bool *target)`.

**Part B**: Show how Compare-and-Swap (CAS) can implement a lock-free counter increment. Write pseudocode for `atomic_increment(int *counter)` using `CompareAndSwap(int *value, int expected, int new_val)`.

**Part C**: A spinlock using TAS satisfies mutual exclusion but may violate bounded waiting. Describe a scenario with 3 processes where one process never acquires the lock. Then describe the modification needed (ticket lock) to guarantee bounded waiting.

### Exercise 5: Busy Waiting vs Blocking

A system has 100 threads competing for a critical section that takes on average 2ms to execute. The system has 4 CPU cores.

1. If threads use a **spinlock** (busy waiting), how much CPU time is wasted while 99 threads wait for 1 thread to finish? Express as percentage of 4-core capacity.
2. If threads use a **mutex** (blocking), what happens to waiting threads and why does this not waste CPU?
3. Give one scenario where a spinlock is **preferred** over a blocking mutex, and explain why.
4. What is the "dual-mode" strategy used by Linux's `mutex_spin_on_owner()`? Why does it combine both approaches?

---

## Next Steps

- [08_Synchronization_Tools.md](./08_Synchronization_Tools.md) - Mutex, semaphore, monitor

---

## References

- [OSTEP - Concurrency: Locks](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf)
- [The Art of Multiprocessor Programming (Herlihy & Shavit)](https://www.elsevier.com/books/the-art-of-multiprocessor-programming/)
- [C11 Atomic Operations](https://en.cppreference.com/w/c/atomic)
