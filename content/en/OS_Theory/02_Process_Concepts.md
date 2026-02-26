# Process Concepts

**Previous**: [OS Overview](./01_OS_Overview.md) | **Next**: [Threads and Multithreading](./03_Threads_and_Multithreading.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define a process and distinguish it from a program
2. Describe the process memory layout, including the text, data, BSS, heap, and stack sections
3. Explain the contents of a Process Control Block (PCB) and its role in process management
4. Trace process state transitions through the 5-state and 7-state models
5. Distinguish between process creation methods using fork() and program replacement using exec()
6. Analyze the direct and indirect costs of context switching
7. Explain zombie and orphan processes and how the OS handles them

---

A program sitting on disk is like a recipe in a cookbook -- inert text. A process is that recipe being actively cooked: ingredients allocated, oven preheated, timer running. Understanding processes is understanding how your computer brings programs to life. This lesson covers process memory structure, the Process Control Block (PCB), process state transitions, and context switching -- the core mechanisms that allow an OS to manage running programs.

## Table of Contents

1. [What is a Process?](#1-what-is-a-process)
2. [Process Memory Structure](#2-process-memory-structure)
3. [Process Control Block (PCB)](#3-process-control-block-pcb)
4. [Process State Transitions](#4-process-state-transitions)
5. [Context Switch](#5-context-switch)
6. [Process Creation and Termination](#6-process-creation-and-termination)
7. [Practice Problems](#7-practice-problems)

---

## 1. What is a Process?

### Program vs Process

```
┌────────────────────────────────────────────────────────┐
│                  Program vs Process                     │
├────────────────────┬───────────────────────────────────┤
│      Program       │              Process               │
├────────────────────┼───────────────────────────────────┤
│ Static entity      │ Dynamic entity                    │
│ Stored on disk     │ Loaded in memory                  │
│ Executable file    │ Executing file                    │
│ Passive            │ Active                            │
│ Doesn't change     │ State constantly changes          │
└────────────────────┴───────────────────────────────────┘

Program ──(load)──▶ Process
          Load into
          memory
```

### Components of a Process

```
Process = Code + Data + Stack + Heap + PCB

┌─────────────────────────────────────┐
│              Process                 │
├─────────────────────────────────────┤
│  ┌───────────────────────────────┐  │
│  │    Text (Code) Section         │  │  Instructions to execute
│  ├───────────────────────────────┤  │
│  │    Data Section                │  │  Global/static variables
│  ├───────────────────────────────┤  │
│  │    Heap                        │  │  Dynamic allocation
│  ├───────────────────────────────┤  │
│  │    Stack                       │  │  Local vars, function calls
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │    PCB (stored in kernel)      │  │  Process metadata
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 2. Process Memory Structure

### Memory Layout

```
High address (0xFFFFFFFF)
┌─────────────────────────────────────┐
│             Kernel Space             │  OS only (no user access)
├─────────────────────────────────────┤ ← 0xC0000000 (Linux 32-bit)
│                                     │
│              Stack                   │  Local variables, parameters
│              ↓ Grows downward        │  Return addresses
│                                     │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│                                     │
│              ↑ Grows upward          │
│              Heap                    │  malloc, new
│                                     │
├─────────────────────────────────────┤
│              BSS                     │  Uninitialized global/static
├─────────────────────────────────────┤
│              Data                    │  Initialized global/static
├─────────────────────────────────────┤
│              Text (Code)             │  Program code (read-only)
└─────────────────────────────────────┘
Low address (0x00000000)
```

### Section Details

```c
#include <stdio.h>
#include <stdlib.h>

// BSS section: uninitialized global
int uninit_global;

// Data section: initialized global
int init_global = 42;

// Data section: static variable
static int static_var = 100;

void example_function(int param) {    // param: stack
    int local_var = 10;               // stack
    static int func_static = 0;       // Data section
    int *heap_ptr;

    heap_ptr = malloc(sizeof(int));   // Allocate on heap
    *heap_ptr = 20;

    printf("local: %d, heap: %d\n", local_var, *heap_ptr);

    free(heap_ptr);                   // Free heap memory
}

// Text section: this code itself
int main() {
    example_function(5);
    return 0;
}
```

### Memory Region Characteristics

```
┌──────────┬──────────┬──────────┬────────────────────────┐
│  Section  │ Read     │ Write    │         Purpose         │
├──────────┼──────────┼──────────┼────────────────────────┤
│ Text     │   O      │   X      │ Program code            │
│ Data     │   O      │   O      │ Initialized global/static│
│ BSS      │   O      │   O      │ Uninitialized global/static│
│ Heap     │   O      │   O      │ Dynamic allocation      │
│ Stack    │   O      │   O      │ Local vars, function calls│
└──────────┴──────────┴──────────┴────────────────────────┘
```

### Stack Frame

```
Function call stack structure:

int add(int a, int b) {
    int result = a + b;
    return result;
}

int main() {
    int x = add(3, 5);
    return 0;
}

┌─────────────────────────────┐ ← High address
│        ...                  │
├─────────────────────────────┤
│    main() stack frame       │
│  ┌───────────────────────┐  │
│  │ x (local variable)    │  │
│  │ Previous frame pointer│  │
│  │ Return address        │  │
│  └───────────────────────┘  │
├─────────────────────────────┤
│    add() stack frame        │
│  ┌───────────────────────┐  │
│  │ result (local var)    │  │
│  │ Previous frame pointer│  │
│  │ Return address        │  │
│  │ b = 5 (parameter)     │  │
│  │ a = 3 (parameter)     │  │
│  └───────────────────────┘  │
├─────────────────────────────┤ ← Stack Pointer (SP)
│        ...                  │
└─────────────────────────────┘ ← Low address
```

---

## 3. Process Control Block (PCB)

### What is PCB?

```
PCB (Process Control Block) = Data structure containing all info to manage a process
                            = Maintained by kernel
                            = Stored in process table

┌───────────────────────────────────────────────────────┐
│                    PCB Structure                       │
├───────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Process Identifier (PID)                         │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Process State (Ready, Running, Waiting...)       │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Program Counter (PC) - next instruction address │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ CPU Registers (general purpose, SP, flags...)    │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ CPU Scheduling Info (priority, scheduling queue) │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Memory Management Info (page table, segment table)│  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Accounting Info (CPU time used, start time...)   │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ I/O Status Info (open files, I/O devices...)     │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Linux task_struct (Simplified)

```c
// Linux kernel process structure (simplified)
struct task_struct {
    // Process identification
    pid_t pid;                    // Process ID — uniquely identifies this process so the kernel
                                  // can track, schedule, and signal it among thousands of others
    pid_t tgid;                   // Thread group ID — groups threads that share an address space
                                  // so signals/exits affect the whole thread group, not just one thread

    // Process state
    volatile long state;          // TASK_RUNNING, TASK_INTERRUPTIBLE...
                                  // The scheduler checks this field to decide whether the process
                                  // is eligible for CPU time or must wait for an event

    // Scheduling info
    int prio;                     // Dynamic priority — adjusted at runtime so interactive processes
                                  // get a responsiveness boost while CPU-hogs are deprioritized
    int static_prio;              // Static priority — set by the user (via nice); provides the
                                  // baseline from which dynamic priority is calculated
    struct sched_entity se;       // Scheduling entity — encapsulates CFS accounting (vruntime)
                                  // so the scheduler can fairly divide CPU time among all processes

    // CPU context
    struct thread_struct thread;  // CPU register state — saves where execution stopped (PC, SP,
                                  // general registers) so the process can resume exactly where
                                  // it was interrupted after a context switch

    // Memory management
    struct mm_struct *mm;         // Memory descriptor — points to the page tables and VMA list;
                                  // without this, the kernel cannot translate virtual addresses
                                  // or enforce per-process memory isolation

    // File system
    struct files_struct *files;   // Open file table — tracks every file descriptor the process
                                  // owns so the kernel can route read/write calls to the right file
    struct fs_struct *fs;         // File system info — stores root dir and current working dir
                                  // so path resolution works correctly for this process

    // Process relationships
    struct task_struct *parent;   // Parent process — needed so wait()/SIGCHLD can propagate
                                  // the child's exit status to the correct parent
    struct list_head children;    // Children processes list — lets the parent iterate over all
                                  // its children (e.g., to reap zombies or forward signals)
    struct list_head sibling;     // Sibling processes list — links processes that share the same
                                  // parent, enabling efficient traversal of the process tree

    // Signals
    struct signal_struct *signal; // Pending/blocked signal info — the kernel checks this on return
                                  // from kernel mode to deliver asynchronous notifications (SIGTERM, etc.)

    // Timing info
    u64 utime, stime;            // User/system CPU time — used for accounting and scheduling
                                  // decisions; also exposed via /proc so admins can find CPU hogs
    u64 start_time;              // Start time — records when the process was created so the kernel
                                  // (and tools like ps) can compute elapsed wall-clock time
};
```

### Process Table

```
┌─────────────────────────────────────────────────────────┐
│                   Process Table                          │
├─────┬──────────────────────────────────────────────────┤
│ PID │                    PCB                            │
├─────┼──────────────────────────────────────────────────┤
│  1  │ init: state=Running, priority=20, mem=4MB...     │
├─────┼──────────────────────────────────────────────────┤
│  2  │ kthreadd: state=Sleeping, priority=10...         │
├─────┼──────────────────────────────────────────────────┤
│ 100 │ bash: state=Ready, priority=20, mem=8MB...       │
├─────┼──────────────────────────────────────────────────┤
│ 101 │ vim: state=Waiting, priority=20, mem=12MB...     │
├─────┼──────────────────────────────────────────────────┤
│ ... │ ...                                              │
└─────┴──────────────────────────────────────────────────┘
```

---

## 4. Process State Transitions

### 5-State Model

```
                        New
                            │
                            │ Admitted
                            ▼
         ┌──────────────┐ Dispatch ┌──────────────┐
         │              │─────────▶│              │
         │   Ready      │          │   Running    │──────┐ Exit
         │              │◀─────────│              │      │
         │              │ Interrupt │              │      │
         └──────────────┘(Timeout)  └──────────────┘      │
                ▲                         │              ▼
                │                         │         ┌──────────┐
                │    I/O or               │         │Terminated│
                │   Event Complete        │         │          │
                │                         │         └──────────┘
                │                         │ I/O or
                │                         │ Event Wait
                │                         ▼
                │              ┌──────────────┐
                └──────────────│   Waiting    │
                               │              │
                               └──────────────┘
```

### State Descriptions

```
┌────────────────┬────────────────────────────────────────┐
│      State      │                Description             │
├────────────────┼────────────────────────────────────────┤
│ New            │ Process being created                  │
├────────────────┼────────────────────────────────────────┤
│ Ready          │ Waiting for CPU assignment             │
│                │ Ready to execute                       │
├────────────────┼────────────────────────────────────────┤
│ Running        │ Executing instructions on CPU          │
│                │ Only one process at a time (single CPU)│
├────────────────┼────────────────────────────────────────┤
│ Waiting        │ Waiting for I/O or event completion    │
│                │ Transitions to Ready when I/O completes│
├────────────────┼────────────────────────────────────────┤
│ Terminated     │ Execution completed, releasing resources│
│                │                                        │
└────────────────┴────────────────────────────────────────┘
```

### State Transition Conditions

```
┌─────────────────┬─────────────────────────────────────────┐
│   Transition     │                Condition                │
├─────────────────┼─────────────────────────────────────────┤
│ New → Ready     │ OS admits process                       │
├─────────────────┼─────────────────────────────────────────┤
│ Ready → Running │ Scheduler assigns CPU (dispatch)        │
├─────────────────┼─────────────────────────────────────────┤
│ Running → Ready │ Time slice expired (timeout)            │
│                 │ Higher priority process arrives (preempt)│
├─────────────────┼─────────────────────────────────────────┤
│ Running → Wait  │ I/O request, event wait                 │
├─────────────────┼─────────────────────────────────────────┤
│ Wait → Ready    │ I/O complete, event occurs              │
├─────────────────┼─────────────────────────────────────────┤
│ Running → Term  │ exit() call, normal/abnormal termination│
└─────────────────┴─────────────────────────────────────────┘
```

### 7-State Model (Including Swapping)

```
                           ┌──────────────────────────────────┐
                           │                                  │
                           │                 Swap out          │
                           ▼                   │              │
┌─────────┐           ┌─────────┐          ┌───┴─────┐        │
│  New    │──────────▶│  Ready  │◀────────▶│ Ready   │        │
│         │           │         │  Swap in  │ Suspend │        │
└─────────┘           └─────────┘          └─────────┘        │
                           │                                  │
                      Dispatch                                │
                           │                                  │
                           ▼                                  │
┌─────────┐           ┌─────────┐                             │
│  Term   │◀──────────│ Running │                             │
│         │           │         │                             │
└─────────┘           └─────────┘                             │
                           │                                  │
                      I/O Request                             │
                           │                                  │
                           ▼                   Swap out       │
                      ┌─────────┐          ┌─────────┐        │
                      │ Waiting │◀────────▶│ Waiting │────────┘
                      │         │  Swap in  │ Suspend │
                      └─────────┘          └─────────┘

Suspend state: Process swapped out from memory to disk
```

---

## 5. Context Switch

### What is a Context Switch?

```
Context Switch = Process of switching CPU to another process
               = Save current process state, restore new process state

┌────────────────────────────────────────────────────────────┐
│                    Context Switch Process                   │
└────────────────────────────────────────────────────────────┘

Process P0              Operating System              Process P1
    │                      │                      │
    │  Executing            │                      │  Waiting
    │                      │                      │
    │──Interrupt/Syscall──▶│                      │
    │                      │                      │
    │                  Save state to PCB0         │
    │                      │                      │
    │                  Restore state from PCB1    │
    │                      │                      │
    │                      │──────────────────────▶│
    │  Waiting             │                      │  Executing
    │                      │                      │
    │                      │◀──Interrupt/Syscall──│
    │                      │                      │
    │                  Save state to PCB1         │
    │                      │                      │
    │                  Restore state from PCB0    │
    │                      │                      │
    │◀─────────────────────│                      │
    │  Executing           │                      │  Waiting
```

### Information Saved/Restored in Context Switch

```
┌────────────────────────────────────────────────────────┐
│               Context                                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  CPU Registers:                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • Program Counter (PC)                            │  │
│  │ • Stack Pointer (SP)                             │  │
│  │ • Base Pointer (BP)                              │  │
│  │ • General Purpose Registers (RAX, RBX, RCX, RDX...)│  │
│  │ • Status Register (FLAGS)                        │  │
│  │ • Floating Point Registers                       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Memory Management Info:                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • Page Table Base Register                       │  │
│  │ • Segment Registers                              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Context Switch Cost

```
┌────────────────────────────────────────────────────────┐
│               Context Switch Cost                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Direct cost:                                          │
│  • Register save/restore: ~hundreds of nanoseconds     │
│  • Kernel mode transition: ~hundreds of nanoseconds    │
│                                                        │
│  Indirect cost (larger):                               │
│  • TLB flush: hundreds~thousands of cycles             │
│  • Cache miss increase (cache pollution)               │
│  • Pipeline flush                                      │
│                                                        │
│  Typical total cost: 1~10 microseconds                 │
│                                                        │
└────────────────────────────────────────────────────────┘

Timeline view of context switch:

P0 exec  │ Context Switch │ P1 exec  │ Context Switch │ P0 exec
━━━━━━━━│     Overhead    │━━━━━━━━│     Overhead    │━━━━━━━
        │← ~1-10 μs →│        │← ~1-10 μs →│
                 ↑ No useful work during this time
```

---

## 6. Process Creation and Termination

### Process Creation with fork()

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid;
    int x = 10;

    printf("Parent process starting, PID: %d\n", getpid());

    pid = fork();  // Fork point

    if (pid < 0) {
        // fork failed
        perror("fork failed");
        return 1;
    }
    else if (pid == 0) {
        // Child process
        printf("Child: PID=%d, Parent PID=%d\n", getpid(), getppid());
        x = x + 10;
        printf("Child: x = %d\n", x);
    }
    else {
        // Parent process
        printf("Parent: PID=%d, Child PID=%d\n", getpid(), pid);
        wait(NULL);  // Wait for child termination
        printf("Parent: x = %d\n", x);  // Still 10
    }

    return 0;
}

/*
Output:
Parent process starting, PID: 1234
Parent: PID=1234, Child PID=1235
Child: PID=1235, Parent PID=1234
Child: x = 20
Parent: x = 10
*/
```

### How fork() Works

```
Before fork():
┌─────────────────────────────────┐
│        Parent Process (PID: 100) │
│  ┌─────────────────────────┐    │
│  │ x = 10                  │    │
│  │ Code/Data/Stack/Heap    │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘

After fork():
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│        Parent Process (PID: 100) │    │        Child Process (PID: 101)  │
│  ┌─────────────────────────┐    │    │  ┌─────────────────────────┐    │
│  │ x = 10                  │    │    │  │ x = 10 (copied)         │    │
│  │ Code/Data/Stack/Heap    │    │    │  │ Code/Data/Stack/Heap    │    │
│  │ fork() returns: 101     │    │    │  │ fork() returns: 0       │    │
│  └─────────────────────────┘    │    │  └─────────────────────────┘    │
└─────────────────────────────────┘    └─────────────────────────────────┘
          │                                       │
          │  Two processes are independent        │
          │  (separate memory spaces)             │
          ▼                                       ▼
```

### Program Execution with exec()

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: execute ls command
        printf("Child: executing ls\n");

        // execl: exec with list of arguments
        execl("/bin/ls", "ls", "-l", NULL);

        // If exec succeeds, this code won't execute
        perror("exec failed");
        return 1;
    }
    else {
        // Parent process
        wait(NULL);
        printf("Parent: child terminated\n");
    }

    return 0;
}
```

### How exec() Works

```
Before exec():
┌─────────────────────────────────┐
│        Child Process             │
│  ┌─────────────────────────┐    │
│  │ Original program code    │    │
│  │ Original data            │    │
│  │ Original stack/heap      │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘

After exec("/bin/ls"):
┌─────────────────────────────────┐
│        Child Process             │
│  ┌─────────────────────────┐    │
│  │ ls program code          │    │  ← Completely replaced
│  │ ls data                  │    │    with new program
│  │ New stack/heap           │    │
│  └─────────────────────────┘    │
│                                 │
│  PID remains the same            │
└─────────────────────────────────┘
```

### Process Termination

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) {
        printf("Child process terminating\n");
        exit(42);  // Exit with status code 42
    }
    else {
        int status;
        wait(&status);  // Collect child's exit status

        if (WIFEXITED(status)) {
            printf("Child exit code: %d\n", WEXITSTATUS(status));
        }
    }

    return 0;
}

/*
Output:
Child process terminating
Child exit code: 42
*/
```

### Zombie and Orphan Processes

```
Zombie Process:
- Child has terminated but parent hasn't called wait()
- PCB remains (stores exit status)
- Resources released but process table entry maintained

┌──────────────────────────────────────────────┐
│  Parent Process (PID: 100)                   │
│  - Hasn't called wait()                      │
└──────────────────────────────────────────────┘
          │
          │ (Relationship maintained)
          ▼
┌──────────────────────────────────────────────┐
│  Zombie Process (PID: 101)                   │
│  - State: Z (Zombie)                         │
│  - Code/Data/Stack released                  │
│  - Only PCB remains                          │
└──────────────────────────────────────────────┘


Orphan Process:
- Parent terminated before child
- init (PID 1) or systemd becomes new parent

┌──────────────────────────────────────────────┐
│  init (PID: 1)                               │
│  - New parent of orphan processes            │
│  - Periodically calls wait()                 │
└──────────────────────────────────────────────┘
          │
          │ (Adoption)
          ▼
┌──────────────────────────────────────────────┐
│  Orphan Process (PID: 102)                   │
│  - Original parent (PID: 100) terminated     │
│  - PPID changed to 1                         │
└──────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Problem 1: Memory Region Identification

Identify the memory region where each variable is stored.

```c
int global_var = 100;        // (   )
int uninitialized;           // (   )
const char* str = "hello";   // (   )

void func() {
    int local = 10;          // (   )
    static int stat = 20;    // (   )
    int* ptr = malloc(4);    // ptr: (   ), *ptr: (   )
}
```

Options: Text, Data, BSS, Stack, Heap

<details>
<summary>Show Answer</summary>

```
int global_var = 100;        // (Data)
int uninitialized;           // (BSS)
const char* str = "hello";   // str: Data, "hello": Text (read-only)

void func() {
    int local = 10;          // (Stack)
    static int stat = 20;    // (Data)
    int* ptr = malloc(4);    // ptr: (Stack), *ptr: (Heap)
}
```

</details>

### Problem 2: Process State Transitions

Explain the process state transitions in the following situations.

1. Process A is using CPU, time slice expires
2. Process B requests file read
3. Process C's file read completes
4. Scheduler selects process D

<details>
<summary>Show Answer</summary>

1. A: Running → Ready (timeout/interrupt)
2. B: Running → Waiting (I/O request)
3. C: Waiting → Ready (I/O complete)
4. D: Ready → Running (dispatch)

</details>

### Problem 3: fork() Output Prediction

Predict the output of the following code.

```c
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("A\n");
    fork();
    printf("B\n");
    fork();
    printf("C\n");
    return 0;
}
```

<details>
<summary>Show Answer</summary>

```
A       (1 output - original process)
B       (2 outputs - 2 processes after first fork)
B
C       (4 outputs - 4 processes after second fork)
C
C
C

Total: A 1 time, B 2 times, C 4 times

Process branching:
        main
          │
    ┌─────┴─────┐
    │  fork()   │
    │           │
  main        child1
    │           │
 ┌──┴──┐     ┌──┴──┐
 │fork()│    │fork()│
 │     │    │     │
main  c2   c1    c3

4 processes each print "C"
```

</details>

### Problem 4: PCB Information

Explain the changes to PCB information in the following situations.

1. When a process transitions from Running to Waiting
2. When a context switch occurs

<details>
<summary>Show Answer</summary>

1. Running → Waiting transition:
   - PCB's state field changes from Running to Waiting
   - I/O status info records the waiting I/O operation
   - Process moves to waiting queue

2. Context switch:
   - Save current process's PC, registers, stack pointer to PCB
   - Restore PC, registers, stack pointer from new process's PCB
   - Update memory management info (page table)
   - Change new process's state to Running

</details>

### Problem 5: Context Switch Cost

Describe two direct costs and two indirect costs of context switching.

<details>
<summary>Show Answer</summary>

**Direct costs:**
1. Register save/restore: Time to save current process's registers to PCB and restore new process's registers
2. Kernel mode transition: Overhead of transitioning from user mode to kernel mode and back

**Indirect costs:**
1. Cache pollution: New process's data not in cache, increasing cache misses
2. TLB flush: New process uses different virtual address space, invalidating TLB entries

</details>

---

## Hands-On Exercises

### Exercise 1: Process Lifecycle Simulation

Run `examples/OS_Theory/02_process_demo.py` and observe the output.

**Tasks:**
1. Add a new process "P5" with priority 3 and trace its state transitions: NEW → READY → RUNNING → WAITING → READY → RUNNING → TERMINATED
2. Modify the `ProcessTable` to track the total time each process spends in each state
3. Add a `kill_process(pid)` method that forcibly transitions any state to TERMINATED

### Exercise 2: Process Tree Exploration

Use system tools to explore the process hierarchy on your machine:

```bash
# Linux
pstree -p | head -30

# macOS
ps -axo pid,ppid,comm | head -30
```

**Tasks:**
1. Identify the init/launchd process (PID 1) and trace 3 levels of its child processes
2. What is the PPID of your current shell? Trace the ancestry back to PID 1
3. Write a Python script using `os.getpid()` and `os.getppid()` that prints its own ancestry

### Exercise 3: Context Switch Overhead

Measure context switch overhead using pipe-based ping-pong between two processes:

```python
import os, time

def measure_context_switches(n=10000):
    r1, w1 = os.pipe()
    r2, w2 = os.pipe()

    pid = os.fork()
    if pid == 0:
        for _ in range(n):
            os.read(r1, 1)
            os.write(w2, b'x')
        os._exit(0)
    else:
        start = time.perf_counter()
        for _ in range(n):
            os.write(w1, b'x')
            os.read(r2, 1)
        elapsed = time.perf_counter() - start
        os.wait()
        print(f"{n} round-trips: {elapsed*1000:.1f} ms")
        print(f"Per switch: {elapsed/n*1e6:.1f} µs")

measure_context_switches()
```

**Tasks:**
1. Run the script and interpret the per-switch latency
2. How does this compare to the theoretical minimum (register save/restore time)?
3. What additional costs beyond register save/restore contribute to context switch overhead?

---

## Exercises

### Exercise 1: Memory Layout Analysis

For each variable or expression in the program below, state the memory section (Text, Data, BSS, Heap, or Stack) where it resides and briefly explain why.

```c
#include <stdio.h>
#include <stdlib.h>

int server_port = 8080;          // (1)
char *app_name;                  // (2)
static int request_count = 0;   // (3)

void handle_request(int id) {
    char buf[256];               // (4)
    static int call_num = 0;     // (5)
    int *data = malloc(1024);    // (6) where is data itself? where does it point?
    free(data);
}
```

### Exercise 2: Process State Transitions

A system has four processes: P1, P2, P3, and P4. Trace the state of each process at each time step based on the events listed below. Use: New, Ready, Running, Waiting, Terminated.

| Time | Event |
|------|-------|
| t=0 | P1 created, P2 created |
| t=1 | P1 dispatched to CPU |
| t=2 | P3 created |
| t=3 | P1 requests file read (I/O) |
| t=4 | P2 dispatched; P4 created |
| t=5 | P1's I/O completes |
| t=6 | P2's time slice expires |
| t=7 | P1 dispatched; P3 dispatched (multicore) |
| t=8 | P3 calls exit() |

Fill in the table:

| Process | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| P1 | | | | | | | | | |
| P2 | | | | | | | | | |
| P3 | | | | | | | | | |
| P4 | | | | | | | | | |

### Exercise 3: fork() Output Prediction

Predict the exact output of the following program, including the number of times each line is printed. Assume no buffering issues and that PIDs are assigned as 1000, 1001, 1002 in order.

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int x = 0;

int main() {
    printf("start\n");

    pid_t p1 = fork();
    if (p1 == 0) {
        x = 10;
        printf("child1: x=%d\n", x);
        return 0;
    }

    pid_t p2 = fork();
    if (p2 == 0) {
        x = 20;
        printf("child2: x=%d\n", x);
        return 0;
    }

    wait(NULL);
    wait(NULL);
    printf("parent: x=%d\n", x);
    return 0;
}
```

1. How many processes are created in total (including the original)?
2. What is the value of `x` in the parent at the end? Why?
3. Can the two child output lines appear in either order? Why?

### Exercise 4: Zombie and Orphan Processes

Read the following code and answer the questions.

```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    pid_t pid = fork();

    if (pid > 0) {
        // Parent
        printf("Parent sleeping...\n");
        sleep(60);   // Does NOT call wait()
        printf("Parent done\n");
    } else {
        // Child
        printf("Child exiting immediately\n");
        exit(0);
    }
    return 0;
}
```

1. What state does the child process enter after calling `exit(0)` while the parent sleeps? Why?
2. How could you observe this state using command-line tools on Linux?
3. What change to the parent code would prevent this situation?
4. Suppose the parent itself crashes before `sleep(60)` ends. What happens to the child? What process adopts it?

### Exercise 5: Context Switch Cost Estimation

A system performs 1,000 context switches per second. Each switch costs approximately 5 microseconds for direct overhead (register save/restore) and an additional 15 microseconds for indirect overhead (TLB flush, cache warm-up).

1. Calculate the total CPU time lost to context switching per second
2. Express this as a percentage of a 1 GHz single-core CPU's total cycles per second
3. If the system reduces context switches to 500 per second by doubling the time quantum, what is the new overhead percentage?
4. Name two workload types where reducing context switches would hurt performance even if it reduces overhead

---

## Next Steps

- [03_Threads_and_Multithreading.md](./03_Threads_and_Multithreading.md) - Thread concepts and multithreading models

---

## References

- [OSTEP - Processes](https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf)
- [Linux man pages - fork](https://man7.org/linux/man-pages/man2/fork.2.html)
- [Linux Kernel - task_struct](https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h)
