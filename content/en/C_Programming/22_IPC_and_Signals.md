# Inter-Process Communication and Signals

**Previous**: [Network Programming in C](./21_Network_Programming.md) | **Next**: [Testing and Profiling in C](./23_Testing_and_Profiling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create anonymous pipes for parent-child communication and redirect standard I/O with `dup2`
2. Use named pipes (FIFOs) to exchange data between unrelated processes
3. Map shared memory regions with `shm_open` and `mmap`, and synchronize access with POSIX semaphores
4. Send and receive structured messages through POSIX message queues with priority ordering
5. Install signal handlers using `sigaction` and write async-signal-safe code
6. Block and unblock signals with `sigprocmask` to protect critical sections
7. Reap child processes automatically using a `SIGCHLD` handler with non-blocking `waitpid`

---

When a single process is not enough -- when you need a pipeline of workers, a watchdog that restarts crashed services, or a shared scoreboard between cooperating daemons -- you need inter-process communication. IPC is the plumbing behind every Unix shell pipeline, every database that coordinates multiple backends, and every container orchestrator that manages worker processes.

> **Analogy -- Texts, Whiteboards, and Mailboxes**: Inter-process communication mechanisms map to everyday communication. A **signal** is like an urgent text notification -- it interrupts whatever you're doing with a brief code (SIGTERM = "please stop"). **Shared memory** is a whiteboard in a shared office -- any process can read or scribble on it instantly, but you need rules (semaphores) so nobody erases someone else's work mid-sentence. A **message queue** is a mailbox: senders drop letters in and receivers pick them up in order, even if they arrive at different times.

**Difficulty**: Advanced

---

## Table of Contents

1. [Pipes](#1-pipes)
2. [Named Pipes (FIFOs)](#2-named-pipes-fifos)
3. [Shared Memory](#3-shared-memory)
4. [POSIX Message Queues](#4-posix-message-queues)
5. [Signals](#5-signals)
6. [Practice Problems](#6-practice-problems)
7. [References](#7-references)

---

## 1. Pipes

### 1.1 Anonymous Pipes

Pipes provide unidirectional data flow between related processes (parent-child).

```
┌────────────────────────────────────────────────────────┐
│                  Pipe Communication                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌───────────┐    pipe    ┌───────────┐               │
│  │  Parent   │───────────▶│  Child    │               │
│  │  (Writer) │   fd[1]    │  (Reader) │               │
│  │           │  ────────▶ │           │               │
│  └───────────┘   fd[0]    └───────────┘               │
│                                                        │
│  pipe(fd) creates:                                     │
│    fd[0] = read end                                    │
│    fd[1] = write end                                   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // Child: read from pipe
        close(pipefd[1]);  // Close unused write end

        char buffer[256];
        ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);
        if (n > 0) {
            buffer[n] = '\0';
            printf("Child received: %s\n", buffer);
        }

        close(pipefd[0]);
        exit(EXIT_SUCCESS);
    } else {
        // Parent: write to pipe
        close(pipefd[0]);  // Close unused read end

        const char *msg = "Hello from parent!";
        write(pipefd[1], msg, strlen(msg));

        close(pipefd[1]);
        wait(NULL);  // Wait for child
    }

    return 0;
}
```

### 1.2 Bidirectional Communication with Two Pipes

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int parent_to_child[2], child_to_parent[2];
    pipe(parent_to_child);
    pipe(child_to_parent);

    pid_t pid = fork();
    if (pid == 0) {
        // Child
        close(parent_to_child[1]);
        close(child_to_parent[0]);

        char buf[256];
        ssize_t n = read(parent_to_child[0], buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Child got: %s\n", buf);

        const char *reply = "Got it, thanks!";
        write(child_to_parent[1], reply, strlen(reply));

        close(parent_to_child[0]);
        close(child_to_parent[1]);
        exit(0);
    }

    // Parent
    close(parent_to_child[0]);
    close(child_to_parent[1]);

    const char *msg = "Task: process data";
    write(parent_to_child[1], msg, strlen(msg));
    close(parent_to_child[1]);

    char buf[256];
    ssize_t n = read(child_to_parent[0], buf, sizeof(buf) - 1);
    buf[n] = '\0';
    printf("Parent got reply: %s\n", buf);

    close(child_to_parent[0]);
    wait(NULL);
    return 0;
}
```

### 1.3 Pipe with exec (Shell-like Piping)

```c
// Simulate: ls -la | grep ".c"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int pipefd[2];
    pipe(pipefd);

    pid_t pid1 = fork();
    if (pid1 == 0) {
        // First child: ls -la
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);  // stdout → pipe write
        close(pipefd[1]);
        execlp("ls", "ls", "-la", NULL);
        perror("execlp ls");
        exit(1);
    }

    pid_t pid2 = fork();
    if (pid2 == 0) {
        // Second child: grep ".c"
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);  // stdin ← pipe read
        close(pipefd[0]);
        execlp("grep", "grep", ".c", NULL);
        perror("execlp grep");
        exit(1);
    }

    // Parent: close both ends and wait
    close(pipefd[0]);
    close(pipefd[1]);
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);

    return 0;
}
```

---

## 2. Named Pipes (FIFOs)

FIFOs allow communication between unrelated processes through a filesystem entry.

### 2.1 Creating and Using FIFOs

```c
// --- Writer process ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define FIFO_PATH "/tmp/myfifo"

int main(void) {
    // Create FIFO (ignore error if it already exists)
    mkfifo(FIFO_PATH, 0666);

    int fd = open(FIFO_PATH, O_WRONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    const char *messages[] = {"Hello", "World", "Done"};
    for (int i = 0; i < 3; i++) {
        write(fd, messages[i], strlen(messages[i]) + 1);
        printf("Sent: %s\n", messages[i]);
        sleep(1);
    }

    close(fd);
    return 0;
}
```

```c
// --- Reader process ---
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define FIFO_PATH "/tmp/myfifo"

int main(void) {
    int fd = open(FIFO_PATH, O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    char buffer[256];
    ssize_t n;
    while ((n = read(fd, buffer, sizeof(buffer))) > 0) {
        printf("Received: %s\n", buffer);
    }

    close(fd);
    unlink(FIFO_PATH);  // Clean up
    return 0;
}
```

---

## 3. Shared Memory

Shared memory is the fastest IPC mechanism because data does not need to be copied between processes.

### 3.1 POSIX Shared Memory

```
┌──────────────────────────────────────────────────────────┐
│              Shared Memory Architecture                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐       Physical Memory       ┌─────────┐│
│  │  Process A  │       ┌──────────────┐      │Process B ││
│  │             │       │              │      │          ││
│  │  Virtual    │──────▶│  Shared      │◀─────│ Virtual  ││
│  │  Address    │  mmap │  Region      │ mmap │ Address  ││
│  │  0x7f...    │       │              │      │ 0x7f...  ││
│  │             │       └──────────────┘      │          ││
│  └─────────────┘                             └─────────┘│
│                                                          │
│  ⚠ Requires synchronization (semaphore/mutex)           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

```c
// --- Producer ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME "/my_shm"
#define SEM_NAME "/my_sem"
#define SHM_SIZE 4096

typedef struct {
    int count;
    char data[256];
} shared_data_t;

int main(void) {
    // Create shared memory
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, sizeof(shared_data_t));

    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, shm_fd, 0);

    // Create semaphore for synchronization
    sem_t *sem = sem_open(SEM_NAME, O_CREAT, 0666, 0);

    // Write data
    shm->count = 42;
    snprintf(shm->data, sizeof(shm->data),
             "Hello from producer (PID=%d)", getpid());

    printf("Producer wrote: count=%d, data=%s\n",
           shm->count, shm->data);

    // Signal consumer
    sem_post(sem);

    // Cleanup
    sem_close(sem);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);

    return 0;
}
```

```c
// --- Consumer ---
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME "/my_shm"
#define SEM_NAME "/my_sem"

typedef struct {
    int count;
    char data[256];
} shared_data_t;

int main(void) {
    // Open shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ, MAP_SHARED, shm_fd, 0);

    // Wait for producer
    sem_t *sem = sem_open(SEM_NAME, 0);
    sem_wait(sem);

    // Read data
    printf("Consumer read: count=%d, data=%s\n",
           shm->count, shm->data);

    // Cleanup
    sem_close(sem);
    sem_unlink(SEM_NAME);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);
    shm_unlink(SHM_NAME);

    return 0;
}
```

---

## 4. POSIX Message Queues

Message queues provide structured message passing with priority support.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mqueue.h>
#include <fcntl.h>

#define QUEUE_NAME "/my_queue"
#define MAX_MSG_SIZE 256
#define MAX_MSGS 10

// Sender
void sender(void) {
    struct mq_attr attr = {
        .mq_flags = 0,
        .mq_maxmsg = MAX_MSGS,
        .mq_msgsize = MAX_MSG_SIZE,
        .mq_curmsgs = 0
    };

    mqd_t mq = mq_open(QUEUE_NAME, O_CREAT | O_WRONLY, 0666, &attr);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        exit(1);
    }

    const char *msgs[] = {"High priority!", "Normal message", "Low priority"};
    unsigned int priorities[] = {10, 5, 1};

    for (int i = 0; i < 3; i++) {
        mq_send(mq, msgs[i], strlen(msgs[i]) + 1, priorities[i]);
        printf("Sent (prio=%u): %s\n", priorities[i], msgs[i]);
    }

    mq_close(mq);
}

// Receiver
void receiver(void) {
    mqd_t mq = mq_open(QUEUE_NAME, O_RDONLY);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        exit(1);
    }

    char buffer[MAX_MSG_SIZE];
    unsigned int priority;

    // Messages arrive highest priority first
    for (int i = 0; i < 3; i++) {
        ssize_t bytes = mq_receive(mq, buffer, MAX_MSG_SIZE, &priority);
        if (bytes >= 0) {
            printf("Received (prio=%u): %s\n", priority, buffer);
        }
    }

    mq_close(mq);
    mq_unlink(QUEUE_NAME);
}
```

---

## 5. Signals

### 5.1 Signal Overview

Signals are software interrupts delivered to a process to notify it of events.

```
┌──────────────────────────────────────────────────────────┐
│  Common Signals                                          │
├─────────┬──────────────────────────────────────────────┤
│ Signal  │ Description                                    │
├─────────┼──────────────────────────────────────────────┤
│ SIGINT  │ Interrupt (Ctrl+C)                             │
│ SIGTERM │ Termination request                            │
│ SIGKILL │ Forced kill (cannot be caught)                 │
│ SIGCHLD │ Child process stopped or terminated            │
│ SIGUSR1 │ User-defined signal 1                          │
│ SIGUSR2 │ User-defined signal 2                          │
│ SIGALRM │ Timer alarm                                    │
│ SIGPIPE │ Broken pipe (write to closed socket)           │
│ SIGSEGV │ Segmentation fault                             │
│ SIGSTOP │ Stop process (cannot be caught)                │
│ SIGCONT │ Continue stopped process                       │
└─────────┴──────────────────────────────────────────────┘
```

### 5.2 Signal Handling with sigaction

Always prefer `sigaction()` over `signal()` for portable, reliable behavior.

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

volatile sig_atomic_t running = 1;

void handle_sigint(int sig) {
    (void)sig;  // Suppress unused warning
    running = 0;
    // Only async-signal-safe functions here!
    write(STDOUT_FILENO, "\nCaught SIGINT, shutting down...\n", 33);
}

void handle_sigusr1(int sig, siginfo_t *info, void *context) {
    (void)sig;
    (void)context;
    // siginfo_t gives us sender information
    printf("SIGUSR1 from PID %d\n", info->si_pid);
}

int main(void) {
    // Setup SIGINT handler
    struct sigaction sa_int = {0};
    sa_int.sa_handler = handle_sigint;
    sigemptyset(&sa_int.sa_mask);
    sa_int.sa_flags = 0;
    sigaction(SIGINT, &sa_int, NULL);

    // Setup SIGUSR1 handler with siginfo
    struct sigaction sa_usr = {0};
    sa_usr.sa_sigaction = handle_sigusr1;
    sigemptyset(&sa_usr.sa_mask);
    sa_usr.sa_flags = SA_SIGINFO;
    sigaction(SIGUSR1, &sa_usr, NULL);

    // Ignore SIGPIPE (common in network programs)
    signal(SIGPIPE, SIG_IGN);

    printf("PID: %d - Press Ctrl+C or send SIGUSR1\n", getpid());

    while (running) {
        printf("Working...\n");
        sleep(2);
    }

    printf("Clean shutdown complete\n");
    return 0;
}
```

### 5.3 Signal Masking

```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    sigset_t block_set, old_set;

    // Block SIGINT during critical section
    sigemptyset(&block_set);
    sigaddset(&block_set, SIGINT);

    sigprocmask(SIG_BLOCK, &block_set, &old_set);

    // ---- Critical section ----
    printf("SIGINT blocked. Ctrl+C won't interrupt.\n");
    sleep(5);
    printf("Critical section done.\n");
    // ---- End critical section ----

    // Restore original mask
    sigprocmask(SIG_SETMASK, &old_set, NULL);
    printf("SIGINT unblocked. Ctrl+C works again.\n");

    sleep(5);
    return 0;
}
```

### 5.4 Reaping Child Processes with SIGCHLD

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

void handle_sigchld(int sig) {
    (void)sig;
    // Reap all terminated children (non-blocking)
    int status;
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        if (WIFEXITED(status)) {
            // Child exited normally
        }
    }
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handle_sigchld;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    // Fork multiple children
    for (int i = 0; i < 5; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            printf("Child %d (PID=%d) working...\n", i, getpid());
            sleep(i + 1);
            printf("Child %d done\n", i);
            exit(i);
        }
    }

    // Parent continues working
    printf("Parent (PID=%d) waiting...\n", getpid());
    sleep(10);
    printf("Parent done\n");

    return 0;
}
```

---

## 6. Practice Problems

### Problem 1: Producer-Consumer with Shared Memory

Implement a producer-consumer system using POSIX shared memory and semaphores:
- Producer writes integers 1-100 into a circular buffer in shared memory
- Consumer reads and prints them
- Use semaphores for synchronization

### Problem 2: Multi-process Pipeline

Create a three-stage pipeline using pipes:
- Stage 1: Reads lines from a file
- Stage 2: Converts to uppercase
- Stage 3: Counts and prints word frequency

### Problem 3: Watchdog Process

Write a watchdog that:
- Forks a child worker process
- Monitors it with SIGCHLD
- Restarts it automatically if it crashes
- Handles SIGTERM for graceful shutdown of both processes

---

## 7. References

- W. Richard Stevens, *Advanced Programming in the UNIX Environment* (3rd ed.)
- `man 7 pipe`, `man 7 fifo`, `man 7 shm_overview`, `man 7 mq_overview`
- `man 7 signal`, `man 2 sigaction`, `man 2 sigprocmask`

---

## Exercises

### Exercise 1: Pipe Chain Tracing

Write a program that creates a chain of three processes — P1, P2, P3 — connected by two anonymous pipes:

1. P1 generates numbers 1 through 10, writing each as text (e.g., `"1\n"`) to pipe A.
2. P2 reads from pipe A, doubles each number, and writes the result to pipe B.
3. P3 reads from pipe B and prints each result to stdout.

Each process must close the pipe ends it does not use immediately after `fork`. After writing the code, answer: what happens to P2 and P3 if P1 closes its write end without writing anything? Verify your answer experimentally.

### Exercise 2: FIFO-Based Logger

Implement a two-program logging system using a named pipe (FIFO):

1. **Logger daemon**: Creates `/tmp/app_log.fifo` with `mkfifo`, opens it for reading, and continuously reads log lines, prepending a timestamp (`[HH:MM:SS]`) and writing them to a file `app.log`.
2. **Application**: Opens `/tmp/app_log.fifo` for writing and sends five log messages one second apart, then closes the FIFO.
3. After the application exits, the logger should detect EOF and print `"Log complete"` before cleaning up with `unlink`.

Run both programs in separate terminals and verify that `app.log` contains all five timestamped messages.

### Exercise 3: Shared Counter with Semaphore Protection

Demonstrate a race condition and then fix it with a POSIX semaphore:

1. Create a shared memory region containing a single `int counter` initialized to 0.
2. Fork 4 child processes; each increments the counter 10,000 times **without** synchronization. Run and observe that the final value is often less than 40,000 due to the race.
3. Add a named semaphore initialized to 1. Wrap each increment in `sem_wait` / `sem_post`. Run again and confirm the final value is exactly 40,000.
4. Clean up all shared resources (`shm_unlink`, `sem_unlink`) in the parent after all children exit.

### Exercise 4: Safe Signal Handler with Pipe Self-Trick

Rewrite the `SIGINT` handler from Section 5.2 using the "self-pipe trick" for signal-safe I/O multiplexing:

1. Create a `pipe(selfpipe)` before entering the main loop.
2. In the `SIGINT` signal handler (installed with `sigaction`), write a single byte `'S'` to `selfpipe[1]` — this is async-signal-safe.
3. In the main loop, use `select()` or `poll()` to wait on both `STDIN_FILENO` and `selfpipe[0]`.
4. When `selfpipe[0]` becomes readable, read the byte, print `"Graceful shutdown initiated"`, and exit.

Explain in a comment block why calling `printf` directly inside a signal handler is unsafe and what problems the self-pipe trick avoids.

### Exercise 5: POSIX Message Queue Priority Scheduler

Build a simple priority-based task scheduler using a POSIX message queue:

1. Define a `task_t` struct containing a `char description[128]` and an `int priority` (1 = low, 5 = medium, 10 = high).
2. Write a producer that enqueues 6 tasks with varying priorities and descriptions in random order.
3. Write a consumer that dequeues tasks one at a time (highest priority first, as guaranteed by `mq_receive`) and prints each task's priority and description.
4. Confirm that the consumer always processes higher-priority tasks before lower-priority ones, regardless of the order the producer sent them.
5. Clean up with `mq_unlink` after all tasks are processed.

---
