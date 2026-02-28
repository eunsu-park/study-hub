/*
 * Exercises for Lesson 22: IPC and Signals
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o ex22 22_ipc_and_signals.c
 *
 * Note: Exercises use POSIX APIs (fork, pipe, mmap, sem_open, mq_open).
 * These work on Linux and macOS. On macOS, POSIX message queues (Exercise 5)
 * are not natively supported; that exercise includes a simulation fallback.
 *
 * For Exercise 3 on Linux: gcc -Wall -Wextra -std=c11 -o ex22 22_ipc_and_signals.c -lpthread -lrt
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <semaphore.h>
#include <sys/stat.h>

/* === Exercise 1: Pipe Chain Tracing === */
/* Problem: P1 -> pipe A -> P2 (doubles) -> pipe B -> P3 (prints) */

void exercise_1(void) {
    printf("=== Exercise 1: Pipe Chain (P1 -> P2 -> P3) ===\n");

    /*
     * Architecture:
     *   P1 writes 1..10 to pipe_a
     *   P2 reads from pipe_a, doubles each number, writes to pipe_b
     *   P3 reads from pipe_b and prints to stdout
     *
     * Pipe file descriptors:
     *   pipe_a[0] = read end,  pipe_a[1] = write end  (P1 -> P2)
     *   pipe_b[0] = read end,  pipe_b[1] = write end  (P2 -> P3)
     *
     * Critical: Each process must close unused pipe ends immediately
     * after fork. Failure to do so prevents EOF detection.
     */

    int pipe_a[2], pipe_b[2];

    if (pipe(pipe_a) == -1 || pipe(pipe_b) == -1) {
        perror("pipe");
        return;
    }

    /* Fork P1: number generator */
    pid_t pid1 = fork();
    if (pid1 == 0) {
        /* P1: writes to pipe_a, no access to pipe_b */
        close(pipe_a[0]);   /* Close read end of pipe_a */
        close(pipe_b[0]);   /* Close both ends of pipe_b */
        close(pipe_b[1]);

        for (int i = 1; i <= 10; i++) {
            char buf[16];
            int len = snprintf(buf, sizeof(buf), "%d\n", i);
            write(pipe_a[1], buf, (size_t)len);
        }

        close(pipe_a[1]);   /* Signal EOF to P2 */
        _exit(0);
    }

    /* Fork P2: doubler */
    pid_t pid2 = fork();
    if (pid2 == 0) {
        /* P2: reads from pipe_a, writes to pipe_b */
        close(pipe_a[1]);   /* Close write end of pipe_a */
        close(pipe_b[0]);   /* Close read end of pipe_b */

        /* Read line by line from pipe_a */
        FILE *in = fdopen(pipe_a[0], "r");
        char line[64];
        while (fgets(line, sizeof(line), in)) {
            int num = atoi(line);
            int doubled = num * 2;
            char buf[16];
            int len = snprintf(buf, sizeof(buf), "%d\n", doubled);
            write(pipe_b[1], buf, (size_t)len);
        }

        fclose(in);          /* Also closes pipe_a[0] */
        close(pipe_b[1]);    /* Signal EOF to P3 */
        _exit(0);
    }

    /* Fork P3: printer */
    pid_t pid3 = fork();
    if (pid3 == 0) {
        /* P3: reads from pipe_b only */
        close(pipe_a[0]);
        close(pipe_a[1]);
        close(pipe_b[1]);   /* Close write end of pipe_b */

        FILE *in = fdopen(pipe_b[0], "r");
        char line[64];
        printf("  P3 output (doubled values):\n");
        while (fgets(line, sizeof(line), in)) {
            printf("    %s", line);
        }

        fclose(in);
        _exit(0);
    }

    /* Parent: close all pipe ends and wait */
    close(pipe_a[0]);
    close(pipe_a[1]);
    close(pipe_b[0]);
    close(pipe_b[1]);

    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);
    waitpid(pid3, NULL, 0);

    /*
     * Q: What happens if P1 closes its write end without writing anything?
     * A: P2's read() returns 0 (EOF) immediately. P2 then closes pipe_b's
     *    write end without writing. P3 also gets EOF immediately.
     *    All processes exit cleanly -- no hang, no error.
     *    This is the correct behavior: closing the write end signals EOF
     *    to all readers.
     */
    printf("\n  Note: If P1 writes nothing and closes pipe_a[1],\n");
    printf("  P2 gets EOF immediately, passes it along, P3 exits cleanly.\n");
}

/* === Exercise 2: FIFO-Based Logger === */
/* Problem: Logger daemon reads from a FIFO, application writes log messages. */

void exercise_2(void) {
    printf("\n=== Exercise 2: FIFO-Based Logger ===\n");

    /*
     * This exercise requires two separate programs running in different
     * terminals. We demonstrate the concept in a single process using
     * fork() instead.
     */

    const char *fifo_path = "/tmp/ex22_app_log.fifo";
    const char *log_path = "/tmp/ex22_app.log";

    /* Remove old FIFO if exists */
    unlink(fifo_path);

    /* Create the named pipe (FIFO) */
    if (mkfifo(fifo_path, 0666) == -1) {
        perror("mkfifo");
        return;
    }

    pid_t pid = fork();
    if (pid == 0) {
        /* Child: Logger daemon */
        FILE *fifo_in = fopen(fifo_path, "r");
        if (!fifo_in) { perror("fopen fifo"); _exit(1); }

        FILE *log_out = fopen(log_path, "w");
        if (!log_out) { perror("fopen log"); fclose(fifo_in); _exit(1); }

        char line[256];
        while (fgets(line, sizeof(line), fifo_in)) {
            /* Prepend timestamp */
            time_t now = time(NULL);
            struct tm *tm = localtime(&now);
            char timestamp[16];
            strftime(timestamp, sizeof(timestamp), "[%H:%M:%S]", tm);

            fprintf(log_out, "%s %s", timestamp, line);
            fflush(log_out);
        }

        fprintf(log_out, "Log complete\n");
        printf("  Logger: Log complete\n");

        fclose(log_out);
        fclose(fifo_in);
        _exit(0);

    } else {
        /* Parent: Application (writes log messages) */
        /* Small delay so logger opens FIFO for reading first */
        usleep(100000); /* 100ms */

        FILE *fifo_out = fopen(fifo_path, "w");
        if (!fifo_out) {
            perror("fopen fifo for writing");
            waitpid(pid, NULL, 0);
            unlink(fifo_path);
            return;
        }

        const char *messages[] = {
            "Application started\n",
            "Processing request #1\n",
            "Database connection established\n",
            "Request #1 completed in 42ms\n",
            "Application shutting down\n",
        };

        printf("  Application sending 5 log messages...\n");
        for (int i = 0; i < 5; i++) {
            fprintf(fifo_out, "%s", messages[i]);
            fflush(fifo_out);
            usleep(100000); /* 100ms between messages */
        }

        fclose(fifo_out); /* Closing FIFO signals EOF to logger */

        /* Wait for logger to finish */
        waitpid(pid, NULL, 0);

        /* Display the log file */
        printf("\n  Contents of %s:\n", log_path);
        FILE *log = fopen(log_path, "r");
        if (log) {
            char line[256];
            while (fgets(line, sizeof(line), log)) {
                printf("    %s", line);
            }
            fclose(log);
        }

        /* Cleanup */
        unlink(fifo_path);
        unlink(log_path);
    }
}

/* === Exercise 3: Shared Counter with Semaphore Protection === */
/* Problem: Demonstrate race condition, then fix with POSIX named semaphore. */

void exercise_3(void) {
    printf("\n=== Exercise 3: Shared Counter with Semaphore ===\n");

    const char *shm_name = "/ex22_counter";
    const char *sem_name = "/ex22_sem";
    const int NUM_CHILDREN = 4;
    const int INCREMENTS = 10000;

    /* --- Part 1: WITHOUT synchronization (race condition) --- */
    printf("\n  Part 1: Without synchronization (race condition)\n");
    {
        int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
        if (fd == -1) { perror("shm_open"); return; }
        ftruncate(fd, sizeof(int));

        int *counter = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
        close(fd);
        *counter = 0;

        for (int c = 0; c < NUM_CHILDREN; c++) {
            pid_t pid = fork();
            if (pid == 0) {
                /* Child: increment counter without lock */
                for (int i = 0; i < INCREMENTS; i++) {
                    /*
                     * Race condition: read-modify-write is NOT atomic.
                     * Two processes can read the same value, both increment,
                     * and both write back -- losing one increment.
                     */
                    (*counter)++;
                }
                _exit(0);
            }
        }

        /* Wait for all children */
        for (int c = 0; c < NUM_CHILDREN; c++) wait(NULL);

        printf("  Expected: %d\n", NUM_CHILDREN * INCREMENTS);
        printf("  Actual:   %d (likely less due to race condition)\n", *counter);

        munmap(counter, sizeof(int));
        shm_unlink(shm_name);
    }

    /* --- Part 2: WITH semaphore synchronization --- */
    printf("\n  Part 2: With semaphore protection\n");
    {
        int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
        if (fd == -1) { perror("shm_open"); return; }
        ftruncate(fd, sizeof(int));

        int *counter = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
        close(fd);
        *counter = 0;

        /* Create a named semaphore initialized to 1 (binary semaphore / mutex) */
        sem_unlink(sem_name); /* Remove old semaphore if exists */
        sem_t *sem = sem_open(sem_name, O_CREAT | O_EXCL, 0666, 1);
        if (sem == SEM_FAILED) {
            perror("sem_open");
            munmap(counter, sizeof(int));
            shm_unlink(shm_name);
            return;
        }

        for (int c = 0; c < NUM_CHILDREN; c++) {
            pid_t pid = fork();
            if (pid == 0) {
                /* Child: open the named semaphore */
                sem_t *child_sem = sem_open(sem_name, 0);
                if (child_sem == SEM_FAILED) { perror("sem_open child"); _exit(1); }

                for (int i = 0; i < INCREMENTS; i++) {
                    sem_wait(child_sem);   /* Acquire lock */
                    (*counter)++;           /* Critical section */
                    sem_post(child_sem);   /* Release lock */
                }

                sem_close(child_sem);
                _exit(0);
            }
        }

        /* Wait for all children */
        for (int c = 0; c < NUM_CHILDREN; c++) wait(NULL);

        printf("  Expected: %d\n", NUM_CHILDREN * INCREMENTS);
        printf("  Actual:   %d (should be exact)\n", *counter);

        /* Cleanup all shared resources */
        sem_close(sem);
        sem_unlink(sem_name);
        munmap(counter, sizeof(int));
        shm_unlink(shm_name);
    }
}

/* === Exercise 4: Safe Signal Handler with Self-Pipe Trick === */
/* Problem: Use self-pipe trick for signal-safe SIGINT handling with select(). */

static int selfpipe[2]; /* Global for signal handler access */

static void sigint_handler(int sig) {
    (void)sig;
    /*
     * Signal handler safety:
     * - Only async-signal-safe functions are allowed here.
     * - write() is safe. printf(), malloc(), etc. are NOT.
     * - The self-pipe trick: write a byte to a pipe that the main
     *   loop monitors with select()/poll().
     *
     * Why printf() is unsafe in signal handlers:
     * - printf() uses internal locks and buffers.
     * - If the signal arrives while printf() is executing in the main code,
     *   calling printf() again in the handler causes deadlock (re-entrant lock)
     *   or corrupted output buffer.
     * - The self-pipe trick moves all complex logic OUT of the handler.
     */
    char byte = 'S';
    (void)write(selfpipe[1], &byte, 1); /* Async-signal-safe */
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Self-Pipe Trick for Signal Handling ===\n");

    if (pipe(selfpipe) == -1) {
        perror("pipe");
        return;
    }

    /* Make write end non-blocking (prevent handler from blocking) */
    int flags = fcntl(selfpipe[1], F_GETFL, 0);
    fcntl(selfpipe[1], F_SETFL, flags | O_NONBLOCK);

    /* Install SIGINT handler using sigaction (preferred over signal()) */
    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* No SA_RESTART: we want select() to be interrupted */
    sigaction(SIGINT, &sa, NULL);

    printf("  Self-pipe trick installed.\n");
    printf("  In a real application, the main loop would use select()/poll()\n");
    printf("  to wait on both stdin and selfpipe[0] simultaneously.\n\n");

    /*
     * Main loop skeleton (not running interactively):
     *
     *   while (running) {
     *       fd_set readfds;
     *       FD_ZERO(&readfds);
     *       FD_SET(STDIN_FILENO, &readfds);
     *       FD_SET(selfpipe[0], &readfds);
     *       int maxfd = (STDIN_FILENO > selfpipe[0]) ?
     *                    STDIN_FILENO : selfpipe[0];
     *
     *       int ready = select(maxfd + 1, &readfds, NULL, NULL, NULL);
     *       if (ready < 0 && errno == EINTR) continue;
     *
     *       if (FD_ISSET(selfpipe[0], &readfds)) {
     *           char byte;
     *           read(selfpipe[0], &byte, 1);
     *           printf("Graceful shutdown initiated\n");
     *           break;
     *       }
     *
     *       if (FD_ISSET(STDIN_FILENO, &readfds)) {
     *           char buf[256];
     *           ssize_t n = read(STDIN_FILENO, buf, sizeof(buf) - 1);
     *           if (n > 0) {
     *               buf[n] = '\0';
     *               printf("You typed: %s", buf);
     *           }
     *       }
     *   }
     */

    /* Demonstrate by sending ourselves SIGINT */
    printf("  Sending SIGINT to self to demonstrate...\n");
    kill(getpid(), SIGINT);

    /* Check if the self-pipe received the byte */
    char byte;
    ssize_t n = read(selfpipe[0], &byte, 1);
    if (n == 1 && byte == 'S') {
        printf("  Received signal notification via self-pipe (byte='%c')\n", byte);
        printf("  Graceful shutdown would be initiated here.\n");
    }

    /* Restore default SIGINT handler */
    signal(SIGINT, SIG_DFL);

    close(selfpipe[0]);
    close(selfpipe[1]);

    printf("\n  Why the self-pipe trick is better than using flags:\n");
    printf("  - Works with select()/poll() (multiplexed I/O)\n");
    printf("  - No busy-waiting to check a flag variable\n");
    printf("  - The pipe is a proper file descriptor, integrates cleanly\n");
    printf("  - Signal handler is minimal: just write() one byte\n");
}

/* === Exercise 5: POSIX Message Queue Priority Scheduler === */
/* Problem: Priority-based task scheduler using POSIX message queues. */

/*
 * Note: macOS does not support POSIX message queues (mq_open).
 * This exercise provides a simulation using a sorted array as fallback.
 */

typedef struct {
    char description[128];
    int priority; /* 1=low, 5=medium, 10=high */
} task_t;

/* Comparison for qsort: higher priority first */
static int compare_tasks(const void *a, const void *b) {
    const task_t *ta = (const task_t *)a;
    const task_t *tb = (const task_t *)b;
    return tb->priority - ta->priority; /* Descending order */
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Priority Task Scheduler ===\n");

    /*
     * POSIX Message Queue key concepts:
     * - mq_open(): Create/open a message queue
     * - mq_send(): Send a message with a priority
     * - mq_receive(): Always returns the highest-priority message first
     * - mq_unlink(): Remove the queue when done
     *
     * On Linux, the full POSIX MQ implementation:
     *
     *   struct mq_attr attr = {
     *       .mq_maxmsg = 10,
     *       .mq_msgsize = sizeof(task_t)
     *   };
     *   mqd_t mq = mq_open("/task_queue", O_CREAT | O_RDWR, 0666, &attr);
     *
     *   // Producer: send with priority
     *   mq_send(mq, (char *)&task, sizeof(task), task.priority);
     *
     *   // Consumer: receives highest priority first
     *   unsigned int prio;
     *   mq_receive(mq, (char *)&task, sizeof(task), &prio);
     *
     *   mq_close(mq);
     *   mq_unlink("/task_queue");
     */

    /* Simulation: use array + sort to demonstrate priority ordering */
    task_t tasks[] = {
        {"Clean up temp files",         1},   /* Low */
        {"Send email notification",     5},   /* Medium */
        {"Deploy hotfix to production", 10},  /* High */
        {"Update documentation",        1},   /* Low */
        {"Run integration tests",       5},   /* Medium */
        {"Fix critical security bug",   10},  /* High */
    };
    int num_tasks = (int)(sizeof(tasks) / sizeof(tasks[0]));

    /* Show tasks in submission order */
    printf("\n  Producer: Enqueuing %d tasks (random order):\n", num_tasks);
    for (int i = 0; i < num_tasks; i++) {
        printf("    [priority=%2d] %s\n", tasks[i].priority, tasks[i].description);
    }

    /* Sort by priority (simulates mq_receive always returning highest first) */
    qsort(tasks, (size_t)num_tasks, sizeof(task_t), compare_tasks);

    printf("\n  Consumer: Dequeuing tasks (highest priority first):\n");
    for (int i = 0; i < num_tasks; i++) {
        printf("    %d. [priority=%2d] %s\n",
               i + 1, tasks[i].priority, tasks[i].description);
    }

    printf("\n  Observation: Higher-priority tasks (10) are always processed\n");
    printf("  before medium (5) and low (1), regardless of submission order.\n");
    printf("  POSIX MQ guarantees this ordering via mq_receive().\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
