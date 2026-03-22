/*
 * Exercises for Lesson 09: Process Management
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex09 09_process_management.c
 * Note: POSIX-specific (Linux/macOS). Not portable to Windows.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <errno.h>

/* === Exercise 1: Fork/Exec Wrapper === */
/* Problem: Create a safe wrapper around fork+exec that handles errors
 *          and returns the child's exit status. */

int run_command(const char *path, char *const argv[]) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        /* Child process */
        execvp(path, argv);
        /* If execvp returns, it failed */
        perror("execvp");
        _exit(127);  /* Use _exit, not exit, in child after fork */
    }

    /* Parent process: wait for child */
    int status;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        printf("Child killed by signal %d\n", WTERMSIG(status));
        return -1;
    }
    return -1;
}

void exercise_1(void) {
    printf("=== Exercise 1: Fork/Exec Wrapper ===\n");

    /* Run 'echo' command */
    printf("Running 'echo Hello from child process':\n");
    char *argv1[] = {"echo", "Hello from child process", NULL};
    int ret = run_command("echo", argv1);
    printf("Exit status: %d\n", ret);

    /* Run 'ls' with flags */
    printf("\nRunning 'ls -la /tmp' (first 5 entries):\n");
    char *argv2[] = {"ls", "-la", "/tmp", NULL};
    ret = run_command("ls", argv2);
    printf("Exit status: %d\n", ret);

    /* Run a command that doesn't exist */
    printf("\nRunning nonexistent command:\n");
    char *argv3[] = {"nonexistent_command_xyz", NULL};
    ret = run_command("nonexistent_command_xyz", argv3);
    printf("Exit status: %d\n", ret);

    /*
     * Key points:
     * - Always use _exit() in child after fork (not exit()) to avoid
     *   flushing stdio buffers twice
     * - execvp searches PATH; execv requires full path
     * - Check WIFEXITED and WIFSIGNALED to interpret wait status
     * - Exit code 127 conventionally means "command not found"
     */
}

/* === Exercise 2: Process Tree Printer === */
/* Problem: Fork multiple children and print the process tree. */

void exercise_2(void) {
    printf("\n=== Exercise 2: Process Tree Printer ===\n");

    pid_t parent = getpid();
    printf("Parent PID: %d\n", parent);

    #define NUM_CHILDREN 3
    pid_t children[NUM_CHILDREN];

    for (int i = 0; i < NUM_CHILDREN; i++) {
        pid_t pid = fork();

        if (pid < 0) {
            perror("fork");
            break;
        }

        if (pid == 0) {
            /* Child process */
            printf("  Child %d: PID=%d, PPID=%d\n", i, getpid(), getppid());

            /* Each child creates one grandchild */
            pid_t grandchild = fork();
            if (grandchild == 0) {
                printf("    Grandchild of child %d: PID=%d, PPID=%d\n",
                       i, getpid(), getppid());
                _exit(0);
            } else if (grandchild > 0) {
                waitpid(grandchild, NULL, 0);
            }
            _exit(0);
        }

        children[i] = pid;
    }

    /* Parent waits for all children */
    for (int i = 0; i < NUM_CHILDREN; i++) {
        int status;
        waitpid(children[i], &status, 0);
        printf("Parent: child %d (PID=%d) exited with status %d\n",
               i, children[i], WEXITSTATUS(status));
    }

    printf("\nProcess tree summary:\n");
    printf("  Parent (%d)\n", parent);
    for (int i = 0; i < NUM_CHILDREN; i++) {
        printf("  +-- Child %d (%d)\n", i, children[i]);
        printf("      +-- Grandchild\n");
    }
}

/* === Exercise 3: Zombie Prevention === */
/* Problem: Demonstrate zombie processes and techniques to prevent them. */

/* SIGCHLD handler for automatic reaping */
volatile sig_atomic_t child_reaped = 0;

void sigchld_handler(int sig) {
    (void)sig;
    /* Reap all finished children (non-blocking) */
    while (waitpid(-1, NULL, WNOHANG) > 0) {
        child_reaped++;
    }
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Zombie Prevention ===\n");

    /*
     * Method 1: Explicit wait (already shown above)
     * Parent calls waitpid() for each child.
     */
    printf("\nMethod 1: Explicit waitpid()\n");
    pid_t pid = fork();
    if (pid == 0) {
        printf("  Child (PID=%d) running\n", getpid());
        _exit(42);
    }
    int status;
    waitpid(pid, &status, 0);
    printf("  Parent reaped child, exit code: %d\n", WEXITSTATUS(status));

    /*
     * Method 2: SIGCHLD handler for automatic reaping
     * Useful when parent is busy and can't block on waitpid.
     */
    printf("\nMethod 2: SIGCHLD handler\n");

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigchld_handler;
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    child_reaped = 0;
    for (int i = 0; i < 3; i++) {
        pid = fork();
        if (pid == 0) {
            printf("  Child %d (PID=%d) exiting\n", i, getpid());
            _exit(0);
        }
    }
    /* Give children time to exit and be reaped by handler */
    usleep(100000);  /* 100ms */
    printf("  Children reaped by SIGCHLD handler: %d\n", child_reaped);

    /*
     * Method 3: Double fork (daemon pattern)
     * The grandchild is orphaned and adopted by init/systemd,
     * so the original parent doesn't need to wait.
     */
    printf("\nMethod 3: Double fork (grandchild becomes orphan)\n");
    pid = fork();
    if (pid == 0) {
        /* First child */
        pid_t grandchild = fork();
        if (grandchild == 0) {
            /* Grandchild: adopted by init when first child exits */
            printf("  Grandchild PID=%d running independently\n", getpid());
            _exit(0);
        }
        /* First child exits immediately */
        _exit(0);
    }
    waitpid(pid, NULL, 0);  /* Reap first child */
    usleep(50000);  /* Let grandchild run */
    printf("  First child reaped; grandchild handled by init\n");

    /* Restore default SIGCHLD */
    sa.sa_handler = SIG_DFL;
    sigaction(SIGCHLD, &sa, NULL);

    /*
     * Summary of zombie prevention:
     * 1. waitpid() — simple, synchronous
     * 2. SIGCHLD handler — async, good for servers
     * 3. Double fork — for daemon processes
     * 4. SA_NOCLDWAIT flag or SIG_IGN for SIGCHLD (Linux-specific)
     */
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
