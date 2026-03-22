/*
 * process_demo.c
 *
 * Demonstrates POSIX process management: fork(), exec(), wait(),
 * and basic signal handling.
 *
 * Build:  gcc -Wall -Wextra -std=c11 -o process_demo process_demo.c
 * Run:    ./process_demo
 *
 * Note: POSIX-only (Linux / macOS).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

/* ── Helper: run a command in a child process via execvp ──────── */
static int run_command(const char *desc, char *const argv[])
{
    printf("\n--- %s ---\n", desc);
    fflush(stdout);

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        /* Child process */
        execvp(argv[0], argv);
        /* execvp only returns on error */
        fprintf(stderr, "execvp(%s): %s\n", argv[0], strerror(errno));
        _exit(127);
    }

    /* Parent process — wait for child */
    int status;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        printf("[parent] child %d exited with code %d\n", pid, code);
        return code;
    } else if (WIFSIGNALED(status)) {
        printf("[parent] child %d killed by signal %d\n", pid, WTERMSIG(status));
        return -1;
    }
    return -1;
}

int main(void)
{
    printf("=== Process Management Demo ===\n");
    printf("Parent PID: %d\n", getpid());

    /* 1. Simple fork — child prints and exits */
    printf("\n--- fork demo ---\n");
    fflush(stdout);

    pid_t pid = fork();
    if (pid == 0) {
        printf("[child  PID %d] Hello from child! Parent is %d\n",
               getpid(), getppid());
        _exit(0);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
        printf("[parent PID %d] Child %d finished\n", getpid(), pid);
    }

    /* 2. exec: run "echo" via execvp */
    char *echo_argv[] = {"echo", "Hello from exec'd echo!", NULL};
    run_command("exec demo (echo)", echo_argv);

    /* 3. exec: run "ls -l /tmp" */
    char *ls_argv[] = {"ls", "-l", "/tmp", NULL};
    run_command("exec demo (ls -l /tmp)", ls_argv);

    /* 4. Multiple children — fan-out and collect */
    printf("\n--- multiple children ---\n");
    fflush(stdout);

    enum { N_CHILDREN = 3 };
    pid_t children[N_CHILDREN];

    for (int i = 0; i < N_CHILDREN; i++) {
        children[i] = fork();
        if (children[i] == 0) {
            printf("[child %d] PID %d doing work...\n", i, getpid());
            usleep(100000 * (i + 1));  /* stagger finish times */
            printf("[child %d] done\n", i);
            _exit(i);
        }
    }

    /* Collect all children */
    for (int i = 0; i < N_CHILDREN; i++) {
        int status;
        pid_t w = waitpid(children[i], &status, 0);
        if (WIFEXITED(status))
            printf("[parent] child %d (PID %d) exited with %d\n",
                   i, w, WEXITSTATUS(status));
    }

    printf("\nAll children collected. Parent exiting.\n");
    return 0;
}
