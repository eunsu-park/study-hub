/**
 * Signal Handling Demo
 *
 * Demonstrates proper signal handling with sigaction:
 * - Graceful shutdown on SIGINT/SIGTERM
 * - Custom behavior on SIGUSR1/SIGUSR2
 * - Child process reaping with SIGCHLD
 *
 * Build: make
 * Usage: ./signal_handler
 *   Then send signals: kill -USR1 <pid>, Ctrl+C, etc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>

// Why: volatile prevents the compiler from caching "running" in a register —
// sig_atomic_t guarantees atomic reads/writes, so the signal handler and main
// loop see changes without locks or memory barriers
static volatile sig_atomic_t running = 1;
static volatile sig_atomic_t usr1_count = 0;
static volatile sig_atomic_t usr2_count = 0;

/* SIGINT / SIGTERM handler: graceful shutdown */
static void handle_shutdown(int sig) {
    const char *name = (sig == SIGINT) ? "SIGINT" : "SIGTERM";
    // Why: write() is async-signal-safe but printf() is NOT — printf uses internal
    // locks/buffers that may be in an inconsistent state when the signal fires,
    // causing deadlocks or corruption
    write(STDOUT_FILENO, "\nReceived ", 10);
    write(STDOUT_FILENO, name, strlen(name));
    write(STDOUT_FILENO, ", shutting down...\n", 18);
    running = 0;
}

/* SIGUSR1 handler */
static void handle_usr1(int sig) {
    (void)sig;
    usr1_count++;
}

/* SIGUSR2 handler with siginfo */
static void handle_usr2(int sig, siginfo_t *info, void *context) {
    (void)sig;
    (void)context;
    usr2_count++;
    /* We can safely use write() here */
    char buf[128];
    int n = snprintf(buf, sizeof(buf),
                     "SIGUSR2 from PID %d (uid=%d)\n",
                     info->si_pid, info->si_uid);
    write(STDOUT_FILENO, buf, n);
}

/* SIGCHLD handler: reap children */
static void handle_sigchld(int sig) {
    (void)sig;
    int status;
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        char buf[64];
        int n = snprintf(buf, sizeof(buf),
                         "Child %d exited (status=%d)\n",
                         pid, WEXITSTATUS(status));
        write(STDOUT_FILENO, buf, n);
    }
}

/* Install a signal handler */
static void install_handler(int sig, void (*handler)(int), int flags) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = flags;
    if (sigaction(sig, &sa, NULL) < 0) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    printf("Signal Handler Demo\n");
    printf("PID: %d\n\n", getpid());
    printf("Available signals:\n");
    printf("  Ctrl+C     → SIGINT  (graceful shutdown)\n");
    printf("  kill -USR1 → SIGUSR1 (counter increment)\n");
    printf("  kill -USR2 → SIGUSR2 (with sender info)\n");
    printf("  kill -TERM → SIGTERM (graceful shutdown)\n\n");

    /* Install handlers */
    install_handler(SIGINT,  handle_shutdown, 0);
    install_handler(SIGTERM, handle_shutdown, 0);
    // Why: SA_RESTART makes interrupted system calls (read, write, sleep) resume
    // automatically after the signal — without it, they would return -1 with
    // errno=EINTR, requiring manual retry logic throughout the code
    install_handler(SIGUSR1, handle_usr1, SA_RESTART);
    install_handler(SIGCHLD, handle_sigchld, SA_RESTART | SA_NOCLDSTOP);

    /* SIGUSR2 with SA_SIGINFO for sender information */
    struct sigaction sa_usr2;
    memset(&sa_usr2, 0, sizeof(sa_usr2));
    sa_usr2.sa_sigaction = handle_usr2;
    sigemptyset(&sa_usr2.sa_mask);
    sa_usr2.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGUSR2, &sa_usr2, NULL);

    // Why: SIGPIPE's default action kills the process — ignoring it lets us handle
    // broken-pipe errors via write()'s return value instead of crashing
    signal(SIGPIPE, SIG_IGN);

    /* Fork a child to demonstrate SIGCHLD */
    pid_t child = fork();
    if (child == 0) {
        printf("[Child %d] Running for 3 seconds...\n", getpid());
        sleep(3);
        printf("[Child %d] Exiting\n", getpid());
        exit(42);
    }
    printf("Forked child PID=%d\n\n", child);

    /* Main loop */
    int iteration = 0;
    while (running) {
        printf("Tick %d (USR1=%d, USR2=%d)\n",
               ++iteration, (int)usr1_count, (int)usr2_count);
        sleep(2);
    }

    printf("\nClean shutdown. Final counts: USR1=%d, USR2=%d\n",
           (int)usr1_count, (int)usr2_count);
    return 0;
}
