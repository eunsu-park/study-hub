// pipe.c
// Pipe implementation
// Execute multiple commands connected with |
// Compile: gcc -c pipe.c or link with other files

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_PIPES 10

// Count number of commands separated by pipes
int count_pipes(char** args) {
    int count = 0;
    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) count++;
    }
    return count;
}

// Split args at pipe positions
// commands[0] = first command's args
// commands[1] = second command's args
// ...
int split_by_pipe(char** args, char*** commands) {
    int cmd_count = 0;
    commands[cmd_count++] = args;

    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) {
            args[i] = NULL;  // Set pipe position to NULL
            if (args[i + 1]) {
                commands[cmd_count++] = &args[i + 1];
            }
        }
    }

    return cmd_count;
}

// Execute pipe
void execute_pipe(char** args) {
    char** commands[MAX_PIPES + 1];
    int cmd_count = split_by_pipe(args, commands);

    if (cmd_count == 1) {
        // No pipe: normal execution
        pid_t pid = fork();
        if (pid == 0) {
            execvp(commands[0][0], commands[0]);
            perror(commands[0][0]);
            exit(EXIT_FAILURE);
        } else if (pid > 0) {
            wait(NULL);
        } else {
            perror("fork");
        }
        return;
    }

    // Why: N commands need N-1 pipes — each pipe connects one command's stdout
    // to the next command's stdin, forming a data pipeline
    int pipes[MAX_PIPES][2];  // Pipe file descriptors

    // Create pipes
    for (int i = 0; i < cmd_count - 1; i++) {
        if (pipe(pipes[i]) < 0) {
            perror("pipe");
            return;
        }
    }

    // Execute each command
    for (int i = 0; i < cmd_count; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // Child process

            // Why: the first command reads from real stdin (no redirection needed),
            // and the last command writes to real stdout — only middle commands
            // need both ends redirected
            if (i > 0) {
                dup2(pipes[i - 1][0], STDIN_FILENO);
            }

            if (i < cmd_count - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            // Why: child must close ALL pipe fds after dup2 — the duplicated fds
            // already point to the right pipe ends, and keeping originals open
            // prevents EOF detection (readers block forever waiting for writers)
            for (int j = 0; j < cmd_count - 1; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            // Execute command
            execvp(commands[i][0], commands[i]);
            perror(commands[i][0]);
            exit(EXIT_FAILURE);

        } else if (pid < 0) {
            perror("fork");
            return;
        }
    }

    // Parent: close all pipes
    for (int i = 0; i < cmd_count - 1; i++) {
        close(pipes[i][0]);
        close(pipes[i][1]);
    }

    // Wait for all child processes
    for (int i = 0; i < cmd_count; i++) {
        wait(NULL);
    }
}

// Test main function
#ifdef TEST_PIPE
#define MAX_INPUT 1024
#define MAX_ARGS 64

int parse_args(char* input, char** args) {
    int argc = 0;
    char* token = strtok(input, " \t\n");
    while (token && argc < MAX_ARGS - 1) {
        args[argc++] = token;
        token = strtok(NULL, " \t\n");
    }
    args[argc] = NULL;
    return argc;
}

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    printf("=== Pipe Test ===\n");
    printf("Example commands:\n");
    printf("  ls -l | grep \".c\"\n");
    printf("  cat /etc/passwd | wc -l\n");
    printf("  ps aux | grep bash | head -5\n");
    printf("  ls | sort | uniq\n");
    printf("\nQuit: exit or Ctrl+D\n\n");

    while (1) {
        printf("pipe> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n");
            break;
        }

        if (input[0] == '\n') continue;

        // Copy input
        char input_copy[MAX_INPUT];
        strncpy(input_copy, input, sizeof(input_copy));

        int argc = parse_args(input_copy, args);
        if (argc == 0) continue;

        if (strcmp(args[0], "exit") == 0) {
            printf("Exiting.\n");
            break;
        }

        // Execute pipe
        execute_pipe(args);
    }

    return 0;
}
#endif
