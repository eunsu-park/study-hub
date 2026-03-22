// minishell_v1.c
// Basic shell structure: read -> parse -> execute -> repeat
// Compile: gcc -o minishell_v1 minishell_v1.c -Wall
// Run: ./minishell_v1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_INPUT 1024
#define MAX_ARGS 64

// Split input by whitespace
int parse_input(char* input, char** args) {
    int argc = 0;
    char* token = strtok(input, " \t\n");

    while (token != NULL && argc < MAX_ARGS - 1) {
        args[argc++] = token;
        token = strtok(NULL, " \t\n");
    }
    args[argc] = NULL;

    return argc;
}

// Execute command
void execute(char** args) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
        return;
    }

    if (pid == 0) {
        // Child process: execute command
        execvp(args[0], args);
        // If execvp fails
        perror(args[0]);
        exit(EXIT_FAILURE);
    } else {
        // Parent process: wait for child to finish
        int status;
        waitpid(pid, &status, 0);
    }
}

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    printf("\n=== Mini Shell v1 ===\n");
    printf("Basic command execution shell\n");
    printf("Quit: exit command or Ctrl+D\n\n");

    while (1) {
        // Print prompt
        printf("minish> ");
        fflush(stdout);

        // Read input
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n");
            break;  // EOF (Ctrl+D)
        }

        // Ignore empty input
        if (input[0] == '\n') continue;

        // Parse
        int argc = parse_input(input, args);
        if (argc == 0) continue;

        // exit command
        if (strcmp(args[0], "exit") == 0) {
            printf("Exiting the shell.\n");
            break;
        }

        // Execute
        execute(args);
    }

    return 0;
}
