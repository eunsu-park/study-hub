// redirect.c
// I/O redirection implementation
// Handles > (output), >> (append), < (input) operators
// Compile: gcc -c redirect.c or link with other files

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

typedef struct {
    char* input_file;   // < file
    char* output_file;  // > or >> file
    int append;         // 1 if >>
} Redirect;

// Parse redirection
// Remove redirection tokens from args and store in Redirect struct
void parse_redirect(char** args, Redirect* redir) {
    redir->input_file = NULL;
    redir->output_file = NULL;
    redir->append = 0;

    int i = 0;
    int j = 0;

    while (args[i] != NULL) {
        if (strcmp(args[i], "<") == 0) {
            // Input redirection
            if (args[i + 1]) {
                redir->input_file = args[i + 1];
                i += 2;
                continue;
            }
        } else if (strcmp(args[i], ">") == 0) {
            // Output redirection (overwrite)
            if (args[i + 1]) {
                redir->output_file = args[i + 1];
                redir->append = 0;
                i += 2;
                continue;
            }
        } else if (strcmp(args[i], ">>") == 0) {
            // Output redirection (append)
            if (args[i + 1]) {
                redir->output_file = args[i + 1];
                redir->append = 1;
                i += 2;
                continue;
            }
        }

        // Why: non-redirect args are compacted in-place (with separate i/j indices) so
        // the command array is clean for execvp — it must not see "<", ">", or filenames
        args[j++] = args[i++];
    }
    args[j] = NULL;
}

// Apply redirection (called in child process)
int apply_redirect(Redirect* redir) {
    // Input redirection
    if (redir->input_file) {
        int fd = open(redir->input_file, O_RDONLY);
        if (fd < 0) {
            perror(redir->input_file);
            return -1;
        }
        dup2(fd, STDIN_FILENO);
        close(fd);
    }

    // Output redirection
    // Why: O_TRUNC vs O_APPEND implements > vs >> — truncate discards existing
    // content, append preserves it; O_CREAT ensures the file is created if missing
    if (redir->output_file) {
        int flags = O_WRONLY | O_CREAT;
        flags |= redir->append ? O_APPEND : O_TRUNC;

        int fd = open(redir->output_file, flags, 0644);
        if (fd < 0) {
            perror(redir->output_file);
            return -1;
        }
        dup2(fd, STDOUT_FILENO);
        close(fd);
    }

    return 0;
}

// Execute command with redirection
// Why: redirections are applied in the child AFTER fork — this preserves the
// parent's original stdin/stdout so the shell can keep reading user input
void execute_with_redirect(char** args) {
    Redirect redir;
    parse_redirect(args, &redir);

    if (args[0] == NULL) return;

    pid_t pid = fork();

    if (pid == 0) {
        // Child: apply redirection then execute
        if (apply_redirect(&redir) < 0) {
            exit(EXIT_FAILURE);
        }
        execvp(args[0], args);
        perror(args[0]);
        exit(EXIT_FAILURE);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
    } else {
        perror("fork");
    }
}

// Test main function
#ifdef TEST_REDIRECT
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

    printf("=== Redirection Test ===\n");
    printf("Example commands:\n");
    printf("  ls -l > output.txt\n");
    printf("  cat < input.txt\n");
    printf("  echo \"Hello\" >> output.txt\n");
    printf("  wc -l < /etc/passwd\n");
    printf("\nQuit: exit or Ctrl+D\n\n");

    while (1) {
        printf("redirect> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n");
            break;
        }

        if (input[0] == '\n') continue;

        // Copy input (strtok modifies the original)
        char input_copy[MAX_INPUT];
        strncpy(input_copy, input, sizeof(input_copy));

        int argc = parse_args(input_copy, args);
        if (argc == 0) continue;

        if (strcmp(args[0], "exit") == 0) {
            printf("Exiting.\n");
            break;
        }

        // Execute with redirection
        execute_with_redirect(args);
    }

    return 0;
}
#endif
