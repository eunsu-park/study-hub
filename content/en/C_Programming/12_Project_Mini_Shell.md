# Project 11: Mini Shell

**Previous**: [Project 10: Terminal Snake Game](./11_Project_Snake_Game.md) | **Next**: [Project 12: Multithreaded Programming](./13_Project_Multithreading.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement the Read-Parse-Execute loop that forms the core of any command-line shell
2. Apply `fork` to create child processes and `execvp` to replace them with external programs
3. Implement built-in commands (`cd`, `pwd`, `echo`, `export`) that must execute in the shell's own process
4. Design a redirection parser that detects `<`, `>`, and `>>` tokens and applies `dup2` to reroute file descriptors
5. Build a pipe executor that chains multiple commands with `pipe`, `dup2`, and coordinated `fork`/`wait` calls
6. Configure a `SIGINT` handler so that Ctrl+C interrupts the foreground command without killing the shell
7. Implement environment-variable expansion (`$VAR`) and wildcard globbing (`*`, `?`) in command arguments

---

A shell is the program you use every day to launch other programs, yet most developers never look inside one. Building a mini shell from scratch is one of the most rewarding systems-programming exercises because it ties together processes, file descriptors, pipes, signals, and environment variables -- the fundamental building blocks of Unix. By the end of this project you will understand what happens between the moment you press Enter and the moment your command's output appears on screen.

## Prerequisites
- String processing
- File I/O
- Pointers and dynamic memory

---

## Stage 1: Basic Shell Structure

The basic shell operation: **Read → Parse → Execute → Repeat**

### Simplest Shell

```c
// minishell_v1.c
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
            printf("Exiting shell.\n");
            break;
        }

        // Execute
        execute(args);
    }

    return 0;
}
```

### Compile and Test

```bash
gcc -o minish minishell_v1.c
./minish

minish> ls -l
minish> pwd
minish> echo hello world
minish> exit
```

---

## Stage 2: Built-in Commands

Some commands must be handled by the shell itself, not external programs.

### Implementing Built-in Commands

```c
// builtins.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Built-in command names
const char* builtin_names[] = {
    "cd",
    "pwd",
    "echo",
    "exit",
    "help",
    "export",
    "env",
    NULL
};

// cd: Change directory
int builtin_cd(char** args) {
    const char* path;

    if (args[1] == NULL) {
        // No argument: go to home directory
        path = getenv("HOME");
        if (path == NULL) {
            fprintf(stderr, "cd: HOME environment variable not set\n");
            return 1;
        }
    } else if (strcmp(args[1], "-") == 0) {
        // cd -: go to previous directory
        path = getenv("OLDPWD");
        if (path == NULL) {
            fprintf(stderr, "cd: OLDPWD environment variable not set\n");
            return 1;
        }
        printf("%s\n", path);
    } else if (strcmp(args[1], "~") == 0) {
        path = getenv("HOME");
    } else {
        path = args[1];
    }

    // Save current directory
    char oldpwd[1024];
    getcwd(oldpwd, sizeof(oldpwd));

    if (chdir(path) != 0) {
        perror("cd");
        return 1;
    }

    // Update OLDPWD, PWD environment variables
    setenv("OLDPWD", oldpwd, 1);

    char newpwd[1024];
    getcwd(newpwd, sizeof(newpwd));
    setenv("PWD", newpwd, 1);

    return 0;
}

// pwd: Print current directory
int builtin_pwd(char** args) {
    (void)args;  // Unused

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("%s\n", cwd);
        return 0;
    }
    perror("pwd");
    return 1;
}

// echo: Print arguments
int builtin_echo(char** args) {
    int newline = 1;
    int start = 1;

    // -n option: print without newline
    if (args[1] && strcmp(args[1], "-n") == 0) {
        newline = 0;
        start = 2;
    }

    for (int i = start; args[i]; i++) {
        printf("%s", args[i]);
        if (args[i + 1]) printf(" ");
    }

    if (newline) printf("\n");
    return 0;
}

// help: Display help
int builtin_help(char** args) {
    (void)args;

    printf("\n=== Mini Shell Help ===\n\n");
    printf("Built-in commands:\n");
    printf("  cd [directory]  - Change directory\n");
    printf("  pwd             - Print current directory\n");
    printf("  echo [text]     - Print text\n");
    printf("  export VAR=val  - Set environment variable\n");
    printf("  env             - List environment variables\n");
    printf("  help            - Display this help\n");
    printf("  exit            - Exit shell\n");
    printf("\nExternal commands are searched in PATH.\n\n");

    return 0;
}

// export: Set environment variable
int builtin_export(char** args) {
    if (args[1] == NULL) {
        // No arguments: print environment variables
        extern char** environ;
        for (char** env = environ; *env; env++) {
            printf("export %s\n", *env);
        }
        return 0;
    }

    // Parse VAR=value format
    for (int i = 1; args[i]; i++) {
        char* eq = strchr(args[i], '=');
        if (eq) {
            *eq = '\0';
            setenv(args[i], eq + 1, 1);
            *eq = '=';
        } else {
            // No =: set empty value
            setenv(args[i], "", 1);
        }
    }

    return 0;
}

// env: Print environment variables
int builtin_env(char** args) {
    (void)args;

    extern char** environ;
    for (char** env = environ; *env; env++) {
        printf("%s\n", *env);
    }
    return 0;
}

// Check if built-in command and execute
// Return: -1 (not built-in), 0+ (execution result)
int execute_builtin(char** args) {
    if (args[0] == NULL) return -1;

    if (strcmp(args[0], "cd") == 0) return builtin_cd(args);
    if (strcmp(args[0], "pwd") == 0) return builtin_pwd(args);
    if (strcmp(args[0], "echo") == 0) return builtin_echo(args);
    if (strcmp(args[0], "help") == 0) return builtin_help(args);
    if (strcmp(args[0], "export") == 0) return builtin_export(args);
    if (strcmp(args[0], "env") == 0) return builtin_env(args);

    return -1;  // Not built-in
}
```

---

## Stage 3: Implementing Redirection

Handle `>`, `>>`, `<` operators.

### Redirection Parser

```c
// redirect.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    char* input_file;   // < file
    char* output_file;  // > or >> file
    int append;         // 1 if >>
} Redirect;

// Parse redirection
// Remove redirection from args and store in Redirect struct
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

        // Not a redirection argument
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
```

### Using Redirection

```c
// execute with redirection
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
```

Test:
```bash
minish> ls -l > files.txt
minish> cat < files.txt
minish> echo "additional content" >> files.txt
minish> wc -l < files.txt
```

---

## Stage 4: Implementing Pipes

Connect commands with `|` operator.

### Pipe Processing

```c
// pipe.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_PIPES 10

// Count pipe-separated commands
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
            args[i] = NULL;  // Replace pipe with NULL
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
        execute_with_redirect(commands[0]);
        return;
    }

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

            // Connect previous pipe's read end to stdin
            if (i > 0) {
                dup2(pipes[i - 1][0], STDIN_FILENO);
            }

            // Connect next pipe's write end to stdout
            if (i < cmd_count - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            // Close all pipes
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
```

Test:
```bash
minish> ls -l | grep ".c"
minish> cat file.txt | wc -l
minish> ps aux | grep bash | head -5
```

---

## Stage 5: Complete Mini Shell

### Full Code

```c
// minishell.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

#define MAX_INPUT 1024
#define MAX_ARGS 64
#define MAX_PIPES 10

// ============ Global Variables ============
static int last_exit_status = 0;

// ============ Signal Handler ============
void sigint_handler(int sig) {
    (void)sig;
    printf("\n");
    // Don't print prompt again (handled in main loop)
}

// ============ Utilities ============

// Trim whitespace from both ends
char* trim(char* str) {
    while (*str == ' ' || *str == '\t') str++;

    if (*str == '\0') return str;

    char* end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\n')) {
        *end-- = '\0';
    }

    return str;
}

// ============ Parsing ============

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

// ============ Redirection ============

typedef struct {
    char* infile;
    char* outfile;
    int append;
} Redirect;

void parse_redirect(char** args, Redirect* r) {
    r->infile = NULL;
    r->outfile = NULL;
    r->append = 0;

    int i = 0, j = 0;
    while (args[i]) {
        if (strcmp(args[i], "<") == 0 && args[i+1]) {
            r->infile = args[i+1];
            i += 2;
        } else if (strcmp(args[i], ">") == 0 && args[i+1]) {
            r->outfile = args[i+1];
            r->append = 0;
            i += 2;
        } else if (strcmp(args[i], ">>") == 0 && args[i+1]) {
            r->outfile = args[i+1];
            r->append = 1;
            i += 2;
        } else {
            args[j++] = args[i++];
        }
    }
    args[j] = NULL;
}

int setup_redirect(Redirect* r) {
    if (r->infile) {
        int fd = open(r->infile, O_RDONLY);
        if (fd < 0) { perror(r->infile); return -1; }
        dup2(fd, STDIN_FILENO);
        close(fd);
    }
    if (r->outfile) {
        int flags = O_WRONLY | O_CREAT | (r->append ? O_APPEND : O_TRUNC);
        int fd = open(r->outfile, flags, 0644);
        if (fd < 0) { perror(r->outfile); return -1; }
        dup2(fd, STDOUT_FILENO);
        close(fd);
    }
    return 0;
}

// ============ Built-in Commands ============

int builtin_cd(char** args) {
    const char* path = args[1] ? args[1] : getenv("HOME");

    if (strcmp(path, "-") == 0) {
        path = getenv("OLDPWD");
        if (!path) {
            fprintf(stderr, "cd: OLDPWD not set\n");
            return 1;
        }
        printf("%s\n", path);
    } else if (strcmp(path, "~") == 0) {
        path = getenv("HOME");
    }

    char oldpwd[1024];
    getcwd(oldpwd, sizeof(oldpwd));

    if (chdir(path) != 0) {
        perror("cd");
        return 1;
    }

    setenv("OLDPWD", oldpwd, 1);
    char newpwd[1024];
    getcwd(newpwd, sizeof(newpwd));
    setenv("PWD", newpwd, 1);

    return 0;
}

int builtin_pwd(void) {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) {
        printf("%s\n", cwd);
        return 0;
    }
    perror("pwd");
    return 1;
}

int builtin_echo(char** args) {
    int newline = 1, start = 1;
    if (args[1] && strcmp(args[1], "-n") == 0) {
        newline = 0;
        start = 2;
    }

    for (int i = start; args[i]; i++) {
        // Environment variable expansion ($VAR)
        if (args[i][0] == '$') {
            char* val = getenv(args[i] + 1);
            printf("%s", val ? val : "");
        } else {
            printf("%s", args[i]);
        }
        if (args[i + 1]) printf(" ");
    }
    if (newline) printf("\n");
    return 0;
}

int builtin_export(char** args) {
    if (!args[1]) {
        extern char** environ;
        for (char** e = environ; *e; e++) {
            printf("export %s\n", *e);
        }
        return 0;
    }

    for (int i = 1; args[i]; i++) {
        char* eq = strchr(args[i], '=');
        if (eq) {
            *eq = '\0';
            setenv(args[i], eq + 1, 1);
        }
    }
    return 0;
}

int builtin_unset(char** args) {
    for (int i = 1; args[i]; i++) {
        unsetenv(args[i]);
    }
    return 0;
}

int builtin_help(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════╗\n");
    printf("║        Mini Shell Help                ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ Built-in commands:                    ║\n");
    printf("║   cd [dir]    Change directory        ║\n");
    printf("║   pwd         Print current directory ║\n");
    printf("║   echo [...]  Print text              ║\n");
    printf("║   export V=X  Set environment var     ║\n");
    printf("║   unset VAR   Unset environment var   ║\n");
    printf("║   help        Display this help       ║\n");
    printf("║   exit [N]    Exit shell              ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ Redirection:                          ║\n");
    printf("║   cmd > file  Redirect output to file║\n");
    printf("║   cmd >> file Append output to file  ║\n");
    printf("║   cmd < file  Read input from file   ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ Pipes:                                ║\n");
    printf("║   cmd1 | cmd2 Pipe output to next cmd║\n");
    printf("╚═══════════════════════════════════════╝\n");
    printf("\n");
    return 0;
}

// Execute built-in command (-1: not built-in)
int run_builtin(char** args) {
    if (!args[0]) return -1;

    if (strcmp(args[0], "cd") == 0) return builtin_cd(args);
    if (strcmp(args[0], "pwd") == 0) return builtin_pwd();
    if (strcmp(args[0], "echo") == 0) return builtin_echo(args);
    if (strcmp(args[0], "export") == 0) return builtin_export(args);
    if (strcmp(args[0], "unset") == 0) return builtin_unset(args);
    if (strcmp(args[0], "help") == 0) return builtin_help();

    return -1;
}

// ============ Pipe Execution ============

int split_pipe(char** args, char*** cmds) {
    int n = 0;
    cmds[n++] = args;

    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) {
            args[i] = NULL;
            if (args[i + 1]) {
                cmds[n++] = &args[i + 1];
            }
        }
    }
    return n;
}

void run_pipeline(char** args) {
    char** cmds[MAX_PIPES + 1];
    int n = split_pipe(args, cmds);

    // No pipe: single command execution
    if (n == 1) {
        Redirect r;
        parse_redirect(cmds[0], &r);

        if (!cmds[0][0]) return;

        // Check built-in
        int builtin_result = run_builtin(cmds[0]);
        if (builtin_result != -1) {
            last_exit_status = builtin_result;
            return;
        }

        // External command
        pid_t pid = fork();
        if (pid == 0) {
            setup_redirect(&r);
            execvp(cmds[0][0], cmds[0]);
            fprintf(stderr, "%s: command not found\n", cmds[0][0]);
            exit(127);
        } else if (pid > 0) {
            int status;
            waitpid(pid, &status, 0);
            last_exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
        }
        return;
    }

    // With pipes
    int pipes[MAX_PIPES][2];
    for (int i = 0; i < n - 1; i++) {
        pipe(pipes[i]);
    }

    for (int i = 0; i < n; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // Connect input
            if (i > 0) {
                dup2(pipes[i-1][0], STDIN_FILENO);
            }
            // Connect output
            if (i < n - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            // Close all pipes
            for (int j = 0; j < n - 1; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            // Handle redirection (only for first/last command)
            Redirect r;
            parse_redirect(cmds[i], &r);
            if (i == 0 && r.infile) {
                int fd = open(r.infile, O_RDONLY);
                if (fd >= 0) { dup2(fd, STDIN_FILENO); close(fd); }
            }
            if (i == n - 1 && r.outfile) {
                int flags = O_WRONLY | O_CREAT | (r.append ? O_APPEND : O_TRUNC);
                int fd = open(r.outfile, flags, 0644);
                if (fd >= 0) { dup2(fd, STDOUT_FILENO); close(fd); }
            }

            execvp(cmds[i][0], cmds[i]);
            fprintf(stderr, "%s: command not found\n", cmds[i][0]);
            exit(127);
        }
    }

    // Parent: close pipes and wait
    for (int i = 0; i < n - 1; i++) {
        close(pipes[i][0]);
        close(pipes[i][1]);
    }

    int status;
    for (int i = 0; i < n; i++) {
        wait(&status);
    }
    last_exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

// ============ Prompt ============

void print_prompt(void) {
    char cwd[256];
    char* dir = getcwd(cwd, sizeof(cwd));

    // Display home directory as ~
    char* home = getenv("HOME");
    if (home && dir && strncmp(dir, home, strlen(home)) == 0) {
        printf("\033[1;34m~%s\033[0m", dir + strlen(home));
    } else {
        printf("\033[1;34m%s\033[0m", dir ? dir : "?");
    }

    // Change color based on exit code
    if (last_exit_status == 0) {
        printf(" \033[1;32m❯\033[0m ");
    } else {
        printf(" \033[1;31m❯\033[0m ");
    }

    fflush(stdout);
}

// ============ Main ============

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    // Set signal handler
    signal(SIGINT, sigint_handler);

    printf("\n\033[1;36m=== Mini Shell ===\033[0m\n");
    printf("Type 'help' for help\n\n");

    while (1) {
        print_prompt();

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\nexit\n");
            break;
        }

        char* trimmed = trim(input);
        if (*trimmed == '\0') continue;

        // Ignore comments
        if (trimmed[0] == '#') continue;

        // Copy input (strtok modifies original)
        char input_copy[MAX_INPUT];
        strncpy(input_copy, trimmed, sizeof(input_copy));

        // Parse
        int argc = parse_args(input_copy, args);
        if (argc == 0) continue;

        // exit command
        if (strcmp(args[0], "exit") == 0) {
            int code = args[1] ? atoi(args[1]) : last_exit_status;
            printf("exit\n");
            exit(code);
        }

        // Execute
        run_pipeline(args);
    }

    return last_exit_status;
}
```

### Compile and Run

```bash
gcc -o minishell minishell.c -Wall -Wextra
./minishell
```

### Test Examples

```bash
=== Mini Shell ===
Type 'help' for help

~ ❯ help
~ ❯ pwd
/Users/username
~ ❯ cd /tmp
/tmp ❯ ls -la
/tmp ❯ echo $HOME
/Users/username
/tmp ❯ export MY_VAR=hello
/tmp ❯ echo $MY_VAR
hello
/tmp ❯ ls -l | grep ".txt" | wc -l
/tmp ❯ cat /etc/passwd | head -5 > first5.txt
/tmp ❯ cat first5.txt
/tmp ❯ cd -
/Users/username
~ ❯ exit
```

---

## Stage 6: Additional Features

### History Feature

```c
#define HISTORY_SIZE 100

static char* history[HISTORY_SIZE];
static int history_count = 0;

void add_history(const char* cmd) {
    if (history_count < HISTORY_SIZE) {
        history[history_count++] = strdup(cmd);
    } else {
        // Remove oldest
        free(history[0]);
        memmove(history, history + 1, (HISTORY_SIZE - 1) * sizeof(char*));
        history[HISTORY_SIZE - 1] = strdup(cmd);
    }
}

int builtin_history(char** args) {
    int n = history_count;
    if (args[1]) {
        n = atoi(args[1]);
        if (n > history_count) n = history_count;
    }

    int start = history_count - n;
    for (int i = start; i < history_count; i++) {
        printf("%5d  %s\n", i + 1, history[i]);
    }
    return 0;
}

void free_history(void) {
    for (int i = 0; i < history_count; i++) {
        free(history[i]);
    }
}
```

### Background Execution (&)

```c
// Check for &
int is_background(char** args) {
    int i = 0;
    while (args[i]) i++;

    if (i > 0 && strcmp(args[i - 1], "&") == 0) {
        args[i - 1] = NULL;  // Remove &
        return 1;
    }
    return 0;
}

// Modified execution function
void run_command(char** args) {
    int bg = is_background(args);

    pid_t pid = fork();

    if (pid == 0) {
        // Child
        execvp(args[0], args);
        perror(args[0]);
        exit(127);
    } else if (pid > 0) {
        if (bg) {
            printf("[%d] %d\n", 1, pid);
            // Background: don't wait
        } else {
            int status;
            waitpid(pid, &status, 0);
        }
    }
}
```

### Wildcard Expansion (*)

```c
#include <glob.h>

// Wildcard expansion
int expand_wildcards(char** args, char** expanded, int max_expanded) {
    int count = 0;

    for (int i = 0; args[i] && count < max_expanded - 1; i++) {
        // Arguments containing * or ?
        if (strchr(args[i], '*') || strchr(args[i], '?')) {
            glob_t results;
            int ret = glob(args[i], GLOB_NOCHECK | GLOB_TILDE, NULL, &results);

            if (ret == 0) {
                for (size_t j = 0; j < results.gl_pathc && count < max_expanded - 1; j++) {
                    expanded[count++] = strdup(results.gl_pathv[j]);
                }
            }
            globfree(&results);
        } else {
            expanded[count++] = args[i];
        }
    }

    expanded[count] = NULL;
    return count;
}
```

---

## Exercises

### Exercise 1: Semicolon Support
Implement sequential execution of multiple commands with `cmd1 ; cmd2` format.

### Exercise 2: && and || Operators
- `cmd1 && cmd2`: Execute cmd2 only if cmd1 succeeds
- `cmd1 || cmd2`: Execute cmd2 only if cmd1 fails

### Exercise 3: Quote Handling
Process `echo "hello world"` so that "hello world" is treated as a single argument.

### Exercise 4: Tab Completion
Use the readline library to implement tab auto-completion.

---

## Key Concepts Summary

| Function | Description |
|----------|-------------|
| `fork()` | Clone process |
| `exec*()` | Execute program |
| `wait()` | Wait for child process |
| `pipe()` | Create pipe |
| `dup2()` | Duplicate file descriptor |
| `open()` | Open file |
| `signal()` | Register signal handler |

| Concept | Description |
|---------|-------------|
| Pipe | Unidirectional inter-process communication |
| Redirection | Change input/output direction |
| Environment variables | Settings passed to processes |
| Signal | Notification sent to processes |

---

## Next Steps

Once you've completed the mini shell, proceed to the next project:
- [Project 12: Multithreaded Programming](13_Project_Multithreading.md) - Using pthread
