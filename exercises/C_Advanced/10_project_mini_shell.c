/*
 * Exercises for Lesson 12: Project Mini Shell
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex12 12_project_mini_shell.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* === Exercise 1: Command Parsing and Tokenizing === */
/* Problem: Split a command line into tokens, handling quotes and whitespace. */

#define MAX_TOKENS 64
#define MAX_CMD_LEN 256

typedef struct {
    char *tokens[MAX_TOKENS];
    int count;
} TokenList;

void tokenize_simple(const char *input, TokenList *tl) {
    /*
     * Simple tokenizer: split on whitespace.
     * strtok modifies the input string, so we work on a copy.
     *
     * Limitations of this simple approach:
     * - Does not handle quoted strings ("hello world" as one token)
     * - Does not handle escape characters (\ )
     * - Does not handle special characters (|, >, <, &)
     */
    static char buf[MAX_CMD_LEN];
    strncpy(buf, input, MAX_CMD_LEN - 1);
    buf[MAX_CMD_LEN - 1] = '\0';

    tl->count = 0;
    char *token = strtok(buf, " \t\n");
    while (token && tl->count < MAX_TOKENS) {
        tl->tokens[tl->count++] = token;
        token = strtok(NULL, " \t\n");
    }
}

void tokenize_advanced(const char *input, TokenList *tl) {
    /*
     * Advanced tokenizer: handles double-quoted strings.
     * "hello world" is treated as a single token.
     *
     * State machine with two states:
     * - NORMAL: whitespace separates tokens, quote starts quoted mode
     * - QUOTED: everything until closing quote is one token
     */
    static char buf[MAX_CMD_LEN];
    static char token_buf[MAX_TOKENS][64];
    strncpy(buf, input, MAX_CMD_LEN - 1);
    buf[MAX_CMD_LEN - 1] = '\0';

    tl->count = 0;
    int i = 0;

    while (buf[i] && tl->count < MAX_TOKENS) {
        /* Skip whitespace */
        while (buf[i] && isspace((unsigned char)buf[i])) i++;
        if (!buf[i]) break;

        int tok_idx = 0;
        if (buf[i] == '"') {
            /* Quoted token */
            i++; /* Skip opening quote */
            while (buf[i] && buf[i] != '"') {
                token_buf[tl->count][tok_idx++] = buf[i++];
            }
            if (buf[i] == '"') i++; /* Skip closing quote */
        } else {
            /* Regular token */
            while (buf[i] && !isspace((unsigned char)buf[i])) {
                token_buf[tl->count][tok_idx++] = buf[i++];
            }
        }
        token_buf[tl->count][tok_idx] = '\0';
        tl->tokens[tl->count] = token_buf[tl->count];
        tl->count++;
    }
}

void exercise_1(void) {
    printf("=== Exercise 1: Command Parsing and Tokenizing ===\n");

    const char *commands[] = {
        "ls -la /home/user",
        "echo hello world",
        "grep -r \"search term\" .",
        "cat   file.txt",            /* Extra spaces */
        "",                          /* Empty */
        "echo \"hello world\" done",
    };
    int n_cmds = (int)(sizeof(commands) / sizeof(commands[0]));

    for (int c = 0; c < n_cmds; c++) {
        printf("\nInput: \"%s\"\n", commands[c]);

        TokenList simple, advanced;
        tokenize_simple(commands[c], &simple);
        tokenize_advanced(commands[c], &advanced);

        printf("  Simple   (%d tokens):", simple.count);
        for (int i = 0; i < simple.count; i++) printf(" [%s]", simple.tokens[i]);
        printf("\n");

        printf("  Advanced (%d tokens):", advanced.count);
        for (int i = 0; i < advanced.count; i++) printf(" [%s]", advanced.tokens[i]);
        printf("\n");
    }
}

/* === Exercise 2: Fork and Exec (Conceptual) === */
/* Problem: Explain fork+exec model with simulated execution. */

typedef struct {
    int pid;
    const char *command;
    const char *status;
    int exit_code;
} ProcessInfo;

void exercise_2(void) {
    printf("\n=== Exercise 2: Fork and Exec ===\n");

    /*
     * The Unix process creation model:
     *
     * 1. fork() - Creates an exact copy of the current process
     *    - Returns 0 to the child process
     *    - Returns child's PID to the parent
     *    - Returns -1 on error
     *
     * 2. exec*() - Replaces the current process image with a new program
     *    - execvp(file, argv) - search PATH for 'file'
     *    - execv(path, argv)  - use exact path
     *    - execl(path, arg0, arg1, ..., NULL) - variadic version
     *
     * 3. wait()/waitpid() - Parent waits for child to complete
     *    - Collects exit status
     *    - Prevents zombie processes
     *
     * Pseudocode for shell command execution:
     *   pid = fork();
     *   if (pid == 0) {
     *       // Child process
     *       execvp(argv[0], argv);
     *       perror("exec failed");  // Only reached if exec fails
     *       exit(1);
     *   } else if (pid > 0) {
     *       // Parent process
     *       waitpid(pid, &status, 0);
     *   } else {
     *       perror("fork failed");
     *   }
     */

    /* Simulate process execution */
    ProcessInfo processes[] = {
        {1001, "ls -la",       "EXITED", 0},
        {1002, "grep foo bar", "EXITED", 1},    /* grep returns 1 if no match */
        {1003, "sleep 10",     "KILLED", 9},     /* SIGKILL */
        {1004, "cat /nofile",  "EXITED", 1},
        {1005, "echo hello",   "EXITED", 0},
    };
    int n_procs = (int)(sizeof(processes) / sizeof(processes[0]));

    printf("Simulated process execution:\n\n");
    printf("%-6s  %-20s  %-10s  %-6s\n", "PID", "Command", "Status", "Exit");
    printf("------  --------------------  ----------  ------\n");

    for (int i = 0; i < n_procs; i++) {
        printf("%-6d  %-20s  %-10s  %-6d\n",
               processes[i].pid, processes[i].command,
               processes[i].status, processes[i].exit_code);
    }

    printf("\nKey concepts:\n");
    printf("  - fork() is cheap (copy-on-write)\n");
    printf("  - exec() does not return on success\n");
    printf("  - Always check fork() return value\n");
    printf("  - Always waitpid() to avoid zombies\n");
}

/* === Exercise 3: Pipe Chaining === */
/* Problem: Parse and simulate pipe chains like "cmd1 | cmd2 | cmd3". */

#define MAX_PIPES 8

typedef struct {
    TokenList commands[MAX_PIPES + 1];
    int n_commands;
} PipeChain;

PipeChain parse_pipeline(const char *input) {
    /*
     * Pipeline parsing:
     * 1. Split input on '|' to get individual commands
     * 2. Tokenize each command separately
     *
     * Implementation with pipe():
     * - For N commands, create N-1 pipes
     * - Each pipe has fd[0] (read) and fd[1] (write)
     * - Command i writes to pipe[i][1], command i+1 reads from pipe[i][0]
     * - First command reads from stdin, last writes to stdout
     */
    PipeChain chain = { .n_commands = 0 };
    char buf[MAX_CMD_LEN];
    strncpy(buf, input, MAX_CMD_LEN - 1);
    buf[MAX_CMD_LEN - 1] = '\0';

    char *segment = strtok(buf, "|");
    while (segment && chain.n_commands <= MAX_PIPES) {
        /* Trim leading whitespace */
        while (*segment && isspace((unsigned char)*segment)) segment++;
        tokenize_advanced(segment, &chain.commands[chain.n_commands]);
        chain.n_commands++;
        segment = strtok(NULL, "|");
    }
    return chain;
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Pipe Chaining ===\n");

    const char *pipelines[] = {
        "ls -la",
        "cat file.txt | grep error",
        "ps aux | grep python | wc -l",
        "cat /var/log/syslog | grep error | sort | uniq -c | head -10",
    };
    int n_pipes = (int)(sizeof(pipelines) / sizeof(pipelines[0]));

    for (int p = 0; p < n_pipes; p++) {
        printf("\nPipeline: %s\n", pipelines[p]);
        PipeChain chain = parse_pipeline(pipelines[p]);
        printf("  Commands: %d, Pipes needed: %d\n",
               chain.n_commands, chain.n_commands - 1);

        for (int c = 0; c < chain.n_commands; c++) {
            printf("  [%d] ", c);
            for (int t = 0; t < chain.commands[c].count; t++) {
                printf("%s ", chain.commands[c].tokens[t]);
            }
            if (c < chain.n_commands - 1) printf("  -> pipe -> ");
            printf("\n");
        }
    }

    printf("\nPipe implementation pattern:\n");
    printf("  int fd[2];\n");
    printf("  pipe(fd);         // fd[0]=read, fd[1]=write\n");
    printf("  if (fork()==0) {  // Writer (left side)\n");
    printf("    close(fd[0]);\n");
    printf("    dup2(fd[1], STDOUT_FILENO);\n");
    printf("    exec(cmd1);\n");
    printf("  } else {          // Reader (right side)\n");
    printf("    close(fd[1]);\n");
    printf("    dup2(fd[0], STDIN_FILENO);\n");
    printf("    exec(cmd2);\n");
    printf("  }\n");
}

/* === Exercise 4: I/O Redirection === */
/* Problem: Parse and simulate input/output redirection (>, <, >>). */

typedef struct {
    char *argv[MAX_TOKENS];
    int argc;
    char *input_file;    /* < file */
    char *output_file;   /* > file */
    int append;          /* >> instead of > */
} RedirectedCmd;

RedirectedCmd parse_redirections(const char *input) {
    /*
     * Redirection parsing:
     * - Find >, >>, < tokens
     * - The token after the redirect symbol is the filename
     * - Remove redirect symbols and filenames from argv
     *
     * Implementation uses dup2():
     *   int fd = open(file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
     *   dup2(fd, STDOUT_FILENO);  // Redirect stdout to file
     *   close(fd);
     */
    static char buf[MAX_CMD_LEN];
    static char tokens[MAX_TOKENS][64];
    strncpy(buf, input, MAX_CMD_LEN - 1);
    buf[MAX_CMD_LEN - 1] = '\0';

    RedirectedCmd cmd = { .argc = 0, .input_file = NULL,
                          .output_file = NULL, .append = 0 };

    char *tok = strtok(buf, " \t");
    while (tok) {
        if (strcmp(tok, ">>") == 0) {
            tok = strtok(NULL, " \t");
            if (tok) {
                cmd.output_file = tok;
                cmd.append = 1;
            }
        } else if (strcmp(tok, ">") == 0) {
            tok = strtok(NULL, " \t");
            if (tok) {
                cmd.output_file = tok;
                cmd.append = 0;
            }
        } else if (strcmp(tok, "<") == 0) {
            tok = strtok(NULL, " \t");
            if (tok) cmd.input_file = tok;
        } else {
            strcpy(tokens[cmd.argc], tok);
            cmd.argv[cmd.argc] = tokens[cmd.argc];
            cmd.argc++;
        }
        tok = strtok(NULL, " \t");
    }
    cmd.argv[cmd.argc] = NULL;
    return cmd;
}

void exercise_4(void) {
    printf("\n=== Exercise 4: I/O Redirection ===\n");

    const char *commands[] = {
        "ls -la",
        "echo hello > output.txt",
        "cat < input.txt",
        "sort < data.txt > sorted.txt",
        "echo log >> app.log",
        "grep error < log.txt > errors.txt",
    };
    int n_cmds = (int)(sizeof(commands) / sizeof(commands[0]));

    for (int i = 0; i < n_cmds; i++) {
        printf("\nCommand: \"%s\"\n", commands[i]);
        RedirectedCmd cmd = parse_redirections(commands[i]);

        printf("  argv (%d):", cmd.argc);
        for (int j = 0; j < cmd.argc; j++) printf(" [%s]", cmd.argv[j]);
        printf("\n");

        printf("  stdin:  %s\n", cmd.input_file ? cmd.input_file : "(terminal)");
        printf("  stdout: %s%s\n",
               cmd.output_file ? cmd.output_file : "(terminal)",
               cmd.append ? " (append)" : "");
    }

    printf("\nRedirection implementation:\n");
    printf("  // Output: > file\n");
    printf("  fd = open(file, O_WRONLY|O_CREAT|O_TRUNC, 0644);\n");
    printf("  dup2(fd, STDOUT_FILENO);\n");
    printf("  // Append: >> file\n");
    printf("  fd = open(file, O_WRONLY|O_CREAT|O_APPEND, 0644);\n");
    printf("  // Input:  < file\n");
    printf("  fd = open(file, O_RDONLY);\n");
    printf("  dup2(fd, STDIN_FILENO);\n");
}

/* === Exercise 5: Built-in Commands (cd, exit, history) === */
/* Problem: Implement shell built-in commands that cannot be external processes. */

typedef int (*BuiltinFunc)(char **argv, int argc);

int builtin_cd(char **argv, int argc) {
    /*
     * cd must be a built-in because:
     * If cd were external, fork+exec would change the child's directory,
     * but the parent (shell) would stay in the original directory.
     * The shell process itself must call chdir().
     */
    if (argc < 2) {
        printf("  cd: would change to HOME directory\n");
        printf("  (uses getenv(\"HOME\") or getpwuid(getuid())->pw_dir)\n");
    } else {
        printf("  cd: would call chdir(\"%s\")\n", argv[1]);
    }
    return 0;
}

int builtin_exit(char **argv, int argc) {
    int code = 0;
    if (argc > 1) code = atoi(argv[1]);
    printf("  exit: would terminate shell with code %d\n", code);
    return code;
}

int builtin_history(char **argv, int argc) {
    (void)argv; (void)argc;
    /* Simulate command history */
    const char *hist[] = {"ls -la", "cd /home", "cat file.txt", "grep error log"};
    printf("  Command history:\n");
    for (int i = 0; i < 4; i++) {
        printf("    %d  %s\n", i + 1, hist[i]);
    }
    return 0;
}

typedef struct {
    const char *name;
    BuiltinFunc func;
    const char *help;
} BuiltinEntry;

void exercise_5(void) {
    printf("\n=== Exercise 5: Built-in Commands ===\n");

    BuiltinEntry builtins[] = {
        {"cd",      builtin_cd,      "Change directory"},
        {"exit",    builtin_exit,    "Exit the shell"},
        {"history", builtin_history, "Show command history"},
    };
    int n_builtins = (int)(sizeof(builtins) / sizeof(builtins[0]));

    printf("Registered built-in commands:\n");
    for (int i = 0; i < n_builtins; i++) {
        printf("  %-10s - %s\n", builtins[i].name, builtins[i].help);
    }

    /* Simulate command dispatch */
    const char *test_cmds[] = {"cd /home/user", "exit 0", "history", "ls -la"};
    int n_tests = (int)(sizeof(test_cmds) / sizeof(test_cmds[0]));

    printf("\nCommand dispatch:\n");
    for (int t = 0; t < n_tests; t++) {
        TokenList tl;
        tokenize_simple(test_cmds[t], &tl);
        if (tl.count == 0) continue;

        printf("\n> %s\n", test_cmds[t]);

        int found = 0;
        for (int b = 0; b < n_builtins; b++) {
            if (strcmp(tl.tokens[0], builtins[b].name) == 0) {
                builtins[b].func(tl.tokens, tl.count);
                found = 1;
                break;
            }
        }
        if (!found) {
            printf("  -> external command: would fork+exec \"%s\"\n", tl.tokens[0]);
        }
    }

    printf("\nWhy built-ins cannot be external:\n");
    printf("  - cd: must modify parent process's working directory\n");
    printf("  - exit: must terminate the shell process itself\n");
    printf("  - export: must modify parent process's environment\n");
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
