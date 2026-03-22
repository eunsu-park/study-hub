# 프로젝트: 미니 셸

**이전**: [프로세스 관리](./09_Process_Management.md) | **다음**: [멀티스레딩](./11_Multithreading.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 모든 명령줄 셸의 핵심인 읽기-파싱-실행 루프를 구현할 수 있다
2. `fork`를 사용하여 자식 프로세스를 생성하고 `execvp`로 외부 프로그램으로 교체할 수 있다
3. 셸 자체의 프로세스에서 실행되어야 하는 내장 명령 (`cd`, `pwd`, `echo`, `export`)을 구현할 수 있다
4. `<`, `>`, `>>` 토큰을 감지하고 `dup2`로 파일 디스크립터를 재지정하는 리다이렉션 파서를 설계할 수 있다
5. `pipe`, `dup2`, 조율된 `fork`/`wait` 호출로 여러 명령을 연결하는 파이프 실행기를 구축할 수 있다
6. Ctrl+C가 전경 명령을 중단하되 셸을 종료하지 않도록 `SIGINT` 핸들러를 설정할 수 있다
7. 명령 인수에서 환경 변수 확장 (`$VAR`)과 와일드카드 글로빙 (`*`, `?`)을 구현할 수 있다

---

셸은 다른 프로그램을 실행하기 위해 매일 사용하는 프로그램이지만, 대부분의 개발자는 그 내부를 들여다보지 않습니다. 미니 셸을 처음부터 구축하는 것은 프로세스, 파일 디스크립터, 파이프, 시그널, 환경 변수 -- Unix의 기본 구성 요소를 한데 묶기 때문에 가장 보람 있는 시스템 프로그래밍 연습 중 하나입니다. 이 프로젝트를 마치면 Enter를 누르는 순간부터 명령의 출력이 화면에 나타나는 순간까지 무슨 일이 일어나는지 이해하게 될 것입니다.

## 사전 지식
- 문자열 처리
- 파일 I/O
- 포인터와 동적 메모리

---

## 단계 1: 기본 셸 구조

기본 셸 동작: **읽기 -> 파싱 -> 실행 -> 반복**

### 가장 간단한 셸

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

### 컴파일 및 테스트

```bash
gcc -o minish minishell_v1.c
./minish

minish> ls -l
minish> pwd
minish> echo hello world
minish> exit
```

---

## 단계 2: 내장 명령

일부 명령은 외부 프로그램이 아닌 셸 자체에서 처리해야 합니다.

### 내장 명령 구현

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

    printf("\n=== 미니 셸 도움말 ===\n\n");
    printf("내장 명령:\n");
    printf("  cd [directory]  - 디렉토리 변경\n");
    printf("  pwd             - 현재 디렉토리 출력\n");
    printf("  echo [text]     - 텍스트 출력\n");
    printf("  export VAR=val  - 환경 변수 설정\n");
    printf("  env             - 환경 변수 목록\n");
    printf("  help            - 이 도움말 표시\n");
    printf("  exit            - 셸 종료\n");
    printf("\n외부 명령은 PATH에서 검색됩니다.\n\n");

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

## 단계 3: 리다이렉션 구현

`>`, `>>`, `<` 연산자를 처리합니다.

### 리다이렉션 파서

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

### 리다이렉션 사용

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

테스트:
```bash
minish> ls -l > files.txt
minish> cat < files.txt
minish> echo "additional content" >> files.txt
minish> wc -l < files.txt
```

---

## 단계 4: 파이프 구현

`|` 연산자로 명령을 연결합니다.

### 파이프 처리

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

테스트:
```bash
minish> ls -l | grep ".c"
minish> cat file.txt | wc -l
minish> ps aux | grep bash | head -5
```

---

## 단계 5: 완전한 미니 셸

### 전체 코드

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
    printf("=== 미니 셸 도움말 ===\n\n");
    printf("내장 명령:\n");
    printf("  cd [dir]    디렉토리 변경\n");
    printf("  pwd         현재 디렉토리 출력\n");
    printf("  echo [...]  텍스트 출력\n");
    printf("  export V=X  환경 변수 설정\n");
    printf("  unset VAR   환경 변수 해제\n");
    printf("  help        이 도움말 표시\n");
    printf("  exit [N]    셸 종료\n\n");
    printf("리다이렉션:\n");
    printf("  cmd > file  출력을 파일로 리다이렉트\n");
    printf("  cmd >> file 출력을 파일에 추가\n");
    printf("  cmd < file  파일에서 입력 읽기\n\n");
    printf("파이프:\n");
    printf("  cmd1 | cmd2 출력을 다음 명령으로 파이프\n\n");
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
        printf(" \033[1;32m>\033[0m ");
    } else {
        printf(" \033[1;31m>\033[0m ");
    }

    fflush(stdout);
}

// ============ Main ============

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    // Set signal handler
    signal(SIGINT, sigint_handler);

    printf("\n\033[1;36m=== 미니 셸 ===\033[0m\n");
    printf("'help'를 입력하면 도움말을 볼 수 있습니다\n\n");

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

### 컴파일 및 실행

```bash
gcc -o minishell minishell.c -Wall -Wextra
./minishell
```

### 테스트 예시

```bash
=== 미니 셸 ===
'help'를 입력하면 도움말을 볼 수 있습니다

~ > help
~ > pwd
/Users/username
~ > cd /tmp
/tmp > ls -la
/tmp > echo $HOME
/Users/username
/tmp > export MY_VAR=hello
/tmp > echo $MY_VAR
hello
/tmp > ls -l | grep ".txt" | wc -l
/tmp > cat /etc/passwd | head -5 > first5.txt
/tmp > cat first5.txt
/tmp > cd -
/Users/username
~ > exit
```

---

## 단계 6: 추가 기능

### 히스토리 기능

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

### 백그라운드 실행 (&)

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

### 와일드카드 확장 (*)

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

## 연습 문제

### 연습 문제 1: 세미콜론 지원
`cmd1 ; cmd2` 형식으로 여러 명령의 순차 실행을 구현하세요.

### 연습 문제 2: && 및 || 연산자
- `cmd1 && cmd2`: cmd1이 성공한 경우에만 cmd2 실행
- `cmd1 || cmd2`: cmd1이 실패한 경우에만 cmd2 실행

### 연습 문제 3: 인용부호 처리
`echo "hello world"`를 처리하여 "hello world"가 하나의 인수로 취급되도록 하세요.

### 연습 문제 4: 탭 자동완성
readline 라이브러리를 사용하여 탭 자동완성을 구현하세요.

### 연습 문제 5: 히어 도큐먼트
명령에 여러 줄 입력을 stdin으로 전달할 수 있는 `<<EOF` 스타일의 히어 도큐먼트를 구현하세요.

---

## 핵심 개념 요약

| 함수 | 설명 |
|------|------|
| `fork()` | 프로세스 복제 |
| `exec*()` | 프로그램 실행 |
| `wait()` | 자식 프로세스 대기 |
| `pipe()` | 파이프 생성 |
| `dup2()` | 파일 디스크립터 복제 |
| `open()` | 파일 열기 |
| `signal()` | 시그널 핸들러 등록 |

| 개념 | 설명 |
|------|------|
| 파이프 | 단방향 프로세스 간 통신 |
| 리다이렉션 | 입출력 방향 변경 |
| 환경 변수 | 프로세스에 전달되는 설정 |
| 시그널 | 프로세스에 전송되는 알림 |

---

## 다음 단계

미니 셸을 완성했다면 다음 레슨으로 진행하세요:
- [C 멀티스레딩](./11_Multithreading.md) -- pthread를 사용한 동시성 프로그래밍
