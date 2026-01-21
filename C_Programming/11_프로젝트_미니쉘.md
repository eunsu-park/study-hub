# 프로젝트 11: 미니 쉘

간단한 명령어 쉘을 직접 구현해봅니다.

## 학습 목표
- 프로세스 생성 (fork)
- 프로그램 실행 (exec 계열)
- 파이프와 리다이렉션
- 시그널 처리 기초

## 사전 지식
- 문자열 처리
- 파일 I/O
- 포인터와 동적 메모리

---

## 1단계: 기본 쉘 구조

쉘의 기본 동작: **읽기 → 파싱 → 실행 → 반복**

### 가장 간단한 쉘

```c
// minishell_v1.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_INPUT 1024
#define MAX_ARGS 64

// 입력을 공백으로 분리
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

// 명령어 실행
void execute(char** args) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork 실패");
        return;
    }

    if (pid == 0) {
        // 자식 프로세스: 명령어 실행
        execvp(args[0], args);
        // execvp 실패시
        perror(args[0]);
        exit(EXIT_FAILURE);
    } else {
        // 부모 프로세스: 자식 종료 대기
        int status;
        waitpid(pid, &status, 0);
    }
}

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    while (1) {
        // 프롬프트 출력
        printf("minish> ");
        fflush(stdout);

        // 입력 읽기
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n");
            break;  // EOF (Ctrl+D)
        }

        // 빈 입력 무시
        if (input[0] == '\n') continue;

        // 파싱
        int argc = parse_input(input, args);
        if (argc == 0) continue;

        // exit 명령어
        if (strcmp(args[0], "exit") == 0) {
            printf("쉘을 종료합니다.\n");
            break;
        }

        // 실행
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

## 2단계: 내장 명령어 (Built-in Commands)

일부 명령어는 외부 프로그램이 아닌 쉘 자체에서 처리해야 합니다.

### 내장 명령어 구현

```c
// builtins.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// 내장 명령어 이름들
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

// cd: 디렉토리 변경
int builtin_cd(char** args) {
    const char* path;

    if (args[1] == NULL) {
        // 인자 없으면 홈 디렉토리
        path = getenv("HOME");
        if (path == NULL) {
            fprintf(stderr, "cd: HOME 환경변수가 설정되지 않음\n");
            return 1;
        }
    } else if (strcmp(args[1], "-") == 0) {
        // cd - : 이전 디렉토리
        path = getenv("OLDPWD");
        if (path == NULL) {
            fprintf(stderr, "cd: OLDPWD 환경변수가 설정되지 않음\n");
            return 1;
        }
        printf("%s\n", path);
    } else if (strcmp(args[1], "~") == 0) {
        path = getenv("HOME");
    } else {
        path = args[1];
    }

    // 현재 디렉토리 저장
    char oldpwd[1024];
    getcwd(oldpwd, sizeof(oldpwd));

    if (chdir(path) != 0) {
        perror("cd");
        return 1;
    }

    // OLDPWD, PWD 환경변수 갱신
    setenv("OLDPWD", oldpwd, 1);

    char newpwd[1024];
    getcwd(newpwd, sizeof(newpwd));
    setenv("PWD", newpwd, 1);

    return 0;
}

// pwd: 현재 디렉토리 출력
int builtin_pwd(char** args) {
    (void)args;  // 사용하지 않음

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("%s\n", cwd);
        return 0;
    }
    perror("pwd");
    return 1;
}

// echo: 인자 출력
int builtin_echo(char** args) {
    int newline = 1;
    int start = 1;

    // -n 옵션: 줄바꿈 없이 출력
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

// help: 도움말
int builtin_help(char** args) {
    (void)args;

    printf("\n=== Mini Shell 도움말 ===\n\n");
    printf("내장 명령어:\n");
    printf("  cd [디렉토리]  - 디렉토리 변경\n");
    printf("  pwd           - 현재 디렉토리 출력\n");
    printf("  echo [텍스트]  - 텍스트 출력\n");
    printf("  export VAR=값  - 환경변수 설정\n");
    printf("  env           - 환경변수 목록\n");
    printf("  help          - 이 도움말\n");
    printf("  exit          - 쉘 종료\n");
    printf("\n외부 명령어는 PATH에서 검색됩니다.\n\n");

    return 0;
}

// export: 환경변수 설정
int builtin_export(char** args) {
    if (args[1] == NULL) {
        // 인자 없으면 환경변수 목록 출력
        extern char** environ;
        for (char** env = environ; *env; env++) {
            printf("export %s\n", *env);
        }
        return 0;
    }

    // VAR=value 형식 파싱
    for (int i = 1; args[i]; i++) {
        char* eq = strchr(args[i], '=');
        if (eq) {
            *eq = '\0';
            setenv(args[i], eq + 1, 1);
            *eq = '=';
        } else {
            // = 없으면 빈 값으로 설정
            setenv(args[i], "", 1);
        }
    }

    return 0;
}

// env: 환경변수 출력
int builtin_env(char** args) {
    (void)args;

    extern char** environ;
    for (char** env = environ; *env; env++) {
        printf("%s\n", *env);
    }
    return 0;
}

// 내장 명령어인지 확인하고 실행
// 반환: -1 (내장 명령어 아님), 0+ (실행 결과)
int execute_builtin(char** args) {
    if (args[0] == NULL) return -1;

    if (strcmp(args[0], "cd") == 0) return builtin_cd(args);
    if (strcmp(args[0], "pwd") == 0) return builtin_pwd(args);
    if (strcmp(args[0], "echo") == 0) return builtin_echo(args);
    if (strcmp(args[0], "help") == 0) return builtin_help(args);
    if (strcmp(args[0], "export") == 0) return builtin_export(args);
    if (strcmp(args[0], "env") == 0) return builtin_env(args);

    return -1;  // 내장 명령어 아님
}
```

---

## 3단계: 리다이렉션 구현

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
    char* input_file;   // < 파일
    char* output_file;  // > 또는 >> 파일
    int append;         // >> 인 경우 1
} Redirect;

// 리다이렉션 파싱
// args에서 리다이렉션 제거하고 Redirect 구조체에 저장
void parse_redirect(char** args, Redirect* redir) {
    redir->input_file = NULL;
    redir->output_file = NULL;
    redir->append = 0;

    int i = 0;
    int j = 0;

    while (args[i] != NULL) {
        if (strcmp(args[i], "<") == 0) {
            // 입력 리다이렉션
            if (args[i + 1]) {
                redir->input_file = args[i + 1];
                i += 2;
                continue;
            }
        } else if (strcmp(args[i], ">") == 0) {
            // 출력 리다이렉션 (덮어쓰기)
            if (args[i + 1]) {
                redir->output_file = args[i + 1];
                redir->append = 0;
                i += 2;
                continue;
            }
        } else if (strcmp(args[i], ">>") == 0) {
            // 출력 리다이렉션 (추가)
            if (args[i + 1]) {
                redir->output_file = args[i + 1];
                redir->append = 1;
                i += 2;
                continue;
            }
        }

        // 리다이렉션이 아닌 인자
        args[j++] = args[i++];
    }
    args[j] = NULL;
}

// 리다이렉션 적용 (자식 프로세스에서 호출)
int apply_redirect(Redirect* redir) {
    // 입력 리다이렉션
    if (redir->input_file) {
        int fd = open(redir->input_file, O_RDONLY);
        if (fd < 0) {
            perror(redir->input_file);
            return -1;
        }
        dup2(fd, STDIN_FILENO);
        close(fd);
    }

    // 출력 리다이렉션
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

### 리다이렉션 사용 예

```c
// execute with redirection
void execute_with_redirect(char** args) {
    Redirect redir;
    parse_redirect(args, &redir);

    if (args[0] == NULL) return;

    pid_t pid = fork();

    if (pid == 0) {
        // 자식: 리다이렉션 적용 후 실행
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
minish> echo "추가 내용" >> files.txt
minish> wc -l < files.txt
```

---

## 4단계: 파이프 구현

`|` 연산자로 명령어를 연결합니다.

### 파이프 처리

```c
// pipe.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_PIPES 10

// 파이프로 분리된 명령어 수 카운트
int count_pipes(char** args) {
    int count = 0;
    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) count++;
    }
    return count;
}

// 파이프 위치에서 args 분리
// commands[0] = 첫 번째 명령어의 args
// commands[1] = 두 번째 명령어의 args
// ...
int split_by_pipe(char** args, char*** commands) {
    int cmd_count = 0;
    commands[cmd_count++] = args;

    for (int i = 0; args[i]; i++) {
        if (strcmp(args[i], "|") == 0) {
            args[i] = NULL;  // 파이프 위치를 NULL로
            if (args[i + 1]) {
                commands[cmd_count++] = &args[i + 1];
            }
        }
    }

    return cmd_count;
}

// 파이프 실행
void execute_pipe(char** args) {
    char** commands[MAX_PIPES + 1];
    int cmd_count = split_by_pipe(args, commands);

    if (cmd_count == 1) {
        // 파이프 없음: 일반 실행
        execute_with_redirect(commands[0]);
        return;
    }

    int pipes[MAX_PIPES][2];  // 파이프 파일 디스크립터

    // 파이프 생성
    for (int i = 0; i < cmd_count - 1; i++) {
        if (pipe(pipes[i]) < 0) {
            perror("pipe");
            return;
        }
    }

    // 각 명령어 실행
    for (int i = 0; i < cmd_count; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // 자식 프로세스

            // 이전 파이프의 읽기 끝을 stdin으로
            if (i > 0) {
                dup2(pipes[i - 1][0], STDIN_FILENO);
            }

            // 다음 파이프의 쓰기 끝을 stdout으로
            if (i < cmd_count - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            // 모든 파이프 닫기
            for (int j = 0; j < cmd_count - 1; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            // 명령어 실행
            execvp(commands[i][0], commands[i]);
            perror(commands[i][0]);
            exit(EXIT_FAILURE);

        } else if (pid < 0) {
            perror("fork");
            return;
        }
    }

    // 부모: 모든 파이프 닫기
    for (int i = 0; i < cmd_count - 1; i++) {
        close(pipes[i][0]);
        close(pipes[i][1]);
    }

    // 모든 자식 프로세스 대기
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

## 5단계: 완성된 미니 쉘

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

// ============ 전역 변수 ============
static int last_exit_status = 0;

// ============ 시그널 핸들러 ============
void sigint_handler(int sig) {
    (void)sig;
    printf("\n");
    // 프롬프트 다시 출력하지 않음 (메인 루프에서 처리)
}

// ============ 유틸리티 ============

// 문자열 양쪽 공백 제거
char* trim(char* str) {
    while (*str == ' ' || *str == '\t') str++;

    if (*str == '\0') return str;

    char* end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\n')) {
        *end-- = '\0';
    }

    return str;
}

// ============ 파싱 ============

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

// ============ 리다이렉션 ============

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

// ============ 내장 명령어 ============

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
        // 환경변수 확장 ($VAR)
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
    printf("║        Mini Shell 도움말              ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ 내장 명령어:                          ║\n");
    printf("║   cd [dir]    디렉토리 변경           ║\n");
    printf("║   pwd         현재 디렉토리           ║\n");
    printf("║   echo [...]  텍스트 출력             ║\n");
    printf("║   export V=X  환경변수 설정           ║\n");
    printf("║   unset VAR   환경변수 삭제           ║\n");
    printf("║   help        이 도움말               ║\n");
    printf("║   exit [N]    쉘 종료                 ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ 리다이렉션:                           ║\n");
    printf("║   cmd > file  출력을 파일로           ║\n");
    printf("║   cmd >> file 출력을 파일에 추가      ║\n");
    printf("║   cmd < file  파일에서 입력           ║\n");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ 파이프:                               ║\n");
    printf("║   cmd1 | cmd2 출력을 다음 명령 입력으로 ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    printf("\n");
    return 0;
}

// 내장 명령어 실행 (-1: 내장 아님)
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

// ============ 파이프 실행 ============

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

    // 파이프 없으면 단일 명령 실행
    if (n == 1) {
        Redirect r;
        parse_redirect(cmds[0], &r);

        if (!cmds[0][0]) return;

        // 내장 명령어 체크
        int builtin_result = run_builtin(cmds[0]);
        if (builtin_result != -1) {
            last_exit_status = builtin_result;
            return;
        }

        // 외부 명령어
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

    // 파이프가 있는 경우
    int pipes[MAX_PIPES][2];
    for (int i = 0; i < n - 1; i++) {
        pipe(pipes[i]);
    }

    for (int i = 0; i < n; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // 입력 연결
            if (i > 0) {
                dup2(pipes[i-1][0], STDIN_FILENO);
            }
            // 출력 연결
            if (i < n - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            // 모든 파이프 닫기
            for (int j = 0; j < n - 1; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            // 리다이렉션 처리 (첫/마지막 명령에만 적용)
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

    // 부모: 파이프 닫고 대기
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

// ============ 프롬프트 ============

void print_prompt(void) {
    char cwd[256];
    char* dir = getcwd(cwd, sizeof(cwd));

    // 홈 디렉토리를 ~로 표시
    char* home = getenv("HOME");
    if (home && dir && strncmp(dir, home, strlen(home)) == 0) {
        printf("\033[1;34m~%s\033[0m", dir + strlen(home));
    } else {
        printf("\033[1;34m%s\033[0m", dir ? dir : "?");
    }

    // 종료 코드에 따라 색상 변경
    if (last_exit_status == 0) {
        printf(" \033[1;32m❯\033[0m ");
    } else {
        printf(" \033[1;31m❯\033[0m ");
    }

    fflush(stdout);
}

// ============ 메인 ============

int main(void) {
    char input[MAX_INPUT];
    char* args[MAX_ARGS];

    // 시그널 핸들러 설정
    signal(SIGINT, sigint_handler);

    printf("\n\033[1;36m=== Mini Shell ===\033[0m\n");
    printf("'help' 입력하여 도움말 보기\n\n");

    while (1) {
        print_prompt();

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\nexit\n");
            break;
        }

        char* trimmed = trim(input);
        if (*trimmed == '\0') continue;

        // 주석 무시
        if (trimmed[0] == '#') continue;

        // 입력 복사 (strtok이 원본 수정)
        char input_copy[MAX_INPUT];
        strncpy(input_copy, trimmed, sizeof(input_copy));

        // 파싱
        int argc = parse_args(input_copy, args);
        if (argc == 0) continue;

        // exit 명령어
        if (strcmp(args[0], "exit") == 0) {
            int code = args[1] ? atoi(args[1]) : last_exit_status;
            printf("exit\n");
            exit(code);
        }

        // 실행
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

### 테스트 예제

```bash
=== Mini Shell ===
'help' 입력하여 도움말 보기

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

## 6단계: 추가 기능

### 히스토리 기능

```c
#define HISTORY_SIZE 100

static char* history[HISTORY_SIZE];
static int history_count = 0;

void add_history(const char* cmd) {
    if (history_count < HISTORY_SIZE) {
        history[history_count++] = strdup(cmd);
    } else {
        // 오래된 것 제거
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
// & 체크
int is_background(char** args) {
    int i = 0;
    while (args[i]) i++;

    if (i > 0 && strcmp(args[i - 1], "&") == 0) {
        args[i - 1] = NULL;  // & 제거
        return 1;
    }
    return 0;
}

// 실행 함수 수정
void run_command(char** args) {
    int bg = is_background(args);

    pid_t pid = fork();

    if (pid == 0) {
        // 자식
        execvp(args[0], args);
        perror(args[0]);
        exit(127);
    } else if (pid > 0) {
        if (bg) {
            printf("[%d] %d\n", 1, pid);
            // 백그라운드: 대기하지 않음
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

// 와일드카드 확장
int expand_wildcards(char** args, char** expanded, int max_expanded) {
    int count = 0;

    for (int i = 0; args[i] && count < max_expanded - 1; i++) {
        // * 또는 ? 포함된 인자
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

### 연습 1: 세미콜론 지원
`cmd1 ; cmd2` 형태로 여러 명령어를 순차 실행하도록 구현하세요.

### 연습 2: && 와 || 연산자
- `cmd1 && cmd2`: cmd1 성공시에만 cmd2 실행
- `cmd1 || cmd2`: cmd1 실패시에만 cmd2 실행

### 연습 3: 따옴표 처리
`echo "hello world"`에서 "hello world"를 하나의 인자로 처리하세요.

### 연습 4: 탭 자동완성
readline 라이브러리를 사용하여 탭 자동완성을 구현하세요.

---

## 핵심 개념 정리

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
| 파이프 | 프로세스 간 단방향 통신 |
| 리다이렉션 | 입출력 방향 변경 |
| 환경변수 | 프로세스에 전달되는 설정 |
| 시그널 | 프로세스에 보내는 알림 |

---

## 다음 단계

미니 쉘을 완성했다면 다음 프로젝트로 넘어가세요:
- [프로젝트 12: 멀티스레드 프로그래밍](12_프로젝트_멀티스레드.md) - pthread 활용
