# 프로세스 관리

**이전**: [프로젝트: 파일 암호화](./08_Project_File_Encryption.md) | **다음**: [프로젝트: 미니 셸](./10_Project_Mini_Shell.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Unix 프로세스 모델과 PID, PPID, 프로세스 상태의 역할을 설명할 수 있다
2. fork를 사용하여 자식 프로세스를 생성하고 반환 값을 통해 부모와 자식을 구분할 수 있다
3. exec 함수 패밀리를 사용하여 프로세스 이미지를 교체할 수 있다
4. wait와 waitpid를 사용하여 부모-자식 실행을 동기화할 수 있다
5. getenv, setenv, environ 배열로 환경 변수를 관리할 수 있다

---

Unix 시스템에서 실행하는 모든 프로그램은 프로세스입니다 -- 자체 주소 공간, 파일 디스크립터, 스케줄링 상태를 가진 실행 파일의 인스턴스입니다. 프로세스가 어떻게 생성되고, 상태를 어떻게 전달하며, 새로운 프로그램으로 어떻게 교체되는지 이해하는 것은 시스템 프로그래밍에 필수적입니다. fork+exec 모델은 셸, 작업 스케줄러, 프로세스 감시자의 기반입니다. 이 레슨에서는 C 코드로 프로세스를 생성, 모니터링, 제어하는 도구를 제공합니다.

## 1. Unix 프로세스 모델

### PID와 PPID

모든 프로세스는 커널이 할당하는 고유한 **프로세스 ID (PID)**를 가집니다. 또한 자신을 생성한 프로세스를 식별하는 **부모 프로세스 ID (PPID)**를 가집니다.

```c
#include <stdio.h>
#include <unistd.h>

int main(void) {
    printf("PID:  %d\n", getpid());
    printf("PPID: %d\n", getppid());
    return 0;
}
```

### 프로세스 상태

프로세스는 수명 동안 여러 상태를 거칩니다:

| 상태 | 설명 |
|------|------|
| **실행 (R)** | 현재 CPU에서 실행 중 |
| **대기 (S)** | 이벤트 대기 중 (I/O, 시그널, 타이머) |
| **정지 (T)** | 시그널에 의해 중단됨 (SIGSTOP, SIGTSTP) |
| **좀비 (Z)** | 종료되었으나 부모에 의해 아직 회수되지 않음 |
| **소멸 (X)** | 완전히 정리됨 (일시적 상태) |

### Linux의 /proc

Linux에서 `/proc` 파일시스템은 프로세스별 정보를 가상 파일로 노출합니다:

```c
#include <stdio.h>
#include <unistd.h>

void print_proc_status(void) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/status", getpid());

    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        // Print selected fields
        if (strncmp(line, "Name:", 5) == 0 ||
            strncmp(line, "State:", 6) == 0 ||
            strncmp(line, "Pid:", 4) == 0 ||
            strncmp(line, "PPid:", 5) == 0 ||
            strncmp(line, "VmRSS:", 6) == 0) {
            printf("%s", line);
        }
    }
    fclose(f);
}
```

주요 `/proc/<pid>/` 항목:

| 파일 | 내용 |
|------|------|
| `status` | 프로세스 상태, 메모리 사용량, UID |
| `cmdline` | 명령줄 인수 (NUL로 구분) |
| `fd/` | 열린 파일 디스크립터 디렉토리 |
| `maps` | 메모리 매핑된 영역 |
| `environ` | 환경 변수 |

---

## 2. fork() -- 자식 프로세스 생성

### fork의 동작 방식

`fork()`는 호출한 프로세스를 복제하여 새로운 프로세스를 생성합니다. 자식은 부모의 거의 정확한 복사본입니다 -- 같은 코드, 같은 데이터, 같은 열린 파일 디스크립터 -- 하지만 새로운 PID를 가집니다.

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main(void) {
    printf("Before fork: PID = %d\n", getpid());

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) {
        // Child process
        printf("Child:  PID = %d, PPID = %d\n", getpid(), getppid());
    } else {
        // Parent process
        printf("Parent: PID = %d, child PID = %d\n", getpid(), pid);
    }

    return 0;
}
```

### 반환 값의 의미

| 반환 값 | 의미 |
|---------|------|
| **음수** | `fork()` 실패 (자식이 생성되지 않음) |
| **0** | 자식 프로세스에 있음 |
| **양수** | 부모 프로세스에 있음; 값은 자식의 PID |

### 쓰기 시 복사 (Copy-on-Write)

최신 커널은 **쓰기 시 복사(COW)**를 사용합니다: `fork()` 후 부모와 자식은 같은 물리 메모리 페이지를 공유합니다. 한 프로세스가 쓰기를 할 때만 페이지가 복사됩니다. 이로 인해 큰 프로세스에서도 `fork()`가 빠릅니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int shared_var = 42;

    pid_t pid = fork();
    if (pid == 0) {
        // Child modifies its own copy (COW triggers here)
        shared_var = 100;
        printf("Child: shared_var = %d (addr: %p)\n",
               shared_var, (void *)&shared_var);
        exit(0);
    }

    wait(NULL);
    // Parent still sees original value
    printf("Parent: shared_var = %d (addr: %p)\n",
           shared_var, (void *)&shared_var);

    return 0;
}
```

출력:
```
Child: shared_var = 100 (addr: 0x7fff...)
Parent: shared_var = 42 (addr: 0x7fff...)
```

주소가 같아 보이지만 (가상 주소), 자식의 쓰기 후에는 물리 페이지가 다릅니다.

---

## 3. exec() 패밀리 -- 프로세스 이미지 교체

`exec` 함수들은 현재 프로세스 이미지를 새로운 프로그램으로 교체합니다. PID는 그대로 유지되지만, 코드, 데이터, 스택은 완전히 교체됩니다.

### exec 변형

| 함수 | 경로 | 인수 | 환경 |
|------|------|------|------|
| `execl` | 전체 경로 | 가변 인수 목록 | 상속 |
| `execlp` | PATH 검색 | 가변 인수 목록 | 상속 |
| `execle` | 전체 경로 | 가변 인수 목록 | 명시적 |
| `execv` | 전체 경로 | 배열 | 상속 |
| `execvp` | PATH 검색 | 배열 | 상속 |
| `execve` | 전체 경로 | 배열 | 명시적 |

명명 규칙:
- **l** = list (가변 인수)
- **v** = vector (인수 배열)
- **p** = PATH 검색
- **e** = 명시적 환경

### 예제

```c
#include <stdio.h>
#include <unistd.h>

int main(void) {
    printf("About to exec ls...\n");

    // execl: full path, variadic args, NULL-terminated
    execl("/bin/ls", "ls", "-l", "-a", NULL);

    // If exec returns, it failed
    perror("execl failed");
    return 1;
}
```

```c
// execvp: searches PATH, args as array
char *args[] = {"ls", "-l", "-a", NULL};
execvp("ls", args);
perror("execvp failed");
```

```c
// execle: explicit environment
char *args[] = {"env", NULL};
char *envp[] = {"MY_VAR=hello", "PATH=/usr/bin", NULL};
execle("/usr/bin/env", "env", NULL, envp);
```

### 중요 사항

- `exec`은 성공 시 **반환하지 않습니다**. `exec` 호출 이후의 코드는 `exec`이 실패한 경우에만 실행됩니다.
- 열린 파일 디스크립터는 `exec`을 통해 보존됩니다 (`FD_CLOEXEC`로 표시된 경우 제외).
- `SIG_DFL`이나 `SIG_IGN`으로 설정된 시그널 처리는 보존됩니다; 사용자 정의 핸들러는 `SIG_DFL`로 초기화됩니다.

---

## 4. wait()와 waitpid() -- 자식 회수

자식이 종료되면 부모가 종료 상태를 가져올 때까지 **좀비**가 됩니다. `wait` 함수 패밀리가 좀비 자식을 회수합니다.

### 기본 wait

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    pid_t pid = fork();

    if (pid == 0) {
        printf("Child (PID %d) doing work...\n", getpid());
        sleep(2);
        printf("Child exiting with status 42\n");
        exit(42);
    }

    // Parent waits for any child
    int status;
    pid_t child = wait(&status);

    printf("Child %d terminated\n", child);

    if (WIFEXITED(status)) {
        printf("Normal exit, status = %d\n", WEXITSTATUS(status));
    }
    if (WIFSIGNALED(status)) {
        printf("Killed by signal %d\n", WTERMSIG(status));
    }

    return 0;
}
```

### 특정 자식을 위한 waitpid

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    pid_t pids[3];

    // Fork three children
    for (int i = 0; i < 3; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            sleep(i + 1);
            printf("Child %d (PID %d) done\n", i, getpid());
            exit(i * 10);
        }
    }

    // Wait for each child in order
    for (int i = 0; i < 3; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        if (WIFEXITED(status)) {
            printf("Child %d exited with %d\n", i, WEXITSTATUS(status));
        }
    }

    return 0;
}
```

### 상태 검사 매크로

| 매크로 | 참을 반환하는 경우 |
|--------|-------------------|
| `WIFEXITED(status)` | 자식이 정상 종료함 (`exit` 호출 또는 `main`에서 반환) |
| `WEXITSTATUS(status)` | 종료 코드 (`WIFEXITED`가 참인 경우에만 유효) |
| `WIFSIGNALED(status)` | 자식이 시그널에 의해 종료됨 |
| `WTERMSIG(status)` | 자식을 종료시킨 시그널 번호 |
| `WIFSTOPPED(status)` | 자식이 정지됨 (예: SIGSTOP에 의해) |
| `WSTOPSIG(status)` | 자식을 정지시킨 시그널 |

### WNOHANG을 사용한 비차단 대기

```c
// Check if any child has terminated without blocking
int status;
pid_t pid = waitpid(-1, &status, WNOHANG);

if (pid > 0) {
    printf("Child %d has terminated\n", pid);
} else if (pid == 0) {
    printf("No child has terminated yet\n");
} else {
    // pid == -1: error or no children
    perror("waitpid");
}
```

---

## 5. 프로세스 종료

### exit() vs _exit()

| 함수 | 설명 |
|------|------|
| `exit(status)` | stdio 버퍼를 플러시하고, `atexit` 핸들러를 호출한 후 종료 |
| `_exit(status)` | 정리 없이 즉시 종료 |

실패한 `exec()` 후 자식 프로세스에서는 `_exit()`를 사용하여 버퍼된 출력이 이중으로 플러시되는 것을 방지합니다:

```c
pid_t pid = fork();
if (pid == 0) {
    execvp(args[0], args);
    // exec failed -- use _exit to avoid flushing parent's buffers
    perror("exec");
    _exit(127);
}
```

### atexit() 핸들러

```c
#include <stdio.h>
#include <stdlib.h>

void cleanup1(void) {
    printf("Cleanup 1: closing database\n");
}

void cleanup2(void) {
    printf("Cleanup 2: saving config\n");
}

int main(void) {
    atexit(cleanup1);  // Called second (LIFO order)
    atexit(cleanup2);  // Called first

    printf("Program running...\n");
    // exit(0) or returning from main triggers atexit handlers
    return 0;
}
```

출력:
```
Program running...
Cleanup 2: saving config
Cleanup 1: closing database
```

### 종료 상태 규칙

| 상태 | 관례 |
|------|------|
| 0 | 성공 |
| 1 | 일반 오류 |
| 2 | 셸 명령 오용 |
| 126 | 명령 실행 불가 |
| 127 | 명령을 찾을 수 없음 |
| 128+N | 시그널 N에 의해 종료됨 |

---

## 6. 환경 변수

### getenv과 setenv

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // Read an environment variable
    const char *home = getenv("HOME");
    if (home) {
        printf("HOME = %s\n", home);
    }

    // Set an environment variable (1 = overwrite if exists)
    setenv("MY_APP_MODE", "debug", 1);
    printf("MY_APP_MODE = %s\n", getenv("MY_APP_MODE"));

    // Remove an environment variable
    unsetenv("MY_APP_MODE");
    printf("MY_APP_MODE after unset: %s\n",
           getenv("MY_APP_MODE") ? getenv("MY_APP_MODE") : "(null)");

    return 0;
}
```

### environ 배열

전역 변수 `environ`은 `"KEY=VALUE"` 문자열의 NULL 종료 배열입니다:

```c
#include <stdio.h>

extern char **environ;

int main(void) {
    // Print all environment variables
    for (char **env = environ; *env != NULL; env++) {
        printf("%s\n", *env);
    }
    return 0;
}
```

### 환경 변수 상속

자식 프로세스는 부모의 환경 복사본을 상속합니다. 자식에서의 변경은 부모에 영향을 미치지 않습니다:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    setenv("PARENT_VAR", "hello", 1);

    pid_t pid = fork();
    if (pid == 0) {
        // Child can read parent's variable
        printf("Child sees: %s\n", getenv("PARENT_VAR"));

        // Child modifies it -- parent is unaffected
        setenv("PARENT_VAR", "modified", 1);
        printf("Child changed to: %s\n", getenv("PARENT_VAR"));
        exit(0);
    }

    wait(NULL);
    printf("Parent still sees: %s\n", getenv("PARENT_VAR"));
    return 0;
}
```

---

## 7. fork+exec 패턴 -- 표준 프로세스 생성 패턴

`fork()` + `exec()`의 조합은 새로운 프로그램을 실행하는 표준 Unix 방식입니다. 부모가 fork하고, 자식이 exec을 호출하여 자신을 교체하며, 부모는 대기합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int run_command(char *const argv[]) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        // Child: replace with new program
        execvp(argv[0], argv);
        // If we get here, exec failed
        perror(argv[0]);
        _exit(127);
    }

    // Parent: wait for child
    int status;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        fprintf(stderr, "Command killed by signal %d\n", WTERMSIG(status));
        return 128 + WTERMSIG(status);
    }

    return -1;
}

int main(void) {
    // Run "ls -la"
    char *cmd1[] = {"ls", "-la", NULL};
    int rc = run_command(cmd1);
    printf("\nls exited with status %d\n\n", rc);

    // Run "date"
    char *cmd2[] = {"date", NULL};
    rc = run_command(cmd2);
    printf("\ndate exited with status %d\n", rc);

    return 0;
}
```

### 오류 처리 모범 사례

1. 항상 `fork()`의 반환 값을 확인합니다.
2. 실패한 `exec()` 후 자식에서는 `exit()`가 아닌 `_exit()`를 사용합니다.
3. 셸 규칙에 맞추어 "명령을 찾을 수 없음"에는 종료 코드 127을 사용합니다.
4. `WEXITSTATUS`에 접근하기 전에 `WIFEXITED`를 확인합니다.

---

## 8. 데몬 프로세스(daemon process) 생성

**데몬(daemon)**은 터미널에서 분리된 백그라운드 프로세스입니다. 표준 생성 패턴은 이중 포크(double-fork)를 사용하여 데몬이 제어 터미널을 절대로 다시 획득할 수 없도록 보장합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

void daemonize(void) {
    // 단계 1: 첫 번째 fork -- 부모 종료로 자식을 고아 프로세스로 만듦
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(EXIT_FAILURE); }
    if (pid > 0) exit(EXIT_SUCCESS);  // Parent exits

    // 단계 2: 새 세션 생성 -- 자식이 세션 리더가 되어
    // 제어 터미널에서 분리됨
    if (setsid() < 0) { perror("setsid"); exit(EXIT_FAILURE); }

    // 단계 3: 두 번째 fork -- 프로세스가 세션 리더가 아니게 하여
    // 제어 터미널을 절대로 획득할 수 없도록 보장
    pid = fork();
    if (pid < 0) { perror("fork"); exit(EXIT_FAILURE); }
    if (pid > 0) exit(EXIT_SUCCESS);  // First child exits

    // 단계 4: 작업 환경 정리
    umask(0);           // Clear file creation mask
    chdir("/");         // Avoid locking any mounted filesystem

    // 단계 5: stdin/stdout/stderr를 /dev/null로 리다이렉트
    int devnull = open("/dev/null", O_RDWR);
    dup2(devnull, STDIN_FILENO);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    if (devnull > STDERR_FILENO) close(devnull);
}

int main(void) {
    daemonize();

    // 데몬 본체: 터미널 없이 백그라운드에서 실행
    while (1) {
        // ... 주기적인 작업 수행 ...
        sleep(10);
    }
    return 0;
}
```

**이중 포크(double-fork)가 필요한 이유?** 첫 번째 `fork` 후 자식은 `setsid()`를 호출하여 제어 터미널이 없는 새 세션의 리더가 됩니다. 그러나 세션 리더는 tty 장치를 열어서 터미널을 획득할 *수 있습니다*. 두 번째 `fork`는 더 이상 세션 리더가 아닌 손자 프로세스를 생성하여, 어떤 상황에서도 제어 터미널을 획득하는 것이 불가능하게 만듭니다. `chdir("/")`와 `/dev/null` 리다이렉트를 결합하면, 결과 프로세스는 사용자의 로그인 환경에서 완전히 격리됩니다.

---

## 9. 실용 예제 -- 간단한 프로세스 실행기

사용자로부터 명령을 읽고 각 명령을 별도의 프로세스로 실행하는 프로그램입니다:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_INPUT 256
#define MAX_ARGS 32

int parse_command(char *input, char **args) {
    int argc = 0;
    char *token = strtok(input, " \t\n");
    while (token && argc < MAX_ARGS - 1) {
        args[argc++] = token;
        token = strtok(NULL, " \t\n");
    }
    args[argc] = NULL;
    return argc;
}

int main(void) {
    char input[MAX_INPUT];
    char *args[MAX_ARGS];

    printf("Simple Process Launcher (type 'quit' to exit)\n");

    while (1) {
        printf("launcher> ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) {
            printf("\n");
            break;
        }

        int argc = parse_command(input, args);
        if (argc == 0) continue;

        if (strcmp(args[0], "quit") == 0) break;

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            continue;
        }

        if (pid == 0) {
            execvp(args[0], args);
            fprintf(stderr, "%s: command not found\n", args[0]);
            _exit(127);
        }

        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            printf("[exit %d]\n", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            printf("[killed by signal %d]\n", WTERMSIG(status));
        }
    }

    printf("Goodbye!\n");
    return 0;
}
```

컴파일 및 테스트:

```bash
gcc -Wall -Wextra -o launcher launcher.c
./launcher
launcher> ls -l
launcher> echo hello world
launcher> date
launcher> quit
```

---

## 연습 문제

### 연습 문제 1: 프로세스 트리

4개의 프로세스 체인 (부모 -> 자식 -> 손자 -> 증손자)을 생성하는 프로그램을 작성하세요. 각 프로세스는 자신의 PID와 PPID를 출력한 다음, 자식을 기다린 후 종료해야 합니다. 출력이 올바른 부모-자식 관계를 보여주는지 확인하세요.

### 연습 문제 2: 병렬 명령 실행기

주어진 명령들을 동시에 실행하기 위해 `n`개의 자식 프로세스를 fork하는 함수 `run_parallel(char *commands[], int n)`을 작성하고, 모든 자식을 기다리세요. 각 명령의 종료 상태를 완료 순서대로 출력하세요. 다양한 실행 시간의 명령 (예: `sleep 1`, `sleep 3`, `ls`)으로 테스트하세요.

### 연습 문제 3: 환경 변수 출력기

명령줄 인수로 변수 이름을 받는 프로그램을 작성하세요. 변수가 존재하면 값을 출력합니다. 존재하지 않으면 사용자에게 값을 입력받아 `setenv`로 설정한 다음, `fork` + `exec`으로 `env` 명령을 실행하여 자식이 새 변수를 상속받았음을 증명하세요.

### 연습 문제 4: 좀비 감지기

자식을 fork하고, 자식은 즉시 종료하지만 부모는 `wait`를 호출하기 전에 30초간 대기하는 프로그램을 작성하세요. 부모가 대기하는 동안 다른 터미널을 열고 `ps aux | grep Z`를 사용하여 좀비를 관찰하세요. 그런 다음 프로그램을 수정하여 자식을 즉시 회수하도록 하세요.

### 연습 문제 5: 명령 파이프라인

두 명령의 파이프라인 (`ls | wc -l`과 같은)을 구현하는 프로그램을 작성하세요. `pipe()`, `fork()`, `dup2()`를 사용하여 첫 번째 명령의 stdout을 두 번째 명령의 stdin에 연결하세요. 부모는 두 자식 모두를 기다려야 합니다.

---

## 다음 단계

프로세스 관리에 대한 확실한 이해를 바탕으로, fork, exec, 파이프, 리다이렉션을 결합하여 완전한 기능을 갖춘 명령 인터프리터인 [미니 셸](./10_Project_Mini_Shell.md)을 구축할 준비가 되었습니다.
