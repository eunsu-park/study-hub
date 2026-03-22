# 프로세스 간 통신과 시그널

**이전**: [네트워크 프로그래밍](./12_Network_Programming.md) | **다음**: [임베디드 시스템](./14_Embedded_Systems.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 부모-자식 통신을 위한 익명 파이프를 생성하고 `dup2`로 표준 I/O를 리다이렉트할 수 있다
2. 명명된 파이프(FIFO)를 사용하여 관련 없는 프로세스 간에 데이터를 교환할 수 있다
3. `shm_open`과 `mmap`으로 공유 메모리 영역을 매핑하고 POSIX 세마포어로 접근을 동기화할 수 있다
4. POSIX 메시지 큐를 통해 우선순위 순서로 구조화된 메시지를 보내고 받을 수 있다
5. `sigaction`을 사용하여 시그널 핸들러를 설치하고 비동기 시그널 안전(async-signal-safe) 코드를 작성할 수 있다
6. `sigprocmask`로 시그널을 차단하고 해제하여 임계 구역을 보호할 수 있다
7. `SIGCHLD` 핸들러와 비차단 `waitpid`를 사용하여 자식 프로세스를 자동으로 회수할 수 있다

---

단일 프로세스로는 충분하지 않을 때 -- 워커 파이프라인, 충돌한 서비스를 재시작하는 감시자, 협력하는 데몬들 사이의 공유 스코어보드가 필요할 때 -- 프로세스 간 통신이 필요합니다. IPC는 모든 Unix 셸 파이프라인, 여러 백엔드를 조율하는 모든 데이터베이스, 워커 프로세스를 관리하는 모든 컨테이너 오케스트레이터 뒤의 배관입니다.

> **비유 -- 문자, 화이트보드, 우편함**: 프로세스 간 통신 메커니즘은 일상적인 통신과 대응됩니다. **시그널**은 긴급 문자 알림과 같습니다 -- 짧은 코드로 하던 일을 중단시킵니다 (SIGTERM = "멈춰 주세요"). **공유 메모리**는 공유 사무실의 화이트보드입니다 -- 어떤 프로세스든 즉시 읽거나 쓸 수 있지만, 누군가의 작성 중에 다른 사람이 지우지 않도록 규칙(세마포어)이 필요합니다. **메시지 큐**는 우편함입니다: 발신자가 편지를 넣고 수신자가 순서대로 가져가며, 서로 다른 시간에 도착해도 됩니다.

**난이도**: 고급

---

## 목차

1. [파이프](#1-파이프)
2. [명명된 파이프 (FIFO)](#2-명명된-파이프-fifo)
3. [공유 메모리](#3-공유-메모리)
4. [POSIX 메시지 큐](#4-posix-메시지-큐)
5. [시그널](#5-시그널)
6. [연습 문제](#6-연습-문제)
7. [참고 자료](#7-참고-자료)

---

## 1. 파이프

### 1.1 익명 파이프

파이프는 관련 프로세스(부모-자식) 간에 단방향 데이터 흐름을 제공합니다.

> **파이프 통신**
>
> - 부모 (쓰기자) -> `fd[1]` -> 파이프 -> `fd[0]` -> 자식 (읽기자)
> - `pipe(fd)` 생성: `fd[0]` = 읽기 끝, `fd[1]` = 쓰기 끝

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // Child: read from pipe
        close(pipefd[1]);  // Close unused write end

        char buffer[256];
        ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);
        if (n > 0) {
            buffer[n] = '\0';
            printf("Child received: %s\n", buffer);
        }

        close(pipefd[0]);
        exit(EXIT_SUCCESS);
    } else {
        // Parent: write to pipe
        close(pipefd[0]);  // Close unused read end

        const char *msg = "Hello from parent!";
        write(pipefd[1], msg, strlen(msg));

        close(pipefd[1]);
        wait(NULL);  // Wait for child
    }

    return 0;
}
```

### 1.2 두 개의 파이프를 이용한 양방향 통신

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int parent_to_child[2], child_to_parent[2];
    pipe(parent_to_child);
    pipe(child_to_parent);

    pid_t pid = fork();
    if (pid == 0) {
        // Child
        close(parent_to_child[1]);
        close(child_to_parent[0]);

        char buf[256];
        ssize_t n = read(parent_to_child[0], buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Child got: %s\n", buf);

        const char *reply = "Got it, thanks!";
        write(child_to_parent[1], reply, strlen(reply));

        close(parent_to_child[0]);
        close(child_to_parent[1]);
        exit(0);
    }

    // Parent
    close(parent_to_child[0]);
    close(child_to_parent[1]);

    const char *msg = "Task: process data";
    write(parent_to_child[1], msg, strlen(msg));
    close(parent_to_child[1]);

    char buf[256];
    ssize_t n = read(child_to_parent[0], buf, sizeof(buf) - 1);
    buf[n] = '\0';
    printf("Parent got reply: %s\n", buf);

    close(child_to_parent[0]);
    wait(NULL);
    return 0;
}
```

### 1.3 exec과 함께 사용하는 파이프 (셸 방식 파이핑)

```c
// Simulate: ls -la | grep ".c"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int pipefd[2];
    pipe(pipefd);

    pid_t pid1 = fork();
    if (pid1 == 0) {
        // First child: ls -la
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);  // stdout -> pipe write
        close(pipefd[1]);
        execlp("ls", "ls", "-la", NULL);
        perror("execlp ls");
        exit(1);
    }

    pid_t pid2 = fork();
    if (pid2 == 0) {
        // Second child: grep ".c"
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);  // stdin <- pipe read
        close(pipefd[0]);
        execlp("grep", "grep", ".c", NULL);
        perror("execlp grep");
        exit(1);
    }

    // Parent: close both ends and wait
    close(pipefd[0]);
    close(pipefd[1]);
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);

    return 0;
}
```

---

## 2. 명명된 파이프 (FIFO)

FIFO는 파일시스템 항목을 통해 관련 없는 프로세스 간 통신을 허용합니다.

### 2.1 FIFO 생성 및 사용

```c
// --- 쓰기 프로세스 ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define FIFO_PATH "/tmp/myfifo"

int main(void) {
    // Create FIFO (ignore error if it already exists)
    mkfifo(FIFO_PATH, 0666);

    int fd = open(FIFO_PATH, O_WRONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    const char *messages[] = {"Hello", "World", "Done"};
    for (int i = 0; i < 3; i++) {
        write(fd, messages[i], strlen(messages[i]) + 1);
        printf("Sent: %s\n", messages[i]);
        sleep(1);
    }

    close(fd);
    return 0;
}
```

```c
// --- 읽기 프로세스 ---
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define FIFO_PATH "/tmp/myfifo"

int main(void) {
    int fd = open(FIFO_PATH, O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    char buffer[256];
    ssize_t n;
    while ((n = read(fd, buffer, sizeof(buffer))) > 0) {
        printf("Received: %s\n", buffer);
    }

    close(fd);
    unlink(FIFO_PATH);  // Clean up
    return 0;
}
```

---

## 3. 공유 메모리

공유 메모리는 데이터가 프로세스 간에 복사될 필요가 없기 때문에 가장 빠른 IPC 메커니즘입니다.

### 3.1 POSIX 공유 메모리

```c
// --- 생산자 ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME "/my_shm"
#define SEM_NAME "/my_sem"
#define SHM_SIZE 4096

typedef struct {
    int count;
    char data[256];
} shared_data_t;

int main(void) {
    // Create shared memory
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, sizeof(shared_data_t));

    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, shm_fd, 0);

    // Create semaphore for synchronization
    sem_t *sem = sem_open(SEM_NAME, O_CREAT, 0666, 0);

    // Write data
    shm->count = 42;
    snprintf(shm->data, sizeof(shm->data),
             "Hello from producer (PID=%d)", getpid());

    printf("Producer wrote: count=%d, data=%s\n",
           shm->count, shm->data);

    // Signal consumer
    sem_post(sem);

    // Cleanup
    sem_close(sem);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);

    return 0;
}
```

```c
// --- 소비자 ---
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME "/my_shm"
#define SEM_NAME "/my_sem"

typedef struct {
    int count;
    char data[256];
} shared_data_t;

int main(void) {
    // Open shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ, MAP_SHARED, shm_fd, 0);

    // Wait for producer
    sem_t *sem = sem_open(SEM_NAME, 0);
    sem_wait(sem);

    // Read data
    printf("Consumer read: count=%d, data=%s\n",
           shm->count, shm->data);

    // Cleanup
    sem_close(sem);
    sem_unlink(SEM_NAME);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);
    shm_unlink(SHM_NAME);

    return 0;
}
```

---

## 4. POSIX 메시지 큐

메시지 큐는 우선순위를 지원하는 구조화된 메시지 전달을 제공합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mqueue.h>
#include <fcntl.h>

#define QUEUE_NAME "/my_queue"
#define MAX_MSG_SIZE 256
#define MAX_MSGS 10

// 송신자
void sender(void) {
    struct mq_attr attr = {
        .mq_flags = 0,
        .mq_maxmsg = MAX_MSGS,
        .mq_msgsize = MAX_MSG_SIZE,
        .mq_curmsgs = 0
    };

    mqd_t mq = mq_open(QUEUE_NAME, O_CREAT | O_WRONLY, 0666, &attr);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        exit(1);
    }

    const char *msgs[] = {"High priority!", "Normal message", "Low priority"};
    unsigned int priorities[] = {10, 5, 1};

    for (int i = 0; i < 3; i++) {
        mq_send(mq, msgs[i], strlen(msgs[i]) + 1, priorities[i]);
        printf("Sent (prio=%u): %s\n", priorities[i], msgs[i]);
    }

    mq_close(mq);
}

// 수신자
void receiver(void) {
    mqd_t mq = mq_open(QUEUE_NAME, O_RDONLY);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        exit(1);
    }

    char buffer[MAX_MSG_SIZE];
    unsigned int priority;

    // Messages arrive highest priority first
    for (int i = 0; i < 3; i++) {
        ssize_t bytes = mq_receive(mq, buffer, MAX_MSG_SIZE, &priority);
        if (bytes >= 0) {
            printf("Received (prio=%u): %s\n", priority, buffer);
        }
    }

    mq_close(mq);
    mq_unlink(QUEUE_NAME);
}
```

---

## 5. 시그널

### 5.1 시그널 개요

| 시그널 | 설명 |
|--------|------|
| SIGINT | 인터럽트 (Ctrl+C) |
| SIGTERM | 종료 요청 |
| SIGKILL | 강제 종료 (포착 불가) |
| SIGCHLD | 자식 프로세스 정지 또는 종료 |
| SIGUSR1 | 사용자 정의 시그널 1 |
| SIGUSR2 | 사용자 정의 시그널 2 |
| SIGALRM | 타이머 알람 |
| SIGPIPE | 깨진 파이프 (닫힌 소켓에 쓰기) |
| SIGSEGV | 세그멘테이션 오류 |
| SIGSTOP | 프로세스 정지 (포착 불가) |
| SIGCONT | 정지된 프로세스 계속 |

### 5.2 sigaction을 사용한 시그널 처리

이식 가능하고 신뢰할 수 있는 동작을 위해 항상 `signal()` 대신 `sigaction()`을 사용하세요.

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

volatile sig_atomic_t running = 1;

void handle_sigint(int sig) {
    (void)sig;  // Suppress unused warning
    running = 0;
    // Only async-signal-safe functions here!
    write(STDOUT_FILENO, "\nCaught SIGINT, shutting down...\n", 33);
}

void handle_sigusr1(int sig, siginfo_t *info, void *context) {
    (void)sig;
    (void)context;
    // siginfo_t gives us sender information
    printf("SIGUSR1 from PID %d\n", info->si_pid);
}

int main(void) {
    // Setup SIGINT handler
    struct sigaction sa_int = {0};
    sa_int.sa_handler = handle_sigint;
    sigemptyset(&sa_int.sa_mask);
    sa_int.sa_flags = 0;
    sigaction(SIGINT, &sa_int, NULL);

    // Setup SIGUSR1 handler with siginfo
    struct sigaction sa_usr = {0};
    sa_usr.sa_sigaction = handle_sigusr1;
    sigemptyset(&sa_usr.sa_mask);
    sa_usr.sa_flags = SA_SIGINFO;
    sigaction(SIGUSR1, &sa_usr, NULL);

    // Ignore SIGPIPE (common in network programs)
    signal(SIGPIPE, SIG_IGN);

    printf("PID: %d - Press Ctrl+C or send SIGUSR1\n", getpid());

    while (running) {
        printf("Working...\n");
        sleep(2);
    }

    printf("Clean shutdown complete\n");
    return 0;
}
```

### 5.3 시그널 마스킹

```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    sigset_t block_set, old_set;

    // Block SIGINT during critical section
    sigemptyset(&block_set);
    sigaddset(&block_set, SIGINT);

    sigprocmask(SIG_BLOCK, &block_set, &old_set);

    // ---- 임계 구역 ----
    printf("SIGINT blocked. Ctrl+C won't interrupt.\n");
    sleep(5);
    printf("Critical section done.\n");
    // ---- 임계 구역 끝 ----

    // Restore original mask
    sigprocmask(SIG_SETMASK, &old_set, NULL);
    printf("SIGINT unblocked. Ctrl+C works again.\n");

    sleep(5);
    return 0;
}
```

### 5.4 SIGCHLD를 사용한 자식 프로세스 회수

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

void handle_sigchld(int sig) {
    (void)sig;
    // Reap all terminated children (non-blocking)
    int status;
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        if (WIFEXITED(status)) {
            // Child exited normally
        }
    }
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handle_sigchld;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    // Fork multiple children
    for (int i = 0; i < 5; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            printf("Child %d (PID=%d) working...\n", i, getpid());
            sleep(i + 1);
            printf("Child %d done\n", i);
            exit(i);
        }
    }

    // Parent continues working
    printf("Parent (PID=%d) waiting...\n", getpid());
    sleep(10);
    printf("Parent done\n");

    return 0;
}
```

---

## 6. 연습 문제

### 연습 문제 1: 파이프 체인 추적

세 개의 프로세스 -- P1, P2, P3 -- 를 두 개의 익명 파이프로 연결하는 프로그램을 작성하세요:

1. P1은 숫자 1~10을 생성하여 각각을 텍스트(예: `"1\n"`)로 파이프 A에 씁니다.
2. P2는 파이프 A에서 읽고, 각 숫자를 2배로 만들어 파이프 B에 씁니다.
3. P3는 파이프 B에서 읽고 각 결과를 stdout에 출력합니다.

각 프로세스는 `fork` 직후에 사용하지 않는 파이프 끝을 즉시 닫아야 합니다. 코드를 작성한 후 답하세요: P1이 아무것도 쓰지 않고 쓰기 끝을 닫으면 P2와 P3에 무슨 일이 일어납니까? 실험적으로 답을 확인하세요.

### 연습 문제 2: FIFO 기반 로거

명명된 파이프(FIFO)를 사용하는 두 프로그램 로깅 시스템을 구현하세요:

1. **로거 데몬**: `mkfifo`로 `/tmp/app_log.fifo`를 생성하고, 읽기 모드로 열어 로그 라인을 계속 읽으며, 타임스탬프(`[HH:MM:SS]`)를 앞에 붙여 `app.log` 파일에 씁니다.
2. **애플리케이션**: `/tmp/app_log.fifo`를 쓰기 모드로 열고 1초 간격으로 5개의 로그 메시지를 보낸 후 FIFO를 닫습니다.
3. 애플리케이션 종료 후, 로거는 EOF를 감지하고 `unlink`로 정리하기 전에 `"Log complete"`를 출력해야 합니다.

별도의 터미널에서 두 프로그램을 실행하고 `app.log`에 5개의 타임스탬프 메시지가 모두 포함되어 있는지 확인하세요.

### 연습 문제 3: 세마포어 보호가 적용된 공유 카운터

경쟁 조건을 시연한 후 POSIX 세마포어로 수정하세요:

1. 0으로 초기화된 단일 `int counter`를 포함하는 공유 메모리 영역을 생성합니다.
2. 4개의 자식 프로세스를 fork하고, 각각 동기화 **없이** 카운터를 10,000번 증가시킵니다. 실행하고 최종 값이 경쟁으로 인해 40,000보다 작은 경우가 많음을 관찰합니다.
3. 1로 초기화된 명명된 세마포어를 추가합니다. 각 증가를 `sem_wait` / `sem_post`로 감쌉니다. 다시 실행하고 최종 값이 정확히 40,000인지 확인합니다.
4. 모든 자식이 종료된 후 부모에서 모든 공유 자원(`shm_unlink`, `sem_unlink`)을 정리합니다.

### 연습 문제 4: 파이프 셀프 트릭을 사용한 안전한 시그널 핸들러

섹션 5.2의 `SIGINT` 핸들러를 시그널 안전 I/O 다중화를 위한 "셀프 파이프 트릭"을 사용하여 다시 작성하세요:

1. 메인 루프에 들어가기 전에 `pipe(selfpipe)`를 생성합니다.
2. `SIGINT` 시그널 핸들러(`sigaction`으로 설치)에서 `selfpipe[1]`에 단일 바이트 `'S'`를 씁니다 -- 이것은 비동기 시그널 안전합니다.
3. 메인 루프에서 `select()` 또는 `poll()`을 사용하여 `STDIN_FILENO`와 `selfpipe[0]`을 모두 대기합니다.
4. `selfpipe[0]`이 읽기 가능해지면 바이트를 읽고 `"Graceful shutdown initiated"`를 출력한 후 종료합니다.

주석 블록에서 시그널 핸들러 내부에서 `printf`를 직접 호출하는 것이 왜 안전하지 않은지, 셀프 파이프 트릭이 어떤 문제를 피하는지 설명하세요.

### 연습 문제 5: POSIX 메시지 큐 우선순위 스케줄러

POSIX 메시지 큐를 사용하여 간단한 우선순위 기반 작업 스케줄러를 구축하세요:

1. `char description[128]`과 `int priority` (1 = 낮음, 5 = 중간, 10 = 높음)를 포함하는 `task_t` 구조체를 정의합니다.
2. 다양한 우선순위와 설명을 가진 6개의 작업을 임의의 순서로 큐에 넣는 생산자를 작성합니다.
3. 한 번에 하나씩 작업을 큐에서 꺼내는 (가장 높은 우선순위 먼저, `mq_receive`가 보장) 소비자를 작성하고 각 작업의 우선순위와 설명을 출력합니다.
4. 생산자가 보낸 순서에 관계없이 소비자가 항상 높은 우선순위 작업을 먼저 처리하는지 확인합니다.
5. 모든 작업이 처리된 후 `mq_unlink`로 정리합니다.

---

## 7. 참고 자료

- W. Richard Stevens, *Advanced Programming in the UNIX Environment* (3rd ed.)
- `man 7 pipe`, `man 7 fifo`, `man 7 shm_overview`, `man 7 mq_overview`
- `man 7 signal`, `man 2 sigaction`, `man 2 sigprocmask`

---
