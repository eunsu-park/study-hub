# 프로세스 간 통신(Inter-Process Communication)과 시그널(Signals)

**이전**: [C 네트워크 프로그래밍](./21_Network_Programming.md) | **다음**: [C 테스팅과 프로파일링](./23_Testing_and_Profiling.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 부모-자식 통신을 위한 익명 파이프(Anonymous Pipe)를 생성하고 `dup2`로 표준 입출력을 재연결합니다
2. 명명된 파이프(Named Pipe, FIFO)를 사용하여 관련 없는 프로세스 간 데이터를 교환합니다
3. `shm_open`과 `mmap`으로 공유 메모리(Shared Memory) 영역을 매핑하고, POSIX 세마포어(Semaphore)로 접근을 동기화합니다
4. 우선순위 정렬을 지원하는 POSIX 메시지 큐(Message Queue)를 통해 구조화된 메시지를 송수신합니다
5. `sigaction`으로 시그널 핸들러를 설치하고 비동기 시그널 안전(Async-Signal-Safe) 코드를 작성합니다
6. `sigprocmask`로 시그널을 차단 및 해제하여 임계 구역(Critical Section)을 보호합니다
7. 비차단(Non-Blocking) `waitpid`를 사용하는 `SIGCHLD` 핸들러로 자식 프로세스를 자동으로 수거합니다

---

단일 프로세스로는 부족할 때 — 워커들의 파이프라인이 필요하거나, 충돌한 서비스를 재시작하는 감시자(Watchdog)가 필요하거나, 협력하는 데몬들 간의 공유 스코어보드가 필요할 때 — 프로세스 간 통신(IPC)이 필요합니다. IPC는 모든 Unix 쉘 파이프라인, 여러 백엔드를 조율하는 모든 데이터베이스, 그리고 워커 프로세스를 관리하는 모든 컨테이너 오케스트레이터의 배관입니다.

> **비유 — 문자 메시지, 화이트보드, 우편함**: 프로세스 간 통신 메커니즘은 일상의 소통 방식에 비유할 수 있습니다. **시그널(Signal)**은 긴급 문자 알림과 같습니다 — 짧은 코드(SIGTERM = "멈춰주세요")로 하던 일을 중단시킵니다. **공유 메모리(Shared Memory)**는 공유 사무실의 화이트보드와 같습니다 — 어떤 프로세스든 즉시 읽거나 쓸 수 있지만, 누군가 문장 중간에 다른 사람의 내용을 지우지 않으려면 규칙(세마포어)이 필요합니다. **메시지 큐(Message Queue)**는 우편함과 같습니다 — 발신자는 편지를 넣고 수신자는 순서대로 가져가며, 도착 시간이 달라도 괜찮습니다.

**난이도**: 고급

---

## 목차

1. [파이프](#1-파이프)
2. [명명된 파이프 (FIFOs)](#2-명명된-파이프-fifos)
3. [공유 메모리](#3-공유-메모리)
4. [POSIX 메시지 큐](#4-posix-메시지-큐)
5. [시그널](#5-시그널)
6. [연습 문제](#6-연습-문제)
7. [참고 자료](#7-참고-자료)

---

## 1. 파이프

### 1.1 익명 파이프(Anonymous Pipes)

파이프는 관련된 프로세스(부모-자식) 간에 단방향 데이터 흐름을 제공합니다.

```
┌────────────────────────────────────────────────────────┐
│                  Pipe Communication                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌───────────┐    pipe    ┌───────────┐               │
│  │  Parent   │───────────▶│  Child    │               │
│  │  (Writer) │   fd[1]    │  (Reader) │               │
│  │           │  ────────▶ │           │               │
│  └───────────┘   fd[0]    └───────────┘               │
│                                                        │
│  pipe(fd) creates:                                     │
│    fd[0] = read end                                    │
│    fd[1] = write end                                   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

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

### 1.2 두 개의 파이프를 사용한 양방향 통신

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

### 1.3 exec와 함께 사용하는 파이프(Shell-like Piping, 셸 파이핑)

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
        dup2(pipefd[1], STDOUT_FILENO);  // stdout → pipe write
        close(pipefd[1]);
        execlp("ls", "ls", "-la", NULL);
        perror("execlp ls");
        exit(1);
    }

    pid_t pid2 = fork();
    if (pid2 == 0) {
        // Second child: grep ".c"
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);  // stdin ← pipe read
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

## 2. 명명된 파이프(Named Pipes, FIFOs)

FIFO는 파일 시스템 엔트리를 통해 관련 없는 프로세스 간 통신을 가능하게 합니다.

### 2.1 FIFO 생성 및 사용

```c
// --- Writer process ---
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
// --- Reader process ---
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

## 3. 공유 메모리(Shared Memory)

공유 메모리는 프로세스 간에 데이터를 복사할 필요가 없기 때문에 가장 빠른 IPC 메커니즘입니다.

### 3.1 POSIX 공유 메모리

```
┌──────────────────────────────────────────────────────────┐
│              Shared Memory Architecture                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐       Physical Memory       ┌─────────┐│
│  │  Process A  │       ┌──────────────┐      │Process B ││
│  │             │       │              │      │          ││
│  │  Virtual    │──────▶│  Shared      │◀─────│ Virtual  ││
│  │  Address    │  mmap │  Region      │ mmap │ Address  ││
│  │  0x7f...    │       │              │      │ 0x7f...  ││
│  │             │       └──────────────┘      │          ││
│  └─────────────┘                             └─────────┘│
│                                                          │
│  ⚠ Requires synchronization (semaphore/mutex)           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

```c
// --- Producer ---
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
// --- Consumer ---
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

## 4. POSIX 메시지 큐(Message Queues)

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

// Sender
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

// Receiver
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

## 5. 시그널(Signals)

### 5.1 시그널 개요

시그널은 프로세스에 이벤트를 알리기 위해 전달되는 소프트웨어 인터럽트입니다.

```
┌──────────────────────────────────────────────────────────┐
│  Common Signals                                          │
├─────────┬──────────────────────────────────────────────┤
│ Signal  │ Description                                    │
├─────────┼──────────────────────────────────────────────┤
│ SIGINT  │ Interrupt (Ctrl+C)                             │
│ SIGTERM │ Termination request                            │
│ SIGKILL │ Forced kill (cannot be caught)                 │
│ SIGCHLD │ Child process stopped or terminated            │
│ SIGUSR1 │ User-defined signal 1                          │
│ SIGUSR2 │ User-defined signal 2                          │
│ SIGALRM │ Timer alarm                                    │
│ SIGPIPE │ Broken pipe (write to closed socket)           │
│ SIGSEGV │ Segmentation fault                             │
│ SIGSTOP │ Stop process (cannot be caught)                │
│ SIGCONT │ Continue stopped process                       │
└─────────┴──────────────────────────────────────────────┘
```

### 5.2 sigaction을 사용한 시그널 처리

이식성과 신뢰성 있는 동작을 위해 `signal()` 대신 항상 `sigaction()`을 사용하세요.

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

### 5.3 시그널 마스킹(Signal Masking)

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

    // ---- Critical section ----
    printf("SIGINT blocked. Ctrl+C won't interrupt.\n");
    sleep(5);
    printf("Critical section done.\n");
    // ---- End critical section ----

    // Restore original mask
    sigprocmask(SIG_SETMASK, &old_set, NULL);
    printf("SIGINT unblocked. Ctrl+C works again.\n");

    sleep(5);
    return 0;
}
```

### 5.4 SIGCHLD를 사용한 자식 프로세스 수거(Reaping)

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

### 문제 1: 공유 메모리를 사용한 생산자-소비자(Producer-Consumer)

POSIX 공유 메모리와 세마포어를 사용하여 생산자-소비자 시스템을 구현하세요:
- 생산자는 공유 메모리의 순환 버퍼(Circular Buffer)에 정수 1-100을 씁니다
- 소비자는 그것을 읽고 출력합니다
- 동기화를 위해 세마포어를 사용합니다

### 문제 2: 다중 프로세스 파이프라인

파이프를 사용하여 3단계 파이프라인을 만드세요:
- 1단계: 파일에서 줄을 읽습니다
- 2단계: 대문자로 변환합니다
- 3단계: 단어 빈도를 계산하고 출력합니다

### 문제 3: 워치독 프로세스(Watchdog Process)

다음 기능을 가진 워치독을 작성하세요:
- 자식 워커 프로세스를 포크합니다
- SIGCHLD로 모니터링합니다
- 충돌 시 자동으로 재시작합니다
- 두 프로세스 모두의 우아한 종료(Graceful Shutdown)를 위해 SIGTERM을 처리합니다

---

## 7. 참고 자료

- W. Richard Stevens, *Advanced Programming in the UNIX Environment* (3rd ed.)
- `man 7 pipe`, `man 7 fifo`, `man 7 shm_overview`, `man 7 mq_overview`
- `man 7 signal`, `man 2 sigaction`, `man 2 sigprocmask`

---

## 연습 문제

### 연습 1: 파이프 체인(Pipe Chain) 추적

세 개의 프로세스 P1, P2, P3를 두 개의 익명 파이프(anonymous pipe)로 연결하는 프로그램을 작성하세요:

1. P1은 숫자 1부터 10을 생성하여 각각을 텍스트(예: `"1\n"`)로 파이프 A에 씁니다.
2. P2는 파이프 A에서 읽고, 각 숫자를 두 배로 만들어 파이프 B에 씁니다.
3. P3는 파이프 B에서 읽어 각 결과를 stdout에 출력합니다.

각 프로세스는 `fork` 직후에 사용하지 않는 파이프 끝(pipe end)을 반드시 닫아야 합니다. 코드를 작성한 후 다음에 답하세요: P1이 아무것도 쓰지 않고 쓰기 끝(write end)을 닫으면 P2와 P3는 어떻게 되나요? 실험으로 답을 확인하세요.

### 연습 2: FIFO 기반 로거(Logger)

명명된 파이프(named pipe, FIFO)를 사용하는 두 프로그램으로 구성된 로깅 시스템을 구현하세요:

1. **로거 데몬(Logger daemon)**: `mkfifo`로 `/tmp/app_log.fifo`를 생성하고 읽기용으로 열어, 로그 줄을 지속적으로 읽으며 타임스탬프(`[HH:MM:SS]`)를 앞에 붙여 `app.log` 파일에 씁니다.
2. **애플리케이션(Application)**: `/tmp/app_log.fifo`를 쓰기용으로 열어 1초 간격으로 5개의 로그 메시지를 전송한 후 FIFO를 닫습니다.
3. 애플리케이션이 종료된 후, 로거는 EOF(End Of File)를 감지하고 `"Log complete"`를 출력한 뒤 `unlink`로 정리합니다.

두 프로그램을 별도의 터미널에서 실행하고, `app.log`에 다섯 개의 타임스탬프가 있는 메시지가 모두 포함되어 있는지 확인하세요.

### 연습 3: 세마포어(Semaphore) 보호를 사용한 공유 카운터(Shared Counter)

경쟁 조건(race condition)을 실연하고 POSIX 세마포어(POSIX semaphore)로 수정하세요:

1. 0으로 초기화된 단일 `int counter`를 포함하는 공유 메모리(shared memory) 영역을 생성하세요.
2. 4개의 자식 프로세스(child process)를 포크(fork)하고, 각각이 **동기화 없이** 카운터를 10,000번 증가시키도록 하세요. 실행 후 경쟁 조건으로 인해 최종 값이 종종 40,000보다 적다는 것을 관찰하세요.
3. 1로 초기화된 명명된 세마포어(named semaphore)를 추가하세요. 각 증가 연산을 `sem_wait` / `sem_post`로 감싸세요. 다시 실행하여 최종 값이 정확히 40,000임을 확인하세요.
4. 모든 자식이 종료된 후 부모에서 모든 공유 리소스(`shm_unlink`, `sem_unlink`)를 정리하세요.

### 연습 4: 파이프 셀프 트릭(Self-Pipe Trick)을 사용한 안전한 시그널 핸들러

5.2절의 `SIGINT` 핸들러를 "셀프 파이프 트릭(self-pipe trick)"을 사용하여 시그널 안전 I/O 다중화(signal-safe I/O multiplexing)로 재작성하세요:

1. 메인 루프 진입 전에 `pipe(selfpipe)`를 생성하세요.
2. `sigaction`으로 설치된 `SIGINT` 시그널 핸들러에서, `selfpipe[1]`에 단일 바이트 `'S'`를 씁니다 — 이것은 비동기 시그널 안전(async-signal-safe)합니다.
3. 메인 루프에서 `select()` 또는 `poll()`을 사용하여 `STDIN_FILENO`와 `selfpipe[0]` 모두를 대기하세요.
4. `selfpipe[0]`가 읽기 가능해지면, 바이트를 읽고, `"Graceful shutdown initiated"`를 출력하고, 종료하세요.

시그널 핸들러 내부에서 직접 `printf`를 호출하는 것이 왜 안전하지 않은지, 그리고 셀프 파이프 트릭이 어떤 문제를 방지하는지 주석 블록으로 설명하세요.

### 연습 5: POSIX 메시지 큐(Message Queue) 우선순위 스케줄러

POSIX 메시지 큐를 사용하는 간단한 우선순위 기반 태스크 스케줄러를 구축하세요:

1. `char description[128]`과 `int priority`(1 = 낮음, 5 = 보통, 10 = 높음) 필드를 포함하는 `task_t` 구조체를 정의하세요.
2. 다양한 우선순위와 설명을 가진 6개의 태스크를 무작위 순서로 큐에 넣는 생산자(producer)를 작성하세요.
3. 태스크를 하나씩 꺼내는(가장 높은 우선순위 먼저, `mq_receive`가 보장하는 대로) 소비자(consumer)를 작성하고, 각 태스크의 우선순위와 설명을 출력하세요.
4. 생산자가 전송한 순서와 관계없이, 소비자가 항상 높은 우선순위 태스크를 먼저 처리하는지 확인하세요.
5. 모든 태스크가 처리된 후 `mq_unlink`로 정리하세요.

---

**이전**: [C 네트워크 프로그래밍](./21_Network_Programming.md) | **다음**: [C 테스팅과 프로파일링](./23_Testing_and_Profiling.md)
