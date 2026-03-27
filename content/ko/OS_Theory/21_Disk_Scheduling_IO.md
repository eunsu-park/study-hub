[이전: 고급 가상 메모리](./20_Advanced_Virtual_Memory.md)

---

# 21. 디스크 스케줄링과 현대 I/O

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 전통적인 디스크 스케줄링 알고리즘과 그 트레이드오프를 설명할 수 있다
2. NVMe 아키텍처와 이것이 I/O 스케줄링을 어떻게 변화시키는지 설명할 수 있다
3. 고성능 비동기 I/O를 위한 io_uring을 구현할 수 있다
4. 블로킹, 논블로킹, epoll, io_uring I/O 패턴을 비교할 수 있다
5. Linux 프로파일링 도구를 사용하여 I/O 성능을 분석할 수 있다

---

## 목차

1. [전통적인 디스크 스케줄링](#1-전통적인-디스크-스케줄링)
2. [Linux의 I/O 스케줄러](#2-linux의-io-스케줄러)
3. [NVMe와 현대 스토리지](#3-nvme와-현대-스토리지)
4. [비동기 I/O 패턴](#4-비동기-io-패턴)
5. [io_uring 심층 분석](#5-io_uring-심층-분석)
6. [I/O 성능 분석](#6-io-성능-분석)
7. [Direct I/O와 Zero-Copy](#7-direct-io와-zero-copy)
8. [연습문제](#8-연습문제)

---

## 1. 전통적인 디스크 스케줄링

### 1.1 HDD 구조와 접근 시간

```
HDD 접근 시간 = 탐색 시간 + 회전 지연 + 전송 시간

탐색 시간: 읽기/쓰기 헤드를 올바른 트랙으로 이동 (0.5-15 ms)
회전 지연: 섹터가 헤드 아래로 회전할 때까지 대기 (0-8 ms @ 7200 RPM)
전송 시간: 데이터 읽기/쓰기 (일반적으로 < 1 ms)

합계: 랜덤 접근당 4-20 ms (SSD의 경우 ~0.1 ms!)

이것이 HDD에서 디스크 스케줄링이 중요한 이유:
  탐색 거리 최소화 = 헤드 이동 최소화 = 더 빠른 I/O
```

### 1.2 스케줄링 알고리즘

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_REQUESTS 100

/*
 * FCFS (First Come First Served):
 * 도착 순서대로 요청 처리.
 * 단순하지만 긴 탐색 거리가 발생할 수 있음.
 */
int fcfs_schedule(int *requests, int n, int head_pos) {
    int total_movement = 0;
    int current = head_pos;

    printf("FCFS: %d", current);
    for (int i = 0; i < n; i++) {
        total_movement += abs(requests[i] - current);
        current = requests[i];
        printf(" -> %d", current);
    }
    printf("\nTotal movement: %d cylinders\n", total_movement);
    return total_movement;
}

/*
 * SSTF (Shortest Seek Time First):
 * 항상 가장 가까운 요청을 다음에 처리.
 * FCFS보다 좋지만 기아 상태가 발생할 수 있음.
 */
int sstf_schedule(int *requests, int n, int head_pos) {
    int total_movement = 0;
    int current = head_pos;
    int serviced[MAX_REQUESTS] = {0};

    printf("SSTF: %d", current);
    for (int i = 0; i < n; i++) {
        int min_dist = __INT_MAX__;
        int min_idx = -1;

        for (int j = 0; j < n; j++) {
            if (!serviced[j]) {
                int dist = abs(requests[j] - current);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
        }

        serviced[min_idx] = 1;
        total_movement += min_dist;
        current = requests[min_idx];
        printf(" -> %d", current);
    }
    printf("\nTotal movement: %d cylinders\n", total_movement);
    return total_movement;
}

/*
 * SCAN (엘리베이터 알고리즘):
 * 헤드가 한 방향으로 이동하며 요청 처리.
 * 끝에서 방향 반전.
 */
int compare_int(const void *a, const void *b) {
    return *(int *)a - *(int *)b;
}

int scan_schedule(int *requests, int n, int head_pos, int max_cylinder) {
    int sorted[MAX_REQUESTS];
    for (int i = 0; i < n; i++) sorted[i] = requests[i];
    qsort(sorted, n, sizeof(int), compare_int);

    int total_movement = 0;
    int current = head_pos;

    printf("SCAN: %d", current);

    /* 먼저 오른쪽으로 이동 */
    for (int i = 0; i < n; i++) {
        if (sorted[i] >= current) {
            total_movement += abs(sorted[i] - current);
            current = sorted[i];
            printf(" -> %d", current);
        }
    }

    /* 끝까지 이동 */
    total_movement += abs(max_cylinder - current);
    current = max_cylinder;

    /* 왼쪽으로 이동 */
    for (int i = n - 1; i >= 0; i--) {
        if (sorted[i] < head_pos) {
            total_movement += abs(sorted[i] - current);
            current = sorted[i];
            printf(" -> %d", current);
        }
    }

    printf("\nTotal movement: %d cylinders\n", total_movement);
    return total_movement;
}

int main(void) {
    int requests[] = {98, 183, 37, 122, 14, 124, 65, 67};
    int n = 8;
    int head_pos = 53;
    int max_cylinder = 199;

    printf("Requests: ");
    for (int i = 0; i < n; i++) printf("%d ", requests[i]);
    printf("\nInitial head position: %d\n\n", head_pos);

    fcfs_schedule(requests, n, head_pos);
    printf("\n");
    sstf_schedule(requests, n, head_pos);
    printf("\n");
    scan_schedule(requests, n, head_pos, max_cylinder);

    return 0;
}
```

---

## 2. Linux의 I/O 스케줄러

### 2.1 Linux I/O 스케줄러

```
Linux 블록 I/O 스케줄러:

1. mq-deadline (회전 디스크의 기본값):
   - 읽기 및 쓰기 데드라인 큐 유지
   - 어떤 요청도 데드라인보다 오래 대기하지 않도록 보장
   - 읽기 데드라인: 500ms, 쓰기 데드라인: 5000ms
   - 기아 상태 방지

2. bfq (Budget Fair Queueing):
   - 프로세스별 I/O 대역폭 보장
   - 대화형 워크로드(데스크탑)에 적합
   - mq-deadline보다 높은 CPU 오버헤드

3. kyber:
   - 빠른 장치(NVMe SSD)를 위해 설계
   - 읽기와 쓰기에 토큰 버킷 사용
   - 매우 낮은 CPU 오버헤드

4. none (스케줄러 없음):
   - 요청을 장치에 직접 전달
   - 하드웨어 큐가 있는 NVMe에 최적
   - 빠른 장치에 가장 낮은 지연 시간
```

### 2.2 스케줄러 확인 및 변경

```c
/*
 * 현재 스케줄러 확인:
 *   cat /sys/block/sda/queue/scheduler
 *
 * 스케줄러 변경:
 *   echo "mq-deadline" > /sys/block/sda/queue/scheduler
 *
 * 큐 깊이 확인:
 *   cat /sys/block/nvme0n1/queue/nr_requests
 */

#include <stdio.h>
#include <string.h>

void show_io_scheduler(const char *device) {
    char path[256];
    char buf[256];
    FILE *fp;

    snprintf(path, sizeof(path), "/sys/block/%s/queue/scheduler", device);
    fp = fopen(path, "r");
    if (fp) {
        fgets(buf, sizeof(buf), fp);
        printf("Device %s scheduler: %s", device, buf);
        fclose(fp);
    }

    snprintf(path, sizeof(path), "/sys/block/%s/queue/nr_requests", device);
    fp = fopen(path, "r");
    if (fp) {
        fgets(buf, sizeof(buf), fp);
        printf("Queue depth: %s", buf);
        fclose(fp);
    }

    snprintf(path, sizeof(path), "/sys/block/%s/queue/rotational", device);
    fp = fopen(path, "r");
    if (fp) {
        fgets(buf, sizeof(buf), fp);
        printf("Rotational: %s", buf);
        fclose(fp);
    }
}
```

---

## 3. NVMe와 현대 스토리지

### 3.1 NVMe 아키텍처

```
NVMe vs 레거시 스토리지 스택:

레거시 (SATA/SAS):
  애플리케이션 -> VFS -> 블록 레이어 -> I/O 스케줄러
    -> SCSI 레이어 -> AHCI 드라이버 -> SATA 컨트롤러 -> SSD
  큐 깊이: 32
  지연 시간: ~100 μs

NVMe:
  애플리케이션 -> VFS -> 블록 레이어 -> NVMe 드라이버 -> NVMe 컨트롤러 -> SSD
  큐 깊이: 큐당 65,535, 최대 65,535개 큐!
  지연 시간: ~10 μs

  NVMe가 제거하는 것:
    - SCSI 변환 레이어
    - 단일 명령 큐 병목
    - 큐별 잠금

  각 CPU 코어가 자체 제출/완료 큐를 가질 수 있음:
    Core 0 ──▶ SQ0/CQ0 ──▶ NVMe 컨트롤러
    Core 1 ──▶ SQ1/CQ1 ──▶ NVMe 컨트롤러
    Core 2 ──▶ SQ2/CQ2 ──▶ NVMe 컨트롤러
    Core 3 ──▶ SQ3/CQ3 ──▶ NVMe 컨트롤러
```

---

## 4. 비동기 I/O 패턴

### 4.1 I/O 모델 비교

```
다섯 가지 I/O 모델 (단순한 것부터 고급까지):

1. 블로킹 I/O:
   read()가 데이터 준비될 때까지 블록.
   단순하지만 연결당 하나의 스레드 필요.

2. 논블로킹 I/O:
   준비되지 않으면 read()가 EAGAIN 반환.
   애플리케이션이 반복적으로 폴링 (바쁜 대기).

3. I/O 멀티플렉싱 (select/poll/epoll):
   여러 FD 중 하나라도 준비될 때까지 대기.
   Linux에서 epoll이 가장 확장성 있음.

4. 시그널 기반 I/O:
   데이터 준비 시 커널이 SIGIO 전송.
   거의 사용하지 않음 (복잡하고 제한적).

5. 비동기 I/O (io_uring):
   I/O 제출 후 커널이 백그라운드에서 완료.
   애플리케이션이 완료 큐를 확인.
   고성능 서버에 최적.
```

### 4.2 epoll 예제

```c
#include <stdio.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define MAX_EVENTS 1024
#define BUF_SIZE 4096

int make_non_blocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void epoll_server(int port) {
    /* 서버 소켓 생성 */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(port),
    };
    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(server_fd, 128);
    make_non_blocking(server_fd);

    /* epoll 인스턴스 생성 */
    int epoll_fd = epoll_create1(0);
    struct epoll_event ev = {
        .events = EPOLLIN,
        .data.fd = server_fd,
    };
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev);

    struct epoll_event events[MAX_EVENTS];
    char buf[BUF_SIZE];

    printf("Server listening on port %d\n", port);

    while (1) {
        int n = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);

        for (int i = 0; i < n; i++) {
            if (events[i].data.fd == server_fd) {
                /* 새 연결 수락 */
                int client_fd = accept(server_fd, NULL, NULL);
                make_non_blocking(client_fd);

                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = client_fd;
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
            } else {
                /* 클라이언트로부터 읽기 */
                ssize_t bytes = read(events[i].data.fd, buf, BUF_SIZE);
                if (bytes <= 0) {
                    close(events[i].data.fd);
                } else {
                    write(events[i].data.fd, buf, bytes);  /* 에코 */
                }
            }
        }
    }

    close(epoll_fd);
    close(server_fd);
}
```

---

## 5. io_uring 심층 분석

### 5.1 io_uring 아키텍처

```
io_uring (Linux 5.1+): Linux I/O의 미래.

커널과 사용자 공간 사이에 공유되는 두 개의 링 버퍼:

  사용자 공간                   커널
  ┌─────────────┐
  │ Submission   │──제출──▶    I/O 요청을
  │ Queue (SQ)   │              백그라운드에서 처리
  └─────────────┘
  ┌─────────────┐
  │ Completion   │◀──완료──    결과 준비됨
  │ Queue (CQ)   │
  └─────────────┘

장점:
  - 제출+완료에 시스템 호출 제로 (폴링 모드)
  - 배칭: 한 번에 여러 요청 제출
  - 메모리 복사 없음: 공유 링 버퍼
  - 모든 I/O 작업 지원: read, write, send, recv 등
```

### 5.2 io_uring 예제

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>

#define QUEUE_DEPTH 64
#define BUF_SIZE 4096

/*
 * 간단한 io_uring 파일 읽기 예제.
 * 컴파일: gcc -o uring_read uring_read.c -luring
 */

void uring_read_file(const char *filename) {
    struct io_uring ring;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    int ret;

    /* io_uring 초기화 */
    ret = io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-ret));
        return;
    }

    /* 파일 열기 */
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        perror("open");
        io_uring_queue_exit(&ring);
        return;
    }

    /* 읽기 버퍼 준비 */
    char *buf = malloc(BUF_SIZE);
    struct iovec iov = {
        .iov_base = buf,
        .iov_len = BUF_SIZE,
    };

    /* 읽기 요청 제출 */
    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_readv(sqe, fd, &iov, 1, 0);  /* 오프셋 0 */
    sqe->user_data = 42;  /* 식별을 위한 태그 */

    ret = io_uring_submit(&ring);
    printf("Submitted %d request(s)\n", ret);

    /* 완료 대기 */
    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
        fprintf(stderr, "io_uring_wait_cqe: %s\n", strerror(-ret));
    } else {
        printf("Read completed: %d bytes, user_data=%llu\n",
               cqe->res, (unsigned long long)cqe->user_data);

        if (cqe->res > 0) {
            printf("Content: %.100s...\n", buf);
        }

        io_uring_cqe_seen(&ring, cqe);
    }

    free(buf);
    close(fd);
    io_uring_queue_exit(&ring);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    uring_read_file(argv[1]);
    return 0;
}
```

---

## 6. I/O 성능 분석

### 6.1 Linux I/O 프로파일링 도구

```
I/O 분석을 위한 필수 도구:

iostat: 장치별 I/O 통계
  $ iostat -xz 1
  Device  r/s   w/s  rMB/s  wMB/s  await  svctm  %util
  nvme0n1 5000  3000  500    300    0.1    0.05   25%

blktrace: 상세한 블록 I/O 추적
  $ blktrace -d /dev/nvme0n1 -o trace
  $ blkparse -i trace

fio: 유연한 I/O 테스터
  $ fio --name=test --ioengine=io_uring --bs=4k \
        --iodepth=64 --rw=randread --size=1G

perf: 시스템 프로파일러
  $ perf stat -e 'block:*' -- command

bpftrace: 동적 추적
  $ bpftrace -e 'tracepoint:block:block_rq_issue { @[comm] = count(); }'
```

---

## 7. Direct I/O와 Zero-Copy

### 7.1 Direct I/O (O_DIRECT)

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

/*
 * Direct I/O는 페이지 캐시를 우회.
 *
 * 일반 I/O:    앱 -> 페이지 캐시 -> 디스크
 * Direct I/O:  앱 -> 디스크 (캐시 우회)
 *
 * Direct I/O를 사용할 때:
 *   - 데이터베이스 엔진 (자체 캐시 관리)
 *   - 대규모 순차 읽기 (캐시 오염)
 *   - 예측 가능한 지연 시간이 필요한 경우
 *
 * 요구사항:
 *   - 버퍼가 정렬되어야 함 (일반적으로 512 또는 4096 바이트)
 *   - I/O 크기가 정렬되어야 함
 *   - 파일 오프셋이 정렬되어야 함
 */

void direct_io_example(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open with O_DIRECT");
        return;
    }

    /* 정렬된 버퍼 (O_DIRECT에 필요) */
    size_t align = 4096;
    size_t size = 4096;
    void *buf = NULL;
    posix_memalign(&buf, align, size);

    ssize_t bytes = read(fd, buf, size);
    if (bytes > 0) {
        printf("Direct I/O read %zd bytes\n", bytes);
    }

    free(buf);
    close(fd);
}
```

### 7.2 sendfile을 이용한 Zero-Copy

```c
#include <sys/sendfile.h>

/*
 * sendfile: 사용자 공간을 거치지 않고 파일 디스크립터 간
 * 데이터 전송.
 *
 * 전통적 방식:
 *   read(file_fd, buf, n)     -> 커널에서 사용자로 복사
 *   write(socket_fd, buf, n)  -> 사용자에서 커널로 복사
 *   합계: 2번 복사, 4번 컨텍스트 스위치
 *
 * sendfile:
 *   sendfile(socket_fd, file_fd, offset, n)
 *   합계: 사용자 공간 복사 0번, 2번 컨텍스트 스위치
 */

void serve_file(int client_fd, const char *filename) {
    int file_fd = open(filename, O_RDONLY);
    struct stat sb;
    fstat(file_fd, &sb);

    /* Zero-copy 전송: 커널이 모든 것을 처리 */
    off_t offset = 0;
    sendfile(client_fd, file_fd, &offset, sb.st_size);

    close(file_fd);
}
```

---

## 8. 연습문제

### 연습문제 1: 디스크 스케줄링 시뮬레이터

디스크 스케줄링 알고리즘을 구현하고 비교하세요:
1. FCFS, SSTF, SCAN, C-SCAN, LOOK을 구현하세요
2. 10,000 실린더 디스크에서 100개의 랜덤 요청 생성
3. 각 알고리즘의 총 헤드 이동량 계산
4. 초기 헤드 위치를 변화시키며 영향 측정
5. 그래프: 다른 요청 패턴에 대한 알고리즘별 총 이동량

### 연습문제 2: io_uring 파일 복사

io_uring을 사용한 고성능 파일 복사 구축:
1. 전통적인 read/write로 파일 복사 구현
2. io_uring으로 구현 (배치 제출)
3. 파일 크기별 처리량 비교: 1 MB, 100 MB, 1 GB
4. 큐 깊이 변화: 1, 16, 64, 256
5. cp와 dd 성능과 비교

### 연습문제 3: I/O 스케줄러 벤치마크

Linux I/O 스케줄러 벤치마크:
1. fio를 사용하여 랜덤 읽기, 순차 읽기, 혼합 워크로드 테스트
2. none, mq-deadline, bfq, kyber로 테스트
3. 각각의 IOPS, 지연 시간(p50, p99), 처리량 측정
4. SSD와 HDD 모두에서 테스트 (가능한 경우)
5. 추천: 어떤 워크로드에 어떤 스케줄러?

### 연습문제 4: epoll vs io_uring 서버

네트워크 서버를 구축하고 비교:
1. epoll로 에코 서버 구현
2. io_uring으로 에코 서버 구현
3. wrk 또는 ab를 사용하여 두 서버 벤치마크 (10K 동시 연결)
4. 측정: 초당 요청, 지연 시간, CPU 사용량
5. 분석: io_uring의 장점이 언제 나타나는가?

### 연습문제 5: Zero-Copy 성능

Zero-copy I/O의 이점 측정:
1. read()+write()로 대용량 파일(1 GB)을 TCP로 전송
2. sendfile()로 동일 파일 전송
3. mmap() + write()로 전송
4. 비교: 처리량, CPU 사용량, 메모리 사용량
5. perf를 사용하여 컨텍스트 스위치와 복사 작업 수 측정

---

*레슨 21 끝*
