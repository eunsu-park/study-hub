[Previous: Advanced Virtual Memory](./20_Advanced_Virtual_Memory.md)

---

# 21. Disk Scheduling and Modern I/O

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain traditional disk scheduling algorithms and their tradeoffs
2. Describe NVMe architecture and how it changes I/O scheduling
3. Implement io_uring for high-performance asynchronous I/O
4. Compare blocking, non-blocking, epoll, and io_uring I/O patterns
5. Analyze I/O performance using Linux profiling tools

---

## Table of Contents

1. [Traditional Disk Scheduling](#1-traditional-disk-scheduling)
2. [I/O Schedulers in Linux](#2-io-schedulers-in-linux)
3. [NVMe and Modern Storage](#3-nvme-and-modern-storage)
4. [Asynchronous I/O Patterns](#4-asynchronous-io-patterns)
5. [io_uring Deep Dive](#5-io_uring-deep-dive)
6. [I/O Performance Analysis](#6-io-performance-analysis)
7. [Direct I/O and Zero-Copy](#7-direct-io-and-zero-copy)
8. [Exercises](#8-exercises)

---

## 1. Traditional Disk Scheduling

### 1.1 HDD Anatomy and Access Time

```
HDD access time = Seek time + Rotational latency + Transfer time

Seek time: Move read/write head to correct track (0.5-15 ms)
Rotational latency: Wait for sector to rotate under head (0-8 ms @ 7200 RPM)
Transfer time: Read/write data (typically < 1 ms)

Total: 4-20 ms per random access (vs ~0.1 ms for SSD!)

This is why disk scheduling matters for HDDs:
  Minimizing seek distance = minimizing head movement = faster I/O
```

### 1.2 Scheduling Algorithms

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_REQUESTS 100

/*
 * FCFS (First Come First Served):
 * Process requests in arrival order.
 * Simple but can result in long seek distances.
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
 * Always service the closest request next.
 * Better than FCFS but can cause starvation.
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
 * SCAN (Elevator Algorithm):
 * Move head in one direction, servicing requests.
 * Reverse direction at the end.
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

    /* Move right first */
    for (int i = 0; i < n; i++) {
        if (sorted[i] >= current) {
            total_movement += abs(sorted[i] - current);
            current = sorted[i];
            printf(" -> %d", current);
        }
    }

    /* Go to end */
    total_movement += abs(max_cylinder - current);
    current = max_cylinder;

    /* Move left */
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

## 2. I/O Schedulers in Linux

### 2.1 Linux I/O Schedulers

```
Linux Block I/O Schedulers:

1. mq-deadline (default for rotating disks):
   - Maintains read and write deadline queues
   - Ensures no request waits longer than deadline
   - Read deadline: 500ms, Write deadline: 5000ms
   - Prevents starvation

2. bfq (Budget Fair Queueing):
   - Per-process I/O bandwidth guarantees
   - Good for interactive workloads (desktop)
   - Higher CPU overhead than mq-deadline

3. kyber:
   - Designed for fast devices (NVMe SSDs)
   - Uses token buckets for read and write
   - Very low CPU overhead

4. none (no scheduler):
   - Passes requests directly to device
   - Best for NVMe with hardware queues
   - Lowest latency for fast devices
```

### 2.2 Checking and Changing Schedulers

```c
/*
 * Check current scheduler:
 *   cat /sys/block/sda/queue/scheduler
 *
 * Change scheduler:
 *   echo "mq-deadline" > /sys/block/sda/queue/scheduler
 *
 * Check queue depth:
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

## 3. NVMe and Modern Storage

### 3.1 NVMe Architecture

```
NVMe vs Legacy Storage Stack:

Legacy (SATA/SAS):
  Application -> VFS -> Block Layer -> I/O Scheduler
    -> SCSI Layer -> AHCI Driver -> SATA Controller -> SSD
  Queue depth: 32
  Latency: ~100 μs

NVMe:
  Application -> VFS -> Block Layer -> NVMe Driver -> NVMe Controller -> SSD
  Queue depth: 65,535 per queue, up to 65,535 queues!
  Latency: ~10 μs

  NVMe eliminates:
    - SCSI translation layer
    - Single command queue bottleneck
    - Per-queue locks

  Each CPU core can have its own submission/completion queue:
    Core 0 ──▶ SQ0/CQ0 ──▶ NVMe Controller
    Core 1 ──▶ SQ1/CQ1 ──▶ NVMe Controller
    Core 2 ──▶ SQ2/CQ2 ──▶ NVMe Controller
    Core 3 ──▶ SQ3/CQ3 ──▶ NVMe Controller
```

---

## 4. Asynchronous I/O Patterns

### 4.1 I/O Models Comparison

```
Five I/O models (from simple to advanced):

1. Blocking I/O:
   read() blocks until data ready.
   Simple but one thread per connection.

2. Non-blocking I/O:
   read() returns EAGAIN if not ready.
   Application polls repeatedly (busy-waiting).

3. I/O Multiplexing (select/poll/epoll):
   Wait for ANY of multiple FDs to be ready.
   epoll is most scalable on Linux.

4. Signal-driven I/O:
   Kernel sends SIGIO when data ready.
   Rarely used (complex, limited).

5. Asynchronous I/O (io_uring):
   Submit I/O, kernel completes in background.
   Application checks completion queue.
   Best for high-performance servers.
```

### 4.2 epoll Example

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
    /* Create server socket */
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

    /* Create epoll instance */
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
                /* Accept new connection */
                int client_fd = accept(server_fd, NULL, NULL);
                make_non_blocking(client_fd);

                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = client_fd;
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
            } else {
                /* Read from client */
                ssize_t bytes = read(events[i].data.fd, buf, BUF_SIZE);
                if (bytes <= 0) {
                    close(events[i].data.fd);
                } else {
                    write(events[i].data.fd, buf, bytes);  /* Echo */
                }
            }
        }
    }

    close(epoll_fd);
    close(server_fd);
}
```

---

## 5. io_uring Deep Dive

### 5.1 io_uring Architecture

```
io_uring (Linux 5.1+): The future of Linux I/O.

Two ring buffers shared between kernel and userspace:

  Userspace                    Kernel
  ┌─────────────┐
  │ Submission   │──submit──▶  Process I/O requests
  │ Queue (SQ)   │              in background
  └─────────────┘
  ┌─────────────┐
  │ Completion   │◀──complete── Results ready
  │ Queue (CQ)   │
  └─────────────┘

Benefits:
  - Zero system calls for submission+completion (polled mode)
  - Batching: submit many requests at once
  - No memory copies: shared ring buffers
  - Supports ALL I/O operations: read, write, send, recv, etc.
```

### 5.2 io_uring Example

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
 * Simple io_uring file read example.
 * Compile: gcc -o uring_read uring_read.c -luring
 */

void uring_read_file(const char *filename) {
    struct io_uring ring;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    int ret;

    /* Initialize io_uring */
    ret = io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-ret));
        return;
    }

    /* Open file */
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        perror("open");
        io_uring_queue_exit(&ring);
        return;
    }

    /* Prepare read buffer */
    char *buf = malloc(BUF_SIZE);
    struct iovec iov = {
        .iov_base = buf,
        .iov_len = BUF_SIZE,
    };

    /* Submit read request */
    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_readv(sqe, fd, &iov, 1, 0);  /* offset 0 */
    sqe->user_data = 42;  /* Tag for identification */

    ret = io_uring_submit(&ring);
    printf("Submitted %d request(s)\n", ret);

    /* Wait for completion */
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

## 6. I/O Performance Analysis

### 6.1 Linux I/O Profiling Tools

```
Essential tools for I/O analysis:

iostat: I/O statistics per device
  $ iostat -xz 1
  Device  r/s   w/s  rMB/s  wMB/s  await  svctm  %util
  nvme0n1 5000  3000  500    300    0.1    0.05   25%

blktrace: Detailed block I/O tracing
  $ blktrace -d /dev/nvme0n1 -o trace
  $ blkparse -i trace

fio: Flexible I/O tester
  $ fio --name=test --ioengine=io_uring --bs=4k \
        --iodepth=64 --rw=randread --size=1G

perf: System profiler
  $ perf stat -e 'block:*' -- command

bpftrace: Dynamic tracing
  $ bpftrace -e 'tracepoint:block:block_rq_issue { @[comm] = count(); }'
```

---

## 7. Direct I/O and Zero-Copy

### 7.1 Direct I/O (O_DIRECT)

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

/*
 * Direct I/O bypasses the page cache.
 *
 * Normal I/O:    App -> Page Cache -> Disk
 * Direct I/O:    App -> Disk (bypass cache)
 *
 * When to use Direct I/O:
 *   - Database engines (manage their own cache)
 *   - Large sequential reads (cache pollution)
 *   - When you need predictable latency
 *
 * Requirements:
 *   - Buffer must be aligned (typically 512 or 4096 bytes)
 *   - I/O size must be aligned
 *   - File offset must be aligned
 */

void direct_io_example(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open with O_DIRECT");
        return;
    }

    /* Aligned buffer (required for O_DIRECT) */
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

### 7.2 Zero-Copy with sendfile

```c
#include <sys/sendfile.h>

/*
 * sendfile: transfer data between file descriptors without
 * copying through userspace.
 *
 * Traditional:
 *   read(file_fd, buf, n)  -> kernel to user copy
 *   write(socket_fd, buf, n) -> user to kernel copy
 *   Total: 2 copies, 4 context switches
 *
 * sendfile:
 *   sendfile(socket_fd, file_fd, offset, n)
 *   Total: 0 user-space copies, 2 context switches
 */

void serve_file(int client_fd, const char *filename) {
    int file_fd = open(filename, O_RDONLY);
    struct stat sb;
    fstat(file_fd, &sb);

    /* Zero-copy transfer: kernel handles everything */
    off_t offset = 0;
    sendfile(client_fd, file_fd, &offset, sb.st_size);

    close(file_fd);
}
```

---

## 8. Exercises

### Exercise 1: Disk Scheduling Simulator

Implement and compare disk scheduling algorithms:
1. Implement FCFS, SSTF, SCAN, C-SCAN, and LOOK
2. Generate 100 random requests on a 10,000-cylinder disk
3. Compute total head movement for each algorithm
4. Vary initial head position and measure impact
5. Plot: total movement vs algorithm for different request patterns

### Exercise 2: io_uring File Copy

Build a high-performance file copy using io_uring:
1. Implement file copy with traditional read/write
2. Implement with io_uring (batched submissions)
3. Compare throughput for files: 1 MB, 100 MB, 1 GB
4. Vary queue depth: 1, 16, 64, 256
5. Compare with cp and dd performance

### Exercise 3: I/O Scheduler Benchmark

Benchmark Linux I/O schedulers:
1. Use fio to test random read, sequential read, mixed workload
2. Test with: none, mq-deadline, bfq, kyber
3. Measure IOPS, latency (p50, p99), throughput for each
4. Test on both SSD and HDD (if available)
5. Recommend: which scheduler for which workload?

### Exercise 4: epoll vs io_uring Server

Build and compare network servers:
1. Implement echo server with epoll
2. Implement echo server with io_uring
3. Use wrk or ab to benchmark both (10K concurrent connections)
4. Measure: requests/second, latency, CPU usage
5. Analyze: when does io_uring's advantage show?

### Exercise 5: Zero-Copy Performance

Measure zero-copy I/O benefits:
1. Serve a large file (1 GB) over TCP using read()+write()
2. Serve the same file using sendfile()
3. Serve using mmap() + write()
4. Compare: throughput, CPU usage, memory usage
5. Use perf to count context switches and copy operations

---

*End of Lesson 21*
