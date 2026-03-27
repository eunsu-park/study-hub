[이전: I/O와 IPC](./18_IO_and_IPC.md)

---

# 20. 고급 가상 메모리

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다단계 TLB와 TLB shootdown을 포함한 TLB 관리 전략을 설명할 수 있다
2. Huge page 할당을 구현하고 성능 이점을 설명할 수 있다
3. NUMA 토폴로지와 메모리 할당 결정에 미치는 영향을 분석할 수 있다
4. Memory-mapped I/O 메커니즘과 디바이스 드라이버에서의 활용을 설명할 수 있다
5. 현대 하드웨어에 맞는 애플리케이션 메모리 접근 패턴을 프로파일링하고 최적화할 수 있다

---

## 목차

1. [TLB 심층 분석](#1-tlb-심층-분석)
2. [Huge Pages](#2-huge-pages)
3. [NUMA 아키텍처](#3-numa-아키텍처)
4. [Memory-Mapped I/O](#4-memory-mapped-io)
5. [커널 메모리 관리](#5-커널-메모리-관리)
6. [Copy-on-Write와 메모리 공유](#6-copy-on-write와-메모리-공유)
7. [메모리 성능 최적화](#7-메모리-성능-최적화)
8. [연습문제](#8-연습문제)

---

## 1. TLB 심층 분석

### 1.1 TLB 구조와 동작

```
TLB (Translation Lookaside Buffer):
  페이지 테이블 항목(가상 → 물리 매핑)의 캐시.

TLB 없이:
  가상 주소 → 페이지 테이블 탐색 (4단계의 경우 4번의 메모리 접근!)
  → 물리 주소
  시간: ~200 사이클

TLB 있을 때 (히트):
  가상 주소 → TLB 조회 (1 사이클) → 물리 주소
  시간: ~1 사이클

TLB 히트율은 성능에 매우 중요.
일반적인 히트율: 잘 동작하는 워크로드에서 99% 이상.

TLB 구조:
  ┌──────────────────────────────────────────┐
  │ VPN (Virtual Page Number) │ PPN │ Flags  │
  ├──────────────────────────────────────────┤
  │ 0x7fff1000                │ 0x3a2│ RWXU  │
  │ 0x400000                  │ 0x1f8│ RX-U  │
  │ 0x601000                  │ 0x2c1│ RW-U  │
  │ ...                       │ ...  │ ...   │
  └──────────────────────────────────────────┘
```

### 1.2 다단계 TLB

```
현대 CPU는 계층적 TLB를 사용:

L1 ITLB (명령어):  64-128 항목, 1 사이클
L1 DTLB (데이터):  64-128 항목, 1 사이클
L2 통합 TLB:       1024-4096 항목, ~7 사이클
페이지 테이블 워크: ~200 사이클 (페이지 워커 하드웨어 사용)

                    ┌────────┐
   가상 주소 ───▶│ L1 TLB │──히트──▶ 물리 주소 (1 사이클)
                    └───┬────┘
                       미스
                    ┌───▼────┐
                    │ L2 TLB │──히트──▶ 물리 주소 (7 사이클)
                    └───┬────┘
                       미스
                    ┌───▼──────────┐
                    │ Page Walker  │──▶ 물리 주소 (200 사이클)
                    │ (하드웨어)    │
                    └──────────────┘
```

### 1.3 TLB Shootdown

```c
/*
 * TLB Shootdown: 멀티프로세서 시스템에서 페이지 매핑이 변경되면,
 * 해당 매핑을 캐싱하고 있는 모든 코어가 이를 무효화해야 합니다.
 *
 * 이를 위해 Inter-Processor Interrupt (IPI)가 필요 - 비용이 큼!
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

/*
 * TLB 영향 측정 시뮬레이션.
 * 실제 커널에서 TLB shootdown은 다음을 사용:
 *   - invlpg 명령어 (단일 항목 무효화)
 *   - cr3 재로드 (전체 TLB 플러시)
 *   - IPI로 다른 코어에 알림
 */

void demonstrate_tlb_impact(void) {
    const size_t PAGE_SIZE = 4096;
    const size_t NUM_PAGES = 256;
    const size_t TOTAL_SIZE = PAGE_SIZE * NUM_PAGES;

    /* 메모리 할당 */
    char *mem = mmap(NULL, TOTAL_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        return;
    }

    /* 순차 접근: 좋은 TLB 동작 (적은 수의 고유 페이지) */
    volatile char sink;
    for (int iter = 0; iter < 1000; iter++) {
        for (size_t i = 0; i < TOTAL_SIZE; i += 64) {
            sink = mem[i];
        }
    }

    /* 랜덤 접근: 나쁜 TLB 동작 (많은 수의 고유 페이지) */
    size_t *random_offsets = malloc(NUM_PAGES * sizeof(size_t));
    for (size_t i = 0; i < NUM_PAGES; i++) {
        random_offsets[i] = (rand() % NUM_PAGES) * PAGE_SIZE;
    }

    for (int iter = 0; iter < 1000; iter++) {
        for (size_t i = 0; i < NUM_PAGES; i++) {
            sink = mem[random_offsets[i]];
        }
    }

    free(random_offsets);
    munmap(mem, TOTAL_SIZE);
}
```

---

## 2. Huge Pages

### 2.1 Huge Pages가 필요한 이유

```
표준 페이지 크기: 4 KB
Huge page 크기: 2 MB (x86), 1 GB (x86), 64 KB (ARM)

작은 페이지의 문제:
  애플리케이션이 1 GB 메모리 사용:
  4 KB 페이지: 262,144 페이지 → 262,144개의 TLB 항목 필요
  하지만 TLB는 ~1000-4000개 항목만 보유!
  → 지속적인 TLB 미스 → 지속적인 페이지 워크 → 느림

Huge pages 사용 시:
  2 MB 페이지: 512 페이지 → 512개의 TLB 항목
  1 GB 페이지: 1 페이지 → 1개의 TLB 항목!
  → TLB가 전체 작업 세트를 커버 → 빠름

성능 향상: 메모리 집중 워크로드에서 10-30%.
```

### 2.2 Linux에서 Huge Pages 사용하기

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)  /* 2 MB */

/*
 * 방법 1: MAP_HUGETLB를 사용한 mmap
 */
void *allocate_huge_mmap(size_t size) {
    /* Huge page 경계로 올림 */
    size_t aligned = (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);

    void *ptr = mmap(NULL, aligned,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                     -1, 0);

    if (ptr == MAP_FAILED) {
        perror("mmap(MAP_HUGETLB)");
        return NULL;
    }

    printf("Allocated %zu bytes with huge pages at %p\n", aligned, ptr);
    return ptr;
}

/*
 * 방법 2: MADV_HUGEPAGE를 사용한 madvise (Transparent Huge Pages)
 */
void *allocate_thp(size_t size) {
    size_t aligned = (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);

    void *ptr = mmap(NULL, aligned,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);

    if (ptr == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }

    /* 커널에 힌트: transparent huge pages 사용 */
    if (madvise(ptr, aligned, MADV_HUGEPAGE) != 0) {
        perror("madvise(MADV_HUGEPAGE)");
    }

    printf("Allocated %zu bytes with THP hint at %p\n", aligned, ptr);
    return ptr;
}

/*
 * 벤치마크: 4K 페이지 vs huge pages 비교
 */
void benchmark_page_sizes(void) {
    const size_t SIZE = 256 * 1024 * 1024;  /* 256 MB */
    const int ITERATIONS = 10;

    /* 일반 페이지로 할당 */
    char *regular = mmap(NULL, SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    /* Huge pages로 할당 */
    char *huge = allocate_huge_mmap(SIZE);

    if (!regular || !huge) return;

    /* 모든 페이지 접근 (페이지 폴트 유발) */
    memset(regular, 0, SIZE);
    if (huge) memset(huge, 0, SIZE);

    /* 랜덤 접근 벤치마크 */
    volatile char sink;
    size_t stride = 4096;  /* 페이지당 하나 접근 */

    printf("Random access benchmark (%zu MB):\n", SIZE / (1024*1024));

    /* 일반 페이지 */
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (size_t off = 0; off < SIZE; off += stride) {
            size_t idx = ((off * 2654435761UL) % SIZE) & ~(stride - 1);
            sink = regular[idx];
        }
    }
    printf("  Regular pages: done\n");

    /* Huge pages */
    if (huge) {
        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (size_t off = 0; off < SIZE; off += stride) {
                size_t idx = ((off * 2654435761UL) % SIZE) & ~(stride - 1);
                sink = huge[idx];
            }
        }
        printf("  Huge pages: done\n");
    }

    munmap(regular, SIZE);
    if (huge) munmap(huge, SIZE);
}

int main(void) {
    benchmark_page_sizes();
    return 0;
}
```

---

## 3. NUMA 아키텍처

### 3.1 NUMA 토폴로지

```
UMA (Uniform Memory Access) - 구형 시스템:
  모든 CPU가 동일한 지연 시간으로 모든 메모리에 접근.

    CPU0  CPU1  CPU2  CPU3
      \    |    |    /
       ┌───┴────┴───┐
       │  메모리 버스  │
       └───────┬─────┘
               │
          ┌────┴────┐
          │  메모리   │
          └──────────┘

NUMA (Non-Uniform Memory Access) - 현대 서버:
  각 CPU는 로컬 메모리(빠름)와 리모트 메모리(느림)를 가짐.

    ┌─────────────┐         ┌─────────────┐
    │   노드 0     │ QPI/UPI │   노드 1     │
    │ CPU0  CPU1  │◄────────▶│ CPU2  CPU3  │
    │ ┌─────────┐ │         │ ┌─────────┐ │
    │ │  메모리   │ │         │ │  메모리   │ │
    │ │ (로컬)   │ │         │ │ (로컬)   │ │
    │ └─────────┘ │         │ └─────────┘ │
    └─────────────┘         └─────────────┘

  로컬 접근:  ~80 ns
  리모트 접근: ~140 ns (1.75배 느림!)
```

### 3.2 NUMA 인식 프로그래밍

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <sched.h>

/*
 * 컴파일: gcc -o numa_demo numa_demo.c -lnuma
 */

void demonstrate_numa(void) {
    if (numa_available() < 0) {
        printf("NUMA not available\n");
        return;
    }

    /* NUMA 토폴로지 조회 */
    int num_nodes = numa_num_configured_nodes();
    int num_cpus = numa_num_configured_cpus();
    printf("NUMA nodes: %d, CPUs: %d\n", num_nodes, num_cpus);

    for (int node = 0; node < num_nodes; node++) {
        long free_mem;
        long total = numa_node_size(node, &free_mem);
        printf("Node %d: %ld MB total, %ld MB free\n",
               node, total / (1024*1024), free_mem / (1024*1024));

        /* 이 노드에 속하는 CPU는? */
        struct bitmask *cpus = numa_allocate_cpumask();
        numa_node_to_cpus(node, cpus);
        printf("  CPUs: ");
        for (int cpu = 0; cpu < num_cpus; cpu++) {
            if (numa_bitmask_isbitset(cpus, cpu)) {
                printf("%d ", cpu);
            }
        }
        printf("\n");
        numa_free_cpumask(cpus);
    }

    /* 특정 NUMA 노드에 할당 */
    size_t size = 1024 * 1024;  /* 1 MB */

    void *local = numa_alloc_onnode(size, 0);
    printf("\nAllocated on node 0: %p\n", local);

    void *interleaved = numa_alloc_interleaved(size);
    printf("Interleaved allocation: %p\n", interleaved);

    /* 스레드를 특정 노드에 바인딩 */
    numa_run_on_node(0);
    printf("Thread bound to node 0\n");

    numa_free(local, size);
    numa_free(interleaved, size);
}

/*
 * NUMA 인식 데이터 구조 레이아웃
 */
typedef struct {
    int *data;
    size_t size;
    int numa_node;
} numa_array_t;

numa_array_t *create_numa_array(size_t count, int node) {
    numa_array_t *arr = malloc(sizeof(numa_array_t));
    arr->size = count;
    arr->numa_node = node;

    /* 특정 NUMA 노드에 데이터 할당 */
    arr->data = numa_alloc_onnode(count * sizeof(int), node);
    if (!arr->data) {
        free(arr);
        return NULL;
    }

    return arr;
}
```

---

## 4. Memory-Mapped I/O

### 4.1 파일 I/O를 위한 mmap

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

/*
 * Memory-mapped 파일 I/O:
 * 파일 내용을 가상 주소 공간에 직접 매핑.
 * 명시적인 read()/write() 호출이 필요 없음!
 *
 * 프로세스 가상 메모리:
 * ┌──────────────┐
 * │    스택       │
 * ├──────────────┤
 * │    ...        │
 * ├──────────────┤
 * │  mmap 영역    │◄── 파일 내용이 여기에 매핑
 * │  (file.dat)   │    읽기/쓰기가 직접 파일로 전달
 * ├──────────────┤
 * │    힙         │
 * ├──────────────┤
 * │    코드       │
 * └──────────────┘
 */

void mmap_file_example(const char *filename) {
    /* 파일 열기 */
    int fd = open(filename, O_RDWR);
    if (fd < 0) {
        perror("open");
        return;
    }

    /* 파일 크기 가져오기 */
    struct stat sb;
    fstat(fd, &sb);
    size_t size = sb.st_size;

    /* 파일을 메모리에 매핑 */
    char *mapped = mmap(NULL, size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    /* 이제 파일 내용을 메모리처럼 접근 가능! */
    printf("First 100 bytes: %.100s\n", mapped);

    /* 메모리에 쓰기로 파일 수정 */
    mapped[0] = 'H';
    mapped[1] = 'i';

    /* 변경사항을 디스크에 플러시 */
    msync(mapped, size, MS_SYNC);

    /* 정리 */
    munmap(mapped, size);
    close(fd);
}

/*
 * 성능 비교: read() vs mmap
 */
void compare_io_methods(const char *filename) {
    struct stat sb;
    stat(filename, &sb);
    size_t size = sb.st_size;

    /* 방법 1: 전통적인 read() */
    int fd = open(filename, O_RDONLY);
    char *buf = malloc(size);
    read(fd, buf, size);

    /* 처리: 줄바꿈 개수 세기 */
    long count1 = 0;
    for (size_t i = 0; i < size; i++) {
        if (buf[i] == '\n') count1++;
    }
    free(buf);
    close(fd);

    /* 방법 2: mmap */
    fd = open(filename, O_RDONLY);
    char *mapped = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    /* 힌트: 순차적으로 읽을 예정 */
    madvise(mapped, size, MADV_SEQUENTIAL);

    long count2 = 0;
    for (size_t i = 0; i < size; i++) {
        if (mapped[i] == '\n') count2++;
    }

    munmap(mapped, size);
    close(fd);

    printf("read():  %ld newlines\n", count1);
    printf("mmap():  %ld newlines\n", count2);
}
```

### 4.2 디바이스 Memory-Mapped I/O

```c
/*
 * 디바이스 MMIO: 하드웨어 레지스터가 가상 주소 공간에 매핑.
 * 디바이스 드라이버가 하드웨어와 통신하는 데 사용.
 *
 * 물리 메모리 맵:
 * 0x00000000 - 0x7FFFFFFF: RAM
 * 0xE0000000 - 0xE0000FFF: GPU 레지스터 (MMIO)
 * 0xF0000000 - 0xF000FFFF: 네트워크 카드 레지스터 (MMIO)
 *
 * 이 주소에 읽기/쓰기를 하면 하드웨어와 통신!
 */

#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

/* 예제: MMIO를 통한 하드웨어 레지스터 읽기 */
void read_device_register(off_t phys_addr, size_t size) {
    int fd = open("/dev/mem", O_RDONLY);
    if (fd < 0) {
        perror("open /dev/mem");
        return;
    }

    /* 물리 주소를 가상 공간에 매핑 */
    volatile uint32_t *regs = mmap(NULL, size,
                                    PROT_READ,
                                    MAP_SHARED,
                                    fd, phys_addr);
    if (regs == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    /* 레지스터 읽기 (volatile로 최적화 방지) */
    uint32_t status = regs[0];
    printf("Device status register: 0x%08x\n", status);

    munmap((void *)regs, size);
    close(fd);
}
```

---

## 5. 커널 메모리 관리

### 5.1 슬랩 할당자

```
Linux 커널 메모리 할당자:

버디 시스템:
  2의 거듭제곱 페이지 단위로 메모리 할당.
  큰 할당에 적합하지만, 작은 객체에는 메모리 낭비.

슬랩 할당자:
  같은 크기의 자주 할당되는 객체를 캐싱.
  반복적인 초기화/소멸을 회피.

  ┌──────────────────────────────────┐
  │          kmem_cache               │
  │  (예: "task_struct" 캐시)         │
  ├──────────────────────────────────┤
  │  슬랩 1: [obj][obj][obj][obj]    │ ← 가득 참
  │  슬랩 2: [obj][obj][   ][   ]    │ ← 부분 사용
  │  슬랩 3: [   ][   ][   ][   ]    │ ← 비어 있음
  └──────────────────────────────────┘

  장점:
  - 고정 크기 객체에 대한 단편화 없음
  - 캐싱된 객체로 재초기화 회피
  - NUMA 인식: 노드별 슬랩 리스트
```

### 5.2 OOM Killer

```c
/*
 * Linux OOM (Out of Memory) Killer:
 * 시스템 메모리가 부족하면 커널이 프로세스를 종료.
 *
 * OOM 점수: /proc/<pid>/oom_score
 *   높은 점수 = 종료될 가능성이 더 높음
 *
 * 요소:
 *   - 메모리 사용량 (높을수록 더 가능성 높음)
 *   - oom_score_adj (-1000 ~ 1000, 사용자 설정 가능)
 *   - 프로세스 수명 (새 프로세스가 더 가능성 높음)
 */

#include <stdio.h>
#include <stdlib.h>

void show_oom_info(pid_t pid) {
    char path[256];
    FILE *fp;

    /* OOM 점수 읽기 */
    snprintf(path, sizeof(path), "/proc/%d/oom_score", pid);
    fp = fopen(path, "r");
    if (fp) {
        int score;
        fscanf(fp, "%d", &score);
        printf("PID %d OOM score: %d\n", pid, score);
        fclose(fp);
    }

    /* OOM 조정값 읽기 */
    snprintf(path, sizeof(path), "/proc/%d/oom_score_adj", pid);
    fp = fopen(path, "r");
    if (fp) {
        int adj;
        fscanf(fp, "%d", &adj);
        printf("PID %d OOM adj: %d\n", pid, adj);
        fclose(fp);
    }
}

/*
 * 중요 프로세스를 OOM으로부터 보호:
 * echo -1000 > /proc/<pid>/oom_score_adj
 */
```

---

## 6. Copy-on-Write와 메모리 공유

### 6.1 COW 구현

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>

/*
 * Copy-on-Write (COW):
 * fork() 후, 부모와 자식이 동일한 물리 페이지를 공유.
 * 페이지는 읽기 전용으로 표시.
 * 한 프로세스가 쓰기를 할 때만 커널이 페이지를 복사.
 *
 * 쓰기 전:
 *   부모 VAS ──▶ 물리 페이지 ◀── 자식 VAS
 *                  (읽기 전용)
 *
 * 자식이 쓰기를 한 후:
 *   부모 VAS ──▶ 물리 페이지 (원본)
 *   자식 VAS ──▶ 물리 페이지 (복사본, 이제 쓰기 가능)
 */

void demonstrate_cow(void) {
    /* 100 MB 할당 */
    size_t size = 100 * 1024 * 1024;
    char *data = malloc(size);
    memset(data, 'A', size);  /* 모든 페이지 접근 */

    printf("Parent: allocated %zu MB\n", size / (1024*1024));

    pid_t pid = fork();

    if (pid == 0) {
        /* 자식 프로세스 */
        /* 이 시점에서 자식은 COW를 통해 부모와 모든 페이지를 공유 */
        printf("Child: shares pages with parent (COW)\n");

        /* 첫 번째 페이지만 수정 - 이 페이지만 복사됨 */
        data[0] = 'B';
        printf("Child: modified 1 page (1 physical copy)\n");

        /* 나머지 25599 페이지는 여전히 공유! */
        printf("Child: 99.996%% of memory still shared\n");

        free(data);
        _exit(0);
    } else {
        wait(NULL);
        printf("Parent: data[0] still = '%c' (unchanged)\n", data[0]);
        free(data);
    }
}

int main(void) {
    demonstrate_cow();
    return 0;
}
```

---

## 7. 메모리 성능 최적화

### 7.1 캐시 친화적 접근 패턴

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4096

/*
 * 행 우선 vs 열 우선 접근:
 * 배열은 메모리에 행 단위로 저장.
 * 행 단위 접근 = 순차 접근 = 캐시 친화적.
 * 열 단위 접근 = 보폭 접근 = 캐시 비친화적.
 */

void row_major_access(int matrix[N][N]) {
    long sum = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += matrix[i][j];  /* 순차 접근: 빠름 */
        }
    }
}

void col_major_access(int matrix[N][N]) {
    long sum = 0;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            sum += matrix[i][j];  /* 보폭 접근: 느림 */
        }
    }
}

void benchmark_access_patterns(void) {
    int (*matrix)[N] = malloc(N * N * sizeof(int));

    /* 초기화 */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = i + j;

    clock_t start, end;

    start = clock();
    for (int iter = 0; iter < 10; iter++)
        row_major_access(matrix);
    end = clock();
    printf("Row-major: %.3f s\n",
           (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    for (int iter = 0; iter < 10; iter++)
        col_major_access(matrix);
    end = clock();
    printf("Col-major: %.3f s\n",
           (double)(end - start) / CLOCKS_PER_SEC);

    free(matrix);
}

int main(void) {
    benchmark_access_patterns();
    return 0;
}
```

### 7.2 메모리 프리페칭

```c
#include <immintrin.h>

/*
 * 소프트웨어 프리페칭 힌트:
 * CPU에게 필요하기 전에 데이터 로딩을 시작하도록 지시.
 */
void prefetch_example(int *data, size_t n) {
    long sum = 0;
    const int PREFETCH_DISTANCE = 16;  /* 16개 요소 앞서서 */

    for (size_t i = 0; i < n; i++) {
        /* 곧 필요할 데이터를 프리페치 */
        if (i + PREFETCH_DISTANCE < n) {
            _mm_prefetch(&data[i + PREFETCH_DISTANCE], _MM_HINT_T0);
        }
        sum += data[i];
    }
}
```

---

## 8. 연습문제

### 연습문제 1: TLB 성능 측정

시스템에서 TLB 영향을 측정하세요:
1. N개의 페이지를 랜덤 순서로 접근하는 프로그램을 작성하세요
2. N을 10에서 100,000 페이지까지 변화시키세요
3. 요소당 접근 시간을 측정하세요
4. 그래프로 그리기: 고유 페이지 수 vs 접근 시간
5. TLB 용량을 식별하세요 (성능이 떨어지는 지점)

### 연습문제 2: Huge Page 벤치마크

일반 페이지 vs huge pages를 비교하세요:
1. 일반 페이지(4 KB)로 1 GB 할당
2. Huge pages(2 MB)로 1 GB 할당
3. 두 가지 모두에 대해 랜덤 접근 벤치마크 수행
4. `perf stat`을 사용하여 각각의 TLB 미스 측정
5. Huge pages로 인한 성능 향상을 계산하세요

### 연습문제 3: NUMA 인식 할당자

NUMA 인식 메모리 할당자를 구축하세요:
1. libnuma를 사용하여 NUMA 토폴로지 감지
2. 특정 노드에 할당하는 `numa_malloc(size, node)` 구현
3. 벤치마크: 로컬 할당 vs 리모트 할당
4. 큰 공유 버퍼를 위한 인터리브 할당 구현
5. 로컬 접근과 리모트 접근의 지연 시간 차이를 보여주세요

### 연습문제 4: Memory-Mapped 파일 처리

mmap을 사용한 고성능 파일 처리기를 구축하세요:
1. 랜덤 정수가 포함된 1 GB 테스트 파일 생성
2. read()로 처리: 임계값보다 큰 값 세기
3. mmap()으로 처리: 동일한 작업
4. 비교: 처리량, 시스템 호출(strace), 페이지 폴트
5. madvise() 힌트를 추가하고 개선 효과를 측정하세요

### 연습문제 5: COW Fork 분석기

Copy-on-write 동작을 분석하세요:
1. 다양한 양의 메모리 할당 (100 MB, 500 MB, 1 GB)
2. Fork하고 fork() 시간 측정
3. 자식에서: 0%, 10%, 50%, 100%의 페이지 수정
4. 부모와 자식의 RSS (상주 세트 크기) 모니터링
5. COW 페이지 폴트를 보여주는 시간에 따른 메모리 사용량 그래프

---

*레슨 20 끝*
