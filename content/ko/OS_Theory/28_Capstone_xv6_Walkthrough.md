[이전: 현대 스케줄러](./27_Modern_Schedulers.md)

---

# 28. 캡스톤: xv6 커널 워크스루

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. xv6 커널 소스 코드를 탐색하고 그 구조를 이해할 수 있다
2. 전원 켜기부터 첫 번째 사용자 프로세스까지의 부팅 과정을 추적할 수 있다
3. 페이지 테이블과 할당을 포함한 메모리 관리 서브시스템을 분석할 수 있다
4. 프로세스 생성, 스케줄링, 시스템 호출 구현을 설명할 수 있다
5. 디스크 레이아웃부터 파일 작업까지 xv6 파일 시스템 구조를 설명할 수 있다

---

## 목차

1. [xv6 소개](#1-xv6-소개)
2. [부팅 과정](#2-부팅-과정)
3. [메모리 관리](#3-메모리-관리)
4. [프로세스 관리](#4-프로세스-관리)
5. [시스템 호출](#5-시스템-호출)
6. [파일 시스템](#6-파일-시스템)
7. [트랩과 인터럽트](#7-트랩과-인터럽트)
8. [캡스톤 프로젝트](#8-캡스톤-프로젝트)

---

## 1. xv6 소개

### 1.1 xv6란?

```
xv6: MIT의 교육용 운영체제.

Unix V6 (1975) 기반으로, x86/RISC-V용 ANSI C로 재작성.
MIT 6.828 / 6.S081 및 수백 개의 다른 OS 과정에서 사용.

xv6를 공부하는 이유:
  - ~8,000줄의 C로 된 완전한, 작동하는 Unix 유사 OS
  - 우리가 공부한 모든 핵심 OS 개념을 구현
  - 며칠이면 전체를 읽을 수 있을 만큼 작음
  - 현실적일 만큼 충분히 복잡
  - QEMU에서 실행 (실제 하드웨어 불필요)

xv6-riscv 소스: https://github.com/mit-pdos/xv6-riscv

파일 구조:
  kernel/
  ├── main.c          # 커널 진입점
  ├── proc.c          # 프로세스 관리
  ├── proc.h          # 프로세스 구조체
  ├── vm.c            # 가상 메모리
  ├── kalloc.c        # 물리 메모리 할당자
  ├── trap.c          # 트랩/인터럽트 처리
  ├── syscall.c       # 시스템 호출 디스패처
  ├── sysfile.c       # 파일 시스템 호출
  ├── sysproc.c       # 프로세스 시스템 호출
  ├── fs.c            # 파일 시스템
  ├── bio.c           # 블록 I/O (버퍼 캐시)
  ├── log.c           # 크래시 복구를 위한 로깅
  ├── pipe.c          # 파이프 구현
  ├── spinlock.c      # 스핀락 구현
  ├── sleeplock.c     # 슬립 락
  ├── console.c       # 콘솔 I/O
  └── uart.c          # 시리얼 포트 드라이버
  user/
  ├── sh.c            # 셸
  ├── ls.c            # ls 명령어
  └── ...             # 기타 사용자 프로그램
```

---

## 2. 부팅 과정

### 2.1 전원 켜기부터 main()까지

```
xv6-riscv 부팅 시퀀스:

1. QEMU가 커널을 0x80000000에 로드
   하드웨어가 PC를 진입점으로 설정

2. entry.S: 각 CPU의 스택 설정
   la sp, stack0
   li a0, 1024*4   # CPU당 4096 바이트 스택
   csrr a1, mhartid
   addi a1, a1, 1
   mul a0, a0, a1
   add sp, sp, a0
   call start

3. start.c: 머신 모드 설정
   - mstatus를 슈퍼바이저 모드로 설정
   - mepc를 main으로 설정
   - 부팅용 페이지 테이블 설정
   - 타이머 인터럽트 설정
   - mret으로 슈퍼바이저 모드 → main()

4. main.c: 커널 초기화
```

### 2.2 main() 초기화

```c
/*
 * xv6 main.c - 커널 초기화 시퀀스
 * (xv6-riscv 소스에서 단순화)
 */

void main(void)
{
    if (cpuid() == 0) {
        /* CPU 0만 일회성 초기화 수행 */
        consoleinit();    /* 콘솔 (UART) */
        printfinit();     /* Printf */
        printf("\nxv6 kernel is booting\n\n");

        kinit();          /* 물리 메모리 할당자 */
        kvminit();        /* 커널 페이지 테이블 */
        kvminithart();    /* 페이징 켜기 */
        procinit();       /* 프로세스 테이블 */
        trapinit();       /* 트랩 벡터 */
        trapinithart();   /* CPU별 트랩 설정 */
        plicinit();       /* 인터럽트 컨트롤러 */
        plicinithart();   /* CPU별 인터럽트 설정 */
        binit();          /* 버퍼 캐시 */
        iinit();          /* 아이노드 테이블 */
        fileinit();       /* 파일 테이블 */
        virtio_disk_init(); /* 디스크 드라이버 */
        userinit();       /* 첫 번째 사용자 프로세스! */

        /* 다른 CPU에 시작 신호 */
        __sync_synchronize();
        started = 1;
    } else {
        /* 다른 CPU는 대기 후 설정 */
        while (started == 0)
            ;
        __sync_synchronize();
        kvminithart();
        trapinithart();
        plicinithart();
    }

    scheduler();  /* 반환하지 않음 - 스케줄러 루프 실행 */
}
```

---

## 3. 메모리 관리

### 3.1 물리 메모리 할당자

```c
/*
 * xv6 kalloc.c: 프리 리스트 기반 물리 페이지 할당자.
 *
 * 메모리 레이아웃:
 *   0x80000000: 커널 코드/데이터 시작
 *   ...
 *   end:        커널 끝 (링커가 정의)
 *   ...
 *   PHYSTOP:    물리 메모리 끝 (128 MB)
 *
 * 빈 페이지는 프리 리스트로 연결.
 * 각 빈 페이지의 첫 8바이트가 다음 빈 페이지를 가리킴.
 */

struct run {
    struct run *next;
};

struct {
    struct spinlock lock;
    struct run *freelist;
} kmem;

/* 물리 페이지 해제 */
void kfree(void *pa) {
    struct run *r;

    /* use-after-free를 잡기 위해 쓰레기 값으로 채우기 */
    memset(pa, 1, PGSIZE);

    acquire(&kmem.lock);
    r = (struct run *)pa;
    r->next = kmem.freelist;
    kmem.freelist = r;
    release(&kmem.lock);
}

/* 물리 페이지 하나 할당. 메모리 부족 시 0 반환. */
void *kalloc(void) {
    struct run *r;

    acquire(&kmem.lock);
    r = kmem.freelist;
    if (r)
        kmem.freelist = r->next;
    release(&kmem.lock);

    if (r)
        memset((char *)r, 5, PGSIZE);  /* 쓰레기 값으로 채우기 */
    return (void *)r;
}
```

### 3.2 페이지 테이블

```c
/*
 * xv6 vm.c: RISC-V Sv39 3단계 페이지 테이블.
 *
 * 가상 주소 (39비트):
 *   [38:30] L2 인덱스 (9비트) → 루트 페이지 테이블
 *   [29:21] L1 인덱스 (9비트) → 두 번째 레벨
 *   [20:12] L0 인덱스 (9비트) → 세 번째 레벨
 *   [11:0]  페이지 오프셋 (12비트)
 *
 * 각 PTE: 54비트
 *   [53:10] 물리 페이지 번호
 *   [9:0]   플래그 (V, R, W, X, U 등)
 */

/* 프로세스를 위한 사용자 페이지 테이블 생성 */
pagetable_t uvmcreate(void) {
    pagetable_t pagetable;
    pagetable = (pagetable_t)kalloc();
    if (pagetable == 0)
        return 0;
    memset(pagetable, 0, PGSIZE);
    return pagetable;
}

/* 페이지 테이블에 페이지 매핑 */
int mappages(pagetable_t pagetable, uint64 va, uint64 size,
             uint64 pa, int perm)
{
    uint64 a, last;
    pte_t *pte;

    a = PGROUNDDOWN(va);
    last = PGROUNDDOWN(va + size - 1);

    for (;;) {
        if ((pte = walk(pagetable, a, 1)) == 0)
            return -1;
        if (*pte & PTE_V)
            panic("mappages: remap");
        *pte = PA2PTE(pa) | perm | PTE_V;

        if (a == last) break;
        a += PGSIZE;
        pa += PGSIZE;
    }
    return 0;
}
```

---

## 4. 프로세스 관리

### 4.1 프로세스 구조체

```c
/*
 * xv6 proc.h: 프로세스 제어 블록
 */

enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

struct proc {
    struct spinlock lock;

    /* 프로세스 상태 */
    enum procstate state;
    int pid;
    int killed;
    int xstate;           /* 종료 상태 */

    /* 스케줄링 */
    struct proc *parent;
    void *chan;            /* 슬립 채널 */

    /* 메모리 */
    pagetable_t pagetable; /* 사용자 페이지 테이블 */
    uint64 sz;             /* 프로세스 메모리 크기 */
    struct trapframe *trapframe; /* 저장된 레지스터 */

    /* swtch()를 위한 컨텍스트 */
    struct context context;

    /* 파일 시스템 */
    struct file *ofile[NOFILE]; /* 열린 파일 */
    struct inode *cwd;          /* 현재 디렉토리 */
    char name[16];              /* 프로세스 이름 */
};
```

### 4.2 Fork 구현

```c
/*
 * xv6 proc.c: fork()는 현재 프로세스의 복사본을 생성.
 */

int fork(void) {
    int i, pid;
    struct proc *np;  /* 새 프로세스 */
    struct proc *p = myproc();  /* 현재 프로세스 */

    /* 프로세스 슬롯 할당 */
    if ((np = allocproc()) == 0)
        return -1;

    /* 사용자 메모리 복사 (부모 → 자식) */
    if (uvmcopy(p->pagetable, np->pagetable, p->sz) < 0) {
        freeproc(np);
        release(&np->lock);
        return -1;
    }
    np->sz = p->sz;

    /* 저장된 레지스터 복사 (자식은 fork에서 0 반환) */
    *(np->trapframe) = *(p->trapframe);
    np->trapframe->a0 = 0;  /* 자식에서 fork는 0 반환 */

    /* 열린 파일 디스크립터 복사 */
    for (i = 0; i < NOFILE; i++) {
        if (p->ofile[i])
            np->ofile[i] = filedup(p->ofile[i]);
    }
    np->cwd = idup(p->cwd);

    safestrcpy(np->name, p->name, sizeof(p->name));
    pid = np->pid;

    release(&np->lock);

    /* 부모 설정 */
    acquire(&wait_lock);
    np->parent = p;
    release(&wait_lock);

    /* 자식을 실행 가능 상태로 */
    acquire(&np->lock);
    np->state = RUNNABLE;
    release(&np->lock);

    return pid;  /* 부모는 자식의 PID 반환 */
}
```

### 4.3 스케줄러

```c
/*
 * xv6 proc.c: 라운드 로빈 스케줄러.
 * 각 CPU가 이 루프를 영원히 실행.
 */

void scheduler(void) {
    struct proc *p;
    struct cpu *c = mycpu();

    c->proc = 0;
    for (;;) {
        /* 데드락을 피하기 위해 인터럽트 활성화 */
        intr_on();

        /* 모든 프로세스를 순회하며 RUNNABLE 찾기 */
        for (p = proc; p < &proc[NPROC]; p++) {
            acquire(&p->lock);
            if (p->state == RUNNABLE) {
                p->state = RUNNING;
                c->proc = p;

                /* 프로세스 페이지 테이블과 컨텍스트로 전환 */
                swtch(&c->context, &p->context);

                /* 프로세스 실행 완료 (현재로서는) */
                c->proc = 0;
            }
            release(&p->lock);
        }
    }
}
```

---

## 5. 시스템 호출

### 5.1 시스템 호출 경로

```
사용자 프로그램이 write(fd, buf, n) 호출:

1. user/usys.S (생성됨):
   write:
     li a7, SYS_write    # 시스콜 번호를 a7에
     ecall               # 커널로 트랩
     ret

2. kernel/trap.c: usertrap()
   ecall 감지, syscall() 호출

3. kernel/syscall.c: syscall()
   trapframe에서 a7 읽기
   sys_write()로 디스패치

4. kernel/sysfile.c: sys_write()
   trapframe에서 인자 읽기 (fd, buf, n)
   filewrite() 호출

5. usertrapret()를 통해 사용자 공간으로 복귀
```

### 5.2 새 시스템 호출 추가

```c
/*
 * xv6에 새 시스템 호출을 추가하는 단계:
 *
 * 1. kernel/syscall.h에 시스콜 번호 추가:
 *    #define SYS_mysyscall 22
 *
 * 2. kernel/syscall.c에 함수 프로토타입 추가:
 *    extern uint64 sys_mysyscall(void);
 *    [SYS_mysyscall] sys_mysyscall,
 *
 * 3. kernel/sysproc.c에 구현:
 */

uint64 sys_mysyscall(void) {
    int arg;
    argint(0, &arg);  /* 첫 번째 인자 읽기 */

    printf("mysyscall called with arg=%d by pid=%d\n",
           arg, myproc()->pid);

    return arg * 2;  /* 반환값 */
}

/*
 * 4. user/usys.pl에 사용자 공간 스텁 추가:
 *    entry("mysyscall");
 *
 * 5. user/user.h에 선언 추가:
 *    int mysyscall(int);
 *
 * 6. 사용자 프로그램에서 사용:
 *    int result = mysyscall(42);  // 84 반환
 */
```

---

## 6. 파일 시스템

### 6.1 디스크 레이아웃

```
xv6 파일 시스템 레이아웃:

  블록 0: 부트 섹터 (xv6에서 미사용)
  블록 1: 슈퍼블록 (파일시스템 메타데이터)
  블록 2-31: 로그 블록 (크래시 복구용)
  블록 32-44: 아이노드 블록 (200개 아이노드)
  블록 45: 비트맵 블록 (빈 블록 추적)
  블록 46+: 데이터 블록

  아이노드 구조:
  ┌──────────────────────────────────┐
  │ type   │ nlinks │ size           │
  ├──────────────────────────────────┤
  │ addrs[0]  → 데이터 블록 0        │
  │ addrs[1]  → 데이터 블록 1        │
  │ ...                              │
  │ addrs[11] → 데이터 블록 11       │  ← 12개 직접 블록
  │ addrs[12] → 간접 블록 ──┐       │  ← 1개 간접 블록
  └──────────────────────────│──────┘
                              ▼
                ┌──────────────────┐
                │ 블록 포인터 0     │
                │ 블록 포인터 1     │
                │ ...              │  256개의 추가 블록 포인터
                │ 블록 포인터 255   │
                └──────────────────┘

  최대 파일 크기: (12 + 256) × 1024 = 274,432 바이트
```

### 6.2 파일 작업

```c
/*
 * xv6 fs.c: 핵심 파일 시스템 작업.
 *
 * 레이어:
 *   시스템 호출 (sysfile.c)
 *     ↓
 *   파일 디스크립터 (file.c)
 *     ↓
 *   아이노드 (fs.c)
 *     ↓
 *   로깅 (log.c)
 *     ↓
 *   버퍼 캐시 (bio.c)
 *     ↓
 *   디스크 드라이버 (virtio_disk.c)
 */

/* 아이노드에서 읽기 (단순화) */
int readi(struct inode *ip, int user_dst, uint64 dst,
          uint off, uint n)
{
    uint tot, m;
    struct buf *bp;

    for (tot = 0; tot < n; tot += m, off += m, dst += m) {
        uint addr = bmap(ip, off / BSIZE);
        if (addr == 0) break;

        bp = bread(ip->dev, addr);  /* 디스크에서 블록 읽기 */
        m = min(n - tot, BSIZE - off % BSIZE);

        if (either_copyout(user_dst, dst,
                           bp->data + (off % BSIZE), m) == -1) {
            brelse(bp);
            tot = -1;
            break;
        }
        brelse(bp);  /* 버퍼 해제 */
    }
    return tot;
}
```

---

## 7. 트랩과 인터럽트

### 7.1 트랩 처리

```c
/*
 * xv6 trap.c: 통합 트랩 처리.
 *
 * 세 가지 유형의 트랩:
 *   1. 시스템 호출 (ecall 명령어)
 *   2. 예외 (페이지 폴트, 잘못된 명령어)
 *   3. 디바이스 인터럽트 (타이머, UART, 디스크)
 *
 * 흐름:
 *   트랩 발생 → uservec (레지스터 저장)
 *   → usertrap() (트랩 처리)
 *   → usertrapret() (복원 후 반환)
 */

void usertrap(void) {
    struct proc *p = myproc();

    /* 사용자 프로그램 카운터 저장 */
    p->trapframe->epc = r_sepc();

    if (r_scause() == 8) {
        /* 시스템 호출 */
        if (killed(p))
            exit(-1);

        /* ecall 명령어를 지나도록 PC 전진 */
        p->trapframe->epc += 4;

        intr_on();  /* 시스콜 중 인터럽트 활성화 */
        syscall();

    } else if ((r_scause() & 0x8000000000000000L) &&
               (r_scause() & 0xff) == 9) {
        /* 디바이스 인터럽트 */
        int irq = plic_claim();
        if (irq == UART0_IRQ) {
            uartintr();
        } else if (irq == VIRTIO0_IRQ) {
            virtio_disk_intr();
        }
        if (irq) plic_complete(irq);

    } else {
        /* 예외 (예: 페이지 폴트) */
        printf("usertrap(): unexpected scause=0x%lx pid=%d\n",
               r_scause(), p->pid);
        setkilled(p);
    }

    if (killed(p))
        exit(-1);

    /* 타이머 인터럽트 → CPU 양보 */
    if (which_dev == 2)
        yield();

    usertrapret();
}
```

---

## 8. 캡스톤 프로젝트

### 프로젝트 A: Copy-on-Write Fork 추가

xv6에 COW fork 구현:
1. fork()를 수정하여 복사 대신 페이지 공유
2. 부모와 자식 모두에서 공유 페이지를 읽기 전용으로 표시
3. 페이지 폴트 처리: 쓰기 시 페이지를 복사하고 다시 매핑
4. 공유 페이지에 대한 참조 카운트 추적
5. 테스트: fork가 더 빠르고 메모리 사용량이 더 낮은지 검증
6. 에지 케이스: 같은 페이지를 공유하는 여러 fork

### 프로젝트 B: 지연 페이지 할당 구현

sbrk()에 지연 할당 추가:
1. sbrk()를 수정하여 프로세스 크기만 업데이트 (할당하지 않음)
2. 페이지 폴트 처리: 폴트가 발생한 페이지를 온디맨드로 할당 및 매핑
3. 잘못된 접근 처리: sbrk 경계를 넘는 접근 시 프로세스 종료
4. 테스트: 1 GB 할당, 1 MB만 접근, 낮은 메모리 사용량 검증
5. 벤치마크: 즉시 할당 vs 지연 할당 비교

### 프로젝트 C: 로그 구조 파일 시스템 추가

xv6의 파일 시스템을 로그 구조로 교체:
1. 로그 구조 레이아웃 설계: 모든 쓰기가 로그 끝으로
2. 세그먼트 구조와 쓰기 버퍼링 구현
3. 가비지 컬렉션 (세그먼트 정리) 구현
4. 빠른 조회를 위한 아이노드 맵 유지
5. 벤치마크: 순차 쓰기 처리량 개선

### 프로젝트 D: 가상 메모리 기능 구현

xv6에 고급 VM 추가:
1. 메모리 매핑 파일을 위한 mmap() 구현
2. 영역 해제를 위한 munmap() 구현
3. MAP_SHARED와 MAP_PRIVATE 지원
4. 디맨드 페이징 mmap을 위한 페이지 폴트 처리
5. mmap을 데이터 파일에 사용하는 간단한 데이터베이스로 테스트

### 프로젝트 E: 네트워크 스택 추가

xv6를 위한 최소 네트워크 스택 구축:
1. 이더넷 프레임 전송/수신 구현 (virtio-net)
2. ARP (주소 해석) 구현
3. IP (패킷 라우팅) 구현
4. UDP (간단한 데이터그램 프로토콜) 구현
5. 간단한 에코 서버와 클라이언트 구축
6. 보너스: TCP 연결 설정 구현

---

## 추가 자료

### xv6 리소스
- [xv6 Book (MIT)](https://pdos.csail.mit.edu/6.828/2023/xv6/book-riscv-rev3.pdf)
- [xv6 Source (GitHub)](https://github.com/mit-pdos/xv6-riscv)
- [MIT 6.S081 Labs](https://pdos.csail.mit.edu/6.828/2023/schedule.html)

### 관련 과정
- MIT 6.S081: Operating System Engineering
- Stanford CS140: Operating Systems
- University of Wisconsin: OSTEP

---

*레슨 28 끝 - 운영체제 이론 과정을 완료하셨습니다!*
