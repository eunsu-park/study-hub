[Previous: Modern Schedulers](./27_Modern_Schedulers.md)

---

# 28. Capstone: xv6 Kernel Walkthrough

## Learning Objectives

After completing this lesson, you will be able to:

1. Navigate the xv6 kernel source code and understand its organization
2. Trace the boot process from power-on to first user process
3. Analyze the memory management subsystem including page tables and allocation
4. Explain process creation, scheduling, and system call implementation
5. Describe the xv6 file system structure from disk layout to file operations

---

## Table of Contents

1. [Introduction to xv6](#1-introduction-to-xv6)
2. [Boot Process](#2-boot-process)
3. [Memory Management](#3-memory-management)
4. [Process Management](#4-process-management)
5. [System Calls](#5-system-calls)
6. [File System](#6-file-system)
7. [Traps and Interrupts](#7-traps-and-interrupts)
8. [Capstone Projects](#8-capstone-projects)

---

## 1. Introduction to xv6

### 1.1 What Is xv6?

```
xv6: A teaching operating system from MIT.

Based on Unix V6 (1975), rewritten in ANSI C for x86/RISC-V.
Used in MIT 6.828 / 6.S081 and hundreds of other OS courses.

Why study xv6?
  - Complete, working Unix-like OS in ~8,000 lines of C
  - Implements all core OS concepts we've studied
  - Small enough to read entirely in a few days
  - Complex enough to be realistic
  - Runs on QEMU (no real hardware needed)

xv6-riscv source: https://github.com/mit-pdos/xv6-riscv

File structure:
  kernel/
  ├── main.c          # Kernel entry point
  ├── proc.c          # Process management
  ├── proc.h          # Process structures
  ├── vm.c            # Virtual memory
  ├── kalloc.c        # Physical memory allocator
  ├── trap.c          # Trap/interrupt handling
  ├── syscall.c       # System call dispatcher
  ├── sysfile.c       # File system calls
  ├── sysproc.c       # Process system calls
  ├── fs.c            # File system
  ├── bio.c           # Block I/O (buffer cache)
  ├── log.c           # Logging for crash recovery
  ├── pipe.c          # Pipe implementation
  ├── spinlock.c      # Spinlock implementation
  ├── sleeplock.c     # Sleep lock
  ├── console.c       # Console I/O
  └── uart.c          # Serial port driver
  user/
  ├── sh.c            # Shell
  ├── ls.c            # ls command
  └── ...             # Other user programs
```

---

## 2. Boot Process

### 2.1 From Power On to main()

```
xv6-riscv boot sequence:

1. QEMU loads kernel at 0x80000000
   Hardware sets PC to entry point

2. entry.S: Set up stack for each CPU
   la sp, stack0
   li a0, 1024*4   # 4096 bytes per CPU stack
   csrr a1, mhartid
   addi a1, a1, 1
   mul a0, a0, a1
   add sp, sp, a0
   call start

3. start.c: Machine mode setup
   - Set mstatus to supervisor mode
   - Set mepc to main
   - Set up page table for boot
   - Timer interrupt setup
   - mret to supervisor mode → main()

4. main.c: Kernel initialization
```

### 2.2 main() Initialization

```c
/*
 * xv6 main.c - kernel initialization sequence
 * (Simplified from xv6-riscv source)
 */

void main(void)
{
    if (cpuid() == 0) {
        /* Only CPU 0 does one-time initialization */
        consoleinit();    /* Console (UART) */
        printfinit();     /* Printf */
        printf("\nxv6 kernel is booting\n\n");

        kinit();          /* Physical memory allocator */
        kvminit();        /* Kernel page table */
        kvminithart();    /* Turn on paging */
        procinit();       /* Process table */
        trapinit();       /* Trap vectors */
        trapinithart();   /* Per-CPU trap setup */
        plicinit();       /* Interrupt controller */
        plicinithart();   /* Per-CPU interrupt setup */
        binit();          /* Buffer cache */
        iinit();          /* Inode table */
        fileinit();       /* File table */
        virtio_disk_init(); /* Disk driver */
        userinit();       /* First user process! */

        /* Signal other CPUs to start */
        __sync_synchronize();
        started = 1;
    } else {
        /* Other CPUs wait then set up */
        while (started == 0)
            ;
        __sync_synchronize();
        kvminithart();
        trapinithart();
        plicinithart();
    }

    scheduler();  /* Never returns - runs scheduler loop */
}
```

---

## 3. Memory Management

### 3.1 Physical Memory Allocator

```c
/*
 * xv6 kalloc.c: Free-list based physical page allocator.
 *
 * Memory layout:
 *   0x80000000: Kernel code/data start
 *   ...
 *   end:        End of kernel (defined by linker)
 *   ...
 *   PHYSTOP:    End of physical memory (128 MB)
 *
 * Free pages are linked in a free list.
 * Each free page's first 8 bytes point to the next free page.
 */

struct run {
    struct run *next;
};

struct {
    struct spinlock lock;
    struct run *freelist;
} kmem;

/* Free a physical page */
void kfree(void *pa) {
    struct run *r;

    /* Fill with junk to catch use-after-free */
    memset(pa, 1, PGSIZE);

    acquire(&kmem.lock);
    r = (struct run *)pa;
    r->next = kmem.freelist;
    kmem.freelist = r;
    release(&kmem.lock);
}

/* Allocate one physical page. Returns 0 if out of memory. */
void *kalloc(void) {
    struct run *r;

    acquire(&kmem.lock);
    r = kmem.freelist;
    if (r)
        kmem.freelist = r->next;
    release(&kmem.lock);

    if (r)
        memset((char *)r, 5, PGSIZE);  /* Fill with junk */
    return (void *)r;
}
```

### 3.2 Page Tables

```c
/*
 * xv6 vm.c: RISC-V Sv39 three-level page tables.
 *
 * Virtual address (39-bit):
 *   [38:30] L2 index (9 bits) → root page table
 *   [29:21] L1 index (9 bits) → second level
 *   [20:12] L0 index (9 bits) → third level
 *   [11:0]  Page offset (12 bits)
 *
 * Each PTE: 54 bits
 *   [53:10] Physical page number
 *   [9:0]   Flags (V, R, W, X, U, etc.)
 */

/* Create a user page table for a process */
pagetable_t uvmcreate(void) {
    pagetable_t pagetable;
    pagetable = (pagetable_t)kalloc();
    if (pagetable == 0)
        return 0;
    memset(pagetable, 0, PGSIZE);
    return pagetable;
}

/* Map pages in a page table */
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

## 4. Process Management

### 4.1 Process Structure

```c
/*
 * xv6 proc.h: Process control block
 */

enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

struct proc {
    struct spinlock lock;

    /* Process state */
    enum procstate state;
    int pid;
    int killed;
    int xstate;           /* Exit status */

    /* Scheduling */
    struct proc *parent;
    void *chan;            /* Sleep channel */

    /* Memory */
    pagetable_t pagetable; /* User page table */
    uint64 sz;             /* Size of process memory */
    struct trapframe *trapframe; /* Saved registers */

    /* Context for swtch() */
    struct context context;

    /* File system */
    struct file *ofile[NOFILE]; /* Open files */
    struct inode *cwd;          /* Current directory */
    char name[16];              /* Process name */
};
```

### 4.2 Fork Implementation

```c
/*
 * xv6 proc.c: fork() creates a copy of the current process.
 */

int fork(void) {
    int i, pid;
    struct proc *np;  /* New process */
    struct proc *p = myproc();  /* Current process */

    /* Allocate process slot */
    if ((np = allocproc()) == 0)
        return -1;

    /* Copy user memory (parent → child) */
    if (uvmcopy(p->pagetable, np->pagetable, p->sz) < 0) {
        freeproc(np);
        release(&np->lock);
        return -1;
    }
    np->sz = p->sz;

    /* Copy saved registers (child returns 0 from fork) */
    *(np->trapframe) = *(p->trapframe);
    np->trapframe->a0 = 0;  /* fork returns 0 in child */

    /* Copy open file descriptors */
    for (i = 0; i < NOFILE; i++) {
        if (p->ofile[i])
            np->ofile[i] = filedup(p->ofile[i]);
    }
    np->cwd = idup(p->cwd);

    safestrcpy(np->name, p->name, sizeof(p->name));
    pid = np->pid;

    release(&np->lock);

    /* Set parent */
    acquire(&wait_lock);
    np->parent = p;
    release(&wait_lock);

    /* Make child runnable */
    acquire(&np->lock);
    np->state = RUNNABLE;
    release(&np->lock);

    return pid;  /* Parent returns child's PID */
}
```

### 4.3 Scheduler

```c
/*
 * xv6 proc.c: Round-robin scheduler.
 * Each CPU runs this loop forever.
 */

void scheduler(void) {
    struct proc *p;
    struct cpu *c = mycpu();

    c->proc = 0;
    for (;;) {
        /* Enable interrupts to avoid deadlock */
        intr_on();

        /* Loop over all processes, find RUNNABLE */
        for (p = proc; p < &proc[NPROC]; p++) {
            acquire(&p->lock);
            if (p->state == RUNNABLE) {
                p->state = RUNNING;
                c->proc = p;

                /* Switch to process page table and context */
                swtch(&c->context, &p->context);

                /* Process is done running (for now) */
                c->proc = 0;
            }
            release(&p->lock);
        }
    }
}
```

---

## 5. System Calls

### 5.1 System Call Path

```
User program calls write(fd, buf, n):

1. user/usys.S (generated):
   write:
     li a7, SYS_write    # syscall number in a7
     ecall               # trap to kernel
     ret

2. kernel/trap.c: usertrap()
   Detects ecall, calls syscall()

3. kernel/syscall.c: syscall()
   Reads a7 from trapframe
   Dispatches to sys_write()

4. kernel/sysfile.c: sys_write()
   Reads arguments from trapframe (fd, buf, n)
   Calls filewrite()

5. Returns to user space via usertrapret()
```

### 5.2 Adding a New System Call

```c
/*
 * Steps to add a new system call to xv6:
 *
 * 1. Add syscall number in kernel/syscall.h:
 *    #define SYS_mysyscall 22
 *
 * 2. Add function prototype in kernel/syscall.c:
 *    extern uint64 sys_mysyscall(void);
 *    [SYS_mysyscall] sys_mysyscall,
 *
 * 3. Implement in kernel/sysproc.c:
 */

uint64 sys_mysyscall(void) {
    int arg;
    argint(0, &arg);  /* Read first argument */

    printf("mysyscall called with arg=%d by pid=%d\n",
           arg, myproc()->pid);

    return arg * 2;  /* Return value */
}

/*
 * 4. Add user-space stub in user/usys.pl:
 *    entry("mysyscall");
 *
 * 5. Add declaration in user/user.h:
 *    int mysyscall(int);
 *
 * 6. Use in user program:
 *    int result = mysyscall(42);  // Returns 84
 */
```

---

## 6. File System

### 6.1 Disk Layout

```
xv6 File System Layout:

  Block 0: Boot sector (unused by xv6)
  Block 1: Superblock (filesystem metadata)
  Block 2-31: Log blocks (for crash recovery)
  Block 32-44: Inode blocks (200 inodes)
  Block 45: Bitmap block (free block tracking)
  Block 46+: Data blocks

  Inode structure:
  ┌──────────────────────────────────┐
  │ type   │ nlinks │ size           │
  ├──────────────────────────────────┤
  │ addrs[0]  → data block 0        │
  │ addrs[1]  → data block 1        │
  │ ...                              │
  │ addrs[11] → data block 11       │  ← 12 direct blocks
  │ addrs[12] → indirect block ──┐  │  ← 1 indirect block
  └──────────────────────────────│──┘
                                 ▼
                   ┌──────────────────┐
                   │ block ptr 0      │
                   │ block ptr 1      │
                   │ ...              │  256 more block pointers
                   │ block ptr 255    │
                   └──────────────────┘

  Max file size: (12 + 256) × 1024 = 274,432 bytes
```

### 6.2 File Operations

```c
/*
 * xv6 fs.c: Core file system operations.
 *
 * Layers:
 *   System calls (sysfile.c)
 *     ↓
 *   File descriptors (file.c)
 *     ↓
 *   Inodes (fs.c)
 *     ↓
 *   Logging (log.c)
 *     ↓
 *   Buffer cache (bio.c)
 *     ↓
 *   Disk driver (virtio_disk.c)
 */

/* Read from an inode (simplified) */
int readi(struct inode *ip, int user_dst, uint64 dst,
          uint off, uint n)
{
    uint tot, m;
    struct buf *bp;

    for (tot = 0; tot < n; tot += m, off += m, dst += m) {
        uint addr = bmap(ip, off / BSIZE);
        if (addr == 0) break;

        bp = bread(ip->dev, addr);  /* Read block from disk */
        m = min(n - tot, BSIZE - off % BSIZE);

        if (either_copyout(user_dst, dst,
                           bp->data + (off % BSIZE), m) == -1) {
            brelse(bp);
            tot = -1;
            break;
        }
        brelse(bp);  /* Release buffer */
    }
    return tot;
}
```

---

## 7. Traps and Interrupts

### 7.1 Trap Handling

```c
/*
 * xv6 trap.c: Unified trap handling.
 *
 * Three types of traps:
 *   1. System call (ecall instruction)
 *   2. Exception (page fault, illegal instruction)
 *   3. Device interrupt (timer, UART, disk)
 *
 * Flow:
 *   Trap occurs → uservec (save registers)
 *   → usertrap() (handle trap)
 *   → usertrapret() (restore and return)
 */

void usertrap(void) {
    struct proc *p = myproc();

    /* Save user program counter */
    p->trapframe->epc = r_sepc();

    if (r_scause() == 8) {
        /* System call */
        if (killed(p))
            exit(-1);

        /* Advance PC past ecall instruction */
        p->trapframe->epc += 4;

        intr_on();  /* Enable interrupts during syscall */
        syscall();

    } else if ((r_scause() & 0x8000000000000000L) &&
               (r_scause() & 0xff) == 9) {
        /* Device interrupt */
        int irq = plic_claim();
        if (irq == UART0_IRQ) {
            uartintr();
        } else if (irq == VIRTIO0_IRQ) {
            virtio_disk_intr();
        }
        if (irq) plic_complete(irq);

    } else {
        /* Exception (e.g., page fault) */
        printf("usertrap(): unexpected scause=0x%lx pid=%d\n",
               r_scause(), p->pid);
        setkilled(p);
    }

    if (killed(p))
        exit(-1);

    /* Timer interrupt → yield CPU */
    if (which_dev == 2)
        yield();

    usertrapret();
}
```

---

## 8. Capstone Projects

### Project A: Add Copy-on-Write Fork

Implement COW fork in xv6:
1. Modify fork() to share pages instead of copying
2. Mark shared pages as read-only in both parent and child
3. Handle page fault: on write, copy the page and remap
4. Track reference counts for shared pages
5. Test: verify fork is faster and memory usage is lower
6. Edge case: multiple forks sharing the same page

### Project B: Implement Lazy Page Allocation

Add lazy allocation to sbrk():
1. Modify sbrk() to only update the process size (don't allocate)
2. Handle page fault: allocate and map the faulting page on demand
3. Handle invalid access: kill process if access beyond sbrk boundary
4. Test: allocate 1 GB, only touch 1 MB, verify low memory usage
5. Benchmark: compare eager vs lazy allocation

### Project C: Add a Log-Structured File System

Replace xv6's file system with a log-structured one:
1. Design log-structured layout: all writes go to log tail
2. Implement segment structure and write buffering
3. Implement garbage collection (segment cleaning)
4. Maintain an inode map for fast lookups
5. Benchmark: sequential write throughput improvement

### Project D: Implement Virtual Memory Features

Add advanced VM to xv6:
1. Implement mmap() for memory-mapped files
2. Implement munmap() to unmap regions
3. Support MAP_SHARED and MAP_PRIVATE
4. Handle page faults for demand-paged mmap
5. Test with a simple database that uses mmap for data files

### Project E: Add Network Stack

Build a minimal network stack for xv6:
1. Implement Ethernet frame send/receive (virtio-net)
2. Implement ARP (address resolution)
3. Implement IP (packet routing)
4. Implement UDP (simple datagram protocol)
5. Build a simple echo server and client
6. Bonus: implement TCP connection establishment

---

## Further Reading

### xv6 Resources
- [xv6 Book (MIT)](https://pdos.csail.mit.edu/6.828/2023/xv6/book-riscv-rev3.pdf)
- [xv6 Source (GitHub)](https://github.com/mit-pdos/xv6-riscv)
- [MIT 6.S081 Labs](https://pdos.csail.mit.edu/6.828/2023/schedule.html)

### Related Courses
- MIT 6.S081: Operating System Engineering
- Stanford CS140: Operating Systems
- University of Wisconsin: OSTEP

---

*End of Lesson 28 - Congratulations on completing the Operating System Theory course!*
