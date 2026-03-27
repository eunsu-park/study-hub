# Virtual Memory ⭐⭐⭐

**Previous**: [Segmentation](./13_Segmentation.md) | **Next**: [Page Replacement](./15_Page_Replacement.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of virtual memory and its benefits for multiprogramming and memory isolation
2. Describe demand paging and how it reduces memory usage by loading pages only when needed
3. Trace the page fault handling sequence step-by-step, from trap to instruction restart
4. Calculate the effective memory access time given page fault rates and service times
5. Explain how the valid/invalid bit and dirty bit work in page table entries
6. Distinguish pure demand paging from prepaging and describe when each is appropriate
7. Implement Copy-on-Write semantics and explain how it accelerates fork()
8. Describe memory-mapped files (mmap) and their advantages over traditional file I/O

---

Virtual memory is perhaps the most transformative idea in OS design. It lets a process use more memory than physically exists, enables memory isolation between processes, and makes sharing efficient. Every modern OS relies on it -- and understanding it explains why your computer can run dozens of applications simultaneously without running out of RAM.

## Table of Contents

1. [Virtual Memory Concept](#1-virtual-memory-concept)
2. [Demand Paging](#2-demand-paging)
3. [Page Fault](#3-page-fault)
4. [Copy-on-Write](#4-copy-on-write)
5. [Memory Mapped Files](#5-memory-mapped-files)
6. [Performance Analysis](#6-performance-analysis)
7. [Practice Problems](#practice-problems)

---

## 1. Virtual Memory Concept

> Virtual memory is like a hotel's room reservation system. The hotel (physical memory) has 100 rooms, but it has accepted 200 reservations (processes). This works because not all guests are in their rooms at the same time -- some are at the pool, some are out sightseeing. The hotel only needs enough rooms for the guests actually present. Similarly, virtual memory lets the OS promise more memory than physically exists, loading pages from disk only when they are actually needed.

### 1.1 What is Virtual Memory?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Virtual Memory Concept                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Process perspective:                  Actual situation:                │
│                                                                          │
│   "I have exclusive                     Physical memory: 2GB            │
│    use of 4GB memory!"                  + Disk swap: 8GB                │
│                                                                          │
│   ┌──────────────────┐                ┌──────────────────┐              │
│   │ Virtual address  │                │    Physical RAM  │              │
│   │ space (4GB)      │                │     (2GB)        │              │
│   │                  │                │                  │              │
│   │ ┌──────────────┐ │    Only some  │ ┌──────────────┐ │              │
│   │ │ Page 0       │─┼───in memory──▶│ │ Frame 100    │ │              │
│   │ ├──────────────┤ │               │ ├──────────────┤ │              │
│   │ │ Page 1       │─┼─────────────▶│ │ Frame 205    │ │              │
│   │ ├──────────────┤ │               │ ├──────────────┤ │              │
│   │ │ Page 2       │ │               │ │ Other process│ │              │
│   │ ├──────────────┤ │               │ │      ...     │ │              │
│   │ │     ...      │ │               │ └──────────────┘ │              │
│   │ │ (most on     │ │                                   │              │
│   │ │  disk)       │ │               ┌──────────────────┐              │
│   │ ├──────────────┤ │               │    Disk          │              │
│   │ │ Page N       │ │    Rest on    │  (Swap area)     │              │
│   │ └──────────────┘ │    disk       │                  │              │
│   └──────────────────┘       ──────▶ │ Page 2, 3...     │              │
│                                       └──────────────────┘              │
│                                                                          │
│   Key: Frequently used parts in memory, rest on disk                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Advantages of Virtual Memory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Virtual Memory Advantages                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Run programs larger than physical memory                            │
│     - Run 4GB program on 2GB RAM                                        │
│     - Load only necessary parts into memory                             │
│                                                                          │
│  2. Run more processes simultaneously                                   │
│     - Each process doesn't use entire memory                            │
│     - Improved multiprogramming level                                   │
│                                                                          │
│  3. Easy memory sharing                                                 │
│     - Shared libraries (libc.so etc)                                    │
│     - Shared memory IPC                                                 │
│     - Copy-on-Write after fork()                                        │
│                                                                          │
│  4. Accelerate process creation                                         │
│     - fork(): Copy only page tables                                     │
│     - exec(): Load only needed pages                                    │
│                                                                          │
│  5. I/O optimization                                                    │
│     - Map files to memory (mmap)                                        │
│     - Reduce unnecessary copying                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Demand Paging

### 2.1 Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Demand Paging                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional approach:                 Demand paging:                  │
│   ┌──────────────────┐                 ┌──────────────────┐             │
│   │ At program start │                 │ At program start │             │
│   │ Load entire      │                 │ Load nothing     │             │
│   │ program to memory│                 │                  │             │
│   └──────────────────┘                 └──────────────────┘             │
│                                                 │                        │
│                                                 ▼                        │
│                                        ┌──────────────────┐             │
│                                        │ First instruction│             │
│                                        │ → Page fault     │             │
│                                        │ → Load that page │             │
│                                        │                  │             │
│                                        └──────────────────┘             │
│                                                 │                        │
│                                                 ▼                        │
│                                        ┌──────────────────┐             │
│                                        │ Continue running │             │
│                                        │ Load only needed │             │
│                                        │ pages            │             │
│                                        └──────────────────┘             │
│                                                                          │
│   Lazy Loading: "Load when needed"                                      │
│   Pure Demand Paging: Never preload                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Valid/Invalid Bit

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Page Table and Valid Bit                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Page Table                                                            │
│   ┌────────────┬────────────┬─────────┐                                 │
│   │ Page Number│Frame Number│  V bit  │                                 │
│   ├────────────┼────────────┼─────────┤                                 │
│   │     0      │     4      │    1    │ ← In memory                     │
│   │     1      │     -      │    0    │ ← On disk (page fault)          │
│   │     2      │     7      │    1    │ ← In memory                     │
│   │     3      │     -      │    0    │ ← Not allocated yet             │
│   │     4      │     2      │    1    │ ← In memory                     │
│   │     5      │     -      │    0    │ ← On disk                       │
│   │     6      │     -      │    i    │ ← Invalid (illegal access)      │
│   └────────────┴────────────┴─────────┘                                 │
│                                                                          │
│   V=1: Valid - Page is in memory                                        │
│   V=0: Invalid - Page is not in memory                                  │
│                                                                          │
│   Invalid cases:                                                        │
│   1. In disk swap area → Need to load                                   │
│   2. Never accessed yet → Need to allocate                              │
│   3. Invalid address → Segmentation Fault                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Page Fault

### 3.1 Page Fault Handling Process (10 Steps)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Page Fault Handling Process                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │  1. CPU references memory → Page table lookup                   │   │
│   │     ↓                                                            │   │
│   │  2. Valid bit = 0 detected → Page fault trap                    │   │
│   │     ↓                                                            │   │
│   │  3. OS gains control                                            │   │
│   │     - Save registers, process state                             │   │
│   │     ↓                                                            │   │
│   │  4. Address validity check                                      │   │
│   │     - Terminate process if illegal address                      │   │
│   │     - Continue if valid                                         │   │
│   │     ↓                                                            │   │
│   │  5. Find free frame                                             │   │
│   │     - If none, perform page replacement                         │   │
│   │     ↓                                                            │   │
│   │  6. Start reading page from disk                                │   │
│   │     - Disk I/O request                                          │   │
│   │     - Process goes to waiting state                             │   │
│   │     ↓                                                            │   │
│   │  7. CPU executes other process                                  │   │
│   │     - Context switch                                            │   │
│   │     ↓                                                            │   │
│   │  8. Disk I/O complete interrupt                                 │   │
│   │     - Page loaded to frame                                      │   │
│   │     ↓                                                            │   │
│   │  9. Update page table                                           │   │
│   │     - Record frame number                                       │   │
│   │     - Valid bit = 1                                             │   │
│   │     ↓                                                            │   │
│   │  10. Restart original instruction                               │   │
│   │      - Resume process                                           │   │
│   │      - No page fault this time                                  │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Hardware Support Requirements

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Hardware Requirements for Page Fault Handling           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Page Table                                                          │
│     - Valid/invalid bit support                                         │
│     - Reference bit, dirty bit                                          │
│                                                                          │
│  2. Secondary Memory (Swap space)                                       │
│     - Disk space for storing pages                                      │
│     - Faster access than regular file system                            │
│                                                                          │
│  3. Instruction Restart                                                 │
│     - Re-execute same instruction after page fault                      │
│     - Recover partially executed instruction                            │
│                                                                          │
│  Example: ADD instruction causes page fault                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  ADD R1, [A], [B]   ; Add values at A and B, store in R1         │   │
│  │                                                                   │   │
│  │  1. Page fault during [A] fetch → Restart from beginning         │   │
│  │  2. Page fault during [B] fetch → Must re-fetch A                │   │
│  │                                                                   │   │
│  │  Complex case:                                                    │   │
│  │  MVC (Move Character) - Block copy instruction                   │   │
│  │  - Page fault during copy → Need to undo partial copy            │   │
│  │  - More complex if source and destination overlap                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Page Fault Handler Code (Pseudocode)

```c
#include <stdint.h>
#include <stdbool.h>

#define NUM_FRAMES 1024
#define PAGE_SIZE 4096

typedef struct {
    uint32_t frame_number;
    bool valid;
    bool dirty;
    bool referenced;
} PageTableEntry;

typedef struct {
    bool allocated;
    int process_id;
    int page_number;
} Frame;

Frame frame_table[NUM_FRAMES];
PageTableEntry* current_page_table;
int current_process_id;

// Page fault handler
void page_fault_handler(uint32_t virtual_address) {
    uint32_t page_number = virtual_address / PAGE_SIZE;

    printf("[Page Fault] Process %d, Page %u\n",
           current_process_id, page_number);

    // 1. Address validity check
    if (!is_valid_address(current_process_id, virtual_address)) {
        // Illegal access - terminate process
        printf("Segmentation Fault!\n");
        terminate_process(current_process_id);
        return;
    }

    // 2. Find free frame
    int frame = find_free_frame();
    if (frame == -1) {
        // No free frame - need page replacement
        frame = select_victim_frame();  // Use LRU etc
        evict_page(frame);              // Remove victim page
    }

    // 3. Load page from disk
    uint32_t disk_address = get_disk_address(current_process_id, page_number);
    schedule_disk_read(disk_address, frame);

    // 4. Transition process to waiting state
    block_process(current_process_id);

    // 5. Execute other process (scheduler)
    schedule_next_process();

    // --- After disk I/O completes in interrupt handler ---
}

// Disk I/O complete interrupt handler
void disk_io_complete_handler(int frame, int process_id, int page_number) {
    // 6. Update page table
    PageTableEntry* pte = get_page_table_entry(process_id, page_number);
    pte->frame_number = frame;
    pte->valid = true;
    pte->dirty = false;
    pte->referenced = true;

    // 7. Update frame table
    frame_table[frame].allocated = true;
    frame_table[frame].process_id = process_id;
    frame_table[frame].page_number = page_number;

    // 8. Transition process to ready state
    unblock_process(process_id);

    // When process is rescheduled, instruction restarts
}
```

---

## 4. Copy-on-Write

### 4.1 Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Copy-on-Write (COW)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional approach on fork():                                       │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │                                                                │     │
│   │   Parent process             Child process                    │     │
│   │   ┌──────────────┐          ┌──────────────┐                  │     │
│   │   │ Memory space │   Copy   │ Memory space │                  │     │
│   │   │ (100MB)      │ ──────▶ │ (100MB copy) │                  │     │
│   │   └──────────────┘          └──────────────┘                  │     │
│   │                                                                │     │
│   │   Problem: 200MB memory usage                                 │     │
│   │   All copied memory wasted if child calls exec()!            │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│   Copy-on-Write approach:                                               │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │                                                                │     │
│   │   After fork():                                                │     │
│   │   Parent PT   Child PT              Physical Memory           │     │
│   │   ┌──────┐  ┌──────┐             ┌──────────────┐             │     │
│   │   │ Page 0│──┼──────┼───────────▶│ Shared page 0│ (read-only) │     │
│   │   │ Page 1│──┼──────┼───────────▶│ Shared page 1│ (read-only) │     │
│   │   │ Page 2│──┼──────┼───────────▶│ Shared page 2│ (read-only) │     │
│   │   └──────┘  └──────┘             └──────────────┘             │     │
│   │                                                                │     │
│   │   Memory usage: 100MB (shared!)                               │     │
│   │                                                                │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Copy on Write

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COW Page Copy on Write                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Child attempts to write to page 1:                                    │
│                                                                          │
│   1. Write protection violation (page is read-only)                     │
│   2. OS confirms it's a COW page                                        │
│   3. Allocate and copy new page                                         │
│   4. Update child's page table                                          │
│   5. Execute write                                                      │
│                                                                          │
│   Result:                                                               │
│   Parent PT   Child PT              Physical Memory                     │
│   ┌──────┐  ┌──────┐             ┌──────────────┐                      │
│   │ Page 0│──┼──────┼───────────▶│ Shared page 0│ (read-only)          │
│   │ Page 1│──┼──────┼──┐        ├──────────────┤                      │
│   │ Page 2│──┼──────┼──┼───────▶│ Shared page 2│ (read-only)          │
│   └──────┘  └───┬──┘  │        └──────────────┘                      │
│                  │     │                                                │
│                  │     └────────▶┌──────────────┐                      │
│                  │               │ Parent page 1│ (read/write)         │
│                  │               └──────────────┘                      │
│                  │                                                      │
│                  └──────────────▶┌──────────────┐                      │
│                                  │ Child page 1 │ (read/write)         │
│                                  │ (copy)       │                      │
│                                  └──────────────┘                      │
│                                                                          │
│   Now page 1 is independent, pages 0,2 still shared                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 COW Implementation

```c
// COW page fault handler (on write)
void cow_page_fault_handler(int process_id, uint32_t virtual_address) {
    uint32_t page_number = virtual_address / PAGE_SIZE;
    PageTableEntry* pte = get_page_table_entry(process_id, page_number);

    // Check if COW page
    if (!pte->cow_flag) {
        // Real protection violation - terminate process
        terminate_process(process_id);
        return;
    }

    int old_frame = pte->frame_number;
    int ref_count = get_reference_count(old_frame);

    if (ref_count > 1) {
        // Other process also using this page - need copy
        int new_frame = allocate_frame();
        if (new_frame == -1) {
            new_frame = select_victim_and_evict();
        }

        // Copy page contents
        memcpy(frame_to_address(new_frame),
               frame_to_address(old_frame),
               PAGE_SIZE);

        // Decrement reference count
        decrement_reference_count(old_frame);

        // Update page table
        pte->frame_number = new_frame;
        pte->cow_flag = false;
        pte->writable = true;

        // New frame reference count = 1
        set_reference_count(new_frame, 1);
    } else {
        // Only this process using - no copy needed, allow write
        pte->cow_flag = false;
        pte->writable = true;
    }

    // Invalidate TLB
    invalidate_tlb_entry(virtual_address);
}
```

---

## 5. Memory Mapped Files

### 5.1 mmap() Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Memory Mapped File (mmap)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional file I/O:                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   Process                 Kernel                    Disk        │   │
│   │   ┌──────────┐          ┌──────────┐           ┌──────────┐    │   │
│   │   │ Buffer   │◀── read()─│ Kernel   │◀──────────│   File   │    │   │
│   │   └──────────┘          │ buffer   │           └──────────┘    │   │
│   │                         └──────────┘                            │   │
│   │                                                                  │   │
│   │   1. System call overhead                                       │   │
│   │   2. Data copied twice (disk→kernel→user)                       │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Memory mapping:                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   Process virtual address space                                 │   │
│   │   ┌──────────────────────────────────────────────┐             │   │
│   │   │        ...                                    │             │   │
│   │   ├──────────────────────────────────────────────┤             │   │
│   │   │        Mapped region                          │             │   │
│   │   │   ┌────────────────────────────────────┐     │             │   │
│   │   │   │        File contents                │◀────┼─┐           │   │
│   │   │   │     (direct access)                 │     │ │           │   │
│   │   │   └────────────────────────────────────┘     │ │           │   │
│   │   ├──────────────────────────────────────────────┤ │           │   │
│   │   │        ...                                    │ │           │   │
│   │   └──────────────────────────────────────────────┘ │           │   │
│   │                                                     │ Auto load│   │
│   │   Disk                                              │ by page  │   │
│   │   ┌──────────────────────────────────────────────┐ │ fault    │   │
│   │   │              File                             │◀┘           │   │
│   │   └──────────────────────────────────────────────┘             │   │
│   │                                                                  │   │
│   │   Advantages:                                                   │   │
│   │   - No copying (Zero-Copy)                                      │   │
│   │   - Direct pointer access                                       │   │
│   │   - Multiple processes can share file                           │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 mmap() Usage Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

// Read file example
void read_file_with_mmap(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return;
    }

    // Get file size
    struct stat sb;
    fstat(fd, &sb);

    // Map file to memory
    char* addr = mmap(NULL,               // Kernel chooses address
                      sb.st_size,         // Mapping size
                      PROT_READ,          // Read-only
                      MAP_PRIVATE,        // Private mapping
                      fd,                 // File descriptor
                      0);                 // Offset

    if (addr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    // Now can directly access file contents via addr
    printf("File size: %ld bytes\n", sb.st_size);
    printf("First 100 bytes:\n%.100s\n", addr);

    // Unmap
    munmap(addr, sb.st_size);
    close(fd);
}

// File modification example (shared mapping)
void modify_file_with_mmap(const char* filename) {
    int fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("open");
        return;
    }

    struct stat sb;
    fstat(fd, &sb);

    // Shared mapping - changes reflected to file
    char* addr = mmap(NULL, sb.st_size,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED,          // Shared mapping!
                      fd, 0);

    if (addr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    // Modify file contents
    if (sb.st_size >= 5) {
        memcpy(addr, "Hello", 5);  // Change first 5 bytes
    }

    // Force sync changes
    msync(addr, sb.st_size, MS_SYNC);

    munmap(addr, sb.st_size);
    close(fd);

    printf("File modification complete\n");
}

// Anonymous mapping (for shared memory)
void* create_shared_memory(size_t size) {
    void* addr = mmap(NULL, size,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS,  // Map without file
                      -1, 0);

    if (addr == MAP_FAILED) {
        return NULL;
    }

    return addr;
}

int main() {
    read_file_with_mmap("/etc/passwd");
    return 0;
}
```

### 5.3 Mapping Options

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        mmap() Options                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Protection flags (prot):                                               │
│  ┌───────────────┬────────────────────────────────────┐                 │
│  │ PROT_READ     │ Read allowed                       │                 │
│  │ PROT_WRITE    │ Write allowed                      │                 │
│  │ PROT_EXEC     │ Execute allowed                    │                 │
│  │ PROT_NONE     │ No access (guard page)             │                 │
│  └───────────────┴────────────────────────────────────┘                 │
│                                                                          │
│  Flags:                                                                 │
│  ┌───────────────┬────────────────────────────────────┐                 │
│  │ MAP_SHARED    │ Changes reflected to file, shareable│                │
│  │ MAP_PRIVATE   │ COW - changes visible to self only │                 │
│  │ MAP_ANONYMOUS │ Memory only without file (fd=-1)   │                 │
│  │ MAP_FIXED     │ Map at exact specified address     │                 │
│  │ MAP_LOCKED    │ Lock pages in memory (no swap)     │                 │
│  └───────────────┴────────────────────────────────────┘                 │
│                                                                          │
│  Usage examples:                                                        │
│  - Load executable: MAP_PRIVATE, PROT_READ|PROT_EXEC                    │
│  - Data file: MAP_SHARED, PROT_READ|PROT_WRITE                          │
│  - Heap expansion: MAP_PRIVATE|MAP_ANONYMOUS                            │
│  - Shared memory: MAP_SHARED|MAP_ANONYMOUS                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Performance Analysis

### 6.1 Effective Access Time (With Page Fault)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Effective Access Time With Page Fault                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Parameters:                                                           │
│   - p: Page fault probability (0 ≤ p ≤ 1)                              │
│   - ma: Memory access time (e.g., 100ns)                               │
│   - pft: Page fault service time (e.g., 8ms = 8,000,000ns)             │
│                                                                          │
│   Effective Access Time (EAT):                                          │
│   EAT = (1 - p) × ma + p × pft                                          │
│                                                                          │
│   Example:                                                              │
│   ma = 100ns, pft = 8ms, allow 10% performance degradation             │
│                                                                          │
│   Acceptable EAT = 100ns × 1.1 = 110ns                                  │
│                                                                          │
│   110 = (1-p) × 100 + p × 8,000,000                                     │
│   110 = 100 - 100p + 8,000,000p                                         │
│   10 = 7,999,900p                                                       │
│   p = 10 / 7,999,900                                                    │
│   p ≈ 0.00000125                                                        │
│   p ≈ 1/800,000                                                         │
│                                                                          │
│   Conclusion: Allow only 1 page fault per 800,000 accesses!            │
│                                                                          │
│   This is why page replacement algorithms are important                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Page Fault Time Breakdown

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Page Fault Service Time Breakdown                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                                                              │       │
│   │   1. Interrupt handling         1-100 μs                    │       │
│   │      - Trap occurs                                          │       │
│   │      - Save registers/state                                 │       │
│   │                                                              │       │
│   │   2. Page fault processing      1-100 μs                    │       │
│   │      - Address validity check                               │       │
│   │      - Find free frame                                      │       │
│   │                                                              │       │
│   │   3. Disk I/O                   8-10 ms  ◀── Most time!     │       │
│   │      - Seek time                                            │       │
│   │      - Rotational latency                                   │       │
│   │      - Transfer time                                        │       │
│   │                                                              │       │
│   │   4. Interrupt handling         1-100 μs                    │       │
│   │      - I/O complete interrupt                               │       │
│   │      - Update tables                                        │       │
│   │                                                              │       │
│   │   5. Process restart            1-100 μs                    │       │
│   │      - Restore registers                                    │       │
│   │      - Restart instruction                                  │       │
│   │                                                              │       │
│   └─────────────────────────────────────────────────────────────┘       │
│                                                                          │
│   Total time: About 8-10ms (mostly disk I/O)                            │
│                                                                          │
│   With SSD: Reduced significantly to 0.1-0.5ms                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Problem 1: Page Fault Scenario
A process accesses pages in the following order. With 3 frames in memory and all initially empty, calculate the number of page faults.

Access sequence: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5

<details>
<summary>Show Answer</summary>

```
Assuming FIFO replacement:

Access  Memory(3 frames)  Fault?
 1      [1, -, -]         Fault
 2      [1, 2, -]         Fault
 3      [1, 2, 3]         Fault
 4      [4, 2, 3]         Fault (replace 1)
 1      [4, 1, 3]         Fault (replace 2)
 2      [4, 1, 2]         Fault (replace 3)
 5      [5, 1, 2]         Fault (replace 4)
 1      [5, 1, 2]         Hit
 2      [5, 1, 2]         Hit
 3      [3, 1, 2]         Fault (replace 5)
 4      [3, 4, 2]         Fault (replace 1)
 5      [3, 4, 5]         Fault (replace 2)

Total page faults: 10
```

</details>

### Problem 2: EAT Calculation
If memory access time is 50ns and page fault service time is 10ms, what should the page fault probability be to keep performance degradation within 5%?

<details>
<summary>Show Answer</summary>

```
ma = 50ns
pft = 10ms = 10,000,000ns
Acceptable EAT = 50 × 1.05 = 52.5ns

EAT = (1-p) × ma + p × pft
52.5 = (1-p) × 50 + p × 10,000,000
52.5 = 50 - 50p + 10,000,000p
2.5 = 9,999,950p
p = 2.5 / 9,999,950
p ≈ 2.5 × 10^-7
p ≈ 1 / 4,000,000

Conclusion: At most 1 page fault per 4 million accesses
```

</details>

### Problem 3: Copy-on-Write
A parent process has 100 pages and calls fork(). After the child modifies 10 pages and terminates, how many pages physically exist?

<details>
<summary>Show Answer</summary>

```
1. After fork():
   - 100 pages shared between parent and child (read-only)
   - Physical pages: 100

2. After child modifies 10 pages:
   - COW occurs on each modification, allocate new page
   - Physical pages: 100(original) + 10(copies) = 110

3. After child terminates:
   - Child's 10 copies freed
   - Shared 90 pages' reference count decreases (to 1)
   - Physical pages: 100 (only parent remains)

Physical pages at each point:
- After fork(): 100
- After child modification: 110
- After child termination: 100
```

</details>

### Problem 4: mmap() Analysis
Predict the output of the following code.

```c
int fd = open("test.txt", O_RDWR);  // Contents: "AAAAAAAAAA"
char* p1 = mmap(NULL, 10, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
char* p2 = mmap(NULL, 10, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);

p1[0] = 'B';
p2[1] = 'C';

printf("p1: %.10s\n", p1);
printf("p2: %.10s\n", p2);
// What if we re-read the file?
```

<details>
<summary>Show Answer</summary>

```
p1: MAP_SHARED - changes reflected to file
p2: MAP_PRIVATE - changes visible only within process (COW)

p1[0] = 'B':
- First character of file changed to B
- p1 shares with file, so change is reflected

p2[1] = 'C':
- COW occurs - p2's page copied
- Second character of copy changed to C
- File not changed

Output:
p1: BAAAAAAAAA  (p1[0] is B, file also changed)
p2: BCAAAAAAAA  (p1's change + p2's own change)

Re-reading file: BAAAAAAAAA
(p2's C is visible only within process, not reflected to file)
```

</details>

### Problem 5: Demand Paging Design
When designing a new OS, you're deciding whether to use Pure Demand Paging or Prefetching. Explain the pros and cons of each approach.

<details>
<summary>Show Answer</summary>

```
Pure Demand Paging:
Pros:
- No unnecessary page loading (only exactly what's needed)
- Minimize initial memory usage
- Simple implementation

Cons:
- Many page faults at program start
- Performance degradation with non-local access patterns

Prefetching:
Pros:
- Preload expected pages to reduce faults
- Effective for sequential access patterns
- Utilize locality principle

Cons:
- Memory waste on prediction failures
- Complex implementation (need good prediction algorithm)
- Possible unnecessary I/O

Real systems:
- Linux: Uses both
  - Default: Demand paging
  - On sequential access detection: readahead (prefetch)
  - madvise(MADV_SEQUENTIAL): Prefetch hint

Recommendation:
- Default to demand paging
- Adaptive prefetching when sequential access pattern detected
```

</details>

---

## Next Steps

Learn about page replacement algorithms in [15_Page_Replacement.md](./15_Page_Replacement.md)!

---

## References

- Silberschatz, "Operating System Concepts" Chapter 10
- Linux man pages: `mmap(2)`, `fork(2)`
- Tanenbaum, "Modern Operating Systems" Chapter 3
- Linux kernel source: `mm/memory.c`, `mm/mmap.c`
