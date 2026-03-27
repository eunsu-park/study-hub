# File System Implementation ⭐⭐⭐⭐

**Previous**: [File System Basics](./16_File_System_Basics.md) | **Next**: [I/O and IPC](./18_IO_and_IPC.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare contiguous, linked, and indexed file allocation methods with their trade-offs
2. Explain the FAT (File Allocation Table) structure and its limitations for large disks
3. Describe the inode-based allocation used in Unix/Linux and calculate maximum file sizes
4. Explain journaling and how it prevents file system corruption after crashes
5. Distinguish between write-ahead logging modes (journal, ordered, writeback) in ext4
6. Compare RAID levels (0, 1, 5, 6, 10) by capacity, performance, and fault tolerance
7. Trace the complete process of file deletion from directory entry removal to block deallocation

---

When you save a file, the OS must decide where on disk to place it, how to track its blocks, and how to recover if the power fails mid-write. File system implementation answers these questions -- and the choices directly affect speed, reliability, and maximum file size. From the simple FAT used on USB drives to the journaling ext4 powering Linux servers, the techniques in this lesson are at work on every storage device you use.

## Table of Contents

1. [File System Structure](#1-file-system-structure)
2. [Disk Block Allocation](#2-disk-block-allocation)
3. [inode Structure](#3-inode-structure)
4. [Directory Implementation](#4-directory-implementation)
5. [File System Examples](#5-file-system-examples)
6. [Journaling](#6-journaling)
7. [RAID](#7-raid)
8. [Practice Problems](#practice-problems)

---

## 1. File System Structure

### 1.1 Disk Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Typical File System Layout                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                         Entire Disk                              │   │
│   ├──────┬──────┬──────┬──────┬──────┬───────────────────────────────┤   │
│   │ MBR  │Partition│Partition│Partition│      │                      │   │
│   │      │  1   │  2   │  3   │ ...  │       Free space           │   │
│   └──────┴──────┴──────┴──────┴──────┴───────────────────────────────┘   │
│                  │                                                       │
│                  ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Partition Structure (ext4)                  │   │
│   ├──────┬──────────┬──────────┬────────────────────────────────────┤   │
│   │ Boot │  Super   │  Group   │            Block Groups            │   │
│   │Block │  Block   │Descriptor│   0    │   1    │   2   │  ...   │   │
│   └──────┴──────────┴──────────┴────────────────────────────────────┘   │
│                                           │                              │
│                                           ▼                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        Block Group Structure                     │   │
│   ├──────────┬──────────┬──────────┬──────────────────────────────────┤  │
│   │  Block   │  inode   │  inode   │          Data Blocks            │  │
│   │  Bitmap  │  Bitmap  │  Table   │                                  │  │
│   └──────────┴──────────┴──────────┴──────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Major Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      File System Components                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┬────────────────────────────────────────────────┐    │
│  │   Component    │                    Role                        │    │
│  ├────────────────┼────────────────────────────────────────────────┤    │
│  │ Boot Block     │ Bootstrap code (OS loading)                    │    │
│  │                │ Block 0 of partition                           │    │
│  ├────────────────┼────────────────────────────────────────────────┤    │
│  │ Superblock     │ File system metadata                           │    │
│  │                │ - Block size, total blocks/inodes              │    │
│  │                │ - Free blocks/inodes count                     │    │
│  │                │ - Mount count, last check time                 │    │
│  ├────────────────┼────────────────────────────────────────────────┤    │
│  │ Block Bitmap   │ Usage status of each block (0/1)               │    │
│  │                │ Fast free block search                         │    │
│  ├────────────────┼────────────────────────────────────────────────┤    │
│  │ inode Bitmap   │ Usage status of each inode (0/1)               │    │
│  ├────────────────┼────────────────────────────────────────────────┤    │
│  │ inode Table    │ Stores all inodes                              │    │
│  │                │ Core of file metadata                          │    │
│  ├────────────────┼────────────────────────────────────────────────┤    │
│  │ Data Blocks    │ Actual file/directory content                  │    │
│  └────────────────┴────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Disk Block Allocation

### 2.1 Contiguous Allocation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Contiguous Allocation                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Directory Entry:                                                       │
│   ┌───────────┬─────────┬────────┐                                      │
│   │ Filename  │Start Block│ Length│                                      │
│   ├───────────┼─────────┼────────┤                                      │
│   │ file_a    │    0    │   3    │                                      │
│   │ file_b    │    6    │   2    │                                      │
│   │ file_c    │   10    │   4    │                                      │
│   └───────────┴─────────┴────────┘                                      │
│                                                                          │
│   Disk Blocks:                                                           │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐        │
│   │ A │ A │ A │   │   │   │ B │ B │   │   │ C │ C │ C │ C │   │        │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘        │
│    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14            │
│                                                                          │
│   Advantages:                                                            │
│   - Fast sequential access (minimal disk head movement)                  │
│   - Easy direct access: block n = start + n                             │
│                                                                          │
│   Disadvantages:                                                         │
│   - External fragmentation (blocks 3-5, 8-9 wasted)                     │
│   - Difficult to grow files                                             │
│   - Must know size in advance                                           │
│                                                                          │
│   Used in: CD-ROM, DVD (read-only)                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Linked Allocation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Linked Allocation                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Directory Entry:                                                       │
│   ┌───────────┬─────────┬─────────┐                                     │
│   │ Filename  │Start Block│End Block│                                     │
│   ├───────────┼─────────┼─────────┤                                     │
│   │ file_a    │    0    │   12    │                                     │
│   └───────────┴─────────┴─────────┘                                     │
│                                                                          │
│   Disk Blocks (each block includes next block pointer):                 │
│                                                                          │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐        │
│   │ Block 0   │   │ Block 5   │   │ Block 8   │   │ Block 12  │        │
│   │ Data      │──▶│ Data      │──▶│ Data      │──▶│ Data      │        │
│   │ Next: 5   │   │ Next: 8   │   │ Next: 12  │   │ Next: -1  │        │
│   └───────────┘   └───────────┘   └───────────┘   └───────────┘        │
│        0               5               8              12                │
│                                                                          │
│   Advantages:                                                            │
│   - No external fragmentation                                            │
│   - Files can grow dynamically                                           │
│                                                                          │
│   Disadvantages:                                                         │
│   - Inefficient direct access (must follow chain n times for block n)   │
│   - Pointer space overhead (4 bytes per block)                          │
│   - Reliability issues (file lost if pointer corrupted)                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 FAT (File Allocation Table)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             FAT Structure                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   FAT (cached in memory)          Disk Blocks                           │
│   ┌─────────────────┐            ┌───────────────────────────────────┐  │
│   │Index│Next Block │            │                                   │  │
│   ├─────┼──────────┤            │  0  │  1  │  2  │  3  │  4  │ ... │  │
│   │  0  │    5     │────────────┼─ A ─┼─────┼─ B ─┼─────┼─────┼─────┤  │
│   │  1  │   FREE   │            │     │     │     │     │     │     │  │
│   │  2  │    4     │────────────┼─────┼─────┼─────┼─────┼─ B ─┼─────┤  │
│   │  3  │   FREE   │            │                                   │  │
│   │  4  │   EOF    │────────────┼─────┼─ A ─┼─────┼─────┼─────┼─────┤  │
│   │  5  │    7     │            │                                   │  │
│   │  6  │   FREE   │            │                                   │  │
│   │  7  │   EOF    │            └───────────────────────────────────┘  │
│   └─────┴──────────┘                                                    │
│                                                                          │
│   file_a: 0 → 5 → 7 → EOF                                              │
│   file_b: 2 → 4 → EOF                                                   │
│                                                                          │
│   Advantages:                                                            │
│   - Fast direct access if FAT in memory                                 │
│   - Pointers outside data blocks improve reliability                    │
│                                                                          │
│   Disadvantages:                                                         │
│   - FAT can be large (for large disks)                                 │
│   - Entire file system problems if FAT corrupted                        │
│                                                                          │
│   Used in: MS-DOS, Windows (FAT32), USB drives                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Indexed Allocation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Indexed Allocation                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Directory Entry:                                                       │
│   ┌───────────┬───────────┐                                             │
│   │ Filename  │Index Block│                                             │
│   ├───────────┼───────────┤                                             │
│   │ file_a    │     8     │                                             │
│   └───────────┴───────────┘                                             │
│                    │                                                     │
│                    ▼                                                     │
│   Index Block (block 8):          Data Blocks:                          │
│   ┌───────────────────┐          ┌───────────────────────────────────┐  │
│   │ [0] → Block 3     │          │                                   │  │
│   │ [1] → Block 10    │          │  3  │ 10  │ 15  │ 21  │  ...     │  │
│   │ [2] → Block 15    │          │Data │Data │Data │Data │          │  │
│   │ [3] → Block 21    │          │     │     │     │     │          │  │
│   │ [4] → -1 (end)    │          └───────────────────────────────────┘  │
│   │ ...               │                                                 │
│   └───────────────────┘                                                 │
│                                                                          │
│   Advantages:                                                            │
│   - Direct access possible: block n = index[n]                          │
│   - No external fragmentation                                            │
│                                                                          │
│   Disadvantages:                                                         │
│   - Index block overhead                                                 │
│   - Inefficient for small files                                         │
│                                                                          │
│   Extensions:                                                            │
│   - Linked index: link index blocks                                     │
│   - Multi-level index: tree structure (Unix inode)                      │
│   - Combined approach: direct + indirect pointers (Unix inode)          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. inode Structure

### 3.1 Unix/Linux inode

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        inode Structure (Unix/ext4)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                         inode                                    │   │
│   ├─────────────────────────────────────────────────────────────────┤   │
│   │  mode (permissions, type)                                       │   │
│   │  uid (owner)                                                    │   │
│   │  gid (group)                                                    │   │
│   │  size (file size)                                               │   │
│   │  atime (access time)                                            │   │
│   │  mtime (modification time)                                      │   │
│   │  ctime (status change time)                                     │   │
│   │  link count (hard link count)                                   │   │
│   ├─────────────────────────────────────────────────────────────────┤   │
│   │  Block pointers (data location)                                 │   │
│   │                                                                  │   │
│   │  ┌─────────────────────────────────────────────────────────┐    │   │
│   │  │ Direct pointers [0-11]  → Data blocks (12 blocks)      │    │   │
│   │  ├─────────────────────────────────────────────────────────┤    │   │
│   │  │ Single indirect [12]    → Index block → Data           │    │   │
│   │  ├─────────────────────────────────────────────────────────┤    │   │
│   │  │ Double indirect [13]    → Index → Index → Data         │    │   │
│   │  ├─────────────────────────────────────────────────────────┤    │   │
│   │  │ Triple indirect [14]    → Index → Index → Index → Data │    │   │
│   │  └─────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Direct/Indirect Block Pointers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  inode Block Pointer Structure                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Block size = 4KB, Pointer size = 4 bytes                              │
│   Pointers/block = 4KB / 4B = 1024                                      │
│                                                                          │
│   inode                                                                  │
│   ┌───────────────┐                                                     │
│   │ Direct [0]    │────────────────────────────────────▶ [Data block]  │
│   │ Direct [1]    │────────────────────────────────────▶ [Data block]  │
│   │ ...           │                                                      │
│   │ Direct [11]   │────────────────────────────────────▶ [Data block]  │
│   ├───────────────┤                                                     │
│   │ Single indirect[12]│──▶┌──────────┐                               │
│   │               │    │Index block│──▶ [Data] × 1024                  │
│   │               │    └──────────┘                                     │
│   ├───────────────┤                                                     │
│   │ Double indirect[13]│──▶┌──────────┐    ┌──────────┐               │
│   │               │    │Index block│──▶│Index block│──▶ [Data] × 1024 │
│   │               │    │ × 1024    │    └──────────┘    × 1024         │
│   │               │    └──────────┘                                     │
│   ├───────────────┤                                                     │
│   │ Triple indirect[14]│──▶ (3-level index)                           │
│   └───────────────┘                                                     │
│                                                                          │
│   Maximum file size calculation (4KB blocks):                           │
│   Direct: 12 × 4KB = 48KB                                               │
│   Single indirect: 1024 × 4KB = 4MB                                     │
│   Double indirect: 1024 × 1024 × 4KB = 4GB                              │
│   Triple indirect: 1024 × 1024 × 1024 × 4KB = 4TB                       │
│   Total: ~4TB                                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 inode Size Calculation Example

```c
// Find nth block location of file
#define BLOCK_SIZE 4096
#define PTRS_PER_BLOCK (BLOCK_SIZE / sizeof(uint32_t))  // 1024

#define DIRECT_BLOCKS 12
#define SINGLE_INDIRECT_LIMIT (DIRECT_BLOCKS + PTRS_PER_BLOCK)
#define DOUBLE_INDIRECT_LIMIT (SINGLE_INDIRECT_LIMIT + PTRS_PER_BLOCK * PTRS_PER_BLOCK)

uint32_t get_block_number(inode_t* inode, uint32_t logical_block) {
    if (logical_block < DIRECT_BLOCKS) {
        // Direct block
        return inode->direct[logical_block];
    }

    logical_block -= DIRECT_BLOCKS;

    if (logical_block < PTRS_PER_BLOCK) {
        // Single indirect
        uint32_t* indirect = read_block(inode->single_indirect);
        return indirect[logical_block];
    }

    logical_block -= PTRS_PER_BLOCK;

    if (logical_block < PTRS_PER_BLOCK * PTRS_PER_BLOCK) {
        // Double indirect
        uint32_t index1 = logical_block / PTRS_PER_BLOCK;
        uint32_t index2 = logical_block % PTRS_PER_BLOCK;

        uint32_t* level1 = read_block(inode->double_indirect);
        uint32_t* level2 = read_block(level1[index1]);
        return level2[index2];
    }

    // Triple indirect (omitted)
    return 0;
}
```

---

## 4. Directory Implementation

### 4.1 Directory Entry Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Directory Implementation                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Linear List (simple implementation):                                  │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │ Directory file content                                            │  │
│   ├────────────┬────────────────┬──────────────────────────────────────┤ │
│   │ inode num  │ Entry length   │ Filename                             │ │
│   ├────────────┼────────────────┼──────────────────────────────────────┤ │
│   │    2       │     12         │ .                                   │ │
│   │    2       │     12         │ ..                                  │ │
│   │   15       │     16         │ file1.txt                           │ │
│   │   23       │     20         │ documents                           │ │
│   │   45       │     24         │ long_filename.doc                   │ │
│   └────────────┴────────────────┴──────────────────────────────────────┘ │
│                                                                          │
│   ext4 directory entry (struct ext4_dir_entry_2):                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ inode (4 bytes)           : inode number                        │   │
│   │ rec_len (2 bytes)         : total length of this entry          │   │
│   │ name_len (1 byte)         : filename length                     │   │
│   │ file_type (1 byte)        : file type (file/dir/link/etc)       │   │
│   │ name (variable)           : filename                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Hash Table Directory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Hash Table Directory                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Filename → Hash function → Bucket index                               │
│                                                                          │
│   Hash Table                                                             │
│   ┌─────────┐                                                           │
│   │    0    │ → [readme.txt, inode 15]                                 │
│   ├─────────┤                                                           │
│   │    1    │ → [config.json, inode 23] → [data.csv, inode 45]         │
│   ├─────────┤                                                           │
│   │    2    │ → (empty)                                                 │
│   ├─────────┤                                                           │
│   │    3    │ → [main.c, inode 67] → [test.c, inode 89]                │
│   ├─────────┤                                                           │
│   │   ...   │                                                           │
│   └─────────┘                                                           │
│                                                                          │
│   Advantages:                                                            │
│   - O(1) average search (linear list is O(n))                           │
│   - Effective for large directories                                     │
│                                                                          │
│   ext4's htree (HTree Index):                                           │
│   - B-tree based hash directory                                         │
│   - Fast search even with thousands of files                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. File System Examples

### 5.1 FAT32

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FAT32 Structure                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Reserved │   FAT   │  FAT    │       Data Region              │   │
│   │  Region  │   1     │   2     │  (Clusters)                     │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Features:                                                              │
│   - Cluster size: 4KB ~ 32KB                                            │
│   - FAT entry: 32 bits (28 bits used)                                   │
│   - Max file size: 4GB - 1 (32-bit size field)                         │
│   - Max volume size: 2TB                                                │
│                                                                          │
│   Directory entry (32 bytes):                                           │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │ Filename (8 bytes) + Extension (3 bytes) = 8.3 format        │     │
│   │ Attributes (1 byte): read-only, hidden, system, directory etc│     │
│   │ Creation/modification/access times (10 bytes)                 │     │
│   │ Starting cluster number (4 bytes)                             │     │
│   │ File size (4 bytes)                                           │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│   VFAT: Long filename support (uses separate entries)                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 ext4

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ext4 Key Features                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Extents                                                             │
│      Represent consecutive blocks as single entry → Efficient for large files│
│      ┌──────────────────────────────────────────────┐                   │
│      │ Starting block: 1000, Length: 500            │                   │
│      │ → References blocks 1000~1499 at once        │                   │
│      └──────────────────────────────────────────────┘                   │
│                                                                          │
│   2. Journaling                                                          │
│      - Metadata journaling: Default mode, fast                          │
│      - Data journaling: Journals data too, safe                         │
│                                                                          │
│   3. Large Volume Support                                                │
│      - Max file size: 16TB                                              │
│      - Max volume size: 1EB (Exabyte)                                   │
│                                                                          │
│   4. Other Features                                                      │
│      - Delayed allocation: Write optimization                            │
│      - Multi-block allocation: Allocate multiple blocks at once         │
│      - Directory htree: Fast search                                      │
│      - Online defragmentation                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Journaling

### 6.1 Need for Journaling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Journaling Concept                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Problem: File system inconsistency                                     │
│                                                                          │
│   File deletion process (without journaling):                            │
│   1. Remove entry from directory                                        │
│   2. Add inode to free list                    ← Crash here!            │
│   3. Add data blocks to free list                                       │
│                                                                          │
│   Problems after crash:                                                  │
│   - Deleted from directory                                              │
│   - But inode and blocks still marked 'in use'                          │
│   - Space inaccessible + unrecoverable                                  │
│                                                                          │
│   Solution: Journaling                                                   │
│   Record to log (journal) before change → Recover based on log after crash│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Journaling Operation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Journaling Operation                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                       Journal Area                               │   │
│   ├─────────────────────────────────────────────────────────────────┤   │
│   │ TxB │ Data blocks │ Metadata │ TxE │ TxB │ ... │ TxE │ ...     │   │
│   └──┬──┴───────────┴────────────┴──┬──┘                            │   │
│      │                              │                                │   │
│      │      Transaction 1           │      Transaction 2             │   │
│      │                              │                                │   │
│   TxB: Transaction Begin                                              │   │
│   TxE: Transaction End (Commit)                                       │   │
│                                                                          │
│   Write process:                                                         │
│   1. Record Transaction Begin                                            │
│   2. Record blocks to be changed in journal                             │
│   3. Record Transaction End (commit)                                     │
│   4. Write data to actual location (Checkpoint)                         │
│   5. Remove transaction from journal                                     │
│                                                                          │
│   Recovery process (at boot):                                            │
│   1. Examine journal                                                     │
│   2. Completed transactions (has TxE): rewrite to actual location       │
│   3. Incomplete transactions (no TxE): ignore (rollback)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Journal Modes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ext4 Journal Modes                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────┬──────────────────────────────────────────────────┐   │
│  │    Mode       │                     Description                  │   │
│  ├───────────────┼──────────────────────────────────────────────────┤   │
│  │ journal       │ Journal both data + metadata                     │   │
│  │               │ Safest but slowest (2x writes)                   │   │
│  ├───────────────┼──────────────────────────────────────────────────┤   │
│  │ ordered       │ Journal metadata only                            │   │
│  │ (default)     │ Write data first, then journal metadata          │   │
│  │               │ Guarantees data consistency, reasonable performance│  │
│  ├───────────────┼──────────────────────────────────────────────────┤   │
│  │ writeback     │ Journal metadata only                            │   │
│  │               │ No guarantee on data write order                 │   │
│  │               │ Fastest but can lose data on crash               │   │
│  └───────────────┴──────────────────────────────────────────────────┘   │
│                                                                          │
│   # Specify option at mount                                             │
│   $ mount -o data=ordered /dev/sda1 /mnt                                │
│   $ mount -o data=journal /dev/sda1 /mnt                                │
│   $ mount -o data=writeback /dev/sda1 /mnt                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. RAID

### 7.1 RAID Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAID Overview                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   RAID = Redundant Array of Independent Disks                           │
│                                                                          │
│   Goals:                                                                 │
│   1. Performance improvement (Striping - data distribution)              │
│   2. Reliability improvement (Mirroring/Parity - redundant storage)     │
│   3. Large capacity (combine multiple disks)                             │
│                                                                          │
│   Implementation methods:                                                │
│   - Hardware RAID: RAID controller card                                  │
│   - Software RAID: Implemented in OS (mdadm)                            │
│   - Hybrid: Motherboard built-in RAID                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 RAID Levels

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       RAID 0 (Striping)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────────────────────────────────────────┐                 │
│   │               Data: A B C D E F G H               │                 │
│   └───────────────────────────────────────────────────┘                 │
│                           │                                              │
│           ┌───────────────┼───────────────┐                             │
│           ▼               ▼               ▼                             │
│      ┌─────────┐     ┌─────────┐     ┌─────────┐                       │
│      │ Disk 0  │     │ Disk 1  │     │ Disk 2  │                       │
│      │  A D G  │     │  B E H  │     │  C F    │                       │
│      └─────────┘     └─────────┘     └─────────┘                       │
│                                                                          │
│   Features:                                                              │
│   - Data distributed across all disks (striping)                         │
│   - Read/write speed: n× improvement                                    │
│   - Capacity: disk count × single capacity                               │
│   - No redundancy: one disk failure → total data loss                   │
│   - Use case: Performance critical, data loss acceptable                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       RAID 1 (Mirroring)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────────────────────────────────────────┐                 │
│   │               Data: A B C D                        │                 │
│   └───────────────────────────────────────────────────┘                 │
│                           │                                              │
│           ┌───────────────┴───────────────┐                             │
│           ▼                               ▼                             │
│      ┌─────────┐                     ┌─────────┐                       │
│      │ Disk 0  │                     │ Disk 1  │                       │
│      │ A B C D │ ◀── Identical ──▶  │ A B C D │                       │
│      │(Original)│                    │ (Mirror)│                       │
│      └─────────┘                     └─────────┘                       │
│                                                                          │
│   Features:                                                              │
│   - Perfect copy (mirroring)                                            │
│   - Read speed: 2× (parallel read from both disks)                      │
│   - Write speed: Same (must write to both)                              │
│   - Capacity: Single disk capacity (50% efficiency)                     │
│   - One disk failure preserves data                                     │
│   - Use case: Critical data, system drives                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                  RAID 5 (Striping with Parity)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┬─────────────┬─────────────┬─────────────┐             │
│   │  Disk 0     │  Disk 1     │  Disk 2     │  Disk 3     │             │
│   ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│   │     A1      │     A2      │     A3      │   Ap (parity)│            │
│   ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│   │     B1      │     B2      │  Bp (parity)│     B3      │             │
│   ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│   │     C1      │  Cp (parity)│     C2      │     C3      │             │
│   ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│   │  Dp (parity)│     D1      │     D2      │     D3      │             │
│   └─────────────┴─────────────┴─────────────┴─────────────┘             │
│                                                                          │
│   Parity calculation: Ap = A1 XOR A2 XOR A3                             │
│   Recovery: A1 = Ap XOR A2 XOR A3 (if A1 lost)                          │
│                                                                          │
│   Features:                                                              │
│   - Parity distributed (RAID 4 uses dedicated disk)                     │
│   - Capacity: (n-1) × single capacity                                   │
│   - Tolerates 1 disk failure                                            │
│   - Improved read speed, write slightly slower (parity calculation)     │
│   - Use case: Most servers                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                  RAID 6 (Dual Parity)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┬─────────────┬─────────────┬─────────────┐             │
│   │  Disk 0     │  Disk 1     │  Disk 2     │  Disk 3     │             │
│   ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│   │     A1      │     A2      │   Ap (P)    │   Aq (Q)    │             │
│   ├─────────────┼─────────────┼─────────────┼─────────────┤             │
│   │     B1      │   Bp (P)    │   Bq (Q)    │     B2      │             │
│   └─────────────┴─────────────┴─────────────┴─────────────┘             │
│                                                                          │
│   Features:                                                              │
│   - 2 parities (P, Q) - different algorithms                            │
│   - Tolerates 2 simultaneous disk failures                              │
│   - Capacity: (n-2) × single capacity                                   │
│   - Requires minimum 4 disks                                            │
│   - Safer than RAID 5, lower write performance                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     RAID 10 (1+0, Mirror + Stripe)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                          Data: A B C D                                  │
│                               │                                         │
│               ┌───────────────┴───────────────┐                        │
│               ▼                               ▼                        │
│        ┌─────────────┐                 ┌─────────────┐                │
│        │  Stripe 0   │                 │  Stripe 1   │                │
│        │     A C     │                 │     B D     │                │
│        └──────┬──────┘                 └──────┬──────┘                │
│               │                               │                        │
│        ┌──────┴──────┐                 ┌──────┴──────┐                │
│        ▼             ▼                 ▼             ▼                │
│   ┌─────────┐   ┌─────────┐       ┌─────────┐   ┌─────────┐          │
│   │ Disk 0  │   │ Disk 1  │       │ Disk 2  │   │ Disk 3  │          │
│   │   A C   │   │   A C   │       │   B D   │   │   B D   │          │
│   │(Original)│   │ (Mirror)│       │(Original)│   │ (Mirror)│          │
│   └─────────┘   └─────────┘       └─────────┘   └─────────┘          │
│                                                                          │
│   Features:                                                              │
│   - Combines RAID 1 (mirror) + RAID 0 (stripe)                          │
│   - Performance: Close to RAID 0                                        │
│   - Reliability: Close to RAID 1                                        │
│   - Capacity: 50% efficiency                                            │
│   - Tolerates 1 disk per mirror pair (up to 2 disks)                   │
│   - Use case: High performance + high reliability needed                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 RAID Level Comparison

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           RAID Level Comparison                                 │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────────────┤
│  RAID    │ Min      │ Capacity │ Read     │ Write    │ Failure Tolerance       │
│  Level   │ Disks    │ Efficiency│ Speed    │ Speed    │                         │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────────────┤
│  RAID 0  │    2     │  100%    │  n×      │  n×      │ 0 disks (dangerous)     │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────────────┤
│  RAID 1  │    2     │  50%     │  2×      │  1×      │ 1 disk                  │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────────────┤
│  RAID 5  │    3     │  (n-1)/n │  Fast    │  Medium  │ 1 disk                  │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────────────┤
│  RAID 6  │    4     │  (n-2)/n │  Fast    │  Slow    │ 2 disks                 │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────────────┤
│  RAID 10 │    4     │  50%     │  Fast    │  Fast    │ 1 per pair              │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────────────────┘
```

---

## Practice Problems

### Problem 1: inode Block Pointers
In a Unix file system with 4KB blocks and 4-byte pointers, how many indirect blocks are needed to store a 100MB file?

<details>
<summary>Show Answer</summary>

```
Block size: 4KB = 4096 bytes
Pointers/block: 4096 / 4 = 1024
File size: 100MB = 102,400KB = 25,600 blocks

Direct blocks: 12 → 12 blocks covered
Single indirect: 1024 → 1024 blocks covered
Double indirect: 1024 × 1024 = 1,048,576 blocks covered

Required blocks: 25,600

Direct: 12 used
Single indirect: 1024 used (1 indirect block)
Remaining blocks: 25,600 - 12 - 1024 = 24,564

Double indirect:
- Level-1 indirect blocks: ceil(24564 / 1024) = 24
- Level-2 indirect block: 1

Total indirect blocks:
- Single indirect: 1
- Double indirect level-1: 1
- Double indirect level-2: 24
Total: 26
```

</details>

### Problem 2: FAT Chain
Given the following FAT table, list the cluster chain for file A (start: 3).

| Cluster | Value |
|---------|-------|
| 0 | FREE |
| 1 | 8 |
| 2 | FREE |
| 3 | 7 |
| 4 | EOF |
| 5 | 1 |
| 6 | FREE |
| 7 | 4 |
| 8 | EOF |

<details>
<summary>Show Answer</summary>

```
File A starting: cluster 3

Track chain:
3 → FAT[3]=7 → FAT[7]=4 → FAT[4]=EOF

File A cluster chain: 3 → 7 → 4 → EOF

File size: 3 clusters

Note: another file starting at cluster 5:
5 → FAT[5]=1 → FAT[1]=8 → FAT[8]=EOF
Chain: 5 → 1 → 8 → EOF
```

</details>

### Problem 3: Journaling Recovery
After system crash, the journal has the following state. What is the file system state after recovery?

```
Journal contents:
[TxB_1] [Block 100: DataA] [Block 101: MetadataA] [TxE_1]
[TxB_2] [Block 200: DataB] [Block 201: MetadataB]
(No TxE_2)
```

<details>
<summary>Show Answer</summary>

```
Recovery process:

1. Check Transaction 1:
   - Both TxB_1 and TxE_1 exist
   - Completed transaction
   - Rewrite blocks 100, 101 to actual location (redo)

2. Check Transaction 2:
   - Only TxB_2, no TxE_2
   - Incomplete transaction
   - Ignore this transaction (undo)
   - Do not apply blocks 200, 201 changes

Result:
- Transaction 1 changes: Applied (file A changes complete)
- Transaction 2 changes: Not applied (no changes to file B)

File system is in consistent state with Transaction 1 only
```

</details>

### Problem 4: RAID Capacity Calculation
Calculate usable capacity when configuring the following RAID with 6× 2TB disks:

1. RAID 0
2. RAID 1
3. RAID 5
4. RAID 6
5. RAID 10

<details>
<summary>Show Answer</summary>

```
Disks: 6 × 2TB = 12TB total capacity

1. RAID 0: 6 × 2TB = 12TB (100%)
   - Use all space
   - No redundancy

2. RAID 1: 2 × 2TB = 4TB (3 pairs in mirror)
   OR simple 2-disk mirror: 2TB
   - 6 disks as 3 pairs, mirroring → each pair 2TB
   - Total: 3 × 2TB = 6TB (50%)

3. RAID 5: (6-1) × 2TB = 10TB (83%)
   - 1 disk worth for parity

4. RAID 6: (6-2) × 2TB = 8TB (67%)
   - 2 disks worth for parity

5. RAID 10: 6/2 × 2TB = 6TB (50%)
   - 3 stripes, each mirrored
   - OR: (disk count / 2) × single capacity
```

</details>

### Problem 5: File Deletion Process
Explain the process when deleting `/home/user/file.txt` in Unix file system from inode and block perspective.

<details>
<summary>Show Answer</summary>

```
Delete command: rm /home/user/file.txt

1. Path resolution:
   - Find /home directory's inode
   - Find /home/user directory's inode
   - Find file.txt's inode number (e.g., inode 12345)

2. Permission check:
   - Check write permission on directory (/home/user)
   - Check if file is write-locked

3. Remove directory entry:
   - Delete "file.txt" entry from /home/user directory file
   - Adjust rec_len to merge space

4. Decrease inode link count:
   - inode 12345's link_count--
   - If link_count > 0, stop here (other links exist)

5. If link_count == 0 and no open files:
   - Free inode 12345 in inode bitmap (mark as 0)
   - Check inode's data block pointers
   - Free all data blocks in block bitmap
   - Free indirect blocks too

6. Update superblock:
   - Increase free inode count
   - Increase free block count

Note: Actual data not erased
- Blocks just marked 'available'
- Recoverable until overwritten with new data
```

</details>

---

## Next Steps

Continue to [18_IO_and_IPC.md](./18_IO_and_IPC.md) to learn I/O systems and inter-process communication!

---

## References

- Silberschatz, "Operating System Concepts" Chapters 14-15
- Linux kernel source: `fs/ext4/`, `fs/fat/`
- ext4 documentation: https://www.kernel.org/doc/html/latest/filesystems/ext4/
- RAID fundamentals: https://en.wikipedia.org/wiki/RAID
