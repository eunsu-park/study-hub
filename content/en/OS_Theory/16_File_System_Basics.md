# File System Basics ⭐⭐⭐

**Previous**: [Page Replacement](./15_Page_Replacement.md) | **Next**: [File System Implementation](./17_File_System_Implementation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define what a file is from the OS perspective and explain how it abstracts raw disk storage
2. Describe common file attributes and operations including open, read, write, seek, and close
3. Distinguish sequential from direct and indexed access methods
4. Compare flat, hierarchical, and acyclic-graph directory structures
5. Explain file protection mechanisms using access control lists and capability lists
6. Analyze the open file table structure and how file descriptors work across fork()
7. Distinguish hard links from symbolic links and explain their trade-offs

---

Files are the primary abstraction for persistent data. Every document, photo, configuration, and database on your computer is a file. The file system is how the OS organizes, names, and protects these files -- turning a raw disk of magnetic sectors or flash cells into the familiar folder hierarchy you navigate every day. Understanding file system basics is the foundation for everything from system administration to database design.

## Table of Contents

1. [File Concept](#1-file-concept)
2. [File Attributes](#2-file-attributes)
3. [File Operations and System Calls](#3-file-operations-and-system-calls)
4. [Directory Structures](#4-directory-structures)
5. [File Access Methods](#5-file-access-methods)
6. [File System Mounting](#6-file-system-mounting)
7. [Practice Problems](#practice-problems)

---

## 1. File Concept

### 1.1 What is a File?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         File Definition                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   File = Named collection of related information                        │
│                                                                          │
│   Operating System Perspective:                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   Disk (Physical Storage Device)                                │   │
│   │   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐            │   │
│   │   │Block0│Block1│Block2│Block3│Block4│Block5│Block6│Block7│    │   │
│   │   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘            │   │
│   │       ↑         ↑     ↑         ↑     ↑     ↑                   │   │
│   │       └────┬────┘     └────┬────┘     └──┬──┘                   │   │
│   │            │               │             │                       │   │
│   │         File A          File B        File C                     │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   User Perspective:                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   Logical Storage Units                                          │   │
│   │                                                                  │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│   │   │ report.docx  │  │ photo.jpg    │  │ program.exe  │          │   │
│   │   │ (Document)   │  │ (Image)      │  │ (Executable) │          │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘          │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   File System: Abstraction layer connecting these two views             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 File Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          File Types                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────┬──────────────┬─────────────────────────────────┐    │
│   │     Type      │  Extension   │            Description          │    │
│   ├───────────────┼──────────────┼─────────────────────────────────┤    │
│   │ Executable    │ .exe, .com   │ Machine code                    │    │
│   │               │ .bin, (none) │                                 │    │
│   ├───────────────┼──────────────┼─────────────────────────────────┤    │
│   │ Object        │ .o, .obj     │ Compiled object code            │    │
│   ├───────────────┼──────────────┼─────────────────────────────────┤    │
│   │ Source Code   │ .c, .py, .js │ Programming language source     │    │
│   ├───────────────┼──────────────┼─────────────────────────────────┤    │
│   │ Text          │ .txt, .md    │ Plain text                      │    │
│   ├───────────────┼──────────────┼─────────────────────────────────┤    │
│   │ Library       │ .a, .so      │ Static/dynamic library          │    │
│   │               │ .lib, .dll   │                                 │    │
│   ├───────────────┼──────────────┼─────────────────────────────────┤    │
│   │ Image         │ .jpg, .png   │ Graphics data                   │    │
│   │               │ .gif, .bmp   │                                 │    │
│   ├───────────────┼──────────────┼─────────────────────────────────┤    │
│   │ Archive       │ .zip, .tar   │ Compressed archive              │    │
│   │               │ .gz, .7z     │                                 │    │
│   └───────────────┴──────────────┴─────────────────────────────────┘    │
│                                                                          │
│   Unix doesn't require extensions (convention only)                     │
│   Type detection by content: file command                                │
│                                                                          │
│   $ file /bin/ls                                                        │
│   /bin/ls: ELF 64-bit LSB shared object, x86-64...                     │
│                                                                          │
│   $ file report.pdf                                                     │
│   report.pdf: PDF document, version 1.4                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. File Attributes

### 2.1 Basic Attributes (Metadata)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         File Attributes                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────┬───────────────────────────────────────────────┐   │
│   │    Attribute    │                  Description                  │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Name            │ Human-readable file name                      │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Identifier      │ Unique number in file system (inode number)   │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Type            │ Regular file, directory, symlink, device, etc │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Location        │ Storage location on disk (block pointers)     │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Size            │ File size in bytes                            │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Protection      │ Access permissions (rwxrwxrwx)                │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Owner           │ File owner (UID)                              │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Group           │ File group (GID)                              │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Timestamps      │ Creation, modification, access time           │   │
│   └─────────────────┴───────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Unix/Linux stat Structure

```c
#include <sys/stat.h>
#include <time.h>

// stat structure (stores file attributes)
struct stat {
    dev_t     st_dev;      // Device ID
    ino_t     st_ino;      // inode number
    mode_t    st_mode;     // File mode (type + permissions)
    nlink_t   st_nlink;    // Number of hard links
    uid_t     st_uid;      // Owner UID
    gid_t     st_gid;      // Group GID
    dev_t     st_rdev;     // Device ID (special files)
    off_t     st_size;     // Total size in bytes
    blksize_t st_blksize;  // I/O block size
    blkcnt_t  st_blocks;   // Number of 512B blocks allocated

    // Timestamps
    struct timespec st_atim;  // Last access time
    struct timespec st_mtim;  // Last modification time
    struct timespec st_ctim;  // Last status change time
};

// File attribute query example
void print_file_info(const char* path) {
    struct stat sb;

    if (stat(path, &sb) == -1) {
        perror("stat");
        return;
    }

    printf("File: %s\n", path);
    printf("inode: %lu\n", sb.st_ino);
    printf("Size: %ld bytes\n", sb.st_size);
    printf("Blocks: %ld\n", sb.st_blocks);
    printf("Permissions: %o\n", sb.st_mode & 0777);
    printf("Owner: %d\n", sb.st_uid);
    printf("Links: %lu\n", sb.st_nlink);

    // File type
    if (S_ISREG(sb.st_mode))  printf("Type: Regular file\n");
    if (S_ISDIR(sb.st_mode))  printf("Type: Directory\n");
    if (S_ISLNK(sb.st_mode))  printf("Type: Symbolic link\n");
    if (S_ISCHR(sb.st_mode))  printf("Type: Character device\n");
    if (S_ISBLK(sb.st_mode))  printf("Type: Block device\n");

    printf("Modified: %s", ctime(&sb.st_mtime));
}
```

### 2.3 Interpreting ls -l Output

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ls -l Output Analysis                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   $ ls -l                                                               │
│   -rw-r--r-- 1 user group 4096 Jan 15 10:30 document.txt               │
│   │├──┤├─┤│  │  │     │     │      │     │     └── Filename            │
│   ││  │ │ │  │  │     │     │      │     └── Time                      │
│   ││  │ │ │  │  │     │     │      └── Date                            │
│   ││  │ │ │  │  │     │     └── Size (bytes)                           │
│   ││  │ │ │  │  │     └── Group                                        │
│   ││  │ │ │  │  └── Owner                                               │
│   ││  │ │ │  └── Hard link count                                        │
│   ││  │ │ └── Others permissions                                        │
│   ││  │ └── Group permissions                                           │
│   ││  └── Owner permissions                                             │
│   │└── File type                                                        │
│   └── - regular, d directory, l link, c char device, b block device    │
│                                                                          │
│   Permissions:                                                           │
│   r (4) = read                                                          │
│   w (2) = write                                                         │
│   x (1) = execute (for directory: access)                               │
│                                                                          │
│   Example: rw-r--r-- = 644 = Owner: rw, Group: r, Others: r            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. File Operations and System Calls

### 3.1 Basic File Operations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        File Operations                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────┬───────────────────────────────────────────────┐   │
│   │   Operation     │                  Description                  │   │
│   ├─────────────────┼───────────────────────────────────────────────┤   │
│   │ Create          │ Create new file, add entry to directory      │   │
│   │ Open            │ Prepare file access, return file descriptor  │   │
│   │ Read            │ Read data from current position              │   │
│   │ Write           │ Write data at current position               │   │
│   │ Seek            │ Change current position in file              │   │
│   │ Close           │ End file access, release resources           │   │
│   │ Delete          │ Remove file, delete directory entry          │   │
│   │ Truncate        │ Set file size to 0 (delete content)          │   │
│   └─────────────────┴───────────────────────────────────────────────┘   │
│                                                                          │
│   Additional operations:                                                 │
│   - Append: Add data to end of file                                     │
│   - Rename: Change file name                                            │
│   - Get/Set Attributes: Query/modify attributes                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 System Call Examples

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

// Create and write file
void create_and_write(const char* path) {
    // O_CREAT: create if doesn't exist
    // O_WRONLY: write-only
    // O_TRUNC: delete existing content
    // 0644: permissions (rw-r--r--)
    int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd == -1) {
        perror("open");
        exit(1);
    }

    const char* data = "Hello, File System!\n";
    ssize_t written = write(fd, data, strlen(data));

    if (written == -1) {
        perror("write");
    } else {
        printf("%zd bytes written\n", written);
    }

    close(fd);
}

// Read file
void read_file(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return;
    }

    char buffer[1024];
    ssize_t bytes_read;

    // read() returns bytes read, 0 is EOF
    while ((bytes_read = read(fd, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytes_read] = '\0';
        printf("%s", buffer);
    }

    if (bytes_read == -1) {
        perror("read");
    }

    close(fd);
}

// File position seek
void seek_example(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) return;

    // Seek to end to check size
    off_t size = lseek(fd, 0, SEEK_END);
    printf("File size: %ld bytes\n", size);

    // Return to beginning
    lseek(fd, 0, SEEK_SET);

    // Move forward 10 bytes
    lseek(fd, 10, SEEK_CUR);

    char buffer[100];
    read(fd, buffer, 10);
    buffer[10] = '\0';
    printf("10 chars from offset 10: %s\n", buffer);

    close(fd);
}

// File copy implementation
int copy_file(const char* src, const char* dst) {
    int src_fd = open(src, O_RDONLY);
    if (src_fd == -1) return -1;

    int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dst_fd == -1) {
        close(src_fd);
        return -1;
    }

    char buffer[4096];
    ssize_t bytes;

    while ((bytes = read(src_fd, buffer, sizeof(buffer))) > 0) {
        ssize_t written = write(dst_fd, buffer, bytes);
        if (written != bytes) {
            close(src_fd);
            close(dst_fd);
            return -1;
        }
    }

    close(src_fd);
    close(dst_fd);
    return 0;
}

int main() {
    create_and_write("test.txt");
    read_file("test.txt");
    seek_example("test.txt");
    copy_file("test.txt", "test_copy.txt");
    return 0;
}
```

### 3.3 Open File Table

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Open File Table Structure                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Process A                Process B                                    │
│   ┌──────────────────┐      ┌──────────────────┐                       │
│   │ fd table         │      │ fd table         │                       │
│   ├──────────────────┤      ├──────────────────┤                       │
│   │ 0: stdin         │      │ 0: stdin         │                       │
│   │ 1: stdout        │      │ 1: stdout        │                       │
│   │ 2: stderr        │      │ 2: stderr        │                       │
│   │ 3: ─────────────┐│      │ 3: ─────────────┐│                       │
│   │ 4: ────────────┐││      └─────────────────┼┘                       │
│   └────────────────┼┼┘                        │                        │
│                    ││                         │                        │
│                    ▼▼                         │                        │
│   System-wide Open File Table                 │                        │
│   ┌─────────────────────────────────────────┐ │                        │
│   │ Entry 1:                                │ │                        │
│   │   - File offset: 100                    │◀┘                        │
│   │   - Access mode: O_RDONLY               │                          │
│   │   - Reference count: 2                  │                          │
│   │   - inode pointer: ──────────────────────┼───┐                      │
│   ├─────────────────────────────────────────┤   │                      │
│   │ Entry 2:                                │   │                      │
│   │   - File offset: 500                    │   │                      │
│   │   - Access mode: O_RDWR                 │   │                      │
│   │   - Reference count: 1                  │   │                      │
│   │   - inode pointer: ──────────────────────┼───┼───┐                  │
│   └─────────────────────────────────────────┘   │   │                  │
│                                                 │   │                  │
│   In-Memory inode Table                         │   │                  │
│   ┌─────────────────────────────────────────┐   │   │                  │
│   │ inode 1234:                             │◀──┘   │                  │
│   │   - File size: 4096                     │       │                  │
│   │   - Disk block locations                │       │                  │
│   │   - Permissions, owner, etc.            │       │                  │
│   ├─────────────────────────────────────────┤       │                  │
│   │ inode 5678:                             │◀──────┘                  │
│   │   - File size: 8192                     │                          │
│   │   - ...                                 │                          │
│   └─────────────────────────────────────────┘                          │
│                                                                          │
│   After fork(): fd table copied, open file entry shared (offset shared!)│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structures

### 4.1 Single-Level Directory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Single-Level Directory                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                         Root Directory                           │   │
│   ├─────────────────────────────────────────────────────────────────┤   │
│   │ cat │ bo │ a │ test │ data │ mail │ cont │ hex │ records │     │   │
│   └──┬──┴──┬─┴──┬─┴───┬──┴───┬──┴───┬──┴───┬──┴──┬──┴────┬────┘     │   │
│      │     │    │     │      │      │      │     │       │           │   │
│      ▼     ▼    ▼     ▼      ▼      ▼      ▼     ▼       ▼           │   │
│     file  file file  file   file   file   file  file    file         │   │
│                                                                          │
│   Problems:                                                              │
│   - Name collision: All users cannot use same names                     │
│   - Management difficulty: Hard to find files as count grows            │
│   - No grouping: Cannot organize related files                          │
│                                                                          │
│   Used in early OS (CP/M)                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Two-Level Directory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Two-Level Directory                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                    Master File Directory (MFD)                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │            user1              user2              user3          │   │
│   └───────────────┬────────────────┬─────────────────┬──────────────┘   │
│                   │                │                 │                  │
│                   ▼                ▼                 ▼                  │
│        User File Directory (UFD)                                         │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │
│   │ cat   test  a   │ │ cat   data  b   │ │ hex   mail     │          │
│   └──┬──────┬────┬──┘ └──┬──────┬────┬──┘ └──┬──────┬──────┘          │
│      │      │    │       │      │    │       │      │                  │
│      ▼      ▼    ▼       ▼      ▼    ▼       ▼      ▼                  │
│                                                                          │
│   Advantages:                                                            │
│   - Independent namespace per user                                       │
│   - user1's cat ≠ user2's cat                                           │
│                                                                          │
│   Disadvantages:                                                         │
│   - Difficult to share files between users                              │
│   - No subdirectories                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Tree-Structured Directory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Tree-Structured Directory                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                              /                                           │
│                              │                                           │
│           ┌──────────────────┼──────────────────┐                       │
│           │                  │                  │                       │
│          bin               home               etc                       │
│           │                  │                  │                       │
│     ┌─────┴─────┐    ┌───────┼───────┐    ┌────┴────┐                  │
│    ls          cat   │       │       │   passwd   hosts                │
│                    user1   user2   user3                                │
│                      │       │       │                                  │
│                 ┌────┴────┐  │    documents                             │
│               docs     code  data     │                                  │
│                │         │    │    ┌──┴──┐                              │
│            ┌───┼───┐   main.c  │   report  notes                        │
│          a.txt b.txt         file.txt                                   │
│                                                                          │
│   Features:                                                              │
│   - Arbitrary depth subdirectories allowed                               │
│   - Absolute path: /home/user1/docs/a.txt                               │
│   - Relative path: ./docs/a.txt (from current directory)                │
│   - Current directory (.), parent directory (..)                        │
│                                                                          │
│   Used by most modern operating systems                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Acyclic Graph Directory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Acyclic Graph Directory                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                              /                                           │
│                              │                                           │
│           ┌──────────────────┼──────────────────┐                       │
│           │                  │                  │                       │
│          home              shared              │                        │
│           │                  │                 │                        │
│     ┌─────┴─────┐           project ◀─────────┘                        │
│   user1       user2          │ ▲                                        │
│     │           │      ┌─────┴─────┐                                    │
│  myproj ───────────────│           │                                    │
│                        │      shared file                                │
│                   ourproj ────┘                                          │
│                                                                          │
│   File/directory sharing methods:                                       │
│                                                                          │
│   1. Hard Link:                                                          │
│      - Different name pointing to same inode                            │
│      - ln target link_name                                              │
│      - On delete, only link count decreases                             │
│      - Cannot use on directories (prevents cycles)                      │
│                                                                          │
│   2. Symbolic Link:                                                      │
│      - Stores path to another file                                      │
│      - ln -s target link_name                                           │
│      - Broken link if original deleted (dangling link)                  │
│      - Can use on directories                                           │
│                                                                          │
│   $ ln /shared/project myproj          # Hard link                      │
│   $ ln -s /shared/project myproj       # Symbolic link                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. File Access Methods

### 5.1 Sequential Access

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Sequential Access                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   File                                                                   │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                            │
│   │ A │ B │ C │ D │ E │ F │ G │ H │ I │ J │                            │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                            │
│             ↑                                                            │
│          Current position                                                │
│                                                                          │
│   Operations:                                                            │
│   - read(): read from current position, advance                         │
│   - write(): write at current position, advance                         │
│   - reset(): return to beginning                                        │
│                                                                          │
│   Features:                                                              │
│   - Originated from tape-based systems                                   │
│   - Commonly used in editors, compilers                                  │
│   - Suitable for log file processing                                     │
│                                                                          │
│   Example: Sequential read to end of file                               │
│   while (read(fd, &buf, size) > 0) {                                    │
│       process(buf);                                                      │
│   }                                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Direct/Random Access

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Direct Access                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   File (logical blocks)                                                  │
│   Block:  0     1     2     3     4     5     6     7     8     9       │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│   │     │     │     │     │     │     │     │     │     │     │        │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
│           ↑                       ↑                                      │
│        Block 1                  Block 5                                  │
│         read                    read                                     │
│                                                                          │
│   Operations:                                                            │
│   - read(n): read block n                                               │
│   - write(n): write block n                                             │
│   - seek(n): move to block n                                            │
│                                                                          │
│   Features:                                                              │
│   - Disk-based systems                                                   │
│   - Suitable for databases                                               │
│   - Random position access                                               │
│                                                                          │
│   Example: Direct record access                                          │
│   #define RECORD_SIZE 100                                               │
│   lseek(fd, record_num * RECORD_SIZE, SEEK_SET);                        │
│   read(fd, &record, RECORD_SIZE);                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Indexed Access

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Indexed Access                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Index File                          Data File                          │
│   ┌────────────────────┐                 ┌─────────────────────────┐    │
│   │ Key     │Block Ptr │                 │ Block 0                  │    │
│   ├─────────┼──────────┤                 ├─────────────────────────┤    │
│   │ "Apple" │     7    │────────────────▶│ Block 7: Apple data     │    │
│   │ "Banana"│     3    │────────────┐   ├─────────────────────────┤    │
│   │ "Cherry"│    12    │──────────┐ │   │ Block 3: Banana data    │◀──┤
│   │ "Date"  │     5    │────────┐ │ │   ├─────────────────────────┤    │
│   │  ...    │   ...    │        │ │ │   │ Block 12: Cherry data   │◀──┘
│   └─────────┴──────────┘        │ │ │   ├─────────────────────────┤
│                                  │ │ │   │ Block 5: Date data      │◀──┐
│                                  │ │ └──▶│                         │   │
│                                  │ │     └─────────────────────────┘   │
│                                  │ │                                    │
│                                  └─┼────────────────────────────────────┘
│                                    │
│                                    └── (Find location from index, direct access)
│
│   Features:                                                              │
│   - Fast search in large files                                          │
│   - Multi-level index if index itself too large                         │
│   - Database B+tree index                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. File System Mounting

### 6.1 Mount Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         File System Mounting                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Before mount:                                                          │
│                                                                          │
│   Root file system (/)          USB drive                               │
│           /                          Independent tree                    │
│          ╱│╲                              │                             │
│        bin home etc                      ╱ ╲                            │
│             │                         photo  doc                         │
│           user1                                                          │
│            │                                                             │
│          mnt (empty directory)                                           │
│                                                                          │
│   After mount (mount /dev/sdb1 /home/user1/mnt):                        │
│                                                                          │
│           /                                                              │
│          ╱│╲                                                            │
│        bin home etc                                                      │
│             │                                                            │
│           user1                                                          │
│            │                                                             │
│          mnt ◀─────────── Mount point                                   │
│          ╱ ╲                                                            │
│       photo  doc   ← USB drive contents visible here                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Mount Commands

```bash
# Check mount status
$ mount
/dev/sda1 on / type ext4 (rw,relatime)
/dev/sdb1 on /mnt/usb type vfat (rw,user)

# Mount USB drive
$ sudo mount /dev/sdb1 /mnt/usb

# Mount with options
$ sudo mount -t ext4 -o ro,noexec /dev/sdc1 /mnt/backup

# Mount options:
# ro: read-only
# rw: read/write
# noexec: cannot execute programs
# nosuid: ignore setuid
# user: allow normal users to mount

# Unmount
$ sudo umount /mnt/usb

# /etc/fstab - automatic mount at boot
# Device            Mount point     Type    Options            Dump Pass
/dev/sda1          /               ext4    defaults           1    1
/dev/sda2          /home           ext4    defaults           1    2
/dev/sdb1          /mnt/data       ext4    defaults,nofail    0    2
UUID=xxxx-xxxx     /mnt/usb        vfat    user,noauto        0    0
```

---

## Practice Problems

### Problem 1: File Attribute Interpretation
Interpret each field from the following `ls -l` output:

```
-rwxr-x--- 2 alice developers 8192 Mar 15 14:30 script.sh
```

<details>
<summary>Show Answer</summary>

```
- : Regular file (d for directory, l for link)
rwx : Owner (alice) permissions - read/write/execute
r-x : Group (developers) permissions - read/execute
--- : Others permissions - none
2 : Hard link count (2 names for this file)
alice : Owner
developers : Owner group
8192 : File size (8KB)
Mar 15 14:30 : Last modification time
script.sh : File name

Numeric permissions: 750 (7=rwx, 5=r-x, 0=---)
```

</details>

### Problem 2: System Call Sequence
Write the system call sequence to append "Hello World" to `/tmp/log.txt`.

<details>
<summary>Show Answer</summary>

```c
int fd = open("/tmp/log.txt", O_WRONLY | O_APPEND | O_CREAT, 0644);
// O_APPEND: always write at end of file
// O_CREAT: create if doesn't exist
// 0644: permission setting

if (fd == -1) {
    perror("open failed");
    exit(1);
}

const char* msg = "Hello World\n";
ssize_t written = write(fd, msg, strlen(msg));

if (written == -1) {
    perror("write failed");
}

close(fd);
```

</details>

### Problem 3: Hard Link vs Symbolic Link
Explain the behavior of each link type in this scenario:

```bash
$ echo "original" > file.txt
$ ln file.txt hardlink.txt       # Hard link
$ ln -s file.txt symlink.txt     # Symbolic link
$ rm file.txt
$ cat hardlink.txt
$ cat symlink.txt
```

<details>
<summary>Show Answer</summary>

```
$ cat hardlink.txt
original
→ Hard link points to same inode
→ Data still exists even after file.txt deleted
→ Link count just decreased by 1

$ cat symlink.txt
cat: symlink.txt: No such file or directory
→ Symbolic link stores path "file.txt"
→ Broken link (dangling link) after file.txt deleted
→ Target doesn't exist, so error

Hard link characteristics:
- Shares same inode number
- Equal status with original
- Cannot use on directories

Symbolic link characteristics:
- Separate inode (stores path string)
- Dependent on original
- Can use on directories
- Can point to different file systems
```

</details>

### Problem 4: Directory Navigation
Canonicalize the path `/home/user/docs/../code/./main.c`.

<details>
<summary>Show Answer</summary>

```
/home/user/docs/../code/./main.c

1. Navigate to /home/user/docs
2. .. goes up: /home/user
3. Navigate to code: /home/user/code
4. . is current directory (ignore): /home/user/code
5. main.c file: /home/user/code/main.c

Canonicalized path: /home/user/code/main.c

Verify in Linux:
$ realpath /home/user/docs/../code/./main.c
/home/user/code/main.c
```

</details>

### Problem 5: Open File Table
Process A opens a file and creates process B via fork(). If both processes write 100 bytes each to the same file, what is the final file size?

<details>
<summary>Show Answer</summary>

```
After fork(), child copies parent's fd table.
Both processes' fds share the same open file entry.
Therefore they share the same offset!

Scenario:
1. Parent open(), offset = 0
2. fork(), child shares same open file entry
3. Parent write(100), offset = 100
4. Child write(100), offset = 200 (continues writing)

Final file size: 200 bytes

If child had separately open()ed:
- Separate open file entry created
- Separate offset
- Both starting from 0 would overwrite!
- Final size: 100 bytes

Note: In reality, results may vary based on write() atomicity
and buffering. O_APPEND usage recommended.
```

</details>

---

## Next Steps

Continue to [17_File_System_Implementation.md](./17_File_System_Implementation.md) to learn file system internals!

---

## References

- Silberschatz, "Operating System Concepts" Chapter 13
- Linux man pages: `open(2)`, `read(2)`, `write(2)`, `stat(2)`
- POSIX file system standards
- The Linux Programming Interface by Michael Kerrisk
