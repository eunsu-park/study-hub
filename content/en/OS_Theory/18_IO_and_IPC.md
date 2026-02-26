# I/O and IPC ⭐⭐⭐

**Previous**: [File System Implementation](./17_File_System_Implementation.md) | **Next**: None (final lesson)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain I/O hardware organization including ports, buses, and device controllers
2. Compare programmed I/O (polling), interrupt-driven I/O, and DMA transfer methods
3. Describe the role of device drivers and the layered I/O subsystem architecture
4. Compare IPC mechanisms including pipes, message queues, shared memory, and sockets
5. Explain when to use shared memory versus message passing based on performance and complexity trade-offs
6. Implement basic IPC using anonymous pipes, named pipes (FIFOs), and POSIX shared memory
7. Design a simple client-server communication using TCP sockets

---

A computer is useless without I/O -- keyboards, displays, disks, and networks are how it interacts with the world. Inter-process communication extends this idea to processes talking to each other. Together, I/O and IPC are the nervous system of the operating system, connecting isolated processes to hardware and to each other. This lesson ties together everything you have learned about processes, memory, and file systems into the complete picture of how an OS manages a running system.

## Table of Contents

1. [I/O Hardware](#1-io-hardware)
2. [I/O Methods](#2-io-methods)
3. [Device Drivers](#3-device-drivers)
4. [Buffering Strategies](#4-buffering-strategies)
5. [IPC Overview](#5-ipc-overview)
6. [Pipes](#6-pipes)
7. [Shared Memory](#7-shared-memory)
8. [Message Queues and Sockets](#8-message-queues-and-sockets)
9. [Practice Problems](#practice-problems)

---

## 1. I/O Hardware

### 1.1 I/O Device Classification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        I/O Device Classification                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌────────────────────────┬────────────────────────────────────────┐   │
│   │      Classification    │              Examples                  │   │
│   ├────────────────────────┼────────────────────────────────────────┤   │
│   │ Block Device           │ Hard disk, SSD, USB storage           │   │
│   │                        │ - Fixed-size block access              │   │
│   │                        │ - Random access possible               │   │
│   ├────────────────────────┼────────────────────────────────────────┤   │
│   │ Character Device       │ Keyboard, mouse, printer, serial port │   │
│   │                        │ - Byte stream access                   │   │
│   │                        │ - Sequential access                    │   │
│   ├────────────────────────┼────────────────────────────────────────┤   │
│   │ Network Device         │ Ethernet, WiFi, Bluetooth             │   │
│   │                        │ - Packet-based                         │   │
│   │                        │ - Socket interface                     │   │
│   └────────────────────────┴────────────────────────────────────────┘   │
│                                                                          │
│   Linux device files:                                                    │
│   /dev/sda       - First SCSI/SATA disk (block)                         │
│   /dev/tty1      - First terminal (character)                           │
│   /dev/null      - Null device (character)                              │
│   /dev/random    - Random number generator (character)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 I/O Hardware Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      I/O Hardware Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                            CPU                                    │  │
│   └───────────────────────────────┬──────────────────────────────────┘  │
│                                   │                                      │
│                                   ▼                                      │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                        System Bus                                 │  │
│   └──────┬─────────────┬─────────────┬─────────────┬────────────────┘  │
│          │             │             │             │                    │
│          ▼             ▼             ▼             ▼                    │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│   │  Memory    │ │ Disk       │ │ Graphics   │ │ Network    │          │
│   │ Controller │ │ Controller │ │ Controller │ │ Controller │          │
│   └────────────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘          │
│                        │              │              │                  │
│                        ▼              ▼              ▼                  │
│                   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│                   │  HDD    │   │  GPU    │   │  NIC    │              │
│                   │  SSD    │   │         │   │         │              │
│                   └─────────┘   └─────────┘   └─────────┘              │
│                                                                          │
│   Device Controller Components:                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Status Register  │ Command Register │ Data Register           │   │
│   │  (Status)         │ (Command)        │ (Data)                  │   │
│   │  - Ready/complete │ - Read/write     │ - I/O data buffer       │   │
│   │  - Error          │ - Control cmds   │                         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. I/O Methods

### 2.1 Polling (Programmed I/O)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Polling Method                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   CPU repeatedly checks device status                                    │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  CPU                          Device Controller                  │   │
│   │                                                                  │   │
│   │  1. Send command ──────────────────▶ Command register           │   │
│   │                                                                  │   │
│   │  2. while (status == busy) {    ◀── Status register             │   │
│   │        // Keep checking (CPU waste!)                            │   │
│   │     }                                                            │   │
│   │                                                                  │   │
│   │  3. Data transfer ──────────────▶/◀── Data register             │   │
│   │                                                                  │   │
│   │  4. Check completion            ◀── Status register             │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Advantages:                                                            │
│   - Simple implementation                                                │
│   - Low overhead for fast devices                                       │
│                                                                          │
│   Disadvantages:                                                         │
│   - CPU time waste (Busy Waiting)                                       │
│   - Inefficient for slow devices                                        │
│                                                                          │
│   Use case: Fast network devices (high throughput)                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Interrupt-Driven I/O

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Interrupt Method                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Time ──────────────────────────────────────────────────▶              │
│                                                                          │
│   CPU:    [Other work]              [Interrupt handler]  [Other work]   │
│                │                              ▲                          │
│                │ I/O request                  │ Interrupt                │
│                ▼                              │ Complete signal          │
│   Device:     [I/O operation executing.......] │                        │
│                                               │                          │
│                                                                          │
│   Interrupt handling process:                                            │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │  1. Device generates interrupt signal                           │   │
│   │                                                                  │   │
│   │  2. CPU suspends current work                                   │   │
│   │     - Save registers                                            │   │
│   │     - Save PC (Program Counter)                                 │   │
│   │                                                                  │   │
│   │  3. Look up interrupt vector table                              │   │
│   │     Interrupt number → Handler address                          │   │
│   │                                                                  │   │
│   │  4. Execute interrupt handler (ISR)                             │   │
│   │     - Check device status                                       │   │
│   │     - Transfer data                                             │   │
│   │     - Wake waiting process                                      │   │
│   │                                                                  │   │
│   │  5. Restore registers/PC, resume original work                  │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Advantages: CPU can do other work while waiting for I/O               │
│   Disadvantages: Interrupt overhead, inefficient for frequent I/O       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 DMA (Direct Memory Access)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             DMA Method                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Direct data transfer between device and memory without CPU            │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                                                                   │  │
│   │      ┌──────┐                    ┌─────────────────┐             │  │
│   │      │ CPU  │──(1) Setup────────▶│  DMA Controller │             │  │
│   │      │      │◀─(4) Interrupt─────│                 │             │  │
│   │      └──────┘                    │  - Source addr  │             │  │
│   │         │                        │  - Dest addr    │             │  │
│   │         │ (Do other work)        │  - Byte count   │             │  │
│   │         │                        │  - Direction    │             │  │
│   │         │                        └────────┬────────┘             │  │
│   │         │                                 │                      │  │
│   │         │                         (2)(3) Direct transfer         │  │
│   │         ▼                                 │                      │  │
│   │   ┌──────────┐                           │                      │  │
│   │   │  Memory  │◀──────────────────────────┘                      │  │
│   │   │          │                           │                      │  │
│   │   └──────────┘                           │                      │  │
│   │                                          │                      │  │
│   │                              ┌───────────▼───────────┐          │  │
│   │                              │      Disk Device      │          │  │
│   │                              └───────────────────────┘          │  │
│   │                                                                   │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   DMA transfer process:                                                  │
│   1. CPU sets transfer information in DMA controller                    │
│   2. DMA uses bus to transfer data (Cycle Stealing)                     │
│   3. CPU performs cache/register operations during transfer             │
│   4. DMA generates interrupt when transfer complete                     │
│                                                                          │
│   Advantages: Minimal CPU load for large data transfers                 │
│   Disadvantages: DMA controller cost, bus contention                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Device Drivers

### 3.1 Driver Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Device Driver Layers                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     User Applications                            │   │
│   │                   (open, read, write, ioctl)                    │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │ System calls                           │
│                                ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                         VFS Layer                                │   │
│   │               (Virtual File System)                              │   │
│   │            Device-independent abstraction interface              │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │                                        │
│          ┌─────────────────────┼─────────────────────┐                 │
│          ▼                     ▼                     ▼                 │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐         │
│   │Block Driver │       │ Char Driver │       │  Network    │         │
│   │  (Block)    │       │ (Character) │       │   Driver    │         │
│   └──────┬──────┘       └──────┬──────┘       └──────┬──────┘         │
│          │                     │                     │                 │
│          ▼                     ▼                     ▼                 │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐         │
│   │ SCSI/SATA   │       │  TTY/Serial │       │  Ethernet   │         │
│   │   Driver    │       │   Driver    │       │   Driver    │         │
│   └──────┬──────┘       └──────┬──────┘       └──────┬──────┘         │
│          │                     │                     │                 │
│          ▼                     ▼                     ▼                 │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐         │
│   │   Hardware  │       │   Hardware  │       │   Hardware  │         │
│   └─────────────┘       └─────────────┘       └─────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Linux Driver Example

```c
// Simple character device driver structure
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "mydevice"

static int major_number;
static char device_buffer[1024];
static int open_count = 0;

// Open device
static int device_open(struct inode *inode, struct file *file) {
    open_count++;
    printk(KERN_INFO "mydevice: opened %d time(s)\n", open_count);
    return 0;
}

// Close device
static int device_release(struct inode *inode, struct file *file) {
    printk(KERN_INFO "mydevice: closed\n");
    return 0;
}

// Read from device
static ssize_t device_read(struct file *file, char __user *buffer,
                           size_t length, loff_t *offset) {
    int bytes_to_read = min(length, sizeof(device_buffer) - (size_t)*offset);

    if (*offset >= sizeof(device_buffer)) return 0;

    if (copy_to_user(buffer, device_buffer + *offset, bytes_to_read)) {
        return -EFAULT;
    }

    *offset += bytes_to_read;
    return bytes_to_read;
}

// Write to device
static ssize_t device_write(struct file *file, const char __user *buffer,
                            size_t length, loff_t *offset) {
    int bytes_to_write = min(length, sizeof(device_buffer) - 1);

    if (copy_from_user(device_buffer, buffer, bytes_to_write)) {
        return -EFAULT;
    }

    device_buffer[bytes_to_write] = '\0';
    return bytes_to_write;
}

// File operations structure
static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = device_open,
    .release = device_release,
    .read = device_read,
    .write = device_write,
};

// Module initialization
static int __init mydevice_init(void) {
    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_number < 0) {
        printk(KERN_ALERT "Failed to register device\n");
        return major_number;
    }
    printk(KERN_INFO "mydevice: registered with major number %d\n",
           major_number);
    return 0;
}

// Module cleanup
static void __exit mydevice_exit(void) {
    unregister_chrdev(major_number, DEVICE_NAME);
    printk(KERN_INFO "mydevice: unregistered\n");
}

module_init(mydevice_init);
module_exit(mydevice_exit);
MODULE_LICENSE("GPL");
```

---

## 4. Buffering Strategies

### 4.1 Buffering Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Buffering Types                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Single Buffering                                                    │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                        │
│   │ Device  │ ───▶ │  Buffer │ ───▶ │ Process │                        │
│   └─────────┘      └─────────┘      └─────────┘                        │
│                                                                          │
│   Problem: Device waits while buffer being processed                    │
│                                                                          │
│   2. Double Buffering                                                    │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                        │
│   │ Device  │ ───▶ │ Buffer A│ ───▶ │ Process │                        │
│   │         │      ├─────────┤      │         │                        │
│   │         │ ───▶ │ Buffer B│ ───▶ │         │                        │
│   └─────────┘      └─────────┘      └─────────┘                        │
│                                                                          │
│   Device writes to A while process reads from B (parallel processing)   │
│                                                                          │
│   3. Circular Buffering                                                  │
│                                                                          │
│        ┌───────────────────────────────────────┐                        │
│        │           Circular Buffer Queue       │                        │
│        │    ┌───┬───┬───┬───┬───┬───┬───┐    │                        │
│        │    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │    │                        │
│        │    └───┴───┴───┴───┴───┴───┴───┘    │                        │
│        │        ↑               ↑             │                        │
│        │      head            tail            │                        │
│        │   (consume pos)   (produce pos)      │                        │
│        └───────────────────────────────────────┘                        │
│                                                                          │
│   Efficient for producer-consumer pattern                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Spooling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Spooling                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Simultaneous Peripheral Operations On-Line                            │
│                                                                          │
│   Printer spooling example:                                             │
│                                                                          │
│   ┌─────────┐                                                           │
│   │Process 1│───┐                                                      │
│   └─────────┘   │    ┌─────────────────┐    ┌─────────────┐            │
│                 │    │                  │    │             │            │
│   ┌─────────┐   ├───▶│   Spool Dir     │───▶│  Printer    │            │
│   │Process 2│───┤    │   (Disk Queue)  │    │  Daemon     │───▶ Printer│
│   └─────────┘   │    │                  │    │(Sequential) │            │
│                 │    │  job1.spl       │    └─────────────┘            │
│   ┌─────────┐   │    │  job2.spl       │                                │
│   │Process 3│───┘    │  job3.spl       │                                │
│   └─────────┘         └─────────────────┘                                │
│                                                                          │
│   Features:                                                              │
│   - Virtualizes slow device (printer)                                   │
│   - Multiple processes can "print" simultaneously                        │
│   - Actual printing happens sequentially                                 │
│   - Process returns immediately (asynchronous)                           │
│                                                                          │
│   Other examples: Mail spool, batch job queue                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. IPC Overview

### 5.1 IPC Methods Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IPC Methods Comparison                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┬──────────────────────────────────────────────────┐ │
│  │     Method      │                    Features                      │ │
│  ├─────────────────┼──────────────────────────────────────────────────┤ │
│  │ Pipe            │ - Unidirectional (anonymous), bidirectional(named)│ │
│  │                 │ - Parent-child process communication             │ │
│  │                 │ - Byte stream                                    │ │
│  ├─────────────────┼──────────────────────────────────────────────────┤ │
│  │ Shared Memory   │ - Fastest (minimal kernel involvement)           │ │
│  │                 │ - Requires synchronization (semaphores, etc)     │ │
│  │                 │ - Suitable for large data                        │ │
│  ├─────────────────┼──────────────────────────────────────────────────┤ │
│  │ Message Queue   │ - Structured messages                            │ │
│  │                 │ - Asynchronous communication                     │ │
│  │                 │ - Priority support                               │ │
│  ├─────────────────┼──────────────────────────────────────────────────┤ │
│  │ Signal          │ - Asynchronous notification                      │ │
│  │                 │ - Limited information transfer                   │ │
│  │                 │ - Similar to interrupt                           │ │
│  ├─────────────────┼──────────────────────────────────────────────────┤ │
│  │ Socket          │ - Network communication                          │ │
│  │                 │ - Cross-system communication possible            │ │
│  │                 │ - TCP/UDP protocols                              │ │
│  └─────────────────┴──────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Pipes

### 6.1 Anonymous Pipe

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Anonymous Pipe                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Unidirectional communication between parent-child processes            │
│                                                                          │
│   ┌─────────────┐                     ┌─────────────┐                   │
│   │   Parent    │      Pipe           │   Child     │                   │
│   │  Process    │                     │  Process    │                   │
│   │             │  ┌───────────────┐  │             │                   │
│   │ write(fd[1])│─▶│===============│─▶│ read(fd[0]) │                   │
│   │             │  └───────────────┘  │             │                   │
│   │  close(fd[0])                     │ close(fd[1])│                   │
│   └─────────────┘                     └─────────────┘                   │
│                                                                          │
│   fd[0]: Read end                                                        │
│   fd[1]: Write end                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

```c
// Anonymous pipe example
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main() {
    int pipefd[2];
    pid_t pid;
    char buffer[1024];

    // Create pipe
    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(1);
    }

    pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(1);
    }

    if (pid == 0) {
        // Child process: read from pipe
        // Why: Close the unused write end in the reader process. A pipe stays open
        // as long as ANY write end exists. If the child keeps fd[1] open, read()
        // will block forever waiting for EOF even after the parent closes its write
        // end -- because the pipe still has an open writer (the child itself).
        close(pipefd[1]);  // Close write end

        ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
        printf("Child received: %.*s\n", (int)n, buffer);

        close(pipefd[0]);
        exit(0);
    } else {
        // Parent process: write to pipe
        // Why: Close the unused read end in the writer process. If the parent keeps
        // fd[0] open and the child exits, writing to a pipe with no readers would
        // normally generate SIGPIPE. But if the parent still holds the read end,
        // the kernel thinks there is still a reader, so write() blocks instead of
        // signaling an error. Always close the end you don't use.
        close(pipefd[0]);  // Close read end

        const char* message = "Hello from parent!";
        write(pipefd[1], message, strlen(message));
        printf("Parent sent: %s\n", message);

        close(pipefd[1]);
        wait(NULL);
    }

    return 0;
}
```

> **Why fork+exec with pipes?** `fork()` creates a child process that inherits all open file descriptors from the parent -- including both ends of the pipe. `exec()` then replaces the child's program image but preserves open file descriptors. This two-step pattern is exactly how shell piping (`cmd1 | cmd2`) works: the shell creates a pipe, forks twice, redirects stdout/stdin to the pipe ends, then execs each command. The new programs never know they are connected by a pipe -- they just read from stdin and write to stdout as usual.

### 6.2 Named Pipe (FIFO)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Named Pipe (FIFO)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Pipe with name in file system                                         │
│   Communication between unrelated processes possible                    │
│                                                                          │
│   ┌─────────────┐    /tmp/myfifo    ┌─────────────┐                    │
│   │ Process A   │                   │ Process B   │                    │
│   │             │  ┌────────────┐   │             │                    │
│   │ write() ───┼─▶│  FIFO File  │──▶│ read()     │                    │
│   │             │  └────────────┘   │             │                    │
│   └─────────────┘                   └─────────────┘                    │
│                                                                          │
│   $ mkfifo /tmp/myfifo   # Create                                      │
│   $ ls -l /tmp/myfifo                                                   │
│   prw-r--r-- 1 user group 0 Jan 15 10:00 /tmp/myfifo                   │
│   # 'p' indicates pipe type                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

```c
// FIFO creation and usage
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#define FIFO_PATH "/tmp/myfifo"

// Writer process
void writer() {
    mkfifo(FIFO_PATH, 0666);

    int fd = open(FIFO_PATH, O_WRONLY);
    const char* message = "Hello via FIFO!";
    write(fd, message, strlen(message));
    close(fd);
}

// Reader process
void reader() {
    int fd = open(FIFO_PATH, O_RDONLY);
    char buffer[1024];
    ssize_t n = read(fd, buffer, sizeof(buffer));
    buffer[n] = '\0';
    printf("Received: %s\n", buffer);
    close(fd);
}
```

---

## 7. Shared Memory

### 7.1 POSIX Shared Memory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Shared Memory Structure                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Process A                         Process B                           │
│   ┌─────────────────┐               ┌─────────────────┐                │
│   │Virtual Addr Space│               │Virtual Addr Space│                │
│   │                 │               │                 │                │
│   │ ┌─────────────┐ │               │ ┌─────────────┐ │                │
│   │ │    Code     │ │               │ │    Code     │ │                │
│   │ ├─────────────┤ │               │ ├─────────────┤ │                │
│   │ │   Data      │ │               │ │   Data      │ │                │
│   │ ├─────────────┤ │               │ ├─────────────┤ │                │
│   │ │Shared Memory│◀┼───────────────┼▶│Shared Memory│ │                │
│   │ │ (0x7000)    │ │      │        │ │ (0x9000)    │ │                │
│   │ ├─────────────┤ │      │        │ ├─────────────┤ │                │
│   │ │    Heap     │ │      │        │ │    Heap     │ │                │
│   │ └─────────────┘ │      │        │ └─────────────┘ │                │
│   └─────────────────┘      │        └─────────────────┘                │
│                            │                                            │
│                            ▼                                            │
│                  ┌─────────────────────┐                               │
│                  │   Physical Memory   │                               │
│                  │                     │                               │
│                  │    [Shared Region]  │                               │
│                  │   Frames 100-110    │                               │
│                  └─────────────────────┘                               │
│                                                                          │
│   Same physical memory mapped to different virtual addresses            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

```c
// POSIX shared memory example
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>

#define SHM_NAME "/my_shm"
#define SHM_SIZE 4096

typedef struct {
    int counter;
    char message[256];
} SharedData;

int main() {
    // Create shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, SHM_SIZE);

    // Map memory
    SharedData* shared = mmap(NULL, SHM_SIZE,
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, 0);

    // Initialize
    shared->counter = 0;
    strcpy(shared->message, "Hello!");

    pid_t pid = fork();

    if (pid == 0) {
        // Child: read/modify shared memory
        printf("Child reads: counter=%d, message=%s\n",
               shared->counter, shared->message);

        shared->counter = 100;
        strcpy(shared->message, "Modified by child");

        munmap(shared, SHM_SIZE);
        exit(0);
    } else {
        // Parent: wait then check
        wait(NULL);

        printf("Parent reads: counter=%d, message=%s\n",
               shared->counter, shared->message);

        // Cleanup
        munmap(shared, SHM_SIZE);
        shm_unlink(SHM_NAME);
    }

    return 0;
}
```

### 7.2 Synchronization Need

> **Why synchronization is essential**: Shared memory is the fastest IPC mechanism because data is never copied -- both processes read and write the same physical memory region with no kernel involvement per access. However, this speed comes at a cost: since two processes can write to the same memory location simultaneously, you get data races without explicit synchronization. Unlike pipes or message queues where the kernel serializes access, shared memory puts the responsibility entirely on the programmer to use mutexes, semaphores, or other synchronization primitives.

```c
// Shared memory with semaphore synchronization
#include <semaphore.h>

typedef struct {
    sem_t mutex;       // Mutual exclusion
    sem_t items;       // Item count
    sem_t spaces;      // Empty space count
    int buffer[10];
    int in, out;
} SharedBuffer;

// Producer
void producer(SharedBuffer* sb, int item) {
    sem_wait(&sb->spaces);  // Wait for empty space
    sem_wait(&sb->mutex);   // Critical section

    sb->buffer[sb->in] = item;
    sb->in = (sb->in + 1) % 10;

    sem_post(&sb->mutex);   // Release critical section
    sem_post(&sb->items);   // Signal item added
}

// Consumer
int consumer(SharedBuffer* sb) {
    sem_wait(&sb->items);   // Wait for item
    sem_wait(&sb->mutex);   // Critical section

    int item = sb->buffer[sb->out];
    sb->out = (sb->out + 1) % 10;

    sem_post(&sb->mutex);   // Release critical section
    sem_post(&sb->spaces);  // Signal empty space added

    return item;
}
```

---

## 8. Message Queues and Sockets

### 8.1 POSIX Message Queue

```c
// Message queue example
#include <mqueue.h>
#include <stdio.h>
#include <string.h>

#define QUEUE_NAME "/my_queue"

typedef struct {
    long type;
    char text[256];
} Message;

// Sender
void sender() {
    struct mq_attr attr = {
        .mq_maxmsg = 10,
        .mq_msgsize = sizeof(Message)
    };

    mqd_t mq = mq_open(QUEUE_NAME, O_CREAT | O_WRONLY, 0666, &attr);

    Message msg;
    msg.type = 1;
    strcpy(msg.text, "Hello via Message Queue!");

    mq_send(mq, (char*)&msg, sizeof(msg), 0);
    mq_close(mq);
}

// Receiver
void receiver() {
    mqd_t mq = mq_open(QUEUE_NAME, O_RDONLY);

    Message msg;
    mq_receive(mq, (char*)&msg, sizeof(msg), NULL);

    printf("Received: %s\n", msg.text);

    mq_close(mq);
    mq_unlink(QUEUE_NAME);
}
```

### 8.2 Socket Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Socket Communication                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Server                                   Client                        │
│                                                                          │
│   socket() ─────────────────────────────── socket()                     │
│      │                                         │                        │
│      ▼                                         │                        │
│   bind()                                       │                        │
│      │                                         │                        │
│      ▼                                         │                        │
│   listen()                                     │                        │
│      │                                         │                        │
│      ▼                                         ▼                        │
│   accept() ◀───────── Connection req ──────── connect()                 │
│      │                                         │                        │
│      │       ┌──────────────────────────┐     │                        │
│      ▼       │      Data Exchange        │     ▼                        │
│   read() ◀───│─────────────────────────▶│───write()                    │
│   write()────│─────────────────────────▶│───read()                     │
│      │       └──────────────────────────┘     │                        │
│      ▼                                         ▼                        │
│   close() ─────────────────────────────── close()                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

```c
// TCP server example
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080

int main() {
    int server_fd, client_fd;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);
    char buffer[1024] = {0};

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind
    bind(server_fd, (struct sockaddr*)&address, sizeof(address));

    // Listen
    listen(server_fd, 3);
    printf("Server listening on port %d\n", PORT);

    // Accept client connection
    client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);

    // Receive data
    read(client_fd, buffer, sizeof(buffer));
    printf("Received: %s\n", buffer);

    // Send response
    send(client_fd, "Hello from server", 17, 0);

    close(client_fd);
    close(server_fd);
    return 0;
}
```

---

## Practice Problems

### Problem 1: I/O Method Comparison
Suggest appropriate use cases for polling, interrupts, and DMA.

<details>
<summary>Show Answer</summary>

```
1. Polling:
   - Very fast devices (high-speed network cards)
   - Very short response time
   - When avoiding interrupt overhead
   Example: 10Gbps network, low-latency trading systems

2. Interrupts:
   - Slow devices (keyboard, mouse)
   - Infrequent I/O
   - When CPU needs to do other work
   Example: General input devices, low-speed serial ports

3. DMA (Direct Memory Access):
   - Large data transfers
   - Block devices (disk, SSD)
   - High throughput required
   Example: Disk I/O, video capture, large network transfers
```

</details>

### Problem 2: Pipe Communication
Explain the internal operation of the following shell command from a pipe perspective.

```bash
$ cat file.txt | grep "error" | wc -l
```

<details>
<summary>Show Answer</summary>

```
1. Shell creates two pipes:
   pipe1: cat → grep
   pipe2: grep → wc

2. Creates three child processes:

   Process 1 (cat):
   - Redirect stdout to pipe1[1]
   - exec("cat", "file.txt")
   - Write file content to pipe1

   Process 2 (grep):
   - Redirect stdin to pipe1[0]
   - Redirect stdout to pipe2[1]
   - exec("grep", "error")
   - Read from pipe1, filter "error", write to pipe2

   Process 3 (wc):
   - Redirect stdin to pipe2[0]
   - exec("wc", "-l")
   - Read from pipe2, count lines

3. Data flow:
   file.txt → cat → pipe1 → grep → pipe2 → wc → stdout
```

</details>

### Problem 3: Shared Memory vs Message Passing
Compare advantages and disadvantages of shared memory and message queues when implementing producer-consumer problem.

<details>
<summary>Show Answer</summary>

```
Shared Memory:

Advantages:
- Fastest IPC (no kernel involvement)
- Efficient for large data
- Flexible data structures

Disadvantages:
- Must implement synchronization directly (semaphores, mutexes)
- Must handle data race conditions
- Single system only
- Complex memory management

Message Queue:

Advantages:
- Built-in synchronization (OS handles)
- Structured message passing
- Priority support
- Clear message boundaries

Disadvantages:
- Data copy overhead
- Message size limits
- Slower than shared memory

Selection criteria:
- Large/high-speed: Shared memory
- Simplicity/safety: Message queue
- Distributed system: Sockets or network message queue
```

</details>

### Problem 4: DMA Calculation
Compare CPU usage time for DMA vs PIO (Programmed I/O) when reading 1MB file from disk.

- Block size: 512 bytes
- PIO: 100 CPU cycles per block
- DMA: 1000 cycles setup, 500 cycles interrupt
- CPU clock: 1GHz

<details>
<summary>Show Answer</summary>

```
File size: 1MB = 1,048,576 bytes
Block count: 1,048,576 / 512 = 2,048 blocks

PIO method:
- CPU cycles = 2,048 × 100 = 204,800 cycles
- CPU time = 204,800 / 1,000,000,000 = 0.2048 ms

DMA method:
- Setup: 1,000 cycles
- Completion interrupt: 500 cycles
- Total CPU cycles = 1,000 + 500 = 1,500 cycles
- CPU time = 1,500 / 1,000,000,000 = 0.0015 ms

Comparison:
- PIO: 0.2048 ms
- DMA: 0.0015 ms
- DMA is ~136× more efficient

Additional consideration:
- DMA has setup overhead, inefficient for very small transfers
- Break-even point: 1500 / 100 = 15 blocks = 7.5 KB
- DMA advantageous for transfers > 7.5 KB
```

</details>

### Problem 5: Socket Programming
Explain differences between TCP and UDP sockets and suggest suitable applications for each.

<details>
<summary>Show Answer</summary>

```
TCP (Transmission Control Protocol):

Features:
- Connection-oriented (3-way handshake)
- Reliability guaranteed (ordering, retransmission)
- Flow control, congestion control
- Byte stream

Suitable applications:
- Web (HTTP/HTTPS)
- Email (SMTP, IMAP)
- File transfer (FTP, SCP)
- Database connections
- SSH

UDP (User Datagram Protocol):

Features:
- Connectionless
- No reliability (loss possible)
- No ordering guarantee
- Datagram-based
- Low overhead

Suitable applications:
- Real-time streaming (video, audio)
- Online games
- DNS queries
- VoIP
- IoT sensor data

Selection criteria:
- Reliability required: TCP
- Speed/low latency priority: UDP
- Some loss acceptable: UDP
- Accurate delivery needed: TCP
```

</details>

---

## Next Steps

You've completed the OS Theory learning materials! Recommended next learning paths:

### Advanced Learning
- **[Linux](../Linux/)**: Apply learned concepts in actual OS
  - Process management: `ps`, `top`, `kill`
  - File systems: `mount`, `df`, `du`
  - Networking: `netstat`, `ss`, `iptables`

### Related Fields
- **[Computer_Architecture](../Computer_Architecture/)**: Hardware perspective
  - Memory hierarchy
  - Cache memory
  - I/O systems

### Practical Projects
- Mini shell implementation (process creation, pipes)
- Memory allocator implementation
- File system simulator
- Scheduler simulator

---

## References

- Silberschatz, "Operating System Concepts" Chapters 12-13
- Stevens, "Advanced Programming in the UNIX Environment"
- Linux man pages: `pipe(2)`, `mmap(2)`, `socket(2)`, `shm_open(3)`
- Linux kernel documentation: https://www.kernel.org/doc/html/latest/
- Tanenbaum, "Modern Operating Systems" Chapters 5-6
