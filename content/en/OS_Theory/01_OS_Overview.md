# Operating System Overview

**Next**: [Process Concepts](./02_Process_Concepts.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what an operating system does and why it is necessary
2. Distinguish between the resource manager and service provider perspectives of an OS
3. Compare monolithic, microkernel, and hybrid kernel architectures and identify real-world examples of each
4. Describe the system call mechanism, including the transition from user mode to kernel mode
5. Explain how interrupt handling works, from signal detection through ISR execution to state restoration
6. Trace the historical evolution of operating systems from batch processing to modern time-sharing systems

---

Every app you use -- your browser, your IDE, your music player -- relies on the OS to manage memory, schedule CPU time, and talk to hardware. Without an OS, every programmer would need to write their own disk driver and memory allocator from scratch. This lesson covers the definition of operating systems, their roles, development history, and kernel architecture -- the foundational concepts that everything else in OS theory builds upon.

## Table of Contents

1. [What is an Operating System?](#1-what-is-an-operating-system)
2. [Roles of an Operating System](#2-roles-of-an-operating-system)
3. [History of Operating Systems](#3-history-of-operating-systems)
4. [Kernel Architecture](#4-kernel-architecture)
5. [System Calls](#5-system-calls)
6. [Interrupt Handling](#6-interrupt-handling)
7. [Practice Problems](#7-practice-problems)

---

## 1. What is an Operating System?

### Definition

```
Operating System = Intermediary between hardware and users

┌─────────────────────────────────────────┐
│         Users/Application Programs       │
├─────────────────────────────────────────┤
│          Operating System (OS)           │
│  - Resource Manager                      │
│  - Control Program                       │
├─────────────────────────────────────────┤
│              Hardware                    │
│    CPU, Memory, Disk, I/O Devices        │
└─────────────────────────────────────────┘
```

### Two Perspectives of OS

```
┌────────────────────────────────────────────────────┐
│              Perspective 1: Resource Manager        │
├────────────────────────────────────────────────────┤
│  - CPU time distribution                           │
│  - Memory space allocation                         │
│  - Disk/I/O device management                      │
│  - Resolve resource conflicts between programs     │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│              Perspective 2: Service Provider        │
├────────────────────────────────────────────────────┤
│  - File system access                              │
│  - Process execution environment                   │
│  - User interface (CLI/GUI)                        │
│  - Network communication                           │
└────────────────────────────────────────────────────┘
```

---

## 2. Roles of an Operating System

### Core Functions

```
┌───────────────────────────────────────────────────────┐
│                OS Core Functions                       │
├───────────────┬───────────────────────────────────────┤
│ Process Mgmt  │ Process creation/termination/scheduling│
├───────────────┼───────────────────────────────────────┤
│ Memory Mgmt   │ Memory allocation/deallocation, virtual│
├───────────────┼───────────────────────────────────────┤
│ File System   │ File create/delete/read/write          │
├───────────────┼───────────────────────────────────────┤
│ I/O Mgmt      │ Device drivers, buffering, spooling    │
├───────────────┼───────────────────────────────────────┤
│ Security      │ Access control, user authentication    │
├───────────────┼───────────────────────────────────────┤
│ Networking    │ Protocol stack, socket interface       │
└───────────────┴───────────────────────────────────────┘
```

### Resource Management Example

```c
// Process reading a file
// (How OS mediates)

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    // 1. Open file (request to OS)
    int fd = open("data.txt", O_RDONLY);

    // 2. Read file (OS handles disk I/O)
    char buffer[1024];
    ssize_t bytes = read(fd, buffer, sizeof(buffer));

    // 3. Close file (OS releases resource)
    close(fd);

    return 0;
}

/*
User Program         Operating System      Hardware
     │                    │                  │
     │── open() ──────────▶                  │
     │                    │── Disk access ───▶
     │                    │◀── Data ─────────│
     │◀── fd returned ────│                  │
*/
```

---

## 3. History of Operating Systems

### Evolution by Generation

```
┌─────────────────────────────────────────────────────────────┐
│              Operating System Evolution History              │
├───────────┬─────────────────────────────────────────────────┤
│   Era      │                  Features                       │
├───────────┼─────────────────────────────────────────────────┤
│ 1st Gen    │ No OS, manual operation                         │
│ 1940-50s   │ Programmers enter machine code directly         │
├───────────┼─────────────────────────────────────────────────┤
│ 2nd Gen    │ Batch System                                    │
│ 1950-60s   │ Job automation, resident monitor                │
├───────────┼─────────────────────────────────────────────────┤
│ 3rd Gen    │ Multiprogramming, time-sharing                  │
│ 1960-70s   │ Multiple programs execute concurrently,         │
│            │ interactive usage                                │
├───────────┼─────────────────────────────────────────────────┤
│ 4th Gen    │ Distributed systems, network OS                 │
│ 1970-Now   │ GUI, multiprocessor, cloud                      │
└───────────┴─────────────────────────────────────────────────┘
```

### Batch System

```
┌───────────────────────────────────────────────────────┐
│              Batch System Operation                    │
├───────────────────────────────────────────────────────┤
│                                                       │
│   Job1 ──▶ Job2 ──▶ Job3 ──▶ Job4 ──▶ ...           │
│                                                       │
│   • Only one job executes at a time                   │
│   • CPU idle time occurs (waiting for I/O)           │
│   • Improved efficiency with automatic job switching  │
│                                                       │
└───────────────────────────────────────────────────────┘

Timeline:
┌──────┬──────────────────┬──────┬──────────────────┐
│Job1  │   I/O wait (waste)│Job2  │   I/O wait       │
└──────┴──────────────────┴──────┴──────────────────┘
   CPU        CPU idle         CPU        CPU idle
```

### Multiprogramming

```
┌───────────────────────────────────────────────────────┐
│            Multiprogramming Operation                  │
├───────────────────────────────────────────────────────┤
│                                                       │
│   Multiple programs loaded in memory simultaneously   │
│                                                       │
│   ┌─────────┐                                         │
│   │  OS     │                                         │
│   ├─────────┤                                         │
│   │ Program1│ ◀── CPU executing                       │
│   ├─────────┤                                         │
│   │ Program2│ ◀── Waiting for I/O                     │
│   ├─────────┤                                         │
│   │ Program3│ ◀── Ready state                         │
│   └─────────┘                                         │
│                                                       │
└───────────────────────────────────────────────────────┘

Timeline:
┌──────┬──────┬──────┬──────┬──────┬──────┐
│Job1  │Job2  │Job3  │Job1  │Job3  │Job2  │
└──────┴──────┴──────┴──────┴──────┴──────┘
   CPU continuously performs different tasks
```

### Time-Sharing System

```
┌───────────────────────────────────────────────────────┐
│            Time-Sharing System Operation               │
├───────────────────────────────────────────────────────┤
│                                                       │
│   Multiple users use computer simultaneously          │
│   Each user allocated short time slice                │
│                                                       │
│   UserA ──┬── 10ms ──┬── 10ms ──┬── ...              │
│   UserB ──┼── 10ms ──┼── 10ms ──┼── ...              │
│   UserC ──┴── 10ms ──┴── 10ms ──┴── ...              │
│                                                       │
│   → Each user feels like they have exclusive access   │
│   → Response time important (interactive system)      │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## 4. Kernel Architecture

### What is a Kernel?

```
Kernel = Core part of the operating system
       = Directly interacts with hardware
       = Always resident in memory

┌─────────────────────────────────────────┐
│           User Space                     │
│  ┌───────┐ ┌───────┐ ┌───────┐          │
│  │ App1  │ │ App2  │ │ App3  │          │
│  └───────┘ └───────┘ └───────┘          │
├─────────────────────────────────────────┤
│           Kernel Space                   │
│  ┌───────────────────────────────────┐  │
│  │           Kernel                   │  │
│  │  • Process management              │  │
│  │  • Memory management               │  │
│  │  • File system                     │  │
│  │  • Device drivers                  │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│              Hardware                    │
└─────────────────────────────────────────┘
```

### Monolithic Kernel

```
┌───────────────────────────────────────────┐
│              Monolithic Kernel             │
├───────────────────────────────────────────┤
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │             Kernel                   │  │
│  │  ┌──────┬──────┬──────┬──────────┐  │  │
│  │  │Process│Memory │File   │ Network  │  │  │
│  │  │ Mgmt  │ Mgmt  │System │          │  │  │
│  │  ├──────┴──────┴──────┴──────────┤  │  │
│  │  │      Device Drivers            │  │  │
│  │  └────────────────────────────────┘  │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Advantages: Excellent performance        │
│  Disadvantages: Hard to maintain, bugs    │
│                 have wide impact          │
│  Examples: Linux, Unix, MS-DOS            │
│                                           │
└───────────────────────────────────────────┘
```

### Microkernel

```
┌───────────────────────────────────────────┐
│              Microkernel                   │
├───────────────────────────────────────────┤
│                                           │
│  User space:                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ │
│  │File   │ │Device │ │Network│ │Process   │ │
│  │Server │ │Driver │ │Server │ │Server    │ │
│  └──────┘ └──────┘ └──────┘ └──────────┘ │
│            ▲          ▲          ▲        │
│            │    IPC   │          │        │
│            ▼          ▼          ▼        │
│  ┌─────────────────────────────────────┐  │
│  │      Microkernel (minimal)          │  │
│  │   - Basic IPC                       │  │
│  │   - Basic scheduling                │  │
│  │   - Basic memory management         │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Advantages: Stability, easy maintenance  │
│  Disadvantages: Performance degradation   │
│                 due to IPC overhead       │
│  Examples: Minix, QNX, L4                 │
│                                           │
└───────────────────────────────────────────┘
```

### Hybrid Kernel

```
┌───────────────────────────────────────────┐
│              Hybrid Kernel                 │
├───────────────────────────────────────────┤
│                                           │
│  Combines advantages of micro + monolithic│
│                                           │
│  User space:                              │
│  ┌──────┐ ┌──────┐                        │
│  │ App  │ │Subsys│                        │
│  └──────┘ └──────┘                        │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │           Hybrid Kernel              │  │
│  │  ┌──────────────────────────────┐   │  │
│  │  │ FileSystem │ Network │ Graphics│   │  │
│  │  └──────────────────────────────┘   │  │
│  │  ┌──────────────────────────────┐   │  │
│  │  │       Microkernel Core        │   │  │
│  │  └──────────────────────────────┘   │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  Examples: Windows NT, macOS (XNU)        │
│                                           │
└───────────────────────────────────────────┘
```

### Kernel Architecture Comparison

```
┌─────────────┬──────────┬──────────┬──────────┐
│  Feature     │Monolithic│Microkernel│ Hybrid   │
├─────────────┼──────────┼──────────┼──────────┤
│ Performance │ High     │ Low      │ Medium   │
│ Stability   │ Low      │ High     │ Medium   │
│ Maintenance │ Hard     │ Easy     │ Medium   │
│ Modularity  │ Low      │ High     │ High     │
│ Code Size   │ Large    │ Small    │ Medium   │
├─────────────┼──────────┼──────────┼──────────┤
│ Examples    │ Linux    │ Minix    │ Windows  │
│             │ FreeBSD  │ QNX      │ macOS    │
└─────────────┴──────────┴──────────┴──────────┘
```

---

## 5. System Calls

### What is a System Call?

```
System Call = Interface for user programs to request OS services

┌───────────────────────────────────────────┐
│              User Mode                     │
│                                           │
│    ┌─────────────────────────────────┐    │
│    │      User Program                │    │
│    │                                 │    │
│    │   read(fd, buffer, size);       │    │
│    │          │                      │    │
│    │          ▼                      │    │
│    │   ┌──────────────────┐          │    │
│    │   │ System Call Interface│       │    │
│    │   │ (Library function)  │        │    │
│    │   └──────────────────┘          │    │
│    └─────────────│───────────────────┘    │
├──────────────────│────────────────────────┤
│              Kernel Mode (trap)            │
│                  ▼                        │
│    ┌─────────────────────────────────┐    │
│    │      System Call Handler         │    │
│    │                                 │    │
│    │   → Perform actual file read     │    │
│    │   → Return result to user        │    │
│    └─────────────────────────────────┘    │
└───────────────────────────────────────────┘
```

### System Call Processing Steps

```
Steps:
1. User program calls system call
2. Library sets parameters in registers
3. Software interrupt (trap) occurs
4. User mode → Kernel mode transition
5. Find handler using system call number
6. Execute handler
7. Kernel mode → User mode return
8. Return result

┌────────────────────────────────────────────────┐
│                System Call Table                │
├────────┬───────────────────────────────────────┤
│ Number │              System Call               │
├────────┼───────────────────────────────────────┤
│   0    │ read()                                │
│   1    │ write()                               │
│   2    │ open()                                │
│   3    │ close()                               │
│   ...  │ ...                                   │
│  39    │ fork()                                │
│  60    │ exit()                                │
└────────┴───────────────────────────────────────┘
```

### System Call Categories

```c
// 1. Process Control
fork();     // Create process
exec();     // Execute program
exit();     // Terminate process
wait();     // Wait for child process

// 2. File Management
open();     // Open file
read();     // Read file
write();    // Write file
close();    // Close file

// 3. Device Management
ioctl();    // Control device
read();     // Read from device
write();    // Write to device

// 4. Information Maintenance
getpid();   // Get process ID
time();     // Get time

// 5. Communication
socket();   // Create socket
send();     // Send data
recv();     // Receive data
```

### System Call Example

```c
// File writing example in Linux
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

int main() {
    const char *msg = "Hello, OS!\n";

    // System call 1: open()
    int fd = open("output.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        // System call: write() to stderr
        write(2, "Error opening file\n", 19);
        return 1;
    }

    // System call 2: write()
    ssize_t bytes_written = write(fd, msg, strlen(msg));

    // System call 3: close()
    close(fd);

    return 0;
}

/*
Trace system calls with strace:
$ strace ./a.out

openat(AT_FDCWD, "output.txt", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 3
write(3, "Hello, OS!\n", 11) = 11
close(3) = 0
*/
```

---

## 6. Interrupt Handling

### What is an Interrupt?

```
Interrupt = Signal that notifies CPU of event requiring attention

┌────────────────────────────────────────────────┐
│                 Interrupt Types                 │
├──────────────────┬─────────────────────────────┤
│ Hardware Interrupt│   Software Interrupt        │
├──────────────────┼─────────────────────────────┤
│ • Timer          │ • System call (trap)        │
│ • Keyboard input │ • Exception                 │
│ • Mouse movement │   - Division by zero        │
│ • Disk I/O done  │   - Invalid memory access   │
│ • Network packet │   - Page fault              │
└──────────────────┴─────────────────────────────┘
```

### Interrupt Handling Process

```
┌───────────────────────────────────────────────────────┐
│                  Interrupt Handling Process            │
└───────────────────────────────────────────────────────┘

1. CPU executing instructions
   │
   ▼
2. Interrupt occurs (e.g., keyboard input)
   │
   ▼
3. Save current state (PC, registers)
   │
   ▼
4. Find handler address in interrupt vector table
   │
   ▼
5. Execute Interrupt Service Routine (ISR)
   │
   ▼
6. Restore saved state
   │
   ▼
7. Resume interrupted program

┌─────────────────────────────────────────────┐
│            Interrupt Vector Table            │
├───────┬─────────────────────────────────────┤
│ Number│           Handler Address            │
├───────┼─────────────────────────────────────┤
│   0   │ 0x00001000 (Division error)         │
│   1   │ 0x00001100 (Debug)                  │
│   2   │ 0x00001200 (NMI)                    │
│  ...  │ ...                                 │
│  32   │ 0x00002000 (Timer)                  │
│  33   │ 0x00002100 (Keyboard)               │
└───────┴─────────────────────────────────────┘
```

### Interrupt Timeline

```
Time →

Process execution: ████████░░░░░░░░████████████████████
                      ↑     ↑
                   Interrupt ISR
                   occurs   done

Details:
┌────────┬────────┬────────────────┬────────────────┐
│ Instruction│ Save   │  Interrupt     │   Instruction  │
│ execution  │ state  │  handler exec  │   resume       │
└────────┴────────┴────────────────┴────────────────┘
```

### Interrupt Priority

```
High   ┌─────────────────────────┐
   ↑   │ Power Failure            │
   │   ├─────────────────────────┤
   │   │ Machine Check            │
   │   ├─────────────────────────┤
   │   │ External Interrupt (NMI) │
   │   ├─────────────────────────┤
   │   │ Timer Interrupt          │
   │   ├─────────────────────────┤
   │   │ I/O Interrupt            │
   │   ├─────────────────────────┤
   │   │ Software Interrupt       │
   ↓   └─────────────────────────┘
Low
```

### Interrupt Handler Example (Conceptual)

```c
// Interrupt Service Routine
// Actually written with assembly

// Timer interrupt handler (conceptual code)
void timer_interrupt_handler(void) {
    // 1. Decrease current process's time slice
    current_process->time_slice--;

    // 2. Update system time
    system_time++;

    // 3. Schedule if time slice is 0
    if (current_process->time_slice == 0) {
        schedule();  // Select next process
    }

    // 4. Signal interrupt completion
    send_EOI();  // End Of Interrupt
}

// Keyboard interrupt handler (conceptual code)
void keyboard_interrupt_handler(void) {
    // 1. Read scan code from keyboard controller
    uint8_t scancode = inb(KEYBOARD_DATA_PORT);

    // 2. Convert scan code to ASCII
    char key = scancode_to_ascii(scancode);

    // 3. Store in keyboard buffer
    keyboard_buffer_put(key);

    // 4. Wake up waiting processes
    wakeup_keyboard_waiters();

    // 5. Send EOI
    send_EOI();
}
```

---

## 7. Practice Problems

### Problem 1: Basic Concepts

Fill in the blanks.

1. The two perspectives of an operating system are ________ and service provider.
2. The goal of multiprogramming is to reduce ________ to increase CPU utilization.
3. The kernel structure where all functions are integrated is called ________.

<details>
<summary>Show Answer</summary>

1. Resource manager
2. CPU idle time
3. Monolithic kernel

</details>

### Problem 2: System Call Classification

Classify the following system calls into appropriate categories.

```
fork(), open(), socket(), getpid(), ioctl()
```

| Process Control | File Management | Device Management | Information Maintenance | Communication |
|----------------|----------------|------------------|----------------------|--------------|
|                |                |                  |                      |              |

<details>
<summary>Show Answer</summary>

| Process Control | File Management | Device Management | Information Maintenance | Communication |
|----------------|----------------|------------------|----------------------|--------------|
| fork() | open() | ioctl() | getpid() | socket() |

</details>

### Problem 3: Kernel Architecture Comparison

Choose the matching kernel architecture for each description.

A. Monolithic Kernel
B. Microkernel
C. Hybrid Kernel

1. ( ) Kernel contains only minimal functions, rest runs in user space
2. ( ) Architecture used by Linux
3. ( ) Architecture used by Windows NT
4. ( ) Performance degradation due to IPC overhead is a disadvantage

<details>
<summary>Show Answer</summary>

1. (B) Microkernel
2. (A) Monolithic Kernel
3. (C) Hybrid Kernel
4. (B) Microkernel

</details>

### Problem 4: Interrupt Handling

Arrange the interrupt handling process in the correct order.

A. Execute interrupt handler
B. Save current state
C. Interrupt occurs
D. Restore saved state
E. Find handler in interrupt vector table

<details>
<summary>Show Answer</summary>

C → B → E → A → D

1. (C) Interrupt occurs
2. (B) Save current state
3. (E) Find handler in interrupt vector table
4. (A) Execute interrupt handler
5. (D) Restore saved state

</details>

### Problem 5: System Call Code Analysis

List the system calls that occur in the following code in order.

```c
#include <unistd.h>
#include <fcntl.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) {
        int fd = open("child.txt", O_WRONLY | O_CREAT, 0644);
        write(fd, "child", 5);
        close(fd);
    } else {
        wait(NULL);
        write(1, "parent\n", 7);
    }

    return 0;
}
```

<details>
<summary>Show Answer</summary>

1. fork() - Create child process

Child process:
2. open() - Open file
3. write() - Write to file
4. close() - Close file
5. exit() - Terminate process (implicit)

Parent process:
6. wait() - Wait for child process
7. write() - Write to standard output
8. exit() - Terminate process (implicit)

</details>

---

## Hands-On Exercises

### Exercise 1: Explore System Calls

Use the `strace` (Linux) or `dtruss` (macOS) command to trace system calls made by a simple program.

```bash
# Linux
strace -c ls /tmp

# macOS
sudo dtruss ls /tmp 2>&1 | head -30
```

**Tasks:**
1. Identify which system call categories (process, file, device, information) appear in the trace
2. Count how many `open`/`openat` calls are made — why so many for a simple `ls`?
3. Compare the trace of `ls /tmp` vs `ls /nonexistent` — what differs?

### Exercise 2: Kernel Architecture Comparison

Research and fill in the following table for three real operating systems:

| Feature | Linux | Windows NT | QNX |
|---------|-------|------------|-----|
| Kernel type | ? | ? | ? |
| Scheduler in kernel? | ? | ? | ? |
| File system in kernel? | ? | ? | ? |
| Device drivers in kernel? | ? | ? | ? |
| IPC mechanism | ? | ? | ? |

### Exercise 3: Interrupt Latency Measurement

Write a Python script that measures timer interrupt granularity:

```python
import time

# Measure the actual resolution of time.sleep()
intervals = []
for _ in range(100):
    start = time.perf_counter()
    time.sleep(0.001)  # request 1ms sleep
    elapsed = time.perf_counter() - start
    intervals.append(elapsed * 1000)  # convert to ms

print(f"Requested: 1.000 ms")
print(f"Actual avg: {sum(intervals)/len(intervals):.3f} ms")
print(f"Min: {min(intervals):.3f} ms, Max: {max(intervals):.3f} ms")
```

**Tasks:**
1. Run the script and explain why the actual sleep time differs from the requested time
2. How does the OS timer interrupt frequency affect this result?
3. Try different sleep values (0.0001, 0.01, 0.1) and plot the error percentage

---

## Exercises

### Exercise 1: OS Role Identification

For each scenario below, identify which OS role (Resource Manager or Service Provider) is primarily at work and name the specific OS subsystem involved.

1. A video player requests 256 MB of RAM to buffer frames
2. A user types `ls` in the terminal and sees a directory listing
3. Two processes try to write to the same file simultaneously — one is blocked
4. A web browser opens a TCP socket to contact a remote server
5. A running program divides by zero and the OS terminates it

### Exercise 2: Kernel Architecture Trade-offs

The three major kernel designs each make different trade-offs. For each real operating system below, identify its kernel type and explain one concrete consequence of that choice:

| OS | Kernel Type | One Practical Consequence |
|----|-------------|--------------------------|
| Linux | ? | ? |
| macOS (XNU) | ? | ? |
| QNX | ? | ? |
| Windows NT | ? | ? |

Then answer: Why do most mainstream desktop OSes use hybrid or monolithic kernels rather than the theoretically cleaner microkernel design?

### Exercise 3: System Call Tracing

Write a minimal C program that performs the following sequence of operations and list every system call it makes, in order:

1. Create a new file named `hello.txt`
2. Write the string `"Hello, OS!"` to it
3. Close the file
4. Fork a child process
5. In the child: open and read back `hello.txt`, then exit
6. In the parent: wait for the child, then delete `hello.txt`

For each system call, state which category it belongs to (process control, file management, device management, information maintenance, or communication).

### Exercise 4: Interrupt Handling Deep Dive

Consider the following interrupt priority table:

```
Priority 1 (highest): Power failure
Priority 2: Machine check
Priority 3: Timer interrupt
Priority 4: Keyboard interrupt
Priority 5 (lowest): Software interrupt
```

Answer the following:

1. A keyboard interrupt occurs while the OS is handling a timer interrupt. What happens? Trace the full execution sequence.
2. Why is it important to save and restore CPU state (registers, PC) during interrupt handling? What would go wrong if the OS skipped the save step?
3. Describe what the timer interrupt handler does in a time-sharing OS. Why must it run frequently (e.g., every 10 ms)?

### Exercise 5: Design a Minimal OS

You are designing an OS for a single-function embedded device (e.g., a microwave controller). Answer each question with a justification:

1. **Kernel type**: Would you choose monolithic, microkernel, or hybrid? Why?
2. **Interrupts needed**: List the hardware interrupts this device would need to handle
3. **System calls needed**: Which categories of system calls are necessary, and which can be omitted entirely?
4. **Preemption**: Does this OS need preemptive scheduling? Why or why not?

---

## Next Steps

- [02_Process_Concepts.md](./02_Process_Concepts.md) - Process memory structure and state transitions

---

## References

- [Operating System Concepts (Silberschatz)](https://www.os-book.com/)
- [OSTEP - Introduction](https://pages.cs.wisc.edu/~remzi/OSTEP/intro.pdf)
- [Linux Kernel Documentation](https://www.kernel.org/doc/)
- [xv6 Operating System](https://pdos.csail.mit.edu/6.828/2021/xv6.html)
