[Previous: OS Security](./25_OS_Security.md)

---

# 26. Microkernel Design

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare monolithic and microkernel architectures with concrete examples
2. Explain L4 and seL4 microkernel designs and their IPC mechanisms
3. Describe capability-based security and its formal verification
4. Analyze the performance tradeoffs of microkernels vs monolithic kernels
5. Evaluate hybrid kernel designs in modern operating systems

---

## Table of Contents

1. [Monolithic vs Microkernel](#1-monolithic-vs-microkernel)
2. [Microkernel Principles](#2-microkernel-principles)
3. [L4 Microkernel Family](#3-l4-microkernel-family)
4. [seL4: Formally Verified Kernel](#4-sel4-formally-verified-kernel)
5. [Capability-Based Security](#5-capability-based-security)
6. [IPC Performance](#6-ipc-performance)
7. [Hybrid Approaches](#7-hybrid-approaches)
8. [Exercises](#8-exercises)

---

## 1. Monolithic vs Microkernel

### 1.1 Architecture Comparison

```
Monolithic Kernel (Linux, Windows NT kernel):
  ┌──────────────────────────────────────┐
  │              User Space               │
  │   Applications, Libraries, Daemons    │
  ├══════════════════════════════════════╡ ← Syscall boundary
  │              Kernel Space              │
  │  Process Mgmt │ Memory Mgmt │ VFS    │
  │  Networking   │ Device Drivers        │
  │  File Systems │ Security │ Crypto    │
  │  ALL in one address space             │
  └──────────────────────────────────────┘

  + Fast (no IPC for kernel services)
  + Simple communication between components
  - Any driver bug can crash entire kernel
  - Large trusted computing base (TCB)

Microkernel (L4, seL4, Minix 3):
  ┌────────────────────────────────────────┐
  │              User Space                 │
  │  Apps │ FS Server │ Net Server │ Drivers│
  │       │  (user)   │  (user)    │ (user) │
  ├══════════════════════════════════════════╡
  │         Microkernel (~10K lines)        │
  │  IPC │ Scheduling │ Memory (basic)     │
  └────────────────────────────────────────┘

  + Small TCB → easier to verify, more secure
  + Driver crash doesn't crash kernel
  + Components can be restarted independently
  - IPC overhead for all kernel service calls
  - More complex system design
```

### 1.2 Code Size Comparison

```
Kernel code sizes (approximate):

Linux kernel:      ~30 million lines of code
Windows NT kernel: ~50 million lines
seL4 microkernel:  ~10,000 lines of C
L4 microkernel:    ~15,000 lines
Minix 3 kernel:    ~6,000 lines

Rule of thumb: 1-25 bugs per 1,000 lines of code
  Linux: ~30,000 - 750,000 potential bugs in kernel
  seL4: Mathematically PROVEN to have zero bugs
        (in the functional specification)
```

---

## 2. Microkernel Principles

### 2.1 Minimality Principle

```
A microkernel should provide ONLY mechanisms that MUST be in kernel:

1. IPC (Inter-Process Communication):
   Must be in kernel because it crosses address spaces.

2. Thread/Process Management:
   Must be in kernel because it requires privilege to context-switch.

3. Address Space Management:
   Must be in kernel because it manages the MMU.

Everything else is a USER-SPACE server:
  - File systems → FS server process
  - Network stack → Network server process
  - Device drivers → Driver processes
  - Memory policy → Pager processes
```

### 2.2 IPC as the Foundation

```c
/*
 * In a microkernel, IPC is the critical operation.
 * Everything goes through IPC:
 *
 * App wants to read a file:
 *   App ──IPC──▶ VFS Server ──IPC──▶ FS Server ──IPC──▶ Disk Driver
 *                                                            │
 *   App ◀──IPC── VFS Server ◀──IPC── FS Server ◀──IPC──────┘
 *
 * In monolithic kernel, this is just:
 *   App ──syscall──▶ VFS → FS → Driver → VFS ──return──▶ App
 *   (All in kernel space, no IPC needed)
 *
 * Microkernel IPC must be FAST to be competitive.
 * L4 IPC: ~100 ns (vs ~50 ns for a syscall on Linux)
 */

/* Pseudocode for L4-style synchronous IPC */
typedef struct {
    unsigned long sender;
    unsigned long msg[64];  /* Message registers */
} ipc_message_t;

/* Send message and wait for reply (call) */
int ipc_call(unsigned long dest, ipc_message_t *msg) {
    /* Transfers control directly to destination thread */
    /* Kernel only mediates: validates capability, switches */
    return 0;  /* Simplified */
}

/* Wait for any message (receive) */
int ipc_recv(ipc_message_t *msg) {
    /* Block until a message arrives */
    return 0;  /* Simplified */
}
```

---

## 3. L4 Microkernel Family

### 3.1 L4 History

```
L4 Family Tree:

L4 (1993, Jochen Liedtke):
  ├── Original hand-written assembly implementation
  ├── Proved that microkernel IPC can be fast
  └── ~100 cycles for IPC (was ~1000 in Mach)

L4Ka::Pistachio (2001, University of Karlsruhe):
  ├── Portable C++ implementation
  └── Multi-architecture support

NICTA/UNSW L4 → OKL4 (commercial, used in billions of phones):
  └── Qualcomm modem firmware runs on OKL4

seL4 (2009, NICTA/UNSW):
  ├── Formally verified (mathematical proof of correctness)
  ├── Capability-based security
  └── Used in military, automotive, aerospace
```

---

## 4. seL4: Formally Verified Kernel

### 4.1 What Formal Verification Means

```
seL4's verification proves:

1. Functional Correctness:
   The C implementation correctly implements the abstract specification.
   "The code does what the spec says."

2. Integrity:
   Memory isolation is not violated.
   "Process A cannot access Process B's memory."

3. Confidentiality:
   Information doesn't leak through covert channels (with some limits).

4. Worst-Case Execution Time:
   Bounded execution time for all kernel operations.
   "Every syscall completes within X microseconds."

The proof chain:
  Abstract Spec (Haskell)
       ↕ proved equivalent
  Executable Spec (Haskell)
       ↕ proved equivalent
  C Implementation
       ↕ proved equivalent
  Binary (via verified compiler or translation validation)
```

### 4.2 seL4 Capabilities

```
seL4 Capability System:

Everything is controlled by capabilities (unforgeable tokens):
  - Thread capabilities: create/manage threads
  - Memory capabilities: map/unmap pages
  - Endpoint capabilities: send/receive IPC messages
  - Notification capabilities: async signaling
  - IRQ capabilities: handle interrupts

Capability operations:
  seL4_Call(ep_cap, msg)     Send message, wait for reply
  seL4_Send(ep_cap, msg)     Send message, don't wait
  seL4_Recv(ep_cap, msg)     Wait for message
  seL4_Yield()               Voluntarily yield CPU

Security model:
  If you don't have a capability, you CAN'T do the operation.
  Capabilities can be delegated but not forged.
  The kernel mediates all capability operations.
```

---

## 5. Capability-Based Security

### 5.1 Capabilities vs Access Control Lists

```
ACL (traditional):
  File: /etc/shadow
  ACL:  root:rw, shadow-group:r
  Check: Is the requesting process's UID/GID in the ACL?

  Problem: "Confused deputy" attack
  Program with UID root accidentally opens /etc/shadow
  when it only needed /tmp/log

Capabilities:
  Process has explicit capability to access each resource.
  open(/etc/shadow) requires FILE_CAP for /etc/shadow.
  If process wasn't given that capability, access denied.

  Solves confused deputy: process only has capabilities
  it was explicitly given, nothing more.
```

### 5.2 Capability Implementation

```c
/*
 * Simplified capability-based file system.
 */

typedef struct {
    int type;           /* CAPABILITY_FILE, CAPABILITY_DIR, etc. */
    int object_id;      /* Which object this capability refers to */
    int permissions;    /* READ, WRITE, EXECUTE, etc. */
    int delegatable;    /* Can this capability be shared? */
} capability_t;

typedef struct {
    capability_t caps[256];
    int n_caps;
} capability_space_t;

int cap_check(capability_space_t *cspace, int object_id, int perm) {
    for (int i = 0; i < cspace->n_caps; i++) {
        if (cspace->caps[i].object_id == object_id &&
            (cspace->caps[i].permissions & perm) == perm) {
            return 1;  /* Authorized */
        }
    }
    return 0;  /* Denied */
}

int cap_delegate(capability_space_t *from, capability_space_t *to,
                 int cap_index, int new_perms) {
    capability_t *cap = &from->caps[cap_index];

    if (!cap->delegatable) return -1;
    if ((new_perms & cap->permissions) != new_perms) return -1;

    /* Create new capability with equal or fewer permissions */
    to->caps[to->n_caps] = *cap;
    to->caps[to->n_caps].permissions = new_perms;
    to->n_caps++;

    return 0;
}
```

---

## 6. IPC Performance

### 6.1 IPC Optimization Techniques

```
Making microkernel IPC fast:

1. Direct Process Switch:
   Sender's timeslice transfers to receiver.
   No scheduler invocation needed!

2. Register-Based Messages:
   Short messages passed in CPU registers.
   No memory copies for small messages.

3. Lazy Scheduling:
   Don't update scheduler data structures for every IPC.
   Only update when truly needed.

4. Kernel Entry/Exit Optimization:
   Minimize register saves/restores.
   Use hardware support (SYSENTER/SYSCALL).

L4 IPC Benchmark Results:
  Platform          | IPC latency
  ------------------|-------------
  x86 (Pentium)     | ~100 cycles
  ARM (Cortex-A9)   | ~150 cycles
  x86-64 (modern)   | ~200 cycles

  For comparison:
  Linux syscall:      ~150-300 cycles
  Linux pipe:         ~3000-5000 cycles
```

---

## 7. Hybrid Approaches

### 7.1 Real-World Hybrid Designs

```
Few systems are purely monolithic or microkernel:

macOS/iOS (XNU):
  Mach microkernel + BSD monolithic layer
  IPC used for some services, direct calls for others

Windows NT:
  Hybrid: kernel-mode drivers but microkernel-like subsystems
  Win32, POSIX subsystems as user-mode servers (originally)

Linux + KVM:
  Monolithic kernel acts as hypervisor
  Can host microkernels as guests

QNX:
  True microkernel for automotive/medical
  Drivers in user space, POSIX compatible

Fuchsia (Google):
  Zircon microkernel
  Capability-based security
  Target: phones, IoT, laptops
```

---

## 8. Exercises

### Exercise 1: Microkernel IPC Simulator

Simulate microkernel IPC:
1. Implement a simple synchronous IPC mechanism using pipes/sockets
2. Create "server" processes for: file system, memory manager, device driver
3. Route all operations through IPC (no direct function calls)
4. Measure IPC overhead compared to direct function calls
5. Implement message batching and measure improvement

### Exercise 2: Capability System

Build a capability-based access control system:
1. Define capability structure: object ID, permissions, delegatable
2. Implement: create, check, delegate, revoke operations
3. Create scenario: web server with minimal capabilities
4. Show that confused deputy attack is prevented
5. Compare with traditional ACL: code complexity and security

### Exercise 3: Monolithic vs Microkernel Tradeoff Analysis

Analyze the architecture tradeoff:
1. Implement a simple service as: (a) library call, (b) IPC-based server
2. Measure latency and throughput for both
3. Simulate a "driver crash" - show microkernel recovers, monolithic doesn't
4. Calculate: at what IPC cost does microkernel become unacceptable?
5. Write analysis: when to choose microkernel vs monolithic

### Exercise 4: User-Space Device Driver

Write a device driver in user space:
1. Create a virtual "device" that produces data
2. Write driver as a user-space process communicating via IPC
3. Write a kernel-space version for comparison
4. Measure: latency, throughput, reliability
5. Simulate driver crash and automatic restart

### Exercise 5: seL4 Capability Exploration

Explore seL4 concepts (in simulation):
1. Implement a capability space with create/delete/invoke operations
2. Model seL4 endpoints and notifications
3. Build a simple 3-process system: client, server, resource manager
4. Demonstrate: delegation chain and capability revocation
5. Prove (informally): that capabilities cannot be forged

---

*End of Lesson 26*
