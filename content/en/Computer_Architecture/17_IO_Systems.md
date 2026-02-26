# I/O Systems

**Previous**: [16_Virtual_Memory.md](./16_Virtual_Memory.md) | **Next**: [18_Parallel_Processing_Multicore.md](./18_Parallel_Processing_Multicore.md)

**Difficulty**: ⭐⭐⭐

**Prerequisites**: CPU architecture, memory systems

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the bus architecture connecting CPU, memory, and I/O devices
2. Compare programmed I/O, interrupt-driven I/O, and DMA
3. Explain how DMA transfers data without CPU involvement
4. Describe the evolution from shared buses to point-to-point interconnects (PCIe)
5. Explain memory-mapped I/O vs port-mapped I/O
6. Analyze I/O bottlenecks and their impact on system performance

---

A computer that cannot communicate with the outside world is useless. I/O systems connect the CPU to disks, networks, displays, and keyboards -- and their design determines whether your SSD feels fast or slow, whether your network card can keep up with gigabit traffic, and whether your GPU can feed frames to your display fast enough. I/O is often the true bottleneck in system performance.

## Table of Contents

1. [I/O System Overview](#1-io-system-overview)
2. [Programmed I/O (Polling)](#2-programmed-io-polling)
3. [Interrupt-Driven I/O](#3-interrupt-driven-io)
4. [DMA (Direct Memory Access)](#4-dma-direct-memory-access)
5. [Bus Architecture](#5-bus-architecture)
6. [I/O Interfaces](#6-io-interfaces)
7. [Modern I/O Systems](#7-modern-io-systems)
8. [Practice Problems](#8-practice-problems)

---

## 1. I/O System Overview

### 1.1 Diversity of I/O Devices

```
┌─────────────────────────────────────────────────────────────┐
│                  I/O Device Classification                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Devices        Output Devices         Storage Devices│
│  ┌──────────┐       ┌──────────┐          ┌──────────┐     │
│  │ Keyboard │       │ Monitor  │          │ HDD/SSD  │     │
│  │ Mouse    │       │ Printer  │          │ USB      │     │
│  │ Scanner  │       │ Speaker  │          │ SD Card  │     │
│  │ Microphone│      │ LED      │          │ Optical  │     │
│  │ Touch    │       │          │          │ Drive    │     │
│  └──────────┘       └──────────┘          └──────────┘     │
│                                                             │
│  Communication Devices                                      │
│  ┌────────────────────────────────────────────────────────┐│
│  │ Network Card (NIC), WiFi, Bluetooth, USB Hub          ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘

I/O Device Characteristics:
┌─────────────────┬──────────────┬───────────────────────────┐
│     Device      │   Data Rate  │      Characteristics      │
├─────────────────┼──────────────┼───────────────────────────┤
│ Keyboard        │  ~100 B/s    │ Slow, async, character    │
├─────────────────┼──────────────┼───────────────────────────┤
│ Mouse           │  ~1 KB/s     │ Slow, sync, event-based   │
├─────────────────┼──────────────┼───────────────────────────┤
│ Gigabit Ethernet│  125 MB/s    │ High-speed, packet-based  │
├─────────────────┼──────────────┼───────────────────────────┤
│ SATA SSD        │  ~600 MB/s   │ High-speed, block-based   │
├─────────────────┼──────────────┼───────────────────────────┤
│ NVMe SSD        │  ~7 GB/s     │ Ultra-fast, parallel      │
├─────────────────┼──────────────┼───────────────────────────┤
│ 4K Display      │  ~20 GB/s    │ Ultra-fast, streaming     │
└─────────────────┴──────────────┴───────────────────────────┘
```

### 1.2 I/O System Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    I/O System Layers                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Application                             │   │
│   │         read(), write(), printf()                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │         Operating System I/O Subsystem              │   │
│   │      Buffering, Caching, Spooling, Scheduling       │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Device Driver                           │   │
│   │    Device-specific control code, interrupt handling │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Hardware Controller                     │   │
│   │        I/O ports, registers, bus interface          │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │               I/O Device                             │   │
│   │          Physical device (disk, keyboard, etc.)     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 I/O Control Method Comparison

```
┌───────────────────┬────────────────────┬────────────────────┐
│                   │  Programmed I/O    │   Interrupt I/O    │
│    Characteristic │    (Polling)       │                    │
├───────────────────┼────────────────────┼────────────────────┤
│ CPU Involvement   │      High          │       Medium       │
├───────────────────┼────────────────────┼────────────────────┤
│ CPU Efficiency    │      Low           │       High         │
├───────────────────┼────────────────────┼────────────────────┤
│ Implementation    │      Low           │       Medium       │
│   Complexity      │                    │                    │
├───────────────────┼────────────────────┼────────────────────┤
│ Suitable Devices  │ Fast, predictable  │  Slow, async       │
├───────────────────┼────────────────────┼────────────────────┤
│ Data Transfer     │  Via CPU           │   Via CPU          │
└───────────────────┴────────────────────┴────────────────────┘

┌───────────────────┬────────────────────┐
│                   │       DMA          │
│    Characteristic │                    │
├───────────────────┼────────────────────┤
│ CPU Involvement   │       Low          │
├───────────────────┼────────────────────┤
│ CPU Efficiency    │     Very High      │
├───────────────────┼────────────────────┤
│ Implementation    │       High         │
│   Complexity      │                    │
├───────────────────┼────────────────────┤
│ Suitable Devices  │   Bulk transfer    │
├───────────────────┼────────────────────┤
│ Data Transfer     │   Direct memory    │
└───────────────────┴────────────────────┘
```

---

## 2. Programmed I/O (Polling)

### 2.1 Polling Concept

```
Definition: CPU periodically checks I/O device status for data transfer

┌─────────────────────────────────────────────────────────────┐
│                    Polling Operation                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│      CPU                           I/O Device               │
│       │                               │                     │
│       │  1. Check status (ready?)     │                     │
│       │──────────────────────────────▶│                     │
│       │                               │                     │
│       │  2. "Not ready"                │                     │
│       │◀──────────────────────────────│                     │
│       │                               │                     │
│       │  3. Check status (again)       │                     │
│       │──────────────────────────────▶│                     │
│       │                               │                     │
│       │  4. "Not ready"                │                     │
│       │◀──────────────────────────────│                     │
│       │         ...repeat...           │                     │
│       │  N. Check status               │                     │
│       │──────────────────────────────▶│                     │
│       │                               │                     │
│       │  N+1. "Ready"                  │                     │
│       │◀──────────────────────────────│                     │
│       │                               │                     │
│       │  N+2. Read data                │                     │
│       │◀──────────────────────────────│                     │
│       │                               │                     │
│                                                             │
│   Problem: CPU waits and does nothing (Busy Waiting)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 I/O Ports and Registers

```
I/O Device Controller Registers:

┌─────────────────────────────────────────────────────────────┐
│              I/O Controller Registers                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Status Register                                     │    │
│  │ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  │    │
│  │ │Busy │Ready│Error│ IRQ │ ... │ ... │ ... │ ... │  │    │
│  │ └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘  │    │
│  │ - Displays device status (read-only)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Control Register                                    │    │
│  │ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  │    │
│  │ │Start│ IE  │Mode │ Dir │ ... │ ... │ ... │ ... │  │    │
│  │ └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘  │    │
│  │ - Device control commands (write)                   │    │
│  │ - IE: Interrupt Enable                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Data Register                                       │    │
│  │ ┌───────────────────────────────────────────────┐   │    │
│  │ │                Data (8/16/32 bits)            │   │    │
│  │ └───────────────────────────────────────────────┘   │    │
│  │ - Actual transfer data                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Polling Programming Example

```c
// Simple polling-based I/O code

#define STATUS_REG  0x3F8  // Status register address
#define DATA_REG    0x3F9  // Data register address
#define READY_BIT   0x01   // Ready bit mask

// Character output (polling)
// Why polling is used here despite being wasteful: for fast devices like a UART
// running at the CPU's own speed, the ready bit flips almost immediately. The
// loop body executes only a few times, so the simplicity of polling (no interrupt
// setup, no context switch) outweighs the brief busy-wait cost.
void putchar_polling(char c) {
    // Wait until device is ready (Busy Wait)
    // Why we check READY_BIT in a loop: the UART transmit buffer may still hold
    // a previous character. Writing before it drains would overwrite and lose data.
    while ((inb(STATUS_REG) & READY_BIT) == 0) {
        // CPU keeps looping and checking
        // Unable to do anything else
    }

    // Transfer data
    outb(DATA_REG, c);
}

// String output
void print_string_polling(const char* str) {
    while (*str) {
        putchar_polling(*str++);
    }
}

// Character input (polling)
char getchar_polling(void) {
    // Wait until input data is available
    while ((inb(STATUS_REG) & READY_BIT) == 0) {
        // Busy Wait
    }

    return inb(DATA_REG);
}
```

### 2.4 Advantages and Disadvantages of Polling

```
Advantages:
┌─────────────────────────────────────────────────────────────┐
│ - Simple implementation                                     │
│ - Minimal hardware requirements                             │
│ - Predictable timing                                        │
│ - Efficient for fast devices (when data is ready instantly) │
│ - Minimizes jitter in real-time systems                     │
└─────────────────────────────────────────────────────────────┘

Disadvantages:
┌─────────────────────────────────────────────────────────────┐
│ - Wastes CPU time (Busy Waiting)                           │
│ - Very inefficient for slow devices                         │
│ - Difficult to handle multiple devices                      │
│ - Increased power consumption                               │
└─────────────────────────────────────────────────────────────┘

CPU Time Waste Calculation:
- Serial port: 115200 bps = 11520 characters/sec
- Per character: ~87us
- 3GHz CPU: 87us = 261,000 cycles
- CPU waits 260,000 cycles to transfer one character!
```

---

## 3. Interrupt-Driven I/O

### 3.1 Interrupt Concept

```
Definition: I/O device asynchronously signals CPU to request processing

┌─────────────────────────────────────────────────────────────┐
│                  Interrupt Operation                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│      CPU (performing tasks)                I/O Device       │
│       │                                      │              │
│       │  1. Issue I/O command                │              │
│       │─────────────────────────────────────▶│              │
│       │                                      │              │
│       │  2. Perform other tasks              │              │
│       │  (Don't wait for I/O completion)     │              │
│       │                                      │              │
│       │         ...time passes...            │ 3. Process I/O│
│       │                                      │              │
│       │  4. Interrupt signal (IRQ)           │              │
│       │◀═════════════════════════════════════│              │
│       │                                      │              │
│       │  5. Save current state               │              │
│       │  6. Execute interrupt handler        │              │
│       │  7. Transfer data                    │              │
│       │  8. Resume original task             │              │
│       │                                      │              │
│                                                             │
│   Advantage: CPU can perform other tasks while waiting      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Interrupt Processing Steps

```
┌─────────────────────────────────────────────────────────────┐
│               Detailed Interrupt Processing Steps            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Device activates IRQ line                               │
│     └── Signal sent to interrupt controller (PIC/APIC)      │
│                                                             │
│  2. Interrupt controller sends interrupt request to CPU     │
│     └── Check interrupt priority                            │
│     └── Forward to CPU if not masked                        │
│                                                             │
│  3. CPU checks interrupt after completing current instruction│
│     └── Check interrupt flag (IF)                           │
│     └── Start processing if interrupts enabled              │
│                                                             │
│  4. Save CPU state                                          │
│     └── Flags register → stack                              │
│     └── CS:IP (or RIP) → stack                              │
│     └── Disable interrupts (prevent nesting)                │
│                                                             │
│  5. Reference interrupt vector table                        │
│     └── Look up handler address by interrupt number         │
│                                                             │
│  6. Execute Interrupt Service Routine (ISR)                 │
│     └── Execute device-specific handling code               │
│     └── Send interrupt acknowledge signal                   │
│                                                             │
│  7. Send EOI (End of Interrupt)                             │
│     └── Notify interrupt controller processing complete     │
│                                                             │
│  8. Return with IRET instruction                            │
│     └── Restore saved state                                 │
│     └── Return to original code                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Interrupt Vector Table

```
x86 Interrupt Vector Table (IDT):

┌─────────────────────────────────────────────────────────────┐
│             Interrupt Descriptor Table (IDT)                 │
├──────┬──────────────────────────────────────────────────────┤
│ Vec  │ Description                                          │
├──────┼──────────────────────────────────────────────────────┤
│  0   │ Divide Error (#DE)                                   │
│  1   │ Debug Exception (#DB)                                │
│  2   │ NMI (Non-Maskable Interrupt)                         │
│  3   │ Breakpoint (#BP)                                     │
│  6   │ Invalid Opcode (#UD)                                 │
│  8   │ Double Fault (#DF)                                   │
│ 13   │ General Protection Fault (#GP)                       │
│ 14   │ Page Fault (#PF)                                     │
│ ...  │ ...                                                  │
│ 32   │ IRQ 0: Timer (PIT)                                   │
│ 33   │ IRQ 1: Keyboard                                      │
│ 34   │ IRQ 2: Cascade (PIC2 connection)                     │
│ 35   │ IRQ 3: COM2/COM4                                     │
│ 36   │ IRQ 4: COM1/COM3                                     │
│ ...  │ ...                                                  │
│ 46   │ IRQ 14: Primary IDE                                  │
│ 47   │ IRQ 15: Secondary IDE                                │
│ ...  │ ...                                                  │
│ 128  │ System Call (Linux: int 0x80)                        │
│ ...  │ ...                                                  │
│ 255  │ Reserved                                             │
└──────┴──────────────────────────────────────────────────────┘

IDT Entry Structure (64-bit):
┌─────────────────────────────────────────────────────────────┐
│  63        48 47 46  44 43    40 39  35 34  32             │
│  ├───────────┼──┼──────┼────────┼─────┼─────┤              │
│  │  Offset   │P │ DPL  │  Type  │ IST │  0  │              │
│  │   [63:48] │  │      │        │     │     │              │
│  └───────────┴──┴──────┴────────┴─────┴─────┘              │
│  31              16 15                0                     │
│  ├─────────────────┼─────────────────┤                     │
│  │  Segment Sel.   │  Offset [15:0]  │                     │
│  └─────────────────┴─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Interrupt-Based I/O Programming

```c
// Interrupt-based keyboard driver example

#define KEYBOARD_IRQ    1
#define KEYBOARD_PORT   0x60

// Keyboard buffer
// Why volatile: the ISR and main code run asynchronously — the ISR modifies
// buffer_head from interrupt context while getchar_interrupt reads it in normal
// context. Without volatile, the compiler may cache buffer_head in a register
// and never see the ISR's update, causing an infinite wait loop.
volatile char keyboard_buffer[256];
volatile int buffer_head = 0;
volatile int buffer_tail = 0;

// Interrupt handler (ISR)
void keyboard_handler(void) {
    // 1. Read scancode
    // Why read immediately: the keyboard controller holds only one scancode.
    // If we delay, the next keystroke overwrites it before we read — losing input.
    unsigned char scancode = inb(KEYBOARD_PORT);

    // 2. Store in buffer
    keyboard_buffer[buffer_head] = scancode;
    buffer_head = (buffer_head + 1) % 256;

    // 3. Send EOI (notify interrupt complete)
    // Why EOI is mandatory: the PIC blocks all interrupts at or below this IRQ's
    // priority until it receives EOI. Forgetting EOI freezes the keyboard (and
    // potentially the timer, since IRQ 0 is lower-numbered but same priority group).
    outb(0x20, 0x20);  // EOI to PIC
}

// Read character (blocking)
char getchar_interrupt(void) {
    // Wait until data is in buffer
    // (In practice, use sleep/wakeup)
    while (buffer_tail == buffer_head) {
        // CPU sleep or run other processes
        asm("hlt");  // Halt until interrupt
    }

    char c = keyboard_buffer[buffer_tail];
    buffer_tail = (buffer_tail + 1) % 256;
    return c;
}

// Register interrupt handler
void init_keyboard(void) {
    // Register handler in IDT
    set_interrupt_handler(32 + KEYBOARD_IRQ, keyboard_handler);

    // Enable interrupt
    enable_irq(KEYBOARD_IRQ);
}
```

### 3.5 Advantages and Disadvantages of Interrupts

```
Advantages:
┌─────────────────────────────────────────────────────────────┐
│ - Improved CPU efficiency (perform other tasks while waiting)│
│ - Suitable for asynchronous event processing                │
│ - Easy to handle multiple devices                           │
│ - Power efficient (CPU can enter sleep state)               │
└─────────────────────────────────────────────────────────────┘

Disadvantages:
┌─────────────────────────────────────────────────────────────┐
│ - Increased implementation complexity                       │
│ - Interrupt overhead (context save/restore)                 │
│ - Interrupt latency exists                                  │
│ - High overhead with frequent interrupts (Interrupt Storm)  │
└─────────────────────────────────────────────────────────────┘

Interrupt Overhead:
- State save: ~100 cycles
- Handler entry: ~50 cycles
- Cache/TLB effects: ~100+ cycles
- Total: ~500-1000 cycles/interrupt

100,000 interrupts/sec @ 3GHz:
Overhead = 100,000 × 500 / 3,000,000,000 ≈ 1.7% CPU
```

---

## 4. DMA (Direct Memory Access)

### 4.1 DMA Concept

```
Definition: Direct data transfer between I/O device and memory without CPU

┌─────────────────────────────────────────────────────────────┐
│              DMA vs CPU-based Transfer Comparison            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CPU-based Transfer (Programmed I/O):                       │
│                                                             │
│    Memory ←──── CPU ────→ I/O Device                        │
│           read      write                                   │
│                                                             │
│    - CPU involvement for every byte                         │
│    - CPU handles data movement                              │
│    - High CPU time consumption                              │
│                                                             │
│  DMA Transfer:                                              │
│                                                             │
│    Memory ◀══════════════▶ I/O Device                       │
│              │                                              │
│              │ DMA Controller                               │
│              └─────────┐                                    │
│                        │                                    │
│           CPU ─────────┘ (setup only)                       │
│                                                             │
│    - CPU handles only transfer setup                        │
│    - DMA controller performs transfer                       │
│    - Notify via interrupt when transfer completes           │
│    - CPU can perform other tasks                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 DMA Operation Process

```
┌─────────────────────────────────────────────────────────────┐
│                    DMA Transfer Process                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CPU configures DMA controller                           │
│     ┌─────────────────────────────────────────────────────┐│
│     │ - Source address (memory or I/O)                    ││
│     │ - Destination address                                ││
│     │ - Transfer size (bytes)                              ││
│     │ - Transfer direction (read/write)                    ││
│     │ - Transfer mode (block/cycle stealing)               ││
│     └─────────────────────────────────────────────────────┘│
│                                                             │
│  2. CPU issues DMA start command                            │
│                                                             │
│  3. DMA controller requests bus (Bus Request)               │
│     - Sends HOLD signal to CPU                              │
│                                                             │
│  4. CPU grants bus (Bus Grant)                              │
│     - Yields bus control with HLDA signal                   │
│     - CPU performs tasks that don't use the bus             │
│                                                             │
│  5. DMA controller transfers data                           │
│     ┌────────┐         ┌────────┐                          │
│     │ Memory │◀═══════▶│  I/O   │                          │
│     └────────┘   DMA   └────────┘                          │
│                  Bus                                        │
│                                                             │
│  6. When transfer completes                                 │
│     - Return bus                                            │
│     - Generate interrupt to CPU                             │
│                                                             │
│  7. CPU handles completion                                  │
│     - Check status                                          │
│     - Setup next transfer if needed                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 DMA Controller Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    DMA Controller                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Control Registers                       │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ Command Register                             │    │    │
│  │  │ - DMA operation mode settings                │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ Mode Register                                │    │    │
│  │  │ - Transfer direction, mode (single/block/demand)│ │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ Status Register                              │    │    │
│  │  │ - Completion status, request status          │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Channel 0                              │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │    │
│  │  │ Address Reg │ │ Count Reg   │ │ Page Reg    │   │    │
│  │  │  0x0000     │ │   1024      │ │   0x00      │   │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Channel 1                              │    │
│  │  ... (same structure)                               │    │
│  └──────────────────────────────────────────────────────┘    │
│  ... Channel 2, 3, ...                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 DMA Transfer Modes

```
1. Block Transfer (Block/Burst Mode):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ──────┬────────────────────────────────┬──────             │
│   CPU  │           DMA                  │ CPU               │
│  usage │     exclusive bus usage        │usage             │
│  ──────┴────────────────────────────────┴──────             │
│                                                             │
│  - Transfer entire block at once                            │
│  - Fastest transfer                                         │
│  - CPU may wait long time                                   │
│  - Suitable for large data                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

2. Cycle Stealing:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──                          │
│  C │D │C │D │C │D │C │D │C │D │C                           │
│  P │M │P │M │P │M │P │M │P │M │P                           │
│  U │A │U │A │U │A │U │A │U │A │U                           │
│  ──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──                          │
│                                                             │
│  - Transfer one word/byte at a time                         │
│  - CPU and DMA alternate bus usage                          │
│  - Minimal CPU impact                                       │
│  - Slower transfer speed                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

3. Demand Transfer (Demand Mode):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Transfer whenever device is ready (DREQ signal based)      │
│  Adapts to device speed                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 DMA Programming Example

```c
// DMA disk read example (simplified)

#define DMA_CHANNEL     2
#define DMA_ADDR_REG    0x04  // Channel 2 address
#define DMA_COUNT_REG   0x05  // Channel 2 count
#define DMA_PAGE_REG    0x81  // Channel 2 page
#define DMA_MODE_REG    0x0B
#define DMA_MASK_REG    0x0A

// DMA transfer setup
// Why this multi-step register dance: the legacy ISA DMA controller (8237A) uses
// 8-bit I/O ports to program 16/24-bit addresses and counts. Each 16-bit value
// must be written as two successive byte-writes (low byte, then high byte),
// synchronized by the flip-flop. This awkward protocol is a relic of the 8-bit
// IBM PC bus — modern DMA (bus-mastering PCIe) avoids it entirely.
void setup_dma_read(void* buffer, size_t count) {
    uint32_t addr = (uint32_t)buffer;

    // 1. Mask DMA channel (disable)
    // Why mask first: prevents the DMA controller from acting on a half-configured
    // channel. Without masking, a stale DREQ signal could trigger a transfer to
    // the old address while we're writing the new one.
    outb(DMA_MASK_REG, DMA_CHANNEL | 0x04);

    // 2. Reset flip-flop
    outb(0x0C, 0);

    // 3. Set mode (read, channel 2, single mode)
    outb(DMA_MODE_REG, 0x46);

    // 4. Set address (lower 16 bits)
    outb(DMA_ADDR_REG, addr & 0xFF);
    outb(DMA_ADDR_REG, (addr >> 8) & 0xFF);

    // 5. Set page (upper bits)
    outb(DMA_PAGE_REG, (addr >> 16) & 0xFF);

    // 6. Set count (count - 1)
    outb(DMA_COUNT_REG, (count - 1) & 0xFF);
    outb(DMA_COUNT_REG, ((count - 1) >> 8) & 0xFF);

    // 7. Unmask DMA channel (enable)
    outb(DMA_MASK_REG, DMA_CHANNEL);
}

// Disk read command + DMA
void read_disk_dma(void* buffer, uint32_t sector, uint16_t count) {
    // Setup DMA
    setup_dma_read(buffer, count * 512);

    // Issue disk read command to disk controller
    issue_disk_read_command(sector, count);

    // CPU can perform other tasks
    // Interrupt occurs when transfer completes
}

// DMA completion interrupt handler
void dma_complete_handler(void) {
    // Handle transfer completion
    // Check status, buffer available
    signal_dma_complete();
}
```

---

## 5. Bus Architecture

### 5.1 Types of Buses

```
┌─────────────────────────────────────────────────────────────┐
│                    System Bus Structure                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                      CPU                             │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │                                │
│                       Front-Side Bus                        │
│                      (or QPI/UPI)                           │
│                            │                                │
│  ┌─────────────────────────┴───────────────────────────┐    │
│  │              Memory Controller Hub                   │    │
│  │                   (Northbridge)                      │    │
│  └───────────┬─────────────┬─────────────┬─────────────┘    │
│              │             │             │                  │
│          Memory         PCIe         DMI                   │
│           Bus           x16                                │
│              │             │             │                  │
│  ┌───────────┴───┐    ┌────┴────┐       │                  │
│  │    DRAM       │    │   GPU   │       │                  │
│  └───────────────┘    └─────────┘       │                  │
│                                         │                  │
│  ┌──────────────────────────────────────┴──────────────┐   │
│  │             I/O Controller Hub                      │   │
│  │                 (Southbridge)                       │   │
│  └───────┬──────────┬──────────┬──────────┬───────────┘   │
│          │          │          │          │                │
│        SATA       USB        PCIe      Audio              │
│          │          │        x1         │                  │
│    ┌─────┴─────┐ ┌──┴──┐  ┌──┴──┐  ┌───┴───┐             │
│    │ HDD/SSD  │ │ USB │  │ NIC │  │Codec  │             │
│    └──────────┘ │Devs │  └─────┘  └───────┘             │
│                 └─────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Bus Characteristics

```
Bus Components:
┌─────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Data Bus                            │  │
│  │  - Data transfer lines                                 │  │
│  │  - Width: 8, 16, 32, 64 bits                           │  │
│  │  - Bidirectional                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Address Bus                          │  │
│  │  - Memory/I/O addressing                               │  │
│  │  - Width: 20, 32, 36, 40+ bits                         │  │
│  │  - Unidirectional (CPU → device)                       │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Control Bus                          │  │
│  │  - Control signal transfer                             │  │
│  │  - Read, Write, IRQ, DMA request, etc.                 │  │
│  │  - Bidirectional                                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Major Bus Standards:
┌────────────────┬──────────────┬───────────────┬─────────────┐
│     Bus        │   Bandwidth  │    Purpose    │   Features  │
├────────────────┼──────────────┼───────────────┼─────────────┤
│ PCIe 4.0 x16   │  ~64 GB/s    │ GPU, high-speed│ Serial, lanes│
├────────────────┼──────────────┼───────────────┼─────────────┤
│ PCIe 5.0 x16   │  ~128 GB/s   │ Next-gen GPU  │ Latest std  │
├────────────────┼──────────────┼───────────────┼─────────────┤
│ SATA III       │  ~600 MB/s   │ Storage       │ Serial      │
├────────────────┼──────────────┼───────────────┼─────────────┤
│ NVMe (PCIe 4)  │  ~7 GB/s     │ High-speed SSD│ Low latency │
├────────────────┼──────────────┼───────────────┼─────────────┤
│ USB 3.2 Gen2  │  ~1.25 GB/s  │ Peripherals   │ Universal   │
├────────────────┼──────────────┼───────────────┼─────────────┤
│ Thunderbolt 4 │  ~5 GB/s     │ High-speed    │ Daisy-chain │
└────────────────┴──────────────┴───────────────┴─────────────┘
```

### 5.3 Bus Arbitration

```
Preventing collisions when multiple devices share bus:

1. Centralized Arbitration:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐           │
│  │Device 0│  │Device 1│  │Device 2│  │Device 3│           │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘           │
│      │  REQ      │  REQ      │  REQ      │  REQ            │
│      │           │           │           │                  │
│      └─────┬─────┴─────┬─────┴─────┬─────┘                  │
│            │           │           │                        │
│            ▼           ▼           ▼                        │
│  ┌──────────────────────────────────────────────┐          │
│  │              Bus Arbiter                      │          │
│  │           (Central arbiter)                   │          │
│  │  - Receive requests                           │          │
│  │  - Issue GRANT signal by priority             │          │
│  └──────────────────────────────────────────────┘          │
│            │           │           │                        │
│      GRANT │     GRANT │     GRANT │                        │
│            ▼           ▼           ▼                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

2. Priority Schemes:
- Fixed priority: Fixed priority per device
- Round robin: Fair allocation in rotation
- Dynamic priority: Adjust based on usage patterns
```

---

## 6. I/O Interfaces

### 6.1 I/O Addressing

```
1. Isolated I/O (Port-Mapped I/O):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Memory Address Space         I/O Address Space            │
│  ┌────────────────────┐        ┌────────────────────┐      │
│  │ 0x0000_0000        │        │ 0x0000             │      │
│  │                    │        │                    │      │
│  │      Memory        │        │    I/O Ports       │      │
│  │                    │        │                    │      │
│  │ 0xFFFF_FFFF        │        │ 0xFFFF             │      │
│  └────────────────────┘        └────────────────────┘      │
│                                                             │
│  - Uses separate address space                              │
│  - Uses IN, OUT instructions                                │
│  - Used in x86 architecture                                 │
│                                                             │
│  Example:                                                   │
│  outb(0x3F8, data);   // Write to port 0x3F8                │
│  data = inb(0x3F8);   // Read from port 0x3F8               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

2. Memory-Mapped I/O (MMIO):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Unified Address Space                                      │
│  ┌────────────────────────────────────────────┐            │
│  │ 0x0000_0000                                │            │
│  │                                            │            │
│  │      System Memory (RAM)                   │            │
│  │                                            │            │
│  │ 0x7FFF_FFFF                                │            │
│  ├────────────────────────────────────────────┤            │
│  │ 0x8000_0000                                │            │
│  │                                            │            │
│  │      I/O Device Registers                  │            │
│  │      (accessed like memory)                │            │
│  │                                            │            │
│  │ 0xFFFF_FFFF                                │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│  - Access I/O with normal memory instructions               │
│  - Mainly used in ARM, RISC-V, etc.                         │
│  - Most devices in modern PCs also use MMIO                 │
│                                                             │
│  Example:                                                   │
│  // Why volatile: without it the compiler may optimize away │
│  // repeated reads (caching the value in a register) or     │
│  // reorder/eliminate writes — but I/O registers have side  │
│  // effects: reading clears status bits, writing triggers   │
│  // hardware actions. volatile forces every access to go    │
│  // through to the actual memory-mapped address.            │
│  volatile uint32_t* reg = (uint32_t*)0xFE200000;           │
│  *reg = value;         // Write to I/O register            │
│  value = *reg;         // Read from I/O register           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Device Driver Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Device Driver Structure                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Driver Entry Points                     │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  init()      - Driver initialization                 │   │
│  │  open()      - Open device                           │   │
│  │  close()     - Close device                          │   │
│  │  read()      - Read data                             │   │
│  │  write()     - Write data                            │   │
│  │  ioctl()     - Control commands                      │   │
│  │  interrupt_handler() - Interrupt handling            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Driver Internal Data                    │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  - Device state                                      │   │
│  │  - Buffers                                           │   │
│  │  - Wait queues                                       │   │
│  │  - Configuration info                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Linux device driver example:
static struct file_operations my_fops = {
    .owner   = THIS_MODULE,
    .open    = my_open,
    .release = my_close,
    .read    = my_read,
    .write   = my_write,
    .unlocked_ioctl = my_ioctl,
};
```

---

## 7. Modern I/O Systems

### 7.1 NVMe (Non-Volatile Memory Express)

```
┌─────────────────────────────────────────────────────────────┐
│                    NVMe Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Traditional SATA/AHCI vs NVMe:                             │
│                                                             │
│  SATA/AHCI:                                                 │
│  - Single command queue (depth 32)                          │
│  - Designed for HDD era                                     │
│  - High latency                                             │
│                                                             │
│  NVMe:                                                      │
│  - 64K queues, 64K commands per queue                       │
│  - Optimized for SSD                                        │
│  - Low latency                                              │
│  - Direct PCIe connection                                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   CPU / Driver                       │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│           ┌────────────────┼────────────────┐              │
│           │                │                │              │
│           ▼                ▼                ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Submit Q 0  │  │ Submit Q 1  │  │ Submit Q N  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 NVMe Controller                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         │                │                │                │
│         ▼                ▼                ▼                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │Complete Q 0 │  │Complete Q 1 │  │Complete Q N │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 USB System

```
┌─────────────────────────────────────────────────────────────┐
│                    USB Architecture                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  USB Generation Speeds:                                     │
│  ┌──────────────┬──────────────────────────────────────┐   │
│  │ USB 2.0     │  480 Mbps (High Speed)               │   │
│  │ USB 3.0     │  5 Gbps (SuperSpeed)                 │   │
│  │ USB 3.1     │  10 Gbps (SuperSpeed+)               │   │
│  │ USB 3.2     │  20 Gbps (SuperSpeed USB 20Gbps)     │   │
│  │ USB4        │  40 Gbps                              │   │
│  └──────────────┴──────────────────────────────────────┘   │
│                                                             │
│  USB Transfer Types:                                        │
│  ┌──────────────┬──────────────────────────────────────┐   │
│  │ Control     │ Setup, control (small data)          │   │
│  │ Bulk        │ Large data (storage devices)         │   │
│  │ Interrupt   │ Small, periodic (keyboard, mouse)    │   │
│  │ Isochronous │ Real-time, periodic (audio, video)   │   │
│  └──────────────┴──────────────────────────────────────┘   │
│                                                             │
│  USB Topology:                                              │
│                                                             │
│       Host Controller                                       │
│            │                                                │
│            ▼                                                │
│        Root Hub                                             │
│        /   |   \                                            │
│       /    |    \                                           │
│    Hub  Device  Device                                      │
│    / \                                                      │
│   /   \                                                     │
│ Dev   Dev                                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 I/O Virtualization

```
┌─────────────────────────────────────────────────────────────┐
│                   I/O Virtualization Techniques              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Emulation:                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Guest OS  →  Hypervisor  →  Physical Device        │   │
│  │            (I/O trap and emulation)                  │   │
│  └─────────────────────────────────────────────────────┘   │
│  - Slow, high compatibility                                 │
│                                                             │
│  2. Para-virtualization (virtio):                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Guest OS (virtio driver) → Hypervisor → Device     │   │
│  └─────────────────────────────────────────────────────┘   │
│  - Optimized virtual interface                              │
│  - Guest OS modification required                           │
│                                                             │
│  3. Direct Device Assignment (VFIO):                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Guest OS  →  Physical Device (direct access)       │   │
│  │            (Memory isolation via IOMMU)              │   │
│  └─────────────────────────────────────────────────────┘   │
│  - Native performance                                       │
│  - Device cannot be shared                                  │
│                                                             │
│  4. SR-IOV (Single Root I/O Virtualization):               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Split one physical device into multiple virtual    │   │
│  │  functions (VF), each VM accesses independent VF    │   │
│  └─────────────────────────────────────────────────────┘   │
│  - Native performance + sharing possible                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

1. Explain the differences between polling, interrupt, and DMA methods.

2. What is the role of the interrupt vector table?

3. What are the 3 pieces of information that must be configured in a DMA controller?

### Intermediate Problems

4. Choose the appropriate I/O method for each scenario:
   - (a) Keyboard input processing
   - (b) Reading 10MB file from disk
   - (c) High-speed network packet processing (10Gbps)

5. What is an "Interrupt Storm" in interrupt-based I/O?

6. Compare the advantages and disadvantages of memory-mapped I/O vs port-mapped I/O.

### Advanced Problems

7. Calculate CPU efficiency under the following conditions:
   - CPU clock: 3GHz
   - Disk transfer rate: 500MB/s
   - DMA block size: 4KB
   - DMA completion interrupt processing time: 1000 cycles

8. Explain USB's 4 transfer types and give device examples suitable for each.

9. Explain how SR-IOV improves I/O performance in virtualized environments.

<details>
<summary>Answers</summary>

1. I/O Method Comparison:
   - Polling: CPU continuously checks status, simple but wastes CPU
   - Interrupt: Device notifies completion, CPU efficient but has overhead
   - DMA: Direct memory transfer, efficient for bulk data but complex

2. Interrupt Vector Table:
   - Maps interrupt numbers to handler addresses
   - Jumps to corresponding handler when interrupt occurs

3. DMA Configuration Info:
   - Source/destination address
   - Transfer size (bytes)
   - Transfer direction (read/write)

4. Appropriate I/O Methods:
   - (a) Interrupt (slow, asynchronous)
   - (b) DMA (bulk block transfer)
   - (c) DMA + polling or NAPI (high-speed, many packets)

5. Interrupt Storm:
   - Situation where too many interrupts cause CPU to only handle interrupts
   - Cannot perform normal tasks
   - Solution: Interrupt coalescing, switch to polling (NAPI)

6. I/O Addressing Comparison:
   - Port-mapped: Separate address space, requires separate instructions, saves address space
   - Memory-mapped: Unified address space, uses normal instructions, cache considerations needed

7. CPU Efficiency Calculation:
   - Transfer rate 500MB/s, block 4KB → 125,000 transfers/sec
   - 1000 cycles interrupt handling per transfer
   - Total interrupt cycles: 125,000,000
   - CPU efficiency: 1 - (125M / 3G) = 1 - 0.042 = 95.8%

8. USB Transfer Types:
   - Control: Device setup, status checking (all USB devices)
   - Bulk: Large data (USB drives, printers)
   - Interrupt: Small periodic (keyboard, mouse)
   - Isochronous: Real-time (webcam, USB audio)

9. SR-IOV Principle:
   - Creates multiple virtual functions (VF) on one physical device
   - Each VM directly accesses dedicated VF
   - Native performance without hypervisor intervention
   - IOMMU ensures memory isolation

</details>

---

## Next Steps

- [18_Parallel_Processing_Multicore.md](./18_Parallel_Processing_Multicore.md) - Multicore architecture and parallel programming

---

## References

- Computer Organization and Design (Patterson & Hennessy)
- Operating System Concepts (Silberschatz et al.)
- [NVMe Specification](https://nvmexpress.org/specifications/)
- [USB Specification](https://www.usb.org/documents)
- Linux Device Drivers (Corbet, Rubini, Kroah-Hartman)
