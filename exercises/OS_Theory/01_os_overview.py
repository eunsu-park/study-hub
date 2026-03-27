"""
Exercises for Lesson 01: Operating System Overview
Topic: OS_Theory

Solutions to practice problems from the lesson.
"""


# === Exercise 1: OS Role Identification ===
# Problem: For each scenario, identify which OS role (Resource Manager or
# Service Provider) is primarily at work and name the specific OS subsystem.

def exercise_1():
    """Identify OS roles for given scenarios."""
    scenarios = [
        {
            "description": "A video player requests 256 MB of RAM to buffer frames",
            "role": "Resource Manager",
            "subsystem": "Memory Management",
            "explanation": (
                "The OS allocates physical memory from the available pool. "
                "This is resource management because the OS decides how much "
                "of a finite resource (RAM) to grant to this process."
            ),
        },
        {
            "description": "A user types 'ls' in the terminal and sees a directory listing",
            "role": "Service Provider",
            "subsystem": "File System + Process Management",
            "explanation": (
                "The OS provides a file listing service through the shell interface. "
                "It creates a process for 'ls', which calls readdir() syscalls, and "
                "the OS returns directory entries -- this is service provision."
            ),
        },
        {
            "description": "Two processes try to write to the same file simultaneously -- one is blocked",
            "role": "Resource Manager",
            "subsystem": "File System (I/O Management / Synchronization)",
            "explanation": (
                "The OS manages concurrent access to the shared file resource. "
                "It enforces locking/ordering to prevent data corruption -- "
                "a classic resource management task."
            ),
        },
        {
            "description": "A web browser opens a TCP socket to contact a remote server",
            "role": "Service Provider",
            "subsystem": "Networking (Protocol Stack)",
            "explanation": (
                "The OS provides a socket API that abstracts away TCP/IP details. "
                "The browser calls socket() and connect(), and the OS handles "
                "protocol negotiation, routing, etc."
            ),
        },
        {
            "description": "A running program divides by zero and the OS terminates it",
            "role": "Resource Manager",
            "subsystem": "Process Management (Exception Handling)",
            "explanation": (
                "The hardware raises a divide-by-zero exception (interrupt). "
                "The OS catches it, determines the process caused a fatal error, "
                "and terminates it to protect system stability -- managing the "
                "CPU resource and enforcing protection."
            ),
        },
    ]

    for i, s in enumerate(scenarios, 1):
        print(f"Scenario {i}: {s['description']}")
        print(f"  Role: {s['role']}")
        print(f"  Subsystem: {s['subsystem']}")
        print(f"  Explanation: {s['explanation']}")
        print()


# === Exercise 2: Kernel Architecture Trade-offs ===
# Problem: Identify kernel type for each OS and explain one practical consequence.

def exercise_2():
    """Analyze kernel architecture trade-offs for real OSes."""
    os_table = [
        {
            "os": "Linux",
            "kernel_type": "Monolithic (with loadable modules)",
            "consequence": (
                "A buggy device driver can crash the entire kernel because all "
                "drivers run in kernel space with full privileges. This is why "
                "kernel panics from faulty GPU or filesystem drivers are possible."
            ),
        },
        {
            "os": "macOS (XNU)",
            "kernel_type": "Hybrid",
            "consequence": (
                "XNU combines a Mach microkernel core (for IPC, virtual memory) "
                "with a BSD monolithic layer (for POSIX, networking, filesystems). "
                "This gives decent performance while keeping a clean IPC mechanism, "
                "but adds complexity in the kernel codebase."
            ),
        },
        {
            "os": "QNX",
            "kernel_type": "Microkernel",
            "consequence": (
                "Device drivers and filesystems run as user-space processes. "
                "If a driver crashes, the kernel continues running and can restart "
                "the driver. This makes QNX extremely reliable -- critical for "
                "automotive and medical systems -- but IPC overhead slows I/O."
            ),
        },
        {
            "os": "Windows NT",
            "kernel_type": "Hybrid",
            "consequence": (
                "The HAL (Hardware Abstraction Layer) makes Windows portable "
                "across architectures, but critical subsystems like the graphics "
                "driver (win32k.sys) run in kernel mode for performance, "
                "which historically has been a major source of BSoD crashes."
            ),
        },
    ]

    print("Kernel Architecture Comparison:")
    print(f"{'OS':<16} {'Kernel Type':<40} ")
    print("-" * 80)
    for entry in os_table:
        print(f"{entry['os']:<16} {entry['kernel_type']:<40}")
        print(f"  Consequence: {entry['consequence']}")
        print()

    print("\nWhy do most desktop OSes use hybrid or monolithic rather than microkernel?")
    print(
        "Answer: Performance. Desktop workloads involve heavy I/O and system call\n"
        "traffic. In a pure microkernel, every file read, network packet, and\n"
        "graphics operation requires IPC between user-space servers and the\n"
        "microkernel -- each IPC involves two context switches and message copying.\n"
        "Monolithic/hybrid designs keep hot paths (filesystem, networking, graphics)\n"
        "in kernel space, avoiding this overhead. The theoretical elegance of\n"
        "microkernels is outweighed by the practical performance cost for\n"
        "general-purpose desktop usage."
    )


# === Exercise 3: System Call Tracing ===
# Problem: Write a minimal C program description and list every system call.

def exercise_3():
    """List system calls for a file-create-fork-read-delete sequence."""
    print("Program sequence and system calls:\n")

    syscalls = [
        ("open('hello.txt', O_WRONLY|O_CREAT|O_TRUNC, 0644)", "File Management",
         "Creates/opens the file for writing"),
        ("write(fd, 'Hello, OS!', 10)", "File Management",
         "Writes the string to the file"),
        ("close(fd)", "File Management",
         "Closes the file descriptor"),
        ("fork()", "Process Control",
         "Creates a child process (duplicates the parent)"),
        ("--- Child process ---", "", ""),
        ("open('hello.txt', O_RDONLY)", "File Management",
         "Child opens the file for reading"),
        ("read(fd, buffer, 10)", "File Management",
         "Child reads the file contents"),
        ("close(fd)", "File Management",
         "Child closes the file"),
        ("exit(0)", "Process Control",
         "Child terminates"),
        ("--- Parent process ---", "", ""),
        ("wait(NULL)", "Process Control",
         "Parent waits for child to terminate"),
        ("unlink('hello.txt')", "File Management",
         "Parent deletes the file"),
        ("exit(0)", "Process Control",
         "Parent terminates (implicit return from main)"),
    ]

    print(f"{'System Call':<50} {'Category':<25} {'Purpose'}")
    print("-" * 100)
    for call, category, purpose in syscalls:
        if call.startswith("---"):
            print(f"\n  {call}")
        else:
            print(f"  {call:<48} {category:<25} {purpose}")


# === Exercise 4: Interrupt Handling Deep Dive ===
# Problem: Answer questions about interrupt priority and handling.

def exercise_4():
    """Analyze interrupt priority scenarios."""

    print("Question 1: Keyboard interrupt during timer interrupt handling\n")
    print("Priority table: Timer=3 (higher), Keyboard=4 (lower)")
    print()
    print("Execution trace:")
    print("  1. CPU is handling timer interrupt (ISR for timer running)")
    print("  2. Keyboard interrupt arrives")
    print("  3. Keyboard priority (4) < Timer priority (3), so keyboard")
    print("     interrupt is MASKED/DEFERRED -- it does NOT preempt the timer ISR")
    print("  4. Timer ISR completes, sends EOI")
    print("  5. CPU checks pending interrupts, finds keyboard interrupt")
    print("  6. Keyboard ISR now executes")
    print("  7. Keyboard ISR completes, sends EOI")
    print("  8. CPU returns to the interrupted user process")
    print()
    print("Key point: Lower-priority interrupts are deferred until higher-priority")
    print("handlers finish. If a HIGHER-priority interrupt arrived during the timer")
    print("ISR, it WOULD preempt (nested interrupt).\n")

    print("Question 2: Why save/restore CPU state?\n")
    print("If the OS skipped saving registers before running the ISR:")
    print("  - The ISR would overwrite the process's register values")
    print("  - When returning to the interrupted process, its PC would be wrong")
    print("    (jumping to garbage address), SP would point to wrong stack location,")
    print("    and computation results in general registers would be lost")
    print("  - The process would crash or produce incorrect results")
    print("  - In effect, the interrupt would corrupt any running program\n")

    print("Question 3: Timer interrupt in a time-sharing OS\n")
    print("The timer interrupt handler:")
    print("  1. Decrements the current process's remaining time quantum")
    print("  2. Updates system clock/timekeeping variables")
    print("  3. Checks sleeping processes for wakeup deadlines")
    print("  4. If time quantum reaches 0: calls the scheduler to")
    print("     select the next process (context switch)")
    print("  5. Sends EOI to interrupt controller")
    print()
    print("Why frequent (e.g., every 10ms)?")
    print("  - Ensures interactive responsiveness: users notice delays > 50-100ms")
    print("  - With 10ms granularity and 10 processes, each gets a turn every 100ms")
    print("  - Too infrequent: system feels sluggish to interactive users")
    print("  - Too frequent: excessive context switch overhead wastes CPU")


# === Exercise 5: Design a Minimal OS ===
# Problem: Design an OS for a single-function embedded device.

def exercise_5():
    """Design a minimal OS for a microwave controller."""
    print("Embedded Microwave Controller OS Design\n")

    print("1. Kernel Type: Monolithic (or even no OS / bare-metal)")
    print("   Justification: The device has a single fixed function. A monolithic")
    print("   kernel (or even a superloop) is simplest and fastest. No need for")
    print("   modularity or fault isolation -- if any component fails, the whole")
    print("   device is unusable anyway. A microkernel's IPC overhead would waste")
    print("   limited CPU resources with no benefit.\n")

    print("2. Hardware Interrupts Needed:")
    interrupts = [
        ("Timer interrupt", "Track cooking duration, trigger beep at completion"),
        ("Keypad interrupt", "Detect button presses (start, stop, time settings)"),
        ("Door sensor interrupt", "Detect door open/close for safety interlock"),
        ("Temperature sensor interrupt", "Monitor internal temperature for safety"),
        ("Power failure interrupt", "Safely shut down magnetron if power drops"),
    ]
    for name, purpose in interrupts:
        print(f"   - {name}: {purpose}")
    print()

    print("3. System Call Categories:")
    print("   Necessary:")
    print("     - Device Management: Read sensors, control magnetron, drive display")
    print("     - Information Maintenance: Get/set timer values, read system clock")
    print("   Unnecessary:")
    print("     - File Management: No filesystem needed (no persistent storage)")
    print("     - Communication: No networking")
    print("     - Process Control: Single task, no fork/exec needed\n")

    print("4. Preemptive Scheduling: No")
    print("   Justification: Only one task runs (the control loop). There is no")
    print("   need to time-share the CPU among competing processes. The system")
    print("   uses interrupt-driven I/O (button presses, timer expiry) with a")
    print("   main event loop. An RTOS with preemption adds complexity and")
    print("   overhead with no benefit for a single-function device.")
    print("   Exception: If safety-critical code must preempt cooking logic")
    print("   (e.g., door-open immediately stops magnetron), a simple priority-")
    print("   based preemptive scheduler could be justified -- but an interrupt")
    print("   handler suffices for this.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: OS Role Identification ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Kernel Architecture Trade-offs ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: System Call Tracing ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Interrupt Handling Deep Dive ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Design a Minimal OS ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
