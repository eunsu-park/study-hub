"""
Exercises for Lesson 02: Process Concepts
Topic: OS_Theory

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Memory Layout Analysis ===
# Problem: For each variable, state the memory section and explain why.

def exercise_1():
    """Identify memory sections for variables in a C program."""
    variables = [
        {
            "code": 'int server_port = 8080;         // (1)',
            "section": "Data",
            "reason": (
                "Initialized global variable. The compiler places it in the "
                "Data section because it has a known initial value (8080) that "
                "must persist for the program's lifetime."
            ),
        },
        {
            "code": 'char *app_name;                  // (2)',
            "section": "BSS",
            "reason": (
                "Uninitialized global pointer. The BSS section holds global/static "
                "variables without explicit initializers. The OS zero-initializes "
                "the entire BSS section at load time (so app_name starts as NULL)."
            ),
        },
        {
            "code": 'static int request_count = 0;   // (3)',
            "section": "Data (or BSS -- implementation-dependent)",
            "reason": (
                "Static variable initialized to 0. Some compilers place it in "
                "BSS (since 0 is the default), others in Data. Either way, it "
                "has static storage duration -- it persists across function calls."
            ),
        },
        {
            "code": 'char buf[256];                   // (4) inside function',
            "section": "Stack",
            "reason": (
                "Local array inside a function. Stack memory is allocated when "
                "handle_request() is called and freed when it returns. The 256 "
                "bytes are part of the function's stack frame."
            ),
        },
        {
            "code": 'static int call_num = 0;         // (5) inside function',
            "section": "Data (or BSS)",
            "reason": (
                "Static local variable. Despite being declared inside a function, "
                "the 'static' keyword gives it static storage duration -- it lives "
                "in the Data/BSS section, NOT on the stack. Its value persists "
                "across calls to handle_request()."
            ),
        },
        {
            "code": 'int *data = malloc(1024);        // (6)',
            "section": "data pointer: Stack; *data (allocated block): Heap",
            "reason": (
                "The pointer variable 'data' is a local variable on the Stack. "
                "The 1024 bytes it points to are allocated on the Heap by malloc(). "
                "This is a crucial distinction: the pointer and the memory it "
                "references live in different sections."
            ),
        },
    ]

    print("Memory Layout Analysis:\n")
    for v in variables:
        print(f"Code:    {v['code']}")
        print(f"Section: {v['section']}")
        print(f"Reason:  {v['reason']}")
        print()


# === Exercise 2: Process State Transitions ===
# Problem: Trace the state of each process at each time step.

def exercise_2():
    """Trace process states through a series of events."""
    # States: N=New, Rd=Ready, Rn=Running, W=Waiting, T=Terminated, -=not exist
    events = [
        "t=0: P1 created, P2 created",
        "t=1: P1 dispatched to CPU",
        "t=2: P3 created",
        "t=3: P1 requests file read (I/O)",
        "t=4: P2 dispatched; P4 created",
        "t=5: P1's I/O completes",
        "t=6: P2's time slice expires",
        "t=7: P1 dispatched; P3 dispatched (multicore)",
        "t=8: P3 calls exit()",
    ]

    # State table: rows = processes, cols = time steps
    states = {
        "P1": ["Ready", "Running", "Running", "Waiting", "Waiting", "Ready", "Ready", "Running", "Running"],
        "P2": ["Ready", "Ready", "Ready", "Ready", "Running", "Running", "Ready", "Ready", "Ready"],
        "P3": ["-", "-", "Ready", "Ready", "Ready", "Ready", "Ready", "Running", "Terminated"],
        "P4": ["-", "-", "-", "-", "Ready", "Ready", "Ready", "Ready", "Ready"],
    }

    print("Process State Transition Table:\n")
    print(f"{'Time':<6}", end="")
    for t in range(9):
        print(f"{'t=' + str(t):<12}", end="")
    print()
    print("-" * 114)

    for proc in ["P1", "P2", "P3", "P4"]:
        print(f"{proc:<6}", end="")
        for state in states[proc]:
            print(f"{state:<12}", end="")
        print()

    print("\nEvent log:")
    for event in events:
        print(f"  {event}")

    print("\nExplanation of key transitions:")
    print("  t=3: P1 Running->Waiting (I/O request, non-preemptive decision point)")
    print("  t=5: P1 Waiting->Ready (I/O complete, not immediately dispatched)")
    print("  t=6: P2 Running->Ready (time slice expired, preempted)")
    print("  t=7: Multicore allows P1 and P3 to both be Running simultaneously")
    print("  t=8: P3 Running->Terminated (explicit exit() call)")


# === Exercise 3: fork() Output Prediction ===
# Problem: Predict the exact output of a fork program.

def exercise_3():
    """Predict fork() output and analyze process creation."""
    print("Program analysis:\n")
    print("```c")
    print("int x = 0;  // global variable")
    print("int main() {")
    print('    printf("start\\n");')
    print("    pid_t p1 = fork();")
    print("    if (p1 == 0) { x = 10; printf(...); return 0; }")
    print("    pid_t p2 = fork();")
    print("    if (p2 == 0) { x = 20; printf(...); return 0; }")
    print("    wait(NULL); wait(NULL);")
    print('    printf("parent: x=%d\\n", x);')
    print("}")
    print("```\n")

    print("Predicted output:")
    print('  start')
    print('  child1: x=10    (from first child, PID 1001)')
    print('  child2: x=20    (from second child, PID 1002)')
    print('  parent: x=0     (from original process, PID 1000)')
    print()

    print("Note: child1 and child2 lines may appear in either order.\n")

    print("Question 1: How many processes are created in total?")
    print("  Answer: 3 processes total")
    print("  - Original parent (PID 1000)")
    print("  - Child1 from first fork() (PID 1001) -- returns 0 immediately")
    print("  - Child2 from second fork() (PID 1002)")
    print("  Note: Child1 returns before the second fork(), so it does NOT")
    print("  create another child. Only the parent reaches the second fork().\n")

    print("Question 2: What is the value of x in the parent at the end?")
    print("  Answer: x = 0")
    print("  Each fork() creates an independent copy of the address space.")
    print("  Child1 sets its copy of x to 10, child2 sets its copy to 20,")
    print("  but these changes are invisible to the parent. The parent's x")
    print("  was never modified after initialization.\n")

    print("Question 3: Can the two child output lines appear in either order?")
    print("  Answer: Yes. The two children are independent processes.")
    print("  The OS scheduler can run them in any order. Child1 might")
    print("  print before or after child2, depending on scheduling decisions.")
    print("  The only guarantee is that 'start' is first and 'parent: x=0'")
    print("  is last (due to wait() calls).")


# === Exercise 4: Zombie and Orphan Processes ===
# Problem: Analyze zombie and orphan scenarios from code.

def exercise_4():
    """Analyze zombie and orphan process scenarios."""
    print("Code analysis: parent sleeps 60s without calling wait()\n")

    print("Question 1: Child state after exit(0)?")
    print("  The child enters the ZOMBIE state (Z).")
    print("  Why: The child has terminated (called exit(0)), so it has")
    print("  released its memory, file descriptors, and other resources.")
    print("  However, its PCB (process table entry) still exists because")
    print("  the parent has not yet called wait() to collect the exit status.")
    print("  The kernel keeps the PCB so the parent can eventually retrieve")
    print("  the child's exit code. Until then, the child is a zombie.\n")

    print("Question 2: How to observe zombie state on Linux?")
    print("  $ ps aux | grep Z")
    print("  # Look for processes with 'Z' or 'Z+' in the STAT column")
    print("  # Or more specifically:")
    print("  $ ps -eo pid,ppid,stat,cmd | grep defunct")
    print("  # Zombie processes show as '<defunct>' in the CMD column\n")

    print("Question 3: What change prevents the zombie?")
    print("  Option A: Add wait() or waitpid() after the sleep:")
    print("    sleep(60);")
    print("    wait(NULL);  // reap the zombie")
    print()
    print("  Option B: Install a SIGCHLD handler that calls wait():")
    print("    signal(SIGCHLD, SIG_IGN);  // kernel auto-reaps children")
    print()
    print("  Option C: Double-fork technique (grandchild is adopted by init).\n")

    print("Question 4: What if the parent crashes before sleep ends?")
    print("  The child (zombie) becomes an ORPHAN process.")
    print("  - init (PID 1) or systemd adopts it (PPID changes to 1)")
    print("  - init periodically calls wait() on all adopted children")
    print("  - The zombie is reaped shortly after adoption")
    print("  - So paradoxically, the parent crashing FIXES the zombie!")


# === Exercise 5: Context Switch Cost Estimation ===
# Problem: Calculate CPU overhead from context switching.

def exercise_5():
    """Calculate context switch overhead."""
    # Given values
    switches_per_sec = 1000
    direct_cost_us = 5        # microseconds
    indirect_cost_us = 15     # microseconds
    total_cost_us = direct_cost_us + indirect_cost_us  # 20 us per switch
    cpu_freq_ghz = 1.0
    cycles_per_sec = cpu_freq_ghz * 1e9

    print("Context Switch Cost Estimation\n")
    print(f"Given: {switches_per_sec} context switches/second")
    print(f"  Direct overhead: {direct_cost_us} us (register save/restore)")
    print(f"  Indirect overhead: {indirect_cost_us} us (TLB flush, cache warm-up)")
    print(f"  Total per switch: {total_cost_us} us")
    print(f"  CPU: {cpu_freq_ghz} GHz single-core\n")

    # Q1: Total CPU time lost per second
    total_time_us = switches_per_sec * total_cost_us
    total_time_ms = total_time_us / 1000
    print(f"Q1: Total CPU time lost per second:")
    print(f"  {switches_per_sec} switches * {total_cost_us} us = {total_time_us} us = {total_time_ms} ms\n")

    # Q2: Percentage of CPU's total cycles
    total_cycles_lost = total_time_us * 1e-6 * cycles_per_sec
    percentage = (total_time_us / 1e6) * 100  # fraction of 1 second
    print(f"Q2: As percentage of 1 GHz CPU capacity:")
    print(f"  {total_time_ms} ms out of 1000 ms = {percentage:.1f}%")
    print(f"  ({total_cycles_lost:,.0f} cycles out of {cycles_per_sec:,.0f} cycles/sec)\n")

    # Q3: Reduced to 500 switches/sec
    new_switches = 500
    new_total_us = new_switches * total_cost_us
    new_percentage = (new_total_us / 1e6) * 100
    print(f"Q3: With {new_switches} switches/sec (doubled time quantum):")
    print(f"  {new_switches} * {total_cost_us} us = {new_total_us} us = {new_total_us/1000} ms")
    print(f"  New overhead: {new_percentage:.1f}% (halved from {percentage:.1f}%)\n")

    # Q4: Workloads where reducing context switches hurts
    print("Q4: Two workload types where reducing context switches hurts:")
    print()
    print("  1. Interactive/real-time workloads:")
    print("     Doubling the time quantum increases response time. A user typing")
    print("     in an editor might wait twice as long for keystroke feedback.")
    print("     For real-time systems, longer quanta mean missed deadlines.")
    print()
    print("  2. I/O-heavy mixed workloads:")
    print("     If short I/O-bound processes must wait behind long CPU-bound")
    print("     processes due to fewer preemptions, I/O devices sit idle longer.")
    print("     This reduces I/O throughput and overall system utilization.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Memory Layout Analysis ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Process State Transitions ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: fork() Output Prediction ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Zombie and Orphan Processes ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Context Switch Cost Estimation ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
