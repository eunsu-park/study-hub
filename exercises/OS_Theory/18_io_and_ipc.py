"""
Exercises for Lesson 18: I/O and IPC
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers I/O method comparison (polling/interrupts/DMA), pipe communication,
shared memory vs message passing, DMA calculation, and socket programming.
"""


# === Exercise 1: I/O Method Comparison ===
# Problem: Suggest appropriate use cases for polling, interrupts, and DMA.

def exercise_1():
    """Compare polling, interrupt-driven, and DMA I/O methods."""
    print("I/O Method Comparison and Use Cases\n")

    methods = [
        {
            "name": "Polling (Programmed I/O)",
            "mechanism": (
                "CPU repeatedly checks device status register in a tight loop. "
                "When device is ready, CPU transfers data one byte/word at a time."
            ),
            "cpu_involvement": "100% -- CPU is busy-waiting the entire time",
            "latency": "Very low (no interrupt overhead, no context switch)",
            "throughput": "Low for large transfers (CPU bound)",
            "use_cases": [
                "10Gbps+ network cards (DPDK-style): interrupt overhead exceeds poll cost",
                "Low-latency trading systems: microsecond response times required",
                "Embedded systems with single-purpose CPUs: CPU has nothing else to do",
                "Very fast devices where data arrives continuously",
            ],
            "avoid_when": "CPU is needed for other tasks; device is slow or intermittent",
        },
        {
            "name": "Interrupt-Driven I/O",
            "mechanism": (
                "CPU initiates I/O, then does other work. Device signals completion "
                "via hardware interrupt. ISR handles the data transfer."
            ),
            "cpu_involvement": "Only during ISR execution (brief)",
            "latency": "Moderate (interrupt delivery + context switch ~1-5us)",
            "throughput": "Good for small transfers; overhead per interrupt for large ones",
            "use_cases": [
                "Keyboard and mouse: infrequent, unpredictable events",
                "Serial ports: low data rate, CPU should do other work",
                "General-purpose I/O on multi-tasking systems",
                "Timer interrupts for preemptive scheduling",
            ],
            "avoid_when": "Very high-frequency events (interrupt storm); very large transfers",
        },
        {
            "name": "DMA (Direct Memory Access)",
            "mechanism": (
                "CPU programs DMA controller with source, destination, and count. "
                "DMA controller transfers data directly between device and memory. "
                "CPU is interrupted only when entire transfer is complete."
            ),
            "cpu_involvement": "Setup + completion interrupt only (minimal)",
            "latency": "Higher initial setup, but much less total CPU time",
            "throughput": "Very high (bus-speed transfers without CPU involvement)",
            "use_cases": [
                "Disk/SSD I/O: large block transfers (4KB-1MB+)",
                "Video capture: continuous high-bandwidth data streams",
                "Network transfers: large packets or bulk data",
                "Audio playback: streaming data to sound card",
            ],
            "avoid_when": "Very small transfers (setup overhead > transfer time)",
        },
    ]

    for m in methods:
        print(f"{'=' * 50}")
        print(f"{m['name']}")
        print(f"{'=' * 50}")
        print(f"  Mechanism: {m['mechanism']}")
        print(f"  CPU involvement: {m['cpu_involvement']}")
        print(f"  Latency: {m['latency']}")
        print(f"  Throughput: {m['throughput']}")
        print(f"  Best use cases:")
        for uc in m['use_cases']:
            print(f"    - {uc}")
        print(f"  Avoid when: {m['avoid_when']}")
        print()

    print("Decision matrix:")
    print(f"  {'Criterion':<30} {'Polling':<15} {'Interrupt':<15} {'DMA'}")
    print("  " + "-" * 70)
    criteria = [
        ("CPU efficiency",    "Poor",  "Good",  "Excellent"),
        ("Response latency",  "Best",  "Good",  "Moderate"),
        ("Large data transfer","Poor", "Moderate","Best"),
        ("Implementation",    "Simple","Moderate","Complex"),
        ("Hardware required",  "None", "PIC/APIC","DMA controller"),
    ]
    for name, p, i, d in criteria:
        print(f"  {name:<30} {p:<15} {i:<15} {d}")


# === Exercise 2: Pipe Communication ===
# Problem: Explain how `cat file.txt | grep "error" | wc -l` works internally.

def exercise_2():
    """Trace pipe communication in a shell pipeline."""
    print('Command: cat file.txt | grep "error" | wc -l\n')

    print("Step 1: Shell creates pipes")
    print("  pipe1 = pipe()  -> [read_fd, write_fd] for cat->grep")
    print("  pipe2 = pipe()  -> [read_fd, write_fd] for grep->wc\n")

    print("Step 2: Shell fork()s three child processes\n")

    processes = [
        {
            "name": "Process 1 (cat)",
            "steps": [
                "Close unused pipe ends (pipe1[0], pipe2[0], pipe2[1])",
                "Redirect stdout -> pipe1[1]  (dup2(pipe1[1], STDOUT_FILENO))",
                "exec('cat', 'file.txt')",
                "cat reads file.txt, writes to stdout (which is now pipe1[1])",
                "When done, exits -> pipe1[1] closed -> EOF for grep",
            ],
        },
        {
            "name": "Process 2 (grep)",
            "steps": [
                "Close unused pipe ends (pipe1[1], pipe2[0])",
                "Redirect stdin  -> pipe1[0]  (dup2(pipe1[0], STDIN_FILENO))",
                "Redirect stdout -> pipe2[1]  (dup2(pipe2[1], STDOUT_FILENO))",
                "exec('grep', 'error')",
                "grep reads from stdin (pipe1), filters lines, writes to stdout (pipe2)",
                "When pipe1 reaches EOF, grep finishes -> pipe2[1] closed",
            ],
        },
        {
            "name": "Process 3 (wc)",
            "steps": [
                "Close unused pipe ends (pipe1[0], pipe1[1], pipe2[1])",
                "Redirect stdin  -> pipe2[0]  (dup2(pipe2[0], STDIN_FILENO))",
                "exec('wc', '-l')",
                "wc reads from stdin (pipe2), counts lines",
                "When pipe2 reaches EOF, wc prints count to stdout (terminal)",
            ],
        },
    ]

    for p in processes:
        print(f"  {p['name']}:")
        for i, step in enumerate(p['steps'], 1):
            print(f"    {i}. {step}")
        print()

    print("Step 3: Data flow\n")
    print("  file.txt -> [cat] -> pipe1 -> [grep 'error'] -> pipe2 -> [wc -l] -> terminal")
    print()

    print("  Kernel buffer in each pipe (default 64KB on Linux):")
    print("  - If pipe is full: writer blocks (backpressure)")
    print("  - If pipe is empty: reader blocks (waiting for data)")
    print("  - This provides automatic flow control between stages\n")

    print("Step 4: Shell waits for all children")
    print("  Parent shell calls waitpid() for each child")
    print("  Returns exit status of the last command (wc) as pipeline result")
    print("  Shell variable $PIPESTATUS (bash) holds exit codes of all stages\n")

    print("Key kernel mechanisms:")
    print("  - pipe() creates a kernel buffer with two file descriptors")
    print("  - dup2() redirects stdin/stdout to pipe endpoints")
    print("  - exec() replaces process image but preserves file descriptors")
    print("  - Closing write end of pipe signals EOF to all readers")
    print("  - Pipes are unidirectional: data flows one way only")


# === Exercise 3: Shared Memory vs Message Passing ===
# Problem: Compare advantages and disadvantages for producer-consumer.

def exercise_3():
    """Compare shared memory and message passing for producer-consumer."""
    print("Producer-Consumer: Shared Memory vs Message Passing\n")

    print("=" * 55)
    print("Shared Memory (shm_open / mmap)")
    print("=" * 55)
    print()

    print("  How it works:")
    print("    1. Producer creates shared memory region")
    print("    2. Both processes mmap() the same physical pages")
    print("    3. Producer writes data directly to shared region")
    print("    4. Consumer reads data directly from shared region")
    print("    5. Synchronization: semaphores or mutexes (user-managed)")
    print()

    shm_pros = [
        "Fastest IPC: no data copying (zero-copy)",
        "After initial setup, no kernel involvement for data transfer",
        "Flexible: any data structure can be shared",
        "Efficient for large data (video frames, large buffers)",
        "Low latency: just write to memory",
    ]
    shm_cons = [
        "Must implement synchronization manually (semaphores, mutexes)",
        "Race condition bugs are easy to introduce",
        "Only works on the same machine (not network-capable)",
        "Complex memory management (who allocates/frees?)",
        "No built-in message boundaries (stream-oriented)",
        "Debugging is harder (shared state)",
    ]

    print("  Advantages:")
    for p in shm_pros:
        print(f"    + {p}")
    print()
    print("  Disadvantages:")
    for c in shm_cons:
        print(f"    - {c}")
    print()

    print("=" * 55)
    print("Message Passing (mq_send / mq_receive or pipe)")
    print("=" * 55)
    print()

    print("  How it works:")
    print("    1. Create message queue (mq_open or pipe)")
    print("    2. Producer sends structured messages (mq_send)")
    print("    3. Kernel copies message from producer to queue buffer")
    print("    4. Consumer receives messages (mq_receive)")
    print("    5. Kernel copies message from queue buffer to consumer")
    print("    6. Synchronization: built-in (kernel manages)")
    print()

    msg_pros = [
        "Built-in synchronization (OS handles send/receive blocking)",
        "Clear message boundaries (structured data units)",
        "Priority support (higher-priority messages delivered first)",
        "Simpler programming model (no shared state to manage)",
        "Can extend to network communication (sockets)",
        "Easier debugging (messages can be logged/inspected)",
    ]
    msg_cons = [
        "Data copy overhead: user->kernel->user (two copies)",
        "Message size limits (e.g., POSIX mq default ~8KB)",
        "Slower than shared memory for large data",
        "Kernel involvement on every send/receive (syscall overhead)",
        "Queue can fill up (need overflow handling)",
    ]

    print("  Advantages:")
    for p in msg_pros:
        print(f"    + {p}")
    print()
    print("  Disadvantages:")
    for c in msg_cons:
        print(f"    - {c}")
    print()

    print("Selection Guide:")
    print(f"  {'Criterion':<30} {'Shared Memory':<20} {'Message Passing'}")
    print("  " + "-" * 65)
    guide = [
        ("Large data / high throughput",  "Preferred",  "Avoid"),
        ("Simple correctness",            "Harder",     "Easier"),
        ("Distributed / networked",       "Not possible","Natural fit"),
        ("Low latency (same machine)",    "Best",       "Good"),
        ("Multiple producers/consumers",  "Complex sync","Built-in"),
        ("Debugging / testing",           "Harder",     "Easier"),
    ]
    for criterion, shm, msg in guide:
        print(f"  {criterion:<30} {shm:<20} {msg}")


# === Exercise 4: DMA Calculation ===
# Problem: Compare CPU usage for DMA vs PIO when reading 1MB from disk.

def exercise_4():
    """Calculate and compare CPU usage for DMA vs PIO."""
    file_size = 1 * 1024 * 1024  # 1MB in bytes
    block_size = 512             # bytes
    pio_cycles_per_block = 100   # CPU cycles per block for PIO
    dma_setup_cycles = 1000      # DMA setup
    dma_interrupt_cycles = 500   # DMA completion interrupt
    cpu_clock_ghz = 1            # 1 GHz

    cpu_clock_hz = cpu_clock_ghz * 1_000_000_000
    num_blocks = file_size // block_size

    print(f"Parameters:")
    print(f"  File size: {file_size // (1024 * 1024)}MB = {file_size:,} bytes")
    print(f"  Block size: {block_size} bytes")
    print(f"  Number of blocks: {file_size:,} / {block_size} = {num_blocks:,}")
    print(f"  CPU clock: {cpu_clock_ghz}GHz = {cpu_clock_hz:,} Hz")
    print(f"  PIO: {pio_cycles_per_block} CPU cycles per block")
    print(f"  DMA: {dma_setup_cycles} cycles setup + {dma_interrupt_cycles} cycles interrupt\n")

    # PIO calculation
    pio_total_cycles = num_blocks * pio_cycles_per_block
    pio_time_s = pio_total_cycles / cpu_clock_hz
    pio_time_ms = pio_time_s * 1000

    print(f"PIO (Programmed I/O):")
    print(f"  CPU cycles = {num_blocks:,} blocks x {pio_cycles_per_block} cycles/block = {pio_total_cycles:,} cycles")
    print(f"  CPU time = {pio_total_cycles:,} / {cpu_clock_hz:,} = {pio_time_ms:.4f} ms")
    print(f"  CPU is 100% occupied during entire transfer\n")

    # DMA calculation
    dma_total_cycles = dma_setup_cycles + dma_interrupt_cycles
    dma_time_s = dma_total_cycles / cpu_clock_hz
    dma_time_ms = dma_time_s * 1000

    print(f"DMA (Direct Memory Access):")
    print(f"  CPU cycles = {dma_setup_cycles} (setup) + {dma_interrupt_cycles} (interrupt) = {dma_total_cycles:,} cycles")
    print(f"  CPU time = {dma_total_cycles:,} / {cpu_clock_hz:,} = {dma_time_ms:.6f} ms")
    print(f"  CPU is free during the actual data transfer!\n")

    # Comparison
    speedup = pio_total_cycles / dma_total_cycles
    savings_pct = (1 - dma_total_cycles / pio_total_cycles) * 100

    print(f"Comparison:")
    print(f"  PIO CPU time:  {pio_time_ms:.4f} ms ({pio_total_cycles:,} cycles)")
    print(f"  DMA CPU time:  {dma_time_ms:.6f} ms ({dma_total_cycles:,} cycles)")
    print(f"  DMA is {speedup:.0f}x more CPU-efficient")
    print(f"  CPU savings: {savings_pct:.2f}%\n")

    # Break-even analysis
    # PIO cost = N * pio_cycles_per_block
    # DMA cost = dma_setup_cycles + dma_interrupt_cycles
    # Break-even: N * pio_cycles = dma_setup + dma_interrupt
    breakeven_blocks = (dma_setup_cycles + dma_interrupt_cycles) / pio_cycles_per_block
    breakeven_bytes = breakeven_blocks * block_size

    print(f"Break-even analysis:")
    print(f"  DMA overhead = {dma_total_cycles} cycles")
    print(f"  PIO cost per block = {pio_cycles_per_block} cycles")
    print(f"  Break-even: {dma_total_cycles} / {pio_cycles_per_block} = {breakeven_blocks} blocks")
    print(f"  = {breakeven_bytes / 1024:.1f} KB")
    print(f"  For transfers > {breakeven_bytes / 1024:.1f} KB, DMA is more efficient")
    print(f"  For transfers < {breakeven_bytes / 1024:.1f} KB, PIO has less overhead")


# === Exercise 5: Socket Programming (TCP vs UDP) ===
# Problem: Compare TCP and UDP sockets with suitable applications.

def exercise_5():
    """Compare TCP and UDP socket characteristics and use cases."""
    print("TCP vs UDP Socket Comparison\n")

    print("=" * 50)
    print("TCP (Transmission Control Protocol)")
    print("=" * 50)
    print()

    tcp_features = [
        ("Connection type",  "Connection-oriented (3-way handshake: SYN, SYN-ACK, ACK)"),
        ("Reliability",      "Guaranteed delivery with acknowledgments and retransmission"),
        ("Ordering",         "Guaranteed in-order delivery (sequence numbers)"),
        ("Flow control",     "Sliding window protocol adjusts to receiver's capacity"),
        ("Congestion control","Built-in (slow start, congestion avoidance, fast recovery)"),
        ("Data model",       "Byte stream (no message boundaries)"),
        ("Overhead",         "Higher (20-byte header minimum, state maintenance)"),
    ]

    print(f"  Features:")
    for feature, desc in tcp_features:
        print(f"    {feature}: {desc}")
    print()

    tcp_apps = [
        "Web browsing (HTTP/HTTPS): page data must arrive complete and in order",
        "Email (SMTP, IMAP): messages must be delivered reliably",
        "File transfer (FTP, SCP): every byte must arrive correctly",
        "Database connections: queries and results must be reliable",
        "SSH: interactive sessions require reliable ordered stream",
    ]
    print(f"  Suitable applications:")
    for app in tcp_apps:
        print(f"    - {app}")
    print()

    print("=" * 50)
    print("UDP (User Datagram Protocol)")
    print("=" * 50)
    print()

    udp_features = [
        ("Connection type",  "Connectionless (no handshake, just send)"),
        ("Reliability",      "No guarantee (packets may be lost, duplicated)"),
        ("Ordering",         "No ordering guarantee (packets may arrive out of order)"),
        ("Flow control",     "None (application must handle)"),
        ("Congestion control","None (application must handle)"),
        ("Data model",       "Datagram (preserves message boundaries)"),
        ("Overhead",         "Lower (8-byte header, no state)"),
    ]

    print(f"  Features:")
    for feature, desc in udp_features:
        print(f"    {feature}: {desc}")
    print()

    udp_apps = [
        "Real-time video/audio streaming: late data is useless, skip and continue",
        "Online games: fresh state updates more important than old ones",
        "DNS queries: single request/response, retransmit if lost",
        "VoIP: real-time voice cannot wait for retransmissions",
        "IoT sensor data: periodic readings where losing one is acceptable",
    ]
    print(f"  Suitable applications:")
    for app in udp_apps:
        print(f"    - {app}")
    print()

    # Side-by-side comparison
    print("Side-by-side comparison:")
    print(f"  {'Property':<25} {'TCP':<25} {'UDP'}")
    print("  " + "-" * 65)
    comparisons = [
        ("Reliability",      "Guaranteed",          "Best-effort"),
        ("Ordering",         "In-order delivery",   "No ordering"),
        ("Speed",            "Slower (overhead)",    "Faster (minimal)"),
        ("Connection setup", "3-way handshake",     "None"),
        ("Header size",      "20+ bytes",           "8 bytes"),
        ("Congestion control","Built-in",           "None"),
        ("Message boundaries","No (stream)",        "Yes (datagram)"),
        ("Use when",         "Correctness matters", "Speed matters"),
    ]
    for prop, tcp, udp in comparisons:
        print(f"  {prop:<25} {tcp:<25} {udp}")

    print()
    print("Socket API comparison (pseudocode):")
    print()
    print("  TCP Server:                    TCP Client:")
    print("    socket(SOCK_STREAM)            socket(SOCK_STREAM)")
    print("    bind(port)                     connect(server_addr)")
    print("    listen(backlog)                send(data)")
    print("    accept() -> new_fd             recv(buffer)")
    print("    recv(buffer)                   close()")
    print("    send(data)")
    print("    close()")
    print()
    print("  UDP Server:                    UDP Client:")
    print("    socket(SOCK_DGRAM)             socket(SOCK_DGRAM)")
    print("    bind(port)                     sendto(data, server_addr)")
    print("    recvfrom(buffer)               recvfrom(buffer)")
    print("    sendto(data, client_addr)      close()")
    print("    close()")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: I/O Method Comparison ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Pipe Communication ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Shared Memory vs Message Passing ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: DMA Calculation ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Socket Programming (TCP vs UDP) ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
