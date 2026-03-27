"""
Exercises for Lesson 21: Disk Scheduling and Modern I/O
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers disk scheduling algorithm comparison (FCFS, SSTF, SCAN, C-SCAN, LOOK),
I/O model latency analysis, and zero-copy transfer benefit calculation.
"""


# === Exercise 1: Disk Scheduling Algorithm Comparison ===
# Problem: Implement FCFS, SSTF, SCAN, C-SCAN, and LOOK algorithms
# and compare total head movement for a given request queue.

def exercise_1():
    """Compare disk scheduling algorithms on the same request queue."""
    requests = [98, 183, 37, 122, 14, 124, 65, 67]
    head_pos = 53
    max_cylinder = 199

    print("Disk Scheduling Algorithm Comparison")
    print(f"Request queue: {requests}")
    print(f"Initial head position: {head_pos}")
    print(f"Disk cylinders: 0 - {max_cylinder}\n")

    def fcfs(reqs, head):
        """First Come First Served."""
        order = list(reqs)
        movement = 0
        current = head
        for r in order:
            movement += abs(r - current)
            current = r
        return movement, order

    def sstf(reqs, head):
        """Shortest Seek Time First."""
        remaining = list(reqs)
        order = []
        movement = 0
        current = head
        while remaining:
            closest = min(remaining, key=lambda r: abs(r - current))
            movement += abs(closest - current)
            current = closest
            order.append(closest)
            remaining.remove(closest)
        return movement, order

    def scan(reqs, head, max_cyl):
        """SCAN (Elevator): move right to end, then left."""
        sorted_reqs = sorted(reqs)
        right = [r for r in sorted_reqs if r >= head]
        left = [r for r in sorted_reqs if r < head]
        left.reverse()

        order = right + [max_cyl] + left
        movement = 0
        current = head
        for r in order:
            movement += abs(r - current)
            current = r

        # Remove max_cyl from display order (it is a boundary, not a request)
        display_order = right + left
        return movement, display_order

    def cscan(reqs, head, max_cyl):
        """C-SCAN: move right to end, jump to 0, continue right."""
        sorted_reqs = sorted(reqs)
        right = [r for r in sorted_reqs if r >= head]
        left = [r for r in sorted_reqs if r < head]

        order = right + [max_cyl, 0] + left
        movement = 0
        current = head
        for r in order:
            movement += abs(r - current)
            current = r

        display_order = right + left
        return movement, display_order

    def look(reqs, head):
        """LOOK: like SCAN but only goes as far as the last request."""
        sorted_reqs = sorted(reqs)
        right = [r for r in sorted_reqs if r >= head]
        left = [r for r in sorted_reqs if r < head]
        left.reverse()

        order = right + left
        movement = 0
        current = head
        for r in order:
            movement += abs(r - current)
            current = r
        return movement, order

    algorithms = [
        ("FCFS", lambda: fcfs(requests, head_pos)),
        ("SSTF", lambda: sstf(requests, head_pos)),
        ("SCAN", lambda: scan(requests, head_pos, max_cylinder)),
        ("C-SCAN", lambda: cscan(requests, head_pos, max_cylinder)),
        ("LOOK", lambda: look(requests, head_pos)),
    ]

    results = []
    for name, algo in algorithms:
        movement, order = algo()
        results.append((name, movement, order))

    for name, movement, order in results:
        path = " -> ".join(str(r) for r in order)
        print(f"--- {name} ---")
        print(f"  Path: {head_pos} -> {path}")
        print(f"  Total head movement: {movement} cylinders\n")

    # Summary ranking
    results.sort(key=lambda x: x[1])
    print("Ranking (least movement first):")
    for rank, (name, movement, _) in enumerate(results, 1):
        print(f"  {rank}. {name}: {movement} cylinders")

    print(f"\nAnalysis:")
    print(f"  SSTF typically minimizes movement but can starve distant requests.")
    print(f"  SCAN/LOOK provide bounded wait times (no starvation).")
    print(f"  C-SCAN provides more uniform wait times than SCAN.")
    print(f"  FCFS is fair but has the highest movement on average.")
    print(f"  For SSDs/NVMe, scheduling barely matters (near-zero seek time).")


# === Exercise 2: I/O Model Latency Analysis ===
# Problem: Compare blocking, non-blocking, epoll, and io_uring I/O patterns
# by modeling request processing throughput.

def exercise_2():
    """Model and compare I/O patterns for a network server."""
    # Parameters
    connection_setup_us = 50       # microseconds
    request_parse_us = 10          # microseconds
    compute_us = 100               # microseconds per request
    disk_io_us = 500               # microseconds (SSD random read)
    response_send_us = 20          # microseconds
    context_switch_us = 5          # microseconds
    epoll_overhead_us = 2          # per event
    uring_overhead_us = 0.5        # per event (shared ring buffer)

    total_requests = 10000
    concurrent_connections = 1000

    print("I/O Model Latency Comparison")
    print(f"Per-request breakdown:")
    print(f"  Connection setup: {connection_setup_us} us")
    print(f"  Request parse:    {request_parse_us} us")
    print(f"  Compute:          {compute_us} us")
    print(f"  Disk I/O:         {disk_io_us} us")
    print(f"  Response send:    {response_send_us} us")
    print(f"  Context switch:   {context_switch_us} us")
    print(f"Total requests: {total_requests:,}")
    print(f"Concurrent connections: {concurrent_connections:,}\n")

    per_request_work = request_parse_us + compute_us + response_send_us

    models = {}

    # Model 1: Blocking I/O (one thread per connection)
    # Each thread blocks during disk I/O
    blocking_per_req = per_request_work + disk_io_us + context_switch_us * 2
    # Thread overhead: context switches scale with thread count
    blocking_thread_overhead = concurrent_connections * context_switch_us
    blocking_total_us = total_requests * blocking_per_req + blocking_thread_overhead
    models["Blocking (1 thread/conn)"] = blocking_total_us

    # Model 2: Non-blocking poll (busy-waiting)
    # CPU spins checking readiness -- wastes CPU but no context switch
    poll_checks_per_io = 50  # average polls before I/O ready
    poll_cost_us = 0.1       # per poll check
    nonblock_per_req = per_request_work + disk_io_us + poll_checks_per_io * poll_cost_us
    nonblock_total_us = total_requests * nonblock_per_req
    models["Non-blocking (poll)"] = nonblock_total_us

    # Model 3: epoll multiplexing
    # Single thread handles all connections via event notification
    epoll_per_req = per_request_work + disk_io_us + epoll_overhead_us
    # epoll_wait batches events -- amortized syscall cost
    epoll_syscall_overhead = (total_requests / 64) * context_switch_us  # batch of 64
    epoll_total_us = total_requests * epoll_per_req + epoll_syscall_overhead
    models["epoll"] = epoll_total_us

    # Model 4: io_uring
    # Zero-copy ring buffer, batched submissions, no syscalls in polled mode
    uring_per_req = per_request_work + disk_io_us + uring_overhead_us
    # Submissions batched, completions via shared memory (no syscall)
    uring_total_us = total_requests * uring_per_req
    models["io_uring"] = uring_total_us

    print(f"{'Model':<30} {'Per-Req (us)':<15} {'Total (ms)':<15} {'RPS':<12} {'vs Best'}")
    print("-" * 80)

    best_total = min(models.values())

    for name, total_us in models.items():
        per_req = total_us / total_requests
        total_ms = total_us / 1000
        rps = total_requests / (total_us / 1e6) if total_us > 0 else 0
        ratio = total_us / best_total
        print(f"{name:<30} {per_req:>10.1f}     {total_ms:>10.1f}     "
              f"{rps:>8,.0f}     {ratio:.2f}x")

    print(f"\nKey insights:")
    print(f"  1. Blocking I/O wastes threads and context switches.")
    print(f"  2. Non-blocking poll wastes CPU cycles spinning.")
    print(f"  3. epoll efficiently multiplexes but still needs syscalls.")
    print(f"  4. io_uring achieves lowest overhead with shared ring buffers.")
    print(f"  5. For NVMe SSDs (low latency), io_uring advantage grows")
    print(f"     because syscall overhead becomes a larger fraction of total time.")


# === Exercise 3: Zero-Copy Transfer Benefit ===
# Problem: Calculate the performance benefit of zero-copy I/O
# (sendfile) vs traditional read+write for file serving.

def exercise_3():
    """Analyze zero-copy I/O benefits for file serving."""
    # System parameters
    syscall_cost_us = 2            # microseconds per syscall
    context_switch_us = 5          # microseconds per switch
    memcpy_bandwidth_gbps = 20     # GB/s memory copy bandwidth
    dma_bandwidth_gbps = 6         # GB/s DMA (disk to memory)
    network_bandwidth_gbps = 10    # GB/s NIC

    file_sizes_mb = [1, 10, 100, 1000]

    print("Zero-Copy I/O Benefit Analysis")
    print(f"Memory copy bandwidth: {memcpy_bandwidth_gbps} GB/s")
    print(f"DMA bandwidth: {dma_bandwidth_gbps} GB/s")
    print(f"Network bandwidth: {network_bandwidth_gbps} GB/s")
    print(f"Syscall cost: {syscall_cost_us} us")
    print(f"Context switch cost: {context_switch_us} us\n")

    def traditional_transfer(size_bytes):
        """read() + write(): 2 copies, 4 context switches."""
        # Step 1: read() syscall -> DMA from disk to kernel buffer
        dma_time = size_bytes / (dma_bandwidth_gbps * 1e9) * 1e6  # us
        # Step 2: kernel copies to user buffer
        copy1_time = size_bytes / (memcpy_bandwidth_gbps * 1e9) * 1e6
        # Step 3: write() syscall -> user buffer copied to kernel socket buffer
        copy2_time = size_bytes / (memcpy_bandwidth_gbps * 1e9) * 1e6
        # Step 4: DMA from socket buffer to NIC
        nic_time = size_bytes / (network_bandwidth_gbps * 1e9) * 1e6

        overhead = syscall_cost_us * 2 + context_switch_us * 4
        total = dma_time + copy1_time + copy2_time + nic_time + overhead
        copies = 2
        switches = 4
        syscalls = 2
        return total, copies, switches, syscalls

    def sendfile_transfer(size_bytes):
        """sendfile(): 0 user copies, 2 context switches."""
        # DMA from disk to kernel buffer
        dma_time = size_bytes / (dma_bandwidth_gbps * 1e9) * 1e6
        # DMA from kernel buffer directly to NIC (no user copy!)
        nic_time = size_bytes / (network_bandwidth_gbps * 1e9) * 1e6

        overhead = syscall_cost_us * 1 + context_switch_us * 2
        total = dma_time + nic_time + overhead
        copies = 0
        switches = 2
        syscalls = 1
        return total, copies, switches, syscalls

    def mmap_transfer(size_bytes):
        """mmap() + write(): 1 copy, 4 context switches."""
        # DMA from disk to kernel buffer (shared via mmap)
        dma_time = size_bytes / (dma_bandwidth_gbps * 1e9) * 1e6
        # write() copies from mmap'd region to socket buffer
        copy_time = size_bytes / (memcpy_bandwidth_gbps * 1e9) * 1e6
        # DMA from socket buffer to NIC
        nic_time = size_bytes / (network_bandwidth_gbps * 1e9) * 1e6

        overhead = syscall_cost_us * 2 + context_switch_us * 4
        total = dma_time + copy_time + nic_time + overhead
        copies = 1
        switches = 4
        syscalls = 2
        return total, copies, switches, syscalls

    methods = [
        ("read+write", traditional_transfer),
        ("sendfile", sendfile_transfer),
        ("mmap+write", mmap_transfer),
    ]

    for size_mb in file_sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        size_label = f"{size_mb} MB" if size_mb < 1024 else f"{size_mb // 1024} GB"

        print(f"--- File size: {size_label} ---")
        print(f"  {'Method':<16} {'Time (us)':<14} {'Copies':<8} "
              f"{'Switches':<10} {'Syscalls':<10} {'Throughput'}")
        print("  " + "-" * 70)

        for name, func in methods:
            total, copies, switches, syscalls = func(size_bytes)
            throughput_gbps = (size_bytes * 8) / (total * 1e-6) / 1e9

            print(f"  {name:<16} {total:>10.1f}     {copies:<8} "
                  f"{switches:<10} {syscalls:<10} {throughput_gbps:.2f} Gbps")
        print()

    # Summary
    print("Summary:")
    print("  sendfile advantages:")
    print("    - Zero user-space copies (data stays in kernel)")
    print("    - Fewer context switches (1 syscall vs 2)")
    print("    - CPU freed for other work (no memcpy)")
    print("    - Best for static file serving (nginx, web servers)")
    print()
    print("  When NOT to use sendfile:")
    print("    - Data needs modification before sending (compression, encryption)")
    print("    - Scatter-gather I/O patterns")
    print("    - Non-file data sources")
    print()
    print("  Modern alternative: io_uring with IORING_OP_SPLICE")
    print("    - Combines zero-copy with async I/O")
    print("    - Even fewer syscalls (batched submission)")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Disk Scheduling Algorithm Comparison ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: I/O Model Latency Analysis ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Zero-Copy Transfer Benefit ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
