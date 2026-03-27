"""
Exercises for Lesson 24: eBPF and Kernel Tracing
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers eBPF program flow simulation, BPF map operations,
and XDP packet processing pipeline modeling.
"""


# === Exercise 1: eBPF Program Lifecycle Simulation ===
# Problem: Model the eBPF program lifecycle including verification,
# JIT compilation, attachment to hooks, and map interaction.

def exercise_1():
    """Simulate the eBPF program lifecycle and verification process."""

    class BPFVerifier:
        """Simulate the eBPF verifier's safety checks."""

        MAX_INSTRUCTIONS = 1_000_000
        MAX_STACK_DEPTH = 512  # bytes
        VALID_HELPERS = {
            "bpf_get_current_pid_tgid", "bpf_get_current_comm",
            "bpf_probe_read_user_str", "bpf_map_lookup_elem",
            "bpf_map_update_elem", "bpf_map_delete_elem",
            "bpf_perf_event_output", "bpf_ktime_get_ns",
            "bpf_trace_printk", "bpf_ringbuf_output",
        }

        def verify(self, program):
            """Run verification checks on a BPF program."""
            results = []

            # Check 1: Instruction count
            if program["instruction_count"] > self.MAX_INSTRUCTIONS:
                results.append(("FAIL", "instruction_count",
                                f"Too many instructions: {program['instruction_count']} "
                                f"> {self.MAX_INSTRUCTIONS}"))
            else:
                results.append(("PASS", "instruction_count",
                                f"{program['instruction_count']} instructions"))

            # Check 2: No unbounded loops
            if program.get("has_unbounded_loop", False):
                results.append(("FAIL", "bounded_loops",
                                "Contains unbounded loop (infinite loop risk)"))
            else:
                results.append(("PASS", "bounded_loops",
                                "All loops are bounded"))

            # Check 3: Stack usage
            if program.get("stack_usage", 0) > self.MAX_STACK_DEPTH:
                results.append(("FAIL", "stack_depth",
                                f"Stack usage {program['stack_usage']} > "
                                f"{self.MAX_STACK_DEPTH} bytes"))
            else:
                results.append(("PASS", "stack_depth",
                                f"{program.get('stack_usage', 0)} bytes used"))

            # Check 4: Memory safety (no out-of-bounds access)
            if program.get("has_oob_access", False):
                results.append(("FAIL", "memory_safety",
                                "Potential out-of-bounds memory access"))
            else:
                results.append(("PASS", "memory_safety",
                                "All memory accesses verified safe"))

            # Check 5: Valid helper functions
            invalid_helpers = [h for h in program.get("helpers_used", [])
                               if h not in self.VALID_HELPERS]
            if invalid_helpers:
                results.append(("FAIL", "helper_functions",
                                f"Invalid helpers: {invalid_helpers}"))
            else:
                results.append(("PASS", "helper_functions",
                                f"All {len(program.get('helpers_used', []))} helpers valid"))

            # Check 6: Program terminates (all paths reach exit)
            if not program.get("all_paths_terminate", True):
                results.append(("FAIL", "termination",
                                "Not all code paths reach program exit"))
            else:
                results.append(("PASS", "termination",
                                "All paths terminate"))

            return results

    # Test programs
    programs = [
        {
            "name": "syscall_tracer (valid)",
            "type": "BPF_PROG_TYPE_TRACEPOINT",
            "instruction_count": 45,
            "stack_usage": 128,
            "has_unbounded_loop": False,
            "has_oob_access": False,
            "all_paths_terminate": True,
            "helpers_used": [
                "bpf_get_current_pid_tgid",
                "bpf_get_current_comm",
                "bpf_perf_event_output",
            ],
        },
        {
            "name": "malicious_loop (rejected)",
            "type": "BPF_PROG_TYPE_KPROBE",
            "instruction_count": 200,
            "stack_usage": 64,
            "has_unbounded_loop": True,
            "has_oob_access": False,
            "all_paths_terminate": False,
            "helpers_used": ["bpf_trace_printk"],
        },
        {
            "name": "buffer_overflow (rejected)",
            "type": "BPF_PROG_TYPE_XDP",
            "instruction_count": 80,
            "stack_usage": 256,
            "has_unbounded_loop": False,
            "has_oob_access": True,
            "all_paths_terminate": True,
            "helpers_used": ["bpf_map_lookup_elem", "kernel_write"],
        },
    ]

    verifier = BPFVerifier()

    print("eBPF Program Lifecycle Simulation\n")

    for prog in programs:
        print(f"--- Program: {prog['name']} ---")
        print(f"  Type: {prog['type']}")
        print(f"  Instructions: {prog['instruction_count']}")
        print(f"  Helpers: {', '.join(prog['helpers_used'])}\n")

        # Step 1: Verification
        print(f"  Step 1: Verification")
        results = verifier.verify(prog)
        all_passed = True
        for status, check, detail in results:
            marker = "+" if status == "PASS" else "X"
            print(f"    [{marker}] {check}: {detail}")
            if status == "FAIL":
                all_passed = False

        if all_passed:
            print(f"\n  Verification: PASSED")

            # Step 2: JIT compilation
            jit_size = prog["instruction_count"] * 8  # ~8 bytes per native instruction
            print(f"\n  Step 2: JIT Compilation")
            print(f"    BPF bytecode -> native x86_64")
            print(f"    Output size: {jit_size} bytes")

            # Step 3: Attachment
            hook_map = {
                "BPF_PROG_TYPE_KPROBE": "kprobe:sys_open",
                "BPF_PROG_TYPE_TRACEPOINT": "tracepoint:raw_syscalls:sys_enter",
                "BPF_PROG_TYPE_XDP": "xdp:eth0",
            }
            hook = hook_map.get(prog["type"], "unknown")
            print(f"\n  Step 3: Attach to hook")
            print(f"    Hook: {hook}")
            print(f"    Status: ACTIVE")
        else:
            print(f"\n  Verification: REJECTED")
            print(f"  Program will NOT be loaded into kernel.")
        print()

    print("Why the verifier matters:")
    print("  - Kernel crashes affect ALL processes (not just the BPF user)")
    print("  - No undo: a rogue kernel module can corrupt memory permanently")
    print("  - eBPF verifier guarantees: no crashes, no infinite loops,")
    print("    no unauthorized memory access, bounded execution time")


# === Exercise 2: BPF Map Operations ===
# Problem: Implement common BPF map types (hash, array, LRU hash)
# and demonstrate their use for kernel-userspace data sharing.

def exercise_2():
    """Simulate BPF map operations for data collection."""

    class BPFHashMap:
        """Simulate BPF_MAP_TYPE_HASH."""
        def __init__(self, name, max_entries):
            self.name = name
            self.max_entries = max_entries
            self.data = {}

        def lookup(self, key):
            return self.data.get(key)

        def update(self, key, value):
            if len(self.data) >= self.max_entries and key not in self.data:
                return False  # Map full
            self.data[key] = value
            return True

        def delete(self, key):
            if key in self.data:
                del self.data[key]
                return True
            return False

        def items(self):
            return self.data.items()

    class BPFArrayMap:
        """Simulate BPF_MAP_TYPE_ARRAY."""
        def __init__(self, name, max_entries):
            self.name = name
            self.max_entries = max_entries
            self.data = [0] * max_entries

        def lookup(self, index):
            if 0 <= index < self.max_entries:
                return self.data[index]
            return None

        def update(self, index, value):
            if 0 <= index < self.max_entries:
                self.data[index] = value
                return True
            return False

    class BPFLRUHashMap:
        """Simulate BPF_MAP_TYPE_LRU_HASH."""
        def __init__(self, name, max_entries):
            self.name = name
            self.max_entries = max_entries
            self.data = {}
            self.access_order = []

        def _touch(self, key):
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

        def lookup(self, key):
            if key in self.data:
                self._touch(key)
                return self.data[key]
            return None

        def update(self, key, value):
            if key in self.data:
                self.data[key] = value
                self._touch(key)
                return True, None
            if len(self.data) >= self.max_entries:
                # Evict LRU entry
                evicted_key = self.access_order.pop(0)
                del self.data[evicted_key]
                self.data[key] = value
                self.access_order.append(key)
                return True, evicted_key
            self.data[key] = value
            self.access_order.append(key)
            return True, None

    print("BPF Map Operations Simulation\n")

    # Scenario 1: Syscall counter using hash map
    print("--- Scenario 1: Syscall Counter (Hash Map) ---\n")
    syscall_count = BPFHashMap("syscall_count", max_entries=1024)

    # Simulate kernel-side: counting syscalls by PID
    events = [
        (1001, "read"), (1001, "write"), (1002, "open"),
        (1001, "read"), (1003, "mmap"), (1002, "read"),
        (1001, "write"), (1001, "read"), (1003, "close"),
        (1002, "write"), (1001, "read"), (1001, "close"),
    ]

    for pid, syscall in events:
        current = syscall_count.lookup(pid)
        if current is None:
            syscall_count.update(pid, {"total": 0})
            current = syscall_count.lookup(pid)
        current["total"] += 1

    # Simulate userspace: reading results
    print(f"  Syscall counts by PID:")
    print(f"  {'PID':<8} {'Count'}")
    print("  " + "-" * 20)
    for pid, stats in sorted(syscall_count.items(), key=lambda x: -x[1]["total"]):
        print(f"  {pid:<8} {stats['total']}")

    # Scenario 2: Per-CPU histogram using array map
    print(f"\n--- Scenario 2: Latency Histogram (Array Map) ---\n")
    # Bucket indices: 0=<1us, 1=1-2us, 2=2-4us, 3=4-8us, etc. (log2 buckets)
    num_buckets = 12
    latency_hist = BPFArrayMap("latency_hist", max_entries=num_buckets)

    import random
    random.seed(42)

    # Simulate latency measurements (in microseconds)
    latencies = [random.expovariate(1 / 5.0) for _ in range(1000)]

    for lat in latencies:
        if lat < 1:
            bucket = 0
        else:
            bucket = min(int(lat).bit_length(), num_buckets - 1)
        current = latency_hist.lookup(bucket)
        latency_hist.update(bucket, current + 1)

    print(f"  Read latency histogram (1000 samples):")
    bucket_labels = ["<1us", "1-2us", "2-4us", "4-8us", "8-16us",
                     "16-32us", "32-64us", "64-128us", "128-256us",
                     "256-512us", "512us-1ms", ">1ms"]

    max_count = max(latency_hist.lookup(i) for i in range(num_buckets))
    bar_width = 40

    for i in range(num_buckets):
        count = latency_hist.lookup(i)
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        label = bucket_labels[i] if i < len(bucket_labels) else f"bucket_{i}"
        print(f"  {label:>12}: {bar:<{bar_width}} {count}")

    # Scenario 3: Connection tracking with LRU hash
    print(f"\n--- Scenario 3: Connection Tracking (LRU Hash Map) ---\n")
    conn_track = BPFLRUHashMap("connections", max_entries=4)

    connections = [
        ("10.0.0.1:5000", "established"),
        ("10.0.0.2:5001", "established"),
        ("10.0.0.3:5002", "syn_sent"),
        ("10.0.0.4:5003", "established"),
        # This one should evict the LRU entry
        ("10.0.0.5:5004", "established"),
        # Access an existing one to keep it alive
        ("10.0.0.4:5003", "established"),
        # This evicts next LRU
        ("10.0.0.6:5005", "syn_sent"),
    ]

    for conn, state in connections:
        success, evicted = conn_track.update(conn, state)
        if evicted:
            print(f"  Add {conn} ({state}) -> evicted LRU entry: {evicted}")
        else:
            print(f"  Add {conn} ({state})")

    print(f"\n  Final map contents ({len(conn_track.data)}/{conn_track.max_entries}):")
    for key, value in conn_track.data.items():
        print(f"    {key}: {value}")
    print(f"  Access order (LRU -> MRU): {conn_track.access_order}")


# === Exercise 3: XDP Packet Processing Pipeline ===
# Problem: Simulate an XDP program that implements basic firewall rules,
# rate limiting, and packet counting.

def exercise_3():
    """Simulate XDP packet processing with firewall and rate limiting."""
    # XDP actions
    XDP_PASS = "XDP_PASS"
    XDP_DROP = "XDP_DROP"
    XDP_TX = "XDP_TX"        # bounce back
    XDP_REDIRECT = "XDP_REDIRECT"

    # Firewall rules
    blocked_ips = {"10.0.0.99", "192.168.1.200", "172.16.0.50"}

    # Rate limiting: max packets per source IP per second
    rate_limit = 5

    # Statistics
    stats = {
        "total": 0,
        "passed": 0,
        "dropped_firewall": 0,
        "dropped_rate": 0,
        "dropped_invalid": 0,
    }

    # Per-IP rate tracking (simulated BPF map)
    rate_map = {}

    # Per-protocol counter (simulated BPF map)
    proto_count = {"TCP": 0, "UDP": 0, "ICMP": 0, "OTHER": 0}

    def xdp_process(packet):
        """Simulate XDP packet processing."""
        stats["total"] += 1

        # Step 1: Parse packet (bounds checking)
        if not packet.get("valid", True):
            stats["dropped_invalid"] += 1
            return XDP_DROP, "invalid packet"

        src_ip = packet["src_ip"]
        dst_ip = packet["dst_ip"]
        proto = packet["protocol"]
        timestamp = packet["timestamp"]

        # Step 2: Protocol counting
        if proto in proto_count:
            proto_count[proto] += 1
        else:
            proto_count["OTHER"] += 1

        # Step 3: Firewall check
        if src_ip in blocked_ips:
            stats["dropped_firewall"] += 1
            return XDP_DROP, f"blocked IP: {src_ip}"

        # Step 4: Rate limiting
        if src_ip not in rate_map:
            rate_map[src_ip] = {"count": 0, "window_start": timestamp}

        entry = rate_map[src_ip]
        # Reset window every second
        if timestamp - entry["window_start"] >= 1.0:
            entry["count"] = 0
            entry["window_start"] = timestamp

        entry["count"] += 1
        if entry["count"] > rate_limit:
            stats["dropped_rate"] += 1
            return XDP_DROP, f"rate limit exceeded: {src_ip}"

        # Step 5: Pass to network stack
        stats["passed"] += 1
        return XDP_PASS, "allowed"

    # Generate test packets
    packets = [
        # Normal traffic
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.0, "valid": True},
        {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.100", "protocol": "UDP",
         "timestamp": 0.1, "valid": True},
        {"src_ip": "10.0.0.3", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.2, "valid": True},
        # Blocked IP
        {"src_ip": "10.0.0.99", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.3, "valid": True},
        {"src_ip": "192.168.1.200", "dst_ip": "10.0.0.100", "protocol": "ICMP",
         "timestamp": 0.4, "valid": True},
        # Invalid packet
        {"src_ip": "???", "dst_ip": "???", "protocol": "???",
         "timestamp": 0.5, "valid": False},
        # Rate limit test: 10.0.0.1 sends 7 more packets (exceeds limit of 5)
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.5, "valid": True},
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.6, "valid": True},
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.7, "valid": True},
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.8, "valid": True},
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.85, "valid": True},
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 0.9, "valid": True},
        # After rate window resets
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.100", "protocol": "TCP",
         "timestamp": 1.1, "valid": True},
    ]

    print("XDP Packet Processing Pipeline Simulation\n")
    print(f"  Firewall blocked IPs: {blocked_ips}")
    print(f"  Rate limit: {rate_limit} packets/IP/second\n")

    print(f"  {'#':<4} {'Source IP':<18} {'Proto':<8} {'Time':<8} "
          f"{'Action':<12} {'Reason'}")
    print("  " + "-" * 70)

    for i, pkt in enumerate(packets):
        action, reason = xdp_process(pkt)
        src = pkt["src_ip"] if pkt.get("valid") else "(invalid)"
        proto = pkt["protocol"] if pkt.get("valid") else "???"
        marker = " ***" if action == XDP_DROP else ""
        print(f"  {i + 1:<4} {src:<18} {proto:<8} {pkt['timestamp']:<8.1f} "
              f"{action:<12} {reason}{marker}")

    # Summary
    print(f"\n--- Processing Summary ---\n")
    print(f"  Total packets:      {stats['total']}")
    print(f"  Passed:             {stats['passed']}")
    print(f"  Dropped (firewall): {stats['dropped_firewall']}")
    print(f"  Dropped (rate):     {stats['dropped_rate']}")
    print(f"  Dropped (invalid):  {stats['dropped_invalid']}")
    drop_total = stats['dropped_firewall'] + stats['dropped_rate'] + stats['dropped_invalid']
    print(f"  Drop rate:          {drop_total / stats['total'] * 100:.1f}%")

    print(f"\n  Protocol distribution:")
    for proto, count in sorted(proto_count.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {proto}: {count}")

    print(f"\n  XDP performance advantages:")
    print(f"    - Processes packets BEFORE sk_buff allocation (kernel fast path)")
    print(f"    - 24M packets/sec per core (vs ~1M for iptables)")
    print(f"    - Zero memory allocation for dropped packets")
    print(f"    - Used by Cloudflare, Facebook (Katran), Cilium for DDoS mitigation")
    print(f"    - Can offload to NIC hardware (XDP offload mode)")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: eBPF Program Lifecycle Simulation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: BPF Map Operations ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: XDP Packet Processing Pipeline ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
