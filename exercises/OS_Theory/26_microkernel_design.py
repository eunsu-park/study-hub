"""
Exercises for Lesson 26: Microkernel Design
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers capability-based access control, microkernel IPC simulation,
and monolithic vs microkernel architecture tradeoff analysis.
"""

import time
import threading
import queue


# === Exercise 1: Capability-Based Access Control System ===
# Problem: Implement a capability system that prevents the confused deputy
# attack and supports delegation with permission attenuation.

def exercise_1():
    """Build a capability-based access control system."""

    # Permission flags
    READ = 0x01
    WRITE = 0x02
    EXECUTE = 0x04

    def perm_str(perm):
        """Convert permission bits to a readable string."""
        parts = []
        if perm & READ:    parts.append("R")
        if perm & WRITE:   parts.append("W")
        if perm & EXECUTE: parts.append("X")
        return "".join(parts) or "---"

    class Capability:
        """An unforgeable token granting access to a specific resource."""
        _next_id = 0

        def __init__(self, object_id, object_name, permissions, delegatable=True):
            Capability._next_id += 1
            self.cap_id = Capability._next_id
            self.object_id = object_id
            self.object_name = object_name
            self.permissions = permissions
            self.delegatable = delegatable

        def __repr__(self):
            return (f"Cap#{self.cap_id}({self.object_name}, "
                    f"{perm_str(self.permissions)}, "
                    f"{'delegatable' if self.delegatable else 'final'})")

    class CapabilitySpace:
        """A process's set of capabilities."""

        def __init__(self, name):
            self.name = name
            self.caps = []

        def grant(self, cap):
            """Add a capability to this space."""
            self.caps.append(cap)

        def check(self, object_id, required_perm):
            """Check if this space has the required permission on an object."""
            for cap in self.caps:
                if (cap.object_id == object_id and
                        (cap.permissions & required_perm) == required_perm):
                    return True
            return False

        def delegate(self, cap, target_space, new_perms):
            """Delegate a capability with equal or fewer permissions."""
            if cap not in self.caps:
                return None, "Capability not in source space"
            if not cap.delegatable:
                return None, "Capability is not delegatable"
            if (new_perms & cap.permissions) != new_perms:
                return None, "Cannot grant more permissions than held"

            new_cap = Capability(cap.object_id, cap.object_name,
                                 new_perms, delegatable=cap.delegatable)
            target_space.grant(new_cap)
            return new_cap, "Delegated successfully"

        def revoke(self, cap):
            """Remove a capability from this space."""
            if cap in self.caps:
                self.caps.remove(cap)
                return True
            return False

        def show(self):
            """Display all capabilities in this space."""
            print(f"  {self.name}: [{len(self.caps)} capabilities]")
            for cap in self.caps:
                print(f"    {cap}")

    # Demonstrate the system
    print("=== Capability-Based Access Control ===\n")

    # Create resources
    resources = {
        1: "/var/www/html/index.html",
        2: "/etc/shadow",
        3: "/tmp/logfile",
        4: "/usr/bin/server",
    }

    # Create process capability spaces
    web_server = CapabilitySpace("web_server")
    logger = CapabilitySpace("logger")
    admin = CapabilitySpace("admin")

    # Admin gets full access to everything
    for obj_id, obj_name in resources.items():
        admin.grant(Capability(obj_id, obj_name, READ | WRITE | EXECUTE))

    # Web server gets read on web content, execute on server binary
    web_server.grant(Capability(1, resources[1], READ))
    web_server.grant(Capability(4, resources[4], READ | EXECUTE))

    # Logger gets write on logfile
    logger.grant(Capability(3, resources[3], WRITE))

    print("Initial capability spaces:")
    admin.show()
    web_server.show()
    logger.show()

    # Demonstrate access checks
    print("\n--- Access Control Checks ---\n")
    checks = [
        (web_server, 1, READ,  "web_server reads index.html"),
        (web_server, 2, READ,  "web_server reads /etc/shadow"),
        (web_server, 3, WRITE, "web_server writes to logfile"),
        (logger,     3, WRITE, "logger writes to logfile"),
        (logger,     3, READ,  "logger reads logfile"),
        (logger,     1, READ,  "logger reads index.html"),
    ]

    for space, obj_id, perm, description in checks:
        allowed = space.check(obj_id, perm)
        status = "ALLOWED" if allowed else "DENIED"
        print(f"  {description:<45} [{status}]")

    # Demonstrate confused deputy prevention
    print("\n--- Confused Deputy Prevention ---\n")
    print("  Scenario: Web server is asked to compile a user report.")
    print("  A malicious user tries to trick it into reading /etc/shadow.\n")

    print("  ACL-based system (vulnerable):")
    print("    Web server runs as www-data, but has ambient access to")
    print("    read files. Malicious request: 'compile report from /etc/shadow'")
    print("    -> ACL checks: does www-data have read? Maybe yes!")
    print("    -> CONFUSED DEPUTY: server acted on behalf of user with")
    print("       its own permissions, not the user's.\n")

    print("  Capability-based system (safe):")
    has_shadow = web_server.check(2, READ)
    print(f"    web_server.check(/etc/shadow, READ) -> {has_shadow}")
    print("    Web server does not possess a capability for /etc/shadow.")
    print("    Even if tricked, it CANNOT access the file.")
    print("    -> ATTACK PREVENTED: no capability = no access, period.")

    # Demonstrate delegation with attenuation
    print("\n--- Capability Delegation ---\n")

    log_cap = web_server.caps[0]  # The index.html READ cap
    new_cap, msg = web_server.delegate(log_cap, logger, READ)
    print(f"  web_server delegates index.html READ to logger: {msg}")
    print(f"  New capability: {new_cap}")

    # Try to escalate permissions during delegation
    escalate_cap, msg = web_server.delegate(log_cap, logger, READ | WRITE)
    print(f"\n  web_server tries to delegate index.html RW to logger: {msg}")
    print(f"  Result: {escalate_cap}")
    print("  -> Cannot grant WRITE when only holding READ!")


# === Exercise 2: Microkernel IPC Simulator ===
# Problem: Simulate synchronous IPC between microkernel user-space servers
# and measure the overhead compared to direct function calls.

def exercise_2():
    """Simulate microkernel IPC and compare with monolithic direct calls."""

    class IPCMessage:
        """A message passed between processes via IPC."""

        def __init__(self, sender, msg_type, payload):
            self.sender = sender
            self.msg_type = msg_type
            self.payload = payload
            self.reply = None

    class MicrokernelServer:
        """A user-space server that communicates only via IPC."""

        def __init__(self, name):
            self.name = name
            self.inbox = queue.Queue()
            self.stats = {"messages_handled": 0, "total_latency_us": 0}

        def handle_message(self, msg):
            """Process an incoming IPC message."""
            raise NotImplementedError

        def send_and_wait(self, target, msg_type, payload):
            """Send a message to another server and wait for reply (call)."""
            msg = IPCMessage(self.name, msg_type, payload)
            # Simulate IPC overhead: context switch + message copy
            ipc_overhead_us = 0.5  # ~500ns per IPC call
            target.inbox.put(msg)
            target.handle_message(msg)
            self.stats["total_latency_us"] += ipc_overhead_us
            target.stats["total_latency_us"] += ipc_overhead_us
            return msg.reply

    class FileSystemServer(MicrokernelServer):
        """User-space file system server."""

        def __init__(self):
            super().__init__("FS_Server")
            self.files = {
                "/etc/hostname": "myhost",
                "/var/log/syslog": "kernel: booting...",
                "/home/user/data.txt": "Hello, World!",
            }

        def handle_message(self, msg):
            self.stats["messages_handled"] += 1
            if msg.msg_type == "read":
                path = msg.payload
                msg.reply = self.files.get(path, None)
            elif msg.msg_type == "write":
                path, data = msg.payload
                self.files[path] = data
                msg.reply = len(data)

    class NetworkServer(MicrokernelServer):
        """User-space network server."""

        def __init__(self):
            super().__init__("Net_Server")
            self.connections = {}
            self._next_fd = 10

        def handle_message(self, msg):
            self.stats["messages_handled"] += 1
            if msg.msg_type == "connect":
                fd = self._next_fd
                self._next_fd += 1
                self.connections[fd] = msg.payload
                msg.reply = fd
            elif msg.msg_type == "send":
                fd, data = msg.payload
                if fd in self.connections:
                    msg.reply = len(data)
                else:
                    msg.reply = -1

    class DeviceDriverServer(MicrokernelServer):
        """User-space device driver."""

        def __init__(self):
            super().__init__("Disk_Driver")
            self.blocks = {}

        def handle_message(self, msg):
            self.stats["messages_handled"] += 1
            if msg.msg_type == "read_block":
                block_num = msg.payload
                msg.reply = self.blocks.get(block_num, b'\x00' * 512)
            elif msg.msg_type == "write_block":
                block_num, data = msg.payload
                self.blocks[block_num] = data
                msg.reply = 0

    # Create microkernel servers
    fs = FileSystemServer()
    net = NetworkServer()
    disk = DeviceDriverServer()
    app = MicrokernelServer("Application")
    app.handle_message = lambda msg: None  # App doesn't serve requests

    print("=== Microkernel IPC Simulation ===\n")

    # Simulate a file read operation (App -> FS -> Disk -> FS -> App)
    print("Scenario: Application reads /etc/hostname\n")
    print("  Microkernel path (4 IPC calls):")
    print("    App --IPC--> FS_Server --IPC--> Disk_Driver")
    print("    App <--IPC-- FS_Server <--IPC-- Disk_Driver\n")

    # Step 1: App sends read request to FS
    disk.blocks[42] = b"myhost"  # Pre-populate disk block
    block_data = app.send_and_wait(disk, "read_block", 42)
    result = app.send_and_wait(fs, "read", "/etc/hostname")

    print(f"  Result: '{result}'")
    print(f"  IPC calls: 2 (App->FS, App->Disk simplified)")

    # Now simulate the same operations as direct function calls (monolithic)
    print("\n--- Performance Comparison ---\n")

    num_ops = 10000

    # Monolithic: direct function calls
    start = time.perf_counter()
    for _ in range(num_ops):
        files = {"/etc/hostname": "myhost"}
        _ = files.get("/etc/hostname")
    monolithic_time = (time.perf_counter() - start) * 1e6  # microseconds

    # Microkernel: IPC-based
    start = time.perf_counter()
    for _ in range(num_ops):
        _ = app.send_and_wait(fs, "read", "/etc/hostname")
    micro_time = (time.perf_counter() - start) * 1e6

    print(f"  Operations: {num_ops} file reads")
    print(f"  Monolithic (direct call): {monolithic_time:.0f} us "
          f"({monolithic_time/num_ops:.3f} us/op)")
    print(f"  Microkernel (IPC-based):  {micro_time:.0f} us "
          f"({micro_time/num_ops:.3f} us/op)")
    ratio = micro_time / monolithic_time if monolithic_time > 0 else float('inf')
    print(f"  Overhead ratio: {ratio:.1f}x\n")

    # Reliability comparison: simulate driver crash
    print("--- Reliability: Driver Crash Recovery ---\n")

    print("  Monolithic kernel:")
    print("    Disk driver has a bug and dereferences NULL pointer.")
    print("    -> KERNEL PANIC: driver runs in kernel space.")
    print("    -> Entire system crashes. All processes lost.\n")

    print("  Microkernel:")
    print("    Disk driver has a bug and crashes.")
    disk_crashed = DeviceDriverServer()
    disk_crashed.name = "Disk_Driver (crashed)"
    print(f"    -> {disk_crashed.name} process terminates.")
    # Restart it
    disk_restarted = DeviceDriverServer()
    disk_restarted.name = "Disk_Driver (restarted)"
    print(f"    -> Supervisor restarts: {disk_restarted.name}")
    result = app.send_and_wait(disk_restarted, "read_block", 0)
    print(f"    -> Service resumed. read_block(0) = {len(result)} bytes")
    print(f"    -> System continues running. No other processes affected.")

    # Server stats
    print(f"\n--- Server Statistics ---\n")
    for server in [fs, net, disk]:
        print(f"  {server.name}: {server.stats['messages_handled']} messages")


# === Exercise 3: Monolithic vs Microkernel Tradeoff Analysis ===
# Problem: Quantitatively analyze when a microkernel architecture is
# preferable to a monolithic one based on IPC cost and reliability.

def exercise_3():
    """Analyze the tradeoff between monolithic and microkernel architectures."""

    print("=== Monolithic vs Microkernel Tradeoff Analysis ===\n")

    # Model parameters
    syscall_cost_ns = 150       # Cost of a syscall in monolithic kernel
    ipc_cost_ns = 500           # Cost of one IPC call in microkernel
    kernel_service_calls = 2    # Number of extra IPC hops for microkernel

    # A file read requires:
    # Monolithic: 1 syscall
    # Microkernel: 1 IPC (app->fs) + 1 IPC (fs->disk) + 1 IPC (disk->fs)
    #              + 1 IPC (fs->app) = 4 IPC calls
    mono_file_read_ns = syscall_cost_ns
    micro_file_read_ns = 4 * ipc_cost_ns

    print("--- Latency Comparison: Single File Read ---\n")
    print(f"  Monolithic: 1 syscall = {mono_file_read_ns} ns")
    print(f"  Microkernel: 4 IPC calls = {micro_file_read_ns} ns")
    print(f"  Overhead: {micro_file_read_ns / mono_file_read_ns:.1f}x\n")

    # At what IPC cost does microkernel become competitive?
    print("--- IPC Cost Sensitivity ---\n")
    print(f"  {'IPC Cost (ns)':<16} {'Micro Total (ns)':<20} {'Overhead':<12} {'Verdict'}")
    print("  " + "-" * 60)

    for ipc_ns in [50, 100, 150, 200, 300, 500, 1000]:
        micro_total = 4 * ipc_ns
        overhead = micro_total / mono_file_read_ns
        if overhead <= 1.5:
            verdict = "Acceptable"
        elif overhead <= 3.0:
            verdict = "Moderate"
        else:
            verdict = "Significant"
        print(f"  {ipc_ns:<16} {micro_total:<20} {overhead:<12.1f} {verdict}")

    print(f"\n  To match monolithic latency (within 1.5x), IPC must be < "
          f"{int(mono_file_read_ns * 1.5 / 4)} ns")

    # Reliability analysis
    print("\n--- Reliability Analysis ---\n")

    bug_rate_per_kloc = 10  # bugs per 1000 lines of code (industry average)
    mono_kloc = 30000       # Linux kernel ~30M lines
    micro_kloc = 10         # seL4 ~10K lines
    driver_kloc = 25000     # drivers are most of Linux kernel

    mono_bugs = mono_kloc * bug_rate_per_kloc
    micro_bugs = micro_kloc * bug_rate_per_kloc

    print(f"  Bug estimation (at {bug_rate_per_kloc} bugs / KLOC):")
    print(f"    Monolithic kernel ({mono_kloc} KLOC): ~{mono_bugs:,} potential bugs")
    print(f"    Microkernel ({micro_kloc} KLOC):      ~{micro_bugs:,} potential bugs")
    print(f"    Bug reduction: {(1 - micro_bugs/mono_bugs) * 100:.2f}%\n")

    # Crash impact analysis
    print("  Crash impact comparison:\n")

    # Simulate failure rates over time
    mtbf_driver_hours = 5000  # Mean time between driver failures
    hours_per_year = 8760
    driver_failures_per_year = hours_per_year / mtbf_driver_hours

    print(f"  Assuming driver MTBF = {mtbf_driver_hours} hours:")
    print(f"  Expected driver failures per year: {driver_failures_per_year:.1f}\n")

    print(f"  {'Metric':<30} {'Monolithic':<20} {'Microkernel'}")
    print("  " + "-" * 65)

    comparisons = [
        ("Driver crash impact", "System crash", "Driver restart"),
        ("Recovery time", "Full reboot (~60s)", "Restart process (~1s)"),
        ("Data loss risk", "High (all processes)", "Low (only driver state)"),
        ("Downtime per year", f"~{driver_failures_per_year * 60:.0f} seconds",
         f"~{driver_failures_per_year * 1:.0f} seconds"),
        ("Formal verification", "Impractical (30M LOC)", "Achieved (seL4)"),
        ("TCB size", f"{mono_kloc * 1000:,} LOC", f"{micro_kloc * 1000:,} LOC"),
    ]

    for metric, mono, micro in comparisons:
        print(f"  {metric:<30} {mono:<20} {micro}")

    # When to choose which
    print("\n--- Decision Framework ---\n")

    scenarios = [
        ("High-throughput server",      "Monolithic",
         "IPC overhead unacceptable for millions of I/O ops/sec"),
        ("Safety-critical (automotive)", "Microkernel",
         "Formal verification + fault isolation required"),
        ("Desktop / laptop",            "Hybrid",
         "Performance + some isolation (macOS XNU, Windows NT)"),
        ("Military / aerospace",        "Microkernel (seL4)",
         "Mathematically proven security properties needed"),
        ("Embedded IoT device",         "Microkernel",
         "Small TCB, minimal attack surface"),
        ("Cloud infrastructure",        "Monolithic + containers",
         "Best performance with namespace-based isolation"),
    ]

    print(f"  {'Use Case':<30} {'Recommendation':<20} {'Rationale'}")
    print("  " + "-" * 80)
    for scenario, choice, rationale in scenarios:
        print(f"  {scenario:<30} {choice:<20} {rationale}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Capability-Based Access Control System ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Microkernel IPC Simulator ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Monolithic vs Microkernel Tradeoff Analysis ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
