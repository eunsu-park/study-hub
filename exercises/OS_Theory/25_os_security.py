"""
Exercises for Lesson 25: OS Security
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers Linux capability modeling, seccomp-style syscall filtering,
and defense-in-depth security analysis.
"""


# === Exercise 1: Linux Capability Simulator ===
# Problem: Model Linux capabilities and demonstrate the principle of least
# privilege by granting only the minimum capabilities a service needs.

def exercise_1():
    """Simulate Linux capability sets and privilege dropping."""

    # Define capability constants (mirroring real Linux capabilities)
    CAPS = {
        "CAP_NET_BIND_SERVICE": "Bind to privileged ports (<1024)",
        "CAP_NET_RAW":          "Use raw sockets (e.g., ping)",
        "CAP_SYS_ADMIN":        "Broad system administration",
        "CAP_DAC_OVERRIDE":     "Bypass file permission checks",
        "CAP_SYS_PTRACE":       "Trace any process",
        "CAP_NET_ADMIN":        "Network configuration",
        "CAP_CHOWN":            "Change file ownership",
        "CAP_SETUID":           "Set UID of process",
        "CAP_SETGID":           "Set GID of process",
        "CAP_KILL":             "Send signals to any process",
    }

    class CapabilitySet:
        """Model of Linux per-process capability sets."""

        def __init__(self, name, effective=None, permitted=None,
                     inheritable=None, bounding=None):
            self.name = name
            self.effective = set(effective or [])
            self.permitted = set(permitted or [])
            self.inheritable = set(inheritable or [])
            self.bounding = set(bounding or CAPS.keys())

        def has_cap(self, cap):
            """Check if a capability is in the effective set."""
            return cap in self.effective

        def drop_cap(self, cap):
            """Drop a capability from all sets."""
            self.effective.discard(cap)
            self.permitted.discard(cap)
            self.inheritable.discard(cap)
            # Bounding set can only shrink, never grow
            self.bounding.discard(cap)

        def drop_to_minimum(self, needed_caps):
            """Drop all capabilities except those in needed_caps."""
            all_caps = set(CAPS.keys())
            for cap in all_caps:
                if cap not in needed_caps:
                    self.drop_cap(cap)

        def show(self):
            """Display the current capability sets."""
            print(f"  Process: {self.name}")
            print(f"    Effective:   {sorted(self.effective) or '{empty}'}")
            print(f"    Permitted:   {sorted(self.permitted) or '{empty}'}")
            print(f"    Inheritable: {sorted(self.inheritable) or '{empty}'}")
            print(f"    Bounding:    {len(self.bounding)} capabilities")

    # Scenario: A web server process starts as root, then drops privileges
    print("=== Scenario: Web Server Privilege Dropping ===\n")

    # Root process has ALL capabilities
    all_caps = set(CAPS.keys())
    root_proc = CapabilitySet("nginx (root)", effective=all_caps,
                              permitted=all_caps, inheritable=all_caps)
    print("Before dropping privileges:")
    root_proc.show()

    # Web server only needs to bind to port 80/443
    needed = {"CAP_NET_BIND_SERVICE"}
    root_proc.drop_to_minimum(needed)
    root_proc.name = "nginx (worker)"

    print("\nAfter dropping to minimum privileges:")
    root_proc.show()

    # Verify access control
    print("\n--- Access Control Checks ---")
    operations = [
        ("CAP_NET_BIND_SERVICE", "Bind to port 80"),
        ("CAP_SYS_ADMIN",       "Mount filesystem"),
        ("CAP_DAC_OVERRIDE",    "Read /etc/shadow"),
        ("CAP_NET_RAW",         "Send raw packets"),
        ("CAP_KILL",            "Kill other processes"),
    ]

    for cap, operation in operations:
        allowed = root_proc.has_cap(cap)
        status = "ALLOWED" if allowed else "DENIED"
        print(f"  {operation:<30} [{status}]  (requires {cap})")

    # Compare: traditional SUID vs capabilities
    print("\n--- Comparison: SUID Root vs Capabilities ---\n")

    print("  Traditional SUID approach:")
    print("    ping is SUID root → process gets ALL root privileges")
    print("    Risk: bug in ping → attacker gets full root access\n")

    print("  Capabilities approach:")
    print("    ping has CAP_NET_RAW only → process gets ONLY raw socket")
    print("    Risk: bug in ping → attacker can send raw packets (limited)\n")

    print("  Impact reduction:")
    print(f"    SUID: {len(CAPS)} capabilities exposed")
    print(f"    Capability-based: 1 capability exposed")
    print(f"    Attack surface reduced by {(len(CAPS)-1)/len(CAPS)*100:.0f}%")


# === Exercise 2: Seccomp Syscall Filter Simulator ===
# Problem: Simulate a seccomp-bpf filter that restricts which system calls
# a process can make, and demonstrate sandboxing effectiveness.

def exercise_2():
    """Simulate seccomp-bpf syscall filtering for process sandboxing."""

    # Define system calls with their numbers and categories
    SYSCALLS = {
        "read":        (0,   "io",       "Read from file descriptor"),
        "write":       (1,   "io",       "Write to file descriptor"),
        "open":        (2,   "fs",       "Open file"),
        "close":       (3,   "io",       "Close file descriptor"),
        "stat":        (4,   "fs",       "Get file status"),
        "mmap":        (9,   "memory",   "Map memory"),
        "mprotect":    (10,  "memory",   "Set memory protection"),
        "munmap":      (11,  "memory",   "Unmap memory"),
        "brk":         (12,  "memory",   "Change data segment size"),
        "ioctl":       (16,  "device",   "Device control"),
        "socket":      (41,  "network",  "Create socket"),
        "connect":     (42,  "network",  "Connect socket"),
        "accept":      (43,  "network",  "Accept connection"),
        "sendto":      (44,  "network",  "Send data on socket"),
        "recvfrom":    (45,  "network",  "Receive data from socket"),
        "bind":        (49,  "network",  "Bind socket to address"),
        "listen":      (50,  "network",  "Listen for connections"),
        "fork":        (57,  "process",  "Create child process"),
        "execve":      (59,  "process",  "Execute program"),
        "exit_group":  (231, "process",  "Exit all threads"),
        "ptrace":      (101, "debug",    "Trace process"),
        "mount":       (165, "fs",       "Mount filesystem"),
    }

    # seccomp actions
    ALLOW = "ALLOW"
    KILL = "KILL"
    ERRNO = "ERRNO(EPERM)"
    LOG = "LOG"

    class SeccompFilter:
        """Simplified seccomp-bpf filter."""

        def __init__(self, name, default_action=KILL):
            self.name = name
            self.default_action = default_action
            self.rules = {}  # syscall_name -> action

        def allow(self, syscall_name):
            self.rules[syscall_name] = ALLOW

        def deny_with_errno(self, syscall_name):
            self.rules[syscall_name] = ERRNO

        def log_only(self, syscall_name):
            self.rules[syscall_name] = LOG

        def check(self, syscall_name):
            """Evaluate the filter for a given syscall."""
            return self.rules.get(syscall_name, self.default_action)

    # Build a sandbox profile for a compute-only process
    print("=== Seccomp Profile: Compute-Only Sandbox ===\n")

    sandbox = SeccompFilter("compute_sandbox", default_action=KILL)
    sandbox.allow("read")
    sandbox.allow("write")
    sandbox.allow("mmap")
    sandbox.allow("mprotect")
    sandbox.allow("munmap")
    sandbox.allow("brk")
    sandbox.allow("close")
    sandbox.allow("exit_group")

    # Test all syscalls against the filter
    print(f"Profile: {sandbox.name}")
    print(f"Default action: {sandbox.default_action}\n")

    print(f"  {'Syscall':<14} {'Nr':<5} {'Category':<10} {'Action':<16} Description")
    print("  " + "-" * 75)

    allowed_count = 0
    blocked_count = 0
    for name, (nr, category, desc) in sorted(SYSCALLS.items(),
                                              key=lambda x: x[1][0]):
        action = sandbox.check(name)
        marker = "  " if action == ALLOW else "X "
        print(f"  {marker}{name:<12} {nr:<5} {category:<10} {action:<16} {desc}")
        if action == ALLOW:
            allowed_count += 1
        else:
            blocked_count += 1

    print(f"\n  Summary: {allowed_count} allowed, {blocked_count} blocked "
          f"out of {len(SYSCALLS)} syscalls")

    # Simulate an attack sequence
    print("\n--- Attack Simulation ---\n")

    attack_sequence = [
        ("write",   "Attacker executes normally (write output)"),
        ("socket",  "Attacker tries to open network connection"),
        ("fork",    "Attacker tries to create child process"),
        ("execve",  "Attacker tries to execute /bin/sh"),
        ("open",    "Attacker tries to read /etc/passwd"),
        ("mount",   "Attacker tries to mount filesystem"),
        ("ptrace",  "Attacker tries to trace another process"),
    ]

    print("  Simulating attack from inside sandboxed process:\n")
    for syscall, description in attack_sequence:
        action = sandbox.check(syscall)
        if action == ALLOW:
            result = "Proceeds normally"
        elif action == KILL:
            result = "PROCESS KILLED (SIGSYS)"
        else:
            result = f"Denied: {action}"
        print(f"  Step: {description}")
        print(f"    {syscall}() -> {result}\n")

    print("  Result: Attack blocked at step 2. Sandbox prevented all")
    print("  network, process, filesystem, and debug operations.")


# === Exercise 3: Defense-in-Depth Security Analysis ===
# Problem: Model a multi-layer defense system and analyze how each layer
# mitigates specific attack vectors.

def exercise_3():
    """Analyze a defense-in-depth security architecture."""

    # Define security layers
    layers = [
        {
            "name": "Hardware Security",
            "mechanisms": ["TPM (Trusted Platform Module)",
                           "Secure Boot (UEFI)",
                           "NX/XD bit (No-Execute)"],
            "mitigates": ["boot-level rootkits", "firmware tampering",
                          "code execution on stack/heap"],
        },
        {
            "name": "Kernel Hardening",
            "mechanisms": ["ASLR (Address Space Layout Randomization)",
                           "KASLR (Kernel ASLR)",
                           "Stack Canaries (-fstack-protector)",
                           "KPTI (Kernel Page Table Isolation)"],
            "mitigates": ["buffer overflow exploits", "ROP attacks",
                          "stack smashing", "Meltdown side-channel"],
        },
        {
            "name": "Process Isolation",
            "mechanisms": ["Linux Capabilities",
                           "seccomp-bpf filtering",
                           "Namespaces (mount, PID, net, user)",
                           "cgroups (resource limits)"],
            "mitigates": ["privilege escalation", "unauthorized syscalls",
                          "container escape", "resource exhaustion (DoS)"],
        },
        {
            "name": "Mandatory Access Control",
            "mechanisms": ["AppArmor (path-based MAC)",
                           "SELinux (label-based MAC)"],
            "mitigates": ["unauthorized file access",
                          "confused deputy attacks",
                          "lateral movement after compromise"],
        },
        {
            "name": "Application Security",
            "mechanisms": ["Input validation",
                           "Memory-safe languages (Rust, Go)",
                           "Sandboxed runtimes (WASM)"],
            "mitigates": ["injection attacks (SQL, command)",
                          "memory corruption",
                          "arbitrary code execution"],
        },
    ]

    print("=== Defense-in-Depth: Security Layer Analysis ===\n")

    for i, layer in enumerate(layers, 1):
        print(f"Layer {i}: {layer['name']}")
        print(f"  Mechanisms:")
        for m in layer["mechanisms"]:
            print(f"    - {m}")
        print(f"  Mitigates:")
        for m in layer["mitigates"]:
            print(f"    * {m}")
        print()

    # Attack scenario analysis
    print("--- Attack Scenario: Buffer Overflow Exploit ---\n")

    attacks = [
        {
            "step": "Attacker sends crafted input to overwrite stack buffer",
            "blocked_by": "Stack Canary",
            "layer": 2,
            "bypass": "Canary value leaked or brute-forced",
        },
        {
            "step": "Attacker overwrites return address to jump to shellcode",
            "blocked_by": "NX bit (non-executable stack)",
            "layer": 1,
            "bypass": "Use ROP (Return-Oriented Programming) instead",
        },
        {
            "step": "Attacker uses ROP to chain gadgets from known addresses",
            "blocked_by": "ASLR (randomized addresses)",
            "layer": 2,
            "bypass": "Information leak reveals base address",
        },
        {
            "step": "Attacker calls execve('/bin/sh') via ROP chain",
            "blocked_by": "seccomp (execve not allowed)",
            "layer": 3,
            "bypass": "Find allowed syscall that achieves goal",
        },
        {
            "step": "Attacker tries to read /etc/shadow",
            "blocked_by": "SELinux / AppArmor (MAC denies access)",
            "layer": 4,
            "bypass": "Very difficult -- MAC is kernel-enforced",
        },
    ]

    for i, attack in enumerate(attacks, 1):
        print(f"  Step {i}: {attack['step']}")
        print(f"    Blocked by: {attack['blocked_by']} (Layer {attack['layer']})")
        print(f"    Bypass:     {attack['bypass']}")
        print()

    # Build defense matrix
    print("--- Defense Matrix: Attack Type vs Mitigation ---\n")

    attack_types = [
        "Buffer Overflow",
        "Privilege Escalation",
        "Syscall Abuse",
        "Unauthorized File Access",
        "Resource Exhaustion",
    ]

    mitigations = [
        ("Stack Canary",    [True,  False, False, False, False]),
        ("ASLR",            [True,  False, False, False, False]),
        ("NX bit",          [True,  False, False, False, False]),
        ("Capabilities",    [False, True,  False, False, False]),
        ("seccomp",         [False, True,  True,  False, False]),
        ("AppArmor/SELinux",[False, True,  False, True,  False]),
        ("cgroups",         [False, False, False, False, True]),
    ]

    header = f"  {'Mitigation':<18}"
    for a in attack_types:
        header += f"{a[:12]:<14}"
    print(header)
    print("  " + "-" * (18 + 14 * len(attack_types)))

    for name, effectiveness in mitigations:
        line = f"  {name:<18}"
        for eff in effectiveness:
            line += f"{'YES':<14}" if eff else f"{'---':<14}"
        print(line)

    print(f"\n  Key insight: No single mitigation covers all attack types.")
    print(f"  Each layer catches attacks that slipped past previous layers.")
    print(f"  An attacker must defeat ALL relevant layers to succeed.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Linux Capability Simulator ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Seccomp Syscall Filter Simulator ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Defense-in-Depth Security Analysis ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
