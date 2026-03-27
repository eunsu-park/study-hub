"""
Exercises for Lesson 23: Container Internals
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers Linux namespace isolation modeling, cgroups v2 resource limiting
simulation, and overlay filesystem layer resolution.
"""


# === Exercise 1: Linux Namespace Isolation Model ===
# Problem: Model the different Linux namespaces and demonstrate how
# each namespace type isolates specific system resources.

def exercise_1():
    """Model Linux namespace isolation for a container."""
    # Simulate namespace views: host vs container

    host_state = {
        "pid_ns": {
            "processes": {
                1: ("systemd", 0),
                452: ("sshd", 0),
                1023: ("nginx", 33),
                1024: ("nginx-worker", 33),
                2048: ("postgres", 26),
            },
        },
        "uts_ns": {
            "hostname": "prod-server-01",
            "domainname": "example.com",
        },
        "net_ns": {
            "interfaces": {
                "lo": "127.0.0.1",
                "eth0": "192.168.1.100",
                "docker0": "172.17.0.1",
            },
            "routes": [
                ("default", "192.168.1.1", "eth0"),
                ("172.17.0.0/16", "172.17.0.1", "docker0"),
            ],
        },
        "mnt_ns": {
            "mounts": [
                ("/", "ext4", "/dev/sda1"),
                ("/home", "ext4", "/dev/sda2"),
                ("/proc", "proc", "proc"),
                ("/sys", "sysfs", "sysfs"),
            ],
        },
        "user_ns": {
            "uid_map": {0: 0, 1000: 1000, 33: 33},
            "gid_map": {0: 0, 1000: 1000, 33: 33},
        },
        "ipc_ns": {
            "shm_segments": [("key=0x1234", "size=4096", "owner=1000")],
            "semaphores": [("key=0x5678", "nsems=5", "owner=0")],
        },
    }

    # Container namespace: isolated view
    container_state = {
        "pid_ns": {
            "processes": {
                1: ("app-server", 0),
                15: ("app-worker-1", 1000),
                16: ("app-worker-2", 1000),
            },
            "note": "PID 1 inside container is PID 5001 on host",
        },
        "uts_ns": {
            "hostname": "my-container",
            "domainname": "(none)",
        },
        "net_ns": {
            "interfaces": {
                "lo": "127.0.0.1",
                "eth0": "172.17.0.2",
            },
            "routes": [
                ("default", "172.17.0.1", "eth0"),
            ],
        },
        "mnt_ns": {
            "mounts": [
                ("/", "overlay", "overlay"),
                ("/proc", "proc", "proc"),
                ("/sys", "sysfs", "sysfs"),
                ("/tmp", "tmpfs", "tmpfs"),
            ],
        },
        "user_ns": {
            "uid_map": {0: 100000, 1000: 101000},
            "gid_map": {0: 100000, 1000: 101000},
            "note": "root(0) inside -> uid 100000 outside (unprivileged!)",
        },
        "ipc_ns": {
            "shm_segments": [],
            "semaphores": [],
            "note": "Clean IPC namespace, no host IPC visible",
        },
    }

    namespaces = [
        ("PID Namespace (CLONE_NEWPID)", "pid_ns"),
        ("UTS Namespace (CLONE_NEWUTS)", "uts_ns"),
        ("Network Namespace (CLONE_NEWNET)", "net_ns"),
        ("Mount Namespace (CLONE_NEWNS)", "mnt_ns"),
        ("User Namespace (CLONE_NEWUSER)", "user_ns"),
        ("IPC Namespace (CLONE_NEWIPC)", "ipc_ns"),
    ]

    print("Linux Namespace Isolation Model\n")

    for ns_name, ns_key in namespaces:
        print(f"--- {ns_name} ---\n")
        host = host_state[ns_key]
        container = container_state[ns_key]

        print(f"  Host view:")
        for key, value in host.items():
            if key == "processes":
                for pid, (name, uid) in value.items():
                    print(f"    PID {pid}: {name} (uid={uid})")
            elif key == "interfaces":
                for iface, ip in value.items():
                    print(f"    {iface}: {ip}")
            elif key == "routes":
                for dest, gw, dev in value:
                    print(f"    {dest} via {gw} dev {dev}")
            elif key == "mounts":
                for mount_pt, fstype, device in value:
                    print(f"    {mount_pt} ({fstype}, {device})")
            elif key == "uid_map" or key == "gid_map":
                label = "UID" if "uid" in key else "GID"
                for inside, outside in value.items():
                    print(f"    {label} {inside} -> {outside}")
            elif key == "shm_segments" or key == "semaphores":
                for item in value:
                    print(f"    {', '.join(item)}")
            else:
                print(f"    {key}: {value}")

        print(f"\n  Container view:")
        for key, value in container.items():
            if key == "note":
                print(f"    Note: {value}")
            elif key == "processes":
                for pid, (name, uid) in value.items():
                    print(f"    PID {pid}: {name} (uid={uid})")
            elif key == "interfaces":
                for iface, ip in value.items():
                    print(f"    {iface}: {ip}")
            elif key == "routes":
                for dest, gw, dev in value:
                    print(f"    {dest} via {gw} dev {dev}")
            elif key == "mounts":
                for mount_pt, fstype, device in value:
                    print(f"    {mount_pt} ({fstype}, {device})")
            elif key == "uid_map" or key == "gid_map":
                label = "UID" if "uid" in key else "GID"
                for inside, outside in value.items():
                    print(f"    {label} {inside} -> {outside} (host)")
            elif key == "shm_segments" or key == "semaphores":
                if not value:
                    print(f"    (empty)")
                for item in value:
                    print(f"    {', '.join(item)}")
            else:
                print(f"    {key}: {value}")
        print()

    print("Key takeaway: Each namespace provides a separate view of one")
    print("resource type. Combined, they create the illusion of an isolated OS.")


# === Exercise 2: cgroups v2 Resource Limiting Simulation ===
# Problem: Simulate cgroups resource enforcement for CPU, memory, and PIDs.

def exercise_2():
    """Simulate cgroups v2 resource limiting behavior."""
    # Container cgroup configuration
    cgroup_config = {
        "name": "webapp-container",
        "cpu_quota_us": 50000,      # 50ms out of 100ms period = 50% CPU
        "cpu_period_us": 100000,    # 100ms
        "memory_max_bytes": 256 * 1024 * 1024,  # 256 MB
        "memory_swap_max_bytes": 0,  # No swap
        "pids_max": 100,
    }

    print("cgroups v2 Resource Limiting Simulation\n")
    print(f"Container: {cgroup_config['name']}")
    cpu_pct = cgroup_config["cpu_quota_us"] / cgroup_config["cpu_period_us"] * 100
    mem_mb = cgroup_config["memory_max_bytes"] / (1024 * 1024)
    print(f"  CPU limit: {cpu_pct:.0f}% ({cgroup_config['cpu_quota_us']} / "
          f"{cgroup_config['cpu_period_us']} us)")
    print(f"  Memory limit: {mem_mb:.0f} MB")
    print(f"  Swap: {'disabled' if cgroup_config['memory_swap_max_bytes'] == 0 else 'enabled'}")
    print(f"  Max PIDs: {cgroup_config['pids_max']}\n")

    # Simulate workload scenarios
    scenarios = [
        {
            "name": "Normal operation",
            "cpu_demand_pct": 30,
            "memory_demand_mb": 150,
            "pid_count": 20,
        },
        {
            "name": "CPU-intensive burst",
            "cpu_demand_pct": 120,
            "memory_demand_mb": 150,
            "pid_count": 20,
        },
        {
            "name": "Memory pressure",
            "cpu_demand_pct": 40,
            "memory_demand_mb": 300,
            "pid_count": 20,
        },
        {
            "name": "Fork bomb attempt",
            "cpu_demand_pct": 50,
            "memory_demand_mb": 100,
            "pid_count": 500,
        },
    ]

    for scenario in scenarios:
        print(f"--- Scenario: {scenario['name']} ---")
        print(f"  Demand: CPU={scenario['cpu_demand_pct']}%, "
              f"Memory={scenario['memory_demand_mb']} MB, "
              f"PIDs={scenario['pid_count']}\n")

        issues = []

        # CPU enforcement
        actual_cpu = min(scenario["cpu_demand_pct"], cpu_pct)
        throttled = scenario["cpu_demand_pct"] > cpu_pct
        if throttled:
            throttle_pct = (1 - cpu_pct / scenario["cpu_demand_pct"]) * 100
            issues.append(f"CPU throttled: {throttle_pct:.0f}% of cycles throttled")
            print(f"  CPU: Requested {scenario['cpu_demand_pct']}%, "
                  f"allowed {actual_cpu:.0f}% -> THROTTLED")
        else:
            print(f"  CPU: Requested {scenario['cpu_demand_pct']}%, "
                  f"allowed {actual_cpu:.0f}% -> OK")

        # Memory enforcement
        if scenario["memory_demand_mb"] > mem_mb:
            if cgroup_config["memory_swap_max_bytes"] == 0:
                issues.append("OOM killed: memory exceeded with no swap")
                print(f"  Memory: Requested {scenario['memory_demand_mb']} MB, "
                      f"limit {mem_mb:.0f} MB -> OOM KILL")
            else:
                print(f"  Memory: Requested {scenario['memory_demand_mb']} MB, "
                      f"limit {mem_mb:.0f} MB -> SWAPPING")
        else:
            print(f"  Memory: Requested {scenario['memory_demand_mb']} MB, "
                  f"limit {mem_mb:.0f} MB -> OK")

        # PID enforcement
        if scenario["pid_count"] > cgroup_config["pids_max"]:
            issues.append(f"Fork rejected: {scenario['pid_count'] - cgroup_config['pids_max']}"
                          f" processes denied")
            print(f"  PIDs: Requested {scenario['pid_count']}, "
                  f"limit {cgroup_config['pids_max']} -> FORK DENIED")
        else:
            print(f"  PIDs: Requested {scenario['pid_count']}, "
                  f"limit {cgroup_config['pids_max']} -> OK")

        if issues:
            print(f"\n  Enforcement actions:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  All resources within limits.")
        print()

    # cgroup filesystem layout
    print("--- cgroup v2 Filesystem Layout ---\n")
    cg_name = cgroup_config["name"]
    files = [
        (f"/sys/fs/cgroup/{cg_name}/cpu.max",
         f"{cgroup_config['cpu_quota_us']} {cgroup_config['cpu_period_us']}"),
        (f"/sys/fs/cgroup/{cg_name}/cpu.stat",
         "usage_usec 142857\nthrottled_usec 57143\nnr_throttled 12"),
        (f"/sys/fs/cgroup/{cg_name}/memory.max",
         f"{cgroup_config['memory_max_bytes']}"),
        (f"/sys/fs/cgroup/{cg_name}/memory.current",
         "157286400"),
        (f"/sys/fs/cgroup/{cg_name}/memory.swap.max",
         f"{cgroup_config['memory_swap_max_bytes']}"),
        (f"/sys/fs/cgroup/{cg_name}/pids.max",
         f"{cgroup_config['pids_max']}"),
        (f"/sys/fs/cgroup/{cg_name}/pids.current",
         "20"),
        (f"/sys/fs/cgroup/{cg_name}/cgroup.procs",
         "5001\n5015\n5016"),
    ]

    for path, content in files:
        print(f"  {path}")
        for line in content.split("\n"):
            print(f"    {line}")


# === Exercise 3: Overlay Filesystem Layer Resolution ===
# Problem: Simulate how OverlayFS resolves file lookups across layers
# and handles writes (copy-up) and deletes (whiteout).

def exercise_3():
    """Simulate OverlayFS file operations with multiple layers."""
    # Define image layers (lower = read-only)
    layer0_base = {
        "/bin/sh": ("binary", 120000),
        "/bin/ls": ("binary", 135000),
        "/etc/passwd": ("root:x:0:0::/root:/bin/sh\nnobody:x:65534:65534::/:", 87),
        "/etc/hostname": ("base-image", 10),
        "/lib/libc.so": ("binary", 2400000),
        "/tmp/": ("dir", 0),
    }

    layer1_packages = {
        "/usr/bin/python3": ("binary", 5800000),
        "/usr/lib/python3/os.py": ("source", 38000),
        "/etc/pip.conf": ("[global]\nindex-url = https://pypi.org/simple", 50),
    }

    layer2_app = {
        "/app/server.py": ("source", 15000),
        "/app/config.yaml": ("port: 8080\ndebug: false", 30),
        "/etc/hostname": ("app-container", 13),  # overrides layer0
    }

    layers = [
        ("Layer 0 (base image)", layer0_base),
        ("Layer 1 (packages)", layer1_packages),
        ("Layer 2 (application)", layer2_app),
    ]

    # Upper layer (read-write, starts empty)
    upper = {}

    # Whiteout markers for deleted files
    whiteouts = set()

    print("OverlayFS Layer Resolution Simulation\n")

    for name, layer in layers:
        print(f"  {name}:")
        for path, (content, size) in sorted(layer.items()):
            kind = "dir" if path.endswith("/") else "file"
            print(f"    {path} ({kind}, {size} bytes)")
    print(f"  Upper (read-write): (empty)\n")

    def resolve_file(path):
        """Look up a file through the overlay stack."""
        # Check whiteouts first
        if path in whiteouts:
            return None, None, "whiteout"

        # Check upper layer (container writes)
        if path in upper:
            return upper[path], "upper", "found"

        # Check lower layers top-down
        for i in range(len(layers) - 1, -1, -1):
            _, layer = layers[i]
            if path in layer:
                return layer[path], f"layer{i}", "found"

        return None, None, "not found"

    # Demonstrate file lookups
    print("--- File Lookups (Read Operations) ---\n")
    lookup_paths = [
        "/bin/sh",
        "/usr/bin/python3",
        "/app/server.py",
        "/etc/hostname",     # exists in layer0 AND layer2
        "/nonexistent",
    ]

    print(f"  {'Path':<25} {'Found In':<12} {'Size':<10} {'Note'}")
    print("  " + "-" * 60)

    for path in lookup_paths:
        result, location, status = resolve_file(path)
        if status == "found":
            _, size = result
            note = ""
            # Check for override
            count = sum(1 for _, layer in layers if path in layer)
            if count > 1:
                note = "(overrides lower layer)"
            print(f"  {path:<25} {location:<12} {size:<10} {note}")
        else:
            print(f"  {path:<25} {'---':<12} {'---':<10} not found")

    # Demonstrate write (copy-up)
    print(f"\n--- Write Operation (Copy-Up) ---\n")
    write_path = "/etc/passwd"
    print(f"  Writing to {write_path} (exists in layer0, read-only)")
    original, _, _ = resolve_file(write_path)
    new_content = original[0] + "\napp:x:1000:1000::/app:/bin/sh"
    upper[write_path] = (new_content, len(new_content))
    print(f"  1. Copy original from layer0 to upper layer")
    print(f"  2. Modify in upper layer (added user 'app')")
    print(f"  3. Original in layer0 is unchanged")
    result, location, _ = resolve_file(write_path)
    print(f"  4. Now resolves from: {location} (size: {result[1]} bytes)")

    # Demonstrate new file creation
    print(f"\n--- New File Creation ---\n")
    new_path = "/app/data.json"
    upper[new_path] = ('{"count": 0}', 14)
    print(f"  Created {new_path} in upper layer")
    result, location, _ = resolve_file(new_path)
    print(f"  Resolves from: {location}")

    # Demonstrate delete (whiteout)
    print(f"\n--- Delete Operation (Whiteout) ---\n")
    delete_path = "/etc/pip.conf"
    print(f"  Deleting {delete_path} (exists in layer1, read-only)")
    whiteouts.add(delete_path)
    print(f"  1. Cannot actually remove from read-only layer")
    print(f"  2. Created whiteout marker in upper layer")
    result, location, status = resolve_file(delete_path)
    print(f"  3. Lookup result: {status} (file appears deleted)")
    print(f"  4. Original still exists in layer1 (unchanged)")

    # Show final merged view
    print(f"\n--- Final Merged View ---\n")
    all_paths = set()
    for _, layer in layers:
        all_paths.update(layer.keys())
    all_paths.update(upper.keys())

    print(f"  {'Path':<30} {'Source':<12} {'Size':<10} {'Status'}")
    print("  " + "-" * 60)

    for path in sorted(all_paths):
        result, location, status = resolve_file(path)
        if status == "found":
            _, size = result
            modified = " (modified)" if path in upper and any(
                path in layer for _, layer in layers) else ""
            created = " (new)" if path in upper and not any(
                path in layer for _, layer in layers) else ""
            print(f"  {path:<30} {location:<12} {size:<10} {modified}{created}")
        elif status == "whiteout":
            print(f"  {path:<30} {'---':<12} {'---':<10} (deleted/whiteout)")

    print(f"\n  Storage efficiency:")
    lower_total = sum(size for _, layer in layers
                      for _, (_, size) in layer.items())
    upper_total = sum(size for _, (_, size) in upper.items())
    print(f"  Lower layers (shared, read-only): {lower_total:,} bytes")
    print(f"  Upper layer (per-container):      {upper_total:,} bytes")
    print(f"  Multiple containers sharing the same image only pay")
    print(f"  the upper layer cost for their specific modifications.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Linux Namespace Isolation Model ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: cgroups v2 Resource Limiting Simulation ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Overlay Filesystem Layer Resolution ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
