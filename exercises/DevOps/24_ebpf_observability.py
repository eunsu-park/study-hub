#!/usr/bin/env python3
"""Exercises for Lesson 24: eBPF Observability
Topic: DevOps
"""


def exercise_1():
    """bpftrace probe design for production scenarios."""
    print("=== Exercise 1: bpftrace Probe Design ===\n")
    probes = [
        {
            "scenario": "DNS query rate by process",
            "command": (
                "bpftrace -e 'uprobe:/lib/.../libc.so.6:getaddrinfo "
                "{ @dns[comm] = count(); } "
                "interval:s:1 { print(@dns); clear(@dns); }'"
            ),
        },
        {
            "scenario": "write() syscall latency for specific PID",
            "command": (
                "bpftrace -e 'tracepoint:syscalls:sys_enter_write /pid == PID/ "
                "{ @start[tid] = nsecs; } "
                "tracepoint:syscalls:sys_exit_write /pid == PID && @start[tid]/ "
                "{ @write_us = hist((nsecs - @start[tid]) / 1000); delete(@start[tid]); }'"
            ),
        },
        {
            "scenario": "Total bytes written per process",
            "command": (
                "bpftrace -e 'kretprobe:vfs_write /retval > 0/ "
                "{ @bytes[comm] = sum(retval); }'"
            ),
        },
    ]
    for probe in probes:
        print(f"Scenario: {probe['scenario']}")
        print(f"  {probe['command']}")
        print()


def exercise_2():
    """eBPF vs OTel decision for different scenarios."""
    print("=== Exercise 2: eBPF vs OTel Decision ===\n")
    decisions = [
        ("Legacy Java app, no source code, needs HTTP metrics",
         "eBPF (Beyla)", "No code changes possible"),
        ("New Python microservice, custom business attributes",
         "OpenTelemetry", "Custom attributes require app-level instrumentation"),
        ("Identify K8s pods with excessive DNS lookups",
         "eBPF (Cilium Hubble)", "DNS is kernel-level; OTel cannot observe it"),
        ("Go service with GC-related latency spikes",
         "Both", "eBPF for kernel/scheduling, OTel for affected requests"),
        ("Network policy enforcement monitoring",
         "eBPF (Hubble/Tetragon)", "Policies enforced at kernel level"),
    ]
    for scenario, choice, reason in decisions:
        print(f"Scenario: {scenario}")
        print(f"  Choice: {choice}")
        print(f"  Reason: {reason}")
        print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
