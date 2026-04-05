#!/usr/bin/env python3
"""Example: eBPF Observability — Kernel-Level Tracing Without Instrumentation

Demonstrates eBPF observability concepts: simulated kernel probe events,
network flow tracking, syscall latency histograms, and zero-instrumentation
service map generation from socket-level data.
Related lesson: 24_eBPF_Observability.md
"""

# =============================================================================
# WHY eBPF OBSERVABILITY?
# Traditional observability requires application instrumentation. eBPF
# programs run in the Linux kernel and can observe syscalls, network packets,
# and function calls WITHOUT modifying application code. Tools like Cilium
# Hubble, Pixie, and Grafana Beyla leverage eBPF for auto-instrumentation.
# =============================================================================

import random
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 1. SIMULATED KERNEL EVENTS
# =============================================================================

@dataclass
class SyscallEvent:
    """Represents a captured syscall event from a kernel probe."""
    pid: int
    comm: str          # Process name
    syscall: str       # e.g., read, write, connect, accept
    latency_ns: int    # Nanoseconds
    timestamp: float = field(default_factory=time.time)
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkEvent:
    """Represents a captured network event (socket-level)."""
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str      # TCP, UDP
    bytes_sent: int = 0
    bytes_recv: int = 0
    latency_ns: int = 0
    pid: int = 0
    comm: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class HTTPEvent:
    """HTTP request/response captured from socket data (L7 parsing)."""
    method: str
    path: str
    status_code: int
    latency_ms: float
    src_service: str
    dst_service: str
    content_length: int = 0
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# 2. SYSCALL LATENCY ANALYZER
# =============================================================================

@dataclass
class SyscallHistogram:
    """Histogram of syscall latencies (log2-based buckets like bcc/bpftrace)."""
    syscall: str
    # Buckets: [0-1us, 1-2us, 2-4us, 4-8us, ..., 512ms-1s, 1s+]
    bucket_counts: list[int] = field(default_factory=lambda: [0] * 20)
    total_count: int = 0
    total_ns: int = 0

    def record(self, latency_ns: int) -> None:
        self.total_count += 1
        self.total_ns += latency_ns
        # Log2 bucketing (microsecond-based)
        us = max(1, latency_ns // 1000)
        bucket = min(int(us).bit_length() - 1, len(self.bucket_counts) - 1)
        self.bucket_counts[bucket] += 1

    @property
    def avg_us(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.total_ns / self.total_count) / 1000

    def display(self) -> str:
        """ASCII histogram display (bpftrace-style)."""
        lines = [f"Syscall: {self.syscall} (n={self.total_count}, avg={self.avg_us:.1f}us)"]
        max_count = max(self.bucket_counts) if self.bucket_counts else 1
        for i, count in enumerate(self.bucket_counts):
            if count == 0:
                continue
            lower = 2 ** i
            upper = 2 ** (i + 1)
            bar_len = int(count / max_count * 40) if max_count else 0
            bar = "#" * bar_len
            lines.append(f"  [{lower:>8} - {upper:>8}) us: {count:>6} |{bar}")
        return "\n".join(lines)


# =============================================================================
# 3. SERVICE MAP FROM NETWORK FLOWS
# =============================================================================

@dataclass
class ServiceMapBuilder:
    """Build a service dependency map from network events (no instrumentation)."""
    # Map process name to resolved service name
    process_to_service: dict[str, str] = field(default_factory=dict)
    # Edges: (src_svc, dst_svc) -> traffic stats
    edges: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def register_process(self, comm: str, service_name: str) -> None:
        self.process_to_service[comm] = service_name

    def ingest_http_event(self, event: HTTPEvent) -> None:
        """Add an HTTP event to the service map."""
        key = (event.src_service, event.dst_service)
        if key not in self.edges:
            self.edges[key] = {
                "request_count": 0, "error_count": 0,
                "total_latency_ms": 0.0, "bytes_total": 0,
            }
        edge = self.edges[key]
        edge["request_count"] += 1
        if event.status_code >= 500:
            edge["error_count"] += 1
        edge["total_latency_ms"] += event.latency_ms
        edge["bytes_total"] += event.content_length

    def get_service_map(self) -> list[dict[str, Any]]:
        """Return the service map as a list of edges with metrics."""
        result = []
        for (src, dst), stats in self.edges.items():
            avg_latency = (stats["total_latency_ms"] / stats["request_count"]
                           if stats["request_count"] else 0)
            error_rate = (stats["error_count"] / stats["request_count"]
                          if stats["request_count"] else 0)
            result.append({
                "source": src, "destination": dst,
                "requests": stats["request_count"],
                "error_rate": round(error_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "throughput_bytes": stats["bytes_total"],
            })
        return result


# =============================================================================
# 4. PROCESS-LEVEL RESOURCE TRACKER
# =============================================================================

@dataclass
class ProcessProfile:
    """Resource profile for a process (built from eBPF events)."""
    pid: int
    comm: str
    syscall_counts: dict[str, int] = field(default_factory=dict)
    total_cpu_ns: int = 0
    net_bytes_sent: int = 0
    net_bytes_recv: int = 0
    file_reads: int = 0
    file_writes: int = 0

    def record_syscall(self, event: SyscallEvent) -> None:
        self.syscall_counts[event.syscall] = self.syscall_counts.get(event.syscall, 0) + 1
        self.total_cpu_ns += event.latency_ns
        if event.syscall in ("read", "recvfrom", "recvmsg"):
            self.file_reads += 1
            self.net_bytes_recv += event.args.get("bytes", 0)
        elif event.syscall in ("write", "sendto", "sendmsg"):
            self.file_writes += 1
            self.net_bytes_sent += event.args.get("bytes", 0)


# =============================================================================
# 5. DATA GENERATOR
# =============================================================================

def generate_events(n_syscalls: int = 500, n_http: int = 200) -> tuple:
    """Generate synthetic eBPF events for demonstration."""
    random.seed(42)
    services = {
        "nginx": "frontend",
        "python3": "api-server",
        "java": "order-service",
        "postgres": "database",
    }
    syscall_events = []
    for _ in range(n_syscalls):
        comm = random.choice(list(services.keys()))
        syscall = random.choice(["read", "write", "connect", "accept", "sendto", "recvfrom"])
        latency = int(random.expovariate(1 / 50_000))  # ~50us avg
        syscall_events.append(SyscallEvent(
            pid=random.randint(1000, 9999), comm=comm,
            syscall=syscall, latency_ns=latency,
            args={"bytes": random.randint(64, 8192)},
        ))

    http_events = []
    svc_list = list(services.values())
    for _ in range(n_http):
        src = random.choice(svc_list[:-1])
        dst = random.choice([s for s in svc_list if s != src])
        http_events.append(HTTPEvent(
            method=random.choice(["GET", "POST", "PUT"]),
            path=random.choice(["/api/orders", "/api/users", "/healthz", "/api/payments"]),
            status_code=random.choices([200, 201, 400, 500], weights=[85, 5, 5, 5])[0],
            latency_ms=random.expovariate(1 / 50),
            src_service=src, dst_service=dst,
            content_length=random.randint(100, 5000),
        ))

    return syscall_events, http_events, services


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    syscall_events, http_events, services = generate_events()

    # --- Syscall Latency Histograms ---
    print("=" * 60)
    print("eBPF Syscall Latency Histograms")
    print("=" * 60)
    histograms: dict[str, SyscallHistogram] = {}
    for event in syscall_events:
        if event.syscall not in histograms:
            histograms[event.syscall] = SyscallHistogram(syscall=event.syscall)
        histograms[event.syscall].record(event.latency_ns)
    for name in ["read", "write", "connect"]:
        if name in histograms:
            print(histograms[name].display())
            print()

    # --- Service Map ---
    print("=" * 60)
    print("Auto-Generated Service Map (from socket data)")
    print("=" * 60)
    builder = ServiceMapBuilder(process_to_service=services)
    for event in http_events:
        builder.ingest_http_event(event)
    for edge in builder.get_service_map():
        print(f"  {edge['source']:>15} -> {edge['destination']:<15} "
              f"reqs={edge['requests']:>4} err={edge['error_rate']:.2%} "
              f"lat={edge['avg_latency_ms']:.1f}ms")

    # --- Process Profiles ---
    print(f"\n{'=' * 60}")
    print("Process Resource Profiles")
    print("=" * 60)
    profiles: dict[str, ProcessProfile] = {}
    for event in syscall_events:
        if event.comm not in profiles:
            profiles[event.comm] = ProcessProfile(pid=event.pid, comm=event.comm)
        profiles[event.comm].record_syscall(event)
    for prof in profiles.values():
        cpu_ms = prof.total_cpu_ns / 1_000_000
        top_syscalls = sorted(prof.syscall_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {prof.comm:>10}: cpu={cpu_ms:.1f}ms, "
              f"reads={prof.file_reads}, writes={prof.file_writes}, "
              f"top={top_syscalls}")
