"""
Exercises for Lesson 03: Network Fundamentals Review
Topic: System_Design

Solutions to practice problems and hands-on exercises.
Covers DNS resolution, HTTP protocol comparison, and REST vs gRPC payload analysis.
"""

import json
import struct
import time
import random
from collections import defaultdict


# === Exercise 1: DNS Resolution Tracer ===
# Problem: Build a DNS resolution simulator demonstrating recursive vs iterative queries.

class DNSServer:
    """Simulates a DNS server at a specific hierarchy level."""

    def __init__(self, name, records=None, referrals=None):
        self.name = name
        self.records = records or {}      # domain -> IP
        self.referrals = referrals or {}  # suffix -> next server
        self.cache = {}
        self.query_count = 0

    def query(self, domain):
        self.query_count += 1

        # Check cache first
        if domain in self.cache:
            return {"answer": self.cache[domain], "cached": True}

        # Direct record
        if domain in self.records:
            return {"answer": self.records[domain], "cached": False}

        # Referral to next level
        for suffix, server in self.referrals.items():
            if domain.endswith(suffix):
                return {"referral": server}

        return {"error": "NXDOMAIN"}


def build_dns_hierarchy():
    """Build Root -> TLD -> Authoritative DNS hierarchy."""
    auth_server = DNSServer("auth.example.com", records={
        "www.example.com": "93.184.216.34",
        "api.example.com": "93.184.216.35",
        "mail.example.com": "93.184.216.36",
    })

    tld_server = DNSServer("tld.com", referrals={
        ".example.com": auth_server,
    })

    root_server = DNSServer("root", referrals={
        ".com": tld_server,
        ".org": DNSServer("tld.org"),
    })

    return root_server, tld_server, auth_server


def recursive_resolve(server, domain, messages=None):
    """Recursive resolution: each server queries the next level."""
    if messages is None:
        messages = []

    result = server.query(domain)
    messages.append(f"Query {server.name} for {domain}")

    if "answer" in result:
        messages.append(f"Answer from {server.name}: {result['answer']}"
                        f"{' (cached)' if result.get('cached') else ''}")
        return result["answer"], messages

    if "referral" in result:
        next_server = result["referral"]
        messages.append(f"Referral from {server.name} -> {next_server.name}")
        answer, messages = recursive_resolve(next_server, domain, messages)
        # Cache the result
        server.cache[domain] = answer
        return answer, messages

    return None, messages


def iterative_resolve(root, domain):
    """Iterative resolution: resolver follows referrals itself."""
    messages = []
    current_server = root

    while True:
        result = current_server.query(domain)
        messages.append(f"Query {current_server.name} for {domain}")

        if "answer" in result:
            messages.append(f"Answer: {result['answer']}"
                            f"{' (cached)' if result.get('cached') else ''}")
            return result["answer"], messages

        if "referral" in result:
            next_server = result["referral"]
            messages.append(f"Referral -> {next_server.name}")
            current_server = next_server
        else:
            messages.append("NXDOMAIN")
            return None, messages


def exercise_1():
    """DNS resolution simulator: recursive vs iterative."""
    root, tld, auth = build_dns_hierarchy()
    domain = "www.example.com"

    print("--- Recursive Resolution ---")
    answer, msgs = recursive_resolve(root, domain)
    for m in msgs:
        print(f"  {m}")
    print(f"Messages exchanged: {len(msgs)}")

    print("\n--- Iterative Resolution ---")
    root2, tld2, auth2 = build_dns_hierarchy()  # Fresh servers
    answer, msgs = iterative_resolve(root2, domain)
    for m in msgs:
        print(f"  {m}")
    print(f"Messages exchanged: {len(msgs)}")

    # Demonstrate caching
    print("\n--- Second lookup (with caching) ---")
    root3, tld3, auth3 = build_dns_hierarchy()
    # First lookup populates cache
    recursive_resolve(root3, domain)
    # Second lookup should hit cache
    answer, msgs = recursive_resolve(root3, domain)
    for m in msgs:
        print(f"  {m}")
    print(f"Messages exchanged: {len(msgs)} (vs 5+ without cache)")


# === Exercise 2: HTTP/1.1 vs HTTP/2 Comparison ===
# Problem: Simulate performance difference between HTTP/1.1 and HTTP/2.

class Resource:
    """A web resource with a name and download time."""
    def __init__(self, name, size_kb, download_time_ms):
        self.name = name
        self.size_kb = size_kb
        self.download_time_ms = download_time_ms


def simulate_http11(resources, max_connections=6):
    """HTTP/1.1: max 6 parallel connections, each blocks until complete."""
    connections = [0.0] * max_connections  # Track when each connection is free
    timeline = []

    for res in resources:
        # Find the connection that becomes free earliest
        earliest_idx = connections.index(min(connections))
        start = connections[earliest_idx]
        end = start + res.download_time_ms
        connections[earliest_idx] = end
        timeline.append((res.name, start, end))

    total_time = max(connections)
    return total_time, timeline


def simulate_http2(resources):
    """HTTP/2: all requests multiplexed on 1 connection, interleaved frames."""
    # All requests start simultaneously, but share bandwidth
    # Total time is dominated by the largest resource since they share the pipe
    # Simplified model: all download in parallel, finish at max(individual times)
    # In reality, multiplexing means they interleave frames

    # More realistic: total bytes / bandwidth, but all start at t=0
    total_time = max(r.download_time_ms for r in resources)

    timeline = []
    for res in resources:
        timeline.append((res.name, 0, res.download_time_ms))

    return total_time, timeline


def exercise_2():
    """HTTP/1.1 vs HTTP/2 performance comparison."""
    resources = [
        Resource("style.css", 50, 80),
        Resource("app.js", 200, 150),
        Resource("vendor.js", 500, 300),
        Resource("hero.jpg", 800, 400),
        Resource("logo.png", 30, 50),
        Resource("icon1.svg", 5, 30),
        Resource("icon2.svg", 5, 30),
        Resource("font.woff2", 100, 120),
        Resource("thumb1.jpg", 150, 180),
        Resource("thumb2.jpg", 150, 180),
    ]

    print(f"Page requires {len(resources)} resources")
    print(f"Total size: {sum(r.size_kb for r in resources)} KB")
    print()

    # HTTP/1.1
    h1_time, h1_timeline = simulate_http11(resources)
    print(f"HTTP/1.1 (6 parallel connections):")
    print(f"  Total page load time: {h1_time:.0f} ms")
    for name, start, end in sorted(h1_timeline, key=lambda x: x[1]):
        bar_start = int(start / 20)
        bar_len = max(1, int((end - start) / 20))
        bar = " " * bar_start + "#" * bar_len
        print(f"  {name:>15}: [{bar:<30}] {start:.0f}-{end:.0f}ms")

    print()

    # HTTP/2
    h2_time, h2_timeline = simulate_http2(resources)
    print(f"HTTP/2 (multiplexed on 1 connection):")
    print(f"  Total page load time: {h2_time:.0f} ms")
    for name, start, end in sorted(h2_timeline, key=lambda x: x[2]):
        bar_len = max(1, int((end - start) / 20))
        bar = "#" * bar_len
        print(f"  {name:>15}: [{bar:<30}] {start:.0f}-{end:.0f}ms")

    improvement = (1 - h2_time / h1_time) * 100
    print(f"\nHTTP/2 improvement: {improvement:.1f}% faster")

    # Head-of-line blocking demonstration
    print("\n--- Head-of-line Blocking Demo ---")
    slow_resources = list(resources)
    # Make one resource very slow (simulating a slow API call)
    slow_resources[2] = Resource("slow_api.js", 500, 2000)

    h1_slow, _ = simulate_http11(slow_resources)
    h2_slow, _ = simulate_http2(slow_resources)
    print(f"With one 2-second slow resource:")
    print(f"  HTTP/1.1: {h1_slow:.0f} ms (slow resource blocks a connection)")
    print(f"  HTTP/2:   {h2_slow:.0f} ms (other resources unaffected)")


# === Exercise 3: REST vs gRPC Payload Comparison ===
# Problem: Compare JSON (REST) vs binary (protobuf-like) serialization.

def json_serialize(record):
    """Serialize a record as JSON (REST-style)."""
    return json.dumps(record).encode("utf-8")


def binary_serialize(record):
    """Simulate Protocol Buffers encoding using struct.pack.

    Simplified binary format:
    - Strings: 2-byte length prefix + UTF-8 bytes
    - Integers: 4-byte int32
    - Lists: 2-byte count + each element
    """
    parts = []

    # Field 1: name (string)
    name_bytes = record["name"].encode("utf-8")
    parts.append(struct.pack("!H", len(name_bytes)) + name_bytes)

    # Field 2: age (int32)
    parts.append(struct.pack("!I", record["age"]))

    # Field 3: email (string)
    email_bytes = record["email"].encode("utf-8")
    parts.append(struct.pack("!H", len(email_bytes)) + email_bytes)

    # Field 4: addresses (list of strings)
    addresses = record["addresses"]
    parts.append(struct.pack("!H", len(addresses)))
    for addr in addresses:
        addr_bytes = addr.encode("utf-8")
        parts.append(struct.pack("!H", len(addr_bytes)) + addr_bytes)

    return b"".join(parts)


def exercise_3():
    """REST vs gRPC payload size comparison."""
    sample_record = {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com",
        "addresses": [
            "123 Main St, New York, NY 10001",
            "456 Oak Ave, San Francisco, CA 94102",
        ],
    }

    print("Single Record Comparison:")
    print("-" * 50)
    json_bytes = json_serialize(sample_record)
    binary_bytes = binary_serialize(sample_record)
    print(f"  JSON size:   {len(json_bytes)} bytes")
    print(f"  Binary size: {len(binary_bytes)} bytes")
    print(f"  Savings:     {(1 - len(binary_bytes)/len(json_bytes)):.1%}")

    # Compare for 1, 10, 100 records
    print(f"\n{'Records':>10} | {'JSON (bytes)':>14} | {'Binary (bytes)':>16} | {'Savings':>8}")
    print("-" * 60)

    for count in [1, 10, 100]:
        records = [sample_record] * count

        json_total = len(json.dumps(records).encode("utf-8"))
        binary_total = sum(len(binary_serialize(r)) for r in records)
        # Add 4-byte count prefix for binary list
        binary_total += 4

        savings = (1 - binary_total / json_total) * 100
        print(f"{count:>10} | {json_total:>14,} | {binary_total:>16,} | {savings:>7.1f}%")

    print("\nAnalysis:")
    print("  - Binary encoding is ~30-50% smaller than JSON")
    print("  - The savings compound with more records")
    print("  - For high-throughput microservices (1000s of calls/sec),")
    print("    this translates to significant bandwidth savings")
    print("  - For simple/infrequent APIs, JSON readability may outweigh size savings")


# === Exercise 4: HTTP Version Selection ===
# Problem: Choose appropriate HTTP version for different scenarios.

def exercise_4():
    """HTTP version selection for different scenarios."""
    scenarios = [
        {
            "scenario": "Mobile app (frequent network switches)",
            "choice": "HTTP/3",
            "reason": "Connection Migration survives WiFi/cellular switches. "
                      "0-RTT reconnection for fast recovery. Better battery efficiency.",
        },
        {
            "scenario": "Legacy system integration",
            "choice": "HTTP/1.1",
            "reason": "Widest compatibility. No conflicts with existing infrastructure. "
                      "Can upgrade gradually if needed.",
        },
        {
            "scenario": "News site with many images",
            "choice": "HTTP/2",
            "reason": "Multiplexing loads many images simultaneously over one connection. "
                      "Header compression (HPACK) reduces overhead. "
                      "Server Push can pre-send critical CSS/JS.",
        },
    ]

    print("HTTP Version Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['scenario']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 5: REST vs gRPC Selection ===
# Problem: Choose communication method for 10 microservices in Go, Java, Python.

def exercise_5():
    """REST vs gRPC selection for microservices."""
    print("Communication Method Selection:")
    print("=" * 60)
    print("\nScenario:")
    print("  - 10 internal services")
    print("  - Languages: Go, Java, Python")
    print("  - Need real-time data synchronization")

    print("\nRecommendation: gRPC")
    print()
    reasons = [
        ("Internal communication", "Not public-facing, no browser support needed"),
        ("Multi-language", "Go, Java, Python all have first-class gRPC support. "
         "Protobuf auto-generates client/server code for consistency."),
        ("Real-time sync", "Bidirectional streaming for real-time data exchange. "
         "Server streaming for event push."),
        ("Performance", "Efficient for high-frequency inter-service communication. "
         "Protobuf binary serialization saves bandwidth."),
    ]

    for i, (title, detail) in enumerate(reasons, 1):
        print(f"  {i}. {title}")
        print(f"     {detail}")

    print("\nImplementation notes:")
    print("  - Manage common .proto files in a shared repository")
    print("  - Generate gRPC client/server code per service")
    print("  - Use REST (HTTP/JSON) only for external-facing APIs")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: DNS Resolution Tracer ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: HTTP/1.1 vs HTTP/2 Comparison ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: REST vs gRPC Payload Comparison ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: HTTP Version Selection ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: REST vs gRPC Selection ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
