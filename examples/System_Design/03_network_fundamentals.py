"""
Network Fundamentals — DNS, TCP, HTTP Lifecycle Simulation

Demonstrates:
- DNS resolution process (recursive and iterative)
- TCP three-way handshake
- HTTP request/response lifecycle
- Latency breakdown of a full web request

Theory:
- DNS translates domain names to IP addresses through a hierarchy:
  browser cache → OS cache → recursive resolver → root → TLD → authoritative.
- TCP establishes reliable connections via a three-way handshake
  (SYN → SYN-ACK → ACK), adding 1 RTT before data transfer.
- TLS adds another 1-2 RTTs for key exchange.
- HTTP/1.1 uses persistent connections; HTTP/2 adds multiplexing;
  HTTP/3 uses QUIC (UDP) to eliminate head-of-line blocking.

Adapted from System Design Lesson 03.
"""

from dataclasses import dataclass, field
from enum import Enum


# ── DNS Resolution Simulation ─────────────────────────────────────────

@dataclass
class DNSRecord:
    domain: str
    ip: str
    ttl: int       # seconds
    record_type: str = "A"


class DNSCache:
    """Simple DNS cache with TTL."""

    def __init__(self):
        self.cache: dict[str, tuple[str, int]] = {}  # domain → (ip, expiry_tick)
        self.tick = 0

    def get(self, domain: str) -> str | None:
        if domain in self.cache:
            ip, expiry = self.cache[domain]
            if self.tick < expiry:
                return ip
            del self.cache[domain]
        return None

    def put(self, domain: str, ip: str, ttl: int) -> None:
        self.cache[domain] = (ip, self.tick + ttl)


# Why: DNS resolution involves multiple network hops in the worst case.
# Understanding this hierarchy explains why CDNs and DNS caching matter so
# much — a cold DNS lookup can add 100-200ms before any data transfer begins.
class DNSResolver:
    """Simulates recursive DNS resolution."""

    def __init__(self):
        self.local_cache = DNSCache()
        self.hops: list[str] = []

        # Simulated authoritative records
        self.authoritative: dict[str, DNSRecord] = {
            "example.com": DNSRecord("example.com", "93.184.216.34", 3600),
            "api.example.com": DNSRecord("api.example.com", "93.184.216.35", 300),
            "cdn.example.com": DNSRecord("cdn.example.com", "104.16.132.229", 60),
        }

    def resolve(self, domain: str) -> tuple[str | None, list[str]]:
        """Resolve domain to IP, returning (ip, hops)."""
        self.hops = []

        # Step 1: Check local cache
        cached = self.local_cache.get(domain)
        if cached:
            self.hops.append(f"Local cache HIT: {domain} → {cached}")
            return cached, self.hops

        self.hops.append(f"Local cache MISS: {domain}")

        # Step 2: Query root nameserver
        self.hops.append(f"Query root NS: . → knows .com TLD")

        # Step 3: Query TLD nameserver
        tld = domain.split(".")[-1]
        self.hops.append(f"Query TLD NS: .{tld} → knows {domain} authoritative")

        # Step 4: Query authoritative nameserver
        if domain in self.authoritative:
            record = self.authoritative[domain]
            self.hops.append(
                f"Query authoritative NS: {domain} → {record.ip} (TTL={record.ttl}s)"
            )
            self.local_cache.put(domain, record.ip, record.ttl)
            return record.ip, self.hops

        self.hops.append(f"NXDOMAIN: {domain} not found")
        return None, self.hops


# ── TCP Handshake Simulation ──────────────────────────────────────────

class TCPState(Enum):
    CLOSED = "CLOSED"
    SYN_SENT = "SYN_SENT"
    SYN_RECEIVED = "SYN_RECEIVED"
    ESTABLISHED = "ESTABLISHED"
    FIN_WAIT = "FIN_WAIT"
    TIME_WAIT = "TIME_WAIT"


@dataclass
class TCPConnection:
    """Simulates TCP three-way handshake and connection lifecycle."""
    client_state: TCPState = TCPState.CLOSED
    server_state: TCPState = TCPState.CLOSED
    client_seq: int = 0
    server_seq: int = 0
    events: list[str] = field(default_factory=list)
    rtt_ms: float = 50.0

    # Why: The three-way handshake costs exactly 1.5 RTTs (SYN, SYN-ACK, ACK)
    # before any data can be sent. This is why connection reuse (keep-alive)
    # and connection pooling are critical for performance.
    def handshake(self) -> float:
        """Perform three-way handshake. Returns total time in ms."""
        import random
        self.client_seq = random.randint(1000, 9999)
        self.server_seq = random.randint(1000, 9999)
        total_ms = 0.0

        # SYN
        self.client_state = TCPState.SYN_SENT
        self.events.append(
            f"Client → Server: SYN (seq={self.client_seq})"
        )
        total_ms += self.rtt_ms / 2

        # SYN-ACK
        self.server_state = TCPState.SYN_RECEIVED
        self.events.append(
            f"Server → Client: SYN-ACK (seq={self.server_seq}, "
            f"ack={self.client_seq + 1})"
        )
        total_ms += self.rtt_ms / 2

        # ACK
        self.client_state = TCPState.ESTABLISHED
        self.server_state = TCPState.ESTABLISHED
        self.events.append(
            f"Client → Server: ACK (ack={self.server_seq + 1})"
        )
        total_ms += self.rtt_ms / 2

        return total_ms

    def close(self) -> float:
        """Four-way connection teardown."""
        total_ms = 0.0
        self.events.append(f"Client → Server: FIN")
        self.client_state = TCPState.FIN_WAIT
        total_ms += self.rtt_ms / 2

        self.events.append(f"Server → Client: ACK")
        total_ms += self.rtt_ms / 2

        self.events.append(f"Server → Client: FIN")
        total_ms += self.rtt_ms / 2

        self.events.append(f"Client → Server: ACK")
        self.client_state = TCPState.TIME_WAIT
        self.server_state = TCPState.CLOSED
        total_ms += self.rtt_ms / 2

        return total_ms


# ── HTTP Lifecycle ────────────────────────────────────────────────────

@dataclass
class LatencyBreakdown:
    """Full latency breakdown of an HTTP request."""
    dns_ms: float = 0.0
    tcp_handshake_ms: float = 0.0
    tls_handshake_ms: float = 0.0
    request_send_ms: float = 0.0
    server_processing_ms: float = 0.0
    response_receive_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (self.dns_ms + self.tcp_handshake_ms + self.tls_handshake_ms +
                self.request_send_ms + self.server_processing_ms +
                self.response_receive_ms)


# Why: Breaking down an HTTP request into its component phases reveals where
# time is actually spent. For a cold connection, DNS + TCP + TLS can easily
# exceed the server processing time — motivating CDNs, connection reuse,
# and HTTP/2 multiplexing.
def simulate_http_request(url: str, rtt_ms: float = 50.0,
                          dns_cached: bool = False,
                          connection_reused: bool = False,
                          use_tls: bool = True) -> LatencyBreakdown:
    """Simulate full HTTP request lifecycle."""
    breakdown = LatencyBreakdown()

    # DNS
    if dns_cached:
        breakdown.dns_ms = 0.1  # local cache lookup
    else:
        breakdown.dns_ms = rtt_ms * 2  # ~2 RTTs for recursive resolution

    # TCP
    if connection_reused:
        breakdown.tcp_handshake_ms = 0.0
    else:
        breakdown.tcp_handshake_ms = rtt_ms * 1.5  # 3-way handshake

    # TLS
    if use_tls and not connection_reused:
        breakdown.tls_handshake_ms = rtt_ms * 2  # TLS 1.2: 2 RTTs

    # Request + Response
    breakdown.request_send_ms = rtt_ms / 2
    breakdown.server_processing_ms = 20.0  # typical server processing
    breakdown.response_receive_ms = rtt_ms / 2

    return breakdown


# ── Demos ─────────────────────────────────────────────────────────────

def demo_dns():
    print("=" * 60)
    print("DNS RESOLUTION PROCESS")
    print("=" * 60)

    resolver = DNSResolver()

    # Cold resolution
    print(f"\n  --- First lookup (cold cache) ---")
    ip, hops = resolver.resolve("example.com")
    for hop in hops:
        print(f"    {hop}")
    print(f"  Result: {ip}")

    # Cached resolution
    print(f"\n  --- Second lookup (cached) ---")
    ip, hops = resolver.resolve("example.com")
    for hop in hops:
        print(f"    {hop}")
    print(f"  Result: {ip}")

    # Different subdomains
    print(f"\n  --- Subdomain lookup ---")
    ip, hops = resolver.resolve("api.example.com")
    for hop in hops:
        print(f"    {hop}")
    print(f"  Result: {ip}")


def demo_tcp_handshake():
    print("\n" + "=" * 60)
    print("TCP THREE-WAY HANDSHAKE")
    print("=" * 60)

    conn = TCPConnection(rtt_ms=50.0)
    handshake_time = conn.handshake()

    print(f"\n  RTT: {conn.rtt_ms} ms")
    print(f"\n  Handshake sequence:")
    for i, event in enumerate(conn.events):
        print(f"    {i+1}. {event}")
    print(f"\n  Handshake time: {handshake_time:.1f} ms (1.5 RTTs)")
    print(f"  Client state: {conn.client_state.value}")
    print(f"  Server state: {conn.server_state.value}")

    # Teardown
    conn.events.clear()
    close_time = conn.close()
    print(f"\n  Connection teardown:")
    for i, event in enumerate(conn.events):
        print(f"    {i+1}. {event}")
    print(f"  Teardown time: {close_time:.1f} ms (2 RTTs)")


def demo_http_lifecycle():
    print("\n" + "=" * 60)
    print("HTTP REQUEST LATENCY BREAKDOWN")
    print("=" * 60)

    scenarios = [
        ("Cold HTTPS request",     False, False, True),
        ("DNS cached, new conn",   True,  False, True),
        ("Reused connection",      True,  True,  True),
        ("HTTP (no TLS)",          False, False, False),
    ]

    rtt = 50.0
    print(f"\n  RTT: {rtt} ms\n")
    print(f"  {'Scenario':<28} {'DNS':>6} {'TCP':>6} {'TLS':>6} "
          f"{'Req':>6} {'Proc':>6} {'Resp':>6} {'Total':>8}")
    print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*8}")

    for name, dns_c, conn_r, tls in scenarios:
        b = simulate_http_request("https://example.com", rtt, dns_c, conn_r, tls)
        print(f"  {name:<28} {b.dns_ms:>5.0f} {b.tcp_handshake_ms:>5.0f} "
              f"{b.tls_handshake_ms:>5.0f} {b.request_send_ms:>5.0f} "
              f"{b.server_processing_ms:>5.0f} {b.response_receive_ms:>5.0f} "
              f"{b.total_ms:>7.0f} ms")

    print(f"\n  Key takeaways:")
    print(f"    - Cold HTTPS adds ~{rtt * 5.5:.0f} ms of overhead before server "
          f"even processes")
    print(f"    - Connection reuse saves {rtt * 3.5:.0f} ms per request")
    print(f"    - DNS caching saves {rtt * 2:.0f} ms per request")


def demo_protocol_comparison():
    print("\n" + "=" * 60)
    print("HTTP PROTOCOL COMPARISON")
    print("=" * 60)

    print(f"\n  {'Feature':<30} {'HTTP/1.1':>12} {'HTTP/2':>12} {'HTTP/3':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    features = [
        ("Transport",           "TCP",          "TCP",          "QUIC (UDP)"),
        ("Multiplexing",        "No",           "Yes",          "Yes"),
        ("Head-of-line block",  "Yes",          "TCP-level",    "No"),
        ("Header compression",  "No",           "HPACK",        "QPACK"),
        ("Server push",         "No",           "Yes",          "Yes"),
        ("Connection setup",    "1-2 RTT",      "1-2 RTT",     "0-1 RTT"),
        ("Connection migration","No",           "No",           "Yes"),
    ]
    for feat, h1, h2, h3 in features:
        print(f"  {feat:<30} {h1:>12} {h2:>12} {h3:>12}")


if __name__ == "__main__":
    demo_dns()
    demo_tcp_handshake()
    demo_http_lifecycle()
    demo_protocol_comparison()
