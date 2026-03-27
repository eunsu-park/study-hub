"""
DNS Resolver Simulation

Demonstrates:
- DNS record types (A, AAAA, CNAME, MX, NS)
- Recursive vs iterative resolution
- DNS caching with TTL
- DNS hierarchy (root → TLD → authoritative)

Theory:
- DNS maps domain names to IP addresses (and other records).
- Hierarchy: Root servers → TLD servers (.com, .org) →
  Authoritative servers (example.com).
- Recursive resolver: queries on behalf of client, follows
  referrals until it gets the answer.
- Caching: responses cached for TTL seconds, reducing
  query load on authoritative servers.

Adapted from Networking Lesson 12.
"""

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DNSRecord:
    name: str
    rtype: str  # A, AAAA, CNAME, MX, NS
    value: str
    ttl: int = 3600
    priority: int = 0  # For MX records


class DNSZone:
    """A DNS zone with its records."""

    def __init__(self, name: str):
        self.name = name
        self.records: list[DNSRecord] = []

    def add_record(self, name: str, rtype: str, value: str,
                   ttl: int = 3600, priority: int = 0) -> None:
        self.records.append(DNSRecord(name, rtype, value, ttl, priority))

    def query(self, name: str, rtype: str) -> list[DNSRecord]:
        return [r for r in self.records
                if r.name == name and r.rtype == rtype]

    def query_any(self, name: str) -> list[DNSRecord]:
        return [r for r in self.records if r.name == name]


class DNSServer:
    """A DNS server that hosts one or more zones."""

    def __init__(self, name: str):
        self.name = name
        self.zones: dict[str, DNSZone] = {}

    def add_zone(self, zone: DNSZone) -> None:
        self.zones[zone.name] = zone

    def resolve(self, name: str, rtype: str) -> tuple[list[DNSRecord], str]:
        """Try to resolve. Returns (records, status).
        Status: 'ANSWER', 'REFERRAL', 'NXDOMAIN'
        """
        # Why: A DNS server may host multiple zones (e.g., example.com and
        # sub.example.com). We select the most specific (longest) zone name
        # that matches, mirroring how authoritative servers determine which
        # zone file is responsible for a given query name.
        best_zone = None
        for zone_name, zone in self.zones.items():
            if name.endswith(zone_name) or name == zone_name:
                if best_zone is None or len(zone_name) > len(best_zone.name):
                    best_zone = zone

        if best_zone is None:
            return [], "NXDOMAIN"

        # Direct match
        records = best_zone.query(name, rtype)
        if records:
            return records, "ANSWER"

        # Why: CNAME records act as aliases — if the exact name has no direct
        # record of the requested type, we check for a CNAME and restart the
        # query for the canonical name. This indirection is how "www.example.com"
        # can point to "example.com" without duplicating A/AAAA records.
        cnames = best_zone.query(name, "CNAME")
        if cnames:
            return cnames, "CNAME"

        # NS referral (for subdomains)
        parts = name.split(".")
        for i in range(len(parts)):
            sub = ".".join(parts[i:])
            ns_records = best_zone.query(sub, "NS")
            if ns_records and sub != best_zone.name:
                return ns_records, "REFERRAL"

        return [], "NXDOMAIN"


# Why: DNS caching is essential for performance — without it, every DNS query
# would traverse the entire hierarchy (root → TLD → authoritative), adding
# ~100ms+ latency. TTL-based expiration ensures stale records are eventually
# refreshed, balancing performance against DNS record update propagation.
class DNSCache:
    """Simple DNS cache with TTL."""

    def __init__(self):
        self.cache: dict[tuple[str, str], tuple[list[DNSRecord], float]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, name: str, rtype: str, current_time: float = 0) -> list[DNSRecord] | None:
        key = (name, rtype)
        if key in self.cache:
            records, expire_time = self.cache[key]
            if current_time < expire_time:
                self.hits += 1
                return records
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, name: str, rtype: str, records: list[DNSRecord],
            current_time: float = 0) -> None:
        if records:
            # Why: We use the minimum TTL from the record set as the cache
            # expiration. If one record expires sooner, the entire set should
            # be re-queried to avoid serving a mix of fresh and stale data.
            min_ttl = min(r.ttl for r in records)
            self.cache[(name, rtype)] = (records, current_time + min_ttl)


# Why: A recursive resolver (like 8.8.8.8 or 1.1.1.1) does all the work
# on behalf of the client — chasing referrals from root to TLD to
# authoritative. The alternative (iterative resolution) pushes this work
# to the client, which is impractical for end-user devices.
class RecursiveResolver:
    """Recursive DNS resolver."""

    def __init__(self, servers: dict[str, DNSServer]):
        self.servers = servers
        self.cache = DNSCache()
        self.query_log: list[str] = []

    def resolve(self, name: str, rtype: str = "A",
                current_time: float = 0) -> list[DNSRecord]:
        """Recursively resolve a DNS query."""
        self.query_log = []

        # Check cache
        cached = self.cache.get(name, rtype, current_time)
        if cached:
            self.query_log.append(f"  CACHE HIT: {name} {rtype}")
            return cached

        # Start from root
        result = self._recursive_query(name, rtype, "root")

        if result:
            self.cache.put(name, rtype, result, current_time)
        return result

    def _recursive_query(self, name: str, rtype: str,
                         server_name: str) -> list[DNSRecord]:
        server = self.servers.get(server_name)
        if not server:
            return []

        self.query_log.append(f"  Query {server_name}: {name} {rtype}")
        records, status = server.resolve(name, rtype)

        if status == "ANSWER":
            self.query_log.append(f"  Answer from {server_name}: "
                                  f"{[r.value for r in records]}")
            return records

        if status == "CNAME":
            self.query_log.append(f"  CNAME: {name} → {records[0].value}")
            return self._recursive_query(records[0].value, rtype, "root")

        if status == "REFERRAL":
            ns_name = records[0].value
            self.query_log.append(f"  Referral to {ns_name}")
            # Find the NS server
            for sname, srv in self.servers.items():
                if sname == ns_name or any(
                    z.name == ns_name for z in srv.zones.values()
                ):
                    return self._recursive_query(name, rtype, sname)

        return []


# ── Build DNS Infrastructure ──────────────────────────────────────────

def build_dns_system() -> dict[str, DNSServer]:
    """Build a simulated DNS hierarchy."""
    servers = {}

    # Root server
    root = DNSServer("root")
    root_zone = DNSZone(".")
    root_zone.add_record("com", "NS", "tld-com")
    root_zone.add_record("org", "NS", "tld-org")
    root.add_zone(root_zone)
    servers["root"] = root

    # .com TLD server
    tld_com = DNSServer("tld-com")
    com_zone = DNSZone("com")
    com_zone.add_record("example.com", "NS", "ns1-example")
    com_zone.add_record("google.com", "NS", "ns1-google")
    tld_com.add_zone(com_zone)
    servers["tld-com"] = tld_com

    # .org TLD server
    tld_org = DNSServer("tld-org")
    org_zone = DNSZone("org")
    org_zone.add_record("wikipedia.org", "NS", "ns1-wikipedia")
    tld_org.add_zone(org_zone)
    servers["tld-org"] = tld_org

    # example.com authoritative
    ns_example = DNSServer("ns1-example")
    ex_zone = DNSZone("example.com")
    ex_zone.add_record("example.com", "A", "93.184.216.34")
    ex_zone.add_record("www.example.com", "CNAME", "example.com")
    ex_zone.add_record("mail.example.com", "A", "93.184.216.35")
    ex_zone.add_record("example.com", "MX", "mail.example.com",
                       priority=10)
    ex_zone.add_record("example.com", "AAAA", "2606:2800:220:1:248:1893:25c8:1946")
    ns_example.add_zone(ex_zone)
    servers["ns1-example"] = ns_example

    # google.com authoritative
    ns_google = DNSServer("ns1-google")
    g_zone = DNSZone("google.com")
    g_zone.add_record("google.com", "A", "142.250.80.46")
    g_zone.add_record("www.google.com", "A", "142.250.80.46")
    g_zone.add_record("mail.google.com", "CNAME", "googlemail.l.google.com")
    g_zone.add_record("googlemail.l.google.com", "A", "142.250.80.17")
    ns_google.add_zone(g_zone)
    servers["ns1-google"] = ns_google

    # wikipedia.org authoritative
    ns_wiki = DNSServer("ns1-wikipedia")
    w_zone = DNSZone("wikipedia.org")
    w_zone.add_record("wikipedia.org", "A", "208.80.154.224")
    w_zone.add_record("en.wikipedia.org", "A", "208.80.154.224")
    ns_wiki.add_zone(w_zone)
    servers["ns1-wikipedia"] = ns_wiki

    return servers


# ── Demos ──────────────────────────────────────────────────────────────

def demo_recursive_resolution():
    print("=" * 60)
    print("RECURSIVE DNS RESOLUTION")
    print("=" * 60)

    servers = build_dns_system()
    resolver = RecursiveResolver(servers)

    queries = [
        ("example.com", "A"),
        ("www.example.com", "A"),  # CNAME chase
        ("google.com", "A"),
        ("en.wikipedia.org", "A"),
    ]

    for name, rtype in queries:
        print(f"\n  Resolving {name} ({rtype}):")
        results = resolver.resolve(name, rtype)
        for entry in resolver.query_log:
            print(f"  {entry}")
        if results:
            for r in results:
                print(f"  Result: {r.name} → {r.value}")
        else:
            print(f"  NXDOMAIN")


def demo_caching():
    print("\n" + "=" * 60)
    print("DNS CACHING")
    print("=" * 60)

    servers = build_dns_system()
    resolver = RecursiveResolver(servers)

    print(f"\n  First query (cache miss):")
    results = resolver.resolve("example.com", "A", current_time=0)
    for entry in resolver.query_log:
        print(f"  {entry}")
    print(f"    Queries made: {len([l for l in resolver.query_log if l.startswith('  Query')])}")

    print(f"\n  Second query (cache hit):")
    results = resolver.resolve("example.com", "A", current_time=1)
    for entry in resolver.query_log:
        print(f"  {entry}")

    print(f"\n  After TTL expires (t=4000):")
    results = resolver.resolve("example.com", "A", current_time=4000)
    for entry in resolver.query_log:
        print(f"  {entry}")
    print(f"    Queries made: {len([l for l in resolver.query_log if l.startswith('  Query')])}")

    cache = resolver.cache
    print(f"\n  Cache stats: hits={cache.hits}, misses={cache.misses}")


def demo_record_types():
    print("\n" + "=" * 60)
    print("DNS RECORD TYPES")
    print("=" * 60)

    servers = build_dns_system()
    ns = servers["ns1-example"]

    records = ns.zones["example.com"].records
    print(f"\n  example.com zone records:")
    print(f"    {'Name':<25} {'Type':<8} {'Value':<40} {'TTL':>5}")
    print(f"    {'-'*25} {'-'*8} {'-'*40} {'-'*5}")
    for r in records:
        print(f"    {r.name:<25} {r.rtype:<8} {r.value:<40} {r.ttl:>5}")

    print(f"""
  Record Types:
    A      IPv4 address
    AAAA   IPv6 address
    CNAME  Canonical name (alias)
    MX     Mail exchange (with priority)
    NS     Name server (delegation)
    TXT    Text (SPF, DKIM, verification)
    SOA    Start of authority (zone metadata)""")


if __name__ == "__main__":
    demo_recursive_resolution()
    demo_caching()
    demo_record_types()
