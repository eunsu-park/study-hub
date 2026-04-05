"""
Exercises for Lesson 02: Scalability Basics
Topic: System_Design

Solutions to practice problems and hands-on exercises from the lesson.
Covers scaling strategies, stateless/stateful architecture, CAP theorem,
and PACELC analysis.
"""

import random
import time
import math
from collections import defaultdict


# === Exercise 1: Choose Scaling Method ===
# Problem: Choose appropriate scaling method for these scenarios.
# a) Early startup, 1,000 daily users
# b) Large e-commerce, 10x traffic on Black Friday
# c) Database handling complex analytics queries
# d) CDN edge servers

def exercise_1():
    """Scaling method selection for different scenarios."""
    scenarios = [
        {
            "name": "Early startup, 1,000 daily users",
            "choice": "Vertical Scaling",
            "reason": "Small scale, simple approach is most cost-efficient. "
                      "No need for distributed system complexity yet.",
        },
        {
            "name": "Large e-commerce, 10x traffic on Black Friday",
            "choice": "Horizontal Scaling",
            "reason": "Need elastic auto-scaling to handle traffic spikes. "
                      "Scale out during peak, scale in after. Cost-efficient.",
        },
        {
            "name": "Database handling complex analytics queries",
            "choice": "Vertical Scaling",
            "reason": "Analytics queries benefit from more CPU/RAM on a single node. "
                      "Cross-node joins are expensive in distributed DBs.",
        },
        {
            "name": "CDN edge servers",
            "choice": "Horizontal Scaling",
            "reason": "Need geographic distribution across many locations. "
                      "Each edge server is independent, perfect for scale-out.",
        },
    ]

    print("Scaling Method Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)  # a, b, c, d
        print(f"\n{label}) {s['name']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 2: Stateless Service Simulation ===
# Problem: Design shopping cart feature for stateless architecture.
# Hands-On: Demonstrate difference between stateful and stateless servers.

class StatefulServer:
    """Server that stores session data in local memory."""

    def __init__(self, server_id):
        self.server_id = server_id
        self.sessions = {}  # session_id -> cart data

    def handle_request(self, session_id, action, item=None):
        if action == "add":
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append(item)
            return f"Server {self.server_id}: Added '{item}' to cart"
        elif action == "view":
            cart = self.sessions.get(session_id, None)
            if cart is None:
                return f"Server {self.server_id}: SESSION NOT FOUND!"
            return f"Server {self.server_id}: Cart = {cart}"


class StatelessServer:
    """Server that reads from an external session store."""

    def __init__(self, server_id, session_store):
        self.server_id = server_id
        self.store = session_store  # shared external store

    def handle_request(self, session_id, action, item=None):
        if action == "add":
            if session_id not in self.store:
                self.store[session_id] = []
            self.store[session_id].append(item)
            return f"Server {self.server_id}: Added '{item}' to cart"
        elif action == "view":
            cart = self.store.get(session_id, [])
            return f"Server {self.server_id}: Cart = {cart}"


class LoadBalancer:
    """Routes requests randomly across servers."""

    def __init__(self, servers):
        self.servers = servers

    def route(self, session_id, action, item=None):
        server = random.choice(self.servers)
        return server.handle_request(session_id, action, item)


def exercise_2():
    """Stateless vs stateful architecture simulation."""
    random.seed(42)

    # --- Stateful architecture ---
    print("--- Stateful Architecture ---")
    stateful_servers = [StatefulServer(i) for i in range(3)]
    stateful_lb = LoadBalancer(stateful_servers)

    session_id = "user_123"
    print(stateful_lb.route(session_id, "add", "laptop"))
    print(stateful_lb.route(session_id, "add", "mouse"))
    print(stateful_lb.route(session_id, "view"))
    print(stateful_lb.route(session_id, "view"))

    # Measure session loss rate
    print("\nSession loss rate test (1000 requests):")
    loss_count = 0
    total_views = 0
    for _ in range(100):
        sid = f"user_{random.randint(1, 50)}"
        # Add item to random server
        stateful_lb.route(sid, "add", "item")
        # View from random server
        for _ in range(10):
            result = stateful_lb.route(sid, "view")
            total_views += 1
            if "SESSION NOT FOUND" in result:
                loss_count += 1
    print(f"Total views: {total_views}, Session losses: {loss_count}")
    print(f"Session loss rate: {loss_count / total_views:.1%}")

    # --- Stateless architecture ---
    print("\n--- Stateless Architecture ---")
    shared_store = {}  # External store (simulating Redis)
    stateless_servers = [StatelessServer(i, shared_store) for i in range(3)]
    stateless_lb = LoadBalancer(stateless_servers)

    session_id = "user_123"
    print(stateless_lb.route(session_id, "add", "laptop"))
    print(stateless_lb.route(session_id, "add", "mouse"))
    print(stateless_lb.route(session_id, "view"))
    print(stateless_lb.route(session_id, "view"))

    # Measure session loss rate (should be 0%)
    print("\nSession loss rate test (1000 requests):")
    shared_store.clear()
    loss_count = 0
    total_views = 0
    for _ in range(100):
        sid = f"user_{random.randint(1, 50)}"
        stateless_lb.route(sid, "add", "item")
        for _ in range(10):
            result = stateless_lb.route(sid, "view")
            total_views += 1
            if "SESSION NOT FOUND" in result:
                loss_count += 1
    print(f"Total views: {total_views}, Session losses: {loss_count}")
    print(f"Session loss rate: {loss_count / total_views:.1%}")


# === Exercise 3: CAP Theorem Choice ===
# Problem: Choose CP or AP for these services.

def exercise_3():
    """CAP theorem analysis for different services."""
    scenarios = [
        {
            "service": "Bank transfer system",
            "choice": "CP",
            "reason": "Balance accuracy is critical. Users would rather see "
                      "'service unavailable' than incorrect balance. "
                      "Strong consistency prevents double spending.",
        },
        {
            "service": "Facebook like count",
            "choice": "AP",
            "reason": "Service must continue even if like counts differ temporarily "
                      "between nodes. Eventual consistency is acceptable. "
                      "Users tolerate seeing 1,523 vs 1,525 likes.",
        },
        {
            "service": "Online booking system",
            "choice": "CP",
            "reason": "Must prevent double booking. Strong consistency ensures "
                      "the same seat/room cannot be sold twice. "
                      "Better to reject a request than create a conflict.",
        },
        {
            "service": "DNS server",
            "choice": "AP",
            "reason": "Responding with slightly stale data is far better than "
                      "not responding at all. DNS is naturally eventually consistent "
                      "with TTL-based updates.",
        },
    ]

    print("CAP Theorem Analysis:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['service']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 4: PACELC Analysis ===
# Problem: Choose PACELC combination for a global service.
# Conditions:
# - Global service (multiple regions)
# - Read latency < 100ms required
# - Data loss unacceptable

def exercise_4():
    """PACELC analysis for a global service."""
    print("PACELC Analysis:")
    print("=" * 60)
    print("\nRequirements:")
    print("  - Global service (multiple regions)")
    print("  - Read latency < 100ms required")
    print("  - Data loss unacceptable")

    print("\nAnalysis:")
    print("  - Global service -> Partition tolerance (P) is required")
    print("  - Read latency < 100ms -> Latency (L) is important")
    print("  - Data loss unacceptable -> Consistency (C) is important")

    print("\nRecommendation: PC/EL")
    print("  During Partition: Maintain Consistency")
    print("    - Some requests may fail to preserve data integrity")
    print("    - No data loss during network splits")
    print("  During Normal operation (Else): Low Latency")
    print("    - Read from local replicas for fast response")
    print("    - Async replication for read performance")

    print("\nPractical Implementation:")
    print("  - Reads: Local replicas (optimized for latency)")
    print("  - Writes: Synchronous to quorum (optimized for consistency)")
    print("  - Technology: Cassandra with LOCAL_QUORUM or CockroachDB")


# === Exercise 5: Stateless Design - Shopping Cart (Practice Problem) ===
# Problem: Design shopping cart for stateless architecture.

def exercise_5():
    """Shopping cart design for stateless architecture."""
    print("Shopping Cart - Stateless Design:")
    print("=" * 60)
    print("\nRecommended: Redis (Centralized Session Store)")
    print()
    print("Reasons:")
    print("  1. Stateless web servers - same handling regardless of server")
    print("  2. Fast read/write - frequent cart lookups/updates")
    print("  3. TTL support - automatic cart expiration (e.g., 7 days)")
    print("  4. Easy failure recovery - Redis Sentinel/Cluster")

    # Simulate the recommended design
    class RedisCartStore:
        """Simulated Redis-based cart store."""
        def __init__(self):
            self.store = {}
            self.ttls = {}

        def add_item(self, user_id, item, quantity=1):
            key = f"cart:{user_id}"
            if key not in self.store:
                self.store[key] = {}
            self.store[key][item] = self.store[key].get(item, 0) + quantity
            self.ttls[key] = time.time() + 604800  # 7-day TTL

        def get_cart(self, user_id):
            key = f"cart:{user_id}"
            if key in self.store and time.time() < self.ttls.get(key, 0):
                return self.store[key]
            return {}

        def remove_item(self, user_id, item):
            key = f"cart:{user_id}"
            if key in self.store and item in self.store[key]:
                del self.store[key][item]

    store = RedisCartStore()
    store.add_item("user_1", "laptop", 1)
    store.add_item("user_1", "mouse", 2)
    store.add_item("user_1", "keyboard", 1)

    print(f"\nDemo: User cart contents: {store.get_cart('user_1')}")
    store.remove_item("user_1", "mouse")
    print(f"After removing mouse: {store.get_cart('user_1')}")


# === Exercise 6: Scaling Cost Calculator (Hands-On Exercise 3) ===
# Problem: Compare vertical vs. horizontal scaling costs for growing traffic.

def exercise_6():
    """Scaling cost calculator: vertical vs. horizontal."""
    print("Scaling Cost Comparison:")
    print("=" * 60)

    # Vertical scaling: cost doubles per capacity tier
    # (actually more than doubles in practice)
    vertical_tiers = [
        (1, 100),     # 1x capacity, $100
        (2, 250),     # 2x capacity, $250
        (4, 600),     # 4x capacity, $600
        (8, 1400),    # 8x capacity, $1400
        (16, 3200),   # 16x capacity, $3200
    ]

    # Horizontal scaling: N servers at $100 each + $50 LB overhead
    server_cost = 100
    lb_overhead = 50

    print(f"\n{'Capacity':>10} | {'Vertical $':>12} | {'Horizontal $':>14} | {'Winner':>10}")
    print("-" * 55)

    crossover_point = None
    for capacity, v_cost in vertical_tiers:
        h_cost = capacity * server_cost + lb_overhead
        winner = "Vertical" if v_cost <= h_cost else "Horizontal"
        if winner == "Horizontal" and crossover_point is None:
            crossover_point = capacity
        print(f"{capacity:>8}x | ${v_cost:>10,} | ${h_cost:>12,} | {winner:>10}")

    if crossover_point:
        print(f"\nCrossover point: Horizontal becomes cheaper at {crossover_point}x capacity")
    else:
        print("\nVertical is cheaper at all tested tiers")

    # Detailed breakdown
    print("\nDetailed Cost Efficiency ($/capacity unit):")
    for capacity, v_cost in vertical_tiers:
        h_cost = capacity * server_cost + lb_overhead
        v_efficiency = v_cost / capacity
        h_efficiency = h_cost / capacity
        print(f"  {capacity:>2}x: Vertical=${v_efficiency:.0f}/unit, "
              f"Horizontal=${h_efficiency:.1f}/unit")


# === Exercise 7: CAP Theorem Demonstration (Hands-On Exercise 2) ===
# Problem: Build a 3-node key-value store with CP and AP modes.

class DistributedKV:
    """Simple 3-node key-value store demonstrating CP vs AP behavior."""

    def __init__(self, mode="CP"):
        self.mode = mode  # "CP" or "AP"
        self.nodes = {0: {}, 1: {}, 2: {}}
        self.partitioned = set()  # Set of partitioned node IDs
        self.quorum = 2  # Need 2 of 3 for quorum

    def partition(self, node_id):
        """Simulate network partition by isolating a node."""
        self.partitioned.add(node_id)
        return f"Node {node_id} partitioned (isolated)"

    def heal(self, node_id):
        """Heal partition for a node."""
        self.partitioned.discard(node_id)
        # Sync data from healthy nodes
        healthy = [n for n in self.nodes if n not in self.partitioned]
        if healthy:
            source = healthy[0]
            self.nodes[node_id] = dict(self.nodes[source])
        return f"Node {node_id} healed and synced"

    def _available_nodes(self):
        return [n for n in self.nodes if n not in self.partitioned]

    def write(self, key, value):
        """Write a key-value pair."""
        available = self._available_nodes()

        if self.mode == "CP":
            if len(available) < self.quorum:
                return f"WRITE REJECTED: Only {len(available)} nodes available, " \
                       f"need {self.quorum} for quorum"
            # Write to all available nodes (consistent)
            for node_id in available:
                self.nodes[node_id][key] = value
            return f"WRITE OK: {key}={value} written to {len(available)} nodes"
        else:  # AP mode
            # Write to whatever is available
            for node_id in available:
                self.nodes[node_id][key] = value
            return f"WRITE OK: {key}={value} written to {len(available)} nodes " \
                   f"(partitioned nodes have stale data)"

    def read(self, key):
        """Read a key."""
        available = self._available_nodes()

        if self.mode == "CP":
            if len(available) < self.quorum:
                return f"READ REJECTED: Only {len(available)} nodes available"
            # Read from first available (all consistent in CP)
            value = self.nodes[available[0]].get(key, "NOT FOUND")
            return f"READ: {key}={value} (consistent, from quorum)"
        else:  # AP mode
            # Read from first available (might be stale)
            value = self.nodes[available[0]].get(key, "NOT FOUND")
            return f"READ: {key}={value} (may be stale if partition occurred)"


def exercise_7():
    """CAP theorem demonstration with CP vs AP modes."""
    print("CAP Theorem Demonstration:")
    print("=" * 60)

    # --- CP Mode ---
    print("\n--- CP Mode (Consistency + Partition Tolerance) ---")
    cp_kv = DistributedKV(mode="CP")
    print(cp_kv.write("balance", "1000"))
    print(cp_kv.read("balance"))
    print(cp_kv.partition(2))
    print(cp_kv.write("balance", "900"))  # Still works (2/3 quorum)
    print(cp_kv.partition(1))              # Now only 1 node available
    print(cp_kv.write("balance", "800"))  # REJECTED - no quorum
    print(cp_kv.read("balance"))           # REJECTED - no quorum

    # --- AP Mode ---
    print("\n--- AP Mode (Availability + Partition Tolerance) ---")
    ap_kv = DistributedKV(mode="AP")
    print(ap_kv.write("balance", "1000"))
    print(ap_kv.read("balance"))
    print(ap_kv.partition(2))
    print(ap_kv.write("balance", "900"))  # Works on available nodes
    print(ap_kv.partition(1))
    print(ap_kv.write("balance", "800"))  # Still works (1 node)
    print(ap_kv.read("balance"))           # Returns value (might be stale)

    # Show inconsistency
    print("\nNode states after AP writes:")
    for node_id, data in ap_kv.nodes.items():
        status = "PARTITIONED" if node_id in ap_kv.partitioned else "ACTIVE"
        print(f"  Node {node_id} [{status}]: {data}")
    print("  -> Nodes show different values = INCONSISTENCY")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Choose Scaling Method ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Stateless vs Stateful Simulation ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: CAP Theorem Choice ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: PACELC Analysis ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Stateless Shopping Cart Design ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Scaling Cost Calculator ===")
    print("=" * 60)
    exercise_6()

    print("\n" + "=" * 60)
    print("=== Exercise 7: CAP Theorem Demonstration ===")
    print("=" * 60)
    exercise_7()

    print("\nAll exercises completed!")
