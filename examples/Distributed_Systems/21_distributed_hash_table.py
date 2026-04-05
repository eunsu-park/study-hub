"""
Distributed Hash Table: Chord Protocol Simulator

Simulates the Chord distributed hash table protocol where nodes are
organized in a ring and use finger tables for O(log N) routing.
Demonstrates key lookup, node join/leave, and stabilisation.

Key concepts:
- Chord ring: nodes arranged by hash on a circular identifier space
- Finger tables: each node maintains O(log N) routing entries
- Key lookup: O(log N) hops via finger table routing
- Node join: update successor, predecessor, and finger tables
- Stabilisation: periodic protocol to fix routing after churn

Usage:
    python 21_distributed_hash_table.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Chord Node
# ---------------------------------------------------------------------------

M = 8  # Identifier space: 2^M = 256 positions

def chord_hash(key: str) -> int:
    """Hash a key into the Chord identifier space."""
    h = hashlib.sha1(key.encode()).hexdigest()
    return int(h, 16) % (2 ** M)


def in_range(x: int, start: int, end: int, inclusive_end: bool = False) -> bool:
    """Check if x is in (start, end] on the ring (modular arithmetic)."""
    if inclusive_end:
        if start < end:
            return start < x <= end
        elif start > end:
            return x > start or x <= end
        else:
            return True  # Full ring
    else:
        if start < end:
            return start < x < end
        elif start > end:
            return x > start or x < end
        else:
            return x != start


@dataclass
class FingerEntry:
    start: int        # finger[i].start = (n + 2^i) mod 2^M
    node_id: int      # The node responsible for this finger


class ChordNode:
    """A node in the Chord DHT."""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.successor: int = node_id
        self.predecessor: int | None = None
        self.finger_table: list[FingerEntry] = []
        self.data: dict[str, str] = {}

        # Initialize finger table starts
        for i in range(M):
            start = (node_id + 2 ** i) % (2 ** M)
            self.finger_table.append(FingerEntry(start, node_id))

    def find_successor(self, key_id: int, network: "ChordNetwork",
                       hops: int = 0) -> tuple[int, int]:
        """Find the successor node for key_id. Returns (node_id, hops)."""
        if in_range(key_id, self.node_id, self.successor, inclusive_end=True):
            return self.successor, hops

        # Find closest preceding node in finger table
        closest = self._closest_preceding_node(key_id)
        if closest == self.node_id:
            return self.successor, hops

        next_node = network.get_node(closest)
        if next_node:
            return next_node.find_successor(key_id, network, hops + 1)
        return self.successor, hops

    def _closest_preceding_node(self, key_id: int) -> int:
        """Find the closest preceding finger for key_id."""
        for i in range(M - 1, -1, -1):
            finger_id = self.finger_table[i].node_id
            if in_range(finger_id, self.node_id, key_id):
                return finger_id
        return self.node_id


class ChordNetwork:
    """Manages a Chord DHT network."""

    def __init__(self):
        self.nodes: dict[int, ChordNode] = {}

    def get_node(self, node_id: int) -> ChordNode | None:
        return self.nodes.get(node_id)

    def add_node(self, node_id: int) -> ChordNode:
        """Add a new node to the network."""
        node = ChordNode(node_id)
        self.nodes[node_id] = node

        if len(self.nodes) == 1:
            # First node: points to itself
            node.successor = node_id
            node.predecessor = node_id
        else:
            # Find an existing node to bootstrap from
            existing = next(n for nid, n in self.nodes.items() if nid != node_id)
            succ_id, _ = existing.find_successor(node_id, self)
            node.successor = succ_id

            # Update predecessor of successor
            succ_node = self.get_node(succ_id)
            if succ_node:
                node.predecessor = succ_node.predecessor
                succ_node.predecessor = node_id

            # Update predecessor's successor
            if node.predecessor is not None:
                pred_node = self.get_node(node.predecessor)
                if pred_node:
                    pred_node.successor = node_id

        # Build finger tables for all nodes
        self._fix_all_fingers()
        return node

    def _fix_all_fingers(self) -> None:
        """Rebuild finger tables for all nodes."""
        for node in self.nodes.values():
            for i in range(M):
                start = (node.node_id + 2 ** i) % (2 ** M)
                # Find the node responsible for this start position
                succ_id = self._simple_find_successor(start)
                node.finger_table[i].node_id = succ_id

    def _simple_find_successor(self, key_id: int) -> int:
        """Find successor by walking the ring (for finger table building)."""
        sorted_ids = sorted(self.nodes.keys())
        for nid in sorted_ids:
            if nid >= key_id:
                return nid
        return sorted_ids[0]  # Wrap around

    def lookup(self, key: str, from_node: int) -> tuple[int, int]:
        """Look up a key starting from from_node. Returns (responsible_node, hops)."""
        key_id = chord_hash(key)
        node = self.get_node(from_node)
        if node:
            return node.find_successor(key_id, self)
        return -1, -1

    def put(self, key: str, value: str, from_node: int) -> tuple[int, int]:
        """Store a key-value pair. Returns (stored_at_node, hops)."""
        resp_id, hops = self.lookup(key, from_node)
        node = self.get_node(resp_id)
        if node:
            node.data[key] = value
        return resp_id, hops

    def get(self, key: str, from_node: int) -> tuple[str | None, int, int]:
        """Get a value. Returns (value, responsible_node, hops)."""
        resp_id, hops = self.lookup(key, from_node)
        node = self.get_node(resp_id)
        if node:
            return node.data.get(key), resp_id, hops
        return None, resp_id, hops


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_chord_ring() -> None:
    """Build a Chord ring and demonstrate routing."""
    print("=" * 70)
    print(f"Chord DHT (M={M}, ring size={2**M})")
    print("=" * 70)

    net = ChordNetwork()

    # Add nodes at specific positions
    node_ids = [0, 32, 64, 96, 128, 192]
    for nid in node_ids:
        net.add_node(nid)

    print(f"\n  Nodes on ring: {sorted(net.nodes.keys())}")
    print(f"\n  Node details:")
    for nid in sorted(net.nodes.keys()):
        node = net.get_node(nid)
        fingers = [f.node_id for f in node.finger_table[:4]]
        print(f"    Node {nid:>3}: succ={node.successor:>3}, "
              f"pred={node.predecessor}, fingers={fingers}...")


def demo_key_lookup() -> None:
    """Demonstrate O(log N) key lookups."""
    print("\n" + "=" * 70)
    print("Key Lookup: O(log N) Routing")
    print("=" * 70)

    net = ChordNetwork()
    for nid in [0, 32, 64, 96, 128, 160, 192, 224]:
        net.add_node(nid)

    # Look up various keys
    test_keys = ["alice", "bob", "carol", "dave", "eve",
                 "frank", "grace", "heidi"]

    print(f"\n  8 nodes on ring, looking up keys from Node 0:\n")
    print(f"  {'Key':<10} {'Hash':>6} {'Responsible':>12} {'Hops':>6}")
    print("  " + "-" * 40)

    total_hops = 0
    for key in test_keys:
        key_hash = chord_hash(key)
        resp_id, hops = net.lookup(key, from_node=0)
        total_hops += hops
        print(f"  {key:<10} {key_hash:>6} {resp_id:>12} {hops:>6}")

    avg_hops = total_hops / len(test_keys)
    import math
    expected = math.log2(len(net.nodes))
    print(f"\n  Average hops: {avg_hops:.1f} (expected O(log N) = {expected:.1f})")


def demo_data_storage() -> None:
    """Store and retrieve data in the DHT."""
    print("\n" + "=" * 70)
    print("DHT Data Storage")
    print("=" * 70)

    net = ChordNetwork()
    for nid in [0, 64, 128, 192]:
        net.add_node(nid)

    # Store data
    entries = [
        ("user:alice", "age=30"),
        ("user:bob", "age=25"),
        ("session:123", "token=abc"),
        ("cache:homepage", "html=..."),
    ]

    print(f"\n  Storing {len(entries)} entries:\n")
    for key, value in entries:
        stored_at, hops = net.put(key, value, from_node=0)
        print(f"    PUT {key} => Node {stored_at} ({hops} hops)")

    # Retrieve data
    print(f"\n  Retrieving:")
    for key, _ in entries:
        value, node_id, hops = net.get(key, from_node=128)
        print(f"    GET {key} from Node 128 => '{value}' at Node {node_id} "
              f"({hops} hops)")

    # Show data distribution
    print(f"\n  Data distribution across nodes:")
    for nid in sorted(net.nodes.keys()):
        node = net.get_node(nid)
        print(f"    Node {nid:>3}: {len(node.data)} keys — "
              f"{list(node.data.keys())}")


def demo_node_join() -> None:
    """Demonstrate adding a node to the ring."""
    print("\n" + "=" * 70)
    print("Node Join: Adding a New Node")
    print("=" * 70)

    net = ChordNetwork()
    for nid in [0, 64, 128, 192]:
        net.add_node(nid)

    # Store some data
    for i in range(20):
        net.put(f"key_{i}", f"val_{i}", from_node=0)

    # Show distribution before
    print(f"\n  Before joining Node 96:")
    for nid in sorted(net.nodes.keys()):
        node = net.get_node(nid)
        print(f"    Node {nid:>3}: {len(node.data)} keys")

    # Add new node
    net.add_node(96)

    # In a real system, keys that now belong to Node 96 would be transferred
    # from its successor. We simulate this:
    succ = net.get_node(128)
    transferred = 0
    if succ:
        keys_to_move = []
        for key in list(succ.data.keys()):
            key_hash = chord_hash(key)
            resp_id, _ = net.lookup(key, from_node=96)
            if resp_id == 96:
                keys_to_move.append(key)

        for key in keys_to_move:
            net.nodes[96].data[key] = succ.data.pop(key)
            transferred += 1

    print(f"\n  After joining Node 96 (transferred {transferred} keys):")
    for nid in sorted(net.nodes.keys()):
        node = net.get_node(nid)
        print(f"    Node {nid:>3}: {len(node.data)} keys")


def demo_comparison() -> None:
    """Compare DHT protocols."""
    print("\n" + "=" * 70)
    print("DHT Protocol Comparison")
    print("=" * 70)

    print("""
  ┌─────────────┬──────────┬───────────┬──────────────┬───────────────┐
  │ Protocol    │ Lookup   │ State/node│ Key feature  │ Used by       │
  ├─────────────┼──────────┼───────────┼──────────────┼───────────────┤
  │ Chord       │ O(log N) │ O(log N)  │ Finger table │ (Academic)    │
  │ Kademlia    │ O(log N) │ O(log N)  │ XOR metric   │ BitTorrent    │
  │ Pastry      │ O(log N) │ O(log N)  │ Prefix route │ PAST, Scribe  │
  │ CAN         │ O(N^1/d) │ O(d)      │ d-dim space  │ (Academic)    │
  └─────────────┴──────────┴───────────┴──────────────┴───────────────┘

  Kademlia advantages over Chord:
  - XOR distance is symmetric: A's distance to B == B's distance to A
  - Parallel lookups: query multiple nodes simultaneously
  - Iterative refinement: closer to target with each hop
  - Proven in production (BitTorrent DHT, Ethereum)
""")


if __name__ == "__main__":
    demo_chord_ring()
    demo_key_lookup()
    demo_data_storage()
    demo_node_join()
    demo_comparison()
    print("Done.")
