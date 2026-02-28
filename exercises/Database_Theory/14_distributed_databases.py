"""
Exercises for Lesson 14: Distributed Databases
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers quorum calculations, 2PC protocol, Raft consensus, distributed deadlock,
consistent hashing, and distributed query optimization.
"""

import hashlib
import math


# === Exercise 3: Quorum Calculations ===
# Problem: Calculate quorum parameters for different configurations.

def exercise_3():
    """Quorum calculations for distributed systems."""
    print("Quorum System: N replicas, W write quorum, R read quorum")
    print("Strong consistency requires: W + R > N")
    print()

    # Part 1: N = 5
    N = 5
    print(f"Given: N = {N}")
    print()

    # 1. What W and R guarantee strong consistency?
    print("1. Values of W and R for strong consistency (W + R > N):")
    valid_combos = []
    for W in range(1, N + 1):
        for R in range(1, N + 1):
            if W + R > N:
                valid_combos.append((W, R))
    print(f"   Example combinations: W=3,R=3 | W=4,R=2 | W=5,R=1 | W=2,R=4 | W=1,R=5")
    print(f"   General rule: W + R > {N} (i.e., W + R >= {N + 1})")
    print()

    # 2. W=3, R=3: fault tolerance
    W, R = 3, 3
    write_tolerance = N - W
    read_tolerance = N - R
    print(f"2. W={W}, R={R}:")
    print(f"   Write tolerance: N - W = {N} - {W} = {write_tolerance} node failures")
    print(f"   Read tolerance:  N - R = {N} - {R} = {read_tolerance} node failures")
    print(f"   Consistent? W+R={W+R} > N={N}: {'YES' if W+R > N else 'NO'}")
    print()

    # 3. W=1 (fast writes), what R?
    W = 1
    R_needed = N - W + 1
    print(f"3. W={W} (fast writes): R must be >= {R_needed} for consistency")
    print(f"   W+R = {W}+{R_needed} = {W+R_needed} > {N}: YES")
    print(f"   Read latency: must contact {R_needed} of {N} nodes (high latency)")
    print(f"   Write latency: only 1 node needed (very fast, but risky)")
    print()

    # 4. N=7, W=4, R=4
    N2, W2, R2 = 7, 4, 4
    print(f"4. N={N2}, W={W2}, R={R2}:")
    print(f"   Consistent? W+R={W2+R2} > N={N2}: {'YES' if W2+R2 > N2 else 'NO'}")
    print(f"   Write tolerance: {N2-W2} failures")
    print(f"   Read tolerance: {N2-R2} failures")
    print(f"   Overall tolerance: min({N2-W2}, {N2-R2}) = {min(N2-W2, N2-R2)} failures")


# === Exercise 4: 2PC Protocol Trace ===
# Problem: Trace 2PC for a distributed transfer.

def exercise_4():
    """Trace 2PC protocol for distributed transfer."""
    print("Transaction T: Transfer $500 from Account A (Node 1) to Account B (Node 2)")
    print()

    # Scenario 1: Both vote YES
    print("1. Both nodes vote YES (normal commit):")
    messages = [
        ("Coordinator", "Node 1, Node 2", "PREPARE", "Phase 1: Coordinator sends PREPARE"),
        ("Node 1", "Coordinator", "VOTE YES", "Node 1: Lock A, check balance >= 500, vote YES"),
        ("Node 2", "Coordinator", "VOTE YES", "Node 2: Lock B, prepare to credit, vote YES"),
        ("Coordinator", "Node 1, Node 2", "COMMIT", "Phase 2: All YES -> Coordinator sends COMMIT"),
        ("Node 1", "Coordinator", "ACK", "Node 1: Debit A by $500, release lock, ACK"),
        ("Node 2", "Coordinator", "ACK", "Node 2: Credit B by $500, release lock, ACK"),
    ]
    for sender, receiver, msg, desc in messages:
        print(f"  {sender:>12} -> {receiver:<20} [{msg}]  {desc}")
    print()

    # Scenario 2: Node 2 votes NO
    print("2. Node 2 votes NO (abort):")
    messages2 = [
        ("Coordinator", "Node 1, Node 2", "PREPARE", "Phase 1: Coordinator sends PREPARE"),
        ("Node 1", "Coordinator", "VOTE YES", "Node 1: Lock A, check balance, vote YES"),
        ("Node 2", "Coordinator", "VOTE NO", "Node 2: Cannot lock B (or other issue), vote NO"),
        ("Coordinator", "Node 1, Node 2", "ABORT", "Phase 2: Any NO -> Coordinator sends ABORT"),
        ("Node 1", "Coordinator", "ACK", "Node 1: Rollback debit, release lock, ACK"),
        ("Node 2", "Coordinator", "ACK", "Node 2: No changes needed, release lock, ACK"),
    ]
    for sender, receiver, msg, desc in messages2:
        print(f"  {sender:>12} -> {receiver:<20} [{msg}]  {desc}")
    print()

    # Scenario 3: Coordinator crashes after receiving YES votes
    print("3. Coordinator crashes AFTER receiving all YES votes, BEFORE sending COMMIT:")
    print()
    print("  Timeline:")
    print("    1. Coordinator receives VOTE YES from Node 1 and Node 2")
    print("    2. Coordinator DECIDES to commit (writes COMMIT to own log)")
    print("    3. Coordinator CRASHES before sending COMMIT messages")
    print()
    print("  Participant state:")
    print("    Node 1: IN DOUBT (voted YES, waiting for decision)")
    print("    Node 2: IN DOUBT (voted YES, waiting for decision)")
    print()
    print("  Problems:")
    print("    - Both nodes are BLOCKED: they hold locks on A and B")
    print("    - They cannot decide on their own (might be commit or abort)")
    print("    - They must WAIT for coordinator to recover")
    print("    - Wait time: potentially minutes to hours (until coordinator restarts)")
    print()
    print("  Recovery:")
    print("    - Coordinator restarts, reads its log, finds COMMIT decision")
    print("    - Sends COMMIT to both nodes")
    print("    - Nodes commit and release locks")
    print()
    print("  This is the fundamental weakness of 2PC: coordinator is a single point of failure.")
    print("  3PC or Paxos-based commit protocols address this at the cost of complexity.")


# === Exercise 5: Raft Consensus ===
# Problem: Trace Raft leader election and log replication.

def exercise_5():
    """Raft consensus scenario analysis."""
    print("5-node Raft cluster: A (leader, term 3), B, C, D, E")
    print()

    # 1. Leader election after A crashes
    print("1. Leader election after A crashes:")
    print("  - A crashes. B, C, D, E detect no heartbeat after election timeout.")
    print("  - One node (say B) times out first, increments term to 4, requests votes.")
    print("  - B sends RequestVote to C, D, E.")
    print("  - Voting rule: nodes vote for the first candidate in a new term")
    print("    IF the candidate's log is at least as up-to-date as the voter's log.")
    print("  - B's log includes [term 3, SET x=5] (replicated to B before crash).")
    print("  - C also has this entry. D and E do not.")
    print("  - C, D, E vote for B (B's log is at least as up-to-date as D and E's).")
    print("  - B gets 3 votes (B, C, D or E) = majority. B becomes leader in term 4.")
    print()

    # 2. Is [term 3, SET x=5] committed?
    print("2. Is entry [term 3, SET x=5] committed?")
    print("  - Entry was replicated to A, B, C (3 out of 5 nodes).")
    print("  - Majority = 3. The entry WAS committed (A counted it as committed before crash).")
    print("  - However, after A crashes, only B and C have it (2 out of 4 surviving).")
    print("  - Since the entry WAS committed before the crash, Raft guarantees it will")
    print("    appear in any future leader's log (Leader Completeness Property).")
    print()

    # 3. Will B include the entry?
    print("3. Will B (new leader, term 4) include [term 3, SET x=5]?")
    print("  - YES. B has this entry in its log.")
    print("  - B was elected because its log was sufficiently up-to-date.")
    print("  - B will replicate this entry to D and E during normal operation.")
    print("  - After replication: B, C, D, E all have the entry (4 out of 4).")
    print("  - The committed entry is NEVER lost -- this is Raft's safety guarantee.")


# === Exercise 6: Distributed Deadlock ===
# Problem: Detect and resolve distributed deadlock.

def exercise_6():
    """Distributed deadlock detection."""
    print("Distributed Deadlock Scenario:")
    print()
    print("  Node 1:")
    print("    T1 holds lock on row A, waiting for lock on row B (held by T2)")
    print("    T2 holds lock on row B")
    print()
    print("  Node 2:")
    print("    T2 waiting for lock on row C (held by T3)")
    print("    T3 holds lock on row C, waiting for lock on row A (held by T1 on Node 1)")
    print()

    # 1. Global wait-for graph
    print("1. Global Wait-For Graph:")
    print("   T1 -> T2  (T1 waits for B, held by T2)")
    print("   T2 -> T3  (T2 waits for C, held by T3)")
    print("   T3 -> T1  (T3 waits for A, held by T1)")
    print()
    print("   Graph: T1 -> T2 -> T3 -> T1")
    print("          ^                   |")
    print("          +-------------------+")
    print()

    # 2. Deadlock detection
    print("2. Deadlock detected: YES")
    print("   Cycle: T1 -> T2 -> T3 -> T1")
    print("   Victim selection: abort the youngest transaction (highest timestamp).")
    print("   If TS(T1) < TS(T2) < TS(T3): abort T3 (youngest).")
    print("   T3 releases lock on C -> T2 can proceed -> T2 releases B -> T1 proceeds.")
    print()

    # 3. Centralized deadlock detector
    print("3. Centralized deadlock detector:")
    print("   - A designated coordinator collects local wait-for graphs from each node.")
    print("   - Node 1 sends: T1 -> T2")
    print("   - Node 2 sends: T2 -> T3, T3 -> T1")
    print("   - Coordinator merges into global graph, detects cycle.")
    print("   - Problem: communication delay may cause false positives (phantom deadlocks)")
    print("     if a transaction completes between graph collection and analysis.")
    print()

    # 4. Timeout-based approach
    print("4. Timeout-based approach:")
    print("   - If a transaction waits longer than threshold (e.g., 5 seconds), abort it.")
    print("   - Simple to implement, no global coordinator needed.")
    print("   - Risks:")
    print("     - False positives: long-running but valid transactions get aborted")
    print("     - Wasted work: may abort a transaction that would have completed")
    print("     - Choosing timeout: too short = false positives, too long = wasted time")


# === Exercise 8: Consistent Hashing ===
# Problem: Key assignment in consistent hashing ring.

def exercise_8():
    """Consistent hashing ring operations."""
    print("Consistent Hashing Ring [0, 360)")
    print()

    # Nodes at positions
    nodes = {"A": 45, "B": 120, "C": 200, "D": 310}
    keys = {"k1": 30, "k2": 90, "k3": 150, "k4": 210, "k5": 330, "k6": 10}

    print("Nodes:", {n: f"{p}" for n, p in sorted(nodes.items(), key=lambda x: x[1])})
    print("Keys:", {k: f"{v}" for k, v in sorted(keys.items())})
    print()

    def find_responsible_node(key_pos, node_positions):
        """Find the first node clockwise from key position."""
        sorted_nodes = sorted(node_positions.items(), key=lambda x: x[1])
        for name, pos in sorted_nodes:
            if pos >= key_pos:
                return name
        return sorted_nodes[0][0]  # Wrap around

    # 1. Key assignments
    print("1. Key assignments (each key -> first node clockwise):")
    for key in sorted(keys.keys()):
        pos = keys[key]
        node = find_responsible_node(pos, nodes)
        print(f"   {key} (pos={pos}) -> Node {node} (pos={nodes[node]})")
    print()

    # 2. Add Node E at 170
    print("2. Add Node E at position 170:")
    nodes_with_e = {**nodes, "E": 170}
    reassigned = []
    for key in sorted(keys.keys()):
        pos = keys[key]
        old_node = find_responsible_node(pos, nodes)
        new_node = find_responsible_node(pos, nodes_with_e)
        if old_node != new_node:
            reassigned.append((key, pos, old_node, new_node))
            print(f"   {key} (pos={pos}): Node {old_node} -> Node {new_node} (REASSIGNED)")
        else:
            print(f"   {key} (pos={pos}): Node {old_node} (unchanged)")
    print(f"   Only {len(reassigned)} key(s) reassigned (minimal disruption).")
    print()

    # 3. Node B fails
    print("3. Node B fails (removed from ring):")
    nodes_without_b = {n: p for n, p in nodes.items() if n != "B"}
    reassigned_b = []
    for key in sorted(keys.keys()):
        pos = keys[key]
        old_node = find_responsible_node(pos, nodes)
        new_node = find_responsible_node(pos, nodes_without_b)
        if old_node != new_node:
            reassigned_b.append((key, pos, old_node, new_node))
            print(f"   {key} (pos={pos}): Node {old_node} -> Node {new_node} (REASSIGNED)")
        else:
            print(f"   {key} (pos={pos}): Node {old_node} (unchanged)")
    print(f"   {len(reassigned_b)} key(s) reassigned to the next node clockwise.")
    print()

    # 4. Virtual nodes
    print("4. With virtual nodes (3 per physical node):")
    virtual_nodes = {
        "A": [45, 165, 285], "B": [120, 240, 350],
        "C": [200, 320, 80], "D": [310, 70, 190]
    }

    # Flatten to (position, physical_node)
    vnode_map = {}
    for physical, positions in virtual_nodes.items():
        for pos in positions:
            vnode_map[pos] = physical

    print("  Virtual node positions:")
    for pos in sorted(vnode_map.keys()):
        print(f"    Position {pos:>3} -> Physical Node {vnode_map[pos]}")
    print()

    print("  Key assignments with virtual nodes:")
    for key in sorted(keys.keys()):
        pos = keys[key]
        # Find first vnode clockwise
        sorted_positions = sorted(vnode_map.keys())
        responsible = None
        for vpos in sorted_positions:
            if vpos >= pos:
                responsible = vnode_map[vpos]
                rpos = vpos
                break
        if responsible is None:
            rpos = sorted_positions[0]
            responsible = vnode_map[rpos]
        print(f"    {key} (pos={pos}) -> vnode at {rpos} -> Physical Node {responsible}")


# === Exercise 10: Distributed Query Optimization ===
# Problem: Compare distributed join strategies.

def exercise_10():
    """Compare distributed join strategies."""
    # Data
    n_orders = 10_000_000
    order_row_size = 500  # bytes
    n_customers = 100_000
    customer_row_size = 200
    tokyo_pct = 0.05
    n_tokyo = int(n_customers * tokyo_pct)
    orders_per_customer = n_orders / n_customers  # 100

    print("Distributed Query Optimization")
    print()
    print(f"  Node 1: orders ({n_orders:,} rows, {order_row_size} bytes/row)")
    print(f"  Node 2: customers ({n_customers:,} rows, {customer_row_size} bytes/row)")
    print(f"  Query: SELECT c.name, o.total FROM orders o JOIN customers c")
    print(f"         ON o.customer_id = c.customer_id WHERE c.city = 'Tokyo'")
    print(f"  Tokyo customers: {tokyo_pct:.0%} = {n_tokyo:,}")
    print(f"  Orders per customer: ~{orders_per_customer:.0f}")
    print()

    strategies = [
        {
            "name": "1. Ship-whole: Send customers to Node 1",
            "transfer_bytes": n_customers * customer_row_size,
            "description": "Send entire customers table to Node 1, join locally."
        },
        {
            "name": "2. Ship-whole (reverse): Send orders to Node 2",
            "transfer_bytes": n_orders * order_row_size,
            "description": "Send entire orders table to Node 2, join locally."
        },
        {
            "name": "3. Semi-join: Send Tokyo customer_ids to Node 1",
            "transfer_bytes": (
                n_tokyo * 8 +  # customer_ids to Node 1 (8 bytes each)
                int(n_tokyo * orders_per_customer) * (8 + 500)  # matching orders back
            ),
            "description": "Step 1: Send Tokyo customer_ids to Node 1 (~40 KB).\n"
                          "         Step 2: Node 1 returns matching orders (~500K rows x 500 bytes).\n"
                          "         Step 3: Join at Node 2."
        },
        {
            "name": "4. Bloom filter: Send Bloom filter of Tokyo customer_ids",
            "transfer_bytes": (
                n_tokyo * 2 +  # Bloom filter (~10 KB, 2 bytes per element with 1% FPR)
                int(n_tokyo * orders_per_customer * 1.01) * 500  # matching orders (1% false positives)
            ),
            "description": "Step 1: Create Bloom filter of Tokyo customer_ids (~10 KB).\n"
                          "         Step 2: Node 1 filters orders using Bloom filter.\n"
                          "         Step 3: Send matching orders to Node 2 (includes ~1% false positives)."
        }
    ]

    print(f"  {'Strategy':<55} {'Transfer (MB)':>12}")
    print(f"  {'-'*55} {'-'*12}")
    for s in strategies:
        mb = s["transfer_bytes"] / (1024 * 1024)
        print(f"  {s['name']:<55} {mb:>10.1f} MB")
    print()

    for s in strategies:
        mb = s["transfer_bytes"] / (1024 * 1024)
        latency = mb  # 1 ms per MB
        print(f"  {s['name']}")
        print(f"    Transfer: {mb:.1f} MB ({latency:.1f} ms at 1 ms/MB)")
        print(f"    Details: {s['description']}")
        print()

    print("  WINNER: Strategy 3 (Semi-join) or 4 (Bloom filter)")
    print("    Both send only ~40 KB to identify Tokyo customers,")
    print("    then transfer only the matching orders (~250 MB).")
    print("    Bloom filter is slightly more efficient (smaller filter) but")
    print("    includes ~1% false positives that are filtered at Node 2.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 3: Quorum Calculations ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 4: 2PC Protocol Trace ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: Raft Consensus ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: Distributed Deadlock ===")
    print("=" * 70)
    exercise_6()

    print("=" * 70)
    print("=== Exercise 8: Consistent Hashing ===")
    print("=" * 70)
    exercise_8()

    print("=" * 70)
    print("=== Exercise 10: Distributed Query Optimization ===")
    print("=" * 70)
    exercise_10()

    print("\nAll exercises completed!")
