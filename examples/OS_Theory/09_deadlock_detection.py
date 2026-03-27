"""
Deadlock Detection and Avoidance

Demonstrates:
- Wait-for graph cycle detection (deadlock detection)
- Banker's algorithm (deadlock avoidance)
- Resource allocation graph visualization

Theory:
- Deadlock conditions: mutual exclusion, hold-and-wait,
  no preemption, circular wait (all four must hold)
- Detection: build wait-for graph, find cycles
- Avoidance: Banker's algorithm checks if granting a request
  leaves the system in a safe state (a safe sequence exists)

Adapted from OS Theory Lesson 09.
"""

from collections import defaultdict


# ── Wait-For Graph and Cycle Detection ──────────────────────────────────

class WaitForGraph:
    """Wait-for graph for deadlock detection.

    Nodes are processes. Edge P_i → P_j means P_i is waiting
    for a resource held by P_j. A cycle indicates deadlock.
    """

    def __init__(self):
        self.edges: dict[str, list[str]] = defaultdict(list)

    def add_edge(self, waiter: str, holder: str) -> None:
        self.edges[waiter].append(holder)

    def detect_cycle(self) -> list[str] | None:
        """DFS-based cycle detection. Returns cycle path or None."""
        visited: set[str] = set()
        # rec_stack tracks nodes on the CURRENT DFS path — a node in both
        # visited and rec_stack means we've found a back edge (cycle). A node
        # only in visited but not rec_stack was fully explored in a previous
        # DFS subtree and is safe to skip.
        rec_stack: set[str] = set()
        parent: dict[str, str] = {}

        def dfs(node: str) -> list[str] | None:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.edges.get(node, []):
                if neighbor not in visited:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result is not None:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle — reconstruct it
                    cycle = [neighbor]
                    current = node
                    while current != neighbor:
                        cycle.append(current)
                        current = parent.get(current, neighbor)
                    cycle.append(neighbor)
                    cycle.reverse()
                    return cycle

            rec_stack.discard(node)
            return None

        all_nodes = set(self.edges.keys())
        for targets in self.edges.values():
            all_nodes.update(targets)

        for node in all_nodes:
            if node not in visited:
                cycle = dfs(node)
                if cycle is not None:
                    return cycle
        return None

    def display(self) -> None:
        print("  Wait-For Graph:")
        for waiter, holders in sorted(self.edges.items()):
            for holder in holders:
                print(f"    {waiter} → {holder}")


def demo_deadlock_detection():
    """Demonstrate deadlock detection via wait-for graph."""
    print("=" * 60)
    print("DEADLOCK DETECTION (Wait-For Graph)")
    print("=" * 60)

    # Scenario 1: No deadlock
    print("\n  --- Scenario 1: No Deadlock ---\n")
    g1 = WaitForGraph()
    g1.add_edge("P1", "P2")
    g1.add_edge("P2", "P3")
    g1.add_edge("P4", "P3")
    g1.display()
    cycle = g1.detect_cycle()
    print(f"  Cycle: {cycle}")
    print(f"  Deadlock: {'Yes' if cycle else 'No'}")

    # Scenario 2: Deadlock (P1→P2→P3→P1)
    print("\n  --- Scenario 2: Deadlock ---\n")
    g2 = WaitForGraph()
    g2.add_edge("P1", "P2")
    g2.add_edge("P2", "P3")
    g2.add_edge("P3", "P1")
    g2.add_edge("P4", "P2")
    g2.display()
    cycle = g2.detect_cycle()
    print(f"  Cycle: {' → '.join(cycle) if cycle else 'None'}")
    print(f"  Deadlock: {'Yes' if cycle else 'No'}")

    # Scenario 3: Multiple potential cycles
    print("\n  --- Scenario 3: Complex Wait-For Graph ---\n")
    g3 = WaitForGraph()
    g3.add_edge("P1", "P2")
    g3.add_edge("P2", "P3")
    g3.add_edge("P3", "P4")
    g3.add_edge("P4", "P2")  # cycle: P2→P3→P4→P2
    g3.add_edge("P5", "P1")
    g3.display()
    cycle = g3.detect_cycle()
    print(f"  Cycle: {' → '.join(cycle) if cycle else 'None'}")
    print(f"  Deadlock: {'Yes' if cycle else 'No'}")


# ── Banker's Algorithm ──────────────────────────────────────────────────

class BankersAlgorithm:
    """Banker's algorithm for deadlock avoidance.

    Maintains:
    - available[j]: instances of resource j available
    - max_demand[i][j]: max demand of process i for resource j
    - allocation[i][j]: currently allocated to process i
    - need[i][j] = max_demand[i][j] - allocation[i][j]
    """

    def __init__(
        self,
        available: list[int],
        max_demand: list[list[int]],
        allocation: list[list[int]],
    ):
        self.n_proc = len(max_demand)
        self.n_res = len(available)
        self.available = available[:]
        self.max_demand = [row[:] for row in max_demand]
        self.allocation = [row[:] for row in allocation]
        self.need = [
            [max_demand[i][j] - allocation[i][j] for j in range(self.n_res)]
            for i in range(self.n_proc)
        ]

    def is_safe(self) -> tuple[bool, list[int]]:
        """Check if current state is safe. Returns (safe, sequence)."""
        # 'work' simulates available resources as processes complete — we
        # copy to avoid mutating actual state during the hypothetical check
        work = self.available[:]
        finish = [False] * self.n_proc
        sequence: list[int] = []

        while True:
            found = False
            for i in range(self.n_proc):
                # A process can finish if its remaining need fits within
                # currently available resources — this is the key invariant
                # of the safety algorithm
                if not finish[i] and all(
                    self.need[i][j] <= work[j] for j in range(self.n_res)
                ):
                    # Process i can finish — release its held resources back
                    # to the work vector, potentially enabling other processes
                    for j in range(self.n_res):
                        work[j] += self.allocation[i][j]
                    finish[i] = True
                    sequence.append(i)
                    found = True

            if not found:
                break

        safe = all(finish)
        return safe, sequence

    def request(self, pid: int, req: list[int]) -> bool:
        """Try to grant resource request. Returns True if granted."""
        # Check request ≤ need
        for j in range(self.n_res):
            if req[j] > self.need[pid][j]:
                print(f"    Error: P{pid} exceeds max claim")
                return False

        # Check request ≤ available
        for j in range(self.n_res):
            if req[j] > self.available[j]:
                print(f"    P{pid} must wait (insufficient resources)")
                return False

        # Pretend to allocate
        for j in range(self.n_res):
            self.available[j] -= req[j]
            self.allocation[pid][j] += req[j]
            self.need[pid][j] -= req[j]

        # Tentatively allocate, then check safety — if unsafe, we must roll
        # back to the pre-request state (optimistic check with undo)
        safe, seq = self.is_safe()
        if safe:
            print(f"    Request granted. Safe sequence: "
                  f"{' → '.join(f'P{s}' for s in seq)}")
            return True
        else:
            # Roll back — the tentative allocation would leave no safe
            # sequence, meaning deadlock could occur; deny the request
            for j in range(self.n_res):
                self.available[j] += req[j]
                self.allocation[pid][j] -= req[j]
                self.need[pid][j] += req[j]
            print(f"    Request denied (would lead to unsafe state)")
            return False

    def display(self) -> None:
        res_names = [f"R{j}" for j in range(self.n_res)]
        header = "      " + "  ".join(f"{r:>3}" for r in res_names)

        print(f"  Available: {self.available}")
        print(f"\n  {'Allocation':>14}  {'Max':>14}  {'Need':>14}")
        print(f"  {header}  {header}  {header}")
        for i in range(self.n_proc):
            alloc = "  ".join(f"{v:>3}" for v in self.allocation[i])
            mx = "  ".join(f"{v:>3}" for v in self.max_demand[i])
            nd = "  ".join(f"{v:>3}" for v in self.need[i])
            print(f"  P{i}:  {alloc}    {mx}    {nd}")


def demo_bankers():
    """Demonstrate Banker's algorithm."""
    print("\n" + "=" * 60)
    print("BANKER'S ALGORITHM (Deadlock Avoidance)")
    print("=" * 60)

    # Classic example: 5 processes, 3 resource types (A, B, C)
    available = [3, 3, 2]
    max_demand = [
        [7, 5, 3],  # P0
        [3, 2, 2],  # P1
        [9, 0, 2],  # P2
        [2, 2, 2],  # P3
        [4, 3, 3],  # P4
    ]
    allocation = [
        [0, 1, 0],  # P0
        [2, 0, 0],  # P1
        [3, 0, 2],  # P2
        [2, 1, 1],  # P3
        [0, 0, 2],  # P4
    ]

    banker = BankersAlgorithm(available, max_demand, allocation)

    print("\n  Initial state:")
    banker.display()

    safe, seq = banker.is_safe()
    print(f"\n  Safe state: {safe}")
    if safe:
        print(f"  Safe sequence: {' → '.join(f'P{s}' for s in seq)}")

    # Test requests
    print("\n  --- Request: P1 asks for [1, 0, 2] ---")
    banker.request(1, [1, 0, 2])

    print("\n  --- Request: P4 asks for [3, 3, 0] ---")
    banker.request(4, [3, 3, 0])

    print("\n  --- Request: P0 asks for [0, 2, 0] ---")
    banker.request(0, [0, 2, 0])


# ── Resource Allocation Graph ───────────────────────────────────────────

def demo_resource_allocation_graph():
    """Show resource allocation graph concepts."""
    print("\n" + "=" * 60)
    print("RESOURCE ALLOCATION GRAPH")
    print("=" * 60)

    print("""
  Resource Allocation Graph (text representation):

  Processes: P1, P2, P3
  Resources: R1 (1 instance), R2 (2 instances), R3 (1 instance)

  Assignment edges (resource → process):
    R1 → P2     (R1 is held by P2)
    R2 → P1     (one R2 instance held by P1)
    R2 → P2     (one R2 instance held by P2)

  Request edges (process → resource):
    P1 → R1     (P1 is waiting for R1)
    P2 → R3     (P2 is waiting for R3)
    P3 → R3     (P3 is waiting for R3)

  Analysis:
    - P1 holds R2, waits for R1 (held by P2)
    - P2 holds R1, R2, waits for R3
    - P3 waits for R3
    - No cycle → no deadlock

  If we add P3 → R2 (P3 waits for R2, all held):
    P3 → R2 → P1 → R1 → P2 → R3 → ... (no cycle back to P3)
    Still no deadlock (for multi-instance, need Banker's)
""")


if __name__ == "__main__":
    demo_deadlock_detection()
    demo_bankers()
    demo_resource_allocation_graph()
