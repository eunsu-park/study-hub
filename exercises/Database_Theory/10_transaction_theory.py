"""
Exercises for Lesson 10: Transaction Theory
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers ACID properties, serializability testing via precedence graphs,
schedule classification (recoverable, cascadeless, strict), and isolation levels.
"""

from collections import defaultdict


# ============================================================
# Precedence Graph Implementation
# ============================================================

class PrecedenceGraph:
    """Directed graph for conflict serializability testing."""

    def __init__(self):
        self.edges = defaultdict(set)
        self.nodes = set()

    def add_edge(self, src, dst, reason=""):
        """Add a directed edge from src to dst."""
        self.nodes.add(src)
        self.nodes.add(dst)
        self.edges[src].add((dst, reason))

    def has_cycle(self):
        """Detect cycle using DFS. Returns (has_cycle, cycle_path)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.nodes}
        parent = {n: None for n in self.nodes}

        def dfs(node, path):
            color[node] = GRAY
            path.append(node)
            for neighbor, _ in self.edges.get(node, set()):
                if color[neighbor] == GRAY:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
                elif color[neighbor] == WHITE:
                    result = dfs(neighbor, path)
                    if result:
                        return result
            path.pop()
            color[node] = BLACK
            return None

        for node in self.nodes:
            if color[node] == WHITE:
                cycle = dfs(node, [])
                if cycle:
                    return True, cycle

        return False, []

    def topological_sort(self):
        """Return topological ordering (serial schedule equivalent)."""
        if self.has_cycle()[0]:
            return None
        in_degree = defaultdict(int)
        for n in self.nodes:
            in_degree[n] = 0
        for src in self.edges:
            for dst, _ in self.edges[src]:
                in_degree[dst] += 1

        queue = [n for n in self.nodes if in_degree[n] == 0]
        result = []
        while queue:
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            for neighbor, _ in self.edges.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return result

    def display(self):
        """Print the graph."""
        print("  Nodes:", sorted(self.nodes))
        print("  Edges:")
        for src in sorted(self.edges.keys()):
            for dst, reason in sorted(self.edges[src]):
                print(f"    {src} -> {dst}  ({reason})")


def parse_schedule(schedule_str):
    """Parse a schedule string like 'r1(A) w2(A) c1 c2' into operations."""
    ops = []
    for token in schedule_str.split():
        if token.startswith(('r', 'w')):
            op_type = token[0]
            # Find transaction number and data item
            paren_idx = token.index('(')
            txn = f"T{token[1:paren_idx]}"
            item = token[paren_idx + 1:-1]
            ops.append((op_type, txn, item))
        elif token.startswith('c'):
            txn = f"T{token[1:]}"
            ops.append(('c', txn, None))
    return ops


def build_precedence_graph(ops):
    """Build precedence graph from schedule operations."""
    graph = PrecedenceGraph()

    # Extract all transactions
    txns = set()
    for op_type, txn, item in ops:
        txns.add(txn)
    for txn in txns:
        graph.nodes.add(txn)

    # Find conflicts
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            op1_type, txn1, item1 = ops[i]
            op2_type, txn2, item2 = ops[j]

            if txn1 == txn2 or item1 is None or item2 is None:
                continue
            if item1 != item2:
                continue

            # Conflict: at least one is a write, different transactions, same item
            is_conflict = (op1_type == 'w' or op2_type == 'w')
            if is_conflict:
                conflict_type = f"{op1_type}{op2_type}({item1})"
                graph.add_edge(txn1, txn2, conflict_type)

    return graph


# ============================================================
# Exercise Solutions
# ============================================================

# === Exercise 1: ACID Violations (Banking) ===
# Problem: For each ACID property, give a concrete example of what goes wrong.

def exercise_1():
    """ACID violation examples in banking."""
    violations = [
        {
            "property": "Atomicity",
            "violation": "Transfer $500 from A to B: A is debited by $500, but system crashes "
                        "before B is credited. A lost $500, B gained nothing.",
            "protection": "Transaction rollback on failure -- either both operations complete or neither does."
        },
        {
            "property": "Consistency",
            "violation": "Account balance constraint: balance >= 0. A transfer of $500 from A "
                        "(balance $300) would leave A with -$200, violating the constraint.",
            "protection": "DBMS checks constraints before commit. Transaction aborted if constraint violated."
        },
        {
            "property": "Isolation",
            "violation": "T1 reads A=$1000, T2 reads A=$1000. Both transfer $500 to B. "
                        "Both set A=$500. A should be $0 but lost update leaves it at $500.",
            "protection": "Concurrency control (locking/MVCC) ensures serializable execution."
        },
        {
            "property": "Durability",
            "violation": "Transaction commits (confirmed to user), but system crashes before "
                        "changes are written to disk. After restart, the transfer is lost.",
            "protection": "Write-Ahead Logging (WAL): log records flushed to stable storage before commit."
        }
    ]

    for v in violations:
        print(f"{v['property']}:")
        print(f"  Violation: {v['violation']}")
        print(f"  Protection: {v['protection']}")
        print()


# === Exercise 4: Conflict Serializability ===
# Problem: Test schedules for conflict serializability.

def exercise_4():
    """Test conflict serializability using precedence graphs."""
    schedules = [
        ("(a)", "r1(A) r2(B) w1(B) w2(A) c1 c2"),
        ("(b)", "r1(A) w2(A) w1(A) r2(A) c1 c2"),
        ("(c)", "r3(B) r1(A) w3(A) r2(B) w2(A) w1(B) c1 c2 c3"),
        ("(d)", "r1(A) r2(B) r3(C) w1(B) w2(C) w3(A) c1 c2 c3"),
    ]

    for label, schedule_str in schedules:
        print(f"Schedule {label}: {schedule_str}")
        ops = parse_schedule(schedule_str)
        graph = build_precedence_graph(ops)
        graph.display()

        has_cycle, cycle = graph.has_cycle()
        if has_cycle:
            print(f"  Cycle detected: {' -> '.join(cycle)}")
            print(f"  NOT conflict serializable.")
        else:
            topo = graph.topological_sort()
            print(f"  No cycle. Conflict serializable.")
            print(f"  Equivalent serial order: {' -> '.join(topo)}")
        print()


# === Exercise 5: Schedule Analysis ===
# Problem: Analyze a schedule for multiple properties.

def exercise_5():
    """Complete schedule analysis."""
    schedule_str = "r1(X) r2(X) w2(X) r1(Y) w1(Y) w2(Y) c1 c2"
    print(f"Schedule: {schedule_str}")
    print()

    ops = parse_schedule(schedule_str)
    graph = build_precedence_graph(ops)

    # (a) Precedence graph
    print("(a) Precedence graph:")
    graph.display()
    print()

    # (b) Conflict serializable?
    has_cycle, cycle = graph.has_cycle()
    print(f"(b) Conflict serializable?")
    if has_cycle:
        print(f"    NO -- cycle: {' -> '.join(cycle)}")
    else:
        topo = graph.topological_sort()
        print(f"    YES -- equivalent serial order: {' -> '.join(topo)}")
    print()

    # (c) Recoverable?
    # A schedule is recoverable if no T commits before all Ts it read from have committed.
    print("(c) Recoverable?")
    print("    T1 reads X (initial value), T2 writes X, T1 reads Y (initial value)")
    print("    T1 does NOT read any value written by T2.")
    print("    T2 writes Y after w2(Y), but T1's w1(Y) comes before w2(Y).")
    print("    T2 does NOT read any value written by T1.")
    print("    c1 before c2. No transaction commits before one it read from.")
    print("    YES, recoverable.")
    print()

    # (d) Cascadeless?
    print("(d) Cascadeless?")
    print("    A schedule is cascadeless if every read sees only committed writes.")
    print("    r1(X): reads initial X value (not from any uncommitted T). OK.")
    print("    r2(X): reads initial X value (T1 only read X, didn't write it yet). OK.")
    print("    r1(Y): reads initial Y value. OK.")
    print("    No transaction reads a dirty (uncommitted) write.")
    print("    YES, cascadeless.")
    print()

    # (e) Strict?
    print("(e) Strict?")
    print("    A schedule is strict if no T reads or writes a data item")
    print("    written by an uncommitted T.")
    print("    w2(X) then... T2's X is not read/written by T1 after w2(X). OK.")
    print("    w1(Y) then w2(Y): T2 writes Y that T1 also wrote, BEFORE T1 commits.")
    print("    w2(Y) overwrites T1's uncommitted write of Y.")
    print("    NO, not strict (T2 writes Y before T1 commits, and T1 wrote Y).")


# === Exercise 6: Constructing Specific Schedules ===
# Problem: Construct schedules with specific properties.

def exercise_6():
    """Construct schedules with specific properties."""

    # (a) View serializable but NOT conflict serializable
    print("(a) View serializable but NOT conflict serializable:")
    print("    Schedule: w1(A) w2(A) w3(A) c1 c2 c3")
    print("    Precedence graph: T1->T2->T3 (no cycle), but also consider blind writes.")
    print()
    print("    Actually, a classic example is:")
    print("    S: r1(A) w2(A) w1(A) w3(A) c1 c2 c3")
    print("    Precedence graph: T1->T2 (r1w2 on A), T2->T1 (w2w1 on A) -> CYCLE")
    print("    So NOT conflict serializable.")
    print("    But view equivalent to T1,T2,T3 serial: T3's blind write of A is the final write,")
    print("    and r1 reads the initial value (same as in T1 first).")
    print("    View serializable because initial reads and final writes match.")
    print()

    # (b) Recoverable but NOT cascadeless
    print("(b) Recoverable but NOT cascadeless:")
    print("    Schedule: w1(A) r2(A) c1 c2")
    print("    T2 reads A written by T1 BEFORE T1 commits (dirty read).")
    print("    But T1 commits before T2 commits -> recoverable.")
    print("    NOT cascadeless: T2 reads uncommitted data from T1.")
    print()

    # (c) Cascadeless but NOT strict
    print("(c) Cascadeless but NOT strict:")
    print("    Schedule: w1(A) w2(A) c1 c2")
    print("    No reads of dirty data (cascadeless: only writes, no dirty reads).")
    print("    But T2 writes A before T1 commits -> not strict.")
    print("    If T1 aborts, T2's write must also be undone (because T2 overwrote T1's value).")


# === Exercise 7: Isolation Levels ===
# Problem: Identify minimum isolation level for each scenario.

def exercise_7():
    """Determine minimum isolation level."""
    scenarios = [
        {
            "scenario": "(a) Inventory system: reading uncommitted quantity could lead to overselling",
            "anomaly": "Dirty read",
            "min_level": "READ COMMITTED",
            "explanation": "Prevents dirty reads. T will only see quantities from committed transactions."
        },
        {
            "scenario": "(b) Report summing account balances (must be consistent)",
            "anomaly": "Non-repeatable read (and/or phantom reads from transfers in progress)",
            "min_level": "REPEATABLE READ",
            "explanation": "Ensures each row read by the report stays stable throughout the transaction. "
                          "Balances won't change mid-report."
        },
        {
            "scenario": "(c) Employee count per department (count must not change within transaction)",
            "anomaly": "Phantom read (new employees inserted during the count)",
            "min_level": "SERIALIZABLE",
            "explanation": "Prevents phantoms: no new rows can appear in the department "
                          "between the two COUNT queries."
        }
    ]

    for s in scenarios:
        print(f"{s['scenario']}")
        print(f"  Anomaly to prevent: {s['anomaly']}")
        print(f"  Minimum isolation level: {s['min_level']}")
        print(f"  Explanation: {s['explanation']}")
        print()


# === Exercise 8: Write Skew under Snapshot Isolation ===
# Problem: Demonstrate write skew anomaly.

def exercise_8():
    """Demonstrate write skew under Snapshot Isolation."""
    print("Write Skew Anomaly under Snapshot Isolation")
    print()
    print("Scenario: Hospital on-call constraint")
    print("  Rule: At least one doctor must be on call at all times.")
    print("  Initial state: doctor A is on call, doctor B is on call.")
    print()

    # Simulate
    doctors = {"A": True, "B": True}
    print(f"  Initial: A on_call={doctors['A']}, B on_call={doctors['B']}")
    print(f"  Invariant: COUNT(on_call=TRUE) >= 1")
    print()

    print("  T1 (Snapshot at time 0):        T2 (Snapshot at time 0):")
    print("    reads: A=true, B=true           reads: A=true, B=true")
    print("    checks: 2 on-call, safe         checks: 2 on-call, safe")
    print("    sets A = false                  sets B = false")
    print("    commits                         commits")
    print()

    # Result
    doctors["A"] = False
    doctors["B"] = False
    on_call_count = sum(1 for v in doctors.values() if v)
    print(f"  Final state: A on_call={doctors['A']}, B on_call={doctors['B']}")
    print(f"  On-call count: {on_call_count}")
    print(f"  INVARIANT VIOLATED! No doctor on call.")
    print()

    print("  Why this happens under SI but not SERIALIZABLE:")
    print("    - Under SI, T1 and T2 see the same snapshot (both see A=true, B=true).")
    print("    - Their write sets don't overlap (T1 writes A, T2 writes B).")
    print("    - SI's first-committer-wins only detects write-write conflicts.")
    print("    - Under true Serializable: T1 runs first (A=false), then T2 checks")
    print("      and sees only B=true. T2 sets B=false -> invariant violated -> ABORT T2.")
    print()

    print("  SQL to demonstrate:")
    sql = """
    -- T1
    BEGIN ISOLATION LEVEL SERIALIZABLE;
    SELECT COUNT(*) FROM doctors WHERE on_call = TRUE;  -- returns 2
    UPDATE doctors SET on_call = FALSE WHERE name = 'A';
    COMMIT;

    -- T2 (concurrent)
    BEGIN ISOLATION LEVEL SERIALIZABLE;
    SELECT COUNT(*) FROM doctors WHERE on_call = TRUE;  -- returns 2
    UPDATE doctors SET on_call = FALSE WHERE name = 'B';
    COMMIT;  -- Under true SERIALIZABLE: this would be aborted (serialization failure)
    """
    print(sql)


# === Exercise 10: Serial Schedule Count ===
# Problem: Prove n! serial schedules and explain precedence graph efficiency.

def exercise_10():
    """Serial schedule count and precedence graph efficiency."""
    import math

    print("Number of serial schedules for n transactions:")
    for n in range(1, 8):
        count = math.factorial(n)
        print(f"  n={n}: {count:,} serial schedules")
    print()

    print("Why testing all n! serial schedules is impractical:")
    n = 20
    print(f"  For n=20 transactions: {math.factorial(20):,} serial schedules")
    print(f"  That's ~2.4 x 10^18 -- infeasible even at 10^9 tests/second.")
    print()

    print("Precedence graph alternative:")
    print("  1. Build precedence graph: O(n^2 * m) where m = operations per transaction")
    print("  2. Detect cycle: O(V + E) using DFS, where V=n, E=O(n^2)")
    print("  3. Total: O(n^2 * m) -- polynomial time!")
    print()

    print("  The precedence graph captures ALL conflict information in a single structure.")
    print("  A schedule is conflict serializable <=> the precedence graph is acyclic.")
    print("  If acyclic, any topological sort gives a valid serial order.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: ACID Violations ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 4: Conflict Serializability ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: Schedule Analysis ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: Constructing Specific Schedules ===")
    print("=" * 70)
    exercise_6()

    print("=" * 70)
    print("=== Exercise 7: Isolation Levels ===")
    print("=" * 70)
    exercise_7()

    print("=" * 70)
    print("=== Exercise 8: Write Skew under SI ===")
    print("=" * 70)
    exercise_8()

    print("=" * 70)
    print("=== Exercise 10: Serial Schedule Count ===")
    print("=" * 70)
    exercise_10()

    print("\nAll exercises completed!")
