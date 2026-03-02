"""
Strongly Connected Components (SCC)
Strongly Connected Components

Finds the maximal sets of vertices in a directed graph where every vertex is reachable from every other.
"""

from typing import List, Tuple, Set
from collections import defaultdict


# =============================================================================
# 1. Kosaraju's Algorithm
# =============================================================================

def kosaraju_scc(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Find SCCs using Kosaraju's Algorithm
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)

    1. Build finish order stack via forward DFS
    2. DFS on reverse graph in stack order
    """
    # Forward/reverse graphs
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)

    for u, v in edges:
        graph[u].append(v)
        reverse_graph[v].append(u)

    # Phase 1: Record finish order via forward DFS
    visited = [False] * n
    finish_stack = []

    def dfs1(node: int):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs1(neighbor)
        finish_stack.append(node)

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    # Phase 2: Find SCCs via reverse DFS
    visited = [False] * n
    sccs = []

    def dfs2(node: int, component: List[int]):
        visited[node] = True
        component.append(node)
        for neighbor in reverse_graph[node]:
            if not visited[neighbor]:
                dfs2(neighbor, component)

    while finish_stack:
        node = finish_stack.pop()
        if not visited[node]:
            component = []
            dfs2(node, component)
            sccs.append(component)

    return sccs


# =============================================================================
# 2. Tarjan's Algorithm
# =============================================================================

def tarjan_scc(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Find SCCs using Tarjan's Algorithm
    Time Complexity: O(V + E)
    Space Complexity: O(V)

    low[v] = minimum discovery time reachable from v
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    disc = [-1] * n  # Discovery time
    low = [-1] * n   # Low-link value
    on_stack = [False] * n
    stack = []
    sccs = []
    time = [0]  # Global time counter

    def dfs(node: int):
        disc[node] = low[node] = time[0]
        time[0] += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in graph[node]:
            if disc[neighbor] == -1:  # Unvisited
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
            elif on_stack[neighbor]:  # On stack (back edge)
                low[node] = min(low[node], disc[neighbor])

        # SCC root found
        if low[node] == disc[node]:
            component = []
            while True:
                v = stack.pop()
                on_stack[v] = False
                component.append(v)
                if v == node:
                    break
            sccs.append(component)

    for i in range(n):
        if disc[i] == -1:
            dfs(i)

    return sccs


# =============================================================================
# 3. SCC Condensation Graph (DAG)
# =============================================================================

def build_scc_dag(n: int, edges: List[Tuple[int, int]]) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    Find SCCs and build condensation graph (DAG)
    Returns: (sccs, scc_graph, node_to_scc)
    """
    sccs = tarjan_scc(n, edges)

    # Map each node to its SCC
    node_to_scc = [-1] * n
    for i, component in enumerate(sccs):
        for node in component:
            node_to_scc[node] = i

    # Edges between SCCs (DAG)
    scc_edges = set()
    for u, v in edges:
        scc_u = node_to_scc[u]
        scc_v = node_to_scc[v]
        if scc_u != scc_v:
            scc_edges.add((scc_u, scc_v))

    scc_graph = defaultdict(list)
    for u, v in scc_edges:
        scc_graph[u].append(v)

    return sccs, dict(scc_graph), node_to_scc


# =============================================================================
# 4. 2-SAT Problem
# =============================================================================

class TwoSAT:
    """
    2-SAT Problem Solver
    n boolean variables with 2-CNF clauses

    Variable x_i: node 2*i (true), node 2*i+1 (false)
    Clause (a or b): not_a -> b, not_b -> a
    """

    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

    def _var(self, x: int, negated: bool) -> int:
        """Convert variable to graph node"""
        return 2 * x + (1 if negated else 0)

    def _neg(self, node: int) -> int:
        """Negation of a node"""
        return node ^ 1

    def add_clause(self, x: int, neg_x: bool, y: int, neg_y: bool):
        """
        Add clause: (x or y)
        neg_x: whether x is negated
        neg_y: whether y is negated
        """
        # (x or y) is equivalent to (not_x -> y) and (not_y -> x)
        node_x = self._var(x, neg_x)
        node_y = self._var(y, neg_y)

        # not_x -> y
        self.graph[self._neg(node_x)].append(node_y)
        self.reverse_graph[node_y].append(self._neg(node_x))

        # not_y -> x
        self.graph[self._neg(node_y)].append(node_x)
        self.reverse_graph[node_x].append(self._neg(node_y))

    def solve(self) -> Tuple[bool, List[bool]]:
        """
        Solve 2-SAT
        Returns: (satisfiability, value of each variable)
        """
        total_nodes = 2 * self.n

        # Find SCCs using Kosaraju's
        visited = [False] * total_nodes
        finish_stack = []

        def dfs1(node: int):
            visited[node] = True
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    dfs1(neighbor)
            finish_stack.append(node)

        for i in range(total_nodes):
            if not visited[i]:
                dfs1(i)

        visited = [False] * total_nodes
        scc_id = [-1] * total_nodes
        current_scc = 0

        def dfs2(node: int):
            visited[node] = True
            scc_id[node] = current_scc
            for neighbor in self.reverse_graph[node]:
                if not visited[neighbor]:
                    dfs2(neighbor)

        while finish_stack:
            node = finish_stack.pop()
            if not visited[node]:
                dfs2(node)
                current_scc += 1

        # Check satisfiability
        for i in range(self.n):
            if scc_id[2 * i] == scc_id[2 * i + 1]:
                return False, []

        # Construct solution (later discovered SCC = smaller SCC ID = higher topological order)
        assignment = [False] * self.n
        for i in range(self.n):
            # Smaller scc_id means later in topological order -> that value is true
            assignment[i] = scc_id[2 * i] > scc_id[2 * i + 1]

        return True, assignment


# =============================================================================
# 5. Practical Problem: School Reachability
# =============================================================================

def min_roads_to_connect(n: int, roads: List[Tuple[int, int]]) -> int:
    """
    Minimum number of roads to add so all schools can reach each other
    = max(number of SCCs with in-degree 0, number of SCCs with out-degree 0)
    (0 if there is only 1 SCC)
    """
    if not roads:
        return n - 1 if n > 1 else 0

    sccs, scc_graph, node_to_scc = build_scc_dag(n, roads)

    if len(sccs) == 1:
        return 0

    # In-degree/out-degree of each SCC
    in_degree = [0] * len(sccs)
    out_degree = [0] * len(sccs)

    for scc_u, neighbors in scc_graph.items():
        out_degree[scc_u] = len(neighbors)
        for scc_v in neighbors:
            in_degree[scc_v] += 1

    # Count SCCs with in-degree/out-degree of 0
    sources = sum(1 for d in in_degree if d == 0)
    sinks = sum(1 for d in out_degree if d == 0)

    return max(sources, sinks)


# =============================================================================
# 6. Practical Problem: Critical Nodes (Similar to Articulation Points)
# =============================================================================

def find_critical_nodes(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """
    Find nodes whose removal increases the number of SCCs
    (Simple brute force implementation)
    """
    original_scc_count = len(tarjan_scc(n, edges))
    critical = []

    for remove_node in range(n):
        # Graph excluding the removed node
        new_edges = [(u, v) for u, v in edges if u != remove_node and v != remove_node]

        # Remap nodes
        remaining = [i for i in range(n) if i != remove_node]
        if not remaining:
            continue

        node_map = {old: new for new, old in enumerate(remaining)}
        remapped_edges = [(node_map[u], node_map[v]) for u, v in new_edges
                          if u in node_map and v in node_map]

        new_scc_count = len(tarjan_scc(len(remaining), remapped_edges))

        if new_scc_count > original_scc_count - 1:  # -1 accounts for the removed node's SCC
            critical.append(remove_node)

    return critical


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Strongly Connected Components (SCC) Examples")
    print("=" * 60)

    # Graph structure
    #   0 -> 1 -> 2
    #   ^    |    |
    #   4 <- 3 -> 5 -> 6
    #        ^        |
    #        +--------+

    n = 7
    edges = [
        (0, 1), (1, 2), (1, 3), (2, 5),
        (3, 4), (4, 0), (3, 5), (5, 6), (6, 3)
    ]

    # 1. Kosaraju's Algorithm
    print("\n[1] Kosaraju's Algorithm")
    sccs = kosaraju_scc(n, edges)
    print(f"    Edges: {edges}")
    print(f"    SCCs: {sccs}")

    # 2. Tarjan's Algorithm
    print("\n[2] Tarjan's Algorithm")
    sccs = tarjan_scc(n, edges)
    print(f"    SCCs: {sccs}")

    # 3. SCC Condensation DAG
    print("\n[3] SCC Condensation Graph (DAG)")
    sccs, scc_graph, node_to_scc = build_scc_dag(n, edges)
    print(f"    SCCs: {sccs}")
    print(f"    Node->SCC: {node_to_scc}")
    print(f"    SCC edges: {dict(scc_graph)}")

    # 4. 2-SAT Problem
    print("\n[4] 2-SAT Problem")
    # (x0 or x1) and (not_x0 or x2) and (not_x1 or not_x2)
    sat = TwoSAT(3)
    sat.add_clause(0, False, 1, False)  # x0 or x1
    sat.add_clause(0, True, 2, False)   # not_x0 or x2
    sat.add_clause(1, True, 2, True)    # not_x1 or not_x2

    solvable, assignment = sat.solve()
    print(f"    Clauses: (x0 or x1) and (not_x0 or x2) and (not_x1 or not_x2)")
    print(f"    Satisfiable: {solvable}")
    if solvable:
        print(f"    Solution: x0={assignment[0]}, x1={assignment[1]}, x2={assignment[2]}")

    # Unsatisfiable 2-SAT
    print("\n    Unsatisfiable case:")
    sat2 = TwoSAT(1)
    sat2.add_clause(0, False, 0, False)  # x0 or x0 = x0
    sat2.add_clause(0, True, 0, True)    # not_x0 or not_x0 = not_x0
    solvable2, _ = sat2.solve()
    print(f"    Clauses: x0 and not_x0")
    print(f"    Satisfiable: {solvable2}")

    # 5. School Connectivity
    print("\n[5] School Connectivity Problem")
    school_roads = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3)]
    min_roads = min_roads_to_connect(5, school_roads)
    print(f"    Roads: {school_roads}")
    print(f"    Additional roads needed: {min_roads}")

    # 6. Algorithm Comparison
    print("\n[6] Kosaraju vs Tarjan Comparison")
    print("    | Property        | Kosaraju      | Tarjan        |")
    print("    |-----------------|---------------|---------------|")
    print("    | Time Complexity | O(V + E)      | O(V + E)      |")
    print("    | DFS Passes      | 2             | 1             |")
    print("    | Reverse Graph   | Required      | Not required  |")
    print("    | Implementation  | Easy          | Moderate      |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
