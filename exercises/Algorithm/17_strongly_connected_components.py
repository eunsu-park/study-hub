"""
Exercises for Lesson 17: Strongly Connected Components
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: Count SCCs (Tarjan/Kosaraju), Domino (SCC+DAG), 2-SAT basic.
"""


# === Exercise 1: Count SCCs (Tarjan's Algorithm) ===
# Problem: Count the number of Strongly Connected Components in a directed graph.
# Approach: Tarjan's algorithm uses a DFS with a stack and low-link values
#   to identify SCCs in a single pass.

def exercise_1():
    """Solution: Tarjan's algorithm for SCC detection."""
    def count_sccs_tarjan(n, adj):
        """
        n: number of vertices (0-indexed)
        adj: adjacency list for directed graph
        Returns: (number of SCCs, list of SCCs)
        """
        index_counter = [0]
        stack = []
        on_stack = [False] * n
        index = [-1] * n
        lowlink = [-1] * n
        sccs = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in adj[v]:
                if index[w] == -1:
                    # w has not been visited
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    # w is on stack, hence in current SCC
                    lowlink[v] = min(lowlink[v], index[w])

            # If v is the root of an SCC
            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in range(n):
            if index[v] == -1:
                strongconnect(v)

        return len(sccs), sccs

    # Test case 1: Simple SCC
    # 0 -> 1 -> 2 -> 0 (one SCC), 2 -> 3 (3 is separate SCC)
    n = 4
    adj = [[] for _ in range(n)]
    adj[0].append(1)
    adj[1].append(2)
    adj[2].append(0)
    adj[2].append(3)

    count, sccs = count_sccs_tarjan(n, adj)
    print(f"Graph 1: {count} SCCs")
    for scc in sccs:
        print(f"  SCC: {sorted(scc)}")
    assert count == 2

    # Test case 2: All separate
    n = 3
    adj = [[] for _ in range(n)]
    adj[0].append(1)
    adj[1].append(2)

    count, sccs = count_sccs_tarjan(n, adj)
    print(f"\nGraph 2 (chain): {count} SCCs")
    assert count == 3  # each node is its own SCC

    # Test case 3: Two cycles
    # 0 -> 1 -> 0, 2 -> 3 -> 4 -> 2, 1 -> 2
    n = 5
    adj = [[] for _ in range(n)]
    adj[0].append(1)
    adj[1].append(0)
    adj[1].append(2)
    adj[2].append(3)
    adj[3].append(4)
    adj[4].append(2)

    count, sccs = count_sccs_tarjan(n, adj)
    print(f"\nGraph 3 (two cycles): {count} SCCs")
    for scc in sccs:
        print(f"  SCC: {sorted(scc)}")
    assert count == 2  # {0,1} and {2,3,4}

    print("All Tarjan SCC tests passed!")


# === Exercise 2: Count SCCs (Kosaraju's Algorithm) ===
# Problem: Same as above but using Kosaraju's two-pass algorithm.
# Approach: (1) DFS on original graph to get finish order,
#   (2) DFS on reversed graph in reverse finish order.

def exercise_2():
    """Solution: Kosaraju's algorithm for SCC detection."""
    def count_sccs_kosaraju(n, adj):
        # Step 1: DFS on original graph, record finish order
        visited = [False] * n
        finish_order = []

        def dfs1(v):
            visited[v] = True
            for u in adj[v]:
                if not visited[u]:
                    dfs1(u)
            finish_order.append(v)

        for v in range(n):
            if not visited[v]:
                dfs1(v)

        # Step 2: Build reversed graph
        rev_adj = [[] for _ in range(n)]
        for v in range(n):
            for u in adj[v]:
                rev_adj[u].append(v)

        # Step 3: DFS on reversed graph in reverse finish order
        visited = [False] * n
        sccs = []

        def dfs2(v, scc):
            visited[v] = True
            scc.append(v)
            for u in rev_adj[v]:
                if not visited[u]:
                    dfs2(u, scc)

        for v in reversed(finish_order):
            if not visited[v]:
                scc = []
                dfs2(v, scc)
                sccs.append(scc)

        return len(sccs), sccs

    # Same test cases as Tarjan
    # Test case 1
    n = 4
    adj = [[] for _ in range(n)]
    adj[0].append(1)
    adj[1].append(2)
    adj[2].append(0)
    adj[2].append(3)

    count, sccs = count_sccs_kosaraju(n, adj)
    print(f"Kosaraju - Graph 1: {count} SCCs")
    assert count == 2

    # Test case 2
    n = 5
    adj = [[] for _ in range(n)]
    adj[0].append(1)
    adj[1].append(0)
    adj[1].append(2)
    adj[2].append(3)
    adj[3].append(4)
    adj[4].append(2)

    count, sccs = count_sccs_kosaraju(n, adj)
    print(f"Kosaraju - Graph 2 (two cycles): {count} SCCs")
    for scc in sccs:
        print(f"  SCC: {sorted(scc)}")
    assert count == 2

    print("All Kosaraju SCC tests passed!")


# === Exercise 3: Condensation Graph ===
# Problem: Build the DAG obtained by contracting each SCC into a single node.
#   Count the number of nodes with in-degree 0 in the condensation graph.
#   This tells us the minimum number of "domino pushes" needed.

def exercise_3():
    """Solution: SCC condensation + in-degree analysis."""
    def min_pushes(n, adj):
        """
        After condensing SCCs, count nodes with in-degree 0.
        These are the minimum number of starting points needed.
        """
        # Step 1: Find SCCs using Tarjan
        index_counter = [0]
        stack = []
        on_stack = [False] * n
        index = [-1] * n
        lowlink = [-1] * n
        comp = [-1] * n  # which SCC each node belongs to
        sccs = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in adj[v]:
                if index[w] == -1:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = len(sccs)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in range(n):
            if index[v] == -1:
                strongconnect(v)

        num_sccs = len(sccs)
        if num_sccs == 1:
            return 1

        # Step 2: Build condensation graph, count in-degrees
        in_degree = [0] * num_sccs
        seen_edges = set()
        for v in range(n):
            for w in adj[v]:
                if comp[v] != comp[w]:
                    edge = (comp[v], comp[w])
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        in_degree[comp[w]] += 1

        # Count nodes with in-degree 0
        return sum(1 for d in in_degree if d == 0)

    # Test case 1: one large SCC
    n = 3
    adj = [[1], [2], [0]]
    result = min_pushes(n, adj)
    print(f"Single cycle (3 nodes): min pushes = {result}")
    assert result == 1

    # Test case 2: chain of SCCs
    # SCC1: {0,1}, SCC2: {2}, SCC3: {3}
    # 0->1->0 (cycle), 1->2, 2->3
    n = 4
    adj = [[1], [0, 2], [3], []]
    result = min_pushes(n, adj)
    print(f"Chain of 3 SCCs: min pushes = {result}")
    assert result == 1  # only SCC containing {0,1} has in-degree 0

    # Test case 3: two independent chains
    # 0->1, 2->3 (no connection between the two)
    n = 4
    adj = [[1], [], [3], []]
    result = min_pushes(n, adj)
    print(f"Two independent chains: min pushes = {result}")
    assert result == 2  # two sources

    print("All Condensation Graph tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Count SCCs (Tarjan) ===")
    exercise_1()
    print("\n=== Exercise 2: Count SCCs (Kosaraju) ===")
    exercise_2()
    print("\n=== Exercise 3: Condensation Graph (Domino) ===")
    exercise_3()
    print("\nAll exercises completed!")
