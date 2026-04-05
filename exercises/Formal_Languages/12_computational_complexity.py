"""
Exercises for Lesson 12: Computational Complexity
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""

from itertools import combinations


# === Exercise 1: Complexity Classification ===
# Problem: For each problem, identify the best known complexity class:
# 1. Shortest path in an unweighted graph
# 2. Determine if a Boolean formula in 3-CNF is satisfiable
# 3. Determine if a QBF is true
# 4. 2-coloring a graph

def exercise_1():
    """Complexity classification of problems."""

    problems = [
        {
            "name": "Shortest path in an unweighted graph",
            "class": "P",
            "explanation": (
                "BFS (Breadth-First Search) finds the shortest path in O(V + E) time, "
                "which is polynomial in the input size. Since the input encodes V vertices "
                "and E edges, this is clearly in P."
            ),
            "algorithm": "BFS",
        },
        {
            "name": "3-SAT: Is a 3-CNF formula satisfiable?",
            "class": "NP-complete",
            "explanation": (
                "3-SAT is the canonical NP-complete problem. It is in NP because a "
                "satisfying assignment serves as a polynomial-size certificate that can "
                "be verified in polynomial time (evaluate the formula). It is NP-hard "
                "by the Cook-Levin theorem (SAT is NP-complete) combined with the "
                "reduction from SAT to 3-SAT."
            ),
            "algorithm": "Backtracking, DPLL, CDCL (exponential worst-case)",
        },
        {
            "name": "QBF: Is a quantified Boolean formula true?",
            "class": "PSPACE-complete",
            "explanation": (
                "TQBF (True Quantified Boolean Formula) is the canonical PSPACE-complete "
                "problem. It generalizes SAT by allowing universal quantifiers (for-all) "
                "in addition to existential ones. The alternation of quantifiers requires "
                "polynomial space to evaluate (by recursion), and every PSPACE problem "
                "can be reduced to TQBF."
            ),
            "algorithm": "Recursive evaluation using O(n) space",
        },
        {
            "name": "2-coloring a graph",
            "class": "P",
            "explanation": (
                "A graph is 2-colorable if and only if it is bipartite. "
                "Bipartiteness can be tested using BFS/DFS in O(V + E) time: "
                "start coloring from any vertex, alternate colors along edges, "
                "and check for conflicts. This is clearly polynomial."
            ),
            "algorithm": "BFS/DFS bipartiteness check, O(V + E)",
        },
    ]

    for i, prob in enumerate(problems, 1):
        print(f"Problem {i}: {prob['name']}")
        print(f"  Complexity class: {prob['class']}")
        print(f"  Algorithm: {prob['algorithm']}")
        print(f"  Explanation: {prob['explanation']}")
        print()

    # Demonstration: 2-coloring via BFS
    print("Demonstration: 2-coloring via BFS")
    print("-" * 40)

    def is_bipartite(adj):
        """Check if graph is bipartite (2-colorable) using BFS."""
        n = len(adj)
        color = [-1] * n
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, color
        return True, color

    # Example: bipartite graph (path graph: 0-1-2-3)
    adj1 = [[1], [0, 2], [1, 3], [2]]
    result1, colors1 = is_bipartite(adj1)
    print(f"  Graph 1 (path 0-1-2-3): bipartite={result1}, colors={colors1}")

    # Example: non-bipartite (triangle: 0-1-2-0)
    adj2 = [[1, 2], [0, 2], [0, 1]]
    result2, colors2 = is_bipartite(adj2)
    print(f"  Graph 2 (triangle 0-1-2): bipartite={result2}")

    # Example: bipartite (complete bipartite K2,3)
    adj3 = [[2, 3, 4], [2, 3, 4], [0, 1], [0, 1], [0, 1]]
    result3, colors3 = is_bipartite(adj3)
    print(f"  Graph 3 (K2,3): bipartite={result3}, colors={colors3}")


# === Exercise 2: Polynomial Reduction ===
# Problem: Show that INDEPENDENT SET <=_P VERTEX COVER.
# Hint: S is an independent set iff V \ S is a vertex cover.

def exercise_2():
    """Polynomial reduction from INDEPENDENT SET to VERTEX COVER."""

    print("Reduction: INDEPENDENT SET <=_P VERTEX COVER")
    print("=" * 60)
    print()
    print("Definitions:")
    print("  INDEPENDENT SET: Given graph G=(V,E) and integer k,")
    print("    does G have an independent set of size >= k?")
    print("    (A set S where no two vertices in S are adjacent.)")
    print()
    print("  VERTEX COVER: Given graph G=(V,E) and integer k,")
    print("    does G have a vertex cover of size <= k?")
    print("    (A set C where every edge has at least one endpoint in C.)")
    print()
    print("Reduction function f:")
    print("  f(<G, k>) = <G, |V| - k>")
    print()
    print("  That is, the graph stays the same, and we change the parameter")
    print("  from k to |V| - k.")
    print()
    print("Correctness proof:")
    print("  Claim: S is an independent set in G iff V\\S is a vertex cover in G.")
    print()
    print("  (=>) Suppose S is an independent set.")
    print("  Let C = V \\ S. Consider any edge (u, v) in E.")
    print("  Since S is independent, u and v cannot BOTH be in S.")
    print("  So at least one of u, v is in V \\ S = C.")
    print("  Therefore C is a vertex cover.")
    print()
    print("  (<=) Suppose C = V \\ S is a vertex cover.")
    print("  Consider any two vertices u, v in S.")
    print("  Since u not in C and v not in C, and C covers all edges,")
    print("  the edge (u,v) cannot exist (it would be uncovered).")
    print("  Therefore S is an independent set.")
    print()
    print("  Combining:")
    print("  G has an independent set of size >= k")
    print("    iff G has a vertex cover of size <= |V| - k")
    print("    iff <G, |V|-k> in VERTEX COVER")
    print()
    print("  f is polynomial-time computable (just arithmetic on k).")
    print("  Therefore INDEPENDENT SET <=_P VERTEX COVER. QED.")

    # Demonstration
    print("\nDemonstration:")
    print("-" * 40)

    # Graph: pentagon (5-cycle)
    #   0 - 1
    #   |   |
    #   4   2
    #    \ /
    #     3
    V = {0, 1, 2, 3, 4}
    E = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    n = len(V)

    def is_independent_set(S, edges):
        """Check if S is an independent set."""
        for u, v in edges:
            if u in S and v in S:
                return False
        return True

    def is_vertex_cover(C, edges):
        """Check if C is a vertex cover."""
        for u, v in edges:
            if u not in C and v not in C:
                return False
        return True

    print(f"  Graph: 5-cycle (0-1-2-3-4-0)")
    print(f"  |V| = {n}")
    print()

    # Find all independent sets of size k
    for k in range(n + 1):
        ind_sets = []
        for combo in combinations(V, k):
            S = set(combo)
            if is_independent_set(S, E):
                ind_sets.append(S)

        complement_k = n - k
        vc_sets = []
        for combo in combinations(V, complement_k):
            C = set(combo)
            if is_vertex_cover(C, E):
                vc_sets.append(C)

        has_ind = len(ind_sets) > 0
        has_vc = len(vc_sets) > 0
        status = "OK" if has_ind == has_vc else "MISMATCH"

        print(f"  k={k}: Independent set size >= {k}? {has_ind} | "
              f"Vertex cover size <= {complement_k}? {has_vc} [{status}]")
        if ind_sets and k <= 2:
            print(f"         Example IS: {ind_sets[0]}, complement VC: {V - ind_sets[0]}")


# === Exercise 3: NP Membership ===
# Problem: Show each problem is in NP by describing certificate + verifier:
# 1. COMPOSITE: Given n, is n composite?
# 2. GRAPH ISOMORPHISM: Given G1, G2, are they isomorphic?
# 3. SET COVER: Given sets S1,...,Sm and integer k, can k sets cover all elements?

def exercise_3():
    """NP membership proofs via certificates and verifiers."""

    print("Part 1: COMPOSITE is in NP")
    print("=" * 60)
    print("  Problem: Given integer n, is n composite (not prime)?")
    print()
    print("  Certificate: Two integers a, b > 1 such that a * b = n.")
    print()
    print("  Verifier V(n, (a, b)):")
    print("    1. Check a > 1 and b > 1.")
    print("    2. Compute a * b.")
    print("    3. Accept if a * b = n.")
    print()
    print("  Certificate size: a and b are at most n, so they need")
    print("    O(log n) bits each -- polynomial in the input size.")
    print("  Verification time: multiplication is O(log^2 n) -- polynomial.")
    print()
    print("  Note: COMPOSITE (equivalently PRIMES) is actually in P")
    print("  (AKS primality test, 2002), but the NP certificate is simpler.")

    # Demonstration
    print("\n  Demonstration:")
    test_nums = [4, 15, 17, 100, 97, 561]
    for n in test_nums:
        # Find factors
        factor = None
        for a in range(2, int(n**0.5) + 1):
            if n % a == 0:
                factor = (a, n // a)
                break
        if factor:
            a, b = factor
            verified = a > 1 and b > 1 and a * b == n
            print(f"    n={n}: certificate=({a}, {b}), "
                  f"a*b={a*b}, verified={verified} -> COMPOSITE")
        else:
            print(f"    n={n}: no certificate found -> PRIME")

    print()
    print()
    print("Part 2: GRAPH ISOMORPHISM is in NP")
    print("=" * 60)
    print("  Problem: Given graphs G1=(V1,E1) and G2=(V2,E2),")
    print("  are they isomorphic?")
    print()
    print("  Certificate: A bijection f: V1 -> V2.")
    print()
    print("  Verifier V((G1, G2), f):")
    print("    1. Check f is a bijection (|V1| = |V2|, f is one-to-one and onto).")
    print("    2. For each edge (u, v) in E1, check that (f(u), f(v)) in E2.")
    print("    3. For each edge (u, v) in E2, check that (f^{-1}(u), f^{-1}(v)) in E1.")
    print("    4. Accept if all checks pass.")
    print()
    print("  Certificate size: O(|V| log |V|) -- the mapping for each vertex.")
    print("  Verification time: O(|V| + |E|) -- polynomial.")

    # Demonstration
    print("\n  Demonstration:")
    # G1: 0-1, 1-2, 2-0 (triangle)
    # G2: a-b, b-c, c-a (triangle, different labels)
    G1_edges = [(0, 1), (1, 2), (2, 0)]
    G2_edges = [("a", "b"), ("b", "c"), ("c", "a")]
    mapping = {0: "a", 1: "b", 2: "c"}

    print(f"    G1 edges: {G1_edges}")
    print(f"    G2 edges: {G2_edges}")
    print(f"    Certificate (mapping): {mapping}")

    all_mapped = True
    for u, v in G1_edges:
        mapped_edge = (mapping[u], mapping[v])
        # Check both directions since edges are undirected
        in_G2 = mapped_edge in G2_edges or (mapped_edge[1], mapped_edge[0]) in G2_edges
        print(f"    ({u},{v}) -> ({mapping[u]},{mapping[v]}): in G2? {in_G2}")
        if not in_G2:
            all_mapped = False
    print(f"    Isomorphic: {all_mapped}")

    print()
    print()
    print("Part 3: SET COVER is in NP")
    print("=" * 60)
    print("  Problem: Given universe U, sets S1,...,Sm, and integer k,")
    print("  can we select at most k sets whose union equals U?")
    print()
    print("  Certificate: A subset I of {1, ..., m} with |I| <= k.")
    print()
    print("  Verifier V((U, S1,...,Sm, k), I):")
    print("    1. Check |I| <= k.")
    print("    2. Compute the union: C = union of S_i for i in I.")
    print("    3. Accept if C = U.")
    print()
    print("  Certificate size: O(m) -- at most m indices.")
    print("  Verification time: O(m * |U|) -- compute union and compare.")

    # Demonstration
    print("\n  Demonstration:")
    U = {1, 2, 3, 4, 5}
    sets = [
        {1, 2, 3},   # S1
        {2, 4},       # S2
        {3, 4, 5},   # S3
        {1, 5},       # S4
    ]
    k = 2

    print(f"    U = {U}")
    for i, s in enumerate(sets, 1):
        print(f"    S{i} = {s}")
    print(f"    k = {k}")

    # Try all k-subsets
    found = False
    for combo in combinations(range(len(sets)), k):
        union = set()
        for idx in combo:
            union |= sets[idx]
        if union == U:
            indices = [i + 1 for i in combo]
            print(f"    Certificate: I = {indices}")
            print(f"    Union = {union}")
            print(f"    |I| = {len(indices)} <= k = {k}: True")
            print(f"    Union = U: {union == U}")
            print(f"    -> ACCEPT (valid cover)")
            found = True
            break

    if not found:
        print(f"    No {k}-set cover exists -> REJECT")

    # Also show a 3-set cover
    print(f"\n    All possible {k}-subset covers:")
    for combo in combinations(range(len(sets)), k):
        union = set()
        for idx in combo:
            union |= sets[idx]
        indices = [i + 1 for i in combo]
        covers = union == U
        print(f"      S{indices}: union={union}, covers U? {covers}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Complexity Classification ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Polynomial Reduction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: NP Membership ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
