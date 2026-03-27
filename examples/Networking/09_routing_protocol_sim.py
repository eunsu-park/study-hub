"""
Routing Protocol Simulator

Demonstrates:
- Distance Vector routing (Bellman-Ford)
- Link-State routing (Dijkstra)
- Count-to-infinity problem
- Split horizon / poison reverse

Theory:
- Distance Vector (DV): Each router shares its distance table with
  neighbors. Bellman-Ford algorithm converges to shortest paths.
  Used in RIP. Vulnerable to count-to-infinity.
- Link-State (LS): Each router floods its link states to all routers.
  Every router runs Dijkstra on the complete topology.
  Used in OSPF. Faster convergence, more memory.

Adapted from Networking Lesson 09.
"""

from collections import defaultdict
import heapq

INF = float("inf")


# ── Network Topology ───────────────────────────────────────────────────

# Why: Modeling the network as an undirected weighted graph is the natural
# abstraction — nodes are routers, edges are links, and weights are costs
# (hop count for RIP, bandwidth-derived for OSPF). Both DV and LS
# algorithms operate on this same graph but use fundamentally different
# information-sharing strategies.
class Topology:
    """Network graph for routing simulations."""

    def __init__(self):
        self.links: dict[str, dict[str, int]] = defaultdict(dict)

    def add_link(self, a: str, b: str, cost: int) -> None:
        self.links[a][b] = cost
        self.links[b][a] = cost

    def remove_link(self, a: str, b: str) -> None:
        self.links[a].pop(b, None)
        self.links[b].pop(a, None)

    def neighbors(self, node: str) -> dict[str, int]:
        return dict(self.links.get(node, {}))

    @property
    def nodes(self) -> list[str]:
        return sorted(set(self.links.keys()))


# ── Distance Vector ───────────────────────────────────────────────────

class DistanceVector:
    """Distance Vector routing protocol simulation."""

    # Why: In DV routing, each router only knows its neighbors' distance tables
    # (not the full topology). This mirrors RIP's design: routers periodically
    # broadcast their distance vectors to neighbors. The trade-off vs. LS is
    # less memory/bandwidth but slower convergence and vulnerability to loops.
    def __init__(self, topology: Topology, split_horizon: bool = False):
        self.topo = topology
        self.split_horizon = split_horizon
        # distance[node][dest] = (cost, next_hop)
        self.tables: dict[str, dict[str, tuple[int, str]]] = {}
        self.iterations = 0

        # Initialize: each node knows only its direct neighbors
        for node in topology.nodes:
            self.tables[node] = {}
            self.tables[node][node] = (0, node)
            for neighbor, cost in topology.neighbors(node).items():
                self.tables[node][neighbor] = (cost, neighbor)

    def iterate(self) -> bool:
        """Run one round of DV exchange. Returns True if tables changed."""
        changed = False
        self.iterations += 1

        for node in self.topo.nodes:
            for neighbor, link_cost in self.topo.neighbors(node).items():
                # Receive neighbor's distance table
                for dest, (cost, _) in self.tables[neighbor].items():
                    # Why: Split horizon prevents a router from advertising a route
                    # back to the neighbor it learned the route from. Without this,
                    # after a link failure, routers can form a routing loop where
                    # each thinks the other has a valid path (count-to-infinity).
                    if self.split_horizon and _ == node:
                        continue

                    new_cost = link_cost + cost
                    current = self.tables[node].get(dest, (INF, ""))

                    if new_cost < current[0]:
                        self.tables[node][dest] = (new_cost, neighbor)
                        changed = True

        return changed

    def converge(self, max_iter: int = 50) -> int:
        """Run until convergence. Returns number of iterations."""
        for _ in range(max_iter):
            if not self.iterate():
                break
        return self.iterations

    def print_table(self, node: str) -> None:
        table = self.tables.get(node, {})
        print(f"    {'Dest':<8} {'Cost':>6} {'Next Hop':<10}")
        print(f"    {'-'*8} {'-'*6} {'-'*10}")
        for dest in sorted(table):
            cost, next_hop = table[dest]
            cost_str = str(cost) if cost < INF else "∞"
            print(f"    {dest:<8} {cost_str:>6} {next_hop:<10}")


# ── Link-State ─────────────────────────────────────────────────────────

class LinkState:
    """Link-State routing protocol simulation (Dijkstra)."""

    def __init__(self, topology: Topology):
        self.topo = topology

    # Why: In Link-State routing, every router has the complete topology
    # (flooded via LSAs) and independently runs Dijkstra. This guarantees
    # loop-free routing and fast convergence, at the cost of more memory
    # and flooding overhead. OSPF uses this approach in production networks.
    def dijkstra(self, source: str) -> dict[str, tuple[int, str]]:
        """Run Dijkstra from source. Returns {dest: (cost, next_hop)}."""
        dist: dict[str, int] = {source: 0}
        prev: dict[str, str | None] = {source: None}
        visited: set[str] = set()
        heap: list[tuple[int, str]] = [(0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            for v, weight in self.topo.neighbors(u).items():
                new_dist = d + weight
                if new_dist < dist.get(v, INF):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        # Why: Dijkstra gives us the shortest-path tree (prev pointers), but
        # routers need next-hop information (not full paths). We trace back from
        # each destination to find the first hop — that's all a forwarding table
        # needs to make per-packet decisions.
        result: dict[str, tuple[int, str]] = {}
        for dest in self.topo.nodes:
            if dest == source:
                result[dest] = (0, source)
                continue
            if dest not in prev:
                continue
            # Trace back to find next hop
            current = dest
            while prev.get(current) != source and prev.get(current) is not None:
                current = prev[current]
            result[dest] = (dist.get(dest, INF), current)

        return result

    def all_routes(self) -> dict[str, dict[str, tuple[int, str]]]:
        """Compute routing tables for all nodes."""
        return {node: self.dijkstra(node) for node in self.topo.nodes}


# ── Demos ──────────────────────────────────────────────────────────────

def demo_distance_vector():
    print("=" * 60)
    print("DISTANCE VECTOR ROUTING")
    print("=" * 60)

    topo = Topology()
    topo.add_link("A", "B", 1)
    topo.add_link("B", "C", 2)
    topo.add_link("A", "C", 10)
    topo.add_link("C", "D", 1)

    print(f"\n  Topology: A—1—B—2—C—1—D, A—10—C")

    dv = DistanceVector(topo)
    print(f"\n  Initial tables (direct neighbors only):")
    for node in topo.nodes:
        print(f"\n  Router {node}:")
        dv.print_table(node)

    iters = dv.converge()
    print(f"\n  After convergence ({iters} iterations):")
    for node in topo.nodes:
        print(f"\n  Router {node}:")
        dv.print_table(node)


def demo_link_state():
    print("\n" + "=" * 60)
    print("LINK-STATE ROUTING (DIJKSTRA)")
    print("=" * 60)

    topo = Topology()
    topo.add_link("A", "B", 1)
    topo.add_link("B", "C", 2)
    topo.add_link("A", "C", 10)
    topo.add_link("C", "D", 1)

    print(f"\n  Topology: A—1—B—2—C—1—D, A—10—C")

    ls = LinkState(topo)
    routes = ls.all_routes()

    for node in topo.nodes:
        print(f"\n  Router {node} (Dijkstra):")
        table = routes[node]
        print(f"    {'Dest':<8} {'Cost':>6} {'Next Hop':<10}")
        print(f"    {'-'*8} {'-'*6} {'-'*10}")
        for dest in sorted(table):
            cost, nh = table[dest]
            print(f"    {dest:<8} {cost:>6} {nh:<10}")


def demo_dv_vs_ls():
    print("\n" + "=" * 60)
    print("DV vs LS COMPARISON")
    print("=" * 60)

    topo = Topology()
    topo.add_link("A", "B", 1)
    topo.add_link("B", "C", 2)
    topo.add_link("A", "C", 10)
    topo.add_link("C", "D", 1)
    topo.add_link("B", "D", 5)

    dv = DistanceVector(topo)
    iters = dv.converge()

    ls = LinkState(topo)
    ls_routes = ls.all_routes()

    print(f"\n  DV converged in {iters} iterations")
    print(f"  LS computed instantly (Dijkstra)")

    print(f"\n  Route comparison (source: A):")
    print(f"    {'Dest':<6} {'DV Cost':>8} {'DV NH':>6} {'LS Cost':>8} {'LS NH':>6} {'Match':>6}")
    print(f"    {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*6}")

    for dest in topo.nodes:
        dv_cost, dv_nh = dv.tables["A"].get(dest, (INF, "?"))
        ls_cost, ls_nh = ls_routes["A"].get(dest, (INF, "?"))
        match = "✓" if dv_cost == ls_cost else "✗"
        print(f"    {dest:<6} {dv_cost:>8} {dv_nh:>6} {ls_cost:>8} {ls_nh:>6} {match:>6}")


def demo_count_to_infinity():
    print("\n" + "=" * 60)
    print("COUNT-TO-INFINITY PROBLEM")
    print("=" * 60)

    topo = Topology()
    topo.add_link("A", "B", 1)
    topo.add_link("B", "C", 1)

    print(f"\n  Topology: A—1—B—1—C")

    # Converge normally
    dv = DistanceVector(topo)
    dv.converge()
    print(f"\n  Normal state — A's route to C: cost={dv.tables['A']['C'][0]}")

    # Now remove the B-C link
    topo.remove_link("B", "C")
    print(f"  Link B—C removed!")

    # B still thinks A can reach C (through B!) — count to infinity
    # Simulate a few iterations to show the problem
    print(f"\n  DV iterations after link failure (no split horizon):")
    print(f"    {'Iter':>6}  {'B→C cost':>10}  {'A→C cost':>10}")
    print(f"    {'-'*6}  {'-'*10}  {'-'*10}")

    # Why: After the B-C link fails, we manually set B's cost to C as INF
    # to simulate B detecting the link is down. The problem is that A still
    # advertises its old route to C (via B), so B re-learns a bogus path
    # through A, and the cost increments by 2 each round — the classic
    # count-to-infinity problem that plagues distance vector protocols.
    dv.tables["B"]["C"] = (INF, "")

    for i in range(8):
        dv.iterate()
        b_cost = dv.tables["B"].get("C", (INF, ""))[0]
        a_cost = dv.tables["A"].get("C", (INF, ""))[0]
        b_str = str(int(b_cost)) if b_cost < INF else "∞"
        a_str = str(int(a_cost)) if a_cost < INF else "∞"
        print(f"    {i+1:>6}  {b_str:>10}  {a_str:>10}")
        if b_cost >= 16:  # RIP's infinity
            print(f"\n  Stopped at RIP infinity (16). "
                  f"Took {i+1} iterations!")
            break

    # With split horizon
    print(f"\n  With split horizon:")
    dv_sh = DistanceVector(topo, split_horizon=True)
    dv_sh.tables["B"]["C"] = (INF, "")
    dv_sh.converge(max_iter=5)
    b_cost = dv_sh.tables["B"].get("C", (INF, ""))[0]
    a_cost = dv_sh.tables["A"].get("C", (INF, ""))[0]
    print(f"    B→C: {'∞' if b_cost >= INF else b_cost}")
    print(f"    A→C: {'∞' if a_cost >= INF else a_cost}")
    print(f"    Split horizon prevents the routing loop!")


if __name__ == "__main__":
    demo_distance_vector()
    demo_link_state()
    demo_dv_vs_ls()
    demo_count_to_infinity()
