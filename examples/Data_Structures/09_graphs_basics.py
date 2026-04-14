"""
09 Graphs Basics
================
Demonstrates graph representations, BFS, DFS,
cycle detection, and topological sort.
"""

from collections import deque


class Graph:
    """Graph using adjacency list."""
    def __init__(self, directed=False):
        self._adj = {}
        self._directed = directed

    def add_edge(self, u, v):
        self._adj.setdefault(u, []).append(v)
        if not self._directed:
            self._adj.setdefault(v, []).append(u)
        else:
            self._adj.setdefault(v, [])

    def neighbors(self, v): return self._adj.get(v, [])
    def vertices(self): return self._adj.keys()


def bfs(graph, start):
    visited = {start}; queue = deque([start]); order = []
    while queue:
        v = queue.popleft(); order.append(v)
        for n in graph.neighbors(v):
            if n not in visited: visited.add(n); queue.append(n)
    return order


def dfs(graph, start):
    visited = set(); stack = [start]; order = []
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v); order.append(v)
            for n in reversed(graph.neighbors(v)):
                if n not in visited: stack.append(n)
    return order


def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph.vertices()}
    def visit(v):
        color[v] = GRAY
        for n in graph.neighbors(v):
            if color[n] == GRAY: return True
            if color[n] == WHITE and visit(n): return True
        color[v] = BLACK; return False
    return any(visit(v) for v in graph.vertices() if color[v] == WHITE)


def topological_sort(graph):
    in_deg = {v: 0 for v in graph.vertices()}
    for v in graph.vertices():
        for n in graph.neighbors(v): in_deg[n] += 1
    queue = deque([v for v in graph.vertices() if in_deg[v] == 0])
    result = []
    while queue:
        v = queue.popleft(); result.append(v)
        for n in graph.neighbors(v):
            in_deg[n] -= 1
            if in_deg[n] == 0: queue.append(n)
    return result


if __name__ == "__main__":
    g = Graph()
    for u, v in [('A','B'), ('A','C'), ('B','D'), ('B','E'), ('C','F'), ('E','F')]:
        g.add_edge(u, v)
    print(f"BFS from A: {bfs(g, 'A')}")
    print(f"DFS from A: {dfs(g, 'A')}")

    dg = Graph(directed=True)
    for u, v in [('A','B'), ('A','C'), ('B','D'), ('C','D'), ('D','E')]:
        dg.add_edge(u, v)
    print(f"\nDAG has cycle: {has_cycle_directed(dg)}")
    print(f"Topological sort: {topological_sort(dg)}")

    cg = Graph(directed=True)
    for u, v in [('A','B'), ('B','C'), ('C','A')]:
        cg.add_edge(u, v)
    print(f"Cyclic graph has cycle: {has_cycle_directed(cg)}")
