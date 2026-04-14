"""
Exercise 09: Graphs Basics

Practice graph traversals, cycle detection, and topological sort.
"""

from collections import deque


def bfs_shortest_path(graph, start, end):
    """Find shortest path in unweighted graph using BFS.

    Args:
        graph: Dict of adjacency lists {node: [neighbors]}.
        start: Starting node.
        end: Target node.

    Returns:
        List representing the shortest path, or None if no path.

    >>> g = {'A': ['B','C'], 'B': ['A','D'], 'C': ['A','D'], 'D': ['B','C','E'], 'E': ['D']}
    >>> bfs_shortest_path(g, 'A', 'E')
    ['A', 'B', 'D', 'E']
    """
    # TODO: Implement this
    pass


def count_connected_components(graph):
    """Count connected components in undirected graph.

    Args:
        graph: Dict of adjacency lists.

    Returns:
        Number of connected components.

    >>> count_connected_components({'A': ['B'], 'B': ['A'], 'C': ['D'], 'D': ['C'], 'E': []})
    3
    """
    # TODO: Implement this
    pass


def has_cycle_directed(graph):
    """Detect cycle in a directed graph.

    Args:
        graph: Dict of adjacency lists (directed).

    Returns:
        True if cycle exists.

    >>> has_cycle_directed({'A': ['B'], 'B': ['C'], 'C': ['A']})
    True
    >>> has_cycle_directed({'A': ['B'], 'B': ['C'], 'C': []})
    False
    """
    # TODO: Implement this
    pass


def topological_sort(graph):
    """Topological sort of a DAG using Kahn's algorithm.

    Args:
        graph: Dict of adjacency lists (directed).

    Returns:
        List of vertices in topological order.
        Raises ValueError if graph has a cycle.

    >>> topological_sort({'A': ['B','C'], 'B': ['D'], 'C': ['D'], 'D': []})
    ['A', 'B', 'C', 'D']
    """
    # TODO: Implement this
    pass


def is_bipartite(graph):
    """Check if an undirected graph is bipartite (2-colorable).

    >>> is_bipartite({'A': ['B','C'], 'B': ['A'], 'C': ['A']})
    True
    >>> is_bipartite({'A': ['B','C'], 'B': ['A','C'], 'C': ['A','B']})
    False
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    g = {'A': ['B','C'], 'B': ['A','D'], 'C': ['A','D'], 'D': ['B','C','E'], 'E': ['D']}
    path = bfs_shortest_path(g, 'A', 'E')
    assert path is not None and path[0] == 'A' and path[-1] == 'E' and len(path) == 4
    print("bfs_shortest_path: PASSED")

    assert count_connected_components({'A':['B'], 'B':['A'], 'C':['D'], 'D':['C'], 'E':[]}) == 3
    print("count_connected_components: PASSED")

    assert has_cycle_directed({'A':['B'], 'B':['C'], 'C':['A']}) is True
    assert has_cycle_directed({'A':['B'], 'B':['C'], 'C':[]}) is False
    print("has_cycle_directed: PASSED")

    result = topological_sort({'A':['B','C'], 'B':['D'], 'C':['D'], 'D':[]})
    assert result.index('A') < result.index('B') < result.index('D')
    assert result.index('A') < result.index('C') < result.index('D')
    print("topological_sort: PASSED")

    assert is_bipartite({'A':['B','C'], 'B':['A'], 'C':['A']}) is True
    assert is_bipartite({'A':['B','C'], 'B':['A','C'], 'C':['A','B']}) is False
    print("is_bipartite: PASSED")

    print("\nAll tests passed!")
