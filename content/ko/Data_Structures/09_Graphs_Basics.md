# 그래프 기초

**이전**: [힙](./08_Heaps.md) | **다음**: [집합과 맵](./10_Sets_and_Maps.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 그래프 용어를 정의할 수 있다: 정점, 엣지, 방향/무방향, 가중, 차수
2. 인접 리스트와 인접 행렬을 사용하여 그래프를 표현할 수 있다
3. BFS와 DFS 순회를 구현할 수 있다
4. 방향 그래프와 무방향 그래프에서 순환을 감지할 수 있다
5. 연결 요소를 찾고 연결성을 판별할 수 있다
6. 방향 비순환 그래프 (DAG)에서 위상 정렬을 수행할 수 있다
7. 주어진 문제에 적합한 그래프 표현을 선택할 수 있다

---

**그래프**는 **엣지**로 연결된 **정점** (노드)의 모음입니다. 그래프는 관계를 모델링하는 가장 일반적인 자료구조이며, 소셜 네트워크, 지도, 의존성 시스템 등 수많은 응용에 사용됩니다.

## 그래프 용어

```
무방향 그래프:                 방향 그래프 (다이그래프):
    A --- B                        A ---> B
    |   / |                        |     /|
    |  /  |                        v   v  v
    C --- D                        C ---> D
```

| 용어 | 정의 |
|------|------|
| **정점 (노드)** | 그래프의 한 점 |
| **엣지** | 두 정점 간의 연결 |
| **차수** | 정점에 연결된 엣지의 수 |
| **진입 차수** | (방향) 들어오는 엣지의 수 |
| **진출 차수** | (방향) 나가는 엣지의 수 |
| **경로** | 엣지로 연결된 정점의 순서 |
| **순환** | 시작과 끝이 같은 경로 |
| **연결** | 모든 정점 쌍 사이에 경로 존재 |
| **DAG** | 방향 비순환 그래프 (순환 없음) |

## 그래프 표현

### 인접 리스트

각 정점이 이웃 목록을 저장합니다. 희소 그래프에 가장 적합:

```python
class Graph:
    """인접 리스트 (딕셔너리)를 사용한 그래프."""
    
    def __init__(self, directed=False):
        self._adj = {}
        self._directed = directed
    
    def add_edge(self, u, v):
        self._adj.setdefault(u, []).append(v)
        if not self._directed:
            self._adj.setdefault(v, []).append(u)
    
    def neighbors(self, v):
        return self._adj.get(v, [])
```

```
무방향 그래프의 인접 리스트:
A: [B, C]       메모리: O(V + E)
B: [A, C, D]
C: [A, B, D]
D: [B, C]
```

### 인접 행렬

`matrix[i][j] = 1`이면 엣지 (i, j)가 존재:

```
인접 행렬 (A=0, B=1, C=2, D=3):
     A  B  C  D       메모리: O(V^2)
A [  0  1  1  0 ]
B [  1  0  1  1 ]
C [  1  1  0  1 ]
D [  0  1  1  0 ]
```

### 비교

| 측면 | 인접 리스트 | 인접 행렬 |
|------|-----------|----------|
| 공간 | O(V + E) | O(V^2) |
| 엣지 확인 | O(차수) | **O(1)** |
| 이웃 순회 | O(차수) | O(V) |
| 적합한 용도 | 희소 그래프 | 밀집 그래프 |

## 너비 우선 탐색 (BFS)

BFS는 현재 깊이의 모든 이웃을 탐색한 후 더 깊이 이동합니다. **큐**를 사용합니다.

```python
from collections import deque

def bfs(graph, start):
    """너비 우선 탐색."""
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        for neighbor in graph.neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order

def bfs_shortest_path(graph, start, end):
    """BFS를 사용한 최단 경로 (비가중)."""
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        vertex, path = queue.popleft()
        for neighbor in graph.neighbors(vertex):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None
```

## 깊이 우선 탐색 (DFS)

DFS는 각 가지를 따라 가능한 한 깊이 탐색한 후 역추적합니다. **스택** (또는 재귀)을 사용합니다.

```python
def dfs_iterative(graph, start):
    """깊이 우선 탐색 (반복, 스택 사용)."""
    visited = set()
    stack = [start]
    order = []
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            for neighbor in reversed(graph.neighbors(vertex)):
                if neighbor not in visited:
                    stack.append(neighbor)
    return order
```

### BFS vs DFS

| 측면 | BFS | DFS |
|------|-----|-----|
| 자료구조 | 큐 | 스택 (또는 재귀) |
| 탐색 방식 | 레벨별 | 가지별 |
| 최단 경로 | 예 (비가중) | 아니오 |
| 적합한 용도 | 최단 경로, 레벨 순서 | 순환 감지, 위상 정렬 |

## 순환 감지

### 방향 그래프

```python
def has_cycle_directed(graph):
    """착색을 사용한 방향 그래프 순환 감지."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph.vertices()}
    
    def dfs(vertex):
        color[vertex] = GRAY
        for neighbor in graph.neighbors(vertex):
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[vertex] = BLACK
        return False
    
    return any(dfs(v) for v in graph.vertices() if color[v] == WHITE)
```

## 위상 정렬

모든 엣지 (u, v)에 대해 u가 v 앞에 오는 DAG의 선형 순서:

```python
def topological_sort_kahn(graph):
    """칸의 알고리즘을 사용한 위상 정렬 (BFS 기반)."""
    in_degree = {v: 0 for v in graph.vertices()}
    for v in graph.vertices():
        for neighbor in graph.neighbors(v):
            in_degree[neighbor] += 1
    
    queue = deque([v for v in graph.vertices() if in_degree[v] == 0])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbor in graph.neighbors(vertex):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(in_degree):
        raise ValueError("그래프에 순환 존재 -- 위상 정렬 불가")
    return result
```

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 그래프 | 관계를 모델링하는 정점 + 엣지 |
| 인접 리스트 | O(V+E) 공간, 희소 그래프에 적합 |
| 인접 행렬 | O(V^2) 공간, O(1) 엣지 조회 |
| BFS | 큐 기반, 최단 경로 (비가중) |
| DFS | 스택 기반, 깊이 우선 탐색 |
| 순환 감지 | 방향 그래프에 착색 기반 DFS |
| 연결 요소 | 미방문 각 정점에서 BFS/DFS |
| 위상 정렬 | 엣지 방향을 존중하는 선형 순서 |

---

**다음**: [집합과 맵](./10_Sets_and_Maps.md) -- 수학적 집합 연산과 맵 추상화를 탐구합니다.
