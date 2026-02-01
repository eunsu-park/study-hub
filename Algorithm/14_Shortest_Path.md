# 최단 경로 (Shortest Path)

## 개요

가중치가 있는 그래프에서 두 정점 사이의 최단 경로를 찾는 알고리즘을 학습합니다. Dijkstra, Bellman-Ford, Floyd-Warshall 알고리즘을 다룹니다.

---

## 목차

1. [최단 경로 알고리즘 비교](#1-최단-경로-알고리즘-비교)
2. [다익스트라 (Dijkstra)](#2-다익스트라-dijkstra)
3. [벨만-포드 (Bellman-Ford)](#3-벨만-포드-bellman-ford)
4. [플로이드-워셜 (Floyd-Warshall)](#4-플로이드-워셜-floyd-warshall)
5. [0-1 BFS](#5-0-1-bfs)
6. [연습 문제](#6-연습-문제)

---

## 1. 최단 경로 알고리즘 비교

```
┌─────────────────┬─────────────┬───────────────┬─────────────────┐
│ 알고리즘        │ 시간 복잡도 │ 음수 가중치   │ 용도            │
├─────────────────┼─────────────┼───────────────┼─────────────────┤
│ BFS             │ O(V+E)      │ ✗             │ 가중치 없음     │
│ Dijkstra        │ O(E log V)  │ ✗             │ 단일 출발점     │
│ Bellman-Ford    │ O(VE)       │ ✓             │ 단일 출발점     │
│ Floyd-Warshall  │ O(V³)       │ ✓             │ 모든 쌍         │
│ 0-1 BFS         │ O(V+E)      │ 0,1만         │ 0/1 가중치      │
└─────────────────┴─────────────┴───────────────┴─────────────────┘
```

---

## 2. 다익스트라 (Dijkstra)

### 개념

```
단일 출발점에서 모든 정점까지의 최단 거리
조건: 음수 가중치 없음

원리:
1. 시작점 거리 = 0, 나머지 = ∞
2. 방문하지 않은 정점 중 최단 거리 정점 선택
3. 해당 정점을 통해 인접 정점 거리 갱신
4. 모든 정점 방문할 때까지 반복
```

### 동작 예시

```
        (B)
       / 1 \
      4     2
     /       \
   (A)       (D)
     \       /
      2     1
       \ 3 /
        (C)

시작: A

단계  방문    dist[A] dist[B] dist[C] dist[D]
----  -----   ------  ------  ------  ------
0     {}      0       ∞       ∞       ∞
1     {A}     0       4       2       ∞       ← A 방문, B,C 갱신
2     {A,C}   0       4       2       5       ← C 방문, D 갱신 (2+3)
3     {A,C,B} 0       4       2       5       ← B 방문
4     {all}   0       4       2       5       ← D 방문

결과: A→B: 4, A→C: 2, A→D: 5
```

### 구현 (우선순위 큐)

```c
// C - 인접 리스트 + 배열 (간단한 버전)
#define INF 1000000000
#define MAX_V 10001

typedef struct {
    int to, weight;
} Edge;

Edge adj[MAX_V][MAX_V];
int adjSize[MAX_V];
int dist[MAX_V];
int visited[MAX_V];
int V, E;

void dijkstra(int start) {
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[start] = 0;

    for (int i = 0; i < V; i++) {
        // 최소 거리 정점 찾기
        int u = -1;
        int minDist = INF;
        for (int j = 0; j < V; j++) {
            if (!visited[j] && dist[j] < minDist) {
                minDist = dist[j];
                u = j;
            }
        }

        if (u == -1) break;
        visited[u] = 1;

        // 인접 정점 갱신
        for (int j = 0; j < adjSize[u]; j++) {
            int v = adj[u][j].to;
            int w = adj[u][j].weight;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
}
```

```cpp
// C++ - 우선순위 큐 사용 (효율적)
#include <queue>
#include <vector>
using namespace std;

typedef pair<int, int> pii;  // {거리, 정점}

vector<int> dijkstra(const vector<vector<pii>>& adj, int start) {
    int V = adj.size();
    vector<int> dist(V, INT_MAX);
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        // 이미 처리된 정점이면 스킵
        if (d > dist[u]) continue;

        for (auto [v, weight] : adj[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

// 사용 예
// adj[u].push_back({v, weight});  // u → v, 가중치 weight
// auto dist = dijkstra(adj, 0);
```

```python
# Python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]  # (거리, 정점)

    while pq:
        d, u = heapq.heappop(pq)

        # 이미 처리된 정점이면 스킵
        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist

# 사용 예
# graph = defaultdict(list)
# graph[0].append((1, 4))  # 0 → 1, 가중치 4
# graph[0].append((2, 2))  # 0 → 2, 가중치 2
# dist = dijkstra(graph, 0)
```

### 경로 복원

```cpp
// C++
pair<vector<int>, vector<int>> dijkstraWithPath(
    const vector<vector<pii>>& adj, int start) {

    int V = adj.size();
    vector<int> dist(V, INT_MAX);
    vector<int> parent(V, -1);
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        for (auto [v, weight] : adj[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;  // 경로 저장
                pq.push({dist[v], v});
            }
        }
    }

    return {dist, parent};
}

// 경로 출력
vector<int> getPath(const vector<int>& parent, int end) {
    vector<int> path;
    for (int v = end; v != -1; v = parent[v]) {
        path.push_back(v);
    }
    reverse(path.begin(), path.end());
    return path;
}
```

```python
# Python
def dijkstra_with_path(graph, start):
    dist = {node: float('inf') for node in graph}
    parent = {node: None for node in graph}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))

    return dist, parent

def get_path(parent, end):
    path = []
    v = end
    while v is not None:
        path.append(v)
        v = parent[v]
    return path[::-1]
```

---

## 3. 벨만-포드 (Bellman-Ford)

### 개념

```
단일 출발점에서 모든 정점까지의 최단 거리
특징: 음수 가중치 허용, 음수 사이클 탐지 가능

원리:
1. 시작점 거리 = 0, 나머지 = ∞
2. 모든 간선에 대해 거리 갱신 (V-1회 반복)
3. V번째 반복에서 갱신되면 음수 사이클 존재
```

### 동작 예시

```
        (B)
       / 1 \
      4     2
     /       \
   (A)       (D)
     \       /
      2     -5    ← 음수 가중치
       \ 3 /
        (C)

간선: (A,B,4), (A,C,2), (B,D,2), (C,B,1), (C,D,3)

반복 1: dist = [0, 4, 2, 5]
반복 2: dist = [0, 3, 2, 5]  ← C→B로 B 갱신 (2+1=3)
반복 3: dist = [0, 3, 2, 5]  ← 변화 없음

결과: A→B: 3, A→C: 2, A→D: 5
```

### 구현

```c
// C
#define INF 1000000000

typedef struct {
    int from, to, weight;
} Edge;

Edge edges[20001];
int dist[501];
int V, E;

int bellmanFord(int start) {
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[start] = 0;

    // V-1번 반복
    for (int i = 0; i < V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = edges[j].from;
            int v = edges[j].to;
            int w = edges[j].weight;

            if (dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }

    // 음수 사이클 확인
    for (int j = 0; j < E; j++) {
        int u = edges[j].from;
        int v = edges[j].to;
        int w = edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            return -1;  // 음수 사이클 존재
        }
    }

    return 0;
}
```

```cpp
// C++
struct Edge {
    int from, to, weight;
};

vector<int> bellmanFord(int V, const vector<Edge>& edges, int start) {
    vector<int> dist(V, INT_MAX);
    dist[start] = 0;

    // V-1번 반복
    for (int i = 0; i < V - 1; i++) {
        for (const auto& e : edges) {
            if (dist[e.from] != INT_MAX &&
                dist[e.from] + e.weight < dist[e.to]) {
                dist[e.to] = dist[e.from] + e.weight;
            }
        }
    }

    // 음수 사이클 확인
    for (const auto& e : edges) {
        if (dist[e.from] != INT_MAX &&
            dist[e.from] + e.weight < dist[e.to]) {
            return {};  // 음수 사이클
        }
    }

    return dist;
}
```

```python
# Python
def bellman_ford(V, edges, start):
    dist = [float('inf')] * V
    dist[start] = 0

    # V-1번 반복
    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # 음수 사이클 확인
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # 음수 사이클

    return dist
```

---

## 4. 플로이드-워셜 (Floyd-Warshall)

### 개념

```
모든 정점 쌍 사이의 최단 거리
특징: 음수 가중치 허용, DP 기반

원리:
k를 경유하는 경로 vs 직접 경로
dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

모든 k에 대해 반복
```

### 동작 예시

```
     1     2
(0) ─→ (1) ─→ (2)
 └──────3──────→

초기:
      0     1     2
  0 [ 0,    1,    3]
  1 [INF,   0,    2]
  2 [INF,  INF,   0]

k=1 (1을 경유):
  dist[0][2] = min(3, dist[0][1]+dist[1][2])
             = min(3, 1+2) = 3

결과: 0→2 최단거리 = 3
```

### 구현

```c
// C
#define INF 1000000000
#define MAX_V 500

int dist[MAX_V][MAX_V];
int V;

void floydWarshall() {
    // k를 경유점으로
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
    }
}

// 초기화
void initGraph() {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) dist[i][j] = 0;
            else dist[i][j] = INF;
        }
    }
}
```

```cpp
// C++
void floydWarshall(vector<vector<int>>& dist) {
    int V = dist.size();

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}

// 초기화
vector<vector<int>> initDist(int V) {
    vector<vector<int>> dist(V, vector<int>(V, INT_MAX));
    for (int i = 0; i < V; i++) {
        dist[i][i] = 0;
    }
    return dist;
}
```

```python
# Python
def floyd_warshall(V, edges):
    INF = float('inf')

    # 초기화
    dist = [[INF] * V for _ in range(V)]
    for i in range(V):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = w

    # 플로이드-워셜
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
```

### 경로 복원

```cpp
// C++
void floydWarshallWithPath(vector<vector<int>>& dist,
                           vector<vector<int>>& next) {
    int V = dist.size();

    // 초기화
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] != INT_MAX && i != j) {
                next[i][j] = j;
            } else {
                next[i][j] = -1;
            }
        }
    }

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next[i][j] = next[i][k];
                }
            }
        }
    }
}

vector<int> getPath(const vector<vector<int>>& next, int from, int to) {
    if (next[from][to] == -1) return {};

    vector<int> path = {from};
    while (from != to) {
        from = next[from][to];
        path.push_back(from);
    }
    return path;
}
```

---

## 5. 0-1 BFS

### 개념

```
간선 가중치가 0 또는 1인 그래프에서 최단 거리
→ 덱(Deque) 사용하여 O(V+E)에 해결

원리:
- 가중치 0인 간선: 덱 앞에 추가
- 가중치 1인 간선: 덱 뒤에 추가

효과: 항상 최소 거리 정점이 앞에 위치
```

### 구현

```cpp
// C++
vector<int> zeroOneBFS(const vector<vector<pii>>& adj, int start) {
    int V = adj.size();
    vector<int> dist(V, INT_MAX);
    deque<int> dq;

    dist[start] = 0;
    dq.push_front(start);

    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();

        for (auto [v, weight] : adj[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;

                if (weight == 0) {
                    dq.push_front(v);
                } else {
                    dq.push_back(v);
                }
            }
        }
    }

    return dist;
}
```

```python
# Python
from collections import deque

def zero_one_bfs(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    dq = deque([start])

    while dq:
        u = dq.popleft()

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight

                if weight == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)

    return dist
```

---

## 6. 연습 문제

### 문제 1: 특정 거리의 도시 찾기

시작 도시에서 정확히 K 거리인 모든 도시를 찾으세요.

<details>
<summary>정답 코드</summary>

```python
import heapq

def cities_at_distance_k(n, edges, start, k):
    graph = [[] for _ in range(n + 1)]
    for a, b in edges:
        graph[a].append(b)  # 단방향

    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in graph[u]:
            if dist[u] + 1 < dist[v]:
                dist[v] = dist[u] + 1
                heapq.heappush(pq, (dist[v], v))

    result = [i for i in range(1, n + 1) if dist[i] == k]
    return sorted(result) if result else [-1]
```

</details>

### 문제 2: 음수 사이클 탐지

음수 사이클이 존재하는지 확인하세요.

<details>
<summary>정답 코드</summary>

```python
def has_negative_cycle(V, edges):
    dist = [0] * V  # 모든 정점에서 시작 가능

    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return True

    return False
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 알고리즘 |
|--------|------|--------|----------|
| ⭐⭐ | [최단경로](https://www.acmicpc.net/problem/1753) | 백준 | Dijkstra |
| ⭐⭐ | [Network Delay Time](https://leetcode.com/problems/network-delay-time/) | LeetCode | Dijkstra |
| ⭐⭐⭐ | [타임머신](https://www.acmicpc.net/problem/11657) | 백준 | Bellman-Ford |
| ⭐⭐⭐ | [플로이드](https://www.acmicpc.net/problem/11404) | 백준 | Floyd-Warshall |
| ⭐⭐⭐ | [경로 찾기](https://www.acmicpc.net/problem/11403) | 백준 | Floyd-Warshall |
| ⭐⭐⭐⭐ | [Cheapest Flights](https://leetcode.com/problems/cheapest-flights-within-k-stops/) | LeetCode | 변형 Dijkstra |

---

## 알고리즘 선택 가이드

```
질문 1: 가중치가 있는가?
├── No → BFS (O(V+E))
└── Yes ↓

질문 2: 음수 가중치가 있는가?
├── No → Dijkstra (O(E log V))
└── Yes ↓

질문 3: 음수 사이클 탐지가 필요한가?
├── Yes → Bellman-Ford (O(VE))
└── No ↓

질문 4: 모든 쌍 최단거리가 필요한가?
├── Yes → Floyd-Warshall (O(V³))
└── No → Bellman-Ford (O(VE))

추가: 가중치가 0, 1만 있으면 → 0-1 BFS (O(V+E))
```

---

## 다음 단계

- [15_Minimum_Spanning_Tree.md](./15_Minimum_Spanning_Tree.md) - Kruskal, Prim, Union-Find

---

## 참고 자료

- [Dijkstra Visualization](https://visualgo.net/en/sssp)
- [Shortest Path Problems](https://cp-algorithms.com/graph/shortest_paths.html)
- Introduction to Algorithms (CLRS) - Chapter 24, 25
