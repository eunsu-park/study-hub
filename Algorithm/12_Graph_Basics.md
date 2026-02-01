# 그래프 기초 (Graph Basics)

## 개요

그래프는 정점(vertex)과 간선(edge)으로 이루어진 자료구조로, 네트워크, 관계, 경로 등을 표현하는 데 사용됩니다. 이 레슨에서는 그래프의 기본 개념과 DFS, BFS 탐색을 학습합니다.

---

## 목차

1. [그래프 기본 개념](#1-그래프-기본-개념)
2. [그래프 표현](#2-그래프-표현)
3. [깊이 우선 탐색 (DFS)](#3-깊이-우선-탐색-dfs)
4. [너비 우선 탐색 (BFS)](#4-너비-우선-탐색-bfs)
5. [DFS vs BFS](#5-dfs-vs-bfs)
6. [그래프 응용](#6-그래프-응용)
7. [연습 문제](#7-연습-문제)

---

## 1. 그래프 기본 개념

### 그래프 용어

```
그래프 G = (V, E)
- V: 정점(Vertex) 집합
- E: 간선(Edge) 집합

예시:
    (1)───(2)
     │   / │
     │  /  │
    (3)───(4)

V = {1, 2, 3, 4}
E = {(1,2), (1,3), (2,3), (2,4), (3,4)}
```

### 그래프 종류

```
1. 방향/무방향 그래프

   무방향 그래프:        방향 그래프:
   (1)───(2)            (1)───→(2)
                         ↑       │
                         │       ↓
                        (4)←──(3)

2. 가중치 그래프

    (1)──5──(2)
     │     / │
     3   2   4
     │ /     │
    (3)──1──(4)

3. 연결/비연결 그래프

   연결:               비연결:
   (1)─(2)            (1)─(2)  (4)
    │ / │                 │
   (3)─(4)            (3)─┘
```

### 그래프 특성

```
- 차수(Degree): 정점에 연결된 간선 수
  - 무방향: degree(v)
  - 방향: in-degree(v), out-degree(v)

- 경로(Path): 정점들의 연속
- 사이클(Cycle): 시작과 끝이 같은 경로
- 연결 요소(Connected Component): 연결된 정점들의 집합
```

---

## 2. 그래프 표현

### 2.1 인접 행렬 (Adjacency Matrix)

```
무방향 그래프:
    (0)───(1)
     │   / │
     │  /  │
    (2)───(3)

인접 행렬:
       0  1  2  3
    0 [0, 1, 1, 0]
    1 [1, 0, 1, 1]
    2 [1, 1, 0, 1]
    3 [0, 1, 1, 0]

특징:
- 공간: O(V²)
- 간선 존재 확인: O(1)
- 인접 정점 순회: O(V)
- 밀집 그래프에 적합
```

```c
// C
#define MAX_V 100

int graph[MAX_V][MAX_V];
int V;  // 정점 수

void addEdge(int u, int v) {
    graph[u][v] = 1;
    graph[v][u] = 1;  // 무방향
}

int hasEdge(int u, int v) {
    return graph[u][v];
}
```

```cpp
// C++
class GraphMatrix {
private:
    vector<vector<int>> adj;
    int V;

public:
    GraphMatrix(int v) : V(v), adj(v, vector<int>(v, 0)) {}

    void addEdge(int u, int v) {
        adj[u][v] = 1;
        adj[v][u] = 1;  // 무방향
    }

    bool hasEdge(int u, int v) {
        return adj[u][v] == 1;
    }
};
```

```python
# Python
class GraphMatrix:
    def __init__(self, v):
        self.V = v
        self.adj = [[0] * v for _ in range(v)]

    def add_edge(self, u, v):
        self.adj[u][v] = 1
        self.adj[v][u] = 1  # 무방향

    def has_edge(self, u, v):
        return self.adj[u][v] == 1
```

### 2.2 인접 리스트 (Adjacency List)

```
무방향 그래프:
    (0)───(1)
     │   / │
     │  /  │
    (2)───(3)

인접 리스트:
0: [1, 2]
1: [0, 2, 3]
2: [0, 1, 3]
3: [1, 2]

특징:
- 공간: O(V + E)
- 간선 존재 확인: O(degree)
- 인접 정점 순회: O(degree)
- 희소 그래프에 적합
```

```c
// C - 연결 리스트 방식
#define MAX_V 100

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node* adj[MAX_V];
int V;

void addEdge(int u, int v) {
    // u -> v
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = adj[u];
    adj[u] = newNode;

    // v -> u (무방향)
    newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = u;
    newNode->next = adj[v];
    adj[v] = newNode;
}
```

```cpp
// C++
class GraphList {
private:
    int V;
    vector<vector<int>> adj;

public:
    GraphList(int v) : V(v), adj(v) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);  // 무방향
    }

    const vector<int>& neighbors(int u) {
        return adj[u];
    }
};
```

```python
# Python
class GraphList:
    def __init__(self, v):
        self.V = v
        self.adj = [[] for _ in range(v)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)  # 무방향

    def neighbors(self, u):
        return self.adj[u]

# 또는 defaultdict 사용
from collections import defaultdict

graph = defaultdict(list)
graph[0].append(1)
graph[1].append(0)
```

### 표현 방식 비교

```
┌────────────────┬───────────────┬───────────────┐
│                │ 인접 행렬     │ 인접 리스트   │
├────────────────┼───────────────┼───────────────┤
│ 공간           │ O(V²)         │ O(V + E)      │
│ 간선 확인      │ O(1)          │ O(degree)     │
│ 인접 정점 순회 │ O(V)          │ O(degree)     │
│ 간선 추가      │ O(1)          │ O(1)          │
│ 적합한 경우    │ 밀집 그래프   │ 희소 그래프   │
└────────────────┴───────────────┴───────────────┘
```

---

## 3. 깊이 우선 탐색 (DFS)

### 개념

```
DFS: 한 방향으로 깊이 들어가다가 막히면 되돌아옴 (백트래킹)
→ 스택 또는 재귀로 구현

       (0)
      / | \
    (1)(2)(3)
    /     / \
  (4)   (5)(6)

DFS 순서 (0에서 시작):
0 → 1 → 4 → (백트래킹) → 2 → (백트래킹) → 3 → 5 → (백트래킹) → 6

방문 순서: 0, 1, 4, 2, 3, 5, 6
```

### 재귀 구현

```c
// C
#define MAX_V 100

int adj[MAX_V][MAX_V];
int visited[MAX_V];
int V;

void dfs(int v) {
    visited[v] = 1;
    printf("%d ", v);

    for (int i = 0; i < V; i++) {
        if (adj[v][i] && !visited[i]) {
            dfs(i);
        }
    }
}
```

```cpp
// C++
class Graph {
private:
    int V;
    vector<vector<int>> adj;

public:
    Graph(int v) : V(v), adj(v) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void dfs(int start) {
        vector<bool> visited(V, false);
        dfsUtil(start, visited);
    }

private:
    void dfsUtil(int v, vector<bool>& visited) {
        visited[v] = true;
        cout << v << " ";

        for (int neighbor : adj[v]) {
            if (!visited[neighbor]) {
                dfsUtil(neighbor, visited);
            }
        }
    }
};
```

```python
# Python
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

    return visited
```

### 스택 구현 (반복문)

```cpp
// C++
void dfsIterative(int start) {
    vector<bool> visited(V, false);
    stack<int> st;

    st.push(start);

    while (!st.empty()) {
        int v = st.top();
        st.pop();

        if (visited[v]) continue;

        visited[v] = true;
        cout << v << " ";

        // 역순으로 push하면 순서 유지
        for (auto it = adj[v].rbegin(); it != adj[v].rend(); it++) {
            if (!visited[*it]) {
                st.push(*it);
            }
        }
    }
}
```

```python
# Python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]

    while stack:
        v = stack.pop()

        if v in visited:
            continue

        visited.add(v)
        print(v, end=' ')

        # 역순으로 추가하면 순서 유지
        for neighbor in reversed(graph[v]):
            if neighbor not in visited:
                stack.append(neighbor)

    return visited
```

---

## 4. 너비 우선 탐색 (BFS)

### 개념

```
BFS: 현재 정점의 모든 인접 정점을 먼저 방문
→ 큐로 구현

       (0)
      / | \
    (1)(2)(3)
    /     / \
  (4)   (5)(6)

BFS 순서 (0에서 시작):
레벨 0: 0
레벨 1: 1, 2, 3
레벨 2: 4, 5, 6

방문 순서: 0, 1, 2, 3, 4, 5, 6
```

### 구현

```c
// C
#define MAX_V 100

int adj[MAX_V][MAX_V];
int visited[MAX_V];
int V;

void bfs(int start) {
    int queue[MAX_V];
    int front = 0, rear = 0;

    visited[start] = 1;
    queue[rear++] = start;

    while (front < rear) {
        int v = queue[front++];
        printf("%d ", v);

        for (int i = 0; i < V; i++) {
            if (adj[v][i] && !visited[i]) {
                visited[i] = 1;
                queue[rear++] = i;
            }
        }
    }
}
```

```cpp
// C++
void bfs(int start) {
    vector<bool> visited(V, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int v = q.front();
        q.pop();

        cout << v << " ";

        for (int neighbor : adj[v]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

```python
# Python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        v = queue.popleft()
        print(v, end=' ')

        for neighbor in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited
```

### BFS로 최단 거리

```cpp
// C++ - 가중치 없는 그래프에서 최단 거리
vector<int> shortestPath(int start) {
    vector<int> dist(V, -1);
    queue<int> q;

    dist[start] = 0;
    q.push(start);

    while (!q.empty()) {
        int v = q.front();
        q.pop();

        for (int neighbor : adj[v]) {
            if (dist[neighbor] == -1) {
                dist[neighbor] = dist[v] + 1;
                q.push(neighbor);
            }
        }
    }

    return dist;
}
```

```python
# Python
def shortest_path(graph, start):
    dist = {start: 0}
    queue = deque([start])

    while queue:
        v = queue.popleft()

        for neighbor in graph[v]:
            if neighbor not in dist:
                dist[neighbor] = dist[v] + 1
                queue.append(neighbor)

    return dist
```

---

## 5. DFS vs BFS

### 비교표

```
┌────────────────┬─────────────────────┬─────────────────────┐
│                │ DFS                 │ BFS                 │
├────────────────┼─────────────────────┼─────────────────────┤
│ 자료구조       │ 스택/재귀           │ 큐                  │
│ 메모리         │ O(h) - 깊이         │ O(w) - 너비         │
│ 완전 탐색      │ 가능                │ 가능                │
│ 최단 경로      │ ✗                   │ ✓ (가중치 없음)     │
│ 사이클 탐지    │ ✓                   │ ✓                   │
│ 경로 존재 확인 │ ✓                   │ ✓                   │
└────────────────┴─────────────────────┴─────────────────────┘
```

### 언제 무엇을 사용?

```
DFS 사용:
- 모든 노드를 방문해야 할 때
- 경로 특징을 저장해야 할 때
- 백트래킹 문제
- 사이클 탐지
- 위상 정렬

BFS 사용:
- 최단 경로/최소 비용 (가중치 없음)
- 레벨 순회
- 가까운 것부터 탐색
```

### 시각적 비교

```
       (1)
      / | \
    (2)(3)(4)
    / \   / \
  (5)(6)(7)(8)

DFS (스택):
방문 순서: 1 → 2 → 5 → 6 → 3 → 4 → 7 → 8
깊이 우선: 한 갈래를 끝까지 탐색

BFS (큐):
방문 순서: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
너비 우선: 같은 레벨을 먼저 탐색
```

---

## 6. 그래프 응용

### 6.1 연결 요소 찾기

```cpp
// C++
int countConnectedComponents() {
    vector<bool> visited(V, false);
    int count = 0;

    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            dfs(i, visited);  // 또는 bfs
            count++;
        }
    }

    return count;
}
```

```python
# Python
def count_connected_components(graph, n):
    visited = set()
    count = 0

    for i in range(n):
        if i not in visited:
            dfs_recursive(graph, i, visited)
            count += 1

    return count
```

### 6.2 사이클 탐지 (무방향 그래프)

```cpp
// C++
bool hasCycleDFS(int v, int parent, vector<bool>& visited) {
    visited[v] = true;

    for (int neighbor : adj[v]) {
        if (!visited[neighbor]) {
            if (hasCycleDFS(neighbor, v, visited)) {
                return true;
            }
        } else if (neighbor != parent) {
            // 방문했는데 부모가 아님 = 사이클
            return true;
        }
    }

    return false;
}

bool hasCycle() {
    vector<bool> visited(V, false);

    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            if (hasCycleDFS(i, -1, visited)) {
                return true;
            }
        }
    }

    return false;
}
```

```python
# Python
def has_cycle(graph, n):
    visited = set()

    def dfs(v, parent):
        visited.add(v)

        for neighbor in graph[v]:
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True

        return False

    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True

    return False
```

### 6.3 이분 그래프 판별

```
이분 그래프: 정점을 두 그룹으로 나눌 수 있고,
           같은 그룹 내 정점끼리는 간선이 없음

   (1)─(2)             (1)   (2)
    │ × │     →         │     │
   (3)─(4)             (3)   (4)

색칠 방법: 인접한 정점은 다른 색
```

```cpp
// C++
bool isBipartite() {
    vector<int> color(V, -1);  // -1: 미방문, 0/1: 색깔

    for (int i = 0; i < V; i++) {
        if (color[i] == -1) {
            queue<int> q;
            q.push(i);
            color[i] = 0;

            while (!q.empty()) {
                int v = q.front();
                q.pop();

                for (int neighbor : adj[v]) {
                    if (color[neighbor] == -1) {
                        color[neighbor] = 1 - color[v];
                        q.push(neighbor);
                    } else if (color[neighbor] == color[v]) {
                        return false;  // 인접한데 같은 색
                    }
                }
            }
        }
    }

    return true;
}
```

```python
# Python
from collections import deque

def is_bipartite(graph, n):
    color = [-1] * n

    for start in range(n):
        if color[start] == -1:
            queue = deque([start])
            color[start] = 0

            while queue:
                v = queue.popleft()

                for neighbor in graph[v]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[v]
                        queue.append(neighbor)
                    elif color[neighbor] == color[v]:
                        return False

    return True
```

### 6.4 2D 그리드 탐색

```
그리드를 그래프로 취급:
- 각 셀 = 정점
- 상하좌우 = 간선

미로 찾기, 섬 개수 세기 등에 활용
```

```cpp
// C++ - 섬 개수 세기
int numIslands(vector<vector<char>>& grid) {
    int m = grid.size();
    int n = grid[0].size();
    int count = 0;

    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};

    function<void(int, int)> dfs = [&](int x, int y) {
        if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] != '1') {
            return;
        }

        grid[x][y] = '0';  // 방문 표시

        for (int i = 0; i < 4; i++) {
            dfs(x + dx[i], y + dy[i]);
        }
    };

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                dfs(i, j);
                count++;
            }
        }
    }

    return count;
}
```

```python
# Python
def num_islands(grid):
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(x, y):
        if x < 0 or x >= m or y < 0 or y >= n or grid[x][y] != '1':
            return

        grid[x][y] = '0'  # 방문 표시

        dfs(x + 1, y)
        dfs(x - 1, y)
        dfs(x, y + 1)
        dfs(x, y - 1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1

    return count
```

---

## 7. 연습 문제

### 문제 1: 경로 존재 확인

두 정점 사이에 경로가 존재하는지 확인하세요.

<details>
<summary>정답 코드</summary>

```python
def has_path(graph, start, end):
    visited = set()
    stack = [start]

    while stack:
        v = stack.pop()

        if v == end:
            return True

        if v in visited:
            continue

        visited.add(v)

        for neighbor in graph[v]:
            if neighbor not in visited:
                stack.append(neighbor)

    return False
```

</details>

### 문제 2: 모든 경로 찾기

두 정점 사이의 모든 경로를 찾으세요.

<details>
<summary>정답 코드</summary>

```python
def find_all_paths(graph, start, end, path=None):
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return [path]

    paths = []

    for neighbor in graph[start]:
        if neighbor not in path:  # 사이클 방지
            new_paths = find_all_paths(graph, neighbor, end, path)
            paths.extend(new_paths)

    return paths
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [DFS와 BFS](https://www.acmicpc.net/problem/1260) | 백준 | 기본 탐색 |
| ⭐ | [연결 요소의 개수](https://www.acmicpc.net/problem/11724) | 백준 | 연결 요소 |
| ⭐⭐ | [Number of Islands](https://leetcode.com/problems/number-of-islands/) | LeetCode | 그리드 DFS |
| ⭐⭐ | [미로 탐색](https://www.acmicpc.net/problem/2178) | 백준 | BFS 최단거리 |
| ⭐⭐ | [Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/) | LeetCode | 이분 그래프 |
| ⭐⭐⭐ | [단지번호붙이기](https://www.acmicpc.net/problem/2667) | 백준 | 그리드 탐색 |

---

## 템플릿 정리

### DFS 템플릿

```python
# 재귀
def dfs(graph, v, visited):
    visited.add(v)
    for neighbor in graph[v]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# 반복
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            for neighbor in graph[v]:
                stack.append(neighbor)
```

### BFS 템플릿

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for neighbor in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 2D 그리드 탐색

```python
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n
```

---

## 다음 단계

- [13_Topological_Sort.md](./13_Topological_Sort.md) - 위상 정렬

---

## 참고 자료

- [Graph Visualization](https://visualgo.net/en/dfsbfs)
- [BFS/DFS Tutorial](https://www.geeksforgeeks.org/difference-between-bfs-and-dfs/)
- Introduction to Algorithms (CLRS) - Chapter 22
