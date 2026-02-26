# 25. 네트워크 플로우 (Network Flow)

## 학습 목표
- 최대 유량 문제의 개념 이해
- Ford-Fulkerson 알고리즘 구현
- Edmonds-Karp 알고리즘 구현
- 이분 매칭 문제 해결
- 최소 컷 정리 활용

## 1. 네트워크 플로우란?

### 정의

네트워크 플로우를 파이프 시스템을 통해 흐르는 물이라고 생각해 보세요. 각 파이프에는 최대 용량이 있고, 합류점에서 물은 보존되어야 하며(들어오는 양 = 나가는 양), 목표는 소스(source)에서 싱크(sink)로의 처리량을 최대화하는 것입니다.

**유량 네트워크**는 다음으로 구성됩니다:
- **방향 그래프** G = (V, E)
- **용량 함수** c(u, v): 간선이 수용할 수 있는 최대 유량
- **소스(source)** s: 유량의 시작점
- **싱크(sink)** t: 유량의 도착점

```
          3
    ┌───→ B ───→┐
    │           │ 2
  S │     4     ↓     T
    │ 2 → C ─→ D │ 3
    └───────────→┘

S: 소스, T: 싱크
숫자: 간선 용량
```

### 유량의 조건

1. **용량 제한**: 0 ≤ f(u,v) ≤ c(u,v) *(파이프 지름이 허용하는 것 이상의 물을 흘릴 수 없음)*
2. **유량 보존**: 소스/싱크 제외, 들어오는 유량 = 나가는 유량 *(합류점에서 물이 축적되지 않음 -- 들어온 모든 물방울은 반드시 나가야 함)*
3. **반대칭성**: f(u,v) = -f(v,u) *(u에서 v로 3단위가 흐르면, v에서 u로는 -3으로 표현되어 잔여 그래프 연산을 가능하게 함)*

### 최대 유량 문제

소스에서 싱크로 보낼 수 있는 **최대 유량** 찾기

```
┌─────────────────────────────────────────────────┐
│              네트워크 플로우 활용                 │
├─────────────────────────────────────────────────┤
│  • 이분 매칭 (작업 배정, 팀 구성)                │
│  • 최대/최소 컷 문제                             │
│  • 프로젝트 선택 문제                           │
│  • 이미지 분할 (컴퓨터 비전)                     │
│  • 교통/통신 네트워크 최적화                     │
└─────────────────────────────────────────────────┘
```

---

## 2. Ford-Fulkerson 알고리즘

### 핵심 아이디어

1. **증가 경로(Augmenting Path)** 찾기: 소스→싱크로 추가 유량을 보낼 수 있는 경로
2. 경로를 따라 유량 증가
3. 더 이상 증가 경로가 없을 때까지 반복

### 잔여 그래프 (Residual Graph)

```
원본 간선: u →(c)→ v, 현재 유량 f

잔여 그래프:
  u →(c-f)→ v  (정방향: 남은 용량)
  u ←(f)← v    (역방향: 취소 가능한 유량)
```

### 동작 예시

```
초기:                    Step 1: S→A→B→T 경로
      4                       (유량 2 추가)
  ┌──→A──→┐
  │   ↓   │2
S │3  1   ↓ T            잔여 그래프 업데이트
  │   ↓   │
  └──→B──→┘
      3

Step 2: S→B→T 경로       최종 최대 유량: 5
  (유량 3 추가)
```

### 구현 (DFS 기반)

```python
from collections import defaultdict

class MaxFlow:
    def __init__(self, n):
        self.n = n
        self.capacity = defaultdict(lambda: defaultdict(int))
        self.flow = defaultdict(lambda: defaultdict(int))

    def add_edge(self, u, v, cap):
        self.capacity[u][v] += cap
        # 명시적 역방향 간선 불필요: self.flow[v][u]는 0에서 시작하므로
        # 잔여 용량(cap - flow) = 정방향에서 cap이 되고,
        # 아래 flow[neighbor][source] -= result가 역방향 부기를 처리함

    def dfs(self, source, sink, visited, min_cap):
        if source == sink:
            return min_cap  # 싱크 도달 -- 이 경로의 병목 용량을 반환

        visited.add(source)

        for neighbor in self.capacity[source]:
            # 잔여 용량 = 원래 용량 - 이미 흐르고 있는 유량
            residual = self.capacity[source][neighbor] - self.flow[source][neighbor]

            if neighbor not in visited and residual > 0:
                # 이 DFS 경로를 따라 지금까지 본 가장 좁은 병목을 추적
                new_cap = min(min_cap, residual)
                result = self.dfs(neighbor, sink, visited, new_cap)

                if result > 0:
                    self.flow[source][neighbor] += result
                    # 역방향 간선: flow[neighbor][source]를 감소시키는 것은
                    # 용량을 "크레딧"으로 돌려주는 것으로, 미래의 경로가
                    # 차선의 경로로 보내진 유량을 재라우팅할 수 있게 해줌
                    self.flow[neighbor][source] -= result
                    return result

        return 0

    def max_flow(self, source, sink):
        total_flow = 0

        # 증가 경로가 없을 때까지 계속 찾기 --
        # 각 반복은 최소 1단위의 유량을 보내므로 루프가 종료됨
        while True:
            visited = set()
            path_flow = self.dfs(source, sink, visited, float('inf'))

            if path_flow == 0:
                break  # 증가 경로 없음 -- 잔여 그래프가 소진됨

            total_flow += path_flow

        return total_flow

# 사용 예시
mf = MaxFlow(4)
# 0: S, 1: A, 2: B, 3: T
mf.add_edge(0, 1, 4)  # S→A
mf.add_edge(0, 2, 3)  # S→B
mf.add_edge(1, 2, 1)  # A→B
mf.add_edge(1, 3, 2)  # A→T
mf.add_edge(2, 3, 3)  # B→T

print("최대 유량:", mf.max_flow(0, 3))  # 5
```

### 시간 복잡도

- **O(E × max_flow)**: DFS 기반 (정수 용량일 때)
- 비합리적 용량이면 수렴하지 않을 수 있음

---

## 3. Edmonds-Karp 알고리즘

### 핵심 아이디어

Ford-Fulkerson + **BFS**로 최단 증가 경로 선택

→ 항상 **O(VE²)** 보장

### 구현

```python
from collections import defaultdict, deque

class EdmondsKarp:
    def __init__(self, n):
        self.n = n
        self.capacity = [[0] * n for _ in range(n)]
        self.graph = defaultdict(list)

    def add_edge(self, u, v, cap):
        self.capacity[u][v] += cap
        self.graph[u].append(v)
        self.graph[v].append(u)  # 역방향도 추가 -- 잔여 그래프 탐색에 필요

    def bfs(self, source, sink, parent):
        # BFS는 최단 증가 경로를 찾음 -- Edmonds-Karp는 항상 간선 수가 가장 적은
        # 경로를 선택하여 O(VE²)를 보장하며, DFS 기반 Ford-Fulkerson의
        # 병리적인 O(E × max_flow) 동작을 방지함
        visited = [False] * self.n
        visited[source] = True
        queue = deque([source])

        while queue:
            node = queue.popleft()

            for neighbor in self.graph[node]:
                # 잔여 용량이 남아 있는 간선만 탐색
                if not visited[neighbor] and self.capacity[node][neighbor] > 0:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    if neighbor == sink:
                        return True  # 경로 발견 -- BFS 조기 종료
                    queue.append(neighbor)

        return False  # 싱크까지의 경로 없음 -- 최대 유량 도달

    def max_flow(self, source, sink):
        parent = [-1] * self.n
        total_flow = 0

        # 각 BFS 호출은 하나의 증가 경로를 찾고 최소 하나의 간선을 포화시키므로,
        # 종료까지 최대 O(VE)번의 증가 경로 반복이 있음
        while self.bfs(source, sink, parent):
            # 경로를 따라 최소 잔여 용량 찾기 (병목 간선)
            path_flow = float('inf')
            node = sink
            while node != source:
                prev = parent[node]
                path_flow = min(path_flow, self.capacity[prev][node])
                node = prev

            # 용량 업데이트: 정방향 간선은 감소, 역방향 간선은 증가.
            # capacity[node][prev]를 증가시키면 미래 경로를 위한 "취소" 옵션을
            # 만들게 됨 -- 이것이 알고리즘이 이전의 차선 라우팅 결정을
            # 되돌릴 수 있게 하는 핵심 메커니즘
            node = sink
            while node != source:
                prev = parent[node]
                self.capacity[prev][node] -= path_flow
                self.capacity[node][prev] += path_flow
                node = prev

            total_flow += path_flow
            parent = [-1] * self.n  # 다음 BFS를 위해 부모 배열 초기화

        return total_flow

# 사용 예시
ek = EdmondsKarp(6)
# 더 복잡한 예제
edges = [
    (0, 1, 16), (0, 2, 13),
    (1, 2, 10), (1, 3, 12),
    (2, 1, 4), (2, 4, 14),
    (3, 2, 9), (3, 5, 20),
    (4, 3, 7), (4, 5, 4)
]
for u, v, cap in edges:
    ek.add_edge(u, v, cap)

print("최대 유량:", ek.max_flow(0, 5))  # 23
```

### C++ 구현

```cpp
#include <bits/stdc++.h>
using namespace std;

class EdmondsKarp {
private:
    int n;
    vector<vector<int>> capacity, adj;

public:
    EdmondsKarp(int n) : n(n), capacity(n, vector<int>(n, 0)), adj(n) {}

    void addEdge(int u, int v, int cap) {
        capacity[u][v] += cap;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int maxFlow(int source, int sink) {
        int totalFlow = 0;

        while (true) {
            vector<int> parent(n, -1);
            queue<int> q;
            q.push(source);
            parent[source] = source;

            while (!q.empty() && parent[sink] == -1) {
                int node = q.front(); q.pop();
                for (int next : adj[node]) {
                    if (parent[next] == -1 && capacity[node][next] > 0) {
                        parent[next] = node;
                        q.push(next);
                    }
                }
            }

            if (parent[sink] == -1) break;

            int pathFlow = INT_MAX;
            for (int v = sink; v != source; v = parent[v]) {
                pathFlow = min(pathFlow, capacity[parent[v]][v]);
            }

            for (int v = sink; v != source; v = parent[v]) {
                capacity[parent[v]][v] -= pathFlow;
                capacity[v][parent[v]] += pathFlow;
            }

            totalFlow += pathFlow;
        }

        return totalFlow;
    }
};
```

---

## 4. 최대 유량 최소 컷 정리

### 정리

**최대 유량 = 최소 컷의 용량**

직관적으로, 컷(cut)은 병목(bottleneck)이라고 생각할 수 있습니다 -- 제거하면 소스에서 싱크로의 모든 경로가 끊어지는 간선 집합입니다. 최소 컷은 가장 좁은 병목이며, 모든 유량은 반드시 이 컷을 통과해야 하므로 그 총 용량을 초과할 수 없습니다. 반대로, 최대 유량 알고리즘은 이 병목을 정확히 포화시킵니다 -- 알고리즘이 종료될 때 잔여 그래프 경계를 가로지르는 포화된 간선들이 최소 컷을 형성합니다. 잔여 그래프의 역방향 간선은 이 과정에서 필수적입니다: 차선의 경로로 보내진 유량을 재라우팅할 수 있게 해서, 이전 결정을 효과적으로 "되돌려" 최종 유량 구성이 최소 컷을 최대 용량으로 통과하도록 합니다.

### 컷이란?

그래프를 소스 측(S)과 싱크 측(T)으로 나누는 간선 집합

```
         컷
          │
    ┌─────│─────┐
S ──┤  A  │  B  ├── T
    └─────│─────┘
          │
컷의 용량 = S측→T측 간선 용량의 합
```

### 최소 컷 찾기

```python
def find_min_cut(self, source, sink):
    """
    최대 유량 계산 후 최소 컷을 구성하는 간선들 반환
    """
    # 먼저 최대 유량 계산 -- 최소 컷을 가로지르는 모든 병목 간선을 포화시킴
    max_flow_value = self.max_flow(source, sink)

    # 최대 유량 후, 잔여 그래프에서 소스로부터 BFS 수행.
    # 소스에서 도달 가능한 노드들이 최소 컷의 "S측"을 형성.
    # 모든 증가 경로가 소진되었으므로 싱크는 도달 불가능.
    visited = [False] * self.n
    queue = deque([source])
    visited[source] = True

    while queue:
        node = queue.popleft()
        for neighbor in range(self.n):
            # 잔여 용량이 남아 있는 간선만 탐색
            if not visited[neighbor] and self.capacity[node][neighbor] > 0:
                visited[neighbor] = True
                queue.append(neighbor)

    # 최소 컷 간선은 도달 가능(S)측에서 도달 불가능(T)측으로 교차하는 간선.
    # 원본 그래프에 존재했던 간선을 식별하기 위해 원본 용량(현재 잔여 용량이 아닌)을 확인 --
    # 일부는 완전히 포화되어 잔여가 0이 되었으며 최소 컷 경계를 형성함.
    # original_capacity 추적 없이는 원래 그래프에 없던 간선(용량 0)과
    # 유량에 의해 포화된 간선을 구별할 수 없음.
    min_cut_edges = []
    for u in range(self.n):
        if visited[u]:
            for v in range(self.n):
                if not visited[v] and self.original_capacity[u][v] > 0:
                    min_cut_edges.append((u, v))

    return max_flow_value, min_cut_edges
```

---

## 5. 이분 매칭 (Bipartite Matching)

### 문제

이분 그래프에서 최대 매칭 찾기

```
왼쪽 그룹        오른쪽 그룹
   A ─────────── 1
   B ─────────── 2
   C ─────────── 3

최대 매칭: A-1, B-2, C-3
```

### 유량 네트워크로 변환

```
              1
           ┌──┤
     ┌──A──┤  │
     │     └──┤
S ───┼──B─────┼─── T
     │     ┌──┤
     └──C──┤  │
           └──┤
              1

모든 간선 용량 = 1
최대 유량 = 최대 매칭
```

### 구현

```python
class BipartiteMatching:
    def __init__(self, left_size, right_size):
        self.left = left_size
        self.right = right_size
        self.adj = [[] for _ in range(left_size)]

    def add_edge(self, left_node, right_node):
        self.adj[left_node].append(right_node)

    def dfs(self, node, visited, match_right):
        for right in self.adj[node]:
            if visited[right]:
                continue
            # 이 DFS가 같은 오른쪽 노드를 재방문하는 것을 방지하기 위해 방문 표시 --
            # 그렇지 않으면 증가 경로 탐색에서 무한 재귀가 발생함
            visited[right] = True

            # 오른쪽 노드가 미매칭이거나, 현재 매칭 파트너가 다른 매칭을 찾을 수 있음.
            # 이 재귀 호출은 증가 경로 단계: 기존 매칭 파트너를 다른 오른쪽 노드로
            # "밀어내어" 이 자리를 비우려고 시도함.
            if match_right[right] == -1 or self.dfs(match_right[right], visited, match_right):
                match_right[right] = node  # 이 오른쪽 노드를 현재 왼쪽 노드에 (재)배정
                return True

        return False  # 이 왼쪽 노드에서 증가 경로를 찾지 못함

    def max_matching(self):
        match_right = [-1] * self.right  # match_right[r] = 오른쪽 노드 r에 매칭된 왼쪽 노드
        result = 0

        for left_node in range(self.left):
            # 왼쪽 노드마다 새 visited 배열 -- 이전 반복에 차단되지 않고
            # 모든 대안 증가 경로를 탐색할 수 있도록 보장
            visited = [False] * self.right
            if self.dfs(left_node, visited, match_right):
                result += 1

        return result, match_right

# 사용 예시: 학생-프로젝트 배정
bm = BipartiteMatching(3, 3)
# 학생 0: 프로젝트 0, 1 가능
bm.add_edge(0, 0)
bm.add_edge(0, 1)
# 학생 1: 프로젝트 0, 2 가능
bm.add_edge(1, 0)
bm.add_edge(1, 2)
# 학생 2: 프로젝트 1, 2 가능
bm.add_edge(2, 1)
bm.add_edge(2, 2)

count, matching = bm.max_matching()
print(f"최대 매칭: {count}")
for right, left in enumerate(matching):
    if left != -1:
        print(f"  학생 {left} → 프로젝트 {right}")
```

### 헝가리안 알고리즘 (Kuhn's Algorithm)

위 DFS 기반 알고리즘이 바로 헝가리안 알고리즘의 간소화 버전입니다.

**시간 복잡도**: O(V × E)

### C++ 구현

```cpp
#include <bits/stdc++.h>
using namespace std;

class BipartiteMatching {
private:
    int left_size, right_size;
    vector<vector<int>> adj;
    vector<int> matchRight;
    vector<bool> visited;

    bool dfs(int node) {
        for (int right : adj[node]) {
            if (visited[right]) continue;
            visited[right] = true;

            if (matchRight[right] == -1 || dfs(matchRight[right])) {
                matchRight[right] = node;
                return true;
            }
        }
        return false;
    }

public:
    BipartiteMatching(int l, int r)
        : left_size(l), right_size(r), adj(l), matchRight(r, -1) {}

    void addEdge(int left, int right) {
        adj[left].push_back(right);
    }

    int maxMatching() {
        int result = 0;
        for (int i = 0; i < left_size; i++) {
            visited.assign(right_size, false);
            if (dfs(i)) result++;
        }
        return result;
    }
};
```

---

## 6. 실전 문제 패턴

### 패턴 1: 작업 배정 문제

```python
def job_assignment(workers, jobs, can_do):
    """
    workers: 작업자 수
    jobs: 작업 수
    can_do[i]: 작업자 i가 할 수 있는 작업 목록
    """
    bm = BipartiteMatching(workers, jobs)
    for worker, job_list in enumerate(can_do):
        for job in job_list:
            bm.add_edge(worker, job)

    return bm.max_matching()
```

### 패턴 2: 정점 분리

노드에 용량이 있을 때 노드를 두 개로 분리

```python
def vertex_capacity_flow(n, edges, vertex_cap, source, sink):
    """
    노드 i → i_in (2*i), i_out (2*i + 1)
    i_in → i_out 용량 = vertex_cap[i]
    """
    new_n = 2 * n
    mf = EdmondsKarp(new_n)

    # 노드 내부 간선
    for i in range(n):
        mf.add_edge(2*i, 2*i + 1, vertex_cap[i])

    # 원본 간선: u_out → v_in
    for u, v, cap in edges:
        mf.add_edge(2*u + 1, 2*v, cap)

    return mf.max_flow(2*source + 1, 2*sink)
```

### 패턴 3: 다중 소스/싱크

```python
def multi_source_sink(n, edges, sources, sinks, source_cap, sink_cap):
    """
    가상의 슈퍼 소스, 슈퍼 싱크 추가
    """
    super_source = n
    super_sink = n + 1
    mf = EdmondsKarp(n + 2)

    # 원본 간선
    for u, v, cap in edges:
        mf.add_edge(u, v, cap)

    # 슈퍼 소스 → 각 소스
    for s, cap in zip(sources, source_cap):
        mf.add_edge(super_source, s, cap)

    # 각 싱크 → 슈퍼 싱크
    for t, cap in zip(sinks, sink_cap):
        mf.add_edge(t, super_sink, cap)

    return mf.max_flow(super_source, super_sink)
```

### 패턴 4: 경로 분리 (Edge-disjoint paths)

```python
def count_edge_disjoint_paths(n, edges, source, sink):
    """
    간선을 공유하지 않는 경로의 최대 개수
    = 각 간선 용량 1로 설정 후 최대 유량
    """
    mf = EdmondsKarp(n)
    for u, v in edges:
        mf.add_edge(u, v, 1)
    return mf.max_flow(source, sink)

def count_vertex_disjoint_paths(n, edges, source, sink):
    """
    정점을 공유하지 않는 경로의 최대 개수
    = 정점 분리 + 각 간선 용량 1
    """
    # 정점 분리: 노드 i → 2*i (in), 2*i+1 (out)
    mf = EdmondsKarp(2 * n)

    # 내부 간선 (소스/싱크 제외)
    for i in range(n):
        if i == source or i == sink:
            mf.add_edge(2*i, 2*i + 1, float('inf'))
        else:
            mf.add_edge(2*i, 2*i + 1, 1)

    # 원본 간선
    for u, v in edges:
        mf.add_edge(2*u + 1, 2*v, 1)

    return mf.max_flow(2*source + 1, 2*sink)
```

### 패턴 5: 최소 경로 커버

DAG에서 모든 정점을 커버하는 최소 경로 수

```python
def min_path_cover(n, edges):
    """
    최소 경로 커버 = n - 최대 매칭
    """
    bm = BipartiteMatching(n, n)
    for u, v in edges:
        bm.add_edge(u, v)

    max_match, _ = bm.max_matching()
    return n - max_match
```

---

## 7. Dinic 알고리즘 (고급)

더 빠른 최대 유량 알고리즘: **O(V²E)**

### 핵심 아이디어

1. BFS로 레벨 그래프 구성
2. DFS로 blocking flow 찾기
3. 반복

```python
from collections import deque

class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        # 각 정방향 간선은 역방향 간선의 참조 인덱스를 저장하고 그 반대도 마찬가지.
        # 이 쌍 인덱스 트릭은 유량 업데이트 시 역방향 간선을 O(1)로 찾을 수 있게 하여,
        # 인접 리스트에서 해당 역방향 간선을 검색할 필요를 없앰.
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])  # 용량 0인 역방향 간선

    def bfs(self, source, sink):
        # BFS는 각 노드에 "레벨"(잔여 그래프에서 소스까지의 거리)을 할당.
        # 이 레벨 그래프는 DFS가 싱크 방향으로만 진행하도록 제한하여
        # 사이클을 방지하고 각 차단 유량(blocking flow) 단계가 진전되도록 보장.
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for next_node, cap, _ in self.graph[node]:
                if cap > 0 and self.level[next_node] == -1:
                    self.level[next_node] = self.level[node] + 1
                    queue.append(next_node)

        return self.level[sink] != -1  # False이면 경로 없음 -- 최대 유량 도달

    def dfs(self, node, sink, flow):
        if node == sink:
            return flow  # 싱크 도달 -- 이 경로의 병목 용량을 반환

        while self.iter[node] < len(self.graph[node]):
            edge = self.graph[node][self.iter[node]]
            next_node, cap, rev = edge

            # 정확히 한 레벨 더 깊은 간선만 따라감 (레벨 그래프 제약)
            if cap > 0 and self.level[next_node] == self.level[node] + 1:
                d = self.dfs(next_node, sink, min(flow, cap))
                if d > 0:
                    edge[1] -= d                        # 정방향 간선 용량 감소
                    self.graph[next_node][rev][1] += d  # 역방향 간선 용량 증가 (잔여)
                    return d

            self.iter[node] += 1  # 이 간선 소진 -- 다음 이웃으로 이동

        return 0  # 레벨 그래프에서 막다른 길; 백트랙

    def max_flow(self, source, sink):
        flow = 0

        # 외부 루프: 각 차단 유량(blocking flow) 단계 후 BFS로 레벨 그래프를 재구성.
        # 각 BFS 단계는 최단 증가 경로 길이를 최소 1 증가시키므로,
        # 최대 O(V)번의 BFS 단계가 있어 총 O(V²E) 복잡도가 됨.
        while self.bfs(source, sink):
            self.iter = [0] * self.n  # iter[node]는 다음 시도할 간선을 추적 -- 재방문 방지
            while True:
                f = self.dfs(source, sink, float('inf'))
                if f == 0:
                    break  # 차단 유량 완료 -- 레벨 그래프 재구성
                flow += f

        return flow
```

---

## 8. 시간 복잡도 정리

| 알고리즘 | 시간 복잡도 | 비고 |
|---------|------------|------|
| Ford-Fulkerson (DFS) | O(E × max_flow) | 정수 용량 |
| Edmonds-Karp (BFS) | O(VE²) | 항상 보장 |
| Dinic | O(V²E) | 일반 그래프 |
| Dinic (이분 그래프) | O(E√V) | 이분 매칭에 유리 |
| Kuhn's (이분 매칭) | O(VE) | 직접 구현 간단 |

---

## 9. 자주 하는 실수

### 실수 1: 역방향 간선 누락

```python
# 잘못된 코드
def add_edge(self, u, v, cap):
    self.capacity[u][v] = cap  # 역방향 없음!

# 올바른 코드
def add_edge(self, u, v, cap):
    self.capacity[u][v] += cap
    # 역방향은 자동 처리 (flow로 음수 가능)
```

### 실수 2: BFS에서 용량 0인 간선 통과

```python
# 잘못된 코드
if not visited[neighbor]:  # 용량 체크 없음!

# 올바른 코드
if not visited[neighbor] and self.capacity[node][neighbor] > 0:
```

### 실수 3: 양방향 간선 처리

```python
# 무방향 간선 (u-v) 추가 시
add_edge(u, v, cap)
add_edge(v, u, cap)  # 반대도 추가
```

---

## 10. 연습 문제

| 난이도 | 문제 유형 | 핵심 개념 |
|--------|----------|-----------|
| ★★☆ | 이분 매칭 기본 | Kuhn's 알고리즘 |
| ★★★ | 최대 유량 기본 | Edmonds-Karp |
| ★★★ | 경로 분리 | 간선/정점 분리 |
| ★★★★ | 최소 컷 | Max-Flow Min-Cut |
| ★★★★ | 프로젝트 선택 | 유량 모델링 |

---

## 다음 단계

- [26_Computational_Geometry.md](./26_Computational_Geometry.md) - 기하 알고리즘

---

## 학습 점검

1. Ford-Fulkerson에서 잔여 그래프란?
2. 최대 유량 = 최소 컷인 이유는?
3. 이분 매칭을 유량으로 변환하는 방법은?
4. Edmonds-Karp가 Ford-Fulkerson보다 나은 이유는?
