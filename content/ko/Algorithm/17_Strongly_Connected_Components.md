# 24. 강한 연결 요소 (Strongly Connected Components)

## 학습 목표
- 강한 연결 요소(SCC)의 개념 이해
- Tarjan 알고리즘 구현
- Kosaraju 알고리즘 구현
- 2-SAT 문제 해결
- 응축 그래프(DAG) 활용

## 1. 강한 연결 요소란?

### 정의

**방향 그래프**에서 **강한 연결 요소(SCC)**란 모든 정점 쌍 (u, v)에 대해 u→v, v→u 경로가 모두 존재하는 **최대** 부분 그래프입니다.

```
    ┌───────────┐
    │  SCC 1    │
    │  ┌→1→─┐   │
    │  │    ↓   │         ┌───────┐
    │  4←──2    │────────→│ SCC 2 │
    │      ↓    │         │  5→6  │
    │      3←───│         │  ↑ ↓  │
    └───────────┘         │  8←7  │
                          └───────┘
```

### 특성

1. **최대성**: SCC에 정점을 추가하면 더 이상 강하게 연결되지 않음
2. **분할성**: 모든 정점은 정확히 하나의 SCC에 속함
3. **DAG 구조**: SCC들을 하나의 노드로 보면 DAG(비순환 방향 그래프)가 됨

### SCC의 활용

```
┌─────────────────────────────────────────────────┐
│              SCC 활용 분야                        │
├─────────────────────────────────────────────────┤
│  • 2-SAT 문제 해결                               │
│  • 데드락 감지                                   │
│  • 소셜 네트워크 분석 (커뮤니티 탐지)              │
│  • 웹 페이지 클러스터링                          │
│  • 컴파일러 최적화 (순환 의존성 분석)             │
└─────────────────────────────────────────────────┘
```

---

## 2. Kosaraju 알고리즘

### 핵심 아이디어

1. 원본 그래프에서 DFS로 종료 순서 기록
2. 역방향 그래프에서 종료 순서의 역순으로 DFS
3. 각 DFS 트리가 하나의 SCC

### 동작 원리

```
Step 1: 원본 그래프 DFS (종료 순서 기록)
        1 → 2 → 3
        ↑       ↓
        └───────┘

        종료 순서: [3, 2, 1]

Step 2: 그래프 역방향 변환
        1 ← 2 ← 3
        ↓       ↑
        └───────┘

Step 3: 역순으로 DFS (1부터 시작)
        1 → 3 → 2 → (1로 돌아옴)

        SCC: {1, 2, 3}
```

### 구현

```python
from collections import defaultdict

class KosarajuSCC:
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.reverse_graph[v].append(u)

    def find_sccs(self):
        # Step 1: 원본 그래프에서 DFS, 종료 순서 기록.
        # 노드는 도달 가능한 모든 노드가 완료된 후에야 종료(추가)되므로,
        # "싱크" SCC(다른 SCC로의 나가는 간선이 없는)의 노드가 먼저 종료된다.
        # 이 순서를 역순으로 하면 step 2에서 "소스" SCC부터 시작하게 된다 —
        # 모든 곳에 도달할 수 있지만 외부에서 도달할 수 없는 노드.
        visited = [False] * self.n
        finish_order = []

        def dfs1(node):
            visited[node] = True
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    dfs1(neighbor)
            finish_order.append(node)  # 모든 후손 방문 후에 추가

        for i in range(self.n):
            if not visited[i]:
                dfs1(i)

        # Step 2: 역방향 그래프에서 종료 순서의 역순으로 DFS.
        # 모든 간선을 뒤집으면 "역방향 그래프에서 u로부터 도달 가능"은
        # "원본 그래프에서 u에 도달 가능"과 같다. 가장 높은 종료 시간의
        # 노드(소스 SCC 루트)에서 역방향 그래프 DFS를 시작하면
        # 같은 SCC 안에 갇히게 된다 — 간선이 뒤집혔기 때문에
        # 다른 SCC로 탈출할 수 없다. 따라서 여기서 각 DFS 트리가
        # 정확히 하나의 SCC가 된다.
        visited = [False] * self.n
        sccs = []

        def dfs2(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in self.reverse_graph[node]:
                if not visited[neighbor]:
                    dfs2(neighbor, component)

        # 종료 순서의 역순으로 처리
        for node in reversed(finish_order):
            if not visited[node]:
                component = []
                dfs2(node, component)
                sccs.append(component)

        return sccs

# 사용 예시
scc = KosarajuSCC(8)
edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,6), (6,4), (6,7)]
for u, v in edges:
    scc.add_edge(u, v)

result = scc.find_sccs()
print("SCCs:", result)
# SCCs: [[7], [4, 6, 5], [3], [0, 2, 1]]
```

### C++ 구현

```cpp
#include <bits/stdc++.h>
using namespace std;

class KosarajuSCC {
private:
    int n;
    vector<vector<int>> graph, reverseGraph;
    vector<bool> visited;
    vector<int> finishOrder;

    void dfs1(int node) {
        visited[node] = true;
        for (int next : graph[node]) {
            if (!visited[next]) dfs1(next);
        }
        finishOrder.push_back(node);
    }

    void dfs2(int node, vector<int>& component) {
        visited[node] = true;
        component.push_back(node);
        for (int next : reverseGraph[node]) {
            if (!visited[next]) dfs2(next, component);
        }
    }

public:
    KosarajuSCC(int n) : n(n), graph(n), reverseGraph(n) {}

    void addEdge(int u, int v) {
        graph[u].push_back(v);
        reverseGraph[v].push_back(u);
    }

    vector<vector<int>> findSCCs() {
        // Step 1
        visited.assign(n, false);
        for (int i = 0; i < n; i++) {
            if (!visited[i]) dfs1(i);
        }

        // Step 2
        visited.assign(n, false);
        vector<vector<int>> sccs;

        for (int i = n - 1; i >= 0; i--) {
            int node = finishOrder[i];
            if (!visited[node]) {
                vector<int> component;
                dfs2(node, component);
                sccs.push_back(component);
            }
        }

        return sccs;
    }
};
```

### 시간 복잡도

- **시간**: O(V + E) - DFS 2번
- **공간**: O(V + E) - 역방향 그래프 저장

---

## 3. Tarjan 알고리즘

### 핵심 아이디어

단일 DFS로 SCC를 찾는 알고리즘. **발견 시간**과 **low-link 값**을 사용합니다.

- **발견 시간 (disc)**: 노드를 처음 방문한 시점
- **Low-link (low)**: 해당 노드에서 도달 가능한 가장 작은 발견 시간

### 동작 원리

```
노드 방문 시:
1. disc[node] = low[node] = 현재 시간
2. 스택에 push
3. 이웃 탐색:
   - 미방문: 재귀 호출 후 low[node] = min(low[node], low[neighbor])
   - 스택에 있음: low[node] = min(low[node], disc[neighbor])
4. disc[node] == low[node]이면 SCC의 루트
   → 스택에서 node까지 pop하여 SCC 구성
```

### 시각화

```
DFS 진행:
노드:  1 → 2 → 3 → 1(back edge)
disc: [1,  2,  3]
low:  [1,  1,  1]  ← 3에서 1로 가는 back edge로 인해 업데이트

노드 1에서: disc[1] == low[1] = 1
→ SCC 발견: {1, 2, 3}
```

### 구현

```python
class TarjanSCC:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.disc = [-1] * n      # 발견 시간
        self.low = [-1] * n       # low-link 값
        self.on_stack = [False] * n
        self.stack = []
        self.time = 0
        self.sccs = []

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs(self, node):
        self.disc[node] = self.low[node] = self.time
        self.time += 1
        self.stack.append(node)
        self.on_stack[node] = True  # 스택 소속 여부를 "방문 여부"와 별도로 추적

        for neighbor in self.graph[node]:
            if self.disc[neighbor] == -1:
                # 트리 간선(Tree edge): 재귀 후 서브트리에서 도달 가능한 가장 작은
                # disc 값을 역전파 — 깊은 곳의 역방향 간선(back edge)이 경로 상의
                # 모든 조상 low 값에 영향을 미치는 원리
                self.dfs(neighbor)
                self.low[node] = min(self.low[node], self.low[neighbor])
            elif self.on_stack[neighbor]:
                # 현재 DFS 스택에 남아있는 조상으로의 역방향 간선(back edge):
                # 이 노드가 해당 조상까지 "탈출" 가능하므로 low 갱신.
                # on_stack 확인 이유: 이미 팝되어 완성된 SCC에 속한 이웃은
                # 다른 컴포넌트에 있으므로, 교차 간선(cross-edge)을 따라가면
                # 두 개의 별도 SCC를 잘못 병합하게 됨
                self.low[node] = min(self.low[node], self.disc[neighbor])

        # disc[node] == low[node]이면 이 노드 아래에서 위로 올라가는 역방향 간선이
        # 없다는 의미 — 이 노드가 해당 SCC의 최상위(가장 오래된) 노드(루트).
        # 이 노드까지 포함하여 스택에서 팝하면 하나의 SCC가 완성됨
        if self.disc[node] == self.low[node]:
            component = []
            while True:
                top = self.stack.pop()
                self.on_stack[top] = False
                component.append(top)
                if top == node:
                    break
            self.sccs.append(component)

    def find_sccs(self):
        for i in range(self.n):
            if self.disc[i] == -1:
                self.dfs(i)
        return self.sccs

# 사용 예시
tarjan = TarjanSCC(8)
edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,6), (6,4), (6,7)]
for u, v in edges:
    tarjan.add_edge(u, v)

result = tarjan.find_sccs()
print("SCCs:", result)
# SCCs: [[2, 1, 0], [6, 5, 4], [3], [7]]
```

### C++ 구현

```cpp
#include <bits/stdc++.h>
using namespace std;

class TarjanSCC {
private:
    int n, timer = 0;
    vector<vector<int>> graph;
    vector<int> disc, low;
    vector<bool> onStack;
    stack<int> st;
    vector<vector<int>> sccs;

    void dfs(int node) {
        disc[node] = low[node] = timer++;
        st.push(node);
        onStack[node] = true;

        for (int next : graph[node]) {
            if (disc[next] == -1) {
                dfs(next);
                low[node] = min(low[node], low[next]);
            } else if (onStack[next]) {
                low[node] = min(low[node], disc[next]);
            }
        }

        if (disc[node] == low[node]) {
            vector<int> component;
            while (true) {
                int top = st.top(); st.pop();
                onStack[top] = false;
                component.push_back(top);
                if (top == node) break;
            }
            sccs.push_back(component);
        }
    }

public:
    TarjanSCC(int n) : n(n), graph(n), disc(n, -1), low(n, -1), onStack(n, false) {}

    void addEdge(int u, int v) {
        graph[u].push_back(v);
    }

    vector<vector<int>> findSCCs() {
        for (int i = 0; i < n; i++) {
            if (disc[i] == -1) dfs(i);
        }
        return sccs;
    }
};
```

### Kosaraju vs Tarjan 비교

| 특성 | Kosaraju | Tarjan |
|------|----------|--------|
| DFS 횟수 | 2번 | 1번 |
| 추가 공간 | 역방향 그래프 O(V+E) | 스택 O(V) |
| 구현 복잡도 | 직관적 | 약간 복잡 |
| 온라인 처리 | 불가능 | 가능 |

---

## 4. 응축 그래프 (Condensation Graph)

### 개념

각 SCC를 하나의 노드로 **압축**하면 **DAG**가 됩니다.

```
원본 그래프:                    응축 그래프 (DAG):
    1 ←→ 2                         SCC0
    ↓     ↓                          ↓
    4 ←→ 3 → 5 → 6                 SCC1 → SCC2
              ↑   ↓
              8 ← 7

SCC0 = {1,2,3,4}
SCC1 = {5,6,7,8}  → 하나의 순환
SCC2 = ... (만약 있다면)
```

### 구현

```python
def build_condensation_graph(n, edges):
    # 1. SCC 찾기
    tarjan = TarjanSCC(n)
    for u, v in edges:
        tarjan.add_edge(u, v)
    sccs = tarjan.find_sccs()

    # 2. 각 노드가 속한 SCC 번호 매핑
    scc_id = [-1] * n
    for i, component in enumerate(sccs):
        for node in component:
            scc_id[node] = i

    # 3. 응축 그래프 구성
    num_sccs = len(sccs)
    condensed = [set() for _ in range(num_sccs)]

    for u, v in edges:
        su, sv = scc_id[u], scc_id[v]
        if su != sv:
            condensed[su].add(sv)

    # set을 list로 변환
    condensed = [list(neighbors) for neighbors in condensed]

    return sccs, scc_id, condensed

# 사용 예시
edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,6), (6,4)]
sccs, scc_id, dag = build_condensation_graph(7, edges)

print("SCCs:", sccs)
print("DAG edges:")
for i, neighbors in enumerate(dag):
    for j in neighbors:
        print(f"  SCC{i} → SCC{j}")
```

### 응축 그래프 활용

```python
def count_reachable_nodes(n, edges):
    """
    각 노드에서 도달 가능한 노드 수 계산
    응축 그래프에서 DAG DP로 효율적 계산
    """
    sccs, scc_id, dag = build_condensation_graph(n, edges)
    num_sccs = len(sccs)

    # 위상 정렬 순서로 처리
    in_degree = [0] * num_sccs
    for u in range(num_sccs):
        for v in dag[u]:
            in_degree[v] += 1

    # 각 SCC에서 도달 가능한 노드 집합
    reachable = [set(sccs[i]) for i in range(num_sccs)]

    from collections import deque
    queue = deque([i for i in range(num_sccs) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for next_node in dag[node]:
            in_degree[next_node] -= 1
            if in_degree[next_node] == 0:
                queue.append(next_node)

    # 역순으로 도달 가능 노드 집합 전파
    for scc in reversed(order):
        for next_scc in dag[scc]:
            reachable[scc] |= reachable[next_scc]

    # 각 원본 노드의 도달 가능 노드 수
    result = [0] * n
    for i in range(n):
        result[i] = len(reachable[scc_id[i]])

    return result
```

---

## 5. 2-SAT 문제

### 개념

2-SAT은 각 절(clause)에 **정확히 2개의 리터럴**이 있는 논리식의 만족 가능성을 판별합니다.

```
예: (x₁ ∨ x₂) ∧ (¬x₁ ∨ x₃) ∧ (¬x₂ ∨ ¬x₃)

리터럴: x₁, ¬x₁, x₂, ¬x₂, x₃, ¬x₃
절: (a ∨ b) 형태
```

### 그래프 변환

**핵심**: (a ∨ b) = (¬a → b) ∧ (¬b → a)

```
절 (x₁ ∨ x₂):
  ¬x₁ → x₂  (x₁이 거짓이면 x₂는 참)
  ¬x₂ → x₁  (x₂가 거짓이면 x₁은 참)

변수 xi에 대해:
  노드 2i: xi
  노드 2i+1: ¬xi
```

### 만족 가능성 판별

**정리**: 2-SAT 식이 만족 가능 ⟺ 어떤 변수 x에 대해서도 x와 ¬x가 같은 SCC에 속하지 않음

### 구현

```python
class TwoSAT:
    def __init__(self, n):
        """n개의 변수 (0 ~ n-1)"""
        self.n = n
        self.graph = [[] for _ in range(2 * n)]

    def add_clause(self, a, neg_a, b, neg_b):
        """
        (a ∨ b) 절 추가
        a, b: 변수 인덱스 (0 ~ n-1)
        neg_a, neg_b: True면 부정
        """
        # 리터럴 → 노드 번호
        def to_node(var, negated):
            return 2 * var + (1 if negated else 0)

        node_a = to_node(a, neg_a)
        node_b = to_node(b, neg_b)
        not_a = to_node(a, not neg_a)
        not_b = to_node(b, not neg_b)

        # ¬a → b, ¬b → a
        self.graph[not_a].append(node_b)
        self.graph[not_b].append(node_a)

    def add_or(self, a, b):
        """(a ∨ b) - 둘 중 하나는 참"""
        self.add_clause(a, False, b, False)

    def add_implies(self, a, b):
        """a → b (a이면 b)"""
        # a → b = ¬a ∨ b
        self.add_clause(a, True, b, False)

    def add_xor(self, a, b):
        """a XOR b (정확히 하나만 참)"""
        # (a ∨ b) ∧ (¬a ∨ ¬b)
        self.add_clause(a, False, b, False)
        self.add_clause(a, True, b, True)

    def add_equal(self, a, b):
        """a = b (둘 다 같은 값)"""
        # (a → b) ∧ (b → a)
        self.add_implies(a, b)
        self.add_implies(b, a)

    def set_true(self, a):
        """변수 a를 반드시 참으로"""
        # a ∨ a
        self.add_clause(a, False, a, False)

    def set_false(self, a):
        """변수 a를 반드시 거짓으로"""
        # ¬a ∨ ¬a
        self.add_clause(a, True, a, True)

    def solve(self):
        """
        만족 가능하면 각 변수의 값 반환, 불가능하면 None
        """
        # Tarjan으로 SCC 찾기
        n = 2 * self.n
        disc = [-1] * n
        low = [-1] * n
        on_stack = [False] * n
        stack = []
        timer = [0]
        scc_id = [-1] * n
        scc_count = [0]

        def dfs(node):
            disc[node] = low[node] = timer[0]
            timer[0] += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in self.graph[node]:
                if disc[neighbor] == -1:
                    dfs(neighbor)
                    low[node] = min(low[node], low[neighbor])
                elif on_stack[neighbor]:
                    low[node] = min(low[node], disc[neighbor])

            if disc[node] == low[node]:
                while True:
                    top = stack.pop()
                    on_stack[top] = False
                    scc_id[top] = scc_count[0]
                    if top == node:
                        break
                scc_count[0] += 1

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        # 만족 가능성 확인
        for i in range(self.n):
            if scc_id[2 * i] == scc_id[2 * i + 1]:
                return None  # x와 ¬x가 같은 SCC

        # 값 결정: SCC 번호가 더 큰 쪽이 참
        # (Tarjan은 역위상 순서로 SCC 번호 부여)
        result = [False] * self.n
        for i in range(self.n):
            # x의 SCC 번호 > ¬x의 SCC 번호면 x = True
            result[i] = scc_id[2 * i] > scc_id[2 * i + 1]

        return result

# 사용 예시
sat = TwoSAT(3)  # 변수: x0, x1, x2

# (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2)
sat.add_or(0, 1)                    # x0 ∨ x1
sat.add_clause(0, True, 2, False)   # ¬x0 ∨ x2
sat.add_clause(1, True, 2, True)    # ¬x1 ∨ ¬x2

result = sat.solve()
if result:
    print("만족 가능:", result)
    # 검증
    x0, x1, x2 = result
    print(f"x0={x0}, x1={x1}, x2={x2}")
    clause1 = x0 or x1
    clause2 = (not x0) or x2
    clause3 = (not x1) or (not x2)
    print(f"검증: {clause1 and clause2 and clause3}")
else:
    print("만족 불가능")
```

### 2-SAT 응용 문제

```python
def team_assignment(n, conflicts):
    """
    n명을 2개 팀으로 나누기
    conflicts[i] = (a, b): a와 b는 같은 팀 불가
    """
    sat = TwoSAT(n)

    for a, b in conflicts:
        # a와 b가 다른 팀
        # team[a] XOR team[b] = True
        sat.add_xor(a, b)

    result = sat.solve()
    if result is None:
        return None

    team1 = [i for i in range(n) if result[i]]
    team2 = [i for i in range(n) if not result[i]]
    return team1, team2

# 예: 0-1 충돌, 1-2 충돌
result = team_assignment(3, [(0, 1), (1, 2)])
print(result)  # ([0, 2], [1]) 또는 ([1], [0, 2])
```

---

## 6. 실전 문제 패턴

### 패턴 1: SCC 개수 세기

```python
def count_sccs(n, edges):
    tarjan = TarjanSCC(n)
    for u, v in edges:
        tarjan.add_edge(u, v)
    return len(tarjan.find_sccs())
```

### 패턴 2: 모든 노드 도달 가능한 시작점

```python
def find_universal_source(n, edges):
    """
    모든 노드에 도달 가능한 시작점이 있는가?
    응축 그래프에서 진입 차수가 0인 SCC가 1개면 가능
    """
    sccs, scc_id, dag = build_condensation_graph(n, edges)
    num_sccs = len(sccs)

    in_degree = [0] * num_sccs
    for u in range(num_sccs):
        for v in dag[u]:
            in_degree[v] += 1

    sources = [i for i in range(num_sccs) if in_degree[i] == 0]

    if len(sources) == 1:
        # 해당 SCC의 아무 노드나 반환
        return sccs[sources[0]][0]
    return -1
```

### 패턴 3: 최소 간선 추가로 강한 연결

```python
def min_edges_to_strongly_connect(n, edges):
    """
    전체 그래프를 하나의 SCC로 만들기 위해 추가할 최소 간선 수
    """
    if n <= 1:
        return 0

    sccs, scc_id, dag = build_condensation_graph(n, edges)
    num_sccs = len(sccs)

    if num_sccs == 1:
        return 0

    in_degree = [0] * num_sccs
    out_degree = [0] * num_sccs

    for u in range(num_sccs):
        for v in dag[u]:
            out_degree[u] += 1
            in_degree[v] += 1

    sources = sum(1 for d in in_degree if d == 0)  # 진입 차수 0
    sinks = sum(1 for d in out_degree if d == 0)   # 진출 차수 0

    return max(sources, sinks)
```

### 패턴 4: 방향 그래프에서 사이클 노드

```python
def nodes_in_cycles(n, edges):
    """사이클에 포함된 모든 노드 찾기"""
    tarjan = TarjanSCC(n)
    for u, v in edges:
        tarjan.add_edge(u, v)

    sccs = tarjan.find_sccs()

    # 크기가 2 이상인 SCC의 노드들
    cycle_nodes = set()
    for component in sccs:
        if len(component) > 1:
            cycle_nodes.update(component)

    # 크기 1인 SCC도 자기 자신으로 가는 간선이 있으면 사이클
    edge_set = set(edges)
    for component in sccs:
        if len(component) == 1:
            node = component[0]
            if (node, node) in edge_set:
                cycle_nodes.add(node)

    return cycle_nodes
```

---

## 7. 시간 복잡도 정리

| 연산 | Kosaraju | Tarjan |
|------|----------|--------|
| SCC 찾기 | O(V + E) | O(V + E) |
| 응축 그래프 | O(V + E) | O(V + E) |
| 2-SAT | - | O(V + E) |
| 공간 | O(V + E) × 2 | O(V + E) |

---

## 8. 자주 하는 실수

### 실수 1: on_stack 체크 누락

```python
# 잘못된 코드
if disc[neighbor] != -1:
    low[node] = min(low[node], disc[neighbor])

# 올바른 코드
if disc[neighbor] != -1 and on_stack[neighbor]:
    low[node] = min(low[node], disc[neighbor])
```

### 실수 2: 2-SAT 노드 번호 계산 오류

```python
# 변수 i에 대해:
# - i 자체: 2*i
# - ¬i: 2*i + 1

# 잘못된 부정
def negate(node):
    return node + 1  # 틀림!

# 올바른 부정
def negate(node):
    return node ^ 1  # XOR로 0↔1 전환
```

### 실수 3: 응축 그래프 중복 간선

```python
# 중복 간선 발생 가능
for u, v in edges:
    if scc_id[u] != scc_id[v]:
        condensed[scc_id[u]].append(scc_id[v])  # 중복!

# set으로 해결
condensed = [set() for _ in range(num_sccs)]
for u, v in edges:
    if scc_id[u] != scc_id[v]:
        condensed[scc_id[u]].add(scc_id[v])
```

---

## 9. 연습 문제

| 난이도 | 문제 유형 | 핵심 개념 |
|--------|----------|-----------|
| ★★☆ | SCC 개수 세기 | 기본 Tarjan/Kosaraju |
| ★★☆ | 도미노 | SCC + DAG 분석 |
| ★★★ | 2-SAT 기본 | 그래프 변환 + SCC |
| ★★★ | 팀 배정 | 2-SAT 응용 |
| ★★★★ | 최소 간선 추가 | 응축 그래프 활용 |

---

## 다음 단계

- [18_Dynamic_Programming.md](./18_Dynamic_Programming.md) - DP, 메모이제이션

---

## 학습 점검

1. Tarjan 알고리즘에서 low-link 값의 의미는?
2. 2-SAT에서 (a ∨ b)를 어떻게 그래프로 변환하는가?
3. 응축 그래프가 항상 DAG인 이유는?
4. Kosaraju가 역방향 그래프를 사용하는 이유는?
