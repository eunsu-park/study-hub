# 최소 신장 트리 (Minimum Spanning Tree)

## 개요

최소 신장 트리(MST)는 그래프의 모든 정점을 연결하면서 간선 가중치 합이 최소인 트리입니다. Kruskal과 Prim 알고리즘, 그리고 Union-Find 자료구조를 학습합니다.

---

## 목차

1. [MST 개념](#1-mst-개념)
2. [Union-Find](#2-union-find)
3. [크루스칼 (Kruskal)](#3-크루스칼-kruskal)
4. [프림 (Prim)](#4-프림-prim)
5. [알고리즘 비교](#5-알고리즘-비교)
6. [연습 문제](#6-연습-문제)

---

## 1. MST 개념

### 신장 트리 (Spanning Tree)

```
신장 트리: 그래프의 모든 정점을 포함하면서
          사이클이 없는 부분 그래프

조건:
- 정점 수: V
- 간선 수: V-1
- 모든 정점이 연결됨
- 사이클 없음
```

### 최소 신장 트리 (MST)

```
MST: 신장 트리 중 간선 가중치 합이 최소인 것

    (1)──4──(2)
    │╲      │╲
    2  1    5  3
    │    ╲  │    ╲
   (3)──6──(4)──7──(5)

MST (가중치 합: 11):
    (1)──4──(2)
     ╲       ╲
      1       3
        ╲      ╲
        (4)──────(5)
         │
         2(3에 연결 아님, 그림 상 (3)에 연결)

실제 MST:
(1)-1-(4), (1)-2-(3), (2)-4-(1), (2)-3-(5)
→ 1+2+4+3 = 10? 또는 다른 조합
```

### MST 속성

```
1. 컷 속성: 그래프를 두 집합으로 나눌 때,
   교차하는 간선 중 최소 가중치 간선은 MST에 포함

2. 사이클 속성: 사이클에서 최대 가중치 간선은
   MST에 포함되지 않음

3. 유일성: 모든 간선 가중치가 다르면 MST는 유일
```

---

## 2. Union-Find (Disjoint Set Union)

### 개념

```
서로소 집합: 공통 원소가 없는 집합들

연산:
- Find(x): x가 속한 집합의 대표 원소 반환
- Union(x, y): x와 y가 속한 집합을 합침

용도:
- 사이클 탐지
- 연결 요소 관리
- Kruskal 알고리즘
```

### 기본 구현

```c
// C
#define MAX_N 100001

int parent[MAX_N];

void init(int n) {
    for (int i = 0; i < n; i++) {
        parent[i] = i;  // 자기 자신이 부모
    }
}

int find(int x) {
    if (parent[x] == x) {
        return x;
    }
    return find(parent[x]);
}

void unite(int x, int y) {
    int px = find(x);
    int py = find(y);
    if (px != py) {
        parent[px] = py;
    }
}
```

### 최적화 1: 경로 압축 (Path Compression)

```
Find 시 경로상의 모든 노드를 루트에 직접 연결

     (5)              (5)
      │               /|\
     (3)      →     (1)(2)(3)
     /│              │
   (1)(2)            (4)
    │
   (4)

시간 복잡도: 거의 O(1) (Amortized)
```

```c
// C - 경로 압축
int find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);  // 재귀적으로 루트 연결
    }
    return parent[x];
}
```

### 최적화 2: 랭크 합치기 (Union by Rank)

```
작은 트리를 큰 트리에 합침

  트리1 (랭크 2)    트리2 (랭크 1)
       (a)              (b)
      / │ \              │
    (c)(d)(e)           (f)

합친 후:
       (a)
      /│╲  \
    (c)(d)(e)(b)
              │
             (f)
```

```c
// C - 경로 압축 + 랭크 합치기
int parent[MAX_N];
int rank_arr[MAX_N];

void init(int n) {
    for (int i = 0; i < n; i++) {
        parent[i] = i;
        rank_arr[i] = 0;
    }
}

int find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

void unite(int x, int y) {
    int px = find(x);
    int py = find(y);

    if (px == py) return;

    // 랭크가 작은 트리를 큰 트리에 붙임
    if (rank_arr[px] < rank_arr[py]) {
        parent[px] = py;
    } else if (rank_arr[px] > rank_arr[py]) {
        parent[py] = px;
    } else {
        parent[py] = px;
        rank_arr[px]++;
    }
}
```

### C++/Python 구현

```cpp
// C++
class UnionFind {
private:
    vector<int> parent, rank_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;

        return true;
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

```python
# Python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

---

## 3. 크루스칼 (Kruskal)

### 개념

```
간선을 가중치 순으로 정렬하고,
사이클이 생기지 않는 간선을 선택

원리:
1. 모든 간선을 가중치 오름차순 정렬
2. 가장 작은 간선부터 선택
3. 사이클이 생기면 건너뜀 (Union-Find로 확인)
4. V-1개 간선을 선택하면 종료

시간 복잡도: O(E log E)
```

### 동작 예시

```
그래프:
   (0)──7──(1)
    │╲    ╱│
    5  8 9  7
    │    ╲╱ │
   (2)──5──(3)

간선 정렬: (2,3,5), (0,2,5), (0,1,7), (1,3,7), (0,3,8), (1,2,9)

선택 과정:
1. (2,3,5) 선택 → 사이클 X ✓
2. (0,2,5) 선택 → 사이클 X ✓
3. (0,1,7) 선택 → 사이클 X ✓
4. V-1=3개 간선 선택 완료

MST: (2,3), (0,2), (0,1)
가중치 합: 5+5+7 = 17
```

### 구현

```c
// C
#define MAX_E 100001

typedef struct {
    int u, v, weight;
} Edge;

Edge edges[MAX_E];
int parent[MAX_E];

int cmp(const void* a, const void* b) {
    return ((Edge*)a)->weight - ((Edge*)b)->weight;
}

int find(int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]);
    return parent[x];
}

int kruskal(int V, int E) {
    // 초기화
    for (int i = 0; i < V; i++)
        parent[i] = i;

    // 정렬
    qsort(edges, E, sizeof(Edge), cmp);

    int mstWeight = 0;
    int edgeCount = 0;

    for (int i = 0; i < E && edgeCount < V - 1; i++) {
        int pu = find(edges[i].u);
        int pv = find(edges[i].v);

        if (pu != pv) {
            parent[pu] = pv;
            mstWeight += edges[i].weight;
            edgeCount++;
        }
    }

    return mstWeight;
}
```

```cpp
// C++
struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

int kruskal(int V, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());

    UnionFind uf(V);
    int mstWeight = 0;
    int edgeCount = 0;

    for (const auto& e : edges) {
        if (edgeCount >= V - 1) break;

        if (uf.unite(e.u, e.v)) {
            mstWeight += e.weight;
            edgeCount++;
        }
    }

    return mstWeight;
}
```

```python
# Python
def kruskal(V, edges):
    edges.sort(key=lambda x: x[2])  # 가중치 기준 정렬
    uf = UnionFind(V)

    mst_weight = 0
    edge_count = 0

    for u, v, w in edges:
        if edge_count >= V - 1:
            break

        if uf.union(u, v):
            mst_weight += w
            edge_count += 1

    return mst_weight
```

---

## 4. 프림 (Prim)

### 개념

```
시작 정점에서 MST를 점점 확장

원리:
1. 임의의 정점에서 시작
2. MST에 포함된 정점에서 나가는 간선 중
   가장 작은 가중치 간선 선택
3. 새로운 정점을 MST에 추가
4. 모든 정점이 포함되면 종료

시간 복잡도:
- 우선순위 큐: O(E log V)
- 인접 행렬: O(V²)
```

### 동작 예시

```
그래프 (0에서 시작):
   (0)──7──(1)
    │╲    ╱│
    5  8 9  7
    │    ╲╱ │
   (2)──5──(3)

단계:
1. 시작: MST = {0}
   인접 간선: (0,1,7), (0,2,5), (0,3,8)
   선택: (0,2,5) → MST = {0,2}

2. 인접 간선: (0,1,7), (0,3,8), (2,3,5)
   선택: (2,3,5) → MST = {0,2,3}

3. 인접 간선: (0,1,7), (3,1,7)
   선택: (0,1,7) 또는 (3,1,7) → MST = {0,1,2,3}

결과: 가중치 합 = 5+5+7 = 17
```

### 구현 (우선순위 큐)

```cpp
// C++
int prim(int V, const vector<vector<pair<int,int>>>& adj) {
    vector<bool> inMST(V, false);
    // {weight, vertex}
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;

    int mstWeight = 0;
    pq.push({0, 0});  // 시작 정점

    while (!pq.empty()) {
        auto [w, u] = pq.top();
        pq.pop();

        if (inMST[u]) continue;

        inMST[u] = true;
        mstWeight += w;

        for (auto [v, weight] : adj[u]) {
            if (!inMST[v]) {
                pq.push({weight, v});
            }
        }
    }

    return mstWeight;
}
```

```python
# Python
import heapq

def prim(V, adj):
    in_mst = [False] * V
    pq = [(0, 0)]  # (weight, vertex)
    mst_weight = 0

    while pq:
        w, u = heapq.heappop(pq)

        if in_mst[u]:
            continue

        in_mst[u] = True
        mst_weight += w

        for v, weight in adj[u]:
            if not in_mst[v]:
                heapq.heappush(pq, (weight, v))

    return mst_weight
```

### 구현 (인접 행렬, V²)

```cpp
// C++ - 밀집 그래프에 유리
int primMatrix(int V, const vector<vector<int>>& adj) {
    vector<int> key(V, INT_MAX);
    vector<bool> inMST(V, false);

    key[0] = 0;
    int mstWeight = 0;

    for (int count = 0; count < V; count++) {
        // 최소 key 값을 가진 정점 선택
        int u = -1;
        for (int v = 0; v < V; v++) {
            if (!inMST[v] && (u == -1 || key[v] < key[u])) {
                u = v;
            }
        }

        inMST[u] = true;
        mstWeight += key[u];

        // 인접 정점 key 갱신
        for (int v = 0; v < V; v++) {
            if (adj[u][v] && !inMST[v] && adj[u][v] < key[v]) {
                key[v] = adj[u][v];
            }
        }
    }

    return mstWeight;
}
```

---

## 5. 알고리즘 비교

### Kruskal vs Prim

```
┌─────────────┬──────────────────┬──────────────────┐
│             │ Kruskal          │ Prim             │
├─────────────┼──────────────────┼──────────────────┤
│ 접근 방식   │ 간선 중심        │ 정점 중심        │
│ 자료구조    │ Union-Find       │ 우선순위 큐      │
│ 시간 복잡도 │ O(E log E)       │ O(E log V)       │
│ 적합한 경우 │ 희소 그래프      │ 밀집 그래프      │
│ 구현 복잡도 │ 상대적 간단      │ 상대적 복잡      │
└─────────────┴──────────────────┴──────────────────┘
```

### 선택 기준

```
희소 그래프 (E ≈ V): Kruskal 유리
밀집 그래프 (E ≈ V²): Prim 유리

간선 리스트로 주어짐: Kruskal 유리
인접 리스트로 주어짐: Prim 유리
```

---

## 6. 연습 문제

### 문제 1: 최소 스패닝 트리

주어진 그래프의 MST 가중치 합을 구하세요.

<details>
<summary>정답 코드</summary>

```python
def solution(V, edges):
    # Kruskal
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(V)

    total = 0
    count = 0

    for u, v, w in edges:
        if count >= V - 1:
            break
        if uf.union(u, v):
            total += w
            count += 1

    return total
```

</details>

### 문제 2: 도시 분할 계획

N개 마을을 2개 그룹으로 나누고, 각 그룹을 최소 비용으로 연결하세요.

<details>
<summary>힌트</summary>

MST를 구한 후 가장 큰 간선 하나를 제거하면 2개 그룹이 됨

</details>

<details>
<summary>정답 코드</summary>

```python
def divide_villages(V, edges):
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(V)

    mst_edges = []

    for u, v, w in edges:
        if uf.union(u, v):
            mst_edges.append(w)
            if len(mst_edges) == V - 1:
                break

    # 가장 큰 간선 제거
    return sum(mst_edges) - max(mst_edges)
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 알고리즘 |
|--------|------|--------|----------|
| ⭐⭐ | [최소 스패닝 트리](https://www.acmicpc.net/problem/1197) | 백준 | Kruskal/Prim |
| ⭐⭐ | [상근이의 여행](https://www.acmicpc.net/problem/9372) | 백준 | MST 개념 |
| ⭐⭐⭐ | [도시 분할 계획](https://www.acmicpc.net/problem/1647) | 백준 | MST 응용 |
| ⭐⭐⭐ | [네트워크 연결](https://www.acmicpc.net/problem/1922) | 백준 | MST |
| ⭐⭐⭐ | [Min Cost to Connect](https://leetcode.com/problems/min-cost-to-connect-all-points/) | LeetCode | Prim |

---

## 템플릿 정리

### Union-Find

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### Kruskal

```python
def kruskal(V, edges):
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(V)
    total = 0
    for u, v, w in edges:
        if uf.union(u, v):
            total += w
    return total
```

### Prim

```python
def prim(V, adj):
    in_mst = [False] * V
    pq = [(0, 0)]
    total = 0
    while pq:
        w, u = heapq.heappop(pq)
        if in_mst[u]:
            continue
        in_mst[u] = True
        total += w
        for v, weight in adj[u]:
            if not in_mst[v]:
                heapq.heappush(pq, (weight, v))
    return total
```

---

## 다음 단계

- [16_LCA와_트리쿼리.md](./16_LCA와_트리쿼리.md) - LCA, 트리 쿼리

---

## 참고 자료

- [MST Visualization](https://visualgo.net/en/mst)
- [Union-Find Tutorial](https://cp-algorithms.com/data_structures/disjoint_set_union.html)
- Introduction to Algorithms (CLRS) - Chapter 23
