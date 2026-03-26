# 위상 정렬 (Topological Sort)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 위상 정렬(topological sort)의 개념을 설명하고 적용 가능한 조건(DAG 요건)을 서술할 수 있다
2. 진입 차수(in-degree) 추적과 BFS를 이용한 칸(Kahn) 알고리즘을 구현하여 위상 순서를 생성할 수 있다
3. 후위 순서(post-order) 스택을 사용한 DFS 기반 위상 정렬을 구현할 수 있다
4. 칸 알고리즘과 DFS 색칠(coloring) 방법을 이용하여 방향 그래프에서 사이클(cycle)을 감지할 수 있다
5. 수강 신청 일정 및 빌드 시스템과 같은 의존성 순서 정렬 문제에 위상 정렬을 적용할 수 있다

---

## 개요

위상 정렬은 방향 비순환 그래프(DAG)의 정점들을 선형으로 정렬하는 알고리즘입니다. 모든 간선 (u, v)에 대해 u가 v보다 먼저 나타나도록 정렬합니다.

---

## 목차

1. [위상 정렬 개념](#1-위상-정렬-개념)
2. [Kahn 알고리즘 (BFS)](#2-kahn-알고리즘-bfs)
3. [DFS 기반 위상 정렬](#3-dfs-기반-위상-정렬)
4. [사이클 탐지](#4-사이클-탐지)
5. [활용 문제](#5-활용-문제)
6. [연습 문제](#6-연습-문제)

---

## 1. 위상 정렬 개념

### 1.1 DAG (Directed Acyclic Graph)

```
DAG: 방향 비순환 그래프
- 방향 간선을 가짐
- 사이클이 없음

위상 정렬이 가능한 조건: 그래프가 DAG

예시: 수강 신청

과목A → 과목B → 과목D
  ↓       ↓
과목C → 과목E

유효한 수강 순서:
A → B → C → D → E  또는
A → C → B → D → E  또는
A → C → B → E → D
```

### 1.2 진입 차수와 진출 차수

```
진입 차수 (In-degree): 정점으로 들어오는 간선 수
진출 차수 (Out-degree): 정점에서 나가는 간선 수

예시:
    A → B
    ↓   ↓
    C → D

정점 | 진입 | 진출
-----|------|------
  A  |   0  |   2
  B  |   1  |   1
  C  |   1  |   1
  D  |   2  |   0

진입 차수가 0인 정점: 시작점 (의존성 없음)
```

### 1.3 위상 순서의 특성

```
- 여러 위상 순서가 존재할 수 있음
- DAG에서만 위상 정렬 가능
- 사이클 존재 시 위상 정렬 불가능

위상 정렬 알고리즘:
1. Kahn 알고리즘 (BFS) - O(V + E)
2. DFS 기반 - O(V + E)
```

---

## 2. Kahn 알고리즘 (BFS)

### 2.1 알고리즘 원리

```
1. 모든 정점의 진입 차수 계산
2. 진입 차수가 0인 정점을 큐에 삽입
3. 큐에서 정점을 꺼내 결과에 추가
4. 해당 정점에서 나가는 간선 제거 (인접 정점 진입 차수 감소)
5. 새로 진입 차수가 0이 된 정점을 큐에 삽입
6. 큐가 빌 때까지 반복

시각화:
초기: in_degree = [0, 1, 1, 2]
      A(0) → B(1) → D(2)
        ↓      ↓
       C(1) ←──┘

단계 1: A 선택 (진입차수 0)
        큐: [A], 결과: [A]
        B, C의 진입차수 감소 → B(0), C(0)

단계 2: B 선택 (또는 C)
        큐: [B, C], 결과: [A, B]
        D의 진입차수 감소 → D(1)

단계 3: C 선택
        큐: [C], 결과: [A, B, C]
        D의 진입차수 감소 → D(0)

단계 4: D 선택
        큐: [D], 결과: [A, B, C, D]

최종: [A, B, C, D]
```

### 2.2 구현

```python
from collections import deque

def topological_sort_kahn(n, edges):
    """
    Kahn 알고리즘 (BFS 기반 위상 정렬)
    n: 정점 수 (0부터 n-1)
    edges: [(u, v), ...] u → v 간선
    시간: O(V + E)
    """
    # 그래프 구성 및 진입 차수 계산
    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1  # 각 노드의 선행 조건 수를 카운트

    # 진입 차수가 0인 노드는 미해결 의존성이 없으므로
    # 즉시 스케줄링해도 안전하다
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    result = []

    while queue:
        node = queue.popleft()
        result.append(node)  # 이 노드는 이제 "완료"됨

        for neighbor in graph[node]:
            # 'node' 완료는 각 이웃의 선행 조건 하나를 충족시킨다;
            # 카운트가 0이 되면 스케줄링 가능 상태가 된다
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 사이클이 존재하면 일부 노드는 진입 차수가 0에 도달하지 못해
    # result에 포함되지 않는다 — 길이 확인으로 추가 기록 없이 이를 감지한다
    if len(result) != n:
        return []  # 사이클 존재

    return result


# 예시
n = 6
edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
print(topological_sort_kahn(n, edges))  # [4, 5, 2, 0, 3, 1] 또는 다른 유효한 순서
```

### 2.3 C++ 구현

```cpp
#include <vector>
#include <queue>
using namespace std;

vector<int> topologicalSortKahn(int n, vector<pair<int, int>>& edges) {
    vector<vector<int>> graph(n);
    vector<int> inDegree(n, 0);

    for (auto& [u, v] : edges) {
        graph[u].push_back(v);
        inDegree[v]++;
    }

    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    vector<int> result;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : graph[node]) {
            if (--inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    if (result.size() != n) {
        return {};  // 사이클 존재
    }

    return result;
}
```

---

## 3. DFS 기반 위상 정렬

### 3.1 알고리즘 원리

```
1. 모든 정점에 대해 DFS 수행
2. DFS가 끝나는 순서대로 스택에 삽입
3. 스택을 역순으로 출력

아이디어:
- DFS에서 정점 v의 탐색이 끝나면 v에서 갈 수 있는 모든 정점은 이미 처리됨
- 따라서 스택에 늦게 들어간 정점이 위상 순서상 먼저 와야 함

시각화:
    A → B → D
    ↓
    C

DFS 방문 순서: A → B → D(완료) → B(완료) → C(완료) → A(완료)
스택 (완료 순): [D, B, C, A]
결과 (역순): [A, C, B, D] 또는 [A, B, D, C]
```

### 3.2 구현

```python
def topological_sort_dfs(n, edges):
    """
    DFS 기반 위상 정렬
    시간: O(V + E)
    """
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    # 3색 표시: 미방문(0) → 방문 중(1) → 완료(2)
    # 2개가 아닌 3개 상태를 사용하면 사이클 감지가 가능하다 —
    # '방문 중' 노드를 다시 만나면 아직 DFS 경로 위에 있다는 뜻(사이클)
    visited = [0] * n
    result = []
    has_cycle = False

    def dfs(node):
        nonlocal has_cycle
        if has_cycle:
            return  # 사이클이 발견되면 즉시 중단

        visited[node] = 1  # 방문 중: 이 노드는 현재 DFS 경로 위에 있음

        for neighbor in graph[node]:
            if visited[neighbor] == 1:
                # 재귀 스택에 있는 노드에 도달 — 역방향 간선(back-edge) = 사이클
                has_cycle = True
                return
            if visited[neighbor] == 0:
                dfs(neighbor)

        visited[node] = 2  # 완료: 모든 후손이 끝남 — 추가해도 안전
        # 모든 후손 이후에 추가하면 u→v일 때 v가 u보다 먼저 추가된다;
        # 마지막에 역순으로 뒤집으면 올바른 위상 순서가 된다
        result.append(node)

    for i in range(n):
        if visited[i] == 0:
            dfs(i)

    if has_cycle:
        return []

    return result[::-1]  # 역순: 마지막에 끝난 노드가 위상 순서에서 먼저 온다


# 예시
n = 6
edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
print(topological_sort_dfs(n, edges))  # [5, 4, 2, 3, 1, 0] 또는 다른 유효한 순서
```

### 3.3 스택 사용 (비재귀)

```python
def topological_sort_iterative(n, edges):
    """반복적 DFS 위상 정렬"""
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    visited = [False] * n
    result = []

    for start in range(n):
        if visited[start]:
            continue

        # 각 스택 항목은 'processed' 플래그를 가지며 재귀 DFS의 두 단계를 시뮬레이션한다:
        # 첫 번째 pop은 자식 탐색을 시작하고,
        # 두 번째 pop(자식 반환 후)은 후위 순서 추가를 수행한다
        stack = [(start, False)]  # (노드, 처리 완료 여부)

        while stack:
            node, processed = stack.pop()

            if processed:
                # 후위 순서: 모든 후손이 방문됨 — 노드 추가
                result.append(node)
                continue

            if visited[node]:
                continue  # 다른 경로를 통해 이미 처리됨

            visited[node] = True
            # processed=True로 다시 push하여 서브트리 이후에 이 노드를 기록
            stack.append((node, True))

            # 자식을 processed=False로 push; node의 후위 순서 마커보다
            # 먼저 탐색됨 (LIFO 순서가 DFS 깊이 우선 동작을 구현)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    stack.append((neighbor, False))

    return result[::-1]  # 후위 순서는 역위상 순서
```

---

## 4. 사이클 탐지

### 4.1 Kahn 알고리즘에서의 사이클 탐지

```python
def has_cycle_kahn(n, edges):
    """
    위상 정렬 결과의 길이가 n보다 작으면 사이클 존재
    """
    result = topological_sort_kahn(n, edges)
    return len(result) != n

# 사이클이 있는 경우
edges_with_cycle = [(0, 1), (1, 2), (2, 0)]  # 0 → 1 → 2 → 0
print(has_cycle_kahn(3, edges_with_cycle))  # True
```

### 4.2 DFS 색상 기반 사이클 탐지

```python
def has_cycle_dfs(n, edges):
    """
    DFS 색상 기법으로 사이클 탐지
    WHITE(0): 미방문
    GRAY(1): 방문 중 (재귀 스택에 있음)
    BLACK(2): 완료
    """
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node):
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # 사이클 발견
            if color[neighbor] == WHITE:
                if dfs(neighbor):
                    return True

        color[node] = BLACK
        return False

    for i in range(n):
        if color[i] == WHITE:
            if dfs(i):
                return True

    return False
```

### 4.3 사이클 경로 찾기

```python
def find_cycle(n, edges):
    """사이클이 있다면 사이클 경로 반환"""
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    parent = [-1] * n

    def dfs(node):
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                # 사이클 발견! 경로 복원
                cycle = [neighbor]
                curr = node
                while curr != neighbor:
                    cycle.append(curr)
                    curr = parent[curr]
                cycle.append(neighbor)
                return cycle[::-1]

            if color[neighbor] == WHITE:
                parent[neighbor] = node
                result = dfs(neighbor)
                if result:
                    return result

        color[node] = BLACK
        return None

    for i in range(n):
        if color[i] == WHITE:
            result = dfs(i)
            if result:
                return result

    return None

# 예시
edges = [(0, 1), (1, 2), (2, 3), (3, 1)]  # 1 → 2 → 3 → 1 사이클
print(find_cycle(4, edges))  # [1, 2, 3, 1]
```

---

## 5. 활용 문제

### 5.1 수강 신청 (기본)

```python
def can_finish(num_courses, prerequisites):
    """
    모든 과목을 수강할 수 있는지 확인
    prerequisites[i] = [a, b] : b를 먼저 들어야 a를 들을 수 있음
    """
    result = topological_sort_kahn(num_courses,
                                    [(b, a) for a, b in prerequisites])
    return len(result) == num_courses

# LeetCode 207. Course Schedule
print(can_finish(2, [[1, 0]]))  # True: 0 → 1
print(can_finish(2, [[1, 0], [0, 1]]))  # False: 사이클
```

### 5.2 수강 순서 찾기

```python
def find_order(num_courses, prerequisites):
    """
    가능한 수강 순서 반환
    불가능하면 빈 리스트 반환
    """
    return topological_sort_kahn(num_courses,
                                  [(b, a) for a, b in prerequisites])

# LeetCode 210. Course Schedule II
print(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
# [0, 1, 2, 3] 또는 [0, 2, 1, 3]
```

### 5.3 빌드 순서

```python
def build_order(projects, dependencies):
    """
    프로젝트 빌드 순서 결정
    dependencies: [(a, b), ...] - b를 빌드한 후 a를 빌드
    """
    # 프로젝트 이름 → 인덱스 매핑
    proj_to_idx = {p: i for i, p in enumerate(projects)}
    n = len(projects)

    edges = [(proj_to_idx[b], proj_to_idx[a]) for a, b in dependencies]
    order = topological_sort_kahn(n, edges)

    if not order:
        return None  # 순환 의존성

    return [projects[i] for i in order]

# 예시
projects = ['a', 'b', 'c', 'd', 'e', 'f']
dependencies = [('a', 'd'), ('f', 'b'), ('b', 'd'), ('f', 'a'), ('d', 'c')]
print(build_order(projects, dependencies))  # ['e', 'f', 'c', 'b', 'd', 'a'] 등
```

### 5.4 작업 스케줄링 (가장 빠른 완료 시간)

```python
def earliest_completion(n, edges, times):
    """
    각 작업의 가장 빠른 완료 시간 계산
    edges: [(u, v), ...] u 완료 후 v 시작 가능
    times[i]: 작업 i의 소요 시간
    """
    from collections import deque

    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # earliest[i] = 작업 i의 가장 빠른 시작 시간 (선행 조건이 없는 작업은 0)
    earliest = [0] * n

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    while queue:
        node = queue.popleft()

        for neighbor in graph[node]:
            # 이웃은 모든 선행 조건이 완료된 후에만 시작할 수 있다;
            # 모든 들어오는 경로에서 max를 취해 진정한 가장 빠른 시작 시간을 찾는다
            earliest[neighbor] = max(earliest[neighbor],
                                      earliest[node] + times[node])
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # 모든 선행자가 처리됨 — 가장 빠른 시작 시간이 확정됨
                queue.append(neighbor)

    # 완료 = 시작 + 소요시간; 전체 최대값이 임계 경로(critical path) 길이
    completion = [earliest[i] + times[i] for i in range(n)]
    return completion, max(completion)

# 예시
n = 4
edges = [(0, 2), (1, 2), (2, 3)]
times = [3, 2, 5, 4]
completion, total = earliest_completion(n, edges, times)
print(f"완료 시간: {completion}")  # [3, 2, 8, 12]
print(f"전체 소요: {total}")  # 12
```

### 5.5 사전 순 가장 빠른 위상 순서

```python
import heapq

def lexicographic_topological_sort(n, edges):
    """
    사전 순으로 가장 빠른 위상 순서
    최소 힙 사용
    """
    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # 최소 힙 사용
    heap = []
    for i in range(n):
        if in_degree[i] == 0:
            heapq.heappush(heap, i)

    result = []

    while heap:
        node = heapq.heappop(heap)
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    return result if len(result) == n else []

# 예시
n = 4
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
print(lexicographic_topological_sort(n, edges))  # [0, 1, 2, 3]
```

### 5.6 위상 정렬 경로의 수

```python
def count_topological_sorts(n, edges):
    """
    가능한 위상 정렬의 수 (백트래킹)
    주의: 지수적 시간 복잡도
    """
    graph = [[] for _ in range(n)]
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    count = 0

    def backtrack(path):
        nonlocal count

        if len(path) == n:
            count += 1
            return

        for i in range(n):
            if i not in path and in_degree[i] == 0:
                # 선택
                for neighbor in graph[i]:
                    in_degree[neighbor] -= 1
                path.add(i)

                backtrack(path)

                # 복원
                path.remove(i)
                for neighbor in graph[i]:
                    in_degree[neighbor] += 1

    backtrack(set())
    return count

# 예시 (작은 n에서만 사용)
n = 4
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
print(count_topological_sorts(n, edges))  # 2: [0,1,2,3], [0,2,1,3]
```

### 5.7 Alien Dictionary

```python
def alien_order(words):
    """
    외계어 알파벳 순서 추론
    words: 사전 순으로 정렬된 외계어 단어 목록
    """
    from collections import defaultdict

    # 모든 문자 수집
    chars = set()
    for word in words:
        chars.update(word)

    graph = defaultdict(set)
    in_degree = {c: 0 for c in chars}

    # 인접한 단어들을 비교하여 순서 추론
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # 예외: w1이 w2의 접두사인데 더 길면 불가능
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""

        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # 위상 정렬
    from collections import deque
    queue = deque([c for c in chars if in_degree[c] == 0])
    result = []

    while queue:
        c = queue.popleft()
        result.append(c)

        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(chars):
        return ""  # 사이클 (불가능한 입력)

    return "".join(result)

# LeetCode 269. Alien Dictionary
words = ["wrt", "wrf", "er", "ett", "rftt"]
print(alien_order(words))  # "wertf"
```

---

## 6. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐ | [줄 세우기](https://www.acmicpc.net/problem/2252) | 백준 | 기본 위상정렬 |
| ⭐⭐ | [Course Schedule](https://leetcode.com/problems/course-schedule/) | LeetCode | 사이클 탐지 |
| ⭐⭐ | [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) | LeetCode | 위상정렬 순서 |
| ⭐⭐⭐ | [작업](https://www.acmicpc.net/problem/2056) | 백준 | 최소 시간 |
| ⭐⭐⭐ | [게임 개발](https://www.acmicpc.net/problem/1516) | 백준 | 작업 스케줄링 |
| ⭐⭐⭐ | [문제집](https://www.acmicpc.net/problem/1766) | 백준 | 사전 순 위상정렬 |
| ⭐⭐⭐⭐ | [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) | LeetCode | 순서 추론 |

---

## 알고리즘 비교

```
┌─────────────────┬─────────────┬─────────────┬────────────────────┐
│ 알고리즘         │ 시간        │ 공간        │ 특징                │
├─────────────────┼─────────────┼─────────────┼────────────────────┤
│ Kahn (BFS)      │ O(V + E)    │ O(V)        │ 진입차수 기반       │
│ DFS 기반        │ O(V + E)    │ O(V)        │ 종료 역순          │
├─────────────────┼─────────────┼─────────────┼────────────────────┤
│ 사전 순 (힙)     │ O((V+E)logV)│ O(V)        │ 최소 힙 사용        │
└─────────────────┴─────────────┴─────────────┴────────────────────┘

V = 정점 수, E = 간선 수
```

---

## 다음 단계

- [14_Shortest_Path.md](./14_Shortest_Path.md) - Dijkstra, Bellman-Ford

---

## 참고 자료

- [Topological Sorting](https://cp-algorithms.com/graph/topological-sort.html)
- Introduction to Algorithms (CLRS) - Chapter 22.4
