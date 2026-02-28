# 실전 문제 풀이 (Problem Solving in Practice)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 처음 보는 알고리즘 문제에 대해 이해 → 예제 분석 → 알고리즘 선택 → 구현 → 검증의 5단계 구조적 문제 풀이 프로세스를 적용할 수 있다
2. 입력 크기 제약 조건으로부터 필요한 시간 복잡도를 추정하고 코딩 전에 부적합한 알고리즘을 제거할 수 있다
3. 키워드와 구조적 패턴(예: 최단 경로, 부분집합 열거, 구간 쿼리)으로 문제 유형을 인식하고 적합한 알고리즘에 매핑할 수 있다
4. 제한된 시간 내에서 난이도에 따른 유형별 전략을 선택하여 실행할 수 있다
5. 코딩 테스트나 대회에서 문제 우선순위를 정하고 시간을 효율적으로 배분하여 관리할 수 있다
6. 코딩 인터뷰(coding interview)에서 명확한 사고를 보여주고 올바른 해법을 도출하기 위한 구조적 의사소통 및 코딩 기법을 적용할 수 있다

---

## 개요

코딩 테스트와 알고리즘 대회를 위한 실전 문제 풀이 전략과 유형별 접근법을 다룹니다.

---

## 목차

1. [문제 풀이 프로세스](#1-문제-풀이-프로세스)
2. [유형 판별법](#2-유형-판별법)
3. [난이도별 전략](#3-난이도별-전략)
4. [유형별 핵심 문제](#4-유형별-핵심-문제)
5. [시간 관리 전략](#5-시간-관리-전략)
6. [코딩 인터뷰 팁](#6-코딩-인터뷰-팁)

---

## 1. 문제 풀이 프로세스

### 1.1 5단계 접근법

```
┌─────────────────────────────────────────────────────────┐
│                    문제 풀이 5단계                       │
├─────────────────────────────────────────────────────────┤
│  1. 문제 이해     → 입력/출력/제약조건 파악              │
│  2. 예제 분석     → 손으로 풀어보기, 패턴 발견           │
│  3. 알고리즘 선택  → 유형 판별, 시간복잡도 검증          │
│  4. 구현          → 코드 작성, 엣지 케이스 처리          │
│  5. 검증          → 테스트 케이스, 디버깅                │
└─────────────────────────────────────────────────────────┘
```

### 1.2 시간복잡도 계산법

```
입력 크기 N에 따른 허용 복잡도 (1초 기준):

┌─────────────┬───────────────────┬─────────────────┐
│ 입력 크기    │ 최대 허용 복잡도   │ 적합한 알고리즘  │
├─────────────┼───────────────────┼─────────────────┤
│ N ≤ 10      │ O(N!)             │ 완전탐색, 백트래킹│
│ N ≤ 20      │ O(2^N)            │ 비트마스킹, 백트래킹│
│ N ≤ 500     │ O(N³)             │ 플로이드-워셜    │
│ N ≤ 5,000   │ O(N²)             │ DP, 브루트포스   │
│ N ≤ 100,000 │ O(N log N)        │ 정렬, 이분탐색   │
│ N ≤ 10^7    │ O(N)              │ 투포인터, 해시   │
│ N ≤ 10^18   │ O(log N)          │ 이분탐색, 수학   │
└─────────────┴───────────────────┴─────────────────┘
```

### 1.3 문제 읽기 체크리스트

```python
# 문제 분석 템플릿
def analyze_problem():
    """
    체크리스트:
    [ ] 입력 범위 확인 (N, M의 최대값)
    [ ] 시간 제한 확인 (보통 1~2초)
    [ ] 메모리 제한 확인 (보통 256MB)
    [ ] 특수 케이스 확인 (0, 1, 음수, 빈 입력)
    [ ] 출력 형식 확인 (소수점, 개행, 공백)
    """
    pass
```

---

## 2. 유형 판별법

### 2.1 키워드 기반 판별

```
┌────────────────────────┬────────────────────────────────┐
│ 키워드                  │ 알고리즘                        │
├────────────────────────┼────────────────────────────────┤
│ 최단 거리, 최소 비용     │ BFS, 다익스트라, 플로이드       │
│ 경로의 수, 방법의 수     │ DP, 조합론                     │
│ 최댓값/최솟값 구하기     │ 이분탐색, DP, 그리디           │
│ ~가 가능한가?           │ 이분탐색 (파라메트릭 서치)      │
│ 모든 경우, 순서         │ 백트래킹, 순열                  │
│ 연결, 그룹              │ Union-Find, DFS/BFS            │
│ 구간 합, 누적           │ 프리픽스 합, 세그먼트 트리       │
│ 연속된 부분             │ 슬라이딩 윈도우, 투포인터       │
│ 문자열 매칭             │ KMP, 해시, 트라이               │
└────────────────────────┴────────────────────────────────┘
```

### 2.2 자료구조 선택 가이드

```
┌────────────────────────┬────────────────────────────────┐
│ 필요한 연산             │ 자료구조                        │
├────────────────────────┼────────────────────────────────┤
│ 빠른 삽입/삭제 (앞/뒤)   │ 덱 (Deque)                     │
│ 빠른 검색 (키-값)       │ 해시맵/딕셔너리                 │
│ 정렬된 상태 유지        │ TreeMap, 힙                    │
│ 최대/최소 빠른 접근     │ 힙 (Priority Queue)            │
│ 중복 제거               │ Set, 해시셋                    │
│ 순서 있는 고유값        │ OrderedDict, TreeSet           │
│ 구간 쿼리               │ 세그먼트 트리, 펜윅 트리        │
└────────────────────────┴────────────────────────────────┘
```

### 2.3 문제 유형 결정 트리

```
                    문제 시작
                        │
           ┌────────────┴────────────┐
           │ 최적화 문제인가?         │
           └────────────┬────────────┘
                 ┌──────┴──────┐
                YES            NO
                 │              │
    ┌────────────┴───┐    ┌────┴────┐
    │ 탐욕으로 가능? │    │ 탐색/나열 │
    └────────────┬───┘    └────┬────┘
         ┌───────┴───────┐     │
        YES              NO    │
         │                │     │
      그리디            DP    ┌┴─────────┐
                              │ 모든 경우? │
                              └┬─────────┘
                        ┌──────┴──────┐
                       YES            NO
                        │              │
                   백트래킹        그래프 탐색
                   완전탐색       (DFS/BFS)
```

---

## 3. 난이도별 전략

### 3.1 Easy (브론즈~실버)

```
핵심 포인트:
✓ 문제를 그대로 구현
✓ 기본 자료구조 활용
✓ 시간복잡도 크게 신경 안 써도 됨

주요 유형:
- 단순 구현/시뮬레이션
- 기본 정렬/탐색
- 1차원 DP
- 기본 그래프 탐색

예시 접근:
1. 문제 조건 그대로 코드로 옮기기
2. 예제 케이스 통과 확인
3. 엣지 케이스 (0, 1, 최대값) 테스트
```

```python
# Easy 문제 템플릿 - 두 수의 합
def two_sum_easy(nums, target):
    """
    브루트포스로 충분 (N ≤ 1000)
    O(N²) 허용
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```

### 3.2 Medium (골드)

```
핵심 포인트:
✓ 알고리즘 선택이 중요
✓ 시간복잡도 검증 필수
✓ 최적화 기법 적용

주요 유형:
- 이분탐색 응용
- 그래프 알고리즘 (다익스트라, MST)
- 2차원 DP
- 투포인터/슬라이딩 윈도우
- 트리 DP

예시 접근:
1. 유형 판별 → 알고리즘 선택
2. 시간복잡도 계산 → 가능 여부 확인
3. 구현 → 최적화
```

```python
# Medium 문제 템플릿 - 두 수의 합 (최적화)
def two_sum_medium(nums, target):
    """
    해시맵으로 최적화 (N ≤ 100,000)
    O(N) 필요
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### 3.3 Hard (플래티넘 이상)

```
핵심 포인트:
✓ 여러 알고리즘 조합
✓ 고급 자료구조 필요
✓ 창의적 접근 요구

주요 유형:
- 세그먼트 트리/펜윅 트리
- 고급 그래프 (SCC, 2-SAT)
- 비트마스킹 DP
- 볼록껍질, 기하
- 문자열 고급 (접미사 배열, 매니커)

예시 접근:
1. 문제 분해 → 부분 문제 정의
2. 알려진 알고리즘 적용 가능성 검토
3. 관찰 → 최적화 아이디어 도출
```

---

## 4. 유형별 핵심 문제

### 4.1 배열/문자열

```python
# 유형 1: 슬라이딩 윈도우 - 최대 연속 합
def max_subarray_sum(arr, k):
    """
    크기 k인 부분 배열의 최대 합
    시간: O(N)
    """
    n = len(arr)
    if n < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# 유형 2: 투포인터 - 정렬된 배열에서 두 수의 합
def two_sum_sorted(arr, target):
    """
    정렬된 배열에서 합이 target인 두 수 찾기
    시간: O(N)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1

    return []

# 유형 3: 프리픽스 합 - 구간 합 쿼리
class PrefixSum:
    def __init__(self, arr):
        self.prefix = [0]
        for x in arr:
            self.prefix.append(self.prefix[-1] + x)

    def query(self, l, r):
        """[l, r] 구간 합 (0-indexed)"""
        return self.prefix[r + 1] - self.prefix[l]
```

### 4.2 그래프

```python
from collections import deque
import heapq

# 유형 1: BFS - 최단 거리 (가중치 없음)
def bfs_shortest(graph, start, end):
    """
    가중치 없는 그래프의 최단 거리
    시간: O(V + E)
    BFS가 가중치 없는 최단 경로에 올바른 이유: 홉 수 순서로 노드를 확장하므로 --
    노드에 처음 도달했을 때가 최단 경로를 통한 것임.
    """
    n = len(graph)
    dist = [-1] * n
    dist[start] = 0

    queue = deque([start])

    while queue:
        curr = queue.popleft()

        if curr == end:
            return dist[end]

        for next_node in graph[curr]:
            if dist[next_node] == -1:  # 아직 미방문 -- 첫 도달 = 최단 경로
                dist[next_node] = dist[curr] + 1
                queue.append(next_node)

    return -1  # start에서 end에 도달 불가

# 유형 2: 다익스트라 - 최단 거리 (가중치 있음)
def dijkstra(graph, start):
    """
    가중치 있는 그래프의 최단 거리
    graph: 인접 리스트 [(next, weight), ...]
    시간: O((V + E) log V)
    다익스트라가 동작하는 원리: 현재 가장 가까운 미확정 노드를 항상 확장 --
    모든 간선 가중치가 비음수이므로 탐욕적 선택이 안전함.
    """
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0

    pq = [(0, start)]  # (거리, 노드) 최소 힙 -- 가장 가까운 노드를 먼저 처리

    while pq:
        d, curr = heapq.heappop(pq)

        # 오래된 힙 항목 건너뛰기 -- 노드가 다른 거리로 여러 번 삽입될 수 있으며;
        # 가장 작은 거리(첫 번째 pop)만이 최종
        if d > dist[curr]:
            continue

        for next_node, weight in graph[curr]:
            new_dist = dist[curr] + weight
            if new_dist < dist[next_node]:
                dist[next_node] = new_dist
                heapq.heappush(pq, (new_dist, next_node))

    return dist

# 유형 3: Union-Find - 연결 요소
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # 각 노드가 자기 자신을 루트로 시작
        self.rank = [0] * n           # 랭크(rank)가 합치는 방향을 안내하여 트리를 평평하게 유지

    def find(self, x):
        if self.parent[x] != x:
            # 경로 압축: 경로의 모든 노드가 루트를 직접 가리키게 만들어
            # 이후 find가 거의 O(1)이 됨 -- 이것이 상각 효율성의 핵심
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # 이미 같은 컴포넌트 -- 병합 불필요
        # 랭크 기반 합치기: 짧은 트리를 긴 트리 아래에 붙여 높이를 O(log n)으로 유지
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1  # 같은 랭크의 트리를 합칠 때만 높이가 증가
        return True
```

### 4.3 동적 프로그래밍

```python
# 유형 1: 1차원 DP - 계단 오르기
def climb_stairs(n):
    """
    n개의 계단을 1칸 또는 2칸씩 오르는 방법의 수
    시간: O(N), 공간: O(1)
    피보나치와 동일한 점화식: ways(n) = ways(n-1) + ways(n-2)
    마지막 단계가 1칸 또는 2칸이기 때문.
    """
    if n <= 2:
        return n

    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr  # 윈도우 슬라이드 -- O(N) 배열 대신 O(1) 공간

    return prev1

# 유형 2: 2차원 DP - 0/1 배낭
def knapsack_01(weights, values, capacity):
    """
    용량 제한 내 최대 가치
    시간: O(N * W), 공간: O(W)
    """
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # 역순으로 용량을 순회하여 각 아이템이 최대 한 번만 고려되도록 함.
        # 정방향이면 dp[w - weights[i]]에 이미 아이템 i가 포함되어
        # 같은 아이템을 여러 번 선택하는 것을 잘못 허용함 (무한 배낭 문제가 됨).
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

# 유형 3: 문자열 DP - LCS
def lcs_length(s1, s2):
    """
    최장 공통 부분 수열의 길이
    시간: O(N * M)
    dp[i][j] = s1[0:i]와 s2[0:j]의 LCS 길이.
    문자가 일치하면 이전 쌍에서 LCS를 확장;
    일치하지 않으면 어느 한 문자열에서 한 문자를 빼는 것 중 최선을 취함.
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1  # 문자 일치 -- LCS를 1 확장
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # 한 문자 없이 최선을 취함

    return dp[n][m]

# 유형 4: 구간 DP - 행렬 곱셈 순서
def matrix_chain(dims):
    """
    행렬 곱셈의 최소 연산 횟수
    dims: 행렬 차원 [d0, d1, d2, ...] → (d0×d1) × (d1×d2) × ...
    시간: O(N³)
    증가하는 구간 길이로 채워서 더 작은 부분 문제가 그것에 의존하는
    더 큰 문제보다 먼저 준비되도록 함.
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            # 모든 가능한 분할점 k를 시도; [i..k] × [k+1..j]를 곱하는 비용은
            # 각 서브 체인의 비용 + 최종 곱셈 dims[i]*dims[k+1]*dims[j+1]
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]
```

### 4.4 이분탐색

```python
# 유형 1: 값 찾기 - lower_bound / upper_bound
def lower_bound(arr, target):
    """target 이상인 첫 위치"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def upper_bound(arr, target):
    """target 초과인 첫 위치"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left

# 유형 2: 파라메트릭 서치 - 나무 자르기
def cut_trees(heights, target):
    """
    절단기 높이를 정해 target 이상의 나무를 얻는 최대 높이
    시간: O(N log max(H))
    """
    def can_get(cut_height):
        total = sum(max(0, h - cut_height) for h in heights)
        return total >= target

    left, right = 0, max(heights)
    result = 0

    while left <= right:
        mid = (left + right) // 2
        if can_get(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result

# 유형 3: 이분탐색 + 그리디 - 공유기 설치
def install_routers(houses, n):
    """
    n개의 공유기를 설치할 때 최대 최소 거리
    시간: O(N log D)
    """
    houses.sort()

    def can_install(min_dist):
        count = 1
        last = houses[0]
        for h in houses[1:]:
            if h - last >= min_dist:
                count += 1
                last = h
        return count >= n

    left, right = 1, houses[-1] - houses[0]
    result = 0

    while left <= right:
        mid = (left + right) // 2
        if can_install(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result
```

### 4.5 백트래킹

```python
# 유형 1: 순열 생성
def permutations(nums):
    """모든 순열 생성 - O(N! * N)"""
    result = []
    used = [False] * len(nums)  # 현재 경로에 이미 있는 원소 추적

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])  # 복사 -- path는 그 자리에서 변경되므로 스냅샷이 필요
            return

        for i, num in enumerate(nums):
            if used[i]:
                continue
            used[i] = True   # 선택: 내려가기 전에 사용 표시
            path.append(num)
            backtrack(path)
            path.pop()       # 선택 취소: 다음 반복이 다른 원소를 시도할 수 있도록 되돌림
            used[i] = False

    backtrack([])
    return result

# 유형 2: 조합 생성
def combinations(nums, k):
    """크기 k인 모든 조합 - O(C(N,K) * K)"""
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)  # i+1부터 시작하여 같은 원소 재사용 방지
            path.pop()

    backtrack(0, [])
    return result

# 유형 3: N-Queens
def solve_n_queens(n):
    """N-Queens 해의 개수"""
    count = 0
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)  # row - col + n - 1 (각 \ 대각선마다 고유)
    diag2 = [False] * (2 * n - 1)  # row + col          (각 / 대각선마다 고유)

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1  # n개의 퀸이 충돌 없이 모두 배치됨
            return

        for col in range(n):
            d1 = row - col + n - 1
            d2 = row + col

            # 가지치기: 어떤 제약이든 위반되면 이 열 건너뛰기 -- 이것이
            # 전체 하위 트리를 제거하여 백트래킹이 브루트포스보다 훨씬 빠른 이유
            if cols[col] or diag1[d1] or diag2[d2]:
                continue

            cols[col] = diag1[d1] = diag2[d2] = True
            backtrack(row + 1)
            cols[col] = diag1[d1] = diag2[d2] = False  # 다음 열 시도 전에 되돌리기

    backtrack(0)
    return count
```

---

## 5. 시간 관리 전략

### 5.1 문제 배분 전략

```
┌─────────────────────────────────────────────────────────┐
│              코딩 테스트 시간 배분 (3시간 기준)           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [전체 훑기]     15분   모든 문제 읽기, 난이도 파악       │
│       ↓                                                 │
│  [Easy 문제]    45분   확실히 풀 수 있는 문제 해결        │
│       ↓                                                 │
│  [Medium 문제]  90분   핵심 문제, 부분 점수 노리기        │
│       ↓                                                 │
│  [Hard 문제]    20분   아이디어만이라도 구현              │
│       ↓                                                 │
│  [검토]         10분   런타임 에러, 엣지 케이스 확인      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 문제 우선순위

```
우선순위 결정 기준:

1. 배점 대비 난이도
   - 쉬운 문제 먼저 (확실한 점수 확보)
   - 부분 점수 있으면 어려운 문제도 도전

2. 유형 친숙도
   - 연습한 유형 먼저
   - 새로운 유형은 후순위

3. 시간 제한
   - 시간 많이 걸릴 문제는 나중에
   - 구현량 많은 문제 주의
```

### 5.3 막혔을 때 대처법

```
1. 5분 룰
   - 5분간 진전 없으면 다른 문제로

2. 단순화
   - 입력 크기 줄여서 생각
   - 특수 케이스부터 해결

3. 역으로 생각
   - 출력에서 역추적
   - "이걸 구하려면 뭐가 필요하지?"

4. 패턴 찾기
   - 예제 손으로 따라가기
   - 규칙성 발견

5. 부분 점수
   - 작은 케이스만 해결
   - 브루트포스로라도 제출
```

---

## 6. 코딩 인터뷰 팁

### 6.1 면접 진행 방식

```
┌─────────────────────────────────────────────────────────┐
│                   코딩 인터뷰 단계                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 문제 설명 (5분)                                     │
│     - 면접관이 문제 제시                                │
│     - 이해 안 되면 질문                                 │
│                                                         │
│  2. 접근법 논의 (10분)                                  │
│     - 생각을 말로 설명                                  │
│     - 면접관과 아이디어 교환                            │
│     - 시간/공간 복잡도 언급                             │
│                                                         │
│  3. 코딩 (20-25분)                                      │
│     - 설명하면서 코딩                                   │
│     - 막히면 힌트 요청 OK                               │
│                                                         │
│  4. 테스트 (5분)                                        │
│     - 예제로 손으로 트레이스                            │
│     - 엣지 케이스 논의                                  │
│                                                         │
│  5. 최적화/후속 질문 (5분)                              │
│     - 개선 방법 논의                                    │
│     - 변형 문제 대응                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 6.2 의사소통 전략

```python
# 좋은 예시: 생각 과정 공유

"""
면접관: 배열에서 두 수의 합이 target인 쌍을 찾으세요.

나: 문제를 이해해 보겠습니다.
    - 배열이 주어지고
    - 합이 target이 되는 두 인덱스를 반환
    - 정렬되어 있나요? → (질문)

    먼저 브루트포스로 생각하면 O(N²)이 됩니다.
    모든 쌍을 확인하는 방법인데요.

    더 효율적으로... 해시맵을 쓰면 O(N)이 가능합니다.
    각 숫자를 보면서 target - num이 이미 해시에 있는지 확인하면 됩니다.

    이 접근법이 괜찮을까요? → (확인)
    그럼 코드를 작성해 보겠습니다.
"""
```

### 6.3 자주 묻는 질문 유형

```
1. Two Sum 변형
   - 정렬된 배열 → 투포인터
   - 세 수의 합 → 정렬 + 투포인터
   - 가장 가까운 합 → 정렬 + 투포인터

2. 연결 리스트
   - 사이클 탐지 → 플로이드 알고리즘
   - 중간 노드 → 빠른/느린 포인터
   - 역순 → 반복 또는 재귀

3. 트리
   - 순회 → 재귀/스택
   - 최대 깊이 → DFS
   - LCA → 재귀

4. 그래프
   - 연결 확인 → DFS/BFS
   - 최단 경로 → BFS
   - 사이클 → DFS + 방문 상태

5. 동적 프로그래밍
   - 계단 오르기 → 피보나치
   - 최대 부분 합 → Kadane
   - 동전 거스름돈 → 무한 배낭
```

---

## 추천 문제 (플랫폼별)

### 백준 (BOJ)

| 난이도 | 문제 | 유형 |
|--------|------|------|
| 실버 | [수 찾기 (1920)](https://www.acmicpc.net/problem/1920) | 이분탐색 |
| 실버 | [DFS와 BFS (1260)](https://www.acmicpc.net/problem/1260) | 그래프 탐색 |
| 골드 | [최단경로 (1753)](https://www.acmicpc.net/problem/1753) | 다익스트라 |
| 골드 | [LCS (9251)](https://www.acmicpc.net/problem/9251) | DP |
| 골드 | [N-Queen (9663)](https://www.acmicpc.net/problem/9663) | 백트래킹 |
| 플래 | [최솟값 찾기 (11003)](https://www.acmicpc.net/problem/11003) | 모노톤 덱 |

### LeetCode

| 난이도 | 문제 | 유형 |
|--------|------|------|
| Easy | [Two Sum](https://leetcode.com/problems/two-sum/) | 해시맵 |
| Easy | [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) | 스택 |
| Medium | [3Sum](https://leetcode.com/problems/3sum/) | 투포인터 |
| Medium | [Coin Change](https://leetcode.com/problems/coin-change/) | DP |
| Medium | [Number of Islands](https://leetcode.com/problems/number-of-islands/) | DFS/BFS |
| Hard | [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) | 이분탐색 |

### 프로그래머스

| 레벨 | 문제 | 유형 |
|------|------|------|
| Lv2 | 타겟 넘버 | DFS/BFS |
| Lv2 | 게임 맵 최단거리 | BFS |
| Lv3 | 네트워크 | Union-Find |
| Lv3 | 등굣길 | DP |

---

## 학습 로드맵

```
┌─────────────────────────────────────────────────────────┐
│                    실력 향상 로드맵                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [1개월차] 기초 다지기                                   │
│    - 배열, 문자열, 스택, 큐                              │
│    - 기본 정렬, 이분탐색                                │
│    - 하루 1문제 Easy                                    │
│                                                         │
│  [2개월차] 핵심 알고리즘                                 │
│    - DFS, BFS, 그래프 기초                              │
│    - 1차원 DP                                           │
│    - 하루 1문제 Easy/Medium                             │
│                                                         │
│  [3개월차] 심화 학습                                     │
│    - 다익스트라, 유니온파인드                            │
│    - 2차원 DP, 백트래킹                                 │
│    - 하루 1-2문제 Medium                                │
│                                                         │
│  [4개월차 이후] 실전 연습                                │
│    - 모의고사 (시간 제한)                               │
│    - Hard 문제 도전                                     │
│    - 약점 유형 보완                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 연습 문제

### 종합 문제

| 번호 | 문제 | 난이도 | 힌트 |
|------|------|--------|------|
| 1 | 배열에서 K번째 큰 수 | ⭐⭐ | 힙 또는 퀵셀렉트 |
| 2 | 미로 최단거리 | ⭐⭐ | BFS |
| 3 | 최대 연속 부분합 | ⭐⭐ | Kadane's Algorithm |
| 4 | 단어 변환 | ⭐⭐⭐ | BFS |
| 5 | 가장 긴 증가 부분 수열 | ⭐⭐⭐ | DP + 이분탐색 |

---

## 참고 자료

- [알고리즘 문제 해결 전략 (종만북)](https://book.algospot.com/)
- [LeetCode Patterns](https://seanprashad.com/leetcode-patterns/)
- [Codeforces](https://codeforces.com/) - 실시간 대회
- [AtCoder](https://atcoder.jp/) - 일본 알고리즘 대회

---

## 체크리스트: 코딩 테스트 준비

```
필수 유형 (반드시 풀 수 있어야 함):
□ 이분탐색 - lower_bound, 파라메트릭 서치
□ BFS - 최단거리, 레벨 탐색
□ DFS - 연결요소, 사이클
□ DP - 1차원, 2차원 기초
□ 그리디 - 정렬 후 선택
□ 투포인터 - 합 문제
□ 해시맵 - 빈도수, 중복 체크

중급 유형 (대부분의 테스트에 출제):
□ 다익스트라 - 가중치 최단경로
□ 유니온파인드 - 그룹화
□ 백트래킹 - 순열, 조합
□ 슬라이딩 윈도우 - 연속 구간
□ 트리 순회 - 전/중/후위

고급 유형 (어려운 테스트):
□ 세그먼트 트리 - 구간 쿼리
□ 위상정렬 - 의존성 순서
□ LCA - 공통 조상
□ 비트마스킹 DP - 상태 압축
```

---

## 이전 단계

- [10_Heaps_and_Priority_Queues.md](./10_Heaps_and_Priority_Queues.md) - 힙과 우선순위 큐

## 다음 단계

- [Heavy-Light Decomposition](./30_Heavy_Light_Decomposition.md) - 트리 경로 쿼리 O(log²N)
