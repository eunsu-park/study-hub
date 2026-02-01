# 29. 고급 DP 최적화 (Advanced DP Optimization)

## 학습 목표
- Convex Hull Trick (CHT) 이해와 구현
- 분할 정복 최적화 (D&C Optimization)
- Knuth 최적화 적용
- 각 최적화 기법의 적용 조건 판별
- 대표 문제 유형 파악

## 1. 개요

### DP 최적화가 필요한 경우

```
┌─────────────────────────────────────────────────┐
│              DP 점화식 형태                       │
├─────────────────────────────────────────────────┤
│  dp[i] = min(dp[j] + cost(j, i))                │
│          j < i                                   │
│                                                  │
│  기본: O(N²) → 최적화 필요!                      │
└─────────────────────────────────────────────────┘

최적화 기법별 조건:
┌────────────────┬──────────────────────────────────┐
│ CHT            │ cost = a[j] * b[i] 형태          │
│ D&C Opt        │ opt[i-1] ≤ opt[i] (단조성)       │
│ Knuth Opt      │ opt[i][j-1] ≤ opt[i][j] ≤ opt[i+1][j] │
└────────────────┴──────────────────────────────────┘
```

---

## 2. Convex Hull Trick (CHT)

### 적용 조건

점화식이 다음 형태일 때:
```
dp[i] = min/max(dp[j] + a[j] * b[i] + c[j] + d[i])
        j < i

여기서:
- a[j]: j에만 의존
- b[i]: i에만 의존
- c[j]: j에만 의존 (상수 취급)
- d[i]: i에만 의존 (모든 j에 동일)
```

### 핵심 아이디어

각 j에 대해 직선 `y = a[j] * x + (dp[j] + c[j])`를 정의하면, `dp[i]`는 `x = b[i]`에서 이 직선들의 최솟값(또는 최댓값)입니다.

```
    y
    |    /
    |   /    ← 직선들의 하한 envelope
    |  /  \
    | /    \
    |/______\_____ x
        b[i]

CHT = 직선들의 하한(또는 상한) envelope 관리
```

### 기본 구현 (정렬된 경우)

```python
class ConvexHullTrickMin:
    """
    최솟값 쿼리용 CHT
    조건: 직선의 기울기가 단조 감소, 쿼리 x가 단조 증가
    """
    def __init__(self):
        self.lines = []  # (기울기, y절편)
        self.ptr = 0

    def bad(self, l1, l2, l3):
        """l2가 불필요한지 확인"""
        # 교점(l1, l2).x >= 교점(l2, l3).x 면 l2 불필요
        return (l3[1] - l1[1]) * (l1[0] - l2[0]) <= (l2[1] - l1[1]) * (l1[0] - l3[0])

    def add_line(self, m, b):
        """직선 y = mx + b 추가"""
        line = (m, b)
        while len(self.lines) >= 2 and self.bad(self.lines[-2], self.lines[-1], line):
            self.lines.pop()
        self.lines.append(line)

    def query(self, x):
        """x에서의 최솟값"""
        if not self.lines:
            return float('inf')

        # 포인터 이동 (x가 단조 증가할 때)
        while self.ptr < len(self.lines) - 1:
            m1, b1 = self.lines[self.ptr]
            m2, b2 = self.lines[self.ptr + 1]
            if m1 * x + b1 > m2 * x + b2:
                self.ptr += 1
            else:
                break

        m, b = self.lines[self.ptr]
        return m * x + b

# 사용 예시: dp[i] = min(dp[j] + a[j] * b[i])
def solve_with_cht(a, b):
    n = len(a)
    dp = [0] * n
    cht = ConvexHullTrickMin()

    # 첫 번째 직선 추가 (j=0)
    cht.add_line(a[0], dp[0])

    for i in range(1, n):
        dp[i] = cht.query(b[i])
        cht.add_line(a[i], dp[i])

    return dp[n-1]
```

### Li Chao Tree (범용 CHT)

기울기나 쿼리 순서가 정렬되지 않은 경우

```python
class LiChaoTree:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.tree = {}  # 노드별 직선 저장

    def add_line(self, m, b, node=1, lo=None, hi=None):
        """직선 y = mx + b 추가"""
        if lo is None:
            lo, hi = self.lo, self.hi

        if lo > hi:
            return

        mid = (lo + hi) // 2

        if node not in self.tree:
            self.tree[node] = (m, b)
            return

        old_m, old_b = self.tree[node]

        # mid에서 어느 직선이 더 좋은지 비교
        left_better = m * lo + b < old_m * lo + old_b
        mid_better = m * mid + b < old_m * mid + old_b

        if mid_better:
            self.tree[node] = (m, b)
            m, b = old_m, old_b

        if lo == hi:
            return

        # 교차가 어느 쪽에서 일어나는지에 따라 재귀
        if left_better != mid_better:
            self.add_line(m, b, 2 * node, lo, mid)
        else:
            self.add_line(m, b, 2 * node + 1, mid + 1, hi)

    def query(self, x, node=1, lo=None, hi=None):
        """x에서의 최솟값"""
        if lo is None:
            lo, hi = self.lo, self.hi

        if node not in self.tree:
            return float('inf')

        m, b = self.tree[node]
        result = m * x + b

        if lo == hi:
            return result

        mid = (lo + hi) // 2
        if x <= mid:
            result = min(result, self.query(x, 2 * node, lo, mid))
        else:
            result = min(result, self.query(x, 2 * node + 1, mid + 1, hi))

        return result

# 사용 예시
tree = LiChaoTree(-1000000, 1000000)
tree.add_line(2, 1)    # y = 2x + 1
tree.add_line(-1, 5)   # y = -x + 5
print(tree.query(0))   # 1
print(tree.query(3))   # 2 (min of 7, 2)
```

### C++ 구현

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct Line {
    ll m, b;
    ll eval(ll x) { return m * x + b; }
};

class CHT {
private:
    deque<Line> lines;

    bool bad(Line l1, Line l2, Line l3) {
        // l2가 불필요한지 확인
        return (__int128)(l3.b - l1.b) * (l1.m - l2.m) <=
               (__int128)(l2.b - l1.b) * (l1.m - l3.m);
    }

public:
    void addLine(ll m, ll b) {
        Line line = {m, b};
        while (lines.size() >= 2 && bad(lines[lines.size()-2], lines[lines.size()-1], line))
            lines.pop_back();
        lines.push_back(line);
    }

    ll query(ll x) {
        // 이진 탐색 또는 포인터 사용
        int lo = 0, hi = lines.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (lines[mid].eval(x) > lines[mid + 1].eval(x))
                lo = mid + 1;
            else
                hi = mid;
        }
        return lines[lo].eval(x);
    }
};
```

---

## 3. 분할 정복 최적화 (D&C Optimization)

### 적용 조건

```
dp[i][j] = min(dp[i-1][k] + cost(k, j))  for k < j
           k

조건: opt[i][j] ≤ opt[i][j+1]
      (최적 분할점의 단조성)

이 조건이 성립하는 경우:
- cost 함수가 사각 부등식 (Quadrangle Inequality)을 만족
- cost(a, c) + cost(b, d) ≤ cost(a, d) + cost(b, c)  (a ≤ b ≤ c ≤ d)
```

### 알고리즘

```
1. dp[i][mid]의 최적 k를 찾음
2. dp[i][lo..mid-1]은 k_lo..k_mid 범위에서만 탐색
3. dp[i][mid+1..hi]는 k_mid..k_hi 범위에서만 탐색
4. O(N² / 레벨) = O(N log N) per row → O(KN log N) total
```

### 구현

```python
def dnc_optimization(n, k, cost):
    """
    dp[k][n] 계산 (k개의 그룹으로 n개 원소 분할)
    cost(i, j): 구간 [i, j)의 비용
    """
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0

    def compute(row, dp_lo, dp_hi, opt_lo, opt_hi):
        if dp_lo > dp_hi:
            return

        dp_mid = (dp_lo + dp_hi) // 2
        best_cost = INF
        best_opt = opt_lo

        for opt in range(opt_lo, min(opt_hi, dp_mid) + 1):
            current_cost = dp[row - 1][opt] + cost(opt, dp_mid)
            if current_cost < best_cost:
                best_cost = current_cost
                best_opt = opt

        dp[row][dp_mid] = best_cost

        # 분할 정복
        compute(row, dp_lo, dp_mid - 1, opt_lo, best_opt)
        compute(row, dp_mid + 1, dp_hi, best_opt, opt_hi)

    for row in range(1, k + 1):
        compute(row, 1, n, 0, n - 1)

    return dp[k][n]

# 예: 배열을 k개 구간으로 나눠 비용 최소화
def solve():
    arr = [1, 3, 2, 4, 5, 2]
    n = len(arr)
    k = 3

    # 전처리: prefix sum으로 구간 합 빠르게 계산
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    def range_sum(i, j):
        return prefix[j] - prefix[i]

    # cost(i, j) = (구간 [i, j)의 합)²
    def cost(i, j):
        s = range_sum(i, j)
        return s * s

    result = dnc_optimization(n, k, cost)
    print(f"최소 비용: {result}")
```

### C++ 구현

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

int n, k;
vector<ll> arr, prefix;
vector<vector<ll>> dp;

ll cost(int i, int j) {
    ll sum = prefix[j] - prefix[i];
    return sum * sum;
}

void compute(int row, int lo, int hi, int opt_lo, int opt_hi) {
    if (lo > hi) return;

    int mid = (lo + hi) / 2;
    ll best = LLONG_MAX;
    int best_opt = opt_lo;

    for (int opt = opt_lo; opt <= min(opt_hi, mid - 1); opt++) {
        ll val = dp[row - 1][opt] + cost(opt, mid);
        if (val < best) {
            best = val;
            best_opt = opt;
        }
    }

    dp[row][mid] = best;

    compute(row, lo, mid - 1, opt_lo, best_opt);
    compute(row, mid + 1, hi, best_opt, opt_hi);
}

ll solve() {
    dp.assign(k + 1, vector<ll>(n + 1, LLONG_MAX));
    dp[0][0] = 0;

    for (int row = 1; row <= k; row++) {
        compute(row, 1, n, 0, n - 1);
    }

    return dp[k][n];
}
```

---

## 4. Knuth 최적화

### 적용 조건

```
dp[i][j] = min(dp[i][k] + dp[k][j]) + cost(i, j)
           i < k < j

조건:
1. cost가 사각 부등식 만족
   cost(a, c) + cost(b, d) ≤ cost(a, d) + cost(b, c)

2. 단조성
   cost(b, c) ≤ cost(a, d)  when a ≤ b ≤ c ≤ d

결과: opt[i][j-1] ≤ opt[i][j] ≤ opt[i+1][j]
```

### 알고리즘

```
O(N³) → O(N²)

for length in range(2, n+1):
    for i in range(n - length + 1):
        j = i + length
        # k의 범위: opt[i][j-1] ~ opt[i+1][j]
```

### 구현 (최적 이진 탐색 트리)

```python
def optimal_bst(freq):
    """
    최적 이진 탐색 트리 구성 비용
    freq[i]: i번째 키의 탐색 빈도
    """
    n = len(freq)
    INF = float('inf')

    # prefix sum for range cost
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]

    def cost(i, j):
        return prefix[j] - prefix[i]

    # dp[i][j]: 구간 [i, j)의 최소 비용
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    opt = [[0] * (n + 1) for _ in range(n + 1)]

    # 길이 1
    for i in range(n):
        dp[i][i + 1] = freq[i]
        opt[i][i + 1] = i

    # 길이 2 이상
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length
            dp[i][j] = INF

            # Knuth 최적화: opt[i][j-1] ~ opt[i+1][j]
            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 <= n and j <= n else j - 1

            for k in range(lo, min(hi, j - 1) + 1):
                val = dp[i][k] + dp[k + 1][j] + cost(i, j)
                if val < dp[i][j]:
                    dp[i][j] = val
                    opt[i][j] = k

    return dp[0][n]

# 사용 예시
frequencies = [25, 10, 20, 5, 15, 25]
print(f"최적 BST 비용: {optimal_bst(frequencies)}")
```

### 대표 문제: 행렬 체인 곱셈

```python
def matrix_chain_multiplication(dims):
    """
    행렬 체인 곱셈의 최소 연산 횟수
    dims[i]: i번째 행렬의 행 수 (마지막은 열 수)
    """
    n = len(dims) - 1
    INF = float('inf')

    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    for i in range(n):
        opt[i][i] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            lo = opt[i][j - 1] if j > i else i
            hi = opt[i + 1][j] if i + 1 < n else j

            for k in range(lo, hi + 1):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k

    return dp[0][n - 1]

# 예: 4개 행렬 (10x30, 30x5, 5x60, 60x10)
print(matrix_chain_multiplication([10, 30, 5, 60, 10]))  # 4200
```

---

## 5. 적용 조건 판별 가이드

### 결정 트리

```
점화식: dp[i] = min(dp[j] + cost(j, i))

cost가 a[j] * b[i] 형태인가?
├── Yes → CHT 사용
│         ├── 기울기 정렬됨? → 스택/덱 CHT
│         └── 정렬 안됨? → Li Chao Tree
└── No
    ↓
opt[i][j] 단조성이 있는가?
├── Yes → D&C 최적화 또는 Knuth 최적화
└── No → 일반 DP (O(N²))
```

### 사각 부등식 확인

```python
def check_quadrangle_inequality(cost, n):
    """
    cost(a, c) + cost(b, d) ≤ cost(a, d) + cost(b, c)
    for all a ≤ b ≤ c ≤ d
    """
    for a in range(n):
        for b in range(a, n):
            for c in range(b, n):
                for d in range(c, n):
                    lhs = cost(a, c) + cost(b, d)
                    rhs = cost(a, d) + cost(b, c)
                    if lhs > rhs:
                        return False
    return True
```

---

## 6. 시간 복잡도 정리

| 기법 | 시간 복잡도 | 적용 조건 |
|------|------------|----------|
| 기본 DP | O(N²) 또는 O(N³) | - |
| CHT (정렬) | O(N) | cost = a[j] * b[i] |
| CHT (일반) | O(N log N) | cost = a[j] * b[i] |
| Li Chao Tree | O(N log N) | cost = a[j] * b[i] |
| D&C 최적화 | O(KN log N) | opt 단조성 |
| Knuth 최적화 | O(N²) | 사각 부등식 |

---

## 7. 대표 문제

### 문제 1: 특별한 우체국 (CHT)

```python
def special_post_office(villages, costs):
    """
    우체국 설치 비용 최소화
    dp[i] = min(dp[j] + cost[j] * dist[i] + ...)
    """
    n = len(villages)
    cht = ConvexHullTrickMin()

    dp = [0] * (n + 1)
    cht.add_line(costs[0], dp[0])

    for i in range(1, n + 1):
        dp[i] = cht.query(villages[i - 1])
        if i < n:
            cht.add_line(costs[i], dp[i])

    return dp[n]
```

### 문제 2: 구간 분할 (D&C)

```python
def partition_array(arr, k):
    """
    배열을 k개 구간으로 분할하여 각 구간 합의 제곱 최소화
    """
    return dnc_optimization(len(arr), k, lambda i, j: sum(arr[i:j])**2)
```

### 문제 3: 파일 합치기 (Knuth)

```python
def merge_files(sizes):
    """
    연속된 파일을 합칠 때 최소 비용
    """
    n = len(sizes)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + sizes[i]

    dp = [[0] * (n + 1) for _ in range(n + 1)]
    opt = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        opt[i][i + 1] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length
            dp[i][j] = float('inf')

            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 < n else j - 1

            for k in range(lo, hi + 1):
                val = dp[i][k] + dp[k][j] + prefix[j] - prefix[i]
                if val < dp[i][j]:
                    dp[i][j] = val
                    opt[i][j] = k

    return dp[0][n]
```

---

## 8. 자주 하는 실수

### 실수 1: CHT 기울기 순서

```python
# 최솟값 CHT: 기울기 단조 감소
# 최댓값 CHT: 기울기 단조 증가

# 순서가 맞지 않으면 Li Chao Tree 사용
```

### 실수 2: 범위 체크

```python
# D&C 최적화에서
for opt in range(opt_lo, min(opt_hi, dp_mid) + 1):
    #                    ^^^ dp_mid보다 작아야 함
```

### 실수 3: 정수 오버플로

```cpp
// cost = a[j] * b[i]에서 long long 필요
typedef long long ll;
ll cost = (ll)a[j] * b[i];
```

---

## 9. 연습 문제

| 난이도 | 문제 유형 | 핵심 개념 |
|--------|----------|-----------|
| ★★★ | 특공대 (BOJ 4008) | CHT |
| ★★★ | 나무 자르기 | CHT |
| ★★★★ | 탈옥 | D&C 최적화 |
| ★★★★ | 파일 합치기 | Knuth 최적화 |
| ★★★★★ | IOI 문제들 | 복합 |

---

## 10. 학습 로드맵

```
1. 기본 DP 숙달
   ↓
2. CHT 이해 (직선 관리)
   ↓
3. D&C 최적화 (단조성 활용)
   ↓
4. Knuth 최적화 (사각 부등식)
   ↓
5. 복합 문제 풀이
```

---

## 마무리

이것으로 알고리즘 학습 자료 30개 레슨이 완료되었습니다!

### 전체 레슨 요약

| 범위 | 주제 |
|------|------|
| 01-05 | 기초 (복잡도, 배열, 문자열, 재귀, 정렬) |
| 06-10 | 핵심 (이분탐색, 스택/큐, 트리, 힙, 그래프) |
| 11-15 | 심화 (DFS/BFS, 최단경로, MST, DP, 실전) |
| 16-20 | 인터뷰 필수 (해시, 문자열, 수학, 위상정렬, 비트마스크) |
| 21-25 | 고급 자료구조/그래프 (세그트리, 트라이, 펜윅, SCC, 플로우) |
| 26-29 | 특수 주제 (LCA, 기하, 게임이론, DP최적화) |

---

## 학습 점검

1. CHT가 적용 가능한 점화식 형태는?
2. D&C 최적화의 opt 단조성이란?
3. Knuth 최적화에서 사각 부등식의 역할은?
4. Li Chao Tree가 필요한 경우는?
