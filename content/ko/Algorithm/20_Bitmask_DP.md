# 비트마스크 DP (Bitmask Dynamic Programming)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. AND, OR, XOR, 시프트(shift), 비트 설정/해제/토글 등 기본 비트 연산을 수행하고 자주 쓰이는 비트 조작 기법을 적용할 수 있다
2. 작은 집합의 부분집합을 정수 비트마스크(bitmask)로 표현하고 모든 부분집합 또는 부분마스크(submask)를 효율적으로 열거할 수 있다
3. 어떤 원소가 사용되었는지를 인코딩하는 상태를 정의하여 비트마스크 DP(Bitmask DP) 풀이를 설계할 수 있다
4. O(2^n × n) 시간 복잡도로 비트마스크 DP를 이용하여 외판원 문제(TSP, Traveling Salesman Problem)를 구현할 수 있다
5. 상태 전환 시 아이템 부분집합을 추적해야 하는 배정(assignment) 및 매칭(matching) 문제에 비트마스크 DP를 적용할 수 있다

---

## 개요

비트마스크 DP는 집합의 상태를 비트로 표현하여 DP를 수행하는 기법입니다. n이 작을 때(n ≤ 20) 부분집합 문제를 효율적으로 해결할 수 있습니다.

---

## 목차

1. [비트 연산 기초](#1-비트-연산-기초)
2. [부분집합 표현](#2-부분집합-표현)
3. [비트마스크 DP 패턴](#3-비트마스크-dp-패턴)
4. [외판원 문제 (TSP)](#4-외판원-문제-tsp)
5. [활용 문제](#5-활용-문제)
6. [연습 문제](#6-연습-문제)

---

## 1. 비트 연산 기초

### 1.1 기본 비트 연산

```
AND (&): 둘 다 1일 때 1
  1010 & 1100 = 1000

OR (|): 하나라도 1이면 1
  1010 | 1100 = 1110

XOR (^): 다르면 1
  1010 ^ 1100 = 0110

NOT (~): 비트 반전
  ~1010 = 0101 (주의: 실제로는 모든 비트 반전)

Left Shift (<<): 왼쪽으로 이동 (× 2^n)
  1 << 3 = 8 (1000₂)

Right Shift (>>): 오른쪽으로 이동 (÷ 2^n)
  8 >> 2 = 2 (10₂)
```

### 1.2 유용한 비트 트릭

```python
# 1. i번째 비트 확인 (0-indexed)
def check_bit(mask, i):
    return (mask >> i) & 1
    # 또는: return mask & (1 << i) != 0

# 2. i번째 비트 켜기
def set_bit(mask, i):
    return mask | (1 << i)

# 3. i번째 비트 끄기
def clear_bit(mask, i):
    return mask & ~(1 << i)

# 4. i번째 비트 토글
def toggle_bit(mask, i):
    return mask ^ (1 << i)

# 5. 켜진 비트 개수 (popcount)
def count_bits(mask):
    count = 0
    while mask:
        count += mask & 1
        mask >>= 1
    return count
# Python: bin(mask).count('1')
# C++: __builtin_popcount(mask)

# 6. 최하위 켜진 비트 (LSB)
def lowest_bit(mask):
    return mask & -mask  # 또는 mask & (~mask + 1)

# 7. 최하위 켜진 비트 제거
def remove_lowest_bit(mask):
    return mask & (mask - 1)

# 8. 모든 비트 1로 채우기 (n개)
def all_ones(n):
    return (1 << n) - 1

# 예시
mask = 0b1010  # 10
print(check_bit(mask, 1))  # 1 (True)
print(check_bit(mask, 2))  # 0 (False)
print(bin(set_bit(mask, 0)))  # 0b1011
print(bin(clear_bit(mask, 3)))  # 0b10
```

### 1.3 C++ 비트 연산

```cpp
#include <bitset>

// i번째 비트 확인
bool checkBit(int mask, int i) {
    return (mask >> i) & 1;
}

// 켜진 비트 개수
int countBits(int mask) {
    return __builtin_popcount(mask);
}
// long long: __builtin_popcountll(mask)

// 최하위 켜진 비트 위치 (0-indexed)
int lowestBitPos(int mask) {
    return __builtin_ctz(mask);  // count trailing zeros
}

// 최상위 켜진 비트 위치
int highestBitPos(int mask) {
    return 31 - __builtin_clz(mask);  // count leading zeros
}
```

---

## 2. 부분집합 표현

### 2.1 집합을 비트마스크로

```
집합 S = {0, 1, 2, 3, 4}의 부분집합

공집합:      00000 = 0
{0}:         00001 = 1
{1}:         00010 = 2
{0, 1}:      00011 = 3
{2}:         00100 = 4
{0, 2}:      00101 = 5
{0, 1, 2}:   00111 = 7
전체집합:    11111 = 31

mask = 13 = 01101₂ → {0, 2, 3}
```

### 2.2 부분집합 순회

```python
def iterate_subsets(n):
    """크기 n인 집합의 모든 부분집합 순회"""
    for mask in range(1 << n):  # 0 ~ 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        print(f"{mask:0{n}b}: {subset}")

iterate_subsets(3)
# 000: []
# 001: [0]
# 010: [1]
# 011: [0, 1]
# 100: [2]
# 101: [0, 2]
# 110: [1, 2]
# 111: [0, 1, 2]
```

### 2.3 특정 마스크의 부분집합만 순회

```python
def iterate_submasks(mask):
    """
    mask의 모든 부분집합(공집합 제외) 순회
    예: mask = 5 (101) → 5, 4, 1

    (submask - 1) & mask가 작동하는 이유:
    1을 빼면 submask의 최하위 설정 비트가 0이 되고 그보다 낮은 모든 비트가 1이 된다.
    mask와 AND 연산을 하면 원래 mask에 설정되지 않았던 비트가 제거되어,
    부분집합 범위 안에 머물게 된다.
    이 방법은 O(2^popcount(mask)) 단계로 모든 부분마스크를 방문하며 —
    모든 2^n 마스크를 순회하고 필터링하는 것보다 훨씬 효율적이다.
    """
    submask = mask
    while submask > 0:
        print(bin(submask))
        submask = (submask - 1) & mask
    print("0 (empty)")

iterate_submasks(5)  # 101, 100, 001, 0
```

### 2.4 집합 연산

```python
def set_operations(a, b, n):
    """
    집합 연산 (n = 전체 집합 크기)
    """
    print(f"A = {bin(a)}, B = {bin(b)}")

    # 합집합 (Union)
    union = a | b
    print(f"A ∪ B = {bin(union)}")

    # 교집합 (Intersection)
    inter = a & b
    print(f"A ∩ B = {bin(inter)}")

    # 차집합 (A - B)
    diff = a & ~b
    print(f"A - B = {bin(diff)}")

    # 여집합 (Complement)
    full = (1 << n) - 1
    comp_a = full ^ a
    print(f"A' = {bin(comp_a)}")

    # 대칭차 (XOR)
    sym_diff = a ^ b
    print(f"A △ B = {bin(sym_diff)}")

# 예시
set_operations(0b1010, 0b1100, 4)
```

---

## 3. 비트마스크 DP 패턴

### 3.1 기본 패턴

```python
def bitmask_dp_template(n, data):
    """
    비트마스크 DP 기본 템플릿
    상태: dp[mask] = mask 상태에서의 최적값
    """
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0  # 초기 상태

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        for i in range(n):
            if mask & (1 << i):  # i가 이미 선택됨
                continue

            new_mask = mask | (1 << i)
            # 상태 전이
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost(mask, i))

    return dp[(1 << n) - 1]  # 모든 원소 선택
```

### 3.2 2차원 비트마스크 DP

```python
def bitmask_dp_2d(n, start, data):
    """
    dp[mask][i] = mask 상태에서 현재 위치가 i일 때의 최적값
    TSP 등에 사용
    """
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue

            for next_node in range(n):
                if mask & (1 << next_node):
                    continue

                new_mask = mask | (1 << next_node)
                cost = data[last][next_node]
                dp[new_mask][next_node] = min(
                    dp[new_mask][next_node],
                    dp[mask][last] + cost
                )

    return dp
```

---

## 4. 외판원 문제 (TSP)

### 4.1 문제 정의

```
TSP (Traveling Salesman Problem):
- n개의 도시를 모두 방문하고 출발점으로 돌아오는 최소 비용 경로
- 브루트포스: O(n!)
- 비트마스크 DP: O(n² × 2^n)

상태:
dp[mask][i] = mask에 포함된 도시들을 방문하고
              현재 도시 i에 있을 때의 최소 비용

전이:
dp[mask | (1<<j)][j] = min(dp[mask][i] + dist[i][j])
(j가 mask에 없고, i에서 j로 이동)
```

### 4.2 구현

```python
def tsp(dist):
    """
    외판원 문제 - 비트마스크 DP
    dist[i][j] = 도시 i에서 j로 가는 비용
    시간: O(n² × 2^n)
    공간: O(n × 2^n)

    비트마스크를 사용하는 이유: 정확히 어떤 도시들을 방문했는지 추적해야 한다.
    비트마스크는 이를 하나의 정수로 인코딩한다(비트 i = 도시 i 방문),
    O(1) 멤버십 테스트와 집합 연산이 가능하다. 전체 상태 공간은 2^n × n으로
    지수적이지만 n <= 20에서는 다룰 수 있으며, 모든 n! 순열에 대한
    브루트포스보다 훨씬 낫다.
    """
    n = len(dist)
    INF = float('inf')

    # dp[mask][i] = mask에 포함된 도시만 정확히 방문하고 도시 i에서 끝날 때의 최소 비용
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 도시 0만 방문(mask=0b1), 비용 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                # last가 방문 집합에 있어야 함 — 비일관적 상태는 건너뜀
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue  # 이미 방문함 — 재방문 불가
                if dist[last][next_city] == INF:
                    continue

                new_mask = mask | (1 << next_city)
                dp[new_mask][next_city] = min(
                    dp[new_mask][next_city],
                    dp[mask][last] + dist[last][next_city]
                )

    # 모든 도시 방문 후 출발점(0)으로 돌아가기.
    # 왕복 비용은 도시 0으로 돌아가는 마지막 구간을 더한다.
    full_mask = (1 << n) - 1
    result = INF
    for last in range(n):
        if dp[full_mask][last] != INF and dist[last][0] != INF:
            result = min(result, dp[full_mask][last] + dist[last][0])

    return result if result != INF else -1


# 예시
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(tsp(dist))  # 80: 0→1→3→2→0
```

### 4.3 경로 복원

```python
def tsp_with_path(dist):
    """TSP + 경로 복원"""
    n = len(dist)
    INF = float('inf')

    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue
                if dist[last][next_city] == INF:
                    continue

                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]

                if new_cost < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_cost
                    parent[new_mask][next_city] = last

    # 최소 비용 찾기
    full_mask = (1 << n) - 1
    min_cost = INF
    last_city = -1

    for i in range(n):
        if dp[full_mask][i] != INF and dist[i][0] != INF:
            total = dp[full_mask][i] + dist[i][0]
            if total < min_cost:
                min_cost = total
                last_city = i

    if last_city == -1:
        return -1, []

    # 경로 복원
    path = [0]  # 마지막에 0으로 돌아감
    mask = full_mask
    curr = last_city

    while curr != -1:
        path.append(curr)
        prev = parent[mask][curr]
        mask ^= (1 << curr)
        curr = prev

    path.reverse()
    return min_cost, path

# 예시
cost, path = tsp_with_path(dist)
print(f"최소 비용: {cost}")  # 80
print(f"경로: {path}")  # [0, 1, 3, 2, 0]
```

### 4.4 C++ 구현

```cpp
#include <vector>
#include <algorithm>
using namespace std;

const int INF = 1e9;

int tsp(vector<vector<int>>& dist) {
    int n = dist.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    dp[1][0] = 0;

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int last = 0; last < n; last++) {
            if (dp[mask][last] == INF) continue;
            if (!(mask & (1 << last))) continue;

            for (int next = 0; next < n; next++) {
                if (mask & (1 << next)) continue;
                if (dist[last][next] == INF) continue;

                int newMask = mask | (1 << next);
                dp[newMask][next] = min(dp[newMask][next],
                                        dp[mask][last] + dist[last][next]);
            }
        }
    }

    int fullMask = (1 << n) - 1;
    int result = INF;

    for (int last = 0; last < n; last++) {
        if (dp[fullMask][last] != INF && dist[last][0] != INF) {
            result = min(result, dp[fullMask][last] + dist[last][0]);
        }
    }

    return result == INF ? -1 : result;
}
```

---

## 5. 활용 문제

### 5.1 집합 커버 문제

```python
def min_set_cover(n, sets):
    """
    최소 집합 커버: 모든 원소를 커버하는 최소 부분집합 수
    sets[i] = i번째 집합의 원소들 (비트마스크)
    """
    m = len(sets)
    full = (1 << n) - 1

    # dp[mask] = mask를 커버하는 최소 집합 수
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        for s in sets:
            new_mask = mask | s
            dp[new_mask] = min(dp[new_mask], dp[mask] + 1)

    return dp[full] if dp[full] != INF else -1

# 예시
# n=4, 집합들: {0,1}, {1,2}, {2,3}, {0,3}
sets = [0b0011, 0b0110, 0b1100, 0b1001]
print(min_set_cover(4, sets))  # 2
```

### 5.2 할당 문제 (Assignment Problem)

```python
def min_assignment(cost):
    """
    n명의 사람에게 n개의 작업 할당
    cost[i][j] = 사람 i가 작업 j를 할 때 비용
    각 사람은 하나의 작업만 수행
    """
    n = len(cost)
    INF = float('inf')

    # dp[mask] = mask에 해당하는 작업들이 할당된 상태에서 최소 비용
    # 사람 i는 popcount(mask) 이전 사람들이 처리
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        person = bin(mask).count('1')  # 현재 할당할 사람
        if person >= n:
            continue

        for job in range(n):
            if mask & (1 << job):
                continue

            new_mask = mask | (1 << job)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[person][job])

    return dp[(1 << n) - 1]

# 예시
cost = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]
print(min_assignment(cost))  # 13: (0→1, 1→0, 2→2, 3→3) 또는 다른 최적 할당
```

### 5.3 해밀턴 경로

```python
def count_hamiltonian_paths(n, adj):
    """
    해밀턴 경로 개수: 모든 정점을 정확히 한 번 방문하는 경로
    adj[i][j] = True if edge (i, j) exists
    """
    # dp[mask][i] = mask 방문, i에서 끝나는 경로 수
    dp = [[0] * n for _ in range(1 << n)]

    # 시작점 설정
    for i in range(n):
        dp[1 << i][i] = 1

    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask][last] == 0:
                continue

            for next_node in range(n):
                if mask & (1 << next_node):
                    continue
                if not adj[last][next_node]:
                    continue

                new_mask = mask | (1 << next_node)
                dp[new_mask][next_node] += dp[mask][last]

    # 모든 정점 방문한 경로 수
    full = (1 << n) - 1
    return sum(dp[full][i] for i in range(n))

# 예시: 완전 그래프
n = 4
adj = [[i != j for j in range(n)] for i in range(n)]
print(count_hamiltonian_paths(n, adj))  # 24 = 4!
```

### 5.4 부분집합 합

```python
def subset_sum_bitmask(arr, target):
    """
    부분집합 합이 target인 경우의 수
    """
    n = len(arr)
    count = 0

    for mask in range(1 << n):
        total = sum(arr[i] for i in range(n) if mask & (1 << i))
        if total == target:
            count += 1

    return count

# 더 효율적: Meet in the Middle (n이 클 때)
def subset_sum_mitm(arr, target):
    """
    Meet in the Middle: O(2^(n/2))
    """
    n = len(arr)
    mid = n // 2

    # 왼쪽 절반의 모든 부분집합 합
    left_sums = []
    for mask in range(1 << mid):
        total = sum(arr[i] for i in range(mid) if mask & (1 << i))
        left_sums.append(total)

    # 정렬
    left_sums.sort()

    # 오른쪽 절반 + 이분탐색
    from bisect import bisect_left, bisect_right
    count = 0

    for mask in range(1 << (n - mid)):
        total = sum(arr[mid + i] for i in range(n - mid) if mask & (1 << i))
        need = target - total
        # left_sums에서 need의 개수
        count += bisect_right(left_sums, need) - bisect_left(left_sums, need)

    return count

# 예시
arr = [1, 2, 3, 4, 5]
print(subset_sum_bitmask(arr, 10))  # 3: {1,4,5}, {2,3,5}, {1,2,3,4}
```

### 5.5 SOS DP (Sum over Subsets)

```python
def sos_dp(arr):
    """
    SOS DP: 각 mask에 대해 부분집합 합 계산
    F[mask] = sum(A[i]) for all i that is subset of mask
    시간: O(n × 2^n)
    """
    n = len(arr).bit_length()
    N = 1 << n

    # arr을 N 크기로 확장
    F = arr + [0] * (N - len(arr))

    for i in range(n):
        for mask in range(N):
            if mask & (1 << i):
                F[mask] += F[mask ^ (1 << i)]

    return F

# 예시
arr = [1, 2, 3, 4]  # 인덱스: 00, 01, 10, 11
result = sos_dp(arr)
# result[3] = arr[0] + arr[1] + arr[2] + arr[3] = 10 (11의 모든 부분집합)
# result[2] = arr[0] + arr[2] = 4 (10의 부분집합)
```

---

## 6. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐⭐ | [외판원 순회](https://www.acmicpc.net/problem/2098) | 백준 | TSP |
| ⭐⭐⭐ | [발전소](https://www.acmicpc.net/problem/1102) | 백준 | 비트마스크 DP |
| ⭐⭐⭐ | [Shortest Hamilton Path](https://codeforces.com/problemset/problem/8/C) | CF | 해밀턴 경로 |
| ⭐⭐⭐ | [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) | LeetCode | 부분집합 |
| ⭐⭐⭐⭐ | [스티커](https://www.acmicpc.net/problem/1562) | 백준 | 비트마스크 DP |
| ⭐⭐⭐⭐ | [Can I Win](https://leetcode.com/problems/can-i-win/) | LeetCode | 게임 이론 |

---

## 비트마스크 DP 체크리스트

```
□ n이 충분히 작은가? (n ≤ 20)
□ 상태를 집합으로 표현할 수 있는가?
□ 점화식을 세울 수 있는가?
□ 메모리가 충분한가? (2^n × factor)
□ 기저 상태 설정
□ 상태 전이 방향 확인 (작은 → 큰 or 큰 → 작은)
```

---

## 시간/공간 복잡도

```
┌──────────────────────┬─────────────────┬─────────────────┐
│ 문제 유형             │ 시간            │ 공간             │
├──────────────────────┼─────────────────┼─────────────────┤
│ 기본 비트마스크 DP    │ O(n × 2^n)     │ O(2^n)          │
│ TSP (외판원)          │ O(n² × 2^n)    │ O(n × 2^n)      │
│ 할당 문제             │ O(n × 2^n)     │ O(2^n)          │
│ SOS DP               │ O(n × 2^n)     │ O(2^n)          │
│ Meet in the Middle   │ O(2^(n/2))     │ O(2^(n/2))      │
└──────────────────────┴─────────────────┴─────────────────┘
```

---

## 다음 단계

- [21_Math_and_Number_Theory.md](./21_Math_and_Number_Theory.md) - 수학과 정수론

---

## 참고 자료

- [Bitmask DP](https://cp-algorithms.com/algebra/all-submasks.html)
- [SOS DP](https://codeforces.com/blog/entry/45223)
