# 동적 프로그래밍 (Dynamic Programming)

## 개요

동적 프로그래밍(DP)은 복잡한 문제를 간단한 하위 문제로 나누어 해결하는 알고리즘 설계 기법입니다. 중복되는 하위 문제의 결과를 저장하여 효율성을 높입니다.

---

## 목차

1. [DP 개념](#1-dp-개념)
2. [메모이제이션 vs 타뷸레이션](#2-메모이제이션-vs-타뷸레이션)
3. [기초 DP 문제](#3-기초-dp-문제)
4. [1D DP](#4-1d-dp)
5. [2D DP](#5-2d-dp)
6. [문자열 DP](#6-문자열-dp)
7. [연습 문제](#7-연습-문제)

---

## 1. DP 개념

### DP의 조건

```
동적 프로그래밍 적용 조건:

1. 최적 부분 구조 (Optimal Substructure)
   - 문제의 최적 해가 부분 문제의 최적 해로 구성됨
   - 예: 최단 경로의 부분 경로도 최단 경로

2. 중복 부분 문제 (Overlapping Subproblems)
   - 같은 부분 문제가 여러 번 반복됨
   - 예: 피보나치에서 fib(3)이 여러 번 계산됨
```

### DP vs 분할 정복

```
┌────────────────┬─────────────────┬─────────────────┐
│                │ DP              │ 분할 정복       │
├────────────────┼─────────────────┼─────────────────┤
│ 부분 문제 중복 │ 있음 (저장)     │ 없음            │
│ 계산 방식      │ 저장 후 재사용  │ 독립적 계산     │
│ 예시           │ 피보나치, LCS   │ 병합정렬, 퀵정렬│
└────────────────┴─────────────────┴─────────────────┘
```

### 피보나치로 이해하기

```
피보나치: fib(n) = fib(n-1) + fib(n-2)

일반 재귀 (중복 계산):
                  fib(5)
                 /      \
            fib(4)      fib(3)
           /    \       /    \
       fib(3)  fib(2) fib(2) fib(1)
       /   \
   fib(2) fib(1)

→ fib(3) 2번, fib(2) 3번 계산
→ 시간 복잡도: O(2^n)

DP (저장 후 재사용):
fib(1)=1 → fib(2)=1 → fib(3)=2 → fib(4)=3 → fib(5)=5

→ 각 값 1번씩만 계산
→ 시간 복잡도: O(n)
```

---

## 2. 메모이제이션 vs 타뷸레이션

### 메모이제이션 (Top-Down)

```
위에서 아래로: 재귀 + 캐싱

특징:
- 필요한 부분만 계산 (Lazy Evaluation)
- 재귀 사용 (스택 오버플로우 주의)
- 직관적인 점화식 구현
```

```cpp
// C++ - 메모이제이션
int memo[100];

int fib(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];

    memo[n] = fib(n-1) + fib(n-2);
    return memo[n];
}
```

```python
# Python - 메모이제이션
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

# 또는 딕셔너리 사용
def fib_memo(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

### 타뷸레이션 (Bottom-Up)

```
아래에서 위로: 반복문 + 테이블

특징:
- 모든 부분 문제 계산
- 반복문 사용 (스택 오버플로우 없음)
- 공간 최적화 가능
```

```c
// C - 타뷸레이션
int fib(int n) {
    if (n <= 1) return n;

    int dp[n + 1];
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }

    return dp[n];
}

// 공간 최적화: O(1)
int fibOptimized(int n) {
    if (n <= 1) return n;

    int prev2 = 0, prev1 = 1;

    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}
```

```python
# Python - 타뷸레이션
def fib(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# 공간 최적화
def fib_optimized(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1

    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2

    return prev1
```

---

## 3. 기초 DP 문제

### 3.1 계단 오르기

```
문제: n개의 계단, 한 번에 1칸 또는 2칸 오를 수 있음
      n번째 계단에 도달하는 방법의 수는?

점화식: dp[i] = dp[i-1] + dp[i-2]
- dp[i-1]: 한 칸 전에서 1칸 오르기
- dp[i-2]: 두 칸 전에서 2칸 오르기

예: n=4
dp[1]=1, dp[2]=2, dp[3]=3, dp[4]=5
```

```python
def climb_stairs(n):
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2

    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]
```

### 3.2 동전 거스름돈 (최소 개수)

```
문제: 동전 종류가 주어질 때, 금액을 만드는 최소 동전 개수

동전: [1, 3, 4], 금액: 6

dp[i] = 금액 i를 만드는 최소 동전 개수

dp[0]=0
dp[1]=min(dp[0]+1)=1                    (1원 1개)
dp[2]=min(dp[1]+1)=2                    (1원 2개)
dp[3]=min(dp[2]+1, dp[0]+1)=1           (3원 1개)
dp[4]=min(dp[3]+1, dp[1]+1, dp[0]+1)=1  (4원 1개)
dp[5]=min(dp[4]+1, dp[2]+1, dp[1]+1)=2  (4+1 또는 3+1+1)
dp[6]=min(dp[5]+1, dp[3]+1, dp[2]+1)=2  (3+3)
```

```cpp
// C++
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);  // 불가능한 값으로 초기화
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }

    return dp[amount] > amount ? -1 : dp[amount];
}
```

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

### 3.3 동전 거스름돈 (경우의 수)

```
문제: 동전 종류가 주어질 때, 금액을 만드는 경우의 수

동전: [1, 2, 5], 금액: 5

점화식: dp[i] += dp[i - coin]
주의: 순서를 고려하지 않으므로 동전 종류별로 순회

dp 초기: [1, 0, 0, 0, 0, 0]

coin=1: [1, 1, 1, 1, 1, 1]
coin=2: [1, 1, 2, 2, 3, 3]
coin=5: [1, 1, 2, 2, 3, 4]

답: 4가지 (1+1+1+1+1, 1+1+1+2, 1+2+2, 5)
```

```python
def coin_combinations(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]
```

---

## 4. 1D DP

### 4.1 최대 부분 배열 합 (Kadane's Algorithm)

```
문제: 연속된 부분 배열의 최대 합

배열: [-2, 1, -3, 4, -1, 2, 1, -5, 4]

dp[i] = i에서 끝나는 최대 부분 배열 합
dp[i] = max(arr[i], dp[i-1] + arr[i])

dp: [-2, 1, -2, 4, 3, 5, 6, 1, 5]

최대값: 6 (부분 배열: [4, -1, 2, 1])
```

```cpp
// C++
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];

    for (int i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]);
        maxSum = max(maxSum, currentSum);
    }

    return maxSum;
}
```

```python
def max_sub_array(nums):
    max_sum = current_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum
```

### 4.2 집 도둑 (House Robber)

```
문제: 인접한 집은 털 수 없음, 최대 금액은?

금액: [2, 7, 9, 3, 1]

dp[i] = i번째 집까지 고려했을 때 최대 금액
dp[i] = max(dp[i-1], dp[i-2] + arr[i])
- dp[i-1]: i번째 집 안 털기
- dp[i-2] + arr[i]: i번째 집 털기

dp[0]=2
dp[1]=max(2, 7)=7
dp[2]=max(7, 2+9)=11
dp[3]=max(11, 7+3)=11
dp[4]=max(11, 11+1)=12

답: 12 (2 + 9 + 1)
```

```cpp
// C++
int rob(vector<int>& nums) {
    if (nums.empty()) return 0;
    if (nums.size() == 1) return nums[0];

    int prev2 = 0, prev1 = 0;

    for (int num : nums) {
        int curr = max(prev1, prev2 + num);
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}
```

```python
def rob(nums):
    if not nums:
        return 0

    prev2, prev1 = 0, 0

    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)

    return prev1
```

### 4.3 최장 증가 부분 수열 (LIS)

```
문제: 가장 긴 증가하는 부분 수열의 길이

배열: [10, 9, 2, 5, 3, 7, 101, 18]

dp[i] = i에서 끝나는 LIS 길이
dp[i] = max(dp[j] + 1) for all j < i where arr[j] < arr[i]

dp: [1, 1, 1, 2, 2, 3, 4, 4]

답: 4 (예: [2, 3, 7, 101] 또는 [2, 5, 7, 101])
```

```cpp
// C++ - O(n²)
int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}

// C++ - O(n log n) 이진 탐색
int lengthOfLISFast(vector<int>& nums) {
    vector<int> tails;

    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }

    return tails.size();
}
```

```python
import bisect

def length_of_lis(nums):
    # O(n log n)
    tails = []

    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)
```

---

## 5. 2D DP

### 5.1 0/1 배낭 문제 (Knapsack)

```
문제: 배낭 용량 W, 각 물건의 무게와 가치가 주어질 때
     배낭에 담을 수 있는 최대 가치는?

물건: [(무게, 가치)] = [(2,3), (3,4), (4,5), (5,6)]
용량: W = 5

dp[i][w] = i번째 물건까지 고려, 용량 w일 때 최대 가치

        w=0  1   2   3   4   5
i=0      0   0   3   3   3   3   (물건1: 무게2, 가치3)
i=1      0   0   3   4   4   7   (물건2: 무게3, 가치4)
i=2      0   0   3   4   5   7   (물건3: 무게4, 가치5)
i=3      0   0   3   4   5   7   (물건4: 무게5, 가치6)

답: 7 (물건1 + 물건2: 무게 2+3=5, 가치 3+4=7)
```

```cpp
// C++
int knapsack(int W, vector<int>& weights, vector<int>& values) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            // 물건 i-1을 넣지 않는 경우
            dp[i][w] = dp[i-1][w];

            // 물건 i-1을 넣는 경우
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i][w],
                               dp[i-1][w - weights[i-1]] + values[i-1]);
            }
        }
    }

    return dp[n][W];
}

// 공간 최적화: O(W)
int knapsackOptimized(int W, vector<int>& weights, vector<int>& values) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {  // 역순!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    return dp[W];
}
```

```python
def knapsack(W, weights, values):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][W]
```

### 5.2 그리드 경로

```
문제: m×n 그리드에서 좌상단→우하단 경로 수
     오른쪽, 아래로만 이동 가능

점화식: dp[i][j] = dp[i-1][j] + dp[i][j-1]

3×3 그리드:
1 1 1
1 2 3
1 3 6

답: 6
```

```cpp
// C++
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1));

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }

    return dp[m-1][n-1];
}
```

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]
```

### 5.3 최소 경로 합

```
문제: 그리드의 각 칸에 비용이 있을 때, 최소 비용 경로

그리드:
1 3 1
1 5 1
4 2 1

dp[i][j] = (i,j)까지의 최소 비용
dp:
1  4  5
2  7  6
6  8  7

답: 7 (1→3→1→1→1)
```

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    dp[0][0] = grid[0][0]

    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[m-1][n-1]
```

---

## 6. 문자열 DP

### 6.1 최장 공통 부분 수열 (LCS)

```
문제: 두 문자열의 최장 공통 부분 수열 길이

s1 = "ABCDGH"
s2 = "AEDFHR"

dp[i][j] = s1[0..i-1]과 s2[0..j-1]의 LCS 길이

    ""  A  E  D  F  H  R
""   0  0  0  0  0  0  0
A    0  1  1  1  1  1  1
B    0  1  1  1  1  1  1
C    0  1  1  1  1  1  1
D    0  1  1  2  2  2  2
G    0  1  1  2  2  2  2
H    0  1  1  2  2  3  3

답: 3 (ADH)
```

```cpp
// C++
int longestCommonSubsequence(string s1, string s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}
```

```python
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

### 6.2 편집 거리 (Edit Distance)

```
문제: 한 문자열을 다른 문자열로 변환하는 최소 연산 수
     연산: 삽입, 삭제, 교체

s1 = "horse"
s2 = "ros"

dp[i][j] = s1[0..i-1]을 s2[0..j-1]로 변환하는 최소 연산

    ""  r  o  s
""   0  1  2  3
h    1  1  2  3
o    2  2  1  2
r    3  2  2  2
s    4  3  3  2
e    5  4  4  3

답: 3 (horse → rorse → rose → ros)
```

```cpp
// C++
int minDistance(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({dp[i-1][j],      // 삭제
                                    dp[i][j-1],      // 삽입
                                    dp[i-1][j-1]});  // 교체
            }
        }
    }

    return dp[m][n];
}
```

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # 삭제
                                   dp[i][j-1],      # 삽입
                                   dp[i-1][j-1])    # 교체

    return dp[m][n]
```

---

## 7. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) | LeetCode | 기초 |
| ⭐ | [피보나치 함수](https://www.acmicpc.net/problem/1003) | 백준 | 기초 |
| ⭐⭐ | [Coin Change](https://leetcode.com/problems/coin-change/) | LeetCode | 동전 |
| ⭐⭐ | [LCS](https://www.acmicpc.net/problem/9251) | 백준 | 문자열 |
| ⭐⭐ | [0/1 Knapsack](https://www.acmicpc.net/problem/12865) | 백준 | 배낭 |
| ⭐⭐⭐ | [LIS](https://www.acmicpc.net/problem/11053) | 백준 | LIS |
| ⭐⭐⭐ | [Edit Distance](https://leetcode.com/problems/edit-distance/) | LeetCode | 문자열 |

---

## DP 문제 접근법

```
1. 상태 정의: dp[i]가 무엇을 의미하는지 정의
2. 점화식 수립: dp[i]와 이전 상태의 관계
3. 초기값 설정: 기저 사례
4. 계산 순서: 의존성에 따른 순서
5. 답 추출: 최종 답이 어디에 있는지
```

---

## 다음 단계

- [19_Greedy_Algorithms.md](./19_Greedy_Algorithms.md) - 탐욕 알고리즘

---

## 참고 자료

- [DP Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
- [VisuAlgo - DP](https://visualgo.net/en/recursion)
- Introduction to Algorithms (CLRS) - Chapter 15
