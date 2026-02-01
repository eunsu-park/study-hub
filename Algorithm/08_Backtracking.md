# 백트래킹 (Backtracking)

## 개요

백트래킹은 해를 찾는 도중 막히면 되돌아가서 다시 해를 찾는 기법입니다. 가지치기(pruning)를 통해 불필요한 탐색을 줄입니다.

---

## 목차

1. [백트래킹 개념](#1-백트래킹-개념)
2. [순열과 조합](#2-순열과-조합)
3. [N-Queens](#3-n-queens)
4. [부분집합](#4-부분집합)
5. [스도쿠](#5-스도쿠)
6. [연습 문제](#6-연습-문제)

---

## 1. 백트래킹 개념

### 기본 원리

```
백트래킹:
1. 해를 하나씩 구성해 나감
2. 조건을 만족하지 않으면 이전 단계로 되돌아감
3. 가지치기로 탐색 공간 축소

DFS + 조건 검사 + 되돌아가기
```

### 상태 공간 트리

```
N=3일 때 순열 탐색:

                    []
         /          |          \
       [1]         [2]         [3]
       / \         / \         / \
    [1,2][1,3] [2,1][2,3] [3,1][3,2]
      |    |     |    |     |    |
   [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]

가지치기 예: 첫 원소가 조건 위반 시 해당 서브트리 전체 스킵
```

### 기본 템플릿

```python
def backtrack(candidate):
    if is_solution(candidate):
        output(candidate)
        return

    for next_choice in choices(candidate):
        if is_valid(next_choice):  # 가지치기
            candidate.append(next_choice)
            backtrack(candidate)
            candidate.pop()  # 되돌리기
```

---

## 2. 순열과 조합

### 2.1 순열 (Permutation)

```
n개 중 r개를 순서 있게 나열
nPr = n! / (n-r)!

[1, 2, 3]의 모든 순열:
[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
```

```cpp
// C++
void permute(vector<int>& nums, int start, vector<vector<int>>& result) {
    if (start == nums.size()) {
        result.push_back(nums);
        return;
    }

    for (int i = start; i < nums.size(); i++) {
        swap(nums[start], nums[i]);
        permute(nums, start + 1, result);
        swap(nums[start], nums[i]);  // 되돌리기
    }
}

vector<vector<int>> permutations(vector<int>& nums) {
    vector<vector<int>> result;
    permute(nums, 0, result);
    return result;
}
```

```python
def permutations(nums):
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result

# 또는 itertools
from itertools import permutations
list(permutations([1, 2, 3]))
```

### 2.2 조합 (Combination)

```
n개 중 r개를 순서 없이 선택
nCr = n! / (r! × (n-r)!)

[1, 2, 3, 4]에서 2개 선택:
[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]
```

```cpp
// C++
void combine(int n, int r, int start, vector<int>& current,
             vector<vector<int>>& result) {
    if (current.size() == r) {
        result.push_back(current);
        return;
    }

    for (int i = start; i <= n; i++) {
        current.push_back(i);
        combine(n, r, i + 1, current, result);
        current.pop_back();
    }
}

vector<vector<int>> combinations(int n, int r) {
    vector<vector<int>> result;
    vector<int> current;
    combine(n, r, 1, current, result);
    return result;
}
```

```python
def combinations(n, r):
    result = []

    def backtrack(start, current):
        if len(current) == r:
            result.append(current[:])
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result

# 또는 itertools
from itertools import combinations
list(combinations([1, 2, 3, 4], 2))
```

### 2.3 중복 순열/조합

```python
# 중복 순열: 같은 원소 여러 번 선택 가능
def permutations_with_repetition(nums, r):
    result = []

    def backtrack(current):
        if len(current) == r:
            result.append(current[:])
            return

        for num in nums:
            current.append(num)
            backtrack(current)
            current.pop()

    backtrack([])
    return result

# 중복 조합
def combinations_with_repetition(nums, r):
    result = []

    def backtrack(start, current):
        if len(current) == r:
            result.append(current[:])
            return

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i, current)  # i+1이 아닌 i
            current.pop()

    backtrack(0, [])
    return result
```

---

## 3. N-Queens

### 문제

```
N×N 체스판에 N개의 퀸을 서로 공격할 수 없게 배치

퀸의 공격 범위: 가로, 세로, 대각선

4×4 예시 (하나의 해):
. Q . .
. . . Q
Q . . .
. . Q .
```

### 알고리즘

```
행 단위로 퀸 배치:
1. 첫 행에 퀸 배치 시도
2. 다음 행에 퀸 배치 (충돌 검사)
3. 충돌하면 백트래킹
4. N개 배치 완료하면 해 출력

충돌 검사:
- 같은 열: cols[col] == True
- 대각선1 (↘): row - col 값이 같음
- 대각선2 (↙): row + col 값이 같음
```

### 구현

```cpp
// C++
class NQueens {
private:
    int n;
    vector<bool> cols, diag1, diag2;
    vector<vector<string>> results;

    void backtrack(int row, vector<int>& queens) {
        if (row == n) {
            results.push_back(generateBoard(queens));
            return;
        }

        for (int col = 0; col < n; col++) {
            if (cols[col] || diag1[row - col + n - 1] || diag2[row + col])
                continue;

            queens[row] = col;
            cols[col] = diag1[row - col + n - 1] = diag2[row + col] = true;

            backtrack(row + 1, queens);

            cols[col] = diag1[row - col + n - 1] = diag2[row + col] = false;
        }
    }

    vector<string> generateBoard(const vector<int>& queens) {
        vector<string> board(n, string(n, '.'));
        for (int i = 0; i < n; i++) {
            board[i][queens[i]] = 'Q';
        }
        return board;
    }

public:
    vector<vector<string>> solveNQueens(int n) {
        this->n = n;
        cols.assign(n, false);
        diag1.assign(2 * n - 1, false);
        diag2.assign(2 * n - 1, false);

        vector<int> queens(n);
        backtrack(0, queens);

        return results;
    }
};
```

```python
def solve_n_queens(n):
    results = []
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row, queens):
        if row == n:
            board = ['.' * q + 'Q' + '.' * (n - q - 1) for q in queens]
            results.append(board)
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1, queens + [col])

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0, [])
    return results

# 해의 개수만 세기
def count_n_queens(n):
    count = 0
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count
```

---

## 4. 부분집합

### 모든 부분집합 생성

```
[1, 2, 3]의 부분집합:
[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]

총 2^n개
```

```cpp
// C++
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;

    function<void(int)> backtrack = [&](int start) {
        result.push_back(current);

        for (int i = start; i < nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(i + 1);
            current.pop_back();
        }
    };

    backtrack(0);
    return result;
}

// 비트마스크 방법
vector<vector<int>> subsetsBitmask(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> result;

    for (int mask = 0; mask < (1 << n); mask++) {
        vector<int> subset;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                subset.push_back(nums[i]);
            }
        }
        result.push_back(subset);
    }

    return result;
}
```

```python
def subsets(nums):
    result = []

    def backtrack(start, current):
        result.append(current[:])

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result

# 비트마스크
def subsets_bitmask(nums):
    n = len(nums)
    result = []

    for mask in range(1 << n):
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        result.append(subset)

    return result
```

### 합이 target인 부분집합

```python
def subset_sum(nums, target):
    result = []

    def backtrack(start, current, current_sum):
        if current_sum == target:
            result.append(current[:])
            return

        if current_sum > target:  # 가지치기
            return

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current, current_sum + nums[i])
            current.pop()

    backtrack(0, [], 0)
    return result
```

---

## 5. 스도쿠

### 문제

```
9×9 격자, 각 행/열/3×3 박스에 1-9가 한 번씩

5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
------+-------+------
8 . . | . 6 . | . . 3
4 . . | 8 . 3 | . . 1
7 . . | . 2 . | . . 6
------+-------+------
. 6 . | . . . | 2 8 .
. . . | 4 1 9 | . . 5
. . . | . 8 . | . 7 9
```

### 구현

```python
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # 행 검사
        if num in board[row]:
            return False

        # 열 검사
        for r in range(9):
            if board[r][col] == num:
                return False

        # 3×3 박스 검사
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False

        return True

    def solve():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(board, row, col, num):
                            board[row][col] = num

                            if solve():
                                return True

                            board[row][col] = '.'  # 백트래킹

                    return False  # 모든 숫자 실패

        return True  # 빈 칸 없음 = 완료

    solve()
```

---

## 6. 연습 문제

### 문제 1: 문자열의 모든 순열

중복 문자가 있을 때 중복 없이 순열 생성

<details>
<summary>정답 코드</summary>

```python
def permute_unique(nums):
    result = []
    nums.sort()

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return

        for i in range(len(remaining)):
            # 중복 스킵
            if i > 0 and remaining[i] == remaining[i-1]:
                continue

            backtrack(current + [remaining[i]],
                     remaining[:i] + remaining[i+1:])

    backtrack([], nums)
    return result
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐ | [N과 M](https://www.acmicpc.net/problem/15649) | 백준 | 순열 |
| ⭐⭐ | [N-Queens](https://www.acmicpc.net/problem/9663) | 백준 | N-Queens |
| ⭐⭐ | [Subsets](https://leetcode.com/problems/subsets/) | LeetCode | 부분집합 |
| ⭐⭐⭐ | [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) | LeetCode | 스도쿠 |
| ⭐⭐⭐ | [Combination Sum](https://leetcode.com/problems/combination-sum/) | LeetCode | 조합 |

---

## 백트래킹 템플릿

```python
def backtrack(state):
    if is_goal(state):
        save_solution(state)
        return

    for choice in get_choices(state):
        if is_valid(choice, state):
            make_choice(state, choice)
            backtrack(state)
            undo_choice(state, choice)
```

---

## 다음 단계

- [09_Trees_and_BST.md](./09_Trees_and_BST.md) - 트리, BST

---

## 참고 자료

- [Backtracking](https://www.geeksforgeeks.org/backtracking-algorithms/)
- Introduction to Algorithms (CLRS) - Backtracking
