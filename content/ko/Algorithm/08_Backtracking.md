# 백트래킹 (Backtracking)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 백트래킹(backtracking) 패러다임을 가지치기(pruning)가 적용된 깊이 우선 탐색(DFS)으로 설명하고, 완전 탐색(brute force) 대비 탐색 공간을 축소하는 방식을 설명할 수 있다
2. 일반적인 백트래킹 템플릿(선택, 탐색, 선택 취소)을 구현하고 순열(permutation)과 조합(combination) 생성에 적용할 수 있다
3. 제약 기반 가지치기를 사용하여 N-Queens 문제를 백트래킹으로 풀고, 상태 공간 트리(state space tree)를 추적할 수 있다
4. 재귀적 백트래킹 접근 방식을 사용하여 주어진 집합의 모든 부분집합(power set, 멱집합)을 생성할 수 있다
5. 전방 검사(forward checking)를 사용하여 유효하지 않은 분기를 조기에 가지치기 하는 스도쿠(Sudoku)와 같은 제약 충족 문제(constraint satisfaction problem)에 백트래킹을 적용할 수 있다
6. 분기 인자(branching factor)와 최대 깊이를 기준으로 백트래킹 솔루션의 시간 복잡도를 분석할 수 있다

---

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
        // 기저 조건: 모든 위치가 고정됨 — 이 순열을 기록
        result.push_back(nums);
        return;
    }

    for (int i = start; i < nums.size(); i++) {
        // nums[i]를 'start' 위치에 교환하여 배치; 별도의 "사용됨" 배열을
        // 할당하지 않아도 됨 — 제자리 교환은 O(1)
        swap(nums[start], nums[i]);
        permute(nums, start + 1, result);
        // 교환을 되돌려 배열을 복원; 이 없이는 다음 반복에서
        // 손상된 배열 상태를 보게 됨
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
            # nums[:]로 스냅샷 생성 — nums를 직접 추가하면
            # 백트래킹 시 변경되는 참조를 저장하게 됨
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            # 교환으로 nums[i]를 start 위치에 "선택"; 제자리 교환은
            # 매 호출마다 새 리스트를 만드는 것 대비 O(1) 공간 유지
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            # 교환 복원으로 "선택 취소" — 원래 순서를 되돌려
            # 루프의 다음 반복이 변경되지 않은 꼬리를 보게 함
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
    diag1 = set()  # row - col: 좌상→우하 대각선에서 같은 값
    diag2 = set()  # row + col: 우상→좌하 대각선에서 같은 값

    def backtrack(row, queens):
        if row == n:
            # N개의 퀸이 충돌 없이 모두 배치됨 — 보드를 생성하고 기록
            board = ['.' * q + 'Q' + '.' * (n - q - 1) for q in queens]
            results.append(board)
            return

        for col in range(n):
            # 가지치기: 세 가지 공격 제약 중 하나라도 해당하면 이 열을 건너뜀;
            # set 사용으로 이전 행을 스캔하지 않고 O(1) 조회 가능
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # 더 깊이 진행하기 전에 열과 양쪽 대각선을 점유 표시
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1, queens + [col])

            # 표시를 해제하여 다음 열 선택이 깨끗한 상태에서 시작하게 함
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
        # 행 검사 — O(9) 스캔; 조기 반환으로 불필요한 검사 방지
        if num in board[row]:
            return False

        # 열 검사 — 위/아래에 같은 숫자가 없는지 확인
        for r in range(9):
            if board[r][col] == num:
                return False

        # 3×3 박스 검사 — 정수 나눗셈으로 어떤 (row,col)이든 박스 원점에 매핑
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
                    # 각 숫자를 시도; is_valid가 트리 하위에서 모순이
                    # 발생하기를 기다리지 않고 유효하지 않은 배치를 조기에 가지치기
                    for num in '123456789':
                        if is_valid(board, row, col, num):
                            board[row][col] = num

                            # 재귀; 하위 트리가 해를 산출하면 True를 위로 전파
                            if solve():
                                return True

                            board[row][col] = '.'  # 백트래킹: 배치를 취소하고 다음 숫자 시도

                    # 유효한 배치 없이 모든 숫자를 소진 — 실패 신호
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
