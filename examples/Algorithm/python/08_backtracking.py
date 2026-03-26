"""
Backtracking
Backtracking Algorithms

A technique that backtracks when hitting a dead end while searching for a solution.
Enables efficient exhaustive search.
"""

from typing import List


# =============================================================================
# 1. Permutations
# =============================================================================
def permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations of an array
    Time Complexity: O(n! * n)
    """
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i + 1:])
            path.pop()

    backtrack([], nums)
    return result


def permutations_inplace(nums: List[int]) -> List[List[int]]:
    """Permutations (swap method)"""
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


# =============================================================================
# 2. Combinations
# =============================================================================
def combinations(n: int, k: int) -> List[List[int]]:
    """
    All combinations of choosing k from 1~n
    Time Complexity: O(C(n,k) * k)
    """
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        # Prune if not enough numbers remain
        need = k - len(path)
        available = n - start + 1
        if available < need:
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find combinations that sum to target (same number can be used multiple times)
    """
    result = []
    candidates.sort()

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # Start from i (allow reuse)
            path.pop()

    backtrack(0, [], target)
    return result


# =============================================================================
# 3. Subsets
# =============================================================================
def subsets(nums: List[int]) -> List[List[int]]:
    """
    All subsets of an array
    Time Complexity: O(2^n * n)
    """
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    """All subsets of an array with duplicate elements"""
    result = []
    nums.sort()

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            # Skip duplicates at the same level
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


# =============================================================================
# 4. N-Queens Problem
# =============================================================================
def solve_n_queens(n: int) -> List[List[str]]:
    """
    N-Queens: Place n queens on an n x n chessboard so they don't attack each other
    """
    result = []
    board = [['.'] * n for _ in range(n)]

    # Sets for column and diagonal checks
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            # Restore
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result


def count_n_queens(n: int) -> int:
    """Count the number of N-Queens solutions"""
    count = [0]
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        if row == n:
            count[0] += 1
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
    return count[0]


# =============================================================================
# 5. Sudoku Solver
# =============================================================================
def solve_sudoku(board: List[List[str]]) -> bool:
    """
    Solve 9x9 Sudoku (in-place)
    Returns True on success
    """
    def is_valid(row, col, num):
        # Row check
        if num in board[row]:
            return False

        # Column check
        for r in range(9):
            if board[r][col] == num:
                return False

        # 3x3 box check
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False

        return True

    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num
                            if backtrack():
                                return True
                            board[row][col] = '.'
                    return False
        return True

    return backtrack()


# =============================================================================
# 6. Word Search
# =============================================================================
def word_search(board: List[List[str]], word: str) -> bool:
    """
    Find a word in a 2D grid (moving only to adjacent cells)
    """
    if not board or not board[0]:
        return False

    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if board[r][c] != word[idx]:
            return False

        # Mark as visited
        temp = board[r][c]
        board[r][c] = '#'

        # Search 4 directions
        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))

        # Restore
        board[r][c] = temp

        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False


# =============================================================================
# 7. Generate Parentheses
# =============================================================================
def generate_parentheses(n: int) -> List[str]:
    """
    Generate all valid combinations of n pairs of parentheses
    """
    result = []

    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return

        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()

        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()

    backtrack([], 0, 0)
    return result


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Backtracking Examples")
    print("=" * 60)

    # 1. Permutations
    print("\n[1] Permutations")
    nums = [1, 2, 3]
    perms = permutations(nums)
    print(f"    Permutations of {nums} ({len(perms)} total):")
    for p in perms[:6]:  # Show first 6
        print(f"    {p}")

    # 2. Combinations
    print("\n[2] Combinations")
    n, k = 4, 2
    combs = combinations(n, k)
    print(f"    C({n}, {k}) = {len(combs)} combinations:")
    print(f"    {combs}")

    print("\n    Combination Sum")
    candidates = [2, 3, 6, 7]
    target = 7
    result = combination_sum(candidates, target)
    print(f"    Combinations from {candidates} summing to {target}: {result}")

    # 3. Subsets
    print("\n[3] Subsets")
    nums = [1, 2, 3]
    subs = subsets(nums)
    print(f"    Subsets of {nums} ({len(subs)} total): {subs}")

    print("\n    Subsets with Duplicates")
    nums_dup = [1, 2, 2]
    subs_dup = subsets_with_dup(nums_dup)
    print(f"    Subsets of {nums_dup}: {subs_dup}")

    # 4. N-Queens
    print("\n[4] N-Queens Problem")
    for n in [4, 8]:
        count = count_n_queens(n)
        print(f"    {n}-Queens solution count: {count}")

    print("\n    4-Queens Solution Example:")
    solutions = solve_n_queens(4)
    for row in solutions[0]:
        print(f"    {row}")

    # 5. Generate Parentheses
    print("\n[5] Generate Parentheses")
    n = 3
    parens = generate_parentheses(n)
    print(f"    {n} pairs of parentheses ({len(parens)} total):")
    print(f"    {parens}")

    # 6. Word Search
    print("\n[6] Word Search")
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    print("    Board:")
    for row in board:
        print(f"    {row}")
    for word in ["ABCCED", "SEE", "ABCB"]:
        result = word_search([row[:] for row in board], word)
        print(f"    '{word}' exists? {result}")

    print("\n" + "=" * 60)
    print("Backtracking Pattern Summary")
    print("=" * 60)
    print("""
    Backtracking Template:

    def backtrack(state):
        if termination_condition:
            add to results
            return

        for choice in choices:
            if invalid choice:
                continue  # Pruning

            apply choice
            backtrack(next_state)
            undo choice  # Backtrack

    Key Points:
    1. Determine state representation
    2. Define termination condition
    3. Reduce unnecessary search with pruning
    4. Restore state (undo changes)
    """)


if __name__ == "__main__":
    main()
