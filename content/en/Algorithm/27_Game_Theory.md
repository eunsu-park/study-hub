# 28. Game Theory

## Learning Objectives
- Understanding basic concepts of combinatorial game theory
- Analyzing winning and losing states
- Solving Nim game
- Utilizing Sprague-Grundy theorem
- Implementing minimax algorithm

## 1. Combinatorial Game Theory Basics

### Characteristics of Combinatorial Games

```
┌─────────────────────────────────────────────────┐
│          Combinatorial Game                      │
├─────────────────────────────────────────────────┤
│  1. Two players take turns                      │
│  2. Perfect information (all information public)│
│  3. No element of chance (deterministic)        │
│  4. Finite number of states                     │
│  5. Player who cannot move loses                │
└─────────────────────────────────────────────────┘
```

### Winning State (W) vs Losing State (L)

```
Losing state (L): All reachable states are W
Winning state (W): At least one reachable state is L

    Start
      |
      W ←── Can force opponent to L
     / \
    L   W
    |   |
    W   L ←── All moves lead to W only

Rule: Start from L = lose, start from W = win
```

### Basic Example: Stone Taking Game

```python
def stone_game(n, moves):
    """
    Can take moves[i] stones from n stones
    Player who takes last stone wins
    """
    # dp[i] = True if winning state
    dp = [False] * (n + 1)

    for i in range(1, n + 1):
        for move in moves:
            if i >= move and not dp[i - move]:
                dp[i] = True
                break

    return dp[n]

# Example: 10 stones, can take 1, 2, or 3 at a time
print(stone_game(10, [1, 2, 3]))  # True
```

---

## 2. Nim Game

### Rules

- Multiple piles of stones
- Each turn, take 1 or more stones from one pile
- Player who takes the last stone wins

```
Piles: [3, 4, 5]

Player 1: Take 2 from pile 2 → [3, 2, 5]
Player 2: Take 3 from pile 3 → [3, 2, 2]
...
```

### Nim Theorem

**XOR Sum = 0 means losing, ≠ 0 means winning**

```
Example: [3, 4, 5]
3 = 011
4 = 100
5 = 101
---------
XOR = 010 = 2 ≠ 0 → First player wins

Example: [1, 2, 3]
1 = 01
2 = 10
3 = 11
--------
XOR = 00 = 0 → First player loses
```

### Implementation

```python
def nim_game(piles):
    """
    Determine winner of Nim game
    Returns: True if first player wins
    The XOR theorem holds because XOR = 0 means every bit is "balanced" across piles —
    any move that changes one pile will unbalance some bit, and the second player
    can always restore balance, eventually leaving the first player with all-zero piles.
    """
    xor_sum = 0
    for pile in piles:
        xor_sum ^= pile  # Accumulate bitwise XOR across all piles

    return xor_sum != 0  # Non-zero XOR means first player can always make it zero

def nim_winning_move(piles):
    """
    Find optimal move to win
    Returns: (pile index, stones to take) or None
    Strategy: reduce some pile to make total XOR = 0, forcing the opponent into a losing state.
    """
    xor_sum = 0
    for pile in piles:
        xor_sum ^= pile

    if xor_sum == 0:
        return None  # Already in losing state — every move creates a winning state for opponent

    # Find a pile where XOR-ing with xor_sum reduces it.
    # target = pile ^ xor_sum gives the value that pile must become so that
    # the new XOR sum is zero; target < pile ensures we remove stones (not add).
    for i, pile in enumerate(piles):
        target = pile ^ xor_sum
        if target < pile:
            return (i, pile - target)  # Take (pile - target) stones from pile i

    return None

# Usage example
piles = [3, 4, 5]
if nim_game(piles):
    print("First player wins!")
    move = nim_winning_move(piles)
    print(f"Optimal move: take {move[1]} from pile {move[0]}")
else:
    print("Second player wins!")
```

### Variant: Misère Nim

Version where player who takes last stone **loses**

```python
def misere_nim(piles):
    """
    Misère Nim: Taking last stone means losing
    """
    xor_sum = 0
    all_ones = True

    for pile in piles:
        xor_sum ^= pile
        if pile > 1:
            all_ones = False

    if all_ones:
        # If all piles ≤ 1, lose if odd number of piles
        return len([p for p in piles if p > 0]) % 2 == 0
    else:
        # Same as regular Nim
        return xor_sum != 0
```

---

## 3. Sprague-Grundy Theorem

### Grundy Number (Nimber)

Every combinatorial game state has a **Grundy number**, which is the size of an equivalent Nim pile.

```
Grundy(state) = mex({Grundy(next_state) for all next_state})

mex = minimum excludant (smallest non-negative integer not in set)
mex({0, 1, 2}) = 3
mex({0, 2, 3}) = 1
mex({1, 2}) = 0
```

### Rules

```
Grundy = 0: losing state (L)
Grundy > 0: winning state (W)

Combining independent games:
Grundy(G1 + G2) = Grundy(G1) XOR Grundy(G2)
```

### Implementation

```python
def calculate_grundy(state, get_next_states, memo=None):
    """
    Calculate Grundy number for a state.
    The Grundy number encodes the entire strategic value of a game position as a
    single integer equivalent to a Nim pile of that size — this is why you can
    combine independent games by XOR-ing their Grundy numbers.
    """
    if memo is None:
        memo = {}

    if state in memo:
        return memo[state]  # Memoization avoids exponential re-computation of overlapping states

    next_states = get_next_states(state)

    if not next_states:
        # Terminal state: no moves available means the current player loses (normal play convention)
        # A losing state is equivalent to an empty Nim pile — Grundy number 0
        memo[state] = 0
        return 0

    # Collect Grundy numbers of all reachable next states
    grundy_set = set()
    for next_state in next_states:
        grundy_set.add(calculate_grundy(next_state, get_next_states, memo))

    # mex (minimum excludant) = smallest non-negative integer NOT in grundy_set.
    # It represents the "first gap" — the current state acts like a Nim pile you
    # cannot reduce to any size equal to a reachable position's Grundy number.
    mex = 0
    while mex in grundy_set:
        mex += 1

    memo[state] = mex
    return mex

# Example: Take 1-3 stones game
def get_next_states(n):
    """States after taking 1, 2, or 3 stones from n"""
    return [n - k for k in [1, 2, 3] if n >= k]

for n in range(10):
    g = calculate_grundy(n, get_next_states)
    print(f"Grundy({n}) = {g}")
# Output: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1
# Pattern: n % 4
```

### Combined Games

```python
def combined_game_winner(games):
    """
    Combination of independent games
    games: list of Grundy numbers for each game
    """
    xor_sum = 0
    for g in games:
        xor_sum ^= g

    return xor_sum != 0

# Example: Three stone games (5, 7, 3 stones)
grundy_5 = calculate_grundy(5, get_next_states)  # 1
grundy_7 = calculate_grundy(7, get_next_states)  # 3
grundy_3 = calculate_grundy(3, get_next_states)  # 3

# Combined: 1 XOR 3 XOR 3 = 1 ≠ 0 → First player wins
print(combined_game_winner([grundy_5, grundy_7, grundy_3]))
```

---

## 4. Minimax Algorithm

### Concept

Calculate outcome when two players play optimally

```
Max player: tries to maximize score
Min player: tries to minimize score

        Max
       / | \
      3  5  2
     /|  |  |\
Min 3 1  5  2 4

Max chooses 5 (maximum)
```

### Basic Implementation

```python
def minimax(state, depth, is_maximizing, get_moves, evaluate, is_terminal):
    """
    Minimax algorithm
    state: current game state
    depth: search depth
    is_maximizing: True if Max player's turn
    get_moves: function returning possible moves
    evaluate: function returning state score
    is_terminal: function checking if game is over
    """
    if depth == 0 or is_terminal(state):
        return evaluate(state)

    moves = get_moves(state)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = minimax(next_state, depth - 1, False,
                                 get_moves, evaluate, is_terminal)
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = minimax(next_state, depth - 1, True,
                                 get_moves, evaluate, is_terminal)
            min_eval = min(min_eval, eval_score)
        return min_eval
```

### Alpha-Beta Pruning

```python
def alphabeta(state, depth, alpha, beta, is_maximizing,
              get_moves, evaluate, is_terminal):
    """
    Minimax optimized with alpha-beta pruning.
    alpha = the best score Max can guarantee so far (lower bound for Max)
    beta  = the best score Min can guarantee so far (upper bound for Max)
    We prune when beta <= alpha because Max will never choose a branch that
    gives Min a value better than what Max can already guarantee elsewhere.
    In the best case (perfect move ordering), this reduces O(b^d) to O(b^(d/2)).
    """
    if depth == 0 or is_terminal(state):
        return evaluate(state)  # Leaf node — return static evaluation

    moves = get_moves(state)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = alphabeta(next_state, depth - 1, alpha, beta, False,
                                   get_moves, evaluate, is_terminal)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)  # Update Max's lower bound
            if beta <= alpha:
                break  # Min already has a better option elsewhere — this subtree is irrelevant
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            next_state = apply_move(state, move)
            eval_score = alphabeta(next_state, depth - 1, alpha, beta, True,
                                   get_moves, evaluate, is_terminal)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)  # Update Min's upper bound
            if beta <= alpha:
                break  # Max already has a better option elsewhere — this subtree is irrelevant
        return min_eval

# Call
score = alphabeta(initial_state, max_depth, float('-inf'), float('inf'), True,
                  get_moves, evaluate, is_terminal)
```

---

## 5. Tic-Tac-Toe Example

```python
class TicTacToe:
    def __init__(self):
        self.board = [[' '] * 3 for _ in range(3)]
        self.current_player = 'X'

    def get_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def make_move(self, row, col):
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def undo_move(self, row, col):
        self.board[row][col] = ' '
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        # Check rows, columns, diagonals
        lines = []
        for i in range(3):
            lines.append([self.board[i][j] for j in range(3)])  # Row
            lines.append([self.board[j][i] for j in range(3)])  # Column
        lines.append([self.board[i][i] for i in range(3)])      # Diagonal
        lines.append([self.board[i][2-i] for i in range(3)])    # Anti-diagonal

        for line in lines:
            if line[0] == line[1] == line[2] != ' ':
                return line[0]
        return None

    def is_terminal(self):
        if self.check_winner():
            return True
        return len(self.get_moves()) == 0

    def evaluate(self):
        winner = self.check_winner()
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        return 0

    def minimax(self, is_maximizing):
        if self.is_terminal():
            return self.evaluate()

        if is_maximizing:
            best = float('-inf')
            for move in self.get_moves():
                self.make_move(*move)
                best = max(best, self.minimax(False))
                self.undo_move(*move)
            return best
        else:
            best = float('inf')
            for move in self.get_moves():
                self.make_move(*move)
                best = min(best, self.minimax(True))
                self.undo_move(*move)
            return best

    def best_move(self):
        """Find optimal move for current player"""
        best_score = float('-inf') if self.current_player == 'X' else float('inf')
        best_move = None
        is_max = self.current_player == 'X'

        for move in self.get_moves():
            self.make_move(*move)
            score = self.minimax(not is_max)
            self.undo_move(*move)

            if is_max and score > best_score:
                best_score = score
                best_move = move
            elif not is_max and score < best_score:
                best_score = score
                best_move = move

        return best_move

# Usage example
game = TicTacToe()
while not game.is_terminal():
    move = game.best_move()
    print(f"{game.current_player} plays at {move}")
    game.make_move(*move)

winner = game.check_winner()
print(f"Winner: {winner if winner else 'Draw'}")
```

---

## 6. Practical Problem Patterns

### Pattern 1: Simple Win/Loss Games

```python
def simple_game_winner(n):
    """
    n stones, take 1-3 at a time
    Last to take wins
    """
    return n % 4 != 0

# Solve quickly with Grundy pattern
```

### Pattern 2: Multi-Pile Games

```python
def multi_pile_game(piles, max_take):
    """
    Take 1~max_take from each pile
    """
    # Grundy for each pile = pile % (max_take + 1)
    xor_sum = 0
    for pile in piles:
        xor_sum ^= (pile % (max_take + 1))

    return xor_sum != 0
```

### Pattern 3: Graph Games

```python
def graph_game(adj, start):
    """
    Token movement game on graph
    Lose if cannot move
    """
    n = len(adj)
    grundy = [-1] * n

    def calc_grundy(node):
        if grundy[node] != -1:
            return grundy[node]

        if not adj[node]:  # Cannot move
            grundy[node] = 0
            return 0

        reachable = set()
        for next_node in adj[node]:
            reachable.add(calc_grundy(next_node))

        mex = 0
        while mex in reachable:
            mex += 1

        grundy[node] = mex
        return mex

    return calc_grundy(start) != 0
```

### Pattern 4: Game Splitting

```python
def split_game(n):
    """
    Split n stones into two non-empty piles
    Lose if cannot split
    """
    grundy = [0] * (n + 1)

    for i in range(2, n + 1):
        reachable = set()
        for j in range(1, i // 2 + 1):
            # Split i into j and i-j
            reachable.add(grundy[j] ^ grundy[i - j])

        mex = 0
        while mex in reachable:
            mex += 1
        grundy[i] = mex

    return grundy[n] != 0
```

---

## 7. Advanced Techniques

### Nimber Multiplication

Nimbers define multiplication in addition to addition (XOR).

```python
def nim_multiply(a, b):
    """
    Nimber multiplication (for small numbers)
    """
    if a < 2 or b < 2:
        return a * b

    # Power of 2 decomposition
    highest_a = a.bit_length() - 1
    highest_b = b.bit_length() - 1

    if highest_a > highest_b:
        a, b = b, a
        highest_a, highest_b = highest_b, highest_a

    if highest_a == 0:
        return a * b

    # Recursive calculation (complex, usually use table)
    # Simplified version only handles simple cases
    return a * b  # Actually more complex
```

### Game Tree Storage

```python
class GameTree:
    def __init__(self):
        self.cache = {}

    def solve(self, state, get_next, hash_state):
        """
        Game tree search with memoization
        """
        h = hash_state(state)
        if h in self.cache:
            return self.cache[h]

        next_states = get_next(state)
        if not next_states:
            self.cache[h] = False  # No move = lose
            return False

        # If any next state is losing, current is winning
        for ns in next_states:
            if not self.solve(ns, get_next, hash_state):
                self.cache[h] = True
                return True

        self.cache[h] = False
        return False
```

---

## 8. Time Complexity Summary

| Algorithm | Time Complexity | Notes |
|-----------|----------------|-------|
| Nim game | O(n) | n = number of piles |
| Grundy calculation | O(states × branches) | Memoization essential |
| Minimax | O(b^d) | b = branching, d = depth |
| Alpha-beta | O(b^(d/2)) ~ O(b^d) | Best with optimal ordering |

---

## 9. Common Mistakes

### Mistake 1: mex Calculation Error

```python
# Incorrect: finding minimum
mex = min(grundy_set)

# Correct: smallest non-negative not in set
mex = 0
while mex in grundy_set:
    mex += 1
```

### Mistake 2: XOR Operator Precedence

```python
# Incorrect
result = a ^ b == 0  # Not (a ^ b) == 0!

# Correct
result = (a ^ b) == 0
```

### Mistake 3: Terminal Condition

```python
# No possible moves means Grundy = 0
if not next_states:
    return 0  # Losing state
```

---

## 10. Practice Problems

| Difficulty | Problem Type | Key Concept |
|-----------|--------------|-------------|
| ★★☆ | Stone taking | Basic W/L analysis |
| ★★☆ | Nim game | XOR application |
| ★★★ | Grundy calculation | mex function |
| ★★★ | Combined games | XOR combination |
| ★★★★ | Minimax | Alpha-beta |

---

## Next Steps

- [28_Advanced_DP_Optimization.md](./28_Advanced_DP_Optimization.md) - Advanced DP optimization

---

## Learning Checklist

1. Why does XOR sum = 0 mean losing in Nim?
2. What is the mex function for Grundy numbers?
3. How to calculate Grundy for combined independent games?
4. How does alpha-beta pruning work?
