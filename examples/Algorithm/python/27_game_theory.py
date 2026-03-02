"""
Game Theory
Game Theory Algorithms

Algorithms for combinatorial game theory and optimal strategies.
"""

from typing import List, Dict, Tuple, Set, Optional
from functools import lru_cache


# =============================================================================
# 1. Nim Game
# =============================================================================

def nim_xor(piles: List[int]) -> int:
    """
    XOR value (Nim-sum) of the Nim game
    XOR != 0: first player wins, XOR == 0: second player wins
    """
    result = 0
    for pile in piles:
        result ^= pile
    return result


def nim_winning_move(piles: List[int]) -> Optional[Tuple[int, int]]:
    """
    Find a winning move in the Nim game
    Returns: (pile index, remaining count) or None
    """
    xor = nim_xor(piles)

    if xor == 0:
        return None  # Losing position, no winning move

    for i, pile in enumerate(piles):
        target = pile ^ xor
        if target < pile:
            return (i, target)  # Leave target stones in the pile

    return None


def nim_game_simulation(piles: List[int], verbose: bool = False) -> int:
    """
    Nim game simulation
    Returns: winner (0: first player, 1: second player)
    """
    current = 0  # Current player

    while max(piles) > 0:
        move = nim_winning_move(piles)

        if move is None:
            # Losing position: make any move
            for i, pile in enumerate(piles):
                if pile > 0:
                    move = (i, pile - 1)
                    break

        pile_idx, new_count = move
        if verbose:
            print(f"    Player {current}: pile {pile_idx} from {piles[pile_idx]} to {new_count}")

        piles[pile_idx] = new_count
        current = 1 - current

    return 1 - current  # The last player to take wins


# =============================================================================
# 2. Sprague-Grundy Theorem
# =============================================================================

def mex(s: Set[int]) -> int:
    """
    Minimum Excludant
    Smallest non-negative integer not in the set
    """
    i = 0
    while i in s:
        i += 1
    return i


def calculate_grundy(position: int, moves: List[int], memo: Dict[int, int] = None) -> int:
    """
    Compute the Sprague-Grundy number
    position: current state (e.g., number of stones)
    moves: list of possible move amounts
    """
    if memo is None:
        memo = {}

    if position in memo:
        return memo[position]

    if position == 0:
        memo[position] = 0
        return 0

    reachable = set()
    for move in moves:
        if position >= move:
            reachable.add(calculate_grundy(position - move, moves, memo))

    result = mex(reachable)
    memo[position] = result
    return result


def multi_pile_grundy(piles: List[int], moves: List[int]) -> int:
    """
    Overall Grundy number for a multi-pile game
    XOR of each pile's Grundy number
    """
    memo = {}
    total_grundy = 0

    for pile in piles:
        grundy = calculate_grundy(pile, moves, memo)
        total_grundy ^= grundy

    return total_grundy


# =============================================================================
# 3. Nim Variants
# =============================================================================

def staircase_nim(stairs: List[int]) -> int:
    """
    Staircase Nim
    XOR of odd-indexed stairs = Grundy number
    """
    xor = 0
    for i in range(0, len(stairs), 2):  # Odd indices (even in 0-indexed)
        xor ^= stairs[i]
    return xor


def misere_nim(piles: List[int]) -> bool:
    """
    Misere Nim (last player to take loses)
    Returns: True if first player wins
    """
    xor = nim_xor(piles)
    all_one_or_zero = all(p <= 1 for p in piles)

    if all_one_or_zero:
        # If the count of piles with 1 stone is odd, second player wins
        ones = sum(1 for p in piles if p == 1)
        return ones % 2 == 0
    else:
        return xor != 0


def poker_nim(piles: List[int], k: int) -> bool:
    """
    Poker Nim: can also add stones to a pile (up to k)
    Returns: True if first player wins
    Rule: same as normal Nim (XOR != 0 means first player wins)
    """
    return nim_xor(piles) != 0


# =============================================================================
# 4. Minimax Algorithm
# =============================================================================

def minimax(position, depth: int, is_maximizing: bool,
            evaluate, get_moves, is_terminal) -> int:
    """
    Minimax Algorithm
    position: current game state
    depth: search depth
    is_maximizing: whether it is the maximizing player's turn
    evaluate: state evaluation function
    get_moves: function returning available moves
    is_terminal: function checking terminal state
    """
    if depth == 0 or is_terminal(position):
        return evaluate(position)

    moves = get_moves(position)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = minimax(new_position, depth - 1, False,
                                evaluate, get_moves, is_terminal)
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = minimax(new_position, depth - 1, True,
                                evaluate, get_moves, is_terminal)
            min_eval = min(min_eval, eval_score)
        return min_eval


def apply_move(position, move):
    """Return new state after applying a move (abstract function)"""
    # Implementation depends on the specific game
    pass


# =============================================================================
# 5. Alpha-Beta Pruning
# =============================================================================

def alpha_beta(position, depth: int, alpha: float, beta: float,
               is_maximizing: bool, evaluate, get_moves, is_terminal) -> int:
    """
    Alpha-Beta Pruning
    alpha: best guaranteed value for the maximizing player
    beta: best guaranteed value for the minimizing player
    """
    if depth == 0 or is_terminal(position):
        return evaluate(position)

    moves = get_moves(position)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = alpha_beta(new_position, depth - 1, alpha, beta,
                                   False, evaluate, get_moves, is_terminal)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Pruning
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_position = apply_move(position, move)
            eval_score = alpha_beta(new_position, depth - 1, alpha, beta,
                                   True, evaluate, get_moves, is_terminal)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Pruning
        return min_eval


# =============================================================================
# 6. Tic-Tac-Toe Implementation
# =============================================================================

class TicTacToe:
    """Tic-Tac-Toe Game"""

    def __init__(self):
        self.board = [[' '] * 3 for _ in range(3)]
        self.current_player = 'X'

    def get_moves(self) -> List[Tuple[int, int]]:
        """Return available moves"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def make_move(self, row: int, col: int) -> bool:
        """Make a move"""
        if self.board[row][col] != ' ':
            return False
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def undo_move(self, row: int, col: int):
        """Undo a move"""
        self.board[row][col] = ' '
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self) -> Optional[str]:
        """Check for a winner"""
        # Rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        # Columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        # Diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        return None

    def is_terminal(self) -> bool:
        """Check if the game is over"""
        return self.check_winner() is not None or len(self.get_moves()) == 0

    def evaluate(self) -> int:
        """Evaluate state (from X's perspective)"""
        winner = self.check_winner()
        if winner == 'X':
            return 10
        elif winner == 'O':
            return -10
        return 0

    def minimax(self, is_maximizing: bool) -> int:
        """Minimax"""
        if self.is_terminal():
            return self.evaluate()

        if is_maximizing:
            max_eval = float('-inf')
            for row, col in self.get_moves():
                self.make_move(row, col)
                eval_score = self.minimax(False)
                self.undo_move(row, col)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for row, col in self.get_moves():
                self.make_move(row, col)
                eval_score = self.minimax(True)
                self.undo_move(row, col)
                min_eval = min(min_eval, eval_score)
            return min_eval

    def best_move(self) -> Tuple[int, int]:
        """Find the best move"""
        best_score = float('-inf') if self.current_player == 'X' else float('inf')
        best_move = None

        for row, col in self.get_moves():
            self.make_move(row, col)
            score = self.minimax(self.current_player == 'X')
            self.undo_move(row, col)

            if self.current_player == 'X':
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (row, col)

        return best_move

    def display(self):
        """Display the board"""
        for i, row in enumerate(self.board):
            print("    " + " | ".join(row))
            if i < 2:
                print("    " + "-" * 9)


# =============================================================================
# 7. Stone Game
# =============================================================================

@lru_cache(maxsize=None)
def stone_game_dp(piles: Tuple[int, ...], left: int, right: int) -> int:
    """
    Stone Game: can only take from either end
    Returns maximum score difference the first player can achieve
    """
    if left > right:
        return 0

    # First player picks left
    pick_left = piles[left] - stone_game_dp(piles, left + 1, right)
    # First player picks right
    pick_right = piles[right] - stone_game_dp(piles, left, right - 1)

    return max(pick_left, pick_right)


def stone_game(piles: List[int]) -> bool:
    """
    Stone Game: True if first player wins
    """
    n = len(piles)
    diff = stone_game_dp(tuple(piles), 0, n - 1)
    return diff > 0


# =============================================================================
# 8. Bash Game
# =============================================================================

def bash_game(n: int, k: int) -> bool:
    """
    Bash Game: take at most k stones from n stones
    The player who takes the last stone wins
    Returns: True if first player wins
    """
    return n % (k + 1) != 0


def bash_game_optimal_move(n: int, k: int) -> int:
    """Optimal move in Bash Game (number of stones to take)"""
    if n % (k + 1) == 0:
        return 1  # Losing position, make any move
    return n % (k + 1)


# =============================================================================
# 9. Wythoff's Game
# =============================================================================

def wythoff_game(a: int, b: int) -> bool:
    """
    Wythoff's Game: take equal amounts from both piles or any amount from one pile
    The player who takes the last stone wins
    Returns: True if first player wins
    """
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio

    if a > b:
        a, b = b, a

    k = b - a
    ak = int(k * phi)

    return a != ak


# =============================================================================
# 10. Euclid's Game
# =============================================================================

def euclid_game(a: int, b: int) -> bool:
    """
    Euclid's Game: subtract a multiple of the smaller number from the larger
    The player who makes 0 wins
    Returns: True if first player wins
    """
    if a < b:
        a, b = b, a

    if b == 0:
        return False  # Already over

    # Recursive analysis
    turn = True  # True: first player's turn
    while b > 0:
        if a >= 2 * b or a == b:
            return turn
        a, b = b, a - b
        turn = not turn

    return not turn


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Game Theory Examples")
    print("=" * 60)

    # 1. Nim Game
    print("\n[1] Nim Game")
    piles = [3, 4, 5]
    xor = nim_xor(piles)
    move = nim_winning_move(piles)
    print(f"    Piles: {piles}")
    print(f"    XOR: {xor} ({'first player wins' if xor != 0 else 'second player wins'})")
    if move:
        print(f"    Winning move: reduce pile {move[0]} to {move[1]}")

    # Simulation
    print("\n    Game simulation:")
    piles_copy = [3, 4, 5]
    winner = nim_game_simulation(piles_copy, verbose=True)
    print(f"    Winner: Player {winner}")

    # 2. Sprague-Grundy
    print("\n[2] Sprague-Grundy Theorem")
    moves = [1, 3, 4]  # Can take 1, 3, or 4 at a time
    memo = {}
    for n in range(10):
        g = calculate_grundy(n, moves, memo)
        print(f"    G({n}) = {g}", end="  ")
    print()

    # Multiple piles
    piles = [7, 5]
    total_g = multi_pile_grundy(piles, moves)
    print(f"    Piles {piles}, moves {moves}")
    print(f"    Total Grundy: {total_g} ({'first player wins' if total_g != 0 else 'second player wins'})")

    # 3. Nim Variants
    print("\n[3] Nim Variants")
    # Misere Nim
    piles_misere = [1, 2, 3]
    print(f"    Misere Nim {piles_misere}: first player {'wins' if misere_nim(piles_misere) else 'loses'}")

    # Staircase Nim
    stairs = [3, 1, 2, 4]  # Stairs 1, 2, 3, 4
    print(f"    Staircase Nim {stairs}: Grundy = {staircase_nim(stairs)}")

    # 4. Tic-Tac-Toe
    print("\n[4] Tic-Tac-Toe Minimax")
    game = TicTacToe()
    game.board = [['X', 'O', 'X'],
                  [' ', 'O', ' '],
                  [' ', ' ', ' ']]
    game.current_player = 'X'
    print("    Current state:")
    game.display()
    best = game.best_move()
    print(f"    Best move for X: {best}")

    # 5. Stone Game
    print("\n[5] Stone Game")
    piles = [5, 3, 4, 5]
    print(f"    Piles: {piles}")
    result = stone_game(piles)
    diff = stone_game_dp(tuple(piles), 0, len(piles) - 1)
    print(f"    First player {'wins' if result else 'loses'} (score difference: {diff})")

    # 6. Bash Game
    print("\n[6] Bash Game")
    n, k = 10, 3
    print(f"    n={n}, k={k}")
    print(f"    First player {'wins' if bash_game(n, k) else 'loses'}")
    if bash_game(n, k):
        print(f"    Optimal move: take {bash_game_optimal_move(n, k)} stones")

    # 7. Wythoff's Game
    print("\n[7] Wythoff's Game")
    test_cases = [(1, 2), (3, 5), (4, 7), (5, 8)]
    for a, b in test_cases:
        result = wythoff_game(a, b)
        print(f"    ({a}, {b}): first player {'wins' if result else 'loses'}")

    # 8. Euclid's Game
    print("\n[8] Euclid's Game")
    test_cases = [(25, 7), (24, 10), (100, 45)]
    for a, b in test_cases:
        result = euclid_game(a, b)
        print(f"    ({a}, {b}): first player {'wins' if result else 'loses'}")

    # 9. Algorithm Summary
    print("\n[9] Game Theory Algorithm Summary")
    print("    | Game           | Winning Condition              |")
    print("    |----------------|--------------------------------|")
    print("    | Nim Game       | XOR != 0                       |")
    print("    | Misere Nim     | Complex condition              |")
    print("    | Bash Game      | n % (k+1) != 0                 |")
    print("    | Wythoff's Game | Golden ratio losing positions  |")
    print("    | General Games  | Sprague-Grundy Theorem         |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
