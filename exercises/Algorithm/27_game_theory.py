"""
Exercises for Lesson 27: Game Theory
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: Stone taking (W/L analysis), Nim game (XOR), Grundy numbers (mex),
Combined games, Minimax with alpha-beta.
"""


# === Exercise 1: Stone Taking Game (Win/Lose Analysis) ===
# Problem: Two players take turns removing 1-3 stones from a pile.
#   The player who takes the last stone wins.
#   Determine if the first player wins given n stones.

def exercise_1():
    """Solution: Pattern analysis - first player loses iff n % 4 == 0."""
    def first_player_wins_simple(n):
        """Direct mathematical solution: lose only when n is a multiple of 4."""
        return n % 4 != 0

    # Also verify with DP approach
    def first_player_wins_dp(n):
        """DP approach: dp[i] = True if the player whose turn it is wins with i stones."""
        if n == 0:
            return False
        dp = [False] * (n + 1)
        # dp[0] = False (no stones = lose)
        for i in range(1, n + 1):
            # Current player wins if any move leads to opponent losing
            for take in range(1, 4):
                if i - take >= 0 and not dp[i - take]:
                    dp[i] = True
                    break
        return dp[n]

    for n in range(21):
        result_simple = first_player_wins_simple(n)
        result_dp = first_player_wins_dp(n)
        status = "WIN" if result_simple else "LOSE"
        print(f"n={n:2d}: {status}")
        assert result_simple == result_dp

    print("\nPattern: First player loses when n is a multiple of 4.")
    print("All Stone Taking tests passed!")


# === Exercise 2: Nim Game ===
# Problem: Given k piles of stones, two players take turns removing any number
#   of stones from one pile. The player who takes the last stone wins.
#   Determine winner using XOR (Sprague-Grundy theorem for Nim).

def exercise_2():
    """Solution: XOR of all pile sizes. First player wins iff XOR != 0."""
    def nim_winner(piles):
        """
        Returns True if the first player wins.
        By Sprague-Grundy theorem for Nim:
        - XOR of all pile sizes = 0 => second player wins (P-position)
        - XOR != 0 => first player wins (N-position)
        """
        xor_sum = 0
        for pile in piles:
            xor_sum ^= pile
        return xor_sum != 0

    tests = [
        ([1, 2, 3], False),      # 1^2^3 = 0 -> P2 wins
        ([1, 1], False),         # 1^1 = 0 -> P2 wins
        ([1, 2, 4], True),       # 1^2^4 = 7 -> P1 wins
        ([3, 4, 5], True),       # 3^4^5 = 2 -> P1 wins
        ([0], False),            # no stones -> P1 loses
        ([5], True),             # single pile -> P1 takes all
        ([1, 1, 1], True),       # 1^1^1 = 1 -> P1 wins
    ]

    for piles, expected in tests:
        result = nim_winner(piles)
        xor_val = 0
        for p in piles:
            xor_val ^= p
        winner = "P1" if result else "P2"
        print(f"Piles={piles}, XOR={xor_val}, Winner={winner}")
        assert result == expected

    print("All Nim Game tests passed!")


# === Exercise 3: Grundy Numbers (Sprague-Grundy) ===
# Problem: Calculate Grundy numbers for a general game.
#   Game: Remove 1, 2, or 3 stones. Last to move wins.
#   Grundy number = mex(Grundy values of reachable states)
#   mex = minimum excludant (smallest non-negative integer not in the set)

def exercise_3():
    """Solution: Grundy number computation using mex."""
    def mex(s):
        """Minimum excludant: smallest non-negative integer not in set s."""
        i = 0
        while i in s:
            i += 1
        return i

    def compute_grundy(n, moves):
        """
        Compute Grundy numbers for states 0..n.
        moves: list of valid move sizes (e.g., [1, 2, 3])
        """
        grundy = [0] * (n + 1)

        for i in range(1, n + 1):
            reachable = set()
            for m in moves:
                if i - m >= 0:
                    reachable.add(grundy[i - m])
            grundy[i] = mex(reachable)

        return grundy

    # Standard game: moves = {1, 2, 3}
    grundy = compute_grundy(20, [1, 2, 3])
    print("Grundy numbers (moves {1,2,3}):")
    for i in range(21):
        print(f"  G({i}) = {grundy[i]}", end="")
        if i % 5 == 4:
            print()
    print()
    # Pattern: G(n) = n % 4
    for i in range(21):
        assert grundy[i] == i % 4

    # Custom game: moves = {1, 3, 4}
    grundy2 = compute_grundy(15, [1, 3, 4])
    print("Grundy numbers (moves {1,3,4}):")
    for i in range(16):
        print(f"  G({i}) = {grundy2[i]}")
    print()

    print("All Grundy Number tests passed!")


# === Exercise 4: Combined Games (Sprague-Grundy XOR) ===
# Problem: Multiple independent games are played simultaneously.
#   A player must make a move in exactly one game. The player who cannot move loses.
#   The combined game's Grundy value = XOR of individual Grundy values.

def exercise_4():
    """Solution: XOR of Grundy values for combined independent games."""
    def mex(s):
        i = 0
        while i in s:
            i += 1
        return i

    def combined_game_winner(game_states, moves_per_game):
        """
        game_states: list of current stone counts for each game
        moves_per_game: list of allowed move sets for each game
        Returns: True if first player wins
        """
        xor_sum = 0
        for state, moves in zip(game_states, moves_per_game):
            # Compute Grundy number for this game's current state
            grundy = [0] * (state + 1)
            for i in range(1, state + 1):
                reachable = set()
                for m in moves:
                    if i - m >= 0:
                        reachable.add(grundy[i - m])
                grundy[i] = mex(reachable)

            xor_sum ^= grundy[state]

        return xor_sum != 0

    # Three games: stones = [5, 3, 7], all with moves {1, 2, 3}
    # Grundy: G(5)=1, G(3)=3, G(7)=3 -> XOR = 1^3^3 = 1 -> P1 wins
    result = combined_game_winner([5, 3, 7], [[1, 2, 3]] * 3)
    print(f"Games [5,3,7] with moves {{1,2,3}}: P1 wins = {result}")
    assert result is True

    # G(4)=0, G(8)=0 -> XOR = 0 -> P2 wins
    result = combined_game_winner([4, 8], [[1, 2, 3]] * 2)
    print(f"Games [4,8] with moves {{1,2,3}}: P1 wins = {result}")
    assert result is False

    print("All Combined Games tests passed!")


# === Exercise 5: Minimax with Alpha-Beta Pruning ===
# Problem: Implement minimax for a simple game tree.
#   Used for two-player zero-sum games (e.g., Tic-Tac-Toe).

def exercise_5():
    """Solution: Minimax with alpha-beta pruning."""
    def minimax(node, depth, is_maximizing, alpha, beta, evaluate, get_children):
        """
        Generic minimax with alpha-beta pruning.
        node: current game state
        depth: remaining search depth
        is_maximizing: True if maximizing player's turn
        evaluate: function to evaluate leaf nodes
        get_children: function to get child states
        """
        if depth == 0 or not get_children(node):
            return evaluate(node)

        if is_maximizing:
            max_eval = float('-inf')
            for child in get_children(node):
                val = minimax(child, depth - 1, False, alpha, beta, evaluate, get_children)
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break  # Beta pruning
            return max_eval
        else:
            min_eval = float('inf')
            for child in get_children(node):
                val = minimax(child, depth - 1, True, alpha, beta, evaluate, get_children)
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break  # Alpha pruning
            return min_eval

    # Example: simple game tree represented as nested lists
    # Leaf values at depth 3: [[3, 5], [6, 9]], [[1, 2], [0, -1]]
    # This represents:
    #          MAX
    #        /     \
    #      MIN     MIN
    #     / \     / \
    #   MAX MAX MAX MAX
    #   /\  /\  /\  /\
    #  3 5 6 9 1 2 0 -1

    tree = [[[[3, 5], [6, 9]], [[1, 2], [0, -1]]]]

    def evaluate(node):
        return node

    def get_children(node):
        if isinstance(node, list):
            return node
        return []

    result = minimax(tree[0], 3, True, float('-inf'), float('inf'),
                     evaluate, get_children)
    print(f"Minimax result: {result}")
    # MAX level: max(MIN(MAX(3,5), MAX(6,9)), MIN(MAX(1,2), MAX(0,-1)))
    # = max(MIN(5, 9), MIN(2, 0))
    # = max(5, 0)
    # = 5
    assert result == 5

    print("All Minimax tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Stone Taking Game ===")
    exercise_1()
    print("\n=== Exercise 2: Nim Game ===")
    exercise_2()
    print("\n=== Exercise 3: Grundy Numbers ===")
    exercise_3()
    print("\n=== Exercise 4: Combined Games ===")
    exercise_4()
    print("\n=== Exercise 5: Minimax with Alpha-Beta ===")
    exercise_5()
    print("\nAll exercises completed!")
