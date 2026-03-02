/*
 * Game Theory
 * Nim Game, Sprague-Grundy, Minimax
 *
 * Algorithms for finding optimal strategies in two-player games.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#define MAX_N 1001

/* =============================================================================
 * 1. Nim Game
 * ============================================================================= */

/* Nim game: take stones from multiple piles
 * The player who takes the last stone wins
 * If XOR is 0, current player loses; otherwise wins */
bool nim_game(int piles[], int n) {
    int xor_sum = 0;
    for (int i = 0; i < n; i++) {
        xor_sum ^= piles[i];
    }
    return xor_sum != 0;
}

/* Winning strategy: find a move that makes XOR equal to 0 */
void nim_winning_move(int piles[], int n, int* pile_idx, int* take) {
    int xor_sum = 0;
    for (int i = 0; i < n; i++) {
        xor_sum ^= piles[i];
    }

    if (xor_sum == 0) {
        *pile_idx = -1;
        *take = -1;
        return;
    }

    for (int i = 0; i < n; i++) {
        int target = piles[i] ^ xor_sum;
        if (target < piles[i]) {
            *pile_idx = i;
            *take = piles[i] - target;
            return;
        }
    }
}

/* =============================================================================
 * 2. Sprague-Grundy Theorem
 * ============================================================================= */

/* Grundy number (Nimber) computation
 * mex: minimum excludant (smallest non-negative integer not in the set) */
int mex(int set[], int n) {
    bool* exists = calloc(n + 1, sizeof(bool));
    for (int i = 0; i < n; i++) {
        if (set[i] <= n) exists[set[i]] = true;
    }
    int result = 0;
    while (exists[result]) result++;
    free(exists);
    return result;
}

/* Grundy numbers for a simple game (can take 1~3 stones) */
int* grundy_simple(int max_n) {
    int* grundy = calloc(max_n + 1, sizeof(int));
    grundy[0] = 0;

    for (int i = 1; i <= max_n; i++) {
        int moves[3];
        int move_count = 0;

        for (int take = 1; take <= 3 && take <= i; take++) {
            moves[move_count++] = grundy[i - take];
        }

        grundy[i] = mex(moves, move_count);
    }

    return grundy;
}

/* General Grundy number computation (allowed take amounts given) */
int* grundy_general(int max_n, int allowed[], int allowed_count) {
    int* grundy = calloc(max_n + 1, sizeof(int));

    for (int i = 1; i <= max_n; i++) {
        int* moves = malloc(allowed_count * sizeof(int));
        int move_count = 0;

        for (int j = 0; j < allowed_count; j++) {
            if (allowed[j] <= i) {
                moves[move_count++] = grundy[i - allowed[j]];
            }
        }

        grundy[i] = mex(moves, move_count);
        free(moves);
    }

    return grundy;
}

/* Grundy number for combined games (XOR operation) */
int combined_grundy(int grundy_values[], int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result ^= grundy_values[i];
    }
    return result;
}

/* =============================================================================
 * 3. Minimax Algorithm
 * ============================================================================= */

#define BOARD_SIZE 3

typedef struct {
    int board[BOARD_SIZE][BOARD_SIZE];
    int current_player;  /* 1: X, -1: O */
} TicTacToe;

void ttt_init(TicTacToe* game) {
    memset(game->board, 0, sizeof(game->board));
    game->current_player = 1;
}

int ttt_check_winner(TicTacToe* game) {
    /* Check rows */
    for (int i = 0; i < 3; i++) {
        int sum = game->board[i][0] + game->board[i][1] + game->board[i][2];
        if (sum == 3) return 1;
        if (sum == -3) return -1;
    }

    /* Check columns */
    for (int j = 0; j < 3; j++) {
        int sum = game->board[0][j] + game->board[1][j] + game->board[2][j];
        if (sum == 3) return 1;
        if (sum == -3) return -1;
    }

    /* Check diagonals */
    int diag1 = game->board[0][0] + game->board[1][1] + game->board[2][2];
    int diag2 = game->board[0][2] + game->board[1][1] + game->board[2][0];
    if (diag1 == 3 || diag2 == 3) return 1;
    if (diag1 == -3 || diag2 == -3) return -1;

    return 0;
}

bool ttt_is_full(TicTacToe* game) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (game->board[i][j] == 0) return false;
        }
    }
    return true;
}

/* Minimax with Alpha-Beta Pruning */
int minimax(TicTacToe* game, int depth, int alpha, int beta, bool is_maximizing) {
    int winner = ttt_check_winner(game);
    if (winner != 0) return winner * 10;
    if (ttt_is_full(game)) return 0;

    if (is_maximizing) {
        int max_eval = INT_MIN;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (game->board[i][j] == 0) {
                    game->board[i][j] = 1;
                    int eval = minimax(game, depth + 1, alpha, beta, false);
                    game->board[i][j] = 0;
                    if (eval > max_eval) max_eval = eval;
                    if (eval > alpha) alpha = eval;
                    if (beta <= alpha) return max_eval;
                }
            }
        }
        return max_eval;
    } else {
        int min_eval = INT_MAX;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (game->board[i][j] == 0) {
                    game->board[i][j] = -1;
                    int eval = minimax(game, depth + 1, alpha, beta, true);
                    game->board[i][j] = 0;
                    if (eval < min_eval) min_eval = eval;
                    if (eval < beta) beta = eval;
                    if (beta <= alpha) return min_eval;
                }
            }
        }
        return min_eval;
    }
}

/* Find the best move */
void find_best_move(TicTacToe* game, int* best_row, int* best_col) {
    int best_val = INT_MIN;
    *best_row = -1;
    *best_col = -1;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (game->board[i][j] == 0) {
                game->board[i][j] = 1;
                int move_val = minimax(game, 0, INT_MIN, INT_MAX, false);
                game->board[i][j] = 0;

                if (move_val > best_val) {
                    *best_row = i;
                    *best_col = j;
                    best_val = move_val;
                }
            }
        }
    }
}

/* =============================================================================
 * 4. Other Famous Games
 * ============================================================================= */

/* Bash game: take 1~k stones from n stones */
bool bash_game(int n, int k) {
    return (n % (k + 1)) != 0;
}

/* Wythoff game: take equal amounts from both piles or any amount from one pile */
bool wythoff_game(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    int k = b - a;
    int cold_a = (int)(k * phi);
    return !(a == cold_a);
}

/* Euclid game: GCD game */
bool euclid_game(int a, int b) {
    if (a < b) { int t = a; a = b; b = t; }
    if (b == 0) return false;

    int moves = 0;
    while (b > 0) {
        if (a >= 2 * b) return (moves % 2 == 0);
        a = a - b;
        if (a < b) { int t = a; a = b; b = t; }
        moves++;
    }
    return (moves % 2 == 1);
}

/* Stone Game - DP-based */
int stone_game(int piles[], int n) {
    int** dp = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dp[i] = calloc(n, sizeof(int));
        dp[i][i] = piles[i];
    }

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            int take_left = piles[i] - dp[i + 1][j];
            int take_right = piles[j] - dp[i][j - 1];
            dp[i][j] = (take_left > take_right) ? take_left : take_right;
        }
    }

    int result = dp[0][n - 1];
    for (int i = 0; i < n; i++) free(dp[i]);
    free(dp);
    return result;
}

/* =============================================================================
 * Test
 * ============================================================================= */

void print_board(TicTacToe* game) {
    char symbols[] = {'O', '.', 'X'};
    for (int i = 0; i < 3; i++) {
        printf("      ");
        for (int j = 0; j < 3; j++) {
            printf("%c ", symbols[game->board[i][j] + 1]);
        }
        printf("\n");
    }
}

int main(void) {
    printf("============================================================\n");
    printf("Game Theory Examples\n");
    printf("============================================================\n");

    /* 1. Nim Game */
    printf("\n[1] Nim Game\n");
    int piles1[] = {3, 4, 5};
    printf("    Piles: [3, 4, 5]\n");
    printf("    XOR: %d ^ %d ^ %d = %d\n", 3, 4, 5, 3 ^ 4 ^ 5);
    printf("    First player wins: %s\n", nim_game(piles1, 3) ? "yes" : "no");

    int pile_idx, take;
    nim_winning_move(piles1, 3, &pile_idx, &take);
    if (pile_idx >= 0) {
        printf("    Winning strategy: take %d from pile %d\n", take, pile_idx);
    }

    int piles2[] = {1, 2, 3};
    printf("    Piles: [1, 2, 3]\n");
    printf("    XOR: %d ^ %d ^ %d = %d\n", 1, 2, 3, 1 ^ 2 ^ 3);
    printf("    First player wins: %s\n", nim_game(piles2, 3) ? "yes" : "no");

    /* 2. Sprague-Grundy */
    printf("\n[2] Sprague-Grundy Theorem\n");
    int* grundy = grundy_simple(15);
    printf("    Game: can take 1~3 stones\n");
    printf("    Grundy numbers: ");
    for (int i = 0; i <= 10; i++) {
        printf("G(%d)=%d ", i, grundy[i]);
    }
    printf("\n");
    printf("    Pattern: 0, 1, 2, 3, 0, 1, 2, 3, ... (period 4)\n");
    free(grundy);

    /* Custom game */
    int allowed[] = {1, 3, 4};
    grundy = grundy_general(15, allowed, 3);
    printf("    Game: can take 1, 3, or 4 stones\n");
    printf("    Grundy numbers: ");
    for (int i = 0; i <= 10; i++) {
        printf("G(%d)=%d ", i, grundy[i]);
    }
    printf("\n");
    free(grundy);

    /* 3. Minimax (Tic-Tac-Toe) */
    printf("\n[3] Minimax Algorithm (Tic-Tac-Toe)\n");
    TicTacToe game;
    ttt_init(&game);

    int row, col;
    find_best_move(&game, &row, &col);
    printf("    Best first move on empty board: (%d, %d)\n", row, col);

    /* Set up a position */
    game.board[0][0] = 1;   /* X */
    game.board[1][1] = -1;  /* O */
    game.board[2][2] = 1;   /* X */

    printf("    Current state:\n");
    print_board(&game);

    find_best_move(&game, &row, &col);
    printf("    X's best move: (%d, %d)\n", row, col);

    /* 4. Other Games */
    printf("\n[4] Other Famous Games\n");

    printf("    Bash game (n=10, k=3): %s\n",
           bash_game(10, 3) ? "first player wins" : "second player wins");
    printf("    Bash game (n=12, k=3): %s\n",
           bash_game(12, 3) ? "first player wins" : "second player wins");

    printf("    Wythoff game (3, 5): %s\n",
           wythoff_game(3, 5) ? "first player wins" : "second player wins");

    printf("    Euclid game (10, 6): %s\n",
           euclid_game(10, 6) ? "first player wins" : "second player wins");

    /* 5. Stone Game */
    printf("\n[5] Stone Game (DP)\n");
    int stones[] = {5, 3, 4, 5};
    printf("    Stone array: [5, 3, 4, 5]\n");
    int diff = stone_game(stones, 4);
    printf("    First-Second player score diff: %d\n", diff);
    printf("    First player %s\n", diff > 0 ? "wins" : (diff < 0 ? "loses" : "draws"));

    /* 6. Complexity */
    printf("\n[6] Complexity\n");
    printf("    | Algorithm         | Time          |\n");
    printf("    |-------------------|---------------|\n");
    printf("    | Nim XOR          | O(n)          |\n");
    printf("    | Grundy compute   | O(n x k)      |\n");
    printf("    | Minimax          | O(b^d)        |\n");
    printf("    | Alpha-Beta       | O(b^(d/2))    |\n");
    printf("    | Stone Game DP    | O(n^2)        |\n");
    printf("    b: branching factor, d: depth, k: possible moves\n");

    printf("\n============================================================\n");

    return 0;
}
