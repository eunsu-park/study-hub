/*
 * Game Theory
 * Nim, Sprague-Grundy, Minimax, Alpha-Beta Pruning
 *
 * Algorithms for finding optimal strategies in games.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include <climits>

using namespace std;

// =============================================================================
// 1. Nim Game
// =============================================================================

// XOR of all piles is 0 -> second player wins, otherwise first player wins
bool canWinNim(const vector<int>& piles) {
    int xorSum = 0;
    for (int pile : piles) {
        xorSum ^= pile;
    }
    return xorSum != 0;
}

// Find the winning first move
pair<int, int> findNimMove(const vector<int>& piles) {
    int xorSum = 0;
    for (int pile : piles) {
        xorSum ^= pile;
    }

    if (xorSum == 0) {
        return {-1, -1};  // Cannot win
    }

    for (int i = 0; i < (int)piles.size(); i++) {
        int target = piles[i] ^ xorSum;
        if (target < piles[i]) {
            return {i, piles[i] - target};  // Remove this many from pile i
        }
    }

    return {-1, -1};
}

// =============================================================================
// 2. Sprague-Grundy Theorem
// =============================================================================

// Compute Grundy number
int grundyNumber(int n, function<vector<int>(int)> getMoves, vector<int>& memo) {
    if (memo[n] != -1) return memo[n];

    unordered_set<int> reachable;
    for (int next : getMoves(n)) {
        reachable.insert(grundyNumber(next, getMoves, memo));
    }

    // MEX (Minimum Excludant) computation
    int mex = 0;
    while (reachable.count(mex)) mex++;

    return memo[n] = mex;
}

// Grundy number of multiple games is XOR
int multiGameGrundy(const vector<int>& grundyNumbers) {
    int result = 0;
    for (int g : grundyNumbers) {
        result ^= g;
    }
    return result;
}

// =============================================================================
// 3. Minimax
// =============================================================================

class TicTacToe {
private:
    vector<vector<int>> board;  // 0: empty, 1: X, 2: O
    const int EMPTY = 0, X = 1, O = 2;

    int checkWin() {
        // Rows
        for (int i = 0; i < 3; i++) {
            if (board[i][0] != EMPTY &&
                board[i][0] == board[i][1] && board[i][1] == board[i][2]) {
                return board[i][0];
            }
        }
        // Columns
        for (int j = 0; j < 3; j++) {
            if (board[0][j] != EMPTY &&
                board[0][j] == board[1][j] && board[1][j] == board[2][j]) {
                return board[0][j];
            }
        }
        // Diagonals
        if (board[0][0] != EMPTY &&
            board[0][0] == board[1][1] && board[1][1] == board[2][2]) {
            return board[0][0];
        }
        if (board[0][2] != EMPTY &&
            board[0][2] == board[1][1] && board[1][1] == board[2][0]) {
            return board[0][2];
        }
        return 0;
    }

    bool isFull() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[i][j] == EMPTY) return false;
            }
        }
        return true;
    }

public:
    TicTacToe() : board(3, vector<int>(3, 0)) {}

    // Minimax: find optimal move
    int minimax(bool isMaximizing) {
        int winner = checkWin();
        if (winner == X) return 1;
        if (winner == O) return -1;
        if (isFull()) return 0;

        if (isMaximizing) {
            int bestScore = INT_MIN;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (board[i][j] == EMPTY) {
                        board[i][j] = X;
                        bestScore = max(bestScore, minimax(false));
                        board[i][j] = EMPTY;
                    }
                }
            }
            return bestScore;
        } else {
            int bestScore = INT_MAX;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (board[i][j] == EMPTY) {
                        board[i][j] = O;
                        bestScore = min(bestScore, minimax(true));
                        board[i][j] = EMPTY;
                    }
                }
            }
            return bestScore;
        }
    }

    pair<int, int> findBestMove(int player) {
        int bestScore = (player == X) ? INT_MIN : INT_MAX;
        pair<int, int> bestMove = {-1, -1};

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[i][j] == EMPTY) {
                    board[i][j] = player;
                    int score = minimax(player == O);
                    board[i][j] = EMPTY;

                    if (player == X && score > bestScore) {
                        bestScore = score;
                        bestMove = {i, j};
                    } else if (player == O && score < bestScore) {
                        bestScore = score;
                        bestMove = {i, j};
                    }
                }
            }
        }

        return bestMove;
    }

    void makeMove(int row, int col, int player) {
        board[row][col] = player;
    }

    void print() {
        for (int i = 0; i < 3; i++) {
            cout << "      ";
            for (int j = 0; j < 3; j++) {
                char c = board[i][j] == EMPTY ? '.' : (board[i][j] == X ? 'X' : 'O');
                cout << c << " ";
            }
            cout << endl;
        }
    }
};

// =============================================================================
// 4. Alpha-Beta Pruning
// =============================================================================

int alphabeta(vector<vector<int>>& board, int depth, int alpha, int beta,
              bool isMax, function<int(const vector<vector<int>>&)> evaluate,
              function<vector<pair<int,int>>(const vector<vector<int>>&, int)> getMoves,
              int player) {

    if (depth == 0) {
        return evaluate(board);
    }

    auto moves = getMoves(board, player);
    if (moves.empty()) {
        return evaluate(board);
    }

    if (isMax) {
        int maxEval = INT_MIN;
        for (auto [r, c] : moves) {
            board[r][c] = player;
            int eval = alphabeta(board, depth - 1, alpha, beta, false,
                                 evaluate, getMoves, 3 - player);
            board[r][c] = 0;
            maxEval = max(maxEval, eval);
            alpha = max(alpha, eval);
            if (beta <= alpha) break;  // Pruning
        }
        return maxEval;
    } else {
        int minEval = INT_MAX;
        for (auto [r, c] : moves) {
            board[r][c] = player;
            int eval = alphabeta(board, depth - 1, alpha, beta, true,
                                 evaluate, getMoves, 3 - player);
            board[r][c] = 0;
            minEval = min(minEval, eval);
            beta = min(beta, eval);
            if (beta <= alpha) break;  // Pruning
        }
        return minEval;
    }
}

// =============================================================================
// 5. Stone Game Variants
// =============================================================================

// Stone game where you can take 1, 2, or 3 stones
bool stoneGame123(int n) {
    // If n is a multiple of 4, second player wins
    return n % 4 != 0;
}

// Stone game where you can take powers of 2
bool stoneGamePowerOf2(int n) {
    vector<int> grundy(n + 1, 0);

    for (int i = 1; i <= n; i++) {
        unordered_set<int> reachable;
        for (int p = 1; p <= i; p *= 2) {
            reachable.insert(grundy[i - p]);
        }
        int mex = 0;
        while (reachable.count(mex)) mex++;
        grundy[i] = mex;
    }

    return grundy[n] != 0;
}

// =============================================================================
// 6. Bash Game
// =============================================================================

// Stone game where you can take at most k stones
bool bashGame(int n, int k) {
    return n % (k + 1) != 0;
}

// =============================================================================
// 7. Wythoff's Game
// =============================================================================

// Two piles, can take from one pile or equal amount from both
bool wythoffGame(int a, int b) {
    if (a > b) swap(a, b);

    const double phi = (1 + sqrt(5)) / 2;
    int coldA = (int)(phi * (b - a));

    return a != coldA;
}

// =============================================================================
// Test
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "Game Theory Example" << endl;
    cout << "============================================================" << endl;

    // 1. Nim Game
    cout << "\n[1] Nim Game" << endl;
    vector<int> piles = {3, 4, 5};
    cout << "    Piles: [3, 4, 5]" << endl;
    cout << "    XOR: " << (3 ^ 4 ^ 5) << endl;
    cout << "    First player wins: " << (canWinNim(piles) ? "Yes" : "No") << endl;

    auto [pileIdx, removeCount] = findNimMove(piles);
    if (pileIdx != -1) {
        cout << "    Optimal move: remove " << removeCount << " from pile " << pileIdx << endl;
    }

    // 2. Sprague-Grundy
    cout << "\n[2] Sprague-Grundy" << endl;
    // Grundy numbers for a game where you can take 1, 2, or 3
    auto moves123 = [](int n) -> vector<int> {
        vector<int> result;
        if (n >= 1) result.push_back(n - 1);
        if (n >= 2) result.push_back(n - 2);
        if (n >= 3) result.push_back(n - 3);
        return result;
    };

    vector<int> memo(20, -1);
    memo[0] = 0;
    cout << "    Stone game (1,2,3) Grundy numbers:" << endl;
    cout << "    ";
    for (int i = 0; i < 10; i++) {
        cout << "G(" << i << ")=" << grundyNumber(i, moves123, memo) << " ";
    }
    cout << endl;

    // 3. Minimax (Tic-Tac-Toe)
    cout << "\n[3] Tic-Tac-Toe Minimax" << endl;
    TicTacToe game;
    game.makeMove(0, 0, 1);  // X
    game.makeMove(1, 1, 2);  // O
    cout << "    Current state:" << endl;
    game.print();

    auto [bestRow, bestCol] = game.findBestMove(1);  // X's optimal move
    cout << "    X's optimal move: (" << bestRow << ", " << bestCol << ")" << endl;

    // 4. Stone Game Variants
    cout << "\n[4] Stone Game Variants" << endl;
    cout << "    10 stones (take 1,2,3): " << (stoneGame123(10) ? "First wins" : "Second wins") << endl;
    cout << "    12 stones (take 1,2,3): " << (stoneGame123(12) ? "First wins" : "Second wins") << endl;

    // 5. Bash Game
    cout << "\n[5] Bash Game" << endl;
    cout << "    10 stones, max 3: " << (bashGame(10, 3) ? "First wins" : "Second wins") << endl;
    cout << "    12 stones, max 3: " << (bashGame(12, 3) ? "First wins" : "Second wins") << endl;

    // 6. Wythoff's Game
    cout << "\n[6] Wythoff's Game" << endl;
    cout << "    Piles (3, 5): " << (wythoffGame(3, 5) ? "First wins" : "Second wins") << endl;
    cout << "    Piles (1, 2): " << (wythoffGame(1, 2) ? "First wins" : "Second wins") << endl;

    // 7. Game Theory Summary
    cout << "\n[7] Game Theory Summary" << endl;
    cout << "    | Game          | Win Condition          |" << endl;
    cout << "    |---------------|------------------------|" << endl;
    cout << "    | Nim           | XOR != 0               |" << endl;
    cout << "    | Bash (k)      | n % (k+1) != 0         |" << endl;
    cout << "    | 1,2,3 stones  | n % 4 != 0             |" << endl;
    cout << "    | Wythoff       | a != floor(phi*(b-a))  |" << endl;
    cout << "    | General game  | Grundy != 0            |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
