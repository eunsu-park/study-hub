/*
 * Backtracking
 * N-Queens, Permutations, Combinations, Sudoku
 *
 * Explores all possibilities while pruning unpromising paths.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

/* =============================================================================
 * 1. N-Queens Problem
 * ============================================================================= */

bool is_safe(int board[], int row, int col, int n) {
    for (int i = 0; i < row; i++) {
        /* Same column */
        if (board[i] == col)
            return false;
        /* Diagonal */
        if (abs(board[i] - col) == abs(i - row))
            return false;
    }
    return true;
}

void print_queens_board(int board[], int n) {
    for (int i = 0; i < n; i++) {
        printf("      ");
        for (int j = 0; j < n; j++) {
            printf("%c ", board[i] == j ? 'Q' : '.');
        }
        printf("\n");
    }
}

int solve_queens(int board[], int row, int n, int print_solutions) {
    if (row == n) {
        if (print_solutions) {
            printf("    Solution:\n");
            print_queens_board(board, n);
        }
        return 1;
    }

    int count = 0;
    for (int col = 0; col < n; col++) {
        if (is_safe(board, row, col, n)) {
            board[row] = col;
            count += solve_queens(board, row + 1, n, print_solutions);
            board[row] = -1;  /* Backtrack */
        }
    }

    return count;
}

int n_queens(int n, int print_solutions) {
    int* board = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) board[i] = -1;

    int count = solve_queens(board, 0, n, print_solutions);
    free(board);
    return count;
}

/* =============================================================================
 * 2. Permutations
 * ============================================================================= */

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void permute_impl(int arr[], int start, int end, int* count) {
    if (start == end) {
        (*count)++;
        printf("      [");
        for (int i = 0; i <= end; i++) {
            printf("%d", arr[i]);
            if (i < end) printf(", ");
        }
        printf("]\n");
        return;
    }

    for (int i = start; i <= end; i++) {
        swap(&arr[start], &arr[i]);
        permute_impl(arr, start + 1, end, count);
        swap(&arr[start], &arr[i]);  /* Backtrack */
    }
}

int permutations(int arr[], int n) {
    int count = 0;
    permute_impl(arr, 0, n - 1, &count);
    return count;
}

/* =============================================================================
 * 3. Combinations
 * ============================================================================= */

void combine_impl(int n, int k, int start, int* current, int idx, int* count) {
    if (idx == k) {
        (*count)++;
        printf("      [");
        for (int i = 0; i < k; i++) {
            printf("%d", current[i]);
            if (i < k - 1) printf(", ");
        }
        printf("]\n");
        return;
    }

    for (int i = start; i <= n; i++) {
        current[idx] = i;
        combine_impl(n, k, i + 1, current, idx + 1, count);
    }
}

int combinations(int n, int k) {
    int* current = malloc(k * sizeof(int));
    int count = 0;
    combine_impl(n, k, 1, current, 0, &count);
    free(current);
    return count;
}

/* =============================================================================
 * 4. Subsets
 * ============================================================================= */

void subsets_impl(int arr[], int n, int* current, int current_size, int index) {
    /* Print current subset */
    printf("      {");
    for (int i = 0; i < current_size; i++) {
        printf("%d", current[i]);
        if (i < current_size - 1) printf(", ");
    }
    printf("}\n");

    for (int i = index; i < n; i++) {
        current[current_size] = arr[i];
        subsets_impl(arr, n, current, current_size + 1, i + 1);
    }
}

void subsets(int arr[], int n) {
    int* current = malloc(n * sizeof(int));
    subsets_impl(arr, n, current, 0, 0);
    free(current);
}

/* =============================================================================
 * 5. Sudoku Solver
 * ============================================================================= */

#define SUDOKU_SIZE 9

bool is_valid_sudoku(int board[9][9], int row, int col, int num) {
    /* Row check */
    for (int x = 0; x < 9; x++) {
        if (board[row][x] == num) return false;
    }

    /* Column check */
    for (int x = 0; x < 9; x++) {
        if (board[x][col] == num) return false;
    }

    /* 3x3 box check */
    int start_row = row - row % 3;
    int start_col = col - col % 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[start_row + i][start_col + j] == num)
                return false;
        }
    }

    return true;
}

bool solve_sudoku(int board[9][9]) {
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            if (board[row][col] == 0) {
                for (int num = 1; num <= 9; num++) {
                    if (is_valid_sudoku(board, row, col, num)) {
                        board[row][col] = num;
                        if (solve_sudoku(board))
                            return true;
                        board[row][col] = 0;  /* Backtrack */
                    }
                }
                return false;  /* No valid number found */
            }
        }
    }
    return true;  /* All cells filled */
}

void print_sudoku(int board[9][9]) {
    for (int i = 0; i < 9; i++) {
        if (i % 3 == 0 && i != 0)
            printf("      ------+-------+------\n");
        printf("      ");
        for (int j = 0; j < 9; j++) {
            if (j % 3 == 0 && j != 0) printf("| ");
            printf("%d ", board[i][j]);
        }
        printf("\n");
    }
}

/* =============================================================================
 * 6. String Permutations
 * ============================================================================= */

void string_permute(char str[], int start, int end) {
    if (start == end) {
        printf("      %s\n", str);
        return;
    }

    for (int i = start; i <= end; i++) {
        char temp = str[start];
        str[start] = str[i];
        str[i] = temp;

        string_permute(str, start + 1, end);

        temp = str[start];
        str[start] = str[i];
        str[i] = temp;
    }
}

/* =============================================================================
 * 7. Combination Sum
 * ============================================================================= */

void combination_sum_impl(int candidates[], int n, int target, int start,
                          int* current, int current_size, int* count) {
    if (target == 0) {
        (*count)++;
        printf("      [");
        for (int i = 0; i < current_size; i++) {
            printf("%d", current[i]);
            if (i < current_size - 1) printf(", ");
        }
        printf("]\n");
        return;
    }

    for (int i = start; i < n; i++) {
        if (candidates[i] > target) break;

        current[current_size] = candidates[i];
        combination_sum_impl(candidates, n, target - candidates[i],
                            i, current, current_size + 1, count);
    }
}

int combination_sum(int candidates[], int n, int target) {
    int* current = malloc(target * sizeof(int));
    int count = 0;
    combination_sum_impl(candidates, n, target, 0, current, 0, &count);
    free(current);
    return count;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Backtracking Examples\n");
    printf("============================================================\n");

    /* 1. N-Queens */
    printf("\n[1] N-Queens Problem\n");
    printf("    4-Queens solutions:\n");
    int solutions_4 = n_queens(4, 1);
    printf("    4-Queens solution count: %d\n", solutions_4);
    printf("    8-Queens solution count: %d\n", n_queens(8, 0));

    /* 2. Permutations */
    printf("\n[2] Permutations\n");
    int arr2[] = {1, 2, 3};
    printf("    Permutations of [1, 2, 3]:\n");
    int perm_count = permutations(arr2, 3);
    printf("    Total: %d\n", perm_count);

    /* 3. Combinations */
    printf("\n[3] Combinations\n");
    printf("    C(4, 2):\n");
    int comb_count = combinations(4, 2);
    printf("    Total: %d\n", comb_count);

    /* 4. Subsets */
    printf("\n[4] Subsets\n");
    int arr4[] = {1, 2, 3};
    printf("    Subsets of {1, 2, 3}:\n");
    subsets(arr4, 3);

    /* 5. Sudoku */
    printf("\n[5] Sudoku Solver\n");
    int sudoku[9][9] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    printf("    Initial state:\n");
    print_sudoku(sudoku);

    if (solve_sudoku(sudoku)) {
        printf("\n    Solved:\n");
        print_sudoku(sudoku);
    }

    /* 6. String Permutations */
    printf("\n[6] String Permutations\n");
    char str[] = "ABC";
    printf("    Permutations of 'ABC':\n");
    string_permute(str, 0, 2);

    /* 7. Combination Sum */
    printf("\n[7] Combination Sum\n");
    int candidates[] = {2, 3, 6, 7};
    printf("    [2,3,6,7], target=7:\n");
    combination_sum(candidates, 4, 7);

    /* 8. Backtracking Summary */
    printf("\n[8] Backtracking Pattern\n");
    printf("    1. Check solution (base case)\n");
    printf("    2. Generate candidates\n");
    printf("    3. Feasibility check (pruning)\n");
    printf("    4. Recursive call\n");
    printf("    5. Restore state (backtrack)\n");

    printf("\n============================================================\n");

    return 0;
}
