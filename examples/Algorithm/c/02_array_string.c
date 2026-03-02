/*
 * Array and String
 * Two Pointer, Sliding Window, Prefix Sum
 *
 * Techniques for efficiently handling arrays and strings.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* =============================================================================
 * 1. Two Pointers
 * ============================================================================= */

/* Find a pair in a sorted array whose sum equals target */
bool two_sum_sorted(int arr[], int n, int target, int* i, int* j) {
    int left = 0, right = n - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) {
            *i = left;
            *j = right;
            return true;
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return false;
}

/* Reverse an array */
void reverse_array(int arr[], int left, int right) {
    while (left < right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
        left++;
        right--;
    }
}

/* Remove duplicates (sorted array) */
int remove_duplicates(int arr[], int n) {
    if (n == 0) return 0;

    int write = 1;
    for (int read = 1; read < n; read++) {
        if (arr[read] != arr[read - 1]) {
            arr[write++] = arr[read];
        }
    }
    return write;
}

/* =============================================================================
 * 2. Sliding Window
 * ============================================================================= */

/* Maximum sum of a fixed-size window */
int max_sum_subarray(int arr[], int n, int k) {
    if (n < k) return -1;

    int window_sum = 0;
    for (int i = 0; i < k; i++) {
        window_sum += arr[i];
    }

    int max_sum = window_sum;
    for (int i = k; i < n; i++) {
        window_sum += arr[i] - arr[i - k];
        if (window_sum > max_sum) {
            max_sum = window_sum;
        }
    }

    return max_sum;
}

/* Minimum length subarray with sum >= target */
int min_subarray_len(int arr[], int n, int target) {
    int left = 0;
    int sum = 0;
    int min_len = n + 1;

    for (int right = 0; right < n; right++) {
        sum += arr[right];

        while (sum >= target) {
            int len = right - left + 1;
            if (len < min_len) min_len = len;
            sum -= arr[left++];
        }
    }

    return (min_len == n + 1) ? 0 : min_len;
}

/* Maximum length of consecutive 1s after flipping at most k zeros to 1 */
int longest_ones(int arr[], int n, int k) {
    int left = 0;
    int zeros = 0;
    int max_len = 0;

    for (int right = 0; right < n; right++) {
        if (arr[right] == 0) zeros++;

        while (zeros > k) {
            if (arr[left] == 0) zeros--;
            left++;
        }

        int len = right - left + 1;
        if (len > max_len) max_len = len;
    }

    return max_len;
}

/* =============================================================================
 * 3. Prefix Sum
 * ============================================================================= */

/* Build prefix sum array */
int* build_prefix_sum(int arr[], int n) {
    int* prefix = malloc((n + 1) * sizeof(int));
    prefix[0] = 0;
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + arr[i];
    }
    return prefix;
}

/* Range sum query [left, right] */
int range_sum(int prefix[], int left, int right) {
    return prefix[right + 1] - prefix[left];
}

/* 2D Prefix Sum */
typedef struct {
    int** prefix;
    int rows;
    int cols;
} PrefixSum2D;

PrefixSum2D* build_prefix_sum_2d(int** matrix, int rows, int cols) {
    PrefixSum2D* ps = malloc(sizeof(PrefixSum2D));
    ps->rows = rows;
    ps->cols = cols;
    ps->prefix = malloc((rows + 1) * sizeof(int*));

    for (int i = 0; i <= rows; i++) {
        ps->prefix[i] = calloc(cols + 1, sizeof(int));
    }

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            ps->prefix[i][j] = matrix[i-1][j-1]
                             + ps->prefix[i-1][j]
                             + ps->prefix[i][j-1]
                             - ps->prefix[i-1][j-1];
        }
    }

    return ps;
}

int query_2d(PrefixSum2D* ps, int r1, int c1, int r2, int c2) {
    return ps->prefix[r2+1][c2+1]
         - ps->prefix[r1][c2+1]
         - ps->prefix[r2+1][c1]
         + ps->prefix[r1][c1];
}

void free_prefix_sum_2d(PrefixSum2D* ps) {
    for (int i = 0; i <= ps->rows; i++) {
        free(ps->prefix[i]);
    }
    free(ps->prefix);
    free(ps);
}

/* =============================================================================
 * 4. String Processing
 * ============================================================================= */

/* Palindrome check */
bool is_palindrome(const char* s) {
    int left = 0;
    int right = strlen(s) - 1;

    while (left < right) {
        if (s[left] != s[right]) return false;
        left++;
        right--;
    }
    return true;
}

/* Anagram check */
bool is_anagram(const char* s1, const char* s2) {
    if (strlen(s1) != strlen(s2)) return false;

    int count[26] = {0};

    for (int i = 0; s1[i]; i++) {
        count[s1[i] - 'a']++;
        count[s2[i] - 'a']--;
    }

    for (int i = 0; i < 26; i++) {
        if (count[i] != 0) return false;
    }
    return true;
}

/* Longest common prefix */
char* longest_common_prefix(char* strs[], int n) {
    if (n == 0) return "";

    char* prefix = malloc(strlen(strs[0]) + 1);
    strcpy(prefix, strs[0]);

    for (int i = 1; i < n; i++) {
        int j = 0;
        while (prefix[j] && strs[i][j] && prefix[j] == strs[i][j]) {
            j++;
        }
        prefix[j] = '\0';
    }

    return prefix;
}

/* =============================================================================
 * 5. Kadane's Algorithm (Maximum Subarray Sum)
 * ============================================================================= */

int max_subarray_sum(int arr[], int n) {
    int max_ending_here = arr[0];
    int max_so_far = arr[0];

    for (int i = 1; i < n; i++) {
        max_ending_here = (arr[i] > max_ending_here + arr[i])
                        ? arr[i] : max_ending_here + arr[i];
        if (max_ending_here > max_so_far) {
            max_so_far = max_ending_here;
        }
    }

    return max_so_far;
}

/* =============================================================================
 * Test
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]");
}

int main(void) {
    printf("============================================================\n");
    printf("Array and String Examples\n");
    printf("============================================================\n");

    /* 1. Two Pointers */
    printf("\n[1] Two Pointers - Two Sum\n");
    int arr1[] = {2, 7, 11, 15};
    int i, j;
    if (two_sum_sorted(arr1, 4, 9, &i, &j)) {
        printf("    Array: [2,7,11,15], target=9\n");
        printf("    Indices: (%d, %d)\n", i, j);
    }

    printf("\n[2] Remove Duplicates\n");
    int arr2[] = {1, 1, 2, 2, 3, 4, 4};
    printf("    Original: ");
    print_array(arr2, 7);
    int new_len = remove_duplicates(arr2, 7);
    printf("\n    Result: ");
    print_array(arr2, new_len);
    printf(" (length: %d)\n", new_len);

    /* 2. Sliding Window */
    printf("\n[3] Sliding Window - Maximum Sum\n");
    int arr3[] = {2, 1, 5, 1, 3, 2};
    printf("    Array: [2,1,5,1,3,2], k=3\n");
    printf("    Maximum sum: %d\n", max_sum_subarray(arr3, 6, 3));

    printf("\n[4] Minimum Length with Sum >= Target\n");
    int arr4[] = {2, 3, 1, 2, 4, 3};
    printf("    Array: [2,3,1,2,4,3], target=7\n");
    printf("    Minimum length: %d\n", min_subarray_len(arr4, 6, 7));

    printf("\n[5] Max Consecutive 1s After Flipping k Zeros\n");
    int arr5[] = {1, 1, 0, 0, 1, 1, 1, 0, 1};
    printf("    Array: [1,1,0,0,1,1,1,0,1], k=2\n");
    printf("    Maximum length: %d\n", longest_ones(arr5, 9, 2));

    /* 3. Prefix Sum */
    printf("\n[6] Prefix Sum\n");
    int arr6[] = {1, 2, 3, 4, 5};
    int* prefix = build_prefix_sum(arr6, 5);
    printf("    Array: [1,2,3,4,5]\n");
    printf("    Range sum [1,3]: %d\n", range_sum(prefix, 1, 3));
    free(prefix);

    /* 4. String */
    printf("\n[7] Palindrome Check\n");
    printf("    'racecar': %s\n", is_palindrome("racecar") ? "true" : "false");
    printf("    'hello': %s\n", is_palindrome("hello") ? "true" : "false");

    printf("\n[8] Anagram Check\n");
    printf("    'listen', 'silent': %s\n",
           is_anagram("listen", "silent") ? "true" : "false");

    /* 5. Kadane */
    printf("\n[9] Kadane's Algorithm - Maximum Subarray Sum\n");
    int arr9[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    printf("    Array: [-2,1,-3,4,-1,2,1,-5,4]\n");
    printf("    Maximum sum: %d\n", max_subarray_sum(arr9, 9));

    /* 10. Algorithm Summary */
    printf("\n[10] Technique Summary\n");
    printf("    | Technique      | Time       | Use Case                |\n");
    printf("    |----------------|------------|-------------------------|\n");
    printf("    | Two Pointers   | O(n)       | Sorted array, both ends |\n");
    printf("    | Sliding Window | O(n)       | Contiguous subarray     |\n");
    printf("    | Prefix Sum     | O(n)/O(1)  | Range sum queries       |\n");
    printf("    | Kadane         | O(n)       | Maximum subarray sum    |\n");

    printf("\n============================================================\n");

    return 0;
}
