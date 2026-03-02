/*
 * Time Complexity
 * Big O Notation and Complexity Analysis
 *
 * A method for analyzing algorithm efficiency.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <functional>

using namespace std;

// =============================================================================
// 1. O(1) - Constant Time
// =============================================================================

int constantTime(const vector<int>& arr) {
    // Always takes the same time regardless of array size
    return arr[0];
}

// =============================================================================
// 2. O(log n) - Logarithmic Time
// =============================================================================

int binarySearch(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }

    return -1;
}

// =============================================================================
// 3. O(n) - Linear Time
// =============================================================================

int linearSearch(const vector<int>& arr, int target) {
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

int sumArray(const vector<int>& arr) {
    int sum = 0;
    for (int x : arr) {
        sum += x;
    }
    return sum;
}

// =============================================================================
// 4. O(n log n) - Linearithmic Time
// =============================================================================

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> L(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> R(arr.begin() + mid + 1, arr.begin() + right + 1);

    size_t i = 0, j = 0;
    int k = left;

    while (i < L.size() && j < R.size()) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < L.size()) arr[k++] = L[i++];
    while (j < R.size()) arr[k++] = R[j++];
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// =============================================================================
// 5. O(n^2) - Quadratic Time
// =============================================================================

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int countPairs(const vector<int>& arr) {
    int count = 0;
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            count++;
        }
    }
    return count;  // n*(n-1)/2
}

// =============================================================================
// 6. O(2^n) - Exponential Time
// =============================================================================

int fibonacciRecursive(int n) {
    if (n <= 1) return n;
    return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
}

// O(n) optimized version
int fibonacciIterative(int n) {
    if (n <= 1) return n;

    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// =============================================================================
// 7. Time Measurement Utility
// =============================================================================

template<typename Func, typename... Args>
double measureTime(Func func, Args&&... args) {
    auto start = chrono::high_resolution_clock::now();
    func(forward<Args>(args)...);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    return diff.count();
}

// =============================================================================
// 8. Space Complexity Examples
// =============================================================================

// O(1) space
void reverseInPlace(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n / 2; i++) {
        swap(arr[i], arr[n - 1 - i]);
    }
}

// O(n) space
vector<int> reverseWithCopy(const vector<int>& arr) {
    vector<int> result(arr.rbegin(), arr.rend());
    return result;
}

// =============================================================================
// Test
// =============================================================================

void printArray(const vector<int>& arr, int limit = 10) {
    cout << "    [";
    for (size_t i = 0; i < arr.size() && i < (size_t)limit; i++) {
        cout << arr[i];
        if (i < arr.size() - 1 && i < (size_t)limit - 1) cout << ", ";
    }
    if (arr.size() > (size_t)limit) cout << ", ...";
    cout << "]" << endl;
}

int main() {
    cout << "============================================================" << endl;
    cout << "Time Complexity Examples" << endl;
    cout << "============================================================" << endl;

    // 1. O(1)
    cout << "\n[1] O(1) - Constant Time" << endl;
    vector<int> arr1 = {5, 2, 8, 1, 9};
    cout << "    First element: " << constantTime(arr1) << endl;

    // 2. O(log n)
    cout << "\n[2] O(log n) - Binary Search" << endl;
    vector<int> arr2 = {1, 3, 5, 7, 9, 11, 13, 15};
    int idx = binarySearch(arr2, 7);
    cout << "    Array: [1,3,5,7,9,11,13,15]" << endl;
    cout << "    Position of 7: " << idx << endl;

    // 3. O(n)
    cout << "\n[3] O(n) - Linear Search" << endl;
    vector<int> arr3 = {4, 2, 7, 1, 9, 3};
    cout << "    Array sum: " << sumArray(arr3) << endl;

    // 4. O(n log n)
    cout << "\n[4] O(n log n) - Merge Sort" << endl;
    vector<int> arr4 = {64, 34, 25, 12, 22, 11, 90};
    cout << "    Before sorting: ";
    printArray(arr4);
    mergeSort(arr4, 0, arr4.size() - 1);
    cout << "    After sorting: ";
    printArray(arr4);

    // 5. O(n^2)
    cout << "\n[5] O(n^2) - Bubble Sort" << endl;
    vector<int> arr5 = {64, 34, 25, 12, 22, 11, 90};
    bubbleSort(arr5);
    cout << "    After sorting: ";
    printArray(arr5);
    cout << "    Number of pairs from 5 elements: " << countPairs(vector<int>(5)) << endl;

    // 6. O(2^n) vs O(n)
    cout << "\n[6] O(2^n) vs O(n) - Fibonacci" << endl;
    cout << "    Fibonacci(20) recursive: " << fibonacciRecursive(20) << endl;
    cout << "    Fibonacci(20) iterative: " << fibonacciIterative(20) << endl;
    cout << "    Fibonacci(40) iterative: " << fibonacciIterative(40) << endl;

    // 7. Space Complexity
    cout << "\n[7] Space Complexity" << endl;
    vector<int> arr7 = {1, 2, 3, 4, 5};
    cout << "    Original: ";
    printArray(arr7);
    reverseInPlace(arr7);
    cout << "    O(1) space reverse: ";
    printArray(arr7);

    // 8. Complexity Summary
    cout << "\n[8] Complexity Summary" << endl;
    cout << "    | Complexity | 1000 ops    | Example           |" << endl;
    cout << "    |------------|-------------|-------------------|" << endl;
    cout << "    | O(1)       | 1           | Array indexing    |" << endl;
    cout << "    | O(log n)   | 10          | Binary search     |" << endl;
    cout << "    | O(n)       | 1000        | Linear search     |" << endl;
    cout << "    | O(n log n) | 10000       | Merge sort        |" << endl;
    cout << "    | O(n^2)     | 1000000     | Bubble sort       |" << endl;
    cout << "    | O(2^n)     | Very large  | All subsets       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
