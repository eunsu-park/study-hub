/*
 * Searching Algorithms
 * Binary Search, Lower/Upper Bound, Parametric Search
 *
 * Efficient searching techniques.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// =============================================================================
// 1. Basic Binary Search
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

// Recursive version
int binarySearchRecursive(const vector<int>& arr, int target, int left, int right) {
    if (left > right) return -1;

    int mid = left + (right - left) / 2;

    if (arr[mid] == target)
        return mid;
    else if (arr[mid] < target)
        return binarySearchRecursive(arr, target, mid + 1, right);
    else
        return binarySearchRecursive(arr, target, left, mid - 1);
}

// =============================================================================
// 2. Lower Bound / Upper Bound
// =============================================================================

// First position >= target
int lowerBound(const vector<int>& arr, int target) {
    int left = 0, right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

// First position > target
int upperBound(const vector<int>& arr, int target) {
    int left = 0, right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

// Count occurrences of target
int countOccurrences(const vector<int>& arr, int target) {
    return upperBound(arr, target) - lowerBound(arr, target);
}

// =============================================================================
// 3. Search in Rotated Sorted Array
// =============================================================================

int searchRotated(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target)
            return mid;

        // Left half is sorted
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid])
                right = mid - 1;
            else
                left = mid + 1;
        }
        // Right half is sorted
        else {
            if (arr[mid] < target && target <= arr[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }

    return -1;
}

// Find minimum in rotated array
int findMin(const vector<int>& arr) {
    int left = 0, right = arr.size() - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] > arr[right])
            left = mid + 1;
        else
            right = mid;
    }

    return arr[left];
}

// =============================================================================
// 4. Parametric Search
// =============================================================================

// Wood cutting (maximum height)
long long cutWood(const vector<int>& trees, int h) {
    long long total = 0;
    for (int tree : trees) {
        if (tree > h) {
            total += tree - h;
        }
    }
    return total;
}

int maxCuttingHeight(const vector<int>& trees, long long need) {
    int left = 0;
    int right = *max_element(trees.begin(), trees.end());
    int result = 0;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (cutWood(trees, mid) >= need) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

// Minimum shipping capacity
bool canShip(const vector<int>& weights, int capacity, int days) {
    int currentWeight = 0;
    int dayCount = 1;

    for (int w : weights) {
        if (w > capacity) return false;
        if (currentWeight + w > capacity) {
            dayCount++;
            currentWeight = w;
        } else {
            currentWeight += w;
        }
    }

    return dayCount <= days;
}

int shipWithinDays(const vector<int>& weights, int days) {
    int left = *max_element(weights.begin(), weights.end());
    int right = 0;
    for (int w : weights) right += w;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canShip(weights, mid, days))
            right = mid;
        else
            left = mid + 1;
    }

    return left;
}

// =============================================================================
// 5. Real Number Binary Search
// =============================================================================

// Square root (with decimal precision)
double sqrtBinary(double x, double precision = 1e-10) {
    if (x < 0) return -1;
    if (x < 1) {
        double lo = x, hi = 1;
        while (hi - lo > precision) {
            double mid = (lo + hi) / 2;
            if (mid * mid < x)
                lo = mid;
            else
                hi = mid;
        }
        return (lo + hi) / 2;
    }

    double lo = 1, hi = x;
    while (hi - lo > precision) {
        double mid = (lo + hi) / 2;
        if (mid * mid < x)
            lo = mid;
        else
            hi = mid;
    }
    return (lo + hi) / 2;
}

// =============================================================================
// 6. 2D Array Search
// =============================================================================

// Search in row-sorted and column-sorted 2D array
bool searchMatrix(const vector<vector<int>>& matrix, int target) {
    if (matrix.empty()) return false;

    int rows = matrix.size();
    int cols = matrix[0].size();
    int row = 0, col = cols - 1;

    while (row < rows && col >= 0) {
        if (matrix[row][col] == target)
            return true;
        else if (matrix[row][col] > target)
            col--;
        else
            row++;
    }

    return false;
}

// =============================================================================
// Test
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "Searching Algorithm Examples" << endl;
    cout << "============================================================" << endl;

    // 1. Binary Search
    cout << "\n[1] Binary Search" << endl;
    vector<int> arr = {1, 3, 5, 7, 9, 11, 13, 15};
    cout << "    Array: [1,3,5,7,9,11,13,15]" << endl;
    cout << "    Position of 7: " << binarySearch(arr, 7) << endl;
    cout << "    Position of 6: " << binarySearch(arr, 6) << endl;

    // 2. Lower/Upper Bound
    cout << "\n[2] Lower/Upper Bound" << endl;
    vector<int> arr2 = {1, 2, 2, 2, 3, 4, 5};
    cout << "    Array: [1,2,2,2,3,4,5]" << endl;
    cout << "    lower_bound(2): " << lowerBound(arr2, 2) << endl;
    cout << "    upper_bound(2): " << upperBound(arr2, 2) << endl;
    cout << "    Count of 2: " << countOccurrences(arr2, 2) << endl;

    // 3. Rotated Sorted Array
    cout << "\n[3] Rotated Sorted Array" << endl;
    vector<int> rotated = {4, 5, 6, 7, 0, 1, 2};
    cout << "    Array: [4,5,6,7,0,1,2]" << endl;
    cout << "    Position of 0: " << searchRotated(rotated, 0) << endl;
    cout << "    Minimum: " << findMin(rotated) << endl;

    // 4. Parametric Search
    cout << "\n[4] Parametric Search" << endl;
    vector<int> trees = {20, 15, 10, 17};
    cout << "    Trees: [20,15,10,17], needed: 7" << endl;
    cout << "    Max cutting height: " << maxCuttingHeight(trees, 7) << endl;

    vector<int> weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cout << "    Packages: [1-10], 5-day shipping" << endl;
    cout << "    Min capacity: " << shipWithinDays(weights, 5) << endl;

    // 5. Real Number Binary Search
    cout << "\n[5] Real Number Binary Search" << endl;
    cout << "    sqrt(2) = " << sqrtBinary(2) << endl;
    cout << "    sqrt(10) = " << sqrtBinary(10) << endl;

    // 6. 2D Array Search
    cout << "\n[6] 2D Array Search" << endl;
    vector<vector<int>> matrix = {
        {1, 4, 7, 11},
        {2, 5, 8, 12},
        {3, 6, 9, 16},
        {10, 13, 14, 17}
    };
    cout << "    Find 5: " << (searchMatrix(matrix, 5) ? "found" : "not found") << endl;
    cout << "    Find 15: " << (searchMatrix(matrix, 15) ? "found" : "not found") << endl;

    // 7. Complexity Summary
    cout << "\n[7] Complexity Summary" << endl;
    cout << "    | Algorithm         | Time           |" << endl;
    cout << "    |-------------------|----------------|" << endl;
    cout << "    | Linear Search     | O(n)           |" << endl;
    cout << "    | Binary Search     | O(log n)       |" << endl;
    cout << "    | Parametric Search | O(log M * f(n))|" << endl;
    cout << "    | 2D Matrix Search  | O(m + n)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
