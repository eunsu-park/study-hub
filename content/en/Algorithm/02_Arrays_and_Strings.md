# Arrays and Strings

## Learning Objectives

After completing this lesson, you will be able to:

1. Recall the time complexity of core array operations (access, insertion, deletion, search) and explain when each applies
2. Implement the two pointers technique to solve problems such as pair sums and palindrome checking in O(n) time
3. Apply the sliding window pattern to efficiently compute subarray or substring statistics without redundant computation
4. Use prefix sums to answer range-sum queries in O(1) after O(n) preprocessing
5. Process strings by recognizing character frequency counting as a hash-map pattern and apply it to anagram and duplicate detection problems
6. Select the appropriate array/string technique (two pointers, sliding window, or prefix sum) given a problem's constraints and goals

---

## Overview

Arrays and strings are the most fundamental data structures. This lesson covers key techniques frequently used in array/string problems: two pointers, sliding window, and prefix sum.

---

## Table of Contents

1. [Array Basics](#1-array-basics)
2. [Two Pointers Technique](#2-two-pointers-technique)
3. [Sliding Window](#3-sliding-window)
4. [Prefix Sum](#4-prefix-sum)
5. [String Processing](#5-string-processing)
6. [Frequency Counting](#6-frequency-counting)
7. [Practice Problems](#7-practice-problems)

---

## 1. Array Basics

### Array Characteristics

```
┌─────────────────────────────────────────────────────┐
│ Operation      │ Time        │ Description          │
├────────────────┼─────────────┼──────────────────────┤
│ Index access   │ O(1)        │ arr[i]               │
│ Append         │ O(1)*       │ Dynamic array avg    │
│ Insert middle  │ O(n)        │ Element shift needed │
│ Delete         │ O(n)        │ Element shift needed │
│ Search         │ O(n)        │ Unsorted             │
│ Search (sorted)│ O(log n)    │ Binary search        │
└─────────────────────────────────────────────────────┘
```

### Array Traversal Patterns

```cpp
// C++ - Basic traversal
vector<int> arr = {1, 2, 3, 4, 5};

// Index-based
for (int i = 0; i < arr.size(); i++) {
    cout << arr[i] << " ";
}

// Range-based
for (int x : arr) {
    cout << x << " ";
}

// Reverse traversal
for (int i = arr.size() - 1; i >= 0; i--) {
    cout << arr[i] << " ";
}
```

```python
# Python
arr = [1, 2, 3, 4, 5]

# Basic traversal
for x in arr:
    print(x, end=" ")

# With index
for i, x in enumerate(arr):
    print(f"arr[{i}] = {x}")

# Reverse traversal
for x in reversed(arr):
    print(x, end=" ")
```

---

## 2. Two Pointers Technique

### Concept

```
Two Pointers: Use two pointers to traverse an array
→ Can reduce O(n²) to O(n)

Types:
1. Start from both ends (sorted array)
2. Move in same direction (slow/fast pointers)
```

### 2.1 Two Pointers from Both Ends

**Problem: Find pair with sum equal to target in sorted array**

```
Array: [1, 2, 4, 6, 8, 10]
Target: 10

left → [1]  [2]  [4]  [6]  [8]  [10] ← right
        ↓                          ↓
      1 + 10 = 11 > 10 → right--

left → [1]  [2]  [4]  [6]  [8]  [10]
        ↓                    ↓
      1 + 8 = 9 < 10 → left++

       [1]  [2]  [4]  [6]  [8]  [10]
             ↓              ↓
      2 + 8 = 10 ✓ Found!
```

```c
// C
void twoSum(int arr[], int n, int target) {
    // Start from both ends — works because the array is sorted:
    // if the sum is too small, moving left pointer right increases it;
    // if too large, moving right pointer left decreases it. An unsorted
    // array would require O(n²) nested scan instead.
    int left = 0;
    int right = n - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];

        if (sum == target) {
            printf("Found: %d + %d = %d\n",
                   arr[left], arr[right], target);
            return;
        } else if (sum < target) {
            left++;   // Need a larger value — move toward bigger elements
        } else {
            right--;  // Need a smaller value — move toward smaller elements
        }
    }
    printf("Not found\n");
}
```

```cpp
// C++
pair<int, int> twoSum(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];

        if (sum == target) {
            return {left, right};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return {-1, -1};
}
```

```python
# Python
def two_sum(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return (left, right)
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return (-1, -1)
```

### 2.2 Palindrome Check

```
String: "racecar"

left → [r] [a] [c] [e] [c] [a] [r] ← right
        ↓                       ↓
       'r' == 'r' ✓

       [r] [a] [c] [e] [c] [a] [r]
            ↓               ↓
       'a' == 'a' ✓

       [r] [a] [c] [e] [c] [a] [r]
                ↓       ↓
       'c' == 'c' ✓

       [r] [a] [c] [e] [c] [a] [r]
                    ↓
       left >= right → Palindrome!
```

```c
// C
#include <string.h>
#include <stdbool.h>

bool isPalindrome(const char* s) {
    int left = 0;
    int right = strlen(s) - 1;

    while (left < right) {
        if (s[left] != s[right]) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}
```

```cpp
// C++
bool isPalindrome(const string& s) {
    int left = 0;
    int right = s.length() - 1;

    while (left < right) {
        if (s[left] != s[right]) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}
```

```python
# Python
def is_palindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1

    return True

# Pythonic way
def is_palindrome_simple(s):
    return s == s[::-1]
```

### 2.3 Same Direction Two Pointers (Slow/Fast Pointers)

**Problem: Remove duplicates from sorted array (In-place)**

```
Array: [1, 1, 2, 2, 2, 3]

slow ↓
fast ↓
     [1] [1] [2] [2] [2] [3]
     slow is unique element position, fast explores

     [1] [1] [2] [2] [2] [3]
      ↓   ↓
     arr[slow] == arr[fast] → fast++

     [1] [1] [2] [2] [2] [3]
      ↓       ↓
     arr[slow] != arr[fast] → slow++, copy

     [1] [2] [2] [2] [2] [3]
          ↓   ↓
     ...

Result: [1, 2, 3, _, _, _], 3 unique elements
```

```c
// C
int removeDuplicates(int arr[], int n) {
    if (n == 0) return 0;

    // slow marks the write position — everything at [0..slow] is the
    // deduplicated prefix. Separating read (fast) from write (slow) lets
    // us modify the array in-place without needing extra memory.
    int slow = 0;

    for (int fast = 1; fast < n; fast++) {
        // Only write when we find a new unique value; duplicate runs
        // in a sorted array are contiguous, so a simple != check suffices
        if (arr[fast] != arr[slow]) {
            slow++;
            arr[slow] = arr[fast];
        }
    }

    return slow + 1;  // Number of unique elements
}
```

```cpp
// C++
int removeDuplicates(vector<int>& arr) {
    if (arr.empty()) return 0;

    int slow = 0;

    for (int fast = 1; fast < arr.size(); fast++) {
        if (arr[fast] != arr[slow]) {
            slow++;
            arr[slow] = arr[fast];
        }
    }

    return slow + 1;
}
```

```python
# Python
def remove_duplicates(arr):
    if not arr:
        return 0

    slow = 0

    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1
```

---

## 3. Sliding Window

### Concept

```
Sliding Window: Move a fixed or variable-sized window through array
→ Effective for contiguous subarray/substring problems
→ Can reduce O(n²) to O(n)

Types:
1. Fixed-size window (size k)
2. Variable-size window (satisfies condition)
```

### 3.1 Fixed-Size Window

**Problem: Maximum sum of contiguous subarray of size k**

```
Array: [1, 4, 2, 10, 2, 3, 1, 0, 20]
k = 3

Window movement:
[1, 4, 2] → sum: 7
   [4, 2, 10] → sum: 16
      [2, 10, 2] → sum: 14
         [10, 2, 3] → sum: 15
            [2, 3, 1] → sum: 6
               [3, 1, 0] → sum: 4
                  [1, 0, 20] → sum: 21 ← Maximum!

Optimization: Don't sum k elements each time,
       add new element - remove old element
```

```c
// C - Naive: O(n*k)
int maxSumNaive(int arr[], int n, int k) {
    int maxSum = 0;

    for (int i = 0; i <= n - k; i++) {
        int sum = 0;
        for (int j = 0; j < k; j++) {
            sum += arr[i + j];
        }
        if (sum > maxSum) {
            maxSum = sum;
        }
    }

    return maxSum;
}

// C - Sliding Window: O(n)
int maxSumSliding(int arr[], int n, int k) {
    // Seed the window with the first k elements; we need a valid baseline
    // before we can start sliding — computing it separately avoids an off-by-one
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;

    // Slide window by subtracting the element that falls off the left edge
    // and adding the new element on the right — O(1) update instead of O(k)
    // recomputation, which is the entire point of the sliding window technique
    for (int i = k; i < n; i++) {
        windowSum += arr[i] - arr[i - k];  // Add new, remove old
        if (windowSum > maxSum) {
            maxSum = windowSum;
        }
    }

    return maxSum;
}
```

```cpp
// C++
int maxSumSliding(const vector<int>& arr, int k) {
    int n = arr.size();
    if (n < k) return -1;

    // First window sum
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;

    // Sliding
    for (int i = k; i < n; i++) {
        windowSum += arr[i] - arr[i - k];
        maxSum = max(maxSum, windowSum);
    }

    return maxSum;
}
```

```python
# Python
def max_sum_sliding(arr, k):
    n = len(arr)
    if n < k:
        return -1

    # First window sum
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Sliding
    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

### 3.2 Variable-Size Window

**Problem: Minimum length subarray with sum >= target**

```
Array: [2, 3, 1, 2, 4, 3], target = 7

left=0, right=0: [2] → sum=2 < 7, right++
left=0, right=1: [2,3] → sum=5 < 7, right++
left=0, right=2: [2,3,1] → sum=6 < 7, right++
left=0, right=3: [2,3,1,2] → sum=8 >= 7, len=4, left++
left=1, right=3: [3,1,2] → sum=6 < 7, right++
left=1, right=4: [3,1,2,4] → sum=10 >= 7, len=4, left++
left=2, right=4: [1,2,4] → sum=7 >= 7, len=3, left++
left=3, right=4: [2,4] → sum=6 < 7, right++
left=3, right=5: [2,4,3] → sum=9 >= 7, len=3, left++
left=4, right=5: [4,3] → sum=7 >= 7, len=2 ← Minimum!

Answer: 2
```

```c
// C
int minSubArrayLen(int arr[], int n, int target) {
    // Initialize to n + 1 (one beyond the maximum possible length) so any
    // valid window will be smaller — avoids a special "not found" flag
    int minLen = n + 1;
    int left = 0;
    int sum = 0;

    for (int right = 0; right < n; right++) {
        sum += arr[right];

        // Shrink from the left as long as the window still meets the condition —
        // we want the *shortest* valid window, so we squeeze greedily
        while (sum >= target) {
            int len = right - left + 1;
            if (len < minLen) {
                minLen = len;
            }
            sum -= arr[left];
            left++;
        }
    }

    return (minLen == n + 1) ? 0 : minLen;
}
```

```cpp
// C++
int minSubArrayLen(const vector<int>& arr, int target) {
    int n = arr.size();
    int minLen = INT_MAX;
    int left = 0;
    int sum = 0;

    for (int right = 0; right < n; right++) {
        sum += arr[right];

        while (sum >= target) {
            minLen = min(minLen, right - left + 1);
            sum -= arr[left];
            left++;
        }
    }

    return (minLen == INT_MAX) ? 0 : minLen;
}
```

```python
# Python
def min_sub_array_len(arr, target):
    n = len(arr)
    min_len = float('inf')
    left = 0
    current_sum = 0

    for right in range(n):
        current_sum += arr[right]

        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1

    return 0 if min_len == float('inf') else min_len
```

### 3.3 String Sliding Window

**Problem: Longest substring without repeating characters**

```
String: "abcabcbb"

[a] → set: {a}, length: 1
[a,b] → set: {a,b}, length: 2
[a,b,c] → set: {a,b,c}, length: 3 ← Maximum
[a,b,c,a] → 'a' duplicate! Move left to remove 'a'
[b,c,a] → set: {b,c,a}, length: 3
[b,c,a,b] → 'b' duplicate! Move left
...

Answer: 3 ("abc" or "bca" or "cab")
```

```cpp
// C++
int lengthOfLongestSubstring(const string& s) {
    // A set tracks which characters are currently in the window — O(1)
    // membership test lets us detect duplicates without rescanning the window
    unordered_set<char> seen;
    int maxLen = 0;
    int left = 0;

    for (int right = 0; right < s.length(); right++) {
        // Remove from the left until the new character s[right] is no longer a
        // duplicate — we can't simply jump left to the duplicate's position
        // because the set would go out of sync with the actual window contents
        while (seen.count(s[right])) {
            seen.erase(s[left]);
            left++;
        }

        seen.insert(s[right]);
        maxLen = max(maxLen, right - left + 1);
    }

    return maxLen;
}
```

```python
# Python
def length_of_longest_substring(s):
    seen = set()
    max_len = 0
    left = 0

    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1

        seen.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len
```

---

## 4. Prefix Sum

### Concept

```
Prefix Sum (Cumulative Sum): Calculate range sum in O(1)

Original array:  [1, 2, 3, 4, 5]
Prefix sum:      [1, 3, 6, 10, 15]

prefix[i] = arr[0] + arr[1] + ... + arr[i]

Range sum [i, j]:
sum(i, j) = prefix[j] - prefix[i-1]
          = (arr[0]+...+arr[j]) - (arr[0]+...+arr[i-1])
          = arr[i] + ... + arr[j]
```

### Prefix Sum Visualization

```
Index:       0    1    2    3    4
Original:   [1]  [2]  [3]  [4]  [5]
Prefix:     [1]  [3]  [6] [10] [15]

sum(1, 3) = arr[1] + arr[2] + arr[3]
          = 2 + 3 + 4 = 9

          = prefix[3] - prefix[0]
          = 10 - 1 = 9 ✓
```

### Implementation

```c
// C
// Build prefix sum array
void buildPrefixSum(int arr[], int prefix[], int n) {
    prefix[0] = arr[0];
    for (int i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }
}

// Range sum query [left, right]
int rangeSum(int prefix[], int left, int right) {
    if (left == 0) {
        return prefix[right];
    }
    return prefix[right] - prefix[left - 1];
}
```

```cpp
// C++
class PrefixSum {
private:
    vector<int> prefix;

public:
    PrefixSum(const vector<int>& arr) {
        int n = arr.size();
        prefix.resize(n + 1, 0);

        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + arr[i];
        }
    }

    // Sum of range [left, right]
    int query(int left, int right) {
        return prefix[right + 1] - prefix[left];
    }
};

// Usage example
// vector<int> arr = {1, 2, 3, 4, 5};
// PrefixSum ps(arr);
// cout << ps.query(1, 3);  // 9
```

```python
# Python
class PrefixSum:
    def __init__(self, arr):
        n = len(arr)
        self.prefix = [0] * (n + 1)

        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + arr[i]

    def query(self, left, right):
        """Sum of range [left, right]"""
        return self.prefix[right + 1] - self.prefix[left]


# Usage example
arr = [1, 2, 3, 4, 5]
ps = PrefixSum(arr)
print(ps.query(1, 3))  # 9
```

### 2D Prefix Sum

```
Sum of submatrix in 2D array

Original matrix:            2D prefix:
[1, 2, 3]               [1,  3,  6]
[4, 5, 6]     →         [5, 12, 21]
[7, 8, 9]               [12, 27, 45]

Sum of submatrix (1,1)~(2,2):
= prefix[2][2] - prefix[0][2] - prefix[2][0] + prefix[0][0]
= 45 - 6 - 12 + 1 = 28

Verify: 5+6+8+9 = 28 ✓
```

```cpp
// C++ - 2D Prefix Sum
class PrefixSum2D {
private:
    vector<vector<int>> prefix;

public:
    PrefixSum2D(const vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        // Allocate (m+1) × (n+1) with an extra row and column of zeros —
        // the sentinel border avoids special-casing i==0 or j==0 in the formula
        prefix.resize(m + 1, vector<int>(n + 1, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // Inclusion-exclusion: add top and left prefixes, subtract
                // the top-left region counted twice, then add current cell
                prefix[i + 1][j + 1] = matrix[i][j]
                                     + prefix[i][j + 1]
                                     + prefix[i + 1][j]
                                     - prefix[i][j];
            }
        }
    }

    // Sum of submatrix (r1,c1)~(r2,c2) — O(1) query regardless of submatrix size
    int query(int r1, int c1, int r2, int c2) {
        // Reverse inclusion-exclusion: subtract two non-overlapping outer strips,
        // add back the corner region that was subtracted twice
        return prefix[r2 + 1][c2 + 1]
             - prefix[r1][c2 + 1]
             - prefix[r2 + 1][c1]
             + prefix[r1][c1];
    }
};
```

```python
# Python - 2D Prefix Sum
class PrefixSum2D:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m):
            for j in range(n):
                self.prefix[i + 1][j + 1] = (matrix[i][j]
                                            + self.prefix[i][j + 1]
                                            + self.prefix[i + 1][j]
                                            - self.prefix[i][j])

    def query(self, r1, c1, r2, c2):
        """Sum of submatrix (r1,c1)~(r2,c2)"""
        return (self.prefix[r2 + 1][c2 + 1]
              - self.prefix[r1][c2 + 1]
              - self.prefix[r2 + 1][c1]
              + self.prefix[r1][c1])
```

---

## 5. String Processing

### 5.1 String Reversal

```c
// C
void reverseString(char* s) {
    int left = 0;
    int right = strlen(s) - 1;

    while (left < right) {
        char temp = s[left];
        s[left] = s[right];
        s[right] = temp;
        left++;
        right--;
    }
}
```

```cpp
// C++
void reverseString(string& s) {
    int left = 0, right = s.length() - 1;

    while (left < right) {
        swap(s[left], s[right]);
        left++;
        right--;
    }
}

// STL
// reverse(s.begin(), s.end());
```

```python
# Python
def reverse_string(s):
    return s[::-1]

# In-place (list)
def reverse_string_list(chars):
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
```

### 5.2 Anagram Check

```
Anagram: Different words composed of same characters
Example: "listen" ↔ "silent"

Method 1: Sort and compare - O(n log n)
Method 2: Frequency comparison - O(n)
```

```cpp
// C++ - Frequency method
bool isAnagram(const string& s1, const string& s2) {
    if (s1.length() != s2.length()) return false;

    int count[26] = {0};

    for (char c : s1) count[c - 'a']++;
    for (char c : s2) count[c - 'a']--;

    for (int i = 0; i < 26; i++) {
        if (count[i] != 0) return false;
    }

    return true;
}
```

```python
# Python
from collections import Counter

def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)

# Or sorting method
def is_anagram_sort(s1, s2):
    return sorted(s1) == sorted(s2)
```

---

## 6. Frequency Counting

### Frequency Using HashMap

```cpp
// C++ - Character frequency
unordered_map<char, int> countFrequency(const string& s) {
    unordered_map<char, int> freq;
    for (char c : s) {
        freq[c]++;
    }
    return freq;
}

// Find most frequent character
char mostFrequent(const string& s) {
    unordered_map<char, int> freq;
    for (char c : s) {
        freq[c]++;
    }

    char result = '\0';
    int maxCount = 0;

    for (auto& [c, count] : freq) {
        if (count > maxCount) {
            maxCount = count;
            result = c;
        }
    }

    return result;
}
```

```python
# Python
from collections import Counter

def count_frequency(s):
    return Counter(s)

# Most frequent character
def most_frequent(s):
    freq = Counter(s)
    return freq.most_common(1)[0][0]
```

### Frequency Using Array (Alphabets)

```c
// C - Lowercase only
void countFrequency(const char* s, int freq[]) {
    // freq array should be size 26 initialized to 0
    while (*s) {
        freq[*s - 'a']++;
        s++;
    }
}

// Usage
int freq[26] = {0};
countFrequency("hello", freq);
// freq['h'-'a'] = 1, freq['e'-'a'] = 1, freq['l'-'a'] = 2, freq['o'-'a'] = 1
```

---

## 7. Practice Problems

### Problem 1: Array Rotation

Rotate array to the right by k positions.

```
Input: [1, 2, 3, 4, 5], k = 2
Output: [4, 5, 1, 2, 3]
```

<details>
<summary>Hint</summary>

Can solve with O(1) space using three reversals:
1. Reverse entire array
2. Reverse first k elements
3. Reverse remaining elements

</details>

<details>
<summary>Solution Code</summary>

```python
def rotate(arr, k):
    n = len(arr)
    k = k % n  # k might be larger than n

    def reverse(start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    reverse(0, n - 1)      # [5,4,3,2,1]
    reverse(0, k - 1)      # [4,5,3,2,1]
    reverse(k, n - 1)      # [4,5,1,2,3]

# Time: O(n), Space: O(1)
```

</details>

### Problem 2: Count Subarrays with Sum k

Count the number of contiguous subarrays with sum equal to k.

```
Input: [1, 1, 1], k = 2
Output: 2 (two subarrays [1,1])
```

<details>
<summary>Hint</summary>

Use prefix sum + hash map
prefix[j] - prefix[i] = k
→ prefix[i] = prefix[j] - k

</details>

<details>
<summary>Solution Code</summary>

```python
from collections import defaultdict

def subarray_sum(arr, k):
    count = 0
    prefix_sum = 0
    prefix_count = defaultdict(int)
    prefix_count[0] = 1  # Empty prefix

    for num in arr:
        prefix_sum += num

        # If prefix_sum - k appeared before
        # Sum from that point to current is k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]

        prefix_count[prefix_sum] += 1

    return count

# Time: O(n), Space: O(n)
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|-----------|---------|---------|------|
| ⭐ | [Two Sum](https://leetcode.com/problems/two-sum/) | LeetCode | Hash map |
| ⭐ | [Range Sum Query](https://www.acmicpc.net/problem/11659) | Baekjoon | Prefix sum |
| ⭐⭐ | [3Sum](https://leetcode.com/problems/3sum/) | LeetCode | Two pointers |
| ⭐⭐ | [Longest Substring Without Repeating](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | LeetCode | Sliding window |
| ⭐⭐ | [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) | LeetCode | Kadane's algorithm |
| ⭐⭐⭐ | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | LeetCode | Sliding window |

---

## Next Steps

- [03_Stacks_and_Queues.md](./03_Stacks_and_Queues.md) - Algorithms using stacks/queues

---

## References

- [Two Pointers Technique](https://www.geeksforgeeks.org/two-pointers-technique/)
- [Sliding Window Problems](https://leetcode.com/tag/sliding-window/)
- [Prefix Sum Tutorial](https://usaco.guide/silver/prefix-sums)
