# 배열과 문자열 (Arrays and Strings)

## 개요

배열과 문자열은 가장 기본적인 자료구조입니다. 이 레슨에서는 배열/문자열 문제에서 자주 사용되는 핵심 테크닉인 2포인터, 슬라이딩 윈도우, 프리픽스 합을 학습합니다.

---

## 목차

1. [배열 기초](#1-배열-기초)
2. [2포인터 기법](#2-2포인터-기법)
3. [슬라이딩 윈도우](#3-슬라이딩-윈도우)
4. [프리픽스 합](#4-프리픽스-합)
5. [문자열 처리](#5-문자열-처리)
6. [빈도수 카운팅](#6-빈도수-카운팅)
7. [연습 문제](#7-연습-문제)

---

## 1. 배열 기초

### 배열의 특성

```
┌─────────────────────────────────────────────────────┐
│ 연산           │ 시간 복잡도 │ 설명                 │
├────────────────┼─────────────┼──────────────────────┤
│ 인덱스 접근    │ O(1)        │ arr[i]               │
│ 끝에 삽입     │ O(1)*       │ 동적 배열 평균       │
│ 중간 삽입     │ O(n)        │ 원소 이동 필요       │
│ 삭제          │ O(n)        │ 원소 이동 필요       │
│ 탐색          │ O(n)        │ 정렬 안됨            │
│ 탐색 (정렬됨) │ O(log n)    │ 이진 탐색            │
└─────────────────────────────────────────────────────┘
```

### 배열 순회 패턴

```cpp
// C++ - 기본 순회
vector<int> arr = {1, 2, 3, 4, 5};

// 인덱스 기반
for (int i = 0; i < arr.size(); i++) {
    cout << arr[i] << " ";
}

// 범위 기반
for (int x : arr) {
    cout << x << " ";
}

// 역순 순회
for (int i = arr.size() - 1; i >= 0; i--) {
    cout << arr[i] << " ";
}
```

```python
# Python
arr = [1, 2, 3, 4, 5]

# 기본 순회
for x in arr:
    print(x, end=" ")

# 인덱스와 함께
for i, x in enumerate(arr):
    print(f"arr[{i}] = {x}")

# 역순 순회
for x in reversed(arr):
    print(x, end=" ")
```

---

## 2. 2포인터 기법

### 개념

```
2포인터: 두 개의 포인터를 사용하여 배열을 탐색
→ O(n²)을 O(n)으로 줄일 수 있음

유형:
1. 양끝에서 시작 (정렬된 배열)
2. 같은 방향으로 이동 (느린/빠른 포인터)
```

### 2.1 양끝에서 시작하는 2포인터

**문제: 정렬된 배열에서 두 수의 합이 target인 쌍 찾기**

```
배열: [1, 2, 4, 6, 8, 10]
타겟: 10

left → [1]  [2]  [4]  [6]  [8]  [10] ← right
        ↓                          ↓
      1 + 10 = 11 > 10 → right--

left → [1]  [2]  [4]  [6]  [8]  [10]
        ↓                    ↓
      1 + 8 = 9 < 10 → left++

       [1]  [2]  [4]  [6]  [8]  [10]
             ↓              ↓
      2 + 8 = 10 ✓ 찾음!
```

```c
// C
void twoSum(int arr[], int n, int target) {
    int left = 0;
    int right = n - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];

        if (sum == target) {
            printf("Found: %d + %d = %d\n",
                   arr[left], arr[right], target);
            return;
        } else if (sum < target) {
            left++;
        } else {
            right--;
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

### 2.2 팰린드롬 검사

```
문자열: "racecar"

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
       left >= right → 팰린드롬!
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

### 2.3 같은 방향 2포인터 (느린/빠른 포인터)

**문제: 정렬된 배열에서 중복 제거 (In-place)**

```
배열: [1, 1, 2, 2, 2, 3]

slow ↓
fast ↓
     [1] [1] [2] [2] [2] [3]
     slow는 고유 원소 위치, fast는 탐색

     [1] [1] [2] [2] [2] [3]
      ↓   ↓
     arr[slow] == arr[fast] → fast++

     [1] [1] [2] [2] [2] [3]
      ↓       ↓
     arr[slow] != arr[fast] → slow++, copy

     [1] [2] [2] [2] [2] [3]
          ↓   ↓
     ...

결과: [1, 2, 3, _, _, _], 고유 원소 3개
```

```c
// C
int removeDuplicates(int arr[], int n) {
    if (n == 0) return 0;

    int slow = 0;

    for (int fast = 1; fast < n; fast++) {
        if (arr[fast] != arr[slow]) {
            slow++;
            arr[slow] = arr[fast];
        }
    }

    return slow + 1;  // 고유 원소 개수
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

## 3. 슬라이딩 윈도우

### 개념

```
슬라이딩 윈도우: 고정 또는 가변 크기의 윈도우를 이동하며 탐색
→ 연속된 부분 배열/문자열 문제에 효과적
→ O(n²)을 O(n)으로 줄일 수 있음

유형:
1. 고정 크기 윈도우 (크기 k)
2. 가변 크기 윈도우 (조건 만족)
```

### 3.1 고정 크기 윈도우

**문제: 크기 k인 연속 부분 배열의 최대 합**

```
배열: [1, 4, 2, 10, 2, 3, 1, 0, 20]
k = 3

윈도우 이동:
[1, 4, 2] → 합: 7
   [4, 2, 10] → 합: 16
      [2, 10, 2] → 합: 14
         [10, 2, 3] → 합: 15
            [2, 3, 1] → 합: 6
               [3, 1, 0] → 합: 4
                  [1, 0, 20] → 합: 21 ← 최대!

최적화: 매번 k개를 더하지 않고,
       새 원소 추가 - 이전 원소 제거
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
    // 첫 윈도우 합 계산
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;

    // 윈도우 슬라이딩
    for (int i = k; i < n; i++) {
        windowSum += arr[i] - arr[i - k];  // 새 원소 추가, 이전 원소 제거
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

    // 첫 윈도우 합
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;

    // 슬라이딩
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

    # 첫 윈도우 합
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # 슬라이딩
    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

### 3.2 가변 크기 윈도우

**문제: 합이 target 이상인 최소 길이 부분 배열**

```
배열: [2, 3, 1, 2, 4, 3], target = 7

left=0, right=0: [2] → 합=2 < 7, right++
left=0, right=1: [2,3] → 합=5 < 7, right++
left=0, right=2: [2,3,1] → 합=6 < 7, right++
left=0, right=3: [2,3,1,2] → 합=8 >= 7, 길이=4, left++
left=1, right=3: [3,1,2] → 합=6 < 7, right++
left=1, right=4: [3,1,2,4] → 합=10 >= 7, 길이=4, left++
left=2, right=4: [1,2,4] → 합=7 >= 7, 길이=3, left++
left=3, right=4: [2,4] → 합=6 < 7, right++
left=3, right=5: [2,4,3] → 합=9 >= 7, 길이=3, left++
left=4, right=5: [4,3] → 합=7 >= 7, 길이=2 ← 최소!

답: 2
```

```c
// C
int minSubArrayLen(int arr[], int n, int target) {
    int minLen = n + 1;  // 불가능한 값으로 초기화
    int left = 0;
    int sum = 0;

    for (int right = 0; right < n; right++) {
        sum += arr[right];

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

### 3.3 문자열 슬라이딩 윈도우

**문제: 중복 없는 최장 부분 문자열**

```
문자열: "abcabcbb"

[a] → 집합: {a}, 길이: 1
[a,b] → 집합: {a,b}, 길이: 2
[a,b,c] → 집합: {a,b,c}, 길이: 3 ← 최대
[a,b,c,a] → 'a' 중복! left를 이동하여 'a' 제거
[b,c,a] → 집합: {b,c,a}, 길이: 3
[b,c,a,b] → 'b' 중복! left 이동
...

답: 3 ("abc" 또는 "bca" 또는 "cab")
```

```cpp
// C++
int lengthOfLongestSubstring(const string& s) {
    unordered_set<char> seen;
    int maxLen = 0;
    int left = 0;

    for (int right = 0; right < s.length(); right++) {
        // 중복이 있으면 left 이동
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

## 4. 프리픽스 합

### 개념

```
프리픽스 합 (누적 합): 배열의 구간 합을 O(1)에 계산

원본 배열:    [1, 2, 3, 4, 5]
프리픽스 합:  [1, 3, 6, 10, 15]

prefix[i] = arr[0] + arr[1] + ... + arr[i]

구간 합 [i, j]:
sum(i, j) = prefix[j] - prefix[i-1]
          = (arr[0]+...+arr[j]) - (arr[0]+...+arr[i-1])
          = arr[i] + ... + arr[j]
```

### 프리픽스 합 시각화

```
인덱스:      0    1    2    3    4
원본:       [1]  [2]  [3]  [4]  [5]
프리픽스:   [1]  [3]  [6] [10] [15]

sum(1, 3) = arr[1] + arr[2] + arr[3]
          = 2 + 3 + 4 = 9

          = prefix[3] - prefix[0]
          = 10 - 1 = 9 ✓
```

### 구현

```c
// C
// 프리픽스 합 배열 생성
void buildPrefixSum(int arr[], int prefix[], int n) {
    prefix[0] = arr[0];
    for (int i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }
}

// 구간 합 쿼리 [left, right]
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

    // 구간 [left, right]의 합
    int query(int left, int right) {
        return prefix[right + 1] - prefix[left];
    }
};

// 사용 예
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
        """구간 [left, right]의 합"""
        return self.prefix[right + 1] - self.prefix[left]


# 사용 예
arr = [1, 2, 3, 4, 5]
ps = PrefixSum(arr)
print(ps.query(1, 3))  # 9
```

### 2D 프리픽스 합

```
2차원 배열에서 부분 행렬의 합

원본 행렬:              2D 프리픽스:
[1, 2, 3]               [1,  3,  6]
[4, 5, 6]     →         [5, 12, 21]
[7, 8, 9]               [12, 27, 45]

부분 행렬 (1,1)~(2,2)의 합:
= prefix[2][2] - prefix[0][2] - prefix[2][0] + prefix[0][0]
= 45 - 6 - 12 + 1 = 28

검증: 5+6+8+9 = 28 ✓
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
        prefix.resize(m + 1, vector<int>(n + 1, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                prefix[i + 1][j + 1] = matrix[i][j]
                                     + prefix[i][j + 1]
                                     + prefix[i + 1][j]
                                     - prefix[i][j];
            }
        }
    }

    // (r1,c1)~(r2,c2) 부분 행렬의 합
    int query(int r1, int c1, int r2, int c2) {
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
        """(r1,c1)~(r2,c2) 부분 행렬의 합"""
        return (self.prefix[r2 + 1][c2 + 1]
              - self.prefix[r1][c2 + 1]
              - self.prefix[r2 + 1][c1]
              + self.prefix[r1][c1])
```

---

## 5. 문자열 처리

### 5.1 문자열 뒤집기

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

# In-place (리스트)
def reverse_string_list(chars):
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
```

### 5.2 아나그램 검사

```
아나그램: 같은 문자들로 구성된 다른 단어
예: "listen" ↔ "silent"

방법 1: 정렬 후 비교 - O(n log n)
방법 2: 빈도수 비교 - O(n)
```

```cpp
// C++ - 빈도수 방법
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

# 또는 정렬 방법
def is_anagram_sort(s1, s2):
    return sorted(s1) == sorted(s2)
```

---

## 6. 빈도수 카운팅

### 해시맵을 이용한 빈도수

```cpp
// C++ - 문자 빈도수
unordered_map<char, int> countFrequency(const string& s) {
    unordered_map<char, int> freq;
    for (char c : s) {
        freq[c]++;
    }
    return freq;
}

// 가장 많이 나타난 문자 찾기
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

# 가장 많이 나타난 문자
def most_frequent(s):
    freq = Counter(s)
    return freq.most_common(1)[0][0]
```

### 배열을 이용한 빈도수 (알파벳)

```c
// C - 소문자만 있는 경우
void countFrequency(const char* s, int freq[]) {
    // freq 배열은 크기 26으로 0으로 초기화되어 있어야 함
    while (*s) {
        freq[*s - 'a']++;
        s++;
    }
}

// 사용
int freq[26] = {0};
countFrequency("hello", freq);
// freq['h'-'a'] = 1, freq['e'-'a'] = 1, freq['l'-'a'] = 2, freq['o'-'a'] = 1
```

---

## 7. 연습 문제

### 문제 1: 배열 회전

배열을 오른쪽으로 k칸 회전하세요.

```
입력: [1, 2, 3, 4, 5], k = 2
출력: [4, 5, 1, 2, 3]
```

<details>
<summary>힌트</summary>

세 번의 뒤집기로 O(1) 공간으로 해결 가능:
1. 전체 뒤집기
2. 앞 k개 뒤집기
3. 나머지 뒤집기

</details>

<details>
<summary>정답 코드</summary>

```python
def rotate(arr, k):
    n = len(arr)
    k = k % n  # k가 n보다 클 수 있음

    def reverse(start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    reverse(0, n - 1)      # [5,4,3,2,1]
    reverse(0, k - 1)      # [4,5,3,2,1]
    reverse(k, n - 1)      # [4,5,1,2,3]

# 시간: O(n), 공간: O(1)
```

</details>

### 문제 2: 부분 배열 합이 k인 개수

합이 k인 연속 부분 배열의 개수를 구하세요.

```
입력: [1, 1, 1], k = 2
출력: 2 (부분배열 [1,1]이 2개)
```

<details>
<summary>힌트</summary>

프리픽스 합 + 해시맵 활용
prefix[j] - prefix[i] = k
→ prefix[i] = prefix[j] - k

</details>

<details>
<summary>정답 코드</summary>

```python
from collections import defaultdict

def subarray_sum(arr, k):
    count = 0
    prefix_sum = 0
    prefix_count = defaultdict(int)
    prefix_count[0] = 1  # 빈 프리픽스

    for num in arr:
        prefix_sum += num

        # prefix_sum - k가 이전에 나온 적 있으면
        # 그 지점부터 현재까지의 합이 k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]

        prefix_count[prefix_sum] += 1

    return count

# 시간: O(n), 공간: O(n)
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [Two Sum](https://leetcode.com/problems/two-sum/) | LeetCode | 해시맵 |
| ⭐ | [구간 합 구하기 4](https://www.acmicpc.net/problem/11659) | 백준 | 프리픽스 합 |
| ⭐⭐ | [3Sum](https://leetcode.com/problems/3sum/) | LeetCode | 2포인터 |
| ⭐⭐ | [Longest Substring Without Repeating](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | LeetCode | 슬라이딩 윈도우 |
| ⭐⭐ | [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) | LeetCode | 카데인 알고리즘 |
| ⭐⭐⭐ | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | LeetCode | 슬라이딩 윈도우 |

---

## 다음 단계

- [03_Stacks_and_Queues.md](./03_Stacks_and_Queues.md) - 스택/큐를 활용한 알고리즘

---

## 참고 자료

- [Two Pointers Technique](https://www.geeksforgeeks.org/two-pointers-technique/)
- [Sliding Window Problems](https://leetcode.com/tag/sliding-window/)
- [Prefix Sum Tutorial](https://usaco.guide/silver/prefix-sums)
