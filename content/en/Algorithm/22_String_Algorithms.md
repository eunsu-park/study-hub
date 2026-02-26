# String Algorithms

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement brute-force string pattern matching and analyze its O(n × m) time complexity
2. Build the KMP failure function and use it to perform O(n + m) pattern matching without redundant comparisons
3. Implement the Rabin-Karp rolling hash algorithm and explain how it achieves average-case O(n + m) matching
4. Construct the Z-array and use the Z-algorithm to find all pattern occurrences in linear time
5. Design polynomial string hashing to compare substrings in O(1) after O(n) preprocessing
6. Apply string algorithms to solve problems such as period detection, anagram search, and multiple pattern matching

---

## Overview

String pattern matching algorithms efficiently find specific patterns in text. Covers various techniques from brute force O(nm) to KMP/Z-algorithm O(n+m).

---

## Table of Contents

1. [Brute Force Matching](#1-brute-force-matching)
2. [KMP Algorithm](#2-kmp-algorithm)
3. [Rabin-Karp Algorithm](#3-rabin-karp-algorithm)
4. [Z-Algorithm](#4-z-algorithm)
5. [String Hashing](#5-string-hashing)
6. [Application Problems](#6-application-problems)
7. [Practice Problems](#7-practice-problems)

---

## 1. Brute Force Matching

### 1.1 Basic Idea

```
Text:    A B C D A B C E A B C D
Pattern: A B C D

Position 0: ABCD = ABCD ✓ (found!)
Position 1: BCDA ≠ ABCD ✗
Position 2: CDAB ≠ ABCD ✗
...
Position 8: ABCD = ABCD ✓ (found!)
```

### 1.2 Implementation

```python
def brute_force(text, pattern):
    """
    Brute force pattern matching
    Time: O(n * m), Space: O(1)
    n = len(text), m = len(pattern)
    """
    n, m = len(text), len(pattern)
    result = []

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            result.append(i)

    return result

# Example
text = "ABCDABCEABCD"
pattern = "ABCD"
print(brute_force(text, pattern))  # [0, 8]
```

```cpp
// C++
vector<int> bruteForce(const string& text, const string& pattern) {
    int n = text.length(), m = pattern.length();
    vector<int> result;

    for (int i = 0; i <= n - m; i++) {
        bool match = true;
        for (int j = 0; j < m; j++) {
            if (text[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) result.push_back(i);
    }

    return result;
}
```

---

## 2. KMP Algorithm

### 2.1 Core Idea

```
KMP (Knuth-Morris-Pratt):
- On mismatch, use information from already matched portion
- Failure function determines skip position
- Time: O(n + m)

Failure function (π array):
- π[i] = longest proper prefix of pattern[0..i] that is also suffix

Example: pattern = "ABCABD"
Index:   0  1  2  3  4  5
Char:    A  B  C  A  B  D
π[i]:    0  0  0  1  2  0

π[4] = 2 → In "ABCAB", "AB" is both prefix and suffix
```

### 2.2 Building Failure Function

```
pattern = "ABAAB"

i=0: "A"      → π[0] = 0 (by definition)
i=1: "AB"     → no prefix=suffix → π[1] = 0
i=2: "ABA"    → "A" matches → π[2] = 1
i=3: "ABAA"   → "A" matches → π[3] = 1
i=4: "ABAAB"  → "AB" matches → π[4] = 2

π = [0, 0, 1, 1, 2]
```

```python
def compute_failure(pattern):
    """Compute failure function - O(m)"""
    m = len(pattern)
    pi = [0] * m  # π[0] = 0 by definition — no proper prefix can equal the whole string
    j = 0  # j tracks the length of the longest matching prefix-suffix seen so far

    for i in range(1, m):
        # On mismatch, fall back to the next candidate prefix length.
        # π[j-1] tells us: "if pattern[0..j-1] matched but pattern[j] didn't,
        # the longest still-valid prefix is π[j-1] characters."
        # This reuse of already-computed values keeps the total work O(m).
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]

        # If match, extend the current prefix-suffix by one character
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j  # Record that pattern[0..i] has a prefix-suffix of length j

    return pi

# Example
print(compute_failure("ABAAB"))  # [0, 0, 1, 1, 2]
print(compute_failure("ABCABD"))  # [0, 0, 0, 1, 2, 0]
```

### 2.3 KMP Matching

```python
def kmp_search(text, pattern):
    """
    KMP pattern matching
    Time: O(n + m), Space: O(m)
    KMP never re-examines a text character — once text[i] is processed, i only moves forward.
    The failure function handles backward movement in the pattern, not the text.
    """
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    pi = compute_failure(pattern)
    result = []
    j = 0  # j = number of pattern characters matched so far

    for i in range(n):
        # On mismatch, fall back in the pattern using failure function.
        # This is O(n) amortized: j can only increase n times across the whole loop,
        # so the total number of fallback steps cannot exceed n either.
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            if j == m - 1:
                # Complete match — record start position
                result.append(i - m + 1)
                # Use failure function to set up for overlapping matches:
                # pi[j] tells us the longest prefix of pattern that is also a suffix
                # of the match just found, so we don't restart from scratch
                j = pi[j]
            else:
                j += 1

    return result


# Example
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
print(kmp_search(text, pattern))  # [10]

text = "AAAAAA"
pattern = "AA"
print(kmp_search(text, pattern))  # [0, 1, 2, 3, 4]
```

### 2.4 C++ Implementation

```cpp
#include <vector>
#include <string>
using namespace std;

vector<int> computeFailure(const string& pattern) {
    int m = pattern.length();
    vector<int> pi(m, 0);
    int j = 0;

    for (int i = 1; i < m; i++) {
        while (j > 0 && pattern[i] != pattern[j]) {
            j = pi[j - 1];
        }
        if (pattern[i] == pattern[j]) {
            pi[i] = ++j;
        }
    }

    return pi;
}

vector<int> kmpSearch(const string& text, const string& pattern) {
    int n = text.length(), m = pattern.length();
    vector<int> pi = computeFailure(pattern);
    vector<int> result;
    int j = 0;

    for (int i = 0; i < n; i++) {
        while (j > 0 && text[i] != pattern[j]) {
            j = pi[j - 1];
        }
        if (text[i] == pattern[j]) {
            if (j == m - 1) {
                result.push_back(i - m + 1);
                j = pi[j];
            } else {
                j++;
            }
        }
    }

    return result;
}
```

### 2.5 KMP Visualization

```
Text:    A B A B D A B A C D A B A B C A B A B
Pattern: A B A B C A B A B
π:       0 0 1 2 0 1 2 3 4

Matching process:
i=0: A=A ✓ j=1
i=1: B=B ✓ j=2
i=2: A=A ✓ j=3
i=3: B=B ✓ j=4
i=4: D≠C ✗ j=π[3]=2, D≠A ✗ j=π[1]=0, D≠A ✗ j=0
i=5: A=A ✓ j=1
...
i=10: A=A ✓ j=1
i=11: B=B ✓ j=2
...
i=18: B=B ✓ j=9 → Complete match! (position 10)
```

---

## 3. Rabin-Karp Algorithm

### 3.1 Core Idea

```
Rabin-Karp:
- Compare strings using hash function
- Rolling hash for O(1) next hash computation
- Average O(n + m), Worst O(nm) (hash collision)

Rolling hash:
hash("ABC") = A*d² + B*d + C
hash("BCD") = (hash("ABC") - A*d²) * d + D

d = base (usually 31 or 256)
```

### 3.2 Implementation

```python
def rabin_karp(text, pattern, d=256, q=101):
    """
    Rabin-Karp pattern matching
    d: base (number of characters)
    q: modulus (large prime)
    Time: Average O(n + m), Worst O(nm)
    The rolling hash makes each window shift O(1) instead of O(m):
    we subtract the outgoing character's contribution and add the incoming one.
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    result = []
    # h = d^(m-1) mod q — the positional weight of the leftmost character in the window.
    # We need this to subtract the outgoing character's contribution during rolling.
    h = pow(d, m - 1, q)

    # Compute initial hash values using polynomial rolling hash
    p_hash = 0  # Pattern hash
    t_hash = 0  # Text window hash

    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % q
        t_hash = (d * t_hash + ord(text[i])) % q

    # Sliding window — O(n) total across all iterations due to O(1) rolling hash
    for i in range(n - m + 1):
        # Hash comparison is O(1); character comparison is O(m) but only done on
        # hash matches, which are rare on random input — giving O(n+m) average case
        if p_hash == t_hash:
            if text[i:i + m] == pattern:  # Verify to guard against hash collisions
                result.append(i)

        # Rolling hash: remove leftmost character, add rightmost new character.
        # Subtracting ord(text[i]) * h removes the d^(m-1) contribution of the old leading char.
        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if t_hash < 0:
                t_hash += q  # Python % can return negative for negative inputs — normalize

    return result


# Example
text = "ABABDABACDABABCABAB"
pattern = "ABAB"
print(rabin_karp(text, pattern))  # [0, 10, 15]
```

### 3.3 Multiple Pattern Search

```python
def rabin_karp_multiple(text, patterns, d=256, q=101):
    """Search multiple patterns simultaneously"""
    n = len(text)
    result = {p: [] for p in patterns}

    # Group patterns by length
    by_length = {}
    for p in patterns:
        m = len(p)
        if m not in by_length:
            by_length[m] = {}
        # Compute pattern hash
        p_hash = 0
        for c in p:
            p_hash = (d * p_hash + ord(c)) % q
        if p_hash not in by_length[m]:
            by_length[m][p_hash] = []
        by_length[m][p_hash].append(p)

    # Search for each length
    for m, hash_to_patterns in by_length.items():
        if m > n:
            continue

        h = pow(d, m - 1, q)
        t_hash = 0

        for i in range(m):
            t_hash = (d * t_hash + ord(text[i])) % q

        for i in range(n - m + 1):
            if t_hash in hash_to_patterns:
                for p in hash_to_patterns[t_hash]:
                    if text[i:i + m] == p:
                        result[p].append(i)

            if i < n - m:
                t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
                if t_hash < 0:
                    t_hash += q

    return result
```

---

## 4. Z-Algorithm

### 4.1 Core Idea

```
Z array:
- Z[i] = length of longest common prefix of s[i:] and s

Example: s = "aabxaab"
Index:   0  1  2  3  4  5  6
Char:    a  a  b  x  a  a  b
Z[i]:    -  1  0  0  3  1  0

Z[1] = 1: Common prefix of "abxaab" and "aabxaab" = "a" (length 1)
Z[4] = 3: Common prefix of "aab" and "aabxaab" = "aab" (length 3)

Pattern matching:
- Construct s = pattern + "$" + text
- Z[i] == len(pattern) means match
```

### 4.2 Computing Z Array

```python
def z_function(s):
    """
    Compute Z array
    Time: O(n) — the Z-box avoids re-examining characters inside already-matched regions
    """
    n = len(s)
    z = [0] * n
    z[0] = n  # By convention, the full string is the common prefix with itself

    l, r = 0, 0  # Z-box: the rightmost matching window found so far, [l, r)

    for i in range(1, n):
        if i < r:
            # i is inside a known Z-box. We already know s[i..r-1] matches s[i-l..r-l-1].
            # So z[i] is at least min(r - i, z[i - l]), saving us from re-examining
            # those characters — this is what achieves O(n) overall.
            z[i] = min(r - i, z[i - l])

        # Attempt to extend the match beyond what we already know
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        # Extend the Z-box to the rightmost position seen so far
        if i + z[i] > r:
            l, r = i, i + z[i]

    return z


# Example
print(z_function("aabxaab"))  # [7, 1, 0, 0, 3, 1, 0]
print(z_function("aaaaa"))    # [5, 4, 3, 2, 1]
```

### 4.3 Z-Algorithm Pattern Matching

```python
def z_search(text, pattern):
    """
    Pattern matching using Z-algorithm
    Time: O(n + m)
    """
    concat = pattern + "$" + text
    z = z_function(concat)
    m = len(pattern)

    result = []
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            result.append(i - m - 1)

    return result


# Example
text = "ABABDABACDABABCABAB"
pattern = "ABAB"
print(z_search(text, pattern))  # [0, 10, 15]
```

### 4.4 C++ Implementation

```cpp
#include <vector>
#include <string>
using namespace std;

vector<int> zFunction(const string& s) {
    int n = s.length();
    vector<int> z(n, 0);
    z[0] = n;
    int l = 0, r = 0;

    for (int i = 1; i < n; i++) {
        if (i < r) {
            z[i] = min(r - i, z[i - l]);
        }
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if (i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }

    return z;
}

vector<int> zSearch(const string& text, const string& pattern) {
    string concat = pattern + "$" + text;
    vector<int> z = zFunction(concat);
    int m = pattern.length();
    vector<int> result;

    for (int i = m + 1; i < concat.length(); i++) {
        if (z[i] == m) {
            result.push_back(i - m - 1);
        }
    }

    return result;
}
```

---

## 5. String Hashing

### 5.1 Polynomial Hash

```python
def polynomial_hash(s, base=31, mod=10**9 + 9):
    """
    Polynomial rolling hash
    hash(s) = s[0]*base^(n-1) + s[1]*base^(n-2) + ... + s[n-1]
    """
    h = 0
    for c in s:
        h = (h * base + ord(c) - ord('a') + 1) % mod
    return h
```

### 5.2 Prefix Hash (Range Hash)

```python
class StringHash:
    """
    String hash (O(1) range hash queries)
    Uses the same prefix-sum idea as prefix sums for integers, but for polynomial hashes.
    After O(n) preprocessing, any substring hash is computable in O(1).
    """
    def __init__(self, s, base=31, mod=10**9 + 9):
        self.base = base
        self.mod = mod
        self.n = len(s)

        # prefix[i] = hash of s[0:i] — analogous to a prefix sum array
        self.prefix = [0] * (self.n + 1)
        # power[i] = base^i mod mod — precomputed to avoid repeated exponentiation
        self.power = [1] * (self.n + 1)

        for i in range(self.n):
            # Treat each character as a digit in base-31: s[0]*31^(n-1) + s[1]*31^(n-2) + ...
            # Mapping to 1..26 instead of 0..25 avoids the ambiguity where leading 'a's are invisible
            self.prefix[i + 1] = (self.prefix[i] * base + ord(s[i]) - ord('a') + 1) % mod
            self.power[i + 1] = (self.power[i] * base) % mod

    def get_hash(self, l, r):
        """Hash of s[l:r+1] (0-indexed)
        Subtracting prefix[l] scaled by power[r-l+1] removes the contribution of
        characters before position l — same logic as range_sum = prefix[r+1] - prefix[l].
        """
        h = (self.prefix[r + 1] - self.prefix[l] * self.power[r - l + 1]) % self.mod
        return (h + self.mod) % self.mod  # Add mod to handle Python's negative modulo


# Usage example
s = "abcabc"
sh = StringHash(s)

print(sh.get_hash(0, 2))  # Hash of "abc"
print(sh.get_hash(3, 5))  # Hash of "abc" (should be same)
print(sh.get_hash(0, 2) == sh.get_hash(3, 5))  # True
```

### 5.3 Double Hash (Collision Prevention)

```python
class DoubleHash:
    """Two hashes to minimize collision probability"""
    def __init__(self, s):
        self.h1 = StringHash(s, base=31, mod=10**9 + 7)
        self.h2 = StringHash(s, base=37, mod=10**9 + 9)

    def get_hash(self, l, r):
        return (self.h1.get_hash(l, r), self.h2.get_hash(l, r))
```

---

## 6. Application Problems

### 6.1 Longest Palindromic Substring

```python
def longest_palindrome_substring(s):
    """
    Expand around center method
    Time: O(n²)
    """
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]

    result = ""
    for i in range(len(s)):
        # Odd length palindrome
        odd = expand(i, i)
        if len(odd) > len(result):
            result = odd

        # Even length palindrome
        even = expand(i, i + 1)
        if len(even) > len(result):
            result = even

    return result


# Example
print(longest_palindrome_substring("babad"))  # "bab" or "aba"
```

### 6.2 Repeated Substring Pattern (KMP Application)

```python
def repeated_substring_pattern(s):
    """
    Check if string is made of repeated pattern
    Example: "abab" → True (ab repeated 2 times)
    """
    n = len(s)
    pi = compute_failure(s)

    # Check last failure function value
    length = pi[n - 1]

    # Pattern unit length
    pattern_length = n - length

    # Check if n is multiple of pattern_length and actually repeats
    return length > 0 and n % pattern_length == 0


# Example
print(repeated_substring_pattern("abab"))   # True
print(repeated_substring_pattern("abcab"))  # False
```

### 6.3 Longest Common Substring (Hash + Binary Search)

```python
def longest_common_substring(s1, s2):
    """
    Binary search + rolling hash
    Time: O((n+m) log(min(n,m)))
    """
    def get_hashes(s, length, base=31, mod=10**9 + 9):
        """Hashes of all substrings of given length"""
        if length > len(s):
            return set()

        hashes = set()
        h = 0
        power = pow(base, length - 1, mod)

        for i in range(length):
            h = (h * base + ord(s[i])) % mod

        hashes.add(h)

        for i in range(length, len(s)):
            h = (h - ord(s[i - length]) * power) % mod
            h = (h * base + ord(s[i])) % mod
            hashes.add(h)

        return hashes

    def check(length):
        """Check if common substring of given length exists"""
        h1 = get_hashes(s1, length)
        h2 = get_hashes(s2, length)
        return len(h1 & h2) > 0

    left, right = 0, min(len(s1), len(s2))
    result = 0

    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result


# Example
print(longest_common_substring("abcdxyz", "xyzabcd"))  # 4 ("abcd" or "xyz")
```

### 6.4 Find Anagrams

```python
def find_anagrams(s, p):
    """
    Find start indices of p's anagrams in s
    Sliding window + hashmap
    """
    from collections import Counter

    result = []
    p_count = Counter(p)
    s_count = Counter()
    m = len(p)

    for i, c in enumerate(s):
        s_count[c] += 1

        # Maintain window size
        if i >= m:
            left = s[i - m]
            s_count[left] -= 1
            if s_count[left] == 0:
                del s_count[left]

        if s_count == p_count:
            result.append(i - m + 1)

    return result


# Example
print(find_anagrams("cbaebabacd", "abc"))  # [0, 6]
```

### 6.5 String Compression

```python
def compress_string(s):
    """
    Compress consecutive same characters
    "aabcccccaaa" → "a2b1c5a3"
    """
    if not s:
        return s

    result = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + str(count))
            count = 1

    result.append(s[-1] + str(count))

    compressed = "".join(result)
    return compressed if len(compressed) < len(s) else s
```

---

## 7. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Algorithm |
|--------|------|--------|----------|
| ⭐⭐ | [Find](https://www.acmicpc.net/problem/1786) | BOJ | KMP |
| ⭐⭐ | [Implement strStr()](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/) | LeetCode | KMP |
| ⭐⭐ | [Repeated String Match](https://leetcode.com/problems/repeated-string-match/) | LeetCode | KMP/Rabin-Karp |
| ⭐⭐⭐ | [Substring](https://www.acmicpc.net/problem/16916) | BOJ | KMP |
| ⭐⭐⭐ | [Longest Happy Prefix](https://leetcode.com/problems/longest-happy-prefix/) | LeetCode | KMP failure |
| ⭐⭐⭐ | [Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/) | LeetCode | KMP |
| ⭐⭐⭐⭐ | [Advertisement](https://www.acmicpc.net/problem/1305) | BOJ | KMP |

---

## Algorithm Comparison

```
┌──────────────┬─────────────┬─────────────┬────────────────┐
│ Algorithm    │ Time        │ Space       │ Characteristics│
├──────────────┼─────────────┼─────────────┼────────────────┤
│ Brute Force  │ O(nm)       │ O(1)        │ Simple, short  │
│ KMP          │ O(n+m)      │ O(m)        │ Exact, general │
│ Rabin-Karp   │ O(n+m) avg  │ O(1)        │ Multi-pattern  │
│ Z-Algorithm  │ O(n+m)      │ O(n+m)      │ Simple impl    │
└──────────────┴─────────────┴─────────────┴────────────────┘

n = text length, m = pattern length
```

---

## Next Steps

- [23_Segment_Tree.md](./23_Segment_Tree.md) - Segment Tree

---

## References

- [String Matching Visualization](https://www.cs.usfca.edu/~galles/visualization/StringMatch.html)
- Introduction to Algorithms (CLRS) - Chapter 32
