# 문자열 알고리즘 (String Algorithms)

## 개요

문자열 패턴 매칭 알고리즘은 텍스트에서 특정 패턴을 효율적으로 찾는 방법입니다. 브루트포스 O(nm)에서 KMP/Z-알고리즘 O(n+m)까지 다양한 기법을 다룹니다.

---

## 목차

1. [브루트포스 매칭](#1-브루트포스-매칭)
2. [KMP 알고리즘](#2-kmp-알고리즘)
3. [Rabin-Karp 알고리즘](#3-rabin-karp-알고리즘)
4. [Z-알고리즘](#4-z-알고리즘)
5. [문자열 해시](#5-문자열-해시)
6. [활용 문제](#6-활용-문제)
7. [연습 문제](#7-연습-문제)

---

## 1. 브루트포스 매칭

### 1.1 기본 아이디어

```
텍스트:  A B C D A B C E A B C D
패턴:    A B C D

위치 0: ABCD = ABCD ✓ (찾음!)
위치 1: BCDA ≠ ABCD ✗
위치 2: CDAB ≠ ABCD ✗
...
위치 8: ABCD = ABCD ✓ (찾음!)
```

### 1.2 구현

```python
def brute_force(text, pattern):
    """
    브루트포스 패턴 매칭
    시간: O(n * m), 공간: O(1)
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

# 예시
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

## 2. KMP 알고리즘

### 2.1 핵심 아이디어

```
KMP (Knuth-Morris-Pratt):
- 불일치 발생 시, 이미 매칭된 부분의 정보를 활용
- 실패 함수(failure function)로 건너뛸 위치 결정
- 시간: O(n + m)

실패 함수 (π 배열):
- π[i] = pattern[0..i]에서 접두사=접미사인 최대 길이

예시: pattern = "ABCABD"
인덱스:  0  1  2  3  4  5
문자:    A  B  C  A  B  D
π[i]:    0  0  0  1  2  0

π[4] = 2 → "ABCAB"에서 "AB"가 접두사이자 접미사
```

### 2.2 실패 함수 구성

```
pattern = "ABAAB"

i=0: "A"      → π[0] = 0 (정의)
i=1: "AB"     → 접두사=접미사 없음 → π[1] = 0
i=2: "ABA"    → "A" 매칭 → π[2] = 1
i=3: "ABAA"   → "A" 매칭 → π[3] = 1
i=4: "ABAAB"  → "AB" 매칭 → π[4] = 2

π = [0, 0, 1, 1, 2]
```

```python
def compute_failure(pattern):
    """실패 함수 계산 - O(m)"""
    m = len(pattern)
    pi = [0] * m  # π[0] = 0
    j = 0  # 현재 매칭된 길이

    for i in range(1, m):
        # 불일치 시 j를 pi[j-1]로 이동
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]

        # 일치하면 j 증가
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j

    return pi

# 예시
print(compute_failure("ABAAB"))  # [0, 0, 1, 1, 2]
print(compute_failure("ABCABD"))  # [0, 0, 0, 1, 2, 0]
```

### 2.3 KMP 매칭

```python
def kmp_search(text, pattern):
    """
    KMP 패턴 매칭
    시간: O(n + m), 공간: O(m)
    """
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    pi = compute_failure(pattern)
    result = []
    j = 0  # pattern에서의 현재 위치

    for i in range(n):
        # 불일치 시 j를 실패 함수로 이동
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        # 일치하면 j 증가
        if text[i] == pattern[j]:
            if j == m - 1:
                # 완전 매칭!
                result.append(i - m + 1)
                j = pi[j]  # 다음 매칭을 위해 이동
            else:
                j += 1

    return result


# 예시
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
print(kmp_search(text, pattern))  # [10]

text = "AAAAAA"
pattern = "AA"
print(kmp_search(text, pattern))  # [0, 1, 2, 3, 4]
```

### 2.4 C++ 구현

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

### 2.5 KMP 시각화

```
텍스트:  A B A B D A B A C D A B A B C A B A B
패턴:    A B A B C A B A B
π:       0 0 1 2 0 1 2 3 4

매칭 과정:
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
i=18: B=B ✓ j=9 → 완전 매칭! (위치 10)
```

---

## 3. Rabin-Karp 알고리즘

### 3.1 핵심 아이디어

```
Rabin-Karp:
- 해시 함수로 문자열 비교
- 롤링 해시로 O(1)에 다음 해시 계산
- 평균 O(n + m), 최악 O(nm) (해시 충돌 시)

롤링 해시:
hash("ABC") = A*d² + B*d + C
hash("BCD") = (hash("ABC") - A*d²) * d + D

d = 기수 (보통 31 또는 256)
```

### 3.2 구현

```python
def rabin_karp(text, pattern, d=256, q=101):
    """
    Rabin-Karp 패턴 매칭
    d: 기수 (문자 종류 수)
    q: 모듈러 (큰 소수)
    시간: 평균 O(n + m), 최악 O(nm)
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    result = []
    h = pow(d, m - 1, q)  # d^(m-1) mod q

    # 초기 해시값 계산
    p_hash = 0  # 패턴 해시
    t_hash = 0  # 텍스트 윈도우 해시

    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % q
        t_hash = (d * t_hash + ord(text[i])) % q

    # 슬라이딩 윈도우
    for i in range(n - m + 1):
        # 해시가 같으면 실제 비교
        if p_hash == t_hash:
            if text[i:i + m] == pattern:
                result.append(i)

        # 다음 윈도우 해시 계산 (롤링)
        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if t_hash < 0:
                t_hash += q

    return result


# 예시
text = "ABABDABACDABABCABAB"
pattern = "ABAB"
print(rabin_karp(text, pattern))  # [0, 10, 15]
```

### 3.3 다중 패턴 검색

```python
def rabin_karp_multiple(text, patterns, d=256, q=101):
    """여러 패턴 동시 검색"""
    n = len(text)
    result = {p: [] for p in patterns}

    # 패턴을 길이별로 그룹화
    by_length = {}
    for p in patterns:
        m = len(p)
        if m not in by_length:
            by_length[m] = {}
        # 패턴 해시 계산
        p_hash = 0
        for c in p:
            p_hash = (d * p_hash + ord(c)) % q
        if p_hash not in by_length[m]:
            by_length[m][p_hash] = []
        by_length[m][p_hash].append(p)

    # 각 길이에 대해 검색
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

## 4. Z-알고리즘

### 4.1 핵심 아이디어

```
Z 배열:
- Z[i] = s[i:]와 s의 최장 공통 접두사 길이

예시: s = "aabxaab"
인덱스:  0  1  2  3  4  5  6
문자:    a  a  b  x  a  a  b
Z[i]:    -  1  0  0  3  1  0

Z[1] = 1: "abxaab"와 "aabxaab"의 공통 접두사 = "a" (길이 1)
Z[4] = 3: "aab"와 "aabxaab"의 공통 접두사 = "aab" (길이 3)

패턴 매칭:
- s = pattern + "$" + text 구성
- Z[i] == len(pattern)이면 매칭
```

### 4.2 Z 배열 계산

```python
def z_function(s):
    """
    Z 배열 계산
    시간: O(n)
    """
    n = len(s)
    z = [0] * n
    z[0] = n  # 정의상 전체 문자열

    l, r = 0, 0  # Z-box의 왼쪽, 오른쪽 경계

    for i in range(1, n):
        if i < r:
            # Z-box 내부: 이전 정보 활용
            z[i] = min(r - i, z[i - l])

        # 확장 시도
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        # Z-box 업데이트
        if i + z[i] > r:
            l, r = i, i + z[i]

    return z


# 예시
print(z_function("aabxaab"))  # [7, 1, 0, 0, 3, 1, 0]
print(z_function("aaaaa"))    # [5, 4, 3, 2, 1]
```

### 4.3 Z-알고리즘 패턴 매칭

```python
def z_search(text, pattern):
    """
    Z-알고리즘을 이용한 패턴 매칭
    시간: O(n + m)
    """
    concat = pattern + "$" + text
    z = z_function(concat)
    m = len(pattern)

    result = []
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            result.append(i - m - 1)

    return result


# 예시
text = "ABABDABACDABABCABAB"
pattern = "ABAB"
print(z_search(text, pattern))  # [0, 10, 15]
```

### 4.4 C++ 구현

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

## 5. 문자열 해시

### 5.1 다항식 해시

```python
def polynomial_hash(s, base=31, mod=10**9 + 9):
    """
    다항식 롤링 해시
    hash(s) = s[0]*base^(n-1) + s[1]*base^(n-2) + ... + s[n-1]
    """
    h = 0
    for c in s:
        h = (h * base + ord(c) - ord('a') + 1) % mod
    return h
```

### 5.2 프리픽스 해시 (구간 해시)

```python
class StringHash:
    """
    문자열 해시 (구간 해시 쿼리 O(1))
    """
    def __init__(self, s, base=31, mod=10**9 + 9):
        self.base = base
        self.mod = mod
        self.n = len(s)

        # 프리픽스 해시
        self.prefix = [0] * (self.n + 1)
        # base의 거듭제곱
        self.power = [1] * (self.n + 1)

        for i in range(self.n):
            self.prefix[i + 1] = (self.prefix[i] * base + ord(s[i]) - ord('a') + 1) % mod
            self.power[i + 1] = (self.power[i] * base) % mod

    def get_hash(self, l, r):
        """s[l:r+1]의 해시값 (0-indexed)"""
        h = (self.prefix[r + 1] - self.prefix[l] * self.power[r - l + 1]) % self.mod
        return (h + self.mod) % self.mod


# 사용 예시
s = "abcabc"
sh = StringHash(s)

print(sh.get_hash(0, 2))  # "abc"의 해시
print(sh.get_hash(3, 5))  # "abc"의 해시 (같아야 함)
print(sh.get_hash(0, 2) == sh.get_hash(3, 5))  # True
```

### 5.3 더블 해시 (충돌 방지)

```python
class DoubleHash:
    """두 개의 해시로 충돌 확률 최소화"""
    def __init__(self, s):
        self.h1 = StringHash(s, base=31, mod=10**9 + 7)
        self.h2 = StringHash(s, base=37, mod=10**9 + 9)

    def get_hash(self, l, r):
        return (self.h1.get_hash(l, r), self.h2.get_hash(l, r))
```

---

## 6. 활용 문제

### 6.1 가장 긴 팰린드롬 부분 문자열

```python
def longest_palindrome_substring(s):
    """
    중심 확장법
    시간: O(n²)
    """
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]

    result = ""
    for i in range(len(s)):
        # 홀수 길이 팰린드롬
        odd = expand(i, i)
        if len(odd) > len(result):
            result = odd

        # 짝수 길이 팰린드롬
        even = expand(i, i + 1)
        if len(even) > len(result):
            result = even

    return result


# 예시
print(longest_palindrome_substring("babad"))  # "bab" 또는 "aba"
```

### 6.2 반복되는 부분 문자열 (KMP 응용)

```python
def repeated_substring_pattern(s):
    """
    문자열이 반복 패턴으로 구성되어 있는지 확인
    예: "abab" → True (ab가 2번 반복)
    """
    n = len(s)
    pi = compute_failure(s)

    # 마지막 실패 함수 값 확인
    length = pi[n - 1]

    # 반복 단위 길이
    pattern_length = n - length

    # n이 pattern_length의 배수이고, 실제로 반복인지 확인
    return length > 0 and n % pattern_length == 0


# 예시
print(repeated_substring_pattern("abab"))   # True
print(repeated_substring_pattern("abcab"))  # False
```

### 6.3 최장 공통 부분 문자열 (해시 + 이분탐색)

```python
def longest_common_substring(s1, s2):
    """
    이분탐색 + 롤링 해시
    시간: O((n+m) log(min(n,m)))
    """
    def get_hashes(s, length, base=31, mod=10**9 + 9):
        """길이 length인 모든 부분 문자열 해시"""
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
        """길이 length의 공통 부분 문자열 존재 여부"""
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


# 예시
print(longest_common_substring("abcdxyz", "xyzabcd"))  # 4 ("abcd" 또는 "xyz")
```

### 6.4 아나그램 찾기

```python
def find_anagrams(s, p):
    """
    s에서 p의 아나그램인 부분 문자열 시작 인덱스
    슬라이딩 윈도우 + 해시맵
    """
    from collections import Counter

    result = []
    p_count = Counter(p)
    s_count = Counter()
    m = len(p)

    for i, c in enumerate(s):
        s_count[c] += 1

        # 윈도우 크기 유지
        if i >= m:
            left = s[i - m]
            s_count[left] -= 1
            if s_count[left] == 0:
                del s_count[left]

        if s_count == p_count:
            result.append(i - m + 1)

    return result


# 예시
print(find_anagrams("cbaebabacd", "abc"))  # [0, 6]
```

### 6.5 문자열 압축

```python
def compress_string(s):
    """
    연속된 같은 문자를 압축
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

## 7. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 알고리즘 |
|--------|------|--------|----------|
| ⭐⭐ | [찾기](https://www.acmicpc.net/problem/1786) | 백준 | KMP |
| ⭐⭐ | [Implement strStr()](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/) | LeetCode | KMP |
| ⭐⭐ | [Repeated String Match](https://leetcode.com/problems/repeated-string-match/) | LeetCode | KMP/Rabin-Karp |
| ⭐⭐⭐ | [부분 문자열](https://www.acmicpc.net/problem/16916) | 백준 | KMP |
| ⭐⭐⭐ | [Longest Happy Prefix](https://leetcode.com/problems/longest-happy-prefix/) | LeetCode | KMP 실패함수 |
| ⭐⭐⭐ | [Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/) | LeetCode | KMP |
| ⭐⭐⭐⭐ | [광고](https://www.acmicpc.net/problem/1305) | 백준 | KMP |

---

## 알고리즘 비교

```
┌──────────────┬─────────────┬─────────────┬────────────────┐
│ 알고리즘      │ 시간        │ 공간        │ 특징            │
├──────────────┼─────────────┼─────────────┼────────────────┤
│ 브루트포스    │ O(nm)       │ O(1)        │ 간단, 짧은 문자열│
│ KMP          │ O(n+m)      │ O(m)        │ 정확, 범용적     │
│ Rabin-Karp   │ O(n+m) 평균  │ O(1)        │ 다중 패턴 효율적 │
│ Z-알고리즘    │ O(n+m)      │ O(n+m)      │ 구현 간단       │
└──────────────┴─────────────┴─────────────┴────────────────┘

n = 텍스트 길이, m = 패턴 길이
```

---

## 다음 단계

- [23_Segment_Tree.md](./23_Segment_Tree.md) - 세그먼트 트리

---

## 참고 자료

- [String Matching Visualization](https://www.cs.usfca.edu/~galles/visualization/StringMatch.html)
- Introduction to Algorithms (CLRS) - Chapter 32
