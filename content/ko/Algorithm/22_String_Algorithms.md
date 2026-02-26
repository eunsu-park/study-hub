# 문자열 알고리즘 (String Algorithms)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 브루트포스(brute-force) 문자열 패턴 매칭을 구현하고 O(n × m) 시간 복잡도를 분석할 수 있다
2. KMP 실패 함수(failure function)를 구성하고 이를 이용해 중복 비교 없이 O(n + m) 패턴 매칭을 수행할 수 있다
3. 라빈-카프(Rabin-Karp) 롤링 해시(rolling hash) 알고리즘을 구현하고 평균 O(n + m) 매칭을 달성하는 원리를 설명할 수 있다
4. Z-배열(Z-array)을 구성하고 Z-알고리즘을 이용하여 선형 시간에 모든 패턴 출현 위치를 찾을 수 있다
5. 다항식 문자열 해시(polynomial string hashing)를 설계하여 O(n) 전처리 후 O(1)에 부분 문자열을 비교할 수 있다
6. 주기 감지(period detection), 애너그램 검색(anagram search), 다중 패턴 매칭(multiple pattern matching) 문제에 문자열 알고리즘을 적용할 수 있다

---

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
    pi = [0] * m  # π[0] = 0 (정의상 — 진접두사(proper prefix)는 전체 문자열과 같을 수 없음)
    j = 0  # j는 지금까지 관찰된 가장 긴 매칭 접두사-접미사(prefix-suffix)의 길이를 추적

    for i in range(1, m):
        # 불일치 시 다음 후보 접두사 길이로 폴백한다.
        # π[j-1]의 의미: "pattern[0..j-1]이 매칭되었으나 pattern[j]가 불일치하면,
        # 여전히 유효한 가장 긴 접두사는 π[j-1] 문자이다."
        # 이미 계산된 값을 재활용하므로 총 작업량이 O(m)으로 유지된다.
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]

        # 일치하면 현재 접두사-접미사를 한 문자 확장
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j  # pattern[0..i]에 길이 j인 접두사-접미사가 있음을 기록

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
    KMP는 텍스트 문자를 절대 다시 검사하지 않는다 — text[i]가 처리되면 i는 앞으로만 이동한다.
    실패 함수가 패턴 내에서의 후퇴를 처리하지, 텍스트에서의 후퇴는 없다.
    """
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    pi = compute_failure(pattern)
    result = []
    j = 0  # j = 지금까지 매칭된 패턴 문자 수

    for i in range(n):
        # 불일치 시 실패 함수를 사용해 패턴 내에서 폴백한다.
        # 이것이 O(n) 분할 상환인 이유: j는 전체 루프에서 최대 n번만 증가할 수 있으므로,
        # 폴백 횟수의 총합도 n을 초과할 수 없다.
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            if j == m - 1:
                # 완전 매칭 — 시작 위치를 기록
                result.append(i - m + 1)
                # 실패 함수를 사용하여 겹치는 매칭(overlapping matches)을 설정:
                # pi[j]는 방금 찾은 매칭의 접미사이면서 패턴의 접두사인
                # 가장 긴 부분의 길이를 알려주므로, 처음부터 다시 시작하지 않아도 된다
                j = pi[j]
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
    롤링 해시(rolling hash)는 각 윈도우 이동을 O(m) 대신 O(1)로 만든다:
    나가는 문자의 기여를 빼고 들어오는 문자를 더한다.
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    result = []
    # h = d^(m-1) mod q — 윈도우에서 가장 왼쪽 문자의 위치 가중치.
    # 롤링 시 나가는 문자의 기여를 빼기 위해 이 값이 필요하다.
    h = pow(d, m - 1, q)

    # 다항식 롤링 해시를 사용하여 초기 해시값 계산
    p_hash = 0  # 패턴 해시
    t_hash = 0  # 텍스트 윈도우 해시

    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % q
        t_hash = (d * t_hash + ord(text[i])) % q

    # 슬라이딩 윈도우 — O(1) 롤링 해시로 인해 전체 반복에서 O(n)
    for i in range(n - m + 1):
        # 해시 비교는 O(1); 문자 비교는 O(m)이지만 해시가 일치할 때만 수행되며,
        # 무작위 입력에서 해시 일치는 드물어 평균 O(n+m)을 달성
        if p_hash == t_hash:
            if text[i:i + m] == pattern:  # 해시 충돌을 방지하기 위해 실제 비교로 검증
                result.append(i)

        # 롤링 해시: 가장 왼쪽 문자를 제거하고 가장 오른쪽 새 문자를 추가.
        # ord(text[i]) * h를 빼면 이전 선두 문자의 d^(m-1) 기여가 제거된다.
        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if t_hash < 0:
                t_hash += q  # Python의 %는 음수 입력에서 음수를 반환할 수 있으므로 정규화

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
    시간: O(n) — Z-box가 이미 매칭된 영역 내의 문자를 재검사하지 않도록 한다
    """
    n = len(s)
    z = [0] * n
    z[0] = n  # 관례상 전체 문자열이 자기 자신과의 공통 접두사이다

    l, r = 0, 0  # Z-box: 지금까지 발견된 가장 오른쪽 매칭 윈도우 [l, r)

    for i in range(1, n):
        if i < r:
            # i가 이미 알려진 Z-box 내부에 있다. s[i..r-1]이 s[i-l..r-l-1]과
            # 매칭됨을 이미 알고 있다. 따라서 z[i]는 최소 min(r - i, z[i - l])이며,
            # 그 문자들을 재검사하지 않아도 된다 — 이것이 O(n)을 달성하는 핵심이다.
            z[i] = min(r - i, z[i - l])

        # 이미 알고 있는 범위를 넘어서 매칭 확장 시도
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        # Z-box를 지금까지 관찰된 가장 오른쪽 위치로 확장
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
    문자열 해시 (O(1) 구간 해시 쿼리)
    정수 배열의 누적 합(prefix sum)과 동일한 아이디어를 다항식 해시에 적용한 것이다.
    O(n) 전처리 후 임의의 부분 문자열 해시를 O(1)에 계산할 수 있다.
    """
    def __init__(self, s, base=31, mod=10**9 + 9):
        self.base = base
        self.mod = mod
        self.n = len(s)

        # prefix[i] = s[0:i]의 해시 — 누적 합 배열과 유사
        self.prefix = [0] * (self.n + 1)
        # power[i] = base^i mod mod — 반복 거듭제곱을 피하기 위해 미리 계산
        self.power = [1] * (self.n + 1)

        for i in range(self.n):
            # 각 문자를 31진법의 자릿수로 취급: s[0]*31^(n-1) + s[1]*31^(n-2) + ...
            # 0..25 대신 1..26으로 매핑하여 선두 'a'가 보이지 않는 모호성을 방지
            self.prefix[i + 1] = (self.prefix[i] * base + ord(s[i]) - ord('a') + 1) % mod
            self.power[i + 1] = (self.power[i] * base) % mod

    def get_hash(self, l, r):
        """s[l:r+1]의 해시값 (0-indexed)
        prefix[l]에 power[r-l+1]을 곱하여 빼면 위치 l 이전 문자들의 기여가 제거된다 —
        range_sum = prefix[r+1] - prefix[l]과 동일한 논리이다.
        """
        h = (self.prefix[r + 1] - self.prefix[l] * self.power[r - l + 1]) % self.mod
        return (h + self.mod) % self.mod  # Python의 음수 모듈러를 처리하기 위해 mod를 더함


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
