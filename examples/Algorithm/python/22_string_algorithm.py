"""
String Algorithms
String Matching and Processing

Implements string search and processing algorithms.
"""

from typing import List, Tuple


# =============================================================================
# 1. KMP Algorithm (Knuth-Morris-Pratt)
# =============================================================================

def kmp_failure(pattern: str) -> List[int]:
    """
    KMP failure function (partial match table) computation
    Time Complexity: O(m), m = pattern length
    """
    m = len(pattern)
    failure = [0] * m
    j = 0  # Length of previous longest prefix

    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]

        if pattern[i] == pattern[j]:
            j += 1
            failure[i] = j

    return failure


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    KMP string search
    Time Complexity: O(n + m)
    Returns: list of starting indices where pattern is found
    """
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    failure = kmp_failure(pattern)
    matches = []
    j = 0  # Pattern index

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]

        if text[i] == pattern[j]:
            if j == m - 1:
                matches.append(i - m + 1)
                j = failure[j]
            else:
                j += 1

    return matches


# =============================================================================
# 2. Rabin-Karp Algorithm
# =============================================================================

def rabin_karp_search(text: str, pattern: str, mod: int = 10**9 + 7) -> List[int]:
    """
    Rabin-Karp string search (rolling hash)
    Time Complexity: average O(n + m), worst O(nm)
    """
    if not pattern or len(pattern) > len(text):
        return []

    n, m = len(text), len(pattern)
    base = 256
    matches = []

    # Compute pattern hash
    pattern_hash = 0
    text_hash = 0
    h = pow(base, m - 1, mod)

    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # Verify to handle hash collision
            if text[i:i + m] == pattern:
                matches.append(i)

        # Compute next window hash
        if i < n - m:
            text_hash = ((text_hash - ord(text[i]) * h) * base + ord(text[i + m])) % mod

    return matches


# =============================================================================
# 3. Z Algorithm
# =============================================================================

def z_function(s: str) -> List[int]:
    """
    Z function computation
    z[i] = length of longest common prefix of s and s[i:]
    Time Complexity: O(n)
    """
    n = len(s)
    z = [0] * n
    z[0] = n

    l, r = 0, 0  # Z-box [l, r)

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] > r:
            l, r = i, i + z[i]

    return z


def z_search(text: str, pattern: str) -> List[int]:
    """String search using Z algorithm"""
    if not pattern:
        return []

    combined = pattern + "$" + text
    z = z_function(combined)
    m = len(pattern)

    return [i - m - 1 for i in range(m + 1, len(combined)) if z[i] == m]


# =============================================================================
# 4. Manacher's Algorithm (Longest Palindrome Substring)
# =============================================================================

def manacher(s: str) -> Tuple[int, int]:
    """
    Find the longest palindrome substring
    Time Complexity: O(n)
    Returns: (start index, length)
    """
    if not s:
        return 0, 0

    # Preprocessing: insert '#' between characters
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = palindrome radius centered at i

    c, r = 0, 0  # Current palindrome center, right boundary

    for i in range(n):
        if i < r:
            p[i] = min(r - i, p[2 * c - i])

        # Attempt expansion
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1

        # Update boundary
        if i + p[i] > r:
            c, r = i, i + p[i]

    # Find longest palindrome
    max_len = max(p)
    center = p.index(max_len)

    # Position in original string
    start = (center - max_len) // 2
    length = max_len

    return start, length


def longest_palindrome(s: str) -> str:
    """Return the longest palindrome substring"""
    start, length = manacher(s)
    return s[start:start + length]


# =============================================================================
# 5. Suffix Array (Simple Implementation)
# =============================================================================

def suffix_array(s: str) -> List[int]:
    """
    Suffix array construction (simple O(n log^2 n) implementation)
    Returns: starting indices of suffixes in lexicographic order
    """
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n

    k = 1
    while k < n:
        # Sort by (rank[i], rank[i+k])
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)

        sa.sort(key=key)

        # Compute new ranks
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i - 1]]
            if key(sa[i]) != key(sa[i - 1]):
                tmp[sa[i]] += 1

        rank = tmp[:]
        k *= 2

    return sa


def lcp_array(s: str, sa: List[int]) -> List[int]:
    """
    LCP array (Longest Common Prefix)
    lcp[i] = length of longest common prefix of s[sa[i]:] and s[sa[i+1]:]
    Time Complexity: O(n)
    """
    n = len(s)
    rank = [0] * n
    for i, idx in enumerate(sa):
        rank[idx] = i

    lcp = [0] * (n - 1)
    h = 0

    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1

    return lcp


# =============================================================================
# 6. Trie-Based String Search
# =============================================================================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.output = []  # For Aho-Corasick


class AhoCorasick:
    """
    Aho-Corasick Algorithm (multi-pattern search)
    Preprocessing: O(sum(|patterns|))
    Search: O(n + m), m = number of matches
    """

    def __init__(self, patterns: List[str]):
        self.root = TrieNode()
        self.patterns = patterns
        self._build_trie()
        self._build_failure()

    def _build_trie(self):
        for idx, pattern in enumerate(self.patterns):
            node = self.root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.output.append(idx)

    def _build_failure(self):
        from collections import deque

        queue = deque()
        self.root.fail = self.root

        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        while queue:
            node = queue.popleft()

            for char, child in node.children.items():
                fail = node.fail
                while fail != self.root and char not in fail.children:
                    fail = fail.fail

                child.fail = fail.children.get(char, self.root)
                if child.fail == child:
                    child.fail = self.root

                child.output += child.fail.output
                queue.append(child)

    def search(self, text: str) -> List[Tuple[int, int]]:
        """
        Search for all patterns in text
        Returns: [(position, pattern_index), ...]
        """
        results = []
        node = self.root

        for i, char in enumerate(text):
            while node != self.root and char not in node.children:
                node = node.fail

            node = node.children.get(char, self.root)

            for pattern_idx in node.output:
                pattern = self.patterns[pattern_idx]
                results.append((i - len(pattern) + 1, pattern_idx))

        return results


# =============================================================================
# 7. Edit Distance
# =============================================================================

def edit_distance(s1: str, s2: str) -> int:
    """
    Levenshtein distance (edit distance)
    Time Complexity: O(mn)
    Space Complexity: O(min(m, n)) optimized
    """
    m, n = len(s1), len(s2)

    # Space optimization: use only two rows
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i

        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])

        prev, curr = curr, prev

    return prev[n]


# =============================================================================
# 8. String Hashing
# =============================================================================

class StringHash:
    """
    Polynomial rolling hash
    Uses double hashing to reduce collisions
    """

    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        self.MOD1 = 10**9 + 7
        self.MOD2 = 10**9 + 9
        self.BASE1 = 31
        self.BASE2 = 37

        self.hash1 = [0] * (self.n + 1)
        self.hash2 = [0] * (self.n + 1)
        self.pow1 = [1] * (self.n + 1)
        self.pow2 = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash1[i + 1] = (self.hash1[i] * self.BASE1 + ord(s[i])) % self.MOD1
            self.hash2[i + 1] = (self.hash2[i] * self.BASE2 + ord(s[i])) % self.MOD2
            self.pow1[i + 1] = self.pow1[i] * self.BASE1 % self.MOD1
            self.pow2[i + 1] = self.pow2[i] * self.BASE2 % self.MOD2

    def get_hash(self, l: int, r: int) -> Tuple[int, int]:
        """Hash value of s[l:r] (0-indexed, half-open interval)"""
        h1 = (self.hash1[r] - self.hash1[l] * self.pow1[r - l]) % self.MOD1
        h2 = (self.hash2[r] - self.hash2[l] * self.pow2[r - l]) % self.MOD2
        return (h1, h2)

    def is_equal(self, l1: int, r1: int, l2: int, r2: int) -> bool:
        """Check if two substrings are equal"""
        if r1 - l1 != r2 - l2:
            return False
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("String Algorithm Examples")
    print("=" * 60)

    # 1. KMP
    print("\n[1] KMP Algorithm")
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    failure = kmp_failure(pattern)
    matches = kmp_search(text, pattern)
    print(f"    Text: {text}")
    print(f"    Pattern: {pattern}")
    print(f"    Failure function: {failure}")
    print(f"    Match positions: {matches}")

    # 2. Rabin-Karp
    print("\n[2] Rabin-Karp Algorithm")
    matches = rabin_karp_search(text, pattern)
    print(f"    Match positions: {matches}")

    # 3. Z Algorithm
    print("\n[3] Z Algorithm")
    s = "aabxaab"
    z = z_function(s)
    print(f"    String: {s}")
    print(f"    Z array: {z}")
    matches = z_search(text, pattern)
    print(f"    Search result: {matches}")

    # 4. Manacher
    print("\n[4] Manacher's Algorithm (Longest Palindrome)")
    s = "babad"
    palindrome = longest_palindrome(s)
    print(f"    String: {s}")
    print(f"    Longest palindrome: {palindrome}")

    s2 = "abacdfgdcaba"
    palindrome2 = longest_palindrome(s2)
    print(f"    String: {s2}")
    print(f"    Longest palindrome: {palindrome2}")

    # 5. Suffix Array
    print("\n[5] Suffix Array")
    s = "banana"
    sa = suffix_array(s)
    lcp = lcp_array(s, sa)
    print(f"    String: {s}")
    print(f"    Suffix array: {sa}")
    print("    Suffixes:")
    for i in sa:
        print(f"      {i}: {s[i:]}")
    print(f"    LCP array: {lcp}")

    # 6. Aho-Corasick
    print("\n[6] Aho-Corasick (Multi-Pattern)")
    patterns = ["he", "she", "his", "hers"]
    text = "ahishers"
    ac = AhoCorasick(patterns)
    results = ac.search(text)
    print(f"    Patterns: {patterns}")
    print(f"    Text: {text}")
    print("    Matches:")
    for pos, idx in results:
        print(f"      Position {pos}: '{patterns[idx]}'")

    # 7. Edit Distance
    print("\n[7] Edit Distance")
    s1, s2 = "kitten", "sitting"
    dist = edit_distance(s1, s2)
    print(f"    '{s1}' -> '{s2}'")
    print(f"    Edit distance: {dist}")

    # 8. String Hashing
    print("\n[8] String Hashing")
    s = "abcabc"
    sh = StringHash(s)
    print(f"    String: {s}")
    print(f"    hash(0:3) = {sh.get_hash(0, 3)}")
    print(f"    hash(3:6) = {sh.get_hash(3, 6)}")
    print(f"    s[0:3] == s[3:6]: {sh.is_equal(0, 3, 3, 6)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
