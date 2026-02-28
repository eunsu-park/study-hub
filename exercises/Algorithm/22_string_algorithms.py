"""
Exercises for Lesson 22: String Algorithms
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: KMP pattern matching, Rabin-Karp, Z-Algorithm.
"""


# === Exercise 1: KMP Pattern Matching ===
# Problem: Find all occurrences of pattern in text using KMP algorithm.
# Approach: Build failure function (partial match table), then search in O(n+m).

def exercise_1():
    """Solution: KMP string matching algorithm."""
    def build_failure(pattern):
        """Build KMP failure function (also called partial match table or lps array)."""
        m = len(pattern)
        failure = [0] * m
        j = 0  # length of previous longest proper prefix suffix

        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = failure[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            failure[i] = j

        return failure

    def kmp_search(text, pattern):
        """Find all starting indices where pattern occurs in text."""
        n, m = len(text), len(pattern)
        if m == 0:
            return []

        failure = build_failure(pattern)
        matches = []
        j = 0  # matched length in pattern

        for i in range(n):
            while j > 0 and text[i] != pattern[j]:
                j = failure[j - 1]
            if text[i] == pattern[j]:
                j += 1
            if j == m:
                matches.append(i - m + 1)
                j = failure[j - 1]

        return matches

    tests = [
        ("ABABDABACDABABCABAB", "ABABCABAB", [10]),
        ("AAAAAA", "AA", [0, 1, 2, 3, 4]),
        ("ABCDEF", "XYZ", []),
        ("hello world", "world", [6]),
        ("ABABABAB", "ABAB", [0, 2, 4]),
    ]

    for text, pattern, expected in tests:
        result = kmp_search(text, pattern)
        print(f'KMP("{text}", "{pattern}") = {result}')
        assert result == expected

    # Test failure function
    failure = build_failure("ABABCABAB")
    print(f"\nFailure function for 'ABABCABAB': {failure}")
    assert failure == [0, 0, 1, 2, 0, 1, 2, 3, 4]

    print("All KMP tests passed!")


# === Exercise 2: Rabin-Karp Pattern Matching ===
# Problem: Find all occurrences of pattern in text using rolling hash.
# Approach: Compute hash of pattern and rolling hash of text windows.

def exercise_2():
    """Solution: Rabin-Karp with polynomial rolling hash."""
    def rabin_karp(text, pattern):
        n, m = len(text), len(pattern)
        if m > n:
            return []

        BASE = 256
        MOD = 1000000007

        # Compute hash of pattern and first window of text
        pattern_hash = 0
        window_hash = 0
        # h = BASE^(m-1) mod MOD, used to remove the leading character
        h = pow(BASE, m - 1, MOD)

        for i in range(m):
            pattern_hash = (pattern_hash * BASE + ord(pattern[i])) % MOD
            window_hash = (window_hash * BASE + ord(text[i])) % MOD

        matches = []

        for i in range(n - m + 1):
            if window_hash == pattern_hash:
                # Hash match: verify character-by-character to avoid false positives
                if text[i:i + m] == pattern:
                    matches.append(i)

            # Slide the window: remove leading char, add trailing char
            if i < n - m:
                window_hash = ((window_hash - ord(text[i]) * h) * BASE
                               + ord(text[i + m])) % MOD
                if window_hash < 0:
                    window_hash += MOD

        return matches

    tests = [
        ("ABABDABACDABABCABAB", "ABABCABAB", [10]),
        ("AAAAAA", "AA", [0, 1, 2, 3, 4]),
        ("hello world hello", "hello", [0, 12]),
        ("ABCDEF", "XYZ", []),
    ]

    for text, pattern, expected in tests:
        result = rabin_karp(text, pattern)
        print(f'Rabin-Karp("{text}", "{pattern}") = {result}')
        assert result == expected

    print("All Rabin-Karp tests passed!")


# === Exercise 3: Z-Algorithm ===
# Problem: Compute the Z-array for a string and use it for pattern matching.
#   Z[i] = length of the longest substring starting at i that matches a prefix of s.

def exercise_3():
    """Solution: Z-Algorithm for O(n+m) pattern matching."""
    def z_function(s):
        n = len(s)
        z = [0] * n
        z[0] = n  # by convention, z[0] = length of string
        l, r = 0, 0

        for i in range(1, n):
            if i < r:
                z[i] = min(r - i, z[i - l])

            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1

            if i + z[i] > r:
                l, r = i, i + z[i]

        return z

    def z_search(text, pattern):
        """Use Z-algorithm to find all occurrences of pattern in text."""
        # Concatenate: pattern + '$' + text
        # The '$' separator ensures no match spans across the boundary
        combined = pattern + '$' + text
        z = z_function(combined)
        m = len(pattern)

        matches = []
        for i in range(m + 1, len(combined)):
            if z[i] == m:
                matches.append(i - m - 1)

        return matches

    # Test Z-function
    z = z_function("aabxaa")
    print(f"Z-function of 'aabxaa': {z}")
    assert z == [6, 1, 0, 0, 2, 1]

    z = z_function("aaaaa")
    print(f"Z-function of 'aaaaa': {z}")
    assert z == [5, 4, 3, 2, 1]

    # Test pattern matching
    tests = [
        ("ABABDABACDABABCABAB", "ABABCABAB", [10]),
        ("AAAAAA", "AA", [0, 1, 2, 3, 4]),
        ("hello world hello", "hello", [0, 12]),
    ]

    for text, pattern, expected in tests:
        result = z_search(text, pattern)
        print(f'Z-search("{text}", "{pattern}") = {result}')
        assert result == expected

    print("All Z-Algorithm tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: KMP Pattern Matching ===")
    exercise_1()
    print("\n=== Exercise 2: Rabin-Karp Pattern Matching ===")
    exercise_2()
    print("\n=== Exercise 3: Z-Algorithm ===")
    exercise_3()
    print("\nAll exercises completed!")
