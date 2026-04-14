"""
11 Strings as Data Structures
=============================
Demonstrates trie, Rabin-Karp pattern matching,
and common string problems.
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children: node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find(prefix) is not None

    def _find(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children: return None
            node = node.children[ch]
        return node

    def autocomplete(self, prefix, limit=10):
        node = self._find(prefix)
        if not node: return []
        results = []
        self._collect(node, prefix, results, limit)
        return results

    def _collect(self, node, current, results, limit):
        if len(results) >= limit: return
        if node.is_end: results.append(current)
        for ch in sorted(node.children):
            self._collect(node.children[ch], current + ch, results, limit)


def rabin_karp(text, pattern):
    """Rabin-Karp string matching."""
    n, m = len(text), len(pattern)
    if m > n: return []
    BASE, MOD = 256, 101
    ph = wh = 0; h = pow(BASE, m-1, MOD)
    for i in range(m):
        ph = (BASE * ph + ord(pattern[i])) % MOD
        wh = (BASE * wh + ord(text[i])) % MOD
    positions = []
    for i in range(n - m + 1):
        if ph == wh and text[i:i+m] == pattern: positions.append(i)
        if i < n - m:
            wh = (BASE * (wh - ord(text[i]) * h) + ord(text[i+m])) % MOD
    return positions


if __name__ == "__main__":
    trie = Trie()
    for w in ["cat", "car", "card", "care", "do", "dog", "done"]:
        trie.insert(w)
    print("Trie demo:")
    for w in ["cat", "cab", "car", "care"]:
        print(f"  search('{w}'): {trie.search(w)}")
    print(f"  starts_with('ca'): {trie.starts_with('ca')}")
    print(f"  autocomplete('ca'): {trie.autocomplete('ca')}")
    print(f"  autocomplete('do'): {trie.autocomplete('do')}")

    print("\nRabin-Karp:")
    text, pat = "AABAACAADAABAABA", "AABA"
    print(f"  Text: '{text}', Pattern: '{pat}'")
    print(f"  Found at: {rabin_karp(text, pat)}")
