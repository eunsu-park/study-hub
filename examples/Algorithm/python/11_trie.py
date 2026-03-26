"""
Trie (Prefix Tree)
Trie Data Structure

Implements tries for string search and prefix-based operations.
"""

from typing import List, Optional, Dict
from collections import defaultdict


# =============================================================================
# 1. Basic Trie (Dictionary-based)
# =============================================================================

class TrieNode:
    """Trie node"""

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        self.count: int = 0  # Number of words starting with this prefix


class Trie:
    """
    Trie (Prefix Tree)
    - Insert: O(m), m = word length
    - Search: O(m)
    - Prefix search: O(m)
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert word - O(m)"""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1

        node.is_end = True

    def search(self, word: str) -> bool:
        """Check if word exists - O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix - O(m)"""
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """Count words starting with prefix - O(m)"""
        node = self._find_node(prefix)
        return node.count if node else 0

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find the node corresponding to prefix"""
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node

    def delete(self, word: str) -> bool:
        """Delete word - O(m)"""

        def _delete(node: TrieNode, word: str, depth: int) -> bool:
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0

            char = word[depth]
            if char not in node.children:
                return False

            should_delete = _delete(node.children[char], word, depth + 1)

            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end

            node.children[char].count -= 1
            return False

        return _delete(self.root, word, 0)


# =============================================================================
# 2. Autocomplete
# =============================================================================

class AutocompleteSystem:
    """Autocomplete system"""

    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word: str, weight: int = 1) -> None:
        """Add word (with weight)"""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True
        node.count = weight  # Here count represents frequency/weight

    def autocomplete(self, prefix: str, limit: int = 5) -> List[str]:
        """List words starting with prefix - O(m + k), k = number of results"""
        node = self.root

        # Find prefix node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all words via DFS
        results = []
        self._collect_words(node, prefix, results)

        # Sort by frequency and return top limit
        results.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in results[:limit]]

    def _collect_words(self, node: TrieNode, prefix: str, results: List) -> None:
        """Collect all complete words from a node"""
        if node.is_end:
            results.append((prefix, node.count))

        for char, child in node.children.items():
            self._collect_words(child, prefix + char, results)


# =============================================================================
# 3. Wildcard Search
# =============================================================================

class WildcardTrie:
    """Trie supporting wildcard '.' character"""

    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        """'.' matches any single character"""
        return self._search(self.root, word, 0)

    def _search(self, node: TrieNode, word: str, index: int) -> bool:
        if index == len(word):
            return node.is_end

        char = word[index]

        if char == '.':
            # Try all child nodes
            for child in node.children.values():
                if self._search(child, word, index + 1):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return self._search(node.children[char], word, index + 1)


# =============================================================================
# 4. Longest Common Prefix (LCP)
# =============================================================================

def longest_common_prefix(words: List[str]) -> str:
    """
    Longest common prefix of a word array
    Using trie
    """
    if not words:
        return ""

    # Build trie
    trie = Trie()
    for word in words:
        trie.insert(word)

    # Traverse from root to branching point
    prefix = []
    node = trie.root

    while node:
        # Continue only when there's exactly one child and it's not an end node
        if len(node.children) == 1 and not node.is_end:
            char = list(node.children.keys())[0]
            prefix.append(char)
            node = node.children[char]
        else:
            break

    return ''.join(prefix)


# =============================================================================
# 5. Word Dictionary
# =============================================================================

class WordDictionary:
    """
    Word add/search dictionary
    - Exact search
    - Prefix search
    - Wildcard search
    """

    def __init__(self):
        self.trie = Trie()
        self.wildcard_trie = WildcardTrie()

    def add_word(self, word: str) -> None:
        self.trie.insert(word)
        self.wildcard_trie.add_word(word)

    def search_exact(self, word: str) -> bool:
        """Exact word search"""
        return self.trie.search(word)

    def search_prefix(self, prefix: str) -> bool:
        """Prefix search"""
        return self.trie.starts_with(prefix)

    def search_pattern(self, pattern: str) -> bool:
        """Pattern search (. = any character)"""
        return self.wildcard_trie.search(pattern)


# =============================================================================
# 6. XOR Trie (Find Maximum XOR Pair)
# =============================================================================

class XORTrie:
    """
    Find maximum XOR pair using bit trie
    Stores each number in binary
    """

    def __init__(self, max_bits: int = 31):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num: int) -> None:
        """Insert number - O(max_bits)"""
        node = self.root

        for i in range(self.max_bits, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def find_max_xor(self, num: int) -> int:
        """Return value that gives maximum XOR with num - O(max_bits)"""
        node = self.root
        result = 0

        for i in range(self.max_bits, -1, -1):
            bit = (num >> i) & 1
            # Choose opposite bit to maximize XOR
            toggle_bit = 1 - bit

            if toggle_bit in node:
                result |= (1 << i)
                node = node[toggle_bit]
            elif bit in node:
                node = node[bit]
            else:
                break

        return result


def find_maximum_xor(nums: List[int]) -> int:
    """
    Maximum XOR of two numbers in an array
    Time Complexity: O(n * max_bits)
    """
    if len(nums) < 2:
        return 0

    xor_trie = XORTrie()
    max_xor = 0

    for num in nums:
        xor_trie.insert(num)
        max_xor = max(max_xor, xor_trie.find_max_xor(num))

    return max_xor


# =============================================================================
# 7. Practical Problem: Word Search II
# =============================================================================

def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """
    Find words from a list in a 2D board
    Trie + DFS
    """
    # Build trie
    trie = Trie()
    for word in words:
        trie.insert(word)

    rows, cols = len(board), len(board[0])
    result = set()

    def dfs(r: int, c: int, node: TrieNode, path: str) -> None:
        if node.is_end:
            result.add(path)

        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        char = board[r][c]
        if char == '#' or char not in node.children:
            return

        board[r][c] = '#'  # Mark as visited
        next_node = node.children[char]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs(r + dr, c + dc, next_node, path + char)

        board[r][c] = char  # Restore

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root, "")

    return list(result)


# =============================================================================
# 8. Prefix/Suffix Simultaneous Search
# =============================================================================

class PrefixSuffixTrie:
    """Trie for simultaneous prefix and suffix search"""

    def __init__(self, words: List[str]):
        self.trie = {}

        for idx, word in enumerate(words):
            # Store in the form suffix#word
            key = word + '#' + word
            for i in range(len(word) + 1):
                node = self.trie
                for char in key[i:]:
                    if char not in node:
                        node[char] = {'idx': -1}
                    node = node[char]
                    node['idx'] = idx  # Store the largest index

    def search(self, prefix: str, suffix: str) -> int:
        """Index of word satisfying both prefix and suffix"""
        key = suffix + '#' + prefix
        node = self.trie

        for char in key:
            if char not in node:
                return -1
            node = node[char]

        return node.get('idx', -1)


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Trie (Prefix Tree) Examples")
    print("=" * 60)

    # 1. Basic Trie
    print("\n[1] Basic Trie")
    trie = Trie()
    words = ["apple", "app", "application", "banana", "band"]
    for word in words:
        trie.insert(word)
    print(f"    Insert: {words}")
    print(f"    search('app'): {trie.search('app')}")
    print(f"    search('application'): {trie.search('application')}")
    print(f"    search('apply'): {trie.search('apply')}")
    print(f"    starts_with('app'): {trie.starts_with('app')}")
    print(f"    count_prefix('app'): {trie.count_prefix('app')}")

    # 2. Autocomplete
    print("\n[2] Autocomplete")
    auto = AutocompleteSystem()
    search_words = [("hello", 5), ("help", 3), ("helicopter", 2), ("hero", 4), ("world", 1)]
    for word, weight in search_words:
        auto.add_word(word, weight)
    print(f"    Words/frequency: {search_words}")
    print(f"    'hel' autocomplete: {auto.autocomplete('hel')}")
    print(f"    'he' autocomplete: {auto.autocomplete('he')}")

    # 3. Wildcard Search
    print("\n[3] Wildcard Search")
    wild = WildcardTrie()
    for word in ["bad", "dad", "mad", "pad"]:
        wild.add_word(word)
    print(f"    Words: ['bad', 'dad', 'mad', 'pad']")
    print(f"    search('.ad'): {wild.search('.ad')}")
    print(f"    search('b..'): {wild.search('b..')}")
    print(f"    search('..d'): {wild.search('..d')}")
    print(f"    search('b.d'): {wild.search('b.d')}")

    # 4. Longest Common Prefix
    print("\n[4] Longest Common Prefix")
    words_lcp = ["flower", "flow", "flight"]
    lcp = longest_common_prefix(words_lcp)
    print(f"    Words: {words_lcp}")
    print(f"    LCP: '{lcp}'")

    # 5. XOR Trie
    print("\n[5] Maximum XOR (XOR Trie)")
    nums = [3, 10, 5, 25, 2, 8]
    max_xor = find_maximum_xor(nums)
    print(f"    Array: {nums}")
    print(f"    Maximum XOR: {max_xor}")
    print(f"    (5 XOR 25 = {5 ^ 25})")

    # 6. Word Dictionary
    print("\n[6] Word Dictionary")
    dictionary = WordDictionary()
    for word in ["hello", "help", "world"]:
        dictionary.add_word(word)
    print(f"    Words: ['hello', 'help', 'world']")
    print(f"    exact 'help': {dictionary.search_exact('help')}")
    print(f"    prefix 'hel': {dictionary.search_prefix('hel')}")
    print(f"    pattern 'h.l.o': {dictionary.search_pattern('h.l.o')}")

    # 7. Word Deletion
    print("\n[7] Word Deletion")
    trie2 = Trie()
    for word in ["apple", "app"]:
        trie2.insert(word)
    print(f"    Insert: ['apple', 'app']")
    print(f"    search('app'): {trie2.search('app')}")
    trie2.delete("app")
    print(f"    After delete('app')")
    print(f"    search('app'): {trie2.search('app')}")
    print(f"    search('apple'): {trie2.search('apple')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
