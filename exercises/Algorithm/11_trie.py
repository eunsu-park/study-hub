"""
Exercises for Lesson 11: Trie
Topic: Algorithm

Solutions to practice problems from the lesson.
The lesson provides recommended problems: Implement Trie, Phone List,
Add and Search Word, Maximum XOR. We implement key ones here.
"""


# === Exercise 1: Implement Trie (Prefix Tree) ===
# Problem: Implement a Trie with insert, search, and startsWith operations.

def exercise_1():
    """Solution: Trie implementation with dictionary-based children."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    class Trie:
        def __init__(self):
            self.root = TrieNode()

        def insert(self, word):
            """Insert a word into the trie."""
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True

        def search(self, word):
            """Return True if the word is in the trie."""
            node = self._find_node(word)
            return node is not None and node.is_end

        def starts_with(self, prefix):
            """Return True if any word in the trie starts with the given prefix."""
            return self._find_node(prefix) is not None

        def _find_node(self, prefix):
            """Navigate the trie following the given prefix. Return the node or None."""
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return None
                node = node.children[char]
            return node

    trie = Trie()

    # Insert words
    words = ["apple", "app", "application", "bat", "ball", "band"]
    for w in words:
        trie.insert(w)
    print(f"Inserted: {words}")

    # Search tests
    search_tests = [
        ("apple", True),
        ("app", True),
        ("ap", False),       # prefix but not a complete word
        ("bat", True),
        ("ban", False),
        ("band", True),
        ("xyz", False),
    ]

    for word, expected in search_tests:
        result = trie.search(word)
        print(f"  search('{word}') = {result}")
        assert result == expected

    # Prefix tests
    prefix_tests = [
        ("app", True),
        ("ap", True),
        ("ba", True),
        ("ban", True),
        ("xyz", False),
        ("", True),  # empty prefix matches root
    ]

    print()
    for prefix, expected in prefix_tests:
        result = trie.starts_with(prefix)
        print(f"  starts_with('{prefix}') = {result}")
        assert result == expected

    print("All Trie implementation tests passed!")


# === Exercise 2: Add and Search Word (with wildcards) ===
# Problem: Design a data structure that supports adding words and searching
#   with '.' wildcard that matches any single character.

def exercise_2():
    """Solution: Trie with DFS for wildcard matching."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    class WordDictionary:
        def __init__(self):
            self.root = TrieNode()

        def add_word(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True

        def search(self, word):
            """Search with '.' wildcard support using DFS."""
            def dfs(node, idx):
                if idx == len(word):
                    return node.is_end

                char = word[idx]
                if char == '.':
                    # Try all children
                    for child in node.children.values():
                        if dfs(child, idx + 1):
                            return True
                    return False
                else:
                    if char not in node.children:
                        return False
                    return dfs(node.children[char], idx + 1)

            return dfs(self.root, 0)

    wd = WordDictionary()
    wd.add_word("bad")
    wd.add_word("dad")
    wd.add_word("mad")

    tests = [
        ("pad", False),
        ("bad", True),
        (".ad", True),    # matches bad, dad, mad
        ("b..", True),    # matches bad
        ("b.d", True),    # matches bad
        ("...", True),    # matches bad, dad, mad
        ("....", False),  # too long
        ("b", False),     # too short
    ]

    for word, expected in tests:
        result = wd.search(word)
        print(f"search('{word}') = {result}")
        assert result == expected

    print("All Wildcard Search tests passed!")


# === Exercise 3: Maximum XOR of Two Numbers ===
# Problem: Given an array of integers, find the maximum XOR of any two numbers.
#   Input: [3, 10, 5, 25, 2, 8]
#   Output: 28 (5 XOR 25 = 28)
# Approach: Build a binary trie of all numbers, then for each number greedily
#   choose the opposite bit at each level to maximize XOR.

def exercise_3():
    """Solution: XOR Trie for O(n * 32) = O(n) time."""
    def find_maximum_xor(nums):
        if len(nums) < 2:
            return 0

        # Determine the maximum number of bits needed
        max_num = max(nums)
        max_bits = max_num.bit_length() if max_num > 0 else 1

        # Build binary trie
        # Each node is [left_child, right_child] where left=0, right=1
        root = [None, None]

        def insert(num):
            node = root
            for i in range(max_bits - 1, -1, -1):
                bit = (num >> i) & 1
                if node[bit] is None:
                    node[bit] = [None, None]
                node = node[bit]

        def query(num):
            """Find the number in trie that gives maximum XOR with num."""
            node = root
            xor_val = 0
            for i in range(max_bits - 1, -1, -1):
                bit = (num >> i) & 1
                # Greedily choose the opposite bit to maximize XOR
                opposite = 1 - bit
                if node[opposite] is not None:
                    xor_val |= (1 << i)
                    node = node[opposite]
                else:
                    node = node[bit]
            return xor_val

        # Insert all numbers into trie
        for num in nums:
            insert(num)

        # Query each number to find maximum XOR
        max_xor = 0
        for num in nums:
            max_xor = max(max_xor, query(num))

        return max_xor

    tests = [
        ([3, 10, 5, 25, 2, 8], 28),    # 5 XOR 25 = 28
        ([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70], 127),
        ([0], 0),
        ([1, 2], 3),
        ([8, 4, 2, 1], 12),            # 8 XOR 4 = 12
    ]

    for nums, expected in tests:
        result = find_maximum_xor(nums)
        print(f"max_xor({nums}) = {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print("All Maximum XOR tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Implement Trie ===")
    exercise_1()
    print("\n=== Exercise 2: Add and Search Word ===")
    exercise_2()
    print("\n=== Exercise 3: Maximum XOR of Two Numbers ===")
    exercise_3()
    print("\nAll exercises completed!")
