# Strings as Data Structures

**Previous**: [Sets and Maps](./10_Sets_and_Maps.md) | **Next**: [Sorting Fundamentals](./12_Sorting_Fundamentals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand how strings are stored in memory (immutability, interning)
2. Implement a trie (prefix tree) for efficient prefix-based operations
3. Apply basic pattern matching algorithms (brute-force, Rabin-Karp)
4. Use string hashing for efficient comparisons
5. Explain the difference between naive O(nm) and efficient O(n+m) matching
6. Implement an autocomplete system using a trie
7. Solve common string problems using appropriate data structures

---

Strings may look like simple sequences of characters, but they have rich data structure properties. This lesson covers specialized structures and algorithms for string processing, including **tries** (prefix trees), **string hashing**, and **pattern matching**.

## String Fundamentals in Python

### Immutability

Python strings are immutable. Every modification creates a new string:

```python
s = "hello"
# s[0] = "H"  # TypeError: 'str' does not support item assignment

# Concatenation creates a new string each time
s = s + " world"  # New string object created

# Efficient approach: join a list
parts = ["hello", " ", "world"]
result = "".join(parts)  # Single allocation
```

### String Interning

Python caches small strings to save memory:

```python
a = "hello"
b = "hello"
a is b  # True -- same object (interned)

a = "hello world!"
b = "hello world!"
a is b  # May be False -- not always interned
```

### String as Array of Characters

```
String: "HELLO"
Index:   0  1  2  3  4
       +--+--+--+--+--+
       |H |E |L |L |O |
       +--+--+--+--+--+

# Common operations
len("HELLO")           # 5, O(1)
"HELLO"[2]             # 'L', O(1)
"HELLO"[1:4]           # "ELL", O(k)
"LL" in "HELLO"        # True, O(n*m)
"HELLO".find("LL")     # 2, O(n*m)
```

## Trie (Prefix Tree)

A **trie** is a tree where each node represents a character, and paths from root to nodes form prefixes of stored strings. It enables O(m) lookup where m is the word length, regardless of how many words are stored.

```
Stored words: "cat", "car", "card", "care", "do", "dog"

          (root)
         /      \
        c        d
        |        |
        a        o
       / \       |
      t   r      g
         / \
        d   e

Search "car":  root -> c -> a -> r  (found!)
Search "cab":  root -> c -> a -> b  (not found, no 'b' child)
```

### Trie Implementation

```python
class TrieNode:
    """A node in a Trie."""
    
    __slots__ = ('children', 'is_end', 'count')
    
    def __init__(self):
        self.children = {}      # char -> TrieNode
        self.is_end = False     # Marks end of a word
        self.count = 0          # Number of words with this prefix


class Trie:
    """Trie (prefix tree) implementation."""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert a word -- O(m) where m = len(word)."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    def search(self, word):
        """Check if a word exists -- O(m)."""
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix -- O(m)."""
        return self._find_node(prefix) is not None
    
    def count_prefix(self, prefix):
        """Count words starting with prefix -- O(m)."""
        node = self._find_node(prefix)
        return node.count if node else 0
    
    def _find_node(self, prefix):
        """Navigate to the node at the end of prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def autocomplete(self, prefix, limit=10):
        """Return up to `limit` words starting with prefix."""
        node = self._find_node(prefix)
        if node is None:
            return []
        
        results = []
        self._collect_words(node, prefix, results, limit)
        return results
    
    def _collect_words(self, node, current, results, limit):
        if len(results) >= limit:
            return
        if node.is_end:
            results.append(current)
        for char in sorted(node.children):
            self._collect_words(node.children[char], current + char, 
                              results, limit)
    
    def delete(self, word):
        """Delete a word from the trie."""
        self._delete(self.root, word, 0)
    
    def _delete(self, node, word, depth):
        if depth == len(word):
            if node.is_end:
                node.is_end = False
                return len(node.children) == 0
            return False
        
        char = word[depth]
        if char not in node.children:
            return False
        
        child = node.children[char]
        child.count -= 1
        should_delete = self._delete(child, word, depth + 1)
        
        if should_delete:
            del node.children[char]
            return len(node.children) == 0 and not node.is_end
        
        return False
```

### Trie vs Hash Table for String Lookup

| Feature | Trie | Hash Table |
|---------|------|-----------|
| Exact lookup | O(m) | O(m) average |
| Prefix search | O(m) | O(n) scan all keys |
| Autocomplete | O(m + k) | O(n) scan |
| Space | Can be large (pointers) | Compact |
| Sorted iteration | Natural (DFS) | Need separate sort |

## Pattern Matching

### Brute-Force (Naive)

```python
def brute_force_search(text, pattern):
    """Find all occurrences of pattern in text -- O(n*m)."""
    n, m = len(text), len(pattern)
    positions = []
    
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
    
    return positions
```

### Rabin-Karp (String Hashing)

Uses a rolling hash to compare substrings in O(1):

```python
def rabin_karp(text, pattern):
    """Rabin-Karp string matching -- O(n+m) average.
    
    Uses rolling hash to avoid recomputing hash from scratch.
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    BASE = 256
    MOD = 101  # A prime number
    
    # Compute hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    h = pow(BASE, m - 1, MOD)  # BASE^(m-1) % MOD
    
    for i in range(m):
        pattern_hash = (BASE * pattern_hash + ord(pattern[i])) % MOD
        window_hash = (BASE * window_hash + ord(text[i])) % MOD
    
    positions = []
    for i in range(n - m + 1):
        if pattern_hash == window_hash:
            # Verify character by character (avoid hash collision false positive)
            if text[i:i + m] == pattern:
                positions.append(i)
        
        # Roll the hash: remove leading char, add trailing char
        if i < n - m:
            window_hash = (BASE * (window_hash - ord(text[i]) * h) 
                          + ord(text[i + m])) % MOD
            if window_hash < 0:
                window_hash += MOD
    
    return positions
```

```
Rolling Hash Example:
Text:    "ABCDE"   Pattern: "BCD"   Window size: 3

Window 1: "ABC" -> hash = h1
Window 2: "BCD" -> hash = h1 - h(A)*BASE^2 + h(D)  (rolling update!)
Window 3: "CDE" -> hash = h2 - h(B)*BASE^2 + h(E)
```

## Common String Problems

### Longest Common Prefix

```python
def longest_common_prefix(words):
    """Find the longest common prefix -- O(n*m).
    
    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    """
    if not words:
        return ""
    prefix = words[0]
    for word in words[1:]:
        while not word.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
```

### Anagram Check

```python
from collections import Counter

def is_anagram(s1, s2):
    """Check if two strings are anagrams -- O(n).
    
    >>> is_anagram("listen", "silent")
    True
    """
    return Counter(s1) == Counter(s2)
```

### First Non-Repeating Character

```python
def first_unique_char(s):
    """Find first non-repeating character -- O(n).
    
    >>> first_unique_char("leetcode")
    0
    """
    counts = Counter(s)
    for i, char in enumerate(s):
        if counts[char] == 1:
            return i
    return -1
```

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| String immutability | Modifications create new objects; use join() |
| Trie | O(m) lookup, natural prefix operations |
| Autocomplete | Trie + DFS to collect words with prefix |
| Brute-force matching | O(nm) -- compare at every position |
| Rabin-Karp | Rolling hash for O(n+m) average matching |
| String hashing | O(1) substring comparison via rolling hash |
| Anagram/counting | Counter for character frequency problems |

---

**Next**: [Sorting Fundamentals](./12_Sorting_Fundamentals.md) -- Learn classic sorting algorithms and their analysis.
