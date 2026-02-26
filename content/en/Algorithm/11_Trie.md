# Trie

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Trie data structure and describe how it stores strings using shared prefixes
2. Implement a Trie with insert, search, and prefix-check operations in Python
3. Analyze the time and space complexity of Trie operations compared to hash maps and sorted arrays
4. Apply Trie structures to solve autocomplete and dictionary search problems
5. Implement an XOR Trie to solve maximum XOR problems on integer arrays

---

## Overview

A Trie is a tree data structure for efficiently storing and searching strings. Also called a Prefix Tree, it is used for autocomplete, dictionary search, and more.

---

## Table of Contents

1. [Trie Concepts](#1-trie-concepts)
2. [Basic Implementation](#2-basic-implementation)
3. [Trie Operations](#3-trie-operations)
4. [XOR Trie](#4-xor-trie)
5. [Application Problems](#5-application-problems)
6. [Practice Problems](#6-practice-problems)

---

## 1. Trie Concepts

### 1.1 Structure

```
Words: "apple", "app", "application", "bat", "ball"

           (root)
          /      \
        a          b
        |          |
        p          a
        |         / \
        p        t   l
       / \       |   |
      l   l      $   l
      |   |          |
      e   i          $
      |   |
      $   c
          |
          a
          |
          t
          |
          i
          |
          o
          |
          n
          |
          $

$ = End of word marker (isEnd)

Features:
- Root is an empty node
- Each edge represents one character
- Common prefixes are shared
```

### 1.2 Time Complexity

```
m = string length

┌─────────────┬─────────────┬────────────────┐
│ Operation   │ Time        │ Description    │
├─────────────┼─────────────┼────────────────┤
│ Insert      │ O(m)        │ String length  │
│ Search      │ O(m)        │ String length  │
│ Prefix      │ O(m)        │ Prefix length  │
│ Delete      │ O(m)        │ String length  │
└─────────────┴─────────────┴────────────────┘

Space: O(total characters) or O(n × m × alphabet size)
```

### 1.3 Trie vs HashSet

```
┌────────────────┬─────────────┬─────────────┐
│ Criteria       │ Trie        │ HashSet     │
├────────────────┼─────────────┼─────────────┤
│ Search         │ O(m)        │ O(m) avg    │
│ Prefix Search  │ O(p) ✓      │ O(n × m) ✗  │
│ Sorted Iter    │ Possible ✓  │ Not possible✗│
│ Space          │ Lower       │ Higher      │
│ Autocomplete   │ Optimal ✓   │ Inefficient ✗│
└────────────────┴─────────────┴─────────────┘

p = prefix length, n = number of words
```

---

## 2. Basic Implementation

### 2.1 Array-based (Fixed Alphabet)

```python
class TrieNode:
    def __init__(self):
        # Fixed-size array of 26 slots gives O(1) child access by index
        # (trade-off: wastes memory when alphabet usage is sparse)
        self.children = [None] * 26  # a-z
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def _char_to_index(self, c):
        # Map 'a'→0 … 'z'→25 so each character becomes a direct array index
        return ord(c) - ord('a')

    def insert(self, word):
        """Insert word - O(m)"""
        node = self.root
        for c in word:
            idx = self._char_to_index(c)
            # Lazily create nodes — only allocate memory for characters that appear
            if node.children[idx] is None:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        # Mark end here, not during traversal, so prefixes ('app') and full words
        # ('apple') share the same path but are distinguished by this flag
        node.is_end = True

    def search(self, word):
        """Search word - O(m)"""
        node = self.root
        for c in word:
            idx = self._char_to_index(c)
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        # Checking is_end distinguishes "app" (a word) from "appl" (just a prefix)
        return node.is_end

    def starts_with(self, prefix):
        """Check if prefix exists - O(p)"""
        node = self.root
        for c in prefix:
            idx = self._char_to_index(c)
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return True


# Usage example
trie = Trie()
trie.insert("apple")
trie.insert("app")
trie.insert("application")

print(trie.search("app"))       # True
print(trie.search("appl"))      # False
print(trie.starts_with("appl")) # True
```

### 2.2 Dictionary-based (Flexible Alphabet)

```python
class TrieNodeDict:
    def __init__(self):
        self.children = {}  # char → TrieNode
        self.is_end = False

class TrieDict:
    def __init__(self):
        self.root = TrieNodeDict()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNodeDict()
            node = node.children[c]
        node.is_end = True

    def search(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True

    def delete(self, word):
        """Delete word"""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False  # Word not found
                # Clear the end-of-word marker; the node itself may still be needed
                # as a prefix node for other words, so we don't delete it yet
                node.is_end = False
                # Signal to the parent that this node can be removed only if it
                # has no children (i.e., it's not shared with any other word)
                return len(node.children) == 0

            c = word[depth]
            if c not in node.children:
                return False

            should_delete = _delete(node.children[c], word, depth + 1)

            if should_delete:
                # Safe to prune this child only because it has no further dependents
                del node.children[c]
                # Propagate the pruning signal upward if this node is also no longer
                # needed — neither an end marker nor a branching point for other words
                return len(node.children) == 0 and not node.is_end

            return False

        _delete(self.root, word, 0)
```

### 2.3 C++ Implementation

```cpp
#include <string>
#include <unordered_map>
using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEnd = false;
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->isEnd = true;
    }

    bool search(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return node->isEnd;
    }

    bool startsWith(const string& prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return true;
    }
};
```

---

## 3. Trie Operations

### 3.1 Autocomplete (Find All Words)

```python
class AutocompleteTrie(TrieDict):
    def autocomplete(self, prefix):
        """Return all words starting with prefix"""
        node = self.root
        for c in prefix:
            if c not in node.children:
                return []
            node = node.children[c]

        result = []
        self._collect_words(node, prefix, result)
        return result

    def _collect_words(self, node, current, result):
        if node.is_end:
            result.append(current)

        for c, child in node.children.items():
            self._collect_words(child, current + c, result)


# Usage example
trie = AutocompleteTrie()
for word in ["apple", "app", "application", "apply", "banana"]:
    trie.insert(word)

print(trie.autocomplete("app"))
# ['app', 'apple', 'application', 'apply']
```

### 3.2 Counting Words

```python
class CountingTrie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {'count': 0}
            node = node[c]
            node['count'] += 1  # Number of words with this prefix
        node['$'] = True  # End of word

    def count_prefix(self, prefix):
        """Count words starting with prefix"""
        node = self.root
        for c in prefix:
            if c not in node:
                return 0
            node = node[c]
        return node.get('count', 0)

    def count_words(self):
        """Count total words"""
        def dfs(node):
            count = 1 if '$' in node else 0
            for c, child in node.items():
                if c not in ['count', '$']:
                    count += dfs(child)
            return count
        return dfs(self.root)


# Usage example
trie = CountingTrie()
for word in ["apple", "app", "application"]:
    trie.insert(word)

print(trie.count_prefix("app"))  # 3
print(trie.count_prefix("appl"))  # 2
print(trie.count_words())  # 3
```

### 3.3 Longest Common Prefix

```python
def longest_common_prefix(words):
    """Longest common prefix of all words"""
    if not words:
        return ""

    trie = TrieDict()
    for word in words:
        trie.insert(word)

    prefix = []
    node = trie.root

    while True:
        # Continue if only one child and not end of word
        if len(node.children) != 1 or node.is_end:
            break

        c, child = next(iter(node.children.items()))
        prefix.append(c)
        node = child

    return ''.join(prefix)


# Example
words = ["flower", "flow", "flight"]
print(longest_common_prefix(words))  # "fl"
```

### 3.4 Wildcard Search

```python
class WildcardTrie(TrieDict):
    def search_with_wildcard(self, word):
        """'.' matches any character"""
        return self._search(self.root, word, 0)

    def _search(self, node, word, idx):
        if idx == len(word):
            return node.is_end

        c = word[idx]
        if c == '.':
            # Search all children
            for child in node.children.values():
                if self._search(child, word, idx + 1):
                    return True
            return False
        else:
            if c not in node.children:
                return False
            return self._search(node.children[c], word, idx + 1)


# Example
trie = WildcardTrie()
trie.insert("bad")
trie.insert("dad")
trie.insert("mad")

print(trie.search_with_wildcard("pad"))  # False
print(trie.search_with_wildcard("bad"))  # True
print(trie.search_with_wildcard(".ad"))  # True
print(trie.search_with_wildcard("b.."))  # True
```

---

## 4. XOR Trie

### 4.1 Concept

```
XOR Trie: Trie storing integers as binary numbers
- Used for finding maximum XOR pair
- Store each bit from most significant

Example: Store 3, 10, 5 (4 bits)
3  = 0011
10 = 1010
5  = 0101

        root
       /    \
      0      1
     / \      \
    0   1      0
    |   |      |
    1   0      1
    |   |      |
    1   1      0
    ↓   ↓      ↓
    3   5      10
```

### 4.2 Maximum XOR Pair

```python
class XORTrie:
    def __init__(self, max_bits=30):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num):
        """Insert number"""
        node = self.root
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def find_max_xor(self, num):
        """Find XOR result with number that maximizes XOR with num"""
        node = self.root
        result = 0

        # Process from most significant bit to least significant bit;
        # greedy bit-by-bit maximization works because each bit position
        # is more significant than all lower bits combined
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            # XOR is 1 when bits differ, so the opposite bit always maximizes
            # the contribution at this position
            opposite = 1 - bit

            if opposite in node:
                # Take the path that sets this XOR bit to 1
                result |= (1 << i)
                node = node[opposite]
            elif bit in node:
                # Opposite bit not available — this XOR bit will be 0
                node = node[bit]
            else:
                break

        return result


def find_maximum_xor(nums):
    """Find maximum XOR pair in array"""
    if len(nums) < 2:
        return 0

    trie = XORTrie()
    max_xor = 0

    for num in nums:
        trie.insert(num)
        max_xor = max(max_xor, trie.find_max_xor(num))

    return max_xor


# Example
nums = [3, 10, 5, 25, 2, 8]
print(find_maximum_xor(nums))  # 28 (5 XOR 25 = 28)
```

### 4.3 Range XOR Maximum

```python
class PersistentXORTrie:
    """
    Maximum XOR with k in range [l, r]
    Offline queries + Persistent Trie
    """
    def __init__(self, max_bits=30):
        self.max_bits = max_bits
        self.nodes = [[0, 0]]  # [left_child, right_child]
        self.count = [0]  # Number of numbers passing each node
        self.roots = [0]  # Root per version

    def insert(self, prev_root, num):
        """Add num from previous version"""
        new_root = len(self.nodes)
        self.nodes.append([0, 0])
        self.count.append(0)

        curr = new_root
        prev = prev_root

        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1

            # Create new node
            child = len(self.nodes)
            self.nodes.append([0, 0])
            self.count.append(0)

            # Copy opposite child from previous version
            self.nodes[curr][1 - bit] = self.nodes[prev][1 - bit] if prev else 0
            self.nodes[curr][bit] = child

            # Update count
            self.count[child] = (self.count[self.nodes[prev][bit]] if prev else 0) + 1

            curr = child
            prev = self.nodes[prev][bit] if prev else 0

        self.roots.append(new_root)
        return new_root

    def query(self, l_root, r_root, num):
        """Maximum XOR with num between versions (l, r]"""
        result = 0
        l_node = l_root
        r_node = r_root

        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            opposite = 1 - bit

            l_opp_count = self.count[self.nodes[l_node][opposite]] if l_node else 0
            r_opp_count = self.count[self.nodes[r_node][opposite]] if r_node else 0

            if r_opp_count - l_opp_count > 0:
                result |= (1 << i)
                l_node = self.nodes[l_node][opposite] if l_node else 0
                r_node = self.nodes[r_node][opposite]
            else:
                l_node = self.nodes[l_node][bit] if l_node else 0
                r_node = self.nodes[r_node][bit]

        return result
```

---

## 5. Application Problems

### 5.1 Add and Search Word

```python
# LeetCode 211. Design Add and Search Words Data Structure
class WordDictionary:
    def __init__(self):
        self.trie = WildcardTrie()

    def addWord(self, word):
        self.trie.insert(word)

    def search(self, word):
        return self.trie.search_with_wildcard(word)
```

### 5.2 Replace Words

```python
def replace_words(dictionary, sentence):
    """
    Replace each word in sentence with dictionary prefix
    dictionary = ["cat", "bat", "rat"]
    sentence = "the cattle was rattled by the battery"
    → "the cat was rat by the bat"
    """
    trie = TrieDict()
    for word in dictionary:
        trie.insert(word)

    def find_root(word):
        node = trie.root
        for i, c in enumerate(word):
            if c not in node.children:
                return word
            node = node.children[c]
            if node.is_end:
                return word[:i + 1]
        return word

    words = sentence.split()
    return ' '.join(find_root(word) for word in words)


# Example
dictionary = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"
print(replace_words(dictionary, sentence))
# "the cat was rat by the bat"
```

### 5.3 Concatenated Words

```python
def find_all_concatenated_words(words):
    """
    Find words that can be formed by concatenating other words
    """
    trie = TrieDict()
    for word in words:
        if word:
            trie.insert(word)

    def can_form(word, start, count):
        if start == len(word):
            return count >= 2

        node = trie.root
        for i in range(start, len(word)):
            c = word[i]
            if c not in node.children:
                return False
            node = node.children[c]
            if node.is_end:
                if can_form(word, i + 1, count + 1):
                    return True

        return False

    result = []
    for word in words:
        if word and can_form(word, 0, 0):
            result.append(word)

    return result


# Example
words = ["cat", "cats", "catsdogcats", "dog", "dogcatsdog",
         "hippopotamuses", "rat", "ratcatdogcat"]
print(find_all_concatenated_words(words))
# ["catsdogcats", "dogcatsdog", "ratcatdogcat"]
```

### 5.4 Suffix Trie

```python
class SuffixTrie:
    """Trie storing all suffixes of a string"""

    def __init__(self, text):
        self.root = {}
        self._build(text)

    def _build(self, text):
        for i in range(len(text)):
            node = self.root
            for c in text[i:]:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node['$'] = i  # Store starting index

    def search(self, pattern):
        """All starting positions where pattern exists"""
        node = self.root
        for c in pattern:
            if c not in node:
                return []
            node = node[c]

        # Collect all $ under this node
        result = []
        self._collect(node, result)
        return result

    def _collect(self, node, result):
        if '$' in node:
            result.append(node['$'])
        for c, child in node.items():
            if c != '$':
                self._collect(child, result)


# Example
st = SuffixTrie("banana")
print(st.search("ana"))  # [1, 3]
print(st.search("nan"))  # [2]
```

---

## 6. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐ | [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/) | LeetCode | Basic implementation |
| ⭐⭐⭐ | [Phone List](https://www.acmicpc.net/problem/5052) | BOJ | Prefix |
| ⭐⭐⭐ | [Design Search Autocomplete](https://leetcode.com/problems/design-search-autocomplete-system/) | LeetCode | Autocomplete |
| ⭐⭐⭐ | [Add and Search Word](https://leetcode.com/problems/design-add-and-search-words-data-structure/) | LeetCode | Wildcard |
| ⭐⭐⭐⭐ | [Maximum XOR of Two Numbers](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) | LeetCode | XOR Trie |
| ⭐⭐⭐⭐ | [String Set](https://www.acmicpc.net/problem/14425) | BOJ | Search |

---

## Time/Space Complexity

```
┌─────────────────┬─────────────┬─────────────────────┐
│ Operation       │ Time        │ Space               │
├─────────────────┼─────────────┼─────────────────────┤
│ Insert          │ O(m)        │ O(m × alphabet)     │
│ Search          │ O(m)        │ -                   │
│ Prefix Search   │ O(p)        │ -                   │
│ Autocomplete    │ O(p + k)    │ O(result length)    │
│ XOR Maximum     │ O(log MAX)  │ O(n × log MAX)      │
└─────────────────┴─────────────┴─────────────────────┘

m = word length, p = prefix length, k = result count
```

---

## Next Steps

- [12_Graph_Basics.md](./12_Graph_Basics.md) - Graph Basics

---

## References

- [Trie](https://cp-algorithms.com/string/trie.html)
- [XOR Trie](https://codeforces.com/blog/entry/65408)
