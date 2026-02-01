# 트라이 (Trie)

## 개요

트라이(Trie)는 문자열을 효율적으로 저장하고 검색하는 트리 자료구조입니다. 접두사 트리(Prefix Tree)라고도 불리며, 자동완성, 사전 검색 등에 활용됩니다.

---

## 목차

1. [트라이 개념](#1-트라이-개념)
2. [기본 구현](#2-기본-구현)
3. [트라이 연산](#3-트라이-연산)
4. [XOR 트라이](#4-xor-트라이)
5. [활용 문제](#5-활용-문제)
6. [연습 문제](#6-연습-문제)

---

## 1. 트라이 개념

### 1.1 구조

```
단어: "apple", "app", "application", "bat", "ball"

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

$ = 단어 끝 표시 (isEnd)

특징:
- 루트는 빈 노드
- 각 간선은 문자 하나를 나타냄
- 공통 접두사를 공유
```

### 1.2 시간 복잡도

```
m = 문자열 길이

┌─────────────┬─────────────┬────────────────┐
│ 연산         │ 시간        │ 설명            │
├─────────────┼─────────────┼────────────────┤
│ 삽입         │ O(m)        │ 문자열 길이만큼 │
│ 검색         │ O(m)        │ 문자열 길이만큼 │
│ 접두사 검색  │ O(m)        │ 접두사 길이만큼 │
│ 삭제         │ O(m)        │ 문자열 길이만큼 │
└─────────────┴─────────────┴────────────────┘

공간: O(총 문자 수) 또는 O(n × m × 알파벳 크기)
```

### 1.3 트라이 vs 해시셋

```
┌────────────────┬─────────────┬─────────────┐
│ 기준           │ 트라이       │ 해시셋       │
├────────────────┼─────────────┼─────────────┤
│ 검색           │ O(m)        │ O(m) 평균    │
│ 접두사 검색    │ O(p) ✓      │ O(n × m) ✗  │
│ 정렬된 순회    │ 가능 ✓      │ 불가능 ✗    │
│ 공간 효율      │ 낮음        │ 높음         │
│ 자동완성       │ 최적 ✓      │ 비효율 ✗    │
└────────────────┴─────────────┴─────────────┘

p = 접두사 길이, n = 단어 수
```

---

## 2. 기본 구현

### 2.1 배열 기반 (고정 알파벳)

```python
class TrieNode:
    def __init__(self):
        self.children = [None] * 26  # a-z
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def _char_to_index(self, c):
        return ord(c) - ord('a')

    def insert(self, word):
        """단어 삽입 - O(m)"""
        node = self.root
        for c in word:
            idx = self._char_to_index(c)
            if node.children[idx] is None:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        node.is_end = True

    def search(self, word):
        """단어 검색 - O(m)"""
        node = self.root
        for c in word:
            idx = self._char_to_index(c)
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return node.is_end

    def starts_with(self, prefix):
        """접두사 존재 여부 - O(p)"""
        node = self.root
        for c in prefix:
            idx = self._char_to_index(c)
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return True


# 사용 예시
trie = Trie()
trie.insert("apple")
trie.insert("app")
trie.insert("application")

print(trie.search("app"))       # True
print(trie.search("appl"))      # False
print(trie.starts_with("appl")) # True
```

### 2.2 딕셔너리 기반 (유연한 알파벳)

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
        """단어 삭제"""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False  # 단어 없음
                node.is_end = False
                return len(node.children) == 0

            c = word[depth]
            if c not in node.children:
                return False

            should_delete = _delete(node.children[c], word, depth + 1)

            if should_delete:
                del node.children[c]
                return len(node.children) == 0 and not node.is_end

            return False

        _delete(self.root, word, 0)
```

### 2.3 C++ 구현

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

## 3. 트라이 연산

### 3.1 자동완성 (모든 단어 찾기)

```python
class AutocompleteTrie(TrieDict):
    def autocomplete(self, prefix):
        """접두사로 시작하는 모든 단어 반환"""
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


# 사용 예시
trie = AutocompleteTrie()
for word in ["apple", "app", "application", "apply", "banana"]:
    trie.insert(word)

print(trie.autocomplete("app"))
# ['app', 'apple', 'application', 'apply']
```

### 3.2 단어 개수 세기

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
            node['count'] += 1  # 이 접두사를 가진 단어 수
        node['$'] = True  # 단어 끝

    def count_prefix(self, prefix):
        """접두사로 시작하는 단어 수"""
        node = self.root
        for c in prefix:
            if c not in node:
                return 0
            node = node[c]
        return node.get('count', 0)

    def count_words(self):
        """전체 단어 수"""
        def dfs(node):
            count = 1 if '$' in node else 0
            for c, child in node.items():
                if c not in ['count', '$']:
                    count += dfs(child)
            return count
        return dfs(self.root)


# 사용 예시
trie = CountingTrie()
for word in ["apple", "app", "application"]:
    trie.insert(word)

print(trie.count_prefix("app"))  # 3
print(trie.count_prefix("appl"))  # 2
print(trie.count_words())  # 3
```

### 3.3 가장 긴 공통 접두사

```python
def longest_common_prefix(words):
    """모든 단어의 가장 긴 공통 접두사"""
    if not words:
        return ""

    trie = TrieDict()
    for word in words:
        trie.insert(word)

    prefix = []
    node = trie.root

    while True:
        # 자식이 하나이고, 단어 끝이 아니면 계속
        if len(node.children) != 1 or node.is_end:
            break

        c, child = next(iter(node.children.items()))
        prefix.append(c)
        node = child

    return ''.join(prefix)


# 예시
words = ["flower", "flow", "flight"]
print(longest_common_prefix(words))  # "fl"
```

### 3.4 와일드카드 검색

```python
class WildcardTrie(TrieDict):
    def search_with_wildcard(self, word):
        """'.'은 모든 문자와 매칭"""
        return self._search(self.root, word, 0)

    def _search(self, node, word, idx):
        if idx == len(word):
            return node.is_end

        c = word[idx]
        if c == '.':
            # 모든 자식 탐색
            for child in node.children.values():
                if self._search(child, word, idx + 1):
                    return True
            return False
        else:
            if c not in node.children:
                return False
            return self._search(node.children[c], word, idx + 1)


# 예시
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

## 4. XOR 트라이

### 4.1 개념

```
XOR 트라이: 정수를 이진수로 저장하는 트라이
- 최대 XOR 쌍 찾기에 활용
- 각 비트를 높은 자리부터 저장

예시: 3, 10, 5를 저장 (4비트)
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

### 4.2 최대 XOR 쌍

```python
class XORTrie:
    def __init__(self, max_bits=30):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num):
        """숫자 삽입"""
        node = self.root
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def find_max_xor(self, num):
        """num과 XOR했을 때 최대값을 만드는 수의 XOR 결과"""
        node = self.root
        result = 0

        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            # 반대 비트를 선택하면 XOR이 최대
            opposite = 1 - bit

            if opposite in node:
                result |= (1 << i)
                node = node[opposite]
            elif bit in node:
                node = node[bit]
            else:
                break

        return result


def find_maximum_xor(nums):
    """배열에서 최대 XOR 쌍 찾기"""
    if len(nums) < 2:
        return 0

    trie = XORTrie()
    max_xor = 0

    for num in nums:
        trie.insert(num)
        max_xor = max(max_xor, trie.find_max_xor(num))

    return max_xor


# 예시
nums = [3, 10, 5, 25, 2, 8]
print(find_maximum_xor(nums))  # 28 (5 XOR 25 = 28)
```

### 4.3 구간 XOR 최대

```python
class PersistentXORTrie:
    """
    구간 [l, r]에서 k와 XOR 최대값
    오프라인 쿼리 + Persistent Trie
    """
    def __init__(self, max_bits=30):
        self.max_bits = max_bits
        self.nodes = [[0, 0]]  # [left_child, right_child]
        self.count = [0]  # 각 노드를 지나는 숫자 개수
        self.roots = [0]  # 버전별 루트

    def insert(self, prev_root, num):
        """이전 버전에서 num 추가"""
        new_root = len(self.nodes)
        self.nodes.append([0, 0])
        self.count.append(0)

        curr = new_root
        prev = prev_root

        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1

            # 새 노드 생성
            child = len(self.nodes)
            self.nodes.append([0, 0])
            self.count.append(0)

            # 반대쪽 자식은 이전 버전에서 복사
            self.nodes[curr][1 - bit] = self.nodes[prev][1 - bit] if prev else 0
            self.nodes[curr][bit] = child

            # 카운트 업데이트
            self.count[child] = (self.count[self.nodes[prev][bit]] if prev else 0) + 1

            curr = child
            prev = self.nodes[prev][bit] if prev else 0

        self.roots.append(new_root)
        return new_root

    def query(self, l_root, r_root, num):
        """버전 (l, r] 사이에서 num과 최대 XOR"""
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

## 5. 활용 문제

### 5.1 단어 추가 및 검색

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

### 5.2 단어 대체하기

```python
def replace_words(dictionary, sentence):
    """
    문장의 각 단어를 사전의 접두사로 대체
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


# 예시
dictionary = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"
print(replace_words(dictionary, sentence))
# "the cat was rat by the bat"
```

### 5.3 연결 단어

```python
def find_all_concatenated_words(words):
    """
    다른 단어들을 이어붙여 만들 수 있는 단어 찾기
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


# 예시
words = ["cat", "cats", "catsdogcats", "dog", "dogcatsdog",
         "hippopotamuses", "rat", "ratcatdogcat"]
print(find_all_concatenated_words(words))
# ["catsdogcats", "dogcatsdog", "ratcatdogcat"]
```

### 5.4 Suffix Trie (접미사 트라이)

```python
class SuffixTrie:
    """문자열의 모든 접미사를 저장하는 트라이"""

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
            node['$'] = i  # 시작 인덱스 저장

    def search(self, pattern):
        """패턴이 존재하는 모든 시작 위치"""
        node = self.root
        for c in pattern:
            if c not in node:
                return []
            node = node[c]

        # 이 노드 아래의 모든 $ 수집
        result = []
        self._collect(node, result)
        return result

    def _collect(self, node, result):
        if '$' in node:
            result.append(node['$'])
        for c, child in node.items():
            if c != '$':
                self._collect(child, result)


# 예시
st = SuffixTrie("banana")
print(st.search("ana"))  # [1, 3]
print(st.search("nan"))  # [2]
```

---

## 6. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐⭐ | [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/) | LeetCode | 기본 구현 |
| ⭐⭐⭐ | [전화번호 목록](https://www.acmicpc.net/problem/5052) | 백준 | 접두사 |
| ⭐⭐⭐ | [Design Search Autocomplete](https://leetcode.com/problems/design-search-autocomplete-system/) | LeetCode | 자동완성 |
| ⭐⭐⭐ | [Add and Search Word](https://leetcode.com/problems/design-add-and-search-words-data-structure/) | LeetCode | 와일드카드 |
| ⭐⭐⭐⭐ | [Maximum XOR of Two Numbers](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) | LeetCode | XOR 트라이 |
| ⭐⭐⭐⭐ | [문자열 집합](https://www.acmicpc.net/problem/14425) | 백준 | 검색 |

---

## 시간/공간 복잡도

```
┌─────────────────┬─────────────┬─────────────────────┐
│ 연산             │ 시간        │ 공간                 │
├─────────────────┼─────────────┼─────────────────────┤
│ 삽입             │ O(m)        │ O(m × 알파벳)        │
│ 검색             │ O(m)        │ -                   │
│ 접두사 검색      │ O(p)        │ -                   │
│ 자동완성         │ O(p + k)    │ O(결과 길이)         │
│ XOR 최대         │ O(log MAX)  │ O(n × log MAX)      │
└─────────────────┴─────────────┴─────────────────────┘

m = 단어 길이, p = 접두사 길이, k = 결과 개수
```

---

## 다음 단계

- [12_Graph_Basics.md](./12_Graph_Basics.md) - 그래프 기초

---

## 참고 자료

- [Trie](https://cp-algorithms.com/string/trie.html)
- [XOR Trie](https://codeforces.com/blog/entry/65408)
