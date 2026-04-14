# 문자열 자료구조

**이전**: [집합과 맵](./10_Sets_and_Maps.md) | **다음**: [정렬 기초](./12_Sorting_Fundamentals.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 문자열이 메모리에 저장되는 방식을 이해할 수 있다 (불변성, 인터닝)
2. 효율적인 접두사 기반 연산을 위한 트라이 (접두사 트리)를 구현할 수 있다
3. 기본 패턴 매칭 알고리즘을 적용할 수 있다 (브루트포스, 라빈-카프)
4. 효율적인 비교를 위한 문자열 해싱을 사용할 수 있다
5. 순진한 O(nm)과 효율적인 O(n+m) 매칭의 차이를 설명할 수 있다
6. 트라이를 사용한 자동완성 시스템을 구현할 수 있다
7. 적절한 자료구조를 사용하여 일반적인 문자열 문제를 해결할 수 있다

---

문자열은 단순한 문자 시퀀스처럼 보이지만 풍부한 자료구조 속성을 가지고 있습니다. 이 레슨에서는 **트라이** (접두사 트리), **문자열 해싱**, **패턴 매칭**을 포함한 문자열 처리를 위한 특화된 구조와 알고리즘을 다룹니다.

## 트라이 (접두사 트리)

**트라이**는 각 노드가 문자를 나타내고, 루트에서 노드까지의 경로가 저장된 문자열의 접두사를 형성하는 트리입니다. 저장된 단어 수와 관계없이 O(m) 조회를 가능하게 합니다 (m은 단어 길이).

```
저장된 단어: "cat", "car", "card", "care", "do", "dog"

          (root)
         /      \
        c        d
        |        |
        a        o
       / \       |
      t   r      g
         / \
        d   e
```

### 트라이 구현

```python
class TrieNode:
    """트라이의 노드."""
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0


class Trie:
    """트라이 (접두사 트리) 구현."""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """단어 삽입 -- O(m), m = len(word)."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    def search(self, word):
        """단어 존재 확인 -- O(m)."""
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix):
        """접두사로 시작하는 단어 존재 확인 -- O(m)."""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def autocomplete(self, prefix, limit=10):
        """접두사로 시작하는 최대 `limit`개의 단어를 반환."""
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
```

## 패턴 매칭

### 라빈-카프 (문자열 해싱)

롤링 해시를 사용하여 부분 문자열을 O(1)에 비교:

```python
def rabin_karp(text, pattern):
    """라빈-카프 문자열 매칭 -- 평균 O(n+m).
    
    롤링 해시를 사용하여 처음부터 해시를 다시 계산하는 것을 방지.
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []
    BASE, MOD = 256, 101
    pattern_hash = window_hash = 0
    h = pow(BASE, m - 1, MOD)
    
    for i in range(m):
        pattern_hash = (BASE * pattern_hash + ord(pattern[i])) % MOD
        window_hash = (BASE * window_hash + ord(text[i])) % MOD
    
    positions = []
    for i in range(n - m + 1):
        if pattern_hash == window_hash and text[i:i + m] == pattern:
            positions.append(i)
        if i < n - m:
            window_hash = (BASE * (window_hash - ord(text[i]) * h)
                          + ord(text[i + m])) % MOD
            if window_hash < 0:
                window_hash += MOD
    return positions
```

## 트라이 vs 해시 테이블

| 기능 | 트라이 | 해시 테이블 |
|------|--------|-----------|
| 정확한 조회 | O(m) | 평균 O(m) |
| 접두사 검색 | O(m) | O(n) 전체 스캔 |
| 자동완성 | O(m + k) | O(n) 스캔 |
| 공간 | 클 수 있음 (포인터) | 컴팩트 |
| 정렬 순회 | 자연스러움 (DFS) | 별도 정렬 필요 |

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 문자열 불변성 | 수정은 새 객체 생성; join() 사용 |
| 트라이 | O(m) 조회, 자연스러운 접두사 연산 |
| 자동완성 | 트라이 + DFS로 접두사 단어 수집 |
| 브루트포스 매칭 | O(nm) -- 모든 위치에서 비교 |
| 라빈-카프 | 롤링 해시로 평균 O(n+m) 매칭 |
| 문자열 해싱 | 롤링 해시로 O(1) 부분 문자열 비교 |

---

**다음**: [정렬 기초](./12_Sorting_Fundamentals.md) -- 고전적인 정렬 알고리즘과 분석을 배웁니다.
