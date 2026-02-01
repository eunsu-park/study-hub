# 해시 테이블 (Hash Table)

## 개요

해시 테이블은 키-값 쌍을 저장하는 자료구조로, 평균 O(1) 시간에 삽입, 삭제, 검색이 가능합니다. 코딩 인터뷰에서 가장 많이 사용되는 자료구조 중 하나입니다.

---

## 목차

1. [해시 테이블 개념](#1-해시-테이블-개념)
2. [해시 함수](#2-해시-함수)
3. [충돌 해결](#3-충돌-해결)
4. [직접 구현](#4-직접-구현)
5. [언어별 사용법](#5-언어별-사용법)
6. [활용 문제](#6-활용-문제)
7. [연습 문제](#7-연습-문제)

---

## 1. 해시 테이블 개념

### 1.1 기본 원리

```
해시 테이블 구조:

키(Key) → 해시 함수 → 인덱스 → 버킷(Bucket)

예시: 이름 → 전화번호 저장

"Alice" → hash("Alice") → 3 → bucket[3] = "010-1234-5678"
"Bob"   → hash("Bob")   → 7 → bucket[7] = "010-9876-5432"

┌─────────────────────────────────────┐
│         해시 테이블 (크기 10)        │
├─────┬───────────────────────────────┤
│  0  │                               │
│  1  │                               │
│  2  │                               │
│  3  │ "Alice" → "010-1234-5678"     │
│  4  │                               │
│  5  │                               │
│  6  │                               │
│  7  │ "Bob" → "010-9876-5432"       │
│  8  │                               │
│  9  │                               │
└─────┴───────────────────────────────┘
```

### 1.2 시간 복잡도

```
┌────────────┬─────────┬─────────┐
│ 연산       │ 평균    │ 최악     │
├────────────┼─────────┼─────────┤
│ 삽입       │ O(1)    │ O(n)    │
│ 삭제       │ O(1)    │ O(n)    │
│ 검색       │ O(1)    │ O(n)    │
└────────────┴─────────┴─────────┘

* 최악의 경우: 모든 키가 같은 버킷에 충돌
* 좋은 해시 함수와 적절한 테이블 크기로 O(1) 유지
```

### 1.3 해시맵 vs 해시셋

```
해시맵 (HashMap):
- 키-값 쌍 저장
- 예: {"apple": 3, "banana": 5}

해시셋 (HashSet):
- 키만 저장 (중복 불가)
- 예: {"apple", "banana", "cherry"}
```

---

## 2. 해시 함수

### 2.1 좋은 해시 함수의 조건

```
1. 결정적 (Deterministic)
   - 같은 입력 → 항상 같은 출력

2. 균일 분포 (Uniform Distribution)
   - 버킷에 고르게 분산

3. 빠른 계산
   - O(1) 또는 O(키 길이)
```

### 2.2 Division Method

```python
def hash_division(key, table_size):
    """나눗셈 방법: key mod table_size"""
    if isinstance(key, str):
        # 문자열 → 정수 변환
        key = sum(ord(c) for c in key)
    return key % table_size

# 예시
print(hash_division(123, 10))  # 3
print(hash_division("abc", 10))  # (97+98+99) % 10 = 4
```

### 2.3 Multiplication Method

```python
def hash_multiplication(key, table_size):
    """곱셈 방법: floor(m * (k*A mod 1))"""
    A = 0.6180339887  # 황금비의 소수 부분 (권장)
    if isinstance(key, str):
        key = sum(ord(c) for c in key)
    return int(table_size * ((key * A) % 1))

# 예시
print(hash_multiplication(123, 10))  # 7
```

### 2.4 문자열 해시 (다항식 롤링 해시)

```python
def hash_string(s, table_size, base=31):
    """
    다항식 해시: s[0]*base^(n-1) + s[1]*base^(n-2) + ... + s[n-1]
    base: 보통 31 또는 37 사용 (소수)
    """
    hash_value = 0
    for c in s:
        hash_value = hash_value * base + ord(c)
    return hash_value % table_size

# 예시
print(hash_string("hello", 1000))  # 일정한 값
```

```cpp
// C++ - 문자열 해시
long long hashString(const string& s, int tableSize, int base = 31) {
    long long hashValue = 0;
    long long power = 1;

    for (int i = s.length() - 1; i >= 0; i--) {
        hashValue = (hashValue + (s[i] - 'a' + 1) * power) % tableSize;
        power = (power * base) % tableSize;
    }

    return hashValue;
}
```

---

## 3. 충돌 해결

### 3.1 충돌이란?

```
충돌 (Collision): 서로 다른 키가 같은 인덱스에 매핑

hash("Alice") = 3
hash("Carol") = 3  ← 충돌!

해결 방법:
1. Chaining (체이닝)
2. Open Addressing (개방 주소법)
```

### 3.2 Chaining (체이닝)

```
같은 버킷에 연결 리스트로 저장

bucket[3]: "Alice" → "Carol" → "Dave" → null

┌─────┬──────────────────────────────┐
│  0  │ null                         │
│  1  │ null                         │
│  2  │ null                         │
│  3  │ "Alice" → "Carol" → "Dave"   │
│  4  │ null                         │
│  5  │ "Bob" → null                 │
└─────┴──────────────────────────────┘
```

```python
class HashTableChaining:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        index = self._hash(key)
        # 기존 키 업데이트
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        # 새 키 추가
        self.table[index].append((key, value))

    def get(self, key):
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def remove(self, key):
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return True
        return False
```

### 3.3 Open Addressing - Linear Probing

```
충돌 시 다음 빈 슬롯을 순차적으로 탐색

hash("Alice") = 3 → bucket[3]에 저장
hash("Carol") = 3 → 충돌! → bucket[4] 확인 → 저장

┌─────┬─────────┐
│  0  │         │
│  1  │         │
│  2  │         │
│  3  │ "Alice" │ ← 원래 위치
│  4  │ "Carol" │ ← 선형 탐사로 이동
│  5  │         │
└─────┴─────────┘
```

```python
class HashTableLinearProbing:
    def __init__(self, size=10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        if self.count >= self.size * 0.7:  # 부하율 70% 초과 시 리사이징 필요
            self._resize()

        index = self._hash(key)
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value  # 업데이트
                return
            index = (index + 1) % self.size

        self.keys[index] = key
        self.values[index] = value
        self.count += 1

    def get(self, key):
        index = self._hash(key)
        start = index
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size
            if index == start:  # 한 바퀴 돌았으면
                break
        return None

    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for i, key in enumerate(old_keys):
            if key is not None:
                self.put(key, old_values[i])
```

### 3.4 Open Addressing - Quadratic Probing

```
충돌 시 1, 4, 9, 16, ... (i²) 칸씩 이동

index = (hash(key) + i²) % size

장점: 클러스터링 완화
단점: 테이블 크기가 소수일 때 최적
```

### 3.5 Open Addressing - Double Hashing

```
두 번째 해시 함수로 이동 간격 결정

index = (hash1(key) + i * hash2(key)) % size

hash2(key) = 1 + (key % (size - 1))  # 0이 되면 안 됨
```

```python
class HashTableDoubleHashing:
    def __init__(self, size=11):  # 소수 권장
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size

    def _hash1(self, key):
        return hash(key) % self.size

    def _hash2(self, key):
        # 0이 되면 안 되므로 1 + ...
        return 1 + (hash(key) % (self.size - 1))

    def put(self, key, value):
        index = self._hash1(key)
        step = self._hash2(key)
        i = 0

        while self.keys[index] is not None and self.keys[index] != key:
            i += 1
            index = (self._hash1(key) + i * step) % self.size

        self.keys[index] = key
        self.values[index] = value
```

---

## 4. 직접 구현

### 4.1 완전한 해시맵 (Python)

```python
class HashMap:
    def __init__(self, initial_capacity=16, load_factor=0.75):
        self.capacity = initial_capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        if self.size >= self.capacity * self.load_factor:
            self._resize()

        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        bucket.append((key, value))
        self.size += 1

    def get(self, key, default=None):
        index = self._hash(key)
        for k, v in self.buckets[index]:
            if k == key:
                return v
        return default

    def remove(self, key):
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return True
        return False

    def contains(self, key):
        return self.get(key) is not None

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def __len__(self):
        return self.size

    def __str__(self):
        items = []
        for bucket in self.buckets:
            for k, v in bucket:
                items.append(f"{k}: {v}")
        return "{" + ", ".join(items) + "}"


# 사용 예시
hm = HashMap()
hm.put("apple", 3)
hm.put("banana", 5)
hm.put("cherry", 2)
print(hm.get("apple"))  # 3
print(hm.contains("grape"))  # False
hm.remove("banana")
print(hm)  # {apple: 3, cherry: 2}
```

### 4.2 해시셋 구현 (Python)

```python
class HashSet:
    def __init__(self):
        self.map = HashMap()

    def add(self, key):
        self.map.put(key, True)

    def remove(self, key):
        return self.map.remove(key)

    def contains(self, key):
        return self.map.contains(key)

    def __len__(self):
        return len(self.map)


# 사용 예시
hs = HashSet()
hs.add(1)
hs.add(2)
hs.add(1)  # 중복, 무시됨
print(hs.contains(1))  # True
print(len(hs))  # 2
```

### 4.3 C++ 구현

```cpp
#include <iostream>
#include <list>
#include <vector>
using namespace std;

template<typename K, typename V>
class HashMap {
private:
    struct Entry {
        K key;
        V value;
        Entry(K k, V v) : key(k), value(v) {}
    };

    vector<list<Entry>> buckets;
    int capacity;
    int count;

    int hash(K key) {
        return std::hash<K>{}(key) % capacity;
    }

    void resize() {
        vector<list<Entry>> oldBuckets = buckets;
        capacity *= 2;
        buckets.assign(capacity, list<Entry>());
        count = 0;

        for (auto& bucket : oldBuckets) {
            for (auto& entry : bucket) {
                put(entry.key, entry.value);
            }
        }
    }

public:
    HashMap(int cap = 16) : capacity(cap), count(0) {
        buckets.resize(capacity);
    }

    void put(K key, V value) {
        if (count >= capacity * 0.75) {
            resize();
        }

        int index = hash(key);
        for (auto& entry : buckets[index]) {
            if (entry.key == key) {
                entry.value = value;
                return;
            }
        }
        buckets[index].push_back(Entry(key, value));
        count++;
    }

    V* get(K key) {
        int index = hash(key);
        for (auto& entry : buckets[index]) {
            if (entry.key == key) {
                return &entry.value;
            }
        }
        return nullptr;
    }

    bool remove(K key) {
        int index = hash(key);
        auto& bucket = buckets[index];
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->key == key) {
                bucket.erase(it);
                count--;
                return true;
            }
        }
        return false;
    }

    int size() { return count; }
};
```

---

## 5. 언어별 사용법

### 5.1 Python

```python
# 딕셔너리 (해시맵)
d = {}
d["apple"] = 3
d["banana"] = 5
print(d.get("apple", 0))  # 3
print("apple" in d)  # True
del d["apple"]

# 딕셔너리 컴프리헨션
squares = {x: x**2 for x in range(5)}

# defaultdict
from collections import defaultdict
dd = defaultdict(int)  # 기본값 0
dd["a"] += 1

# Counter
from collections import Counter
c = Counter("abracadabra")
print(c.most_common(3))  # [('a', 5), ('b', 2), ('r', 2)]

# 셋 (해시셋)
s = set()
s.add(1)
s.add(2)
s.discard(1)  # 없어도 에러 안 남
print(2 in s)  # True
```

### 5.2 C++

```cpp
#include <unordered_map>
#include <unordered_set>
#include <map>  // 정렬된 맵 (레드-블랙 트리)

// unordered_map (해시맵)
unordered_map<string, int> um;
um["apple"] = 3;
um["banana"] = 5;

if (um.find("apple") != um.end()) {
    cout << um["apple"] << endl;  // 3
}

um.erase("apple");

// 반복
for (auto& [key, value] : um) {
    cout << key << ": " << value << endl;
}

// unordered_set (해시셋)
unordered_set<int> us;
us.insert(1);
us.insert(2);
us.erase(1);
cout << us.count(2) << endl;  // 1 (있음)

// 커스텀 해시 함수 (pair 등)
struct PairHash {
    size_t operator()(const pair<int, int>& p) const {
        return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
    }
};
unordered_set<pair<int, int>, PairHash> ps;
```

### 5.3 Java

```java
import java.util.*;

// HashMap
Map<String, Integer> map = new HashMap<>();
map.put("apple", 3);
map.put("banana", 5);
System.out.println(map.get("apple"));  // 3
System.out.println(map.getOrDefault("grape", 0));  // 0
map.remove("apple");

// 반복
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

// HashSet
Set<Integer> set = new HashSet<>();
set.add(1);
set.add(2);
set.remove(1);
System.out.println(set.contains(2));  // true

// computeIfAbsent (빈도수 카운팅에 유용)
map.computeIfAbsent("cherry", k -> 0);
map.merge("cherry", 1, Integer::sum);
```

---

## 6. 활용 문제

### 6.1 Two Sum

```python
def two_sum(nums, target):
    """
    합이 target인 두 인덱스 찾기
    시간: O(n), 공간: O(n)
    """
    seen = {}  # 값 → 인덱스

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []

# 예시
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

### 6.2 빈도수 카운팅

```python
def count_frequency(arr):
    """원소별 빈도수"""
    freq = {}
    for x in arr:
        freq[x] = freq.get(x, 0) + 1
    return freq

# 또는 Counter 사용
from collections import Counter
freq = Counter([1, 2, 2, 3, 3, 3])
print(freq)  # Counter({3: 3, 2: 2, 1: 1})
```

### 6.3 중복 확인

```python
def has_duplicate(nums):
    """중복 원소 존재 여부"""
    return len(nums) != len(set(nums))

def find_duplicates(nums):
    """중복 원소 찾기"""
    seen = set()
    duplicates = []
    for num in nums:
        if num in seen:
            duplicates.append(num)
        seen.add(num)
    return duplicates
```

### 6.4 아나그램 그룹화

```python
def group_anagrams(strs):
    """
    아나그램끼리 그룹화
    예: ["eat", "tea", "tan", "ate", "nat", "bat"]
    → [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for s in strs:
        # 정렬된 문자열을 키로 사용
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

### 6.5 가장 긴 연속 수열

```python
def longest_consecutive(nums):
    """
    가장 긴 연속 수열 길이
    예: [100, 4, 200, 1, 3, 2] → 4 (1, 2, 3, 4)
    시간: O(n)
    """
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # 수열의 시작점만 탐색 (num-1이 없어야 시작점)
        if num - 1 not in num_set:
            current = num
            length = 1

            while current + 1 in num_set:
                current += 1
                length += 1

            max_length = max(max_length, length)

    return max_length
```

### 6.6 Subarray Sum Equals K

```python
def subarray_sum(nums, k):
    """
    합이 k인 부분 배열 개수
    프리픽스 합 + 해시맵 활용
    시간: O(n)
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # 합 → 등장 횟수

    for num in nums:
        prefix_sum += num

        # prefix_sum - k가 이전에 나왔다면
        # 그 지점부터 현재까지 합이 k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]

        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count

# 예시
print(subarray_sum([1, 1, 1], 2))  # 2
print(subarray_sum([1, 2, 3], 3))  # 2
```

### 6.7 LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used 캐시
    get/put O(1)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        # 최근 사용으로 이동
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if len(self.cache) > self.capacity:
            # 가장 오래된 항목 제거
            self.cache.popitem(last=False)


# 사용 예시
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)  # 2 제거
print(cache.get(2))  # -1
```

---

## 7. 연습 문제

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [Two Sum](https://leetcode.com/problems/two-sum/) | LeetCode | 기본 해시맵 |
| ⭐ | [Valid Anagram](https://leetcode.com/problems/valid-anagram/) | LeetCode | 빈도수 |
| ⭐⭐ | [Group Anagrams](https://leetcode.com/problems/group-anagrams/) | LeetCode | 키 설계 |
| ⭐⭐ | [숫자 카드 2](https://www.acmicpc.net/problem/10816) | 백준 | 빈도수 |
| ⭐⭐ | [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) | LeetCode | 프리픽스 합 |
| ⭐⭐⭐ | [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) | LeetCode | 해시셋 |
| ⭐⭐⭐ | [LRU Cache](https://leetcode.com/problems/lru-cache/) | LeetCode | OrderedDict |

---

## 해시 테이블 복잡도 정리

```
┌─────────────────┬─────────────────┬─────────────────┐
│ 자료구조         │ 평균            │ 최악             │
├─────────────────┼─────────────────┼─────────────────┤
│ 해시맵 삽입      │ O(1)           │ O(n)            │
│ 해시맵 검색      │ O(1)           │ O(n)            │
│ 해시맵 삭제      │ O(1)           │ O(n)            │
├─────────────────┼─────────────────┼─────────────────┤
│ 해시셋 추가      │ O(1)           │ O(n)            │
│ 해시셋 검색      │ O(1)           │ O(n)            │
│ 해시셋 삭제      │ O(1)           │ O(n)            │
└─────────────────┴─────────────────┴─────────────────┘

공간 복잡도: O(n)
```

---

## 다음 단계

- [05_Sorting_Algorithms.md](./05_Sorting_Algorithms.md) - 정렬 알고리즘

---

## 참고 자료

- [Hash Table Visualization](https://www.cs.usfca.edu/~galles/visualization/OpenHash.html)
- Introduction to Algorithms (CLRS) - Chapter 11
