# Hash Table

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how a hash function maps keys to bucket indices and why O(1) average-case lookup is achievable
2. Compare chaining and open addressing as collision resolution strategies, including their trade-offs in load factor and memory usage
3. Implement a hash table from scratch in C or Python, including insert, delete, and search operations
4. Use built-in hash map structures (Python dict, C++ unordered_map) to solve two-sum, grouping, and frequency-counting problems
5. Identify scenarios where a hash table's worst-case O(n) behavior occurs and describe mitigations such as resizing and better hash functions
6. Design solutions to common interview patterns (duplicate detection, subarray sum, grouping by key) using hash tables

---

## Overview

A hash table is a data structure that stores key-value pairs, enabling insertion, deletion, and search operations in O(1) average time. It is one of the most commonly used data structures in coding interviews.

---

## Table of Contents

1. [Hash Table Concepts](#1-hash-table-concepts)
2. [Hash Functions](#2-hash-functions)
3. [Collision Resolution](#3-collision-resolution)
4. [Direct Implementation](#4-direct-implementation)
5. [Language-Specific Usage](#5-language-specific-usage)
6. [Practical Problems](#6-practical-problems)
7. [Practice Problems](#7-practice-problems)

---

## 1. Hash Table Concepts

### 1.1 Basic Principles

```
Hash Table Structure:

Key → Hash Function → Index → Bucket

Example: Storing name → phone number

"Alice" → hash("Alice") → 3 → bucket[3] = "010-1234-5678"
"Bob"   → hash("Bob")   → 7 → bucket[7] = "010-9876-5432"

┌─────────────────────────────────────┐
│       Hash Table (Size 10)          │
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

### 1.2 Time Complexity

```
┌────────────┬─────────┬─────────┐
│ Operation  │ Average │ Worst   │
├────────────┼─────────┼─────────┤
│ Insert     │ O(1)    │ O(n)    │
│ Delete     │ O(1)    │ O(n)    │
│ Search     │ O(1)    │ O(n)    │
└────────────┴─────────┴─────────┘

* Worst case: All keys collide in the same bucket
* Maintain O(1) with good hash function and appropriate table size
```

### 1.3 HashMap vs HashSet

```
HashMap:
- Stores key-value pairs
- Example: {"apple": 3, "banana": 5}

HashSet:
- Stores keys only (no duplicates)
- Example: {"apple", "banana", "cherry"}
```

---

## 2. Hash Functions

### 2.1 Properties of a Good Hash Function

```
1. Deterministic
   - Same input → always same output

2. Uniform Distribution
   - Evenly distributed across buckets

3. Fast Computation
   - O(1) or O(key length)
```

### 2.2 Division Method

```python
def hash_division(key, table_size):
    """Division method: key mod table_size"""
    if isinstance(key, str):
        # Convert string → integer
        key = sum(ord(c) for c in key)
    return key % table_size

# Examples
print(hash_division(123, 10))  # 3
print(hash_division("abc", 10))  # (97+98+99) % 10 = 4
```

### 2.3 Multiplication Method

```python
def hash_multiplication(key, table_size):
    """Multiplication method: floor(m * (k*A mod 1))"""
    # The golden ratio's fractional part is theoretically optimal for minimizing
    # clustering — it is maximally irrational, so multiples of it are as evenly
    # spread around the unit interval as possible
    A = 0.6180339887  # Fractional part of golden ratio (recommended)
    if isinstance(key, str):
        key = sum(ord(c) for c in key)
    # Multiply key by A, take only the fractional part (mod 1), then scale to
    # [0, table_size) — this makes the hash independent of table_size, which is
    # an advantage over the division method when the table is frequently resized
    return int(table_size * ((key * A) % 1))

# Example
print(hash_multiplication(123, 10))  # 7
```

### 2.4 String Hash (Polynomial Rolling Hash)

```python
def hash_string(s, table_size, base=31):
    """
    Polynomial hash: s[0]*base^(n-1) + s[1]*base^(n-2) + ... + s[n-1]
    base: typically 31 or 37 (prime)
    """
    hash_value = 0
    for c in s:
        hash_value = hash_value * base + ord(c)
    return hash_value % table_size

# Example
print(hash_string("hello", 1000))  # consistent value
```

```cpp
// C++ - String Hash
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

## 3. Collision Resolution

### 3.1 What is Collision?

```
Collision: Different keys mapping to the same index

hash("Alice") = 3
hash("Carol") = 3  ← Collision!

Solutions:
1. Chaining
2. Open Addressing
```

### 3.2 Chaining

```
Store in linked list at the same bucket

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
        # Update existing key
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        # Add new key
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
Search for next empty slot sequentially on collision

hash("Alice") = 3 → store in bucket[3]
hash("Carol") = 3 → collision! → check bucket[4] → store

┌─────┬─────────┐
│  0  │         │
│  1  │         │
│  2  │         │
│  3  │ "Alice" │ ← Original position
│  4  │ "Carol" │ ← Moved by linear probing
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
        # Resize at 70% load factor — beyond this threshold, probe chains grow
        # long enough that average O(1) performance degrades toward O(n)
        if self.count >= self.size * 0.7:
            self._resize()

        index = self._hash(key)
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value  # Update existing key in-place
                return
            # Wrap around with modulo so probing treats the table as circular —
            # without this, we would fall off the end and miss empty slots at index 0
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
            # Stop if we've wrapped all the way around — the key does not exist
            # (an absent key is confirmed only by a None slot or a full circle)
            if index == start:  # Made full circle
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
Move by 1, 4, 9, 16, ... (i²) steps on collision

index = (hash(key) + i²) % size

Advantage: Reduces clustering
Disadvantage: Optimal when table size is prime
```

### 3.5 Open Addressing - Double Hashing

```
Use second hash function to determine step size

index = (hash1(key) + i * hash2(key)) % size

hash2(key) = 1 + (key % (size - 1))  # Must not be 0
```

```python
class HashTableDoubleHashing:
    def __init__(self, size=11):  # Prime number recommended
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size

    def _hash1(self, key):
        return hash(key) % self.size

    def _hash2(self, key):
        # Must not be 0, so 1 + ...
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

## 4. Direct Implementation

### 4.1 Complete HashMap (Python)

```python
class HashMap:
    def __init__(self, initial_capacity=16, load_factor=0.75):
        # Power-of-two initial capacity (16) is conventional — modern hash maps
        # use bitwise AND instead of modulo for speed when capacity is a power of 2
        self.capacity = initial_capacity
        # 0.75 load factor is the empirically found sweet spot: below this,
        # memory is wasted; above this, collision chains degrade average O(1)
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        if self.size >= self.capacity * self.load_factor:
            self._resize()  # Double capacity to restore low load factor

        index = self._hash(key)
        bucket = self.buckets[index]

        # Scan the chain first — upsert semantics mean an existing key must be
        # updated, not duplicated (duplicate keys would break get() correctness)
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
        self.capacity *= 2  # Double to amortize resize cost over future inserts
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

        # Re-hash all existing entries — bucket indices change because
        # hash(key) % capacity depends on the new (larger) capacity
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


# Usage example
hm = HashMap()
hm.put("apple", 3)
hm.put("banana", 5)
hm.put("cherry", 2)
print(hm.get("apple"))  # 3
print(hm.contains("grape"))  # False
hm.remove("banana")
print(hm)  # {apple: 3, cherry: 2}
```

### 4.2 HashSet Implementation (Python)

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


# Usage example
hs = HashSet()
hs.add(1)
hs.add(2)
hs.add(1)  # Duplicate, ignored
print(hs.contains(1))  # True
print(len(hs))  # 2
```

### 4.3 C++ Implementation

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

## 5. Language-Specific Usage

### 5.1 Python

```python
# Dictionary (HashMap)
d = {}
d["apple"] = 3
d["banana"] = 5
print(d.get("apple", 0))  # 3
print("apple" in d)  # True
del d["apple"]

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}

# defaultdict
from collections import defaultdict
dd = defaultdict(int)  # Default value 0
dd["a"] += 1

# Counter
from collections import Counter
c = Counter("abracadabra")
print(c.most_common(3))  # [('a', 5), ('b', 2), ('r', 2)]

# Set (HashSet)
s = set()
s.add(1)
s.add(2)
s.discard(1)  # No error if not found
print(2 in s)  # True
```

### 5.2 C++

```cpp
#include <unordered_map>
#include <unordered_set>
#include <map>  // Sorted map (Red-Black tree)

// unordered_map (HashMap)
unordered_map<string, int> um;
um["apple"] = 3;
um["banana"] = 5;

if (um.find("apple") != um.end()) {
    cout << um["apple"] << endl;  // 3
}

um.erase("apple");

// Iteration
for (auto& [key, value] : um) {
    cout << key << ": " << value << endl;
}

// unordered_set (HashSet)
unordered_set<int> us;
us.insert(1);
us.insert(2);
us.erase(1);
cout << us.count(2) << endl;  // 1 (exists)

// Custom hash function (for pair, etc.)
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

// Iteration
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

// HashSet
Set<Integer> set = new HashSet<>();
set.add(1);
set.add(2);
set.remove(1);
System.out.println(set.contains(2));  // true

// computeIfAbsent (useful for frequency counting)
map.computeIfAbsent("cherry", k -> 0);
map.merge("cherry", 1, Integer::sum);
```

---

## 6. Practical Problems

### 6.1 Two Sum

```python
def two_sum(nums, target):
    """
    Find two indices that sum to target
    Time: O(n), Space: O(n)
    """
    seen = {}  # value → index

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []

# Example
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

### 6.2 Frequency Counting

```python
def count_frequency(arr):
    """Element frequency count"""
    freq = {}
    for x in arr:
        freq[x] = freq.get(x, 0) + 1
    return freq

# Or use Counter
from collections import Counter
freq = Counter([1, 2, 2, 3, 3, 3])
print(freq)  # Counter({3: 3, 2: 2, 1: 1})
```

### 6.3 Duplicate Detection

```python
def has_duplicate(nums):
    """Check if duplicate exists"""
    return len(nums) != len(set(nums))

def find_duplicates(nums):
    """Find duplicate elements"""
    seen = set()
    duplicates = []
    for num in nums:
        if num in seen:
            duplicates.append(num)
        seen.add(num)
    return duplicates
```

### 6.4 Group Anagrams

```python
def group_anagrams(strs):
    """
    Group anagrams together
    Example: ["eat", "tea", "tan", "ate", "nat", "bat"]
    → [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

### 6.5 Longest Consecutive Sequence

```python
def longest_consecutive(nums):
    """
    Find longest consecutive sequence length
    Example: [100, 4, 200, 1, 3, 2] → 4 (1, 2, 3, 4)
    Time: O(n)
    """
    # Convert to set for O(1) membership checks — sorting would be O(n log n)
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # Only start a new sequence from its lowest element; if num-1 exists,
        # this num is not a starting point — skipping non-starts avoids
        # re-scanning every element of every chain, keeping overall cost O(n)
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
    Count subarrays with sum equal to k
    Using prefix sum + hashmap
    Time: O(n)
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # sum → occurrence count

    for num in nums:
        prefix_sum += num

        # If prefix_sum - k appeared before,
        # sum from that point to now equals k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]

        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count

# Examples
print(subarray_sum([1, 1, 1], 2))  # 2
print(subarray_sum([1, 2, 3], 3))  # 2
```

### 6.7 LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used cache
    get/put O(1)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # OrderedDict remembers insertion order and supports O(1) move_to_end —
        # combining a dict (O(1) lookup) with order tracking achieves O(1) get/put
        # without the manual doubly-linked list + hash map that LRU normally needs
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        # Mark as most recently used by moving to the "end" of the ordered dict —
        # the "front" (oldest end) is what we evict, so access promotes to the back
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # Update and refresh recency in one step — inserting without moving
            # would leave a stale position in the order
            self.cache.move_to_end(key)
        self.cache[key] = value

        if len(self.cache) > self.capacity:
            # Evict the least recently used item at the front (last=False)
            self.cache.popitem(last=False)


# Usage example
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)  # Removes key 2
print(cache.get(2))  # -1
```

---

## 7. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [Two Sum](https://leetcode.com/problems/two-sum/) | LeetCode | Basic HashMap |
| ⭐ | [Valid Anagram](https://leetcode.com/problems/valid-anagram/) | LeetCode | Frequency |
| ⭐⭐ | [Group Anagrams](https://leetcode.com/problems/group-anagrams/) | LeetCode | Key Design |
| ⭐⭐ | [Number Cards 2](https://www.acmicpc.net/problem/10816) | Baekjoon | Frequency |
| ⭐⭐ | [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) | LeetCode | Prefix Sum |
| ⭐⭐⭐ | [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) | LeetCode | HashSet |
| ⭐⭐⭐ | [LRU Cache](https://leetcode.com/problems/lru-cache/) | LeetCode | OrderedDict |

---

## Hash Table Complexity Summary

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Data Structure  │ Average         │ Worst           │
├─────────────────┼─────────────────┼─────────────────┤
│ HashMap Insert  │ O(1)           │ O(n)            │
│ HashMap Search  │ O(1)           │ O(n)            │
│ HashMap Delete  │ O(1)           │ O(n)            │
├─────────────────┼─────────────────┼─────────────────┤
│ HashSet Add     │ O(1)           │ O(n)            │
│ HashSet Search  │ O(1)           │ O(n)            │
│ HashSet Delete  │ O(1)           │ O(n)            │
└─────────────────┴─────────────────┴─────────────────┘

Space Complexity: O(n)
```

---

## Next Steps

- [05_Sorting_Algorithms.md](./05_Sorting_Algorithms.md) - Sorting Algorithms

---

## References

- [Hash Table Visualization](https://www.cs.usfca.edu/~galles/visualization/OpenHash.html)
- Introduction to Algorithms (CLRS) - Chapter 11
