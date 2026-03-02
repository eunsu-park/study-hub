"""
Hash Table
Hash Table Implementation

Implements hash functions and collision resolution methods.
"""

from typing import Any, List, Optional, Tuple
from collections import OrderedDict


# =============================================================================
# 1. Hash Functions
# =============================================================================

def hash_division(key: int, table_size: int) -> int:
    """Division hash function - O(1)"""
    return key % table_size


def hash_multiplication(key: int, table_size: int, A: float = 0.6180339887) -> int:
    """Multiplication hash function (Knuth's suggested A value) - O(1)"""
    return int(table_size * ((key * A) % 1))


def hash_string(s: str, table_size: int) -> int:
    """String hash function (polynomial rolling hash) - O(n)"""
    hash_val = 0
    base = 31
    for char in s:
        hash_val = (hash_val * base + ord(char)) % table_size
    return hash_val


# =============================================================================
# 2. Chaining
# =============================================================================

class HashTableChaining:
    """Hash table with chaining collision resolution"""

    def __init__(self, size: int = 10):
        self.size = size
        self.table: List[List[Tuple[Any, Any]]] = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key: Any) -> int:
        """Hash function"""
        if isinstance(key, int):
            return key % self.size
        return hash(key) % self.size

    def put(self, key: Any, value: Any) -> None:
        """Insert key-value pair - Average O(1), Worst O(n)"""
        index = self._hash(key)

        # Update existing key
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return

        # Add new key
        self.table[index].append((key, value))
        self.count += 1

    def get(self, key: Any) -> Optional[Any]:
        """Look up value by key - Average O(1), Worst O(n)"""
        index = self._hash(key)

        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def remove(self, key: Any) -> bool:
        """Delete key-value pair - Average O(1), Worst O(n)"""
        index = self._hash(key)

        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                self.count -= 1
                return True
        return False

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None

    def load_factor(self) -> float:
        return self.count / self.size


# =============================================================================
# 3. Open Addressing - Linear Probing
# =============================================================================

class HashTableLinearProbing:
    """Hash table with linear probing collision resolution"""

    DELETED = object()  # Deletion marker

    def __init__(self, size: int = 10):
        self.size = size
        self.keys: List[Any] = [None] * size
        self.values: List[Any] = [None] * size
        self.count = 0

    def _hash(self, key: Any) -> int:
        if isinstance(key, int):
            return key % self.size
        return hash(key) % self.size

    def _probe(self, key: Any, for_insert: bool = False) -> int:
        """Find slot using linear probing"""
        index = self._hash(key)
        first_deleted = -1

        for i in range(self.size):
            probe_index = (index + i) % self.size

            if self.keys[probe_index] is None:
                if for_insert and first_deleted != -1:
                    return first_deleted
                return probe_index

            if self.keys[probe_index] is self.DELETED:
                if first_deleted == -1:
                    first_deleted = probe_index
                continue

            if self.keys[probe_index] == key:
                return probe_index

        return first_deleted if first_deleted != -1 else -1

    def put(self, key: Any, value: Any) -> bool:
        """Insert key-value pair - Average O(1), Worst O(n)"""
        if self.count >= self.size * 0.7:  # Load factor 70%
            self._resize()

        index = self._probe(key, for_insert=True)
        if index == -1:
            return False

        if self.keys[index] is None or self.keys[index] is self.DELETED:
            self.count += 1

        self.keys[index] = key
        self.values[index] = value
        return True

    def get(self, key: Any) -> Optional[Any]:
        """Look up value by key - Average O(1), Worst O(n)"""
        index = self._probe(key)

        if index != -1 and self.keys[index] not in (None, self.DELETED):
            return self.values[index]
        return None

    def remove(self, key: Any) -> bool:
        """Delete key-value pair (using deletion marker) - Average O(1), Worst O(n)"""
        index = self._probe(key)

        if index != -1 and self.keys[index] not in (None, self.DELETED):
            self.keys[index] = self.DELETED
            self.values[index] = None
            self.count -= 1
            return True
        return False

    def _resize(self) -> None:
        """Double the table size"""
        old_keys = self.keys
        old_values = self.values

        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0

        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)


# =============================================================================
# 4. Open Addressing - Double Hashing
# =============================================================================

class HashTableDoubleHashing:
    """Hash table with double hashing collision resolution"""

    DELETED = object()

    def __init__(self, size: int = 11):  # Prime number recommended
        self.size = size
        self.keys: List[Any] = [None] * size
        self.values: List[Any] = [None] * size
        self.count = 0

    def _hash1(self, key: Any) -> int:
        if isinstance(key, int):
            return key % self.size
        return hash(key) % self.size

    def _hash2(self, key: Any) -> int:
        """Second hash function (returns non-zero value)"""
        if isinstance(key, int):
            return 7 - (key % 7)  # 7 is a prime smaller than size
        return 7 - (hash(key) % 7)

    def _probe(self, key: Any, for_insert: bool = False) -> int:
        """Find slot using double hashing"""
        h1 = self._hash1(key)
        h2 = self._hash2(key)
        first_deleted = -1

        for i in range(self.size):
            index = (h1 + i * h2) % self.size

            if self.keys[index] is None:
                if for_insert and first_deleted != -1:
                    return first_deleted
                return index

            if self.keys[index] is self.DELETED:
                if first_deleted == -1:
                    first_deleted = index
                continue

            if self.keys[index] == key:
                return index

        return first_deleted if first_deleted != -1 else -1

    def put(self, key: Any, value: Any) -> bool:
        index = self._probe(key, for_insert=True)
        if index == -1:
            return False

        if self.keys[index] is None or self.keys[index] is self.DELETED:
            self.count += 1

        self.keys[index] = key
        self.values[index] = value
        return True

    def get(self, key: Any) -> Optional[Any]:
        index = self._probe(key)
        if index != -1 and self.keys[index] not in (None, self.DELETED):
            return self.values[index]
        return None

    def remove(self, key: Any) -> bool:
        index = self._probe(key)
        if index != -1 and self.keys[index] not in (None, self.DELETED):
            self.keys[index] = self.DELETED
            self.values[index] = None
            self.count -= 1
            return True
        return False


# =============================================================================
# 5. LRU Cache (Least Recently Used Cache)
# =============================================================================

class LRUCache:
    """
    LRU Cache Implementation
    O(1) get/put using OrderedDict
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        """Look up key and move to most recently used - O(1)"""
        if key not in self.cache:
            return None

        # Move to most recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """Insert key-value pair - O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove the oldest entry
                self.cache.popitem(last=False)

        self.cache[key] = value

    def __str__(self) -> str:
        return str(list(self.cache.items()))


# =============================================================================
# 6. Practical Problem: Two Sum
# =============================================================================

def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Find indices of two numbers that sum to target
    Time Complexity: O(n), Space Complexity: O(n)
    """
    seen = {}  # value -> index

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i

    return None


# =============================================================================
# 7. Practical Problem: Frequency Count
# =============================================================================

def frequency_count(arr: List[Any]) -> dict:
    """
    Count element frequencies in an array
    Time Complexity: O(n), Space Complexity: O(k) (k = number of unique elements)
    """
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq


def most_frequent(arr: List[Any]) -> Optional[Any]:
    """Find the most frequent element"""
    freq = frequency_count(arr)
    if not freq:
        return None
    return max(freq, key=freq.get)


# =============================================================================
# 8. Practical Problem: Subarray Sum Equals K
# =============================================================================

def subarray_sum_count(nums: List[int], k: int) -> int:
    """
    Count contiguous subarrays with sum equal to k
    Uses prefix sum + hash map
    Time Complexity: O(n), Space Complexity: O(n)
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # prefix sum -> occurrence count

    for num in nums:
        prefix_sum += num

        # If prefix_sum - k existed before, the subarray between them has sum k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]

        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count


# =============================================================================
# 9. Practical Problem: Group Anagrams
# =============================================================================

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    Group anagrams together
    Time Complexity: O(n * k log k), n=number of strings, k=max string length
    """
    groups = {}

    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    return list(groups.values())


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Hash Table Examples")
    print("=" * 60)

    # 1. Hash Function Tests
    print("\n[1] Hash Functions")
    print(f"    hash_division(42, 10) = {hash_division(42, 10)}")
    print(f"    hash_multiplication(42, 10) = {hash_multiplication(42, 10)}")
    print(f"    hash_string('hello', 10) = {hash_string('hello', 10)}")

    # 2. Chaining Test
    print("\n[2] Chaining Method")
    ht_chain = HashTableChaining(5)
    for i, name in enumerate(['Alice', 'Bob', 'Charlie', 'David', 'Eve']):
        ht_chain.put(name, i * 10)
    print(f"    get('Charlie') = {ht_chain.get('Charlie')}")
    print(f"    'Bob' in table = {'Bob' in ht_chain}")
    print(f"    load_factor = {ht_chain.load_factor():.2f}")

    # 3. Linear Probing Test
    print("\n[3] Linear Probing Method")
    ht_linear = HashTableLinearProbing(10)
    for i in range(7):
        ht_linear.put(i * 5, f"value_{i}")
    print(f"    get(10) = {ht_linear.get(10)}")
    ht_linear.remove(10)
    print(f"    get(10) after remove = {ht_linear.get(10)}")

    # 4. LRU Cache Test
    print("\n[4] LRU Cache")
    cache = LRUCache(3)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    print(f"    Cache: {cache}")
    cache.get('a')  # Move 'a' to most recent
    cache.put('d', 4)  # 'b' gets evicted
    print(f"    After get('a'), put('d'): {cache}")

    # 5. Two Sum
    print("\n[5] Two Sum")
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"    nums = {nums}, target = {target}")
    print(f"    Result: indices {result}")

    # 6. Frequency Count
    print("\n[6] Frequency Count")
    arr = ['a', 'b', 'a', 'c', 'a', 'b']
    freq = frequency_count(arr)
    print(f"    Array: {arr}")
    print(f"    Frequency: {freq}")
    print(f"    Most frequent: {most_frequent(arr)}")

    # 7. Subarray Sum
    print("\n[7] Subarray Sum Equals K Count")
    nums = [1, 1, 1]
    k = 2
    count = subarray_sum_count(nums, k)
    print(f"    nums = {nums}, k = {k}")
    print(f"    Result: {count} subarrays")

    # 8. Group Anagrams
    print("\n[8] Group Anagrams")
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    groups = group_anagrams(strs)
    print(f"    Input: {strs}")
    print(f"    Groups: {groups}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
