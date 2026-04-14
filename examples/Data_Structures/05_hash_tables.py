"""
05 Hash Tables
==============
Demonstrates hash table with chaining, Python dict features,
defaultdict, Counter, and custom hashable objects.
"""

from collections import defaultdict, Counter


class HashTableChaining:
    """Hash table with separate chaining."""

    def __init__(self, capacity=8):
        self._capacity = capacity
        self._size = 0
        self._buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        return hash(key) % self._capacity

    def put(self, key, value):
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / self._capacity > 0.75:
            self._resize(self._capacity * 2)

    def get(self, key):
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def __len__(self):
        return self._size

    def _resize(self, new_cap):
        old = self._buckets
        self._capacity = new_cap
        self._buckets = [[] for _ in range(new_cap)]
        self._size = 0
        for bucket in old:
            for k, v in bucket:
                self.put(k, v)


def demo_hash_table():
    """Demonstrate custom hash table."""
    ht = HashTableChaining()
    data = [("alice", 30), ("bob", 25), ("charlie", 35), ("dave", 28)]
    for name, age in data:
        ht.put(name, age)
    print(f"Size: {len(ht)}")
    for name, _ in data:
        print(f"  ht['{name}'] = {ht.get(name)}")
    print(f"  'alice' in ht: {'alice' in ht}")
    print(f"  'eve' in ht: {'eve' in ht}")


def demo_dict_features():
    """Demonstrate Python dict features."""
    words = ["apple", "banana", "avocado", "blueberry", "cherry"]
    groups = defaultdict(list)
    for w in words:
        groups[w[0]].append(w)
    print(f"Groups: {dict(groups)}")

    text = "abracadabra"
    c = Counter(text)
    print(f"Counter('{text}'): {c}")
    print(f"Most common 3: {c.most_common(3)}")


def demo_custom_hashable():
    """Demonstrate hashing custom objects."""
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def __hash__(self):
            return hash((self.x, self.y))
        def __eq__(self, other):
            return isinstance(other, Point) and self.x == other.x and self.y == other.y
        def __repr__(self):
            return f"Point({self.x}, {self.y})"

    distances = {Point(0, 0): 0.0, Point(3, 4): 5.0, Point(1, 1): 1.414}
    for pt, dist in distances.items():
        print(f"  {pt}: distance = {dist}")
    print(f"  Point(3,4) in dict: {Point(3, 4) in distances}")


if __name__ == "__main__":
    for title, func in [
        ("Custom Hash Table", demo_hash_table),
        ("Python Dict Features", demo_dict_features),
        ("Custom Hashable", demo_custom_hashable),
    ]:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
