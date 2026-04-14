"""
10 Sets and Maps
================
Demonstrates set operations, dict patterns, defaultdict,
Counter, OrderedDict, and LRU cache.
"""

from collections import defaultdict, Counter, OrderedDict


def demo_set_operations():
    a, b = {1,2,3,4}, {3,4,5,6}
    print(f"A = {a}, B = {b}")
    print(f"Union:        {a | b}")
    print(f"Intersection: {a & b}")
    print(f"Difference:   {a - b}")
    print(f"Sym diff:     {a ^ b}")


def demo_two_sum():
    def two_sum(nums, target):
        seen = {}
        for i, num in enumerate(nums):
            if target - num in seen: return [seen[target-num], i]
            seen[num] = i
        return []
    print(f"two_sum([2,7,11,15], 9) = {two_sum([2,7,11,15], 9)}")


def demo_group_anagrams():
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    groups = defaultdict(list)
    for w in words: groups[tuple(sorted(w))].append(w)
    print(f"Anagram groups: {list(groups.values())}")


def demo_counter():
    c = Counter("abracadabra")
    print(f"Counter: {c}")
    print(f"Most common 3: {c.most_common(3)}")


class LRUCache:
    def __init__(self, capacity):
        self._cache = OrderedDict(); self._cap = capacity
    def get(self, key):
        if key not in self._cache: return -1
        self._cache.move_to_end(key); return self._cache[key]
    def put(self, key, value):
        if key in self._cache: self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._cap: self._cache.popitem(last=False)


def demo_lru():
    cache = LRUCache(3)
    for k, v in [(1,"a"),(2,"b"),(3,"c")]: cache.put(k, v)
    print(f"Cache: {dict(cache._cache)}")
    cache.get(1); cache.put(4, "d")
    print(f"After access(1), put(4): {dict(cache._cache)}")


if __name__ == "__main__":
    for title, func in [("Set Operations", demo_set_operations),
        ("Two Sum", demo_two_sum), ("Group Anagrams", demo_group_anagrams),
        ("Counter", demo_counter), ("LRU Cache", demo_lru)]:
        print(f"\n{'='*50}\n  {title}\n{'='*50}"); func()
