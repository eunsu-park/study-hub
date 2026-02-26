"""
Cache Simulator

Demonstrates:
- Direct-mapped cache
- Set-associative cache (N-way)
- Fully-associative cache
- LRU replacement policy
- Cache performance metrics

Theory:
- Cache exploits locality (temporal and spatial) to reduce
  average memory access time.
- Address decomposition: [tag | index | offset]
- Direct-mapped: 1 way per set — fast but conflict misses
- N-way set-associative: N ways per set — fewer conflicts
- Fully-associative: all lines in one set — fewest misses, expensive

Adapted from Computer Architecture Lesson 15.
"""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheLine:
    valid: bool = False
    tag: int = 0
    dirty: bool = False
    data: int = 0  # simplified: just store the address


class Cache:
    """Configurable cache simulator.

    Parameters:
    - cache_size: total cache size in bytes
    - block_size: cache line (block) size in bytes
    - associativity: number of ways (1 = direct-mapped,
      num_lines = fully-associative)
    """

    def __init__(self, cache_size: int, block_size: int, associativity: int):
        self.cache_size = cache_size
        self.block_size = block_size
        self.associativity = associativity

        self.num_lines = cache_size // block_size
        self.num_sets = self.num_lines // associativity

        # Calculate bit widths
        self.offset_bits = (block_size - 1).bit_length()
        self.index_bits = max(0, (self.num_sets - 1).bit_length())
        self.tag_bits = 32 - self.offset_bits - self.index_bits

        # OrderedDict per set: insertion order tracks LRU position.
        # The least-recently-used entry is always at the front (oldest),
        # so eviction is O(1) via popitem(last=False).  This avoids
        # maintaining explicit timestamps or a separate LRU linked list.
        self.sets: list[OrderedDict[int, CacheLine]] = [
            OrderedDict() for _ in range(self.num_sets)
        ]

        # Statistics
        self.hits = 0
        self.misses = 0
        self.compulsory = 0
        self.conflict = 0
        self.capacity = 0
        self._all_seen: set[int] = set()  # for miss classification

    def _decompose(self, addr: int) -> tuple[int, int, int]:
        """Decompose address into (tag, index, offset)."""
        # Address layout: [tag | index | offset].  The offset selects a
        # byte within the block (exploiting spatial locality); the index
        # selects the set (direct mapping via modulo); the tag identifies
        # which block is stored (used for comparison on lookup).
        offset = addr & ((1 << self.offset_bits) - 1)
        index = (addr >> self.offset_bits) & ((1 << self.index_bits) - 1)
        tag = addr >> (self.offset_bits + self.index_bits)
        return tag, index, offset

    def access(self, addr: int) -> bool:
        """Access cache. Returns True on hit, False on miss."""
        tag, index, offset = self._decompose(addr)
        cache_set = self.sets[index]
        block_addr = addr >> self.offset_bits

        if tag in cache_set:
            # Hit — move to end to mark as most-recently-used.
            # In a direct-mapped cache (associativity=1) this is a no-op
            # since there is only one slot.  For set-associative caches
            # this maintains the LRU ordering within the set.
            cache_set.move_to_end(tag)
            self.hits += 1
            return True

        # Miss
        self.misses += 1

        # Classify miss
        if block_addr not in self._all_seen:
            self.compulsory += 1
            self._all_seen.add(block_addr)
        elif len(cache_set) < self.associativity:
            self.compulsory += 1  # still loading
        else:
            # Could be conflict or capacity
            self.conflict += 1

        # Evict if necessary — the first item in the OrderedDict is the
        # least-recently-used entry.  In a direct-mapped cache this always
        # evicts the single existing entry; in N-way the victim is chosen
        # purely by recency, not by address — which is why higher
        # associativity reduces conflict misses at the cost of more
        # comparators in hardware.
        if len(cache_set) >= self.associativity:
            cache_set.popitem(last=False)

        # Insert new line
        cache_set[tag] = CacheLine(valid=True, tag=tag, data=addr)
        return False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def config_str(self) -> str:
        if self.associativity == 1:
            mapping = "Direct-Mapped"
        elif self.associativity == self.num_lines:
            mapping = "Fully-Associative"
        else:
            mapping = f"{self.associativity}-Way Set-Associative"
        return (f"{mapping}, {self.cache_size}B cache, "
                f"{self.block_size}B blocks, {self.num_sets} sets")

    def display_state(self) -> None:
        print(f"  Cache State ({self.config_str()}):")
        for i, cache_set in enumerate(self.sets):
            if not cache_set:
                continue
            tags = [f"0x{tag:04X}" for tag in cache_set]
            print(f"    Set {i:>2}: [{', '.join(tags)}]")


def simulate(cache: Cache, addresses: list[int], label: str = "") -> dict:
    """Run access sequence through cache."""
    trace = []
    for addr in addresses:
        hit = cache.access(addr)
        trace.append((addr, hit))

    return {
        "label": label,
        "config": cache.config_str(),
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate": cache.hit_rate,
        "trace": trace,
    }


# ── Demos ───────────────────────────────────────────────────────────────

def demo_direct_mapped():
    """Demonstrate direct-mapped cache."""
    print("=" * 60)
    print("DIRECT-MAPPED CACHE")
    print("=" * 60)

    # 256B cache, 16B blocks → 16 lines, direct-mapped
    cache = Cache(cache_size=256, block_size=16, associativity=1)
    print(f"\n  Config: {cache.config_str()}")
    print(f"  Address bits: tag={cache.tag_bits}, index={cache.index_bits}, "
          f"offset={cache.offset_bits}")

    # Access pattern demonstrating conflict misses
    # Addresses 0x00 and 0x100 map to same set (index=0)
    addresses = [0x00, 0x10, 0x20, 0x00, 0x100, 0x00, 0x100, 0x00]

    print(f"\n  {'Addr':>6}  {'Tag':>5}  {'Index':>6}  {'Result':>7}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*7}")
    for addr in addresses:
        tag, index, offset = cache._decompose(addr)
        hit = cache.access(addr)
        # Undo double counting (access was already called in simulate)
        # Just show the trace
        print(f"  0x{addr:04X}  0x{tag:03X}  {index:>6}  {'HIT' if hit else 'MISS'}")

    print(f"\n  Note: 0x000 and 0x100 conflict on set 0 → thrashing")


def demo_associativity_comparison():
    """Compare different associativity levels."""
    print("\n" + "=" * 60)
    print("ASSOCIATIVITY COMPARISON")
    print("=" * 60)

    # Pattern that causes conflicts in direct-mapped
    addresses = []
    # Access 4 addresses that map to same set in direct-mapped
    for _ in range(5):
        for base in [0x000, 0x100, 0x200, 0x300]:
            addresses.append(base)

    configs = [
        (256, 16, 1, "Direct-Mapped"),
        (256, 16, 2, "2-Way"),
        (256, 16, 4, "4-Way"),
        (256, 16, 16, "Fully-Assoc"),
    ]

    print(f"\n  Access pattern: cycle through 4 conflicting addresses, 5 times")
    print(f"  Total accesses: {len(addresses)}")

    print(f"\n  {'Config':<20} {'Hits':>6} {'Misses':>8} {'Hit Rate':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10}")

    for size, block, assoc, label in configs:
        cache = Cache(cache_size=size, block_size=block, associativity=assoc)
        for addr in addresses:
            cache.access(addr)
        print(f"  {label:<20} {cache.hits:>6} {cache.misses:>8} "
              f"{cache.hit_rate:>9.1%}")


def demo_locality():
    """Demonstrate spatial and temporal locality effects."""
    print("\n" + "=" * 60)
    print("LOCALITY PATTERNS")
    print("=" * 60)

    cache_size = 256
    block_size = 16
    assoc = 4

    # Pattern 1: Sequential (good spatial locality)
    sequential = list(range(0, 256, 4))  # stride-4 through 256 bytes
    c1 = Cache(cache_size, block_size, assoc)
    for addr in sequential:
        c1.access(addr)

    # Pattern 2: Strided (partial spatial locality)
    strided = list(range(0, 1024, 64))  # stride-64
    c2 = Cache(cache_size, block_size, assoc)
    for addr in strided:
        c2.access(addr)

    # Pattern 3: Random-like (poor locality)
    import random
    random.seed(42)
    rand_addrs = [random.randint(0, 4095) for _ in range(64)]
    c3 = Cache(cache_size, block_size, assoc)
    for addr in rand_addrs:
        c3.access(addr)

    # Pattern 4: Working set with reuse (temporal locality)
    working = [0, 16, 32, 48] * 16  # 4 blocks accessed repeatedly
    c4 = Cache(cache_size, block_size, assoc)
    for addr in working:
        c4.access(addr)

    print(f"\n  Cache: {cache_size}B, {block_size}B blocks, {assoc}-way")
    print(f"\n  {'Pattern':<25} {'Accesses':>9} {'Hit Rate':>10}")
    print(f"  {'-'*25} {'-'*9} {'-'*10}")
    for name, cache, n in [
        ("Sequential (stride 4)", c1, len(sequential)),
        ("Strided (stride 64)", c2, len(strided)),
        ("Random", c3, len(rand_addrs)),
        ("Working set (reuse)", c4, len(working)),
    ]:
        print(f"  {name:<25} {n:>9} {cache.hit_rate:>9.1%}")


def demo_amat():
    """Calculate Average Memory Access Time."""
    print("\n" + "=" * 60)
    print("AVERAGE MEMORY ACCESS TIME (AMAT)")
    print("=" * 60)

    hit_time = 1       # cycles
    miss_penalty = 100  # cycles

    print(f"\n  AMAT = Hit Time + Miss Rate × Miss Penalty")
    print(f"  Hit Time = {hit_time} cycle, Miss Penalty = {miss_penalty} cycles")

    print(f"\n  {'Hit Rate':>10}  {'Miss Rate':>10}  {'AMAT':>8}  {'Speedup':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

    for hr in [0.50, 0.80, 0.90, 0.95, 0.99]:
        mr = 1 - hr
        amat = hit_time + mr * miss_penalty
        speedup = miss_penalty / amat
        print(f"  {hr:>9.0%}  {mr:>9.0%}  {amat:>7.1f}  {speedup:>9.1f}x")


if __name__ == "__main__":
    demo_direct_mapped()
    demo_associativity_comparison()
    demo_locality()
    demo_amat()
