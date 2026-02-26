"""
TLB and Page Table Walk Simulator

Demonstrates:
- TLB (Translation Lookaside Buffer) with LRU eviction
- Multi-level page table walk
- Address translation pipeline
- TLB performance with different access patterns

Theory:
- Virtual address → TLB lookup → hit: get PFN directly,
  miss: walk page table, update TLB.
- Page table walk: traverse hierarchical page tables in memory
  (each level requires one memory access).
- Effective memory access time depends heavily on TLB hit rate:
  EMAT = TLB_hit_rate × (TLB_time + mem_time)
       + TLB_miss_rate × (TLB_time + walk_time + mem_time)

Adapted from Computer Architecture Lesson 16.
"""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class PageTableEntry:
    frame: int = -1
    valid: bool = False
    accessed: bool = False
    dirty: bool = False


class TLB:
    """Translation Lookaside Buffer with LRU replacement."""

    def __init__(self, size: int = 16):
        self.size = size
        self.entries: OrderedDict[int, int] = OrderedDict()  # VPN → PFN
        self.hits = 0
        self.misses = 0

    def lookup(self, vpn: int) -> int | None:
        if vpn in self.entries:
            self.hits += 1
            # Move to end = mark as most-recently-used.  TLB is fully
            # associative (any VPN can be in any slot), so LRU is the
            # standard replacement policy — it keeps the working set's
            # pages resident as long as the TLB is large enough.
            self.entries.move_to_end(vpn)
            return self.entries[vpn]
        self.misses += 1
        return None

    def insert(self, vpn: int, pfn: int) -> int | None:
        evicted = None
        if vpn in self.entries:
            self.entries.move_to_end(vpn)
            self.entries[vpn] = pfn
        else:
            if len(self.entries) >= self.size:
                evicted_vpn, _ = self.entries.popitem(last=False)
                evicted = evicted_vpn
            self.entries[vpn] = pfn
        return evicted

    def flush(self) -> None:
        # TLB flush is required on context switch (process change) because
        # different processes have different page tables.  Without a flush,
        # stale VPN-to-PFN mappings from the old process would produce
        # incorrect translations for the new process.
        self.entries.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TwoLevelPageTable:
    """Two-level page table for address translation."""

    def __init__(self, l1_bits: int = 10, l2_bits: int = 10, offset_bits: int = 12):
        self.l1_bits = l1_bits
        self.l2_bits = l2_bits
        self.offset_bits = offset_bits
        self.l1_size = 1 << l1_bits
        self.l2_size = 1 << l2_bits

        self.l1_table: list[list[PageTableEntry] | None] = [None] * self.l1_size
        self.walk_count = 0
        self.mem_accesses_for_walks = 0

    def map(self, vpn: int, pfn: int) -> None:
        l1_idx = vpn >> self.l2_bits
        l2_idx = vpn & (self.l2_size - 1)

        if self.l1_table[l1_idx] is None:
            self.l1_table[l1_idx] = [PageTableEntry() for _ in range(self.l2_size)]

        self.l1_table[l1_idx][l2_idx] = PageTableEntry(frame=pfn, valid=True)

    def walk(self, vpn: int) -> int | None:
        """Walk the page table. Returns PFN or None (page fault)."""
        self.walk_count += 1

        # Split VPN into two indices for the two-level table.  Using a
        # hierarchical structure avoids allocating a single flat table
        # for the entire virtual address space — only populated regions
        # have L2 tables allocated (sparse coverage).
        l1_idx = vpn >> self.l2_bits
        l2_idx = vpn & (self.l2_size - 1)

        # Level 1 access (1 memory access)
        self.mem_accesses_for_walks += 1
        if self.l1_table[l1_idx] is None:
            return None  # page fault

        # Level 2 access (1 memory access)
        self.mem_accesses_for_walks += 1
        entry = self.l1_table[l1_idx][l2_idx]
        if not entry.valid:
            return None  # page fault

        entry.accessed = True
        return entry.frame


class AddressTranslator:
    """Complete address translation with TLB and page table."""

    def __init__(self, tlb_size: int = 16, page_size: int = 4096):
        self.page_size = page_size
        self.offset_bits = (page_size - 1).bit_length()
        l2_bits = 10
        l1_bits = 10

        self.tlb = TLB(size=tlb_size)
        self.page_table = TwoLevelPageTable(l1_bits, l2_bits, self.offset_bits)
        self.page_faults = 0

    def setup_mappings(self, mappings: dict[int, int]) -> None:
        """Set up VPN → PFN mappings."""
        for vpn, pfn in mappings.items():
            self.page_table.map(vpn, pfn)

    def translate(self, virtual_addr: int) -> tuple[int | None, str]:
        """Translate virtual address. Returns (physical_addr, event_type)."""
        vpn = virtual_addr >> self.offset_bits
        offset = virtual_addr & (self.page_size - 1)

        # Step 1: TLB lookup — on hit, translation costs just ~1 cycle
        # (the TLB is a small, fast SRAM near the CPU).
        pfn = self.tlb.lookup(vpn)
        if pfn is not None:
            physical = (pfn << self.offset_bits) | offset
            return physical, "TLB_HIT"

        # Step 2: TLB miss — must walk the page table in memory.
        # Each level costs one memory access (~100 cycles), so a 2-level
        # walk is ~200 cycles.  This is why TLB hit rate is critical:
        # even a small miss rate causes a large performance penalty.
        pfn = self.page_table.walk(vpn)
        if pfn is None:
            self.page_faults += 1
            return None, "PAGE_FAULT"

        # Step 3: Fill TLB with the discovered mapping so future
        # accesses to the same page are fast (temporal locality).
        self.tlb.insert(vpn, pfn)
        physical = (pfn << self.offset_bits) | offset
        return physical, "TLB_MISS"


# ── Demos ───────────────────────────────────────────────────────────────

def demo_address_translation():
    """Step-by-step address translation."""
    print("=" * 60)
    print("ADDRESS TRANSLATION WITH TLB")
    print("=" * 60)

    translator = AddressTranslator(tlb_size=4, page_size=4096)

    # Map some pages
    mappings = {0: 5, 1: 8, 2: 1, 3: 12, 4: 3, 5: 7, 10: 20}
    translator.setup_mappings(mappings)

    # Access sequence
    addresses = [
        0x0000,  # VPN 0 → TLB miss, walk
        0x0100,  # VPN 0 → TLB hit (same page)
        0x1234,  # VPN 1 → TLB miss
        0x0200,  # VPN 0 → TLB hit
        0x2000,  # VPN 2 → TLB miss
        0x3000,  # VPN 3 → TLB miss
        0x4000,  # VPN 4 → TLB miss (evicts VPN 0)
        0x0300,  # VPN 0 → TLB miss again (was evicted)
        0xA000,  # VPN 10 → TLB miss
        0x6000,  # VPN 6 → page fault (not mapped)
    ]

    print(f"\n  TLB size: 4 entries, Page size: 4KB")
    print(f"  Mapped pages: {list(mappings.keys())}")
    print(f"\n  {'Virtual':>10}  {'VPN':>5}  {'Event':>12}  {'Physical':>12}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*12}  {'-'*12}")

    for vaddr in addresses:
        vpn = vaddr >> 12
        paddr, event = translator.translate(vaddr)
        pstr = f"0x{paddr:08X}" if paddr is not None else "---"
        print(f"  0x{vaddr:08X}  {vpn:>5}  {event:>12}  {pstr}")

    print(f"\n  TLB: {translator.tlb.hits} hits, {translator.tlb.misses} misses "
          f"({translator.tlb.hit_rate:.1%})")
    print(f"  Page walks: {translator.page_table.walk_count} "
          f"({translator.page_table.mem_accesses_for_walks} memory accesses)")
    print(f"  Page faults: {translator.page_faults}")


def demo_tlb_performance():
    """Compare TLB performance with different patterns."""
    print("\n" + "=" * 60)
    print("TLB PERFORMANCE COMPARISON")
    print("=" * 60)

    page_size = 4096
    n_pages = 64

    # Set up translator with all pages mapped
    def run_pattern(name: str, addresses: list[int], tlb_size: int) -> None:
        translator = AddressTranslator(tlb_size=tlb_size, page_size=page_size)
        mappings = {i: i + 100 for i in range(n_pages)}
        translator.setup_mappings(mappings)

        for addr in addresses:
            translator.translate(addr)

        total = translator.tlb.hits + translator.tlb.misses
        print(f"  {name:<30} {total:>7} {translator.tlb.hit_rate:>9.1%} "
              f"{translator.page_table.walk_count:>7}")

    # Patterns
    import random
    random.seed(42)

    # Sequential: stride through pages
    sequential = [i * page_size + 100 for i in range(n_pages)] * 4

    # Working set: 8 pages accessed repeatedly
    working = [random.choice(range(8)) * page_size for _ in range(500)]

    # Random: all pages uniformly
    rand_access = [random.randint(0, n_pages - 1) * page_size for _ in range(500)]

    for tlb_size in [8, 16, 32]:
        print(f"\n  TLB size: {tlb_size}")
        print(f"  {'Pattern':<30} {'Total':>7} {'Hit Rate':>9} {'Walks':>7}")
        print(f"  {'-'*30} {'-'*7} {'-'*9} {'-'*7}")

        run_pattern("Sequential", sequential, tlb_size)
        run_pattern("Working set (8 pages)", working, tlb_size)
        run_pattern("Random (64 pages)", rand_access, tlb_size)


def demo_emat():
    """Calculate Effective Memory Access Time with TLB."""
    print("\n" + "=" * 60)
    print("EFFECTIVE MEMORY ACCESS TIME (EMAT)")
    print("=" * 60)

    tlb_time = 1      # ns
    mem_time = 100     # ns
    page_levels = 2    # 2-level page table

    print(f"\n  TLB access: {tlb_time} ns")
    print(f"  Memory access: {mem_time} ns")
    print(f"  Page table levels: {page_levels}")

    print(f"\n  EMAT = h × (TLB + MEM) + (1-h) × (TLB + levels×MEM + MEM)")
    print(f"       = h × {tlb_time + mem_time} + (1-h) × {tlb_time + page_levels * mem_time + mem_time}")

    print(f"\n  {'TLB Hit Rate':>13}  {'EMAT (ns)':>10}  {'Slowdown':>10}")
    print(f"  {'-'*13}  {'-'*10}  {'-'*10}")

    baseline = mem_time  # no virtual memory
    for hr in [0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 0.999]:
        emat = hr * (tlb_time + mem_time) + (1 - hr) * (tlb_time + page_levels * mem_time + mem_time)
        slowdown = emat / baseline
        print(f"  {hr:>12.1%}  {emat:>9.1f}  {slowdown:>9.2f}x")


if __name__ == "__main__":
    demo_address_translation()
    demo_tlb_performance()
    demo_emat()
