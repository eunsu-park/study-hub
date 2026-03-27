"""
Paging and TLB Simulation

Demonstrates:
- Single-level page table address translation
- Two-level (hierarchical) page table
- Translation Lookaside Buffer (TLB) with LRU eviction
- Memory access pattern analysis

Theory:
- Paging divides virtual memory into fixed-size pages and physical
  memory into frames of the same size.
- Page table maps virtual page numbers (VPN) to physical frame
  numbers (PFN).
- TLB caches recent translations for fast lookup.
- Two-level page tables reduce memory overhead by only allocating
  second-level tables as needed.

Adapted from OS Theory Lesson 12.
"""

from collections import OrderedDict


# ── Single-Level Page Table ─────────────────────────────────────────────

class PageTableEntry:
    """A single page table entry."""

    def __init__(
        self,
        frame: int = -1,
        valid: bool = False,
        dirty: bool = False,
        referenced: bool = False,
        protection: str = "rw",
    ):
        self.frame = frame
        self.valid = valid
        self.dirty = dirty
        self.referenced = referenced
        self.protection = protection

    def __repr__(self) -> str:
        if not self.valid:
            return "[ INVALID ]"
        flags = ""
        flags += "D" if self.dirty else "-"
        flags += "R" if self.referenced else "-"
        return f"[frame={self.frame:3d} {flags} {self.protection}]"


class PageTable:
    """Single-level page table."""

    def __init__(self, page_size: int, virtual_bits: int, physical_bits: int):
        self.page_size = page_size
        self.offset_bits = page_size.bit_length() - 1
        self.num_pages = 2 ** (virtual_bits - self.offset_bits)
        self.num_frames = 2 ** (physical_bits - self.offset_bits)
        self.entries = [PageTableEntry() for _ in range(self.num_pages)]

    def map_page(self, vpn: int, frame: int, protection: str = "rw") -> None:
        self.entries[vpn] = PageTableEntry(
            frame=frame, valid=True, protection=protection
        )

    def translate(self, virtual_addr: int) -> int | None:
        """Translate virtual address to physical address."""
        # Extract VPN by shifting out the offset bits; extract offset by masking
        # with (page_size - 1). This works because page_size is a power of 2, so
        # the low-order bits form the offset and high-order bits form the VPN.
        vpn = virtual_addr >> self.offset_bits
        offset = virtual_addr & (self.page_size - 1)

        if vpn >= self.num_pages:
            print(f"    Error: VPN {vpn} out of range")
            return None

        entry = self.entries[vpn]
        if not entry.valid:
            print(f"    Page fault: VPN {vpn} not mapped")
            return None

        entry.referenced = True
        # Reconstruct physical address: frame number shifted left provides the
        # base address of the frame, OR'd with the offset to get the exact byte.
        # This mirrors how real MMU hardware concatenates frame bits and offset bits.
        physical_addr = (entry.frame << self.offset_bits) | offset
        return physical_addr

    def display(self) -> None:
        print(f"  Page Table ({self.num_pages} entries, "
              f"page size={self.page_size}B):")
        for i, entry in enumerate(self.entries):
            if entry.valid:
                print(f"    VPN {i:3d} → {entry}")


def demo_page_table():
    """Demonstrate single-level page table translation."""
    print("=" * 60)
    print("SINGLE-LEVEL PAGE TABLE")
    print("=" * 60)

    # 16-bit virtual, 15-bit physical, 4KB pages
    page_size = 4096  # 4KB
    pt = PageTable(page_size=page_size, virtual_bits=16, physical_bits=15)

    # Map some pages
    pt.map_page(0, frame=5)
    pt.map_page(1, frame=2)
    pt.map_page(3, frame=7)
    pt.map_page(5, frame=0, protection="r")

    pt.display()

    # Translate addresses
    test_addrs = [0x0000, 0x0010, 0x1234, 0x3FFF, 0x5100, 0x8000, 0x2000]
    print(f"\n  Address Translations (page size = {page_size}):")
    print(f"  {'Virtual':>10}  {'VPN':>5}  {'Offset':>8}  {'Physical':>10}  Status")
    print(f"  {'-'*10}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*12}")

    for vaddr in test_addrs:
        vpn = vaddr >> 12
        offset = vaddr & 0xFFF
        paddr = pt.translate(vaddr)
        if paddr is not None:
            status = "OK"
            pstr = f"0x{paddr:04X}"
        else:
            status = "PAGE FAULT"
            pstr = "---"
        print(f"  0x{vaddr:04X}      {vpn:>3}    0x{offset:03X}    {pstr:>10}  {status}")


# ── Two-Level Page Table ────────────────────────────────────────────────

class TwoLevelPageTable:
    """Two-level (hierarchical) page table.

    Virtual address split: [L1 index | L2 index | offset]
    Only allocates L2 tables when needed → saves memory.
    """

    def __init__(self, l1_bits: int, l2_bits: int, offset_bits: int):
        self.l1_bits = l1_bits
        self.l2_bits = l2_bits
        self.offset_bits = offset_bits
        self.l1_size = 2 ** l1_bits
        self.l2_size = 2 ** l2_bits

        # L1 table: pointers to L2 tables (None = not allocated).
        # This is the key space optimization: a flat table for a 32-bit address
        # space with 4KB pages needs 1M entries, but two-level only allocates
        # L2 tables for address regions actually in use.
        self.l1_table: list[list[PageTableEntry] | None] = [None] * self.l1_size
        self.l2_tables_allocated = 0

    def map_page(self, vpn: int, frame: int) -> None:
        l1_idx = vpn >> self.l2_bits
        l2_idx = vpn & (self.l2_size - 1)

        # Allocate L2 table if needed
        if self.l1_table[l1_idx] is None:
            self.l1_table[l1_idx] = [PageTableEntry() for _ in range(self.l2_size)]
            self.l2_tables_allocated += 1

        self.l1_table[l1_idx][l2_idx] = PageTableEntry(
            frame=frame, valid=True
        )

    def translate(self, virtual_addr: int) -> int | None:
        total_vpn_bits = self.l1_bits + self.l2_bits
        vpn = virtual_addr >> self.offset_bits
        offset = virtual_addr & ((1 << self.offset_bits) - 1)

        l1_idx = vpn >> self.l2_bits
        l2_idx = vpn & (self.l2_size - 1)

        # L1 lookup
        if self.l1_table[l1_idx] is None:
            return None  # page fault

        # L2 lookup
        entry = self.l1_table[l1_idx][l2_idx]
        if not entry.valid:
            return None  # page fault

        return (entry.frame << self.offset_bits) | offset

    def memory_usage(self) -> dict:
        """Calculate memory used by page tables."""
        entry_size = 4  # bytes per PTE
        l1_mem = self.l1_size * entry_size
        l2_mem = self.l2_tables_allocated * self.l2_size * entry_size

        # Compare with flat table
        total_pages = self.l1_size * self.l2_size
        flat_mem = total_pages * entry_size

        return {
            "l1_memory": l1_mem,
            "l2_memory": l2_mem,
            "total": l1_mem + l2_mem,
            "flat_equivalent": flat_mem,
            "savings_pct": (1 - (l1_mem + l2_mem) / flat_mem) * 100,
        }


def demo_two_level():
    """Demonstrate two-level page table."""
    print("\n" + "=" * 60)
    print("TWO-LEVEL PAGE TABLE")
    print("=" * 60)

    # 32-bit virtual: 10-bit L1 + 10-bit L2 + 12-bit offset
    pt = TwoLevelPageTable(l1_bits=10, l2_bits=10, offset_bits=12)

    # Map a few scattered pages
    mappings = [
        (0, 100),        # VPN 0 → frame 100
        (1, 101),        # VPN 1 → frame 101 (same L1 entry)
        (1024, 200),     # VPN 1024 → frame 200 (different L1 entry)
        (500000, 300),   # VPN 500000 → frame 300 (far away)
    ]

    for vpn, frame in mappings:
        pt.map_page(vpn, frame)
        l1 = vpn >> 10
        l2 = vpn & 0x3FF
        print(f"  Mapped VPN {vpn:>6} (L1={l1:>3}, L2={l2:>3}) → frame {frame}")

    usage = pt.memory_usage()
    print(f"\n  Memory usage:")
    print(f"    L1 table:        {usage['l1_memory']:>8} bytes")
    print(f"    L2 tables ({pt.l2_tables_allocated}):   {usage['l2_memory']:>8} bytes")
    print(f"    Total:           {usage['total']:>8} bytes")
    print(f"    Flat equivalent: {usage['flat_equivalent']:>8} bytes")
    print(f"    Savings:         {usage['savings_pct']:>7.1f}%")


# ── TLB Simulation ─────────────────────────────────────────────────────

class TLB:
    """Translation Lookaside Buffer with LRU eviction."""

    def __init__(self, size: int):
        self.size = size
        # OrderedDict for LRU: most recently used at end — this gives O(1)
        # lookup AND O(1) eviction (popitem from front), matching the
        # performance characteristics of a real hardware TLB with LRU
        self.entries: OrderedDict[int, int] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def lookup(self, vpn: int) -> int | None:
        if vpn in self.entries:
            self.hits += 1
            self.entries.move_to_end(vpn)  # mark as recently used
            return self.entries[vpn]
        self.misses += 1
        return None

    def insert(self, vpn: int, frame: int) -> int | None:
        """Insert mapping, evict LRU if full. Returns evicted VPN or None."""
        evicted = None
        if vpn in self.entries:
            self.entries.move_to_end(vpn)
            self.entries[vpn] = frame
        else:
            if len(self.entries) >= self.size:
                evicted_vpn, _ = self.entries.popitem(last=False)  # remove LRU
                evicted = evicted_vpn
            self.entries[vpn] = frame
        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def display(self) -> None:
        print(f"  TLB ({len(self.entries)}/{self.size} entries):")
        for vpn, frame in self.entries.items():
            print(f"    VPN {vpn:>3} → frame {frame:>3}")


def demo_tlb():
    """Demonstrate TLB with different access patterns."""
    print("\n" + "=" * 60)
    print("TLB SIMULATION")
    print("=" * 60)

    # Page table as simple dict
    page_table = {i: i * 3 + 7 for i in range(32)}  # VPN → frame

    def simulate(name: str, accesses: list[int], tlb_size: int) -> None:
        tlb = TLB(tlb_size)
        print(f"\n  --- {name} (TLB size={tlb_size}) ---")
        print(f"  Access sequence: {accesses[:20]}{'...' if len(accesses) > 20 else ''}")

        for vpn in accesses:
            frame = tlb.lookup(vpn)
            if frame is None:
                # TLB miss → consult page table
                frame = page_table[vpn]
                evicted = tlb.insert(vpn, frame)

        print(f"  Hits: {tlb.hits}, Misses: {tlb.misses}")
        print(f"  Hit rate: {tlb.hit_rate:.1%}")

    # Pattern 1: Sequential (poor locality for small TLB)
    sequential = list(range(16)) * 3
    simulate("Sequential", sequential, tlb_size=4)

    # Pattern 2: Temporal locality (same pages accessed repeatedly)
    temporal = [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3] * 3
    simulate("Temporal Locality", temporal, tlb_size=4)

    # Pattern 3: Working set fits in TLB
    working_set = [5, 7, 5, 3, 7, 5, 3, 5, 7, 3] * 5
    simulate("Working Set (fits)", working_set, tlb_size=4)

    # Pattern 4: Thrashing (working set > TLB)
    thrashing = [0, 1, 2, 3, 4] * 10
    simulate("Thrashing (5 pages, 4 TLB)", thrashing, tlb_size=4)


if __name__ == "__main__":
    demo_page_table()
    demo_two_level()
    demo_tlb()
