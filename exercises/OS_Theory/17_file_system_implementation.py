"""
Exercises for Lesson 17: File System Implementation
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers inode block pointers, FAT chain traversal, journaling recovery,
RAID capacity calculation, and file deletion process.
"""

import math


# === Exercise 1: inode Block Pointers ===
# Problem: Calculate indirect blocks needed for a 100MB file.

def exercise_1():
    """Calculate indirect block requirements for inode-based file storage."""
    block_size = 4096       # 4KB
    pointer_size = 4        # 4 bytes
    file_size_mb = 100
    file_size = file_size_mb * 1024 * 1024  # bytes

    pointers_per_block = block_size // pointer_size
    total_blocks = math.ceil(file_size / block_size)

    print(f"System parameters:")
    print(f"  Block size: {block_size // 1024}KB = {block_size} bytes")
    print(f"  Pointer size: {pointer_size} bytes")
    print(f"  Pointers per indirect block: {block_size} / {pointer_size} = {pointers_per_block}")
    print(f"  File size: {file_size_mb}MB = {file_size:,} bytes")
    print(f"  Total data blocks needed: {file_size:,} / {block_size} = {total_blocks:,}\n")

    # inode structure: 12 direct + 1 single indirect + 1 double indirect + 1 triple indirect
    direct_count = 12
    single_indirect_count = pointers_per_block
    double_indirect_count = pointers_per_block * pointers_per_block
    triple_indirect_count = pointers_per_block ** 3

    print(f"inode block pointer structure:")
    print(f"  Direct blocks (12):           {direct_count:>12,} blocks")
    print(f"  Single indirect (1 level):    {single_indirect_count:>12,} blocks")
    print(f"  Double indirect (2 levels):   {double_indirect_count:>12,} blocks")
    print(f"  Triple indirect (3 levels):   {triple_indirect_count:>12,} blocks")
    max_size = (direct_count + single_indirect_count + double_indirect_count + triple_indirect_count) * block_size
    print(f"  Maximum file size:            {max_size / (1024**4):.1f} TB\n")

    # Calculate how many blocks from each level
    remaining = total_blocks

    # Direct blocks
    direct_used = min(remaining, direct_count)
    remaining -= direct_used
    print(f"Block allocation for {file_size_mb}MB file ({total_blocks:,} blocks):")
    print(f"  Direct blocks used: {direct_used}")

    # Single indirect
    single_used = min(remaining, single_indirect_count)
    remaining -= single_used
    single_indirect_blocks = 1 if single_used > 0 else 0
    print(f"  Single indirect blocks used: {single_used} (needs {single_indirect_blocks} indirect block)")

    # Double indirect
    double_used = min(remaining, double_indirect_count)
    remaining -= double_used
    if double_used > 0:
        double_l2_blocks = math.ceil(double_used / pointers_per_block)
        double_l1_blocks = 1
        double_total_indirect = double_l1_blocks + double_l2_blocks
    else:
        double_l2_blocks = 0
        double_l1_blocks = 0
        double_total_indirect = 0
    print(f"  Double indirect blocks used: {double_used}")
    print(f"    Level-1 (top) indirect blocks: {double_l1_blocks}")
    print(f"    Level-2 indirect blocks: {double_l2_blocks}")

    # Triple indirect (not needed for 100MB)
    triple_used = remaining
    triple_total_indirect = 0
    if triple_used > 0:
        triple_l3 = math.ceil(triple_used / pointers_per_block)
        triple_l2 = math.ceil(triple_l3 / pointers_per_block)
        triple_l1 = 1
        triple_total_indirect = triple_l1 + triple_l2 + triple_l3
        print(f"  Triple indirect blocks used: {triple_used}")
    else:
        print(f"  Triple indirect: not needed")

    total_indirect = single_indirect_blocks + double_total_indirect + triple_total_indirect
    print(f"\nTotal indirect blocks needed: {total_indirect}")
    print(f"  Single indirect: {single_indirect_blocks}")
    print(f"  Double indirect level-1: {double_l1_blocks}")
    print(f"  Double indirect level-2: {double_l2_blocks}")
    print(f"  Total: {single_indirect_blocks} + {double_l1_blocks} + {double_l2_blocks} = {total_indirect}")

    overhead = total_indirect * block_size
    overhead_pct = overhead / file_size * 100
    print(f"\nMetadata overhead: {total_indirect} blocks * {block_size // 1024}KB = {overhead // 1024}KB")
    print(f"  ({overhead_pct:.2f}% of file size)")


# === Exercise 2: FAT Chain Traversal ===
# Problem: Trace FAT chain for file A starting at cluster 3.

def exercise_2():
    """Traverse FAT (File Allocation Table) cluster chains."""
    # FAT table: cluster -> next cluster (or FREE/EOF)
    fat = {
        0: "FREE",
        1: 8,
        2: "FREE",
        3: 7,
        4: "EOF",
        5: 1,
        6: "FREE",
        7: 4,
        8: "EOF",
    }

    print(f"FAT Table:")
    print(f"  {'Cluster':<10} {'Value'}")
    print("  " + "-" * 20)
    for cluster, value in sorted(fat.items()):
        print(f"  {cluster:<10} {value}")
    print()

    def trace_chain(fat_table, start):
        """Follow a FAT chain from start to EOF."""
        chain = [start]
        current = start
        while fat_table[current] != "EOF":
            next_cluster = fat_table[current]
            if next_cluster == "FREE":
                print(f"  ERROR: Chain leads to FREE cluster at {current}!")
                break
            chain.append(next_cluster)
            current = next_cluster
        return chain

    # File A starts at cluster 3
    print("File A (starting cluster: 3):")
    chain_a = trace_chain(fat, 3)
    chain_str = " -> ".join(str(c) for c in chain_a) + " -> EOF"
    print(f"  Chain: {chain_str}")
    print(f"  Clusters used: {len(chain_a)}")
    print(f"  Traversal: ", end="")
    for i, c in enumerate(chain_a):
        if i > 0:
            print(f" -> FAT[{chain_a[i-1]}]={c}", end="")
        else:
            print(f"Start={c}", end="")
    print(f" -> FAT[{chain_a[-1]}]=EOF\n")

    # Discover other file chains
    print("Discovering other file chains:\n")

    # Find all chain starts (clusters pointed to by directory entries, not by other FAT entries)
    pointed_to = set()
    for cluster, value in fat.items():
        if isinstance(value, int):
            pointed_to.add(value)

    # Potential start clusters: allocated clusters not pointed to by other clusters
    allocated = {c for c, v in fat.items() if v != "FREE"}
    potential_starts = allocated - pointed_to

    for start in sorted(potential_starts):
        chain = trace_chain(fat, start)
        chain_str = " -> ".join(str(c) for c in chain) + " -> EOF"
        print(f"  Chain starting at {start}: {chain_str} ({len(chain)} clusters)")

    print(f"\nFree clusters: {sorted(c for c, v in fat.items() if v == 'FREE')}")

    # Fragmentation analysis
    print(f"\nFragmentation analysis:")
    for start in sorted(potential_starts):
        chain = trace_chain(fat, start)
        sequential = all(chain[i] + 1 == chain[i + 1] for i in range(len(chain) - 1))
        if sequential:
            print(f"  Chain from {start}: contiguous (no fragmentation)")
        else:
            print(f"  Chain from {start}: fragmented (clusters not sequential)")
            print(f"    Cluster sequence: {chain}")
            print(f"    Disk head must seek between non-adjacent clusters")


# === Exercise 3: Journaling Recovery ===
# Problem: Determine file system state after crash with partial journal.

def exercise_3():
    """Analyze journaling recovery after a system crash."""
    print("Journal contents after crash:\n")
    print("  [TxB_1] [Block 100: DataA] [Block 101: MetadataA] [TxE_1]")
    print("  [TxB_2] [Block 200: DataB] [Block 201: MetadataB]")
    print("  (No TxE_2 -- crash occurred before Transaction 2 completed)\n")

    print("=" * 50)
    print("Recovery process:\n")

    print("Phase 1: Scan journal for transaction boundaries")
    print("  - Transaction 1: TxB_1 found, TxE_1 found -> COMPLETE")
    print("  - Transaction 2: TxB_2 found, TxE_2 NOT found -> INCOMPLETE\n")

    print("Phase 2: Process Transaction 1 (COMPLETE)")
    print("  Action: REDO (replay to ensure changes are on disk)")
    print("  - Write Block 100 (DataA) to its actual location on disk")
    print("  - Write Block 101 (MetadataA) to its actual location on disk")
    print("  - Transaction 1 changes are guaranteed to be applied")
    print("  - Even if they were already on disk, re-applying is idempotent\n")

    print("Phase 3: Process Transaction 2 (INCOMPLETE)")
    print("  Action: DISCARD (do not apply)")
    print("  - Block 200 (DataB): NOT written to disk")
    print("  - Block 201 (MetadataB): NOT written to disk")
    print("  - Transaction 2 is treated as if it never happened")
    print("  - The operation that generated Tx2 must be retried by the application\n")

    print("Phase 4: Clear processed journal entries\n")

    print("=" * 50)
    print("\nFinal file system state:")
    print("  - Transaction 1 changes: APPLIED (file A modifications complete)")
    print("  - Transaction 2 changes: NOT APPLIED (file B unchanged)")
    print("  - File system is CONSISTENT\n")

    print("Why this works:")
    print("  The journal provides atomicity for each transaction:")
    print("  - Complete transactions (with TxE) are guaranteed durable")
    print("  - Incomplete transactions are rolled back (never partially applied)")
    print("  - No torn writes: either ALL of a transaction's blocks are applied,")
    print("    or NONE of them are\n")

    print("Journal modes comparison:")
    modes = [
        ("Journal (full)",   "Both data and metadata journaled",
         "Safest, slowest: all writes go to journal first"),
        ("Ordered (default)", "Only metadata journaled, data written first",
         "Data written before metadata commit ensures consistency"),
        ("Writeback",        "Only metadata journaled, data order unspecified",
         "Fastest, but data may be stale after crash"),
    ]
    print(f"  {'Mode':<20} {'What's journaled':<42} {'Trade-off'}")
    print("  " + "-" * 90)
    for mode, what, tradeoff in modes:
        print(f"  {mode:<20} {what:<42} {tradeoff}")


# === Exercise 4: RAID Capacity Calculation ===
# Problem: Calculate usable capacity for 6x 2TB disks across RAID levels.

def exercise_4():
    """Calculate RAID capacity for different configurations."""
    num_disks = 6
    disk_size_tb = 2
    total_tb = num_disks * disk_size_tb

    print(f"Configuration: {num_disks} x {disk_size_tb}TB disks = {total_tb}TB raw capacity\n")

    raids = [
        {
            "level": "RAID 0 (Striping)",
            "usable": num_disks * disk_size_tb,
            "formula": f"{num_disks} x {disk_size_tb}TB",
            "pct": 100,
            "redundancy": "None -- any single disk failure loses ALL data",
            "min_disks": 2,
        },
        {
            "level": "RAID 1 (Mirroring)",
            "usable": (num_disks // 2) * disk_size_tb,
            "formula": f"({num_disks}/2) x {disk_size_tb}TB (3 mirrored pairs)",
            "pct": 50,
            "redundancy": "Can lose 1 disk per mirrored pair (up to 3 disks if from different pairs)",
            "min_disks": 2,
        },
        {
            "level": "RAID 5 (Distributed Parity)",
            "usable": (num_disks - 1) * disk_size_tb,
            "formula": f"({num_disks}-1) x {disk_size_tb}TB",
            "pct": (num_disks - 1) / num_disks * 100,
            "redundancy": "Can lose any 1 disk",
            "min_disks": 3,
        },
        {
            "level": "RAID 6 (Double Parity)",
            "usable": (num_disks - 2) * disk_size_tb,
            "formula": f"({num_disks}-2) x {disk_size_tb}TB",
            "pct": (num_disks - 2) / num_disks * 100,
            "redundancy": "Can lose any 2 disks simultaneously",
            "min_disks": 4,
        },
        {
            "level": "RAID 10 (Striped Mirrors)",
            "usable": (num_disks // 2) * disk_size_tb,
            "formula": f"({num_disks}/2) x {disk_size_tb}TB (3 striped mirror pairs)",
            "pct": 50,
            "redundancy": "Can lose 1 disk per pair; best read performance",
            "min_disks": 4,
        },
    ]

    print(f"{'RAID Level':<30} {'Formula':<35} {'Usable':<10} {'Efficiency':<12} {'Redundancy'}")
    print("-" * 120)
    for r in raids:
        print(f"{r['level']:<30} {r['formula']:<35} {r['usable']}TB{'':<6} {r['pct']:.0f}%{'':<8} {r['redundancy']}")

    print(f"\nRecommendation by use case:")
    print(f"  - Maximum performance, no redundancy needed: RAID 0 (12TB)")
    print(f"  - General server with good redundancy:       RAID 5 (10TB)")
    print(f"  - High-reliability (can't risk rebuild):     RAID 6 (8TB)")
    print(f"  - Database (high IOPS + redundancy):         RAID 10 (6TB)")
    print(f"  - Backup mirror:                             RAID 1 (6TB)")


# === Exercise 5: File Deletion Process ===
# Problem: Trace the process of deleting /home/user/file.txt in Unix.

def exercise_5():
    """Trace Unix file deletion from inode and block perspective."""
    print("Command: rm /home/user/file.txt\n")

    steps = [
        ("1. Path Resolution", [
            "Resolve '/' -> root inode (inode 2)",
            "Read root directory data blocks, find 'home' -> inode 100",
            "Read inode 100 data blocks, find 'user' -> inode 200",
            "Read inode 200 data blocks, find 'file.txt' -> inode 12345",
        ]),
        ("2. Permission Check", [
            "Check WRITE permission on /home/user directory (inode 200)",
            "  The user must have write permission on the DIRECTORY",
            "  (not the file itself!) to remove a directory entry",
            "Check if file has the sticky bit set on the directory",
            "  If sticky bit is set (like /tmp), only owner can delete",
        ]),
        ("3. Remove Directory Entry", [
            "Remove 'file.txt' entry from /home/user's directory data",
            "In ext4: set the entry's inode to 0, merge rec_len with neighbor",
            "Update directory modification time (mtime)",
            "This is the only operation that 'deletes' the filename",
        ]),
        ("4. Decrement Inode Link Count", [
            "inode 12345: link_count-- (e.g., 1 -> 0)",
            "If link_count > 0: STOP HERE",
            "  Other hard links still reference this inode",
            "  Data remains accessible through other filenames",
        ]),
        ("5. Check Open File References", [
            "If link_count == 0 but file is still open by a process:",
            "  Mark inode as 'orphan' (unlinked but still in use)",
            "  Data blocks are NOT freed yet",
            "  When last process closes the file, proceed to step 6",
            "This is why 'rm' of a file being read doesn't break the reader",
        ]),
        ("6. Free Data Blocks", [
            "Read inode 12345's block pointers:",
            "  Free direct blocks (up to 12 blocks)",
            "  Free single indirect blocks + the indirect block itself",
            "  Free double indirect blocks + both levels of indirect blocks",
            "  Free triple indirect blocks + all three levels",
            "Mark each block as free in the BLOCK BITMAP",
        ]),
        ("7. Free Inode", [
            "Mark inode 12345 as free in the INODE BITMAP",
            "Update superblock: increment free inode count",
            "Update superblock: increment free block count",
            "Update block group descriptor (if using block groups)",
        ]),
    ]

    for title, details in steps:
        print(f"{title}:")
        for detail in details:
            print(f"  {detail}")
        print()

    print("Important notes:")
    print("  - The actual data bytes are NOT erased or zeroed!")
    print("  - Blocks are merely marked as 'available' in the bitmap")
    print("  - Data remains on disk until overwritten by new allocations")
    print("  - This is why file recovery tools can recover 'deleted' files")
    print("  - For secure deletion, use 'shred' to overwrite data blocks")
    print()

    # Simulate the deletion
    print("--- Simulation ---\n")
    inode = {
        "number": 12345,
        "link_count": 2,
        "size": 50000,
        "direct_blocks": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        "single_indirect": 112,
        "open_count": 0,
    }

    print(f"Before deletion:")
    print(f"  Inode {inode['number']}: link_count={inode['link_count']}, "
          f"size={inode['size']}, open_count={inode['open_count']}")
    print(f"  Data blocks: {inode['direct_blocks'][:6]}... + indirect")

    # First rm: decrease link count
    inode["link_count"] -= 1
    print(f"\nAfter first 'rm' (removing one hard link):")
    print(f"  Inode {inode['number']}: link_count={inode['link_count']}")
    print(f"  link_count > 0: data preserved (another hard link exists)")

    # Second rm: link count reaches 0
    inode["link_count"] -= 1
    print(f"\nAfter second 'rm' (removing last hard link):")
    print(f"  Inode {inode['number']}: link_count={inode['link_count']}")
    if inode["open_count"] > 0:
        print(f"  open_count > 0: defer cleanup (file still in use)")
    else:
        print(f"  link_count=0 and open_count=0: free all resources")
        num_blocks = len(inode["direct_blocks"]) + 1  # +1 for indirect
        print(f"  Freed {num_blocks} data blocks + 1 indirect block")
        print(f"  Freed inode {inode['number']}")
        print(f"  Updated block bitmap: {num_blocks + 1} blocks marked free")
        print(f"  Updated inode bitmap: inode {inode['number']} marked free")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: inode Block Pointers ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: FAT Chain Traversal ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Journaling Recovery ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: RAID Capacity Calculation ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: File Deletion Process ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
