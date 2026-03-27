"""
Exercises for Lesson 28: Capstone — xv6 Kernel Walkthrough
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers xv6-style physical memory allocation, process table and fork
simulation, and inode-based file system layout analysis.
"""


# === Exercise 1: Physical Memory Allocator (kalloc/kfree) ===
# Problem: Implement xv6's free-list based physical page allocator and
# demonstrate allocation, deallocation, and use-after-free detection.

def exercise_1():
    """Simulate xv6's kalloc/kfree physical memory allocator."""

    PAGE_SIZE = 4096  # 4 KB pages

    class PhysicalPage:
        """Represents a single physical memory page."""

        def __init__(self, address):
            self.address = address
            self.data = bytearray(PAGE_SIZE)

        def fill(self, value):
            """Fill the page with a byte pattern (for debugging)."""
            for i in range(PAGE_SIZE):
                self.data[i] = value & 0xFF

        def __repr__(self):
            return f"Page(0x{self.address:08x})"

    class FreeListAllocator:
        """xv6-style free-list physical page allocator.

        Free pages are linked together. Each free page's first bytes
        store a pointer (index) to the next free page.
        """

        def __init__(self, start_addr, end_addr):
            self.start_addr = start_addr
            self.end_addr = end_addr
            self.pages = {}           # address -> PhysicalPage
            self.freelist = None      # Head of free list (address)
            self.free_count = 0
            self.total_pages = 0
            self.alloc_count = 0
            self.free_op_count = 0

            # Initialize: create all pages and add to free list
            addr = start_addr
            while addr < end_addr:
                page = PhysicalPage(addr)
                self.pages[addr] = page
                self._free_page(addr, initial=True)
                self.total_pages += 1
                addr += PAGE_SIZE

        def _free_page(self, addr, initial=False):
            """Add a page to the head of the free list."""
            page = self.pages[addr]
            # Fill with junk pattern (0x01) to catch use-after-free
            if not initial:
                page.fill(0x01)

            # Store next pointer in the first 8 bytes of the page
            if self.freelist is not None:
                # Encode next pointer
                next_addr = self.freelist
                for i in range(8):
                    page.data[i] = (next_addr >> (i * 8)) & 0xFF
            else:
                for i in range(8):
                    page.data[i] = 0

            self.freelist = addr
            self.free_count += 1

        def kalloc(self):
            """Allocate one physical page. Returns address or None."""
            if self.freelist is None:
                return None  # Out of memory

            addr = self.freelist
            page = self.pages[addr]

            # Read next pointer from the page's first 8 bytes
            next_addr = 0
            for i in range(8):
                next_addr |= page.data[i] << (i * 8)

            self.freelist = next_addr if next_addr != 0 else None
            self.free_count -= 1
            self.alloc_count += 1

            # Fill with junk pattern (0x05) to catch reads of uninitialized data
            page.fill(0x05)

            return addr

        def kfree(self, addr):
            """Free a physical page back to the allocator."""
            if addr not in self.pages:
                print(f"  ERROR: kfree(0x{addr:08x}) - invalid address!")
                return
            self._free_page(addr)
            self.free_op_count += 1

        def stats(self):
            return {
                "total_pages": self.total_pages,
                "free_pages": self.free_count,
                "allocated_pages": self.total_pages - self.free_count,
                "kalloc_calls": self.alloc_count,
                "kfree_calls": self.free_op_count,
            }

    # Simulate with 128 KB of physical memory (32 pages)
    PHYS_START = 0x80000000
    PHYS_END = PHYS_START + 32 * PAGE_SIZE

    print("=== xv6 Physical Memory Allocator ===\n")
    print(f"Physical memory: 0x{PHYS_START:08x} - 0x{PHYS_END:08x}")
    print(f"Page size: {PAGE_SIZE} bytes")
    print(f"Total pages: {(PHYS_END - PHYS_START) // PAGE_SIZE}\n")

    alloc = FreeListAllocator(PHYS_START, PHYS_END)
    s = alloc.stats()
    print(f"After initialization: {s['free_pages']} free pages\n")

    # Allocate some pages
    print("--- Allocation Sequence ---\n")
    allocated = []
    for i in range(5):
        addr = alloc.kalloc()
        allocated.append(addr)
        print(f"  kalloc() -> 0x{addr:08x}")

    s = alloc.stats()
    print(f"\n  Free: {s['free_pages']}, Allocated: {s['allocated_pages']}\n")

    # Free some pages
    print("--- Free Sequence ---\n")
    for addr in allocated[:3]:
        print(f"  kfree(0x{addr:08x})")
        alloc.kfree(addr)

    s = alloc.stats()
    print(f"\n  Free: {s['free_pages']}, Allocated: {s['allocated_pages']}\n")

    # Re-allocate and show LIFO behavior
    print("--- Re-allocation (LIFO order) ---\n")
    print("  Free list is LIFO: last freed page is first allocated.\n")
    for i in range(3):
        addr = alloc.kalloc()
        print(f"  kalloc() -> 0x{addr:08x}")

    # Demonstrate use-after-free detection
    print("\n--- Use-After-Free Detection ---\n")
    addr = alloc.kalloc()
    page = alloc.pages[addr]

    # Write meaningful data
    msg = b"IMPORTANT DATA"
    for i, b in enumerate(msg):
        page.data[i] = b
    print(f"  Wrote '{msg.decode()}' to page 0x{addr:08x}")

    # Free the page
    alloc.kfree(addr)
    print(f"  kfree(0x{addr:08x})")

    # Try to read the data back
    read_back = bytes(page.data[:len(msg)])
    is_junk = all(b == 0x01 for b in page.data[:len(msg)])
    print(f"  Read back: {read_back.hex()} ({'junk pattern' if is_junk else 'stale data'})")
    print(f"  -> Page was filled with 0x01 on free, original data destroyed.")
    print(f"  -> This is how xv6 detects use-after-free bugs.\n")

    # Final stats
    s = alloc.stats()
    print(f"Final statistics:")
    for k, v in s.items():
        print(f"  {k}: {v}")


# === Exercise 2: Process Table and Fork Simulation ===
# Problem: Simulate xv6's process table, fork(), and round-robin scheduler.

def exercise_2():
    """Simulate xv6 process management: fork, exit, wait, and scheduling."""

    NPROC = 16  # Maximum number of processes (xv6 uses 64)

    class TrapFrame:
        """Saved CPU registers (simplified)."""

        def __init__(self):
            self.a0 = 0     # Return value / first argument
            self.epc = 0    # Program counter
            self.sp = 0     # Stack pointer

        def copy_from(self, other):
            self.a0 = other.a0
            self.epc = other.epc
            self.sp = other.sp

    class Process:
        """xv6-style process control block."""

        UNUSED = "UNUSED"
        RUNNABLE = "RUNNABLE"
        RUNNING = "RUNNING"
        SLEEPING = "SLEEPING"
        ZOMBIE = "ZOMBIE"

        def __init__(self):
            self.state = Process.UNUSED
            self.pid = 0
            self.parent = None
            self.name = ""
            self.sz = 0               # Memory size
            self.trapframe = TrapFrame()
            self.ofile = [None] * 16  # Open file descriptors
            self.xstate = 0           # Exit status
            self.memory = {}          # Simulated memory pages

    class ProcessTable:
        """xv6-style process table with fork/exit/wait/scheduler."""

        def __init__(self):
            self.procs = [Process() for _ in range(NPROC)]
            self.next_pid = 1
            self.current = None
            self.schedule_log = []

        def allocproc(self):
            """Find an unused slot and initialize it."""
            for p in self.procs:
                if p.state == Process.UNUSED:
                    p.pid = self.next_pid
                    self.next_pid += 1
                    p.state = Process.RUNNABLE
                    p.trapframe = TrapFrame()
                    p.memory = {}
                    return p
            return None  # No free slots

        def userinit(self):
            """Create the first user process (init)."""
            p = self.allocproc()
            if p is None:
                return None
            p.name = "init"
            p.sz = 4096
            p.memory[0] = bytearray(4096)  # One page of code/data
            p.trapframe.epc = 0  # Start at address 0
            p.trapframe.sp = 4096
            return p

        def fork(self, parent):
            """Create a child process (copy of parent)."""
            child = self.allocproc()
            if child is None:
                return -1  # No free slots

            # Copy memory
            child.sz = parent.sz
            for addr, data in parent.memory.items():
                child.memory[addr] = bytearray(data)

            # Copy trapframe
            child.trapframe.copy_from(parent.trapframe)
            child.trapframe.a0 = 0  # fork() returns 0 in child

            # Copy open file descriptors (increment refcount in real xv6)
            for i in range(len(parent.ofile)):
                child.ofile[i] = parent.ofile[i]

            child.parent = parent
            child.name = parent.name
            child.state = Process.RUNNABLE

            return child.pid  # fork() returns child PID in parent

        def exit(self, proc, status):
            """Terminate a process."""
            proc.xstate = status
            proc.state = Process.ZOMBIE
            # Reparent children to init
            for p in self.procs:
                if p.parent == proc and p.state != Process.UNUSED:
                    p.parent = self.procs[0]  # init

        def wait(self, parent):
            """Wait for a child process to exit. Returns (pid, status)."""
            for p in self.procs:
                if p.parent == parent and p.state == Process.ZOMBIE:
                    pid = p.pid
                    status = p.xstate
                    # Free the process slot
                    p.state = Process.UNUSED
                    p.pid = 0
                    p.parent = None
                    p.memory = {}
                    return pid, status
            return -1, 0  # No zombie children

        def scheduler(self, num_ticks):
            """Round-robin scheduler (simplified xv6 scheduler)."""
            tick = 0
            idx = 0
            while tick < num_ticks:
                found = False
                for i in range(NPROC):
                    p = self.procs[(idx + i) % NPROC]
                    if p.state == Process.RUNNABLE:
                        p.state = Process.RUNNING
                        self.current = p
                        self.schedule_log.append((tick, p.pid, p.name))
                        # Process runs for one tick
                        p.state = Process.RUNNABLE
                        self.current = None
                        idx = (idx + i + 1) % NPROC
                        tick += 1
                        found = True
                        break
                if not found:
                    tick += 1  # Idle tick

        def show_table(self):
            """Display the process table."""
            print(f"  {'PID':<5} {'State':<12} {'Name':<10} "
                  f"{'Size':<8} {'Parent'}")
            print("  " + "-" * 45)
            for p in self.procs:
                if p.state != Process.UNUSED:
                    parent_pid = p.parent.pid if p.parent else "-"
                    print(f"  {p.pid:<5} {p.state:<12} {p.name:<10} "
                          f"{p.sz:<8} {parent_pid}")

    print("=== xv6 Process Management Simulation ===\n")

    pt = ProcessTable()

    # Step 1: Create init process
    print("Step 1: Create init process (userinit)")
    init = pt.userinit()
    pt.show_table()

    # Step 2: Fork shell from init
    print("\nStep 2: init forks shell")
    shell_pid = pt.fork(init)
    # Find shell process and rename it
    for p in pt.procs:
        if p.pid == shell_pid:
            p.name = "sh"
            break
    pt.show_table()

    # Step 3: Shell forks to run a command
    print("\nStep 3: shell forks to execute 'ls'")
    shell = next(p for p in pt.procs if p.pid == shell_pid)
    ls_pid = pt.fork(shell)
    for p in pt.procs:
        if p.pid == ls_pid:
            p.name = "ls"
            break
    pt.show_table()

    # Step 4: Fork another command
    print("\nStep 4: shell forks to execute 'cat'")
    cat_pid = pt.fork(shell)
    for p in pt.procs:
        if p.pid == cat_pid:
            p.name = "cat"
            break
    pt.show_table()

    # Step 5: Run the scheduler
    print("\nStep 5: Run round-robin scheduler (12 ticks)\n")
    pt.scheduler(12)

    print(f"  {'Tick':<6} {'PID':<5} {'Name'}")
    print("  " + "-" * 20)
    for tick, pid, name in pt.schedule_log:
        print(f"  {tick:<6} {pid:<5} {name}")

    # Step 6: ls exits, shell waits
    print("\nStep 6: 'ls' exits with status 0")
    ls_proc = next(p for p in pt.procs if p.pid == ls_pid)
    pt.exit(ls_proc, 0)

    reaped_pid, status = pt.wait(shell)
    print(f"  shell calls wait() -> pid={reaped_pid}, status={status}")
    pt.show_table()

    # Step 7: cat exits
    print("\nStep 7: 'cat' exits with status 0")
    cat_proc = next(p for p in pt.procs if p.pid == cat_pid)
    pt.exit(cat_proc, 0)
    reaped_pid, status = pt.wait(shell)
    print(f"  shell calls wait() -> pid={reaped_pid}, status={status}")
    pt.show_table()

    print(f"\n  Key xv6 process lifecycle:")
    print(f"  UNUSED -> allocproc() -> RUNNABLE -> scheduler -> RUNNING")
    print(f"  RUNNING -> yield/sleep -> RUNNABLE/SLEEPING")
    print(f"  RUNNING -> exit() -> ZOMBIE -> parent wait() -> UNUSED")


# === Exercise 3: xv6 File System Layout ===
# Problem: Simulate the xv6 file system structure including superblock,
# inodes, bitmap, and data blocks with direct and indirect addressing.

def exercise_3():
    """Simulate xv6 inode-based file system layout and operations."""

    BSIZE = 1024       # Block size (bytes)
    NDIRECT = 12       # Direct block pointers per inode
    NINDIRECT = 256    # Indirect block pointers (BSIZE / sizeof(uint))
    MAXFILE = NDIRECT + NINDIRECT  # Max blocks per file
    NINODES = 200      # Number of inodes

    class Superblock:
        """File system metadata."""

        def __init__(self, nblocks):
            self.magic = 0x10203040
            self.size = nblocks       # Total blocks
            self.nblocks = nblocks - 46  # Data blocks
            self.ninodes = NINODES
            self.nlog = 30            # Log blocks
            self.logstart = 2
            self.inodestart = 32
            self.bmapstart = 45

    class Inode:
        """On-disk inode structure."""

        T_DIR = 1
        T_FILE = 2

        def __init__(self, inum):
            self.inum = inum
            self.type = 0         # 0=free, 1=dir, 2=file
            self.nlink = 0        # Number of links
            self.size = 0         # File size in bytes
            self.addrs = [0] * (NDIRECT + 1)  # 12 direct + 1 indirect

    class FileSystem:
        """Simplified xv6 file system."""

        def __init__(self, nblocks=1000):
            self.sb = Superblock(nblocks)
            self.inodes = [Inode(i) for i in range(NINODES)]
            self.bitmap = [False] * nblocks  # True = allocated
            self.blocks = {}   # block_num -> data
            self.next_data_block = 46  # First data block

            # Mark system blocks as used
            for i in range(46):
                self.bitmap[i] = True

            # Initialize root directory (inode 1)
            root = self.inodes[1]
            root.type = Inode.T_DIR
            root.nlink = 1

        def alloc_block(self):
            """Allocate a data block from the bitmap."""
            for i in range(self.next_data_block, self.sb.size):
                if not self.bitmap[i]:
                    self.bitmap[i] = True
                    self.blocks[i] = bytearray(BSIZE)
                    return i
            return 0  # No free blocks

        def free_block(self, bnum):
            """Free a data block."""
            self.bitmap[bnum] = False
            if bnum in self.blocks:
                del self.blocks[bnum]

        def alloc_inode(self, itype):
            """Allocate an inode."""
            for i in range(1, NINODES):
                if self.inodes[i].type == 0:
                    self.inodes[i].type = itype
                    self.inodes[i].nlink = 1
                    self.inodes[i].size = 0
                    return self.inodes[i]
            return None

        def bmap(self, inode, bn):
            """Map a logical block number to a physical block number.
            Allocates blocks as needed (like xv6 bmap)."""
            if bn < NDIRECT:
                if inode.addrs[bn] == 0:
                    inode.addrs[bn] = self.alloc_block()
                return inode.addrs[bn]
            else:
                bn -= NDIRECT
                if bn >= NINDIRECT:
                    return 0  # File too large

                # Allocate indirect block if needed
                if inode.addrs[NDIRECT] == 0:
                    inode.addrs[NDIRECT] = self.alloc_block()

                indirect_block = inode.addrs[NDIRECT]
                # Read pointer from indirect block
                # In real xv6, this reads from disk. We simulate with dict.
                key = (indirect_block, bn)
                if key not in self.blocks:
                    self.blocks[key] = self.alloc_block()
                return self.blocks[key]

        def write_file(self, inode, data):
            """Write data to a file inode."""
            offset = 0
            bn = 0
            while offset < len(data):
                block_num = self.bmap(inode, bn)
                if block_num == 0:
                    break
                chunk = data[offset:offset + BSIZE]
                if block_num not in self.blocks:
                    self.blocks[block_num] = bytearray(BSIZE)
                self.blocks[block_num][:len(chunk)] = chunk
                offset += BSIZE
                bn += 1
            inode.size = len(data)

        def read_file(self, inode, length=None):
            """Read data from a file inode."""
            if length is None:
                length = inode.size
            data = bytearray()
            bn = 0
            while len(data) < length:
                block_num = self.bmap(inode, bn)
                if block_num == 0 or block_num not in self.blocks:
                    break
                remaining = min(BSIZE, length - len(data))
                data.extend(self.blocks[block_num][:remaining])
                bn += 1
            return bytes(data[:length])

        def file_stats(self, inode):
            """Show file block mapping."""
            blocks_used = 0
            direct_used = 0
            indirect_used = 0

            for i in range(NDIRECT):
                if inode.addrs[i] != 0:
                    blocks_used += 1
                    direct_used += 1

            if inode.addrs[NDIRECT] != 0:
                blocks_used += 1  # The indirect block itself
                for bn in range(NINDIRECT):
                    key = (inode.addrs[NDIRECT], bn)
                    if key in self.blocks:
                        blocks_used += 1
                        indirect_used += 1

            return {
                "size": inode.size,
                "blocks": blocks_used,
                "direct": direct_used,
                "indirect": indirect_used,
                "max_size": MAXFILE * BSIZE,
            }

    print("=== xv6 File System Layout ===\n")

    fs = FileSystem(nblocks=1000)

    # Show disk layout
    print("Disk Layout:")
    print(f"  Block 0:     Boot sector (unused)")
    print(f"  Block 1:     Superblock (magic=0x{fs.sb.magic:08x})")
    print(f"  Block 2-31:  Log ({fs.sb.nlog} blocks for crash recovery)")
    print(f"  Block 32-44: Inode blocks ({NINODES} inodes)")
    print(f"  Block 45:    Bitmap (free block tracking)")
    print(f"  Block 46+:   Data blocks ({fs.sb.nblocks} available)")
    print(f"  Total:       {fs.sb.size} blocks ({fs.sb.size * BSIZE // 1024} KB)\n")

    # Create files of various sizes
    print("--- Creating Files ---\n")

    # Small file (fits in direct blocks)
    small_inode = fs.alloc_inode(Inode.T_FILE)
    small_data = b"Hello, xv6!" * 10  # 110 bytes
    fs.write_file(small_inode, small_data)

    # Medium file (uses several direct blocks)
    medium_inode = fs.alloc_inode(Inode.T_FILE)
    medium_data = bytes(range(256)) * 40  # 10,240 bytes (~10 KB)
    fs.write_file(medium_inode, medium_data)

    # Large file (requires indirect block)
    large_inode = fs.alloc_inode(Inode.T_FILE)
    large_data = bytes([0xAB]) * (15 * BSIZE)  # 15,360 bytes (~15 KB)
    fs.write_file(large_inode, large_data)

    files = [
        ("small.txt",  small_inode,  small_data),
        ("medium.bin", medium_inode, medium_data),
        ("large.dat",  large_inode,  large_data),
    ]

    print(f"  {'File':<14} {'Inode':<7} {'Size':<10} {'Direct':<9} "
          f"{'Indirect':<10} {'Total Blocks'}")
    print("  " + "-" * 60)

    for name, inode, data in files:
        stats = fs.file_stats(inode)
        print(f"  {name:<14} {inode.inum:<7} {stats['size']:<10} "
              f"{stats['direct']:<9} {stats['indirect']:<10} {stats['blocks']}")

    # Verify data integrity
    print(f"\n--- Data Integrity Check ---\n")
    for name, inode, original_data in files:
        read_back = fs.read_file(inode)
        matches = read_back == original_data
        print(f"  {name}: read {len(read_back)} bytes, "
              f"integrity {'OK' if matches else 'FAIL'}")

    # Show inode block addressing
    print(f"\n--- Inode Block Addressing (large.dat) ---\n")

    print(f"  Direct blocks (addrs[0..11]):")
    for i in range(NDIRECT):
        addr = large_inode.addrs[i]
        if addr != 0:
            print(f"    addrs[{i:>2}] -> block {addr}")

    if large_inode.addrs[NDIRECT] != 0:
        print(f"\n  Indirect block: addrs[{NDIRECT}] -> block "
              f"{large_inode.addrs[NDIRECT]}")
        print(f"  Indirect entries:")
        count = 0
        for bn in range(NINDIRECT):
            key = (large_inode.addrs[NDIRECT], bn)
            if key in fs.blocks:
                print(f"    indirect[{bn}] -> block {fs.blocks[key]}")
                count += 1
        if count == 0:
            print(f"    (none)")

    # Show maximum file size calculation
    print(f"\n--- Maximum File Size ---\n")
    print(f"  Direct blocks:   {NDIRECT} x {BSIZE} = "
          f"{NDIRECT * BSIZE:,} bytes")
    print(f"  Indirect blocks: {NINDIRECT} x {BSIZE} = "
          f"{NINDIRECT * BSIZE:,} bytes")
    print(f"  Maximum:         ({NDIRECT} + {NINDIRECT}) x {BSIZE} = "
          f"{MAXFILE * BSIZE:,} bytes ({MAXFILE * BSIZE // 1024} KB)")
    print(f"\n  To support larger files, xv6 labs add double-indirect")
    print(f"  blocks: {NDIRECT - 1} direct + 1 indirect + 1 double-indirect")
    print(f"  = {(NDIRECT - 1) + NINDIRECT + NINDIRECT * NINDIRECT} blocks "
          f"= {((NDIRECT - 1) + NINDIRECT + NINDIRECT * NINDIRECT) * BSIZE // 1024:,} KB")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Physical Memory Allocator (kalloc/kfree) ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Process Table and Fork Simulation ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: xv6 File System Layout ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
