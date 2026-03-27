"""
Filesystem Simulation

Demonstrates:
- Inode-based filesystem (Unix-style)
- FAT (File Allocation Table) filesystem
- Directory structure and path resolution

Theory:
- Inode: fixed-size structure storing file metadata + block pointers.
  Direct, indirect, double-indirect pointers for scalability.
- FAT: table mapping each block to the next block in the chain.
  Simple but poor random access and fragmentation issues.
- Directories: special files mapping names to inodes (Unix) or
  starting blocks (FAT).

Adapted from OS Theory Lesson 17.
"""

from dataclasses import dataclass, field
from datetime import datetime


# ── Inode-Based Filesystem ──────────────────────────────────────────────

@dataclass
class Inode:
    """Unix-style inode."""
    ino: int
    is_dir: bool = False
    size: int = 0
    blocks: list[int] = field(default_factory=list)
    link_count: int = 1
    owner: str = "root"
    permissions: str = "rwxr-xr-x"
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __repr__(self) -> str:
        kind = "d" if self.is_dir else "-"
        return (f"Inode(ino={self.ino}, {kind}{self.permissions}, "
                f"size={self.size}, blocks={self.blocks})")


class InodeFS:
    """Simplified inode-based filesystem simulator."""

    def __init__(self, total_blocks: int = 64, block_size: int = 4096):
        self.block_size = block_size
        self.total_blocks = total_blocks
        # Blocks 0 and 1 are reserved: block 0 holds the superblock (filesystem
        # metadata like size, free count), block 1 holds the root directory.
        # Separating metadata from data blocks allows the FS to locate the root
        # without any other information.
        self.free_blocks = list(range(2, total_blocks))
        self.inodes: dict[int, Inode] = {}
        self.next_ino = 0

        # Directory entries are stored separately from inodes — this mirrors
        # real Unix filesystems where a directory's data blocks contain
        # (name, inode) pairs, and the inode itself only stores metadata.
        # This separation enables hard links: multiple names can point to
        # the same inode.
        self.dir_entries: dict[int, dict[str, int]] = {}

        # Create root directory
        root = self._alloc_inode(is_dir=True)
        self.root_ino = root.ino
        self.dir_entries[root.ino] = {".": root.ino, "..": root.ino}

    def _alloc_inode(self, is_dir: bool = False) -> Inode:
        ino = self.next_ino
        self.next_ino += 1
        inode = Inode(ino=ino, is_dir=is_dir)
        self.inodes[ino] = inode
        return inode

    def _alloc_block(self) -> int | None:
        if not self.free_blocks:
            return None
        return self.free_blocks.pop(0)

    def _resolve_path(self, path: str) -> int | None:
        """Resolve path to inode number."""
        parts = [p for p in path.strip("/").split("/") if p]
        current = self.root_ino

        for part in parts:
            if current not in self.dir_entries:
                return None
            entries = self.dir_entries[current]
            if part not in entries:
                return None
            current = entries[part]
        return current

    def mkdir(self, path: str) -> bool:
        """Create a directory."""
        parts = path.strip("/").split("/")
        name = parts[-1]
        parent_path = "/" + "/".join(parts[:-1])
        parent_ino = self._resolve_path(parent_path)

        if parent_ino is None:
            print(f"    Error: parent '{parent_path}' not found")
            return False

        if name in self.dir_entries.get(parent_ino, {}):
            print(f"    Error: '{name}' already exists")
            return False

        new_dir = self._alloc_inode(is_dir=True)
        self.dir_entries[parent_ino][name] = new_dir.ino
        # Every directory contains "." (self) and ".." (parent) entries — these
        # aren't just convenience aliases; ".." is how the kernel resolves
        # relative paths like "cd ..", and "." is why a new directory increases
        # the parent's link_count (the parent gains a ".." back-reference)
        self.dir_entries[new_dir.ino] = {
            ".": new_dir.ino,
            "..": parent_ino,
        }
        self.inodes[parent_ino].link_count += 1
        return True

    def create_file(self, path: str, size: int) -> bool:
        """Create a file with given size (allocates blocks)."""
        parts = path.strip("/").split("/")
        name = parts[-1]
        parent_path = "/" + "/".join(parts[:-1])
        parent_ino = self._resolve_path(parent_path)

        if parent_ino is None:
            print(f"    Error: parent '{parent_path}' not found")
            return False

        new_file = self._alloc_inode(is_dir=False)
        new_file.size = size

        # Allocate blocks
        n_blocks = (size + self.block_size - 1) // self.block_size
        for _ in range(n_blocks):
            block = self._alloc_block()
            if block is None:
                print(f"    Error: disk full")
                return False
            new_file.blocks.append(block)

        self.dir_entries[parent_ino][name] = new_file.ino
        return True

    def ls(self, path: str = "/") -> None:
        """List directory contents."""
        ino = self._resolve_path(path)
        if ino is None:
            print(f"    Error: '{path}' not found")
            return

        inode = self.inodes[ino]
        if not inode.is_dir:
            print(f"    {path}: not a directory")
            return

        entries = self.dir_entries.get(ino, {})
        print(f"  ls {path}:")
        for name, child_ino in sorted(entries.items()):
            if name in (".", ".."):
                continue
            child = self.inodes[child_ino]
            kind = "d" if child.is_dir else "-"
            print(f"    {kind}{child.permissions}  "
                  f"{child.link_count:>2}  {child.owner:<6}  "
                  f"{child.size:>8}  {name}")

    def stat(self, path: str) -> None:
        """Show inode details for a path."""
        ino = self._resolve_path(path)
        if ino is None:
            print(f"    Error: '{path}' not found")
            return
        inode = self.inodes[ino]
        print(f"  stat {path}:")
        print(f"    Inode: {inode.ino}")
        print(f"    Type:  {'directory' if inode.is_dir else 'file'}")
        print(f"    Size:  {inode.size}")
        print(f"    Blocks: {inode.blocks}")
        print(f"    Links: {inode.link_count}")


def demo_inode_fs():
    """Demonstrate inode-based filesystem."""
    print("=" * 60)
    print("INODE-BASED FILESYSTEM")
    print("=" * 60)

    fs = InodeFS(total_blocks=64, block_size=4096)

    print("\n  Creating directory structure...")
    fs.mkdir("/home")
    fs.mkdir("/home/alice")
    fs.mkdir("/etc")
    fs.mkdir("/var")
    fs.mkdir("/var/log")

    print("  Creating files...")
    fs.create_file("/etc/passwd", 2048)
    fs.create_file("/etc/hosts", 512)
    fs.create_file("/home/alice/notes.txt", 8500)
    fs.create_file("/home/alice/photo.jpg", 150000)
    fs.create_file("/var/log/syslog", 32000)

    print()
    fs.ls("/")
    print()
    fs.ls("/home/alice")
    print()
    fs.stat("/home/alice/photo.jpg")

    # Show disk usage
    used = fs.total_blocks - len(fs.free_blocks)
    print(f"\n  Disk usage: {used}/{fs.total_blocks} blocks "
          f"({used * fs.block_size / 1024:.0f} KB / "
          f"{fs.total_blocks * fs.block_size / 1024:.0f} KB)")


# ── FAT Filesystem ──────────────────────────────────────────────────────

class FATFS:
    """Simplified FAT filesystem simulator.

    FAT entry values:
    - 0: free block
    - -1: end of chain
    - n: next block in chain
    """

    FREE = 0
    END = -1

    def __init__(self, total_blocks: int = 32, block_size: int = 512):
        self.block_size = block_size
        self.total_blocks = total_blocks
        # FAT: index = block number, value = next block or special
        self.fat = [self.FREE] * total_blocks
        self.fat[0] = self.END  # block 0 reserved for FAT

        # Root directory: list of (name, start_block, size, is_dir)
        self.root: list[dict] = []

    def _alloc_blocks(self, n: int) -> list[int] | None:
        """Allocate n contiguous-in-chain blocks."""
        free = [i for i in range(1, self.total_blocks) if self.fat[i] == self.FREE]
        if len(free) < n:
            return None
        allocated = free[:n]

        # Chain them as a linked list via FAT entries — each block's FAT slot
        # points to the next block in the file. This linked structure means
        # sequential reads require following the chain (O(n) to reach block k),
        # which is FAT's main disadvantage over inode-based direct/indirect
        # block pointers that offer O(1) random access.
        for i in range(len(allocated) - 1):
            self.fat[allocated[i]] = allocated[i + 1]
        self.fat[allocated[-1]] = self.END

        return allocated

    def create_file(self, name: str, size: int) -> bool:
        """Create a file in the root directory."""
        n_blocks = max(1, (size + self.block_size - 1) // self.block_size)
        blocks = self._alloc_blocks(n_blocks)
        if blocks is None:
            print(f"    Error: disk full")
            return False

        self.root.append({
            "name": name,
            "start": blocks[0],
            "size": size,
        })
        return True

    def delete_file(self, name: str) -> bool:
        """Delete a file, freeing its blocks."""
        for i, entry in enumerate(self.root):
            if entry["name"] == name:
                # Walk the chain and free blocks
                block = entry["start"]
                while block != self.END:
                    next_block = self.fat[block]
                    self.fat[block] = self.FREE
                    block = next_block
                self.root.pop(i)
                return True
        return False

    def read_chain(self, name: str) -> list[int]:
        """Return the block chain for a file."""
        for entry in self.root:
            if entry["name"] == name:
                chain = []
                block = entry["start"]
                while block != self.END:
                    chain.append(block)
                    block = self.fat[block]
                return chain
        return []

    def display_fat(self) -> None:
        """Display the FAT table."""
        print("  FAT Table:")
        header = "    Block: " + " ".join(f"{i:>3}" for i in range(self.total_blocks))
        values = "    Value: " + " ".join(
            f"{'END' if v == self.END else 'FRE' if v == self.FREE else f'{v:>3}'}"
            for v in self.fat
        )
        print(header)
        print(values)

    def display_files(self) -> None:
        """Display root directory."""
        print("  Root Directory:")
        print(f"    {'Name':<15} {'Size':>8} {'Start':>6} {'Chain'}")
        print(f"    {'-'*15} {'-'*8} {'-'*6} {'-'*20}")
        for entry in self.root:
            chain = self.read_chain(entry["name"])
            chain_str = " → ".join(str(b) for b in chain)
            print(f"    {entry['name']:<15} {entry['size']:>8} "
                  f"{entry['start']:>6} {chain_str}")


def demo_fat():
    """Demonstrate FAT filesystem."""
    print("\n" + "=" * 60)
    print("FAT FILESYSTEM")
    print("=" * 60)

    fs = FATFS(total_blocks=16, block_size=512)

    print("\n  Creating files...")
    fs.create_file("readme.txt", 300)
    fs.create_file("data.csv", 1800)
    fs.create_file("image.bmp", 1200)
    fs.create_file("log.txt", 600)

    print()
    fs.display_files()
    print()
    fs.display_fat()

    # Delete middle file → fragmentation
    print("\n  Deleting data.csv...")
    fs.delete_file("data.csv")

    # Create new file → fills gaps
    print("  Creating report.pdf (1500 bytes)...")
    fs.create_file("report.pdf", 1500)

    print()
    fs.display_files()
    print()
    fs.display_fat()

    # Count fragmentation
    free_blocks = sum(1 for v in fs.fat if v == FATFS.FREE)
    used = fs.total_blocks - free_blocks - 1  # -1 for reserved block 0
    print(f"\n  Disk: {used} used, {free_blocks} free out of {fs.total_blocks} blocks")


if __name__ == "__main__":
    demo_inode_fs()
    demo_fat()
