"""
Exercises for Lesson 16: File System Basics
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers file attribute interpretation, system call sequences, hard vs symbolic links,
directory path canonicalization, and open file table behavior across fork().
"""


# === Exercise 1: File Attribute Interpretation ===
# Problem: Interpret each field from an `ls -l` output line.

def exercise_1():
    """Interpret ls -l output fields."""
    ls_output = "-rwxr-x--- 2 alice developers 8192 Mar 15 14:30 script.sh"

    print(f"ls -l output:\n  {ls_output}\n")

    # Parse each field
    fields = {
        "File type":      ("-", "Regular file (d=directory, l=symlink, b=block, c=char, p=pipe, s=socket)"),
        "Owner perms":    ("rwx", "Read + Write + Execute for owner (alice)"),
        "Group perms":    ("r-x", "Read + Execute for group (developers); no Write"),
        "Others perms":   ("---", "No permissions for others"),
        "Hard link count": ("2", "Two directory entries (hard links) point to this inode"),
        "Owner":          ("alice", "File owner (UID mapped to username)"),
        "Group":          ("developers", "File group (GID mapped to group name)"),
        "Size":           ("8192", "File size in bytes (8KB)"),
        "Modification":   ("Mar 15 14:30", "Last modification timestamp (mtime)"),
        "Filename":       ("script.sh", "File name in the directory entry"),
    }

    print(f"  {'Field':<18} {'Value':<15} {'Interpretation'}")
    print("  " + "-" * 75)
    for field, (value, interp) in fields.items():
        print(f"  {field:<18} {value:<15} {interp}")

    # Numeric permission
    print(f"\nNumeric (octal) permissions:")
    perms = {"rwx": 7, "r-x": 5, "---": 0, "rw-": 6, "r--": 4, "--x": 1, "-w-": 2, "-wx": 3}
    owner_perm = "rwx"
    group_perm = "r-x"
    other_perm = "---"
    octal = f"{perms[owner_perm]}{perms[group_perm]}{perms[other_perm]}"
    print(f"  {owner_perm} = {perms[owner_perm]}, {group_perm} = {perms[group_perm]}, {other_perm} = {perms[other_perm]}")
    print(f"  Numeric mode: {octal}")
    print(f"  chmod {octal} script.sh   # equivalent command")

    # Additional insight on hard link count
    print(f"\nHard link count = 2 means:")
    print(f"  There are 2 filenames that refer to the same inode.")
    print(f"  Deleting one name (rm) reduces the count to 1.")
    print(f"  The file data is only freed when the count reaches 0")
    print(f"  AND no process has the file open.")


# === Exercise 2: System Call Sequence ===
# Problem: Write the system call sequence to append "Hello World" to /tmp/log.txt.

def exercise_2():
    """Show the system call sequence for file append operation."""
    print("Task: Append 'Hello World' to /tmp/log.txt\n")

    print("System call sequence (C code):\n")
    print("```c")
    print('#include <fcntl.h>')
    print('#include <unistd.h>')
    print('#include <string.h>')
    print('#include <stdlib.h>')
    print('#include <stdio.h>')
    print("")
    print("int main() {")
    print('    // 1. open() -- O_APPEND ensures writes go to end of file')
    print('    //            -- O_CREAT creates file if it does not exist')
    print('    //            -- 0644 sets permissions: rw-r--r--')
    print('    int fd = open("/tmp/log.txt", O_WRONLY | O_APPEND | O_CREAT, 0644);')
    print("    if (fd == -1) {")
    print('        perror("open failed");')
    print("        exit(1);")
    print("    }")
    print("")
    print('    // 2. write() -- write the message to the file')
    print('    const char* msg = "Hello World\\n";')
    print("    ssize_t written = write(fd, msg, strlen(msg));")
    print("    if (written == -1) {")
    print('        perror("write failed");')
    print("    }")
    print("")
    print("    // 3. close() -- release the file descriptor")
    print("    close(fd);")
    print("    return 0;")
    print("}")
    print("```\n")

    # Trace the kernel operations
    print("What happens inside the kernel for each call:\n")
    calls = [
        ("open()", [
            "Search directory tree: / -> tmp -> log.txt",
            "Check permissions against process UID/GID",
            "Allocate file descriptor in process's fd table",
            "Create/find entry in system-wide open file table",
            "Set file position to end (O_APPEND)",
            "Return fd number to user space",
        ]),
        ("write()", [
            "Validate fd (is it open? is it writable?)",
            "O_APPEND: atomically seek to end before writing",
            "Copy data from user buffer to kernel buffer (page cache)",
            "Mark page cache pages as dirty",
            "Update file size in inode if file grew",
            "Update mtime (modification time) in inode",
            "Return number of bytes written",
        ]),
        ("close()", [
            "Release process's fd table entry",
            "Decrement reference count in system open file table",
            "If ref count == 0: release open file table entry",
            "If no more references to inode: may write dirty pages to disk",
            "Flush file metadata if needed",
        ]),
    ]

    for syscall, steps in calls:
        print(f"  {syscall}:")
        for i, step in enumerate(steps, 1):
            print(f"    {i}. {step}")
        print()


# === Exercise 3: Hard Link vs Symbolic Link ===
# Problem: Predict behavior when original file is deleted.

def exercise_3():
    """Demonstrate hard link vs symbolic link behavior after file deletion."""
    print("Scenario:\n")
    print("  $ echo 'original' > file.txt")
    print("  $ ln file.txt hardlink.txt        # Hard link")
    print("  $ ln -s file.txt symlink.txt      # Symbolic link")
    print("  $ rm file.txt")
    print("  $ cat hardlink.txt")
    print("  $ cat symlink.txt\n")

    print("--- What happens at each step ---\n")

    print("1. echo 'original' > file.txt")
    print("   - Creates inode (e.g., inode 12345) with data 'original'")
    print("   - Creates directory entry: 'file.txt' -> inode 12345")
    print("   - inode 12345 link_count = 1\n")

    print("2. ln file.txt hardlink.txt")
    print("   - Creates new directory entry: 'hardlink.txt' -> inode 12345")
    print("   - SAME inode as file.txt (same inode number)")
    print("   - inode 12345 link_count = 2")
    print("   - Both names are EQUAL -- neither is 'the original'\n")

    print("3. ln -s file.txt symlink.txt")
    print("   - Creates NEW inode (e.g., inode 67890)")
    print("   - inode 67890 stores the string 'file.txt' (the target path)")
    print("   - Creates directory entry: 'symlink.txt' -> inode 67890")
    print("   - inode 12345 link_count still = 2 (symlinks don't affect it)\n")

    print("4. rm file.txt")
    print("   - Removes directory entry 'file.txt'")
    print("   - inode 12345 link_count decremented: 2 -> 1")
    print("   - link_count > 0, so data is NOT freed")
    print("   - 'hardlink.txt' still points to inode 12345\n")

    print("5. cat hardlink.txt")
    print("   Output: original")
    print("   - hardlink.txt -> inode 12345, which still has the data")
    print("   - Hard links are direct references to the inode")
    print("   - The data survives as long as any hard link exists\n")

    print("6. cat symlink.txt")
    print("   Output: cat: symlink.txt: No such file or directory")
    print("   - symlink.txt -> inode 67890 -> path 'file.txt'")
    print("   - OS tries to resolve 'file.txt', but it was deleted")
    print("   - This is a DANGLING (broken) symbolic link\n")

    print("Comparison summary:")
    print(f"  {'Property':<30} {'Hard Link':<25} {'Symbolic Link'}")
    print("  " + "-" * 75)
    comparisons = [
        ("Inode", "Same as original", "Different (stores path)"),
        ("After rm original", "Data preserved", "Dangling link (broken)"),
        ("Cross-filesystem", "No", "Yes"),
        ("Link to directories", "No (usually)", "Yes"),
        ("Space overhead", "Directory entry only", "New inode + path string"),
        ("Link count affected", "Yes (+1)", "No"),
        ("Relative paths", "N/A (inode ref)", "Relative to symlink location"),
    ]
    for prop, hard, sym in comparisons:
        print(f"  {prop:<30} {hard:<25} {sym}")


# === Exercise 4: Directory Path Canonicalization ===
# Problem: Canonicalize /home/user/docs/../code/./main.c

def exercise_4():
    """Canonicalize a file path by resolving . and .. components."""
    path = "/home/user/docs/../code/./main.c"

    print(f"Input path: {path}\n")

    # Split and process components
    components = path.split("/")
    stack = []

    print("Step-by-step resolution:")
    step = 0
    for component in components:
        if component == "" or component == ".":
            if component == ".":
                step += 1
                print(f"  Step {step}: '{component}' -> current directory, skip")
            continue
        elif component == "..":
            step += 1
            if stack:
                removed = stack.pop()
                print(f"  Step {step}: '..' -> go up, remove '{removed}'")
                print(f"           Stack: /{'/'.join(stack)}")
            else:
                print(f"  Step {step}: '..' -> already at root, ignore")
        else:
            step += 1
            stack.append(component)
            print(f"  Step {step}: '{component}' -> push to stack")
            print(f"           Stack: /{'/'.join(stack)}")

    canonical = "/" + "/".join(stack)
    print(f"\nCanonical path: {canonical}\n")

    print("Detailed trace:")
    print("  /home/user/docs/../code/./main.c")
    print("  1. Start at /")
    print("  2. Enter home:  /home")
    print("  3. Enter user:  /home/user")
    print("  4. Enter docs:  /home/user/docs")
    print("  5. '..' -> up:  /home/user")
    print("  6. Enter code:  /home/user/code")
    print("  7. '.' -> stay: /home/user/code")
    print("  8. main.c file: /home/user/code/main.c\n")

    print("Verification:")
    print("  $ realpath /home/user/docs/../code/./main.c")
    print("  /home/user/code/main.c")

    # Additional examples
    print("\n--- Additional path canonicalization examples ---\n")
    test_paths = [
        "/a/b/c/../../d",
        "/a/./b/./c/./d",
        "/a/b/../../../c",
        "/../a/b",
        "/a/b/c/../d/../e",
    ]

    for p in test_paths:
        parts = p.split("/")
        s = []
        for part in parts:
            if part == "" or part == ".":
                continue
            elif part == "..":
                if s:
                    s.pop()
            else:
                s.append(part)
        result = "/" + "/".join(s)
        print(f"  {p:<35} -> {result}")


# === Exercise 5: Open File Table and fork() ===
# Problem: Determine final file size when parent and child both write 100 bytes.

def exercise_5():
    """Analyze shared open file table entries after fork()."""
    write_size = 100  # bytes each

    print("Scenario: Parent opens a file, then fork(). Both write 100 bytes.\n")

    print("Step-by-step analysis:\n")

    print("1. Parent calls open():")
    print("   - Process A's fd table: fd 3 -> open file entry #42")
    print("   - Open file entry #42: offset=0, refcount=1, inode ptr\n")

    print("2. Parent calls fork():")
    print("   - Child B inherits fd table (including fd 3)")
    print("   - Both A and B share the SAME open file entry #42")
    print("   - Open file entry #42: offset=0, refcount=2")
    print("   - This is the key: they share the OFFSET!\n")

    print("3. Parent writes 100 bytes:")
    print("   - Writes at offset 0, advances offset to 100")
    print("   - Open file entry #42: offset=100")
    print("   - File size: 100 bytes\n")

    print("4. Child writes 100 bytes:")
    print("   - Reads SHARED offset: 100")
    print("   - Writes at offset 100, advances offset to 200")
    print("   - Open file entry #42: offset=200")
    print("   - File size: 200 bytes\n")

    total_size = write_size * 2
    print(f"Final file size: {total_size} bytes\n")
    print(f"  The writes are sequential, not overlapping,")
    print(f"  because the shared offset advances for both processes.\n")

    print("--- What if the child independently open()ed the same file? ---\n")

    print("  If the child calls open() separately instead of inheriting fd:")
    print("  - Parent's fd 3 -> open file entry #42 (offset=0)")
    print("  - Child's fd 3 -> open file entry #99 (offset=0, SEPARATE)")
    print("  - Each has its own offset!")
    print("")
    print("  Parent writes 100 bytes at offset 0 -> file contains 100 bytes")
    print("  Child writes 100 bytes at offset 0 -> OVERWRITES parent's data!")
    print(f"  Final file size: {write_size} bytes (data from whichever wrote last)\n")

    print("Practical implication:")
    print("  After fork(), shared fd entries allow coordinated writing (e.g., logs).")
    print("  For independent writing to the same file, use O_APPEND to ensure")
    print("  each write atomically moves to end-of-file before writing.")
    print("  This prevents data loss from offset race conditions.")

    print(f"\n  Diagram:")
    print(f"  fork() scenario (shared entry):")
    print(f"    Process A fd[3] ──┐")
    print(f"                      ├──> Open File Entry #42 (shared offset)")
    print(f"    Process B fd[3] ──┘         │")
    print(f"                                └──> Inode (file data)")
    print(f"")
    print(f"  Independent open() scenario (separate entries):")
    print(f"    Process A fd[3] ──> Open File Entry #42 (offset A) ──┐")
    print(f"                                                          ├──> Inode")
    print(f"    Process B fd[3] ──> Open File Entry #99 (offset B) ──┘")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: File Attribute Interpretation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: System Call Sequence ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Hard Link vs Symbolic Link ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Directory Path Canonicalization ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Open File Table and fork() ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
