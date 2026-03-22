# 11. Git Internals

**Previous**: [Monorepo Management](./10_Monorepo_Management.md) | **Next**: [Git Bisect and Debugging](./12_Git_Bisect_and_Debugging.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Git's content-addressable object model and the four object types (blob, tree, commit, tag)
2. Navigate and understand the `.git` directory structure
3. Use plumbing commands (`hash-object`, `cat-file`, `ls-tree`, `rev-parse`, `update-ref`) to interact with Git at a low level
4. Explain how commits form a Directed Acyclic Graph (DAG) and how branches are simply pointers
5. Understand pack files, garbage collection, and how Git optimizes storage
6. Distinguish between plumbing and porcelain commands and know when to use each

---

Most Git users interact exclusively with high-level commands like `commit`, `push`, and `merge`. But underneath these friendly commands lies a remarkably elegant content-addressable filesystem. Understanding Git's internals transforms you from a Git user into someone who truly understands what Git is doing -- and more importantly, why things go wrong and how to fix them.

## Table of Contents
1. [The .git Directory](#1-the-git-directory)
2. [Git's Object Model](#2-gits-object-model)
3. [Content-Addressable Storage](#3-content-addressable-storage)
4. [Plumbing vs Porcelain](#4-plumbing-vs-porcelain)
5. [The DAG: How Commits Form History](#5-the-dag-how-commits-form-history)
6. [Pack Files and Garbage Collection](#6-pack-files-and-garbage-collection)
7. [Low-Level Command Reference](#7-low-level-command-reference)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. The .git Directory

When you run `git init`, Git creates a `.git` directory that contains everything Git needs. The working directory is just a checkout of one version; the real repository lives inside `.git`.

### 1.1 Directory Structure

```
.git/
├── HEAD                 # Points to current branch ref
├── config               # Repository-specific configuration
├── description          # Used by GitWeb (rarely modified)
├── index                # Staging area (binary file)
├── packed-refs          # Packed references for efficiency
├── objects/             # All content (blobs, trees, commits, tags)
│   ├── info/
│   └── pack/            # Pack files for compression
├── refs/                # Pointers to commit objects
│   ├── heads/           # Branch tips
│   ├── tags/            # Tag references
│   └── remotes/         # Remote tracking branches
├── hooks/               # Client/server-side hook scripts
├── info/                # Global excludes, etc.
│   └── exclude          # Like .gitignore but not committed
└── logs/                # Reflog entries
    ├── HEAD
    └── refs/
```

### 1.2 The HEAD File

`HEAD` is the simplest yet most important file. It tells Git which branch you are on.

```bash
# View HEAD contents
cat .git/HEAD
# ref: refs/heads/main

# When in detached HEAD state
cat .git/HEAD
# a1b2c3d4e5f6... (direct SHA-1 hash)
```

### 1.3 The Index (Staging Area)

The index is a binary file (`.git/index`) that stores the staging area. It sits between your working directory and the object database.

```bash
# View the index contents
git ls-files --stage
# 100644 ce013625030ba8dba906f756967f9e9ca394464a 0	README.md
# 100644 8baef1b4abc478178b004d62031cf7fe6db6f903 0	src/main.py

# The three columns are: mode, SHA-1, stage number, filename
# Stage 0 = normal, 1/2/3 = merge conflict stages
```

### 1.4 The refs Directory

Branches and tags are simply files containing SHA-1 hashes.

```bash
# A branch is just a file with a commit hash
cat .git/refs/heads/main
# e83c5163316f89bfbde7d9ab23ca2e25604af290

# A tag (lightweight) is identical
cat .git/refs/tags/v1.0
# e83c5163316f89bfbde7d9ab23ca2e25604af290

# Remote tracking branches
cat .git/refs/remotes/origin/main
# e83c5163316f89bfbde7d9ab23ca2e25604af290
```

---

## 2. Git's Object Model

Git has exactly four object types. Everything in Git -- every file, every directory snapshot, every commit -- is stored as one of these objects.

### 2.1 Blob Objects (File Content)

A blob stores the contents of a single file. It does NOT store the filename, permissions, or any metadata -- just raw content.

```bash
# Create a blob manually
echo "Hello, Git internals!" | git hash-object -w --stdin
# Returns: 8d0e41234f24b6da002d962a26c2495ea16a425f

# The object is stored at:
# .git/objects/8d/0e41234f24b6da002d962a26c2495ea16a425f
#              ^^ first 2 chars = directory
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ remaining chars = filename
```

Key properties of blobs:
- Content-only: two files with identical content share the same blob
- No filename stored: the tree object maps names to blobs
- Compressed with zlib

### 2.2 Tree Objects (Directory Snapshots)

A tree object represents a directory. It maps filenames to blobs (files) or other trees (subdirectories).

```bash
# View a tree object
git ls-tree HEAD
# 100644 blob ce013625030ba8dba906f756967f9e9ca394464a    README.md
# 040000 tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579    src
# 100644 blob 8baef1b4abc478178b004d62031cf7fe6db6f903    Makefile

# Mode values:
# 100644 = regular file
# 100755 = executable file
# 040000 = subdirectory (tree)
# 120000 = symbolic link
# 160000 = gitlink (submodule)
```

Tree structure visualized:

```
tree (root)
├── blob "README.md"  → ce0136...
├── blob "Makefile"   → 8baef1...
└── tree "src/"       → d8329f...
    ├── blob "main.py"    → a1b2c3...
    └── blob "utils.py"   → d4e5f6...
```

### 2.3 Commit Objects

A commit object ties everything together. It contains:
- A pointer to a tree object (the project snapshot)
- Zero or more parent commits (zero for initial commit, one for normal, two+ for merges)
- Author information (name, email, timestamp)
- Committer information (can differ from author)
- Commit message

```bash
# View a commit object
git cat-file -p HEAD
# tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579
# parent a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# author John Doe <john@example.com> 1709200000 +0900
# committer John Doe <john@example.com> 1709200000 +0900
#
# Add user authentication module
```

### 2.4 Tag Objects (Annotated Tags)

Annotated tags are full objects (unlike lightweight tags, which are just refs).

```bash
# Create an annotated tag
git tag -a v1.0 -m "Release version 1.0"

# View the tag object
git cat-file -p v1.0
# object e83c5163316f89bfbde7d9ab23ca2e25604af290
# type commit
# tag v1.0
# tagger John Doe <john@example.com> 1709200000 +0900
#
# Release version 1.0
```

### 2.5 Object Relationships

```
tag v1.0
  │
  ▼
commit c3 ─────► tree (root snapshot)
  │                ├── blob README.md
  │                └── tree src/
  │                     └── blob main.py
  ▼
commit c2 ─────► tree (older snapshot)
  │                ├── blob README.md
  │                └── tree src/
  │                     └── blob main.py
  ▼
commit c1 ─────► tree (initial snapshot)
                   └── blob README.md
```

---

## 3. Content-Addressable Storage

Git is fundamentally a content-addressable filesystem. The address (key) of every object is the SHA-1 hash of its content.

### 3.1 How Hashing Works

```bash
# Git hashes content with a header: "<type> <size>\0<content>"
# For a blob:
echo -n "Hello, Git internals!" | git hash-object --stdin
# 8d0e41234f24b6da002d962a26c2495ea16a425f

# Equivalent manual computation:
echo -en "blob 21\0Hello, Git internals!" | shasum
# 8d0e41234f24b6da002d962a26c2495ea16a425f  -

# The header format: "blob <content-length>\0"
```

### 3.2 Implications of Content-Addressable Storage

**Integrity**: Any corruption changes the hash, breaking the chain. Git detects corruption automatically.

```bash
# Verify repository integrity
git fsck
# Checking object directories: 100% (256/256), done.
# Checking objects: 100% (1234/1234), done.
```

**Deduplication**: Identical content is stored only once, regardless of how many files or commits reference it.

```bash
# Two files with identical content share one blob
echo "shared content" > file_a.txt
echo "shared content" > file_b.txt
git add file_a.txt file_b.txt

git ls-files --stage
# 100644 abc123... 0  file_a.txt
# 100644 abc123... 0  file_b.txt   # Same hash!
```

**Immutability**: Objects can never be modified. A "change" creates a new object with a new hash.

### 3.3 Object Storage on Disk

```bash
# Objects are stored as zlib-compressed files
# Path: .git/objects/<first-2-chars>/<remaining-38-chars>

# View raw object (Python)
python3 -c "
import zlib, sys
with open('.git/objects/8d/0e41234f24b6da002d962a26c2495ea16a425f', 'rb') as f:
    print(zlib.decompress(f.read()))
"
# b'blob 21\x00Hello, Git internals!'
```

---

## 4. Plumbing vs Porcelain

Git commands are divided into two categories, named after bathroom fixtures.

### 4.1 Porcelain Commands (User-Facing)

These are the commands you use daily:

```bash
git add          git commit       git push
git pull         git merge        git rebase
git log          git status       git diff
git branch       git checkout     git switch
git stash        git tag          git fetch
git clone        git remote       git reset
```

### 4.2 Plumbing Commands (Low-Level)

These are the building blocks that porcelain commands use internally:

```bash
# Object manipulation
git hash-object     # Compute object hash / write to database
git cat-file        # Display object content, type, or size
git write-tree      # Write index as tree object
git commit-tree     # Create commit from tree object
git mktag           # Create tag object

# Index manipulation
git update-index    # Register file contents in the index
git read-tree       # Read tree into the index
git ls-files        # Show information about files in the index

# Reference manipulation
git update-ref      # Safely update a reference
git symbolic-ref    # Read/update symbolic references (like HEAD)

# Inspection
git ls-tree         # List contents of a tree object
git rev-parse       # Parse revision identifiers
git rev-list        # List commit objects in reverse chronological order
git for-each-ref    # Iterate over references
git diff-tree       # Compare two tree objects
```

### 4.3 Building a Commit with Plumbing Commands

Here is how `git add` + `git commit` work under the hood:

```bash
# Step 1: Store file content as a blob
BLOB_HASH=$(echo "Hello World" | git hash-object -w --stdin)
echo "Blob: $BLOB_HASH"

# Step 2: Add blob to the index
git update-index --add --cacheinfo 100644 $BLOB_HASH hello.txt

# Step 3: Write the index as a tree object
TREE_HASH=$(git write-tree)
echo "Tree: $TREE_HASH"

# Step 4: Create a commit pointing to the tree
COMMIT_HASH=$(echo "Initial commit via plumbing" | \
  git commit-tree $TREE_HASH)
echo "Commit: $COMMIT_HASH"

# Step 5: Update the branch to point to the new commit
git update-ref refs/heads/main $COMMIT_HASH

# Step 6: Point HEAD to the branch
git symbolic-ref HEAD refs/heads/main

# Now 'git log' shows our commit!
git log --oneline
```

---

## 5. The DAG: How Commits Form History

### 5.1 Understanding the DAG

Git's history is a Directed Acyclic Graph (DAG). Each commit points to its parent(s), forming a graph that flows in one direction (backwards in time) and never loops.

```
# Linear history
A ← B ← C ← D  (main)

# Branch and merge
A ← B ← C ← F  (main)
     ↖         ↗
      D ← E     (feature)

# Multiple merge parents
A ← B ← C ← G  (main)
     ↖       ↗
      D ← E
     ↖       ↗
      F ─────   (hotfix)
```

### 5.2 Traversing the DAG

```bash
# List all commits reachable from HEAD
git rev-list HEAD
# Shows SHA-1 of every commit, newest first

# Count total commits
git rev-list --count HEAD
# 142

# Commits reachable from main but not feature
git rev-list main --not feature
# (commits on main after the branch point)

# Common ancestor of two branches
git merge-base main feature
# a1b2c3d4...

# Graph visualization
git log --all --graph --oneline --decorate
# * e83c516 (HEAD -> main) Merge feature
# |\
# | * a1b2c3d (feature) Add feature
# | * d4e5f6a Work on feature
# |/
# * 1234567 Base commit
```

### 5.3 Reachability

An object is **reachable** if Git can find it by following references. Unreachable objects are candidates for garbage collection.

```
refs/heads/main → commit C → commit B → commit A
                      │           │           │
                      ▼           ▼           ▼
                   tree T3     tree T2     tree T1
                      │           │           │
                      ▼           ▼           ▼
                   blobs...    blobs...    blobs...

All objects above are reachable from refs/heads/main.
```

```bash
# Find unreachable objects
git fsck --unreachable
# unreachable blob 8d0e412...
# unreachable commit a1b2c3d...

# Find dangling objects (unreachable and not referenced by other unreachable objects)
git fsck --dangling
```

### 5.4 Ancestry References

```bash
# Parent references
HEAD^       # First parent of HEAD
HEAD^2      # Second parent (only meaningful for merge commits)
HEAD^^      # Grandparent (first parent's first parent)

# Ancestor references
HEAD~1      # Same as HEAD^
HEAD~2      # Same as HEAD^^
HEAD~3      # Great-grandparent

# Combining them
HEAD~2^2    # Second parent of the grandparent

# Practical example
git log --oneline HEAD~5..HEAD   # Last 5 commits
git diff HEAD~3 HEAD             # Changes in last 3 commits
```

---

## 6. Pack Files and Garbage Collection

### 6.1 Loose vs Packed Objects

Initially, every object is stored as a separate file (loose object). This is inefficient for large repositories.

```bash
# Count loose objects
git count-objects
# 1234 objects, 5678 kilobytes

# Detailed statistics
git count-objects -v
# count: 1234           # loose objects
# size: 5678            # loose object disk size (KB)
# in-pack: 45678        # packed objects
# packs: 3              # number of pack files
# size-pack: 12345      # pack file disk size (KB)
# prune-packable: 0     # loose objects also in packs
# garbage: 0            # files in objects dir that aren't objects
# size-garbage: 0
```

### 6.2 How Pack Files Work

Pack files store objects using delta compression. Instead of storing every version of a file, Git stores one full version plus deltas (differences) from other versions.

```bash
# Manually create a pack file
git repack -a -d
# -a: pack all objects
# -d: remove redundant loose objects

# List pack file contents
git verify-pack -v .git/objects/pack/pack-*.idx
# SHA-1  type  size  size-in-pack  offset  depth  base-SHA-1
# a1b2c3 commit 234  180           12
# d4e5f6 tree   120  95            192
# 789abc blob   5678 1234          287     2      fedcba...
#                                          ^ delta depth
#                                                 ^ base object
```

### 6.3 Garbage Collection

```bash
# Run garbage collection
git gc
# Enumerating objects: 1234, done.
# Counting objects: 100% (1234/1234), done.
# Delta compression using up to 8 threads
# Compressing objects: 100% (567/567), done.
# Writing objects: 100% (1234/1234), done.

# Aggressive GC (slower, better compression)
git gc --aggressive

# Auto GC (runs when thresholds are met)
git gc --auto
# Triggers when loose objects > gc.auto (default 6700)
# or when packs > gc.autoPackLimit (default 50)
```

### 6.4 Pruning

```bash
# Remove unreachable objects older than 2 weeks (default)
git prune

# Preview what would be removed
git prune --dry-run

# Remove immediately (dangerous -- no grace period)
git prune --expire=now

# Reflog expiration (prerequisite for pruning)
git reflog expire --expire=90.days --all
git gc --prune=now
```

### 6.5 Removing Large Files from History

When a large file is accidentally committed, even deleting it doesn't reclaim space because the blob remains in history.

```bash
# Find large objects
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  grep ^blob | sort -k3 -n -r | head -10
# blob a1b2c3d4 104857600 data/huge_file.bin

# Remove with git-filter-repo (recommended over filter-branch)
pip install git-filter-repo
git filter-repo --invert-paths --path data/huge_file.bin

# Force garbage collection after rewriting history
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## 7. Low-Level Command Reference

### 7.1 git hash-object

```bash
# Compute SHA-1 without storing
echo "test content" | git hash-object --stdin
# d670460b4b4aece5915caf5c68d12f560a9fe3e4

# Compute and store in object database
echo "test content" | git hash-object -w --stdin

# Hash a file
git hash-object myfile.txt

# Hash and store a file
git hash-object -w myfile.txt
```

### 7.2 git cat-file

```bash
# Display object content
git cat-file -p HEAD          # Pretty-print any object
git cat-file -p HEAD:file.txt # Content of file at HEAD

# Display object type
git cat-file -t HEAD
# commit

# Display object size
git cat-file -s HEAD
# 234

# Batch mode (efficient for many objects)
echo "HEAD" | git cat-file --batch
echo "HEAD" | git cat-file --batch-check  # Type and size only
```

### 7.3 git ls-tree

```bash
# List tree contents (non-recursive)
git ls-tree HEAD
# 100644 blob ce01362... README.md
# 040000 tree d8329fc... src

# Recursive listing (all files)
git ls-tree -r HEAD
# 100644 blob ce01362... README.md
# 100644 blob a1b2c3d... src/main.py
# 100644 blob d4e5f6a... src/utils.py

# Show only names
git ls-tree --name-only HEAD

# List specific subdirectory
git ls-tree HEAD src/

# Include object size
git ls-tree -l HEAD
# 100644 blob ce01362...    1234    README.md
```

### 7.4 git rev-parse

```bash
# Convert symbolic name to SHA-1
git rev-parse HEAD
# e83c5163316f89bfbde7d9ab23ca2e25604af290

git rev-parse main
git rev-parse HEAD~3
git rev-parse v1.0

# Show the .git directory
git rev-parse --git-dir
# .git

# Show the repository root
git rev-parse --show-toplevel
# /home/user/project

# Check if inside a Git repository
git rev-parse --is-inside-work-tree
# true

# Parse revision ranges
git rev-parse main...feature   # Symmetric difference
git rev-parse main..feature    # In feature but not main
```

### 7.5 git update-ref

```bash
# Set a reference to a specific commit
git update-ref refs/heads/new-branch $COMMIT_HASH

# Delete a reference
git update-ref -d refs/heads/old-branch

# Safe update (verify old value first)
git update-ref refs/heads/main $NEW_HASH $OLD_HASH
# Fails if main doesn't point to $OLD_HASH (prevents race conditions)

# Create a lightweight tag
git update-ref refs/tags/v2.0 $COMMIT_HASH
```

### 7.6 git for-each-ref

```bash
# List all references with formatting
git for-each-ref --format='%(refname:short) %(objecttype) %(objectname:short)' refs/heads/
# main commit e83c516
# feature commit a1b2c3d
# dev commit d4e5f6a

# Sort by committer date
git for-each-ref --sort=-committerdate --format='%(refname:short) %(committerdate:relative)' refs/heads/
# feature 2 hours ago
# main 1 day ago
# dev 3 days ago

# Find merged branches
git for-each-ref --merged=main refs/heads/ --format='%(refname:short)'
```

---

## 8. Practice Exercises

### Exercise 1: Explore the Object Database

```bash
# 1. Create a new repository and make a commit
git init internals-lab && cd internals-lab
echo "first file" > hello.txt
git add hello.txt
git commit -m "Initial commit"

# 2. Tasks:
# a) Find the blob hash of hello.txt using git ls-files --stage
# b) Use git cat-file -p to view the blob content
# c) Use git cat-file -t to confirm the object type
# d) Use git ls-tree HEAD to see the root tree
# e) Use git cat-file -p on the commit object to see its tree pointer
# f) Manually verify the hash: echo -en "blob 10\0first file" | shasum
```

### Exercise 2: Build a Commit with Plumbing Commands Only

```bash
# Create a complete commit without using any porcelain commands.
# Use only: hash-object, update-index, write-tree, commit-tree, update-ref

# 1. Create a blob:
#    echo "plumbing test" | git hash-object -w --stdin
#
# 2. Add to index:
#    git update-index --add --cacheinfo 100644 <blob-hash> test.txt
#
# 3. Write tree:
#    git write-tree
#
# 4. Create commit:
#    echo "Plumbing commit" | git commit-tree <tree-hash> -p HEAD
#
# 5. Update branch:
#    git update-ref refs/heads/main <commit-hash>
#
# 6. Verify with git log
```

### Exercise 3: Investigate Object Relationships

```bash
# 1. Create a repository with 3 commits, each modifying different files
# 2. Draw the DAG by hand using git rev-list and git cat-file
# 3. For each commit:
#    a) Find its tree hash
#    b) List all blobs in that tree
#    c) Identify which blobs are shared between commits
# 4. Run git count-objects before and after git gc
# 5. Verify that identical content shares the same blob hash
```

### Exercise 4: Pack File Analysis

```bash
# 1. Create a repository with a large text file
dd if=/dev/urandom bs=1024 count=100 | base64 > large_file.txt
git add large_file.txt && git commit -m "Add large file"

# 2. Make 5 small modifications to large_file.txt, committing each
# 3. Run git count-objects -v to see loose object stats
# 4. Run git gc and check git count-objects -v again
# 5. Use git verify-pack -v to examine delta chains
# 6. Compare the total size of loose objects vs packed objects
# 7. Explain why the packed size is smaller (delta compression)
```

### Exercise 5: Recovery with Plumbing Commands

```bash
# 1. Create a commit, note its hash
# 2. Reset --hard to the previous commit (simulating an accident)
# 3. The "lost" commit still exists as a dangling object
# 4. Find it using: git fsck --dangling
# 5. Recover it using: git update-ref refs/heads/recovered <hash>
# 6. Verify recovery with git log recovered
```

---

## Next Steps

- [Git Bisect and Debugging](./12_Git_Bisect_and_Debugging.md) - Use Git for debugging
- [Git Documentation - Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain) - Official reference
- [Pro Git Book - Chapter 10](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain) - Deep dive into internals

## References

- [Git Internals - Pro Git Book](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)
- [Git Object Model](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
- [Git Packfiles](https://git-scm.com/book/en/v2/Git-Internals-Packfiles)
- [git-filter-repo](https://github.com/newren/git-filter-repo)
- [SHA-1 and Git](https://git-scm.com/docs/hash-function-transition)

---

[← Previous: Monorepo Management](10_Monorepo_Management.md) | [Next: Git Bisect and Debugging →](12_Git_Bisect_and_Debugging.md) | [Table of Contents](00_Overview.md)
