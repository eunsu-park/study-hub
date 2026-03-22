#!/usr/bin/env bash
# =============================================================================
# 10_git_internals.sh — Git Object Model & Internals Demo
# =============================================================================
# Demonstrates: git cat-file, ls-tree, rev-parse, hash-object, fsck
#
# Git is fundamentally a content-addressable filesystem. Every piece of data
# (files, directories, commits) is stored as an object identified by its
# SHA-1 hash. Understanding this model demystifies Git's behavior.
#
# Object types:
#   blob   — File contents (no filename, just data)
#   tree   — Directory listing (maps names to blobs/trees)
#   commit — Snapshot pointer (tree + parent + author + message)
#   tag    — Annotated tag (points to a commit with metadata)
#
# Usage: bash 10_git_internals.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Git Internals Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

cd "${WORK_DIR}"
git init
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

# ===========================================================================
# 1. The .git directory — Git's brain
# ===========================================================================
echo "--- 1. Inside the .git directory ---"
echo ""
echo "Key directories and files:"
echo "  .git/objects/   — Object database (blobs, trees, commits)"
echo "  .git/refs/      — Branch and tag pointers"
echo "  .git/HEAD       — Current branch pointer"
echo "  .git/index      — Staging area (binary file)"
echo "  .git/config     — Repository configuration"
echo ""

echo "Current HEAD contents:"
cat .git/HEAD
echo ""

# ===========================================================================
# 2. Creating objects manually with hash-object
# ===========================================================================
# Why: Understanding hash-object shows that Git stores content, not files.
# Two files with identical content share the same blob object.
echo "--- 2. Creating blob objects ---"

# Write content directly to the object database
# -w: actually write (not just compute hash), --stdin: read from stdin
BLOB_SHA=$(echo "Hello, Git internals!" | git hash-object -w --stdin)
echo "Created blob object: ${BLOB_SHA}"

# Inspect the object type and content
echo "Object type:"
git cat-file -t "${BLOB_SHA}"

echo "Object size:"
git cat-file -s "${BLOB_SHA}"

echo "Object content:"
git cat-file -p "${BLOB_SHA}"
echo ""

# Demonstrate content addressing — same content = same hash
BLOB_SHA2=$(echo "Hello, Git internals!" | git hash-object --stdin)
echo "Same content produces same hash: ${BLOB_SHA2}"
echo "Hashes match: $( [[ "${BLOB_SHA}" == "${BLOB_SHA2}" ]] && echo 'YES' || echo 'NO' )"
echo ""

# ===========================================================================
# 3. Commits, trees, and blobs — the object graph
# ===========================================================================
echo "--- 3. Building the object graph ---"

# Create some files and commit them
mkdir -p src
echo "print('hello')" > src/main.py
echo "# Config" > src/config.py
echo "README content" > README.md

git add .
git commit -m "Initial commit: project structure"
echo ""

# Get the commit SHA
COMMIT_SHA=$(git rev-parse HEAD)
echo "Commit SHA: ${COMMIT_SHA}"
echo ""

# Inspect the commit object
echo "--- 3a. Commit object ---"
echo "A commit stores: tree reference, parent(s), author, committer, message"
echo ""
git cat-file -p "${COMMIT_SHA}"
echo ""

# Extract the tree SHA from the commit
TREE_SHA=$(git cat-file -p "${COMMIT_SHA}" | grep '^tree ' | awk '{print $2}')
echo "--- 3b. Tree object (root directory) ---"
echo "Tree SHA: ${TREE_SHA}"
echo "A tree maps filenames to blob/tree SHAs with file mode permissions."
echo ""
git cat-file -p "${TREE_SHA}"
echo ""

# Inspect the src/ subtree
SRC_TREE_SHA=$(git cat-file -p "${TREE_SHA}" | grep 'src$' | awk '{print $3}')
echo "--- 3c. Subtree object (src/ directory) ---"
echo "Subtree SHA: ${SRC_TREE_SHA}"
echo ""
git cat-file -p "${SRC_TREE_SHA}"
echo ""

# Inspect a blob (file content)
MAIN_PY_SHA=$(git cat-file -p "${SRC_TREE_SHA}" | grep 'main.py' | awk '{print $3}')
echo "--- 3d. Blob object (src/main.py) ---"
echo "Blob SHA: ${MAIN_PY_SHA}"
echo ""
echo "Content:"
git cat-file -p "${MAIN_PY_SHA}"
echo ""

# ===========================================================================
# 4. Visualizing the object graph
# ===========================================================================
echo "--- 4. Object graph visualization ---"
echo ""
echo "  commit ${COMMIT_SHA:0:7}"
echo "    |"
echo "    +-- tree ${TREE_SHA:0:7}  (root)"
echo "    |     |"
echo "    |     +-- blob ...  README.md"
echo "    |     +-- tree ${SRC_TREE_SHA:0:7}  src/"
echo "    |           |"
echo "    |           +-- blob ...  config.py"
echo "    |           +-- blob ${MAIN_PY_SHA:0:7}  main.py"
echo ""

# ===========================================================================
# 5. How branches work — they're just pointers
# ===========================================================================
# Why this matters: A branch is literally a 41-byte file containing a SHA.
# Creating a branch is instant because Git just writes a file.
echo "--- 5. Branches are just pointers ---"

echo "The file .git/refs/heads/main contains:"
cat .git/refs/heads/main
echo ""

echo "git rev-parse confirms:"
git rev-parse main
echo ""

# Create a branch and show it's just a file
git branch experiment
echo "New branch file created:"
cat .git/refs/heads/experiment
echo ""

echo "HEAD points to a branch (symbolic reference):"
cat .git/HEAD
echo ""

# ===========================================================================
# 6. How git log traverses the graph
# ===========================================================================
echo "--- 6. Commit chain traversal ---"

# Create more commits to build a chain
echo "v2" >> README.md
git add README.md
git commit -m "Second commit"

echo "v3" >> README.md
git add README.md
git commit -m "Third commit"

echo "Commit chain (each commit points to its parent):"
CURRENT=$(git rev-parse HEAD)
while [[ -n "${CURRENT}" ]]; do
    MSG=$(git cat-file -p "${CURRENT}" | tail -1)
    PARENT=$(git cat-file -p "${CURRENT}" | grep '^parent ' | awk '{print $2}' || true)
    if [[ -n "${PARENT}" ]]; then
        echo "  ${CURRENT:0:7} (${MSG}) --> parent: ${PARENT:0:7}"
    else
        echo "  ${CURRENT:0:7} (${MSG}) --> (root commit, no parent)"
    fi
    CURRENT="${PARENT}"
done
echo ""

# ===========================================================================
# 7. git ls-tree — inspect directory structure at any point
# ===========================================================================
echo "--- 7. ls-tree: directory listing from any commit ---"

echo "Current tree (HEAD):"
git ls-tree -r HEAD --name-only
echo ""

echo "Tree at first commit (HEAD~2):"
git ls-tree -r HEAD~2 --name-only
echo ""

# ===========================================================================
# 8. Annotated tags — full objects in the database
# ===========================================================================
echo "--- 8. Tag objects ---"

git tag -a v1.0 -m "First stable release"
TAG_SHA=$(git rev-parse v1.0)
echo "Tag object SHA: ${TAG_SHA}"
echo ""

echo "Tag object content (note: it points to a commit):"
git cat-file -p "${TAG_SHA}"
echo ""

echo "Tag object type:"
git cat-file -t "${TAG_SHA}"
echo ""

# ===========================================================================
# 9. Verify repository integrity
# ===========================================================================
# Why fsck: Checks the entire object database for corruption. Useful after
# disk errors, interrupted operations, or before important operations.
echo "--- 9. Repository integrity check ---"
git fsck --no-dangling 2>&1
echo ""

# ===========================================================================
# 10. Object storage on disk
# ===========================================================================
echo "--- 10. How objects are stored on disk ---"

echo "Objects are stored in .git/objects/ using first 2 chars as directory:"
echo "Example for commit ${COMMIT_SHA:0:7}:"
echo "  Path: .git/objects/${COMMIT_SHA:0:2}/${COMMIT_SHA:2}"
echo "  Exists: $( [[ -f ".git/objects/${COMMIT_SHA:0:2}/${COMMIT_SHA:2}" ]] && echo 'YES' || echo 'NO' )"
echo ""

echo "Total objects in repository:"
git count-objects -v
echo ""

echo "=== Summary ==="
echo "Git internals demonstrated:"
echo "  git hash-object  — Create blob objects from content"
echo "  git cat-file     — Inspect any object (type, size, content)"
echo "  git ls-tree      — List tree (directory) contents"
echo "  git rev-parse    — Resolve references to SHA hashes"
echo "  git fsck         — Verify object database integrity"
echo ""
echo "Key insight: Git is a content-addressable filesystem."
echo "  - Same content always produces the same SHA hash"
echo "  - Branches and tags are just pointers (tiny files)"
echo "  - Commits form a linked list through parent references"
echo "  - Trees map filenames to content (blobs) and subdirs (trees)"
echo ""
echo "Demo complete."
