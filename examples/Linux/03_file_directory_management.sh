#!/usr/bin/env bash
# =============================================================================
# 03_file_directory_management.sh — mkdir, cp, mv, rm, find, touch
#
# PURPOSE: The daily commands for shaping a directory tree: creating,
#          copying, moving, removing, and searching. Everything happens
#          inside a tempdir cleaned up at exit — safe on any system.
#
# USAGE:
#   ./03_file_directory_management.sh [--create|--copy-move|--remove|--find|--all]
# =============================================================================

set -euo pipefail

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }
tree_like() {
    # Minimal tree-view fallback so the demo runs without the `tree` binary.
    find "$1" -print | sed -e "s|^$1||" -e "s|/[^/]*|  |g;s|^|  |"
}

# ---------------------------------------------------------------------------
# 1. Creating files and directories
# ---------------------------------------------------------------------------
demo_create() {
    section "1. Creating Files and Directories"

    explain "touch — create an empty file (or update mtime if it exists):"
    show "touch notes.txt log.txt"
    touch notes.txt log.txt
    ls -l notes.txt log.txt

    explain "mkdir -p — create a chain of directories, no error if they exist:"
    show "mkdir -p project/src/utils project/tests"
    mkdir -p project/src/utils project/tests
    tree_like "$(pwd)/project"

    explain "Write content into a file with a here-doc (<<'EOF') or >:"
    show "cat > project/README.md <<'EOF' ... EOF"
    cat > project/README.md <<'EOF'
# Demo Project
One-line summary of what this project does.
EOF
    cat project/README.md
}

# ---------------------------------------------------------------------------
# 2. Copying and moving
# ---------------------------------------------------------------------------
demo_copy_move() {
    section "2. Copying and Moving"

    # Ensure we have something to copy even if this section runs alone.
    mkdir -p project/src
    echo "placeholder" > project/README.md

    explain "cp — copy a file. -r to copy a directory recursively. -p to preserve timestamps & perms:"
    show "cp project/README.md project/README.bak"
    cp project/README.md project/README.bak
    show "cp -rp project/src project/src_backup"
    cp -rp project/src project/src_backup

    ls -l project/README.md project/README.bak

    explain "mv — move OR rename (same command; no destination dir → rename):"
    show "mv project/README.bak project/OLD_README.md"
    mv project/README.bak project/OLD_README.md
    ls project

    # Why 'mv' feels dangerous: it overwrites silently by default. Use -i
    # (interactive, prompts before overwrite) or -n (never clobber) to be safe.
    explain "mv -n — refuse to overwrite an existing file (safer for scripts):"
    show "mv -n notes.txt log.txt || echo '  (mv -n refused to overwrite)'"
    mv -n notes.txt log.txt 2>/dev/null || echo "  (mv -n refused to overwrite existing log.txt)"
}

# ---------------------------------------------------------------------------
# 3. Removing — with care
# ---------------------------------------------------------------------------
demo_remove() {
    section "3. Removing Files — The Dangerous One"

    # Staging area so -r and wildcards have something to act on.
    mkdir -p junk/a/b junk/a/c
    touch junk/a/b/x junk/a/c/y junk/a/z
    tree_like "$(pwd)/junk"

    explain "rm — removes files. By default it does NOT remove directories."
    show "rm junk/a/z"
    rm junk/a/z

    explain "rm -r — recursive, removes a directory and its contents."
    show "rm -r junk/a/b"
    rm -r junk/a/b

    # Why -i is a beginner's safety net: interactive prompts prevent the
    # classic 'accidentally rm -rf the wrong path'. Use -I (capital) for one
    # prompt per recursive delete instead of one per file.
    explain "rm -i — prompts before each file (press 'n' here to decline):"
    show "yes n | rm -ri junk | head -5   # simulate declining"
    yes n 2>/dev/null | rm -ri junk 2>&1 | head -5 || true

    explain "NEVER type 'rm -rf /' or 'rm -rf \$VAR/' where \$VAR might be empty."
    explain "Guard scripts with: [[ -n \"\$target\" && \"\$target\" != / ]] before recursive rm."
}

# ---------------------------------------------------------------------------
# 4. find — locate by name, type, time, or size
# ---------------------------------------------------------------------------
demo_find() {
    section "4. find — Locate Files by Attributes"

    # Build a small tree with varied ages and sizes
    mkdir -p tree/{alpha,beta}
    echo "small" > tree/alpha/tiny.txt
    head -c 5000 /dev/urandom > tree/beta/biggish.bin 2>/dev/null || dd if=/dev/zero of=tree/beta/biggish.bin bs=1024 count=5 status=none
    touch -t 202001010000 tree/alpha/ancient.txt   # fixed old timestamp

    explain "Basic find — every file under a path:"
    show "find tree"
    find tree

    explain "Filter by name pattern (-name, -iname for case-insensitive):"
    show "find tree -name '*.txt'"
    find tree -name '*.txt'

    explain "Filter by type: -type f (file), -type d (dir), -type l (symlink):"
    show "find tree -type d"
    find tree -type d

    explain "Filter by size: +10k = greater than 10 KiB:"
    show "find tree -type f -size +2k"
    find tree -type f -size +2k

    explain "Filter by mtime: -mtime +30 = modified more than 30 days ago:"
    show "find tree -mtime +30 -type f"
    find tree -mtime +30 -type f

    explain "Execute a command on each match (-exec ... {} \\;):"
    show "find tree -type f -exec wc -l {} \\;"
    find tree -type f -exec wc -l {} \; 2>/dev/null || true
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --create)    demo_create ;;
        --copy-move) demo_copy_move ;;
        --remove)    demo_remove ;;
        --find)      demo_find ;;
        --all|*)
            demo_create
            demo_copy_move
            demo_remove
            demo_find
            ;;
    esac
}

main "$@"
