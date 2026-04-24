#!/usr/bin/env bash
# =============================================================================
# 02_filesystem_navigation.sh — FHS, Paths, and Directory Listing
#
# PURPOSE: Shows the Filesystem Hierarchy Standard (FHS), the difference
#          between absolute and relative paths, how ls flags change the
#          view, and what symbolic vs hard links are. All operations are
#          read-only or happen inside a temporary directory cleaned up at
#          exit — safe on any system.
#
# USAGE:
#   ./02_filesystem_navigation.sh [--fhs|--paths|--ls|--links|--all]
# =============================================================================

set -euo pipefail

# Temporary workspace. trap ensures cleanup even if we abort early.
# Why: demonstrating links and paths requires creating files; doing it in
# $HOME would pollute the user's tree.
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show_cmd() { printf "[CMD]  %s\n" "$1"; }

# ---------------------------------------------------------------------------
# 1. Filesystem Hierarchy Standard
# ---------------------------------------------------------------------------
demo_fhs() {
    section "1. Filesystem Hierarchy Standard (FHS)"

    explain "Every Linux-like system follows a standard layout. Knowing it lets"
    explain "you guess where to look. The essentials:"

    # A small lookup table. Keeping it in the script keeps the lesson portable.
    cat <<'EOF'
  /bin, /usr/bin     essential & general-purpose executables
  /sbin, /usr/sbin   system administration binaries (often need sudo)
  /etc               system-wide configuration files (plain text)
  /home              per-user home directories
  /var               variable data: logs, caches, spool files
  /tmp               ephemeral files; cleared on reboot on most systems
  /opt               optional, third-party software installed outside FHS
  /proc, /sys        virtual filesystems exposing kernel/runtime state
  /dev               device nodes (disks, terminals, random sources)
EOF

    explain "Let's peek at a few:"
    show_cmd "ls /etc | head -5"
    ls /etc | head -5 || true

    show_cmd "cat /proc/uptime   # seconds since boot, idle seconds"
    cat /proc/uptime 2>/dev/null || echo "  (/proc not available on this OS)"
}

# ---------------------------------------------------------------------------
# 2. Absolute vs relative paths
# ---------------------------------------------------------------------------
demo_paths() {
    section "2. Absolute vs Relative Paths"

    explain "Absolute path: starts with '/' — interpreted from the filesystem root."
    explain "Relative path: does NOT start with '/' — interpreted from the current directory."
    explain "Special segments: '.' = current dir, '..' = parent dir, '~' = your home."

    cd "$WORKDIR"

    mkdir -p projects/demo
    echo "hello" > projects/demo/note.txt

    show_cmd "pwd"
    pwd

    show_cmd "cat projects/demo/note.txt     # relative"
    cat projects/demo/note.txt

    show_cmd "cat \"\$WORKDIR/projects/demo/note.txt\"     # absolute"
    cat "$WORKDIR/projects/demo/note.txt"

    show_cmd "cd projects/demo && cat ../../\$(basename \$WORKDIR)/projects/demo/note.txt"
    (cd projects/demo && cat ../demo/note.txt)

    show_cmd "realpath projects/demo/note.txt   # canonical absolute path"
    realpath projects/demo/note.txt 2>/dev/null || echo "  (realpath unavailable)"
}

# ---------------------------------------------------------------------------
# 3. ls: the same directory, different views
# ---------------------------------------------------------------------------
demo_ls() {
    section "3. ls — Different Views of the Same Directory"

    cd "$WORKDIR"
    mkdir -p sample
    touch sample/alpha sample/beta sample/gamma sample/.hidden
    echo "content" > sample/data.txt

    explain "Plain ls shows non-hidden files only:"
    show_cmd "ls sample"
    ls sample

    explain "ls -a shows hidden dotfiles too:"
    show_cmd "ls -a sample"
    ls -a sample

    explain "ls -l shows permissions, owner, size, mtime:"
    show_cmd "ls -l sample"
    ls -l sample

    explain "ls -lh uses human-readable sizes; ls -ltr sorts by time (oldest first):"
    show_cmd "ls -lhtr sample"
    ls -lhtr sample
}

# ---------------------------------------------------------------------------
# 4. Symbolic vs hard links
# ---------------------------------------------------------------------------
demo_links() {
    section "4. Symbolic vs Hard Links"

    cd "$WORKDIR"
    echo "original data" > original.txt

    explain "A HARD link is another name for the same underlying file."
    explain "A SYMBOLIC (soft) link is a tiny file that points to a path."

    show_cmd "ln original.txt hardlink.txt        # hard link"
    ln original.txt hardlink.txt

    show_cmd "ln -s original.txt symlink.txt      # symbolic link"
    ln -s original.txt symlink.txt

    show_cmd "ls -l original.txt hardlink.txt symlink.txt"
    ls -l original.txt hardlink.txt symlink.txt

    # Why this matters: deleting the original DOES break the symlink but
    # NOT the hard link — because both hard-link names are equal citizens.
    explain "Delete the original and observe:"
    show_cmd "rm original.txt && ls -l hardlink.txt symlink.txt"
    rm original.txt
    ls -l hardlink.txt 2>/dev/null && cat hardlink.txt
    ls -l symlink.txt 2>/dev/null
    cat symlink.txt 2>/dev/null && echo "  symlink resolved" || echo "  symlink is now broken (target gone)"
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --fhs)   demo_fhs ;;
        --paths) demo_paths ;;
        --ls)    demo_ls ;;
        --links) demo_links ;;
        --all|*)
            demo_fhs
            demo_paths
            demo_ls
            demo_links
            ;;
    esac
}

main "$@"
