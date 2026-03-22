#!/bin/bash
# Exercises for Lesson 01: Linux Basics
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Navigating Man Pages and Help Systems ===
# Problem: Use man, --help, info, and whatis to explore command documentation.
exercise_1() {
    echo "=== Exercise 1: Navigating Man Pages and Help Systems ==="
    echo ""
    echo "Scenario: You just joined a team managing a Linux server and need to"
    echo "quickly look up unfamiliar commands without internet access."
    echo ""

    echo "--- Part A: Find the man page section for passwd ---"
    echo "Solution:"
    echo "  man -f passwd"
    echo ""
    echo "  Explanation:"
    echo "    -f (equivalent to 'whatis') shows all man page entries for 'passwd'."
    echo "    Output shows passwd(1) for the command and passwd(5) for the file format."
    echo "    Use 'man 5 passwd' to read the file format section specifically."
    echo ""

    echo "--- Part B: Search for commands related to 'disk' ---"
    echo "Solution:"
    echo "  man -k disk"
    echo "  apropos disk"
    echo ""
    echo "  Explanation:"
    echo "    -k (equivalent to 'apropos') searches all man page descriptions for"
    echo "    the keyword 'disk'. Returns commands like fdisk, df, lsblk, etc."
    echo "    Useful when you know what you want to do but not the command name."
    echo ""

    echo "--- Part C: Get a quick one-line description of a command ---"
    echo "Solution:"
    echo "  whatis ls cp mv rm"
    echo ""
    echo "  Explanation:"
    echo "    whatis prints the NAME section from each command's man page."
    echo "    Accepts multiple commands at once for quick comparison."
    echo ""

    echo "--- Part D: Use --help for a concise option summary ---"
    echo "Solution:"
    echo "  ls --help"
    echo "  ls --help 2>&1 | grep -- '-l'"
    echo ""
    echo "  Explanation:"
    echo "    --help is faster than man for checking a specific flag."
    echo "    Piping through grep isolates the option you care about."
    echo "    The '--' after grep prevents '-l' from being parsed as a grep flag."
    echo ""

    echo "Verification:"
    echo "  man -k . | wc -l   # Count total available man pages on the system"
    if command -v man &>/dev/null; then
        count=$(man -k . 2>/dev/null | wc -l)
        echo "  This system has approximately $count man pages available."
    fi
}

# === Exercise 2: Identifying System Information ===
# Problem: Gather comprehensive system information using built-in commands.
exercise_2() {
    echo "=== Exercise 2: Identifying System Information ==="
    echo ""
    echo "Scenario: You need to document a server's hardware and OS configuration"
    echo "before performing a major upgrade."
    echo ""

    echo "--- Part A: Kernel and architecture information ---"
    echo "Solution:"
    echo "  uname -a           # All system info in one line"
    echo "  uname -r           # Kernel release only (e.g., 5.15.0-91-generic)"
    echo "  uname -m           # Machine architecture (x86_64, aarch64)"
    echo ""
    echo "  Explanation:"
    echo "    -a = all: kernel name, hostname, release, version, machine, OS"
    echo "    -r = release: essential for checking driver/module compatibility"
    echo "    -m = machine: confirms 64-bit (x86_64) vs 32-bit (i686)"
    echo ""

    echo "--- Part B: OS distribution details ---"
    echo "Solution:"
    echo "  cat /etc/os-release"
    echo "  lsb_release -a        # If lsb-release package is installed"
    echo "  hostnamectl            # systemd systems: OS, kernel, architecture"
    echo ""
    echo "  Explanation:"
    echo "    /etc/os-release is the standard file on all modern distros."
    echo "    lsb_release provides a uniform interface across distributions."
    echo "    hostnamectl combines hostname, OS, kernel, and virtualization info."
    echo ""

    echo "--- Part C: Hardware summary ---"
    echo "Solution:"
    echo "  lscpu                    # CPU model, cores, threads, cache"
    echo "  free -h                  # Memory usage in human-readable format"
    echo "  lsblk                    # Block device tree (disks and partitions)"
    echo ""
    echo "  Explanation:"
    echo "    lscpu reads /proc/cpuinfo in a structured format."
    echo "    free -h shows total/used/available RAM and swap (-h = human-readable)."
    echo "    lsblk shows the disk→partition→mount hierarchy without needing root."
    echo ""

    echo "--- Current system info ---"
    echo "  Kernel: $(uname -r 2>/dev/null || echo 'N/A')"
    echo "  Arch:   $(uname -m 2>/dev/null || echo 'N/A')"
    echo "  Host:   $(hostname 2>/dev/null || echo 'N/A')"
}

# === Exercise 3: Terminal Shortcuts and History ===
# Problem: Use shell history and keyboard shortcuts for efficient terminal usage.
exercise_3() {
    echo "=== Exercise 3: Terminal Shortcuts and History Usage ==="
    echo ""
    echo "Scenario: You frequently repeat complex commands and want to work"
    echo "faster in the terminal without retyping everything."
    echo ""

    echo "--- Part A: Search and reuse command history ---"
    echo "Solution:"
    echo "  history              # Show numbered command history"
    echo "  history 20           # Show last 20 commands"
    echo "  !42                  # Re-execute command number 42"
    echo "  !!                   # Re-execute the last command"
    echo "  sudo !!              # Re-run last command with sudo (very common)"
    echo "  !ssh                 # Re-run the most recent command starting with 'ssh'"
    echo ""
    echo "  Explanation:"
    echo "    History is stored in ~/.bash_history (persisted between sessions)."
    echo "    HISTSIZE controls in-memory count, HISTFILESIZE controls file count."
    echo "    '!!' is the most-used shortcut — saves retyping after a permission error."
    echo ""

    echo "--- Part B: Reverse search with Ctrl+R ---"
    echo "Solution:"
    echo "  Press Ctrl+R, then type part of a previous command"
    echo "  Press Ctrl+R again to cycle through older matches"
    echo "  Press Enter to execute, or Ctrl+G to cancel"
    echo ""
    echo "  Explanation:"
    echo "    Reverse incremental search (Ctrl+R) is the fastest way to find"
    echo "    a specific command from your history. It matches anywhere in the"
    echo "    command line, not just the beginning."
    echo ""

    echo "--- Part C: Essential keyboard shortcuts ---"
    echo "Solution:"
    echo "  Ctrl+A    Move cursor to beginning of line"
    echo "  Ctrl+E    Move cursor to end of line"
    echo "  Ctrl+U    Cut everything before cursor"
    echo "  Ctrl+K    Cut everything after cursor"
    echo "  Ctrl+W    Cut the word before cursor"
    echo "  Ctrl+Y    Paste (yank) the last cut text"
    echo "  Ctrl+L    Clear screen (same as 'clear' command)"
    echo "  Ctrl+C    Interrupt (kill) current process"
    echo "  Ctrl+D    Exit shell / send EOF"
    echo "  Ctrl+Z    Suspend current process (resume with 'fg')"
    echo ""
    echo "  Explanation:"
    echo "    These are readline shortcuts (inherited from Emacs keybindings)."
    echo "    They work in bash, zsh, and most interactive CLI programs."
    echo "    Ctrl+U and Ctrl+Y together let you 'park' a half-typed command."
    echo ""

    echo "--- Part D: Customize history behavior ---"
    echo "Solution (add to ~/.bashrc):"
    echo "  export HISTSIZE=10000              # Remember 10,000 commands in memory"
    echo "  export HISTFILESIZE=20000          # Store 20,000 commands in file"
    echo "  export HISTCONTROL=ignoreboth      # Skip duplicates and space-prefixed"
    echo "  export HISTTIMEFORMAT='%F %T  '   # Timestamp each entry"
    echo "  shopt -s histappend                # Append to history, don't overwrite"
    echo ""
    echo "  Explanation:"
    echo "    ignoreboth = ignoredups + ignorespace. Commands starting with a"
    echo "    space are not recorded (useful for commands containing passwords)."
    echo "    histappend prevents multiple terminal sessions from overwriting"
    echo "    each other's history."
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
