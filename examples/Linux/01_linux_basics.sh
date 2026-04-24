#!/usr/bin/env bash
# =============================================================================
# 01_linux_basics.sh — Shell Basics, Built-ins, and Getting Help
#
# PURPOSE: A first look at interacting with a Linux shell. Shows the
#          difference between a command and a built-in, how to discover
#          what a command does, and how the shell remembers history.
#          Every action is read-only — safe on any system.
#
# USAGE:
#   ./01_linux_basics.sh [--hello|--help-systems|--builtins|--history|--all]
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show_cmd() { printf "[CMD]  %s\n" "$1"; }

# ---------------------------------------------------------------------------
# 1. Hello, shell
# ---------------------------------------------------------------------------
demo_hello() {
    section "1. Your First Commands"

    explain "A shell is a program that reads commands and runs them."
    explain "echo prints its arguments; pwd prints the working directory;"
    explain "whoami prints the current user; date prints the current time."

    show_cmd "echo 'Hello, Linux!'"
    echo "Hello, Linux!"

    show_cmd "pwd"
    pwd

    show_cmd "whoami"
    whoami

    show_cmd "date"
    date
}

# ---------------------------------------------------------------------------
# 2. How to get help — three parallel help systems
# ---------------------------------------------------------------------------
demo_help_systems() {
    section "2. Three Ways to Get Help"

    explain "man <cmd>        — long-form manual, organized by section"
    explain "<cmd> --help     — short usage summary, printed instantly"
    explain "type <name>      — tells you whether something is a built-in, alias, or file"

    # Why `type` matters: the same name can refer to a built-in, an alias,
    # or an executable on PATH — and they can behave differently. Knowing
    # which one you're calling is the first step to debugging surprises.
    show_cmd "type echo"
    type echo || true

    show_cmd "type cd"
    type cd || true

    show_cmd "type ls"
    type ls || true

    show_cmd "echo --help | head -3   # most commands support --help"
    echo --help 2>/dev/null | head -3 || true
}

# ---------------------------------------------------------------------------
# 3. Built-ins vs external commands
# ---------------------------------------------------------------------------
demo_builtins() {
    section "3. Built-ins vs External Commands"

    explain "A built-in runs inside the shell itself (fast, no fork)."
    explain "An external command is a file on disk the shell executes."
    explain "'cd' MUST be a built-in — it changes the shell's own state."

    show_cmd "command -v ls        # path of the external binary"
    command -v ls

    show_cmd "command -v cd        # cd is a builtin, so no path"
    command -v cd || echo "  (cd is a shell built-in, not an executable)"

    show_cmd "compgen -b | head -10   # list shell built-ins (bash-specific)"
    compgen -b 2>/dev/null | head -10 || explain "compgen unavailable in this shell"
}

# ---------------------------------------------------------------------------
# 4. Command history
# ---------------------------------------------------------------------------
demo_history() {
    section "4. Command History"

    explain "The shell remembers previous commands. You can search, recall, and re-run them."
    explain "  Up/Down arrows  — step through history"
    explain "  Ctrl-R          — reverse search"
    explain "  !!              — repeat last command"
    explain "  !<N>            — re-run command #N from history"
    explain "  history         — show the list (this script runs in a subshell,"
    explain "                     so you will not see your interactive history here)"

    show_cmd "history | tail -5   # if run interactively"
    # In a script subshell, history is usually empty — that is expected.
    history 2>/dev/null | tail -5 || echo "  (history unavailable in non-interactive subshell — expected)"
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --hello)        demo_hello ;;
        --help-systems) demo_help_systems ;;
        --builtins)     demo_builtins ;;
        --history)      demo_history ;;
        --all|*)
            demo_hello
            demo_help_systems
            demo_builtins
            demo_history
            ;;
    esac
}

main "$@"
