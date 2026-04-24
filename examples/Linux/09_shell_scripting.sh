#!/usr/bin/env bash
# =============================================================================
# 09_shell_scripting.sh — Variables, Control Flow, Functions, Parameter Expansion
#
# PURPOSE: A tour of the shell scripting constructs you will reach for most
#          often. Illustrates safe defaults (set -euo pipefail), proper
#          quoting, and common parameter-expansion patterns. All examples
#          are read-only; running this script produces only stdout text.
#
# USAGE:
#   ./09_shell_scripting.sh [--vars|--flow|--funcs|--expansion|--all]
# =============================================================================

# ---------------------------------------------------------------------------
# Safe defaults — put these at the top of EVERY non-trivial script.
# ---------------------------------------------------------------------------
# -e : exit on any unchecked command failure
# -u : error on unset variables
# -o pipefail : a pipeline's exit status is the last non-zero command's
#
# Why: default shell behavior keeps running after errors. That silently masks
# bugs in production scripts. These three flags turn silent failures into
# loud ones at the earliest point.
set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }

# ---------------------------------------------------------------------------
# 1. Variables and quoting
# ---------------------------------------------------------------------------
demo_vars() {
    section "1. Variables and Quoting"

    local name="Alice"
    local greeting="Hello, World"

    explain "Double quotes expand variables and command substitution:"
    show 'echo "Hi, $name — today is $(date +%A)"'
    echo "Hi, $name — today is $(date +%A)"

    explain "Single quotes are literal — NO expansion:"
    show "echo 'Hi, \$name — today is \$(date +%A)'"
    echo 'Hi, $name — today is $(date +%A)'

    # Why quote: an unquoted $var splits on whitespace and globs. This is the
    # #1 source of bugs in shell scripts. Quote by default; only omit quotes
    # when you deliberately want word-splitting (e.g., expanding a list).
    explain "Quoting matters when values contain spaces:"
    show 'echo "$greeting" vs echo $greeting'
    echo "Result A: $greeting"
    # shellcheck disable=SC2086   # intentional demo of word-splitting
    echo "Result B: " $greeting
    echo "  (both lines look similar here, but passing unquoted to functions"
    echo "   that count arguments can surprise you — quote unless you mean otherwise)"
}

# ---------------------------------------------------------------------------
# 2. Control flow — if, for, while, case
# ---------------------------------------------------------------------------
demo_flow() {
    section "2. Control Flow"

    explain "if / elif / else — bracket form [[ ]] (bash) supports richer tests:"
    local n=7
    show '[[ $n -gt 10 ]] && echo big || [[ $n -gt 5 ]] && echo medium || echo small'
    if [[ $n -gt 10 ]]; then
        echo "  n=$n -> big"
    elif [[ $n -gt 5 ]]; then
        echo "  n=$n -> medium"
    else
        echo "  n=$n -> small"
    fi

    explain "for — iterate over a list or a glob:"
    show 'for fruit in apple banana cherry; do ...'
    for fruit in apple banana cherry; do
        echo "  - $fruit (length ${#fruit})"
    done

    explain "C-style for — numeric ranges:"
    show 'for ((i=1; i<=3; i++)); do ...'
    for ((i = 1; i <= 3; i++)); do
        echo "  count: $i"
    done

    explain "while — loop until a condition fails:"
    local countdown=3
    while (( countdown > 0 )); do
        echo "  T-$countdown..."
        (( countdown-- ))
    done

    explain "case — pattern dispatch (cleaner than long if/elif chains):"
    local day
    day="$(date +%a)"
    case "$day" in
        Sat|Sun) echo "  $day — weekend" ;;
        *)       echo "  $day — weekday" ;;
    esac
}

# ---------------------------------------------------------------------------
# 3. Functions
# ---------------------------------------------------------------------------
demo_funcs() {
    section "3. Functions"

    # Why declare 'local': without it, variables leak to the global scope
    # and can clobber callers — a subtle source of bugs in larger scripts.
    greet() {
        local who="${1:-friend}"   # default if $1 is empty/unset
        echo "Hello, $who!"
    }

    sum() {
        local total=0
        local n
        for n in "$@"; do
            (( total += n ))
        done
        echo "$total"   # functions "return" via stdout; exit status is the return code
    }

    show 'greet                # no arg → uses default'
    greet

    show 'greet "Alice"'
    greet "Alice"

    show 'sum 1 2 3 4 5'
    local result
    result="$(sum 1 2 3 4 5)"
    echo "  sum = $result"
}

# ---------------------------------------------------------------------------
# 4. Parameter expansion — the quiet superpower
# ---------------------------------------------------------------------------
demo_expansion() {
    section "4. Parameter Expansion"

    local path="/var/log/app/service.log"
    local name=""

    explain "\${var:-default}  — use default if var is empty/unset:"
    show 'echo "${name:-anonymous}"'
    echo "  ${name:-anonymous}"

    explain "\${var:=default} — also ASSIGNS the default back to var:"
    show 'echo "${name:=guest}"'
    echo "  ${name:=guest}"
    echo "  (name is now '$name')"

    explain "\${#var} — length of var:"
    show 'echo "${#path}"'
    echo "  ${#path}"

    explain "\${var##*/} — strip longest match of '*/' from the front (basename):"
    show 'echo "${path##*/}"'
    echo "  ${path##*/}"

    explain "\${var%/*}  — strip shortest match of '/*' from the end (dirname):"
    show 'echo "${path%/*}"'
    echo "  ${path%/*}"

    explain "\${var/old/new} — replace first match; /old/ deletes:"
    show 'echo "${path/log/LOG}"'
    echo "  ${path/log/LOG}"

    explain "\${var//old/new} — replace ALL matches:"
    show 'echo "${path//\//:}"   # path separator swap'
    echo "  ${path//\//:}"
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --vars)      demo_vars ;;
        --flow)      demo_flow ;;
        --funcs)     demo_funcs ;;
        --expansion) demo_expansion ;;
        --all|*)
            demo_vars
            demo_flow
            demo_funcs
            demo_expansion
            ;;
    esac
}

main "$@"
