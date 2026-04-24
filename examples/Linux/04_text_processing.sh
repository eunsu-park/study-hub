#!/usr/bin/env bash
# =============================================================================
# 04_text_processing.sh — grep, sed, awk, cut, sort, uniq, tr, wc
#
# PURPOSE: The Unix text-processing toolkit. Each demo uses a small sample
#          file generated inside a tempdir, so the script is self-contained
#          and runs on any system.
#
# USAGE:
#   ./04_text_processing.sh [--grep|--sed|--awk|--pipeline|--all]
# =============================================================================

set -euo pipefail

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show_cmd() { printf "[CMD]  %s\n" "$1"; }

# Sample CSV for all demos. Keep it small and readable.
SAMPLE="$WORKDIR/users.csv"
cat > "$SAMPLE" <<'EOF'
id,name,role,city,salary
1,Alice,engineer,Seoul,78000
2,Bob,manager,Tokyo,95000
3,Carol,engineer,Seoul,82000
4,Dave,analyst,Busan,65000
5,Eve,engineer,Tokyo,88000
6,Frank,manager,Seoul,110000
EOF

# ---------------------------------------------------------------------------
# 1. grep — finding patterns
# ---------------------------------------------------------------------------
demo_grep() {
    section "1. grep — finding patterns"

    explain "Plain literal match:"
    show_cmd "grep 'engineer' users.csv"
    grep 'engineer' "$SAMPLE"

    explain "Case-insensitive (-i), count only (-c), line numbers (-n):"
    show_cmd "grep -in 'seoul' users.csv"
    grep -in 'seoul' "$SAMPLE"

    show_cmd "grep -c 'engineer' users.csv   # count matching lines"
    grep -c 'engineer' "$SAMPLE"

    explain "Invert match (-v) — everything EXCEPT the header row:"
    show_cmd "grep -v '^id,' users.csv"
    grep -v '^id,' "$SAMPLE"

    explain "Extended regex (-E) — engineers or managers:"
    show_cmd "grep -E '(engineer|manager)' users.csv"
    grep -E '(engineer|manager)' "$SAMPLE"
}

# ---------------------------------------------------------------------------
# 2. sed — streaming edits
# ---------------------------------------------------------------------------
demo_sed() {
    section "2. sed — stream editor"

    explain "Substitute (s/old/new/): replace first match on each line."
    show_cmd "sed 's/Seoul/SEL/' users.csv"
    sed 's/Seoul/SEL/' "$SAMPLE"

    explain "Global on the line (s/.../.../g) — all matches, not just the first:"
    show_cmd "echo 'foo bar foo' | sed 's/foo/FOO/g'"
    echo 'foo bar foo' | sed 's/foo/FOO/g'

    explain "Delete header line (1d) — useful in pipelines:"
    show_cmd "sed '1d' users.csv | head -3"
    sed '1d' "$SAMPLE" | head -3

    explain "Print only lines 2-4 (-n with p):"
    show_cmd "sed -n '2,4p' users.csv"
    sed -n '2,4p' "$SAMPLE"
}

# ---------------------------------------------------------------------------
# 3. awk — field-aware processing
# ---------------------------------------------------------------------------
demo_awk() {
    section "3. awk — field-aware processing"

    explain "awk splits each line on whitespace (or -F<sep>) into \$1, \$2, ..."
    explain "NR is the current record number. \$0 is the whole line."

    show_cmd "awk -F, 'NR>1 {print \$2, \$3}' users.csv   # name and role, skip header"
    awk -F, 'NR>1 {print $2, $3}' "$SAMPLE"

    explain "Filter by field value — engineers only:"
    show_cmd "awk -F, '\$3==\"engineer\" {print \$2, \$5}' users.csv"
    awk -F, '$3=="engineer" {print $2, $5}' "$SAMPLE"

    explain "Aggregate: average salary by role, in one awk program:"
    show_cmd 'awk -F, ... (see script)'
    awk -F, '
        NR > 1 { sum[$3] += $5; count[$3]++ }
        END    { for (role in sum) printf "  %-10s %d (avg over %d)\n", role, sum[role]/count[role], count[role] }
    ' "$SAMPLE"
}

# ---------------------------------------------------------------------------
# 4. The Unix pipeline — composing small tools
# ---------------------------------------------------------------------------
demo_pipeline() {
    section "4. Pipelines — composing cut, sort, uniq, tr, wc"

    explain "Goal: count users per city, sorted by count desc."
    explain "Pipeline: strip header → extract column 4 → sort → count runs → sort desc."

    show_cmd "tail -n +2 users.csv | cut -d, -f4 | sort | uniq -c | sort -rn"
    tail -n +2 "$SAMPLE" | cut -d, -f4 | sort | uniq -c | sort -rn

    explain "Count total lines / words / bytes with wc:"
    show_cmd "wc users.csv"
    wc "$SAMPLE"

    explain "tr translates or deletes characters (great for normalization):"
    show_cmd "echo 'Hello, World' | tr '[:upper:]' '[:lower:]'"
    echo 'Hello, World' | tr '[:upper:]' '[:lower:]'

    show_cmd "echo 'a,b,c,d' | tr ',' '\\n'   # split CSV row into lines"
    echo 'a,b,c,d' | tr ',' '\n'
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    explain "Sample file: $SAMPLE"
    cat "$SAMPLE"

    case "$mode" in
        --grep)     demo_grep ;;
        --sed)      demo_sed ;;
        --awk)      demo_awk ;;
        --pipeline) demo_pipeline ;;
        --all|*)
            demo_grep
            demo_sed
            demo_awk
            demo_pipeline
            ;;
    esac
}

main "$@"
