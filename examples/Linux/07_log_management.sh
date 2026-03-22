#!/usr/bin/env bash
# =============================================================================
# 07_log_management.sh - Log Management, Rotation, and Analysis
#
# PURPOSE: Demonstrates journalctl queries, rsyslog configuration patterns,
#          logrotate setup, and structured log analysis with jq. Generates
#          sample logs for hands-on practice.
#
# USAGE:
#   ./07_log_management.sh [--journal|--rsyslog|--rotate|--analysis|--all]
#
# MODES:
#   --journal   journalctl filtering and export techniques
#   --rsyslog   rsyslog configuration patterns
#   --rotate    logrotate configuration and manual rotation
#   --analysis  Structured log parsing with jq and awk
#   --all       Run all sections (default)
#
# CONCEPTS COVERED:
#   - journalctl: priority filtering, time ranges, follow mode, export
#   - rsyslog: facility/severity, forwarding, templates
#   - logrotate: size/time-based rotation, compression, retention
#   - Structured logging: JSON format, jq queries, log aggregation
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/log_demo}"
mkdir -p "${OUTPUT_DIR}"

section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }
explain() { echo -e "${GREEN}[INFO]${RESET} $1"; }
show_cmd() { echo -e "${YELLOW}[CMD]${RESET} $1"; }

# ---------------------------------------------------------------------------
# Generate sample structured logs for analysis demonstrations
# ---------------------------------------------------------------------------
generate_sample_logs() {
    local logfile="${OUTPUT_DIR}/app.jsonl"

    # Why: JSON Lines (one JSON object per line) is the modern standard for
    # structured logs. Unlike plain text, it enables precise filtering by
    # field (e.g., all errors from the auth module in the last hour).
    cat > "${logfile}" <<'JSONL'
{"ts":"2026-02-27T10:00:01Z","level":"INFO","module":"auth","msg":"User login successful","user":"alice","ip":"10.0.1.5","latency_ms":45}
{"ts":"2026-02-27T10:00:02Z","level":"WARN","module":"api","msg":"Rate limit approaching","user":"bob","endpoint":"/api/data","rate":85}
{"ts":"2026-02-27T10:00:03Z","level":"ERROR","module":"db","msg":"Connection pool exhausted","pool_size":20,"active":20,"waiting":5}
{"ts":"2026-02-27T10:00:04Z","level":"INFO","module":"api","msg":"Request completed","method":"GET","path":"/api/users","status":200,"latency_ms":12}
{"ts":"2026-02-27T10:00:05Z","level":"ERROR","module":"auth","msg":"Invalid token","user":"charlie","ip":"203.0.113.42","reason":"expired"}
{"ts":"2026-02-27T10:00:06Z","level":"INFO","module":"worker","msg":"Job completed","job_id":"j-1234","duration_s":3.2,"items_processed":1500}
{"ts":"2026-02-27T10:00:07Z","level":"WARN","module":"disk","msg":"Disk usage high","mount":"/var/log","usage_pct":87}
{"ts":"2026-02-27T10:00:08Z","level":"INFO","module":"api","msg":"Request completed","method":"POST","path":"/api/orders","status":201,"latency_ms":89}
{"ts":"2026-02-27T10:00:09Z","level":"ERROR","module":"api","msg":"Internal server error","method":"GET","path":"/api/reports","status":500,"latency_ms":2001}
{"ts":"2026-02-27T10:00:10Z","level":"INFO","module":"auth","msg":"User login successful","user":"dave","ip":"10.0.2.8","latency_ms":38}
JSONL
    echo "${logfile}"
}

# ---------------------------------------------------------------------------
# 1. journalctl Techniques
# ---------------------------------------------------------------------------
demo_journal() {
    section "1. journalctl Filtering & Export"

    explain "Time-based filtering:"
    show_cmd "journalctl --since '2026-02-27 00:00:00' --until '2026-02-27 12:00:00'"
    show_cmd "journalctl --since '30 min ago'"
    show_cmd "journalctl --since today"
    echo ""

    explain "Priority-based filtering:"
    # Why: Syslog priorities are hierarchical. -p err shows err + crit +
    # alert + emerg. This is far more useful than grepping for 'error'.
    echo "  Priorities: emerg(0) alert(1) crit(2) err(3) warn(4) notice(5) info(6) debug(7)"
    show_cmd "journalctl -p err                        # Errors and above"
    show_cmd "journalctl -p warning..err               # Range: warnings to errors"
    echo ""

    explain "Combining filters (AND logic):"
    # Why: Multiple filters are ANDed. This lets you narrow to exactly
    # the logs you need without piping through grep chains.
    show_cmd "journalctl -u nginx.service -p err --since '1h ago'"
    show_cmd "journalctl _UID=1000 -p warning          # By user ID"
    show_cmd "journalctl _COMM=python3 -p err          # By command name"
    echo ""

    explain "Exporting for analysis:"
    show_cmd "journalctl -u myapp -o json > /tmp/myapp_logs.json"
    show_cmd "journalctl -u myapp -o export > /tmp/myapp.export  # Binary format"
    echo ""

    explain "Persistent journal configuration (/etc/systemd/journald.conf):"
    echo "  Storage=persistent          # Store on disk (default: auto)"
    echo "  SystemMaxUse=500M           # Max total journal size"
    echo "  SystemMaxFileSize=50M       # Max single file size"
    echo "  MaxRetentionSec=30day       # Auto-delete after 30 days"
    echo ""
    # Why: Without explicit limits, the journal grows until the filesystem
    # runs out of space, which can cause cascading failures.

    if command -v journalctl &>/dev/null; then
        explain "Live example — recent kernel messages:"
        journalctl -k --no-pager -n 3 -o short-iso 2>/dev/null || echo "  (insufficient permissions)"
    fi
}

# ---------------------------------------------------------------------------
# 2. rsyslog Configuration
# ---------------------------------------------------------------------------
demo_rsyslog() {
    section "2. rsyslog Configuration Patterns"

    local rsyslog_conf="${OUTPUT_DIR}/rsyslog_custom.conf"

    explain "rsyslog: The traditional Linux syslog daemon"
    explain "Facility = source category, Severity = urgency level"
    echo ""
    echo "  Facilities: auth, authpriv, cron, daemon, kern, local0-7, mail, user..."
    echo "  Severities: emerg, alert, crit, err, warning, notice, info, debug"
    echo ""

    explain "Generating example rsyslog config: ${rsyslog_conf}"
    cat > "${rsyslog_conf}" <<'CONF'
# /etc/rsyslog.d/50-custom.conf
# Why: Drop-in configs in /etc/rsyslog.d/ are loaded alphabetically.
# Using 50- prefix ensures it runs after default configs (00-20).

# ---- Local file routing ----
# Why: Separating logs by facility makes troubleshooting faster.
# auth/authpriv → security audits, mail → email server issues.
auth,authpriv.*         /var/log/auth.log
mail.*                  /var/log/mail.log
cron.*                  /var/log/cron.log

# Why: local0-7 are reserved for custom applications.
# Convention: local0=app1, local1=app2, etc.
local0.*                /var/log/myapp.log

# ---- Remote forwarding (TCP with TLS) ----
# Why: Centralized logging is critical for security and compliance.
# TCP (@@) is reliable unlike UDP (@) which can silently drop messages.
*.* @@(o)logserver.example.com:6514

# ---- Template for structured output ----
# Why: Templates let you output in JSON, which log aggregators
# (ELK, Splunk, Loki) can parse without custom regex.
template(name="json-template" type="list") {
    constant(value="{")
    constant(value="\"timestamp\":\"")  property(name="timereported" dateFormat="rfc3339")
    constant(value="\",\"host\":\"")    property(name="hostname")
    constant(value="\",\"severity\":\"") property(name="syslogseverity-text")
    constant(value="\",\"facility\":\"") property(name="syslogfacility-text")
    constant(value="\",\"tag\":\"")     property(name="syslogtag")
    constant(value="\",\"msg\":\"")     property(name="msg" format="jsonf")
    constant(value="\"}\n")
}

# Apply JSON template to local0 logs
local0.* action(type="omfile" file="/var/log/myapp-structured.json"
               template="json-template")
CONF

    echo -e "  ${GREEN}Written to ${rsyslog_conf}${RESET}"
    echo ""

    explain "Testing rsyslog config and sending test messages:"
    show_cmd "rsyslogd -N1                              # Validate config syntax"
    show_cmd "logger -p local0.info 'Test message'      # Send to local0.info"
    show_cmd "logger -p local0.err 'Error occurred'     # Send to local0.err"
}

# ---------------------------------------------------------------------------
# 3. Log Rotation
# ---------------------------------------------------------------------------
demo_rotate() {
    section "3. logrotate Configuration"

    local rotate_conf="${OUTPUT_DIR}/logrotate_myapp.conf"

    explain "logrotate prevents log files from consuming all disk space."
    explain "It runs daily via cron or systemd timer."
    echo ""

    explain "Generating logrotate config: ${rotate_conf}"
    # Why: Each directive addresses a specific operational concern:
    #   rotate N — keep N rotated files (balance history vs disk)
    #   compress — save 60-80% disk with gzip
    #   delaycompress — don't compress the most recent rotation (apps may still write)
    #   copytruncate — for apps that can't reopen log files after rotation
    cat > "${rotate_conf}" <<'CONF'
# /etc/logrotate.d/myapp
/var/log/myapp/*.log {
    # Rotation schedule
    daily                   # Rotate every day (also: weekly, monthly)
    rotate 14               # Keep 14 rotated files (2 weeks of history)
    missingok               # Don't error if the log file is missing
    notifempty              # Don't rotate if file is empty (saves inodes)

    # Compression
    compress                # gzip old log files
    delaycompress           # Keep the most recent rotation uncompressed
                            # Why: Some apps buffer writes and may still
                            # append to the just-rotated file briefly.

    # Size-based trigger (overrides daily if file is small)
    minsize 10M             # Only rotate if file is >= 10MB
    maxsize 500M            # Force rotation if file exceeds 500MB

    # Post-rotation action
    # Why: Many daemons hold the old file descriptor open. postrotate
    # sends a signal so the daemon reopens its log file.
    postrotate
        /usr/bin/systemctl reload myapp.service 2>/dev/null || true
    endscript

    # For apps that can't handle file rotation gracefully:
    # copytruncate           # Copy then truncate (instead of rename)
    # Why: copytruncate has a small window where log lines can be lost.
    # Prefer postrotate + signal when the app supports it.
}
CONF

    echo -e "  ${GREEN}Written to ${rotate_conf}${RESET}"
    echo ""

    explain "Manual rotation commands:"
    show_cmd "logrotate -d /etc/logrotate.conf         # Dry-run (debug mode)"
    show_cmd "logrotate -f /etc/logrotate.d/myapp      # Force immediate rotation"
    echo ""

    explain "Checking logrotate status:"
    show_cmd "cat /var/lib/logrotate/status             # Last rotation timestamps"
}

# ---------------------------------------------------------------------------
# 4. Structured Log Analysis
# ---------------------------------------------------------------------------
demo_analysis() {
    section "4. Structured Log Analysis with jq"

    local logfile
    logfile=$(generate_sample_logs)
    explain "Generated sample JSON logs: ${logfile}"
    echo ""

    if ! command -v jq &>/dev/null; then
        explain "jq not installed — showing commands and expected output"
        echo "  Install: brew install jq (macOS) or apt install jq (Linux)"
        echo ""
        show_cmd "jq -r 'select(.level==\"ERROR\") | .msg' ${logfile}"
        echo "  Connection pool exhausted"
        echo "  Invalid token"
        echo "  Internal server error"
        return
    fi

    explain "Filter by log level (all errors):"
    # Why: jq's select() is far more reliable than grep 'ERROR' because
    # grep would also match messages that happen to contain the word ERROR.
    show_cmd "jq -r 'select(.level==\"ERROR\") | \"[\\(.ts)] \\(.module): \\(.msg)\"' ${logfile}"
    jq -r 'select(.level=="ERROR") | "[\(.ts)] \(.module): \(.msg)"' "${logfile}"
    echo ""

    explain "Group and count by module:"
    show_cmd "jq -r '.module' ${logfile} | sort | uniq -c | sort -rn"
    jq -r '.module' "${logfile}" | sort | uniq -c | sort -rn
    echo ""

    explain "Extract high-latency requests (>50ms):"
    show_cmd "jq 'select(.latency_ms != null and .latency_ms > 50)' ${logfile}"
    jq -c 'select(.latency_ms != null and .latency_ms > 50) | {path, latency_ms, status}' "${logfile}"
    echo ""

    explain "Compute average latency from API requests:"
    show_cmd "jq -s '[.[] | select(.module==\"api\" and .latency_ms) | .latency_ms] | add/length' ${logfile}"
    jq -s '[.[] | select(.module=="api" and .latency_ms) | .latency_ms] | add/length' "${logfile}"
    echo ""

    explain "Find unique IP addresses from auth events:"
    show_cmd "jq -r 'select(.module==\"auth\" and .ip) | .ip' ${logfile} | sort -u"
    jq -r 'select(.module=="auth" and .ip) | .ip' "${logfile}" | sort -u
    echo ""

    explain "Classic awk one-liners for unstructured logs:"
    echo "  # Count HTTP status codes from access.log"
    show_cmd "awk '{print \$9}' /var/log/nginx/access.log | sort | uniq -c | sort -rn"
    echo "  # Top 10 IP addresses by request count"
    show_cmd "awk '{print \$1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head -10"
    echo "  # Extract 5xx errors"
    show_cmd "awk '\$9 >= 500' /var/log/nginx/access.log"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}=========================================${RESET}"
    echo -e "${BOLD} Log Management & Analysis Demo${RESET}"
    echo -e "${BOLD}=========================================${RESET}"
    echo -e "Output directory: ${OUTPUT_DIR}/\n"

    local mode="${1:---all}"
    case "${mode}" in
        --journal)  demo_journal ;;
        --rsyslog)  demo_rsyslog ;;
        --rotate)   demo_rotate ;;
        --analysis) demo_analysis ;;
        --all)
            demo_journal
            demo_rsyslog
            demo_rotate
            demo_analysis
            ;;
        *) echo "Usage: $0 [--journal|--rsyslog|--rotate|--analysis|--all]"; exit 1 ;;
    esac

    echo -e "\n${GREEN}${BOLD}Done.${RESET}"
}

main "$@"
