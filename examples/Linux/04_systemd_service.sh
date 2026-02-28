#!/usr/bin/env bash
# =============================================================================
# 04_systemd_service.sh - Systemd Service, Timer, and Socket Activation
#
# PURPOSE: Demonstrates creating custom systemd units including services,
#          timers (cron replacement), socket activation, and journal queries.
#          All operations generate unit files but only install on request.
#
# USAGE:
#   ./04_systemd_service.sh [--service|--timer|--socket|--journal|--all]
#
# MODES:
#   --service  Create and explain a custom service unit
#   --timer    Create a systemd timer (cron alternative)
#   --socket   Demonstrate socket-activated service
#   --journal  journalctl query examples
#   --all      Run all sections (default)
#
# CONCEPTS COVERED:
#   - Unit file structure: [Unit], [Service], [Install] sections
#   - Service types: simple, forking, oneshot, notify
#   - Restart policies and resource limits
#   - Timer units: OnCalendar, OnBootSec, Persistent
#   - Socket activation: lazy start on first connection
#   - Journal: structured logging, filtering, disk usage
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Why: Writing generated unit files to /tmp keeps the host system clean.
# In production, unit files go to /etc/systemd/system/ (admin-created)
# or /usr/lib/systemd/system/ (package-installed).
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/systemd_demo}"
mkdir -p "${OUTPUT_DIR}"

section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }
explain() { echo -e "${GREEN}[INFO]${RESET} $1"; }
show_cmd() { echo -e "${YELLOW}[CMD]${RESET} $1"; }

# ---------------------------------------------------------------------------
# 1. Custom Service Unit
# ---------------------------------------------------------------------------
demo_service() {
    section "1. Custom Systemd Service"

    local unit_file="${OUTPUT_DIR}/myapp.service"

    explain "Generating service unit file: ${unit_file}"
    # Why: Each section serves a distinct purpose:
    #   [Unit]    — metadata, dependencies, ordering
    #   [Service] — how to run the process
    #   [Install] — when to start (which target)
    cat > "${unit_file}" <<'UNIT'
[Unit]
# Description shown in systemctl status and journal
Description=My Application Server
# Why: After=network-online.target ensures the network is fully up.
# Using network.target only means "network config started", not "connected".
After=network-online.target
Wants=network-online.target
# Start our app only after PostgreSQL is ready
After=postgresql.service
Requires=postgresql.service

[Service]
# Why: Type=simple (default) means systemd considers the service started
# as soon as the ExecStart process is forked. Use Type=notify if the app
# signals readiness via sd_notify().
Type=simple
User=appuser
Group=appgroup
WorkingDirectory=/opt/myapp

# Environment and secrets
# Why: EnvironmentFile keeps secrets out of the unit file itself,
# which is readable by any user via systemctl cat.
EnvironmentFile=/etc/myapp/env
Environment=PYTHONUNBUFFERED=1

ExecStartPre=/opt/myapp/scripts/check_config.sh
ExecStart=/opt/myapp/venv/bin/python -m myapp.server
ExecReload=/bin/kill -HUP $MAINPID

# Restart policy
# Why: on-failure restarts only on non-zero exit, crash, or signal kill.
# always would restart even on clean exit (undesirable for oneshot tasks).
Restart=on-failure
RestartSec=5
StartLimitIntervalSec=300
StartLimitBurst=5

# Security hardening
# Why: These directives implement defense-in-depth. Even if the app is
# compromised, it cannot write to most of the filesystem or escalate.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/opt/myapp/data /var/log/myapp

# Resource limits
# Why: Without limits, a memory leak can OOM-kill unrelated services.
MemoryMax=512M
CPUQuota=80%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=myapp

[Install]
# Why: multi-user.target is the standard "system is up" target (like runlevel 3).
WantedBy=multi-user.target
UNIT

    echo -e "  ${GREEN}Written to ${unit_file}${RESET}"
    echo ""

    explain "Service types comparison:"
    echo "  simple   — ExecStart IS the main process (most common)"
    echo "  forking  — ExecStart forks a daemon; systemd tracks the PID file"
    echo "  oneshot  — Runs once to completion (e.g., initialization scripts)"
    echo "  notify   — App calls sd_notify() when ready (most reliable)"
    echo ""

    explain "Key management commands:"
    show_cmd "systemctl daemon-reload              # Reload unit files after changes"
    show_cmd "systemctl enable --now myapp.service  # Enable + start in one command"
    show_cmd "systemctl status myapp.service        # Check status and recent logs"
    show_cmd "systemctl restart myapp.service       # Stop + start"
    show_cmd "systemctl reload myapp.service        # Send SIGHUP (graceful)"
}

# ---------------------------------------------------------------------------
# 2. Timer Unit (Cron Alternative)
# ---------------------------------------------------------------------------
demo_timer() {
    section "2. Systemd Timer (Cron Alternative)"

    local timer_file="${OUTPUT_DIR}/backup.timer"
    local service_file="${OUTPUT_DIR}/backup.service"

    explain "Generating timer unit: ${timer_file}"
    # Why: Timers are superior to cron because they support:
    #   - Calendar expressions with second precision
    #   - RandomizedDelaySec (prevents thundering herd)
    #   - Persistent=true (catch up missed runs after reboot)
    #   - Dependency on other units
    #   - Journal integration for logging
    cat > "${timer_file}" <<'UNIT'
[Unit]
Description=Daily Database Backup Timer

[Timer]
# Why: OnCalendar uses systemd calendar syntax, which is more expressive
# than cron. "Mon..Fri *-*-* 02:00:00" = weekdays at 2 AM.
OnCalendar=*-*-* 02:00:00
# Spread jobs randomly over 30 minutes to avoid I/O spikes
RandomizedDelaySec=1800
# If the system was off at 2 AM, run backup on next boot
Persistent=true
# Accuracy: allow systemd to coalesce with nearby timers to save wakeups
AccuracySec=60

[Install]
WantedBy=timers.target
UNIT

    explain "Generating companion service: ${service_file}"
    cat > "${service_file}" <<'UNIT'
[Unit]
Description=Daily Database Backup

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup-db.sh
User=backup
Nice=19
IOSchedulingClass=idle
UNIT

    echo -e "  ${GREEN}Written to ${timer_file} and ${service_file}${RESET}"
    echo ""

    explain "Useful timer commands:"
    show_cmd "systemctl list-timers --all          # Show all active timers"
    show_cmd "systemctl enable --now backup.timer   # Activate the timer"
    show_cmd "journalctl -u backup.service          # Check last backup run"
    echo ""

    explain "OnCalendar syntax examples:"
    echo "  *-*-* 00:00:00          — Every day at midnight"
    echo "  Mon *-*-* 09:00:00      — Every Monday at 9 AM"
    echo "  *-*-01 00:00:00         — First day of every month"
    echo "  *:0/15                  — Every 15 minutes"
    echo ""
    show_cmd "systemd-analyze calendar 'Mon..Fri *-*-* 02:00:00'  # Validate syntax"
}

# ---------------------------------------------------------------------------
# 3. Socket Activation
# ---------------------------------------------------------------------------
demo_socket() {
    section "3. Socket Activation"

    local socket_file="${OUTPUT_DIR}/myapp.socket"
    local service_file="${OUTPUT_DIR}/myapp-socket.service"

    explain "Socket activation: systemd listens on a port and starts the service"
    explain "only when the first connection arrives. Benefits:"
    echo "  - Faster boot (services start on demand)"
    echo "  - Zero-downtime restarts (socket buffers during restart)"
    echo "  - Dependency ordering without explicit After= chains"
    echo ""

    explain "Generating socket unit: ${socket_file}"
    cat > "${socket_file}" <<'UNIT'
[Unit]
Description=My App Socket

[Socket]
# Why: ListenStream creates a TCP socket. Systemd holds it open and passes
# the file descriptor to the service via socket activation protocol.
ListenStream=8080
# Why: Accept=false means one service instance handles all connections
# (typical for web servers). Accept=true spawns per-connection instances.
Accept=false
# Limit connection backlog
Backlog=128

[Install]
WantedBy=sockets.target
UNIT

    explain "Generating socket-activated service: ${service_file}"
    cat > "${service_file}" <<'UNIT'
[Unit]
Description=My App (Socket Activated)
# Why: The service requires its socket. If the socket is stopped,
# the service will be stopped too.
Requires=myapp.socket

[Service]
Type=simple
ExecStart=/opt/myapp/bin/server --fd 3
# Why: NonBlocking=true sets O_NONBLOCK on the inherited fd, which
# most async frameworks (asyncio, tokio, libuv) require.
NonBlocking=true
# Restart just the service, not the socket (zero-downtime pattern)
Restart=on-failure
UNIT

    echo -e "  ${GREEN}Written to ${socket_file} and ${service_file}${RESET}"
    echo ""
    show_cmd "systemctl enable --now myapp.socket   # Listen without starting service"
    show_cmd "curl http://localhost:8080/            # First request triggers service start"
}

# ---------------------------------------------------------------------------
# 4. Journal Queries
# ---------------------------------------------------------------------------
demo_journal() {
    section "4. Journalctl Queries"

    explain "The systemd journal replaces syslog with structured, indexed logging."
    echo ""

    explain "Basic filtering:"
    show_cmd "journalctl -u myapp.service              # Logs for one service"
    show_cmd "journalctl -u myapp.service --since '1h ago'"
    show_cmd "journalctl -u myapp.service -p err       # Only errors and above"
    echo ""

    explain "Priority levels (syslog-compatible):"
    echo "  0=emerg  1=alert  2=crit  3=err  4=warning  5=notice  6=info  7=debug"
    echo ""

    explain "Kernel messages:"
    show_cmd "journalctl -k                            # Equivalent to dmesg"
    show_cmd "journalctl -k -p err                     # Kernel errors only"
    echo ""

    explain "Follow mode (like tail -f):"
    show_cmd "journalctl -u myapp.service -f"
    echo ""

    explain "Output formats:"
    show_cmd "journalctl -u myapp -o json-pretty       # Full structured data"
    show_cmd "journalctl -u myapp -o short-iso         # ISO timestamps"
    show_cmd "journalctl -u myapp -o cat               # Message text only"
    echo ""

    explain "Boot-scoped queries:"
    show_cmd "journalctl -b                            # Current boot only"
    show_cmd "journalctl -b -1                         # Previous boot"
    show_cmd "journalctl --list-boots                  # List all recorded boots"
    echo ""

    explain "Disk usage management:"
    show_cmd "journalctl --disk-usage                  # Total journal size"
    show_cmd "journalctl --vacuum-size=500M            # Shrink to 500 MB"
    show_cmd "journalctl --vacuum-time=30d             # Remove logs older than 30 days"
    echo ""

    # Why: Showing a live example (if journalctl is available) makes this
    # tangible rather than purely theoretical.
    if command -v journalctl &>/dev/null; then
        explain "Live example — last 5 system log entries:"
        journalctl --no-pager -n 5 -o short-iso 2>/dev/null || echo "  (insufficient permissions)"
    else
        explain "(journalctl not available on this system — macOS uses log show instead)"
        show_cmd "log show --last 5m --predicate 'eventMessage contains \"error\"'"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}==========================================${RESET}"
    echo -e "${BOLD} Systemd Service, Timer & Socket Demo${RESET}"
    echo -e "${BOLD}==========================================${RESET}"
    echo -e "Generated unit files → ${OUTPUT_DIR}/\n"

    local mode="${1:---all}"
    case "${mode}" in
        --service) demo_service ;;
        --timer)   demo_timer ;;
        --socket)  demo_socket ;;
        --journal) demo_journal ;;
        --all)
            demo_service
            demo_timer
            demo_socket
            demo_journal
            ;;
        *) echo "Usage: $0 [--service|--timer|--socket|--journal|--all]"; exit 1 ;;
    esac

    echo -e "\n${GREEN}${BOLD}Generated files:${RESET}"
    ls -la "${OUTPUT_DIR}"/*.service "${OUTPUT_DIR}"/*.timer "${OUTPUT_DIR}"/*.socket 2>/dev/null || true
    echo -e "\n${GREEN}${BOLD}Done.${RESET}"
}

main "$@"
