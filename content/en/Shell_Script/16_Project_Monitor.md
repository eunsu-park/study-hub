# Project: System Monitoring Tool

**Difficulty**: ⭐⭐⭐⭐

**Previous**: [Project: Deployment Automation](./15_Project_Deployment.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Build a real-time terminal dashboard using `tput` for cursor control, color formatting, and responsive layout
2. Implement system metric collectors that parse CPU, memory, disk, and network data from `/proc` and standard Unix utilities
3. Configure a threshold-based alerting system with cooldown periods to prevent notification spam
4. Send alert notifications to Slack or Discord webhooks and via email from a shell script
5. Write a log aggregation module that parses system logs, detects error patterns, and summarizes findings
6. Integrate monitoring scripts with cron for periodic health checks and automated HTML report generation
7. Handle terminal resize events (`SIGWINCH`) to adapt the dashboard layout dynamically

---

When a production server runs out of disk space at 3 AM or CPU usage spikes during a traffic surge, you need monitoring in place before the problem occurs. Commercial tools like Datadog and Prometheus are powerful but require infrastructure of their own. A bash-based monitoring tool runs on any server with zero dependencies, gives you full control over what metrics to collect and how to alert, and serves as a capstone project that integrates terminal UI, signal handling, process management, and system internals from every previous lesson.

## 1. Overview

### What We're Building

A comprehensive terminal-based system monitoring dashboard written in pure bash. This tool provides real-time visibility into:

- CPU usage and load averages
- Memory and swap utilization
- Disk space across filesystems
- Network I/O statistics
- Active processes and their resource consumption
- System logs with error detection
- Alert notifications when thresholds are exceeded

Think of it as a lightweight, customizable alternative to tools like `htop`, `glances`, or Datadog agents — but completely self-contained in bash.

### Key Features

1. **Real-time Dashboard**: Live-updating terminal UI with colors and layouts
2. **Configurable Alerts**: Threshold-based warnings sent to Slack/Discord webhooks
3. **Log Aggregation**: Parse and summarize system logs for errors/warnings
4. **Cron Integration**: Run periodic checks and generate reports
5. **Cross-platform**: Works on Linux and macOS (with OS-specific adaptations)
6. **Zero Dependencies**: Pure bash using standard Unix tools

### Why Pure Bash?

- **Universality**: Runs on any server with bash installed
- **Transparency**: No black-box monitoring agents
- **Customization**: Easily modify metrics, thresholds, and output format
- **Learning**: Demonstrates advanced bash techniques (tput, /proc parsing, signal handling)

---

## 2. Design

### Architecture

```
Monitor System
├── Data Collection
│   ├── CPU metrics (/proc/stat, top)
│   ├── Memory metrics (/proc/meminfo, free)
│   ├── Disk metrics (df)
│   ├── Network metrics (/proc/net/dev)
│   └── Process metrics (ps, top)
│
├── Display Engine
│   ├── Terminal UI (tput)
│   ├── Color formatting
│   ├── Layout management
│   └── Responsive design (SIGWINCH)
│
├── Alerting System
│   ├── Threshold configuration
│   ├── Alert cooldown (prevent spam)
│   ├── Webhook notifications (Slack, Discord)
│   └── Email alerts (mail command)
│
└── Reporting
    ├── Log parsing
    ├── Historical data collection
    ├── HTML report generation
    └── CSV export
```

### Data Flow

```
1. Collect metrics from system APIs (/proc, df, ps)
2. Parse and normalize data
3. Compare against thresholds
4. If threshold exceeded:
   a. Check cooldown period
   b. Send alert via webhook/email
   c. Record alert event
5. Format data for terminal display
6. Render to screen using tput
7. Wait refresh interval
8. Repeat
```

### Configuration

Thresholds and settings stored in a config file:

```bash
# monitor.conf
REFRESH_INTERVAL=2      # Seconds between updates
CPU_ALERT_THRESHOLD=80  # Percent
MEM_ALERT_THRESHOLD=85  # Percent
DISK_ALERT_THRESHOLD=90 # Percent
ALERT_COOLDOWN=300      # Seconds (5 minutes)
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

---

## 3. Terminal UI with tput

### What is tput?

`tput` is a terminal control utility that manipulates cursor position, colors, and screen clearing using the terminfo database.

### Essential tput Commands

```bash
# Cursor movement
tput cup Y X          # Move cursor to row Y, column X (0-indexed)
tput cuu N            # Move cursor up N lines
tput cud N            # Move cursor down N lines

# Screen control
tput clear            # Clear entire screen
tput el               # Clear to end of line
tput ed               # Clear to end of screen

# Colors
tput setaf N          # Set foreground color (0-7)
tput setab N          # Set background color (0-7)
tput bold             # Enable bold
tput sgr0             # Reset all attributes

# Terminal info
tput cols             # Get number of columns
tput lines            # Get number of rows
```

### Color Reference

```bash
# ANSI color codes (tput setaf)
0 = Black
1 = Red
2 = Green
3 = Yellow
4 = Blue
5 = Magenta
6 = Cyan
7 = White
```

### Building a Dashboard Layout

```bash
#!/usr/bin/env bash

# Clear screen and hide cursor
tput clear
tput civis  # Hide cursor

# Draw header
tput cup 0 0
tput bold; tput setaf 6  # Bold cyan
echo "╔════════════════════════════════════════════════════════════════╗"
tput cup 1 0
echo "║          SYSTEM MONITOR - $(hostname)                          ║"
tput cup 2 0
echo "╚════════════════════════════════════════════════════════════════╝"
tput sgr0

# Draw CPU section
tput cup 4 0
tput setaf 3; echo "CPU Usage:"
tput sgr0
tput cup 5 2
echo "Load Average: 1.23, 0.98, 0.76"
tput cup 6 2
echo "CPU: [████████░░] 78%"

# Draw Memory section
tput cup 8 0
tput setaf 3; echo "Memory:"
tput sgr0
tput cup 9 2
echo "Used: 12.4 GB / 16.0 GB (77%)"

# Show cursor again before exiting
tput cnorm
```

### Progress Bars

Visual representation of percentages:

```bash
draw_bar() {
    local percent="$1"
    local width="${2:-50}"
    local filled=$(( percent * width / 100 ))
    local empty=$(( width - filled ))

    # Color based on threshold
    if [ "$percent" -ge 90 ]; then
        tput setaf 1  # Red
    elif [ "$percent" -ge 70 ]; then
        tput setaf 3  # Yellow
    else
        tput setaf 2  # Green
    fi

    # Draw bar
    printf "["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%%\n" "$percent"

    tput sgr0
}

# Usage
draw_bar 78 40
```

### Handling Terminal Resize (SIGWINCH)

Redraw on terminal size change:

```bash
resize_handler() {
    TERM_COLS=$(tput cols)
    TERM_LINES=$(tput lines)
    tput clear
    # Redraw entire dashboard
}

trap resize_handler SIGWINCH

# Get initial terminal size
TERM_COLS=$(tput cols)
TERM_LINES=$(tput lines)
```

---

## 4. System Metrics Collection

### CPU Usage

Calculate CPU usage from `/proc/stat`:

```bash
get_cpu_usage() {
    # Read /proc/stat twice with a delay to calculate usage
    local prev_idle prev_total idle total

    # First reading
    read -r cpu prev_idle prev_total <<< $(awk '/^cpu / {
        idle = $5
        total = 0
        for (i=2; i<=NF; i++) total += $i
        print idle, total
    }' /proc/stat)

    sleep 0.5

    # Second reading
    read -r cpu idle total <<< $(awk '/^cpu / {
        idle = $5
        total = 0
        for (i=2; i<=NF; i++) total += $i
        print idle, total
    }' /proc/stat)

    # Calculate percentage
    local idle_delta=$((idle - prev_idle))
    local total_delta=$((total - prev_total))
    local usage=$((100 * (total_delta - idle_delta) / total_delta))

    echo "$usage"
}
```

**macOS alternative** (using `top`):

```bash
get_cpu_usage_macos() {
    top -l 2 -n 0 -F -s 1 | \
        awk '/^CPU usage:/ {print 100 - $7}' | \
        tail -1 | \
        cut -d. -f1
}
```

### Load Average

```bash
get_load_average() {
    awk '{print $1, $2, $3}' /proc/loadavg
}

# macOS compatible version
get_load_average_cross() {
    uptime | awk -F'load averages?: ' '{print $2}'
}
```

### Memory Usage

```bash
get_memory_usage() {
    # Linux: /proc/meminfo
    if [ -f /proc/meminfo ]; then
        awk '/^MemTotal:/ {total=$2}
             /^MemAvailable:/ {avail=$2}
             END {
                 used = total - avail
                 percent = int(100 * used / total)
                 printf "%.1f,%.1f,%d\n", used/1024/1024, total/1024/1024, percent
             }' /proc/meminfo
    else
        # macOS: use vm_stat
        vm_stat | awk '
            /Pages free/ {free=$3}
            /Pages active/ {active=$3}
            /Pages inactive/ {inactive=$3}
            /Pages speculative/ {spec=$3}
            /Pages wired/ {wired=$3}
            END {
                # Page size is typically 4096 bytes
                total_gb = (free + active + inactive + spec + wired) * 4096 / 1024 / 1024 / 1024
                used_gb = (active + wired) * 4096 / 1024 / 1024 / 1024
                percent = int(100 * used_gb / total_gb)
                printf "%.1f,%.1f,%d\n", used_gb, total_gb, percent
            }'
    fi
}
```

### Disk Usage

```bash
get_disk_usage() {
    df -h / | awk 'NR==2 {
        gsub(/%/, "", $5)
        print $3, $2, $5
    }'
}

# All mounted filesystems
get_all_disks() {
    df -h | awk 'NR>1 && $1 ~ /^\// {
        gsub(/%/, "", $5)
        printf "%-20s %8s / %8s (%3d%%)\n", $6, $3, $2, $5
    }'
}
```

### Network I/O

```bash
get_network_io() {
    local interface="${1:-eth0}"

    if [ -f /proc/net/dev ]; then
        # Linux
        awk -v iface="$interface" '
            $1 == iface":" {
                printf "RX: %d bytes, TX: %d bytes\n", $2, $10
            }' /proc/net/dev
    else
        # macOS: use netstat
        netstat -ib | awk -v iface="$interface" '
            $1 == iface && $3 ~ /Link/ {
                printf "RX: %d bytes, TX: %d bytes\n", $7, $10
            }'
    fi
}
```

### Process Metrics

```bash
get_top_processes() {
    local count="${1:-5}"

    # Linux
    ps aux --sort=-%mem | head -n $((count + 1)) | awk 'NR>1 {
        printf "%-10s %5.1f%% %5.1f%% %s\n", $1, $3, $4, $11
    }'
}

# Top CPU consumers
get_top_cpu_processes() {
    ps aux --sort=-%cpu | head -6 | awk 'NR>1 {
        printf "%-15s %6.1f%% %s\n", $1, $3, $11
    }'
}
```

---

## 5. Alerting System

### Threshold Configuration

```bash
# Default thresholds
CPU_ALERT_THRESHOLD="${CPU_ALERT_THRESHOLD:-80}"
MEM_ALERT_THRESHOLD="${MEM_ALERT_THRESHOLD:-85}"
DISK_ALERT_THRESHOLD="${DISK_ALERT_THRESHOLD:-90}"
LOAD_ALERT_THRESHOLD="${LOAD_ALERT_THRESHOLD:-4.0}"

# Alert cooldown (don't spam)
ALERT_COOLDOWN="${ALERT_COOLDOWN:-300}"  # 5 minutes
ALERT_HISTORY="/tmp/monitor_alerts.log"
```

### Alert Check Functions

```bash
check_cpu_threshold() {
    local cpu_usage="$1"

    if [ "$cpu_usage" -ge "$CPU_ALERT_THRESHOLD" ]; then
        if should_send_alert "cpu"; then
            send_alert "CPU" "CPU usage is ${cpu_usage}% (threshold: ${CPU_ALERT_THRESHOLD}%)"
            record_alert "cpu"
        fi
    fi
}

check_memory_threshold() {
    local mem_percent="$1"

    if [ "$mem_percent" -ge "$MEM_ALERT_THRESHOLD" ]; then
        if should_send_alert "memory"; then
            send_alert "MEMORY" "Memory usage is ${mem_percent}% (threshold: ${MEM_ALERT_THRESHOLD}%)"
            record_alert "memory"
        fi
    fi
}

check_disk_threshold() {
    local disk_percent="$1"

    if [ "$disk_percent" -ge "$DISK_ALERT_THRESHOLD" ]; then
        if should_send_alert "disk"; then
            send_alert "DISK" "Disk usage is ${disk_percent}% (threshold: ${DISK_ALERT_THRESHOLD}%)"
            record_alert "disk"
        fi
    fi
}
```

### Alert Cooldown

Prevent alert spam using cooldown periods:

```bash
should_send_alert() {
    local alert_type="$1"
    local now=$(date +%s)
    local last_alert_time=0

    # Check last alert time
    if [ -f "$ALERT_HISTORY" ]; then
        last_alert_time=$(awk -v type="$alert_type" \
            '$1 == type {print $2}' "$ALERT_HISTORY" 2>/dev/null || echo 0)
    fi

    local time_since_alert=$((now - last_alert_time))

    if [ "$time_since_alert" -ge "$ALERT_COOLDOWN" ]; then
        return 0  # OK to send
    else
        return 1  # Too soon
    fi
}

record_alert() {
    local alert_type="$1"
    local now=$(date +%s)

    # Update or append alert record
    if [ -f "$ALERT_HISTORY" ]; then
        # Update existing entry or add new one
        grep -v "^${alert_type} " "$ALERT_HISTORY" > "$ALERT_HISTORY.tmp" 2>/dev/null || true
        echo "${alert_type} ${now}" >> "$ALERT_HISTORY.tmp"
        mv "$ALERT_HISTORY.tmp" "$ALERT_HISTORY"
    else
        echo "${alert_type} ${now}" > "$ALERT_HISTORY"
    fi
}
```

### Webhook Notifications (Slack/Discord)

```bash
send_slack_alert() {
    local title="$1"
    local message="$2"
    local webhook_url="${SLACK_WEBHOOK_URL}"

    if [ -z "$webhook_url" ]; then
        return 0
    fi

    local payload=$(cat <<EOF
{
    "username": "System Monitor",
    "icon_emoji": ":warning:",
    "attachments": [{
        "color": "danger",
        "title": "${title}",
        "text": "${message}",
        "footer": "$(hostname)",
        "ts": $(date +%s)
    }]
}
EOF
)

    curl -X POST -H 'Content-type: application/json' \
        --data "$payload" \
        "$webhook_url" &>/dev/null
}

send_discord_alert() {
    local title="$1"
    local message="$2"
    local webhook_url="${DISCORD_WEBHOOK_URL}"

    if [ -z "$webhook_url" ]; then
        return 0
    fi

    local payload=$(cat <<EOF
{
    "username": "System Monitor",
    "embeds": [{
        "title": "${title}",
        "description": "${message}",
        "color": 15158332,
        "footer": {
            "text": "$(hostname)"
        },
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    }]
}
EOF
)

    curl -X POST -H 'Content-type: application/json' \
        --data "$payload" \
        "$webhook_url" &>/dev/null
}

send_alert() {
    local title="$1"
    local message="$2"

    # Try Slack
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        send_slack_alert "$title" "$message"
    fi

    # Try Discord
    if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
        send_discord_alert "$title" "$message"
    fi

    # Fallback to email
    if command -v mail &>/dev/null && [ -n "${ALERT_EMAIL:-}" ]; then
        echo "$message" | mail -s "[ALERT] $title" "$ALERT_EMAIL"
    fi

    # Log locally
    echo "[$(date)] $title: $message" >> /var/log/monitor_alerts.log
}
```

---

## 6. Log Aggregation

### Parsing System Logs

```bash
parse_syslog_errors() {
    local log_file="/var/log/syslog"
    local time_range="${1:-1h}"  # Last 1 hour

    # Convert time range to minutes
    local minutes=60
    case "$time_range" in
        *h) minutes=$((${time_range%h} * 60)) ;;
        *m) minutes=${time_range%m} ;;
    esac

    # Get timestamp for filtering
    local since=$(date -d "$minutes minutes ago" '+%b %d %H:%M' 2>/dev/null || \
                  date -v-${minutes}M '+%b %d %H:%M')

    if [ -f "$log_file" ]; then
        awk -v since="$since" '
            $0 ~ since {found=1}
            found && /error|fail|critical/i {print}
        ' "$log_file"
    fi
}

# Count error types
count_log_errors() {
    local log_file="${1:-/var/log/syslog}"

    if [ ! -f "$log_file" ]; then
        echo "Log file not found: $log_file"
        return 1
    fi

    awk '
        /ERROR/ {error++}
        /WARNING/ {warning++}
        /CRITICAL/ {critical++}
        /FATAL/ {fatal++}
        END {
            printf "CRITICAL: %d, FATAL: %d, ERROR: %d, WARNING: %d\n",
                   critical, fatal, error, warning
        }
    ' "$log_file"
}
```

### Application Log Parsing

```bash
parse_app_logs() {
    local app_log="/var/log/myapp/app.log"
    local lines="${1:-100}"

    if [ ! -f "$app_log" ]; then
        return 0
    fi

    # Get recent errors
    tail -n "$lines" "$app_log" | \
    grep -E "ERROR|Exception|Traceback" | \
    tail -10
}

# Summarize log patterns
summarize_logs() {
    local log_file="$1"

    echo "=== Log Summary ==="
    echo ""

    # Error counts by hour
    echo "Errors per hour (last 24h):"
    awk '/ERROR/ {
        match($0, /([0-9]{2}):/, hour)
        if (hour[1]) hours[hour[1]]++
    }
    END {
        for (h in hours) printf "  %s:00 - %d errors\n", h, hours[h]
    }' "$log_file" | sort

    echo ""

    # Top error messages
    echo "Top 5 error patterns:"
    grep "ERROR" "$log_file" | \
        awk -F'ERROR' '{print $2}' | \
        sort | uniq -c | sort -rn | head -5
}
```

---

## 7. Complete Monitor Script

Here's the full monitoring dashboard implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# System Monitor - Real-time Dashboard
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/monitor.conf"

# Defaults
REFRESH_INTERVAL="${REFRESH_INTERVAL:-2}"
CPU_ALERT_THRESHOLD="${CPU_ALERT_THRESHOLD:-80}"
MEM_ALERT_THRESHOLD="${MEM_ALERT_THRESHOLD:-85}"
DISK_ALERT_THRESHOLD="${DISK_ALERT_THRESHOLD:-90}"
ALERT_COOLDOWN="${ALERT_COOLDOWN:-300}"

# Load config if exists
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# ============================================================================
# Terminal Setup
# ============================================================================

setup_terminal() {
    tput clear
    tput civis  # Hide cursor
    trap cleanup EXIT
    trap resize_handler SIGWINCH
}

cleanup() {
    tput clear
    tput cnorm  # Show cursor
    tput sgr0   # Reset colors
}

resize_handler() {
    TERM_COLS=$(tput cols)
    TERM_LINES=$(tput lines)
    tput clear
}

# ============================================================================
# Metrics Collection
# ============================================================================

get_cpu_usage() {
    if [ -f /proc/stat ]; then
        local prev_idle prev_total idle total

        read -r cpu prev_idle prev_total <<< $(awk '/^cpu / {
            idle = $5; total = 0
            for (i=2; i<=NF; i++) total += $i
            print idle, total
        }' /proc/stat)

        sleep 0.2

        read -r cpu idle total <<< $(awk '/^cpu / {
            idle = $5; total = 0
            for (i=2; i<=NF; i++) total += $i
            print idle, total
        }' /proc/stat)

        local idle_delta=$((idle - prev_idle))
        local total_delta=$((total - prev_total))
        echo $((100 * (total_delta - idle_delta) / total_delta))
    else
        # macOS fallback
        echo 50
    fi
}

get_memory_info() {
    if [ -f /proc/meminfo ]; then
        awk '/^MemTotal:/ {total=$2}
             /^MemAvailable:/ {avail=$2}
             END {
                 used = total - avail
                 percent = int(100 * used / total)
                 printf "%.1f %.1f %d\n", used/1024/1024, total/1024/1024, percent
             }' /proc/meminfo
    else
        echo "8.0 16.0 50"
    fi
}

get_disk_info() {
    df -h / | awk 'NR==2 {
        gsub(/%/, "", $5)
        print $3, $2, $5
    }'
}

get_load_avg() {
    if [ -f /proc/loadavg ]; then
        awk '{print $1, $2, $3}' /proc/loadavg
    else
        uptime | awk -F'load averages?: ' '{print $2}' | awk '{print $1, $2, $3}' | tr -d ','
    fi
}

get_network_stats() {
    local interface="${1:-eth0}"

    if [ -f /proc/net/dev ]; then
        awk -v iface="$interface" '
            $1 == iface":" {
                rx = $2 / 1024 / 1024
                tx = $10 / 1024 / 1024
                printf "%.2f %.2f\n", rx, tx
            }' /proc/net/dev
    else
        echo "0.00 0.00"
    fi
}

# ============================================================================
# Display Functions
# ============================================================================

draw_header() {
    tput cup 0 0
    tput bold; tput setaf 6
    local width=$((TERM_COLS - 1))
    printf "╔%${width}s╗\n" | tr ' ' '═'
    printf "║ %-$((width - 2))s ║\n" "SYSTEM MONITOR - $(hostname) - $(date +'%Y-%m-%d %H:%M:%S')"
    printf "╚%${width}s╝\n" | tr ' ' '═'
    tput sgr0
}

draw_bar() {
    local percent="$1"
    local width="${2:-40}"
    local filled=$(( percent * width / 100 ))
    local empty=$(( width - filled ))

    if [ "$percent" -ge 90 ]; then
        tput setaf 1
    elif [ "$percent" -ge 70 ]; then
        tput setaf 3
    else
        tput setaf 2
    fi

    printf "["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%%\n" "$percent"

    tput sgr0
}

draw_metrics() {
    local row=4

    # CPU
    tput cup $row 2
    tput bold; tput setaf 3
    echo "CPU Usage:"
    tput sgr0
    row=$((row + 1))

    local cpu_usage=$(get_cpu_usage)
    tput cup $row 4
    draw_bar "$cpu_usage" 50

    row=$((row + 1))
    read -r load1 load5 load15 <<< $(get_load_avg)
    tput cup $row 4
    printf "Load Average: %.2f, %.2f, %.2f\n" "$load1" "$load5" "$load15"

    # Memory
    row=$((row + 2))
    tput cup $row 2
    tput bold; tput setaf 3
    echo "Memory:"
    tput sgr0
    row=$((row + 1))

    read -r mem_used mem_total mem_percent <<< $(get_memory_info)
    tput cup $row 4
    printf "Used: %.1f GB / %.1f GB\n" "$mem_used" "$mem_total"
    row=$((row + 1))
    tput cup $row 4
    draw_bar "$mem_percent" 50

    # Disk
    row=$((row + 2))
    tput cup $row 2
    tput bold; tput setaf 3
    echo "Disk (/):"
    tput sgr0
    row=$((row + 1))

    read -r disk_used disk_total disk_percent <<< $(get_disk_info)
    tput cup $row 4
    printf "Used: %s / %s\n" "$disk_used" "$disk_total"
    row=$((row + 1))
    tput cup $row 4
    draw_bar "$disk_percent" 50

    # Network
    row=$((row + 2))
    tput cup $row 2
    tput bold; tput setaf 3
    echo "Network (eth0):"
    tput sgr0
    row=$((row + 1))

    read -r rx_mb tx_mb <<< $(get_network_stats "eth0")
    tput cup $row 4
    printf "RX: %.2f MB | TX: %.2f MB\n" "$rx_mb" "$tx_mb"

    # Top Processes
    row=$((row + 2))
    tput cup $row 2
    tput bold; tput setaf 3
    echo "Top CPU Processes:"
    tput sgr0
    row=$((row + 1))

    tput cup $row 4
    ps aux --sort=-%cpu 2>/dev/null | head -6 | tail -5 | \
        awk '{printf "%-12s %5.1f%% %s\n", $1, $3, $11}' | \
        while IFS= read -r line; do
            tput cup $row 4
            echo "$line"
            row=$((row + 1))
        done

    # Footer
    tput cup $((TERM_LINES - 2)) 2
    tput setaf 8
    echo "Press Ctrl+C to exit | Refresh: ${REFRESH_INTERVAL}s"
    tput sgr0
}

# ============================================================================
# Alert Functions
# ============================================================================

check_thresholds() {
    local cpu_usage="$1"
    local mem_percent="$2"
    local disk_percent="$3"

    [ "$cpu_usage" -ge "$CPU_ALERT_THRESHOLD" ] && \
        send_alert "High CPU Usage" "CPU at ${cpu_usage}% (threshold: ${CPU_ALERT_THRESHOLD}%)"

    [ "$mem_percent" -ge "$MEM_ALERT_THRESHOLD" ] && \
        send_alert "High Memory Usage" "Memory at ${mem_percent}% (threshold: ${MEM_ALERT_THRESHOLD}%)"

    [ "$disk_percent" -ge "$DISK_ALERT_THRESHOLD" ] && \
        send_alert "Low Disk Space" "Disk at ${disk_percent}% (threshold: ${DISK_ALERT_THRESHOLD}%)"
}

send_alert() {
    local title="$1"
    local message="$2"

    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"[$(hostname)] ${title}: ${message}\"}" \
            "$SLACK_WEBHOOK_URL" &>/dev/null &
    fi

    echo "[$(date)] ALERT: $title - $message" >> /var/log/monitor.log
}

# ============================================================================
# Main Loop
# ============================================================================

main() {
    setup_terminal

    TERM_COLS=$(tput cols)
    TERM_LINES=$(tput lines)

    while true; do
        draw_header
        draw_metrics

        # Collect current values for alerting
        cpu_usage=$(get_cpu_usage)
        read -r mem_used mem_total mem_percent <<< $(get_memory_info)
        read -r disk_used disk_total disk_percent <<< $(get_disk_info)

        # Check thresholds (in background to avoid blocking)
        check_thresholds "$cpu_usage" "$mem_percent" "$disk_percent" &

        sleep "$REFRESH_INTERVAL"
    done
}

main "$@"
```

---

## 8. Cron Integration

### Periodic Monitoring

Run health checks every 5 minutes:

```bash
# Crontab entry
*/5 * * * * /opt/monitor/monitor.sh --check-only >> /var/log/monitor_cron.log 2>&1
```

Monitor script with `--check-only` mode:

```bash
if [ "${1:-}" = "--check-only" ]; then
    # Collect metrics
    cpu=$(get_cpu_usage)
    read mem_used mem_total mem_percent <<< $(get_memory_info)
    read disk_used disk_total disk_percent <<< $(get_disk_info)

    # Check thresholds
    check_thresholds "$cpu" "$mem_percent" "$disk_percent"

    # Log
    echo "[$(date)] CPU: ${cpu}% | MEM: ${mem_percent}% | DISK: ${disk_percent}%"

    exit 0
fi
```

### Generating Reports

Daily HTML report:

```bash
generate_html_report() {
    local output="/var/www/html/monitor_report.html"

    cat > "$output" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>System Monitor Report</title>
    <style>
        body { font-family: monospace; margin: 20px; }
        .metric { margin: 10px 0; }
        .bar { background: #e0e0e0; height: 20px; width: 400px; position: relative; }
        .bar-fill { background: #4CAF50; height: 100%; }
        .bar-fill.warning { background: #FFC107; }
        .bar-fill.danger { background: #F44336; }
    </style>
</head>
<body>
    <h1>System Monitor Report - $(hostname)</h1>
    <p>Generated: $(date)</p>

    <div class="metric">
        <strong>CPU Usage:</strong> ${cpu_usage}%
        <div class="bar">
            <div class="bar-fill" style="width: ${cpu_usage}%"></div>
        </div>
    </div>

    <div class="metric">
        <strong>Memory:</strong> ${mem_used} GB / ${mem_total} GB (${mem_percent}%)
        <div class="bar">
            <div class="bar-fill" style="width: ${mem_percent}%"></div>
        </div>
    </div>

    <div class="metric">
        <strong>Disk:</strong> ${disk_used} / ${disk_total} (${disk_percent}%)
        <div class="bar">
            <div class="bar-fill" style="width: ${disk_percent}%"></div>
        </div>
    </div>
</body>
</html>
EOF

    echo "Report generated: $output"
}
```

### CSV Export for Historical Data

```bash
log_metrics_csv() {
    local csv_file="/var/log/monitor_metrics.csv"

    # Create header if file doesn't exist
    if [ ! -f "$csv_file" ]; then
        echo "timestamp,cpu_percent,mem_percent,disk_percent,load_1min" > "$csv_file"
    fi

    # Collect metrics
    local cpu=$(get_cpu_usage)
    read -r mem_used mem_total mem_percent <<< $(get_memory_info)
    read -r disk_used disk_total disk_percent <<< $(get_disk_info)
    read -r load1 load5 load15 <<< $(get_load_avg)

    # Append to CSV
    echo "$(date +%s),$cpu,$mem_percent,$disk_percent,$load1" >> "$csv_file"
}
```

---

## 9. Usage Examples

### Run the Dashboard

```bash
chmod +x monitor.sh
./monitor.sh
```

### Configure Alerts

Edit `monitor.conf`:

```bash
# monitor.conf
REFRESH_INTERVAL=2
CPU_ALERT_THRESHOLD=80
MEM_ALERT_THRESHOLD=85
DISK_ALERT_THRESHOLD=90
ALERT_COOLDOWN=300

SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
ALERT_EMAIL="ops@example.com"
```

### Cron Setup for Periodic Checks

```bash
# Install cron job
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/monitor/monitor.sh --check-only") | crontab -

# Daily HTML report at 8 AM
(crontab -l 2>/dev/null; echo "0 8 * * * /opt/monitor/generate_report.sh") | crontab -
```

---

## 10. Extensions

### 1. Remote Monitoring

Monitor multiple servers via SSH:

```bash
monitor_remote() {
    local hosts=("web01" "web02" "db01")

    for host in "${hosts[@]}"; do
        echo "=== ${host} ==="
        ssh "$host" 'bash -s' < monitor.sh --check-only
        echo ""
    done
}
```

### 2. Web Dashboard

Serve real-time metrics over HTTP:

```bash
# Simple HTTP server that serves JSON metrics
while true; do
    echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{
        \"cpu\": $(get_cpu_usage),
        \"memory\": $(get_memory_info | awk '{print $3}'),
        \"disk\": $(get_disk_info | awk '{print $3}')
    }" | nc -l -p 8080 -q 1
done
```

### 3. Database Storage

Store metrics in SQLite for historical analysis:

```bash
store_metrics() {
    local db="/var/lib/monitor/metrics.db"

    sqlite3 "$db" <<EOF
CREATE TABLE IF NOT EXISTS metrics (
    timestamp INTEGER,
    cpu_percent INTEGER,
    mem_percent INTEGER,
    disk_percent INTEGER
);
INSERT INTO metrics VALUES (
    $(date +%s),
    $(get_cpu_usage),
    $(get_memory_info | awk '{print $3}'),
    $(get_disk_info | awk '{print $3}')
);
EOF
}
```

### 4. Grafana-Compatible Metrics

Export metrics in Prometheus format:

```bash
# Expose metrics at http://localhost:9090/metrics
serve_prometheus_metrics() {
    while true; do
        cat <<EOF | nc -l -p 9090 -q 1
HTTP/1.1 200 OK
Content-Type: text/plain

# HELP system_cpu_usage CPU usage percentage
# TYPE system_cpu_usage gauge
system_cpu_usage $(get_cpu_usage)

# HELP system_memory_usage Memory usage percentage
# TYPE system_memory_usage gauge
system_memory_usage $(get_memory_info | awk '{print $3}')

# HELP system_disk_usage Disk usage percentage
# TYPE system_disk_usage gauge
system_disk_usage $(get_disk_info | awk '{print $3}')
EOF
    done
}
```

## Exercises

### Exercise 1: Implement Cross-Platform Metric Collectors

The monitoring tool uses `/proc` files which are Linux-only. Add macOS support by detecting the OS and using the appropriate command:
- Write `get_cpu_usage()` that uses `/proc/stat` on Linux and `top -l 1` (parsed with `awk`) on macOS
- Write `get_memory_info()` that reads `/proc/meminfo` on Linux and `vm_stat` on macOS
- Write `get_disk_info()` that uses `df -h` on both but handles the different column layouts
- Test each function on your system and verify it returns a numeric percentage

Wrap each function with an OS check using `uname -s`:
```bash
OS=$(uname -s)
case "$OS" in
    Linux)  ... ;;
    Darwin) ... ;;
    *) echo "Unsupported OS: $OS" >&2; return 1 ;;
esac
```

### Exercise 2: Build a Threshold Alert System with Cooldown

Implement a standalone `alerter.sh` that:
- Accepts `CPU_THRESHOLD`, `MEM_THRESHOLD`, and `DISK_THRESHOLD` environment variables (defaults: 80, 85, 90)
- Checks current CPU, memory, and disk percentages by calling the metric functions from Exercise 1
- For each threshold exceeded, prints an alert message like `ALERT: CPU at 92% (threshold: 80%)`
- Implements a cooldown: once an alert fires, stores the timestamp in `/tmp/alert_cooldown_<metric>` and skips firing again until 5 minutes have elapsed
- Logs all alerts (including skipped cooldown alerts) to `/var/log/monitor_alerts.log` with a timestamp

Test by temporarily lowering a threshold below your current usage to trigger an alert, then running again within 5 minutes to confirm the cooldown suppresses the second alert.

### Exercise 3: Add Terminal Resize Handling

Extend the dashboard to respond to terminal resize events:
- Install a `trap 'redraw_dashboard' SIGWINCH` handler
- The `redraw_dashboard` function should read the new terminal dimensions with `tput lines` and `tput cols`
- Clear and redraw the entire dashboard using the new dimensions
- If the terminal is too small (fewer than 24 lines or 80 columns), display a single message: `"Terminal too small. Resize to at least 80x24."`
- Use `tput cup 0 0` (move cursor to top-left) rather than `clear` to avoid screen flicker on redraw

Test by running the dashboard and resizing your terminal window — the layout should adapt without losing history or flickering.

### Exercise 4: Generate an Automated HTML Report

Write a `generate_report.sh` that produces a self-contained HTML report of system health:
- Collect one snapshot of CPU, memory, disk, and load average
- List the top 5 processes by CPU usage (from `ps aux --sort=-%cpu`)
- Check the last 50 lines of `/var/log/syslog` (or `/var/log/system.log` on macOS) for lines matching `ERROR|WARN|CRIT`
- Write all findings into a styled HTML file with a table for metrics, a table for top processes, and a `<pre>` block for log excerpts
- Include the generation timestamp and hostname in the report header

Set up a cron job that runs this script every day at 7 AM and saves the output to `/var/reports/$(date +%Y%m%d)_report.html`.

### Exercise 5: Write Integration Tests for the Monitor

Write a Bats test suite `test_monitor.bats` that tests the monitoring tool's key behaviors without actually requiring elevated privileges or specific hardware:
- Test that `get_cpu_usage` returns a value between 0 and 100
- Test that `get_memory_info` returns three space-separated values (used, total, percent)
- Test that `get_disk_info` returns at least one line of output
- Test that the alert system fires when a threshold of 0 is set (guaranteed to trigger)
- Mock the metric functions to return fixed values (e.g., CPU=95) and verify the alert message content matches the expected format

---

**Previous**: [Project: Deployment Automation](./15_Project_Deployment.md)
