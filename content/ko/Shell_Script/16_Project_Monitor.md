# 프로젝트: 시스템 모니터링 도구(System Monitoring Tool)

**난이도**: ⭐⭐⭐⭐

**이전**: [배포 자동화](./15_Project_Deployment.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `tput`을 사용하여 커서 제어(cursor control), 컬러 포맷팅, 반응형 레이아웃을 갖춘 실시간 터미널 대시보드(terminal dashboard)를 구축할 수 있습니다.
2. `/proc`과 표준 Unix 유틸리티에서 CPU, 메모리, 디스크, 네트워크 데이터를 파싱하는 시스템 메트릭 수집기(metric collector)를 구현할 수 있습니다.
3. 알림 스팸을 방지하는 쿨다운 기간(cooldown period)이 포함된 임계값 기반 경고(threshold-based alerting) 시스템을 설정할 수 있습니다.
4. 셸 스크립트에서 Slack 또는 Discord 웹훅(webhook)과 이메일로 경고 알림을 전송할 수 있습니다.
5. 시스템 로그를 파싱하고 오류 패턴을 감지하며 결과를 요약하는 로그 집계(log aggregation) 모듈을 작성할 수 있습니다.
6. 주기적인 헬스 체크(health check)와 자동화된 HTML 보고서 생성을 위해 모니터링 스크립트를 cron과 연동할 수 있습니다.
7. 터미널 크기 변경 이벤트(`SIGWINCH`)를 처리하여 대시보드 레이아웃을 동적으로 조정할 수 있습니다.

---

프로덕션 서버가 새벽 3시에 디스크 공간이 부족해지거나 트래픽 급증으로 CPU 사용률이 폭등할 때, 문제가 발생하기 전에 모니터링이 갖춰져 있어야 합니다. Datadog이나 Prometheus 같은 상용 도구는 강력하지만 자체 인프라가 필요합니다. bash 기반 모니터링 도구는 의존성 없이 어떤 서버에서도 실행되고, 수집할 메트릭과 경고 방식을 완전히 제어할 수 있으며, 이전 모든 레슨의 터미널 UI, 시그널 처리(signal handling), 프로세스 관리(process management), 시스템 내부 구조를 통합하는 마무리 프로젝트로 작동합니다.

## 1. 개요

### 우리가 만들 것

순수 bash로 작성된 포괄적인 터미널 기반 시스템 모니터링 대시보드입니다. 이 도구는 다음에 대한 실시간 가시성을 제공합니다:

- CPU 사용량과 부하 평균
- 메모리와 스왑 활용도
- 파일 시스템 전체의 디스크 공간
- 네트워크 I/O 통계
- 활성 프로세스와 리소스 소비
- 에러 탐지를 통한 시스템 로그
- 임계값 초과 시 경고 알림

`htop`, `glances`, Datadog 에이전트와 같은 도구의 경량 커스터마이징 가능한 대안으로 생각하되, bash로 완전히 자체 포함됩니다.

### 주요 기능

1. **실시간 대시보드**: 컬러와 레이아웃이 있는 실시간 업데이트 터미널 UI
2. **설정 가능한 경고**: Slack/Discord 웹훅으로 전송되는 임계값 기반 경고
3. **로그 집계**: 에러/경고에 대한 시스템 로그 파싱 및 요약
4. **Cron 통합**: 주기적 확인 실행 및 보고서 생성
5. **크로스 플랫폼**: Linux와 macOS에서 작동 (OS 특화 적응 포함)
6. **제로 의존성**: 표준 유닉스 도구를 사용하는 순수 bash

### 왜 순수 Bash인가?

- **보편성**: bash가 설치된 모든 서버에서 실행
- **투명성**: 블랙박스 모니터링 에이전트 없음
- **커스터마이징**: 메트릭, 임계값, 출력 형식을 쉽게 수정
- **학습**: 고급 bash 기술 시연 (tput, /proc 파싱, 시그널 처리)

---

## 2. 설계

### 아키텍처

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

### 데이터 흐름

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

### 설정

임계값과 설정은 설정 파일에 저장:

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

## 3. tput을 사용한 터미널 UI

### tput이란?

`tput`은 terminfo 데이터베이스를 사용하여 커서 위치, 색상, 화면 지우기를 조작하는 터미널 제어 유틸리티입니다.

### 필수 tput 명령어

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

### 색상 참조

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

### 대시보드 레이아웃 구축

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

### 진행 막대

백분율의 시각적 표현:

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

### 터미널 크기 조정 처리 (SIGWINCH)

터미널 크기 변경 시 다시 그리기:

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

## 4. 시스템 메트릭 수집

### CPU 사용량

`/proc/stat`에서 CPU 사용량 계산:

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

**macOS 대안** (`top` 사용):

```bash
get_cpu_usage_macos() {
    top -l 2 -n 0 -F -s 1 | \
        awk '/^CPU usage:/ {print 100 - $7}' | \
        tail -1 | \
        cut -d. -f1
}
```

### 부하 평균

```bash
get_load_average() {
    awk '{print $1, $2, $3}' /proc/loadavg
}

# macOS compatible version
get_load_average_cross() {
    uptime | awk -F'load averages?: ' '{print $2}'
}
```

### 메모리 사용량

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

### 디스크 사용량

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

### 네트워크 I/O

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

### 프로세스 메트릭

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

## 5. 경고 시스템

### 임계값 설정

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

### 경고 확인 함수

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

### 경고 쿨다운

쿨다운 기간을 사용하여 경고 스팸 방지:

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

### 웹훅 알림 (Slack/Discord)

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

## 6. 로그 집계

### 시스템 로그 파싱

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

### 애플리케이션 로그 파싱

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

## 7. 완전한 모니터 스크립트

완전한 모니터링 대시보드 구현입니다:

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

## 8. Cron 통합

### 주기적 모니터링

5분마다 헬스 체크 실행:

```bash
# Crontab entry
*/5 * * * * /opt/monitor/monitor.sh --check-only >> /var/log/monitor_cron.log 2>&1
```

`--check-only` 모드가 있는 모니터 스크립트:

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

### 보고서 생성

일일 HTML 보고서:

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

### 기록 데이터를 위한 CSV 내보내기

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

## 9. 사용 예시

### 대시보드 실행

```bash
chmod +x monitor.sh
./monitor.sh
```

### 경고 설정

`monitor.conf` 편집:

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

### 주기적 확인을 위한 Cron 설정

```bash
# Install cron job
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/monitor/monitor.sh --check-only") | crontab -

# Daily HTML report at 8 AM
(crontab -l 2>/dev/null; echo "0 8 * * * /opt/monitor/generate_report.sh") | crontab -
```

---

## 10. 확장

### 1. 원격 모니터링

SSH를 통해 여러 서버 모니터링:

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

### 2. 웹 대시보드

HTTP를 통해 실시간 메트릭 제공:

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

### 3. 데이터베이스 저장

기록 분석을 위해 SQLite에 메트릭 저장:

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

### 4. Grafana 호환 메트릭

Prometheus 형식으로 메트릭 내보내기:

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

## 연습 문제

### 연습 1: 크로스 플랫폼(Cross-Platform) 메트릭 수집기 구현하기

모니터링 도구는 Linux 전용인 `/proc` 파일을 사용합니다. OS를 감지하고 적절한 명령어를 사용하여 macOS 지원을 추가하세요:
- Linux에서는 `/proc/stat`을, macOS에서는 `top -l 1`(awk로 파싱)을 사용하는 `get_cpu_usage()` 작성
- Linux에서는 `/proc/meminfo`를, macOS에서는 `vm_stat`을 읽는 `get_memory_info()` 작성
- 두 OS에서 `df -h`를 사용하되 다른 열 레이아웃을 처리하는 `get_disk_info()` 작성
- 각 함수를 시스템에서 테스트하고 숫자 백분율을 반환하는지 확인

`uname -s`를 사용하여 각 함수에 OS 확인을 추가하세요:
```bash
OS=$(uname -s)
case "$OS" in
    Linux)  ... ;;
    Darwin) ... ;;
    *) echo "Unsupported OS: $OS" >&2; return 1 ;;
esac
```

### 연습 2: 쿨다운(Cooldown)이 있는 임계값 알림 시스템 구축하기

다음을 수행하는 독립 실행형 `alerter.sh`를 구현하세요:
- `CPU_THRESHOLD`, `MEM_THRESHOLD`, `DISK_THRESHOLD` 환경 변수 허용 (기본값: 80, 85, 90)
- 연습 1의 메트릭 함수를 호출하여 현재 CPU, 메모리, 디스크 백분율 확인
- 임계값이 초과된 각 항목에 대해 `ALERT: CPU at 92% (threshold: 80%)` 같은 알림 메시지 출력
- 쿨다운(cooldown) 구현: 알림이 발동되면 타임스탬프를 `/tmp/alert_cooldown_<metric>`에 저장하고 5분이 경과할 때까지 다시 발동하지 않음
- 모든 알림(쿨다운으로 건너뛴 알림 포함)을 타임스탬프와 함께 `/var/log/monitor_alerts.log`에 기록

임계값을 현재 사용률보다 낮게 일시적으로 낮춰 알림을 트리거하여 테스트하고, 5분 이내에 다시 실행하여 쿨다운이 두 번째 알림을 억제하는지 확인하세요.

### 연습 3: 터미널 크기 조정 처리 추가하기

터미널 크기 조정 이벤트에 응답하도록 대시보드를 확장하세요:
- `trap 'redraw_dashboard' SIGWINCH` 핸들러 설치
- `redraw_dashboard` 함수가 `tput lines`와 `tput cols`로 새 터미널 크기를 읽어야 함
- 새 크기를 사용하여 전체 대시보드를 지우고 다시 그리기
- 터미널이 너무 작으면(24줄 또는 80열 미만), 단일 메시지 표시: `"Terminal too small. Resize to at least 80x24."`
- 화면 깜박임을 피하기 위해 `clear` 대신 `tput cup 0 0`(커서를 좌상단으로 이동) 사용

대시보드를 실행하고 터미널 창 크기를 조정하여 테스트 — 히스토리 손실이나 깜박임 없이 레이아웃이 적응되어야 합니다.

### 연습 4: 자동화된 HTML 보고서 생성하기

시스템 상태의 자가 포함(self-contained) HTML 보고서를 생성하는 `generate_report.sh`를 작성하세요:
- CPU, 메모리, 디스크, 로드 평균(load average)의 스냅샷 수집 한 번
- CPU 사용량 기준 상위 5개 프로세스 나열 (`ps aux --sort=-%cpu`에서)
- `ERROR|WARN|CRIT`와 일치하는 줄에 대해 `/var/log/syslog`(macOS는 `/var/log/system.log`)의 마지막 50줄 확인
- 메트릭용 테이블, 상위 프로세스용 테이블, 로그 발췌용 `<pre>` 블록이 있는 스타일이 적용된 HTML 파일에 모든 결과 작성
- 보고서 헤더에 생성 타임스탬프와 호스트명 포함

이 스크립트를 매일 오전 7시에 실행하고 출력을 `/var/reports/$(date +%Y%m%d)_report.html`에 저장하는 크론(cron) 작업을 설정하세요.

### 연습 5: 모니터를 위한 통합 테스트 작성하기

상승된 권한이나 특정 하드웨어를 실제로 요구하지 않고 모니터링 도구의 핵심 동작을 테스트하는 Bats 테스트 스위트(test suite) `test_monitor.bats`를 작성하세요:
- `get_cpu_usage`가 0~100 사이의 값을 반환하는지 테스트
- `get_memory_info`가 공백으로 구분된 세 개의 값(사용량, 총량, 백분율)을 반환하는지 테스트
- `get_disk_info`가 최소 한 줄의 출력을 반환하는지 테스트
- 임계값이 0으로 설정될 때(반드시 트리거됨) 알림 시스템이 발동하는지 테스트
- 메트릭 함수를 고정 값(예: CPU=95)을 반환하도록 모킹(mocking)하고 알림 메시지 내용이 예상 형식과 일치하는지 확인

---

**이전**: [15_Project_Deployment.md](./15_Project_Deployment.md)
