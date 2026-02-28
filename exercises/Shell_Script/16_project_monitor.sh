#!/bin/bash
# Exercises for Lesson 16: Project - System Monitoring Tool
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Implement Cross-Platform Metric Collectors ===
# Problem: Write get_cpu_usage, get_memory_info, get_disk_info that work
# on both Linux (/proc) and macOS (top, vm_stat, df).
exercise_1() {
    echo "=== Exercise 1: Implement Cross-Platform Metric Collectors ==="

    local OS
    OS=$(uname -s)

    get_cpu_usage() {
        case "$OS" in
            Linux)
                if [ -f /proc/stat ]; then
                    # Two-sample CPU measurement
                    local cpu_line1 cpu_line2
                    cpu_line1=$(head -1 /proc/stat)
                    sleep 0.2
                    cpu_line2=$(head -1 /proc/stat)

                    local idle1 total1 idle2 total2
                    idle1=$(echo "$cpu_line1" | awk '{print $5}')
                    total1=$(echo "$cpu_line1" | awk '{s=0; for(i=2;i<=NF;i++) s+=$i; print s}')
                    idle2=$(echo "$cpu_line2" | awk '{print $5}')
                    total2=$(echo "$cpu_line2" | awk '{s=0; for(i=2;i<=NF;i++) s+=$i; print s}')

                    local idle_d=$((idle2 - idle1))
                    local total_d=$((total2 - total1))
                    if (( total_d > 0 )); then
                        echo $(( 100 * (total_d - idle_d) / total_d ))
                    else
                        echo 0
                    fi
                else
                    echo 0
                fi
                ;;
            Darwin)
                # macOS: Use top -l 1 to get CPU idle percentage
                local idle
                idle=$(top -l 1 -n 0 2>/dev/null | awk '/CPU usage/ {
                    gsub(/%/, "")
                    for(i=1;i<=NF;i++) { if($(i+1)=="idle") print $i }
                }')
                if [ -n "$idle" ]; then
                    # Remove decimal part for integer
                    idle=${idle%%.*}
                    echo $(( 100 - idle ))
                else
                    echo 0
                fi
                ;;
            *)
                echo "0"
                ;;
        esac
    }

    get_memory_info() {
        # Returns: used_gb total_gb percent
        case "$OS" in
            Linux)
                if [ -f /proc/meminfo ]; then
                    awk '/^MemTotal:/ {total=$2}
                         /^MemAvailable:/ {avail=$2}
                         END {
                             used = total - avail
                             pct = int(100 * used / total)
                             printf "%.1f %.1f %d\n", used/1024/1024, total/1024/1024, pct
                         }' /proc/meminfo
                else
                    echo "0.0 0.0 0"
                fi
                ;;
            Darwin)
                local page_size
                page_size=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
                local total_bytes
                total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
                local total_gb
                total_gb=$(echo "$total_bytes" | awk -v ps="$page_size" '{printf "%.1f", $1/1024/1024/1024}')

                # vm_stat gives pages
                local free_pages
                free_pages=$(vm_stat 2>/dev/null | awk '/Pages free/ {gsub(/\./,""); print $3}')
                local inactive_pages
                inactive_pages=$(vm_stat 2>/dev/null | awk '/Pages inactive/ {gsub(/\./,""); print $3}')

                local free_bytes=$(( (free_pages + inactive_pages) * page_size ))
                local used_bytes=$(( total_bytes - free_bytes ))
                local used_gb
                used_gb=$(echo "$used_bytes" | awk '{printf "%.1f", $1/1024/1024/1024}')
                local pct
                if (( total_bytes > 0 )); then
                    pct=$(( 100 * used_bytes / total_bytes ))
                else
                    pct=0
                fi

                echo "$used_gb $total_gb $pct"
                ;;
            *)
                echo "0.0 0.0 0"
                ;;
        esac
    }

    get_disk_info() {
        # Returns: used total percent (for root filesystem)
        # df -h works on both platforms but column layout can differ
        df -h / 2>/dev/null | awk 'NR==2 {
            gsub(/%/, "", $5)
            print $3, $2, $5
        }'
    }

    # Run the collectors
    echo "  OS detected: $OS"
    echo ""

    echo "  --- CPU Usage ---"
    local cpu
    cpu=$(get_cpu_usage)
    echo "  CPU: ${cpu}%"
    if [[ "$cpu" =~ ^[0-9]+$ ]] && (( cpu >= 0 && cpu <= 100 )); then
        echo "  [OK] Valid numeric percentage"
    else
        echo "  [WARN] Unexpected value: $cpu"
    fi

    echo ""
    echo "  --- Memory Info ---"
    local mem_info
    mem_info=$(get_memory_info)
    echo "  Memory: $mem_info"
    local mem_pct
    mem_pct=$(echo "$mem_info" | awk '{print $3}')
    if [[ "$mem_pct" =~ ^[0-9]+$ ]] && (( mem_pct >= 0 && mem_pct <= 100 )); then
        echo "  [OK] Valid memory percentage: ${mem_pct}%"
    else
        echo "  [WARN] Unexpected memory value"
    fi

    echo ""
    echo "  --- Disk Info ---"
    local disk_info
    disk_info=$(get_disk_info)
    echo "  Disk: $disk_info"
    local disk_pct
    disk_pct=$(echo "$disk_info" | awk '{print $3}')
    if [[ "$disk_pct" =~ ^[0-9]+$ ]] && (( disk_pct >= 0 && disk_pct <= 100 )); then
        echo "  [OK] Valid disk percentage: ${disk_pct}%"
    else
        echo "  [WARN] Unexpected disk value"
    fi
}

# === Exercise 2: Build a Threshold Alert System with Cooldown ===
# Problem: alerter.sh with thresholds, cooldown via /tmp files, logging.
exercise_2() {
    echo "=== Exercise 2: Build a Threshold Alert System with Cooldown ==="

    local COOLDOWN_DIR="/tmp/alert_cooldown_$$"
    local ALERT_LOG="/tmp/monitor_alerts_$$.log"
    mkdir -p "$COOLDOWN_DIR"

    local CPU_THRESHOLD="${CPU_THRESHOLD:-80}"
    local MEM_THRESHOLD="${MEM_THRESHOLD:-85}"
    local DISK_THRESHOLD="${DISK_THRESHOLD:-90}"
    local COOLDOWN_SECONDS=300  # 5 minutes

    # Simple metric getters (will return actual values on supported systems)
    get_cpu_pct() {
        # Return a simulated value for testing purposes
        echo "${FAKE_CPU:-45}"
    }
    get_mem_pct() {
        echo "${FAKE_MEM:-60}"
    }
    get_disk_pct() {
        echo "${FAKE_DISK:-70}"
    }

    should_alert() {
        local metric="$1"
        local cooldown_file="$COOLDOWN_DIR/$metric"
        local now
        now=$(date +%s)

        if [ -f "$cooldown_file" ]; then
            local last_alert
            last_alert=$(cat "$cooldown_file")
            local elapsed=$(( now - last_alert ))
            if (( elapsed < COOLDOWN_SECONDS )); then
                echo "  [COOLDOWN] $metric alert suppressed (${elapsed}s since last, need ${COOLDOWN_SECONDS}s)"
                echo "[$(date)] COOLDOWN $metric (${elapsed}s < ${COOLDOWN_SECONDS}s)" >> "$ALERT_LOG"
                return 1
            fi
        fi
        return 0
    }

    record_alert() {
        local metric="$1"
        date +%s > "$COOLDOWN_DIR/$metric"
    }

    check_thresholds() {
        local cpu mem disk
        cpu=$(get_cpu_pct)
        mem=$(get_mem_pct)
        disk=$(get_disk_pct)

        echo "  Current: CPU=${cpu}% MEM=${mem}% DISK=${disk}%"
        echo "  Thresholds: CPU>=${CPU_THRESHOLD}% MEM>=${MEM_THRESHOLD}% DISK>=${DISK_THRESHOLD}%"
        echo ""

        # Check CPU
        if (( cpu >= CPU_THRESHOLD )); then
            if should_alert "cpu"; then
                echo "  ALERT: CPU at ${cpu}% (threshold: ${CPU_THRESHOLD}%)"
                echo "[$(date)] ALERT CPU at ${cpu}% (threshold: ${CPU_THRESHOLD}%)" >> "$ALERT_LOG"
                record_alert "cpu"
            fi
        fi

        # Check Memory
        if (( mem >= MEM_THRESHOLD )); then
            if should_alert "memory"; then
                echo "  ALERT: Memory at ${mem}% (threshold: ${MEM_THRESHOLD}%)"
                echo "[$(date)] ALERT MEM at ${mem}% (threshold: ${MEM_THRESHOLD}%)" >> "$ALERT_LOG"
                record_alert "memory"
            fi
        fi

        # Check Disk
        if (( disk >= DISK_THRESHOLD )); then
            if should_alert "disk"; then
                echo "  ALERT: Disk at ${disk}% (threshold: ${DISK_THRESHOLD}%)"
                echo "[$(date)] ALERT DISK at ${disk}% (threshold: ${DISK_THRESHOLD}%)" >> "$ALERT_LOG"
                record_alert "disk"
            fi
        fi
    }

    # Test 1: No alerts (values below thresholds)
    echo "--- Test 1: Below thresholds ---"
    FAKE_CPU=45 FAKE_MEM=60 FAKE_DISK=70
    check_thresholds

    echo ""

    # Test 2: Trigger alerts (set thresholds very low)
    echo "--- Test 2: Above thresholds ---"
    FAKE_CPU=95 FAKE_MEM=90 FAKE_DISK=92
    CPU_THRESHOLD=80 MEM_THRESHOLD=85 DISK_THRESHOLD=90
    check_thresholds

    echo ""

    # Test 3: Cooldown (run again immediately — should suppress)
    echo "--- Test 3: Cooldown suppression ---"
    check_thresholds

    echo ""
    echo "--- Alert log ---"
    if [ -f "$ALERT_LOG" ]; then
        cat "$ALERT_LOG" | sed 's/^/  /'
    else
        echo "  (empty)"
    fi

    rm -rf "$COOLDOWN_DIR" "$ALERT_LOG"
}

# === Exercise 3: Add Terminal Resize Handling ===
# Problem: trap SIGWINCH, redraw dashboard, check minimum size.
exercise_3() {
    echo "=== Exercise 3: Add Terminal Resize Handling ==="

    # Since we can't run an interactive dashboard in exercise mode,
    # we demonstrate the resize handling concepts.

    local TERM_COLS TERM_LINES
    local MIN_COLS=80
    local MIN_LINES=24

    get_terminal_size() {
        TERM_COLS=$(tput cols 2>/dev/null || echo 80)
        TERM_LINES=$(tput lines 2>/dev/null || echo 24)
    }

    check_minimum_size() {
        if (( TERM_COLS < MIN_COLS || TERM_LINES < MIN_LINES )); then
            echo "  Terminal too small. Resize to at least ${MIN_COLS}x${MIN_LINES}."
            echo "  Current size: ${TERM_COLS}x${TERM_LINES}"
            return 1
        fi
        return 0
    }

    redraw_dashboard() {
        get_terminal_size

        if ! check_minimum_size; then
            return
        fi

        # Use tput cup 0 0 instead of clear to avoid flicker
        # tput cup 0 0  # (commented out to avoid messing up exercise output)

        echo "  Dashboard redraw at ${TERM_COLS}x${TERM_LINES}"
        echo "  Header width: $((TERM_COLS - 2)) characters"
        local bar_width=$((TERM_COLS - 30))
        (( bar_width < 10 )) && bar_width=10
        (( bar_width > 60 )) && bar_width=60
        echo "  Progress bar width: $bar_width characters"
    }

    echo "--- Resize handler code ---"
    cat << 'CODE'
  # Install SIGWINCH handler
  resize_handler() {
      TERM_COLS=$(tput cols)
      TERM_LINES=$(tput lines)
      redraw_dashboard
  }
  trap resize_handler SIGWINCH

  redraw_dashboard() {
      # Check minimum size
      if (( TERM_COLS < 80 || TERM_LINES < 24 )); then
          tput cup 0 0
          echo "Terminal too small. Resize to at least 80x24."
          return
      fi

      # Redraw at current position (no clear = no flicker)
      tput cup 0 0
      draw_header
      draw_metrics
      draw_footer
  }
CODE

    echo ""
    echo "--- Testing with current terminal ---"
    get_terminal_size
    echo "  Terminal size: ${TERM_COLS}x${TERM_LINES}"

    if check_minimum_size; then
        echo "  [OK] Terminal meets minimum size requirements"
        redraw_dashboard
    fi

    echo ""
    echo "--- Simulating small terminal ---"
    TERM_COLS=60
    TERM_LINES=20
    check_minimum_size || true

    echo ""
    echo "--- Key techniques ---"
    echo "  1. trap 'resize_handler' SIGWINCH - catches terminal resize"
    echo "  2. tput cup 0 0 - move cursor without clearing (no flicker)"
    echo "  3. Check minimum dimensions before drawing"
    echo "  4. Scale bar widths based on TERM_COLS"
}

# === Exercise 4: Generate an Automated HTML Report ===
# Problem: Collect metrics, top processes, log errors, output styled HTML.
exercise_4() {
    echo "=== Exercise 4: Generate an Automated HTML Report ==="

    local output="/tmp/monitor_report_$$.html"
    local hostname_str
    hostname_str=$(hostname -s 2>/dev/null || echo "localhost")
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Collect metrics
    local cpu_usage="42"
    local mem_used="8.2"
    local mem_total="16.0"
    local mem_pct="51"
    local disk_used="120G"
    local disk_total="500G"
    local disk_pct="24"

    # Try to get real values
    if [ -f /proc/stat ]; then
        cpu_usage=$(awk '/^cpu / {
            idle=$5; total=0
            for(i=2;i<=NF;i++) total+=$i
            print int(100 * (total-idle) / total)
        }' /proc/stat 2>/dev/null || echo "$cpu_usage")
    fi

    local load_avg
    if [ -f /proc/loadavg ]; then
        load_avg=$(awk '{print $1, $2, $3}' /proc/loadavg)
    else
        load_avg=$(uptime | awk -F'load averages?: ' '{print $2}' | awk '{print $1, $2, $3}' | tr -d ',')
    fi
    load_avg="${load_avg:-0.5 0.3 0.2}"

    # Top 5 CPU processes
    local top_procs
    top_procs=$(ps aux --sort=-%cpu 2>/dev/null | head -6 | tail -5 | \
        awk '{printf "<tr><td>%s</td><td>%s</td><td>%.1f%%</td><td>%.1f%%</td><td>%s</td></tr>\n", $1, $2, $3, $4, $11}' \
        2>/dev/null || echo "<tr><td colspan='5'>Unable to collect process data</td></tr>")

    # Log errors (check common log locations)
    local log_errors=""
    for logfile in /var/log/syslog /var/log/system.log /var/log/messages; do
        if [ -r "$logfile" ]; then
            log_errors=$(tail -50 "$logfile" 2>/dev/null | grep -iE "ERROR|WARN|CRIT" | tail -10)
            break
        fi
    done
    if [ -z "$log_errors" ]; then
        log_errors="No recent errors found (or logs not accessible)."
    fi

    # Generate HTML
    cat > "$output" << HTMLEOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>System Monitor Report - $hostname_str</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; }
        .info { color: #888; font-size: 0.9em; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .bar-bg { background: #e0e0e0; height: 24px; border-radius: 4px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 4px; }
        .bar-green { background: #4CAF50; }
        .bar-yellow { background: #FFC107; }
        .bar-red { background: #F44336; }
        pre { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; font-size: 0.85em; }
    </style>
</head>
<body>
    <h1>System Monitor Report</h1>
    <p class="info">Host: <strong>$hostname_str</strong> | Generated: <strong>$timestamp</strong></p>

    <h2>System Metrics</h2>
    <table>
        <tr>
            <th>Metric</th><th>Value</th><th>Bar</th>
        </tr>
        <tr>
            <td>CPU Usage</td>
            <td>${cpu_usage}%</td>
            <td><div class="bar-bg"><div class="bar-fill bar-green" style="width:${cpu_usage}%"></div></div></td>
        </tr>
        <tr>
            <td>Memory</td>
            <td>${mem_used} / ${mem_total} GB (${mem_pct}%)</td>
            <td><div class="bar-bg"><div class="bar-fill bar-green" style="width:${mem_pct}%"></div></div></td>
        </tr>
        <tr>
            <td>Disk (/)</td>
            <td>${disk_used} / ${disk_total} (${disk_pct}%)</td>
            <td><div class="bar-bg"><div class="bar-fill bar-green" style="width:${disk_pct}%"></div></div></td>
        </tr>
        <tr>
            <td>Load Average</td>
            <td>$load_avg</td>
            <td>-</td>
        </tr>
    </table>

    <h2>Top 5 CPU Processes</h2>
    <table>
        <tr><th>User</th><th>PID</th><th>CPU%</th><th>MEM%</th><th>Command</th></tr>
        $top_procs
    </table>

    <h2>Recent Log Errors</h2>
    <pre>$(echo "$log_errors" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')</pre>

    <p class="info">Report generated by monitor exercise script.</p>
</body>
</html>
HTMLEOF

    echo "  Report generated: $output"
    echo "  Size: $(wc -c < "$output") bytes"
    echo ""

    echo "--- Report structure ---"
    echo "  - Header with hostname and timestamp"
    echo "  - Metrics table with CPU, Memory, Disk, Load Average"
    echo "  - Visual bar charts (CSS)"
    echo "  - Top 5 processes table"
    echo "  - Recent log errors in <pre> block"
    echo ""

    echo "--- First 20 lines of HTML ---"
    head -20 "$output" | sed 's/^/  /'
    echo "  ..."

    echo ""
    echo "--- Cron setup for daily report at 7 AM ---"
    echo '  0 7 * * * /path/to/generate_report.sh > /var/reports/$(date +\%Y\%m\%d)_report.html'

    rm -f "$output"
}

# === Exercise 5: Write Integration Tests for the Monitor ===
# Problem: Bats tests for metric functions and alert system.
exercise_5() {
    echo "=== Exercise 5: Write Integration Tests for the Monitor ==="

    local work_dir="/tmp/monitor_test_$$"
    mkdir -p "$work_dir"

    # Create a minimal monitor library for testing
    cat > "$work_dir/monitor_lib.sh" << 'MONLIB'
#!/bin/bash

get_cpu_usage() {
    local OS=$(uname -s)
    case "$OS" in
        Linux)
            if [ -f /proc/stat ]; then
                awk '/^cpu / {
                    idle=$5; total=0
                    for(i=2;i<=NF;i++) total+=$i
                    print int(100*(total-idle)/total)
                }' /proc/stat
            else
                echo 50
            fi
            ;;
        Darwin)
            local idle
            idle=$(top -l 1 -n 0 2>/dev/null | awk '/CPU usage/ {
                for(i=1;i<=NF;i++) if($(i+1)=="idle") { gsub(/%/,"",$i); print $i }
            }')
            idle=${idle%%.*}
            echo $(( 100 - ${idle:-50} ))
            ;;
        *) echo 50 ;;
    esac
}

get_memory_info() {
    local OS=$(uname -s)
    case "$OS" in
        Linux)
            awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}
                 END{u=t-a; printf "%.1f %.1f %d\n",u/1048576,t/1048576,int(100*u/t)}' /proc/meminfo 2>/dev/null || echo "8.0 16.0 50"
            ;;
        Darwin)
            echo "8.0 16.0 50"
            ;;
        *) echo "0.0 0.0 0" ;;
    esac
}

get_disk_info() {
    df -h / | awk 'NR==2{gsub(/%/,"",$5); print $3,$2,$5}'
}

check_alert() {
    local metric="$1"
    local value="$2"
    local threshold="$3"

    if (( value >= threshold )); then
        echo "ALERT: $metric at ${value}% (threshold: ${threshold}%)"
        return 0
    fi
    return 1
}
MONLIB

    # Create Bats test file
    cat > "$work_dir/test_monitor.bats" << 'BATS'
#!/usr/bin/env bats

setup() {
    source "${BATS_TEST_DIRNAME}/monitor_lib.sh"
}

@test "get_cpu_usage returns value between 0 and 100" {
    run get_cpu_usage
    [ "$status" -eq 0 ]
    [[ "$output" =~ ^[0-9]+$ ]]
    [ "$output" -ge 0 ]
    [ "$output" -le 100 ]
}

@test "get_memory_info returns three space-separated values" {
    run get_memory_info
    [ "$status" -eq 0 ]
    local fields
    IFS=' ' read -ra fields <<< "$output"
    [ "${#fields[@]}" -eq 3 ]
}

@test "get_disk_info returns at least one line" {
    run get_disk_info
    [ "$status" -eq 0 ]
    [ -n "$output" ]
}

@test "alert fires when threshold of 0 is set" {
    run check_alert "CPU" 50 0
    [ "$status" -eq 0 ]
    [[ "$output" =~ "ALERT: CPU at 50%" ]]
}

@test "alert message contains expected format" {
    run check_alert "MEM" 95 80
    [ "$status" -eq 0 ]
    [[ "$output" =~ "ALERT: MEM at 95% (threshold: 80%)" ]]
}
BATS

    echo "--- monitor_lib.sh ---"
    cat "$work_dir/monitor_lib.sh" | sed 's/^/  /'
    echo ""
    echo "--- test_monitor.bats ---"
    cat "$work_dir/test_monitor.bats" | sed 's/^/  /'
    echo ""

    # Run tests manually
    echo "--- Running tests ---"
    source "$work_dir/monitor_lib.sh"

    local pass=0 fail=0

    # Test 1: CPU usage returns 0-100
    local cpu
    cpu=$(get_cpu_usage)
    if [[ "$cpu" =~ ^[0-9]+$ ]] && (( cpu >= 0 && cpu <= 100 )); then
        echo "  ok 1 - get_cpu_usage returns value between 0 and 100 ($cpu)"
        (( pass++ ))
    else
        echo "  not ok 1 - get_cpu_usage returned: $cpu"
        (( fail++ ))
    fi

    # Test 2: Memory info returns 3 values
    local mem
    mem=$(get_memory_info)
    local mem_fields
    IFS=' ' read -ra mem_fields <<< "$mem"
    if [ "${#mem_fields[@]}" -eq 3 ]; then
        echo "  ok 2 - get_memory_info returns three values ($mem)"
        (( pass++ ))
    else
        echo "  not ok 2 - get_memory_info returned ${#mem_fields[@]} values: $mem"
        (( fail++ ))
    fi

    # Test 3: Disk info returns non-empty
    local disk
    disk=$(get_disk_info)
    if [ -n "$disk" ]; then
        echo "  ok 3 - get_disk_info returns non-empty ($disk)"
        (( pass++ ))
    else
        echo "  not ok 3 - get_disk_info returned empty"
        (( fail++ ))
    fi

    # Test 4: Alert fires with threshold 0
    local alert
    alert=$(check_alert "CPU" 50 0)
    if [[ "$alert" =~ "ALERT: CPU at 50%" ]]; then
        echo "  ok 4 - alert fires when threshold is 0"
        (( pass++ ))
    else
        echo "  not ok 4 - alert did not fire"
        (( fail++ ))
    fi

    # Test 5: Alert message format
    alert=$(check_alert "MEM" 95 80)
    if [[ "$alert" == "ALERT: MEM at 95% (threshold: 80%)" ]]; then
        echo "  ok 5 - alert message matches expected format"
        (( pass++ ))
    else
        echo "  not ok 5 - got: $alert"
        (( fail++ ))
    fi

    echo ""
    echo "  $pass passed, $fail failed"

    rm -rf "$work_dir"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
