#!/bin/bash
# Exercises for Lesson 11: System Monitoring
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Resource Monitoring ===
# Problem: Monitor system resources using command-line tools to identify
#          memory pressure, CPU bottlenecks, and I/O saturation.
exercise_1() {
    echo "=== Exercise 1: Resource Monitoring ==="
    echo ""
    echo "Scenario: A production web server is responding slowly during peak hours."
    echo "Use monitoring tools to identify whether the bottleneck is CPU, memory, or I/O."
    echo ""

    echo "--- Part A: Memory Analysis with free and vmstat ---"
    echo "Solution:"
    echo "  free -h                                 # Human-readable memory summary"
    echo "  free -h -s 5                            # Refresh every 5 seconds"
    echo "  vmstat 2 10                             # Sample every 2 sec, 10 iterations"
    echo ""
    echo "  Key vmstat columns to watch:"
    echo "    r   - Processes waiting for CPU (runnable). High r = CPU bottleneck"
    echo "    b   - Processes in uninterruptible sleep. High b = I/O bottleneck"
    echo "    si  - Swap in (KB/s). Non-zero = memory pressure"
    echo "    so  - Swap out (KB/s). Non-zero = memory pressure"
    echo "    us  - User CPU time %. High = application CPU usage"
    echo "    sy  - System CPU time %. High = kernel/syscall overhead"
    echo "    wa  - Wait I/O %. High = disk I/O bottleneck"
    echo "    id  - Idle %. Low = system is busy"
    echo ""
    echo "  Explanation:"
    echo "    In 'free' output, 'available' is the true free memory (includes cache)"
    echo "    Linux aggressively caches disk data; high 'buff/cache' is normal and healthy"
    echo "    Swap usage alone is not alarming; active swapping (si/so > 0) is the concern"
    echo ""

    echo "--- Part B: I/O Monitoring with iostat ---"
    echo "Solution:"
    echo "  iostat -xz 2 5                          # Extended stats, skip idle, 2s interval"
    echo ""
    echo "  Key iostat columns:"
    echo "    %util   - Device utilization. >80% indicates saturation"
    echo "    await   - Average I/O wait time (ms). High = slow storage"
    echo "    r_await - Read wait time (ms)"
    echo "    w_await - Write wait time (ms)"
    echo "    avgqu-sz - Average queue length. High = requests queuing up"
    echo "    rkB/s, wkB/s - Read/write throughput"
    echo ""
    echo "  Explanation:"
    echo "    -x shows extended statistics with all columns"
    echo "    -z suppresses output for devices with zero activity"
    echo "    Compare await with typical values: SSD ~0.1-1ms, HDD ~5-15ms"
    echo "    High %util with low throughput indicates random I/O (seek-bound)"
    echo ""

    echo "--- Part C: Historical Monitoring with sar ---"
    echo "Solution:"
    echo "  sar -u 2 5                              # CPU usage: 2s interval, 5 samples"
    echo "  sar -r 2 5                              # Memory usage"
    echo "  sar -d 2 5                              # Disk I/O"
    echo "  sar -n DEV 2 5                          # Network interface statistics"
    echo ""
    echo "  # View historical data (from /var/log/sa/)"
    echo "  sar -u -f /var/log/sa/sa13              # CPU for the 13th of the month"
    echo "  sar -r -s 09:00:00 -e 17:00:00          # Memory between 9am and 5pm today"
    echo ""
    echo "  Explanation:"
    echo "    sar is part of the sysstat package (install: apt install sysstat)"
    echo "    Data is collected every 10 minutes by a cron job (/etc/cron.d/sysstat)"
    echo "    Enable data collection: edit /etc/default/sysstat, set ENABLED=\"true\""
    echo "    Historical data is invaluable for correlating performance with events"
    echo ""

    # Safe read-only check
    echo "--- Current Memory Status ---"
    if command -v free &>/dev/null; then
        free -h 2>/dev/null || echo "  (free command not available)"
    else
        echo "  (free command not available on this system)"
    fi
}

# === Exercise 2: Cron Job Scheduling ===
# Problem: Set up automated tasks using crontab with proper scheduling,
#          logging, and error handling.
exercise_2() {
    echo "=== Exercise 2: Cron Job Scheduling ==="
    echo ""
    echo "Scenario: Configure automated maintenance tasks including log rotation,"
    echo "database backups, and health checks on a production server."
    echo ""

    echo "--- Part A: Crontab Syntax and Common Patterns ---"
    echo "Solution:"
    cat << 'CRONTAB'
  # Crontab format:
  # MIN  HOUR  DOM  MON  DOW  COMMAND
  # 0-59 0-23  1-31 1-12 0-7  (0 and 7 = Sunday)

  # Every 5 minutes: health check
  */5 * * * * /opt/scripts/health_check.sh >> /var/log/health.log 2>&1

  # Daily at 2:30 AM: database backup
  30 2 * * * /opt/scripts/db_backup.sh >> /var/log/backup.log 2>&1

  # Every Monday at 6 AM: weekly report
  0 6 * * 1 /opt/scripts/weekly_report.sh 2>&1 | mail -s "Weekly Report" admin@company.com

  # First day of each month at midnight: log rotation
  0 0 1 * * /opt/scripts/rotate_logs.sh >> /var/log/rotation.log 2>&1

  # Every weekday at 8 AM and 6 PM: system stats
  0 8,18 * * 1-5 /opt/scripts/system_stats.sh >> /var/log/stats.log 2>&1
CRONTAB
    echo ""
    echo "  Explanation:"
    echo "    */5 means 'every 5 units' (works for any field)"
    echo "    1-5 means Monday through Friday"
    echo "    8,18 means 'at 8 and at 18'"
    echo "    2>&1 redirects stderr to stdout (capture errors in log)"
    echo "    Always use full paths in cron (no PATH environment by default)"
    echo ""

    echo "--- Part B: Crontab Management Commands ---"
    echo "Solution:"
    echo "  crontab -e                   # Edit current user's crontab"
    echo "  crontab -l                   # List current user's crontab"
    echo "  crontab -r                   # Remove all cron jobs (dangerous!)"
    echo "  sudo crontab -e -u www-data  # Edit crontab for a specific user"
    echo ""
    echo "  # System-wide cron directories (drop scripts here, no crontab -e needed):"
    echo "  /etc/cron.d/          # Custom schedule files (crontab syntax + username field)"
    echo "  /etc/cron.hourly/     # Scripts run every hour"
    echo "  /etc/cron.daily/      # Scripts run daily (usually 6:25 AM)"
    echo "  /etc/cron.weekly/     # Scripts run weekly"
    echo "  /etc/cron.monthly/    # Scripts run monthly"
    echo ""
    echo "  Explanation:"
    echo "    Scripts in cron.daily/ etc. are run by anacron (handles missed runs)"
    echo "    Files in /etc/cron.d/ include a username field after DOW"
    echo "    Check /var/log/syslog or /var/log/cron for cron execution logs"
    echo ""

    echo "--- Part C: Robust Cron Job Script Pattern ---"
    echo "Solution:"
    cat << 'SCRIPT'
  #!/bin/bash
  # /opt/scripts/db_backup.sh - Production database backup

  LOCK_FILE="/var/run/db_backup.lock"
  LOG_FILE="/var/log/db_backup.log"

  # Prevent concurrent runs with flock
  exec 200>"$LOCK_FILE"
  if ! flock -n 200; then
      echo "$(date): Backup already running, exiting" >> "$LOG_FILE"
      exit 0
  fi

  # Set PATH explicitly (cron has minimal PATH)
  export PATH="/usr/local/bin:/usr/bin:/bin"

  # Logging function
  log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"; }

  log "Backup started"
  if pg_dump -U postgres mydb | gzip > "/backup/mydb_$(date +%Y%m%d).sql.gz"; then
      log "Backup completed successfully"
  else
      log "ERROR: Backup failed with exit code $?"
      echo "Database backup failed" | mail -s "ALERT: Backup Failure" admin@company.com
  fi
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    flock prevents overlapping runs if a job takes longer than the interval"
    echo "    Explicit PATH ensures commands are found (cron's PATH is minimal)"
    echo "    Error notification via mail ensures failures are not silently ignored"
    echo ""

    echo "--- Verification ---"
    echo "  grep CRON /var/log/syslog               # Check cron execution on Debian/Ubuntu"
    echo "  journalctl -u cron --since '1 hour ago'  # Check with systemd"
    echo "  systemctl status cron                     # Verify cron daemon is running"
}

# === Exercise 3: Log Monitoring and Alerting ===
# Problem: Set up real-time log monitoring and simple alerting mechanisms.
exercise_3() {
    echo "=== Exercise 3: Log Monitoring and Alerting ==="
    echo ""
    echo "Scenario: Monitor application and system logs in real time, set up"
    echo "alerts for critical events, and aggregate information from multiple logs."
    echo ""

    echo "--- Part A: Real-Time Log Watching ---"
    echo "Solution:"
    echo "  # Follow a log file in real time"
    echo "  tail -f /var/log/syslog"
    echo ""
    echo "  # Follow multiple files simultaneously"
    echo "  tail -f /var/log/syslog /var/log/auth.log"
    echo ""
    echo "  # Follow with grep filter (only show errors)"
    echo "  tail -f /var/log/syslog | grep --line-buffered -i 'error\\|fail\\|critical'"
    echo ""
    echo "  # Use journalctl for systemd-managed logs"
    echo "  journalctl -f -u nginx                   # Follow nginx logs"
    echo "  journalctl -f -p err                      # Follow only error-priority and above"
    echo "  journalctl --since '10 minutes ago' -u ssh # Recent SSH activity"
    echo ""
    echo "  Explanation:"
    echo "    tail -f keeps the file open and outputs new lines as they appear"
    echo "    --line-buffered is critical with grep to prevent output buffering delays"
    echo "    journalctl -p levels: emerg, alert, crit, err, warning, notice, info, debug"
    echo "    journalctl -f is preferred over tail -f for systemd services"
    echo ""

    echo "--- Part B: Periodic Monitoring with watch ---"
    echo "Solution:"
    echo "  # Watch disk space every 5 seconds"
    echo "  watch -n 5 df -h"
    echo ""
    echo "  # Watch connection count by state"
    echo "  watch -n 2 'ss -tan | awk '\\''NR>1{print \$1}'\\'' | sort | uniq -c | sort -rn'"
    echo ""
    echo "  # Highlight changes between refreshes"
    echo "  watch -d -n 10 'cat /proc/meminfo | head -5'"
    echo ""
    echo "  Explanation:"
    echo "    -n sets the refresh interval in seconds (default: 2)"
    echo "    -d highlights differences between consecutive outputs"
    echo "    watch re-runs the command and refreshes the terminal display"
    echo "    Press Ctrl+C to stop; useful for live dashboards in a tmux pane"
    echo ""

    echo "--- Part C: Simple Alert Script ---"
    echo "Solution:"
    cat << 'SCRIPT'
  #!/bin/bash
  # /opt/scripts/disk_alert.sh - Alert when disk usage exceeds threshold
  # Run via cron: */10 * * * * /opt/scripts/disk_alert.sh

  THRESHOLD=85
  ALERT_EMAIL="admin@company.com"
  HOSTNAME=$(hostname)

  # Parse df output, skip header and tmpfs
  df -h --output=pcent,target | tail -n +2 | while read -r usage mount; do
      # Strip % sign and compare
      pct=${usage%\%}
      if (( pct > THRESHOLD )); then
          SUBJECT="DISK ALERT: ${mount} at ${usage} on ${HOSTNAME}"
          MESSAGE="Warning: ${mount} is ${usage} full on ${HOSTNAME} at $(date)"
          echo "$MESSAGE" | mail -s "$SUBJECT" "$ALERT_EMAIL"
          logger -t disk_alert "$MESSAGE"    # Also log to syslog
      fi
  done
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    df --output=pcent,target gives just the percentage and mount point"
    echo "    \${usage%\\%} strips the trailing % sign for numeric comparison"
    echo "    logger writes to syslog (viewable with journalctl -t disk_alert)"
    echo "    mail sends email alerts (requires mailutils or postfix configured)"
    echo ""

    echo "--- Verification ---"
    echo "  # Test the alert script manually:"
    echo "  bash -x /opt/scripts/disk_alert.sh       # Run with debug trace"
    echo "  logger -t test 'Hello syslog'             # Test syslog writing"
    echo "  journalctl -t test --since '1 minute ago' # Verify syslog entry"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
