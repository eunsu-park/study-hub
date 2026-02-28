#!/bin/bash
# Exercises for Lesson 18: Log Management
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: journalctl Query ===
# Problem: Write commands to query logs with specific conditions.
exercise_1() {
    echo "=== Exercise 1: journalctl Query ==="
    echo ""

    echo "--- Part A: Only nginx service error logs (today) ---"
    echo "Solution:"
    echo "  journalctl -u nginx -p err --since today"
    echo ""
    echo "  Flags explained:"
    echo "    -u nginx       # Filter by systemd unit (nginx.service)"
    echo "    -p err         # Priority filter: err and above (err, crit, alert, emerg)"
    echo "    --since today  # Time range: from midnight today"
    echo ""
    echo "  Priority levels (0=highest): emerg > alert > crit > err > warning > notice > info > debug"
    echo ""

    echo "--- Part B: Output logs for PID 1234 in JSON ---"
    echo "Solution:"
    echo "  journalctl _PID=1234 -o json-pretty"
    echo ""
    echo "  Flags explained:"
    echo "    _PID=1234       # Match field: filter by process ID"
    echo "    -o json-pretty  # Output format: indented JSON"
    echo ""
    echo "  Other useful output formats:"
    echo "    -o short     # Default syslog-like format"
    echo "    -o verbose   # All journal fields"
    echo "    -o json      # Compact JSON (one entry per line, good for piping)"
    echo "    -o cat        # Message text only (no metadata)"
    echo ""

    echo "--- Part C: Kernel warning+ messages from the last hour ---"
    echo "Solution:"
    echo "  journalctl -k -p warning --since '1 hour ago'"
    echo ""
    echo "  Flags explained:"
    echo "    -k              # Kernel messages only (same as dmesg, but with timestamps)"
    echo "    -p warning      # Warning priority and above"
    echo "    --since '...'   # Accepts: 'today', 'yesterday', '2 hours ago', '2024-01-15 10:00'"
    echo ""

    # Safe demonstration: show a sample journalctl command if available
    if command -v journalctl &>/dev/null; then
        echo "--- Live demo: Recent kernel messages ---"
        journalctl -k -p warning --since "1 hour ago" --no-pager -n 5 2>/dev/null \
            || echo "(No kernel messages in the last hour or insufficient permissions)"
    else
        echo "(journalctl not available on this system - commands shown above for reference)"
    fi
}

# === Exercise 2: rsyslog Filter ===
# Problem: Write rsyslog rules for auth logging, failure filtering, and remote forwarding.
exercise_2() {
    echo "=== Exercise 2: rsyslog Filter ==="
    echo ""
    echo "Scenario: Configure rsyslog to separate auth logs, capture failures, and forward errors."
    echo ""

    echo "Solution: /etc/rsyslog.d/custom.conf"
    echo ""
    cat << 'RSYSLOG'
# === Custom rsyslog configuration ===
# File: /etc/rsyslog.d/custom.conf

# Rule 1: Save ALL auth facility messages to a dedicated file
# 'auth.*' matches any priority level from the auth facility
auth.*    /var/log/auth-all.log

# Rule 2: Property-based filter for messages containing "Failed"
# This catches failed login attempts, failed sudo, etc.
# ':msg, contains' performs substring matching on the message text
:msg, contains, "Failed"    /var/log/failures.log

# Rule 3: Forward error and above to remote syslog server
# '*.err' = all facilities, error priority and above
# '@@' = TCP transport (single '@' would be UDP)
# Port 514 is the standard syslog port
*.err    @@192.168.1.100:514
RSYSLOG
    echo ""
    echo "Key rsyslog concepts:"
    echo "  Facility: auth, kern, mail, daemon, local0-7, etc."
    echo "  Priority: emerg, alert, crit, err, warning, notice, info, debug"
    echo "  Selector: facility.priority (e.g., auth.* = auth facility, all priorities)"
    echo ""
    echo "  Transport prefixes:"
    echo "    @   = UDP (fast, unreliable)"
    echo "    @@  = TCP (reliable, recommended for important logs)"
    echo ""
    echo "  Filter types:"
    echo "    Traditional: auth.warning    (facility.priority)"
    echo "    Property:    :msg, contains, \"text\"    (match message content)"
    echo "    Expression:  if \$msg contains 'text' then /var/log/file.log"
    echo ""
    echo "To apply changes:"
    echo "  sudo systemctl restart rsyslog"
    echo "  sudo rsyslogd -N1   # Validate config syntax before restart"
}

# === Exercise 3: logrotate Configuration ===
# Problem: Configure log rotation for /var/log/myapp/ with specific requirements.
exercise_3() {
    echo "=== Exercise 3: logrotate Configuration ==="
    echo ""
    echo "Scenario: Rotate application logs daily, keep 30 days, compress with xz,"
    echo "          rotate when >100MB, and signal the app after rotation."
    echo ""

    echo "Solution: /etc/logrotate.d/myapp"
    echo ""
    cat << 'LOGROTATE'
/var/log/myapp/*.log {
    daily                       # Rotate every day
    rotate 30                   # Keep 30 rotated files (30 days of history)
    size 100M                   # Also rotate if file exceeds 100MB
                                # Note: with 'daily' AND 'size', rotation happens
                                # whichever condition is met first

    compress                    # Compress rotated logs
    compresscmd /usr/bin/xz     # Use xz instead of default gzip
    compressext .xz             # Set the compressed file extension
    delaycompress               # Don't compress the most recent rotated file
                                # (allows processes to finish writing)

    missingok                   # Don't error if log file is missing
    notifempty                  # Don't rotate if file is empty
    create 0644 root root       # Create new log file with these permissions

    postrotate
        # Send SIGHUP to the application after rotation
        # SIGHUP conventionally tells daemons to reopen log files
        [ -f /var/run/myapp.pid ] && kill -HUP $(cat /var/run/myapp.pid)
    endscript
}
LOGROTATE
    echo ""
    echo "Key logrotate concepts:"
    echo ""
    echo "  Rotation triggers:"
    echo "    daily/weekly/monthly  # Time-based rotation"
    echo "    size 100M             # Size-based rotation"
    echo "    minsize 50M           # Rotate on schedule only if >= size"
    echo "    maxsize 500M          # Rotate mid-cycle if file too large"
    echo ""
    echo "  Compression options:"
    echo "    compress              # Enable compression"
    echo "    compresscmd /usr/bin/xz   # xz offers better compression than gzip"
    echo "    delaycompress         # Wait one cycle before compressing"
    echo ""
    echo "  Scripts (run in order):"
    echo "    prerotate/endscript   # Before rotation"
    echo "    postrotate/endscript  # After rotation"
    echo "    firstaction/endscript # Before all logs processed"
    echo "    lastaction/endscript  # After all logs processed"
    echo ""
    echo "To test the configuration (dry run):"
    echo "  sudo logrotate -d /etc/logrotate.d/myapp    # Debug mode (no changes)"
    echo "  sudo logrotate -f /etc/logrotate.d/myapp    # Force rotation (for testing)"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
