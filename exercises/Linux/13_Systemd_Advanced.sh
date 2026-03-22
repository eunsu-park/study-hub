#!/bin/bash
# Exercises for Lesson 13: Systemd Advanced
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Creating Custom Systemd Service Units ===
# Problem: Write systemd unit files for a custom application with proper
#          dependencies, resource limits, and security hardening.
exercise_1() {
    echo "=== Exercise 1: Creating Custom Systemd Service Units ==="
    echo ""
    echo "Scenario: Deploy a Node.js web application as a systemd service with"
    echo "automatic restart, resource limits, and security sandboxing."
    echo ""

    echo "--- Part A: Basic Service Unit File ---"
    echo "Solution: /etc/systemd/system/webapp.service"
    cat << 'UNIT'
  [Unit]
  Description=My Web Application
  Documentation=https://github.com/company/webapp
  After=network.target postgresql.service
  Wants=postgresql.service

  [Service]
  Type=simple
  User=webapp
  Group=webapp
  WorkingDirectory=/opt/webapp
  ExecStart=/usr/bin/node /opt/webapp/server.js
  ExecReload=/bin/kill -HUP $MAINPID
  Restart=on-failure
  RestartSec=5
  StartLimitBurst=5
  StartLimitIntervalSec=60

  # Environment
  Environment=NODE_ENV=production
  Environment=PORT=3000
  EnvironmentFile=-/etc/webapp/env    # '-' means don't fail if file missing

  # Logging
  StandardOutput=journal
  StandardError=journal
  SyslogIdentifier=webapp

  [Install]
  WantedBy=multi-user.target
UNIT
    echo ""
    echo "  Explanation:"
    echo "    After= ensures network and database are up before starting"
    echo "    Wants= is a weak dependency (webapp starts even if postgresql fails)"
    echo "    Requires= would be a hard dependency (webapp fails if postgresql fails)"
    echo "    Type=simple means systemd considers the service started immediately"
    echo "    Restart=on-failure only restarts on non-zero exit (not on clean stop)"
    echo "    StartLimitBurst/IntervalSec prevents restart loops (max 5 in 60s)"
    echo ""

    echo "--- Part B: Resource Limits and Security Hardening ---"
    echo "Solution: Add to the [Service] section"
    cat << 'HARDENING'
  # Resource Limits
  LimitNOFILE=65535              # Max open file descriptors
  MemoryMax=512M                 # Hard memory limit (OOM killed if exceeded)
  MemoryHigh=384M                # Throttle at this level before hitting Max
  CPUQuota=200%                  # Max 2 CPU cores worth of time

  # Security Sandboxing
  NoNewPrivileges=yes            # Prevent privilege escalation
  ProtectSystem=strict           # Mount / as read-only (except /dev, /proc, /sys)
  ProtectHome=yes                # Hide /home, /root, /run/user
  ReadWritePaths=/opt/webapp/data /var/log/webapp
  PrivateTmp=yes                 # Isolated /tmp for this service
  ProtectKernelModules=yes       # Block module loading
  ProtectKernelTunables=yes      # Block sysctl writes
  ProtectControlGroups=yes       # Block cgroup modifications
  RestrictNamespaces=yes         # Block namespace creation
  RestrictSUIDSGID=yes           # Block SUID/SGID file creation
HARDENING
    echo ""
    echo "  Explanation:"
    echo "    ProtectSystem=strict makes the entire filesystem read-only except specified paths"
    echo "    ReadWritePaths whitelists specific directories the service needs to write to"
    echo "    PrivateTmp gives the service its own /tmp (prevents data leaks between services)"
    echo "    These settings create a security sandbox without containers"
    echo ""

    echo "--- Part C: Managing the Service ---"
    echo "Solution:"
    echo "  # Reload systemd after changing unit files"
    echo "  sudo systemctl daemon-reload"
    echo ""
    echo "  # Enable and start"
    echo "  sudo systemctl enable --now webapp.service"
    echo ""
    echo "  # Check status and logs"
    echo "  sudo systemctl status webapp"
    echo "  journalctl -u webapp -f                     # Follow logs"
    echo "  journalctl -u webapp --since '1 hour ago'   # Recent logs"
    echo ""
    echo "  # Analyze security score"
    echo "  systemd-analyze security webapp.service      # Shows hardening score"
    echo ""
    echo "  Explanation:"
    echo "    daemon-reload is REQUIRED after any unit file change"
    echo "    enable --now combines 'enable' (auto-start at boot) and 'start' (start now)"
    echo "    systemd-analyze security rates hardening from 0 (best) to 10 (worst)"
}

# === Exercise 2: Systemd Timers ===
# Problem: Replace cron jobs with systemd timers for better logging,
#          dependency management, and reliability.
exercise_2() {
    echo "=== Exercise 2: Systemd Timers (Replacing Cron) ==="
    echo ""
    echo "Scenario: Convert cron-based backup and maintenance tasks to systemd timers"
    echo "for better integration with systemd logging and service management."
    echo ""

    echo "--- Part A: Creating a Timer Unit ---"
    echo "Solution: Two files needed — a .service and a .timer"
    echo ""
    echo "  /etc/systemd/system/backup.service:"
    cat << 'SERVICE'
  [Unit]
  Description=Database Backup
  After=postgresql.service

  [Service]
  Type=oneshot
  User=backup
  ExecStart=/opt/scripts/db_backup.sh
  StandardOutput=journal
  StandardError=journal
SERVICE
    echo ""
    echo "  /etc/systemd/system/backup.timer:"
    cat << 'TIMER'
  [Unit]
  Description=Run database backup daily at 2:30 AM

  [Timer]
  OnCalendar=*-*-* 02:30:00
  RandomizedDelaySec=300
  Persistent=true

  [Install]
  WantedBy=timers.target
TIMER
    echo ""
    echo "  Explanation:"
    echo "    Type=oneshot means the service runs once and exits (not a daemon)"
    echo "    OnCalendar uses systemd calendar format: YYYY-MM-DD HH:MM:SS"
    echo "    RandomizedDelaySec adds jitter to prevent thundering herd"
    echo "    Persistent=true runs missed timers (e.g., if the server was off at 2:30 AM)"
    echo "    The timer and service must share the same base name (backup.timer -> backup.service)"
    echo ""

    echo "--- Part B: Common Timer Schedules ---"
    echo "Solution:"
    cat << 'SCHEDULES'
  # OnCalendar format: DayOfWeek YYYY-MM-DD HH:MM:SS

  # Every 15 minutes
  OnCalendar=*:0/15

  # Every hour at minute 0
  OnCalendar=hourly                    # Shorthand
  OnCalendar=*-*-* *:00:00             # Equivalent explicit form

  # Every day at midnight
  OnCalendar=daily

  # Every Monday at 6 AM
  OnCalendar=Mon *-*-* 06:00:00

  # First of every month at midnight
  OnCalendar=*-*-01 00:00:00

  # Weekdays only at 9 AM and 5 PM
  OnCalendar=Mon..Fri *-*-* 09,17:00:00

  # Alternative: monotonic timers (relative to boot or last run)
  OnBootSec=5min                       # 5 minutes after boot
  OnUnitActiveSec=1h                   # 1 hour after last activation
SCHEDULES
    echo ""
    echo "  Test a calendar expression:"
    echo "  systemd-analyze calendar 'Mon *-*-* 06:00:00'"
    echo "  # Shows next 5 trigger times — invaluable for verifying schedules"
    echo ""
    echo "  Explanation:"
    echo "    Monotonic timers (OnBootSec, OnUnitActiveSec) are relative, not calendar-based"
    echo "    Combine calendar and monotonic: run at boot AND on a schedule"
    echo "    systemd-analyze calendar validates and previews your time expressions"
    echo ""

    echo "--- Part C: Managing and Monitoring Timers ---"
    echo "Solution:"
    echo "  # Enable and start the timer (not the service!)"
    echo "  sudo systemctl enable --now backup.timer"
    echo ""
    echo "  # List all active timers with next/last run times"
    echo "  systemctl list-timers --all"
    echo ""
    echo "  # Manually trigger the associated service"
    echo "  sudo systemctl start backup.service"
    echo ""
    echo "  # Check timer and service status"
    echo "  systemctl status backup.timer"
    echo "  systemctl status backup.service"
    echo ""
    echo "  # View logs for the timer-triggered service"
    echo "  journalctl -u backup.service --since today"
    echo ""
    echo "  Explanation:"
    echo "    You enable the .timer, not the .service (the timer activates the service)"
    echo "    list-timers shows NEXT and LAST trigger times plus time until next"
    echo "    Manually starting the .service is great for testing without waiting"
    echo ""

    echo "  Advantages of timers over cron:"
    echo "    - Full journalctl logging (no separate log file management)"
    echo "    - Persistent=true catches up on missed runs"
    echo "    - Dependencies (After=, Wants=) ensure prerequisites are met"
    echo "    - Resource limits and security sandboxing from systemd"
    echo "    - RandomizedDelaySec prevents simultaneous execution across servers"
}

# === Exercise 3: Socket Activation and Dependency Management ===
# Problem: Set up socket-activated services and manage complex service
#          dependencies with systemd.
exercise_3() {
    echo "=== Exercise 3: Socket Activation and Dependency Management ==="
    echo ""
    echo "Scenario: Create a socket-activated service that only starts when a"
    echo "connection arrives, and manage a multi-service application stack with"
    echo "proper dependency ordering."
    echo ""

    echo "--- Part A: Socket-Activated Service ---"
    echo "Solution: Two files — a .socket and a .service"
    echo ""
    echo "  /etc/systemd/system/myapi.socket:"
    cat << 'SOCKET'
  [Unit]
  Description=My API Socket

  [Socket]
  ListenStream=8080
  Accept=no
  NoDelay=true
  ReusePort=true
  Backlog=128

  [Install]
  WantedBy=sockets.target
SOCKET
    echo ""
    echo "  /etc/systemd/system/myapi.service:"
    cat << 'SERVICE'
  [Unit]
  Description=My API Service
  Requires=myapi.socket
  After=myapi.socket network.target

  [Service]
  Type=notify
  User=apiuser
  ExecStart=/opt/api/bin/server --fd 3
  Restart=on-failure
  WatchdogSec=30

  [Install]
  WantedBy=multi-user.target
SERVICE
    echo ""
    echo "  Explanation:"
    echo "    Socket activation: systemd holds the socket, starts the service on first connection"
    echo "    Accept=no: the service receives all connections (vs Accept=yes: one instance per connection)"
    echo "    The service receives the socket as file descriptor 3 (first inherited fd)"
    echo "    Type=notify means the service signals readiness to systemd (sd_notify)"
    echo "    WatchdogSec=30 kills the service if it doesn't ping systemd every 30 seconds"
    echo ""

    echo "--- Part B: Dependency Management ---"
    echo "Solution: Application stack with database, cache, and web app"
    echo ""
    echo "  Dependency directives:"
    cat << 'DEPS'
  # Strong dependencies (fail together)
  Requires=postgresql.service          # If postgresql stops, this unit stops too
  BindsTo=postgresql.service           # Even stronger: stop on ANY state change

  # Weak dependencies (independent failure)
  Wants=redis.service                  # Start redis, but don't fail if redis fails

  # Ordering (does NOT imply dependency)
  After=postgresql.service redis.service  # Wait for these to start first
  Before=nginx.service                    # Start before nginx

  # Conditional (skip if not present)
  Requisite=network-online.target      # Fail immediately if not already active
DEPS
    echo ""
    echo "  Explanation:"
    echo "    Requires= + After= is the common pattern (depend + order)"
    echo "    Wants= + After= is for optional dependencies"
    echo "    After=/Before= alone only controls order, not dependency"
    echo "    Without After=, dependent services start simultaneously (parallel boot)"
    echo ""

    echo "--- Part C: Target Units for Application Stacks ---"
    echo "Solution: /etc/systemd/system/myapp.target"
    cat << 'TARGET'
  [Unit]
  Description=My Application Stack
  Requires=postgresql.service redis.service webapp.service
  After=postgresql.service redis.service
  Wants=nginx.service worker.service

  [Install]
  WantedBy=multi-user.target
TARGET
    echo ""
    echo "  Usage:"
    echo "  sudo systemctl enable --now myapp.target   # Start entire stack"
    echo "  sudo systemctl stop myapp.target            # Stop entire stack"
    echo "  sudo systemctl isolate myapp.target         # Start only this target's units"
    echo ""
    echo "  # Visualize dependencies"
    echo "  systemctl list-dependencies myapp.target"
    echo "  systemd-analyze dot myapp.target | dot -Tsvg > deps.svg"
    echo ""
    echo "  Explanation:"
    echo "    Targets group related services (like runlevels but more flexible)"
    echo "    'isolate' stops everything NOT required by the target (use with caution)"
    echo "    list-dependencies shows the dependency tree recursively"
    echo "    systemd-analyze dot generates a graphviz dependency graph"
    echo ""

    echo "--- Verification ---"
    echo "  systemctl status myapi.socket              # Check socket is listening"
    echo "  curl http://localhost:8080                  # Triggers service activation"
    echo "  systemctl status myapi.service             # Now shows active (running)"
    echo "  systemd-analyze blame                      # Boot time per service"
    echo "  systemd-analyze critical-chain              # Critical path of boot"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
