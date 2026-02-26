# Advanced systemd

**Previous**: [Security and Firewall](./12_Security_and_Firewall.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the systemd architecture, unit types, and unit file locations
2. Write custom service unit files with appropriate Type, Exec, and Restart directives
3. Apply security hardening options (ProtectSystem, PrivateTmp, NoNewPrivileges) to service units
4. Create timer units as a modern replacement for cron jobs
5. Implement socket activation to start services on demand and improve boot performance
6. Configure unit dependencies and ordering with Requires, Wants, After, and Before
7. Query, filter, and manage system logs using journalctl and journald configuration
8. Analyze boot performance and dependency chains with systemd-analyze

---

systemd is the init system and service manager that runs as PID 1 on virtually all modern Linux distributions. It replaced the older SysVinit and Upstart systems, bringing parallel service startup, on-demand activation, and unified logging. Whether you are deploying a web application, scheduling backups, or debugging why a service fails to start, you will be working with systemd. This lesson goes beyond the basic `systemctl start/stop` commands to show you how to write your own unit files and leverage systemd's full capabilities.

## Table of Contents
1. [systemd Architecture](#1-systemd-architecture)
2. [Writing Service Units](#2-writing-service-units)
3. [Timer Units](#3-timer-units)
4. [Socket Activation](#4-socket-activation)
5. [Dependencies and Ordering](#5-dependencies-and-ordering)
6. [journald Logging](#6-journald-logging)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. systemd Architecture

### 1.1 systemd Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    systemd Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  systemd (PID 1)                     │   │
│  │  • System initialization and service management     │   │
│  │  • Parallel service startup                          │   │
│  │  • Socket/D-Bus activation                           │   │
│  │  • cgroups-based resource management                 │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│    ┌──────┼──────────────────────────────┐                 │
│    │      │                              │                  │
│    ▼      ▼                              ▼                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐             │
│  │udevd │ │logind│ │journald│ │networkd│ │resolved│         │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘             │
│                                                             │
│  Unit types:                                                │
│  • .service  - Services/daemons                             │
│  • .socket   - Sockets                                      │
│  • .timer    - Timers (cron replacement)                    │
│  • .target   - Groups (runlevel replacement)                │
│  • .mount    - Mount points                                 │
│  • .device   - Devices                                      │
│  • .path     - Path monitoring                              │
│  • .slice    - Resource groups                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Unit File Locations

```bash
# Unit file locations (in priority order)
/etc/systemd/system/        # System administrator settings (highest priority)
/run/systemd/system/        # Runtime generated units
/usr/lib/systemd/system/    # Package installed units

# User units
~/.config/systemd/user/     # User settings
/etc/systemd/user/          # Global user settings
/usr/lib/systemd/user/      # Package installed user units

# Check unit file location
systemctl show -p FragmentPath nginx.service
systemctl cat nginx.service
```

### 1.3 Basic systemctl Commands

```bash
# Service management
systemctl start nginx
systemctl stop nginx
systemctl restart nginx
systemctl reload nginx       # Reload configuration only
systemctl status nginx

# Auto-start on boot
systemctl enable nginx
systemctl disable nginx
systemctl is-enabled nginx

# Service masking (prevent starting completely)
systemctl mask nginx
systemctl unmask nginx

# Unit list
systemctl list-units
systemctl list-units --type=service
systemctl list-units --state=failed

# Unit file list
systemctl list-unit-files

# Check dependencies
systemctl list-dependencies nginx.service
systemctl list-dependencies --reverse nginx.service

# Daemon reload (after modifying unit files)
systemctl daemon-reload
```

---

## 2. Writing Service Units

### 2.1 Basic Service Unit

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application Service
Documentation=https://myapp.example.com/docs
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=myapp
Group=myapp
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/myapp --config /etc/myapp/config.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5

# Environment variables
Environment=NODE_ENV=production
EnvironmentFile=-/etc/myapp/env  # - means no error if file doesn't exist

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=myapp

[Install]
WantedBy=multi-user.target
```

### 2.2 Service Types

```ini
# Type=simple (default)
# Process is the main process
[Service]
Type=simple
ExecStart=/usr/bin/myapp

# Type=forking
# Process forks and parent exits
[Service]
Type=forking
PIDFile=/var/run/myapp.pid
ExecStart=/usr/bin/myapp --daemon

# Type=oneshot
# One-time task that exits after starting
[Service]
Type=oneshot
ExecStart=/usr/local/bin/setup-script.sh
RemainAfterExit=yes  # Stay active after exit

# Type=notify
# Process notifies with sd_notify() when ready
[Service]
Type=notify
ExecStart=/usr/bin/myapp-with-notify

# Type=dbus
# Ready when D-Bus name is acquired
[Service]
Type=dbus
BusName=org.example.MyApp
ExecStart=/usr/bin/myapp

# Type=idle
# Execute after other jobs complete (boot messages, etc.)
[Service]
Type=idle
ExecStart=/usr/bin/welcome-message
```

### 2.3 Exec Options

```ini
[Service]
# Start/stop commands
ExecStartPre=/usr/bin/myapp-check    # Before start
ExecStart=/usr/bin/myapp             # Main command
ExecStartPost=/usr/bin/myapp-notify  # After start
ExecReload=/bin/kill -HUP $MAINPID   # On reload
ExecStop=/usr/bin/myapp stop         # Stop command
ExecStopPost=/usr/bin/cleanup        # After stop

# Continue even if failed (- prefix)
ExecStartPre=-/usr/bin/optional-check

# Use shell (; for multiple commands)
ExecStart=/bin/sh -c 'echo start && /usr/bin/myapp'

# Timeouts
TimeoutStartSec=30
TimeoutStopSec=30
TimeoutSec=30  # Sets both

# Restart conditions
Restart=no              # Don't restart
Restart=on-success      # Only on clean exit
Restart=on-failure      # Only on non-zero exit
Restart=on-abnormal     # On signal/timeout
Restart=on-watchdog     # On watchdog timeout
Restart=on-abort        # On uncaught signal
Restart=always          # Always restart

RestartSec=5            # Wait time before restart
RestartPreventExitStatus=1 23  # Don't restart on these exit codes
```

### 2.4 Security Options

```ini
[Service]
# User/Group
User=myapp
Group=myapp
DynamicUser=yes  # Create dynamic user (temporary)

# Filesystem access restrictions
ProtectSystem=strict     # /usr, /boot read-only
ProtectHome=yes          # /home inaccessible
PrivateTmp=yes           # Isolated /tmp
ReadWritePaths=/var/lib/myapp
ReadOnlyPaths=/etc/myapp

# Network restrictions
PrivateNetwork=yes       # Isolated network
RestrictAddressFamilies=AF_INET AF_INET6

# System call filtering
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources

# Other security
NoNewPrivileges=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=1G
CPUQuota=50%
```

### 2.5 Complete Example: Node.js App

```ini
# /etc/systemd/system/nodeapp.service
[Unit]
Description=Node.js Application
Documentation=https://example.com/docs
After=network.target mongodb.service
Wants=mongodb.service

[Service]
Type=simple
User=nodeapp
Group=nodeapp
WorkingDirectory=/opt/nodeapp

# Node.js path
Environment=PATH=/opt/nodeapp/node/bin:/usr/bin
Environment=NODE_ENV=production
EnvironmentFile=/etc/nodeapp/env

# Execution
ExecStart=/opt/nodeapp/node/bin/node /opt/nodeapp/app.js
ExecReload=/bin/kill -HUP $MAINPID

# Restart policy
Restart=always
RestartSec=10
WatchdogSec=30

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nodeapp

# Security
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths=/var/lib/nodeapp /var/log/nodeapp

# Resource limits
LimitNOFILE=65536
MemoryMax=2G

[Install]
WantedBy=multi-user.target
```

---

## 3. Timer Units

### 3.1 Timer Basics

```ini
# /etc/systemd/system/backup.timer
[Unit]
Description=Daily Backup Timer

[Timer]
# Real-time (wallclock) timer
OnCalendar=*-*-* 02:00:00  # Daily at 2 AM

# Or monotonic timer
# OnBootSec=15min           # 15 minutes after boot
# OnUnitActiveSec=1h        # 1 hour after last activation

# Accuracy (battery saving)
AccuracySec=1min

# Handle missed runs
Persistent=yes  # Compensate for missed runs while system was off

[Install]
WantedBy=timers.target

---
# /etc/systemd/system/backup.service
[Unit]
Description=Backup Service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
```

### 3.2 OnCalendar Syntax

```bash
# OnCalendar format: DayOfWeek Year-Month-Day Hour:Minute:Second

# Daily at midnight
OnCalendar=daily
OnCalendar=*-*-* 00:00:00

# Hourly
OnCalendar=hourly
OnCalendar=*-*-* *:00:00

# Every Monday at 6 AM
OnCalendar=Mon *-*-* 06:00:00
OnCalendar=weekly

# First day of month
OnCalendar=monthly
OnCalendar=*-*-01 00:00:00

# January 1st every year
OnCalendar=yearly
OnCalendar=*-01-01 00:00:00

# Every 5 minutes
OnCalendar=*:0/5
OnCalendar=*-*-* *:00/5:00

# Weekdays at 9 AM
OnCalendar=Mon..Fri *-*-* 09:00:00

# Specific date
OnCalendar=2024-12-25 00:00:00

# Range
OnCalendar=*-*-* 08..18:00:00  # Every hour (8 AM-6 PM)

# Test timer
systemd-analyze calendar "Mon *-*-* 09:00:00"
systemd-analyze calendar --iterations=5 "daily"
```

### 3.3 Timer Management

```bash
# Start/enable timer
systemctl start backup.timer
systemctl enable backup.timer

# Timer list
systemctl list-timers
systemctl list-timers --all

# Timer status
systemctl status backup.timer

# Run immediately (bypass timer)
systemctl start backup.service

# Migrate from cron
# crontab: 0 2 * * * /usr/local/bin/backup.sh
# → systemd timer: OnCalendar=*-*-* 02:00:00
```

### 3.4 Multiple Schedules

```ini
# /etc/systemd/system/multi-schedule.timer
[Unit]
Description=Multiple Schedule Timer

[Timer]
# Multiple times can be specified
OnCalendar=Mon *-*-* 06:00:00
OnCalendar=Wed *-*-* 06:00:00
OnCalendar=Fri *-*-* 06:00:00

# Or
OnCalendar=Mon,Wed,Fri *-*-* 06:00:00

[Install]
WantedBy=timers.target
```

---

## 4. Socket Activation

### 4.1 Socket Activation Concept

```
┌─────────────────────────────────────────────────────────────┐
│                    Socket Activation Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. On boot                                                 │
│     ┌─────────┐                                            │
│     │ systemd │──▶ Open socket (don't start service)      │
│     └─────────┘                                            │
│          │                                                  │
│          ▼                                                  │
│     ┌─────────┐                                            │
│     │ socket  │ (waiting)                                  │
│     └─────────┘                                            │
│                                                             │
│  2. On connection request                                  │
│     ┌─────────┐      ┌─────────┐      ┌─────────┐         │
│     │ Client  │─────▶│ socket  │─────▶│ systemd │         │
│     └─────────┘      └─────────┘      └─────────┘         │
│                                              │              │
│                                              ▼              │
│                                        Start service        │
│                                              │              │
│                                              ▼              │
│                                        ┌─────────┐         │
│                                        │ service │         │
│                                        └─────────┘         │
│                                                             │
│  Advantages:                                                │
│  • Faster boot time                                         │
│  • Start service only on demand                             │
│  • Keep connections during service restart                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Socket Units

```ini
# /etc/systemd/system/myapp.socket
[Unit]
Description=My App Socket

[Socket]
# TCP socket
ListenStream=8080
# Or specify IP
# ListenStream=127.0.0.1:8080
# Or IPv6
# ListenStream=[::1]:8080

# Unix socket
# ListenStream=/run/myapp/myapp.sock
# SocketUser=myapp
# SocketGroup=myapp
# SocketMode=0660

# UDP socket
# ListenDatagram=8081

# Connection backlog
Backlog=128

# One service instance per connection
Accept=no  # Default, one service handles all connections
# Accept=yes  # New instance per connection (inetd style)

# Link to service (default: same name.service)
# Service=myapp.service

[Install]
WantedBy=sockets.target

---
# /etc/systemd/system/myapp.service
[Unit]
Description=My App Service
Requires=myapp.socket

[Service]
Type=simple
ExecStart=/opt/myapp/bin/myapp
# Socket is passed as fd 3 (or $LISTEN_FDS)

[Install]
WantedBy=multi-user.target
```

### 4.3 Socket Activation Service Example

```python
#!/usr/bin/env python3
# /opt/myapp/bin/myapp.py
# Python server with socket activation support

import socket
import os
import sys

def get_systemd_socket():
    """Get socket passed from systemd"""
    # LISTEN_FDS: number of fds passed
    # LISTEN_PID: target process PID
    listen_fds = int(os.environ.get('LISTEN_FDS', 0))
    listen_pid = int(os.environ.get('LISTEN_PID', 0))

    if listen_pid != os.getpid():
        return None

    if listen_fds >= 1:
        # Starts from fd 3 (0=stdin, 1=stdout, 2=stderr)
        return socket.fromfd(3, socket.AF_INET, socket.SOCK_STREAM)

    return None

def main():
    # Use systemd socket or create new socket
    sock = get_systemd_socket()
    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', 8080))
        sock.listen(128)
        print("Listening on port 8080")
    else:
        print("Using systemd socket")

    while True:
        conn, addr = sock.accept()
        with conn:
            print(f"Connection from {addr}")
            conn.sendall(b"Hello from socket-activated service!\n")

if __name__ == '__main__':
    main()
```

---

## 5. Dependencies and Ordering

### 5.1 Dependency Directives

```ini
[Unit]
# Requires: Required dependency (this unit fails if dependency fails)
Requires=postgresql.service

# Wants: Optional dependency (continue even if dependency fails)
Wants=redis.service

# Requisite: Depend only on already active unit
Requisite=network.target

# BindsTo: Strong dependency (stop this unit if dependency stops)
BindsTo=libvirtd.service

# PartOf: Restart/stop with dependency
PartOf=docker.service

# Conflicts: Cannot run simultaneously
Conflicts=shutdown.target
```

### 5.2 Ordering Directives

```ini
[Unit]
# After: Start after specified units
After=network.target postgresql.service

# Before: Start before specified units
Before=httpd.service

# Dependencies and ordering are separate!
# Wants=postgresql.service → Try to start with postgresql
# After=postgresql.service → Start after postgresql completes

# Correct combination
Wants=postgresql.service
After=postgresql.service
```

### 5.3 Target Units

```ini
# /etc/systemd/system/myapp.target
[Unit]
Description=My Application Stack
Requires=myapp-web.service myapp-worker.service
After=myapp-web.service myapp-worker.service

[Install]
WantedBy=multi-user.target

# Usage
systemctl start myapp.target    # Start all related services
systemctl stop myapp.target     # Stop all related services
systemctl restart myapp.target
```

### 5.4 Dependency Visualization

```bash
# Dependency tree
systemctl list-dependencies nginx.service
systemctl list-dependencies --reverse nginx.service

# Boot sequence analysis
systemd-analyze
systemd-analyze blame
systemd-analyze critical-chain
systemd-analyze critical-chain nginx.service

# Generate graph (SVG)
systemd-analyze dot | dot -Tsvg > systemd.svg
systemd-analyze dot "nginx.service" | dot -Tsvg > nginx-deps.svg
```

---

## 6. journald Logging

### 6.1 journalctl Basics

```bash
# All logs
journalctl

# Specific unit logs
journalctl -u nginx.service

# Real-time logs (tail -f)
journalctl -f
journalctl -fu nginx.service

# Boot logs
journalctl -b          # Current boot
journalctl -b -1       # Previous boot
journalctl --list-boots

# Time range
journalctl --since "2024-01-01"
journalctl --since "1 hour ago"
journalctl --since "2024-01-01" --until "2024-01-02"
journalctl --since yesterday

# Priority filter
journalctl -p err      # error and above
journalctl -p warning  # warning and above
# 0=emerg, 1=alert, 2=crit, 3=err, 4=warning, 5=notice, 6=info, 7=debug

# Kernel messages
journalctl -k
journalctl --dmesg

# JSON output
journalctl -o json
journalctl -o json-pretty

# Disk usage
journalctl --disk-usage

# Clean logs
journalctl --vacuum-size=500M
journalctl --vacuum-time=7d
```

### 6.2 journald Configuration

```ini
# /etc/systemd/journald.conf
[Journal]
# Storage method
Storage=persistent     # Persistent storage (/var/log/journal)
# Storage=volatile     # Memory only (/run/log/journal)
# Storage=auto         # Persistent if /var/log/journal exists

# Size limits
SystemMaxUse=500M      # Maximum disk usage
SystemMaxFileSize=50M  # Maximum individual file size
RuntimeMaxUse=100M     # Maximum runtime (memory)

# Retention period
MaxRetentionSec=1month

# Compression
Compress=yes

# Forward to syslog
ForwardToSyslog=no

# Console output
ForwardToConsole=no

# Rate limiting
RateLimitIntervalSec=30s
RateLimitBurst=10000
```

```bash
# Apply configuration
systemctl restart systemd-journald

# Create persistent storage directory
mkdir -p /var/log/journal
systemd-tmpfiles --create --prefix /var/log/journal
```

### 6.3 Structured Logging

```bash
# Send log with systemd-cat
echo "Hello" | systemd-cat -t myapp -p info

# In scripts
#!/bin/bash
exec 1> >(systemd-cat -t myscript -p info)
exec 2> >(systemd-cat -t myscript -p err)
echo "This goes to journal"
```

```python
# In Python (systemd.journal)
from systemd import journal

journal.send('Hello from Python',
             PRIORITY=journal.LOG_INFO,
             SYSLOG_IDENTIFIER='myapp',
             MYFIELD='custom_value')
```

### 6.4 Advanced Log Filtering

```bash
# Field-based filtering
journalctl _SYSTEMD_UNIT=nginx.service
journalctl _UID=1000
journalctl _PID=1234
journalctl _COMM=nginx

# Multiple conditions (AND)
journalctl _SYSTEMD_UNIT=nginx.service _PID=1234

# OR conditions
journalctl _SYSTEMD_UNIT=nginx.service + _SYSTEMD_UNIT=php-fpm.service

# Specific field output
journalctl -o verbose
journalctl -u nginx --output-fields=MESSAGE,_PID

# With grep
journalctl -u nginx | grep -i error
journalctl -u nginx -g "error|warning"  # -g is grep pattern
```

---

## 7. Practice Exercises

### Exercise 1: Web Application Service
```ini
# Requirements:
# 1. Register Python/Node.js web app as service
# 2. Set restart policy (on-failure)
# 3. Use environment variable file
# 4. Apply security options

# Write service unit:
```

### Exercise 2: Backup Timer
```ini
# Requirements:
# 1. Run backup daily at 3 AM
# 2. Full backup every Sunday
# 3. Compensate for missed runs
# 4. Log to journal

# Write timer and service units:
```

### Exercise 3: Socket Activated Service
```ini
# Requirements:
# 1. Listen on port 9000
# 2. Start service only on connection
# 3. Auto-stop on idle (IdleTimeout)

# Write socket and service units:
```

### Exercise 4: Microservices Stack
```ini
# Requirements:
# 1. API service (api.service)
# 2. Worker service (worker.service)
# 3. Database dependency
# 4. Complete stack target

# Write all unit files:
```

---

## References

- [systemd Documentation](https://www.freedesktop.org/software/systemd/man/)
- [Arch Wiki - systemd](https://wiki.archlinux.org/title/systemd)
- [RHEL systemd Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_basic_system_settings/managing-services-with-systemd_configuring-basic-system-settings)

---

**Previous**: [Security and Firewall](./12_Security_and_Firewall.md)
