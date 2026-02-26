# Log Management

**Previous**: [SELinux and AppArmor](./17_SELinux_AppArmor.md) | **Next**: [Backup and Recovery](./19_Backup_Recovery.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure systemd-journald for persistent storage and query logs with journalctl
2. Write advanced journalctl queries using time, priority, service, and field filters
3. Configure rsyslog rules, filters, and templates for structured log processing
4. Set up logrotate policies to manage log file size and retention
5. Deploy a centralized remote logging infrastructure with rsyslog over TLS
6. Analyze web server and system logs using tools like lnav and GoAccess

---

## Table of Contents

1. [Linux Log System Overview](#1-linux-log-system-overview)
2. [systemd-journald](#2-systemd-journald)
3. [Advanced journalctl Usage](#3-advanced-journalctl-usage)
4. [rsyslog Configuration](#4-rsyslog-configuration)
5. [logrotate](#5-logrotate)
6. [Remote Log Collection](#6-remote-log-collection)
7. [Log Analysis Tools](#7-log-analysis-tools)

---

Logs are the flight recorder of your servers. When an intrusion occurs, a service crashes, or performance degrades, logs are the first place you look for answers. Effective log management means more than just writing files to /var/log -- it means structuring, rotating, centralizing, and querying logs efficiently so that you can detect problems in minutes rather than days.

## 1. Linux Log System Overview

### Log System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Applications / Services                   │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
                ▼                         ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│    systemd-journald       │   │    rsyslog / syslog-ng      │
│    (Binary journal)       │──▶│    (Text log files)         │
└───────────────────────────┘   └─────────────────────────────┘
                │                         │
                ▼                         ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│  /run/log/journal/        │   │  /var/log/*.log             │
│  /var/log/journal/        │   │  Remote server              │
└───────────────────────────┘   └─────────────────────────────┘
```

### Main Log Files

| File | Contents |
|------|----------|
| `/var/log/messages` | General system messages (RHEL/CentOS) |
| `/var/log/syslog` | General system messages (Ubuntu/Debian) |
| `/var/log/auth.log` | Authentication logs (Ubuntu) |
| `/var/log/secure` | Authentication logs (RHEL) |
| `/var/log/kern.log` | Kernel messages |
| `/var/log/dmesg` | Boot time kernel messages |
| `/var/log/cron` | Cron job logs |
| `/var/log/maillog` | Mail server logs |

### Log Priority (Severity)

| Level | Name | Description |
|-------|------|-------------|
| 0 | emerg | System unusable |
| 1 | alert | Immediate action required |
| 2 | crit | Critical error |
| 3 | err | Error |
| 4 | warning | Warning |
| 5 | notice | Normal but noteworthy |
| 6 | info | Informational message |
| 7 | debug | Debug message |

---

## 2. systemd-journald

### journald Configuration

```bash
# Configuration file
sudo vi /etc/systemd/journald.conf
```

```ini
# /etc/systemd/journald.conf
[Journal]
# Storage method: volatile(memory), persistent(disk), auto, none
Storage=persistent

# Maximum size (for disk storage)
SystemMaxUse=500M
SystemKeepFree=1G
SystemMaxFileSize=50M
SystemMaxFiles=100

# Runtime storage (memory)
RuntimeMaxUse=50M

# Log compression
Compress=yes

# Sealing (tamper-evident)
Seal=yes

# Forward to rsyslog
ForwardToSyslog=yes

# Console output
ForwardToConsole=no

# Maximum retention period
MaxRetentionSec=1month

# Rate limiting
RateLimitIntervalSec=30s
RateLimitBurst=10000
```

```bash
# Apply configuration
sudo systemctl restart systemd-journald
```

### Enabling Persistent Storage

```bash
# Create journal directory (persistent storage)
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal

# Set permissions
sudo chown root:systemd-journal /var/log/journal
sudo chmod 2755 /var/log/journal

# Restart journald
sudo systemctl restart systemd-journald
```

### Checking Journal Status

```bash
# Disk usage
journalctl --disk-usage

# Check journal files
journalctl --header

# Verify journal integrity
journalctl --verify
```

---

## 3. Advanced journalctl Usage

### Basic Queries

```bash
# All logs
journalctl

# Reverse order (newest first)
journalctl -r

# Real-time follow
journalctl -f

# Last N lines
journalctl -n 50

# Output without pager
journalctl --no-pager
```

### Time-based Filtering

```bash
# Today's logs
journalctl --since today

# Yesterday's logs
journalctl --since yesterday --until today

# Specific time range
journalctl --since "2024-01-15 10:00:00" --until "2024-01-15 12:00:00"

# Relative time
journalctl --since "1 hour ago"
journalctl --since "30 minutes ago"

# Boot related
journalctl -b          # Current boot
journalctl -b -1       # Previous boot
journalctl --list-boots # Boot list
```

### Service/Unit Filtering

```bash
# Specific service
journalctl -u nginx.service
journalctl -u nginx -u php-fpm

# Kernel messages
journalctl -k

# Specific PID
journalctl _PID=1234

# Specific executable
journalctl /usr/bin/bash

# Specific user
journalctl _UID=1000
```

### Priority Filtering

```bash
# Error and above
journalctl -p err

# Warning and above
journalctl -p warning

# Range specification
journalctl -p err..crit

# Numeric specification
journalctl -p 3
```

### Output Formats

```bash
# JSON format
journalctl -o json
journalctl -o json-pretty

# Verbose output
journalctl -o verbose

# Short output
journalctl -o short
journalctl -o short-precise  # Include microseconds

# cat style (message only)
journalctl -o cat

# Export format
journalctl -o export
```

### Complex Queries

```bash
# Combination (AND)
journalctl -u nginx -p err --since today

# Custom fields
journalctl _SYSTEMD_UNIT=sshd.service _PID=1234

# Message search
journalctl -g "error|fail|critical"

# List field values
journalctl -F _SYSTEMD_UNIT
journalctl -F PRIORITY
```

### Journal Maintenance

```bash
# Delete old logs (time-based)
sudo journalctl --vacuum-time=30d

# Delete old logs (size-based)
sudo journalctl --vacuum-size=500M

# Delete based on file count
sudo journalctl --vacuum-files=10

# Delete all journals
sudo journalctl --rotate
sudo journalctl --vacuum-time=1s
```

---

## 4. rsyslog Configuration

### Basic rsyslog Configuration

```bash
# Main configuration file
sudo vi /etc/rsyslog.conf
```

```bash
# /etc/rsyslog.conf (main sections)

# Load modules
module(load="imuxsock")    # Local system log
module(load="imjournal")   # journald integration
module(load="imklog")      # Kernel log

# Global settings
global(
    workDirectory="/var/lib/rsyslog"
    maxMessageSize="64k"
)

# Default rules
*.info;mail.none;authpriv.none;cron.none    /var/log/messages
authpriv.*                                   /var/log/secure
mail.*                                       -/var/log/maillog
cron.*                                       /var/log/cron
*.emerg                                      :omusrmsg:*
```

### Filter Syntax

```bash
# Basic syntax: facility.priority  action

# facility: auth, authpriv, cron, daemon, kern, mail, user, local0-7, *
# priority: emerg, alert, crit, err, warning, notice, info, debug, none, *

# Examples
kern.*                      /var/log/kern.log        # All kernel messages
*.crit                      /var/log/critical.log    # All critical errors
mail.err                    /var/log/mail-err.log    # Mail errors
*.info;mail.none            /var/log/messages        # info and above, exclude mail
```

### Advanced Filtering

```bash
# /etc/rsyslog.d/custom.conf

# Property-based filter
:programname, isequal, "nginx" /var/log/nginx/access.log
:programname, startswith, "postfix" /var/log/mail/postfix.log

# Message content based
:msg, contains, "error" /var/log/errors.log
:msg, regex, "failed.*authentication" /var/log/auth-failures.log

# Complex conditions
if $programname == 'sshd' and $msg contains 'Failed' then {
    action(type="omfile" file="/var/log/ssh-failures.log")
    stop
}
```

### Using Templates

```bash
# Custom log format
template(name="CustomFormat" type="string"
    string="%timegenerated% %HOSTNAME% %syslogtag%%msg%\n")

# JSON format
template(name="JsonFormat" type="list") {
    constant(value="{")
    constant(value="\"timestamp\":\"")     property(name="timereported" dateFormat="rfc3339")
    constant(value="\",\"host\":\"")       property(name="hostname")
    constant(value="\",\"program\":\"")    property(name="programname")
    constant(value="\",\"severity\":\"")   property(name="syslogseverity-text")
    constant(value="\",\"message\":\"")    property(name="msg" format="json")
    constant(value="\"}\n")
}

# Apply template
*.* action(type="omfile" file="/var/log/json.log" template="JsonFormat")
```

### Conditional Processing

```bash
# RainerScript syntax
if $programname == 'nginx' then {
    if $syslogseverity <= 3 then {
        # Error and above go to separate file
        action(type="omfile" file="/var/log/nginx/error.log")
    } else {
        # Rest go to general log
        action(type="omfile" file="/var/log/nginx/access.log")
    }
    stop
}
```

---

## 5. logrotate

### Basic Configuration

```bash
# Global configuration
sudo vi /etc/logrotate.conf
```

```bash
# /etc/logrotate.conf

# Rotation cycle: daily, weekly, monthly
weekly

# Number of logs to keep
rotate 4

# Create new log file
create

# Use date extension
dateext

# Compression
compress
delaycompress

# Ignore empty log files
notifempty

# Include individual configurations
include /etc/logrotate.d
```

### Application-specific Configuration

```bash
# /etc/logrotate.d/nginx
/var/log/nginx/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}
```

```bash
# /etc/logrotate.d/mysql
/var/log/mysql/*.log {
    daily
    rotate 7
    missingok
    create 640 mysql adm
    compress
    sharedscripts
    postrotate
        test -x /usr/bin/mysqladmin || exit 0
        if [ -f /root/.my.cnf ]; then
            /usr/bin/mysqladmin flush-logs
        fi
    endscript
}
```

### Advanced Options

```bash
# /etc/logrotate.d/custom-app
/var/log/myapp/*.log {
    # Rotation cycle
    daily

    # Number to keep
    rotate 30

    # Size-based rotation
    size 100M

    # Minimum size (don't rotate if smaller)
    minsize 10M

    # Maximum retention period
    maxage 365

    # Compression settings
    compress
    compresscmd /usr/bin/xz
    compressoptions -9
    compressext .xz
    delaycompress

    # No error if file missing
    missingok

    # Don't rotate empty files
    notifempty

    # Create new file
    create 0644 root root

    # Or keep existing file
    # copytruncate

    # Scripts
    prerotate
        echo "About to rotate logs"
    endscript

    postrotate
        systemctl reload myapp
    endscript

    firstaction
        echo "Starting log rotation batch"
    endscript

    lastaction
        echo "Finished log rotation batch"
    endscript
}
```

### Testing logrotate

```bash
# Dry run (no actual execution)
sudo logrotate -d /etc/logrotate.d/nginx

# Force execution
sudo logrotate -f /etc/logrotate.d/nginx

# Verbose output
sudo logrotate -v /etc/logrotate.conf

# Check status file
cat /var/lib/logrotate/status
```

---

## 6. Remote Log Collection

### rsyslog Server Configuration

```bash
# /etc/rsyslog.conf (server)

# Enable UDP reception
module(load="imudp")
input(type="imudp" port="514")

# Enable TCP reception
module(load="imtcp")
input(type="imtcp" port="514")

# Separate logs by host
$template RemoteLogs,"/var/log/remote/%HOSTNAME%/%PROGRAMNAME%.log"
*.* ?RemoteLogs

# Or using RainerScript
template(name="RemoteLogsByHost" type="string"
    string="/var/log/remote/%HOSTNAME%/%$YEAR%-%$MONTH%-%$DAY%.log")

if $fromhost-ip != '127.0.0.1' then {
    action(type="omfile" dynaFile="RemoteLogsByHost")
    stop
}
```

### rsyslog Client Configuration

```bash
# /etc/rsyslog.d/remote.conf (client)

# Send via UDP (@)
*.* @logserver.example.com:514

# Send via TCP (@@)
*.* @@logserver.example.com:514

# Send only specific logs
auth.* @@logserver.example.com:514
*.err @@logserver.example.com:514

# Queue configuration (reliable delivery)
action(
    type="omfwd"
    target="logserver.example.com"
    port="514"
    protocol="tcp"
    queue.type="LinkedList"
    queue.filename="remote_queue"
    queue.saveOnShutdown="on"
    queue.maxDiskSpace="1g"
    action.resumeRetryCount="-1"
)
```

### TLS Encryption Setup

```bash
# Server configuration
module(load="imtcp"
    StreamDriver.Name="gtls"
    StreamDriver.Mode="1"
    StreamDriver.AuthMode="x509/name"
)

global(
    DefaultNetstreamDriver="gtls"
    DefaultNetstreamDriverCAFile="/etc/rsyslog.d/ca.pem"
    DefaultNetstreamDriverCertFile="/etc/rsyslog.d/server-cert.pem"
    DefaultNetstreamDriverKeyFile="/etc/rsyslog.d/server-key.pem"
)

input(type="imtcp" port="6514")
```

```bash
# Client configuration
global(
    DefaultNetstreamDriver="gtls"
    DefaultNetstreamDriverCAFile="/etc/rsyslog.d/ca.pem"
    DefaultNetstreamDriverCertFile="/etc/rsyslog.d/client-cert.pem"
    DefaultNetstreamDriverKeyFile="/etc/rsyslog.d/client-key.pem"
)

action(
    type="omfwd"
    target="logserver.example.com"
    port="6514"
    protocol="tcp"
    StreamDriver="gtls"
    StreamDriverMode="1"
    StreamDriverAuthMode="x509/name"
)
```

### Firewall Configuration

```bash
# RHEL/CentOS (firewalld)
sudo firewall-cmd --permanent --add-port=514/tcp
sudo firewall-cmd --permanent --add-port=514/udp
sudo firewall-cmd --reload

# Ubuntu (ufw)
sudo ufw allow 514/tcp
sudo ufw allow 514/udp
```

---

## 7. Log Analysis Tools

### lnav (Log Navigator)

```bash
# Installation
# Ubuntu/Debian
sudo apt install lnav

# RHEL/CentOS
sudo yum install epel-release
sudo yum install lnav

# Usage
lnav /var/log/syslog
lnav /var/log/nginx/*.log

# Remote log (SSH)
lnav ssh://user@server/var/log/syslog

# Filtering (internal commands)
:filter-in error
:filter-out debug
```

### multitail

```bash
# Installation
sudo apt install multitail  # Ubuntu
sudo yum install multitail  # RHEL

# Monitor multiple files simultaneously
multitail /var/log/syslog /var/log/auth.log

# Color distinction
multitail -ci green /var/log/access.log -ci red /var/log/error.log
```

### GoAccess (Web Log Analysis)

```bash
# Installation
sudo apt install goaccess  # Ubuntu
sudo yum install goaccess  # RHEL

# Real-time analysis in terminal
goaccess /var/log/nginx/access.log -c

# Generate HTML report
goaccess /var/log/nginx/access.log -o report.html --log-format=COMBINED

# Real-time HTML dashboard
goaccess /var/log/nginx/access.log -o /var/www/html/report.html \
    --log-format=COMBINED --real-time-html
```

### Simple Analysis Commands

```bash
# Top requesting IPs
awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head

# HTTP status code distribution
awk '{print $9}' /var/log/nginx/access.log | sort | uniq -c | sort -rn

# Requests by hour
awk '{print $4}' /var/log/nginx/access.log | cut -d: -f2 | sort | uniq -c

# Error message frequency
grep -i error /var/log/syslog | awk '{print $5}' | sort | uniq -c | sort -rn | head

# Failed SSH logins
grep "Failed password" /var/log/auth.log | awk '{print $11}' | sort | uniq -c | sort -rn
```

---

## Practice Problems

### Problem 1: journalctl Query

Write commands to query logs with the following conditions:
1. Only nginx service error logs (today)
2. Output logs for a specific PID (1234) in JSON
3. Kernel warning and above messages from the last hour

### Problem 2: rsyslog Filter

Write rsyslog rules that meet the following requirements:
- Save all auth messages to `/var/log/auth-all.log`
- Also save messages containing "Failed" to `/var/log/failures.log`
- Send error and above logs to remote server `192.168.1.100`

### Problem 3: logrotate Configuration

For logs in `/var/log/myapp/` directory:
- Daily rotation
- Keep for 30 days
- Rotate when exceeding 100MB
- xz compression
- Send SIGHUP to application after rotation

---

## Answers

### Problem 1 Answer

```bash
# 1. nginx error logs (today)
journalctl -u nginx -p err --since today

# 2. PID 1234 JSON output
journalctl _PID=1234 -o json-pretty

# 3. Kernel warning and above (1 hour)
journalctl -k -p warning --since "1 hour ago"
```

### Problem 2 Answer

```bash
# /etc/rsyslog.d/custom.conf

# auth log
auth.*  /var/log/auth-all.log

# Messages containing Failed
:msg, contains, "Failed" /var/log/failures.log

# Remote transmission (error and above)
*.err @@192.168.1.100:514
```

### Problem 3 Answer

```bash
# /etc/logrotate.d/myapp
/var/log/myapp/*.log {
    daily
    rotate 30
    size 100M
    compress
    compresscmd /usr/bin/xz
    compressext .xz
    delaycompress
    missingok
    notifempty
    create 0644 root root
    postrotate
        [ -f /var/run/myapp.pid ] && kill -HUP $(cat /var/run/myapp.pid)
    endscript
}
```

---

---

## References

- [systemd Journal](https://www.freedesktop.org/software/systemd/man/systemd-journald.service.html)
- [rsyslog Documentation](https://www.rsyslog.com/doc/)
- [logrotate Manual](https://linux.die.net/man/8/logrotate)
- `man journalctl`, `man rsyslog.conf`, `man logrotate`

---

**Previous**: [SELinux and AppArmor](./17_SELinux_AppArmor.md) | **Next**: [Backup and Recovery](./19_Backup_Recovery.md)
