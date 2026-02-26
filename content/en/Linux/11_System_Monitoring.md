# System Monitoring

**Previous**: [Network Basics](./10_Network_Basics.md) | **Next**: [Security and Firewall](./12_Security_and_Firewall.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Retrieve system information using uname, hostnamectl, and lsb_release
2. Inspect CPU details and monitor CPU usage with lscpu, top, and vmstat
3. Monitor memory utilization using free and /proc/meminfo
4. Check disk usage and partition information with df, du, lsblk, and fdisk
5. View and analyze system logs using traditional log files and journalctl
6. Schedule recurring tasks with cron and interpret crontab syntax
7. Interpret load averages and diagnose system performance with vmstat and iostat
8. Write monitoring scripts that report system status and alert on thresholds

---

A server that is not being monitored is a server waiting to fail. System monitoring lets you spot problems before they become outages -- a filling disk, a memory leak, a spike in CPU usage, or a failing service. The tools in this lesson form the foundation of observability on Linux, and they work on everything from a Raspberry Pi to a thousand-node cloud cluster.

## 1. System Information

### uname - Kernel Information

```bash
# Full information
uname -a

# Kernel version
uname -r

# Operating system
uname -s

# Hardware
uname -m
```

Output:
```
Linux server01 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux
```

### hostnamectl

```bash
hostnamectl
```

Output:
```
 Static hostname: server01
       Icon name: computer-vm
         Chassis: vm
      Machine ID: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
         Boot ID: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  Virtualization: kvm
Operating System: Ubuntu 22.04.3 LTS
          Kernel: Linux 5.15.0-91-generic
    Architecture: x86-64
```

### lsb_release - Distribution Information

```bash
# Ubuntu/Debian
lsb_release -a

# Or
cat /etc/os-release
```

---

## 2. CPU Information

### /proc/cpuinfo

```bash
# CPU information
cat /proc/cpuinfo

# CPU model only
grep "model name" /proc/cpuinfo | head -1

# CPU core count
grep -c "processor" /proc/cpuinfo
# Or
nproc
```

### lscpu

```bash
lscpu
```

Output:
```
Architecture:          x86_64
CPU(s):                4
Thread(s) per core:    2
Core(s) per socket:    2
Socket(s):             1
Model name:            Intel(R) Core(TM) i5-8250U
CPU MHz:               1600.000
```

### CPU Usage

```bash
# Check in top
top -bn1 | head -5

# vmstat
vmstat 1 5

# mpstat (sysstat package)
mpstat 1 5
```

---

## 3. Memory Information

### free - Memory Usage

```bash
# Basic output
free

# Human-readable
free -h

# Detailed information
free -h --wide
```

Output:
```
              total        used        free      shared  buff/cache   available
Mem:          7.8Gi       3.2Gi       1.5Gi       256Mi       3.1Gi       4.0Gi
Swap:         2.0Gi          0B       2.0Gi
```

| Field | Description |
|-------|-------------|
| total | Total memory |
| used | Used |
| free | Unused |
| shared | Shared memory |
| buff/cache | Buffer/cache |
| available | Available (free + releasable cache) |

### /proc/meminfo

```bash
# Detailed memory information
cat /proc/meminfo

# Specific items
grep -E "MemTotal|MemFree|MemAvailable" /proc/meminfo
```

---

## 4. Disk Information

### df - Filesystem Usage

```bash
# Basic output
df

# Human-readable
df -h

# Filesystem type
df -Th

# Specific path
df -h /home
```

Output:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        50G   15G   33G  32% /
/dev/sda2       100G   45G   50G  48% /home
tmpfs           3.9G     0  3.9G   0% /dev/shm
```

### du - Directory Usage

```bash
# Directory size
du -sh /var/log

# By subdirectory
du -h --max-depth=1 /home

# Largest directories
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -10
```

### lsblk - Block Devices

```bash
lsblk
```

Output:
```
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sda      8:0    0  100G  0 disk
├─sda1   8:1    0   50G  0 part /
├─sda2   8:2    0   45G  0 part /home
└─sda3   8:3    0    5G  0 part [SWAP]
```

### fdisk - Partition Information

```bash
sudo fdisk -l
```

---

## 5. Log Management

### Main Log Files

| Log File | Content |
|----------|---------|
| `/var/log/syslog` | System log (Ubuntu) |
| `/var/log/messages` | System log (CentOS) |
| `/var/log/auth.log` | Authentication log (Ubuntu) |
| `/var/log/secure` | Authentication log (CentOS) |
| `/var/log/kern.log` | Kernel log |
| `/var/log/dmesg` | Boot messages |
| `/var/log/nginx/` | Nginx logs |
| `/var/log/apache2/` | Apache logs |

### Log Viewing

```bash
# System log (recent)
tail -100 /var/log/syslog

# Real-time monitoring
tail -f /var/log/syslog

# Search for errors
grep -i error /var/log/syslog | tail -20

# Monitor multiple logs simultaneously
tail -f /var/log/syslog /var/log/auth.log
```

### journalctl - systemd Logs

```bash
# All logs
journalctl

# Recent logs
journalctl -n 100

# Real-time
journalctl -f

# Specific service
journalctl -u nginx

# Today's logs
journalctl --since today

# Time range
journalctl --since "2024-01-23 00:00" --until "2024-01-23 12:00"

# Since boot
journalctl -b

# Errors only
journalctl -p err

# Kernel logs
journalctl -k
```

### dmesg - Kernel Messages

```bash
# Kernel messages
dmesg

# Recent messages
dmesg | tail -50

# Real-time
dmesg -w

# Human-readable
dmesg -H
```

---

## 6. Cron Jobs

### crontab Basics

```bash
# View current user crontab
crontab -l

# Edit crontab
crontab -e

# Other user's crontab (root)
sudo crontab -u username -l
```

### cron Format

```
* * * * * command
│ │ │ │ │
│ │ │ │ └── Day of week (0-7, 0 and 7 are Sunday)
│ │ │ └──── Month (1-12)
│ │ └────── Day (1-31)
│ └──────── Hour (0-23)
└────────── Minute (0-59)
```

### cron Examples

```bash
# Every minute
* * * * * /path/to/script.sh

# Every hour on the hour
0 * * * * /path/to/script.sh

# Daily at 2 AM
0 2 * * * /path/to/script.sh

# Every Monday at 3 AM
0 3 * * 1 /path/to/script.sh

# 1st of every month at midnight
0 0 1 * * /path/to/script.sh

# Every 5 minutes
*/5 * * * * /path/to/script.sh

# Weekdays at 9 AM
0 9 * * 1-5 /path/to/script.sh

# Multiple times
0 9,12,18 * * * /path/to/script.sh
```

### Practical cron Examples

```bash
# Backup (daily at 3 AM)
0 3 * * * /home/user/scripts/backup.sh >> /var/log/backup.log 2>&1

# Log cleanup (Sunday 4 AM)
0 4 * * 0 find /var/log -name "*.log" -mtime +30 -delete

# System update (Saturday 2 AM)
0 2 * * 6 apt update && apt upgrade -y

# Health check (every 10 minutes)
*/10 * * * * /home/user/scripts/health_check.sh
```

### System cron Directories

```
/etc/cron.d/        # cron configuration files
/etc/cron.daily/    # Daily execution
/etc/cron.hourly/   # Hourly execution
/etc/cron.weekly/   # Weekly execution
/etc/cron.monthly/  # Monthly execution
```

---

## 7. System Load

### uptime - Load Average

```bash
uptime
```

Output:
```
 10:30:00 up 15 days,  3:45,  2 users,  load average: 0.15, 0.10, 0.08
                                                         │     │     │
                                                         │     │     └── 15-min average
                                                         │     └── 5-min average
                                                         └── 1-min average
```

Load average interpretation:
- Lower than CPU core count: idle capacity
- Equal to CPU core count: fully utilized
- Higher than CPU core count: overloaded

### vmstat - Virtual Memory Statistics

```bash
# 1-second interval, 5 times
vmstat 1 5
```

Output:
```
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 1  0      0 1500000 200000 3000000   0    0     5    10  100  200  2  1 97  0  0
```

| Field | Description |
|-------|-------------|
| r | Runnable processes |
| b | Blocked processes |
| swpd | Used swap |
| si/so | Swap in/out |
| bi/bo | Block in/out |
| us | User CPU |
| sy | System CPU |
| id | Idle CPU |
| wa | I/O wait |

### iostat - I/O Statistics

```bash
# Install
sudo apt install sysstat    # Ubuntu
sudo dnf install sysstat    # CentOS

# Usage
iostat -x 1 5
```

---

## 8. Monitoring Scripts

### System Status Report

```bash
#!/bin/bash
# system_report.sh

echo "=== System Status Report ==="
echo "Date: $(date)"
echo

echo "=== System Information ==="
uname -a
echo

echo "=== CPU Usage ==="
top -bn1 | grep "Cpu(s)" | awk '{print "Used: " 100-$8 "%"}'
echo

echo "=== Memory ==="
free -h | grep Mem
echo

echo "=== Disk Usage ==="
df -h | grep -E "^/dev"
echo

echo "=== Load Average ==="
uptime
echo

echo "=== Network Connections ==="
ss -tuln | grep LISTEN | wc -l
echo "Listening ports"
```

### Disk Space Alert

```bash
#!/bin/bash
# disk_alert.sh

THRESHOLD=80

df -h | grep -E "^/dev" | while read line; do
    usage=$(echo "$line" | awk '{print $5}' | tr -d '%')
    mount=$(echo "$line" | awk '{print $6}')

    if [ "$usage" -gt "$THRESHOLD" ]; then
        echo "Warning: $mount usage ${usage}%"
        # Can add email notification here
    fi
done
```

---

## 9. Practice Exercises

### Exercise 1: Check System Information

```bash
# System information
uname -a
hostnamectl

# CPU information
lscpu | head -15

# Memory
free -h

# Disk
df -h
```

### Exercise 2: Log Analysis

```bash
# Check system log
sudo tail -50 /var/log/syslog

# Search for errors
sudo grep -i "error\|fail" /var/log/syslog | tail -20

# Check authentication log
sudo grep "Failed" /var/log/auth.log | tail -10
```

### Exercise 3: journalctl Usage

```bash
# Logs since boot
journalctl -b --no-pager | tail -50

# Today's errors
journalctl --since today -p err

# SSH service logs
journalctl -u sshd -n 20
```

### Exercise 4: cron Setup

```bash
# Edit crontab
crontab -e

# Add test job (log current time every minute)
# * * * * * date >> ~/cron_test.log

# Verify
crontab -l

# Check result after 1 minute
cat ~/cron_test.log
```

### Exercise 5: Resource Monitoring

```bash
# CPU load
uptime

# vmstat 5-second interval
vmstat 5 3

# Top CPU/memory processes in top
ps aux --sort=-%cpu | head -6
ps aux --sort=-%mem | head -6
```

---

**Previous**: [Network Basics](./10_Network_Basics.md) | **Next**: [Security and Firewall](./12_Security_and_Firewall.md)
