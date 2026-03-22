# Troubleshooting Guide

**Previous**: [High Availability Cluster](./25_High_Availability_Cluster.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply a systematic five-step methodology for diagnosing Linux system problems
2. Recover from GRUB, filesystem, and systemd boot failures using recovery mode
3. Diagnose network connectivity issues step-by-step from interface to DNS to firewall
4. Identify and resolve disk space exhaustion, filesystem corruption, and I/O bottlenecks
5. Detect OOM killer events and diagnose memory leaks in running processes
6. Debug process issues including zombies, high CPU usage, and file descriptor exhaustion
7. Build performance baselines and use bottleneck analysis checklists for production systems

---

## Table of Contents

1. [Problem Solving Methodology](#1-problem-solving-methodology)
2. [Boot Issues](#2-boot-issues)
3. [Network Issues](#3-network-issues)
4. [Disk Issues](#4-disk-issues)
5. [Memory Issues](#5-memory-issues)
6. [Process Issues](#6-process-issues)
7. [Performance Analysis](#7-performance-analysis)

---

When a production server goes down at 3 AM, there is no time for trial and error. Effective troubleshooting requires a structured methodology: gather evidence, form hypotheses, test them, and document the resolution. This lesson compiles the diagnostic techniques from the entire Linux series into a practical reference that you can use under pressure to resolve boot failures, network outages, disk problems, memory exhaustion, and performance degradation.

## 1. Problem Solving Methodology

### Systematic Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    Problem Solving Process                   │
│                                                             │
│  1. Problem Definition                                       │
│     └── What are the symptoms?                              │
│     └── When did it start?                                  │
│     └── What changes were made?                             │
│                                                             │
│  2. Information Gathering                                    │
│     └── Check logs                                          │
│     └── Check system status                                 │
│     └── Review configuration                                │
│                                                             │
│  3. Hypothesis Formation                                     │
│     └── List possible causes                                │
│     └── Prioritize                                          │
│                                                             │
│  4. Testing and Verification                                 │
│     └── Test hypothesis                                     │
│     └── Verify results                                      │
│                                                             │
│  5. Resolution and Documentation                             │
│     └── Apply fix                                           │
│     └── Establish prevention measures                       │
│     └── Documentation                                       │
└─────────────────────────────────────────────────────────────┘
```

### Basic Diagnostic Commands

```bash
# System overview
uptime                    # Uptime, load average
uname -a                  # Kernel version
hostnamectl               # Host information
dmidecode -t system       # Hardware information

# Resource overview
free -h                   # Memory
df -h                     # Disk
top -bn1 | head -20       # Top CPU/memory processes

# Recent logs
journalctl -p err -since "1 hour ago"
dmesg | tail -50

# Service status
systemctl --failed
systemctl status <service>
```

### Log Checking Priority

```bash
# 1. System logs
journalctl -xe                    # Recent errors
journalctl -b                     # Current boot
journalctl -p err --since today   # Today's errors

# 2. Kernel messages
dmesg --level=err,warn
dmesg -T | tail -100

# 3. Service-specific logs
journalctl -u nginx -f
journalctl -u postgresql --since "1 hour ago"

# 4. Application logs
tail -f /var/log/nginx/error.log
tail -f /var/log/syslog
```

---

## 2. Boot Issues

### Boot Process

```
BIOS/UEFI → GRUB → Kernel → systemd → Services → Login
    │          │       │        │          │
    └──────────┴───────┴────────┴──────────┴── Failure possible at each stage
```

### GRUB Recovery

```bash
# Press 'e' in GRUB menu to edit
# Add to kernel line:
linux /vmlinuz... root=... single    # Single user mode
linux /vmlinuz... root=... init=/bin/bash  # Direct shell

# Reinstall GRUB (in recovery mode)
mount /dev/sda2 /mnt
mount /dev/sda1 /mnt/boot
mount --bind /dev /mnt/dev
mount --bind /proc /mnt/proc
mount --bind /sys /mnt/sys
chroot /mnt
grub-install /dev/sda
update-grub
exit
reboot
```

### Filesystem Recovery

```bash
# Run fsck (on unmounted filesystem)
fsck -y /dev/sda1

# Check root filesystem from live environment
# 1. Boot to recovery mode or Live USB
# 2. Unmount filesystem then check
umount /dev/sda2
fsck -y /dev/sda2

# XFS recovery
xfs_repair /dev/sda2

# ext4 superblock recovery
mke2fs -n /dev/sda2  # Find backup superblock locations
e2fsck -b 32768 /dev/sda2  # Recover with backup superblock
```

### systemd Boot Issues

```bash
# Boot analysis
systemd-analyze
systemd-analyze blame
systemd-analyze critical-chain

# Check failed units
systemctl --failed
systemctl reset-failed

# Debug specific service
systemctl status nginx.service
journalctl -u nginx.service -b

# Boot to emergency mode
# Add to kernel line in GRUB:
systemd.unit=emergency.target
# Or
systemd.unit=rescue.target
```

### Password Reset

```bash
# 1. Press 'e' in GRUB to edit
# 2. Add to end of linux line: init=/bin/bash
# 3. Press Ctrl+X to boot

# Remount root filesystem as read-write
mount -o remount,rw /

# Change password
passwd root
passwd username

# SELinux relabeling (RHEL/CentOS)
touch /.autorelabel

# Reboot
exec /sbin/init
# Or
reboot -f
```

---

## 3. Network Issues

### Step-by-Step Diagnosis

```bash
# 1. Interface status
ip link show
ip addr show
ethtool eth0

# 2. Routing table
ip route show
ip route get 8.8.8.8

# 3. DNS verification
cat /etc/resolv.conf
nslookup google.com
dig google.com

# 4. Connectivity test
ping -c 3 8.8.8.8           # IP connectivity
ping -c 3 google.com        # DNS resolution + connectivity
traceroute google.com       # Path tracing

# 5. Port check
ss -tlnp                    # Listening ports
ss -tnp                     # Connected sockets
netstat -anp                # Full status
```

### Connection Problem Diagnosis

```bash
# Test specific port connection
nc -zv 192.168.1.100 80
telnet 192.168.1.100 80

# TCP connection status
ss -tn state established
ss -tn state time-wait | wc -l

# Packet capture
tcpdump -i eth0 port 80
tcpdump -i eth0 host 192.168.1.100
tcpdump -i eth0 -w capture.pcap

# MTU issue check
ping -M do -s 1472 192.168.1.1  # 1500 - 28 = 1472
```

### Firewall Issues

```bash
# Check iptables
iptables -L -n -v
iptables -t nat -L -n -v

# Check nftables
nft list ruleset

# Check firewalld (RHEL/CentOS)
firewall-cmd --list-all
firewall-cmd --get-active-zones

# Check UFW (Ubuntu)
ufw status verbose

# Temporarily disable firewall (for testing)
systemctl stop firewalld
iptables -F
```

### DNS Issues

```bash
# DNS resolution test
nslookup example.com
dig example.com
host example.com

# Use specific DNS server
nslookup example.com 8.8.8.8
dig @8.8.8.8 example.com

# DNS cache check/flush
systemd-resolve --statistics
systemd-resolve --flush-caches
# Or
resolvectl flush-caches

# Check /etc/hosts
cat /etc/hosts

# Check nsswitch configuration
cat /etc/nsswitch.conf | grep hosts
```

---

## 4. Disk Issues

### Checking Disk Status

```bash
# Disk usage
df -h
df -i                       # Inode usage

# Partition check
lsblk
fdisk -l
parted -l

# Disk health (SMART)
smartctl -H /dev/sda
smartctl -a /dev/sda

# Disk I/O statistics
iostat -xz 1
iotop -o
```

### Disk Space Issues

```bash
# Find large files
find / -xdev -type f -size +100M -exec ls -lh {} \; 2>/dev/null

# Directory-wise usage
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
du -sh /var/log/*

# Deleted files still holding space (open file handles)
lsof | grep deleted
lsof +L1

# Log file cleanup
journalctl --vacuum-size=100M
find /var/log -name "*.gz" -mtime +30 -delete

# Inode issue (too many files)
find / -xdev -type d -exec sh -c 'echo "$(find "{}" -maxdepth 1 | wc -l) {}"' \; | sort -rn | head
```

### Filesystem Recovery

```bash
# Check read-only mode
mount | grep ' / '

# Remount as read-write
mount -o remount,rw /

# Filesystem errors
dmesg | grep -i "error\|fail\|corrupt"

# Force fsck
touch /forcefsck
reboot

# Or from recovery mode
fsck -y /dev/sda1
```

### LVM Issues

```bash
# Check LVM status
pvs; vgs; lvs
pvdisplay; vgdisplay; lvdisplay

# LVM metadata recovery
vgcfgrestore -l vg_name              # List backups
vgcfgrestore -f /etc/lvm/archive/... vg_name

# Activate VG
vgchange -ay vg_name

# Activate LV
lvchange -ay /dev/vg_name/lv_name
```

---

## 5. Memory Issues

### Checking Memory Status

```bash
# Memory overview
free -h
cat /proc/meminfo

# Per-process memory
ps aux --sort=-%mem | head -20
top -o %MEM

# Swap usage
swapon -s
cat /proc/swaps

# Detailed memory usage
smem -t -k
pmap -x <PID>
```

### OOM Killer Diagnosis

```bash
# Check OOM occurrence
dmesg | grep -i "out of memory"
journalctl -k | grep -i "oom"

# Check OOM score
cat /proc/<PID>/oom_score
cat /proc/<PID>/oom_score_adj

# Adjust OOM score (protect)
echo -1000 > /proc/<PID>/oom_score_adj

# Or in systemd service
# [Service]
# OOMScoreAdjust=-500
```

### Memory Leak Diagnosis

```bash
# Track process memory
while true; do
    ps -o pid,vsz,rss,comm -p <PID>
    sleep 60
done

# Valgrind (development environment)
valgrind --leak-check=full ./myapp

# Check USS/PSS with smem
smem -P nginx
```

### Cache/Buffer Cleanup

```bash
# Cache status
cat /proc/meminfo | grep -E "Cached|Buffers|SReclaimable"

# Clear cache (caution in production)
sync
echo 1 > /proc/sys/vm/drop_caches  # Page cache
echo 2 > /proc/sys/vm/drop_caches  # dentries, inodes
echo 3 > /proc/sys/vm/drop_caches  # All

# Swap cleanup (when memory is available)
swapoff -a && swapon -a
```

---

## 6. Process Issues

### Checking Process Status

```bash
# Process list
ps aux
ps -ef
ps auxf  # Tree format

# Find specific process
pgrep -a nginx
pidof nginx

# Process status
cat /proc/<PID>/status
cat /proc/<PID>/limits

# Process environment variables
cat /proc/<PID>/environ | tr '\0' '\n'
```

### Zombie/Orphan Processes

```bash
# Find zombie processes
ps aux | awk '$8=="Z"'

# Find zombie's parent process
ps -ef | grep <ZOMBIE_PID>

# Check parent process
cat /proc/<ZOMBIE_PID>/status | grep PPid

# Remove zombie (terminate parent)
kill -SIGCHLD <PARENT_PID>
# Or restart parent process
```

### strace/lsof Debugging

```bash
# Trace system calls
strace -p <PID>
strace -p <PID> -e open,read,write
strace -f -p <PID>  # Include child processes

# Check open files
lsof -p <PID>
lsof -c nginx
lsof -i :80
lsof +D /var/log

# Check file descriptors
ls -la /proc/<PID>/fd
cat /proc/<PID>/limits | grep "open files"
```

### Service Issues

```bash
# Service status
systemctl status nginx

# Service logs
journalctl -u nginx -f
journalctl -u nginx --since "10 minutes ago"

# Service restart (detailed)
systemctl restart nginx
systemctl daemon-reload && systemctl restart nginx

# Check service configuration
systemctl cat nginx
systemctl show nginx
```

---

## 7. Performance Analysis

### System Overview

```bash
# Combined status
vmstat 1 10
mpstat -P ALL 1 5
iostat -xz 1 5

# Load average interpretation
# load average: 1.00, 0.75, 0.50
# 1-minute, 5-minute, 15-minute average
# Compare with CPU core count (1.0 = 100% utilization)
nproc  # CPU core count
```

### CPU Analysis

```bash
# CPU usage
top -bn1 | head -20
htop

# Per-process CPU
pidstat 1 5
ps aux --sort=-%cpu | head -10

# Detailed CPU info
mpstat -P ALL 1

# Hotspot detection (perf)
perf top
perf record -g -p <PID> -- sleep 30
perf report
```

### I/O Analysis

```bash
# Disk I/O
iostat -xz 1
iotop -o

# Per-process I/O
pidstat -d 1

# Wait time check
await, svctm in iostat output
# await > 10ms: Slow disk
# %util > 80%: Possible bottleneck

# I/O profiling
blktrace -d /dev/sda -o - | blkparse -i -
```

### Network Analysis

```bash
# Network statistics
netstat -s
ss -s

# Bandwidth monitoring
iftop -i eth0
nethogs eth0

# Connection status
ss -tn state established | wc -l
ss -tn state time-wait | wc -l

# Packet loss check
netstat -s | grep -i "packet loss\|retrans"
```

### Bottleneck Analysis Checklist

```bash
#!/bin/bash
# bottleneck-check.sh

echo "=== System Overview ==="
uptime
echo

echo "=== CPU ==="
mpstat 1 3 | tail -4
echo

echo "=== Memory ==="
free -h
echo

echo "=== Disk I/O ==="
iostat -xz 1 3 | tail -10
echo

echo "=== Network ==="
ss -s
echo

echo "=== Top Processes (CPU) ==="
ps aux --sort=-%cpu | head -6
echo

echo "=== Top Processes (Memory) ==="
ps aux --sort=-%mem | head -6
echo

echo "=== Failed Services ==="
systemctl --failed
echo

echo "=== Recent Errors ==="
journalctl -p err --since "1 hour ago" | tail -20
```

### Performance Baseline

```bash
# Record normal state (regularly)
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M)
OUTPUT_DIR=/var/log/baseline

mkdir -p $OUTPUT_DIR

# System information
vmstat 1 60 > $OUTPUT_DIR/vmstat-$DATE.log &
iostat -xz 1 60 > $OUTPUT_DIR/iostat-$DATE.log &
mpstat -P ALL 1 60 > $OUTPUT_DIR/mpstat-$DATE.log &
sar -n DEV 1 60 > $OUTPUT_DIR/sar-net-$DATE.log &

wait

# Snapshots
ps aux > $OUTPUT_DIR/ps-$DATE.log
free -h > $OUTPUT_DIR/memory-$DATE.log
df -h > $OUTPUT_DIR/disk-$DATE.log
ss -s > $OUTPUT_DIR/network-$DATE.log
```

---

## Practice Problems

### Problem 1: Boot Issue

The system boots to emergency mode. Explain the steps to find and resolve the cause.

### Problem 2: Disk Space

The /var partition is 100% full. Write commands to find the cause and resolve it.

### Problem 3: Network Connection

Cannot access external websites. Write step-by-step diagnostic procedures.

---

## Answers

### Problem 1 Answer

```bash
# 1. Check error messages
journalctl -xb
dmesg | grep -i error

# 2. Common causes
# - /etc/fstab errors
# - Filesystem corruption
# - SELinux issues

# 3. Check /etc/fstab
cat /etc/fstab
# Verify UUID and devices are correct

# 4. Check filesystem
fsck -y /dev/sda1

# 5. Fix fstab (if problematic)
# Add nofail option or comment out problematic entries

# 6. Reboot
reboot
```

### Problem 2 Answer

```bash
# 1. Check overall usage
df -h /var

# 2. Find large directories
du -h --max-depth=1 /var | sort -hr | head -10

# 3. Find large files
find /var -type f -size +100M -exec ls -lh {} \;

# 4. Check common causes
du -sh /var/log
du -sh /var/cache
du -sh /var/lib/docker  # If using Docker

# 5. Clean up logs
journalctl --vacuum-size=100M
find /var/log -name "*.gz" -mtime +7 -delete
truncate -s 0 /var/log/large-file.log

# 6. Check deleted file handles
lsof +L1 | grep /var
# Restart service to release handles
```

### Problem 3 Answer

```bash
# 1. Interface status
ip addr show
ip link show

# 2. Check default gateway
ip route show
ping -c 3 <gateway-ip>

# 3. External IP connectivity
ping -c 3 8.8.8.8

# 4. DNS check (if IP works but domain doesn't)
nslookup google.com
cat /etc/resolv.conf

# 5. Firewall check
iptables -L -n
firewall-cmd --list-all

# 6. Test specific port
nc -zv google.com 443

# 7. Check routing path
traceroute google.com

# Actions based on diagnosis results:
# - No IP: DHCP or manual IP configuration
# - Gateway unreachable: Check network cable/switch
# - External IP unreachable: Check router/firewall
# - Only DNS failing: Fix resolv.conf
```

---

## References

- [Linux Performance](http://www.brendangregg.com/linuxperf.html)
- [Red Hat System Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/system_administrators_guide/index)
- [Ubuntu Server Guide](https://ubuntu.com/server/docs)
- `man strace`, `man lsof`, `man perf`

---

## Conclusion

This document concludes the Linux learning series.

Complete Learning Content:
- 01-03: Linux Basics
- 04-08: Intermediate Administration
- 09-12: Advanced Server Management
- 13-16: Advanced Topics (systemd, Performance, Containers, Storage)
- 17-26: Expert Level (Security, Virtualization, Automation, HA, Troubleshooting)

Return to [00_Overview.md](./00_Overview.md) to review the complete learning roadmap.

---

**Previous**: [High Availability Cluster](./25_High_Availability_Cluster.md)
