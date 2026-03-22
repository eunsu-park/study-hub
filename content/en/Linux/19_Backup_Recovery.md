# Backup and Recovery

**Previous**: [Log Management](./18_Log_Management.md) | **Next**: [Kernel Management](./20_Kernel_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a backup strategy using the 3-2-1 rule with appropriate RTO and RPO targets
2. Perform incremental backups with rsync using hard links for space efficiency
3. Configure Borg Backup repositories with encryption, compression, and automated pruning
4. Create system-level disk images using dd and LVM snapshots
5. Execute a bare-metal disaster recovery procedure from backup to bootable system
6. Automate backup workflows with systemd timers and implement monitoring alerts

---

## Table of Contents

1. [Backup Strategy Overview](#1-backup-strategy-overview)
2. [Advanced rsync Usage](#2-advanced-rsync-usage)
3. [Borg Backup](#3-borg-backup)
4. [tar/cpio Backup](#4-tarcpio-backup)
5. [System Image Backup](#5-system-image-backup)
6. [Disaster Recovery Strategy](#6-disaster-recovery-strategy)
7. [Automation and Monitoring](#7-automation-and-monitoring)

---

Every system will eventually face a failure -- whether from hardware defects, human error, ransomware, or natural disaster. The question is not *if* you will need to recover data, but *when*. A well-designed backup and recovery strategy is the insurance policy that keeps your business running. This lesson covers the tools, techniques, and planning principles that transform backup from a checkbox item into a battle-tested recovery capability.

## 1. Backup Strategy Overview

### 3-2-1 Backup Rule

```
┌─────────────────────────────────────────────────────────────┐
│                    3-2-1 Backup Rule                         │
├─────────────────────────────────────────────────────────────┤
│  3: Maintain 3 copies of data                               │
│     └── Original + 2 backups                                │
│                                                             │
│  2: On 2 different storage types                            │
│     └── Local disk + external disk or NAS                   │
│                                                             │
│  1: 1 copy offsite (remote location)                        │
│     └── Cloud or physical remote site                       │
└─────────────────────────────────────────────────────────────┘
```

### Backup Types

| Type | Description | Advantages | Disadvantages |
|------|-------------|------------|---------------|
| **Full Backup** | Copy all data | Simple recovery | Time/space intensive |
| **Incremental Backup** | Only changes since last backup | Fast, less space | Recovery requires chain |
| **Differential Backup** | Changes since last full backup | Easier recovery than incremental | More space than incremental |
| **Snapshot** | Filesystem state at specific point | Instant creation | Storage dependent |

### RTO and RPO

```
┌─────────────────────────────────────────────────────────────┐
│ RPO (Recovery Point Objective)                              │
│ = Acceptable data loss time                                 │
│ = How much data can we afford to lose since last backup?    │
│                                                             │
│ RTO (Recovery Time Objective)                               │
│ = Acceptable time to service recovery                       │
│ = How quickly must we recover after an incident?            │
└─────────────────────────────────────────────────────────────┘

Timeline:
──────────────────────────────────────────────────────────────
     Last Backup            Incident               Recovery
         │                    │                       │
         │◄─────── RPO ──────►│◄────── RTO ─────────►│
         │    (Data Loss)     │    (Downtime)        │
```

---

## 2. Advanced rsync Usage

### Basic Syntax

```bash
rsync [options] source destination

# Local copy
rsync -av /source/ /backup/

# Remote copy (SSH)
rsync -av /source/ user@server:/backup/
rsync -av user@server:/source/ /backup/
```

### Key Options

```bash
# Basic option combination
rsync -avz --progress /source/ /backup/

# Detailed options
-a, --archive       # Archive mode (same as -rlptgoD)
-v, --verbose       # Verbose output
-z, --compress      # Compress during transfer
-P                  # Combination of --progress --partial
--progress          # Show progress
--partial           # Keep partial files

# Delete options
--delete            # Delete files only in destination
--delete-before     # Delete before transfer
--delete-after      # Delete after transfer
--delete-excluded   # Also delete excluded files

# Synchronization precision options
-c, --checksum      # Compare by checksum (slow)
-u, --update        # Skip if destination is newer
--ignore-existing   # Skip existing files
```

### Exclusion Patterns

```bash
# Exclude specific patterns
rsync -av --exclude='*.log' --exclude='cache/' /source/ /backup/

# Use exclusion file
rsync -av --exclude-from='exclude.txt' /source/ /backup/
```

```bash
# exclude.txt example
*.log
*.tmp
*.cache
.git/
node_modules/
__pycache__/
.DS_Store
Thumbs.db
```

### Incremental Backup Script

```bash
#!/bin/bash
# incremental-backup.sh

# Configuration
SOURCE="/data"
BACKUP_BASE="/backup"
LATEST_LINK="$BACKUP_BASE/latest"
DATE=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_PATH="$BACKUP_BASE/$DATE"

# Use hard links if previous backup exists
if [ -d "$LATEST_LINK" ]; then
    LINK_DEST="--link-dest=$LATEST_LINK"
else
    LINK_DEST=""
fi

# Run rsync
rsync -av --delete \
    $LINK_DEST \
    --exclude='*.tmp' \
    --exclude='cache/' \
    "$SOURCE/" \
    "$BACKUP_PATH/"

# Update latest link
rm -f "$LATEST_LINK"
ln -s "$BACKUP_PATH" "$LATEST_LINK"

# Delete backups older than 30 days
find "$BACKUP_BASE" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_PATH"
```

### SSH Key Setup (for Remote Backup)

```bash
# Generate backup-dedicated key
ssh-keygen -t ed25519 -f ~/.ssh/backup_key -N ""

# Copy key to remote server
ssh-copy-id -i ~/.ssh/backup_key.pub user@backup-server

# Config settings for automation
cat >> ~/.ssh/config << EOF
Host backup-server
    HostName 192.168.1.100
    User backupuser
    IdentityFile ~/.ssh/backup_key
    StrictHostKeyChecking no
EOF

# Run remote backup
rsync -avz -e "ssh -i ~/.ssh/backup_key" /data/ backup-server:/backup/
```

### Bandwidth Limiting

```bash
# Limit to 10MB/s
rsync -av --bwlimit=10000 /source/ /backup/

# Fast speed only outside business hours
if [ $(date +%H) -ge 18 ] || [ $(date +%H) -lt 8 ]; then
    BWLIMIT=""
else
    BWLIMIT="--bwlimit=5000"
fi
rsync -av $BWLIMIT /source/ /backup/
```

---

## 3. Borg Backup

### Borg Introduction

Borg Backup is a backup program supporting deduplication, compression, and encryption.

```bash
# Installation
# Ubuntu/Debian
sudo apt install borgbackup

# RHEL/CentOS
sudo yum install epel-release
sudo yum install borgbackup

# Install via pip
pip install borgbackup
```

### Repository Initialization

```bash
# Create local repository
borg init --encryption=repokey /backup/borg-repo

# Create remote repository
borg init --encryption=repokey user@server:/backup/borg-repo

# Encryption options
# none       - No encryption
# repokey    - Store key in repository (recommended)
# keyfile    - Store key in local file
# repokey-blake2 - Faster hash
```

### Creating Backups

```bash
# Basic backup
borg create /backup/borg-repo::backup-{now} /data

# With options
borg create \
    --verbose \
    --progress \
    --stats \
    --compression lz4 \
    --exclude '*.tmp' \
    --exclude 'cache/' \
    /backup/borg-repo::backup-{now:%Y-%m-%d_%H-%M} \
    /home \
    /etc \
    /var/www
```

### Compression Options

| Option | Description | Speed | Compression Ratio |
|--------|-------------|-------|-------------------|
| `none` | No compression | Fastest | None |
| `lz4` | Fast compression | Fast | Low |
| `zstd` | Balanced compression | Medium | Medium |
| `zlib` | gzip compatible | Slow | High |
| `lzma` | Maximum compression | Very slow | Highest |

### Backup Management

```bash
# List backups
borg list /backup/borg-repo

# Backup details
borg info /backup/borg-repo::backup-2024-01-15

# View backup contents
borg list /backup/borg-repo::backup-2024-01-15

# View specific path only
borg list /backup/borg-repo::backup-2024-01-15 /home/user/

# Compare backups
borg diff /backup/borg-repo::backup-2024-01-14 backup-2024-01-15
```

### Recovery

```bash
# Full recovery
cd /restore
borg extract /backup/borg-repo::backup-2024-01-15

# Recover specific file/directory
borg extract /backup/borg-repo::backup-2024-01-15 home/user/documents

# Recover to original path
cd /
borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx/

# Mount specific point in time (FUSE)
mkdir /mnt/borg
borg mount /backup/borg-repo::backup-2024-01-15 /mnt/borg
# After browsing files
borg umount /mnt/borg
```

### Retention Policy (Pruning)

```bash
# Automatic cleanup
borg prune \
    --keep-hourly=24 \
    --keep-daily=7 \
    --keep-weekly=4 \
    --keep-monthly=12 \
    --keep-yearly=2 \
    /backup/borg-repo

# Dry run (no actual deletion)
borg prune --dry-run --list \
    --keep-daily=7 \
    /backup/borg-repo
```

### Borg Backup Script

```bash
#!/bin/bash
# borg-backup.sh

# Environment variable setup
export BORG_REPO="user@backup-server:/backup/borg-repo"
export BORG_PASSPHRASE="your-secure-passphrase"

# Log file
LOG_FILE="/var/log/borg-backup.log"

# Backup function
backup() {
    echo "Starting backup: $(date)" >> "$LOG_FILE"

    borg create \
        --verbose \
        --filter AME \
        --list \
        --stats \
        --compression lz4 \
        --exclude-caches \
        --exclude '/home/*/.cache' \
        --exclude '/var/tmp/*' \
        --exclude '/var/cache/*' \
        ::'{hostname}-{now:%Y-%m-%d_%H:%M}' \
        /etc \
        /home \
        /var/www \
        /var/lib/mysql \
        2>> "$LOG_FILE"

    backup_exit=$?

    echo "Backup finished with exit code: $backup_exit" >> "$LOG_FILE"
}

# Prune function
prune() {
    echo "Starting prune: $(date)" >> "$LOG_FILE"

    borg prune \
        --list \
        --keep-hourly=24 \
        --keep-daily=7 \
        --keep-weekly=4 \
        --keep-monthly=6 \
        2>> "$LOG_FILE"

    echo "Prune finished" >> "$LOG_FILE"
}

# Repository integrity check (weekly)
check() {
    if [ $(date +%u) -eq 7 ]; then
        echo "Starting check: $(date)" >> "$LOG_FILE"
        borg check 2>> "$LOG_FILE"
        echo "Check finished" >> "$LOG_FILE"
    fi
}

# Execute
backup
prune
check

# Alert on failure
if [ $backup_exit -ne 0 ]; then
    echo "Backup failed!" | mail -s "Borg Backup Alert" admin@example.com
fi
```

---

## 4. tar/cpio Backup

### tar Backup

```bash
# Basic compressed backup
tar -czvf backup.tar.gz /data

# Incremental backup (using snapshot)
tar --create \
    --gzip \
    --listed-incremental=/backup/snapshot.snar \
    --file=/backup/backup-$(date +%Y%m%d).tar.gz \
    /data

# Restore
tar --extract \
    --gzip \
    --listed-incremental=/dev/null \
    --file=/backup/backup-20240115.tar.gz \
    -C /restore

# Exclusion patterns
tar -czvf backup.tar.gz \
    --exclude='*.log' \
    --exclude='cache' \
    /data
```

### cpio Backup

```bash
# Create backup
find /data -print | cpio -ov > backup.cpio

# Compressed backup
find /data -print | cpio -ov | gzip > backup.cpio.gz

# Restore
cpio -iv < backup.cpio

# Restore compressed file
gunzip -c backup.cpio.gz | cpio -iv

# Restore specific files only
cpio -iv "*.conf" < backup.cpio
```

---

## 5. System Image Backup

### Disk Image with dd

```bash
# Full disk backup
sudo dd if=/dev/sda of=/backup/disk.img bs=4M status=progress

# With compression
sudo dd if=/dev/sda bs=4M status=progress | gzip > /backup/disk.img.gz

# Restore
sudo dd if=/backup/disk.img of=/dev/sda bs=4M status=progress

# Restore compressed image
gunzip -c /backup/disk.img.gz | sudo dd of=/dev/sda bs=4M status=progress

# Backup partition only
sudo dd if=/dev/sda1 of=/backup/partition.img bs=4M status=progress
```

### Clonezilla

```bash
# After creating Clonezilla Live USB

# Create disk image (command line)
/usr/sbin/ocs-sr -q2 -c -j2 -z1 -i 4096 -sfsck -senc -p true \
    savedisk img_name sda

# Restore
/usr/sbin/ocs-sr -g auto -e1 auto -e2 -r -j2 -c -scr -p true \
    restoredisk img_name sda
```

### LVM Snapshot Backup

```bash
# Create snapshot
sudo lvcreate -L 10G -s -n data-snap /dev/vg0/data

# Mount snapshot
sudo mkdir /mnt/snapshot
sudo mount -o ro /dev/vg0/data-snap /mnt/snapshot

# Perform backup
rsync -av /mnt/snapshot/ /backup/data/

# Cleanup
sudo umount /mnt/snapshot
sudo lvremove /dev/vg0/data-snap
```

---

## 6. Disaster Recovery Strategy

### DR Plan Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Disaster Recovery Plan                    │
├─────────────────────────────────────────────────────────────┤
│  1. Risk Assessment                                         │
│     - Identify potential threats                            │
│     - Business impact analysis                              │
│                                                             │
│  2. Recovery Objectives                                     │
│     - Define RTO/RPO                                        │
│     - Determine priorities                                  │
│                                                             │
│  3. Backup Strategy                                         │
│     - Backup types and frequency                            │
│     - Storage locations (onsite/offsite)                    │
│                                                             │
│  4. Recovery Procedures                                     │
│     - Step-by-step recovery guide                           │
│     - Contacts and roles                                    │
│                                                             │
│  5. Testing and Maintenance                                 │
│     - Regular recovery testing                              │
│     - Documentation updates                                 │
└─────────────────────────────────────────────────────────────┘
```

### Recovery Checklist

```bash
#!/bin/bash
# disaster-recovery-checklist.sh

echo "=== Disaster Recovery Checklist ==="

# 1. Hardware status check
echo "[1] Hardware Status Check"
lsblk
free -h
cat /proc/cpuinfo | grep "model name" | head -1

# 2. Network connectivity check
echo "[2] Network Connectivity"
ip addr show
ping -c 3 8.8.8.8

# 3. Backup storage access check
echo "[3] Backup Storage Access"
# Local backup
ls -la /backup/
# Remote backup
ssh backup-server "ls -la /backup/"

# 4. Backup integrity verification
echo "[4] Backup Integrity"
# Borg verification
borg check /backup/borg-repo

# 5. Recovery test (sample)
echo "[5] Sample File Recovery Test"
mkdir -p /tmp/recovery-test
borg extract /backup/borg-repo::latest etc/hostname -C /tmp/recovery-test
diff /etc/hostname /tmp/recovery-test/etc/hostname

echo "=== Checklist Complete ==="
```

### Bare Metal Recovery Procedure

```bash
# 1. Boot from recovery media (Ubuntu Live USB, etc.)

# 2. Network configuration
ip addr add 192.168.1.100/24 dev eth0
ip route add default via 192.168.1.1

# 3. Disk partitioning
parted /dev/sda mklabel gpt
parted /dev/sda mkpart primary ext4 1MiB 512MiB    # /boot
parted /dev/sda mkpart primary ext4 512MiB 100%    # /
mkfs.ext4 /dev/sda1
mkfs.ext4 /dev/sda2

# 4. Mount
mount /dev/sda2 /mnt
mkdir /mnt/boot
mount /dev/sda1 /mnt/boot

# 5. Restore from backup
# Using rsync
rsync -av backup-server:/backup/latest/ /mnt/

# Or using Borg
borg extract backup-server:/backup/borg-repo::latest -C /mnt

# 6. Install bootloader via chroot
mount --bind /dev /mnt/dev
mount --bind /proc /mnt/proc
mount --bind /sys /mnt/sys
chroot /mnt

grub-install /dev/sda
update-grub

exit

# 7. Cleanup and reboot
umount -R /mnt
reboot
```

---

## 7. Automation and Monitoring

### Automation with systemd Timer

```bash
# /etc/systemd/system/backup.service
[Unit]
Description=Daily Backup Service
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
User=root
Nice=19
IOSchedulingClass=idle

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/systemd/system/backup.timer
[Unit]
Description=Run backup daily

[Timer]
OnCalendar=*-*-* 02:00:00
RandomizedDelaySec=1800
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
# Enable timer
sudo systemctl enable --now backup.timer

# Check status
systemctl list-timers --all | grep backup
```

### Backup Monitoring Script

```bash
#!/bin/bash
# backup-monitor.sh

BACKUP_DIR="/backup"
MAX_AGE_HOURS=26
ALERT_EMAIL="admin@example.com"
LOGFILE="/var/log/backup-monitor.log"

check_backup_age() {
    local newest=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" | sort -r | head -1)

    if [ -z "$newest" ]; then
        echo "ERROR: No backup found" | tee -a "$LOGFILE"
        return 1
    fi

    local age_seconds=$(($(date +%s) - $(stat -c %Y "$newest")))
    local age_hours=$((age_seconds / 3600))

    if [ $age_hours -gt $MAX_AGE_HOURS ]; then
        echo "WARNING: Latest backup is $age_hours hours old" | tee -a "$LOGFILE"
        return 1
    fi

    echo "OK: Latest backup is $age_hours hours old ($newest)" | tee -a "$LOGFILE"
    return 0
}

check_backup_size() {
    local today=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "$(date +%Y-%m-%d)*" | head -1)
    local yesterday=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "$(date -d yesterday +%Y-%m-%d)*" | head -1)

    if [ -n "$today" ] && [ -n "$yesterday" ]; then
        local size_today=$(du -s "$today" | awk '{print $1}')
        local size_yesterday=$(du -s "$yesterday" | awk '{print $1}')

        # Warn if difference exceeds 50%
        local diff=$((size_today - size_yesterday))
        local threshold=$((size_yesterday / 2))

        if [ ${diff#-} -gt $threshold ]; then
            echo "WARNING: Significant size change: $size_yesterday -> $size_today" | tee -a "$LOGFILE"
            return 1
        fi
    fi

    return 0
}

check_disk_space() {
    local usage=$(df "$BACKUP_DIR" | awk 'NR==2 {print $5}' | tr -d '%')

    if [ $usage -gt 90 ]; then
        echo "CRITICAL: Backup disk usage at ${usage}%" | tee -a "$LOGFILE"
        return 1
    elif [ $usage -gt 80 ]; then
        echo "WARNING: Backup disk usage at ${usage}%" | tee -a "$LOGFILE"
        return 1
    fi

    echo "OK: Backup disk usage at ${usage}%" | tee -a "$LOGFILE"
    return 0
}

# Main execution
echo "=== Backup Monitor: $(date) ===" >> "$LOGFILE"
ERRORS=0

check_backup_age || ((ERRORS++))
check_backup_size || ((ERRORS++))
check_disk_space || ((ERRORS++))

if [ $ERRORS -gt 0 ]; then
    tail -20 "$LOGFILE" | mail -s "Backup Monitor Alert" "$ALERT_EMAIL"
fi

exit $ERRORS
```

### Prometheus Metrics Collection

```bash
#!/bin/bash
# backup-metrics.sh (for node_exporter textfile collector)

METRICS_FILE="/var/lib/node_exporter/textfile_collector/backup.prom"
BACKUP_DIR="/backup"

# Latest backup time
newest=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" | sort -r | head -1)
if [ -n "$newest" ]; then
    backup_timestamp=$(stat -c %Y "$newest")
    echo "backup_last_success_timestamp $backup_timestamp" > "$METRICS_FILE"
fi

# Backup size
backup_size=$(du -sb "$newest" 2>/dev/null | awk '{print $1}')
echo "backup_size_bytes $backup_size" >> "$METRICS_FILE"

# Disk usage
disk_usage=$(df "$BACKUP_DIR" | awk 'NR==2 {print $3}')
disk_total=$(df "$BACKUP_DIR" | awk 'NR==2 {print $2}')
echo "backup_disk_used_bytes $((disk_usage * 1024))" >> "$METRICS_FILE"
echo "backup_disk_total_bytes $((disk_total * 1024))" >> "$METRICS_FILE"
```

---

## Practice Problems

### Problem 1: rsync Incremental Backup

Write an rsync incremental backup script using hard links:
- Source: `/home/user`
- Backup location: `/backup/home`
- Daily backup, link latest backup as `latest` symlink
- Auto-delete backups older than 30 days

### Problem 2: Borg Recovery

Write commands to recover only the `/etc/nginx/` directory from a specific date's backup in a Borg repository.

### Problem 3: DR Testing

Write a quarterly disaster recovery testing procedure. Include:
- Backup integrity verification
- Sample data recovery test
- Full system recovery test (if possible)

---

## Answers

### Problem 1 Answer

```bash
#!/bin/bash

SOURCE="/home/user"
BACKUP_BASE="/backup/home"
DATE=$(date +%Y-%m-%d)
BACKUP_PATH="$BACKUP_BASE/$DATE"
LATEST="$BACKUP_BASE/latest"

# Hard link option
if [ -d "$LATEST" ]; then
    LINK="--link-dest=$LATEST"
else
    LINK=""
fi

# Execute backup
rsync -av --delete $LINK "$SOURCE/" "$BACKUP_PATH/"

# Update latest link
rm -f "$LATEST"
ln -s "$BACKUP_PATH" "$LATEST"

# Delete backups older than 30 days
find "$BACKUP_BASE" -maxdepth 1 -type d -name "20*" -mtime +30 -exec rm -rf {} \;
```

### Problem 2 Answer

```bash
# Check backup list
borg list /backup/borg-repo

# Recover nginx config from specific date backup
borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx

# Recover to different path
mkdir /tmp/restore
cd /tmp/restore
borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx
```

### Problem 3 Answer

```markdown
# Quarterly DR Testing Procedure

## 1. Backup Integrity Verification (Every Quarter)
- [ ] Run Borg check: `borg check /backup/borg-repo`
- [ ] Review backup list: `borg list /backup/borg-repo`
- [ ] Check recent backup details: `borg info /backup/borg-repo::latest`

## 2. Sample Data Recovery Test (Every Quarter)
- [ ] Create test directory
- [ ] Test config file recovery (/etc/)
- [ ] Test data file recovery (/var/www/)
- [ ] Verify recovered file integrity (diff or checksum)

## 3. Full System Recovery Test (Semi-annually)
- [ ] Prepare test VM or physical server
- [ ] Execute bare metal recovery procedure
- [ ] Verify boot
- [ ] Verify service normal operation
- [ ] Verify data integrity

## 4. Documentation and Improvement
- [ ] Document test results
- [ ] Record discovered issues
- [ ] Apply procedure improvements
- [ ] Verify RTO/RPO achievement
```

---

---

## References

- [rsync Manual](https://rsync.samba.org/documentation.html)
- [Borg Backup Documentation](https://borgbackup.readthedocs.io/)
- [Clonezilla](https://clonezilla.org/)
- `man rsync`, `man tar`, `man dd`, `man borg`

---

**Previous**: [Log Management](./18_Log_Management.md) | **Next**: [Kernel Management](./20_Kernel_Management.md)
