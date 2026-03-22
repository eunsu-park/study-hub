# 16. Storage Management

**Previous**: [Container Internals](./15_Container_Internals.md) | **Next**: [SELinux and AppArmor](./17_SELinux_AppArmor.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and manage LVM physical volumes, volume groups, and logical volumes
2. Extend and shrink LVM volumes and use LVM snapshots for consistent backups
3. Compare RAID levels (0, 1, 5, 6, 10) and configure software RAID with mdadm
4. Select the appropriate filesystem (ext4, XFS, Btrfs) based on workload requirements
5. Encrypt disks with LUKS and configure automatic unlocking at boot
6. Combine LUKS encryption with LVM for secure, flexible storage
7. Monitor disk health with SMART and perform routine filesystem maintenance

## Table of Contents
1. [Storage Fundamentals](#1-storage-fundamentals)
2. [LVM](#2-lvm)
3. [RAID](#3-raid)
4. [Filesystems](#4-filesystems)
5. [Disk Encryption](#5-disk-encryption)
6. [Monitoring and Maintenance](#6-monitoring-and-maintenance)
7. [Practice Exercises](#7-practice-exercises)

---

Data is the most valuable asset on any server, and the storage layer is what protects it. A misconfigured RAID array, an unencrypted disk, or a full partition can lead to data loss, security breaches, or extended downtime. This lesson covers the essential storage technologies -- LVM for flexibility, RAID for redundancy, modern filesystems for reliability, and LUKS for encryption -- that every Linux administrator must master to build production-grade storage infrastructure.

## 1. Storage Fundamentals

### 1.1 Storage Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer Structure                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Application                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │           VFS (Virtual File System)     │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    Filesystem (ext4, XFS, Btrfs)        │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    Block Device (LVM, RAID, LUKS)       │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    Disk Driver                           │               │
│  └─────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    Physical Disk (HDD, SSD, NVMe)       │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Checking Disk Information

```bash
# Block device list
lsblk
lsblk -f  # Include filesystem

# Detailed disk information
fdisk -l
parted -l

# SMART information (disk health)
smartctl -a /dev/sda

# Disk UUIDs
blkid

# Disk usage
df -h
df -i  # inode usage
```

### 1.3 Partition Management

```bash
# fdisk (MBR)
fdisk /dev/sdb
# n - new partition
# d - delete partition
# p - print partition table
# w - write and exit

# parted (GPT recommended)
parted /dev/sdb
(parted) mklabel gpt
(parted) mkpart primary ext4 0% 50%
(parted) mkpart primary ext4 50% 100%
(parted) print
(parted) quit

# gdisk (GPT)
gdisk /dev/sdb
```

---

## 2. LVM

### 2.1 LVM Concept

```
┌─────────────────────────────────────────────────────────────┐
│                       LVM Structure                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Logical Volume (LV)                                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │   lv1   │ │   lv2   │ │   lv3   │  ← Create filesystem  │
│  │  (root) │ │  (home) │ │  (data) │                       │
│  └─────────┴─────────┴─────────┘                           │
│           │                                                 │
│           ▼                                                 │
│  Volume Group (VG)                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                       vg0                            │   │
│  │   ← Combine multiple PVs                             │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  Physical Volume (PV)                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │  /dev/  │ │  /dev/  │ │  /dev/  │  ← Physical disks     │
│  │  sda1   │ │  sdb1   │ │  sdc1   │                       │
│  └─────────┘ └─────────┘ └─────────┘                       │
│                                                             │
│  Advantages:                                                │
│  • Dynamic volume resizing                                  │
│  • Combine multiple disks                                   │
│  • Snapshot support                                         │
│  • Online expansion                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Creating LVM

```bash
# 1. Create Physical Volume (PV)
pvcreate /dev/sdb1
pvcreate /dev/sdc1

# Check PV
pvs
pvdisplay /dev/sdb1

# 2. Create Volume Group (VG)
vgcreate vg_data /dev/sdb1 /dev/sdc1

# Check VG
vgs
vgdisplay vg_data

# 3. Create Logical Volume (LV)
lvcreate -L 10G -n lv_home vg_data
lvcreate -l 100%FREE -n lv_data vg_data  # Use all remaining space

# Check LV
lvs
lvdisplay /dev/vg_data/lv_home

# 4. Create filesystem and mount
mkfs.ext4 /dev/vg_data/lv_home
mkdir /mnt/home
mount /dev/vg_data/lv_home /mnt/home

# Add to /etc/fstab
echo '/dev/vg_data/lv_home /home ext4 defaults 0 2' >> /etc/fstab
```

### 2.3 Extending LVM

```bash
# Add new disk to VG
pvcreate /dev/sdd1
vgextend vg_data /dev/sdd1

# Extend LV
lvextend -L +5G /dev/vg_data/lv_home
# Or use all free space
lvextend -l +100%FREE /dev/vg_data/lv_home

# Extend filesystem
# ext4
resize2fs /dev/vg_data/lv_home

# XFS (grow only)
xfs_growfs /mnt/home

# Extend LV + filesystem together
lvextend -r -L +5G /dev/vg_data/lv_home
```

### 2.4 Shrinking LVM

```bash
# ⚠️ Warning: Backup data first!

# Shrink ext4 (unmount required)
umount /mnt/home
e2fsck -f /dev/vg_data/lv_home
resize2fs /dev/vg_data/lv_home 8G
lvreduce -L 8G /dev/vg_data/lv_home
mount /dev/vg_data/lv_home /mnt/home

# XFS cannot be shrunk!
```

### 2.5 LVM Snapshots

```bash
# Create snapshot
lvcreate -L 1G -s -n snap_home /dev/vg_data/lv_home

# Check snapshot
lvs
lvdisplay /dev/vg_data/snap_home

# Mount snapshot (verify before restore)
mount -o ro /dev/vg_data/snap_home /mnt/snapshot

# Restore from snapshot
lvconvert --merge /dev/vg_data/snap_home
# May require reboot

# Remove snapshot
lvremove /dev/vg_data/snap_home
```

---

## 3. RAID

### 3.1 RAID Levels

```
┌─────────────────────────────────────────────────────────────┐
│                      RAID Level Comparison                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RAID 0 (Striping)                                          │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │ A1  │ A2  │ A3  │ A4  │  Pros: Best performance         │
│  │ B1  │ B2  │ B3  │ B4  │  Cons: No fault tolerance       │
│  └─────┴─────┴─────┴─────┘  Capacity: 100%                │
│  Disk1 Disk2 Disk3 Disk4                                   │
│                                                             │
│  RAID 1 (Mirroring)                                         │
│  ┌─────┬─────┐                                              │
│  │ A1  │ A1  │  Pros: Perfect replication                  │
│  │ B1  │ B1  │  Cons: 50% capacity                         │
│  └─────┴─────┘  Capacity: 50%                              │
│  Disk1 Disk2                                                │
│                                                             │
│  RAID 5 (Striping + Parity)                                 │
│  ┌─────┬─────┬─────┐                                        │
│  │ A1  │ A2  │ Ap  │  Pros: Performance + fault (1 disk)   │
│  │ B1  │ Bp  │ B2  │  Cons: Parity calculation on write    │
│  │ Cp  │ C1  │ C2  │  Capacity: (n-1)/n                    │
│  └─────┴─────┴─────┘  Minimum 3 disks                      │
│  Disk1 Disk2 Disk3                                          │
│                                                             │
│  RAID 6 (Double Parity)                                     │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │ A1  │ A2  │ Ap  │ Aq  │  Pros: 2 disk fault tolerance   │
│  │ B1  │ Bp  │ Bq  │ B2  │  Cons: Slower writes            │
│  └─────┴─────┴─────┴─────┘  Capacity: (n-2)/n              │
│                                                             │
│  RAID 10 (1+0, Striped Mirrors)                            │
│  ┌─────┬─────┐ ┌─────┬─────┐                               │
│  │ A1  │ A1  │ │ A2  │ A2  │  Pros: Performance + reliability│
│  │ B1  │ B1  │ │ B2  │ B2  │  Cons: 50% capacity           │
│  └─────┴─────┘ └─────┴─────┘  Minimum 4 disks             │
│  Mirror 1      Mirror 2                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Configuring RAID with mdadm

```bash
# Install mdadm
apt install mdadm

# Create RAID 1
mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb1 /dev/sdc1

# Create RAID 5
mdadm --create /dev/md1 --level=5 --raid-devices=3 /dev/sdd1 /dev/sde1 /dev/sdf1

# Create RAID 10
mdadm --create /dev/md2 --level=10 --raid-devices=4 /dev/sdg1 /dev/sdh1 /dev/sdi1 /dev/sdj1

# Check RAID status
cat /proc/mdstat
mdadm --detail /dev/md0

# Save configuration
mdadm --detail --scan >> /etc/mdadm/mdadm.conf
update-initramfs -u

# Create filesystem
mkfs.ext4 /dev/md0
mount /dev/md0 /mnt/raid
```

### 3.3 RAID Management

```bash
# Simulate disk failure
mdadm --manage /dev/md0 --fail /dev/sdb1

# Remove failed disk
mdadm --manage /dev/md0 --remove /dev/sdb1

# Add new disk
mdadm --manage /dev/md0 --add /dev/sdk1

# Add spare disk
mdadm --manage /dev/md0 --add-spare /dev/sdl1

# Check rebuild status
cat /proc/mdstat
watch -n 1 cat /proc/mdstat

# Stop/start RAID
mdadm --stop /dev/md0
mdadm --assemble /dev/md0 /dev/sdb1 /dev/sdc1
```

### 3.4 RAID Expansion

```bash
# Add disk to RAID 5/6
mdadm --grow /dev/md1 --raid-devices=4 --add /dev/sdm1

# Expand size and filesystem
mdadm --grow /dev/md1 --size=max
resize2fs /dev/md1
```

---

## 4. Filesystems

### 4.1 Filesystem Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                    Filesystem Comparison                        │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│              │     ext4     │     XFS      │     Btrfs        │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ Max Volume   │    1 EB      │    8 EB      │    16 EB         │
│ Max File     │   16 TB      │    8 EB      │    16 EB         │
│ Journaling   │     Yes      │     Yes      │     CoW          │
│ Online Grow  │     Yes      │     Yes      │     Yes          │
│ Online Shrink│     No       │     No       │     Yes          │
│ Snapshots    │   (LVM)      │   (LVM)      │     Yes (native) │
│ Compression  │     No       │     No       │     Yes          │
│ Checksums    │  Metadata    │     No       │     Yes (all)    │
│ Best For     │  General/Boot│  Large/DB    │  NAS/Snapshots   │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

### 4.2 ext4 Management

```bash
# Create
mkfs.ext4 /dev/sdb1
mkfs.ext4 -L "DATA" /dev/sdb1  # With label

# Create with options
mkfs.ext4 -b 4096 -i 16384 -O ^has_journal /dev/sdb1

# Check information
tune2fs -l /dev/sdb1

# Change settings
tune2fs -L "NEW_LABEL" /dev/sdb1  # Label
tune2fs -c 30 /dev/sdb1           # fsck after 30 mounts
tune2fs -i 1m /dev/sdb1           # fsck after 1 month
tune2fs -O ^has_journal /dev/sdb1 # Disable journal

# Filesystem check
e2fsck -f /dev/sdb1
e2fsck -p /dev/sdb1  # Auto-fix

# Defragment (online)
e4defrag /dev/sdb1
```

### 4.3 XFS Management

```bash
# Create
mkfs.xfs /dev/sdb1
mkfs.xfs -L "DATA" -f /dev/sdb1

# Check information
xfs_info /dev/sdb1
xfs_info /mnt/data  # If mounted

# Change label
xfs_admin -L "NEW_LABEL" /dev/sdb1

# Grow filesystem (online)
xfs_growfs /mnt/data

# Filesystem check (unmount required)
xfs_repair /dev/sdb1
xfs_repair -n /dev/sdb1  # Check only

# Defragment (online)
xfs_fsr /mnt/data
```

### 4.4 Btrfs Management

```bash
# Create
mkfs.btrfs /dev/sdb1
mkfs.btrfs -L "DATA" -m raid1 -d raid1 /dev/sdb1 /dev/sdc1

# Check information
btrfs filesystem show
btrfs filesystem df /mnt/btrfs

# Create subvolume
btrfs subvolume create /mnt/btrfs/subvol1

# Snapshot
btrfs subvolume snapshot /mnt/btrfs/subvol1 /mnt/btrfs/snap1
btrfs subvolume snapshot -r /mnt/btrfs/subvol1 /mnt/btrfs/snap_ro  # Read-only

# List subvolumes
btrfs subvolume list /mnt/btrfs

# Delete snapshot
btrfs subvolume delete /mnt/btrfs/snap1

# Enable compression
mount -o compress=zstd /dev/sdb1 /mnt/btrfs

# Add disk
btrfs device add /dev/sdd1 /mnt/btrfs
btrfs balance start /mnt/btrfs

# Scrub (data integrity check)
btrfs scrub start /mnt/btrfs
btrfs scrub status /mnt/btrfs
```

---

## 5. Disk Encryption

### 5.1 LUKS Encryption

```bash
# Create LUKS volume
cryptsetup luksFormat /dev/sdb1
# Enter passphrase

# Open LUKS
cryptsetup open /dev/sdb1 encrypted_disk
# Creates /dev/mapper/encrypted_disk

# Create filesystem
mkfs.ext4 /dev/mapper/encrypted_disk

# Mount
mount /dev/mapper/encrypted_disk /mnt/encrypted

# Unmount and close
umount /mnt/encrypted
cryptsetup close encrypted_disk
```

### 5.2 LUKS Management

```bash
# LUKS information
cryptsetup luksDump /dev/sdb1

# Add key (max 8)
cryptsetup luksAddKey /dev/sdb1

# Remove key
cryptsetup luksRemoveKey /dev/sdb1

# Use key file
dd if=/dev/urandom of=/root/keyfile bs=1024 count=4
chmod 400 /root/keyfile
cryptsetup luksAddKey /dev/sdb1 /root/keyfile

# Open with key file
cryptsetup open /dev/sdb1 encrypted_disk --key-file /root/keyfile
```

### 5.3 Auto-mount on Boot

```bash
# /etc/crypttab
# <name> <device> <key file> <options>
encrypted_disk /dev/sdb1 /root/keyfile luks

# /etc/fstab
/dev/mapper/encrypted_disk /mnt/encrypted ext4 defaults 0 2
```

### 5.4 LUKS + LVM

```bash
# Create encrypted PV
cryptsetup luksFormat /dev/sdb1
cryptsetup open /dev/sdb1 crypt_pv

# Configure LVM
pvcreate /dev/mapper/crypt_pv
vgcreate vg_encrypted /dev/mapper/crypt_pv
lvcreate -l 100%FREE -n lv_data vg_encrypted

# Filesystem
mkfs.ext4 /dev/vg_encrypted/lv_data
```

---

## 6. Monitoring and Maintenance

### 6.1 Disk Status Monitoring

```bash
# SMART monitoring
smartctl -H /dev/sda              # Health status
smartctl -a /dev/sda              # Full information
smartctl -t short /dev/sda        # Short test
smartctl -l selftest /dev/sda     # Test results

# smartd daemon configuration
# /etc/smartd.conf
/dev/sda -a -o on -S on -s (S/../.././02|L/../../6/03)
#                             Daily 2am short test | Saturday 3am long test

# I/O statistics
iostat -x 1

# Disk queue
cat /sys/block/sda/queue/nr_requests
```

### 6.2 Filesystem Maintenance

```bash
# Regular check (ext4)
tune2fs -c 30 /dev/sda1     # fsck after 30 mounts
tune2fs -i 1m /dev/sda1     # fsck every month

# Reserved blocks (ext4)
tune2fs -m 1 /dev/sda1      # Reduce to 1% (default 5%)

# TRIM (SSD)
fstrim -v /                  # Manual TRIM
systemctl enable fstrim.timer  # Periodic TRIM

# Disk usage analysis
du -sh /*
ncdu /
```

### 6.3 Backup Strategy

```bash
# Full disk backup with dd
dd if=/dev/sda of=/backup/sda.img bs=64M status=progress

# Compressed backup
dd if=/dev/sda bs=64M | gzip > /backup/sda.img.gz

# Clone partition
dd if=/dev/sda1 of=/dev/sdb1 bs=64M status=progress

# File backup with rsync
rsync -avz --delete /data/ /backup/data/

# Consistent backup with LVM snapshot
lvcreate -L 1G -s -n snap_backup /dev/vg/lv_data
mount -o ro /dev/vg/snap_backup /mnt/snap
rsync -avz /mnt/snap/ /backup/
umount /mnt/snap
lvremove -f /dev/vg/snap_backup
```

---

## 7. Practice Exercises

### Exercise 1: LVM Configuration
```bash
# Requirements:
# 1. Create VG from 2 disks
# 2. Create 3 LVs (root 20G, home 50G, data remaining)
# 3. Format as ext4, xfs, ext4 respectively
# 4. Configure permanent mount in fstab

# Write commands:
```

### Exercise 2: RAID 5 Configuration
```bash
# Requirements:
# 1. Configure RAID 5 with 4 disks
# 2. Set 1 as spare
# 3. Simulate failure and recovery

# Write commands:
```

### Exercise 3: Encrypted LVM
```bash
# Requirements:
# 1. Encrypt disk with LUKS
# 2. Configure LVM on encrypted volume
# 3. Auto-mount on boot (using key file)

# Write configuration files and commands:
```

### Exercise 4: Btrfs Snapshot Management
```bash
# Requirements:
# 1. Create Btrfs volume
# 2. Design subvolume structure
# 3. Establish snapshot policy (daily, weekly)
# 4. Test restore from snapshot

# Write script:
```

---

---

## References

- [LVM Administrator's Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_and_managing_logical_volumes/)
- [Linux RAID Wiki](https://raid.wiki.kernel.org/)
- [Btrfs Wiki](https://btrfs.wiki.kernel.org/)
- [LUKS Documentation](https://gitlab.com/cryptsetup/cryptsetup)

---

**Previous**: [Container Internals](./15_Container_Internals.md) | **Next**: [SELinux and AppArmor](./17_SELinux_AppArmor.md)
