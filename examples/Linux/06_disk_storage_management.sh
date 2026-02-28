#!/usr/bin/env bash
# =============================================================================
# 06_disk_storage_management.sh - Disk, Filesystem, and LVM Management
#
# PURPOSE: Demonstrates LVM operations, filesystem creation, mount management,
#          fstab configuration, and disk usage analysis. Destructive operations
#          are dry-run by default.
#
# USAGE:
#   ./06_disk_storage_management.sh [--disk|--lvm|--filesystem|--usage|--all]
#
# MODES:
#   --disk        Disk identification and partitioning concepts
#   --lvm         LVM operations (PV, VG, LV lifecycle)
#   --filesystem  Filesystem creation, mounting, fstab
#   --usage       Disk usage analysis and quota management
#   --all         Run all sections (default)
#
# CONCEPTS COVERED:
#   - Block device identification: lsblk, blkid, fdisk
#   - LVM: Physical Volumes → Volume Groups → Logical Volumes
#   - Filesystem types: ext4, xfs, btrfs tradeoffs
#   - Mount options: noexec, nosuid, noatime
#   - fstab: persistent mounts with UUID
#   - Disk usage: du, df, ncdu, quota
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

DRY_RUN="${DRY_RUN:-true}"

section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }
explain() { echo -e "${GREEN}[INFO]${RESET} $1"; }
show_cmd() { echo -e "${YELLOW}[CMD]${RESET} $1"; }
run_safe() { show_cmd "$1"; eval "$1" 2>/dev/null || echo "  (unavailable)"; }
run_or_dry() {
    show_cmd "$1"
    if [[ "${DRY_RUN}" == "false" ]]; then eval "$1"; else echo -e "  ${RED}(dry-run)${RESET}"; fi
}

# ---------------------------------------------------------------------------
# 1. Disk Identification
# ---------------------------------------------------------------------------
demo_disk() {
    section "1. Disk Identification & Partitioning"

    explain "Listing block devices:"
    # Why: lsblk shows the tree relationship between disks, partitions,
    # and LVM volumes. The TYPE column distinguishes disk/part/lvm/crypt.
    if command -v lsblk &>/dev/null; then
        run_safe "lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE"
    else
        # macOS fallback
        run_safe "diskutil list"
    fi
    echo ""

    explain "Identifying filesystems and UUIDs:"
    # Why: UUIDs are stable identifiers — device names like /dev/sdb1 can
    # change across reboots if disk order changes. fstab should use UUID=.
    if command -v blkid &>/dev/null; then
        run_safe "blkid 2>/dev/null | head -5"
    fi
    echo ""

    explain "Partition table types:"
    echo "  MBR (Master Boot Record): Legacy, max 2TB, 4 primary partitions"
    echo "  GPT (GUID Partition Table): Modern, max 9.4ZB, 128+ partitions"
    echo ""

    explain "Partitioning tools:"
    show_cmd "fdisk /dev/sdb           # Interactive MBR/GPT partitioning"
    show_cmd "gdisk /dev/sdb           # GPT-only partitioning"
    show_cmd "parted /dev/sdb print    # Non-interactive partition info"
    echo ""

    explain "Partition alignment (critical for SSDs):"
    # Why: Misaligned partitions cause read-modify-write amplification on
    # SSDs, degrading performance by 10-30%. Modern tools align by default.
    show_cmd "parted /dev/sdb align-check optimal 1"
}

# ---------------------------------------------------------------------------
# 2. LVM (Logical Volume Manager)
# ---------------------------------------------------------------------------
demo_lvm() {
    section "2. LVM — Logical Volume Manager"

    explain "LVM architecture:"
    echo "  Physical Volumes (PV): Raw disks or partitions"
    echo "       ↓"
    echo "  Volume Groups (VG): Pool of storage from one or more PVs"
    echo "       ↓"
    echo "  Logical Volumes (LV): Virtual partitions carved from VG"
    echo ""
    # Why: LVM adds an abstraction layer between physical disks and
    # filesystems. This enables live resizing, snapshots, and spanning
    # multiple physical disks — none of which raw partitions support.

    explain "Creating the LVM stack:"
    run_or_dry "pvcreate /dev/sdb1 /dev/sdc1            # Initialize PVs"
    run_or_dry "vgcreate data_vg /dev/sdb1 /dev/sdc1    # Create VG from PVs"
    run_or_dry "lvcreate -L 50G -n app_lv data_vg        # Create 50GB LV"
    run_or_dry "lvcreate -l 100%FREE -n log_lv data_vg   # Use remaining space"
    echo ""

    explain "Inspecting LVM state:"
    if command -v pvs &>/dev/null; then
        show_cmd "pvs      # Physical volume summary"
        show_cmd "vgs      # Volume group summary"
        show_cmd "lvs      # Logical volume summary"
        pvs 2>/dev/null || echo "  (no PVs found or LVM not installed)"
    else
        explain "(LVM tools not available on this system)"
    fi
    echo ""

    explain "Live resizing (key LVM advantage):"
    # Why: With raw partitions, resizing requires unmounting, repartitioning,
    # and filesystem resize. LVM can extend a mounted filesystem live.
    run_or_dry "lvextend -L +20G /dev/data_vg/app_lv    # Add 20GB"
    run_or_dry "resize2fs /dev/data_vg/app_lv            # Grow ext4 online"
    # For XFS:
    show_cmd "xfs_growfs /mount/point                    # XFS equivalent"
    echo ""

    explain "LVM snapshots (for backups):"
    # Why: Snapshots capture a point-in-time copy using copy-on-write.
    # Backup can read from the frozen snapshot while the live LV continues.
    run_or_dry "lvcreate -s -L 10G -n app_snap /dev/data_vg/app_lv"
    show_cmd "mount /dev/data_vg/app_snap /mnt/snap      # Mount snapshot"
    show_cmd "lvremove /dev/data_vg/app_snap              # Remove when done"
}

# ---------------------------------------------------------------------------
# 3. Filesystem Management
# ---------------------------------------------------------------------------
demo_filesystem() {
    section "3. Filesystem Creation & Mounting"

    explain "Filesystem type comparison:"
    echo "  ext4   — Default Linux FS. Mature, fast fsck, journaled."
    echo "           Best for: general purpose, < 50TB volumes"
    echo "  xfs    — High-performance, scales to exabytes. No shrink."
    echo "           Best for: large files, databases, media servers"
    echo "  btrfs  — Copy-on-write, snapshots, compression, checksums."
    echo "           Best for: NAS, containers, when data integrity matters"
    echo ""

    explain "Creating filesystems:"
    run_or_dry "mkfs.ext4 -L appdata /dev/data_vg/app_lv"
    run_or_dry "mkfs.xfs -L logs /dev/data_vg/log_lv"
    echo ""

    explain "Mounting with options:"
    # Why: Mount options affect security and performance. nosuid prevents
    # setuid binaries from running, noexec prevents script execution,
    # noatime reduces write I/O by not updating access timestamps.
    run_or_dry "mount -o noatime,nosuid /dev/data_vg/app_lv /srv/app"
    echo ""

    explain "Persistent mounts via /etc/fstab:"
    echo "  Format: <device> <mountpoint> <type> <options> <dump> <fsck>"
    echo ""
    echo "  # Why: UUID= is more reliable than /dev/sdX device names"
    echo "  UUID=abc123... /srv/app  ext4  defaults,noatime,nosuid  0 2"
    echo "  UUID=def456... /var/log  xfs   defaults,noatime         0 2"
    echo ""
    # Why: dump (5th field) is usually 0 (disable backup), fsck order (6th)
    # is 1 for root, 2 for others, 0 to skip. Wrong fsck order can cause
    # boot failures when filesystems are checked in parallel incorrectly.
    echo "  # tmpfs for /tmp (RAM-backed, cleared on reboot)"
    echo "  tmpfs  /tmp  tmpfs  defaults,noexec,nosuid,size=2G  0 0"
    echo ""

    explain "Validating fstab before reboot:"
    # Why: A broken fstab can prevent the system from booting. Always
    # test with mount -a first, which mounts all entries without rebooting.
    show_cmd "mount -a    # Mount all fstab entries (test before reboot)"
    show_cmd "findmnt --verify    # Verify fstab syntax (systemd)"
}

# ---------------------------------------------------------------------------
# 4. Disk Usage Analysis
# ---------------------------------------------------------------------------
demo_usage() {
    section "4. Disk Usage Analysis"

    explain "Filesystem-level usage (df):"
    # Why: df shows free space per mounted filesystem. -h = human-readable,
    # -T = show filesystem type. Watch for >85% usage.
    run_safe "df -hT 2>/dev/null | head -10"
    echo ""

    explain "Directory-level usage (du):"
    # Why: When df says a filesystem is full, du finds where the space went.
    # --max-depth limits recursion, -h gives human-readable sizes.
    show_cmd "du -sh /var/log/*  | sort -rh | head -10"
    if [[ -d /var/log ]]; then
        du -sh /var/log/* 2>/dev/null | sort -rh | head -5 || true
    fi
    echo ""

    explain "Finding large files:"
    show_cmd "find /var -type f -size +100M -exec ls -lh {} \\; 2>/dev/null"
    echo ""

    explain "Inode usage (often overlooked):"
    # Why: A filesystem can run out of inodes before running out of space,
    # especially with millions of small files. This causes "No space left
    # on device" even though df shows free space.
    run_safe "df -i 2>/dev/null | head -5"
    echo ""

    explain "Finding deleted-but-open files (phantom space usage):"
    # Why: If a process holds a file descriptor open to a deleted file,
    # the space is not reclaimed until the process closes the fd.
    show_cmd "lsof +L1    # Files with link count 0 (deleted but open)"
    echo ""

    explain "Disk I/O monitoring:"
    if command -v iostat &>/dev/null; then
        run_safe "iostat -xz 1 1 2>/dev/null | head -15"
    else
        show_cmd "iostat -xz 1 3    # Install: apt install sysstat"
    fi

    explain "Key I/O metrics to watch:"
    echo "  %util   — How busy the device is (>70% = bottleneck)"
    echo "  await   — Average I/O wait time in ms (>10ms for SSD = issue)"
    echo "  r/s,w/s — Read/write operations per second"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}============================================${RESET}"
    echo -e "${BOLD} Disk & Storage Management Demo${RESET}"
    echo -e "${BOLD}============================================${RESET}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${RED}Destructive commands run in DRY-RUN mode.${RESET}\n"
    fi

    local mode="${1:---all}"
    case "${mode}" in
        --disk)       demo_disk ;;
        --lvm)        demo_lvm ;;
        --filesystem) demo_filesystem ;;
        --usage)      demo_usage ;;
        --all)
            demo_disk
            demo_lvm
            demo_filesystem
            demo_usage
            ;;
        *) echo "Usage: $0 [--disk|--lvm|--filesystem|--usage|--all]"; exit 1 ;;
    esac

    echo -e "\n${GREEN}${BOLD}Done.${RESET}"
}

main "$@"
