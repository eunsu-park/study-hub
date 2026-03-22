#!/bin/bash
# Exercises for Lesson 19: Backup and Recovery
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: rsync Incremental Backup ===
# Problem: Write an rsync incremental backup script using hard links.
#          Source: /home/user, Destination: /backup/home, daily with 30-day retention.
exercise_1() {
    echo "=== Exercise 1: rsync Incremental Backup with Hard Links ==="
    echo ""
    echo "Scenario: Create a space-efficient daily backup system that uses hard links"
    echo "to avoid duplicating unchanged files across snapshots."
    echo ""

    echo "Solution: backup_incremental.sh"
    echo ""
    cat << 'SCRIPT'
#!/bin/bash
# Incremental backup using rsync hard links
# Each day creates a full directory tree, but unchanged files are hard-linked
# to the previous backup, saving disk space.

SOURCE="/home/user"
BACKUP_BASE="/backup/home"
DATE=$(date +%Y-%m-%d)
BACKUP_PATH="$BACKUP_BASE/$DATE"
LATEST="$BACKUP_BASE/latest"

# Use --link-dest for hard link-based incremental backup
# If a 'latest' symlink exists, use it as the reference for hard links
if [ -d "$LATEST" ]; then
    LINK="--link-dest=$LATEST"
else
    LINK=""
fi

# Execute backup
# -a = archive mode (preserves permissions, timestamps, symlinks, etc.)
# -v = verbose output
# --delete = remove files from backup that no longer exist in source
# --link-dest = create hard links to matching files in the reference directory
rsync -av --delete $LINK "$SOURCE/" "$BACKUP_PATH/"

# Update the 'latest' symlink to point to today's backup
# This becomes the reference for tomorrow's --link-dest
rm -f "$LATEST"
ln -s "$BACKUP_PATH" "$LATEST"

# Delete backups older than 30 days
# -maxdepth 1 = only top-level directories
# -name "20*" = match date-formatted directories (2024-01-15, etc.)
# -mtime +30 = modified more than 30 days ago
find "$BACKUP_BASE" -maxdepth 1 -type d -name "20*" -mtime +30 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_PATH"
SCRIPT
    echo ""
    echo "How hard-link backups work:"
    echo "  Day 1: Full copy of all files (e.g., 10GB)"
    echo "  Day 2: rsync compares to Day 1 via --link-dest"
    echo "         - Changed files: new copy (actual disk usage)"
    echo "         - Unchanged files: hard link to Day 1 copy (zero extra space)"
    echo "  Result: Each backup LOOKS like a full backup but only uses space"
    echo "          proportional to the changes."
    echo ""
    echo "  Why hard links? A hard link is another name for the same data on disk."
    echo "  Deleting one link doesn't affect others. When you delete an old backup,"
    echo "  files that are still hard-linked from newer backups remain intact."
    echo ""
    echo "Cron setup for daily execution at 2 AM:"
    echo "  0 2 * * * /usr/local/bin/backup_incremental.sh >> /var/log/backup.log 2>&1"
}

# === Exercise 2: Borg Recovery ===
# Problem: Recover only /etc/nginx/ from a specific date's backup in a Borg repository.
exercise_2() {
    echo "=== Exercise 2: Borg Backup Recovery ==="
    echo ""
    echo "Scenario: Recover the nginx configuration from a specific backup archive."
    echo ""

    echo "Solution:"
    echo ""
    echo "--- Step 1: List available archives to find the right one ---"
    echo "  borg list /backup/borg-repo"
    echo ""
    echo "  Example output:"
    echo "    backup-2024-01-13  Mon, 2024-01-13 02:00:03"
    echo "    backup-2024-01-14  Tue, 2024-01-14 02:00:02"
    echo "    backup-2024-01-15  Wed, 2024-01-15 02:00:04"
    echo ""

    echo "--- Step 2: Inspect the archive contents (optional) ---"
    echo "  borg list /backup/borg-repo::backup-2024-01-15 | grep etc/nginx"
    echo ""
    echo "  Why: Verify the files exist before extracting."
    echo ""

    echo "--- Step 3: Extract only /etc/nginx/ from the specific archive ---"
    echo "  # Method A: Extract to current directory (creates ./etc/nginx/)"
    echo "  mkdir /tmp/restore && cd /tmp/restore"
    echo "  borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx"
    echo ""
    echo "  # Method B: Extract and restore in place (CAUTION: overwrites current files)"
    echo "  cd /"
    echo "  borg extract /backup/borg-repo::backup-2024-01-15 etc/nginx"
    echo ""
    echo "  Note: Borg paths are relative (no leading /). Use 'etc/nginx' not '/etc/nginx'."
    echo ""

    echo "--- Step 4: Verify and apply ---"
    echo "  # Compare restored with current"
    echo "  diff -r /tmp/restore/etc/nginx /etc/nginx"
    echo ""
    echo "  # Copy restored files into place"
    echo "  sudo cp -a /tmp/restore/etc/nginx/* /etc/nginx/"
    echo "  sudo nginx -t          # Test config syntax"
    echo "  sudo systemctl reload nginx"
    echo ""

    echo "Key Borg concepts:"
    echo "  Repository: /backup/borg-repo (encrypted, deduplicated storage)"
    echo "  Archive: A named snapshot within the repo (backup-2024-01-15)"
    echo "  Deduplication: Borg splits files into chunks; identical chunks stored once"
    echo "  Encryption: Borg repos are encrypted at rest (AES-256)"
}

# === Exercise 3: Disaster Recovery Testing Procedure ===
# Problem: Write a quarterly DR testing procedure with integrity checks and recovery tests.
exercise_3() {
    echo "=== Exercise 3: Disaster Recovery Testing Procedure ==="
    echo ""
    echo "Scenario: Establish a repeatable DR testing process to ensure backups"
    echo "are valid and recovery procedures work when needed."
    echo ""

    echo "Solution: Quarterly DR Testing Checklist"
    echo ""
    cat << 'PROCEDURE'
# ============================================
# Quarterly Disaster Recovery Testing Procedure
# ============================================

# === Phase 1: Backup Integrity Verification (Every Quarter) ===

# 1.1 Verify Borg repository integrity
borg check /backup/borg-repo
# Why: Detects data corruption, missing chunks, broken archives
# This can take a while for large repos; schedule during off-hours

# 1.2 List recent backups and verify schedule compliance
borg list /backup/borg-repo | tail -10
# Check: Are there daily entries? Any gaps?

# 1.3 Check latest archive details
borg info /backup/borg-repo::latest
# Verify: Size is reasonable, timestamp is recent, no errors

# 1.4 Verify rsync backup integrity
ls -la /backup/home/latest
find /backup/home -maxdepth 1 -type d | wc -l
# Check: 'latest' symlink exists and points to recent date


# === Phase 2: Sample Data Recovery Test (Every Quarter) ===

# 2.1 Create isolated test directory
mkdir -p /tmp/dr-test-$(date +%Y%m%d)
cd /tmp/dr-test-$(date +%Y%m%d)

# 2.2 Test config file recovery
borg extract /backup/borg-repo::latest etc/nginx
diff -r etc/nginx /etc/nginx
# Expected: Files match current config (or show known changes)

# 2.3 Test data file recovery
borg extract /backup/borg-repo::latest var/www
ls -la var/www/
# Verify: File count, sizes, and permissions look correct

# 2.4 Verify file integrity with checksums
find var/www -type f -exec md5sum {} \; > restored_checksums.txt
# Compare against production checksums if available


# === Phase 3: Full System Recovery Test (Semi-annually) ===

# 3.1 Prepare test environment
# Option A: VM (preferred)
#   Create a fresh VM matching production specs
# Option B: Spare hardware
#   Boot from live USB

# 3.2 Execute bare metal recovery
# Restore bootloader, partitions, and filesystem
# Restore /etc, /home, /var from Borg backup
# Reinstall packages from saved package list:
#   dpkg --set-selections < package_list.txt && apt-get dselect-upgrade

# 3.3 Verify system functionality
#   - System boots successfully
#   - Network connectivity works
#   - Services start (nginx, postgresql, etc.)
#   - Application responds to requests
#   - Data integrity verified

# 3.4 Measure RTO/RPO
#   RTO (Recovery Time Objective): Time from start to full service
#   RPO (Recovery Point Objective): Age of newest backup used
#   Record both and compare against targets


# === Phase 4: Documentation and Improvement ===

# Document results
echo "DR Test Date: $(date +%Y-%m-%d)" >> /var/log/dr-test-results.log
echo "Phase 1 (Integrity): PASS/FAIL" >> /var/log/dr-test-results.log
echo "Phase 2 (Sample Recovery): PASS/FAIL" >> /var/log/dr-test-results.log
echo "Phase 3 (Full Recovery): PASS/FAIL" >> /var/log/dr-test-results.log
echo "RTO Achieved: XX minutes (target: YY)" >> /var/log/dr-test-results.log
echo "RPO Achieved: XX hours (target: YY)" >> /var/log/dr-test-results.log

# Clean up test artifacts
rm -rf /tmp/dr-test-*
PROCEDURE
    echo ""
    echo "Why regular DR testing matters:"
    echo "  - Backups may silently fail (corruption, missed files, wrong permissions)"
    echo "  - Recovery procedures drift as infrastructure changes"
    echo "  - Staff turnover means knowledge of recovery steps may be lost"
    echo "  - RTO/RPO targets can only be validated through actual testing"
    echo "  - Compliance frameworks (SOC2, ISO27001) often require documented DR tests"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
