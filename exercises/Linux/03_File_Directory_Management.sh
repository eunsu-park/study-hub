#!/bin/bash
# Exercises for Lesson 03: File and Directory Management
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Bulk File Operations ===
# Problem: Perform bulk copy, move, and remove operations using patterns and options.
exercise_1() {
    echo "=== Exercise 1: Bulk File Operations (cp, mv, rm with Patterns) ==="
    echo ""
    echo "Scenario: You need to reorganize a project directory — move logs to an"
    echo "archive, copy configs to a backup, and clean up temporary files."
    echo ""

    echo "--- Part A: Copy files with preservation and patterns ---"
    echo "Solution:"
    echo "  cp -a /etc/nginx/ /backup/nginx-\$(date +%Y%m%d)/   # Archive copy"
    echo "  cp -r /opt/project/src/ /opt/project/src.bak/       # Recursive copy"
    echo "  cp -u /src/*.conf /dest/                             # Copy only newer files"
    echo "  cp --parents etc/nginx/nginx.conf /backup/           # Preserve directory structure"
    echo ""
    echo "  Explanation:"
    echo "    -a (archive) = -dR --preserve=all: copies recursively, preserving"
    echo "    symlinks, permissions, timestamps, ownership, and ACLs."
    echo "    -u (update) skips files that are newer in the destination."
    echo "    --parents recreates the source directory structure in the target."
    echo ""

    echo "--- Part B: Move and rename with safety ---"
    echo "Solution:"
    echo "  mv -i oldname.txt newname.txt            # Interactive — prompts before overwrite"
    echo "  mv -n /tmp/upload/* /data/uploads/       # No-clobber — never overwrite"
    echo "  mv -v /var/log/*.log.1 /archive/logs/    # Verbose — show each move"
    echo ""
    echo "  Explanation:"
    echo "    -i (interactive) asks 'overwrite?' for each conflict. Safe default."
    echo "    -n (no-clobber) silently skips existing files. Ideal for scripts."
    echo "    -v (verbose) prints 'renamed X -> Y' for each operation."
    echo "    On the same filesystem, mv is instant (just renames the directory entry)."
    echo ""

    echo "--- Part C: Safe deletion patterns ---"
    echo "Solution:"
    echo "  rm -i *.tmp                     # Interactive — confirm each deletion"
    echo "  rm -rv /tmp/build-cache/        # Recursive verbose removal"
    echo "  find /tmp -name '*.tmp' -mtime +7 -delete   # Delete old temp files"
    echo ""
    echo "  Explanation:"
    echo "    Always use -i or -v in interactive sessions for safety."
    echo "    find ... -delete is safer than rm with globs because find shows"
    echo "    exactly what matches before you add -delete."
    echo "    NEVER run: rm -rf / or rm -rf * without double-checking your pwd."
    echo ""

    echo "--- Part D: Bulk rename with a loop ---"
    echo "Solution:"
    echo "  # Rename all .txt files to .md"
    echo "  for f in *.txt; do"
    echo "    mv -- \"\$f\" \"\${f%.txt}.md\""
    echo "  done"
    echo ""
    echo "  Explanation:"
    echo "    \${f%.txt} strips the .txt suffix using parameter expansion."
    echo "    '--' prevents filenames starting with '-' from being parsed as flags."
    echo "    Alternative: rename 's/\\.txt\$/.md/' *.txt (Perl rename, if installed)."
}

# === Exercise 2: Archive and Compression ===
# Problem: Create and extract archives using tar, gzip, zip, and xz.
exercise_2() {
    echo "=== Exercise 2: Archive and Compression (tar, gzip, zip, xz) ==="
    echo ""
    echo "Scenario: You need to package application logs for offsite backup,"
    echo "choosing the right compression for speed vs size tradeoffs."
    echo ""

    echo "--- Part A: tar with gzip compression ---"
    echo "Solution:"
    echo "  tar czf logs-backup.tar.gz /var/log/            # Create gzip-compressed archive"
    echo "  tar czf backup.tar.gz -C /opt/project .         # Archive from specific directory"
    echo "  tar tzf logs-backup.tar.gz                       # List contents without extracting"
    echo "  tar xzf logs-backup.tar.gz -C /restore/          # Extract to specific directory"
    echo ""
    echo "  Explanation:"
    echo "    c = create, z = gzip, f = filename, t = list, x = extract."
    echo "    -C changes directory before operating (avoids absolute paths in archive)."
    echo "    gzip is the standard balance of speed and compression ratio."
    echo ""

    echo "--- Part B: tar with xz for maximum compression ---"
    echo "Solution:"
    echo "  tar cJf archive.tar.xz /data/dataset/           # xz compression (best ratio)"
    echo "  tar cJf archive.tar.xz --exclude='*.tmp' /data/  # Exclude patterns"
    echo ""
    echo "  Explanation:"
    echo "    J = xz compression. Produces ~30% smaller files than gzip, but"
    echo "    is significantly slower. Ideal for long-term archival storage."
    echo "    Compression comparison: tar.xz < tar.bz2 < tar.gz (size)."
    echo "    Speed comparison: gzip >> bzip2 > xz (gzip is fastest)."
    echo ""

    echo "--- Part C: zip for cross-platform compatibility ---"
    echo "Solution:"
    echo "  zip -r project.zip /opt/project/                # Create zip recursively"
    echo "  zip -r project.zip /opt/project/ -x '*.git*'    # Exclude .git"
    echo "  unzip -l project.zip                             # List contents"
    echo "  unzip project.zip -d /restore/                   # Extract to directory"
    echo ""
    echo "  Explanation:"
    echo "    zip is the best choice when sharing with Windows/macOS users."
    echo "    -r = recursive. -x = exclude pattern."
    echo "    zip stores each file individually (random access), while tar"
    echo "    is a stream (must read sequentially)."
    echo ""

    echo "Verification:"
    echo "  file archive.tar.gz          # Confirm file type"
    echo "  du -h archive.tar.gz         # Check compressed size"
    echo "  tar tzf archive.tar.gz | wc -l  # Count files in archive"
}

# === Exercise 3: Disk Usage Analysis ===
# Problem: Analyze disk usage to identify space-consuming directories and files.
exercise_3() {
    echo "=== Exercise 3: Disk Usage Analysis (du, df, ncdu Patterns) ==="
    echo ""
    echo "Scenario: The monitoring system alerts that root filesystem is at 90%."
    echo "You need to quickly identify what is consuming the most space."
    echo ""

    echo "--- Part A: Check filesystem-level usage with df ---"
    echo "Solution:"
    echo "  df -h                # Human-readable sizes for all mounted filesystems"
    echo "  df -h /              # Check only root filesystem"
    echo "  df -h /var /home     # Check specific mount points"
    echo "  df -i /              # Check inode usage (can run out even with free space)"
    echo ""
    echo "  Explanation:"
    echo "    -h = human-readable (K, M, G instead of raw blocks)."
    echo "    -i = inode usage. A filesystem with 0% space used but 100% inodes"
    echo "    used cannot create new files. This happens with millions of tiny files."
    echo ""

    echo "--- Part B: Find space-consuming directories with du ---"
    echo "Solution:"
    echo "  du -sh /var/*                   # Summary of each item in /var"
    echo "  du -h --max-depth=1 /var | sort -rh | head -10   # Top 10 largest"
    echo "  du -sh /var/log /var/cache /tmp  # Check common culprits"
    echo ""
    echo "  Explanation:"
    echo "    -s = summary (total per argument, no subdirectory breakdown)."
    echo "    -h = human-readable sizes."
    echo "    --max-depth=1 shows only the first level of subdirectories."
    echo "    sort -rh = reverse human-numeric sort (largest first)."
    echo ""

    echo "--- Part C: Interactive exploration with ncdu ---"
    echo "Solution:"
    echo "  ncdu /var                # Interactive TUI disk usage browser"
    echo "  ncdu -x /               # Exclude other filesystems (-x = one filesystem)"
    echo "  ncdu -e -o report.json / # Export results for later analysis"
    echo ""
    echo "  Explanation:"
    echo "    ncdu (NCurses Disk Usage) provides a navigable, sorted view."
    echo "    Navigate with arrow keys, press 'd' to delete, 'q' to quit."
    echo "    -x prevents crossing filesystem boundaries (skips /proc, /sys, NFS)."
    echo "    If ncdu is not installed: sudo apt install ncdu (or dnf install ncdu)."
    echo ""

    echo "--- Part D: Quick wins for freeing space ---"
    echo "Solution:"
    echo "  sudo journalctl --vacuum-size=100M      # Trim systemd journal logs"
    echo "  sudo apt clean                           # Remove cached .deb packages"
    echo "  find /var/log -name '*.gz' -mtime +90 -delete  # Delete old compressed logs"
    echo "  find /tmp -type f -atime +30 -delete     # Clean old temp files"
    echo ""
    echo "  Explanation:"
    echo "    journalctl --vacuum-size keeps only 100MB of journal logs."
    echo "    apt clean removes /var/cache/apt/archives/ (safe, re-downloadable)."
    echo "    Old compressed logs (.gz) are typically rotated copies already replaced."
    echo ""

    # Safe read-only check on current system
    echo "--- Current disk usage on this system ---"
    if command -v df &>/dev/null; then
        df -h / 2>/dev/null | head -2
    fi
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
