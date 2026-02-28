#!/bin/bash
# Exercises for Lesson 20: Kernel Management
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Module Management ===
# Problem: Blacklist the nouveau driver (assuming NVIDIA proprietary driver usage).
exercise_1() {
    echo "=== Exercise 1: Kernel Module Management ==="
    echo ""
    echo "Scenario: Blacklist the open-source 'nouveau' GPU driver to use"
    echo "the proprietary NVIDIA driver instead."
    echo ""

    echo "--- Step 1: Create the blacklist configuration ---"
    echo "  echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf"
    echo "  echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf"
    echo ""
    echo "  Why two lines?"
    echo "    'blacklist nouveau'          - Prevents modprobe from auto-loading nouveau"
    echo "    'options nouveau modeset=0'  - Disables kernel modesetting as a fallback"
    echo "    Both are needed because blacklist alone doesn't prevent all loading paths."
    echo ""

    echo "--- Step 2: Update initramfs to apply at boot ---"
    echo "  # Ubuntu/Debian:"
    echo "  sudo update-initramfs -u"
    echo ""
    echo "  # RHEL/CentOS/Fedora:"
    echo "  sudo dracut -f"
    echo ""
    echo "  Why: The initial RAM filesystem (initramfs) is loaded before the root filesystem."
    echo "  If nouveau is compiled into initramfs, it loads before modprobe.d rules apply."
    echo "  Rebuilding initramfs ensures the blacklist takes effect early in boot."
    echo ""

    echo "--- Step 3: Verify the configuration ---"
    echo "  cat /etc/modprobe.d/blacklist-nouveau.conf"
    echo ""

    echo "--- Step 4: Verify after reboot ---"
    echo "  lsmod | grep nouveau    # Should return nothing (no output = success)"
    echo "  lsmod | grep nvidia     # Should show nvidia modules loaded"
    echo ""

    # Safe read-only check on current system
    echo "--- Current module status on this system ---"
    if command -v lsmod &>/dev/null; then
        if lsmod 2>/dev/null | grep -q nouveau; then
            echo "  nouveau is currently LOADED"
        else
            echo "  nouveau is NOT loaded (either blacklisted or not applicable)"
        fi
        if lsmod 2>/dev/null | grep -q nvidia; then
            echo "  nvidia driver is LOADED"
        else
            echo "  nvidia driver is NOT loaded"
        fi
    else
        echo "  (lsmod not available on this system)"
    fi
    echo ""

    echo "Additional module management commands:"
    echo "  lsmod                       # List all loaded modules"
    echo "  modinfo <module>            # Show module details (version, params, depends)"
    echo "  sudo modprobe <module>      # Load a module (resolves dependencies)"
    echo "  sudo modprobe -r <module>   # Unload a module"
    echo "  sudo rmmod <module>         # Force-unload a module (no dependency check)"
}

# === Exercise 2: GRUB Configuration ===
# Problem: Configure GRUB with 10s timeout, second entry as default, 4GB memory limit, quiet boot.
exercise_2() {
    echo "=== Exercise 2: GRUB Configuration ==="
    echo ""
    echo "Scenario: Customize GRUB bootloader settings for specific requirements."
    echo ""

    echo "Solution: /etc/default/grub"
    echo ""
    cat << 'GRUB'
# GRUB_DEFAULT: Which menu entry to boot by default
# 0 = first entry, 1 = second entry, "saved" = last chosen
GRUB_DEFAULT=1

# GRUB_TIMEOUT: Seconds to wait before auto-booting default entry
GRUB_TIMEOUT=10

# GRUB_TIMEOUT_STYLE: How to display the menu
# menu = always show, hidden = hide unless key pressed, countdown = show timer
GRUB_TIMEOUT_STYLE=menu

# GRUB_CMDLINE_LINUX_DEFAULT: Kernel parameters for normal boot entries
# quiet = suppress most boot messages
# splash = show graphical splash screen
# mem=4G = limit usable memory to 4GB (useful for testing memory constraints)
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash mem=4G"

# GRUB_CMDLINE_LINUX: Kernel parameters for ALL entries (including recovery)
GRUB_CMDLINE_LINUX=""
GRUB
    echo ""

    echo "To apply GRUB changes:"
    echo "  # Ubuntu/Debian:"
    echo "  sudo update-grub"
    echo ""
    echo "  # RHEL/CentOS (BIOS):"
    echo "  sudo grub2-mkconfig -o /boot/grub2/grub.cfg"
    echo ""
    echo "  # RHEL/CentOS (UEFI):"
    echo "  sudo grub2-mkconfig -o /boot/efi/EFI/centos/grub.cfg"
    echo ""

    echo "Common kernel parameters explained:"
    echo "  quiet           - Suppress kernel boot messages (cleaner boot)"
    echo "  splash          - Show graphical boot splash"
    echo "  mem=4G          - Limit memory (testing, legacy apps)"
    echo "  nomodeset       - Disable kernel mode setting (GPU troubleshooting)"
    echo "  init=/bin/bash  - Boot directly to shell (emergency recovery)"
    echo "  single          - Boot to single-user/rescue mode"
    echo "  rd.break        - Break into initramfs (password reset)"
    echo "  selinux=0       - Disable SELinux (temporary troubleshooting only)"
    echo ""

    echo "To see current kernel parameters:"
    if [ -f /proc/cmdline ]; then
        echo "  cat /proc/cmdline"
        echo "  Current: $(cat /proc/cmdline)"
    else
        echo "  cat /proc/cmdline"
        echo "  (/proc/cmdline not available on this system)"
    fi
}

# === Exercise 3: sysctl Web Server Optimization ===
# Problem: Write sysctl settings optimized for a web server.
exercise_3() {
    echo "=== Exercise 3: sysctl Web Server Optimization ==="
    echo ""
    echo "Scenario: Tune kernel parameters for a high-traffic web server."
    echo ""

    echo "Solution: /etc/sysctl.d/99-webserver.conf"
    echo ""
    cat << 'SYSCTL'
# ==================================================
# Web Server Kernel Tuning
# File: /etc/sysctl.d/99-webserver.conf
# ==================================================

# --- Connection Backlog ---
# Maximum number of connections queued for acceptance by listen()
# Default: 4096. Web servers with high concurrency need much more.
net.core.somaxconn = 65535

# Maximum SYN requests queued (half-open connections)
# Protects against SYN flood while allowing legitimate bursts
net.ipv4.tcp_max_syn_backlog = 65535

# --- File Descriptors ---
# System-wide limit on open files. Each TCP connection = 1 file descriptor.
# 2M handles ~2 million concurrent connections + files
# Default: ~100,000. Must increase for high-concurrency servers.
fs.file-max = 2097152

# --- TCP Tuning ---
# Time (seconds) to hold FIN-WAIT-2 state before closing
# Default: 60. Lower = faster cleanup of dead connections
net.ipv4.tcp_fin_timeout = 15

# Allow reuse of TIME-WAIT sockets for new connections
# Critical for servers making many outbound connections (reverse proxies)
net.ipv4.tcp_tw_reuse = 1

# TCP keepalive: detect dead connections
# Send first probe after 300s of idle (default: 7200s = 2 hours!)
net.ipv4.tcp_keepalive_time = 300
# Send 5 probes (default: 9)
net.ipv4.tcp_keepalive_probes = 5
# 15 seconds between probes (default: 75s)
net.ipv4.tcp_keepalive_intvl = 15

# --- Memory Buffers ---
# Maximum socket receive/send buffer sizes (bytes)
# 16MB allows large window scaling for high-bandwidth connections
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216

# TCP auto-tuning buffer sizes: min, default, max (bytes)
# min=4K (tiny connections), default=12M (normal), max=16M (large transfers)
net.ipv4.tcp_rmem = 4096 12582912 16777216
net.ipv4.tcp_wmem = 4096 12582912 16777216
SYSCTL
    echo ""

    echo "To apply sysctl settings:"
    echo "  sudo sysctl -p /etc/sysctl.d/99-webserver.conf"
    echo ""
    echo "To verify a specific setting:"
    echo "  sysctl net.core.somaxconn"
    echo ""

    # Show current values for key parameters (safe read-only)
    echo "--- Current values on this system ---"
    for param in net.core.somaxconn fs.file-max net.ipv4.tcp_fin_timeout; do
        value=$(sysctl -n "$param" 2>/dev/null)
        if [ -n "$value" ]; then
            echo "  $param = $value"
        fi
    done
    echo ""

    echo "Important notes:"
    echo "  - Also increase per-process limits in /etc/security/limits.conf:"
    echo "    * soft nofile 65535"
    echo "    * hard nofile 65535"
    echo "  - nginx/Apache may have their own connection limits (worker_connections, MaxClients)"
    echo "  - Monitor with: ss -s (socket statistics), netstat -tn | wc -l (connection count)"
    echo "  - Use '99-' prefix to ensure this file loads after other sysctl configs"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
