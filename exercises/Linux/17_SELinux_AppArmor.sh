#!/bin/bash
# Exercises for Lesson 17: SELinux and AppArmor
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: SELinux Context ===
# Problem: Set /opt/webapp as web server content, allow Apache to use port 8443,
#          and allow httpd to access user home directories.
exercise_1() {
    echo "=== Exercise 1: SELinux Context ==="
    echo ""
    echo "Scenario: Configure SELinux contexts and booleans for a web server."
    echo ""

    echo "--- Part A: Set /opt/webapp as web server content ---"
    echo "Solution:"
    echo "  sudo semanage fcontext -a -t httpd_sys_content_t \"/opt/webapp(/.*)?\"   # Add file context rule"
    echo "  sudo restorecon -Rv /opt/webapp                                          # Apply context recursively"
    echo ""
    echo "Why: SELinux labels files with security contexts. Apache (httpd) can only"
    echo "read files labeled httpd_sys_content_t. The semanage command adds a persistent"
    echo "rule, and restorecon applies it to existing files."
    echo ""

    echo "--- Part B: Allow Apache to use port 8443 ---"
    echo "Solution:"
    echo "  sudo semanage port -a -t http_port_t -p tcp 8443   # Register port with SELinux"
    echo ""
    echo "Why: SELinux restricts which ports each service can bind to. By default, httpd"
    echo "can only use standard HTTP ports (80, 443, etc.). Adding 8443 to http_port_t"
    echo "allows Apache to listen on this non-standard port."
    echo ""

    echo "--- Part C: Allow httpd to access user home directories ---"
    echo "Solution:"
    echo "  sudo setsebool -P httpd_enable_homedirs on   # Enable boolean persistently (-P)"
    echo ""
    echo "Why: SELinux booleans are on/off switches for predefined policy rules."
    echo "httpd_enable_homedirs controls whether Apache can read files in /home/*."
    echo "The -P flag makes it persist across reboots."
    echo ""

    # Safe verification commands we can actually run (if SELinux is available)
    if command -v getenforce &>/dev/null; then
        echo "--- Current SELinux Status ---"
        getenforce 2>/dev/null || echo "(SELinux not available on this system)"
    else
        echo "(SELinux tools not installed on this system - commands shown above for reference)"
    fi
}

# === Exercise 2: AppArmor Profile ===
# Problem: Write an AppArmor profile for /usr/local/bin/backup.sh that can
#          read /etc/, write to /var/backup/, execute rsync, and access TCP port 22.
exercise_2() {
    echo "=== Exercise 2: AppArmor Profile ==="
    echo ""
    echo "Scenario: Create a security profile that confines a backup script."
    echo ""

    echo "Solution: AppArmor profile for /usr/local/bin/backup.sh"
    echo ""
    cat << 'PROFILE'
#include <tunables/global>

/usr/local/bin/backup.sh {
  #include <abstractions/base>       # Base system access (libc, etc.)
  #include <abstractions/bash>       # Bash shell access (needed for .sh scripts)

  # Read configuration files
  /etc/** r,                         # Read any file under /etc/ (recursive)

  # Backup directory access
  /var/backup/ r,                    # Read the backup directory itself
  /var/backup/** rw,                 # Read and write files within it

  # Execute rsync for remote sync
  /usr/bin/rsync Px,                 # Execute rsync with its own profile (Px)
                                     # Px = transition to rsync's profile
                                     # Use 'ix' instead if rsync has no profile

  # Network access for SSH (rsync over SSH)
  network inet stream,              # Allow IPv4 TCP connections
  network inet6 stream,             # Allow IPv6 TCP connections
}
PROFILE
    echo ""
    echo "Why each rule matters:"
    echo "  - 'r' = read only; prevents the script from modifying system configs"
    echo "  - 'rw' = read-write; backup dir needs both for creating backup files"
    echo "  - 'Px' = execute with profile transition; limits rsync's own access"
    echo "  - 'network stream' = TCP only; no UDP prevents unexpected protocols"
    echo ""
    echo "To install this profile:"
    echo "  sudo cp backup.sh.apparmor /etc/apparmor.d/usr.local.bin.backup.sh"
    echo "  sudo apparmor_parser -r /etc/apparmor.d/usr.local.bin.backup.sh"
    echo "  sudo aa-enforce /usr/local/bin/backup.sh"
}

# === Exercise 3: SELinux Troubleshooting ===
# Problem: Diagnose and fix a web application not working in SELinux Enforcing mode.
exercise_3() {
    echo "=== Exercise 3: SELinux Troubleshooting ==="
    echo ""
    echo "Scenario: A web application fails in SELinux Enforcing mode but works in Permissive."
    echo ""

    echo "Step-by-step diagnostic procedure:"
    echo ""
    echo "--- Step 1: Check SELinux denials in the audit log ---"
    echo "  sudo ausearch -m avc -ts recent"
    echo ""
    echo "  Why: AVC (Access Vector Cache) messages record every SELinux denial."
    echo "  '-ts recent' limits output to the last 10 minutes."
    echo ""

    echo "--- Step 2: Analyze the root cause ---"
    echo "  sudo ausearch -m avc -ts recent | audit2why"
    echo ""
    echo "  Why: audit2why translates raw AVC messages into human-readable explanations"
    echo "  and suggests specific fixes (boolean to set, context to change, etc.)."
    echo ""

    echo "--- Step 3: Get detailed analysis with sealert (if setroubleshoot is installed) ---"
    echo "  sudo sealert -a /var/log/audit/audit.log"
    echo ""
    echo "  Why: sealert provides comprehensive analysis with confidence scores"
    echo "  and ranked solution suggestions."
    echo ""

    echo "--- Step 4: Apply the appropriate fix based on diagnosis ---"
    echo ""
    echo "  If CONTEXT issue (wrong file labels):"
    echo "    sudo semanage fcontext -a -t <correct_type> '/path/to/files(/.*)?'"
    echo "    sudo restorecon -Rv /path/to/files"
    echo ""
    echo "  If BOOLEAN issue (policy toggle needed):"
    echo "    sudo setsebool -P <boolean_name> on"
    echo ""
    echo "  If PORT issue (non-standard port):"
    echo "    sudo semanage port -a -t <port_type> -p tcp <port_number>"
    echo ""
    echo "  If CUSTOM POLICY needed (no existing rule covers this):"
    echo "    sudo ausearch -m avc -ts recent | audit2allow -M mypolicy"
    echo "    sudo semodule -i mypolicy.pp"
    echo ""

    echo "--- Step 5: Verify the fix ---"
    echo "  sudo setenforce 1          # Ensure Enforcing mode"
    echo "  # Test the web application"
    echo "  curl -I http://localhost    # Verify it works"
    echo "  sudo ausearch -m avc -ts recent   # Should show no new denials"
    echo ""

    echo "Key tools summary:"
    echo "  ausearch    - Search audit logs for SELinux events"
    echo "  audit2why   - Explain why access was denied"
    echo "  audit2allow - Generate policy module to allow denied access"
    echo "  sealert     - User-friendly analysis (requires setroubleshoot)"
    echo "  restorecon  - Reset file contexts to policy defaults"
    echo "  semanage    - Manage SELinux policy (contexts, booleans, ports)"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
