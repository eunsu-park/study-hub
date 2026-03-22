#!/bin/bash
# Exercises for Lesson 06: User and Group Management
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: User Lifecycle ===
# Problem: Create, modify, and remove user accounts with appropriate settings.
exercise_1() {
    echo "=== Exercise 1: User Lifecycle (useradd, usermod, userdel, passwd) ==="
    echo ""
    echo "Scenario: A new developer joins the team. Set up their account, configure"
    echo "it correctly, then clean up when they leave."
    echo ""

    echo "--- Part A: Create a user with full options ---"
    echo "Solution:"
    echo "  sudo useradd -m -s /bin/bash -c 'Alice Kim' -G developers,docker alice"
    echo ""
    echo "  Explanation:"
    echo "    -m = create home directory (/home/alice)"
    echo "    -s /bin/bash = set login shell (default may be /bin/sh)"
    echo "    -c 'Alice Kim' = GECOS comment field (full name)"
    echo "    -G developers,docker = supplementary groups (comma-separated)"
    echo "    Without -m, no home directory is created (common mistake)."
    echo ""

    echo "--- Part B: Set password and aging policy ---"
    echo "Solution:"
    echo "  sudo passwd alice                        # Set password interactively"
    echo "  sudo chage -M 90 alice                   # Password expires after 90 days"
    echo "  sudo chage -W 14 alice                   # Warn 14 days before expiry"
    echo "  sudo chage -m 1 alice                    # Minimum 1 day between changes"
    echo "  sudo chage -l alice                      # List current aging settings"
    echo ""
    echo "  Explanation:"
    echo "    -M = maximum days before password must be changed."
    echo "    -W = warning days before expiry."
    echo "    -m = minimum days between password changes (prevents rapid cycling)."
    echo "    chage -l shows all aging info: last change, expiry, account disable date."
    echo ""

    echo "--- Part C: Modify an existing user ---"
    echo "Solution:"
    echo "  sudo usermod -aG sudo alice              # Add to sudo group (keep existing groups)"
    echo "  sudo usermod -s /bin/zsh alice            # Change shell"
    echo "  sudo usermod -d /home/akim -m alice       # Move home directory"
    echo "  sudo usermod -l akim alice                # Rename login from alice to akim"
    echo ""
    echo "  Explanation:"
    echo "    -aG = APPEND to supplementary groups. Without -a, it REPLACES all groups!"
    echo "    This is the most common usermod mistake — forgetting -a removes the user"
    echo "    from all existing groups."
    echo "    -d with -m moves the home directory contents to the new location."
    echo ""

    echo "--- Part D: Remove a user (offboarding) ---"
    echo "Solution:"
    echo "  sudo userdel alice                       # Remove user, keep home directory"
    echo "  sudo userdel -r alice                    # Remove user AND home directory"
    echo "  sudo find / -user alice -ls 2>/dev/null  # Find orphaned files before deletion"
    echo ""
    echo "  Explanation:"
    echo "    -r removes home directory and mail spool. Use with caution."
    echo "    Best practice: find orphaned files BEFORE deleting the user."
    echo "    After deletion, files owned by the old UID show numeric IDs in ls."
    echo "    Consider: lock the account first (usermod -L), archive data, then delete."
    echo ""

    echo "Verification:"
    echo "  id alice                # Show UID, GID, and all groups"
    echo "  getent passwd alice     # Show passwd entry"
    echo "  groups alice            # List group memberships"
}

# === Exercise 2: Group Management and Membership ===
# Problem: Create and manage groups, configure group membership for team collaboration.
exercise_2() {
    echo "=== Exercise 2: Group Management and Membership ==="
    echo ""
    echo "Scenario: Set up group-based access control for a development team"
    echo "with separate groups for developers, QA, and DevOps."
    echo ""

    echo "--- Part A: Create and configure groups ---"
    echo "Solution:"
    echo "  sudo groupadd developers                 # Create group"
    echo "  sudo groupadd -g 2000 devops             # Create with specific GID"
    echo "  sudo groupadd -r monitoring              # System group (low GID range)"
    echo ""
    echo "  Explanation:"
    echo "    -g specifies GID (useful for consistency across servers)."
    echo "    -r creates a system group with a GID in the system range"
    echo "    (typically < 1000). System groups are for services, not users."
    echo ""

    echo "--- Part B: Manage group membership ---"
    echo "Solution:"
    echo "  sudo gpasswd -a alice developers         # Add alice to developers"
    echo "  sudo gpasswd -a bob developers           # Add bob to developers"
    echo "  sudo gpasswd -d alice developers         # Remove alice from developers"
    echo "  sudo gpasswd -M alice,bob,carol devops   # Set full member list at once"
    echo ""
    echo "  Explanation:"
    echo "    gpasswd -a is equivalent to 'usermod -aG group user' but with"
    echo "    clearer syntax (group-centric vs user-centric)."
    echo "    -M replaces the entire member list — use carefully."
    echo "    -d removes a single user from the group."
    echo ""

    echo "--- Part C: Primary vs supplementary groups ---"
    echo "Solution:"
    echo "  sudo usermod -g developers alice          # Change PRIMARY group"
    echo "  sudo usermod -aG devops,docker alice      # Add SUPPLEMENTARY groups"
    echo "  newgrp developers                         # Switch active group in current session"
    echo ""
    echo "  Explanation:"
    echo "    Primary group (one): determines default group ownership of new files."
    echo "    Supplementary groups (many): grant additional access permissions."
    echo "    -g sets primary, -aG adds supplementary."
    echo "    newgrp starts a new shell with a different active group."
    echo "    'id' shows both primary (gid=) and supplementary (groups=)."
    echo ""

    echo "--- Part D: Group administration and cleanup ---"
    echo "Solution:"
    echo "  sudo groupmod -n engineering developers    # Rename group"
    echo "  sudo groupdel oldteam                      # Delete group"
    echo "  getent group developers                    # View group entry and members"
    echo "  cat /etc/group | grep developers           # Same info from file directly"
    echo ""
    echo "  Explanation:"
    echo "    groupmod -n renames without changing GID (files keep same group)."
    echo "    groupdel fails if any user has the group as their primary group."
    echo "    /etc/group format: name:password:GID:member1,member2"
    echo "    getent uses nsswitch.conf (works with LDAP/NIS, not just local files)."
    echo ""

    echo "Verification:"
    echo "  getent group | grep -E 'developers|devops'  # Check both groups"
    echo "  id alice                                      # Verify alice's memberships"
}

# === Exercise 3: sudo Configuration and Best Practices ===
# Problem: Configure sudo access with proper sudoers entries.
exercise_3() {
    echo "=== Exercise 3: sudo Configuration and sudoers Best Practices ==="
    echo ""
    echo "Scenario: Implement least-privilege sudo access — developers can restart"
    echo "services, DBAs can manage databases, but neither has full root."
    echo ""

    echo "--- Part A: Edit sudoers safely ---"
    echo "Solution:"
    echo "  sudo visudo                    # ALWAYS use visudo (syntax-checks before saving)"
    echo "  sudo visudo -f /etc/sudoers.d/developers   # Edit drop-in file"
    echo ""
    echo "  Explanation:"
    echo "    NEVER edit /etc/sudoers directly with a text editor."
    echo "    visudo locks the file and validates syntax before saving."
    echo "    A syntax error in sudoers can lock everyone out of sudo."
    echo "    Drop-in files in /etc/sudoers.d/ are modular and manageable."
    echo ""

    echo "--- Part B: Grant limited sudo privileges ---"
    echo "Solution (sudoers entries):"
    echo ""
    cat << 'SUDOERS'
# /etc/sudoers.d/developers
# Allow developers to restart application services
%developers ALL=(root) NOPASSWD: /usr/bin/systemctl restart nginx, \
                                 /usr/bin/systemctl restart app.service, \
                                 /usr/bin/systemctl status *

# /etc/sudoers.d/dba
# Allow DBAs to manage PostgreSQL
%dba ALL=(postgres) NOPASSWD: /usr/bin/psql, \
                              /usr/bin/pg_dump, \
                              /usr/bin/pg_restore
SUDOERS
    echo ""
    echo "  Explanation:"
    echo "    %developers = group. ALL = any host. (root) = run as root."
    echo "    NOPASSWD: skips password prompt for these specific commands."
    echo "    Listing exact paths prevents users from running modified binaries."
    echo "    (postgres) lets DBAs run commands as the postgres user, not root."
    echo ""

    echo "--- Part C: Verify and test sudo access ---"
    echo "Solution:"
    echo "  sudo -l                              # List YOUR sudo privileges"
    echo "  sudo -l -U alice                     # List alice's sudo privileges"
    echo "  sudo -v                              # Validate (extend) sudo timeout"
    echo "  sudo -k                              # Kill sudo session (require password next time)"
    echo ""
    echo "  Explanation:"
    echo "    sudo -l is the safest way to verify what a user can do."
    echo "    sudo -v refreshes the credential cache without running a command."
    echo "    sudo -k clears the cache (good practice when leaving a terminal)."
    echo ""

    echo "--- Part D: Security best practices ---"
    echo "Solution:"
    echo "  Defaults    timestamp_timeout=5        # Re-auth after 5 minutes (default: 15)"
    echo "  Defaults    log_output                 # Log all sudo session output"
    echo "  Defaults    logfile=/var/log/sudo.log   # Custom sudo log file"
    echo "  Defaults    passwd_tries=3              # Lock after 3 failed attempts"
    echo "  Defaults    secure_path=\"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\""
    echo ""
    echo "  Explanation:"
    echo "    timestamp_timeout controls how long sudo remembers your password."
    echo "    log_output creates session transcripts for audit trails."
    echo "    secure_path overrides the user's PATH to prevent trojan commands."
    echo "    These go at the top of sudoers or in /etc/sudoers.d/00-defaults."
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
