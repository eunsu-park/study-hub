#!/usr/bin/env bash
# =============================================================================
# 03_user_group_management.sh - User, Group, and Permission Management
#
# PURPOSE: Demonstrates Linux user/group administration, file permission models,
#          sudo configuration, and password policies. All operations are
#          simulated (dry-run) to be safe on any system.
#
# USAGE:
#   ./03_user_group_management.sh [--users|--groups|--perms|--sudo|--all]
#
# MODES:
#   --users   User creation and password policy demonstrations
#   --groups  Group management and membership
#   --perms   File permission models (rwx, ACL, sticky/setuid/setgid)
#   --sudo    Sudoers configuration patterns
#   --all     Run all sections (default)
#
# CONCEPTS COVERED:
#   - /etc/passwd and /etc/shadow structure
#   - Primary vs supplementary groups
#   - Numeric (chmod 755) vs symbolic (chmod u+x) permissions
#   - Special bits: setuid, setgid, sticky
#   - Password aging and account expiration
#   - Sudoers file best practices
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color codes for readability
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Why: DRY_RUN prevents actual system changes. Educational scripts should
# demonstrate commands without modifying the host. Set DRY_RUN=false only
# in a disposable VM or container.
DRY_RUN="${DRY_RUN:-true}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
section() {
    echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"
}

explain() {
    echo -e "${GREEN}[INFO]${RESET} $1"
}

show_cmd() {
    # Why: Displaying the command before "executing" it teaches the reader
    # exactly what they would type, without risking unintended side effects.
    echo -e "${YELLOW}[CMD]${RESET} $1"
}

run_or_dry() {
    show_cmd "$1"
    if [[ "${DRY_RUN}" == "false" ]]; then
        eval "$1"
    else
        echo -e "  ${RED}(dry-run — skipped)${RESET}"
    fi
}

# ---------------------------------------------------------------------------
# 1. User Management
# ---------------------------------------------------------------------------
demo_users() {
    section "1. User Management"

    explain "Creating a new user with a home directory and default shell:"
    # Why: -m creates the home directory, -s sets the login shell.
    # Without -m, the user has no home dir and many tools break.
    run_or_dry "useradd -m -s /bin/bash -c 'Developer Account' devuser"

    explain "Viewing /etc/passwd entry structure (real users on this system):"
    # Why: Understanding passwd fields is fundamental to Linux admin.
    # Format: username:x:UID:GID:GECOS:home:shell
    echo -e "${BOLD}  username : x : UID : GID : GECOS : home_dir : shell${RESET}"
    # Show the current user's entry as a safe example
    grep "^$(whoami):" /etc/passwd 2>/dev/null || echo "  (entry not found — may be in LDAP/NIS)"

    explain "Password aging policies (chage command):"
    show_cmd "chage -l devuser"
    # Why: Password aging enforces rotation. Critical fields:
    #   -M max_days: force password change every N days
    #   -m min_days: prevent changing password too frequently
    #   -W warn_days: warn N days before expiration
    #   -I inactive_days: disable account N days after password expiry
    run_or_dry "chage -M 90 -m 7 -W 14 -I 30 devuser"
    explain "Above sets: max 90 days, min 7 days, warn 14 days, inactive 30 days"

    explain "Locking and unlocking a user account:"
    # Why: Locking (passwd -l) prepends '!' to the password hash in /etc/shadow,
    # preventing login without deleting the account. Useful for investigations.
    run_or_dry "passwd -l devuser"
    run_or_dry "passwd -u devuser"

    explain "Setting account expiration date:"
    run_or_dry "usermod -e 2026-12-31 devuser"

    explain "Deleting a user and their home directory:"
    # Why: -r removes home dir and mail spool. Without it, orphaned files
    # remain and may be assigned to a future user with the same UID.
    run_or_dry "userdel -r devuser"
}

# ---------------------------------------------------------------------------
# 2. Group Management
# ---------------------------------------------------------------------------
demo_groups() {
    section "2. Group Management"

    explain "Creating a new group:"
    run_or_dry "groupadd developers"

    explain "Adding a user to a supplementary group:"
    # Why: -aG appends the group. Without -a, usermod REPLACES all
    # supplementary groups, potentially locking the user out of sudo.
    run_or_dry "usermod -aG developers devuser"

    explain "Listing current user's groups:"
    show_cmd "groups"
    groups 2>/dev/null || id -Gn

    explain "Viewing /etc/group structure:"
    echo -e "${BOLD}  group_name : x : GID : member_list${RESET}"
    # Why: The fourth field is a comma-separated list of supplementary members.
    # The primary group is NOT listed here — it is in /etc/passwd GID field.
    head -5 /etc/group 2>/dev/null || echo "  (cannot read /etc/group)"

    explain "Primary vs supplementary groups:"
    echo "  - Primary group: assigned at creation (newgrp changes it temporarily)"
    echo "  - Supplementary groups: additional memberships (up to ~65k)"
    echo "  - New files inherit the creator's primary group (or dir's group with setgid)"

    explain "Changing a file's group ownership:"
    run_or_dry "chgrp developers /tmp/project_file"

    explain "Removing a group:"
    run_or_dry "groupdel developers"
}

# ---------------------------------------------------------------------------
# 3. File Permissions
# ---------------------------------------------------------------------------
demo_permissions() {
    section "3. File Permission Model"

    explain "Permission bits: read(4) write(2) execute(1)"
    echo "  -rwxr-xr-- = owner(rwx=7) group(r-x=5) other(r--=4) => 754"
    echo ""

    explain "Numeric (octal) vs symbolic chmod:"
    show_cmd "chmod 755 script.sh       # rwxr-xr-x"
    show_cmd "chmod u+x,g-w,o= file    # add exec for owner, remove write for group, none for other"

    explain "Default permissions — umask:"
    show_cmd "umask"
    # Why: umask SUBTRACTS permissions from defaults (666 for files, 777 for dirs).
    # umask 022 → files=644, dirs=755. Setting umask 077 is more secure
    # (only owner can access new files).
    echo "  Current umask: $(umask)"
    echo "  umask 022 → files: 644, dirs: 755"
    echo "  umask 077 → files: 600, dirs: 700 (restrictive)"

    explain "Special permission bits:"
    echo ""
    echo "  ${BOLD}setuid (4xxx)${RESET}: File runs as owner, not invoker"
    show_cmd "chmod u+s /usr/bin/passwd  # Users can change their own password"
    echo "  Why: passwd must write /etc/shadow (owned by root)"
    echo ""
    echo "  ${BOLD}setgid (2xxx)${RESET}: File runs as group; on dirs, new files inherit group"
    show_cmd "chmod g+s /srv/shared/     # All new files get the dir's group"
    echo "  Why: Teams sharing a directory need consistent group ownership"
    echo ""
    echo "  ${BOLD}sticky bit (1xxx)${RESET}: Only owner/root can delete files in directory"
    show_cmd "chmod +t /tmp              # Prevents users from deleting each other's files"
    echo "  Why: /tmp is world-writable but users must not delete others' files"

    explain "POSIX ACLs for fine-grained control:"
    # Why: Standard rwx gives only one owner and one group. ACLs let you
    # grant per-user or per-group permissions without changing ownership.
    show_cmd "setfacl -m u:alice:rw /srv/data/report.csv"
    show_cmd "getfacl /srv/data/report.csv"
}

# ---------------------------------------------------------------------------
# 4. Sudo Configuration
# ---------------------------------------------------------------------------
demo_sudo() {
    section "4. Sudoers Configuration"

    explain "Sudoers file best practices:"
    echo "  1. Never edit /etc/sudoers directly — use visudo"
    echo "  2. Drop files in /etc/sudoers.d/ for modularity"
    echo "  3. Principle of least privilege: grant specific commands"
    echo ""

    explain "Example sudoers entries:"
    # Why: Each example shows a progressively more restrictive policy.
    # The goal is to grant minimal privileges needed for the task.
    echo "  # Full root access (avoid in production)"
    echo '  devuser  ALL=(ALL:ALL) ALL'
    echo ""
    echo "  # Password-less restart of a specific service"
    echo '  deploy   ALL=(root) NOPASSWD: /usr/bin/systemctl restart myapp.service'
    echo ""
    echo "  # Group-based access to docker"
    echo '  %docker  ALL=(root) NOPASSWD: /usr/bin/docker'
    echo ""
    echo "  # Command alias for network diagnostics"
    echo '  Cmnd_Alias NETDIAG = /usr/sbin/tcpdump, /usr/bin/traceroute, /usr/sbin/ss'
    echo '  %netops   ALL=(root) NOPASSWD: NETDIAG'

    explain "Validating sudoers syntax:"
    show_cmd "visudo -c -f /etc/sudoers.d/devuser"
    # Why: A syntax error in sudoers can lock everyone out of sudo.
    # visudo -c validates without applying changes.

    explain "Auditing sudo usage:"
    show_cmd "journalctl _COMM=sudo --since '1 hour ago'"
    echo "  Each sudo invocation is logged with user, command, and timestamp."
}

# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}=======================================${RESET}"
    echo -e "${BOLD} Linux User & Group Management Demo${RESET}"
    echo -e "${BOLD}=======================================${RESET}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${RED}Running in DRY-RUN mode (no system changes).${RESET}"
        echo -e "Set DRY_RUN=false in a disposable VM to execute commands.\n"
    fi

    local mode="${1:---all}"

    case "${mode}" in
        --users)  demo_users ;;
        --groups) demo_groups ;;
        --perms)  demo_permissions ;;
        --sudo)   demo_sudo ;;
        --all)
            demo_users
            demo_groups
            demo_permissions
            demo_sudo
            ;;
        *)
            echo "Usage: $0 [--users|--groups|--perms|--sudo|--all]"
            exit 1
            ;;
    esac

    echo -e "\n${GREEN}${BOLD}Done.${RESET}"
}

main "$@"
