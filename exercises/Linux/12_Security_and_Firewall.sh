#!/bin/bash
# Exercises for Lesson 12: Security and Firewall
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: SSH Hardening ===
# Problem: Harden SSH server configuration to follow security best practices
#          and reduce the attack surface.
exercise_1() {
    echo "=== Exercise 1: SSH Hardening ==="
    echo ""
    echo "Scenario: Your server is exposed to the internet and receiving brute-force"
    echo "SSH login attempts. Harden the SSH configuration to minimize risk."
    echo ""

    echo "--- Part A: sshd_config Best Practices ---"
    echo "Solution: /etc/ssh/sshd_config"
    cat << 'SSHD'
  # Disable root login (force users to sudo)
  PermitRootLogin no

  # Disable password authentication (key-only)
  PasswordAuthentication no
  PubkeyAuthentication yes

  # Limit authentication attempts
  MaxAuthTries 3
  MaxSessions 5

  # Disable unused authentication methods
  ChallengeResponseAuthentication no
  KerberosAuthentication no
  GSSAPIAuthentication no

  # Use SSH Protocol 2 only (Protocol 1 is insecure)
  Protocol 2

  # Limit SSH access to specific users/groups
  AllowUsers deploy admin
  # Or: AllowGroups ssh-users

  # Change default port (security through obscurity, not a real defense alone)
  Port 2222

  # Disconnect idle sessions (300 seconds = 5 minutes)
  ClientAliveInterval 60
  ClientAliveCountMax 5

  # Restrict key exchange and cipher algorithms to strong ones
  KexAlgorithms curve25519-sha256@libssh.org
  Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com
  MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com

  # Disable X11 and TCP forwarding if not needed
  X11Forwarding no
  AllowTcpForwarding no

  # Log level for audit trail
  LogLevel VERBOSE
SSHD
    echo ""
    echo "  Explanation:"
    echo "    PermitRootLogin no forces attackers to guess both username AND key"
    echo "    PasswordAuthentication no eliminates brute-force password attacks entirely"
    echo "    ClientAliveInterval sends keepalive probes; CountMax * Interval = max idle time"
    echo "    VERBOSE logging records key fingerprints used for login (audit trail)"
    echo ""

    echo "--- Part B: Validate and Apply Configuration ---"
    echo "Solution:"
    echo "  # Test configuration syntax BEFORE restarting (prevents lockout!)"
    echo "  sudo sshd -t"
    echo ""
    echo "  # If syntax is OK, restart the SSH service"
    echo "  sudo systemctl restart sshd"
    echo ""
    echo "  # IMPORTANT: Keep your current session open and test a NEW connection"
    echo "  # before closing, in case of misconfiguration"
    echo "  ssh -p 2222 deploy@your-server   # Test from another terminal"
    echo ""
    echo "  Explanation:"
    echo "    sshd -t validates the config file syntax and exits"
    echo "    Never close your active SSH session before verifying the new config works"
    echo "    If locked out, use console access (cloud provider console, IPMI, etc.)"
    echo ""

    echo "--- Part C: SSH Key Security Audit ---"
    echo "Solution:"
    echo "  # Check permissions on SSH files"
    echo "  ls -la ~/.ssh/"
    echo "  # Expected: drwx------ (700) for .ssh directory"
    echo "  #           -rw------- (600) for private keys and authorized_keys"
    echo "  #           -rw-r--r-- (644) for public keys and config"
    echo ""
    echo "  # Fix permissions if needed"
    echo "  chmod 700 ~/.ssh"
    echo "  chmod 600 ~/.ssh/id_ed25519 ~/.ssh/authorized_keys"
    echo "  chmod 644 ~/.ssh/id_ed25519.pub ~/.ssh/config"
    echo ""
    echo "  # List authorized keys with fingerprints"
    echo "  ssh-keygen -lf ~/.ssh/authorized_keys"
    echo ""
    echo "  # Remove old/unknown keys from authorized_keys"
    echo "  # Review each entry — remove any you don't recognize"
    echo ""
    echo "  Explanation:"
    echo "    SSH refuses to use keys with overly permissive permissions"
    echo "    Regular audits of authorized_keys prevent unauthorized access"
    echo "    ssh-keygen -l shows the key fingerprint for identification"
}

# === Exercise 2: Firewall Rules ===
# Problem: Configure firewall rules using both ufw (Debian/Ubuntu) and
#          firewalld (RHEL/CentOS) for a web server.
exercise_2() {
    echo "=== Exercise 2: Firewall Rules ==="
    echo ""
    echo "Scenario: Configure a firewall for a web server that needs HTTP/HTTPS"
    echo "access from the public, SSH from a management subnet only, and"
    echo "database access from application servers only."
    echo ""

    echo "--- Part A: UFW (Uncomplicated Firewall) - Debian/Ubuntu ---"
    echo "Solution:"
    echo "  # Reset to default (deny incoming, allow outgoing)"
    echo "  sudo ufw default deny incoming"
    echo "  sudo ufw default allow outgoing"
    echo ""
    echo "  # Allow HTTP and HTTPS from anywhere"
    echo "  sudo ufw allow 80/tcp comment 'HTTP'"
    echo "  sudo ufw allow 443/tcp comment 'HTTPS'"
    echo ""
    echo "  # Allow SSH only from management subnet"
    echo "  sudo ufw allow from 10.0.100.0/24 to any port 2222 proto tcp comment 'SSH from mgmt'"
    echo ""
    echo "  # Allow PostgreSQL only from app server"
    echo "  sudo ufw allow from 10.0.1.10 to any port 5432 proto tcp comment 'DB from app1'"
    echo "  sudo ufw allow from 10.0.1.11 to any port 5432 proto tcp comment 'DB from app2'"
    echo ""
    echo "  # Rate limit SSH to prevent brute force"
    echo "  sudo ufw limit 2222/tcp comment 'SSH rate limit'"
    echo ""
    echo "  # Enable the firewall"
    echo "  sudo ufw enable"
    echo ""
    echo "  # View rules with numbers"
    echo "  sudo ufw status numbered verbose"
    echo ""
    echo "  Explanation:"
    echo "    'default deny incoming' blocks all inbound traffic not explicitly allowed"
    echo "    'from 10.0.100.0/24' restricts to a specific subnet (CIDR notation)"
    echo "    'limit' allows max 6 connections per 30 seconds from a single IP"
    echo "    Rules are evaluated in order; first match wins"
    echo ""

    echo "--- Part B: firewalld - RHEL/CentOS/Fedora ---"
    echo "Solution:"
    echo "  # Check current zone and rules"
    echo "  sudo firewall-cmd --get-active-zones"
    echo "  sudo firewall-cmd --list-all"
    echo ""
    echo "  # Add services (permanent + immediate)"
    echo "  sudo firewall-cmd --permanent --add-service=http"
    echo "  sudo firewall-cmd --permanent --add-service=https"
    echo ""
    echo "  # Add custom port"
    echo "  sudo firewall-cmd --permanent --add-port=2222/tcp"
    echo ""
    echo "  # Rich rules for source-specific access"
    echo "  sudo firewall-cmd --permanent --add-rich-rule='rule family=\"ipv4\" source address=\"10.0.1.0/24\" port port=\"5432\" protocol=\"tcp\" accept'"
    echo ""
    echo "  # Reload to apply permanent rules"
    echo "  sudo firewall-cmd --reload"
    echo ""
    echo "  Explanation:"
    echo "    --permanent makes rules persist across reboots (without it, rules are temporary)"
    echo "    firewalld uses zones (public, internal, dmz, etc.) for different trust levels"
    echo "    Rich rules provide iptables-like granularity within the firewalld framework"
    echo "    Always --reload after adding --permanent rules"
    echo ""

    echo "--- Part C: Verify and Monitor Firewall ---"
    echo "Solution:"
    echo "  # UFW"
    echo "  sudo ufw status verbose                  # Show all active rules"
    echo "  sudo ufw show added                      # Show rules as ufw commands"
    echo ""
    echo "  # firewalld"
    echo "  sudo firewall-cmd --list-all-zones       # All zones and their rules"
    echo "  sudo firewall-cmd --query-port=443/tcp   # Check if specific port is open"
    echo ""
    echo "  # Test from external host"
    echo "  nmap -p 80,443,2222,5432 your-server-ip  # Port scan (run from outside)"
    echo ""
    echo "  # Check iptables (underlying framework)"
    echo "  sudo iptables -L -n --line-numbers        # List all rules with line numbers"
    echo ""
    echo "  Explanation:"
    echo "    Both ufw and firewalld are frontends for iptables/nftables"
    echo "    Always verify from outside the server (nmap, curl, telnet)"
    echo "    iptables -L shows the actual kernel-level rules being enforced"
}

# === Exercise 3: fail2ban Configuration ===
# Problem: Set up fail2ban to automatically block brute-force attacks
#          and configure custom rules for application logs.
exercise_3() {
    echo "=== Exercise 3: fail2ban Configuration and Intrusion Detection ==="
    echo ""
    echo "Scenario: Protect services from brute-force attacks using fail2ban,"
    echo "and set up basic intrusion detection monitoring."
    echo ""

    echo "--- Part A: fail2ban Installation and SSH Jail ---"
    echo "Solution:"
    echo "  # Install"
    echo "  sudo apt install fail2ban     # Debian/Ubuntu"
    echo "  sudo yum install fail2ban     # RHEL/CentOS (EPEL required)"
    echo ""
    echo "  # Create local override (never edit jail.conf directly)"
    echo "  sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local"
    echo ""
    cat << 'F2B'
  # /etc/fail2ban/jail.local
  [DEFAULT]
  bantime  = 3600       # Ban for 1 hour (seconds)
  findtime = 600        # Look at last 10 minutes of logs
  maxretry = 3          # Ban after 3 failures within findtime
  banaction = iptables-multiport
  action = %(action_mwl)s    # Ban + send email with whois + log lines

  [sshd]
  enabled  = true
  port     = 2222            # Match your actual SSH port
  logpath  = /var/log/auth.log    # Debian/Ubuntu
  # logpath = /var/log/secure     # RHEL/CentOS
  maxretry = 3
  bantime  = 86400           # 24 hours for SSH (stricter)
F2B
    echo ""
    echo "  Explanation:"
    echo "    jail.local overrides jail.conf and survives package upgrades"
    echo "    bantime = how long the IP stays blocked"
    echo "    findtime = window in which maxretry failures trigger a ban"
    echo "    action_mwl = ban + mail + whois lookup + relevant log lines"
    echo ""

    echo "--- Part B: Custom fail2ban Jail for Nginx ---"
    echo "Solution:"
    cat << 'F2B_NGINX'
  # /etc/fail2ban/jail.local (add this section)
  [nginx-http-auth]
  enabled  = true
  port     = http,https
  logpath  = /var/log/nginx/error.log
  maxretry = 5

  [nginx-botsearch]
  enabled  = true
  port     = http,https
  logpath  = /var/log/nginx/access.log
  maxretry = 2
  bantime  = 86400

  # Custom filter for repeated 404s (scanner detection)
  # /etc/fail2ban/filter.d/nginx-404.conf
  [Definition]
  failregex = ^<HOST> - .* "(GET|POST|HEAD) .* HTTP/.*" 404
  ignoreregex =
F2B_NGINX
    echo ""
    echo "  Explanation:"
    echo "    nginx-http-auth catches failed HTTP authentication attempts"
    echo "    nginx-botsearch catches requests to non-existent paths (scanners)"
    echo "    Custom filters use regex with <HOST> placeholder for the attacker IP"
    echo "    failregex must match the log format of your application"
    echo ""

    echo "--- Part C: fail2ban Management and Monitoring ---"
    echo "Solution:"
    echo "  # Start and enable"
    echo "  sudo systemctl enable --now fail2ban"
    echo ""
    echo "  # Check jail status"
    echo "  sudo fail2ban-client status                   # List active jails"
    echo "  sudo fail2ban-client status sshd              # Show banned IPs for sshd"
    echo ""
    echo "  # Manually ban/unban"
    echo "  sudo fail2ban-client set sshd banip 1.2.3.4   # Manually ban an IP"
    echo "  sudo fail2ban-client set sshd unbanip 1.2.3.4 # Unban an IP"
    echo ""
    echo "  # Test a filter against a log file (dry run)"
    echo "  fail2ban-regex /var/log/auth.log /etc/fail2ban/filter.d/sshd.conf"
    echo ""
    echo "  # Check fail2ban's own log"
    echo "  tail -f /var/log/fail2ban.log"
    echo ""
    echo "  Explanation:"
    echo "    fail2ban-regex is essential for testing custom filters before deployment"
    echo "    Always test filters to avoid false positives that lock out legitimate users"
    echo "    fail2ban-client status shows currently banned IPs and ban count"
    echo "    Use 'unbanip' carefully; check if the IP is truly legitimate"
    echo ""

    echo "--- Verification ---"
    echo "  # Simulate failed SSH login to test (from a test IP, NOT your own!):"
    echo "  # ssh -p 2222 nonexistent@your-server   # Will fail 3 times, then IP is banned"
    echo "  sudo fail2ban-client status sshd         # Verify the ban"
    echo "  sudo iptables -L f2b-sshd -n             # See the iptables rule"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
