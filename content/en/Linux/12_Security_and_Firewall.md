# Security and Firewall

**Previous**: [System Monitoring](./11_System_Monitoring.md) | **Next**: [Advanced systemd](./13_Systemd_Advanced.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the principle of least privilege and describe Linux security layers
2. Harden SSH configuration by disabling root login, using key authentication, and limiting access
3. Configure UFW (Ubuntu) to allow, deny, and rate-limit network traffic
4. Configure firewalld (CentOS/RHEL) using zones, services, and rich rules
5. Understand the basics of SELinux (CentOS) and AppArmor (Ubuntu) mandatory access controls
6. Install and configure fail2ban to block brute-force attacks automatically
7. Set up unattended security updates for both Ubuntu and CentOS
8. Perform a security audit checking users, services, open ports, and file permissions

---

Security is not a feature you add at the end -- it is a mindset you apply from the start. A Linux server exposed to the internet will be probed by automated scanners within minutes of going online. This lesson covers the essential hardening steps that every server should have in place: locking down SSH, configuring firewalls, enabling intrusion prevention, and performing regular security audits.

## 1. Security Basic Principles

### Principle of Least Privilege

```
┌─────────────────────────────────────────────────────────┐
│                    Security Layers                       │
├─────────────────────────────────────────────────────────┤
│  1. Physical Security - Server room access control       │
│  2. Network Security - Firewall, VPN                     │
│  3. Host Security - OS configuration, patches            │
│  4. Application Security - Vulnerability management      │
│  5. Data Security - Encryption, backup                   │
└─────────────────────────────────────────────────────────┘
```

### Basic Security Checklist

- [ ] Disable unnecessary services
- [ ] Change default ports (SSH, etc.)
- [ ] Strong password policy
- [ ] Regular security updates
- [ ] Log monitoring
- [ ] Firewall configuration
- [ ] Use SSH key authentication

---

## 2. SSH Security Configuration

### sshd_config Settings

```bash
sudo vi /etc/ssh/sshd_config
```

### Recommended Settings

```bash
# Change port (default 22 → other port)
Port 2222

# Disable root login
PermitRootLogin no

# Disable password authentication (key only)
PasswordAuthentication no

# Disallow empty passwords
PermitEmptyPasswords no

# Allow specific users only
AllowUsers ubuntu deploy

# Allow specific groups only
AllowGroups sshusers

# Limit login attempts
MaxAuthTries 3

# Idle timeout
ClientAliveInterval 300
ClientAliveCountMax 2

# Disable X11 forwarding
X11Forwarding no

# Use protocol 2 only (usually default)
Protocol 2
```

### Apply Configuration

```bash
# Validate configuration
sudo sshd -t

# Restart service
sudo systemctl restart sshd
```

### SSH Key Management

```bash
# Generate key (ed25519 recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Set key permissions (required)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/authorized_keys
```

---

## 3. Firewall - UFW (Ubuntu)

UFW (Uncomplicated Firewall) is Ubuntu's default firewall.

### Basic Commands

```bash
# Check status
sudo ufw status
sudo ufw status verbose
sudo ufw status numbered

# Enable/disable
sudo ufw enable
sudo ufw disable

# Set default policy
sudo ufw default deny incoming    # Deny incoming (default)
sudo ufw default allow outgoing   # Allow outgoing (default)
```

### Add Rules

```bash
# Allow port
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443

# Allow port range
sudo ufw allow 6000:6010/tcp

# Allow by service name
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https

# Allow from specific IP only
sudo ufw allow from 192.168.1.100
sudo ufw allow from 192.168.1.100 to any port 22

# Allow subnet
sudo ufw allow from 192.168.1.0/24

# Specify TCP/UDP
sudo ufw allow 53/tcp
sudo ufw allow 53/udp
```

### Delete Rules

```bash
# Delete by rule number
sudo ufw status numbered
sudo ufw delete 2

# Delete rule directly
sudo ufw delete allow 80
```

### Advanced Settings

```bash
# Rate limiting (DoS prevention)
sudo ufw limit ssh    # Limit SSH connections (6 in 30 seconds)

# Logging
sudo ufw logging on
sudo ufw logging high

# Specific interface
sudo ufw allow in on eth0 to any port 80
```

### Typical Server Configuration

```bash
# Default policy
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH (use changed port if applicable)
sudo ufw allow 2222/tcp

# Web server
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable
sudo ufw enable
```

---

## 4. Firewall - firewalld (CentOS/RHEL)

firewalld is the default firewall for CentOS/RHEL.

### Basic Commands

```bash
# Check status
sudo firewall-cmd --state
sudo systemctl status firewalld

# Enable/disable
sudo systemctl start firewalld
sudo systemctl stop firewalld
sudo systemctl enable firewalld

# Reload configuration
sudo firewall-cmd --reload
```

### Zone Concept

| Zone | Description |
|------|-------------|
| drop | Deny all connections |
| block | Deny connection + ICMP response |
| public | Public (default) |
| external | External (NAT) |
| dmz | DMZ |
| work | Work |
| home | Home |
| internal | Internal |
| trusted | Allow all connections |

```bash
# Check current zone
sudo firewall-cmd --get-default-zone

# List zones
sudo firewall-cmd --get-zones

# Change zone
sudo firewall-cmd --set-default-zone=public
```

### Add Rules

```bash
# Allow service
sudo firewall-cmd --add-service=ssh --permanent
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --add-service=https --permanent

# Allow port
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --add-port=3000-3010/tcp --permanent

# Allow specific IP
sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.100" accept' --permanent

# Specific IP to specific port
sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="22" protocol="tcp" accept' --permanent

# Apply configuration
sudo firewall-cmd --reload
```

### Delete Rules

```bash
# Remove service
sudo firewall-cmd --remove-service=http --permanent

# Remove port
sudo firewall-cmd --remove-port=8080/tcp --permanent

# Apply
sudo firewall-cmd --reload
```

### Check Configuration

```bash
# Current configuration
sudo firewall-cmd --list-all

# Service list
sudo firewall-cmd --list-services

# Port list
sudo firewall-cmd --list-ports
```

---

## 5. SELinux (CentOS/RHEL)

### Check Status

```bash
# Current status
getenforce
sestatus
```

### Modes

| Mode | Description |
|------|-------------|
| Enforcing | Policy enforced (default) |
| Permissive | Log only |
| Disabled | Disabled |

### Change Mode

```bash
# Temporary change (reverts on reboot)
sudo setenforce 0    # Permissive
sudo setenforce 1    # Enforcing

# Permanent change
sudo vi /etc/selinux/config
# SELINUX=enforcing → SELINUX=permissive
```

### SELinux Troubleshooting

```bash
# Check denial logs
sudo ausearch -m avc -ts recent

# Analyze problems (requires audit2why)
sudo ausearch -m avc | audit2why

# Allow port example
sudo semanage port -a -t http_port_t -p tcp 8080
```

---

## 6. AppArmor (Ubuntu)

### Check Status

```bash
# Status
sudo aa-status

# Profile list
ls /etc/apparmor.d/
```

### Modes

```bash
# Profile enforce mode
sudo aa-enforce /etc/apparmor.d/usr.sbin.nginx

# Profile complain mode (log only)
sudo aa-complain /etc/apparmor.d/usr.sbin.nginx

# Disable profile
sudo aa-disable /etc/apparmor.d/usr.sbin.nginx
```

---

## 7. Fail2ban

fail2ban monitors logs to block malicious attempts.

### Installation

```bash
# Ubuntu
sudo apt install fail2ban

# CentOS
sudo dnf install fail2ban
```

### Basic Configuration

```bash
# Copy configuration file
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo vi /etc/fail2ban/jail.local
```

### Configuration Example

```ini
[DEFAULT]
# Ban time (seconds)
bantime = 3600

# Monitoring time (seconds)
findtime = 600

# Maximum retry attempts
maxretry = 5

# Email notification
destemail = admin@example.com
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh,2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400
```

### Management Commands

```bash
# Start/stop
sudo systemctl start fail2ban
sudo systemctl enable fail2ban

# Check status
sudo fail2ban-client status
sudo fail2ban-client status sshd

# Check banned IPs
sudo fail2ban-client status sshd | grep "Banned IP"

# Unban IP
sudo fail2ban-client set sshd unbanip 192.168.1.100

# Ban IP manually
sudo fail2ban-client set sshd banip 192.168.1.100
```

---

## 8. Security Updates

### Ubuntu/Debian

```bash
# Check for updates
sudo apt update
apt list --upgradable

# Security updates only
sudo apt upgrade -y

# Configure automatic updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

### CentOS/RHEL

```bash
# Check for updates
sudo dnf check-update

# Security updates only
sudo dnf upgrade --security

# Configure automatic updates
sudo dnf install dnf-automatic
sudo systemctl enable --now dnf-automatic.timer
```

---

## 9. Security Audit Checklist

### User Audit

```bash
# Accounts without password
sudo awk -F: '($2 == "") {print $1}' /etc/shadow

# UID 0 accounts (other than root)
awk -F: '($3 == 0) {print $1}' /etc/passwd

# Recent login failures
sudo lastb | head -20

# Users with sudo privileges
grep -Po '^sudo.+:\K.*$' /etc/group
```

### Service Audit

```bash
# Running services
systemctl list-units --type=service --state=running

# Open ports
ss -tuln

# Check unnecessary services
systemctl list-unit-files --type=service | grep enabled
```

### File Permission Audit

```bash
# World-writable files
find / -type f -perm -002 2>/dev/null

# SUID files
find / -perm -4000 2>/dev/null

# SGID files
find / -perm -2000 2>/dev/null

# Files without owner
find / -nouser -o -nogroup 2>/dev/null
```

---

## 10. Practice Exercises

### Exercise 1: Harden SSH

```bash
# Backup current configuration
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

# Change configuration (e.g., disable root login)
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# Validate configuration
sudo sshd -t

# Restart
sudo systemctl restart sshd
```

### Exercise 2: Firewall Setup (Ubuntu)

```bash
# Current status
sudo ufw status

# Default policy
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Enable
sudo ufw enable

# Verify
sudo ufw status verbose
```

### Exercise 3: Firewall Setup (CentOS)

```bash
# Current status
sudo firewall-cmd --list-all

# Allow SSH
sudo firewall-cmd --add-service=ssh --permanent

# Apply
sudo firewall-cmd --reload

# Verify
sudo firewall-cmd --list-services
```

### Exercise 4: fail2ban Setup

```bash
# Install
sudo apt install fail2ban    # Ubuntu
# sudo dnf install fail2ban  # CentOS

# Basic configuration
sudo systemctl start fail2ban
sudo systemctl enable fail2ban

# Check status
sudo fail2ban-client status sshd
```

### Exercise 5: Security Audit

```bash
# Check open ports
ss -tuln

# Login failure history
sudo lastb | head -10

# Check security updates
apt list --upgradable 2>/dev/null | grep -i security

# Check SUID files
find /usr/bin -perm -4000 2>/dev/null
```

---

**Previous**: [System Monitoring](./11_System_Monitoring.md) | **Next**: [Advanced systemd](./13_Systemd_Advanced.md)
