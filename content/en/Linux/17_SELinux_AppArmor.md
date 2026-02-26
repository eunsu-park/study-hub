# SELinux and AppArmor

**Previous**: [Storage Management](./16_Storage_Management.md) | **Next**: [Log Management](./18_Log_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between Discretionary Access Control (DAC) and Mandatory Access Control (MAC)
2. Configure SELinux modes and manage security contexts for files, processes, and ports
3. Use SELinux booleans to toggle policy features without writing custom modules
4. Diagnose SELinux denials using audit2why and resolve them with restorecon or custom policies
5. Write and manage AppArmor profiles to confine application access
6. Generate AppArmor profiles automatically using aa-genprof and aa-logprof

---

## Table of Contents

1. [Mandatory Access Control Overview](#1-mandatory-access-control-overview)
2. [SELinux Basics](#2-selinux-basics)
3. [SELinux Policy Management](#3-selinux-policy-management)
4. [SELinux Troubleshooting](#4-selinux-troubleshooting)
5. [AppArmor Basics](#5-apparmor-basics)
6. [AppArmor Profiles](#6-apparmor-profiles)
7. [Practical Scenarios](#7-practical-scenarios)

---

Traditional Unix permissions (DAC) have a critical weakness: if an attacker compromises a process running as root, they own the entire system. Mandatory Access Control closes this gap by enforcing policies that even root cannot override. SELinux and AppArmor are the two dominant MAC implementations in Linux, and they are enabled by default on most enterprise distributions. Knowing how to configure and troubleshoot them is the difference between a secure production server and one that gets disabled to "make things work."

## 1. Mandatory Access Control Overview

### DAC vs MAC

```
┌─────────────────────────────────────────────────────────────┐
│                   Access Control Comparison                  │
├─────────────────────────────────────────────────────────────┤
│  DAC (Discretionary Access Control)                         │
│  - Traditional Unix permission model                        │
│  - File owner determines permissions                        │
│  - Managed with chmod, chown                                │
│  - root can bypass all restrictions                         │
├─────────────────────────────────────────────────────────────┤
│  MAC (Mandatory Access Control)                             │
│  - System policy determines access                          │
│  - Users cannot change policies                             │
│  - Implemented with SELinux, AppArmor                       │
│  - Even root is restricted by policy                        │
└─────────────────────────────────────────────────────────────┘
```

### Security Module Comparison

| Feature | SELinux | AppArmor |
|---------|---------|----------|
| Base Distribution | RHEL/CentOS/Fedora | Ubuntu/Debian/SUSE |
| Approach | Label-based | Path-based |
| Complexity | High | Low |
| Granularity | Very fine | Medium |
| Learning Curve | Steep | Gentle |
| Default Policy | Comprehensive | Limited |

---

## 2. SELinux Basics

### SELinux Modes

```bash
# Check current mode
getenforce
# Enforcing, Permissive, or Disabled

# Check detailed status
sestatus

# Temporary mode change (restored on reboot)
sudo setenforce 0  # Permissive
sudo setenforce 1  # Enforcing
```

### Permanent Mode Change

```bash
# Edit /etc/selinux/config
# RHEL/CentOS
sudo vi /etc/selinux/config
```

```ini
# /etc/selinux/config
SELINUX=enforcing     # enforcing, permissive, disabled
SELINUXTYPE=targeted  # targeted, minimum, mls
```

### SELinux Context

Every file, process, and port is assigned a security context:

```
user:role:type:level
user_u:role_r:type_t:s0
```

```bash
# Check file context
ls -Z /var/www/html/
# -rw-r--r--. root root unconfined_u:object_r:httpd_sys_content_t:s0 index.html

# Check process context
ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0    12345 ?  00:00:01 httpd

# Check user context
id -Z
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023
```

### Common Types

| Type | Description |
|------|-------------|
| `httpd_t` | Apache web server process |
| `httpd_sys_content_t` | Web content files |
| `mysqld_t` | MySQL process |
| `sshd_t` | SSH daemon |
| `user_home_t` | User home directory |
| `tmp_t` | Temporary files |

---

## 3. SELinux Policy Management

### Changing File Context

```bash
# Temporary change (restored on relabeling)
chcon -t httpd_sys_content_t /var/www/custom/index.html

# Recursive directory change
chcon -R -t httpd_sys_content_t /var/www/custom/

# Copy context from another file
chcon --reference=/var/www/html/index.html /var/www/custom/index.html
```

### Permanent Context Settings

```bash
# Add context rule to policy
sudo semanage fcontext -a -t httpd_sys_content_t "/srv/www(/.*)?"

# Apply policy
sudo restorecon -Rv /srv/www

# List context rules
sudo semanage fcontext -l | grep httpd

# Delete rule
sudo semanage fcontext -d "/srv/www(/.*)?"
```

### SELinux Booleans

Booleans are switches that turn specific features of SELinux policy on or off:

```bash
# List all booleans
getsebool -a

# Check specific boolean
getsebool httpd_can_network_connect

# Temporary change
sudo setsebool httpd_can_network_connect on

# Permanent change (-P option)
sudo setsebool -P httpd_can_network_connect on

# Search booleans
getsebool -a | grep httpd
```

### Common Boolean Examples

```bash
# Web server related
httpd_can_network_connect      # Allow external network connections
httpd_can_network_connect_db   # Allow DB connections
httpd_can_sendmail            # Allow sending mail
httpd_enable_homedirs         # Allow access to user home directories

# FTP related
ftpd_anon_write              # Allow anonymous write
ftpd_full_access             # Full filesystem access

# Others
samba_enable_home_dirs       # Samba home directory sharing
```

### Port Context

```bash
# Check port labels
sudo semanage port -l | grep http
# http_port_t                    tcp      80, 81, 443, 488, 8008, 8009, 8443, 9000

# Add new port
sudo semanage port -a -t http_port_t -p tcp 8080

# Delete port
sudo semanage port -d -t http_port_t -p tcp 8080

# Modify port
sudo semanage port -m -t http_port_t -p tcp 8888
```

---

## 4. SELinux Troubleshooting

### Checking Audit Logs

```bash
# Check SELinux denial logs
sudo ausearch -m avc -ts recent

# Logs related to specific service
sudo ausearch -m avc -c httpd

# Convert to readable format
sudo ausearch -m avc -ts recent | audit2why
```

### Using audit2why

```bash
# Analyze denial reasons
sudo cat /var/log/audit/audit.log | audit2why

# Example output:
# type=AVC msg=audit(...): avc:  denied  { read } for  pid=1234
# comm="httpd" name="index.html" dev="sda1" ino=12345
# scontext=system_u:system_r:httpd_t:s0
# tcontext=unconfined_u:object_r:user_home_t:s0 tclass=file
#
# Was caused by:
#   Missing type enforcement (TE) allow rule.
```

### Generating Policy with audit2allow

```bash
# Generate allow rules (review only)
sudo ausearch -m avc -ts recent | audit2allow

# Compile as local module
sudo ausearch -m avc -ts recent | audit2allow -M mypolicy

# Install module
sudo semodule -i mypolicy.pp

# Check installed modules
sudo semodule -l | grep mypolicy

# Remove module
sudo semodule -r mypolicy
```

### Using sealert (GUI/Detailed Analysis)

```bash
# Requires setroubleshoot package
sudo yum install setroubleshoot-server

# Run analysis
sudo sealert -a /var/log/audit/audit.log

# Check real-time alerts
sudo sealert -l "*"
```

### Common Problem Resolution

```bash
# Problem: Web server cannot read files
# 1. Check context
ls -Z /var/www/html/problem_file

# 2. Fix context
sudo restorecon -v /var/www/html/problem_file

# Problem: Cannot use custom port
# 1. Check current port label
sudo semanage port -l | grep 8080

# 2. Add port
sudo semanage port -a -t http_port_t -p tcp 8080

# Problem: Network connection denied
# 1. Check related boolean
getsebool -a | grep httpd_can_network

# 2. Enable boolean
sudo setsebool -P httpd_can_network_connect on
```

---

## 5. AppArmor Basics

### Checking AppArmor Status

```bash
# Ubuntu/Debian
sudo aa-status

# or
sudo apparmor_status
```

Example output:
```
apparmor module is loaded.
38 profiles are loaded.
36 profiles are in enforce mode.
   /snap/snapd/19457/usr/lib/snapd/snap-confine
   /usr/bin/evince
   ...
2 profiles are in complain mode.
   /usr/sbin/cups-browsed
   /usr/sbin/cupsd
```

### AppArmor Modes

```bash
# Enforce mode: Block policy violations
sudo aa-enforce /etc/apparmor.d/usr.sbin.nginx

# Complain mode: Only log violations (no blocking)
sudo aa-complain /etc/apparmor.d/usr.sbin.nginx

# Disable profile
sudo aa-disable /etc/apparmor.d/usr.sbin.nginx

# Reload profile
sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.nginx
```

### Profile Locations

```bash
# System profiles
ls /etc/apparmor.d/

# Main files
/etc/apparmor.d/usr.sbin.nginx    # Nginx profile
/etc/apparmor.d/usr.sbin.mysqld   # MySQL profile
/etc/apparmor.d/abstractions/     # Shared rules
/etc/apparmor.d/tunables/         # Variable definitions
```

---

## 6. AppArmor Profiles

### Profile Structure

```
#include <tunables/global>

/path/to/program {
  #include <abstractions/base>

  # File access rules
  /etc/myapp.conf r,
  /var/log/myapp.log w,
  /usr/lib/myapp/** r,

  # Network rules
  network inet stream,

  # Execution rules
  /usr/bin/helper ix,
}
```

### Permission Flags

| Flag | Meaning |
|------|---------|
| `r` | Read |
| `w` | Write |
| `a` | Append |
| `k` | File lock |
| `l` | Link |
| `m` | Memory map execute |
| `x` | Execute |
| `ix` | Execute with same profile |
| `px` | Execute with different profile |
| `ux` | Execute unconfined |
| `Px` | px + environment scrubbing |
| `Ux` | ux + environment scrubbing |

### Profile Writing Example

```bash
# /etc/apparmor.d/usr.local.bin.myapp
#include <tunables/global>

/usr/local/bin/myapp {
  #include <abstractions/base>
  #include <abstractions/nameservice>

  # Read config files
  /etc/myapp/** r,

  # Data directory
  /var/lib/myapp/ r,
  /var/lib/myapp/** rw,

  # Log files
  /var/log/myapp/ r,
  /var/log/myapp/** rw,
  owner /var/log/myapp/*.log w,

  # Runtime files
  /run/myapp.pid rw,
  /run/myapp.sock rw,

  # Libraries
  /usr/lib/myapp/** rm,

  # Network access
  network inet tcp,
  network inet udp,

  # System call restrictions
  deny @{PROC}/** w,
  deny /sys/** w,

  # Child processes
  /usr/bin/logger Px,
}
```

### Automatic Profile Generation

```bash
# Generate profile with aa-genprof
sudo aa-genprof /usr/local/bin/myapp

# Run the program and perform typical operations
# aa-genprof monitors access and creates profile

# Update existing profile with aa-logprof
sudo aa-logprof
```

### Using Abstractions

```bash
# Common rules in /etc/apparmor.d/abstractions/
# base          - Basic system access
# nameservice   - DNS, NSS, etc.
# authentication - PAM, shadow, etc.
# apache2-common - Apache common rules
# mysql         - MySQL client access
# php           - PHP related access
```

Using in profiles:
```
#include <abstractions/base>
#include <abstractions/nameservice>
```

---

## 7. Practical Scenarios

### Scenario 1: Web Server Custom Directory (SELinux)

```bash
# Problem: 403 error when serving web content from /data/www

# 1. Check current context
ls -Zd /data/www
# drwxr-xr-x. root root unconfined_u:object_r:default_t:s0 /data/www

# 2. Set correct context
sudo semanage fcontext -a -t httpd_sys_content_t "/data/www(/.*)?"
sudo restorecon -Rv /data/www

# 3. Verify
ls -Zd /data/www
# drwxr-xr-x. root root unconfined_u:object_r:httpd_sys_content_t:s0 /data/www
```

### Scenario 2: PHP Application DB Connection (SELinux)

```bash
# Problem: PHP cannot connect to remote MySQL

# 1. Check logs
sudo ausearch -m avc -c httpd | audit2why

# 2. Check boolean
getsebool httpd_can_network_connect_db
# httpd_can_network_connect_db --> off

# 3. Enable boolean
sudo setsebool -P httpd_can_network_connect_db on
```

### Scenario 3: Nginx Custom Port (AppArmor)

```bash
# /etc/apparmor.d/local/nginx
# File for local customization

# Allow additional port
network inet stream,

# Allow additional paths
/data/nginx/** r,
/var/log/nginx-custom/ rw,
/var/log/nginx-custom/** rw,
```

```bash
# Reload profile
sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.nginx
```

### Scenario 4: Docker and SELinux

```bash
# Mounting host volume in Docker container

# Method 1: z option (shared label)
docker run -v /data:/data:z myimage

# Method 2: Z option (private label)
docker run -v /data:/data:Z myimage

# Method 3: Manual label assignment
sudo chcon -Rt svirt_sandbox_file_t /data
docker run -v /data:/data myimage
```

### Scenario 5: Creating New Service Profile (AppArmor)

```bash
# 1. Start in complain mode
sudo aa-complain /usr/local/bin/newservice

# 2. Run service and test all features

# 3. Update profile from logs
sudo aa-logprof

# 4. Switch to enforce mode
sudo aa-enforce /usr/local/bin/newservice

# 5. Test
```

---

## Practice Problems

### Problem 1: SELinux Context

What commands should you use in the following situations?
- Permanently set `/opt/webapp` directory as web server content
- Allow Apache to use port 8443
- Allow httpd to access user home directories

### Problem 2: AppArmor Profile

The `/usr/local/bin/backup.sh` script performs the following tasks:
- Read `/etc/`
- Write to `/var/backup/`
- Execute `rsync`
- Network access to TCP port 22

Write an AppArmor profile for this script.

### Problem 3: Troubleshooting

A web application is not working in SELinux Enforcing mode:
1. List the steps to diagnose the problem
2. What tools should you use?

---

## Answers

### Problem 1 Answer

```bash
# Web content setup
sudo semanage fcontext -a -t httpd_sys_content_t "/opt/webapp(/.*)?"
sudo restorecon -Rv /opt/webapp

# Add port
sudo semanage port -a -t http_port_t -p tcp 8443

# Allow home directory access
sudo setsebool -P httpd_enable_homedirs on
```

### Problem 2 Answer

```
#include <tunables/global>

/usr/local/bin/backup.sh {
  #include <abstractions/base>
  #include <abstractions/bash>

  # Read config
  /etc/** r,

  # Backup directory
  /var/backup/ r,
  /var/backup/** rw,

  # Execute rsync
  /usr/bin/rsync Px,

  # SSH network
  network inet stream,
  network inet6 stream,
}
```

### Problem 3 Answer

```bash
# 1. Check SELinux logs
sudo ausearch -m avc -ts recent

# 2. Analyze cause
sudo ausearch -m avc -ts recent | audit2why

# 3. Detailed analysis (if setroubleshoot installed)
sudo sealert -a /var/log/audit/audit.log

# 4. Apply solution
# - Context issue: restorecon, semanage fcontext
# - Boolean issue: setsebool
# - Port issue: semanage port
# - Policy needed: Create custom module with audit2allow
```

---

## References

- [SELinux User Guide (Red Hat)](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/using_selinux/index)
- [AppArmor Wiki](https://gitlab.com/apparmor/apparmor/-/wikis/home)
- [SELinux Project Wiki](https://selinuxproject.org/page/Main_Page)
- `man semanage`, `man restorecon`, `man audit2why`
- `man apparmor`, `man aa-status`, `man apparmor.d`

---

**Previous**: [Storage Management](./16_Storage_Management.md) | **Next**: [Log Management](./18_Log_Management.md)
