# User and Group Management

**Previous**: [Permissions and Ownership](./05_Permissions_Ownership.md) | **Next**: [Process Management](./07_Process_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the structure and purpose of /etc/passwd, /etc/shadow, and /etc/group
2. Create, modify, and delete user accounts with useradd, usermod, and userdel
3. Manage groups using groupadd, groupmod, groupdel, and gpasswd
4. Switch between users with su and escalate privileges with sudo
5. Configure sudo permissions by editing /etc/sudoers safely with visudo
6. Query user and group information using id, groups, who, w, and last
7. Apply security best practices for password policies and account management

---

Linux is a multi-user operating system by design. Even a single-user workstation runs dozens of system accounts for services like web servers, databases, and mail. Properly managing users and groups is the foundation of access control -- it determines who can log in, what files they can access, and which commands they can run with elevated privileges.

## 1. User-Related Files

Linux stores user information in specific files.

### /etc/passwd

Stores user account information.

```bash
cat /etc/passwd | head -5
```

Output:
```
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
```

```
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
  │    │  │    │     │        │           │
  │    │  │    │     │        │           └── Login shell
  │    │  │    │     │        └── Home directory
  │    │  │    │     └── Description (GECOS)
  │    │  │    └── Primary group ID (GID)
  │    │  └── User ID (UID)
  │    └── Password (x = stored in shadow file)
  └── Username
```

### /etc/shadow

Stores encrypted passwords (readable only by root).

```bash
sudo cat /etc/shadow | head -3
```

Output:
```
root:$6$xxxx...:19000:0:99999:7:::
ubuntu:$6$yyyy...:19000:0:99999:7:::
```

```
ubuntu:$6$...:19000:0:99999:7:::
  │      │     │   │   │   │
  │      │     │   │   │   └── Password expiration warning days
  │      │     │   │   └── Maximum password age
  │      │     │   └── Minimum password age
  │      │     └── Last changed date (days since Jan 1, 1970)
  │      └── Encrypted password
  └── Username
```

### /etc/group

Stores group information.

```bash
cat /etc/group | head -5
```

Output:
```
root:x:0:
daemon:x:1:
ubuntu:x:1000:
sudo:x:27:ubuntu
developers:x:1001:alice,bob
```

```
developers:x:1001:alice,bob
    │      │  │      │
    │      │  │      └── Group members (additional members)
    │      │  └── Group ID (GID)
    │      └── Password (usually not used)
    └── Group name
```

---

## 2. User Management Commands

### useradd - Create User

```bash
# Basic creation
sudo useradd username

# Create with options (recommended)
sudo useradd -m -s /bin/bash -c "John Doe" john

# Key options
# -m : Create home directory
# -s : Specify login shell
# -c : Description (comment)
# -d : Specify home directory path
# -g : Primary group
# -G : Additional groups
# -u : Specify UID
```

```bash
# Create with multiple groups
sudo useradd -m -s /bin/bash -G sudo,developers newuser

# Set password
sudo passwd newuser
```

### adduser - Interactive User Creation (Ubuntu/Debian)

```bash
# Create user interactively (more convenient)
sudo adduser newuser
```

Output:
```
Adding user `newuser' ...
Adding new group `newuser' (1002) ...
Adding new user `newuser' (1002) with group `newuser' ...
Creating home directory `/home/newuser' ...
Copying files from `/etc/skel' ...
New password:
Retype new password:
passwd: password updated successfully
Full Name []: New User
Room Number []:
Work Phone []:
Home Phone []:
Other []:
Is the information correct? [Y/n] y
```

### usermod - Modify User

```bash
# Change shell
sudo usermod -s /bin/zsh username

# Change home directory
sudo usermod -d /home/newhome -m username

# Add to group (keep existing groups)
sudo usermod -aG sudo username
sudo usermod -aG docker,developers username

# Change username
sudo usermod -l newname oldname

# Lock account
sudo usermod -L username

# Unlock account
sudo usermod -U username
```

### userdel - Delete User

```bash
# Delete user only
sudo userdel username

# Delete with home directory and mail
sudo userdel -r username
```

### passwd - Password Management

```bash
# Change own password
passwd

# Change other user's password (root)
sudo passwd username

# Expire password (force change on next login)
sudo passwd -e username

# Lock password
sudo passwd -l username

# Unlock password
sudo passwd -u username

# Check password status
sudo passwd -S username
```

---

## 3. Group Management Commands

### groupadd - Create Group

```bash
# Create group
sudo groupadd developers

# Specify GID
sudo groupadd -g 2000 mygroup
```

### groupmod - Modify Group

```bash
# Change group name
sudo groupmod -n newname oldname

# Change GID
sudo groupmod -g 2001 groupname
```

### groupdel - Delete Group

```bash
sudo groupdel groupname
```

### gpasswd - Manage Group Members

```bash
# Add user to group
sudo gpasswd -a username groupname

# Remove user from group
sudo gpasswd -d username groupname

# Assign group administrator
sudo gpasswd -A adminuser groupname
```

---

## 4. User Switching

### su - Switch User

```bash
# Switch to another user
su username

# Switch to root
su -
su - root

# Switch with environment variables (recommended)
su - username

# Execute single command
su -c 'command' username
```

### sudo - Privilege Escalation

```bash
# Execute command with administrator privileges
sudo command

# Execute command as different user
sudo -u username command

# Open root shell
sudo -i

# Preserve environment variables
sudo -E command

# Clear sudo cache
sudo -k
```

---

## 5. sudo Configuration

### /etc/sudoers

File for configuring sudo permissions. **Always edit with visudo.**

```bash
sudo visudo
```

### Basic Format

```
# Per-user configuration
user   host=(run_as_user) command

# Per-group configuration (% prefix)
%group   host=(run_as_user) command
```

### Configuration Examples

```bash
# root has all privileges
root    ALL=(ALL:ALL) ALL

# sudo group members have all privileges
%sudo   ALL=(ALL:ALL) ALL

# Grant all privileges to specific user
john    ALL=(ALL:ALL) ALL

# Allow sudo without password
john    ALL=(ALL) NOPASSWD: ALL

# Allow specific commands only
backup  ALL=(ALL) /usr/bin/rsync, /usr/bin/tar

# Specific command without password
deploy  ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nginx
```

### /etc/sudoers.d/

You can manage configurations in separate files.

```bash
# Create file
sudo visudo -f /etc/sudoers.d/developers

# Content
%developers ALL=(ALL) NOPASSWD: /usr/bin/docker
```

---

## 6. User Information

### id - User ID Information

```bash
# Current user
id

# Specific user
id username
```

Output:
```
uid=1000(ubuntu) gid=1000(ubuntu) groups=1000(ubuntu),27(sudo),999(docker)
```

### groups - Group Membership

```bash
# Current user groups
groups

# Specific user groups
groups username
```

### who - Logged-in Users

```bash
# Currently logged-in users
who
```

Output:
```
ubuntu   pts/0        2024-01-23 10:00 (192.168.1.100)
```

### w - Detailed Login Information

```bash
w
```

Output:
```
 10:30:00 up 5 days,  3:45,  2 users,  load average: 0.00, 0.01, 0.05
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
ubuntu   pts/0    192.168.1.100    10:00    0.00s  0.03s  0.00s w
john     pts/1    192.168.1.101    10:15    5:00   0.01s  0.01s bash
```

### last - Login History

```bash
# Recent login history
last

# Specific user
last username

# Last 10 entries
last -n 10

# Reboot history
last reboot
```

### lastlog - Last Login

```bash
lastlog
```

---

## 7. System Users

| UID Range | Purpose |
|-----------|---------|
| 0 | root |
| 1-999 | System users |
| 1000+ | Regular users |

### Creating System Users

```bash
# System user (no login)
sudo useradd -r -s /usr/sbin/nologin serviceuser
```

### Common System Users

| User | Purpose |
|------|---------|
| root | System administrator |
| www-data | Web server |
| mysql | MySQL database |
| postgres | PostgreSQL |
| nobody | Minimal privilege processes |

---

## 8. Practical Examples

### Development Team Environment

```bash
# 1. Create developers group
sudo groupadd developers

# 2. Create developer accounts
sudo useradd -m -s /bin/bash -G developers alice
sudo useradd -m -s /bin/bash -G developers bob
sudo passwd alice
sudo passwd bob

# 3. Set up shared directory
sudo mkdir -p /projects/shared
sudo chgrp developers /projects/shared
sudo chmod 2775 /projects/shared

# 4. Grant sudo privileges (Docker commands only)
sudo visudo -f /etc/sudoers.d/developers
# %developers ALL=(ALL) NOPASSWD: /usr/bin/docker
```

### Web Developer Environment

```bash
# Web developer account
sudo useradd -m -s /bin/bash -G www-data,developers webdev
sudo passwd webdev

# Web directory permissions
sudo chown -R webdev:www-data /var/www/mysite
sudo chmod -R 775 /var/www/mysite
```

### Deployment-Only Account

```bash
# Deployment account (login with key only)
sudo useradd -m -s /bin/bash deploy
sudo mkdir -p /home/deploy/.ssh
sudo chmod 700 /home/deploy/.ssh

# SSH key setup
sudo touch /home/deploy/.ssh/authorized_keys
sudo chmod 600 /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh

# Limited sudo privileges
sudo visudo -f /etc/sudoers.d/deploy
# deploy ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart myapp
```

---

## 9. Security Best Practices

### Password Policy

```bash
# Edit /etc/login.defs
sudo vi /etc/login.defs
```

```
PASS_MAX_DAYS   90     # Maximum age
PASS_MIN_DAYS   7      # Minimum age
PASS_WARN_AGE   14     # Expiration warning days
PASS_MIN_LEN    12     # Minimum length
```

### Disable Direct root Login

```bash
# Disable root login via SSH
sudo vi /etc/ssh/sshd_config
# PermitRootLogin no

sudo systemctl restart sshd
```

### Lock Unnecessary Accounts

```bash
# Lock unused accounts
sudo passwd -l unuseduser

# Set shell to nologin
sudo usermod -s /usr/sbin/nologin unuseduser
```

---

## 10. Practice Exercises

### Exercise 1: Check User Information

```bash
# Current user information
id
groups
whoami

# Check /etc/passwd
grep $USER /etc/passwd

# Login history
last -n 5
```

### Exercise 2: User Creation and Deletion

```bash
# Create test user
sudo useradd -m -s /bin/bash -c "Test User" testuser
sudo passwd testuser

# Verify
id testuser
grep testuser /etc/passwd
ls -la /home/testuser

# Delete
sudo userdel -r testuser
```

### Exercise 3: Group Management

```bash
# Create group
sudo groupadd testgroup

# Add user
sudo usermod -aG testgroup $USER

# Verify (re-login required)
groups

# Delete group
sudo groupdel testgroup
```

### Exercise 4: sudo Testing

```bash
# Check sudo privileges
sudo -l

# Execute command with root privileges
sudo whoami

# Execute command as different user
sudo -u www-data whoami
```

---

**Previous**: [Permissions and Ownership](./05_Permissions_Ownership.md) | **Next**: [Process Management](./07_Process_Management.md)
