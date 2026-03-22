# Permissions and Ownership

**Previous**: [Text Processing](./04_Text_Processing.md) | **Next**: [User and Group Management](./06_User_Group_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Read and interpret Linux file permissions in both symbolic and numeric notation
2. Modify file permissions using chmod in both numeric and symbolic modes
3. Change file ownership and group with chown and chgrp
4. Explain the purpose and behavior of SUID, SGID, and the Sticky Bit
5. Configure default permissions using umask
6. Apply correct permissions for real-world scenarios (web servers, SSH keys, shared directories)
7. Diagnose and resolve common "Permission denied" errors

---

Security in Linux starts with permissions. Every file and directory has an owner, a group, and a set of access rules that determine who can read, write, or execute it. A misconfigured permission can either lock legitimate users out of their own files or, worse, expose sensitive data to the entire system. Understanding permissions is essential for both daily work and system hardening.

> **Analogy: Hotel Key Card** -- Linux permissions work like hotel key cards. The owner is the guest who booked the room (full access), the group is the cleaning staff (limited access to common areas), and others are strangers (no access unless explicitly allowed). The `chmod` command is the front desk reprinting key cards with different access levels.

## 1. Understanding File Permissions

Every file in Linux has permissions.

### Permission Structure

```
-rw-r--r-- 1 ubuntu ubuntu 1234 Jan 23 10:00 file.txt
│└─┬──┘└─┬──┘└─┬──┘
│  │     │     │
│  │     │     └── Others
│  │     └── Group
│  └── Owner
└── File type
```

### Permission Types

| Permission | Character | Number | File | Directory |
|------------|-----------|--------|------|-----------|
| Read | r | 4 | Read content | List contents (ls) |
| Write | w | 2 | Modify content | Create/delete files |
| Execute | x | 1 | Execute | Enter (cd) |
| None | - | 0 | No permission | No permission |

### Reading Permissions

```
rwxr-xr--
│││││││││
││││││││└── Others: read only (r--)
│││││└┴┴── Group: read+execute (r-x)
└┴┴┴───── Owner: all permissions (rwx)
```

Numeric conversion:
```
rwx = 4+2+1 = 7
r-x = 4+0+1 = 5
r-- = 4+0+0 = 4

→ 754
```

---

## 2. chmod - Change Permissions

### Numeric Mode

```bash
# Syntax
chmod [permission_number] file

# Examples
chmod 755 script.sh      # rwxr-xr-x
chmod 644 file.txt       # rw-r--r--
chmod 600 secret.key     # rw-------
chmod 777 public/        # rwxrwxrwx (not recommended)
```

### Common Permissions

| Permission | Number | Use Case |
|------------|--------|----------|
| rwxr-xr-x | 755 | Executables, directories |
| rw-r--r-- | 644 | Regular files |
| rw------- | 600 | Sensitive files (keys, configs) |
| rwx------ | 700 | Private directories |
| rwxrwxr-x | 775 | Group-shared directories |

### Symbolic Mode

```bash
# Syntax
chmod [target][operator][permission] file

# Target: u(owner), g(group), o(others), a(all)
# Operator: +(add), -(remove), =(set)
# Permission: r, w, x

# Examples
chmod u+x script.sh      # Add execute for owner
chmod g-w file.txt       # Remove write from group
chmod o=r file.txt       # Set others to read-only
chmod a+r file.txt       # Add read for all

# Multiple permissions
chmod u+rwx,g+rx,o+r file.txt
chmod ug+x script.sh

# Recursive application
chmod -R 755 directory/
```

### Execute Permission Example

```bash
# Grant execute permission to script
chmod +x script.sh
./script.sh

# Or
chmod u+x script.sh
```

---

## 3. chown - Change Owner

```bash
# Syntax
chown [options] owner[:group] file

# Change owner only
chown newuser file.txt

# Change owner and group
chown newuser:newgroup file.txt

# Change group only
chown :newgroup file.txt

# Recursive change
chown -R user:group directory/
```

```bash
# Examples
sudo chown www-data:www-data /var/www/html
sudo chown -R ubuntu:ubuntu ~/projects/
```

---

## 4. chgrp - Change Group

```bash
# Change group only
chgrp developers file.txt

# Recursive change
chgrp -R www-data /var/www/
```

---

## 5. Special Permissions

### SUID (Set User ID)

When executed, runs with the file owner's permissions.

```
-rwsr-xr-x  → s indicates SUID is set
```

```bash
# Set SUID
chmod u+s program
chmod 4755 program

# Typical SUID file
ls -l /usr/bin/passwd
# -rwsr-xr-x 1 root root ... /usr/bin/passwd
```

### SGID (Set Group ID)

When executed, runs with the file group's permissions.
For directories, new files inherit the directory's group.

```
-rwxr-sr-x  → s indicates SGID is set
```

```bash
# Set SGID
chmod g+s directory/
chmod 2755 directory/

# Useful for shared directories
sudo mkdir /shared
sudo chmod 2775 /shared
sudo chgrp developers /shared
# Now files created by developers group members inherit developers group
```

### Sticky Bit

For directories, only file owners can delete their files.

```
drwxrwxrwt  → t indicates Sticky Bit
```

```bash
# Set Sticky Bit
chmod +t directory/
chmod 1777 directory/

# /tmp is a typical example
ls -ld /tmp
# drwxrwxrwt 1 root root 4096 Jan 23 10:00 /tmp
```

### Special Permission Numbers

| Permission | Number | Position |
|------------|--------|----------|
| SUID | 4 | First digit |
| SGID | 2 | First digit |
| Sticky | 1 | First digit |

```bash
# SUID + 755
chmod 4755 file

# SGID + 775
chmod 2775 directory/

# Sticky + 777
chmod 1777 /tmp/
```

---

## 6. umask - Default Permissions

umask determines default permissions for new files/directories.

```
File default: 666 - umask
Directory default: 777 - umask
```

```bash
# Check current umask
umask
# 0022

# Set umask
umask 022    # New files 644, new dirs 755
umask 077    # New files 600, new dirs 700
umask 002    # New files 664, new dirs 775
```

### umask Calculation Example

```
umask = 022

File:      666
         - 022
         ------
           644 (rw-r--r--)

Directory: 777
         - 022
         ------
           755 (rwxr-xr-x)
```

### Permanent Setting

```bash
# Add to ~/.bashrc or ~/.profile
echo "umask 022" >> ~/.bashrc
source ~/.bashrc
```

---

## 7. Permission Check Commands

### ls -l

```bash
ls -l file.txt
# -rw-r--r-- 1 ubuntu ubuntu 1234 Jan 23 10:00 file.txt
```

### stat

```bash
stat file.txt
```

Output:
```
  File: file.txt
  Size: 1234            Blocks: 8          IO Block: 4096   regular file
Access: (0644/-rw-r--r--)  Uid: ( 1000/  ubuntu)   Gid: ( 1000/  ubuntu)
...
```

### getfacl (when ACL supported)

```bash
getfacl file.txt
```

---

## 8. Real-World Scenarios

### Web Server Directory Setup

```bash
# Set up web root directory
sudo mkdir -p /var/www/mysite
sudo chown -R www-data:www-data /var/www/mysite
sudo chmod -R 755 /var/www/mysite

# Upload directory (allow write)
sudo mkdir /var/www/mysite/uploads
sudo chmod 775 /var/www/mysite/uploads

# Config file (read-only)
sudo chmod 640 /var/www/mysite/config.php
sudo chown root:www-data /var/www/mysite/config.php
```

### Shared Directory Setup

```bash
# Development team shared directory
sudo groupadd developers
sudo mkdir /shared/dev
sudo chgrp developers /shared/dev
sudo chmod 2775 /shared/dev

# Add user to group
sudo usermod -aG developers username
```

### SSH Key Permissions

```bash
# SSH directory permissions (required!)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa          # Private key
chmod 644 ~/.ssh/id_rsa.pub      # Public key
chmod 600 ~/.ssh/authorized_keys
chmod 644 ~/.ssh/known_hosts
```

### Script Execute Permissions

```bash
# Grant execute permission to scripts
chmod +x deploy.sh
chmod +x *.sh

# Or 755
chmod 755 backup.sh
```

---

## 9. Permission Troubleshooting

### Permission Denied Error

```bash
# Error: Permission denied
# Solution 1: Check permissions
ls -la file.txt

# Solution 2: Change permissions
chmod 644 file.txt    # Or appropriate permission

# Solution 3: Change owner
sudo chown $USER file.txt

# Solution 4: Use sudo
sudo cat /etc/shadow
```

### Cannot Enter Directory

```bash
# Error: Permission denied (cannot cd)
# Directory needs x permission
chmod +x directory/
```

### Cannot Modify File

```bash
# Error: Cannot modify file
# Solution: Add write permission
chmod u+w file.txt

# Or directory write permission (for new files)
chmod u+w directory/
```

---

## 10. Practice Exercises

### Exercise 1: Reading Permissions

```bash
# Create file
touch test_file.txt
mkdir test_dir

# Check permissions
ls -la test_file.txt test_dir
stat test_file.txt
```

### Exercise 2: chmod Practice

```bash
# Create script
cat > test_script.sh << 'EOF'
#!/bin/bash
echo "Hello from script!"
EOF

# Try to execute (no permission)
./test_script.sh
# Permission denied

# Grant execute permission
chmod +x test_script.sh
./test_script.sh
# Hello from script!

# Various permission settings
chmod 755 test_script.sh    # rwxr-xr-x
chmod 700 test_script.sh    # rwx------
chmod 644 test_script.sh    # rw-r--r--
```

### Exercise 3: Change Ownership

```bash
# Check file ownership
ls -l test_file.txt

# Change group (may need sudo)
sudo chgrp users test_file.txt
ls -l test_file.txt
```

### Exercise 4: umask Test

```bash
# Check current umask
umask

# Change umask and create file
umask 077
touch secret.txt
mkdir private_dir
ls -la secret.txt private_dir

# Restore original
umask 022
```

### Exercise 5: Shared Directory

```bash
# Create shared directory (needs sudo)
sudo mkdir /tmp/shared_test
sudo chmod 1777 /tmp/shared_test

# Create test file
touch /tmp/shared_test/my_file.txt

# Other users cannot delete it
# (due to Sticky bit)
```

---

**Previous**: [Text Processing](./04_Text_Processing.md) | **Next**: [User and Group Management](./06_User_Group_Management.md)
