# Filesystem Navigation

**Previous**: [Linux Basics](./01_Linux_Basics.md) | **Next**: [File and Directory Management](./03_File_Directory_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the Linux Filesystem Hierarchy Standard (FHS) and the purpose of key directories
2. Distinguish between absolute and relative paths
3. Navigate the filesystem using cd, pwd, and special directory symbols (~, .., -, .)
4. List directory contents with ls and interpret long-format output
5. Search for files using find, locate, which, and whereis
6. Inspect file metadata using file and stat commands
7. Apply wildcard patterns (globbing) to match files efficiently

---

Every file and directory in Linux lives within a single, unified tree rooted at `/`. Unlike Windows, which uses separate drive letters (C:, D:), Linux organizes everything -- devices, configuration, user data -- under one hierarchy. Knowing where things live and how to move around is the first step toward becoming productive on any Linux system.

## 1. Linux Directory Structure

Linux follows FHS (Filesystem Hierarchy Standard) with a tree structure.

```
/                          ← Root (top level)
├── bin/                   ← Essential executables (ls, cp, mv, etc.)
├── boot/                  ← Boot files (kernel, bootloader)
├── dev/                   ← Device files (disks, USB, etc.)
├── etc/                   ← System configuration files
├── home/                  ← User home directories
│   ├── user1/
│   └── user2/
├── lib/                   ← Shared libraries
├── media/                 ← Removable media mount points
├── mnt/                   ← Temporary mount points
├── opt/                   ← Additional software packages
├── proc/                  ← Process information (virtual filesystem)
├── root/                  ← Root user home
├── run/                   ← Runtime data
├── sbin/                  ← System administration commands
├── srv/                   ← Service data
├── sys/                   ← Kernel/device information (virtual)
├── tmp/                   ← Temporary files (deleted on reboot)
├── usr/                   ← User programs
│   ├── bin/              ← User commands
│   ├── lib/              ← Libraries
│   ├── local/            ← Locally installed programs
│   └── share/            ← Shared data
└── var/                   ← Variable data
    ├── log/              ← Log files
    ├── cache/            ← Cache
    └── www/              ← Web server files
```

---

## 2. Key Directory Descriptions

| Directory | Description | Examples |
|-----------|-------------|----------|
| `/` | Root, starting point of all directories | - |
| `/home` | Regular user home directories | `/home/ubuntu` |
| `/root` | Root user home | - |
| `/etc` | System configuration files | `/etc/passwd`, `/etc/hosts` |
| `/var` | Variable data like logs, cache | `/var/log/syslog` |
| `/tmp` | Temporary files (all users can write) | - |
| `/usr` | User programs, libraries | `/usr/bin/python3` |
| `/opt` | Third-party software | `/opt/google/chrome` |
| `/bin`, `/sbin` | Essential system commands | `/bin/ls`, `/sbin/reboot` |
| `/dev` | Device files | `/dev/sda`, `/dev/null` |
| `/proc` | Process/kernel information (virtual) | `/proc/cpuinfo` |

---

## 3. Understanding Paths

### Absolute Path

Full path starting from root (`/`).

```bash
# Absolute path examples
/home/ubuntu/documents/file.txt
/etc/nginx/nginx.conf
/var/log/syslog
```

### Relative Path

Path relative to current location.

```bash
# When current location is /home/ubuntu
documents/file.txt      # → /home/ubuntu/documents/file.txt
./documents/file.txt    # → Same meaning (current directory)
../shared/data.txt      # → /home/shared/data.txt
```

### Special Directories

| Symbol | Meaning | Example |
|--------|---------|---------|
| `.` | Current directory | `./script.sh` |
| `..` | Parent directory | `cd ..` |
| `~` | Home directory | `cd ~` = `cd /home/user` |
| `-` | Previous directory | `cd -` |
| `/` | Root directory | `cd /` |

```bash
# Using special directories
cd ~              # To home directory
cd ~/documents    # To home/documents
cd ..             # To parent directory
cd ../..          # Two levels up
cd -              # To previous directory
```

---

## 4. pwd - Check Current Location

```bash
# Print current working directory
pwd
```

Output:
```
/home/ubuntu/projects
```

---

## 5. cd - Change Directory

### Basic Usage

```bash
# Move with absolute path
cd /var/log

# Move with relative path
cd documents

# To home directory
cd
cd ~

# To parent directory
cd ..

# To previous directory
cd -
```

### Usage Examples

```bash
# Check current location
pwd                    # /home/ubuntu

# Move to documents
cd documents
pwd                    # /home/ubuntu/documents

# Move up
cd ..
pwd                    # /home/ubuntu

# To previous directory
cd -
pwd                    # /home/ubuntu/documents
```

---

## 6. ls - List Directory Contents

### Basic Usage

```bash
# Current directory
ls

# Specific directory
ls /var/log

# Multiple directories
ls /home /tmp
```

### Main Options

| Option | Description |
|--------|-------------|
| `-l` | Long format with details |
| `-a` | Include hidden files (all) |
| `-h` | Human-readable sizes |
| `-R` | Recursive into subdirectories |
| `-t` | Sort by modification time |
| `-S` | Sort by file size |
| `-r` | Reverse sort order |
| `-d` | Directory info itself |

### Option Combinations

```bash
# Long format + hidden files
ls -la

# Long format + human-readable
ls -lh

# Most recent first
ls -lt

# Largest first
ls -lS

# Common combination
ls -lah
```

### Interpreting ls -l Output

```
-rw-r--r-- 1 ubuntu ubuntu 4096 Jan 23 14:30 file.txt
│├──┬───┤ │ │      │      │    │            │
││  │   │ │ │      │      │    │            └── Filename
││  │   │ │ │      │      │    └── Modification time
││  │   │ │ │      │      └── File size (bytes)
││  │   │ │ │      └── Group
││  │   │ │ └── Owner
││  │   │ └── Hard link count
││  │   └── Other permissions (r--)
││  └── Group permissions (r--)
│└── Owner permissions (rw-)
└── File type (- file, d directory)
```

### File Type Indicators

| Character | Type |
|-----------|------|
| `-` | Regular file |
| `d` | Directory |
| `l` | Symbolic link |
| `c` | Character device |
| `b` | Block device |
| `s` | Socket |
| `p` | Pipe |

---

## 7. Finding Files

### find - File Search

```bash
# Basic syntax
find [path] [conditions] [actions]

# Find by name
find /home -name "*.txt"

# Case insensitive
find /home -iname "readme*"

# Specify type (f: file, d: directory)
find /var -type f -name "*.log"
find /home -type d -name "config"

# Find by size
find / -size +100M          # Over 100MB
find / -size -1k            # Under 1KB

# Find by modification time
find /var/log -mtime -7     # Modified within 7 days
find /tmp -mtime +30        # Modified over 30 days ago

# Find by permissions
find / -perm 777

# Find by owner
find /home -user ubuntu
```

### Combining find with Actions

```bash
# Delete found files
find /tmp -name "*.tmp" -delete

# Execute command on found files
find /home -name "*.sh" -exec chmod +x {} \;

# Print found files
find /var/log -name "*.log" -print
```

### locate - Fast Search

Fast search using a database.

```bash
# Search for file
locate nginx.conf

# Case insensitive
locate -i readme

# Update database (administrator)
sudo updatedb
```

### which - Command Location

```bash
# Location of command executable
which python3
```

Output:
```
/usr/bin/python3
```

### whereis - Command Related Files

```bash
# Executable, source, and manual locations
whereis nginx
```

Output:
```
nginx: /usr/sbin/nginx /usr/lib/nginx /etc/nginx /usr/share/nginx /usr/share/man/man8/nginx.8.gz
```

---

## 8. File Content Preview

### file - Check File Type

```bash
file document.pdf
file script.sh
file image.jpg
```

Output:
```
document.pdf: PDF document, version 1.4
script.sh: Bourne-Again shell script, ASCII text executable
image.jpg: JPEG image data, JFIF standard 1.01
```

### stat - Detailed File Information

```bash
stat file.txt
```

Output:
```
  File: file.txt
  Size: 1234            Blocks: 8          IO Block: 4096   regular file
Device: 801h/2049d      Inode: 123456      Links: 1
Access: (0644/-rw-r--r--)  Uid: ( 1000/  ubuntu)   Gid: ( 1000/  ubuntu)
Access: 2024-01-23 10:00:00.000000000 +0900
Modify: 2024-01-23 09:30:00.000000000 +0900
Change: 2024-01-23 09:30:00.000000000 +0900
 Birth: 2024-01-20 15:00:00.000000000 +0900
```

---

## 9. Wildcards (Globbing)

| Pattern | Description | Examples |
|---------|-------------|----------|
| `*` | Zero or more characters | `*.txt`, `log*` |
| `?` | Exactly one character | `file?.txt` |
| `[abc]` | One of a, b, c | `file[123].txt` |
| `[a-z]` | Range from a to z | `file[a-z].txt` |
| `[!abc]` | Exclude a, b, c | `file[!0-9].txt` |

```bash
# All txt files
ls *.txt

# Files starting with log
ls log*

# Single digit files
ls file?.txt

# Files ending with digit
ls file[0-9].txt

# Files starting with a-c
ls [a-c]*.txt
```

---

## 10. Practice Exercises

### Exercise 1: Directory Navigation

```bash
# 1. Check current location
pwd

# 2. Move to root
cd /

# 3. Check directory structure
ls -l

# 4. Move to /var/log
cd /var/log

# 5. Check log files
ls -lh

# 6. Return to home
cd ~
```

### Exercise 2: Detailed Information

```bash
# Full listing including hidden files
ls -la ~

# Check recently modified files
ls -lt /var/log | head -10

# Find large files
ls -lhS /var/log | head -5
```

### Exercise 3: Finding Files

```bash
# Find .conf files in home
find ~ -name "*.conf" 2>/dev/null

# Find nginx-related files in /etc
find /etc -name "*nginx*" 2>/dev/null

# Find files over 100MB
find / -size +100M 2>/dev/null | head -10

# Find logs modified within 7 days
find /var/log -mtime -7 -name "*.log"
```

### Exercise 4: System Directory Exploration

```bash
# Check CPU information
cat /proc/cpuinfo | head -20

# Memory information
cat /proc/meminfo | head -10

# System hostname
cat /etc/hostname

# Check logged-in users
cat /etc/passwd | head -5
```

---

**Previous**: [Linux Basics](./01_Linux_Basics.md) | **Next**: [File and Directory Management](./03_File_Directory_Management.md)
