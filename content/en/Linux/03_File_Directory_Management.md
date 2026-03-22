# File and Directory Management

**Previous**: [Filesystem Navigation](./02_Filesystem_Navigation.md) | **Next**: [Text Processing](./04_Text_Processing.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create files and directory structures using touch and mkdir
2. Copy files and directories with cp, choosing appropriate options for backups
3. Move and rename files and directories using mv
4. Safely delete files and directories with rm and rmdir, avoiding destructive mistakes
5. View file contents using cat, less, head, and tail (including real-time log monitoring)
6. Create and manage hard links and symbolic links with ln
7. Compress and extract archives using tar, gzip, and zip
8. Check disk usage with du and df

---

File and directory management is the daily bread of Linux administration. Whether you are deploying code, organizing backups, or cleaning up temporary files, you will use these commands hundreds of times a day. Mastering them -- along with their safety flags -- prevents costly mistakes like accidentally deleting an entire project directory.

## 1. Creating Files/Directories

### touch - Create Empty File

```bash
# Create empty file
touch newfile.txt

# Create multiple files
touch file1.txt file2.txt file3.txt

# Update timestamp if file exists
touch existing_file.txt
```

### mkdir - Create Directory

```bash
# Create single directory
mkdir projects

# Create multiple directories
mkdir dir1 dir2 dir3

# Create nested directories (-p: parents)
mkdir -p projects/web/frontend/src

# Create with permissions
mkdir -m 755 public_dir
```

```bash
# Create directory structure at once
mkdir -p myproject/{src,tests,docs,config}
```

Result:
```
myproject/
├── src/
├── tests/
├── docs/
└── config/
```

---

## 2. Copying Files/Directories

### cp - Copy

```bash
# Copy file
cp source.txt destination.txt

# Copy to another directory
cp file.txt /home/user/backup/

# Copy multiple files
cp file1.txt file2.txt /backup/
```

### Main Options

| Option | Description |
|--------|-------------|
| `-r`, `-R` | Recursive directory copy |
| `-i` | Confirm before overwrite |
| `-v` | Display progress |
| `-p` | Preserve permissions, owner, timestamp |
| `-a` | Archive mode (same as -rpP) |
| `-u` | Copy only newer files |
| `-n` | No overwrite |

```bash
# Copy directory (recursive)
cp -r projects/ projects_backup/

# Interactive copy (confirm overwrite)
cp -i important.txt backup/

# Display progress
cp -v largefile.zip /backup/

# Preserve attributes
cp -p config.txt /backup/

# Archive mode (recommended for backup)
cp -a /var/www/ /backup/www/

# Copy only newer files
cp -u *.txt /backup/
```

---

## 3. Moving and Renaming Files/Directories

### mv - Move/Rename

```bash
# Rename file
mv oldname.txt newname.txt

# Move file
mv file.txt /home/user/documents/

# Move directory
mv projects/ /home/user/

# Move multiple files
mv file1.txt file2.txt /backup/

# Move and rename
mv old_project/ /home/user/new_project/
```

### Main Options

| Option | Description |
|--------|-------------|
| `-i` | Confirm before overwrite |
| `-v` | Display progress |
| `-n` | No overwrite |
| `-u` | Move only if newer |

```bash
# Interactive move
mv -i file.txt /backup/

# Display progress
mv -v *.log /archive/

# Don't overwrite existing files
mv -n newfile.txt /shared/
```

---

## 4. Deleting Files/Directories

### rm - Delete Files

```bash
# Delete file
rm file.txt

# Delete multiple files
rm file1.txt file2.txt file3.txt

# Delete with wildcards
rm *.tmp
rm log_2023*
```

### Main Options

| Option | Description |
|--------|-------------|
| `-r`, `-R` | Recursive directory deletion |
| `-f` | Force deletion (no confirmation) |
| `-i` | Confirm before deletion |
| `-v` | Display deleted files |

```bash
# Delete directory
rm -r directory/

# Force delete (caution!)
rm -f file.txt

# Force delete directory (extreme caution!)
rm -rf old_project/

# Interactive deletion
rm -i important_file.txt

# Display deletion process
rm -rv logs/
```

### rmdir - Delete Empty Directory

```bash
# Can only delete empty directories
rmdir empty_dir/

# Delete parent empty directories
rmdir -p a/b/c/  # Deletes c, b, a in order (all must be empty)
```

### Dangerous Command Warnings

```bash
# Never execute these!
# rm -rf /           # Deletes entire system
# rm -rf /*          # Deletes everything under root
# rm -rf ~/*         # Deletes entire home directory
# rm -rf .           # Deletes current directory

# Safe habits
rm -ri directory/   # Interactive confirmation
ls directory/       # Check contents before deletion
```

---

## 5. Viewing File Contents

### cat - Print Entire Content

```bash
# Print file content
cat file.txt

# Concatenate and print multiple files
cat file1.txt file2.txt

# Show line numbers
cat -n file.txt

# Squeeze blank lines
cat -s file.txt
```

### less - Page-by-Page Viewing

View large files comfortably.

```bash
less largefile.txt
```

| Key | Action |
|-----|--------|
| `Space` / `f` | Next page |
| `b` | Previous page |
| `g` | Beginning of file |
| `G` | End of file |
| `/search` | Search forward |
| `?search` | Search backward |
| `n` | Next search result |
| `N` | Previous search result |
| `q` | Quit |

### more - Simple Page Viewing

```bash
more file.txt
```

### head - Beginning of File

```bash
# First 10 lines (default)
head file.txt

# First 20 lines
head -n 20 file.txt
head -20 file.txt

# First 100 bytes
head -c 100 file.txt
```

### tail - End of File

```bash
# Last 10 lines (default)
tail file.txt

# Last 20 lines
tail -n 20 file.txt

# Real-time monitoring (useful for logs)
tail -f /var/log/syslog

# Monitor multiple files in real-time
tail -f file1.log file2.log
```

---

## 6. Links

### Hard Link vs Symbolic Link

```
┌──────────────────────────────────────────────────────────┐
│                    Hard Link                             │
│                                                          │
│   file.txt ─────┬───▶ [inode 123] ───▶ [data blocks]   │
│                 │                                        │
│   hardlink.txt ─┘                                        │
│                                                          │
│   • Points to same inode                                │
│   • Data preserved even if original deleted             │
│   • Only within same filesystem                         │
│   • Directories not allowed                             │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                Symbolic Link (Soft Link)                 │
│                                                          │
│   file.txt ─────────▶ [inode 123] ───▶ [data blocks]   │
│                 ▲                                        │
│   symlink.txt ──┘  (points to path)                     │
│                                                          │
│   • Points to file path                                 │
│   • Broken link if original deleted                     │
│   • Can cross filesystems                               │
│   • Directories allowed                                 │
└──────────────────────────────────────────────────────────┘
```

### ln - Create Links

```bash
# Create hard link
ln original.txt hardlink.txt

# Create symbolic link
ln -s original.txt symlink.txt

# Symbolic link to directory
ln -s /var/log/ ~/logs

# Force overwrite
ln -sf new_target.txt symlink.txt
```

### Checking Links

```bash
# Check symbolic link (ls -l)
ls -l symlink.txt
```

Output:
```
lrwxrwxrwx 1 user user 12 Jan 23 10:00 symlink.txt -> original.txt
```

```bash
# Check link count (second column in ls -l)
ls -l hardlink.txt original.txt
```

Output:
```
-rw-r--r-- 2 user user 100 Jan 23 10:00 hardlink.txt
-rw-r--r-- 2 user user 100 Jan 23 10:00 original.txt
```

---

## 7. Compression and Archives

### tar - Archive

tar bundles multiple files into one.

| Option | Description |
|--------|-------------|
| `-c` | Create archive |
| `-x` | Extract archive |
| `-t` | List contents |
| `-v` | Verbose output |
| `-f` | Specify filename |
| `-z` | gzip compression (.tar.gz) |
| `-j` | bzip2 compression (.tar.bz2) |
| `-J` | xz compression (.tar.xz) |
| `-C` | Specify extraction directory |

```bash
# Create archive
tar -cvf archive.tar directory/

# gzip compressed archive
tar -czvf archive.tar.gz directory/

# bzip2 compression (higher compression)
tar -cjvf archive.tar.bz2 directory/

# xz compression (highest compression)
tar -cJvf archive.tar.xz directory/

# View archive contents
tar -tvf archive.tar.gz

# Extract archive
tar -xvf archive.tar
tar -xzvf archive.tar.gz

# Extract to specific directory
tar -xzvf archive.tar.gz -C /tmp/

# Extract specific files only
tar -xzvf archive.tar.gz file1.txt file2.txt
```

### gzip / gunzip - Compression

```bash
# Compress (deletes original)
gzip file.txt          # → file.txt.gz

# Decompress
gunzip file.txt.gz     # → file.txt

# Keep original while compressing
gzip -k file.txt

# Compression level (1-9, 9 is highest)
gzip -9 file.txt
```

### zip / unzip - ZIP Compression

```bash
# Compress
zip archive.zip file1.txt file2.txt

# Compress including directory
zip -r archive.zip directory/

# Decompress
unzip archive.zip

# Decompress to specific directory
unzip archive.zip -d /tmp/

# View contents
unzip -l archive.zip
```

### Compression Format Comparison

| Format | Command | Compression | Speed | Compatibility |
|--------|---------|-------------|-------|---------------|
| .gz | gzip | Medium | Fast | High |
| .bz2 | bzip2 | High | Medium | High |
| .xz | xz | Very High | Slow | Medium |
| .zip | zip | Medium | Fast | Highest |

---

## 8. Checking File Type

### file Command

```bash
file document.pdf
file script.sh
file /bin/ls
file archive.tar.gz
```

Output:
```
document.pdf: PDF document, version 1.4
script.sh: Bourne-Again shell script, ASCII text executable
/bin/ls: ELF 64-bit LSB pie executable, x86-64
archive.tar.gz: gzip compressed data
```

---

## 9. Disk Usage

### du - Directory Usage

```bash
# Directory size
du -h directory/

# Summary only
du -sh directory/

# Size of subdirectories in current directory
du -h --max-depth=1

# Find large directories
du -h --max-depth=1 | sort -hr | head -10
```

### df - Disk Free Space

```bash
# Usage by filesystem
df -h

# Filesystem for specific path
df -h /home
```

---

## 10. Practice Exercises

### Exercise 1: Create Project Structure

```bash
# Create project directories
mkdir -p myapp/{src,tests,docs,config}

# Check structure
ls -la myapp/

# Create empty files
touch myapp/src/main.py
touch myapp/tests/test_main.py
touch myapp/config/settings.conf
touch myapp/README.md

# Verify result
find myapp -type f
```

### Exercise 2: File Backup

```bash
# Create backup directory
mkdir -p backup/$(date +%Y%m%d)

# Copy file
cp -v important.txt backup/$(date +%Y%m%d)/

# Backup directory
cp -a myapp/ backup/$(date +%Y%m%d)/myapp_backup/

# Compressed backup
tar -czvf backup/myapp_$(date +%Y%m%d).tar.gz myapp/
```

### Exercise 3: Log File Management

```bash
# Move to log directory
cd /var/log

# Find large log files
ls -lhS *.log 2>/dev/null | head -5

# Check recent logs
tail -20 syslog

# Real-time monitoring
tail -f syslog
# (Exit with Ctrl+C)
```

### Exercise 4: Temporary File Cleanup

```bash
# Check /tmp contents
ls -la /tmp/

# Find temporary files older than 7 days
find /tmp -mtime +7 -type f 2>/dev/null

# Delete files matching pattern (caution)
# find /tmp -name "*.tmp" -mtime +7 -delete
```

### Exercise 5: Using Symbolic Links

```bash
# Link config files
mkdir -p ~/dotfiles
ln -s ~/.bashrc ~/dotfiles/bashrc
ln -s ~/.vimrc ~/dotfiles/vimrc

# Check links
ls -la ~/dotfiles/

# Shortcut to log directory
ln -s /var/log ~/logs
ls ~/logs/
```

---

**Previous**: [Filesystem Navigation](./02_Filesystem_Navigation.md) | **Next**: [Text Processing](./04_Text_Processing.md)
