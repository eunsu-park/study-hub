# Linux Basics

**Next**: [Filesystem Navigation](./02_Filesystem_Navigation.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what Linux is and describe its key characteristics
2. Identify major Linux distributions and their use cases
3. Distinguish between the terminal, shell, and kernel
4. Construct commands using the correct syntax of command, options, and arguments
5. Use the man pages and --help to find information about any command
6. Execute basic system information commands (whoami, hostname, date, uname)
7. Navigate command history using history, Ctrl+R, and bang operators
8. Apply essential keyboard shortcuts for efficient terminal use

---

Linux is the backbone of modern computing infrastructure. From web servers and cloud platforms to smartphones (Android) and supercomputers, Linux powers the vast majority of the systems the world depends on. Understanding Linux is not just a useful skill -- it is a foundational competency for any developer, system administrator, or DevOps engineer. This lesson introduces the core concepts you need to start working confidently in a Linux environment.

## 1. What is Linux?

Linux is an open-source operating system developed by Linus Torvalds in 1991.

```
┌─────────────────────────────────────────────────────────┐
│                   Applications                          │
│          (web servers, databases, dev tools)            │
├─────────────────────────────────────────────────────────┤
│                        Shell                            │
│                  (bash, zsh, sh)                        │
├─────────────────────────────────────────────────────────┤
│                    Linux Kernel                         │
│      (processes, memory, filesystem, networking)        │
├─────────────────────────────────────────────────────────┤
│                      Hardware                           │
│          (CPU, memory, disk, network)                   │
└─────────────────────────────────────────────────────────┘
```

### Linux Characteristics

| Feature | Description |
|---------|-------------|
| Open Source | Source code public, free to use |
| Stability | High stability suitable for server operations |
| Security | Permission-based security model |
| Multi-user | Support for multiple simultaneous user connections |
| Multi-tasking | Multiple processes run concurrently |
| Portability | Runs on various hardware platforms |

---

## 2. Linux Distributions

Various distributions exist based on the Linux kernel.

```
              ┌──────────────────┐
              │   Linux Kernel   │
              └────────┬─────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    Debian    │ │    RHEL      │ │    Arch      │
└──────┬───────┘ └──────┬───────┘ └──────────────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐
│   Ubuntu     │ │   CentOS     │
│   Mint       │ │   Rocky      │
└──────────────┘ │   Fedora     │
                 └──────────────┘
```

### Major Distribution Comparison

| Distribution | Base | Features | Use Cases |
|--------------|------|----------|-----------|
| Ubuntu | Debian | Ease of use, large community | Beginners, desktop, server |
| Debian | - | Stability, strict package policy | Server |
| CentOS/Rocky | RHEL | Enterprise-grade stability | Corporate servers |
| Fedora | RHEL | Latest technology, RHEL testbed | Developers |
| Alpine | - | Lightweight (5MB), security | Containers |
| Arch | - | Latest packages, DIY philosophy | Advanced users |

---

## 3. Terminal and Shell

### Terminal

The terminal is a text-based interface for interacting with the computer.

```
┌─────────────────────────────────────────────────────────┐
│ user@hostname:~$                                        │
│                                                         │
│  ← Space to enter commands and view results            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Shell

The shell is a program that interprets user commands and passes them to the kernel.

| Shell | Description |
|-------|-------------|
| bash | Default shell for most Linux systems (Bourne Again Shell) |
| zsh | Enhanced features, macOS default shell |
| sh | Original shell (Bourne Shell) |
| fish | User-friendly shell |

```bash
# Check current shell
echo $SHELL
```

Output:
```
/bin/bash
```

---

## 4. Command Structure Basics

```
command [options] [arguments]
   │        │          │
   │        │          └── Target of command (files, directories, etc.)
   │        └── Modify command behavior (-a, --all, etc.)
   └── Command to execute
```

### Examples

```bash
# Basic form
ls

# With option
ls -l

# Option + argument
ls -l /home

# Multiple options
ls -la /home

# Long options
ls --all --human-readable
```

### Option Formats

| Format | Example | Description |
|--------|---------|-------------|
| Short option | `-l` | Dash + single letter |
| Combined options | `-la` | Multiple short options combined |
| Long option | `--all` | Double dash + word |

---

## 5. Using Help

### man (Manual Pages)

```bash
# View command manual
man ls
man cp
man chmod

# Search by keyword
man -k "copy file"
```

Navigating man pages:
| Key | Action |
|-----|--------|
| `Space` / `f` | Next page |
| `b` | Previous page |
| `/search_term` | Search |
| `n` | Next search result |
| `q` | Quit |

### --help Option

```bash
# Quick help
ls --help
cp --help
```

### info

```bash
# Detailed information (GNU commands)
info ls
```

---

## 6. Basic Commands

### whoami - Current User

```bash
whoami
```

Output:
```
ubuntu
```

### hostname - System Name

```bash
hostname
```

Output:
```
my-server
```

### date - Current Date/Time

```bash
# Current date/time
date

# Specify format
date "+%Y-%m-%d %H:%M:%S"
```

Output:
```
Tue Jan 23 14:30:00 KST 2024
2024-01-23 14:30:00
```

### cal - Calendar

```bash
# Current month
cal

# Specific year
cal 2024

# Specific month
cal 3 2024
```

### clear - Clear Screen

```bash
clear
# Or Ctrl + L
```

### echo - Print Text

```bash
echo "Hello, Linux!"
echo $HOME
echo "Current path: $(pwd)"
```

---

## 7. Command History

### history Command

```bash
# View history
history

# Last 10 commands only
history 10

# Search history
history | grep "apt"
```

### Using History

| Command | Description |
|---------|-------------|
| `!!` | Execute previous command |
| `!n` | Execute nth history command |
| `!string` | Execute most recent command starting with string |
| `Ctrl + R` | Reverse history search |

```bash
# Re-execute previous command
!!

# Re-execute with sudo
sudo !!

# Execute command 123
!123

# Most recent command starting with ls
!ls
```

### Ctrl + R (Reverse Search)

```bash
# Press Ctrl + R then enter search term
(reverse-i-search)`apt': apt update

# Enter: execute
# Ctrl + R: next result
# Ctrl + G: cancel
```

---

## 8. Keyboard Shortcuts

### Cursor Movement

| Shortcut | Action |
|----------|--------|
| `Ctrl + A` | Beginning of line |
| `Ctrl + E` | End of line |
| `Ctrl + ←/→` | Move by word |

### Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl + U` | Delete before cursor |
| `Ctrl + K` | Delete after cursor |
| `Ctrl + W` | Delete word before cursor |
| `Ctrl + Y` | Paste deleted content |

### Control

| Shortcut | Action |
|----------|--------|
| `Ctrl + C` | Interrupt running command |
| `Ctrl + D` | End input (EOF) / logout |
| `Ctrl + Z` | Suspend process |
| `Ctrl + L` | Clear screen |

---

## 9. Tab Auto-completion

Using the Tab key enables auto-completion of commands and filenames.

```bash
# Command auto-completion
sys[Tab]     → systemctl

# Filename auto-completion
cd /ho[Tab]  → cd /home/

# When multiple candidates exist
cd /[Tab][Tab]  → Display possible list
```

---

## 10. Practice Exercises

### Exercise 1: Check System Information

```bash
# Check current user
whoami

# Check hostname
hostname

# Current date/time
date

# System uptime
uptime

# Kernel information
uname -a
```

### Exercise 2: Using Help

```bash
# Check ls manual
man ls

# Check cp help
cp --help

# Search manuals by keyword
man -k "disk space"
```

### Exercise 3: Using History

```bash
# Check history
history

# Re-execute previous command
!!

# Search previous commands with Ctrl + R
# (Press Ctrl + R and enter search term)
```

---

**Next**: [Filesystem Navigation](./02_Filesystem_Navigation.md)
