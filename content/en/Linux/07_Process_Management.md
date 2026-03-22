# Process Management

**Previous**: [User and Group Management](./06_User_Group_Management.md) | **Next**: [Package Management](./08_Package_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a process is and describe the Linux process lifecycle and states
2. List and inspect running processes using ps, pstree, and top/htop
3. Terminate processes using kill, killall, and pkill with appropriate signals
4. Manage foreground and background jobs using job control (fg, bg, jobs, &)
5. Keep processes running after logout with nohup and disown
6. Manage system services using systemctl (start, stop, enable, status)
7. Adjust process priority using nice and renice

---

Everything happening on a Linux system -- from the init process (PID 1) to your interactive shell session -- is a process. Understanding how to monitor, control, and terminate processes is critical for troubleshooting a slow server, stopping a runaway script, or ensuring a critical service stays running after you log out.

> **Analogy: Restaurant Kitchen** -- Processes are orders being cooked in the kitchen. The kernel is the head chef who decides which dish gets the stove next (scheduling), sends sous chefs to work in parallel (threads), and stops a burnt dish before it sets off the fire alarm (killing a process). `ps` is the order board, and `top` is the kitchen's live status display.

## 1. Process Concepts

A process is an instance of a running program.

```
┌─────────────────────────────────────────────────────────┐
│                      Process                             │
├─────────────────────────────────────────────────────────┤
│  PID: 1234                    (Process ID)               │
│  PPID: 1                      (Parent Process ID)        │
│  UID: 1000                    (Running User)             │
│  State: Running                                          │
│  Memory: 50MB                                            │
│  CPU: 2%                                                 │
└─────────────────────────────────────────────────────────┘
```

### Process States

| State | Code | Description |
|-------|------|-------------|
| Running | R | Running or runnable |
| Sleeping | S | Waiting (interruptible) |
| Disk Sleep | D | Waiting (uninterruptible) |
| Stopped | T | Stopped (Ctrl+Z) |
| Zombie | Z | Terminated but not collected by parent |

### Process Hierarchy

```
init/systemd (PID 1)
├── sshd
│   └── bash
│       └── vim
├── nginx
│   ├── nginx worker
│   └── nginx worker
└── cron
```

---

## 2. ps - Process List

### Basic Usage

```bash
# Current terminal processes
ps

# All processes
ps aux

# Full format
ps -ef
```

### Interpreting ps aux Output

```
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root         1  0.0  0.1 168936 11784 ?        Ss   Jan20   0:08 /sbin/init
ubuntu    1234  0.5  1.2 723456 98765 pts/0    Sl   10:00   0:15 /usr/bin/node app.js
```

| Field | Description |
|-------|-------------|
| USER | Running user |
| PID | Process ID |
| %CPU | CPU usage |
| %MEM | Memory usage |
| VSZ | Virtual memory size |
| RSS | Actual memory usage |
| TTY | Terminal (? = none) |
| STAT | State |
| START | Start time |
| TIME | CPU time |
| COMMAND | Command |

### Key Options

```bash
# All processes (BSD style)
ps aux

# All processes (UNIX style)
ps -ef

# Specific user processes
ps -u ubuntu

# Specific process
ps -p 1234

# Tree format
ps auxf
ps -ef --forest

# Search for specific command
ps aux | grep nginx
```

### pstree - Process Tree

```bash
# Full tree
pstree

# Show PIDs
pstree -p

# Specific user
pstree ubuntu

# From specific PID
pstree -p 1234
```

---

## 3. top - Real-time Monitoring

### Basic Usage

```bash
top
```

Output:
```
top - 10:30:00 up 5 days,  3:45,  2 users,  load average: 0.15, 0.10, 0.05
Tasks: 120 total,   1 running, 119 sleeping,   0 stopped,   0 zombie
%Cpu(s):  2.0 us,  1.0 sy,  0.0 ni, 96.5 id,  0.5 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :   7976.0 total,   2048.0 free,   3500.0 used,   2428.0 buff/cache
MiB Swap:   2048.0 total,   2048.0 free,      0.0 used.   4000.0 avail Mem

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
 1234 ubuntu    20   0  723456  98765  12345 S   5.0   1.2   0:15.23 node
 5678 mysql     20   0 1234567 234567  23456 S   2.0   2.9   1:23.45 mysqld
```

### top Header Explanation

| Item | Description |
|------|-------------|
| load average | 1-min, 5-min, 15-min average load |
| us | User process CPU |
| sy | System (kernel) CPU |
| ni | Nice'd process CPU |
| id | Idle CPU |
| wa | I/O wait |

### top Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `k` | Kill process |
| `r` | Change nice value |
| `M` | Sort by memory |
| `P` | Sort by CPU |
| `1` | Show individual CPUs |
| `c` | Show full command |
| `f` | Select fields |
| `h` | Help |

### htop - Enhanced top

```bash
# Install
# Ubuntu/Debian
sudo apt install htop

# CentOS/RHEL
sudo dnf install htop

# Run
htop
```

htop Features:
- Color interface
- Mouse support
- Scrollable
- Process tree view
- Search functionality

---

## 4. Process Control

### kill - Terminate Process

```bash
# Normal termination (SIGTERM)
kill 1234

# Force kill (SIGKILL)
kill -9 1234
kill -KILL 1234

# List signals
kill -l
```

### Common Signals

| Signal | Number | Description |
|--------|--------|-------------|
| SIGHUP | 1 | Restart/reload configuration |
| SIGINT | 2 | Interrupt (Ctrl+C) |
| SIGQUIT | 3 | Quit with core dump |
| SIGKILL | 9 | Force kill (cannot be ignored) |
| SIGTERM | 15 | Normal termination (default) |
| SIGSTOP | 19 | Pause |
| SIGCONT | 18 | Resume |

```bash
# Graceful termination
kill -TERM 1234

# Force kill (last resort)
kill -9 1234

# Reload configuration
kill -HUP 1234

# Pause process
kill -STOP 1234

# Resume process
kill -CONT 1234
```

### killall - Kill by Name

```bash
# Kill by name
killall nginx

# Force kill
killall -9 node

# Interactive confirmation
killall -i process_name
```

### pkill - Kill by Pattern

```bash
# Pattern matching
pkill -f "python app.py"

# User's processes
pkill -u username

# Specify signal
pkill -9 -f "node server.js"
```

### pgrep - Find Process IDs

```bash
# Find PID
pgrep nginx

# Detailed information
pgrep -a nginx

# Specific user
pgrep -u root sshd
```

---

## 5. Foreground and Background

### Foreground

Occupies the terminal while running.

```bash
# Normal execution (foreground)
./long_running_script.sh
```

### Background

Runs without occupying the terminal.

```bash
# Background execution
./long_running_script.sh &

# Redirect output
./script.sh > output.log 2>&1 &
```

### Job Control

```bash
# Pause foreground job
# Ctrl + Z

# List background jobs
jobs

# Send to background
bg %1

# Bring to foreground
fg %1

# Kill job
kill %1
```

### nohup - Continue After Logout

```bash
# Continue running after logout
nohup ./script.sh &

# Specify output
nohup ./script.sh > output.log 2>&1 &

# Check PID
echo $!
```

### disown - Detach from Terminal

```bash
# Run in background then detach
./script.sh &
disown

# Or immediately detach
./script.sh &
disown %1
```

---

## 6. systemctl - Service Management

### Check Service Status

```bash
# Check status
systemctl status nginx

# List running services
systemctl list-units --type=service

# Active services
systemctl list-units --type=service --state=running

# Failed services
systemctl --failed
```

### Service Control

```bash
# Start
sudo systemctl start nginx

# Stop
sudo systemctl stop nginx

# Restart
sudo systemctl restart nginx

# Reload configuration (without interruption)
sudo systemctl reload nginx

# Restart or reload
sudo systemctl reload-or-restart nginx
```

### Automatic Start on Boot

```bash
# Enable automatic start
sudo systemctl enable nginx

# Disable automatic start
sudo systemctl disable nginx

# Check enabled status
systemctl is-enabled nginx

# Enable and start immediately
sudo systemctl enable --now nginx
```

### View Service Logs

```bash
# Service logs
journalctl -u nginx

# Real-time logs
journalctl -u nginx -f

# Last 100 lines
journalctl -u nginx -n 100

# Today's logs
journalctl -u nginx --since today
```

---

## 7. Process Priority

### nice - Set Priority

nice value: -20 (high priority) ~ 19 (low priority), default 0

```bash
# Run with low priority
nice -n 10 ./heavy_task.sh

# High priority (requires root)
sudo nice -n -10 ./important_task.sh
```

### renice - Change Priority of Running Process

```bash
# Change priority
renice -n 10 -p 1234

# All processes of user
sudo renice -n 5 -u username
```

---

## 8. Practice Exercises

### Exercise 1: Process Monitoring

```bash
# Check current processes
ps aux | head -20

# Find specific process
ps aux | grep sshd

# Process tree
pstree -p | head -30

# Real-time monitoring
top
# (press q to quit)
```

### Exercise 2: Background Jobs

```bash
# Create test script
cat > test_bg.sh << 'EOF'
#!/bin/bash
for i in {1..10}; do
    echo "Count: $i"
    sleep 2
done
EOF
chmod +x test_bg.sh

# Run in foreground then pause with Ctrl+Z
./test_bg.sh
# Ctrl+Z

# List jobs
jobs

# Send to background
bg %1

# Bring back to foreground
fg %1
```

### Exercise 3: Process Termination

```bash
# Run sleep process
sleep 300 &
echo "PID: $!"

# Check process
ps aux | grep sleep

# Kill
kill $!

# Verify
ps aux | grep sleep
```

### Exercise 4: Service Management

```bash
# SSH service status
systemctl status sshd

# Check logs
journalctl -u sshd -n 20

# List running services
systemctl list-units --type=service --state=running
```

### Exercise 5: Resource Monitoring

```bash
# Top 5 CPU-intensive processes
ps aux --sort=-%cpu | head -6

# Top 5 memory-intensive processes
ps aux --sort=-%mem | head -6

# Process count
ps aux | wc -l

# Check for zombie processes
ps aux | grep Z
```

---

**Previous**: [User and Group Management](./06_User_Group_Management.md) | **Next**: [Package Management](./08_Package_Management.md)
