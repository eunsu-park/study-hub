# Shell Scripting

**Previous**: [Package Management](./08_Package_Management.md) | **Next**: [Network Basics](./10_Network_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write and execute shell scripts with proper shebang lines and permissions
2. Declare and use variables, including special variables ($?, $@, $$) and environment variables
3. Read user input with read and process command-line arguments
4. Implement conditional logic using if/elif/else and case statements
5. Write loops (for, while, until) with break and continue controls
6. Define and call functions with parameters, return values, and local variables
7. Write practical automation scripts for backups, health checks, and log analysis
8. Debug shell scripts using set -x, set -e, set -u, and set -o pipefail

---

Shell scripting is the automation superpower of Linux. Instead of typing the same sequence of commands every day, you write them once in a script and let the computer handle the repetition. From simple backup scripts to full deployment pipelines, shell scripts are the glue that holds Linux system administration together. Even in the age of Python and Go, shell scripting remains the fastest way to automate system-level tasks.

## 1. Shell Script Basics

A shell script is a program that collects commands in a file for automation.

### First Script

```bash
#!/bin/bash
# This is a comment
echo "Hello, World!"
```

### shebang (#!)

The first line of a script specifies the interpreter.

```bash
#!/bin/bash        # bash shell
#!/bin/sh          # standard shell
#!/usr/bin/env bash  # find bash in environment (more portable)
```

### Running Scripts

```bash
# 1. Grant execute permission and run
chmod +x script.sh
./script.sh

# 2. Run directly with bash
bash script.sh

# 3. Run with source (in current shell)
source script.sh
. script.sh
```

---

## 2. Variables

### Variable Declaration and Usage

```bash
#!/bin/bash

# Variable declaration (no spaces around =!)
name="John"
age=25
readonly PI=3.14159    # constant

# Variable usage
echo $name
echo ${name}           # recommended (clearer)
echo "Hello, ${name}!"
echo "Age: $age"

# Delete variable
unset name
```

### Special Variables

| Variable | Description |
|----------|-------------|
| `$0` | Script name |
| `$1` ~ `$9` | Positional parameters |
| `$#` | Number of parameters |
| `$@` | All parameters (individual) |
| `$*` | All parameters (string) |
| `$?` | Exit status of last command |
| `$$` | Current process PID |
| `$!` | Last background PID |

```bash
#!/bin/bash
echo "Script: $0"
echo "First argument: $1"
echo "Second argument: $2"
echo "Argument count: $#"
echo "All arguments: $@"
echo "PID: $$"
```

### Environment Variables

```bash
# View environment variables
env
printenv

# Key environment variables
echo $HOME       # Home directory
echo $USER       # Username
echo $PATH       # Execution path
echo $PWD        # Current directory
echo $SHELL      # Current shell

# Set environment variable
export MY_VAR="value"

# Set within script
#!/bin/bash
export PATH="$PATH:/opt/myapp/bin"
```

### Variable Default Values

```bash
# Use default value if variable is unset
name=${name:-"default"}

# Assign default value if variable is unset
name=${name:="default"}

# Error if variable is unset
name=${name:?"Variable not set"}
```

---

## 3. Input

### read Command

```bash
#!/bin/bash

# Basic input
echo -n "Enter your name: "
read name
echo "Hello, $name!"

# With prompt
read -p "Enter your age: " age
echo "You are ${age} years old."

# Password input (hidden)
read -sp "Password: " password
echo
echo "Password set."

# Timeout
read -t 5 -p "Enter within 5 seconds: " input

# Multiple variables
read -p "Name and age: " name age
echo "$name, $age"
```

### Command-line Arguments

```bash
#!/bin/bash
# Usage: ./script.sh arg1 arg2

if [ $# -lt 2 ]; then
    echo "Usage: $0 <name> <age>"
    exit 1
fi

name=$1
age=$2
echo "$name is ${age} years old."
```

---

## 4. Conditionals

### if Statement

```bash
#!/bin/bash

# Basic format
if [ condition ]; then
    command
fi

# if-else
if [ condition ]; then
    command1
else
    command2
fi

# if-elif-else
if [ condition1 ]; then
    command1
elif [ condition2 ]; then
    command2
else
    command3
fi
```

### Comparison Operators

#### Numeric Comparison

| Operator | Description |
|----------|-------------|
| `-eq` | Equal |
| `-ne` | Not equal |
| `-gt` | Greater than |
| `-ge` | Greater or equal |
| `-lt` | Less than |
| `-le` | Less or equal |

```bash
#!/bin/bash
num=10

if [ $num -gt 5 ]; then
    echo "$num is greater than 5."
fi

if [ $num -eq 10 ]; then
    echo "$num is 10."
fi
```

#### String Comparison

| Operator | Description |
|----------|-------------|
| `=` | Equal |
| `!=` | Not equal |
| `-z` | Empty string |
| `-n` | Not empty |

```bash
#!/bin/bash
str="hello"

if [ "$str" = "hello" ]; then
    echo "String is hello."
fi

if [ -z "$empty" ]; then
    echo "Variable is empty."
fi

if [ -n "$str" ]; then
    echo "Variable has value."
fi
```

#### File Tests

| Operator | Description |
|----------|-------------|
| `-e` | File exists |
| `-f` | Regular file |
| `-d` | Directory |
| `-r` | Readable |
| `-w` | Writable |
| `-x` | Executable |
| `-s` | File size > 0 |

```bash
#!/bin/bash
file="/etc/passwd"

if [ -e "$file" ]; then
    echo "File exists."
fi

if [ -f "$file" ]; then
    echo "Regular file."
fi

if [ -d "/home" ]; then
    echo "Directory."
fi

if [ -r "$file" ]; then
    echo "Readable."
fi
```

### Logical Operators

```bash
#!/bin/bash

# AND
if [ $a -gt 0 ] && [ $b -gt 0 ]; then
    echo "Both positive"
fi

# OR
if [ $a -gt 0 ] || [ $b -gt 0 ]; then
    echo "At least one positive"
fi

# NOT
if [ ! -f "file.txt" ]; then
    echo "File doesn't exist."
fi

# Using [[ ]] (recommended)
if [[ $a -gt 0 && $b -gt 0 ]]; then
    echo "Both positive"
fi
```

### case Statement

```bash
#!/bin/bash
read -p "Choose a fruit (apple/banana/orange): " fruit

case $fruit in
    apple)
        echo "You chose apple."
        ;;
    banana)
        echo "You chose banana."
        ;;
    orange)
        echo "You chose orange."
        ;;
    *)
        echo "Unknown fruit."
        ;;
esac
```

---

## 5. Loops

### for Loop

```bash
#!/bin/bash

# List iteration
for name in Alice Bob Charlie; do
    echo "Hello, $name!"
done

# Range iteration
for i in {1..5}; do
    echo "Number: $i"
done

# With increment
for i in {0..10..2}; do
    echo "Even: $i"
done

# C-style
for ((i=0; i<5; i++)); do
    echo "Index: $i"
done

# File list
for file in *.txt; do
    echo "Processing: $file"
done

# Command output
for user in $(cat /etc/passwd | cut -d: -f1); do
    echo "User: $user"
done
```

### while Loop

```bash
#!/bin/bash

# Basic while
count=1
while [ $count -le 5 ]; do
    echo "Count: $count"
    ((count++))
done

# Read file
while read line; do
    echo "Line: $line"
done < file.txt

# Infinite loop
while true; do
    echo "Running... (Ctrl+C to exit)"
    sleep 1
done
```

### until Loop

```bash
#!/bin/bash

# Repeat until condition is true
count=1
until [ $count -gt 5 ]; do
    echo "Count: $count"
    ((count++))
done
```

### break and continue

```bash
#!/bin/bash

# break - exit loop
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        break
    fi
    echo $i
done

# continue - skip to next iteration
for i in {1..5}; do
    if [ $i -eq 3 ]; then
        continue
    fi
    echo $i
done
```

---

## 6. Functions

### Function Definition

```bash
#!/bin/bash

# Method 1
function greet() {
    echo "Hello, $1!"
}

# Method 2
say_bye() {
    echo "Goodbye, $1!"
}

# Call
greet "World"
say_bye "World"
```

### Function Parameters

```bash
#!/bin/bash

print_info() {
    echo "Name: $1"
    echo "Age: $2"
    echo "Argument count: $#"
}

print_info "John" 25
```

### Return Values

```bash
#!/bin/bash

# return for exit status (0-255)
is_even() {
    if [ $(($1 % 2)) -eq 0 ]; then
        return 0    # true
    else
        return 1    # false
    fi
}

if is_even 4; then
    echo "Even number."
fi

# echo for value return
add() {
    echo $(($1 + $2))
}

result=$(add 5 3)
echo "5 + 3 = $result"
```

### Local Variables

```bash
#!/bin/bash

my_func() {
    local local_var="I'm local"
    global_var="I'm global"
    echo "$local_var"
}

my_func
echo "$global_var"     # prints
echo "$local_var"      # doesn't print
```

---

## 7. Arrays

```bash
#!/bin/bash

# Array declaration
fruits=("apple" "banana" "orange")

# Access by index
echo ${fruits[0]}     # apple
echo ${fruits[1]}     # banana

# All elements
echo ${fruits[@]}

# Element count
echo ${#fruits[@]}

# Add element
fruits+=("grape")

# Iterate array
for fruit in "${fruits[@]}"; do
    echo $fruit
done

# With index
for i in "${!fruits[@]}"; do
    echo "$i: ${fruits[$i]}"
done
```

---

## 8. Practical Script Examples

### Backup Script

```bash
#!/bin/bash
# backup.sh - Directory backup script

SOURCE_DIR="${1:-/var/www}"
BACKUP_DIR="${2:-/backup}"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${DATE}.tar.gz"

# Check backup directory
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
fi

# Check source directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: $SOURCE_DIR directory not found."
    exit 1
fi

# Execute backup
echo "Starting backup: $SOURCE_DIR -> $BACKUP_DIR/$BACKUP_FILE"
tar -czvf "$BACKUP_DIR/$BACKUP_FILE" -C "$(dirname $SOURCE_DIR)" "$(basename $SOURCE_DIR)"

if [ $? -eq 0 ]; then
    echo "Backup complete: $BACKUP_DIR/$BACKUP_FILE"

    # Delete backups older than 30 days
    find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete
    echo "Old backups cleaned"
else
    echo "Backup failed!"
    exit 1
fi
```

### Server Health Check Script

```bash
#!/bin/bash
# health_check.sh - Server health check

LOG_FILE="/var/log/health_check.log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_disk() {
    local usage=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
    if [ "$usage" -gt 80 ]; then
        log_message "Warning: Disk usage ${usage}%"
        return 1
    fi
    log_message "Disk: ${usage}% used"
    return 0
}

check_memory() {
    local usage=$(free | awk '/Mem:/ {printf("%.0f", $3/$2 * 100)}')
    if [ "$usage" -gt 80 ]; then
        log_message "Warning: Memory usage ${usage}%"
        return 1
    fi
    log_message "Memory: ${usage}% used"
    return 0
}

check_service() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        log_message "Service $service: running"
        return 0
    else
        log_message "Warning: Service $service stopped"
        return 1
    fi
}

# Main
log_message "===== Health check started ====="
check_disk
check_memory
check_service "sshd"
check_service "nginx" 2>/dev/null || true
log_message "===== Health check completed ====="
```

### Log Analysis Script

```bash
#!/bin/bash
# log_analyzer.sh - Log analysis

LOG_FILE="${1:-/var/log/syslog}"

if [ ! -f "$LOG_FILE" ]; then
    echo "File not found: $LOG_FILE"
    exit 1
fi

echo "=== Log Analysis: $LOG_FILE ==="
echo

echo "=== Error Count ==="
grep -ci "error" "$LOG_FILE"

echo
echo "=== Recent 10 Errors ==="
grep -i "error" "$LOG_FILE" | tail -10

echo
echo "=== Log Count by Hour ==="
awk '{print $3}' "$LOG_FILE" | cut -d: -f1 | sort | uniq -c | sort -k2n
```

### User Creation Script

```bash
#!/bin/bash
# create_users.sh - Batch user creation from file
# Usage: sudo ./create_users.sh users.txt
# users.txt format: username:password:groupname

USER_FILE="$1"

if [ -z "$USER_FILE" ] || [ ! -f "$USER_FILE" ]; then
    echo "Usage: $0 <userfile>"
    exit 1
fi

while IFS=: read -r username password groupname; do
    # Skip empty lines or comments
    [[ -z "$username" || "$username" =~ ^# ]] && continue

    # Check if user exists
    if id "$username" &>/dev/null; then
        echo "User exists: $username (skipped)"
        continue
    fi

    # Create group if it doesn't exist
    if ! getent group "$groupname" &>/dev/null; then
        groupadd "$groupname"
        echo "Group created: $groupname"
    fi

    # Create user
    useradd -m -s /bin/bash -g "$groupname" "$username"
    echo "$username:$password" | chpasswd
    echo "User created: $username (group: $groupname)"

done < "$USER_FILE"

echo "Done!"
```

---

## 9. Debugging

### Debug Mode

```bash
# Trace script execution
bash -x script.sh

# Enable within script
#!/bin/bash
set -x    # Start debug
# commands...
set +x    # Stop debug

# Exit on error
set -e

# Error on undefined variable
set -u

# Detect pipe errors
set -o pipefail

# Best practice (script start)
#!/bin/bash
set -euo pipefail
```

---

## 10. Practice Exercises

### Exercise 1: Simple Calculator

```bash
#!/bin/bash
# calculator.sh

read -p "First number: " num1
read -p "Operator (+, -, *, /): " op
read -p "Second number: " num2

case $op in
    +) result=$((num1 + num2)) ;;
    -) result=$((num1 - num2)) ;;
    \*) result=$((num1 * num2)) ;;
    /) result=$((num1 / num2)) ;;
    *) echo "Invalid operator"; exit 1 ;;
esac

echo "Result: $num1 $op $num2 = $result"
```

### Exercise 2: File Organization Script

```bash
#!/bin/bash
# organize.sh - Organize files by extension

DIR="${1:-.}"

for file in "$DIR"/*; do
    [ -f "$file" ] || continue

    ext="${file##*.}"
    mkdir -p "$DIR/$ext"
    mv "$file" "$DIR/$ext/"
    echo "Moved: $file -> $DIR/$ext/"
done
```

---

**Previous**: [Package Management](./08_Package_Management.md) | **Next**: [Network Basics](./10_Network_Basics.md)
