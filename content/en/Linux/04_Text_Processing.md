# Text Processing

**Previous**: [File and Directory Management](./03_File_Directory_Management.md) | **Next**: [Permissions and Ownership](./05_Permissions_Ownership.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Search for patterns in files using grep with regular expressions
2. Write basic and extended regular expressions for text matching
3. Extract specific fields from structured data using cut
4. Sort and deduplicate data with sort and uniq
5. Count lines, words, and bytes using wc
6. Perform text substitutions and line operations with sed
7. Process structured text data and perform calculations with awk
8. Build multi-stage data processing pipelines using pipes and redirection

---

Text processing is where Linux truly shines. The Unix philosophy of "small tools that do one thing well" means you can chain grep, sed, awk, sort, and other utilities into powerful data pipelines -- all from the command line. System administrators use these tools daily to analyze logs, extract metrics, and transform configuration files. Mastering text processing turns you from a casual Linux user into a power user.

## 1. grep - Text Search

grep is a powerful tool for searching patterns in files.

### Basic Usage

```bash
# Basic search
grep "pattern" file.txt

# Search multiple files
grep "error" *.log

# Recursive directory search
grep -r "TODO" ./src/
```

### Main Options

| Option | Description |
|--------|-------------|
| `-i` | Ignore case |
| `-r`, `-R` | Recursive search |
| `-n` | Show line numbers |
| `-v` | Invert match (exclude pattern) |
| `-c` | Count matching lines |
| `-l` | Show only filenames with matches |
| `-L` | Show filenames without matches |
| `-w` | Match whole words |
| `-A n` | Show n lines after match |
| `-B n` | Show n lines before match |
| `-C n` | Show n lines before and after match (context) |
| `-E` | Extended regular expressions |
| `-o` | Show only matching part |

### Option Usage Examples

```bash
# Case insensitive
grep -i "error" log.txt

# Show line numbers
grep -n "function" script.js

# Lines not matching
grep -v "comment" code.py

# Count matching lines
grep -c "import" *.py

# Show only filenames
grep -l "password" *.conf

# Whole word match
grep -w "log" file.txt    # Matches "log" only, excludes "logging"

# Show context
grep -A 3 "ERROR" app.log    # 3 lines after
grep -B 2 "ERROR" app.log    # 2 lines before
grep -C 2 "ERROR" app.log    # 2 lines before and after

# Recursive search with line numbers
grep -rn "TODO" ./
```

---

## 2. Basic Regular Expressions

| Pattern | Description | Examples |
|---------|-------------|----------|
| `.` | Any single character | `a.c` → abc, adc |
| `*` | Zero or more of preceding | `ab*c` → ac, abc, abbc |
| `^` | Start of line | `^Error` → Error at line start |
| `$` | End of line | `end$` → end at line end |
| `[ ]` | Character class | `[aeiou]` → vowels |
| `[^ ]` | Negated character class | `[^0-9]` → non-digits |
| `\` | Escape | `\.` → literal dot |

```bash
# Line starts with Error
grep "^Error" log.txt

# Line ends with ;
grep ";$" code.c

# 3 characters: a, any, t
grep "a.t" file.txt    # ant, art, act

# Lines starting with digit
grep "^[0-9]" data.txt

# Find empty lines
grep "^$" file.txt

# Comment lines (starting with #)
grep "^#" config.conf
```

### Extended Regular Expressions (-E)

| Pattern | Description | Examples |
|---------|-------------|----------|
| `+` | One or more of preceding | `ab+c` → abc, abbc (not ac) |
| `?` | Zero or one of preceding | `colou?r` → color, colour |
| `|` | OR | `cat|dog` |
| `( )` | Group | `(ab)+` → ab, abab |
| `{n}` | Exactly n times | `a{3}` → aaa |
| `{n,m}` | n to m times | `a{2,4}` → aa, aaa, aaaa |

```bash
# Use extended regex
grep -E "error|warning|critical" log.txt

# IP address pattern
grep -E "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}" access.log

# Email pattern (simple)
grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" file.txt

# Phone number pattern
grep -E "[0-9]{3}-[0-9]{3,4}-[0-9]{4}" contacts.txt
```

---

## 3. cut - Field Extraction

### Basic Usage

```bash
# Extract field by delimiter
cut -d'delimiter' -ffield_number file

# Extract by character position
cut -cstart-end file
```

### Main Options

| Option | Description |
|--------|-------------|
| `-d` | Specify delimiter |
| `-f` | Field number |
| `-c` | Character position |

```bash
# Colon delimiter, field 1 (username)
cut -d':' -f1 /etc/passwd

# Multiple fields
cut -d':' -f1,3,6 /etc/passwd

# Field range
cut -d',' -f2-4 data.csv

# Character position
cut -c1-10 file.txt

# Tab delimiter (default)
cut -f2 file.tsv
```

Example (/etc/passwd):
```bash
cat /etc/passwd | head -3
```
```
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
```

```bash
cut -d':' -f1,6 /etc/passwd | head -3
```
```
root:/root
daemon:/usr/sbin
ubuntu:/home/ubuntu
```

---

## 4. sort - Sorting

### Basic Usage

```bash
# Basic sort (alphabetical)
sort file.txt

# Reverse sort
sort -r file.txt

# Numeric sort
sort -n numbers.txt

# Remove duplicates
sort -u file.txt
```

### Main Options

| Option | Description |
|--------|-------------|
| `-r` | Reverse order |
| `-n` | Numeric sort |
| `-k` | Sort by specific field (key) |
| `-t` | Specify delimiter |
| `-u` | Remove duplicates (unique) |
| `-h` | Human-readable sizes |

```bash
# Numeric sort
sort -n scores.txt

# Reverse numeric sort
sort -rn scores.txt

# Sort by 2nd field (comma delimiter)
sort -t',' -k2 data.csv

# Numeric sort by 3rd field
sort -t':' -k3 -n /etc/passwd

# Human-readable size sort
du -h | sort -h

# Reverse sort by file size
ls -lh | sort -k5 -hr
```

---

## 5. uniq - Remove Duplicates

uniq only handles **consecutive** duplicates, so it's usually used with sort.

```bash
# Remove consecutive duplicates
uniq file.txt

# Show count of duplicates
uniq -c file.txt

# Show only duplicated lines
uniq -d file.txt

# Show only unique lines
uniq -u file.txt
```

```bash
# Use with sort
sort file.txt | uniq

# Count duplicates then sort
sort file.txt | uniq -c | sort -rn

# Top 10 most frequent IPs
cat access.log | cut -d' ' -f1 | sort | uniq -c | sort -rn | head -10
```

---

## 6. wc - Count

```bash
# Lines, words, bytes
wc file.txt
```

Output:
```
  100   500  3000 file.txt
   │     │     │
   │     │     └── Byte count
   │     └── Word count
   └── Line count
```

```bash
# Line count only
wc -l file.txt

# Word count only
wc -w file.txt

# Byte count only
wc -c file.txt

# Multiple files
wc -l *.txt

# With pipe
cat /etc/passwd | wc -l
```

---

## 7. sed - Stream Editor

sed is a text transformation tool.

### Basic Substitution

```bash
# Syntax: s/pattern/replacement/flags
sed 's/old/new/' file.txt        # First occurrence per line
sed 's/old/new/g' file.txt       # All occurrences (global)
sed 's/old/new/gi' file.txt      # Case insensitive
```

### Main Options

| Option | Description |
|--------|-------------|
| `-i` | In-place edit (modify file directly) |
| `-e` | Multiple commands |
| `-n` | Suppress output |

```bash
# Modify file directly
sed -i 's/old/new/g' file.txt

# Modify with backup
sed -i.bak 's/old/new/g' file.txt

# Multiple substitutions
sed -e 's/a/A/g' -e 's/b/B/g' file.txt

# Substitute only specific lines
sed '5s/old/new/' file.txt       # Line 5 only
sed '1,10s/old/new/g' file.txt   # Lines 1-10
```

### Line Deletion

```bash
# Delete specific lines
sed '5d' file.txt               # Delete line 5
sed '1,5d' file.txt             # Delete lines 1-5
sed '/pattern/d' file.txt       # Delete lines containing pattern

# Delete empty lines
sed '/^$/d' file.txt

# Delete comment lines
sed '/^#/d' config.conf
```

### Line Printing

```bash
# Print specific lines
sed -n '5p' file.txt            # Line 5 only
sed -n '1,10p' file.txt         # Lines 1-10
sed -n '/pattern/p' file.txt    # Lines containing pattern

# With line numbers
sed -n '=;p' file.txt
```

---

## 8. awk - Pattern Processing

awk is a programming language for text processing.

### Basic Structure

```bash
awk 'pattern { action }' file
```

### Field Variables

| Variable | Description |
|----------|-------------|
| `$0` | Entire line |
| `$1` | First field |
| `$2` | Second field |
| `$NF` | Last field |
| `NR` | Current line number |
| `NF` | Number of fields |

```bash
# Print first field
awk '{print $1}' file.txt

# Multiple fields
awk '{print $1, $3}' file.txt

# Specify delimiter
awk -F':' '{print $1, $6}' /etc/passwd

# Last field
awk '{print $NF}' file.txt

# With line numbers
awk '{print NR, $0}' file.txt
```

### Conditional Output

```bash
# Conditional filtering
awk '$3 > 100 {print $0}' data.txt

# Pattern matching
awk '/error/ {print $0}' log.txt

# Specific field pattern
awk '$1 ~ /^192/ {print $0}' access.log

# Multiple conditions
awk '$2 > 50 && $3 < 100 {print $1}' data.txt
```

### Calculations

```bash
# Sum
awk '{sum += $1} END {print sum}' numbers.txt

# Average
awk '{sum += $1; count++} END {print sum/count}' numbers.txt

# Maximum
awk 'BEGIN {max=0} $1 > max {max=$1} END {print max}' numbers.txt
```

### Formatting

```bash
# Formatted output
awk '{printf "%-10s %5d\n", $1, $2}' data.txt

# Add header
awk 'BEGIN {print "Name\tScore"} {print $1"\t"$2}' data.txt
```

---

## 9. Pipes and Redirection

### Pipe (|)

Passes command output to another command's input.

```bash
# Connect commands
ls -l | grep ".txt"
cat file.txt | sort | uniq
ps aux | grep nginx | grep -v grep
```

### Output Redirection

| Symbol | Description |
|--------|-------------|
| `>` | Redirect to file (overwrite) |
| `>>` | Append to file |
| `2>` | Redirect error output |
| `2>&1` | Redirect error to standard output |
| `&>` | Redirect both standard and error (bash) |

```bash
# Output to file
ls -l > filelist.txt

# Append to file
echo "new line" >> file.txt

# Error only to file
command 2> error.log

# Both output and error
command > output.txt 2>&1
command &> all.log

# Ignore errors
command 2>/dev/null

# Ignore all output
command > /dev/null 2>&1
```

### Input Redirection

```bash
# Input from file
sort < unsorted.txt

# Here Document
cat << EOF
Multiple lines
of text
input
EOF
```

---

## 10. Practice Exercises

### Exercise 1: Log Analysis

```bash
# Create sample log
cat << 'EOF' > access.log
192.168.1.10 - - [23/Jan/2024:10:15:32] "GET /index.html" 200
192.168.1.20 - - [23/Jan/2024:10:15:33] "GET /api/users" 200
192.168.1.10 - - [23/Jan/2024:10:15:34] "POST /api/login" 401
192.168.1.30 - - [23/Jan/2024:10:15:35] "GET /style.css" 200
192.168.1.10 - - [23/Jan/2024:10:15:36] "GET /api/data" 500
192.168.1.20 - - [23/Jan/2024:10:15:37] "GET /index.html" 200
EOF

# Find errors (4xx, 5xx)
grep -E " [45][0-9]{2}$" access.log

# Requests per IP
cut -d' ' -f1 access.log | sort | uniq -c | sort -rn

# Statistics by status code
awk '{print $NF}' access.log | sort | uniq -c | sort -rn
```

### Exercise 2: Extract User Information

```bash
# Regular users only (UID >= 1000)
awk -F':' '$3 >= 1000 {print $1, $6}' /etc/passwd

# User count by shell
cut -d':' -f7 /etc/passwd | sort | uniq -c | sort -rn

# Users in /home
grep "/home/" /etc/passwd | cut -d':' -f1
```

### Exercise 3: Data Transformation

```bash
# Create CSV
cat << 'EOF' > data.csv
name,score,grade
Alice,95,A
Bob,82,B
Charlie,78,C
David,91,A
EOF

# Score sum
awk -F',' 'NR>1 {sum+=$2} END {print "Total:", sum}' data.csv

# Average
awk -F',' 'NR>1 {sum+=$2; c++} END {print "Average:", sum/c}' data.csv

# A grade students
awk -F',' '$3=="A" {print $1}' data.csv

# Sort by score descending
sort -t',' -k2 -rn data.csv | head -5
```

### Exercise 4: Text Transformation

```bash
# All lowercase to uppercase
cat file.txt | tr 'a-z' 'A-Z'

# Replace specific word
sed 's/error/ERROR/g' log.txt

# Multiple replacements
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt

# Remove empty lines
sed '/^$/d' file.txt

# Spaces to tabs
sed 's/  */\t/g' file.txt
```

### Exercise 5: Complex Pipelines

```bash
# Top 10 largest files
find /var/log -type f -exec ls -l {} \; 2>/dev/null | \
  sort -k5 -rn | head -10

# Memory usage of specific process
ps aux | grep nginx | grep -v grep | \
  awk '{sum += $6} END {print sum/1024 " MB"}'

# Filter errors from real-time log
tail -f /var/log/syslog | grep --line-buffered -i error
```

---

**Previous**: [File and Directory Management](./03_File_Directory_Management.md) | **Next**: [Permissions and Ownership](./05_Permissions_Ownership.md)
