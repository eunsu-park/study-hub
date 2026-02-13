# Lesson 07: String Processing and Text Manipulation

**Difficulty**: ⭐⭐⭐

**Previous**: [06_IO_and_Redirection.md](./06_IO_and_Redirection.md) | **Next**: [08_Regex_in_Bash.md](./08_Regex_in_Bash.md)

---

## 1. Built-in String Operations Review

Before exploring external tools, let's review bash's powerful built-in string operations.

### 1.1 Parameter Expansion Quick Reference

```bash
#!/bin/bash

text="Hello World"

# Length
echo "${#text}"  # Output: 11

# Substring extraction
echo "${text:0:5}"   # Output: Hello
echo "${text:6}"     # Output: World
echo "${text: -5}"   # Output: World (note the space before -)

# Remove from beginning (shortest match)
filename="path/to/file.txt"
echo "${filename#*/}"    # Output: to/file.txt

# Remove from beginning (longest match)
echo "${filename##*/}"   # Output: file.txt

# Remove from end (shortest match)
echo "${filename%.*}"    # Output: path/to/file

# Remove from end (longest match)
echo "${filename%%/*}"   # Output: path
```

### 1.2 String Replacement

```bash
#!/bin/bash

text="foo bar foo baz foo"

# Replace first occurrence
echo "${text/foo/FOO}"    # Output: FOO bar foo baz foo

# Replace all occurrences
echo "${text//foo/FOO}"   # Output: FOO bar FOO baz FOO

# Remove first occurrence
echo "${text/foo}"        # Output:  bar foo baz foo

# Remove all occurrences
echo "${text//foo}"       # Output:  bar  baz

# Replace at beginning
echo "${text/#foo/START}" # Output: START bar foo baz foo

# Replace at end
text2="foo bar foo"
echo "${text2/%foo/END}"  # Output: foo bar END
```

### 1.3 Case Conversion

```bash
#!/bin/bash

text="Hello World"

# Convert to lowercase
echo "${text,,}"          # Output: hello world
echo "${text,}"           # Output: hello World (first char only)

# Convert to uppercase
echo "${text^^}"          # Output: HELLO WORLD
echo "${text^}"           # Output: Hello World (first char only)

# Toggle case (first char)
echo "${text~}"

# Toggle case (all chars)
echo "${text~~}"
```

### 1.4 String Concatenation and Repetition

```bash
#!/bin/bash

# Concatenation
first="Hello"
last="World"
full="$first $last"
echo "$full"  # Output: Hello World

# Append to variable
message="Hello"
message+=" World"
echo "$message"  # Output: Hello World

# String repetition (using printf)
repeat_string() {
    local string=$1
    local count=$2
    printf "%${count}s" | tr ' ' "$string"
}

echo "$(repeat_string '=' 40)"  # Output: ========================================

# Alternative: using bash loops
repeat_string2() {
    local string=$1
    local count=$2
    local result=""
    for ((i=0; i<count; i++)); do
        result+="$string"
    done
    echo "$result"
}

echo "$(repeat_string2 '-' 20)"  # Output: --------------------
```

### 1.5 String Comparison

```bash
#!/bin/bash

str1="hello"
str2="world"

# Equality
[[ $str1 == $str2 ]] && echo "Equal" || echo "Not equal"

# Inequality
[[ $str1 != $str2 ]] && echo "Different"

# Lexicographic comparison
[[ $str1 < $str2 ]] && echo "$str1 comes before $str2"
[[ $str1 > $str2 ]] && echo "$str1 comes after $str2"

# Check if empty
[[ -z $str1 ]] && echo "Empty" || echo "Not empty"

# Check if not empty
[[ -n $str1 ]] && echo "Not empty"

# Pattern matching
[[ $str1 == h* ]] && echo "Starts with h"
[[ $str1 == *o ]] && echo "Ends with o"
```

## 2. printf Formatting

The `printf` command provides powerful string formatting capabilities.

### 2.1 Basic Format Specifiers

```bash
#!/bin/bash

# String
printf "%s\n" "Hello World"

# Integer
printf "%d\n" 42

# Floating point
printf "%f\n" 3.14159
printf "%.2f\n" 3.14159  # Output: 3.14

# Hexadecimal
printf "%x\n" 255   # Output: ff
printf "%X\n" 255   # Output: FF

# Octal
printf "%o\n" 64    # Output: 100

# Character (ASCII)
printf "%c\n" 65    # Output: A
```

### 2.2 Width and Precision

```bash
#!/bin/bash

# Minimum width (right-aligned)
printf "%10s\n" "Hello"     # Output:      Hello

# Left-aligned
printf "%-10s\n" "Hello"    # Output: Hello

# Zero-padded numbers
printf "%05d\n" 42          # Output: 00042

# Precision for floats
printf "%.3f\n" 3.14159     # Output: 3.142

# Width and precision together
printf "%10.2f\n" 3.14159   # Output:       3.14
```

### 2.3 Building Formatted Tables

```bash
#!/bin/bash

# Print table header
printf "%-15s %-10s %10s\n" "Name" "Status" "Count"
printf "%-15s %-10s %10s\n" "===============" "==========" "=========="

# Print data rows
printf "%-15s %-10s %10d\n" "Alice" "Active" 42
printf "%-15s %-10s %10d\n" "Bob" "Inactive" 17
printf "%-15s %-10s %10d\n" "Charlie" "Active" 93

# Output:
# Name            Status          Count
# =============== ==========     ==========
# Alice           Active             42
# Bob             Inactive           17
# Charlie         Active             93
```

### 2.4 printf to Variable

```bash
#!/bin/bash

# Store formatted string in variable
printf -v timestamp "%(%Y-%m-%d %H:%M:%S)T" -1
echo "Current time: $timestamp"

# Format complex strings
printf -v sql_query "SELECT * FROM %s WHERE id = %d" "users" 42
echo "$sql_query"

# Build CSV line
printf -v csv_line "%s,%d,%.2f" "Product A" 100 19.99
echo "$csv_line"
```

### 2.5 Repeating Patterns with printf

```bash
#!/bin/bash

# Print horizontal line
printf '=%.0s' {1..50}
echo

# Print formatted separator
printf '%*s\n' 50 | tr ' ' '-'

# Create progress bar
create_progress_bar() {
    local percent=$1
    local width=50
    local filled=$((percent * width / 100))

    printf "["
    printf "%${filled}s" | tr ' ' '#'
    printf "%$((width - filled))s" | tr ' ' '-'
    printf "] %3d%%\n" "$percent"
}

create_progress_bar 75
# Output: [#####################################---------------] 75%
```

### 2.6 Practical Formatting Examples

```bash
#!/bin/bash

# Format currency
format_currency() {
    local amount=$1
    printf "$%'.2f\n" "$amount"
}

format_currency 1234567.89  # Output: $1,234,567.89

# Format file sizes
format_size() {
    local size=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit=0

    while ((size > 1024 && unit < 4)); do
        size=$((size / 1024))
        ((unit++))
    done

    printf "%.2f %s\n" "$size" "${units[$unit]}"
}

format_size 1048576  # Output: 1.00 MB

# Format duration (seconds to HH:MM:SS)
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))

    printf "%02d:%02d:%02d\n" "$hours" "$minutes" "$secs"
}

format_duration 3665  # Output: 01:01:05
```

## 3. tr Command

The `tr` (translate) command performs character-by-character transformations.

### 3.1 Character Translation

```bash
#!/bin/bash

# Translate characters
echo "hello" | tr 'a-z' 'A-Z'  # Output: HELLO
echo "WORLD" | tr 'A-Z' 'a-z'  # Output: world

# Specific character replacement
echo "hello" | tr 'l' 'L'      # Output: heLLo

# Multiple replacements
echo "hello world" | tr 'elo' 'ELO'  # Output: hELLO wOrLd

# Rotate characters (ROT13)
echo "Hello World" | tr 'A-Za-z' 'N-ZA-Mn-za-m'  # Output: Uryyb Jbeyq
```

### 3.2 Deleting Characters

```bash
#!/bin/bash

# Delete specific characters
echo "hello123world456" | tr -d '0-9'  # Output: helloworld

# Delete whitespace
echo "  hello   world  " | tr -d ' '   # Output: helloworld

# Delete newlines (join lines)
cat multiline.txt | tr -d '\n'

# Remove all vowels
echo "Hello World" | tr -d 'aeiouAEIOU'  # Output: Hll Wrld

# Remove punctuation
echo "Hello, World!" | tr -d '[:punct:]'  # Output: Hello World
```

### 3.3 Squeezing Characters

```bash
#!/bin/bash

# Squeeze repeated characters
echo "hello    world" | tr -s ' '     # Output: hello world

# Squeeze multiple spaces to single space
echo "too    many     spaces" | tr -s '[:space:]' ' '

# Remove duplicate blank lines
cat file.txt | tr -s '\n'

# Squeeze specific characters
echo "booook" | tr -s 'o'              # Output: bok
```

### 3.4 Complement Set

```bash
#!/bin/bash

# Keep only alphanumeric characters (delete everything else)
echo "Hello, World! 123" | tr -cd '[:alnum:]'  # Output: HelloWorld123

# Keep only digits
echo "Price: $19.99" | tr -cd '0-9'            # Output: 1999

# Remove all non-printable characters
cat file.txt | tr -cd '[:print:]\n'
```

### 3.5 Character Classes

```bash
#!/bin/bash

# Available character classes
# [:alnum:]  - Alphanumeric characters
# [:alpha:]  - Alphabetic characters
# [:digit:]  - Digits
# [:lower:]  - Lowercase letters
# [:upper:]  - Uppercase letters
# [:space:]  - Whitespace characters
# [:punct:]  - Punctuation characters
# [:print:]  - Printable characters

# Examples
echo "Hello123" | tr '[:lower:]' '[:upper:]'   # Output: HELLO123
echo "ABC def" | tr '[:upper:]' '[:lower:]'    # Output: abc def
echo "Hello World" | tr -d '[:space:]'         # Output: HelloWorld
echo "test@email.com" | tr -cd '[:alnum:]@.'   # Keep only valid email chars
```

### 3.6 Practical tr Examples

```bash
#!/bin/bash

# Convert DOS/Windows line endings to Unix
tr -d '\r' < dos_file.txt > unix_file.txt

# Create a URL slug from a title
echo "My Blog Post Title!" | tr '[:upper:] ' '[:lower:]-' | tr -cd '[:alnum:]-'
# Output: my-blog-post-title

# Extract phone number digits
echo "Phone: (555) 123-4567" | tr -cd '0-9'    # Output: 5551234567

# Remove control characters from text
cat file.txt | tr -d '[:cntrl:]'

# Convert spaces to underscores in filename
filename="My Document.txt"
new_filename=$(echo "$filename" | tr ' ' '_')
echo "$new_filename"  # Output: My_Document.txt
```

## 4. cut Command

The `cut` command extracts fields or characters from each line.

### 4.1 Character Extraction

```bash
#!/bin/bash

# Extract specific characters
echo "Hello World" | cut -c 1-5      # Output: Hello
echo "Hello World" | cut -c 7-       # Output: World
echo "Hello World" | cut -c -5       # Output: Hello
echo "Hello World" | cut -c 1,7      # Output: HW

# Extract multiple ranges
echo "abcdefghij" | cut -c 1-3,5-7   # Output: abcefg
```

### 4.2 Field Extraction with Delimiter

```bash
#!/bin/bash

# CSV parsing
echo "Alice,30,Engineer" | cut -d',' -f1     # Output: Alice
echo "Alice,30,Engineer" | cut -d',' -f2     # Output: 30
echo "Alice,30,Engineer" | cut -d',' -f1,3   # Output: Alice,Engineer

# Tab-delimited (default)
echo -e "A\tB\tC\tD" | cut -f2               # Output: B

# Extract from file
cut -d':' -f1,3 /etc/passwd  # Extract username and UID

# Multiple fields
echo "one:two:three:four:five" | cut -d':' -f2-4  # Output: two:three:four
```

### 4.3 Byte Extraction

```bash
#!/bin/bash

# Extract bytes (similar to characters for ASCII)
echo "Hello" | cut -b 1-3   # Output: Hel

# Useful for binary data or multi-byte characters
# Note: -b treats multi-byte UTF-8 as separate bytes
```

### 4.4 Complement (Output Suppression)

```bash
#!/bin/bash

# Output all fields except specified ones
echo "A,B,C,D,E" | cut -d',' -f1-3 --complement  # Output: D,E
```

### 4.5 Practical cut Examples

```bash
#!/bin/bash

# Extract IP addresses from log
cut -d' ' -f1 access.log | sort -u

# Get list of users from /etc/passwd
cut -d':' -f1 /etc/passwd

# Extract extension from filename
echo "document.pdf" | rev | cut -d'.' -f1 | rev  # Output: pdf

# Parse command output
ps aux | tail -n +2 | cut -c 66-  # Extract command column

# Extract date from timestamp
echo "2024-02-13 15:30:45" | cut -d' ' -f1  # Output: 2024-02-13

# Parse CSV with specific columns
cut -d',' -f2,4,6 data.csv > extracted.csv
```

## 5. paste and join

These commands merge data from multiple files.

### 5.1 paste Command

```bash
#!/bin/bash

# Merge files side by side
# file1.txt: A B C
# file2.txt: 1 2 3
paste file1.txt file2.txt
# Output:
# A    1
# B    2
# C    3

# Custom delimiter
paste -d',' file1.txt file2.txt
# Output:
# A,1
# B,2
# C,3

# Serial mode (all lines from first file, then second)
paste -s file1.txt file2.txt
# Output:
# A    B    C
# 1    2    3

# Merge multiple files
paste file1.txt file2.txt file3.txt

# Create CSV from multiple files
paste -d',' names.txt ages.txt cities.txt > output.csv
```

### 5.2 join Command

```bash
#!/bin/bash

# Join files on common field
# users.txt:    passwords.txt:
# 1 alice       1 pass123
# 2 bob         2 pass456
# 3 charlie     3 pass789

join users.txt passwords.txt
# Output:
# 1 alice pass123
# 2 bob pass456
# 3 charlie pass789

# Custom delimiter
join -t',' users.csv passwords.csv

# Join on different fields
join -1 2 -2 1 file1.txt file2.txt  # file1 field 2, file2 field 1

# Outer join (include unmatched lines)
join -a1 file1.txt file2.txt  # Include unmatched from file1
join -a2 file1.txt file2.txt  # Include unmatched from file2
join -a1 -a2 file1.txt file2.txt  # Full outer join

# Specify output format
join -o 1.1,1.2,2.2 file1.txt file2.txt
```

### 5.3 Practical Examples

```bash
#!/bin/bash

# Combine first and last names
paste -d' ' first_names.txt last_names.txt > full_names.txt

# Create table from columnar data
paste -d'|' col1.txt col2.txt col3.txt | column -t -s'|'

# Join user info with login history
sort -k1 users.txt > users_sorted.txt
sort -k1 logins.txt > logins_sorted.txt
join -t',' users_sorted.txt logins_sorted.txt > user_logins.txt

# Transpose data (rows to columns)
paste -s -d',' data.txt

# Create numbered list
paste -d' ' <(seq 1 10) items.txt
# Output:
# 1 item1
# 2 item2
# ...
```

## 6. column Command

The `column` command formats output as a table.

### 6.1 Basic Column Formatting

```bash
#!/bin/bash

# Auto-format as table
cat <<EOF | column -t
Name Age City
Alice 30 NYC
Bob 25 LA
Charlie 35 Chicago
EOF
# Output:
# Name     Age  City
# Alice    30   NYC
# Bob      25   LA
# Charlie  35   Chicago

# Custom separator
echo -e "A,B,C\n1,2,3\n4,5,6" | column -t -s','
# Output:
# A  B  C
# 1  2  3
# 4  5  6
```

### 6.2 Fill Columns Before Rows

```bash
#!/bin/bash

# Create columns (newspaper style)
seq 1 20 | column -c 40
# Output (approximate):
# 1  5   9   13  17
# 2  6   10  14  18
# 3  7   11  15  19
# 4  8   12  16  20
```

### 6.3 JSON Formatting

```bash
#!/bin/bash

# Format JSON as table (requires column with -J)
# Note: GNU column has -J flag for JSON, BSD column doesn't

# Alternative: use jq to prepare data, then column
jq -r '.[] | [.name, .age, .city] | @tsv' data.json | column -t
```

### 6.4 Practical Examples

```bash
#!/bin/bash

# Format command output
ps aux | head -n 10 | column -t

# Create aligned configuration file
cat > config.conf <<EOF
port=8080
host=localhost
debug=true
workers=4
EOF

cat config.conf | column -t -s'='
# Output:
# port     8080
# host     localhost
# debug    true
# workers  4

# Format CSV data nicely
column -t -s',' data.csv

# Create aligned menu
cat <<MENU | column -t
1|Start|Launch the application
2|Stop|Terminate the application
3|Restart|Restart the application
4|Status|Check application status
MENU
```

## 7. JSON Processing with jq

`jq` is a powerful command-line JSON processor.

### 7.1 Basic Filters

```bash
#!/bin/bash

# Pretty-print JSON
echo '{"name":"Alice","age":30}' | jq '.'

# Extract field
echo '{"name":"Alice","age":30}' | jq '.name'  # Output: "Alice"

# Extract nested field
echo '{"user":{"name":"Alice","age":30}}' | jq '.user.name'  # Output: "Alice"

# Array element
echo '["a","b","c"]' | jq '.[1]'  # Output: "b"

# Array slice
echo '[1,2,3,4,5]' | jq '.[2:4]'  # Output: [3,4]
```

### 7.2 Array Operations

```bash
#!/bin/bash

# Iterate array
echo '[1,2,3]' | jq '.[]'
# Output:
# 1
# 2
# 3

# Map over array
echo '[1,2,3]' | jq 'map(. * 2)'  # Output: [2,4,6]

# Filter array
echo '[1,2,3,4,5]' | jq 'map(select(. > 2))'  # Output: [3,4,5]

# Array length
echo '[1,2,3,4,5]' | jq 'length'  # Output: 5

# Sum array
echo '[1,2,3,4,5]' | jq 'add'  # Output: 15

# Get unique values
echo '[1,2,2,3,3,3]' | jq 'unique'  # Output: [1,2,3]

# Sort array
echo '[3,1,2]' | jq 'sort'  # Output: [1,2,3]
```

### 7.3 Object Operations

```bash
#!/bin/bash

# Get keys
echo '{"a":1,"b":2,"c":3}' | jq 'keys'  # Output: ["a","b","c"]

# Get values
echo '{"a":1,"b":2,"c":3}' | jq '.[]'
# Output:
# 1
# 2
# 3

# Check if key exists
echo '{"a":1,"b":2}' | jq 'has("a")'  # Output: true

# Add field
echo '{"a":1}' | jq '. + {b: 2}'  # Output: {"a":1,"b":2}

# Delete field
echo '{"a":1,"b":2}' | jq 'del(.b)'  # Output: {"a":1}
```

### 7.4 Conditional Logic

```bash
#!/bin/bash

# If-then-else
echo '{"age":25}' | jq 'if .age >= 18 then "adult" else "minor" end'

# Select with condition
echo '[{"name":"Alice","age":30},{"name":"Bob","age":17}]' | \
    jq '.[] | select(.age >= 18)'
# Output: {"name":"Alice","age":30}

# Multiple conditions
echo '[1,2,3,4,5]' | jq '.[] | select(. > 2 and . < 5)'
# Output:
# 3
# 4
```

### 7.5 String Interpolation

```bash
#!/bin/bash

# String interpolation
echo '{"first":"Alice","last":"Smith"}' | \
    jq '"\(.first) \(.last)"'
# Output: "Alice Smith"

# Build object
echo '{"name":"Alice"}' | \
    jq '{greeting: "Hello, \(.name)!"}'
# Output: {"greeting":"Hello, Alice!"}
```

### 7.6 Practical jq Examples

```bash
#!/bin/bash

# Parse API response
curl -s https://api.github.com/users/torvalds | jq '.name, .location, .public_repos'

# Extract specific fields from array of objects
jq '.users[] | {name, email}' users.json

# Create CSV from JSON
jq -r '.[] | [.name, .age, .city] | @csv' data.json

# Filter and transform
jq '.users | map(select(.active == true) | {name, email})' data.json

# Group by field
jq 'group_by(.category)' items.json

# Flatten nested structure
jq '[.users[].orders[]] | length' data.json

# Update field
jq '.users[] |= if .name == "Alice" then .status = "admin" else . end' data.json

# Merge JSON files
jq -s '.[0] * .[1]' file1.json file2.json

# Pretty-print with custom indentation
jq --indent 4 '.' data.json

# Raw output (no quotes)
jq -r '.name' data.json

# Compact output
jq -c '.' data.json
```

## 8. YAML Processing with yq

`yq` is a YAML processor similar to jq (note: there are multiple tools named yq; examples use mikefarah/yq).

### 8.1 Basic Operations

```bash
#!/bin/bash

# Read value
yq '.database.host' config.yaml

# Read nested array
yq '.servers[0].name' config.yaml

# Read all array elements
yq '.servers[].name' config.yaml

# Get keys
yq 'keys' config.yaml
```

### 8.2 Modifying YAML

```bash
#!/bin/bash

# Update value
yq '.database.port = 5432' config.yaml

# Add new field
yq '.newfield = "value"' config.yaml

# Delete field
yq 'del(.oldfield)' config.yaml

# Update in place
yq -i '.database.host = "localhost"' config.yaml

# Merge YAML files
yq eval-all 'select(fileIndex == 0) * select(fileIndex == 1)' file1.yaml file2.yaml
```

### 8.3 Format Conversion

```bash
#!/bin/bash

# YAML to JSON
yq -o=json '.' config.yaml

# JSON to YAML
yq -P '.' data.json

# Pretty-print YAML
yq '.' config.yaml
```

### 8.4 Practical Examples

```bash
#!/bin/bash

# Extract database credentials
DB_HOST=$(yq '.database.host' config.yaml)
DB_PORT=$(yq '.database.port' config.yaml)
DB_NAME=$(yq '.database.name' config.yaml)

# Update configuration
yq -i ".app.version = \"$NEW_VERSION\"" config.yaml
yq -i ".app.updated_at = \"$(date -Iseconds)\"" config.yaml

# Validate YAML syntax
if yq '.' config.yaml > /dev/null 2>&1; then
    echo "Valid YAML"
else
    echo "Invalid YAML"
fi

# Extract all service ports
yq '.services[].port' docker-compose.yml

# Build environment file from YAML
yq -o=props '.env' config.yaml > .env
```

## 9. Practical Text Processing Pipelines

### 9.1 Log Analysis

```bash
#!/bin/bash

# Extract error messages with timestamps
grep ERROR app.log | cut -d' ' -f1-2,4- | sort | uniq -c

# Count errors by type
grep ERROR app.log | cut -d':' -f3 | sort | uniq -c | sort -rn

# Top 10 IP addresses in access log
cut -d' ' -f1 access.log | sort | uniq -c | sort -rn | head -10

# Parse and reformat log entries
awk -F'[\\[\\]]' '{print $1, $2, $3}' access.log | \
    column -t > formatted.log
```

### 9.2 Data Transformation

```bash
#!/bin/bash

# Convert CSV to JSON
csv_to_json() {
    local csv_file=$1

    # Read header
    IFS=',' read -ra headers < "$csv_file"

    # Process data rows
    tail -n +2 "$csv_file" | while IFS=',' read -ra values; do
        echo "{"
        for i in "${!headers[@]}"; do
            printf '  "%s": "%s"' "${headers[$i]}" "${values[$i]}"
            [[ $i -lt $((${#headers[@]} - 1)) ]] && echo ","
        done
        echo "\n}"
    done | jq -s '.'
}

# Transform field names
jq 'map({username: .user, email_address: .email, full_name: "\(.first) \(.last)"})' \
    input.json > output.json

# Pivot data (rows to columns)
paste -s -d',' data.txt
```

### 9.3 Report Generation

```bash
#!/bin/bash

# Generate system report
generate_report() {
    cat <<EOF | column -t -s'|'
Metric|Value|Unit
---|---|---
CPU Usage|$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%|
Memory Used|$(free -m | awk 'NR==2{print $3}')|MB
Disk Usage|$(df -h / | awk 'NR==2{print $5}')|
Uptime|$(uptime -p)|
Load Average|$(uptime | awk -F'load average:' '{print $2}')|
EOF
}

# Create markdown table from CSV
csv_to_markdown() {
    local csv=$1

    # Header
    head -1 "$csv" | tr ',' '|' | sed 's/^/|/' | sed 's/$/|/'

    # Separator
    head -1 "$csv" | tr ',' '|' | sed 's/[^|]/-/g' | sed 's/^/|/' | sed 's/$/|/'

    # Data
    tail -n +2 "$csv" | tr ',' '|' | sed 's/^/|/' | sed 's/$/|/'
}
```

## Practice Problems

### Problem 1: Advanced CSV Processor
Create a script that:
- Reads CSV file with header row
- Validates each row (correct number of fields, data types)
- Supports filtering rows based on column values (e.g., age > 30)
- Supports sorting by multiple columns
- Supports selecting specific columns
- Outputs in CSV, JSON, or formatted table
- Handles quoted fields with commas correctly

### Problem 2: Log Parser and Analyzer
Build a log analysis tool that:
- Parses common log formats (Apache, Nginx, syslog)
- Extracts timestamp, level, message, source
- Generates statistics (error rate, top errors, time distribution)
- Creates timeline visualization using ASCII characters
- Supports filtering by time range, level, pattern
- Outputs report in markdown or HTML format

### Problem 3: Configuration File Converter
Write a tool that:
- Converts between JSON, YAML, TOML, INI, ENV formats
- Validates syntax for each format
- Preserves comments where possible
- Supports nested structures
- Handles arrays and complex types
- Can extract/update specific values via command-line
- Supports merging multiple config files

### Problem 4: Text Template Engine
Implement a template processor that:
- Reads template file with placeholders (e.g., {{variable}})
- Supports conditionals: {{#if condition}}...{{/if}}
- Supports loops: {{#each items}}...{{/each}}
- Supports includes: {{> include file.txt}}
- Reads variables from JSON/YAML file or environment
- Supports filters: {{variable|upper}}, {{variable|date}}
- Handles nested data structures

### Problem 5: Data Validation Framework
Create a validation tool that:
- Defines validation rules in YAML format
- Validates CSV/JSON/YAML data against rules
- Supports rules: required, type, range, pattern, length, custom
- Reports validation errors with line numbers and field names
- Supports cross-field validation (e.g., end_date > start_date)
- Generates validation report in multiple formats
- Can fix common issues automatically (trim spaces, convert case)

**Previous**: [06_IO_and_Redirection.md](./06_IO_and_Redirection.md) | **Next**: [08_Regex_in_Bash.md](./08_Regex_in_Bash.md)
