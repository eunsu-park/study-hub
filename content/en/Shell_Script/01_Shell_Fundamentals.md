# Shell Fundamentals and Execution Environment

**Next**: [Parameter Expansion and Variable Attributes](./02_Parameter_Expansion.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare bash, sh, zsh, and fish shells in terms of features, POSIX compliance, and scripting suitability
2. Identify POSIX-compliant constructs and avoid common bashisms that break portability
3. Distinguish between login, interactive, non-interactive, and restricted shell modes
4. Trace the startup file loading order for different shell invocation modes
5. Explain exit codes and use them to build robust conditional logic
6. Configure shell behavior using `set` and `shopt` options
7. Write portable shebang lines using `/usr/bin/env`

---

Every shell script runs inside an execution environment that determines which features are available, which configuration files are loaded, and how errors are handled. Misunderstanding these fundamentals leads to scripts that work on your machine but fail in cron jobs, Docker containers, or other systems. This lesson gives you the foundational knowledge to write scripts that behave predictably across different environments.

## 1. Types of Shells

Different shells offer varying features and compatibility levels. Understanding these differences is crucial for writing portable scripts.

### Shell Comparison Table

| Feature | bash | sh (dash) | zsh | fish |
|---------|------|-----------|-----|------|
| POSIX compliant | Mostly | Yes | Mostly | No |
| Arrays | Yes | No | Yes (better) | Yes |
| Associative arrays | Yes (4.0+) | No | Yes | Yes |
| [[ ]] test | Yes | No | Yes | No |
| Process substitution | Yes | No | Yes | Yes |
| Here strings | Yes | No | Yes | Yes |
| Arithmetic (( )) | Yes | No | Yes | No |
| Command completion | Good | Basic | Excellent | Excellent |
| Startup performance | Medium | Fast | Slow | Medium |
| Scripting focus | Yes | Yes | Yes | No |
| Default on Debian/Ubuntu | bash | dash (/bin/sh) | bash | fish |
| Configuration syntax | bash | POSIX sh | zsh-extended | Fish-specific |

### When to Use Each Shell

**bash**: General-purpose scripting, most systems have it, good balance of features and portability.

```bash
#!/bin/bash
# Use bash for scripts needing arrays, [[ ]], or process substitution
declare -A config
config[host]="localhost"
config[port]=8080

if [[ -n "${config[host]}" ]]; then
    echo "Host: ${config[host]}"
fi
```

**sh (POSIX)**: Maximum portability, embedded systems, minimal environments.

```bash
#!/bin/sh
# POSIX-compliant script - no bash-isms
# No arrays, no [[, no process substitution

if [ -n "$HOST" ]; then
    echo "Host: $HOST"
fi

# Use case instead of [[ with regex
case "$filename" in
    *.txt) echo "Text file" ;;
    *.log) echo "Log file" ;;
esac
```

**zsh**: Interactive use, advanced completion, better array handling.

```bash
#!/bin/zsh
# zsh has more powerful array features
array=(one two three)
echo $array[1]  # zsh arrays are 1-indexed (bash uses 0-indexed)

# Advanced globbing
setopt extended_glob
files=(^*.txt)  # all files except .txt
```

**fish**: Interactive shell, user-friendly, NOT for portable scripts.

```fish
#!/usr/bin/fish
# Fish has different syntax - not POSIX compatible
set host localhost
set port 8080

if test -n "$host"
    echo "Host: $host"
end
```

## 2. POSIX Compliance

POSIX (Portable Operating System Interface) defines a standard for shell behavior. POSIX-compliant scripts run on any POSIX shell (sh, bash, dash, ksh, etc.).

### POSIX vs Bash-isms

| Feature | POSIX sh | bash Extension |
|---------|----------|----------------|
| Test command | `[ ]` | `[[ ]]` |
| String comparison | `[ "$a" = "$b" ]` | `[[ $a == $b ]]` |
| Regex matching | (use grep) | `[[ $str =~ regex ]]` |
| Arrays | Not supported | `arr=(1 2 3)` |
| Functions | `func() { }` | `function func { }` |
| Arithmetic | `expr`, `$(( ))` | `let`, `(( ))` |
| Process substitution | Not supported | `<(cmd)`, `>(cmd)` |
| Here strings | Not supported | `<<< "string"` |
| Local variables | Not in POSIX | `local var=value` |

### Writing Portable POSIX Scripts

```bash
#!/bin/sh
# POSIX-compliant script example

# Use [ ] instead of [[ ]]
if [ "$1" = "start" ]; then
    echo "Starting service..."
fi

# Use $(( )) for arithmetic (this IS POSIX)
count=0
count=$((count + 1))

# Use case for pattern matching
case "$filename" in
    *.tar.gz|*.tgz)
        echo "Compressed tarball"
        ;;
    *.zip)
        echo "ZIP archive"
        ;;
    *)
        echo "Unknown format"
        ;;
esac

# Avoid arrays - use positional parameters or temporary files
set -- "item1" "item2" "item3"
for item in "$@"; do
    echo "$item"
done

# Use command substitution $(cmd) not backticks
current_dir=$(pwd)

# Check command existence portably
if command -v docker >/dev/null 2>&1; then
    echo "Docker is installed"
fi
```

## 3. Shell Modes

Shells operate in different modes depending on how they are invoked. This affects which startup files are read.

### Login vs Non-Login Shells

**Login shell**: Started when you log in (SSH, console login, `bash --login`).

**Non-login shell**: Started from an existing session (opening terminal in GUI, running bash from bash).

Test if shell is login shell:

```bash
#!/bin/bash
# Check if running as login shell
if shopt -q login_shell; then
    echo "This is a login shell"
else
    echo "This is a non-login shell"
fi

# Alternative method
case "$-" in
    *l*) echo "Login shell" ;;
    *) echo "Non-login shell" ;;
esac
```

### Interactive vs Non-Interactive Shells

**Interactive**: Terminal attached, accepts user input (normal terminal session).

**Non-interactive**: Running scripts, no terminal interaction.

Test if shell is interactive:

```bash
#!/bin/bash
# Check if running interactively
if [[ $- == *i* ]]; then
    echo "Interactive shell"
else
    echo "Non-interactive shell (script)"
fi

# Alternative method
case "$-" in
    *i*) echo "Interactive" ;;
    *) echo "Non-interactive" ;;
esac

# Check if stdin is a terminal
if [ -t 0 ]; then
    echo "stdin is a terminal"
else
    echo "stdin is not a terminal (piped/redirected)"
fi
```

## 4. Startup Files Loading Order

The order in which bash reads configuration files depends on the shell mode.

### Startup Sequence Diagram

```
Login Shell (bash --login or SSH)
├── /etc/profile (system-wide)
│   └── /etc/profile.d/*.sh (if sourced by /etc/profile)
└── First found:
    ├── ~/.bash_profile
    ├── ~/.bash_login  (if ~/.bash_profile not found)
    └── ~/.profile     (if neither above found)
        └── (many .bash_profile files source ~/.bashrc)

Non-Login Interactive Shell (terminal window)
├── /etc/bash.bashrc (Debian/Ubuntu)
└── ~/.bashrc

Non-Interactive Shell (scripts)
├── $BASH_ENV (if set, file path to source)
└── (typically nothing)

Login Shell Exit
└── ~/.bash_logout
```

### Startup Files Purpose

| File | Purpose | Typical Contents |
|------|---------|------------------|
| `/etc/profile` | System-wide login settings | PATH, LANG, umask |
| `/etc/bash.bashrc` | System-wide interactive settings | PS1, aliases (Debian/Ubuntu) |
| `~/.bash_profile` | User login settings | Source ~/.bashrc, set PATH |
| `~/.bashrc` | User interactive settings | Aliases, functions, PS1 |
| `~/.profile` | POSIX login settings | Portable login settings |
| `~/.bash_logout` | Cleanup on logout | Clear screen, clean temp files |

### Example Startup File Structure

**~/.bash_profile** (login shell entry point):

```bash
# ~/.bash_profile - loaded by login shells

# Set PATH for user binaries
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

# Load .bashrc if it exists (for interactive login shells)
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# Login-specific settings
echo "Last login: $(date)" >> ~/.login_log
```

**~/.bashrc** (interactive shell settings):

```bash
# ~/.bashrc - loaded by interactive non-login shells

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# History settings
HISTCONTROL=ignoreboth
HISTSIZE=10000
HISTFILESIZE=20000
shopt -s histappend

# Prompt
PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

# Aliases
alias ll='ls -lah'
alias grep='grep --color=auto'

# Load functions from separate file
if [ -f ~/.bash_functions ]; then
    . ~/.bash_functions
fi
```

**~/.profile** (POSIX-compatible login settings):

```bash
# ~/.profile - POSIX-compatible login settings
# Used when bash is invoked as sh, or by other POSIX shells

# Set PATH
PATH="$HOME/bin:$PATH"
export PATH

# Environment variables
export EDITOR=vim
export PAGER=less

# If bash, source .bashrc
if [ -n "$BASH_VERSION" ]; then
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi
```

## 5. Exit Codes

Exit codes indicate whether a command succeeded or failed. By convention:

### Exit Code Conventions

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Misuse of shell builtin |
| 126 | Command found but not executable |
| 127 | Command not found |
| 128 | Invalid exit argument |
| 128+N | Fatal error signal N (130 = Ctrl+C (SIGINT=2)) |
| 255 | Exit status out of range |

### Using Exit Codes

```bash
#!/bin/bash

# Check exit code with $?
grep "pattern" file.txt
if [ $? -eq 0 ]; then
    echo "Pattern found"
else
    echo "Pattern not found"
fi

# Better: use command directly in if
if grep "pattern" file.txt > /dev/null; then
    echo "Pattern found"
fi

# Return custom exit codes from functions
validate_input() {
    local input="$1"

    if [ -z "$input" ]; then
        echo "Error: input is empty" >&2
        return 1
    fi

    if ! [[ "$input" =~ ^[0-9]+$ ]]; then
        echo "Error: input must be numeric" >&2
        return 2
    fi

    if [ "$input" -lt 0 ] || [ "$input" -gt 100 ]; then
        echo "Error: input must be between 0 and 100" >&2
        return 3
    fi

    return 0
}

# Use function and check exit code
if validate_input "42"; then
    echo "Input is valid"
else
    case $? in
        1) echo "Empty input" ;;
        2) echo "Not numeric" ;;
        3) echo "Out of range" ;;
    esac
fi

# Exit script with specific code
check_prerequisites() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Error: docker not found" >&2
        exit 127
    fi

    if [ ! -r /etc/config.conf ]; then
        echo "Error: config file not readable" >&2
        exit 1
    fi
}
```

### Exit Code Best Practices

```bash
#!/bin/bash
set -e  # Exit on error (but be careful with this)

# Explicitly handle errors
perform_backup() {
    local source="$1"
    local dest="$2"

    if ! tar czf "$dest" "$source" 2>/dev/null; then
        echo "Backup failed" >&2
        return 1
    fi

    echo "Backup successful"
    return 0
}

# Chain commands with proper error handling
if perform_backup "/data" "/backup/data.tar.gz"; then
    echo "Cleaning up old backups..."
    find /backup -name "*.tar.gz" -mtime +7 -delete
else
    echo "Backup failed, keeping old backups"
    exit 1
fi
```

## 6. Shell Options Overview

Shell options control shell behavior. Two commands manage options:

- **set**: POSIX-standard options
- **shopt**: Bash-specific extended options

### Important set Options

```bash
#!/bin/bash

# Show current options
echo "$-"  # e.g., "himBH" (each letter is an active option)

# Enable options
set -e  # Exit on error (errexit)
set -u  # Error on undefined variables (nounset)
set -o pipefail  # Pipeline fails if any command fails
set -x  # Print commands before executing (xtrace)

# Disable options
set +e  # Don't exit on error
set +x  # Stop printing commands

# Combine options
set -euo pipefail  # Common "strict mode"

# Example: noclobber prevents overwriting files
set -o noclobber
echo "test" > file.txt  # Creates file
echo "test" > file.txt  # Error: file exists
echo "test" >| file.txt  # Override noclobber with >|
```

### Common set Options

| Option | Short | Description |
|--------|-------|-------------|
| `-e` (errexit) | `-e` | Exit if command fails |
| `-u` (nounset) | `-u` | Error on undefined variable |
| `-x` (xtrace) | `-x` | Print commands before execution |
| `-o pipefail` | (long only) | Pipeline fails if any command fails |
| `-o noclobber` | `-C` | Prevent > from overwriting files |
| `-o noglob` | `-f` | Disable pathname expansion |
| `-o vi` | (long only) | Vi-style command line editing |
| `-o emacs` | (long only) | Emacs-style editing (default) |

### Important shopt Options

```bash
#!/bin/bash

# Enable bash extended options
shopt -s extglob  # Extended pattern matching
shopt -s globstar  # ** for recursive glob
shopt -s nullglob  # Non-matching globs expand to nothing
shopt -s dotglob  # Include hidden files in globs
shopt -s nocaseglob  # Case-insensitive globbing

# Disable options
shopt -u dotglob  # Exclude hidden files

# Check if option is set
if shopt -q nullglob; then
    echo "nullglob is enabled"
fi

# Example: nullglob
shopt -s nullglob
files=(*.txt)
if [ ${#files[@]} -eq 0 ]; then
    echo "No .txt files found"
else
    echo "Found ${#files[@]} .txt files"
fi

# Example: globstar
shopt -s globstar
# Find all Python files recursively
for file in **/*.py; do
    echo "$file"
done

# Example: extglob (covered more in Lesson 04)
shopt -s extglob
rm !(*.txt|*.log)  # Remove all except .txt and .log files
```

### Useful shopt Options

| Option | Description |
|--------|-------------|
| `extglob` | Extended pattern matching (!(pat), *(pat), etc.) |
| `globstar` | ** matches recursively |
| `nullglob` | Non-matching globs expand to null, not literal |
| `dotglob` | Include hidden files in pathname expansion |
| `nocaseglob` | Case-insensitive pathname expansion |
| `failglob` | Unmatched globs cause error |
| `checkjobs` | Check running jobs before exiting |
| `autocd` | Change directory by typing directory name |
| `cdspell` | Autocorrect minor cd errors |

### Strict Mode Example

```bash
#!/bin/bash
# Strict mode for safer scripts

set -euo pipefail
IFS=$'\n\t'

# -e: exit on error
# -u: error on undefined variable
# -o pipefail: pipeline fails if any command fails
# IFS: safer word splitting

# Now errors will stop the script
command_that_fails  # Script exits here
echo "This won't execute"

# To handle errors explicitly:
if ! command_that_might_fail; then
    echo "Command failed, handling error"
    # Do cleanup
    exit 1
fi
```

## 7. The env Command and #!/usr/bin/env bash

### Why Use #!/usr/bin/env bash

The shebang line tells the system which interpreter to use. Two approaches:

**Direct path**: `#!/bin/bash`
- Fast (no PATH search)
- Not portable (bash might be in /usr/local/bin)

**env approach**: `#!/usr/bin/env bash`
- Portable (finds bash in PATH)
- Standard in modern scripts
- Works across different systems

```bash
#!/usr/bin/env bash
# This finds bash wherever it is in PATH

# Check where bash is located
which bash

# On macOS: /bin/bash
# On FreeBSD: /usr/local/bin/bash
# On Nix: /nix/store/.../bin/bash
```

### Using env to Set Environment

```bash
#!/usr/bin/env -S bash -euo pipefail
# -S flag allows passing multiple arguments (GNU env 8.30+)

# Alternative for older env:
#!/usr/bin/env bash
set -euo pipefail
```

### env for Clean Environment

```bash
# Run command with clean environment
env -i bash --norc --noprofile

# Run with specific variables only
env -i HOME=/tmp USER=testuser bash

# Remove specific variables
env -u DISPLAY firefox

# Add variables
env FOO=bar ./script.sh

# Inspect environment
env | sort
```

### Script Template with Best Practices

```bash
#!/usr/bin/env bash
# Script: example.sh
# Description: Example script with best practices
# Author: Your Name
# Created: 2026-02-13

# Strict mode
set -euo pipefail
IFS=$'\n\t'

# Trap errors
trap 'echo "Error on line $LINENO" >&2' ERR

# Useful shell options
shopt -s nullglob globstar

# Constants
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Functions
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] <argument>

Options:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -d, --debug     Enable debug mode

Examples:
    $SCRIPT_NAME input.txt
    $SCRIPT_NAME -v input.txt
EOF
}

main() {
    # Main script logic
    echo "Running from: $SCRIPT_DIR"
    echo "Script name: $SCRIPT_NAME"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -d|--debug)
            set -x
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Run main function
main "$@"
```

## Practice Problems

### Problem 1: Shell Detection Script

Write a script that detects:
- Which shell it's running in (bash, zsh, sh, etc.)
- Whether it's a login or non-login shell
- Whether it's interactive or non-interactive
- The version of the shell

**Expected output**:
```
Shell: bash
Version: 5.1.16
Type: non-login, interactive
```

### Problem 2: Startup File Analyzer

Create a script that:
- Lists all bash startup files that exist on the system
- Shows the order they would be loaded for login vs non-login shells
- Displays the first 5 lines of each file
- Checks for common mistakes (like setting aliases in .bash_profile instead of .bashrc)

### Problem 3: Exit Code Logger

Write a function that wraps any command and:
- Logs the command being executed
- Captures and logs the exit code
- Logs execution time
- Appends to a log file: `timestamp | command | exit_code | duration`

**Example usage**:
```bash
log_command ls -la /nonexistent
# Should log: 2026-02-13 10:30:45 | ls -la /nonexistent | 2 | 0.003s
```

### Problem 4: Portable Script Checker

Create a script that analyzes another bash script and reports:
- Non-POSIX constructs used ([[ ]], arrays, etc.)
- Bashisms that would fail in sh
- Suggestions for making it more portable
- A "portability score" (0-100%)

**Hint**: Search for patterns like `[[`, `declare`, `function keyword`, etc.

### Problem 5: Environment Snapshot

Write a script that:
- Saves current environment variables to a file
- Saves current shell options (set -o, shopt -p) to a file
- Can restore the environment from the saved state
- Shows diff between saved and current state

**Usage**:
```bash
./envsnap.sh save snapshot.env
# ... make changes ...
./envsnap.sh diff snapshot.env
./envsnap.sh restore snapshot.env
```

---

**Next**: [Parameter Expansion and Variable Attributes](./02_Parameter_Expansion.md)
