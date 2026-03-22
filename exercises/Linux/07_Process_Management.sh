#!/bin/bash
# Exercises for Lesson 07: Process Management
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Process Inspection ===
# Problem: Examine running processes using ps, top, and related tools.
exercise_1() {
    echo "=== Exercise 1: Process Inspection (ps, top, htop Patterns) ==="
    echo ""
    echo "Scenario: A server is running slow. You need to identify which processes"
    echo "are consuming the most CPU and memory resources."
    echo ""

    echo "--- Part A: ps for snapshot-based process analysis ---"
    echo "Solution:"
    echo "  ps aux                          # All processes, user-oriented format"
    echo "  ps aux --sort=-%mem | head -10  # Top 10 memory consumers"
    echo "  ps aux --sort=-%cpu | head -10  # Top 10 CPU consumers"
    echo "  ps -ef                          # Full-format listing (POSIX style)"
    echo "  ps -eo pid,ppid,user,%cpu,%mem,cmd --sort=-%cpu | head -15"
    echo ""
    echo "  Explanation:"
    echo "    'aux' = BSD-style: a=all users, u=user-oriented, x=include daemons."
    echo "    '-ef' = POSIX-style: -e=all processes, -f=full format."
    echo "    -o selects specific columns (custom output format)."
    echo "    --sort=-key sorts descending (- prefix). +key for ascending."
    echo "    Both styles work; aux is more common in practice."
    echo ""

    echo "--- Part B: Filter processes by name, PID, or user ---"
    echo "Solution:"
    echo "  ps aux | grep nginx             # Search by name (includes grep itself)"
    echo "  ps aux | grep '[n]ginx'         # Trick: avoids matching the grep process"
    echo "  pgrep -la nginx                 # Cleaner: list PIDs and command lines"
    echo "  ps -u www-data                  # All processes by user www-data"
    echo "  ps --ppid 1 -o pid,cmd          # All direct children of PID 1 (init)"
    echo ""
    echo "  Explanation:"
    echo "    grep '[n]ginx' uses a character class for 'n' — matches 'nginx' but"
    echo "    not 'grep [n]ginx' (the grep command itself). Classic shell trick."
    echo "    pgrep is purpose-built for process search: -l=list name, -a=full command."
    echo "    --ppid filters by parent PID, useful for seeing a service's workers."
    echo ""

    echo "--- Part C: Real-time monitoring with top ---"
    echo "Solution:"
    echo "  top                             # Interactive process monitor"
    echo "  top -b -n 1 | head -20         # Batch mode: one snapshot, first 20 lines"
    echo "  top -b -n 5 -d 2 > top.log     # Log 5 snapshots, 2-second intervals"
    echo ""
    echo "  Key top shortcuts (interactive mode):"
    echo "    P = sort by CPU    M = sort by memory    T = sort by time"
    echo "    k = kill process   r = renice process    q = quit"
    echo "    1 = toggle per-CPU display    c = show full command path"
    echo ""
    echo "  Explanation:"
    echo "    -b = batch mode (non-interactive, for piping/logging)."
    echo "    -n = number of iterations. -d = delay between updates."
    echo "    top is always available but htop (if installed) has better usability:"
    echo "    color-coded bars, mouse support, tree view, and easier filtering."
    echo ""

    # Safe read-only check on current system
    echo "--- Current top 5 CPU processes ---"
    if command -v ps &>/dev/null; then
        ps -eo pid,user,%cpu,%mem,cmd --sort=-%cpu 2>/dev/null | head -6
    fi
}

# === Exercise 2: Job Control ===
# Problem: Manage foreground and background jobs in the shell.
exercise_2() {
    echo "=== Exercise 2: Job Control (bg, fg, nohup, &) ==="
    echo ""
    echo "Scenario: You need to run a long-running data export while continuing"
    echo "to work in the same terminal, and ensure it survives if you disconnect."
    echo ""

    echo "--- Part A: Background execution with & ---"
    echo "Solution:"
    echo "  ./export-data.sh &              # Start in background"
    echo "  jobs                            # List background jobs in this shell"
    echo "  jobs -l                         # Include PIDs"
    echo ""
    echo "  Explanation:"
    echo "    & at the end runs the command in the background immediately."
    echo "    The shell prints [job_number] PID when backgrounding."
    echo "    Background jobs still output to the terminal (redirect to avoid):"
    echo "    ./export-data.sh > export.log 2>&1 &"
    echo ""

    echo "--- Part B: Suspend and resume with Ctrl+Z, fg, bg ---"
    echo "Solution:"
    echo "  # Press Ctrl+Z while a command is running → suspends (stops) it"
    echo "  bg                              # Resume suspended job in background"
    echo "  bg %2                           # Resume job #2 specifically"
    echo "  fg                              # Bring most recent background job to foreground"
    echo "  fg %1                           # Bring job #1 to foreground"
    echo ""
    echo "  Explanation:"
    echo "    Ctrl+Z sends SIGTSTP (terminal stop) — the process is PAUSED, not killed."
    echo "    bg resumes it in the background (sends SIGCONT)."
    echo "    fg brings it to the foreground."
    echo "    %N refers to job number from 'jobs' output."
    echo "    Common workflow: start a command, realize it's slow, Ctrl+Z, bg."
    echo ""

    echo "--- Part C: Survive disconnection with nohup and disown ---"
    echo "Solution:"
    echo "  nohup ./long-task.sh > task.log 2>&1 &     # Immune to hangup signal"
    echo "  disown %1                                   # Detach job #1 from shell"
    echo "  disown -a                                   # Detach all background jobs"
    echo ""
    echo "  Explanation:"
    echo "    When you close a terminal, the shell sends SIGHUP to all its jobs."
    echo "    nohup ignores SIGHUP and redirects output to nohup.out by default."
    echo "    disown removes a job from the shell's job table — the shell forgets it,"
    echo "    so no SIGHUP is sent on exit."
    echo "    Use nohup for planned long tasks; disown for jobs already running."
    echo ""

    echo "--- Part D: Modern alternatives ---"
    echo "Solution:"
    echo "  screen -S export ./export-data.sh    # Run in a screen session"
    echo "  screen -r export                      # Reattach later"
    echo ""
    echo "  tmux new -s export './export-data.sh' # Run in a tmux session"
    echo "  tmux attach -t export                  # Reattach later"
    echo ""
    echo "  Explanation:"
    echo "    screen/tmux create persistent terminal sessions that survive disconnection."
    echo "    They are superior to nohup because you can reattach and see live output."
    echo "    tmux is the modern choice (better keybindings, scripting, splits)."
    echo "    For one-off tasks, nohup suffices. For interactive work, use tmux."
}

# === Exercise 3: Signal Handling ===
# Problem: Send signals to processes and understand signal behavior.
exercise_3() {
    echo "=== Exercise 3: Signal Handling (kill, pkill, trap) ==="
    echo ""
    echo "Scenario: A web server process is not responding. You need to gracefully"
    echo "stop it, force-kill it if necessary, and write scripts that handle signals."
    echo ""

    echo "--- Part A: Sending signals with kill ---"
    echo "Solution:"
    echo "  kill 1234                       # Send SIGTERM (15) — graceful shutdown"
    echo "  kill -HUP 1234                  # Send SIGHUP (1) — reload configuration"
    echo "  kill -9 1234                    # Send SIGKILL (9) — force kill (last resort)"
    echo "  kill -0 1234                    # Check if process exists (no signal sent)"
    echo ""
    echo "  Explanation:"
    echo "    SIGTERM (15): default signal. Asks the process to shut down gracefully."
    echo "    The process can catch it, clean up resources, and exit."
    echo "    SIGHUP (1): traditionally means 'hangup'. Many daemons reload config."
    echo "    SIGKILL (9): uncatchable — kernel immediately terminates the process."
    echo "    Use SIGKILL only when SIGTERM fails after a reasonable wait."
    echo "    kill -0 is a safe existence check that sends no actual signal."
    echo ""

    echo "--- Part B: Pattern-based killing with pkill/killall ---"
    echo "Solution:"
    echo "  pkill nginx                     # Kill processes by name pattern"
    echo "  pkill -u www-data               # Kill all processes of a user"
    echo "  pkill -f 'python.*server.py'    # Match full command line (-f)"
    echo "  killall -HUP nginx              # Send HUP to all nginx processes"
    echo "  pkill -9 -P 1234               # Kill all children of PID 1234"
    echo ""
    echo "  Explanation:"
    echo "    pkill matches process names with patterns (regex by default)."
    echo "    -f matches against the full command line, not just the process name."
    echo "    -P matches by parent PID (kill all children of a process)."
    echo "    killall matches exact process names (not patterns)."
    echo "    WARNING: on Solaris, killall kills ALL processes. Use pkill instead."
    echo ""

    echo "--- Part C: Signal trapping in scripts ---"
    echo "Solution:"
    cat << 'SCRIPT'
#!/bin/bash
# Graceful shutdown example

TMPDIR=$(mktemp -d)

cleanup() {
    echo "Caught signal — cleaning up..."
    rm -rf "$TMPDIR"
    echo "Cleanup complete. Exiting."
    exit 0
}

# Trap SIGTERM and SIGINT (Ctrl+C) to run cleanup
trap cleanup SIGTERM SIGINT

echo "Running (PID: $$). Temp dir: $TMPDIR"
echo "Press Ctrl+C or send SIGTERM to test graceful shutdown."

# Simulate long-running work
while true; do
    echo "Working... $(date)"
    sleep 5
done
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    trap COMMAND SIGNAL — runs COMMAND when SIGNAL is received."
    echo "    Common traps: SIGTERM (graceful stop), SIGINT (Ctrl+C), EXIT (always)."
    echo "    trap cleanup EXIT — runs on ANY exit (normal, signal, error)."
    echo "    SIGKILL (9) CANNOT be trapped — that is why it is the last resort."
    echo ""

    echo "--- Part D: Common signals reference ---"
    echo "  Signal    Number  Default Action     Common Use"
    echo "  SIGHUP      1     Terminate          Reload config (daemons)"
    echo "  SIGINT      2     Terminate          Ctrl+C (interrupt)"
    echo "  SIGQUIT     3     Core dump          Ctrl+\\ (quit with dump)"
    echo "  SIGKILL     9     Terminate          Force kill (uncatchable)"
    echo "  SIGTERM    15     Terminate          Graceful shutdown (default)"
    echo "  SIGSTOP    19     Stop               Pause (uncatchable, like Ctrl+Z)"
    echo "  SIGCONT    18     Continue            Resume paused process"
    echo "  SIGUSR1    10     Terminate          User-defined (log rotation, etc.)"
    echo ""
    echo "Verification:"
    echo "  kill -l             # List all signal names and numbers"
    echo "  kill -l 15          # Show name for signal number 15 (TERM)"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
