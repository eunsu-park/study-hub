#!/usr/bin/env bash
# Claude Code — Getting Started Workflow Example
# Demonstrates the read-edit-test-commit loop

# === 1. Installation ===
# npm install -g @anthropic-ai/claude-code
# claude --version

# === 2. Authentication ===
# claude login
# Authenticates via browser OAuth — stores token in ~/.claude/

# === 3. Basic Session ===
# Start an interactive session:
#   claude
#
# Start with a prompt:
#   claude "explain this codebase"
#
# Start in a specific directory:
#   cd ~/projects/myapp && claude

# === 4. Core Slash Commands ===
# /help          — Show all commands
# /clear         — Clear conversation context
# /compact       — Summarize and compress context
# /model         — Switch between models (opus, sonnet, haiku)
# /cost          — Show token usage and cost
# /vim           — Toggle vim keybindings

# === 5. Non-Interactive Mode ===
# Pipe input:
#   echo "fix the type error in src/utils.ts" | claude

# With a specific file:
#   claude -p "review this file for bugs" < src/auth.py

# Output only (no interactive UI):
#   claude --print "explain what this function does" < src/parser.js

# === 6. Read-Edit-Test-Commit Loop ===
# This is the recommended workflow:
#
# Step 1: Claude reads the relevant files
#   > "Look at the auth middleware in src/middleware/"
#
# Step 2: Claude proposes edits (you approve/reject each)
#   > "Add rate limiting to the login endpoint"
#
# Step 3: Run tests to verify
#   > "Run the auth tests"
#
# Step 4: Commit when satisfied
#   > /commit
#   or: "commit these changes with a descriptive message"

# === 7. Useful Flags ===
# claude --model opus      # Use Opus for complex tasks
# claude --allowedTools "Read,Grep,Glob"  # Restrict tools
# claude --verbose         # Show tool calls and details
