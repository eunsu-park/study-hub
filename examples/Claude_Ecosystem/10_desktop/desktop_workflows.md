# Claude Desktop — Workflow Examples

## Parallel Sessions with Worktrees

Claude Desktop can run multiple agents in parallel, each in its own git worktree:

```
Session 1: "Implement the new auth module"
  → Works in worktree: /tmp/claude-worktree-auth-abc123/
  → Creates branch: claude/auth-module

Session 2: "Fix the flaky test in test_api.py"
  → Works in worktree: /tmp/claude-worktree-fix-def456/
  → Creates branch: claude/fix-flaky-test

Both sessions run simultaneously without conflicts.
```

## App Preview

For web projects, Claude Desktop can build and preview your app:

```
User: "Add a dark mode toggle to the settings page"

Claude Desktop:
  1. Reads current settings page
  2. Adds dark mode toggle component
  3. Builds the app (npm run dev)
  4. Opens preview pane showing the result
  5. You see the change live before approving
```

## GitHub Integration

Monitor and fix CI directly from Desktop:

```
User: "The CI is failing on PR #42, can you fix it?"

Claude Desktop:
  1. Fetches PR #42 details via GitHub API
  2. Reads the failing CI logs
  3. Identifies the issue (missing env variable)
  4. Pushes a fix commit
  5. Monitors CI re-run
```

## Settings Location

- macOS: `~/Library/Application Support/Claude/`
- Windows: `%APPDATA%\Claude\`
- Config file: `claude_desktop_config.json`
