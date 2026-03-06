# Claude Desktop Application

**Previous**: [09. IDE Integration](./09_IDE_Integration.md) | **Next**: [11. Cowork: AI Digital Colleague](./11_Cowork.md)

---

Claude Desktop is a standalone application for macOS and Windows that provides a dedicated environment for working with Claude. Unlike the web interface at claude.ai, the Desktop app integrates deeply with your local development environment — offering features like parallel coding sessions with git worktree isolation, an App Preview for running and viewing web applications, and GitHub integration for monitoring pull requests and automatically fixing CI failures. This lesson covers the Desktop app's features, workflows, and how it complements the CLI and IDE experiences.

**Difficulty**: ⭐

**Prerequisites**:
- Lesson 01: Introduction to Claude (understanding the Claude product ecosystem)
- Lesson 02: Claude Code Getting Started (basic familiarity with Claude Code)
- Git basics (branches, commits, worktrees)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand what Claude Desktop is and how it differs from claude.ai and Claude Code CLI
2. Install and configure Claude Desktop on macOS or Windows
3. Use parallel sessions with git worktree isolation for concurrent work
4. Use App Preview to run and view web applications within Claude
5. Leverage GitHub integration for PR monitoring and CI fix workflows
6. Configure Desktop-specific settings
7. Understand how Claude Desktop integrates with Claude Code CLI

---

## Table of Contents

1. [What Is Claude Desktop?](#1-what-is-claude-desktop)
2. [Desktop vs. Web vs. CLI](#2-desktop-vs-web-vs-cli)
3. [Installation and Setup](#3-installation-and-setup)
4. [Parallel Sessions and Git Worktrees](#4-parallel-sessions-and-git-worktrees)
5. [Visual Diff Review](#5-visual-diff-review)
6. [App Preview](#6-app-preview)
7. [GitHub Integration](#7-github-integration)
8. [Desktop Settings and Configuration](#8-desktop-settings-and-configuration)
9. [Integration with Claude Code CLI](#9-integration-with-claude-code-cli)
10. [Session Persistence](#10-session-persistence)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. What Is Claude Desktop?

Claude Desktop is Anthropic's standalone desktop application that brings Claude's AI capabilities to your local machine as a native app. It combines the conversational interface of claude.ai with the local tooling capabilities of Claude Code, wrapped in a purpose-built application designed for development workflows.

```
┌────────────────────────────────────────────────────────────────┐
│                     Claude Desktop                             │
│                                                                │
│  ┌──────┐  ┌──────────────────────────────────────────────┐   │
│  │      │  │                                              │   │
│  │ Side │  │              Main Workspace                  │   │
│  │ bar  │  │                                              │   │
│  │      │  │  Conversation + Code + App Preview           │   │
│  │ ───  │  │                                              │   │
│  │ Sess │  │  Claude: I've updated the login page.        │   │
│  │ ion  │  │  Here's what it looks like:                  │   │
│  │ List │  │                                              │   │
│  │      │  │  ┌──────────────────────────────────┐        │   │
│  │ ───  │  │  │     App Preview                  │        │   │
│  │ PR   │  │  │     (Live web app rendering)     │        │   │
│  │ Mon  │  │  │                                  │        │   │
│  │ itor │  │  │     [Login Page Preview]         │        │   │
│  │      │  │  │                                  │        │   │
│  │      │  │  └──────────────────────────────────┘        │   │
│  │      │  │                                              │   │
│  └──────┘  └──────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

Key characteristics:
- **Native application**: Runs as a macOS or Windows app (not in a browser)
- **Local file access**: Reads and modifies files on your local machine
- **Tool execution**: Runs terminal commands, builds, and tests locally
- **Persistent sessions**: Conversations survive app restarts
- **Git-aware**: Understands your repository structure and can create worktrees

---

## 2. Desktop vs. Web vs. CLI

Claude is available through multiple interfaces. Here is how they compare:

| Feature | claude.ai (Web) | Claude Desktop | Claude Code (CLI) |
|---------|----------------|----------------|-------------------|
| **Platform** | Browser | macOS, Windows | Terminal (any OS) |
| **Local file access** | No (upload only) | Yes | Yes |
| **Terminal commands** | No | Yes | Yes |
| **Git integration** | No | Yes (worktrees, PRs) | Yes (basic git) |
| **App Preview** | No | Yes | No |
| **Parallel sessions** | Tabs (no isolation) | Git worktree isolation | Multiple terminals |
| **PR monitoring** | No | Yes (GitHub) | Via `gh` CLI |
| **MCP support** | Limited | Yes | Yes |
| **Offline** | No | No | No |
| **Best for** | General Q&A, writing | Development, prototyping | Power users, automation |

### When to Use Each

- **claude.ai**: General questions, writing tasks, quick prototypes with artifacts, no local code needed
- **Claude Desktop**: Full development sessions, visual feedback on web apps, PR management, parallel feature work
- **Claude Code CLI**: Server-side work, CI/CD integration, scripting, SSH sessions, automation

---

## 3. Installation and Setup

### macOS Installation

1. Download from [claude.ai/download](https://claude.ai/download) or the Mac App Store
2. Open the downloaded `.dmg` file
3. Drag Claude to the Applications folder
4. Launch Claude from Applications or Spotlight
5. Sign in with your Anthropic account

```bash
# Verify installation
ls /Applications/Claude.app
# or
open -a Claude
```

### Windows Installation

1. Download from [claude.ai/download](https://claude.ai/download)
2. Run the installer (`.exe`)
3. Follow the installation wizard
4. Launch Claude from the Start menu or search

### First-Time Setup

On first launch:
1. **Sign in**: Use your Anthropic account (same as claude.ai)
2. **Grant permissions**: Allow local file system access when prompted
3. **Select project folder**: Choose the directory you want to work in
4. **Configure tools**: Enable/disable local tool access (terminal, file editing)

```
┌──────────────────────────────────────────────┐
│  Welcome to Claude Desktop                   │
│                                              │
│  ☑ Allow file system access                  │
│  ☑ Allow terminal command execution          │
│  ☑ Allow network access                      │
│  ☐ Allow unrestricted tool use               │
│                                              │
│  Project folder: [~/projects/myapp]  [Browse]│
│                                              │
│  [Get Started]                               │
└──────────────────────────────────────────────┘
```

---

## 4. Parallel Sessions and Git Worktrees

One of Claude Desktop's standout features is the ability to run **multiple coding sessions in parallel**, each isolated in its own **git worktree**. This means Claude can work on two features simultaneously without code conflicts.

### What Are Git Worktrees?

Git worktrees allow you to have multiple working directories tied to the same repository. Each worktree checks out a different branch, and changes in one worktree do not affect another.

```bash
# Standard git workflow: one working directory
my-project/     # Only one branch checked out at a time

# With worktrees: multiple working directories
my-project/           # Main worktree (e.g., main branch)
my-project-feature-a/ # Worktree for feature-a branch
my-project-feature-b/ # Worktree for feature-b branch
```

### How Claude Desktop Uses Worktrees

When you start a new parallel session, Claude Desktop:

1. Creates a new git branch for the task
2. Creates a new worktree directory for that branch
3. Runs the session entirely within that worktree
4. When done, the worktree can be merged or deleted

```
┌──────────────────────────────────────────────────────────────┐
│  Claude Desktop - Parallel Sessions                          │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Session 1       │  │  Session 2       │                 │
│  │  Branch: feat-a  │  │  Branch: feat-b  │                 │
│  │  Worktree:       │  │  Worktree:       │                 │
│  │  /tmp/myapp-a/   │  │  /tmp/myapp-b/   │                 │
│  │                  │  │                  │                 │
│  │  "Add user       │  │  "Fix payment    │                 │
│  │   profiles page" │  │   validation"    │                 │
│  │                  │  │                  │                 │
│  │  Status: Working │  │  Status: Testing │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                              │
│  Both sessions work independently — no conflicts             │
└──────────────────────────────────────────────────────────────┘
```

### Starting a Parallel Session

```
1. Click "New Session" in the sidebar
2. Select "Parallel Session (new worktree)"
3. Name the branch (e.g., "feature/user-profiles")
4. Describe the task
5. Claude creates the worktree and begins working
```

### Benefits of Worktree Isolation

- **No merge conflicts during development**: Each session works on separate files in separate directories
- **Independent testing**: Tests run in one worktree do not affect another
- **Clean rollback**: If a session goes wrong, delete the worktree — the main branch is untouched
- **True parallelism**: Claude can implement two features simultaneously

### Merging Results

After a parallel session completes:

```
┌──────────────────────────────────────────────┐
│  Session "feat-user-profiles" Complete        │
│                                              │
│  Changes:                                    │
│  + src/pages/UserProfile.tsx (new)           │
│  + src/api/users.ts (modified)               │
│  + tests/UserProfile.test.tsx (new)          │
│                                              │
│  [Create PR]  [Merge to main]  [Discard]     │
└──────────────────────────────────────────────┘
```

---

## 5. Visual Diff Review

Claude Desktop provides a rich visual diff interface for reviewing code changes, similar to the IDE integration but as a standalone experience.

### Side-by-Side Diff View

```
┌───────────────────────────┬───────────────────────────┐
│  Before                   │  After                    │
│───────────────────────────│───────────────────────────│
│  10: class UserService {  │  10: class UserService {  │
│  11:   db = new DB();     │  11:   constructor(       │
│  12:                      │  12:     private db: DB,  │
│  13:   getUser(id) {      │  13:   ) {}               │
│  14:     return this.db   │  14:                      │
│  15:       .query(id);    │  15:   getUser(id: string)│
│  16:   }                  │  16:     return this.db   │
│  17: }                    │  17:       .query(id);    │
│                           │  18:   }                  │
│                           │  19: }                    │
└───────────────────────────┴───────────────────────────┘
│  [Accept] [Reject] [Edit] │ File 1 of 3 [◀ ▶]       │
└───────────────────────────────────────────────────────┘
```

### Unified Diff View

For those who prefer a unified view:

```
  10   class UserService {
- 11     db = new DB();
- 12
- 13     getUser(id) {
+ 11     constructor(
+ 12       private db: DB,
+ 13     ) {}
+ 14
+ 15     getUser(id: string) {
  16       return this.db
  17         .query(id);
  18     }
  19   }
```

### Change Summary

Claude Desktop provides a summary of all changes across files:

```
┌──────────────────────────────────────────────────────────┐
│  Change Summary                                          │
│                                                          │
│  3 files changed, 24 insertions(+), 12 deletions(-)     │
│                                                          │
│  📄 src/services/user.ts      +8  -4   [View Diff]     │
│  📄 src/app.ts                +12 -6   [View Diff]     │
│  📄 tests/user.test.ts        +4  -2   [View Diff]     │
│                                                          │
│  [Accept All] [Review Each] [Reject All]                 │
└──────────────────────────────────────────────────────────┘
```

---

## 6. App Preview

App Preview is a feature unique to Claude Desktop that allows you to **run web applications directly within the Claude interface** and see live results.

### How App Preview Works

When Claude builds or modifies a web application, it can start a development server and render the result directly in the Desktop app:

```
┌──────────────────────────────────────────────────────────────┐
│  Claude: I've updated the dashboard. Here's the preview:     │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  App Preview - http://localhost:3000/dashboard           ││
│  │  ────────────────────────────────────────────────────────││
│  │                                                          ││
│  │  ┌─────────────────────────────────────────────────┐    ││
│  │  │  Dashboard                          [Settings]  │    ││
│  │  │                                                 │    ││
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  │    ││
│  │  │  │ Users     │  │ Revenue   │  │ Orders    │  │    ││
│  │  │  │ 1,234     │  │ $45,678   │  │ 567       │  │    ││
│  │  │  └───────────┘  └───────────┘  └───────────┘  │    ││
│  │  │                                                 │    ││
│  │  │  [Chart showing weekly trends]                  │    ││
│  │  │                                                 │    ││
│  │  └─────────────────────────────────────────────────┘    ││
│  │                                                          ││
│  │  Console: No errors                         [Refresh]   ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  The layout looks good. Should I adjust the chart colors?    │
└──────────────────────────────────────────────────────────────┘
```

### Starting App Preview

App Preview activates automatically when Claude:
1. Starts a development server (`npm run dev`, `flask run`, etc.)
2. Generates an HTML file
3. Creates a web application with a viewable output

You can also manually request a preview:

```
You: Start the dev server and show me the login page
```

### Console Log Monitoring

The App Preview includes a console panel that shows:
- JavaScript console output (`console.log`, `console.error`)
- Network request status
- Runtime errors and warnings

```
┌──────────────────────────────────────────────────────┐
│  Console Output                                      │
│                                                      │
│  [LOG]  App started on port 3000                     │
│  [LOG]  Connected to database                        │
│  [WARN] Deprecation: findDOMNode is deprecated       │
│  [ERR]  TypeError: Cannot read property 'map' of     │
│         undefined at Dashboard.tsx:45                 │
│                                                      │
│  [Clear]  [Filter: All ▼]  [Auto-scroll ☑]          │
└──────────────────────────────────────────────────────┘
```

### Real-Time Error Detection and Auto-Fixing

When the console shows errors, Claude Desktop can detect and offer to fix them:

```
Claude: I see a TypeError in Dashboard.tsx at line 45 — the
        'orders' array is undefined on initial render. This is
        because the API call hasn't completed yet. I'll add a
        loading state and null check.

        [Auto-Fix] [Show Error Details] [Ignore]
```

Clicking **Auto-Fix** triggers Claude to:
1. Read the error details and stack trace
2. Navigate to the relevant file
3. Apply a fix (e.g., add null check, loading state)
4. Refresh the preview to verify the fix

### Supported Frameworks

App Preview works with any framework that serves on localhost:

| Framework | Command | Auto-detected |
|-----------|---------|--------------|
| React (Vite) | `npm run dev` | Yes |
| Next.js | `npm run dev` | Yes |
| Vue | `npm run dev` | Yes |
| Svelte | `npm run dev` | Yes |
| Flask | `flask run` | Yes |
| Express | `node server.js` | Yes |
| Django | `python manage.py runserver` | Yes |
| Static HTML | Direct file rendering | Yes |

---

## 7. GitHub Integration

Claude Desktop integrates with GitHub to provide PR monitoring, CI status tracking, and automated fixes for CI failures.

### PR Monitoring

The sidebar shows active pull requests for the current repository:

```
┌──────────────────────────────────────┐
│  Pull Requests                       │
│                                      │
│  ▶ #142 Add user profiles     [Open]│
│    Branch: feat/user-profiles        │
│    CI: ✓ Passing                     │
│    Reviews: 1/2 approved             │
│                                      │
│  ▶ #141 Fix payment validation [Open]│
│    Branch: fix/payment-val           │
│    CI: ✗ Failing (2 checks)         │
│    Reviews: 0/2 approved             │
│                                      │
│  ▶ #140 Update dependencies   [Open]│
│    Branch: chore/deps                │
│    CI: ⏳ Running                    │
│    Reviews: Not requested            │
│                                      │
└──────────────────────────────────────┘
```

### CI Check Status

Click on a PR to see detailed CI check status:

```
┌──────────────────────────────────────────────────────────┐
│  PR #141: Fix payment validation                         │
│                                                          │
│  CI Checks:                                              │
│  ✓ lint          Passed (12s)                            │
│  ✗ test          Failed (45s)  [View Logs]               │
│  ✗ build         Failed (23s)  [View Logs]               │
│  ⏭ deploy        Skipped (depends on build)              │
│                                                          │
│  test failure:                                           │
│  FAIL src/payments/__tests__/validate.test.ts            │
│    ● should reject negative amounts                      │
│      Expected: ValidationError                           │
│      Received: undefined                                 │
│                                                          │
│  [Auto-Fix CI]  [View PR on GitHub]                      │
└──────────────────────────────────────────────────────────┘
```

### Auto-Fix for CI Failures

When a CI check fails, Claude Desktop can automatically fix the issue:

```
1. Click [Auto-Fix CI]
2. Claude reads the CI logs and error messages
3. Claude identifies the failing code
4. Claude creates a fix commit
5. Claude pushes to the PR branch
6. CI re-runs automatically

Workflow:
  CI fails → Claude reads logs → Identifies fix → Commits → Pushes → CI passes
```

Example auto-fix flow:

```
Claude: CI check 'test' failed. I found the issue:

        In src/payments/validate.ts:23, the validateAmount()
        function doesn't throw for negative values. The test
        expects a ValidationError for negative amounts.

        Fix: Add a check for amount < 0 at the start of
        validateAmount().

        [Apply Fix and Push]  [Show Diff First]
```

### Code Review with Inline Comments

Claude Desktop can participate in code reviews by adding inline comments to PRs:

```
┌──────────────────────────────────────────────────────────────┐
│  PR #142 Review - src/pages/UserProfile.tsx                  │
│                                                              │
│  45 │ const [user, setUser] = useState(null);                │
│  46 │                                                        │
│  47 │ useEffect(() => {                                      │
│  48 │   fetch(`/api/users/${userId}`)                        │
│     │   🤖 Claude: Missing error handling for the fetch      │
│     │      call. If the API returns a 404 or 500, the        │
│     │      component will show a blank page. Add a           │
│     │      try/catch and set an error state.                  │
│  49 │     .then(r => r.json())                               │
│  50 │     .then(setUser);                                    │
│  51 │ }, [userId]);                                          │
│     │   🤖 Claude: userId is not in the dependency array     │
│     │      type — if it's a string, this could cause         │
│     │      re-renders on reference changes. Consider         │
│     │      memoizing or using a stable reference.            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Desktop Settings and Configuration

### Accessing Settings

```
macOS:  Claude → Settings (or Cmd+,)
Windows: File → Settings (or Ctrl+,)
```

### General Settings

```json
{
  "appearance": {
    "theme": "system",            // "light", "dark", "system"
    "fontSize": 14,
    "fontFamily": "SF Mono",
    "sidebarPosition": "left"
  },
  "sessions": {
    "autoSave": true,             // Persist sessions across restarts
    "maxParallelSessions": 4,
    "defaultWorktreeLocation": "/tmp/claude-worktrees"
  },
  "tools": {
    "allowFileAccess": true,
    "allowTerminal": true,
    "allowNetwork": true,
    "confirmBeforeExecution": true   // Prompt before running commands
  },
  "github": {
    "enabled": true,
    "autoMonitorPRs": true,
    "autoFixCI": false,           // Require manual approval for CI fixes
    "reviewComments": true
  }
}
```

### Project-Specific Settings

Claude Desktop respects the same project-level settings as Claude Code CLI:

- **CLAUDE.md**: Project instructions loaded automatically
- **.claude/settings.json**: Permission rules and tool configurations
- **.claude/agents/**: Custom agent definitions

### Model Selection

```
┌──────────────────────────────────────────┐
│  Model Selection                         │
│                                          │
│  ○ Claude Opus 4.6   (Most capable)      │
│  ● Claude Sonnet 4.6 (Balanced)          │
│  ○ Claude Haiku 4.5  (Fastest)           │
│                                          │
│  Extended thinking: [Off ▼]              │
│  ├── Off                                 │
│  ├── Low (4K budget)                     │
│  ├── Medium (16K budget)                 │
│  └── High (32K budget)                   │
│                                          │
└──────────────────────────────────────────┘
```

---

## 9. Integration with Claude Code CLI

Claude Desktop and Claude Code CLI share the same underlying engine. They complement each other rather than competing.

### Shared Configuration

Both tools read from the same configuration sources:

```
~/.claude/                    # Shared between Desktop and CLI
├── settings.json             # Global settings
├── credentials               # Authentication
└── projects/                 # Project-specific settings

project/.claude/              # Shared project settings
├── settings.json
├── agents/
└── skills/

project/CLAUDE.md             # Shared project instructions
```

### Using Both Together

A common workflow is to use both tools for different aspects of the same project:

```
Workflow: Feature Development

1. Claude Desktop: Start a parallel session for "feature/auth"
   - Claude works on the auth module with App Preview
   - You see the login page rendering in real time

2. Claude Code CLI (in terminal):
   - Meanwhile, use CLI for server-side configuration
   - Set up database migrations
   - Run performance benchmarks

3. Claude Desktop: Review the auth module changes
   - Visual diff review
   - Create a PR from the Desktop app

4. Claude Code CLI:
   - Use CLI to check CI status: gh pr checks 142
   - Run final integration tests
```

### Avoiding Conflicts

If both Claude Desktop and Claude Code CLI are working in the same directory:

```
Conflict scenario:
  Desktop editing src/app.ts  ←──→  CLI editing src/app.ts
  Result: One overwrites the other's changes

Prevention:
  - Use git worktrees in Desktop (separate directory)
  - Work on different files simultaneously
  - Coordinate: finish one session before starting another in the same directory
```

---

## 10. Session Persistence

Claude Desktop preserves your sessions across application restarts.

### What Is Preserved

- **Conversation history**: All messages and tool outputs
- **Session state**: Which files were read, what changes were made
- **Worktree association**: Which git worktree/branch the session uses
- **App Preview state**: Which server was running (but it needs to be restarted)

### What Is NOT Preserved

- **Running processes**: Development servers, tests, builds stop when the app closes
- **In-flight operations**: A tool call that was executing when the app closed is lost
- **Context window position**: The AI starts with full history but may summarize older parts

### Resuming a Session

```
1. Open Claude Desktop
2. Sidebar shows previous sessions:

   Recent Sessions:
   ├── "Add user profiles" (2 hours ago) - feat/user-profiles
   ├── "Fix payment bug" (yesterday) - fix/payment-val
   └── "Refactor database" (3 days ago) - refactor/db-layer

3. Click a session to resume
4. Claude loads the conversation history and continues

Claude: "Welcome back! Last time we were working on the user
         profiles feature. We completed the ProfilePage component
         and the API endpoints. The remaining work is:
         - Add profile picture upload
         - Write tests
         - Update the navigation

         Should I continue with the profile picture upload?"
```

### Session Management

```
┌──────────────────────────────────────────────┐
│  Session Management                          │
│                                              │
│  Active Sessions: 2 of 4 max                 │
│                                              │
│  📌 feat/user-profiles   [Resume] [Archive]  │
│  📌 fix/payment-val      [Resume] [Archive]  │
│                                              │
│  Archived Sessions:                          │
│  📁 refactor/db-layer    [Restore] [Delete]  │
│  📁 chore/update-deps    [Restore] [Delete]  │
│                                              │
│  [New Session]  [Clean Up Worktrees]         │
└──────────────────────────────────────────────┘
```

### Worktree Cleanup

When sessions are archived or deleted, you can clean up the associated git worktrees:

```bash
# Claude Desktop can do this automatically, or you can do it manually:
git worktree list
# /Users/you/myapp               abcd123 [main]
# /tmp/claude-worktrees/myapp-a  ef45678 [feat/user-profiles]
# /tmp/claude-worktrees/myapp-b  90abcde [fix/payment-val]

# Remove a worktree
git worktree remove /tmp/claude-worktrees/myapp-b

# Prune stale worktree references
git worktree prune
```

---

## 11. Exercises

### Exercise 1: Installation and First Session

1. Download and install Claude Desktop
2. Open a project directory
3. Start a conversation: "Explain the structure of this project"
4. Observe how Claude reads local files and presents the information

### Exercise 2: Parallel Sessions

1. Open a git repository in Claude Desktop
2. Create two parallel sessions:
   - Session 1: "Add a footer component to the website"
   - Session 2: "Fix the header navigation links"
3. Observe that each session works in a separate worktree
4. Verify the changes are independent (no conflicts)
5. Merge both sessions' branches

### Exercise 3: App Preview

1. Open a web project (React, Vue, or plain HTML)
2. Ask Claude to make a visual change (e.g., "Change the background to dark mode")
3. Watch the App Preview update in real time
4. Introduce a deliberate error and observe:
   - The console log showing the error
   - Claude offering to auto-fix it

### Exercise 4: GitHub Integration

1. Create a PR from a Claude Desktop session
2. Monitor the CI checks in the sidebar
3. If CI fails, use the [Auto-Fix CI] feature
4. Review the fix before it is pushed

### Exercise 5: CLI + Desktop Workflow

1. Start a task in Claude Desktop (e.g., "Create a new API endpoint")
2. While Claude works, open a terminal and use Claude Code CLI
3. Use the CLI for a complementary task (e.g., "Set up the database migration")
4. Verify both tools' changes work together

---

## 12. References

- [Claude Desktop Download](https://claude.ai/download)
- [Claude Desktop Documentation](https://docs.anthropic.com/en/docs/claude-desktop)
- [Git Worktrees Documentation](https://git-scm.com/docs/git-worktree)
- [GitHub CLI (gh)](https://cli.github.com/)
- [Anthropic Blog: Claude Desktop Features](https://www.anthropic.com/news)

---

## Next Steps

In the next lesson, [Cowork: AI Digital Colleague](./11_Cowork.md), we explore Cowork — Anthropic's product for running Claude as an autonomous digital colleague that can handle broader tasks beyond coding, including project management, document processing, and workflow automation through plugins and MCP connectors.
