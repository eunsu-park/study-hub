# IDE Integration

**Previous**: [08. Agent Teams](./08_Agent_Teams.md) | **Next**: [10. Claude Desktop Application](./10_Claude_Desktop.md)

---

Claude Code is not limited to the terminal. It integrates deeply into the two most popular IDE families — **VS Code** and **JetBrains** — bringing AI-assisted coding directly into your editor. This lesson covers installation, key features, keyboard shortcuts, and workflow tips for getting the most out of Claude Code inside your IDE.

**Difficulty**: ⭐

**Prerequisites**:
- Lesson 02: Claude Code Getting Started (CLI basics)
- Familiarity with either VS Code or a JetBrains IDE

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and configure the Claude Code extension for VS Code
2. Install and configure the Claude Code plugin for JetBrains IDEs
3. Use the Claude Code panel for conversational coding within the IDE
4. Review inline diffs and accept/reject proposed changes
5. Use @-mentions to provide file context efficiently
6. Navigate Plan mode review within the IDE
7. Apply keyboard shortcuts for common operations
8. Compare terminal-only vs. IDE-integrated workflows and choose the right one

---

## Table of Contents

1. [Overview: Terminal + IDE](#1-overview-terminal--ide)
2. [VS Code Extension](#2-vs-code-extension)
3. [JetBrains Plugin](#3-jetbrains-plugin)
4. [Inline Diff Review](#4-inline-diff-review)
5. [@-Mentions for File Context](#5--mentions-for-file-context)
6. [Plan Mode in the IDE](#6-plan-mode-in-the-ide)
7. [Terminal Integration](#7-terminal-integration)
8. [Keyboard Shortcuts Reference](#8-keyboard-shortcuts-reference)
9. [Terminal-Only vs. IDE Workflow](#9-terminal-only-vs-ide-workflow)
10. [Tips for Effective IDE + Claude Code Workflow](#10-tips-for-effective-ide--claude-code-workflow)
11. [Troubleshooting](#11-troubleshooting)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Overview: Terminal + IDE

Claude Code operates at two levels:

1. **CLI (Terminal)**: The core experience. You type natural language in your terminal, and Claude Code reads, edits, and runs code.
2. **IDE Extension/Plugin**: An integrated panel within your editor that wraps the CLI experience with visual enhancements — inline diffs, file-aware context, and editor-native interactions.

```
┌──────────────────────────────────────────────────────────────┐
│                         Your IDE                             │
│                                                              │
│  ┌──────────────────────────────┬──────────────────────────┐ │
│  │                              │    Claude Code Panel     │ │
│  │       Editor Area            │                          │ │
│  │                              │  > Refactor the login    │ │
│  │  - Inline diffs shown here   │    handler to use async  │ │
│  │  - Accept/reject changes     │                          │ │
│  │  - See proposed edits in     │  Claude: I'll refactor   │ │
│  │    context                   │  the login handler...    │ │
│  │                              │                          │ │
│  │                              │  [Plan] [Accept] [Reject]│ │
│  └──────────────────────────────┴──────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Terminal (integrated) — Claude Code CLI also available  │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

The IDE integration is **not** a separate product. It is the same Claude Code engine, presented through your editor's native UI. Your settings, CLAUDE.md, permissions, and hooks all work identically.

---

## 2. VS Code Extension

### 2.1 Installation

**Method 1: VS Code Marketplace**

1. Open VS Code
2. Go to Extensions (`Cmd+Shift+X` on macOS, `Ctrl+Shift+X` on Windows/Linux)
3. Search for "Claude Code"
4. Click **Install** on the official Anthropic extension

**Method 2: Command Line**

```bash
code --install-extension anthropic.claude-code
```

**Method 3: From Claude Code CLI**

If you already have Claude Code installed in the terminal:

```bash
claude install-extension vscode
```

### 2.2 Initial Setup

After installation:

1. The Claude Code icon appears in the Activity Bar (left sidebar)
2. Click the icon to open the Claude Code panel
3. If not already authenticated, you will be prompted to log in
4. The extension detects your existing `~/.claude/` configuration automatically

```
┌─────────────────────────────────┐
│  VS Code Activity Bar          │
│                                 │
│  📁 Explorer                    │
│  🔍 Search                     │
│  🔀 Source Control              │
│  🐛 Run and Debug              │
│  📦 Extensions                  │
│  🤖 Claude Code  ◀── NEW       │
│                                 │
└─────────────────────────────────┘
```

### 2.3 Opening the Claude Code Panel

There are multiple ways to open the Claude Code panel:

| Method | macOS | Windows/Linux |
|--------|-------|---------------|
| Keyboard shortcut | `Cmd+Esc` | `Ctrl+Esc` |
| Activity Bar | Click Claude Code icon | Click Claude Code icon |
| Command Palette | `Cmd+Shift+P` → "Claude Code: Open" | `Ctrl+Shift+P` → "Claude Code: Open" |

The panel opens as a sidebar (default right side) or can be dragged to the bottom panel area.

### 2.4 The Claude Code Panel

The panel provides a conversational interface similar to the CLI:

```
┌─────────────────────────────────────┐
│  Claude Code                    [⚙] │
│─────────────────────────────────────│
│                                     │
│  Session: my-project (active)       │
│  Model: claude-sonnet-4-20250514    │
│  Mode: normal                       │
│                                     │
│  You: Refactor the UserService      │
│  class to use dependency injection  │
│  instead of direct imports.         │
│                                     │
│  Claude: I'll refactor UserService  │
│  to use constructor injection.      │
│  Let me read the current code...    │
│                                     │
│  📄 Reading src/services/user.ts    │
│  ✏️  Editing src/services/user.ts   │
│  📄 Reading src/app.ts              │
│  ✏️  Editing src/app.ts             │
│                                     │
│  [View Changes] [Accept All]        │
│                                     │
│─────────────────────────────────────│
│  > Type your message...        [⏎]  │
└─────────────────────────────────────┘
```

### 2.5 Status Bar Indicators

The VS Code status bar shows Claude Code's current state:

```
┌─────────────────────────────────────────────────────────────┐
│  Status Bar (bottom of VS Code)                             │
│                                                             │
│  🤖 Claude: Ready  |  Mode: Normal  |  Tokens: 12.4K/200K  │
│  └── Agent status    └── Permission    └── Context usage    │
└─────────────────────────────────────────────────────────────┘
```

Status indicators:
- **Ready**: Claude Code is idle, waiting for input
- **Thinking**: Processing your request
- **Editing**: Making changes to files
- **Waiting**: Waiting for your approval (in Plan mode or permission prompt)
- Token counter shows current context window usage

### 2.6 Settings Sync

The VS Code extension shares settings with the CLI:

- **CLAUDE.md**: Same project instructions apply
- **Permissions**: Same allow/deny rules from `.claude/settings.json`
- **Hooks**: Same hook configurations are active
- **Skills**: Same `/skill` commands available
- **Model selection**: Can be changed independently per session

Settings specific to the VS Code extension are in VS Code's settings (`Cmd+,`):

```json
{
  "claude-code.panelPosition": "right",
  "claude-code.fontSize": 14,
  "claude-code.showTokenCount": true,
  "claude-code.autoOpenPanel": false,
  "claude-code.theme": "auto"
}
```

---

## 3. JetBrains Plugin

### 3.1 Installation

The Claude Code plugin is available for all JetBrains IDEs:
- IntelliJ IDEA
- PyCharm
- WebStorm
- GoLand
- PhpStorm
- CLion
- Rider
- RubyMine

**Installation Steps**:

1. Open your JetBrains IDE
2. Go to **Settings/Preferences** → **Plugins**
3. Search for "Claude Code" in the Marketplace tab
4. Click **Install** and restart the IDE

```bash
# Alternative: Install from command line (if JetBrains Toolbox is configured)
# The plugin ID may vary; check JetBrains Marketplace for exact ID
```

### 3.2 Feature Parity with VS Code

The JetBrains plugin provides the same core features as the VS Code extension:

| Feature | VS Code | JetBrains |
|---------|---------|-----------|
| Chat panel | Yes | Yes |
| Inline diffs | Yes | Yes |
| @-mentions | Yes | Yes |
| Plan mode review | Yes | Yes |
| Terminal integration | Yes | Yes |
| Status bar | Yes | Yes |
| Settings sync with CLI | Yes | Yes |

### 3.3 JetBrains-Specific Workflows

JetBrains IDEs offer some unique integration points:

**Inspections Integration**: Claude Code findings can appear alongside JetBrains' built-in code inspections:

```
┌──────────────────────────────────────────────┐
│  Code Inspection Results                      │
│                                              │
│  ⚠️ JetBrains: Unused import 'os'           │
│  ⚠️ JetBrains: Method may be static         │
│  🤖 Claude: Potential SQL injection on L45   │
│  🤖 Claude: Missing null check on L78       │
└──────────────────────────────────────────────┘
```

**Tool Window**: The Claude Code tool window can be docked to any position in the IDE:

```
View → Tool Windows → Claude Code
```

**Action System Integration**: Claude Code actions appear in JetBrains' action system:

```
Ctrl+Shift+A (Find Action) → Type "Claude"

Results:
  Claude Code: Open Panel
  Claude Code: Ask About Selection
  Claude Code: Explain Code
  Claude Code: Generate Tests
  Claude Code: Review File
```

**Right-Click Context Menu**:

```
Right-click on selected code →
  Claude Code →
    Ask About Selection
    Explain This Code
    Refactor Selection
    Generate Tests for Selection
    Find Bugs in Selection
```

---

## 4. Inline Diff Review

One of the most impactful IDE integration features is **inline diff review**. When Claude Code proposes changes to a file, you see them directly in your editor — not just as text in a chat panel.

### How It Works

```
┌──────────────────────────────────────────────────────────────┐
│  src/services/user.ts                                        │
│──────────────────────────────────────────────────────────────│
│  10 │ export class UserService {                             │
│  11 │-  private db = new Database();     // REMOVED (red)    │
│  12 │-  private cache = new Cache();     // REMOVED (red)    │
│  11 │+  constructor(                     // ADDED (green)    │
│  12 │+    private db: Database,          // ADDED (green)    │
│  13 │+    private cache: Cache,          // ADDED (green)    │
│  14 │+  ) {}                             // ADDED (green)    │
│  15 │                                                        │
│  16 │   async getUser(id: string) {                          │
│  17 │     // unchanged code...                               │
│──────────────────────────────────────────────────────────────│
│  [Accept Change] [Reject Change] [Edit Manually]             │
└──────────────────────────────────────────────────────────────┘
```

### Diff Navigation

When Claude Code makes changes to multiple files, you can navigate between them:

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| Next diff | `Cmd+Option+]` | `Ctrl+Alt+]` |
| Previous diff | `Cmd+Option+[` | `Ctrl+Alt+[` |
| Accept current diff | `Cmd+Enter` | `Ctrl+Enter` |
| Reject current diff | `Cmd+Backspace` | `Ctrl+Backspace` |
| Accept all diffs | `Cmd+Shift+Enter` | `Ctrl+Shift+Enter` |
| Reject all diffs | `Cmd+Shift+Backspace` | `Ctrl+Shift+Backspace` |

### Partial Acceptance

You can accept some changes and reject others within the same file:

```
File: src/config.ts

Change 1: Add database pool config     [Accept ✓]
Change 2: Change port from 3000→8080   [Reject ✗]  (I want to keep 3000)
Change 3: Add Redis connection string   [Accept ✓]
```

This granular control is one of the key advantages of the IDE integration over the terminal, where changes are applied atomically.

---

## 5. @-Mentions for File Context

In the IDE integration, you can use **@-mentions** to reference files and code ranges directly in your messages. This provides precise context without needing to describe file locations.

### File Mentions

Type `@` followed by a filename to reference it:

```
You: Refactor @src/auth/login.ts to use the error handling
     pattern from @src/utils/errors.ts
```

Claude Code receives the full contents of both files as context, ensuring it understands the existing code before making changes.

### Line Range Mentions

Reference specific lines within a file:

```
You: The function at @src/api/users.ts:45-80 has a bug.
     It doesn't handle the case where the user ID is null.
```

Claude Code reads only lines 45-80 of the file, focusing its attention on the relevant code.

### Directory Mentions

Reference entire directories:

```
You: Analyze the test coverage in @tests/api/ and identify
     which endpoints are missing test cases.
```

### Symbol Mentions

Some IDE integrations support referencing symbols (functions, classes, variables) directly:

```
You: Explain what @UserService.authenticate does and whether
     it properly handles token expiration.
```

### Autocomplete for @-Mentions

The IDE provides autocomplete when you type `@`:

```
You: Refactor @
              ┌──────────────────────────┐
              │  📄 src/auth/login.ts    │
              │  📄 src/auth/jwt.ts      │
              │  📄 src/auth/register.ts │
              │  📁 src/api/            │
              │  📁 src/models/         │
              └──────────────────────────┘
```

This is faster than typing full file paths and eliminates path errors.

### @-Mention Examples

| Input | What Claude Receives |
|-------|---------------------|
| `@package.json` | Full contents of package.json |
| `@src/app.ts:1-20` | Lines 1-20 of src/app.ts |
| `@tests/` | List of files in tests/ directory |
| `@.env.example` | Contents of .env.example |
| `@tsconfig.json` | Full contents of tsconfig.json |

---

## 6. Plan Mode in the IDE

Plan mode (covered in detail in Lesson 04) has enhanced support in the IDE.

### Activating Plan Mode

```
You: /plan Migrate our Express app to use TypeScript strict mode
```

Or toggle Plan mode from the panel controls:

```
┌────────────────────────────────────────┐
│  Mode: [Normal ▼]                      │
│         ├── Normal                     │
│         ├── Plan                       │
│         └── Auto-accept               │
└────────────────────────────────────────┘
```

### Plan Review Interface

In the IDE, Plan mode presents a structured view of proposed changes:

```
┌──────────────────────────────────────────────────────────────┐
│  Plan: Migrate to TypeScript Strict Mode                     │
│──────────────────────────────────────────────────────────────│
│                                                              │
│  Phase 1: Configuration                                      │
│  ☐ Update tsconfig.json: enable "strict": true               │
│  ☐ Update tsconfig.json: enable "noImplicitAny": true        │
│                                                              │
│  Phase 2: Fix Type Errors (estimated: 47 files)              │
│  ☐ src/models/user.ts: Add types to 3 functions              │
│  ☐ src/models/order.ts: Add types to 5 functions             │
│  ☐ src/api/users.ts: Fix 12 implicit 'any' parameters       │
│  ☐ ... (44 more files)                                       │
│                                                              │
│  Phase 3: Verification                                       │
│  ☐ Run tsc --noEmit to verify zero errors                    │
│  ☐ Run test suite to verify no regressions                   │
│                                                              │
│  [Approve Plan] [Edit Plan] [Cancel]                         │
└──────────────────────────────────────────────────────────────┘
```

You can review the plan, edit it (add/remove steps), and approve it before Claude Code begins execution. During execution, each step is marked as completed.

---

## 7. Terminal Integration

The IDE's integrated terminal works seamlessly with Claude Code.

### Using Claude Code in the IDE Terminal

You can still use the `claude` CLI directly in the IDE's terminal:

```bash
# In the IDE's integrated terminal
claude "Run the test suite and fix any failing tests"
```

This is useful when you prefer the terminal experience for certain tasks while using the IDE panel for others.

### Terminal Command Output

When Claude Code runs terminal commands (via the Bash tool), the output is visible in both:

1. The Claude Code panel (summarized)
2. The IDE's terminal (full output)

```
Claude Code Panel:
  Running: npm test
  Result: 45/47 tests passed, 2 failures
  [Show Full Output]

IDE Terminal:
  $ npm test
  PASS src/auth/__tests__/login.test.ts (2.1s)
  PASS src/api/__tests__/users.test.ts (1.8s)
  FAIL src/api/__tests__/orders.test.ts (3.2s)
    ● OrderAPI › POST /orders › should validate required fields
      Expected: 400
      Received: 500
  ...
```

### Terminal Sharing

When you run commands in the IDE terminal manually, Claude Code can observe the output if you reference it:

```
You: I just ran `npm test` in the terminal and got failures.
     Check the terminal output and fix the failing tests.
```

---

## 8. Keyboard Shortcuts Reference

### VS Code Shortcuts

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| Open Claude Code panel | `Cmd+Esc` | `Ctrl+Esc` |
| Toggle Claude Code panel | `Cmd+Shift+Esc` | `Ctrl+Shift+Esc` |
| Send message | `Enter` | `Enter` |
| New line in input | `Shift+Enter` | `Shift+Enter` |
| Cancel current operation | `Escape` | `Escape` |
| Accept all changes | `Cmd+Shift+Enter` | `Ctrl+Shift+Enter` |
| Reject all changes | `Cmd+Shift+Backspace` | `Ctrl+Shift+Backspace` |
| Next diff | `Cmd+Option+]` | `Ctrl+Alt+]` |
| Previous diff | `Cmd+Option+[` | `Ctrl+Alt+[` |
| Accept current diff | `Cmd+Enter` | `Ctrl+Enter` |
| Reject current diff | `Cmd+Backspace` | `Ctrl+Backspace` |
| Focus Claude Code input | `Cmd+L` | `Ctrl+L` |
| Clear conversation | `Cmd+K` | `Ctrl+K` |
| Ask about selection | `Cmd+Shift+L` | `Ctrl+Shift+L` |

### JetBrains Shortcuts

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| Open Claude Code | `Cmd+Esc` | `Ctrl+Esc` |
| Send message | `Enter` | `Enter` |
| New line | `Shift+Enter` | `Shift+Enter` |
| Cancel operation | `Escape` | `Escape` |
| Accept all changes | `Cmd+Shift+Enter` | `Ctrl+Shift+Enter` |
| Find Claude action | `Cmd+Shift+A` | `Ctrl+Shift+A` |
| Ask about selection | `Cmd+Shift+L` | `Ctrl+Shift+L` |

### Customizing Shortcuts

**VS Code**: Open Keyboard Shortcuts (`Cmd+K Cmd+S`) and search for "Claude Code"

**JetBrains**: Open Settings → Keymap → search for "Claude Code"

```json
// VS Code keybindings.json example
[
  {
    "key": "cmd+shift+c",
    "command": "claude-code.openPanel",
    "when": "editorTextFocus"
  },
  {
    "key": "cmd+shift+e",
    "command": "claude-code.explainSelection",
    "when": "editorHasSelection"
  }
]
```

---

## 9. Terminal-Only vs. IDE Workflow

Both approaches have strengths. Here is a comparison to help you choose:

### Terminal-Only Workflow

**Advantages**:
- No IDE dependency — works over SSH, in containers, on any machine
- Faster startup (no IDE overhead)
- Full screen for Claude Code output
- Better for server-side work, DevOps, and automation
- Scriptable (pipe Claude Code into other tools)
- Works identically on all operating systems

**Disadvantages**:
- No visual diff review (changes described in text)
- Must type full file paths (no @-mention autocomplete)
- Cannot see changes in editor context
- Harder to partially accept/reject changes
- Must switch between terminal and editor manually

```
Terminal workflow:
  Terminal ──▶ Claude Code ──▶ Changes applied ──▶ Open editor to review
```

### IDE-Integrated Workflow

**Advantages**:
- Inline diff review in editor context
- @-mentions with autocomplete for files
- See changes alongside surrounding code
- Granular accept/reject for individual changes
- No context switching between terminal and editor
- Plan mode with structured visual review

**Disadvantages**:
- Requires VS Code or JetBrains IDE
- Additional memory usage from IDE + extension
- Extension updates may lag behind CLI releases
- Some advanced CLI features may not be exposed in the UI
- IDE-specific bugs and quirks

```
IDE workflow:
  Editor ──▶ Claude Code Panel ──▶ Inline diffs ──▶ Accept/reject in place
```

### Recommended Approach

Most developers use **both**, choosing based on the task:

| Task | Recommended |
|------|-------------|
| Quick question about code | IDE (select code → ask) |
| Multi-file refactoring | IDE (inline diff review) |
| Server-side debugging | Terminal (SSH access) |
| CI/CD scripting | Terminal (scriptable) |
| Code review assistance | IDE (visual diffs) |
| Codebase exploration | Terminal or IDE |
| Writing tests | IDE (see test alongside code) |
| Docker/deployment work | Terminal |

---

## 10. Tips for Effective IDE + Claude Code Workflow

### Tip 1: Select Before Asking

Select relevant code in the editor before asking Claude Code a question. The selection is automatically included as context:

```
1. Select lines 45-80 in src/api/users.ts
2. Cmd+Shift+L (Ask about selection)
3. "Why does this function return undefined when the user has no orders?"
```

### Tip 2: Use Split View

Keep Claude Code panel open alongside your editor in split view:

```
┌───────────────────────┬────────────────────┐
│                       │                    │
│   Editor              │   Claude Code      │
│   (your code)         │   (conversation)   │
│                       │                    │
│                       │                    │
└───────────────────────┴────────────────────┘
```

### Tip 3: Review Changes Before Accepting

Always review inline diffs before accepting, even if you trust Claude Code. Look for:
- Unintended side effects in nearby code
- Changes that break the module's interface (public API)
- Hardcoded values that should be configurable
- Missing error handling in the new code

### Tip 4: Use Plan Mode for Large Changes

For changes spanning multiple files, use Plan mode first:

```
You: /plan Add authentication middleware to all API routes
```

Review the plan, then approve. This prevents surprises when Claude Code starts editing.

### Tip 5: Combine Terminal and IDE

For complex workflows, use both:

```
1. IDE: Ask Claude Code to explain the codebase architecture
2. Terminal: Use Claude Code CLI to run migration scripts
3. IDE: Review and refine the generated code
4. Terminal: Run tests and fix failures
```

### Tip 6: Keep the Panel Focused

Avoid long conversations in a single session. If the topic changes significantly, start a new session:

```
Session 1: "Help me debug the login flow" (focused)
Session 2: "Now help me optimize the database queries" (new topic)
```

### Tip 7: Leverage Editor Features Alongside Claude

Use your IDE's built-in features to complement Claude Code:

- **Go to Definition**: Before asking Claude to refactor, use your IDE to understand call sites
- **Find References**: See all usages of a function before Claude modifies it
- **Git Blame**: Understand the history of code before requesting changes
- **Debugger**: Step through code to verify Claude's changes work correctly

---

## 11. Troubleshooting

### Extension Not Loading

```
Symptom: Claude Code icon not appearing in Activity Bar
Solutions:
1. Check VS Code version (requires VS Code 1.85+)
2. Reload VS Code window (Cmd+Shift+P → "Reload Window")
3. Check extension is enabled (Extensions → Claude Code → Enable)
4. Check developer console (Help → Toggle Developer Tools → Console)
```

### Authentication Issues

```
Symptom: "Not authenticated" error in the IDE panel
Solutions:
1. Run 'claude auth' in the terminal first
2. Check ~/.claude/credentials exists
3. Ensure your API key or subscription is valid
4. Try logging out and back in: 'claude auth logout && claude auth login'
```

### Panel Not Responding

```
Symptom: Claude Code panel shows spinner indefinitely
Solutions:
1. Cancel current operation (Escape)
2. Check network connectivity
3. Check Claude Code CLI works in terminal independently
4. Restart the extension (Cmd+Shift+P → "Claude Code: Restart")
```

### Diff Review Not Showing

```
Symptom: Claude says it edited files but no inline diffs appear
Solutions:
1. Check the file is open in the editor
2. Click "View Changes" in the Claude Code panel
3. Check Source Control panel for pending changes
4. Ensure the file is not in .gitignore (diffs use git)
```

### High Memory Usage

```
Symptom: IDE becomes slow when Claude Code is active
Solutions:
1. Close long conversation sessions (start fresh)
2. Reduce claude-code.maxHistoryItems in settings
3. Close unused editor tabs
4. Increase VS Code memory limit in settings
```

---

## 12. Exercises

### Exercise 1: VS Code Setup

1. Install the Claude Code extension in VS Code
2. Open a project with a CLAUDE.md file
3. Use `Cmd+Esc` to open the Claude Code panel
4. Ask it to explain the project structure
5. Verify that CLAUDE.md instructions are being followed

### Exercise 2: Inline Diff Review

1. Ask Claude Code to refactor a function in your project
2. Review the inline diffs in the editor
3. Accept one change and reject another
4. Verify the file reflects your choices

### Exercise 3: @-Mention Workflow

1. Open a project with multiple files
2. In the Claude Code panel, use @-mentions to reference two files
3. Ask Claude to compare the code patterns in both files
4. Use a line range mention (`@file.ts:10-30`) for a specific question

### Exercise 4: Plan Mode Review

1. Switch to Plan mode in the Claude Code panel
2. Ask for a multi-file change (e.g., "Add input validation to all API endpoints")
3. Review the plan that Claude Code proposes
4. Edit the plan to remove a step you disagree with
5. Approve the modified plan and watch execution

### Exercise 5: Terminal + IDE Comparison

Perform the same task using both approaches:

1. **Terminal**: Use `claude` CLI to add a test file for a module
2. **IDE**: Use the Claude Code panel to add a test file for a different module
3. Compare the experience: speed, control, review quality
4. Write down which approach you preferred and why

---

## 13. References

- [Claude Code VS Code Extension](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)
- [Claude Code JetBrains Plugin](https://plugins.jetbrains.com/plugin/claude-code)
- [Claude Code Documentation: IDE Integration](https://docs.anthropic.com/en/docs/claude-code)
- [VS Code Extension API](https://code.visualstudio.com/api)
- [JetBrains Plugin Development](https://plugins.jetbrains.com/docs/intellij/)

---

## Next Steps

In the next lesson, [Claude Desktop Application](./10_Claude_Desktop.md), we explore the standalone Claude Desktop app — a dedicated application for macOS and Windows that offers parallel session management, App Preview for running web apps directly in the interface, and deep GitHub integration for PR monitoring and CI fix workflows.
