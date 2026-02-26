# Git Basics

**Next**: [Basic Commands](./02_Basic_Commands.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a distributed version control system (DVCS) is and why it matters
2. Distinguish between Git (the tool) and GitHub (the hosting service)
3. Install and configure Git on macOS, Linux, or Windows
4. Initialize a new Git repository with `git init`
5. Describe the three-area architecture: working directory, staging area, and repository
6. Perform the basic workflow: `add`, `commit`, and view history with `git log`

---

Whether you are working alone on a personal project or collaborating with a large team, losing track of changes is one of the most common and costly mistakes in software development. Git gives you a complete, replayable history of every modification, the ability to work on multiple ideas in parallel, and a safety net that lets you undo almost anything. Learning Git is the single most impactful productivity skill a developer can acquire.

> **Analogy -- A Time Machine for Code**: Imagine writing an essay and wishing you could go back to the version from yesterday, or see exactly what changed between Tuesday and Thursday. Git is that time machine. Every `git commit` creates a snapshot -- a save point you can return to at any time. Unlike "undo" in a text editor (which only goes back linearly), Git lets you branch into parallel timelines, compare any two snapshots, and even merge alternate histories together.

## 1. What is Git?

Git is a **Distributed Version Control System (DVCS)**. It tracks changes to files and enables multiple people to collaborate.

### Why Use Git?

- **Version Control**: Save all change history of files
- **Backup**: Store code safely
- **Collaboration**: Multiple people can work simultaneously
- **Experimentation**: Test new features safely

### Git vs GitHub

| Git | GitHub |
|-----|--------|
| Version control **tool** | Git repository **hosting service** |
| Works locally | Online platform |
| Used via command line | Provides web interface |

---

## 2. Installing Git

### macOS

```bash
# Install with Homebrew
brew install git

# Or install via Xcode Command Line Tools
xcode-select --install
```

### Windows

Download and install from [Git official website](https://git-scm.com/download/win)

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install git
```

### Verify Installation

```bash
git --version
# Output example: git version 2.43.0
```

---

## 3. Initial Git Setup

You need to configure your user information when using Git for the first time.

### Set Username and Email

```bash
# Set username
git config --global user.name "John Doe"

# Set email
git config --global user.email "john@example.com"
```

### Verify Configuration

```bash
# View all settings
git config --list

# Check specific settings
git config user.name
git config user.email
```

### Set Default Editor (Optional)

```bash
# Set VS Code as default editor
git config --global core.editor "code --wait"

# Use Vim
git config --global core.editor "vim"
```

---

## 4. Creating a Git Repository

### Method 1: Initialize a New Repository

```bash
# Create project folder
mkdir my-project
cd my-project

# Initialize Git repository
git init
```

Output:
```
Initialized empty Git repository in /path/to/my-project/.git/
```

### Method 2: Clone an Existing Repository

```bash
# Clone repository from GitHub
git clone https://github.com/username/repository.git
```

---

## 5. Git's Three Areas

Git manages files in three areas:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Working        │    │  Staging        │    │  Repository     │
│  Directory      │───▶│  Area           │───▶│  (.git)         │
│  (Work space)   │    │  (Staging)      │    │  (Repository)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
      ↑                      ↑                      ↑
   Edit files            git add               git commit
```

1. **Working Directory**: The space where you actually modify files
2. **Staging Area**: The space where files to be committed are gathered
3. **Repository**: The space where committed snapshots are stored

---

## Practice Examples

### Example 1: Create Your First Repository

```bash
# 1. Create and navigate to practice folder
mkdir git-practice
cd git-practice

# 2. Initialize Git repository
git init

# 3. Create file
echo "# My First Git Project" > README.md

# 4. Check status
git status
```

Expected output:
```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	README.md

nothing added to commit but untracked files present (use "git add" to track)
```

### Example 2: Check Configuration

```bash
# View current Git configuration
git config --list --show-origin
```

---

## Key Summary

| Concept | Description |
|------|------|
| `git init` | Initialize new Git repository |
| `git clone` | Clone remote repository |
| `git config` | Modify Git configuration |
| Working Directory | Space for modifying files |
| Staging Area | Space for commit queue |
| Repository | Space for storing change history |

---

## Next Steps

Let's learn basic commands like `add`, `commit`, `status`, `log` in [02_Basic_Commands.md](./02_Basic_Commands.md)!
