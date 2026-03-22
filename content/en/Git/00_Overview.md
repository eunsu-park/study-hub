# Git & GitHub Learning Guide

## Introduction

This folder contains materials for learning the Git version control system and the GitHub collaboration platform. You can progress step by step from basic commands to CI/CD automation.

**Target Audience**: Developer beginners, those who want to learn version control

---

## Learning Roadmap

```
[Git Basics]          [GitHub]              [Advanced]
    │                    │                    │
    ▼                    ▼                    ▼
Git Basics ────────▶ Getting Started ────▶ Advanced Git
    │                  with GitHub            │
    ▼                    │                    ▼
Basic Commands ─────▶ GitHub          ────▶ GitHub Actions
    │                Collaboration
    ▼
Branches
```

---

## Prerequisites

- Basic terminal/command-line usage
- Text editor usage

---

## File List

| Filename | Difficulty | Key Topics |
|--------|--------|----------|
| [01_Git_Basics.md](./01_Git_Basics.md) | ⭐ | Git concepts, installation, initial setup |
| [02_Basic_Commands.md](./02_Basic_Commands.md) | ⭐ | status, add, commit, log |
| [03_Branches.md](./03_Branches.md) | ⭐⭐ | Branch creation, merging, conflict resolution |
| [04_GitHub_Getting_Started.md](./04_GitHub_Getting_Started.md) | ⭐ | Remote repositories, push, pull, clone |
| [05_GitHub_Collaboration.md](./05_GitHub_Collaboration.md) | ⭐⭐ | Pull Requests, code reviews, Fork |
| [06_Git_Advanced.md](./06_Git_Advanced.md) | ⭐⭐⭐ | rebase, cherry-pick, stash, reset |
| [07_GitHub_Actions.md](./07_GitHub_Actions.md) | ⭐⭐⭐ | CI/CD, workflow automation |
| [08_Git_Workflow_Strategies.md](./08_Git_Workflow_Strategies.md) | ⭐⭐⭐ | Git Flow, GitHub Flow, trunk-based |
| [09_Advanced_Git_Techniques.md](./09_Advanced_Git_Techniques.md) | ⭐⭐⭐⭐ | hooks, submodules, worktrees |
| [10_Monorepo_Management.md](./10_Monorepo_Management.md) | ⭐⭐⭐⭐ | Nx, Turborepo, dependency management |
| [11_Git_Internals.md](./11_Git_Internals.md) | ⭐⭐⭐⭐ | Git object model, packfiles, refs |
| [12_Git_Bisect_and_Debugging.md](./12_Git_Bisect_and_Debugging.md) | ⭐⭐⭐ | bisect, blame, debugging workflows |
| [13_Monorepo_Workflows.md](./13_Monorepo_Workflows.md) | ⭐⭐⭐⭐ | Monorepo CI/CD, change detection, scaling |
| [14_Git_Hooks_Advanced.md](./14_Git_Hooks_Advanced.md) | ⭐⭐⭐⭐ | Advanced hooks, automation, enforcement |

---

## Recommended Learning Path

### Stage 1: Git Basics (Local)
1. Git Basics → Basic Commands → Branches

### Stage 2: GitHub Collaboration
2. Getting Started with GitHub → GitHub Collaboration

### Stage 3: Advanced Usage
3. Advanced Git → GitHub Actions → Git Workflow Strategies

### Stage 4: Expert
4. Advanced Git Techniques → Monorepo Management

---

## Quick Start

### Install Git

```bash
# macOS
brew install git

# Ubuntu
sudo apt-get install git

# Verify installation
git --version
```

### Initial Setup

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### First Commit

```bash
git init
git add .
git commit -m "Initial commit"
```

---

## Related Materials

- [Docker Learning](../Docker/00_Overview.md) - Containerizing development environments
- [GitHub Actions](./07_GitHub_Actions.md) - CI/CD automation
