# Introduction and Philosophy

**Next**: [Modes and Basic Navigation](./02_Modes_and_Basic_Navigation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the history and evolution of vi → Vim → Neovim
2. Describe the modal editing philosophy and why it exists
3. Compare Vim with modern editors and identify Vim's unique strengths
4. Install Vim or Neovim on your operating system
5. Launch and complete `vimtutor` as your first hands-on practice

---

Every developer eventually faces a moment: editing a file on a remote server with no GUI, fixing a config file through SSH, or watching a colleague navigate code at impossible speed. That moment is often your first encounter with Vim. Understanding *why* Vim works the way it does — not just *how* — is the key to unlocking its power.

## Table of Contents

1. [A Brief History](#1-a-brief-history)
2. [The Modal Editing Philosophy](#2-the-modal-editing-philosophy)
3. [Why Learn Vim in 2026?](#3-why-learn-vim-in-2026)
4. [vi vs Vim vs Neovim](#4-vi-vs-vim-vs-neovim)
5. [Installation](#5-installation)
6. [Your First Session: vimtutor](#6-your-first-session-vimtutor)
7. [The Learning Curve](#7-the-learning-curve)
8. [Summary](#8-summary)

---

## 1. A Brief History

### ed (1969)
Ken Thompson wrote `ed`, a line editor for Unix. You couldn't see the file — you typed commands and hoped for the best. The only feedback for an error was `?`.

### ex (1976)
Bill Joy extended `ed` into `ex`, adding more powerful line-editing commands. Many of these commands survive in Vim's `:` command-line mode today (`:s`, `:g`, `:w`, `:q`).

### vi (1976)
Joy then added a **visual** mode to `ex` — a full-screen editor called `vi`. For the first time, you could see the text as you edited it. The name literally means "visual." This is why Vim's command-line mode is also called "Ex mode."

### Vim (1991)
Bram Moolenaar released **Vi IMproved** — a clone of vi with major enhancements:
- Multi-level undo (vi only had single undo)
- Visual mode for selection
- Plugin system and scripting (Vimscript)
- Split windows and tabs
- Syntax highlighting
- Extensive help system (`:help`)

### Neovim (2014)
A community fork of Vim focused on modernization:
- Lua as a first-class scripting language (alongside Vimscript)
- Built-in LSP client (Language Server Protocol)
- Built-in terminal emulator
- Treesitter for syntax parsing
- Asynchronous job control
- Better defaults out of the box

---

## 2. The Modal Editing Philosophy

> **Analogy — The Gear Shift**: Think of Vim's modes like a car's gear shift. Normal mode is *drive* — you're moving through text. Insert mode is *park* — you're stationary, adding text at one spot. Visual mode is *reverse* — you're selecting backwards or forwards. You don't try to drive in park; similarly, you don't try to navigate in Insert mode. Switching modes is as natural as shifting gears.

Most editors have a single mode: you type, and letters appear. Vim is different — it has **multiple modes**, each optimized for a different task.

### Why Modes?

The fundamental insight: **you spend far more time reading, navigating, and modifying code than typing new text from scratch.**

Studies of editing behavior show that programmers spend roughly:
- **70%** of their time reading and navigating
- **20%** modifying existing text
- **10%** inserting new text

A single-mode editor optimizes for that 10%. Vim optimizes for the 90%.

### The Core Modes

| Mode | Purpose | How to Enter | Visual Indicator |
|------|---------|-------------|-----------------|
| **Normal** | Navigate and manipulate text | `Esc` (or `Ctrl-[`) | No indicator / `-- NORMAL --` |
| **Insert** | Type new text | `i`, `a`, `o`, etc. | `-- INSERT --` |
| **Visual** | Select text regions | `v`, `V`, `Ctrl-v` | `-- VISUAL --` |
| **Command-line** | Execute commands | `:`, `/`, `?` | `:` prompt at bottom |

Normal mode is the **home base**. You always return to Normal mode between actions. This is the most important habit to develop.

### The Efficiency of Modes

In a traditional editor, to delete a word you might:
1. Move hand to mouse → double-click the word → press Delete

In Vim (Normal mode):
1. Type `dw` (delete word) — two keystrokes, hands never leave home row

To delete 5 lines in a traditional editor:
1. Click start of first line → Shift+click end of fifth line → Delete

In Vim:
1. Type `5dd` — three keystrokes

---

## 3. Why Learn Vim in 2026?

### Universal Availability
Vim (or at least vi) is installed on virtually every Unix/Linux system. When you SSH into a server, Vim is there.

### Speed
Once fluent, Vim users edit text significantly faster than GUI editor users. The keyboard-only workflow eliminates the constant hand-to-mouse context switch.

### Composability
Vim's commands form a **language**: operators (verbs) combine with motions (nouns). Once you learn the grammar, you can express complex edits concisely. You'll learn this in depth in [Lesson 5](./05_Operators_and_Composability.md).

### Ergonomics
Your hands stay on the home row. No reaching for arrow keys, no mouse. Over hours of editing, this reduces strain.

### Vim Keybindings Everywhere
Even if you don't use Vim directly, Vim keybindings are available in:
- VS Code (Vim extension)
- JetBrains IDEs (IdeaVim)
- Sublime Text (Vintage mode)
- Web browsers (Vimium)
- Terminal shells (bash/zsh vi mode)
- Jupyter notebooks
- Many more

Learning Vim is an investment that transfers across tools.

---

## 4. vi vs Vim vs Neovim

| Feature | vi | Vim | Neovim |
|---------|-----|-----|--------|
| Year | 1976 | 1991 | 2014 |
| Multi-level undo | No | Yes | Yes |
| Syntax highlighting | No | Yes | Yes (Treesitter) |
| Plugin system | No | Vimscript | Vimscript + Lua |
| LSP support | No | Via plugins | Built-in |
| Async jobs | No | Vim 8+ | Yes |
| Built-in terminal | No | Vim 8+ | Yes |
| GUI variants | No | gVim | Many (Neovide, etc.) |
| Config file | `.exrc` | `.vimrc` | `init.lua` / `init.vim` |

**Which should you choose?**

- **Learning the basics**: Either Vim or Neovim — the core editing concepts are identical
- **Server administration**: Vim (more widely pre-installed)
- **Modern development environment**: Neovim (LSP, Treesitter, Lua ecosystem)
- **This guide**: Covers both. Lessons 1–13 apply to both; Lesson 14 focuses on Neovim-specific features

---

## 5. Installation

### macOS

```bash
# Vim (comes pre-installed, but often outdated)
vim --version | head -1

# Install latest Vim via Homebrew
brew install vim

# Install Neovim
brew install neovim
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install vim        # Vim
sudo apt install neovim     # Neovim
```

### Fedora/RHEL

```bash
sudo dnf install vim-enhanced   # Vim
sudo dnf install neovim         # Neovim
```

### Windows

```bash
# Via Chocolatey
choco install vim
choco install neovim

# Via Scoop
scoop install vim
scoop install neovim

# Or download from https://www.vim.org/download.php
```

### Verify Installation

```bash
vim --version | head -1
# VIM - Vi IMproved 9.x

nvim --version | head -1
# NVIM v0.10.x
```

---

## 6. Your First Session: vimtutor

The best way to start learning Vim is the built-in tutorial:

```bash
vimtutor
```

This opens a special text file that guides you through basic commands interactively. It takes about 25-30 minutes and covers:

- Moving the cursor (hjkl)
- Entering and exiting Insert mode
- Deleting text
- Undo and redo
- Basic file operations (save, quit)

**Recommendation**: Run `vimtutor` at least 3 times over your first week. Each time, you'll internalize the movements more deeply.

### What to Expect

When you first open Vim (not vimtutor), you'll see something like:

```
~
~
~                    VIM - Vi IMproved
~
~                     version 9.1
~                 by Bram Moolenaar
~
~            Vim is open source and freely distributable
~
~                    type  :q<Enter>       to exit
~                    type  :help<Enter>    for help
~
~
```

The `~` characters indicate lines beyond the end of the file (empty buffer). Don't panic — you now know that `:q` followed by Enter will exit!

---

## 7. The Learning Curve

Vim's learning curve is often exaggerated, but it is real. Here's what to expect:

```
Productivity
    ^
    |        Traditional editors
    |       ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    |      ╱
    |     ╱
    |    ╱         Vim
    |   ╱      ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    |  ╱      ╱
    | ╱   ╱‾‾╱
    |╱  ╱
    |  ╱    ← The dip (1-2 weeks)
    | ╱
    +──────────────────────────────> Time
     Week 1   Week 2   Week 4   Month 2+
```

### Tips for the Learning Curve

1. **Don't go cold turkey** — Use Vim for small tasks first (commit messages, config files)
2. **Learn incrementally** — Master a few commands before adding more
3. **Use cheat sheets** — Keep one next to your monitor (see `examples/VIM/10_vim_cheatsheet.md`)
4. **Resist the mouse** — Force yourself to use keyboard commands
5. **Don't memorize — understand** — Vim's commands are a language, not a list

---

## 8. Summary

| Concept | Key Takeaway |
|---------|-------------|
| History | ed → ex → vi → Vim → Neovim (50+ years of evolution) |
| Modal editing | Different modes for different tasks; Normal mode is home base |
| Why Vim | Universal, fast, composable, ergonomic, transferable skills |
| Installation | `brew install vim/neovim` (macOS), `apt install` (Ubuntu) |
| First step | Run `vimtutor` — the single best starting point |

---

## Exercises

### Exercise 1: Exit Vim

Open Vim by running `vim` in your terminal. You are now in Normal mode staring at a blank buffer. Exit Vim without saving.

<details>
<summary>Show Answer</summary>

Type `:q` and press `Enter`. Because the buffer is empty (no changes were made), Vim exits immediately.

If you accidentally pressed keys and changed something, use `:q!` to force quit without saving.

</details>

### Exercise 2: Identify Vim's Modes

Based on the lesson, match each action with the correct Vim mode:

1. You are typing new code into a function body.
2. You are jumping between lines searching for a bug.
3. You are running `:w` to save the file.
4. You are highlighting three lines to copy them.

<details>
<summary>Show Answer</summary>

1. **Insert mode** — you are inputting new text.
2. **Normal mode** — you are navigating; Normal mode is the home base for movement.
3. **Command-line mode** — `:w` is an Ex command entered at the `:` prompt.
4. **Visual mode** — text selection is done in Visual mode (`V` for linewise).

</details>

### Exercise 3: The Editing Time Distribution

The lesson states that programmers spend roughly 70% / 20% / 10% of their editing time on three activities. Without looking, recall what those three activities are, and explain why this distribution justifies Vim's modal design.

<details>
<summary>Show Answer</summary>

- **70%** reading and navigating code
- **20%** modifying existing text
- **10%** inserting new text from scratch

Vim's design justification: a conventional single-mode editor is optimized only for the 10% (typing). By separating navigation/manipulation into Normal mode, Vim optimizes for the entire 90% (reading + modifying). Modal design is a deliberate trade-off that rewards the most common editing activities.

</details>

### Exercise 4: vi vs Vim vs Neovim Feature Comparison

Without consulting the table in the lesson, answer the following:

1. Which editor first introduced multi-level undo?
2. Which editor has LSP support built-in (no plugin required)?
3. What configuration file does each editor use?

<details>
<summary>Show Answer</summary>

1. **Vim** (1991) — `vi` only had single-level undo.
2. **Neovim** — Vim requires a plugin (e.g., vim-lsp, coc.nvim) for LSP, while Neovim has a built-in LSP client.
3. Configuration files:
   - `vi`: `.exrc`
   - `Vim`: `.vimrc`
   - `Neovim`: `init.lua` or `init.vim`

</details>

### Exercise 5: Run vimtutor and Reflect

Run `vimtutor` in your terminal and complete it from start to finish. After finishing, answer:

1. What keystroke moves the cursor down one line?
2. What command deletes the character under the cursor?
3. What command saves and quits in one step?

<details>
<summary>Show Answer</summary>

1. `j` moves the cursor down (mnemonic: "j" goes down — think of the hanging tail).
2. `x` deletes the character under the cursor (like a typewriter's "cross out").
3. `:wq` writes (saves) the file and then quits. Alternatively, `ZZ` in Normal mode does the same thing.

</details>

---

**Next**: [Modes and Basic Navigation](./02_Modes_and_Basic_Navigation.md) — We'll dive deep into Vim's modes and learn to move around files efficiently.
