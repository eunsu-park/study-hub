# Modes and Basic Navigation

**Previous**: [Introduction and Philosophy](./01_Introduction_and_Philosophy.md) | **Next**: [Essential Editing](./03_Essential_Editing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify and switch between Normal, Insert, Command-line, and Replace modes
2. Navigate using `h`, `j`, `k`, `l` instead of arrow keys
3. Explain why Normal mode is the default and how to return to it
4. Use basic Command-line mode commands (`:w`, `:q`, `:wq`, `:q!`)
5. Recognize the mode indicator at the bottom of the screen

---

The single most important concept in Vim is **modes**. Every frustration a beginner faces — "I can't type!", "Why are letters disappearing!", "How do I exit?!" — comes from not understanding which mode they're in. Master mode awareness, and everything else follows.

## Table of Contents

1. [Understanding Modes](#1-understanding-modes)
2. [Normal Mode — Your Home Base](#2-normal-mode--your-home-base)
3. [Insert Mode — Adding Text](#3-insert-mode--adding-text)
4. [Command-Line Mode](#4-command-line-mode)
5. [Replace Mode](#5-replace-mode)
6. [Basic Navigation with hjkl](#6-basic-navigation-with-hjkl)
7. [Combining Counts with Movement](#7-combining-counts-with-movement)
8. [Essential First-Day Commands](#8-essential-first-day-commands)
9. [Summary](#9-summary)

---

## 1. Understanding Modes

Vim has several modes, but you'll use four primary ones:

```
                    ┌──────────────┐
          i,a,o     │              │    Esc
        ┌──────────▶│  INSERT MODE │──────────┐
        │           │              │           │
        │           └──────────────┘           │
        │                                      ▼
  ┌─────┴────────┐                    ┌──────────────┐
  │              │◀───────────────────│              │
  │ NORMAL MODE  │       Esc          │ VISUAL MODE  │
  │  (default)   │───────────────────▶│              │
  │              │      v, V, Ctrl-V  └──────────────┘
  └─────┬────────┘
        │           ┌──────────────┐           ▲
        │    :,/,?  │  COMMAND-LINE│           │
        └──────────▶│     MODE     │───────────┘
                    │              │   Esc / Enter
                    └──────────────┘
```

**Key rule**: `Esc` always returns you to Normal mode. When in doubt, press `Esc`.

---

## 2. Normal Mode — Your Home Base

Normal mode is where you spend most of your time. In Normal mode, every key is a **command**, not a character to insert.

### Entering Normal Mode

| From | Press | Note |
|------|-------|------|
| Insert mode | `Esc` or `Ctrl-[` | `Ctrl-[` is faster (no reach to Esc key) |
| Visual mode | `Esc` | Returns to Normal |
| Command-line mode | `Esc` | Cancels the command |
| Any mode | `Esc` `Esc` | Double-tap is always safe |

### Why Normal Mode is Default

When you open a file, Vim starts in Normal mode. This surprises beginners who expect to start typing immediately. The reason: **you usually open a file to read or modify it, not to type from scratch**. Normal mode lets you immediately navigate, search, or edit.

### The Mode Indicator

Look at the bottom-left of your Vim window:

```
-- INSERT --        ← You're in Insert mode
-- VISUAL --        ← You're in Visual mode
-- VISUAL LINE --   ← Line-wise Visual mode
-- VISUAL BLOCK --  ← Block Visual mode
:                   ← Command-line mode
(nothing)           ← Normal mode (no indicator)
```

---

## 3. Insert Mode — Adding Text

Insert mode is where Vim behaves like a "normal" editor — keys produce characters.

### Ways to Enter Insert Mode

| Key | Action | Mnemonic |
|-----|--------|----------|
| `i` | Insert before cursor | **i**nsert |
| `a` | Insert after cursor | **a**ppend |
| `I` | Insert at beginning of line | **I**nsert at start |
| `A` | Insert at end of line | **A**ppend to end |
| `o` | Open new line below | **o**pen below |
| `O` | Open new line above | **O**pen above |

Each of these positions the cursor differently before entering Insert mode. For now, just remember `i` (insert at cursor) and `Esc` (return to Normal).

### Exiting Insert Mode

| Key | Action |
|-----|--------|
| `Esc` | Return to Normal mode |
| `Ctrl-[` | Same as Esc (easier to reach) |
| `Ctrl-c` | Return to Normal (skips abbreviation triggers) |

**Pro tip**: Many Vim users remap `Caps Lock` to `Esc` (or `Ctrl`) since it's closer to the home row. You'll learn how in [Lesson 12](./12_Configuration_and_Vimrc.md).

---

## 4. Command-Line Mode

Press `:` in Normal mode to enter Command-line mode. A `:` prompt appears at the bottom of the screen where you type commands.

### Essential Commands

| Command | Action | Mnemonic |
|---------|--------|----------|
| `:w` | Save (write) the file | **w**rite |
| `:q` | Quit Vim | **q**uit |
| `:wq` | Save and quit | **w**rite + **q**uit |
| `:q!` | Quit without saving (force) | **q**uit + force |
| `:w filename` | Save as a new filename | **w**rite to file |
| `:e filename` | Open (edit) a file | **e**dit |
| `:help topic` | Open help for a topic | |

### The "How Do I Exit Vim?" Problem

This is the most common Vim question — it even became a famous Stack Overflow question with over 2.7 million views. Now you know:

```
:q       ← Quit (fails if unsaved changes)
:q!      ← Quit, discarding changes
:wq      ← Save and quit
ZZ       ← Save and quit (Normal mode shortcut)
ZQ       ← Quit without saving (Normal mode shortcut)
```

---

## 5. Replace Mode

A lesser-used but handy mode: Replace mode overwrites existing characters as you type.

| Key | Action |
|-----|--------|
| `r` | Replace single character under cursor (stays in Normal mode) |
| `R` | Enter Replace mode (overwrite until Esc) |

```
Before: Hello World     (cursor on 'W')
r+w:    Hello world     (replaced W with w, back to Normal)

Before: Hello World     (cursor on 'W')
R:      Hello there     (typing 'there' overwrites 'World', press Esc to stop)
```

---

## 6. Basic Navigation with hjkl

In Normal mode, use these keys to move the cursor:

```
          k
          ↑
     h ←     → l
          ↓
          j
```

| Key | Direction | Mnemonic |
|-----|-----------|----------|
| `h` | Left | Leftmost key |
| `j` | Down | "**j**umps down" (j hangs below the baseline) |
| `k` | Up | "**k**icks up" |
| `l` | Right | Rightmost key |

### Why Not Arrow Keys?

Arrow keys work in Vim, but `hjkl` offers advantages:

1. **Home row** — Your fingers never leave typing position
2. **Speed** — No reaching across the keyboard
3. **Composability** — `hjkl` work with counts and operators (e.g., `5j` = move down 5 lines)
4. **Universality** — Many Vim-like tools only support `hjkl`

### Practice Discipline

Force yourself to use `hjkl` from day one. It feels awkward for about a week, then becomes second nature. If you want to enforce this, you can disable arrow keys in your `.vimrc`:

```vim
" Disable arrow keys to force hjkl habit
noremap <Up>    <Nop>
noremap <Down>  <Nop>
noremap <Left>  <Nop>
noremap <Right> <Nop>
```

---

## 7. Combining Counts with Movement

A powerful feature: prefix any movement with a **count** to repeat it.

| Command | Action |
|---------|--------|
| `5j` | Move down 5 lines |
| `10l` | Move right 10 characters |
| `3k` | Move up 3 lines |
| `20h` | Move left 20 characters |

This pattern — **count + command** — appears throughout Vim. It's the beginning of Vim's composable command language.

### Line Number Navigation

With line numbers visible (`:set number`), you can jump directly:

| Command | Action |
|---------|--------|
| `:42` | Go to line 42 |
| `42G` | Go to line 42 (Normal mode) |
| `gg` | Go to first line |
| `G` | Go to last line |

---

## 8. Essential First-Day Commands

Here's the minimal command set to survive your first day with Vim:

### Opening and Closing

```bash
vim filename    # Open a file
vim             # Open Vim with empty buffer
```

### Inside Vim

```
i       → Enter Insert mode (start typing)
Esc     → Return to Normal mode

h/j/k/l → Move cursor (left/down/up/right)

:w      → Save
:q      → Quit
:wq     → Save and quit
:q!     → Quit without saving

u       → Undo
Ctrl-r  → Redo
```

### A Complete Editing Session

```
1. vim hello.txt          ← Open (or create) file
2. i                      ← Enter Insert mode
3. Hello, Vim!            ← Type your text
4. Esc                    ← Return to Normal mode
5. :wq                    ← Save and quit
```

That's it — you can now create and edit files in Vim.

---

## 9. Summary

| Concept | Key Points |
|---------|-----------|
| Normal mode | Default mode; keys are commands, not characters |
| Insert mode | `i` to enter, `Esc` to exit; type text as usual |
| Command-line mode | `:` to enter; `:w` save, `:q` quit, `:wq` both |
| Replace mode | `r` for one char, `R` for continuous overwrite |
| Navigation | `h`(←) `j`(↓) `k`(↑) `l`(→); prefer over arrow keys |
| Counts | `5j` = move down 5 lines; count + command pattern |
| Golden rule | When confused, press `Esc` to return to Normal mode |

---

## Exercises

### Exercise 1: Mode Identification

You open a file with `vim notes.txt`. For each scenario below, identify which key(s) to press:

1. You want to start typing at the current cursor position.
2. You typed some text but want to go back to Normal mode.
3. You want to add a new line below the current line and start typing.
4. You are in Normal mode and want to save the file.

<details>
<summary>Show Answer</summary>

1. Press `i` — enters Insert mode with cursor before the current character.
2. Press `Esc` (or `Ctrl-[`) — returns to Normal mode from any other mode.
3. Press `o` — opens a new blank line below and enters Insert mode.
4. Type `:w` and press `Enter` — the `:` enters Command-line mode, `w` writes the file.

</details>

### Exercise 2: hjkl Navigation

Starting from the current position, write the key sequence to accomplish each movement (use `hjkl` only, no arrow keys):

1. Move down 7 lines.
2. Move left 3 characters.
3. Move to the last line of the file.
4. Move to the first line of the file.

<details>
<summary>Show Answer</summary>

1. `7j` — count (7) followed by direction (j = down).
2. `3h` — count (3) followed by direction (h = left).
3. `G` — capital G jumps to the last line.
4. `gg` — double lowercase g jumps to the first line.

</details>

### Exercise 3: Replace Mode vs. Delete and Insert

You have the line: `The cat sat on the mat`

The cursor is on the `c` in `cat`. You want to change `cat` to `dog`. Describe two different ways to accomplish this:

- Method A: Using replace mode (`r` or `R`)
- Method B: Without using replace mode

<details>
<summary>Show Answer</summary>

**Method A (Replace mode)**:
- Press `R` to enter Replace mode.
- Type `dog` — this overwrites `cat` with `dog` character by character.
- Press `Esc` to return to Normal mode.

**Method B (Delete and insert)**:
- Press `x` three times (or `3x`) to delete `cat`.
- Press `i` to enter Insert mode.
- Type `dog`.
- Press `Esc`.

Method A is more concise when the replacement is the same length as the original. You will learn a more powerful method (`cw` — change word) in Lesson 3.

</details>

### Exercise 4: Save and Quit Variants

Match each scenario to the correct command:

1. You have made changes and want to save them, then exit.
2. You have made changes but realize they were wrong; you want to discard everything and exit.
3. You want to save the file but continue editing.
4. You have made no changes and just want to close the file.

<details>
<summary>Show Answer</summary>

1. `:wq` (or `ZZ` in Normal mode) — write then quit.
2. `:q!` (or `ZQ` in Normal mode) — force quit, discarding changes.
3. `:w` — write (save) without quitting.
4. `:q` — quit; works cleanly when there are no unsaved changes.

</details>

### Exercise 5: The Mode Indicator Challenge

Open Vim (`vim`) and practice switching modes, reading the indicator at the bottom of the screen after each step:

1. Open Vim — what does the bottom-left show?
2. Press `i` — what changes?
3. Press `Esc` — what changes back?
4. Press `v` — what appears now?
5. Press `Esc` then `:` — what appears at the bottom?

<details>
<summary>Show Answer</summary>

1. Nothing (blank) — Normal mode has no indicator by default.
2. `-- INSERT --` appears at the bottom left.
3. The indicator disappears (back to Normal mode).
4. `-- VISUAL --` appears — you are now in character-wise Visual mode.
5. After `Esc` you return to Normal mode, then `:` brings up a `:` prompt at the very bottom — that is Command-line mode.

</details>

---

**Previous**: [Introduction and Philosophy](./01_Introduction_and_Philosophy.md) | **Next**: [Essential Editing](./03_Essential_Editing.md)
