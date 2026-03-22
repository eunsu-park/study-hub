# Essential Editing

**Previous**: [Modes and Basic Navigation](./02_Modes_and_Basic_Navigation.md) | **Next**: [Motions and Navigation](./04_Motions_and_Navigation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use multiple ways to enter Insert mode (`i`, `a`, `o`, `I`, `A`, `O`)
2. Delete characters, words, and lines using `x`, `dw`, and `dd`
3. Yank (copy) and put (paste) text with `yy` and `p`
4. Undo and redo changes with `u` and `Ctrl-r`
5. Save, quit, and combine these operations efficiently

---

With modes and basic navigation under your belt, it's time to learn the fundamental editing operations. These commands will cover 80% of your daily editing needs. Don't try to memorize everything at once — focus on `i`, `Esc`, `x`, `dd`, `u`, and `:wq`, then gradually add more commands to your repertoire.

## Table of Contents

1. [Entering Insert Mode — The Full Picture](#1-entering-insert-mode--the-full-picture)
2. [Deleting Text](#2-deleting-text)
3. [Yank (Copy) and Put (Paste)](#3-yank-copy-and-put-paste)
4. [Changing Text](#4-changing-text)
5. [Undo and Redo](#5-undo-and-redo)
6. [Saving and Quitting](#6-saving-and-quitting)
7. [Repeating Commands](#7-repeating-commands)
8. [Joining Lines](#8-joining-lines)
9. [Practical Editing Workflow](#9-practical-editing-workflow)
10. [Summary](#10-summary)

---

## 1. Entering Insert Mode — The Full Picture

Each Insert command positions the cursor differently before switching to Insert mode:

```
Line: The quick brown fox

Cursor on 'q' of "quick":

  i  → Insert BEFORE 'q':    The |quick brown fox
  a  → Insert AFTER 'q':     The q|uick brown fox
  I  → Insert at line START:  |The quick brown fox
  A  → Insert at line END:    The quick brown fox|
  o  → Open line BELOW:       The quick brown fox
                               |
  O  → Open line ABOVE:       |
                               The quick brown fox
```

### When to Use Which

| Command | Use Case |
|---------|----------|
| `i` | Insert at current position (most common) |
| `a` | Append after cursor (e.g., add text after a character) |
| `I` | Add something to the beginning of a line |
| `A` | Add something to the end of a line (very common) |
| `o` | Start a new line below (extremely common) |
| `O` | Start a new line above |

---

## 2. Deleting Text

All delete commands work in Normal mode. Deleted text is saved in a register (Vim's clipboard) so you can paste it later.

### Character-Level Deletion

| Command | Action |
|---------|--------|
| `x` | Delete character under cursor |
| `X` | Delete character before cursor (like Backspace) |

### Word-Level Deletion

| Command | Action |
|---------|--------|
| `dw` | Delete from cursor to start of next word |
| `de` | Delete from cursor to end of current word |
| `db` | Delete from cursor backwards to start of word |

### Line-Level Deletion

| Command | Action |
|---------|--------|
| `dd` | Delete entire current line |
| `D` | Delete from cursor to end of line |
| `d0` | Delete from cursor to beginning of line |

### Deletion with Counts

| Command | Action |
|---------|--------|
| `3x` | Delete 3 characters |
| `5dd` | Delete 5 lines |
| `2dw` | Delete 2 words |

### Example

```
Before: The quick brown fox jumps over the lazy dog
        ^ cursor here

x    →  he quick brown fox jumps over the lazy dog
dw   →  quick brown fox jumps over the lazy dog
dd   →  (entire line deleted)
D    →  The                (from cursor to end deleted)
```

---

## 3. Yank (Copy) and Put (Paste)

Vim uses its own terminology: **yank** means copy, **put** means paste.

### Yanking (Copying)

| Command | Action |
|---------|--------|
| `yy` | Yank (copy) entire current line |
| `Y` | Yank entire current line (same as `yy`) |
| `yw` | Yank from cursor to start of next word |
| `y$` | Yank from cursor to end of line |

### Putting (Pasting)

| Command | Action |
|---------|--------|
| `p` | Put (paste) after cursor / below current line |
| `P` | Put (paste) before cursor / above current line |

### The Relationship Between Delete and Put

Here's something important: **every delete command also copies the text**. This means `dd` followed by `p` is effectively "cut and paste."

```
Line 1: First line
Line 2: Second line    ← cursor here
Line 3: Third line

dd   → Cuts "Second line" (now in register)
       Line 1: First line
       Line 2: Third line    ← cursor here

p    → Pastes below current line:
       Line 1: First line
       Line 2: Third line
       Line 3: Second line   ← pasted here
```

### Duplicating a Line

One of the most common operations: `yy` then `p` duplicates the current line below.

```
yy   → Yank current line
p    → Paste below

Result: The line appears twice.
```

---

## 4. Changing Text

The **change** command (`c`) deletes text and immediately enters Insert mode — it's a delete + insert combo.

| Command | Action |
|---------|--------|
| `cw` | Change word (delete to next word, enter Insert) |
| `ce` | Change to end of word |
| `cc` | Change entire line (delete line, enter Insert) |
| `C` | Change from cursor to end of line |
| `c$` | Same as `C` |
| `s` | Substitute character (delete char, enter Insert) |
| `S` | Substitute line (same as `cc`) |

### Example

```
Before: The quick brown fox
            ^ cursor on 'q'

cw   → The |brown fox           (cursor in Insert mode at |)
       Type "slow" → The slow brown fox

ce   → The | brown fox          (deleted "quick", Insert mode)
       Type "fast" → The fast brown fox

cc   → |                        (entire line cleared, Insert mode)
       Type anything...
```

---

## 5. Undo and Redo

Vim's undo system is more powerful than most editors — it supports **unlimited multi-level undo** and even an **undo tree** (branches of undo history).

| Command | Action |
|---------|--------|
| `u` | Undo last change |
| `U` | Undo all changes on current line (less commonly used) |
| `Ctrl-r` | Redo (reverse an undo) |

### Undo Granularity

Each time you enter Insert mode, type, and return to Normal mode, that entire Insert session counts as **one undo unit**. So if you type a whole paragraph in Insert mode, `u` will undo the entire paragraph.

**Tip**: Make smaller, more frequent trips to Normal mode (press `Esc` periodically while writing) to create finer-grained undo points.

### Repeat Count with Undo

| Command | Action |
|---------|--------|
| `5u` | Undo last 5 changes |
| `5Ctrl-r` | Redo 5 changes |

---

## 6. Saving and Quitting

### Basic Commands

| Command | Action |
|---------|--------|
| `:w` | Write (save) the file |
| `:q` | Quit (fails if unsaved changes exist) |
| `:wq` | Write and quit |
| `:x` | Write (only if changed) and quit |
| `ZZ` | Same as `:x` (Normal mode shortcut) |
| `:q!` | Quit without saving (discard changes) |
| `ZQ` | Same as `:q!` (Normal mode shortcut) |
| `:wa` | Write all open buffers |
| `:qa` | Quit all open buffers |
| `:wqa` | Write and quit all |

### Save As

```vim
:w newfile.txt        " Save current buffer as newfile.txt
:w! existing.txt      " Overwrite existing.txt (force)
```

### Working with Read-Only Files

```vim
:w !sudo tee %        " Save with sudo (when you forgot to open with sudo)
```

This pipes the buffer through `sudo tee` to write to the file. The `%` represents the current filename.

---

## 7. Repeating Commands

### The Dot Command (`.`)

The `.` key repeats the last change command. This is one of Vim's most powerful features.

```
dd    → Delete a line
.     → Delete another line (repeats dd)
.     → Delete another line
```

```
cw    → Change word to "hello" (type hello, press Esc)
w     → Move to next word
.     → Change this word to "hello" too
```

The dot command is so important that experienced Vim users plan their edits around it. You'll explore this concept deeply in [Lesson 5](./05_Operators_and_Composability.md).

### Repeating Command-Line Commands

| Command | Action |
|---------|--------|
| `@:` | Repeat last command-line command |
| `@@` | Repeat `@:` again |

---

## 8. Joining Lines

| Command | Action |
|---------|--------|
| `J` | Join current line with the line below (adds a space) |
| `gJ` | Join lines without adding a space |
| `3J` | Join 3 lines together |

```
Before:
  Hello
  World

J →
  Hello World
```

---

## 9. Practical Editing Workflow

Here's a realistic editing session combining everything learned:

```
Task: Edit a Python function

1. vim app.py                     ← Open the file
2. /def calculate                 ← Search for the function (Lesson 8)
3. j                              ← Move to next line
4. A                              ← Append at end of line (Insert mode)
5. , timeout=30                   ← Type the new parameter
6. Esc                            ← Back to Normal mode
7. jj                             ← Move down 2 lines
8. dd                             ← Delete a line
9. O                              ← Open line above (Insert mode)
10. # Fixed: added timeout param  ← Type a comment
11. Esc                           ← Back to Normal mode
12. :wq                           ← Save and quit
```

---

## 10. Summary

| Category | Commands | Description |
|----------|----------|-------------|
| Insert | `i`, `a`, `o`, `I`, `A`, `O` | Various ways to enter Insert mode |
| Delete | `x`, `dw`, `dd`, `D` | Delete char, word, line, to end |
| Yank/Put | `yy`, `p`, `P` | Copy line, paste after/before |
| Change | `cw`, `cc`, `C`, `s` | Delete + enter Insert mode |
| Undo | `u`, `Ctrl-r` | Undo, redo |
| Save/Quit | `:w`, `:q`, `:wq`, `ZZ` | Write, quit, both |
| Repeat | `.` | Repeat last change |
| Join | `J` | Join lines |

### The Three Most Important Habits

1. **Stay in Normal mode** — Only enter Insert mode to type, then immediately return
2. **Use `u` freely** — Undo is unlimited, so experiment without fear
3. **Use `.` to repeat** — Plan your edits so `.` can replay them

---

## Exercises

### Exercise 1: Insert Mode Entry Points

Given the line: `function greet(name) {`

Cursor is on the `g` of `greet`. Write the single key to use for each task:

1. Add a comment `// helper` on the line above (new line above, enter insert).
2. Add `return ` at the very start of the line.
3. Add `  // end` on a new blank line below.
4. Add ` async` after the word `function` (cursor currently on `g`; assume you use `b` to move back first — what then?).

<details>
<summary>Show Answer</summary>

1. `O` — opens a new line above and enters Insert mode.
2. `I` — moves cursor to the beginning of the line and enters Insert mode.
3. `o` — opens a new line below and enters Insert mode.
4. After moving the cursor to be at the end of `function` (using `b` to go back then `e` to go to end of word), press `a` — appends after the current character, placing the cursor right after `function`.

</details>

### Exercise 2: Delete Operations

You have this text (cursor on the first character of line 2):

```
Line 1: import os
Line 2: import sys
Line 3: import json
Line 4: import re
```

Perform the following operations in sequence (starting fresh each time):

1. Delete the entire line 2 in one command.
2. Delete just the word `sys` (cursor is on `s` of `sys`).
3. Delete lines 2, 3, and 4 in one command.

<details>
<summary>Show Answer</summary>

1. `dd` — deletes the entire current line.
2. `de` — deletes from the cursor to the end of the current word (`sys`). Alternatively `dw` deletes `sys` and the following space.
3. `3dd` — deletes 3 lines starting from the current line.

</details>

### Exercise 3: Yank, Put, and the Dot Command

You want to duplicate the line `console.log("debug");` five times below itself. Describe the most efficient command sequence.

<details>
<summary>Show Answer</summary>

1. Position cursor on the `console.log("debug");` line.
2. `yy` — yank (copy) the line.
3. `p` — paste below; now you have the line twice.
4. `.` — repeat the last change (paste again); now 3 times.
5. `.` — 4 times.
6. `.` — 5 times.

Total: `yy p . . .` — 6 keystrokes to create 5 duplicates.

Alternatively: `yy 4p` pastes 4 copies at once for a total of 5 lines (original + 4 pastes).

</details>

### Exercise 4: Change vs. Delete

Explain the difference between these two command sequences when the cursor is on the word `old` in the line `The old house`:

- Sequence A: `dw` then `i` then type `new`
- Sequence B: `cw` then type `new`

Which is preferred and why?

<details>
<summary>Show Answer</summary>

Both sequences produce `The new house`, but:

- **Sequence A** (`dw` + `i` + type): Three separate actions — delete word, enter insert mode, type. The dot command (`.`) would only repeat the `i` + type portion, not the delete.
- **Sequence B** (`cw` + type): A single atomic change. The entire operation (delete word + insert new text) is recorded as one undo unit and one dot-repeatable action.

**Sequence B is preferred** because `cw` creates a single repeatable action. If you need to change another occurrence of `old` to `new`, you can simply press `.` after Sequence B.

</details>

### Exercise 5: Undo Tree Exploration

Open Vim with a new file (`vim practice.txt`), then perform these steps:

1. Type `first` in Insert mode, then press `Esc`.
2. Type `A` to append, add ` second`, press `Esc`.
3. Press `u` twice. What is the state of the text?
4. Now press `Ctrl-r` once. What is the state?

<details>
<summary>Show Answer</summary>

1. After step 1: `first`
2. After step 2: `first second`
3. After pressing `u` twice:
   - First `u`: undoes the ` second` append → text is `first`
   - Second `u`: undoes the `first` insert → text is empty (blank buffer)
4. After `Ctrl-r`: redoes the last undone change → text is `first` again.

This demonstrates that each Insert mode session (enter → type → Esc) counts as one undo unit. Two `u` presses undid two separate Insert mode sessions.

</details>

---

**Previous**: [Modes and Basic Navigation](./02_Modes_and_Basic_Navigation.md) | **Next**: [Motions and Navigation](./04_Motions_and_Navigation.md)
