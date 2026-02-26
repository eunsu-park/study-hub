# Registers, Marks, and Macros

**Previous**: [Search and Replace](./08_Search_and_Replace.md) | **Next**: [Buffers, Windows, and Tabs](./10_Buffers_Windows_and_Tabs.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use named registers (`"a`-`"z`) to store and retrieve multiple clipboard entries
2. Access special registers (unnamed, numbered, system clipboard, expression)
3. Set and navigate marks for bookmarking positions across files
4. Record, play, and edit macros for automating repetitive edits
5. Apply macros across ranges and files efficiently

---

> **Analogy — Labeled Clipboard Drawers**: Imagine having 26 labeled drawers (a through z) in your desk, each able to hold a snippet of text. When you copy something, you can choose which drawer to put it in. When you paste, you pick which drawer to retrieve from. That's Vim's register system — 26 named clipboards plus several special-purpose ones, all accessible with a single keystroke.

Registers, marks, and macros are Vim's "power tools" — they dramatically amplify your editing capabilities by letting you store, recall, and automate. Many Vim users consider macros the single most productivity-boosting feature.

## Table of Contents

1. [Registers — Multiple Clipboards](#1-registers--multiple-clipboards)
2. [Special Registers](#2-special-registers)
3. [System Clipboard Integration](#3-system-clipboard-integration)
4. [Marks — Bookmarking Positions](#4-marks--bookmarking-positions)
5. [Macros — Recording and Playback](#5-macros--recording-and-playback)
6. [Advanced Macro Techniques](#6-advanced-macro-techniques)
7. [Practical Workflows](#7-practical-workflows)
8. [Summary](#8-summary)

---

## 1. Registers — Multiple Clipboards

Every delete, yank, or change in Vim stores text in a **register**. By default, the unnamed register (`""`) is used, but you can specify any named register.

### Using Named Registers

To specify a register, prefix the command with `"` followed by a register letter:

```
"ayy     → Yank current line into register a
"ap      → Put (paste) contents of register a
"bdd     → Delete line into register b
"bp      → Put contents of register b
"cy3w    → Yank 3 words into register c
```

### Uppercase Appends

Using an uppercase letter **appends** to the register instead of replacing:

```
"ayy     → Yank line into register a (replaces)
jj
"Ayy     → Append this line to register a (now has 2 lines)
"ap      → Pastes both lines
```

### Viewing Registers

```vim
:registers         " Show all registers
:reg               " Short form
:reg a b c         " Show specific registers
```

The output looks like:

```
--- Registers ---
""   last deleted or yanked text
"0   last yank (not delete)
"1   last delete
"a   contents of register a
"b   contents of register b
"+   system clipboard
```

---

## 2. Special Registers

| Register | Name | Contains |
|----------|------|----------|
| `""` | Unnamed | Last delete, change, or yank |
| `"0` | Yank register | Last yank only (not deletes) |
| `"1`-`"9` | Numbered | Last 9 deletes (1=most recent) |
| `"-` | Small delete | Last delete smaller than one line |
| `"+` | System clipboard | System clipboard (see below) |
| `"*` | Primary selection | X11 primary selection (Linux) |
| `".` | Last insert | Text from last Insert mode session |
| `"%` | Current file | Current filename |
| `"#` | Alternate file | Previously edited filename |
| `":` | Last command | Last command-line command |
| `"/` | Last search | Last search pattern |
| `"=` | Expression | Evaluate expression (see below) |
| `"_` | Black hole | Discards text (true delete) |

### The Unnamed Register Problem

When you delete text with `dd`, it goes into the unnamed register (`""`), overwriting whatever was there. This means:

```
yy       → Yank a line (stored in "" and "0)
jjdd     → Delete a line (overwrites "" but NOT "0!)
p        → Pastes the DELETED line (not the yanked one!)
```

**Solution**: Use `"0p` to paste what you last yanked (ignoring deletes):

```
yy       → Yank a line
jjdd     → Delete a line (doesn't affect "0)
"0p      → Paste the YANKED line (from "0)
```

### The Black Hole Register

Use `"_` to delete without affecting any register:

```
"_dd     → Delete line, nothing stored anywhere
"_dw     → Delete word, registers unchanged
```

Useful when you want to delete text without overwriting your yank register.

### The Expression Register

In Insert mode or Command-line mode, `Ctrl-r =` evaluates an expression:

```
In Insert mode:
Ctrl-r = 2 + 3 Enter    → Inserts "5"
Ctrl-r = system('date')  → Inserts current date
```

---

## 3. System Clipboard Integration

### The `+` and `*` Registers

| Register | macOS | Linux (X11) | Windows |
|----------|-------|-------------|---------|
| `"+` | System clipboard | Clipboard (Ctrl-C/V) | Clipboard |
| `"*` | Same as `"+` | Primary selection (mouse select) | Same as `"+` |

### Copy To / Paste From System Clipboard

```
"+yy     → Yank line to system clipboard
"+p      → Paste from system clipboard

"+y      → (in Visual mode) Yank selection to clipboard
"+P      → Paste before cursor from clipboard
```

### Clipboard Option

To make all yank/delete operations automatically use the system clipboard:

```vim
set clipboard=unnamedplus    " Use + register for all y/d/c
```

With this setting, `yy` and `p` work directly with the system clipboard.

### Check Clipboard Support

```bash
vim --version | grep clipboard
# Look for +clipboard (supported) or -clipboard (not supported)
```

If clipboard is not supported, install a full Vim build:
```bash
# macOS (Homebrew vim has clipboard support)
brew install vim

# Ubuntu
sudo apt install vim-gtk3    # or vim-gtk
```

---

## 4. Marks — Bookmarking Positions

Marks were introduced in [Lesson 4](./04_Motions_and_Navigation.md), but here we'll cover them more thoroughly.

### Local Marks (`a-z`)

File-specific bookmarks:

```
ma       → Set mark 'a' at current position
'a       → Jump to line of mark 'a'
`a       → Jump to exact position of mark 'a'
```

### Global Marks (`A-Z`)

Cross-file bookmarks — these work across any file:

```
mA       → Set global mark 'A' here
'A       → Jump to mark 'A' (opens the file if needed)
```

### Listing Marks

```vim
:marks         " Show all marks
:marks aB      " Show specific marks
```

### Useful Mark Patterns

```
" Mark your position before a big jump
ma           → Bookmark current position
/function    → Search (moves you far away)
'a           → Jump back to where you were

" Mark important code locations
mf           → Mark a function definition
mt           → Mark a test file location
'f           → Jump to the function
't           → Jump to the test

" Use global marks for cross-file navigation
mM           → Mark main file
mT           → Mark test file
'M           → Jump to main file from anywhere
```

### Deleting Marks

```vim
:delmarks a      " Delete mark a
:delmarks a-d    " Delete marks a through d
:delmarks!       " Delete all lowercase marks
```

---

## 5. Macros — Recording and Playback

Macros record a sequence of keystrokes for automated replay.

### Recording a Macro

```
qa       → Start recording into register 'a'
...      → Perform your editing actions
q        → Stop recording
```

### Playing a Macro

```
@a       → Play macro in register 'a'
@@       → Replay last played macro
5@a      → Play macro 'a' five times
```

### Step-by-Step Example

Task: Add semicolons to the end of each line.

```
Line 1: let x = 1
Line 2: let y = 2
Line 3: let z = 3
```

```
qa           → Start recording to register a
A;           → Append semicolon at end of line (A enters Insert at end)
Esc          → Return to Normal mode
j            → Move to next line
q            → Stop recording

@a           → Apply to next line
@@           → Apply to the line after that (or 2@a)
```

Result:
```
Line 1: let x = 1;
Line 2: let y = 2;
Line 3: let z = 3;
```

### The Key to Reliable Macros

**Start and end in a predictable position.** A good macro pattern:

1. **Position**: Move to a known position (`0`, `^`, `gg`, etc.)
2. **Edit**: Perform the edit
3. **Advance**: Move to the position for the next iteration (`j`, `n`, etc.)

```
qa              → Record
0               → Go to start of line (predictable position)
f"              → Find first quote
ci"new text     → Change content inside quotes
Esc             → Back to Normal
j               → Move to next line
q               → Stop
```

---

## 6. Advanced Macro Techniques

### Apply Macro to Range

```vim
:5,10normal @a       " Apply macro 'a' to lines 5-10
:%normal @a          " Apply macro 'a' to every line
:'<,'>normal @a      " Apply to visual selection
```

### Recursive Macros

A macro can call itself, creating a loop that stops at an error (e.g., end of file):

```
qqq              → Clear register q (record nothing)
qq               → Start recording to q
...edits...      → Your edits
j                → Move to next line
@q               → Call itself (recursive)
q                → Stop recording
@q               → Start the recursive macro
```

The macro will repeat until it hits an error (like trying to move past the last line), at which point it stops automatically.

### Editing a Macro

Macros are just text stored in registers. You can edit them:

```
" Paste the macro into the buffer
"ap              → Paste register a

" Edit the text (it's just keystrokes)
... make changes ...

" Yank it back into the register
"ayy             → Yank the modified text back to register a
```

Or use `let`:
```vim
:let @a = "Iprefix: \<Esc>j"    " Set register a directly
```

### Appending to a Macro

Use uppercase register to append:

```
qa...q           → Record macro in a
qA...q           → Append more steps to macro a
```

---

## 7. Practical Workflows

### Workflow 1: Multi-Clipboard

```
" Yanking different things for later use:
"ayy             → Yank line to register a (a function signature)
"byy             → Yank another line to register b (an import)
"cyy             → Yank another to register c (a variable)

" Later, paste them wherever needed:
"ap              → Paste the function signature
"bp              → Paste the import
"cp              → Paste the variable
```

### Workflow 2: Macro for Code Transformation

Task: Convert a list of names to a specific format.

```
Before:
john doe
jane smith
bob wilson

Goal:
name: "John Doe",
name: "Jane Smith",
name: "Bob Wilson",
```

```
qa               → Record
I                → Insert at start
name: "          → Type prefix
Esc              → Exit Insert
~                → Toggle case of 'j' → 'J'
w~               → Move to next word, toggle 'd' → 'D'
A",              → Append at end
Esc              → Exit Insert
j0               → Next line, beginning
q                → Stop recording
2@a              → Apply to remaining 2 lines
```

### Workflow 3: Marks for Code Navigation

```
" Working on feature implementation:
mI               → Mark the Interface definition (global mark)
mF               → Mark the Function implementation
mT               → Mark the Test file

" Jump between them:
'I               → Check interface
'F               → Go to implementation
'T               → Run/check tests
```

---

## 8. Summary

### Registers

| Register | Usage | Example |
|----------|-------|---------|
| `"a`-`"z` | Named storage | `"ayy`, `"ap` |
| `"A`-`"Z` | Append to named | `"Ayy` |
| `""` | Unnamed (default) | `yy`, `p` |
| `"0` | Last yank | `"0p` |
| `"1`-`"9` | Delete history | `"1p` |
| `"+` | System clipboard | `"+yy`, `"+p` |
| `"_` | Black hole | `"_dd` |
| `"=` | Expression | `Ctrl-r =` |

### Marks

| Mark | Scope | Example |
|------|-------|---------|
| `a`-`z` | File-local | `ma`, `` `a `` |
| `A`-`Z` | Global (cross-file) | `mA`, `'A` |
| `` ` `` | Last jump position | ``` `` ``` |
| `.` | Last edit position | `` `. `` |

### Macros

| Action | Command |
|--------|---------|
| Record | `q{a-z}` ... `q` |
| Play | `@{a-z}` |
| Replay last | `@@` |
| Play N times | `{N}@{a-z}` |
| Apply to range | `:{range}normal @{a-z}` |
| Edit macro | Paste → edit → yank back |

---

## Exercises

### Exercise 1: The Unnamed Register Problem

Given the following scenario:

1. You yank a line with `yy`.
2. You move down several lines.
3. You delete an unwanted line with `dd`.
4. You press `p` to paste.

What gets pasted — the yanked line or the deleted line? How do you paste the originally yanked line?

<details>
<summary>Show Answer</summary>

**What gets pasted**: The **deleted** line. The `dd` command wrote the deleted content into the unnamed register `""`, overwriting the yanked content.

**How to paste the originally yanked line**: Use `"0p` — the yank register (`"0`) is only updated by yank operations (`y`, `yy`), not by delete operations (`d`, `dd`). So `"0` still holds the originally yanked line even after the `dd`.

This is one of the most common Vim surprises for new users. The fix is to always use `"0p` when you want to paste something you yanked (as opposed to something you deleted).

</details>

### Exercise 2: Multi-Register Copy/Paste

You are editing a Python file and want to copy three separate code snippets to use later. Describe the complete workflow using named registers `a`, `b`, and `c`.

<details>
<summary>Show Answer</summary>

**Copying to named registers:**
1. Navigate to the first snippet (e.g., a function signature), press `"ayy` — yanks the line into register `a`.
2. Navigate to the second snippet (e.g., an import statement), press `"byy` — yanks into register `b`.
3. Navigate to the third snippet (e.g., a configuration constant), press `"cyy` — yanks into register `c`.

**Verifying the registers:**
- Type `:reg a b c` to inspect the contents of all three registers.

**Pasting them later:**
- `"ap` — pastes the function signature.
- `"bp` — pastes the import.
- `"cp` — pastes the configuration constant.

Named registers persist for the entire Vim session, so you can paste them in any order at any location, even after many other edits.

</details>

### Exercise 3: Record and Apply a Macro

You have this list of items that needs formatting:

```
apple
banana
cherry
date
elderberry
```

You want to transform each line to: `- "apple",`

Record a macro in register `q` to do this, then apply it to all 5 lines. Write out each key press in the macro.

<details>
<summary>Show Answer</summary>

**Recording the macro:**
```
qq         → Start recording into register q
0          → Go to start of line (predictable position)
I- "       → Insert at line start: - "
Esc        → Return to Normal mode
A",        → Append at line end: ",
Esc        → Return to Normal mode
j          → Move to next line (advance for next iteration)
q          → Stop recording
```

**Applying to all lines:**
- Position cursor on line 1.
- Record and play the macro once: `@q` processes line 1 (then moves to line 2).
- Press `4@q` to apply to the remaining 4 lines.
- Or use `:%normal @q` to apply to every line at once (but then the `j` in the macro causes double-advances — better to use `4@q`).

**Result:**
```
- "apple",
- "banana",
- "cherry",
- "date",
- "elderberry",
```

</details>

### Exercise 4: Black Hole and Yank Register

You have this text and want to:
1. Copy `important_function()` to paste later.
2. Delete several unwanted lines without losing your copy.
3. Paste `important_function()` at the end.

Describe the complete workflow using the black hole register.

<details>
<summary>Show Answer</summary>

**Step 1**: Navigate to the `important_function()` line and press `yy` — copies it into `""` and `"0`.

**Step 2**: Navigate to each unwanted line and press `"_dd` — deletes the line but sends it to the black hole register `"_` instead of `""`. Your yanked content in `"0` remains intact.

**Step 3**: Navigate to the destination and press `"0p` (or just `p` if you used `"_dd` — since `"_dd` doesn't overwrite `""`, the unnamed register still holds the yank from step 1).

**Why this matters**: Using `"_dd` ensures the delete operation does NOT overwrite the unnamed register. This means you can safely press plain `p` after black hole deletes and still get the original yanked content.

</details>

### Exercise 5: Edit a Macro

You recorded a macro in register `a` to add `console.log("debug");` at the start of each line. After recording, you realize you want to add a newline before it instead of inline. How do you edit the macro without re-recording it from scratch?

<details>
<summary>Show Answer</summary>

**Method 1: Paste, edit, yank back**

1. Open a blank line in your buffer.
2. Press `"ap` — pastes the raw macro contents (they look like characters: `Iconsole.log("debug");\<Esc>j`).
3. Edit the text: find the `I` (which means "Insert at line start") and add `O` (capital O = open line above) before it, or modify as needed.
4. After editing, visually select just that line with `V`.
5. Press `"ay` — yanks the modified line back into register `a`.
6. Now `@a` runs the modified macro.

**Method 2: Set with `:let`**

```vim
:let @a = "Oconsole.log(\"debug\");\<Esc>j"
```

The `:let @a = "..."` command sets the register directly. Use `\<Esc>` for the Escape key and `\"` for literal quotes inside the string.

</details>

---

**Previous**: [Search and Replace](./08_Search_and_Replace.md) | **Next**: [Buffers, Windows, and Tabs](./10_Buffers_Windows_and_Tabs.md)
