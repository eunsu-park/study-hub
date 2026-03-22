# Buffers, Windows, and Tabs

**Previous**: [Registers, Marks, and Macros](./09_Registers_Marks_and_Macros.md) | **Next**: [Command Line and Advanced Features](./11_Command_Line_and_Advanced_Features.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the relationship between buffers, windows, and tabs in Vim
2. Manage multiple buffers (open, list, switch, delete)
3. Split the screen into multiple windows (horizontal, vertical) and navigate between them
4. Use tabs for grouping window layouts
5. Create an efficient multi-file workflow

---

When you work on real projects, you need to edit multiple files simultaneously. Vim handles this through three concepts that work together: **buffers** (files in memory), **windows** (viewports), and **tabs** (collections of windows). Understanding these correctly is essential — many Vim users misuse tabs because they expect them to work like browser tabs.

## Table of Contents

1. [The Mental Model](#1-the-mental-model)
2. [Buffers — Files in Memory](#2-buffers--files-in-memory)
3. [Windows — Viewports](#3-windows--viewports)
4. [Tabs — Window Layouts](#4-tabs--window-layouts)
5. [The Argument List](#5-the-argument-list)
6. [Practical Multi-File Workflows](#6-practical-multi-file-workflows)
7. [Summary](#7-summary)

---

## 1. The Mental Model

```
┌─── Tab 1 ──────────────────────┐  ┌─── Tab 2 ─────────┐
│  ┌─Window 1──┐ ┌─Window 2──┐  │  │  ┌─Window 1───┐   │
│  │           │ │           │  │  │  │            │   │
│  │ Buffer A  │ │ Buffer B  │  │  │  │ Buffer A   │   │
│  │ (file.py) │ │ (test.py) │  │  │  │ (file.py)  │   │
│  │           │ │           │  │  │  │            │   │
│  └───────────┘ └───────────┘  │  │  └────────────┘   │
└────────────────────────────────┘  └────────────────────┘
```

- **Buffer**: A file loaded into memory. You can have many buffers, but not all need to be visible.
- **Window**: A viewport that displays a buffer. Multiple windows can show the same buffer.
- **Tab**: A collection of windows (a layout). NOT like browser tabs.

**Key insight**: Buffers are the primary unit. You work with buffers; windows and tabs are just different ways to view them.

---

## 2. Buffers — Files in Memory

### Opening Files into Buffers

```vim
:e filename       " Edit (open) a file
:e .              " Open file explorer in current directory
:e **/*.py        " Open file with wildcard (with wildmenu)
```

### Listing Buffers

```vim
:ls               " List all buffers
:buffers          " Same as :ls
```

Output:
```
  1 %a   "app.py"              line 42
  2 #    "test_app.py"         line 1
  3      "utils.py"            line 15
```

| Symbol | Meaning |
|--------|---------|
| `%` | Current buffer (displayed in current window) |
| `#` | Alternate buffer (previous buffer) |
| `a` | Active (loaded and visible) |
| `h` | Hidden (loaded but not displayed) |
| `+` | Modified (unsaved changes) |

### Switching Buffers

| Command | Action |
|---------|--------|
| `:bn` or `:bnext` | Next buffer |
| `:bp` or `:bprev` | Previous buffer |
| `:b {N}` | Switch to buffer number N |
| `:b {name}` | Switch to buffer by partial name |
| `:b#` or `Ctrl-^` | Toggle between current and alternate buffer |
| `:bf` | First buffer |
| `:bl` | Last buffer |

### Partial Name Matching

`:b` supports partial matching and tab completion:

```vim
:b app       " Switch to buffer containing "app" (e.g., app.py)
:b test<Tab> " Tab-complete buffer names
```

### Closing Buffers

| Command | Action |
|---------|--------|
| `:bd` or `:bdelete` | Delete (close) current buffer |
| `:bd {N}` | Delete buffer number N |
| `:bd {name}` | Delete buffer by name |
| `:%bd` | Delete all buffers |
| `:bd!` | Force delete (discard unsaved changes) |

### Hidden Buffers

By default, Vim won't let you switch from a modified buffer. Set `hidden` to allow it:

```vim
set hidden         " Allow switching from unsaved buffers
```

This is almost essential for a multi-buffer workflow.

---

## 3. Windows — Viewports

### Creating Splits

| Command | Action |
|---------|--------|
| `:sp` or `Ctrl-w s` | Horizontal split (same buffer) |
| `:vsp` or `Ctrl-w v` | Vertical split (same buffer) |
| `:sp {file}` | Horizontal split with a new file |
| `:vsp {file}` | Vertical split with a new file |
| `:new` | Horizontal split with empty buffer |
| `:vnew` | Vertical split with empty buffer |

### Visual Layout

```
:sp (horizontal split)      :vsp (vertical split)

┌──────────────────┐        ┌─────────┬─────────┐
│    Window 1      │        │         │         │
│    (buffer A)    │        │ Window 1│ Window 2│
├──────────────────┤        │(buff A) │(buff A) │
│    Window 2      │        │         │         │
│    (buffer A)    │        │         │         │
└──────────────────┘        └─────────┴─────────┘
```

### Navigating Between Windows

All window commands use the `Ctrl-w` prefix:

| Command | Action |
|---------|--------|
| `Ctrl-w h` | Move to window left |
| `Ctrl-w j` | Move to window below |
| `Ctrl-w k` | Move to window above |
| `Ctrl-w l` | Move to window right |
| `Ctrl-w w` | Cycle to next window |
| `Ctrl-w W` | Cycle to previous window |
| `Ctrl-w p` | Go to previous (last accessed) window |

### Resizing Windows

| Command | Action |
|---------|--------|
| `Ctrl-w =` | Make all windows equal size |
| `Ctrl-w _` | Maximize current window height |
| `Ctrl-w \|` | Maximize current window width |
| `Ctrl-w +` | Increase height by 1 line |
| `Ctrl-w -` | Decrease height by 1 line |
| `Ctrl-w >` | Increase width by 1 column |
| `Ctrl-w <` | Decrease width by 1 column |
| `{N}Ctrl-w _` | Set height to N lines |
| `{N}Ctrl-w \|` | Set width to N columns |

### Moving Windows

| Command | Action |
|---------|--------|
| `Ctrl-w r` | Rotate windows (swap positions) |
| `Ctrl-w R` | Rotate in reverse |
| `Ctrl-w x` | Exchange with neighbor |
| `Ctrl-w H` | Move window to far left (full height) |
| `Ctrl-w J` | Move window to bottom (full width) |
| `Ctrl-w K` | Move window to top (full width) |
| `Ctrl-w L` | Move window to far right (full height) |

### Closing Windows

| Command | Action |
|---------|--------|
| `:q` or `Ctrl-w q` | Close current window |
| `:only` or `Ctrl-w o` | Close all windows except current |
| `:qa` | Close all windows and quit |

---

## 4. Tabs — Window Layouts

Vim tabs are NOT like browser tabs. Each tab is a **window layout** — a collection of split windows. You'd use them for different tasks or perspectives on the same project.

### Creating and Closing Tabs

| Command | Action |
|---------|--------|
| `:tabnew` | Open a new tab with an empty buffer |
| `:tabnew {file}` | Open file in new tab |
| `:tabe {file}` | Same as `:tabnew {file}` |
| `:tabclose` | Close current tab |
| `:tabonly` | Close all other tabs |

### Navigating Tabs

| Command | Action |
|---------|--------|
| `gt` | Go to next tab |
| `gT` | Go to previous tab |
| `{N}gt` | Go to tab number N |
| `:tabn` | Next tab |
| `:tabp` | Previous tab |
| `:tabfirst` | First tab |
| `:tablast` | Last tab |

### Listing Tabs

```vim
:tabs         " Show all tabs and their windows
```

### When to Use Tabs

**Good uses**:
- Separate concerns: code tab + test tab + docs tab
- Comparing files: each tab shows a different arrangement
- Temporary workspace: open a tab, do something, close it

**Anti-pattern**: Using tabs as file tabs (one tab per file). Use **buffers** for that — they're more efficient and Vim is designed around buffer-based workflow.

---

## 5. The Argument List

The argument list is the list of files passed to Vim on startup, but you can modify it dynamically.

### Basic Usage

```bash
vim *.py         # Opens all .py files (they become the arglist)
```

```vim
:args            " Show current argument list
:args *.js       " Set arglist to all .js files
:next            " Next file in arglist
:prev            " Previous file
:first           " First file
:last            " Last file
```

### Apply Commands Across Arglist

```vim
" Replace in all arglist files:
:argdo %s/old/new/ge | update

" Run macro across all arglist files:
:argdo normal @a | update
```

### Buffer Equivalent: `:bufdo`

```vim
:bufdo %s/old/new/ge | update    " Replace in all open buffers
```

---

## 6. Practical Multi-File Workflows

### Workflow 1: Two-File Split (Source + Test)

```vim
:e app.py            " Open source file
:vsp test_app.py     " Vertical split with test file
Ctrl-w l             " Switch to test window
" ... edit test ...
Ctrl-w h             " Switch back to source
" ... edit source ...
```

### Workflow 2: Buffer-Based Navigation

```vim
set hidden           " Enable hidden buffers

:e models.py         " Open file 1
:e views.py          " Open file 2
:e urls.py           " Open file 3

:ls                  " See all buffers
:b mod<Tab>          " Switch to models.py (tab completion)
Ctrl-^               " Toggle between last two buffers
:bn                  " Next buffer
:bp                  " Previous buffer
```

### Workflow 3: Tab Workspaces

```vim
" Tab 1: Main code (split view)
:e main.py
:vsp utils.py

" Tab 2: Tests
:tabnew test_main.py
:vsp test_utils.py

" Tab 3: Configuration
:tabnew config.yaml

" Navigate:
gt / gT              " Switch between tabs
Ctrl-w h/l           " Switch windows within tab
```

### Recommended Key Mappings

```vim
" Quick buffer navigation
nnoremap <leader>b :ls<CR>:b<Space>

" Quick window navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Buffer switching
nnoremap [b :bprev<CR>
nnoremap ]b :bnext<CR>
```

---

## 7. Summary

### Buffers

| Command | Action |
|---------|--------|
| `:e {file}` | Open file |
| `:ls` | List buffers |
| `:bn`, `:bp` | Next/previous buffer |
| `:b {name}` | Switch by name |
| `Ctrl-^` | Toggle alternate buffer |
| `:bd` | Close buffer |

### Windows

| Command | Action |
|---------|--------|
| `:sp`, `:vsp` | Horizontal/vertical split |
| `Ctrl-w h/j/k/l` | Navigate windows |
| `Ctrl-w =` | Equalize size |
| `Ctrl-w _`, `Ctrl-w \|` | Maximize height/width |
| `Ctrl-w q` | Close window |
| `Ctrl-w o` | Close all others |

### Tabs

| Command | Action |
|---------|--------|
| `:tabnew {file}` | New tab |
| `gt`, `gT` | Next/previous tab |
| `{N}gt` | Go to tab N |
| `:tabclose` | Close tab |

### The Mental Model, Again

```
Buffers = Files in memory (the data)
Windows = Viewports into buffers (the view)
Tabs    = Window arrangements (the layout)
```

Use **buffers** as your primary file management tool. Use **windows** when you need to see multiple things at once. Use **tabs** sparingly, for distinct workspaces.

---

## Exercises

### Exercise 1: Buffer vs Window vs Tab

Classify each scenario below: is the person using buffers, windows, or tabs correctly or incorrectly? Explain why.

1. A developer opens one tab per file — `app.py` in Tab 1, `models.py` in Tab 2, `views.py` in Tab 3.
2. A developer opens all files with `:e` and navigates between them using `:bn`/`:bp` and `Ctrl-^`.
3. A developer uses `:vsp` to show `app.py` and `test_app.py` side by side in the same tab.
4. A developer creates two tabs: Tab 1 shows the source split with tests, Tab 2 shows documentation.

<details>
<summary>Show Answer</summary>

1. **Incorrect** — This is the classic anti-pattern. Vim tabs are window layouts, not file holders. Using one tab per file is wasteful and defeats the buffer system. Use `:e` to open each file as a buffer and navigate with `:bn`/`:bp` or `:b {name}`.

2. **Correct** — This is the intended buffer workflow. `Ctrl-^` quickly toggles between the two most recently used buffers, which is ideal for jumping back and forth between e.g. source and test.

3. **Correct** — Windows (splits) are exactly for viewing multiple buffers simultaneously. `:vsp` creates a vertical split so you can see both files at once.

4. **Correct** — Using tabs for distinct "workspaces" or concerns (code layout vs docs layout) is a legitimate use of Vim tabs.

</details>

### Exercise 2: Reading the Buffer List

After running `:ls`, you see:

```
  1 %a   "main.py"        line 10
  2 #    "utils.py"       line 1
  3  h   "config.py"      line 5
  4  h+  "README.md"      line 1
```

Answer the following questions:

1. Which file is currently displayed in the active window?
2. What does pressing `Ctrl-^` do right now?
3. Which buffer has unsaved changes?
4. How many buffers are loaded in memory but NOT visible?

<details>
<summary>Show Answer</summary>

1. **`main.py`** — the `%` symbol marks the current buffer (displayed in the current window), and `a` means it is active (visible).

2. **Switches to `utils.py`** — `Ctrl-^` (same as `:b#`) toggles to the alternate buffer, marked with `#`. `utils.py` is the alternate buffer.

3. **`README.md`** — the `+` flag means it has unsaved changes. Buffer 4 (`README.md`) shows `h+` — hidden AND modified.

4. **Two buffers**: `config.py` (buffer 3) and `README.md` (buffer 4), both marked `h` (hidden — loaded but not displayed in any window).

</details>

### Exercise 3: Window Navigation Sequence

You have this window layout (cursor is in Window A):

```
┌──────────────┬──────────────┐
│   Window A   │   Window B   │
│   (main.py)  │  (test.py)   │
├──────────────┴──────────────┤
│          Window C           │
│         (utils.py)          │
└─────────────────────────────┘
```

Write the exact key sequence to:

1. Move from Window A to Window B.
2. Move from Window B to Window C.
3. From Window C, move to Window A.
4. Maximize Window C's height.
5. Restore all windows to equal size.

<details>
<summary>Show Answer</summary>

1. `Ctrl-w l` — moves right from A to B.
2. `Ctrl-w j` — moves down from B to C.
3. `Ctrl-w k` — moves up from C; since A and B are both above, this goes to the window directly above. Alternatively, `Ctrl-w w` cycles forward through windows.
4. `Ctrl-w _` — maximizes the current window's height.
5. `Ctrl-w =` — equalizes all window sizes.

</details>

### Exercise 4: Multi-File Rename with `:bufdo`

You have opened 5 Python files and need to rename the variable `user_id` to `account_id` across all of them. All 5 files are currently loaded as buffers.

1. What single command performs this rename across all open buffers and saves each file?
2. Why is `set hidden` often needed before running this command?
3. What does the `e` flag in `%s/old/new/ge` do, and why is it useful here?

<details>
<summary>Show Answer</summary>

1. `:bufdo %s/user_id/account_id/ge | update`
   - `:bufdo` executes the command in every loaded buffer
   - `%s/user_id/account_id/ge` replaces all occurrences (`g`) and suppresses errors if not found (`e`)
   - `| update` saves each buffer only if it was modified

2. Without `set hidden`, Vim refuses to switch away from a modified buffer (one with unsaved changes). `:bufdo` needs to move through every buffer, so if any buffer has pending changes and `hidden` is not set, the command will fail with an error. Setting `set hidden` allows Vim to keep modified buffers loaded in the background without displaying them.

3. The `e` flag suppresses the error message `"Pattern not found"` when a buffer does not contain `user_id`. Without `e`, `:bufdo` would abort on the first buffer where the pattern is not found, leaving the remaining buffers unprocessed.

</details>

### Exercise 5: Tab Workspace Design

You are starting work on a web project with these files: `app.py`, `models.py`, `views.py`, `test_app.py`, `test_models.py`, and `README.md`.

Design a tab + window layout using Vim commands. Requirements:
- Tab 1: Show source code (you want to see `app.py` and `models.py` side by side)
- Tab 2: Show tests (`test_app.py` and `test_models.py` side by side)
- Tab 3: Show documentation (`README.md` alone)

Write the sequence of Vim commands to set up this layout from scratch.

<details>
<summary>Show Answer</summary>

```vim
" Tab 1: Source code side by side
:e app.py
:vsp models.py

" Tab 2: Tests side by side
:tabnew test_app.py
:vsp test_models.py

" Tab 3: Documentation
:tabnew README.md

" Navigate back to Tab 1
:tabfirst
" or press: 1gt
```

After setup, use `gt` / `gT` to switch between tabs, and `Ctrl-w h` / `Ctrl-w l` to switch between the split windows within a tab.

Note: `app.py`, `models.py`, etc. are also accessible as buffers from any tab — you don't need to re-open them. The tabs just provide different **views** of those buffers.

</details>

---

**Previous**: [Registers, Marks, and Macros](./09_Registers_Marks_and_Macros.md) | **Next**: [Command Line and Advanced Features](./11_Command_Line_and_Advanced_Features.md)
