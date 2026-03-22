# Command Line and Advanced Features

**Previous**: [Buffers, Windows, and Tabs](./10_Buffers_Windows_and_Tabs.md) | **Next**: [Configuration and Vimrc](./12_Configuration_and_Vimrc.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use Ex commands with ranges, counts, and special addresses
2. Filter text through external shell commands with `:!` and `:read`
3. Work with folds to collapse and expand code sections
4. Save and restore editing sessions for project continuity
5. Use additional productivity features (abbreviations, autocompletion, spell check)

---

Vim's command-line mode is a direct descendant of the `ex` line editor. This heritage gives Vim a powerful command language for batch operations on text — operations that would be tedious with normal-mode editing. Combined with features like folds and sessions, these tools make Vim suitable for managing large codebases.

## Table of Contents

1. [Ex Commands and Ranges](#1-ex-commands-and-ranges)
2. [Shell Integration](#2-shell-integration)
3. [Folds](#3-folds)
4. [Sessions and Views](#4-sessions-and-views)
5. [Autocompletion](#5-autocompletion)
6. [Abbreviations](#6-abbreviations)
7. [Spell Checking](#7-spell-checking)
8. [Miscellaneous Tips](#8-miscellaneous-tips)
9. [Summary](#9-summary)

---

## 1. Ex Commands and Ranges

### Range Syntax

Every Ex command can take a range specifying which lines to act on:

```
:[range]command [arguments]
```

| Range | Meaning |
|-------|---------|
| `.` | Current line |
| `$` | Last line |
| `%` | Entire file (shorthand for `1,$`) |
| `1,10` | Lines 1 to 10 |
| `.,$` | Current line to end of file |
| `.,.+5` | Current line and next 5 |
| `'a,'b` | Between marks a and b |
| `'<,'>` | Visual selection |
| `/pat1/,/pat2/` | Between lines matching patterns |

### Useful Ex Commands

| Command | Action |
|---------|--------|
| `:[range]d` | Delete lines |
| `:[range]y` | Yank lines |
| `:[range]m {address}` | Move lines to address |
| `:[range]co {address}` | Copy lines to address |
| `:[range]t {address}` | Same as `:co` |
| `:[range]p` | Print (display) lines |
| `:[range]w {file}` | Write range to file |

### Line Addressing Examples

```vim
:10d                " Delete line 10
:10,20d             " Delete lines 10-20
:10m20              " Move line 10 to after line 20
:10,20m$            " Move lines 10-20 to end of file
:10t20              " Copy line 10 to after line 20
:10,20t0            " Copy lines 10-20 to beginning of file

" Move current line down 5 lines:
:m+5

" Move current line to top:
:m0
```

### Relative Addressing

```vim
:+3d                " Delete line 3 below cursor
:-2d                " Delete line 2 above cursor
:.,+4d              " Delete current line and next 4

" With patterns:
:/function/d        " Delete next line containing "function"
:/start/,/end/d     " Delete from "start" to "end"
```

### The `:normal` Command

Execute Normal-mode commands from the command line:

```vim
:10,20normal A;           " Append ';' to lines 10-20
:%normal I// ;            " Comment every line with //
:'<,'>normal @a           " Run macro 'a' on visual selection
:g/pattern/normal dd      " Delete specific lines using normal mode
```

---

## 2. Shell Integration

### Running Shell Commands

| Command | Action |
|---------|--------|
| `:!{cmd}` | Run shell command and show output |
| `:!!` | Repeat last shell command |
| `:r !{cmd}` | Read command output into buffer |
| `:.!{cmd}` | Filter current line through command |
| `:[range]!{cmd}` | Filter range through command |

### Examples

```vim
:!ls -la                   " List files
:!python %                 " Run current file (% = filename)
:!git status               " Check git status

" Insert command output:
:r !date                   " Insert current date below cursor
:r !ls                     " Insert directory listing

" Use current file info:
:!echo %                   " Print current filename
:!python %                 " Run current Python file
:!wc %                     " Word count of current file
```

### Filtering Text Through Commands

The filter operator `!` sends text through an external command and replaces it with the output:

```vim
" Sort lines 5-10
:5,10!sort

" Sort the entire file
:%!sort

" Sort visually selected lines
:'<,'>!sort

" Format JSON
:%!python -m json.tool

" Convert markdown to HTML
:%!pandoc

" Remove duplicate lines
:%!sort -u

" Column-align on '='
:'<,'>!column -t -s '='
```

### The Normal Mode Filter

In Normal mode, `!` is an operator that takes a motion:

```
!}sort      → Sort from cursor to next blank line
!Gsort      → Sort from cursor to end of file
!!sort      → Sort current line (!! = filter current line)
```

---

## 3. Folds

Folds collapse sections of text, hiding detail so you can see the big picture.

### Fold Methods

```vim
set foldmethod=manual    " Create folds manually
set foldmethod=indent    " Fold based on indentation
set foldmethod=syntax    " Fold based on syntax (language-aware)
set foldmethod=marker    " Fold at {{{ and }}} markers
set foldmethod=expr      " Custom fold expressions
```

### Basic Fold Commands

| Command | Action |
|---------|--------|
| `zf{motion}` | Create fold (manual method) |
| `zo` | Open fold under cursor |
| `zc` | Close fold under cursor |
| `za` | Toggle fold (open ↔ close) |
| `zR` | Open all folds in file |
| `zM` | Close all folds in file |
| `zd` | Delete fold under cursor |
| `zE` | Delete all folds |

### Fold with Visual Selection

```
V{select lines}zf    → Create fold from visual selection
```

### Fold Levels

| Command | Action |
|---------|--------|
| `zm` | Fold more (decrease fold level by 1) |
| `zr` | Fold less (increase fold level by 1) |
| `:set foldlevel=N` | Set specific fold level |

### Example: Indent-Based Folding

```python
class MyClass:           # ── Level 0 (never folded)
    def method_one(self):    # ── Level 1
        for item in items:       # ── Level 2
            process(item)            # ── Level 3

    def method_two(self):    # ── Level 1
        return result
```

With `foldmethod=indent`:
- `zM` collapses everything
- `zr` reveals level 1 (class body with method signatures)
- `zr` again reveals level 2 (method bodies)

### Markers for Manual Sections

```vim
" In your code:
" Section 1 {{{
code here...
" }}}

" Section 2 {{{
more code...
" }}}

set foldmethod=marker    " Folds at {{{ / }}} pairs
```

---

## 4. Sessions and Views

### Sessions

A session saves everything: open buffers, window layout, tabs, cursor positions, and settings.

```vim
:mksession              " Save session to Session.vim
:mksession! project.vim " Save to specific file (! overwrites)
:source Session.vim     " Restore session
```

From the command line:
```bash
vim -S Session.vim      " Open Vim with session
vim -S project.vim      " Open specific session
```

### Views

A view saves the state of a single window (folds, cursor, local options):

```vim
:mkview              " Save view for current window
:loadview            " Restore view
```

### Auto-Save Sessions

```vim
" Add to .vimrc for automatic session management:
autocmd VimLeave * mksession! ~/.vim/sessions/last.vim
```

---

## 5. Autocompletion

Vim has built-in completion without any plugins.

### Insert Mode Completion

| Key | Completion Type |
|-----|----------------|
| `Ctrl-n` | Next word match (generic) |
| `Ctrl-p` | Previous word match (generic) |
| `Ctrl-x Ctrl-f` | Filename completion |
| `Ctrl-x Ctrl-l` | Whole line completion |
| `Ctrl-x Ctrl-k` | Dictionary word completion |
| `Ctrl-x Ctrl-o` | Omni completion (language-aware) |

### Generic Completion

`Ctrl-n` and `Ctrl-p` complete based on words found in:
- Current buffer
- Other open buffers
- Tag files
- Included files

```python
# Type "cal" then Ctrl-n:
calculate_total     ← Matches from buffer
calculate_tax
calendar
```

### Navigating the Completion Menu

| Key | Action |
|-----|--------|
| `Ctrl-n` | Next item |
| `Ctrl-p` | Previous item |
| `Ctrl-y` | Accept selection |
| `Ctrl-e` | Cancel completion |

---

## 6. Abbreviations

Abbreviations auto-expand text as you type in Insert mode.

### Creating Abbreviations

```vim
:ab teh the              " Auto-correct 'teh' to 'the'
:ab @@ user@example.com  " Expand '@@' to email
:ab #b #!/bin/bash       " Expand '#b' to shebang
```

### Insert Mode Only

```vim
:iab rtfm Read The Fine Manual
```

### Removing Abbreviations

```vim
:una teh              " Remove abbreviation
:abc                  " Clear all abbreviations
```

---

## 7. Spell Checking

```vim
:set spell             " Enable spell checking
:set spelllang=en_us   " Set language
:set nospell           " Disable spell checking
```

### Spell Navigation

| Command | Action |
|---------|--------|
| `]s` | Next misspelled word |
| `[s` | Previous misspelled word |
| `z=` | Show spelling suggestions |
| `zg` | Mark word as good (add to dictionary) |
| `zw` | Mark word as wrong |
| `zug` | Undo `zg` |

---

## 8. Miscellaneous Tips

### Command-Line History

| Key | Action |
|-----|--------|
| `q:` | Open command-line history window |
| `q/` | Open search history window |
| `Ctrl-f` | Switch to history window (from `:` prompt) |

### Redirecting Command Output

```vim
:redir @a          " Start redirecting output to register a
:ls                " Command output goes to register a
:redir END         " Stop redirecting
"ap                " Paste the captured output

" Or redirect to a file:
:redir > output.txt
:ls
:redir END
```

### Diff Mode

```bash
vim -d file1 file2     # Open files in diff mode
```

```vim
:diffsplit file2       " Split and diff with another file
:diffthis              " Mark current window for diff
:diffoff               " Turn off diff mode
]c                     " Jump to next change
[c                     " Jump to previous change
do                     " Diff obtain (pull change from other window)
dp                     " Diff put (push change to other window)
```

### Encryption

```vim
:X                     " Set encryption key for current file
:set key=              " Remove encryption (empty key)
```

---

## 9. Summary

| Category | Key Commands |
|----------|-------------|
| Ex ranges | `.`, `$`, `%`, `1,10`, `/pat1/,/pat2/` |
| Line operations | `:d`, `:m`, `:t`, `:normal` |
| Shell | `:!cmd`, `:r !cmd`, `:[range]!cmd` |
| Folds | `zo`/`zc`/`za`, `zR`/`zM`, `zf` |
| Sessions | `:mksession`, `:source`, `vim -S` |
| Completion | `Ctrl-n`/`Ctrl-p`, `Ctrl-x Ctrl-f/l/o` |
| Abbreviations | `:ab`, `:iab`, `:una` |
| Spell | `:set spell`, `]s`/`[s`, `z=`, `zg` |
| Diff | `vim -d`, `]c`/`[c`, `do`/`dp` |

---

## Exercises

### Exercise 1: Ex Range Syntax

Write the Ex command for each task without using search/replace (use line operations instead):

1. Delete lines 5 through 12.
2. Move lines 20–25 to the end of the file.
3. Copy line 10 to just after line 50.
4. Delete the current line and the next 3 lines below it.
5. Move all lines between the marks `a` and `b` to the top of the file (line 0).

<details>
<summary>Show Answer</summary>

1. `:5,12d`
2. `:20,25m$` — `$` is the last line address
3. `:10t50` — `:t` (same as `:co`) copies; the destination is after line 50
4. `:.,+3d` — `.` is current line, `+3` is 3 lines below
5. `:'a,'bm0` — `'a,'b` is the range between marks, `m0` moves to before line 1 (top)

</details>

### Exercise 2: Shell Filtering

You have a file with a JSON blob that is all on one line and hard to read. You also have a section of 20 lines that need to be sorted alphabetically.

1. What command formats the entire file as pretty-printed JSON using Python's built-in tool?
2. What command sorts only lines 30–50 in place?
3. You want to insert the current date and time at the cursor position. What command does this?
4. What is the difference between `:!sort` and `:%!sort`?

<details>
<summary>Show Answer</summary>

1. `:%!python -m json.tool` — pipes the entire file through Python's JSON formatter, which replaces the buffer content with the pretty-printed version.

2. `:30,50!sort` — pipes lines 30–50 through the system `sort` command and replaces those lines with the sorted output.

3. `:r !date` — the `:r` command reads output into the buffer; `!date` runs the shell `date` command. The result is inserted below the cursor.

4. `:!sort` runs `sort` as a shell command and shows the output in a temporary display (does not modify the buffer). `:%!sort` pipes the entire buffer through `sort` and **replaces** the buffer content with the sorted output. The `%` makes it a filter operation on the whole file.

</details>

### Exercise 3: Working with Folds

You have a Python file and want to use fold features for code navigation.

1. Which `foldmethod` would automatically fold Python functions and classes based on their indentation?
2. You are looking at a file where all folds are closed (`zM` was run). What command opens just one level of folds (showing class bodies, but not method bodies)?
3. What is the key sequence to create a manual fold over a visually selected block of lines?
4. You set fold markers in your code using `{{{` and `}}}`. What `foldmethod` setting enables this?

<details>
<summary>Show Answer</summary>

1. `set foldmethod=indent` — folds are determined by indentation level, which aligns well with Python's indentation-based structure.

2. `zr` — "fold less" increases the fold level by 1, revealing one more level of nesting. After `zM` (all closed), pressing `zr` once opens level 1 folds. You can also use `:set foldlevel=1`.

3. Select the lines with `V` (line-wise visual), then press `zf` — this creates a manual fold covering the selected lines.

4. `set foldmethod=marker` — with this setting, Vim looks for `{{{` to open a fold and `}}}` to close it. You can embed these markers in comments in your code.

</details>

### Exercise 4: The `:normal` Command

Explain what each command does, and write a command to accomplish the described task:

1. `:%normal A;` — what does this do?
2. `:'<,'>normal I# ` — what does this do?
3. Write a command that appends the text ` // TODO` to every line that contains the word `function`.

<details>
<summary>Show Answer</summary>

1. `:%normal A;` — executes `A;` (Append `;` at end of line) in Normal mode on every line in the file (`%`). This adds a semicolon to the end of every line — useful for JavaScript files where semicolons are required.

2. `:'<,'>normal I# ` — executes `I# ` (Insert `# ` at the beginning of line) on every line in the visual selection. This comments out all selected lines with a `#` prefix (Python/shell style).

3. `:g/function/normal A // TODO`
   - `:g/function/` — runs the following command on every line containing "function"
   - `normal A // TODO` — in Normal mode, `A` enters Insert mode at end of line, then ` // TODO` is typed

</details>

### Exercise 5: Abbreviations and Productivity Features

1. Create an abbreviation that expands `addr` to `0x00000000` in Insert mode only.
2. You notice you frequently type `pritn` instead of `print`. Write the abbreviation to auto-correct this.
3. You are writing documentation and want to find all misspelled words. What sequence of commands enables spell checking and jumps to the first misspelled word?
4. After finding a misspelled word, how do you see suggestions and accept the first one?

<details>
<summary>Show Answer</summary>

1. `:iab addr 0x00000000` — `:iab` creates an Insert-mode-only abbreviation. Typing `addr` followed by a space or punctuation in Insert mode will expand it.

2. `:ab pritn print` — `:ab` creates a general abbreviation (works in Insert mode). Whenever you type `pritn` followed by a non-word character, it auto-corrects to `print`.

3. `:set spell` followed by `]s` — `:set spell` enables spell checking (misspelled words are highlighted), and `]s` jumps forward to the next misspelled word.

4. Press `z=` to show the list of suggestions. Each suggestion is numbered. Type the number and press `Enter` to accept it. To accept the first suggestion quickly: `1z=`.

</details>

---

**Previous**: [Buffers, Windows, and Tabs](./10_Buffers_Windows_and_Tabs.md) | **Next**: [Configuration and Vimrc](./12_Configuration_and_Vimrc.md)
