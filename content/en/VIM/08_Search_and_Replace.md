# Search and Replace

**Previous**: [Visual Mode](./07_Visual_Mode.md) | **Next**: [Registers, Marks, and Macros](./09_Registers_Marks_and_Macros.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Search forward (`/`) and backward (`?`) and navigate matches with `n`/`N`
2. Use `*` and `#` to search for the word under the cursor
3. Perform substitutions with `:s` and its flags (`g`, `c`, `i`)
4. Write basic regular expressions for pattern matching in Vim
5. Use the `:g` (global) command for powerful line-based operations

---

Searching is how you navigate large files, and substitution is how you make bulk changes. Together, they form one of Vim's most powerful capabilities. Many tasks that require a separate "Find and Replace" dialog in other editors are just a quick command in Vim.

## Table of Contents

1. [Basic Search](#1-basic-search)
2. [Search Options and Highlighting](#2-search-options-and-highlighting)
3. [Word Under Cursor](#3-word-under-cursor)
4. [Substitution Basics](#4-substitution-basics)
5. [Substitution Ranges](#5-substitution-ranges)
6. [Substitution Flags](#6-substitution-flags)
7. [Regular Expressions in Vim](#7-regular-expressions-in-vim)
8. [The Global Command](#8-the-global-command)
9. [Practical Examples](#9-practical-examples)
10. [Summary](#10-summary)

---

## 1. Basic Search

### Forward Search

```
/pattern     → Search forward for "pattern"
n            → Jump to next match
N            → Jump to previous match
```

### Backward Search

```
?pattern     → Search backward for "pattern"
n            → Jump to next match (in backward direction)
N            → Jump to previous match (in forward direction)
```

### Search as a Motion

Search works as a motion with operators:

```
d/function   → Delete everything from cursor to "function"
y/end        → Yank from cursor to "end"
c/TODO       → Change from cursor to "TODO"
```

### Canceling a Search

Press `Esc` while typing a search pattern to cancel.

---

## 2. Search Options and Highlighting

### Highlight Matches

```vim
:set hlsearch      " Highlight all matches
:set nohlsearch    " Turn off highlighting permanently
:nohlsearch        " Clear current highlights (until next search)
:noh               " Short form
```

### Incremental Search

```vim
:set incsearch     " Show matches as you type (live preview)
```

With `incsearch` on, Vim jumps to the first match as you type each character. Press `Enter` to confirm or `Esc` to cancel.

### Case Sensitivity

```vim
:set ignorecase    " Case-insensitive search
:set smartcase     " Case-sensitive if pattern has uppercase
```

With both `ignorecase` and `smartcase` set:
- `/hello` matches "Hello", "HELLO", "hello"
- `/Hello` matches only "Hello" (uppercase forces case-sensitive)

### Per-Search Case Override

| Pattern | Meaning |
|---------|---------|
| `/pattern\c` | Force case-insensitive |
| `/pattern\C` | Force case-sensitive |

---

## 3. Word Under Cursor

| Command | Action |
|---------|--------|
| `*` | Search forward for word under cursor (whole word match) |
| `#` | Search backward for word under cursor |
| `g*` | Search forward (partial match — no word boundaries) |
| `g#` | Search backward (partial match) |

```python
def calculate_total(items):    ← cursor on "calculate_total"
    return calculate_total_v2(items)

*   → Jumps to next "calculate_total" (whole word, won't match calculate_total_v2)
g*  → Jumps to next occurrence including partial matches
```

This is extremely useful for finding all occurrences of a variable or function name.

---

## 4. Substitution Basics

The substitute command replaces text matching a pattern:

```
:[range]s/pattern/replacement/[flags]
```

### Simple Examples

```vim
:s/old/new/        " Replace first 'old' with 'new' on current line
:s/old/new/g       " Replace ALL 'old' with 'new' on current line
:%s/old/new/g      " Replace all 'old' with 'new' in entire file
:%s/old/new/gc     " Replace all, with confirmation for each
```

### Alternative Delimiters

If your pattern contains `/`, use a different delimiter:

```vim
:s#/usr/local#/opt#g        " Replace paths (using # as delimiter)
:s|http://|https://|g       " Replace URLs (using | as delimiter)
```

---

## 5. Substitution Ranges

The range specifies which lines to affect:

| Range | Meaning |
|-------|---------|
| (nothing) | Current line only |
| `%` | Entire file |
| `1,10` | Lines 1 through 10 |
| `.,$` | Current line to end of file |
| `.,+5` | Current line and next 5 lines |
| `'<,'>` | Visual selection (auto-inserted when you press `:` in Visual) |
| `'a,'b` | From mark `a` to mark `b` |
| `/start/,/end/` | From line matching "start" to line matching "end" |

### Examples

```vim
:10,20s/foo/bar/g       " Replace in lines 10-20
:.,$s/foo/bar/g         " Replace from current line to end
:'<,'>s/foo/bar/g       " Replace in visual selection
:/BEGIN/,/END/s/foo/bar/g   " Replace between BEGIN and END markers
```

---

## 6. Substitution Flags

| Flag | Meaning |
|------|---------|
| `g` | **G**lobal — replace all occurrences on each line (not just first) |
| `c` | **C**onfirm each replacement |
| `i` | Case-**i**nsensitive |
| `I` | Case-sensitive |
| `n` | Count matches only (don't replace) |
| `e` | Suppress errors if no match found |

### Confirmation Mode (`c` flag)

With the `c` flag, Vim asks for each match:

```vim
:%s/old/new/gc
```

At each match, you see:

```
replace with new (y/n/a/q/l/^E/^Y)?
```

| Response | Action |
|----------|--------|
| `y` | Replace this match |
| `n` | Skip this match |
| `a` | Replace **a**ll remaining |
| `q` | **Q**uit substitution |
| `l` | Replace this and quit (**l**ast) |
| `Ctrl-e` | Scroll up |
| `Ctrl-y` | Scroll down |

---

## 7. Regular Expressions in Vim

Vim uses its own regex flavor, which differs slightly from PCRE (Perl-compatible).

### Basic Metacharacters

| Pattern | Matches |
|---------|---------|
| `.` | Any single character |
| `*` | Zero or more of previous |
| `\+` | One or more of previous |
| `\?` | Zero or one of previous |
| `\|` | Alternation (OR) |
| `^` | Start of line |
| `$` | End of line |
| `\<` | Start of word boundary |
| `\>` | End of word boundary |

### Character Classes

| Pattern | Matches |
|---------|---------|
| `[abc]` | Any of a, b, or c |
| `[a-z]` | Lowercase letter |
| `[0-9]` | Digit |
| `[^abc]` | NOT a, b, or c |
| `\d` | Digit (same as `[0-9]`) |
| `\w` | Word character (`[a-zA-Z0-9_]`) |
| `\s` | Whitespace |

### Capture Groups

```vim
:%s/\(\w\+\), \(\w\+\)/\2 \1/g
```

This swaps "Last, First" → "First Last":
- `\(` and `\)` create capture groups
- `\1`, `\2` refer to captured groups in the replacement

### Very Magic Mode

Vim's default regex requires escaping many metacharacters. Use `\v` for "very magic" mode where most characters have special meaning (like PCRE):

```vim
" Default (need to escape +, |, (, ))
:%s/\(foo\|bar\)\+/baz/g

" Very magic (no escaping needed)
:%s/\v(foo|bar)+/baz/g
```

---

## 8. The Global Command

The `:g` command executes a command on every line matching a pattern:

```
:[range]g/pattern/command
```

### Basic Usage

```vim
:g/TODO/d              " Delete all lines containing "TODO"
:g/^$/d                " Delete all blank lines
:g/DEBUG/normal dd     " Delete lines containing "DEBUG"
:g/pattern/p           " Print (show) all matching lines
```

### Inverse Global (`:v`)

`:v` is the inverse — it acts on lines that do NOT match:

```vim
:v/important/d         " Delete all lines NOT containing "important"
:v/\S/d                " Delete all lines without non-whitespace (blank lines)
```

### Advanced Global Examples

```vim
" Move all TODO comments to end of file
:g/TODO/m $

" Copy all function definitions to register a
:g/^def /yank A

" Number all non-blank lines
:g/\S/s/^/\=line('.').' '/

" Sort lines matching a pattern
:g/^import/sort
```

### Global + Substitution

```vim
" Only substitute in lines containing "config"
:g/config/s/old/new/g

" Delete trailing whitespace only in comment lines
:g/^#/s/\s\+$//
```

---

## 9. Practical Examples

### Rename a Variable

```vim
" Rename 'oldName' to 'newName' in entire file
:%s/oldName/newName/g

" Only rename whole-word matches (not 'oldNameExtra')
:%s/\<oldName\>/newName/g

" With confirmation
:%s/\<oldName\>/newName/gc
```

### Clean Up Trailing Whitespace

```vim
:%s/\s\+$//g
```

### Convert Tabs to Spaces

```vim
:%s/\t/    /g          " Replace tabs with 4 spaces
```

### Add Semicolons to Line Ends

```vim
:%s/$/;/               " Add semicolon to every line end
```

### Comment/Uncomment Code

```vim
" Comment lines 10-20 (add # at start)
:10,20s/^/# /

" Uncomment (remove # from start)
:10,20s/^# //
```

### Extract Data

```vim
" Show all lines containing email addresses
:g/\w\+@\w\+\.\w\+/p

" Delete all HTML tags
:%s/<[^>]*>//g

" Convert snake_case to camelCase
:%s/_\(\l\)/\u\1/g
```

---

## 10. Summary

| Category | Command | Description |
|----------|---------|-------------|
| Search | `/pattern`, `?pattern` | Forward/backward search |
| Navigate | `n`, `N` | Next/previous match |
| Word search | `*`, `#` | Search word under cursor |
| Substitute | `:%s/old/new/g` | Replace in entire file |
| Confirm | `:%s/old/new/gc` | Replace with confirmation |
| Range | `:10,20s/old/new/g` | Replace in line range |
| Global | `:g/pattern/command` | Execute on matching lines |
| Inverse | `:v/pattern/command` | Execute on non-matching lines |
| Regex | `\v`, `\<\>`, `\(\)` | Very magic, word boundary, groups |
| Highlight | `:set hlsearch`, `:noh` | Toggle match highlighting |

### Quick Reference

```vim
" Most common operations:
/word           " Find 'word'
*               " Find word under cursor
:%s/old/new/g   " Replace all
:%s/old/new/gc  " Replace with confirm
:g/pattern/d    " Delete matching lines
:v/pattern/d    " Delete non-matching lines
```

---

## Exercises

### Exercise 1: Search Navigation

You are editing a Python file with multiple functions. Describe the exact key sequence to:

1. Search forward for the word `return`.
2. Jump to the next occurrence.
3. Jump back to the previous occurrence.
4. Search for the exact word under the cursor (e.g., cursor is on `calculate`) without typing it again.

<details>
<summary>Show Answer</summary>

1. Press `/return` then `Enter` — searches forward for `return`.
2. Press `n` — jumps to the next match in the same direction.
3. Press `N` — jumps to the previous match (reverses the direction).
4. Press `*` — searches forward for the complete word under the cursor as a whole-word match. This is equivalent to `/\<calculate\>` but requires zero typing.

</details>

### Exercise 2: Substitution Syntax

Write the Vim command for each task:

1. Replace all occurrences of `foo` with `bar` in the entire file.
2. Replace `http://` with `https://` everywhere in the file (note the slash in the pattern — use an alternative delimiter).
3. Replace `old_var` with `new_var` only on lines 15 through 30.
4. Replace `DEBUG` with `INFO` in the current line only, asking for confirmation each time.

<details>
<summary>Show Answer</summary>

1. `:%s/foo/bar/g`
2. `:s#http://#https://#g` or `:%s|http://|https://|g` — using `#` or `|` as the delimiter avoids having to escape the `/` in the URL.
3. `:15,30s/old_var/new_var/g`
4. `:s/DEBUG/INFO/gc` — no `%` so it only applies to the current line; `c` flag confirms each substitution.

</details>

### Exercise 3: The Global Command

Using the `:g` or `:v` command, write the command for each task:

1. Delete all lines that contain the word `TODO`.
2. Delete all blank lines (lines with only whitespace or nothing).
3. Show (print) all lines that contain `import`.
4. Delete all lines that do NOT contain `def ` (keep only function definition lines).

<details>
<summary>Show Answer</summary>

1. `:g/TODO/d`
2. `:g/^\s*$/d` — matches lines with only whitespace characters from start (`^`) to end (`$`).
3. `:g/import/p` — the `p` command prints each matching line to the command area. (`:g/import/#` also shows line numbers.)
4. `:v/def /d` — `:v` is the inverse of `:g`; it deletes lines that do NOT match.

</details>

### Exercise 4: Regex Pattern Writing

Write a Vim substitution command for each transformation:

1. Remove all trailing whitespace from every line in the file.
2. Swap the order of two comma-separated words: turn `Smith, John` into `John Smith` (use capture groups).
3. Replace all standalone occurrences of `count` (not `counter`, `discount`, etc.) with `total`.

<details>
<summary>Show Answer</summary>

1. `:%s/\s\+$//g` — `\s\+` matches one or more whitespace characters, `$` anchors to end of line.
2. `:%s/\(\w\+\), \(\w\+\)/\2 \1/g` — `\(\w\+\)` captures the first word (Last), `, ` matches the separator, `\(\w\+\)` captures the second word (First), then `\2 \1` swaps them.
   Or with very magic mode: `:%s/\v(\w+), (\w+)/\2 \1/g`
3. `:%s/\<count\>/total/g` — `\<` and `\>` are word boundary anchors that ensure `count` is matched as a whole word only.

</details>

### Exercise 5: Combined Search and Replace Workflow

You have a JavaScript file where you want to rename the function `processData` to `handleRequest`. The function is used in multiple places.

Describe the safest workflow using interactive confirmation, and explain what each confirmation response does.

<details>
<summary>Show Answer</summary>

**Command**: `:%s/\<processData\>/handleRequest/gc`

**Breakdown**:
- `%` — entire file
- `s/` — substitute
- `\<processData\>` — match the exact whole word `processData` (word boundaries prevent matching `processDataHelper` etc.)
- `/handleRequest/` — replacement text
- `gc` — global (all occurrences per line) + confirm each one

**At each match, respond:**
- `y` — replace this occurrence (use when you're sure)
- `n` — skip this occurrence (use when this is a different variable or intentional exception)
- `a` — replace all remaining occurrences at once (use when you've reviewed enough and are confident)
- `q` — quit without replacing any more (use if you realize you made a mistake)
- `l` — replace this one and quit (use for the last one you want to change)

The `\<...\>` word boundaries make this safe — it won't accidentally rename `processDataHelper` or other identifiers that merely contain `processData`.

</details>

---

**Previous**: [Visual Mode](./07_Visual_Mode.md) | **Next**: [Registers, Marks, and Macros](./09_Registers_Marks_and_Macros.md)
