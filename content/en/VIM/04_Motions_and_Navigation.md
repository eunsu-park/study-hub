# Motions and Navigation

**Previous**: [Essential Editing](./03_Essential_Editing.md) | **Next**: [Operators and Composability](./05_Operators_and_Composability.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Navigate by words using `w`, `b`, `e` and their WORD variants `W`, `B`, `E`
2. Move within lines using `0`, `^`, `$`, `f`, `t`, `F`, `T`
3. Scroll and jump through files using `Ctrl-d`, `Ctrl-u`, `gg`, `G`, and line numbers
4. Use marks to bookmark positions and jump back to them
5. Navigate the jump list and change list with `Ctrl-o`, `Ctrl-i`, and `g;`

---

In Lesson 2, you learned `h`, `j`, `k`, `l` — the fundamental cursor movements. But navigating one character at a time is like walking when you could be driving. This lesson teaches you to move by words, lines, screens, and landmarks — dramatically speeding up your navigation.

## Table of Contents

1. [Word Motions](#1-word-motions)
2. [Line Motions](#2-line-motions)
3. [Character Search Within a Line](#3-character-search-within-a-line)
4. [Screen Motions](#4-screen-motions)
5. [File Motions](#5-file-motions)
6. [Marks](#6-marks)
7. [Jump List and Change List](#7-jump-list-and-change-list)
8. [Matching Brackets](#8-matching-brackets)
9. [Summary](#9-summary)

---

## 1. Word Motions

Vim distinguishes between **word** (lowercase) and **WORD** (uppercase):

- **word**: A sequence of letters, digits, and underscores, OR a sequence of other non-blank characters. Separated by whitespace or punctuation boundaries.
- **WORD**: A sequence of any non-blank characters. Separated only by whitespace.

```
Example:   my_var.method(arg1, arg2)

words:     my_var . method ( arg1 ,   arg2 )     ← 8 words
WORDs:     my_var.method(arg1,  arg2)             ← 2 WORDs
```

### Word Navigation Commands

| Command | Action |
|---------|--------|
| `w` | Move to start of next **word** |
| `W` | Move to start of next **WORD** |
| `b` | Move to start of previous **word** (back) |
| `B` | Move to start of previous **WORD** |
| `e` | Move to end of current/next **word** |
| `E` | Move to end of current/next **WORD** |
| `ge` | Move to end of previous **word** |
| `gE` | Move to end of previous **WORD** |

### Visual Example

```
Text: Hello, World! How are you?
      ^

w:    Hello, World! How are you?     (5 presses to reach "you")
             ^  ^    ^   ^   ^
W:    Hello, World! How are you?     (3 presses to reach "you")
             ^      ^   ^   ^

b:    (from "you") goes: are → How → World! → Hello,
B:    (from "you") goes: are → How → World! → Hello,
```

**When to use WORD motions**: When you want to skip over punctuation quickly (e.g., navigating URLs, file paths, or code with lots of symbols).

---

## 2. Line Motions

| Command | Action |
|---------|--------|
| `0` | Move to first column (column 0) |
| `^` | Move to first non-blank character |
| `$` | Move to end of line |
| `g_` | Move to last non-blank character |
| `+` | Move to first non-blank of next line |
| `-` | Move to first non-blank of previous line |

### Difference Between `0` and `^`

```
    def my_function():
^   ^
0   ^  (caret)

0 goes to column 0 (the space before "def")
^ goes to 'd' (first non-blank character)
```

Use `^` most of the time — you usually want the first meaningful character, not the leading whitespace.

---

## 3. Character Search Within a Line

These commands search for a specific character on the current line:

| Command | Action |
|---------|--------|
| `f{char}` | Move **f**orward to next occurrence of `{char}` |
| `F{char}` | Move backward to previous occurrence of `{char}` |
| `t{char}` | Move forward **t**ill (just before) `{char}` |
| `T{char}` | Move backward till (just after) `{char}` |
| `;` | Repeat last `f`/`F`/`t`/`T` in same direction |
| `,` | Repeat last `f`/`F`/`t`/`T` in opposite direction |

### Example

```
Text: def calculate_total_price(items, tax_rate):
      ^  cursor here

fp → def calculate_total_price(items, tax_rate):
                              ^  (jumped to first 'p' in "price")

;  → def calculate_total_price(items, tax_rate):
                                              ^  (next occurrence — no more p visible, actually jumps to nowhere or stays)

ft → def calculate_total_price(items, tax_rate):
                         ^  (jumped to first 't' in "total")
;  → next 't'
```

### Why `t` Exists

The `t`ill motion is especially useful with operators. For example, to delete everything up to (but not including) a parenthesis:

```
Text: calculate_total_price(items)
      ^

dt(  → (items)    ← deleted up to but NOT including '('
df(  → items)     ← deleted up to AND including '('
```

---

## 4. Screen Motions

These commands scroll the viewport without moving to a specific line:

### Scrolling

| Command | Action |
|---------|--------|
| `Ctrl-d` | Scroll **d**own half a screen |
| `Ctrl-u` | Scroll **u**p half a screen |
| `Ctrl-f` | Scroll **f**orward (down) one full screen |
| `Ctrl-b` | Scroll **b**ack (up) one full screen |
| `Ctrl-e` | Scroll down one line (cursor stays) |
| `Ctrl-y` | Scroll up one line (cursor stays) |

### Cursor Position on Screen

| Command | Action |
|---------|--------|
| `H` | Move cursor to **H**igh (top of screen) |
| `M` | Move cursor to **M**iddle of screen |
| `L` | Move cursor to **L**ow (bottom of screen) |
| `zz` | Center current line on screen |
| `zt` | Move current line to **t**op of screen |
| `zb` | Move current line to **b**ottom of screen |

`zz` is particularly useful: after jumping to a line, center it so you can see context above and below.

---

## 5. File Motions

| Command | Action |
|---------|--------|
| `gg` | Go to first line of file |
| `G` | Go to last line of file |
| `{N}G` | Go to line number N (e.g., `42G`) |
| `:{N}` | Go to line number N (e.g., `:42`) |
| `{N}%` | Go to N% through the file (e.g., `50%` = middle) |

### Paragraph and Sentence Motions

| Command | Action |
|---------|--------|
| `{` | Move to previous blank line (paragraph boundary) |
| `}` | Move to next blank line |
| `(` | Move to previous sentence |
| `)` | Move to next sentence |

These are useful for navigating prose or code separated by blank lines.

---

## 6. Marks

Marks are bookmarks — they save a cursor position so you can jump back later.

### Setting Marks

| Command | Action |
|---------|--------|
| `m{a-z}` | Set a local mark (file-specific) |
| `m{A-Z}` | Set a global mark (works across files) |

### Jumping to Marks

| Command | Action |
|---------|--------|
| `` `{mark} `` | Jump to exact position (line and column) |
| `'{mark}` | Jump to beginning of marked line |

### Special Marks

| Mark | Meaning |
|------|---------|
| `` ` `` or `''` | Position before last jump |
| `` `. `` | Position of last change |
| `` `^ `` | Position of last insert |
| `` `[ `` | Start of last yank/change |
| `` `] `` | End of last yank/change |
| `` `< `` | Start of last visual selection |
| `` `> `` | End of last visual selection |

### Example Workflow

```
1. You're reading a function at line 150
2. ma                    ← Set mark 'a' here
3. Navigate elsewhere to check something
4. 'a                    ← Jump back to line 150
5. `a                    ← Jump to exact column too
```

---

## 7. Jump List and Change List

### Jump List

Vim maintains a list of locations you've "jumped" to. A jump is any motion that moves you far from the cursor (like `G`, `gg`, `/search`, marks, etc.).

| Command | Action |
|---------|--------|
| `Ctrl-o` | Go to previous position in jump list (**o**lder) |
| `Ctrl-i` | Go to next position in jump list (newer) |
| `:jumps` | Show the jump list |

Think of `Ctrl-o` as the "Back" button in a web browser.

### Change List

Vim also tracks where you've made changes:

| Command | Action |
|---------|--------|
| `g;` | Go to previous change position |
| `g,` | Go to next change position |
| `:changes` | Show the change list |

This is extremely useful: "Where was I just editing?" → `g;` takes you there.

---

## 8. Matching Brackets

| Command | Action |
|---------|--------|
| `%` | Jump to matching bracket: `()`, `[]`, `{}` |

```
if (condition) {       ← cursor on '{'
    do_something();
}                      ← % jumps here

if (condition) {       ← % jumps back here
    do_something();
}                      ← cursor on '}'
```

`%` works with `(`, `)`, `[`, `]`, `{`, `}` and can be extended with plugins to match HTML tags, if/endif, etc.

---

## 9. Summary

| Category | Commands | Description |
|----------|----------|-------------|
| Word | `w`/`b`/`e`, `W`/`B`/`E` | Move by word / WORD |
| Line | `0`, `^`, `$`, `g_` | Start, first non-blank, end of line |
| Character | `f`/`F`/`t`/`T`, `;`/`,` | Find char on line, repeat |
| Screen | `Ctrl-d`/`u`/`f`/`b` | Scroll half/full screen |
| Screen position | `H`/`M`/`L`, `zz`/`zt`/`zb` | Cursor / viewport positioning |
| File | `gg`/`G`/`{N}G` | Start/end/line N of file |
| Paragraph | `{`/`}` | Previous/next blank line |
| Marks | `m{a-z}`, `` `{mark} `` | Set and jump to bookmarks |
| Jump list | `Ctrl-o`/`Ctrl-i` | Navigate jump history |
| Change list | `g;`/`g,` | Navigate change history |
| Brackets | `%` | Jump to matching bracket |

### Navigation Strategy

1. **Large jumps**: `gg`, `G`, `/{pattern}`, marks
2. **Medium jumps**: `{`, `}`, `Ctrl-d`, `Ctrl-u`, `H`/`M`/`L`
3. **Small jumps**: `w`, `b`, `e`, `f{char}`
4. **Micro adjustments**: `h`, `j`, `k`, `l`

Work from large to small: jump close, then refine.

---

## Exercises

### Exercise 1: word vs. WORD

Given this line: `http://api.example.com/v2/users?limit=10`

With the cursor at the very beginning (`h`):

1. How many `w` presses does it take to reach `limit`?
2. How many `W` presses does it take to reach `limit`?
3. Explain why the counts differ.

<details>
<summary>Show Answer</summary>

1. **`w` count**: The URL contains many word boundaries (`:`, `/`, `.`, `?`, `=` each count as boundaries), so it takes many presses — approximately 12–14 `w` presses to reach `limit`.
2. **`W` count**: Only whitespace separates WORDs. The entire URL is a single WORD with no spaces inside it, so `W` would jump past the whole URL in 1 press (to the next token after a space, if any). Since `limit` is inside the same WORD as `http://...`, you cannot reach it with `W` — it would overshoot.

3. **Explanation**: `word` boundaries include punctuation changes (letters→symbols→letters), so `w` stops at each `.`, `/`, `?`, `=`, and `=`. `WORD` only stops at whitespace, making it much faster for skipping over long tokens like URLs or complex expressions.

**Practical takeaway**: Use `W` to jump over entire URLs or symbol-heavy tokens, use `w` for fine-grained navigation within them.

</details>

### Exercise 2: Line Navigation

Given this Python line (cursor at column 0, which is a space):
```
    return user.profile.settings["theme"]
```

Write the single command to move to each target position:

1. The `r` in `return`.
2. The `[` before `"theme"`.
3. The `]` at the end.

<details>
<summary>Show Answer</summary>

1. `^` — moves to the first non-blank character (`r` in `return`). Note: `0` would go to the very first column (the space), not `r`.
2. `f[` — finds the next `[` forward on the current line.
3. `$` — moves to the end of the line (the `]`). Alternatively, `f]` would also work.

</details>

### Exercise 3: Marks for Multi-Location Editing

You are editing a 500-line configuration file. You need to:
1. Edit a section near line 50.
2. Then edit a section near line 300.
3. Then return to line 50 to verify your first edit.

Describe the complete workflow using marks.

<details>
<summary>Show Answer</summary>

```
1. Navigate to line 50 (e.g., :50 or 50G)
2. ma                 ← Set mark 'a' at line 50
3. Make your edits at line 50
4. Navigate to line 300 (e.g., :300 or 300G)
5. mb                 ← Set mark 'b' at line 300 (optional but good practice)
6. Make your edits at line 300
7. 'a                 ← Jump back to beginning of line 50
   OR
   `a                 ← Jump back to exact cursor position at line 50
8. Verify your first edit
```

**Tip**: Use lowercase marks `a`-`z` for within-file bookmarks. The backtick (`` ` ``) version (`\`a`) returns to the exact column, while the apostrophe version (`'a`) returns to the first non-blank character of the marked line.

</details>

### Exercise 4: The `%` Jump

You see this code but something looks wrong with the brackets:

```python
def process(data):
    result = [x for x in data if (x > 0 and x < 100]
    return result
```

Describe how you would use `%` to diagnose the bracket mismatch.

<details>
<summary>Show Answer</summary>

1. Place the cursor on the `[` after `result = `.
2. Press `%` — Vim should jump to the matching `]`.
3. Observe that `%` lands on the `]` at the end — but notice the parenthesis `(` inside the list comprehension has no closing `)`.
4. Move to the `(` after `if `.
5. Press `%` — Vim will try to find the matching `)` but it doesn't exist, so `%` will not jump (or will show an error).

This reveals the bug: there is an opening `(` with no closing `)`. The fix is to add `)` before the `]`:
```python
    result = [x for x in data if (x > 0 and x < 100)]
```

</details>

### Exercise 5: Jump List Navigation

You're working on a large file and perform these actions in order:

1. Start at line 1.
2. Press `G` to go to the last line (line 200).
3. Press `50%` to go to line 100.
4. Press `gg` to return to line 1.

After all these jumps, what sequence of `Ctrl-o` presses would return you to line 200? What is the jump list at that point?

<details>
<summary>Show Answer</summary>

The jump list (most recent first) after all four actions:
```
Position 1 (current): line 1   ← gg
Position 2:           line 100 ← 50%
Position 3:           line 200 ← G
Position 4:           line 1   ← starting position
```

To return to line 200:
- Press `Ctrl-o` once → goes to line 100 (one step back in jump list)
- Press `Ctrl-o` again → goes to line 200 (two steps back)

So **two `Ctrl-o` presses** to reach line 200.

To go forward again: press `Ctrl-i` to navigate forward in the jump list.

</details>

---

**Previous**: [Essential Editing](./03_Essential_Editing.md) | **Next**: [Operators and Composability](./05_Operators_and_Composability.md)
