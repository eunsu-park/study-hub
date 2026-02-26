# Operators and Composability

**Previous**: [Motions and Navigation](./04_Motions_and_Navigation.md) | **Next**: [Text Objects](./06_Text_Objects.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Vim's operator + motion grammar and why it makes Vim a "language"
2. Combine operators (`d`, `c`, `y`, `>`, `<`, `gU`, `gu`) with any motion
3. Use counts with operators for precision editing
4. Master the dot command (`.`) to efficiently repeat changes
5. Plan edits around repeatability

---

> **Analogy — Verb + Noun Grammar**: Vim commands are a language. Operators are **verbs** (what to do) and motions are **nouns** (what to act on). Just as in English you combine "delete" + "word" to say "delete a word," in Vim you type `d` + `w` to do exactly that: `dw`. Once you learn 5 verbs and 10 nouns, you have 50 commands — without memorizing 50 separate keystrokes.

This is the most important lesson in the entire guide. The operator + motion grammar is what transforms Vim from a collection of keybindings into a composable editing language. Once this concept clicks, your editing speed will accelerate dramatically.

## Table of Contents

1. [The Grammar of Vim](#1-the-grammar-of-vim)
2. [Operators (Verbs)](#2-operators-verbs)
3. [Combining Operators with Motions](#3-combining-operators-with-motions)
4. [Operators with Counts](#4-operators-with-counts)
5. [The Double-Operator Shortcut](#5-the-double-operator-shortcut)
6. [The Dot Command](#6-the-dot-command)
7. [Thinking in Operators](#7-thinking-in-operators)
8. [Summary](#8-summary)

---

## 1. The Grammar of Vim

Every editing command in Vim follows this formula:

```
[count] operator [count] motion
```

For example:

| Command | Breakdown | English |
|---------|-----------|---------|
| `dw` | `d` + `w` | **d**elete a **w**ord |
| `d3w` | `d` + `3w` | delete 3 words |
| `3dw` | `3` × (`d` + `w`) | delete a word, 3 times |
| `y$` | `y` + `$` | **y**ank to end of line |
| `c2e` | `c` + `2e` | **c**hange 2 word-ends |
| `>j` | `>` + `j` | indent current + next line |
| `gUw` | `gU` + `w` | uppercase a word |

The power: **every new motion you learn works with every operator you know**, and vice versa. Learning is multiplicative, not additive.

---

## 2. Operators (Verbs)

### Core Operators

| Operator | Action | Mnemonic |
|----------|--------|----------|
| `d` | Delete (and copy to register) | **d**elete |
| `c` | Change (delete + enter Insert) | **c**hange |
| `y` | Yank (copy) | **y**ank |
| `>` | Indent right | |
| `<` | Indent left | |
| `=` | Auto-indent | |
| `gU` | Make uppercase | |
| `gu` | Make lowercase | |
| `g~` | Toggle case | |
| `gq` | Format text (wrap to textwidth) | |
| `!` | Filter through external command | |

### How Operators Work

1. Press the operator key — Vim waits for a motion
2. Press a motion — Vim applies the operator over the motion's range
3. Result: text is modified and you're back in Normal mode (or Insert mode for `c`)

---

## 3. Combining Operators with Motions

### Delete (`d`) Combinations

| Command | Action |
|---------|--------|
| `dw` | Delete to next word start |
| `de` | Delete to end of word |
| `db` | Delete back to word start |
| `d$` or `D` | Delete to end of line |
| `d0` | Delete to start of line |
| `d^` | Delete to first non-blank |
| `dG` | Delete to end of file |
| `dgg` | Delete to start of file |
| `dj` | Delete current line + line below |
| `d}` | Delete to next paragraph |
| `df)` | Delete forward through `)` |
| `dt"` | Delete till (before) `"` |
| `d/pattern` | Delete to next match of pattern |

### Yank (`y`) Combinations

| Command | Action |
|---------|--------|
| `yw` | Yank to next word start |
| `ye` | Yank to end of word |
| `y$` | Yank to end of line |
| `y0` | Yank to start of line |
| `yG` | Yank to end of file |
| `ygg` | Yank to start of file |
| `yf;` | Yank forward through `;` |

### Change (`c`) Combinations

| Command | Action |
|---------|--------|
| `cw` | Change to next word (delete + Insert) |
| `ce` | Change to end of word |
| `c$` or `C` | Change to end of line |
| `c0` | Change to start of line |
| `cf"` | Change through `"` |
| `ct)` | Change till `)` |

### Indent (`>`, `<`) Combinations

| Command | Action |
|---------|--------|
| `>j` | Indent current + next line |
| `>}` | Indent to next paragraph |
| `>>` | Indent current line |
| `<<` | Unindent current line |
| `>G` | Indent from here to end of file |

### Case Operators

```
Text: hello world

gUw  → HELLO world      (uppercase one word)
gUe  → HELLO world      (uppercase to end of word)
gU$  → HELLO WORLD      (uppercase to end of line)
guw  → hello world      (lowercase one word)
g~w  → HELLO world      (toggle case of one word)
```

---

## 4. Operators with Counts

Counts can appear before the operator, before the motion, or both:

```
3dw  =  d3w  =  delete 3 words
2y$  =  yank to end of line, 2 times (2 lines)
5>>  =  indent 5 lines
```

### Practical Examples

```
Before: one two three four five six seven
        ^ cursor

d3w  → four five six seven       (deleted 3 words)
2dw  → three four five six seven (deleted 2 words)
3dd  → (deleted 3 lines)         (when on multi-line text)
```

---

## 5. The Double-Operator Shortcut

When you press an operator twice, it applies to the **entire current line**:

| Command | Action |
|---------|--------|
| `dd` | Delete current line |
| `yy` | Yank current line |
| `cc` | Change current line |
| `>>` | Indent current line |
| `<<` | Unindent current line |
| `==` | Auto-indent current line |
| `gUU` | Uppercase current line |
| `guu` | Lowercase current line |
| `g~~` | Toggle case of current line |

This is consistent: **operator + operator = apply to line**.

---

## 6. The Dot Command

The `.` (dot) command repeats the last **change**. A "change" is anything that modifies the text: `d`, `c`, `x`, `p`, `>`, an Insert mode session, etc.

### Basic Usage

```
Scenario: Delete several lines
dd     → Delete first line
.      → Delete next line (repeats dd)
.      → Delete next line
.      → Delete next line
```

```
Scenario: Change a word throughout text
/word      → Search for "word"
cw         → Change it (type replacement, press Esc)
n          → Jump to next occurrence
.          → Repeat the change
n          → Next occurrence
.          → Repeat again
```

### Planning for the Dot Command

Experienced Vim users **craft their edits** to maximize dot-repeatability:

**Less optimal** (not dot-repeatable):
```
I typed "Hello " then Esc  → . would repeat inserting "Hello "
But if I want to add "Hello " at the start of each line, I need:
I → type "Hello " → Esc → j → .  (move down, repeat)
```

**Better** (use `I` to insert at start of line):
```
I          → Insert at line start
Hello      → Type text
Esc        → Return to Normal
j.         → Move down, repeat (adds "Hello " at start of next line)
j.         → Again
```

### What the Dot Command Remembers

The dot command replays the **most recent atomic change**:
- If you typed `dw`, dot repeats `dw`
- If you typed `cw`, typed "new", and pressed `Esc`, dot repeats the entire change (delete word + insert "new")
- If you typed `3dd`, dot repeats `3dd` (deletes 3 lines)

### The Ideal Vim Workflow

```
1. Make a change (the initial edit)
2. Move to the next location
3. Press . to repeat
4. Repeat steps 2-3
```

This pattern — **change, move, repeat** — is the heart of efficient Vim editing.

---

## 7. Thinking in Operators

### The Multiplication Effect

If you know `N` operators and `M` motions, you have `N × M` commands:

```
Operators:  d, c, y, >, <, =, gU, gu  → 8 operators
Motions:    w, e, b, $, 0, ^, gg, G, }, {, f{c}, t{c}, /{pat}  → 13+ motions

Total: 8 × 13 = 104+ unique editing commands
```

Every new operator or motion you learn multiplies your capabilities.

### Example: Learning a New Motion

Suppose you learn `%` (jump to matching bracket). Now you immediately gain:
- `d%` — Delete to matching bracket
- `c%` — Change to matching bracket
- `y%` — Yank to matching bracket
- `>%` — Indent to matching bracket
- `gU%` — Uppercase to matching bracket

Five new commands from learning **one** new motion.

---

## 8. Summary

| Concept | Description |
|---------|-------------|
| Grammar | `[count] operator [count] motion` |
| Core operators | `d` (delete), `c` (change), `y` (yank), `>`/`<` (indent), `gU`/`gu` (case) |
| Double operator | `dd`, `yy`, `cc` — operates on current line |
| Dot command (`.`) | Repeats last change |
| Counts | `3dw` = delete 3 words |
| Multiplication | N operators × M motions = N×M commands |
| Ideal workflow | Change → Move → Repeat (`.`) |

### Key Insight

Vim is not a collection of commands to memorize — it's a **language** to learn. Once you understand the grammar (operator + motion), every new word (operator or motion) you learn expands your vocabulary exponentially.

---

## Exercises

### Exercise 1: Parse the Command

For each Vim command below, identify the operator, motion (and count if present), and describe what it does in plain English:

1. `d3w`
2. `gU$`
3. `>G`
4. `ct;`
5. `ygg`

<details>
<summary>Show Answer</summary>

1. `d3w` — operator: `d` (delete), motion: `3w` (3 words forward). **Delete the next 3 words.**
2. `gU$` — operator: `gU` (uppercase), motion: `$` (end of line). **Uppercase everything from the cursor to end of line.**
3. `>G` — operator: `>` (indent right), motion: `G` (last line). **Indent all lines from the cursor to the end of the file.**
4. `ct;` — operator: `c` (change), motion: `t;` (till just before `;`). **Delete from cursor up to (but not including) the next `;`, then enter Insert mode.**
5. `ygg` — operator: `y` (yank), motion: `gg` (first line). **Copy everything from the cursor to the beginning of the file.**

</details>

### Exercise 2: Choose the Right Operator + Motion

You have this JavaScript line (cursor at `v` of `var`):

```javascript
var userName = getUserInput();
```

Write the single command to accomplish each task:

1. Delete the word `var` and the space after it.
2. Change `getUserInput` to something else (delete it and enter Insert mode).
3. Copy from cursor to end of line.
4. Make `userName` uppercase (cursor is on `u`).

<details>
<summary>Show Answer</summary>

1. `dw` — deletes `var` and moves to the space; or `de` to delete just `var`, then `x` for the space. Most cleanly: `dW` (delete entire WORD including trailing space).
2. Navigate to `g` of `getUserInput`, then `ct(` — change till `(` (deletes `getUserInput` and enters Insert mode, leaving `()` intact).
3. `y$` — yank from cursor to end of line.
4. With cursor on `u` of `userName`, press `gUe` or `gUw` — uppercases to end/start of next word, turning `userName` into `USERNAME` or `USERN` (depends on exact cursor and word boundary). `gUiw` (text object, Lesson 6) is more precise.

</details>

### Exercise 3: Design for Dot-Repeatability

You want to add a semicolon to the end of each line in this JavaScript snippet:

```
const a = 1
const b = 2
const c = 3
const d = 4
```

Describe the most dot-repeatable approach.

<details>
<summary>Show Answer</summary>

**Step-by-step approach:**
1. Place cursor on the first line.
2. Press `A` — jumps to end of line and enters Insert mode.
3. Type `;` (the semicolon).
4. Press `Esc` — return to Normal mode. This records the change: "append `;` at line end".
5. Press `j` — move to next line.
6. Press `.` — repeats the append: adds `;` at the end.
7. Press `j.` for each subsequent line.

The dot command replays `A;Esc` — the entire "append semicolon at line end" operation.

**Even faster**: Use a substitution command (Lesson 8): `:%s/$/;/` adds `;` to every line at once. But the `A;Esc` + `j.` approach demonstrates the dot-repeat principle beautifully.

</details>

### Exercise 4: The Multiplication Effect

You currently know these operators: `d`, `c`, `y`, `>`, `gU`.
You currently know these motions: `w`, `e`, `b`, `$`, `0`, `gg`, `G`, `}`, `f{char}`.

1. How many operator+motion combinations are possible?
2. You just learned the `t{char}` motion. How many new commands does this single addition give you?
3. You just learned the `=` (auto-indent) operator. How many new commands does this give you with your existing motions?

<details>
<summary>Show Answer</summary>

1. 5 operators × 9 motions = **45 combinations**.
2. Learning `t{char}` adds 1 new motion. Now you have 5 operators × 10 motions = 50. The increase is **5 new commands** (one for each operator: `dt{char}`, `ct{char}`, `yt{char}`, `>t{char}`, `gUt{char}`).
3. Learning `=` adds 1 new operator. Now you have 6 operators × 10 motions = 60. The increase is **10 new commands** (one for each existing motion: `=w`, `=e`, `=b`, `=$`, `=0`, `=gg`, `=G`, `=}`, `=f{char}`, `=t{char}`).

This demonstrates that each new operator gives you M new commands (one per existing motion), and each new motion gives you N new commands (one per existing operator).

</details>

### Exercise 5: Complete Editing Task

Starting with this text (cursor at line 1, column 1):

```
hello WORLD
This is a TEST line.
foo BAR baz
```

Describe the exact key sequence to transform it into:

```
hello world
This is a test line.
foo bar baz
```

(Lowercase all-caps words on each line.)

<details>
<summary>Show Answer</summary>

**Line 1**: Cursor on `h`. Press `W` to jump to `WORLD`. Press `gue` to lowercase to end of word → `world`.

**Line 2**: Press `j` to go to line 2. Search forward for uppercase: press `f T` (find `T` of `TEST`). Press `gue` → `test`.

**Line 3**: Press `j` to go to line 3. Press `W` to jump to `BAR`. Press `gue` → `bar`.

Alternatively, for lines 2 and 3 you can use the dot command if the edits are structured similarly:

After doing `gue` on `WORLD`: the change is recorded. Move to `TEST`, press `.` — but dot would lowercase "to end of word" which works here too (same operation). Then move to `BAR`, press `.` again.

So the streamlined sequence is: `W gue j fT . j W .`

</details>

---

**Previous**: [Motions and Navigation](./04_Motions_and_Navigation.md) | **Next**: [Text Objects](./06_Text_Objects.md) — Learn Vim's precision selection mechanism.
