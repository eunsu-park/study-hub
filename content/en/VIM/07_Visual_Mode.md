# Visual Mode

**Previous**: [Text Objects](./06_Text_Objects.md) | **Next**: [Search and Replace](./08_Search_and_Replace.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between character-wise (`v`), line-wise (`V`), and block-wise (`Ctrl-v`) Visual modes
2. Select text using motions and text objects in Visual mode
3. Apply operators to visual selections (delete, yank, change, indent)
4. Use Visual Block mode for column editing (multi-line insert, delete, replace)
5. Reselect previous selections with `gv` and switch selection endpoints with `o`

---

Visual mode lets you **see** what you're selecting before acting on it. While operators + motions/text objects are generally faster for experienced users, Visual mode provides a safety net: you can verify your selection before committing to an action. It's also the only way to perform certain operations like column editing.

## Table of Contents

1. [Three Types of Visual Mode](#1-three-types-of-visual-mode)
2. [Character-wise Visual Mode](#2-character-wise-visual-mode)
3. [Line-wise Visual Mode](#3-line-wise-visual-mode)
4. [Block Visual Mode](#4-block-visual-mode)
5. [Operations on Visual Selections](#5-operations-on-visual-selections)
6. [Useful Visual Mode Techniques](#6-useful-visual-mode-techniques)
7. [Visual Mode vs Operators](#7-visual-mode-vs-operators)
8. [Summary](#8-summary)

---

## 1. Three Types of Visual Mode

| Key | Mode | Selects | Indicator |
|-----|------|---------|-----------|
| `v` | Character-wise | Arbitrary range of characters | `-- VISUAL --` |
| `V` | Line-wise | Whole lines | `-- VISUAL LINE --` |
| `Ctrl-v` | Block-wise | Rectangular block (column) | `-- VISUAL BLOCK --` |

### Switching Between Visual Modes

While in any Visual mode, press a different Visual key to switch:
- In `v` mode, press `V` to switch to line-wise
- In `V` mode, press `Ctrl-v` to switch to block-wise
- Press `Esc` or the same key again to exit Visual mode

---

## 2. Character-wise Visual Mode (`v`)

Press `v` to start selecting character by character. Then use any motion to extend the selection.

```
Text: The quick brown fox jumps over the lazy dog

1. Position cursor on 'q' of "quick"
2. Press v          → Selection starts
3. Press e          → Selects "quick"
4. Press e again    → Extends to "quick brown"
5. Press d          → Deletes selection
```

### Using Motions to Select

All motions work in Visual mode:

| After `v`, press... | Selects |
|---------------------|---------|
| `w` | To next word start |
| `e` | To end of word |
| `$` | To end of line |
| `}` | To next paragraph |
| `f.` | To next period |
| `G` | To end of file |
| `/pattern` | To next match |

### Using Text Objects in Visual Mode

| After `v`, press... | Selects |
|---------------------|---------|
| `iw` | Inner word |
| `i"` | Inner quotes |
| `i(` | Inner parentheses |
| `it` | Inner tag |
| `ip` | Inner paragraph |

```
Text: result = calculate(x, y, z)
                         ^ cursor

v + i( → selects "x, y, z"
v + a( → selects "(x, y, z)"
```

---

## 3. Line-wise Visual Mode (`V`)

Press `V` to select entire lines. Motions extend the selection by whole lines.

```
def function_one():
    return 1            ← cursor here, press V

def function_two():     ← press j (extends selection)
    return 2            ← press j (extends further)

Now press d → all 4 lines deleted
Or press > → all 4 lines indented
```

### Common Line-wise Selections

| Action | Keys |
|--------|------|
| Select current line | `V` |
| Select current + next 4 lines | `V4j` |
| Select to end of file | `VG` |
| Select entire function (between blank lines) | `Vip` |
| Select to matching bracket | `V%` |

---

## 4. Block Visual Mode (`Ctrl-v`)

Block mode selects a **rectangular region** — a column of text. This is unique to Vim and extremely powerful for structured data.

### Basic Column Selection

```
Line 1: apple
Line 2: banana
Line 3: cherry
Line 4: date

1. Cursor on 'a' of "apple"
2. Ctrl-v        → Enter block mode
3. 3j            → Extend down 3 lines
4. e             → Extend to end of word

Selection (highlighted):
Line 1: [apple]
Line 2: [banan]a
Line 3: [cherr]y
Line 4: [date]
```

### Column Insert

Insert text at the same position on multiple lines:

```
Before:
line one
line two
line three

1. Ctrl-v         → Block mode
2. 2j             → Select 3 lines
3. I              → Insert before block (capital I)
4. Type "# "      → Type your text
5. Esc            → Apply to all lines

After:
# line one
# line two
# line three
```

### Column Append

```
Before:
item1
item2
item3

1. Ctrl-v         → Block mode
2. 2j             → Select 3 lines
3. $              → Extend to end of lines
4. A              → Append after block (capital A)
5. Type ","       → Type your text
6. Esc            → Apply to all lines

After:
item1,
item2,
item3,
```

### Column Replace

```
Before:
aaa bbb ccc
aaa bbb ccc
aaa bbb ccc

1. Ctrl-v         → Block mode on first column
2. 2j, 2l         → Select 3×3 block
3. r#             → Replace all selected with #

After:
### bbb ccc
### bbb ccc
### bbb ccc
```

### Column Delete

```
Before:
    line one       (4 spaces indent)
    line two
    line three

1. Ctrl-v         → Block mode
2. 2j             → Select 3 lines
3. 3l             → Select 4 columns
4. d              → Delete

After:
line one
line two
line three
```

---

## 5. Operations on Visual Selections

Once you have a selection, apply any operator:

| Key | Action |
|-----|--------|
| `d` or `x` | Delete selection |
| `c` | Change selection (delete + Insert mode) |
| `y` | Yank (copy) selection |
| `>` | Indent selection |
| `<` | Unindent selection |
| `=` | Auto-indent selection |
| `~` | Toggle case |
| `U` | Uppercase |
| `u` | Lowercase |
| `J` | Join selected lines |
| `gq` | Format/wrap text |
| `:` | Enter command-line for range operations |
| `r{char}` | Replace every character with `{char}` |

### Indent Multiple Lines

```
V       → Select current line
4j      → Extend selection down 4 lines
>       → Indent once (or >> for further indent)

Or to indent more:
V4j     → Select 5 lines
3>      → Indent 3 levels
```

### Sort Lines

```
V       → Line-wise select
{motion to extend}
:sort   → Sort selected lines alphabetically
```

---

## 6. Useful Visual Mode Techniques

### Reselect with `gv`

`gv` reselects the previous visual selection. Useful when you need to apply multiple operations:

```
V5j>    → Select and indent 5 lines
gv>     → Reselect same lines and indent again
```

### Switch Ends with `o`

While in Visual mode, `o` jumps between the start and end of the selection, letting you adjust either boundary:

```
Text: The quick brown fox

v → start at 'T'
w → extend to 'quick' (selection: "The q")
o → cursor jumps to 'T' (start of selection)
b → now you can shrink/grow from the start
```

### Select All

```
ggVG    → Go to first line, Visual line mode, go to last line
```

### Visual Mode Search

You can search while in Visual mode to extend the selection:

```
v         → Start visual
/function → Extend selection to "function"
```

### Increment/Decrement Numbers

```
Selection over numbers:
10
20
30

g Ctrl-a  → Increment each number (becomes 11, 22, 33 — sequential increment)
Ctrl-a    → Increment all by 1 (becomes 11, 21, 31)
```

---

## 7. Visual Mode vs Operators

### When to Use Visual Mode

- **Uncertain selection**: You want to see what you're selecting before acting
- **Irregular ranges**: The selection doesn't map cleanly to a single motion/text object
- **Block operations**: Column editing can only be done in Visual Block mode
- **Teaching/learning**: Visual mode gives visual feedback while learning

### When to Use Operators Directly

- **Known target**: You know exactly what to operate on (e.g., `diw`, `ci"`)
- **Speed**: Operator + motion is fewer keystrokes
- **Dot repeatable**: `dw` is repeatable with `.`, but `vwd` is not (Visual selections aren't repeatable)

### Comparison

| Task | Visual Mode | Operator |
|------|------------|----------|
| Delete word | `viwx` (4 keys) | `diw` (3 keys) |
| Change quotes | `vi"c` (4 keys) | `ci"` (3 keys) |
| Yank to end | `v$y` (3 keys) | `y$` (2 keys) |
| Delete 3 lines | `V2jd` (4 keys) | `3dd` (3 keys) |

Operators are usually more concise and dot-repeatable. Use Visual mode when you need the safety of seeing your selection.

---

## 8. Summary

| Mode | Key | Selection Type | Best For |
|------|-----|---------------|----------|
| Character | `v` | Arbitrary range | Precise text selection |
| Line | `V` | Whole lines | Indenting, moving, deleting lines |
| Block | `Ctrl-v` | Rectangle/column | Column editing, multi-line insert |

| Technique | Keys | Description |
|-----------|------|-------------|
| Reselect | `gv` | Reselect previous selection |
| Switch ends | `o` | Jump between selection boundaries |
| Block insert | `Ctrl-v` → `I` → type → `Esc` | Insert at column |
| Block append | `Ctrl-v` → `$A` → type → `Esc` | Append at end |
| Select all | `ggVG` | Select entire file |

### Practice Tips

1. Start with `V` (line-wise) — it's the most intuitive
2. Use `v` + text objects (`viw`, `vi"`) for practice
3. Try `Ctrl-v` for column editing tasks (commenting, alignment)
4. Gradually transition to direct operators as you gain confidence

---

## Exercises

### Exercise 1: Choose the Right Visual Mode

For each scenario, state which Visual mode (`v`, `V`, or `Ctrl-v`) is most appropriate and why:

1. You want to add `//` to the beginning of 5 lines to comment them out.
2. You want to select just the word `"error"` inside a longer string for replacement.
3. You want to delete 3 complete lines of a function.
4. You want to remove 4 spaces of indentation from the beginning of 6 lines simultaneously.

<details>
<summary>Show Answer</summary>

1. **`Ctrl-v`** (Block Visual) — Column insert allows you to prepend `//` to all 5 lines at once by selecting column 1 across the lines and using `I//Esc`.
2. **`v`** (Character-wise) — You need to select an arbitrary character range within a line, not whole lines.
3. **`V`** (Line-wise) — Whole line operations like deleting complete lines are cleanest in line-wise mode.
4. **`Ctrl-v`** (Block Visual) — Select the first 4 columns across all 6 lines, then press `d` to delete the indentation block.

</details>

### Exercise 2: Block Visual Column Insert

You have this list of items:

```
apple
banana
cherry
date
```

Using Visual Block mode, describe the exact key sequence to transform it into:

```
- apple
- banana
- cherry
- date
```

<details>
<summary>Show Answer</summary>

1. Place cursor on the `a` of `apple` (column 1).
2. Press `Ctrl-v` — enter Visual Block mode.
3. Press `3j` — extend selection down 3 lines (now covering column 1 of all 4 lines).
4. Press `I` — (capital I) enter Insert mode at the start of the block.
5. Type `- ` — (dash space).
6. Press `Esc` — Vim applies the insertion to all 4 lines.

Result: `- ` is prepended to every selected line.

</details>

### Exercise 3: gv and the o Key

You've just used `V3j>` to indent 4 lines. Now you realize you need to indent them one more level. What is the most efficient command?

Bonus: You accidentally selected too far when using character visual mode and want to shrink the selection from the end without restarting. How?

<details>
<summary>Show Answer</summary>

**Main question**: Press `gv>` — `gv` reselects the exact same 4-line visual selection, then `>` indents it again. Much faster than reselecting with `V3j>` again.

**Bonus**: While in Visual mode with the cursor at the far end of the selection, press `o` — this jumps the cursor to the other end (the start of selection). Now you can press motions like `b` or `h` to move the **end** of the selection backwards, effectively shrinking it from what was previously the end.

</details>

### Exercise 4: Visual vs. Operator Efficiency

Rewrite each Visual mode operation as a direct operator + text object combination (no Visual mode):

1. `viwd` — select inner word, then delete
2. `vi"c` — select inside quotes, then change
3. `Vd` — select current line, then delete
4. `vi(y` — select inside parentheses, then yank

<details>
<summary>Show Answer</summary>

1. `viwd` → `diw` — delete inner word (3 keys vs 4)
2. `vi"c` → `ci"` — change inside quotes (3 keys vs 4)
3. `Vd` → `dd` — delete current line (2 keys vs 2 — same!)
4. `vi(y` → `yi(` — yank inside parentheses (3 keys vs 4)

Note: The operator+text object form is always at least as efficient (usually fewer keystrokes) AND it is dot-repeatable, while the Visual mode version is not.

</details>

### Exercise 5: Real-World Block Editing

You have a CSV file where you need to remove the second column from every row:

```
John,Smith,30,Engineer
Jane,Doe,25,Designer
Bob,Jones,35,Manager
```

Using Visual Block mode, describe how to delete the `,Smith`, `,Doe`, `,Jones` parts (the second field including its leading comma).

<details>
<summary>Show Answer</summary>

1. Place cursor on the `,` after `John` (the first comma before `Smith`).
2. Press `Ctrl-v` to enter Visual Block mode.
3. Press `2j` to extend the block down to cover all 3 rows.
4. Press `f,` — move forward to the next `,` (this selects up to and including the comma before `30`, `25`, `35`). The block now spans `,Smith`, `,Doe`, `,Jones`.
5. Press `d` to delete the selected block.

Result:
```
John,30,Engineer
Jane,25,Designer
Bob,35,Manager
```

Note: This works because Visual Block mode applies the same column range to all selected rows, even when the text in each row has different content.

</details>

---

**Previous**: [Text Objects](./06_Text_Objects.md) | **Next**: [Search and Replace](./08_Search_and_Replace.md)
