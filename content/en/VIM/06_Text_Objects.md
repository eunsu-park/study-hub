# Text Objects

**Previous**: [Operators and Composability](./05_Operators_and_Composability.md) | **Next**: [Visual Mode](./07_Visual_Mode.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between **inner** (`i`) and **around** (`a`) text objects
2. Use word, sentence, and paragraph text objects (`iw`, `aw`, `is`, `as`, `ip`, `ap`)
3. Use delimiter-based text objects for quotes, brackets, and tags (`i"`, `a(`, `it`)
4. Combine text objects with operators for precision editing (`diw`, `ci"`, `ya}`)
5. Choose the appropriate text object for common editing scenarios

---

> **Analogy — Precision Surgical Tools**: If motions are like walking to a destination, text objects are like teleporting to exactly the right spot and selecting exactly the right amount of text. They're precision surgical tools — `ci"` says "change everything inside these quotes" without needing to navigate to the quote boundary first. Your cursor just needs to be *anywhere* inside the quotes.

Text objects are one of Vim's most powerful features and what truly sets it apart from other editors. They let you operate on structured units of text — words, sentences, paragraphs, quoted strings, bracketed blocks — regardless of where your cursor is within that unit.

## Table of Contents

1. [What Are Text Objects?](#1-what-are-text-objects)
2. [Inner vs Around](#2-inner-vs-around)
3. [Word Objects](#3-word-objects)
4. [Sentence and Paragraph Objects](#4-sentence-and-paragraph-objects)
5. [Delimiter Objects (Quotes and Brackets)](#5-delimiter-objects-quotes-and-brackets)
6. [Tag Objects](#6-tag-objects)
7. [Block Objects](#7-block-objects)
8. [Practical Examples](#8-practical-examples)
9. [Summary](#9-summary)

---

## 1. What Are Text Objects?

Text objects are special motions that select a **range** of text based on structure. They only work after an **operator** (or in Visual mode) — you can't use them alone in Normal mode.

The syntax:

```
operator  +  a/i  +  object-type
  (verb)    (scope)    (noun)
```

Examples:
- `diw` → **d**elete **i**nner **w**ord
- `ca"` → **c**hange **a**round **"**quotes
- `yi(` → **y**ank **i**nner **(** parentheses

---

## 2. Inner vs Around

Every text object has two variants:

| Prefix | Name | Includes |
|--------|------|----------|
| `i` | **inner** | The content only (not delimiters or surrounding whitespace) |
| `a` | **around** | The content + delimiters or surrounding whitespace |

### Visual Comparison

```
Text: "Hello, World!"

di"  →  ""                 (deletes content, keeps quotes)
da"  →                     (deletes content + quotes)

ci"  →  "|"                (cursor in Insert between quotes)
ca"  →  |                  (everything gone, Insert mode)
```

```
Text: The quick brown fox

diw  →  The  brown fox     (deletes "quick", keeps spaces)
daw  →  The brown fox      (deletes "quick" + one space)
```

**Rule of thumb**:
- Use `i` (inner) when you want to **replace** the content (keep the delimiters)
- Use `a` (around) when you want to **remove** the entire element (including delimiters/spacing)

---

## 3. Word Objects

| Object | Range |
|--------|-------|
| `iw` | Inner word (the word only) |
| `aw` | A word (word + surrounding space) |
| `iW` | Inner WORD |
| `aW` | A WORD |

### Example

```
Text: The quick brown fox
          ^  cursor on 'u'

diw  → The  brown fox          ("quick" deleted)
daw  → The brown fox           ("quick" + space deleted)
ciw  → The | brown fox         (cursor in Insert, ready to type replacement)
yiw  → yanks "quick"           (cursor can be anywhere in the word)
```

The key advantage: **your cursor can be anywhere in the word**. With motions, you'd need to navigate to the word boundary first. With text objects, `diw` deletes the word no matter where within it your cursor sits.

---

## 4. Sentence and Paragraph Objects

| Object | Range |
|--------|-------|
| `is` | Inner sentence |
| `as` | A sentence (includes trailing space) |
| `ip` | Inner paragraph (text between blank lines) |
| `ap` | A paragraph (includes trailing blank line) |

### Sentence Boundaries

Vim considers a sentence to end at `.`, `!`, or `?` followed by whitespace or end-of-line.

```
Text: This is sentence one. This is sentence two. And three.
                             ^ cursor here

dis → This is sentence one.  And three.
das → This is sentence one. And three.
```

### Paragraph Objects

Paragraphs are separated by blank lines. These are extremely useful for code:

```python
def function_one():
    pass

def function_two():    ← cursor anywhere here
    pass

def function_three():
    pass
```

`dip` on `function_two` deletes the entire function (between blank lines). `dap` also removes the trailing blank line.

---

## 5. Delimiter Objects (Quotes and Brackets)

| Object | Range |
|--------|-------|
| `i"` / `a"` | Double-quoted string |
| `i'` / `a'` | Single-quoted string |
| `` i` `` / `` a` `` | Backtick-quoted string |
| `i(` or `i)` / `a(` or `a)` | Parentheses |
| `i[` or `i]` / `a[` or `a]` | Square brackets |
| `i{` or `i}` / `a{` or `a}` | Curly braces |
| `i<` or `i>` / `a<` or `a>` | Angle brackets |

### Quote Examples

```python
message = "Hello, World!"
               ^ cursor here

di"  → message = ""
da"  → message =
ci"  → message = "|"         (Insert mode between quotes)
yi"  → yanks "Hello, World!" content
```

### Bracket Examples

```python
result = calculate(x, y, z)
                    ^ cursor here

di(  → result = calculate()
da(  → result = calculate
ci(  → result = calculate(|)     (Insert mode inside parens)
```

```javascript
const config = {
    host: "localhost",    ← cursor anywhere inside braces
    port: 8080,
};

di{  →  const config = {};
da{  →  const config = ;
```

### Nesting

Text objects respect nesting:

```python
outer(inner(deep), value)
             ^ cursor here

di(  → outer(inner(), value)     (only inner parens)
```

But if you want the outer pair, you'd need to position at the outer level or use a count.

---

## 6. Tag Objects

For HTML/XML/JSX:

| Object | Range |
|--------|-------|
| `it` | Inner tag content |
| `at` | A tag (including opening and closing tags) |

```html
<div class="container">Hello, World!</div>
                        ^ cursor here

dit  → <div class="container"></div>
dat  → (entire element removed)
cit  → <div class="container">|</div>    (Insert mode)
```

This works with nested tags too:

```html
<ul>
  <li>Item one</li>      ← cursor on "one"
  <li>Item two</li>
</ul>

dit  → <li></li>          (inner tag of <li>)
```

---

## 7. Block Objects

For programming language blocks (Vim calls them "Block"):

| Object | Range | Use case |
|--------|-------|----------|
| `iB` or `i{` | Inner `{...}` block | Function/class body |
| `aB` or `a{` | A `{...}` block | Entire block including braces |
| `ib` or `i(` | Inner `(...)` block | Function arguments |
| `ab` or `a(` | A `(...)` block | Including parens |

### Code Example

```javascript
function greet(name, greeting) {
    if (name) {
        console.log(greeting + ", " + name);    ← cursor here
    }
    return true;
}

diB (inner block of enclosing {}):
function greet(name, greeting) {
    if (name) {
    }
    return true;
}

On the outer function level:
daB:
function greet(name, greeting)
```

---

## 8. Practical Examples

### Changing a Function Argument

```python
def process(old_argument):
                ^ cursor anywhere in argument

ciw  →  def process(|):          (change the argument name)
```

### Replacing a String

```python
name = "John Doe"
         ^ cursor anywhere in the string

ci"  →  name = "|"               (ready to type new name)
```

### Deleting a Dictionary/Object Entry

```python
config = {
    "host": "localhost",
    "port": 8080,              ← cursor on this line
    "debug": True,
}

dd   →  deletes the line (simple and effective for one line)
```

### Changing HTML Content

```html
<h1>Old Title</h1>
      ^ cursor here

cit  →  <h1>|</h1>              (type new title)
```

### Wrapping Pattern: `ysi"` (with surround plugin)

While not built-in, the `vim-surround` plugin extends text objects beautifully:
- `cs"'` — Change surrounding `"` to `'`
- `ds"` — Delete surrounding `"`
- `ysiw"` — Add `"` around word

You'll learn about plugins in [Lesson 13](./13_Plugins_and_Ecosystem.md).

### Quick Reference Combinations

| Task | Command |
|------|---------|
| Delete word (any position in word) | `diw` |
| Change quoted string | `ci"` or `ci'` |
| Yank function arguments | `yi(` |
| Delete paragraph | `dip` |
| Indent code block | `>i{` |
| Select inside HTML tag | `vit` |
| Delete including brackets | `da[` |
| Change function body | `ci{` |

---

## 9. Summary

| Object Type | Inner (`i`) | Around (`a`) |
|-------------|-------------|--------------|
| Word | `iw` | `aw` |
| WORD | `iW` | `aW` |
| Sentence | `is` | `as` |
| Paragraph | `ip` | `ap` |
| `"` quotes | `i"` | `a"` |
| `'` quotes | `i'` | `a'` |
| `()` parens | `i(` or `i)` | `a(` or `a)` |
| `[]` brackets | `i[` or `i]` | `a[` or `a]` |
| `{}` braces | `i{` or `i}` | `a{` or `a}` |
| `<>` angles | `i<` or `i>` | `a<` or `a>` |
| HTML tag | `it` | `at` |

### Decision Guide

```
Want to REPLACE content?   → Use inner (i): ci", ciw, ci(
Want to REMOVE entirely?   → Use around (a): da", daw, da(
Want to COPY content?      → Use inner (i): yi", yiw
Want to SELECT?            → Use in Visual: vi", viw, vit
```

### Why Text Objects Matter

1. **Position-independent** — Cursor can be anywhere inside the object
2. **Precise** — Select exactly what you need
3. **Fast** — One command replaces navigate + select + act
4. **Composable** — Work with all operators
5. **Readable** — `ci"` reads as "change inside quotes"

---

## Exercises

### Exercise 1: Inner vs. Around

Given the text: `result = calculate(x + y, z * 2)`

Cursor is somewhere inside `x + y, z * 2`. For each command, describe the resulting text:

1. `di(`
2. `da(`
3. `ci(`
4. `yi(`

<details>
<summary>Show Answer</summary>

1. `di(` — deletes everything inside the parentheses, leaving: `result = calculate()`
2. `da(` — deletes the parentheses and their content, leaving: `result = calculate`
3. `ci(` — deletes the content inside and enters Insert mode, leaving cursor between `(` and `)`: `result = calculate(|)`
4. `yi(` — copies `x + y, z * 2` into the default register; text unchanged.

</details>

### Exercise 2: Choose the Right Text Object

For each editing task, write the single command (operator + text object):

1. You're inside a JSON string `"active"` and want to replace the word with a different value.
2. You're inside a Python function `def process(data, config):` and want to copy all the arguments.
3. You're inside an HTML `<p>Old content</p>` and want to clear the content, ready to type new text.
4. You're inside a Python dict `{"key": "value"}` and want to delete everything including the braces.

<details>
<summary>Show Answer</summary>

1. `ci"` — change inner quotes: deletes `active` and enters Insert mode between `""`.
2. `yi(` — yank inner parentheses: copies `data, config` (without the parens).
3. `cit` — change inner tag: deletes `Old content` and enters Insert mode, leaving the `<p></p>` tags.
4. `da{` — delete around braces: removes `{"key": "value"}` entirely.

</details>

### Exercise 3: Paragraph Object for Code

You have this Python file:

```python
def setup():
    initialize_db()
    load_config()

def main():
    setup()
    run_app()

def cleanup():
    close_db()
    save_state()
```

Cursor is on the line `run_app()`. Write the commands to:

1. Delete the entire `main` function block (not including the blank lines around it).
2. Delete the entire `main` function block including the trailing blank line.

<details>
<summary>Show Answer</summary>

1. `dip` — delete inner paragraph: removes the lines between blank lines (the `def main():`, `setup()`, `run_app()` lines), leaving the surrounding blank lines.
2. `dap` — delete around paragraph: removes the function block AND the trailing blank line.

Note: Your cursor must be somewhere within the `main` function block (not on a blank line) for these to target the correct paragraph.

</details>

### Exercise 4: Nested Text Objects

Given: `outer("inner value", more)`

Cursor is on the `v` of `value`. Answer these questions:

1. What does `di"` do?
2. What does `da"` do?
3. After `da"`, what does the text look like?
4. How would you delete everything inside the outer parentheses (including `"inner value"` and `, more`)?

<details>
<summary>Show Answer</summary>

1. `di"` — deletes the content of the nearest enclosing quotes: removes `inner value`, leaving `outer("", more)`.
2. `da"` — deletes the quotes and their content: removes `"inner value"`, leaving `outer(, more)`.
3. After `da"`: `outer(, more)` — note the leading comma and space remain.
4. Move cursor to be inside `outer(...)` but outside the quotes, then `di(` — deletes everything inside the outer parentheses: `outer()`.

</details>

### Exercise 5: Real-World Editing Scenario

You are editing this JavaScript configuration:

```javascript
const config = {
    apiUrl: "https://old-api.example.com/v1",
    timeout: 5000,
    retries: 3,
};
```

Describe the most efficient way (fewest keystrokes) to:

1. Change the URL to a new address.
2. Change the entire object content to new settings.

<details>
<summary>Show Answer</summary>

**Task 1 (Change the URL)**:
- Position cursor anywhere inside `https://old-api.example.com/v1` (the string content).
- Press `ci"` — deletes the URL content and enters Insert mode between the quotes.
- Type the new URL.
- Press `Esc`.

This is 3 actions: `ci"` + type + `Esc`. No need to navigate to the string boundaries first.

**Task 2 (Change entire object content)**:
- Position cursor anywhere inside the `{...}` block.
- Press `ci{` (or `ciB`) — deletes all content between `{` and `}` and enters Insert mode.
- Type the new settings.
- Press `Esc`.

This is also 3 actions regardless of how many lines the original content spans.

</details>

---

**Previous**: [Operators and Composability](./05_Operators_and_Composability.md) | **Next**: [Visual Mode](./07_Visual_Mode.md)
