# VIM Command Cheatsheet

## Modes

| Key | Mode | Purpose |
|-----|------|---------|
| `Esc` | Normal | Navigate and command (default) |
| `i` | Insert | Type text |
| `v` | Visual | Select text |
| `V` | Visual Line | Select lines |
| `Ctrl-v` | Visual Block | Select columns |
| `:` | Command-line | Execute commands |
| `R` | Replace | Overwrite text |

---

## Movement

### Basic
| Key | Action |
|-----|--------|
| `h` `j` `k` `l` | Left, Down, Up, Right |
| `w` / `W` | Next word / WORD |
| `b` / `B` | Previous word / WORD |
| `e` / `E` | End of word / WORD |
| `0` | Start of line |
| `^` | First non-blank |
| `$` | End of line |

### Line Search
| Key | Action |
|-----|--------|
| `f{c}` | Forward to char |
| `F{c}` | Backward to char |
| `t{c}` | Forward till char |
| `T{c}` | Backward till char |
| `;` | Repeat f/t |
| `,` | Repeat f/t reverse |

### File Navigation
| Key | Action |
|-----|--------|
| `gg` | First line |
| `G` | Last line |
| `{N}G` | Line N |
| `Ctrl-d` | Half page down |
| `Ctrl-u` | Half page up |
| `Ctrl-f` | Full page down |
| `Ctrl-b` | Full page up |
| `H` / `M` / `L` | Screen top/mid/bottom |
| `%` | Matching bracket |
| `{` / `}` | Prev/next paragraph |

### Scrolling
| Key | Action |
|-----|--------|
| `zz` | Center current line |
| `zt` | Current line to top |
| `zb` | Current line to bottom |

---

## Operators (Verbs)

| Key | Action |
|-----|--------|
| `d` | Delete |
| `c` | Change (delete + Insert) |
| `y` | Yank (copy) |
| `>` / `<` | Indent / Unindent |
| `=` | Auto-indent |
| `gU` / `gu` | Uppercase / Lowercase |
| `g~` | Toggle case |
| `!` | Filter through command |

**Grammar**: `[count] operator [count] motion/text-object`

---

## Text Objects

| Object | Inner (`i`) | Around (`a`) |
|--------|-------------|--------------|
| Word | `iw` | `aw` |
| WORD | `iW` | `aW` |
| Sentence | `is` | `as` |
| Paragraph | `ip` | `ap` |
| `"` | `i"` | `a"` |
| `'` | `i'` | `a'` |
| `()` | `i(` | `a(` |
| `[]` | `i[` | `a[` |
| `{}` | `i{` | `a{` |
| `<>` | `i<` | `a<` |
| Tag | `it` | `at` |

**Common combos**: `diw` `ci"` `ya(` `dip` `>i{` `vit`

---

## Editing

### Insert Mode Entry
| Key | Where |
|-----|-------|
| `i` / `a` | Before / After cursor |
| `I` / `A` | Start / End of line |
| `o` / `O` | New line below / above |

### Delete
| Key | Action |
|-----|--------|
| `x` | Delete char |
| `dw` | Delete word |
| `dd` | Delete line |
| `D` | Delete to end of line |
| `d$` | Delete to end of line |

### Change
| Key | Action |
|-----|--------|
| `cw` | Change word |
| `cc` | Change line |
| `C` | Change to end of line |
| `s` | Substitute char |
| `S` | Substitute line |

### Copy & Paste
| Key | Action |
|-----|--------|
| `yy` | Yank line |
| `yw` | Yank word |
| `y$` | Yank to end |
| `p` / `P` | Paste after / before |

### Other
| Key | Action |
|-----|--------|
| `u` | Undo |
| `Ctrl-r` | Redo |
| `.` | Repeat last change |
| `J` | Join lines |
| `~` | Toggle case (char) |
| `r{c}` | Replace char with {c} |

---

## Search & Replace

| Command | Action |
|---------|--------|
| `/pattern` | Search forward |
| `?pattern` | Search backward |
| `n` / `N` | Next / Previous match |
| `*` / `#` | Search word under cursor |
| `:%s/old/new/g` | Replace all in file |
| `:%s/old/new/gc` | Replace with confirm |
| `:s/old/new/g` | Replace on current line |
| `:g/pattern/d` | Delete matching lines |
| `:v/pattern/d` | Delete non-matching |

---

## Registers

| Register | Purpose |
|----------|---------|
| `""` | Unnamed (default) |
| `"0` | Last yank |
| `"1`-`"9` | Delete history |
| `"a`-`"z` | Named (user) |
| `"+` | System clipboard |
| `"_` | Black hole (discard) |

**Usage**: `"ayy` (yank to a), `"ap` (paste from a)

---

## Macros

| Key | Action |
|-----|--------|
| `q{a-z}` | Start recording |
| `q` | Stop recording |
| `@{a-z}` | Play macro |
| `@@` | Replay last macro |
| `{N}@{a-z}` | Play N times |

---

## Marks

| Key | Action |
|-----|--------|
| `m{a-z}` | Set local mark |
| `m{A-Z}` | Set global mark |
| `` `{mark} `` | Jump to mark (exact) |
| `'{mark}` | Jump to mark (line) |
| `` `` `` | Last jump position |
| `` `. `` | Last edit position |

---

## Buffers, Windows, Tabs

### Buffers
| Command | Action |
|---------|--------|
| `:e file` | Open file |
| `:ls` | List buffers |
| `:bn` / `:bp` | Next / Previous |
| `:b name` | Switch by name |
| `Ctrl-^` | Toggle alternate |
| `:bd` | Close buffer |

### Windows
| Command | Action |
|---------|--------|
| `:sp` / `:vsp` | Horizontal / Vertical split |
| `Ctrl-w h/j/k/l` | Navigate windows |
| `Ctrl-w =` | Equal size |
| `Ctrl-w _` / `\|` | Maximize height / width |
| `Ctrl-w q` | Close window |
| `Ctrl-w o` | Close all others |

### Tabs
| Command | Action |
|---------|--------|
| `:tabnew` | New tab |
| `gt` / `gT` | Next / Previous tab |
| `:tabclose` | Close tab |

---

## File Commands

| Command | Action |
|---------|--------|
| `:w` | Save |
| `:q` | Quit |
| `:wq` / `ZZ` | Save and quit |
| `:q!` / `ZQ` | Quit without saving |
| `:wa` | Save all |
| `:qa` | Quit all |

---

## Folds

| Key | Action |
|-----|--------|
| `zo` | Open fold |
| `zc` | Close fold |
| `za` | Toggle fold |
| `zR` | Open all |
| `zM` | Close all |

---

## Spell Check

| Key | Action |
|-----|--------|
| `]s` / `[s` | Next / Previous misspelling |
| `z=` | Suggestions |
| `zg` | Mark as good |
