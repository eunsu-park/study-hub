# Configuration and Vimrc

**Previous**: [Command Line and Advanced Features](./11_Command_Line_and_Advanced_Features.md) | **Next**: [Plugins and Ecosystem](./13_Plugins_and_Ecosystem.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and organize a `.vimrc` file with essential settings
2. Configure display options, search behavior, and editing preferences
3. Create custom key mappings with `map`, `nmap`, `imap` and use `<leader>`
4. Write autocommands (`autocmd`) for filetype-specific behavior
5. Use filetype detection to customize Vim per language

---

> **Analogy — Car Dashboard Customization**: Your `.vimrc` is like customizing a car's dashboard. You choose which gauges to display (line numbers, status bar), remap controls to fit your preferences (key mappings), and set up automatic behaviors (autocmds — like automatic headlights). Everyone's `.vimrc` is different, reflecting their workflow.

Out of the box, Vim's defaults are quite spartan. A well-configured `.vimrc` transforms Vim from a bare-bones editor into a comfortable development environment. This lesson teaches you to build your own configuration from scratch.

## Table of Contents

1. [The .vimrc File](#1-the-vimrc-file)
2. [Essential Settings](#2-essential-settings)
3. [Key Mappings](#3-key-mappings)
4. [The Leader Key](#4-the-leader-key)
5. [Autocommands](#5-autocommands)
6. [Filetype Settings](#6-filetype-settings)
7. [Status Line](#7-status-line)
8. [Color Schemes](#8-color-schemes)
9. [Building Your .vimrc Incrementally](#9-building-your-vimrc-incrementally)
10. [Summary](#10-summary)

---

## 1. The .vimrc File

### Location

| Editor | File | Location |
|--------|------|----------|
| Vim | `.vimrc` | `~/.vimrc` or `~/.vim/vimrc` |
| Neovim | `init.vim` | `~/.config/nvim/init.vim` |
| Neovim (Lua) | `init.lua` | `~/.config/nvim/init.lua` |

### Creating Your First .vimrc

```bash
touch ~/.vimrc       # Create the file
vim ~/.vimrc         # Edit it
```

### Reloading Without Restarting

```vim
:source ~/.vimrc     " Reload vimrc
:so %                " If you're editing the .vimrc file (% = current file)
```

### Comments

```vim
" This is a comment in Vimscript
" Comments start with a double-quote
```

---

## 2. Essential Settings

### Display Settings

```vim
set number              " Show line numbers
set relativenumber      " Relative line numbers (combined with number = hybrid)
set cursorline          " Highlight current line
set showmatch           " Highlight matching brackets
set showcmd             " Show partial command in status bar
set showmode            " Show current mode (INSERT, VISUAL, etc.)
set laststatus=2        " Always show status line
set ruler               " Show cursor position in status line
set signcolumn=yes      " Always show sign column (prevents text shifting)
set colorcolumn=80      " Show column guide at 80 characters
set wrap                " Wrap long lines visually
set linebreak           " Wrap at word boundaries (not mid-word)
set scrolloff=8         " Keep 8 lines visible above/below cursor
set sidescrolloff=8     " Keep 8 columns visible left/right of cursor
```

### Search Settings

```vim
set hlsearch            " Highlight search matches
set incsearch           " Incremental search (show matches as you type)
set ignorecase          " Case-insensitive search
set smartcase           " Case-sensitive if uppercase in pattern
```

### Editing Settings

```vim
set tabstop=4           " Tab displays as 4 spaces
set shiftwidth=4        " Indent by 4 spaces
set expandtab           " Use spaces instead of tabs
set softtabstop=4       " Tab key inserts 4 spaces
set autoindent          " Copy indent from current line to new line
set smartindent         " Smart auto-indentation
set backspace=indent,eol,start  " Backspace works as expected
```

### File Behavior

```vim
set encoding=utf-8      " UTF-8 encoding
set fileencoding=utf-8  " Save files as UTF-8
set hidden              " Allow switching from unsaved buffers
set autoread            " Auto-reload files changed outside Vim
set noswapfile          " Don't create swap files
set nobackup            " Don't create backup files
set undofile            " Persistent undo (survives quitting Vim)
set undodir=~/.vim/undodir  " Directory for undo files
```

### UI Behavior

```vim
set wildmenu            " Enhanced command-line completion
set wildmode=list:longest,full  " Completion behavior
set mouse=a             " Enable mouse support (all modes)
set splitbelow          " New horizontal splits open below
set splitright          " New vertical splits open to the right
set updatetime=300      " Faster completion (default: 4000ms)
set timeoutlen=500      " Mapping timeout (ms)
```

---

## 3. Key Mappings

### Mapping Commands

| Command | Scope | Recursive |
|---------|-------|-----------|
| `map` | Normal, Visual, Operator-pending | Yes |
| `nmap` | Normal only | Yes |
| `imap` | Insert only | Yes |
| `vmap` | Visual only | Yes |
| `noremap` | Normal, Visual, Op-pending | **No** |
| `nnoremap` | Normal only | **No** |
| `inoremap` | Insert only | **No** |
| `vnoremap` | Visual only | **No** |

**Always use `noremap` variants** to avoid unexpected behavior from recursive mappings.

### Syntax

```vim
nnoremap {keys} {action}
```

### Practical Mappings

```vim
" Quick save
nnoremap <C-s> :w<CR>
inoremap <C-s> <Esc>:w<CR>a

" Quick quit
nnoremap <leader>q :q<CR>

" Clear search highlighting
nnoremap <Esc><Esc> :nohlsearch<CR>

" Move lines up and down
nnoremap <A-j> :m .+1<CR>==
nnoremap <A-k> :m .-2<CR>==
vnoremap <A-j> :m '>+1<CR>gv=gv
vnoremap <A-k> :m '<-2<CR>gv=gv

" Better window navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Buffer navigation
nnoremap [b :bprev<CR>
nnoremap ]b :bnext<CR>

" Keep visual selection when indenting
vnoremap < <gv
vnoremap > >gv

" Y behaves like D and C (yank to end of line)
nnoremap Y y$

" Center screen after jumps
nnoremap n nzzzv
nnoremap N Nzzzv
nnoremap <C-d> <C-d>zz
nnoremap <C-u> <C-u>zz
```

### Special Keys

| Notation | Key |
|----------|-----|
| `<CR>` | Enter/Return |
| `<Esc>` | Escape |
| `<Tab>` | Tab |
| `<BS>` | Backspace |
| `<Space>` | Space bar |
| `<C-x>` | Ctrl+x |
| `<A-x>` or `<M-x>` | Alt+x |
| `<S-x>` | Shift+x |
| `<leader>` | Leader key (default: `\`) |
| `<Nop>` | No operation (disable a key) |

### Removing Mappings

```vim
nunmap {keys}          " Remove Normal mode mapping
iunmap {keys}          " Remove Insert mode mapping
mapclear               " Remove all mappings
```

---

## 4. The Leader Key

The leader key is a prefix for custom shortcuts, avoiding conflicts with built-in commands.

### Setting the Leader

```vim
let mapleader = " "           " Space as leader (very popular)
let maplocalleader = ","      " Local leader for filetype-specific maps
```

### Leader Mappings

```vim
" File operations
nnoremap <leader>w :w<CR>               " Save
nnoremap <leader>q :q<CR>               " Quit
nnoremap <leader>x :x<CR>               " Save and quit

" Buffer operations
nnoremap <leader>b :ls<CR>:b<Space>     " List and switch buffers
nnoremap <leader>d :bd<CR>              " Close buffer

" Search
nnoremap <leader>h :nohlsearch<CR>      " Clear highlights

" Window operations
nnoremap <leader>v :vsplit<CR>          " Vertical split
nnoremap <leader>s :split<CR>           " Horizontal split

" Quick edit vimrc
nnoremap <leader>ev :edit $MYVIMRC<CR>
nnoremap <leader>sv :source $MYVIMRC<CR>
```

---

## 5. Autocommands

Autocommands execute commands automatically in response to events.

### Syntax

```vim
autocmd {event} {pattern} {command}
```

### Common Events

| Event | Trigger |
|-------|---------|
| `BufRead`, `BufNewFile` | Opening a file |
| `BufWritePre` | Before saving a file |
| `BufWritePost` | After saving a file |
| `FileType` | When filetype is detected |
| `VimEnter` | After Vim starts |
| `VimLeave` | Before Vim exits |
| `InsertEnter` | Entering Insert mode |
| `InsertLeave` | Leaving Insert mode |

### Practical Autocommands

```vim
" Remove trailing whitespace on save
autocmd BufWritePre * :%s/\s\+$//e

" Return to last edit position when opening files
autocmd BufReadPost * if line("'\"") > 0 && line("'\"") <= line("$")
    \ | execute "normal! g'\"" | endif

" Auto-resize splits when terminal is resized
autocmd VimResized * wincmd =

" Highlight yanked text briefly
autocmd TextYankPost * silent! lua vim.highlight.on_yank()
```

### Autocommand Groups

Group autocommands to prevent duplicates on reload:

```vim
augroup MyAutocommands
    autocmd!                          " Clear group first
    autocmd BufWritePre * :%s/\s\+$//e
    autocmd FileType python setlocal tabstop=4
    autocmd FileType javascript setlocal tabstop=2
augroup END
```

The `autocmd!` at the start clears the group, preventing duplicate commands when you re-source your `.vimrc`.

---

## 6. Filetype Settings

### Enabling Filetype Detection

```vim
filetype on            " Enable filetype detection
filetype plugin on     " Load filetype-specific plugins
filetype indent on     " Load filetype-specific indentation
" Or all at once:
filetype plugin indent on
```

### Per-Filetype Settings

```vim
augroup FileTypeSettings
    autocmd!
    " Python: 4-space tabs
    autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab

    " JavaScript/TypeScript: 2-space tabs
    autocmd FileType javascript,typescript setlocal tabstop=2 shiftwidth=2 expandtab

    " Go: real tabs
    autocmd FileType go setlocal tabstop=4 shiftwidth=4 noexpandtab

    " Markdown: wrap and spell check
    autocmd FileType markdown setlocal wrap linebreak spell

    " Makefiles: must use tabs
    autocmd FileType make setlocal noexpandtab
augroup END
```

### Filetype Plugin Files

For extensive per-filetype configuration, create files in:

```
~/.vim/ftplugin/python.vim     " Loaded for Python files
~/.vim/ftplugin/javascript.vim " Loaded for JavaScript files
```

Contents of `~/.vim/ftplugin/python.vim`:
```vim
setlocal tabstop=4
setlocal shiftwidth=4
setlocal expandtab
setlocal colorcolumn=79
nnoremap <buffer> <leader>r :!python %<CR>
```

---

## 7. Status Line

### Basic Custom Status Line

```vim
set laststatus=2     " Always show status line

set statusline=
set statusline+=%#PmenuSel#
set statusline+=\ %M              " Modified flag
set statusline+=\ %f              " Filename
set statusline+=%=                " Right-align from here
set statusline+=\ %y              " Filetype
set statusline+=\ %{&encoding}    " Encoding
set statusline+=\ %p%%            " Percentage through file
set statusline+=\ %l:%c           " Line:Column
set statusline+=\
```

### Status Line Components

| Component | Displays |
|-----------|---------|
| `%f` | File path (relative) |
| `%F` | File path (absolute) |
| `%t` | Filename only |
| `%m` | Modified flag `[+]` |
| `%r` | Read-only flag `[RO]` |
| `%y` | Filetype `[python]` |
| `%l` | Current line |
| `%c` | Current column |
| `%p` | Percentage through file |
| `%=` | Separator (left/right alignment) |

Most users prefer a status line plugin (like `lualine` or `lightline`) — see [Lesson 13](./13_Plugins_and_Ecosystem.md).

---

## 8. Color Schemes

### Built-in Color Schemes

```vim
:colorscheme desert    " Apply desert theme
:colorscheme slate     " Apply slate theme
:colorscheme default   " Reset to default
```

List available schemes:
```vim
:colorscheme <Tab>     " Tab-complete through available schemes
```

### True Color Support

```vim
if has('termguicolors')
    set termguicolors    " Enable 24-bit RGB color
endif
```

### Popular Color Schemes (Install as Plugins)

- **gruvbox** — Warm, retro palette
- **tokyonight** — Modern dark/light
- **catppuccin** — Pastel colors
- **onedark** — Inspired by Atom
- **nord** — Arctic, blue-ish

### Syntax Highlighting

```vim
syntax on              " Enable syntax highlighting
syntax enable          " Same, but preserves user color settings
```

---

## 9. Building Your .vimrc Incrementally

### The Approach

1. **Start minimal** — Don't copy someone's 500-line config
2. **Add settings as needed** — When something annoys you, fix it
3. **Understand everything** — Never add a line you don't understand
4. **Comment your changes** — Future you will thank present you

### Starter .vimrc

See `examples/VIM/06_minimal_vimrc.vim` for a well-commented minimal config.

### Common Mistakes to Avoid

1. **Copying massive configs** — You won't understand half of it
2. **Too many plugins too soon** — Learn core Vim first
3. **Fighting Vim's nature** — Don't try to make Vim behave like VS Code
4. **Not using groups for autocmds** — Leads to duplicate commands

---

## 10. Summary

| Category | Key Concepts |
|----------|-------------|
| File | `~/.vimrc`, `:source`, Vimscript comments (`"`) |
| Display | `number`, `relativenumber`, `cursorline`, `scrolloff` |
| Search | `hlsearch`, `incsearch`, `ignorecase`, `smartcase` |
| Editing | `tabstop`, `shiftwidth`, `expandtab`, `autoindent` |
| Mappings | `nnoremap`, `inoremap`, `<leader>`, `<CR>`, `<Nop>` |
| Autocmds | `autocmd`, `augroup`, `FileType`, `BufWritePre` |
| Filetype | `filetype plugin indent on`, `setlocal`, `ftplugin/` |
| Appearance | `colorscheme`, `statusline`, `termguicolors` |

### The Golden Rule

> Your `.vimrc` should grow organically. Every line should solve a problem you've actually encountered.

---

## Exercises

### Exercise 1: Interpreting Settings

For each setting below, explain what it does and why you might want it:

1. `set relativenumber`
2. `set scrolloff=8`
3. `set undofile`
4. `set splitright`

<details>
<summary>Show Answer</summary>

1. `set relativenumber` — shows line numbers relative to the cursor. The current line shows its absolute number; lines above and below show their distance (1, 2, 3...). This makes vertical jump commands like `5j`, `12k`, or `d8j` much easier to use because you can read the jump distance directly from the line number.

2. `set scrolloff=8` — keeps at least 8 lines visible above and below the cursor when scrolling. This provides context — you always see what is around the current line rather than the cursor jumping to the very edge of the screen.

3. `set undofile` — saves undo history to disk. Without this, undo history is lost when you close a file. With `undofile`, you can undo changes made in previous sessions even after restarting Vim.

4. `set splitright` — when you create a vertical split (`:vsp`), the new window opens to the **right** instead of the default left. This matches the natural left-to-right reading direction and feels more intuitive.

</details>

### Exercise 2: `nmap` vs `nnoremap`

1. What is the key difference between `nmap` and `nnoremap`?
2. Given the following mappings, explain the danger of using `nmap`:

```vim
nmap j gj
nmap gj j
```

3. Why is `nnoremap` almost always the correct choice?

<details>
<summary>Show Answer</summary>

1. `nmap` creates a **recursive** mapping — the right-hand side is interpreted as input, so if it contains mapped keys, those mappings are followed too. `nnoremap` creates a **non-recursive** mapping — the right-hand side is interpreted as literal Vim commands, ignoring any other mappings.

2. With those two `nmap` definitions:
   - Pressing `j` would trigger `gj`
   - But `gj` is also mapped to `j`
   - `j` triggers `gj` again
   - This creates an **infinite loop** that hangs Vim. `nmap` chains the mappings recursively.

3. `nnoremap` ignores other mappings on the right-hand side, so you always know exactly what will happen. It avoids accidental chains and infinite loops. The only time you intentionally want `nmap` is when you explicitly want one mapping to trigger another — which is rarely needed and easy to debug poorly.

</details>

### Exercise 3: Writing Autocommands

Write the autocommand (or group of autocommands) for each task:

1. Automatically remove trailing whitespace every time any file is saved.
2. When editing Markdown files, enable word wrap (`wrap linebreak`) and spell checking (`spell`).
3. After saving a Python file, automatically run it with `:!python %`.

Wrap all three in a proper `augroup` block.

<details>
<summary>Show Answer</summary>

```vim
augroup MyProductivity
    autocmd!
    " 1. Remove trailing whitespace on save (all files)
    autocmd BufWritePre * :%s/\s\+$//e

    " 2. Markdown: word wrap and spell check
    autocmd FileType markdown setlocal wrap linebreak spell

    " 3. Auto-run Python file after saving
    autocmd BufWritePost *.py :!python %
augroup END
```

The `autocmd!` at the beginning of the group clears all previously registered autocommands in `MyProductivity`. This prevents duplicates when you re-source your `.vimrc` — without it, each `:source` would add another copy of each autocommand.

Note: Task 3 uses `BufWritePost` (after saving) so the file is already written when Python runs it. Using `BufWritePre` (before saving) would run the old version of the file.

</details>

### Exercise 4: Leader Key Design

A developer sets `let mapleader = " "` (Space as leader) and wants to create these mappings:

1. `<leader>w` — save the current file
2. `<leader>q` — close the current buffer (not just the window)
3. `<leader>ev` — open the `.vimrc` file for editing
4. `<leader>sv` — reload the `.vimrc` file

Write all four as proper `nnoremap` commands.

<details>
<summary>Show Answer</summary>

```vim
let mapleader = " "

nnoremap <leader>w :w<CR>
nnoremap <leader>q :bd<CR>
nnoremap <leader>ev :edit $MYVIMRC<CR>
nnoremap <leader>sv :source $MYVIMRC<CR>
```

Notes:
- `<CR>` is required after Ex commands to submit them (equivalent to pressing Enter)
- `:w` saves the file; `:bd` deletes (closes) the buffer — different from `:q` which closes the window
- `$MYVIMRC` is a special Vim variable that holds the path to the active vimrc file
- Space as leader is comfortable because it is a large key that does nothing useful in Normal mode by default

</details>

### Exercise 5: Filetype Configuration

You are writing a `.vimrc` for a team that works in three languages with different conventions:

- **Python**: 4-space indentation, `expandtab`, 79-column guide
- **JavaScript**: 2-space indentation, `expandtab`
- **Go**: real tab characters, `noexpandtab`, tab width 4

Write the complete `augroup` block for these filetype-specific settings.

<details>
<summary>Show Answer</summary>

```vim
augroup LanguageSettings
    autocmd!

    " Python: PEP 8 style
    autocmd FileType python setlocal
        \ tabstop=4
        \ shiftwidth=4
        \ softtabstop=4
        \ expandtab
        \ colorcolumn=79

    " JavaScript: common JS convention
    autocmd FileType javascript,typescript setlocal
        \ tabstop=2
        \ shiftwidth=2
        \ softtabstop=2
        \ expandtab

    " Go: gofmt uses real tabs
    autocmd FileType go setlocal
        \ tabstop=4
        \ shiftwidth=4
        \ noexpandtab

augroup END
```

Key points:
- `setlocal` applies settings only to the current buffer (unlike `set` which is global)
- The `\` at the start of continuation lines allows multi-line autocommand arguments in Vimscript
- Setting both `tabstop`, `shiftwidth`, and `softtabstop` ensures consistent behavior regardless of how indentation is triggered

</details>

---

**Previous**: [Command Line and Advanced Features](./11_Command_Line_and_Advanced_Features.md) | **Next**: [Plugins and Ecosystem](./13_Plugins_and_Ecosystem.md)
