# Plugins and Ecosystem

**Previous**: [Configuration and Vimrc](./12_Configuration_and_Vimrc.md) | **Next**: [Neovim and Modern Workflows](./14_Neovim_and_Modern_Workflows.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and manage plugins using vim-plug (Vim) and lazy.nvim (Neovim)
2. Configure essential plugins for file navigation, fuzzy finding, and Git
3. Use surround, commentary, and other text manipulation plugins
4. Understand the basics of writing a simple Vim plugin
5. Evaluate plugins and build a curated, maintainable plugin set

---

Plugins extend Vim's capabilities beyond its already powerful core. The ecosystem offers thousands of plugins, but you only need a handful to build an excellent development environment. This lesson covers plugin management, essential plugins, and the principles for choosing wisely.

## Table of Contents

1. [Plugin Managers](#1-plugin-managers)
2. [Essential Plugins](#2-essential-plugins)
3. [Text Manipulation Plugins](#3-text-manipulation-plugins)
4. [File Navigation Plugins](#4-file-navigation-plugins)
5. [Git Plugins](#5-git-plugins)
6. [Language Support](#6-language-support)
7. [Writing a Simple Plugin](#7-writing-a-simple-plugin)
8. [Plugin Philosophy](#8-plugin-philosophy)
9. [Summary](#9-summary)

---

## 1. Plugin Managers

### vim-plug (Recommended for Vim)

The most popular plugin manager for Vim — minimal, fast, and easy.

**Installation:**
```bash
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

**Usage in .vimrc:**
```vim
call plug#begin('~/.vim/plugged')

Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

call plug#end()
```

**Commands:**
| Command | Action |
|---------|--------|
| `:PlugInstall` | Install plugins |
| `:PlugUpdate` | Update plugins |
| `:PlugClean` | Remove unused plugins |
| `:PlugStatus` | Check plugin status |

### lazy.nvim (Recommended for Neovim)

A modern, Lua-based plugin manager for Neovim with lazy loading.

```lua
-- Bootstrap lazy.nvim (in init.lua)
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git", "clone", "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable", lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)

require("lazy").setup({
  "tpope/vim-surround",
  "numToStr/Comment.nvim",
  { "nvim-telescope/telescope.nvim", dependencies = { "nvim-lua/plenary.nvim" } },
})
```

### Other Plugin Managers

| Manager | Language | Lazy Loading | Notes |
|---------|----------|-------------|-------|
| vim-plug | Vimscript | Basic | Most popular for Vim |
| lazy.nvim | Lua | Advanced | Standard for Neovim |
| packer.nvim | Lua | Yes | Predecessor to lazy.nvim |
| Vundle | Vimscript | No | Older, less maintained |
| pathogen | Vimscript | No | Simple but manual |

---

## 2. Essential Plugins

### The "Must Have" List

These plugins are almost universally recommended:

| Plugin | Purpose | Author |
|--------|---------|--------|
| vim-surround | Surround text objects | tpope |
| vim-commentary | Toggle comments | tpope |
| vim-repeat | Make `.` work with plugins | tpope |
| fzf.vim / telescope.nvim | Fuzzy finding | junegunn / nvim-telescope |
| vim-fugitive | Git integration | tpope |
| NERDTree / nvim-tree | File explorer | scrooloose / nvim-tree |

Tim Pope's plugins (`tpope/*`) are considered the gold standard of Vim plugin quality.

---

## 3. Text Manipulation Plugins

### vim-surround

Add, change, and delete surrounding characters (quotes, brackets, tags).

```vim
Plug 'tpope/vim-surround'
```

| Command | Before | After |
|---------|--------|-------|
| `cs"'` | `"hello"` | `'hello'` |
| `cs'<q>` | `'hello'` | `<q>hello</q>` |
| `ds"` | `"hello"` | `hello` |
| `ysiw"` | `hello` | `"hello"` |
| `yss(` | `hello world` | `( hello world )` |
| `yss)` | `hello world` | `(hello world)` |

**Key mappings:**
- `cs{old}{new}` — **C**hange **s**urrounding
- `ds{char}` — **D**elete **s**urrounding
- `ys{motion}{char}` — Add surrounding (**y**ou **s**urround)
- `S{char}` — Surround in Visual mode

### vim-commentary (or Comment.nvim)

Toggle comments with a single command.

```vim
" Vim
Plug 'tpope/vim-commentary'
```

| Command | Action |
|---------|--------|
| `gcc` | Toggle comment on current line |
| `gc{motion}` | Toggle comment over motion |
| `gcap` | Comment a paragraph |
| `gc` (Visual) | Comment selection |

```lua
-- Neovim (Comment.nvim)
{ "numToStr/Comment.nvim", config = true }
```

### vim-repeat

Makes the `.` command work with plugin actions (surround, commentary, etc.).

```vim
Plug 'tpope/vim-repeat'
```

No configuration needed — just install it and `.` will repeat plugin commands.

### vim-unimpaired

Pairs of bracket mappings for common operations.

```vim
Plug 'tpope/vim-unimpaired'
```

| Command | Action |
|---------|--------|
| `[b` / `]b` | Previous/next buffer |
| `[q` / `]q` | Previous/next quickfix |
| `[l` / `]l` | Previous/next location list |
| `[<Space>` | Add blank line above |
| `]<Space>` | Add blank line below |
| `[e` / `]e` | Exchange line up/down |

---

## 4. File Navigation Plugins

### fzf.vim (Vim) / Telescope (Neovim)

Fuzzy finders are the fastest way to navigate projects.

**fzf.vim:**
```vim
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

" Key mappings
nnoremap <leader>f :Files<CR>       " Find files
nnoremap <leader>g :Rg<CR>          " Grep content
nnoremap <leader>b :Buffers<CR>     " Switch buffers
nnoremap <leader>l :Lines<CR>       " Search lines
nnoremap <leader>c :Commits<CR>     " Git commits
```

| Command | Action |
|---------|--------|
| `:Files` | Fuzzy find files |
| `:Buffers` | Fuzzy find buffers |
| `:Rg {pattern}` | Ripgrep search |
| `:Lines` | Search across all buffer lines |
| `:BLines` | Search in current buffer |
| `:Commits` | Browse git commits |
| `:History` | Recent files |

**Telescope (Neovim):**
```lua
{
  "nvim-telescope/telescope.nvim",
  dependencies = { "nvim-lua/plenary.nvim" },
  keys = {
    { "<leader>f", "<cmd>Telescope find_files<cr>" },
    { "<leader>g", "<cmd>Telescope live_grep<cr>" },
    { "<leader>b", "<cmd>Telescope buffers<cr>" },
  },
}
```

### File Explorer

**NERDTree (Vim):**
```vim
Plug 'preservim/nerdtree'

nnoremap <leader>n :NERDTreeToggle<CR>
nnoremap <leader>N :NERDTreeFind<CR>    " Reveal current file
```

**nvim-tree (Neovim):**
```lua
{
  "nvim-tree/nvim-tree.lua",
  dependencies = { "nvim-tree/nvim-web-devicons" },
  config = function()
    require("nvim-tree").setup()
    vim.keymap.set("n", "<leader>n", "<cmd>NvimTreeToggle<cr>")
  end,
}
```

---

## 5. Git Plugins

### vim-fugitive

The definitive Git plugin for Vim.

```vim
Plug 'tpope/vim-fugitive'
```

| Command | Action |
|---------|--------|
| `:Git` | Open Git status (interactive staging) |
| `:Git diff` | View diff |
| `:Git blame` | Line-by-line blame |
| `:Git log` | View log |
| `:Git commit` | Commit |
| `:Git push` | Push |
| `:Gread` | Revert file to git version |
| `:Gwrite` | Stage current file |

### gitsigns.nvim (Neovim)

Shows git changes in the sign column (gutter).

```lua
{
  "lewis6991/gitsigns.nvim",
  config = function()
    require("gitsigns").setup({
      signs = {
        add = { text = "+" },
        change = { text = "~" },
        delete = { text = "_" },
      },
    })
  end,
}
```

### vim-signify (Vim)

The Vim equivalent of gitsigns:

```vim
Plug 'mhinz/vim-signify'
```

---

## 6. Language Support

### Syntax Highlighting and Indentation

Vim includes basic support for most languages. For enhanced support:

```vim
" Better syntax for many languages
Plug 'sheerun/vim-polyglot'
```

For Neovim, Treesitter provides superior parsing (see [Lesson 14](./14_Neovim_and_Modern_Workflows.md)).

### ALE (Asynchronous Lint Engine)

Linting and fixing for Vim:

```vim
Plug 'dense-analysis/ale'

let g:ale_linters = {
\   'python': ['flake8', 'mypy'],
\   'javascript': ['eslint'],
\}
let g:ale_fixers = {
\   '*': ['remove_trailing_lines', 'trim_whitespace'],
\   'python': ['black', 'isort'],
\   'javascript': ['prettier'],
\}
let g:ale_fix_on_save = 1
```

### CoC.nvim (Conquer of Completion)

Full LSP support for Vim (like VS Code's language features):

```vim
Plug 'neoclide/coc.nvim', {'branch': 'release'}
```

For Neovim, the built-in LSP client is preferred (see [Lesson 14](./14_Neovim_and_Modern_Workflows.md)).

---

## 7. Writing a Simple Plugin

Vim plugins are just Vimscript (or Lua for Neovim) files in specific directories.

### Plugin Directory Structure

```
~/.vim/plugin/          " Auto-loaded scripts
~/.vim/autoload/        " Lazy-loaded functions
~/.vim/ftplugin/        " Filetype-specific scripts
~/.vim/after/plugin/    " Loaded after all plugins
```

### A Simple Plugin: Word Counter

Create `~/.vim/plugin/wordcount.vim`:

```vim
" wordcount.vim — Display word count in status line

function! WordCount()
    let s:old_status = v:statusmsg
    exe "silent normal g\<C-g>"
    if v:statusmsg =~ '--No lines'
        let wordcount = 0
    else
        let wordcount = str2nr(split(v:statusmsg)[11])
    endif
    let v:statusmsg = s:old_status
    return wordcount
endfunction

" Add to status line
set statusline+=\ Words:%{WordCount()}
```

### Custom Commands

```vim
" Create a command to insert the current date
command! InsertDate execute "normal! i" . strftime('%Y-%m-%d')

" Create a command to trim trailing whitespace
command! TrimWhitespace %s/\s\+$//e

" Create a command with arguments
command! -nargs=1 Grep execute 'silent grep!' <q-args> | copen
```

### Custom Functions

```vim
" Toggle between relative and absolute line numbers
function! ToggleLineNumbers()
    if &relativenumber
        set norelativenumber
    else
        set relativenumber
    endif
endfunction
nnoremap <leader>rn :call ToggleLineNumbers()<CR>
```

---

## 8. Plugin Philosophy

### Less is More

Every plugin:
- Adds startup time
- Can introduce bugs
- May conflict with other plugins
- Needs maintenance (updates, compatibility)

### Rules for Choosing Plugins

1. **Learn core Vim first** — Don't install plugins to avoid learning Vim
2. **Solve a real problem** — Install only when you hit a genuine limitation
3. **Check quality signals** — Stars, recent commits, active maintainer
4. **Understand the plugin** — Read the README, know the key mappings
5. **Limit your count** — 10-15 well-chosen plugins beats 50 random ones

### Plugin Audit Questions

Before installing a plugin, ask:
- Can I do this with built-in Vim features?
- Does this replace a workflow I actually use?
- Is it well-maintained?
- Will I use it frequently enough to justify the overhead?

### Recommended Starter Set (10 plugins)

```vim
" Text manipulation
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'tpope/vim-repeat'

" Navigation
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'
Plug 'preservim/nerdtree'

" Git
Plug 'tpope/vim-fugitive'
Plug 'mhinz/vim-signify'

" Language
Plug 'sheerun/vim-polyglot'
Plug 'dense-analysis/ale'
```

---

## 9. Summary

| Category | Vim | Neovim |
|----------|-----|--------|
| Plugin manager | vim-plug | lazy.nvim |
| Fuzzy finder | fzf.vim | telescope.nvim |
| File explorer | NERDTree | nvim-tree.lua |
| Comments | vim-commentary | Comment.nvim |
| Git | vim-fugitive | vim-fugitive + gitsigns |
| Linting | ALE | Built-in LSP |
| Completion | coc.nvim | nvim-cmp + LSP |
| Surround | vim-surround | vim-surround / nvim-surround |

### Getting Started Checklist

1. Install a plugin manager (vim-plug or lazy.nvim)
2. Add vim-surround, vim-commentary, vim-repeat
3. Add a fuzzy finder (fzf or telescope)
4. Add vim-fugitive for Git
5. Add a file explorer if needed
6. Stop here — add more only when you need them

---

## Exercises

### Exercise 1: vim-plug Setup

You are setting up a new Vim installation and want to use vim-plug with three plugins: `tpope/vim-surround`, `tpope/vim-commentary`, and `junegunn/fzf.vim` (which also requires `junegunn/fzf`).

1. Write the complete `.vimrc` section that declares these plugins.
2. After writing the config, what command do you run inside Vim to install them?
3. Later, you decide to remove `junegunn/fzf.vim` and `junegunn/fzf`. After removing those lines from `.vimrc` and reloading, what command cleans up the installed plugin files?

<details>
<summary>Show Answer</summary>

1. The plugin declaration block:

```vim
call plug#begin('~/.vim/plugged')

Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

call plug#end()
```

Note: `junegunn/fzf` must be declared before `junegunn/fzf.vim` because fzf.vim depends on the fzf binary that the first plugin installs via its `do` hook.

2. `:PlugInstall` — vim-plug downloads all declared plugins to `~/.vim/plugged/`.

3. `:PlugClean` — After removing the plugin lines from `.vimrc` and reloading (`:source ~/.vimrc`), `:PlugClean` detects plugins in the plugged directory that are no longer declared and offers to delete them.

</details>

### Exercise 2: vim-surround Operations

For each transformation, write the vim-surround command. The cursor is on the word `hello` in each case:

1. `hello` → `"hello"` (add double quotes around the word)
2. `"hello"` → `'hello'` (change double quotes to single quotes)
3. `'hello'` → `hello` (remove the single quotes)
4. `hello world` (cursor at start of line) → `(hello world)` (wrap the entire line in parentheses, no spaces)

<details>
<summary>Show Answer</summary>

1. `ysiw"` — `ys` (you surround) + `iw` (inner word) + `"` (with double quotes). Cursor must be on or in the word `hello`.

2. `cs"'` — `cs` (change surrounding) + `"` (old) + `'` (new). Works when cursor is anywhere inside the quoted text.

3. `ds'` — `ds` (delete surrounding) + `'` (the surrounding character to remove).

4. `yss)` — `yss` (you surround entire line) + `)` (closing paren, no spaces). Use `(` instead of `)` to get `( hello world )` with spaces: `yss(`.

</details>

### Exercise 3: vim-commentary

You are editing a Python function and need to comment/uncomment code:

```python
def process(data):
    result = transform(data)
    debug_log(result)
    return result
```

1. With cursor on the `debug_log` line, how do you comment out just that line?
2. How do you comment out the entire function body (lines 2–4) using a motion?
3. After commenting, how do you uncomment the same lines again?
4. `vim-repeat` is installed alongside `vim-commentary`. What does this enable?

<details>
<summary>Show Answer</summary>

1. `gcc` — toggle comment on the current line. With cursor on `debug_log(result)`, this adds `# ` at the beginning.

2. Position cursor on `result = transform(data)` (first line of the body), then: `gc2j` — `gc` (toggle comment) + `2j` (motion covering 2 lines down). This comments lines 2, 3, and 4. Alternatively, visually select the three lines with `V2j` then press `gc`.

3. `gc2j` again (or `gcc` per line, or re-select and `gc`) — `gcc`/`gc` toggles: if the lines are already commented, it removes the comments.

4. `vim-repeat` makes the `.` command repeat the last `gcc` or `gc{motion}` action. Without vim-repeat, `.` would only repeat the last built-in Vim action, not plugin actions. With it, you can comment one line, move to another, and press `.` to comment that line too.

</details>

### Exercise 4: Evaluating a Plugin

A colleague suggests installing a plugin that "shows a visual indentation guide" (like VS Code's indent lines). Apply the plugin audit questions from this lesson to evaluate it:

1. Can you do this with built-in Vim features?
2. What quality signals would you check?
3. What is the main cost/risk of adding this plugin?
4. Would you install it? Justify your decision.

<details>
<summary>Show Answer</summary>

This is an open-ended exercise — there is no single correct answer. Here is a model response:

1. **Built-in alternative**: Vim can show `set list` characters for whitespace and uses `foldmethod=indent` for indentation structure, but there is no built-in visual indent guide like VS Code's. So the plugin does provide genuinely new functionality.

2. **Quality signals to check**:
   - GitHub stars (1000+ is generally reliable)
   - Last commit date (within the past 12 months is healthy)
   - Open issues and whether maintainer responds
   - Number of active contributors

3. **Main cost/risk**:
   - Small startup time increase (usually negligible for indent plugins)
   - May not work well with all color schemes or terminal emulators
   - Another plugin to maintain and update

4. **Decision rationale**: Reasonable to install if you regularly work with deeply-nested code (e.g., Python, JavaScript) where tracking indentation level is error-prone. Probably unnecessary if you use `set number` + `set relativenumber` and already navigate by line number. The key principle: install only if it solves a problem you actually experience.

</details>

### Exercise 5: Writing a Custom Command

Write a Vimscript custom command called `Header` that inserts a comment header block above the current line. The header should look like this (for any file type using `#` comments):

```
# ============================================================
# Section: <text argument>
# ============================================================
```

Where `<text argument>` is the argument passed to the command. For example, `:Header Utilities` should insert:

```
# ============================================================
# Section: Utilities
# ============================================================
```

<details>
<summary>Show Answer</summary>

```vim
command! -nargs=1 Header call InsertHeader(<q-args>)

function! InsertHeader(title)
    let separator = "# ============================================================"
    let titleline = "# Section: " . a:title
    call append(line('.') - 1, separator)
    call append(line('.'), titleline)
    call append(line('.') + 1, separator)
endfunction
```

Explanation:
- `command! -nargs=1 Header` defines a command `Header` that takes exactly 1 argument
- `<q-args>` passes the argument as a quoted string to the function
- `append(line('.') - 1, text)` inserts a line before the current line
- `append(line('.'), text)` inserts after the current line (but since we already inserted above, positions shift)
- `a:title` accesses the function argument named `title`

A simpler alternative using `:normal`:
```vim
command! -nargs=1 Header execute "normal! O# ============================================================\n# Section: " . <q-args> . "\n# ============================================================\<Esc>"
```

</details>

---

**Previous**: [Configuration and Vimrc](./12_Configuration_and_Vimrc.md) | **Next**: [Neovim and Modern Workflows](./14_Neovim_and_Modern_Workflows.md)
