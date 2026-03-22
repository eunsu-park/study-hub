# Vim Plugins and Ecosystem — Guide

## Plugin Managers

### vim-plug (Recommended for Vim)

```vim
" ~/.vim/autoload/plug.vim — install first:
" curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
"   https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

call plug#begin('~/.vim/plugged')

" --- Essential Plugins ---
Plug 'tpope/vim-surround'          " Surround text objects: cs'\" ds' ysiw'
Plug 'tpope/vim-commentary'        " Comment: gcc (line), gc (motion/visual)
Plug 'tpope/vim-repeat'            " Make . work with plugin mappings
Plug 'tpope/vim-fugitive'          " Git wrapper: :Git blame, :Gdiff, :Glog

" --- Navigation ---
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'            " Fuzzy finder: :Files, :Rg, :Buffers
Plug 'preservim/nerdtree'          " File tree: :NERDTreeToggle

" --- Editing ---
Plug 'jiangmiao/auto-pairs'        " Auto-close brackets/quotes
Plug 'mg979/vim-visual-multi'      " Multiple cursors

" --- Appearance ---
Plug 'vim-airline/vim-airline'      " Status line
Plug 'morhetz/gruvbox'             " Color scheme

" --- Language Support ---
Plug 'dense-analysis/ale'          " Async lint engine
Plug 'neoclide/coc.nvim', {'branch': 'release'}  " LSP client

call plug#end()

" Commands:
"   :PlugInstall    — install plugins
"   :PlugUpdate     — update plugins
"   :PlugClean      — remove unused plugins
"   :PlugStatus     — check status
```

### lazy.nvim (Recommended for Neovim)

```lua
-- ~/.config/nvim/lua/plugins.lua

-- Bootstrap lazy.nvim
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
  -- Telescope (fuzzy finder)
  {
    "nvim-telescope/telescope.nvim",
    dependencies = { "nvim-lua/plenary.nvim" },
    keys = {
      { "<leader>ff", "<cmd>Telescope find_files<cr>", desc = "Find Files" },
      { "<leader>fg", "<cmd>Telescope live_grep<cr>", desc = "Live Grep" },
      { "<leader>fb", "<cmd>Telescope buffers<cr>", desc = "Buffers" },
    },
  },

  -- Treesitter (syntax highlighting)
  {
    "nvim-treesitter/nvim-treesitter",
    build = ":TSUpdate",
    config = function()
      require("nvim-treesitter.configs").setup({
        ensure_installed = { "python", "javascript", "lua", "vim" },
        highlight = { enable = true },
        indent = { enable = true },
      })
    end,
  },

  -- LSP
  {
    "neovim/nvim-lspconfig",
    dependencies = {
      "williamboman/mason.nvim",
      "williamboman/mason-lspconfig.nvim",
    },
  },

  -- Autocompletion
  {
    "hrsh7th/nvim-cmp",
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
      "hrsh7th/cmp-buffer",
      "L3MON4D3/LuaSnip",
    },
  },

  -- Color scheme
  {
    "catppuccin/nvim",
    name = "catppuccin",
    priority = 1000,
    config = function()
      vim.cmd.colorscheme("catppuccin")
    end,
  },
})
```

---

## Essential Plugins by Category

### 1. Tim Pope's Essentials (must-have)

| Plugin | Purpose | Key Commands |
|--------|---------|-------------|
| `vim-surround` | Add/change/delete surroundings | `cs'"` change `'` to `"`, `ds"` delete `"`, `ysiw)` surround word |
| `vim-commentary` | Toggle comments | `gcc` comment line, `gcap` comment paragraph, `gc` in visual |
| `vim-repeat` | Make `.` work with plugins | Just works transparently |
| `vim-fugitive` | Git integration | `:Git blame`, `:Gdiff`, `:Glog`, `:Git push` |
| `vim-unimpaired` | Paired shortcuts | `]q`/`[q` quickfix, `]b`/`[b` buffers, `]e`/`[e` move lines |

### 2. Navigation

| Plugin | Purpose | Key Commands |
|--------|---------|-------------|
| `fzf.vim` | Fuzzy file/text search | `:Files`, `:Rg`, `:Buffers`, `:Lines` |
| `telescope.nvim` | Neovim fuzzy finder | `<leader>ff` files, `<leader>fg` grep |
| `nerdtree` | File explorer tree | `:NERDTreeToggle`, `o` open, `s` split |
| `oil.nvim` | Neovim file manager | Edit directory like a buffer |

### 3. LSP and Completion

| Plugin | Purpose | Key Commands |
|--------|---------|-------------|
| `coc.nvim` | LSP client for Vim | `gd` go to def, `K` hover, `<leader>rn` rename |
| `nvim-lspconfig` | Native Neovim LSP | `gd`, `gr` references, `<leader>ca` code action |
| `nvim-cmp` | Autocompletion | Tab/Shift-Tab to navigate, Enter to confirm |

### 4. Editing Enhancement

| Plugin | Purpose | Key Commands |
|--------|---------|-------------|
| `auto-pairs` | Auto-close brackets | Automatic on `(`, `[`, `{`, `"`, `'` |
| `vim-visual-multi` | Multiple cursors | `Ctrl-n` select word, `Ctrl-Down` add cursor |
| `vim-abolish` | Smart substitution | `:S/word/replacement/g` (case-preserving) |

---

## Plugin Configuration Patterns

### Key Mappings for Plugins

```vim
" --- fzf.vim ---
nnoremap <leader>f :Files<CR>
nnoremap <leader>g :Rg<CR>
nnoremap <leader>b :Buffers<CR>
nnoremap <leader>l :Lines<CR>

" --- NERDTree ---
nnoremap <leader>n :NERDTreeToggle<CR>
nnoremap <leader>N :NERDTreeFind<CR>

" --- fugitive ---
nnoremap <leader>gs :Git<CR>
nnoremap <leader>gd :Gdiff<CR>
nnoremap <leader>gb :Git blame<CR>
nnoremap <leader>gl :Glog<CR>

" --- coc.nvim ---
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)
nmap <leader>rn <Plug>(coc-rename)
```

### Conditional Plugin Loading

```vim
" Only load heavy plugins when needed
Plug 'fatih/vim-go', { 'for': 'go' }           " Go: load only for .go files
Plug 'rust-lang/rust.vim', { 'for': 'rust' }   " Rust: load only for .rs files
Plug 'preservim/nerdtree', { 'on': 'NERDTreeToggle' }  " Load on first use
```

---

## Choosing Plugins: Decision Guide

```
Need file search?
  ├── Vim → fzf.vim
  └── Neovim → telescope.nvim

Need LSP/completion?
  ├── Vim → coc.nvim
  └── Neovim → nvim-lspconfig + nvim-cmp

Need file explorer?
  ├── Tree view → NERDTree (Vim) or neo-tree (Neovim)
  └── Buffer-based → oil.nvim (Neovim)

Need syntax highlighting?
  ├── Vim → vim-polyglot (regex-based)
  └── Neovim → nvim-treesitter (AST-based, more accurate)

Need statusline?
  ├── Vim → vim-airline or lightline
  └── Neovim → lualine.nvim
```

---

## Minimal Recommended Setup

### For Vim (5 plugins)
```vim
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'junegunn/fzf.vim'
Plug 'tpope/vim-fugitive'
Plug 'dense-analysis/ale'
```

### For Neovim (5 plugins)
```lua
{ "nvim-telescope/telescope.nvim" }
{ "nvim-treesitter/nvim-treesitter" }
{ "neovim/nvim-lspconfig" }
{ "hrsh7th/nvim-cmp" }
{ "lewis6991/gitsigns.nvim" }
```
