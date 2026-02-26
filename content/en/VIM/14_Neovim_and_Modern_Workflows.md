# Neovim and Modern Workflows

**Previous**: [Plugins and Ecosystem](./13_Plugins_and_Ecosystem.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how Neovim differs from Vim and when to choose each
2. Write a basic Lua configuration (`init.lua`) for Neovim
3. Set up LSP (Language Server Protocol) for IDE-like features
4. Configure Treesitter for advanced syntax highlighting and text objects
5. Build an IDE-level workflow with terminal integration, Git, and debugging

---

Neovim started as a fork of Vim in 2014, aiming to modernize the codebase while keeping full compatibility. Today, Neovim has become the preferred choice for developers who want a modal editor with IDE-level features. Everything you've learned in Lessons 1–13 applies to Neovim — this lesson covers what Neovim adds on top.

## Table of Contents

1. [Neovim vs Vim](#1-neovim-vs-vim)
2. [Lua Configuration Basics](#2-lua-configuration-basics)
3. [LSP — Language Server Protocol](#3-lsp--language-server-protocol)
4. [Treesitter](#4-treesitter)
5. [Terminal Integration](#5-terminal-integration)
6. [Git Integration](#6-git-integration)
7. [DAP — Debug Adapter Protocol](#7-dap--debug-adapter-protocol)
8. [Building a Complete IDE Setup](#8-building-a-complete-ide-setup)
9. [Summary](#9-summary)

---

## 1. Neovim vs Vim

### Key Differences

| Feature | Vim | Neovim |
|---------|-----|--------|
| Scripting | Vimscript (+ Vim9 script) | Vimscript + **Lua** (first-class) |
| LSP | Via plugins (coc.nvim) | **Built-in** LSP client |
| Treesitter | Not supported | **Built-in** |
| Terminal | `:terminal` (Vim 8+) | Enhanced `:terminal` |
| Defaults | Minimal | Sensible defaults |
| Config file | `~/.vimrc` | `~/.config/nvim/init.lua` |
| API | Limited | Rich Lua API |
| UI | Terminal only (gVim for GUI) | External UIs (Neovide, etc.) |
| Community | Stable, slower development | Active, rapid development |

### When to Choose Vim

- Server environments where only `vi`/`vim` is available
- You prefer stability over cutting-edge features
- Your existing `.vimrc` works perfectly
- You don't need IDE features (LSP, Treesitter)

### When to Choose Neovim

- You want IDE-level features (autocomplete, go-to-definition, linting)
- You prefer Lua over Vimscript
- You want the latest plugin ecosystem
- You're starting fresh (no legacy config)

---

## 2. Lua Configuration Basics

### Config File Location

```
~/.config/nvim/init.lua       " Main config file
~/.config/nvim/lua/           " Lua modules directory
```

### Vimscript to Lua Translation

```lua
-- Setting options (set number → vim.opt.number)
vim.opt.number = true
vim.opt.relativenumber = true
vim.opt.tabstop = 4
vim.opt.shiftwidth = 4
vim.opt.expandtab = true
vim.opt.hlsearch = true
vim.opt.ignorecase = true
vim.opt.smartcase = true
vim.opt.termguicolors = true
vim.opt.scrolloff = 8
vim.opt.signcolumn = "yes"
vim.opt.clipboard = "unnamedplus"

-- Global variables (let g:mapleader = " " → vim.g.mapleader)
vim.g.mapleader = " "
vim.g.maplocalleader = ","

-- Key mappings (nnoremap → vim.keymap.set)
vim.keymap.set("n", "<C-h>", "<C-w>h", { desc = "Move to left window" })
vim.keymap.set("n", "<C-j>", "<C-w>j", { desc = "Move to below window" })
vim.keymap.set("n", "<C-k>", "<C-w>k", { desc = "Move to above window" })
vim.keymap.set("n", "<C-l>", "<C-w>l", { desc = "Move to right window" })

-- Clear search highlighting
vim.keymap.set("n", "<Esc>", "<cmd>nohlsearch<CR>")
```

### Modular Configuration

Split your config into modules:

```
~/.config/nvim/
├── init.lua              " Entry point
└── lua/
    ├── options.lua       " vim.opt settings
    ├── keymaps.lua       " Key mappings
    └── plugins/
        ├── init.lua      " Plugin manager setup
        ├── lsp.lua       " LSP configuration
        └── treesitter.lua
```

In `init.lua`:
```lua
require("options")
require("keymaps")
require("plugins")
```

### Autocommands in Lua

```lua
-- Create autocommand group
local augroup = vim.api.nvim_create_augroup("MyGroup", { clear = true })

-- Remove trailing whitespace on save
vim.api.nvim_create_autocmd("BufWritePre", {
  group = augroup,
  pattern = "*",
  callback = function()
    vim.cmd([[%s/\s\+$//e]])
  end,
})

-- Highlight yanked text
vim.api.nvim_create_autocmd("TextYankPost", {
  group = augroup,
  callback = function()
    vim.highlight.on_yank({ timeout = 200 })
  end,
})
```

---

## 3. LSP — Language Server Protocol

LSP gives Neovim IDE features: autocompletion, go-to-definition, hover documentation, rename, diagnostics, and more.

### How LSP Works

```
┌─────────────┐         ┌──────────────────┐
│   Neovim    │◀───────▶│  Language Server  │
│  (client)   │  JSON   │  (pyright, tsserver, │
│             │   RPC   │   gopls, clangd...)  │
└─────────────┘         └──────────────────────┘
```

Neovim communicates with external language servers that understand your code.

### Setup with mason.nvim + nvim-lspconfig

```lua
-- Plugin list (in lazy.nvim)
{
  "williamboman/mason.nvim",          -- Install language servers
  "williamboman/mason-lspconfig.nvim", -- Bridge mason ↔ lspconfig
  "neovim/nvim-lspconfig",            -- Configure LSP clients
}
```

```lua
-- Configuration
require("mason").setup()
require("mason-lspconfig").setup({
  ensure_installed = {
    "pyright",      -- Python
    "ts_ls",        -- TypeScript
    "lua_ls",       -- Lua
    "clangd",       -- C/C++
    "gopls",        -- Go
  },
})

-- Configure each server
local lspconfig = require("lspconfig")

lspconfig.pyright.setup({})
lspconfig.ts_ls.setup({})
lspconfig.lua_ls.setup({
  settings = {
    Lua = {
      diagnostics = { globals = { "vim" } },
    },
  },
})
```

### LSP Keybindings

```lua
vim.api.nvim_create_autocmd("LspAttach", {
  callback = function(event)
    local opts = { buffer = event.buf }

    vim.keymap.set("n", "gd", vim.lsp.buf.definition, opts)
    vim.keymap.set("n", "gD", vim.lsp.buf.declaration, opts)
    vim.keymap.set("n", "gr", vim.lsp.buf.references, opts)
    vim.keymap.set("n", "gi", vim.lsp.buf.implementation, opts)
    vim.keymap.set("n", "K", vim.lsp.buf.hover, opts)
    vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename, opts)
    vim.keymap.set("n", "<leader>ca", vim.lsp.buf.code_action, opts)
    vim.keymap.set("n", "[d", vim.diagnostic.goto_prev, opts)
    vim.keymap.set("n", "]d", vim.diagnostic.goto_next, opts)
  end,
})
```

### Autocompletion with nvim-cmp

```lua
{
  "hrsh7th/nvim-cmp",
  dependencies = {
    "hrsh7th/cmp-nvim-lsp",    -- LSP source
    "hrsh7th/cmp-buffer",       -- Buffer words
    "hrsh7th/cmp-path",         -- File paths
    "L3MON4D3/LuaSnip",        -- Snippet engine
    "saadparwaiz1/cmp_luasnip", -- Snippet source
  },
}
```

```lua
local cmp = require("cmp")
local luasnip = require("luasnip")

cmp.setup({
  snippet = {
    expand = function(args)
      luasnip.lsp_expand(args.body)
    end,
  },
  mapping = cmp.mapping.preset.insert({
    ["<C-Space>"] = cmp.mapping.complete(),
    ["<CR>"] = cmp.mapping.confirm({ select = true }),
    ["<Tab>"] = cmp.mapping(function(fallback)
      if cmp.visible() then
        cmp.select_next_item()
      elseif luasnip.expand_or_jumpable() then
        luasnip.expand_or_jump()
      else
        fallback()
      end
    end, { "i", "s" }),
  }),
  sources = cmp.config.sources({
    { name = "nvim_lsp" },
    { name = "luasnip" },
    { name = "buffer" },
    { name = "path" },
  }),
})
```

---

## 4. Treesitter

Treesitter provides accurate syntax highlighting, code folding, and text objects by building a real parse tree of your code.

### Setup

```lua
{
  "nvim-treesitter/nvim-treesitter",
  build = ":TSUpdate",
  config = function()
    require("nvim-treesitter.configs").setup({
      ensure_installed = {
        "python", "javascript", "typescript", "lua",
        "c", "cpp", "go", "rust", "html", "css",
        "json", "yaml", "markdown", "bash", "vim",
      },
      highlight = { enable = true },
      indent = { enable = true },
      incremental_selection = {
        enable = true,
        keymaps = {
          init_selection = "<C-space>",
          node_incremental = "<C-space>",
          scope_incremental = false,
          node_decremental = "<bs>",
        },
      },
    })
  end,
}
```

### Treesitter Text Objects

```lua
{
  "nvim-treesitter/nvim-treesitter-textobjects",
  dependencies = { "nvim-treesitter/nvim-treesitter" },
  config = function()
    require("nvim-treesitter.configs").setup({
      textobjects = {
        select = {
          enable = true,
          lookahead = true,
          keymaps = {
            ["af"] = "@function.outer",  -- Select around function
            ["if"] = "@function.inner",  -- Select inside function
            ["ac"] = "@class.outer",
            ["ic"] = "@class.inner",
            ["aa"] = "@parameter.outer",
            ["ia"] = "@parameter.inner",
          },
        },
        move = {
          enable = true,
          goto_next_start = {
            ["]f"] = "@function.outer",
            ["]c"] = "@class.outer",
          },
          goto_previous_start = {
            ["[f"] = "@function.outer",
            ["[c"] = "@class.outer",
          },
        },
      },
    })
  end,
}
```

With these text objects:
- `daf` — Delete a function (the entire function!)
- `vif` — Select inside a function body
- `daa` — Delete a parameter
- `]f` / `[f` — Jump between functions

---

## 5. Terminal Integration

Neovim's built-in terminal lets you run shell commands without leaving the editor.

### Basic Terminal

```vim
:terminal           " Open terminal in current window
:split | terminal   " Open terminal in horizontal split
:vsplit | terminal  " Open terminal in vertical split
```

### Terminal Mode

In terminal mode, you interact with the shell. To return to Normal mode:

| Key | Action |
|-----|--------|
| `Ctrl-\` `Ctrl-n` | Exit terminal mode → Normal mode |
| `i` or `a` | Enter terminal mode (from Normal mode in terminal buffer) |

### toggleterm.nvim

A popular plugin for a better terminal experience:

```lua
{
  "akinsho/toggleterm.nvim",
  config = function()
    require("toggleterm").setup({
      open_mapping = [[<C-\>]],
      direction = "float",      -- or "horizontal", "vertical"
      float_opts = { border = "curved" },
    })
  end,
}
```

### Running Code

```lua
-- Run current file based on filetype
vim.keymap.set("n", "<leader>r", function()
  local ft = vim.bo.filetype
  local runners = {
    python = "python3 %",
    javascript = "node %",
    typescript = "npx tsx %",
    go = "go run %",
    rust = "cargo run",
    c = "gcc % -o /tmp/a.out && /tmp/a.out",
  }
  local cmd = runners[ft]
  if cmd then
    vim.cmd("split | terminal " .. vim.fn.expand(cmd))
  end
end, { desc = "Run current file" })
```

---

## 6. Git Integration

### vim-fugitive (Works in Both Vim and Neovim)

The essential Git plugin (see [Lesson 13](./13_Plugins_and_Ecosystem.md)).

### gitsigns.nvim (Neovim)

Shows line-by-line git changes and provides hunk operations:

```lua
{
  "lewis6991/gitsigns.nvim",
  config = function()
    require("gitsigns").setup({
      on_attach = function(bufnr)
        local gs = require("gitsigns")
        local opts = { buffer = bufnr }

        vim.keymap.set("n", "]h", gs.next_hunk, opts)        -- Next change
        vim.keymap.set("n", "[h", gs.prev_hunk, opts)        -- Previous change
        vim.keymap.set("n", "<leader>hs", gs.stage_hunk, opts) -- Stage hunk
        vim.keymap.set("n", "<leader>hr", gs.reset_hunk, opts) -- Reset hunk
        vim.keymap.set("n", "<leader>hp", gs.preview_hunk, opts) -- Preview
        vim.keymap.set("n", "<leader>hb", gs.blame_line, opts)  -- Blame
      end,
    })
  end,
}
```

---

## 7. DAP — Debug Adapter Protocol

Neovim supports debugging through the Debug Adapter Protocol.

```lua
{
  "mfussenegger/nvim-dap",
  dependencies = {
    "rcarriga/nvim-dap-ui",
    "nvim-neotest/nvim-nio",
  },
  config = function()
    local dap = require("dap")
    local dapui = require("dapui")

    dapui.setup()

    -- Auto-open/close UI
    dap.listeners.after.event_initialized["dapui_config"] = dapui.open
    dap.listeners.before.event_terminated["dapui_config"] = dapui.close

    -- Keymaps
    vim.keymap.set("n", "<F5>", dap.continue)
    vim.keymap.set("n", "<F10>", dap.step_over)
    vim.keymap.set("n", "<F11>", dap.step_into)
    vim.keymap.set("n", "<F12>", dap.step_out)
    vim.keymap.set("n", "<leader>db", dap.toggle_breakpoint)
  end,
}
```

### Python Debugging

```lua
-- Install debugpy: pip install debugpy
dap.adapters.python = {
  type = "executable",
  command = "python",
  args = { "-m", "debugpy.adapter" },
}

dap.configurations.python = {
  {
    type = "python",
    request = "launch",
    name = "Launch file",
    program = "${file}",
    pythonPath = function()
      return "/usr/bin/python3"
    end,
  },
}
```

---

## 8. Building a Complete IDE Setup

### The Full Stack

```
┌─────────────────────────────────────┐
│              Neovim                  │
│  ┌─────────────────────────────┐    │
│  │  Lazy.nvim (Plugin Manager) │    │
│  └─────────────────────────────┘    │
│  ┌───────────┐ ┌───────────────┐    │
│  │ Treesitter│ │ nvim-lspconfig│    │
│  │ (parsing) │ │   (LSP)      │    │
│  └───────────┘ └───────────────┘    │
│  ┌───────────┐ ┌───────────────┐    │
│  │ nvim-cmp  │ │  Telescope    │    │
│  │(complete) │ │(fuzzy find)   │    │
│  └───────────┘ └───────────────┘    │
│  ┌───────────┐ ┌───────────────┐    │
│  │ gitsigns  │ │  nvim-dap     │    │
│  │  (git)    │ │  (debug)      │    │
│  └───────────┘ └───────────────┘    │
└─────────────────────────────────────┘
```

### Recommended Complete Setup

See `examples/VIM/09_init_lua.lua` for a complete, commented Neovim configuration.

### Distribution Alternatives

If you want a pre-configured setup:

| Distribution | Description |
|-------------|-------------|
| **LazyVim** | Feature-rich, well-maintained |
| **NvChad** | Beautiful defaults |
| **AstroNvim** | Community-driven |
| **kickstart.nvim** | Minimal starting point (recommended for learning) |

These provide a starting configuration you can customize. However, building your own config teaches you more about how everything works.

---

## 9. Summary

| Feature | Key Components |
|---------|---------------|
| Config | `init.lua`, `vim.opt`, `vim.keymap.set`, `vim.api` |
| LSP | mason.nvim + nvim-lspconfig + nvim-cmp |
| Treesitter | nvim-treesitter + textobjects |
| Fuzzy find | telescope.nvim |
| Git | vim-fugitive + gitsigns.nvim |
| Terminal | Built-in `:terminal` or toggleterm.nvim |
| Debug | nvim-dap + nvim-dap-ui |
| File explorer | nvim-tree.lua |

### The Neovim Learning Path

1. **Master core Vim** (Lessons 1–12) — This is your foundation
2. **Switch to Neovim** — Your Vim knowledge transfers directly
3. **Learn Lua basics** — Enough to write `init.lua`
4. **Add LSP** — The biggest quality-of-life improvement
5. **Add Treesitter** — Better highlighting and text objects
6. **Add telescope** — Fuzzy finding everything
7. **Customize incrementally** — Add plugins as you identify needs

---

## Exercises

### Exercise 1: Vim vs Neovim Decision

For each scenario, decide whether classic Vim or Neovim is the better choice, and explain why:

1. You are administering a remote Linux server and need to edit a config file in `/etc/`. The server has only `vim` installed.
2. You are a Python developer on your local machine and want go-to-definition, inline error diagnostics, and autocompletion.
3. You have a 600-line `.vimrc` that took two years to build and works perfectly.
4. You are starting Vim for the first time and have no existing configuration.

<details>
<summary>Show Answer</summary>

1. **Vim** — There is no choice here: only `vim` is available. This is the most common real-world scenario where Vim knowledge is essential. Everything from Lessons 1–12 applies directly.

2. **Neovim** — These are exactly the IDE features that Neovim provides natively through its built-in LSP client. In Vim, you would need the heavyweight `coc.nvim` plugin to achieve the same, which is more complex to set up and maintain.

3. **Either, but Vim is practical** — Your existing `.vimrc` works in both Vim and Neovim (Neovim has backward compatibility). However, if it works perfectly and you have no strong reason to switch, staying with Vim is the pragmatic choice. You can migrate to Neovim gradually.

4. **Neovim** — Starting fresh with no legacy config, Neovim is the better choice. It has sensible defaults, a richer ecosystem, and Lua as a first-class configuration language. Starting with `init.lua` and modern plugins from day one sets you up for a better long-term experience.

</details>

### Exercise 2: Vimscript to Lua Translation

Translate each Vimscript line to its Lua equivalent for Neovim's `init.lua`:

1. `set number`
2. `set tabstop=2`
3. `let g:mapleader = ","`
4. `nnoremap <leader>w :w<CR>`
5. `set clipboard=unnamedplus`

<details>
<summary>Show Answer</summary>

1. `vim.opt.number = true`

2. `vim.opt.tabstop = 2`

3. `vim.g.mapleader = ","`

4. `vim.keymap.set("n", "<leader>w", "<cmd>w<CR>")`
   — The `<cmd>...<CR>` form is preferred in Lua mappings over `:...<CR>` because `<cmd>` doesn't enter and leave Command-line mode.

5. `vim.opt.clipboard = "unnamedplus"`

Summary of the mapping pattern:
- `set {option}` → `vim.opt.{option} = true`
- `set {option}={value}` → `vim.opt.{option} = value`
- `let g:{var} = value` → `vim.g.{var} = value`
- `nnoremap {lhs} {rhs}` → `vim.keymap.set("n", "{lhs}", "{rhs}")`

</details>

### Exercise 3: LSP Workflow

Explain the role of each component in the LSP setup chain:

```
mason.nvim → mason-lspconfig.nvim → nvim-lspconfig → nvim-cmp
```

1. What does `mason.nvim` do?
2. What does `mason-lspconfig.nvim` do?
3. What does `nvim-lspconfig` do?
4. What does `nvim-cmp` do, and why is it a separate plugin from the LSP client?

<details>
<summary>Show Answer</summary>

1. **mason.nvim** — A package manager for development tools. It downloads and installs language servers (like `pyright`, `ts_ls`, `clangd`), linters, and formatters. Think of it as `apt` or `brew` but for editor tools. It stores them in a Neovim-managed directory.

2. **mason-lspconfig.nvim** — A bridge between mason and lspconfig. Without it, you would have to manually ensure the right server name is used in both tools (they sometimes use different names). It also provides `ensure_installed` to automatically install listed servers when Neovim starts.

3. **nvim-lspconfig** — Configures Neovim's built-in LSP client to communicate with specific language servers. It provides sensible default settings for each server and a simple API (`lspconfig.pyright.setup({})`) to activate them.

4. **nvim-cmp** — A completion engine that shows a menu of suggestions. The LSP client receives raw completion data from language servers, but it doesn't have a UI to display them. nvim-cmp provides the visual completion menu and handles user interaction (accept, navigate items). It is separate because you might want to show completions from multiple sources (LSP, buffer words, file paths, snippets) in the same menu — nvim-cmp aggregates all of these.

</details>

### Exercise 4: Treesitter Text Objects

With the Treesitter textobjects plugin configured as shown in this lesson, describe what each command does when editing a Python file:

```python
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
```

1. Cursor is on the `return a + b` line. You press `daf`.
2. Cursor is on the `def multiply` line. You press `vif`.
3. Cursor is on `a` inside `def add(self, a, b)`. You press `daa`.
4. Cursor is anywhere inside `add`. You press `]f`.

<details>
<summary>Show Answer</summary>

1. **`daf`** (delete around function) — Deletes the entire `add` method including its `def` line, body, and any surrounding blank lines. The file would be left with only the `multiply` method inside `Calculator`.

2. **`vif`** (visual select inside function) — Visually selects the body of `multiply`, i.e., just `return a * b` (the lines inside the function, but not the `def` line itself).

3. **`daa`** (delete around parameter) — Deletes the parameter `a` from the function signature, including the comma separator. The function signature becomes `def add(self, b):`.

4. **`]f`** (jump to next function) — Moves the cursor to the start of the next function definition, which is `def multiply(self, a, b):`.

</details>

### Exercise 5: IDE Feature Comparison

You are describing Neovim's IDE features to a colleague who uses VS Code. Match each VS Code feature to its Neovim equivalent (plugin/command):

| VS Code Feature | Neovim Equivalent |
|----------------|-------------------|
| Go to Definition (F12) | ? |
| Peek at hover docs (Ctrl+K, Ctrl+I) | ? |
| Rename symbol (F2) | ? |
| Show all references | ? |
| Next diagnostic error | ? |
| Open integrated terminal (Ctrl+`) | ? |

<details>
<summary>Show Answer</summary>

| VS Code Feature | Neovim Equivalent |
|----------------|-------------------|
| Go to Definition (F12) | `gd` — `vim.lsp.buf.definition` |
| Peek at hover docs (Ctrl+K, Ctrl+I) | `K` — `vim.lsp.buf.hover` |
| Rename symbol (F2) | `<leader>rn` — `vim.lsp.buf.rename` |
| Show all references | `gr` — `vim.lsp.buf.references` |
| Next diagnostic error | `]d` — `vim.diagnostic.goto_next` |
| Open integrated terminal (Ctrl+`) | `:terminal` or `<C-\>` with toggleterm.nvim |

All of these are provided by Neovim's built-in LSP client with the keybindings configured in the `LspAttach` autocommand shown in this lesson. The exact key bindings shown above follow the conventions from the lesson — your own config may use different keys.

</details>

---

**Previous**: [Plugins and Ecosystem](./13_Plugins_and_Ecosystem.md)
