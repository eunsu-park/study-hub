-- ============================================================================
-- Neovim init.lua — Basic Lua configuration
-- ============================================================================
-- Place this file at: ~/.config/nvim/init.lua
-- This demonstrates the Lua equivalents of common Vimscript settings
-- and a basic plugin setup with lazy.nvim.
-- ============================================================================

-- ============================================================================
-- OPTIONS
-- ============================================================================

-- Why: vim.opt is the Lua equivalent of :set.
-- vim.opt.number = true  is the same as  :set number

-- Display
vim.opt.number = true              -- Show line numbers
vim.opt.relativenumber = true      -- Relative line numbers
vim.opt.cursorline = true          -- Highlight current line
vim.opt.showmatch = true           -- Highlight matching brackets
vim.opt.signcolumn = "yes"         -- Always show sign column
vim.opt.colorcolumn = "80"         -- Column guide at 80 chars
vim.opt.scrolloff = 8              -- Keep 8 lines visible around cursor
vim.opt.sidescrolloff = 8          -- Keep 8 columns visible
vim.opt.wrap = true                -- Wrap long lines
vim.opt.linebreak = true           -- Wrap at word boundaries
vim.opt.termguicolors = true       -- True color support

-- Search
vim.opt.hlsearch = true            -- Highlight search matches
vim.opt.incsearch = true           -- Show matches while typing
vim.opt.ignorecase = true          -- Case-insensitive search
vim.opt.smartcase = true           -- Case-sensitive if uppercase used

-- Tabs and Indentation
vim.opt.tabstop = 4                -- Tab = 4 spaces
vim.opt.shiftwidth = 4             -- Indent = 4 spaces
vim.opt.softtabstop = 4            -- Tab key inserts 4 spaces
vim.opt.expandtab = true           -- Spaces instead of tabs
vim.opt.autoindent = true          -- Copy indent from current line
vim.opt.smartindent = true         -- Smart indentation

-- File Handling
vim.opt.encoding = "utf-8"
vim.opt.hidden = true              -- Allow unsaved buffer switching
vim.opt.autoread = true            -- Auto-reload changed files
vim.opt.swapfile = false           -- No swap files
vim.opt.backup = false             -- No backup files
vim.opt.undofile = true            -- Persistent undo
vim.opt.undodir = vim.fn.stdpath("data") .. "/undodir"

-- UI Behavior
vim.opt.splitbelow = true          -- New horizontal splits open below
vim.opt.splitright = true          -- New vertical splits open right
vim.opt.mouse = "a"                -- Enable mouse in all modes
vim.opt.updatetime = 300           -- Faster CursorHold events
vim.opt.timeoutlen = 500           -- Key sequence timeout (ms)
vim.opt.clipboard = "unnamedplus"  -- Use system clipboard
vim.opt.wildmode = "list:longest,full"

-- ============================================================================
-- LEADER KEY
-- ============================================================================

-- Why: Set leader before any mappings that use it.
-- Space is the most popular leader key in modern configs.
vim.g.mapleader = " "
vim.g.maplocalleader = ","

-- ============================================================================
-- KEY MAPPINGS
-- ============================================================================

-- Why: vim.keymap.set(mode, lhs, rhs, opts) is the Lua mapping API.
-- The 'desc' field helps with which-key and :map documentation.
local map = vim.keymap.set

-- File operations
map("n", "<C-s>", "<cmd>w<CR>", { desc = "Save file" })
map("i", "<C-s>", "<Esc><cmd>w<CR>a", { desc = "Save file (insert)" })
map("n", "<leader>w", "<cmd>w<CR>", { desc = "Save file" })
map("n", "<leader>q", "<cmd>q<CR>", { desc = "Quit" })

-- Window navigation (Ctrl+hjkl)
map("n", "<C-h>", "<C-w>h", { desc = "Move to left window" })
map("n", "<C-j>", "<C-w>j", { desc = "Move to below window" })
map("n", "<C-k>", "<C-w>k", { desc = "Move to above window" })
map("n", "<C-l>", "<C-w>l", { desc = "Move to right window" })

-- Buffer navigation
map("n", "[b", "<cmd>bprev<CR>", { desc = "Previous buffer" })
map("n", "]b", "<cmd>bnext<CR>", { desc = "Next buffer" })
map("n", "<leader><leader>", "<C-^>", { desc = "Toggle last buffer" })

-- Search
map("n", "<Esc>", "<cmd>nohlsearch<CR>", { desc = "Clear search highlight" })
map("n", "n", "nzzzv", { desc = "Next match (centered)" })
map("n", "N", "Nzzzv", { desc = "Previous match (centered)" })

-- Scrolling (centered)
map("n", "<C-d>", "<C-d>zz", { desc = "Scroll down (centered)" })
map("n", "<C-u>", "<C-u>zz", { desc = "Scroll up (centered)" })

-- Line movement (Alt+j/k)
map("n", "<A-j>", "<cmd>m .+1<CR>==", { desc = "Move line down" })
map("n", "<A-k>", "<cmd>m .-2<CR>==", { desc = "Move line up" })
map("v", "<A-j>", ":m '>+1<CR>gv=gv", { desc = "Move selection down" })
map("v", "<A-k>", ":m '<-2<CR>gv=gv", { desc = "Move selection up" })

-- Visual indentation (keep selection)
map("v", "<", "<gv", { desc = "Indent left (keep selection)" })
map("v", ">", ">gv", { desc = "Indent right (keep selection)" })

-- Y consistency (yank to end of line)
map("n", "Y", "y$", { desc = "Yank to end of line" })

-- Splits
map("n", "<leader>v", "<cmd>vsplit<CR>", { desc = "Vertical split" })
map("n", "<leader>s", "<cmd>split<CR>", { desc = "Horizontal split" })

-- Quick config edit
map("n", "<leader>ev", "<cmd>edit $MYVIMRC<CR>", { desc = "Edit init.lua" })

-- ============================================================================
-- AUTOCOMMANDS
-- ============================================================================

-- Why: vim.api.nvim_create_augroup and nvim_create_autocmd are the Lua API
-- for autocommands. The { clear = true } prevents duplicates on reload.
local augroup = vim.api.nvim_create_augroup("UserConfig", { clear = true })

-- Highlight yanked text briefly
vim.api.nvim_create_autocmd("TextYankPost", {
    group = augroup,
    desc = "Highlight on yank",
    callback = function()
        vim.highlight.on_yank({ timeout = 200 })
    end,
})

-- Remove trailing whitespace on save
vim.api.nvim_create_autocmd("BufWritePre", {
    group = augroup,
    desc = "Remove trailing whitespace",
    pattern = "*",
    callback = function()
        local save = vim.fn.winsaveview()
        vim.cmd([[keeppatterns %s/\s\+$//e]])
        vim.fn.winrestview(save)
    end,
})

-- Return to last edit position
vim.api.nvim_create_autocmd("BufReadPost", {
    group = augroup,
    desc = "Return to last edit position",
    callback = function()
        local mark = vim.api.nvim_buf_get_mark(0, '"')
        local line_count = vim.api.nvim_buf_line_count(0)
        if mark[1] > 0 and mark[1] <= line_count then
            pcall(vim.api.nvim_win_set_cursor, 0, mark)
        end
    end,
})

-- Auto-resize splits on window resize
vim.api.nvim_create_autocmd("VimResized", {
    group = augroup,
    desc = "Auto-resize splits",
    command = "wincmd =",
})

-- ============================================================================
-- FILETYPE SETTINGS
-- ============================================================================

local ft_group = vim.api.nvim_create_augroup("FileTypeSettings", { clear = true })

-- Why: Different languages have different indentation conventions.
local filetype_settings = {
    python     = { tabstop = 4, shiftwidth = 4, expandtab = true },
    javascript = { tabstop = 2, shiftwidth = 2, expandtab = true },
    typescript = { tabstop = 2, shiftwidth = 2, expandtab = true },
    html       = { tabstop = 2, shiftwidth = 2, expandtab = true },
    css        = { tabstop = 2, shiftwidth = 2, expandtab = true },
    lua        = { tabstop = 2, shiftwidth = 2, expandtab = true },
    go         = { tabstop = 4, shiftwidth = 4, expandtab = false },
    make       = { expandtab = false },
    yaml       = { tabstop = 2, shiftwidth = 2, expandtab = true },
    markdown   = { wrap = true, linebreak = true, spell = true },
}

for ft, settings in pairs(filetype_settings) do
    vim.api.nvim_create_autocmd("FileType", {
        group = ft_group,
        pattern = ft,
        callback = function()
            for k, v in pairs(settings) do
                vim.opt_local[k] = v
            end
        end,
    })
end

-- ============================================================================
-- PLUGIN SETUP (lazy.nvim)
-- ============================================================================

-- Why: Bootstrap lazy.nvim if not installed.
-- This block downloads lazy.nvim on first run.
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
    vim.fn.system({
        "git", "clone", "--filter=blob:none",
        "https://github.com/folke/lazy.nvim.git",
        "--branch=stable",
        lazypath,
    })
end
vim.opt.rtp:prepend(lazypath)

-- Why: require("lazy").setup() loads all plugins.
-- Each plugin spec is a table with the GitHub repo and optional config.
require("lazy").setup({

    -- Color scheme
    {
        "catppuccin/nvim",
        name = "catppuccin",
        priority = 1000,  -- Load before other plugins
        config = function()
            vim.cmd.colorscheme("catppuccin")
        end,
    },

    -- Text manipulation
    { "tpope/vim-surround" },
    {
        "numToStr/Comment.nvim",
        config = true,  -- Calls require("Comment").setup()
    },
    { "tpope/vim-repeat" },

    -- Fuzzy finder
    {
        "nvim-telescope/telescope.nvim",
        dependencies = { "nvim-lua/plenary.nvim" },
        keys = {
            { "<leader>f", "<cmd>Telescope find_files<cr>", desc = "Find files" },
            { "<leader>g", "<cmd>Telescope live_grep<cr>", desc = "Live grep" },
            { "<leader>b", "<cmd>Telescope buffers<cr>", desc = "Buffers" },
            { "<leader>h", "<cmd>Telescope help_tags<cr>", desc = "Help tags" },
        },
    },

    -- File explorer
    {
        "nvim-tree/nvim-tree.lua",
        dependencies = { "nvim-tree/nvim-web-devicons" },
        keys = {
            { "<leader>n", "<cmd>NvimTreeToggle<cr>", desc = "Toggle file tree" },
        },
        config = function()
            require("nvim-tree").setup({
                view = { width = 30 },
            })
        end,
    },

    -- Git
    { "tpope/vim-fugitive" },
    {
        "lewis6991/gitsigns.nvim",
        config = function()
            require("gitsigns").setup({
                signs = {
                    add = { text = "│" },
                    change = { text = "│" },
                    delete = { text = "_" },
                },
            })
        end,
    },

    -- Status line
    {
        "nvim-lualine/lualine.nvim",
        dependencies = { "nvim-tree/nvim-web-devicons" },
        config = function()
            require("lualine").setup({
                options = {
                    theme = "catppuccin",
                    section_separators = "",
                    component_separators = "│",
                },
            })
        end,
    },

})

-- ============================================================================
-- End of init.lua
-- For LSP, Treesitter, and DAP setup, see Lesson 14 examples.
-- ============================================================================
