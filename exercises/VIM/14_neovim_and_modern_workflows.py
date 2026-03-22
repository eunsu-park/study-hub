"""
Exercises for Lesson 14: Neovim and Modern Workflows
Topic: VIM

Solutions to practice problems from the lesson.
Since VIM exercises are command-based, solutions show the commands
and explain what they do.
"""


# === Exercise 1: Vim vs Neovim Decision ===
# Problem: For each scenario, choose Vim or Neovim and explain why.

def exercise_1():
    """Solution: Vim vs Neovim decision for each scenario."""
    scenarios = [
        ("1. Remote Linux server, only vim installed, need to edit /etc/ config",
         "Vim",
         "No choice -- only vim is available. This is the most common scenario\n"
         "    where core Vim skills (Lessons 1-12) are essential."),
        ("2. Python developer wanting go-to-definition, diagnostics, autocomplete",
         "Neovim",
         "These are exactly the IDE features Neovim provides natively through\n"
         "    its built-in LSP client. Vim would need the heavy coc.nvim plugin."),
        ("3. Existing 600-line .vimrc that works perfectly",
         "Either, but Vim is practical",
         "Your .vimrc works in both (backward compatibility). If everything works\n"
         "    and there is no strong reason to switch, staying with Vim is pragmatic.\n"
         "    You can migrate gradually if desired."),
        ("4. Starting Vim for the first time, no existing config",
         "Neovim",
         "Starting fresh, Neovim is better: sensible defaults, richer ecosystem,\n"
         "    Lua as a first-class config language. init.lua + modern plugins from\n"
         "    day one sets up a better long-term experience."),
    ]
    # Why: The choice depends on context. Vim is universally available.
    # Neovim is the better choice for modern development when starting fresh.
    for scenario, choice, explanation in scenarios:
        print(f"  {scenario}")
        print(f"    Choice: {choice}")
        print(f"    Why:    {explanation}\n")


# === Exercise 2: Vimscript to Lua Translation ===
# Problem: Translate each Vimscript line to Lua for init.lua.

def exercise_2():
    """Solution: Vimscript to Lua translation table."""
    translations = [
        ("set number",
         "vim.opt.number = true",
         "Boolean options use true/false in Lua."),
        ("set tabstop=2",
         "vim.opt.tabstop = 2",
         "Numeric options use = value."),
        ('let g:mapleader = ","',
         'vim.g.mapleader = ","',
         "Global variables use vim.g namespace."),
        ('nnoremap <leader>w :w<CR>',
         'vim.keymap.set("n", "<leader>w", "<cmd>w<CR>")',
         '<cmd>...<CR> is preferred over :...<CR> in Lua (avoids mode transitions).'),
        ("set clipboard=unnamedplus",
         'vim.opt.clipboard = "unnamedplus"',
         "String options use quoted strings."),
    ]
    # Why: Lua is Neovim's first-class scripting language, offering better
    # performance and a cleaner API than Vimscript.
    print(f"  {'Vimscript':<40s} {'Lua'}")
    print(f"  {'-' * 40} {'-' * 50}")
    for vimscript, lua, note in translations:
        print(f"  {vimscript:<40s} {lua}")
        print(f"  {'':40s} Note: {note}\n")

    print("  Pattern summary:")
    print("    set {option}         -> vim.opt.{option} = true")
    print("    set {option}={value} -> vim.opt.{option} = value")
    print("    let g:{var} = value  -> vim.g.{var} = value")
    print('    nnoremap {lhs} {rhs} -> vim.keymap.set("n", "{lhs}", "{rhs}")')


# === Exercise 3: LSP Workflow ===
# Problem: Explain the role of each component in the LSP setup chain.

def exercise_3():
    """Solution: LSP component chain explanation."""
    print("  Chain: mason.nvim -> mason-lspconfig.nvim -> nvim-lspconfig -> nvim-cmp\n")

    components = [
        ("1. mason.nvim",
         "Package manager for development tools.",
         "Downloads and installs language servers (pyright, ts_ls, clangd),\n"
         "    linters, and formatters. Think of it as apt/brew for editor tools."),
        ("2. mason-lspconfig.nvim",
         "Bridge between mason and lspconfig.",
         "Ensures the right server names are used in both tools (they sometimes\n"
         "    differ). Provides ensure_installed to auto-install listed servers."),
        ("3. nvim-lspconfig",
         "Configures Neovim's built-in LSP client.",
         "Provides sensible defaults for each server and a simple API:\n"
         "    lspconfig.pyright.setup({}) to activate them."),
        ("4. nvim-cmp",
         "Completion engine with visual menu.",
         "The LSP client gets raw completion data but has no UI to display it.\n"
         "    nvim-cmp provides the visual menu and aggregates multiple sources\n"
         "    (LSP, buffer words, file paths, snippets) in one place."),
    ]
    # Why: Each component handles a specific concern in the LSP pipeline:
    # installation, configuration, client communication, and UI presentation.
    for component, role, detail in components:
        print(f"  {component}")
        print(f"    Role:   {role}")
        print(f"    Detail: {detail}\n")


# === Exercise 4: Treesitter Text Objects ===
# Problem: Describe what Treesitter textobject commands do in Python code.

def exercise_4():
    """Solution: Treesitter text object commands on Python code."""
    print("  Code:")
    print("    class Calculator:")
    print("        def add(self, a, b):")
    print("            return a + b")
    print()
    print("        def multiply(self, a, b):")
    print("            return a * b\n")

    commands = [
        ("1. daf (cursor on 'return a + b')",
         "Delete around function",
         "Deletes the entire 'add' method including its def line, body,\n"
         "    and surrounding blank lines. Only 'multiply' remains."),
        ("2. vif (cursor on 'def multiply')",
         "Visual select inside function",
         "Selects the body of multiply: just 'return a * b'\n"
         "    (the lines inside the function, but NOT the def line itself)."),
        ("3. daa (cursor on 'a' in 'def add(self, a, b)')",
         "Delete around parameter",
         "Deletes the parameter 'a' including the comma separator.\n"
         "    Signature becomes: def add(self, b):"),
        ("4. ]f (cursor anywhere inside 'add')",
         "Jump to next function",
         "Moves cursor to the start of the next function definition:\n"
         "    def multiply(self, a, b):"),
    ]
    # Why: Treesitter builds a real parse tree, enabling language-aware text objects.
    # daf, vif, daa operate on actual language constructs (functions, parameters),
    # not just text patterns.
    for cmd_desc, action, explanation in commands:
        print(f"  {cmd_desc}")
        print(f"    Action: {action}")
        print(f"    Effect: {explanation}\n")


# === Exercise 5: IDE Feature Comparison ===
# Problem: Match VS Code features to Neovim equivalents.

def exercise_5():
    """Solution: VS Code to Neovim feature mapping."""
    mappings = [
        ("Go to Definition (F12)",
         "gd",
         "vim.lsp.buf.definition"),
        ("Peek at hover docs (Ctrl+K, Ctrl+I)",
         "K",
         "vim.lsp.buf.hover"),
        ("Rename symbol (F2)",
         "<leader>rn",
         "vim.lsp.buf.rename"),
        ("Show all references",
         "gr",
         "vim.lsp.buf.references"),
        ("Next diagnostic error",
         "]d",
         "vim.diagnostic.goto_next"),
        ("Open integrated terminal (Ctrl+`)",
         ":terminal (or <C-\\> with toggleterm.nvim)",
         "Built-in :terminal or toggleterm plugin"),
    ]
    # Why: Neovim's built-in LSP client provides the same IDE features as VS Code.
    # The keybindings are configured in the LspAttach autocommand.
    print(f"  {'VS Code Feature':<40s} {'Neovim Key':<15s} {'API/Plugin'}")
    print(f"  {'-' * 40} {'-' * 15} {'-' * 35}")
    for vscode, nvim_key, api in mappings:
        print(f"  {vscode:<40s} {nvim_key:<15s} {api}")

    print()
    print("  All LSP features are provided by Neovim's built-in LSP client.")
    print("  Keybindings are configured in the LspAttach autocommand from the lesson.")
    print("  Your own config may use different keys.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Vim vs Neovim Decision", exercise_1),
        ("Exercise 2: Vimscript to Lua Translation", exercise_2),
        ("Exercise 3: LSP Workflow", exercise_3),
        ("Exercise 4: Treesitter Text Objects", exercise_4),
        ("Exercise 5: IDE Feature Comparison", exercise_5),
    ]
    for title, func in exercises:
        print(f"=== {title} ===")
        func()
        print()
    print("All exercises completed!")
