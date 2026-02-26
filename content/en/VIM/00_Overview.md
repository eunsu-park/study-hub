# VIM Study Guide

## Introduction

VIM (Vi IMproved) is one of the most powerful and efficient text editors ever created. Born from the Unix tradition, VIM's modal editing philosophy lets you manipulate text at the speed of thought — once you learn its language. This guide takes you from first steps to an IDE-level development workflow.

Whether you need to quickly edit a config file on a remote server, efficiently refactor code, or build a fully customized development environment, VIM skills pay dividends for your entire career.

---

## Learning Roadmap

```
                        VIM Learning Path
    ════════════════════════════════════════════════

    Phase 1: Survival          Phase 2: Efficiency
    ┌─────────────────┐        ┌─────────────────────┐
    │ L01 Philosophy   │───────▶│ L04 Motions          │
    │ L02 Modes        │       │ L05 Operators         │
    │ L03 Basic Editing│       │ L06 Text Objects      │
    └─────────────────┘        │ L07 Visual Mode       │
                               │ L08 Search & Replace  │
                               └───────────┬───────────┘
                                           │
                               Phase 3: Mastery
                               ┌───────────┴───────────┐
                               │ L09 Registers & Macros │
                               │ L10 Buffers & Windows  │
                               │ L11 Command Line       │
                               │ L12 Configuration      │
                               │ L13 Plugins            │
                               │ L14 Neovim & Modern    │
                               └───────────────────────┘
```

---

## Prerequisites

- Basic terminal/command-line usage ([Shell Script](../Shell_Script/00_Overview.md))
- Familiarity with any programming language (helpful but not required)

---

## File List

### Phase 1: Survival (⭐)

| File | Difficulty | Key Topics |
|------|-----------|------------|
| [01_Introduction_and_Philosophy.md](./01_Introduction_and_Philosophy.md) | ⭐ | History, modal editing, vi/vim/neovim, installation, vimtutor |
| [02_Modes_and_Basic_Navigation.md](./02_Modes_and_Basic_Navigation.md) | ⭐ | Normal/Insert/Command-line modes, hjkl, mode switching |
| [03_Essential_Editing.md](./03_Essential_Editing.md) | ⭐ | Insert commands, delete/yank/put, undo/redo, save/quit |

### Phase 2: Efficiency (⭐⭐)

| File | Difficulty | Key Topics |
|------|-----------|------------|
| [04_Motions_and_Navigation.md](./04_Motions_and_Navigation.md) | ⭐⭐ | Word/line/screen motions, f/t character search, jumping |
| [05_Operators_and_Composability.md](./05_Operators_and_Composability.md) | ⭐⭐ | Operator + motion grammar, d/c/y combos, dot repeat |
| [06_Text_Objects.md](./06_Text_Objects.md) | ⭐⭐ | Inner/around objects, word/sentence/paragraph/bracket/quote |
| [07_Visual_Mode.md](./07_Visual_Mode.md) | ⭐⭐ | Character/line/block visual, selection + operators, reselect |
| [08_Search_and_Replace.md](./08_Search_and_Replace.md) | ⭐⭐ | Forward/backward search, substitution, regex, global commands |

### Phase 3: Mastery (⭐⭐⭐)

| File | Difficulty | Key Topics |
|------|-----------|------------|
| [09_Registers_Marks_and_Macros.md](./09_Registers_Marks_and_Macros.md) | ⭐⭐⭐ | Named registers, marks, macro recording/playback, clipboard |
| [10_Buffers_Windows_and_Tabs.md](./10_Buffers_Windows_and_Tabs.md) | ⭐⭐⭐ | Buffer management, split windows, tabs, arglist |
| [11_Command_Line_and_Advanced_Features.md](./11_Command_Line_and_Advanced_Features.md) | ⭐⭐⭐ | Ex commands, external commands, ranges, folds, sessions |
| [12_Configuration_and_Vimrc.md](./12_Configuration_and_Vimrc.md) | ⭐⭐⭐ | .vimrc structure, options, key mappings, autocmd, filetype |
| [13_Plugins_and_Ecosystem.md](./13_Plugins_and_Ecosystem.md) | ⭐⭐⭐ | Plugin managers, essential plugins, writing simple plugins |
| [14_Neovim_and_Modern_Workflows.md](./14_Neovim_and_Modern_Workflows.md) | ⭐⭐⭐ | Neovim, Lua config, LSP, Treesitter, terminal, Git integration |

---

## Example Files

Example files are in [`examples/VIM/`](../../../examples/VIM/):

| File | Description |
|------|-------------|
| `01_basic_motions.txt` | Motion practice text with exercise instructions |
| `02_operators_practice.txt` | Operator + motion combination exercises |
| `03_text_objects_practice.py` | Python code for text object practice |
| `04_search_replace.txt` | Search and substitution pattern exercises |
| `05_macro_examples.txt` | Macro recording exercises (repetitive editing) |
| `06_minimal_vimrc.vim` | Minimal .vimrc with annotated options |
| `07_intermediate_vimrc.vim` | Intermediate config (mappings, autocmd, statusline) |
| `08_advanced_vimrc.vim` | Advanced config (functions, conditionals, plugins) |
| `09_init_lua.lua` | Neovim init.lua basic setup |
| `10_vim_cheatsheet.md` | Command reference cheatsheet |

---

## Recommended Learning Path

### Week 1: Survival
Complete L01–L03. Run `vimtutor` daily. Force yourself to use VIM for small edits.

### Week 2–3: Building Speed
Work through L04–L08. The "operator + motion" grammar (L05) is the single most important concept — once it clicks, everything accelerates.

### Week 4+: Customization & Power
Explore L09–L14 based on your needs. Set up your `.vimrc` (L12), add essential plugins (L13), and consider Neovim (L14) for a modern setup.

---

## Practice Environment

```bash
# Check if vim is installed
vim --version | head -1

# macOS (comes pre-installed, or install latest)
brew install vim        # Latest Vim
brew install neovim     # Neovim

# Ubuntu/Debian
sudo apt install vim    # Vim
sudo apt install neovim # Neovim

# Start the built-in tutorial (best first step!)
vimtutor
```

---

## Related Topics

- [Shell_Script](../Shell_Script/00_Overview.md) — Command-line skills (VIM lives in the terminal)
- [Linux](../Linux/00_Overview.md) — System administration where VIM is the default editor
- [Git](../Git/00_Overview.md) — Commit messages, interactive rebase use VIM by default

---

## References

- [Vim Official Documentation](https://www.vim.org/docs.php)
- [Neovim Documentation](https://neovim.io/doc/)
- [Practical Vim by Drew Neil](https://pragprog.com/titles/dnvim2/practical-vim-second-edition/)
- [Learn Vimscript the Hard Way](https://learnvimscriptthehardway.stevelosh.com/)
- [Vim Adventures](https://vim-adventures.com/) — Game-based learning
