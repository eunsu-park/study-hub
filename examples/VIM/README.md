# VIM Examples

This directory contains practice files and configuration examples for learning Vim and Neovim.

## Files Overview

| # | File | Topic | Key Demos |
|---|------|-------|-----------|
| 01 | `01_basic_motions.txt` | Basic Motions | h/j/k/l, word motions, line motions, screen motions |
| 02 | `02_operators_practice.txt` | Operators | Operator + motion combinations, d/c/y/g~ |
| 03 | `03_text_objects_practice.py` | Text Objects | iw/aw, i(/a(, i"/a", Python-specific practice |
| 04 | `04_search_replace.txt` | Search and Replace | /, ?, :s, :g, regex patterns |
| 05 | `05_macro_examples.txt` | Macros | Recording (q), playback (@), recursive macros |
| 06 | `06_minimal_vimrc.vim` | Minimal .vimrc | Sensible defaults, commented options |
| 07 | `07_intermediate_vimrc.vim` | Intermediate .vimrc | Key mappings, autocmds, status line |
| 08 | `08_advanced_vimrc.vim` | Advanced .vimrc | Custom functions, conditionals, plugin integration |
| 09 | `09_init_lua.lua` | Neovim init.lua | Lua equivalents of Vimscript settings |
| 10 | `10_vim_cheatsheet.md` | Command Cheatsheet | Quick reference for modes, motions, operators |
| 11 | `11_modes_practice.txt` | Modes Practice | Normal, Insert, Visual, Command-line switching |
| 12 | `12_editing_practice.txt` | Essential Editing | Insert/append, delete, change, undo/redo |
| 13 | `13_visual_mode_practice.txt` | Visual Mode | Character/line/block selection, operators on selections |
| 14 | `14_buffers_windows_tabs.txt` | Buffers, Windows, Tabs | Container hierarchy, splits, tab pages |
| 15 | `15_command_line_advanced.txt` | Command-line and Advanced | Ex commands, ranges, external filters |
| 16 | `16_plugins_guide.md` | Plugins and Ecosystem | Plugin managers, essential plugins, configuration |

## Using Practice Files

Open any `.txt` practice file in Vim and follow the embedded instructions:

```bash
vim 01_basic_motions.txt
```

Configuration files can be copied to your home directory:

```bash
cp 06_minimal_vimrc.vim ~/.vimrc
```

For Neovim Lua configuration:

```bash
cp 09_init_lua.lua ~/.config/nvim/init.lua
```
