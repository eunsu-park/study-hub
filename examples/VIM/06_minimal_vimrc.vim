" ============================================================================
" Minimal .vimrc — A sensible starting point
" ============================================================================
" Copy this file to ~/.vimrc to use it.
" Each option is commented to explain what it does and why.
" ============================================================================

" --- Essential: Modern Vim defaults ---
" Why: Vim starts in 'compatible' mode by default, which emulates old vi
" behavior. This disables that, enabling all Vim improvements.
set nocompatible

" --- File Encoding ---
" Why: UTF-8 is the universal standard. Without this, special characters
" (accented letters, emoji, CJK) may display incorrectly.
set encoding=utf-8
set fileencoding=utf-8

" --- Display ---
" Why: Line numbers help you navigate (e.g., :42 or 42G to jump to line 42).
" Relative numbers make it easy to count lines for motions (e.g., 5j).
set number
set relativenumber

" Why: Highlights the line your cursor is on, making it easier to find.
set cursorline

" Why: When your cursor is on a bracket, this briefly highlights the match.
set showmatch

" Why: Shows partial commands (like d or y) in the bottom-right as you type.
set showcmd

" Why: Shows which mode you're in (INSERT, VISUAL, etc.) at the bottom.
set showmode

" Why: Keeps 8 lines visible above and below the cursor when scrolling.
" Without this, the cursor can reach the screen edge before scrolling starts.
set scrolloff=8

" Why: Show long lines wrapped at word boundaries, not mid-character.
set wrap
set linebreak

" --- Search ---
" Why: Highlights all matches of your search pattern across the file.
set hlsearch

" Why: Shows matches incrementally as you type the search pattern.
set incsearch

" Why: Makes search case-insensitive by default.
set ignorecase

" Why: If your search pattern contains uppercase, search becomes case-sensitive.
" Combined with ignorecase: /hello matches "Hello", but /Hello only matches "Hello".
set smartcase

" --- Tabs and Indentation ---
" Why: 4 spaces is the most common convention. expandtab converts Tab key to spaces.
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab

" Why: Copies the indentation of the current line when starting a new one.
set autoindent

" Why: Adds smart indentation for code (e.g., after { or : in some languages).
set smartindent

" --- Backspace Behavior ---
" Why: Without this, Backspace may not work over indentation, line breaks,
" or text inserted before the current Insert mode session.
set backspace=indent,eol,start

" --- File Handling ---
" Why: Allows switching between buffers without saving first.
" Without this, :bn and :bp require you to save before switching.
set hidden

" Why: If a file is changed outside Vim (e.g., by git), auto-reload it.
set autoread

" Why: Disable swap files and backups. Modern systems have auto-save
" and version control, making these redundant.
set noswapfile
set nobackup
set nowritebackup

" --- Command Line ---
" Why: Shows a visual menu of completions when pressing Tab in command mode.
set wildmenu
set wildmode=list:longest,full

" --- Splits ---
" Why: New splits open in the intuitive direction (below and right).
set splitbelow
set splitright

" --- Status Line ---
" Why: Always show the status line (file name, position, etc.).
set laststatus=2

" --- Performance ---
" Why: Faster screen updates for better responsiveness.
set updatetime=300

" --- Syntax and Filetype ---
" Why: Enables color-coded syntax highlighting.
syntax on

" Why: Enables filetype detection, plugins, and indentation rules.
" This is essential for per-language behavior (Python indent, etc.).
filetype plugin indent on

" --- Recommended: Clear search highlighting with Esc ---
nnoremap <Esc><Esc> :nohlsearch<CR>

" ============================================================================
" That's it! This ~60-line config gives you a comfortable editing experience.
" Grow your .vimrc gradually — only add settings you understand and need.
" ============================================================================
