" ============================================================================
" Advanced .vimrc — Functions, conditionals, and plugin integration
" ============================================================================
" Builds on 06_minimal and 07_intermediate. This config demonstrates:
" - Custom functions for complex behavior
" - Conditional settings based on environment
" - Plugin management with vim-plug
" - Project-specific settings
" ============================================================================

" ---- Plugin Management with vim-plug ----
" Why: vim-plug is the most popular Vim plugin manager — minimal, fast,
" and supports lazy loading. Install it first:
"   curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
"     https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
"
" Then run :PlugInstall after saving this file.

call plug#begin('~/.vim/plugged')

" --- Text Manipulation ---
" Why: vim-surround adds 'cs', 'ds', 'ys' for changing/deleting/adding
" surrounding characters. Example: cs"' changes "hello" to 'hello'.
Plug 'tpope/vim-surround'

" Why: gcc to toggle comment on a line, gc{motion} for a range.
Plug 'tpope/vim-commentary'

" Why: Makes . (dot repeat) work with plugin commands like surround/commentary.
Plug 'tpope/vim-repeat'

" Why: Pairs of bracket mappings: [b/]b for buffers, [q/]q for quickfix, etc.
Plug 'tpope/vim-unimpaired'

" --- Navigation ---
" Why: Fuzzy finder for files, buffers, lines, git commits, and more.
" fzf is a standalone tool; fzf.vim integrates it into Vim.
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

" Why: File tree explorer. Toggle with a key mapping.
Plug 'preservim/nerdtree'

" --- Git ---
" Why: The gold standard for Git integration in Vim.
" :Git for status, :Git blame for per-line blame, etc.
Plug 'tpope/vim-fugitive'

" Why: Shows git diff markers (added/modified/deleted) in the sign column.
Plug 'mhinz/vim-signify'

" --- Language Support ---
" Why: A single plugin for better syntax highlighting across many languages.
Plug 'sheerun/vim-polyglot'

" Why: Asynchronous linting and fixing. Runs linters in the background.
Plug 'dense-analysis/ale'

" --- Appearance ---
" Why: A popular warm color scheme with good contrast.
Plug 'morhetz/gruvbox'

call plug#end()

" ============================================================================
" CONDITIONAL SETTINGS
" ============================================================================

" Why: True color support makes color schemes look much better,
" but only works in terminals that support it (iTerm2, Alacritty, etc.).
if has('termguicolors')
    set termguicolors
endif

" Why: Apply color scheme only if the plugin is installed.
" silent! suppresses errors if the scheme isn't available.
set background=dark
silent! colorscheme gruvbox

" Why: Different settings for GUI (gVim/MacVim) vs terminal Vim.
if has('gui_running')
    set guifont=JetBrains\ Mono:h14
    set guioptions-=T       " Remove toolbar
    set guioptions-=m       " Remove menu bar
    set guioptions-=r       " Remove right scrollbar
    set guioptions-=L       " Remove left scrollbar
endif

" Why: Check if clipboard support is available before setting it.
if has('clipboard')
    set clipboard=unnamedplus
endif

" ============================================================================
" PLUGIN CONFIGURATION
" ============================================================================

" --- fzf.vim ---
" Why: Leader-based shortcuts for the most common fuzzy-find operations.
nnoremap <leader>f :Files<CR>
nnoremap <leader>g :Rg<CR>
nnoremap <leader>b :Buffers<CR>
nnoremap <leader>l :Lines<CR>
nnoremap <leader>c :Commits<CR>
nnoremap <leader>m :Marks<CR>

" Why: Show fzf in a floating window (cleaner appearance).
let g:fzf_layout = { 'window': { 'width': 0.8, 'height': 0.6 } }

" --- NERDTree ---
" Why: Toggle the file explorer, or reveal the current file in the tree.
nnoremap <leader>n :NERDTreeToggle<CR>
nnoremap <leader>N :NERDTreeFind<CR>

" Why: Close NERDTree if it's the only window left (avoids empty Vim).
autocmd BufEnter * if winnr('$') == 1 && exists('b:NERDTree')
    \ && b:NERDTree.isTabTree() | quit | endif

" Why: Show hidden (dot) files in NERDTree.
let NERDTreeShowHidden=1

" --- ALE (Linter/Fixer) ---
" Why: Configure which linters and fixers to use per language.
let g:ale_linters = {
\   'python': ['flake8', 'mypy'],
\   'javascript': ['eslint'],
\   'typescript': ['eslint'],
\   'go': ['golangci-lint'],
\}

let g:ale_fixers = {
\   '*': ['remove_trailing_lines', 'trim_whitespace'],
\   'python': ['black', 'isort'],
\   'javascript': ['prettier'],
\   'typescript': ['prettier'],
\   'go': ['gofmt'],
\}

" Why: Auto-fix on save — formats code every time you write the file.
let g:ale_fix_on_save = 1

" Why: Navigate between lint errors quickly.
nnoremap [e :ALEPreviousWrap<CR>
nnoremap ]e :ALENextWrap<CR>

" --- vim-signify ---
" Why: Show line-level git changes with subtle markers.
let g:signify_sign_add    = '│'
let g:signify_sign_change = '│'
let g:signify_sign_delete = '_'

" ============================================================================
" CUSTOM FUNCTIONS
" ============================================================================

" --- Toggle Between Relative and Absolute Line Numbers ---
" Why: Relative numbers are great for motions (5j, 10k) but absolute
" numbers are better when someone says "look at line 42." Toggle between them.
function! ToggleLineNumbers()
    if &relativenumber
        set norelativenumber
        set number
        echo "Absolute line numbers"
    else
        set relativenumber
        set number
        echo "Relative line numbers"
    endif
endfunction
nnoremap <leader>rn :call ToggleLineNumbers()<CR>

" --- Smart Tab Completion ---
" Why: Tab completes words from buffer when in the middle of a word,
" but inserts a real tab at the start of a line.
function! SmartTab()
    if col('.') > 1 && getline('.')[col('.') - 2] =~ '\w'
        return "\<C-n>"
    else
        return "\<Tab>"
    endif
endfunction
inoremap <expr> <Tab> SmartTab()
inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"

" --- Strip Trailing Whitespace (with cursor preservation) ---
" Why: The simple :%s/\s+$//e approach moves the cursor.
" This function preserves cursor position and search register.
function! StripTrailingWhitespace()
    let l:save = winsaveview()
    keeppatterns %s/\s\+$//e
    call winrestview(l:save)
endfunction
autocmd BufWritePre * call StripTrailingWhitespace()

" --- Quick Terminal ---
" Why: Open a small terminal split at the bottom for running commands.
function! OpenTerminal()
    botright split
    resize 12
    terminal
endfunction
nnoremap <leader>t :call OpenTerminal()<CR>

" --- Run Current File ---
" Why: Execute the current file based on its filetype.
" Supports common scripting languages.
function! RunFile()
    write
    let l:ft = &filetype
    if l:ft == 'python'
        execute '!python3 %'
    elseif l:ft == 'javascript'
        execute '!node %'
    elseif l:ft == 'typescript'
        execute '!npx tsx %'
    elseif l:ft == 'sh' || l:ft == 'bash'
        execute '!bash %'
    elseif l:ft == 'go'
        execute '!go run %'
    elseif l:ft == 'c'
        execute '!gcc % -o /tmp/vim_run && /tmp/vim_run'
    elseif l:ft == 'cpp'
        execute '!g++ -std=c++17 % -o /tmp/vim_run && /tmp/vim_run'
    else
        echo "No runner configured for filetype: " . l:ft
    endif
endfunction
nnoremap <leader>r :call RunFile()<CR>

" ============================================================================
" CUSTOM COMMANDS
" ============================================================================

" Why: Quick commands for common operations.
command! TrimWhitespace call StripTrailingWhitespace()
command! Config edit $MYVIMRC
command! Reload source $MYVIMRC

" Why: Create a scratch buffer for temporary notes.
command! Scratch new | setlocal buftype=nofile bufhidden=wipe noswapfile

" Why: Show the highlight group under the cursor (useful for theme tweaking).
command! SynGroup echo synIDattr(synID(line('.'), col('.'), 1), 'name')

" ============================================================================
" PROJECT-SPECIFIC SETTINGS
" ============================================================================

" Why: Allow per-project .vimrc files for project-specific settings.
" The 'secure' option prevents untrusted configs from running shell commands.
set exrc
set secure

" ============================================================================
