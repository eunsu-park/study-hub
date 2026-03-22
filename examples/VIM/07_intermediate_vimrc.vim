" ============================================================================
" Intermediate .vimrc — Key mappings, autocmds, and status line
" ============================================================================
" Builds on 06_minimal_vimrc.vim. Assumes you're comfortable with basic Vim
" and want to customize your workflow.
" ============================================================================

" --- Leader Key ---
" Why: Space as leader is popular because it's the largest key and easy to
" reach from home row. Many modern configs use this convention.
let mapleader = " "
let maplocalleader = ","

" ============================================================================
" KEY MAPPINGS
" ============================================================================
" Always use 'noremap' variants to avoid recursive mapping surprises.

" --- File Operations ---
" Why: Ctrl-s is muscle memory from other editors. Works in Normal and Insert.
nnoremap <C-s> :w<CR>
inoremap <C-s> <Esc>:w<CR>a

" Why: Quick access to common operations via leader key.
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>x :x<CR>

" --- Window Navigation ---
" Why: Ctrl-w + hjkl is too many keystrokes. Ctrl+hjkl is faster.
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" --- Window Splits ---
nnoremap <leader>v :vsplit<CR>
nnoremap <leader>s :split<CR>

" --- Buffer Navigation ---
" Why: [b and ]b follow Vim's convention for "previous/next" bracket mappings.
nnoremap [b :bprev<CR>
nnoremap ]b :bnext<CR>
nnoremap <leader>b :ls<CR>:b<Space>

" Why: Ctrl-^ toggles between last two buffers, but it's hard to reach.
nnoremap <leader><leader> <C-^>

" --- Search ---
" Why: Clear search highlighting without typing :nohlsearch every time.
nnoremap <leader>h :nohlsearch<CR>

" Why: Center the screen after jumping to a search result.
" Prevents disorientation when n/N jumps you to a distant match.
nnoremap n nzzzv
nnoremap N Nzzzv

" --- Line Movement ---
" Why: Alt+j/k moves the current line up or down (like VS Code).
" == re-indents after moving.
nnoremap <A-j> :m .+1<CR>==
nnoremap <A-k> :m .-2<CR>==
vnoremap <A-j> :m '>+1<CR>gv=gv
vnoremap <A-k> :m '<-2<CR>gv=gv

" --- Visual Mode Improvements ---
" Why: After indenting a visual selection, the selection is lost.
" These mappings re-select after indent so you can indent multiple levels.
vnoremap < <gv
vnoremap > >gv

" --- Y Consistency ---
" Why: By default, Y yanks the entire line (like yy). But D deletes to end,
" and C changes to end. Y should be consistent: yank to end of line.
nnoremap Y y$

" --- Quick Edit Config ---
" Why: Quickly edit and reload your vimrc without leaving Vim.
nnoremap <leader>ev :edit $MYVIMRC<CR>
nnoremap <leader>sv :source $MYVIMRC<CR>

" --- Scrolling ---
" Why: Center screen after half-page jumps to maintain context.
nnoremap <C-d> <C-d>zz
nnoremap <C-u> <C-u>zz

" --- Disable Arrow Keys (Training Wheels) ---
" Why: Forces you to use hjkl. Remove these once you've built the habit.
" Uncomment if you want to enforce this:
" noremap <Up>    <Nop>
" noremap <Down>  <Nop>
" noremap <Left>  <Nop>
" noremap <Right> <Nop>

" ============================================================================
" AUTOCOMMANDS
" ============================================================================

" Why: augroup + autocmd! prevents duplicate commands when re-sourcing .vimrc.
augroup MyAutocommands
    autocmd!

    " Why: Removes invisible trailing whitespace on every save.
    " The 'e' flag suppresses errors when no matches are found.
    autocmd BufWritePre * :%s/\s\+$//e

    " Why: Returns to the last cursor position when reopening a file.
    " This is one of the most appreciated quality-of-life features.
    autocmd BufReadPost * if line("'\"") > 0 && line("'\"") <= line("$")
        \ | execute "normal! g'\"" | endif

    " Why: Automatically rebalance window sizes when terminal is resized.
    autocmd VimResized * wincmd =

    " Why: Highlight briefly when yanking (Vim 8.1.1270+).
    if exists('##TextYankPost')
        autocmd TextYankPost * silent! lua vim.highlight.on_yank()
    endif
augroup END

" ============================================================================
" FILETYPE-SPECIFIC SETTINGS
" ============================================================================

augroup FileTypeSettings
    autocmd!

    " Why: Python uses 4-space indentation (PEP 8).
    autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab
    autocmd FileType python setlocal colorcolumn=79

    " Why: JavaScript/TypeScript community standard is 2-space indentation.
    autocmd FileType javascript,typescript setlocal tabstop=2 shiftwidth=2 expandtab

    " Why: HTML/CSS also typically use 2-space indentation.
    autocmd FileType html,css setlocal tabstop=2 shiftwidth=2 expandtab

    " Why: Go uses real tabs (gofmt enforces this).
    autocmd FileType go setlocal tabstop=4 shiftwidth=4 noexpandtab

    " Why: Markdown should wrap visually and enable spell checking.
    autocmd FileType markdown setlocal wrap linebreak spell spelllang=en_us

    " Why: Makefiles require real tabs (spaces break them).
    autocmd FileType make setlocal noexpandtab

    " Why: YAML is indentation-sensitive; 2 spaces is standard.
    autocmd FileType yaml setlocal tabstop=2 shiftwidth=2 expandtab
augroup END

" ============================================================================
" STATUS LINE
" ============================================================================

" Why: A custom status line provides useful info at a glance without plugins.
set laststatus=2

set statusline=
set statusline+=\ %{toupper(mode())}    " Current mode
set statusline+=\ │                      " Separator
set statusline+=\ %f                     " File path (relative)
set statusline+=\ %m                     " Modified flag [+]
set statusline+=\ %r                     " Read-only flag [RO]
set statusline+=%=                       " Right-align from here
set statusline+=\ %y                     " Filetype [python]
set statusline+=\ │                      " Separator
set statusline+=\ %{&encoding}           " Encoding (utf-8)
set statusline+=\ │                      " Separator
set statusline+=\ %p%%                   " Percentage through file
set statusline+=\ │                      " Separator
set statusline+=\ %l:%c                  " Line:Column
set statusline+=\                        " Trailing space

" ============================================================================
" ADDITIONAL COMFORT SETTINGS
" ============================================================================

" Why: Persistent undo — your undo history survives quitting Vim.
if has('persistent_undo')
    set undofile
    set undodir=~/.vim/undodir
    " Create the directory if it doesn't exist
    if !isdirectory(expand('~/.vim/undodir'))
        call mkdir(expand('~/.vim/undodir'), 'p')
    endif
endif

" Why: Show invisible characters when you toggle with :set list
set listchars=tab:→\ ,trail:·,extends:»,precedes:«,nbsp:␣

" Why: Enable mouse support in all modes (useful for scrolling, resizing splits).
set mouse=a

" Why: Faster timeout for key sequences (default 1000ms is sluggish).
set timeoutlen=500
set ttimeoutlen=10

" ============================================================================
