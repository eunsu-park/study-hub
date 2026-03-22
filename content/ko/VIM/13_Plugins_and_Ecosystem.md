# 플러그인과 생태계

**이전**: [설정과 Vimrc](./12_Configuration_and_Vimrc.md) | **다음**: [Neovim과 현대적 워크플로우](./14_Neovim_and_Modern_Workflows.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. vim-plug(Vim)와 lazy.nvim(Neovim)을 사용하여 플러그인(Plugin)을 설치하고 관리하기
2. 파일 탐색, 퍼지 검색(fuzzy finding), Git을 위한 필수 플러그인 설정하기
3. surround, commentary 등 텍스트 조작 플러그인 사용하기
4. 간단한 Vim 플러그인을 작성하는 기초 이해하기
5. 플러그인을 평가하고 관리하기 쉬운 플러그인 세트 구축하기

---

플러그인은 Vim의 이미 강력한 핵심 기능을 더욱 확장합니다. 생태계에는 수천 개의 플러그인이 있지만, 훌륭한 개발 환경을 만들기 위해 필요한 것은 소수에 불과합니다. 이 레슨에서는 플러그인 관리, 필수 플러그인, 그리고 현명하게 선택하는 원칙을 다룹니다.

## 목차

1. [플러그인 매니저](#1-플러그인-매니저)
2. [필수 플러그인](#2-필수-플러그인)
3. [텍스트 조작 플러그인](#3-텍스트-조작-플러그인)
4. [파일 탐색 플러그인](#4-파일-탐색-플러그인)
5. [Git 플러그인](#5-git-플러그인)
6. [언어 지원](#6-언어-지원)
7. [간단한 플러그인 작성하기](#7-간단한-플러그인-작성하기)
8. [플러그인 철학](#8-플러그인-철학)
9. [요약](#9-요약)

---

## 1. 플러그인 매니저

### vim-plug (Vim에 권장)

Vim에서 가장 인기 있는 플러그인 매니저 — 가볍고 빠르며 사용이 간편합니다.

**설치:**
```bash
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

**.vimrc에서 사용:**
```vim
call plug#begin('~/.vim/plugged')

Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

call plug#end()
```

**명령어:**
| 명령어 | 동작 |
|---------|--------|
| `:PlugInstall` | 플러그인 설치 |
| `:PlugUpdate` | 플러그인 업데이트 |
| `:PlugClean` | 사용하지 않는 플러그인 제거 |
| `:PlugStatus` | 플러그인 상태 확인 |

### lazy.nvim (Neovim에 권장)

지연 로딩(lazy loading)을 지원하는 Neovim용 현대적인 Lua 기반 플러그인 매니저입니다.

```lua
-- lazy.nvim 부트스트랩 (init.lua에 추가)
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

### 다른 플러그인 매니저

| 매니저 | 언어 | 지연 로딩 | 비고 |
|---------|----------|-------------|-------|
| vim-plug | Vimscript | 기본 지원 | Vim에서 가장 인기 있음 |
| lazy.nvim | Lua | 고급 지원 | Neovim 표준 |
| packer.nvim | Lua | 지원 | lazy.nvim의 전신 |
| Vundle | Vimscript | 미지원 | 구식, 유지 보수 부족 |
| pathogen | Vimscript | 미지원 | 단순하지만 수동 관리 필요 |

---

## 2. 필수 플러그인

### "꼭 설치해야 할" 목록

거의 모든 사용자에게 권장되는 플러그인:

| 플러그인 | 목적 | 작성자 |
|--------|---------|--------|
| vim-surround | 텍스트 오브젝트 둘러싸기 | tpope |
| vim-commentary | 주석 토글 | tpope |
| vim-repeat | 플러그인에서 `.` 명령 지원 | tpope |
| fzf.vim / telescope.nvim | 퍼지 검색 | junegunn / nvim-telescope |
| vim-fugitive | Git 통합 | tpope |
| NERDTree / nvim-tree | 파일 탐색기 | scrooloose / nvim-tree |

Tim Pope의 플러그인(`tpope/*`)은 Vim 플러그인 품질의 표준으로 여겨집니다.

---

## 3. 텍스트 조작 플러그인

### vim-surround

따옴표, 괄호, 태그 등 둘러싸는 문자를 추가, 변경, 삭제합니다.

```vim
Plug 'tpope/vim-surround'
```

| 명령어 | 변경 전 | 변경 후 |
|---------|--------|-------|
| `cs"'` | `"hello"` | `'hello'` |
| `cs'<q>` | `'hello'` | `<q>hello</q>` |
| `ds"` | `"hello"` | `hello` |
| `ysiw"` | `hello` | `"hello"` |
| `yss(` | `hello world` | `( hello world )` |
| `yss)` | `hello world` | `(hello world)` |

**주요 매핑:**
- `cs{이전}{새로운}` — 둘러싸기 **변경(C**hange **s**urrounding)
- `ds{문자}` — 둘러싸기 **삭제(D**elete **s**urrounding)
- `ys{모션}{문자}` — 둘러싸기 **추가(y**ou **s**urround)
- `S{문자}` — 비주얼 모드에서 둘러싸기

### vim-commentary (또는 Comment.nvim)

단일 명령으로 주석을 토글합니다.

```vim
" Vim
Plug 'tpope/vim-commentary'
```

| 명령어 | 동작 |
|---------|--------|
| `gcc` | 현재 줄 주석 토글 |
| `gc{모션}` | 모션 범위 주석 토글 |
| `gcap` | 문단 주석 처리 |
| `gc` (비주얼) | 선택 영역 주석 처리 |

```lua
-- Neovim (Comment.nvim)
{ "numToStr/Comment.nvim", config = true }
```

### vim-repeat

플러그인 동작(surround, commentary 등)에서 `.` 명령이 동작하도록 합니다.

```vim
Plug 'tpope/vim-repeat'
```

별도의 설정 없이 설치만 하면 `.`이 플러그인 명령을 반복합니다.

### vim-unimpaired

일반적인 작업에 대한 대괄호 쌍 매핑을 제공합니다.

```vim
Plug 'tpope/vim-unimpaired'
```

| 명령어 | 동작 |
|---------|--------|
| `[b` / `]b` | 이전/다음 버퍼 |
| `[q` / `]q` | 이전/다음 quickfix |
| `[l` / `]l` | 이전/다음 위치 목록 |
| `[<Space>` | 위에 빈 줄 추가 |
| `]<Space>` | 아래에 빈 줄 추가 |
| `[e` / `]e` | 줄 위/아래로 이동 |

---

## 4. 파일 탐색 플러그인

### fzf.vim (Vim) / Telescope (Neovim)

퍼지 파인더(Fuzzy Finder)는 프로젝트를 탐색하는 가장 빠른 방법입니다.

**fzf.vim:**
```vim
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

" 키 매핑
nnoremap <leader>f :Files<CR>       " 파일 찾기
nnoremap <leader>g :Rg<CR>          " 내용 검색
nnoremap <leader>b :Buffers<CR>     " 버퍼 전환
nnoremap <leader>l :Lines<CR>       " 줄 검색
nnoremap <leader>c :Commits<CR>     " Git 커밋
```

| 명령어 | 동작 |
|---------|--------|
| `:Files` | 파일 퍼지 검색 |
| `:Buffers` | 버퍼 퍼지 검색 |
| `:Rg {패턴}` | Ripgrep 검색 |
| `:Lines` | 모든 버퍼 줄 검색 |
| `:BLines` | 현재 버퍼에서 검색 |
| `:Commits` | Git 커밋 탐색 |
| `:History` | 최근 파일 |

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

### 파일 탐색기

**NERDTree (Vim):**
```vim
Plug 'preservim/nerdtree'

nnoremap <leader>n :NERDTreeToggle<CR>
nnoremap <leader>N :NERDTreeFind<CR>    " 현재 파일 위치 표시
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

## 5. Git 플러그인

### vim-fugitive

Vim을 위한 결정적인 Git 플러그인입니다.

```vim
Plug 'tpope/vim-fugitive'
```

| 명령어 | 동작 |
|---------|--------|
| `:Git` | Git 상태 열기 (인터랙티브 스테이징) |
| `:Git diff` | 차이점 보기 |
| `:Git blame` | 줄별 blame 보기 |
| `:Git log` | 로그 보기 |
| `:Git commit` | 커밋 |
| `:Git push` | 푸시 |
| `:Gread` | 파일을 Git 버전으로 되돌리기 |
| `:Gwrite` | 현재 파일 스테이징 |

### gitsigns.nvim (Neovim)

기호 열(gutter)에 Git 변경 사항을 표시합니다.

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

gitsigns의 Vim 버전입니다:

```vim
Plug 'mhinz/vim-signify'
```

---

## 6. 언어 지원

### 문법 강조 및 들여쓰기

Vim은 대부분의 언어를 기본으로 지원합니다. 향상된 지원이 필요하다면:

```vim
" 다양한 언어의 향상된 문법 지원
Plug 'sheerun/vim-polyglot'
```

Neovim에서는 Treesitter가 더 뛰어난 파싱을 제공합니다 ([레슨 14](./14_Neovim_and_Modern_Workflows.md) 참고).

### ALE (비동기 린트 엔진, Asynchronous Lint Engine)

Vim을 위한 린팅(linting) 및 수정 도구:

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

### CoC.nvim (자동완성 정복, Conquer of Completion)

Vim에서 VS Code의 언어 기능에 준하는 완전한 LSP 지원:

```vim
Plug 'neoclide/coc.nvim', {'branch': 'release'}
```

Neovim에서는 내장 LSP 클라이언트를 사용하는 것이 권장됩니다 ([레슨 14](./14_Neovim_and_Modern_Workflows.md) 참고).

---

## 7. 간단한 플러그인 작성하기

Vim 플러그인은 특정 디렉토리에 위치한 Vimscript (또는 Neovim의 경우 Lua) 파일에 불과합니다.

### 플러그인 디렉토리 구조

```
~/.vim/plugin/          " 자동으로 로드되는 스크립트
~/.vim/autoload/        " 필요할 때 지연 로드되는 함수
~/.vim/ftplugin/        " 파일 유형별 스크립트
~/.vim/after/plugin/    " 모든 플러그인 이후에 로드됨
```

### 간단한 플러그인: 단어 수 카운터

`~/.vim/plugin/wordcount.vim` 파일을 생성하세요:

```vim
" wordcount.vim — 상태 줄에 단어 수 표시

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

" 상태 줄에 추가
set statusline+=\ Words:%{WordCount()}
```

### 커스텀 명령어

```vim
" 현재 날짜를 삽입하는 명령어 생성
command! InsertDate execute "normal! i" . strftime('%Y-%m-%d')

" 후행 공백을 제거하는 명령어 생성
command! TrimWhitespace %s/\s\+$//e

" 인자를 받는 명령어 생성
command! -nargs=1 Grep execute 'silent grep!' <q-args> | copen
```

### 커스텀 함수

```vim
" 상대적/절대적 줄 번호 토글
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

## 8. 플러그인 철학

### 적을수록 좋다

모든 플러그인은:
- 시작 시간을 늘립니다
- 버그를 유발할 수 있습니다
- 다른 플러그인과 충돌할 수 있습니다
- 유지 보수가 필요합니다 (업데이트, 호환성)

### 플러그인 선택 원칙

1. **먼저 Vim 핵심 기능을 익히세요** — Vim 학습을 피하기 위해 플러그인을 설치하지 마세요
2. **실제 문제를 해결하세요** — 진짜 한계에 부딪혔을 때만 설치하세요
3. **품질 신호를 확인하세요** — 별점, 최근 커밋, 활발한 유지 관리자
4. **플러그인을 이해하세요** — README를 읽고, 키 매핑을 파악하세요
5. **수를 제한하세요** — 잘 선택된 10~15개가 무작위 50개보다 낫습니다

### 플러그인 점검 질문

플러그인을 설치하기 전에 자문하세요:
- Vim 내장 기능으로도 할 수 있지 않은가?
- 실제로 내가 사용하는 워크플로우를 대체하는가?
- 잘 유지 관리되고 있는가?
- 오버헤드를 감수할 만큼 자주 사용하게 될까?

### 권장 시작 세트 (10개)

```vim
" 텍스트 조작
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'tpope/vim-repeat'

" 탐색
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'
Plug 'preservim/nerdtree'

" Git
Plug 'tpope/vim-fugitive'
Plug 'mhinz/vim-signify'

" 언어
Plug 'sheerun/vim-polyglot'
Plug 'dense-analysis/ale'
```

---

## 9. 요약

| 분류 | Vim | Neovim |
|----------|-----|--------|
| 플러그인 매니저 | vim-plug | lazy.nvim |
| 퍼지 파인더 | fzf.vim | telescope.nvim |
| 파일 탐색기 | NERDTree | nvim-tree.lua |
| 주석 | vim-commentary | Comment.nvim |
| Git | vim-fugitive | vim-fugitive + gitsigns |
| 린팅 | ALE | 내장 LSP |
| 자동완성 | coc.nvim | nvim-cmp + LSP |
| 둘러싸기 | vim-surround | vim-surround / nvim-surround |

### 시작 체크리스트

1. 플러그인 매니저 설치 (vim-plug 또는 lazy.nvim)
2. vim-surround, vim-commentary, vim-repeat 추가
3. 퍼지 파인더 추가 (fzf 또는 telescope)
4. Git을 위한 vim-fugitive 추가
5. 필요하다면 파일 탐색기 추가
6. 여기서 멈추고 — 필요할 때만 더 추가하세요

---

## 연습 문제

### 연습 1: vim-plug 설정

새 Vim 설치를 설정하며 `tpope/vim-surround`, `tpope/vim-commentary`, `junegunn/fzf.vim`(`junegunn/fzf`도 필요) 세 개의 플러그인을 vim-plug로 설치하려 합니다.

1. 이 플러그인을 선언하는 완전한 `.vimrc` 섹션을 작성하세요.
2. 설정 작성 후 Vim 내에서 설치하는 명령어는?
3. 나중에 `junegunn/fzf.vim`과 `junegunn/fzf`를 제거하기로 했습니다. `.vimrc`에서 해당 줄을 삭제하고 리로드한 뒤 설치된 플러그인 파일을 정리하는 명령어는?

<details>
<summary>정답 보기</summary>

1. 플러그인 선언 블록:

```vim
call plug#begin('~/.vim/plugged')

Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

call plug#end()
```

참고: `junegunn/fzf`는 `junegunn/fzf.vim` 전에 선언해야 합니다. fzf.vim은 첫 번째 플러그인이 `do` 훅으로 설치하는 fzf 바이너리에 의존하기 때문입니다.

2. `:PlugInstall` — vim-plug가 선언된 모든 플러그인을 `~/.vim/plugged/`에 다운로드합니다.

3. `:PlugClean` — `.vimrc`에서 플러그인 줄을 삭제하고 리로드(`:source ~/.vimrc`)한 뒤, `:PlugClean`이 plugged 디렉토리에서 더 이상 선언되지 않은 플러그인을 감지하고 삭제를 제안합니다.

</details>

### 연습 2: vim-surround 조작

각 변환에 필요한 vim-surround 명령어를 작성하세요. 각 경우에 커서는 `hello` 단어에 있습니다:

1. `hello` → `"hello"` (단어에 큰따옴표 추가)
2. `"hello"` → `'hello'` (큰따옴표를 작은따옴표로 변경)
3. `'hello'` → `hello` (작은따옴표 제거)
4. `hello world` (커서가 줄 시작에 있음) → `(hello world)` (전체 줄을 공백 없이 괄호로 감싸기)

<details>
<summary>정답 보기</summary>

1. `ysiw"` — `ys`(you surround) + `iw`(inner word) + `"`(큰따옴표로). 커서는 `hello` 단어 위 또는 안에 있어야 합니다.

2. `cs"'` — `cs`(change surrounding) + `"`(이전) + `'`(새로운). 따옴표로 감싸인 텍스트 안 어디서든 동작합니다.

3. `ds'` — `ds`(delete surrounding) + `'`(제거할 둘러싸기 문자).

4. `yss)` — `yss`(you surround entire line) + `)`(공백 없는 닫는 괄호). `)`대신 `(`을 사용하면 `( hello world )`처럼 공백이 추가됩니다: `yss(`.

</details>

### 연습 3: vim-commentary

Python 함수를 편집하면서 코드를 주석 처리/해제해야 합니다:

```python
def process(data):
    result = transform(data)
    debug_log(result)
    return result
```

1. `debug_log` 줄에 커서가 있을 때, 그 줄만 주석 처리하는 방법은?
2. 모션을 사용해 전체 함수 본문(2–4번 줄)을 주석 처리하는 방법은?
3. 주석 처리 후 같은 줄들의 주석을 해제하려면?
4. `vim-repeat`가 `vim-commentary`와 함께 설치되면 무엇을 가능하게 합니까?

<details>
<summary>정답 보기</summary>

1. `gcc` — 현재 줄의 주석을 토글합니다. `debug_log(result)` 줄에서 실행하면 시작 부분에 `# `가 추가됩니다.

2. `result = transform(data)` 줄(본문 첫 번째 줄)에 커서를 놓고: `gc2j` — `gc`(주석 토글) + `2j`(아래 2줄을 포함하는 모션). 2, 3, 4번 줄이 주석 처리됩니다. 또는 세 줄을 `V2j`로 비주얼 선택 후 `gc`를 누릅니다.

3. 다시 `gc2j`(또는 줄별로 `gcc`, 또는 재선택 후 `gc`) — `gcc`/`gc`는 토글입니다: 이미 주석 처리된 줄이면 주석을 제거합니다.

4. `vim-repeat`는 `.` 명령이 마지막 `gcc` 또는 `gc{모션}` 동작을 반복하게 합니다. 없으면 `.`는 마지막 Vim 내장 동작만 반복하고 플러그인 동작은 반복하지 않습니다. 설치하면 한 줄을 주석 처리한 후 다른 줄로 이동하여 `.`만 눌러도 그 줄을 주석 처리할 수 있습니다.

</details>

### 연습 4: 플러그인 평가하기

동료가 "시각적 들여쓰기 가이드"를 보여주는 플러그인(VS Code의 들여쓰기 선 같은)을 설치하자고 제안합니다. 이 레슨의 플러그인 점검 질문을 적용하여 평가하세요:

1. Vim 내장 기능으로 할 수 있습니까?
2. 어떤 품질 신호를 확인하겠습니까?
3. 이 플러그인 추가의 주요 비용/위험은?
4. 설치하겠습니까? 결정을 정당화하세요.

<details>
<summary>정답 보기</summary>

이것은 열린 질문입니다 — 단일 정답이 없습니다. 모범 답안은 다음과 같습니다:

1. **내장 대안**: Vim은 `set list`로 공백 문자를 표시하고 `foldmethod=indent`로 들여쓰기 구조를 사용하지만, VS Code 같은 시각적 들여쓰기 가이드는 내장되어 있지 않습니다. 따라서 이 플러그인은 진정으로 새로운 기능을 제공합니다.

2. **확인할 품질 신호**:
   - GitHub 별점 (1000+ 이상이면 일반적으로 신뢰할 수 있음)
   - 마지막 커밋 날짜 (최근 12개월 내가 건강한 상태)
   - 열린 이슈와 유지 관리자의 응답 여부
   - 활발한 기여자 수

3. **주요 비용/위험**:
   - 소폭 시작 시간 증가 (들여쓰기 플러그인은 보통 무시할 수 있는 수준)
   - 모든 색상 테마나 터미널 에뮬레이터와 잘 동작하지 않을 수 있음
   - 유지 관리 및 업데이트가 필요한 플러그인 하나 추가

4. **결정 근거**: Python, JavaScript 등 깊게 중첩된 코드를 주로 작업하며 들여쓰기 수준 추적이 오류를 유발하는 경우 설치가 합리적입니다. `set number` + `set relativenumber`를 이미 사용하고 줄 번호로 탐색한다면 불필요할 수 있습니다. 핵심 원칙: 실제로 겪는 문제를 해결할 때만 설치하세요.

</details>

### 연습 5: 커스텀 명령어 작성하기

현재 줄 위에 주석 헤더 블록을 삽입하는 `Header`라는 Vimscript 커스텀 명령어를 작성하세요. 헤더는 다음과 같은 형태여야 합니다(`#` 주석을 사용하는 파일 유형):

```
# ============================================================
# Section: <텍스트 인수>
# ============================================================
```

여기서 `<텍스트 인수>`는 명령어에 전달된 인수입니다. 예를 들어 `:Header Utilities`는 다음을 삽입해야 합니다:

```
# ============================================================
# Section: Utilities
# ============================================================
```

<details>
<summary>정답 보기</summary>

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

설명:
- `command! -nargs=1 Header`는 정확히 1개의 인수를 받는 `Header` 명령어를 정의합니다
- `<q-args>`는 인수를 따옴표 문자열로 함수에 전달합니다
- `append(line('.') - 1, text)`는 현재 줄 앞에 줄을 삽입합니다
- `append(line('.'), text)`는 현재 줄 뒤에 삽입합니다 (위에 이미 삽입했으므로 위치가 이동됩니다)
- `a:title`은 `title`이라는 이름의 함수 인수에 접근합니다

`:normal`을 사용하는 더 간단한 대안:
```vim
command! -nargs=1 Header execute "normal! O# ============================================================\n# Section: " . <q-args> . "\n# ============================================================\<Esc>"
```

</details>

---

**이전**: [설정과 Vimrc](./12_Configuration_and_Vimrc.md) | **다음**: [Neovim과 현대적 워크플로우](./14_Neovim_and_Modern_Workflows.md)
