# 설정과 Vimrc

**이전**: [명령줄과 고급 기능](./11_Command_Line_and_Advanced_Features.md) | **다음**: [플러그인과 생태계](./13_Plugins_and_Ecosystem.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 필수 설정을 포함한 `.vimrc` 파일을 생성하고 구성하기
2. 화면 표시 옵션, 검색 동작, 편집 환경 설정하기
3. `map`, `nmap`, `imap`으로 커스텀 키 매핑(Key Mapping)을 만들고 `<leader>`를 활용하기
4. 파일 유형별 동작을 위한 자동 명령(autocommand, `autocmd`) 작성하기
5. 파일 유형 감지(filetype detection)를 사용해 언어별로 Vim 환경 커스터마이징하기

---

> **비유 — 자동차 대시보드 커스터마이징**: `.vimrc`는 자동차 대시보드를 내 취향대로 꾸미는 것과 같습니다. 어떤 계기판을 표시할지 선택하고(줄 번호, 상태 바), 컨트롤을 자신의 작업 방식에 맞게 재배치하며(키 매핑), 자동 동작을 설정합니다(autocmd — 자동 전조등처럼). 모든 사람의 `.vimrc`는 저마다의 작업 흐름을 반영하기 때문에 제각각 다릅니다.

기본 상태의 Vim은 매우 단순합니다. 잘 구성된 `.vimrc`는 Vim을 기초적인 편집기에서 편안한 개발 환경으로 탈바꿈시킵니다. 이 레슨에서는 처음부터 자신만의 설정을 구축하는 방법을 배웁니다.

## 목차

1. [.vimrc 파일](#1-vimrc-파일)
2. [필수 설정](#2-필수-설정)
3. [키 매핑](#3-키-매핑)
4. [리더 키](#4-리더-키)
5. [자동 명령](#5-자동-명령)
6. [파일 유형 설정](#6-파일-유형-설정)
7. [상태 줄](#7-상태-줄)
8. [색상 테마](#8-색상-테마)
9. [점진적으로 .vimrc 구성하기](#9-점진적으로-vimrc-구성하기)
10. [요약](#10-요약)

---

## 1. .vimrc 파일

### 위치

| 편집기 | 파일 | 위치 |
|--------|------|----------|
| Vim | `.vimrc` | `~/.vimrc` 또는 `~/.vim/vimrc` |
| Neovim | `init.vim` | `~/.config/nvim/init.vim` |
| Neovim (Lua) | `init.lua` | `~/.config/nvim/init.lua` |

### 첫 번째 .vimrc 만들기

```bash
touch ~/.vimrc       # 파일 생성
vim ~/.vimrc         # 편집
```

### 재시작 없이 리로드하기

```vim
:source ~/.vimrc     " vimrc 다시 불러오기
:so %                " .vimrc 파일을 편집 중일 때 (% = 현재 파일)
```

### 주석(Comments)

```vim
" Vimscript의 주석입니다
" 주석은 큰따옴표로 시작합니다
```

---

## 2. 필수 설정

### 화면 표시 설정

```vim
set number              " 줄 번호 표시
set relativenumber      " 상대적 줄 번호 (number와 함께 사용하면 하이브리드 모드)
set cursorline          " 현재 줄 강조
set showmatch           " 대응하는 괄호 강조
set showcmd             " 상태 바에 부분 명령어 표시
set showmode            " 현재 모드 표시 (INSERT, VISUAL 등)
set laststatus=2        " 상태 줄 항상 표시
set ruler               " 상태 줄에 커서 위치 표시
set signcolumn=yes      " 기호 열 항상 표시 (텍스트 밀림 방지)
set colorcolumn=80      " 80자 위치에 열 가이드 표시
set wrap                " 긴 줄을 시각적으로 줄바꿈
set linebreak           " 단어 경계에서 줄바꿈 (단어 중간에서 자르지 않음)
set scrolloff=8         " 커서 위아래로 8줄 항상 표시
set sidescrolloff=8     " 커서 좌우로 8열 항상 표시
```

### 검색 설정

```vim
set hlsearch            " 검색 결과 강조
set incsearch           " 점진적 검색 (입력하면서 결과 표시)
set ignorecase          " 대소문자 구분 없이 검색
set smartcase           " 패턴에 대문자가 있으면 대소문자 구분
```

### 편집 설정

```vim
set tabstop=4           " 탭을 4칸 공백으로 표시
set shiftwidth=4        " 들여쓰기 시 4칸 공백 사용
set expandtab           " 탭 대신 공백 사용
set softtabstop=4       " Tab 키를 누르면 4칸 공백 삽입
set autoindent          " 현재 줄의 들여쓰기를 새 줄에 복사
set smartindent         " 스마트 자동 들여쓰기
set backspace=indent,eol,start  " Backspace를 기대한 대로 동작시킴
```

### 파일 동작

```vim
set encoding=utf-8      " UTF-8 인코딩
set fileencoding=utf-8  " 파일을 UTF-8로 저장
set hidden              " 저장하지 않은 버퍼에서도 다른 파일로 전환 허용
set autoread            " Vim 외부에서 변경된 파일 자동 리로드
set noswapfile          " 스왑 파일 생성 안 함
set nobackup            " 백업 파일 생성 안 함
set undofile            " 영구 실행 취소 (Vim 종료 후에도 유지)
set undodir=~/.vim/undodir  " 실행 취소 파일 저장 디렉토리
```

### UI 동작

```vim
set wildmenu            " 향상된 명령줄 자동완성
set wildmode=list:longest,full  " 자동완성 동작 방식
set mouse=a             " 마우스 지원 활성화 (모든 모드)
set splitbelow          " 수평 분할 시 아래에 새 창 열기
set splitright          " 수직 분할 시 오른쪽에 새 창 열기
set updatetime=300      " 빠른 자동완성 (기본값: 4000ms)
set timeoutlen=500      " 매핑 타임아웃 (밀리초)
```

---

## 3. 키 매핑

### 매핑 명령어

| 명령어 | 적용 범위 | 재귀적 |
|---------|-------|-----------|
| `map` | 일반, 비주얼, 연산자 대기 모드 | 예 |
| `nmap` | 일반 모드만 | 예 |
| `imap` | 입력 모드만 | 예 |
| `vmap` | 비주얼 모드만 | 예 |
| `noremap` | 일반, 비주얼, 연산자 대기 모드 | **아니오** |
| `nnoremap` | 일반 모드만 | **아니오** |
| `inoremap` | 입력 모드만 | **아니오** |
| `vnoremap` | 비주얼 모드만 | **아니오** |

**항상 `noremap` 계열을 사용하세요** — 재귀적 매핑으로 인한 예상치 못한 동작을 방지합니다.

### 문법

```vim
nnoremap {키} {동작}
```

### 실용적인 매핑

```vim
" 빠른 저장
nnoremap <C-s> :w<CR>
inoremap <C-s> <Esc>:w<CR>a

" 빠른 종료
nnoremap <leader>q :q<CR>

" 검색 강조 지우기
nnoremap <Esc><Esc> :nohlsearch<CR>

" 줄 위아래로 이동
nnoremap <A-j> :m .+1<CR>==
nnoremap <A-k> :m .-2<CR>==
vnoremap <A-j> :m '>+1<CR>gv=gv
vnoremap <A-k> :m '<-2<CR>gv=gv

" 더 나은 창 이동
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" 버퍼 탐색
nnoremap [b :bprev<CR>
nnoremap ]b :bnext<CR>

" 들여쓰기 후 비주얼 선택 유지
vnoremap < <gv
vnoremap > >gv

" Y가 D, C처럼 동작하게 설정 (줄 끝까지 복사)
nnoremap Y y$

" 점프 후 화면 가운데 정렬
nnoremap n nzzzv
nnoremap N Nzzzv
nnoremap <C-d> <C-d>zz
nnoremap <C-u> <C-u>zz
```

### 특수 키 표기

| 표기 | 키 |
|----------|-----|
| `<CR>` | Enter/Return |
| `<Esc>` | Escape |
| `<Tab>` | Tab |
| `<BS>` | Backspace |
| `<Space>` | 스페이스바 |
| `<C-x>` | Ctrl+x |
| `<A-x>` 또는 `<M-x>` | Alt+x |
| `<S-x>` | Shift+x |
| `<leader>` | 리더 키 (기본값: `\`) |
| `<Nop>` | 아무 동작 없음 (키 비활성화) |

### 매핑 제거

```vim
nunmap {키}          " 일반 모드 매핑 제거
iunmap {키}          " 입력 모드 매핑 제거
mapclear               " 모든 매핑 제거
```

---

## 4. 리더 키

리더 키(Leader Key)는 커스텀 단축키의 접두사로, 내장 명령어와의 충돌을 방지합니다.

### 리더 키 설정

```vim
let mapleader = " "           " 스페이스를 리더로 설정 (매우 일반적)
let maplocalleader = ","      " 파일 유형별 매핑을 위한 로컬 리더
```

### 리더 매핑 예시

```vim
" 파일 작업
nnoremap <leader>w :w<CR>               " 저장
nnoremap <leader>q :q<CR>               " 종료
nnoremap <leader>x :x<CR>               " 저장 후 종료

" 버퍼 작업
nnoremap <leader>b :ls<CR>:b<Space>     " 버퍼 목록 표시 후 전환
nnoremap <leader>d :bd<CR>              " 버퍼 닫기

" 검색
nnoremap <leader>h :nohlsearch<CR>      " 강조 지우기

" 창 작업
nnoremap <leader>v :vsplit<CR>          " 수직 분할
nnoremap <leader>s :split<CR>           " 수평 분할

" vimrc 빠른 편집
nnoremap <leader>ev :edit $MYVIMRC<CR>
nnoremap <leader>sv :source $MYVIMRC<CR>
```

---

## 5. 자동 명령

자동 명령(Autocommand)은 특정 이벤트에 반응하여 명령을 자동으로 실행합니다.

### 문법

```vim
autocmd {이벤트} {패턴} {명령}
```

### 주요 이벤트

| 이벤트 | 발생 시점 |
|-------|---------|
| `BufRead`, `BufNewFile` | 파일을 열 때 |
| `BufWritePre` | 파일을 저장하기 전 |
| `BufWritePost` | 파일을 저장한 후 |
| `FileType` | 파일 유형이 감지될 때 |
| `VimEnter` | Vim 시작 후 |
| `VimLeave` | Vim 종료 전 |
| `InsertEnter` | 입력 모드 진입 시 |
| `InsertLeave` | 입력 모드 종료 시 |

### 실용적인 자동 명령

```vim
" 저장 시 후행 공백 제거
autocmd BufWritePre * :%s/\s\+$//e

" 파일을 열 때 마지막 편집 위치로 돌아가기
autocmd BufReadPost * if line("'\"") > 0 && line("'\"") <= line("$")
    \ | execute "normal! g'\"" | endif

" 터미널 크기 변경 시 분할 창 자동 조정
autocmd VimResized * wincmd =

" 복사한 텍스트를 잠깐 강조
autocmd TextYankPost * silent! lua vim.highlight.on_yank()
```

### 자동 명령 그룹

리로드 시 중복을 방지하기 위해 자동 명령을 그룹으로 묶으세요:

```vim
augroup MyAutocommands
    autocmd!                          " 그룹을 먼저 초기화
    autocmd BufWritePre * :%s/\s\+$//e
    autocmd FileType python setlocal tabstop=4
    autocmd FileType javascript setlocal tabstop=2
augroup END
```

시작 부분의 `autocmd!`는 그룹을 초기화하여, `.vimrc`를 다시 소스할 때 명령이 중복 등록되는 것을 방지합니다.

---

## 6. 파일 유형 설정

### 파일 유형 감지 활성화

```vim
filetype on            " 파일 유형 감지 활성화
filetype plugin on     " 파일 유형별 플러그인 로드
filetype indent on     " 파일 유형별 들여쓰기 로드
" 또는 한 번에:
filetype plugin indent on
```

### 파일 유형별 설정

```vim
augroup FileTypeSettings
    autocmd!
    " Python: 4칸 탭
    autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab

    " JavaScript/TypeScript: 2칸 탭
    autocmd FileType javascript,typescript setlocal tabstop=2 shiftwidth=2 expandtab

    " Go: 실제 탭 문자 사용
    autocmd FileType go setlocal tabstop=4 shiftwidth=4 noexpandtab

    " Markdown: 줄바꿈 및 맞춤법 검사
    autocmd FileType markdown setlocal wrap linebreak spell

    " Makefile: 탭 문자 필수
    autocmd FileType make setlocal noexpandtab
augroup END
```

### 파일 유형 플러그인 파일

파일 유형별로 광범위한 설정이 필요할 경우 다음 위치에 파일을 만드세요:

```
~/.vim/ftplugin/python.vim     " Python 파일에 대해 로드됨
~/.vim/ftplugin/javascript.vim " JavaScript 파일에 대해 로드됨
```

`~/.vim/ftplugin/python.vim` 내용 예시:
```vim
setlocal tabstop=4
setlocal shiftwidth=4
setlocal expandtab
setlocal colorcolumn=79
nnoremap <buffer> <leader>r :!python %<CR>
```

---

## 7. 상태 줄

### 기본 커스텀 상태 줄

```vim
set laststatus=2     " 상태 줄 항상 표시

set statusline=
set statusline+=%#PmenuSel#
set statusline+=\ %M              " 수정 플래그
set statusline+=\ %f              " 파일 이름
set statusline+=%=                " 여기서부터 오른쪽 정렬
set statusline+=\ %y              " 파일 유형
set statusline+=\ %{&encoding}    " 인코딩
set statusline+=\ %p%%            " 파일 내 위치 (퍼센트)
set statusline+=\ %l:%c           " 줄:열
set statusline+=\
```

### 상태 줄 구성 요소

| 구성 요소 | 표시 내용 |
|-----------|---------|
| `%f` | 파일 경로 (상대 경로) |
| `%F` | 파일 경로 (절대 경로) |
| `%t` | 파일 이름만 |
| `%m` | 수정 플래그 `[+]` |
| `%r` | 읽기 전용 플래그 `[RO]` |
| `%y` | 파일 유형 `[python]` |
| `%l` | 현재 줄 |
| `%c` | 현재 열 |
| `%p` | 파일 내 위치 (퍼센트) |
| `%=` | 구분자 (좌/우 정렬 기준) |

대부분의 사용자는 상태 줄 플러그인(`lualine` 또는 `lightline`)을 선호합니다 — [레슨 13](./13_Plugins_and_Ecosystem.md)을 참고하세요.

---

## 8. 색상 테마

### 내장 색상 테마

```vim
:colorscheme desert    " desert 테마 적용
:colorscheme slate     " slate 테마 적용
:colorscheme default   " 기본 테마로 초기화
```

사용 가능한 테마 목록:
```vim
:colorscheme <Tab>     " Tab으로 사용 가능한 테마 탐색
```

### 트루 컬러(True Color) 지원

```vim
if has('termguicolors')
    set termguicolors    " 24비트 RGB 색상 활성화
endif
```

### 인기 색상 테마 (플러그인으로 설치)

- **gruvbox** — 따뜻하고 복고풍의 팔레트
- **tokyonight** — 현대적인 다크/라이트 테마
- **catppuccin** — 파스텔 색상
- **onedark** — Atom에서 영감을 받은 테마
- **nord** — 차갑고 파란빛의 Arctic 테마

### 문법 강조(Syntax Highlighting)

```vim
syntax on              " 문법 강조 활성화
syntax enable          " 동일하지만 사용자 색상 설정 유지
```

---

## 9. 점진적으로 .vimrc 구성하기

### 접근 방법

1. **최소한으로 시작하기** — 다른 사람의 500줄짜리 설정을 그대로 복사하지 마세요
2. **필요할 때 설정 추가하기** — 불편한 점이 생기면 그때 해결하세요
3. **모든 것을 이해하기** — 이해하지 못한 줄은 절대 추가하지 마세요
4. **변경 사항 주석 달기** — 미래의 자신이 현재의 자신에게 감사할 것입니다

### 시작용 .vimrc

`examples/VIM/06_minimal_vimrc.vim` 파일에서 상세한 주석이 달린 최소 설정을 확인하세요.

### 흔히 저지르는 실수

1. **거대한 설정 복사** — 절반도 이해하지 못한 채 사용하게 됩니다
2. **너무 이른 플러그인 남발** — 먼저 Vim의 핵심 기능부터 익히세요
3. **Vim의 본성에 저항하기** — Vim을 VS Code처럼 만들려고 하지 마세요
4. **autocmd에 그룹 미사용** — 명령 중복 등록으로 이어집니다

---

## 10. 요약

| 분류 | 핵심 개념 |
|----------|-------------|
| 파일 | `~/.vimrc`, `:source`, Vimscript 주석 (`"`) |
| 화면 | `number`, `relativenumber`, `cursorline`, `scrolloff` |
| 검색 | `hlsearch`, `incsearch`, `ignorecase`, `smartcase` |
| 편집 | `tabstop`, `shiftwidth`, `expandtab`, `autoindent` |
| 매핑 | `nnoremap`, `inoremap`, `<leader>`, `<CR>`, `<Nop>` |
| 자동 명령 | `autocmd`, `augroup`, `FileType`, `BufWritePre` |
| 파일 유형 | `filetype plugin indent on`, `setlocal`, `ftplugin/` |
| 외관 | `colorscheme`, `statusline`, `termguicolors` |

### 황금률

> `.vimrc`는 유기적으로 성장해야 합니다. 모든 줄은 실제로 겪은 문제를 해결해야 합니다.

---

## 연습 문제

### 연습 1: 설정 해석하기

아래 각 설정이 무엇을 하는지, 그리고 왜 필요한지 설명하세요:

1. `set relativenumber`
2. `set scrolloff=8`
3. `set undofile`
4. `set splitright`

<details>
<summary>정답 보기</summary>

1. `set relativenumber` — 커서 기준으로 상대적인 줄 번호를 표시합니다. 현재 줄에는 절대 번호가, 위아래 줄에는 거리(1, 2, 3...)가 표시됩니다. `5j`, `12k`, `d8j` 같은 수직 점프 명령을 사용할 때 점프 거리를 줄 번호에서 바로 읽을 수 있어 훨씬 편리합니다.

2. `set scrolloff=8` — 스크롤할 때 커서 위아래로 최소 8줄이 항상 보이게 합니다. 이를 통해 맥락을 항상 확인할 수 있습니다 — 커서가 화면 끝 모서리로 이동하지 않고 현재 줄 주변 내용을 볼 수 있습니다.

3. `set undofile` — 실행 취소 히스토리를 디스크에 저장합니다. 이 설정이 없으면 파일을 닫을 때 실행 취소 히스토리가 사라집니다. `undofile`을 사용하면 Vim을 재시작한 후에도 이전 세션에서 했던 변경사항을 되돌릴 수 있습니다.

4. `set splitright` — 수직 분할(`:vsp`) 시 새 창이 기본값인 왼쪽 대신 **오른쪽**에 열립니다. 좌→우 방향의 자연스러운 읽기 방향과 일치하여 더 직관적으로 느껴집니다.

</details>

### 연습 2: `nmap` vs `nnoremap`

1. `nmap`과 `nnoremap`의 핵심 차이점은 무엇입니까?
2. 다음 매핑에서 `nmap`을 사용하면 어떤 문제가 발생합니까?

```vim
nmap j gj
nmap gj j
```

3. 왜 `nnoremap`이 거의 항상 올바른 선택입니까?

<details>
<summary>정답 보기</summary>

1. `nmap`은 **재귀적(recursive)** 매핑을 만듭니다 — 오른쪽의 입력이 해석될 때 다른 매핑된 키도 따라갑니다. `nnoremap`은 **비재귀적(non-recursive)** 매핑을 만듭니다 — 오른쪽이 다른 매핑을 무시하고 Vim 명령 그대로 해석됩니다.

2. 위 두 `nmap` 정의로:
   - `j`를 누르면 `gj`가 실행됨
   - 하지만 `gj`도 `j`에 매핑되어 있음
   - `j`가 다시 `gj`를 실행함
   - 이것이 **무한 루프**를 만들어 Vim이 멈춥니다. `nmap`은 매핑을 재귀적으로 연결합니다.

3. `nnoremap`은 오른쪽의 다른 매핑을 무시하므로 어떤 일이 일어날지 항상 정확히 알 수 있습니다. 우발적인 체인과 무한 루프를 방지합니다. `nmap`을 의도적으로 사용하는 경우는 하나의 매핑이 다른 매핑을 명시적으로 트리거해야 할 때뿐인데, 이는 거의 필요 없고 디버그하기도 어렵습니다.

</details>

### 연습 3: 자동 명령 작성

각 작업에 대한 자동 명령(또는 자동 명령 그룹)을 작성하세요:

1. 모든 파일이 저장될 때마다 후행 공백(trailing whitespace)을 자동으로 제거
2. Markdown 파일 편집 시 줄바꿈(`wrap linebreak`)과 맞춤법 검사(`spell`) 활성화
3. Python 파일을 저장한 후 `:!python %`로 자동 실행

세 가지를 모두 적절한 `augroup` 블록으로 감싸세요.

<details>
<summary>정답 보기</summary>

```vim
augroup MyProductivity
    autocmd!
    " 1. 저장 시 후행 공백 제거 (모든 파일)
    autocmd BufWritePre * :%s/\s\+$//e

    " 2. Markdown: 줄바꿈 및 맞춤법 검사
    autocmd FileType markdown setlocal wrap linebreak spell

    " 3. Python 파일 저장 후 자동 실행
    autocmd BufWritePost *.py :!python %
augroup END
```

그룹 시작 부분의 `autocmd!`는 `MyProductivity`에 이전에 등록된 모든 자동 명령을 지웁니다. 이를 통해 `.vimrc`를 다시 소스할 때 중복 등록을 방지합니다 — 없으면 `:source`할 때마다 각 자동 명령의 복사본이 추가됩니다.

참고: 작업 3은 파일이 이미 저장된 후 Python이 실행되도록 `BufWritePost`(저장 후)를 사용합니다. `BufWritePre`(저장 전)를 사용하면 이전 버전의 파일이 실행됩니다.

</details>

### 연습 4: 리더 키 설계

개발자가 `let mapleader = " "`(스페이스를 리더로)를 설정하고 다음 매핑을 만들려 합니다:

1. `<leader>w` — 현재 파일 저장
2. `<leader>q` — 현재 버퍼 닫기 (창만 닫는 게 아님)
3. `<leader>ev` — `.vimrc` 파일 열어서 편집
4. `<leader>sv` — `.vimrc` 파일 다시 불러오기

네 가지를 모두 `nnoremap` 명령으로 작성하세요.

<details>
<summary>정답 보기</summary>

```vim
let mapleader = " "

nnoremap <leader>w :w<CR>
nnoremap <leader>q :bd<CR>
nnoremap <leader>ev :edit $MYVIMRC<CR>
nnoremap <leader>sv :source $MYVIMRC<CR>
```

참고:
- Ex 명령 뒤에 `<CR>`이 있어야 실행됩니다(Enter 키 누르기에 해당)
- `:w`는 파일 저장; `:bd`는 버퍼 삭제(닫기) — 창만 닫는 `:q`와 다름
- `$MYVIMRC`는 활성 vimrc 파일 경로를 담고 있는 Vim의 특수 변수
- 스페이스를 리더로 사용하는 것은 노멀 모드에서 기본적으로 아무 동작도 하지 않는 큰 키이기 때문에 편리합니다

</details>

### 연습 5: 파일 유형별 설정

세 가지 언어의 서로 다른 컨벤션을 가진 팀을 위한 `.vimrc`를 작성합니다:

- **Python**: 4칸 들여쓰기, `expandtab`, 79열 가이드
- **JavaScript**: 2칸 들여쓰기, `expandtab`
- **Go**: 실제 탭 문자, `noexpandtab`, 탭 너비 4

이 파일 유형별 설정을 위한 완전한 `augroup` 블록을 작성하세요.

<details>
<summary>정답 보기</summary>

```vim
augroup LanguageSettings
    autocmd!

    " Python: PEP 8 스타일
    autocmd FileType python setlocal
        \ tabstop=4
        \ shiftwidth=4
        \ softtabstop=4
        \ expandtab
        \ colorcolumn=79

    " JavaScript: 일반적인 JS 컨벤션
    autocmd FileType javascript,typescript setlocal
        \ tabstop=2
        \ shiftwidth=2
        \ softtabstop=2
        \ expandtab

    " Go: gofmt은 실제 탭을 사용
    autocmd FileType go setlocal
        \ tabstop=4
        \ shiftwidth=4
        \ noexpandtab

augroup END
```

핵심 포인트:
- `setlocal`은 설정을 현재 버퍼에만 적용합니다(전역으로 적용하는 `set`과 달리)
- 줄 계속을 위해 `\`로 시작하는 Vimscript 문법으로 여러 줄에 걸쳐 작성 가능
- `tabstop`, `shiftwidth`, `softtabstop`을 모두 설정하면 들여쓰기가 어떤 방식으로 트리거되든 일관된 동작이 보장됩니다

</details>

---

**이전**: [명령줄과 고급 기능](./11_Command_Line_and_Advanced_Features.md) | **다음**: [플러그인과 생태계](./13_Plugins_and_Ecosystem.md)
