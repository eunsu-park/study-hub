# Neovim과 현대적 워크플로우

**이전**: [플러그인과 생태계](./13_Plugins_and_Ecosystem.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Neovim이 Vim과 어떻게 다른지, 각각 언제 선택해야 하는지 설명하기
2. Neovim의 기본 Lua 설정(`init.lua`) 작성하기
3. IDE 수준의 기능을 위한 LSP(Language Server Protocol) 설정하기
4. 고급 구문 하이라이팅과 텍스트 오브젝트를 위한 트리시터(Treesitter) 설정하기
5. 터미널 통합, Git, 디버깅을 갖춘 IDE 수준의 워크플로우 구축하기

---

Neovim은 2014년 Vim의 포크(fork)로 시작되었으며, 완전한 호환성을 유지하면서 코드베이스를 현대화하는 것을 목표로 했습니다. 오늘날 Neovim은 IDE 수준의 기능을 갖춘 모달 에디터를 원하는 개발자들이 선호하는 선택이 되었습니다. 레슨 1~13에서 배운 모든 내용은 Neovim에도 그대로 적용됩니다. 이 레슨은 Neovim이 그 위에 추가하는 것들을 다룹니다.

## 목차

1. [Neovim vs Vim](#1-neovim-vs-vim)
2. [Lua 설정 기초](#2-lua-설정-기초)
3. [LSP — Language Server Protocol](#3-lsp--language-server-protocol)
4. [트리시터(Treesitter)](#4-트리시터treesitter)
5. [터미널 통합](#5-터미널-통합)
6. [Git 통합](#6-git-통합)
7. [DAP — Debug Adapter Protocol](#7-dap--debug-adapter-protocol)
8. [완전한 IDE 환경 구축하기](#8-완전한-ide-환경-구축하기)
9. [요약](#9-요약)

---

## 1. Neovim vs Vim

### 주요 차이점

| 기능 | Vim | Neovim |
|---------|-----|--------|
| 스크립팅 | Vimscript (+ Vim9 script) | Vimscript + **Lua** (일급 지원) |
| LSP | 플러그인 경유 (coc.nvim) | **내장** LSP 클라이언트 |
| 트리시터 | 미지원 | **내장** |
| 터미널 | `:terminal` (Vim 8+) | 향상된 `:terminal` |
| 기본값 | 최소한 | 합리적인 기본값 |
| 설정 파일 | `~/.vimrc` | `~/.config/nvim/init.lua` |
| API | 제한적 | 풍부한 Lua API |
| UI | 터미널 전용 (GUI는 gVim) | 외부 UI 지원 (Neovide 등) |
| 커뮤니티 | 안정적, 느린 개발 | 활발함, 빠른 개발 |

### Vim을 선택하는 경우

- `vi`/`vim`만 사용 가능한 서버 환경
- 최신 기능보다 안정성을 선호하는 경우
- 기존 `.vimrc`가 완벽하게 작동하는 경우
- LSP, 트리시터 등 IDE 기능이 필요하지 않은 경우

### Neovim을 선택하는 경우

- 자동완성, 정의로 이동, 린팅 등 IDE 수준의 기능을 원하는 경우
- Vimscript보다 Lua를 선호하는 경우
- 최신 플러그인 생태계를 원하는 경우
- 레거시 설정 없이 새로 시작하는 경우

---

## 2. Lua 설정 기초

### 설정 파일 위치

```
~/.config/nvim/init.lua       " 메인 설정 파일
~/.config/nvim/lua/           " Lua 모듈 디렉토리
```

### Vimscript에서 Lua로 변환

```lua
-- 옵션 설정 (set number → vim.opt.number)
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

-- 전역 변수 (let g:mapleader = " " → vim.g.mapleader)
vim.g.mapleader = " "
vim.g.maplocalleader = ","

-- 키 매핑 (nnoremap → vim.keymap.set)
vim.keymap.set("n", "<C-h>", "<C-w>h", { desc = "Move to left window" })
vim.keymap.set("n", "<C-j>", "<C-w>j", { desc = "Move to below window" })
vim.keymap.set("n", "<C-k>", "<C-w>k", { desc = "Move to above window" })
vim.keymap.set("n", "<C-l>", "<C-w>l", { desc = "Move to right window" })

-- 검색 하이라이팅 지우기
vim.keymap.set("n", "<Esc>", "<cmd>nohlsearch<CR>")
```

### 모듈식 설정

설정을 모듈로 분리하세요:

```
~/.config/nvim/
├── init.lua              " 진입점
└── lua/
    ├── options.lua       " vim.opt 설정
    ├── keymaps.lua       " 키 매핑
    └── plugins/
        ├── init.lua      " 플러그인 매니저 설정
        ├── lsp.lua       " LSP 설정
        └── treesitter.lua
```

`init.lua`에서:
```lua
require("options")
require("keymaps")
require("plugins")
```

### Lua에서 자동명령(Autocommand) 사용

```lua
-- 자동명령 그룹 생성
local augroup = vim.api.nvim_create_augroup("MyGroup", { clear = true })

-- 저장 시 후행 공백 제거
vim.api.nvim_create_autocmd("BufWritePre", {
  group = augroup,
  pattern = "*",
  callback = function()
    vim.cmd([[%s/\s\+$//e]])
  end,
})

-- 복사한 텍스트 하이라이팅
vim.api.nvim_create_autocmd("TextYankPost", {
  group = augroup,
  callback = function()
    vim.highlight.on_yank({ timeout = 200 })
  end,
})
```

---

## 3. LSP — Language Server Protocol

LSP(Language Server Protocol)는 Neovim에 자동완성, 정의로 이동, 호버 문서, 이름 변경, 진단 등 IDE 기능을 제공합니다.

### LSP 동작 방식

```
┌─────────────┐         ┌──────────────────┐
│   Neovim    │◀───────▶│  Language Server  │
│  (client)   │  JSON   │  (pyright, tsserver, │
│             │   RPC   │   gopls, clangd...)  │
└─────────────┘         └──────────────────────┘
```

Neovim은 코드를 이해하는 외부 언어 서버와 통신합니다.

### mason.nvim + nvim-lspconfig로 설정하기

```lua
-- 플러그인 목록 (lazy.nvim 사용 시)
{
  "williamboman/mason.nvim",          -- 언어 서버 설치
  "williamboman/mason-lspconfig.nvim", -- mason ↔ lspconfig 브리지
  "neovim/nvim-lspconfig",            -- LSP 클라이언트 설정
}
```

```lua
-- 설정
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

-- 각 서버 설정
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

### LSP 키 바인딩

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

### nvim-cmp로 자동완성 설정

```lua
{
  "hrsh7th/nvim-cmp",
  dependencies = {
    "hrsh7th/cmp-nvim-lsp",    -- LSP 소스
    "hrsh7th/cmp-buffer",       -- 버퍼 단어
    "hrsh7th/cmp-path",         -- 파일 경로
    "L3MON4D3/LuaSnip",        -- 스니펫 엔진
    "saadparwaiz1/cmp_luasnip", -- 스니펫 소스
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

## 4. 트리시터(Treesitter)

트리시터(Treesitter)는 코드의 실제 파스 트리(parse tree)를 구축하여 정확한 구문 하이라이팅, 코드 폴딩, 텍스트 오브젝트를 제공합니다.

### 설정

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

### 트리시터 텍스트 오브젝트

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
            ["af"] = "@function.outer",  -- 함수 바깥쪽 선택
            ["if"] = "@function.inner",  -- 함수 안쪽 선택
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

이 텍스트 오브젝트를 활용하면:
- `daf` — 함수 전체 삭제 (함수 전체를!)
- `vif` — 함수 본문 안쪽 선택
- `daa` — 매개변수 삭제
- `]f` / `[f` — 함수 간 이동

---

## 5. 터미널 통합

Neovim의 내장 터미널은 에디터를 떠나지 않고도 셸 명령을 실행할 수 있게 해줍니다.

### 기본 터미널

```vim
:terminal           " 현재 창에서 터미널 열기
:split | terminal   " 가로 분할로 터미널 열기
:vsplit | terminal  " 세로 분할로 터미널 열기
```

### 터미널 모드

터미널 모드에서는 셸과 직접 상호작용합니다. 일반(Normal) 모드로 돌아가려면:

| 키 | 동작 |
|-----|--------|
| `Ctrl-\` `Ctrl-n` | 터미널 모드 종료 → 일반 모드 |
| `i` 또는 `a` | 터미널 모드 진입 (터미널 버퍼에서 일반 모드로 있을 때) |

### toggleterm.nvim

더 나은 터미널 경험을 위한 인기 플러그인:

```lua
{
  "akinsho/toggleterm.nvim",
  config = function()
    require("toggleterm").setup({
      open_mapping = [[<C-\>]],
      direction = "float",      -- 또는 "horizontal", "vertical"
      float_opts = { border = "curved" },
    })
  end,
}
```

### 코드 실행

```lua
-- 파일 타입에 따라 현재 파일 실행
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

## 6. Git 통합

### vim-fugitive (Vim과 Neovim 모두에서 작동)

필수 Git 플러그인입니다 ([레슨 13](./13_Plugins_and_Ecosystem.md) 참조).

### gitsigns.nvim (Neovim)

줄 단위 Git 변경 사항을 표시하고 헝크(hunk) 작업을 제공합니다:

```lua
{
  "lewis6991/gitsigns.nvim",
  config = function()
    require("gitsigns").setup({
      on_attach = function(bufnr)
        local gs = require("gitsigns")
        local opts = { buffer = bufnr }

        vim.keymap.set("n", "]h", gs.next_hunk, opts)        -- 다음 변경
        vim.keymap.set("n", "[h", gs.prev_hunk, opts)        -- 이전 변경
        vim.keymap.set("n", "<leader>hs", gs.stage_hunk, opts) -- 헝크 스테이지
        vim.keymap.set("n", "<leader>hr", gs.reset_hunk, opts) -- 헝크 리셋
        vim.keymap.set("n", "<leader>hp", gs.preview_hunk, opts) -- 미리보기
        vim.keymap.set("n", "<leader>hb", gs.blame_line, opts)  -- 블레임
      end,
    })
  end,
}
```

---

## 7. DAP — Debug Adapter Protocol

Neovim은 DAP(Debug Adapter Protocol)를 통해 디버깅을 지원합니다.

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

    -- UI 자동 열기/닫기
    dap.listeners.after.event_initialized["dapui_config"] = dapui.open
    dap.listeners.before.event_terminated["dapui_config"] = dapui.close

    -- 키 매핑
    vim.keymap.set("n", "<F5>", dap.continue)
    vim.keymap.set("n", "<F10>", dap.step_over)
    vim.keymap.set("n", "<F11>", dap.step_into)
    vim.keymap.set("n", "<F12>", dap.step_out)
    vim.keymap.set("n", "<leader>db", dap.toggle_breakpoint)
  end,
}
```

### Python 디버깅

```lua
-- debugpy 설치: pip install debugpy
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

## 8. 완전한 IDE 환경 구축하기

### 전체 스택 구성

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

### 권장 완전 설정

완전히 주석이 달린 Neovim 설정 예제는 `examples/VIM/09_init_lua.lua`를 참조하세요.

### 배포판 대안

미리 설정된 환경을 원한다면:

| 배포판 | 설명 |
|-------------|-------------|
| **LazyVim** | 기능이 풍부하고 잘 관리됨 |
| **NvChad** | 아름다운 기본 설정 |
| **AstroNvim** | 커뮤니티 주도 |
| **kickstart.nvim** | 최소한의 시작점 (학습용으로 권장) |

이들은 커스터마이즈할 수 있는 시작 설정을 제공합니다. 그러나 직접 설정을 구축하면 모든 동작 원리를 더 잘 이해할 수 있습니다.

---

## 9. 요약

| 기능 | 핵심 구성 요소 |
|---------|---------------|
| 설정 | `init.lua`, `vim.opt`, `vim.keymap.set`, `vim.api` |
| LSP | mason.nvim + nvim-lspconfig + nvim-cmp |
| 트리시터 | nvim-treesitter + textobjects |
| 퍼지 검색 | telescope.nvim |
| Git | vim-fugitive + gitsigns.nvim |
| 터미널 | 내장 `:terminal` 또는 toggleterm.nvim |
| 디버그 | nvim-dap + nvim-dap-ui |
| 파일 탐색기 | nvim-tree.lua |

### Neovim 학습 경로

1. **핵심 Vim 마스터하기** (레슨 1~12) — 이것이 기반입니다
2. **Neovim으로 전환** — Vim 지식이 그대로 이전됩니다
3. **Lua 기초 학습** — `init.lua`를 작성할 수 있을 정도면 충분합니다
4. **LSP 추가** — 가장 큰 삶의 질 향상
5. **트리시터 추가** — 더 나은 하이라이팅과 텍스트 오브젝트
6. **telescope 추가** — 모든 것을 퍼지 검색으로
7. **점진적 커스터마이즈** — 필요를 파악하면서 플러그인 추가

---

## 연습 문제

### 연습 1: Vim vs Neovim 선택

각 시나리오에서 클래식 Vim과 Neovim 중 어느 것이 더 나은 선택인지 결정하고 이유를 설명하세요:

1. 원격 Linux 서버를 관리하며 `/etc/`의 설정 파일을 편집해야 합니다. 서버에는 `vim`만 설치되어 있습니다.
2. 로컬 머신에서 Python 개발을 하며 정의로 이동, 인라인 에러 진단, 자동완성이 필요합니다.
3. 2년에 걸쳐 구축한 완벽하게 동작하는 600줄짜리 `.vimrc`가 있습니다.
4. Vim을 처음 시작하며 기존 설정이 없습니다.

<details>
<summary>정답 보기</summary>

1. **Vim** — 선택의 여지가 없습니다: `vim`만 사용 가능합니다. 이것이 Vim 지식이 필수인 가장 흔한 실제 시나리오입니다. 레슨 1~12의 모든 내용이 그대로 적용됩니다.

2. **Neovim** — 이것이 바로 Neovim이 내장 LSP 클라이언트로 기본 제공하는 IDE 기능들입니다. Vim에서 동일한 기능을 구현하려면 무거운 `coc.nvim` 플러그인이 필요하며, 설정과 유지 관리가 더 복잡합니다.

3. **둘 다 가능하지만 Vim이 실용적** — 기존 `.vimrc`는 Vim과 Neovim 모두에서 작동합니다(Neovim은 하위 호환성이 있습니다). 하지만 완벽하게 작동하고 전환할 강력한 이유가 없다면 Vim을 유지하는 것이 실용적인 선택입니다. 점진적으로 Neovim으로 마이그레이션할 수 있습니다.

4. **Neovim** — 레거시 설정 없이 새로 시작할 때 Neovim이 더 나은 선택입니다. 합리적인 기본값, 풍부한 생태계, 일급 설정 언어로서의 Lua를 갖추고 있습니다. 처음부터 `init.lua`와 현대적인 플러그인으로 시작하면 장기적으로 더 나은 경험을 할 수 있습니다.

</details>

### 연습 2: Vimscript에서 Lua로 변환

각 Vimscript 줄을 Neovim의 `init.lua`에 맞는 Lua 형식으로 변환하세요:

1. `set number`
2. `set tabstop=2`
3. `let g:mapleader = ","`
4. `nnoremap <leader>w :w<CR>`
5. `set clipboard=unnamedplus`

<details>
<summary>정답 보기</summary>

1. `vim.opt.number = true`

2. `vim.opt.tabstop = 2`

3. `vim.g.mapleader = ","`

4. `vim.keymap.set("n", "<leader>w", "<cmd>w<CR>")`
   — Lua 매핑에서는 `:...<CR>` 대신 `<cmd>...<CR>` 형식을 권장합니다. `<cmd>`는 커맨드라인 모드에 진입/종료하지 않기 때문입니다.

5. `vim.opt.clipboard = "unnamedplus"`

변환 패턴 요약:
- `set {option}` → `vim.opt.{option} = true`
- `set {option}={value}` → `vim.opt.{option} = value`
- `let g:{var} = value` → `vim.g.{var} = value`
- `nnoremap {lhs} {rhs}` → `vim.keymap.set("n", "{lhs}", "{rhs}")`

</details>

### 연습 3: LSP 워크플로우

LSP 설정 체인에서 각 구성 요소의 역할을 설명하세요:

```
mason.nvim → mason-lspconfig.nvim → nvim-lspconfig → nvim-cmp
```

1. `mason.nvim`은 무엇을 합니까?
2. `mason-lspconfig.nvim`은 무엇을 합니까?
3. `nvim-lspconfig`은 무엇을 합니까?
4. `nvim-cmp`는 무엇을 하며, 왜 LSP 클라이언트와 별개의 플러그인입니까?

<details>
<summary>정답 보기</summary>

1. **mason.nvim** — 개발 도구를 위한 패키지 매니저입니다. `pyright`, `ts_ls`, `clangd` 같은 언어 서버, 린터, 포맷터를 다운로드하고 설치합니다. 에디터 도구를 위한 `apt`나 `brew`라고 생각하면 됩니다. Neovim이 관리하는 디렉토리에 저장합니다.

2. **mason-lspconfig.nvim** — mason과 lspconfig 사이의 브리지입니다. 없으면 두 도구에서 올바른 서버 이름을 수동으로 맞춰야 합니다(두 도구가 다른 이름을 사용하는 경우가 있음). `ensure_installed`를 통해 Neovim 시작 시 나열된 서버를 자동으로 설치할 수도 있습니다.

3. **nvim-lspconfig** — 특정 언어 서버와 통신하도록 Neovim의 내장 LSP 클라이언트를 설정합니다. 각 서버에 합리적인 기본 설정을 제공하고 활성화를 위한 간단한 API(`lspconfig.pyright.setup({})`)를 제공합니다.

4. **nvim-cmp** — 제안 목록을 표시하는 자동완성 엔진입니다. LSP 클라이언트는 언어 서버에서 원시 자동완성 데이터를 받지만 이를 표시하는 UI가 없습니다. nvim-cmp가 시각적 자동완성 메뉴를 제공하고 사용자 상호작용(수락, 항목 탐색)을 처리합니다. 별개의 플러그인인 이유는 LSP, 버퍼 단어, 파일 경로, 스니펫 등 여러 소스의 자동완성을 같은 메뉴에 표시할 수 있기 때문입니다 — nvim-cmp가 이 모든 것을 집계합니다.

</details>

### 연습 4: 트리시터 텍스트 오브젝트

이 레슨에서 설정한 트리시터 텍스트오브젝트 플러그인으로 Python 파일을 편집할 때 각 명령이 무엇을 하는지 설명하세요:

```python
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
```

1. 커서가 `return a + b` 줄에 있을 때 `daf`를 누릅니다.
2. 커서가 `def multiply` 줄에 있을 때 `vif`를 누릅니다.
3. 커서가 `def add(self, a, b)` 안의 `a`에 있을 때 `daa`를 누릅니다.
4. 커서가 `add` 안 어딘가에 있을 때 `]f`를 누릅니다.

<details>
<summary>정답 보기</summary>

1. **`daf`**(delete around function) — `def` 줄, 본문, 주변 빈 줄을 포함한 `add` 메서드 전체를 삭제합니다. 파일에는 `Calculator` 안에 `multiply` 메서드만 남습니다.

2. **`vif`**(visual select inside function) — `multiply`의 본문, 즉 `return a * b`만 비주얼 선택합니다(`def` 줄 제외, 함수 내부의 줄들만).

3. **`daa`**(delete around parameter) — 함수 시그니처에서 매개변수 `a`를 쉼표 구분자와 함께 삭제합니다. 함수 시그니처가 `def add(self, b):`가 됩니다.

4. **`]f`**(jump to next function) — 커서를 다음 함수 정의의 시작으로 이동합니다. `def multiply(self, a, b):`로 이동합니다.

</details>

### 연습 5: IDE 기능 비교

VS Code를 사용하는 동료에게 Neovim의 IDE 기능을 설명하려 합니다. 각 VS Code 기능을 Neovim의 해당 기능(플러그인/명령)과 매칭하세요:

| VS Code 기능 | Neovim 해당 기능 |
|----------------|-------------------|
| 정의로 이동 (F12) | ? |
| 호버 문서 보기 (Ctrl+K, Ctrl+I) | ? |
| 심볼 이름 변경 (F2) | ? |
| 모든 참조 보기 | ? |
| 다음 진단 오류로 이동 | ? |
| 통합 터미널 열기 (Ctrl+`) | ? |

<details>
<summary>정답 보기</summary>

| VS Code 기능 | Neovim 해당 기능 |
|----------------|-------------------|
| 정의로 이동 (F12) | `gd` — `vim.lsp.buf.definition` |
| 호버 문서 보기 (Ctrl+K, Ctrl+I) | `K` — `vim.lsp.buf.hover` |
| 심볼 이름 변경 (F2) | `<leader>rn` — `vim.lsp.buf.rename` |
| 모든 참조 보기 | `gr` — `vim.lsp.buf.references` |
| 다음 진단 오류로 이동 | `]d` — `vim.diagnostic.goto_next` |
| 통합 터미널 열기 (Ctrl+`) | `:terminal` 또는 toggleterm.nvim으로 `<C-\>` |

이 모든 기능은 이 레슨의 `LspAttach` 자동명령에서 설정한 키 바인딩으로 Neovim의 내장 LSP 클라이언트가 제공합니다. 위의 키 바인딩은 레슨의 컨벤션을 따르며, 실제 설정에서는 다른 키를 사용할 수 있습니다.

</details>

---

**이전**: [플러그인과 생태계](./13_Plugins_and_Ecosystem.md)
