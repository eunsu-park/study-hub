# VIM 학습 가이드

## 소개

VIM(Vi IMproved)은 역대 가장 강력하고 효율적인 텍스트 편집기 중 하나입니다. Unix 전통에서 탄생한 VIM의 모달 편집(Modal Editing) 철학은 그 언어를 익히고 나면 생각의 속도로 텍스트를 조작할 수 있게 해줍니다. 이 가이드는 첫 걸음부터 IDE 수준의 개발 워크플로우까지 안내합니다.

원격 서버에서 설정 파일을 빠르게 수정하거나, 코드를 효율적으로 리팩토링하거나, 완전히 커스터마이즈된 개발 환경을 구축할 때 모두 VIM 실력은 경력 전반에 걸쳐 큰 자산이 됩니다.

---

## 학습 로드맵

```
                        VIM 학습 경로
    ════════════════════════════════════════════════

    1단계: 생존           2단계: 효율
    ┌─────────────────┐        ┌─────────────────────┐
    │ L01 철학         │───────▶│ L04 모션             │
    │ L02 모드         │       │ L05 연산자            │
    │ L03 기본 편집    │       │ L06 텍스트 객체        │
    └─────────────────┘        │ L07 비주얼 모드        │
                               │ L08 검색 & 치환        │
                               └───────────┬───────────┘
                                           │
                               3단계: 숙달
                               ┌───────────┴───────────┐
                               │ L09 레지스터 & 매크로   │
                               │ L10 버퍼 & 창           │
                               │ L11 명령줄              │
                               │ L12 설정               │
                               │ L13 플러그인            │
                               │ L14 Neovim & 모던 워크플로│
                               └───────────────────────┘
```

---

## 사전 요구사항

- 기본 터미널/명령줄 사용법 ([Shell Script](../Shell_Script/00_Overview.md))
- 프로그래밍 언어 기본 지식 (도움이 되지만 필수는 아님)

---

## 파일 목록

### 1단계: 생존 (⭐)

| 파일 | 난이도 | 주요 주제 |
|------|--------|-----------|
| [01_Introduction_and_Philosophy.md](./01_Introduction_and_Philosophy.md) | ⭐ | 역사, 모달 편집, vi/vim/neovim, 설치, vimtutor |
| [02_Modes_and_Basic_Navigation.md](./02_Modes_and_Basic_Navigation.md) | ⭐ | 노멀/입력/명령줄 모드, hjkl, 모드 전환 |
| [03_Essential_Editing.md](./03_Essential_Editing.md) | ⭐ | 입력 명령, 삭제/복사/붙여넣기, 실행취소/다시실행, 저장/종료 |

### 2단계: 효율 (⭐⭐)

| 파일 | 난이도 | 주요 주제 |
|------|--------|-----------|
| [04_Motions_and_Navigation.md](./04_Motions_and_Navigation.md) | ⭐⭐ | 단어/줄/화면 모션, f/t 문자 검색, 점프 |
| [05_Operators_and_Composability.md](./05_Operators_and_Composability.md) | ⭐⭐ | 연산자 + 모션 문법, d/c/y 조합, 점 반복 |
| [06_Text_Objects.md](./06_Text_Objects.md) | ⭐⭐ | 내부/주변 객체, 단어/문장/문단/괄호/따옴표 |
| [07_Visual_Mode.md](./07_Visual_Mode.md) | ⭐⭐ | 문자/줄/블록 비주얼, 선택 + 연산자, 재선택 |
| [08_Search_and_Replace.md](./08_Search_and_Replace.md) | ⭐⭐ | 앞/뒤 검색, 치환, 정규식, 전역 명령 |

### 3단계: 숙달 (⭐⭐⭐)

| 파일 | 난이도 | 주요 주제 |
|------|--------|-----------|
| [09_Registers_Marks_and_Macros.md](./09_Registers_Marks_and_Macros.md) | ⭐⭐⭐ | 명명 레지스터, 마크, 매크로 녹화/재생, 클립보드 |
| [10_Buffers_Windows_and_Tabs.md](./10_Buffers_Windows_and_Tabs.md) | ⭐⭐⭐ | 버퍼 관리, 분할 창, 탭, 인자 목록 |
| [11_Command_Line_and_Advanced_Features.md](./11_Command_Line_and_Advanced_Features.md) | ⭐⭐⭐ | Ex 명령, 외부 명령, 범위, 폴드, 세션 |
| [12_Configuration_and_Vimrc.md](./12_Configuration_and_Vimrc.md) | ⭐⭐⭐ | .vimrc 구조, 옵션, 키 매핑, autocmd, 파일 타입 |
| [13_Plugins_and_Ecosystem.md](./13_Plugins_and_Ecosystem.md) | ⭐⭐⭐ | 플러그인 매니저, 핵심 플러그인, 간단한 플러그인 작성 |
| [14_Neovim_and_Modern_Workflows.md](./14_Neovim_and_Modern_Workflows.md) | ⭐⭐⭐ | Neovim, Lua 설정, LSP, Treesitter, 터미널, Git 통합 |

---

## 예제 파일

예제 파일은 [`examples/VIM/`](../../../examples/VIM/)에 있습니다:

| 파일 | 설명 |
|------|------|
| `01_basic_motions.txt` | 연습 지침이 포함된 모션 연습 텍스트 |
| `02_operators_practice.txt` | 연산자 + 모션 조합 연습 |
| `03_text_objects_practice.py` | 텍스트 객체 연습용 Python 코드 |
| `04_search_replace.txt` | 검색 및 치환 패턴 연습 |
| `05_macro_examples.txt` | 매크로 녹화 연습 (반복 편집) |
| `06_minimal_vimrc.vim` | 주석이 달린 옵션과 함께하는 최소한의 .vimrc |
| `07_intermediate_vimrc.vim` | 중급 설정 (매핑, autocmd, 상태표시줄) |
| `08_advanced_vimrc.vim` | 고급 설정 (함수, 조건문, 플러그인) |
| `09_init_lua.lua` | Neovim init.lua 기본 설정 |
| `10_vim_cheatsheet.md` | 명령어 참조 치트시트 |

---

## 권장 학습 경로

### 1주차: 생존
L01–L03을 완료합니다. 매일 `vimtutor`를 실행합니다. 작은 편집에 VIM을 강제로 사용해 보세요.

### 2–3주차: 속도 향상
L04–L08을 학습합니다. "연산자 + 모션" 문법(L05)은 가장 중요한 개념입니다 — 이것이 이해되는 순간 모든 것이 빨라집니다.

### 4주차 이후: 커스터마이징 & 파워 활용
필요에 따라 L09–L14를 탐색합니다. `.vimrc`를 설정하고(L12), 핵심 플러그인을 추가하며(L13), 모던 설정을 위해 Neovim(L14)을 고려해 보세요.

---

## 실습 환경

```bash
# vim 설치 확인
vim --version | head -1

# macOS (기본 설치됨, 또는 최신 버전 설치)
brew install vim        # 최신 Vim
brew install neovim     # Neovim

# Ubuntu/Debian
sudo apt install vim    # Vim
sudo apt install neovim # Neovim

# 내장 튜토리얼 시작 (최고의 첫 걸음!)
vimtutor
```

---

## 관련 주제

- [Shell_Script](../Shell_Script/00_Overview.md) — 명령줄 기술 (VIM은 터미널에서 동작합니다)
- [Linux](../Linux/00_Overview.md) — VIM이 기본 편집기인 시스템 관리
- [Git](../Git/00_Overview.md) — 커밋 메시지, 대화형 리베이스에서 기본적으로 VIM 사용

---

## 참고 자료

- [Vim 공식 문서](https://www.vim.org/docs.php)
- [Neovim 문서](https://neovim.io/doc/)
- [Practical Vim (Drew Neil 저)](https://pragprog.com/titles/dnvim2/practical-vim-second-edition/)
- [Learn Vimscript the Hard Way](https://learnvimscriptthehardway.stevelosh.com/)
- [Vim Adventures](https://vim-adventures.com/) — 게임 기반 학습
