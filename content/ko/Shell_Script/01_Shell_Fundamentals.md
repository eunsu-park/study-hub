# Shell 기초와 실행 환경

**다음**: [매개변수 확장과 변수 속성](./02_Parameter_Expansion.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 기능, POSIX 준수성, 스크립팅 적합성 측면에서 bash, sh, zsh, fish 쉘을 비교할 수 있습니다
2. POSIX 호환 구문을 식별하고 이식성을 해치는 bashism을 피할 수 있습니다
3. 로그인, 대화형(interactive), 비대화형(non-interactive), 제한(restricted) 쉘 모드를 구별할 수 있습니다
4. 쉘 호출 방식에 따른 시작 파일 로딩 순서를 추적할 수 있습니다
5. 종료 코드(exit code)를 설명하고 이를 활용해 견고한 조건부 로직을 구성할 수 있습니다
6. `set`과 `shopt` 옵션을 사용해 쉘 동작을 설정할 수 있습니다
7. `/usr/bin/env`를 사용해 이식 가능한 shebang 줄을 작성할 수 있습니다

---

모든 쉘 스크립트는 실행 환경(execution environment) 안에서 동작하며, 이 환경은 어떤 기능을 사용할 수 있는지, 어떤 설정 파일이 로드되는지, 오류를 어떻게 처리하는지를 결정합니다. 이러한 기초를 제대로 이해하지 못하면 로컬에서는 잘 동작하는 스크립트가 cron 작업, Docker 컨테이너, 또는 다른 시스템에서 실패하는 상황이 발생합니다. 이 레슨은 다양한 환경에서 예측 가능하게 동작하는 스크립트를 작성하기 위한 기초 지식을 제공합니다.

## 1. 쉘의 종류

다양한 쉘은 서로 다른 기능과 호환성 수준을 제공합니다. 이러한 차이점을 이해하는 것은 이식 가능한 스크립트를 작성하는 데 중요합니다.

### 쉘 비교 표

| 기능 | bash | sh (dash) | zsh | fish |
|---------|------|-----------|-----|------|
| POSIX 호환 | 대부분 | 예 | 대부분 | 아니오 |
| 배열(Arrays) | 예 | 아니오 | 예 (더 좋음) | 예 |
| 연관 배열(Associative arrays) | 예 (4.0+) | 아니오 | 예 | 예 |
| [[ ]] 테스트 | 예 | 아니오 | 예 | 아니오 |
| 프로세스 치환(Process substitution) | 예 | 아니오 | 예 | 예 |
| Here 문자열(Here strings) | 예 | 아니오 | 예 | 예 |
| 산술 연산 (( )) | 예 | 아니오 | 예 | 아니오 |
| 명령 완성(Command completion) | 좋음 | 기본 | 탁월 | 탁월 |
| 시작 성능 | 중간 | 빠름 | 느림 | 중간 |
| 스크립팅 초점 | 예 | 예 | 예 | 아니오 |
| Debian/Ubuntu 기본값 | bash | dash (/bin/sh) | bash | fish |
| 설정 문법 | bash | POSIX sh | zsh-extended | Fish-specific |

### 각 쉘을 사용해야 하는 경우

**bash**: 범용 스크립팅, 대부분의 시스템에서 사용 가능, 기능과 이식성의 좋은 균형.

```bash
#!/bin/bash
# 배열, [[, 프로세스 치환이 필요한 스크립트에는 bash 사용
declare -A config
config[host]="localhost"
config[port]=8080

if [[ -n "${config[host]}" ]]; then
    echo "Host: ${config[host]}"
fi
```

**sh (POSIX)**: 최대 이식성, 임베디드 시스템, 최소 환경.

```bash
#!/bin/sh
# POSIX 호환 스크립트 - bashisms 없음
# 배열 없음, [[ 없음, 프로세스 치환 없음

if [ -n "$HOST" ]; then
    echo "Host: $HOST"
fi

# 정규식을 사용하는 [[ 대신 case 사용
case "$filename" in
    *.txt) echo "Text file" ;;
    *.log) echo "Log file" ;;
esac
```

**zsh**: 대화형 사용, 고급 완성 기능, 더 나은 배열 처리.

```bash
#!/bin/zsh
# zsh는 더 강력한 배열 기능을 가짐
array=(one two three)
echo $array[1]  # zsh 배열은 1-indexed (bash는 0-indexed 사용)

# 고급 글로빙
setopt extended_glob
files=(^*.txt)  # .txt를 제외한 모든 파일
```

**fish**: 대화형 쉘, 사용자 친화적, 이식 가능한 스크립트에는 적합하지 않음.

```fish
#!/usr/bin/fish
# Fish는 다른 문법을 가짐 - POSIX 호환하지 않음
set host localhost
set port 8080

if test -n "$host"
    echo "Host: $host"
end
```

## 2. POSIX 호환성

POSIX (Portable Operating System Interface)는 쉘 동작에 대한 표준을 정의합니다. POSIX 호환 스크립트는 모든 POSIX 쉘(sh, bash, dash, ksh 등)에서 실행됩니다.

### POSIX vs Bash-isms

| 기능 | POSIX sh | bash 확장 |
|---------|----------|----------------|
| 테스트 명령 | `[ ]` | `[[ ]]` |
| 문자열 비교 | `[ "$a" = "$b" ]` | `[[ $a == $b ]]` |
| 정규식 매칭 | (grep 사용) | `[[ $str =~ regex ]]` |
| 배열 | 지원 안 됨 | `arr=(1 2 3)` |
| 함수 | `func() { }` | `function func { }` |
| 산술 연산 | `expr`, `$(( ))` | `let`, `(( ))` |
| 프로세스 치환 | 지원 안 됨 | `<(cmd)`, `>(cmd)` |
| Here 문자열 | 지원 안 됨 | `<<< "string"` |
| 지역 변수 | POSIX에 없음 | `local var=value` |

### 이식 가능한 POSIX 스크립트 작성

```bash
#!/bin/sh
# POSIX 호환 스크립트 예제

# [[ ]] 대신 [ ] 사용
if [ "$1" = "start" ]; then
    echo "Starting service..."
fi

# 산술 연산에 $(( )) 사용 (이것은 POSIX입니다)
count=0
count=$((count + 1))

# 패턴 매칭에 case 사용
case "$filename" in
    *.tar.gz|*.tgz)
        echo "Compressed tarball"
        ;;
    *.zip)
        echo "ZIP archive"
        ;;
    *)
        echo "Unknown format"
        ;;
esac

# 배열 피하기 - 위치 매개변수나 임시 파일 사용
set -- "item1" "item2" "item3"
for item in "$@"; do
    echo "$item"
done

# 백틱이 아닌 명령 치환 $(cmd) 사용
current_dir=$(pwd)

# 이식 가능한 방식으로 명령 존재 확인
if command -v docker >/dev/null 2>&1; then
    echo "Docker is installed"
fi
```

## 3. 쉘 모드

쉘은 호출 방식에 따라 다른 모드로 작동합니다. 이는 어떤 시작 파일을 읽을지에 영향을 줍니다.

### Login vs Non-Login 쉘

**Login 쉘**: 로그인할 때 시작됨(SSH, 콘솔 로그인, `bash --login`).

**Non-login 쉘**: 기존 세션에서 시작됨(GUI에서 터미널 열기, bash에서 bash 실행).

쉘이 login 쉘인지 테스트:

```bash
#!/bin/bash
# login 쉘로 실행 중인지 확인
if shopt -q login_shell; then
    echo "This is a login shell"
else
    echo "This is a non-login shell"
fi

# 대체 방법
case "$-" in
    *l*) echo "Login shell" ;;
    *) echo "Non-login shell" ;;
esac
```

### Interactive vs Non-Interactive 쉘

**Interactive**: 터미널 연결됨, 사용자 입력 수락(일반 터미널 세션).

**Non-interactive**: 스크립트 실행, 터미널 상호작용 없음.

쉘이 대화형인지 테스트:

```bash
#!/bin/bash
# 대화형으로 실행 중인지 확인
if [[ $- == *i* ]]; then
    echo "Interactive shell"
else
    echo "Non-interactive shell (script)"
fi

# 대체 방법
case "$-" in
    *i*) echo "Interactive" ;;
    *) echo "Non-interactive" ;;
esac

# stdin이 터미널인지 확인
if [ -t 0 ]; then
    echo "stdin is a terminal"
else
    echo "stdin is not a terminal (piped/redirected)"
fi
```

## 4. 시작 파일 로딩 순서

bash가 설정 파일을 읽는 순서는 쉘 모드에 따라 다릅니다.

### 시작 순서 다이어그램

```
Login Shell (bash --login 또는 SSH)
├── /etc/profile (시스템 전체)
│   └── /etc/profile.d/*.sh (/etc/profile이 소싱하는 경우)
└── 첫 번째로 발견된 파일:
    ├── ~/.bash_profile
    ├── ~/.bash_login  (~/.bash_profile이 없는 경우)
    └── ~/.profile     (위 두 파일이 모두 없는 경우)
        └── (많은 .bash_profile 파일이 ~/.bashrc를 소싱함)

Non-Login Interactive Shell (터미널 윈도우)
├── /etc/bash.bashrc (Debian/Ubuntu)
└── ~/.bashrc

Non-Interactive Shell (스크립트)
├── $BASH_ENV (설정된 경우, 소싱할 파일 경로)
└── (일반적으로 없음)

Login Shell 종료
└── ~/.bash_logout
```

### 시작 파일 목적

| 파일 | 목적 | 일반적인 내용 |
|------|---------|------------------|
| `/etc/profile` | 시스템 전체 login 설정 | PATH, LANG, umask |
| `/etc/bash.bashrc` | 시스템 전체 대화형 설정 | PS1, 별칭 (Debian/Ubuntu) |
| `~/.bash_profile` | 사용자 login 설정 | ~/.bashrc 소싱, PATH 설정 |
| `~/.bashrc` | 사용자 대화형 설정 | 별칭, 함수, PS1 |
| `~/.profile` | POSIX login 설정 | 이식 가능한 login 설정 |
| `~/.bash_logout` | 로그아웃 시 정리 | 화면 지우기, 임시 파일 정리 |

### 시작 파일 구조 예제

**~/.bash_profile** (login 쉘 진입점):

```bash
# ~/.bash_profile - login 쉘이 로딩

# 사용자 바이너리를 위한 PATH 설정
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

# .bashrc가 존재하면 로딩 (대화형 login 쉘용)
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# Login 전용 설정
echo "Last login: $(date)" >> ~/.login_log
```

**~/.bashrc** (대화형 쉘 설정):

```bash
# ~/.bashrc - 대화형 non-login 쉘이 로딩

# 대화형으로 실행 중이 아니면, 아무것도 하지 않음
case $- in
    *i*) ;;
      *) return;;
esac

# 히스토리 설정
HISTCONTROL=ignoreboth
HISTSIZE=10000
HISTFILESIZE=20000
shopt -s histappend

# 프롬프트
PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

# 별칭
alias ll='ls -lah'
alias grep='grep --color=auto'

# 별도 파일에서 함수 로딩
if [ -f ~/.bash_functions ]; then
    . ~/.bash_functions
fi
```

**~/.profile** (POSIX 호환 login 설정):

```bash
# ~/.profile - POSIX 호환 login 설정
# bash가 sh로 호출될 때, 또는 다른 POSIX 쉘에서 사용

# PATH 설정
PATH="$HOME/bin:$PATH"
export PATH

# 환경 변수
export EDITOR=vim
export PAGER=less

# bash인 경우, .bashrc 소싱
if [ -n "$BASH_VERSION" ]; then
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi
```

## 5. 종료 코드(Exit Codes)

종료 코드는 명령이 성공했는지 실패했는지를 나타냅니다. 관례적으로:

### 종료 코드 규칙

| 코드 | 의미 |
|------|---------|
| 0 | 성공 |
| 1 | 일반 오류 |
| 2 | 쉘 내장 명령의 잘못된 사용 |
| 126 | 명령을 찾았지만 실행 불가 |
| 127 | 명령을 찾을 수 없음 |
| 128 | 잘못된 exit 인자 |
| 128+N | 치명적 오류 시그널 N (130 = Ctrl+C (SIGINT=2)) |
| 255 | 종료 상태 범위 초과 |

### 종료 코드 사용하기

```bash
#!/bin/bash

# $?로 종료 코드 확인
grep "pattern" file.txt
if [ $? -eq 0 ]; then
    echo "Pattern found"
else
    echo "Pattern not found"
fi

# 더 좋은 방법: if에서 직접 명령 사용
if grep "pattern" file.txt > /dev/null; then
    echo "Pattern found"
fi

# 함수에서 사용자 정의 종료 코드 반환
validate_input() {
    local input="$1"

    if [ -z "$input" ]; then
        echo "Error: input is empty" >&2
        return 1
    fi

    if ! [[ "$input" =~ ^[0-9]+$ ]]; then
        echo "Error: input must be numeric" >&2
        return 2
    fi

    if [ "$input" -lt 0 ] || [ "$input" -gt 100 ]; then
        echo "Error: input must be between 0 and 100" >&2
        return 3
    fi

    return 0
}

# 함수 사용 및 종료 코드 확인
if validate_input "42"; then
    echo "Input is valid"
else
    case $? in
        1) echo "Empty input" ;;
        2) echo "Not numeric" ;;
        3) echo "Out of range" ;;
    esac
fi

# 특정 코드로 스크립트 종료
check_prerequisites() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Error: docker not found" >&2
        exit 127
    fi

    if [ ! -r /etc/config.conf ]; then
        echo "Error: config file not readable" >&2
        exit 1
    fi
}
```

### 종료 코드 모범 사례

```bash
#!/bin/bash
set -e  # 오류 시 종료 (이것은 주의해서 사용)

# 명시적으로 오류 처리
perform_backup() {
    local source="$1"
    local dest="$2"

    if ! tar czf "$dest" "$source" 2>/dev/null; then
        echo "Backup failed" >&2
        return 1
    fi

    echo "Backup successful"
    return 0
}

# 적절한 오류 처리로 명령 체이닝
if perform_backup "/data" "/backup/data.tar.gz"; then
    echo "Cleaning up old backups..."
    find /backup -name "*.tar.gz" -mtime +7 -delete
else
    echo "Backup failed, keeping old backups"
    exit 1
fi
```

## 6. 쉘 옵션 개요

쉘 옵션은 쉘 동작을 제어합니다. 두 명령으로 옵션을 관리합니다:

- **set**: POSIX 표준 옵션
- **shopt**: Bash 전용 확장 옵션

### 중요한 set 옵션

```bash
#!/bin/bash

# 현재 옵션 표시
echo "$-"  # 예: "himBH" (각 문자는 활성 옵션)

# 옵션 활성화
set -e  # 오류 시 종료 (errexit)
set -u  # 정의되지 않은 변수 사용 시 오류 (nounset)
set -o pipefail  # 파이프라인에서 하나라도 실패하면 실패
set -x  # 실행 전 명령 출력 (xtrace)

# 옵션 비활성화
set +e  # 오류 시 종료 안 함
set +x  # 명령 출력 중지

# 옵션 결합
set -euo pipefail  # 일반적인 "strict mode"

# 예제: noclobber는 파일 덮어쓰기 방지
set -o noclobber
echo "test" > file.txt  # 파일 생성
echo "test" > file.txt  # 오류: 파일 존재
echo "test" >| file.txt  # >|로 noclobber 무시
```

### 일반적인 set 옵션

| 옵션 | 짧은 형태 | 설명 |
|--------|-------|-------------|
| `-e` (errexit) | `-e` | 명령 실패 시 종료 |
| `-u` (nounset) | `-u` | 정의되지 않은 변수 사용 시 오류 |
| `-x` (xtrace) | `-x` | 실행 전 명령 출력 |
| `-o pipefail` | (긴 형태만) | 파이프라인에서 하나라도 실패하면 실패 |
| `-o noclobber` | `-C` | >로 파일 덮어쓰기 방지 |
| `-o noglob` | `-f` | 경로명 확장 비활성화 |
| `-o vi` | (긴 형태만) | Vi 스타일 명령줄 편집 |
| `-o emacs` | (긴 형태만) | Emacs 스타일 편집 (기본값) |

### 중요한 shopt 옵션

```bash
#!/bin/bash

# bash 확장 옵션 활성화
shopt -s extglob  # 확장 패턴 매칭
shopt -s globstar  # 재귀 glob을 위한 **
shopt -s nullglob  # 매칭 안 되는 glob은 빈 것으로 확장
shopt -s dotglob  # glob에 숨김 파일 포함
shopt -s nocaseglob  # 대소문자 구분 안 하는 글로빙

# 옵션 비활성화
shopt -u dotglob  # 숨김 파일 제외

# 옵션이 설정되었는지 확인
if shopt -q nullglob; then
    echo "nullglob is enabled"
fi

# 예제: nullglob
shopt -s nullglob
files=(*.txt)
if [ ${#files[@]} -eq 0 ]; then
    echo "No .txt files found"
else
    echo "Found ${#files[@]} .txt files"
fi

# 예제: globstar
shopt -s globstar
# 재귀적으로 모든 Python 파일 찾기
for file in **/*.py; do
    echo "$file"
done

# 예제: extglob (Lesson 04에서 더 자세히 다룸)
shopt -s extglob
rm !(*.txt|*.log)  # .txt와 .log 파일을 제외한 모든 파일 삭제
```

### 유용한 shopt 옵션

| 옵션 | 설명 |
|--------|-------------|
| `extglob` | 확장 패턴 매칭 (!(pat), *(pat), 등) |
| `globstar` | **는 재귀적으로 매칭 |
| `nullglob` | 매칭 안 되는 glob은 null로 확장, 리터럴 아님 |
| `dotglob` | 경로명 확장에 숨김 파일 포함 |
| `nocaseglob` | 대소문자 구분 안 하는 경로명 확장 |
| `failglob` | 매칭 안 되는 glob은 오류 발생 |
| `checkjobs` | 종료 전 실행 중인 작업 확인 |
| `autocd` | 디렉터리명만 입력해도 디렉터리 변경 |
| `cdspell` | cd 오류 자동 수정 |

### Strict Mode 예제

```bash
#!/bin/bash
# 더 안전한 스크립트를 위한 strict mode

set -euo pipefail
IFS=$'\n\t'

# -e: 오류 시 종료
# -u: 정의되지 않은 변수 사용 시 오류
# -o pipefail: 파이프라인에서 하나라도 실패하면 실패
# IFS: 더 안전한 단어 분할

# 이제 오류가 스크립트를 중지시킴
command_that_fails  # 스크립트가 여기서 종료
echo "This won't execute"

# 명시적으로 오류 처리하려면:
if ! command_that_might_fail; then
    echo "Command failed, handling error"
    # 정리 작업 수행
    exit 1
fi
```

## 7. env 명령과 #!/usr/bin/env bash

### #!/usr/bin/env bash를 사용하는 이유

셔뱅(shebang) 라인은 시스템에 어떤 인터프리터를 사용할지 알려줍니다. 두 가지 접근 방식:

**직접 경로**: `#!/bin/bash`
- 빠름 (PATH 검색 없음)
- 이식 가능하지 않음 (bash가 /usr/local/bin에 있을 수 있음)

**env 접근**: `#!/usr/bin/env bash`
- 이식 가능 (PATH에서 bash를 찾음)
- 현대 스크립트의 표준
- 다양한 시스템에서 작동

```bash
#!/usr/bin/env bash
# 이것은 PATH에서 bash가 어디 있든 찾아냄

# bash가 어디에 있는지 확인
which bash

# macOS: /bin/bash
# FreeBSD: /usr/local/bin/bash
# Nix: /nix/store/.../bin/bash
```

### env를 사용하여 환경 설정

```bash
#!/usr/bin/env -S bash -euo pipefail
# -S 플래그는 여러 인자 전달 허용 (GNU env 8.30+)

# 이전 env를 위한 대안:
#!/usr/bin/env bash
set -euo pipefail
```

### 깨끗한 환경을 위한 env

```bash
# 깨끗한 환경으로 명령 실행
env -i bash --norc --noprofile

# 특정 변수만으로 실행
env -i HOME=/tmp USER=testuser bash

# 특정 변수 제거
env -u DISPLAY firefox

# 변수 추가
env FOO=bar ./script.sh

# 환경 검사
env | sort
```

### 모범 사례를 적용한 스크립트 템플릿

```bash
#!/usr/bin/env bash
# Script: example.sh
# Description: 모범 사례를 적용한 예제 스크립트
# Author: Your Name
# Created: 2026-02-13

# Strict mode
set -euo pipefail
IFS=$'\n\t'

# 오류 트랩
trap 'echo "Error on line $LINENO" >&2' ERR

# 유용한 쉘 옵션
shopt -s nullglob globstar

# 상수
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# 함수
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] <argument>

Options:
    -h, --help      이 도움말 메시지 표시
    -v, --verbose   상세 출력 활성화
    -d, --debug     디버그 모드 활성화

Examples:
    $SCRIPT_NAME input.txt
    $SCRIPT_NAME -v input.txt
EOF
}

main() {
    # 메인 스크립트 로직
    echo "Running from: $SCRIPT_DIR"
    echo "Script name: $SCRIPT_NAME"
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -d|--debug)
            set -x
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# 메인 함수 실행
main "$@"
```

## 연습 문제

### 문제 1: 쉘 감지 스크립트

다음을 감지하는 스크립트를 작성하세요:
- 실행 중인 쉘 (bash, zsh, sh 등)
- login 쉘인지 non-login 쉘인지
- 대화형인지 비대화형인지
- 쉘 버전

**예상 출력**:
```
Shell: bash
Version: 5.1.16
Type: non-login, interactive
```

### 문제 2: 시작 파일 분석기

다음을 수행하는 스크립트를 작성하세요:
- 시스템에 존재하는 모든 bash 시작 파일 나열
- login vs non-login 쉘에서 로딩되는 순서 표시
- 각 파일의 첫 5줄 표시
- 일반적인 실수 확인 (.bash_profile 대신 .bashrc에 별칭 설정 등)

### 문제 3: 종료 코드 로거

모든 명령을 래핑하는 함수를 작성하세요:
- 실행되는 명령 로깅
- 종료 코드 캡처 및 로깅
- 실행 시간 로깅
- 로그 파일에 추가: `timestamp | command | exit_code | duration`

**사용 예제**:
```bash
log_command ls -la /nonexistent
# 로깅되어야 함: 2026-02-13 10:30:45 | ls -la /nonexistent | 2 | 0.003s
```

### 문제 4: 이식 가능한 스크립트 검사기

다른 bash 스크립트를 분석하고 다음을 보고하는 스크립트를 작성하세요:
- 사용된 비POSIX 구조 ([[, 배열 등)
- sh에서 실패할 Bashisms
- 더 이식 가능하게 만들기 위한 제안
- "이식성 점수" (0-100%)

**힌트**: `[[`, `declare`, `function 키워드` 등의 패턴을 검색하세요.

### 문제 5: 환경 스냅샷

다음을 수행하는 스크립트를 작성하세요:
- 현재 환경 변수를 파일에 저장
- 현재 쉘 옵션(set -o, shopt -p)을 파일에 저장
- 저장된 상태에서 환경을 복원할 수 있음
- 저장된 상태와 현재 상태의 차이 표시

**사용법**:
```bash
./envsnap.sh save snapshot.env
# ... 변경 수행 ...
./envsnap.sh diff snapshot.env
./envsnap.sh restore snapshot.env
```

---

**다음**: [02_Parameter_Expansion.md](./02_Parameter_Expansion.md)
