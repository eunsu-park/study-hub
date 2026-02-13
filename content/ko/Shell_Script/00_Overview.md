# Shell Scripting 학습 가이드

## 소개

이 폴더는 프로그래밍 분야로서의 쉘 스크립팅을 체계적으로 학습할 수 있도록 제공합니다. Bash를 주 쉘로 사용하여, 기본적인 스크립팅을 넘어서는 고급 기법, 실전 자동화 패턴, 전문적인 모범 사례를 다룹니다.

**대상 독자**: Linux 토픽을 완료한 학습자 (특히 Lesson 09: Shell Scripting 기초)

---

## 학습 로드맵

```
[Foundation]              [Intermediate]             [Advanced]
    |                         |                          |
    v                         v                          v
Shell Basics/Env ------> Functions/Libs ---------> Portability/Best Practices
    |                         |                          |
    v                         v                          v
Parameter Expansion ----> I/O & Redirection ------> Testing
    |                         |
    v                         v                     [Projects]
Arrays & Data ----------> String/Regex ----------> Task Runner
    |                         |                          |
    v                         v                          v
Adv. Control Flow ------> Process/Error ----------> Deployment
                              |                          |
                              v                          v
                         Arg Parsing/CLI ---------> Monitoring Tool
```

---

## 선수 학습

- Linux 기초 및 터미널 사용 익숙도
- [Linux/09_Shell_Scripting.md](../Linux/09_Shell_Scripting.md) - 변수, 조건문, 반복문, 함수, 배열, 디버깅 기초
- [Linux/04_Text_Processing.md](../Linux/04_Text_Processing.md) - grep, sed, awk 기초

---

## 파일 목록

### Foundation (복습 + 기초 심화)

| 파일 | 난이도 | 핵심 주제 |
|------|-----------|------------|
| [01_Shell_Fundamentals.md](./01_Shell_Fundamentals.md) | ⭐ | 쉘 종류(bash/sh/zsh/dash), POSIX, login/non-login, profile/bashrc 로딩, exit codes |
| [02_Parameter_Expansion.md](./02_Parameter_Expansion.md) | ⭐⭐ | 문자열 조작, ${var#}, ${var//}, 부분문자열, 간접 참조, declare |
| [03_Arrays_and_Data.md](./03_Arrays_and_Data.md) | ⭐⭐ | 연관 배열, 스택/큐 시뮬레이션, CSV 파싱, 설정 로딩 |
| [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md) | ⭐⭐ | [[ ]] vs [ ] vs (( )), extglob, select 메뉴, bc를 이용한 연산 |

### Intermediate (심화 기법)

| 파일 | 난이도 | 핵심 주제 |
|------|-----------|------------|
| [05_Functions_and_Libraries.md](./05_Functions_and_Libraries.md) | ⭐⭐ | 반환 패턴, 재귀, 함수 라이브러리, 네임스페이스, 콜백 |
| [06_IO_and_Redirection.md](./06_IO_and_Redirection.md) | ⭐⭐⭐ | 파일 디스크립터, here documents, 프로세스 치환, named pipes, 파이프 함정 |
| [07_String_Processing.md](./07_String_Processing.md) | ⭐⭐⭐ | 내장 문자열 연산, printf, tr/cut/paste/join, JSON/YAML을 위한 jq/yq |
| [08_Regex_in_Bash.md](./08_Regex_in_Bash.md) | ⭐⭐⭐ | =~ 연산자, BASH_REMATCH, 확장 정규식, glob vs regex, 실전 검증 |
| [09_Process_Management.md](./09_Process_Management.md) | ⭐⭐⭐ | 백그라운드 작업, 서브쉘, 시그널 & trap, 정리 패턴, coproc |
| [10_Error_Handling.md](./10_Error_Handling.md) | ⭐⭐⭐ | set -euo pipefail 심화, trap ERR, 에러 프레임워크, ShellCheck, 로깅 |
| [11_Argument_Parsing.md](./11_Argument_Parsing.md) | ⭐⭐⭐ | getopts, getopt, 자가 문서화 도움말, 색상 출력, 진행률 표시 |

### Advanced (전문적 기법)

| 파일 | 난이도 | 핵심 주제 |
|------|-----------|------------|
| [12_Portability_and_Best_Practices.md](./12_Portability_and_Best_Practices.md) | ⭐⭐⭐⭐ | POSIX vs bash vs zsh, bashisms, Google Shell Style Guide, 보안, 성능 |
| [13_Testing.md](./13_Testing.md) | ⭐⭐⭐⭐ | Bats 프레임워크, 단위 테스트 패턴, 모킹, TDD, CI 통합 |

### Projects (실전 적용)

| 파일 | 난이도 | 핵심 주제 |
|------|-----------|------------|
| [14_Project_Task_Runner.md](./14_Project_Task_Runner.md) | ⭐⭐⭐ | Makefile 스타일 태스크 러너, 의존성 관리, 병렬 실행 |
| [15_Project_Deployment.md](./15_Project_Deployment.md) | ⭐⭐⭐⭐ | SSH 배포, 롤링 배포, Docker 엔트리포인트, 롤백 |
| [16_Project_Monitor.md](./16_Project_Monitor.md) | ⭐⭐⭐⭐ | 실시간 대시보드, 알림, 로그 집계, cron 통합 |

---

## 권장 학습 순서

1. **Foundation (1주차)**: 01 → 02 → 03 → 04
   - 쉘 기초를 빠르게 복습한 후, 파라미터 확장과 배열을 깊이 학습
2. **Intermediate (2-3주차)**: 05 → 06 → 07 → 08 → 09 → 10 → 11
   - 핵심 스크립팅 기법: I/O, 정규식, 프로세스, 에러 처리
3. **Advanced (4주차)**: 12 → 13
   - 이식성, 모범 사례, 테스팅
4. **Projects (5주차)**: 14 → 15 → 16
   - 실전 프로젝트에 모든 학습 내용 적용

---

## 실습 환경

```bash
# bash 버전 확인 (연관 배열을 위해 4.0+ 권장)
bash --version

# 정적 분석을 위한 ShellCheck 설치
# macOS
brew install shellcheck

# Ubuntu/Debian
sudo apt install shellcheck

# 테스팅을 위한 Bats 설치 (Lesson 13)
brew install bats-core  # macOS
# 또는 소스에서 설치: https://github.com/bats-core/bats-core
```

---

## 관련 토픽

- [Linux/](../Linux/00_Overview.md) - Linux 기초, 쉘 기본
- [Git/](../Git/00_Overview.md) - 버전 관리 (스크립트는 Git 훅에서 자주 사용)
- [Docker/](../Docker/00_Overview.md) - 컨테이너 엔트리포인트는 쉘 스크립트 사용
- [MLOps/](../MLOps/00_Overview.md) - 자동화 파이프라인

---

## 참고 자료

- [GNU Bash Manual](https://www.gnu.org/software/bash/manual/)
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- [ShellCheck Wiki](https://www.shellcheck.net/wiki/)
- [Bats-core Documentation](https://bats-core.readthedocs.io/)
- [Advanced Bash-Scripting Guide](https://tldp.org/LDP/abs/html/)
