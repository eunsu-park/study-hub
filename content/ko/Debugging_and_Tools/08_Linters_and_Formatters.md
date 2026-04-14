# 린터와 포매터

**이전**: [테스트 기초](./07_Testing_Basics.md) | **다음**: [타입 체킹](./09_Type_Checking.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 린터(버그 찾기)와 포매터(스타일 적용)의 차이 설명하기
2. `pylint`로 버그, 코드 스멜, 스타일 위반 찾기
3. `flake8`을 가볍고 빠른 린팅 도구로 사용하기
4. `ruff`를 현대적이고 초고속인 올인원 린터로 사용하기
5. `black`으로 코드를 일관된 스타일로 자동 포맷하기
6. 설정 파일(`pyproject.toml`, `.flake8`)로 린터 설정하기
7. pre-commit 훅을 설정하여 커밋 전에 린터 자동 실행하기
8. 린터 출력을 해석하고 흔한 경고 수정하기

---

린터와 포매터는 코드가 실행되기 **전에** 버그를 잡고 코드 스타일을 강제하는 자동화 도구입니다. 린터는 소스 코드를 읽고 잠재적 문제를 표시합니다: 사용되지 않는 변수, 도달 불가능한 코드, 스타일 위반, 일부 논리 에러까지. 포매터는 일관된 스타일을 따르도록 코드를 자동으로 다시 작성합니다.

> **예방 vs 치료:** 디버깅은 버그가 발생한 후에 찾습니다. 린터는 발생하기 전에 찾습니다. 최고의 버그는 테스트 스위트에 도달하지 않는 버그입니다.

---

## 1. 린터 vs 포매터

| 측면 | 린터 | 포매터 |
|------|------|--------|
| 목적 | 버그와 코드 스멜 찾기 | 일관된 스타일 강제 |
| 동작 | 경고를 보고 | 파일을 다시 작성 |
| 예시 | pylint, flake8, ruff | black, autopep8, yapf |
| 초점 | 정확성 + 스타일 | 스타일만 |

---

## 2. pylint: 포괄적인 린터

```bash
pip install pylint
pylint my_script.py
```

### pylint 메시지 코드 이해

```
C0114 → Convention (스타일)
W0611 → Warning (가능한 문제)
E0001 → Error (확실한 버그)
R0903 → Refactor (코드 스멜)
F0001 → Fatal (pylint 처리 불가)
```

### 유용한 pylint 검사

| 검사 | 잡는 것 |
|------|---------|
| `W0611` | 사용되지 않는 import |
| `W0612` | 사용되지 않는 변수 |
| `E0602` | 정의되지 않은 변수 |
| `W0104` | 효과가 없는 문 |
| `W0621` | 외부 스코프 이름 재정의 |

---

## 3. flake8: 가벼운 린터

```bash
pip install flake8
flake8 my_script.py
```

pylint보다 빠르고 간단합니다. 세 가지 도구를 결합: `pycodestyle` (스타일), `pyflakes` (에러), `mccabe` (복잡도).

---

## 4. ruff: 현대적 초고속 린터

```bash
pip install ruff
ruff check my_script.py      # 린팅
ruff format my_script.py     # 포맷 (black과 유사)
```

### ruff의 장점

- Rust로 작성 -- pylint나 flake8보다 **10-100배 빠름**
- pylint, flake8, isort, pyupgrade 등을 하나의 도구로 대체
- 많은 규칙에 대한 자동 수정 기능
- 활발한 개발과 성장하는 규칙 세트

```bash
# 수정 가능한 것을 자동 수정
ruff check --fix my_script.py
```

### 설정 (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle 에러
    "W",   # pycodestyle 경고
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
]
ignore = ["E501"]  # 줄이 너무 긺
```

---

## 5. black: 코드 포매터

```bash
pip install black
black my_script.py          # 파일 포맷
black --check my_script.py  # 수정 없이 확인
```

black은 **주관적** -- 설정 옵션이 의도적으로 매우 적습니다. `ruff format`은 `black`과 호환되는 더 빠른 대안입니다.

---

## 6. Pre-Commit 훅

### 설정

```bash
pip install pre-commit
```

`.pre-commit-config.yaml` 생성:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
```

```bash
pre-commit install          # 훅 설치
pre-commit run --all-files  # 수동 실행
```

---

## 7. 린터 비교

| 기능 | pylint | flake8 | ruff |
|------|--------|--------|------|
| 속도 | 느림 | 보통 | 매우 빠름 |
| 자동 수정 | 없음 | 없음 | 있음 (많은 규칙) |
| 포맷팅 | 없음 | 없음 | 있음 (`ruff format`) |
| import 정렬 | 없음 | 없음 (플러그인) | 있음 (내장) |

### 초보자 권장

**ruff**로 시작하세요: 빠르고, 포괄적이며, 자동 수정이 있고, 포맷팅과 import 정렬을 포함합니다.

```bash
pip install ruff
ruff check --fix .     # 린팅 및 수정
ruff format .          # 포맷
```

---

## 8. 흔한 경고 해석 및 수정

| 코드 | 의미 | 수정 |
|------|------|------|
| F401 | 사용되지 않는 import | import 제거 |
| F841 | 사용되지 않는 변수 | 변수 사용하거나 `_`로 접두사 |
| E711 | None과 `==` 비교 | `is None` 사용 |
| E722 | 맨 except (모든 것을 잡음) | `except Exception:` 사용 |
| E501 | 줄이 너무 긺 | 여러 줄로 나누기 |

---

## 요약

- 린터는 버그와 코드 스멜을 찾고; 포매터는 일관된 스타일을 강제
- `pylint`은 포괄적이지만 느림; `flake8`은 빠르지만 덜 철저
- `ruff`가 현대적 선택: 빠르고, 포괄적이며, 자동 수정과 포맷팅 제공
- `black` (또는 `ruff format`)이 자동 포맷팅으로 스타일 논쟁을 제거
- pre-commit 훅이 매 커밋 전에 자동으로 검사 실행
- `ruff`로 시작하세요 -- pylint, flake8, isort, black을 하나의 도구로 대체
- 린터 경고를 항상 수정하세요 -- 종종 실제 버그를 드러냅니다

---

## 연습문제

1. 제공된 Python 파일에 `ruff check`를 실행하고 모든 경고 수정하기
2. ruff 설정이 포함된 `pyproject.toml` 구성하기
3. `ruff format`을 설정하여 코드 자동 포맷하기
4. ruff 훅이 포함된 pre-commit 설정 만들기

**이전**: [테스트 기초](./07_Testing_Basics.md) | **다음**: [타입 체킹](./09_Type_Checking.md)
