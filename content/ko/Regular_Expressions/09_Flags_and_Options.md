# 플래그와 옵션

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. `re.IGNORECASE`(`re.I`)로 대소문자 무시 매칭을 수행할 수 있다
2. `re.MULTILINE`(`re.M`)으로 행 단위 앵커 동작을 적용할 수 있다
3. `re.DOTALL`(`re.S`)로 `.`가 개행도 매칭하게 할 수 있다
4. `re.VERBOSE`(`re.X`)로 읽기 쉬운 패턴을 작성할 수 있다
5. 비트 OR 연산자로 여러 플래그를 결합할 수 있다
6. 패턴 내 인라인 플래그(`(?i)`, `(?m)`, `(?s)`, `(?x)`)를 사용할 수 있다
7. `re.ASCII`와 기본 유니코드 매칭의 차이를 이해할 수 있다
8. 실전 패턴에 플래그를 효과적으로 적용할 수 있다

---

## 1. 정규식 플래그 개요

Python의 `re` 모듈은 패턴 동작을 변경하는 플래그를 제공합니다:

```
플래그             축약      효과
────              ─────     ──────
re.IGNORECASE     re.I      대소문자 무시 매칭
re.MULTILINE      re.M      ^와 $가 행 경계를 매칭
re.DOTALL         re.S      .가 개행 문자도 매칭
re.VERBOSE        re.X      패턴에 주석과 공백 허용
re.ASCII          re.A      \w, \d, \s가 ASCII만 매칭
```

---

## 2. `re.IGNORECASE` (`re.I`)

대소문자에 관계없이 패턴을 매칭합니다:

```python
import re

text = "Python PYTHON python PyThOn"

# 플래그 없이: 대소문자 구분 (기본)
print(re.findall(r'python', text))
# ['python']

# IGNORECASE 사용: 모든 대소문자 매칭
print(re.findall(r'python', text, re.IGNORECASE))
# ['Python', 'PYTHON', 'python', 'PyThOn']
```

### 실전 예제

```python
import re

# 텍스트에서 대소문자 무시 검색
article = """
Python is a programming language.
PYTHON was created by Guido van Rossum.
Learning python is fun!
"""

mentions = re.findall(r'\bpython\b', article, re.I)
print(f"Python 언급 횟수: {len(mentions)}")
# Python 언급 횟수: 3
```

---

## 3. `re.MULTILINE` (`re.M`)

`^`와 `$`가 문자열 경계가 아닌 행 경계에서 매칭되도록 변경합니다:

```python
import re

text = """Line 1: Hello
Line 2: World
Line 3: Python"""

# MULTILINE 없이: ^는 문자열 시작만 매칭
print(re.findall(r'^Line \d', text))
# ['Line 1']

# MULTILINE 사용: ^가 각 행 시작을 매칭
print(re.findall(r'^Line \d', text, re.MULTILINE))
# ['Line 1', 'Line 2', 'Line 3']
```

### 실전 예제

```python
import re

config = """# Database settings
host=localhost
port=5432
# Connection pool
max_connections=10
timeout=30"""

# 주석 행 제거
clean = re.sub(r'^#.*$\n?', '', config, flags=re.M)
print(clean)
```

---

## 4. `re.DOTALL` (`re.S`)

`.`가 개행을 포함한 **모든** 문자와 매칭되게 합니다:

```python
import re

text = """<div>
Hello
World
</div>"""

# DOTALL 없이: .가 \n과 매칭되지 않음
match = re.search(r'<div>(.+)</div>', text)
print(match)  # None (행을 넘어 매칭할 수 없음)

# DOTALL 사용: .가 \n도 매칭
match = re.search(r'<div>(.+)</div>', text, re.DOTALL)
print(match.group(1))
# "\nHello\nWorld\n"
```

```
re.DOTALL 없이:
    . 매칭: a b c 1 2 3 ! @ # (\n 제외한 모든 문자)
    
re.DOTALL 사용:
    . 매칭: a b c 1 2 3 ! @ # \n (진정한 모든 문자)

DOTALL 없이 대안: . 대신 [\s\S] 사용
    [\s\S]는 항상 개행을 포함한 모든 문자와 매칭
```

---

## 5. `re.VERBOSE` (`re.X`)

가독성을 위해 공백과 주석이 있는 패턴을 작성할 수 있습니다:

```python
import re

# VERBOSE 없이: 읽기 어려움
pattern_compact = r'^(?:(?P<scheme>\w+)://)(?P<host>[^/:]+)(?::(?P<port>\d+))?(?P<path>/[^\s?]*)?(?:\?(?P<query>\S+))?$'

# VERBOSE 사용: 읽기 쉽고 문서화됨
pattern_verbose = re.compile(r"""
    ^                           # 문자열 시작
    (?:(?P<scheme>\w+)://)      # 스킴 (http, https, ftp)
    (?P<host>[^/:]+)            # 호스트명
    (?::(?P<port>\d+))?         # 선택적 포트 번호
    (?P<path>/[^\s?]*)?         # 선택적 경로
    (?:\?(?P<query>\S+))?       # 선택적 쿼리 문자열
    $                           # 문자열 끝
""", re.VERBOSE)

url = "https://example.com:8080/api/users?page=1"
match = pattern_verbose.search(url)
if match:
    print(match.groupdict())
```

### VERBOSE 모드 규칙

```
VERBOSE 모드에서:
- 공백이 무시됨 (문자 클래스 안이나 이스케이프된 경우 제외)
- #은 행 끝까지의 주석을 시작
- 리터럴 공백을 매칭하려면 \s, [ ], 또는 \ (이스케이프된 공백) 사용
- 리터럴 #을 매칭하려면 \# 또는 [#] 사용
```

---

## 6. 플래그 결합

비트 OR 연산자 `|`로 여러 플래그를 결합합니다:

```python
import re

text = """Hello World
hello python
HELLO REGEX"""

# IGNORECASE와 MULTILINE 결합
pattern = r'^hello\b'
matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
print(matches)  # ['Hello', 'hello', 'HELLO']

# 세 가지 플래그 결합
html = """<div>
Content with
multiple lines
</div>"""

match = re.search(
    r"""
    <div>       # 여는 태그
    \s*(.+?)    # 내용 (게으른)
    \s*</div>   # 닫는 태그
    """,
    html,
    re.VERBOSE | re.DOTALL | re.IGNORECASE
)
print(match.group(1))
```

---

## 7. 인라인 플래그

패턴 내에 `(?flags)` 형태로 플래그를 직접 삽입할 수 있습니다:

```python
import re

# 인라인 IGNORECASE
print(re.findall(r'(?i)python', "Python PYTHON python"))
# ['Python', 'PYTHON', 'python']

# 인라인 MULTILINE
text = "Line 1\nLine 2\nLine 3"
print(re.findall(r'(?m)^Line \d', text))
# ['Line 1', 'Line 2', 'Line 3']
```

### 인라인 플래그 문자

```
인라인    동등한 플래그
──────    ───────────────
(?i)      re.IGNORECASE
(?m)      re.MULTILINE
(?s)      re.DOTALL
(?x)      re.VERBOSE
(?a)      re.ASCII
(?imsx)   여러 플래그 결합
```

### 범위 지정 인라인 플래그

```python
import re

# 패턴의 일부에만 플래그 적용
# (?i:...) 해당 그룹만 대소문자 무시
pattern = r'Hello (?i:world)'  # "world"만 대소문자 무시
print(re.search(pattern, "Hello WORLD"))   # Match
print(re.search(pattern, "HELLO WORLD"))   # None (Hello는 대소문자 구분)
```

---

## 8. `re.ASCII` (`re.A`)

`\w`, `\d`, `\s`, `\b`가 유니코드 또는 ASCII만 매칭하도록 제어합니다:

```python
import re

# Python 3 기본: 유니코드 매칭
text = "hello world"

# 기본: \w가 유니코드 단어 문자와 매칭
print(re.findall(r'\w+', text))

# ASCII 사용: [a-zA-Z0-9_]로 제한
print(re.findall(r'\w+', text, re.ASCII))
```

---

## 9. 실전 플래그 조합

### 로그 처리 (MULTILINE + VERBOSE)

```python
import re

log = """2024-01-15 08:30:45 ERROR Connection failed
2024-01-15 08:30:46 INFO Retrying
2024-01-15 08:30:47 ERROR Timeout expired"""

pattern = re.compile(r"""
    ^                       # 행 시작
    (\d{4}-\d{2}-\d{2})    # 날짜
    \s+
    (\d{2}:\d{2}:\d{2})    # 시간
    \s+
    (ERROR|WARN)            # 레벨 (에러와 경고만)
    \s+
    (.+)                    # 메시지
    $                       # 행 끝
""", re.VERBOSE | re.MULTILINE)

for match in pattern.finditer(log):
    date, time, level, msg = match.groups()
    print(f"[{level}] {date} {time}: {msg}")
```

### 설정 파서 (VERBOSE + MULTILINE)

```python
import re

config = """
# Server configuration
server.host = localhost
server.port = 8080

# Database configuration
db.host = 192.168.1.100
db.port = 5432
db.name = myapp
"""

pattern = re.compile(r"""
    ^                   # 행 시작
    (?!\s*\#)           # 주석 행이 아님
    (\w+(?:\.\w+)*)     # 키 (점 표기법)
    \s*=\s*             # 등호와 선택적 공백
    (.+?)               # 값
    \s*$                # 행 끝 (후행 공백 제거)
""", re.VERBOSE | re.MULTILINE)

config_dict = dict(pattern.findall(config))
for key, value in config_dict.items():
    print(f"{key} = {value}")
```

---

## 10. 플래그 선택 가이드

```
대소문자 무시 매칭이 필요한가?
└── 예 -> re.IGNORECASE (re.I)

멀티라인 텍스트에서 행별 ^$가 필요한가?
└── 예 -> re.MULTILINE (re.M)

.가 행 바꿈을 넘어 매칭해야 하는가?
└── 예 -> re.DOTALL (re.S)

패턴이 복잡하여 문서화가 필요한가?
└── 예 -> re.VERBOSE (re.X)

ASCII 텍스트만 처리하는가 (유니코드 불필요)?
└── 예 -> re.ASCII (re.A)
```

---

## 요약

| 플래그 | 축약 | 인라인 | 효과 |
|--------|------|--------|------|
| `re.IGNORECASE` | `re.I` | `(?i)` | 대소문자 무시 매칭 |
| `re.MULTILINE` | `re.M` | `(?m)` | `^`/`$`가 행 경계를 매칭 |
| `re.DOTALL` | `re.S` | `(?s)` | `.`가 개행도 매칭 |
| `re.VERBOSE` | `re.X` | `(?x)` | 주석과 공백 허용 |
| `re.ASCII` | `re.A` | `(?a)` | `\w`, `\d`, `\s`가 ASCII만 매칭 |

핵심 정리:
- `|`로 플래그 결합: `re.I | re.M | re.S`
- 30자를 넘는 패턴에는 `re.VERBOSE` 사용
- 인라인 플래그 `(?i)`는 전체 패턴에 영향 (범위 지정은 `(?i:...)`)
- `re.DOTALL`과 `re.MULTILINE`은 서로 다른 목적이며 자주 함께 사용됨

---

## 다음 강의

[10_자주 쓰는 패턴](./10_Common_Patterns.md)에서는 일반적인 검증 및 추출 작업을 위한 패턴을 구성하고 분석합니다.
