# 앵커와 경계

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 단일 행 및 멀티라인 모드에서 `^`와 `$` 앵커를 사용할 수 있다
2. 단어 경계 `\b`를 적용하여 완전한 단어를 매칭할 수 있다
3. 비단어 경계 `\B`를 사용하여 부분 단어를 매칭할 수 있다
4. `\A`, `\Z`와 `^`, `$`의 차이를 이해할 수 있다
5. 앵커를 수량자 및 문자 클래스와 결합할 수 있다
6. 멀티라인 모드(`re.MULTILINE`)를 행 단위 매칭에 적용할 수 있다
7. 경계 단언을 사용하여 실전 문제를 해결할 수 있다

---

## 1. 앵커란?

앵커는 문자열의 **위치**를 매칭하며, 문자를 매칭하지 않습니다. 너비가 0이므로 텍스트를 소비하지 않습니다:

```
문자열:  H e l l o ,   W o r l d !
         ↑                         ↑
         ^                         $
     문자열 시작                문자열 끝

         ↑   ↑ ↑   ↑ ↑ ↑   ↑   ↑ ↑ ↑ ↑ ↑
         \b  \b\b   \b\b\b  \b  \b\b \b\b\b
         단어 경계 (\w와 \W 사이)
```

---

## 2. 시작과 끝 앵커: `^`와 `$`

### `^` -- 문자열의 시작

```python
import re

lines = [
    "Python is great",
    "I love Python",
    "Python rocks",
]

for line in lines:
    if re.search(r'^Python', line):
        print(f"Python으로 시작: {line}")

# Python으로 시작: Python is great
# Python으로 시작: Python rocks
```

### `$` -- 문자열의 끝

```python
import re

files = ["report.pdf", "data.csv", "image.pdf", "notes.txt"]

for f in files:
    if re.search(r'\.pdf$', f):
        print(f"PDF 파일: {f}")

# PDF 파일: report.pdf
# PDF 파일: image.pdf
```

### `^`와 `$`를 결합한 전체 문자열 검증

```python
import re

def validate_username(username):
    """사용자명: 3~16자 영숫자, 밑줄 허용."""
    return bool(re.fullmatch(r'^[a-zA-Z]\w{2,15}$', username))

test_cases = [
    ("alice", True),
    ("Bob_99", True),
    ("ab", False),        # 너무 짧음
    ("1alice", False),    # 숫자로 시작
    ("a" * 17, False),   # 너무 김
]

for username, expected in test_cases:
    result = validate_username(username)
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] '{username}' -> {result}")
```

---

## 3. 멀티라인 모드: `re.MULTILINE`

`re.MULTILINE` 없이는 `^`와 `$`가 **전체 문자열**의 시작과 끝만 매칭합니다. `re.MULTILINE`을 사용하면 **각 행**의 시작과 끝을 매칭합니다:

```python
import re

text = """Line 1: Hello
Line 2: World
Line 3: Python"""

# MULTILINE 없이: ^ 는 문자열 시작만 매칭
print(re.findall(r'^Line \d', text))
# ['Line 1']

# MULTILINE 사용: ^ 는 각 행 시작을 매칭
print(re.findall(r'^Line \d', text, re.MULTILINE))
# ['Line 1', 'Line 2', 'Line 3']
```

```
re.MULTILINE 없이:

    "Line 1: Hello\nLine 2: World\nLine 3: Python"
     ^                                             $
     여기만                                  여기만

re.MULTILINE 사용:

    "Line 1: Hello\nLine 2: World\nLine 3: Python"
     ^             ^              ^                $
     ^             $^             $^               $
     각 \n이 새로운 행 경계를 생성
```

### 패턴과 일치하는 행 추출

```python
import re

log = """2024-01-15 INFO: Server started
2024-01-15 ERROR: Connection failed
2024-01-16 INFO: Request received
2024-01-16 ERROR: Timeout expired
2024-01-16 WARN: High memory usage"""

# 모든 ERROR 행 찾기
errors = re.findall(r'^.*ERROR.*$', log, re.MULTILINE)
for e in errors:
    print(e)

# 2024-01-15 ERROR: Connection failed
# 2024-01-16 ERROR: Timeout expired
```

---

## 4. `\A`와 `\Z` -- 절대 앵커

`\A`와 `\Z`는 멀티라인 모드에서도 항상 **전체 문자열**의 시작/끝만 매칭합니다:

```python
import re

text = """First line
Second line
Third line"""

# MULTILINE의 ^는 각 행 시작을 매칭
print(re.findall(r'^\w+', text, re.MULTILINE))
# ['First', 'Second', 'Third']

# \A는 항상 절대적 시작만 매칭
print(re.findall(r'\A\w+', text, re.MULTILINE))
# ['First']
```

```
앵커 비교:

    모드          ^/$ 매칭              \A/\Z 매칭
    ────          ─────────            ───────────
    기본          문자열 시작/끝         문자열 시작/끝
    MULTILINE     행 시작/끝            문자열 시작/끝 (변경 없음)
```

---

## 5. 단어 경계: `\b`

`\b` 앵커는 **단어 문자와 비단어 문자 사이의 경계**를 매칭합니다:

```python
import re

text = "cat scatter category caterpillar"

# \b 없이: 단어 안의 "cat"도 매칭
print(re.findall(r'cat', text))
# ['cat', 'cat', 'cat', 'cat']

# \b 사용: 완전한 단어 "cat"만 매칭
print(re.findall(r'\bcat\b', text))
# ['cat']
```

```
단어 경계 시각화:

    c a t   s c a t t e r   c a t e g o r y
    ↑     ↑ ↑             ↑ ↑
    \b    \b \b            \b \b

    \bcat\b  여기서만 매칭:
    [cat]  scatter  category  caterpillar
     ───
     ✓       ✗        ✗         ✗
```

### 자주 쓰는 `\b` 패턴

```python
import re

# 완전한 단어만 매칭
text = "I love JavaScript, not just Java"
print(re.findall(r'\bJava\b', text))
# ['Java']  ("JavaScript" 제외)

# 접두사로 시작하는 단어 매칭
text = "preview, preprocess, present, compress"
print(re.findall(r'\bpre\w+', text))
# ['preview', 'preprocess', 'present']

# 접미사로 끝나는 단어 매칭
text = "running, jumping, sing, nothing"
print(re.findall(r'\w+ing\b', text))
# ['running', 'jumping', 'sing', 'nothing']
```

---

## 6. 비단어 경계: `\B`

`\B`는 `\b`가 매칭되지 **않는** 위치를 매칭합니다 -- 단어 내부 또는 외부의 위치:

```python
import re

text = "cat scatter category caterpillar"

# \B: 단어 경계가 아닌 곳 -- 다른 단어 내부의 "cat" 매칭
print(re.findall(r'\Bcat\B', text))
# ['cat']  ("scatter"에서만 "cat"이 완전히 내부에 있음)

print(re.findall(r'\Bcat', text))
# ['cat', 'cat']  (scatter, category)
```

---

## 7. 앵커와 다른 기능 결합

### 형식 검증

```python
import re

# 간소화된 이메일 검증
emails = ["user@example.com", "@invalid", "user@", "a@b.c"]
for email in emails:
    if re.fullmatch(r'^[\w.+-]+@[\w-]+\.[\w.]+$', email):
        print(f"유효:   {email}")
    else:
        print(f"무효: {email}")
```

### 첫/마지막 단어 추출

```python
import re

text = "The quick brown fox jumps over the lazy dog"

# 첫 번째 단어
print(re.search(r'^\w+', text).group())  # "The"

# 마지막 단어
print(re.search(r'\w+$', text).group())  # "dog"

# 각 행의 첫 번째 단어
multiline = """Hello World
Goodbye Moon
Greetings Star"""
print(re.findall(r'^\w+', multiline, re.MULTILINE))
# ['Hello', 'Goodbye', 'Greetings']
```

### 빈 행 찾기

```python
import re

text = """Line 1

Line 3

Line 5"""

# 빈 행 찾기
blank_lines = re.findall(r'^$', text, re.MULTILINE)
print(f"빈 행 수: {len(blank_lines)}")  # 2
```

---

## 8. 치환에서의 앵커

### 후행 공백 제거

```python
import re

code = "def hello():   \n    pass  \n    return True   \n"

# 각 행에서 후행 공백 제거
clean = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
print(repr(clean))
# 'def hello():\n    pass\n    return True\n'
```

---

## 9. 실전 예제

### 예제 1: IP 주소 형식 검증

```python
import re

def is_valid_ip_format(ip):
    """문자열이 IPv4 주소처럼 보이는지 확인 (기본 형식 검사)."""
    return bool(re.fullmatch(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ip))

tests = ["192.168.1.1", "10.0.0.1", "999.999.999.999", "1.2.3", "a.b.c.d"]
for ip in tests:
    print(f"{ip:20s} -> {is_valid_ip_format(ip)}")
```

### 예제 2: 문장 찾기

```python
import re

text = "Hello world. How are you? I'm fine! Thanks."

# 문장 찾기 (. ? 또는 !로 끝나는)
sentences = re.findall(r'[A-Z][^.!?]*[.!?]', text)
for s in sentences:
    print(s)
```

### 예제 3: 변수명 매칭

```python
import re

code = "count = 0; _private = True; 2bad = False; my_var = 42"

# 유효한 Python 식별자: 문자 또는 밑줄로 시작
identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
print(identifiers)
# ['count', '_private', 'bad', 'False', 'my_var']
```

---

## 요약

| 앵커 | 의미 | 멀티라인 동작 |
|------|------|--------------|
| `^` | 문자열의 시작 | 각 행의 시작 (`re.MULTILINE` 사용 시) |
| `$` | 문자열의 끝 | 각 행의 끝 (`re.MULTILINE` 사용 시) |
| `\A` | 문자열의 절대 시작 | 항상 문자열의 시작 |
| `\Z` | 문자열의 절대 끝 | 항상 문자열의 끝 |
| `\b` | 단어 경계 | 해당 없음 (멀티라인에 영향받지 않음) |
| `\B` | 비단어 경계 | 해당 없음 (멀티라인에 영향받지 않음) |

핵심 정리:
- 앵커는 문자가 아닌 **위치**를 매칭
- `\b`는 **완전한 단어** 매칭에 필수적
- 텍스트를 행 단위로 처리할 때 `re.MULTILINE` 사용
- `\A`와 `\Z`는 멀티라인 모드의 영향을 받지 않음

---

## 다음 강의

[06_그룹과 캡처](./06_Groups_and_Capturing.md)에서는 패턴의 일부를 그룹화하고 캡처된 텍스트를 추출하는 방법을 배웁니다.
