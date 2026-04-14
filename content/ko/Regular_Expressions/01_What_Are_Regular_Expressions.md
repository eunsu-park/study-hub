# 정규 표현식이란

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 정규 표현식이 무엇인지, 그 역사적 기원을 설명할 수 있다
2. 소프트웨어 개발에서 정규식의 대표적 활용 사례를 파악할 수 있다
3. 정규식과 유한 오토마타의 관계를 이해할 수 있다
4. Python의 `re` 모듈로 기본 패턴 매칭을 수행할 수 있다
5. `re.match()`, `re.search()`, `re.findall()`의 차이를 구분할 수 있다
6. Match 객체로 매칭된 텍스트와 위치를 추출할 수 있다
7. 원시 문자열(`r""`)을 사용하여 백슬래시 이스케이프 문제를 피할 수 있다

---

## 1. 정규 표현식이란?

**정규 표현식**(regular expression, regex 또는 regexp)은 검색 패턴을 정의하는 문자 시퀀스입니다. 이 패턴을 사용하여 문자열 내의 텍스트를 매칭, 검색, 치환할 수 있습니다.

정규식은 텍스트 패턴을 기술하기 위해 특별히 설계된 미니 언어라고 생각하면 됩니다:

```
패턴:  \d{3}-\d{4}
       ───┬── ─┬─ ───┬── ─┬─
          |    |     |    |
          |    |     |    └── 정확히 4자리 숫자
          |    |     └────── 리터럴 하이픈
          |    └──────────── 정확히 3자리 숫자
          └───────────────── \d는 "임의의 숫자"를 의미

매칭됨:  "555-1234"  "800-5678"  "123-4567"
불일치:  "55-1234"   "5555-123"  "abc-defg"
```

---

## 2. 간략한 역사

정규 표현식은 컴퓨터 과학에 깊은 뿌리를 두고 있습니다:

| 연도 | 사건 |
|------|------|
| 1951 | Stephen Kleene가 수학적 표기법으로 "정규 집합"을 기술 |
| 1968 | Ken Thompson이 QED 텍스트 편집기에 정규식 구현 |
| 1973 | Unix에서 `grep` (Global Regular Expression Print) 탄생 |
| 1986 | POSIX가 정규식 문법 표준화 (BRE와 ERE) |
| 1987 | Larry Wall이 Perl을 만들어 고급 정규식 기능 대중화 |
| 1997 | Philip Hazel이 PCRE (Perl Compatible Regular Expressions) 개발 |
| 2003 | Python의 `re` 모듈이 현재 사용하는 기능 세트로 안정화 |

이론적 기반은 **형식 언어 이론**에서 비롯됩니다: 정규 표현식은 **유한 오토마타**(상태 기계)가 인식할 수 있는 언어 클래스를 정확히 기술합니다.

```
패턴 "ab+c"에 대한 유한 오토마타:

    ┌───┐  a   ┌───┐  b   ┌───┐  b   ┌───┐  c   ╔═══╗
    │ S │ ───> │ 1 │ ───> │ 2 │ ───> │ 2 │ ───> ║ ✓ ║
    └───┘      └───┘      └───┘      └───┘      ╚═══╝
                                       ↑    │
                                       └────┘
                                      ('b'에서 반복)

    S = 시작 상태
    ✓ = 수락 상태 (매칭 성공)
```

---

## 3. 왜 정규 표현식을 배워야 하는가?

정규 표현식은 소프트웨어 개발의 곳곳에 등장합니다:

### 유효성 검사
```
"user@example.com"이 유효한 이메일인가?
"2024-01-15"가 날짜 형식에 맞는가?
"P@ssw0rd!"가 강력한 비밀번호인가?
```

### 검색과 치환
```
문서에서 모든 전화번호 찾기
"color"를 "colour"로 바꾸기 (단, "Colorado"는 제외)
HTML에서 모든 URL 추출
```

### 데이터 추출
```
로그 파일에서 에러 메시지 파싱
혼합 구분자가 있는 CSV에서 필드 추출
릴리스 노트에서 버전 번호 추출
```

### 텍스트 처리
```
소스 코드 토큰화
텍스트를 문장 단위로 분할
사용자 입력 정리 (HTML 태그 제거, 공백 정규화)
```

---

## 4. Python의 `re` 모듈

Python은 정규식 작업을 위해 표준 라이브러리에 `re` 모듈을 제공합니다. 별도 설치가 필요 없습니다.

```python
import re
```

### 4.1 첫 번째 패턴 매칭

```python
import re

text = "The year is 2024 and the month is 12."

# 4자리 숫자 검색
match = re.search(r'\d{4}', text)

if match:
    print(f"발견: {match.group()}")      # 발견: 2024
    print(f"위치: {match.start()}")      # 위치: 12
```

### 4.2 원시 문자열: `r""`이 중요한 이유

정규 표현식에서 백슬래시는 특별한 의미를 가집니다. Python 문자열도 이스케이프 시퀀스에 백슬래시를 사용합니다. 이 때문에 충돌이 발생합니다:

```
원시 문자열 없이:
    "\n"   -> Python이 개행 문자로 해석
    "\\n"  -> Python이 리터럴 \n으로 해석 (정규식 엔진이 보는 값)

원시 문자열 사용:
    r"\n"  -> Python이 \n을 그대로 유지 (정규식 엔진이 보는 값)
```

**정규식 패턴에는 항상 원시 문자열(`r""`)을 사용하세요:**

```python
# 나쁜 예: 원시 문자열 없이
pattern = "\d+"          # \d가 잘못 해석될 수 있음
pattern = "\\d+"         # 동작하지만 가독성이 떨어짐

# 좋은 예: 원시 문자열 사용
pattern = r"\d+"         # 명확하고 올바름
```

시각적 비교:

```
Python 문자열    정규식 엔진이 보는 값
─────────────    ──────────────────────
"\d+"            \d+      (우연히 동작 -- \d는 Python 이스케이프가 아님)
"\b"             ←BELL→   (Python 백스페이스, 단어 경계가 아님!)
r"\b"            \b       (올바름: 단어 경계)
"\\"             \        (백슬래시 하나)
r"\\"            \\       (백슬래시 두 개 -- 아마 의도한 것이 아님)
```

---

## 5. `re` 모듈의 핵심 함수

### 5.1 `re.search()` -- 첫 번째 매칭 찾기

전체 문자열을 스캔하여 첫 번째 매칭을 반환하거나, 없으면 `None`을 반환합니다:

```python
import re

text = "Error 404: Page not found at 15:30:00"

match = re.search(r'\d+', text)
if match:
    print(match.group())  # "404"
```

### 5.2 `re.match()` -- 문자열 시작에서 매칭

문자열의 **시작 부분**에서만 매칭을 시도합니다:

```python
import re

# match()는 문자열의 시작만 검사
print(re.match(r'\d+', "404 error"))     # <Match '404'>
print(re.match(r'\d+', "Error 404"))     # None (숫자로 시작하지 않음)
print(re.search(r'\d+', "Error 404"))    # <Match '404'> (search는 찾아냄)
```

```
re.match() vs re.search():

    문자열: "Error 404: Not Found"
             ^
             |
    match()  여기만 검사
    search() 스캔 ──────────────>
```

### 5.3 `re.findall()` -- 모든 매칭 찾기

겹치지 않는 모든 매칭의 리스트를 반환합니다:

```python
import re

text = "Prices: $10.50, $23.99, $5.00"

prices = re.findall(r'\$\d+\.\d{2}', text)
print(prices)  # ['$10.50', '$23.99', '$5.00']
```

### 5.4 `re.finditer()` -- 매칭 순회하기

Match 객체의 이터레이터를 반환합니다(`findall`보다 상세한 정보 제공):

```python
import re

text = "2024-01-15 Error: Connection failed\n2024-01-16 Info: Retry succeeded"

for match in re.finditer(r'\d{4}-\d{2}-\d{2}', text):
    print(f"날짜: {match.group()} 위치 {match.start()}")

# 날짜: 2024-01-15 위치 0
# 날짜: 2024-01-16 위치 36
```

### 5.5 `re.fullmatch()` -- 전체 문자열 매칭

전체 문자열이 패턴과 일치해야 합니다(검증에 유용):

```python
import re

# 날짜 형식 검증
print(re.fullmatch(r'\d{4}-\d{2}-\d{2}', "2024-01-15"))       # Match
print(re.fullmatch(r'\d{4}-\d{2}-\d{2}', "Date: 2024-01-15")) # None
```

---

## 6. Match 객체

정규식이 매칭되면 Python은 유용한 메서드를 가진 `Match` 객체를 반환합니다:

```python
import re

text = "My phone number is 555-867-5309."
match = re.search(r'(\d{3})-(\d{3})-(\d{4})', text)

if match:
    print(match.group())     # '555-867-5309'  (전체 매칭)
    print(match.group(0))    # '555-867-5309'  (group()과 동일)
    print(match.group(1))    # '555'           (첫 번째 캡처 그룹)
    print(match.group(2))    # '867'           (두 번째 캡처 그룹)
    print(match.group(3))    # '5309'          (세 번째 캡처 그룹)
    print(match.groups())    # ('555', '867', '5309')
    print(match.start())     # 19              (시작 위치)
    print(match.end())       # 31              (끝 위치)
    print(match.span())      # (19, 31)        (시작, 끝 튜플)
```

```
Match 객체 구조:
                                    
    text = "My phone number is 555-867-5309."
                               ↑           ↑
                          start=19      end=31

    .group()  -> "555-867-5309"    전체 매칭
    .group(1) -> "555"             ──┐
    .group(2) -> "867"               ├── 캡처 그룹
    .group(3) -> "5309"            ──┘
    .span()   -> (19, 31)         원본 텍스트에서의 위치
```

---

## 7. 패턴 컴파일

패턴을 여러 번 사용한다면 컴파일하여 성능을 향상시킬 수 있습니다:

```python
import re

# 한 번 컴파일, 여러 번 사용
phone_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')

texts = [
    "Call 555-867-5309 for info",
    "Fax: 555-123-4567",
    "No phone here",
]

for text in texts:
    match = phone_pattern.search(text)
    if match:
        print(f"발견: {match.group()}")
```

`re.compile()`의 장점:
- **성능**: 패턴이 한 번만 컴파일되고, 매 호출마다 다시 컴파일되지 않음
- **가독성**: 패턴에 설명적인 변수 이름 부여 가능
- **재사용**: 동일한 컴파일된 패턴을 여러 곳에서 사용

---

## 8. 간단한 실제 예제

지금까지 배운 것을 결합하여 로그 라인에서 데이터를 추출해 봅시다:

```python
import re

log_line = "[2024-01-15 08:30:45] ERROR server.py:142 - Connection timeout after 30s"

# 패턴 분석:
#   \[(\d{4}-\d{2}-\d{2})\s+  - 대괄호 안의 날짜
#   (\d{2}:\d{2}:\d{2})\]\s+  - 대괄호 안의 시간
#   (\w+)\s+                   - 로그 레벨
#   (\S+):(\d+)\s+-\s+        - 파일:행 번호
#   (.+)                       - 메시지

pattern = re.compile(
    r'\[(\d{4}-\d{2}-\d{2})\s+'   # 날짜
    r'(\d{2}:\d{2}:\d{2})\]\s+'   # 시간
    r'(\w+)\s+'                     # 레벨
    r'(\S+):(\d+)\s+-\s+'          # 파일:행
    r'(.+)'                         # 메시지
)

match = pattern.search(log_line)
if match:
    date, time, level, file, line, message = match.groups()
    print(f"날짜:    {date}")     # 2024-01-15
    print(f"시간:    {time}")     # 08:30:45
    print(f"레벨:    {level}")    # ERROR
    print(f"파일:    {file}")     # server.py
    print(f"행:      {line}")     # 142
    print(f"메시지:  {message}")  # Connection timeout after 30s
```

---

## 9. 정규식 vs 문자열 메서드

모든 것에 정규식을 사용할 필요는 없습니다. 간단한 작업에는 Python의 문자열 메서드가 더 빠릅니다:

| 작업 | 문자열 메서드 | 정규식 |
|------|-------------|--------|
| "http"로 시작하는지 확인 | `s.startswith("http")` | `re.match(r'http', s)` |
| 정확한 단어 치환 | `s.replace("old", "new")` | `re.sub(r'old', 'new', s)` |
| 단일 구분자로 분할 | `s.split(",")` | `re.split(r',', s)` |
| 부분 문자열 포함 여부 | `"hello" in s` | `re.search(r'hello', s)` |

**경험 법칙**: 고정 텍스트 작업에는 문자열 메서드를 사용하세요. **패턴** -- 가변 텍스트, 선택적 부분, 대안, 반복 -- 이 필요할 때 정규식을 사용하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| 정규 표현식 | 텍스트 매칭을 위한 패턴 언어 |
| `re` 모듈 | Python 내장 정규식 라이브러리 |
| 원시 문자열 (`r""`) | Python이 백슬래시를 해석하는 것을 방지 |
| `re.search()` | 문자열 어디서든 첫 번째 매칭 찾기 |
| `re.match()` | 문자열 시작에서만 매칭 |
| `re.findall()` | 모든 매칭의 리스트 반환 |
| `re.finditer()` | Match 객체의 이터레이터 반환 |
| `re.fullmatch()` | 전체 문자열 매칭 |
| `re.compile()` | 재사용을 위해 패턴 사전 컴파일 |
| Match 객체 | 매칭된 텍스트, 그룹, 위치 정보 포함 |

---

## 다음 강의

[02_리터럴 매칭과 메타문자](./02_Literal_Matching_and_Metacharacters.md)에서는 정규식 패턴의 구성 요소인 리터럴 문자와 메타문자를 다룹니다.
