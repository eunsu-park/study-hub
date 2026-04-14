# 리터럴 매칭과 메타문자

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 정규식 패턴에서 리터럴 텍스트를 매칭할 수 있다
2. 핵심 메타문자(`. ^ $ | \ ( ) [ ] { } * + ?`)를 식별하고 사용할 수 있다
3. 메타문자를 이스케이프하여 문자 그대로 매칭할 수 있다
4. 점(`.`)을 사용하여 임의의 문자를 매칭할 수 있다
5. 교대 연산자(`|`)를 사용하여 여러 대안을 매칭할 수 있다
6. `^`와 `$` 앵커를 기본 수준에서 이해할 수 있다
7. 리터럴과 메타문자를 조합하여 간단한 패턴을 구성할 수 있다

---

## 1. 리터럴 매칭

가장 단순한 정규식 패턴은 일반 텍스트입니다. 각 문자가 자기 자신과 매칭됩니다:

```python
import re

text = "The cat sat on the mat"

# 리터럴 매칭: "cat" 찾기
match = re.search(r'cat', text)
print(match.group())  # "cat"

# findall: "at"의 모든 출현 찾기
print(re.findall(r'at', text))  # ['at', 'at', 'at']
```

```
패턴: "cat"

    T h e   c a t   s a t   o n   t h e   m a t
                ─────
                매칭

패턴의 각 문자가 정확히 일치해야 합니다:
    c -> c  ✓
    a -> a  ✓
    t -> t  ✓
```

### 대소문자 구분

기본적으로 정규식 매칭은 **대소문자를 구분**합니다:

```python
import re

text = "Python is great. PYTHON is powerful."

print(re.findall(r'Python', text))   # ['Python']
print(re.findall(r'PYTHON', text))   # ['PYTHON']
print(re.findall(r'python', text))   # []
```

---

## 2. 메타문자

정규식에는 특별한 의미를 가진 12개의 문자가 있습니다. 이것이 **메타문자**입니다:

```
. ^ $ * + ? { } [ ] \ | ( )
```

각 메타문자는 고유한 역할을 합니다:

```
메타문자        용도                         예시
─────────       ──────                       ───────
.               임의의 문자 (\n 제외)         a.c  "abc", "a1c" 매칭
^               문자열/행의 시작              ^Hello  "Hello world" 매칭
$               문자열/행의 끝               world$  "Hello world" 매칭
*               0회 이상 반복               ab*c  "ac", "abc", "abbc" 매칭
+               1회 이상 반복               ab+c  "abc", "abbc" 매칭
?               0회 또는 1회               colou?r  "color", "colour" 매칭
{n,m}           n~m회 반복                  \d{2,4}  "12", "123", "1234" 매칭
[...]           문자 클래스                  [aeiou]  임의의 모음 매칭
\               이스케이프 / 특수 시퀀스      \.  리터럴 점 매칭
|               교대 (OR)                   cat|dog  "cat" 또는 "dog" 매칭
(...)           그룹화 / 캡처               (ab)+  "ab", "abab" 매칭
```

---

## 3. 점(`.`) -- 임의의 문자

점은 개행 문자(`\n`)를 제외한 **임의의 단일 문자**와 매칭됩니다:

```python
import re

# . 는 임의의 단일 문자와 매칭
print(re.findall(r'c.t', "cat cot cut c1t c_t c\nt"))
# ['cat', 'cot', 'cut', 'c1t', 'c_t']
# 참고: "c\nt"는 매칭되지 않음 (점은 기본적으로 개행과 매칭되지 않음)
```

```
패턴: c.t

    "cat"  ->  c[a]t  ✓   (a가 .과 매칭)
    "cot"  ->  c[o]t  ✓   (o가 .과 매칭)
    "cut"  ->  c[u]t  ✓   (u가 .과 매칭)
    "c1t"  ->  c[1]t  ✓   (1이 .과 매칭)
    "c t"  ->  c[ ]t  ✓   (공백이 .과 매칭)
    "ct"   ->  c[]t   ✗   (.는 정확히 하나의 문자를 필요로 함)
    "c\nt" ->  c[\n]t ✗   (.는 기본적으로 개행과 매칭되지 않음)
```

### 여러 개의 점

```python
import re

# 점 두 개: 'a'와 'd' 사이에 임의의 두 문자를 매칭
print(re.findall(r'a..d', "abcd a12d a  d aXYd"))
# ['abcd', 'a12d', 'a  d']
```

---

## 4. 앵커: `^`와 `$`

앵커는 문자를 매칭하지 않고 문자열의 **위치**를 매칭합니다:

### `^` -- 문자열의 시작

```python
import re

text = "Python is great"

print(re.search(r'^Python', text))   # Match: 시작에 "Python"
print(re.search(r'^is', text))       # None: "is"가 시작에 없음
```

### `$` -- 문자열의 끝

```python
import re

text = "Python is great"

print(re.search(r'great$', text))    # Match: 끝에 "great"
print(re.search(r'Python$', text))   # None: "Python"이 끝에 없음
```

### `^`와 `$` 결합

```python
import re

# 전체 문자열 매칭
print(re.search(r'^Python$', "Python"))          # Match
print(re.search(r'^Python$', "Python is great")) # None

# 간단한 형식 검증
print(re.search(r'^\d{5}$', "12345"))   # Match (미국 우편번호)
print(re.search(r'^\d{5}$', "1234"))    # None (너무 짧음)
print(re.search(r'^\d{5}$', "123456"))  # None (너무 긺)
```

```
앵커 시각화:

    ^ P y t h o n   i s   g r e a t $
    ↑                                 ↑
    ^= 시작 위치                       $= 끝 위치

    ^Python  -> 매칭됨 (Python이 시작에 있음)
    great$   -> 매칭됨 (great이 끝에 있음)
    ^great   -> 불일치 (great이 시작에 없음)
```

---

## 5. 파이프(`|`) -- 교대

파이프 연산자는 **OR**을 의미합니다 -- 왼쪽 또는 오른쪽 패턴과 매칭합니다:

```python
import re

text = "I have a cat and a dog and a bird"

# "cat" 또는 "dog" 매칭
print(re.findall(r'cat|dog', text))   # ['cat', 'dog']

# "cat" 또는 "dog" 또는 "bird" 매칭
print(re.findall(r'cat|dog|bird', text))   # ['cat', 'dog', 'bird']
```

### 교대의 범위

`|` 연산자는 우선순위가 낮아서 **전체** 표현식을 분할합니다:

```python
import re

# "gray|grey"는 "gray" 또는 "grey"를 매칭
print(re.findall(r'gray|grey', "gray and grey"))   # ['gray', 'grey']

# 그룹과 함께 사용: "gr(a|e)y"
print(re.findall(r'gr(a|e)y', "gray and grey"))    # ['a', 'e']
# 주의: findall은 전체 매칭이 아닌 캡처된 그룹을 반환!

# 전체 매칭을 얻으려면 finditer를 사용:
for m in re.finditer(r'gr(a|e)y', "gray and grey"):
    print(m.group())  # "gray", "grey"
```

---

## 6. 백슬래시(`\`) -- 이스케이프 문자

백슬래시는 두 가지 역할을 합니다:

### 역할 1: 메타문자 이스케이프

메타문자를 문자 그대로 매칭하려면 앞에 `\`를 붙입니다:

```python
import re

# 이스케이프 없이: . 는 임의의 문자와 매칭
print(re.findall(r'.', "a.b"))     # ['a', '.', 'b']

# 이스케이프 사용: \. 는 리터럴 점만 매칭
print(re.findall(r'\.', "a.b"))    # ['.']

# 리터럴 달러 기호 매칭
price = "The price is $9.99"
print(re.search(r'\$\d+\.\d{2}', price).group())  # "$9.99"
```

### 모든 메타문자 이스케이프

```
문자 그대로 매칭하려면     정규식에서 사용
──────────────────        ────────────
.                         \.
^                         \^
$                         \$
*                         \*
+                         \+
?                         \?
{                         \{
}                         \}
[                         \[
]                         \]
\                         \\
|                         \|
(                         \(
)                         \)
```

### 역할 2: 특수 시퀀스

백슬래시는 특수 문자 시퀀스도 생성합니다(이후 강의에서 다룸):

```
\d    임의의 숫자 (0-9)
\w    임의의 단어 문자 (a-z, A-Z, 0-9, _)
\s    임의의 공백 문자 (스페이스, 탭, 개행)
\b    단어 경계
\n    개행
\t    탭
```

### `re.escape()`로 동적 패턴 처리

사용자 입력으로 정규식을 구성할 때는 특수 문자를 이스케이프하세요:

```python
import re

user_input = "Is this real? (yes/no)"

# re.escape()가 모든 메타문자 앞에 백슬래시를 추가
escaped = re.escape(user_input)
print(escaped)  # 'Is\\ this\\ real\\?\\ \\(yes/no\\)'

# 정규식에서 안전하게 사용 가능
text = "Question: Is this real? (yes/no)"
match = re.search(re.escape(user_input), text)
print(match.group())  # "Is this real? (yes/no)"
```

---

## 7. 리터럴과 메타문자 결합

점점 복잡한 패턴을 만들어 봅시다:

### 예제 1: 날짜 형식 패턴 매칭

```python
import re

text = "Events on 2024-01-15 and 2024-12-31"

# \d = 임의의 숫자 (다음 강의에서 다룰 축약 표기)
dates = re.findall(r'\d\d\d\d-\d\d-\d\d', text)
print(dates)  # ['2024-01-15', '2024-12-31']
```

### 예제 2: 간단한 파일 확장자 매칭

```python
import re

files = "report.pdf, data.csv, image.png, script.py"

# .pdf 또는 .csv로 끝나는 파일명 매칭
matches = re.findall(r'\w+\.(?:pdf|csv)', files)
print(matches)  # ['report.pdf', 'data.csv']
```

### 예제 3: 특정 텍스트로 시작하거나 끝나는 행 매칭

```python
import re

log = """INFO: Server started
ERROR: Connection failed
INFO: Request received
ERROR: Timeout"""

# "ERROR"로 시작하는 행 찾기
errors = re.findall(r'^ERROR.*', log, re.MULTILINE)
print(errors)
# ['ERROR: Connection failed', 'ERROR: Timeout']
```

---

## 8. 메타문자 관련 흔한 실수

### 실수 1: 이스케이프를 잊음

```python
import re

# 잘못된 예: . 가 임의의 문자와 매칭
re.search(r'192.168.1.1', "192X168Y1Z1")  # 매칭됨! (의도하지 않음)

# 올바른 예: 점을 이스케이프
re.search(r'192\.168\.1\.1', "192X168Y1Z1")  # None (올바름)
```

### 실수 2: 원시 문자열을 잊음

```python
import re

# 잘못된 예: \b가 Python의 백스페이스 문자
re.findall('\bword\b', "a word here")     # [] -- 동작하지 않음!

# 올바른 예: 원시 문자열 사용
re.findall(r'\bword\b', "a word here")    # ['word']
```

### 실수 3: 교대의 범위

```python
import re

# 잘못된 예: "cat" 또는 "dogs"를 매칭 ("cats"나 "dogs"가 아님)
re.findall(r'cat|dogs', "cats and dogs")  # ['cat', 'dogs']

# 올바른 예: 공유 부분에 그룹 사용
re.findall(r'(?:cat|dog)s', "cats and dogs")  # ['cats', 'dogs']
```

---

## 9. 패턴 구성 전략

정규식 패턴을 만들 때는 다음 접근법을 따르세요:

```
1단계: 매칭하고 싶은 예시를 나열
       "192.168.1.1", "10.0.0.1", "172.16.0.1"

2단계: 고정된 부분 (리터럴) 식별
       점으로 구분된 숫자들

3단계: 가변적인 부분 (메타문자 필요) 식별
       각 위치에 1~3자리 숫자

4단계: 패턴을 조각씩 구성
       \d+\.\d+\.\d+\.\d+

5단계: 매칭과 비매칭 예시 모두로 테스트
       "192.168.1.1"     -> 매칭되어야 함
       "999.999.999.999" -> 매칭됨 (나중에 정교화)
       "abc.def.ghi.jkl" -> 매칭되면 안 됨 ✓
```

---

## 요약

| 개념 | 설명 |
|------|------|
| 리터럴 매칭 | 문자가 자기 자신과 매칭 (`abc`는 "abc"와 매칭) |
| `.` (점) | 개행을 제외한 임의의 단일 문자와 매칭 |
| `^` (캐럿) | 문자열의 시작과 매칭 (멀티라인 모드에서는 각 행의 시작) |
| `$` (달러) | 문자열의 끝과 매칭 (멀티라인 모드에서는 각 행의 끝) |
| `\|` (파이프) | 교대 -- 왼쪽 또는 오른쪽 패턴과 매칭 |
| `\` (백슬래시) | 메타문자를 이스케이프하거나 특수 시퀀스 생성 |
| `re.escape()` | 문자열의 모든 메타문자를 자동으로 이스케이프 |
| 원시 문자열 | 정규식 패턴에는 항상 `r""`을 사용 |

---

## 다음 강의

[03_문자 클래스](./03_Character_Classes.md)에서는 문자 클래스와 축약 표기법을 사용하여 특정 문자 집합을 매칭하는 방법을 배웁니다.
