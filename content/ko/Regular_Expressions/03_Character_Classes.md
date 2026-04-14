# 문자 클래스

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 대괄호 `[...]`를 사용하여 사용자 정의 문자 클래스를 정의할 수 있다
2. 문자 범위(`[a-z]`, `[0-9]`)를 사용하여 간결한 패턴을 작성할 수 있다
3. `[^...]`로 문자 클래스를 부정할 수 있다
4. 축약 문자 클래스(`\d`, `\w`, `\s`)와 그 부정형을 사용할 수 있다
5. 대괄호 안팎에서의 문자 클래스 차이를 이해할 수 있다
6. 문자 클래스를 다른 정규식 기능과 결합할 수 있다
7. 문자 클래스에서 유니코드 문자를 처리할 수 있다

---

## 1. 문자 클래스란?

**문자 클래스**(또는 문자 집합)는 정의된 집합에서 **하나의 문자**를 매칭합니다. 대괄호 `[...]`로 감싸서 표현합니다:

```python
import re

text = "bag big bog bug"

# [aeiou]는 임의의 단일 모음과 매칭
print(re.findall(r'b[aeiou]g', text))
# ['bag', 'big', 'bog', 'bug']
```

```
패턴: b[aeiou]g

    분석:
    b         - 리터럴 'b'
    [aeiou]   - 집합 {a, e, i, o, u}에서 임의의 한 문자
    g         - 리터럴 'g'

    "bag" -> b[a]g ✓  (a가 [aeiou]에 포함)
    "beg" -> b[e]g ✓  (e가 [aeiou]에 포함)
    "big" -> b[i]g ✓  (i가 [aeiou]에 포함)
    "bog" -> b[o]g ✓  (o가 [aeiou]에 포함)
    "bug" -> b[u]g ✓  (u가 [aeiou]에 포함)
    "byg" -> b[y]g ✗  (y가 [aeiou]에 미포함)
```

---

## 2. 문자 범위

대괄호 안에서 하이픈을 사용하여 범위를 지정합니다:

```python
import re

# [a-z]는 임의의 소문자와 매칭
print(re.findall(r'[a-z]+', "Hello World 123"))
# ['ello', 'orld']

# [A-Z]는 임의의 대문자와 매칭
print(re.findall(r'[A-Z]+', "Hello World 123"))
# ['H', 'W']

# [0-9]는 임의의 숫자와 매칭
print(re.findall(r'[0-9]+', "Hello World 123"))
# ['123']

# [a-zA-Z]는 임의의 영문자(대소문자)와 매칭
print(re.findall(r'[a-zA-Z]+', "Hello World 123"))
# ['Hello', 'World']
```

```
범위의 원리 (ASCII/유니코드 코드 포인트 기반):

    [a-z]   = a b c d e f g h i j k l m n o p q r s t u v w x y z
               97 ────────────────────────────────────────────> 122

    [A-Z]   = A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
               65 ────────────────────────────────────────────> 90

    [0-9]   = 0 1 2 3 4 5 6 7 8 9
               48 ──────────────> 57
```

---

## 3. 부정 문자 클래스

`[` 바로 뒤에 `^`를 넣으면 집합에 **포함되지 않은** 임의의 문자와 매칭합니다:

```python
import re

# [^aeiou]는 모음이 아닌 임의의 문자와 매칭
text = "Hello World"
print(re.findall(r'[^aeiou]', text))
# ['H', 'l', 'l', ' ', 'W', 'r', 'l', 'd']

# [^0-9]는 숫자가 아닌 임의의 문자와 매칭
print(re.findall(r'[^0-9]+', "abc123def456"))
# ['abc', 'def']
```

```
부정 시각화:

    [aeiou]    매칭: a e i o u
               거부: 나머지 전부

    [^aeiou]   매칭: 나머지 전부
               거부: a e i o u

    [^...]를 "이 문자들을 제외한 모든 것"으로 생각하세요
```

**중요**: `^`는 `[` 바로 뒤 **첫 번째 문자**일 때만 부정을 의미합니다. 대괄호 안의 다른 위치에서는 리터럴 `^`입니다.

---

## 4. 대괄호 안의 특수 문자

대부분의 메타문자는 `[...]` 안에서 특수한 의미를 잃습니다:

```python
import re

# 대괄호 안에서 대부분의 메타문자는 리터럴
text = "a.b a+b a*b a?b"
print(re.findall(r'a[.+*?]b', text))
# ['a.b', 'a+b', 'a*b', 'a?b']
```

`[...]` 안에서 특수한 문자:

```
문자    [...]안에서의 동작            리터럴로 사용하려면
────    ────────────────────        ────────────────────
]       대괄호를 닫음                맨 앞에 배치: []abc] 또는 이스케이프: [\]]
\       이스케이프 문자              이스케이프: [\\]
^       부정 (맨 앞일 때)            맨 앞이 아닌 곳에 배치: [a^b]
-       범위 구분자                  맨 앞/뒤에 배치: [-abc] 또는 [abc-]
```

---

## 5. 축약 문자 클래스

Python 정규식은 자주 쓰는 문자 클래스에 대한 축약 표기를 제공합니다:

```
축약       동등한 표현         의미
─────      ──────────         ───────
\d         [0-9]              임의의 숫자
\D         [^0-9]             숫자가 아닌 문자
\w         [a-zA-Z0-9_]       단어 문자
\W         [^a-zA-Z0-9_]      단어 문자가 아닌 문자
\s         [ \t\n\r\f\v]      공백 문자
\S         [^ \t\n\r\f\v]     공백이 아닌 문자
```

```python
import re

text = "User: alice_99, Age: 25, Email: alice@test.com"

# \d - 숫자
print(re.findall(r'\d+', text))
# ['99', '25']

# \w - 단어 문자 (영문자, 숫자, 밑줄)
print(re.findall(r'\w+', text))
# ['User', 'alice_99', 'Age', '25', 'Email', 'alice', 'test', 'com']

# \s - 공백 문자
print(re.findall(r'\s+', text))
# [' ', ' ', ' ', ' ', ' ']
```

### 대문자 = 부정

패턴은 간단합니다: 소문자 축약은 특정 클래스와 매칭하고, 대문자는 그 반대와 매칭합니다.

```
\d  (숫자)        <-->  \D  (숫자 아닌 것)
\w  (단어 문자)    <-->  \W  (단어 문자 아닌 것)
\s  (공백)         <-->  \S  (공백 아닌 것)
```

---

## 6. 축약 표기와 문자 클래스 결합

대괄호 안에서 축약 표기와 리터럴 문자를 혼합할 수 있습니다:

```python
import re

# 숫자 또는 하이픈 매칭 (전화번호용)
text = "Call 555-867-5309"
print(re.findall(r'[\d-]+', text))
# ['555-867-5309']

# 단어 문자 또는 점 매칭 (파일명용)
text = "file.txt image.png README"
print(re.findall(r'[\w.]+', text))
# ['file.txt', 'image.png', 'README']
```

---

## 7. 점 vs 문자 클래스

점(`.`)은 거의 모든 문자와 매칭됩니다. 문자 클래스는 정밀한 제어를 제공합니다:

```
비교: . vs [...]

    .         \n을 제외한 임의의 문자와 매칭
    [.]       리터럴 점만 매칭
    [^\n]     .과 동등 (\n을 제외한 임의의 문자와 매칭)
    [\s\S]    \n을 포함한 임의의 문자와 매칭
```

```python
import re

text = "a.b acb a\nb"

# . 는 임의의 문자와 매칭 (개행 제외)
print(re.findall(r'a.b', text))     # ['a.b', 'acb']

# [.]는 리터럴 점만 매칭
print(re.findall(r'a[.]b', text))   # ['a.b']

# [\s\S]는 개행을 포함한 모든 것과 매칭
print(re.findall(r'a[\s\S]b', text))  # ['a.b', 'acb', 'a\nb']
```

---

## 8. 유니코드와 문자 클래스

Python 3에서 `\w`, `\d`, `\s`는 기본적으로 유니코드를 인식합니다:

```python
import re

# \w는 유니코드 단어 문자와 매칭
text = "hello world"
print(re.findall(r'\w+', text))
# ['hello', 'world']

# ASCII로만 제한하려면 re.ASCII 플래그 사용
print(re.findall(r'\w+', text, re.ASCII))
# ['hello', 'world']
```

---

## 9. 실전 예제

### 예제 1: 16진수 색상 코드 매칭

```python
import re

css = "colors: #FF0000, #00ff00, #0000FF, #abc, not #xyz"
hex_colors = re.findall(r'#[0-9a-fA-F]{3,6}', css)
print(hex_colors)  # ['#FF0000', '#00ff00', '#0000FF', '#abc']
```

### 예제 2: 공백 정리

```python
import re

text = "  Hello   World   !  "
# 여러 공백을 단일 공백으로 치환
clean = re.sub(r'\s+', ' ', text).strip()
print(f"'{clean}'")  # 'Hello World !'
```

### 예제 3: 이니셜 추출

```python
import re

name = "John Michael Smith"
initials = re.findall(r'[A-Z]', name)
print("".join(initials))  # "JMS"
```

---

## 요약

| 개념 | 문법 | 예시 |
|------|------|------|
| 문자 클래스 | `[abc]` | a, b 또는 c와 매칭 |
| 범위 | `[a-z]` | 소문자와 매칭 |
| 부정 | `[^abc]` | a, b, c를 제외한 모든 것과 매칭 |
| 숫자 | `\d` / `[0-9]` | 임의의 숫자와 매칭 |
| 숫자 아닌 것 | `\D` / `[^0-9]` | 숫자가 아닌 것과 매칭 |
| 단어 문자 | `\w` / `[a-zA-Z0-9_]` | 영문자, 숫자, 밑줄과 매칭 |
| 단어 문자 아닌 것 | `\W` / `[^a-zA-Z0-9_]` | 단어 문자가 아닌 것과 매칭 |
| 공백 | `\s` | 스페이스, 탭, 개행 등과 매칭 |
| 공백 아닌 것 | `\S` | 공백이 아닌 것과 매칭 |

---

## 다음 강의

[04_수량자](./04_Quantifiers.md)에서는 문자나 그룹이 몇 번 반복되어야 하는지 지정하는 방법을 배웁니다.
