# 전방탐색과 후방탐색

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 너비 0 단언이 일반 매칭과 어떻게 다른지 이해할 수 있다
2. 긍정 전방탐색 `(?=...)`으로 뒤따라오는 내용을 단언할 수 있다
3. 부정 전방탐색 `(?!...)`으로 뒤따라오지 않는 내용을 단언할 수 있다
4. 긍정 후방탐색 `(?<=...)`으로 앞에 오는 내용을 단언할 수 있다
5. 부정 후방탐색 `(?<!...)`으로 앞에 오지 않는 내용을 단언할 수 있다
6. 여러 탐색 단언을 하나의 패턴에 결합할 수 있다
7. Python에서 후방탐색의 길이 제한을 이해할 수 있다
8. 탐색을 사용하여 실전 텍스트 처리 문제를 해결할 수 있다

---

## 1. 탐색 단언이란?

탐색 단언은 현재 위치 전후에 패턴이 존재하는지 확인하면서, **문자를 소비하지 않습니다**. 앵커처럼 너비가 0입니다:

```
유형                 구문          의미
────                 ──────        ───────
긍정 전방탐색        (?=...)       뒤따라오는 것이 반드시 매칭
부정 전방탐색        (?!...)       뒤따라오는 것이 매칭되면 안 됨
긍정 후방탐색        (?<=...)      앞에 오는 것이 반드시 매칭
부정 후방탐색        (?<!...)      앞에 오는 것이 매칭되면 안 됨
```

```
너비 0 개념:

    일반 매칭:     f o o b a r
                   ─────────── 소비됨 (커서가 이동)

    전방탐색:      f o o | b a r
                          ↑
                   커서가 여기에 머무르며, "앞을 엿봄"
```

---

## 2. 긍정 전방탐색: `(?=...)`

`(?=...)` 안의 패턴이 앞쪽에서 매칭 **가능한** 위치를 매칭합니다:

```python
import re

# "bar"가 뒤따라오는 "foo"만 매칭
text = "foobar foobaz foo"
print(re.findall(r'foo(?=bar)', text))
# ['foo']  -- bar 앞의 foo만

# 참고: "bar"는 매칭 결과에 포함되지 않음!
match = re.search(r'foo(?=bar)', text)
print(match.group())  # "foo" ("foobar"가 아님)
```

### 실전 예제

```python
import re

# 쉼표가 뒤따라오는 단어 찾기
text = "apple, banana, cherry and grape"
print(re.findall(r'\w+(?=,)', text))
# ['apple', 'banana']

# 단위가 뒤따라오는 숫자 찾기
text = "100px, 50em, 200px, 75%"
print(re.findall(r'\d+(?=px)', text))
# ['100', '200']  -- px가 붙은 숫자만
```

---

## 3. 부정 전방탐색: `(?!...)`

`(?!...)` 안의 패턴이 앞쪽에서 매칭 **불가능한** 위치를 매칭합니다:

```python
import re

# "bar"가 뒤따라오지 않는 "foo"만 매칭
text = "foobar foobaz foo"
print(re.findall(r'foo(?!bar)', text))
# ['foo', 'foo']  -- foobaz와 독립된 foo
```

### 흔한 부정 전방탐색 패턴

```python
import re

# 주석이 아닌 행 매칭
code = """# This is a comment
print("hello")
# Another comment
x = 42"""
non_comments = re.findall(r'^(?!#).*$', code, re.MULTILINE)
print(non_comments)
# ['print("hello")', 'x = 42']
```

---

## 4. 긍정 후방탐색: `(?<=...)`

`(?<=...)` 안의 패턴이 뒤쪽에서 매칭 **가능한** 위치를 매칭합니다:

```python
import re

# "$" 뒤에 오는 숫자만 매칭
text = "Price: $50, Quantity: 10, Total: $500"
print(re.findall(r'(?<=\$)\d+', text))
# ['50', '500']  -- $ 뒤의 숫자만
```

```
긍정 후방탐색 시각화:

    패턴: (?<=\$)\d+

    "Price: $50, Quantity: 10"
            ↑──
            └── 후방탐색이 숫자 앞에 "$"가 있는지 확인
            
    "$50" -> (?<=\$) 확인: 앞에 $가 있는가? 예 -> "50" 매칭
    "10"  -> (?<=\$) 확인: 앞에 $가 있는가? 아니오 -> 건너뜀
```

### 더 많은 예제

```python
import re

# "=" 뒤의 값 추출
text = "name=John age=30 city=NYC"
print(re.findall(r'(?<=\=)\w+', text))
# ['John', '30', 'NYC']

# 괄호 안의 텍스트 추출 (후방탐색 + 전방탐색)
text = "Hello (World) and (Python)"
print(re.findall(r'(?<=\()\w+(?=\))', text))
# ['World', 'Python']
```

---

## 5. 부정 후방탐색: `(?<!...)`

`(?<!...)` 안의 패턴이 뒤쪽에서 매칭 **불가능한** 위치를 매칭합니다:

```python
import re

# "$"가 앞에 오지 않는 숫자 매칭
text = "Price: $50, Quantity: 10, Total: $500"
print(re.findall(r'(?<!\$)\b\d+', text))
# ['10']
```

---

## 6. 후방탐색 길이 제한

Python의 `re` 모듈에서 후방탐색 패턴은 **고정 길이**여야 합니다:

```python
import re

# 유효: 고정 길이 후방탐색
re.findall(r'(?<=abc)\w+', "abcdef")      # OK: "abc"는 3글자
re.findall(r'(?<=\d{3})\w+', "123abc")    # OK: \d{3}은 3글자

# 무효: 가변 길이 후방탐색
try:
    re.findall(r'(?<=\d+)\w+', "123abc")  # 에러!
except re.error as e:
    print(f"에러: {e}")
    # 에러: look-behind requires fixed-width pattern
```

### 가변 길이 후방탐색의 우회 방법

```python
import re

# 가변 후방탐색 대신 캡처 그룹을 사용
text = "price: $1234"

# 불가: (?<=\$\d*)\d+
# 해결책: 캡처 그룹
match = re.search(r'\$(\d+)', text)
print(match.group(1))  # "1234"
```

---

## 7. 여러 탐색 결합

여러 탐색 단언을 연결할 수 있습니다:

```python
import re

# 비밀번호 검증: 대문자, 소문자, 숫자, 특수문자가 최소 하나씩
pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%]).{8,}$'

print(bool(re.match(pattern, "P@ssw0rd")))   # True
print(bool(re.match(pattern, "password")))    # False
print(bool(re.match(pattern, "SHORT1!")))     # False
```

```
동일 위치에서의 다중 전방탐색:

    패턴: ^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%]).{8,}$

    위치 ^:
    ├── (?=.*[A-Z])    앞을 봄: 대문자가 있는가? ✓
    ├── (?=.*[a-z])    앞을 봄: 소문자가 있는가? ✓
    ├── (?=.*\d)       앞을 봄: 숫자가 있는가? ✓
    ├── (?=.*[!@#$%])  앞을 봄: 특수문자가 있는가? ✓
    └── .{8,}$         이제 실제로 8+ 문자를 끝까지 매칭

    모든 전방탐색이 동일한 위치(문자열 시작)에서 확인합니다.
    문자를 소비하지 않으므로 모두 ^에서 시작합니다.
```

---

## 8. 치환에서의 탐색

탐색은 텍스트를 소비하지 않으므로 `re.sub()`에서 강력합니다:

```python
import re

# 큰 숫자에 쉼표 추가: 1234567 -> 1,234,567
def add_commas(n):
    return re.sub(r'(?<=\d)(?=(\d{3})+(?!\d))', ',', str(n))

print(add_commas(1234567))      # "1,234,567"
print(add_commas(1000000000))   # "1,000,000,000"
print(add_commas(42))           # "42"
```

### 더 많은 치환 예제

```python
import re

# 대문자 앞에 공백 삽입 (camelCase -> camel Case)
text = "camelCaseToSeparateWords"
result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
print(result)  # "camel Case To Separate Words"
```

---

## 9. 실전 예제

### 예제 1: 통화 기호 없이 가격 추출

```python
import re

text = "Items: $19.99, EUR29.50, $5.00, JPY1500"

# $ 기호 뒤의 숫자 얻기
usd_prices = re.findall(r'(?<=\$)\d+\.?\d*', text)
print(f"USD: {usd_prices}")  # ['19.99', '5.00']

# EUR 뒤의 숫자 얻기
eur_prices = re.findall(r'(?<=EUR)\d+\.?\d*', text)
print(f"EUR: {eur_prices}")  # ['29.50']
```

### 예제 2: 파일 확장자 검증

```python
import re

files = ["report.pdf", "data.csv", "image.exe", "script.py", "notes.txt"]

# .exe로 끝나지 않는 파일명 매칭
safe_files = [f for f in files if re.search(r'^.*(?<!\.exe)$', f)]
print(safe_files)  # ['report.pdf', 'data.csv', 'script.py', 'notes.txt']
```

---

## 10. 탐색 치트시트

```
단언              방향          긍정/부정         예시
─────────         ─────────    ────────────      ───────
(?=pattern)       전방 →       긍정 (반드시)      \w+(?=\.)
(?!pattern)       전방 →       부정 (아니어야)    \d+(?!px)
(?<=pattern)      후방 ←       긍정 (반드시)      (?<=\$)\d+
(?<!pattern)      후방 ←       부정 (아니어야)    (?<!\d)\w+

핵심 규칙:
- 탐색은 너비 0 (문자를 소비하지 않음)
- Python의 re에서 후방탐색은 고정 길이여야 함
- 여러 탐색을 동일 위치에서 연결 가능
- 탐색 안에 임의의 패턴 사용 가능 (그룹, 수량자 등)
```

---

## 요약

| 단언 | 구문 | 의미 |
|------|------|------|
| 긍정 전방탐색 | `(?=...)` | 뒤따라오는 것이 반드시 매칭 |
| 부정 전방탐색 | `(?!...)` | 뒤따라오는 것이 매칭되면 안 됨 |
| 긍정 후방탐색 | `(?<=...)` | 앞에 오는 것이 반드시 매칭 |
| 부정 후방탐색 | `(?<!...)` | 앞에 오는 것이 매칭되면 안 됨 |

핵심 정리:
- 탐색은 너비 0 -- 문자를 소비하지 않음
- Python 후방탐색은 고정 길이 패턴이 필요
- 복잡한 검증에 여러 전방탐색을 연결 (예: 비밀번호)
- `re.sub()`에서 탐색을 사용하면 텍스트를 치환하지 않고 삽입 가능

---

## 다음 강의

[08_치환과 분할](./08_Substitution_and_Splitting.md)에서는 `re.sub()`과 `re.split()`을 사용한 텍스트 변환을 마스터합니다.
