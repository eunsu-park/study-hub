# 치환과 분할

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. `re.sub()`으로 패턴 기반 텍스트 치환을 수행할 수 있다
2. 치환 문자열에서 캡처 그룹을 참조할 수 있다
3. 동적 치환을 위한 콜백 함수를 작성할 수 있다
4. `re.subn()`으로 치환 횟수를 파악할 수 있다
5. `re.split()`으로 복잡한 패턴에 따라 문자열을 분할할 수 있다
6. `maxsplit`과 그룹을 사용하여 분할 동작을 제어할 수 있다
7. 분할의 엣지 케이스를 처리할 수 있다

---

## 1. `re.sub()`의 기본 치환

`re.sub(pattern, replacement, string, count=0, flags=0)`는 패턴의 모든 매칭을 치환합니다:

```python
import re

text = "I have 3 cats and 2 dogs"

# 모든 숫자를 "#"으로 치환
result = re.sub(r'\d', '#', text)
print(result)  # "I have # cats and # dogs"

# 숫자 시퀀스 치환
result = re.sub(r'\d+', 'N', text)
print(result)  # "I have N cats and N dogs"
```

### `count` 매개변수

치환 횟수를 제한합니다:

```python
import re

text = "aaa bbb ccc aaa bbb"

# 처음 2개만 치환
result = re.sub(r'aaa|bbb', 'XXX', text, count=2)
print(result)  # "XXX XXX ccc aaa bbb"
```

---

## 2. 치환에서의 그룹 참조

치환 문자열에서 `\1`, `\2` 또는 `\g<name>`으로 캡처된 그룹을 참조합니다:

```python
import re

# 이름 순서 바꾸기
text = "Smith, John"
result = re.sub(r'(\w+), (\w+)', r'\2 \1', text)
print(result)  # "John Smith"

# 날짜 형식 변환: MM/DD/YYYY -> YYYY-MM-DD
text = "01/15/2024 and 12/31/2024"
result = re.sub(r'(\d{2})/(\d{2})/(\d{4})', r'\3-\1-\2', text)
print(result)  # "2024-01-15 and 2024-12-31"
```

### 명명 그룹 참조

```python
import re

# \g<name>을 사용한 명명 그룹 참조
text = "2024-01-15"
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
result = re.sub(pattern, r'\g<day>/\g<month>/\g<year>', text)
print(result)  # "15/01/2024"
```

```
치환 문자열에서의 그룹 참조 구문:

    \1, \2, \3      번호 기반 그룹
    \g<1>, \g<2>    번호 기반 그룹 (명시적, 모호함 방지)
    \g<name>        명명 그룹
```

---

## 3. `re.sub()`에서의 콜백 함수

동적 변환을 위해 **함수**를 치환으로 전달합니다:

```python
import re

text = "I have 3 cats and 12 dogs"

# 모든 숫자를 2배로
def double_number(match):
    num = int(match.group())
    return str(num * 2)

result = re.sub(r'\d+', double_number, text)
print(result)  # "I have 6 cats and 24 dogs"
```

콜백은 **Match 객체**를 받고 **문자열**을 반환해야 합니다:

```python
import re

# 온도 변환: "72F" -> "22.2C"
text = "Today: 72F, Tomorrow: 85F, Record: 104F"

def fahrenheit_to_celsius(match):
    f = int(match.group(1))
    c = (f - 32) * 5 / 9
    return f"{c:.1f}C"

result = re.sub(r'(\d+)F', fahrenheit_to_celsius, text)
print(result)  # "Today: 22.2C, Tomorrow: 29.4C, Record: 40.0C"
```

### 람다 콜백

```python
import re

# 'p'로 시작하는 모든 단어를 대문자로
text = "python programming is powerful and practical"
result = re.sub(r'\bp\w+', lambda m: m.group().upper(), text)
print(result)  # "PYTHON PROGRAMMING is POWERFUL and PRACTICAL"
```

---

## 4. `re.subn()` -- 치환과 횟수

`re.subn()`은 `(새_문자열, 치환_횟수)` 튜플을 반환합니다:

```python
import re

text = "cat bat rat cat mat"

result, count = re.subn(r'[cbr]at', 'dog', text)
print(f"결과: {result}")      # "dog dog dog dog mat"
print(f"치환 횟수: {count}")  # 4

# 치환이 발생했는지 확인하는 데 유용
text = "no matches here"
result, count = re.subn(r'\d+', 'NUM', text)
if count == 0:
    print("치환 없음")
```

---

## 5. `re.split()`의 기본 분할

`re.split(pattern, string, maxsplit=0, flags=0)`는 패턴 매칭 위치에서 문자열을 분할합니다:

```python
import re

# 임의의 공백으로 분할
text = "Hello   World\tPython\nRegex"
print(re.split(r'\s+', text))
# ['Hello', 'World', 'Python', 'Regex']

# 선택적 공백이 있는 쉼표로 분할
text = "apple, banana,cherry ,  grape"
print(re.split(r'\s*,\s*', text))
# ['apple', 'banana', 'cherry', 'grape']

# 여러 구분자로 분할
text = "one;two,three:four|five"
print(re.split(r'[;,:|]', text))
# ['one', 'two', 'three', 'four', 'five']
```

### `maxsplit` 매개변수

```python
import re

text = "one,two,three,four,five"

# 최대 3개 부분으로 분할
print(re.split(r',', text, maxsplit=2))
# ['one', 'two', 'three,four,five']
```

---

## 6. 캡처 그룹이 있는 분할

분할 패턴에 캡처 그룹이 포함되면 캡처된 텍스트가 결과에 **포함**됩니다:

```python
import re

text = "one1two2three3four"

# 그룹 없이: 구분자가 제거됨
print(re.split(r'\d', text))
# ['one', 'two', 'three', 'four']

# 그룹 사용: 구분자가 포함됨
print(re.split(r'(\d)', text))
# ['one', '1', 'two', '2', 'three', '3', 'four']
```

```
캡처 그룹이 있는 분할 시각화:

    입력:    "one1two2three"
    패턴:    (\d)

    분할 위치:
    "one" | 1 | "two" | 2 | "three"
           ↑            ↑
           구분자        구분자

    () 없이: ['one', 'two', 'three']    (구분자 버림)
    () 사용: ['one', '1', 'two', '2', 'three']  (구분자 유지)
```

### 실용적 활용: 구분자 유지

```python
import re

# 문장을 분할하되 구두점은 유지
text = "Hello! How are you? I'm fine. Thanks!"
parts = re.split(r'([.!?])\s*', text)
print(parts)
# ['Hello', '!', 'How are you', '?', "I'm fine", '.', 'Thanks', '!', '']

# 구두점을 포함한 문장 재구성
sentences = []
for i in range(0, len(parts) - 1, 2):
    if parts[i]:
        sentences.append(parts[i] + parts[i+1])
print(sentences)
# ['Hello!', 'How are you?', "I'm fine.", 'Thanks!']
```

---

## 7. 분할의 엣지 케이스

### 연속 구분자에서의 빈 문자열

```python
import re

text = "one,,two,,,three"
print(re.split(r',', text))
# ['one', '', 'two', '', '', 'three']

# 빈 문자열 필터링
parts = [p for p in re.split(r',', text) if p]
print(parts)  # ['one', 'two', 'three']
```

### 시작 또는 끝의 패턴

```python
import re

text = ",one,two,three,"
print(re.split(r',', text))
# ['', 'one', 'two', 'three', '']
# 주의: 시작과 끝에 빈 문자열
```

---

## 8. 실전 예제

### 예제 1: 템플릿 변수 치환

```python
import re

template = "Hello, {{name}}! You have {{count}} new messages."
variables = {"name": "Alice", "count": "5"}

def replace_var(match):
    var_name = match.group(1)
    return variables.get(var_name, match.group())

result = re.sub(r'\{\{(\w+)\}\}', replace_var, template)
print(result)  # "Hello, Alice! You have 5 new messages."
```

### 예제 2: HTML 태그 제거

```python
import re

html = "<p>Hello <b>World</b>! Click <a href='url'>here</a>.</p>"

# 모든 HTML 태그 제거
clean = re.sub(r'<[^>]+>', '', html)
print(clean)  # "Hello World! Click here."
```

### 예제 3: 공백 정규화

```python
import re

text = "  Hello   World  \n\n  How   are   you?  \n  "

# 공백 축소, 양끝 제거
clean = re.sub(r'\s+', ' ', text).strip()
print(f"'{clean}'")
# 'Hello World How are you?'
```

### 예제 4: 로그 파일 파싱 및 변환

```python
import re

log = """[2024-01-15 08:30:45] ERROR: Connection refused
[2024-01-15 08:30:46] INFO: Retrying connection
[2024-01-15 08:30:47] ERROR: Connection timeout
[2024-01-15 08:30:50] INFO: Connection established"""

# ERROR 행에서 타임스탬프 추출
errors = re.findall(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ERROR: (.+)',
    log
)
for timestamp, message in errors:
    print(f"{timestamp} - {message}")

# 로그 형식 변환: [timestamp] LEVEL: msg -> timestamp|LEVEL|msg
result = re.sub(
    r'\[(.+?)\] (\w+): (.+)',
    r'\1|\2|\3',
    log
)
print(result)
```

---

## 9. 성능 고려 사항

```python
import re

# 루프에서 사용하는 패턴은 컴파일
pattern = re.compile(r'\b\w+\b')

# 간단한 경우 문자열 메서드 사용 -- 정규식보다 빠름
text = "hello world"
# 좋음: text.replace("hello", "hi")     <- 더 빠름
# 괜찮음: re.sub(r'hello', 'hi', text)  <- 정규식 오버헤드

# 패턴이 필요할 때만 re.sub 사용
# 나쁨:  re.sub(r'hello', 'hi', text)    <- 과도함
# 좋음:  re.sub(r'\bhello\b', 'hi', text) <- 단어 경계가 필요
```

---

## 요약

| 함수 | 용도 | 반환값 |
|------|------|--------|
| `re.sub(pat, repl, s)` | 모든 매칭 치환 | 새 문자열 |
| `re.sub(pat, func, s)` | 콜백으로 치환 | 새 문자열 |
| `re.subn(pat, repl, s)` | 치환 및 횟수 | (새_문자열, 횟수) |
| `re.split(pat, s)` | 패턴으로 분할 | 문자열 리스트 |

핵심 치환 구문:
- `\1`, `\2` -- 번호 기반 그룹 참조
- `\g<1>`, `\g<name>` -- 명시적 그룹 참조
- 콜백 함수는 Match 객체를 받고 문자열을 반환

---

## 다음 강의

[09_플래그와 옵션](./09_Flags_and_Options.md)에서는 패턴 해석 방식을 변경하는 정규식 플래그를 탐구합니다.
