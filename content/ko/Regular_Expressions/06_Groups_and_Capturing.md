# 그룹과 캡처

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 괄호 `()`를 사용하여 캡처 그룹을 생성할 수 있다
2. `match.group()`과 `match.groups()`로 캡처된 텍스트를 추출할 수 있다
3. 역참조(`\1`, `\2`)를 적용하여 반복된 텍스트를 매칭할 수 있다
4. `(?P<name>...)` 구문으로 명명 그룹을 생성할 수 있다
5. 비캡처 그룹 `(?:...)`을 캡처 없이 그룹화에 사용할 수 있다
6. 중첩 그룹과 그룹 번호 매기기를 이해할 수 있다
7. 그룹에 수량자와 교대를 적용할 수 있다
8. `re.findall()`의 그룹 관련 동작을 이해할 수 있다

---

## 1. 기본 캡처 그룹

괄호 `()`는 두 가지 목적의 **캡처 그룹**을 생성합니다:
1. **그룹화**: 여러 문자를 하나의 단위로 취급
2. **캡처**: 매칭된 텍스트를 나중에 사용하기 위해 저장

```python
import re

text = "Date: 2024-01-15"

# 그룹 없이: 전체 매칭만 얻음
match = re.search(r'\d{4}-\d{2}-\d{2}', text)
print(match.group())  # "2024-01-15"

# 그룹 사용: 개별 부분을 캡처
match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
print(match.group())   # "2024-01-15"  (전체 매칭)
print(match.group(1))  # "2024"        (년도)
print(match.group(2))  # "01"          (월)
print(match.group(3))  # "15"          (일)
print(match.groups())  # ('2024', '01', '15')
```

```
패턴: (\d{4})-(\d{2})-(\d{2})

    2  0  2  4  -  0  1  -  1  5
    ──────────     ─────     ─────
    그룹 1         그룹 2    그룹 3
    \d{4}          \d{2}     \d{2}

    group(0) = "2024-01-15"   <- 전체 매칭
    group(1) = "2024"         <- 첫 번째 (...)
    group(2) = "01"           <- 두 번째 (...)
    group(3) = "15"           <- 세 번째 (...)
```

---

## 2. 그룹 번호 매기기

그룹은 **여는 괄호**의 위치에 따라 왼쪽에서 오른쪽으로 번호가 매겨집니다:

```python
import re

text = "John Smith (age 30)"
pattern = r'((\w+)\s(\w+))\s\(age\s(\d+)\)'
match = re.search(pattern, text)

print(match.group(0))  # "John Smith (age 30)" - 전체 매칭
print(match.group(1))  # "John Smith"          - 외부 그룹
print(match.group(2))  # "John"                - 첫 번째 내부 그룹
print(match.group(3))  # "Smith"               - 두 번째 내부 그룹
print(match.group(4))  # "30"                  - 나이 그룹
```

```
그룹 번호 매기기 (여는 괄호를 왼쪽에서 오른쪽으로 셈):

    ( ( \w+ ) \s ( \w+ ) ) \s \( age \s ( \d+ ) \)
    ↑ ↑           ↑               ↑
    1 2           3               4
```

---

## 3. 그룹이 있는 `re.findall()`

`findall()`이 그룹을 만나면 전체 매칭이 아닌 캡처된 그룹을 반환합니다:

```python
import re

text = "2024-01-15 and 2024-12-31"

# 그룹 없이: 전체 매칭 반환
print(re.findall(r'\d{4}-\d{2}-\d{2}', text))
# ['2024-01-15', '2024-12-31']

# 하나의 그룹: 그룹 내용의 리스트 반환
print(re.findall(r'(\d{4})-\d{2}-\d{2}', text))
# ['2024', '2024']  -- 캡처된 년도만!

# 여러 그룹: 튜플의 리스트 반환
print(re.findall(r'(\d{4})-(\d{2})-(\d{2})', text))
# [('2024', '01', '15'), ('2024', '12', '31')]
```

그룹을 사용하면서 전체 매칭을 얻으려면:
- `finditer()`를 사용하여 각 매칭에서 `.group()` 호출
- 비캡처 그룹 `(?:...)` 사용

---

## 4. 명명 그룹: `(?P<name>...)`

명명 그룹은 패턴을 더 읽기 쉽고 유지보수하기 쉽게 만듭니다:

```python
import re

text = "2024-01-15 08:30:45"
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\s+(?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})'

match = re.search(pattern, text)
if match:
    print(match.group('year'))    # "2024"
    print(match.group('month'))   # "01"
    print(match.group('day'))     # "15"
    print(match.groupdict())
    # {'year': '2024', 'month': '01', 'day': '15',
    #  'hour': '08', 'min': '30', 'sec': '45'}
```

### 치환에서의 명명 그룹

```python
import re

# 날짜 형식 변환: YYYY-MM-DD -> DD/MM/YYYY
text = "Date: 2024-01-15"
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
result = re.sub(pattern, r'\g<day>/\g<month>/\g<year>', text)
print(result)  # "Date: 15/01/2024"
```

```
명명 그룹 구문:

    정의:      (?P<name>pattern)
    참조:      \g<name> (치환문에서)
    접근:      match.group('name')
    딕셔너리:  match.groupdict()
```

---

## 5. 비캡처 그룹: `(?:...)`

그룹화가 필요하지만 캡처는 필요 없을 때 사용합니다:

```python
import re

# 캡처 그룹: findall이 캡처된 텍스트 반환
print(re.findall(r'(https?)://(\S+)', "http://a.com https://b.com"))
# [('http', 'a.com'), ('https', 'b.com')]

# 비캡처 그룹: findall이 전체 매칭 반환
print(re.findall(r'(?:https?)://\S+', "http://a.com https://b.com"))
# ['http://a.com', 'https://b.com']
```

### 비캡처 그룹의 활용

```python
import re

# 교대와 함께 (?:...) 사용
text = "gray grey"
print(re.findall(r'gr(?:a|e)y', text))    # ['gray', 'grey']
print(re.findall(r'gr(a|e)y', text))      # ['a', 'e']  -- 의도와 다름!

# 수량자와 함께 (?:...) 사용
text = "ababab cd abab"
print(re.findall(r'(?:ab)+', text))        # ['ababab', 'abab']
print(re.findall(r'(ab)+', text))          # ['ab', 'ab']  -- 마지막 캡처만
```

---

## 6. 역참조: `\1`, `\2`, ...

역참조는 이전 그룹이 캡처한 **동일한 텍스트**와 매칭합니다:

```python
import re

# \1은 그룹 1이 매칭한 것을 다시 참조
# 중복 단어 찾기
text = "the the cat sat sat on the the mat"
print(re.findall(r'\b(\w+)\s+\1\b', text))
# ['the', 'sat', 'the']
```

```
역참조 시각화:

    패턴: \b(\w+)\s+\1\b

    "the the" -> (\w+)가 "the" 캡처, \1이 "the"인지 확인
                  ───                        ───
                 그룹 1           그룹 1과 정확히 일치해야 함

    "the cat" -> (\w+)가 "the" 캡처, \1이 "the"인지 확인
                  ───                        ───
                 "the"               "cat" != "the"  ✗
```

### 더 많은 역참조 예제

```python
import re

# 매칭되는 닫기 태그가 있는 HTML 태그 매칭
html = "<b>bold</b> <i>italic</i> <b>broken</i>"
pattern = r'<(\w+)>.*?</\1>'
print(re.findall(pattern, html))
# ['b', 'i']  -- 깨진 태그는 매칭되지 않음

# 반복된 문자 매칭 ("aa", "bb" 등)
text = "aardvark bookkeeper"
print(re.findall(r'(.)\1', text))
# ['a', 'o', 'k', 'e']

# 인용된 문자열 매칭 (양쪽 동일한 인용부호)
text = '''She said "hello" and 'goodbye' but not "mixed'"""
pattern = r'''(["'])(.*?)\1'''
print(re.findall(pattern, text))
# [('"', 'hello'), ("'", 'goodbye')]
```

---

## 7. 수량자가 있는 그룹

그룹이 수량자로 반복되면 **마지막** 캡처만 저장됩니다:

```python
import re

# (ab)+ 는 마지막 "ab"만 캡처
match = re.search(r'(ab)+', "ababab")
print(match.group())   # "ababab"  (전체 매칭)
print(match.group(1))  # "ab"     (마지막 캡처만)
```

---

## 8. 실전 예제

### 예제 1: 키-값 쌍 파싱

```python
import re

config = """
host=localhost
port=5432
database=mydb
user=admin
password=s3cret
"""

pairs = re.findall(r'^(\w+)=(.+)$', config, re.MULTILINE)
config_dict = dict(pairs)
print(config_dict)
# {'host': 'localhost', 'port': '5432', 'database': 'mydb',
#  'user': 'admin', 'password': 's3cret'}
```

### 예제 2: URL 구성 요소 추출

```python
import re

url = "https://www.example.com:8080/path/to/page?q=search&lang=en"
pattern = r'(?P<scheme>\w+)://(?P<host>[^/:]+)(?::(?P<port>\d+))?(?P<path>/[^?]*)?(?:\?(?P<query>.+))?'

match = re.search(pattern, url)
if match:
    for key, value in match.groupdict().items():
        print(f"{key:10s}: {value}")
```

### 예제 3: 이름 순서 바꾸기

```python
import re

names = "Smith, John\nDoe, Jane\nPark, Eunsu"
# "성, 이름" -> "이름 성"
result = re.sub(r'(\w+),\s*(\w+)', r'\2 \1', names)
print(result)
# John Smith
# Jane Doe
# Eunsu Park
```

### 예제 4: 중복 단어 찾기

```python
import re

text = "The the quick brown fox fox jumped over the lazy lazy dog"
dupes = re.findall(r'\b(\w+)\s+\1\b', text, re.IGNORECASE)
print(f"중복 단어: {dupes}")
# 중복 단어: ['The', 'fox', 'lazy']
```

---

## 9. 그룹 참조 치트시트

```
구문                용도                 예시
──────              ───────             ───────
(...)               캡처 그룹           (\d+) 숫자를 캡처
(?:...)             비캡처 그룹         (?:ab)+ 캡처 없이 그룹화
(?P<name>...)       명명 그룹           (?P<year>\d{4})
\1, \2              역참조             (\w+)\s+\1
(?P=name)           명명 역참조         (?P=tag)
\g<1>, \g<name>     치환문 참조         re.sub(r'(...)', r'\g<1>', ...)
match.group(n)      번호로 접근         match.group(1)
match.group('name') 이름으로 접근       match.group('year')
match.groups()      모든 그룹(튜플)     ('2024', '01', '15')
match.groupdict()   명명 그룹(딕셔너리) {'year': '2024', ...}
```

---

## 요약

| 개념 | 구문 | 용도 |
|------|------|------|
| 캡처 그룹 | `(...)` | 매칭된 텍스트를 그룹화하고 캡처 |
| 비캡처 그룹 | `(?:...)` | 캡처 없이 그룹화 |
| 명명 그룹 | `(?P<name>...)` | 이름을 가진 캡처 |
| 역참조 | `\1`, `\2` | 이전 그룹과 동일한 텍스트 매칭 |
| 명명 역참조 | `(?P=name)` | 명명 그룹과 동일한 텍스트 매칭 |
| 그룹 접근 | `.group(n)` | 번호로 캡처된 텍스트 얻기 |
| 전체 그룹 | `.groups()` | 모든 캡처 그룹의 튜플 |
| 그룹 딕셔너리 | `.groupdict()` | 명명 그룹의 딕셔너리 |

---

## 다음 강의

[07_전방탐색과 후방탐색](./07_Lookahead_and_Lookbehind.md)에서는 텍스트를 소비하지 않고 위치 앞뒤를 확인하는 너비 0 단언을 배웁니다.
