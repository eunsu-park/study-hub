# 수량자

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 기본 수량자(`*`, `+`, `?`)를 사용하여 반복을 제어할 수 있다
2. `{n}`, `{n,}`, `{n,m}`으로 정확한 반복 횟수를 지정할 수 있다
3. 탐욕적 매칭과 게으른(비탐욕적) 매칭을 구분할 수 있다
4. 정규식 엔진이 수량자와 함께 역추적하는 방식을 이해할 수 있다
5. 문자 클래스와 그룹에 수량자를 적용할 수 있다
6. 상황에 맞는 적절한 수량자를 선택할 수 있다
7. 흔한 수량자 함정을 피할 수 있다

---

## 1. 세 가지 기본 수량자

수량자는 앞의 요소가 몇 번 반복되어야 하는지 지정합니다:

```
수량자     의미             매칭
──────     ──────────────   ─────────────────────────
*          0회 이상         "", "a", "aa", "aaa", ...
+          1회 이상         "a", "aa", "aaa", ...
?          0회 또는 1회     "", "a"
```

### `*` -- 0회 이상

```python
import re

# ab*c: b가 0회 이상 등장 가능
print(re.findall(r'ab*c', "ac abc abbc abbbc"))
# ['ac', 'abc', 'abbc', 'abbbc']
```

```
패턴: ab*c

    "ac"     ->  a[]c       ✓  (b가 0개)
    "abc"    ->  a[b]c      ✓  (b가 1개)
    "abbc"   ->  a[bb]c     ✓  (b가 2개)
    "abbbc"  ->  a[bbb]c    ✓  (b가 3개)
    "adc"    ->  a[d]c      ✗  (d는 b가 아님)
```

### `+` -- 1회 이상

```python
import re

# ab+c: b가 최소 1회 이상 등장해야 함
print(re.findall(r'ab+c', "ac abc abbc abbbc"))
# ['abc', 'abbc', 'abbbc']
# 참고: "ac"는 매칭되지 않음 (최소 하나의 b가 필요)
```

### `?` -- 0회 또는 1회

```python
import re

# colou?r: u는 선택적
print(re.findall(r'colou?r', "color and colour"))
# ['color', 'colour']

# https?: s는 선택적
print(re.findall(r'https?://', "http://a.com and https://b.com"))
# ['http://', 'https://']
```

---

## 2. 정확한 반복: `{n}`, `{n,}`, `{n,m}`

반복 횟수를 정밀하게 제어합니다:

```
구문       의미
──────     ───────
{n}        정확히 n회
{n,}       n회 이상
{n,m}      n~m회 (포함)
{,m}       최대 m회 (0~m)
```

```python
import re

# {n}: 정확히 n회
print(re.findall(r'\d{4}', "12 123 1234 12345"))
# ['1234', '1234']

# {n,}: n회 이상
print(re.findall(r'\d{3,}', "1 12 123 1234 12345"))
# ['123', '1234', '12345']

# {n,m}: n~m회
print(re.findall(r'\d{2,4}', "1 12 123 1234 12345"))
# ['12', '123', '1234', '1234']  (탐욕적: 가능한 많이)
```

### 실전 예제

```python
import re

# 미국 우편번호: 정확히 5자리
zip_pattern = r'^\d{5}$'
print(re.fullmatch(zip_pattern, "12345"))   # Match
print(re.fullmatch(zip_pattern, "1234"))    # None

# ZIP+4: 5자리, 선택적 하이픈과 4자리
zip4_pattern = r'^\d{5}(-\d{4})?$'
print(re.fullmatch(zip4_pattern, "12345"))       # Match
print(re.fullmatch(zip4_pattern, "12345-6789"))  # Match

# 비밀번호: 8~20자
pwd_pattern = r'^.{8,20}$'
print(re.fullmatch(pwd_pattern, "short"))           # None
print(re.fullmatch(pwd_pattern, "longenough123"))   # Match
```

---

## 3. 탐욕적 vs 게으른 매칭

### 탐욕적 (기본)

기본적으로 수량자는 **탐욕적**입니다 -- 가능한 많은 텍스트를 매칭합니다:

```python
import re

text = "<b>bold</b> and <i>italic</i>"

# 탐욕적: .* 가 가능한 많이 매칭
print(re.search(r'<.*>', text).group())
# '<b>bold</b> and <i>italic</i>'   -- 전부 매칭!
```

```
탐욕적 매칭: <.*>

    입력: <b>bold</b> and <i>italic</i>
          ─────────────────────────────────
          <                               >
          ↑ 매칭 시작          모두 매칭 ↑

    .*가 모든 것을 삼키고, 마지막 >를 찾을 때까지 역추적
```

### 게으른 (비탐욕적)

수량자 뒤에 `?`를 추가하면 **게으른** 매칭이 됩니다 -- 가능한 적은 텍스트를 매칭합니다:

```python
import re

text = "<b>bold</b> and <i>italic</i>"

# 게으른: .*? 가 가능한 적게 매칭
print(re.findall(r'<.*?>', text))
# ['<b>', '</b>', '<i>', '</i>']
```

```
게으른 매칭: <.*?>

    입력: <b>bold</b> and <i>italic</i>
          ───
          < >
          ↑ 첫 번째 >에서 멈춤

    .*?가 가능한 적은 문자를 취하고, 첫 번째 >에서 멈춤
```

### 전체 게으른 수량자 표

```
탐욕적    게으른     의미
──────    ────       ───────
*         *?         0회 이상 (적은 쪽 선호)
+         +?         1회 이상 (적은 쪽 선호)
?         ??         0회 또는 1회 (0회 선호)
{n,m}     {n,m}?     n~m회 (적은 쪽 선호)
{n,}      {n,}?      n회 이상 (적은 쪽 선호)
```

---

## 4. 역추적의 동작 원리

탐욕적 vs 게으른을 이해하려면 역추적을 이해해야 합니다:

### 탐욕적 역추적

```
패턴: ".*" 입력: He said "hello" and "world"

1단계: " 가 첫 번째 "와 매칭
2단계: .* 가 문자열 끝까지 모든 것을 매칭
3단계: 엔진이 닫는 "를 매칭해야 함
       한 문자씩 역추적하여 마지막 "를 찾음

결과: "hello" and "world"
```

### 게으른 역추적

```
패턴: ".*?" 입력: He said "hello" and "world"

1단계: " 가 첫 번째 "와 매칭
2단계: .*? 가 0개 문자로 시작
       닫는 "를 즉시 매칭 시도
3단계: h는 "가 아님 -- .*?를 한 문자 확장
4단계: 계속 확장하다가 "를 만나면 매칭!

결과: "hello"
```

---

## 5. 문자 클래스와 수량자

수량자는 바로 앞의 요소에 적용됩니다:

```python
import re

# \d+ : 1개 이상의 숫자
print(re.findall(r'\d+', "Price: $12.99, Qty: 5"))
# ['12', '99', '5']

# [a-z]+ : 1개 이상의 소문자
print(re.findall(r'[a-z]+', "Hello World"))
# ['ello', 'orld']

# \w{3,} : 3개 이상의 단어 문자
print(re.findall(r'\w{3,}', "I am a developer at Google"))
# ['developer', 'Google']
```

---

## 6. 흔한 실수

### 실수 1: `*`는 빈 문자열과 매칭

```python
import re

# * 는 0회 매칭을 허용 -- 빈 문자열이 생성됨
print(re.findall(r'\d*', "abc"))
# ['', '', '', '']  -- 각 위치에서 빈 문자열과 매칭!

# 대신 +를 사용하여 최소 1회 매칭을 요구
print(re.findall(r'\d+', "abc"))
# []  -- 숫자 없음 (명확한 결과)
```

### 실수 2: 탐욕적 매칭이 너무 많이 매칭

```python
import re

html = '<span class="name">John</span>'

# 잘못: 탐욕적이 여러 태그를 관통
print(re.search(r'<.*>', html).group())
# '<span class="name">John</span>'

# 올바름: 게으른 수량자 사용
print(re.search(r'<.*?>', html).group())
# '<span class="name">'

# 더 좋음: 부정 문자 클래스 사용 (더 효율적)
print(re.search(r'<[^>]+>', html).group())
# '<span class="name">'
```

---

## 7. 수량자 선택 가이드

```
해당 요소가 반드시 등장해야 하는가?
│
├── 아니오 (선택적) ────────────────> ? 사용 (0회 또는 1회)
│   └── 반복될 수 있는가? ──> 예 ──> * 사용 (0회 이상)
│
└── 예 (필수) ──────────────────────> + 사용 (1회 이상)
    └── 정확한 횟수? ──────> 예 ──> {n} 또는 {n,m} 사용

가능한 적게 매칭하고 싶은가?
└── 예 ──> 수량자 뒤에 ? 추가 (게으른)
```

---

## 요약

| 수량자 | 의미 | 탐욕적 예시 | 게으른 예시 |
|--------|------|-------------|-------------|
| `*` | 0회 이상 | `a*`는 ""와 "aaa" 매칭 | `a*?`는 "" 선호 |
| `+` | 1회 이상 | `a+`는 "a"와 "aaa" 매칭 | `a+?`는 "a" 선호 |
| `?` | 0회 또는 1회 | `a?`는 ""와 "a" 매칭 | `a??`는 "" 선호 |
| `{n}` | 정확히 n회 | `a{3}`은 "aaa" 매칭 | 해당 없음 |
| `{n,}` | n회 이상 | `a{2,}`는 "aa", "aaa" 매칭 | `a{2,}?`는 "aa" 선호 |
| `{n,m}` | n~m회 | `a{2,4}`는 "aa"~"aaaa" 매칭 | `a{2,4}?`는 "aa" 선호 |

핵심 정리:
- **탐욕적** (기본): 가능한 많이 매칭, 필요시 역추적
- **게으른** (`?` 접미사): 가능한 적게 매칭, 필요시 확장
- 가능하면 `.*?` 대신 `[^X]*`를 사용 (더 효율적)

---

## 다음 강의

[05_앵커와 경계](./05_Anchors_and_Boundaries.md)에서는 앵커와 단어 경계를 사용한 위치 기반 매칭을 배웁니다.
