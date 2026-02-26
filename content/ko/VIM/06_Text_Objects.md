# 텍스트 객체(Text Objects)

**이전**: [연산자와 조합성](./05_Operators_and_Composability.md) | **다음**: [비주얼 모드](./07_Visual_Mode.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. **내부(inner)** (`i`)와 **주변(around)** (`a`) 텍스트 객체의 차이를 구분한다
2. 단어, 문장, 문단 텍스트 객체를 사용한다 (`iw`, `aw`, `is`, `as`, `ip`, `ap`)
3. 따옴표, 괄호, 태그에 대한 구분자 기반 텍스트 객체를 사용한다 (`i"`, `a(`, `it`)
4. 텍스트 객체와 연산자를 조합하여 정밀 편집을 수행한다 (`diw`, `ci"`, `ya}`)
5. 일반적인 편집 시나리오에 적합한 텍스트 객체를 선택한다

---

> **비유 — 정밀 수술 도구**: 모션(motion)이 목적지까지 걸어가는 것이라면, 텍스트 객체는 정확히 원하는 위치로 순간이동하여 딱 필요한 만큼의 텍스트를 선택하는 것입니다. 이것은 정밀 수술 도구와 같습니다. `ci"`는 "이 따옴표 안의 모든 것을 바꿔라"라고 말하며, 따옴표 경계로 이동할 필요가 없습니다. 커서가 따옴표 안의 *어느 곳에나* 있으면 됩니다.

텍스트 객체는 Vim의 가장 강력한 기능 중 하나이며, Vim을 다른 편집기와 진정으로 차별화하는 요소입니다. 텍스트 객체는 구조화된 텍스트 단위(단어, 문장, 문단, 따옴표로 묶인 문자열, 괄호로 묶인 블록 등)를 해당 단위 내에서 커서 위치에 관계없이 조작할 수 있게 해줍니다.

## 목차

1. [텍스트 객체란 무엇인가?](#1-텍스트-객체란-무엇인가)
2. [내부(Inner) vs 주변(Around)](#2-내부inner-vs-주변around)
3. [단어 객체](#3-단어-객체)
4. [문장과 문단 객체](#4-문장과-문단-객체)
5. [구분자 객체 (따옴표와 괄호)](#5-구분자-객체-따옴표와-괄호)
6. [태그 객체](#6-태그-객체)
7. [블록 객체](#7-블록-객체)
8. [실용 예제](#8-실용-예제)
9. [요약](#9-요약)

---

## 1. 텍스트 객체란 무엇인가?

텍스트 객체는 구조를 기반으로 텍스트의 **범위(range)**를 선택하는 특수 모션입니다. 텍스트 객체는 **연산자(operator)** 뒤에서만(또는 비주얼 모드에서만) 동작하며, 노멀 모드(Normal mode)에서 단독으로 사용할 수 없습니다.

구문:

```
operator  +  a/i  +  object-type
  (동사)    (범위)      (명사)
```

예시:
- `diw` → **d**elete **i**nner **w**ord (내부 단어 삭제)
- `ca"` → **c**hange **a**round **"**quotes (따옴표 포함 변경)
- `yi(` → **y**ank **i**nner **(** parentheses (괄호 내부 복사)

---

## 2. 내부(Inner) vs 주변(Around)

모든 텍스트 객체에는 두 가지 변형이 있습니다:

| 접두사 | 이름 | 포함 범위 |
|--------|------|----------|
| `i` | **내부(inner)** | 내용만 (구분자나 주변 공백 제외) |
| `a` | **주변(around)** | 내용 + 구분자 또는 주변 공백 |

### 시각적 비교

```
Text: "Hello, World!"

di"  →  ""                 (내용 삭제, 따옴표 유지)
da"  →                     (내용 + 따옴표 삭제)

ci"  →  "|"                (커서가 따옴표 사이 삽입 모드에 위치)
ca"  →  |                  (모두 삭제, 삽입 모드)
```

```
Text: The quick brown fox

diw  →  The  brown fox     ("quick" 삭제, 공백 유지)
daw  →  The brown fox      ("quick" + 공백 하나 삭제)
```

**경험 법칙**:
- `i` (내부)는 내용을 **교체**하려 할 때 사용 (구분자 유지)
- `a` (주변)는 요소 전체를 **제거**하려 할 때 사용 (구분자/공백 포함)

---

## 3. 단어 객체

| 객체 | 범위 |
|--------|-------|
| `iw` | 내부 단어(word) (단어만) |
| `aw` | 단어(word) + 주변 공백 |
| `iW` | 내부 WORD |
| `aW` | WORD + 주변 공백 |

### 예시

```
Text: The quick brown fox
          ^  'u'에 커서

diw  → The  brown fox          ("quick" 삭제)
daw  → The brown fox           ("quick" + 공백 삭제)
ciw  → The | brown fox         (삽입 모드로 전환, 교체 준비)
yiw  → "quick" 복사           (커서가 단어 어느 곳에나 있어도 됨)
```

핵심 장점: **커서가 단어 내 어느 곳에 있어도 동작합니다**. 모션을 사용한다면 단어 경계로 이동해야 하지만, 텍스트 객체를 사용하면 `diw`가 커서 위치에 관계없이 단어를 삭제합니다.

---

## 4. 문장과 문단 객체

| 객체 | 범위 |
|--------|-------|
| `is` | 내부 문장(sentence) |
| `as` | 문장 + 뒤따르는 공백 |
| `ip` | 내부 문단(paragraph) (빈 줄 사이의 텍스트) |
| `ap` | 문단 + 뒤따르는 빈 줄 |

### 문장 경계

Vim은 `.`, `!`, `?` 뒤에 공백 또는 줄 끝이 오면 문장이 끝난다고 간주합니다.

```
Text: This is sentence one. This is sentence two. And three.
                             ^ 커서 위치

dis → This is sentence one.  And three.
das → This is sentence one. And three.
```

### 문단 객체

문단은 빈 줄로 구분됩니다. 코드에서 매우 유용합니다:

```python
def function_one():
    pass

def function_two():    ← 커서가 어느 위치에나
    pass

def function_three():
    pass
```

`function_two`에서 `dip`를 실행하면 함수 전체(빈 줄 사이)가 삭제됩니다. `dap`는 뒤따르는 빈 줄도 함께 제거합니다.

---

## 5. 구분자 객체 (따옴표와 괄호)

| 객체 | 범위 |
|--------|-------|
| `i"` / `a"` | 큰따옴표 문자열 |
| `i'` / `a'` | 작은따옴표 문자열 |
| `` i` `` / `` a` `` | 백틱(backtick) 문자열 |
| `i(` 또는 `i)` / `a(` 또는 `a)` | 소괄호 |
| `i[` 또는 `i]` / `a[` 또는 `a]` | 대괄호 |
| `i{` 또는 `i}` / `a{` 또는 `a}` | 중괄호 |
| `i<` 또는 `i>` / `a<` 또는 `a>` | 꺾쇠 괄호 |

### 따옴표 예시

```python
message = "Hello, World!"
               ^ 커서 위치

di"  → message = ""
da"  → message =
ci"  → message = "|"         (따옴표 사이 삽입 모드)
yi"  → "Hello, World!" 내용 복사
```

### 괄호 예시

```python
result = calculate(x, y, z)
                    ^ 커서 위치

di(  → result = calculate()
da(  → result = calculate
ci(  → result = calculate(|)     (괄호 안 삽입 모드)
```

```javascript
const config = {
    host: "localhost",    ← 중괄호 내부 어느 위치에나 커서
    port: 8080,
};

di{  →  const config = {};
da{  →  const config = ;
```

### 중첩(Nesting)

텍스트 객체는 중첩 구조를 올바르게 처리합니다:

```python
outer(inner(deep), value)
             ^ 커서 위치

di(  → outer(inner(), value)     (내부 괄호만)
```

바깥쪽 괄호 쌍을 대상으로 하려면 바깥쪽 레벨로 커서를 이동하거나 카운트를 사용해야 합니다.

---

## 6. 태그 객체

HTML/XML/JSX용:

| 객체 | 범위 |
|--------|-------|
| `it` | 태그 내부 내용 |
| `at` | 태그 전체 (여는 태그와 닫는 태그 포함) |

```html
<div class="container">Hello, World!</div>
                        ^ 커서 위치

dit  → <div class="container"></div>
dat  → (전체 요소 삭제)
cit  → <div class="container">|</div>    (삽입 모드)
```

중첩 태그에서도 동작합니다:

```html
<ul>
  <li>Item one</li>      ← "one"에 커서
  <li>Item two</li>
</ul>

dit  → <li></li>          (<li>의 내부 태그 내용)
```

---

## 7. 블록 객체

프로그래밍 언어 블록용 (Vim에서는 "Block"이라 부름):

| 객체 | 범위 | 사용 사례 |
|--------|-------|----------|
| `iB` 또는 `i{` | `{...}` 블록 내부 | 함수/클래스 본문 |
| `aB` 또는 `a{` | `{...}` 블록 전체 | 중괄호 포함 전체 블록 |
| `ib` 또는 `i(` | `(...)` 블록 내부 | 함수 인자 |
| `ab` 또는 `a(` | `(...)` 블록 전체 | 괄호 포함 |

### 코드 예시

```javascript
function greet(name, greeting) {
    if (name) {
        console.log(greeting + ", " + name);    ← 커서 위치
    }
    return true;
}

diB (감싸는 {}의 내부 블록):
function greet(name, greeting) {
    if (name) {
    }
    return true;
}

바깥쪽 함수 레벨에서:
daB:
function greet(name, greeting)
```

---

## 8. 실용 예제

### 함수 인자 변경

```python
def process(old_argument):
                ^ 인자 내 어느 곳에나 커서

ciw  →  def process(|):          (인자 이름 변경)
```

### 문자열 교체

```python
name = "John Doe"
         ^ 문자열 내 어느 곳에나 커서

ci"  →  name = "|"               (새 이름 입력 준비)
```

### 딕셔너리/객체 항목 삭제

```python
config = {
    "host": "localhost",
    "port": 8080,              ← 이 줄에 커서
    "debug": True,
}

dd   →  해당 줄 삭제 (한 줄일 때는 단순하고 효과적)
```

### HTML 내용 변경

```html
<h1>Old Title</h1>
      ^ 커서 위치

cit  →  <h1>|</h1>              (새 제목 입력)
```

### 감싸기 패턴: `ysi"` (surround 플러그인 사용 시)

기본 내장 기능은 아니지만, `vim-surround` 플러그인이 텍스트 객체를 훌륭하게 확장합니다:
- `cs"'` — 감싸는 `"`를 `'`로 변경
- `ds"` — 감싸는 `"` 삭제
- `ysiw"` — 단어 주위에 `"` 추가

플러그인에 대해서는 [레슨 13](./13_Plugins_and_Ecosystem.md)에서 배웁니다.

### 빠른 참조 조합

| 작업 | 명령 |
|------|---------|
| 단어 삭제 (단어 내 어느 위치에서나) | `diw` |
| 따옴표 내 문자열 변경 | `ci"` 또는 `ci'` |
| 함수 인자 복사 | `yi(` |
| 문단 삭제 | `dip` |
| 코드 블록 들여쓰기 | `>i{` |
| HTML 태그 내부 선택 | `vit` |
| 대괄호 포함 삭제 | `da[` |
| 함수 본문 변경 | `ci{` |

---

## 9. 요약

| 객체 유형 | 내부(`i`) | 주변(`a`) |
|-------------|-------------|--------------|
| 단어(Word) | `iw` | `aw` |
| WORD | `iW` | `aW` |
| 문장(Sentence) | `is` | `as` |
| 문단(Paragraph) | `ip` | `ap` |
| `"` 큰따옴표 | `i"` | `a"` |
| `'` 작은따옴표 | `i'` | `a'` |
| `()` 소괄호 | `i(` 또는 `i)` | `a(` 또는 `a)` |
| `[]` 대괄호 | `i[` 또는 `i]` | `a[` 또는 `a]` |
| `{}` 중괄호 | `i{` 또는 `i}` | `a{` 또는 `a}` |
| `<>` 꺾쇠 괄호 | `i<` 또는 `i>` | `a<` 또는 `a>` |
| HTML 태그 | `it` | `at` |

### 선택 가이드

```
내용을 교체(REPLACE)하려면?  → 내부(i) 사용: ci", ciw, ci(
완전히 제거(REMOVE)하려면?   → 주변(a) 사용: da", daw, da(
내용을 복사(COPY)하려면?     → 내부(i) 사용: yi", yiw
선택(SELECT)하려면?          → 비주얼에서 사용: vi", viw, vit
```

### 텍스트 객체가 중요한 이유

1. **위치 독립적** — 커서가 객체 내부 어느 곳에나 있어도 됨
2. **정밀함** — 필요한 것만 정확하게 선택
3. **빠름** — 명령 하나가 이동 + 선택 + 작업을 대체
4. **조합 가능** — 모든 연산자와 함께 동작
5. **가독성** — `ci"`는 "따옴표 안을 변경"으로 읽힘

---

## 연습 문제

### 연습 문제 1: 내부(Inner) vs 주변(Around)

텍스트: `result = calculate(x + y, z * 2)`

커서가 `x + y, z * 2` 내부 어딘가에 있습니다. 각 명령의 결과 텍스트를 설명하세요:

1. `di(`
2. `da(`
3. `ci(`
4. `yi(`

<details>
<summary>정답 보기</summary>

1. `di(` — 괄호 내부의 모든 것을 삭제하여 남는 것: `result = calculate()`
2. `da(` — 괄호와 내용을 삭제하여 남는 것: `result = calculate`
3. `ci(` — 내용을 삭제하고 삽입 모드(Insert mode)로 진입, `(` 와 `)` 사이에 커서: `result = calculate(|)`
4. `yi(` — `x + y, z * 2`를 기본 레지스터에 복사; 텍스트는 변경 없음.

</details>

### 연습 문제 2: 올바른 텍스트 객체 선택

각 편집 작업에 대해 단일 명령(연산자 + 텍스트 객체)을 작성하세요:

1. JSON 문자열 `"active"` 안에 있고 다른 값으로 교체하고 싶습니다.
2. Python 함수 `def process(data, config):` 안에 있고 모든 인자를 복사하고 싶습니다.
3. HTML `<p>Old content</p>` 안에 있고 내용을 지우고 새 텍스트를 입력하려 합니다.
4. Python 딕셔너리 `{"key": "value"}` 안에 있고 중괄호를 포함해 모두 삭제하고 싶습니다.

<details>
<summary>정답 보기</summary>

1. `ci"` — 따옴표 내부 변경: `active`를 삭제하고 `""` 사이 삽입 모드(Insert mode)로 진입.
2. `yi(` — 괄호 내부 복사: 괄호 없이 `data, config`를 복사.
3. `cit` — 태그 내부 변경: `Old content`를 삭제하고 삽입 모드(Insert mode)로 진입, `<p></p>` 태그 유지.
4. `da{` — 중괄호 포함 삭제: `{"key": "value"}`를 완전히 제거.

</details>

### 연습 문제 3: 코드에서의 문단 객체

다음 Python 파일이 있습니다:

```python
def setup():
    initialize_db()
    load_config()

def main():
    setup()
    run_app()

def cleanup():
    close_db()
    save_state()
```

커서가 `run_app()` 줄에 있습니다. 다음 명령을 작성하세요:

1. 주변의 빈 줄을 제외하고 `main` 함수 블록 전체를 삭제합니다.
2. 뒤따르는 빈 줄을 포함하여 `main` 함수 블록 전체를 삭제합니다.

<details>
<summary>정답 보기</summary>

1. `dip` — 내부 문단 삭제: 빈 줄 사이의 줄들(`def main():`, `setup()`, `run_app()`)을 제거하고 주변 빈 줄은 유지합니다.
2. `dap` — 주변 문단 삭제: 함수 블록과 뒤따르는 빈 줄을 모두 제거합니다.

참고: 이 명령이 올바른 문단을 대상으로 하려면 커서가 `main` 함수 블록 내부(빈 줄이 아닌 곳)에 있어야 합니다.

</details>

### 연습 문제 4: 중첩된 텍스트 객체

텍스트: `outer("inner value", more)`

커서가 `value`의 `v` 위에 있습니다. 다음 질문에 답하세요:

1. `di"`는 무엇을 합니까?
2. `da"`는 무엇을 합니까?
3. `da"` 후 텍스트는 어떻게 보입니까?
4. 바깥쪽 괄호 내부의 모든 것(`"inner value"`와 `, more` 포함)을 삭제하려면 어떻게 합니까?

<details>
<summary>정답 보기</summary>

1. `di"` — 가장 가까운 따옴표 내부 내용을 삭제: `inner value`를 제거하여 `outer("", more)`가 됩니다.
2. `da"` — 따옴표와 내용을 삭제: `"inner value"`를 제거하여 `outer(, more)`가 됩니다.
3. `da"` 후: `outer(, more)` — 앞의 쉼표와 공백이 남습니다.
4. 커서를 `outer(...)` 내부이지만 따옴표 바깥으로 이동한 후 `di(` — 바깥쪽 괄호 내부를 모두 삭제: `outer()`가 됩니다.

</details>

### 연습 문제 5: 실제 편집 시나리오

다음 JavaScript 설정을 편집하고 있습니다:

```javascript
const config = {
    apiUrl: "https://old-api.example.com/v1",
    timeout: 5000,
    retries: 3,
};
```

가장 효율적인 방법(최소 키 입력)으로 다음을 수행하세요:

1. URL을 새 주소로 변경합니다.
2. 전체 객체 내용을 새 설정으로 변경합니다.

<details>
<summary>정답 보기</summary>

**작업 1 (URL 변경)**:
- 커서를 `https://old-api.example.com/v1`(문자열 내용) 어딘가에 놓습니다.
- `ci"`를 누릅니다 — URL 내용을 삭제하고 따옴표 사이 삽입 모드(Insert mode)로 진입.
- 새 URL을 입력합니다.
- `Esc`를 누릅니다.

문자열 경계까지 이동할 필요 없이 3가지 동작: `ci"` + 입력 + `Esc`.

**작업 2 (전체 객체 내용 변경)**:
- 커서를 `{...}` 블록 내부 어딘가에 놓습니다.
- `ci{` (또는 `ciB`)를 누릅니다 — `{` 와 `}` 사이의 모든 내용을 삭제하고 삽입 모드(Insert mode)로 진입.
- 새 설정을 입력합니다.
- `Esc`를 누릅니다.

원본 내용이 몇 줄이든 관계없이 3가지 동작입니다.

</details>

---

**이전**: [연산자와 조합성](./05_Operators_and_Composability.md) | **다음**: [비주얼 모드](./07_Visual_Mode.md)
