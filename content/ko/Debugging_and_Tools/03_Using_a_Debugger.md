# 디버거 사용법

**이전**: [print 디버깅](./02_Print_Debugging.md) | **다음**: [흔한 버그 패턴](./04_Common_Bug_Patterns.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `pdb`, `breakpoint()`, 명령줄 실행으로 Python 디버거 시작하기
2. step, next, continue, return 명령으로 코드 탐색하기
3. 변수 검사, 표현식 평가, 실행 중 상태 수정하기
4. 특정 조건에서만 멈추는 조건부 중단점 설정하기
5. `pdb.post_mortem()`으로 예외 발생 후 디버깅하기
6. `up`, `down`, `where` 명령으로 호출 스택 탐색하기
7. IDE 디버거(VS Code, PyCharm)의 기초 이해하기

---

디버거는 프로그램을 **아무 지점에서나 일시 정지**하고, 모든 변수를 **검사**하며, 코드를 **한 줄씩** 따라가고, 값을 **즉석에서 수정**할 수 있게 해줍니다. print 디버깅이 빠르긴 하지만, 디버거는 프로그램 상태의 완전한 투시도를 제공합니다. `pdb`(Python 내장 디버거) 사용법을 배우는 것은 수많은 좌절의 시간을 절약해줄 핵심 기술입니다.

> **비유:** print 디버깅은 건물에 창문을 달아 안을 들여다보는 것입니다. 디버거는 모든 방을 걸어 다니고, 모든 서랍을 열고, 가구를 재배치할 수 있는 것입니다 -- 시간이 멈춰있는 동안에요.

---

## 1. 디버거 시작하기

### 1.1 `breakpoint()` 사용 (Python 3.7+, 권장)

```python
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    breakpoint()  # 여기서 실행이 일시 정지됨
    return total / len(numbers)

result = calculate_average([10, 20, 30])
```

Python이 `breakpoint()`를 만나면 대화형 `pdb` 프롬프트로 진입합니다:

```
> /path/to/script.py(5)calculate_average()
-> return total / len(numbers)
(Pdb) 
```

### 1.2 `import pdb; pdb.set_trace()` 사용

3.7 이전 버전의 동등한 방법 (여전히 작동):

```python
import pdb; pdb.set_trace()  # breakpoint()와 동일
```

### 1.3 명령줄에서 실행

```bash
# 처음부터 디버거 제어 하에 스크립트 시작
python -m pdb script.py

# 디버거가 첫 번째 줄에서 정지함
> /path/to/script.py(1)<module>()
-> import sys
(Pdb) 
```

### 1.4 breakpoint() 비활성화

```bash
# 모든 중단점 비활성화 (프로덕션에서 유용)
PYTHONBREAKPOINT=0 python script.py

# 다른 디버거 사용
PYTHONBREAKPOINT=ipdb.set_trace python script.py
```

---

## 2. 필수 pdb 명령어

### 명령어 참조

```
┌──────────────────────────────────────────────────────────────┐
│  탐색                                                        │
├──────────────────────────────────────────────────────────────┤
│  n (next)       현재 줄 실행, 함수 호출 건너뛰기              │
│  s (step)       현재 줄 실행, 함수 안으로 들어가기             │
│  c (continue)   다음 중단점까지 계속 실행                     │
│  r (return)     현재 함수가 반환될 때까지 실행                 │
│  unt (until) N  N번 줄까지 실행                              │
├──────────────────────────────────────────────────────────────┤
│  검사                                                        │
├──────────────────────────────────────────────────────────────┤
│  p expr         표현식의 값 출력                              │
│  pp expr        표현식의 값을 보기 좋게 출력                   │
│  l (list)       현재 줄 주변의 소스 코드 표시                  │
│  ll (longlist)  현재 함수의 전체 소스 표시                    │
│  a (args)       현재 함수의 인자 표시                         │
│  w (where)      호출 스택 표시 (트레이스백)                   │
│  whatis expr    표현식의 타입 표시                            │
├──────────────────────────────────────────────────────────────┤
│  중단점                                                      │
├──────────────────────────────────────────────────────────────┤
│  b N            N번 줄에 중단점 설정                          │
│  b func         함수 진입 시 중단점 설정                      │
│  b N, cond      조건부 중단점 (cond=True일 때만 정지)         │
│  cl N           N번 중단점 제거                              │
│  bl             모든 중단점 목록                              │
│  disable N      N번 중단점 비활성화 (유지하되 정지하지 않음)   │
│  enable N       N번 중단점 재활성화                           │
├──────────────────────────────────────────────────────────────┤
│  스택 탐색                                                   │
├──────────────────────────────────────────────────────────────┤
│  u (up)         호출 스택에서 한 프레임 위로                   │
│  d (down)       호출 스택에서 한 프레임 아래로                 │
├──────────────────────────────────────────────────────────────┤
│  제어                                                        │
├──────────────────────────────────────────────────────────────┤
│  q (quit)       디버거와 프로그램 종료                        │
│  restart        프로그램 재시작                               │
│  h (help)       도움말 표시; h <명령>으로 상세 정보            │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 디버깅 실습

### 버그가 있는 예제 코드

```python
# buggy_stats.py
def compute_stats(data):
    """평균과 표준편차를 계산합니다."""
    n = len(data)
    mean = sum(data) / n
    
    variance = sum((x - mean) for x in data) / n  # 버그: ** 2가 빠져있음
    std_dev = variance ** 0.5
    
    return {"mean": mean, "std_dev": std_dev}

result = compute_stats([2, 4, 4, 4, 5, 5, 7, 9])
print(f"Mean: {result['mean']}, Std Dev: {result['std_dev']}")
```

### 단계별 디버깅 세션

```
$ python -m pdb buggy_stats.py
> buggy_stats.py(1)<module>()
-> def compute_stats(data):
(Pdb) b 6            # 6번 줄(분산 계산)에 중단점 설정
Breakpoint 1 at buggy_stats.py:6
(Pdb) c               # 중단점까지 계속 실행
> buggy_stats.py(6)compute_stats()
-> variance = sum((x - mean) for x in data) / n
(Pdb) p mean           # 평균값 확인
5.0
(Pdb) p data           # 입력 데이터 확인
[2, 4, 4, 4, 5, 5, 7, 9]
(Pdb) p [(x - mean) for x in data]    # 각 항 확인
[-3.0, -1.0, -1.0, -1.0, 0.0, 0.0, 2.0, 4.0]
(Pdb) p sum([(x - mean) for x in data])
0.0                     # 편차의 합은 항상 0!
(Pdb) # 발견! (x - mean)**2가 필요한데 (x - mean)만 있음
(Pdb) p sum([(x - mean)**2 for x in data]) / n
4.0                     # 올바른 분산
(Pdb) q
```

---

## 4. 조건부 중단점

특정 조건이 참일 때만 정지:

```python
def process_records(records):
    for i, record in enumerate(records):
        breakpoint()  # 이러면 1000번 정지!
        result = transform(record)
```

대신 조건부 중단점 사용:

```
(Pdb) b 3, i == 500           # 500번째 반복에서만 정지
(Pdb) b 3, record["status"] == "error"  # 에러 레코드에서만 정지
(Pdb) b 3, len(record) > 10   # 크기가 큰 레코드에서만 정지
```

또는 코드에서:

```python
def process_records(records):
    for i, record in enumerate(records):
        if record.get("status") == "error":
            breakpoint()  # 에러 레코드에서만 정지
        result = transform(record)
```

---

## 5. 사후 디버깅 (Post-Mortem)

예외가 발생한 후 `pdb.post_mortem()`으로 크래시 시점의 상태를 검사할 수 있습니다:

```python
import pdb

try:
    result = buggy_function()
except Exception:
    pdb.post_mortem()  # 크래시 지점에서 디버거 열기
```

명령줄에서:

```bash
# 처리되지 않은 예외 발생 시 자동으로 디버거 진입
python -m pdb script.py
# 크래시가 발생하면 pdb가 크래시 지점에서 (Pdb) 프롬프트를 제공
```

### 대화형 Python에서 `pdb.pm()` 사용

```python
>>> import my_module
>>> my_module.buggy_function()
Traceback (most recent call last):
  ...
ValueError: invalid value
>>> import pdb; pdb.pm()   # 마지막 예외를 디버깅
> my_module.py(10)buggy_function()
-> result = int(value)
(Pdb) p value
'not_a_number'
```

---

## 6. 스택 탐색

깊이 중첩된 호출에서 멈췄을 때, `up`과 `down`으로 호출 스택을 이동합니다:

```python
def level_3(x):
    breakpoint()
    return x * 2

def level_2(x):
    return level_3(x + 10)

def level_1(x):
    return level_2(x + 5)

level_1(1)
```

```
(Pdb) w                     # 전체 호출 스택 표시
  /path/script.py(10)<module>()
-> level_1(1)
  /path/script.py(8)level_1()
-> return level_2(x + 5)
  /path/script.py(5)level_2()
-> return level_3(x + 10)
> /path/script.py(2)level_3()    ← 현재 프레임
-> return x * 2
(Pdb) p x                   # level_3에서의 x
16
(Pdb) u                     # level_2로 올라가기
> /path/script.py(5)level_2()
(Pdb) p x                   # level_2에서의 x
6
(Pdb) u                     # level_1으로 올라가기
> /path/script.py(8)level_1()
(Pdb) p x                   # level_1에서의 x
1
(Pdb) d                     # level_2로 다시 내려가기
```

---

## 7. 디버깅 중 값 수정

디버그 세션 중에 변수 값을 변경할 수 있습니다:

```python
def divide(a, b):
    breakpoint()
    return a / b

divide(10, 0)
```

```
(Pdb) p b
0
(Pdb) !b = 5         # pdb 명령과 구분하기 위해 ! 접두사 사용
(Pdb) p b
5
(Pdb) c               # 계속 -- 이제 10/0 대신 10/5로 나눔
```

**주의**: pdb 명령과 충돌을 피하기 위해 대입문 앞에 `!`를 사용하세요 (예: `n = 5` 대신 `!n = 5`, `n`은 "next" 명령이므로).

---

## 8. 향상된 디버거

### 8.1 `ipdb` -- IPython 기반 디버거

```bash
pip install ipdb
```

```python
import ipdb; ipdb.set_trace()
# 또는: PYTHONBREAKPOINT=ipdb.set_trace python script.py
```

장점: 탭 완성, 구문 강조, 향상된 `?` 도움말.

### 8.2 `pdb++` (`pdbpp`)

```bash
pip install pdbpp
```

자동으로 `pdb`를 향상된 버전으로 대체:
- 구문 강조
- `sticky` 모드 (항상 주변 코드를 표시)
- 탭 완성

---

## 9. IDE 디버거

### VS Code

1. Python 파일 열기
2. 거터(줄 번호 왼쪽)를 클릭하여 빨간 중단점 설정
3. `F5`를 눌러 디버깅 시작 (또는 `실행 > 디버깅 시작`)
4. 디버그 도구 모음 사용: 계속(`F5`), 건너뛰기(`F10`), 들어가기(`F11`)
5. 왼쪽 "변수" 패널에서 변수 검사
6. "감시" 패널에 표현식 추가
7. "디버그 콘솔"에서 표현식 평가

### PyCharm

1. 거터를 클릭하여 중단점 설정
2. 우클릭하고 "디버그" 선택 (또는 벌레 아이콘 사용)
3. `F8` (건너뛰기), `F7` (들어가기), `F9` (재개) 사용
4. "변수" 패널에 모든 지역 변수가 자동으로 표시됨
5. "표현식 평가" (`Alt+F8`)로 임의의 코드 실행 가능

### 공통 IDE 기능

```
┌──────────────────────────────────────────────┐
│  기능                  단축키 (VS Code)       │
├──────────────────────────────────────────────┤
│  중단점 토글           F9 / 거터 클릭         │
│  디버깅 시작           F5                     │
│  디버깅 중지           Shift+F5               │
│  건너뛰기              F10                    │
│  들어가기              F11                    │
│  나가기               Shift+F11              │
│  계속                  F5                     │
│  재시작               Ctrl+Shift+F5          │
└──────────────────────────────────────────────┘
```

---

## 10. 디버깅 팁과 모범 사례

### 10.1 넓게 시작하여 좁혀가기

1. 의심되는 버그 영역 앞에 중단점 설정
2. `n`으로 건너뛰며 잘못된 값을 발견할 때까지 진행
3. 다음 실행에서 정확한 문제 줄에 중단점 설정
4. `s`로 함수 호출 안으로 들어가기

### 10.2 `commands`로 자동 동작 설정

```
(Pdb) b 15                    # 15번 줄에 중단점
(Pdb) commands 1              # 1번 중단점이 발동하면...
(com) p x, y, total           # ...이 값들을 출력하고
(com) c                       # ...계속 실행
(com) end                     # 명령 끝
```

이제 15번 줄에 도달할 때마다 자동으로 값을 출력하고 계속합니다 -- 코드를 수정하지 않고 print 문을 추가한 것과 같은 효과입니다.

### 10.3 재귀 함수 디버깅

```python
def factorial(n):
    breakpoint()
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

조건부 중단점 사용:
```
(Pdb) b 2, n == 1             # 기저 조건에서만 정지
```

또는 `r` (return)로 재귀 호출을 빠르게 건너뛰세요.

---

## 요약

- `breakpoint()`는 디버거를 호출하는 현대적 방법 (Python 3.7+)
- `n`은 건너뛰기, `s`는 들어가기, `c`는 계속, `r`은 현재 함수에서 반환
- `p`와 `pp`로 값을 검사; `!` 접두사로 변수 수정
- 조건부 중단점으로 매 반복마다 멈추는 것을 방지
- `pdb.post_mortem()`으로 크래시 후 디버깅 가능
- `up`/`down`으로 호출 스택 탐색
- IDE 디버거는 같은 개념에 대한 시각적 인터페이스 제공
- 기본을 이해하려면 `pdb`부터 시작하고, 편의를 위해 IDE 디버거 사용

---

## 연습문제

1. `breakpoint()`를 사용하여 함수 안에서 일시 정지하고 변수 검사하기
2. `n`으로 루프를 따라가며 값의 변화 추적하기
3. 특정 조건에서만 멈추는 조건부 중단점 설정하기
4. 사후 디버깅을 사용하여 크래시 조사하기
5. `up`과 `down`으로 호출 스택 탐색하기

**이전**: [print 디버깅](./02_Print_Debugging.md) | **다음**: [흔한 버그 패턴](./04_Common_Bug_Patterns.md)
