# 변수와 데이터 타입

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [배열과 연산](./03_Arrays_and_Operations.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 모든 IDL 숫자 타입의 변수를 생성하고 할당하기
2. IDL 데이터 타입의 전체 범위 이해하기 (BYTE부터 COMPLEX까지)
3. 타입 변환 함수 사용하기 (FIX, FLOAT, DOUBLE, STRING, BYTE)
4. HELP, SIZE, N_ELEMENTS로 변수 검사하기
5. ISA와 TYPENAME으로 변수 타입 확인하기
6. 특수 값 (NaN, Infinity, NULL) 다루기
7. IDL의 타입 승격 규칙 이해하기

---

IDL은 동적 타입 언어입니다. 변수 타입을 명시적으로 선언하지 않으며, 할당하는 값에 의해 타입이 결정됩니다. 그러나 수치 정밀도, 메모리 사용량, 성능 모두 선택한 타입에 따라 달라지므로 데이터 타입을 이해하는 것이 중요합니다.

## 변수 할당

IDL에서 변수는 `=` 연산자로 간단한 할당을 통해 생성됩니다:

```idl
; 스칼라 변수
x = 42           ; 정수 (INT, 16비트 부호 있는)
y = 3.14         ; 부동소수점 (FLOAT, 32비트)
name = 'Alice'   ; 문자열

; IDL 변수 이름은 대소문자를 구분하지 않습니다
MyVar = 100
PRINT, myvar     ;       100
PRINT, MYVAR     ;       100
```

---

## IDL 데이터 타입

### 숫자 타입

| 타입 | IDL 이름 | 바이트 | 범위 | 생성 구문 |
|------|----------|--------|------|-----------|
| 바이트 | BYTE | 1 | 0~255 | `x = 0B` 또는 `x = BYTE(42)` |
| 정수 | INT | 2 | -32,768~32,767 | `x = 0` 또는 `x = 0S` |
| 부호 없는 정수 | UINT | 2 | 0~65,535 | `x = 0U` |
| 롱 | LONG | 4 | -2^31~2^31-1 | `x = 0L` |
| 부호 없는 롱 | ULONG | 4 | 0~2^32-1 | `x = 0UL` |
| 64비트 롱 | LONG64 | 8 | -2^63~2^63-1 | `x = 0LL` |
| Float | FLOAT | 4 | ~1.2e-38~3.4e38 | `x = 0.0` |
| Double | DOUBLE | 8 | ~2.2e-308~1.8e308 | `x = 0.0D0` |
| 복소수 | COMPLEX | 8 | 두 개의 float | `x = COMPLEX(1.0, 2.0)` |
| 더블 복소수 | DCOMPLEX | 16 | 두 개의 double | `x = DCOMPLEX(1.0D, 2.0D)` |

### 비숫자 타입

| 타입 | 설명 | 예제 |
|------|------|------|
| STRING | 문자열 | `s = 'Hello'` |
| POINTER | 힙 변수에 대한 포인터 | `p = PTR_NEW(42)` |
| OBJECT | 객체 참조 | `obj = OBJ_NEW('classname')` |
| LIST | 순서가 있는 컬렉션 (IDL 8+) | `lst = LIST(1, 'a', 3.14)` |
| HASH | 키-값 쌍 (IDL 8+) | `h = HASH('key', 'value')` |

---

## 특정 타입의 변수 생성

### Byte (BYTE)

바이트는 부호 없는 8비트 정수 (0-255)로, 이미지 데이터에 일반적으로 사용됩니다:

```idl
b1 = 0B
b2 = 255B
b3 = BYTE(42)
b4 = BYTE('A')     ; 'A'의 ASCII 값 = 65

; 문자열에서 바이트 배열
bytes = BYTE('Hello')
PRINT, bytes        ;   72  101  108  108  111
```

### Float (FLOAT)와 Double (DOUBLE)

```idl
; Float (32비트, ~7자리 정밀도)
f = 3.14
f2 = 1.0E6

; Double (64비트, ~15자리 정밀도)
d = 3.14159265358979D0
d2 = 1.0D6

; 과학 계산에서 정밀도가 중요합니다
PRINT, 1.0 / 3.0           ;     0.333333  (7자리)
PRINT, 1.0D0 / 3.0D0       ;      0.33333333333333  (15자리)
```

### 복소수

```idl
z1 = COMPLEX(3.0, 4.0)     ; 3 + 4i
PRINT, REAL_PART(z1)        ;     3.00000
PRINT, IMAGINARY(z1)        ;     4.00000
PRINT, ABS(z1)              ;     5.00000   (sqrt(3^2 + 4^2))
```

### 문자열

```idl
s1 = 'Hello, World!'
s2 = "Double quotes also work"
PRINT, STRLEN(s1)           ;       13

; 문자열 연결
greeting = 'Hello' + ', ' + 'World!'
```

---

## 타입 변환 함수

```idl
; BYTE() — 바이트로 변환
PRINT, BYTE(65)              ;   65

; FIX() — 정수 (INT)로 변환
PRINT, FIX(3.7)              ;       3  (버림, 반올림 아님)

; FLOAT() — Float로 변환
PRINT, FLOAT(42)             ;      42.0000

; DOUBLE() — Double로 변환
PRINT, DOUBLE(42)            ;       42.000000

; STRING() — 문자열로 변환
PRINT, STRING(42)            ;       42
PRINT, STRING(3.14, FORMAT='(F6.3)')  ;  3.140
```

---

## 변수 검사

### HELP

```idl
a = 42
b = 3.14D0
c = 'Hello'
d = FINDGEN(10)

HELP, a, b, c, d
; A               INT       =       42
; B               DOUBLE    =        3.1400000000000
; C               STRING    = 'Hello'
; D               FLOAT     = Array[10]
```

### SIZE

```idl
x = FLTARR(100, 200)
PRINT, SIZE(x, /N_DIMENSIONS)   ;        2
PRINT, SIZE(x, /DIMENSIONS)     ;      100     200
PRINT, SIZE(x, /TYPE)           ;        4  (FLOAT)
PRINT, SIZE(x, /TNAME)          ; FLOAT
PRINT, SIZE(x, /N_ELEMENTS)     ;    20000
```

### N_ELEMENTS

```idl
arr = FINDGEN(5, 3)
PRINT, N_ELEMENTS(arr)      ;       15

; 정의되지 않은 변수의 N_ELEMENTS는 0을 반환 — 존재 확인에 유용
PRINT, N_ELEMENTS(undefined_var)  ;        0
```

### ISA와 TYPENAME (IDL 8+)

```idl
x = 3.14D0
PRINT, ISA(x, 'DOUBLE')         ;    1  (참)
PRINT, ISA(x, /NUMBER)          ;    1  (참)
PRINT, TYPENAME(x)              ; DOUBLE
```

---

## 타입 승격 규칙

타입을 혼합하여 표현식을 사용하면 IDL은 결과를 더 정밀한 타입으로 승격합니다:

```idl
; INT + FLOAT -> FLOAT
result = 5 + 3.0
HELP, result
; RESULT          FLOAT     =       8.00000

; FLOAT + DOUBLE -> DOUBLE
result = 3.14 + 1.0D0
HELP, result
; RESULT          DOUBLE    =        4.1400000000000
```

### 정밀도 함정

```idl
; 일반적인 버그 원인:
PRINT, 1/3           ;       0  (정수 나눗셈!)
PRINT, 1.0/3.0       ;     0.333333  (부동소수점 나눗셈)
PRINT, 1.0D0/3.0D0   ;      0.33333333333333  (더블 나눗셈)
```

---

## 특수 값

### NaN (Not a Number)과 Infinity

```idl
nan_float = !VALUES.F_NAN
inf_float = !VALUES.F_INFINITY

; NaN 테스트
PRINT, FINITE(nan_float)             ;        0  (거짓)
PRINT, FINITE(42.0)                  ;        1  (참)
PRINT, FINITE(nan_float, /NAN)       ;        1  (참 - NaN임)

; 배열에서 NaN 값 교체
data = [1.0, !VALUES.F_NAN, 3.0, !VALUES.F_NAN, 5.0]
good = WHERE(FINITE(data), count)
IF count GT 0 THEN PRINT, 'Good values:', data[good]
```

### NULL (IDL 8+)

```idl
x = !NULL
; 루프 전 변수 초기화에 유용
result = !NULL
FOR i = 0, 4 DO result = [result, i^2]
PRINT, result        ;        0       1       4       9      16
```

---

## 요약

| 개념 | 설명 |
|------|------|
| 동적 타이핑 | 변수는 할당된 값의 타입을 가짐 |
| 타입 접미사 | `B` (byte), `S` (int), `L` (long), `LL` (long64), `U` (unsigned), `D` (double) |
| HELP | 변수 이름, 타입, 값/차원 표시 |
| SIZE | 차원 및 타입 정보 반환 |
| N_ELEMENTS | 전체 요소 수 반환 (정의되지 않으면 0) |
| ISA | /NUMBER, /ARRAY 등의 키워드로 타입 확인 |
| 타입 변환 | FIX, FLOAT, DOUBLE, STRING, BYTE, LONG 등 |
| 타입 승격 | 혼합 타입 표현식은 더 정밀한 타입으로 승격 |
| 특수 값 | `!VALUES.F_NAN`, `!VALUES.F_INFINITY`, `!NULL` |
| FINITE | NaN 및 Infinity 테스트 |

---

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [배열과 연산](./03_Arrays_and_Operations.md)
