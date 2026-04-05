# 배열과 연산

**이전**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md) | **다음**: [연산자와 표현식](./04_Operators_and_Expressions.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 생성 함수 (INDGEN, FINDGEN, DINDGEN)와 제로 채움 함수 (INTARR, FLTARR, DBLARR, BYTARR)로 배열 생성하기
2. MAKE_ARRAY로 유연한 배열 구성하기
3. 첨자 표기법으로 배열 인덱싱 및 슬라이싱하기
4. WHERE 함수로 조건에 맞는 요소 찾기
5. 배열에 대해 요소별 산술 수행하기
6. REFORM으로 배열 형태 변환하고 배열 차원 이해하기
7. N_ELEMENTS와 N_PARAMS 구분하기

---

배열은 IDL의 기본 데이터 구조입니다. 개별 요소를 루프로 돌리는 다른 언어와 달리, IDL은 전체 배열에 대해 한 번에 연산하도록 설계되었습니다. 이 배열 지향 접근법은 더 간결하고 요소별 처리보다 극적으로 빠릅니다.

## 배열 생성

### 생성 함수 (인덱스 배열)

연속적인 인덱스 값으로 채워진 배열을 생성합니다:

```idl
; INDGEN — 정수 인덱스 배열
a = INDGEN(5)
PRINT, a             ;        0       1       2       3       4

; FINDGEN — Float 인덱스 배열
b = FINDGEN(5)

; DINDGEN — Double 인덱스 배열
c = DINDGEN(5)

; 2D 배열 (3열 x 4행)
arr2d = INDGEN(3, 4)
PRINT, arr2d
;        0       1       2
;        3       4       5
;        6       7       8
;        9      10      11
```

### 제로 채움 배열

```idl
; INTARR — 0으로 채워진 정수 배열
ia = INTARR(5)

; FLTARR — 0으로 채워진 Float 배열
fa = FLTARR(3, 4)

; DBLARR — 0으로 채워진 Double 배열
da = DBLARR(100)

; BYTARR — 0으로 채워진 Byte 배열
ba = BYTARR(5)
```

### MAKE_ARRAY

```idl
; 타입 코드를 지정하여 생성
arr = MAKE_ARRAY(5, 3, TYPE=4)     ; 5x3 FLOAT 배열

; 초기값으로 생성
arr = MAKE_ARRAY(10, VALUE=99.0)

; /INDEX 키워드로 생성 (*INDGEN과 유사)
arr = MAKE_ARRAY(5, /FLOAT, /INDEX)
```

### 배열 리터럴

```idl
; 대괄호로 직접 배열 생성
x = [1, 2, 3, 4, 5]
names = ['Alice', 'Bob', 'Charlie']

; 배열 연결
a = [1, 2, 3]
b = [4, 5, 6]
c = [a, b]
PRINT, c             ;        1       2       3       4       5       6
```

---

## 배열 인덱싱

IDL은 0부터 시작하는 인덱싱과 대괄호를 사용합니다:

```idl
arr = [10, 20, 30, 40, 50]

; 단일 요소
PRINT, arr[0]        ;       10
PRINT, arr[-1]       ;       50  (끝에서부터, IDL 8+)

; 범위 (콜론)
PRINT, arr[2:4]      ;       30      40      50

; 끝까지 (* 사용)
PRINT, arr[3:*]      ;       40      50
```

### 다차원 인덱싱

```idl
arr = INDGEN(4, 3)
; 단일 요소: arr[열, 행]
PRINT, arr[1, 2]     ;        9

; 행 슬라이스 (행 1의 모든 열)
PRINT, arr[*, 1]     ;        4       5       6       7

; 열 슬라이스 (열 2의 모든 행)
PRINT, arr[2, *]     ;        2       6      10
```

---

## WHERE 함수

`WHERE`는 IDL에서 가장 중요한 함수 중 하나입니다. 조건을 만족하는 배열 요소의 인덱스를 반환합니다:

```idl
data = [3, 7, 1, 9, 4, 6, 2, 8, 5]

; 5보다 큰 요소 찾기
idx = WHERE(data GT 5, count)
PRINT, 'Count:', count       ;        4
PRINT, 'Values:', data[idx]  ;        7       9       6       8

; 일치하는 항목이 없으면 -1 반환
idx = WHERE(data GT 100, count)
PRINT, 'Count:', count       ;        0
PRINT, 'Index:', idx         ;       -1
```

### WHERE와 COMPLEMENT

```idl
data = FINDGEN(10)
good = WHERE(data GE 5, n_good, COMPLEMENT=bad, NCOMPLEMENT=n_bad)
PRINT, 'Good:', data[good]
PRINT, 'Bad:', data[bad]
```

### 일반적인 WHERE 패턴

```idl
; NaN 값 제거
data = [1.0, !VALUES.F_NAN, 3.0, !VALUES.F_NAN, 5.0]
good = WHERE(FINITE(data), count)
IF count GT 0 THEN clean_data = data[good]

; 값 교체
data = FINDGEN(10)
idx = WHERE(data LT 3)
IF idx[0] NE -1 THEN data[idx] = -999.0
```

---

## 배열 산술

IDL은 배열에 대해 요소별로 산술을 수행합니다:

```idl
a = [1.0, 2.0, 3.0, 4.0, 5.0]
b = [10.0, 20.0, 30.0, 40.0, 50.0]

PRINT, a + b         ;      11.0000      22.0000 ...
PRINT, a * b         ;      10.0000      40.0000 ...
PRINT, a ^ 2         ;      1.00000      4.00000 ...

; 스칼라-배열 연산
PRINT, a * 10        ;      10.0000      20.0000 ...
```

### 배열 통계

```idl
data = RANDOMN(seed, 1000)
PRINT, 'Mean:', MEAN(data)
PRINT, 'Median:', MEDIAN(data)
PRINT, 'Std Dev:', STDDEV(data)
PRINT, 'Min:', MIN(data)
PRINT, 'Max:', MAX(data)
PRINT, 'Total:', TOTAL(data)
```

---

## 배열 조작

### REFORM — 배열 형태 변환

```idl
; 1D 배열을 2D로 변환
arr = INDGEN(12)
arr2d = REFORM(arr, 3, 4)

; 축소된 차원 제거 (squeeze)
x = FLTARR(10, 1, 5)    ; Array[10, 1, 5]
y = REFORM(x)            ; Array[10, 5]
```

### REBIN — 정수 배율로 크기 변경

```idl
small = INDGEN(3, 2)
big = REBIN(small, 6, 4)   ; 각 차원을 2배로

big = FINDGEN(100)
small = REBIN(big, 10)     ; 10개씩 평균
```

### SORT, UNIQ, REVERSE, SHIFT

```idl
data = [3, 1, 4, 1, 5, 9, 2, 6]
idx = SORT(data)
PRINT, data[idx]     ; 정렬된 결과

; 유일한 요소
u = UNIQ(data, SORT(data))
PRINT, data[u]

; 뒤집기
PRINT, REVERSE([1, 2, 3, 4, 5])

; 이동
PRINT, SHIFT([1, 2, 3, 4, 5], 2)
```

---

## N_ELEMENTS vs N_PARAMS

```idl
; N_ELEMENTS — 변수의 요소 수
arr = FINDGEN(100)
PRINT, N_ELEMENTS(arr)       ;      100

; N_PARAMS — 루틴에 전달된 위치 매개변수 수
PRO example_proc, a, b, c
  PRINT, 'Number of parameters:', N_PARAMS()
END
```

---

## 요약

| 개념 | 설명 |
|------|------|
| INDGEN, FINDGEN, DINDGEN | 인덱스 값으로 배열 생성 |
| INTARR, FLTARR, DBLARR | 0으로 채워진 배열 생성 |
| MAKE_ARRAY | 유연한 배열 구성 |
| `arr[i]`, `arr[i:j]` | 인덱싱과 슬라이싱 |
| WHERE | 조건에 맞는 인덱스 찾기 |
| 요소별 연산 | 배열에 대한 `+`, `-`, `*`, `/`, `^` |
| REFORM | 배열 차원 변환 |
| REBIN / CONGRID | 배열 크기 변경 |
| N_ELEMENTS | 변수의 요소 수 세기 |
| N_PARAMS | 루틴의 위치 매개변수 수 세기 |

---

**이전**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md) | **다음**: [연산자와 표현식](./04_Operators_and_Expressions.md)
