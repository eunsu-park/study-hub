# 문자열 처리

**이전**: [프로시저와 함수](./06_Procedures_and_Functions.md) | **다음**: [파일 I/O](./08_File_IO.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. STRMID, STRPOS, STRTRIM, STRLEN으로 문자열 조작하기
2. STRSPLIT과 STRJOIN으로 문자열 분할 및 결합하기
3. STRUPCASE와 STRLOWCASE로 대소문자 변경하기
4. STRING 함수와 FORMAT 키워드로 출력 포맷팅하기
5. printf 스타일 포맷 코드 사용하기
6. READS로 문자열 파싱하기
7. STREGEX로 정규 표현식 적용하기

---

문자열 처리는 데이터 파일 파싱, 출력 포맷팅, 파일 이름 구성, FITS 헤더 작업에 필수적입니다.

## 기본 문자열 연산

```idl
s = 'Hello, World!'
PRINT, STRLEN(s)         ;       13

; 공백 제거
s = '   Hello   '
PRINT, STRTRIM(s, 2)     ; Hello  (양쪽)

; 부분 문자열 추출
PRINT, STRMID(s, 3, 5)   ; Hello

; 부분 문자열 위치 찾기
PRINT, STRPOS(s, 'Hello')

; 대소문자 변환
PRINT, STRUPCASE('hello')  ; HELLO
PRINT, STRLOWCASE('HELLO') ; hello
```

## 문자열 분할과 결합

```idl
; 분할
words = STRSPLIT('Hello World IDL', ' ', /EXTRACT)
csv = STRSPLIT('a,b,c,d', ',', /EXTRACT)

; 결합
PRINT, STRJOIN(['Hello', 'World'], ' ')    ; Hello World
PRINT, STRJOIN(['1', '2', '3'], ',')       ; 1,2,3
```

## FORMAT을 사용한 문자열 포맷팅

```idl
; Fortran 스타일 포맷 코드
PRINT, STRING(42, FORMAT='(I6)')          ;    42
PRINT, STRING(3.14, FORMAT='(F8.4)')      ;  3.1400
PRINT, STRING(1.23e10, FORMAT='(E12.4)')  ;  1.2300E+10
PRINT, FORMAT='("Name: ", A-15, " Age: ", I3)', 'Alice', 30
```

| 코드 | 설명 | 예제 |
|------|------|------|
| `I` | 정수 | `FORMAT='(I5)'` |
| `F` | 고정소수점 실수 | `FORMAT='(F8.3)'` |
| `E` | 과학 표기법 | `FORMAT='(E12.4)'` |
| `A` | 문자열 | `FORMAT='(A10)'` |

## READS — 문자열 파싱

```idl
line = '42 3.14 Hello'
a = 0 & b = 0.0 & c = ''
READS, line, a, b, c
PRINT, 'a =', a, ' b =', b, ' c = ', c
```

## STREGEX — 정규 표현식

```idl
s = 'The temperature is 293.15 K'
match = STREGEX(s, '[0-9]+\.?[0-9]*', /EXTRACT)
PRINT, match                                ; 293.15

; 캡처 그룹
s = 'Date: 2024-07-15'
result = STREGEX(s, '([0-9]{4})-([0-9]{2})-([0-9]{2})', /SUBEXPR, /EXTRACT)
PRINT, 'Year:', result[1]    ; 2024
PRINT, 'Month:', result[2]   ; 07
PRINT, 'Day:', result[3]     ; 15
```

---

## 요약

| 함수 | 설명 |
|------|------|
| `STRLEN(s)` | 문자열 길이 |
| `STRMID(s, pos, len)` | 부분 문자열 추출 |
| `STRPOS(s, search)` | 부분 문자열 위치 찾기 |
| `STRTRIM(s, flag)` | 공백 제거 (0=오른쪽, 1=왼쪽, 2=양쪽) |
| `STRSPLIT(s, delim, /EXTRACT)` | 문자열을 배열로 분할 |
| `STRJOIN(arr, sep)` | 배열을 문자열로 결합 |
| `STRUPCASE(s)` | 대문자로 변환 |
| `STRLOWCASE(s)` | 소문자로 변환 |
| `STRING(val, FORMAT=fmt)` | 값을 문자열로 포맷 |
| `READS, s, vars` | 문자열을 변수로 파싱 |
| `STREGEX(s, pattern)` | 정규 표현식 매칭 |

---

**이전**: [프로시저와 함수](./06_Procedures_and_Functions.md) | **다음**: [파일 I/O](./08_File_IO.md)
