# 제어 흐름

**이전**: [연산자와 표현식](./04_Operators_and_Expressions.md) | **다음**: [프로시저와 함수](./06_Procedures_and_Functions.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단일 줄 및 블록 형태로 IF/THEN/ELSE 문 작성하기
2. 다양한 증분값으로 FOR 루프 사용하기
3. WHILE 및 REPEAT/UNTIL 루프 구현하기
4. CASE와 SWITCH로 다중 분기 선택하기
5. BREAK와 CONTINUE로 루프 제어하기
6. BEGIN...END 블록 구문 이해하기
7. GOTO 사용 시기 인식하기 (그리고 사용하지 말아야 할 때)

---

제어 흐름문은 IDL이 코드를 실행하는 순서를 결정합니다. 핵심은 다중 문장 블록이 `BEGIN...END` 구분자를 필요로 한다는 것입니다.

## IF / THEN / ELSE

### 단일 줄 IF

```idl
x = 15
IF x GT 10 THEN PRINT, 'x is greater than 10'
IF x MOD 2 EQ 0 THEN PRINT, 'Even' ELSE PRINT, 'Odd'
```

### 블록 IF (BEGIN...END)

```idl
IF x GT 10 THEN BEGIN
  PRINT, 'x is greater than 10'
  PRINT, 'x = ', x
ENDIF

IF x MOD 2 EQ 0 THEN BEGIN
  PRINT, 'x is even'
ENDIF ELSE BEGIN
  PRINT, 'x is odd'
ENDELSE

; IF/ELSE IF/ELSE 체인
score = 85
IF score GE 90 THEN BEGIN
  grade = 'A'
ENDIF ELSE IF score GE 80 THEN BEGIN
  grade = 'B'
ENDIF ELSE BEGIN
  grade = 'F'
ENDELSE
```

### 블록 구분자

| 구조 | 시작 | 끝 |
|------|------|-----|
| IF...THEN | `BEGIN` | `ENDIF` |
| ELSE | `BEGIN` | `ENDELSE` |
| FOR | `BEGIN` | `ENDFOR` |
| WHILE | `BEGIN` | `ENDWHILE` |
| REPEAT | `BEGIN` | `ENDREP` |
| CASE/SWITCH 절 | `BEGIN` | `END` |

## FOR 루프

```idl
FOR i = 0, 9 DO PRINT, i

FOR i = 0, 9 DO BEGIN
  PRINT, 'Iteration:', i, '  Square:', i^2
ENDFOR

; 증분값 지정
FOR i = 0, 10, 2 DO PRINT, i    ; 0, 2, 4, 6, 8, 10

; 역순
FOR i = 10, 0, -1 DO PRINT, i
```

## WHILE 루프

```idl
count = 0
WHILE count LT 5 DO BEGIN
  PRINT, 'Count:', count
  count = count + 1
ENDWHILE
```

## REPEAT / UNTIL

REPEAT는 본문을 최소 한 번 실행한 후 조건을 확인합니다:

```idl
count = 0
REPEAT BEGIN
  PRINT, 'Count:', count
  count = count + 1
ENDREP UNTIL count GE 5
```

## CASE 문

CASE는 첫 번째 일치하는 절을 실행하고 종료하는 다중 분기입니다:

```idl
day = 3
CASE day OF
  1: PRINT, 'Monday'
  2: PRINT, 'Tuesday'
  3: PRINT, 'Wednesday'
  ELSE: PRINT, 'Other day'
ENDCASE
```

## SWITCH 문

SWITCH는 CASE와 유사하지만 다음 절로 계속 진행합니다 (폴스루):

```idl
; BREAK로 폴스루 방지
level = 2
SWITCH level OF
  1: BEGIN & PRINT, 'Level 1' & BREAK & END
  2: BEGIN & PRINT, 'Level 2' & BREAK & END
  3: BEGIN & PRINT, 'Level 3' & BREAK & END
ENDSWITCH
```

**권장**: 대부분의 경우 SWITCH 대신 CASE를 사용하세요.

## BREAK와 CONTINUE

```idl
; BREAK — 루프를 조기 종료
FOR i = 0, 100 DO BEGIN
  IF i^2 GT 50 THEN BREAK
  PRINT, i, i^2
ENDFOR

; CONTINUE — 현재 반복의 나머지를 건너뜀
FOR i = 0, 9 DO BEGIN
  IF i MOD 2 EQ 0 THEN CONTINUE
  PRINT, 'Odd:', i
ENDFOR
```

## 줄 계속

IDL은 줄 끝에 `$`를 사용하여 다음 줄에 이어서 작성합니다:

```idl
result = SIN(x) * COS(y) + $
         TAN(z) * ALOG(w)

PLOT, time, flux, $
  TITLE='Solar X-ray Flux', $
  XTITLE='Time (hours)', $
  YTITLE='Flux (W/m^2)'
```

---

## 요약

| 구조 | 구문 | 참고 |
|------|------|------|
| IF/THEN | `IF cond THEN stmt` | 단일 줄 |
| IF/THEN 블록 | `IF cond THEN BEGIN...ENDIF` | 다중 줄 |
| FOR | `FOR var=start, stop [, step] DO BEGIN...ENDFOR` | 카운트 루프 |
| WHILE | `WHILE cond DO BEGIN...ENDWHILE` | 사전 테스트 루프 |
| REPEAT/UNTIL | `REPEAT BEGIN...ENDREP UNTIL cond` | 사후 테스트 루프 |
| CASE | `CASE expr OF val: stmt ... ENDCASE` | 폴스루 없음 |
| SWITCH | `SWITCH expr OF val: stmt ... ENDSWITCH` | 폴스루 |
| BREAK | `BREAK` | 루프/case 종료 |
| CONTINUE | `CONTINUE` | 다음 반복으로 건너뜀 |
| 줄 계속 | `$` | 다음 줄에 이어서 |

---

**이전**: [연산자와 표현식](./04_Operators_and_Expressions.md) | **다음**: [프로시저와 함수](./06_Procedures_and_Functions.md)
