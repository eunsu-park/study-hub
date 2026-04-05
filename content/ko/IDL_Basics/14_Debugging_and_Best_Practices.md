# 디버깅과 모범 사례

**이전**: [날짜와 시간](./13_Date_and_Time.md) | **다음**: [프로젝트: 태양 광도 곡선](./15_Project_Solar_Light_Curve.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. STOP으로 중단점 설정하고 .CONTINUE로 재개하기
2. 디버깅 중 HELP와 PRINT로 변수 검사하기
3. .COMPILE, .RUN, RETALL로 컴파일 관리하기
4. RESOLVE_ALL로 종속성 확인하기
5. HEAP_GC와 포인터 정리로 메모리 관리하기
6. 읽기 쉽고 유지보수 가능한 IDL 코딩 규칙 따르기
7. 루프 대신 벡터화 연산으로 효율적인 코드 작성하기

---

## 디버깅 도구

### STOP — 중단점

```idl
PRO process_data, filename
  data = READFITS(filename, header)
  STOP    ; 실행이 여기서 멈춤
  ; 프롬프트에서 변수를 검사한 후 .CONTINUE로 재개
  processed = data / SXPAR(header, 'EXPTIME')
END
```

STOP 지점에서 모든 변수를 검사할 수 있습니다:

```idl
IDL> HELP, data
IDL> PRINT, MIN(data), MAX(data)
IDL> .CONTINUE    ; 실행 재개
```

### .CONTINUE, .STEP

```idl
IDL> .CONTINUE       ; 실행 재개
IDL> .STEP           ; 한 줄 실행 후 정지
IDL> .OUT            ; 현재 루틴이 반환될 때까지 계속
```

### HELP — 변수 검사

```idl
HELP, my_array       ; 타입과 차원 표시
HELP                 ; 현재 스코프의 모든 변수
HELP, /STRUCTURES    ; 구조체만 표시
HELP, /MEMORY        ; 메모리 사용량
```

## 오류 처리

### CATCH

```idl
PRO safe_process, filename
  CATCH, error_status
  IF error_status NE 0 THEN BEGIN
    PRINT, 'Error: ' + !ERROR_STATE.MSG
    CATCH, /CANCEL
    RETURN
  ENDIF
  data = READFITS(filename, header)
  CATCH, /CANCEL
END
```

### MESSAGE

```idl
PRO validate_input, data
  IF N_ELEMENTS(data) EQ 0 THEN $
    MESSAGE, 'Input is undefined'
  IF SIZE(data, /N_DIMENSIONS) NE 2 THEN $
    MESSAGE, 'Input must be 2D'
  MESSAGE, 'Input validated', /INFORMATIONAL
END
```

## 메모리 관리

```idl
; 가비지 컬렉션
HEAP_GC

; 포인터 해제
ptr = PTR_NEW(FINDGEN(1000000))
PTR_FREE, ptr

; TEMPORARY로 복사 방지
result = TEMPORARY(big_array) * 2.0
; big_array는 이제 정의되지 않음
```

## 효율성 팁: 벡터화!

IDL에서 가장 중요한 성능 팁: **배열 연산이 가능할 때 루프를 피하세요**.

```idl
; 나쁜 예: 요소별 루프
FOR i = 0L, n - 1 DO BEGIN
  IF data[i] GT 0 THEN result[i] = SQRT(data[i])
ENDFOR

; 좋은 예: 벡터화 연산
positive = WHERE(data GT 0, count)
IF count GT 0 THEN result[positive] = SQRT(data[positive])
; 일반적으로 10-100배 빠름
```

### 더 많은 벡터화 예제

```idl
; 나쁜 예: 루프로 임계값 처리
FOR i = 0L, N_ELEMENTS(data) - 1 DO $
  IF data[i] LT 0 THEN data[i] = 0

; 좋은 예: 벡터화
data = data > 0    ; 최소 연산자로 0에서 클리핑

; 나쁜 예: 루프로 합계
total_val = 0.0
FOR i = 0L, N_ELEMENTS(data) - 1 DO total_val += data[i]

; 좋은 예: 내장 함수
total_val = TOTAL(data)
```

## 일반적인 함정

```idl
; 정수 나눗셈
PRINT, 1/3              ;        0  (잘못됨)
PRINT, 1.0/3.0          ;     0.333333  (올바름)

; BEGIN/END 누락
; 잘못됨: 첫 문장만 IF 본문
IF x GT 0 THEN
  PRINT, 'Positive'
  y = SQRT(x)            ; 이것은 항상 실행됨!

; 올바름:
IF x GT 0 THEN BEGIN
  PRINT, 'Positive'
  y = SQRT(x)
ENDIF

; WHERE 결과 확인 안 함
idx = WHERE(data GT threshold, count)
IF count GT 0 THEN good_data = data[idx]    ; 올바름

; LUN 해제 안 함
OPENR, lun, 'data.txt', /GET_LUN
; ... 항상 FREE_LUN 호출해야 함 ...
FREE_LUN, lun
```

## 코딩 규칙

```idl
;+
; NAME:
;   compute_temperature
; PURPOSE:
;   Calculate brightness temperature
; INPUTS:
;   flux       - Spectral flux in W/m^2/Hz
;   wavelength - Wavelength in meters
; OUTPUTS:
;   Returns brightness temperature in Kelvin
;-
FUNCTION compute_temperature, flux, wavelength
  ; 설명적인 변수 이름 사용
  h = 6.626D-34    ; Planck constant
  c = 3.0D8         ; Speed of light
  k = 1.381D-23     ; Boltzmann constant
  RETURN, h * c / (wavelength * k * ALOG(2.0D * h * c^2 / (wavelength^5 * flux) + 1.0D))
END
```

---

## 요약

| 도구 | 설명 |
|------|------|
| `STOP` | 중단점 삽입 |
| `.CONTINUE` | STOP 후 재개 |
| `.STEP` | 한 줄 실행 |
| `HELP, var` | 변수 검사 |
| `RETALL` | 메인 레벨로 복귀 |
| `CATCH` | 구조화된 오류 처리 |
| `MESSAGE` | 오류/정보 메시지 발생 |
| `HEAP_GC` | 힙 변수 가비지 컬렉션 |
| `TEMPORARY()` | 메모리 재사용 |

### 모범 사례 체크리스트

- 루프 대신 벡터화 연산 사용
- 설명적인 변수 및 루틴 이름 사용
- 모든 루틴에 헤더 문서 추가
- WHERE 결과를 사용하기 전에 확인
- 파일을 열면 항상 FREE_LUN 호출
- 정밀도가 중요한 계산에 DOUBLE 사용
- 엣지 케이스로 테스트 (빈 배열, NaN, 0으로 나누기)

---

**이전**: [날짜와 시간](./13_Date_and_Time.md) | **다음**: [프로젝트: 태양 광도 곡선](./15_Project_Solar_Light_Curve.md)
