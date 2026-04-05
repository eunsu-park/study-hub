# 파일 I/O

**이전**: [문자열 처리](./07_String_Processing.md) | **다음**: [구조체](./09_Structures.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. OPENR, OPENW, OPENU로 파일 열고 GET_LUN/FREE_LUN으로 논리 유닛 관리하기
2. READF와 PRINTF로 텍스트 파일 읽기/쓰기
3. READU와 WRITEU로 바이너리 파일 읽기/쓰기
4. POINT_LUN으로 파일 내 탐색하고 EOF로 파일 끝 감지하기
5. SAVE/RESTORE로 IDL 변수 저장하고 복원하기
6. READ_ASCII와 READ_CSV로 구조화된 텍스트 파일 읽기
7. FILE_SEARCH로 파일 검색하고 FILE_TEST로 파일 속성 테스트하기

---

## 파일 열기와 닫기

```idl
; OPENR — 읽기용 열기, OPENW — 쓰기용 열기, OPENU — 업데이트용 열기
; /GET_LUN을 사용하여 IDL이 논리 유닛 번호를 할당하게 합니다 (권장)

OPENR, lun, 'data.txt', /GET_LUN
; ... 데이터 읽기 ...
FREE_LUN, lun    ; 닫기 및 LUN 해제
```

## 텍스트 파일 읽기

```idl
OPENR, lun, 'data.txt', /GET_LUN
line = ''
WHILE ~EOF(lun) DO BEGIN
  READF, lun, line
  PRINT, line
ENDWHILE
FREE_LUN, lun
```

### FORMAT을 사용한 읽기

```idl
name = '' & age = 0 & score = 0.0
OPENR, lun, 'data.txt', /GET_LUN
WHILE ~EOF(lun) DO BEGIN
  READF, lun, FORMAT='(A10, I5, F6.1)', name, age, score
  PRINT, STRTRIM(name, 2), age, score
ENDWHILE
FREE_LUN, lun
```

## 텍스트 파일 쓰기

```idl
OPENW, lun, 'output.txt', /GET_LUN
PRINTF, lun, 'This is line 1'
PRINTF, lun, FORMAT='(F8.3, "  ", F10.6)', 1.234, 5.678
FREE_LUN, lun
```

## 바이너리 파일 I/O

```idl
; 쓰기
data = FINDGEN(1000)
OPENW, lun, 'data.bin', /GET_LUN
WRITEU, lun, data
FREE_LUN, lun

; 읽기 — 정확한 형식을 알아야 함
data = FLTARR(1000)
OPENR, lun, 'data.bin', /GET_LUN
READU, lun, data
FREE_LUN, lun
```

## SAVE와 RESTORE

```idl
; 변수 저장
x = FINDGEN(100)
y = SIN(x / 10.0)
SAVE, x, y, FILENAME='results.sav'

; 변수 복원
RESTORE, 'results.sav'
PRINT, N_ELEMENTS(x)
```

## 파일 시스템 작업

```idl
; 파일 검색
fits_files = FILE_SEARCH('/data/*.fits', COUNT=n_files)

; 파일 존재 여부 확인
IF ~FILE_TEST(filename) THEN PRINT, 'File not found'

; 디렉토리 생성
FILE_MKDIR, '/tmp/output'
```

---

## 요약

| 작업 | 프로시저/함수 | 설명 |
|------|--------------|------|
| 읽기용 열기 | `OPENR, lun, file, /GET_LUN` | 기존 파일 열기 |
| 쓰기용 열기 | `OPENW, lun, file, /GET_LUN` | 파일 생성/덮어쓰기 |
| 파일 닫기 | `FREE_LUN, lun` | 닫기 및 LUN 해제 |
| 텍스트 읽기 | `READF, lun, vars` | 포맷된 텍스트 읽기 |
| 텍스트 쓰기 | `PRINTF, lun, vars` | 포맷된 텍스트 쓰기 |
| 바이너리 읽기 | `READU, lun, vars` | 비포맷 바이너리 읽기 |
| 바이너리 쓰기 | `WRITEU, lun, vars` | 비포맷 바이너리 쓰기 |
| 변수 저장 | `SAVE, vars, FILENAME=f` | .sav 파일로 저장 |
| 변수 복원 | `RESTORE, filename` | .sav 파일에서 로드 |
| 파일 찾기 | `FILE_SEARCH(pattern)` | 파일 검색 |
| 파일 테스트 | `FILE_TEST(file)` | 파일 속성 확인 |

---

**이전**: [문자열 처리](./07_String_Processing.md) | **다음**: [구조체](./09_Structures.md)
