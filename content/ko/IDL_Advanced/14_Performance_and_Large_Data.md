# 14. 성능과 대용량 데이터

**이전**: [IDL-Python 브릿지](./13_IDL_Python_Bridge.md) | **다음**: [캡스톤: 태양 이벤트 분석](./15_Capstone_Solar_Event_Analysis.md)

---

## 학습 목표

1. ASSOC로 대용량 데이터셋에 메모리 매핑 파일 접근을 사용한다
2. SYSTIME과 PROFILER로 IDL 코드를 벤치마킹한다
3. 느린 스칼라 루프를 피하는 벡터화 코드를 작성한다
4. TEMPORARY, HEAP_GC, PTR_FREE로 메모리를 관리한다
5. 대량 파일 컬렉션의 배치 처리를 최적화한다

---

## 1. 벡터화 vs 루프

IDL에서 가장 중요한 최적화: **스칼라 루프를 배열 연산으로 교체**.

```idl
n = 10000000L
x = RANDOMU(seed, n)
y = RANDOMU(seed, n)

; 방법 1: 스칼라 루프 (느림)
FOR i = 0L, n-1 DO result[i] = SQRT(x[i]^2 + y[i]^2)

; 방법 2: 벡터화 (빠름)
result = SQRT(x^2 + y^2)
; 일반적인 속도 향상: 50-200배
```

### 일반적인 벡터화 패턴

```idl
; 조건부 할당
; 느림: FOR 루프에서 IF 사용
; 빠름:
data = data > 0.0
; 또는:
neg = WHERE(data LT 0, count)
IF count GT 0 THEN data[neg] = 0.0

; 누적
; 느림: FOR 루프
; 빠름: TOTAL(data)

; 스무딩
; 느림: 픽셀별 FOR 루프
; 빠름: SMOOTH(img, 5, /EDGE_TRUNCATE)
```

---

## 2. TEMPORARY 함수

`TEMPORARY`는 입력 변수의 메모리를 즉시 해제하여 출력에 재사용합니다.

```idl
; TEMPORARY 없이: 피크 메모리 = 2 * sizeof(data)
data = FLTARR(4096, 4096)
result = ALOG10(data + 1.0)  ; data가 여전히 할당됨

; TEMPORARY 사용: 피크 메모리 절반
data = FLTARR(4096, 4096)
result = ALOG10(TEMPORARY(data) + 1.0)
; data는 이제 정의되지 않음 (해제됨)
```

---

## 3. 메모리 관리

```idl
; 메모리 사용량 확인
HELP, /MEMORY
mem = MEMORY(/CURRENT)
PRINT, '현재 힙: ', mem / 1e6, ' MB'

; 변수 해제
large_array = 0  ; 메모리 해제 (스칼라 0으로 대체)

; 포인터 해제
PTR_FREE, ptr

; 가비지 컬렉션
HEAP_GC, /VERBOSE  ; 참조되지 않는 포인터/객체 찾아 해제
```

---

## 4. ASSOC — 메모리 매핑 파일 접근

RAM보다 큰 파일을 처리할 때 필수적입니다.

```idl
; 메모리 매핑으로 대용량 파일 읽기
OPENR, lun, 'large_cube.dat', /GET_LUN
cube = ASSOC(lun, FLTARR(4096, 4096))
; cube는 메모리에 없음 — 파일 매핑

; 개별 프레임 접근 (해당 프레임만 디스크에서 읽음)
frame0 = cube[0]
frame100 = cube[100]

; 전체 파일 로드 없이 시간 평균 계산
temporal_sum = FLTARR(4096, 4096)
FOR t = 0L, 499 DO temporal_sum += cube[t]
temporal_mean = temporal_sum / 500.0

FREE_LUN, lun
```

---

## 5. 벤치마킹

```idl
; 기본 타이밍
t0 = SYSTIME(1)
; ... 벤치마킹할 코드 ...
PRINT, '경과: ', SYSTIME(1) - t0, ' 초'

; PROFILER
PROFILER, /SYSTEM
my_procedure, data
PROFILER, /REPORT  ; 타이밍 리포트 출력
```

---

## 6. 배치 처리 패턴

```idl
PRO batch_process, input_dir, output_dir
    files = FILE_SEARCH(input_dir + '/*.fits', COUNT=nf)
    t_start = SYSTIME(1)

    FOR i = 0L, nf-1 DO BEGIN
        read_sdo, files[i], index, data
        aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER

        outfile = output_dir + '/' + FILE_BASENAME(files[i])
        mwritefits, oindex, odata, OUTFILE=outfile

        ; 진행 보고
        IF (i MOD 50) EQ 0 THEN BEGIN
            elapsed = SYSTIME(1) - t_start
            rate = (i+1) / elapsed
            eta = (nf - i - 1) / rate
            PRINT, STRING(i+1, nf, eta, FORMAT='(I5, "/", I5, " ETA: ", F7.1, "s")')
        ENDIF

        data = 0 & odata = 0  ; 매 반복마다 메모리 해제
    ENDFOR
END
```

---

## 7. SAVE/RESTORE 최적화

```idl
; 중간 결과 캐싱
cache_file = 'calibrated_cache.sav'
IF FILE_TEST(cache_file) THEN BEGIN
    RESTORE, cache_file
ENDIF ELSE BEGIN
    ; 비용이 큰 계산 수행
    ; ...
    SAVE, data_cube, FILENAME=cache_file
ENDELSE
```

---

## 요약

| 기법 | 핵심 함수 | 효과 |
|------|----------|------|
| 벡터화 | 배열 연산자 | 50-200배 속도 향상 |
| TEMPORARY | `TEMPORARY()` | 피크 메모리 절반 |
| ASSOC | `ASSOC()` | RAM보다 큰 파일 |
| 프로파일링 | `SYSTIME(1)`, `PROFILER` | 병목 식별 |
| 메모리 관리 | `HEAP_GC`, `MEMORY()` | 충돌 방지 |

---

**이전**: [IDL-Python 브릿지](./13_IDL_Python_Bridge.md) | **다음**: [캡스톤: 태양 이벤트 분석](./15_Capstone_Solar_Event_Analysis.md)
