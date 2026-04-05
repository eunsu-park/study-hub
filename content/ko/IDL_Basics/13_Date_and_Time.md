# 날짜와 시간

**이전**: [FITS 파일 처리](./12_FITS_File_Handling.md) | **다음**: [디버깅과 모범 사례](./14_Debugging_and_Best_Practices.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. SYSTIME으로 현재 시간 가져오기
2. JULDAY와 CALDAT로 율리우스 날짜 작업하기
3. 날짜 문자열을 숫자 구성 요소로 파싱하기
4. ANYTIM으로 유연한 시간 변환하기 (SolarSoft)
5. 시간 산술 수행하기 (시간, 일 추가, 간격 찾기)
6. 플롯 축 라벨용 날짜 포맷팅하기
7. 시계열 분석을 위한 시간 배열 생성하기

---

날짜와 시간 처리는 과학 데이터 분석에서 필수적입니다. 관측 타임스탬프, 노출 시간, 시계열 모두 정확한 시간 표현과 변환이 필요합니다.

## SYSTIME — 현재 시스템 시간

```idl
PRINT, SYSTIME()            ; 문자열로 현재 시간
PRINT, SYSTIME(1)           ; 1970년 1월 1일 이후 초
PRINT, SYSTIME(/UTC)        ; UTC (현지 시간이 아님)
PRINT, SYSTIME(/JULIAN)     ; 현재 율리우스 날짜
```

## 율리우스 날짜

율리우스 날짜 (JD)는 기원전 4713년 1월 1일 이후의 연속적인 일수입니다.

### JULDAY — 달력에서 율리우스 날짜로

```idl
; JULDAY(월, 일, 년 [, 시, 분, 초])
jd = JULDAY(7, 15, 2024, 12, 0, 0)
; 매개변수 순서에 주의: 월, 일, 년 (년, 월, 일이 아님)
```

### CALDAT — 율리우스 날짜에서 달력으로

```idl
CALDAT, jd, month, day, year, hour, minute, second
PRINT, FORMAT='(I4, "-", I02, "-", I02, " ", I02, ":", I02, ":", I02)', $
  year, month, day, hour, minute, second
```

## 시간 산술

```idl
start_jd = JULDAY(7, 15, 2024, 12, 0, 0)

; 율리우스 날짜에서 1.0 = 1일
next_day = start_jd + 1.0D0          ; 1일 추가
plus_6h = start_jd + 6.0D0 / 24.0D0  ; 6시간 추가
plus_30m = start_jd + 30.0D0 / 1440.0D0  ; 30분 추가

; 시간 차이
jd1 = JULDAY(7, 15, 2024, 12, 0, 0)
jd2 = JULDAY(7, 20, 2024, 18, 30, 0)
diff_days = jd2 - jd1
diff_hours = diff_days * 24.0D0
```

## 시간 배열 생성

```idl
start_jd = JULDAY(7, 15, 2024, 0, 0, 0)
cadence_sec = 12.0D0    ; 12초 간격 (AIA와 같음)
n_steps = 7200           ; 12초 간격으로 24시간

time_jd = start_jd + DINDGEN(n_steps) * cadence_sec / 86400.0D0
time_hours = (time_jd - start_jd) * 24.0D0
```

## 플롯용 날짜 포맷팅

```idl
; 사용자 정의 틱 포맷 함수
FUNCTION time_tick_format, axis, index, value
  CALDAT, value, month, day, year, hour, minute
  months = ['Jan','Feb','Mar','Apr','May','Jun',$
            'Jul','Aug','Sep','Oct','Nov','Dec']
  RETURN, STRING(FORMAT='(I02, ":", I02, " ", A3, " ", I02)', $
    hour, minute, months[month-1], day)
END
```

## 연중 일수 (DOY)

```idl
year = 2024 & month = 7 & day = 15
jd_date = JULDAY(month, day, year)
jd_jan1 = JULDAY(1, 1, year)
doy = LONG(jd_date - jd_jan1) + 1
PRINT, 'DOY:', doy    ; 197
```

---

## 요약

| 함수/프로시저 | 설명 |
|--------------|------|
| `SYSTIME()` | 현재 시스템 시간 (문자열 또는 초) |
| `SYSTIME(/JULIAN)` | 현재 율리우스 날짜 |
| `JULDAY(M, D, Y, H, MN, S)` | 달력에서 율리우스 날짜로 |
| `CALDAT, JD, M, D, Y, H, MN, S` | 율리우스 날짜에서 달력으로 |
| `LABEL_DATE` | 플롯 축용 율리우스 날짜 포맷 |

| 시간 산술 | 값 |
|----------|-----|
| 1일 | 1.0 JD |
| 1시간 | 1.0/24.0 JD |
| 1분 | 1.0/1440.0 JD |
| 1초 | 1.0/86400.0 JD |

---

**이전**: [FITS 파일 처리](./12_FITS_File_Handling.md) | **다음**: [디버깅과 모범 사례](./14_Debugging_and_Best_Practices.md)
