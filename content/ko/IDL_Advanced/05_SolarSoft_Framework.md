# 05. SolarSoft 프레임워크

**이전**: [객체지향 IDL](./04_Object_Oriented_IDL.md) | **다음**: [SDO/AIA 분석](./06_SDO_AIA_Analysis.md)

---

## 학습 목표

1. SolarSoft(SSW)를 시스템에 설치하고 구성한다
2. SSW 디렉토리 구조와 장비 트리를 탐색한다
3. SSW 환경 변수와 sswidl 시작을 사용한다
4. SSW 시간 처리 유틸리티(ANYTIM, UTC2TAI 등)를 활용한다
5. SSW_JSOC와 VSO를 통해 데이터를 조회한다

---

## 1. SolarSoft란?

SolarSoft(SSW)는 태양물리학을 위한 종합 IDL 라이브러리 시스템입니다:

- **장비 보정 파이프라인**: SDO/AIA, SDO/HMI, SOHO/EIT, Hinode, STEREO 등
- **좌표 변환**: heliographic, heliocentric, Carrington
- **시간 유틸리티**: 유연한 시간 파싱, TAI/UTC 변환
- **데이터 접근**: JSOC, VSO, SDAC

---

## 2. 설치 및 설정

```bash
export SSW=/usr/local/ssw
mkdir -p $SSW && cd $SSW
wget https://www.lmsal.com/solarsoft/ssw_install.tar
tar xf ssw_install.tar

# IDL에서 장비 설치
# ssw_install, /sdo, /aia, /hmi, /goes, /hessi

export SSW_INSTR="aia hmi goes hessi"
source $SSW/gen/setup/setup.ssw

# SolarSoft IDL 시작
sswidl
```

---

## 3. SSW 시간 유틸리티

### ANYTIM — 범용 시간 파서

```idl
; 거의 모든 시간 문자열을 표준 형식으로 변환
t1 = ANYTIM('2024-01-15 12:00:00')
t2 = ANYTIM('15-Jan-2024 12:00:00')

; 출력 형식 옵션
t_ccsds = ANYTIM('2024-01-15', /CCSDS)  ; '2024-01-15T00:00:00.000'
t_vms   = ANYTIM('2024-01-15', /VMS)    ; '15-Jan-2024 00:00:00.000'

; 시간 연산
t0 = ANYTIM('2024-01-15T12:00:00')
t1 = t0 + 3600.0  ; 1시간 추가
```

### TIM2CARR — Carrington 회전

```idl
carr = TIM2CARR('2024-01-15T12:00:00')
PRINT, 'Carrington 회전: ', LONG(carr)

; 역변환
time = CARR2TIM(2277.0)
```

---

## 4. SSW 데이터 구조

### Index/Data 패러다임

```idl
; SSW 표준 패턴: read_xxx가 index(헤더)와 data(이미지)를 반환
MREADFITS, 'solar_image.fits', index, data
PRINT, 'Date: ', index.date_obs

; SSW Map 구조
index2map, index, data, map
PLOT_MAP, map, /LIMB

; Sub-map 추출
map_sub = SUB_MAP(map, XRANGE=[-500, 500], YRANGE=[-500, 500])
```

---

## 5. 데이터 접근: VSO와 JSOC

```idl
; Virtual Solar Observatory (VSO) 검색
vso_search, '2024-01-15T00:00:00', '2024-01-15T01:00:00', $
    INSTRUMENT='aia', WAVE='171', results, /FLAT

; 데이터 다운로드
vso_get, results, OUT_DIR='./data/', FILENAMES=downloaded_files
```

---

## 6. SSW 유틸리티 루틴

```idl
; 태양 천문력
sun = GET_SUN('2024-01-15T12:00:00')
PRINT, 'B0 각도:     ', sun.b0, ' deg'
PRINT, 'L0 (Carr lon):', sun.l0, ' deg'
PRINT, 'R_sun:        ', sun.sd, ' arcsec'

; 파일명에서 시간 추출
time = SSW_FILE2TIME('aia.lev1.171A_2024-01-15T12_00_00.fits')
```

---

## 요약

| 주제 | 핵심 루틴 | 용도 |
|------|----------|------|
| 설치 | `ssw_install`, `setup.ssw` | SSW 설정 |
| 시간 | `ANYTIM`, `UTC2TAI`, `TIM2CARR` | 시간 변환 |
| 데이터 I/O | `MREADFITS`, `MWRITEFITS` | FITS 파일 접근 |
| 맵 | `INDEX2MAP`, `PLOT_MAP` | 태양 맵 구조 |
| 데이터 접근 | `VSO_SEARCH`, `VSO_GET` | 데이터 다운로드 |
| 천문력 | `GET_SUN`, `PB0R` | 태양 기하학 |

---

**이전**: [객체지향 IDL](./04_Object_Oriented_IDL.md) | **다음**: [SDO/AIA 분석](./06_SDO_AIA_Analysis.md)
