# 08. GOES와 RHESSI

**이전**: [SDO/HMI 분석](./07_SDO_HMI_Analysis.md) | **다음**: [스펙트럼 분석](./09_Spectral_Analysis.md)

---

## 학습 목표

1. GOES X선 광도곡선을 읽고 플레어 등급 레이블과 함께 플롯한다
2. GOES 이벤트 목록과 플레어 카탈로그를 활용한다
3. RHESSI 이미징과 분광 개념을 이해한다
4. `hsi_image`로 RHESSI 이미지를 생성한다
5. OSPEX로 RHESSI 스펙트럼 분석을 수행한다

---

## 1. GOES X선 데이터

| 등급 | 플럭스 (W/m^2, 1-8 A) |
|------|----------------------|
| A | < 1e-7 |
| B | 1e-7 ~ 1e-6 |
| C | 1e-6 ~ 1e-5 |
| M | 1e-5 ~ 1e-4 |
| X | > 1e-4 |

```idl
; GOES 데이터 읽기
rd_goes, data, tarray, $
    TSTART='2024-01-01', TEND='2024-01-02', /ONE_MINUTE

; 광도곡선 플롯
PLOT, (tarray - tarray[0]) / 3600.0, data.lo, /YLOG, $
    XTITLE='Time (hours)', YTITLE='Flux (W m!U-2!N)', $
    TITLE='GOES 1-8 A X선 플럭스', $
    YRANGE=[1e-9, 1e-3]
```

---

## 2. GOES 플레어 카탈로그

```idl
rd_gev, gev, TSTART='2024-01-01', TEND='2024-01-31'

FOR i = 0, N_ELEMENTS(gev)-1 DO $
    PRINT, gev[i].st$date, ' Class: ', gev[i].class, ' AR: ', gev[i].noaa
```

---

## 3. RHESSI 이미징

```idl
obj = HSI_IMAGE()
obj->SET, OBS_TIME_INTERVAL = $
    ANYTIM(['2024-01-15 12:00:00', '2024-01-15 12:02:00'])
obj->SET, IMAGE_DIM = [128, 128]
obj->SET, PIXEL_SIZE = [2.0, 2.0]
obj->SET, ENERGY_BAND = [6.0, 12.0]
obj->SET, IMAGE_ALGORITHM = 'CLEAN'

image = obj->GETDATA()
OBJ_DESTROY, obj
```

### RHESSI 이미지 알고리즘

| 알고리즘 | 설명 | 적합한 용도 |
|----------|------|------------|
| BACK_PROJECTION | 단순 역투영 | 빠른 확인 |
| CLEAN | 반복 디컨볼루션 | 일반 사용 |
| MEM_NJIT | 최대 엔트로피 | 확장 소스 |
| PIXON | Pixon 방법 | 복잡한 형태 |

---

## 4. OSPEX 스펙트럼 분석

```idl
o = OSPEX()
o->SET, SPEX_SPECFILE = 'hsi_spectrum.fits'
o->SET, SPEX_DRMFILE = 'hsi_drm.fits'
o->SET, FIT_FUNCTION = 'vth+thick2'  ; 열적 + 두꺼운 표적
o->DOFIT

fit_params = o->GET(/SPEX_SUMM_PARAMS)
PRINT, '온도: ', fit_params[1], ' keV'
```

| 피팅 함수 | 설명 |
|----------|------|
| `vth` | 등온 (CHIANTI) |
| `thick2` | 두꺼운 표적 제동복사 |
| `thin2` | 얇은 표적 제동복사 |
| `bpow` | 꺾인 멱법칙 |

---

## 요약

| 주제 | 핵심 루틴 | 용도 |
|------|----------|------|
| GOES 데이터 | `rd_goes` | X선 광도곡선 |
| GOES 플롯 | `utplot` | 시간축 서식 플롯 |
| 플레어 카탈로그 | `rd_gev` | 이벤트 목록 |
| RHESSI 이미징 | `hsi_image` | X선 이미지 재구성 |
| RHESSI 스펙트럼 | OSPEX | 스펙트럼 피팅 |

---

**이전**: [SDO/HMI 분석](./07_SDO_HMI_Analysis.md) | **다음**: [스펙트럼 분석](./09_Spectral_Analysis.md)
