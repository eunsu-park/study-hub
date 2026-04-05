# 11. 곡선 피팅

**이전**: [이미지 처리](./10_Image_Processing.md) | **다음**: [NetCDF와 HDF5](./12_NetCDF_and_HDF5.md)

---

## 학습 목표

1. POLY_FIT과 LINFIT으로 다항식/선형 피팅을 수행한다
2. GAUSSFIT으로 가우시안 피팅을 수행한다
3. CURVEFIT으로 비선형 최소제곱 피팅을 사용한다
4. MPFIT(Markwardt)으로 견고한 매개변수 추정을 적용한다
5. 카이제곱 분석과 신뢰 구간을 계산한다

---

## 1. 선형 및 다항식 피팅

```idl
; 선형 피팅: y = a + b*x
result = LINFIT(x, y, SIGMA=sigma)
PRINT, '절편: ', result[0], ' +/- ', sigma[0]
PRINT, '기울기: ', result[1], ' +/- ', sigma[1]

; 다항식 피팅 (2차)
coeffs = POLY_FIT(x, y, 2, SIGMA=sigma, CHISQ=chisq)
```

---

## 2. 가우시안 피팅

```idl
; GAUSSFIT — 단일 가우시안 피팅
yfit = GAUSSFIT(x, y, coeffs, NTERMS=4)
; coeffs = [진폭, 중심, 시그마, 배경]

PRINT, 'FWHM: ', 2.354 * coeffs[2]
```

---

## 3. CURVEFIT — 비선형 최소제곱

```idl
; 사용자 정의 함수: PRO func, x, params, ymodel, pder
PRO exp_decay, x, p, ymod, pder
    ymod = p[0] * EXP(-p[1] * x) + p[2]
    IF N_PARAMS() GE 4 THEN BEGIN
        pder = FLTARR(N_ELEMENTS(x), 3)
        pder[*, 0] = EXP(-p[1] * x)
        pder[*, 1] = -p[0] * x * EXP(-p[1] * x)
        pder[*, 2] = 1.0
    ENDIF
END

params = [8.0, 0.2, 1.5]
yfit = CURVEFIT(x, y, weights, params, sigma, $
    FUNCTION_NAME='exp_decay', CHISQ=chisq)
```

---

## 4. MPFIT — Markwardt Levenberg-Marquardt

MPFIT은 IDL에서 곡선 피팅의 표준입니다. 매개변수 제약, 고정, 더 나은 수렴을 제공합니다.

```idl
FUNCTION my_model, x, p
    RETURN, p[0] * EXP(-0.5*((x - p[1])/p[2])^2) + p[3]
END

; MPFITFUN으로 피팅
params = MPFITFUN('my_model', x, y, yerr, p0, $
    PERROR=perror, BESTNORM=bestnorm, DOF=dof)

PRINT, '축소된 카이제곱: ', bestnorm / dof
```

### PARINFO를 이용한 매개변수 제약

```idl
parinfo = REPLICATE({value: 0.D, fixed: 0, limited: [0,0], $
    limits: [0.D, 0.D]}, 4)
parinfo[0].value = 4.0
parinfo[0].limited = [1, 0]  ; 하한 활성
parinfo[0].limits[0] = 0.0   ; 진폭 > 0
```

---

## 5. 카이제곱 분석

```idl
; 축소된 카이제곱: chi2_red = chi2 / dof
residuals = y - my_model(x, params)
chi2 = TOTAL((residuals / yerr)^2)
chi2_red = chi2 / (N_ELEMENTS(x) - N_ELEMENTS(params))
; chi2_red ~ 1.0이면 좋은 피팅
```

---

## 6. 실전: 태양 스펙트럼 선 피팅

```idl
; 방출선 프로파일 피팅 (예: EIS, IRIS)
FUNCTION spectral_line, x, p
    gaussian = p[0] * EXP(-0.5*((x - p[1])/p[2])^2)
    background = p[3] + p[4] * (x - MEAN(x))
    RETURN, gaussian + background
END

params = MPFITFUN('spectral_line', wavelength, spectrum, spec_err, p0, $
    PARINFO=parinfo, PERROR=perror)

; 도플러 속도 계산
v_doppler = (params[1] - rest_wavelength) / rest_wavelength * 3e5  ; km/s
```

---

## 요약

| 방법 | IDL 함수 | 적합한 용도 |
|------|---------|------------|
| 선형 | `LINFIT` | y = a + bx |
| 다항식 | `POLY_FIT` | y = sum(c_i * x^i) |
| 가우시안 | `GAUSSFIT` | 단일 가우시안 피크 |
| 비선형 | `CURVEFIT` | 일반 비선형 모델 |
| MPFIT | `MPFITFUN` | 제약 비선형 (권장) |

---

**이전**: [이미지 처리](./10_Image_Processing.md) | **다음**: [NetCDF와 HDF5](./12_NetCDF_and_HDF5.md)
