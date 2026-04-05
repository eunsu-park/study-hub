# 13. IDL-Python 브릿지

**이전**: [NetCDF와 HDF5](./12_NetCDF_and_HDF5.md) | **다음**: [성능과 대용량 데이터](./14_Performance_and_Large_Data.md)

---

## 학습 목표

1. 내장 Python 브릿지로 IDL에서 Python 함수를 호출한다
2. pIDLy와 hissw로 Python에서 IDL을 호출한다
3. IDL과 Python 간의 데이터 타입 매핑을 이해한다
4. 하이브리드 IDL-Python 워크플로우를 설계한다
5. IDL에서 Python으로의 마이그레이션 전략을 계획한다

---

## 1. IDL 내장 Python 브릿지

```idl
; Python 모듈 가져오기
np = PYTHON.IMPORT('numpy')
arr = np.array([1.0, 2.0, 3.0])
PRINT, np.mean(arr)

; SunPy 사용
sunpy_map = PYTHON.IMPORT('sunpy.map')
smap = sunpy_map.Map('aia_171.fits')
data = FLOAT(smap.data)
```

---

## 2. 데이터 타입 매핑

| IDL 타입 | Python 타입 | 참고 |
|----------|-------------|------|
| `FLOAT` | `numpy.float32` | |
| `DOUBLE` | `numpy.float64` | |
| `STRING` | `str` | |
| Array | `numpy.ndarray` | 열 우선 -> 행 우선 주의! |

**중요**: IDL은 열 우선(Fortran), Python은 행 우선(C) 순서입니다. 브릿지가 이를 처리하지만 차원 순서에 유의해야 합니다.

---

## 3. Python에서 IDL 호출

### pIDLy

```python
import pidly
idl = pidly.IDL()
idl('x = FINDGEN(100) * 0.1')
x = idl.x  # numpy 배열 반환
idl.close()
```

### hissw

```python
import hissw
ssw_env = hissw.Environment(ssw_packages=['sdo/aia'])
script = "read_sdo, '{{ file }}', index, data"
outputs = ssw_env.run(script, args={'file': 'aia_171.fits'}, save=['data'])
```

---

## 4. 마이그레이션 전략

### 동등 라이브러리

| IDL / SSW | Python 동등물 |
|-----------|--------------|
| Core IDL | NumPy, SciPy |
| READFITS | astropy.io.fits |
| PLOT, CONTOUR | Matplotlib |
| SMOOTH, CONVOL | scipy.ndimage |
| CURVEFIT, MPFIT | scipy.optimize.curve_fit, lmfit |
| SolarSoft | SunPy |
| AIA_PREP | aiapy |
| WCS 루틴 | astropy.wcs |
| ANYTIM | astropy.time |

### 마이그레이션 단계

1. **1단계**: 병행 사용 — SSW 보정은 hissw, 새 분석은 Python
2. **2단계**: I/O 교체 — SunPy/Astropy로 데이터 접근 전환
3. **3단계**: 보정 교체 — aiapy 등이 충분히 성숙하면 교체
4. **4단계**: 완전 Python — Python 동등물이 없는 레거시 코드만 IDL 유지

### IDL에 남겨야 할 것

- Python 동등물이 없는 미션별 보정 파이프라인
- 잘 테스트된 레거시 분석 코드
- OSPEX 스펙트럼 피팅 (Python 버전보다 IDL이 더 성숙)

---

## 요약

| 접근 방식 | 방향 | 적합한 용도 |
|----------|------|------------|
| IDL Python 브릿지 | IDL -> Python | Python 라이브러리 빠른 접근 |
| pIDLy | Python -> IDL | Python에서 IDL 스크립팅 |
| hissw | Python -> SSW IDL | SSW 보정 파이프라인 |
| SunPy/Astropy | 순수 Python | 새 프로젝트 |

---

**이전**: [NetCDF와 HDF5](./12_NetCDF_and_HDF5.md) | **다음**: [성능과 대용량 데이터](./14_Performance_and_Large_Data.md)
