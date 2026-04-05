# 12. NetCDF와 HDF5

**이전**: [곡선 피팅](./11_Curve_Fitting.md) | **다음**: [IDL-Python 브릿지](./13_IDL_Python_Bridge.md)

---

## 학습 목표

1. IDL의 NCDF 루틴으로 NetCDF 파일을 읽고 쓴다
2. IDL의 H5 루틴으로 HDF5 파일을 읽고 쓴다
3. CDF 파일로 우주물리학 데이터(CDAWeb/SPDF)를 다룬다
4. HDF5와 NetCDF의 계층적 데이터 구조를 탐색한다

---

## 1. NetCDF 파일 I/O

```idl
; NetCDF 파일 읽기
ncid = NCDF_OPEN('solar_wind.nc', /NOWRITE)
varid = NCDF_VARID(ncid, 'velocity')
NCDF_VARGET, ncid, varid, velocity
NCDF_ATTGET, ncid, varid, 'units', units_bytes
NCDF_CLOSE, ncid

; NetCDF 파일 쓰기
ncid = NCDF_CREATE('output.nc', /CLOBBER)
time_dimid = NCDF_DIMDEF(ncid, 'time', /UNLIMITED)
x_dimid = NCDF_DIMDEF(ncid, 'x', 100)
data_varid = NCDF_VARDEF(ncid, 'magnetic_field', [x_dimid, time_dimid], /FLOAT)
NCDF_ATTPUT, ncid, data_varid, 'units', 'Gauss'
NCDF_CONTROL, ncid, /ENDEF
NCDF_VARPUT, ncid, data_varid, data_array
NCDF_CLOSE, ncid
```

---

## 2. HDF5 파일 I/O

```idl
; HDF5 파일 읽기
file_id = H5F_OPEN('satellite_data.h5')
dataset_id = H5D_OPEN(file_id, '/measurements/magnetic_field')
data = H5D_READ(dataset_id)

; 속성 읽기
attr_id = H5A_OPEN_NAME(dataset_id, 'units')
units = H5A_READ(attr_id)
H5A_CLOSE, attr_id

H5D_CLOSE, dataset_id
H5F_CLOSE, file_id

; 빠른 HDF5 탐색
result = H5_PARSE('satellite_data.h5', /READ_DATA)
```

---

## 3. CDF 파일 (Common Data Format)

CDF는 NASA CDAWeb/SPDF를 통해 배포되는 우주물리학 데이터의 표준 형식입니다.

```idl
cdf_id = CDF_OPEN('ace_swepam_data.cdf')
CDF_VARGET, cdf_id, 'Epoch', epoch_data, /ZVAR
CDF_VARGET, cdf_id, 'V_GSE', velocity, /ZVAR
CDF_VARGET, cdf_id, 'Np', density, /ZVAR

; CDF 에포크를 읽기 가능한 시간으로 변환
CDF_EPOCH, epoch_data[0], yr, mo, dy, hr, mn, sc, ms, /BREAKDOWN_EPOCH

CDF_CLOSE, cdf_id
```

---

## 4. 형식 비교

| 특징 | FITS | NetCDF | HDF5 | CDF |
|------|------|--------|------|-----|
| 주요 분야 | 천문학 | 기후/대기 | 일반 과학 | 우주물리학 |
| 계층적 | 아니오 | 그룹 (v4) | 예 | 아니오 |
| 압축 | Rice, GZIP | GZIP (v4) | GZIP, SZIP | GZIP |
| IDL 지원 | READFITS | NCDF_* | H5*_ | CDF_* |

---

## 요약

| 형식 | 핵심 루틴 | 열기/생성 |
|------|----------|----------|
| NetCDF | `NCDF_OPEN`, `NCDF_VARGET` | `NCDF_CREATE` |
| HDF5 | `H5F_OPEN`, `H5D_READ` | `H5F_CREATE` |
| CDF | `CDF_OPEN`, `CDF_VARGET` | `CDF_CREATE` |

---

**이전**: [곡선 피팅](./11_Curve_Fitting.md) | **다음**: [IDL-Python 브릿지](./13_IDL_Python_Bridge.md)
