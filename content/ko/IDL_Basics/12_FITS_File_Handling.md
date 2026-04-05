# FITS 파일 처리

**이전**: [이미지 표시](./11_Image_Display.md) | **다음**: [날짜와 시간](./13_Date_and_Time.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. FITS 파일 형식과 천문학에서의 역할 이해하기
2. READFITS와 MRDFITS로 FITS 파일 읽기
3. WRITEFITS와 MWRFITS로 FITS 파일 쓰기
4. SXPAR과 FXADDPAR로 FITS 헤더 조작하기
5. 다중 확장 FITS (MEF) 파일 다루기
6. FITS 데이터 타입 (이미지, 테이블, 바이너리 테이블) 처리하기
7. SDO/AIA와 HMI 데이터를 실제 예제로 읽기

---

FITS (Flexible Image Transport System)는 천문학과 우주 과학의 표준 파일 형식입니다. 1970년대에 개발되어 거의 모든 천문 관측의 주요 데이터 형식이었습니다.

## FITS란?

FITS 파일은 하나 이상의 Header-Data Unit (HDU)으로 구성됩니다. 헤더는 80자 키워드 레코드의 ASCII 텍스트이고, 데이터는 이미지나 테이블입니다.

### BITPIX 값

| BITPIX | 데이터 타입 | IDL 타입 |
|--------|-----------|----------|
| 8 | 부호 없는 바이트 | BYTE |
| 16 | 16비트 부호 있는 정수 | INT |
| 32 | 32비트 부호 있는 정수 | LONG |
| -32 | 32비트 부동소수점 | FLOAT |
| -64 | 64비트 부동소수점 | DOUBLE |

## FITS 파일 읽기

```idl
; 데이터와 헤더 읽기
data = READFITS('image.fits', header)

; 특정 키워드 추출
naxis1 = SXPAR(header, 'NAXIS1')
wavelnth = SXPAR(header, 'WAVELNTH')
date_obs = SXPAR(header, 'DATE-OBS')

; 키워드 존재 여부 확인
value = SXPAR(header, 'MISSING_KEY', COUNT=count)
IF count EQ 0 THEN PRINT, 'Keyword not found'
```

## FITS 파일 쓰기

```idl
data = DIST(256)
MKHDR, header, data    ; 데이터에서 헤더 자동 생성
SXADDPAR, header, 'TELESCOP', 'Simulated', 'Telescope name'
SXADDPAR, header, 'WAVELNTH', 171, 'Wavelength in Angstroms'
WRITEFITS, 'output.fits', data, header
```

## 다중 확장 FITS (MEF)

```idl
; MRDFITS로 확장 읽기
primary = MRDFITS('multi.fits', 0, primary_header)
ext1 = MRDFITS('multi.fits', 1, ext1_header)

; MWRFITS로 확장 쓰기
MWRFITS, primary_data, 'multi.fits', header, /CREATE
MWRFITS, image2, 'multi.fits'    ; 확장 추가

; 바이너리 테이블 확장 쓰기
catalog = REPLICATE({ra: 0.0D, dec: 0.0D, flux: 0.0}, 100)
MWRFITS, catalog, 'catalog.fits', /CREATE
```

## SDO/AIA 예제

```idl
data = READFITS('aia_171.fits', header)
wavelength = SXPAR(header, 'WAVELNTH')
exptime = SXPAR(header, 'EXPTIME')
data_norm = data / exptime
display = BYTSCL(ALOG10(data_norm > 1.0), MIN=0, MAX=4)
LOADCT, 1, /SILENT
TV, CONGRID(display, 512, 512)
```

## 헤더 조작

```idl
; 키워드 수정
SXADDPAR, header, 'EXPTIME', 4.0, 'Updated exposure time'
SXDELPAR, header, 'OBSOLETE_KEY'
SXADDPAR, header, 'HISTORY', 'Processed on ' + SYSTIME()
```

---

## 요약

| 함수/프로시저 | 설명 |
|--------------|------|
| `READFITS(file, header)` | FITS 파일 읽기 |
| `WRITEFITS, file, data, header` | FITS 파일 쓰기 |
| `MRDFITS(file, ext, header)` | 다중 확장 FITS 읽기 |
| `MWRFITS, data, file` | FITS 확장 쓰기/추가 |
| `SXPAR(header, key)` | 헤더 키워드 값 읽기 |
| `SXADDPAR, header, key, val` | 헤더 키워드 추가/수정 |
| `MKHDR, header, data` | 최소 헤더 생성 |

---

**이전**: [이미지 표시](./11_Image_Display.md) | **다음**: [날짜와 시간](./13_Date_and_Time.md)
