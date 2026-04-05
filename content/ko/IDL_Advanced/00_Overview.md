# IDL 고급 — 학습 가이드

## 소개

이 폴더는 SolarSoft(SSW)를 활용한 과학 데이터 분석에 초점을 맞춘 **포괄적인 IDL(Interactive Data Language) 고급 커리큘럼**을 제공합니다. IDL 기초에서 언어 기본을 다루었다면, 이 과정에서는 고급 배열 조작, 출판 품질의 시각화, 태양 관측 장비 파이프라인(SDO/AIA, SDO/HMI, GOES, RHESSI), 스펙트럼 분석, 이미지 처리, 곡선 피팅, Python과의 상호운용성, 대규모 데이터 성능 최적화를 깊이 다룹니다.

IDL은 태양물리학과 우주과학의 핵심 언어로 남아있습니다. SolarSoft 라이브러리 모음은 수십 년에 걸쳐 정제된 보정 파이프라인, 좌표 변환, 데이터 접근 유틸리티를 제공합니다. 이러한 도구를 숙달하는 것은 태양권물리학 연구에 필수적입니다.

## 학습 내용

- 고급 배열 조작 및 다차원 데이터 기법
- 출판 품질 플로팅: 다중 패널, 등고선, 표면, PostScript 출력
- 지도 투영법과 태양 좌표계
- 객체지향 IDL과 위젯 프로그래밍
- SolarSoft 프레임워크: 설치, 장비 트리, 유틸리티 루틴
- SDO/AIA 보정, 다중 파장 분석, DEM 추정
- SDO/HMI 자기장 분석과 Carrington 지도
- GOES X선 광도곡선과 RHESSI 스펙트럼/이미지 분석
- 스펙트럼 분석: FFT, 웨이블릿, Lomb-Scargle
- 이미지 처리: 필터링, 형태학, 에지 검출, 특징 추적
- 곡선 피팅: CURVEFIT, MPFIT, Gaussian 피팅, 카이제곱 분석
- 과학 파일 형식: NetCDF, HDF5, CDF
- IDL-Python 브릿지와 마이그레이션 전략
- 성능 최적화와 대용량 데이터셋 처리
- 캡스톤: 종합 태양 플레어 이벤트 분석

## 사전 요구 사항

| 토픽 | 필요 수준 |
|------|----------|
| **[IDL 기초](../IDL_Basics/00_Overview.md)** | 능숙 — 변수, 배열, 제어 흐름, 기본 플로팅, FITS I/O |
| **[Solar_Physics](../Solar_Physics/00_Overview.md)** | 기본 — 태양 대기, 플레어, CME |
| Linux/Shell | 기본 — 명령줄, 환경 변수 |

## 학습 경로

```
┌─────────────────────────────────┐
│  Block 1: 고급 기초              │  L01–L04
│  배열, 플로팅, 지도, OOP        │
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│  Block 2: SolarSoft &           │  L05–L08
│  태양 관측 장비                  │  SSW, AIA, HMI, GOES, RHESSI
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│  Block 3: 분석 기법              │  L09–L12
│  스펙트럼, 이미지, 피팅, I/O    │
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│  Block 4: 통합                   │  L13–L15
│  Python 브릿지, 성능,           │
│  캡스톤 프로젝트                 │
└─────────────────────────────────┘
```

## 레슨 목록

| # | 파일명 | 설명 |
|---|--------|------|
| **Block 1: 고급 기초** |
| 01 | `01_Advanced_Array_Techniques.md` | REFORM, REBIN, CONGRID, TOTAL, MEDIAN, SMOOTH, CONVOL, IMAGE_STATISTICS |
| 02 | `02_Advanced_Plotting.md` | 다중 패널 플롯, CONTOUR, SURFACE, PLOTS, PostScript 출력 |
| 03 | `03_Map_Projections.md` | MAP_SET, MAP_CONTINENTS, 좌표 변환, WCS |
| 04 | `04_Object_Oriented_IDL.md` | 클래스 정의, 상속, 위젯 프로그래밍 |
| **Block 2: SolarSoft & 태양 관측 장비** |
| 05 | `05_SolarSoft_Framework.md` | SSW 설치, 장비 트리, 유틸리티 루틴 |
| 06 | `06_SDO_AIA_Analysis.md` | AIA_PREP, 다중 파장 분석, DEM 기초 |
| 07 | `07_SDO_HMI_Analysis.md` | 자기도, 벡터 자기장, Carrington 지도 |
| 08 | `08_GOES_and_RHESSI.md` | GOES 광도곡선, RHESSI 이미징 및 분광 |
| **Block 3: 분석 기법** |
| 09 | `09_Spectral_Analysis.md` | FFT, 웨이블릿, Lomb-Scargle, 스펙트럼 필터링 |
| 10 | `10_Image_Processing.md` | 필터링, 형태학, 에지 검출, 특징 추적 |
| 11 | `11_Curve_Fitting.md` | CURVEFIT, MPFIT, GAUSSFIT, 카이제곱 분석 |
| 12 | `12_NetCDF_and_HDF5.md` | NetCDF, HDF5, CDF 파일 I/O |
| **Block 4: 통합** |
| 13 | `13_IDL_Python_Bridge.md` | Python 브릿지, pIDLy, hissw, 마이그레이션 전략 |
| 14 | `14_Performance_and_Large_Data.md` | ASSOC, 벡터화, 메모리 관리, 프로파일링 |
| 15 | `15_Capstone_Solar_Event_Analysis.md` | 종합 태양 플레어 분석 프로젝트 |

## 환경 설정

### SolarSoft 설치

레슨 05-08과 15에는 SolarSoft(SSW)가 필요합니다:

```bash
# SolarSoft 다운로드
export SSW=/usr/local/ssw
mkdir -p $SSW
cd $SSW
wget https://www.lmsal.com/solarsoft/ssw_install.tar
tar xf ssw_install.tar

# 장비 패키지 설치
ssw_install, /sdo, /aia, /hmi, /goes, /hessi

# 환경 변수 설정
export SSW=/usr/local/ssw
export SSW_INSTR="aia hmi goes hessi"
source $SSW/gen/setup/setup.ssw
```

### SolarSoft IDL 시작

```bash
sswidl          # SolarSoft 환경이 로드된 IDL 실행
```

## 관련 자료

- **[IDL 기초](../IDL_Basics/00_Overview.md)** — 언어 기본, 이 과정의 선수 과목
- **[Solar_Physics](../Solar_Physics/00_Overview.md)** — 태양 대기, 플레어, CME
- **[Space_Weather](../Space_Weather/00_Overview.md)** — 지자기 폭풍, 예보
- **[Plasma_Physics](../Plasma_Physics/00_Overview.md)** — MHD, 플라즈마 파동, 재결합

---

*[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 라이선스에 따라 배포됩니다*
