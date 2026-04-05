# IDL 기초

IDL (Interactive Data Language)은 태양 물리학, 우주 과학, 천문학, 지구 과학 분야에서 널리 사용되는 프로그래밍 언어 및 환경입니다. Research Systems Inc. (현재 NV5 Geospatial Solutions의 Harris Geospatial Solutions)에서 개발한 IDL은 배열 기반 수치 계산, 이미지 처리, 과학적 시각화에 뛰어납니다. GDL (GNU Data Language)은 대부분의 IDL 구문 및 루틴과 호환되는 무료 오픈소스 대안입니다.

이 토픽은 설치와 기본 문법부터 파일 I/O, 플로팅, 태양 물리학 프로젝트까지 IDL 기초를 다룹니다. 위성 데이터, 망원경 관측, 수치 시뮬레이션을 다루든, 이 레슨들은 IDL 또는 GDL을 사용하여 과학 데이터를 읽고, 처리하고, 시각화하는 기술을 제공합니다.

## 학습 내용

이 토픽은 다음의 실습 중심 내용을 제공합니다:
- **시작하기**: 설치 (IDL 및 GDL), IDLDE, 명령줄 사용, 배치 모드
- **핵심 언어**: 변수, 데이터 타입, 배열, 연산자, 제어 흐름
- **프로시저와 함수**: 키워드와 위치 매개변수를 사용한 재사용 가능한 IDL 프로그램 작성
- **문자열**: 문자열 조작 함수, 포맷팅, 정규 표현식
- **파일 I/O**: 텍스트 파일, 바이너리 파일, SAVE/RESTORE, CSV 읽기
- **구조체**: 익명 및 명명된 구조체, 구조체 배열, 동적 생성
- **플로팅**: PLOT, OPLOT, XYOUTS, PostScript 출력, 출판 품질 그림
- **이미지 표시**: TV, TVSCL, 컬러 테이블, 바이트 스케일링, 디바이스 독립 그래픽
- **FITS 파일**: FITS 파일 읽기/쓰기, 헤더 조작, 다중 확장 FITS
- **날짜와 시간**: 율리우스 날짜, 시간 파싱, 플롯용 포맷팅
- **디버깅**: STOP, HELP, RETALL, 메모리 관리, 코딩 모범 사례
- **프로젝트**: FITS 데이터로 태양 광도 곡선 만들기

## 사전 요구사항

- [프로그래밍](../Programming/00_Overview.md) — 일반적인 프로그래밍 개념 (변수, 제어 흐름, 함수)에 대한 기본 이해

사전 IDL 경험은 필요하지 않습니다. 변수, 루프, 함수와 같은 기본 프로그래밍 개념을 이해한다면 시작할 준비가 된 것입니다.

## 학습 로드맵

```
                          IDL 기초 — 학습 경로
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 01 시작하기   │──▶│ 02 변수와        │──▶│ 03 배열과              │  │
  │  │              │   │    데이터 타입    │   │    연산                │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 06 프로시저  │◀──│ 05 제어 흐름     │◀──│ 04 연산자와            │  │
  │  │  와 함수     │   │                  │   │    표현식              │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 07 문자열    │──▶│ 08 파일 I/O      │──▶│ 09 구조체              │  │
  │  │    처리      │   │                  │   │                        │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 12 FITS 파일 │◀──│ 11 이미지        │◀──│ 10 기본                │  │
  │  │    처리      │   │    표시          │   │    플로팅              │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 13 날짜와    │──▶│ 14 디버깅과      │──▶│ 15 프로젝트: 태양      │  │
  │  │    시간      │   │  모범 사례       │   │    광도 곡선           │  │
  │  └──────────────┘   └──────────────────┘   └────────────────────────┘  │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## 레슨

| # | 제목 | 난이도 | 핵심 내용 |
|---|------|--------|----------|
| 01 | [시작하기](01_Getting_Started.md) | ⭐ | 설치, IDLDE, 명령줄, Hello World, GDL, 배치 모드 |
| 02 | [변수와 데이터 타입](02_Variables_and_Data_Types.md) | ⭐ | BYTE, INT, LONG, FLOAT, DOUBLE, COMPLEX, STRING, 타입 변환 |
| 03 | [배열과 연산](03_Arrays_and_Operations.md) | ⭐ | 배열 생성, 인덱싱, 슬라이싱, WHERE, REFORM, 배열 수학 |
| 04 | [연산자와 표현식](04_Operators_and_Expressions.md) | ⭐ | 산술, 관계, 논리, 비트, 문자열 연결 |
| 05 | [제어 흐름](05_Control_Flow.md) | ⭐ | IF/THEN/ELSE, FOR, WHILE, REPEAT, CASE, SWITCH, BEGIN/END |
| 06 | [프로시저와 함수](06_Procedures_and_Functions.md) | ⭐ | PRO, FUNCTION, 키워드, _EXTRA, 스코프, COMMON 블록 |
| 07 | [문자열 처리](07_String_Processing.md) | ⭐ | STRMID, STRPOS, STRSPLIT, STRJOIN, FORMAT, STREGEX |
| 08 | [파일 I/O](08_File_IO.md) | ⭐ | OPENR/OPENW, GET_LUN, READF/PRINTF, 바이너리 I/O, SAVE/RESTORE |
| 09 | [구조체](09_Structures.md) | ⭐⭐ | 익명/명명 구조체, CREATE_STRUCT, 구조체 배열 |
| 10 | [기본 플로팅](10_Basic_Plotting.md) | ⭐ | PLOT, OPLOT, XYOUTS, 축 키워드, PostScript 출력 |
| 11 | [이미지 표시](11_Image_Display.md) | ⭐⭐ | TV, TVSCL, LOADCT, BYTSCL, CONGRID, REBIN, 컬러 테이블 |
| 12 | [FITS 파일 처리](12_FITS_File_Handling.md) | ⭐⭐ | READFITS, WRITEFITS, 헤더, MRDFITS, 다중 확장 FITS |
| 13 | [날짜와 시간](13_Date_and_Time.md) | ⭐ | SYSTIME, JULDAY, CALDAT, 시간 파싱, ANYTIM |
| 14 | [디버깅과 모범 사례](14_Debugging_and_Best_Practices.md) | ⭐⭐ | STOP, HELP, RETALL, HEAP_GC, 코딩 규칙, 벡터화 |
| 15 | [프로젝트: 태양 광도 곡선](15_Project_Solar_Light_Curve.md) | ⭐⭐ | FITS 읽기, 시계열, 출판 품질 플롯, PostScript |

## 권장 학습 순서

01부터 15까지 순서대로 진행하세요. 각 레슨은 이전 레슨에서 소개된 개념을 기반으로 합니다:

1. **환경 설정 (레슨 1)**: IDL 또는 GDL 설치 및 실행
2. **언어 기초 (레슨 2-5)**: 변수, 배열, 연산자, 제어 흐름은 모든 IDL 프로그램의 뼈대를 형성합니다
3. **모듈화 코드 (레슨 6)**: 코드를 프로시저와 함수로 정리
4. **문자열 처리 (레슨 7)**: 텍스트 데이터 파싱 및 포맷팅
5. **데이터 I/O와 구조체 (레슨 8-9)**: 파일 읽기/쓰기 및 복잡한 데이터 구성
6. **시각화 (레슨 10-11)**: 플롯 생성 및 이미지 표시
7. **과학 데이터 (레슨 12-13)**: FITS 파일 및 날짜/시간 작업
8. **전문 기술 (레슨 14)**: 코드 디버깅 및 모범 사례 준수
9. **종합 프로젝트 (레슨 15)**: 모든 것을 태양 물리학 워크플로우로 통합

## 환경 설정

### 옵션 1: IDL (상용)

IDL은 NV5 Geospatial (이전 Harris Geospatial)의 상용 제품입니다. 라이선스가 필요합니다.

```
다운로드: https://www.nv5geospatialsoftware.com/Products/IDL
```

### 옵션 2: GDL (무료, 오픈소스)

GDL (GNU Data Language)은 대부분의 IDL 구문과 호환되는 무료 오픈소스 대안입니다.

```bash
# macOS (Homebrew)
brew install gnudatalanguage

# Ubuntu / Debian
sudo apt-get install gnudatalanguage

# Fedora / RHEL
sudo dnf install gdl

# 소스에서 빌드
git clone https://github.com/gnudatalanguage/gdl.git
cd gdl && mkdir build && cd build
cmake .. && make -j4 && sudo make install
```

설치를 확인하세요:

```bash
# IDL
idl -e "PRINT, 'Hello from IDL'"

# GDL
gdl -e "PRINT, 'Hello from GDL'"
```

각 레슨의 예제 코드는 `examples/IDL_Basics/`에서 사용할 수 있습니다.

## 관련 자료

- [태양 물리학](../Solar_Physics/00_Overview.md) — 태양 관측 데이터 분석과 태양물리학
- [우주 기상](../Space_Weather/00_Overview.md) — 우주 기상 모델링과 예측
- [프로그래밍](../Programming/00_Overview.md) — 언어에 독립적인 프로그래밍 개념

---

**라이선스**: 콘텐츠는 CC BY-NC 4.0으로 라이선스됩니다
