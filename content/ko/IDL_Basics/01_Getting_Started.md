# IDL 시작하기

**다음**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. IDL이 무엇이며 과학 컴퓨팅에서 왜 사용되는지 설명하기
2. 시스템에 IDL 또는 GDL 설치하기
3. IDLDE (IDL 개발 환경)와 명령줄 인터페이스 사용하기
4. PRINT를 사용하여 첫 번째 IDL 프로그램 작성하고 실행하기
5. IDL과 GDL의 차이점 이해하기
6. .RUN, .COMPILE, RETALL 같은 기본 IDL 명령어 사용하기
7. 배치 모드에서 IDL 프로그램 실행하기
8. IDL 라이선스 및 대안 이해하기

---

IDL (Interactive Data Language)은 1970년대부터 과학 컴퓨팅의 핵심 도구였습니다. 콜로라도 대학교 LASP (Laboratory for Atmospheric and Space Physics)의 David Stern이 개발한 IDL은 태양 물리학, 천문학, 우주 과학, 원격 탐사, 의료 영상 분야의 데이터 분석에 표준 도구가 되었습니다. 배열 지향 구문, 내장 시각화, 풍부한 과학 루틴 라이브러리는 다차원 데이터셋 작업에 특히 적합합니다.

## 왜 IDL인가?

### 역사적 중요성

IDL은 주요 과학 미션과 프로젝트에서 선택된 언어입니다:

- **NASA Solar Dynamics Observatory (SDO)**: IDL로 구축된 데이터 파이프라인
- **SOHO (Solar and Heliospheric Observatory)**: 주요 분석 언어
- **SolarSoft (SSW)**: 수천 개의 루틴을 갖춘 태양 물리학용 대규모 IDL 라이브러리
- **허블 우주 망원경**: 많은 분석 도구가 IDL로 작성됨
- **GOES X-ray Sensor**: IDL을 사용한 표준 데이터 분석

### 주요 강점

- **배열 지향**: 명시적 루프 없이 전체 배열에 대한 연산
- **내장 시각화**: 최소한의 코드로 출판 품질 플로팅
- **과학 라이브러리**: 광범위한 수학, 통계, 이미지 처리, 신호 처리
- **FITS 지원**: FITS (Flexible Image Transport System) 형식의 네이티브 지원
- **대화형**: 명령 프롬프트에서 아이디어를 테스트한 후 스크립트로 저장
- **성숙한 생태계**: 수십 년간 검증된 우주 과학 루틴

### IDL vs. 현대 대안

```
언어         │ 라이선스     │ 배열 구문    │ 시각화        │ FITS 지원    │ 레거시 코드
─────────────┼──────────────┼──────────────┼───────────────┼──────────────┼────────────
IDL          │ 상용         │ 우수         │ 내장          │ 네이티브     │ 방대
GDL          │ 무료 (GPL)   │ 우수         │ 내장          │ 네이티브     │ 호환
Python       │ 무료 (BSD)   │ NumPy        │ Matplotlib    │ astropy.io   │ 성장 중
MATLAB       │ 상용         │ 우수         │ 내장          │ 제한적       │ 다름
Julia        │ 무료 (MIT)   │ 우수         │ Plots.jl      │ FITSIO.jl    │ 최소
```

---

## IDL 설치

### 상용 IDL

IDL은 NV5 Geospatial Solutions에서 배포합니다.

```
공식 웹사이트: https://www.nv5geospatialsoftware.com/Products/IDL
```

#### 라이선스 유형

- **풀 라이선스**: 모든 기능을 포함한 완전한 IDL
- **IDL 가상 머신**: 미리 컴파일된 .sav 파일 실행용 무료 런타임 (편집 불가)
- **학생 라이선스**: 학술용 할인
- **플로팅 라이선스**: 네트워크를 통해 공유 (대학/연구소에서 일반적)

#### Linux에서 설치

```bash
# NV5 웹사이트에서 설치 프로그램 다운로드
# 설치 프로그램 실행
chmod +x idl_installer.sh
./idl_installer.sh

# PATH에 IDL 추가 (일반적인 위치)
export IDL_DIR=/usr/local/harris/idl
export PATH=$IDL_DIR/bin:$PATH

# 영구 설정을 위해 ~/.bashrc 또는 ~/.zshrc에 추가
echo 'export IDL_DIR=/usr/local/harris/idl' >> ~/.bashrc
echo 'export PATH=$IDL_DIR/bin:$PATH' >> ~/.bashrc
```

### GDL: 무료 대안

GDL (GNU Data Language)은 대부분의 IDL 구문과 호환되는 무료 오픈소스 구현입니다. 학습 목적으로 GDL은 훌륭한 선택입니다.

#### GDL 설치

```bash
# macOS (Homebrew)
brew install gnudatalanguage

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install gnudatalanguage

# Fedora / RHEL
sudo dnf install gdl

# 소스에서 설치 (최신 기능)
git clone https://github.com/gnudatalanguage/gdl.git
cd gdl
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

---

## IDL 환경

### IDLDE (IDL 개발 환경)

IDLDE는 상용 IDL에 포함된 그래픽 개발 환경입니다:

- **편집기**: 구문 강조가 있는 코드 편집기 (여러 탭)
- **콘솔**: 하단의 대화형 명령 프롬프트
- **변수 감시**: 실행 중 변수 값 검사
- **프로젝트 탐색기**: 파일 및 디렉토리 탐색

### 명령줄 인터페이스

```bash
# IDL 대화형 세션 시작
idl

# GDL 대화형 세션 시작
gdl
```

다음과 같은 프롬프트가 표시됩니다:

```
IDL>
```

### IDL 프롬프트

IDL 프롬프트에서 명령을 대화형으로 입력할 수 있습니다:

```idl
IDL> PRINT, 'Hello, World!'
Hello, World!

IDL> x = 42
IDL> PRINT, x
      42

IDL> PRINT, x * 2 + 10
      94
```

---

## Hello, World!

가장 간단한 IDL 프로그램은 `PRINT` 프로시저를 사용합니다:

```idl
IDL> PRINT, 'Hello, World!'
Hello, World!
```

### 첫 번째 스크립트

`hello.pro` 파일을 생성합니다:

```idl
; hello.pro - 나의 첫 IDL 프로그램
PRO hello
  PRINT, 'Hello, World!'
  PRINT, 'IDL 프로그래밍에 오신 것을 환영합니다!'

  ; 기본 산술
  x = 10
  y = 20
  PRINT, 'x + y =', x + y

  ; 배열 생성
  arr = [1, 2, 3, 4, 5]
  PRINT, 'Array:', arr
  PRINT, 'Sum:', TOTAL(arr)
  PRINT, 'Mean:', MEAN(arr)
END
```

IDL 프롬프트에서 실행:

```idl
IDL> .RUN hello
% Compiled module: HELLO.
IDL> hello
```

---

## 필수 명령어

### 컴파일 및 실행 명령어

IDL은 시스템 수준 작업을 위해 점 명령어 (마침표로 시작하는 명령어)를 사용합니다:

```idl
; 파일 컴파일
IDL> .COMPILE filename.pro

; 컴파일하고 실행 (이름 없는 프로그램의 경우)
IDL> .RUN filename.pro

; 이전에 컴파일된 프로시저 실행
IDL> procedure_name
```

### 세션 관리

```idl
; 세션 초기화 — 모든 중단점이나 오류에서 메인 레벨로 복귀
IDL> RETALL

; STOP (중단점) 후 실행 계속
IDL> .CONTINUE

; 코드를 한 줄씩 실행
IDL> .STEP

; IDL 종료
IDL> EXIT
```

### 도움말 얻기

```idl
; 변수에 대한 정보 표시
IDL> x = FINDGEN(10)
IDL> HELP, x
X               FLOAT     = Array[10]

; 현재 스코프의 모든 변수 목록
IDL> HELP
```

---

## IDL 프로그램 구조

### 메인 레벨 프로그램

PRO 또는 FUNCTION 선언이 없는 코드 블록으로, `.RUN`으로 직접 실행됩니다:

```idl
; main_example.pro - 메인 레벨 프로그램
x = FINDGEN(100)
y = SIN(x / 10.0)
PLOT, x, y, TITLE='Sine Wave'
PRINT, 'Plot complete.'
END
```

### 명명된 프로시저

```idl
; greet.pro
PRO greet, name
  IF N_PARAMS() EQ 0 THEN name = 'World'
  PRINT, 'Hello, ' + name + '!'
END
```

### 명명된 함수

```idl
; add_numbers.pro
FUNCTION add_numbers, a, b
  RETURN, a + b
END
```

### 파일 명명 규칙

- **파일당 하나의 루틴**: 파일 이름은 루틴 이름과 일치해야 합니다
- **소문자에 .pro 확장자**: `my_routine.pro`는 `PRO my_routine`을 포함
- **IDL은 경로를 검색**: IDL은 `!PATH`에 있는 루틴을 자동으로 찾아서 컴파일합니다

---

## IDL 경로

```idl
; 현재 경로 보기
IDL> PRINT, !PATH

; 경로에 디렉토리 추가
IDL> !PATH = '/home/user/my_idl_code:' + !PATH

; 재귀적 디렉토리 포함을 위해 EXPAND_PATH 사용
IDL> !PATH = EXPAND_PATH('+/home/user/my_idl_code') + ':' + !PATH
```

---

## 배치 모드

IDL 스크립트를 비대화형으로 실행할 수 있습니다:

```bash
# 스크립트 실행 후 종료
idl -e "PRINT, 'Hello from batch mode'"

# .pro 파일을 배치 모드로 실행
idl < my_script.pro

# GDL 등가
gdl -e "PRINT, 'Hello from GDL batch mode'"
```

---

## IDL 시스템 변수

IDL에는 동작을 제어하는 내장 시스템 변수가 있습니다. 모두 `!`로 시작합니다:

```idl
; 수학 상수
PRINT, !PI          ; 3.14159...
PRINT, !DTOR        ; 도에서 라디안으로 변환 계수
PRINT, !RADEG       ; 라디안에서 도로 변환 계수

; 특수 값
PRINT, !VALUES.F_NAN      ; Float NaN
PRINT, !VALUES.F_INFINITY ; Float 무한대
```

---

## 요약

| 개념 | 설명 |
|------|------|
| `PRINT` | 콘솔에 값 출력 |
| `.RUN` | 파일 컴파일 및 실행 |
| `.COMPILE` | 실행 없이 컴파일 |
| `RETALL` | 메인 레벨로 복귀 |
| `.CONTINUE` | STOP 후 계속 |
| `HELP` | 변수 및 루틴 검사 |
| `EXIT` | IDL 종료 |
| `!PATH` | .pro 파일 검색 경로 |
| `@filename` | 파일을 줄 단위로 실행 |
| 시스템 변수 | `!PI`, `!DTOR`, `!VALUES` 등 |

이제 IDL 또는 GDL이 설치되었고 기본 프로그램을 작성하고 실행할 수 있습니다. 다음 레슨에서는 IDL의 데이터 타입과 변수를 자세히 살펴보겠습니다.

---

**다음**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md)
