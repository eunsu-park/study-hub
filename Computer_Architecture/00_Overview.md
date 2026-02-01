# 컴퓨터 구조 학습 가이드

## 소개

이 폴더는 컴퓨터 구조(Computer Architecture)를 체계적으로 학습하기 위한 자료를 담고 있습니다. 데이터 표현부터 CPU 아키텍처, 메모리 시스템, 병렬 처리까지 컴퓨터가 어떻게 작동하는지 이해할 수 있습니다.

**대상 독자**: 프로그래밍 기초를 아는 개발자, CS 기초를 학습하려는 사람

---

## 학습 로드맵

```
[기초]                    [중급]                    [고급]
  │                         │                         │
  ▼                         ▼                         ▼
컴퓨터 개요 ────────▶ 명령어 집합 ───────▶ 파이프라이닝
  │                         │                         │
  ▼                         ▼                         ▼
데이터 표현 ────────▶ 제어장치 ─────────▶ 캐시 메모리
  │                         │                         │
  ▼                         ▼                         ▼
논리 게이트 ────────▶ CPU 구조 ─────────▶ 가상 메모리
  │                                                   │
  ▼                                                   ▼
순차 논리 ──────────────────────────────▶ 병렬/멀티코어
```

---

## 선수 지식

- 프로그래밍 기초 (변수, 제어문, 함수)
- 기본 수학 (이진수, 논리 연산)
- C 또는 Python 중 하나 이상의 언어

---

## 파일 목록

### 기초 개념 (01-05)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Computer_System_Overview.md](./01_Computer_System_Overview.md) | ⭐ | 컴퓨터 역사, 폰 노이만 구조, 하드웨어 구성 |
| [02_Data_Representation_Basics.md](./02_Data_Representation_Basics.md) | ⭐ | 이진수, 8진수, 16진수, 진법 변환 |
| [03_Integer_Float_Representation.md](./03_Integer_Float_Representation.md) | ⭐⭐ | 2의 보수, IEEE 754 부동소수점 |
| [04_Logic_Gates.md](./04_Logic_Gates.md) | ⭐ | AND, OR, NOT, 불 대수 |
| [05_Combinational_Logic.md](./05_Combinational_Logic.md) | ⭐⭐ | 가산기, 멀티플렉서, 디코더 |

### CPU 아키텍처 (06-10)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [06_Sequential_Logic.md](./06_Sequential_Logic.md) | ⭐⭐ | 플립플롭, 레지스터, 카운터 |
| [07_CPU_Architecture_Basics.md](./07_CPU_Architecture_Basics.md) | ⭐⭐ | ALU, 레지스터 파일, 데이터패스 |
| [08_Control_Unit.md](./08_Control_Unit.md) | ⭐⭐⭐ | 하드와이어드/마이크로프로그램 제어 |
| [09_Instruction_Set_Architecture.md](./09_Instruction_Set_Architecture.md) | ⭐⭐⭐ | CISC vs RISC, 주소 지정 방식 |
| [10_Assembly_Language_Basics.md](./10_Assembly_Language_Basics.md) | ⭐⭐⭐ | x86/ARM 기초, 기본 명령어 |

### 성능 향상 기법 (11-13)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [11_Pipelining.md](./11_Pipelining.md) | ⭐⭐⭐ | 파이프라인 단계, 해저드, 포워딩 |
| [12_Branch_Prediction.md](./12_Branch_Prediction.md) | ⭐⭐⭐ | 정적/동적 분기 예측, BTB |
| [13_Superscalar_Out_of_Order.md](./13_Superscalar_Out_of_Order.md) | ⭐⭐⭐⭐ | ILP, 레지스터 리네이밍 |

### 메모리 시스템 (14-16)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [14_Memory_Hierarchy.md](./14_Memory_Hierarchy.md) | ⭐⭐ | 지역성, SRAM/DRAM, 메모리 계층 |
| [15_Cache_Memory.md](./15_Cache_Memory.md) | ⭐⭐⭐ | 직접/연관/집합연관 사상, 교체 정책 |
| [16_Virtual_Memory.md](./16_Virtual_Memory.md) | ⭐⭐⭐⭐ | 페이지 테이블, TLB, 페이지 교체 |

### 입출력 및 병렬 처리 (17-18)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [17_IO_Systems.md](./17_IO_Systems.md) | ⭐⭐⭐ | 인터럽트, DMA, 버스 |
| [18_Parallel_Processing_Multicore.md](./18_Parallel_Processing_Multicore.md) | ⭐⭐⭐⭐ | SIMD/MIMD, 캐시 일관성, Amdahl |

---

## 추천 학습 순서

### 1단계: 기초 개념 (1주)
```
01_Computer_System_Overview → 02_Data_Representation_Basics → 03_Integer_Float_Representation
```

### 2단계: 디지털 논리 (1주)
```
04_Logic_Gates → 05_Combinational_Logic → 06_Sequential_Logic
```

### 3단계: CPU 아키텍처 (2주)
```
07_CPU_Architecture_Basics → 08_Control_Unit → 09_Instruction_Set_Architecture → 10_Assembly_Language_Basics
```

### 4단계: 성능 향상 (1~2주)
```
11_Pipelining → 12_Branch_Prediction → 13_Superscalar_Out_of_Order
```

### 5단계: 메모리 시스템 (1~2주)
```
14_Memory_Hierarchy → 15_Cache_Memory → 16_Virtual_Memory
```

### 6단계: 입출력 및 병렬 (1주)
```
17_IO_Systems → 18_Parallel_Processing_Multicore
```

---

## 실습 환경

### 시뮬레이터

```bash
# MARS (MIPS 시뮬레이터)
# https://courses.missouristate.edu/kenvollmar/mars/

# Logisim (논리 회로 시뮬레이터)
# https://www.cburch.com/logisim/

# CPU 시뮬레이터
# https://cpuvisualsimulator.github.io/
```

### 어셈블리 실습

```bash
# x86 (Linux)
nasm -f elf64 hello.asm -o hello.o
ld hello.o -o hello

# GCC 어셈블리 출력
gcc -S -O0 program.c -o program.s
```

---

## 복잡도 빠른 참조

| 구성 요소 | 일반적인 지연 시간 |
|-----------|-------------------|
| 레지스터 접근 | ~1 사이클 |
| L1 캐시 | ~4 사이클 |
| L2 캐시 | ~10 사이클 |
| L3 캐시 | ~40 사이클 |
| 메인 메모리 | ~100+ 사이클 |
| SSD | ~10,000+ 사이클 |
| HDD | ~10,000,000+ 사이클 |

---

## 관련 자료

### 다른 폴더와의 연계

| 폴더 | 관련 내용 |
|------|----------|
| [C_Programming/](../C_Programming/00_Overview.md) | 포인터, 메모리 관리 |
| [Algorithm/](../Algorithm/00_Overview.md) | 복잡도 분석, 캐시 최적화 |
| [Linux/](../Linux/00_Overview.md) | 프로세스, 메모리 관리 |

### 외부 자료

- [Computer Organization and Design (Patterson & Hennessy)](https://www.elsevier.com/books/computer-organization-and-design/)
- [Nand2Tetris](https://www.nand2tetris.org/)
- [CPU 시각화](https://www.youtube.com/watch?v=cNN_tTXABUA)

---

## 학습 팁

1. **시뮬레이터 활용**: Logisim으로 논리 회로 직접 구현
2. **어셈블리 실습**: 간단한 프로그램을 어셈블리로 작성
3. **캐시 분석**: perf나 cachegrind로 캐시 미스 분석
4. **단계별 이해**: 각 레슨의 연습 문제 반드시 풀기
5. **시각화**: 파이프라인, 캐시 동작 등 그림으로 이해

