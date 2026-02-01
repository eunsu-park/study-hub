# 운영체제 이론 학습 가이드

## 소개

이 폴더는 운영체제(Operating System) 이론을 체계적으로 학습하기 위한 자료를 담고 있습니다. 프로세스 관리부터 메모리 관리, 파일 시스템까지 운영체제의 핵심 개념을 단계별로 학습할 수 있습니다.

**대상 독자**: C/C++ 프로그래밍 경험이 있는 개발자, CS 기초를 학습하려는 사람

---

## 학습 로드맵

```
[운영체제 기초]              [CPU/동기화]                [메모리/파일]
     │                          │                           │
     ▼                          ▼                           ▼
운영체제 개요 ─────────▶ CPU 스케줄링 기초 ────▶ 메모리 관리 기초
     │                          │                           │
     ▼                          ▼                           ▼
프로세스 개념 ─────────▶ 스케줄링 알고리즘 ────▶ 가상 메모리
     │                          │                           │
     ▼                          ▼                           ▼
스레드/멀티스레딩 ──────▶ 고급 스케줄링 ────────▶ 페이지 교체
     │                          │                           │
     ▼                          ▼                           ▼
     └──────────────▶ 동기화 기초 ─────────▶ 파일 시스템
                            │                           │
                            ▼                           ▼
                       동기화 도구 ────────────▶ I/O 시스템
                            │
                            ▼
                        데드락
```

---

## 선수 지식

- **C/C++ 프로그래밍**: 포인터, 메모리 관리, 멀티스레드 기초
- **컴퓨터 구조 기초**: CPU, 메모리 계층, 인터럽트
- **기본 자료구조**: 큐, 스택, 연결 리스트
- **기본 알고리즘**: 복잡도 분석 (Big O)

---

## 파일 목록

### 운영체제 기초 (01-03)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_OS_Overview.md](./01_OS_Overview.md) | ⭐ | OS 정의, 역할, 역사, 커널 구조 |
| [02_Process_Concepts.md](./02_Process_Concepts.md) | ⭐⭐ | 프로세스 메모리 구조, PCB, 상태 전이 |
| [03_Threads_and_Multithreading.md](./03_Threads_and_Multithreading.md) | ⭐⭐ | 스레드 vs 프로세스, 멀티스레딩 모델 |

### CPU 스케줄링 (04-06)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [04_CPU_Scheduling_Basics.md](./04_CPU_Scheduling_Basics.md) | ⭐⭐ | CPU/I/O burst, 스케줄링 목표, 스케줄러 종류 |
| [05_Scheduling_Algorithms.md](./05_Scheduling_Algorithms.md) | ⭐⭐⭐ | FCFS, SJF, SRTF, Priority, RR, 간트 차트 |
| [06_Advanced_Scheduling.md](./06_Advanced_Scheduling.md) | ⭐⭐⭐ | MLFQ, 멀티프로세서 스케줄링, 실시간 스케줄링 |

### 프로세스 동기화 (07-09)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [07_Synchronization_Basics.md](./07_Synchronization_Basics.md) | ⭐⭐⭐ | 경쟁 상태, 임계 구역, Peterson's Solution |
| [08_Synchronization_Tools.md](./08_Synchronization_Tools.md) | ⭐⭐⭐ | 뮤텍스, 세마포어, 모니터, 고전 동기화 문제 |
| [09_Deadlock.md](./09_Deadlock.md) | ⭐⭐⭐ | 데드락 조건, 예방, 회피, 탐지, 은행원 알고리즘 |

### 메모리 관리 (10-13)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [10_Memory_Management_Basics.md](./10_Memory_Management_Basics.md) | ⭐⭐ | 주소 바인딩, 스와핑, 메모리 할당 개요 |
| [11_Contiguous_Memory_Allocation.md](./11_Contiguous_Memory_Allocation.md) | ⭐⭐⭐ | First-fit, Best-fit, 단편화, 압축 |
| [12_Paging.md](./12_Paging.md) | ⭐⭐⭐ | 페이지 테이블, TLB, 다단계 페이징 |
| [13_Segmentation.md](./13_Segmentation.md) | ⭐⭐⭐ | 세그먼트 테이블, 페이징과 비교 |

### 가상 메모리 (14-15)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [14_Virtual_Memory.md](./14_Virtual_Memory.md) | ⭐⭐⭐ | 요구 페이징, 페이지 폴트, 유효/무효 비트 |
| [15_Page_Replacement.md](./15_Page_Replacement.md) | ⭐⭐⭐ | FIFO, LRU, LFU, Clock, 스레싱 |

### 파일 시스템과 I/O (16-18)

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [16_File_System_Basics.md](./16_File_System_Basics.md) | ⭐⭐ | 파일 개념, 디렉토리 구조, 접근 방법 |
| [17_File_System_Implementation.md](./17_File_System_Implementation.md) | ⭐⭐⭐ | 할당 방법, FAT, inode, 저널링 |
| [18_IO_and_IPC.md](./18_IO_and_IPC.md) | ⭐⭐⭐ | I/O 하드웨어, DMA, IPC 통신 |

---

## 추천 학습 순서

### 1단계: 운영체제 기초 (1주)
```
01_OS_Overview → 02_Process_Concepts → 03_Threads_and_Multithreading
```
운영체제의 기본 개념과 프로세스/스레드의 차이를 이해합니다.

### 2단계: CPU 스케줄링 (1~2주)
```
04_CPU_Scheduling_Basics → 05_Scheduling_Algorithms → 06_Advanced_Scheduling
```
CPU 스케줄링의 목표와 다양한 알고리즘을 학습합니다.

### 3단계: 프로세스 동기화 (1~2주)
```
07_Synchronization_Basics → 08_Synchronization_Tools → 09_Deadlock
```
동시성 문제와 해결 방법을 심도 있게 학습합니다.

### 4단계: 메모리 관리 (1~2주)
```
10_Memory_Management_Basics → 11_Contiguous_Memory_Allocation → 12_Paging → 13_Segmentation
```

### 5단계: 가상 메모리 (1주)
```
14_Virtual_Memory → 15_Page_Replacement
```

### 6단계: 파일/I/O (1주)
```
16_File_System_Basics → 17_File_System_Implementation → 18_IO_and_IPC
```

---

## 실습 환경

### 필수 도구

```bash
# Linux 환경 (권장)
# Ubuntu, Fedora, 또는 macOS

# GCC 컴파일러
gcc --version
g++ --version

# pthread 라이브러리 (멀티스레드)
# Linux에서는 기본 포함

# 프로세스 모니터링
ps aux
top
htop
```

### 시스템 정보 확인

```bash
# CPU 정보
cat /proc/cpuinfo

# 메모리 정보
cat /proc/meminfo
free -h

# 프로세스 상태
cat /proc/[PID]/status
```

---

## 핵심 개념 빠른 참조

### 프로세스 상태 전이

```
         생성
          │
          ▼
       ┌─────┐  디스패치   ┌─────┐
       │준비 │───────────▶│실행 │
       │Ready│◀───────────│Run  │
       └─────┘  타임아웃   └─────┘
          ▲                   │
          │     I/O 완료      │ I/O 요청
          │                   ▼
          │              ┌─────┐
          └──────────────│대기 │
                         │Wait │
                         └─────┘
```

### 스케줄링 알고리즘 비교

| 알고리즘 | 선점 | 기아 | 특징 |
|----------|------|------|------|
| FCFS | 비선점 | 없음 | 단순, Convoy 효과 |
| SJF | 비선점 | 가능 | 최적 평균 대기시간 |
| SRTF | 선점 | 가능 | 선점형 SJF |
| Priority | 둘 다 | 가능 | Aging으로 해결 |
| RR | 선점 | 없음 | 시분할, 타임 퀀텀 중요 |
| MLFQ | 선점 | 가능 | 적응형, 실용적 |

### 동기화 도구 비교

| 도구 | 값 범위 | 사용 사례 |
|------|---------|----------|
| 뮤텍스 | 0/1 | 상호 배제 |
| 이진 세마포어 | 0/1 | 상호 배제 |
| 카운팅 세마포어 | 0~N | 자원 카운팅 |
| 모니터 | - | 고급 동기화 |

---

## 관련 자료

### 다른 폴더와의 연계

| 폴더 | 관련 내용 |
|------|----------|
| [Linux/](../Linux/00_Overview.md) | 리눅스 시스템 프로그래밍, 프로세스 관리 |
| [Computer_Architecture/](../Computer_Architecture/00_Overview.md) | CPU 구조, 메모리 계층, 인터럽트 |
| [C_Programming/](../C_Programming/00_Overview.md) | 시스템 호출, 멀티스레드 프로그래밍 |
| [Networking/](../Networking/00_Overview.md) | 소켓 프로그래밍, I/O 모델 |

### 외부 자료

- [Operating System Concepts (공룡책)](https://www.os-book.com/)
- [OSTEP (무료 온라인 교재)](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [MIT 6.828: Operating System Engineering](https://pdos.csail.mit.edu/6.828/)
- [Linux Kernel Development (Robert Love)](https://www.oreilly.com/library/view/linux-kernel-development/9780768696974/)

---

## 학습 팁

1. **실습 필수**: 코드를 직접 작성하고 실행해보기
2. **시각화**: 프로세스 상태, 스케줄링 간트 차트 직접 그리기
3. **리눅스 활용**: /proc 파일시스템으로 실제 시스템 상태 확인
4. **단계별 학습**: 기초 개념을 완전히 이해한 후 다음 단계로
5. **문제 풀이**: 각 레슨의 연습 문제 반드시 풀기

