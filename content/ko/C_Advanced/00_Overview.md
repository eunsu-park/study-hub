# C 고급

## 소개

이 토픽은 고급 C 프로그래밍을 다룹니다: 포인터 마스터, 시스템 프로그래밍, 자료 구조, 동시성, 크로스 플랫폼 개발.

**선수과목**: [C_Basics](../C_Basics/00_Overview.md) (포인터, 구조체, 동적 메모리를 포함한 C 기초 지식 또는 동등한 수준)

---

## 학습 로드맵

```
[포인터 & 자료 구조]                [시스템 프로그래밍]              [도구 & 플랫폼]
  |                                    |                              |
  v                                    v                              v
고급 포인터 ----------+          프로세스 관리 ----+           임베디드 시스템
  |                   |            |                |              |
  v                   |            v                |              v
메모리 관리           |          미니 셸            |           디버깅 & 프로파일링
  |                   |            |                |              |
  v                   |            v                |              v
동적 배열             |          멀티스레딩         |           크로스 플랫폼 개발
  |                   |            |                |              |
  v                   |            v                |              v
연결 리스트           |          네트워크 프로그래밍 |           스네이크 게임 (종합)
  |                   |            |                |
  v                   |            v                |
스택 & 큐            |          IPC & 시그널 ------+
  |                   |
  v                   |
해시 테이블           |
  |                   |
  v                   |
파일 암호화 ----------+
```

---

## 파일 목록

| # | 제목 | 난이도 | 핵심 내용 |
|---|------|--------|----------|
| [01](./01_Advanced_Pointers.md) | 고급 포인터 | ⭐⭐⭐ | 함수 포인터, void*, 포인터 배열, const 정확성 |
| [02](./02_Advanced_Memory_Management.md) | 고급 메모리 관리 | ⭐⭐⭐ | 메모리 레이아웃, mmap, 커스텀 할당자, 메모리 풀 |
| [03](./03_Bit_Operations.md) | 비트 연산 | ⭐⭐ | 비트 연산자, 비트 마스킹, 레지스터 조작 |
| [04](./04_Project_Dynamic_Array.md) | 프로젝트: 동적 배열 | ⭐⭐ | malloc/realloc, 가변 배열, 분할 상환 비용 |
| [05](./05_Project_Linked_List.md) | 프로젝트: 연결 리스트 | ⭐⭐⭐ | 단일/이중 연결, 삽입, 삭제, 역전 |
| [06](./06_Project_Stack_Queue.md) | 프로젝트: 스택과 큐 | ⭐⭐ | LIFO/FIFO, 배열 및 연결 리스트 구현 |
| [07](./07_Project_Hash_Table.md) | 프로젝트: 해시 테이블 | ⭐⭐⭐ | 해시 함수, 체이닝, 개방 주소법 |
| [08](./08_Project_File_Encryption.md) | 프로젝트: 파일 암호화 | ⭐⭐ | XOR 암호, 바이트 단위 파일 처리 |
| [09](./09_Process_Management.md) | 프로세스 관리 | ⭐⭐⭐ | fork, exec, wait, 프로세스 생명주기 |
| [10](./10_Project_Mini_Shell.md) | 프로젝트: 미니 셸 | ⭐⭐⭐⭐ | 셸 구현, 파이프, 리다이렉션 |
| [11](./11_Multithreading.md) | 멀티스레딩 | ⭐⭐⭐⭐ | pthreads, 뮤텍스, 조건 변수, 스레드 풀 |
| [12](./12_Network_Programming.md) | 네트워크 프로그래밍 | ⭐⭐⭐⭐ | TCP/UDP 소켓, 클라이언트-서버, select/poll |
| [13](./13_IPC_and_Signals.md) | IPC와 시그널 | ⭐⭐⭐⭐ | 파이프, 공유 메모리, 메시지 큐, 시그널 |
| [14](./14_Embedded_Systems.md) | 임베디드 시스템 | ⭐⭐⭐ | GPIO, 시리얼, I2C/SPI, volatile, 레지스터 접근 |
| [15](./15_Debugging_and_Profiling.md) | 디버깅과 프로파일링 | ⭐⭐⭐ | GDB 고급, Valgrind, ASan, gprof, 단위 테스트 |
| [16](./16_Cross_Platform_Development.md) | 크로스 플랫폼 개발 | ⭐⭐⭐ | 이식성, CMake, 플랫폼 추상화 |
| [17](./17_Project_Snake_Game.md) | 프로젝트: 스네이크 게임 | ⭐⭐⭐ | 터미널 제어, 게임 루프, ncurses |

---

## 추천 학습 순서

### 경로 1: 포인터 & 자료 구조
1. 고급 포인터 -> 메모리 관리 -> 동적 배열 -> 연결 리스트 -> 스택 & 큐 -> 해시 테이블 -> 파일 암호화

### 경로 2: 시스템 프로그래밍
2. 프로세스 관리 -> 미니 셸 -> 멀티스레딩 -> 네트워크 프로그래밍 -> IPC & 시그널

### 경로 3: 도구 & 플랫폼
3. 임베디드 시스템 -> 디버깅 & 프로파일링 -> 크로스 플랫폼 개발 -> 스네이크 게임 (종합)

---

## 실습 환경

```bash
# GCC 버전 확인 (C11 지원 필요)
gcc --version

# 경고 및 디버그 정보 포함하여 컴파일
gcc -Wall -Wextra -std=c11 -g program.c -o program

# Valgrind로 실행 (Linux/macOS)
valgrind --leak-check=full ./program

# AddressSanitizer로 컴파일
gcc -fsanitize=address -g program.c -o program
```

---

## 관련 자료

- [C_Basics/](../C_Basics/00_Overview.md) - C 기초 (변수, 제어 흐름, 함수, 기본 포인터)
- [Linux/](../Linux/00_Overview.md) - 리눅스 환경과 셸 스크립팅
- [OS_Theory/](../OS_Theory/00_Overview.md) - 운영체제 개념 (프로세스, 메모리, 스케줄링)
- [Computer_Architecture/](../Computer_Architecture/00_Overview.md) - 하드웨어 기초
- [Algorithm/](../Algorithm/00_Overview.md) - 자료 구조와 알고리즘
- [Networking/](../Networking/00_Overview.md) - 네트워크 프로토콜과 아키텍처
