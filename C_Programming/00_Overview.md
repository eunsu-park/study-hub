# C 프로그래밍 학습 가이드

## 소개

이 폴더는 C 프로그래밍을 체계적으로 학습하기 위한 자료를 담고 있습니다. 기초 문법부터 임베디드 시스템까지, 실습 프로젝트를 통해 단계별로 학습할 수 있습니다.

**대상 독자**: 프로그래밍 입문자 ~ 중급자

---

## 학습 로드맵

```
[기초]           [중급]              [고급]           [임베디드]
  │                │                   │                  │
  ▼                ▼                   ▼                  ▼
환경설정 ──▶ 동적배열 ──────▶ 뱀게임 ──────▶ 임베디드 기초
  │           │                   │              │
  ▼           ▼                   ▼              ▼
기초복습 ──▶ 연결리스트 ──▶ 미니쉘 ────▶ 비트연산 심화
  │           │                   │              │
  ▼           ▼                   ▼              ▼
계산기 ────▶ 파일암호화 ──▶ 멀티스레드 ─▶ GPIO 제어
  │           │                                  │
  ▼           ▼                                  ▼
숫자맞추기 ─▶ 스택과큐                      시리얼통신
  │           │
  ▼           ▼
주소록 ────▶ 해시테이블
```

---

## 선수 지식

- 기본적인 컴퓨터 사용법
- 터미널/명령줄 사용 경험 (권장)
- 텍스트 에디터 또는 IDE 사용법

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Environment_Setup.md](./01_Environment_Setup.md) | ⭐ | 개발 환경 구축, 컴파일러 설치 |
| [02_C_Basics_Review.md](./02_C_Basics_Review.md) | ⭐ | 변수, 자료형, 연산자, 제어문, 함수 |
| [03_Project_Calculator.md](./03_Project_Calculator.md) | ⭐ | 함수, switch-case, scanf |
| [04_Project_Number_Guessing.md](./04_Project_Number_Guessing.md) | ⭐ | 반복문, 랜덤, 조건문 |
| [05_Project_Address_Book.md](./05_Project_Address_Book.md) | ⭐⭐ | 구조체, 배열, 파일 I/O |
| [06_Project_Dynamic_Array.md](./06_Project_Dynamic_Array.md) | ⭐⭐ | malloc, realloc, free |
| [07_Project_Linked_List.md](./07_Project_Linked_List.md) | ⭐⭐⭐ | 포인터, 동적 자료구조 |
| [08_Project_File_Encryption.md](./08_Project_File_Encryption.md) | ⭐⭐ | 파일 처리, 비트 연산 |
| [09_Project_Stack_Queue.md](./09_Project_Stack_Queue.md) | ⭐⭐ | 자료구조, LIFO/FIFO |
| [10_Project_Hash_Table.md](./10_Project_Hash_Table.md) | ⭐⭐⭐ | 해싱, 충돌 처리 |
| [11_Project_Snake_Game.md](./11_Project_Snake_Game.md) | ⭐⭐⭐ | 터미널 제어, 게임 루프 |
| [12_Project_Mini_Shell.md](./12_Project_Mini_Shell.md) | ⭐⭐⭐⭐ | fork, exec, 파이프 |
| [13_Project_Multithreading.md](./13_Project_Multithreading.md) | ⭐⭐⭐⭐ | pthread, 동기화 |
| [14_Embedded_Basics.md](./14_Embedded_Basics.md) | ⭐ | Arduino, GPIO 기초 |
| [15_Bit_Operations.md](./15_Bit_Operations.md) | ⭐⭐ | 비트 마스킹, 레지스터 |
| [16_Project_GPIO_Control.md](./16_Project_GPIO_Control.md) | ⭐⭐ | LED, 버튼, 디바운싱 |
| [17_Project_Serial_Communication.md](./17_Project_Serial_Communication.md) | ⭐⭐ | UART, 명령어 파싱 |
| [18_Debugging_Memory_Analysis.md](./18_Debugging_Memory_Analysis.md) | ⭐⭐⭐ | GDB, Valgrind, AddressSanitizer |
| [19_Advanced_Embedded_Protocols.md](./19_Advanced_Embedded_Protocols.md) | ⭐⭐⭐ | PWM, I2C, SPI, ADC |
| [20_Advanced_Pointers.md](./20_Advanced_Pointers.md) | ⭐⭐⭐ | 포인터 산술, 다중 포인터, 함수 포인터, 동적 메모리 |

---

## 추천 학습 순서

### 초급 (C 입문)
1. 환경설정 → 기초 복습 → 계산기 → 숫자맞추기 → 주소록

### 중급 (자료구조 & 포인터)
2. 포인터 심화 → 동적배열 → 연결리스트 → 파일암호화 → 스택과큐 → 해시테이블

### 고급 (시스템 프로그래밍)
3. 뱀게임 → 미니쉘 → 멀티스레드

### 임베디드 (Arduino)
4. 임베디드 기초 → 비트연산 심화 → GPIO 제어 → 시리얼통신 → 고급임베디드 프로토콜

### 디버깅 (선택)
5. 디버깅과 메모리분석 (모든 과정 완료 후 권장)

---

## 실습 코드

예제 코드는 [examples/](./examples/) 폴더에서 확인할 수 있습니다.

```bash
# 전체 빌드
cd examples
make

# 개별 실행
make run-calculator
make run-list
make run-thread
```

---

## 관련 자료

- [Docker 학습](../Docker/00_Overview.md) - 개발 환경 컨테이너화
- [Git 학습](../Git/00_Overview.md) - 버전 관리
