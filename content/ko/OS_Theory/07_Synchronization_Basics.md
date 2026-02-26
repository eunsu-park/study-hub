# 동기화 기초

**이전**: [고급 스케줄링](./06_Advanced_Scheduling.md) | **다음**: [동기화 도구](./08_Synchronization_Tools.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 경쟁 상태(Race Condition)를 정의하고 동시 프로그램에서 발생하는 이유를 설명할 수 있습니다
2. 임계 구역(Critical Section) 해결 조건 세 가지(상호 배제, 진행, 한정 대기)를 식별할 수 있습니다
3. Peterson's 알고리즘을 단계별로 추적하고 `flag[]`와 `turn` 두 변수가 모두 필요한 이유를 설명할 수 있습니다
4. Test-and-Set과 Compare-and-Swap 하드웨어 명령어를 설명할 수 있습니다
5. 바쁜 대기(Busy Waiting)와 블로킹 동기화(Blocking Synchronization)를 구별할 수 있습니다

---

동시성 버그는 타이밍에 의존하기 때문에 찾고 재현하기가 가장 어렵습니다. 경쟁 상태(Race Condition)는 백만 번 실행 중 한 번 프로그램을 충돌시킬 수 있지만, 그 한 번이 데이터베이스를 손상시키거나 보안 침해를 일으킬 수 있습니다. 동기화를 마스터하는 것이 "대부분의 경우에" 작동하는 프로그램과 증명 가능하게 올바른 프로그램을 구분합니다.

## 목차

1. [경쟁 상태 (Race Condition)](#1-경쟁-상태-race-condition)
2. [임계 구역 문제](#2-임계-구역-문제)
3. [임계 구역 해결 조건](#3-임계-구역-해결-조건)
4. [Peterson's Solution](#4-petersons-solution)
5. [하드웨어 지원](#5-하드웨어-지원)
6. [연습 문제](#6-연습-문제)

---

## 1. 경쟁 상태 (Race Condition)

> 두 사람이 반대편에서 동시에 회전문을 통과하려 한다고 상상해보세요. 조율 없이는 문이 막혀 누구도 통과하지 못합니다. 이것이 경쟁 상태(Race Condition)의 본질입니다 -- 여러 주체가 적절한 동기화 없이 공유 자원에 접근하여 예측할 수 없는 결과를 초래합니다.

### 정의

```
경쟁 상태 (Race Condition)
= 여러 프로세스/스레드가 공유 데이터에 동시 접근할 때
  실행 순서에 따라 결과가 달라지는 상황

┌─────────────────────────────────────────────────────────┐
│                    경쟁 상태 예시                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  공유 변수: counter = 5                                 │
│                                                         │
│  스레드 1: counter++                                   │
│  스레드 2: counter++                                   │
│                                                         │
│  기대 결과: counter = 7                                │
│  실제 결과: counter = 6 (또는 7, 비결정적)              │
│                                                         │
│  왜?                                                    │
│  counter++는 원자적(atomic)이지 않음                    │
│  1. 레지스터 = counter (읽기)                          │
│  2. 레지스터 = 레지스터 + 1 (증가)                     │
│  3. counter = 레지스터 (쓰기)                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 경쟁 상태 발생 과정

```
┌─────────────────────────────────────────────────────────┐
│              counter++ 경쟁 상태 상세                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  초기값: counter = 5                                   │
│                                                         │
│  스레드 1                 스레드 2                      │
│  ─────────                ─────────                    │
│  R1 = counter (R1=5)                                   │
│                          R2 = counter (R2=5)           │
│  R1 = R1 + 1 (R1=6)                                    │
│                          R2 = R2 + 1 (R2=6)            │
│  counter = R1 (counter=6)                              │
│                          counter = R2 (counter=6)      │
│                                                         │
│  최종 counter = 6 (7이 아님!)                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 시간축:                                          │  │
│  │                                                  │  │
│  │  T1: ─읽기─────증가─────쓰기──                   │  │
│  │  T2: ────읽기─────증가──────쓰기──               │  │
│  │            ↑                                     │  │
│  │     아직 T1이 쓰기 전에 읽음                      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### C 코드 예제

```c
#include <stdio.h>
#include <pthread.h>

int counter = 0;  // 공유 변수

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        counter++;  // 경쟁 상태!
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // 예상: 2000000
    // 실제: 그보다 작은 값 (매 실행마다 다름)
    printf("Counter: %d\n", counter);

    return 0;
}

/*
실행 결과 예시:
$ ./race_condition
Counter: 1523847
$ ./race_condition
Counter: 1678234
$ ./race_condition
Counter: 1432156
*/
```

### 은행 계좌 예제

```
┌─────────────────────────────────────────────────────────┐
│                 은행 계좌 경쟁 상태                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  계좌 잔액: balance = 1000                              │
│                                                         │
│  스레드 1 (인출 200)       스레드 2 (입금 500)          │
│  ───────────────────       ───────────────────          │
│  temp = balance (1000)                                  │
│                            temp = balance (1000)        │
│  temp = temp - 200 (800)                                │
│                            temp = temp + 500 (1500)     │
│  balance = temp (800)                                   │
│                            balance = temp (1500)        │
│                                                         │
│  최종: balance = 1500                                  │
│  올바른 결과: 1000 - 200 + 500 = 1300                  │
│                                                         │
│  200원이 사라짐!                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
// 은행 계좌 경쟁 상태 코드
#include <stdio.h>
#include <pthread.h>

int balance = 1000;

void* withdraw(void* arg) {
    int amount = *(int*)arg;
    int temp = balance;
    // 컨텍스트 스위치 발생 가능 지점
    temp = temp - amount;
    balance = temp;
    return NULL;
}

void* deposit(void* arg) {
    int amount = *(int*)arg;
    int temp = balance;
    // 컨텍스트 스위치 발생 가능 지점
    temp = temp + amount;
    balance = temp;
    return NULL;
}

int main() {
    pthread_t t1, t2;
    int withdraw_amount = 200;
    int deposit_amount = 500;

    pthread_create(&t1, NULL, withdraw, &withdraw_amount);
    pthread_create(&t2, NULL, deposit, &deposit_amount);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Final balance: %d\n", balance);
    // 예상: 1300, 실제: 800 또는 1500 또는 1300

    return 0;
}
```

---

## 2. 임계 구역 문제

### 임계 구역 정의

```
┌─────────────────────────────────────────────────────────┐
│                    임계 구역 개념                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  임계 구역 (Critical Section)                           │
│  = 공유 자원에 접근하는 코드 영역                        │
│  = 한 번에 하나의 프로세스만 실행해야 하는 구역          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                프로세스 구조                     │    │
│  │                                                 │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         진입 구역 (Entry Section)       │   │    │
│  │  │         - 임계 구역 진입 허가 요청       │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                      │                         │    │
│  │                      ▼                         │    │
│  │  ╔═════════════════════════════════════════╗   │    │
│  │  ║         임계 구역 (Critical Section)     ║   │    │
│  │  ║         - 공유 자원 접근 코드            ║   │    │
│  │  ╚═════════════════════════════════════════╝   │    │
│  │                      │                         │    │
│  │                      ▼                         │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         퇴출 구역 (Exit Section)        │   │    │
│  │  │         - 임계 구역 퇴장 알림            │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                      │                         │    │
│  │                      ▼                         │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         나머지 구역 (Remainder Section) │   │    │
│  │  │         - 공유 자원 미사용 코드          │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 코드 구조

```c
// 일반적인 임계 구역 구조
while (true) {
    // 진입 구역 (Entry Section)
    // 임계 구역 진입 허가 획득

    // ===== 임계 구역 (Critical Section) =====
    // 공유 자원에 접근하는 코드
    counter++;
    // ==========================================

    // 퇴출 구역 (Exit Section)
    // 임계 구역 사용 완료 알림

    // 나머지 구역 (Remainder Section)
    // 공유 자원을 사용하지 않는 코드
}
```

---

## 3. 임계 구역 해결 조건

### 세 가지 필수 조건

```
┌─────────────────────────────────────────────────────────┐
│              임계 구역 해결을 위한 세 조건               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 상호 배제 (Mutual Exclusion)                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  하나의 프로세스가 임계 구역에 있으면               │  │
│  │  다른 프로세스는 들어올 수 없음                     │  │
│  │                                                   │  │
│  │  ┌──────────────────────────────────────────────┐ │  │
│  │  │                                              │ │  │
│  │  │  P1:  ╔════ 임계 구역 ════╗                  │ │  │
│  │  │  P2:  ═══대기═══▶│       │                  │ │  │
│  │  │                  │       │                  │ │  │
│  │  │                  ╚═══════╝                  │ │  │
│  │  │                                              │ │  │
│  │  └──────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. 진행 (Progress)                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  임계 구역이 비어있고, 진입하려는 프로세스가 있으면  │  │
│  │  진입할 프로세스를 결정해야 하며                    │  │
│  │  이 결정은 무한히 연기될 수 없음                    │  │
│  │                                                   │  │
│  │  → 무한정 진입 거부 금지                          │  │
│  │  → 나머지 구역에 있는 프로세스가 결정에 참여 금지   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. 한정 대기 (Bounded Waiting)                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  프로세스가 임계 구역 진입을 요청한 후              │  │
│  │  다른 프로세스가 진입할 수 있는 횟수에 제한이 있음   │  │
│  │                                                   │  │
│  │  → 기아(Starvation) 방지                          │  │
│  │  → 언젠가는 반드시 진입할 수 있음을 보장            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 조건 위반 예시

```
┌─────────────────────────────────────────────────────────┐
│                    조건 위반 예시                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 상호 배제 위반:                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  P1과 P2가 동시에 임계 구역에 있음                  │  │
│  │  → 경쟁 상태 발생                                  │  │
│  │                                                   │  │
│  │  P1:  ╔══════════════════╗                        │  │
│  │  P2:       ╔══════════════════╗  ← 동시 진입!     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. 진행 위반:                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  임계 구역이 비어있는데 아무도 못 들어감            │  │
│  │                                                   │  │
│  │  P1: 대기 중 ────────────────▶                    │  │
│  │  P2: 대기 중 ────────────────▶                    │  │
│  │  임계 구역: [비어있음]  ← 둘 다 진입 못함!         │  │
│  │                                                   │  │
│  │  예: 잘못된 턴 변수 사용                           │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. 한정 대기 위반:                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  P1이 계속 임계 구역에 들어가고 P2는 영원히 대기    │  │
│  │                                                   │  │
│  │  P1: 임계구역 → 임계구역 → 임계구역 → ...        │  │
│  │  P2: 대기 ────────────────────────▶ (기아)       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Peterson's Solution

### 알고리즘 설명

```
┌─────────────────────────────────────────────────────────┐
│                 Peterson's Solution                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  두 프로세스 간의 상호 배제를 소프트웨어로 해결          │
│                                                         │
│  공유 변수:                                             │
│  • flag[2]: 각 프로세스의 임계 구역 진입 의사           │
│  • turn: 누구 차례인지 표시                            │
│                                                         │
│  핵심 아이디어:                                         │
│  • 진입하고 싶다고 표시 (flag[i] = true)               │
│  • 상대방에게 양보 (turn = j)                          │
│  • 상대방이 진입 의사가 있고 상대방 차례면 대기         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 구현 코드

```c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

// 공유 변수
volatile bool flag[2] = {false, false};
volatile int turn = 0;

int shared_counter = 0;

void* process_0(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // 진입 구역 (Entry Section)
        flag[0] = true;      // 진입 의사 표시
        turn = 1;            // 상대방에게 양보
        while (flag[1] && turn == 1) {
            // 바쁜 대기 (Busy Waiting)
            // 상대방이 임계 구역에 있고, 상대방 차례면 대기
        }

        // 임계 구역 (Critical Section)
        shared_counter++;

        // 퇴출 구역 (Exit Section)
        flag[0] = false;     // 임계 구역 퇴장

        // 나머지 구역 (Remainder Section)
    }
    return NULL;
}

void* process_1(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // 진입 구역 (Entry Section)
        flag[1] = true;      // 진입 의사 표시
        turn = 0;            // 상대방에게 양보
        while (flag[0] && turn == 0) {
            // 바쁜 대기
        }

        // 임계 구역 (Critical Section)
        shared_counter++;

        // 퇴출 구역 (Exit Section)
        flag[1] = false;

        // 나머지 구역
    }
    return NULL;
}

int main() {
    pthread_t t0, t1;

    pthread_create(&t0, NULL, process_0, NULL);
    pthread_create(&t1, NULL, process_1, NULL);

    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    printf("Counter: %d\n", shared_counter);  // 2000000
    return 0;
}
```

### 정확성 증명

```
┌─────────────────────────────────────────────────────────┐
│              Peterson's Solution 정확성                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 상호 배제 (Mutual Exclusion) ✓                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  P0가 임계 구역에 있으려면:                        │  │
│  │    flag[1] = false 또는 turn = 0                  │  │
│  │                                                   │  │
│  │  P1가 임계 구역에 있으려면:                        │  │
│  │    flag[0] = false 또는 turn = 1                  │  │
│  │                                                   │  │
│  │  둘 다 임계 구역에 있으려면:                       │  │
│  │    turn = 0 AND turn = 1 (불가능!)                │  │
│  │                                                   │  │
│  │  → 상호 배제 보장                                 │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. 진행 (Progress) ✓                                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  P0만 진입하려는 경우:                             │  │
│  │    flag[0] = true, turn = 1, flag[1] = false     │  │
│  │    while 조건: flag[1](false) && turn==1         │  │
│  │    → while 탈출, 진입 가능                        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. 한정 대기 (Bounded Waiting) ✓                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  P0가 대기 중이고 P1이 임계 구역 사용 후:          │  │
│  │    P1이 다시 진입하려면 turn = 0으로 설정          │  │
│  │    → P0가 진입 (turn == 0이므로 P0 우선)          │  │
│  │                                                   │  │
│  │  최대 1번 대기 후 진입 보장                        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Peterson's Solution 한계

```
┌─────────────────────────────────────────────────────────┐
│            Peterson's Solution 한계                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 두 프로세스에만 적용 가능                           │
│     → n개 프로세스로 확장하려면 복잡해짐                │
│                                                         │
│  2. 바쁜 대기 (Busy Waiting)                            │
│     → CPU 시간 낭비                                    │
│     → 스핀락(Spinlock)이라고도 함                      │
│                                                         │
│  3. 현대 CPU에서 작동 보장 안 됨                        │
│     → 컴파일러/CPU 명령어 재배치 가능                   │
│     → 메모리 배리어(Memory Barrier) 필요               │
│                                                         │
│  4. 성능 문제                                           │
│     → 멀티코어에서 캐시 일관성 오버헤드                 │
│                                                         │
│  현대 시스템에서는:                                     │
│  • 하드웨어 지원 (원자적 명령어)                        │
│  • 운영체제 제공 동기화 도구 (뮤텍스, 세마포어)          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 하드웨어 지원

### Test-and-Set (TAS)

```
┌─────────────────────────────────────────────────────────┐
│                    Test-and-Set                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  원자적으로 수행되는 하드웨어 명령어:                    │
│  1. 현재 값을 읽고                                      │
│  2. 새 값(보통 true)으로 설정                          │
│                                                         │
│  의사 코드:                                             │
│  ```                                                   │
│  bool test_and_set(bool *target) {                     │
│      bool rv = *target;    // 현재 값 읽기             │
│      *target = true;       // true로 설정              │
│      return rv;            // 이전 값 반환             │
│  }                                                     │
│  // 이 전체가 원자적으로 실행됨 (인터럽트 불가)          │
│  ```                                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### TAS를 이용한 상호 배제

```c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>

// 원자적 불리언 락
atomic_bool lock = false;

int shared_counter = 0;

// test_and_set 구현 (실제로는 하드웨어 명령어)
bool test_and_set(atomic_bool *target) {
    // C11 atomic: atomic_exchange와 동일
    return atomic_exchange(target, true);
}

void* critical_section(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // 진입 구역: 락 획득 시도
        while (test_and_set(&lock)) {
            // 바쁜 대기 (스핀)
            // 락이 이미 true면 계속 대기
        }

        // 임계 구역
        shared_counter++;

        // 퇴출 구역: 락 해제
        lock = false;
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, critical_section, NULL);
    pthread_create(&t2, NULL, critical_section, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", shared_counter);  // 2000000
    return 0;
}
```

### Compare-and-Swap (CAS)

```
┌─────────────────────────────────────────────────────────┐
│                  Compare-and-Swap (CAS)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  원자적으로 수행되는 하드웨어 명령어:                    │
│  1. 현재 값을 기대 값과 비교                            │
│  2. 같으면 새 값으로 교체                               │
│  3. 이전 값 반환                                        │
│                                                         │
│  의사 코드:                                             │
│  ```                                                   │
│  bool compare_and_swap(int *word, int expected, int new_val) { │
│      int temp = *word;                                 │
│      if (temp == expected) {                           │
│          *word = new_val;                              │
│          return true;                                  │
│      }                                                 │
│      return false;                                     │
│  }                                                     │
│  // 이 전체가 원자적으로 실행됨                         │
│  ```                                                   │
│                                                         │
│  x86: CMPXCHG 명령어                                   │
│  ARM: LDREX/STREX 명령어 조합                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### CAS를 이용한 상호 배제

```c
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int lock = 0;  // 0: 사용 가능, 1: 사용 중

int shared_counter = 0;

void* critical_section(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // 진입 구역: CAS로 락 획득 시도
        int expected = 0;
        while (!atomic_compare_exchange_weak(&lock, &expected, 1)) {
            // CAS 실패 시 expected가 현재 lock 값으로 업데이트됨
            expected = 0;  // 다시 0으로 리셋하고 재시도
            // 바쁜 대기
        }

        // 임계 구역
        shared_counter++;

        // 퇴출 구역: 락 해제
        atomic_store(&lock, 0);
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, critical_section, NULL);
    pthread_create(&t2, NULL, critical_section, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", shared_counter);  // 2000000
    return 0;
}
```

### CAS를 이용한 락-프리 카운터

```c
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        int old_val, new_val;
        do {
            old_val = atomic_load(&counter);
            new_val = old_val + 1;
        } while (!atomic_compare_exchange_weak(&counter, &old_val, new_val));
        // CAS 성공할 때까지 반복
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", counter);  // 2000000
    return 0;
}
```

### 하드웨어 명령어 비교

```
┌──────────────────┬─────────────────────────┬─────────────────────────┐
│      특성         │      Test-and-Set      │    Compare-and-Swap    │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ 반환값           │ 이전 값                 │ 성공/실패 + 현재값      │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ 조건부 변경      │ 불가능 (항상 설정)       │ 가능 (일치 시만 변경)   │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ 활용             │ 스핀락                  │ 락-프리 알고리즘        │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ ABA 문제         │ 해당 없음               │ 발생 가능               │
├──────────────────┼─────────────────────────┼─────────────────────────┤
│ 복잡도           │ 단순                    │ 약간 복잡               │
└──────────────────┴─────────────────────────┴─────────────────────────┘

ABA 문제:
A → B → A로 값이 바뀌어도 CAS는 변화를 감지 못함
해결: 버전 번호 추가 또는 더블-워드 CAS 사용
```

---

## 6. 연습 문제

### 문제 1: 경쟁 상태 식별

다음 코드에서 경쟁 상태가 발생할 수 있는 부분을 찾고 설명하세요.

```c
int balance = 1000;

void transfer(int from, int to, int amount) {
    if (balance >= amount) {
        balance = balance - amount;
        // ... 이체 처리 ...
    }
}
```

<details>
<summary>정답 보기</summary>

**경쟁 상태 위치:**

`if (balance >= amount)`와 `balance = balance - amount` 사이에서 경쟁 상태 발생 가능

**시나리오:**
- 잔액: 1000, 두 스레드가 각각 700원 이체 시도

```
스레드1: if (1000 >= 700) → true
스레드2: if (1000 >= 700) → true (아직 감소 안 됨)
스레드1: balance = 1000 - 700 = 300
스레드2: balance = 1000 - 700 = 300 (잘못된 연산!)
```

결과: 1400원이 이체됨 (잔액 -400원이 되어야 하지만 300원)

**해결:** 임계 구역 보호 필요 (뮤텍스 등)

</details>

### 문제 2: 임계 구역 조건

다음 중 임계 구역 해결 조건에 해당하지 않는 것은?

A. 상호 배제 (Mutual Exclusion)
B. 진행 (Progress)
C. 한정 대기 (Bounded Waiting)
D. 공정성 (Fairness)

<details>
<summary>정답 보기</summary>

**정답: D. 공정성**

임계 구역 문제의 세 가지 필수 조건:
1. 상호 배제 - 한 번에 하나만
2. 진행 - 무한정 대기 금지
3. 한정 대기 - 유한 횟수 내 진입 보장

공정성은 바람직하지만 필수 조건은 아님.

</details>

### 문제 3: Peterson's Solution 분석

Peterson's Solution에서 두 프로세스가 동시에 임계 구역에 진입할 수 없는 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

**핵심 논리:**

P0가 임계 구역에 있으려면:
- `flag[1] == false` OR `turn == 0`

P1이 임계 구역에 있으려면:
- `flag[0] == false` OR `turn == 1`

두 프로세스가 동시에 진입하려면 위 두 조건이 모두 참이어야 함.

그러나:
- 둘 다 진입하려면 `flag[0] == true` AND `flag[1] == true`
- 따라서 두 번째 조건(turn)이 결정적
- `turn`은 0 또는 1 중 하나만 가능
- `turn == 0 AND turn == 1`은 불가능!

따라서 최대 하나의 프로세스만 임계 구역에 진입 가능.

</details>

### 문제 4: TAS 구현

Test-and-Set을 사용하여 다음 요구사항을 만족하는 락을 구현하세요:
- `lock()`: 락 획득
- `unlock()`: 락 해제
- 여러 스레드에서 안전하게 사용 가능

<details>
<summary>정답 보기</summary>

```c
#include <stdatomic.h>
#include <stdbool.h>

typedef struct {
    atomic_bool locked;
} spinlock_t;

void spinlock_init(spinlock_t *lock) {
    atomic_store(&lock->locked, false);
}

void lock(spinlock_t *lock) {
    while (atomic_exchange(&lock->locked, true)) {
        // 스핀 (바쁜 대기)
        // 선택적: CPU 양보
        // sched_yield();
    }
}

void unlock(spinlock_t *lock) {
    atomic_store(&lock->locked, false);
}

// 사용 예:
spinlock_t my_lock;
spinlock_init(&my_lock);

lock(&my_lock);
// 임계 구역
unlock(&my_lock);
```

</details>

### 문제 5: 바쁜 대기의 문제점

바쁜 대기(Busy Waiting)의 문제점을 설명하고, 해결 방법을 제시하세요.

<details>
<summary>정답 보기</summary>

**바쁜 대기의 문제점:**

1. **CPU 시간 낭비**
   - 대기 중에도 CPU 사이클 소비
   - 다른 프로세스가 실행할 수 있는 시간 빼앗음

2. **우선순위 역전**
   - 높은 우선순위 프로세스가 스핀
   - 낮은 우선순위 프로세스가 락을 가지고 있어 진행 못 함

3. **전력 소모**
   - 모바일/임베디드 시스템에서 배터리 소모

**해결 방법:**

1. **블로킹 락 (Blocking Lock)**
   - 대기 시 프로세스를 Sleep 상태로 전환
   - 락 해제 시 Wake up
   - 예: 뮤텍스, 세마포어

2. **하이브리드 접근**
   - 짧은 시간 스핀 후 블로킹
   - Linux의 futex, Java의 synchronized

3. **yield 사용**
   - 스핀 대신 다른 스레드에 양보
   - `sched_yield()` 호출

</details>

---

## 실습 과제

### 실습 1: 피터슨 알고리즘 검증

`examples/OS_Theory/07_sync_primitives.py`를 실행하고 피터슨 알고리즘(Peterson's Algorithm) 데모를 분석하세요.

**과제:**
1. `PetersonLock`에서 `turn` 변수를 제거하세요. `demo_peterson()`을 실행하고 상호 배제(Mutual Exclusion)가 실패하는 이유를 설명하세요
2. 반복 횟수를 500,000으로 늘리고 5번 실행하세요. 실제 카운트 vs 기대값을 기록하세요. 관찰된 최대 오차는 얼마인가요?
3. 피터슨 알고리즘을 N개 스레드로 일반화하는 `FilterLock`을 구현하세요 (N-1 레벨의 flag 배열 사용)

### 실습 2: 원자적 연산 비교

다양한 동기화 프리미티브의 성능을 비교하세요:

```python
import threading, time

def benchmark_lock(lock_type, label, n=1_000_000):
    counter = [0]
    lock = lock_type()

    def worker():
        for _ in range(n // 2):
            lock.acquire()
            counter[0] += 1
            lock.release()

    t0 = threading.Thread(target=worker)
    t1 = threading.Thread(target=worker)
    start = time.perf_counter()
    t0.start(); t1.start()
    t0.join(); t1.join()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s, count={counter[0]}")
```

**과제:**
1. `threading.Lock`, `threading.RLock`, `threading.Semaphore(1)`을 벤치마크하세요. 성능별로 순위를 매기세요
2. `Lock`이 일반적으로 `RLock`과 `Semaphore`보다 빠른 이유를 설명하세요
3. 락 없이 테스트하고 오류율을 보고하세요 — Python의 GIL이 결과에 어떤 영향을 미치나요?

### 실습 3: 배리어 구현

`examples/OS_Theory/07_sync_primitives.py`의 `SimpleBarrier`를 연구하세요.

**과제:**
1. 현재 배리어는 세대별로 `Event`를 사용합니다. 대신 `Condition` 변수를 사용하는 대안을 구현하세요
2. 정확히 하나의 스레드를 "리더"로 식별하는 `barrier.wait()` 반환 값을 추가하세요 (배리어 단계당 하나의 스레드에만 True 반환)
3. 8개 스레드와 5개 단계로 배리어를 테스트하고, 모든 Phase N이 Phase N+1 전에 완료되는지 검증하세요

---

## 연습 문제

### 연습 1: 경쟁 조건(Race Condition) 식별

간단한 은행 계좌를 구현하는 다음 코드를 살펴보세요. 두 스레드가 `transfer()`를 동시에 실행합니다.

```c
#include <stdio.h>
#include <pthread.h>

int account_A = 1000;
int account_B = 500;

void transfer(int *from, int *to, int amount) {
    if (*from >= amount) {       // 단계 1: 잔액 확인
        *from -= amount;         // 단계 2: 차감
        *to   += amount;         // 단계 3: 입금
    }
}

void *thread1(void *arg) { transfer(&account_A, &account_B, 200); return NULL; }
void *thread2(void *arg) { transfer(&account_A, &account_B, 900); return NULL; }
```

1. 이 코드에서 임계 구역(critical section)을 식별하세요
2. 두 이체(transfer) 모두 처음 확인을 통과한다는 전제 하에, thread1과 thread2의 특정 인터리빙(interleaving)으로 account_A가 음수 잔액이 되는 상황을 설명하세요
3. 시스템 내 총 금액은 처음에 얼마인가요? 경쟁 조건이 이 합계를 변경할 수 있나요? 설명하세요
4. 올바른 동작을 보장하기 위해 전역 `pthread_mutex_t`를 사용하여 코드를 수정하세요

### 연습 2: 임계 구역 요구사항 분석

두 프로세스 임계 구역 문제에 대한 각 "해결책"이 세 가지 요구사항(상호 배제(mutual exclusion), 진행(progress), 유한 대기(bounded waiting)) 중 어느 것을 위반하는지 식별하세요. 두 프로세스 P0와 P1을 가정합니다.

**제안된 해결책 A** — 단순 turn 변수:
```c
int turn = 0;
// 프로세스 Pi:
while (turn != i);  // 대기
// 임계 구역
turn = 1 - i;       // 다른 프로세스에 turn 양보
```

**제안된 해결책 B** — 플래그만, turn 없음:
```c
bool flag[2] = {false, false};
// 프로세스 Pi:
flag[i] = true;
while (flag[1-i]);  // 대기
// 임계 구역
flag[i] = false;
```

각 해결책에 대해:
1. 상호 배제를 만족하나요? 증명하거나 반례를 제시하세요
2. 진행을 만족하나요? 증명하거나 반례를 제시하세요
3. 유한 대기를 만족하나요? 설명하세요

### 연습 3: 피터슨 알고리즘(Peterson's Algorithm) 단계별 추적

프로세스 P0와 P1을 위한 피터슨 알고리즘(Peterson's Algorithm):
```c
bool flag[2] = {false, false};
int turn;

// 프로세스 Pi (i=0 또는 1):
flag[i] = true;
turn = 1 - i;
while (flag[1-i] && turn == 1-i);  // 바쁜 대기
// --- 임계 구역 ---
flag[i] = false;
```

P0와 P1이 동시에 진입을 시도하는 다음 인터리빙을 추적하세요. 각 단계 후 `flag[0]`, `flag[1]`, `turn`의 값을 기록하세요. 어떤 프로세스가 먼저 임계 구역에 진입하나요?

| 단계 | 동작 | flag[0] | flag[1] | turn | 결과 |
|------|------|---------|---------|------|------|
| 1 | P0: flag[0]=true | | | | |
| 2 | P1: flag[1]=true | | | | |
| 3 | P0: turn=1 | | | | |
| 4 | P1: turn=0 | | | | |
| 5 | P0: while 확인 | | | | |
| 6 | P1: while 확인 | | | | |

설명: `turn`을 마지막에 설정하는 것이 정확히 하나의 프로세스만 진행됨을 보장하는 이유는 무엇인가요?

### 연습 4: TAS와 CAS 의미론

**파트 A**: 테스트-앤-셋(Test-and-Set, TAS)을 사용하여 스핀락(spinlock)을 구현하세요. `TestAndSet(bool *target)`을 사용하는 `lock()`과 `unlock()`의 의사 코드(pseudocode)를 작성하세요.

**파트 B**: 비교-앤-교환(Compare-and-Swap, CAS)을 사용하여 락 없는(lock-free) 카운터 증가를 구현하세요. `CompareAndSwap(int *value, int expected, int new_val)`을 사용하는 `atomic_increment(int *counter)`의 의사 코드를 작성하세요.

**파트 C**: TAS를 사용한 스핀락은 상호 배제를 만족하지만 유한 대기를 위반할 수 있습니다. 3개 프로세스 중 하나가 절대 락을 획득하지 못하는 시나리오를 설명하세요. 그런 다음 유한 대기를 보장하기 위해 필요한 수정(티켓 락(ticket lock))을 설명하세요.

### 연습 5: 바쁜 대기(Busy Waiting)와 블로킹(Blocking)

시스템에 100개의 스레드가 평균 2ms가 걸리는 임계 구역을 두고 경쟁합니다. 시스템에는 CPU 코어 4개가 있습니다.

1. 스레드가 **스핀락(spinlock)**(바쁜 대기)을 사용한다면, 99개의 스레드가 1개 스레드가 완료되기를 기다리는 동안 낭비되는 CPU 시간은 얼마인가요? 4코어 용량의 비율로 나타내세요.
2. 스레드가 **뮤텍스(mutex)**(블로킹)를 사용한다면 대기 스레드에게 어떤 일이 발생하며 왜 CPU가 낭비되지 않나요?
3. 블로킹 뮤텍스보다 스핀락이 **선호되는** 시나리오를 하나 제시하고 이유를 설명하세요.
4. 리눅스(Linux)의 `mutex_spin_on_owner()`가 사용하는 "이중 모드(dual-mode)" 전략은 무엇인가요? 두 접근 방식을 결합하는 이유는 무엇인가요?

---

## 다음 단계

- [동기화 도구](./08_Synchronization_Tools.md) - 뮤텍스, 세마포어, 모니터

---

## 참고 자료

- [OSTEP - Concurrency: Locks](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf)
- [The Art of Multiprocessor Programming (Herlihy & Shavit)](https://www.elsevier.com/books/the-art-of-multiprocessor-programming/)
- [C11 Atomic Operations](https://en.cppreference.com/w/c/atomic)

