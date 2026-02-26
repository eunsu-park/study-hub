# 동기화 도구

**이전**: [동기화 기초](./07_Synchronization_Basics.md) | **다음**: [데드락](./09_Deadlock.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 뮤텍스(Mutex)와 이진 세마포어(Binary Semaphore)를 구별하고 소유권 차이를 설명할 수 있습니다
2. 자원 풀링을 위한 카운팅 세마포어(Counting Semaphore)를 구현할 수 있습니다
3. 세마포어를 사용하여 생산자-소비자(Producer-Consumer) 문제를 해결할 수 있습니다
4. 독자-저자(Readers-Writers) 문제와 식사하는 철학자(Dining Philosophers) 문제를 해결할 수 있습니다
5. 모니터(Monitor)가 공유 상태를 캡슐화하여 동기화를 단순화하는 방법을 설명할 수 있습니다

---

원시 락(Raw Lock)과 Test-and-Set은 대부분의 프로그래머에게 너무 저수준입니다. 더 높은 수준의 도구인 세마포어(Semaphore), 모니터(Monitor), 조건 변수(Condition Variable)는 동시 접근을 조율하는 구조화된 방법을 제공합니다. 이것들은 모든 데이터베이스, 웹 서버, 운영체제 커널 내부에서 사용되는 구성 요소입니다.

## 목차

1. [뮤텍스 (Mutex)](#1-뮤텍스-mutex)
2. [세마포어 (Semaphore)](#2-세마포어-semaphore)
3. [모니터 (Monitor)](#3-모니터-monitor)
4. [조건 변수 (Condition Variable)](#4-조건-변수-condition-variable)
5. [고전 동기화 문제](#5-고전-동기화-문제)
6. [연습 문제](#6-연습-문제)

---

## 1. 뮤텍스 (Mutex)

### 개념

```
┌─────────────────────────────────────────────────────────┐
│                    뮤텍스 (Mutex)                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Mutex = Mutual Exclusion의 약자                        │
│        = 이진 락 (Binary Lock)                          │
│        = 한 번에 하나의 스레드만 임계 구역 진입 허용      │
│                                                         │
│  상태:                                                  │
│  • 잠김 (Locked): 한 스레드가 락을 소유                  │
│  • 열림 (Unlocked): 사용 가능                           │
│                                                         │
│  기본 연산:                                             │
│  • lock(): 락 획득 (다른 스레드 대기)                    │
│  • unlock(): 락 해제                                   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │          뮤텍스 동작 시각화                       │    │
│  │                                                 │    │
│  │  스레드1: ─lock()─┬───임계구역───┬─unlock()─     │    │
│  │  스레드2: ─lock()─│─대기─────────│─임계구역─     │    │
│  │                  │              ▲               │    │
│  │                  └──────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### pthread_mutex 사용

```c
#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        pthread_mutex_lock(&mutex);    // 락 획득
        counter++;                      // 임계 구역
        pthread_mutex_unlock(&mutex);  // 락 해제
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Counter: %d\n", counter);  // 2000000 (정확!)
    return 0;
}
```

### 뮤텍스 초기화 방법

```c
// 방법 1: 정적 초기화
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 방법 2: 동적 초기화
pthread_mutex_t mutex;
pthread_mutex_init(&mutex, NULL);  // 속성 NULL = 기본

// 사용 후 정리 (동적 초기화 시)
pthread_mutex_destroy(&mutex);

// 속성 설정 예시
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);  // 재귀 뮤텍스
pthread_mutex_init(&mutex, &attr);
pthread_mutexattr_destroy(&attr);
```

### 뮤텍스 유형

```
┌─────────────────────────────────────────────────────────┐
│                    뮤텍스 유형                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. PTHREAD_MUTEX_NORMAL (기본)                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • 재귀 lock 시 데드락                             │  │
│  │  • 소유하지 않은 스레드가 unlock 시 정의되지 않음   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. PTHREAD_MUTEX_RECURSIVE                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • 같은 스레드가 여러 번 lock 가능                  │  │
│  │  • lock 횟수만큼 unlock 필요                       │  │
│  │  • 재귀 함수에서 유용                              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. PTHREAD_MUTEX_ERRORCHECK                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • 재귀 lock 시 에러 반환                          │  │
│  │  • 소유하지 않은 unlock 시 에러 반환               │  │
│  │  • 디버깅용                                        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 세마포어 (Semaphore)

### 개념

```
┌─────────────────────────────────────────────────────────┐
│                   세마포어 (Semaphore)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Dijkstra가 1965년 제안                                 │
│  = 정수 값을 가지는 동기화 객체                         │
│  = 여러 개의 자원을 관리할 수 있음                       │
│                                                         │
│  기본 연산:                                             │
│  • wait() / P() / down() / acquire()                   │
│    - 값이 0보다 크면 감소시키고 진행                    │
│    - 값이 0이면 대기                                    │
│                                                         │
│  • signal() / V() / up() / release()                   │
│    - 값을 증가시킴                                      │
│    - 대기 중인 프로세스가 있으면 깨움                   │
│                                                         │
│  종류:                                                  │
│  • 이진 세마포어 (Binary): 0 또는 1 (뮤텍스와 유사)     │
│  • 카운팅 세마포어 (Counting): 0 이상의 정수            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### P/V 연산 (wait/signal)

```
┌─────────────────────────────────────────────────────────┐
│                    P/V 연산 정의                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  P(S) / wait(S):                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  wait(S) {                                        │  │
│  │      while (S <= 0) {                             │  │
│  │          // 대기 (바쁜 대기 또는 블로킹)           │  │
│  │      }                                            │  │
│  │      S = S - 1;                                   │  │
│  │  }                                                │  │
│  │  // P = Proberen (Dutch: to test)                │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  V(S) / signal(S):                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  signal(S) {                                      │  │
│  │      S = S + 1;                                   │  │
│  │      // 대기 중인 프로세스가 있으면 깨움           │  │
│  │  }                                                │  │
│  │  // V = Verhogen (Dutch: to increment)           │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  중요: P와 V는 원자적으로 수행되어야 함                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### POSIX 세마포어 사용

```c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t semaphore;
int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        sem_wait(&semaphore);    // P 연산: 대기 및 감소
        counter++;
        sem_post(&semaphore);    // V 연산: 증가 및 시그널
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    // 세마포어 초기화 (값=1: 이진 세마포어)
    sem_init(&semaphore, 0, 1);  // 0=스레드 간 공유

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    sem_destroy(&semaphore);

    printf("Counter: %d\n", counter);  // 2000000
    return 0;
}
```

### 카운팅 세마포어 예제

```c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define MAX_CONNECTIONS 3

sem_t connection_sem;

void* client(void* arg) {
    int id = *(int*)arg;

    printf("클라이언트 %d: 연결 대기 중\n", id);

    sem_wait(&connection_sem);  // 연결 슬롯 획득

    printf("클라이언트 %d: 연결됨 (작업 중...)\n", id);
    sleep(2);  // 작업 시뮬레이션

    printf("클라이언트 %d: 연결 해제\n", id);
    sem_post(&connection_sem);  // 연결 슬롯 반환

    return NULL;
}

int main() {
    pthread_t threads[10];
    int ids[10];

    // 최대 3개의 동시 연결 허용
    sem_init(&connection_sem, 0, MAX_CONNECTIONS);

    for (int i = 0; i < 10; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, client, &ids[i]);
    }

    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }

    sem_destroy(&connection_sem);
    return 0;
}

/*
출력 예시:
클라이언트 0: 연결 대기 중
클라이언트 0: 연결됨 (작업 중...)
클라이언트 1: 연결 대기 중
클라이언트 1: 연결됨 (작업 중...)
클라이언트 2: 연결 대기 중
클라이언트 2: 연결됨 (작업 중...)
클라이언트 3: 연결 대기 중       <- 3개 이상은 대기
...
*/
```

### 뮤텍스 vs 세마포어

```
┌──────────────────┬─────────────────────┬─────────────────────┐
│      특성         │       뮤텍스        │      세마포어       │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 값 범위          │ 0 또는 1            │ 0 이상              │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 소유권           │ 있음 (락 소유자만   │ 없음 (누구나        │
│                  │ unlock 가능)        │ signal 가능)        │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 용도             │ 상호 배제           │ 자원 카운팅,        │
│                  │                     │ 신호 전달           │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 재귀적 획득      │ 가능 (RECURSIVE)    │ 불가능              │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 우선순위 상속    │ 지원 가능           │ 일반적으로 미지원   │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 사용 예          │ 공유 데이터 보호    │ 생산자-소비자,      │
│                  │                     │ 연결 풀 관리        │
└──────────────────┴─────────────────────┴─────────────────────┘
```

---

## 3. 모니터 (Monitor)

### 개념

```
┌─────────────────────────────────────────────────────────┐
│                    모니터 (Monitor)                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  모니터 = 동기화를 캡슐화한 고급 추상화                  │
│         = 공유 데이터 + 연산 + 동기화를 하나로 묶음      │
│         = 한 번에 하나의 프로세스만 모니터 내부 진입     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                   모니터                        │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         공유 데이터 (Private)           │   │    │
│  │  │         int counter;                    │   │    │
│  │  │         int buffer[N];                  │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         조건 변수                        │   │    │
│  │  │         condition notEmpty;             │   │    │
│  │  │         condition notFull;              │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │         프로시저 (Public)               │   │    │
│  │  │         void insert(int item);          │   │    │
│  │  │         int remove();                   │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  │                                                │    │
│  │          ← 진입 큐 (Entry Queue)              │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  특징:                                                  │
│  • 컴파일러가 상호 배제를 자동 보장                     │
│  • Java synchronized, Python Lock 등                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Java에서의 모니터

```java
// Java synchronized를 이용한 모니터
public class Counter {
    private int count = 0;

    // synchronized 메서드 = 모니터의 프로시저
    public synchronized void increment() {
        count++;  // 자동으로 상호 배제 보장
    }

    public synchronized void decrement() {
        count--;
    }

    public synchronized int getCount() {
        return count;
    }
}

// 사용 예
Counter counter = new Counter();

Thread t1 = new Thread(() -> {
    for (int i = 0; i < 1000000; i++) {
        counter.increment();
    }
});

Thread t2 = new Thread(() -> {
    for (int i = 0; i < 1000000; i++) {
        counter.increment();
    }
});

t1.start(); t2.start();
t1.join(); t2.join();
System.out.println(counter.getCount());  // 2000000
```

---

## 4. 조건 변수 (Condition Variable)

### 개념

```
┌─────────────────────────────────────────────────────────┐
│               조건 변수 (Condition Variable)             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  조건 변수 = 특정 조건이 참이 될 때까지 대기하게 하는 도구│
│            = 뮤텍스와 함께 사용                         │
│                                                         │
│  연산:                                                  │
│  • wait(cond, mutex):                                  │
│    1. 뮤텍스 해제                                      │
│    2. 조건 변수에서 대기                               │
│    3. 깨어나면 뮤텍스 재획득                           │
│                                                         │
│  • signal(cond) / pthread_cond_signal():               │
│    - 대기 중인 스레드 하나 깨움                         │
│                                                         │
│  • broadcast(cond) / pthread_cond_broadcast():         │
│    - 대기 중인 모든 스레드 깨움                         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                   동작 흐름                      │    │
│  │                                                 │    │
│  │  Thread A           Thread B                   │    │
│  │  ─────────           ─────────                 │    │
│  │  lock(mutex)                                   │    │
│  │  while (!조건)                                 │    │
│  │      wait(cond) ───┐                          │    │
│  │          │ 뮤텍스해제 │                          │    │
│  │          │ 대기    │  lock(mutex)             │    │
│  │          │        │  조건 변경                 │    │
│  │          │        │  signal(cond)             │    │
│  │          │◀───────│  unlock(mutex)            │    │
│  │  뮤텍스재획득                                   │    │
│  │  임계 구역 계속                                 │    │
│  │  unlock(mutex)                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### pthread 조건 변수 사용

```c
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool ready = false;
int data = 0;

void* producer(void* arg) {
    pthread_mutex_lock(&mutex);

    data = 42;
    ready = true;
    printf("생산자: 데이터 준비 완료\n");

    pthread_cond_signal(&cond);  // 소비자 깨움
    pthread_mutex_unlock(&mutex);

    return NULL;
}

void* consumer(void* arg) {
    pthread_mutex_lock(&mutex);

    while (!ready) {  // while 루프 사용! (spurious wakeup 방지)
        printf("소비자: 데이터 대기 중...\n");
        pthread_cond_wait(&cond, &mutex);
    }

    printf("소비자: 데이터 수신 = %d\n", data);
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main() {
    pthread_t prod, cons;

    pthread_create(&cons, NULL, consumer, NULL);
    sleep(1);  // 소비자가 먼저 대기하도록
    pthread_create(&prod, NULL, producer, NULL);

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    return 0;
}
```

### Spurious Wakeup

```
┌─────────────────────────────────────────────────────────┐
│               Spurious Wakeup (가짜 깨우기)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  문제: signal 없이 wait에서 깨어날 수 있음               │
│                                                         │
│  잘못된 코드:                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  if (!ready) {                                    │  │
│  │      pthread_cond_wait(&cond, &mutex);  // 위험! │  │
│  │  }                                                │  │
│  │  // 조건이 참이 아닌데 실행될 수 있음              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  올바른 코드:                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  while (!ready) {                                 │  │
│  │      pthread_cond_wait(&cond, &mutex);            │  │
│  │  }                                                │  │
│  │  // 깨어나면 조건을 다시 확인                      │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  규칙: 항상 while 루프 안에서 wait() 호출!              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 고전 동기화 문제

### 생산자-소비자 문제 (Bounded Buffer)

```
┌─────────────────────────────────────────────────────────┐
│              생산자-소비자 문제                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  설정:                                                  │
│  • 고정 크기 버퍼 (N개)                                 │
│  • 생산자: 버퍼에 아이템 추가                           │
│  • 소비자: 버퍼에서 아이템 제거                         │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                                                   │  │
│  │  생산자 ──▶ [  버퍼  ] ──▶ 소비자                │  │
│  │           ┌───────────┐                          │  │
│  │           │ ? │ ? │ ? │                          │  │
│  │           └───────────┘                          │  │
│  │           N = 3                                  │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  동기화 요구사항:                                       │
│  1. 버퍼가 가득 차면 생산자 대기                        │
│  2. 버퍼가 비어있으면 소비자 대기                       │
│  3. 버퍼 접근 시 상호 배제                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 5

int buffer[BUFFER_SIZE];
int in = 0, out = 0;

sem_t empty;  // 빈 슬롯 수 (초기값: BUFFER_SIZE)
sem_t full;   // 찬 슬롯 수 (초기값: 0)
pthread_mutex_t mutex;

void* producer(void* arg) {
    for (int i = 0; i < 10; i++) {
        int item = i;

        sem_wait(&empty);           // 빈 슬롯 대기
        pthread_mutex_lock(&mutex);

        buffer[in] = item;
        printf("생산: %d (위치 %d)\n", item, in);
        in = (in + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&mutex);
        sem_post(&full);            // 찬 슬롯 증가

        usleep(100000);  // 0.1초
    }
    return NULL;
}

void* consumer(void* arg) {
    for (int i = 0; i < 10; i++) {
        sem_wait(&full);            // 찬 슬롯 대기
        pthread_mutex_lock(&mutex);

        int item = buffer[out];
        printf("소비: %d (위치 %d)\n", item, out);
        out = (out + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&mutex);
        sem_post(&empty);           // 빈 슬롯 증가

        usleep(150000);  // 0.15초
    }
    return NULL;
}

int main() {
    pthread_t prod, cons;

    sem_init(&empty, 0, BUFFER_SIZE);
    sem_init(&full, 0, 0);
    pthread_mutex_init(&mutex, NULL);

    pthread_create(&prod, NULL, producer, NULL);
    pthread_create(&cons, NULL, consumer, NULL);

    pthread_join(prod, NULL);
    pthread_join(cons, NULL);

    return 0;
}
```

### 독자-저자 문제 (Readers-Writers)

```
┌─────────────────────────────────────────────────────────┐
│                 독자-저자 문제                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  설정:                                                  │
│  • 공유 데이터베이스                                    │
│  • 독자 (Reader): 데이터 읽기만 함                      │
│  • 저자 (Writer): 데이터 수정                           │
│                                                         │
│  규칙:                                                  │
│  • 여러 독자가 동시에 읽기 가능                         │
│  • 저자가 쓰는 동안 다른 접근 불가                      │
│  • 저자는 배타적 접근 필요                              │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                                                   │  │
│  │  독자1 ──읽기──┐                                  │  │
│  │  독자2 ──읽기──┼──▶ [ 데이터베이스 ]              │  │
│  │  독자3 ──읽기──┘          ↑                       │  │
│  │                          │                       │  │
│  │  저자  ────────쓰기───────┘                       │  │
│  │       (배타적 접근)                               │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t write_lock = PTHREAD_MUTEX_INITIALIZER;
int read_count = 0;
int shared_data = 0;

void* reader(void* arg) {
    int id = *(int*)arg;

    pthread_mutex_lock(&mutex);
    read_count++;
    if (read_count == 1) {
        pthread_mutex_lock(&write_lock);  // 첫 독자가 저자 차단
    }
    pthread_mutex_unlock(&mutex);

    // 읽기 수행 (임계 구역 아님, 여러 독자 동시 가능)
    printf("독자 %d: 데이터 = %d\n", id, shared_data);
    usleep(100000);

    pthread_mutex_lock(&mutex);
    read_count--;
    if (read_count == 0) {
        pthread_mutex_unlock(&write_lock);  // 마지막 독자가 저자 허용
    }
    pthread_mutex_unlock(&mutex);

    return NULL;
}

void* writer(void* arg) {
    int id = *(int*)arg;

    pthread_mutex_lock(&write_lock);

    // 쓰기 수행 (배타적 접근)
    shared_data++;
    printf("저자 %d: 데이터를 %d로 변경\n", id, shared_data);
    usleep(200000);

    pthread_mutex_unlock(&write_lock);

    return NULL;
}

int main() {
    pthread_t readers[5], writers[2];
    int ids[5] = {1, 2, 3, 4, 5};
    int wids[2] = {1, 2};

    for (int i = 0; i < 5; i++)
        pthread_create(&readers[i], NULL, reader, &ids[i]);
    for (int i = 0; i < 2; i++)
        pthread_create(&writers[i], NULL, writer, &wids[i]);

    for (int i = 0; i < 5; i++)
        pthread_join(readers[i], NULL);
    for (int i = 0; i < 2; i++)
        pthread_join(writers[i], NULL);

    return 0;
}
```

### 식사하는 철학자 문제 (Dining Philosophers)

```
┌─────────────────────────────────────────────────────────┐
│              식사하는 철학자 문제                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  설정:                                                  │
│  • 5명의 철학자, 5개의 젓가락                           │
│  • 각 철학자는 생각하거나 먹음                          │
│  • 먹으려면 양쪽 젓가락이 필요                          │
│                                                         │
│           ┌───────────────────────────┐                 │
│           │          P0              │                 │
│           │      ◇       ◇          │                 │
│           │     C4       C0          │                 │
│           │                          │                 │
│        P4 ◇                      ◇ P1│                 │
│           │   C3             C1     │                 │
│           │                          │                 │
│        P3 ◇──────C2──────◇ P2       │                 │
│           │                          │                 │
│           └───────────────────────────┘                 │
│                                                         │
│  문제: 데드락 발생 가능!                                │
│  모든 철학자가 왼쪽 젓가락을 집으면 → 아무도 못 먹음     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define N 5
#define LEFT(i) (i)
#define RIGHT(i) ((i + 1) % N)

pthread_mutex_t chopsticks[N];

void* philosopher(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 3; i++) {
        // 생각하기
        printf("철학자 %d: 생각 중...\n", id);
        usleep(100000);

        // 데드락 방지: 짝수 철학자는 왼쪽 먼저, 홀수는 오른쪽 먼저
        if (id % 2 == 0) {
            pthread_mutex_lock(&chopsticks[LEFT(id)]);
            pthread_mutex_lock(&chopsticks[RIGHT(id)]);
        } else {
            pthread_mutex_lock(&chopsticks[RIGHT(id)]);
            pthread_mutex_lock(&chopsticks[LEFT(id)]);
        }

        // 먹기
        printf("철학자 %d: 먹는 중...\n", id);
        usleep(200000);

        // 젓가락 내려놓기
        pthread_mutex_unlock(&chopsticks[LEFT(id)]);
        pthread_mutex_unlock(&chopsticks[RIGHT(id)]);
    }

    return NULL;
}

int main() {
    pthread_t philosophers[N];
    int ids[N];

    for (int i = 0; i < N; i++)
        pthread_mutex_init(&chopsticks[i], NULL);

    for (int i = 0; i < N; i++) {
        ids[i] = i;
        pthread_create(&philosophers[i], NULL, philosopher, &ids[i]);
    }

    for (int i = 0; i < N; i++)
        pthread_join(philosophers[i], NULL);

    return 0;
}
```

---

## 6. 연습 문제

### 문제 1: 세마포어 값

초기값이 5인 세마포어에 대해 P, P, V, P, P, P 연산을 순서대로 수행하면 최종 세마포어 값은?

<details>
<summary>정답 보기</summary>

**연산 순서와 값 변화:**
- 초기값: 5
- P: 5 - 1 = 4
- P: 4 - 1 = 3
- V: 3 + 1 = 4
- P: 4 - 1 = 3
- P: 3 - 1 = 2
- P: 2 - 1 = 1

**최종 값: 1**

</details>

### 문제 2: 생산자-소비자

생산자-소비자 문제에서 empty와 full 세마포어의 역할을 설명하고, 순서가 중요한 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

**empty 세마포어:**
- 빈 버퍼 슬롯 수 관리
- 초기값: 버퍼 크기 (N)
- 생산자가 P 연산 (슬롯 할당)
- 소비자가 V 연산 (슬롯 반환)

**full 세마포어:**
- 채워진 버퍼 슬롯 수 관리
- 초기값: 0
- 생산자가 V 연산 (아이템 추가 알림)
- 소비자가 P 연산 (아이템 대기)

**순서가 중요한 이유:**
- 세마포어 P 연산과 뮤텍스 획득 순서
- 잘못된 순서: mutex lock → sem_wait → 데드락 가능
- 올바른 순서: sem_wait → mutex lock → 데드락 방지

</details>

### 문제 3: 모니터 구현

다음 세마포어 코드를 모니터(조건 변수) 방식으로 변환하세요.

```c
sem_t sem;
sem_init(&sem, 0, 0);

// 생산자
sem_post(&sem);

// 소비자
sem_wait(&sem);
```

<details>
<summary>정답 보기</summary>

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int count = 0;

// 생산자
pthread_mutex_lock(&mutex);
count++;
pthread_cond_signal(&cond);
pthread_mutex_unlock(&mutex);

// 소비자
pthread_mutex_lock(&mutex);
while (count == 0) {
    pthread_cond_wait(&cond, &mutex);
}
count--;
pthread_mutex_unlock(&mutex);
```

</details>

### 문제 4: 식사하는 철학자 데드락

식사하는 철학자 문제에서 데드락이 발생하는 시나리오를 설명하고, 세 가지 해결 방법을 제시하세요.

<details>
<summary>정답 보기</summary>

**데드락 시나리오:**
모든 철학자가 동시에 왼쪽 젓가락을 집으면:
- P0: C0 보유, C4 대기
- P1: C1 보유, C0 대기
- P2: C2 보유, C1 대기
- P3: C3 보유, C2 대기
- P4: C4 보유, C3 대기
→ 순환 대기 → 데드락!

**해결 방법:**

1. **비대칭 락 획득:**
   - 짝수 철학자: 왼쪽 먼저
   - 홀수 철학자: 오른쪽 먼저
   - 순환 대기 깨짐

2. **동시 젓가락 획득:**
   - 양쪽 젓가락을 동시에 획득할 수 있을 때만 집음
   - 중앙 뮤텍스로 보호

3. **최대 N-1명 제한:**
   - 세마포어로 최대 4명만 식탁에 앉도록 제한
   - 최소 한 명은 양쪽 젓가락 사용 가능

</details>

### 문제 5: 조건 변수 사용

왜 조건 변수의 wait() 호출을 while 루프 안에서 해야 하는지 설명하세요.

<details>
<summary>정답 보기</summary>

**이유 1: Spurious Wakeup (가짜 깨우기)**
- 시스템에 의해 signal 없이 깨어날 수 있음
- 조건을 다시 확인하지 않으면 잘못된 상태에서 진행

**이유 2: 다중 대기자**
- 여러 스레드가 같은 조건 변수에서 대기 중
- broadcast 후 모두 깨어나지만, 한 스레드만 조건 만족 가능
- 나머지는 다시 대기해야 함

**이유 3: 조건 변경**
- 깨어난 후 조건이 다시 거짓이 될 수 있음
- 다른 스레드가 먼저 자원 사용 가능

**올바른 패턴:**
```c
pthread_mutex_lock(&mutex);
while (!condition) {
    pthread_cond_wait(&cond, &mutex);
}
// 조건이 참임이 보장됨
pthread_mutex_unlock(&mutex);
```

</details>

---

## 실습 과제

### 실습 1: 생산자-소비자 변형

`examples/OS_Theory/08_producer_consumer.py`를 실행하고 유한 버퍼를 실험하세요.

**과제:**
1. 버퍼 용량을 1(단일 요소 버퍼)로 변경하세요. 용량 5와 비교하여 처리량에 어떤 영향을 미치나요?
2. 불균형 시나리오를 만드세요: 5개의 빠른 생산자(0.01초 지연)와 1개의 느린 소비자(0.1초 지연). 버퍼 활용 패턴을 관찰하세요
3. 높은 우선순위 항목이 먼저 소비되는 우선순위 생산자-소비자(Priority Producer-Consumer)를 구현하세요 (힌트: deque 대신 힙 사용)

### 실습 2: 식사하는 철학자 해결책

`examples/OS_Theory/08_producer_consumer.py`를 사용하여 식사하는 철학자 문제(Dining Philosophers)의 다양한 해결책을 탐구하세요.

**과제:**
1. "웨이터" 해결책을 구현하세요: 동시에 식사를 시도할 수 있는 철학자 수를 N-1로 제한하는 세마포어 추가
2. "Chandy-Misra" 해결책을 구현하세요: 포크는 처음에 더럽고, 사용 후 더럽게 표시되며, 더러운 포크는 요청하는 이웃에게 반드시 전달
3. 세 가지 해결책(자원 계층, 웨이터, Chandy-Misra)을 1000번의 식사로 비교하고 철학자당 평균 대기 시간을 측정하세요

### 실습 3: 읽기-쓰기 락

동시 읽기는 허용하지만 배타적 쓰기를 보장하는 읽기-쓰기 락(Read-Write Lock)을 구현하세요:

**과제:**
1. `acquire_read()`, `release_read()`, `acquire_write()`, `release_write()` 메서드가 있는 `ReadWriteLock` 클래스를 만드세요
2. "독자 우선"(Readers-preference) 정책을 구현하세요 (활성 작성자가 없으면 독자는 대기하지 않음)
3. 5개 독자 스레드(각 100회 읽기)와 2개 작성자 스레드(각 20회 쓰기)로 테스트하세요. 쓰기 중에 읽기가 발생하지 않는지 확인하세요

---

## 연습 문제

### 연습 1: 뮤텍스(Mutex)와 세마포어(Semaphore) 의미론

각 동기화(synchronization) 시나리오에서 **뮤텍스** 또는 **세마포어**를 선택하고 이유를 설명하세요. 세마포어를 선택한다면 이진(binary)인지 카운팅(counting)인지, 초기값은 얼마인지 명시하세요.

| 시나리오 | 도구 (뮤텍스/세마포어) | 초기값 | 이유 |
|----------|----------------------|--------|------|
| 4개 스레드가 업데이트하는 공유 `int counter` 보호 | | | |
| 동시 데이터베이스 연결을 10개로 제한 | | | |
| 큐에 새 작업이 추가되었음을 워커 스레드에 알림 | | | |
| 설정 파일을 한 번에 정확히 하나의 스레드만 읽도록 보장 | | | |
| 생산자가 소비자보다 앞서 실행되도록 조정 (크기=5 유한 버퍼) | | | |

### 연습 2: 세마포어(Semaphore) 추적

시스템이 크기 3의 버퍼에서 생산자(producer)와 소비자(consumer)를 조정하기 위해 두 세마포어를 사용합니다:
- `empty` (초기값 3): 빈 슬롯 수
- `full` (초기값 0): 꽉 찬 슬롯 수
- `mutex` (초기값 1): 버퍼 접근 보호

다음 일련의 작업을 추적하고 각 호출 후 세마포어 값을 기록하세요. 프로세스가 블록된다면 "BLOCK"을 쓰세요.

| 단계 | 작업 | empty | full | mutex | 비고 |
|------|------|-------|------|-------|------|
| 초기 | — | 3 | 0 | 1 | |
| 1 | 생산자: P(empty) | | | | |
| 2 | 생산자: P(mutex) | | | | |
| 3 | 생산자: 아이템 추가 | | | | |
| 4 | 생산자: V(mutex) | | | | |
| 5 | 생산자: V(full) | | | | |
| 6 | 생산자: P(empty) | | | | |
| 7 | 생산자: P(mutex) | | | | |
| 8 | 생산자: 아이템 추가 | | | | |
| 9 | 생산자: V(mutex) | | | | |
| 10 | 생산자: V(full) | | | | |
| 11 | 생산자: P(empty) | | | | |
| 12 | 생산자: P(mutex) | | | | |
| 13 | 생산자: 아이템 추가 | | | | |
| 14 | 생산자: V(mutex) | | | | |
| 15 | 생산자: V(full) | | | | |
| 16 | 생산자: P(empty) | | | | |

### 연습 3: 모니터(Monitor)와 조건 변수(Condition Variable) 설계

최대 용량 N인 `BoundedQueue` 모니터를 의사 코드(pseudocode)로 설계하세요. `enqueue(item)`과 `dequeue()`를 지원해야 합니다. 모니터는:
- 큐가 꽉 찼을 때 `enqueue()`를 블록해야 합니다
- 큐가 비었을 때 `dequeue()`를 블록해야 합니다
- 두 조건 변수(condition variable): `not_full`과 `not_empty`를 사용해야 합니다

```pseudocode
monitor BoundedQueue:
    queue = []
    capacity = N
    condition not_full
    condition not_empty

    procedure enqueue(item):
        // TODO: 채워주세요

    procedure dequeue():
        // TODO: 채워주세요
        return item
```

1. `wait()`와 `signal()`을 사용하여 `enqueue`와 `dequeue` 프로시저를 채워주세요
2. 조건 확인이 `if`문 대신 `while` 루프에 있어야 하는 이유는 무엇인가요? (힌트: 허위 깨어남(spurious wakeup))
3. `signal()`이 **신호-후-대기(signal-and-wait)** 의미론을 사용한다면 구현에서 무엇이 달라지나요?

### 연습 4: 식사하는 철학자(Dining Philosophers) 교착 상태 분석

다섯 철학자가 원형 테이블에 앉아 있습니다. 각자 왼쪽 포크를 집은 다음 오른쪽 포크를 집습니다. 다음 잘못된 구현을 고려하세요:

```c
void philosopher(int i) {
    while (true) {
        think();
        lock(fork[i]);           // 왼쪽 포크 집기
        lock(fork[(i+1) % 5]);  // 오른쪽 포크 집기
        eat();
        unlock(fork[i]);
        unlock(fork[(i+1) % 5]);
    }
}
```

1. 교착 상태(deadlock) 시나리오를 보여주세요: 5명의 철학자 모두 정확히 하나의 포크를 들고 있어 아무도 진행할 수 없는 특정 상태를 보여주세요
2. **자원 계층(resource hierarchy)** 해결책 적용: 철학자는 항상 번호가 낮은 포크를 먼저 획득해야 합니다. 이 수정을 사용하도록 `philosopher()`를 다시 작성하고 순환 대기(circular wait) 조건을 깨뜨리는 이유를 설명하세요
3. 자원 계층 수정을 적용해도 기아(starvation)가 발생할 수 있나요? 구체적인 시나리오를 제시하거나 발생할 수 없음을 증명하세요.

### 연습 5: 독자-저자(Readers-Writers) 문제 공정성

독자 우선(readers' preference) 방식의 고전적인 독자-저자 해결책은 저자(writer)를 기아 상태에 빠뜨릴 수 있습니다. 독자(reader) 도착률 λ_r = 10/초, 저자 도착률 λ_w = 1/초이고 읽기는 50ms, 쓰기는 20ms가 걸리는 시스템을 고려하세요.

1. 독자 우선에서 저자가 도착하여 5명의 활성 독자를 발견합니다. 새 독자가 계속 도착합니다. 저자가 접근 권한을 얻을 수 있나요? 기아 메커니즘을 설명하세요.
2. 저자 우선(writers' preference, 어떤 저자가 대기 중이거나 쓰고 있으면 독자는 대기)을 세마포어를 사용한 의사 코드로 구현하세요. 어떤 변수가 필요한가요?
3. 저자 우선에서 저자가 락을 보유하고 있습니다. 20명의 독자가 대기 중입니다. 저자가 완료합니다. 몇 명의 독자가 진행할 수 있나요? 한 번에 전부인가요 아니면 하나씩인가요?
4. 독자와 저자 모두의 기아를 방지하는 공정한 해결책을 설명하세요. 핵심 메커니즘은 무엇인가요?

---

## 다음 단계

- [데드락](./09_Deadlock.md) - 데드락 조건, 예방, 회피, 탐지

---

## 참고 자료

- [OSTEP - Condition Variables](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-cv.pdf)
- [POSIX Threads Programming](https://computing.llnl.gov/tutorials/pthreads/)
- [Java Concurrency in Practice](https://jcip.net/)

