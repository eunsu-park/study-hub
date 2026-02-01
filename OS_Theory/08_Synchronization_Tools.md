# 동기화 도구

## 개요

운영체제와 프로그래밍 언어는 동기화를 위한 다양한 도구를 제공합니다. 이 레슨에서는 뮤텍스, 세마포어, 모니터, 조건 변수를 학습하고, 고전적인 동기화 문제들을 해결합니다.

---

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

## 다음 단계

- [09_Deadlock.md](./09_Deadlock.md) - 데드락 조건, 예방, 회피, 탐지

---

## 참고 자료

- [OSTEP - Condition Variables](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-cv.pdf)
- [POSIX Threads Programming](https://computing.llnl.gov/tutorials/pthreads/)
- [Java Concurrency in Practice](https://jcip.net/)

