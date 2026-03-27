[이전: 디스크 스케줄링과 I/O](./21_Disk_Scheduling_IO.md)

---

# 22. 실시간 운영체제

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 경성, 준경성, 연성 실시간 요구사항을 구별할 수 있다
2. Rate Monotonic과 Earliest Deadline First 스케줄링을 설명할 수 있다
3. 임베디드 시스템을 위한 FreeRTOS와 Zephyr 아키텍처를 설명할 수 있다
4. 우선순위 상속을 포함한 우선순위 역전 해결책을 구현할 수 있다
5. 태스크 스케줄 가능성과 최악 실행 시간을 분석할 수 있다

---

## 목차

1. [실시간 개념](#1-실시간-개념)
2. [RTOS 스케줄링 알고리즘](#2-rtos-스케줄링-알고리즘)
3. [스케줄 가능성 분석](#3-스케줄-가능성-분석)
4. [우선순위 역전](#4-우선순위-역전)
5. [FreeRTOS 개요](#5-freertos-개요)
6. [Zephyr RTOS](#6-zephyr-rtos)
7. [RTOS 설계 패턴](#7-rtos-설계-패턴)
8. [연습문제](#8-연습문제)

---

## 1. 실시간 개념

### 1.1 실시간 분류

```
실시간 시스템:
  정확성은 결과와 타이밍 모두에 의존.

경성 실시간 (Hard Real-Time):
  데드라인 미스 = 시스템 실패 (치명적)
  예: 에어백 전개, 비행 제어, 심박 조율기
  보장: 데드라인 충족에 대한 수학적 증명

준경성 실시간 (Firm Real-Time):
  데드라인 미스 = 결과가 무용 (치명적이지는 않음)
  예: 비디오 프레임 렌더링, 금융 거래
  늦은 결과는 폐기

연성 실시간 (Soft Real-Time):
  데드라인 미스 = 품질 저하 (여전히 어느 정도 유용)
  예: 오디오 스트리밍, 웹 서버 응답
  늦은 결과는 가치가 감소
```

### 1.2 핵심 RTOS 개념

```
RTOS vs 범용 OS:

특성            | RTOS            | 범용 OS
----------------|-----------------|------------------
스케줄링         | 우선순위 기반    | 공정성 기반
지연 시간        | 결정적          | 최선 노력
인터럽트 지연    | < 10 μs        | 100 μs - 10 ms
컨텍스트 스위치   | < 5 μs         | 1-100 μs
메모리           | 정적/소규모     | 동적/대규모
응답 시간        | 보장됨          | 통계적
```

---

## 2. RTOS 스케줄링 알고리즘

### 2.1 Rate Monotonic Scheduling (RMS)

```c
#include <stdio.h>
#include <math.h>

/*
 * Rate Monotonic Scheduling:
 *   - 정적 우선순위 할당
 *   - 짧은 주기 = 높은 우선순위
 *   - 고정 우선순위 선점형 스케줄러 중 최적
 *
 *   이용률 한계 (Liu & Layland):
 *   U = Σ (Ci/Ti) ≤ n(2^(1/n) - 1)
 *
 *   n=1: 100%, n=2: 82.8%, n=3: 78.0%, n→∞: 69.3% (ln 2)
 */

typedef struct {
    int id;
    double period;      /* T: 최소 도착 간격 */
    double execution;   /* C: 최악 실행 시간 */
    double deadline;    /* D: 상대적 데드라인 (RMS에서는 = 주기) */
    int priority;       /* 낮은 숫자 = 높은 우선순위 */
} rtos_task_t;

double rm_utilization_bound(int n) {
    return n * (pow(2.0, 1.0 / n) - 1.0);
}

int rm_schedulability_test(rtos_task_t *tasks, int n) {
    double utilization = 0;
    for (int i = 0; i < n; i++) {
        utilization += tasks[i].execution / tasks[i].period;
    }

    double bound = rm_utilization_bound(n);

    printf("Utilization: %.4f\n", utilization);
    printf("RM bound (n=%d): %.4f\n", n, bound);
    printf("Result: %s\n",
           utilization <= bound ? "SCHEDULABLE" : "INCONCLUSIVE");

    /* 참고: U > bound이면 여전히 스케줄 가능할 수 있음.
     * 확인하려면 정확한 분석(응답 시간 분석)이 필요. */
    return utilization <= bound;
}

/*
 * 응답 시간 분석 (RMS의 정확한 테스트):
 *   R_i = C_i + Σ_{j∈hp(i)} ⌈R_i / T_j⌉ · C_j
 *   R_i가 수렴할 때까지 반복. R_i ≤ D_i이면 스케줄 가능.
 */
double response_time_analysis(rtos_task_t *tasks, int n, int task_idx) {
    double r = tasks[task_idx].execution;
    double prev_r;

    for (int iter = 0; iter < 100; iter++) {
        prev_r = r;
        r = tasks[task_idx].execution;

        for (int j = 0; j < task_idx; j++) {
            r += ceil(prev_r / tasks[j].period) * tasks[j].execution;
        }

        if (fabs(r - prev_r) < 0.0001) break;
        if (r > tasks[task_idx].deadline) return r;  /* 미스! */
    }

    return r;
}

int main(void) {
    rtos_task_t tasks[] = {
        {1, 100.0, 20.0, 100.0, 1},  /* T=100, C=20 */
        {2, 150.0, 30.0, 150.0, 2},  /* T=150, C=30 */
        {3, 350.0, 80.0, 350.0, 3},  /* T=350, C=80 */
    };
    int n = 3;

    printf("=== Rate Monotonic Schedulability ===\n");
    rm_schedulability_test(tasks, n);

    printf("\n=== Response Time Analysis ===\n");
    for (int i = 0; i < n; i++) {
        double r = response_time_analysis(tasks, n, i);
        printf("Task %d: WCRT = %.1f, Deadline = %.1f -> %s\n",
               tasks[i].id, r, tasks[i].deadline,
               r <= tasks[i].deadline ? "OK" : "MISS");
    }

    return 0;
}
```

### 2.2 Earliest Deadline First (EDF)

```c
/*
 * EDF (Earliest Deadline First):
 *   - 동적 우선순위: 가장 가까운 데드라인 = 최고 우선순위
 *   - 단일 프로세서 선점형 스케줄링에서 최적
 *   - U ≤ 1.0 (100%)일 때만 스케줄 가능!
 *   - RMS보다 구현이 더 복잡
 */

int edf_schedulability_test(rtos_task_t *tasks, int n) {
    double utilization = 0;
    for (int i = 0; i < n; i++) {
        utilization += tasks[i].execution / tasks[i].period;
    }

    printf("EDF Utilization: %.4f\n", utilization);
    printf("EDF Bound: 1.0000\n");
    printf("Result: %s\n",
           utilization <= 1.0 ? "SCHEDULABLE" : "NOT SCHEDULABLE");

    return utilization <= 1.0;
}
```

---

## 3. 스케줄 가능성 분석

### 3.1 최악 실행 시간 (WCET)

```
WCET 분석:
  태스크가 실행하는 데 걸릴 수 있는 최대 시간 결정.

정적 분석:
  - 코드 경로, 루프 한계, 캐시 동작 분석
  - 보수적: 과대 추정할 수 있음
  - 도구: aiT, Bound-T, OTAWA

측정 기반:
  - 태스크를 여러 번 실행하고 최대값 기록
  - 과소 추정할 수 있음! (놓친 최악 경로)
  - 통계적 방법과 함께 사용

혼합:
  - 정적 분석과 측정을 결합
  - 복잡한 시스템에 가장 실용적
```

### 3.2 지터와 지연 시간 분석

```c
#include <stdio.h>
#include <time.h>
#include <unistd.h>

/*
 * 인터럽트 지연과 스케줄링 지터 측정.
 */
void measure_jitter(int n_samples) {
    struct timespec expected, actual;
    double jitters[10000];
    int count = 0;

    long interval_ns = 1000000;  /* 1 ms 목표 주기 */

    clock_gettime(CLOCK_MONOTONIC, &expected);

    for (int i = 0; i < n_samples && i < 10000; i++) {
        /* 다음 예상 깨어남 시간 계산 */
        expected.tv_nsec += interval_ns;
        if (expected.tv_nsec >= 1000000000L) {
            expected.tv_sec++;
            expected.tv_nsec -= 1000000000L;
        }

        /* 예상 시간까지 슬립 */
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &expected, NULL);

        /* 실제 깨어남 시간 측정 */
        clock_gettime(CLOCK_MONOTONIC, &actual);

        /* 지터 계산 (예상과의 차이) */
        double jitter = (actual.tv_sec - expected.tv_sec) * 1e6 +
                       (actual.tv_nsec - expected.tv_nsec) / 1e3;  /* μs */
        jitters[count++] = jitter;
    }

    /* 통계 */
    double min_j = jitters[0], max_j = jitters[0], sum = 0;
    for (int i = 0; i < count; i++) {
        if (jitters[i] < min_j) min_j = jitters[i];
        if (jitters[i] > max_j) max_j = jitters[i];
        sum += jitters[i];
    }

    printf("Jitter analysis (%d samples):\n", count);
    printf("  Min: %.2f μs\n", min_j);
    printf("  Max: %.2f μs\n", max_j);
    printf("  Avg: %.2f μs\n", sum / count);
}
```

---

## 4. 우선순위 역전

### 4.1 Mars Pathfinder 버그

```
우선순위 역전: 고우선순위 태스크가 저우선순위 태스크에 의해 블록됨.

고전적인 예: Mars Pathfinder (1997)

  태스크 H (높은 우선순위): 기상 데이터 수집
  태스크 M (중간 우선순위): 통신 태스크
  태스크 L (낮은 우선순위): 정보 버스 관리

  시퀀스:
  1. L이 공유 버스의 뮤텍스 획득
  2. H가 L을 선점하고, 같은 뮤텍스 획득 시도 -> 블록됨
  3. M이 L을 선점하고 실행 (뮤텍스 불필요)
  4. H는 L을 기다리지만, M이 실행 중이라 L이 실행 못 함!

  결과: 워치독 타이머 발동, 시스템 리셋.

  해결: 우선순위 상속 프로토콜
  L이 H가 필요한 뮤텍스를 보유하면, 일시적으로 L을 H의 우선순위로 올림.
```

### 4.2 우선순위 상속 구현

```c
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

/*
 * 우선순위 상속 프로토콜:
 * 고우선순위 태스크가 저우선순위 태스크가 보유한 뮤텍스에서 블록되면,
 * 저우선순위 태스크가 일시적으로 고우선순위를 상속.
 */

void setup_priority_inheritance(void) {
    pthread_mutex_t mutex;
    pthread_mutexattr_t attr;

    /* 우선순위 상속으로 뮤텍스 설정 */
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutexattr_destroy(&attr);

    printf("Mutex created with PTHREAD_PRIO_INHERIT\n");

    /*
     * 이제 고우선순위 스레드가 이 뮤텍스에서 블록되면,
     * 뮤텍스 보유자의 우선순위가 자동으로 올라감.
     *
     * 우선순위 천장 프로토콜 (대안):
     * 뮤텍스 우선순위를 획득할 수 있는 태스크의 최고 우선순위로 설정.
     * 뮤텍스 획득 시 즉시 태스크 우선순위가 올라감.
     */

    pthread_mutexattr_t ceil_attr;
    pthread_mutex_t ceil_mutex;
    pthread_mutexattr_init(&ceil_attr);
    pthread_mutexattr_setprotocol(&ceil_attr, PTHREAD_PRIO_PROTECT);
    pthread_mutexattr_setprioceiling(&ceil_attr, 90);  /* 최고 잠재적 사용자 */
    pthread_mutex_init(&ceil_mutex, &ceil_attr);
    pthread_mutexattr_destroy(&ceil_attr);

    printf("Mutex created with priority ceiling = 90\n");

    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&ceil_mutex);
}
```

---

## 5. FreeRTOS 개요

### 5.1 FreeRTOS 아키텍처

```
FreeRTOS: 세계에서 가장 인기 있는 RTOS (400억+ 다운로드)

아키텍처:
  ┌────────────────────────────┐
  │      애플리케이션 태스크     │
  ├────────────────────────────┤
  │  큐   세마포어    타이머     │
  ├────────────────────────────┤
  │     FreeRTOS 커널           │
  │  스케줄러  │  메모리 관리    │
  ├────────────────────────────┤
  │    하드웨어 추상화           │
  └────────────────────────────┘

주요 특징:
  - 선점형 또는 협력형 스케줄링
  - 설정 가능한 틱 레이트 (일반적으로 1 kHz)
  - 다양한 메모리 할당 방식
  - 태스크 알림 (경량 세마포어)
  - 스트림 및 메시지 버퍼
```

### 5.2 FreeRTOS 태스크 예제

```c
/* FreeRTOS 태스크 생성 및 스케줄링 예제 */
/* 의사 코드 - FreeRTOS SDK 필요 */

#include <stdio.h>

/* 시뮬레이션된 FreeRTOS 타입과 함수 */
typedef void* TaskHandle_t;
typedef unsigned long TickType_t;

#define pdMS_TO_TICKS(ms) ((ms) / 1)  /* 단순화 */
#define configMAX_PRIORITIES 5

/* 센서 읽기 태스크 - 100ms마다 실행 */
void sensor_task(void *params) {
    const TickType_t period = pdMS_TO_TICKS(100);

    while (1) {
        /* 센서 읽기 */
        int value = 0;  /* read_adc(); */
        printf("[Sensor] Reading: %d\n", value);

        /* 큐를 통해 처리 태스크로 전송 */
        /* xQueueSend(sensor_queue, &value, 0); */

        /* 다음 주기까지 대기 */
        /* vTaskDelayUntil(&last_wake, period); */
    }
}

/* 모터 제어 태스크 - 최고 우선순위, 10ms마다 실행 */
void motor_task(void *params) {
    const TickType_t period = pdMS_TO_TICKS(10);

    while (1) {
        /* 큐에서 설정값 읽기 */
        int setpoint = 0;
        /* xQueueReceive(motor_queue, &setpoint, 0); */

        /* PID 제어 계산 */
        int output = setpoint;  /* pid_compute(setpoint, current); */

        /* 모터 출력 적용 */
        /* set_pwm(output); */
        printf("[Motor] Output: %d\n", output);

        /* 다음 주기까지 대기 */
        /* vTaskDelayUntil(&last_wake, period); */
    }
}

/* 통신 태스크 - 최저 우선순위 */
void comm_task(void *params) {
    while (1) {
        /* 가능할 때 UART/WiFi로 데이터 전송 */
        printf("[Comm] Transmitting data...\n");

        /* vTaskDelay(pdMS_TO_TICKS(1000)); */
    }
}

int main(void) {
    printf("FreeRTOS Task Example (simulated)\n");

    /* 실제 FreeRTOS에서:
     * xTaskCreate(motor_task, "Motor", 256, NULL, 4, NULL);   // 최고
     * xTaskCreate(sensor_task, "Sensor", 256, NULL, 3, NULL);
     * xTaskCreate(comm_task, "Comm", 512, NULL, 1, NULL);     // 최저
     * vTaskStartScheduler();
     */

    sensor_task(NULL);  /* 시뮬레이션 실행 */
    return 0;
}
```

---

## 6. Zephyr RTOS

### 6.1 Zephyr 아키텍처

```
Zephyr: Linux Foundation의 현대적 RTOS

주요 차별점:
  - 500개 이상의 보드에 대한 네이티브 지원
  - 내장 네트워킹 (Bluetooth, WiFi, Thread, 6LoWPAN)
  - 하드웨어 설명을 위한 디바이스 트리 (Linux처럼)
  - CMake 기반 빌드 시스템
  - 메모리 보호 (MPU 지원)
  - POSIX 호환성 레이어

아키텍처:
  ┌──────────────────────────────────┐
  │        애플리케이션               │
  ├──────────────────────────────────┤
  │  네트워킹  │ Bluetooth │ USB    │
  ├──────────────────────────────────┤
  │  드라이버   │ 센서      │ GPIO   │
  ├──────────────────────────────────┤
  │       Zephyr 커널                │
  │  스레드  │ 스케줄링  │ IPC      │
  │  메모리  │ 타이머    │ 동기화   │
  ├──────────────────────────────────┤
  │    하드웨어 추상화 레이어         │
  └──────────────────────────────────┘
```

---

## 7. RTOS 설계 패턴

### 7.1 일반적인 RTOS 패턴

```
1. 주기적 태스크 패턴:
   태스크가 고정 간격으로 깨어남 (예: 10ms마다)
   용도: 센서 샘플링, 제어 루프

2. 이벤트 기반 패턴:
   태스크가 이벤트/인터럽트 발생까지 슬립
   용도: 버튼 누름, 네트워크 패킷 도착

3. 생산자-소비자:
   하나의 태스크가 데이터 생산, 다른 태스크가 소비
   메시지 큐로 연결
   용도: 센서 → 처리 → 액추에이터 파이프라인

4. 워치독 패턴:
   모니터링 태스크가 모든 다른 태스크를 주기적으로 확인
   어떤 태스크가 응답을 멈추면 시스템 리셋
   용도: 안전 필수 시스템

5. 더블 버퍼 패턴:
   생산자가 버퍼 A에 쓰는 동안 소비자가 버퍼 B를 읽음
   버퍼를 원자적으로 교체
   용도: DMA, 오디오 처리
```

---

## 8. 연습문제

### 연습문제 1: RMS 스케줄 가능성

태스크 세트의 스케줄 가능성을 분석하세요:
1. Rate Monotonic의 이용률 한계 테스트 구현
2. 정확한 응답 시간 분석 구현
3. 5개의 다른 태스크 세트로 테스트 (각 3-5개 태스크)
4. 이용률 테스트에서 "미결정"이지만 RTA로 스케줄 가능한 태스크 세트 찾기
5. 시각화: 하나의 하이퍼피리어드에 대한 RMS 스케줄의 간트 차트

### 연습문제 2: EDF 시뮬레이터

Earliest Deadline First 스케줄러 구축:
1. C로 EDF 스케줄링 시뮬레이터 구현
2. 다른 주기를 가진 주기적 태스크 지원
3. 태스크 실행을 보여주는 간트 차트 생성
4. RMS로는 불가능한 태스크 세트를 EDF가 스케줄링하는 것 시연
5. 이용률이 100%를 초과할 때 무슨 일이 일어나는지 보여주기 (데드라인 미스)

### 연습문제 3: 우선순위 역전 시연

우선순위 역전을 시연하고 해결하세요:
1. 뮤텍스를 공유하는 3개의 다른 우선순위 pthread 생성
2. 우선순위 역전 보여주기: 중간이 실행되는 동안 고우선순위 스레드가 블록
3. PTHREAD_PRIO_INHERIT로 해결
4. 측정: 우선순위 상속 유무에 따른 블록 시간
5. 우선순위 천장 프로토콜을 구현하고 비교

### 연습문제 4: RTOS 태스크 설계

완전한 RTOS 애플리케이션 설계:
1. 온도 컨트롤러를 위한 5개 태스크 명세: 센서, PID, 디스플레이, 통신, 워치독
2. 각각에 대해 주기, 우선순위, WCET 할당
3. RMS와 EDF로 스케줄 가능성 검증
4. 메시지 큐를 사용한 태스크 통신 구현
5. 워치독 모니터링 및 복구 추가

### 연습문제 5: 지터 측정

Linux에서 실시간 성능 측정:
1. clock_nanosleep을 사용한 주기적 태스크(1 kHz) 생성
2. 10,000회 반복에 걸쳐 지터 측정
3. 비교: 일반 Linux vs SCHED_FIFO vs SCHED_RR
4. 각각에 대한 지터 분포 그래프 (히스토그램)
5. 스레드 어피니티를 적용하고 개선 효과 측정

---

*레슨 22 끝*
