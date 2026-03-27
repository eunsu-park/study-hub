[이전: 마이크로커널 설계](./26_Microkernel_Design.md)

---

# 27. 현대 스케줄러

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Linux CFS (Completely Fair Scheduler)와 가상 런타임 개념을 설명할 수 있다
2. Linux 6.6에서 CFS를 대체한 EEVDF 스케줄러를 설명할 수 있다
3. 데드라인 스케줄링을 구현하고 그 보장을 분석할 수 있다
4. 이기종 코어(big.LITTLE)에 대한 스케줄링 과제를 이해할 수 있다
5. Linux, Windows, macOS의 현대 스케줄링 접근법을 비교할 수 있다

---

## 목차

1. [Linux 스케줄러의 진화](#1-linux-스케줄러의-진화)
2. [CFS: Completely Fair Scheduler](#2-cfs-completely-fair-scheduler)
3. [EEVDF: Earliest Eligible Virtual Deadline First](#3-eevdf-earliest-eligible-virtual-deadline-first)
4. [SCHED_DEADLINE](#4-sched_deadline)
5. [이기종 코어 스케줄링](#5-이기종-코어-스케줄링)
6. [에너지 인식 스케줄링](#6-에너지-인식-스케줄링)
7. [스케줄러 비교](#7-스케줄러-비교)
8. [연습문제](#8-연습문제)

---

## 1. Linux 스케줄러의 진화

### 1.1 타임라인

```
Linux 스케줄러 역사:

2.4 (2001): O(n) 스케줄러
  - 최고 우선순위를 찾기 위해 모든 태스크 스캔
  - 많은 프로세스에서 확장성 불량

2.6 (2003): O(1) 스케줄러 (Ingo Molnar)
  - 상수 시간 태스크 선택
  - 활성/만료 배열
  - 대화형 태스크를 위한 휴리스틱

2.6.23 (2007): CFS (Con Kolivas, Ingo Molnar)
  - 가상 런타임을 통한 공정 스케줄링
  - 레드-블랙 트리 O(log n)
  - 휴리스틱 불필요

6.6 (2023): EEVDF (Peter Zijlstra)
  - CFS 대체
  - 더 나은 지연 시간 보장
  - 가상 데드라인 개념
  - 더 간단한 파라미터 튜닝
```

---

## 2. CFS: Completely Fair Scheduler

### 2.1 가상 런타임 개념

```
CFS 핵심 아이디어: 각 태스크의 "가상 런타임"(vruntime) 추적.
  vruntime = 실제_런타임 / 가중치

  높은 가중치 (nice -20) → vruntime이 천천히 증가 → 더 많이 실행
  낮은 가중치 (nice 19)  → vruntime이 빠르게 증가 → 덜 실행

  항상 가장 낮은 vruntime을 가진 태스크를 선택.
  모든 태스크가 결국 같은 vruntime을 가지므로 이것이 "공정".

레드-블랙 트리:
  vruntime 기준으로 균형 BST에 태스크 정렬.
  가장 왼쪽 노드 선택 = 캐시된 포인터로 O(1).
  삽입/제거 = O(log n).

  vruntime 축 →
  ┌────┬─────┬──────┬──────┬────────┐
  │ T3 │ T1  │  T5  │  T2  │   T4   │
  │ 5ms│ 8ms │ 12ms │ 15ms │  20ms  │
  └────┴─────┴──────┴──────┴────────┘
   ↑
   다음 실행 (가장 낮은 vruntime)
```

### 2.2 CFS 구현 세부사항

```c
#include <stdio.h>
#include <stdlib.h>

/*
 * 단순화된 CFS 시뮬레이터.
 */

#define MAX_TASKS 100

typedef struct {
    int pid;
    double vruntime;     /* 가상 런타임 (ns) */
    int nice;            /* Nice 값 (-20 ~ 19) */
    double weight;       /* nice에서 유도 */
    int is_running;
} cfs_task_t;

/* Nice-to-weight 매핑 (단순화) */
double nice_to_weight(int nice) {
    /* 가중치는 nice 5단계마다 대략 2배 */
    /* nice 0 = weight 1024 */
    double base = 1024.0;
    return base * (1.0 / (1.0 + nice * 0.1));  /* 단순화 */
}

typedef struct {
    cfs_task_t tasks[MAX_TASKS];
    int n_tasks;
    double min_granularity;  /* 최소 타임 슬라이스 (ms) */
    double target_latency;   /* 스케줄링 주기 (ms) */
} cfs_scheduler_t;

void cfs_init(cfs_scheduler_t *sched) {
    sched->n_tasks = 0;
    sched->min_granularity = 0.75;  /* 0.75 ms */
    sched->target_latency = 6.0;    /* 6 ms */
}

void cfs_add_task(cfs_scheduler_t *sched, int pid, int nice) {
    cfs_task_t *task = &sched->tasks[sched->n_tasks++];
    task->pid = pid;
    task->nice = nice;
    task->weight = nice_to_weight(nice);
    task->vruntime = 0;
    task->is_running = 0;

    /* 새 태스크: 기아 방지를 위해 현재 최솟값으로 vruntime 설정 */
    double min_vruntime = 1e18;
    for (int i = 0; i < sched->n_tasks - 1; i++) {
        if (sched->tasks[i].vruntime < min_vruntime) {
            min_vruntime = sched->tasks[i].vruntime;
        }
    }
    if (sched->n_tasks > 1) {
        task->vruntime = min_vruntime;
    }
}

/* 가장 낮은 vruntime의 태스크 선택 (레드-블랙 트리의 가장 왼쪽) */
cfs_task_t *cfs_pick_next(cfs_scheduler_t *sched) {
    cfs_task_t *best = NULL;
    double min_vruntime = 1e18;

    for (int i = 0; i < sched->n_tasks; i++) {
        if (sched->tasks[i].vruntime < min_vruntime) {
            min_vruntime = sched->tasks[i].vruntime;
            best = &sched->tasks[i];
        }
    }

    return best;
}

/* 태스크의 타임 슬라이스 계산 */
double cfs_time_slice(cfs_scheduler_t *sched, cfs_task_t *task) {
    double total_weight = 0;
    for (int i = 0; i < sched->n_tasks; i++) {
        total_weight += sched->tasks[i].weight;
    }

    double slice = sched->target_latency * (task->weight / total_weight);

    /* 최소 단위 강제 */
    if (slice < sched->min_granularity) {
        slice = sched->min_granularity;
    }

    return slice;
}

/* 한 번의 스케줄링 라운드 시뮬레이션 */
void cfs_simulate(cfs_scheduler_t *sched, int rounds) {
    printf("CFS Simulation (%d rounds):\n", rounds);
    printf("%-5s %-8s %-10s %-10s\n", "Round", "PID", "Slice(ms)", "VRuntime");

    for (int r = 0; r < rounds; r++) {
        cfs_task_t *task = cfs_pick_next(sched);
        if (!task) break;

        double slice = cfs_time_slice(sched, task);

        /* vruntime 업데이트: 가중치의 역수로 가중 */
        double vruntime_delta = slice * (1024.0 / task->weight);
        task->vruntime += vruntime_delta;

        printf("%-5d %-8d %-10.2f %-10.2f\n",
               r + 1, task->pid, slice, task->vruntime);
    }
}

int main(void) {
    cfs_scheduler_t sched;
    cfs_init(&sched);

    cfs_add_task(&sched, 1, 0);    /* 일반 우선순위 */
    cfs_add_task(&sched, 2, -5);   /* 더 높은 우선순위 */
    cfs_add_task(&sched, 3, 10);   /* 더 낮은 우선순위 */

    cfs_simulate(&sched, 15);
    return 0;
}
```

---

## 3. EEVDF: Earliest Eligible Virtual Deadline First

### 3.1 EEVDF 개념

```
EEVDF는 가상 데드라인을 추가하여 CFS를 개선:

CFS 문제:
  정렬에 vruntime만 사용.
  "낮은 지연이 필요한 것"과 "처리량이 필요한 것"을 구별할 수 없음

EEVDF 추가:
  가상 데드라인 = vruntime + (time_slice / weight)

  태스크는 vruntime ≤ min_vruntime일 때 "적격"
  적격 태스크 중에서 가장 이른 데드라인을 가진 것을 선택

  이것은 자연스럽게:
  - 짧은 태스크: 이른 데드라인 → 낮은 지연
  - 긴 태스크: 늦은 데드라인 → 좋은 처리량
  - CFS의 "깨어남 선점" 휴리스틱 불필요

EEVDF 선택:
  1. 필터: 적격 태스크만 (공정한 몫 확보)
  2. 적격 중: 가장 이른 가상 데드라인 선택
  3. 결과: 지연에 민감한 태스크가 빠르게 처리됨
```

### 3.2 EEVDF vs CFS 비교

```
특성              | CFS              | EEVDF
-----------------|------------------|------------------
선택 기준         | 가장 낮은 vruntime| 가장 이른 적격 데드라인
지연 시간 제어    | 휴리스틱          | 알고리즘에 내장
선점              | 깨어남 휴리스틱   | 데드라인 비교
튜닝 파라미터     | 많은 sysctl      | 더 적은 파라미터
공정성            | 좋음              | 더 좋음 (증명 가능)
지연 시간         | 가변적            | 더 예측 가능
복잡도            | 중간              | 약간 높음
Linux 버전        | 2.6.23 - 6.5     | 6.6+
```

---

## 4. SCHED_DEADLINE

### 4.1 Linux의 EDF

```c
#include <stdio.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>

/*
 * SCHED_DEADLINE: Linux의 EDF 기반 실시간 스케줄러.
 *
 * 각 태스크가 지정:
 *   Runtime:  C - 주기당 WCET
 *   Deadline: D - 상대 데드라인
 *   Period:   T - 최소 도착 간격
 *
 * 커널 보장: 태스크가 T 마이크로초마다 C 마이크로초를 받고,
 * 해제 후 D 마이크로초 내에 완료.
 *
 * 입장 제어: 시스템이 과부하되면 새 태스크 거부.
 */

struct sched_attr {
    unsigned int size;
    unsigned int sched_policy;
    unsigned long long sched_flags;
    int sched_nice;
    unsigned int sched_priority;
    unsigned long long sched_runtime;    /* ns */
    unsigned long long sched_deadline;   /* ns */
    unsigned long long sched_period;     /* ns */
};

int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags) {
    return syscall(SYS_sched_setattr, pid, attr, flags);
}

void set_deadline_scheduling(void) {
    struct sched_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.sched_policy = 6;  /* SCHED_DEADLINE */

    /* 10ms 런타임, 30ms 데드라인, 30ms 주기 */
    attr.sched_runtime  = 10 * 1000 * 1000;  /* 10 ms (ns 단위) */
    attr.sched_deadline = 30 * 1000 * 1000;  /* 30 ms (ns 단위) */
    attr.sched_period   = 30 * 1000 * 1000;  /* 30 ms (ns 단위) */

    int ret = sched_setattr(0, &attr, 0);
    if (ret != 0) {
        perror("sched_setattr");
        printf("Note: requires root or CAP_SYS_NICE\n");
    } else {
        printf("SCHED_DEADLINE set: runtime=10ms, deadline=30ms, period=30ms\n");
    }
}
```

---

## 5. 이기종 코어 스케줄링

### 5.1 big.LITTLE / 하이브리드 아키텍처

```
현대 CPU는 다른 유형의 코어를 가짐:

ARM big.LITTLE:
  Big 코어 (Cortex-A78):    높은 성능, 높은 전력
  LITTLE 코어 (Cortex-A55): 낮은 성능, 낮은 전력

Intel 하이브리드 (Alder Lake+):
  P-코어 (성능):  높은 IPC, 하이퍼스레딩
  E-코어 (효율):  낮은 IPC, 낮은 전력, HT 없음

스케줄링 과제:
  어떤 태스크를 어떤 코어에?

  컴퓨팅 집중 → Big/P-코어 (성능 극대화)
  백그라운드 태스크 → LITTLE/E-코어 (전력 절약)
  지연에 민감 → Big/P-코어 (빠른 응답)
  배치 처리 → E-코어 (전력 효율)

Linux 해결책: Energy-Aware Scheduling (EAS)
  CPU 용량과 에너지 모델을 사용하여 결정.
```

### 5.2 태스크 배치 결정

```c
/*
 * 단순화된 이기종 스케줄러.
 */

typedef enum {
    CORE_BIG,
    CORE_LITTLE,
} core_type_t;

typedef struct {
    int id;
    core_type_t type;
    int capacity;      /* 상대적 연산 용량 */
    int power_cost;    /* 상대적 전력 소비 */
    int current_load;  /* 현재 사용률 (0-1024) */
} cpu_core_t;

typedef struct {
    int pid;
    int utilization;   /* CPU 사용률 (0-1024) */
    int latency_req;   /* 1 = 지연에 민감, 0 = 처리량 */
} sched_task_t;

cpu_core_t *select_core(cpu_core_t *cores, int n_cores,
                         sched_task_t *task) {
    cpu_core_t *best = NULL;
    int best_score = -1;

    for (int i = 0; i < n_cores; i++) {
        int available = cores[i].capacity - cores[i].current_load;
        if (available < task->utilization) continue;

        int score = 0;

        if (task->latency_req) {
            /* 지연에 민감한 태스크에는 big 코어 선호 */
            score = cores[i].capacity * 2 - cores[i].power_cost;
        } else {
            /* 백그라운드 태스크에는 효율 코어 선호 */
            score = cores[i].capacity - cores[i].power_cost * 2;
        }

        if (score > best_score) {
            best_score = score;
            best = &cores[i];
        }
    }

    return best;
}
```

---

## 6. 에너지 인식 스케줄링

### 6.1 Linux의 EAS

```
Energy-Aware Scheduling (EAS):

목표: 성능 요구를 충족하면서 에너지 최소화.

CPU별 에너지 모델:
  에너지 = Σ (용량 × 해당_용량_전력 × 시간)

EAS 결정:
  각 태스크 깨어남 시 계산:
  1. big 코어에 배치할 때의 에너지
  2. LITTLE 코어에 배치할 때의 에너지
  3. 성능을 충족하는 가장 낮은 에너지 옵션 선택

  또한 고려:
  - 태스크 사용률 이력 (PELT: Per-Entity Load Tracking)
  - CPU 주파수 (DVFS 통합)
  - 열적 제약
  - 마이그레이션 비용 (캐시 워밍업)
```

---

## 7. 스케줄러 비교

### 7.1 운영체제 간 비교

```
운영체제별 스케줄러 비교:

Linux (EEVDF):
  - SCHED_NORMAL에 대한 공정성 기반
  - SCHED_FIFO/RR에 대한 우선순위 기반
  - SCHED_DEADLINE에 대한 EDF
  - 모바일/노트북용 에너지 인식

Windows:
  - 32단계 우선순위 기반
  - 대화형 태스크에 동적 우선순위 부스팅
  - 포그라운드 프로세스 우대
  - 우선순위에 따라 스레드 퀀텀(타임 슬라이스) 변동

macOS:
  - Mach 스케줄러 (감쇄 사용량)
  - 스레드 서비스 품질(QoS) 레벨
  - QoS: UserInteractive > UserInitiated > Utility > Background
  - QoS 기반 자동 승격/강등

실시간:
  - FreeRTOS: 고정 우선순위 선점형
  - Zephyr: EDF 지원이 있는 우선순위
  - QNX: 적응형 파티셔닝 + 우선순위
```

---

## 8. 연습문제

### 연습문제 1: CFS 시뮬레이터

CFS 시뮬레이터 구축:
1. 태스크 정렬을 위한 레드-블랙 트리(또는 정렬 리스트) 구현
2. 지원: nice 값 (-20 ~ 19), 동적 태스크 추가/제거
3. 다양한 nice 값의 10개 태스크로 1초 시뮬레이션
4. 공정성 검증: 각 태스크가 가중치에 비례한 CPU를 받는지
5. 그래프: 각 태스크의 실제 런타임 vs 예상 런타임

### 연습문제 2: EEVDF 시뮬레이터

EEVDF 스케줄링 구현:
1. CFS 시뮬레이터에 가상 데드라인 계산 추가
2. 적격성 검사 구현
3. 태스크 선택 비교: CFS (vruntime) vs EEVDF (적격 + 데드라인)
4. 지연에 민감한 태스크와 배치 태스크가 있는 워크로드 생성
5. EEVDF가 대화형 태스크에 더 나은 지연 시간을 제공함을 보여주기

### 연습문제 3: SCHED_DEADLINE 실험

Linux SCHED_DEADLINE 사용:
1. 주기적 태스크 작성 (예: 5ms 런타임, 20ms 주기)
2. SCHED_DEADLINE 파라미터 설정 후 입장 확인
3. 실제 실행 시간과 데드라인 준수 측정
4. 경쟁하는 SCHED_NORMAL 태스크를 추가하고 격리 확인
5. 테스트: 런타임이 예산을 초과하면 어떻게 되는가?

### 연습문제 4: 이기종 스케줄링

big.LITTLE 스케줄링 시뮬레이션:
1. 4개의 big 코어(capacity=1024)와 4개의 LITTLE 코어(capacity=512) 모델링
2. 다양한 사용률과 지연 요구사항의 20개 태스크 생성
3. 3가지 전략 구현: big 전용, LITTLE 전용, 이기종 인식
4. 비교: 성능, 전력 소비, 공정성
5. 이기종 인식이 최고의 에너지-성능 비율을 달성함을 보여주기

### 연습문제 5: 스케줄링 오버헤드 측정

실제 스케줄러 동작 측정:
1. 컨텍스트 스위치 지연 시간을 측정하는 벤치마크 작성
2. 다른 태스크 수로 테스트: 1, 10, 100, 1000
3. SCHED_NORMAL, SCHED_FIFO, SCHED_RR 비교
4. perf를 사용하여 스케줄러 오버헤드 측정
5. 그래프: 각 스케줄러 클래스의 지연 시간 분포

---

*레슨 27 끝*
