[Previous: Disk Scheduling and I/O](./21_Disk_Scheduling_IO.md)

---

# 22. Real-Time Operating Systems

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish hard, firm, and soft real-time requirements
2. Explain Rate Monotonic and Earliest Deadline First scheduling
3. Describe FreeRTOS and Zephyr architectures for embedded systems
4. Implement priority inversion solutions including priority inheritance
5. Analyze task schedulability and worst-case execution time

---

## Table of Contents

1. [Real-Time Concepts](#1-real-time-concepts)
2. [RTOS Scheduling Algorithms](#2-rtos-scheduling-algorithms)
3. [Schedulability Analysis](#3-schedulability-analysis)
4. [Priority Inversion](#4-priority-inversion)
5. [FreeRTOS Overview](#5-freertos-overview)
6. [Zephyr RTOS](#6-zephyr-rtos)
7. [RTOS Design Patterns](#7-rtos-design-patterns)
8. [Exercises](#8-exercises)

---

## 1. Real-Time Concepts

### 1.1 Real-Time Classification

```
Real-Time Systems:
  Correctness depends on BOTH the result AND the timing.

Hard Real-Time:
  Deadline miss = system failure (catastrophic)
  Examples: airbag deployment, flight control, pacemaker
  Guarantee: Mathematical proof that deadlines will be met

Firm Real-Time:
  Deadline miss = result is useless (but not catastrophic)
  Examples: video frame rendering, financial trading
  Late results are discarded

Soft Real-Time:
  Deadline miss = degraded quality (still somewhat useful)
  Examples: audio streaming, web server response
  Late results have reduced value
```

### 1.2 Key RTOS Concepts

```
RTOS vs General-Purpose OS:

Feature          | RTOS            | General-Purpose OS
-----------------|-----------------|-----------------
Scheduling       | Priority-based  | Fairness-based
Latency          | Deterministic   | Best-effort
Interrupt latency| < 10 μs        | 100 μs - 10 ms
Context switch   | < 5 μs         | 1-100 μs
Memory           | Static/small    | Dynamic/large
Response time    | Guaranteed      | Statistical
```

---

## 2. RTOS Scheduling Algorithms

### 2.1 Rate Monotonic Scheduling (RMS)

```c
#include <stdio.h>
#include <math.h>

/*
 * Rate Monotonic Scheduling:
 *   - Static priority assignment
 *   - Shorter period = higher priority
 *   - Optimal among fixed-priority preemptive schedulers
 *
 *   Utilization bound (Liu & Layland):
 *   U = Σ (Ci/Ti) ≤ n(2^(1/n) - 1)
 *
 *   n=1: 100%, n=2: 82.8%, n=3: 78.0%, n→∞: 69.3% (ln 2)
 */

typedef struct {
    int id;
    double period;      /* T: minimum inter-arrival time */
    double execution;   /* C: worst-case execution time */
    double deadline;    /* D: relative deadline (= period for RMS) */
    int priority;       /* Lower number = higher priority */
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

    /* Note: If U > bound, may still be schedulable.
     * Need exact analysis (response time analysis) to confirm. */
    return utilization <= bound;
}

/*
 * Response Time Analysis (exact test for RMS):
 *   R_i = C_i + Σ_{j∈hp(i)} ⌈R_i / T_j⌉ · C_j
 *   Iterate until R_i converges. Schedulable if R_i ≤ D_i.
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
        if (r > tasks[task_idx].deadline) return r;  /* Miss! */
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
 *   - Dynamic priority: closest deadline = highest priority
 *   - Optimal for uniprocessor preemptive scheduling
 *   - Schedulable if and only if U ≤ 1.0 (100%!)
 *   - More complex to implement than RMS
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

## 3. Schedulability Analysis

### 3.1 Worst-Case Execution Time (WCET)

```
WCET Analysis:
  Determining the maximum time a task can take to execute.

Static analysis:
  - Analyze code paths, loop bounds, cache behavior
  - Conservative: may overestimate
  - Tools: aiT, Bound-T, OTAWA

Measurement-based:
  - Run task many times, record maximum
  - May underestimate! (missed worst-case paths)
  - Use with statistical methods

Hybrid:
  - Combine static analysis with measurements
  - Most practical for complex systems
```

### 3.2 Jitter and Latency Analysis

```c
#include <stdio.h>
#include <time.h>
#include <unistd.h>

/*
 * Measure interrupt latency and scheduling jitter.
 */
void measure_jitter(int n_samples) {
    struct timespec expected, actual;
    double jitters[10000];
    int count = 0;

    long interval_ns = 1000000;  /* 1 ms target period */

    clock_gettime(CLOCK_MONOTONIC, &expected);

    for (int i = 0; i < n_samples && i < 10000; i++) {
        /* Calculate next expected wakeup */
        expected.tv_nsec += interval_ns;
        if (expected.tv_nsec >= 1000000000L) {
            expected.tv_sec++;
            expected.tv_nsec -= 1000000000L;
        }

        /* Sleep until expected time */
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &expected, NULL);

        /* Measure actual wakeup time */
        clock_gettime(CLOCK_MONOTONIC, &actual);

        /* Calculate jitter (difference from expected) */
        double jitter = (actual.tv_sec - expected.tv_sec) * 1e6 +
                       (actual.tv_nsec - expected.tv_nsec) / 1e3;  /* μs */
        jitters[count++] = jitter;
    }

    /* Statistics */
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

## 4. Priority Inversion

### 4.1 The Mars Pathfinder Bug

```
Priority Inversion: A high-priority task is blocked by a low-priority task.

The classic example: Mars Pathfinder (1997)

  Task H (High priority): Meteorological data collection
  Task M (Medium priority): Communication task
  Task L (Low priority): Information bus management

  Sequence:
  1. L acquires mutex on shared bus
  2. H preempts L, tries to acquire same mutex -> BLOCKED
  3. M preempts L and runs (doesn't need mutex)
  4. H is stuck waiting for L, but L can't run because M is running!

  Result: Watchdog timer fires, system resets.

  Solution: Priority Inheritance Protocol
  When L holds a mutex that H needs, temporarily boost L to H's priority.
```

### 4.2 Priority Inheritance Implementation

```c
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

/*
 * Priority Inheritance Protocol:
 * When a high-priority task blocks on a mutex held by a low-priority task,
 * the low-priority task temporarily inherits the high priority.
 */

void setup_priority_inheritance(void) {
    pthread_mutex_t mutex;
    pthread_mutexattr_t attr;

    /* Set up mutex with priority inheritance */
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutexattr_destroy(&attr);

    printf("Mutex created with PTHREAD_PRIO_INHERIT\n");

    /*
     * Now when a high-priority thread blocks on this mutex,
     * the mutex holder's priority is automatically boosted.
     *
     * Priority Ceiling Protocol (alternative):
     * Set mutex priority to the highest priority of any task
     * that might acquire it. Task's priority is boosted
     * immediately when acquiring the mutex.
     */

    pthread_mutexattr_t ceil_attr;
    pthread_mutex_t ceil_mutex;
    pthread_mutexattr_init(&ceil_attr);
    pthread_mutexattr_setprotocol(&ceil_attr, PTHREAD_PRIO_PROTECT);
    pthread_mutexattr_setprioceiling(&ceil_attr, 90);  /* Highest potential user */
    pthread_mutex_init(&ceil_mutex, &ceil_attr);
    pthread_mutexattr_destroy(&ceil_attr);

    printf("Mutex created with priority ceiling = 90\n");

    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&ceil_mutex);
}
```

---

## 5. FreeRTOS Overview

### 5.1 FreeRTOS Architecture

```
FreeRTOS: Most popular RTOS worldwide (40B+ downloads)

Architecture:
  ┌────────────────────────────┐
  │      Application Tasks     │
  ├────────────────────────────┤
  │  Queues  Semaphores Timers │
  ├────────────────────────────┤
  │     FreeRTOS Kernel        │
  │  Scheduler │ Memory Mgmt   │
  ├────────────────────────────┤
  │    Hardware Abstraction    │
  └────────────────────────────┘

Key features:
  - Preemptive or cooperative scheduling
  - Configurable tick rate (typically 1 kHz)
  - Multiple memory allocation schemes
  - Task notifications (lightweight semaphores)
  - Stream and message buffers
```

### 5.2 FreeRTOS Task Example

```c
/* FreeRTOS task creation and scheduling example */
/* This is pseudocode - requires FreeRTOS SDK */

#include <stdio.h>

/* Simulated FreeRTOS types and functions */
typedef void* TaskHandle_t;
typedef unsigned long TickType_t;

#define pdMS_TO_TICKS(ms) ((ms) / 1)  /* Simplified */
#define configMAX_PRIORITIES 5

/* Sensor reading task - runs every 100ms */
void sensor_task(void *params) {
    const TickType_t period = pdMS_TO_TICKS(100);

    while (1) {
        /* Read sensor */
        int value = 0;  /* read_adc(); */
        printf("[Sensor] Reading: %d\n", value);

        /* Send to processing task via queue */
        /* xQueueSend(sensor_queue, &value, 0); */

        /* Wait for next period */
        /* vTaskDelayUntil(&last_wake, period); */
    }
}

/* Motor control task - highest priority, runs every 10ms */
void motor_task(void *params) {
    const TickType_t period = pdMS_TO_TICKS(10);

    while (1) {
        /* Read setpoint from queue */
        int setpoint = 0;
        /* xQueueReceive(motor_queue, &setpoint, 0); */

        /* PID control calculation */
        int output = setpoint;  /* pid_compute(setpoint, current); */

        /* Apply motor output */
        /* set_pwm(output); */
        printf("[Motor] Output: %d\n", output);

        /* Wait for next period */
        /* vTaskDelayUntil(&last_wake, period); */
    }
}

/* Communication task - lowest priority */
void comm_task(void *params) {
    while (1) {
        /* Send data over UART/WiFi when available */
        printf("[Comm] Transmitting data...\n");

        /* vTaskDelay(pdMS_TO_TICKS(1000)); */
    }
}

int main(void) {
    printf("FreeRTOS Task Example (simulated)\n");

    /* In real FreeRTOS:
     * xTaskCreate(motor_task, "Motor", 256, NULL, 4, NULL);   // Highest
     * xTaskCreate(sensor_task, "Sensor", 256, NULL, 3, NULL);
     * xTaskCreate(comm_task, "Comm", 512, NULL, 1, NULL);     // Lowest
     * vTaskStartScheduler();
     */

    sensor_task(NULL);  /* Simulated run */
    return 0;
}
```

---

## 6. Zephyr RTOS

### 6.1 Zephyr Architecture

```
Zephyr: Modern RTOS from the Linux Foundation

Key differentiators:
  - Native support for 500+ boards
  - Built-in networking (Bluetooth, WiFi, Thread, 6LoWPAN)
  - Device tree for hardware description (like Linux)
  - CMake-based build system
  - Memory protection (MPU support)
  - POSIX compatibility layer

Architecture:
  ┌──────────────────────────────────┐
  │        Application               │
  ├──────────────────────────────────┤
  │  Networking │ Bluetooth │ USB    │
  ├──────────────────────────────────┤
  │  Drivers    │ Sensors   │ GPIO   │
  ├──────────────────────────────────┤
  │       Zephyr Kernel               │
  │  Threads │ Scheduling │ IPC      │
  │  Memory  │ Timers     │ Sync     │
  ├──────────────────────────────────┤
  │    Hardware Abstraction Layer     │
  └──────────────────────────────────┘
```

---

## 7. RTOS Design Patterns

### 7.1 Common RTOS Patterns

```
1. Periodic Task Pattern:
   Task wakes at fixed intervals (e.g., every 10ms)
   Use: sensor sampling, control loops

2. Event-Driven Pattern:
   Task sleeps until event/interrupt occurs
   Use: button press, network packet arrival

3. Producer-Consumer:
   One task produces data, another consumes
   Connected by message queue
   Use: sensor → processing → actuator pipeline

4. Watchdog Pattern:
   Monitoring task checks all other tasks periodically
   Resets system if any task stops responding
   Use: safety-critical systems

5. Double-Buffer Pattern:
   Producer writes to buffer A while consumer reads buffer B
   Swap buffers atomically
   Use: DMA, audio processing
```

---

## 8. Exercises

### Exercise 1: RMS Schedulability

Analyze schedulability for task sets:
1. Implement utilization bound test for Rate Monotonic
2. Implement exact Response Time Analysis
3. Test with 5 different task sets (3-5 tasks each)
4. Find a task set where utilization test says "inconclusive" but RTA confirms schedulable
5. Visualize: Gantt chart of RMS schedule for one hyperperiod

### Exercise 2: EDF Simulator

Build an Earliest Deadline First scheduler:
1. Implement EDF scheduling simulator in C
2. Support periodic tasks with different periods
3. Generate Gantt chart showing task execution
4. Demonstrate EDF scheduling a task set that RMS cannot
5. Show what happens when utilization exceeds 100% (deadline miss)

### Exercise 3: Priority Inversion Demonstration

Demonstrate and fix priority inversion:
1. Create 3 pthreads with different priorities sharing a mutex
2. Show priority inversion: high-priority thread blocked while medium runs
3. Fix with PTHREAD_PRIO_INHERIT
4. Measure: blocked time with and without priority inheritance
5. Implement priority ceiling protocol and compare

### Exercise 4: RTOS Task Design

Design a complete RTOS application:
1. Specify 5 tasks for a temperature controller: sensor, PID, display, comm, watchdog
2. Assign periods, priorities, and WCET for each
3. Verify schedulability with RMS and EDF
4. Implement task communication using message queues
5. Add watchdog monitoring and recovery

### Exercise 5: Jitter Measurement

Measure real-time performance on Linux:
1. Create a periodic task (1 kHz) using clock_nanosleep
2. Measure jitter over 10,000 iterations
3. Compare: normal Linux vs SCHED_FIFO vs SCHED_RR
4. Plot jitter distribution (histogram) for each
5. Apply thread affinity and measure improvement

---

*End of Lesson 22*
