[Previous: Microkernel Design](./26_Microkernel_Design.md)

---

# 27. Modern Schedulers

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Linux CFS (Completely Fair Scheduler) and its virtual runtime concept
2. Describe the EEVDF scheduler that replaced CFS in Linux 6.6
3. Implement deadline scheduling and analyze its guarantees
4. Understand scheduling challenges for heterogeneous cores (big.LITTLE)
5. Compare modern scheduling approaches across Linux, Windows, and macOS

---

## Table of Contents

1. [Evolution of Linux Schedulers](#1-evolution-of-linux-schedulers)
2. [CFS: Completely Fair Scheduler](#2-cfs-completely-fair-scheduler)
3. [EEVDF: Earliest Eligible Virtual Deadline First](#3-eevdf-earliest-eligible-virtual-deadline-first)
4. [SCHED_DEADLINE](#4-sched_deadline)
5. [Heterogeneous Core Scheduling](#5-heterogeneous-core-scheduling)
6. [Energy-Aware Scheduling](#6-energy-aware-scheduling)
7. [Scheduler Comparison](#7-scheduler-comparison)
8. [Exercises](#8-exercises)

---

## 1. Evolution of Linux Schedulers

### 1.1 Timeline

```
Linux Scheduler History:

2.4 (2001): O(n) scheduler
  - Scan all tasks to find highest priority
  - Poor scaling with many processes

2.6 (2003): O(1) scheduler (Ingo Molnar)
  - Constant-time task selection
  - Active/expired arrays
  - Heuristics for interactive tasks

2.6.23 (2007): CFS (Con Kolivas, Ingo Molnar)
  - Fair scheduling via virtual runtime
  - Red-black tree O(log n)
  - No heuristics needed

6.6 (2023): EEVDF (Peter Zijlstra)
  - Replaces CFS
  - Better latency guarantees
  - Virtual deadline concept
  - Simpler parameter tuning
```

---

## 2. CFS: Completely Fair Scheduler

### 2.1 Virtual Runtime Concept

```
CFS Key Idea: Track "virtual runtime" (vruntime) for each task.
  vruntime = actual_runtime / weight

  Higher weight (nice -20) → vruntime grows slowly → runs more
  Lower weight (nice 19)  → vruntime grows quickly → runs less

  Always pick the task with LOWEST vruntime.
  This is "fair" because all tasks eventually get equal vruntime.

Red-Black Tree:
  Tasks sorted by vruntime in a balanced BST.
  Pick leftmost node = O(1) with cached pointer.
  Insert/remove = O(log n).

  vruntime axis →
  ┌────┬─────┬──────┬──────┬────────┐
  │ T3 │ T1  │  T5  │  T2  │   T4   │
  │ 5ms│ 8ms │ 12ms │ 15ms │  20ms  │
  └────┴─────┴──────┴──────┴────────┘
   ↑
   Next to run (lowest vruntime)
```

### 2.2 CFS Implementation Details

```c
#include <stdio.h>
#include <stdlib.h>

/*
 * Simplified CFS simulator.
 */

#define MAX_TASKS 100

typedef struct {
    int pid;
    double vruntime;     /* Virtual runtime (ns) */
    int nice;            /* Nice value (-20 to 19) */
    double weight;       /* Derived from nice */
    int is_running;
} cfs_task_t;

/* Nice-to-weight mapping (simplified) */
double nice_to_weight(int nice) {
    /* Weight roughly doubles every 5 nice levels */
    /* nice 0 = weight 1024 */
    double base = 1024.0;
    return base * (1.0 / (1.0 + nice * 0.1));  /* Simplified */
}

typedef struct {
    cfs_task_t tasks[MAX_TASKS];
    int n_tasks;
    double min_granularity;  /* Minimum time slice (ms) */
    double target_latency;   /* Scheduling period (ms) */
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

    /* New task: set vruntime to current minimum to avoid starvation */
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

/* Pick task with lowest vruntime (leftmost in red-black tree) */
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

/* Calculate time slice for a task */
double cfs_time_slice(cfs_scheduler_t *sched, cfs_task_t *task) {
    double total_weight = 0;
    for (int i = 0; i < sched->n_tasks; i++) {
        total_weight += sched->tasks[i].weight;
    }

    double slice = sched->target_latency * (task->weight / total_weight);

    /* Enforce minimum granularity */
    if (slice < sched->min_granularity) {
        slice = sched->min_granularity;
    }

    return slice;
}

/* Simulate one scheduling round */
void cfs_simulate(cfs_scheduler_t *sched, int rounds) {
    printf("CFS Simulation (%d rounds):\n", rounds);
    printf("%-5s %-8s %-10s %-10s\n", "Round", "PID", "Slice(ms)", "VRuntime");

    for (int r = 0; r < rounds; r++) {
        cfs_task_t *task = cfs_pick_next(sched);
        if (!task) break;

        double slice = cfs_time_slice(sched, task);

        /* Update vruntime: weighted by inverse of weight */
        double vruntime_delta = slice * (1024.0 / task->weight);
        task->vruntime += vruntime_delta;

        printf("%-5d %-8d %-10.2f %-10.2f\n",
               r + 1, task->pid, slice, task->vruntime);
    }
}

int main(void) {
    cfs_scheduler_t sched;
    cfs_init(&sched);

    cfs_add_task(&sched, 1, 0);    /* Normal priority */
    cfs_add_task(&sched, 2, -5);   /* Higher priority */
    cfs_add_task(&sched, 3, 10);   /* Lower priority */

    cfs_simulate(&sched, 15);
    return 0;
}
```

---

## 3. EEVDF: Earliest Eligible Virtual Deadline First

### 3.1 EEVDF Concept

```
EEVDF improves on CFS by adding virtual deadlines:

CFS problem:
  Only uses vruntime for ordering.
  Can't distinguish between "needs low latency" and "needs throughput"

EEVDF adds:
  Virtual Deadline = vruntime + (time_slice / weight)

  A task is "eligible" when its vruntime ≤ min_vruntime
  Among eligible tasks, pick the one with EARLIEST deadline

  This naturally gives:
  - Short tasks: early deadlines → low latency
  - Long tasks: later deadlines → good throughput
  - No need for CFS's "wake-up preemption" heuristics

EEVDF Selection:
  1. Filter: only eligible tasks (fair share earned)
  2. Among eligible: pick earliest virtual deadline
  3. Result: latency-sensitive tasks get served quickly
```

### 3.2 EEVDF vs CFS Comparison

```
Feature          | CFS              | EEVDF
-----------------|------------------|------------------
Selection metric | Lowest vruntime  | Earliest eligible deadline
Latency control  | Heuristics       | Built into algorithm
Preemption       | wake-up heuristic| Deadline comparison
Tuning knobs     | Many sysctl      | Fewer parameters
Fairness         | Good             | Better (provable)
Latency          | Variable         | More predictable
Complexity       | Medium           | Slightly higher
Linux version    | 2.6.23 - 6.5     | 6.6+
```

---

## 4. SCHED_DEADLINE

### 4.1 EDF in Linux

```c
#include <stdio.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>

/*
 * SCHED_DEADLINE: Linux's EDF-based real-time scheduler.
 *
 * Each task specifies:
 *   Runtime:  C - WCET per period
 *   Deadline: D - relative deadline
 *   Period:   T - minimum inter-arrival time
 *
 * Kernel guarantees: task gets C microseconds every T microseconds,
 * completed within D microseconds of release.
 *
 * Admission control: new task rejected if system would be overloaded.
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

    /* 10ms runtime, 30ms deadline, 30ms period */
    attr.sched_runtime  = 10 * 1000 * 1000;  /* 10 ms in ns */
    attr.sched_deadline = 30 * 1000 * 1000;  /* 30 ms in ns */
    attr.sched_period   = 30 * 1000 * 1000;  /* 30 ms in ns */

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

## 5. Heterogeneous Core Scheduling

### 5.1 big.LITTLE / Hybrid Architectures

```
Modern CPUs have different core types:

ARM big.LITTLE:
  Big cores (Cortex-A78):   High performance, high power
  LITTLE cores (Cortex-A55): Low performance, low power

Intel Hybrid (Alder Lake+):
  P-cores (Performance):  High IPC, hyperthreading
  E-cores (Efficient):    Lower IPC, lower power, no HT

Scheduling challenge:
  Which tasks go on which cores?

  Compute-intensive → Big/P-core (maximize performance)
  Background tasks  → LITTLE/E-core (save power)
  Latency-sensitive → Big/P-core (fast response)
  Batch processing  → E-core (power efficiency)

Linux solution: Energy-Aware Scheduling (EAS)
  Uses CPU capacity and energy model to make decisions.
```

### 5.2 Task Placement Decisions

```c
/*
 * Simplified heterogeneous scheduler.
 */

typedef enum {
    CORE_BIG,
    CORE_LITTLE,
} core_type_t;

typedef struct {
    int id;
    core_type_t type;
    int capacity;      /* Relative compute capacity */
    int power_cost;    /* Relative power consumption */
    int current_load;  /* Current utilization (0-1024) */
} cpu_core_t;

typedef struct {
    int pid;
    int utilization;   /* CPU utilization (0-1024) */
    int latency_req;   /* 1 = latency-sensitive, 0 = throughput */
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
            /* Prefer big cores for latency-sensitive tasks */
            score = cores[i].capacity * 2 - cores[i].power_cost;
        } else {
            /* Prefer efficient cores for background tasks */
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

## 6. Energy-Aware Scheduling

### 6.1 EAS in Linux

```
Energy-Aware Scheduling (EAS):

Objective: minimize energy while meeting performance needs.

Energy model per CPU:
  Energy = Σ (capacity × power_at_capacity × time)

EAS decision:
  For each task wakeup, compute:
  1. Energy if placed on big core
  2. Energy if placed on LITTLE core
  3. Choose lowest energy option that meets performance

  Also considers:
  - Task utilization history (PELT: Per-Entity Load Tracking)
  - CPU frequency (DVFS integration)
  - Thermal constraints
  - Migration cost (cache warm-up)
```

---

## 7. Scheduler Comparison

### 7.1 Cross-OS Comparison

```
Scheduler comparison across operating systems:

Linux (EEVDF):
  - Fairness-based for SCHED_NORMAL
  - Priority-based for SCHED_FIFO/RR
  - EDF for SCHED_DEADLINE
  - Energy-aware for mobile/laptop

Windows:
  - Priority-based with 32 levels
  - Dynamic priority boosting for interactive tasks
  - Favor foreground processes
  - Thread quantum (time slice) varies by priority

macOS:
  - Mach scheduler (decay-usage)
  - Thread Quality of Service (QoS) levels
  - QoS: UserInteractive > UserInitiated > Utility > Background
  - Automatic promotion/demotion based on QoS

Real-time:
  - FreeRTOS: Fixed priority preemptive
  - Zephyr: Priority with EDF support
  - QNX: Adaptive partitioning + priority
```

---

## 8. Exercises

### Exercise 1: CFS Simulator

Build a CFS simulator:
1. Implement red-black tree (or sorted list) for task ordering
2. Support: nice values (-20 to 19), dynamic task add/remove
3. Simulate 1 second with 10 tasks of varying nice values
4. Verify fairness: each task gets CPU proportional to weight
5. Plot: actual runtime vs expected runtime for each task

### Exercise 2: EEVDF Simulator

Implement EEVDF scheduling:
1. Add virtual deadline calculation to CFS simulator
2. Implement eligibility check
3. Compare task selection: CFS (vruntime) vs EEVDF (eligible + deadline)
4. Create workload with latency-sensitive and batch tasks
5. Show that EEVDF gives better latency to interactive tasks

### Exercise 3: SCHED_DEADLINE Experiment

Use Linux SCHED_DEADLINE:
1. Write a periodic task (e.g., 5ms runtime, 20ms period)
2. Set SCHED_DEADLINE parameters and verify admission
3. Measure actual execution time and deadline adherence
4. Add competing SCHED_NORMAL tasks and verify isolation
5. Test: what happens when runtime exceeds budget?

### Exercise 4: Heterogeneous Scheduling

Simulate big.LITTLE scheduling:
1. Model 4 big cores (capacity=1024) and 4 LITTLE cores (capacity=512)
2. Create 20 tasks with varying utilization and latency requirements
3. Implement 3 strategies: big-only, LITTLE-only, heterogeneous-aware
4. Compare: performance, power consumption, fairness
5. Show that heterogeneous-aware achieves best energy-performance ratio

### Exercise 5: Scheduling Overhead Measurement

Measure real scheduler behavior:
1. Write a benchmark that measures context switch latency
2. Test with different numbers of tasks: 1, 10, 100, 1000
3. Compare SCHED_NORMAL, SCHED_FIFO, SCHED_RR
4. Use perf to measure scheduler overhead
5. Plot: latency distribution for each scheduler class

---

*End of Lesson 27*
