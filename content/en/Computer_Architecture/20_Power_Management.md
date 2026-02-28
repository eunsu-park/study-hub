# Power Management

**Previous**: [RISC-V Architecture](./19_RISC_V_Architecture.md)

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: CPU architecture, parallel processing and multicore (Lesson 18)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the sources of power consumption in CMOS circuits (dynamic, static, short-circuit)
2. Describe Dynamic Voltage and Frequency Scaling (DVFS) and calculate power savings
3. Explain clock gating and power gating as techniques to reduce active and leakage power
4. Analyze the dark silicon problem and its impact on multicore scaling
5. Compare energy-proportional computing approaches in modern processors

---

A modern laptop processor can draw 5 watts when reading email and 60 watts when rendering video. A smartphone SoC must survive a full day on a battery that stores roughly 15 Wh. Data centers spend nearly as much on cooling as on computation. Power management is no longer a niche concern -- it shapes every level of computer architecture, from transistor design to operating system scheduling.

---

## Table of Contents

1. [Power Fundamentals in CMOS](#1-power-fundamentals-in-cmos)
2. [Dynamic Voltage and Frequency Scaling](#2-dynamic-voltage-and-frequency-scaling)
3. [Clock Gating](#3-clock-gating)
4. [Power Gating](#4-power-gating)
5. [Dark Silicon](#5-dark-silicon)
6. [Processor Power States](#6-processor-power-states)
7. [Energy-Proportional Computing](#7-energy-proportional-computing)
8. [Thermal Management](#8-thermal-management)
9. [Practice Problems](#9-practice-problems)

---

## 1. Power Fundamentals in CMOS

Total power in a CMOS circuit has three components:

$$P_{total} = P_{dynamic} + P_{static} + P_{short-circuit}$$

### 1.1 Dynamic Power

Dynamic power comes from charging and discharging capacitors when transistors switch:

$$P_{dynamic} = \alpha \cdot C \cdot V_{dd}^2 \cdot f$$

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| α | Activity factor (fraction of gates switching per cycle) | 0.1 - 0.3 |
| C | Total load capacitance | Process-dependent |
| V_dd | Supply voltage | 0.6 - 1.2 V |
| f | Clock frequency | 1 - 5 GHz |

The key insight: power scales with the **square** of voltage. Reducing voltage from 1.0V to 0.7V cuts dynamic power by 51% (0.7² / 1.0² = 0.49).

### 1.2 Static Power (Leakage)

Even when transistors are not switching, current leaks through them:

$$P_{static} = V_{dd} \cdot I_{leak}$$

Leakage current increases exponentially as transistors shrink. At 7nm and below, leakage can account for 30-50% of total chip power.

| Leakage Type | Cause | Mitigation |
|-------------|-------|------------|
| Sub-threshold | Current flows even when gate is "off" | Higher threshold voltage (slower) |
| Gate oxide tunneling | Current tunnels through thin gate oxide | High-k dielectrics (HfO₂) |
| Junction leakage | Reverse-biased p-n junction current | Well biasing |

### 1.3 Short-Circuit Power

During switching transitions, both PMOS and NMOS transistors are briefly on simultaneously, creating a direct path from V_dd to ground. This typically accounts for 5-10% of dynamic power and is minimized by ensuring fast transition times.

---

## 2. Dynamic Voltage and Frequency Scaling

DVFS adjusts both voltage and frequency together because:
- Higher frequency requires higher voltage for reliable switching
- Lower frequency allows lower voltage

### 2.1 DVFS Operating Points

```
┌────────────────────────────────────────────────────────────┐
│           DVFS Operating Points                             │
├──────────┬───────────┬──────────┬───────────┬──────────────┤
│ Mode     │ Frequency │ Voltage  │ Rel Power │ Performance  │
├──────────┼───────────┼──────────┼───────────┼──────────────┤
│ Turbo    │ 4.5 GHz   │ 1.20 V   │ 100%      │ Maximum      │
│ High     │ 3.5 GHz   │ 1.00 V   │  54%      │ 78%          │
│ Normal   │ 2.5 GHz   │ 0.85 V   │  28%      │ 56%          │
│ Low      │ 1.5 GHz   │ 0.70 V   │  12%      │ 33%          │
│ Idle     │ 0.8 GHz   │ 0.60 V   │   5%      │ 18%          │
└──────────┴───────────┴──────────┴───────────┴──────────────┘
```

### 2.2 Power-Performance Trade-off

If we halve the frequency and reduce voltage proportionally:
- Performance: halved (2x slower)
- Dynamic power: reduced by ~8x (because P ∝ V² × f, and V also drops)
- Energy per task: may actually decrease (power drops faster than time increases)

This means running at lower voltage/frequency can be more energy-efficient even though the task takes longer.

### 2.3 DVFS in Practice

| Platform | DVFS Technology | Control |
|----------|----------------|---------|
| Intel | SpeedStep / Speed Shift (HWP) | Hardware-managed with OS hints |
| AMD | Cool'n'Quiet / Precision Boost | Firmware + OS governor |
| ARM | big.LITTLE + DVFS per cluster | OS DVFS driver |
| RISC-V | Platform-specific (SBI calls) | OS or firmware |

Linux exposes DVFS through the `cpufreq` subsystem:

```bash
# View available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
# performance  powersave  ondemand  conservative  schedutil

# Set governor
echo "schedutil" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# View current frequency (kHz)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

---

## 3. Clock Gating

Clock gating stops the clock signal to unused circuit blocks, eliminating their dynamic power consumption.

### 3.1 How It Works

```
                    ┌─────────┐
Clock ──────┐       │         │
            ├─ AND ─┤ Circuit │
Enable ─────┘       │  Block  │
                    └─────────┘

When Enable = 0:
  - Clock signal does not reach the circuit
  - No switching activity → no dynamic power
  - Register values are preserved (static)
```

### 3.2 Granularity Levels

| Level | What Gets Gated | Savings | Latency to Resume |
|-------|----------------|---------|-------------------|
| **Fine-grained** | Individual functional units (ALU, FPU) | 5-15% | 0 cycles |
| **Medium** | Execution pipeline stages | 15-30% | 1-3 cycles |
| **Coarse** | Entire cores or IP blocks | 30-60% | 10-100 cycles |

Modern processors use automatic clock gating: the hardware detects when a unit has no work and gates its clock without software involvement.

### 3.3 Example: Intel Clock Gating

Intel's processors gate clocks at multiple levels:
- **Instruction-level**: Unused execution ports are clock-gated each cycle
- **Core-level**: Idle cores have their clocks stopped (C1 state)
- **Uncore**: L3 cache slices, memory controllers, interconnects can be independently gated

---

## 4. Power Gating

Power gating goes further than clock gating by cutting off the power supply entirely, eliminating both dynamic and leakage power.

### 4.1 Implementation

```
V_dd ──────┤ Power Switch (Sleep Transistor)
            │
         ┌──┴──────────────┐
         │                 │
         │  Logic Block    │ ← Virtual V_dd
         │                 │
         └──┬──────────────┘
            │
GND ────────┘
```

The sleep transistor is a large PMOS device that disconnects the logic block from V_dd. When off:
- No current flows → zero dynamic and leakage power
- All register/memory state is **lost** (must be saved first)

### 4.2 State Retention

To preserve state during power gating:

| Technique | How | Cost |
|-----------|-----|------|
| **Software save/restore** | OS saves registers to DRAM before power-off | High latency (μs-ms) |
| **Retention registers** | Special always-on flip-flops hold critical state | Area overhead (~10%) |
| **Retention voltage** | Keep very low voltage (data retained, no switching) | Small leakage remains |

### 4.3 Power Gating vs Clock Gating

| Aspect | Clock Gating | Power Gating |
|--------|-------------|-------------|
| Eliminates dynamic power | Yes | Yes |
| Eliminates leakage | No | Yes |
| State preservation | Automatic | Requires save/restore |
| Wake-up latency | 0-3 cycles | 10-1000+ cycles |
| Hardware complexity | Low | High (sleep transistors, retention) |

---

## 5. Dark Silicon

### 5.1 The Problem

As transistors shrink, more fit on a chip, but power density limits prevent running them all simultaneously. The fraction of the chip that must remain unpowered at any time is called **dark silicon**.

```
Moore's Law says:  Transistor count doubles every 2 years
Dennard Scaling says: Power density stays constant as transistors shrink
                      (THIS BROKE DOWN around 45nm / ~2006)

Result: We can build chips with 10 billion transistors,
        but can only power ~30-50% of them at once.
```

### 5.2 Estimates

At 8nm technology, studies estimate 50-80% of chip area must be dark at any given time under typical thermal design power (TDP) constraints.

### 5.3 Strategies for Dark Silicon

| Strategy | Description | Example |
|----------|-------------|---------|
| **Dim silicon** | Run all cores at reduced V/f | DVFS (Section 2) |
| **Heterogeneous cores** | Mix big+small cores, activate as needed | ARM big.LITTLE, Intel P+E cores |
| **Specialized accelerators** | Use dark area for special-purpose units | GPU, NPU, video encode/decode, crypto |
| **Chiplets** | Separate dies with independent power domains | AMD Zen (CCD + IOD) |
| **Near-threshold computing** | Operate transistors near their threshold voltage | Research prototypes |

### 5.4 Modern Heterogeneous Design

Apple M-series chips exemplify the heterogeneous approach:

```
┌──────────────────────────────────────────────────┐
│                Apple M4 Pro SoC                   │
├──────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐              │
│  │ Performance   │  │ Efficiency   │              │
│  │ Cores (6)     │  │ Cores (4)    │              │
│  │ High V/f      │  │ Low V/f      │  ← big.LITTLE│
│  │ High IPC       │  │ Low power    │              │
│  └──────────────┘  └──────────────┘              │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ GPU (20 core) │  │ Neural Engine│              │
│  │ Graphics      │  │ ML inference │  ← Accelerators│
│  └──────────────┘  └──────────────┘              │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ Media Engine  │  │ Secure       │              │
│  │ H.264/265/AV1│  │ Enclave      │              │
│  └──────────────┘  └──────────────┘              │
└──────────────────────────────────────────────────┘
```

Only the components needed for the current workload are fully powered. Video playback uses the media engine (low power); ML training uses the Neural Engine and GPU; email uses only efficiency cores.

---

## 6. Processor Power States

### 6.1 ACPI C-States

The Advanced Configuration and Power Interface (ACPI) defines processor idle states:

| State | Name | Description | Wake Latency |
|-------|------|-------------|-------------|
| **C0** | Active | Executing instructions | 0 |
| **C1** | Halt | Clock stopped, voltage maintained | ~1 μs |
| **C1E** | Enhanced Halt | C1 + reduced voltage | ~10 μs |
| **C3** | Sleep | L1/L2 cache flushed, clock off | ~50-100 μs |
| **C6** | Deep Sleep | Core power gated, state saved | ~100-200 μs |
| **C7** | Deeper Sleep | L2 cache also flushed | ~200-400 μs |
| **C10** | Package Sleep | All cores off, only PMC alive | ~1-5 ms |

### 6.2 P-States (Performance States)

P-states control active performance via DVFS:
- P0: Maximum performance (Turbo Boost)
- P1: Maximum non-turbo frequency
- Pn: Minimum operating frequency

The OS scheduler and hardware collaborate to select the right P-state based on workload.

### 6.3 Package vs Core Power States

```
Package C-state = deepest state where ALL cores are idle

Core 0: C6 (deep sleep)
Core 1: C1 (halt)       → Package state = C1 (limited by shallowest)
Core 2: C6 (deep sleep)
Core 3: C6 (deep sleep)

Core 0: C6
Core 1: C6              → Package state = C6 (all cores deep sleep)
Core 2: C6                 Now package-level power savings apply
Core 3: C6
```

---

## 7. Energy-Proportional Computing

### 7.1 The Problem

Servers spend most of their time at 10-50% utilization, but power consumption at idle is still 50-60% of peak. The relationship between utilization and power is far from linear:

```
Power
(Watts)
  200 ├─────────────────────────────────●  Peak
      │                            ●
  150 ├                       ●
      │                  ●
  100 ├─────────────●                     ← Ideal: linear
      │         ●
   50 ├─ ●──                              ← Actual: high idle power
      │  ↑ Idle power (~50% of peak)
    0 ├──┬──┬──┬──┬──┬──┬──┬──┬──┬──
      0  10 20 30 40 50 60 70 80 90 100
                  Utilization (%)
```

### 7.2 Approaches

| Approach | Description | Effectiveness |
|----------|-------------|--------------|
| **DVFS** | Scale voltage/frequency with load | Moderate (dynamic power only) |
| **Power gating** | Turn off idle cores entirely | High (eliminates leakage too) |
| **Server consolidation** | Migrate VMs to fewer servers, power off rest | Very high |
| **Heterogeneous computing** | Route work to most efficient compute unit | High |
| **Near-data processing** | Process data near storage to avoid transfer energy | Moderate |

### 7.3 Power Usage Effectiveness (PUE)

Data center efficiency is measured by PUE:

$$PUE = \frac{\text{Total Facility Power}}{\text{IT Equipment Power}}$$

| PUE | Rating | Meaning |
|-----|--------|---------|
| 2.0+ | Poor | 50%+ overhead on cooling, lighting, UPS |
| 1.4 | Average | Industry average |
| 1.1 | Excellent | Google, Meta achieve this with advanced cooling |
| 1.0 | Ideal | All power goes to computation (impossible in practice) |

---

## 8. Thermal Management

### 8.1 Thermal Design Power (TDP)

TDP is the maximum sustained power the cooling system must dissipate. It is not the maximum power draw (which can exceed TDP during turbo bursts).

| Processor | TDP | Max Power |
|-----------|-----|-----------|
| Intel Core i9-14900K | 125 W | 253 W (turbo) |
| AMD Ryzen 9 7950X | 170 W | 230 W (PPT) |
| Apple M4 Pro | ~30 W | ~40 W |
| ARM Cortex-A78 (per core) | ~1 W | ~1.5 W |

### 8.2 Thermal Throttling

When chip temperature exceeds the junction temperature limit (T_j,max, typically 100-105°C):

1. **DVFS throttling**: Reduce frequency and voltage
2. **Clock modulation**: Insert idle cycles (duty cycling)
3. **Core parking**: Migrate work away from hot cores
4. **Emergency shutdown**: If temperature continues to rise despite throttling

### 8.3 Cooling Technologies

| Technology | Typical Application | Heat Dissipation |
|-----------|-------------------|-----------------|
| Passive heatsink | Low-power SoCs | 5-30 W |
| Active fan + heatsink | Desktop/Laptop CPUs | 65-250 W |
| Vapor chamber | High-end laptops, GPUs | 100-400 W |
| Liquid cooling (AIO) | Enthusiast desktops | 200-500 W |
| Direct liquid cooling | Data centers | 500+ W |
| Immersion cooling | HPC, cryptocurrency | 1000+ W |

---

## 9. Practice Problems

### Problem 1: Power Calculation

A processor has:
- Activity factor α = 0.2
- Load capacitance C = 20 nF
- Supply voltage V_dd = 1.0 V
- Clock frequency f = 3.0 GHz
- Leakage current I_leak = 10 A

Calculate: (a) Dynamic power, (b) Static power, (c) Total power. If voltage is reduced to 0.7V and frequency to 2.0 GHz, what are the new values?

### Problem 2: DVFS Energy Analysis

A task requires 10 billion cycles. Compare running it at:
- Option A: 4 GHz, 1.2V (TDP = 100W dynamic)
- Option B: 2 GHz, 0.8V

For each option, calculate: execution time, dynamic power, and total energy consumed. Which option is more energy-efficient?

### Problem 3: Dark Silicon Budget

A 100 mm² chip at 5nm contains 20 billion transistors. The TDP is 15W (mobile) and each transistor at full activity consumes 0.5 pW on average.
1. What fraction of transistors can be active simultaneously?
2. How does this change at 3nm (assume 1.5x density, same TDP)?
3. Propose a heterogeneous design that uses the dark silicon budget effectively.

### Problem 4: C-State Analysis

A server core alternates between 2ms of computation and 8ms of idle. Available C-states are C1 (1μs wake, saves 30% idle power), C3 (100μs wake, saves 70%), C6 (500μs wake, saves 95%). Which C-state should the OS select during each idle period? Consider the break-even time for each state.

### Problem 5: PUE and Total Cost

A data center runs 10,000 servers at 300W average each. PUE = 1.5. Electricity costs $0.10/kWh.
1. Calculate annual electricity cost.
2. If PUE improves to 1.2, how much money is saved per year?
3. What capital investment in cooling improvements would break even in 3 years?

---

*End of Lesson 20*
