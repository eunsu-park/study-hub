# Lesson 16: Quantum Computing Landscape and Future

[← Previous: Quantum Machine Learning](15_Quantum_Machine_Learning.md) | [Back to Overview](00_Overview.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare major quantum hardware platforms: superconducting, trapped ion, photonic, neutral atom, and topological
2. Evaluate quantum processors using key metrics: qubit count, coherence time, gate fidelity, and connectivity
3. Explain the NISQ era and its limitations for practical quantum computing
4. Assess quantum advantage claims (Google Sycamore, Jiuzhang, and beyond)
5. Navigate the quantum cloud platform ecosystem (IBM Quantum, Amazon Braket, Azure Quantum, Google Cirq)
6. Describe the fault-tolerant quantum computing roadmap and its milestones
7. Identify open problems, research directions, and the relationship to post-quantum cryptography

---

The quantum computing industry today is at a remarkable inflection point. After decades of theoretical development and laboratory demonstrations, quantum computers are becoming accessible cloud services. Companies are competing fiercely to build machines with more qubits, longer coherence times, and lower error rates. Governments are investing billions in quantum research. Yet fundamental questions remain: When will quantum computers solve problems that no classical computer can? Which hardware approach will win? What are the killer applications?

This lesson surveys the quantum computing landscape as of 2025, examining the diverse hardware platforms, cloud services, and software tools. We will be honest about both the progress and the challenges, providing a balanced perspective that cuts through both the hype and the skepticism.

> **Analogy:** The quantum computing industry today is like the early days of classical computing in the 1950s — we have working prototypes, fierce competition between hardware approaches, and are just beginning to discover the killer applications. Just as vacuum tubes, transistors, and integrated circuits competed in the classical era, superconducting qubits, trapped ions, and photonic systems are competing today. The ultimate winner may not yet be on the table.

## Table of Contents

1. [Hardware Platforms](#1-hardware-platforms)
2. [Key Metrics for Quantum Processors](#2-key-metrics-for-quantum-processors)
3. [The NISQ Era](#3-the-nisq-era)
4. [Quantum Advantage and Supremacy](#4-quantum-advantage-and-supremacy)
5. [Quantum Cloud Platforms](#5-quantum-cloud-platforms)
6. [Quantum Software Ecosystem](#6-quantum-software-ecosystem)
7. [Fault-Tolerant Quantum Computing Roadmap](#7-fault-tolerant-quantum-computing-roadmap)
8. [Open Problems and Research Directions](#8-open-problems-and-research-directions)
9. [Post-Quantum Cryptography Connection](#9-post-quantum-cryptography-connection)
10. [Comparison Tables and Timelines](#10-comparison-tables-and-timelines)
11. [Exercises](#11-exercises)

---

## 1. Hardware Platforms

### 1.1 Superconducting Qubits

**Key players**: IBM, Google, Rigetti, IQM, Alice & Bob

**How it works**: Superconducting circuits cooled to ~15 millikelvin (colder than outer space) behave as quantum harmonic oscillators. A **transmon qubit** uses a Josephson junction to create an anharmonic oscillator with unequal energy level spacings, allowing the two lowest levels to serve as $|0\rangle$ and $|1\rangle$.

**Characteristics**:
- Qubit type: Charge/flux-based superconducting circuits
- Operating temperature: ~15 mK (dilution refrigerator)
- Gate speed: 10-100 ns (very fast)
- Coherence time: ~100-500 $\mu$s
- 2-qubit gate fidelity: 99.5-99.9%
- Connectivity: Nearest-neighbor (2D lattice, heavy-hex)
- Scalability: Good — leverages semiconductor fabrication techniques

**Pros**: Fast gates, mature fabrication, largest qubit counts (1000+)
**Cons**: Requires extreme cooling, limited connectivity, relatively short coherence times, crosstalk between nearby qubits

**Milestones**:
- 2019: Google Sycamore (53 qubits, quantum supremacy claim)
- 2022: IBM Osprey (433 qubits)
- 2023: IBM Condor (1,121 qubits)
- 2024: IBM Heron (133 qubits, improved error rates)
- 2025: Continued scaling toward 100K+ qubits roadmap

### 1.2 Trapped Ion Qubits

**Key players**: IonQ, Quantinuum (Honeywell), Alpine Quantum Technologies

**How it works**: Individual ions (typically $^{171}\text{Yb}^+$ or $^{40}\text{Ca}^+$) are trapped in electromagnetic fields (Paul trap or Penning trap). Two internal energy levels of each ion serve as $|0\rangle$ and $|1\rangle$. Gates are performed using precisely tuned laser pulses.

**Characteristics**:
- Qubit type: Hyperfine states of trapped ions
- Operating temperature: Room temperature (ions themselves are laser-cooled)
- Gate speed: 1-100 $\mu$s (slower than superconducting)
- Coherence time: Minutes to hours (!) — orders of magnitude longer
- 2-qubit gate fidelity: 99.5-99.9%
- Connectivity: **All-to-all** (any qubit can interact with any other)
- Scalability: Challenging beyond ~50 qubits per trap; modular approaches under development

**Pros**: Long coherence, all-to-all connectivity, identical qubits (no fabrication variability), high fidelity
**Cons**: Slow gates, scalability challenges, complex laser control systems

**Milestones**:
- 2023: Quantinuum H2 (56 qubits, 99.8% 2-qubit fidelity)
- 2024: IonQ Forte Enterprise (36 algorithmic qubits, high connectivity)
- 2025: Quantinuum pushing toward 100+ qubits with QCCD architecture

### 1.3 Photonic Qubits

**Key players**: Xanadu, PsiQuantum, ORCA Computing

**How it works**: Information is encoded in properties of photons — polarization, path, time bin, or photon number. Gates are implemented using beam splitters, phase shifters, and squeezed light sources. Measurement uses photon detectors.

**Characteristics**:
- Qubit type: Photon states (polarization, squeezed states)
- Operating temperature: Room temperature (mostly)
- Gate speed: Picoseconds (speed of light)
- Coherence time: Effectively infinite (photons do not decohere, but are lost)
- Gate fidelity: Varies widely (photon loss is the main error)
- Connectivity: Flexible (optical networks)
- Scalability: Promising — integrated photonic chips, but photon loss is a challenge

**Two approaches**:
1. **Linear optical QC (LOQC)**: Uses single photons, beam splitters, and measurement-based gates. Requires photon detection for gate teleportation.
2. **Continuous variable (CV)**: Uses squeezed light states. Xanadu's approach with Gaussian Boson Sampling.

**Pros**: Room temperature, natural for networking/communication, fast operations
**Cons**: Photon loss, non-deterministic gates (LOQC), challenging entanglement generation

### 1.4 Neutral Atom Qubits

**Key players**: QuEra Computing, Pasqal, Atom Computing, ColdQuanta/Infleqtion

**How it works**: Individual neutral atoms (typically rubidium or cesium) are trapped in optical tweezers — tightly focused laser beams that hold atoms at specific positions. Qubits are encoded in atomic energy levels. Entangling gates use the **Rydberg interaction**: when an atom is excited to a high-energy Rydberg state, it creates a strong interaction with neighboring atoms, enabling controlled entanglement.

**Characteristics**:
- Qubit type: Ground/Rydberg states of neutral atoms
- Operating temperature: Near absolute zero (laser cooling)
- Gate speed: ~$\mu$s
- Coherence time: ~seconds
- 2-qubit gate fidelity: 99.5%+ (rapidly improving)
- Connectivity: Programmable geometry (atoms can be rearranged!)
- Scalability: Very promising — hundreds of atoms already demonstrated

**Pros**: Scalable (up to 1000+ atoms), reconfigurable geometry, all identical qubits, native multi-qubit gates
**Cons**: Relatively new, atom loss during computation, slower gates than superconducting

**Milestones**:
- 2023: Harvard/QuEra demonstrated 48 logical qubits with error correction
- 2024: Continued scaling to 256+ physical qubits
- 2025: Active commercialization, exploring fault tolerance

### 1.5 Topological Qubits

**Key player**: Microsoft

**How it works**: Information is encoded in non-local topological properties of exotic quasiparticles called **anyons** (specifically, Majorana zero modes). Topological qubits would be inherently protected from local noise by their topological nature — errors would require large-scale disturbances to corrupt the information.

**Characteristics**:
- Qubit type: Majorana zero modes (topological)
- Operating temperature: ~mK
- Gate speed: Theoretical — not yet demonstrated at scale
- Coherence time: Theoretically very long (topological protection)
- Scalability: Theoretical — requires materials science breakthroughs

**Current status (2025)**: Microsoft announced progress on Majorana-based topological qubits, but a fully functional topological qubit with demonstrated error correction has not yet been publicly achieved. This approach is the most speculative but potentially the most revolutionary.

**Pros**: Inherent error protection, potentially transformative
**Cons**: Extremely challenging physics, earliest stage of development, questionable timeline

---

## 2. Key Metrics for Quantum Processors

### 2.1 Metric Definitions

| Metric | Definition | Why it matters |
|--------|-----------|---------------|
| **Qubit count** | Number of physical qubits | Determines problem size |
| **Coherence time** ($T_1$, $T_2$) | How long qubits maintain quantum information | Limits circuit depth |
| **Gate fidelity** | Probability that a gate operation is correct | Determines error rate |
| **1-qubit gate fidelity** | Typically 99.9%+ for all platforms | Less critical bottleneck |
| **2-qubit gate fidelity** | 99%-99.9% depending on platform | Main error source |
| **Circuit depth** | Maximum gates before decoherence | Limits algorithm complexity |
| **Connectivity** | Which qubit pairs can directly interact | Affects SWAP overhead |
| **Readout fidelity** | Accuracy of measurement | Affects final answer quality |
| **Clock speed** | Time per gate layer | Determines computation time |

### 2.2 Quantum Volume

**Quantum Volume** (QV), proposed by IBM, is a single-number benchmark that attempts to capture the overall capability of a quantum processor:

$$QV = 2^n$$

where $n$ is the largest circuit width (number of qubits) for which the processor can reliably execute random circuits of depth $n$.

QV incorporates qubit count, connectivity, gate fidelity, and measurement fidelity into one number. However, it has limitations:
- Favors all-to-all connectivity (trapped ions score well)
- Does not capture application-specific performance
- Saturates for large processors

### 2.3 Other Benchmarks

- **CLOPS** (Circuit Layer Operations Per Second): Measures throughput — how fast can the processor execute circuits?
- **Algorithmic qubits**: IonQ's metric — the number of qubits usable for algorithms after accounting for connectivity and fidelity
- **Logical error rate**: The error rate after quantum error correction (most relevant for fault-tolerant computing)

### 2.4 Platform Comparison (2025)

| Platform | Qubits | $T_2$ | 2Q Fidelity | Connectivity | QV |
|----------|--------|-------|-------------|-------------|-----|
| IBM (superconducting) | 1,121 | ~300 $\mu$s | 99.5% | Heavy-hex (nearest) | 128 |
| Google (superconducting) | 72 | ~100 $\mu$s | 99.7% | 2D grid | ~64 |
| Quantinuum (trapped ion) | 56 | ~3 s | 99.8% | All-to-all | $2^{20}$+ |
| IonQ (trapped ion) | 36 | ~1 s | 99.5% | All-to-all | $2^{25}$ (claimed) |
| QuEra (neutral atom) | 256 | ~1 s | 99.5% | Programmable | — |
| Xanadu (photonic) | ~200 modes | — | — | Graph-based | — |

---

## 3. The NISQ Era

### 3.1 What Is NISQ?

The term **Noisy Intermediate-Scale Quantum** (NISQ) was coined by John Preskill in 2018 to describe the current era of quantum computing:

- **Noisy**: Error rates are too high for full error correction
- **Intermediate-Scale**: 50-1000+ qubits — enough to be classically hard to simulate, but not enough for fault-tolerant algorithms
- **Quantum**: Genuine quantum effects (superposition, entanglement) are present

### 3.2 NISQ Capabilities

What NISQ devices can do:
- Execute circuits of depth ~100-500 (limited by decoherence)
- Demonstrate quantum advantage for specific, carefully chosen problems
- Run variational algorithms (VQE, QAOA) on small instances
- Explore quantum error correction on a small scale

What NISQ devices cannot do:
- Run Shor's algorithm for cryptographically relevant numbers
- Achieve practical quantum advantage for real-world problems (debated)
- Perform fault-tolerant computation
- Scale to thousands of logical qubits

### 3.3 NISQ Algorithms

| Algorithm | Problem | Status |
|-----------|---------|--------|
| VQE | Molecular ground states | Demonstrated for small molecules (H₂, LiH) |
| QAOA | Combinatorial optimization | Demonstrated for small graphs, unclear advantage |
| Variational classifiers | Machine learning | Demonstrated, no advantage over classical |
| Quantum simulation | Condensed matter, chemistry | Most promising near-term application |
| Random circuit sampling | Benchmarking | Quantum advantage demonstrated |

### 3.4 The Utility Era

IBM and others have proposed the concept of a **Utility Era** — a transition from NISQ to useful quantum computing:

- **Error mitigation**: Techniques like zero-noise extrapolation and probabilistic error cancellation partially correct errors without full QEC
- **Circuit knitting**: Breaking large circuits into smaller ones that fit on current hardware
- **Transpilation**: Optimizing circuits for specific hardware topologies

These techniques extend the practical reach of NISQ devices, though the magnitude of the extension is debated.

---

## 4. Quantum Advantage and Supremacy

### 4.1 Definitions

- **Quantum supremacy** (Preskill, 2012): A quantum computer performs a computation that no classical computer can perform in any reasonable time
- **Quantum advantage**: A quantum computer solves a *useful* problem faster than the best classical approach

Supremacy has been demonstrated; advantage for practical problems has not.

### 4.2 Google Sycamore (2019)

**Claim**: A 53-qubit superconducting processor sampled from random quantum circuits in 200 seconds, a task estimated to take a classical supercomputer 10,000 years.

**Challenge**: IBM argued that with enough memory (petabytes), a classical simulation could complete in 2.5 days. Subsequent classical algorithms further reduced the classical time.

**Status**: The claim has been partially upheld — the specific sampling task is genuinely hard for classical computers, but the problem itself has no known practical application.

### 4.3 Jiuzhang (2020)

**Claim**: A photonic quantum computer performed Gaussian Boson Sampling in 200 seconds, estimated to take a classical supercomputer $10^{14}$ seconds ($\approx$ 2.5 billion years).

**Significance**: Demonstrated quantum advantage using a completely different hardware platform (photonic vs superconducting), and for a different computational problem.

### 4.4 Subsequent Developments

- **2021**: University of Science and Technology of China (USTC) demonstrated quantum advantage with a 66-qubit superconducting processor (Zuchongzhi)
- **2023**: IBM demonstrated "utility" of quantum computation — a 127-qubit calculation that gave results agreeing with exact methods but was too large for brute-force classical simulation
- **2024-2025**: Continued demonstrations of quantum utility for specific physics problems, but no practical quantum advantage for commercial problems

### 4.5 The Honest Assessment

| Criterion | Status |
|-----------|--------|
| Quantum supremacy for contrived problems | **Achieved** (2019) |
| Quantum advantage for scientific problems | **Emerging** (2023-2025) |
| Quantum advantage for commercial problems | **Not achieved** |
| Quantum advantage for cryptography | **Decades away** |

---

## 5. Quantum Cloud Platforms

### 5.1 IBM Quantum

- **Hardware**: Superconducting (transmon), largest systems (1000+ qubits)
- **Access**: Free tier (limited), premium plans for larger systems
- **SDK**: Qiskit (Python)
- **Simulators**: Aer (local), cloud-based
- **Unique features**: Largest fleet of quantum processors, Qiskit Runtime for optimized execution

### 5.2 Amazon Braket

- **Hardware**: Access to IonQ (trapped ion), Rigetti (superconducting), QuEra (neutral atom), OQC (superconducting)
- **Access**: Pay-per-use through AWS
- **SDK**: Amazon Braket SDK (Python)
- **Simulators**: Local and managed simulators
- **Unique features**: Multi-vendor access from a single platform, AWS integration

### 5.3 Azure Quantum

- **Hardware**: Access to IonQ, Quantinuum, Rigetti, Pasqal
- **Access**: Pay-per-use through Azure
- **SDK**: Azure Quantum SDK, Q# language
- **Simulators**: Various classical and quantum-inspired
- **Unique features**: Resource estimation tools (planning for fault-tolerant era), integration with Azure cloud

### 5.4 Google Quantum AI

- **Hardware**: Sycamore processor (superconducting)
- **Access**: Primarily for research partners
- **SDK**: Cirq (Python)
- **Simulators**: qsim (high-performance simulator)
- **Unique features**: Focus on quantum error correction research, strong simulation capabilities

### 5.5 Platform Comparison

| Feature | IBM Quantum | Amazon Braket | Azure Quantum | Google QAI |
|---------|-------------|---------------|---------------|-----------|
| Free tier | Yes | Free credits | Free credits | Research |
| Primary SDK | Qiskit | Braket SDK | Q# / Qiskit | Cirq |
| Multi-vendor | No | Yes | Yes | No |
| Max qubits | 1,121 | ~256 (varies) | 56+ | 72 |
| Error correction | Active research | Via vendors | Active research | Leading |
| Education | Extensive | Good | Good | Good |

---

## 6. Quantum Software Ecosystem

### 6.1 Major Frameworks

| Framework | Developer | Language | Strengths |
|-----------|-----------|----------|----------|
| **Qiskit** | IBM | Python | Largest community, broad functionality, hardware access |
| **Cirq** | Google | Python | Low-level control, strong simulation |
| **PennyLane** | Xanadu | Python | Quantum ML focus, automatic differentiation |
| **Tket** | Quantinuum | Python/C++ | Advanced circuit optimization, multi-backend |
| **Q#** | Microsoft | Q# (custom) | Resource estimation, fault-tolerant focus |
| **Braket SDK** | Amazon | Python | Multi-vendor hardware access |
| **QuTiP** | Community | Python | Quantum dynamics simulation |

### 6.2 Framework Selection Guide

- **Learning quantum computing**: Qiskit (best tutorials and community)
- **Quantum ML research**: PennyLane (native autodiff, ML integration)
- **Low-level circuit design**: Cirq (fine-grained control)
- **Circuit optimization**: Tket (best transpiler)
- **Multi-vendor deployment**: Braket SDK or Tket
- **Fault-tolerant planning**: Q# (resource estimation)

### 6.3 Simulation Tools

For near-term research, classical simulators are essential:

| Simulator | Qubits | Speed | Special Features |
|-----------|--------|-------|-----------------|
| Qiskit Aer | ~30 | GPU-accelerated | Noise models |
| Google qsim | ~40 | Very fast (C++) | Optimized for Sycamore-like circuits |
| cuQuantum | ~30-40 | NVIDIA GPU | Tensor network methods |
| MPS/tensor network | 50-100+ | Variable | For low-entanglement circuits |

The crossover point — where quantum computers outperform simulators — depends heavily on the circuit structure. For random deep circuits, this occurs at ~50 qubits. For structured circuits with limited entanglement, classical methods can handle much larger systems.

---

## 7. Fault-Tolerant Quantum Computing Roadmap

### 7.1 From NISQ to Fault Tolerance

The path to useful, fault-tolerant quantum computing involves several stages:

```
2020s: NISQ Era
├── 50-1000+ noisy physical qubits
├── Variational algorithms (VQE, QAOA)
├── Error mitigation techniques
└── Quantum utility demonstrations

2025-2030: Early Fault Tolerance
├── 100-1000+ physical qubits with error correction
├── Logical qubits demonstrated (surface code)
├── Below-threshold error rates achieved
└── Small-scale error-corrected algorithms

2030-2035: Practical Fault Tolerance
├── 10,000-100,000 physical qubits
├── 100+ logical qubits
├── Quantum advantage for chemistry/optimization
└── Quantum simulation of new materials

2035+: Large-Scale Quantum Computing
├── 1,000,000+ physical qubits
├── 1,000+ logical qubits
├── Shor's algorithm for RSA-2048
└── General-purpose quantum computing
```

### 7.2 Key Milestones Achieved

| Year | Milestone | Significance |
|------|-----------|-------------|
| 2019 | Quantum supremacy (Google) | First quantum computation beyond classical reach |
| 2021 | Logical qubit lifetime exceeds physical (various) | Error correction works |
| 2023 | Below-threshold surface code (Google) | Practical error correction viable |
| 2023 | 48 logical qubits (Harvard/QuEra) | Error correction at scale |
| 2024 | Real-time error correction (IBM, Google) | Moving toward fault tolerance |
| 2025 | Continued progress on logical qubit quality | Approaching practical utility |

### 7.3 Resource Estimates for Practical Problems

| Application | Logical qubits | T gates | Physical qubits (est.) | Timeline |
|-------------|---------------|---------|----------------------|----------|
| Factor RSA-2048 | ~4,000 | $\sim 10^{12}$ | $\sim 20 \times 10^6$ | 2035+ |
| Simulate FeMoco (nitrogen fixation) | ~200 | $\sim 10^{10}$ | $\sim 4 \times 10^6$ | 2030+ |
| Pharmaceutical drug design | ~500 | $\sim 10^{11}$ | $\sim 10^7$ | 2032+ |
| Optimization (portfolio, logistics) | ~100 | $\sim 10^8$ | $\sim 10^5$ | 2028+ |

These estimates assume surface code error correction with physical error rates of $\sim 10^{-3}$, requiring roughly 1000-5000 physical qubits per logical qubit.

---

## 8. Open Problems and Research Directions

### 8.1 Fundamental Questions

1. **Quantum advantage boundary**: For which problems do quantum computers provide provable polynomial or exponential speedups? The precise boundary between classically easy and quantumly easy problems is poorly understood.

2. **Barren plateaus**: Can we design variational algorithms that are trainable at scale? The barren plateau problem (Lesson 15) remains a fundamental obstacle.

3. **Quantum error correction overhead**: Can we reduce the overhead from ~1000 physical qubits per logical qubit to ~10-100? This would transform the timeline for practical quantum computing.

4. **Quantum simulation**: Can NISQ devices provide useful quantum simulations of materials and molecules before fault tolerance? This is the most promising near-term application.

### 8.2 Hardware Research

- **Better qubits**: Longer coherence, higher fidelity, faster gates
- **Modular architectures**: Connecting multiple small quantum processors via quantum links
- **3D integration**: Stacking qubit layers for increased density
- **New qubit types**: Cat qubits (hardware-efficient error correction), dual-rail photonic qubits, silicon spin qubits
- **Cryogenic electronics**: Classical control electronics that operate at millikelvin temperatures to reduce wiring complexity

### 8.3 Software and Algorithm Research

- **Better NISQ algorithms**: Algorithms that make the best use of limited, noisy qubits
- **Error mitigation**: Improving techniques like zero-noise extrapolation and probabilistic error cancellation
- **Quantum compilers**: Optimizing circuits for specific hardware constraints
- **Quantum-classical hybrid**: Better ways to combine quantum and classical processing
- **Application discovery**: Finding new domains where quantum computing provides genuine value

### 8.4 Industry vs Academia

The quantum computing field has a unique dynamic:

- **Industry**: Focuses on hardware scaling, cloud services, and near-term applications. Tends toward optimistic timelines.
- **Academia**: Focuses on fundamental theory, algorithm development, and rigorous benchmarking. Tends toward cautious assessments.
- **Government**: Funds both, motivated by national security (cryptography) and economic competitiveness.

A healthy ecosystem requires all three perspectives, with honest communication about what has been achieved and what remains speculative.

---

## 9. Post-Quantum Cryptography Connection

### 9.1 The Threat

Shor's algorithm (Lesson 10) threatens all public-key cryptography based on integer factoring or discrete logarithms:

- **RSA**: Broken by Shor's factoring algorithm
- **Diffie-Hellman**: Broken by quantum discrete log algorithm
- **Elliptic curve cryptography**: Broken by quantum period-finding on elliptic curves

### 9.2 Post-Quantum Cryptography (PQC)

NIST standardized post-quantum algorithms in 2024 (see Security L05):

| Algorithm | Type | Use | Based on |
|-----------|------|-----|----------|
| ML-KEM (Kyber) | Key encapsulation | Key exchange | Module lattices |
| ML-DSA (Dilithium) | Digital signature | Authentication | Module lattices |
| SLH-DSA (SPHINCS+) | Digital signature | Authentication | Hash functions |
| FN-DSA (Falcon) | Digital signature | Authentication | NTRU lattices |

These algorithms are designed to be secure against both classical and quantum attacks.

### 9.3 Timeline Pressure

The "harvest now, decrypt later" threat: adversaries can collect encrypted data today and decrypt it once quantum computers are available. For data with long secrecy requirements (government, healthcare, financial), the transition to PQC must begin immediately — decades before quantum computers can actually break current encryption.

### 9.4 Quantum Key Distribution (QKD)

An alternative approach (Lesson 12): use quantum mechanics itself for key distribution. QKD provides information-theoretic security but has practical limitations (distance, key rate, cost). PQC and QKD are complementary rather than competing solutions.

---

## 10. Comparison Tables and Timelines

### 10.1 Hardware Platform Summary

| Feature | Superconducting | Trapped Ion | Photonic | Neutral Atom | Topological |
|---------|----------------|-------------|----------|-------------|-------------|
| **Company** | IBM, Google | IonQ, Quantinuum | Xanadu, PsiQuantum | QuEra, Pasqal | Microsoft |
| **Max qubits** | 1,121 | 56 | ~200 modes | 256 | ~1 (experimental) |
| **Coherence** | ~500 $\mu$s | Hours | N/A (loss) | Seconds | Theoretically infinite |
| **2Q fidelity** | 99.5% | 99.8% | Varies | 99.5% | N/A |
| **Connectivity** | Nearest-neighbor | All-to-all | Flexible | Programmable | N/A |
| **Gate speed** | ~50 ns | ~10 $\mu$s | ~ps | ~$\mu$s | N/A |
| **Maturity** | Most mature | Mature | Developing | Rising fast | Very early |
| **Scaling** | Fabrication | Modular traps | Integrated photonics | Optical tweezers | Unknown |

### 10.2 Application Timeline (Estimated)

| Application | Required logical qubits | Estimated timeline | Confidence |
|-------------|------------------------|-------------------|-----------|
| Quantum simulation (small) | 10-50 | 2025-2028 | High |
| Drug discovery acceleration | 100-500 | 2030-2035 | Medium |
| Materials science | 200-1,000 | 2030-2035 | Medium |
| Optimization (practical advantage) | 50-200 | 2028-2033 | Medium |
| Machine learning advantage | Unknown | Unknown | Low |
| Break RSA-2048 | ~4,000 | 2035-2045+ | Low |
| General-purpose quantum computing | 10,000+ | 2040+ | Very low |

### 10.3 Investment Landscape

| Region | Notable investments (2020-2025) |
|--------|-------------------------------|
| **USA** | $2B+ government (National Quantum Initiative), massive private investment (IBM, Google, IonQ, Rigetti) |
| **China** | $15B+ government investment, USTC quantum supremacy demonstrations |
| **EU** | $1B+ (Quantum Flagship program), IQM, Pasqal, Alpine Quantum Technologies |
| **UK** | $1B+ (National Quantum Computing Centre), Quantinuum, ORCA |
| **Japan** | $500M+ government, Riken quantum computing center |
| **South Korea** | $40M+ government, Samsung quantum research |
| **Australia** | Silicon Quantum Computing (silicon spin qubits) |

---

## 11. Exercises

### Exercise 1: Hardware Platform Analysis

Based on the comparison table in Section 10.1:
(a) For each of the following applications, argue which hardware platform is best suited and why:
   - Running Shor's algorithm for RSA-2048
   - Near-term variational chemistry (VQE)
   - Quantum key distribution network
   - Combinatorial optimization (QAOA)
(b) What is the most important metric for each application? (qubit count, fidelity, connectivity, coherence, or speed?)
(c) If you could improve only one metric by 10x for each platform, which would give the biggest practical impact?

### Exercise 2: Quantum Volume Calculation

Quantum Volume is defined as $QV = 2^n$ where $n$ is the largest circuit size the processor can reliably execute.

(a) A processor has 20 qubits, 99.5% 2-qubit gate fidelity, and nearest-neighbor connectivity. The circuit of depth $d$ on $n$ qubits requires roughly $3n \cdot d$ SWAP gates for connectivity. Estimate the Quantum Volume by finding the largest $n$ where the total circuit fidelity exceeds 2/3.
(b) The same processor improves to 99.9% fidelity. How does QV change?
(c) A trapped ion processor has 12 qubits, 99.8% fidelity, and all-to-all connectivity (0 SWAP overhead). What is its QV?
(d) Argue whether QV is a good metric for comparing these two processors.

### Exercise 3: Fault-Tolerance Resource Estimation

For a surface code with code distance $d$ and physical error rate $p$:
- Logical error rate: $p_L \approx 0.1 \times (100p)^{(d+1)/2}$
- Physical qubits per logical qubit: $2d^2$

(a) For $p = 10^{-3}$ (current state of the art), what code distance $d$ is needed for $p_L < 10^{-10}$?
(b) How many physical qubits per logical qubit?
(c) To factor RSA-2048 (needing ~4,000 logical qubits), how many total physical qubits are needed?
(d) If hardware improves to $p = 10^{-4}$, how do these numbers change?

### Exercise 4: Quantum Cloud Platform Comparison

Using the documentation of at least two quantum cloud platforms (IBM Quantum and one other):
(a) Write a simple quantum circuit (Bell state preparation + measurement) in each platform's SDK.
(b) Compare the programming models: how is a circuit defined, transpiled, and executed?
(c) What are the pricing models? Estimate the cost of running 10,000 shots of a 10-qubit circuit on each platform.
(d) Which platform would you recommend for a university course on quantum computing? For a startup developing quantum algorithms?

### Exercise 5: Future Prediction Analysis

Many predictions about quantum computing timelines have been made:
(a) Research and list 5 predictions made in 2019-2020 about what quantum computing would achieve by 2025.
(b) How accurate were these predictions?
(c) Based on the actual progress from 2019-2025, what do you predict for 2030?
(d) What are the three biggest technical obstacles that could delay or accelerate the timeline?
(e) If you were advising a government on quantum computing investment, what would you recommend focusing on and why?

---

[← Previous: Quantum Machine Learning](15_Quantum_Machine_Learning.md) | [Back to Overview](00_Overview.md)
