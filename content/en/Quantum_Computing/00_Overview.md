# Quantum Computing

## Overview

Quantum computing harnesses quantum mechanical phenomena — superposition, entanglement, and interference — to process information in fundamentally different ways from classical computers. While classical bits are either 0 or 1, quantum bits (qubits) can exist in superpositions of both states simultaneously, enabling certain computations to achieve exponential speedups.

This topic covers the theoretical foundations (quantum mechanics primer, qubit mathematics, gate model), landmark algorithms (Deutsch-Jozsa, Grover, Shor), error correction, and modern applications (variational algorithms, quantum machine learning). Examples use Qiskit-style pseudocode and NumPy simulations.

## Prerequisites

- **Linear Algebra**: Complex vector spaces, matrix operations, eigenvalues (Math_for_AI L01-L03)
- **Probability**: Basic probability theory (Data_Science L11-L13)
- **Python**: Comfortable with NumPy (Python L01-L08)
- **Cryptography** (optional): RSA for understanding Shor's algorithm (Cryptography_Theory L05)

## Learning Path

```
Foundations (L01-L06)
├── L01: Quantum Mechanics Primer
├── L02: Qubits and the Bloch Sphere
├── L03: Quantum Gates
├── L04: Quantum Circuits
├── L05: Entanglement and Bell States
└── L06: Quantum Measurement

Algorithms (L07-L10)
├── L07: Deutsch-Jozsa Algorithm
├── L08: Grover's Search Algorithm
├── L09: Quantum Fourier Transform
└── L10: Shor's Factoring Algorithm

Advanced Topics (L11-L16)
├── L11: Quantum Error Correction
├── L12: Quantum Teleportation and Communication
├── L13: Variational Quantum Eigensolver (VQE)
├── L14: QAOA and Combinatorial Optimization
├── L15: Quantum Machine Learning
└── L16: Quantum Computing Landscape and Future
```

## Lessons

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Quantum Mechanics Primer](01_Quantum_Mechanics_Primer.md) | Wave-particle duality, superposition, measurement postulate, Dirac notation |
| 02 | [Qubits and the Bloch Sphere](02_Qubits_and_Bloch_Sphere.md) | Single-qubit states, Bloch sphere visualization, multi-qubit systems |
| 03 | [Quantum Gates](03_Quantum_Gates.md) | Pauli gates, Hadamard, phase gates, CNOT, universal gate sets |
| 04 | [Quantum Circuits](04_Quantum_Circuits.md) | Circuit model, circuit depth/width, simulation with matrices |
| 05 | [Entanglement and Bell States](05_Entanglement_and_Bell_States.md) | EPR pairs, Bell states, CHSH inequality, non-locality |
| 06 | [Quantum Measurement](06_Quantum_Measurement.md) | Projective measurement, POVM, measurement bases, partial measurement |
| 07 | [Deutsch-Jozsa Algorithm](07_Deutsch_Jozsa_Algorithm.md) | Oracle model, quantum parallelism, first exponential speedup |
| 08 | [Grover's Search Algorithm](08_Grovers_Search.md) | Amplitude amplification, oracle construction, quadratic speedup |
| 09 | [Quantum Fourier Transform](09_Quantum_Fourier_Transform.md) | QFT circuit, phase estimation, connection to classical FFT |
| 10 | [Shor's Factoring Algorithm](10_Shors_Algorithm.md) | Period finding, modular exponentiation, RSA implications |
| 11 | [Quantum Error Correction](11_Quantum_Error_Correction.md) | Bit-flip/phase-flip codes, Shor code, stabilizer formalism, surface codes |
| 12 | [Quantum Teleportation and Communication](12_Quantum_Teleportation.md) | Teleportation protocol, superdense coding, no-cloning theorem |
| 13 | [Variational Quantum Eigensolver](13_VQE.md) | Variational principle, ansatz design, parameter optimization, molecular simulation |
| 14 | [QAOA and Combinatorial Optimization](14_QAOA.md) | MaxCut, mixing/cost Hamiltonians, parameter landscapes |
| 15 | [Quantum Machine Learning](15_Quantum_Machine_Learning.md) | Quantum feature maps, variational classifiers, quantum kernels, barren plateaus |
| 16 | [Quantum Computing Landscape and Future](16_Landscape_and_Future.md) | Hardware platforms, quantum advantage, NISQ era, fault-tolerant roadmap |

## Relationship to Other Topics

| Topic | Connection |
|-------|-----------|
| Math_for_AI | Linear algebra foundations (complex vector spaces, unitary matrices) |
| Cryptography_Theory | Post-quantum cryptography motivated by Shor's algorithm |
| Signal_Processing | QFT is the quantum analog of discrete Fourier transform |
| Machine_Learning | Quantum ML extends classical ML with quantum feature spaces |
| Mathematical_Methods | Complex analysis, linear algebra used throughout |

## Example Files

Located in `examples/Quantum_Computing/`:

| File | Description |
|------|-------------|
| `01_qubit_simulation.py` | Qubit state vectors, Bloch sphere, measurement simulation |
| `02_quantum_gates.py` | Gate matrices, gate application, universal gate decomposition |
| `03_quantum_circuits.py` | Circuit builder, multi-qubit simulation, entanglement |
| `04_bell_states.py` | Bell state preparation, CHSH game simulation |
| `05_deutsch_jozsa.py` | Deutsch-Jozsa oracle and algorithm implementation |
| `06_grovers_search.py` | Grover's algorithm with oracle, amplitude amplification |
| `07_quantum_fourier.py` | QFT circuit, phase estimation |
| `08_shors_algorithm.py` | Period finding, Shor's algorithm for small numbers |
| `09_error_correction.py` | Bit-flip code, Shor 9-qubit code, syndrome measurement |
| `10_teleportation.py` | Quantum teleportation protocol simulation |
| `11_vqe.py` | VQE for H₂ molecule ground state |
| `12_qaoa.py` | QAOA for MaxCut problem |
| `13_quantum_ml.py` | Quantum feature map, variational classifier |
