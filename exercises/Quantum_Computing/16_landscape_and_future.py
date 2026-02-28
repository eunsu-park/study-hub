"""
Exercises for Lesson 16: Quantum Computing Landscape and Future
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
Exercises 1, 4, 5 are analysis/discussion-based; 2, 3 are computational.
"""

import numpy as np
from math import comb, log2, ceil
from typing import Dict, List


# === Exercise 1: Hardware Platform Analysis ===
# Problem: For each application, argue which hardware platform is best suited.

def exercise_1():
    """Hardware platform analysis for different quantum applications."""
    print("=" * 60)
    print("Exercise 1: Hardware Platform Analysis")
    print("=" * 60)

    platforms = {
        "Superconducting": {
            "qubits": "50-1000+",
            "gate_fidelity": "99.5-99.9%",
            "connectivity": "Nearest-neighbor (grid)",
            "coherence": "~100 us",
            "gate_speed": "~20-100 ns",
            "vendors": "IBM, Google, Rigetti",
        },
        "Trapped Ion": {
            "qubits": "10-50",
            "gate_fidelity": "99.5-99.9%",
            "connectivity": "All-to-all",
            "coherence": "~seconds",
            "gate_speed": "~1-100 us",
            "vendors": "IonQ, Quantinuum",
        },
        "Photonic": {
            "qubits": "Up to 216 (Borealis)",
            "gate_fidelity": "~99%",
            "connectivity": "Programmable",
            "coherence": "N/A (no decoherence)",
            "gate_speed": "~ps (fastest)",
            "vendors": "Xanadu, PsiQuantum",
        },
        "Neutral Atom": {
            "qubits": "100-1000+",
            "gate_fidelity": "~99.5%",
            "connectivity": "Reconfigurable",
            "coherence": "~seconds",
            "gate_speed": "~1 us",
            "vendors": "Atom Computing, QuEra",
        },
    }

    print("\nPlatform Comparison:")
    print("-" * 70)
    for platform, specs in platforms.items():
        print(f"\n  {platform}:")
        for key, val in specs.items():
            print(f"    {key:<15} {val}")

    # (a) Application-specific recommendations
    applications = {
        "Shor's RSA-2048": {
            "best": "Superconducting (long-term) or Trapped Ion",
            "key_metric": "Qubit count + fidelity (needs ~4000 logical qubits)",
            "reasoning": (
                "Needs millions of physical qubits with high fidelity. "
                "Superconducting has the scaling path; trapped ions have "
                "better fidelity but fewer qubits currently."
            ),
        },
        "Near-term VQE": {
            "best": "Trapped Ion or Neutral Atom",
            "key_metric": "Gate fidelity (variational circuits are shallow but need precision)",
            "reasoning": (
                "VQE uses shallow circuits on small systems. All-to-all connectivity "
                "(trapped ions) avoids SWAP overhead. High coherence time helps."
            ),
        },
        "QKD Network": {
            "best": "Photonic",
            "key_metric": "Connectivity and transmission (photons travel naturally)",
            "reasoning": (
                "QKD requires transmitting quantum states over distances. "
                "Photonic systems are inherently suited for quantum communication."
            ),
        },
        "QAOA Optimization": {
            "best": "Neutral Atom or Superconducting",
            "key_metric": "Qubit count (combinatorial problems need many qubits)",
            "reasoning": (
                "QAOA needs many qubits with reasonable fidelity. "
                "Neutral atom arrays offer high qubit counts with reconfigurable geometry. "
                "Native Rydberg interactions map well to optimization problems."
            ),
        },
    }

    print("\n\n(a) Application Recommendations:")
    print("-" * 70)
    for app, details in applications.items():
        print(f"\n  {app}:")
        print(f"    Best platform:  {details['best']}")
        print(f"    Key metric:     {details['key_metric']}")
        print(f"    Reasoning:      {details['reasoning']}")

    # (b) Most important metric per application
    print("\n(b) Most important metric per application:")
    metrics = {
        "Shor's RSA-2048": "Qubit count (need millions of physical qubits)",
        "Near-term VQE": "Gate fidelity (circuit quality matters more than depth)",
        "QKD Network": "Connectivity (need to transmit photons over fiber)",
        "QAOA": "Qubit count + connectivity (problem graph maps to hardware)",
    }
    for app, metric in metrics.items():
        print(f"    {app:<25} {metric}")

    # (c) 10x improvement impact
    print("\n(c) If you could improve ONE metric by 10x:")
    improvements = {
        "Superconducting": "Coherence (100us -> 1ms) - enables deeper circuits, better error correction",
        "Trapped Ion": "Gate speed (1us -> 100ns) - currently the bottleneck for scaling",
        "Photonic": "Gate fidelity (99% -> 99.9%) - enable fault-tolerant operations",
        "Neutral Atom": "Gate fidelity (99.5% -> 99.95%) - cross fault-tolerance threshold",
    }
    for platform, improvement in improvements.items():
        print(f"    {platform:<20} {improvement}")


# === Exercise 2: Quantum Volume Calculation ===
# Problem: Estimate Quantum Volume for different processor configurations.

def exercise_2():
    """Quantum Volume calculation and analysis."""
    print("\n" + "=" * 60)
    print("Exercise 2: Quantum Volume Calculation")
    print("=" * 60)

    def estimate_quantum_volume(
        n_physical: int,
        two_qubit_fidelity: float,
        connectivity: str = "nearest_neighbor",
    ) -> int:
        """
        Estimate Quantum Volume.
        QV = 2^n where n is the largest circuit that succeeds with
        heavy output probability > 2/3.

        For depth-n circuit on n qubits:
        - Each layer has ~n/2 two-qubit gates
        - For nearest-neighbor: need ~3n SWAP gates per layer
        - Total gates ~ n * (n/2 + swap_overhead)
        - Circuit fidelity ~ fidelity^(total_gates)
        """
        max_n = 0

        for n in range(1, n_physical + 1):
            # Gates per layer
            two_qubit_gates_per_layer = n // 2

            # SWAP overhead depends on connectivity
            if connectivity == "all_to_all":
                swap_overhead = 0
            elif connectivity == "nearest_neighbor":
                swap_overhead = 3 * n  # Rough estimate
            else:
                swap_overhead = n

            # Total depth = n layers, each with gates + swaps
            total_two_qubit_gates = n * (two_qubit_gates_per_layer + swap_overhead)

            # Overall circuit fidelity
            circuit_fidelity = two_qubit_fidelity ** total_two_qubit_gates

            # QV requires heavy output probability > 2/3
            # Approximate: need circuit fidelity > 2/3
            if circuit_fidelity > 2 / 3:
                max_n = n
            else:
                break

        return 2 ** max_n if max_n > 0 else 1

    # (a) Superconducting: 20 qubits, 99.5% fidelity, nearest-neighbor
    configs = [
        ("(a) SC, 99.5%, NN", 20, 0.995, "nearest_neighbor"),
        ("(b) SC, 99.9%, NN", 20, 0.999, "nearest_neighbor"),
        ("(c) Ion, 99.8%, A2A", 12, 0.998, "all_to_all"),
    ]

    print(f"\n  Quantum Volume estimation:")
    print(f"  {'Config':<25} {'Qubits':<8} {'Fidelity':<10} {'Connectivity':<15} {'QV'}")
    print("  " + "-" * 70)

    qv_results = {}
    for name, n_q, fid, conn in configs:
        qv = estimate_quantum_volume(n_q, fid, conn)
        qv_results[name] = qv
        n_eff = int(log2(qv)) if qv > 1 else 0
        print(f"  {name:<25} {n_q:<8} {fid:<10.3f} {conn:<15} 2^{n_eff} = {qv}")

    # (d) Is QV a good metric?
    print(f"\n(d) Is QV a good metric for comparing these processors?")
    print(f"    Pros:")
    print(f"      - Captures both qubit count AND gate quality")
    print(f"      - Accounts for connectivity overhead (SWAP gates)")
    print(f"      - Single number for easy comparison")
    print(f"    Cons:")
    print(f"      - Ignores coherence time (matters for variational algorithms)")
    print(f"      - Square circuit assumption may not match real algorithms")
    print(f"      - Different architectures have different optimal workloads")
    print(f"      - A trapped-ion processor with QV=64 may outperform")
    print(f"        a superconducting processor with QV=128 on specific tasks")
    print(f"    Verdict: QV is necessary but not sufficient for comparison")


# === Exercise 3: Fault-Tolerance Resource Estimation ===
# Problem: Surface code resource estimation for factoring RSA-2048.

def exercise_3():
    """Surface code fault-tolerance resource estimation."""
    print("\n" + "=" * 60)
    print("Exercise 3: Fault-Tolerance Resource Estimation")
    print("=" * 60)

    def surface_code_resources(p_physical: float, p_logical_target: float,
                                n_logical: int) -> Dict:
        """
        Estimate surface code resources.

        Surface code scaling:
            p_L ~ 0.1 * (100 * p)^((d+1)/2)
            Physical qubits per logical qubit: 2 * d^2
        """
        # Find minimum code distance d for target logical error rate
        # p_L = 0.1 * (100*p)^((d+1)/2) < p_target
        # (100*p)^((d+1)/2) < 10 * p_target
        # ((d+1)/2) * log(100*p) < log(10*p_target)

        factor = 100 * p_physical
        if factor >= 1:
            return {"error": "Physical error rate too high for surface code (need p < 0.01)"}

        log_factor = np.log(factor)
        log_target = np.log(10 * p_logical_target)

        # (d+1)/2 > log_target / log_factor  (note: log_factor < 0 so inequality flips)
        min_half_d = log_target / log_factor
        min_d = int(ceil(2 * min_half_d - 1))
        if min_d % 2 == 0:
            min_d += 1  # d must be odd

        min_d = max(min_d, 3)  # Minimum practical distance

        physical_per_logical = 2 * min_d ** 2
        total_physical = physical_per_logical * n_logical

        # Actual logical error rate with this d
        p_logical_actual = 0.1 * (100 * p_physical) ** ((min_d + 1) / 2)

        return {
            "code_distance": min_d,
            "physical_per_logical": physical_per_logical,
            "total_physical": total_physical,
            "p_logical_actual": p_logical_actual,
        }

    # RSA-2048 requires ~4000 logical qubits
    n_logical = 4000
    p_target = 1e-10

    print(f"\n  Target: {n_logical} logical qubits, p_L < {p_target}")
    print(f"  Surface code: p_L ~ 0.1 * (100*p)^((d+1)/2)")
    print(f"  Physical qubits per logical: 2*d^2")

    for p_label, p_phys in [("(a) p = 10^-3 (current SOA)", 1e-3),
                             ("(d) p = 10^-4 (improved)", 1e-4)]:
        result = surface_code_resources(p_phys, p_target, n_logical)
        print(f"\n  {p_label}:")

        if "error" in result:
            print(f"    {result['error']}")
        else:
            print(f"    (a) Code distance d = {result['code_distance']}")
            print(f"    (b) Physical qubits per logical: 2*{result['code_distance']}^2 "
                  f"= {result['physical_per_logical']:,}")
            print(f"    (c) Total physical qubits: {result['total_physical']:,} "
                  f"({result['total_physical']/1e6:.1f}M)")
            print(f"        Actual p_L = {result['p_logical_actual']:.2e}")

    # Comparison
    print(f"\n  Summary:")
    r1 = surface_code_resources(1e-3, p_target, n_logical)
    r2 = surface_code_resources(1e-4, p_target, n_logical)
    if "error" not in r1 and "error" not in r2:
        reduction = r1["total_physical"] / r2["total_physical"]
        print(f"    Improving p from 10^-3 to 10^-4 reduces physical qubits by {reduction:.1f}x")
        print(f"    From {r1['total_physical']/1e6:.1f}M to {r2['total_physical']/1e6:.1f}M")
        print(f"    Current state (2025): ~1,000 physical qubits")
        print(f"    Need: {r1['total_physical']/1e6:.0f}M at p=10^-3 (still ~{r1['total_physical']//1000:.0f}x gap)")


# === Exercise 4: Quantum Cloud Platform Comparison ===
# Problem: Compare quantum cloud platforms (analysis-based exercise).

def exercise_4():
    """Quantum cloud platform comparison (structured analysis)."""
    print("\n" + "=" * 60)
    print("Exercise 4: Quantum Cloud Platform Comparison")
    print("=" * 60)

    # (a) Bell state circuit in different SDKs
    print("\n(a) Bell state preparation in two SDKs:")

    print("""
    IBM Qiskit:
    ─────────────
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Transpile and run
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    service = QiskitRuntimeService()
    backend = service.least_busy()
    sampler = Sampler(backend)
    result = sampler.run(qc, shots=1024).result()

    Amazon Braket:
    ─────────────
    from braket.circuits import Circuit
    bell = Circuit().h(0).cnot(0, 1)

    from braket.aws import AwsDevice
    device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    task = device.run(bell, shots=1024)
    result = task.result()
    """)

    # (b) Programming model comparison
    print("(b) Programming model comparison:")
    comparison = {
        "Circuit definition": {
            "IBM": "QuantumCircuit class, gate-by-gate",
            "Braket": "Circuit class, method chaining",
        },
        "Transpilation": {
            "IBM": "Automatic via transpile() or runtime",
            "Braket": "Automatic device-specific compilation",
        },
        "Execution": {
            "IBM": "Sampler/Estimator primitives (2024+)",
            "Braket": "device.run() returns async task",
        },
        "Ecosystem": {
            "IBM": "Qiskit (largest community), Qiskit Nature, ML, Finance",
            "Braket": "Multi-hardware (IonQ, Rigetti, OQC), PennyLane integration",
        },
    }

    for aspect, platforms in comparison.items():
        print(f"\n  {aspect}:")
        for platform, desc in platforms.items():
            print(f"    {platform:<8} {desc}")

    # (c) Pricing estimate
    print("\n(c) Pricing estimate (10,000 shots, 10-qubit circuit):")
    pricing = {
        "IBM Quantum": {
            "model": "Free tier (10 min/month) + pay-as-you-go",
            "estimate": "~$1.60/task (at 10,000 shots on real hardware)",
            "note": "Free simulators available",
        },
        "Amazon Braket": {
            "model": "Per-task + per-shot pricing",
            "estimate": "~$0.30/task + $0.01/shot = ~$100.30 (IonQ)",
            "note": "Simulator: ~$0.075/task + $0.0000375/shot",
        },
    }

    for platform, details in pricing.items():
        print(f"\n  {platform}:")
        for key, val in details.items():
            print(f"    {key:<10} {val}")

    # (d) Recommendations
    print("\n(d) Recommendations:")
    print("    University course: IBM Quantum (free tier, largest documentation,")
    print("    extensive tutorials, Qiskit Textbook, active community)")
    print()
    print("    Startup: Amazon Braket (multi-hardware access lets you benchmark")
    print("    across platforms, AWS integration for production deployment)")


# === Exercise 5: Future Prediction Analysis ===
# Problem: Analyze quantum computing predictions and timelines.

def exercise_5():
    """Quantum computing future prediction analysis."""
    print("\n" + "=" * 60)
    print("Exercise 5: Future Prediction Analysis")
    print("=" * 60)

    # (a) Predictions from 2019-2020
    predictions = [
        {
            "year": 2019,
            "source": "Google (quantum supremacy paper)",
            "prediction": "Quantum advantage in optimization within 5 years",
            "target": 2024,
            "status": "Partially achieved (specific tasks only, not general optimization)",
        },
        {
            "year": 2020,
            "source": "IBM Quantum roadmap",
            "prediction": "1,000+ qubit processor by 2023",
            "target": 2023,
            "status": "Achieved (Condor: 1,121 qubits in 2023)",
        },
        {
            "year": 2019,
            "source": "NSF/DOE reports",
            "prediction": "Fault-tolerant QC within 10-15 years",
            "target": "2029-2034",
            "status": "On track but uncertain (error rates improving but slowly)",
        },
        {
            "year": 2020,
            "source": "McKinsey report",
            "prediction": "Quantum computing market > $1B by 2025",
            "target": 2025,
            "status": "Likely achieved (including services, software, hardware)",
        },
        {
            "year": 2020,
            "source": "Academic consensus",
            "prediction": "Quantum advantage for chemistry (drug discovery) by 2025",
            "target": 2025,
            "status": "Not yet achieved (classical simulation still competitive)",
        },
    ]

    print("\n(a,b) Predictions from 2019-2020 and their accuracy:")
    print("-" * 70)
    for p in predictions:
        print(f"\n  [{p['year']}] {p['source']}:")
        print(f"    Prediction: {p['prediction']}")
        print(f"    Target:     {p['target']}")
        print(f"    Status:     {p['status']}")

    # (c) 2030 predictions based on progress
    print("\n\n(c) Predictions for 2030 based on actual 2019-2025 progress:")
    print("-" * 70)

    predictions_2030 = [
        "10,000-100,000 physical qubits with 99.9%+ 2-qubit gate fidelity",
        "First demonstrations of logical qubit error rates below physical rates",
        "Quantum advantage in specific optimization and simulation problems",
        "Quantum-safe cryptography widely deployed (independent of QC progress)",
        "Quantum computing as a service (QCaaS) mature, integrated with cloud",
    ]

    for i, pred in enumerate(predictions_2030, 1):
        print(f"  {i}. {pred}")

    # (d) Three biggest obstacles
    print("\n\n(d) Three biggest technical obstacles:")
    obstacles = [
        {
            "obstacle": "Error rates and error correction overhead",
            "impact": "DELAY",
            "detail": (
                "Current best: ~10^-3 error rate. Surface code needs ~10^-4. "
                "Each logical qubit needs 1000+ physical qubits. "
                "Breakthrough in error correction codes could ACCELERATE."
            ),
        },
        {
            "obstacle": "Qubit connectivity and crosstalk",
            "impact": "DELAY",
            "detail": (
                "Most architectures have limited connectivity, requiring SWAP "
                "operations that degrade circuits. Solving this (e.g., via "
                "modular architectures) could ACCELERATE progress."
            ),
        },
        {
            "obstacle": "Quantum software and algorithm development",
            "impact": "ACCELERATE",
            "detail": (
                "Better algorithms (like the recent advances in quantum error "
                "correction protocols and fault-tolerant compilation) can "
                "dramatically reduce hardware requirements."
            ),
        },
    ]

    for obs in obstacles:
        print(f"\n  {obs['obstacle']} [{obs['impact']}]")
        print(f"    {obs['detail']}")

    # (e) Government investment recommendation
    print("\n\n(e) Government investment recommendation:")
    recommendations = [
        ("Fundamental research (40%)",
         "Error correction, new qubit architectures, quantum algorithms"),
        ("Workforce development (25%)",
         "University programs, postdoc funding, industry-academia partnerships"),
        ("Infrastructure (20%)",
         "Quantum networks, shared testbeds, cloud access for researchers"),
        ("Application development (10%)",
         "Industry partnerships for chemistry, materials, optimization"),
        ("Standards and security (5%)",
         "Post-quantum cryptography migration, quantum-safe standards"),
    ]

    for area, detail in recommendations:
        print(f"  {area}")
        print(f"    Focus: {detail}")

    print(f"\n  Rationale: Quantum computing is pre-competitive technology.")
    print(f"  Government should focus on foundational research and talent,")
    print(f"  while letting industry drive near-term applications.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
