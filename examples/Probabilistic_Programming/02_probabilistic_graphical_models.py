"""
Probabilistic Graphical Models Examples
- Bayesian networks, d-separation, variable elimination, belief propagation
"""
import numpy as np


# === Example 1: Student Bayesian Network ===
def student_bayesian_network():
    """Build and query the Student BN from Koller & Friedman."""
    p_d = {"easy": 0.6, "hard": 0.4}
    p_i = {"low": 0.7, "high": 0.3}
    p_g = {
        ("easy", "low"):  {"A": 0.3, "B": 0.4, "C": 0.3},
        ("easy", "high"): {"A": 0.9, "B": 0.08, "C": 0.02},
        ("hard", "low"):  {"A": 0.05, "B": 0.25, "C": 0.7},
        ("hard", "high"): {"A": 0.5, "B": 0.3, "C": 0.2},
    }
    p_letter = {"A": {"strong": 0.9, "weak": 0.1},
                "B": {"strong": 0.4, "weak": 0.6},
                "C": {"strong": 0.01, "weak": 0.99}}

    # P(Intelligence=high | Letter=strong) via variable elimination
    f_gi = {}
    for i in ["low", "high"]:
        f_gi[i] = {}
        for g in ["A", "B", "C"]:
            f_gi[i][g] = sum(p_d[d] * p_g[(d, i)][g] for d in ["easy", "hard"])

    f_i_given_l = {}
    for i in ["low", "high"]:
        f_i_given_l[i] = sum(f_gi[i][g] * p_letter[g]["strong"] for g in ["A", "B", "C"])

    unnorm = {i: p_i[i] * f_i_given_l[i] for i in ["low", "high"]}
    z = sum(unnorm.values())
    posterior = {i: unnorm[i] / z for i in ["low", "high"]}
    print("P(Intelligence | Letter=strong):")
    for i, p in posterior.items():
        print(f"  {i}: {p:.4f}")


# === Example 2: Explaining Away ===
def explaining_away():
    """Demonstrate the explaining-away phenomenon."""
    np.random.seed(42)
    n = 100000
    rain = np.random.binomial(1, 0.2, n)
    sprinkler = np.random.binomial(1, 0.3, n)
    p_wet = 0.99 * rain + 0.9 * sprinkler * (1 - rain) + 0.01 * (1 - rain) * (1 - sprinkler)
    wet = np.random.binomial(1, np.clip(p_wet, 0, 1))

    corr_uncond = np.corrcoef(rain, sprinkler)[0, 1]
    mask = wet == 1
    corr_cond = np.corrcoef(rain[mask], sprinkler[mask])[0, 1]
    print(f"\nExplaining Away:")
    print(f"  Corr(Rain, Sprinkler) unconditional: {corr_uncond:.4f}")
    print(f"  Corr(Rain, Sprinkler) | Wet=1:       {corr_cond:.4f}")


# === Example 3: Belief Propagation on Chain ===
def belief_propagation_chain():
    """BP on a 3-node chain: X1 — X2 — X3."""
    phi = {1: np.array([0.3, 0.7]), 2: np.array([0.5, 0.5]), 3: np.array([0.8, 0.2])}
    psi_12 = np.array([[0.9, 0.1], [0.2, 0.8]])
    psi_23 = np.array([[0.7, 0.3], [0.4, 0.6]])

    m_12 = phi[1] @ psi_12
    m_12 /= m_12.sum()
    m_32 = phi[3] @ psi_23.T
    m_32 /= m_32.sum()
    belief_2 = phi[2] * m_12 * m_32
    belief_2 /= belief_2.sum()
    print(f"\nBelief Propagation on Chain:")
    print(f"  Belief at X2: {belief_2}")


if __name__ == "__main__":
    student_bayesian_network()
    explaining_away()
    belief_propagation_chain()
