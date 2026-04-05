# 02. 확률적 그래프 모델(Probabilistic Graphical Models)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 2번째

[이전: 베이지안 사고](./01_Bayesian_Thinking.md) | [다음: MCMC 기초](./03_MCMC_Fundamentals.md)

---

> **프레임워크 참고**: 이 레슨은 그래프 연산에 NumPy와 networkx를 사용합니다.
> PGM 개념은 모든 확률적 프로그래밍 프레임워크의 기반입니다.
>
> 설치: `pip install numpy networkx matplotlib pgmpy`

## 학습 목표(Learning Objectives)

- 베이지안 네트워크와 그 방향 비순환 그래프(DAG) 표현 이해
- 마르코프 랜덤 필드(비방향 그래프 모델) 학습
- DAG에서 조건부 독립을 읽어내는 d-분리 마스터
- 소규모 베이지안 네트워크에서 정확 추론 구현
- PGM 개념과 확률적 프로그래밍의 연결

---

## 1. 왜 그래프 모델인가?(Why Graphical Models?)

실세계 시스템은 많은 상호 관련된 확률 변수를 포함합니다. 확률적 그래프 모델(PGM)은 **조건부 독립** 구조를 활용하여 많은 변수에 대한 결합 분포를 표현하고 추론하는 원칙적인 방법을 제공합니다.

### 1.1 차원의 저주(The Curse of Dimensionality)

```python
import numpy as np

# A joint distribution over n binary variables needs 2^n - 1 parameters
for n in [5, 10, 20, 30]:
    params = 2**n - 1
    print(f"n={n:2d} variables → {params:>12,d} parameters (full joint)")

# With conditional independence, we can factorize and drastically reduce parameters
# A Bayesian network with at most k parents per node: O(n * 2^k)
k = 3
for n in [5, 10, 20, 30]:
    params_bn = n * 2**k
    print(f"n={n:2d}, k={k} → {params_bn:>6d} parameters (Bayesian network)")
```

### 1.2 두 가지 유형의 그래프 모델(Two Types of Graphical Models)

| 속성 | 베이지안 네트워크 | 마르코프 랜덤 필드 |
|------|-----------------|-------------------|
| 그래프 유형 | 방향 비순환 그래프(DAG) | 비방향 그래프 |
| 인수분해 | 조건부 확률 테이블 | 포텐셜 함수 |
| 의미론 | 인과적 / 생성적 | 연관적 / 제약 |
| 정규화 | 이미 정규화됨 | 분배 함수 Z 필요 |
| 사용 예 | 의료 진단, 유전자 조절 | 이미지 분할, 물리학 |

---

## 2. 베이지안 네트워크(Bayesian Networks)

베이지안 네트워크(BN)는 각 노드가 확률 변수를, 각 간선이 직접적인 확률적 의존성을 나타내는 DAG입니다.

### 2.1 인수분해(Factorization)

결합 분포는 다음과 같이 인수분해됩니다:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i | \text{Parents}(X_i))$$

```python
import networkx as nx
import matplotlib.pyplot as plt

# Example: Student network
# Difficulty → Grade ← Intelligence
# Intelligence → SAT
# Grade → Letter

G = nx.DiGraph()
edges = [
    ("Difficulty", "Grade"),
    ("Intelligence", "Grade"),
    ("Intelligence", "SAT"),
    ("Grade", "Letter"),
]
G.add_edges_from(edges)

# The joint factorizes as:
# P(D, I, G, S, L) = P(D) * P(I) * P(G|D,I) * P(S|I) * P(L|G)
# 5 variables but only 5 conditional probability tables

pos = {
    "Difficulty": (0, 1),
    "Intelligence": (2, 1),
    "Grade": (1, 0),
    "SAT": (3, 0),
    "Letter": (1, -1),
}
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=2000, font_size=10, arrows=True,
        arrowsize=20, ax=ax)
ax.set_title("Student Bayesian Network")
plt.tight_layout()
plt.savefig("student_bn.png", dpi=100)
plt.show()
```

### 2.2 조건부 확률 테이블(Conditional Probability Tables, CPTs)

```python
# Define the CPTs for the Student network
# Each CPT: P(child | parents)

# P(Difficulty)
p_difficulty = {"easy": 0.6, "hard": 0.4}

# P(Intelligence)
p_intelligence = {"low": 0.7, "high": 0.3}

# P(Grade | Difficulty, Intelligence)
p_grade = {
    ("easy", "low"):  {"A": 0.3, "B": 0.4, "C": 0.3},
    ("easy", "high"): {"A": 0.9, "B": 0.08, "C": 0.02},
    ("hard", "low"):  {"A": 0.05, "B": 0.25, "C": 0.7},
    ("hard", "high"): {"A": 0.5, "B": 0.3, "C": 0.2},
}

# P(SAT | Intelligence)
p_sat = {
    "low":  {"low_score": 0.95, "high_score": 0.05},
    "high": {"low_score": 0.2, "high_score": 0.8},
}

# P(Letter | Grade)
p_letter = {
    "A": {"strong": 0.9, "weak": 0.1},
    "B": {"strong": 0.4, "weak": 0.6},
    "C": {"strong": 0.01, "weak": 0.99},
}
```

### 2.3 결합 확률과 주변 확률 계산(Computing Joint and Marginal Probabilities)

```python
def compute_joint_probability(d, i, g, s, l):
    """Compute P(D=d, I=i, G=g, S=s, L=l)."""
    return (p_difficulty[d] *
            p_intelligence[i] *
            p_grade[(d, i)][g] *
            p_sat[i][s] *
            p_letter[g][l])

# Example: P(D=hard, I=high, G=A, S=high_score, L=strong)
prob = compute_joint_probability("hard", "high", "A", "high_score", "strong")
print(f"P(hard, high, A, high_score, strong) = {prob:.6f}")

# Marginal: P(Letter = strong) by summing over all other variables
p_strong = 0
for d in p_difficulty:
    for i in p_intelligence:
        for g in ["A", "B", "C"]:
            for s in p_sat[i]:
                p_strong += compute_joint_probability(d, i, g, s, "strong")
print(f"P(Letter=strong) = {p_strong:.4f}")
```

---

## 3. D-분리(D-Separation)

D-분리는 DAG에서 조건부 독립을 읽어내는 그래프 기준입니다. 확률적 그래프 모델에서 가장 중요한 개념 중 하나입니다.

### 3.1 세 가지 표준 구조(Three Canonical Structures)

```python
def visualize_canonical_structures():
    """Visualize the three canonical structures in a BN."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Chain: A → B → C
    G1 = nx.DiGraph([("A", "B"), ("B", "C")])
    pos1 = {"A": (0, 0), "B": (1, 0), "C": (2, 0)}
    nx.draw(G1, pos1, with_labels=True, node_color="lightgreen",
            node_size=1500, arrows=True, arrowsize=20, ax=axes[0])
    axes[0].set_title("Chain: A → B → C\nA ⊥ C | B")

    # 2. Fork: A ← B → C
    G2 = nx.DiGraph([("B", "A"), ("B", "C")])
    pos2 = {"A": (0, 0), "B": (1, 1), "C": (2, 0)}
    nx.draw(G2, pos2, with_labels=True, node_color="lightyellow",
            node_size=1500, arrows=True, arrowsize=20, ax=axes[1])
    axes[1].set_title("Fork: A ← B → C\nA ⊥ C | B")

    # 3. Collider: A → B ← C
    G3 = nx.DiGraph([("A", "B"), ("C", "B")])
    pos3 = {"A": (0, 1), "B": (1, 0), "C": (2, 1)}
    nx.draw(G3, pos3, with_labels=True, node_color="lightsalmon",
            node_size=1500, arrows=True, arrowsize=20, ax=axes[2])
    axes[2].set_title("Collider: A → B ← C\nA ⊥ C (but NOT A ⊥ C | B!)")

    plt.tight_layout()
    plt.savefig("canonical_structures.png", dpi=100)
    plt.show()

visualize_canonical_structures()
```

### 3.2 D-분리 규칙(D-Separation Rules)

노드 X와 Y 사이의 경로는 증거 집합 Z가 주어졌을 때, 다음을 포함하면 **차단**됩니다:

1. **체인(Chain)** A → B → C에서 B ∈ Z (매개자를 알면 흐름이 차단됨)
2. **포크(Fork)** A ← B → C에서 B ∈ Z (공통 원인을 알면 흐름이 차단됨)
3. **충돌체(Collider)** A → B ← C에서 B ∉ Z이고 B의 자손이 Z에 없음

X와 Y 사이의 **모든** 경로가 차단되면 X와 Y는 Z가 주어졌을 때 **d-분리**됩니다.

```python
def is_d_separated(graph, x, y, z_set):
    """
    Check d-separation using the Bayes Ball algorithm (simplified).
    Returns True if x ⊥ y | z_set in the given DAG.
    """
    # Use pgmpy for robust d-separation testing
    from pgmpy.models import BayesianNetwork
    from pgmpy.independencies import Independencies

    model = BayesianNetwork(graph.edges())
    independencies = model.get_independencies()

    # Check if (x ⊥ y | z_set) is in the independencies
    from pgmpy.independencies import IndependenceAssertion
    assertion = IndependenceAssertion(x, y, list(z_set))
    return assertion in independencies


# Student network d-separation examples
print("D-separation tests on Student network:")
print("=" * 50)

# Difficulty ⊥ Intelligence? (no path, so yes)
# Actually they share a child (Grade) — but unconditionally, they are independent
# because Grade is a collider and we haven't conditioned on it
print("D ⊥ I | {} → True (collider Grade not observed)")

# Difficulty ⊥ Intelligence | Grade? (conditioning on collider opens the path)
print("D ⊥ I | {Grade} → False (explaining away)")

# SAT ⊥ Difficulty | Intelligence? (Intelligence blocks the path)
print("SAT ⊥ D | {Intelligence} → True (chain blocked)")

# Letter ⊥ Intelligence? (path through Grade, which is NOT in Z)
print("L ⊥ I | {} → False (path through Grade is active)")
```

### 3.3 설명하기(Explaining Away / Collider Bias)

```python
def explaining_away_demo():
    """Demonstrate the explaining-away phenomenon."""
    # Two independent causes (rain, sprinkler) share a common effect (wet grass)
    # Conditioning on the effect makes the causes dependent

    n = 100000
    rain = np.random.binomial(1, 0.2, n)
    sprinkler = np.random.binomial(1, 0.3, n)

    # Wet grass: likely if rain OR sprinkler
    p_wet = 0.99 * rain + 0.9 * sprinkler * (1 - rain) + 0.01 * (1 - rain) * (1 - sprinkler)
    wet = np.random.binomial(1, np.clip(p_wet, 0, 1))

    # Unconditional: rain and sprinkler are independent
    corr_uncond = np.corrcoef(rain, sprinkler)[0, 1]
    print(f"Correlation(Rain, Sprinkler) unconditional: {corr_uncond:.4f}")

    # Conditional on Wet=1: rain and sprinkler become negatively correlated
    mask = wet == 1
    corr_cond = np.corrcoef(rain[mask], sprinkler[mask])[0, 1]
    print(f"Correlation(Rain, Sprinkler) | Wet=1:       {corr_cond:.4f}")
    print("→ Negative! If grass is wet and it rained, sprinkler is less likely.")

explaining_away_demo()
```

---

## 4. 베이지안 네트워크에서의 정확 추론(Exact Inference in Bayesian Networks)

### 4.1 변수 소거(Variable Elimination)

베이지안 네트워크에서 정확 추론을 위한 핵심 알고리즘입니다. 인수분해를 활용하여 전체 결합을 계산하지 않습니다.

```python
def variable_elimination_example():
    """Compute P(Intelligence=high | Letter=strong) by variable elimination."""
    # Elimination order: D, S, G (eliminate in reverse topological order, keeping I)

    # Factor 1: P(D)
    f_d = {"easy": 0.6, "hard": 0.4}

    # Factor 2: P(I)
    f_i = {"low": 0.7, "high": 0.3}

    # Factor 3: P(G | D, I)
    f_g = p_grade  # already defined above

    # Factor 4: P(S | I)
    f_s = p_sat

    # Factor 5: P(L=strong | G)  — evidence: Letter=strong
    f_l_evidence = {g: p_letter[g]["strong"] for g in ["A", "B", "C"]}

    # Step 1: Eliminate S (not connected to query or evidence through remaining factors)
    # S only appears in f_s, and summing over S gives 1. So we skip it.

    # Step 2: Eliminate D
    # Combine f_d and f_g, sum over D
    f_gi = {}  # P(G|I) after marginalizing D
    for i in ["low", "high"]:
        f_gi[i] = {}
        for g in ["A", "B", "C"]:
            f_gi[i][g] = sum(f_d[d] * f_g[(d, i)][g] for d in ["easy", "hard"])

    # Step 3: Eliminate G
    # Combine f_gi with f_l_evidence, sum over G
    f_i_given_l = {}
    for i in ["low", "high"]:
        f_i_given_l[i] = sum(f_gi[i][g] * f_l_evidence[g] for g in ["A", "B", "C"])

    # Step 4: Multiply by prior P(I) and normalize
    unnormalized = {i: f_i[i] * f_i_given_l[i] for i in ["low", "high"]}
    z = sum(unnormalized.values())
    posterior = {i: unnormalized[i] / z for i in ["low", "high"]}

    print("P(Intelligence | Letter=strong):")
    for i, p in posterior.items():
        print(f"  {i}: {p:.4f}")

variable_elimination_example()
```

### 4.2 pgmpy를 사용한 추론(Using pgmpy for Inference)

```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Build the Student BN with pgmpy
model = BayesianNetwork([
    ("Difficulty", "Grade"),
    ("Intelligence", "Grade"),
    ("Intelligence", "SAT"),
    ("Grade", "Letter"),
])

cpd_d = TabularCPD("Difficulty", 2, [[0.6], [0.4]],
                    state_names={"Difficulty": ["easy", "hard"]})
cpd_i = TabularCPD("Intelligence", 2, [[0.7], [0.3]],
                    state_names={"Intelligence": ["low", "high"]})
cpd_g = TabularCPD("Grade", 3,
    [[0.3, 0.9, 0.05, 0.5],   # A
     [0.4, 0.08, 0.25, 0.3],  # B
     [0.3, 0.02, 0.7, 0.2]],  # C
    evidence=["Difficulty", "Intelligence"],
    evidence_card=[2, 2],
    state_names={"Grade": ["A", "B", "C"],
                 "Difficulty": ["easy", "hard"],
                 "Intelligence": ["low", "high"]})
cpd_s = TabularCPD("SAT", 2,
    [[0.95, 0.2],
     [0.05, 0.8]],
    evidence=["Intelligence"], evidence_card=[2],
    state_names={"SAT": ["low_score", "high_score"],
                 "Intelligence": ["low", "high"]})
cpd_l = TabularCPD("Letter", 2,
    [[0.9, 0.4, 0.01],
     [0.1, 0.6, 0.99]],
    evidence=["Grade"], evidence_card=[3],
    state_names={"Letter": ["strong", "weak"],
                 "Grade": ["A", "B", "C"]})

model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_s, cpd_l)
assert model.check_model()

# Inference
inference = VariableElimination(model)
result = inference.query(["Intelligence"], evidence={"Letter": "strong"})
print(result)

# Query with multiple evidence
result2 = inference.query(["Grade"], evidence={"Difficulty": "hard", "Intelligence": "high"})
print(result2)
```

---

## 5. 마르코프 랜덤 필드(Markov Random Fields)

마르코프 랜덤 필드(MRF)는 비방향 그래프 모델이라고도 하며, 비방향 간선으로 대칭적 의존성을 표현합니다.

### 5.1 구조와 인수분해(Structure and Factorization)

MRF는 결합 분포를 클리크에 대한 **포텐셜 함수**의 곱으로 인수분해합니다:

$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{c \in \text{cliques}} \psi_c(X_c)$$

여기서 $Z = \sum_x \prod_c \psi_c(x_c)$는 분배 함수입니다.

```python
def mrf_example():
    """Simple 4-node MRF for image denoising."""
    # Grid MRF: 2x2 binary image
    # Nodes: (0,0), (0,1), (1,0), (1,1)
    # Edges: horizontal and vertical neighbors

    # Unary potentials (data term): favor observed noisy pixel values
    observed = np.array([[1, 0], [1, 1]])

    # Pairwise potentials: neighbors should agree (smoothness)
    lam = 2.0  # smoothness strength

    def unary_potential(x, obs):
        """Higher potential when x agrees with observation."""
        return np.exp(-0.5 * (x - obs)**2)

    def pairwise_potential(xi, xj):
        """Higher potential when neighbors agree."""
        return np.exp(lam * (xi == xj))

    # Brute-force: enumerate all 2^4 = 16 configurations
    best_config = None
    best_score = -np.inf
    configs = []

    for x00 in [0, 1]:
        for x01 in [0, 1]:
            for x10 in [0, 1]:
                for x11 in [0, 1]:
                    config = np.array([[x00, x01], [x10, x11]])
                    # Unary terms
                    score = np.log(unary_potential(x00, observed[0, 0]))
                    score += np.log(unary_potential(x01, observed[0, 1]))
                    score += np.log(unary_potential(x10, observed[1, 0]))
                    score += np.log(unary_potential(x11, observed[1, 1]))
                    # Pairwise terms (horizontal + vertical)
                    score += np.log(pairwise_potential(x00, x01))
                    score += np.log(pairwise_potential(x10, x11))
                    score += np.log(pairwise_potential(x00, x10))
                    score += np.log(pairwise_potential(x01, x11))

                    configs.append((config.copy(), score))
                    if score > best_score:
                        best_score = score
                        best_config = config.copy()

    print(f"Observed image:\n{observed}")
    print(f"MAP denoised image:\n{best_config}")
    print(f"MAP score: {best_score:.4f}")

mrf_example()
```

### 5.2 인수 그래프(Factor Graphs)

인수 그래프는 방향과 비방향 모델을 모두 포함하는 통합된 표현을 제공합니다.

```python
def visualize_factor_graph():
    """Visualize a factor graph."""
    G = nx.Graph()

    # Variable nodes (circles)
    variables = ["X1", "X2", "X3"]
    # Factor nodes (squares)
    factors = ["f1", "f12", "f23", "f3"]

    G.add_nodes_from(variables, bipartite=0)
    G.add_nodes_from(factors, bipartite=1)

    G.add_edges_from([
        ("X1", "f1"), ("X1", "f12"),
        ("X2", "f12"), ("X2", "f23"),
        ("X3", "f23"), ("X3", "f3"),
    ])

    pos = {
        "X1": (0, 0), "X2": (2, 0), "X3": (4, 0),
        "f1": (0, 1), "f12": (1, 1), "f23": (3, 1), "f3": (4, 1),
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    nx.draw_networkx_nodes(G, pos, nodelist=variables, node_shape='o',
                           node_color='lightblue', node_size=1500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=factors, node_shape='s',
                           node_color='lightyellow', node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    ax.set_title("Factor Graph: P(X1,X2,X3) = f1(X1) · f12(X1,X2) · f23(X2,X3) · f3(X3)")
    plt.tight_layout()
    plt.savefig("factor_graph.png", dpi=100)
    plt.show()

visualize_factor_graph()
```

---

## 6. 믿음 전파(Belief Propagation)

믿음 전파(BP)는 그래프 모델에서 추론을 위한 주요 메시지 전달 알고리즘입니다.

### 6.1 트리에서의 합-곱 알고리즘(Sum-Product Algorithm on Trees)

```python
def belief_propagation_chain():
    """Belief propagation on a 3-node chain: X1 — X2 — X3."""
    # Unary potentials
    phi = {
        1: np.array([0.3, 0.7]),   # P(X1)
        2: np.array([0.5, 0.5]),   # P(X2) - uniform
        3: np.array([0.8, 0.2]),   # P(X3)
    }

    # Pairwise potentials (compatibility)
    psi_12 = np.array([[0.9, 0.1],
                        [0.2, 0.8]])  # prefer agreement
    psi_23 = np.array([[0.7, 0.3],
                        [0.4, 0.6]])

    # Message X1 → X2: m12[x2] = sum_x1 phi_1(x1) * psi_12(x1, x2)
    m_12 = phi[1] @ psi_12
    m_12 /= m_12.sum()  # normalize
    print(f"Message X1→X2: {m_12}")

    # Message X3 → X2: m32[x2] = sum_x3 phi_3(x3) * psi_23^T(x3, x2)
    m_32 = phi[3] @ psi_23.T
    m_32 /= m_32.sum()
    print(f"Message X3→X2: {m_32}")

    # Belief at X2: b(x2) ∝ phi_2(x2) * m_12(x2) * m_32(x2)
    belief_2 = phi[2] * m_12 * m_32
    belief_2 /= belief_2.sum()
    print(f"Belief at X2:  {belief_2}")
    print(f"  P(X2=0) = {belief_2[0]:.4f}, P(X2=1) = {belief_2[1]:.4f}")

belief_propagation_chain()
```

### 6.2 순환 믿음 전파(Loopy Belief Propagation)

그래프에 순환이 있을 때, 수렴할 때까지 메시지 전달을 반복합니다. 이는 근사적이지만 실무에서 종종 잘 작동합니다.

```python
def loopy_bp_grid(size=4, n_iters=20):
    """Loopy BP on a grid MRF (Ising model)."""
    n = size * size
    # Random unary potentials (log scale)
    np.random.seed(42)
    unary = np.random.randn(size, size) * 0.5

    # Pairwise: Ising coupling strength
    J = 0.5

    # Messages: stored as log-ratios for numerical stability
    # message[i][j] = log(m_{i→j}(+1) / m_{i→j}(-1))
    messages = {}
    neighbors = {}

    for i in range(size):
        for j in range(size):
            node = (i, j)
            nbrs = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    nbrs.append((ni, nj))
                    messages[(node, (ni, nj))] = 0.0
            neighbors[node] = nbrs

    # Iterate
    for iteration in range(n_iters):
        max_change = 0
        new_messages = {}
        for node in neighbors:
            for nbr in neighbors[node]:
                # Message from node to nbr
                # m_{node→nbr}(x_nbr) ∝ sum_{x_node} psi(x_node,x_nbr) phi(x_node) prod_{k≠nbr} m_{k→node}(x_node)
                incoming = unary[node[0], node[1]]
                for k in neighbors[node]:
                    if k != nbr:
                        incoming += messages.get((k, node), 0.0)

                # For binary Ising: message = atanh(tanh(J) * tanh(incoming))
                new_msg = np.arctanh(np.tanh(J) * np.tanh(incoming))
                new_messages[(node, nbr)] = new_msg
                max_change = max(max_change, abs(new_msg - messages[(node, nbr)]))

        messages = new_messages
        if iteration % 5 == 0 or max_change < 1e-6:
            print(f"Iter {iteration}: max message change = {max_change:.6f}")
        if max_change < 1e-6:
            break

    # Compute beliefs (marginals)
    beliefs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            node = (i, j)
            log_ratio = unary[i, j]
            for k in neighbors[node]:
                log_ratio += messages[(k, node)]
            beliefs[i, j] = 1 / (1 + np.exp(-2 * log_ratio))  # P(X=+1)

    print(f"\nMarginal beliefs P(X=+1):\n{beliefs.round(3)}")
    return beliefs

beliefs = loopy_bp_grid()
```

---

## 7. 마르코프 블랭킷과 지역 계산(Markov Blanket and Local Computation)

### 7.1 마르코프 블랭킷(Markov Blanket)

베이지안 네트워크에서 노드의 마르코프 블랭킷은 그 부모, 자식, 공동 부모(자식의 다른 부모)로 구성됩니다. 마르코프 블랭킷이 조건부로 주어지면, 노드는 다른 모든 노드와 독립입니다.

```python
def markov_blanket(dag, node):
    """Compute the Markov blanket of a node in a DAG."""
    parents = set(dag.predecessors(node))
    children = set(dag.successors(node))
    coparents = set()
    for child in children:
        coparents.update(dag.predecessors(child))
    coparents.discard(node)

    blanket = parents | children | coparents
    return blanket


# Student network
blanket_grade = markov_blanket(G, "Grade")
print(f"Markov blanket of Grade: {blanket_grade}")
# Should be: {Difficulty, Intelligence, Letter}

blanket_intelligence = markov_blanket(G, "Intelligence")
print(f"Markov blanket of Intelligence: {blanket_intelligence}")
# Should be: {Grade, Difficulty, SAT}
```

### 7.2 마르코프 블랭킷이 MCMC에 중요한 이유(Why Markov Blankets Matter for MCMC)

깁스 샘플링(레슨 03)에서, 각 변수를 전조건부 분포에서 샘플링하는데, 이는 **마르코프 블랭킷에만** 의존합니다:

$$P(X_i | X_{-i}) = P(X_i | \text{MB}(X_i))$$

이를 통해 각 깁스 단계가 **지역** 계산이 됩니다.

---

## 8. 플레이트 표기법(Plate Notation)

플레이트 표기법은 그래프 모델에서 반복 구조(i.i.d. 데이터)를 간결하게 표현하는 방법입니다.

```python
def plate_notation_example():
    """Visualize plate notation for a simple Bayesian model."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw the plate (rectangle)
    rect = plt.Rectangle((0.5, 0.2), 3, 2.5, fill=False,
                          linestyle='--', linewidth=2, edgecolor='gray')
    ax.add_patch(rect)
    ax.text(3.2, 0.3, "N", fontsize=14, color='gray')

    # Nodes
    circle_params = dict(radius=0.3, fill=False, linewidth=2)

    # mu (outside plate - shared parameter)
    mu = plt.Circle((2, 4), **circle_params, edgecolor='blue')
    ax.add_patch(mu)
    ax.text(1.85, 3.9, "μ", fontsize=16, ha='center')

    # sigma (outside plate)
    sigma = plt.Circle((3.5, 4), **circle_params, edgecolor='blue')
    ax.add_patch(sigma)
    ax.text(3.35, 3.9, "σ", fontsize=16, ha='center')

    # y_i (inside plate - observed data)
    yi = plt.Circle((2, 1.5), **circle_params, edgecolor='black')
    ax.add_patch(yi)
    ax.text(1.85, 1.4, "yᵢ", fontsize=16, ha='center')
    # Shade to indicate observed
    yi_filled = plt.Circle((2, 1.5), 0.3, color='lightgray', zorder=0)
    ax.add_patch(yi_filled)

    # Arrows
    ax.annotate("", xy=(2, 1.85), xytext=(2, 3.65),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(2.25, 1.7), xytext=(3.25, 3.75),
                arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.set_title("Plate Notation: yᵢ ~ Normal(μ, σ²) for i=1,...,N")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("plate_notation.png", dpi=100)
    plt.show()

plate_notation_example()
```

---

## 9. PGM에서 확률적 프로그램으로(From PGMs to Probabilistic Programs)

PGM과 확률적 프로그래밍을 연결하는 핵심 통찰: 확률적 프로그램은 코드 구조를 통해 **암묵적으로** 그래프 모델을 정의합니다.

```python
# A PyMC model is essentially a Bayesian network expressed as code
# (Preview of Lesson 04)

# PGM-style specification:
# mu ~ Normal(0, 10)
# sigma ~ HalfNormal(5)
# for i in 1..N:
#     y_i ~ Normal(mu, sigma)

# Probabilistic program equivalent:
# import pymc as pm
# with pm.Model():
#     mu = pm.Normal("mu", mu=0, sigma=10)
#     sigma = pm.HalfNormal("sigma", sigma=5)
#     y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
#     trace = pm.sample()

# The compiler/runtime automatically:
# 1. Builds the computational graph
# 2. Computes log-probability
# 3. Runs MCMC or VI for inference
```

### 9.1 수동 PGM 대비 확률적 프로그램의 장점(Advantages of Probabilistic Programs over Manual PGMs)

| 수동 PGM | 확률적 프로그램 |
|---------|--------------|
| CPT를 수동으로 열거 | 매개변수를 분포로 정의 |
| 고정된 구조 | 확률적 제어 흐름 (확률 변수에 대한 if/else) |
| 대부분 이산 변수 | 연속 + 이산 원활하게 혼합 |
| 추론 알고리즘 별도 | 프레임워크에 추론 내장 |
| 확장 어려움 | 자동 미분 + GPU 가속 |

---

## 10. 조건부 독립 검정(Conditional Independence Testing)

```python
def conditional_independence_test(data, x_col, y_col, z_cols=None, alpha=0.05):
    """
    Test conditional independence X ⊥ Y | Z using partial correlation.
    (Assuming Gaussian distributions for simplicity.)
    """
    from scipy.stats import pearsonr

    if z_cols is None or len(z_cols) == 0:
        # Unconditional test
        r, p = pearsonr(data[x_col], data[y_col])
        independent = p > alpha
        print(f"{x_col} ⊥ {y_col}: r={r:.4f}, p={p:.4f} → {'Independent' if independent else 'Dependent'}")
        return independent
    else:
        # Partial correlation: regress out Z from both X and Y
        from numpy.linalg import lstsq
        Z = data[z_cols].values
        Z_aug = np.column_stack([Z, np.ones(len(Z))])

        x_resid = data[x_col].values - Z_aug @ lstsq(Z_aug, data[x_col].values, rcond=None)[0]
        y_resid = data[y_col].values - Z_aug @ lstsq(Z_aug, data[y_col].values, rcond=None)[0]

        r, p = pearsonr(x_resid, y_resid)
        independent = p > alpha
        print(f"{x_col} ⊥ {y_col} | {z_cols}: r={r:.4f}, p={p:.4f} → "
              f"{'Independent' if independent else 'Dependent'}")
        return independent

# Example: Generate data from A → B → C
import pandas as pd
np.random.seed(42)
n = 1000
A = np.random.randn(n)
B = 0.8 * A + np.random.randn(n) * 0.5
C = 0.6 * B + np.random.randn(n) * 0.5
df = pd.DataFrame({"A": A, "B": B, "C": C})

conditional_independence_test(df, "A", "C")            # Dependent (unconditional)
conditional_independence_test(df, "A", "C", ["B"])      # Independent (chain blocked)
```

---

## 요약(Summary)

| 개념 | 핵심 요점 |
|------|---------|
| 베이지안 네트워크 | 조건부 독립을 인코딩하는 DAG; 결합 = CPT의 곱 |
| 마르코프 랜덤 필드 | 포텐셜 함수가 있는 비방향 모델; 분배 함수 필요 |
| D-분리 | DAG에서 조건부 독립의 그래프 기준 |
| 충돌체 | 충돌체(또는 그 자손)를 조건부로 하면 경로가 열림 |
| 변수 소거 | 한 번에 하나의 변수를 주변화하는 정확 추론 |
| 믿음 전파 | 메시지 전달 알고리즘; 트리에서 정확, 순환 그래프에서 근사 |
| 마르코프 블랭킷 | 부모 + 자식 + 공동 부모; 지역 계산에 충분 |
| 플레이트 표기법 | 반복(i.i.d.) 구조의 간결한 표현 |
| PGM → PPL | 확률적 프로그램은 암묵적으로 그래프 모델을 정의 |

---

## 참고 문헌(References)

1. Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Ch. 8. Springer.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Ch. 10. MIT Press.
4. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.

---

[이전: 베이지안 사고](./01_Bayesian_Thinking.md) | [다음: MCMC 기초 →](./03_MCMC_Fundamentals.md)
