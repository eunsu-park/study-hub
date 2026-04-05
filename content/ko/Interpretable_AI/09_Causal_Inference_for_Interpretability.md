# 레슨 9: 해석 가능성을 위한 인과 추론(Causal Inference for Interpretability)

[이전: 반사실적 설명](./08_Counterfactual_Explanations.md) | [다음: 설명 평가](./10_Evaluating_Explanations.md)

---

## 학습 목표

- 구조적 인과 모델(Structural Causal Models, SCMs)과 방향 비순환 그래프(Directed Acyclic Graphs, DAGs)를 데이터 생성 과정의 형식적 표현으로 이해한다
- Pearl의 do-계산법(do-calculus)을 사용하여 관측적 조건부 확률 P(Y|X=x)과 개입적 분포 P(Y|do(X=x))를 구분한다
- 상관된 특성이 SHAP 및 LIME과 같은 표준 기여도 방법을 어떻게 오도하는지 인식한다
- 인과적으로 근거가 있는 특성 중요도를 위해 인과 SHAP(Causal SHAP)과 개입적 조건부 기대값(interventional conditional expectations)을 구현한다
- DoWhy 라이브러리를 적용하여 ML 파이프라인에서 인과 관계를 탐지하고 추론한다

---

## 1. 표준 설명이 오도할 수 있는 이유

### 1.1 상관관계 함정(The Correlation Trap)

지금까지 학습한 모든 설명 방법 -- SHAP, LIME, 적분 그래디언트(Integrated Gradients), GradCAM -- 은 근본적으로 *관측적* 질문에 답한다: "관측된 분포에 따라 특성 값을 변화시킬 때 모델의 예측이 어떻게 변하는가?" 이것이 문제가 되는 이유는 실제 데이터셋의 특성들이 상관되어 있고, 이러한 상관관계가 오도하는 기여도를 만들어낼 수 있기 때문이다.

폐렴 위험을 예측하는 의료 진단 모델을 생각해보자. 훈련 데이터에 가짜 상관관계가 있다고 가정하자: 구급차로 도착한 환자는 흉부 X선을 촬영할 가능성이 더 높고, 한 병원의 X선 스캐너가 미묘한 워터마크를 삽입한다. 모델은 워터마크를 높은 폐렴 위험과 연관짓는 것을 학습한다. 표준 SHAP은 "워터마크가 포함된 픽셀 영역"이 중요하다고 충실히 보고할 것이다 -- 모델이 실제로 이를 사용하기 때문이다. 하지만 *인과적* 설명은 다르다: 워터마크는 폐렴을 유발하지 않으며, 교란된 인공물(confounded artifact)이다.

```python
"""
Illustration: correlated features produce misleading SHAP values.

Scenario: We predict house price. Square footage (X1) causes price (Y).
Number of bathrooms (X2) is correlated with square footage but adds
little independent causal effect. Standard SHAP may over-attribute
importance to X2 because it conditions on the observed distribution
where X1 and X2 move together.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import shap

np.random.seed(42)

# --- Data generation with known causal structure ---
# True causal model: X1 (sqft) -> Y (price)
#                    X1 -> X2 (bathrooms, caused by sqft)
# X2 has NO direct causal effect on Y, only through X1
n = 2000
X1 = np.random.normal(1500, 400, n)          # square footage
X2 = 0.002 * X1 + np.random.normal(0, 0.3, n)  # bathrooms ~ f(sqft)
noise = np.random.normal(0, 10000, n)

# True price depends ONLY on sqft, not bathrooms
Y = 150 * X1 + noise

df = pd.DataFrame({"sqft": X1, "bathrooms": X2, "price": Y})

# --- Train model ---
model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(df[["sqft", "bathrooms"]], df["price"])

# --- Standard SHAP ---
explainer = shap.Explainer(model, df[["sqft", "bathrooms"]])
shap_values = explainer(df[["sqft", "bathrooms"]])

print("Mean |SHAP| per feature:")
print(f"  sqft:      {np.abs(shap_values.values[:, 0]).mean():.1f}")
print(f"  bathrooms: {np.abs(shap_values.values[:, 1]).mean():.1f}")

# PROBLEM: bathrooms gets non-trivial SHAP importance even though
# it has ZERO causal effect on price. This happens because SHAP
# uses the conditional/marginal distribution where sqft and bathrooms
# are correlated. When bathrooms is "removed" (replaced with marginal
# samples), the model loses information about sqft through the
# correlation, making bathrooms appear important.
```

### 1.2 근본 원인: 조건부 확률 vs 개입(Conditioning vs Intervening)

근본적인 문제는 두 가지 질문 사이의 차이이다:

1. **관측적(Observational)**: "욕실이 3개인 집들 중에서 기대 가격은 얼마인가?"
   이는 데이터에 대한 조건부 확률이다: P(Y | X2 = 3).
2. **개입적(Interventional)**: "다른 모든 것을 자연스러운 상태로 두면서 욕실을 3개로 *설정*하면, 기대 가격은 얼마인가?" 이것은 *개입*이다:
   P(Y | do(X2 = 3)).

관측적 경우에서 욕실 = 3이라는 것을 알면 sqft에 대한 정보도 얻게 된다(상관관계가 있으므로), 따라서 기대 가격이 더 높다. 개입적 경우에서 욕실을 3으로 설정해도 sqft는 변하지 않으므로, 가격에 대한 효과는 0이다.

표준 SHAP은 관측적 질문에 답한다. 인과 SHAP(Causal SHAP)은 개입적 질문에 답한다. 이 차이는 특성들이 인과적으로 관련되어 있을 때 중요하다.

### 1.3 실무에서 이것이 중요한 경우

관측적-개입적 차이가 큰 경우:

- 특성들이 강한 인과적 의존성을 가질 때 (예: 나이 -> 경력 -> 급여)
- 모델이 교란 변수로부터 가짜 상관관계를 학습했을 때
- 이해관계자들이 *실행 가능한* 설명을 필요로 할 때 ("다른 결과를 얻으려면 무엇을 바꿔야 하는가?")
- 규제 요건이 상관관계적 추론이 아닌 인과적 추론을 요구할 때

특성들이 독립적일 때(인과 관계가 없을 때), 관측적 설명과 개입적 설명은 일치하며, 표준 SHAP이 인과적으로 유효하다.

---

## 2. 구조적 인과 모델(Structural Causal Models, SCMs)

### 2.1 정의

구조적 인과 모델 M = (U, V, F, P(U))은 네 가지 구성 요소로 이루어진다:

- **U**: 외생(exogenous) 변수 -- 모델 외부의 무작위성 원천
- **V**: 내생(endogenous) 변수 -- 모델링하는 변수
- **F**: 구조 방정식(structural equations) -- 결정론적 함수 V_i = f_i(pa_i, U_i), 여기서 pa_i는 V_i의 직접 원인(부모)
- **P(U)**: 외생 변수에 대한 분포

구조 방정식은 방향 비순환 그래프(DAG)를 정의하며, 간선은 직접적 인과 관계를 나타낸다.

```python
"""
Formal representation of a Structural Causal Model.

SCM for the house price example:
  U1 ~ N(1500, 400^2)    (exogenous noise for sqft)
  U2 ~ N(0, 0.3^2)       (exogenous noise for bathrooms)
  U3 ~ N(0, 10000^2)     (exogenous noise for price)

  X1 := U1                (sqft is purely exogenous)
  X2 := 0.002*X1 + U2    (bathrooms depends on sqft)
  Y  := 150*X1 + U3      (price depends only on sqft)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class StructuralCausalModel:
    """A minimal SCM implementation for educational purposes.

    This class encodes the causal graph structure and structural equations.
    It supports sampling from the observational distribution and computing
    interventional distributions via the do-operator.
    """
    variables: List[str]
    # Each equation is: variable = f(parents, noise)
    # Stored as: variable -> (parent_names, function(parent_values, noise))
    equations: Dict[str, Tuple[List[str], Callable]]
    # Noise distributions: variable -> sampling_function
    noise_distributions: Dict[str, Callable]

    def sample(self, n: int = 1) -> pd.DataFrame:
        """Sample n observations from the observational distribution.

        Evaluates structural equations in topological order, respecting
        causal dependencies. This produces data as if we simply observed
        the natural data-generating process.
        """
        data = {}
        for var in self.variables:
            parents, func = self.equations[var]
            noise = self.noise_distributions[var](n)
            parent_values = {p: data[p] for p in parents}
            data[var] = func(parent_values, noise)
        return pd.DataFrame(data)

    def intervene(self, n: int, interventions: Dict[str, float]) -> pd.DataFrame:
        """Sample from the interventional distribution P(V | do(X=x)).

        When we intervene on a variable, we REPLACE its structural equation
        with a constant value. This breaks all incoming causal arrows to that
        variable while preserving all other causal relationships. This is the
        key difference from conditioning: downstream effects propagate, but
        upstream causes are disconnected.
        """
        data = {}
        for var in self.variables:
            if var in interventions:
                # Replace structural equation with constant
                # This "cuts" all incoming edges in the DAG
                data[var] = np.full(n, interventions[var])
            else:
                parents, func = self.equations[var]
                noise = self.noise_distributions[var](n)
                parent_values = {p: data[p] for p in parents}
                data[var] = func(parent_values, noise)
        return pd.DataFrame(data)


# --- Build the house price SCM ---
house_scm = StructuralCausalModel(
    variables=["sqft", "bathrooms", "price"],
    equations={
        # sqft has no parents -- it is a root node
        "sqft": ([], lambda parents, noise: noise),
        # bathrooms is caused by sqft (larger houses have more bathrooms)
        "bathrooms": (
            ["sqft"],
            lambda parents, noise: 0.002 * parents["sqft"] + noise
        ),
        # price is caused ONLY by sqft, not bathrooms
        "price": (
            ["sqft"],
            lambda parents, noise: 150 * parents["sqft"] + noise
        ),
    },
    noise_distributions={
        "sqft": lambda n: np.random.normal(1500, 400, n),
        "bathrooms": lambda n: np.random.normal(0, 0.3, n),
        "price": lambda n: np.random.normal(0, 10000, n),
    },
)

# Observational data (natural distribution)
obs_data = house_scm.sample(5000)
print("Observational data statistics:")
print(obs_data.describe().round(1))

# Interventional data: set bathrooms = 5 for everyone
int_data = house_scm.intervene(5000, {"bathrooms": 5.0})
print("\nInterventional data (do(bathrooms=5)):")
print(int_data.describe().round(1))

# Key insight: in the interventional distribution, price does NOT change
# because bathrooms has no causal effect on price. The mean price is
# the same regardless of what value we set bathrooms to.
# In the observational distribution, conditioning on bathrooms=5 would
# imply large sqft (due to correlation), giving a misleadingly high price.
```

### 2.2 방향 비순환 그래프(Directed Acyclic Graphs, DAGs)

DAG는 SCM의 인과 구조를 그래프로 표현한 것이다. 노드는 변수를 나타내고, 방향 간선은 직접적 인과 효과를 나타낸다.

```python
"""
Building and visualizing causal DAGs.

We use the networkx library for graph operations and matplotlib for
visualization. In practice, the DAG encodes domain knowledge about
which variables cause which other variables.
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def build_dag(edges: List[Tuple[str, str]]) -> nx.DiGraph:
    """Build a DAG from a list of (cause, effect) edges.

    Each edge A -> B means "A is a direct cause of B" in the
    structural causal model. The graph must be acyclic: there
    cannot be circular causal dependencies.
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Verify acyclicity -- a fundamental requirement for SCMs
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph contains cycles -- not a valid DAG!")

    return G


def find_confounders(dag: nx.DiGraph, x: str, y: str) -> List[str]:
    """Find common causes (confounders) of X and Y.

    A confounder Z satisfies: Z -> ... -> X and Z -> ... -> Y
    (Z is an ancestor of both X and Y). Confounders create spurious
    associations between X and Y that are not causal.
    """
    ancestors_x = nx.ancestors(dag, x)
    ancestors_y = nx.ancestors(dag, y)
    # Common ancestors (excluding X and Y themselves)
    common = ancestors_x & ancestors_y
    return list(common)


def find_adjustment_set(dag: nx.DiGraph, treatment: str, outcome: str) -> set:
    """Find a valid backdoor adjustment set using Pearl's backdoor criterion.

    To estimate the causal effect of treatment on outcome, we need to
    block all "backdoor paths" (non-causal paths through common ancestors)
    while keeping all causal paths open.

    A set Z satisfies the backdoor criterion if:
    1. No node in Z is a descendant of treatment
    2. Z blocks every path between treatment and outcome that has an
       arrow INTO treatment (backdoor paths)

    This is a simplified implementation using parent adjustment.
    """
    # Simple heuristic: adjust for parents of treatment
    # (This is valid when parents are not descendants of treatment)
    parents = set(dag.predecessors(treatment))
    descendants = nx.descendants(dag, treatment)

    # Remove any parents that are descendants of treatment (would be invalid)
    valid_adjustment = parents - descendants - {outcome}
    return valid_adjustment


# --- Example: Medical diagnosis DAG ---
# Smoking -> Tar deposits -> Cancer
# Smoking -> Anxiety
# Genetics -> Cancer
# Genetics -> Smoking (genetic predisposition)
medical_dag = build_dag([
    ("Genetics", "Smoking"),
    ("Genetics", "Cancer"),
    ("Smoking", "Tar"),
    ("Tar", "Cancer"),
    ("Smoking", "Anxiety"),
])

print("DAG edges:", list(medical_dag.edges()))
print("Confounders of Smoking and Cancer:", find_confounders(medical_dag, "Smoking", "Cancer"))
print("Adjustment set for Smoking -> Cancer:", find_adjustment_set(medical_dag, "Smoking", "Cancer"))

# Genetics is a confounder: it causes both Smoking and Cancer.
# To estimate the causal effect of Smoking on Cancer, we must adjust
# for Genetics (block the backdoor path Smoking <- Genetics -> Cancer).
```

### 2.3 핵심 DAG 구조

세 가지 기본 구조가 DAG에서 정보의 흐름을 결정한다:

```python
"""
The three elemental DAG structures and their independence properties.

Understanding these is essential for causal reasoning because they
determine which variables are conditionally independent.
"""

# 1. CHAIN: A -> B -> C
# Unconditionally: A and C are dependent (information flows through B)
# Conditioning on B: A and C become INDEPENDENT (B "blocks" the path)
# Example: Smoking -> Tar -> Cancer
#   Knowing tar deposits blocks the information flow from smoking to cancer

# 2. FORK (common cause): A <- B -> C
# Unconditionally: A and C are dependent (confounded by B)
# Conditioning on B: A and C become INDEPENDENT (confounding removed)
# Example: Genetics -> Smoking, Genetics -> Cancer
#   Genetics confounds the smoking-cancer relationship

# 3. COLLIDER: A -> B <- C
# Unconditionally: A and C are INDEPENDENT
# Conditioning on B: A and C become DEPENDENT ("explaining away")
# Example: Talent -> Hired <- Luck
#   Among hired people, talent and luck become negatively correlated
#   (if someone was hired and is not talented, they must have been lucky)

def demonstrate_collider_bias():
    """Show how conditioning on a collider creates spurious associations.

    This is called "Berkson's paradox" or "explaining away." It is a
    common source of spurious correlations in selected samples.
    """
    n = 10000
    # Talent and Luck are independent causes
    talent = np.random.normal(50, 15, n)
    luck = np.random.normal(50, 15, n)

    # Hiring depends on both (collider)
    score = talent + luck
    hired = score > 110  # Top performers get hired

    # In the full population, talent and luck are independent
    full_corr = np.corrcoef(talent, luck)[0, 1]
    print(f"Correlation (full population): {full_corr:.4f}")  # ~0

    # Among hired people, talent and luck become negatively correlated
    hired_corr = np.corrcoef(talent[hired], luck[hired])[0, 1]
    print(f"Correlation (hired only):      {hired_corr:.4f}")  # negative

    # This is collider bias: conditioning on the collider (hired)
    # opens a spurious path between its parents (talent and luck)

demonstrate_collider_bias()
```

---

## 3. Pearl의 Do-계산법(Do-Calculus)

### 3.1 do-연산자(do-Operator)

do-연산자는 개입을 표현하기 위한 Pearl의 수학적 장치이다. P(Y | do(X = x))는 단순히 X = x를 *관측*하는 것이 아니라, 외부 개입으로 X를 x로 *설정*했을 때의 Y의 분포를 나타낸다.

그래프적으로, do(X = x)는:
1. DAG에서 X로 들어오는 모든 화살표를 제거
2. X = x를 상수로 설정
3. 나머지 화살표를 통해 효과를 전방으로 전파

```python
"""
The do-operator: the fundamental distinction between seeing and doing.

P(Y | X = x): "What is Y among cases where X happens to be x?"
  -> Uses observed correlations, including those from confounders

P(Y | do(X = x)): "What would Y be if we forced X to be x?"
  -> Removes confounding by breaking incoming arrows to X
"""

def observational_vs_interventional():
    """Demonstrate the quantitative difference between P(Y|X) and P(Y|do(X)).

    We use a confounded scenario where a naive observational analysis
    gives the WRONG answer about the causal effect.
    """
    np.random.seed(42)
    n = 50000

    # --- True causal model ---
    # Confounder Z affects both treatment X and outcome Y
    # Z: socioeconomic status (higher -> more likely to take drug AND healthier)
    Z = np.random.binomial(1, 0.5, n)  # 0: low SES, 1: high SES

    # Treatment X: probability of taking drug depends on SES
    # High SES people are more likely to take the drug (confounding!)
    p_treatment = 0.3 + 0.5 * Z  # P(X=1|Z=0)=0.3, P(X=1|Z=1)=0.8
    X = np.random.binomial(1, p_treatment, n)

    # Outcome Y: recovery probability
    # The drug actually HURTS recovery (negative causal effect)
    # But high SES helps recovery (confounding effect)
    p_recovery = 0.4 - 0.1 * X + 0.4 * Z
    p_recovery = np.clip(p_recovery, 0, 1)
    Y = np.random.binomial(1, p_recovery, n)

    # --- Observational analysis (WRONG) ---
    # P(Y=1 | X=1) vs P(Y=1 | X=0)
    treated_recovery = Y[X == 1].mean()
    untreated_recovery = Y[X == 0].mean()
    obs_effect = treated_recovery - untreated_recovery

    print("=== Observational Analysis (confounded) ===")
    print(f"P(Y=1 | X=1) = {treated_recovery:.3f}")
    print(f"P(Y=1 | X=0) = {untreated_recovery:.3f}")
    print(f"Naive effect:  {obs_effect:+.3f}")
    print("Conclusion: Drug appears HELPFUL (but this is WRONG)")

    # --- Interventional analysis (correct, using adjustment) ---
    # P(Y=1 | do(X=1)) = sum_z P(Y=1|X=1,Z=z) * P(Z=z)
    # This is the backdoor adjustment formula
    effects = []
    for z_val in [0, 1]:
        mask_treated = (X == 1) & (Z == z_val)
        mask_untreated = (X == 0) & (Z == z_val)
        p_z = (Z == z_val).mean()

        # Stratum-specific effects
        treated_in_stratum = Y[mask_treated].mean()
        untreated_in_stratum = Y[mask_untreated].mean()
        effects.append((z_val, treated_in_stratum, untreated_in_stratum, p_z))

    p_y_do_x1 = sum(t * p for _, t, _, p in effects)
    p_y_do_x0 = sum(u * p for _, _, u, p in effects)
    causal_effect = p_y_do_x1 - p_y_do_x0

    print("\n=== Causal Analysis (backdoor adjustment) ===")
    for z_val, t, u, p in effects:
        print(f"Z={z_val}: P(Y=1|X=1,Z={z_val})={t:.3f}, "
              f"P(Y=1|X=0,Z={z_val})={u:.3f}, P(Z={z_val})={p:.3f}")
    print(f"P(Y=1 | do(X=1)) = {p_y_do_x1:.3f}")
    print(f"P(Y=1 | do(X=0)) = {p_y_do_x0:.3f}")
    print(f"Causal effect:     {causal_effect:+.3f}")
    print("Conclusion: Drug is actually HARMFUL (correct!)")

observational_vs_interventional()
```

### 3.2 백도어 보정 공식(The Backdoor Adjustment Formula)

유효한 보정 집합 Z(백도어 기준을 충족하는)가 있을 때, X가 Y에 미치는 인과 효과는:

P(Y | do(X = x)) = sum_z P(Y | X = x, Z = z) * P(Z = z)

이 공식은 인과 추론의 핵심 도구이다. 이것은 다음을 말한다: 교란 변수 Z로 층화하고, 각 층에서 조건부 확률을 계산한 다음, Z의 주변 분포에 대해 평균을 낸다.

```python
"""
The backdoor adjustment formula in action.

This is the mathematical bridge between observational data (which we have)
and interventional quantities (which we want). It works when we can
identify and measure all confounders.
"""

def backdoor_adjustment(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    treatment_value: float = 1.0,
) -> float:
    """Compute P(Y | do(X = treatment_value)) using backdoor adjustment.

    This function implements the adjustment formula for discrete confounders.
    For continuous confounders, you would use regression adjustment or
    inverse probability weighting instead.

    Parameters
    ----------
    df : DataFrame with treatment, outcome, and confounder columns
    treatment : name of the treatment variable
    outcome : name of the outcome variable
    confounders : list of confounder variable names
    treatment_value : the value to intervene on

    Returns
    -------
    The estimated interventional probability/expectation
    """
    # Group by unique confounder combinations
    # For each stratum, compute E[Y | X = treatment_value, Z = z]
    # Then weight by P(Z = z)
    total_n = len(df)
    weighted_sum = 0.0

    if not confounders:
        # No confounders: observational = interventional
        treated = df[df[treatment] == treatment_value]
        return treated[outcome].mean()

    # Create confounder strata
    grouped = df.groupby(confounders)

    for stratum_values, stratum_df in grouped:
        # P(Z = z): proportion of data in this stratum
        p_z = len(stratum_df) / total_n

        # E[Y | X = treatment_value, Z = z]
        treated_in_stratum = stratum_df[
            stratum_df[treatment] == treatment_value
        ]

        if len(treated_in_stratum) > 0:
            e_y_given_x_z = treated_in_stratum[outcome].mean()
            weighted_sum += e_y_given_x_z * p_z

    return weighted_sum


# --- Example: Simpson's Paradox ---
# A drug appears effective overall but is harmful in every subgroup
np.random.seed(42)
n = 10000

gender = np.random.binomial(1, 0.5, n)  # 0: F, 1: M
# Men are more likely to take the drug AND have higher baseline recovery
treatment = np.random.binomial(1, 0.3 + 0.4 * gender, n)
p_recover = 0.6 - 0.15 * treatment + 0.3 * gender
recovery = np.random.binomial(1, np.clip(p_recover, 0, 1), n)

simpson_df = pd.DataFrame({
    "gender": gender,
    "treatment": treatment,
    "recovery": recovery,
})

# Naive (confounded) estimate
naive = simpson_df.groupby("treatment")["recovery"].mean()
print("Naive analysis:")
print(f"  P(recovery | treatment=1) = {naive[1]:.3f}")
print(f"  P(recovery | treatment=0) = {naive[0]:.3f}")
print(f"  Naive effect: {naive[1] - naive[0]:+.3f}")

# Causal (adjusted) estimate
p_do_1 = backdoor_adjustment(simpson_df, "treatment", "recovery", ["gender"], 1)
p_do_0 = backdoor_adjustment(simpson_df, "treatment", "recovery", ["gender"], 0)
print(f"\nCausal analysis (adjusting for gender):")
print(f"  P(recovery | do(treatment=1)) = {p_do_1:.3f}")
print(f"  P(recovery | do(treatment=0)) = {p_do_0:.3f}")
print(f"  Causal effect: {p_do_1 - p_do_0:+.3f}")
```

### 3.3 Do-계산법의 세 가지 규칙(The Three Rules of Do-Calculus)

Pearl의 do-계산법은 개입적 표현식을 관측적 표현식으로 변환할 수 있는(가능한 경우) 세 가지 규칙으로 구성된다:

```python
"""
Pearl's Three Rules of Do-Calculus (conceptual overview).

These rules are COMPLETE: any identifiable causal effect can be
computed from observational data using these three rules.

Rule 1 (Insertion/deletion of observations):
  P(Y | do(X), Z, W) = P(Y | do(X), W)
  if (Y _|_ Z | X, W) in the DAG with incoming arrows to X removed

Rule 2 (Action/observation exchange):
  P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)
  if (Y _|_ Z | X, W) in the DAG with incoming arrows to Z removed

Rule 3 (Insertion/deletion of actions):
  P(Y | do(X), do(Z), W) = P(Y | do(X), W)
  if (Y _|_ Z | X, W) in the DAG with incoming arrows to Z removed
  AND all Z-ancestors with certain path conditions

In practice, you rarely apply these rules by hand. Instead, algorithms
like the ID algorithm (Tian & Pearl, 2002) automate the process.
"""

# The backdoor adjustment is a CONSEQUENCE of these rules.
# The front-door adjustment is another important consequence:

def front_door_example():
    """The front-door criterion: when you cannot measure the confounder.

    DAG: U -> X, U -> Y, X -> M -> Y
    U is an unmeasured confounder. We cannot use backdoor adjustment.
    But if we can measure mediator M, the front-door formula applies:

    P(Y | do(X=x)) = sum_m P(M=m|X=x) * sum_x' P(Y|X=x',M=m) * P(X=x')
    """
    np.random.seed(42)
    n = 20000

    # Unmeasured confounder
    U = np.random.normal(0, 1, n)

    # X is influenced by U
    X = np.random.normal(U, 1, n)

    # Mediator M is caused only by X (no confounding)
    M = 0.5 * X + np.random.normal(0, 0.5, n)

    # Y is caused by M and U (confounded with X through U)
    Y = 2.0 * M + 1.5 * U + np.random.normal(0, 0.5, n)

    # True causal effect of X on Y through M: 0.5 * 2.0 = 1.0
    # Naive regression of Y on X would give ~1.0 + confounding bias

    from sklearn.linear_model import LinearRegression

    # Naive (biased)
    naive_model = LinearRegression().fit(X.reshape(-1, 1), Y)
    print(f"Naive regression coefficient (X->Y): {naive_model.coef_[0]:.3f}")
    print("(Biased upward due to confounding through U)")

    # Front-door estimation
    # Step 1: P(M|X) - regress M on X
    step1 = LinearRegression().fit(X.reshape(-1, 1), M)
    effect_x_on_m = step1.coef_[0]

    # Step 2: P(Y|M,X) - regress Y on M controlling for X
    step2 = LinearRegression().fit(
        np.column_stack([M, X]), Y
    )
    effect_m_on_y = step2.coef_[0]

    causal_effect = effect_x_on_m * effect_m_on_y
    print(f"\nFront-door estimate: {effect_x_on_m:.3f} * {effect_m_on_y:.3f} = {causal_effect:.3f}")
    print("(Close to the true effect of 1.0)")

front_door_example()
```

---

## 4. 인과적 특성 중요도(Causal Feature Importance)

### 4.1 개입적 조건부 기대값(Interventional Conditional Expectations, Janzing et al.)

Janzing et al. (2020)은 SHAP의 표준 조건부 기대값을 *개입적* 조건부 기대값으로 대체할 것을 제안했다. 핵심 아이디어: 특성 X_j의 기여를 계산할 때, 나머지 특성에 대해 조건부 확률을 사용하는 것(상관관계를 보존하는) 대신, X_j에 *개입*한다(원인과의 상관관계를 끊는).

```python
"""
Causal feature importance using interventional conditional expectations.

Standard SHAP value for feature j:
  phi_j = E_S[ E[f(X) | X_S = x_S, X_j = x_j] - E[f(X) | X_S = x_S] ]
  where the inner expectations use the CONDITIONAL distribution of
  unseen features given seen features.

Causal SHAP value for feature j:
  phi_j^causal = E_S[ E[f(X) | do(X_S = x_S, X_j = x_j)] - E[f(X) | do(X_S = x_S)] ]
  where the inner expectations use the INTERVENTIONAL distribution.

The difference: conditional preserves feature correlations (confounding),
interventional breaks them (deconfounding).
"""

from itertools import combinations


def marginal_shap_value(
    model,
    x_instance: np.ndarray,
    X_background: np.ndarray,
    feature_idx: int,
) -> float:
    """Compute marginal (observational) SHAP value for one feature.

    This uses the marginal distribution for unseen features, which is
    equivalent to assuming features are independent. This is what
    SHAP's KernelExplainer with "interventional" feature perturbation does.

    Note: despite the name "interventional" in SHAP's API, this is only
    truly interventional when features are causally independent.
    """
    n_features = x_instance.shape[0]
    n_background = X_background.shape[0]
    phi = 0.0
    n_perms = 0

    # Iterate over all subsets S not containing feature_idx
    other_features = [i for i in range(n_features) if i != feature_idx]

    for size in range(len(other_features) + 1):
        for S in combinations(other_features, size):
            S_with_j = set(S) | {feature_idx}
            S_set = set(S)

            # Create synthetic instances: features in S take x's values,
            # others take background values (marginal distribution)
            X_with = np.tile(X_background, (1, 1)).copy()
            X_without = np.tile(X_background, (1, 1)).copy()

            for feat in S_with_j:
                X_with[:, feat] = x_instance[feat]
            for feat in S_set:
                X_without[:, feat] = x_instance[feat]

            # Marginal contribution
            v_with = model.predict(X_with).mean()
            v_without = model.predict(X_without).mean()

            # Shapley weighting
            s = len(S_set)
            n = n_features
            import math
            weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            phi += weight * (v_with - v_without)

    return phi


def causal_shap_value(
    model,
    x_instance: np.ndarray,
    scm: StructuralCausalModel,
    feature_idx: int,
    feature_names: List[str],
    n_samples: int = 1000,
) -> float:
    """Compute causal SHAP value using the SCM for interventional distributions.

    Instead of using the marginal/conditional distribution to replace unseen
    features, we use the SCM to sample from the interventional distribution.
    This correctly accounts for causal structure.

    The key difference: when feature j is "seen," we intervene do(X_j = x_j)
    rather than conditioning on X_j = x_j. For "unseen" features, we sample
    from their interventional distribution given the seen interventions.
    """
    n_features = x_instance.shape[0]
    phi = 0.0
    other_features = [i for i in range(n_features) if i != feature_idx]

    for size in range(len(other_features) + 1):
        for S in combinations(other_features, size):
            S_with_j = set(S) | {feature_idx}
            S_set = set(S)

            # Interventional samples: do(X_S = x_S) for features in subset
            interventions_with = {
                feature_names[i]: x_instance[i] for i in S_with_j
            }
            interventions_without = {
                feature_names[i]: x_instance[i] for i in S_set
            }

            # Sample remaining features from interventional distribution
            data_with = scm.intervene(n_samples, interventions_with)
            data_without = scm.intervene(n_samples, interventions_without)

            v_with = model.predict(data_with[feature_names].values).mean()
            v_without = model.predict(data_without[feature_names].values).mean()

            s = len(S_set)
            n = n_features
            import math
            weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            phi += weight * (v_with - v_without)

    return phi


# --- Compare marginal vs causal SHAP on the house price model ---
feature_names = ["sqft", "bathrooms"]
X_bg = df[feature_names].values[:100]  # Background samples
x_test = df[feature_names].values[0]   # Instance to explain

print("Instance:", dict(zip(feature_names, x_test)))
print(f"Prediction: ${model.predict(x_test.reshape(1, -1))[0]:,.0f}")

# Marginal SHAP (may over-attribute to bathrooms)
marg_sqft = marginal_shap_value(model, x_test, X_bg, 0)
marg_bath = marginal_shap_value(model, x_test, X_bg, 1)
print(f"\nMarginal SHAP:  sqft={marg_sqft:+.0f}, bathrooms={marg_bath:+.0f}")

# Causal SHAP (correctly attributes zero to bathrooms)
causal_sqft = causal_shap_value(model, x_test, house_scm, 0, feature_names)
causal_bath = causal_shap_value(model, x_test, house_scm, 1, feature_names)
print(f"Causal SHAP:    sqft={causal_sqft:+.0f}, bathrooms={causal_bath:+.0f}")
```

### 4.2 CausalSHAP vs TreeSHAP: 불일치 예시

```python
"""
Full example: CausalSHAP vs TreeSHAP on a dataset where they DISAGREE.

Scenario: Loan approval model
- Income (X1) causes Credit Score (X2): higher income -> better credit history
- Income directly affects loan approval
- Credit Score directly affects loan approval
- BUT the model has also learned a spurious pattern through a confounded
  feature: Zip Code (X3), which correlates with both income and approval
  (redlining effect in training data)

Standard TreeSHAP attributes importance to Zip Code.
Causal SHAP correctly identifies Zip Code's contribution as spurious.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 5000

# --- True causal model ---
# Income -> Credit Score -> Approval
# Income -> Approval (direct effect)
# Zip Code is a confounder: correlated with income (neighborhood effects)
# but has NO legitimate causal effect on creditworthiness

# Hidden confounder: neighborhood wealth
neighborhood_wealth = np.random.normal(50, 15, n)

income = neighborhood_wealth * 1000 + np.random.normal(50000, 10000, n)
credit_score = 0.005 * income + np.random.normal(300, 50, n)
credit_score = np.clip(credit_score, 300, 850)
zip_code = (neighborhood_wealth > 50).astype(int)  # Proxy for wealth

# True approval: depends on income and credit score, NOT zip code
logit = -10 + 0.00005 * income + 0.01 * credit_score
prob = 1 / (1 + np.exp(-logit))
approval = np.random.binomial(1, prob, n)

loan_df = pd.DataFrame({
    "income": income,
    "credit_score": credit_score,
    "zip_code": zip_code,
    "approval": approval,
})

# --- Train model (it may pick up zip_code correlation) ---
features = ["income", "credit_score", "zip_code"]
X_train, X_test, y_train, y_test = train_test_split(
    loan_df[features], loan_df["approval"], test_size=0.2, random_state=42
)

clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
clf.fit(X_train, y_train)
print(f"Model accuracy: {clf.score(X_test, y_test):.3f}")

# --- Standard TreeSHAP ---
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

print("\nTreeSHAP mean |SHAP| per feature:")
for i, feat in enumerate(features):
    print(f"  {feat:15s}: {np.abs(shap_values[:, i]).mean():.4f}")

# TreeSHAP may assign non-trivial importance to zip_code because
# it correlates with income and credit score in the observed data.

# --- Causal analysis: zip_code should have ZERO causal importance ---
# If we had the SCM, Causal SHAP would show this.
# We can approximate by comparing predictions with/without zip_code:

# Interventional test: randomly permute zip_code (breaks its correlation)
X_test_permuted = X_test.copy()
X_test_permuted["zip_code"] = np.random.permutation(X_test["zip_code"].values)

pred_original = clf.predict_proba(X_test)[:, 1]
pred_permuted = clf.predict_proba(X_test_permuted)[:, 1]

print(f"\nPrediction change when intervening on zip_code:")
print(f"  Mean absolute change: {np.abs(pred_original - pred_permuted).mean():.4f}")
print(f"  Max absolute change:  {np.abs(pred_original - pred_permuted).max():.4f}")
print("  (Any non-zero change indicates the model uses the spurious correlation)")
```

---

## 5. 모델 디버깅을 위한 인과적 설명(Causal Explanations for Model Debugging)

### 5.1 가짜 상관관계 탐지(Detecting Spurious Correlations)

인과 추론의 XAI에서 가장 가치 있는 응용 중 하나는 모델이 진정한 인과적 특성이 아닌 가짜 상관관계에 의존하고 있는지를 식별하는 것이다.

```python
"""
Using causal reasoning to debug a model that learned a spurious correlation.

The workflow:
1. Specify the causal DAG (domain knowledge)
2. Identify which features are causally relevant vs spurious
3. Test whether the model uses spurious features via interventional tests
4. Quantify the "spuriousness" of the model's reliance
"""

def spurious_correlation_test(
    model,
    X_test: pd.DataFrame,
    feature_name: str,
    n_permutations: int = 100,
) -> dict:
    """Test whether a model relies on a spurious feature.

    The idea: if feature X has no causal effect on the outcome, then
    randomly permuting X (an intervention that breaks all correlations)
    should not systematically change predictions for any subgroup.

    If predictions DO change, the model is using the spurious correlation.

    Returns statistics about how much the model relies on this feature.
    """
    original_preds = model.predict_proba(X_test)[:, 1]
    permuted_changes = []

    for _ in range(n_permutations):
        X_permuted = X_test.copy()
        X_permuted[feature_name] = np.random.permutation(
            X_test[feature_name].values
        )
        permuted_preds = model.predict_proba(X_permuted)[:, 1]
        changes = permuted_preds - original_preds
        permuted_changes.append(changes)

    all_changes = np.array(permuted_changes)

    return {
        "feature": feature_name,
        "mean_absolute_change": np.abs(all_changes).mean(),
        "max_absolute_change": np.abs(all_changes).max(),
        "std_change": all_changes.std(),
        "pct_predictions_affected": (np.abs(all_changes) > 0.01).mean() * 100,
        "verdict": "SPURIOUS RELIANCE DETECTED"
                   if np.abs(all_changes).mean() > 0.005
                   else "No significant spurious reliance",
    }


# Test each feature in the loan model
print("=== Spurious Correlation Audit ===\n")
for feature in features:
    result = spurious_correlation_test(clf, X_test, feature)
    print(f"Feature: {result['feature']}")
    print(f"  Mean |change|: {result['mean_absolute_change']:.5f}")
    print(f"  % affected:    {result['pct_predictions_affected']:.1f}%")
    print(f"  Verdict:       {result['verdict']}")
    print()
```

### 5.2 ML을 위한 매개 분석: 직접 효과 vs 간접 효과(Mediation Analysis for ML: Direct vs Indirect Effects)

매개 분석(mediation analysis)은 특성의 총 효과를 직접 효과와 간접 효과로 분해한다. 이는 특성이 모델의 예측에 *어떻게* 영향을 미치는지 이해하는 데 중요하다.

```python
"""
Mediation analysis: decomposing feature effects into direct and indirect paths.

Total effect = Direct effect + Indirect effect

Direct effect: X -> Y (not through any mediator)
Indirect effect: X -> M -> Y (through the mediator M)

Example: Does income affect loan approval directly, or only through credit score?
  Income -> Approval (direct)
  Income -> Credit Score -> Approval (indirect, through credit score)
"""


def natural_direct_effect(
    model,
    scm: StructuralCausalModel,
    treatment: str,
    mediator: str,
    outcome: str,
    feature_names: List[str],
    x_treat: float,
    x_control: float,
    n_samples: int = 5000,
) -> float:
    """Compute the Natural Direct Effect (NDE).

    NDE = E[Y(x_treat, M(x_control))] - E[Y(x_control, M(x_control))]

    The NDE measures: "What would happen if we changed treatment from
    x_control to x_treat, but kept the mediator at whatever value it
    would naturally take under x_control?"

    This isolates the DIRECT effect of treatment on outcome, bypassing
    the mediator pathway.
    """
    # Step 1: Sample M under control condition
    control_data = scm.intervene(n_samples, {treatment: x_control})
    mediator_values = control_data[mediator].values

    # Step 2: Predict under treatment with control-condition mediator
    treat_data = scm.intervene(n_samples, {treatment: x_treat})
    treat_data[mediator] = mediator_values  # Override mediator
    y_treat = model.predict(treat_data[feature_names].values).mean()

    # Step 3: Predict under control with control-condition mediator
    y_control = model.predict(control_data[feature_names].values).mean()

    return y_treat - y_control


def natural_indirect_effect(
    model,
    scm: StructuralCausalModel,
    treatment: str,
    mediator: str,
    outcome: str,
    feature_names: List[str],
    x_treat: float,
    x_control: float,
    n_samples: int = 5000,
) -> float:
    """Compute the Natural Indirect Effect (NIE).

    NIE = E[Y(x_control, M(x_treat))] - E[Y(x_control, M(x_control))]

    The NIE measures: "What would happen if we kept treatment at x_control,
    but changed the mediator to what it would be under x_treat?"

    This isolates the INDIRECT effect through the mediator.
    """
    # M under treatment
    treat_data = scm.intervene(n_samples, {treatment: x_treat})
    mediator_treat = treat_data[mediator].values

    # M under control
    control_data = scm.intervene(n_samples, {treatment: x_control})
    mediator_control = control_data[mediator].values

    # Y under control + treatment-mediator
    hybrid_data = control_data.copy()
    hybrid_data[mediator] = mediator_treat
    y_hybrid = model.predict(hybrid_data[feature_names].values).mean()

    # Y under control + control-mediator
    y_control = model.predict(control_data[feature_names].values).mean()

    return y_hybrid - y_control


# Note: Total Effect = NDE + NIE (approximately, for nonlinear models)
# This decomposition reveals whether the model uses features through
# legitimate causal pathways or shortcut correlations.
```

---

## 6. DoWhy 라이브러리

### 6.1 개요

DoWhy는 원칙에 기반한 4단계 워크플로우를 제공하는 인과 추론 Python 라이브러리이다:
1. **모델(Model)**: 인과 DAG 정의
2. **식별(Identify)**: 추정량(adjustment formula) 찾기
3. **추정(Estimate)**: 인과 효과 계산
4. **반박(Refute)**: 추정의 견고성 테스트

```python
"""
DoWhy tutorial: complete causal inference workflow.

DoWhy enforces good practices by requiring explicit causal assumptions
(the DAG) and providing robustness checks (refutation tests).
"""

# pip install dowhy

import dowhy
from dowhy import CausalModel


def dowhy_causal_analysis():
    """Complete DoWhy workflow for estimating a causal effect."""

    # --- Step 0: Generate data with known causal structure ---
    np.random.seed(42)
    n = 5000

    age = np.random.normal(40, 10, n)
    education = 0.3 * age + np.random.normal(12, 2, n)  # Age -> Education
    experience = 0.8 * age - 0.2 * education + np.random.normal(10, 3, n)
    # True causal effect of education on salary: $3000 per year
    salary = 3000 * education + 1500 * experience + 500 * age + np.random.normal(0, 5000, n)

    data = pd.DataFrame({
        "age": age,
        "education": education,
        "experience": experience,
        "salary": salary,
    })

    # --- Step 1: MODEL - Define the causal graph ---
    # This encodes our domain knowledge about what causes what
    causal_graph = """
    digraph {
        age -> education;
        age -> experience;
        age -> salary;
        education -> experience;
        education -> salary;
        experience -> salary;
    }
    """

    model = CausalModel(
        data=data,
        treatment="education",
        outcome="salary",
        graph=causal_graph,
    )

    print("=== Step 1: Causal Model ===")
    print("Treatment: education")
    print("Outcome: salary")
    print("Graph defined with 4 variables and 6 edges")

    # --- Step 2: IDENTIFY - Find the adjustment formula ---
    # DoWhy automatically finds the backdoor adjustment set
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"\n=== Step 2: Identification ===")
    print(identified_estimand)

    # --- Step 3: ESTIMATE - Compute the causal effect ---
    # Multiple estimation methods are available
    estimate_lr = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )
    print(f"\n=== Step 3: Estimation (Linear Regression) ===")
    print(f"Estimated causal effect: ${estimate_lr.value:.0f} per year of education")
    print(f"(True effect: $3000)")

    # Also try propensity score stratification
    estimate_ps = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_stratification",
        target_units="ate",
    )
    print(f"\nEstimated causal effect (Propensity Score): ${estimate_ps.value:.0f}")

    # --- Step 4: REFUTE - Test robustness ---
    print(f"\n=== Step 4: Refutation Tests ===")

    # Test 1: Random common cause
    # Add a random (non-causal) variable. If the estimate changes
    # significantly, our original estimate may be unreliable.
    refute_random = model.refute_estimate(
        identified_estimand,
        estimate_lr,
        method_name="random_common_cause",
    )
    print(f"\nRandom Common Cause Test:")
    print(f"  Original estimate: ${estimate_lr.value:.0f}")
    print(f"  New estimate:      ${refute_random.new_effect:.0f}")
    print(f"  p-value:           {refute_random.refutation_result['p_value']:.4f}")

    # Test 2: Placebo treatment
    # Replace treatment with random noise. The effect should become ~0.
    refute_placebo = model.refute_estimate(
        identified_estimand,
        estimate_lr,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
    )
    print(f"\nPlacebo Treatment Test:")
    print(f"  Original estimate: ${estimate_lr.value:.0f}")
    print(f"  Placebo estimate:  ${refute_placebo.new_effect:.0f}")
    print(f"  (Should be close to $0)")

    # Test 3: Data subset
    # Estimate on a random subset. Should be stable.
    refute_subset = model.refute_estimate(
        identified_estimand,
        estimate_lr,
        method_name="data_subset_refuter",
        subset_fraction=0.8,
    )
    print(f"\nData Subset Test (80%):")
    print(f"  Original estimate: ${estimate_lr.value:.0f}")
    print(f"  Subset estimate:   ${refute_subset.new_effect:.0f}")

    return model, estimate_lr

# model, estimate = dowhy_causal_analysis()
```

### 6.2 ML 모델 디버깅을 위한 DoWhy

```python
"""
Using DoWhy to audit an ML model's feature usage.

Workflow:
1. Train the ML model
2. Define the causal DAG for the features
3. Use DoWhy to estimate the causal effect of each feature on the
   model's PREDICTION (not the true outcome)
4. Compare causal effects with SHAP values to identify discrepancies
"""

def audit_model_with_dowhy(
    model,
    data: pd.DataFrame,
    features: List[str],
    prediction_col: str,
    causal_graph: str,
) -> pd.DataFrame:
    """Audit an ML model by estimating causal effects on its predictions.

    By treating the model's prediction as the "outcome" and estimating
    causal effects of features on predictions, we can identify:
    - Features the model uses that have no causal relationship to the outcome
    - Features the model ignores that have strong causal effects
    - Indirect effects through mediators

    Parameters
    ----------
    model : trained sklearn model
    data : DataFrame with feature columns
    features : list of feature names to audit
    prediction_col : name of the prediction column
    causal_graph : DOT-format causal graph string

    Returns
    -------
    DataFrame with causal effect estimates per feature
    """
    # Add model predictions to the data
    audit_data = data.copy()
    audit_data[prediction_col] = model.predict(data[features])

    results = []

    for feature in features:
        try:
            causal_model = CausalModel(
                data=audit_data,
                treatment=feature,
                outcome=prediction_col,
                graph=causal_graph,
            )

            estimand = causal_model.identify_effect(
                proceed_when_unidentifiable=True
            )

            estimate = causal_model.estimate_effect(
                estimand,
                method_name="backdoor.linear_regression",
            )

            results.append({
                "feature": feature,
                "causal_effect_on_prediction": estimate.value,
                "identified": True,
            })

        except Exception as e:
            results.append({
                "feature": feature,
                "causal_effect_on_prediction": None,
                "identified": False,
                "error": str(e),
            })

    return pd.DataFrame(results)


# Example usage (conceptual):
# audit_results = audit_model_with_dowhy(
#     model=clf,
#     data=loan_df,
#     features=["income", "credit_score", "zip_code"],
#     prediction_col="model_prediction",
#     causal_graph="""
#     digraph {
#         income -> credit_score;
#         income -> model_prediction;
#         credit_score -> model_prediction;
#     }
#     """,
# )
# Note: zip_code is intentionally omitted from the causal graph
# because it has no legitimate causal effect. If the model assigns
# it importance, that's a red flag for spurious correlation usage.
```

---

## 7. 실습: 분류 모델에서 심슨의 역설 탐지(Practical: Detecting Simpson's Paradox in a Classification Model)

### 7.1 설정: 교란된 분류 문제

심슨의 역설(Simpson's Paradox)은 여러 하위 그룹에서 나타나는 추세가 그룹을 결합하면 *역전*되는 경우에 발생한다. 이는 교란의 가장 극적인 예 중 하나이며 ML 공정성과 해석 가능성에 직접적으로 관련된다.

```python
"""
Practical exercise: Detecting and resolving Simpson's Paradox
in a classification model.

Scenario: A hospital evaluates two treatments for kidney stones.
Treatment A appears better overall, but Treatment B is actually
better for BOTH small and large stones. The paradox arises because
Treatment A is preferentially given for small stones (easier cases).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


def generate_kidney_stone_data(n: int = 10000) -> pd.DataFrame:
    """Generate data exhibiting Simpson's Paradox.

    True causal model:
      Stone Size -> Treatment Assignment (confounding)
      Stone Size -> Recovery (direct effect)
      Treatment -> Recovery (direct effect -- Treatment B is better)

    But: overall statistics show Treatment A has higher recovery rate
    because it is assigned to easier cases (small stones).
    """
    np.random.seed(42)

    # Stone size: 0 = small, 1 = large
    stone_size = np.random.binomial(1, 0.5, n)

    # Treatment assignment depends on stone size (confounding!)
    # Small stones get Treatment A more often (it is simpler to administer)
    # Large stones get Treatment B more often (doctors choose it for hard cases)
    p_treatment_b = 0.25 + 0.5 * stone_size  # P(B|small)=0.25, P(B|large)=0.75
    treatment = np.random.binomial(1, p_treatment_b, n)  # 0=A, 1=B

    # Recovery depends on stone size and treatment
    # Treatment B is actually BETTER for both sizes:
    # Small stone: P(recover|A)=0.93, P(recover|B)=0.87 -- wait, this is the
    # classic version where BOTH treatments work less well on large stones
    # Let us make B genuinely better:
    # Small stones + A: 0.85, Small stones + B: 0.90
    # Large stones + A: 0.65, Large stones + B: 0.75
    base_recovery = 0.85 - 0.20 * stone_size  # Small: 0.85, Large: 0.65
    treatment_effect = 0.05 + 0.05 * stone_size  # B is better: +0.05 for small, +0.10 for large
    p_recovery = base_recovery + treatment_effect * treatment
    recovery = np.random.binomial(1, np.clip(p_recovery, 0, 1), n)

    return pd.DataFrame({
        "stone_size": stone_size,
        "treatment": treatment,
        "recovery": recovery,
    })


# --- Generate and explore data ---
data = generate_kidney_stone_data(20000)

print("=== Overall Recovery Rates (Simpson's Paradox) ===")
overall = data.groupby("treatment")["recovery"].mean()
print(f"Treatment A (0): {overall[0]:.3f}")
print(f"Treatment B (1): {overall[1]:.3f}")
print(f"Overall winner:  {'A' if overall[0] > overall[1] else 'B'}")
print("(Treatment A appears better overall!)")

print("\n=== Stratified by Stone Size (reveals the truth) ===")
for size in [0, 1]:
    subset = data[data["stone_size"] == size]
    rates = subset.groupby("treatment")["recovery"].mean()
    size_name = "Small" if size == 0 else "Large"
    print(f"\n{size_name} stones:")
    print(f"  Treatment A: {rates[0]:.3f}")
    print(f"  Treatment B: {rates[1]:.3f}")
    print(f"  Winner: {'A' if rates[0] > rates[1] else 'B'}")

print("\n(Treatment B is better in EVERY subgroup -- Simpson's Paradox!)")
```

### 7.2 교란된 분류기 훈련

```python
"""
Train a classifier on the confounded data and show how standard
interpretability methods are misled.
"""

# --- Train a model that ignores the confounder ---
# This simulates a common mistake: using only treatment to predict recovery
X_confounded = data[["treatment"]].values
y = data["recovery"].values

model_confounded = LogisticRegression(random_state=42)
model_confounded.fit(X_confounded, y)

print("=== Confounded Model (no stone size feature) ===")
print(f"Coefficient for treatment: {model_confounded.coef_[0][0]:.4f}")
print(f"Intercept: {model_confounded.intercept_[0]:.4f}")
if model_confounded.coef_[0][0] < 0:
    print("Model thinks Treatment B is WORSE (wrong!)")

# --- Train the correct model with the confounder ---
X_correct = data[["treatment", "stone_size"]].values

model_correct = LogisticRegression(random_state=42)
model_correct.fit(X_correct, y)

print(f"\n=== Deconfounded Model (includes stone size) ===")
print(f"Coefficient for treatment:  {model_correct.coef_[0][0]:.4f}")
print(f"Coefficient for stone_size: {model_correct.coef_[0][1]:.4f}")
if model_correct.coef_[0][0] > 0:
    print("Model correctly identifies Treatment B as better!")
```

### 7.3 인과 탐지 파이프라인(Causal Detection Pipeline)

```python
"""
A systematic pipeline to detect Simpson's Paradox in classification models.

This pipeline:
1. Checks if overall treatment effects reverse when stratified
2. Quantifies the degree of confounding
3. Recommends causal adjustments
"""


def detect_simpsons_paradox(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    stratification_vars: List[str],
    threshold: float = 0.01,
) -> dict:
    """Detect Simpson's Paradox by comparing overall and stratified effects.

    Simpson's Paradox occurs when the direction of an association REVERSES
    after stratifying by a confounder. This function systematically checks
    for reversals across all stratification variables.

    Parameters
    ----------
    data : DataFrame
    treatment : binary treatment variable (0/1)
    outcome : binary outcome variable (0/1)
    stratification_vars : potential confounders to stratify by
    threshold : minimum effect size to consider

    Returns
    -------
    Dictionary with detection results
    """
    # Overall effect
    overall = data.groupby(treatment)[outcome].mean()
    overall_effect = overall.get(1, 0) - overall.get(0, 0)

    results = {
        "overall_effect": overall_effect,
        "overall_direction": "positive" if overall_effect > 0 else "negative",
        "paradoxes_found": [],
    }

    for var in stratification_vars:
        strata = data[var].unique()
        stratum_effects = []

        for stratum_value in sorted(strata):
            subset = data[data[var] == stratum_value]
            stratum_rates = subset.groupby(treatment)[outcome].mean()
            effect = stratum_rates.get(1, 0) - stratum_rates.get(0, 0)
            stratum_effects.append({
                "stratum": f"{var}={stratum_value}",
                "effect": effect,
                "n": len(subset),
            })

        # Check for reversal: all strata have opposite sign from overall
        all_reversed = all(
            (se["effect"] > threshold and overall_effect < -threshold) or
            (se["effect"] < -threshold and overall_effect > threshold)
            for se in stratum_effects
            if abs(se["effect"]) > threshold
        )

        if all_reversed:
            results["paradoxes_found"].append({
                "stratification_variable": var,
                "overall_effect": overall_effect,
                "stratum_effects": stratum_effects,
                "severity": "HIGH" if abs(overall_effect) > 0.05 else "MODERATE",
            })

    results["paradox_detected"] = len(results["paradoxes_found"]) > 0
    return results


# --- Run the detection pipeline ---
detection = detect_simpsons_paradox(
    data=data,
    treatment="treatment",
    outcome="recovery",
    stratification_vars=["stone_size"],
)

print("=== Simpson's Paradox Detection Results ===")
print(f"Overall treatment effect: {detection['overall_effect']:+.4f}")
print(f"Overall direction: {detection['overall_direction']}")
print(f"Paradox detected: {detection['paradox_detected']}")

if detection["paradox_detected"]:
    for paradox in detection["paradoxes_found"]:
        print(f"\n  Confounding variable: {paradox['stratification_variable']}")
        print(f"  Severity: {paradox['severity']}")
        for se in paradox["stratum_effects"]:
            print(f"    {se['stratum']}: effect = {se['effect']:+.4f} (n={se['n']})")
    print("\n  RECOMMENDATION: Include the confounding variable in the model")
    print("  or use causal adjustment (backdoor formula) for estimation.")
```

### 7.4 완전한 DoWhy 통합

```python
"""
Complete DoWhy pipeline for the Simpson's Paradox example.
"""

def dowhy_simpsons_resolution():
    """Use DoWhy to properly estimate the treatment effect."""

    # Define the causal DAG
    causal_graph = """
    digraph {
        stone_size -> treatment;
        stone_size -> recovery;
        treatment -> recovery;
    }
    """

    causal_model = CausalModel(
        data=data,
        treatment="treatment",
        outcome="recovery",
        graph=causal_graph,
    )

    # Identify the estimand
    estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    print("=== DoWhy Identification ===")
    print(estimand)

    # Estimate using multiple methods
    methods = [
        ("backdoor.linear_regression", "Linear Regression"),
        ("backdoor.propensity_score_matching", "Propensity Score Matching"),
        ("backdoor.propensity_score_stratification", "PS Stratification"),
    ]

    estimates = {}
    for method_name, display_name in methods:
        try:
            est = causal_model.estimate_effect(
                estimand,
                method_name=method_name,
            )
            estimates[display_name] = est.value
            print(f"\n{display_name}: {est.value:+.4f}")
        except Exception as e:
            print(f"\n{display_name}: Error - {e}")

    # Refutation: add random common cause
    print("\n=== Refutation Tests ===")
    if estimates:
        first_method = list(methods)[0][0]
        est = causal_model.estimate_effect(estimand, method_name=first_method)

        refute = causal_model.refute_estimate(
            estimand, est,
            method_name="random_common_cause",
        )
        print(f"Random Common Cause: original={est.value:+.4f}, "
              f"new={refute.new_effect:+.4f}")

    print("\nAll methods agree: Treatment B has a POSITIVE causal effect")
    print("(Simpson's Paradox is resolved by adjusting for stone_size)")

# dowhy_simpsons_resolution()
```

---

## 8. 고급 주제(Advanced Topics)

### 8.1 민감도 분석: DAG가 잘못되었다면?(Sensitivity Analysis: What If the DAG Is Wrong?)

인과 추론의 근본적 한계는 결과가 가정된 DAG에 의존한다는 것이다. 민감도 분석(sensitivity analysis)은 DAG가 잘못 지정되었을 경우 결론이 얼마나 변할지를 정량화한다.

```python
"""
Sensitivity analysis: how robust are causal conclusions to unmeasured confounding?

The E-value (VanderWeele & Ding, 2017) quantifies: how strong would an
unmeasured confounder need to be to explain away the observed causal effect?

A large E-value means the conclusion is robust: even strong unmeasured
confounding could not negate the effect. A small E-value means we should
be cautious.
"""

import math


def compute_e_value(risk_ratio: float) -> float:
    """Compute the E-value for a given risk ratio.

    The E-value is defined as:
    E = RR + sqrt(RR * (RR - 1))

    where RR is the observed risk ratio (must be > 1; if < 1, use 1/RR).

    Interpretation: An unmeasured confounder would need to have a risk ratio
    of at least E-value with BOTH the treatment and the outcome (conditional
    on measured covariates) to fully explain away the observed effect.
    """
    if risk_ratio < 1:
        risk_ratio = 1.0 / risk_ratio

    e_val = risk_ratio + math.sqrt(risk_ratio * (risk_ratio - 1))
    return e_val


# Example: Treatment B has recovery rate 0.82 vs Treatment A's 0.77
# (after adjusting for stone size)
rr = 0.82 / 0.77  # Risk ratio ~1.065
e_val = compute_e_value(rr)

print(f"Risk ratio: {rr:.3f}")
print(f"E-value: {e_val:.3f}")
print(f"\nInterpretation: An unmeasured confounder would need a risk ratio")
print(f"of at least {e_val:.2f} with both treatment and outcome to explain")
print(f"away the observed effect. This is relatively modest, suggesting")
print(f"we should look for additional confounders.")
```

### 8.2 인과 추론과 XAI의 연결(Connecting Causal Inference to XAI)

```python
"""
Summary: How causal inference improves XAI methods.

This table maps standard XAI methods to their causal-aware counterparts.
"""

comparison = pd.DataFrame({
    "Standard XAI Method": [
        "SHAP (KernelSHAP / TreeSHAP)",
        "LIME",
        "Integrated Gradients",
        "Permutation Importance",
        "Partial Dependence Plots",
    ],
    "Limitation": [
        "Uses observational distribution; correlated features split importance",
        "Local linear model ignores causal structure",
        "Baseline choice does not respect causal constraints",
        "Breaking correlations by permutation is only valid under independence",
        "Marginalizes over observed distribution; biased by confounders",
    ],
    "Causal Alternative": [
        "CausalSHAP (Heskes et al., 2020) or Asymmetric SHAP (Frye et al., 2020)",
        "Causal LIME (not yet standardized in literature)",
        "Causal path-specific gradients",
        "Interventional Permutation (permute respecting DAG)",
        "Causal Partial Dependence (intervene instead of marginalize)",
    ],
    "Key Idea": [
        "Use SCM interventions instead of conditional expectations",
        "Perturb features according to causal interventions",
        "Integrate along causally valid paths only",
        "Only permute features that are causally root nodes",
        "Use do(X=x) instead of P(.|X=x) for the PDP curve",
    ],
})

print("=== Standard XAI vs Causal XAI ===")
for _, row in comparison.iterrows():
    print(f"\n{row['Standard XAI Method']}:")
    print(f"  Problem:     {row['Limitation']}")
    print(f"  Solution:    {row['Causal Alternative']}")
    print(f"  Mechanism:   {row['Key Idea']}")
```

---

## 요약

- **관측적 vs 개입적(Observational vs interventional)**: 표준 XAI 방법은 "예측과 무엇이 상관되는가?"에 답하고, 인과적 XAI는 "무엇이 예측을 변화시키는 원인인가?"에 답한다 -- 특성들이 상관되어 있을 때 이 구분이 중요하다.
- **구조적 인과 모델(Structural Causal Models)**: SCM (V, U, F, P(U))은 데이터 생성 과정을 형식적으로 인코딩한다. 관련된 DAG는 인과 경로, 교란 변수, 매개 변수를 드러낸다.
- **세 가지 DAG 구조**: 체인(chains)은 매개 변수에 조건을 걸면 정보를 차단하고, 포크(forks)는 공통 원인에 조건을 걸면 정보를 차단하며, 충돌체(colliders)는 조건을 걸면 정보를 *열어* 가짜 연관을 만든다.
- **Pearl의 do-계산법(do-calculus)**: do-연산자는 DAG에서 들어오는 화살표를 제거하여 외부 개입을 나타낸다. 백도어 보정 공식은 개입적 쿼리를 계산 가능한 관측적 양으로 변환한다.
- **인과적 특성 중요도(Causal feature importance)**: Janzing et al.의 개입적 조건부 기대값과 CausalSHAP은 조건부 분포를 개입적 분포로 대체하여 특성 의존성을 올바르게 처리한다.
- **모델 디버깅**: 인과 추론은 모델이 사용하는 것과 DAG가 합법적이라고 말하는 것을 비교하여 모델이 악용하는 가짜 상관관계를 식별한다.
- **매개 분석(Mediation analysis)**: 총 효과를 직접 및 간접 경로로 분해하여 특성이 예측에 어떻게 영향을 미치는지 드러낸다.
- **DoWhy 라이브러리**: Python에서 인과 추론을 위한 원칙에 기반한 모델-식별-추정-반박 워크플로우를 제공한다.
- **심슨의 역설(Simpson's Paradox)**: 집계 추세가 하위 그룹에서 역전되는 교란의 극적인 예이다; 인과 분석을 통해 탐지하고 해결할 수 있다.

---

## 연습 문제

### 연습 1: 자체 SCM 구축 (초급)

다음 시나리오에 대한 구조적 인과 모델을 만들어라:
- 교육 수준이 직업 유형과 급여 모두에 영향을 미친다
- 직업 유형이 급여에 영향을 미친다
- 교란 변수 "가족 재산"이 교육과 급여 모두에 영향을 미친다

SCM 클래스를 구현하고, 관측 데이터를 생성하고, 단순 회귀가 교육이 급여에 미치는 효과를 과대추정함을 보여라. 그런 다음 백도어 보정을 사용하여 올바른 인과 추정치를 구하라.

### 연습 2: CausalSHAP 불일치 (중급)

섹션 4.2의 대출 승인 데이터셋을 사용하여:
1. 100개 테스트 인스턴스에 대해 표준 TreeSHAP 값을 계산하라
2. 각 특성에 대한 인과적 순열 테스트를 구현하라
3. TreeSHAP과 인과 중요도가 불일치하는 지점을 보여주는 산점도를 만들어라
4. 가짜 zip_code 특성에 가장 큰 영향을 받는 인스턴스를 식별하라

### 연습 3: 심슨의 역설 탐지기 (중급)

`detect_simpsons_paradox` 함수를 확장하여:
1. 연속 교란 변수를 처리하라 (구간화하여)
2. 방향이 얼마나 역전되는지 정량화하는 "역설 심각도 점수"를 계산하라
3. 사용자가 제공한 DAG에서 최소 보정 집합을 추천하라
4. 전체 대 층화된 효과를 보여주는 시각화를 생성하라

### 연습 4: DoWhy 모델 감사 (고급)

선택한 데이터셋(예: Adult Income 또는 German Credit)에서 그래디언트 부스팅 분류기를 훈련시키라. 그런 다음:
1. 도메인 지식을 기반으로 인과 DAG를 정의하라
2. DoWhy를 사용하여 모델 예측에 대한 각 특성의 인과 효과를 추정하라
3. 인과 효과를 SHAP 특성 중요도와 비교하라
4. SHAP과 인과 중요도가 불일치하는 특성을 식별하라
5. 발견 사항과 권장 사항을 요약하는 1페이지 "인과 감사 보고서"를 작성하라

### 연습 5: 민감도 분석 (고급)

연습 4의 DoWhy 분석에 대해:
1. 각 특성의 인과 효과에 대한 E-값을 계산하라
2. "누락 변수 편향" 민감도 분석을 구현하라: 잠재적 미측정 교란 변수 각각에 대해 인과 효과를 얼마나 변화시킬지 추정하라
3. 미측정 교란 강도에 대한 다양한 가정 하에서 인과 효과 추정치를 보여주는 민감도 플롯을 만들어라
4. 어떤 결론이 견고하고 어떤 것이 취약한지 판단하라

---

[이전: 반사실적 설명](./08_Counterfactual_Explanations.md) | [개요](./00_Overview.md) | [다음: 설명 평가](./10_Evaluating_Explanations.md)

**License**: CC BY-NC 4.0
