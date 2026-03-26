# 확률 공리와 법칙

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 집합론적 언어를 사용하여 실험, 표본공간, 사건을 정의하기
2. 콜모고로프 공리 (Kolmogorov axioms) 세 가지를 명시하고 이로부터 기본 확률 성질을 유도하기
3. 조건부 확률 (conditional probability)을 계산하고 곱셈 법칙을 적용하기
4. 분할 문제에 전확률 법칙 (Law of Total Probability) 적용하기
5. 베이즈 정리 (Bayes' theorem)를 사용하여 새로운 증거에 기반해 확률을 갱신하기
6. 두 개 이상의 사건에 대한 통계적 독립성 (independence)을 정의하고 검증하기
7. 시뮬레이션을 사용하여 Python으로 확률 계산 구현하기

---

## 개요

확률론 (probability theory)은 불확실성에 대해 추론하기 위한 엄밀한 수학적 틀을 제공합니다. 직관에 의존하는 대신, 1933년 안드레이 콜모고로프가 제안한 세 가지 공리로부터 확률을 구축합니다. 베이즈 정리부터 중심극한정리까지, 확률론의 모든 결과는 이 공리로부터 도출됩니다.

---

## 목차

1. [실험, 표본공간, 사건](#1-실험-표본공간-사건)
2. [콜모고로프 공리](#2-콜모고로프-공리)
3. [공리로부터 유도되는 성질](#3-공리로부터-유도되는-성질)
4. [조건부 확률](#4-조건부-확률)
5. [전확률 법칙](#5-전확률-법칙)
6. [베이즈 정리](#6-베이즈-정리)
7. [독립성](#7-독립성)
8. [Python 예제](#8-python-예제)
9. [핵심 요약](#9-핵심-요약)

---

## 1. 실험, 표본공간, 사건

### 확률 실험

**확률 실험** (random experiment, 또는 시행)이란 다음 조건을 만족하는 절차입니다:

- 동일한 조건에서 반복할 수 있음
- 가능한 결과의 잘 정의된 집합이 있음
- 실험 전에 결과를 확실하게 예측할 수 없음

**예시**: 주사위 던지기, 동전 던지기, 환자의 혈압 측정.

### 표본공간

**표본공간** (sample space) $\Omega$ (또는 $S$)는 실험의 모든 가능한 결과의 집합입니다.

| 실험 | 표본공간 |
|------|----------|
| 동전 던지기 | $\Omega = \{H, T\}$ |
| 주사위 던지기 | $\Omega = \{1, 2, 3, 4, 5, 6\}$ |
| 동전 두 번 던지기 | $\Omega = \{HH, HT, TH, TT\}$ |
| 전구의 수명 | $\Omega = [0, \infty)$ |

표본공간은 **유한**, **가산 무한** (예: 첫 번째 앞면이 나올 때까지의 동전 던지기 횟수), 또는 **비가산 무한** (예: 연속 측정값)일 수 있습니다.

### 사건

**사건** (event) $A$는 표본공간의 부분집합입니다: $A \subseteq \Omega$.

- **단순 사건** (simple event): 정확히 하나의 결과를 포함, 예: $\{3\}$
- **복합 사건** (compound event): 여러 결과를 포함, 예: "짝수가 나옴" = $\{2, 4, 6\}$
- **확실한 사건** (certain event): $\Omega$ (항상 발생)
- **불가능한 사건** (impossible event): $\emptyset$ (절대 발생하지 않음)

### 사건에 대한 집합 연산

| 연산 | 표기법 | 의미 |
|------|--------|------|
| 합집합 | $A \cup B$ | $A$ 또는 $B$ (또는 둘 다) 발생 |
| 교집합 | $A \cap B$ | $A$와 $B$ 모두 발생 |
| 여사건 | $A^c$ 또는 $\bar{A}$ | $A$가 발생하지 않음 |
| 차집합 | $A \setminus B$ | $A$는 발생하지만 $B$는 발생하지 않음 |
| 상호 배타적 | $A \cap B = \emptyset$ | $A$와 $B$가 동시에 발생할 수 없음 |

**드 모르간 법칙** (De Morgan's Laws):

$$
(A \cup B)^c = A^c \cap B^c, \qquad (A \cap B)^c = A^c \cup B^c
$$

---

## 2. 콜모고로프 공리

**확률 함수** (probability function) $P$는 $\Omega$의 부분집합들로 이루어진 시그마 대수 $\mathcal{F}$의 각 사건에 실수를 부여합니다. 다음을 만족해야 합니다:

### 공리 1: 비음수성

$$P(A) \geq 0 \quad \text{모든 사건 } A \text{에 대해}$$

### 공리 2: 정규화

$$P(\Omega) = 1$$

### 공리 3: 가산 가법성

$A_1, A_2, A_3, \ldots$이 쌍별로 상호 배타적인 사건들 (즉, $i \neq j$일 때 $A_i \cap A_j = \emptyset$)이면:

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

삼중항 $(\Omega, \mathcal{F}, P)$를 **확률 공간** (probability space)이라 합니다.

---

## 3. 공리로부터 유도되는 성질

아래의 모든 성질은 세 가지 공리로부터 논리적으로 도출됩니다.

### 여사건 법칙

$$P(A^c) = 1 - P(A)$$

*증명*: $A$와 $A^c$는 상호 배타적이고 $A \cup A^c = \Omega$. 공리 3에 의해: $P(A) + P(A^c) = P(\Omega) = 1$.

### 공집합의 확률

$$P(\emptyset) = 0$$

### 단조성

$A \subseteq B$이면, $P(A) \leq P(B)$.

### 덧셈 법칙 (일반)

임의의 두 사건 $A$와 $B$에 대해:

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

이는 레슨 01의 포함-배제를 직접 적용한 것입니다.

### 불 부등식 (합집합 상한)

$$P\left(\bigcup_{i=1}^{n} A_i\right) \leq \sum_{i=1}^{n} P(A_i)$$

### 확률의 범위

$$0 \leq P(A) \leq 1 \quad \text{모든 사건 } A \text{에 대해}$$

---

## 4. 조건부 확률

### 정의

$B$가 주어졌을 때 $A$의 **조건부 확률** (conditional probability) ($P(B) > 0$인 경우):

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**해석**: 표본공간을 $B$에 속하는 결과로 제한하고, 그 축소된 공간에서 $A$가 얼마나 가능한지를 묻습니다.

**예제**: 공정한 주사위를 던진다. $A = \{6\}$, $B = \{\text{짝수}\} = \{2, 4, 6\}$로 놓으면:

$$P(A \mid B) = \frac{P(\{6\})}{P(\{2,4,6\})} = \frac{1/6}{3/6} = \frac{1}{3}$$

### 곱셈 법칙

정의를 재배열하면:

$$P(A \cap B) = P(A \mid B) \, P(B) = P(B \mid A) \, P(A)$$

여러 사건에 대한 **연쇄 법칙** (chain rule):

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \, P(A_2 \mid A_1) \, P(A_3 \mid A_1 \cap A_2) \cdots P(A_n \mid A_1 \cap \cdots \cap A_{n-1})$$

**예제**: 표준 52장 카드 덱에서 비복원으로 2장을 뽑을 때, 둘 다 에이스일 확률:

$$P(A_1 \cap A_2) = P(A_1) \cdot P(A_2 \mid A_1) = \frac{4}{52} \cdot \frac{3}{51} = \frac{12}{2652} = \frac{1}{221}$$

---

## 5. 전확률 법칙

### 표본공간의 분할

사건 $B_1, B_2, \ldots, B_n$이 $\Omega$의 **분할** (partition)을 이루려면:

1. 상호 배타적: $i \neq j$일 때 $B_i \cap B_j = \emptyset$
2. 전체를 포괄: $B_1 \cup B_2 \cup \cdots \cup B_n = \Omega$
3. 각각 양의 확률을 가짐: $P(B_i) > 0$

### 법칙

$\{B_1, B_2, \ldots, B_n\}$이 $\Omega$의 분할이면, 임의의 사건 $A$에 대해:

$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) \, P(B_i)$$

**예제**: 공장에 두 대의 기계가 있다. 기계 1은 제품의 60%를 생산하고 (불량률 2%), 기계 2는 40%를 생산한다 (불량률 5%). 무작위로 고른 제품이 불량일 확률은?

$$P(D) = P(D \mid M_1)P(M_1) + P(D \mid M_2)P(M_2) = (0.02)(0.60) + (0.05)(0.40) = 0.032$$

---

## 6. 베이즈 정리

### 진술

분할 $\{B_1, B_2, \ldots, B_n\}$과 $P(A) > 0$인 사건 $A$가 주어졌을 때:

$$P(B_j \mid A) = \frac{P(A \mid B_j) \, P(B_j)}{\sum_{i=1}^{n} P(A \mid B_i) \, P(B_i)} = \frac{P(A \mid B_j) \, P(B_j)}{P(A)}$$

**용어**:

- $P(B_j)$: **사전 확률** (prior probability, 증거를 관찰하기 전)
- $P(A \mid B_j)$: **가능도** (likelihood, 가설이 주어졌을 때 증거의 확률)
- $P(B_j \mid A)$: **사후 확률** (posterior probability, 증거를 관찰한 후)
- $P(A)$: **주변 가능도** (marginal likelihood) 또는 **증거** (evidence)

### 의료 검사 예제

질병 유병률이 1000명 중 1명이다 ($P(D) = 0.001$). 검사의 성능:

- 민감도 (sensitivity, 진양성률): $P(+ \mid D) = 0.99$
- 특이도 (specificity, 진음성률): $P(- \mid D^c) = 0.95$

양성 판정을 받은 경우, $P(D \mid +)$은?

**1단계**: 전확률 법칙으로 $P(+)$ 계산:

$$P(+) = P(+ \mid D)P(D) + P(+ \mid D^c)P(D^c) = (0.99)(0.001) + (0.05)(0.999) = 0.05094$$

**2단계**: 베이즈 정리 적용:

$$P(D \mid +) = \frac{(0.99)(0.001)}{0.05094} = \frac{0.00099}{0.05094} \approx 0.0194$$

외관상 좋은 검사임에도 불구하고, 양성 결과는 실제로 질병이 있을 확률이 1.94%에 불과합니다. 이를 **기저율 오류** (base rate fallacy)라 합니다 -- 질병이 희귀할 때 대부분의 양성은 위양성입니다.

---

## 7. 독립성

### 두 사건에 대한 정의

사건 $A$와 $B$가 **독립** (independent)이란 다음을 의미합니다:

$$P(A \cap B) = P(A) \cdot P(B)$$

동치 조건 ($P(B) > 0$일 때): $P(A \mid B) = P(A)$ -- $B$가 발생했다는 것이 $A$의 확률을 변화시키지 않습니다.

### 상호 독립성

사건 $A_1, A_2, \ldots, A_n$이 **상호 독립** (mutually independent)이란 모든 부분집합 $\{i_1, i_2, \ldots, i_k\} \subseteq \{1, 2, \ldots, n\}$에 대해:

$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \cdot P(A_{i_2}) \cdots P(A_{i_k})$$

**주의**: 쌍별 독립성은 상호 독립성을 보장하지 않습니다.

### 독립 사건의 성질

$A$와 $B$가 독립이면, 다음도 독립입니다:

- $A$와 $B^c$
- $A^c$와 $B$
- $A^c$와 $B^c$

### 독립성 vs. 상호 배타성

- **상호 배타적**: $P(A \cap B) = 0$ -- $A$가 발생하면 $B$는 발생할 수 없음 (비자명 사건에 대한 최대 음의 의존성)
- **독립**: $P(A \cap B) = P(A)P(B)$ -- 하나의 발생이 다른 것에 대해 아무런 정보도 제공하지 않음

$P(A) > 0$이고 $P(B) > 0$이면, 상호 배타적인 사건은 **절대로** 독립이 아닙니다 ($0 \neq P(A)P(B)$이므로).

---

## 8. Python 예제

### 조건부 확률 시뮬레이션

```python
import random

def simulate_conditional(n_trials=1_000_000):
    """Simulate P(die=6 | die is even) by rejection sampling."""
    random.seed(42)
    even_count = 0
    six_and_even = 0

    for _ in range(n_trials):
        roll = random.randint(1, 6)
        if roll % 2 == 0:          # Condition: even
            even_count += 1
            if roll == 6:
                six_and_even += 1

    estimate = six_and_even / even_count
    print(f"P(6 | even) ~ {estimate:.4f}  (exact: 0.3333)")

simulate_conditional()
```

### 베이즈 정리: 의료 검사

```python
def bayes_medical_test(prevalence, sensitivity, specificity):
    """Compute P(Disease | Positive test) using Bayes' theorem."""
    p_d = prevalence
    p_pos_given_d = sensitivity
    p_pos_given_not_d = 1 - specificity

    # Law of total probability
    p_pos = p_pos_given_d * p_d + p_pos_given_not_d * (1 - p_d)

    # Bayes' theorem
    p_d_given_pos = (p_pos_given_d * p_d) / p_pos

    print(f"Prevalence:   {p_d}")
    print(f"Sensitivity:  {sensitivity}")
    print(f"Specificity:  {specificity}")
    print(f"P(+):         {p_pos:.6f}")
    print(f"P(D | +):     {p_d_given_pos:.6f}")
    return p_d_given_pos

bayes_medical_test(prevalence=0.001, sensitivity=0.99, specificity=0.95)
# P(D | +) ~ 0.019417
```

### 시뮬레이션으로 독립성 검증

```python
import random

def test_independence(n_trials=500_000):
    """Test whether two coin flips are independent via simulation."""
    random.seed(0)
    count_a = 0   # First flip is H
    count_b = 0   # Second flip is H
    count_ab = 0  # Both are H

    for _ in range(n_trials):
        flip1 = random.choice(["H", "T"])
        flip2 = random.choice(["H", "T"])
        a = (flip1 == "H")
        b = (flip2 == "H")
        count_a += a
        count_b += b
        count_ab += (a and b)

    p_a = count_a / n_trials
    p_b = count_b / n_trials
    p_ab = count_ab / n_trials

    print(f"P(A)      = {p_a:.4f}")
    print(f"P(B)      = {p_b:.4f}")
    print(f"P(A)*P(B) = {p_a * p_b:.4f}")
    print(f"P(A & B)  = {p_ab:.4f}")
    print(f"Independent? P(AB) ~ P(A)P(B): {abs(p_ab - p_a * p_b) < 0.005}")

test_independence()
```

### 전확률 법칙: 공장 예제

```python
def factory_defect():
    """Compute defect probability using the Law of Total Probability."""
    # Machine 1: 60% of production, 2% defect rate
    # Machine 2: 40% of production, 5% defect rate
    machines = [
        {"name": "M1", "share": 0.60, "defect_rate": 0.02},
        {"name": "M2", "share": 0.40, "defect_rate": 0.05},
    ]

    p_defect = sum(m["share"] * m["defect_rate"] for m in machines)
    print(f"P(Defective) = {p_defect:.4f}")  # 0.0320

    # Reverse: given defective, which machine? (Bayes)
    for m in machines:
        posterior = (m["defect_rate"] * m["share"]) / p_defect
        print(f"P({m['name']} | Defective) = {posterior:.4f}")

factory_defect()
```

---

## 9. 핵심 요약

1. **확률은 세 가지 공리 위에 구축됩니다**: 비음수성, 정규화, 가산 가법성. 확률론의 모든 정리는 이로부터 도출됩니다.

2. **조건부 확률** $P(A \mid B) = P(A \cap B)/P(B)$는 부분적인 정보를 얻은 후 확률 평가를 세밀하게 조정합니다.

3. **베이즈 정리**는 조건부 확률을 역전시킵니다 -- 관찰된 결과로부터 원인의 확률을 계산합니다.

4. **전확률 법칙**은 표본공간의 분할에 대해 합산하여 $P(A)$를 계산할 수 있게 합니다.

5. **독립성** ($P(A \cap B) = P(A)P(B)$)은 "영향 없음"의 수학적 형식화입니다. 상호 배타성과는 근본적으로 다릅니다.

6. **기저율이 중요합니다**: 기저 조건이 희귀할 때는 매우 정확한 검사도 대부분 위양성을 만들어냅니다 (베이즈 정리가 이를 정확하게 정량화합니다).

---

*이전: [01 - 조합론과 셈](./01_Combinatorics_and_Counting.md) | 다음: [03 - 확률변수와 분포](./03_Random_Variables_and_Distributions.md)*
