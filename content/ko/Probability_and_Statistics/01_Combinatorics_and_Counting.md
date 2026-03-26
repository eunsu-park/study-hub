# 조합론과 셈

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 합성 실험의 결과를 세기 위해 덧셈 법칙과 곱셈 법칙 적용하기
2. 반복이 있는 경우와 없는 경우의 순열 (permutation) 계산하기
3. 반복이 있는 경우와 없는 경우의 조합 (combination) 계산하기 (중복조합)
4. 다항 계수 (multinomial coefficient)를 사용하여 다중집합 원소의 배열 수 세기
5. 합집합 문제에 포함-배제 원리 (inclusion-exclusion principle) 적용하기
6. 완전순열 (derangement)을 정의하고 부분계승 공식을 사용하여 $D_n$ 계산하기
7. `math.factorial`과 `itertools`를 사용하여 Python으로 셈 함수 구현하기

---

## 개요

조합론 (combinatorics)은 확률론의 수학적 기초입니다. 사건에 확률을 부여하기 전에, 해당 사건이 발생할 수 있는 경우의 수를 셀 수 있어야 합니다. 이 레슨에서는 이산 확률론 전체를 뒷받침하는 핵심 셈 기법을 다룹니다.

---

## 목차

1. [기본 셈 원리](#1-기본-셈-원리)
2. [순열](#2-순열)
3. [조합](#3-조합)
4. [다항 계수](#4-다항-계수)
5. [포함-배제 원리](#5-포함-배제-원리)
6. [완전순열](#6-완전순열)
7. [Python 예제](#7-python-예제)
8. [핵심 요약](#8-핵심-요약)

---

## 1. 기본 셈 원리

### 덧셈 법칙 (합의 법칙)

사건 $A$가 $m$가지 방법으로, 사건 $B$가 $n$가지 방법으로 일어날 수 있고, $A$와 $B$가 **상호 배타적** (mutually exclusive, 동시에 일어날 수 없음)이면, $A$ 또는 $B$가 일어나는 경우의 수는:

$$m + n \text{ ways}$$

**예제**: 한 학생이 수학 선택 과목(5개) 또는 물리학 선택 과목(3개) 중 하나만 선택할 수 있다. 총 선택 수: $5 + 3 = 8$.

### 곱셈 법칙 (곱의 법칙)

어떤 절차가 두 **순차적 단계**로 구성되어 있고, 1단계가 $m$가지 방법으로, 2단계가 $n$가지 방법으로 수행될 수 있다면 (1단계의 결과와 무관하게), 전체 절차는:

$$m \times n \text{ ways}$$

**예제**: 번호판이 문자 2개와 숫자 3개로 구성된다. 총 번호판 수: $26^2 \times 10^3 = 676{,}000$.

### 일반화된 곱셈 법칙

$k$개의 순차적 단계에서 각각 $n_1, n_2, \ldots, n_k$개의 선택이 있을 때:

$$\text{총 경우의 수} = n_1 \times n_2 \times \cdots \times n_k = \prod_{i=1}^{k} n_i$$

---

## 2. 순열

**순열** (permutation)이란 객체의 순서 있는 배열입니다.

### 반복 없는 순열

$n$개의 서로 다른 객체에서 $r$개를 선택하여 배열하되, **순서가 중요하고** 객체를 재사용하지 않는 경우의 수:

$$P(n, r) = \frac{n!}{(n - r)!}$$

특수한 경우 -- $n$개 객체 전부를 배열: $P(n, n) = n!$

**예제**: 영어 알파벳에서 반복 없이 만들 수 있는 3글자 "단어"의 수는?

$$P(26, 3) = \frac{26!}{23!} = 26 \times 25 \times 24 = 15{,}600$$

### 반복 있는 순열

$n$종류의 객체를 **몇 번이든 사용**할 수 있을 때, $r$-길이 수열의 수:

$$n^r$$

**예제**: 숫자 0--9로 이루어진 4자리 PIN: $10^4 = 10{,}000$.

### 다중집합의 순열

$n$개의 객체 중 $i$번째 유형이 $n_i$번 나타날 때 ($n_1 + n_2 + \cdots + n_k = n$), 서로 다른 순열의 수:

$$\frac{n!}{n_1! \, n_2! \, \cdots \, n_k!}$$

**예제**: "MISSISSIPPI"의 서로 다른 배열 수:

- 총 글자 수: 11
- M: 1, I: 4, S: 4, P: 2

$$\frac{11!}{1! \cdot 4! \cdot 4! \cdot 2!} = \frac{39{,}916{,}800}{1 \cdot 24 \cdot 24 \cdot 2} = 34{,}650$$

---

## 3. 조합

**조합** (combination)이란 객체의 순서 없는 선택입니다.

### 반복 없는 조합

$n$개의 서로 다른 객체에서 $r$개를 선택하되, **순서가 중요하지 않은** 경우의 수:

$$\binom{n}{r} = C(n, r) = \frac{n!}{r!(n - r)!}$$

주요 성질:

- $\binom{n}{0} = \binom{n}{n} = 1$
- $\binom{n}{r} = \binom{n}{n-r}$ (대칭성)
- $\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$ (파스칼의 항등식)

**예제**: 10명 중 4명으로 위원회를 구성하는 경우의 수:

$$\binom{10}{4} = \frac{10!}{4! \cdot 6!} = \frac{10 \times 9 \times 8 \times 7}{4 \times 3 \times 2 \times 1} = 210$$

### 반복 있는 조합 (중복조합, Stars and Bars)

$n$종류에서 **반복을 허용하여** $r$개를 선택하는 경우의 수 (동치: $r$개의 동일한 공을 $n$개의 서로 다른 상자에 넣는 방법 수):

$$\binom{n + r - 1}{r} = \binom{n + r - 1}{n - 1}$$

**Stars and Bars 직관**: $r$개의 항목을 별($\star$)로 표현하고, $n-1$개의 막대($|$)를 사용하여 $n$개 그룹으로 나눕니다. 전체 기호 수는 $r + n - 1$이고, $n-1$개의 막대 위치를 선택합니다.

**예제**: 3가지 맛의 도넛 8개를 구매하는 방법의 수는?

$$\binom{3 + 8 - 1}{8} = \binom{10}{8} = 45$$

### 이항 정리

조합과 대수의 연결:

$$(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}$$

$x = y = 1$로 놓으면: $\sum_{k=0}^{n} \binom{n}{k} = 2^n$ ($n$-원소 집합의 전체 부분집합 수).

---

## 4. 다항 계수

**다항 계수** (multinomial coefficient)는 이항 계수를 두 그룹 이상의 분할로 일반화한 것입니다. $n$개의 객체를 크기 $n_1, n_2, \ldots, n_k$인 $k$개의 그룹으로 나누는 방법의 수 ($\sum n_i = n$):

$$\binom{n}{n_1, n_2, \ldots, n_k} = \frac{n!}{n_1! \, n_2! \, \cdots \, n_k!}$$

### 다항 정리

$$(x_1 + x_2 + \cdots + x_k)^n = \sum_{\substack{n_1 + n_2 + \cdots + n_k = n \\ n_i \geq 0}} \binom{n}{n_1, n_2, \ldots, n_k} \prod_{i=1}^{k} x_i^{n_i}$$

**예제**: $(x + y + z)^6$에서 $x^2 y^3 z$의 계수를 구하라:

$$\binom{6}{2, 3, 1} = \frac{6!}{2! \cdot 3! \cdot 1!} = \frac{720}{2 \cdot 6 \cdot 1} = 60$$

---

## 5. 포함-배제 원리

### 두 집합의 경우

임의의 두 사건 (또는 집합) $A$와 $B$에 대해:

$$|A \cup B| = |A| + |B| - |A \cap B|$$

### 세 집합의 경우

$$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|$$

### 일반 공식

$n$개의 집합 $A_1, A_2, \ldots, A_n$에 대해:

$$\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{k=1}^{n} (-1)^{k+1} \sum_{1 \leq i_1 < i_2 < \cdots < i_k \leq n} |A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}|$$

### 풀이 예제: 제약 조건이 있는 셈

**문제**: 1부터 1000까지의 정수 중 2, 3, 또는 5로 나누어지는 수는 몇 개인가?

$A$ = 2의 배수, $B$ = 3의 배수, $C$ = 5의 배수로 놓으면:

- $|A| = \lfloor 1000/2 \rfloor = 500$
- $|B| = \lfloor 1000/3 \rfloor = 333$
- $|C| = \lfloor 1000/5 \rfloor = 200$
- $|A \cap B| = \lfloor 1000/6 \rfloor = 166$
- $|A \cap C| = \lfloor 1000/10 \rfloor = 100$
- $|B \cap C| = \lfloor 1000/15 \rfloor = 66$
- $|A \cap B \cap C| = \lfloor 1000/30 \rfloor = 33$

$$|A \cup B \cup C| = 500 + 333 + 200 - 166 - 100 - 66 + 33 = 734$$

---

## 6. 완전순열

**완전순열** (derangement)이란 **어떤 원소도 원래 위치에 있지 않은** 순열입니다. $n$개 원소의 완전순열 수를 $D_n$ (또는 $!n$)으로 표기합니다:

$$D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!}$$

이는 다음과 같은 재귀식으로도 표현됩니다:

$$D_n = (n - 1)(D_{n-1} + D_{n-2}), \quad D_0 = 1, \quad D_1 = 0$$

$n$이 클 때, $D_n \approx n!/e$.

**예제**: 4명이 각각 모자를 상자에 넣고 무작위로 재분배한다. 아무도 자기 모자를 받지 않을 확률은?

$$D_4 = 4!\left(\frac{1}{0!} - \frac{1}{1!} + \frac{1}{2!} - \frac{1}{3!} + \frac{1}{4!}\right) = 24\left(1 - 1 + \frac{1}{2} - \frac{1}{6} + \frac{1}{24}\right) = 9$$

$$P(\text{완전순열}) = \frac{D_4}{4!} = \frac{9}{24} = \frac{3}{8} = 0.375$$

---

## 7. Python 예제

### 기본 셈 함수

```python
import math
from itertools import permutations, combinations, combinations_with_replacement

# --- Permutations and Combinations ---
def P(n, r):
    """Permutations: P(n, r) = n! / (n-r)!"""
    return math.factorial(n) // math.factorial(n - r)

def C(n, r):
    """Combinations: C(n, r) = n! / (r! * (n-r)!)"""
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

# Python 3.8+ has math.comb and math.perm
print(f"P(26, 3) = {math.perm(26, 3)}")   # 15600
print(f"C(10, 4) = {math.comb(10, 4)}")    # 210

# Stars and bars: C(n+r-1, r)
n_types, r_items = 3, 8
print(f"Stars & bars: {math.comb(n_types + r_items - 1, r_items)}")  # 45
```

### 다항 계수

```python
import math

def multinomial(n, groups):
    """Compute n! / (n1! * n2! * ... * nk!)"""
    assert sum(groups) == n, "Group sizes must sum to n"
    result = math.factorial(n)
    for g in groups:
        result //= math.factorial(g)
    return result

# MISSISSIPPI arrangements
print(f"MISSISSIPPI: {multinomial(11, [1, 4, 4, 2])}")  # 34650

# Coefficient of x^2 y^3 z in (x+y+z)^6
print(f"Multinomial coeff: {multinomial(6, [2, 3, 1])}")  # 60
```

### 포함-배제

```python
def divisible_count(limit, divisors):
    """Count integers in [1, limit] divisible by at least one divisor."""
    from itertools import combinations
    n = len(divisors)
    total = 0
    for k in range(1, n + 1):
        sign = (-1) ** (k + 1)
        for combo in combinations(divisors, k):
            # LCM of the combo (for pairwise coprime, it is the product)
            lcm = combo[0]
            for d in combo[1:]:
                lcm = lcm * d // math.gcd(lcm, d)
            total += sign * (limit // lcm)
    return total

result = divisible_count(1000, [2, 3, 5])
print(f"Integers 1-1000 divisible by 2, 3, or 5: {result}")  # 734
```

### 완전순열

```python
import math

def derangements(n):
    """Compute D_n using the inclusion-exclusion formula."""
    return sum((-1)**k * math.factorial(n) // math.factorial(k)
               for k in range(n + 1))

for n in range(1, 9):
    d = derangements(n)
    ratio = d / math.factorial(n)
    print(f"D_{n} = {d:>6},  D_{n}/{n}! = {ratio:.6f}")

# Output shows ratio converging to 1/e ~ 0.367879...
```

### itertools를 이용한 열거

```python
from itertools import permutations, combinations

# All 2-element permutations of {A, B, C}
perms = list(permutations("ABC", 2))
print(f"P(3,2) = {len(perms)}: {perms}")
# 6: [('A','B'), ('A','C'), ('B','A'), ('B','C'), ('C','A'), ('C','B')]

# All 2-element combinations of {A, B, C}
combs = list(combinations("ABC", 2))
print(f"C(3,2) = {len(combs)}: {combs}")
# 3: [('A','B'), ('A','C'), ('B','C')]
```

---

## 8. 핵심 요약

| 개념 | 공식 | 적용 상황 |
|------|------|----------|
| 곱셈 법칙 | $n_1 \times n_2 \times \cdots \times n_k$ | 순차적 독립 단계 |
| 순열 (반복 없음) | $P(n,r) = n!/(n-r)!$ | 순서 있는 선택, 서로 다른 항목 |
| 순열 (반복 있음) | $n^r$ | 순서 있는 선택, 항목 재사용 가능 |
| 조합 (반복 없음) | $\binom{n}{r} = n!/[r!(n-r)!]$ | 순서 없는 선택, 서로 다른 항목 |
| 조합 (반복 있음) | $\binom{n+r-1}{r}$ | 순서 없는 선택, 항목 재사용 가능 |
| 다항 계수 | $n!/(n_1! \cdots n_k!)$ | 그룹으로 분할 |
| 포함-배제 | $\sum (-1)^{k+1} \cdots$ | 겹침이 있는 합집합 셈 |
| 완전순열 | $D_n = n! \sum (-1)^k/k!$ | 고정점이 없는 순열 |

**판단 가이드**: 셈 문제에서 두 가지를 물어보세요:

1. **순서가 중요한가?** 예 -> 순열; 아니오 -> 조합
2. **반복이 허용되는가?** 이에 따라 어떤 공식 변형을 사용할지 결정됩니다.

---

*다음 레슨: [02 - 확률 공리와 법칙](./02_Probability_Axioms_and_Rules.md)*
