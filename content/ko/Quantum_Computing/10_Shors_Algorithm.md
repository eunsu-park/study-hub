# 레슨 10: 쇼어의 소인수분해 알고리즘(Shor's Factoring Algorithm)

[← 이전: 양자 푸리에 변환](09_Quantum_Fourier_Transform.md) | [다음: 양자 오류 정정 →](11_Quantum_Error_Correction.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 정수 인수분해(integer factoring)가 고전적으로 계산하기 어려운 이유와 암호학에서 그것이 중요한 이유를 설명할 수 있다
2. 인수분해에서 주기 찾기(period finding)로의 환원(reduction)을 설명할 수 있다
3. QPE와 모듈러 지수화(modular exponentiation)를 사용하는 양자 주기 찾기 서브루틴을 단계별로 설명할 수 있다
4. 작은 수 ($N = 15, 21$)에 대해 쇼어의 알고리즘을 단계별로 추적할 수 있다
5. 쇼어의 알고리즘의 복잡도를 분석할 수 있다: 양자 게이트 $O((\log N)^3)$
6. RSA 및 포스트-양자 암호학(post-quantum cryptography)에 대한 함의를 논의할 수 있다
7. 작은 입력에 대한 쇼어의 알고리즘의 고전적 시뮬레이션을 구현할 수 있다

---

1994년, 피터 쇼어(Peter Shor)는 현대 암호학의 기반을 뒤흔드는 알고리즘을 발표했습니다. 그는 양자 컴퓨터가 큰 정수를 다항 시간(polynomial time)에 인수분해할 수 있다는 것을 보였습니다 — 최선의 고전 알고리즘은 준지수 시간(sub-exponential time)에만 해결할 수 있는 과제입니다. RSA 암호화(인터넷 보안의 근간)의 보안이 인수분해가 계산 불가능하다는 가정에 의존하기 때문에, 쇼어의 알고리즘은 충분히 강력한 양자 컴퓨터가 RSA는 물론 디피-헬만 키 교환(Diffie-Hellman key exchange) 및 타원 곡선 암호화(elliptic curve cryptography)도 깰 수 있음을 의미합니다.

쇼어의 알고리즘은 단순한 이론적 호기심이 아닙니다. 이것이 정부와 기업들이 양자 컴퓨팅 연구에 수십억 달러를 투자하는 주된 이유이며, 포스트-양자 암호화 표준 개발을 추진하는 원동력입니다 (TLS/PKI 맥락은 Security L05 참고). 쇼어의 알고리즘을 이해하려면 레슨 9의 QFT 기계가 필요하며, 양자 계산 우위(quantum computational advantage)의 가장 설득력 있는 예시를 제공합니다.

> **비유:** 쇼어의 알고리즘은 노래에서 리듬을 찾는 것처럼 양자 간섭(quantum interference)을 활용합니다. QFT는 모듈러 지수화의 주기(리듬)를 드러내는데, 이를 고전적으로 하려면 천문학적으로 긴 곡을 들어야 합니다. 10억 년 동안 지속되는 노래의 박자 주파수를 찾아야 한다고 상상해 보세요 — 고전적 방법은 매우 오랫동안 들어야 하지만, 양자 간섭은 짧은 양자 "샘플"만으로 리듬을 식별할 수 있게 해줍니다.

## 목차

1. [인수분해 문제](#1-인수분해-문제)
2. [환원: 인수분해에서 주기 찾기로](#2-환원-인수분해에서-주기-찾기로)
3. [고전적 주기 찾기의 어려움](#3-고전적-주기-찾기의-어려움)
4. [양자 주기 찾기](#4-양자-주기-찾기)
5. [완전한 쇼어의 알고리즘](#5-완전한-쇼어의-알고리즘)
6. [예제 풀이: 15 인수분해](#6-예제-풀이-15-인수분해)
7. [예제 풀이: 21 인수분해](#7-예제-풀이-21-인수분해)
8. [복잡도 분석](#8-복잡도-분석)
9. [암호학에 대한 함의](#9-암호학에-대한-함의)
10. [Python 구현](#10-python-구현)
11. [연습 문제](#11-연습-문제)

---

## 1. 인수분해 문제

### 1.1 문제 서술

**정수 인수분해(Integer Factoring)**: 합성수(composite integer) $N$이 주어졌을 때, $1 < p < N$이고 $p | N$인 비자명(non-trivial) 인수 $p$를 찾으세요.

예를 들어: $N = 15 \to p = 3$ (또는 $p = 5$), $15 = 3 \times 5$이기 때문입니다.

### 1.2 인수분해가 어렵다고 여겨지는 이유

인수분해를 위한 가장 잘 알려진 고전 알고리즘:

| 알고리즘 | 복잡도 | 유형 |
|-----------|-----------|------|
| 시행 나눗셈(Trial division) | $O(\sqrt{N})$ | $\log N$에 대해 지수적 |
| 폴라드 로(Pollard's rho) | $O(N^{1/4})$ | 준지수적 (휴리스틱) |
| 이차체(Quadratic sieve) | $O(e^{c\sqrt{\ln N \ln\ln N}})$ | 준지수적 |
| 일반 수체(General number field sieve, GNFS) | $O(e^{c(\ln N)^{1/3}(\ln\ln N)^{2/3}})$ | 준지수적 |

이 중 어느 것도 입력 크기 $n = \log_2 N$에 대해 다항식이 아닙니다. GNFS가 가장 빠른 알려진 고전 알고리즘이지만, 준지수적 실행 시간은 자릿수가 두 배가 되면 계산 시간이 세제곱 정도 증가한다는 것을 의미합니다.

### 1.3 암호학과의 연결

RSA 암호화는 두 큰 소수의 곱을 인수분해하는 어려움에 의존합니다:

1. **키 생성**: 각각 약 1024비트인 두 큰 소수 $p, q$를 선택합니다. $N = pq$를 계산합니다.
2. **공개 키**: $(N, e)$ (여기서 $e$는 $\phi(N) = (p-1)(q-1)$과 서로소)
3. **개인 키**: $d = e^{-1} \mod \phi(N)$

보안은 다음에 의존합니다: $N$이 주어졌을 때, $p$와 $q$를 찾는 것이 계산적으로 불가능해야 합니다. 인수분해가 쉬웠다면, $\phi(N)$을 계산하고 $d$를 구해 RSA를 완전히 깰 수 있습니다.

현재 RSA 키는 2048-4096비트의 $N$을 사용합니다. GNFS는 이것들을 인수분해하는 데 우주의 나이보다 더 오랜 시간이 걸립니다. 쇼어의 알고리즘은 $n = \log_2 N$일 때 대략 $4000 \cdot n$개의 논리 큐비트와 $O(n^3)$개의 게이트가 필요합니다.

---

## 2. 환원: 인수분해에서 주기 찾기로

쇼어의 알고리즘의 핵심 통찰은 **인수분해가 함수의 주기를 찾는 것으로 환원된다**는 것입니다. 이 환원은 완전히 고전적입니다 — 양자 속도 향상은 오직 주기 찾기 단계에서만 발생합니다.

### 2.1 위수 찾기(Order Finding)

$N$과 서로소인 정수 $a$ (즉, $\gcd(a, N) = 1$)에 대해, $a$의 $N$에 대한 **위수(order)**는 다음을 만족하는 가장 작은 양의 정수 $r$입니다:

$$a^r \equiv 1 \pmod{N}$$

위수는 항상 존재하며 (오일러 정리에 의해, $r | \phi(N)$) $r \leq N$입니다.

### 2.2 위수에서 인수로

**정리**: $r$이 $a$의 $N$에 대한 위수이고 $r$이 짝수이면:

$$a^r - 1 = (a^{r/2} - 1)(a^{r/2} + 1) \equiv 0 \pmod{N}$$

즉, $N | (a^{r/2} - 1)(a^{r/2} + 1)$입니다. 어느 인수도 $N$으로 나누어지지 않으면 (즉, $a^{r/2} \not\equiv \pm 1 \pmod{N}$이면):

$$\gcd(a^{r/2} - 1, N) \quad \text{와} \quad \gcd(a^{r/2} + 1, N)$$

모두 $N$의 비자명 인수입니다.

### 2.3 환원 알고리즘

$N$을 인수분해하기 위해:

1. $\{2, 3, \ldots, N-1\}$에서 무작위 $a$를 선택합니다
2. $\gcd(a, N)$을 계산합니다. $\gcd(a, N) > 1$이면 인수를 찾은 것입니다 (행운!)
3. $a$의 $N$에 대한 위수 $r$을 찾습니다 **(이것이 어려운 단계)**
4. $r$이 홀수이면 1단계로 돌아갑니다
5. $\gcd(a^{r/2} - 1, N)$과 $\gcd(a^{r/2} + 1, N)$을 계산합니다
6. 어느 것이든 비자명 인수를 제공하면 출력합니다; 그렇지 않으면 1단계로 돌아갑니다

**성공 확률**: 무작위 $a$에 대해, $r$이 짝수이고 $a^{r/2} \not\equiv -1 \pmod{N}$일 확률이 최소 $1/2$입니다 ($N$이 두 개 이상의 서로 다른 홀수 소인수를 가질 때). 따라서 평균적으로 $O(1)$번의 반복이 예상됩니다.

### 2.4 Python: 고전적 환원

```python
import numpy as np
from math import gcd

def factoring_via_order(N, a, order_r):
    """Given the order r of a mod N, attempt to find a factor.

    Why use gcd? The key algebraic identity a^r - 1 = (a^{r/2} - 1)(a^{r/2} + 1)
    means N divides the product but (hopefully) not either factor individually.
    The gcd extracts the shared factor between N and each term.
    """
    if order_r % 2 != 0:
        return None, "Order is odd — retry with different a"

    x = pow(a, order_r // 2, N)  # a^{r/2} mod N

    if x == N - 1:  # x ≡ -1 (mod N)
        return None, "a^{r/2} ≡ -1 (mod N) — retry with different a"

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    if factor1 not in (1, N):
        return factor1, f"gcd({x}-1, {N}) = {factor1}"
    if factor2 not in (1, N):
        return factor2, f"gcd({x}+1, {N}) = {factor2}"

    return None, "Both gcds are trivial — retry"

# Example: Factor 15 using a = 7
N = 15
a = 7
# Order of 7 mod 15: 7^1=7, 7^2=49≡4, 7^3=343≡13, 7^4=2401≡1 → r=4
r = 4
factor, msg = factoring_via_order(N, a, r)
print(f"N={N}, a={a}, r={r}: {msg}")
if factor:
    print(f"  {N} = {factor} × {N // factor}")
```

---

## 3. 고전적 주기 찾기의 어려움

### 3.1 주기 찾기 문제

함수 $f(x) = a^x \bmod N$이 주어졌을 때, 주기 $r$ — 즉 모든 $x$에 대해 $f(x + r) = f(x)$를 만족하는 가장 작은 양의 정수 — 를 찾으세요.

### 3.2 고전적 접근법이 실패하는 이유

**브루트포스(Brute force)**: $a^0, a^1, a^2, \ldots$를 $a^r \equiv 1$이 될 때까지 계산합니다. $r$이 $N$만큼 클 수 있고, $N$이 2048비트 수 ($N \sim 2^{2048}$)일 수 있으므로, 천문학적인 수의 단계가 필요합니다.

**생일 공격(Birthday-style attacks)**: 폴라드 로(Pollard's rho) 알고리즘은 $O(\sqrt{r})$의 공간과 시간을 사용하지만, $\sqrt{r}$도 여전히 $\sim 2^{1024}$이 될 수 있어 손에 닿지 않습니다.

근본적인 문제는 고전적으로 $f(x) = a^x \bmod N$에서 많은 점을 평가하지 않고 주기성을 감지하는 방법이 알려져 있지 않다는 것입니다. 양자 역학은 지름길을 제공합니다: **모든 입력의 중첩에서 $f$를 동시에 평가한 후, QFT를 사용해 간섭 패턴에서 주기를 추출합니다**.

---

## 4. 양자 주기 찾기

### 4.1 개요

양자 서브루틴은 두 개의 레지스터를 사용합니다:

- **카운팅 레지스터(Counting register)**: $t$큐비트 ($t = 2n + 1$이고 $n = \lceil \log_2 N \rceil$, 충분한 정밀도)
- **타겟 레지스터(Target register)**: $f(x) = a^x \bmod N$을 저장하는 $n$큐비트

### 4.2 단계별 설명

**1단계: 초기화**

$$|0\rangle^{\otimes t} |0\rangle^{\otimes n}$$

**2단계: 중첩 생성** (카운팅 레지스터에 $H^{\otimes t}$ 적용)

$$\frac{1}{\sqrt{2^t}} \sum_{x=0}^{2^t - 1} |x\rangle |0\rangle$$

**3단계: 모듈러 지수화 적용** (양자 오라클)

$$\frac{1}{\sqrt{2^t}} \sum_{x=0}^{2^t - 1} |x\rangle |a^x \bmod N\rangle$$

이것이 가장 비싼 단계입니다: 제어 모듈러 지수화 회로는 $O(n^3)$개의 게이트를 필요로 합니다.

**4단계: 카운팅 레지스터에 역 QFT 적용**

QFT (엄밀히는 역 QFT) 후, 카운팅 레지스터는 주기 $r$에 대한 정보를 인코딩하는 상태들의 중첩을 포함합니다.

**5단계: 카운팅 레지스터 측정**

측정 결과 $m$은 다음을 근사적으로 만족합니다:

$$\frac{m}{2^t} \approx \frac{s}{r}$$

여기서 $s \in \{0, 1, \ldots, r-1\}$인 어떤 정수입니다. $m/2^t$에 연분수(continued fraction) 알고리즘을 적용하면 $r$을 추출할 수 있습니다.

### 4.3 QFT가 주기를 드러내는 이유

3단계 후, 타겟 레지스터를 (개념적으로) 측정하여 어떤 값 $f_0 = a^{x_0} \bmod N$을 얻었다고 가정합니다. 카운팅 레지스터는 $f_0$으로 매핑되는 모든 $x$ 값의 중첩으로 붕괴됩니다:

$$|\psi\rangle = \frac{1}{\sqrt{A}} \sum_{j=0}^{A-1} |x_0 + jr\rangle$$

여기서 $A \approx 2^t / r$은 유효한 $x$ 값의 수입니다. 이 상태는 주기 $r$을 가집니다.

주기 상태에 QFT를 적용하면 확률이 $2^t / r$의 배수에 집중되어, 정수 $s$에 대해 $s \cdot 2^t / r$ 근처의 측정 결과를 줍니다. 이것은 레슨 9에서 공부한 주기성 검출 성질과 정확히 일치합니다.

### 4.4 연분수 알고리즘(Continued Fraction Algorithm)

측정 결과 $m$으로부터 분수 $m / 2^t$를 형성하고 연분수 전개를 계산합니다. 분모가 $\leq N$인 수렴분수(convergent)가 $r$의 후보를 제공합니다.

예를 들어, $2^t = 256$이고 $m = 85$이면:

$$\frac{85}{256} = 0.33203125 \approx \frac{1}{3}$$

$85/256$의 연분수 전개는 $[0; 3, 85/256 \text{ 세부사항}]$이고, 이것은 수렴분수 $1/3$을 줍니다. 따라서 $r = 3$이 후보 주기입니다.

```python
def continued_fraction(numerator, denominator, max_denom):
    """Extract the period using continued fractions.

    Why continued fractions? The measurement outcome m/2^t is a noisy estimate
    of s/r. Continued fractions find the "simplest" fraction (smallest
    denominator) close to a given real number — exactly what we need to
    recover r from an approximate rational.
    """
    # Compute continued fraction coefficients
    cf = []
    n, d = numerator, denominator
    while d > 0 and len(cf) < 50:
        cf.append(n // d)
        n, d = d, n % d

    # Compute convergents and return the one with denominator ≤ max_denom
    convergents = []
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    for a in cf:
        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
        if k_curr > max_denom:
            break
        convergents.append((h_curr, k_curr))

    return convergents

# Example: recover period from measurement outcome
m = 85
T = 256
N = 15

convergents = continued_fraction(m, T, N)
print(f"m/2^t = {m}/{T} = {m/T:.6f}")
print("Convergents:")
for h, k in convergents:
    print(f"  {h}/{k} = {h/k:.6f}")
```

---

## 5. 완전한 쇼어의 알고리즘

### 5.1 알고리즘 요약

**입력**: 인수분해할 합성수 $N$.

**출력**: $N$의 비자명 인수.

1. **자명한 경우 확인**: $N$이 짝수이면 2를 반환합니다. $N = p^k$ (어떤 소수 $p$와 $k \geq 2$)이면 고전적 방법을 사용합니다.

2. **무작위 $a$ 선택**: $\{2, \ldots, N-1\}$에서 균등하게 $a$를 선택합니다.

3. **GCD 확인**: $g = \gcd(a, N)$을 계산합니다. $g > 1$이면 $g$를 반환합니다.

4. **양자 주기 찾기**:
   - 카운팅 레지스터 ($t = 2\lceil\log_2 N\rceil + 1$큐비트)와 작업 레지스터 ($\lceil\log_2 N\rceil$큐비트) 준비
   - 카운팅 레지스터에 $H^{\otimes t}$ 적용
   - 제어 모듈러 지수화 적용: $|x\rangle|y\rangle \to |x\rangle|y \cdot a^x \bmod N\rangle$
   - 카운팅 레지스터에 $\text{QFT}^{-1}$ 적용
   - 카운팅 레지스터를 측정하여 결과 $m$ 획득

5. **고전적 후처리(Classical post-processing)**:
   - $m / 2^t$에 연분수를 적용하여 후보 주기 $r$을 찾습니다
   - $r$이 홀수이거나 $a^{r/2} \equiv -1 \pmod{N}$이면 2단계로 이동합니다
   - $\gcd(a^{r/2} \pm 1, N)$을 계산하고 비자명 인수를 반환합니다

6. **필요시 반복** (예상 $O(1)$번의 반복).

### 5.2 회로 다이어그램

```
Counting register (t qubits):

|0⟩ ─── H ─── ───────── ───── ───── ┌───────┐
|0⟩ ─── H ─── ───────── ───── U²   │       │
|0⟩ ─── H ─── ───────── U⁴   ───── │ QFT⁻¹ │ ── Measure → m
|0⟩ ─── H ─── U⁸       ───── ───── │       │
        ...                          └───────┘

Work register (n qubits):

|1⟩ ─── ───── target ── target target ────────────
```

여기서 $U|y\rangle = |ay \bmod N\rangle$은 모듈러 곱셈 연산자입니다. 제어-$U^{2^j}$ 게이트들은 제어 큐비트가 $|1\rangle$일 때 $|y\rangle \to |a^{2^j} y \bmod N\rangle$을 구현합니다.

---

## 6. 예제 풀이: 15 인수분해

### 6.1 설정

$N = 15$. $N$을 표현하는 데 $n = \lceil\log_2 15\rceil = 4$비트가 필요합니다.

$a = 7$을 선택합니다 (15와 서로소: $\gcd(7, 15) = 1$ 확인됨).

### 6.2 위수의 고전적 계산 (검증용)

$$7^1 = 7, \quad 7^2 = 49 \equiv 4, \quad 7^3 = 343 \equiv 13, \quad 7^4 = 2401 \equiv 1 \pmod{15}$$

따라서 $r = 4$.

### 6.3 양자 주기 찾기

$t = 2 \times 4 + 1 = 9$개의 카운팅 큐비트를 사용하므로 $2^t = 512$.

모듈러 지수화와 역 QFT 후, 카운팅 레지스터는 $2^t / r = 512 / 4 = 128$의 배수 근처에서 피크를 가집니다:

$$m \in \{0, 128, 256, 384\}$$

각각 약 $\approx 1/4$의 확률로 측정됩니다.

### 6.4 후처리

| 측정값 $m$ | $m/2^t$ | 연분수 | 후보 $r$ | 유효? |
|-----|---------|-------------------|---------------|--------|
| 0 | 0/512 = 0 | $0/1$ | 1 | 아니오 (자명) |
| 128 | 128/512 = 1/4 | $1/4$ | 4 | **예** |
| 256 | 256/512 = 1/2 | $1/2$ | 2 | 시도 필요 |
| 384 | 384/512 = 3/4 | $3/4$ | 4 | **예** |

$r = 4$ (짝수, 좋음):

$$a^{r/2} = 7^2 = 49 \equiv 4 \pmod{15}$$

$4 \not\equiv -1 \pmod{15}$이므로:

$$\gcd(4 - 1, 15) = \gcd(3, 15) = 3$$
$$\gcd(4 + 1, 15) = \gcd(5, 15) = 5$$

**결과**: $15 = 3 \times 5$. 인수분해 성공!

---

## 7. 예제 풀이: 21 인수분해

### 7.1 설정

$N = 21 = 3 \times 7$. $a = 2$를 선택합니다.

### 7.2 위수 계산

$$2^1 = 2, \quad 2^2 = 4, \quad 2^3 = 8, \quad 2^4 = 16, \quad 2^5 = 32 \equiv 11, \quad 2^6 = 64 \equiv 1 \pmod{21}$$

따라서 $r = 6$.

### 7.3 후처리

$r = 6$은 짝수입니다. $a^{r/2} = 2^3 = 8$을 계산합니다.

확인: $8 \not\equiv -1 \pmod{21}$ ($-1 \equiv 20$이므로).

$$\gcd(8 - 1, 21) = \gcd(7, 21) = 7$$
$$\gcd(8 + 1, 21) = \gcd(9, 21) = 3$$

**결과**: $21 = 3 \times 7$. 성공!

### 7.4 $a = 4$를 선택했다면?

$$4^1 = 4, \quad 4^2 = 16, \quad 4^3 = 64 \equiv 1 \pmod{21}$$

$r = 3$은 홀수입니다. 알고리즘은 다른 $a$로 재시도합니다. 이것은 때때로 여러 번 시도가 필요한 이유를 보여줍니다.

---

## 8. 복잡도 분석

### 8.1 게이트 수

쇼어의 알고리즘의 양자 회로는 세 가지 주요 구성 요소를 가집니다:

| 구성 요소 | 게이트 수 |
|-----------|-----------|
| 아다마르 레이어 ($H^{\otimes t}$) | $O(n)$ |
| 모듈러 지수화 ($a^x \bmod N$) | $O(n^3)$ |
| 역 QFT | $O(n^2)$ |

모듈러 지수화가 지배적입니다: $O(n)$번의 순차적 제어 모듈러 곱셈이 필요하고, 각각 $O(n^2)$개의 게이트가 필요합니다. 총합: $n = \lceil\log_2 N\rceil$일 때 $O(n^3)$.

### 8.2 큐비트 수

- 카운팅 레지스터: $t = 2n + 1$큐비트
- 작업 레지스터: $n$큐비트
- 모듈러 산술을 위한 보조(ancilla) 큐비트: $O(n)$

**총합**: $O(n)$큐비트 $= O(\log N)$큐비트.

RSA-2048 ($n = 2048$비트)의 경우, 대략 $4n + O(n) \approx 10{,}000$개의 논리 큐비트가 필요합니다. 양자 오류 정정 오버헤드 (논리 큐비트당 약 1000-10000개의 물리 큐비트)를 적용하면 수백만 개의 물리 큐비트가 필요합니다.

### 8.3 전체 복잡도

$$\text{쇼어의 알고리즘}: O(n^3) = O((\log N)^3) \text{ 개의 양자 게이트}$$

최선의 고전 알고리즘 (GNFS)과 비교하면:

$$\text{GNFS}: O\left(\exp\left(c \cdot (\log N)^{1/3} (\log\log N)^{2/3}\right)\right)$$

2048비트의 $N$에 대해:
- **쇼어의 알고리즘**: $\sim 2048^3 \approx 10^{10}$번의 연산 (실현 가능)
- **GNFS**: $\sim 10^{30}$번의 연산 (불가능)

이것은 *지수적* 양자 속도 향상입니다.

---

## 9. 암호학에 대한 함의

### 9.1 쇼어의 알고리즘이 깨는 것

| 암호 시스템 | 기반 | 쇼어에 의해 깨지는가? |
|-------------|----------|-----------------|
| RSA | 정수 인수분해 | **예** |
| 디피-헬만(Diffie-Hellman) | 이산 로그(Discrete logarithm) | **예** (동일 알고리즘, 약간 수정) |
| 타원 곡선 (ECDSA, ECDH) | 타원 곡선 이산 로그 | **예** |
| AES (대칭) | — | **아니오** (그로버가 $\sqrt{}$ 속도 향상 제공, 키 크기 두 배로 완화) |
| SHA-256 (해싱) | — | **아니오** (그로버가 역상(preimage)에 $\sqrt{}$ 속도 향상 제공) |

### 9.2 타임라인과 위협 평가

2025년 기준:

- **쇼어의 알고리즘으로 인수분해된 가장 큰 수**: 21 (실제 양자 하드웨어에서)
- **양자 방법으로 인수분해된 가장 큰 수**: 변분적 방법을 사용한 작은 수 (<100)
- **RSA-2048 위협 예상 시기**: 2030년대-2040년대 (하드웨어 발전에 따라 다름)
- **지금 수집, 나중에 복호화 (Harvest now, decrypt later)**: 적대적 행위자들이 양자 컴퓨터를 사용 가능해질 때 복호화하기 위해 이미 암호화된 데이터를 수집하고 있을 수 있습니다

### 9.3 포스트-양자 암호학(Post-Quantum Cryptography)

NIST는 2024년 포스트-양자 알고리즘을 표준화했습니다:

- **CRYSTALS-Kyber** (ML-KEM): 격자 기반 키 캡슐화(Lattice-based key encapsulation)
- **CRYSTALS-Dilithium** (ML-DSA): 격자 기반 디지털 서명(Lattice-based digital signatures)
- **SPHINCS+** (SLH-DSA): 해시 기반 디지털 서명(Hash-based digital signatures)

이것들은 고전적 공격과 양자 공격 모두에 안전하다고 여겨집니다 (TLS/PKI 맥락은 Security L05 참고).

---

## 10. Python 구현

### 10.1 완전한 쇼어의 알고리즘 시뮬레이션

```python
import numpy as np
from math import gcd, isqrt

def is_prime_power(N):
    """Check if N is a prime power.

    Why check this first? Shor's algorithm assumes N is composite and not
    a prime power. Prime powers can be factored classically in polynomial
    time using Newton's method for k-th roots.
    """
    for k in range(2, int(np.log2(N)) + 1):
        root = round(N**(1/k))
        for r in [root - 1, root, root + 1]:
            if r > 1 and r**k == N:
                return True, r, k
    return False, None, None

def classical_order_finding(a, N):
    """Find the order of a modulo N by brute force.

    Why brute force? On a real quantum computer, this step is replaced by
    quantum period finding. We use the classical version for simulation
    and verification of small cases.
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            return None  # Should not happen if gcd(a,N) = 1
    return r

def quantum_period_finding_simulation(a, N, n_counting=None):
    """Simulate the quantum period-finding subroutine.

    This simulates what the quantum circuit does: it creates the probability
    distribution that the inverse QFT produces, then samples from it.
    A real quantum computer would use actual quantum gates.
    """
    if n_counting is None:
        n = int(np.ceil(np.log2(N)))
        n_counting = 2 * n + 1

    Q = 2**n_counting  # Size of counting register

    # Build the state after modular exponentiation + inverse QFT
    # For each measurement outcome m, the probability is:
    # P(m) = |1/Q * sum_{x: a^x mod N = f0} exp(-2πi·m·x/Q)|²
    # summed over all possible f0 values

    # For simulation, we compute the order classically, then simulate
    # the measurement distribution
    r = classical_order_finding(a, N)
    if r is None:
        return None, None

    # The probability peaks near multiples of Q/r
    # P(m) is largest when m ≈ s·Q/r for integer s
    probs = np.zeros(Q)
    for m in range(Q):
        # Sum over s from 0 to r-1
        amplitude = 0
        for x0 in range(r):
            # Number of terms in the sum with this residue
            count = (Q - x0 + r - 1) // r
            for j in range(count):
                x = x0 + j * r
                amplitude += np.exp(-2j * np.pi * m * x / Q)
        probs[m] = abs(amplitude / Q)**2

    # Normalize
    probs /= probs.sum()

    # Sample from the distribution
    measurement = np.random.choice(Q, p=probs)

    return measurement, Q

def continued_fractions_period(m, Q, N):
    """Extract candidate period from measurement using continued fractions.

    Why continued fractions? The measurement gives m/Q ≈ s/r for unknown s and r.
    Continued fractions find the best rational approximation with small denominator,
    which recovers r even from an imperfect measurement.
    """
    if m == 0:
        return None

    # Compute continued fraction expansion of m/Q
    candidates = []
    n, d = m, Q
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    while d > 0:
        a_i = n // d
        n, d = d, n % d

        h_prev, h_curr = h_curr, a_i * h_curr + h_prev
        k_prev, k_curr = k_curr, a_i * k_curr + k_prev

        if k_curr > N:
            break
        if k_curr > 0:
            candidates.append(k_curr)

    return candidates

def shors_algorithm(N, max_attempts=20):
    """Complete Shor's algorithm for factoring N.

    Returns a non-trivial factor of N, or None if all attempts fail.
    """
    print(f"Factoring N = {N}")
    print("=" * 50)

    # Step 1: Trivial checks
    if N % 2 == 0:
        print(f"  N is even: factor = 2")
        return 2

    is_pp, base, exp = is_prime_power(N)
    if is_pp:
        print(f"  N = {base}^{exp}: factor = {base}")
        return base

    # Step 2-6: Main loop
    for attempt in range(1, max_attempts + 1):
        a = np.random.randint(2, N)
        print(f"\n  Attempt {attempt}: a = {a}")

        # Step 3: Check GCD
        g = gcd(a, N)
        if g > 1:
            print(f"    Lucky! gcd({a}, {N}) = {g}")
            return g

        # Step 4: Quantum period finding (simulated)
        r = classical_order_finding(a, N)
        print(f"    Order of {a} mod {N}: r = {r}")

        if r is None:
            continue

        # Step 5: Check conditions
        if r % 2 != 0:
            print(f"    r = {r} is odd — retrying")
            continue

        x = pow(a, r // 2, N)
        print(f"    a^(r/2) mod N = {a}^{r//2} mod {N} = {x}")

        if x == N - 1:
            print(f"    a^(r/2) ≡ -1 (mod N) — retrying")
            continue

        # Step 6: Compute factors
        f1 = gcd(x - 1, N)
        f2 = gcd(x + 1, N)
        print(f"    gcd({x}-1, {N}) = {f1}")
        print(f"    gcd({x}+1, {N}) = {f2}")

        if f1 not in (1, N):
            print(f"    SUCCESS: {N} = {f1} × {N // f1}")
            return f1
        if f2 not in (1, N):
            print(f"    SUCCESS: {N} = {f2} × {N // f2}")
            return f2

        print(f"    Both factors trivial — retrying")

    print(f"\n  Failed after {max_attempts} attempts")
    return None

# Run Shor's algorithm on several numbers
for N in [15, 21, 35, 77, 91]:
    factor = shors_algorithm(N)
    print()
```

### 10.2 양자 측정 시뮬레이션

```python
import numpy as np
from math import gcd

def simulate_shor_measurement(a, N, n_counting_extra=3):
    """Simulate the quantum measurement step of Shor's algorithm.

    This shows the probability distribution over measurement outcomes
    and demonstrates how the peaks encode the period.
    """
    n = int(np.ceil(np.log2(N)))
    t = 2 * n + n_counting_extra  # Extra bits for precision
    Q = 2**t

    # Find the true order (for simulation purposes)
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1

    print(f"Simulating Shor's measurement for a={a}, N={N}")
    print(f"  True period: r = {r}")
    print(f"  Counting register: {t} qubits (Q = {Q})")
    print(f"  Expected peaks near multiples of Q/r = {Q/r:.2f}")

    # Compute probability distribution
    # The state after modular exponentiation has the form:
    # (1/√Q) Σ_x |x⟩|a^x mod N⟩
    # After measuring the work register (conceptually), we get:
    # (1/√A) Σ_j |x_0 + jr⟩ for some x_0
    # The inverse QFT maps this to peaks near multiples of Q/r

    probs = np.zeros(Q)
    A = Q // r  # Approximate number of terms

    for m in range(Q):
        # Average over all possible x_0 values (0 to r-1)
        total_prob = 0
        for x0 in range(r):
            amplitude = 0
            j = 0
            x = x0
            while x < Q:
                amplitude += np.exp(-2j * np.pi * m * x / Q)
                x += r
                j += 1
            total_prob += abs(amplitude)**2
        probs[m] = total_prob / (r * Q)

    probs /= probs.sum()

    # Show the peaks
    print(f"\n  Top measurement outcomes:")
    top_indices = np.argsort(probs)[::-1][:min(2*r, 10)]
    for idx in sorted(top_indices):
        if probs[idx] > 0.005:
            ratio = idx / Q
            # Find nearest s/r
            best_s = round(ratio * r)
            print(f"    m = {idx:4d}: P = {probs[idx]:.4f}, "
                  f"m/Q = {ratio:.6f}, nearest s/r = {best_s}/{r} = {best_s/r:.6f}")

    # Simulate sampling
    print(f"\n  Simulating 10 measurements:")
    for trial in range(10):
        m = np.random.choice(Q, p=probs)
        candidates = continued_fractions_period(m, Q, N)
        print(f"    m = {m}, m/Q = {m/Q:.6f}, "
              f"candidate periods: {candidates}")

# Simulate for N=15, a=7
simulate_shor_measurement(7, 15)

print("\n" + "=" * 60)

# Simulate for N=21, a=2
simulate_shor_measurement(2, 21)
```

### 10.3 모듈러 지수화 (고전적)

```python
def modular_exponentiation_trace(a, x, N):
    """Compute a^x mod N using repeated squaring, showing the trace.

    Why repeated squaring? This is the classical algorithm that makes
    modular exponentiation efficient: O(log x) multiplications instead of
    O(x). The quantum circuit uses a similar decomposition, with each
    squaring step becoming a controlled-U^{2^j} operation.
    """
    # Write x in binary: x = x_{t-1} * 2^{t-1} + ... + x_1 * 2 + x_0
    binary = bin(x)[2:]
    t = len(binary)

    print(f"Computing {a}^{x} mod {N}")
    print(f"  x = {x} = {binary}₂ ({t} bits)")

    # Precompute a^{2^j} mod N for j = 0, ..., t-1
    powers = [a % N]
    for j in range(1, t):
        powers.append(powers[-1]**2 % N)
    print(f"  Powers of a: a^(2^j) mod {N} = {powers}")

    # Multiply the powers corresponding to 1-bits
    result = 1
    for j in range(t):
        bit = int(binary[t - 1 - j])
        if bit == 1:
            result = (result * powers[j]) % N
            print(f"  bit {j} = 1: multiply by {powers[j]}, result = {result}")
        else:
            print(f"  bit {j} = 0: skip")

    print(f"  Final: {a}^{x} mod {N} = {result}")
    return result

# Trace modular exponentiation for Shor's algorithm
modular_exponentiation_trace(7, 11, 15)
```

---

## 11. 연습 문제

### 연습 1: 쇼어의 알고리즘 수동 계산

$N = 35$를 인수분해하기 위해 쇼어의 알고리즘을 손으로 적용하세요:
(a) $a = 2$를 선택하세요. 2의 35에 대한 위수를 계산하세요.
(b) $r$이 짝수입니까? 짝수라면 $a^{r/2} \bmod 35$를 계산하세요.
(c) $\gcd(a^{r/2} \pm 1, 35)$를 계산하여 인수를 찾으세요.
(d) $a = 3$으로 반복하세요. 작동합니까? 이유는 무엇입니까?

### 연습 2: 측정 분포

시뮬레이션 코드를 사용하여 $N = 21$, $a = 2$ (주기 $r = 6$)에 대한 쇼어의 알고리즘의 측정 분포를 분석하세요.
(a) 확률 분포에 피크가 몇 개 있습니까?
(b) 각 피크에 대해 어떤 정수 $s$에 대해 $m/Q \approx s/r$임을 확인하세요.
(c) 연분수를 통해 올바른 주기를 성공적으로 얻는 측정 결과의 비율은 얼마입니까?

### 연습 3: 성공 확률 분석

$N = pq$ ($p$와 $q$는 서로 다른 홀수 소수)인 경우:
(a) $N$과 서로소인 $a$의 값이 정확히 $\phi(N) = (p-1)(q-1)$개임을 보이세요.
(b) 이 중 짝수 위수 $r$을 주는 비율은 얼마입니까? (힌트: 중국인의 나머지 정리(Chinese Remainder Theorem) 고려)
(c) $N = 77 = 7 \times 11$에 대해 100번의 무작위 $a$ 선택으로 쇼어의 알고리즘을 시뮬레이션하세요. 첫 번째 시도에서 성공하는 비율은 얼마입니까?

### 연습 4: 포스트-양자 보안

양자 컴퓨터가 초당 $10^{10}$개의 게이트를 실행할 수 있다고 가정하세요.
(a) 쇼어의 알고리즘 ($n = 2048$인 $O(n^3)$ 게이트)을 사용하여 RSA-2048을 인수분해하는 시간을 추정하세요.
(b) RSA-4096에 대한 시간을 추정하세요.
(c) RSA-2048에 필요한 논리 큐비트 수는 얼마입니까? ($2n + O(n)$ 추정치 사용)
(d) 각 논리 큐비트가 오류 정정을 위해 1000개의 물리 큐비트를 필요로 한다면, 필요한 물리 큐비트 수는 얼마입니까?

### 연습 5: 더 큰 수에 대한 위수 찾기 구현

$N = 143 = 11 \times 13$과 $N = 221 = 13 \times 17$을 처리하도록 시뮬레이션을 확장하세요:
(a) 각각에 대해 성공적인 인수분해를 제공하는 모든 $a$ 값을 찾으세요.
(b) 성공 확률 (유효한 $a$ 값의 비율)을 계산하세요.
(c) $N$이 커질수록 측정 분포는 어떻게 변합니까? (분포를 그래프로 그리세요.)

---

[← 이전: 양자 푸리에 변환](09_Quantum_Fourier_Transform.md) | [다음: 양자 오류 정정 →](11_Quantum_Error_Correction.md)
