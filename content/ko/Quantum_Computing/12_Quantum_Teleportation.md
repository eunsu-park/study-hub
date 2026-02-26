# 레슨 12: 양자 텔레포테이션과 통신(Quantum Teleportation and Communication)

[← 이전: 양자 오류 정정](11_Quantum_Error_Correction.md) | [다음: 변분 양자 고유값 해법 →](13_VQE.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 복제 불가 정리(No-Cloning Theorem)를 진술하고 증명하며 그 물리적 의미를 설명할 수 있다
2. 고전적 통신의 역할을 포함하여 양자 텔레포테이션(Quantum Teleportation) 프로토콜을 단계별로 설명할 수 있다
3. 텔레포테이션이 비신호 원리(No-Signaling Principle)나 특수 상대성 이론을 위반하지 않는 이유를 설명할 수 있다
4. 초밀 부호화(Superdense Coding) 구현 — 1개의 큐비트와 공유 얽힘을 사용하여 2개의 고전 비트 전송
5. BB84 양자 키 분배(Quantum Key Distribution) 프로토콜과 그 보안 근거를 개괄할 수 있다
6. 장거리 양자 통신을 위한 양자 중계기(Quantum Repeater) 개념을 설명할 수 있다
7. Python으로 텔레포테이션, 초밀 부호화, BB84를 시뮬레이션할 수 있다

---

1993년 Bennett 등이 처음 제안한 양자 텔레포테이션은 양자 정보 이론에서 가장 놀랍고 반직관적인 현상 중 하나다. 공유된 얽힘 쌍과 두 비트의 고전적 통신만을 사용하여 미지의 양자 상태를 한 위치에서 다른 위치로 전송할 수 있게 한다. 원래의 양자 상태는 과정에서 파괴되며(복제 불가 만족), 양자 정보를 전달하는 물리적 물질이나 에너지는 두 위치 사이를 이동하지 않는다.

공상과학 소설 같은 이름에도 불구하고, 양자 텔레포테이션은 빛보다 빠른 통신을 허용하지 않는다. 프로토콜에 필요한 고전적 비트들은 빛의 속도(또는 그 이하)로 이동하며, 그것들 없이는 수신자에게 오직 무작위 잡음만이 있다. 양자 얽힘과 고전적 통신 사이의 이 미묘한 상호작용은 양자역학에서 정보의 본질에 대한 심오한 진실을 드러낸다.

> **비유:** 양자 텔레포테이션은 문서를 팩스로 전송하는 것과 같습니다. 원본은 파괴되고(복제 불가), 정보는 고전적 및 양자적 채널을 통해 이동하며, 목적지에 동일한 복사본이 나타납니다. 팩스가 전화선(고전적 채널)과 팩스 기계(공유 자원) 모두를 필요로 하는 것처럼, 텔레포테이션도 고전적 비트와 공유된 얽힘 쌍 모두를 필요로 합니다.

## 목차

1. [복제 불가 정리 재고찰](#1-복제-불가-정리-재고찰)
2. [양자 텔레포테이션 프로토콜](#2-양자-텔레포테이션-프로토콜)
3. [텔레포테이션이 비신호 원리를 위반하지 않는 이유](#3-텔레포테이션이-비신호-원리를-위반하지-않는-이유)
4. [초밀 부호화](#4-초밀-부호화)
5. [양자 키 분배: BB84](#5-양자-키-분배-bb84)
6. [양자 중계기](#6-양자-중계기)
7. [Python 구현](#7-python-구현)
8. [연습 문제](#8-연습-문제)

---

## 1. 복제 불가 정리 재고찰

### 1.1 공식 진술

**복제 불가 정리(No-Cloning Theorem)**: 모든 양자 상태 $|\psi\rangle$에 대해 다음을 만족하는 유니타리 연산자 $U$와 고정된 보조 상태(Ancilla State) $|s\rangle$는 존재하지 않는다:

$$U(|\psi\rangle \otimes |s\rangle) = |\psi\rangle \otimes |\psi\rangle$$

### 1.2 증명

그러한 $U$가 존재한다고 가정하자. 두 임의의 상태 $|\psi\rangle$와 $|\phi\rangle$에 대해:

$$U(|\psi\rangle|s\rangle) = |\psi\rangle|\psi\rangle$$
$$U(|\phi\rangle|s\rangle) = |\phi\rangle|\phi\rangle$$

$U$는 유니타리이므로 내적을 보존한다:

$$\langle\psi|\phi\rangle \cdot \langle s|s\rangle = (\langle\psi|\phi\rangle)^2$$

$\langle s|s\rangle = 1$이므로:

$$\langle\psi|\phi\rangle = (\langle\psi|\phi\rangle)^2$$

$c = \langle\psi|\phi\rangle$로 놓으면. $c = c^2$이므로 $c(c-1) = 0$, 즉 $c = 0$ 또는 $c = 1$이다.

이것은 $U$가 동일($c = 1$)하거나 직교($c = 0$)하는 상태만 복제할 수 있음을 의미한다. 일반적인 미지의 양자 상태는 복제할 수 없다. $\square$

### 1.3 물리적 의미

복제 불가 정리는 심오한 시사점을 가진다:

- **양자 텔레포테이션**: 원래 상태는 텔레포테이션 중에 파괴되어야 한다(복사본이 남지 않음)
- **양자 암호학**: 도청자는 양자 키 비트를 방해하지 않고 복사할 수 없다
- **양자 오류 정정**: 단순한 복제로는 양자 정보를 보호할 수 없다 — 얽힘 기반 인코딩을 사용해야 한다(레슨 11)
- **양자 컴퓨팅**: 고전적 비트를 팬아웃(Fan-Out)하는 것처럼 양자 정보를 팬아웃할 수 없다

### 1.4 복제 *가능한* 것

- **알려진 상태**: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$을 알고 있다면(즉, $\alpha$와 $\beta$를 안다면) 원하는 만큼 많은 복사본을 준비할 수 있다
- **직교 상태**: CNOT 게이트는 계산 기저 상태를 복제한다: $\text{CNOT}|b\rangle|0\rangle = |b\rangle|b\rangle$ ($b \in \{0, 1\}$)
- **고전적 정보**: 고전 비트는 자유롭게 복사될 수 있다 (고전적 정보는 직교 상태로 표현됨)

---

## 2. 양자 텔레포테이션 프로토콜

### 2.1 설정

세 당사자와 세 큐비트:

- **앨리스(Alice)**는 큐비트 $A$를 미지의 상태 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$로 가지고 있으며 이를 밥(Bob)에게 텔레포테이션하려 한다
- **앨리스**는 또한 큐비트 $E_A$ (얽힘 쌍의 절반)를 가진다
- **밥**은 큐비트 $E_B$ (얽힘 쌍의 나머지 절반)를 가진다

앨리스와 밥이 공유하는 얽힘 쌍은 벨 상태(Bell State)다:

$$|\Phi^+\rangle_{E_A E_B} = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

### 2.2 초기 상태

완전한 3-큐비트 상태는:

$$|\Psi_0\rangle = |\psi\rangle_A \otimes |\Phi^+\rangle_{E_A E_B}$$

$$= (\alpha|0\rangle + \beta|1\rangle)_A \otimes \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)_{E_A E_B}$$

$$= \frac{1}{\sqrt{2}} [\alpha|0\rangle(|00\rangle + |11\rangle) + \beta|1\rangle(|00\rangle + |11\rangle)]$$

$$= \frac{1}{\sqrt{2}} [\alpha|000\rangle + \alpha|011\rangle + \beta|100\rangle + \beta|111\rangle]$$

### 2.3 1단계: 앨리스가 CNOT 적용

앨리스는 큐비트 $A$를 제어로, $E_A$를 타겟으로 하는 CNOT 게이트를 적용한다:

$$|\Psi_1\rangle = \frac{1}{\sqrt{2}} [\alpha|000\rangle + \alpha|011\rangle + \beta|110\rangle + \beta|101\rangle]$$

### 2.4 2단계: 앨리스가 아다마르 적용

앨리스는 큐비트 $A$에 아다마르(Hadamard) 게이트를 적용한다:

$H|0\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$이고 $H|1\rangle = (|0\rangle - |1\rangle)/\sqrt{2}$를 사용하면:

$$|\Psi_2\rangle = \frac{1}{2} [\alpha(|0\rangle + |1\rangle)|00\rangle + \alpha(|0\rangle + |1\rangle)|11\rangle + \beta(|0\rangle - |1\rangle)|10\rangle + \beta(|0\rangle - |1\rangle)|01\rangle]$$

앨리스의 두 큐비트($A$와 $E_A$)로 재분류하면:

$$|\Psi_2\rangle = \frac{1}{2} [|00\rangle(\alpha|0\rangle + \beta|1\rangle) + |01\rangle(\alpha|1\rangle + \beta|0\rangle) + |10\rangle(\alpha|0\rangle - \beta|1\rangle) + |11\rangle(\alpha|1\rangle - \beta|0\rangle)]$$

### 2.5 3단계: 앨리스가 측정

앨리스는 계산 기저에서 두 큐비트($A$와 $E_A$)를 측정한다. 각 결과는 확률 $1/4$로 발생한다:

| 앨리스의 결과 | 밥의 큐비트 상태 | 필요한 정정 |
|----------------|-------------------|-------------------|
| $\|00\rangle$ | $\alpha\|0\rangle + \beta\|1\rangle$ | 없음 ($I$) |
| $\|01\rangle$ | $\alpha\|1\rangle + \beta\|0\rangle$ | $X$ |
| $\|10\rangle$ | $\alpha\|0\rangle - \beta\|1\rangle$ | $Z$ |
| $\|11\rangle$ | $\alpha\|1\rangle - \beta\|0\rangle$ | $ZX$ |

### 2.6 4단계: 고전적 통신과 정정

앨리스는 **고전적 채널**을 통해 2비트 측정 결과를 밥에게 전송한다. 이 비트들에 기반하여 밥은 해당하는 파울리 정정을 적용한다:

| 고전적 비트 | 정정 |
|---------------|-----------|
| 00 | $I$ (아무것도 안 함) |
| 01 | $X$ (비트 반전) |
| 10 | $Z$ (위상 반전) |
| 11 | $ZX = iY$ (둘 다) |

정정 후, 밥의 큐비트는 정확히 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$이 된다. 텔레포테이션이 완료되었다.

### 2.7 앨리스의 원본에 무슨 일이 일어났는가?

앨리스의 측정으로 그녀의 큐비트 $A$가 $|0\rangle$ 또는 $|1\rangle$로 붕괴되었다 — 원래 상태 $|\psi\rangle$가 파괴되었다. 정보가 복사된 것이 아니라 전송되었다. 복제 불가가 만족된다.

### 2.8 사용된 자원

- **얽힘 쌍 1개** (사전에 공유, 프로토콜 중 소모됨)
- **고전적 비트 2개** (앨리스에서 밥으로 전송됨)
- **큐비트 1개** 텔레포테이션됨

이 자원 계산은 타이트하다: 양자 텔레포테이션은 1개의 큐비트를 텔레포테이션하기 위해 정확히 1 에비트(Ebit, 얽힘 비트)와 2 씨비트(Cbit, 고전적 비트)를 필요로 한다.

---

## 3. 텔레포테이션이 비신호 원리를 위반하지 않는 이유

### 3.1 우려

얽힘은 광년 떨어져 있을 수 있는 앨리스와 밥 사이에 공유된다. 앨리스가 큐비트를 측정하면, 양자역학에 관한 한 밥의 큐비트가 즉각적으로 변한다. 이것이 빛보다 빠른 통신을 허용하는가?

### 3.2 해결

**아니다.** 앨리스가 고전적 비트를 보내기 전에, 밥의 큐비트는 최대 혼합 상태(Maximally Mixed State)에 있다 — 완전히 무작위하고 어떤 정보도 없다:

$$\rho_B = \text{Tr}_{AE_A}(|\Psi_2\rangle\langle\Psi_2|) = \frac{1}{2}I$$

$|\psi\rangle$가 무엇이든 관계없이, 밥은 완전히 무작위한 큐비트를 본다. 앨리스의 고전적 비트 없이는 $\alpha$나 $\beta$에 대한 어떤 정보도 추출할 수 없다.

고전적 비트는 필수적이며 빛보다 빨리 이동할 수 없다. 양자 상태는 오직 고전적 정정을 받은 후에만 복원되며, 이는 빛의 속도에 의해 제한된다.

### 3.3 정보 계산

- **고전적 비트 이전**: 밥은 $|\psi\rangle$에 대해 0비트의 정보를 가진다
- **고전적 비트 이후**: 밥은 $|\psi\rangle$에 대해 완전한 정보를 가진다
- **고전적 비트만으로**: 2개의 고전적 비트는 임의의 큐비트 상태를 기술할 수 없다 (연속 매개변수 $\alpha, \beta$ 필요)

텔레포테이션 프로토콜은 사전 공유된 얽힘을 "양자 채널"로 활용하며, 2개의 고전적 비트로 활성화되면 완전한 연속 양자 정보를 전송한다. 얽힘도 고전적 비트도 단독으로는 충분하지 않다.

---

## 4. 초밀 부호화

### 4.1 텔레포테이션의 쌍대

초밀 부호화(Superdense Coding)는 텔레포테이션의 "쌍대(Dual)"이다: 텔레포테이션이 2개의 고전적 비트 + 1 에비트를 사용하여 1개의 큐비트를 전송하는 반면, 초밀 부호화는 1개의 큐비트 + 1 에비트를 사용하여 2개의 고전적 비트를 전송한다.

### 4.2 프로토콜

**설정**: 앨리스와 밥이 벨 상태 $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$를 공유한다. 앨리스는 첫 번째 큐비트를, 밥은 두 번째 큐비트를 가진다.

**1단계**: 앨리스는 밥에게 2개의 고전적 비트를 전송하려 한다. 그녀는 자신의 큐비트에 다음 네 가지 연산 중 하나를 적용한다:

| 메시지 | 앨리스의 연산 | 결과 상태 |
|---------|-------------------|-----------------|
| 00 | $I$ | $\|\Phi^+\rangle = \frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ |
| 01 | $X$ | $\|\Psi^+\rangle = \frac{1}{\sqrt{2}}(\|10\rangle + \|01\rangle)$ |
| 10 | $Z$ | $\|\Phi^-\rangle = \frac{1}{\sqrt{2}}(\|00\rangle - \|11\rangle)$ |
| 11 | $ZX$ | $\|\Psi^-\rangle = \frac{1}{\sqrt{2}}(\|10\rangle - \|01\rangle)$ |

**2단계**: 앨리스는 자신의 큐비트를 밥에게 전송한다 (큐비트 1개 전송됨).

**3단계**: 밥은 이제 두 큐비트를 모두 가진다. 벨 측정을 수행한다 (첫 번째 큐비트에 CNOT 후 H, 그런 다음 둘 다 측정). 네 개의 벨 상태는 직교하므로, 밥은 완벽하게 구별하여 2비트 메시지를 복원한다.

### 4.3 자원 비교

| 프로토콜 | 전송된 양자 | 전송된 고전적 | 얽힘 | 전송된 정보 |
|----------|-------------|---------------|-------------|----------------------|
| 텔레포테이션 | 0 큐비트 | 2 비트 | 1 에비트 | 1 큐비트 |
| 초밀 부호화 | 1 큐비트 | 0 비트 | 1 에비트 | 고전적 2 비트 |

두 프로토콜은 쌍대적이다: 양자와 고전적 자원을 상호 보완적인 방식으로 교환한다.

---

## 5. 양자 키 분배: BB84

### 5.1 키 분배 문제

앨리스와 밥은 안전하지 않은 채널을 통해 공유 비밀 키를 구축하려 한다. 고전적 키 분배는 사전 공유된 비밀이 필요하거나(대칭 암호화) 계산 난이도 가정에 의존한다(RSA, Diffie-Hellman). 양자 키 분배(QKD, Quantum Key Distribution)는 **정보 이론적 보안(Information-Theoretic Security)**을 제공한다 — 계산 가정이 아닌 물리 법칙에 의해 보장되는 보안.

### 5.2 BB84 프로토콜

1984년 Bennett과 Brassard가 제안했다:

**1단계: 앨리스가 큐비트를 준비하고 전송**

키의 각 비트에 대해 앨리스는 무작위로 선택한다:
- 비트 값 $b \in \{0, 1\}$
- 기저: $Z$ (계산 기저) 또는 $X$ (아다마르 기저)

큐비트를 준비한다:

| 비트 | 기저 | 상태 |
|-----|-------|-------|
| 0 | Z | $\|0\rangle$ |
| 1 | Z | $\|1\rangle$ |
| 0 | X | $\|+\rangle$ |
| 1 | X | $\|-\rangle$ |

**2단계: 밥이 측정**

수신된 각 큐비트에 대해 밥은 무작위로 측정 기저($Z$ 또는 $X$)를 선택하고 측정한다.

- 밥이 앨리스와 **같은 기저**를 선택하면: 확실하게 앨리스의 비트 값을 얻는다
- 밥이 **다른 기저**를 선택하면: 무작위 결과를 얻는다 (50/50)

**3단계: 기저 조정(공개 채널)**

앨리스와 밥은 자신들의 기저 선택을 공개적으로 발표한다(비트 값은 아님). 다른 기저를 사용한 모든 비트를 버린다. 남은 비트(전체의 약 50%)가 **체질된 키(Sifted Key)**를 형성한다.

**4단계: 도청자 탐지**

앨리스와 밥은 체질된 키의 무작위 부분 집합을 희생하여 값을 공개적으로 비교한다. 도청자(이브, Eve)가 큐비트를 가로채고 측정했다면, 오류를 도입했을 것이다(복제 불가 정리와 측정 방해 때문에).

- **도청자 없음**: 비교에서 오류율 0%
- **도청자 있음**: 약 25%의 오류율 (이브가 50%의 시간에 잘못된 기저를 추측하고, 잘못된 기저 측정은 50%의 확률로 잘못된 값을 줌)

오류율이 임계값 아래라면, 남은 체질된 키를 계속 사용한다(프라이버시 증폭과 함께). 오류율이 너무 높으면 중단한다 — 채널이 손상되었다.

### 5.3 보안 근거

BB84 보안은 두 양자역학 원리에 기반한다:

1. **복제 불가 정리**: 이브는 두 기저에서 측정하기 위해 큐비트를 복사할 수 없다
2. **측정 방해**: 잘못된 기저에서 큐비트를 측정하면 비가역적으로 상태가 변해 탐지 가능한 오류가 도입된다

양자 컴퓨터조차 BB84를 깰 수 없다(RSA와 달리). QKD는 계산 가정과 무관한 정보 이론적 보안을 제공한다.

### 5.4 실용적 한계

- **거리**: 광섬유에서의 광자 손실로 중계기 없이 ~100 km로 범위 제한
- **키 레이트**: 일반적으로 킬로비트/초로, 대량 암호화에는 너무 느림(키 교환에만 사용)
- **사이드 채널**: 실제 구현은 하드웨어 취약점을 가질 수 있음(검출기 눈멀기, 트로이 목마 공격)
- **인증**: BB84는 인증된 고전적 채널이 필요함(그 자체로 일부 초기 공유 비밀이 필요)

---

## 6. 양자 중계기

### 6.1 거리 문제

장거리 양자 통신은 광자 손실 문제에 직면한다: 광섬유에서 약 0.2 dB/km 손실로, 100 km 후에는 광자의 ~1%만 도착한다. 고전적 통신은 증폭기(중계기)로 이를 해결하지만, 복제 불가 정리는 단순히 양자 신호를 증폭하는 것을 막는다.

### 6.2 얽힘 교환

양자 중계기의 핵심 아이디어는 **얽힘 교환(Entanglement Swapping)**이다:

1. 앨리스와 중간 노드(찰리, Charlie) 사이에 얽힘 쌍 생성
2. 찰리와 밥 사이에 또 다른 얽힘 쌍 생성
3. 찰리가 자신의 두 큐비트에 벨 측정을 수행
4. 이것이 얽힘을 "교환"한다: 앨리스와 밥이 직접 상호작용하지 않았음에도 이제 얽혀있다

```
앨리스 ~~~~ 찰리 ~~~~ 밥
  A-C 얽힘    C-B 얽힘

찰리가 벨 측정 수행:

앨리스 ~~~~~~~~~~~~~ 밥
    이제 직접 얽혀있다!
```

### 6.3 얽힘 정제

불완전한 얽힘 쌍은 국소 연산과 고전적 통신(LOCC, Local Operations and Classical Communication)을 사용하여 더 적지만 더 높은 품질의 쌍으로 "정제(Distilled)"될 수 있다. 여러 노이즈 있는 벨 쌍이 더 적은 수의 거의 완벽한 벨 쌍으로 변환된다.

### 6.4 양자 중계기 아키텍처

실용적인 양자 중계기 체인은 다음과 같이 작동한다:

1. 총 거리를 세그먼트로 분할
2. 각 세그먼트 내에서 얽힘 생성 (단거리, 관리 가능한 손실)
3. 얽힘 교환을 사용하여 세그먼트에 걸쳐 얽힘 확장
4. 충실도(Fidelity) 유지를 위해 정제 사용
5. 종단 간 얽힘이 확립될 때까지 반복

이 아키텍처는 원칙적으로 수천 킬로미터에 걸친 양자 통신을 가능하게 한다. 연구 프로토타입은 미치우스(Micius) 위성(2017)을 통해 ~1,200 km에 걸친 얽힘을 시연했다.

---

## 7. Python 구현

### 7.1 양자 텔레포테이션

```python
import numpy as np

def teleportation_simulation(alpha, beta, verbose=True):
    """Simulate the quantum teleportation protocol.

    Why simulate the full 3-qubit system? This demonstrates how the
    entanglement between Alice's and Bob's qubits enables the transfer
    of an unknown state through measurement and classical communication.
    """
    # Validate input: |α|² + |β|² should equal 1
    assert abs(abs(alpha)**2 + abs(beta)**2 - 1) < 1e-10, "State not normalized"

    if verbose:
        print("=" * 55)
        print("Quantum Teleportation Protocol")
        print("=" * 55)
        print(f"\nState to teleport: |ψ⟩ = ({alpha:.4f})|0⟩ + ({beta:.4f})|1⟩")

    # === Step 0: Initial state ===
    # |ψ⟩_A ⊗ |Φ+⟩_{E_A E_B}
    # 3 qubits: A, E_A, E_B in that order
    # Basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩

    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 on qubits E_A, E_B
    bell = np.zeros(4, dtype=complex)
    bell[0b00] = 1/np.sqrt(2)  # |00⟩
    bell[0b11] = 1/np.sqrt(2)  # |11⟩

    # Alice's qubit
    psi = np.array([alpha, beta], dtype=complex)

    # Full 3-qubit state: |ψ⟩ ⊗ |Φ+⟩
    state = np.kron(psi, bell)
    if verbose:
        print(f"\nInitial 3-qubit state:")
        _print_state(state, 3)

    # === Step 1: CNOT (A→E_A) ===
    # CNOT flips E_A when A=1
    CNOT = np.eye(8, dtype=complex)
    # Swap |100⟩↔|110⟩ and |101⟩↔|111⟩
    for i in range(8):
        if (i >> 2) & 1:  # If qubit A is |1⟩
            j = i ^ (1 << 1)  # Flip qubit E_A
            CNOT[i, i] = 0
            CNOT[j, j] = 0
            CNOT[i, j] = 1
            CNOT[j, i] = 1

    state = CNOT @ state
    if verbose:
        print("\nAfter CNOT (A controls E_A):")
        _print_state(state, 3)

    # === Step 2: Hadamard on A ===
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I2 = np.eye(2)
    H_full = np.kron(np.kron(H, I2), I2)  # H on qubit A only

    state = H_full @ state
    if verbose:
        print("\nAfter Hadamard on A:")
        _print_state(state, 3)

    # === Step 3: Alice measures qubits A and E_A ===
    # Compute probabilities for each measurement outcome
    probs = {}
    bob_states = {}
    for m in range(4):  # Alice's 2 qubits: 00, 01, 10, 11
        # Project onto |m⟩ on qubits A and E_A
        bob_state = np.zeros(2, dtype=complex)
        for b in range(2):  # Bob's qubit
            idx = (m << 1) | b  # Combine Alice's bits with Bob's bit
            bob_state[b] = state[idx]
        prob = np.sum(np.abs(bob_state)**2)
        if prob > 1e-10:
            probs[m] = prob
            bob_states[m] = bob_state / np.sqrt(prob)

    # Simulate measurement (random outcome)
    outcomes = list(probs.keys())
    probabilities = [probs[m] for m in outcomes]
    measurement = outcomes[np.random.choice(len(outcomes), p=probabilities)]

    if verbose:
        print(f"\nAlice measures: |{measurement:02b}⟩ (prob = {probs[measurement]:.4f})")
        print(f"Bob's qubit before correction: {bob_states[measurement].round(4)}")

    # === Step 4: Bob applies correction ===
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    corrections = {
        0b00: np.eye(2, dtype=complex),
        0b01: X,
        0b10: Z,
        0b11: Z @ X,
    }
    correction_names = {0b00: "I", 0b01: "X", 0b10: "Z", 0b11: "ZX"}

    bob_final = corrections[measurement] @ bob_states[measurement]

    if verbose:
        print(f"\nClassical bits sent: {measurement >> 1}, {measurement & 1}")
        print(f"Bob applies correction: {correction_names[measurement]}")
        print(f"Bob's final state: ({bob_final[0]:.4f})|0⟩ + ({bob_final[1]:.4f})|1⟩")

    # Verify
    fidelity = abs(np.dot(psi.conj(), bob_final))**2
    if verbose:
        print(f"\nFidelity with original: {fidelity:.10f}")
        print("Teleportation " + ("SUCCESS" if fidelity > 0.999 else "FAILED"))

    return fidelity

def _print_state(state, n_qubits):
    """Print a quantum state with labeled basis states."""
    for i in range(len(state)):
        if abs(state[i]) > 1e-10:
            label = f"|{i:0{n_qubits}b}⟩"
            print(f"  {state[i]:+.4f} {label}")

# Test with several states
print("Test 1: |ψ⟩ = |0⟩")
teleportation_simulation(1, 0)

print("\n\nTest 2: |ψ⟩ = |+⟩")
teleportation_simulation(1/np.sqrt(2), 1/np.sqrt(2))

print("\n\nTest 3: |ψ⟩ = (1+2i)|0⟩/√5 + (0)|1⟩... arbitrary state")
alpha = (1 + 2j) / np.sqrt(5 + 4)
beta = 2 / np.sqrt(5 + 4)
# Normalize
norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
teleportation_simulation(alpha/norm, beta/norm)
```

### 7.2 초밀 부호화

```python
import numpy as np

def superdense_coding(message_bits):
    """Simulate the superdense coding protocol.

    Why superdense coding? It demonstrates the "dual" of teleportation:
    1 qubit + 1 ebit can carry 2 classical bits. This exceeds the Holevo
    bound for a single qubit (which can carry at most 1 classical bit
    without entanglement).

    Args:
        message_bits: tuple (b1, b2) where each is 0 or 1
    """
    b1, b2 = message_bits
    print(f"Message to send: ({b1}, {b2})")

    # Shared Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    state = np.zeros(4, dtype=complex)
    state[0b00] = 1/np.sqrt(2)
    state[0b11] = 1/np.sqrt(2)
    print(f"Shared Bell state: (|00⟩ + |11⟩)/√2")

    # Alice's encoding: apply operation to her qubit (qubit 0)
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Alice's operation depends on the message
    # 00 → I, 01 → X, 10 → Z, 11 → ZX
    if (b1, b2) == (0, 0):
        alice_op = I
        op_name = "I"
    elif (b1, b2) == (0, 1):
        alice_op = X
        op_name = "X"
    elif (b1, b2) == (1, 0):
        alice_op = Z
        op_name = "Z"
    else:  # (1, 1)
        alice_op = Z @ X
        op_name = "ZX"

    # Apply Alice's operation to qubit 0
    full_op = np.kron(alice_op, I)
    state = full_op @ state
    print(f"Alice applies {op_name} to her qubit")

    # Alice sends her qubit to Bob (Bob now has both qubits)

    # Bob's decoding: CNOT then Hadamard, then measure
    # CNOT (qubit 0 controls qubit 1)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    state = CNOT @ state

    # Hadamard on qubit 0
    H = np.array([[1,1],[1,-1]]) / np.sqrt(2)
    HI = np.kron(H, I)
    state = HI @ state

    # Measure
    probs = np.abs(state)**2
    result = np.argmax(probs)
    decoded_b1 = (result >> 1) & 1
    decoded_b2 = result & 1

    print(f"Bob measures: |{result:02b}⟩ with probability {probs[result]:.4f}")
    print(f"Decoded message: ({decoded_b1}, {decoded_b2})")
    print(f"Correct: {'YES' if (decoded_b1, decoded_b2) == (b1, b2) else 'NO'}")
    return (decoded_b1, decoded_b2)

print("=" * 50)
print("Superdense Coding Protocol")
print("=" * 50)

for msg in [(0,0), (0,1), (1,0), (1,1)]:
    print()
    superdense_coding(msg)
```

### 7.3 BB84 양자 키 분배

```python
import numpy as np

def bb84_simulation(n_bits=100, eve_present=False, verbose=True):
    """Simulate the BB84 quantum key distribution protocol.

    Why simulate BB84? This demonstrates how quantum mechanics enables
    information-theoretically secure key exchange. The no-cloning theorem
    prevents an eavesdropper from copying qubits, and measurement in the
    wrong basis introduces detectable errors.
    """
    np.random.seed(42)

    # === Alice's preparation ===
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0=Z basis, 1=X basis

    # Prepare quantum states
    # Z basis: 0→|0⟩, 1→|1⟩
    # X basis: 0→|+⟩, 1→|−⟩
    states = []
    for i in range(n_bits):
        if alice_bases[i] == 0:  # Z basis
            if alice_bits[i] == 0:
                states.append(np.array([1, 0], dtype=complex))
            else:
                states.append(np.array([0, 1], dtype=complex))
        else:  # X basis
            if alice_bits[i] == 0:
                states.append(np.array([1, 1], dtype=complex) / np.sqrt(2))
            else:
                states.append(np.array([1, -1], dtype=complex) / np.sqrt(2))

    # === Eve's interception (if present) ===
    eve_bases = None
    if eve_present:
        eve_bases = np.random.randint(0, 2, n_bits)
        for i in range(n_bits):
            # Eve measures in her randomly chosen basis
            if eve_bases[i] == 0:  # Z basis measurement
                prob_0 = abs(states[i][0])**2
                result = 0 if np.random.random() < prob_0 else 1
                # After measurement, state collapses
                if result == 0:
                    states[i] = np.array([1, 0], dtype=complex)
                else:
                    states[i] = np.array([0, 1], dtype=complex)
            else:  # X basis measurement
                # Project onto |+⟩ and |−⟩
                plus = np.array([1, 1]) / np.sqrt(2)
                prob_plus = abs(np.dot(plus, states[i]))**2
                result = 0 if np.random.random() < prob_plus else 1
                if result == 0:
                    states[i] = np.array([1, 1], dtype=complex) / np.sqrt(2)
                else:
                    states[i] = np.array([1, -1], dtype=complex) / np.sqrt(2)

    # === Bob's measurement ===
    bob_bases = np.random.randint(0, 2, n_bits)
    bob_bits = np.zeros(n_bits, dtype=int)

    for i in range(n_bits):
        if bob_bases[i] == 0:  # Z basis measurement
            prob_0 = abs(states[i][0])**2
            bob_bits[i] = 0 if np.random.random() < prob_0 else 1
        else:  # X basis measurement
            plus = np.array([1, 1]) / np.sqrt(2)
            prob_plus = abs(np.dot(plus, states[i]))**2
            bob_bits[i] = 0 if np.random.random() < prob_plus else 1

    # === Basis reconciliation ===
    matching_bases = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]
    n_sifted = len(sifted_alice)

    # === Error estimation ===
    # Use first half for error check, second half as key
    n_check = n_sifted // 2
    check_alice = sifted_alice[:n_check]
    check_bob = sifted_bob[:n_check]
    errors = np.sum(check_alice != check_bob)
    error_rate = errors / n_check if n_check > 0 else 0

    key_alice = sifted_alice[n_check:]
    key_bob = sifted_bob[n_check:]
    key_match = np.all(key_alice == key_bob)

    if verbose:
        print("=" * 55)
        print(f"BB84 Protocol (Eve {'present' if eve_present else 'absent'})")
        print("=" * 55)
        print(f"  Qubits sent: {n_bits}")
        print(f"  Matching bases: {n_sifted} ({100*n_sifted/n_bits:.0f}%)")
        print(f"  Check bits used: {n_check}")
        print(f"  Errors in check: {errors}/{n_check} = {100*error_rate:.1f}%")
        print(f"  Key length: {len(key_alice)} bits")
        print(f"  Keys match: {key_match}")

        if error_rate > 0.11:
            print("  ALERT: Error rate too high — eavesdropper likely detected!")
            print("  Protocol ABORTED.")
        else:
            print("  Key accepted — channel appears secure.")

    return error_rate, key_match

# Run BB84 without eavesdropper
print("\n--- Without Eavesdropper ---")
bb84_simulation(n_bits=200, eve_present=False)

# Run BB84 with eavesdropper
print("\n--- With Eavesdropper ---")
bb84_simulation(n_bits=200, eve_present=True)
```

### 7.4 얽힘 교환

```python
import numpy as np

def entanglement_swapping():
    """Simulate entanglement swapping for quantum repeaters.

    Why entanglement swapping? It's the quantum analog of a relay: it extends
    entanglement across distances that are too far for direct transmission.
    Two short-distance entangled pairs are "joined" into one long-distance pair
    via a Bell measurement at the intermediate node.
    """
    print("=" * 55)
    print("Entanglement Swapping (Quantum Repeater Building Block)")
    print("=" * 55)

    # 4 qubits: A, C1, C2, B
    # A-C1 entangled (Bell pair 1): |Φ+⟩_{A,C1}
    # C2-B entangled (Bell pair 2): |Φ+⟩_{C2,B}
    # Charlie holds C1 and C2

    # Build initial state: |Φ+⟩_{A,C1} ⊗ |Φ+⟩_{C2,B}
    bell = np.zeros(4, dtype=complex)
    bell[0b00] = 1/np.sqrt(2)
    bell[0b11] = 1/np.sqrt(2)

    state = np.kron(bell, bell)  # 16-dimensional (4 qubits)
    print("\nInitial state: |Φ+⟩_{A,C1} ⊗ |Φ+⟩_{C2,B}")

    # Charlie performs Bell measurement on C1, C2 (qubits 1 and 2)
    # First: CNOT with C1 as control, C2 as target
    CNOT_C1C2 = np.eye(16, dtype=complex)
    for i in range(16):
        c1 = (i >> 2) & 1  # qubit 1
        c2 = (i >> 1) & 1  # qubit 2
        if c1 == 1:
            j = i ^ (1 << 1)  # flip qubit 2
            CNOT_C1C2[i, :] = 0
            CNOT_C1C2[i, j] = 1
        # else: identity (already set)

    state = CNOT_C1C2 @ state

    # Hadamard on C1 (qubit 1)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I2 = np.eye(2)
    H_C1 = np.kron(np.kron(I2, H), np.kron(I2, I2))
    state = H_C1 @ state

    # Measure C1 and C2
    # For each measurement outcome (c1, c2), find Alice-Bob state
    print("\nCharlie measures C1 and C2:")
    for c1 in range(2):
        for c2 in range(2):
            # Extract Alice-Bob state for this measurement
            ab_state = np.zeros(4, dtype=complex)
            for a in range(2):
                for b in range(2):
                    idx = (a << 3) | (c1 << 2) | (c2 << 1) | b
                    ab_idx = (a << 1) | b
                    ab_state[ab_idx] = state[idx]

            prob = np.sum(np.abs(ab_state)**2)
            if prob > 1e-10:
                ab_state /= np.sqrt(prob)
                print(f"\n  Charlie gets |{c1}{c2}⟩ (prob={prob:.4f}):")
                print(f"  Alice-Bob state:")
                for i in range(4):
                    if abs(ab_state[i]) > 1e-10:
                        print(f"    {ab_state[i]:+.4f} |{i:02b}⟩")

                # Check if it's a Bell state
                bell_states = {
                    'Φ+': np.array([1,0,0,1])/np.sqrt(2),
                    'Φ-': np.array([1,0,0,-1])/np.sqrt(2),
                    'Ψ+': np.array([0,1,1,0])/np.sqrt(2),
                    'Ψ-': np.array([0,1,-1,0])/np.sqrt(2),
                }
                for name, bs in bell_states.items():
                    if abs(abs(np.dot(bs.conj(), ab_state)) - 1) < 1e-10:
                        print(f"  → This is |{name}⟩! Alice and Bob are now entangled!")

    print("\nResult: Regardless of Charlie's measurement outcome,")
    print("Alice and Bob end up sharing a Bell state (up to known Pauli correction).")
    print("Entanglement has been 'swapped' from A-C1 and C2-B to A-B!")

entanglement_swapping()
```

---

## 8. 연습 문제

### 연습 1: 다른 벨 상태를 사용한 텔레포테이션

$|\Phi^+\rangle$ 대신 $|\Psi^-\rangle = (|01\rangle - |10\rangle)/\sqrt{2}$를 공유된 얽힘 쌍으로 사용하여 텔레포테이션 프로토콜을 반복하라.
(a) 대수를 단계별로 계산하라.
(b) 정정이 어떻게 바뀌는가?
(c) Python 시뮬레이션을 사용하여 검증하라.

### 연습 2: 노이즈가 있는 텔레포테이션 충실도

공유된 벨 쌍이 불완전하다고 가정하자 — 워너(Werner) 상태:

$$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$$

(a) $p$의 어떤 값에서 텔레포테이션 충실도가 $2/3$ (고전적 한계)를 초과할 수 있는가?
(b) $p = 0.5, 0.7, 0.9, 1.0$에 대한 노이즈 있는 벨 쌍으로 텔레포테이션을 시뮬레이션하라. 충실도 대 $p$를 그려라.
(c) $p < 1/3$에서 프로토콜이 무용함(충실도 $\leq 1/2$)을 보여라.

### 연습 3: 초밀 부호화 용량

(a) 공유된 얽힘 없이 단일 큐비트는 최대 1개의 고전적 비트를 전달할 수 있음을 증명하라(홀레보 한계, Holevo Bound).
(b) 1 에비트의 공유 얽힘이 있으면 1개의 큐비트가 정확히 2개의 고전적 비트를 전달할 수 있음을 보여라(초밀 부호화).
(c) 앨리스와 밥이 최대 얽힘이 아닌 상태 $\cos\theta|00\rangle + \sin\theta|11\rangle$를 공유한다면? $\theta$에 따라 용량이 어떻게 변하는가?

### 연습 4: BB84 보안 분석

이브가 있는 경우와 없는 경우 각각 BB84 시뮬레이션을 1000번 실행하라:
(a) 두 경우 모두에 대한 오류율의 히스토그램을 그려라.
(b) 두 분포를 가장 잘 분리하는 임계 오류율은 무엇인가?
(c) 선택한 임계값에 대해 이브가 존재하지만 감지되지 않을 확률(위음성, False Negative)을 계산하라.
(d) 체크 비트의 수에 따라 감지 확률이 어떻게 변하는가?

### 연습 5: 양자 중계기 체인

얽힘 교환 시뮬레이션을 3-노드 체인(앨리스-찰리1-찰리2-밥)으로 확장하라:
(a) 3개의 벨 쌍으로 시작: A-C1, C1'-C2, C2'-B.
(b) 두 중간 노드에서 얽힘 교환을 수행하라.
(c) 앨리스와 밥이 결국 벨 상태를 공유함을 검증하라.
(d) 각 링크의 충실도가 $F$라면, 2번의 교환 후 종단 간 충실도는 무엇인가? $n$번의 교환 후는?

---

[← 이전: 양자 오류 정정](11_Quantum_Error_Correction.md) | [다음: 변분 양자 고유값 해법 →](13_VQE.md)
