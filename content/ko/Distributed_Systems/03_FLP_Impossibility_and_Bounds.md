# 레슨 3: FLP 불가능성과 이론적 한계

[개요](./00_Overview.md) | [이전](./02_Time_Clocks_and_Ordering.md) | [다음](./04_Consistency_Models.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. 합의 문제를 세 가지 속성(agreement, validity, termination)으로 형식적으로 정의하고 각각이 필요한 이유를 설명
2. Bivalent 구성, 핵심 보조 정리, 모순 논증을 포함한 FLP 불가능성 증명을 단계별로 따라가기
3. 합의의 라운드 및 메시지 복잡도 하한을 명시하고 Dolev-Reischuk 한계를 설명
4. FLP를 우회하는 다섯 가지 전략(무작위화, 부분 동기성, 장애 감지기, 타임아웃, 약화된 문제 정의)을 설명
5. Paxos와 Raft 같은 실용적 합의 프로토콜이 FLP 불가능성 결과에도 불구하고 동작하는 이유를 설명

---

## 목차

1. [합의 문제](#1-합의-문제)
2. [준비: 정의와 표기법](#2-준비-정의와-표기법)
3. [FLP 불가능성 정리](#3-flp-불가능성-정리)
4. [상세 증명 스케치](#4-상세-증명-스케치)
5. [합의의 하한](#5-합의의-하한)
6. [정지 문제와의 관계](#6-정지-문제와의-관계)
7. [FLP 우회](#7-flp-우회)
8. [장애 감지기: Chandra-Toueg 프레임워크](#8-장애-감지기-chandra-toueg-프레임워크)
9. [실용적 함의](#9-실용적-함의)
10. [코드: FLP 스타일 적대자 시뮬레이션](#10-코드-flp-스타일-적대자-시뮬레이션)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고 문헌](#13-참고-문헌)

---

## 1. 합의 문제

### 1.1 비형식적 설명

합의 문제는 기만적으로 단순합니다: 프로세스 그룹이 단일 값에 **합의**해야 합니다. 각 프로세스가 값을 제안하고, 모두 제안된 값 중 하나에 대해 결정해야 합니다.

이것은 다음의 기반입니다:
- Leader 선출 (누가 leader인지 합의)
- Atomic broadcast (메시지 순서에 합의)
- 분산 트랜잭션 (commit 또는 abort에 합의)
- 상태 머신 복제 (다음 명령에 합의)
- 블록체인 (다음 블록에 합의)

### 1.2 형식적 정의

`n`개 프로세스 `p₁, p₂, ..., pₙ`에 대한 합의 프로토콜은 세 가지 속성을 만족합니다:

**속성 1 -- Agreement**: 두 올바른 프로세스가 다른 값을 결정하지 않습니다.

```
∀ correct processes pᵢ, pⱼ:
  decide(pᵢ) = decide(pⱼ)
```

**속성 2 -- Validity (비자명성)**: 결정된 값은 어떤 프로세스가 제안한 것이어야 합니다.

```
∃ process pₖ:
  decide(pᵢ) = propose(pₖ)
```

Validity가 없으면, 자명한 프로토콜이 제안과 관계없이 항상 고정 값(예: 0)을 결정할 수 있습니다. Validity는 프로토콜이 실제로 입력을 처리하도록 보장합니다.

**속성 3 -- Termination**: 모든 올바른 프로세스가 결국 결정합니다.

```
∀ correct processes pᵢ:
  eventually ∃ v: decide(pᵢ) = v
```

참고: **Agreement**와 **validity**는 safety 속성입니다. **Termination**은 liveness 속성입니다.

### 1.3 합의의 변형

| 변형 | Agreement | Validity | Termination |
|------|-----------|----------|-------------|
| Uniform consensus | 결정하는 모든 프로세스(장애 포함)가 같은 값 결정 | 어떤 제안이든 | 모든 올바른 프로세스가 결정 |
| Binary consensus | 위와 동일 | 결정 ∈ {0, 1} | 모든 올바른 프로세스가 결정 |
| Multi-valued consensus | 동일 | 어떤 제안이든 | 모든 올바른 프로세스가 결정 |
| Interactive consistency | 각 프로세스가 모든 제안을 학습 | 모든 제안이 수집됨 | 모든 올바른 프로세스가 결정 |

FLP 결과는 가장 단순한 변형인 **binary consensus**에 적용됩니다. 어떤 합의 프로토콜도 binary consensus를 해결할 수 있으므로(제안을 {0, 1}에 매핑), 불가능성은 모든 변형에 확장됩니다.

### 1.4 합의가 어려운 이유

동기 시스템에서 합의는 간단합니다:

```python
def synchronous_consensus(processes, proposals, max_rounds):
    """
    Simple synchronous consensus: exchange all proposals,
    then pick the minimum. Works because we can detect crashes.
    """
    known_values = {pid: proposals[pid] for pid in processes if processes[pid].alive}

    for round_num in range(max_rounds):
        # Each process broadcasts its known values
        for sender in processes:
            if not processes[sender].alive:
                continue
            for receiver in processes:
                if not processes[receiver].alive:
                    continue
                # In synchronous model, this always arrives within Δ
                send(sender, receiver, known_values)

        # Wait exactly Δ time units (guaranteed delivery)
        wait(DELTA)

        # Merge all received values
        for receiver in processes:
            for msg in receive_all(receiver):
                known_values.update(msg)

    # All correct processes have the same set of values → decide minimum
    return min(known_values.values())
```

동기 모델에서는 `f + 1` 라운드(여기서 `f`는 crash 장애의 최대 수) 후에, 모든 올바른 프로세스가 같은 값 집합을 가지며 결정론적으로 결정할 수 있습니다(예: 최솟값 선택).

**비동기** 모델에서는 이 접근법이 실패합니다:
- `Δ` 시간 단위를 대기할 수 없습니다 (상한이 없음)
- Crash를 감지할 수 없습니다 (느린 것 ≡ 죽은 것)
- 모든 메시지가 도착했는지 알 수 없습니다

---

## 2. 준비: 정의와 표기법

FLP 정리를 정확히 명시하고 증명하려면, 형식적 정의가 필요합니다.

### 2.1 FLP를 위한 시스템 모델

FLP 결과는 다음을 가정합니다:

| 가정 | 명세 |
|------|------|
| 프로세스 수 | n ≥ 2 |
| 타이밍 모델 | 완전 비동기 |
| 장애 모델 | 최대 1개 crash-stop 장애 (f = 1) |
| 링크 모델 | Reliable (모든 메시지가 결국 전달) |
| 프로토콜 | 결정론적 |
| 결정 | Binary (0 또는 1 결정) |

장애 가정이 얼마나 약한지 주목하세요: 단 **하나의** crash이고, 링크는 reliable합니다. 이런 유리한 환경에서도, 결정론적 합의는 불가능합니다.

### 2.2 실행 모델

**구성(Configuration)** `C`는 시스템 상태의 완전한 설명입니다: 모든 프로세스의 내부 상태와 전송 중인 메시지의 집합("메시지 버퍼").

**단계(Step)**는 프로세스 `pᵢ`가 수행하는 원자적 행동입니다:
1. 메시지 버퍼에서 메시지 `m`(또는 특별한 null 메시지 `∅`)을 수신
2. 현재 상태와 `m`에 기반하여 새로운 내부 상태로 전이
3. 다른 프로세스에 0개 이상의 메시지를 전송

프로세스 `pᵢ`가 메시지 `m`을 수신하는 단계를 `e = (pᵢ, m)`으로 표기합니다.

**실행(Execution)**(또는 **run**)은 초기 구성에서 시작하는 (아마도 무한한) 단계의 시퀀스입니다:

```
C₀ →[e₁]→ C₁ →[e₂]→ C₂ →[e₃]→ C₃ → ...
```

**스케줄** `σ`은 단계의 시퀀스입니다. 구성 `C`에 스케줄 `σ`을 적용하여 도달하는 구성을 `σ(C)`로 씁니다.

### 2.3 결정 값과 구성 분류

구성 `C`는:

- **0-decided**: 어떤 프로세스가 `C`에서 0을 결정
- **1-decided**: 어떤 프로세스가 `C`에서 1을 결정
- **0-valent**: `C`에서 시작하는 모든 실행에서 결정(있다면)이 0
- **1-valent**: `C`에서 시작하는 모든 실행에서 결정(있다면)이 1
- **Bivalent**: `C`에서 시작하는 두 실행이 존재, 하나는 0을 결정하고 하나는 1을 결정
- **Univalent**: 0-valent 또는 1-valent (bivalent가 아님)

```
Configuration taxonomy:

                Configuration C
               /                \
          Univalent            Bivalent
         /         \              |
    0-valent    1-valent     Both outcomes
    (only 0)    (only 1)     are possible
```

**핵심 통찰**: Bivalent 구성은 프로토콜이 아직 결정에 "커밋"하지 않은 구성입니다. 적대자(스케줄러)가 실행을 어느 방향으로든 이끌 수 있습니다.

### 2.4 초기 구성

초기 구성은 입력(제안) 벡터 `(v₁, v₂, ..., vₙ)`에 의해 결정됩니다. 여기서 `vᵢ ∈ {0, 1}`은 프로세스 `pᵢ`가 제안한 값입니다. 모든 초기 구성에서 메시지 버퍼는 비어 있습니다.

---

## 3. FLP 불가능성 정리

### 3.1 정리 명제

**정리** (Fischer, Lynch, Paterson, 1985): 단 하나의 프로세스라도 crash될 수 있는 reliable 링크를 가진 비동기 시스템에서, 합의 문제를 해결하는 결정론적 프로토콜은 없습니다.

더 정확히: 비동기 모델에서 `n ≥ 2` 프로세스에 대한 어떤 결정론적 합의 프로토콜 `P`에 대해서도, 최대 하나의 프로세스가 crash되더라도 어떤 올바른 프로세스가 결정하지 못하는(termination이 위반되는) `P`의 실행이 존재합니다.

### 3.2 증명 전략

증명은 적대자(메시지 전달 순서와 프로세스 속도를 제어하는 스케줄러)가 시스템을 **항상** bivalent 구성에 유지시켜 어떤 프로세스도 결정하지 못하게 할 수 있음을 보여줍니다. 증명은 세 가지 주요 구성 요소를 가집니다:

1. **보조 정리 1**: Bivalent 초기 구성이 존재합니다.
2. **보조 정리 2**: 어떤 bivalent 구성에서든, 또 다른 bivalent 구성으로 이끄는 단계가 존재합니다.
3. **정리**: 보조 정리 2를 반복 적용하여, 적대자는 결정하지 않는 무한 실행을 구성합니다(장애 모델과 일관되면서 termination을 위반).

---

## 4. 상세 증명 스케치

### 4.1 보조 정리 1: Bivalent 초기 구성의 존재

**주장**: 어떤 결정론적 합의 프로토콜에 대해서도, bivalent인 초기 구성이 존재합니다.

**모순에 의한 증명**: 모든 초기 구성이 univalent(0-valent 또는 1-valent)라고 가정합니다.

두 특정 초기 구성을 고려합니다:
- `C₀ = (0, 0, ..., 0)`: 모든 프로세스가 0을 제안. Validity에 의해, 결정은 0이어야 합니다. 따라서 `C₀`는 0-valent.
- `C₁ = (1, 1, ..., 1)`: 모든 프로세스가 1을 제안. Validity에 의해, 결정은 1이어야 합니다. 따라서 `C₁`은 1-valent.

정확히 한 위치만 다른 초기 구성의 시퀀스를 고려합니다:

```
C₀ = (0, 0, 0, ..., 0, 0)    → 0-valent
C₁ = (1, 0, 0, ..., 0, 0)    → ?-valent
C₂ = (1, 1, 0, ..., 0, 0)    → ?-valent
...
Cₙ = (1, 1, 1, ..., 1, 1)    → 1-valent
```

각 연속 쌍 `Cₖ`과 `Cₖ₊₁`은 정확히 한 프로세스의 제안(프로세스 `pₖ₊₁`)만 다릅니다.

`C₀`가 0-valent이고 `Cₙ`이 1-valent이므로, 비둘기집 원리에 의해 `Cₖ`(0-valent)과 `Cₖ₊₁`(1-valent)인 인접 쌍이 존재해야 합니다.

`Cₖ`과 `Cₖ₊₁`은 프로세스 `pₖ₊₁`의 제안만 다릅니다. `pₖ₊₁`이 맨 처음에 crash되는(어떤 단계도 수행하기 전에) 실행을 고려합니다. 다른 모든 프로세스의 관점에서, `Cₖ`과 `Cₖ₊₁`은 **동일**합니다 (`pₖ₊₁`이 어떤 메시지도 보내지 않으므로 `pₖ₊₁`의 제안을 볼 수 없습니다).

하지만:
- `Cₖ`(0-valent)에서 모든 실행은 0을 결정합니다.
- `Cₖ₊₁`(1-valent)에서 모든 실행은 1을 결정합니다.

`pₖ₊₁`이 즉시 crash되는 실행은 두 경우 모두 다른 모든 프로세스에게 동일하게 보입니다. 이 실행들은 같은 값을 결정해야 하지만(나머지 프로세스들이 동일한 상태를 보므로), 하나는 0을 결정하고 다른 하나는 1을 결정한다고 했습니다. **모순**.

따라서, 우리의 가정이 틀렸습니다: 모든 초기 구성이 univalent인 것은 아닙니다. 적어도 하나는 bivalent입니다. ∎

```
Visual proof of Lemma 1:

C₀ = (0,0,0)  ← 0-valent (by validity)
C₁ = (1,0,0)  ← ?
C₂ = (1,1,0)  ← ?
C₃ = (1,1,1)  ← 1-valent (by validity)

There must be an adjacent pair where valency changes.
Say C₁ is 0-valent and C₂ is 1-valent.
They differ only in p₂'s proposal.

If p₂ crashes immediately:
  C₁ looks like (1, _, 0)  → decides 0
  C₂ looks like (1, _, 0)  → decides 1
  Same view, different decisions → CONTRADICTION

∴ Some Cₖ must be bivalent.
```

### 4.2 보조 정리 2: Bivalent 구성의 지속

**주장**: `C`를 bivalent 구성, `e = (p, m)`을 적용 가능한 단계라 합시다. `D`를 `e`를 적용하지 않고 `C`에서 도달 가능한 구성의 집합, `E = {e(C') | C' ∈ D ∪ {C}}`를 각각에 `e`를 적용하여 얻은 구성의 집합이라 합시다. 그러면 `E`는 bivalent 구성을 포함합니다.

**모순에 의한 증명**: `E`가 bivalent 구성을 포함하지 않는다고 가정합니다 (`E`의 모든 구성이 univalent).

`C`가 bivalent이므로:
- `C`에서 0-decided 구성으로 이끄는 스케줄이 존재합니다.
- `C`에서 1-decided 구성으로 이끄는 스케줄이 존재합니다.

`E`의 구조에 의해, 0-valent인 구성 `E₀ ∈ E`와 1-valent인 구성 `E₁ ∈ E`가 존재해야 합니다. (모두 같은 valency, 예를 들어 0-valent이면, 적대자가 항상 `e`를 적용하여 0을 강제할 수 있어 `C`의 bivalence에 모순.)

이제 `D ∪ {C}`에서 다음과 같은 두 구성 `C₀, C₁`을 고려합니다:
- `e(C₀)`가 0-valent
- `e(C₁)`가 1-valent
- `C₁ = e'(C₀)` (단일 단계 `e' = (p', m')`) (이웃)

이러한 쌍은 `E`에서 0-valent에서 1-valent 구성으로의 경로를 추적할 수 있고, 어떤 단계에서 valency가 변해야 하므로 존재해야 합니다.

**경우 1**: `p ≠ p'` (두 단계가 서로 다른 프로세스를 포함)

`e = (p, m)`과 `e' = (p', m')`이 서로 다른 프로세스를 포함하므로, **교환 가능**(적용 순서가 상관없음)합니다:

```
     C₀ ──[e']──► C₁
      │              │
     [e]            [e]
      ▼              ▼
   e(C₀)         e(C₁)
  0-valent       1-valent

But also:
     C₀ ──[e]──► e(C₀)
      │              │
     [e']           [e']
      ▼              ▼
   e'(C₀)=C₁    e'(e(C₀))

Since e and e' commute: e(e'(C₀)) = e'(e(C₀))
                        e(C₁)     = e'(e(C₀))

So e(C₁) is both 1-valent AND reachable from the 0-valent e(C₀).
But a 0-valent configuration can only reach 0-decided states.
CONTRADICTION.
```

**경우 2**: `p = p'` (두 단계가 같은 프로세스 `p`를 포함)

`C₀`에서 프로세스 `p`가 crash되는(더 이상 단계를 수행하지 않는) 실행을 고려합니다. `σ`를 `C₀`에서 나머지 `n - 1` 프로세스의 결정 실행이라 합시다. `f = 1` (하나의 crash가 허용)이고 `p`가 crash되었으므로, 이 실행은 종료해야 합니다 (합의 프로토콜의 termination 속성에 의해).

`A = σ(C₀)`를 결정된 구성이라 합시다.

`A`는 `p`가 어떤 단계도 수행하지 않고 `C₀`에서 도달 가능합니다. `e = (p, m)`과 `e' = (p', m') = (p, m')` 모두 `p`를 포함하고, `p`가 `σ`에서 단계를 수행하지 않았으므로, 구성 `A`는 `D`에 있습니다.

따라서, `e(A)`는 `E`에 있고 univalent입니다(우리의 가정에 의해). 하지만 `A`는 결정되었으므로(특정 결정 값을 가짐), `e(A)`는 `A`와 같은 결정 값을 가져야 합니다.

하지만 `A`는 `C₀`에서 도달되었고(`e` 후 0-valent로 이끄는), `A`는 `σ`를 통해 `C₁`과도 관련될 수 있습니다. 이것은 0-valent와 1-valent 분기가 합쳐져야 하므로 모순을 만듭니다. (완전한 형식적 논증은 어떤 스케줄이 확장될 수 있는지 주의 깊게 추적해야 하지만, 본질적인 아이디어는 적대자가 `p`의 단계를 지연시켜 충돌하는 결정을 만들 수 있다는 것입니다.) **모순**. ∎

### 4.3 주 정리: 조합하기

보조 정리 1과 2가 주어지면:

1. Bivalent 초기 구성에서 시작합니다 (보조 정리 1이 하나의 존재를 보장).
2. 적대자가 단계별로 실행을 구성합니다. 각 지점에서, 적대자는 보조 정리 2를 사용하여 구성을 bivalent로 유지하는 단계를 선택합니다.
3. 이 구성은 어떤 프로세스도 결정하지 않는 무한 실행을 생산합니다.
4. 이 실행에서 최대 하나의 프로세스가 crash됩니다(적대자는 어떤 프로세스에 대한 메시지든 결국 전달하는 한 지연시킬 수 있고, 최대 하나의 프로세스를 crash시킬 수 있습니다).
5. 이것은 termination 속성을 위반하여, 프로토콜이 합의를 해결하지 못함을 증명합니다. ∎

```
Construction of the non-deciding execution:

  C₀ (bivalent initial, Lemma 1)
   │
   ▼ [adversary chooses step via Lemma 2]
  C₁ (still bivalent)
   │
   ▼ [adversary chooses step via Lemma 2]
  C₂ (still bivalent)
   │
   ▼ [adversary chooses step via Lemma 2]
  C₃ (still bivalent)
   │
   ▼
  ... (forever bivalent, never decides)
```

### 4.4 공정성 보장

미묘한 점: 적대자는 실행이 **공정**하도록 보장해야 합니다 -- 모든 메시지가 결국 전달되고, 모든 올바른 프로세스가 무한히 많은 단계를 수행합니다. 그렇지 않으면 적대자가 메시지를 전달하지 않음으로써 자명하게 진행을 방지할 수 있습니다.

증명은 구성의 각 단계에서, 적대자가 bivalence를 유지하면서 **가장 오래된** 보류 메시지를 전달하도록 선택할 수 있음을 지적하여 이를 처리합니다(보조 정리 2가 이것이 항상 가능함을 보장). 이것은 구성을 bivalent로 유지하면서 공정성을 보장합니다.

### 4.5 증명 요약

```
┌──────────────────────────────────────────────────────────┐
│                   FLP Proof Structure                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  GIVEN: Any deterministic consensus protocol P           │
│         Asynchronous model, reliable links, f = 1        │
│                                                          │
│  LEMMA 1: ∃ bivalent initial configuration               │
│    Proof: Adjacent configs differing in 1 proposal       │
│           Process crash makes them indistinguishable      │
│           Contradiction if all configs are univalent      │
│                                                          │
│  LEMMA 2: From bivalent C, adversary can maintain        │
│           bivalence after any step                        │
│    Proof: By contradiction on commutativity of steps      │
│           Case 1: different processes → commute           │
│           Case 2: same process → crash argument           │
│                                                          │
│  THEOREM: Adversary constructs infinite fair execution    │
│           that never decides (violates termination)       │
│           ∴ P does not solve consensus                    │
│                                                          │
│  CONCLUSION: No deterministic P solves consensus          │
│              in async model with even 1 crash             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 5. 합의의 하한

### 5.1 라운드 복잡도

**동기** 모델에서도 합의는 여러 라운드가 필요합니다.

**정리** (Dolev and Strong, 1983): `f`개 crash 장애가 있는 동기 모델에서, 어떤 결정론적 합의 프로토콜도 최악의 경우 최소 `f + 1` 라운드가 필요합니다.

| 모델 | 장애 유형 | 장애 수 | 최소 라운드 | 최소 메시지 |
|------|----------|---------|-----------|-----------|
| 동기 | Crash | f | f + 1 | O(n × f) |
| 동기 | Byzantine | f < n/3 | f + 1 | O(n² × f) |
| 부분 동기 | Crash | f < n/2 | 2 (GST 이후) | O(n²) |
| 비동기 | Crash | 1 | ∞ (불가능) | N/A |

**f + 1 라운드의 직관**: 라운드 `k`에서, 프로세스는 `k`개 중개자의 체인에서 무언가를 학습했을 수 있습니다. 라운드 `k`에서 crash되는 프로세스는 일부 프로세스에는 정보를 보냈지만 다른 프로세스에는 보내지 못해 비대칭을 만들 수 있습니다. `f` 라운드 후, 최대 `f`개의 이러한 비대칭이 존재할 수 있습니다. `(f + 1)`번째 라운드가 최종 모호성을 해결합니다. 추가 crash가 발생할 수 없기 때문입니다.

### 5.2 Dolev-Reischuk 한계

**정리** (Dolev and Reischuk, 1985): `f ≥ 1`개 crash 장애를 허용하는 어떤 합의 프로토콜도 어떤 실행에서 최소 `Ω(n²)` 메시지를 보내야 합니다.

이것은 `O(n)` 총 메시지만으로는 합의를 해결할 수 없다는 것을 의미합니다 -- 최소한 이차적으로 많은 메시지가 필요합니다.

**증명 스케치**: 전송된 총 메시지 수가 `n²/4` 미만이면, 두 프로세스 `pᵢ`와 `pⱼ` 사이에서 양방향을 합쳐 `n/2` 미만의 메시지가 전송되는 쌍이 존재합니다. 적대자가 그 메시지의 송신자를 crash시킬 수 있어, `pᵢ`와 `pⱼ`가 다른 결정을 요구하는 두 시나리오를 구분할 수 없게 만듭니다.

### 5.3 하한 요약 표

| 한계 | 모델 | 제한하는 것 | 타이트? |
|------|------|-----------|--------|
| FLP | 비동기, f = 1 | Termination 불가능 | 예 (절대적) |
| f + 1 라운드 | 동기, crash | 최소 라운드 | 예 (DS83) |
| Ω(n²) 메시지 | 모든, crash | 최소 메시지 | 거의 (DR85) |
| n ≥ 3f + 1 | 모든, Byzantine | 최소 프로세스 | 예 (PSL80) |
| n ≥ 2f + 1 | 모든, crash | 최소 프로세스 | 예 |

### 5.4 하한의 실용적 함의

이 한계들은 직접적인 엔지니어링 결과를 가집니다:

```
You CANNOT build a consensus protocol that is simultaneously:
  1. Deterministic
  2. Asynchronous (no timing assumptions)
  3. Fault-tolerant (even f = 1)
  4. Always terminates

You MUST sacrifice at least one:
  - Determinism → randomized protocols (Ben-Or)
  - Full asynchrony → partial synchrony (Paxos, Raft)
  - No fault tolerance → trivial (but useless)
  - Guaranteed termination → may block during partitions (Paxos safety)
```

---

## 6. 정지 문제와의 관계

### 6.1 개념적 연결

FLP 불가능성은 계산 가능성 이론의 결정 불가능 결과와 깊은 연관이 있습니다:

| 개념 | 정지 문제 | FLP 불가능성 |
|------|----------|------------|
| 핵심 질문 | "이 프로그램이 정지하는가?" | "모든 프로세스가 결정하는가?" |
| 답 | 결정 불가능 | 보장 불가능 |
| 기법 | 대각선화 | 적대적 스케줄링 |
| 보여주는 것 | 모든 프로그램에 대해 답할 수 있는 알고리즘 없음 | 모든 스케줄에 대해 종료할 수 있는 프로토콜 없음 |
| 적대자 | 자기 참조 프로그램 | 악의적 스케줄러 |

### 6.2 유비

두 결과 모두 특정 문제가 각각의 모델에서 **본질적으로 결정 불가능**함을 보여줍니다:

```
Halting problem:
  "There is no algorithm that can determine, for an arbitrary program
   and input, whether the program halts."

FLP:
  "There is no deterministic algorithm that can guarantee, for all
   possible message schedules, that all correct processes decide."
```

두 경우 모두, 증명은 어떤 제안된 해결책도 물리치는 적대적 입력(자기 참조 프로그램 / 주의 깊게 선택된 스케줄)을 구성합니다.

### 6.3 핵심 차이

그러나 유비는 불완전합니다:

1. **정지 문제는 계산 가능성에 관한 것**: 어떤 알고리즘도 해결할 수 없습니다. 더 많은 시간이나 자원으로도 불가능합니다.
2. **FLP는 모델에 관한 것**: 모델을 변경하면(타이밍 추가, 무작위화 추가), 합의가 해결 가능해집니다. FLP는 계산 가능성 자체가 아닌, *특정 모델에서* 달성 가능한 것에 대한 명제입니다.

```
Halting problem: ABSOLUTELY impossible (no algorithm exists)
FLP:            CONDITIONALLY impossible (in the async deterministic model)
                → Solvable with randomization, partial synchrony, or oracles
```

---

## 7. FLP 우회

FLP 결과는 순수 비동기 결정론적 모델에서 **할 수 없는** 것을 알려줍니다. 모든 실용적 합의 프로토콜은 FLP 가정 중 하나를 약화시켜 동작합니다.

### 7.1 전략 1: 무작위화 (Ben-Or 프로토콜, 1983)

**아이디어**: 프로세스가 무작위 선택을 하도록 허용하여, 적대자가 구성을 bivalent로 유지하는 능력을 깨뜨립니다.

**Ben-Or 프로토콜** (binary consensus를 위해 단순화):

```python
def ben_or_consensus(process_id, initial_value, processes, max_rounds=100):
    """
    Ben-Or's randomized binary consensus protocol.
    Tolerates f < n/2 crash failures.
    Achieves agreement with probability 1 (in expectation O(2^n) rounds).
    """
    v = initial_value  # my current preferred value (0 or 1)
    n = len(processes)

    for r in range(1, max_rounds + 1):
        # Phase 1: Broadcast current value
        broadcast(process_id, ("PHASE1", r, v))
        phase1_msgs = wait_for_messages("PHASE1", r, threshold=n - 1)

        # Count votes
        count_0 = sum(1 for _, _, val in phase1_msgs if val == 0)
        count_1 = sum(1 for _, _, val in phase1_msgs if val == 1)

        # Phase 2: Broadcast proposal
        if count_0 > n // 2:
            proposal = 0
        elif count_1 > n // 2:
            proposal = 1
        else:
            proposal = None  # "?" -- no majority

        broadcast(process_id, ("PHASE2", r, proposal))
        phase2_msgs = wait_for_messages("PHASE2", r, threshold=n - 1)

        # Decision rule
        non_none = [(_, _, val) for _, _, val in phase2_msgs if val is not None]

        if len(non_none) > 0:
            decided_value = non_none[0][2]  # all non-None proposals are the same
            if len(non_none) > n // 2:
                return decided_value  # DECIDE
            else:
                v = decided_value  # adopt the proposed value
        else:
            # No majority in phase 2: flip a coin!
            v = random.choice([0, 1])  # RANDOMIZATION breaks FLP

    return None  # did not converge (extremely unlikely)
```

**속성**:

| 속성 | 보장 |
|------|------|
| Agreement | 항상 (Phase 2의 결정론적 검사) |
| Validity | 항상 (제안된 값만 승리 가능) |
| Termination | 확률 1로 (하지만 기대 라운드가 O(2^n)) |
| Crash 허용 | f < n/2 |

**동작 이유**: Phase 2의 무작위 코인 플립은 적대자가 어떤 bivalent 구성을 유지할지 예측할 수 없게 만듭니다. 0이 아닌 확률로 모든 프로세스가 같은 값을 플립하여 합의에 도달합니다. 많은 라운드에 걸쳐, 이 확률은 1에 접근합니다.

**실용적 우려**: 기대 O(2^n) 라운드는 실제로는 너무 느립니다. 현대의 무작위화 프로토콜(예: 공통 코인 또는 검증 가능한 랜덤 함수 기반)은 기대 O(1) 라운드를 달성합니다.

### 7.2 전략 2: 부분 동기성 (DLS 프로토콜, 1988)

**아이디어**: 시스템이 최종적으로 동기적이라고 가정합니다(GST 이후). Safety는 항상 유지; liveness는 GST 이후 유지.

Dwork-Lynch-Stockmeyer(DLS) 프로토콜은 부분 동기성을 형식적으로 활용한 최초의 프로토콜이었습니다.

**핵심 통찰**: 프로토콜을 다음과 같이 설계합니다:
- **타이밍이 어떻든** (완전 비동기여도), agreement와 validity가 유지됩니다.
- **GST 이후** (메시지가 Δ 내에 도착할 때), termination이 제한된 라운드 수 내에 보장됩니다.

```
Before GST:
  Messages may be delayed arbitrarily.
  Protocol may not make progress.
  BUT: safety is never violated.
  (Processes may re-propose, retry, but never disagree.)

After GST:
  Messages arrive within Δ.
  Failure detector becomes accurate.
  Protocol reaches consensus within O(f) rounds.
```

이것이 Paxos와 Raft가 사용하는 모델입니다(원래 DLS 용어로 설명되지는 않았지만).

### 7.3 전략 3: 장애 감지기 (Chandra-Toueg, 1996)

**아이디어**: 비동기 모델을 어떤 프로세스가 crash되었는지에 대한 (아마도 불완전한) 정보를 제공하는 오라클(장애 감지기)로 보강합니다.

이것은 섹션 8에서 깊이 다룹니다.

### 7.4 전략 4: 불완전한 장애 감지기로서의 타임아웃

**아이디어**: 프로세스가 crash되었는지 추측하기 위해 타임아웃을 사용합니다. 이것은 실제로 eventually perfect 장애 감지기(◇P)의 구현입니다.

```python
class TimeoutBasedDetector:
    """
    Practical failure detector using adaptive timeouts.
    Implements an approximation of ◇P (eventually perfect).
    """

    def __init__(self, initial_timeout_ms: float = 500):
        self.timeout = initial_timeout_ms
        self.suspected: set[str] = set()
        self.last_heard: dict[str, float] = {}
        self.false_positive_count = 0

    def heartbeat_received(self, node_id: str):
        """Process a heartbeat from a node."""
        now = time.time() * 1000
        if node_id in self.suspected:
            # We suspected this node but it is alive → false positive!
            self.suspected.discard(node_id)
            self.false_positive_count += 1
            # Increase timeout to reduce future false positives
            self.timeout *= 1.5
            print(f"  False positive on {node_id}! Increasing timeout to {self.timeout:.0f}ms")
        self.last_heard[node_id] = now

    def check_timeouts(self) -> set[str]:
        """Check which nodes have timed out."""
        now = time.time() * 1000
        for node_id, last_time in self.last_heard.items():
            if now - last_time > self.timeout:
                if node_id not in self.suspected:
                    self.suspected.add(node_id)
                    print(f"  Suspecting {node_id} (no heartbeat for {now - last_time:.0f}ms)")
        return self.suspected.copy()
```

**이것이 실제로 동작하는 이유**: GST 이후, 타임아웃은 결국 실제 메시지 지연보다 큰 값에서 안정화되어, 감지기가 **최종적으로 정확**해집니다. GST 이전에는 오탐을 생성할 수 있지만, 합의 프로토콜의 safety는 영향을 받지 않습니다.

### 7.5 전략 5: 약화된 문제 정의

**아이디어**: 합의를 해결하는 대신, 비동기 모델에서 달성 가능한 더 약한 문제를 해결합니다.

| 문제 | 비동기에서 달성 가능? | 참고 |
|------|---------------------|------|
| 합의 | 아니오 (FLP) | Agreement + validity + termination 필요 |
| k-set agreement | k < n/(n-f)이면 아니오 | 합의의 일반화 |
| 근사 합의 | 예 | 프로세스가 ε 내에서 합의 |
| Reliable broadcast | 예 | 전달하거나 안 하지만, 전달에 합의 |
| Atomic broadcast | 아니오 | 합의와 동치 |
| Eventual consistency | 예 | 제한된 시간 없이 수렴 |

---

## 8. 장애 감지기: Chandra-Toueg 프레임워크

### 8.1 아이디어

Chandra와 Toueg(1996)는 물었습니다: "비동기 모델에 추가되었을 때 합의를 해결 가능하게 만드는 **가장 약한** 메커니즘은 무엇인가?"

그들은 두 가지 속성에 기반한 **장애 감지기 클래스**를 정의했습니다:

**완전성(Completeness)**: 모든 crash된 프로세스가 결국 모든(일부) 올바른 프로세스에 의해 의심됩니다.

| 유형 | 정의 |
|------|------|
| Strong completeness | 모든 crash된 프로세스가 결국 **모든** 올바른 프로세스에 의해 영구적으로 의심됨 |
| Weak completeness | 모든 crash된 프로세스가 결국 **일부** 올바른 프로세스에 의해 영구적으로 의심됨 |

**정확성(Accuracy)**: 올바른 프로세스가 거짓으로 의심되지 않습니다.

| 유형 | 정의 |
|------|------|
| Strong accuracy | 어떤 올바른 프로세스도 의심되지 않음 |
| Weak accuracy | 일부 올바른 프로세스가 의심되지 않음 |
| Eventual strong accuracy | 어떤 시간 이후, 어떤 올바른 프로세스도 의심되지 않음 |
| Eventual weak accuracy | 어떤 시간 이후, 일부 올바른 프로세스가 의심되지 않음 |

### 8.2 장애 감지기 클래스

완전성과 정확성을 결합하면 8개 클래스가 됩니다. 가장 중요한 것들:

| 클래스 | 기호 | 완전성 | 정확성 | 합의? |
|--------|------|--------|--------|-------|
| Perfect | P | Strong | Strong | 예 |
| Strong | S | Strong | Weak | 예 |
| Eventually Perfect | ◇P | Strong | Eventual strong | 예 |
| Eventually Strong | ◇S | Strong | Eventual weak | **예** (가장 약함!) |
| Eventually Weak | ◇W | Weak | Eventual weak | 예 (축소 포함) |

### 8.3 핵심 결과

**정리** (Chandra and Toueg, 1996): ◇S (eventually strong 장애 감지기)는 crash 장애(f < n/2)가 있는 비동기 모델에서 합의를 해결할 수 있는 **가장 약한** 장애 감지기 클래스입니다.

```
◇S guarantees:
  1. Strong completeness: every crashed process is eventually
     permanently suspected by every correct process.
  2. Eventual weak accuracy: there exists a time after which
     SOME correct process is never suspected by any correct process.
```

**직관**: "최종적으로 신뢰되는" 프로세스가 비공식적 leader로 작동합니다. 정확성 조건이 발동한 후, 모든 프로세스가 이 프로세스를 의심하지 않기로 합의하고, 이것이 합의를 완료까지 이끌 수 있습니다.

### 8.4 ◇S를 이용한 합의

```python
def chandra_toueg_consensus(process_id, proposal, processes, failure_detector):
    """
    Chandra-Toueg consensus using a ◇S failure detector.
    Rotating coordinator protocol.
    Tolerates f < n/2 crash failures.
    """
    n = len(processes)
    estimate = proposal     # current estimate of the decision value
    timestamp = 0           # round in which estimate was last updated

    for r in range(1, 1000):  # rounds (will terminate after GST)
        coordinator = processes[r % n]  # rotating coordinator

        # Phase 1: Send estimate to coordinator
        if process_id != coordinator:
            send(coordinator, ("ESTIMATE", r, estimate, timestamp))

        # Coordinator collects estimates
        if process_id == coordinator:
            estimates = collect(n // 2 + 1, timeout=None)  # wait for majority
            # Pick estimate with highest timestamp
            best = max(estimates, key=lambda e: e[2])
            proposal_value = best[1]
            # Phase 2: Propose to all
            broadcast(("PROPOSE", r, proposal_value))

        # All processes wait for coordinator's proposal
        msg = receive_from(coordinator, timeout=failure_detector.timeout(coordinator))

        if msg is not None:
            _, _, proposed = msg
            estimate = proposed
            timestamp = r
            send(coordinator, ("ACK", r))
        else:
            # Coordinator suspected of failure
            send(coordinator, ("NACK", r))

        # Coordinator decides if it got majority ACKs
        if process_id == coordinator:
            acks = count_acks(r)
            if acks > n // 2:
                broadcast(("DECIDE", r, proposal_value))
                return proposal_value

        # Check for decision message
        decision = check_for_decision()
        if decision is not None:
            return decision

    return None  # should not reach here after GST
```

### 8.5 축소: Weak에서 Strong 완전성으로

**정리**: Weak completeness를 가진 어떤 장애 감지기도 정확성을 변경하지 않고 strong completeness를 가진 것으로 변환할 수 있습니다.

**알고리즘**: 프로세스 `pᵢ`가 `pⱼ`를 의심하면(weak completeness → 일부 프로세스가 의심), `pᵢ`가 "나는 `pⱼ`를 의심한다"고 브로드캐스트합니다. 이 메시지를 수신한 모든 프로세스도 `pⱼ`를 의심하기 시작합니다.

이것은 정확성을 장애 감지기 클래스 간의 구분 요소로 집중할 수 있게 해줍니다.

---

## 9. 실용적 함의

### 9.1 Paxos와 Raft가 FLP에도 불구하고 동작하는 이유

| 측면 | FLP와의 관계 |
|------|------------|
| Safety | 항상 보장 (비동기 기간에도) -- 타이밍에 의존하지 않음 |
| Liveness | 부분 동기성에 의존 (leader 선출 + 안정적 leader) |
| Leader 선출 | 타임아웃 사용 (불완전한 장애 감지기 ≈ ◇S) |
| FLP 우회 | Liveness를 위한 부분 동기성; safety는 무조건적 |

```
Raft's relationship to FLP:

  FLP says: You cannot have safety + liveness in async model
  Raft says: We guarantee safety ALWAYS + liveness AFTER leader stabilizes

  During network partition or leader failure:
    - Raft may not make progress (no leader) → liveness violated
    - But Raft NEVER disagrees on committed values → safety preserved

  After partition heals:
    - Leader election succeeds (timeouts ≈ ◇S)
    - Progress resumes
    - All previously committed values are preserved
```

### 9.2 FLP가 실제로 물리는 경우

FLP는 이론적 호기심만이 아닙니다. 실제 시스템에서 나타납니다:

| 시나리오 | FLP 발현 | 완화 |
|---------|----------|------|
| 긴 GC 일시 정지 | Leader가 의심되어 재선출되지만 기존 leader가 여전히 살아있음 → 스플릿 브레인 위험 | Fencing 토큰, 리스 기반 리더십 |
| 대칭 네트워크 파티션 | 두 반쪽이 각각 leader를 선출 → 다수 quorum이 필요 | 어떤 결정에도 다수 필요 |
| 메시지 순서 변경 | 순서 벗어난 전달이 순진한 프로토콜을 혼란시킬 수 있음 | 시퀀스 번호, 로그 기반 프로토콜 |
| 연쇄 타임아웃 | 타임아웃 폭풍이 모든 노드가 서로를 의심 → 진행 없음 | 지수 백오프, pre-vote |

### 9.3 실천과 이론의 격차

```
Theoretical result:
  "Consensus is impossible in the asynchronous model."

Practical reality:
  "Consensus works fine 99.99% of the time because
   networks are usually well-behaved."

The gap:
  - Networks are USUALLY partially synchronous
  - The adversary in FLP is unrealistically powerful
  - Practical protocols sacrifice liveness during rare bad periods
  - The "cost" is occasional unavailability, not incorrectness
```

---

## 10. 코드: FLP 스타일 적대자 시뮬레이션

### 10.1 적대적 스케줄러

```python
"""
Simulate the FLP impossibility by implementing an adversarial scheduler
that prevents consensus by maintaining bivalent configurations.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Decision(Enum):
    UNDECIDED = "undecided"
    ZERO = "0"
    ONE = "1"


@dataclass
class ProcessState:
    """State of a single consensus process."""
    pid: int
    proposal: int          # 0 or 1
    estimate: int          # current estimate
    round_num: int = 0
    decided: bool = False
    decision: Optional[int] = None
    alive: bool = True
    messages_sent: int = 0
    messages_received: int = 0


@dataclass
class PendingMessage:
    """A message in the network buffer."""
    sender: int
    receiver: int
    round_num: int
    value: int
    delivered: bool = False
    delay: int = 0  # adversarial delay


class SimpleConsensusProtocol:
    """
    A simple (deterministic) consensus protocol that the FLP adversary will defeat.

    Protocol:
    - Round-based: each round, every process broadcasts its estimate
    - If a process receives a majority with the same value, it decides
    - Otherwise, it adopts the majority value as its new estimate
    """

    def __init__(self, n: int, proposals: list[int]):
        self.n = n
        self.processes = [
            ProcessState(pid=i, proposal=proposals[i], estimate=proposals[i])
            for i in range(n)
        ]
        self.message_buffer: list[PendingMessage] = []
        self.round = 0
        self.decided_values: dict[int, int] = {}

    def start_round(self):
        """Each process broadcasts its current estimate."""
        self.round += 1
        for p in self.processes:
            if p.alive and not p.decided:
                p.round_num = self.round
                for q in self.processes:
                    if q.pid != p.pid:
                        msg = PendingMessage(
                            sender=p.pid,
                            receiver=q.pid,
                            round_num=self.round,
                            value=p.estimate,
                        )
                        self.message_buffer.append(msg)
                        p.messages_sent += 1

    def deliver_messages(self, deliveries: list[int]):
        """
        Deliver specific messages (by index) from the buffer.
        The adversary controls WHICH messages are delivered.
        """
        received: dict[int, list[int]] = {i: [] for i in range(self.n)}

        for idx in deliveries:
            if idx < len(self.message_buffer):
                msg = self.message_buffer[idx]
                if not msg.delivered:
                    msg.delivered = True
                    proc = self.processes[msg.receiver]
                    if proc.alive and not proc.decided:
                        received[msg.receiver].append(msg.value)
                        proc.messages_received += 1

        # Process received messages
        for pid, values in received.items():
            if not values:
                continue
            proc = self.processes[pid]
            if proc.decided:
                continue

            # Add own estimate
            all_values = values + [proc.estimate]
            count_0 = sum(1 for v in all_values if v == 0)
            count_1 = sum(1 for v in all_values if v == 1)

            # Decision rule: if supermajority (> 2n/3), decide
            threshold = (2 * self.n) // 3 + 1
            if count_0 >= threshold:
                proc.decided = True
                proc.decision = 0
                self.decided_values[pid] = 0
            elif count_1 >= threshold:
                proc.decided = True
                proc.decision = 1
                self.decided_values[pid] = 1
            else:
                # Adopt majority as new estimate
                proc.estimate = 0 if count_0 >= count_1 else 1

    def undelivered_messages(self) -> list[int]:
        """Return indices of undelivered messages."""
        return [i for i, msg in enumerate(self.message_buffer) if not msg.delivered]

    def is_decided(self) -> bool:
        """Check if any process has decided."""
        return len(self.decided_values) > 0

    def all_decided(self) -> bool:
        """Check if all alive processes have decided."""
        return all(
            p.decided for p in self.processes if p.alive
        )

    def status(self) -> str:
        estimates = [p.estimate for p in self.processes if p.alive]
        decisions = {p.pid: p.decision for p in self.processes if p.decided}
        pending = len(self.undelivered_messages())
        return (f"Round {self.round}: estimates={estimates}, "
                f"decisions={decisions}, pending_msgs={pending}")


class FLPAdversary:
    """
    An adversary that exploits FLP-style reasoning to delay consensus.

    Strategy:
    - Identify which messages, if delivered, would lead to a decision
    - Delay those messages
    - Deliver other messages to maintain bivalence
    - Optionally crash one process at the worst moment
    """

    def __init__(self, protocol: SimpleConsensusProtocol, max_delay: int = 10):
        self.protocol = protocol
        self.max_delay = max_delay
        self.rounds_delayed = 0
        self.crash_used = False

    def choose_deliveries(self) -> list[int]:
        """
        Choose which messages to deliver.
        Strategy: maintain disagreement by selectively delivering.
        """
        undelivered = self.protocol.undelivered_messages()
        if not undelivered:
            return []

        # Separate messages by value
        msgs_with_0 = [
            i for i in undelivered
            if self.protocol.message_buffer[i].value == 0
        ]
        msgs_with_1 = [
            i for i in undelivered
            if self.protocol.message_buffer[i].value == 1
        ]

        # Strategy: deliver messages that maintain balance
        # Give 0-messages to some processes and 1-messages to others
        deliveries = []
        n = self.protocol.n

        # Deliver 0-messages to the first half of processes
        for idx in msgs_with_0:
            msg = self.protocol.message_buffer[idx]
            if msg.receiver < n // 2:
                deliveries.append(idx)

        # Deliver 1-messages to the second half
        for idx in msgs_with_1:
            msg = self.protocol.message_buffer[idx]
            if msg.receiver >= n // 2:
                deliveries.append(idx)

        # If we would deliver nothing, deliver a subset to ensure fairness
        if not deliveries and undelivered:
            deliveries = undelivered[:len(undelivered)//2]

        self.rounds_delayed += 1
        return deliveries

    def maybe_crash_process(self):
        """
        Crash a process at the worst moment (only once, per FLP model).
        Strategy: crash the process that is about to break the tie.
        """
        if self.crash_used:
            return

        # Find process whose vote would break a tie
        estimates = [p.estimate for p in self.protocol.processes if p.alive]
        count_0 = sum(1 for e in estimates if e == 0)
        count_1 = sum(1 for e in estimates if e == 1)

        if count_0 == count_1:
            # Perfect tie -- crashing anyone maintains it
            return

        # Crash a process from the majority to create a tie
        majority_val = 0 if count_0 > count_1 else 1
        for p in self.protocol.processes:
            if p.alive and not p.decided and p.estimate == majority_val:
                p.alive = False
                self.crash_used = True
                print(f"  ADVERSARY: Crashed process {p.pid} "
                      f"(estimate={p.estimate}) to maintain bivalence")
                return


def run_flp_simulation(n: int = 5, max_rounds: int = 20):
    """
    Run the FLP adversary against a simple consensus protocol.
    Demonstrates how the adversary prevents consensus.
    """
    print("="*70)
    print("FLP IMPOSSIBILITY SIMULATION")
    print(f"Processes: {n}, Max rounds: {max_rounds}")
    print("="*70)

    # Start with a "nearly balanced" proposal: slightly more 0s
    proposals = [0] * (n // 2 + 1) + [1] * (n // 2)
    print(f"Initial proposals: {proposals}")

    protocol = SimpleConsensusProtocol(n, proposals)
    adversary = FLPAdversary(protocol)

    for r in range(max_rounds):
        print(f"\n--- Round {r + 1} ---")

        # Start the round (processes broadcast estimates)
        protocol.start_round()
        print(f"  {protocol.status()}")

        # Adversary chooses which messages to deliver
        deliveries = adversary.choose_deliveries()
        print(f"  Adversary delivers {len(deliveries)} of "
              f"{len(protocol.undelivered_messages())} messages")

        # Deliver the chosen messages
        protocol.deliver_messages(deliveries)
        print(f"  After delivery: {protocol.status()}")

        # Adversary may crash a process
        adversary.maybe_crash_process()

        # Check if consensus was reached despite adversary
        if protocol.is_decided():
            print(f"\n  CONSENSUS REACHED despite adversary! "
                  f"Decisions: {protocol.decided_values}")
            break

        if protocol.all_decided():
            break
    else:
        print(f"\n{'='*70}")
        print(f"RESULT: Adversary prevented consensus for {max_rounds} rounds")
        print(f"This demonstrates the FLP impossibility in action.")
        print(f"Rounds delayed: {adversary.rounds_delayed}")
        print(f"Process crashed: {adversary.crash_used}")
        print(f"{'='*70}")

    # Final status
    print(f"\nFinal process states:")
    for p in protocol.processes:
        status = "CRASHED" if not p.alive else (
            f"DECIDED {p.decision}" if p.decided else f"UNDECIDED (est={p.estimate})"
        )
        print(f"  P{p.pid}: {status} "
              f"(sent={p.messages_sent}, recv={p.messages_received})")


# Run the simulation
if __name__ == "__main__":
    run_flp_simulation(n=5, max_rounds=20)
    print("\n")
    run_flp_simulation(n=7, max_rounds=15)
```

### 10.2 무작위화가 적대자를 깨뜨리는 방법 시연

```python
def run_randomized_vs_adversary(n: int = 5, trials: int = 100):
    """
    Show that adding randomization allows consensus to eventually
    succeed despite the adversary, breaking the FLP deadlock.
    """
    print("="*70)
    print("RANDOMIZED CONSENSUS vs FLP ADVERSARY")
    print(f"Processes: {n}, Trials: {trials}")
    print("="*70)

    success_rounds = []

    for trial in range(trials):
        proposals = [random.choice([0, 1]) for _ in range(n)]
        estimates = list(proposals)
        decided = False

        for r in range(1, 200):
            # Adversary: deliver messages to maintain disagreement (simplified)
            # But with random coin flips, adversary cannot predict outcomes

            # Simulate a round of Ben-Or-style protocol
            count_0 = sum(1 for e in estimates if e == 0)
            count_1 = sum(1 for e in estimates if e == 1)

            if count_0 > n // 2:
                proposal = 0
            elif count_1 > n // 2:
                proposal = 1
            else:
                proposal = None

            # Decision check
            if proposal is not None:
                # Check if supermajority agrees
                agreement = sum(1 for e in estimates if e == proposal)
                if agreement > (2 * n) // 3:
                    decided = True
                    success_rounds.append(r)
                    break
                else:
                    estimates = [proposal] * n  # all adopt
            else:
                # Random coin flip for each process
                estimates = [random.choice([0, 1]) for _ in range(n)]

        if not decided:
            success_rounds.append(200)

    avg_rounds = sum(success_rounds) / len(success_rounds)
    max_rounds = max(success_rounds)
    min_rounds = min(success_rounds)

    print(f"\nResults over {trials} trials:")
    print(f"  Average rounds to consensus: {avg_rounds:.1f}")
    print(f"  Min rounds: {min_rounds}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Success rate: {sum(1 for r in success_rounds if r < 200)/trials*100:.1f}%")
    print(f"\nConclusion: Randomization breaks the adversary's strategy.")
    print(f"Even though individual rounds may fail, consensus is reached")
    print(f"with probability 1 over sufficiently many rounds.")


if __name__ == "__main__":
    run_randomized_vs_adversary(n=5, trials=100)
```

### 10.3 부분 동기성 시뮬레이션

```python
def demonstrate_partial_synchrony(n: int = 5, gst: int = 10):
    """
    Show how partial synchrony (with a GST) enables consensus.
    Before GST: arbitrary delays. After GST: bounded delays.
    """
    print("="*70)
    print(f"PARTIAL SYNCHRONY SIMULATION (GST at round {gst})")
    print("="*70)

    proposals = [0, 0, 1, 1, 0]
    estimates = list(proposals)
    leader = None
    decided = False

    for r in range(1, 30):
        is_after_gst = r >= gst

        if is_after_gst and leader is None:
            leader = 0  # stable leader election after GST
            print(f"\n  Round {r}: GST reached! Leader elected: P{leader}")

        if not is_after_gst:
            # Before GST: adversary can delay, no stable leader
            # Simulate failed leader election
            candidate = r % n
            heard = random.randint(0, n - 1)
            if heard <= n // 2:
                print(f"  Round {r}: Leader election failed "
                      f"(P{candidate} heard from only {heard}/{n})")
                continue
            # Even if elected, messages may be delayed
            if random.random() < 0.5:
                print(f"  Round {r}: P{candidate} elected but messages delayed")
                continue

        # After GST (or lucky round before): leader-based consensus
        if leader is not None:
            leader_value = estimates[leader]
            # Leader proposes to all
            acks = 0
            for i in range(n):
                if is_after_gst or random.random() > 0.3:
                    estimates[i] = leader_value
                    acks += 1

            if acks > n // 2:
                decided = True
                print(f"\n  Round {r}: CONSENSUS on value {leader_value}!")
                print(f"  All estimates: {estimates}")
                print(f"  Rounds after GST: {r - gst + 1}")
                break
            else:
                print(f"  Round {r}: Leader got {acks}/{n} acks (need > {n//2})")

    if not decided:
        print(f"\n  Consensus not reached in 30 rounds")

    return decided


if __name__ == "__main__":
    demonstrate_partial_synchrony(n=5, gst=10)
```

---

## 11. 요약

### FLP 불가능성 한눈에 보기

```
┌──────────────────────────────────────────────────────────────────┐
│                   FLP IMPOSSIBILITY THEOREM                      │
│                                                                  │
│  MODEL: Asynchronous, reliable links, deterministic protocol     │
│  FAULT: Even 1 crash-stop failure                                │
│  RESULT: Consensus (agreement + validity + termination)          │
│          is IMPOSSIBLE                                           │
│                                                                  │
│  WHY: Adversary can always keep configuration bivalent           │
│       by selectively delaying messages                           │
│                                                                  │
│  PROOF:                                                          │
│    Lemma 1: ∃ bivalent initial configuration                     │
│    Lemma 2: From bivalent, can stay bivalent                     │
│    Theorem: Infinite non-deciding execution exists                │
│                                                                  │
│  CIRCUMVENTIONS:                                                 │
│    1. Randomization      → expected termination (Ben-Or)         │
│    2. Partial synchrony  → termination after GST (Paxos/Raft)   │
│    3. Failure detectors  → ◇S weakest sufficient (Chandra-Toueg)│
│    4. Timeouts           → practical ◇P approximation            │
│    5. Weaker problems    → eventual consistency, etc.            │
│                                                                  │
│  PRACTICAL IMPACT:                                               │
│    - Every consensus protocol MUST sacrifice something           │
│    - Paxos/Raft: sacrifice liveness during async periods         │
│    - Bitcoin: sacrifice deterministic safety                     │
│    - CRDTs: sacrifice strong consistency                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 핵심 정리

1. **FLP는 모델에 대한 것이지, 현실에 대한 것이 아닙니다**: 실제 시스템은 순수 비동기가 아니므로, FLP가 실용적 합의를 방지하지 않습니다.

2. **Safety vs liveness 트레이드오프**: FLP는 모든 프로토콜이 어떤 것을 희생하도록 강제합니다. 실용적 프로토콜은 liveness(일시적 불가용성)를 희생하여 safety(정확성)를 보존합니다.

3. **적대자가 핵심**: FLP의 적대자는 메시지 스케줄링을 제어합니다. 실제로는 어떤 적대자도 그런 정밀도로 네트워크를 제어하지 않습니다.

4. **하한이 설계를 안내합니다**: 합의가 `Ω(n²)` 메시지 또는 `f + 1` 라운드를 필요로 한다는 것을 알면 엔지니어가 현실적인 성능 기대를 설정하는 데 도움이 됩니다.

5. **장애 감지기가 이론과 실천을 잇습니다**: ◇S 추상화는 실제 시스템에서 타임아웃이 제공하는 것을 형식화합니다.

---

## 12. 연습 문제

### 문제 1: 보조 정리 1 검증

Binary 제안을 가진 3-프로세스 시스템에 대해, 8개의 초기 구성을 모두 나열하고 적어도 하나가 bivalent여야 함을 보이세요 (섹션 4.1의 인접성 논증 사용).

### 문제 2: 교환 가능성

프로세스 P1, P2, P3가 구성 `C`에 있습니다:

```
C: P1 has estimate 0, P2 has estimate 1, P3 has estimate 1
Message buffer: {(P1→P2, value=0), (P2→P3, value=1)}
```

단계 `e1 = (P2 receives from P1)`와 `e2 = (P3 receives from P2)`를 양쪽 순서로 적용하세요. 최종 구성이 동일한지 확인하세요 (`p ≠ p'`일 때의 교환 가능성).

### 문제 3: 장애 감지기 분류

각 시나리오에 대해, 적용되는 가장 강한 장애 감지기 클래스를 식별하세요:

1. 5초 후, 모든 crash된 노드가 정확히 식별되고 올바른 노드는 오식별되지 않음.
2. 5초 후, 모든 crash된 노드가 적어도 하나의 올바른 프로세스에 의해 식별되고, 적어도 하나의 올바른 프로세스가 오식별되지 않음.
3. 즉시 그리고 항상, 모든 crash된 노드가 정확히 식별되고 올바른 노드는 오식별되지 않음.

### 문제 4: 라운드 계산

7개 프로세스와 최대 2개 crash가 있는 동기 시스템에서:
1. 합의에 필요한 최소 라운드 수는?
2. 장애 허용을 3개 crash로 늘리면, 몇 라운드가 더 필요한가?
3. 최악의 경우 전송되는 총 메시지 수는?

### 문제 5: 구현 도전

FLP 적대자 시뮬레이션을 확장하여:

1. 메시지 간의 인과적 의존성을 추적하고 bivalence를 깨뜨릴 메시지를 선택적으로 지연하는 더 정교한 적대자를 구현.
2. 설정 가능한 라운드에서 GST를 활성화하는 "부분 동기성 스위치"를 추가하고, GST 후 O(f) 라운드 내에 합의에 도달함을 보임.
3. Ben-Or의 무작위화 프로토콜을 구현하고 적대자에 대해 합의에 도달하는 데 걸리는 라운드 수를 측정.

---

## 13. 참고 문헌

1. Fischer, M. J., Lynch, N. A., & Paterson, M. S. (1985). "Impossibility of Distributed Consensus with One Faulty Process." *Journal of the ACM*, 32(2), 374-382.
2. Dwork, C., Lynch, N., & Stockmeyer, L. (1988). "Consensus in the Presence of Partial Synchrony." *Journal of the ACM*, 35(2), 288-323.
3. Chandra, T. D., & Toueg, S. (1996). "Unreliable Failure Detectors for Reliable Distributed Systems." *Journal of the ACM*, 43(2), 225-267.
4. Chandra, T. D., Hadzilacos, V., & Toueg, S. (1996). "The Weakest Failure Detector for Solving Consensus." *Journal of the ACM*, 43(4), 685-722.
5. Ben-Or, M. (1983). "Another Advantage of Free Choice: Completely Asynchronous Agreement Protocols." *PODC 1983*.
6. Dolev, D., & Reischuk, R. (1985). "Bounds on Information Exchange for Byzantine Agreement." *Journal of the ACM*, 32(1), 191-204.
7. Dolev, D., & Strong, H. R. (1983). "Authenticated Algorithms for Byzantine Agreement." *SIAM Journal on Computing*, 12(4), 656-666.
8. Pease, M., Shostak, R., & Lamport, L. (1980). "Reaching Agreement in the Presence of Faults." *Journal of the ACM*, 27(2), 228-234.
9. Lynch, N. (1996). *Distributed Algorithms*. Morgan Kaufmann. Chapters 5-6.
10. Attiya, H., & Welch, J. (2004). *Distributed Computing: Fundamentals, Simulations, and Advanced Topics*. Wiley. Chapter 5.

---

[다음: 레슨 04 - 일관성 모델 심층 분석](./04_Consistency_Models.md)
