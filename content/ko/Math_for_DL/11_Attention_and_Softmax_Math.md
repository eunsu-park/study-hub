# 레슨 11: 어텐션과 소프트맥스 수학

## 학습 목표

- 스케일드 닷-프로덕트 어텐션 공식을 원리부터 유도한다
- 어텐션 스코어를 $1/\sqrt{d_k}$로 스케일링하는 이유를 설명한다
- 어텐션 메커니즘을 통한 그래디언트 흐름을 분석한다
- 소프트맥스 온도와 어텐션 분포에 미치는 영향을 이해한다
- 소프트맥스의 야코비안을 유도하고 그래디언트 계산에 대한 함의를 이해한다
- 다중 헤드 어텐션을 병렬 부분 공간 투영으로 이해한다
- 어텐션의 계산 복잡도와 효율적 변형의 동기를 분석한다

---

## 1. 닷-프로덕트 어텐션

### 1.1 쿼리, 키, 값

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

- **쿼리**: "무엇을 찾고 있는가?"
- **키**: "무엇을 담고 있는가?"
- **값**: "어떤 정보를 제공하는가?"

---

## 2. 왜 $1/\sqrt{d_k}$로 스케일링하는가?

### 2.1 분산 논거

$q_i$와 $k_j$가 평균 0, 분산 1인 독립 확률 변수라면:

$$\text{Var}(\mathbf{q}^\top \mathbf{k}) = d_k$$

큰 $d_k$에서 일부 닷-프로덕트가 매우 커져, 소프트맥스가 포화 영역(거의 원-핫)에 들어갑니다.

### 2.2 수정

$\sqrt{d_k}$로 나누면 닷-프로덕트의 분산을 1로 정규화:

$$\text{Var}\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}\right) = 1$$

$d_k$에 관계없이 소프트맥스를 민감한(비포화) 영역에 유지합니다.

---

## 3. 소프트맥스 성질

### 3.1 야코비안

$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j), \quad \mathbf{J} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$$

### 3.2 온도 스케일링

$$s_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

| $\tau$ | 효과 |
|--------|------|
| $\tau \to 0^+$ | 하드 argmax (원-핫) |
| $\tau = 1$ | 표준 소프트맥스 |
| $\tau \to \infty$ | 균등 분포 |

**DL 용도**: 지식 증류 ($\tau > 1$), 샘플링 ($\tau < 1$), 대조 학습

---

## 4. 어텐션을 통한 그래디언트 흐름

$\mathbf{O} = \mathbf{A}\mathbf{V}$, $\mathbf{A} = \text{softmax}(\mathbf{S}/\sqrt{d_k})$, $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$에서

$\frac{\partial L}{\partial \mathbf{O}}$가 주어지면:

1. $\frac{\partial L}{\partial \mathbf{V}} = \mathbf{A}^\top \frac{\partial L}{\partial \mathbf{O}}$
2. $\frac{\partial L}{\partial \mathbf{A}} = \frac{\partial L}{\partial \mathbf{O}} \mathbf{V}^\top$
3. 소프트맥스를 통한 그래디언트: $\frac{\partial L}{\partial S_{ij}} = A_{ij}(\bar{A}_{ij} - \sum_l \bar{A}_{il} A_{il})$
4. $\frac{\partial L}{\partial \mathbf{Q}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial \mathbf{S}} \mathbf{K}$

---

## 5. 다중 헤드 어텐션

$h$개 어텐션 헤드를 병렬로 실행, 각각 다른 학습된 투영 사용:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = [\text{head}_1; \ldots; \text{head}_h]\mathbf{W}_O$$

$d_k = d_v = d_\text{model} / h$로, 총 계산량은 전체 차원의 단일 헤드와 동일하지만, 각 헤드가 다른 측면에 주의를 기울일 수 있습니다.

---

## 6. 계산 복잡도

| 연산 | 복잡도 |
|------|--------|
| $\mathbf{Q}\mathbf{K}^\top$ | $O(T^2 d_k)$ |
| $\mathbf{A}\mathbf{V}$ | $O(T^2 d_v)$ |
| **총** | **$O(T^2 d)$** |
| $\mathbf{A}$ 메모리 | 헤드당 $O(T^2)$ |

$O(T^2)$ 스케일링이 컨텍스트 길이를 제한하여, Flash Attention, 선형 어텐션, 희소 어텐션 등의 동기가 됩니다.

### 인과적 마스킹

자기회귀 모델에서 위치 $i$는 $\leq i$ 위치만 참조: $S_{ij} = -\infty$ ($j > i$)를 소프트맥스 전에 설정.

---

## 7. 소프트 딕셔너리 조회로서의 어텐션

어텐션은 키-값 저장소의 **소프트 조회**로 볼 수 있습니다:

$$\text{output}_i = \sum_j P(j | i) \cdot \mathbf{v}_j = \mathbb{E}_{j \sim P(\cdot|i)}[\mathbf{v}_j]$$

가중치는 쿼리-키 유사도에 의해 결정되는 값의 가중 평균입니다.

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 스케일드 닷-프로덕트 | $\text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\mathbf{V}$; 스케일링이 소프트맥스 포화 방지 |
| 스케일링 인수 | $1/\sqrt{d_k}$가 닷-프로덕트 분산을 1로 정규화 |
| 온도 | $\tau < 1$: 날카롭게; $\tau > 1$: 부드럽게; $\tau \to 0$: argmax |
| 어텐션 그래디언트 | 소프트맥스를 통해: $dS_{ij} = A_{ij}(\bar{A}_{ij} - \sum_l \bar{A}_{il} A_{il})$ |
| 다중 헤드 | $h$개 부분 공간에서 병렬 어텐션; 총 비용 동일 |
| 복잡도 | 시간 $O(T^2 d)$, 헤드당 메모리 $O(T^2)$ |
| 인과 마스크 | 자기회귀 모델을 위해 미래 스코어를 $-\infty$로 설정 |

---

## 연습문제

1. 인과 마스킹을 가진 스케일드 닷-프로덕트 어텐션을 구현하고 위치 $i$가 $\leq i$ 위치만 참조함을 검증하세요.
2. 스케일링 없이 $d_k$의 함수로 어텐션 가중치의 엔트로피를 그려 스케일링이 필요한 이유를 보이세요.
3. 다중 헤드 어텐션을 처음부터 구현하고 각 단계의 크기를 검증하세요.
4. 다양한 온도에서의 어텐션 가중치 엔트로피를 비교하고 지식 증류에 대한 함의를 논하세요.
5. $\text{elu}(x) + 1$ 특성 맵을 사용하여 선형 어텐션을 구현하고 표준 어텐션 출력과 비교하세요.

---

**다음**: [12. 종합 정리](12_Putting_It_All_Together.md)
