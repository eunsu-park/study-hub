# 레슨 9: 행렬 분해

## 학습 목표

- 고유 분해를 복습하고 DL 맥락에서 고유값/고유벡터를 해석한다
- 직사각 행렬에 특이값 분해(SVD)를 유도하고 적용한다
- Eckart-Young 정리를 통해 SVD와 저랭크 근사를 연결한다
- PCA가 차원 축소를 위해 고유 분해/SVD를 사용하는 방식을 이해한다
- 저랭크 분해 기법(LoRA)을 신경망 가중치 행렬에 적용한다
- SVD를 가중치 초기화와 학습된 네트워크 분석에 사용한다
- GAN의 스펙트럼 정규화를 위해 SVD로 스펙트럼 노름을 계산한다
- 모델 압축을 위한 절단 SVD를 구현한다

---

## 1. 고유 분해 복습

대칭 행렬 $\mathbf{A} = \mathbf{A}^\top$: $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top$ (직교 고유벡터, 실수 고유값).

**DL 맥락**: 손실 함수의 헤시안은 대칭. 고유값 = 각 주요 방향의 곡률, 최대 고유값이 최대 안전 학습률 결정 ($\eta < 2/\lambda_\max$).

---

## 2. 특이값 분해 (SVD)

### 2.1 정의

임의의 행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$: $\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$

- $\mathbf{U}$: 좌 특이 벡터 (직교)
- $\boldsymbol{\Sigma}$: 특이값의 대각 행렬 ($\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $\mathbf{V}$: 우 특이 벡터 (직교)

### 2.2 SVD와 고유 분해의 관계

- $\mathbf{A}^\top \mathbf{A}$의 고유 분해 $\to$ $\mathbf{V}$와 $\sigma_i^2$
- 특이값 = $\mathbf{A}^\top \mathbf{A}$ 고유값의 제곱근

---

## 3. 저랭크 근사

### 3.1 Eckart-Young 정리

$\mathbf{A}$의 최적 랭크-$k$ 근사는 SVD를 절단하여 얻음:

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

근사 오차: $\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$

### 3.2 압축비

$1000 \times 1000$ 행렬의 랭크-10 근사: $\approx 50$배 압축.

---

## 4. DL에서의 SVD

### 4.1 LoRA

$$\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times d}$$

$d = 4096$, $r = 8$: 256배 적은 매개변수.

### 4.2 스펙트럼 정규화

GAN에서 판별기의 각 층 가중치 행렬의 스펙트럼 노름(최대 특이값)을 제한:

$$\bar{\mathbf{W}} = \frac{\mathbf{W}}{\sigma_1(\mathbf{W})}$$

립시츠 상수를 1로 보장하여 학습을 안정화합니다.

---

## 5. SVD를 통한 PCA

중심화된 데이터 행렬 $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$에서, 주성분은 $\mathbf{V}$의 열이고 분산은 $\sigma_i^2 / (N-1)$.

상위 $k$개 주성분으로 투영: $\mathbf{X}_k = \mathbf{X} \mathbf{V}_k \in \mathbb{R}^{N \times k}$

---

## 6. 콜레스키 분해

대칭 양정부호 행렬 $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$ (하삼각 $\mathbf{L}$).

**DL 용도**: 다변량 가우시안 샘플링 ($\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$), 선형 시스템의 빠른 풀이.

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 고유 분해 | 대칭 행렬에 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$; 곡률 드러냄 |
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$; 임의 행렬에 작동 |
| 저랭크 근사 | 절단 SVD가 최적 랭크-$k$ 적합 (Eckart-Young) |
| LoRA | $\Delta W = BA$ ($r \ll d$)로 미세 조정; 100배+ 매개변수 절감 |
| 스펙트럼 정규화 | $\bar{W} = W / \sigma_1(W)$; GAN 학습 안정화 |
| PCA | 공분산의 고유 분해 = 중심화 데이터의 SVD |
| 콜레스키 | 양정부호 행렬에 $A = LL^\top$; 빠른 샘플링과 풀이 |

---

## 연습문제

1. 거듭제곱법으로 행렬의 최대 특이값과 해당 특이 벡터를 찾는 것을 구현하세요.
2. 절단 SVD로 가중치 행렬을 랭크 $k$로 압축하고 $k$에 대한 복원 오차를 측정하세요.
3. LoRA를 구현하세요: 가중치 행렬을 고정하고 저랭크 $B$와 $A$를 학습하며 순전파를 검증하세요.
4. 고차원 데이터셋(예: 100D)에 PCA를 적용하고 설명된 분산 곡선(스크리 도표)을 그리세요.
5. 거듭제곱법을 사용하여 가중치 행렬의 스펙트럼 정규화를 구현하고 립시츠 한계를 검증하세요.

---

**다음**: [10. 수치 안정성](10_Numerical_Stability.md)
