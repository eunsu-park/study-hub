[이전: 강화학습 입문](./42_Reinforcement_Learning_Intro.md) | [다음: 테스트 타임 적응](./44_Test_Time_Adaptation.md)

---

# 43. 퓨샷 학습(Few-Shot Learning)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 퓨샷 학습(few-shot learning) 문제와 표준 지도 학습과의 차이를 설명할 수 있다
2. 메타 학습(meta-learning) 프레임워크(학습하는 법을 학습)와 에피소드 학습(episodic training)을 설명할 수 있다
3. 퓨샷 분류를 위한 프로토타입 네트워크(Prototypical Networks)를 구현할 수 있다
4. 메트릭 기반(Prototypical, Matching, Relation Networks) 방법과 최적화 기반(MAML) 방법을 비교할 수 있다
5. 레이블 데이터가 제한된 실제 작업에 퓨샷 학습 기법을 적용할 수 있다

---

## 목차

1. [퓨샷 문제](#1-퓨샷-문제)
2. [메타 학습 프레임워크](#2-메타-학습-프레임워크)
3. [메트릭 기반 방법](#3-메트릭-기반-방법)
4. [프로토타입 네트워크](#4-프로토타입-네트워크)
5. [매칭 네트워크](#5-매칭-네트워크)
6. [모델 무관 메타 학습(MAML)](#6-모델-무관-메타-학습maml)
7. [관계 네트워크](#7-관계-네트워크)
8. [실전 고려사항](#8-실전-고려사항)
9. [연습문제](#9-연습문제)

---

## 1. 퓨샷 문제

### 1.1 퓨샷 학습이 필요한 이유

표준 딥러닝은 클래스당 수천 ~ 수백만 개의 레이블 예제를 필요로 합니다. 그러나 현실에서는 레이블 데이터가 부족한 경우가 많습니다:

- **의료 영상**: 희귀 질환은 진단 사례가 매우 적습니다
- **신약 개발**: 실험 결과가 제한된 새로운 분자 구조
- **로보틱스**: 배포 중에 만나는 새로운 물체
- **개인화**: 최소한의 예제로 개별 사용자에게 모델을 적응시키는 경우

퓨샷 학습은 **클래스당 1~5개의 예제**만으로 새로운 클래스를 분류하는 것을 목표로 합니다(1-shot, 5-shot 학습).

### 1.2 문제 정형화

```
표준 분류:
  훈련: 이미지 10,000개 × 100개 클래스
  테스트: 동일한 100개 클래스

퓨샷 분류:
  메타 훈련: "기본" 클래스의 대규모 데이터셋 (예: 64개 클래스, 다수 예제)
  메타 테스트: K개의 예제만 있는 새로운 "신규" 클래스 (K = 1 또는 5)
```

**N-way K-shot** 문제: 클래스당 K개의 예제를 갖는 N개의 새로운 클래스 중에서 분류.

| 용어 | 정의 |
|------|------|
| **서포트 셋(Support set)** | 클래스당 K개의 레이블 예제 (학습 대상) |
| **쿼리 셋(Query set)** | 분류해야 할 레이블 없는 예제 (예측 대상) |
| **에피소드(Episode)** | 하나의 N-way K-shot 태스크 (서포트 + 쿼리) |
| **기본 클래스(Base classes)** | 메타 훈련 중 사용 가능한 클래스 (다수 예제) |
| **신규 클래스(Novel classes)** | 테스트 시 새로운 클래스 (K개 예제만 존재) |

### 1.3 전이 학습과의 차이

| 측면 | 전이 학습(Transfer Learning) | 퓨샷 학습(Few-Shot Learning) |
|------|------------------------------|------------------------------|
| 접근법 | 새 데이터로 사전 훈련 모델 파인튜닝 | 일반화하는 학습 알고리즘 자체를 학습 |
| 필요 데이터 | 수십 ~ 수백 개 예제 | 1~5개 예제 |
| 새로운 클래스 | 파인튜닝 후 고정됨 | 테스트 시 임의의 새 클래스 처리 가능 |
| 훈련 패러다임 | 표준 (배치, 에포크) | 에피소드 (퓨샷 태스크 시뮬레이션) |

---

## 2. 메타 학습 프레임워크

### 2.1 학습하는 법을 학습하기

핵심 통찰: 특정 클래스를 분류하도록 모델을 훈련하는 대신, **적은 예제로 분류하는 방법 자체를 학습**하도록 모델을 훈련합니다. 모델은 다양한 태스크를 통해 귀납적 편향(inductive bias)을 학습하여 새로운 태스크에 전이합니다.

```
표준 학습:
  데이터셋 D → 훈련 → 모델 f → D의 클래스 예측

메타 학습:
  다수의 태스크 T₁, T₂, ..., Tₙ → 메타 훈련 → 메타 학습기 M
  새로운 태스크 T_new (소수 예제) → M → 적응된 모델 → 새 클래스 예측
```

### 2.2 에피소드 학습(Episodic Training)

메타 훈련 중, 퓨샷 시나리오를 시뮬레이션합니다:

```python
def create_episode(dataset, n_way=5, k_shot=5, n_query=15):
    """단일 N-way K-shot 에피소드를 생성합니다.

    1. 데이터셋에서 N개의 클래스를 샘플링
    2. 각 클래스에서 K개의 예제 샘플링 → 서포트 셋
    3. 각 클래스에서 추가 예제 샘플링 → 쿼리 셋
    """
    classes = random.sample(dataset.classes, n_way)

    support = []  # N × K 예제 (레이블 포함)
    query = []    # N × n_query 예제 (레이블 포함)

    for label, cls in enumerate(classes):
        examples = random.sample(dataset[cls], k_shot + n_query)
        support.extend([(x, label) for x in examples[:k_shot]])
        query.extend([(x, label) for x in examples[k_shot:]])

    return support, query
```

각 에피소드는 미니 분류 태스크입니다. 모델은 수천 개의 이런 태스크에 걸쳐 잘 수행하도록 학습됩니다.

---

## 3. 메트릭 기반 방법

가장 직관적인 접근법: 같은 클래스의 예제는 가깝고 다른 클래스는 멀리 위치하는 임베딩 공간(embedding space)을 학습합니다. 테스트 시에는 가장 가까운 클래스 대표점을 찾아 분류합니다.

### 3.1 일반 프레임워크

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  서포트 셋   │──────►│   임베딩     │──────►│   클래스     │
│  (K개 예제)  │       │   네트워크   │       │   프로토타입 │
└──────────────┘       │   f_θ        │       └──────┬───────┘
                       │              │              │
┌──────────────┐       │              │       ┌──────▼───────┐
│  쿼리 이미지 │──────►│              │──────►│   거리       │──► 예측
└──────────────┘       └──────────────┘       │   비교       │
                                              └──────────────┘
```

임베딩 네트워크 f_θ는 모든 태스크에 공유됩니다. 에피소드마다 클래스 프로토타입만 달라집니다.

---

## 4. 프로토타입 네트워크

### 4.1 알고리즘

프로토타입 네트워크(Prototypical Networks, Snell et al., 2017)는 각 클래스의 프로토타입(평균 임베딩)을 계산한 후, 가장 가까운 프로토타입으로 쿼리를 분류합니다:

1. 모든 서포트 예제를 임베딩: $e_i = f_\theta(x_i)$
2. 클래스 프로토타입 계산: $c_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} f_\theta(x_i)$
3. 거리에 대한 소프트맥스(softmax)로 쿼리 분류: $p(y=k|x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'} \exp(-d(f_\theta(x), c_{k'}))}$

여기서 $d$는 일반적으로 유클리드 거리의 제곱입니다.

### 4.2 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """퓨샷 학습 백본에서 사용되는 표준 컨브 블록."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.max_pool2d(F.relu(self.bn(self.conv(x))), 2)


class ProtoNet(nn.Module):
    """퓨샷 분류를 위한 프로토타입 네트워크.

    임베딩 네트워크는 이미지를 특징 공간으로 매핑하며,
    분류는 가장 가까운 프로토타입을 찾는 것으로 귀결됩니다.
    """

    def __init__(self, in_channels=1, hidden_dim=64, embedding_dim=64):
        super().__init__()
        # 4층 ConvNet (Omniglot/miniImageNet의 표준)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        """입력 이미지를 특징 공간으로 임베딩."""
        features = self.encoder(x)
        return features.view(features.size(0), -1)  # 평탄화

    def compute_prototypes(self, support, labels, n_way):
        """서포트 셋 임베딩으로부터 클래스 프로토타입을 계산합니다.

        prototype_k = 클래스 k에 속하는 모든 서포트 임베딩의 평균
        """
        embeddings = self.forward(support)
        prototypes = torch.zeros(n_way, embeddings.size(1),
                                 device=support.device)
        for k in range(n_way):
            mask = (labels == k)
            prototypes[k] = embeddings[mask].mean(dim=0)
        return prototypes

    def classify(self, query, prototypes):
        """프로토타입까지의 거리로 쿼리 예제를 분류합니다.

        각 쿼리 예제에 대한 로그 확률을 반환합니다.
        """
        query_emb = self.forward(query)
        # 유클리드 거리의 제곱
        # dists[i, k] = ||query_i - prototype_k||²
        dists = torch.cdist(query_emb, prototypes, p=2).pow(2)
        # 음수 거리를 로짓(logit)으로 사용 (가까울수록 점수가 높음)
        return F.log_softmax(-dists, dim=1)


def train_episode(model, support, support_labels, query, query_labels,
                  n_way, optimizer):
    """단일 에피소드로 훈련합니다."""
    model.train()
    optimizer.zero_grad()

    prototypes = model.compute_prototypes(support, support_labels, n_way)
    log_probs = model.classify(query, prototypes)
    loss = F.nll_loss(log_probs, query_labels)

    loss.backward()
    optimizer.step()

    # 정확도
    preds = log_probs.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()

    return loss.item(), acc
```

### 4.3 유클리드 거리가 효과적인 이유

프로토타입 네트워크 논문에서 저자들은 유클리드 거리의 제곱을 사용하는 것이 임베딩 공간에서의 선형 분류기와 동치(equivalent)임을 보입니다. 프로토타입은 클래스 중심점(centroid) 역할을 하며, 결정 경계는 프로토타입 주위의 보로노이 셀(Voronoi cell)이 됩니다. 이는 복잡한 거리 함수를 학습하는 것보다 더 단순하고 강건합니다.

---

## 5. 매칭 네트워크

매칭 네트워크(Matching Networks, Vinyals et al., 2016)는 서포트 셋에 대한 어텐션(attention)을 사용합니다:

$$p(y|x, S) = \sum_{i=1}^{|S|} a(x, x_i) \cdot y_i$$

여기서 $a(x, x_i) = \frac{\exp(c(f(x), g(x_i)))}{\sum_j \exp(c(f(x), g(x_j)))}$

ProtoNet과의 주요 차이점:
- 유클리드 거리 대신 코사인 유사도(cosine similarity) 사용
- 전체 컨텍스트 임베딩(Full Context Embedding)을 위해 선택적으로 LSTM을 사용하여 서포트 셋 임베딩을 전체 서포트에 조건화
- 클래스 평균이 아닌 모든 서포트 예제에 대한 가중 합산

---

## 6. 모델 무관 메타 학습(MAML)

### 6.1 아이디어

MAML(Finn et al., 2017)은 완전히 다른 접근법을 취합니다: 임베딩을 학습하는 대신, 몇 번의 경사 하강 스텝만으로 새로운 태스크에 빠르게 파인튜닝될 수 있는 **초기화(initialization)**를 학습합니다.

```
표준 훈련: 무작위 초기화 → 많은 경사 하강 스텝 → 좋은 모델

MAML: 메타 학습된 초기화 → 1~5번의 경사 하강 스텝 → 새 태스크에 좋은 모델
```

### 6.2 알고리즘

```
메타 훈련:
  태스크 T₁, ..., Tₙ 배치마다:
    각 태스크 Tᵢ에 대해:
      1. 현재 파라미터 복사: θ'ᵢ = θ
      2. 내부 루프: Tᵢ의 서포트 셋에 K번 경사 하강
         θ'ᵢ ← θ'ᵢ - α ∇_θ'ᵢ L(θ'ᵢ, support_i)
      3. Tᵢ의 쿼리 셋에서 적응된 θ'ᵢ를 평가 → loss_i

    외부 루프: 쿼리 손실의 합으로 θ 업데이트
    θ ← θ - β ∇_θ Σᵢ L(θ'ᵢ, query_i)
    (주의: 이는 그레이디언트의 그레이디언트를 계산해야 합니다)
```

### 6.3 핵심 통찰

MAML의 외부 루프 경사 하강은 특정 태스크에서의 성능이 아니라 **학습하는 능력** 자체를 최적화합니다. 결과적으로 얻어진 초기화 θ*는 파라미터 공간에서 **몇 번의 경사 하강 스텝만으로 다양한 태스크에 도달할 수 있는** 위치에 놓이게 됩니다.

### 6.4 메트릭 방법과의 비교

| 특징 | ProtoNet | MAML |
|------|----------|------|
| 접근법 | 임베딩 학습 후 가장 가까운 프로토타입 | 초기화 학습 후 파인튜닝 |
| 테스트 시 계산량 | 순전파만 | 순전파 + 역전파 |
| 유연성 | 고정된 거리 함수 | 모델 전체 적응 |
| 계산 비용 | 낮음 | 높음 (2차 미분 필요) |
| 임의 모델에 적용 | 특정 구조 필요 | 예 (모델 무관) |

---

## 7. 관계 네트워크

관계 네트워크(Relation Networks, Sung et al., 2018)는 고정된 메트릭을 사용하는 대신 거리 함수 자체를 학습합니다:

```
┌─────────┐    ┌───────────┐
│ 서포트   │───►│  임베딩   │───┐
│  예제   │    │  모듈     │   │ 연결(Concatenate) ┌────────────┐
└─────────┘    └───────────┘   ├──────────────────►│  관계      │──► 점수
┌─────────┐    ┌───────────┐   │                   │  모듈      │   (0~1)
│  쿼리   │───►│  임베딩   │───┘                   │  (학습됨)  │
│  예제   │    │  모듈     │                        └────────────┘
└─────────┘    └───────────┘
```

관계 모듈(relation module)은 연결된 특징 맵을 입력으로 받아 유사도 점수를 출력하는 소형 CNN입니다. 이를 통해 유클리드 거리나 코사인 거리로는 포착하기 어려운 복잡한 비선형 유사도 측도를 학습할 수 있습니다.

---

## 8. 실전 고려사항

### 8.1 방법 선택 가이드

| 상황 | 권장 방법 |
|------|-----------|
| 단순하고 빠른 추론이 필요한 경우 | 프로토타입 네트워크 |
| 최대 유연성이 필요한 경우 | MAML |
| 극히 적은 예제 (1-shot) | Matching Networks 또는 MAML |
| 도메인 특화 유사도 | Relation Networks |
| 대규모 사전 훈련 백본 사용 가능 | ProtoNet 헤드로 파인튜닝 |

### 8.2 데이터 증강(Data Augmentation)

예제가 1~5개뿐이므로 증강이 매우 중요합니다:
- 랜덤 크롭, 뒤집기, 회전
- 색상 지터(Color jitter)
- 컷아웃(Cutout) / 랜덤 이레이싱(Random erasing)
- 같은 클래스 서포트 예제 간 믹스업(Mixup)

### 8.3 백본 선택

현대 퓨샷 학습은 종종 사전 훈련된 백본을 사용합니다:

| 백본 | 파라미터 수 | 5-way 5-shot 정확도 (miniImageNet) |
|------|------------|-------------------------------------|
| Conv4 (4층 CNN) | 113K | ~65% |
| ResNet-12 | 12M | ~76% |
| WRN-28-10 | 36M | ~80% |
| ViT-Small (사전 훈련) | 22M | ~85% |

### 8.4 벤치마크

| 데이터셋 | 클래스 수 | 이미지 수 | 이미지 크기 | 태스크 |
|---------|----------|----------|------------|--------|
| Omniglot | 1,623 문자 | 32K | 28×28 | 손글씨 문자 |
| miniImageNet | 100 클래스 | 60K | 84×84 | 자연 이미지 |
| tieredImageNet | 608 클래스 | 779K | 84×84 | 계층적 분할 |
| CUB-200 | 조류 200종 | 12K | 84×84 | 세분화 분류 |
| Meta-Dataset | 다중 도메인 | 다양 | 다양 | 크로스 도메인 |

---

## 9. 연습문제

### 연습문제 1: 에피소드 구성

CIFAR-100(100개 클래스, 클래스당 훈련 이미지 500장)에서 5-way 5-shot 에피소드를 생성하는 함수를 작성하세요. 함수는 다음을 수행해야 합니다:
1. 무작위로 5개 클래스 선택
2. 각 클래스에서 무작위로 서포트 5장, 쿼리 15장 선택
3. 레이블을 0~4로 재인덱싱하여 올바른 형태의 텐서로 반환

### 연습문제 2: 프로토타입 네트워크 학습

완전한 프로토타입 네트워크 훈련 루프를 구현하세요:
1. Conv4 백본 사용 (컨브 블록 4개, 필터 64개)
2. Omniglot에서 5-way 1-shot 분류 훈련
3. 600개 테스트 에피소드에 대한 정확도 보고
4. 유클리드 거리와 코사인 거리 비교

### 연습문제 3: MAML vs ProtoNet

간단한 합성 데이터셋에서 MAML과 프로토타입 네트워크를 비교하세요:
1. 20개 클래스에 대한 2D 가우시안 클러스터 생성
2. 10개 클래스에서 두 방법 메타 훈련
3. 나머지 10개 클래스에서 메타 테스트 (5-way 5-shot)
4. 정확도와 실제 훈련 시간 비교

### 연습문제 4: 데이터 증강의 영향

1-shot 정확도에 대한 데이터 증강의 효과를 측정하세요:
1. 증강 없이 ProtoNet 훈련
2. 랜덤 수평 뒤집기 + 색상 지터로 훈련
3. 모든 증강 적용(뒤집기, 회전, 컷아웃, 믹스업)하여 훈련
4. 각 단계에서의 정확도 향상을 보고

### 연습문제 5: 실제 응용

제조 결함 분류를 위한 퓨샷 학습 시스템을 설계하세요:
- 결함 유형 5가지, 각각 예제 이미지 3장만 존재
- 정상(비결함) 이미지는 무제한 제공
- 128×128 크기의 그레이스케일 이미지로 동작해야 함
- 다음을 기술하세요: 백본 선택, 훈련 전략, 평가 프로토콜

---

*레슨 43 끝*
