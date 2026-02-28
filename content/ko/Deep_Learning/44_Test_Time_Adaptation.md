[이전: 퓨샷 학습](./43_Few_Shot_Learning.md)

---

# 44. 테스트 타임 적응(Test-Time Adaptation)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 도메인 시프트(domain shift) 문제와 사전 훈련 모델이 분포가 달라진 데이터에서 성능이 저하되는 이유를 설명할 수 있다
2. 가장 단순한 형태의 테스트 타임 적응(test-time adaptation)으로서 배치 정규화(batch normalization) 적응을 설명할 수 있다
3. 추론 중 모델을 적응시키기 위한 TENT(Test-time Entropy Minimization)를 구현할 수 있다
4. TTA 접근법(BN 적응, TENT, TTT, CoTTA)을 비교하고 각각의 트레이드오프를 이해할 수 있다
5. 배포 환경에서 모델 강건성(robustness)을 향상시키기 위한 테스트 타임 적응 전략을 적용할 수 있다

---

## 목차

1. [도메인 시프트 문제](#1-도메인-시프트-문제)
2. [배치 정규화 적응](#2-배치-정규화-적응)
3. [TENT: 테스트 타임 엔트로피 최소화](#3-tent-테스트-타임-엔트로피-최소화)
4. [테스트 타임 훈련(TTT)](#4-테스트-타임-훈련ttt)
5. [연속 테스트 타임 적응(CoTTA)](#5-연속-테스트-타임-적응cotta)
6. [실전 배포](#6-실전-배포)
7. [연습문제](#7-연습문제)

---

## 1. 도메인 시프트 문제

### 1.1 도메인 시프트란?

소스 데이터(예: 카메라로 촬영한 사진)로 훈련된 모델은 다른 분포의 타겟 데이터(예: 스케치, 손상된 이미지, 다른 병원의 데이터)에서 성능이 크게 저하되는 경우가 많습니다:

```
소스 도메인 (훈련):                  타겟 도메인 (배포):

┌──────────────┐                      ┌──────────────┐
│  실험실의    │  모델 정확도          │  실세계의    │  같은 모델가
│  깨끗한 사진  │  95%                 │  손상된 이미지│  60%로 하락
│  ImageNet    │  ──────────────►     │              │
└──────────────┘                      └──────────────┘

도메인 시프트의 예:
  • 날씨: 맑음 → 안개/비
  • 장비: 카메라 A → 카메라 B
  • 스타일: 사진 → 그림/스케치
  • 손상: 깨끗함 → 노이즈/블러/압축
  • 기관: 병원 A → 병원 B
```

### 1.2 표준 해결책과 한계

| 접근법 | 가능 시점 | 한계 |
|--------|----------|------|
| 도메인 적응(Domain adaptation) | 배포 전 (타겟 데이터 필요) | 타겟 데이터를 미리 확보하지 못할 수 있음 |
| 데이터 증강 | 훈련 중 | 모든 가능한 시프트를 예측할 수 없음 |
| 강건한 아키텍처 | 훈련 중 | 도움이 되지만 격차를 완전히 없애지 못함 |
| **테스트 타임 적응(TTA)** | **추론 중** | **즉석 적응 — 타겟 레이블 불필요** |

TTA는 고유하게도 적응 과정에서 **레이블이 있는 타겟 데이터**도, **소스 데이터에 대한 접근**도 필요하지 않습니다. 모델과 들어오는 테스트 배치만으로 동작합니다.

---

## 2. 배치 정규화 적응

### 2.1 BN 통계량이 중요한 이유

배치 정규화(Batch Normalization)는 훈련 중 계산된 이동 평균(μ)과 분산(σ²)을 저장합니다. 테스트 시에는 이 저장된 통계량으로 입력을 정규화합니다. 그러나 테스트 분포가 훈련 분포와 다르면 이 통계량이 맞지 않습니다.

```python
# 테스트 시 표준 BatchNorm:
# 훈련 중 저장된 running_mean과 running_var를 사용
output = (input - running_mean) / sqrt(running_var + eps) * gamma + beta

# 문제: running_mean과 running_var는 소스 도메인을 나타냄
# 입력이 다른 도메인에서 올 경우, 정규화가 제대로 되지 않음
```

### 2.2 BN Adapt: 테스트 통계량으로 교체

가장 단순한 TTA 방법: 저장된 BN 통계량을 현재 테스트 배치의 통계량으로 교체합니다.

```python
import torch
import torch.nn as nn


def adapt_bn(model, test_loader, num_batches=10):
    """배치 정규화 통계량을 테스트 분포에 적응시킵니다.

    테스트 데이터로 순전파를 실행하여 이동 평균/분산을 업데이트한 후,
    추론을 위해 고정합니다.
    """
    # BN 통계량 초기화
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            module.train()  # 배치 통계량 사용
            module.momentum = None  # 누적 이동 평균 사용

    # 테스트 타임 통계량 수집
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            # 순전파가 BN 이동 통계량을 업데이트
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()
            model(inputs)

    # BN을 eval 모드로 고정
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    return model
```

### 2.3 효과

| 방법 | 깨끗한 이미지 정확도 | 손상된 이미지 정확도 (평균) |
|------|---------------------|----------------------------|
| 표준 (적응 없음) | 76.1% | 43.5% |
| BN Adapt | 76.1% | 57.2% |

BN 적응은 추가 훈련 없이 무료이고, 레이블이 필요 없으며, 손상으로 인한 정확도 손실의 상당 부분을 복구합니다. 한계점: 정규화 통계량만 수정할 뿐, 모델 가중치는 변경하지 않습니다.

---

## 3. TENT: 테스트 타임 엔트로피 최소화

### 3.1 아이디어

TENT(Wang et al., 2021)는 BN 통계량을 넘어 테스트 시 실제로 모델 파라미터를 업데이트합니다. 핵심: 테스트 데이터에 대한 모델 예측의 **엔트로피(entropy)**를 최소화합니다.

```
높은 엔트로피 예측 (불확실):          낮은 엔트로피 예측 (확신):
  고양이: 0.25                          고양이: 0.85
  개:     0.25                          개:     0.10
  새:     0.25                          새:     0.03
  물고기: 0.25                          물고기: 0.02

TENT는 엔트로피를 최소화 → 모델이 확신 있는 예측을 하도록 유도
모델 아키텍처가 좋다면, 확신 있는 예측 = 올바른 예측 (대부분의 경우)
```

### 3.2 알고리즘

```
각 테스트 배치에 대해:
  1. 순전파 → 예측 확률 p 계산
  2. 엔트로피 계산: H(p) = -Σ p_i log(p_i)
  3. 엔트로피 손실 역전파
  4. BatchNorm 레이어의 어파인(affine) 파라미터 (γ, β)만 업데이트
  5. 업데이트된 모델로 예측 수행
```

### 3.3 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


class TENT:
    """테스트 타임 엔트로피 최소화 (TENT).

    배치 정규화 어파인 파라미터만 업데이트하여
    예측의 엔트로피를 최소화함으로써 사전 훈련된 모델을
    테스트 시에 적응시킵니다.
    """

    def __init__(self, model, lr=0.001, steps=1):
        self.model = model
        self.steps = steps

        # BN 어파인 파라미터 (γ와 β)만 수집
        self.params = []
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.requires_grad_(True)
                self.params.extend([module.weight, module.bias])
            else:
                # 다른 모든 파라미터는 고정
                for param in module.parameters(recurse=False):
                    param.requires_grad_(False)

        self.optimizer = SGD(self.params, lr=lr, momentum=0.9)

    def adapt_and_predict(self, inputs):
        """입력 배치에 대해 모델을 적응시킨 후 예측합니다."""
        self.model.train()  # BN이 배치 통계량 사용

        for _ in range(self.steps):
            outputs = self.model(inputs)
            loss = self._entropy_loss(outputs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # 적응된 파라미터로 최종 예측
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs)

    @staticmethod
    def _entropy_loss(logits):
        """소프트맥스 확률의 섀넌 엔트로피(Shannon entropy), 배치 평균."""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy.mean()
```

### 3.4 BN 파라미터만 업데이트하는 이유

모든 파라미터를 업데이트하면 테스트 배치에 과적합됩니다(레이블이 없는 경우 특히 심각). BN 어파인 파라미터(γ, β)는 최소한의 저위험 집합입니다:
- 정규화된 특징의 스케일과 시프트를 제어합니다
- 어떤 특징을 감지할지를 변경하지 않고 분포 시프트를 보완할 수 있습니다
- BN 레이어당 파라미터 2개뿐 (컨브 레이어의 수천 개 대비)

### 3.5 ImageNet-C에서의 결과

| 방법 | 평균 손상 오류율 (%) |
|------|---------------------|
| 표준 | 57.2 |
| BN Adapt | 40.5 |
| TENT (1 스텝) | 38.0 |
| TENT (10 스텝) | 36.2 |

---

## 4. 테스트 타임 훈련(TTT)

### 4.1 접근법

TTT(Sun et al., 2020)는 훈련 중 자기 지도 보조 태스크(예: 이미지 회전 예측)를 추가합니다. 테스트 시에는 이 보조 태스크를 사용하여 테스트 입력으로 공유 인코더를 파인튜닝합니다.

```
훈련:
  ┌─────────┐     ┌────────────┐     ┌───────────────┐
  │  입력   │────►│   공유     │────►│  메인 태스크  │──► 분류 손실
  │  이미지  │     │  인코더   │     │  헤드         │
  └─────────┘     └─────┬──────┘     └───────────────┘
                        │
                        └────────────►┌───────────────┐
                                      │  회전         │──► 자기 지도
                                      │  예측         │     손실
                                      └───────────────┘

테스트 시:
  1. 테스트 이미지 수신
  2. 회전 버전 생성 (0°, 90°, 180°, 270°)
  3. 회전 예측 태스크로 인코더 파인튜닝
  4. 적응된 인코더로 메인 분류 수행
```

### 4.2 핵심 통찰

회전 예측 태스크는 레이블이 필요 없지만, 특징 추출기(feature extractor)가 테스트 데이터 분포에 적응하도록 학습 신호를 제공합니다. 가정: 인코더가 테스트 데이터에서 회전을 올바르게 예측할 수 있다면, 그 특징이 해당 분포에 잘 보정(calibrated)되어 있다는 의미입니다.

---

## 5. 연속 테스트 타임 적응(CoTTA)

### 5.1 과제: 연속적인 분포 변화

배포 환경에서는 분포가 지속적으로 변화할 수 있습니다(예: 하루 동안 날씨가 변하는 상황). TENT를 일련의 다른 분포에 단순히 적용하면 다음 문제가 발생합니다:
- **오류 누적(Error accumulation)**: 잘못된 의사 레이블이 시간이 지남에 따라 복합적으로 증가
- **치명적 망각(Catastrophic forgetting)**: 모델이 소스 도메인의 지식을 잊어버림

### 5.2 CoTTA의 해결책

CoTTA(Wang et al., 2022)는 두 가지 메커니즘으로 이 문제를 해결합니다:

1. **가중 평균 의사 레이블(Weight-averaged pseudo-labels)**: 지수 이동 평균(EMA, Exponential Moving Average) 교사 모델을 사용하여 더 안정적인 의사 레이블 생성
2. **확률적 복원(Stochastic restore)**: 매 스텝마다 일부 파라미터를 소스 값으로 무작위 복원하여 드리프트(drift) 방지

```python
class CoTTA:
    """연속 테스트 타임 적응 (CoTTA).

    안정적인 의사 레이블을 위해 EMA 교사 모델을 사용하고,
    치명적 망각을 방지하기 위해 확률적 복원을 적용합니다.
    """

    def __init__(self, model, lr=0.001, ema_decay=0.999, restore_prob=0.01):
        self.student = model
        self.teacher = self._copy_model(model)  # EMA 교사 모델
        self.source_params = {n: p.clone() for n, p in model.named_parameters()}
        self.ema_decay = ema_decay
        self.restore_prob = restore_prob

        # BN 파라미터만 적응
        params = [p for n, p in model.named_parameters()
                  if 'bn' in n or 'norm' in n]
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def adapt(self, inputs):
        # EMA 교사 모델에서 의사 레이블 획득
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            pseudo_labels = teacher_logits.argmax(dim=1)
            confidence = F.softmax(teacher_logits, dim=1).max(dim=1)[0]

        # 확신 있는 의사 레이블로 학생 모델 훈련
        self.student.train()
        mask = confidence > 0.9  # 확신 있는 예측만 사용
        if mask.sum() > 0:
            student_logits = self.student(inputs[mask])
            loss = F.cross_entropy(student_logits, pseudo_labels[mask])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # EMA 교사 업데이트
        self._update_teacher()

        # 확률적 복원
        self._stochastic_restore()

        return self.student(inputs)

    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(),
                                     self.student.parameters()):
            t_param.data = self.ema_decay * t_param.data + \
                          (1 - self.ema_decay) * s_param.data

    def _stochastic_restore(self):
        for name, param in self.student.named_parameters():
            if name in self.source_params:
                mask = torch.rand_like(param) < self.restore_prob
                param.data[mask] = self.source_params[name][mask]

    @staticmethod
    def _copy_model(model):
        import copy
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher
```

---

## 6. 실전 배포

### 6.1 TTA 사용 시점

| 상황 | 권장 접근법 |
|------|------------|
| 알려진 고정 시프트 (새 카메라) | BN Adapt (가장 단순하고 신뢰할 수 있음) |
| 알 수 없는 손상 (날씨, 노이즈) | TENT (즉석 적응) |
| 점진적으로 변화하는 분포 | CoTTA (연속 시프트 처리) |
| 단일 테스트 이미지 (배치 없음) | TTT 또는 인스턴스 수준 BN 적응 |
| 안전 필수 애플리케이션 | BN Adapt만 (가장 보수적) |

### 6.2 배치 크기 민감도

TTA 방법은 배치 통계량에 의존합니다. 매우 작은 배치(1~8개)는 BN 통계량을 불안정하게 만듭니다:

| 배치 크기 | BN Adapt 개선 | TENT 개선 |
|----------|--------------|-----------|
| 1 | 성능 저하 가능 | 불안정 |
| 16 | 보통 | 양호 |
| 64 | 양호 | 최적 |
| 200 | 우수 | 우수 |

단일 이미지 추론의 경우, 인스턴스 정규화(instance normalization) 또는 특징 수준 적응을 고려하세요.

### 6.3 계산 비용

| 방법 | 추가 순전파 | 추가 역전파 | 메모리 오버헤드 |
|------|-----------|-----------|---------------|
| BN Adapt | N 배치 × 1 | 0 | 무시할 수 있음 |
| TENT (1 스텝) | 배치당 1 | 배치당 1 | +BN 파라미터 그레이디언트 |
| TTT | K 회전 | K 회전 | +보조 헤드 |
| CoTTA | 1 (교사) + 1 (학생) | 1 | +교사 모델 |

---

## 7. 연습문제

### 연습문제 1: 분포 시프트 분석

사전 훈련된 ResNet-50을 다음 조건의 ImageNet 검증 세트에서 평가하세요:
1. 손상 없음 (기준선)
2. 가우시안 노이즈 (σ = 0.1, 0.3, 0.5)
3. 모션 블러 (커널 크기 5, 15, 25)
4. JPEG 압축 (품질 10, 30, 50)

손상 심각도에 따른 정확도를 그래프로 그리세요. 어떤 손상 유형이 가장 큰 정확도 하락을 유발합니까?

### 연습문제 2: BN 적응

BN 적응을 구현하고 연습문제 1의 손상에 테스트하세요:
1. 손상된 데이터 10, 50, 100 배치를 사용하여 적응
2. 적응 전후의 정확도 비교
3. 안정적인 적응을 위해 필요한 최소 배치 수는 얼마입니까?

### 연습문제 3: TENT 구현

TENT를 처음부터 구현하세요:
1. BN 어파인 파라미터만 그레이디언트 추적을 활성화한 사전 훈련 모델 설정
2. 엔트로피 손실 구현
3. 1, 3, 10번의 적응 스텝으로 테스트
4. BN Adapt와 비교: TENT가 추가적인 이점을 제공하는 경우는 언제입니까?

### 연습문제 4: 배치 크기 연구

배치 크기가 TTA 품질에 미치는 영향을 연구하세요:
1. 배치 크기 1, 4, 16, 64, 256으로 TENT 실행
2. 배치 크기 1의 경우, BatchNorm을 인스턴스 정규화로 교체 시도
3. 배치 크기에 따른 정확도를 그래프로 그리고 추세를 설명하세요

### 연습문제 5: 연속 적응

연속 시프트 시나리오를 시뮬레이션하세요:
1. 10가지 서로 다른 손상 유형 시퀀스를 생성하고, 각각 100 배치씩 적용
2. TENT를 단순 적용(오류가 누적됨)하여 시간에 따른 정확도 측정
3. CoTTA(EMA 교사 모델 + 확률적 복원)를 적용하여 비교
4. CoTTA가 단순 TENT에서 발생하는 성능 저하를 방지함을 보이세요

---

*레슨 44 끝*
