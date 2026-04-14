# MLP 밑바닥 구현

**이전**: [학습 파이프라인](./12_Training_Pipeline.md) | **다음**: [기초에서 딥러닝으로](./14_From_Fundamentals_to_Deep_Learning.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 설정 가능한 층과 활성화를 가진 모듈식 MLP 클래스를 설계할 수 있습니다
2. 순전파, 역전파, 가중치 업데이트를 단일 클래스에 구현할 수 있습니다
3. He 초기화, 배치 정규화, 드롭아웃을 함께 적용할 수 있습니다
4. 실제 데이터셋에서 MLP를 학습시키고 경쟁력 있는 정확도를 달성할 수 있습니다
5. 셔플링과 진행 로깅이 포함된 미니배치 학습을 구현할 수 있습니다
6. 적응적 학습을 위해 Adam 옵티마이저를 사용할 수 있습니다
7. 학습 안정성을 위한 기울기 클리핑을 추가할 수 있습니다
8. 기울기 검증으로 일반적인 구현 문제를 디버그할 수 있습니다

---

이것은 캡스톤 레슨입니다: 레슨 01-12의 모든 내용을 합쳐 NumPy만으로 완전하고 작동하는 다층 퍼셉트론을 구축합니다. PyTorch도, TensorFlow도 없이 -- 오직 행렬 연산, 역전파, 최적화만으로. 모든 구성 요소를 직접 구현함으로써, 신경망 프레임워크 내부에서 무슨 일이 일어나는지 깊이 이해하게 될 것입니다.

---

## 1. 구조 개요

```
┌────────────────────────────────────────────────────┐
│                   MLP 클래스                        │
│                                                    │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐   │
│  │  레이어 1 │→│  레이어 2 │→...→│  레이어 L │   │
│  │ Linear    │  │ Linear    │     │ Linear    │   │
│  │ BatchNorm │  │ BatchNorm │     │ (BN 없음) │   │
│  │ ReLU      │  │ ReLU      │     │ Softmax   │   │
│  │ Dropout   │  │ Dropout   │     │(Drop 없음)│   │
│  └──────────┘  └──────────┘       └──────────┘   │
│                                                    │
│  옵티마이저: Adam                                   │
│  손실: 교차 엔트로피                                 │
│  초기화: He 정규                                    │
└────────────────────────────────────────────────────┘
```

---

## 2. 구성 요소

### 2.1 선형 레이어

```python
class Linear:
    """He 초기화를 가진 완전 연결 레이어."""

    def __init__(self, fan_in, fan_out):
        self.W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((fan_out, 1))

    def forward(self, a_prev):
        self.a_prev = a_prev
        return self.W @ a_prev + self.b

    def backward(self, dz):
        m = self.a_prev.shape[1]
        self.dW = (1 / m) * dz @ self.a_prev.T
        self.db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        return self.W.T @ dz
```

### 2.2 활성화 함수, 배치 정규화, 드롭아웃

각 구성 요소는 `forward()`와 `backward()` 메서드를 가진 모듈로 구현합니다.

---

## 3. MLP 클래스 주요 기능

- **설정 가능한 구조**: 임의의 깊이와 너비
- **He 초기화**: ReLU 네트워크용
- **배치 정규화**: 안정적 학습
- **드롭아웃**: 정규화
- **Adam 옵티마이저**: 적응적 학습률
- **교차 엔트로피 손실**: 소프트맥스 출력
- **기울기 클리핑**: 학습 안정성

---

## 4. 학습/추론 모드 전환

```python
model.set_training(True)    # 학습 모드: BN은 배치 통계, 드롭아웃 활성
model.set_training(False)   # 추론 모드: BN은 이동 통계, 드롭아웃 비활성
```

검증/테스트 시 반드시 추론 모드로 전환해야 합니다.

---

## 5. 전체 학습 루프

```python
for epoch in range(n_epochs):
    model.set_training(True)
    
    # 학습 데이터 셔플
    # 미니배치 순회
    for batch in mini_batches:
        y_pred = model.forward(X_batch)      # 순전파
        loss = cross_entropy(y_pred, Y_batch) # 손실 계산
        model.backward(Y_batch)               # 역전파
        optimizer.step(params, grads)          # 가중치 업데이트
    
    # 검증
    model.set_training(False)
    val_pred = model.forward(X_val)
    val_loss = cross_entropy(val_pred, Y_val)
    val_acc = accuracy(val_pred, Y_val)
```

---

## 6. 디버깅 팁

### 6.1 기울기 검증

분석적 기울기를 수치 기울기와 비교합니다:

```
수치적: ∂L/∂θ_i ≈ [L(θ_i + ε) - L(θ_i - ε)] / (2ε)

상대 오차 < 1e-4 → 올바를 가능성 높음
상대 오차 > 1e-3 → 버그 가능성 높음
```

### 6.2 일반적인 버그

```
버그                             │ 증상                    │ 수정
────────────────────────────────┼────────────────────────┼──────────
행렬곱에서 전치 누락            │ 형상 불일치 오류        │ 모든 @ 연산 확인
mean/sum에서 잘못된 축          │ 기울기 검증 실패        │ axis=0 vs axis=1
기울기에서 1/m 누락             │ 학습률에 너무 민감      │ 배치 크기로 나누기
학습 중 BN이 평가 모드         │ 학습 손실 나쁨          │ set_training(True)
log(0) 클리핑 누락              │ NaN 손실               │ log(y + 1e-15)
```

---

## 7. 요약

```
핵심 정리
═══════════════════════════════════════════════════════
1. 모듈식 설계: Linear, BN, ReLU, Dropout, Softmax 블록
2. ReLU 네트워크를 위한 He 초기화
3. 순전파: Linear → BN → ReLU → Dropout (은닉층당)
4. 역전파: 역순, 각 블록에서 연쇄 법칙
5. 안정성을 위한 기울기 클리핑 + Adam 옵티마이저
6. 에포크마다 셔플링과 함께 미니배치 학습
7. 검증/테스트 시 항상 평가 모드로 전환
8. 기울기 검증으로 구현 버그를 조기에 포착
═══════════════════════════════════════════════════════
```

---

## 연습문제

1. MLP 클래스에 L2 정규화를 추가하고 과적합에 미치는 효과를 테스트하세요
2. 학습 루프에 학습률 스케줄링(코사인 어닐링)을 구현하세요
3. 실제 데이터셋(예: sklearn의 digits)에서 MLP를 학습시키고 테스트 정확도를 보고하세요
4. 모델 체크포인팅을 위한 `save()`와 `load()` 메서드를 추가하세요

---

**이전**: [학습 파이프라인](./12_Training_Pipeline.md) | **다음**: [기초에서 딥러닝으로](./14_From_Fundamentals_to_Deep_Learning.md)
