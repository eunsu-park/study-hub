# 신경망 기초 (Neural Network Fundamentals) 학습 가이드

## 개요

신경망은 현대 딥러닝의 핵심 계산 구조입니다. 이 토픽은 고전 머신러닝과 딥러닝 사이의 간극을 메우며, 신경망이 어떻게 작동하는지 밑바닥부터 체계적으로 이해할 수 있도록 구성되어 있습니다. 생물학적 영감과 퍼셉트론에서 시작하여, 활성화 함수, 순전파 구조, 역전파, 최적화, 정규화를 거쳐 -- NumPy만으로 완전한 MLP를 구현하는 것으로 마무리합니다.

**선수 과목**: [Machine Learning](../Machine_Learning/00_Overview.md), [Linear Algebra](../Linear_Algebra/00_Overview.md)

---

## 학습 로드맵

```
생물학적 뉴런 → 퍼셉트론 → 활성화 함수 → 순전파 신경망
                                              ↓
보편 근사 정리 ← 배치 정규화 ← 정규화 ← 가중치 초기화
        ↓                                     ↑
학습 파이프라인 → MLP 밑바닥 구현       손실 함수 → 역전파
        ↓                                     → 경사 하강법 ──┘
기초에서 딥러닝으로
```

---

## 파일 목록

| 파일 | 주제 | 핵심 내용 |
|------|------|----------|
| [01_Biological_to_Artificial_Neurons.md](./01_Biological_to_Artificial_Neurons.md) | 생물학적 뉴런에서 인공 뉴런으로 | McCulloch-Pitts 모델, 역사적 연대기, 뉴런 해부학 |
| [02_Perceptron_and_Linear_Classifiers.md](./02_Perceptron_and_Linear_Classifiers.md) | 퍼셉트론과 선형 분류기 | 퍼셉트론 학습 규칙, 수렴 정리, XOR 문제 |
| [03_Activation_Functions.md](./03_Activation_Functions.md) | 활성화 함수 | Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax, 선택 가이드 |
| [04_Feedforward_Networks.md](./04_Feedforward_Networks.md) | 순전파 신경망 | MLP 구조, 행렬 연산, 순전파 구현 |
| [05_Loss_Functions.md](./05_Loss_Functions.md) | 손실 함수 | MSE, 교차 엔트로피, Hinge 손실, 선택 가이드 |
| [06_Backpropagation.md](./06_Backpropagation.md) | 역전파 | 연쇄 법칙, 계산 그래프, 기울기 유도 |
| [07_Gradient_Descent_Variants.md](./07_Gradient_Descent_Variants.md) | 경사 하강법 변형 | SGD, Momentum, RMSProp, Adam, 학습률 스케줄링 |
| [08_Weight_Initialization.md](./08_Weight_Initialization.md) | 가중치 초기화 | Xavier/Glorot, He/Kaiming, 대칭 깨뜨리기 |
| [09_Regularization.md](./09_Regularization.md) | 정규화 | L1/L2, Dropout, 조기 종료, 데이터 증강 |
| [10_Batch_Normalization.md](./10_Batch_Normalization.md) | 배치 정규화 | 내부 공변량 이동, BN 알고리즘, 추론 모드 |
| [11_Universal_Approximation.md](./11_Universal_Approximation.md) | 보편 근사 정리 | 이론, 시각화, 실질적 한계 |
| [12_Training_Pipeline.md](./12_Training_Pipeline.md) | 학습 파이프라인 | 데이터 분할, 검증, 하이퍼파라미터 튜닝 |
| [13_Building_MLP_from_Scratch.md](./13_Building_MLP_from_Scratch.md) | MLP 밑바닥 구현 | NumPy로 완전한 MLP, 모듈식 레이어 설계 |
| [14_From_Fundamentals_to_Deep_Learning.md](./14_From_Fundamentals_to_Deep_Learning.md) | 기초에서 딥러닝으로 | CNN, RNN, Transformer 미리보기, 다음 단계 |

---

## 환경 설정

### 필요 라이브러리 설치

```bash
pip install numpy matplotlib
```

### 버전 확인

```python
import numpy as np
import matplotlib

print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
```

### 권장 버전
- Python: 3.9+
- NumPy: 1.24+
- Matplotlib: 3.7+

---

## 권장 학습 순서

### 1단계: 기초 (01-03)
- 신경망의 생물학적 영감 이해
- 퍼셉트론과 그 한계 파악
- 활성화 함수와 그 특성 학습

### 2단계: 구조와 학습 (04-07)
- 순전파 신경망 구조 이해
- 손실 함수와 역전파 마스터
- 경사 하강법 최적화 변형 학습

### 3단계: 학습 모범 사례 (08-10)
- 적절한 가중치 초기화 전략
- 과적합 방지를 위한 정규화 기법
- 안정적 학습을 위한 배치 정규화

### 4단계: 이론과 실습 (11-13)
- 보편 근사 정리 이해
- 완전한 학습 파이프라인 구축
- 밑바닥부터 완전한 MLP 구현

### 5단계: 딥러닝으로의 전환 (14)
- CNN, RNN, Transformer 구조 미리보기
- 다음 학습 단계 안내

---

## 이 토픽의 위치

```
Machine Learning (Tier 2)
    │
    ├── 고전 ML 알고리즘 (sklearn)
    │
    ▼
Neural Network Fundamentals (Tier 2)  ◄── 현재 위치
    │
    ├── 신경망의 작동 원리 (밑바닥 구현)
    ├── 역전파와 최적화
    ├── NumPy 전용 구현
    │
    ▼
Deep Learning (Tier 3)
    │
    ├── CNN, RNN, Transformer (PyTorch)
    ├── 고급 아키텍처
    └── GPU 대규모 학습
```

---

## 참고자료

### 교재
- "Neural Networks and Deep Learning" - Michael Nielsen (무료 온라인)
- "Deep Learning" - Goodfellow, Bengio, Courville ("DL 바이블")
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### 온라인 자료
- [3Blue1Brown 신경망 시리즈](https://www.3blue1brown.com/topics/neural-networks)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/)
- [Michael Nielsen의 Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
