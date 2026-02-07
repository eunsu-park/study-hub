# 딥러닝 학습 가이드

## 소개

이 폴더는 딥러닝을 체계적으로 학습하기 위한 자료입니다. PyTorch를 주 프레임워크로 사용하며, 기초 개념은 NumPy로도 구현하여 알고리즘 원리를 깊이 이해할 수 있도록 구성했습니다.

**대상 독자**: Machine_Learning 폴더를 완료한 학습자

---

## 학습 로드맵

```
[기초]                    [CNN]                    [시퀀스]
   │                        │                         │
   ▼                        ▼                         ▼
텐서/오토그래드 ──▶ CNN 기초 ─────────▶ RNN 기초
   │                        │                         │
   ▼                        ▼                         ▼
신경망 기초 ────▶ CNN 심화 ─────────▶ LSTM/GRU
   │               (ResNet)                           │
   ▼                        │                         ▼
역전파 이해 ────▶ 전이학습 ─────────▶ Transformer
                                                      │
                [실전]                                ▼
                   │                            Attention
                   ▼
              학습 최적화
                   │
                   ▼
              모델 저장/배포
                   │
                   ▼
              실전 프로젝트
```

---

## 파일 목록

### 기초 (PyTorch + NumPy 비교)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [01_Tensors_and_Autograd.md](./01_Tensors_and_Autograd.md) | ⭐ | Tensor, 자동 미분, GPU | PyTorch + NumPy |
| [02_Neural_Network_Basics.md](./02_Neural_Network_Basics.md) | ⭐⭐ | 퍼셉트론, MLP, 활성화 함수 | PyTorch + NumPy |
| [03_Backpropagation.md](./03_Backpropagation.md) | ⭐⭐ | 체인 룰, 경사 하강법, 손실 함수 | PyTorch + NumPy |
| [04_Training_Techniques.md](./04_Training_Techniques.md) | ⭐⭐ | 옵티마이저, 배치, 정규화 | PyTorch + NumPy |

### CNN (PyTorch 중심, 기초는 NumPy 비교)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [05_CNN_Basics.md](./05_CNN_Basics.md) | ⭐⭐ | 합성곱, 풀링, 필터 | PyTorch + NumPy |
| [06_CNN_Advanced.md](./06_CNN_Advanced.md) | ⭐⭐⭐ | ResNet, VGG, EfficientNet | PyTorch only |
| [07_Transfer_Learning.md](./07_Transfer_Learning.md) | ⭐⭐⭐ | 사전학습 모델, Fine-tuning | PyTorch only |

### 시퀀스 모델 (PyTorch only)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [08_RNN_Basics.md](./08_RNN_Basics.md) | ⭐⭐⭐ | 순환 신경망, 기울기 소실 | PyTorch only |
| [09_LSTM_GRU.md](./09_LSTM_GRU.md) | ⭐⭐⭐ | 장기 의존성, 게이트 구조 | PyTorch only |
| [10_Attention_Transformer.md](./10_Attention_Transformer.md) | ⭐⭐⭐⭐ | Attention, Transformer | PyTorch only |

### 실전 (PyTorch only)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [11_Training_Optimization.md](./11_Training_Optimization.md) | ⭐⭐⭐ | AMP, 기울기 누적, 하이퍼파라미터 | PyTorch only |
| [12_Model_Saving_Deployment.md](./12_Model_Saving_Deployment.md) | ⭐⭐⭐ | TorchScript, ONNX, 양자화 | PyTorch only |
| [13_Practical_Image_Classification.md](./13_Practical_Image_Classification.md) | ⭐⭐⭐⭐ | CIFAR-10, 데이터 증강, Mixup | PyTorch only |
| [14_Practical_Text_Classification.md](./14_Practical_Text_Classification.md) | ⭐⭐⭐⭐ | 감성 분석, LSTM/Transformer | PyTorch only |

### 생성 모델 (PyTorch only)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [15_Generative_Models_GAN.md](./15_Generative_Models_GAN.md) | ⭐⭐⭐ | GAN, DCGAN, 손실 함수, StyleGAN | PyTorch only |
| [16_Generative_Models_VAE.md](./16_Generative_Models_VAE.md) | ⭐⭐⭐ | VAE, ELBO, Reparameterization, Beta-VAE | PyTorch only |
| [17_Diffusion_Models.md](./17_Diffusion_Models.md) | ⭐⭐⭐⭐ | DDPM, DDIM, U-Net, Stable Diffusion | PyTorch only |
| [18_Attention_Deep_Dive.md](./18_Attention_Deep_Dive.md) | ⭐⭐⭐⭐ | Flash Attention, Sparse Attention, RoPE, ALiBi | PyTorch only |

### 최신 기법 (PyTorch only)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [19_Vision_Transformer.md](./19_Vision_Transformer.md) | ⭐⭐⭐⭐ | ViT 아키텍처, Patch Embedding, CLS 토큰, DeiT, Swin Transformer | PyTorch only |
| [20_CLIP_Multimodal.md](./20_CLIP_Multimodal.md) | ⭐⭐⭐⭐ | CLIP 원리, Zero-shot Classification, Image-Text 매칭, BLIP | PyTorch only |
| [21_Self_Supervised_Learning.md](./21_Self_Supervised_Learning.md) | ⭐⭐⭐⭐ | SimCLR, MoCo, BYOL, MAE, Contrastive Learning | PyTorch only |
| [22_Reinforcement_Learning_Intro.md](./22_Reinforcement_Learning_Intro.md) | ⭐⭐⭐ | MDP 기초, Q-Learning 개념, Policy Gradient 개요 | PyTorch only |

### 객체 탐지/분할 (PyTorch only)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [23_Object_Detection.md](./23_Object_Detection.md) | ⭐⭐⭐⭐ | YOLO, Faster R-CNN, DETR, Mask R-CNN, SAM | PyTorch only |

### 시각화/도구 (PyTorch only)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [24_TensorBoard.md](./24_TensorBoard.md) | ⭐⭐ | 학습 시각화, 메트릭 로깅, 하이퍼파라미터 추적, 프로파일링 | PyTorch only |

---

## NumPy vs PyTorch 비교 가이드

| 주제 | NumPy 구현 | PyTorch 구현 | 비교 포인트 |
|------|-----------|--------------|------------|
| 텐서 | `np.array` | `torch.tensor` | GPU 지원, 자동 미분 |
| 순전파 | 직접 행렬 연산 | `nn.Module.forward` | 코드 간결성 |
| 역전파 | 수동 미분 계산 | `loss.backward()` | 자동 미분의 편리함 |
| 합성곱 | for 루프 구현 | `nn.Conv2d` | 성능 차이 |
| 최적화 | SGD 직접 구현 | `optim.SGD` | 고급 옵티마이저 |

### NumPy 구현의 장점
- 알고리즘의 수학적 원리 이해
- 행렬 연산 직관 향상
- 프레임워크 의존성 없이 개념 파악

### NumPy 구현이 어려운 시점
- CNN 심화 구조 (Skip Connection, Batch Norm)
- RNN/LSTM (복잡한 게이트 구조)
- Transformer (Multi-Head Attention)

---

## 선수 지식

- Python 기초 문법
- NumPy 배열 연산
- Machine_Learning 폴더 내용 (회귀, 분류 개념)
- 기초 미적분 (편미분, 체인 룰)
- 선형대수 기초 (행렬 곱셈)

---

## 환경 설정

### 필수 패키지

```bash
# PyTorch (CUDA 지원)
pip install torch torchvision torchaudio

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 기타 필수
pip install numpy matplotlib
```

### GPU 확인

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 추천 학습 순서

1. **기초 (1주)**: 01 → 02 → 03 → 04
   - NumPy와 PyTorch 코드 모두 실행하며 비교
2. **CNN (1주)**: 05 → 06 → 07
   - 05는 NumPy로 합성곱 직접 구현 추천
3. **시퀀스 (1-2주)**: 08 → 09 → 10
   - Transformer/Attention은 충분히 시간 투자
4. **실전 (1주)**: 11 → 12 → 13 → 14
   - 학습 최적화, 모델 저장, 프로젝트 실습
5. **생성 모델 (2주)**: 15 → 16 → 17 → 18
   - GAN/VAE/Diffusion 순서로 학습
   - Attention 심화는 LLM 학습 전 필수
6. **최신 기법 (1-2주)**: 19 → 20 → 21 → 22
   - ViT, CLIP 등 최신 아키텍처 학습
   - 강화학습 기초는 Reinforcement_Learning 폴더 선행 학습 권장

---

## 관련 자료

- [Machine_Learning/](../Machine_Learning/00_Overview.md) - 선수 과목
- [Python/](../Python/00_Overview.md) - 고급 Python
- [Data_Analysis/](../Data_Analysis/00_Overview.md) - NumPy/Pandas

---

## 참고 링크

- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n (CNN)](http://cs231n.stanford.edu/)
- [CS224n (NLP)](http://web.stanford.edu/class/cs224n/)
