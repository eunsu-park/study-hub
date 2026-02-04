# 10. Vision Transformer (ViT)

## 개요

Vision Transformer (ViT)는 Transformer 아키텍처를 이미지 분류에 적용한 모델입니다. 이미지를 패치로 분할하고, 각 패치를 토큰처럼 처리합니다. "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

---

## 수학적 배경

### 1. 이미지 패치화

```
입력 이미지: x ∈ R^(H × W × C)
패치 크기: P × P

패치 시퀀스:
x_p ∈ R^(N × P² × C)  where N = (H × W) / P²

예시:
- 이미지: 224 × 224 × 3
- 패치: 16 × 16
- N = (224 × 224) / (16 × 16) = 196 패치
- 각 패치: 16 × 16 × 3 = 768 차원
```

### 2. 패치 임베딩

```
Linear Projection:
z_0 = [x_class; x_p¹E; x_p²E; ...; x_pⁿE] + E_pos

여기서:
- x_class: 학습 가능한 [CLS] 토큰
- E ∈ R^(P²C × D): 패치 임베딩 행렬
- E_pos ∈ R^((N+1) × D): 위치 임베딩

z_0 ∈ R^((N+1) × D): 초기 임베딩 시퀀스
```

### 3. Transformer Encoder

```
Encoder block (L layers):

z'_l = MSA(LN(z_{l-1})) + z_{l-1}
z_l = MLP(LN(z'_l)) + z'_l

최종 출력:
y = LN(z_L⁰)  # [CLS] 토큰만 사용

여기서 z_L⁰는 L번째 레이어의 [CLS] 토큰
```

---

## ViT 아키텍처 변형

```
ViT-Base (B/16):
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- MLP size: 3072
- Patch size: 16
- Parameters: 86M

ViT-Large (L/16):
- Hidden size: 1024
- Layers: 24
- Attention heads: 16
- MLP size: 4096
- Patch size: 16
- Parameters: 307M

ViT-Huge (H/14):
- Hidden size: 1280
- Layers: 32
- Attention heads: 16
- MLP size: 5120
- Patch size: 14
- Parameters: 632M
```

---

## 파일 구조

```
10_ViT/
├── README.md
├── pytorch_lowlevel/
│   └── vit_lowlevel.py         # ViT 직접 구현
├── paper/
│   └── vit_paper.py            # 논문 재현
└── exercises/
    ├── 01_patch_embedding.md   # 패치 임베딩 시각화
    └── 02_attention_maps.md    # Attention 시각화
```

---

## 핵심 개념

### 1. CNN vs ViT

```
CNN:
- 지역적 수용 영역 (local receptive field)
- Inductive bias: locality, translation equivariance
- 작은 데이터셋에 유리

ViT:
- 전역 수용 영역 (global from start)
- 최소한의 inductive bias
- 대규모 데이터셋에 유리 (JFT-300M)
- 작은 데이터: pre-training 필요
```

### 2. Position Embedding

```
1D Learnable (ViT 기본):
- N+1개의 학습 가능한 벡터
- 순서 정보 학습

2D Positional (변형):
- (row, col) 별도 임베딩
- 이미지 구조 반영

Sinusoidal:
- 고정된 삼각 함수
- 외삽 가능성
```

### 3. [CLS] Token vs Global Average Pooling

```
[CLS] Token:
- 첫 번째 위치에 추가
- 전체 이미지 표현 집약
- BERT 스타일

Global Average Pooling:
- 모든 패치 평균
- CNN 스타일
- 비슷한 성능
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- F.linear, F.layer_norm 사용
- nn.TransformerEncoder 미사용
- 패치화 직접 구현

### Level 3: Paper Implementation (paper/)
- 논문 정확한 사양
- JFT/ImageNet pre-training
- Fine-tuning 코드

### Level 4: Code Analysis (별도)
- timm 라이브러리 분석
- HuggingFace ViT 분석

---

## 학습 체크리스트

- [ ] 패치 임베딩 수식 이해
- [ ] 위치 임베딩 역할
- [ ] [CLS] 토큰 역할
- [ ] CNN 대비 장단점
- [ ] Attention map 시각화
- [ ] Fine-tuning 전략

---

## 참고 자료

- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- [timm ViT](https://github.com/rwightman/pytorch-image-models)
- [../Deep_Learning/19_Vision_Transformer.md](../Deep_Learning/19_Vision_Transformer.md)
