# 딥러닝 학습 가이드

## 소개

이 폴더는 **딥러닝 이론과 실습**을 아우르는 종합 가이드를 제공하며, 개념 중심 레슨과 처음부터 구현하는 모델 실습을 결합합니다. PyTorch를 주요 프레임워크로 사용하며, 4단계 학습 접근법을 통합한 교육 과정을 따릅니다:

1. **레벨 1 (NumPy 스크래치)**: NumPy만 사용하여 모델을 구축하고 기본 메커니즘 이해
2. **레벨 2 (PyTorch 저수준)**: 고수준 API 없이 PyTorch 기본 요소(텐서, autograd)로 구현
3. **레벨 3 (논문 재현)**: 원논문을 읽고 핵심 아키텍처 재현
4. **레벨 4 (코드 분석)**: 프레임워크 및 연구 저장소의 프로덕션급 구현 분석

이 통합 접근법은 수학적 기초부터 실전 배포까지 딥러닝의 "왜"와 "어떻게"를 모두 이해할 수 있도록 합니다.

## 대상 독자

- **Machine_Learning** 폴더를 완료한 학습자
- Python, NumPy, 기본 ML 개념(경사 하강법, 과적합, train/test 분할)에 익숙한 독자
- 엄격하고 구현 중심의 딥러닝 교육을 원하는 모든 분

## 학습 로드맵

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  기초 과정   │────▶│     CNN      │────▶│  시퀀스 모델 │────▶│ 트랜스포머   │
│   L01-L06    │     │   L07-L12    │     │   L13-L15    │     │   L16-L22    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                        │
                                                                        ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  실전 응용   │◀────│  고급 주제   │◀────│  생성 모델   │◀────│  훈련 필수   │
│   L39-L42    │     │   L34-L38    │     │   L28-L33    │     │   L23-L27    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**권장 학습 경로**:
1. 기초 과정(L01-L06)으로 PyTorch 기본과 역전파 마스터
2. CNN(L07-L12)으로 컴퓨터 비전 기초 학습
3. 시퀀스 모델(L13-L15)로 시간적 데이터 처리
4. 트랜스포머(L16-L22)로 현대 NLP 및 비전의 핵심 학습
5. 훈련 필수(L23-L27)로 최적화, 손실 함수, 정규화 학습
6. 생성 모델(L28-L33)로 GAN, VAE, Diffusion 탐구
7. 고급 주제(L34-L38)로 멀티모달 학습 및 최신 아키텍처 학습
8. 실전 프로젝트(L39-L42)로 지식 적용

## 파일 목록

| 레슨 | 파일명 | 난이도 | 설명 |
|------|--------|--------|------|
| **블록 1: 기초 과정** |
| L01 | `01_Tensors_and_Autograd.md` | ⭐ | 텐서 연산, autograd, 계산 그래프 |
| L02 | `02_Neural_Network_Basics.md` | ⭐⭐ | 활성화 함수, 손실 함수, 순전파/역전파 |
| L03 | `03_Backpropagation.md` | ⭐⭐ | 연쇄 법칙, 그래디언트 흐름, 기울기 소실/폭발 |
| L04 | `04_Training_Techniques.md` | ⭐⭐ | 정규화, 드롭아웃, 배치 정규화, 조기 종료 |
| L05 | `05_Impl_Linear_Logistic.md` | ⭐⭐ | **구현**: 선형 및 로지스틱 회귀 처음부터 구현 |
| L06 | `06_Impl_MLP.md` | ⭐⭐ | **구현**: 다층 퍼셉트론(NumPy → PyTorch) |
| **블록 2: 합성곱 신경망** |
| L07 | `07_CNN_Basics.md` | ⭐⭐ | 합성곱, 풀링, 특징 맵, LeNet |
| L08 | `08_CNN_Advanced.md` | ⭐⭐⭐ | ResNet, Inception, 스킵 연결, 1x1 합성곱 |
| L09 | `09_Transfer_Learning.md` | ⭐⭐⭐ | 사전 훈련 모델, 파인튜닝, 도메인 적응 |
| L10 | `10_Impl_CNN_LeNet.md` | ⭐⭐⭐ | **구현**: MNIST에서 LeNet-5 구현 |
| L11 | `11_Impl_VGG.md` | ⭐⭐⭐ | **구현**: VGG-16 아키텍처 |
| L12 | `12_Impl_ResNet.md` | ⭐⭐⭐ | **구현**: 잔차 네트워크(ResNet-18/34) |
| **블록 3: 시퀀스 모델** |
| L13 | `13_RNN_Basics.md` | ⭐⭐⭐ | 순환 신경망, 은닉 상태, 시퀀스-투-시퀀스 |
| L14 | `14_LSTM_GRU.md` | ⭐⭐⭐ | 장단기 메모리(LSTM), 게이트 순환 유닛(GRU) |
| L15 | `15_Impl_LSTM_GRU.md` | ⭐⭐⭐ | **구현**: 텍스트 분류를 위한 LSTM/GRU |
| **블록 4: 어텐션과 트랜스포머** |
| L16 | `16_Attention_Transformer.md` | ⭐⭐⭐⭐ | 셀프 어텐션, 멀티헤드 어텐션, 트랜스포머 아키텍처 |
| L17 | `17_Attention_Deep_Dive.md` | ⭐⭐⭐⭐ | Query/Key/Value, 스케일된 내적, 위치 인코딩 |
| L18 | `18_Impl_Transformer.md` | ⭐⭐⭐⭐ | **구현**: 트랜스포머 인코더/디코더 처음부터 구현 |
| L19 | `19_Impl_BERT.md` | ⭐⭐⭐⭐ | **구현**: BERT 사전 훈련(마스크드 LM) |
| L20 | `20_Impl_GPT.md` | ⭐⭐⭐⭐ | **구현**: GPT 스타일 자기회귀 모델 |
| L21 | `21_Vision_Transformer.md` | ⭐⭐⭐⭐ | 패치 임베딩, ViT 아키텍처, DeiT |
| L22 | `22_Impl_ViT.md` | ⭐⭐⭐⭐ | **구현**: 이미지 분류를 위한 비전 트랜스포머 |
| **블록 5: 훈련 필수 요소** |
| L23 | `23_Training_Optimization.md` | ⭐⭐⭐ | 학습률 스케줄링, 그래디언트 클리핑, 혼합 정밀도 |
| L24 | `24_Loss_Functions.md` | ⭐⭐⭐ | 교차 엔트로피, 포컬 손실, 대조 손실, 트리플렛 손실 |
| L25 | `25_Optimizers.md` | ⭐⭐⭐ | SGD, Adam, AdamW, 학습률 워밍업 |
| L26 | `26_Normalization_Layers.md` | ⭐⭐⭐ | 배치 정규화, 레이어 정규화, 그룹 정규화, 인스턴스 정규화 |
| L27 | `27_TensorBoard.md` | ⭐⭐ | 로깅, 시각화, 하이퍼파라미터 추적 |
| **블록 6: 생성 모델** |
| L28 | `28_Generative_Models_GAN.md` | ⭐⭐⭐ | 생성자, 판별자, 적대적 훈련 |
| L29 | `29_Impl_GAN.md` | ⭐⭐⭐ | **구현**: 이미지 생성을 위한 DCGAN |
| L30 | `30_Generative_Models_VAE.md` | ⭐⭐⭐ | 변분 오토인코더, 잠재 공간, 재매개변수화 |
| L31 | `31_Impl_VAE.md` | ⭐⭐⭐ | **구현**: MNIST/CIFAR-10에서 VAE |
| L32 | `32_Diffusion_Models.md` | ⭐⭐⭐⭐ | DDPM, 스코어 기반 모델, 노이즈 제거 과정 |
| L33 | `33_Impl_Diffusion.md` | ⭐⭐⭐⭐ | **구현**: 노이즈 제거 확산 확률 모델 |
| **블록 7: 멀티모달 및 고급 주제** |
| L34 | `34_CLIP_Multimodal.md` | ⭐⭐⭐⭐ | 대조 학습, 비전-언어 모델 |
| L35 | `35_Impl_CLIP.md` | ⭐⭐⭐⭐ | **구현**: CLIP 스타일 이미지-텍스트 정렬 |
| L36 | `36_Self_Supervised_Learning.md` | ⭐⭐⭐⭐ | SimCLR, MoCo, BYOL, 대조 사전 훈련 |
| L37 | `37_Modern_Architectures.md` | ⭐⭐⭐⭐ | EfficientNet, ConvNeXt, Swin Transformer, NFNet |
| L38 | `38_Object_Detection.md` | ⭐⭐⭐⭐ | RCNN, YOLO, RetinaNet, DETR, 앵커 프리 방법 |
| **블록 8: 실전 응용 및 배포** |
| L39 | `39_Practical_Image_Classification.md` | ⭐⭐⭐⭐ | 엔드투엔드 프로젝트: 데이터셋, 훈련, 평가, 배포 |
| L40 | `40_Practical_Text_Classification.md` | ⭐⭐⭐⭐ | 엔드투엔드 NLP 프로젝트: 토큰화, 파인튜닝, 추론 |
| L41 | `41_Model_Saving_Deployment.md` | ⭐⭐⭐ | ONNX 내보내기, TorchScript, 모델 서빙(Flask, TorchServe) |
| L42 | `42_Reinforcement_Learning_Intro.md` | ⭐⭐⭐ | DQN 기초, 정책 그래디언트, RL 주제로의 연결 |
| L43 | `43_Few_Shot_Learning.md` | ⭐⭐⭐⭐ | 메타 학습, 프로토타입 네트워크, MAML |
| L44 | `44_Test_Time_Adaptation.md` | ⭐⭐⭐⭐ | 배치 정규화 적응, TENT, CoTTA, 분포 변화 |

**총 44개 레슨** (개념 레슨 30개 + 구현 레슨 14개)

## 구현 철학: 4단계 접근법

이 커리큘럼은 **4단계 진행**을 통해 이론과 실전 코딩을 통합합니다:

| 레벨 | 설명 | 도구 | 예시 레슨 |
|------|------|------|-----------|
| **L1: NumPy 스크래치** | NumPy만 사용하여 모델 구축(PyTorch `nn.Module` 없음). 순전파/역전파를 수동으로 구현. | NumPy 배열, 수동 그래디언트 계산 | L05, L06 |
| **L2: PyTorch 저수준** | PyTorch 텐서와 autograd 사용, 단 `nn.Linear`, `nn.Conv2d` 등은 사용 안 함. 커스텀 모듈 정의. | `torch.Tensor`, `autograd`, 커스텀 `nn.Module` | L10, L11, L15 |
| **L3: 논문 재현** | 원논문(Attention Is All You Need, BERT 등) 읽고 아키텍처 재현. | PyTorch, 논문 의사코드 | L18, L19, L20, L22 |
| **L4: 코드 분석** | 프로덕션 구현(Hugging Face Transformers, torchvision models) 연구 및 디자인 패턴 이해. | GitHub 저장소, 라이브러리 소스 코드 | L37, L38, L41 |

**왜 이 접근법인가?**
- **L1**은 수학을 이해하도록 보장("마법" 라이브러리 없음)
- **L2**는 저수준 제어를 유지하면서 PyTorch 관용구를 가르침
- **L3**은 학술 논문과 코드를 연결
- **L4**는 실무 ML 엔지니어링 준비

## 선수 지식

- **프로그래밍**: Python 숙련도(함수, 클래스, 리스트 컴프리헨션)
- **수학**: 선형대수(행렬 곱셈, 고유값), 미적분(도함수, 연쇄 법칙), 기초 확률
- **머신러닝**: 지도 학습, 손실 함수, 경사 하강법 친숙도(`Machine_Learning` 폴더 참조)
- **라이브러리**: NumPy 기초(배열 인덱싱, 브로드캐스팅)

## 환경 설정

### 설치
```bash
# PyTorch 설치 (CPU 버전)
pip install torch torchvision matplotlib numpy

# GPU 지원 (CUDA 11.8 예시)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 선택 사항: 시각화를 위한 TensorBoard
pip install tensorboard
```

### 설치 확인
```python
import torch
print(torch.__version__)  # 예: 2.1.0
print(torch.cuda.is_available())  # GPU 사용 가능 시 True
```

### 권장 도구
- **IDE**: Python 확장이 있는 VS Code, 실험용 Jupyter 노트북
- **GPU**: L28-L42에는 NVIDIA GPU 권장(대부분 레슨은 Google Colab 무료 버전 사용 가능)

## 관련 자료

- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: 손실 함수, 정규화, 평가 지표 이해를 위한 선수 과정
- **[LLM_and_NLP](../LLM_and_NLP/00_Overview.md)**: 고급 NLP 응용(BERT, GPT 파인튜닝, LangChain)
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: 스케일링 법칙, LoRA, 양자화, RAG
- **[Computer_Vision](../Computer_Vision/00_Overview.md)**: OpenCV, 객체 탐지, SLAM 응용 CV
- **[Reinforcement_Learning](../Reinforcement_Learning/00_Overview.md)**: DQN, PPO, 정책 그래디언트(L42 기반)
- **[Math_for_AI](../Math_for_AI/00_Overview.md)**: 행렬 미적분, 최적화 이론, 확률

## 학습 팁

1. **구현 건너뛰지 않기**: 코드를 직접 타이핑하면(복사하더라도) 근육 기억이 생깁니다. 읽기만 하려는 유혹을 피하세요.
2. **자유롭게 실험하기**: 하이퍼파라미터 변경, 활성화 함수 교체, 의도적으로 코드를 깨서 오류 메시지 확인.
3. **코드와 함께 논문 읽기**: L18-L22의 경우 원논문을 읽으세요. 논문의 표기법이 코드의 변수명과 일치합니다.
4. **작은 데이터로 디버그**: 전체 훈련 전에 작은 데이터셋(10개 샘플)으로 모델을 테스트하여 버그 포착.
5. **활성화 시각화**: TensorBoard(L27)를 사용하여 그래디언트, 가중치, 특징 맵 검사.
6. **커뮤니티 참여**: PyTorch 포럼, r/MachineLearning, Papers with Code 토론.

## 학습 성과

이 폴더를 완료하면 다음을 수행할 수 있습니다:

- ✅ NumPy와 PyTorch를 사용하여 신경망을 처음부터 구현
- ✅ 역전파, 경사 하강법, autograd 내부 동작 설명
- ✅ 이미지 분류를 위한 CNN 구축 및 훈련(ResNet, VGG)
- ✅ 연구 논문에서 트랜스포머, BERT, GPT 구현
- ✅ 생성 모델 훈련(GAN, VAE, Diffusion Models)
- ✅ 실제 데이터셋에 전이 학습 및 파인튜닝 적용
- ✅ 고급 기법으로 훈련 최적화(혼합 정밀도, 그래디언트 클리핑, 학습률 스케줄)
- ✅ ONNX, TorchScript, 웹 프레임워크를 사용한 모델 배포
- ✅ 최첨단 딥러닝 논문 읽고 재현

## 다음 단계

- **NLP의 경우**: 대형 언어 모델, RAG, 프롬프트 엔지니어링을 위해 `LLM_and_NLP`로 진행
- **비전의 경우**: OpenCV, 3D 비전, SLAM을 위해 `Computer_Vision` 탐구
- **효율성의 경우**: 양자화, LoRA, 모델 압축을 위해 `Foundation_Models` 학습
- **RL의 경우**: DQN, PPO, 게임 에이전트를 위해 `Reinforcement_Learning`으로 진행
- **프로덕션의 경우**: 실험 추적, 모델 서빙, CI/CD를 위해 `MLOps` 확인

## 추가 자료

- **공식 문서**: [PyTorch Tutorials](https://pytorch.org/tutorials/), [PyTorch Documentation](https://pytorch.org/docs/)
- **책**: *Deep Learning*(Goodfellow 외), *Dive into Deep Learning*(d2l.ai)
- **강좌**: Stanford CS230, Fast.ai Practical Deep Learning
- **논문**: 구현 및 벤치마크를 위한 [Papers with Code](https://paperswithcode.com/)

---

**즐거운 학습 되세요!** `01_Tensors_and_Autograd.md`부터 시작하여 딥러닝 전문성을 단계별로 쌓아가세요.
