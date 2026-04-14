# PyTorch Fundamentals 학습 가이드

## 개요

PyTorch는 연구와 산업 모두에서 지배적인 딥러닝 라이브러리로 자리잡은 Python 기반 과학 컴퓨팅 프레임워크입니다. 이 토픽에서는 PyTorch의 핵심 추상화인 텐서, 자동 미분, 모듈, 데이터 로딩, 학습 루프 등을 포괄적이고 실습 중심으로 소개하며, 고급 딥러닝 아키텍처를 다루기 전에 필요한 실무 능력을 갖추도록 합니다.

딥러닝 이론 과정과 달리, 이 토픽은 **도구로서의 PyTorch**에 집중합니다: 내부 작동 원리, 관용적인 PyTorch 코드 작성법, 모델 디버깅, 프로파일링 및 배포 방법을 다룹니다.

---

## 학습 목표 (What You'll Learn)

이 토픽을 완료하면 다음을 할 수 있습니다:

- 텐서를 CPU와 GPU에서 생성하고 조작하며, dtype, device, 메모리 레이아웃을 완전히 이해
- PyTorch의 autograd 엔진을 사용하여 그래디언트를 계산하고 계산 그래프를 이해
- `nn.Module`을 사용하여 신경망 아키텍처를 정의하고 적절한 파라미터 관리를 수행
- `Dataset`과 `DataLoader`로 커스텀 데이터셋과 효율적인 데이터 파이프라인을 구축
- 검증, 체크포인팅, 로깅이 포함된 깔끔한 학습 루프를 작성
- 프로덕션 배포를 위해 모델을 저장, 로드, 내보내기
- 일반적인 PyTorch 에러(shape 불일치, 그래디언트 문제, device 불일치)를 디버깅
- 넓은 PyTorch 생태계(torchvision, Lightning, HuggingFace)를 활용

## 사전 요구사항 (Prerequisites)

- **Python_Advanced**: 클래스, 데코레이터, 컨텍스트 매니저, 이터레이터, 타입 힌트
- **Neural_Network_Fundamentals**: 순방향 네트워크, 역전파, 손실 함수, 경사 하강법
- **선형대수 기초**: 행렬 곱셈, 전치, 브로드캐스팅 개념

## 학습 로드맵 (Learning Roadmap)

```
                    ┌───────────────────────────────────────────────────────┐
                    │              PyTorch Fundamentals (14 레슨)           │
                    └───────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ L01: 소개    │────▶│ L02: 텐서   │────▶│ L03: 텐서   │────▶│ L04: Autograd│
  │              │     │              │     │   연산       │     │              │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
                                                                       ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ L08: 학습    │◀────│ L07: Dataset │◀────│ L06: 손실   │◀────│ L05: nn      │
  │   루프       │     │ & DataLoader │     │  & 옵티마이저│     │   Module     │
  └──────┬───────┘     └──────────────┘     └──────────────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ L09: 저장    │────▶│ L10: GPU     │────▶│ L11: 디버깅  │────▶│ L12: 커스텀  │
  │   & 로드     │     │  학습        │     │              │     │  레이어      │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
                                                                       ▼
                                            ┌──────────────┐     ┌──────────────┐
                                            │ L14: PyTorch │◀────│ L13: Torch   │
                                            │  생태계      │     │  Script      │
                                            └──────────────┘     └──────────────┘
```

**추천 학습 순서**: L01부터 L14까지 순차적으로 학습합니다. 레슨은 서로 이어지며 -- L01-L04는 기초, L05-L08은 모델 구축 파이프라인, L09-L14는 프로덕션과 고급 주제를 다룹니다.

---

## 파일 목록 (File List)

| 레슨 | 파일명 | 설명 |
|------|--------|------|
| L01 | `01_Introduction_to_PyTorch.md` | 역사, 생태계, 설치, 첫 텐서 |
| L02 | `02_Tensors.md` | 생성, 속성, dtype, device, 뷰 vs 복사 |
| L03 | `03_Tensor_Operations.md` | 인덱싱, 슬라이싱, 브로드캐스팅, 행렬 연산 |
| L04 | `04_Autograd.md` | requires_grad, backward(), grad, 계산 그래프 |
| L05 | `05_nn_Module.md` | 모듈 정의, forward, parameters(), 모듈 중첩 |
| L06 | `06_Loss_Functions_and_Optimizers.md` | CrossEntropyLoss, Adam, SGD, 학습률 스케줄링 |
| L07 | `07_Dataset_and_DataLoader.md` | Dataset, DataLoader, transforms, 커스텀 데이터셋 |
| L08 | `08_Training_Loop.md` | train/eval 모드, 에포크, 배치, 검증 |
| L09 | `09_Model_Saving_and_Loading.md` | state_dict, 체크포인트, ONNX 내보내기 |
| L10 | `10_GPU_Training.md` | .to(device), DataParallel, 혼합 정밀도 |
| L11 | `11_Debugging_PyTorch.md` | shape 에러, 그래디언트 확인, 훅 |
| L12 | `12_Custom_Layers_and_Functions.md` | autograd.Function, 커스텀 backward |
| L13 | `13_TorchScript_and_Deployment.md` | 트레이싱, 스크립팅, 모바일 배포 |
| L14 | `14_PyTorch_Ecosystem.md` | torchvision, torchaudio, Lightning, HuggingFace |

**총 14개 레슨**

---

## 환경 설정 (Environment Setup)

### 설치

```bash
# PyTorch 설치 (CPU 버전)
pip install torch torchvision torchaudio

# GPU 지원 (CUDA 12.1 예시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 설치 확인
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 추천 도구

- **IDE**: VS Code에 Python 및 Pylance 확장 설치
- **디버거**: Python 디버거 (`breakpoint()`) 또는 VS Code 디버거
- **GPU**: L10에 NVIDIA GPU 권장, 대부분의 레슨은 Google Colab 무료 등급으로 충분

---

## 관련 자료 (Related Materials)

- **[Python_Advanced](../Python_Advanced/00_Overview.md)**: PyTorch 전반에 사용되는 고급 Python 기능
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: PyTorch 기초 위에 CNN, Transformer, GAN을 구현
- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: 고전적 ML 개념 (손실, 정규화, 평가)
- **[CUDA](../CUDA/00_Overview.md)**: PyTorch GPU 백엔드 이해를 위한 GPU 프로그래밍 기초

---

## 학습 팁 (Study Tips)

1. **모든 예제를 직접 타이핑하세요**: 복사-붙여넣기를 하지 마세요. 타이핑을 통해 API에 익숙해집니다.
2. **shape을 집착적으로 확인하세요**: 모든 연산 후 `.shape`을 출력하세요. 습관이 될 때까지 계속합니다.
3. **에러 메시지를 주의 깊게 읽으세요**: PyTorch 에러 메시지는 정보가 풍부합니다 -- 기대값과 실제 shape, dtype, device를 알려줍니다.
4. **디버깅에는 작은 텐서를 사용하세요**: 손으로 검증할 수 있는 2x3이나 3x4 텐서를 만드세요.
5. **공식 문서를 확인하세요**: PyTorch 공식 문서는 훌륭합니다 -- 기본 참고 자료로 활용하세요.

---

**[01_Introduction_to_PyTorch.md](./01_Introduction_to_PyTorch.md)부터 시작하여 PyTorch 여정을 시작하세요.**
